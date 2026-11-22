use super::*;
use calc_flow::{SourceHistoryContext, SourceHistoryReplayFactory, StreamRuntimeConfig};
use std::sync::atomic::{AtomicUsize, Ordering};

struct CountedFactory {
    inner: Arc<dyn SourceHistoryReplayFactory>,
    count: Arc<AtomicUsize>,
}

pub(super) fn counted_factory(
    inner: Arc<dyn SourceHistoryReplayFactory>,
    count: Arc<AtomicUsize>,
) -> Arc<dyn SourceHistoryReplayFactory> {
    Arc::new(CountedFactory { inner, count })
}

impl SourceHistoryReplayFactory for CountedFactory {
    fn create(&self, history: SourceHistoryContext) -> Result<Box<dyn StreamSource>> {
        Ok(Box::new(CountedReader {
            inner: self.inner.create(history)?,
            count: self.count.clone(),
        }))
    }
}

struct CountedReader {
    inner: Box<dyn StreamSource>,
    count: Arc<AtomicUsize>,
}

#[async_trait]
impl StreamSource for CountedReader {
    fn capabilities(&self) -> SourceCapabilities {
        self.inner.capabilities()
    }
    async fn open(&mut self, cursor: Option<Cursor>) -> Result<()> {
        self.inner.open(cursor).await
    }
    async fn next(&mut self) -> Result<Option<SourceEvent>> {
        let event = self.inner.next().await?;
        if let Some(SourceEvent::Data { batch, .. }) = &event {
            self.count.fetch_add(batch.num_rows(), Ordering::Relaxed);
        }
        Ok(event)
    }
    async fn close(&mut self) -> Result<()> {
        self.inner.close().await
    }
}

#[tokio::test]
async fn test_frozen_asof_recovery_replays_only_after_active_state_anchor() {
    recovery_case(35, 33, false, ProjectionMode::None).await;
}

#[tokio::test]
async fn test_frozen_asof_anchor_recovery_replays_exact_suffix_and_carries_unchanged() {
    recovery_case(38, 35, false, ProjectionMode::None).await;
}

#[tokio::test]
async fn test_frozen_asof_repeated_anchors_release_prefixes() {
    recovery_case(68, 66, false, ProjectionMode::None).await;
}

#[tokio::test]
async fn test_frozen_asof_repeated_anchors_keep_an_idle_source_position() {
    recovery_case(68, 66, true, ProjectionMode::None).await;
}

#[tokio::test]
async fn test_frozen_asof_replays_aliased_input_projection() {
    recovery_case(38, 35, false, ProjectionMode::Aliased).await;
}

#[tokio::test]
async fn test_frozen_asof_replays_reordered_projection_chain() {
    recovery_case(38, 35, false, ProjectionMode::ReorderedAlias).await;
}

#[tokio::test]
async fn test_frozen_asof_computed_ingress_uses_native_recovery() {
    recovery_case(6, 3, false, ProjectionMode::Computed).await;
}

#[tokio::test]
async fn test_frozen_asof_filtered_ingress_uses_native_recovery() {
    recovery_case(6, 3, false, ProjectionMode::Filtered).await;
}

async fn recovery_case(rows: u64, cuts: u64, idle_right: bool, projection: ProjectionMode) {
    let root = tempfile::tempdir().unwrap();
    for (side, offset, value) in [("left", 105, 1), ("right", 100, 9)] {
        let directory = root.path().join(side);
        std::fs::create_dir(&directory).unwrap();
        let bytes = (1..=rows)
            .flat_map(|sequence| row(offset + sequence * 100, sequence, value))
            .collect::<Vec<_>>();
        std::fs::write(directory.join("01.json"), bytes).unwrap();
    }
    let paused = [Arc::new(Notify::new()), Arc::new(Notify::new())];
    let gates = [Arc::new(Semaphore::new(0)), Arc::new(Semaphore::new(0))];
    let first = runner_observed(
        root.path(),
        paused.clone(),
        1,
        Duration::ZERO,
        true,
        Some(&gates),
        &ReplayObservation {
            rows: None,
            idle_right,
            projection,
        },
    )
    .with_runtime_config(StreamRuntimeConfig {
        // The fixture drives each cut and must work with only the latest manifest.
        checkpoint_interval: Duration::from_secs(3_600),
        retained_epochs: 1,
        ..StreamRuntimeConfig::default()
    })
    .unwrap()
    .start()
    .await
    .unwrap();
    let mut last = None;
    for cut in 1..=cuts {
        paused_at_cut(&paused).await;
        last = Some(first.trigger_checkpoint().await.unwrap());
        if cut < cuts {
            for gate in &gates {
                gate.add_permits(1);
            }
        }
    }
    let checkpoints = root.path().join("checkpoints");
    let manifest = find_manifest(
        &checkpoints,
        &format!("epoch-{}.json", last.unwrap().as_u64()),
    )
    .expect("final data checkpoint must be readable at its cut");
    let unchanged = first.trigger_checkpoint().await.unwrap();
    let unchanged_manifest =
        find_manifest(&checkpoints, &format!("epoch-{}.json", unchanged.as_u64()))
            .expect("unchanged checkpoint must be readable at its cut");
    assert_eq!(first.cancel().await.state, JobState::Cancelled);
    let replay = !matches!(
        projection,
        ProjectionMode::Computed | ProjectionMode::Filtered
    );
    assert_carried_checkpoint(&manifest, &unchanged_manifest, replay);
    for side in ["left", "right"] {
        std::fs::remove_dir_all(root.path().join(side)).unwrap();
    }
    let replay_count = Arc::new(AtomicUsize::new(0));
    let second = runner_observed(
        root.path(),
        [Arc::new(Notify::new()), Arc::new(Notify::new())],
        1,
        Duration::ZERO,
        true,
        None,
        &ReplayObservation {
            rows: Some(replay_count.clone()),
            idle_right: false,
            projection,
        },
    )
    .start()
    .await
    .unwrap();
    assert_eq!(second.wait().await.state, JobState::Completed);
    assert_outputs(root.path(), rows, projection);
    let rebuilt_rows = replay_count.load(Ordering::Relaxed);
    println!(
        "cuts={cuts}, idle_right={idle_right}, replayed_rows={rebuilt_rows}, output_rows={rows}"
    );
    assert!(
        if replay {
            rebuilt_rows <= 4
        } else {
            rebuilt_rows == 0
        },
        "cold recovery reread {rebuilt_rows} historical rows instead of the anchor suffix"
    );
}

fn assert_carried_checkpoint(
    manifest: &serde_json::Value,
    unchanged: &serde_json::Value,
    replay: bool,
) {
    let operator = &manifest["operators"]["asof"];
    assert_eq!(
        operator["inline_metadata"]
            .as_object()
            .unwrap()
            .contains_key("source_replay"),
        replay
    );
    let blocks = |manifest: &serde_json::Value| {
        manifest["operators"]["asof"]["blocks"]
            .as_array()
            .unwrap()
            .iter()
            .filter(|handle| handle["object_id"] != "asof-replay-control")
            .map(|handle| {
                (
                    handle["object_id"].as_str().unwrap().to_owned(),
                    handle["sha256"].as_str().unwrap().to_owned(),
                )
            })
            .collect::<Vec<_>>()
    };
    assert_eq!(blocks(manifest), blocks(unchanged));
}

fn assert_outputs(root: &Path, rows: u64, projection: ProjectionMode) {
    use datafusion::arrow::array::StringArray;
    let (time, sequence, value) = if projection == ProjectionMode::None {
        ("left__time", "left__sequence", "left__value")
    } else {
        ("left__ts", "left__seq", "left__v")
    };
    let mut seen = std::collections::BTreeSet::new();
    for batch in output_batches(root) {
        for record in batch.table_payload().unwrap().batches() {
            assert_eq!(record.num_columns(), 8);
            for column in record.columns() {
                assert_eq!(column.null_count(), 0);
            }
            let int = |name: &str, row| {
                record
                    .column_by_name(name)
                    .unwrap()
                    .as_any()
                    .downcast_ref::<Int64Array>()
                    .unwrap()
                    .value(row)
            };
            let ts = |name: &str, row| {
                record
                    .column_by_name(name)
                    .unwrap()
                    .as_any()
                    .downcast_ref::<TimestampMicrosecondArray>()
                    .unwrap()
                    .value(row)
            };
            for row in 0..record.num_rows() {
                let seq = int(sequence, row);
                assert!(seen.insert(seq), "duplicate output sequence {seq}");
                for key in ["left__key", "right__key"] {
                    assert_eq!(
                        record
                            .column_by_name(key)
                            .unwrap()
                            .as_any()
                            .downcast_ref::<StringArray>()
                            .unwrap()
                            .value(row),
                        "A"
                    );
                }
                assert_eq!(ts(time, row), 105 + seq * 100);
                assert_eq!(int(value, row), 1);
                assert_eq!(int("right__sequence", row), seq);
                assert_eq!(ts("right__time", row), 100 + seq * 100);
                assert_eq!(int("right__value", row), 9);
            }
        }
    }
    assert_eq!(seen, (1..=i64::try_from(rows).unwrap()).collect());
}
