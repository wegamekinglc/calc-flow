use super::*;
use calc_flow::{SourceHistoryContext, SourceHistoryReplayFactory};
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
    recovery_case(35, 33, false).await;
}

#[tokio::test]
async fn test_frozen_asof_anchor_recovery_replays_exact_suffix_and_carries_unchanged() {
    recovery_case(38, 35, false).await;
}

#[tokio::test]
async fn test_frozen_asof_repeated_anchors_release_prefixes() {
    recovery_case(68, 66, false).await;
}

#[tokio::test]
async fn test_frozen_asof_repeated_anchors_keep_an_idle_source_position() {
    recovery_case(68, 66, true).await;
}

async fn recovery_case(rows: u64, cuts: u64, idle_right: bool) {
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
        },
    )
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
    let unchanged = first.trigger_checkpoint().await.unwrap();
    let unchanged_manifest = find_manifest(
        &root.path().join("checkpoints"),
        &format!("manifest-{:020}.json", unchanged.as_u64()),
    )
    .unwrap();
    assert_eq!(first.cancel().await.state, JobState::Cancelled);
    let manifest = find_manifest(
        &root.path().join("checkpoints"),
        &format!("manifest-{:020}.json", last.unwrap().as_u64()),
    )
    .unwrap();
    let operator = manifest.operators().values().next().unwrap();
    assert!(operator.inline_metadata.contains_key("source_replay"));
    let carried = unchanged_manifest.operators().values().next().unwrap();
    assert_eq!(
        operator
            .segments
            .iter()
            .filter(|handle| handle.segment_id() != "asof-replay-control")
            .map(|handle| (handle.segment_id(), handle.sha256()))
            .collect::<Vec<_>>(),
        carried
            .segments
            .iter()
            .filter(|handle| handle.segment_id() != "asof-replay-control")
            .map(|handle| (handle.segment_id(), handle.sha256()))
            .collect::<Vec<_>>(),
    );
    for side in ["left", "right"] {
        std::fs::remove_dir_all(root.path().join(side)).unwrap();
    }
    let replay_count = Arc::new(AtomicUsize::new(0));
    let second = runner_with_replay_count(
        root.path(),
        [Arc::new(Notify::new()), Arc::new(Notify::new())],
        1,
        Duration::ZERO,
        true,
        None,
        Some(replay_count.clone()),
    )
    .start()
    .await
    .unwrap();
    assert_eq!(second.wait().await.state, JobState::Completed);
    assert_eq!(
        right_values(root.path()),
        vec![9; usize::try_from(rows).unwrap()]
    );
    let rebuilt_rows = replay_count.load(Ordering::Relaxed);
    println!(
        "cuts={cuts}, idle_right={idle_right}, replayed_rows={rebuilt_rows}, output_rows={rows}"
    );
    assert!(
        rebuilt_rows <= 4,
        "cold recovery reread {rebuilt_rows} historical rows instead of the anchor suffix"
    );
}
