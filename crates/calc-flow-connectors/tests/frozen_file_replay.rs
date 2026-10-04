#![cfg(feature = "file")]

use std::{collections::BTreeMap, path::Path, sync::Arc, time::Duration};

use async_trait::async_trait;
use calc_flow::{
    ArrowFieldSpec, AsofJoinSide, AsofStateLimits, CheckpointManifest, Cursor, JobState,
    ManagedCheckpointRuntime, PipelineBuilder, Result, SinkBinding, SourceBinding,
    SourceCapabilities, SourceEvent, SourceSchema, StreamAsofJoinOperator, StreamAsofJoinSpec,
    StreamExecutionPlan, StreamRequirements, StreamSource, StreamingRunner, UdfRegistry,
    WatermarkPolicy,
};
use calc_flow_connectors::{
    FileSinkConfig, FileSourceConfig, FrozenFileSource, TransactionalParquetSink,
};
use datafusion::arrow::array::{Int64Array, TimestampMicrosecondArray};
use serde_json::json;
use tokio::sync::{Notify, Semaphore};

struct PausedFile {
    inner: FrozenFileSource,
    paused: Arc<Notify>,
    delivered: bool,
    recovered: bool,
    native_watermarks: bool,
    pending_watermark: Option<calc_flow::EventTime>,
    gate: Option<Arc<Semaphore>>,
}

#[async_trait]
impl StreamSource for PausedFile {
    fn capabilities(&self) -> SourceCapabilities {
        let mut capabilities = self.inner.capabilities();
        if self.native_watermarks {
            capabilities.native_watermarks = calc_flow::NativeWatermarkCapability::EmitsNative;
        }
        capabilities
    }

    fn history_spec(&self) -> Option<calc_flow::SourceHistorySpec> {
        self.inner.history_spec()
    }
    fn history_replay_factory(&self) -> Option<Arc<dyn calc_flow::SourceHistoryReplayFactory>> {
        self.inner.history_replay_factory()
    }

    fn validate_history(&self, history: &calc_flow::SourceHistoryManifestEntry) -> Result<()> {
        self.inner.validate_history(history)
    }

    async fn prepare_history(&mut self, history: calc_flow::SourceHistoryContext) -> Result<()> {
        self.inner.prepare_history(history).await
    }

    async fn open(&mut self, cursor: Option<Cursor>) -> Result<()> {
        self.recovered = cursor.is_some();
        self.inner.open(cursor).await
    }

    async fn next(&mut self) -> Result<Option<SourceEvent>> {
        if let Some(watermark) = self.pending_watermark.take() {
            return Ok(Some(SourceEvent::Watermark(watermark)));
        }
        if self.delivered && !self.recovered {
            self.paused.notify_one();
            if let Some(gate) = &self.gate {
                gate.acquire().await.unwrap().forget();
            } else {
                std::future::pending::<()>().await;
            }
        }
        let event = self.inner.next().await?;
        if self.native_watermarks
            && let Some(SourceEvent::Data { batch, .. }) = &event
        {
            let time = batch
                .table_payload()?
                .batches()
                .iter()
                .flat_map(|record| {
                    record
                        .column_by_name("time")
                        .unwrap()
                        .as_any()
                        .downcast_ref::<TimestampMicrosecondArray>()
                        .unwrap()
                        .iter()
                        .flatten()
                })
                .max()
                .unwrap();
            self.pending_watermark = Some(calc_flow::EventTime::from_micros(time - 1));
        }
        self.delivered = true;
        Ok(event)
    }

    async fn close(&mut self) -> Result<()> {
        self.inner.close().await
    }
}

fn file_source(path: &Path, max_batch_rows: usize) -> FrozenFileSource {
    let schema = [
        ("key", "string"),
        ("time", "timestamp[us, UTC]"),
        ("sequence", "int64"),
        ("value", "int64"),
    ]
    .into_iter()
    .map(|(name, data_type)| ArrowFieldSpec {
        name: name.into(),
        data_type: data_type.into(),
        nullable: false,
    })
    .collect::<Vec<_>>();
    let options = BTreeMap::from([
        ("path".into(), json!(path.display().to_string())),
        ("format".into(), json!("json")),
        ("schema".into(), json!(schema)),
        ("max_batch_rows".into(), json!(max_batch_rows)),
        ("max_batch_bytes".into(), json!(1024 * 1024)),
    ]);
    FrozenFileSource::new(FileSourceConfig::from_options(&options).unwrap()).unwrap()
}

fn plan(left: &FrozenFileSource, right: &FrozenFileSource) -> StreamExecutionPlan {
    let exact = |source: &FrozenFileSource| match source.capabilities().schema {
        SourceSchema::Exact(schema) => schema,
        SourceSchema::DynamicOrUnknown => panic!("file replay requires exact schemas"),
    };
    let side = |prefix: &str| {
        AsofJoinSide::new(
            vec!["key".into()],
            "time".into(),
            vec!["sequence".into()],
            prefix.into(),
        )
        .unwrap()
    };
    let spec = StreamAsofJoinSpec::new(
        side("left"),
        side("right"),
        Duration::from_micros(10),
        AsofStateLimits::new(10_000, 8 * 1024 * 1024).unwrap(),
    )
    .unwrap();
    PipelineBuilder::new("frozen_asof_recovery")
        .unwrap()
        .add_node(
            "asof",
            Box::new(StreamAsofJoinOperator::new("asof", exact(left), exact(right), spec).unwrap()),
        )
        .unwrap()
        .compile_stream(
            &UdfRegistry::new().snapshot(),
            &StreamRequirements::default(),
        )
        .unwrap()
}

fn runner(root: &Path, paused: [Arc<Notify>; 2]) -> StreamingRunner {
    runner_with_options(root, paused, 1, Duration::from_micros(1))
}

fn runner_with_options(
    root: &Path,
    paused: [Arc<Notify>; 2],
    max_batch_rows: usize,
    max_out_of_orderness: Duration,
) -> StreamingRunner {
    runner_configured(
        root,
        paused,
        max_batch_rows,
        max_out_of_orderness,
        false,
        None,
    )
}

fn runner_configured(
    root: &Path,
    paused: [Arc<Notify>; 2],
    max_batch_rows: usize,
    max_out_of_orderness: Duration,
    native_watermarks: bool,
    gates: Option<&[Arc<Semaphore>; 2]>,
) -> StreamingRunner {
    let [left_paused, right_paused] = paused;
    let left = file_source(&root.join("left"), max_batch_rows);
    let right = file_source(&root.join("right"), max_batch_rows);
    let plan = plan(&left, &right);
    let ids = plan.source_binding_ids();
    let left_id = ids
        .iter()
        .copied()
        .find(|id| id.ends_with("left"))
        .unwrap()
        .to_string();
    let right_id = ids
        .iter()
        .copied()
        .find(|id| id.ends_with("right"))
        .unwrap()
        .to_string();
    let output = plan.sink_binding_ids()[0].to_string();
    let bind = |inner, paused, gate| {
        let source = SourceBinding::new(PausedFile {
            inner,
            paused,
            delivered: false,
            recovered: false,
            native_watermarks,
            pending_watermark: None,
            gate,
        });
        if native_watermarks {
            return source.with_watermark_policy(WatermarkPolicy::SourceProvided);
        }
        source.with_watermark_policy(WatermarkPolicy::BoundedOutOfOrderness {
            event_time_column: "time".into(),
            max_out_of_orderness,
            emit_interval: Duration::from_millis(1),
            idle_timeout: None,
        })
    };
    StreamingRunner::new(
        plan,
        BTreeMap::from([
            (
                left_id,
                bind(left, left_paused, gates.map(|gates| gates[0].clone())),
            ),
            (
                right_id,
                bind(right, right_paused, gates.map(|gates| gates[1].clone())),
            ),
        ]),
        BTreeMap::from([(
            output,
            vec![
                SinkBinding::transactional(
                    "results",
                    TransactionalParquetSink::new(FileSinkConfig {
                        root: root.join("out"),
                        output: "results".into(),
                    })
                    .unwrap(),
                )
                .unwrap(),
            ],
        )]),
        ManagedCheckpointRuntime::new(root.join("checkpoints")).unwrap(),
    )
    .unwrap()
}

fn row(time: u64, sequence: u64, value: i64) -> Vec<u8> {
    format!("{{\"key\":\"A\",\"time\":\"1970-01-01T00:00:00.{time:06}Z\",\"sequence\":{sequence},\"value\":{value}}}\n").into_bytes()
}

fn right_values(root: &Path) -> Vec<i64> {
    use calc_flow::{DecodeBounds, FormatDecoder};
    let codec = calc_flow_connectors::parquet::ParquetCodec::new("1").unwrap();
    let mut found = Vec::new();
    for epoch in std::fs::read_dir(root.join("out/results")).unwrap() {
        let path = epoch.unwrap().path();
        if !path.is_dir()
            || !path
                .file_name()
                .unwrap()
                .to_str()
                .unwrap()
                .starts_with("epoch=")
        {
            continue;
        }
        for part in std::fs::read_dir(path).unwrap() {
            let path = part.unwrap().path();
            if path.extension().and_then(|extension| extension.to_str()) != Some("parquet") {
                continue;
            }
            let batch = codec
                .decode(
                    &std::fs::read(path).unwrap(),
                    &DecodeBounds::new(10_000, 1024 * 1024).unwrap(),
                    &[],
                )
                .unwrap();
            for record in batch.table_payload().unwrap().batches() {
                let column = record
                    .column_by_name("right__value")
                    .unwrap()
                    .as_any()
                    .downcast_ref::<Int64Array>()
                    .unwrap();
                found.extend(column.iter().map(|value| value.unwrap()));
            }
        }
    }
    found.sort_unstable();
    found
}

#[tokio::test]
async fn test_frozen_asof_checkpoint_restart_uses_original_unconsumed_history() {
    let root = tempfile::tempdir().unwrap();
    for side in ["left", "right"] {
        std::fs::create_dir(root.path().join(side)).unwrap();
    }
    std::fs::write(root.path().join("left/01.json"), row(105, 1, 1)).unwrap();
    std::fs::write(root.path().join("left/02.json"), row(205, 2, 2)).unwrap();
    std::fs::write(root.path().join("right/01.json"), row(100, 1, 5)).unwrap();
    std::fs::write(root.path().join("right/02.json"), row(200, 2, 9)).unwrap();
    let paused = [Arc::new(Notify::new()), Arc::new(Notify::new())];
    let first = runner(root.path(), paused.clone()).start().await.unwrap();
    tokio::time::timeout(Duration::from_secs(10), async {
        paused[0].notified().await;
        paused[1].notified().await;
    })
    .await
    .unwrap();
    let completed = tokio::time::timeout(Duration::from_secs(10), first.trigger_checkpoint())
        .await
        .unwrap()
        .unwrap();
    let first_outcome = first.cancel().await;
    assert_eq!(first_outcome.state, JobState::Cancelled);
    assert_eq!(first_outcome.completed_epoch, Some(completed));
    std::fs::remove_file(root.path().join("left/02.json")).unwrap();
    std::fs::write(root.path().join("left/03.json"), row(305, 3, 3)).unwrap();
    std::fs::write(root.path().join("right/02.json"), row(200, 2, 999)).unwrap();
    let second = runner(
        root.path(),
        [Arc::new(Notify::new()), Arc::new(Notify::new())],
    )
    .start()
    .await
    .unwrap();
    let outcome = tokio::time::timeout(Duration::from_secs(10), second.wait())
        .await
        .unwrap();
    assert_eq!(outcome.state, JobState::Completed);
    assert_eq!(right_values(root.path()), [5, 9]);
    std::fs::remove_dir_all(root.path().join("left")).unwrap();
    std::fs::remove_dir_all(root.path().join("right")).unwrap();
    let terminal = runner(
        root.path(),
        [Arc::new(Notify::new()), Arc::new(Notify::new())],
    )
    .start()
    .await
    .unwrap();
    assert_eq!(terminal.wait().await.state, JobState::Completed);
    assert_eq!(right_values(root.path()), [5, 9]);
}

fn find_manifest(root: &Path, name: &str) -> Option<CheckpointManifest> {
    for entry in std::fs::read_dir(root).unwrap() {
        let path = entry.unwrap().path();
        if path.is_dir() {
            if let Some(manifest) = find_manifest(&path, name) {
                return Some(manifest);
            }
        } else if path.file_name().and_then(|name| name.to_str()) == Some(name) {
            return Some(CheckpointManifest::from_bytes(&std::fs::read(path).unwrap()).unwrap());
        }
    }
    None
}

#[tokio::test]
async fn test_frozen_asof_checkpoint_stores_batch_coordinates_instead_of_rows() {
    let root = tempfile::tempdir().unwrap();
    for (side, time, value) in [("left", 105, 1), ("right", 100, 9)] {
        let directory = root.path().join(side);
        std::fs::create_dir(&directory).unwrap();
        let bytes = (1..=1000)
            .flat_map(|sequence| row(time, sequence, value))
            .collect::<Vec<_>>();
        std::fs::write(directory.join("01.json"), bytes).unwrap();
    }
    let paused = [Arc::new(Notify::new()), Arc::new(Notify::new())];
    let first = runner_with_options(root.path(), paused.clone(), 1000, Duration::from_secs(10))
        .start()
        .await
        .unwrap();
    tokio::time::timeout(Duration::from_secs(10), async {
        paused[0].notified().await;
        paused[1].notified().await;
    })
    .await
    .unwrap();
    let epoch = tokio::time::timeout(Duration::from_secs(10), first.trigger_checkpoint())
        .await
        .unwrap()
        .unwrap();
    assert_eq!(first.cancel().await.state, JobState::Cancelled);
    let manifest = find_manifest(
        &root.path().join("checkpoints"),
        &format!("manifest-{:020}.json", epoch.as_u64()),
    )
    .unwrap();
    let operator_bytes = manifest
        .operators()
        .values()
        .flat_map(|operator| &operator.segments)
        .map(calc_flow::StateHandle::byte_len)
        .sum::<u64>();
    println!("retained rows: 2000; checkpoint operator bytes: {operator_bytes}");
    assert!(
        operator_bytes < 4096,
        "two data callbacks should store coordinates, not 2000 retained rows: {operator_bytes} bytes"
    );
    for side in ["left", "right"] {
        std::fs::remove_dir_all(root.path().join(side)).unwrap();
    }
    let recovered = runner_with_options(
        root.path(),
        [Arc::new(Notify::new()), Arc::new(Notify::new())],
        1000,
        Duration::from_secs(10),
    )
    .start()
    .await
    .unwrap();
    let outcome = tokio::time::timeout(Duration::from_secs(10), recovered.wait())
        .await
        .unwrap();
    assert_eq!(outcome.state, JobState::Completed);
    assert_eq!(right_values(root.path()), vec![9; 1000]);
}

async fn paused_at_cut(paused: &[Arc<Notify>; 2]) {
    tokio::time::timeout(Duration::from_secs(10), async {
        paused[0].notified().await;
        paused[1].notified().await;
    })
    .await
    .unwrap();
}

#[tokio::test]
async fn test_frozen_asof_replay_suppresses_already_committed_output() {
    let root = tempfile::tempdir().unwrap();
    for (side, times, values) in [("left", [105, 205], [1, 2]), ("right", [100, 200], [5, 9])] {
        let directory = root.path().join(side);
        std::fs::create_dir(&directory).unwrap();
        let bytes = row(times[0], 1, values[0])
            .into_iter()
            .chain(row(times[1], 2, values[1]))
            .collect::<Vec<_>>();
        std::fs::write(directory.join("01.json"), bytes).unwrap();
    }
    let paused = [Arc::new(Notify::new()), Arc::new(Notify::new())];
    let first = runner_configured(
        root.path(),
        paused.clone(),
        2,
        Duration::from_secs(10),
        true,
        None,
    )
    .start()
    .await
    .unwrap();
    paused_at_cut(&paused).await;
    first.trigger_checkpoint().await.unwrap();
    assert_eq!(first.cancel().await.state, JobState::Cancelled);
    assert_eq!(right_values(root.path()), [5]);
    for side in ["left", "right"] {
        std::fs::remove_dir_all(root.path().join(side)).unwrap();
    }
    let second = runner_configured(
        root.path(),
        [Arc::new(Notify::new()), Arc::new(Notify::new())],
        2,
        Duration::from_secs(10),
        true,
        None,
    )
    .start()
    .await
    .unwrap();
    assert_eq!(second.wait().await.state, JobState::Completed);
    assert_eq!(right_values(root.path()), [5, 9]);
}

#[tokio::test]
async fn test_frozen_asof_replay_compacts_and_resumes_after_33_cuts() {
    let root = tempfile::tempdir().unwrap();
    for (side, time, value) in [("left", 105, 1), ("right", 100, 9)] {
        let directory = root.path().join(side);
        std::fs::create_dir(&directory).unwrap();
        let bytes = (1..=35)
            .flat_map(|sequence| row(time, sequence, value))
            .collect::<Vec<_>>();
        std::fs::write(directory.join("01.json"), bytes).unwrap();
    }
    let paused = [Arc::new(Notify::new()), Arc::new(Notify::new())];
    let gates = [Arc::new(Semaphore::new(0)), Arc::new(Semaphore::new(0))];
    let first = runner_configured(
        root.path(),
        paused.clone(),
        1,
        Duration::from_secs(10),
        false,
        Some(&gates),
    )
    .start()
    .await
    .unwrap();
    for cut in 1..=33 {
        paused_at_cut(&paused).await;
        let epoch = first.trigger_checkpoint().await.unwrap();
        let manifest = find_manifest(
            &root.path().join("checkpoints"),
            &format!("manifest-{:020}.json", epoch.as_u64()),
        )
        .unwrap();
        let segments = &manifest.operators().values().next().unwrap().segments;
        assert_eq!(segments.len(), if cut == 33 { 2 } else { cut + 1 });
        if cut < 33 {
            for gate in &gates {
                gate.add_permits(1);
            }
        }
    }
    assert_eq!(first.cancel().await.state, JobState::Cancelled);
    for side in ["left", "right"] {
        std::fs::remove_dir_all(root.path().join(side)).unwrap();
    }
    let second = runner_with_options(
        root.path(),
        [Arc::new(Notify::new()), Arc::new(Notify::new())],
        1,
        Duration::from_secs(10),
    )
    .start()
    .await
    .unwrap();
    assert_eq!(second.wait().await.state, JobState::Completed);
    assert_eq!(right_values(root.path()), vec![9; 35]);
}
