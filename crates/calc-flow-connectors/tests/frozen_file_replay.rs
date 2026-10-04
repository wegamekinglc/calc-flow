#![cfg(feature = "file")]

use std::{collections::BTreeMap, path::Path, sync::Arc, time::Duration};

use async_trait::async_trait;
use calc_flow::{
    ArrowFieldSpec, AsofJoinSide, AsofStateLimits, Cursor, JobState, ManagedCheckpointRuntime,
    PipelineBuilder, Result, SinkBinding, SourceBinding, SourceCapabilities, SourceEvent,
    SourceSchema, StreamAsofJoinOperator, StreamAsofJoinSpec, StreamExecutionPlan,
    StreamRequirements, StreamSource, StreamingRunner, UdfRegistry, WatermarkPolicy,
};
use calc_flow_connectors::{
    FileSinkConfig, FileSourceConfig, FrozenFileSource, TransactionalParquetSink,
};
use datafusion::arrow::array::Int64Array;
use serde_json::json;
use tokio::sync::Notify;

struct PausedFile {
    inner: FrozenFileSource,
    paused: Arc<Notify>,
    delivered: bool,
    recovered: bool,
}

#[async_trait]
impl StreamSource for PausedFile {
    fn capabilities(&self) -> SourceCapabilities {
        self.inner.capabilities()
    }

    fn history_spec(&self) -> Option<calc_flow::SourceHistorySpec> {
        self.inner.history_spec()
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
        if self.delivered && !self.recovered {
            self.paused.notify_one();
            std::future::pending().await
        } else {
            let event = self.inner.next().await?;
            self.delivered = true;
            Ok(event)
        }
    }

    async fn close(&mut self) -> Result<()> {
        self.inner.close().await
    }
}

fn file_source(path: &Path) -> FrozenFileSource {
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
        ("max_batch_rows".into(), json!(1)),
        ("max_batch_bytes".into(), json!(1024)),
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
        AsofStateLimits::new(100, 1024 * 1024).unwrap(),
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
    let [left_paused, right_paused] = paused;
    let left = file_source(&root.join("left"));
    let right = file_source(&root.join("right"));
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
    let bind = |inner, paused| {
        SourceBinding::new(PausedFile {
            inner,
            paused,
            delivered: false,
            recovered: false,
        })
        .with_watermark_policy(WatermarkPolicy::BoundedOutOfOrderness {
            event_time_column: "time".into(),
            max_out_of_orderness: Duration::from_micros(1),
            emit_interval: Duration::from_millis(1),
            idle_timeout: None,
        })
    };
    StreamingRunner::new(
        plan,
        BTreeMap::from([
            (left_id, bind(left, left_paused)),
            (right_id, bind(right, right_paused)),
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
                    &DecodeBounds::new(10, 1024 * 1024).unwrap(),
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
