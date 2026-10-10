use super::*;
use crate::{ExpressionOperator, PipelineBuilder, StreamRequirements, UdfRegistry};
use datafusion::arrow::array::Int64Array;
use datafusion::arrow::datatypes::{DataType, Field, Schema};
use datafusion::arrow::record_batch::RecordBatch;
use parking_lot::Mutex;
use std::sync::atomic::AtomicUsize;

struct PendingSource;

#[async_trait]
impl StreamSource for PendingSource {
    fn capabilities(&self) -> SourceCapabilities {
        SourceCapabilities {
            replay_positioning: ReplayPositioning::ExactPauseReportAndSeek,
            delivery: SourceDeliveryCapability::Lossless,
            max_batch_rows: 1,
            max_batch_bytes: 1024,
            schema: SourceSchema::DynamicOrUnknown,
            native_watermarks: NativeWatermarkCapability::EmitsNative,
        }
    }

    async fn open(&mut self, _cursor: Option<Cursor>) -> Result<()> {
        Ok(())
    }

    async fn next(&mut self) -> Result<Option<SourceEvent>> {
        std::future::pending().await
    }

    async fn close(&mut self) -> Result<()> {
        Ok(())
    }
}

struct NoopSink;

#[async_trait]
impl StreamSink for NoopSink {
    async fn open(&mut self) -> Result<()> {
        Ok(())
    }

    async fn write(&mut self, _batch: &Batch) -> Result<()> {
        Ok(())
    }

    async fn close(&mut self) -> Result<()> {
        Ok(())
    }
}

fn runner(checkpoint: Option<ManagedCheckpointRuntime>) -> StreamingRunner {
    let plan = PipelineBuilder::new("checkpoint-mode")
        .unwrap()
        .add_node(
            "forward",
            Box::new(
                ExpressionOperator::new("forward", "", vec!["value".into()], None, Vec::new())
                    .unwrap(),
            ),
        )
        .unwrap()
        .compile_stream(
            &UdfRegistry::new().snapshot(),
            &StreamRequirements::default(),
        )
        .unwrap();
    let sources = BTreeMap::from([("input".into(), SourceBinding::new(PendingSource))]);
    let sinks = BTreeMap::from([(
        "output".into(),
        vec![SinkBinding::ordinary("sink", NoopSink).unwrap()],
    )]);
    match checkpoint {
        Some(checkpoint) => StreamingRunner::new(plan, sources, sinks, checkpoint),
        None => StreamingRunner::without_checkpoints(plan, sources, sinks),
    }
    .unwrap()
}

#[test]
fn test_checkpointing_defaults_on_and_rejects_backend_mode_conflicts() {
    assert!(StreamRuntimeConfig::default().checkpointing);
    let error = runner(None)
        .with_runtime_config(StreamRuntimeConfig::default())
        .err()
        .expect("enabling checkpoints requires managed storage");
    assert!(
        matches!(error, CalcFlowError::Streaming(error) if error.category() == StreamingErrorCategory::Validation)
    );

    let directory = tempfile::tempdir().unwrap();
    let error = runner(Some(
        ManagedCheckpointRuntime::new(directory.path().join("managed")).unwrap(),
    ))
    .with_runtime_config(StreamRuntimeConfig {
        checkpointing: false,
        ..StreamRuntimeConfig::default()
    })
    .err()
    .expect("disabling checkpoints cannot silently discard supplied storage");
    assert!(
        matches!(error, CalcFlowError::Streaming(error) if error.category() == StreamingErrorCategory::Validation)
    );
    assert!(!directory.path().join("managed").exists());
}

#[test]
fn test_checkpointing_mode_is_fixed_in_each_owned_job_context() {
    let context = StreamJobContext::new(
        1,
        "fingerprint",
        JsonMap::new(),
        None,
        CancellationToken::new(),
    );
    assert!(context.checkpointing());
    let disabled = context.clone().with_checkpointing(false);
    assert!(!disabled.checkpointing());
    assert!(!disabled.clone().checkpointing());
    assert!(context.checkpointing());
}

#[test]
fn test_checkpointing_hash_records_disabled_mode_without_changing_the_plan() {
    let runner = runner(None);
    let fingerprint = runner.plan.fingerprint().to_owned();
    let enabled = runner
        .plan
        .runtime_config_hash(&StreamRuntimeConfig::default())
        .unwrap();
    let disabled = runner
        .plan
        .runtime_config_hash(&StreamRuntimeConfig {
            checkpointing: false,
            ..StreamRuntimeConfig::default()
        })
        .unwrap();
    assert_eq!(
        enabled,
        "fd02188bac4dfdca45ddd37267c08da51ea4a47c562edb06236b59b056f6e4bb"
    );
    assert_ne!(enabled, disabled);
    assert_eq!(runner.plan.fingerprint(), fingerprint);
}

#[derive(Clone, Default)]
struct Lifecycle {
    opened: Arc<AtomicUsize>,
    closed: Arc<AtomicUsize>,
    output: Arc<Mutex<Vec<RecordBatch>>>,
}

struct FiniteSource {
    lifecycle: Lifecycle,
    batch: Option<Batch>,
    history: bool,
}

#[async_trait]
impl StreamSource for FiniteSource {
    fn capabilities(&self) -> SourceCapabilities {
        SourceCapabilities {
            max_batch_rows: 2,
            ..PendingSource.capabilities()
        }
    }
    fn history_spec(&self) -> Option<crate::SourceHistorySpec> {
        self.history.then(|| {
            crate::SourceHistorySpec::new(
                "test_history",
                crate::SourceHistoryLimits::new(1, 1024, 1024).unwrap(),
            )
            .unwrap()
        })
    }
    async fn open(&mut self, cursor: Option<Cursor>) -> Result<()> {
        assert!(cursor.is_none());
        self.lifecycle.opened.fetch_add(1, Ordering::SeqCst);
        Ok(())
    }
    async fn next(&mut self) -> Result<Option<SourceEvent>> {
        Ok(self.batch.take().map(|batch| SourceEvent::Data {
            batch,
            cursor: Cursor::unbound(vec![1], JsonMap::new()).unwrap(),
        }))
    }
    async fn close(&mut self) -> Result<()> {
        self.lifecycle.closed.fetch_add(1, Ordering::SeqCst);
        Ok(())
    }
}

struct RecordingSink {
    lifecycle: Lifecycle,
    close_delay: Duration,
}

#[async_trait]
impl StreamSink for RecordingSink {
    async fn open(&mut self) -> Result<()> {
        self.lifecycle.opened.fetch_add(1, Ordering::SeqCst);
        Ok(())
    }
    async fn write(&mut self, batch: &Batch) -> Result<()> {
        self.lifecycle
            .output
            .lock()
            .extend_from_slice(batch.table_payload()?.batches());
        Ok(())
    }
    async fn close(&mut self) -> Result<()> {
        tokio::time::sleep(self.close_delay).await;
        self.lifecycle.closed.fetch_add(1, Ordering::SeqCst);
        Ok(())
    }
}

struct EpochSink(Lifecycle);

#[async_trait]
impl TransactionalStreamSink for EpochSink {
    async fn open(&mut self) -> Result<()> {
        self.0.opened.fetch_add(1, Ordering::SeqCst);
        Ok(())
    }
    async fn begin_epoch(&mut self, _: Epoch) -> Result<()> {
        Ok(())
    }
    async fn write(&mut self, _: &Batch) -> Result<()> {
        Ok(())
    }
    async fn pre_commit(&mut self, _: Epoch) -> Result<JsonMap> {
        Ok(JsonMap::new())
    }
    async fn commit(&mut self, _: Epoch, _: &JsonMap) -> Result<()> {
        Ok(())
    }
    async fn abort(&mut self, _: Epoch, _: Option<&JsonMap>) -> Result<()> {
        Ok(())
    }
    async fn recover(&mut self, _: &SinkRecovery) -> Result<()> {
        Ok(())
    }
    async fn close(&mut self) -> Result<()> {
        self.0.closed.fetch_add(1, Ordering::SeqCst);
        Ok(())
    }
}

fn finite_runner(
    sql: bool,
    history: bool,
    sink: SinkBinding,
    source: &Lifecycle,
) -> StreamingRunner {
    let batch = Batch::table(
        vec![
            RecordBatch::try_new(
                Arc::new(Schema::new(vec![Field::new(
                    "value",
                    DataType::Int64,
                    false,
                )])),
                vec![Arc::new(Int64Array::from(vec![7, 11]))],
            )
            .unwrap(),
        ],
        crate::BatchMetadata::new("input", 1, JsonMap::new()).unwrap(),
    )
    .unwrap();
    let mut result = runner(None);
    if sql {
        result.plan = PipelineBuilder::new("checkpoint-mode-sql")
            .unwrap()
            .add_node(
                "sql",
                Box::new(
                    crate::SqlOperator::new(
                        "sql",
                        "SELECT value, COUNT(*) AS count FROM input GROUP BY value",
                        vec!["input".into()],
                        Vec::new(),
                    )
                    .unwrap(),
                ),
            )
            .unwrap()
            .compile_stream(
                &UdfRegistry::new().snapshot(),
                &StreamRequirements::default(),
            )
            .unwrap();
    }
    result.sources = BTreeMap::from([(
        "input".into(),
        SourceBinding::new(FiniteSource {
            lifecycle: source.clone(),
            batch: Some(batch),
            history,
        }),
    )]);
    result.sinks = BTreeMap::from([("output".into(), vec![sink])]);
    result
}

fn ordinary_sink(lifecycle: &Lifecycle, close_delay: Duration) -> SinkBinding {
    SinkBinding::ordinary(
        "sink",
        RecordingSink {
            lifecycle: lifecycle.clone(),
            close_delay,
        },
    )
    .unwrap()
}

#[tokio::test]
async fn test_checkpointing_off_delivers_rows_finishes_and_preserves_edge_config() {
    let source = Lifecycle::default();
    let sink = Lifecycle::default();
    let (start, cleanup) =
        finite_runner(false, false, ordinary_sink(&sink, Duration::ZERO), &source)
            .with_runtime_config(StreamRuntimeConfig {
                checkpointing: false,
                checkpoint_interval: Duration::from_micros(1),
                edge_budget: crate::EdgeBudget::new(3, 4096).unwrap(),
                ..StreamRuntimeConfig::default()
            })
            .unwrap()
            .start_with_cleanup();
    let job = start.await.unwrap();
    let trigger = job.trigger_checkpoint().await.unwrap_err();
    let outcome = job.wait().await;
    let status = job.status();
    drop(job);
    cleanup.await.unwrap();
    assert_eq!(outcome.state, JobState::Completed, "{:?}", outcome.errors);
    assert_eq!(outcome.completed_epoch, None);
    assert_eq!(status.checkpoint, CheckpointStatus::default());
    assert!(
        matches!(trigger, CalcFlowError::Streaming(error) if error.category() == StreamingErrorCategory::Validation)
    );
    assert_eq!(source.closed.load(Ordering::SeqCst), 1);
    assert_eq!(sink.closed.load(Ordering::SeqCst), 1);
    assert_eq!(
        sink.output.lock()[0]
            .column(0)
            .as_any()
            .downcast_ref::<Int64Array>()
            .unwrap()
            .values()
            .as_ref(),
        &[7, 11]
    );
    assert!(
        status
            .edges
            .values()
            .all(|edge| edge.row_limit == 3 && edge.byte_limit == 4096)
    );
    assert_eq!(
        status.delivery["output"].effective,
        crate::DeliveryGuarantee::BestEffort
    );
}

#[tokio::test]
async fn test_checkpointing_off_preserves_sql_state_budget() {
    let source = Lifecycle::default();
    let sink = Lifecycle::default();
    let (start, cleanup) =
        finite_runner(true, false, ordinary_sink(&sink, Duration::ZERO), &source)
            .with_runtime_config(StreamRuntimeConfig {
                checkpointing: false,
                sql_state_budget: Some(crate::StateBudget::new(1, 4096).unwrap()),
                ..StreamRuntimeConfig::default()
            })
            .unwrap()
            .start_with_cleanup();
    let job = start.await.unwrap();
    let outcome = job.wait().await;
    drop(job);
    cleanup.await.unwrap();
    assert_eq!(outcome.state, JobState::Failed);
    assert!(sink.output.lock().is_empty());
    assert_eq!(source.closed.load(Ordering::SeqCst), 1);
    assert_eq!(sink.closed.load(Ordering::SeqCst), 1);
}

#[tokio::test]
async fn test_checkpointing_off_preserves_sink_lifecycle_timeout() {
    let source = Lifecycle::default();
    let sink = Lifecycle::default();
    let (start, cleanup) = finite_runner(
        false,
        false,
        ordinary_sink(&sink, Duration::from_millis(100)),
        &source,
    )
    .with_runtime_config(StreamRuntimeConfig {
        checkpointing: false,
        checkpoint_timeout: Duration::from_millis(10),
        ..StreamRuntimeConfig::default()
    })
    .unwrap()
    .start_with_cleanup();
    let job = start.await.unwrap();
    let outcome = job.wait().await;
    drop(job);
    cleanup.await.unwrap();
    assert_eq!(outcome.state, JobState::Failed);
    assert_eq!(source.closed.load(Ordering::SeqCst), 1);
}

#[tokio::test]
async fn test_checkpointing_off_rejects_epoch_sinks_before_lifecycle() {
    for epoch_idempotent in [false, true] {
        let source = Lifecycle::default();
        let sink = Lifecycle::default();
        let binding = if epoch_idempotent {
            SinkBinding::epoch_idempotent(
                "sink",
                EpochSink(sink.clone()),
                "test_epoch",
                RetentionClass::Unbounded,
            )
        } else {
            SinkBinding::transactional("sink", EpochSink(sink.clone()))
        }
        .unwrap();
        let (start, cleanup) = finite_runner(false, false, binding, &source).start_with_cleanup();
        let result = start.await;
        if let Ok(job) = &result {
            job.cancel().await;
        }
        let error = result.err();
        cleanup.await.unwrap();
        assert!(
            error.is_some(),
            "epoch sink must require checkpoint ownership"
        );
        assert_eq!(source.opened.load(Ordering::SeqCst), 0);
        assert_eq!(sink.opened.load(Ordering::SeqCst), 0);
    }
}

#[tokio::test]
async fn test_checkpointing_off_rejects_history_before_lifecycle() {
    let source = Lifecycle::default();
    let sink = Lifecycle::default();
    let (start, cleanup) =
        finite_runner(false, true, ordinary_sink(&sink, Duration::ZERO), &source)
            .start_with_cleanup();
    let error = start
        .await
        .err()
        .expect("history requires checkpoint storage");
    cleanup.await.unwrap();
    assert!(
        matches!(error, CalcFlowError::Streaming(error) if error.category() == StreamingErrorCategory::Validation)
    );
    assert_eq!(source.opened.load(Ordering::SeqCst), 0);
    assert_eq!(sink.opened.load(Ordering::SeqCst), 0);
}
