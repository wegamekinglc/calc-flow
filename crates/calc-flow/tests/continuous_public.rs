use std::{
    collections::BTreeMap,
    path::{Path, PathBuf},
    sync::{
        Arc, Mutex,
        atomic::{AtomicUsize, Ordering},
    },
};

use async_trait::async_trait;
use calc_flow::{
    Batch, BatchKind, CalcFlowError, ExpressionOperator, JsonMap, PipelineBuilder, Port, Result,
    StreamOperator, StreamRequirements, UdfRegistry, UnionOperator,
    continuous::{
        Cursor, JobState, ManagedCheckpointRuntime, NativeWatermarkCapability, ReplayPositioning,
        SinkBinding, SinkRecovery, SourceBinding, SourceCapabilities, SourceDeliveryCapability,
        SourceEvent, SourceSchema, SourceStatus, StreamSink, StreamSource, StreamingErrorCategory,
        StreamingJob, StreamingRunner, TransactionalStreamSink,
    },
};

struct FiniteSource;

#[async_trait]
impl StreamSource for FiniteSource {
    fn capabilities(&self) -> SourceCapabilities {
        SourceCapabilities {
            replay_positioning: ReplayPositioning::ExactPauseReportAndSeek,
            delivery: SourceDeliveryCapability::Lossless,
            max_batch_rows: 1,
            max_batch_bytes: 1,
            schema: SourceSchema::DynamicOrUnknown,
            native_watermarks: NativeWatermarkCapability::NeverEmits,
        }
    }

    async fn open(&mut self, _cursor: Option<Cursor>) -> Result<()> {
        Ok(())
    }

    async fn next(&mut self) -> Result<Option<SourceEvent>> {
        Ok(None)
    }

    async fn close(&mut self) -> Result<()> {
        Ok(())
    }
}

fn accept_object_safe_source(_source: Box<dyn StreamSource>) {}

fn accept_object_safe_connectors(
    _source: Option<Box<dyn StreamSource>>,
    _sink: Option<Box<dyn StreamSink>>,
    _transactional_sink: Option<Box<dyn TransactionalStreamSink>>,
) {
}

fn accept_send<T: Send>() {}

fn accept_send_sync<T: Send + Sync>() {}

fn accept_source_status_contract(status: &SourceStatus) {
    let _: ReplayPositioning = status.replay_positioning;
    let _: SourceDeliveryCapability = status.delivery;
}

#[test]
fn public_continuous_types_are_available_to_external_crates() {
    accept_object_safe_source(Box::new(FiniteSource));
    accept_object_safe_connectors(None, None, None);
    accept_send::<StreamingRunner>();
    accept_send_sync::<StreamingJob>();

    let root = PathBuf::from("managed-checkpoints");
    let _checkpoints = ManagedCheckpointRuntime::new(root).unwrap();
    let cursor = Cursor::new("source", vec![1], JsonMap::new()).unwrap();

    assert_eq!(cursor.source_id(), Some("source"));
    assert_eq!(cursor.order(), [1]);

    let _ = accept_source_status_contract;
}

#[derive(Clone, Default)]
struct LifecycleProbe {
    capability_reads: Arc<AtomicUsize>,
    source_opens: Arc<AtomicUsize>,
    source_closes: Arc<AtomicUsize>,
    sink_opens: Arc<AtomicUsize>,
    sink_closes: Arc<AtomicUsize>,
    sink_recoveries: Arc<Mutex<Vec<bool>>>,
}

struct PendingSource {
    probe: LifecycleProbe,
}

struct EndingSource {
    probe: LifecycleProbe,
}

struct FailingSource;

fn exact_source_capabilities() -> SourceCapabilities {
    SourceCapabilities {
        replay_positioning: ReplayPositioning::ExactPauseReportAndSeek,
        delivery: SourceDeliveryCapability::Lossless,
        max_batch_rows: 1,
        max_batch_bytes: 1024,
        schema: SourceSchema::DynamicOrUnknown,
        native_watermarks: NativeWatermarkCapability::EmitsNative,
    }
}

#[async_trait]
impl StreamSource for PendingSource {
    fn capabilities(&self) -> SourceCapabilities {
        self.probe.capability_reads.fetch_add(1, Ordering::SeqCst);
        exact_source_capabilities()
    }

    async fn open(&mut self, _cursor: Option<Cursor>) -> Result<()> {
        self.probe.source_opens.fetch_add(1, Ordering::SeqCst);
        Ok(())
    }

    async fn next(&mut self) -> Result<Option<SourceEvent>> {
        std::future::pending().await
    }

    async fn close(&mut self) -> Result<()> {
        self.probe.source_closes.fetch_add(1, Ordering::SeqCst);
        Ok(())
    }
}

#[async_trait]
impl StreamSource for EndingSource {
    fn capabilities(&self) -> SourceCapabilities {
        self.probe.capability_reads.fetch_add(1, Ordering::SeqCst);
        exact_source_capabilities()
    }

    async fn open(&mut self, _cursor: Option<Cursor>) -> Result<()> {
        self.probe.source_opens.fetch_add(1, Ordering::SeqCst);
        Ok(())
    }

    async fn next(&mut self) -> Result<Option<SourceEvent>> {
        Ok(None)
    }

    async fn close(&mut self) -> Result<()> {
        self.probe.source_closes.fetch_add(1, Ordering::SeqCst);
        Ok(())
    }
}

#[async_trait]
impl StreamSource for FailingSource {
    fn capabilities(&self) -> SourceCapabilities {
        exact_source_capabilities()
    }

    async fn open(&mut self, _cursor: Option<Cursor>) -> Result<()> {
        Err(CalcFlowError::Io {
            path: "/secret/connector/path".into(),
            source: std::io::Error::other("secret connector detail"),
        })
    }

    async fn next(&mut self) -> Result<Option<SourceEvent>> {
        unreachable!("a source that fails to open cannot be polled")
    }

    async fn close(&mut self) -> Result<()> {
        Ok(())
    }
}

struct TransactionalSink {
    probe: LifecycleProbe,
}

#[async_trait]
impl TransactionalStreamSink for TransactionalSink {
    async fn open(&mut self) -> Result<()> {
        self.probe.sink_opens.fetch_add(1, Ordering::SeqCst);
        Ok(())
    }

    async fn begin_epoch(&mut self, _epoch: calc_flow::Epoch) -> Result<()> {
        Ok(())
    }

    async fn write(&mut self, _batch: &Batch) -> Result<()> {
        Ok(())
    }

    async fn pre_commit(&mut self, _epoch: calc_flow::Epoch) -> Result<JsonMap> {
        Ok(JsonMap::new())
    }

    async fn commit(&mut self, _epoch: calc_flow::Epoch, _pre_commit: &JsonMap) -> Result<()> {
        Ok(())
    }

    async fn abort(
        &mut self,
        _epoch: calc_flow::Epoch,
        _pre_commit: Option<&JsonMap>,
    ) -> Result<()> {
        Ok(())
    }

    async fn recover(&mut self, recovery: &SinkRecovery) -> Result<()> {
        self.probe
            .sink_recoveries
            .lock()
            .unwrap()
            .push(recovery.terminal());
        Ok(())
    }

    async fn close(&mut self) -> Result<()> {
        self.probe.sink_closes.fetch_add(1, Ordering::SeqCst);
        Ok(())
    }
}

fn continuous_plan() -> calc_flow::StreamExecutionPlan {
    PipelineBuilder::new("public_continuous")
        .unwrap()
        .add_node(
            "expression",
            Box::new(
                ExpressionOperator::new(
                    "expression",
                    "total = a + b",
                    Vec::new(),
                    None,
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
        .unwrap()
}

fn unproven_operator_plan() -> calc_flow::StreamExecutionPlan {
    let operator = UnionOperator::new(
        "external",
        ["left", "right"]
            .into_iter()
            .map(|name| Port::new(name, BatchKind::Table, true, None).unwrap())
            .collect(),
    )
    .unwrap();
    PipelineBuilder::new("unproven_operator")
        .unwrap()
        .add_node("external", Box::new(operator) as Box<dyn StreamOperator>)
        .unwrap()
        .compile_stream(
            &UdfRegistry::new().snapshot(),
            &StreamRequirements::default(),
        )
        .unwrap()
}

fn continuous_runner(root: &Path, probe: LifecycleProbe) -> StreamingRunner {
    let plan = continuous_plan();
    let source_id = plan.source_binding_ids()[0].to_owned();
    let output_id = plan.sink_binding_ids()[0].to_owned();
    StreamingRunner::new(
        plan,
        BTreeMap::from([(
            source_id,
            SourceBinding::new(PendingSource {
                probe: probe.clone(),
            }),
        )]),
        BTreeMap::from([(
            output_id,
            vec![SinkBinding::transactional("sink", TransactionalSink { probe }).unwrap()],
        )]),
        ManagedCheckpointRuntime::new(root).unwrap(),
    )
    .unwrap()
}

fn ending_runner(root: &Path, probe: LifecycleProbe) -> StreamingRunner {
    let plan = continuous_plan();
    let source_id = plan.source_binding_ids()[0].to_owned();
    let output_id = plan.sink_binding_ids()[0].to_owned();
    StreamingRunner::new(
        plan,
        BTreeMap::from([(
            source_id,
            SourceBinding::new(EndingSource {
                probe: probe.clone(),
            }),
        )]),
        BTreeMap::from([(
            output_id,
            vec![SinkBinding::transactional("sink", TransactionalSink { probe }).unwrap()],
        )]),
        ManagedCheckpointRuntime::new(root).unwrap(),
    )
    .unwrap()
}

#[tokio::test]
async fn public_job_checkpoints_and_cancel_settles_connectors() {
    let probe = LifecycleProbe::default();
    let plan = continuous_plan();
    let source_id = plan.source_binding_ids()[0].to_owned();
    let output_id = plan.sink_binding_ids()[0].to_owned();
    let directory = tempfile::tempdir().unwrap();
    let root = directory.path().join("managed");
    let runner = StreamingRunner::new(
        plan,
        BTreeMap::from([(
            source_id,
            SourceBinding::new(PendingSource {
                probe: probe.clone(),
            }),
        )]),
        BTreeMap::from([(
            output_id,
            vec![
                SinkBinding::transactional(
                    "sink",
                    TransactionalSink {
                        probe: probe.clone(),
                    },
                )
                .unwrap(),
            ],
        )]),
        ManagedCheckpointRuntime::new(root).unwrap(),
    )
    .unwrap();

    let job = runner.start().await.unwrap();
    let epoch = job.trigger_checkpoint().await.unwrap();

    assert_eq!(epoch, calc_flow::Epoch::INITIAL);
    assert_eq!(job.status().checkpoint.last_completed_epoch, Some(epoch));
    let outcome = job.cancel().await;
    assert_eq!(outcome.state, JobState::Cancelled);
    assert_eq!(probe.capability_reads.load(Ordering::SeqCst), 1);
    assert_eq!(probe.source_opens.load(Ordering::SeqCst), 1);
    assert_eq!(probe.source_closes.load(Ordering::SeqCst), 1);
    assert_eq!(probe.sink_opens.load(Ordering::SeqCst), 1);
    assert_eq!(probe.sink_closes.load(Ordering::SeqCst), 1);
}

#[test]
fn runner_shape_validation_is_safe_and_side_effect_free() {
    let probe = LifecycleProbe::default();
    let plan = continuous_plan();
    let source_id = plan.source_binding_ids()[0].to_owned();
    let directory = tempfile::tempdir().unwrap();
    let root = directory.path().join("not-created");

    let result = StreamingRunner::new(
        plan,
        BTreeMap::from([(
            source_id,
            SourceBinding::new(PendingSource {
                probe: probe.clone(),
            }),
        )]),
        BTreeMap::new(),
        ManagedCheckpointRuntime::new(&root).unwrap(),
    );

    let error = match result {
        Ok(_) => panic!("missing sink routes must fail runner validation"),
        Err(error) => error,
    };
    let CalcFlowError::Streaming(error) = error else {
        panic!("public runner errors must use the safe streaming boundary");
    };
    assert_eq!(error.category(), StreamingErrorCategory::Validation);
    assert_eq!(error.message(), "streaming job validation failed");
    assert_eq!(probe.capability_reads.load(Ordering::SeqCst), 0);
    assert_eq!(probe.source_opens.load(Ordering::SeqCst), 0);
    assert!(!root.exists());
}

#[tokio::test]
async fn start_errors_drop_raw_connector_details() {
    let probe = LifecycleProbe::default();
    let plan = continuous_plan();
    let source_id = plan.source_binding_ids()[0].to_owned();
    let expected_message = format!("source {source_id:?} open failed");
    let output_id = plan.sink_binding_ids()[0].to_owned();
    let directory = tempfile::tempdir().unwrap();
    let runner = StreamingRunner::new(
        plan,
        BTreeMap::from([(source_id, SourceBinding::new(FailingSource))]),
        BTreeMap::from([(
            output_id,
            vec![SinkBinding::transactional("sink", TransactionalSink { probe }).unwrap()],
        )]),
        ManagedCheckpointRuntime::new(directory.path().join("managed")).unwrap(),
    )
    .unwrap();

    let error = match runner.start().await {
        Ok(_) => panic!("the source open failure must reject start"),
        Err(error) => error,
    };
    let display = error.to_string();
    let debug = format!("{error:?}");
    let CalcFlowError::Streaming(error) = error else {
        panic!("public start errors must use the safe streaming boundary");
    };

    assert_eq!(error.category(), StreamingErrorCategory::Connector);
    assert_eq!(error.message(), expected_message);
    assert!(!display.contains("secret"));
    assert!(!debug.contains("secret"));
}

#[tokio::test]
async fn capability_mismatch_precedes_lifecycle_and_storage_side_effects() {
    let probe = LifecycleProbe::default();
    let plan = unproven_operator_plan();
    let sources = plan
        .source_binding_ids()
        .into_iter()
        .map(|source_id| {
            (
                source_id.to_owned(),
                SourceBinding::new(PendingSource {
                    probe: probe.clone(),
                }),
            )
        })
        .collect();
    let output_id = plan.sink_binding_ids()[0].to_owned();
    let directory = tempfile::tempdir().unwrap();
    let root = directory.path().join("not-created");
    let runner = StreamingRunner::new(
        plan,
        sources,
        BTreeMap::from([(
            output_id,
            vec![
                SinkBinding::transactional(
                    "sink",
                    TransactionalSink {
                        probe: probe.clone(),
                    },
                )
                .unwrap(),
            ],
        )]),
        ManagedCheckpointRuntime::new(&root).unwrap(),
    )
    .unwrap();

    let error = match runner.start().await {
        Ok(_) => panic!("an unproven operator must fail capability preflight"),
        Err(error) => error,
    };
    let CalcFlowError::Streaming(error) = error else {
        panic!("capability failures must use the safe streaming boundary");
    };

    assert_eq!(error.category(), StreamingErrorCategory::Validation);
    assert_eq!(probe.capability_reads.load(Ordering::SeqCst), 2);
    assert_eq!(probe.source_opens.load(Ordering::SeqCst), 0);
    assert_eq!(probe.sink_opens.load(Ordering::SeqCst), 0);
    assert!(!root.exists());
}

#[tokio::test]
async fn sink_recovery_distinguishes_non_terminal_checkpoints() {
    let probe = LifecycleProbe::default();
    let directory = tempfile::tempdir().unwrap();
    let root = directory.path().join("managed");

    let first = continuous_runner(&root, probe.clone())
        .start()
        .await
        .unwrap();
    first.trigger_checkpoint().await.unwrap();
    first.cancel().await;

    let restored = continuous_runner(&root, probe.clone())
        .start()
        .await
        .unwrap();
    assert_eq!(*probe.sink_recoveries.lock().unwrap(), [false]);
    restored.cancel().await;
}

#[tokio::test]
async fn terminal_manifest_recovery_does_not_reopen_sources() {
    let probe = LifecycleProbe::default();
    let directory = tempfile::tempdir().unwrap();
    let root = directory.path().join("managed");

    let first = ending_runner(&root, probe.clone()).start().await.unwrap();
    assert_eq!(first.wait().await.state, JobState::Completed);

    let restored = ending_runner(&root, probe.clone()).start().await.unwrap();
    assert_eq!(restored.wait().await.state, JobState::Completed);
    assert_eq!(*probe.sink_recoveries.lock().unwrap(), [true]);
    assert_eq!(probe.source_opens.load(Ordering::SeqCst), 1);
    assert_eq!(probe.source_closes.load(Ordering::SeqCst), 1);
}
