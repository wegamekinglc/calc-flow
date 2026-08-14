#[path = "support/restart_vector.rs"]
mod restart_vector;

use std::{
    any::Any,
    collections::BTreeMap,
    fmt,
    path::{Path, PathBuf},
    sync::{
        Arc, Mutex,
        atomic::{AtomicUsize, Ordering},
    },
};

use async_trait::async_trait;
use calc_flow::{
    Batch, BatchKind, BatchMetadata, CalcFlowError, ComponentKind, Cursor, DeliveryGuarantee,
    ExpressionOperator, ExternalPayload, JobState, JsonMap, ManagedCheckpointRuntime,
    NativeWatermarkCapability, PipelineBuilder, Port, ReplayPositioning, Result, SinkBinding,
    SinkRecovery, SourceBinding, SourceCapabilities, SourceDeliveryCapability, SourceEvent,
    SourceSchema, SourceStatus, StreamOperator, StreamRequirements, StreamSink, StreamSource,
    StreamingErrorCategory, StreamingJob, StreamingRunner, TransactionalStreamSink, UdfRegistry,
    UnionOperator,
};
use datafusion::arrow::{array::Int64Array, record_batch::RecordBatch};
use restart_vector::{RestartRecordVector, RestartVector, restart_vector};
use tokio::sync::{oneshot, watch};

const SECRET: &str = "credential-canary-value";

struct SecretPayload;

impl fmt::Debug for SecretPayload {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str(SECRET)
    }
}

impl ExternalPayload for SecretPayload {
    fn backend(&self) -> &'static str {
        "secret-probe"
    }

    fn len(&self) -> usize {
        1
    }

    fn estimated_bytes(&self) -> usize {
        1
    }

    fn as_any(&self) -> &dyn Any {
        self
    }
}

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

#[test]
fn public_connector_debug_redacts_batch_and_cursor_payloads() {
    let batch = Batch::external(
        Arc::new(SecretPayload),
        BatchMetadata::new(
            "source",
            0,
            BTreeMap::from([("secret".into(), SECRET.into())]),
        )
        .unwrap(),
    )
    .unwrap();
    let cursor = Cursor::new(
        "source",
        SECRET.as_bytes().to_vec(),
        BTreeMap::from([("secret".into(), SECRET.into())]),
    )
    .unwrap();
    let event = SourceEvent::Data { batch, cursor };

    for rendered in [format!("{event:?}"), format!("{event:#?}")] {
        assert!(!rendered.contains(SECRET));
        assert!(!rendered.contains("credential"));
    }
}

#[derive(Clone, Default)]
struct LifecycleProbe {
    capability_reads: Arc<AtomicUsize>,
    source_opens: Arc<AtomicUsize>,
    source_closes: Arc<AtomicUsize>,
    sink_opens: Arc<AtomicUsize>,
    sink_closes: Arc<AtomicUsize>,
    sink_recoveries: Arc<Mutex<Vec<bool>>>,
    sink_recovery_debugs: Arc<Mutex<Vec<String>>>,
}

struct PendingSource {
    probe: LifecycleProbe,
}

struct EndingSource {
    probe: LifecycleProbe,
}

struct LossySource {
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
impl StreamSource for LossySource {
    fn capabilities(&self) -> SourceCapabilities {
        self.probe.capability_reads.fetch_add(1, Ordering::SeqCst);
        SourceCapabilities {
            delivery: SourceDeliveryCapability::Lossy,
            ..exact_source_capabilities()
        }
    }

    async fn open(&mut self, _cursor: Option<Cursor>) -> Result<()> {
        self.probe.source_opens.fetch_add(1, Ordering::SeqCst);
        Ok(())
    }

    async fn next(&mut self) -> Result<Option<SourceEvent>> {
        std::future::pending().await
    }

    async fn close(&mut self) -> Result<()> {
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
        Ok(BTreeMap::from([("secret".into(), SECRET.into())]))
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
        self.probe
            .sink_recovery_debugs
            .lock()
            .unwrap()
            .extend([format!("{recovery:?}"), format!("{recovery:#?}")]);
        Ok(())
    }

    async fn close(&mut self) -> Result<()> {
        self.probe.sink_closes.fetch_add(1, Ordering::SeqCst);
        Ok(())
    }
}

fn continuous_plan() -> calc_flow::StreamExecutionPlan {
    continuous_plan_with_requirements(&StreamRequirements::default())
}

fn continuous_plan_with_requirements(
    requirements: &StreamRequirements,
) -> calc_flow::StreamExecutionPlan {
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
        .compile_stream(&UdfRegistry::new().snapshot(), requirements)
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
            output_id.clone(),
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
            output_id.clone(),
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
    assert_eq!(error.component_kind(), Some(ComponentKind::Sink));
    assert_eq!(error.component_id(), Some("output"));
    assert_eq!(
        error.message(),
        "sink bindings are missing graph output \"output\""
    );
    assert_eq!(probe.capability_reads.load(Ordering::SeqCst), 0);
    assert_eq!(probe.source_opens.load(Ordering::SeqCst), 0);
    assert!(!root.exists());
}

#[test]
fn runner_shape_errors_keep_safe_participant_coordinates() {
    let plan = continuous_plan();
    let source_id = plan.source_binding_ids()[0].to_owned();
    let output_id = plan.sink_binding_ids()[0].to_owned();
    let directory = tempfile::tempdir().unwrap();
    let checkpoint = || ManagedCheckpointRuntime::new(directory.path().join("managed")).unwrap();

    let unknown_source = match StreamingRunner::new(
        continuous_plan(),
        BTreeMap::from([("extra".into(), SourceBinding::new(FiniteSource))]),
        BTreeMap::from([(
            output_id.clone(),
            vec![
                SinkBinding::transactional(
                    "sink",
                    TransactionalSink {
                        probe: LifecycleProbe::default(),
                    },
                )
                .unwrap(),
            ],
        )]),
        checkpoint(),
    ) {
        Ok(_) => panic!("the unknown source must fail"),
        Err(error) => error,
    };
    let CalcFlowError::Streaming(unknown_source) = unknown_source else {
        panic!("shape errors must use the streaming boundary");
    };
    assert_eq!(unknown_source.component_kind(), Some(ComponentKind::Source));
    assert_eq!(unknown_source.component_id(), Some("extra"));
    assert_eq!(
        unknown_source.message(),
        "source binding \"extra\" does not match a compiled external input"
    );

    let missing_sink = match StreamingRunner::new(
        plan,
        BTreeMap::from([(source_id, SourceBinding::new(FiniteSource))]),
        BTreeMap::new(),
        checkpoint(),
    ) {
        Ok(_) => panic!("the missing sink must fail"),
        Err(error) => error,
    };
    let CalcFlowError::Streaming(missing_sink) = missing_sink else {
        panic!("shape errors must use the streaming boundary");
    };
    assert_eq!(missing_sink.component_kind(), Some(ComponentKind::Sink));
    assert_eq!(missing_sink.component_id(), Some(output_id.as_str()));
    assert_eq!(
        missing_sink.message(),
        format!("sink bindings are missing graph output {output_id:?}")
    );
}

#[test]
fn invalid_shape_ids_are_redacted_before_the_public_boundary() {
    let plan = continuous_plan();
    let output_id = plan.sink_binding_ids()[0].to_owned();
    let canary = "https://user:credential@host";
    let directory = tempfile::tempdir().unwrap();

    let error = match StreamingRunner::new(
        plan,
        BTreeMap::from([(canary.into(), SourceBinding::new(FiniteSource))]),
        BTreeMap::from([(
            output_id,
            vec![
                SinkBinding::transactional(
                    "sink",
                    TransactionalSink {
                        probe: LifecycleProbe::default(),
                    },
                )
                .unwrap(),
            ],
        )]),
        ManagedCheckpointRuntime::new(directory.path().join("managed")).unwrap(),
    ) {
        Ok(_) => panic!("the forbidden source ID must fail shape validation"),
        Err(error) => error,
    };
    let CalcFlowError::Streaming(error) = error else {
        panic!("shape errors must use the streaming boundary");
    };

    assert_eq!(error.category(), StreamingErrorCategory::Validation);
    assert_eq!(error.component_kind(), Some(ComponentKind::Source));
    assert_eq!(error.component_id(), None);
    assert_eq!(
        error.message(),
        "source binding ID is not a portable identifier"
    );
    for rendered in [error.to_string(), format!("{error:?}")] {
        assert!(!rendered.contains(canary));
        assert!(!rendered.contains("credential"));
    }
}

#[tokio::test]
async fn invalid_operator_ids_are_redacted_before_public_projection() {
    let canary = "https://user:credential@host";
    let plan = PipelineBuilder::new("invalid-operator")
        .unwrap()
        .add_node(
            canary,
            Box::new(
                ExpressionOperator::new(canary, "value = value", Vec::new(), None, Vec::new())
                    .unwrap(),
            ),
        )
        .unwrap()
        .compile_stream(
            &UdfRegistry::new().snapshot(),
            &StreamRequirements::default(),
        )
        .unwrap();
    let source_id = plan.source_binding_ids()[0].to_owned();
    let output_id = plan.sink_binding_ids()[0].to_owned();
    let directory = tempfile::tempdir().unwrap();
    let runner = StreamingRunner::new(
        plan,
        BTreeMap::from([(source_id, SourceBinding::new(FiniteSource))]),
        BTreeMap::from([(
            output_id,
            vec![
                SinkBinding::transactional(
                    "sink",
                    TransactionalSink {
                        probe: LifecycleProbe::default(),
                    },
                )
                .unwrap(),
            ],
        )]),
        ManagedCheckpointRuntime::new(directory.path().join("managed")).unwrap(),
    )
    .unwrap();

    let error = runner.start().await.unwrap_err();
    let CalcFlowError::Streaming(error) = error else {
        panic!("operator validation must use the streaming boundary");
    };
    assert_eq!(error.category(), StreamingErrorCategory::Validation);
    assert_eq!(error.component_kind(), Some(ComponentKind::Operator));
    assert_eq!(error.component_id(), None);
    assert_eq!(error.message(), "operator ID is not a portable identifier");
    for rendered in [error.to_string(), format!("{error:?}")] {
        assert!(!rendered.contains(canary));
        assert!(!rendered.contains("credential"));
    }
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
    assert_eq!(error.component_kind(), Some(ComponentKind::Operator));
    assert_eq!(error.component_id(), Some("external"));
    assert_eq!(
        error.message(),
        "operator \"external\" checkpoint capability is invalid"
    );
    assert_eq!(probe.capability_reads.load(Ordering::SeqCst), 2);
    assert_eq!(probe.source_opens.load(Ordering::SeqCst), 0);
    assert_eq!(probe.sink_opens.load(Ordering::SeqCst), 0);
    assert!(!root.exists());
}

#[tokio::test]
async fn delivery_capability_failure_names_the_safe_source_before_lifecycle() {
    let baseline = continuous_plan();
    let output_id = baseline.sink_binding_ids()[0].to_owned();
    let requirements = StreamRequirements {
        delivery: BTreeMap::from([(output_id.clone(), DeliveryGuarantee::ExactlyOnce)]),
    };
    let plan = continuous_plan_with_requirements(&requirements);
    let source_id = plan.source_binding_ids()[0].to_owned();
    let probe = LifecycleProbe::default();
    let directory = tempfile::tempdir().unwrap();
    let root = directory.path().join("not-created");
    let runner = StreamingRunner::new(
        plan,
        BTreeMap::from([(
            source_id.clone(),
            SourceBinding::new(LossySource {
                probe: probe.clone(),
            }),
        )]),
        BTreeMap::from([(
            output_id.clone(),
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
        Ok(_) => panic!("the lossy source must fail exactly-once preflight"),
        Err(error) => error,
    };
    let CalcFlowError::Streaming(error) = error else {
        panic!("capability failures must use the streaming boundary");
    };

    assert_eq!(error.category(), StreamingErrorCategory::Validation);
    assert_eq!(error.component_kind(), Some(ComponentKind::Source));
    assert_eq!(error.component_id(), Some(source_id.as_str()));
    assert_eq!(
        error.message(),
        format!("output {output_id:?} requires exactly_once but source {source_id:?} is lossy")
    );
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
    assert!(
        probe
            .sink_recovery_debugs
            .lock()
            .unwrap()
            .iter()
            .all(|rendered| !rendered.contains(SECRET))
    );
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

const RESTART_VECTOR_WRITE_WAIT_TIMEOUT: std::time::Duration = std::time::Duration::from_secs(10);

#[derive(Clone)]
struct RestartVectorProbe {
    opened_offsets: Arc<Mutex<Vec<usize>>>,
    source_closes: Arc<AtomicUsize>,
    sink_closes: Arc<AtomicUsize>,
    writes: watch::Sender<usize>,
}

impl Default for RestartVectorProbe {
    fn default() -> Self {
        let (writes, _) = watch::channel(0);
        Self {
            opened_offsets: Arc::default(),
            source_closes: Arc::default(),
            sink_closes: Arc::default(),
            writes,
        }
    }
}

impl RestartVectorProbe {
    async fn wait_for_writes(&self, expected: usize) {
        self.wait_for_writes_with_interlock(expected, None).await;
    }

    async fn wait_for_writes_with_interlock(
        &self,
        expected: usize,
        interlock: Option<(oneshot::Sender<()>, oneshot::Receiver<()>)>,
    ) {
        let mut writes = self.writes.subscribe();
        let mut interlock = interlock;
        tokio::time::timeout(RESTART_VECTOR_WRITE_WAIT_TIMEOUT, async {
            loop {
                if *writes.borrow_and_update() >= expected {
                    break;
                }
                if let Some((checked, resume)) = interlock.take() {
                    checked.send(()).expect("write-wait test remains active");
                    resume.await.expect("write-wait test resumes waiter");
                }
                writes
                    .changed()
                    .await
                    .expect("restart-vector write counter remains open");
            }
        })
        .await
        .unwrap_or_else(|_| {
            panic!(
                "restart sink wrote {} of {expected} rows within {:?}",
                *writes.borrow(),
                RESTART_VECTOR_WRITE_WAIT_TIMEOUT
            )
        });
    }
}

#[tokio::test]
async fn restart_write_wait_preserves_update_between_check_and_await() {
    let probe = RestartVectorProbe::default();
    let waiting_probe = probe.clone();
    let (checked_sender, checked_receiver) = oneshot::channel();
    let (resume_sender, resume_receiver) = oneshot::channel();
    let waiter = tokio::spawn(async move {
        waiting_probe
            .wait_for_writes_with_interlock(1, Some((checked_sender, resume_receiver)))
            .await;
    });

    checked_receiver.await.unwrap();
    probe.writes.send_modify(|writes| *writes += 1);
    resume_sender.send(()).unwrap();
    tokio::time::timeout(std::time::Duration::from_millis(500), waiter)
        .await
        .expect("write wait lost an update before registering its await")
        .unwrap();
}

struct RestartVectorSource {
    records: Arc<[RestartRecordVector]>,
    pause_at: Option<usize>,
    offset: usize,
    probe: RestartVectorProbe,
}

#[async_trait]
impl StreamSource for RestartVectorSource {
    fn capabilities(&self) -> SourceCapabilities {
        SourceCapabilities {
            replay_positioning: ReplayPositioning::ExactPauseReportAndSeek,
            delivery: SourceDeliveryCapability::Lossless,
            max_batch_rows: 1,
            max_batch_bytes: 1024,
            schema: SourceSchema::DynamicOrUnknown,
            native_watermarks: NativeWatermarkCapability::NeverEmits,
        }
    }

    async fn open(&mut self, cursor: Option<Cursor>) -> Result<()> {
        self.offset = match cursor {
            Some(cursor) => usize::try_from(
                cursor
                    .payload()
                    .get("offset")
                    .and_then(serde_json::Value::as_u64)
                    .expect("restart cursor carries an integer offset"),
            )
            .unwrap(),
            None => 0,
        };
        self.probe.opened_offsets.lock().unwrap().push(self.offset);
        Ok(())
    }

    async fn next(&mut self) -> Result<Option<SourceEvent>> {
        if self.pause_at == Some(self.offset) {
            tokio::task::yield_now().await;
            return Ok(Some(SourceEvent::Idle));
        }
        let Some(record) = self.records.get(self.offset).copied() else {
            return Ok(None);
        };
        assert_eq!(record.offset, self.offset);
        self.offset += 1;
        let batch = RecordBatch::try_from_iter(vec![(
            "value",
            Arc::new(Int64Array::from(vec![record.value])) as _,
        )])
        .unwrap();
        Ok(Some(SourceEvent::Data {
            batch: Batch::table(
                vec![batch],
                BatchMetadata::new("a6-vector", record.offset as u64, BTreeMap::new())?,
            )?,
            cursor: Cursor::unbound(
                u64::try_from(self.offset).unwrap().to_be_bytes().to_vec(),
                BTreeMap::from([("offset".into(), self.offset.into())]),
            )?,
        }))
    }

    async fn close(&mut self) -> Result<()> {
        self.probe.source_closes.fetch_add(1, Ordering::SeqCst);
        Ok(())
    }
}

struct RestartVectorSink {
    root: PathBuf,
    pending: Vec<i64>,
    probe: RestartVectorProbe,
}

impl RestartVectorSink {
    fn values(pre_commit: &JsonMap) -> Vec<i64> {
        pre_commit["values"]
            .as_array()
            .unwrap()
            .iter()
            .map(|value| value.as_i64().unwrap())
            .collect()
    }

    async fn commit_values(&self, epoch: calc_flow::Epoch, values: &[i64]) -> Result<()> {
        tokio::fs::create_dir_all(&self.root)
            .await
            .map_err(|source| CalcFlowError::Io {
                path: self.root.to_string_lossy().into_owned(),
                source,
            })?;
        let target = self
            .root
            .join(format!("visible-{:020}.json", epoch.as_u64()));
        if target.exists() {
            let observed: Vec<i64> =
                serde_json::from_slice(&tokio::fs::read(&target).await.map_err(|source| {
                    CalcFlowError::Io {
                        path: target.to_string_lossy().into_owned(),
                        source,
                    }
                })?)
                .unwrap();
            assert_eq!(observed, values);
            return Ok(());
        }
        let temporary = self.root.join(format!(".tmp-{:020}.json", epoch.as_u64()));
        tokio::fs::write(&temporary, serde_json::to_vec(values).unwrap())
            .await
            .map_err(|source| CalcFlowError::Io {
                path: temporary.to_string_lossy().into_owned(),
                source,
            })?;
        tokio::fs::rename(&temporary, &target)
            .await
            .map_err(|source| CalcFlowError::Io {
                path: target.to_string_lossy().into_owned(),
                source,
            })
    }
}

#[async_trait]
impl TransactionalStreamSink for RestartVectorSink {
    async fn open(&mut self) -> Result<()> {
        Ok(())
    }

    async fn begin_epoch(&mut self, _epoch: calc_flow::Epoch) -> Result<()> {
        self.pending.clear();
        Ok(())
    }

    async fn write(&mut self, batch: &Batch) -> Result<()> {
        for record in batch.table_payload()?.batches() {
            let values = record
                .column_by_name("doubled")
                .unwrap()
                .as_any()
                .downcast_ref::<Int64Array>()
                .unwrap();
            self.pending.extend(values.values());
            self.probe
                .writes
                .send_modify(|writes| *writes += values.len());
        }
        Ok(())
    }

    async fn pre_commit(&mut self, _epoch: calc_flow::Epoch) -> Result<JsonMap> {
        Ok(BTreeMap::from([(
            "values".into(),
            serde_json::json!(self.pending),
        )]))
    }

    async fn commit(&mut self, epoch: calc_flow::Epoch, pre_commit: &JsonMap) -> Result<()> {
        self.commit_values(epoch, &Self::values(pre_commit)).await
    }

    async fn abort(
        &mut self,
        _epoch: calc_flow::Epoch,
        _pre_commit: Option<&JsonMap>,
    ) -> Result<()> {
        self.pending.clear();
        Ok(())
    }

    async fn recover(&mut self, recovery: &SinkRecovery) -> Result<()> {
        self.commit_values(recovery.epoch(), &Self::values(recovery.pre_commit()))
            .await
    }

    async fn close(&mut self) -> Result<()> {
        self.probe.sink_closes.fetch_add(1, Ordering::SeqCst);
        Ok(())
    }
}

fn restart_vector_plan(vector: &RestartVector) -> calc_flow::StreamExecutionPlan {
    let requirements = StreamRequirements {
        delivery: BTreeMap::from([(
            vector.plan.output_id.clone(),
            DeliveryGuarantee::ExactlyOnce,
        )]),
    };
    PipelineBuilder::new(&vector.plan.name)
        .unwrap()
        .add_node(
            &vector.plan.operator_id,
            Box::new(
                ExpressionOperator::new(
                    &vector.plan.operator_id,
                    &vector.plan.expression,
                    Vec::new(),
                    None,
                    Vec::new(),
                )
                .unwrap(),
            ),
        )
        .unwrap()
        .compile_stream(&UdfRegistry::new().snapshot(), &requirements)
        .unwrap()
}

fn restart_vector_runner(
    vector: &RestartVector,
    managed_root: &Path,
    sink_root: &Path,
    pause_at: Option<usize>,
    probe: RestartVectorProbe,
) -> StreamingRunner {
    StreamingRunner::new(
        restart_vector_plan(vector),
        BTreeMap::from([(
            vector.plan.source_id.clone(),
            SourceBinding::new(RestartVectorSource {
                records: vector.records.clone().into(),
                pause_at,
                offset: 0,
                probe: probe.clone(),
            })
            .with_watermark_policy(calc_flow::WatermarkPolicy::Disabled { idle_timeout: None }),
        )]),
        BTreeMap::from([(
            vector.plan.output_id.clone(),
            vec![
                SinkBinding::transactional(
                    &vector.plan.sink_id,
                    RestartVectorSink {
                        root: sink_root.to_owned(),
                        pending: Vec::new(),
                        probe,
                    },
                )
                .unwrap(),
            ],
        )]),
        ManagedCheckpointRuntime::new(managed_root).unwrap(),
    )
    .unwrap()
}

async fn restart_vector_visible_values(root: &Path) -> Vec<i64> {
    let mut entries = tokio::fs::read_dir(root).await.unwrap();
    let mut paths = Vec::new();
    while let Some(entry) = entries.next_entry().await.unwrap() {
        if entry.file_name().to_str().is_some_and(|name| {
            name.starts_with("visible-")
                && Path::new(name)
                    .extension()
                    .is_some_and(|extension| extension.eq_ignore_ascii_case("json"))
        }) {
            paths.push(entry.path());
        }
    }
    paths.sort();
    let mut values = Vec::new();
    for path in paths {
        values.extend(
            serde_json::from_slice::<Vec<i64>>(&tokio::fs::read(path).await.unwrap()).unwrap(),
        );
    }
    values
}

fn restart_vector_temporary_artifacts(root: &Path) -> usize {
    let mut pending = vec![root.to_owned()];
    let mut count = 0;
    while let Some(directory) = pending.pop() {
        let Ok(entries) = std::fs::read_dir(directory) else {
            continue;
        };
        for entry in entries.map(std::result::Result::unwrap) {
            if entry.file_type().unwrap().is_dir() {
                pending.push(entry.path());
            } else if entry
                .file_name()
                .to_str()
                .is_some_and(|name| name.contains(".tmp") || name.starts_with("tmp"))
            {
                count += 1;
            }
        }
    }
    count
}

#[tokio::test]
async fn shared_restart_vector_is_exactly_once_across_checkpoint_recovery() {
    let vector = restart_vector().await;
    let original = vector.clone();
    let directory = tempfile::tempdir().unwrap();
    let managed_root = directory.path().join("managed");
    let sink_root = directory.path().join("sink");
    let probe = RestartVectorProbe::default();

    let first = restart_vector_runner(
        &vector,
        &managed_root,
        &sink_root,
        Some(vector.checkpoint_after),
        probe.clone(),
    )
    .start()
    .await
    .unwrap();
    probe.wait_for_writes(vector.checkpoint_after).await;
    let epoch = first.trigger_checkpoint().await.unwrap();
    assert_eq!(epoch.as_u64(), vector.expected.checkpoint_epoch);
    let first_outcome = first.cancel().await;
    assert_eq!(first_outcome.state, JobState::Cancelled);
    assert_eq!(first.status().task_count, vector.expected.terminal_tasks);
    assert_eq!(
        first
            .status()
            .edges
            .values()
            .filter(|edge| {
                edge.current_envelopes != 0 || edge.current_rows != 0 || edge.current_bytes != 0
            })
            .count(),
        vector.expected.terminal_charged_edges
    );

    let second = restart_vector_runner(&vector, &managed_root, &sink_root, None, probe.clone())
        .start()
        .await
        .unwrap();
    let second_outcome = second.wait().await;
    assert_eq!(second_outcome.state, JobState::Completed);
    assert_eq!(
        second_outcome.completed_epoch.map(calc_flow::Epoch::as_u64),
        Some(vector.expected.terminal_epoch)
    );
    let status = second.status();
    assert_eq!(status.task_count, vector.expected.terminal_tasks);
    assert_eq!(
        status
            .edges
            .values()
            .filter(|edge| {
                edge.current_envelopes != 0 || edge.current_rows != 0 || edge.current_bytes != 0
            })
            .count(),
        vector.expected.terminal_charged_edges
    );

    let values = restart_vector_visible_values(&sink_root).await;
    let unique = values
        .iter()
        .copied()
        .collect::<std::collections::BTreeSet<_>>();
    let expected = vector
        .expected
        .values
        .iter()
        .copied()
        .collect::<std::collections::BTreeSet<_>>();
    assert_eq!(values, vector.expected.values);
    assert_eq!(values.len() - unique.len(), vector.expected.duplicates);
    assert_eq!(
        expected.difference(&unique).count(),
        vector.expected.missing
    );
    assert_eq!(
        *probe.opened_offsets.lock().unwrap(),
        vector.expected.opened_offsets
    );
    assert_eq!(probe.source_closes.load(Ordering::SeqCst), 2);
    assert_eq!(probe.sink_closes.load(Ordering::SeqCst), 2);
    assert_eq!(
        restart_vector_temporary_artifacts(directory.path()),
        vector.expected.temporary_artifacts
    );
    assert_eq!(vector, original);
}
