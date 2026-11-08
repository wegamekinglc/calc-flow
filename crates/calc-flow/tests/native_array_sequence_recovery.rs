use std::{
    collections::BTreeMap,
    path::{Path, PathBuf},
    sync::{
        Arc, Mutex,
        atomic::{AtomicUsize, Ordering},
    },
    time::Duration,
};

use async_trait::async_trait;
use calc_flow::{
    Batch, BatchKind, BatchMetadata, CheckpointManifest, CheckpointManifestFields, Cursor,
    ExternalOperatorSpec, JobState, JsonMap, ManagedCheckpointRuntime, NativeWatermarkCapability,
    PipelineBuilder, Port, ReplayPositioning, Result, SinkBinding, SourceBinding,
    SourceCapabilities, SourceDeliveryCapability, SourceEvent, SourceSchema, StateHandle,
    StreamOperatorFactory, StreamRequirements, StreamSink, StreamSource, StreamingRunner,
    UdfRegistry,
    array::{NativeArray, NativeArraySequenceFactory},
};
use sha2::{Digest as _, Sha256};

#[derive(Clone, Default)]
struct Opens {
    sources: Arc<AtomicUsize>,
    sinks: Arc<AtomicUsize>,
}
struct EndingSource(Opens);
#[async_trait]
impl StreamSource for EndingSource {
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
    async fn open(&mut self, _cursor: Option<Cursor>) -> Result<()> {
        self.0.sources.fetch_add(1, Ordering::SeqCst);
        Ok(())
    }
    async fn next(&mut self) -> Result<Option<SourceEvent>> {
        Ok(None)
    }
    async fn close(&mut self) -> Result<()> {
        Ok(())
    }
}
struct Output(Opens);
#[async_trait]
impl StreamSink for Output {
    async fn open(&mut self) -> Result<()> {
        self.0.sinks.fetch_add(1, Ordering::SeqCst);
        Ok(())
    }
    async fn write(&mut self, _batch: &Batch) -> Result<()> {
        Ok(())
    }
    async fn close(&mut self) -> Result<()> {
        Ok(())
    }
}
fn runner(root: &Path, opens: Opens) -> StreamingRunner {
    runner_with(
        root,
        opens,
        serde_json::json!({"function":"ewma","span":3}),
        256 * 1024 * 1024,
    )
}
fn runner_with(
    root: &Path,
    opens: Opens,
    kernel: serde_json::Value,
    maximum: usize,
) -> StreamingRunner {
    let plan = sequence_plan(kernel, maximum);
    let source = plan.source_binding_ids()[0].to_owned();
    let sink = plan.sink_binding_ids()[0].to_owned();
    StreamingRunner::new(
        plan,
        BTreeMap::from([(
            source,
            SourceBinding::new(EndingSource(opens.clone()))
                .with_watermark_policy(calc_flow::WatermarkPolicy::Disabled { idle_timeout: None }),
        )]),
        BTreeMap::from([(
            sink,
            vec![SinkBinding::ordinary("output", Output(opens)).unwrap()],
        )]),
        ManagedCheckpointRuntime::new(root).unwrap(),
    )
    .unwrap()
}
fn sequence_operator(
    kernel: serde_json::Value,
    maximum: usize,
) -> Box<dyn calc_flow::StreamOperator> {
    let options = JsonMap::from([
        ("ir_version".into(), 1.into()),
        ("input_dtype".into(), "float64".into()),
        ("kernel".into(), kernel),
        (
            "execution".into(),
            serde_json::json!({"max_allocation_bytes":maximum}),
        ),
    ]);
    let spec = ExternalOperatorSpec::new("native", "array_sequence", "1", options).unwrap();
    NativeArraySequenceFactory::default()
        .create(
            &spec,
            vec![Port::new("input", BatchKind::Array, true, None).unwrap()],
            vec![Port::new("output", BatchKind::Array, true, None).unwrap()],
        )
        .unwrap()
}
fn sequence_plan(kernel: serde_json::Value, maximum: usize) -> calc_flow::StreamExecutionPlan {
    operator_plan(sequence_operator(kernel, maximum))
}
fn operator_plan(operator: Box<dyn calc_flow::StreamOperator>) -> calc_flow::StreamExecutionPlan {
    PipelineBuilder::new("native-sequence-recovery")
        .unwrap()
        .add_node("sequence", operator)
        .unwrap()
        .compile_stream(
            &UdfRegistry::new().snapshot(),
            &StreamRequirements::default(),
        )
        .unwrap()
}
async fn completed(root: &Path, opens: Opens) {
    complete(runner(root, opens)).await;
}
async fn complete(runner: StreamingRunner) {
    let job = runner.start().await.unwrap();
    let outcome = tokio::time::timeout(Duration::from_secs(5), job.wait())
        .await
        .unwrap();
    assert_eq!(outcome.state, JobState::Completed);
}

fn updated_manifest(original: &CheckpointManifest, handle: StateHandle) -> CheckpointManifest {
    let mut operators = original.operators().clone();
    operators.get_mut("sequence").unwrap().segments = vec![handle];
    CheckpointManifest::new(CheckpointManifestFields {
        pipeline_name: original.pipeline_name().into(),
        pipeline_fingerprint: original.pipeline_fingerprint().into(),
        runtime_config_hash: original.runtime_config_hash().into(),
        epoch: original.epoch(),
        created_at: original.created_at(),
        recovery_status: original.recovery_status(),
        sources: original.sources().clone(),
        operators,
        sinks: original.sinks().clone(),
        static_inputs: original.static_inputs().clone(),
    })
    .unwrap()
}
async fn saved(root: &Path) -> (PathBuf, CheckpointManifest, Vec<u8>) {
    let manifest_path = root.join("manifests/manifest-00000000000000000001.json");
    let manifest =
        CheckpointManifest::from_bytes(&tokio::fs::read(&manifest_path).await.unwrap()).unwrap();
    let handle = &manifest.operators()["sequence"].segments[0];
    let bytes = tokio::fs::read(root.join("state").join(handle.relative_path()))
        .await
        .unwrap();
    (manifest_path, manifest, bytes)
}
async fn replace_state(
    root: &Path,
    manifest_path: PathBuf,
    manifest: &CheckpointManifest,
    bytes: Vec<u8>,
) {
    let handle = &manifest.operators()["sequence"].segments[0];
    let checksum = hex::encode(Sha256::digest(&bytes));
    let replacement = StateHandle::new(
        handle.operator_id(),
        handle.epoch(),
        handle.segment_id(),
        handle.relative_path(),
        u64::try_from(bytes.len()).unwrap(),
        &checksum,
    )
    .unwrap();
    let changed = updated_manifest(manifest, replacement);
    tokio::fs::write(root.join("state").join(handle.relative_path()), bytes)
        .await
        .unwrap();
    tokio::fs::write(manifest_path, changed.canonical_bytes().unwrap())
        .await
        .unwrap();
}
async fn corrupt_layout_with_valid_managed_checksum(root: &Path) {
    let (path, manifest, mut bytes) = saved(root).await;
    assert_eq!(&bytes[..8], b"CFASEQ\0\0");
    bytes[8] = 2;
    replace_state(root, path, &manifest, bytes).await;
}
fn native_error(result: Result<calc_flow::StreamingJob>, field: &str) {
    let Err(calc_flow::CalcFlowError::Streaming(error)) = result else {
        panic!("native recovery must reject invalid state before opening connectors")
    };
    assert_eq!(
        error.category(),
        calc_flow::StreamingErrorCategory::CheckpointMismatch
    );
    assert_eq!(
        error.component_kind(),
        Some(calc_flow::ComponentKind::Operator)
    );
    assert_eq!(error.component_id(), Some("sequence"));
    assert!(error.message().contains(field));
}

#[tokio::test]
async fn test_native_sequence_valid_terminal_restart_keeps_sources_closed() {
    let directory = tempfile::tempdir().unwrap();
    let root = directory.path().join("managed");
    let opens = Opens::default();
    completed(&root, opens.clone()).await;
    completed(&root, opens.clone()).await;
    assert_eq!(opens.sources.load(Ordering::SeqCst), 1);
}

#[tokio::test]
async fn test_native_sequence_terminal_rejects_valid_checksum_with_bad_codec_before_open() {
    let directory = tempfile::tempdir().unwrap();
    let root = directory.path().join("managed");
    let opens = Opens::default();
    completed(&root, opens.clone()).await;
    corrupt_layout_with_valid_managed_checksum(&root).await;
    let result = runner(&root, opens.clone()).start().await;
    native_error(result, "state.layout");
    assert_eq!(opens.sources.load(Ordering::SeqCst), 1);
    assert_eq!(opens.sinks.load(Ordering::SeqCst), 1);
}

#[tokio::test]
async fn test_native_sequence_terminal_preflights_encoded_and_candidate_limits_before_open() {
    let directory = tempfile::tempdir().unwrap();
    let root = directory.path().join("managed");
    let opens = Opens::default();
    let kernel = serde_json::json!({"function":"rolling_mean","window":100});
    complete(runner_with(&root, opens.clone(), kernel.clone(), 600)).await;
    let (path, manifest, mut bytes) = saved(&root).await;
    assert_eq!(bytes.len(), 140);
    bytes[68..76].copy_from_slice(&100_u64.to_le_bytes());
    bytes[132..140].copy_from_slice(&100_u64.to_le_bytes());
    bytes.resize(1040, 0);
    replace_state(&root, path, &manifest, bytes).await;
    let result = runner_with(&root, opens.clone(), kernel, 600).start().await;
    native_error(result, "state.peak_bytes");
    assert_eq!(opens.sources.load(Ordering::SeqCst), 1);
    assert_eq!(opens.sinks.load(Ordering::SeqCst), 1);
}

struct Replay {
    values: Vec<Batch>,
    position: usize,
    events: tokio::sync::mpsc::UnboundedReceiver<()>,
    resumed: Arc<Mutex<Vec<Vec<u8>>>>,
    source: String,
}
#[async_trait]
impl StreamSource for Replay {
    fn capabilities(&self) -> SourceCapabilities {
        SourceCapabilities {
            replay_positioning: ReplayPositioning::ExactPauseReportAndSeek,
            delivery: SourceDeliveryCapability::Lossless,
            max_batch_rows: 2,
            max_batch_bytes: 1024,
            schema: SourceSchema::DynamicOrUnknown,
            native_watermarks: NativeWatermarkCapability::NeverEmits,
        }
    }
    async fn open(&mut self, cursor: Option<Cursor>) -> Result<()> {
        if let Some(cursor) = cursor {
            self.position = usize::from(cursor.order()[0]);
            self.resumed.lock().unwrap().push(cursor.order().to_vec());
        }
        Ok(())
    }
    async fn next(&mut self) -> Result<Option<SourceEvent>> {
        if self.events.recv().await.is_none() {
            return Ok(None);
        }
        let batch = self.values[self.position].clone();
        self.position += 1;
        Ok(Some(SourceEvent::Data {
            batch,
            cursor: Cursor::new(
                &self.source,
                vec![u8::try_from(self.position).unwrap()],
                JsonMap::new(),
            )?,
        }))
    }
    async fn close(&mut self) -> Result<()> {
        Ok(())
    }
}
struct Collected(tokio::sync::mpsc::UnboundedSender<Batch>);
#[async_trait]
impl StreamSink for Collected {
    async fn open(&mut self) -> Result<()> {
        Ok(())
    }
    async fn write(&mut self, batch: &Batch) -> Result<()> {
        self.0.send(batch.clone()).unwrap();
        Ok(())
    }
    async fn close(&mut self) -> Result<()> {
        Ok(())
    }
}
fn input(values: Vec<f64>) -> Batch {
    let shape = vec![values.len()];
    Batch::external(
        Arc::new(NativeArray::from_f64(values, shape, None).unwrap()),
        BatchMetadata::default(),
    )
    .unwrap()
}
fn replay_runner(
    root: &Path,
    resumed: Arc<Mutex<Vec<Vec<u8>>>>,
    events: tokio::sync::mpsc::UnboundedReceiver<()>,
    output: tokio::sync::mpsc::UnboundedSender<Batch>,
) -> StreamingRunner {
    let plan = sequence_plan(
        serde_json::json!({"function":"ewma", "span":3}),
        256 * 1024 * 1024,
    );
    replay_with_plan(root, plan, resumed, events, output)
}
fn replay_with_plan(
    root: &Path,
    plan: calc_flow::StreamExecutionPlan,
    resumed: Arc<Mutex<Vec<Vec<u8>>>>,
    events: tokio::sync::mpsc::UnboundedReceiver<()>,
    output: tokio::sync::mpsc::UnboundedSender<Batch>,
) -> StreamingRunner {
    let source = plan.source_binding_ids()[0].to_owned();
    let sink = plan.sink_binding_ids()[0].to_owned();
    StreamingRunner::new(
        plan,
        BTreeMap::from([(
            source.clone(),
            SourceBinding::new(Replay {
                values: vec![input(vec![1.0, 3.0]), input(vec![5.0, 7.0])],
                position: 0,
                events,
                resumed,
                source,
            })
            .with_watermark_policy(calc_flow::WatermarkPolicy::Disabled { idle_timeout: None }),
        )]),
        BTreeMap::from([(
            sink,
            vec![SinkBinding::ordinary("output", Collected(output)).unwrap()],
        )]),
        ManagedCheckpointRuntime::new(root).unwrap(),
    )
    .unwrap()
}
fn values(batch: &Batch) -> Vec<f64> {
    let payload = batch.external_payload().unwrap();
    let array = payload.as_any().downcast_ref::<NativeArray>().unwrap();
    let calc_flow::array::FloatValues::Float64(values) = array.float_view().unwrap().values()
    else {
        panic!("sequence results are float64")
    };
    values.to_vec()
}
#[tokio::test]
async fn test_native_sequence_new_job_replays_after_prefix_without_reseeding() {
    let directory = tempfile::tempdir().unwrap();
    let root = directory.path().join("managed");
    let resumed = Arc::new(Mutex::new(Vec::new()));
    let (send, events) = tokio::sync::mpsc::unbounded_channel();
    let (output, mut received) = tokio::sync::mpsc::unbounded_channel();
    let first = replay_runner(&root, Arc::clone(&resumed), events, output)
        .start()
        .await
        .unwrap();
    send.send(()).unwrap();
    assert_eq!(values(&received.recv().await.unwrap()), [1.0, 2.0]);
    first.trigger_checkpoint().await.unwrap();
    assert_eq!(first.cancel().await.state, JobState::Cancelled);
    drop(send);

    let (send, events) = tokio::sync::mpsc::unbounded_channel();
    let (output, mut received) = tokio::sync::mpsc::unbounded_channel();
    let second = replay_runner(&root, Arc::clone(&resumed), events, output)
        .start()
        .await
        .unwrap();
    assert_eq!(*resumed.lock().unwrap(), [vec![1]]);
    send.send(()).unwrap();
    assert_eq!(values(&received.recv().await.unwrap()), [3.5, 5.25]);
    drop(send);
    assert_eq!(second.wait().await.state, JobState::Completed);
}

#[test]
fn test_native_sequence_static_ingress_is_rejected_for_every_declared_backend() {
    let providers = calc_flow::ProviderRegistry::default();
    providers
        .register_stream(
            "native",
            "array_sequence",
            "1",
            Arc::new(NativeArraySequenceFactory::default()),
        )
        .unwrap();
    for backend in ["native", "numpy", "jax"] {
        let project: calc_flow::ProjectSpec = serde_json::from_value(serde_json::json!({
            "format_version":3,"id":"sequence-static","name":"sequence-static",
            "runtime":{"mode":"stream","options":{}},
            "static_inputs":[{"kind":"array","name":"input","mutability":"static",
                "backend":backend,"dtype":"float64","shape":[0]}],
            "graph":{"name":"sequence-static","nodes":[{"id":"sequence",
                "operator":{"kind":"external","provider":"native","name":"array_sequence","version":"1",
                    "options":{"ir_version":1,"input_dtype":"float64","kernel":{"function":"ewma","span":3}}},
                "input_ports":[{"name":"input","kind":"array"}],"output_ports":[{"name":"output","kind":"array"}]}]},
        })).unwrap();
        let result = calc_flow::compile_stream_project_graph(
            &project,
            &providers,
            &UdfRegistry::new().snapshot(),
            &StreamRequirements::default(),
        );
        assert!(
            matches!(&result, Err(calc_flow::CalcFlowError::InvalidArgument { field, .. }) if field == "static_inputs.input"),
            "an actual sequence requires dynamic ingress, including backend {backend}: {result:?}"
        );
    }
}

const LEGACY_SECRET: &str = "native-lookalike-credential-canary";
struct Lookalike(Box<dyn calc_flow::StreamOperator>, Arc<AtomicUsize>);
impl calc_flow::OperatorMetadata for Lookalike {
    fn name(&self) -> &str {
        self.0.name()
    }
    fn input_ports(&self) -> &[Port] {
        self.0.input_ports()
    }
    fn output_ports(&self) -> &[Port] {
        self.0.output_ports()
    }
    fn configuration(&self) -> JsonMap {
        self.0.configuration()
    }
}
#[async_trait]
impl calc_flow::StreamOperator for Lookalike {
    fn lifecycle(&self) -> calc_flow::StreamOperatorLifecycle {
        self.0.lifecycle()
    }
    async fn process_data(
        &mut self,
        ingress: &str,
        batch: Batch,
        context: &calc_flow::StreamOperatorContext<'_>,
        collector: &mut dyn calc_flow::StreamCollector,
    ) -> Result<()> {
        self.0
            .process_data(ingress, batch, context, collector)
            .await
    }
    async fn on_watermark(
        &mut self,
        watermark: calc_flow::EventTime,
        context: &calc_flow::StreamOperatorContext<'_>,
        collector: &mut dyn calc_flow::StreamCollector,
    ) -> Result<()> {
        self.0.on_watermark(watermark, context, collector).await
    }
    async fn on_end(
        &mut self,
        context: &calc_flow::StreamOperatorContext<'_>,
        collector: &mut dyn calc_flow::StreamCollector,
    ) -> Result<()> {
        self.0.on_end(context, collector).await
    }
    async fn prepare_checkpoint_async(
        &mut self,
        context: &calc_flow::StreamOperatorContext<'_>,
    ) -> Result<()> {
        self.0.prepare_checkpoint_async(context).await
    }
    fn checkpoint(&mut self, epoch: calc_flow::Epoch) -> Result<calc_flow::OperatorStateSnapshot> {
        self.0.checkpoint(epoch)
    }
    async fn restore_async(
        &mut self,
        _: &calc_flow::OperatorStateSnapshot,
        _: &calc_flow::StreamOperatorContext<'_>,
    ) -> Result<()> {
        self.1.fetch_add(1, Ordering::SeqCst);
        Err(calc_flow::CalcFlowError::CheckpointMismatch {
            message: LEGACY_SECRET.into(),
        })
    }
    fn reset(&mut self) -> Result<()> {
        self.0.reset()
    }
}
fn legacy_runner(
    root: &Path,
    restores: Arc<AtomicUsize>,
    events: tokio::sync::mpsc::UnboundedReceiver<()>,
    output: tokio::sync::mpsc::UnboundedSender<Batch>,
) -> StreamingRunner {
    let operator = sequence_operator(
        serde_json::json!({"function":"ewma", "span":3}),
        256 * 1024 * 1024,
    );
    let plan = operator_plan(Box::new(Lookalike(operator, restores)));
    replay_with_plan(root, plan, Arc::new(Mutex::new(Vec::new())), events, output)
}
#[tokio::test]
async fn test_native_name_lookalike_retains_legacy_managed_restore_redaction() {
    let directory = tempfile::tempdir().unwrap();
    let root = directory.path().join("managed");
    let (send, events) = tokio::sync::mpsc::unbounded_channel();
    let (output, mut received) = tokio::sync::mpsc::unbounded_channel();
    let restores = Arc::new(AtomicUsize::new(0));
    let first = legacy_runner(&root, Arc::clone(&restores), events, output)
        .start()
        .await
        .unwrap();
    send.send(()).unwrap();
    received.recv().await.unwrap();
    first.trigger_checkpoint().await.unwrap();
    first.cancel().await;
    drop(send);
    let (_send, events) = tokio::sync::mpsc::unbounded_channel();
    let (output, _received) = tokio::sync::mpsc::unbounded_channel();
    let Err(calc_flow::CalcFlowError::Streaming(error)) =
        legacy_runner(&root, Arc::clone(&restores), events, output)
            .start()
            .await
    else {
        panic!("custom restore must fail at managed entry")
    };
    assert_eq!(restores.load(Ordering::SeqCst), 1);
    assert_eq!(
        error.category(),
        calc_flow::StreamingErrorCategory::Internal
    );
    assert!(!error.message().contains(LEGACY_SECRET));
    assert!(!format!("{error:?}").contains(LEGACY_SECRET));
}
