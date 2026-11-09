//! Managed recovery must validate rolling indicator state before connectors open.

use std::{
    collections::BTreeMap,
    io::Cursor as IoCursor,
    path::{Path, PathBuf},
    sync::{
        Arc, Mutex,
        atomic::{AtomicUsize, Ordering},
    },
    time::Duration,
};

use async_trait::async_trait;
use calc_flow::{
    Batch, BatchMetadata, CancellationToken, CheckpointManifest, CheckpointManifestFields, Cursor,
    EdgeCollector, Epoch, EventTime, JobState, JsonMap, ManagedCheckpointRuntime,
    NativeWatermarkCapability, OperatorMetadata, PipelineBuilder, ReplayPositioning, Result,
    RollingOperator, RollingSpec, SinkBinding, SourceBinding, SourceCapabilities,
    SourceDeliveryCapability, SourceEvent, SourceSchema, StateBudget, StateHandle,
    StreamExecutionPlan, StreamJobContext, StreamOperator, StreamOperatorContext,
    StreamRequirements, StreamSink, StreamSource, StreamingRunner, UdfRegistry,
};
use datafusion::arrow::{
    array::{
        Array, ArrayRef, BinaryArray, Float64Array, StringArray, TimestampMicrosecondArray,
        UInt64Array,
    },
    datatypes::{DataType, Field, Schema, TimeUnit},
    ipc::{reader::FileReader, writer::FileWriter},
    record_batch::RecordBatch,
};
use serde_json::json;
use sha2::{Digest as _, Sha256};
use tokio::sync::mpsc::{UnboundedReceiver, UnboundedSender, unbounded_channel};

#[derive(Clone, Default)]
struct Probe {
    sources: Arc<AtomicUsize>,
    sinks: Arc<AtomicUsize>,
    writes: Arc<AtomicUsize>,
    cursors: Arc<Mutex<Vec<Vec<u8>>>>,
}

fn schema() -> Arc<Schema> {
    Arc::new(Schema::new(vec![
        Field::new(
            "ts",
            DataType::Timestamp(TimeUnit::Microsecond, Some("UTC".into())),
            false,
        ),
        Field::new("entity", DataType::Utf8, false),
        Field::new("seq", DataType::UInt64, false),
        Field::new("price", DataType::Float64, true),
        Field::new("high", DataType::Float64, true),
        Field::new("low", DataType::Float64, true),
        Field::new("close", DataType::Float64, true),
    ]))
}

fn input(position: usize) -> Batch {
    let (times, price, high, low, close) = match position {
        0 => ([1, 2], [1.0, 2.0], [10.0, 12.0], [8.0, 9.0], [9.0, 10.0]),
        1 => ([3, 4], [1.0, 3.0], [11.0, 14.0], [8.0, 11.0], [9.0, 13.0]),
        _ => panic!("the replay fixture contains exactly two batches"),
    };
    let columns: Vec<ArrayRef> = vec![
        Arc::new(TimestampMicrosecondArray::from(times.to_vec()).with_timezone("UTC")),
        Arc::new(StringArray::from(vec!["a"; 2])),
        Arc::new(UInt64Array::from(
            times.map(|value| u64::try_from(value).unwrap()).to_vec(),
        )),
        Arc::new(Float64Array::from(price.to_vec())),
        Arc::new(Float64Array::from(high.to_vec())),
        Arc::new(Float64Array::from(low.to_vec())),
        Arc::new(Float64Array::from(close.to_vec())),
    ];
    Batch::table(
        vec![RecordBatch::try_new(schema(), columns).unwrap()],
        BatchMetadata::default(),
    )
    .unwrap()
}

fn indicator_operator() -> RollingOperator {
    let spec: RollingSpec = serde_json::from_value(json!({
        "configuration_version":1,"state_layout_version":5,
        "partition_by":["entity"],"event_time":"ts","sequence_by":["seq"],
        "outputs":[
            {"kind":"rsi","primitive_version":1,"input":"price","output":"rsi","period":2},
            {"kind":"atr","primitive_version":1,"high":"high","low":"low","close":"close","output":"atr","period":2},
            {"kind":"lag","primitive_version":1,"input":"price","output":"lag","periods":1}
        ],
        "allowed_lateness_micros":0,"late_policy":{"kind":"error","scope":"envelope"},
        "value_policy":"stateful_numeric_v1"
    })).unwrap();
    RollingOperator::new("features", schema(), spec).unwrap()
}

fn plan() -> StreamExecutionPlan {
    let operator = indicator_operator();
    PipelineBuilder::new("rolling-indicator-recovery")
        .unwrap()
        .add_node("features", operator)
        .unwrap()
        .compile_stream(
            &UdfRegistry::new().snapshot(),
            &StreamRequirements::default(),
        )
        .unwrap()
}

#[tokio::test]
async fn test_indicator_unchanged_preparation_shares_bytes_with_distinct_epoch_descriptors() {
    let mut operator = indicator_operator();
    let job = StreamJobContext::new(
        71,
        "a".repeat(64),
        JsonMap::new(),
        None,
        CancellationToken::new(),
    );
    let context = StreamOperatorContext::new(&job, "features", None);
    let mut output = EdgeCollector::new(operator.output_ports().to_vec());
    operator
        .process_data("input", input(0), &context, &mut output)
        .await
        .unwrap();
    operator
        .on_watermark(EventTime::from_micros(2), &context, &mut output)
        .await
        .unwrap();
    operator.prepare_checkpoint_async(&context).await.unwrap();
    let first = operator.checkpoint(Epoch::new(1).unwrap()).unwrap();
    operator.prepare_checkpoint_async(&context).await.unwrap();
    let second = operator.checkpoint(Epoch::new(2).unwrap()).unwrap();
    assert_eq!(
        first.segments.keys().next().unwrap(),
        "base-00000000000000000001-00000000"
    );
    assert_eq!(
        second.segments.keys().next().unwrap(),
        "base-00000000000000000002-00000000"
    );
    let first_bytes = first.segments.values().next().unwrap();
    let second_bytes = second.segments.values().next().unwrap();
    assert_eq!(
        first_bytes.bytes().as_ptr(),
        second_bytes.bytes().as_ptr(),
        "unchanged state must reuse the prepared immutable allocation"
    );
    assert_eq!(first_bytes.sha256(), second_bytes.sha256());
    for (index, snapshot) in [first, second].iter().enumerate() {
        assert_eq!(snapshot.inline_metadata["epoch"], json!(index + 1));
        let mut restored = indicator_operator();
        restored.restore_async(snapshot, &context).await.unwrap();
        let mut output = EdgeCollector::new(restored.output_ports().to_vec());
        restored
            .process_data("input", input(1), &context, &mut output)
            .await
            .unwrap();
        restored.on_end(&context, &mut output).await.unwrap();
        let messages = output.drain("output");
        assert_eq!(messages.len(), 1);
        let batch = messages[0].as_data().unwrap();
        assert_eq!(values(batch, "rsi"), [Some(50.0), Some(100.0 / 1.2)]);
        assert_eq!(values(batch, "atr"), [Some(3.0), Some(4.0)]);
        assert_eq!(values(batch, "lag"), [Some(2.0), Some(1.0)]);
    }
}

fn old_only_plan() -> StreamExecutionPlan {
    let declaration: RollingSpec = serde_json::from_value(json!({
        "configuration_version":1,"state_layout_version":5,
        "partition_by":["entity"],"event_time":"ts","sequence_by":["seq"],
        "outputs":[{"kind":"cumulative_sum","primitive_version":1,"input":"price","output":"total","min_periods":1}],
        "allowed_lateness_micros":0,"late_policy":{"kind":"error","scope":"envelope"},
        "value_policy":"stateful_numeric_v1"
    })).unwrap();
    let mut operator = RollingOperator::new("features", schema(), declaration).unwrap();
    operator
        .set_state_budget(StateBudget::new(32, 2048).unwrap())
        .unwrap();
    PipelineBuilder::new("old-only-layout-five-recovery")
        .unwrap()
        .add_node("features", operator)
        .unwrap()
        .compile_stream(
            &UdfRegistry::new().snapshot(),
            &StreamRequirements::default(),
        )
        .unwrap()
}

struct Replay {
    source: String,
    position: usize,
    pending_watermark: bool,
    events: UnboundedReceiver<()>,
    probe: Probe,
}

#[async_trait]
impl StreamSource for Replay {
    fn capabilities(&self) -> SourceCapabilities {
        SourceCapabilities {
            replay_positioning: ReplayPositioning::ExactPauseReportAndSeek,
            delivery: SourceDeliveryCapability::Lossless,
            max_batch_rows: 2,
            max_batch_bytes: 4096,
            schema: SourceSchema::Exact(schema()),
            native_watermarks: NativeWatermarkCapability::EmitsNative,
        }
    }

    async fn open(&mut self, cursor: Option<Cursor>) -> Result<()> {
        self.probe.sources.fetch_add(1, Ordering::SeqCst);
        if let Some(cursor) = cursor {
            self.position = usize::from(cursor.order()[0]);
            self.probe
                .cursors
                .lock()
                .unwrap()
                .push(cursor.order().to_vec());
        }
        Ok(())
    }

    async fn next(&mut self) -> Result<Option<SourceEvent>> {
        if self.pending_watermark {
            self.pending_watermark = false;
            let watermark = i64::try_from(self.position * 2).unwrap();
            return Ok(Some(SourceEvent::Watermark(EventTime::from_micros(
                watermark,
            ))));
        }
        if self.events.recv().await.is_none() {
            return Ok(None);
        }
        let batch = input(self.position);
        self.position += 1;
        self.pending_watermark = true;
        let cursor = Cursor::new(
            &self.source,
            vec![u8::try_from(self.position).unwrap()],
            JsonMap::new(),
        )?;
        Ok(Some(SourceEvent::Data { batch, cursor }))
    }

    async fn close(&mut self) -> Result<()> {
        Ok(())
    }
}

struct Collected {
    output: UnboundedSender<Batch>,
    probe: Probe,
}

#[async_trait]
impl StreamSink for Collected {
    async fn open(&mut self) -> Result<()> {
        self.probe.sinks.fetch_add(1, Ordering::SeqCst);
        Ok(())
    }
    async fn write(&mut self, batch: &Batch) -> Result<()> {
        self.probe.writes.fetch_add(1, Ordering::SeqCst);
        self.output.send(batch.clone()).unwrap();
        Ok(())
    }
    async fn close(&mut self) -> Result<()> {
        Ok(())
    }
}

fn connect(
    root: &Path,
    probe: &Probe,
) -> (
    StreamingRunner,
    UnboundedSender<()>,
    UnboundedReceiver<Batch>,
) {
    connect_plan(root, probe, plan())
}

fn connect_plan(
    root: &Path,
    probe: &Probe,
    plan: StreamExecutionPlan,
) -> (
    StreamingRunner,
    UnboundedSender<()>,
    UnboundedReceiver<Batch>,
) {
    let source = plan.source_binding_ids()[0].to_owned();
    let sink = plan.sink_binding_ids()[0].to_owned();
    let (send, events) = unbounded_channel();
    let (output, received) = unbounded_channel();
    let runner = StreamingRunner::new(
        plan,
        BTreeMap::from([(
            source.clone(),
            SourceBinding::new(Replay {
                source,
                position: 0,
                pending_watermark: false,
                events,
                probe: probe.clone(),
            }),
        )]),
        BTreeMap::from([(
            sink,
            vec![
                SinkBinding::ordinary(
                    "output",
                    Collected {
                        output,
                        probe: probe.clone(),
                    },
                )
                .unwrap(),
            ],
        )]),
        ManagedCheckpointRuntime::new(root).unwrap(),
    )
    .unwrap();
    (runner, send, received)
}

async fn complete(root: &Path, probe: &Probe, with_rows: bool) {
    let (runner, send, _received) = connect(root, probe);
    finish(runner, send, with_rows).await;
}

async fn finish(runner: StreamingRunner, send: UnboundedSender<()>, with_rows: bool) {
    if with_rows {
        send.send(()).unwrap();
        send.send(()).unwrap();
    }
    drop(send);
    let job = runner.start().await.unwrap();
    let outcome = tokio::time::timeout(Duration::from_secs(5), job.wait())
        .await
        .unwrap();
    assert_eq!(outcome.state, JobState::Completed, "{outcome:?}");
}

#[tokio::test]
async fn test_old_only_layout_five_terminal_keeps_encoded_storage_outside_retained_limit() {
    let directory = tempfile::tempdir().unwrap();
    let root = directory.path().join("managed");
    let probe = Probe::default();
    let (runner, send, _received) = connect_plan(&root, &probe, old_only_plan());
    finish(runner, send, true).await;
    let (_, saved) = manifest(&root).await;
    assert!(saved.operators()["features"].segments[0].byte_len() > 2048);
    let (runner, send, _received) = connect_plan(&root, &probe, old_only_plan());
    finish(runner, send, false).await;
    assert_eq!(probe.sources.load(Ordering::SeqCst), 1);
    assert_eq!(probe.sinks.load(Ordering::SeqCst), 2);
}

#[tokio::test]
async fn test_old_only_layout_five_terminal_validates_metadata_before_open() {
    let directory = tempfile::tempdir().unwrap();
    let root = directory.path().join("managed");
    let probe = Probe::default();
    let (runner, send, _received) = connect_plan(&root, &probe, old_only_plan());
    finish(runner, send, true).await;
    let (path, original) = manifest(&root).await;
    let mut operators = original.operators().clone();
    operators
        .get_mut("features")
        .unwrap()
        .inline_metadata
        .insert("configuration_hash".into(), json!("0".repeat(64)));
    let changed = changed_manifest(&original, operators);
    tokio::fs::write(path, changed.canonical_bytes().unwrap())
        .await
        .unwrap();
    let (runner, _send, _received) = connect_plan(&root, &probe, old_only_plan());
    assert_preopen_failure(runner.start().await, &probe, "configuration");
}

async fn receive(received: &mut UnboundedReceiver<Batch>) -> Batch {
    tokio::time::timeout(Duration::from_secs(5), received.recv())
        .await
        .unwrap()
        .unwrap()
}

fn values(batch: &Batch, name: &str) -> Vec<Option<f64>> {
    batch
        .table_payload()
        .unwrap()
        .batches()
        .iter()
        .flat_map(|record| {
            record
                .column_by_name(name)
                .unwrap()
                .as_any()
                .downcast_ref::<Float64Array>()
                .unwrap()
                .iter()
        })
        .collect()
}

#[tokio::test]
async fn test_indicator_managed_prefix_restarts_without_reseeding_mixed_groups() {
    let directory = tempfile::tempdir().unwrap();
    let root = directory.path().join("managed");
    let probe = Probe::default();
    let (runner, send, mut received) = connect(&root, &probe);
    let first = runner.start().await.unwrap();
    send.send(()).unwrap();
    let prefix = receive(&mut received).await;
    assert_eq!(values(&prefix, "rsi"), [None, None]);
    first.trigger_checkpoint().await.unwrap();
    assert_eq!(first.cancel().await.state, JobState::Cancelled);
    drop(send);

    let (runner, send, mut received) = connect(&root, &probe);
    let second = runner.start().await.unwrap();
    assert_eq!(*probe.cursors.lock().unwrap(), [vec![1]]);
    send.send(()).unwrap();
    let suffix = receive(&mut received).await;
    assert_eq!(values(&suffix, "rsi"), [Some(50.0), Some(100.0 / 1.2)]);
    assert_eq!(values(&suffix, "atr"), [Some(3.0), Some(4.0)]);
    assert_eq!(values(&suffix, "lag"), [Some(2.0), Some(1.0)]);
    drop(send);
    assert_eq!(second.wait().await.state, JobState::Completed);
    assert_eq!(probe.sources.load(Ordering::SeqCst), 2);
    assert_eq!(probe.sinks.load(Ordering::SeqCst), 2);
}

#[tokio::test]
async fn test_indicator_valid_terminal_restart_keeps_source_closed_and_recovers_sink() {
    let directory = tempfile::tempdir().unwrap();
    let root = directory.path().join("managed");
    let probe = Probe::default();
    complete(&root, &probe, true).await;
    let writes = probe.writes.load(Ordering::SeqCst);
    assert!(writes > 0);
    complete(&root, &probe, false).await;
    assert_eq!(probe.sources.load(Ordering::SeqCst), 1);
    assert_eq!(probe.sinks.load(Ordering::SeqCst), 2);
    assert_eq!(probe.writes.load(Ordering::SeqCst), writes);
}

#[tokio::test]
async fn test_indicator_empty_global_terminal_preserves_no_segment_representation() {
    let directory = tempfile::tempdir().unwrap();
    let root = directory.path().join("managed");
    let probe = Probe::default();
    complete(&root, &probe, false).await;
    let (_, manifest) = manifest(&root).await;
    assert!(manifest.operators()["features"].segments.is_empty());
    assert!(
        manifest.operators()["features"].inline_metadata["segment_inventory"]
            .as_array()
            .unwrap()
            .is_empty()
    );
    complete(&root, &probe, false).await;
    assert_eq!(probe.sources.load(Ordering::SeqCst), 1);
    assert_eq!(probe.sinks.load(Ordering::SeqCst), 2);
    assert_eq!(probe.writes.load(Ordering::SeqCst), 0);
}

async fn manifest(root: &Path) -> (PathBuf, CheckpointManifest) {
    let path = root.join("manifests/manifest-00000000000000000001.json");
    let manifest = CheckpointManifest::from_bytes(&tokio::fs::read(&path).await.unwrap()).unwrap();
    (path, manifest)
}

fn changed_manifest(
    original: &CheckpointManifest,
    operators: BTreeMap<String, calc_flow::OperatorManifestEntry>,
) -> CheckpointManifest {
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

fn corrupt_indicator(bytes: &[u8]) -> Vec<u8> {
    let mut reader = FileReader::try_new(IoCursor::new(bytes), None).unwrap();
    let record = reader.next().unwrap().unwrap();
    assert!(reader.next().is_none());
    let index = record.num_columns() - 1;
    let states = record
        .column(index)
        .as_any()
        .downcast_ref::<BinaryArray>()
        .unwrap();
    let mut changed = false;
    let replacement = states
        .iter()
        .map(|state| {
            state.map(|bytes| {
                let mut payload = bytes.to_vec();
                if !changed {
                    assert!(matches!(payload.len(), 56 | 64));
                    // Make has_previous invalid while preserving all Arrow and outer checksums.
                    payload[24..32].copy_from_slice(&2_u64.to_le_bytes());
                    changed = true;
                }
                payload
            })
        })
        .collect::<BinaryArray>();
    assert!(
        changed,
        "the fixture must actually contain a numerical indicator payload"
    );
    let mut columns = record.columns().to_vec();
    columns[index] = Arc::new(replacement);
    let corrupted = RecordBatch::try_new(record.schema(), columns).unwrap();
    let mut encoded = Vec::new();
    let mut writer = FileWriter::try_new(&mut encoded, &corrupted.schema()).unwrap();
    writer.write(&corrupted).unwrap();
    writer.finish().unwrap();
    drop(writer);
    encoded
}

async fn corrupt_saved_indicator(root: &Path) {
    corrupt_saved_segment(root, corrupt_indicator).await;
}

async fn corrupt_saved_segment(root: &Path, corrupt: fn(&[u8]) -> Vec<u8>) {
    let (path, original) = manifest(root).await;
    let entry = &original.operators()["features"];
    assert_eq!(entry.segments.len(), 1);
    let old = &entry.segments[0];
    let segment_path = root.join("state").join(old.relative_path());
    let encoded = corrupt(&tokio::fs::read(&segment_path).await.unwrap());
    let checksum = hex::encode(Sha256::digest(&encoded));
    let replacement = StateHandle::new(
        old.operator_id(),
        old.epoch(),
        old.segment_id(),
        old.relative_path(),
        u64::try_from(encoded.len()).unwrap(),
        &checksum,
    )
    .unwrap();
    let mut operators = original.operators().clone();
    let entry = operators.get_mut("features").unwrap();
    entry.segments = vec![replacement];
    let descriptor = &mut entry.inline_metadata.get_mut("segment_inventory").unwrap()[0]["handle"];
    descriptor["byte_len"] = json!(encoded.len());
    descriptor["sha256"] = json!(checksum);
    let changed = changed_manifest(&original, operators);
    tokio::fs::write(segment_path, encoded).await.unwrap();
    tokio::fs::write(path, changed.canonical_bytes().unwrap())
        .await
        .unwrap();
}

fn assert_preopen_failure(result: Result<calc_flow::StreamingJob>, probe: &Probe, field: &str) {
    let Err(calc_flow::CalcFlowError::Streaming(error)) = result else {
        panic!("invalid indicator state must fail before connectors open");
    };
    assert_eq!(
        error.category(),
        calc_flow::StreamingErrorCategory::CheckpointMismatch
    );
    assert_eq!(
        error.component_kind(),
        Some(calc_flow::ComponentKind::Operator)
    );
    assert_eq!(error.component_id(), Some("features"));
    assert!(error.message().contains(field), "{error}");
    assert_eq!(probe.sources.load(Ordering::SeqCst), 1);
    assert_eq!(probe.sinks.load(Ordering::SeqCst), 1);
}

#[tokio::test]
async fn test_indicator_terminal_rejects_bad_inner_state_with_valid_outer_checksums_before_open() {
    let directory = tempfile::tempdir().unwrap();
    let root = directory.path().join("managed");
    let probe = Probe::default();
    complete(&root, &probe, true).await;
    corrupt_saved_indicator(&root).await;
    let (runner, _send, _received) = connect(&root, &probe);
    assert_preopen_failure(runner.start().await, &probe, "state.format");
}

#[tokio::test]
async fn test_indicator_ordinary_rejects_bad_inner_state_with_valid_outer_checksums_before_open() {
    let directory = tempfile::tempdir().unwrap();
    let root = directory.path().join("managed");
    let probe = Probe::default();
    let (runner, send, mut received) = connect(&root, &probe);
    let first = runner.start().await.unwrap();
    send.send(()).unwrap();
    receive(&mut received).await;
    first.trigger_checkpoint().await.unwrap();
    assert_eq!(first.cancel().await.state, JobState::Cancelled);
    drop(send);
    corrupt_saved_indicator(&root).await;
    let (runner, _send, _received) = connect(&root, &probe);
    assert_preopen_failure(runner.start().await, &probe, "state.format");
}

#[tokio::test]
async fn test_indicator_completed_output_cannot_masquerade_as_empty_global_state() {
    let directory = tempfile::tempdir().unwrap();
    let root = directory.path().join("managed");
    let probe = Probe::default();
    complete(&root, &probe, true).await;
    let (path, original) = manifest(&root).await;
    let mut operators = original.operators().clone();
    let entry = operators.get_mut("features").unwrap();
    assert!(
        entry.inline_metadata["next_output_sequence"]
            .as_u64()
            .unwrap()
            > 0
    );
    entry.segments.clear();
    entry
        .inline_metadata
        .insert("segment_inventory".into(), json!([]));
    let changed = changed_manifest(&original, operators);
    tokio::fs::write(path, changed.canonical_bytes().unwrap())
        .await
        .unwrap();
    let (runner, _send, _received) = connect(&root, &probe);
    assert_preopen_failure(runner.start().await, &probe, "state.segments");
}

const METADATA_CANARY: &str = "rolling-metadata-credential-canary";

async fn assert_invalid_inline_metadata(field: &str, value: serde_json::Value) {
    let directory = tempfile::tempdir().unwrap();
    let root = directory.path().join("managed");
    let probe = Probe::default();
    complete(&root, &probe, true).await;
    let (path, original) = manifest(&root).await;
    let mut operators = original.operators().clone();
    operators
        .get_mut("features")
        .unwrap()
        .inline_metadata
        .insert(field.to_owned(), value);
    let changed = changed_manifest(&original, operators);
    tokio::fs::write(path, changed.canonical_bytes().unwrap())
        .await
        .unwrap();
    let (runner, _send, _received) = connect(&root, &probe);
    let result = runner.start().await;
    let Err(calc_flow::CalcFlowError::Streaming(error)) = result else {
        panic!("malformed checkpoint metadata must fail before opening connectors");
    };
    assert!(
        !error.message().contains(METADATA_CANARY),
        "unvalidated checkpoint metadata leaked: {error}"
    );
    assert_preopen_failure(
        Err(calc_flow::CalcFlowError::Streaming(error)),
        &probe,
        "state.metadata",
    );
}

#[tokio::test]
async fn test_indicator_metadata_type_error_does_not_echo_checkpoint_values() {
    assert_invalid_inline_metadata("state_layout_version", json!(METADATA_CANARY)).await;
}

#[tokio::test]
async fn test_indicator_unknown_metadata_key_is_not_echoed() {
    assert_invalid_inline_metadata(METADATA_CANARY, json!(0)).await;
}

fn mixed_cumulative_plan() -> StreamExecutionPlan {
    let spec: RollingSpec = serde_json::from_value(json!({
        "configuration_version":1,"state_layout_version":5,
        "partition_by":["entity"],"event_time":"ts","sequence_by":["seq"],
        "outputs":[
            {"kind":"rsi","primitive_version":1,"input":"price","output":"rsi","period":2},
            {"kind":"cumulative_sum","primitive_version":1,"input":"price","output":"total","min_periods":1}
        ],
        "allowed_lateness_micros":0,"late_policy":{"kind":"error","scope":"envelope"},
        "value_policy":"stateful_numeric_v1"
    })).unwrap();
    PipelineBuilder::new("private-cumulative-state")
        .unwrap()
        .add_node(
            "features",
            RollingOperator::new("features", schema(), spec).unwrap(),
        )
        .unwrap()
        .compile_stream(
            &UdfRegistry::new().snapshot(),
            &StreamRequirements::default(),
        )
        .unwrap()
}
fn corrupt_cumulative(bytes: &[u8]) -> Vec<u8> {
    let mut reader = FileReader::try_new(IoCursor::new(bytes), None).unwrap();
    let record = reader.next().unwrap().unwrap();
    assert!(reader.next().is_none());
    let index = record.num_columns() - 2;
    let states = record
        .column(index)
        .as_any()
        .downcast_ref::<BinaryArray>()
        .unwrap();
    let mut changed = false;
    let replacement = states
        .iter()
        .map(|state| {
            state.map(|bytes| {
                let mut value: serde_json::Value = serde_json::from_slice(bytes).unwrap();
                value["valid_count"] = json!(METADATA_CANARY);
                changed = true;
                serde_json::to_vec(&value).unwrap()
            })
        })
        .collect::<BinaryArray>();
    assert!(changed, "fixture must contain a cumulative summary");
    let mut columns = record.columns().to_vec();
    columns[index] = Arc::new(replacement);
    let corrupted = RecordBatch::try_new(record.schema(), columns).unwrap();
    let mut encoded = Vec::new();
    let mut writer = FileWriter::try_new(&mut encoded, &corrupted.schema()).unwrap();
    writer.write(&corrupted).unwrap();
    writer.finish().unwrap();
    drop(writer);
    encoded
}

#[tokio::test]
async fn test_indicator_cumulative_payload_type_error_is_private_before_connector_open() {
    let directory = tempfile::tempdir().unwrap();
    let root = directory.path().join("managed");
    let probe = Probe::default();
    let (runner, send, _received) = connect_plan(&root, &probe, mixed_cumulative_plan());
    finish(runner, send, true).await;
    corrupt_saved_segment(&root, corrupt_cumulative).await;
    let (runner, _send, _received) = connect_plan(&root, &probe, mixed_cumulative_plan());
    let Err(calc_flow::CalcFlowError::Streaming(error)) = runner.start().await else {
        panic!("malformed cumulative state must fail before connector open");
    };
    assert!(
        !error.message().contains(METADATA_CANARY),
        "unvalidated cumulative payload leaked: {error}"
    );
    assert_preopen_failure(
        Err(calc_flow::CalcFlowError::Streaming(error)),
        &probe,
        "state.format",
    );
}

#[tokio::test]
async fn test_indicator_oversized_invalid_metadata_is_rejected_before_connectors_open() {
    assert_invalid_inline_metadata(
        "state_layout_version",
        json!(METADATA_CANARY.repeat(40_000)),
    )
    .await;
    assert_invalid_inline_metadata(&METADATA_CANARY.repeat(40_000), json!(0)).await;
}
