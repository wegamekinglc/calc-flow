//! Managed MACD restores preserve carried signal state and reject invalid inner words.

use std::{
    collections::BTreeMap,
    io::Cursor as IoCursor,
    path::{Path, PathBuf},
    sync::{
        Arc,
        atomic::{AtomicUsize, Ordering},
    },
    time::Duration,
};

use async_trait::async_trait;
use calc_flow::{
    Batch, BatchMetadata, CheckpointManifest, CheckpointManifestFields, Cursor, EventTime,
    JobState, ManagedCheckpointRuntime, NativeWatermarkCapability, PipelineBuilder,
    ReplayPositioning, Result, RollingOperator, RollingSpec, SinkBinding, SourceBinding,
    SourceCapabilities, SourceDeliveryCapability, SourceEvent, SourceSchema, StateHandle,
    StreamExecutionPlan, StreamRequirements, StreamSink, StreamSource, StreamingRunner,
    UdfRegistry,
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
    ]))
}
fn input(position: usize) -> Batch {
    let (times, price) = match position {
        0 => ([1, 2], [Some(10.0), Some(14.0)]),
        1 => ([3, 4], [None, None]),
        _ => panic!("two replay batches"),
    };
    let arrays: Vec<ArrayRef> = vec![
        Arc::new(TimestampMicrosecondArray::from(times.to_vec()).with_timezone("UTC")),
        Arc::new(StringArray::from(vec!["a"; 2])),
        Arc::new(UInt64Array::from(
            times.map(|time| u64::try_from(time).unwrap()).to_vec(),
        )),
        Arc::new(Float64Array::from(price.to_vec())),
    ];
    Batch::table(
        vec![RecordBatch::try_new(schema(), arrays).unwrap()],
        BatchMetadata::default(),
    )
    .unwrap()
}
fn plan() -> StreamExecutionPlan {
    let mut outputs = ["line", "signal", "histogram"]
        .map(|component| {
            json!({
                "kind":"macd_component","primitive_version":1,"input":"price","output":component,
                "component":component,"fast_span":1,"slow_span":3,"signal_span":3,
                "min_periods":1,"signal_min_periods":1,
            })
        })
        .to_vec();
    outputs.extend([
        json!({"kind":"rsi","primitive_version":1,"input":"price","output":"rsi","period":1}),
        json!({"kind":"lag","primitive_version":1,"input":"price","output":"lag","periods":1}),
    ]);
    let spec: RollingSpec = serde_json::from_value(json!({
        "configuration_version":1,"state_layout_version":6,
        "partition_by":["entity"],"event_time":"ts","sequence_by":["seq"],"outputs":outputs,
        "allowed_lateness_micros":0,"late_policy":{"kind":"error","scope":"envelope"},
        "value_policy":"stateful_numeric_v1",
    }))
    .unwrap();
    PipelineBuilder::new("macd-managed-recovery")
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
struct Replay {
    id: String,
    position: usize,
    watermark: bool,
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
        }
        Ok(())
    }
    async fn next(&mut self) -> Result<Option<SourceEvent>> {
        if self.watermark {
            self.watermark = false;
            return Ok(Some(SourceEvent::Watermark(EventTime::from_micros(
                i64::try_from(self.position * 2).unwrap(),
            ))));
        }
        if self.events.recv().await.is_none() {
            return Ok(None);
        }
        let batch = input(self.position);
        self.position += 1;
        self.watermark = true;
        let cursor = Cursor::new(
            &self.id,
            vec![u8::try_from(self.position).unwrap()],
            BTreeMap::new(),
        )?;
        Ok(Some(SourceEvent::Data { batch, cursor }))
    }
    async fn close(&mut self) -> Result<()> {
        Ok(())
    }
}
struct Collected {
    probe: Probe,
    send: UnboundedSender<Batch>,
}
#[async_trait]
impl StreamSink for Collected {
    async fn open(&mut self) -> Result<()> {
        self.probe.sinks.fetch_add(1, Ordering::SeqCst);
        Ok(())
    }
    async fn write(&mut self, batch: &Batch) -> Result<()> {
        self.probe.writes.fetch_add(1, Ordering::SeqCst);
        self.send.send(batch.clone()).unwrap();
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
    let plan = plan();
    let source = plan.source_binding_ids()[0].to_owned();
    let sink = plan.sink_binding_ids()[0].to_owned();
    let (send, events) = unbounded_channel();
    let (output, received) = unbounded_channel();
    let runner = StreamingRunner::new(
        plan,
        BTreeMap::from([(
            source.clone(),
            SourceBinding::new(Replay {
                id: source,
                position: 0,
                watermark: false,
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
                        probe: probe.clone(),
                        send: output,
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
async fn save_prefix(root: &Path, probe: &Probe) {
    let (runner, send, mut output) = connect(root, probe);
    let job = runner.start().await.unwrap();
    send.send(()).unwrap();
    let batch = receive(&mut output).await;
    assert_eq!(values(&batch, "signal"), [Some(0.0), Some(1.0)]);
    job.trigger_checkpoint().await.unwrap();
    assert_eq!(job.cancel().await.state, JobState::Cancelled);
    drop(send);
}
async fn complete(root: &Path, probe: &Probe) {
    let (runner, send, _received) = connect(root, probe);
    send.send(()).unwrap();
    send.send(()).unwrap();
    drop(send);
    let job = runner.start().await.unwrap();
    let outcome = tokio::time::timeout(Duration::from_secs(5), job.wait())
        .await
        .unwrap();
    assert_eq!(outcome.state, JobState::Completed, "{outcome:?}");
}
#[tokio::test]
async fn test_macd_managed_normal_restart_keeps_carried_signal_and_mixed_legacy_groups() {
    let dir = tempfile::tempdir().unwrap();
    let root = dir.path().join("managed");
    let probe = Probe::default();
    save_prefix(&root, &probe).await;
    let (runner, send, mut output) = connect(&root, &probe);
    let job = runner.start().await.unwrap();
    send.send(()).unwrap();
    let suffix = receive(&mut output).await;
    assert_eq!(values(&suffix, "line"), [Some(2.0); 2]);
    assert_eq!(values(&suffix, "signal"), [Some(1.5), Some(1.75)]);
    assert_eq!(values(&suffix, "histogram"), [Some(0.5), Some(0.25)]);
    assert_eq!(values(&suffix, "rsi"), [None; 2]);
    assert_eq!(values(&suffix, "lag"), [Some(14.0), None]);
    drop(send);
    assert_eq!(job.wait().await.state, JobState::Completed);
    assert_eq!(probe.sources.load(Ordering::SeqCst), 2);
}
#[tokio::test]
async fn test_macd_managed_terminal_restart_keeps_source_closed() {
    let dir = tempfile::tempdir().unwrap();
    let root = dir.path().join("managed");
    let probe = Probe::default();
    complete(&root, &probe).await;
    let writes = probe.writes.load(Ordering::SeqCst);
    let (runner, send, _received) = connect(&root, &probe);
    drop(send);
    assert_eq!(
        runner.start().await.unwrap().wait().await.state,
        JobState::Completed
    );
    assert_eq!(probe.sources.load(Ordering::SeqCst), 1);
    assert_eq!(probe.sinks.load(Ordering::SeqCst), 2);
    assert_eq!(probe.writes.load(Ordering::SeqCst), writes);
}
async fn manifest(root: &Path) -> (PathBuf, CheckpointManifest) {
    let path = root.join("manifests/manifest-00000000000000000001.json");
    let manifest = CheckpointManifest::from_bytes(&tokio::fs::read(&path).await.unwrap()).unwrap();
    (path, manifest)
}
fn corrupt(bytes: &[u8]) -> Vec<u8> {
    let mut reader = FileReader::try_new(IoCursor::new(bytes), None).unwrap();
    let record = reader.next().unwrap().unwrap();
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
                if payload.len() == 104 {
                    payload[56..64].copy_from_slice(&u64::MAX.to_le_bytes());
                    changed = true;
                }
                payload
            })
        })
        .collect::<BinaryArray>();
    assert!(changed, "actual MACD state present");
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
async fn corrupt_saved_state(root: &Path) {
    let (path, original) = manifest(root).await;
    let old = &original.operators()["features"].segments[0];
    assert_eq!(
        original.operators()["features"].inline_metadata["state_layout_version"],
        json!(6)
    );
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
    tokio::fs::write(segment_path, encoded).await.unwrap();
    tokio::fs::write(
        path,
        changed_manifest(&original, operators)
            .canonical_bytes()
            .unwrap(),
    )
    .await
    .unwrap();
}
async fn assert_corrupt_rejected(root: &Path, probe: &Probe) {
    corrupt_saved_state(root).await;
    let (runner, _send, _) = connect(root, probe);
    let Err(calc_flow::CalcFlowError::Streaming(error)) = runner.start().await else {
        panic!("invalid MACD state must fail before open")
    };
    assert_eq!(
        error.category(),
        calc_flow::StreamingErrorCategory::CheckpointMismatch
    );
    assert_eq!(error.component_id(), Some("features"));
    assert!(error.message().contains("state.format"), "{error}");
    assert_eq!(probe.sources.load(Ordering::SeqCst), 1);
    assert_eq!(probe.sinks.load(Ordering::SeqCst), 1);
}
#[tokio::test]
async fn test_macd_normal_recovery_rejects_bad_count_after_valid_outer_checksums() {
    let dir = tempfile::tempdir().unwrap();
    let root = dir.path().join("managed");
    let probe = Probe::default();
    save_prefix(&root, &probe).await;
    assert_corrupt_rejected(&root, &probe).await;
}
#[tokio::test]
async fn test_macd_terminal_recovery_rejects_bad_count_after_valid_outer_checksums() {
    let dir = tempfile::tempdir().unwrap();
    let root = dir.path().join("managed");
    let probe = Probe::default();
    complete(&root, &probe).await;
    assert_corrupt_rejected(&root, &probe).await;
}
