mod queued_stop;

mod managed_capture;

use super::*;
use crate::runtime::streaming::progress::WatermarkPolicy;
use crate::{Epoch, OperatorStateSnapshot, SqlOperator, StateSegment};
use datafusion::arrow::{
    array::{Array, StringArray},
    datatypes::{DataType, Field, Schema, SchemaRef},
    ipc::writer::FileWriter,
};
use datafusion::execution::memory_pool::MemoryReservation;
use std::sync::{Condvar, Weak};

const NODE: &str = "managed_sql";
const QUERY: &str = "SELECT key, SUM(value) AS total FROM events GROUP BY key";
const DEADLINE: StdDuration = StdDuration::from_secs(5);

fn logical_schema() -> SchemaRef {
    Arc::new(Schema::new_with_metadata(
        vec![
            Field::new("key", DataType::Utf8, false),
            Field::new("value", DataType::Int64, false)
                .with_metadata([("units".into(), "ticks".into())].into()),
            Field::new("unused", DataType::Utf8, true),
        ],
        [("schema-source".into(), "managed-recovery".into())].into(),
    ))
}

fn input(source: &str) -> Batch {
    Batch::table(
        vec![
            RecordBatch::try_new(
                logical_schema(),
                vec![
                    Arc::new(StringArray::from(vec!["a", "a"])),
                    Arc::new(Int64Array::from(vec![2, 5])),
                    Arc::new(StringArray::from(vec![Some("payload"), None])),
                ],
            )
            .unwrap(),
        ],
        BatchMetadata::new(
            source,
            4,
            BTreeMap::from([(
                "fixture".into(),
                serde_json::json!({"immutable": true, "case": source}),
            )]),
        )
        .unwrap(),
    )
    .unwrap()
}

fn expected_normalized_metadata(source: &str) -> BatchMetadata {
    BatchMetadata::new(
        "events",
        0,
        BTreeMap::from([(
            "fixture".into(),
            serde_json::json!({"immutable": true, "case": source}),
        )]),
    )
    .unwrap()
}

async fn sql() -> SqlOperator {
    let operator = SqlOperator::new(NODE, QUERY, vec!["events".into()], vec![]).unwrap();
    let schema = operator
        .infer_schema(&BTreeMap::from([("events".into(), logical_schema())]))
        .await
        .unwrap();
    operator
        .with_ports(
            vec![
                Port::with_schema_ref("events", BatchKind::Table, true, Some(logical_schema()))
                    .unwrap(),
            ],
            Port::with_schema_ref("output", BatchKind::Table, true, Some(schema)).unwrap(),
        )
        .unwrap()
}

#[derive(Default)]
struct Probe {
    source_opens: AtomicUsize,
    source_reads: AtomicUsize,
    source_closes: AtomicUsize,
    start_completed: AtomicBool,
    tail: AtomicBool,
    sink_opens: AtomicUsize,
    sink_recovers: AtomicUsize,
    sink_pre_commits: AtomicUsize,
    sink_commits: AtomicUsize,
    sink_closes: AtomicUsize,
    writes: Mutex<Vec<Batch>>,
}

struct Source {
    binding: String,
    next: u64,
    terminal: bool,
    probe: Arc<Probe>,
}

#[async_trait]
impl StreamSource for Source {
    async fn open(&mut self, cursor: Option<Cursor>) -> Result<()> {
        self.probe.source_opens.fetch_add(1, Ordering::SeqCst);
        if let Some(cursor) = cursor {
            self.next = u64::from_be_bytes(cursor.order().try_into().unwrap());
        }
        Ok(())
    }
    async fn next(&mut self) -> Result<Option<SourceEvent>> {
        self.probe.source_reads.fetch_add(1, Ordering::SeqCst);
        if self.next == 0 {
            self.next = 1;
            return Ok(Some(SourceEvent::Data {
                batch: input(&self.binding),
                cursor: Cursor::unbound(self.next.to_be_bytes().to_vec(), JsonMap::new())?,
            }));
        }
        self.probe.tail.store(true, Ordering::SeqCst);
        if self.terminal {
            Ok(None)
        } else {
            std::future::pending().await
        }
    }
    async fn close(&mut self) -> Result<()> {
        self.probe.source_closes.fetch_add(1, Ordering::SeqCst);
        Ok(())
    }
    fn capabilities(&self) -> SourceCapabilities {
        SourceCapabilities {
            replayable: true,
            max_batch_rows: 2,
            max_batch_bytes: 1 << 20,
        }
    }
    fn native_watermark_capability(
        &self,
    ) -> crate::runtime::streaming::progress::NativeWatermarkCapability {
        crate::runtime::streaming::progress::NativeWatermarkCapability::NeverEmits
    }
}

struct Sink(Arc<Probe>);

#[async_trait]
impl TransactionalStreamSink for Sink {
    async fn open(&mut self) -> Result<()> {
        self.0.sink_opens.fetch_add(1, Ordering::SeqCst);
        Ok(())
    }
    async fn begin_epoch(&mut self, _epoch: Epoch) -> Result<()> {
        Ok(())
    }
    async fn write(&mut self, batch: &Batch) -> Result<()> {
        self.0.writes.lock().push(batch.clone());
        Ok(())
    }
    async fn pre_commit(&mut self, _epoch: Epoch) -> Result<JsonMap> {
        self.0.sink_pre_commits.fetch_add(1, Ordering::SeqCst);
        Ok(JsonMap::new())
    }
    async fn commit(&mut self, _epoch: Epoch, _state: &JsonMap) -> Result<()> {
        self.0.sink_commits.fetch_add(1, Ordering::SeqCst);
        Ok(())
    }
    async fn abort(&mut self, _epoch: Epoch, _state: Option<&JsonMap>) -> Result<()> {
        Ok(())
    }
    async fn recover(&mut self, manifest: &crate::CheckpointManifest) -> Result<()> {
        self.0.sink_recovers.fetch_add(1, Ordering::SeqCst);
        self.commit(
            manifest.epoch(),
            manifest.sinks()["sink"].pre_commit.as_ref().unwrap(),
        )
        .await
    }
    async fn close(&mut self) -> Result<()> {
        self.0.sink_closes.fetch_add(1, Ordering::SeqCst);
        Ok(())
    }
}

async fn spec(source: &str, terminal: bool, probe: &Arc<Probe>) -> ContinuousJobSpec {
    let plan = PipelineBuilder::new("sql-managed-current-recovery")
        .unwrap()
        .add_node(NODE, Box::new(sql().await))
        .unwrap()
        .compile_stream(
            &UdfRegistry::new().snapshot(),
            &StreamRequirements {
                delivery: BTreeMap::from([(
                    "output".into(),
                    crate::DeliveryGuarantee::ExactlyOnce,
                )]),
            },
        )
        .unwrap();
    ContinuousJobSpec {
        context: StreamJobContext::new(
            701,
            plan.fingerprint(),
            JsonMap::new(),
            None,
            CancellationToken::new(),
        ),
        plan,
        sources: vec![NamedSourceBinding {
            binding_id: "events".into(),
            binding: SourceBinding::new(
                Box::new(Source {
                    binding: source.into(),
                    next: 0,
                    terminal,
                    probe: probe.clone(),
                }),
                None,
                0,
            )
            .unwrap()
            .with_watermark_policy(WatermarkPolicy::Disabled { idle_timeout: None }),
        }],
        sinks: vec![NamedSinkBinding {
            output_id: "output".into(),
            sink_id: "sink".into(),
            binding: OrdinarySinkBinding::new_transactional(Box::new(Sink(probe.clone()))),
        }],
        edge_budget: EdgeBudget::new(16, 1 << 20).unwrap(),
        delivery_mode: M2DeliveryMode::ProcessLocalOrdered,
        static_inputs: crate::static_input::PreparedStaticInputs::default(),
    }
}

fn checkpoint(root: &Path) -> CheckpointRuntimeSpec {
    CheckpointRuntimeSpec::managed(
        ManagedCheckpointRuntime::new(root).unwrap(),
        StreamRuntimeConfig {
            checkpoint_interval: StdDuration::from_secs(3_600),
            checkpoint_timeout: DEADLINE,
            ..StreamRuntimeConfig::default()
        },
    )
    .unwrap()
}

async fn capture(root: &Path, source: &str, terminal: bool) -> crate::CheckpointManifest {
    let probe = Arc::new(Probe::default());
    let mut runner = ContinuousRunner::new();
    let job = runner
        .start_checkpointed(spec(source, terminal, &probe).await, checkpoint(root))
        .await
        .unwrap();
    let settled = tokio::time::timeout(DEADLINE, async {
        if terminal {
            return job.wait().await.state == ContinuousJobState::Completed;
        }
        while !probe.tail.load(Ordering::SeqCst) || probe.writes.lock().is_empty() {
            tokio::task::yield_now().await;
        }
        job.trigger_checkpoint().await.unwrap() == Epoch::INITIAL
    })
    .await;
    if !terminal || settled.is_err() {
        let _ = job.cancel().await;
    }
    drop(job);
    runner.shutdown().await.unwrap();
    assert_eq!(runner.registry_counts(), (0, 0));
    assert!(settled.unwrap());
    assert_eq!(probe.source_closes.load(Ordering::SeqCst), 1);
    assert_eq!(probe.sink_closes.load(Ordering::SeqCst), 1);
    let schema = sql().await.output_ports()[0].schema().unwrap().clone();
    assert_capture_output(&probe, source, schema);
    let bytes = tokio::fs::read(root.join("manifests/manifest-00000000000000000001.json"))
        .await
        .unwrap();
    let manifest = crate::CheckpointManifest::from_bytes(&bytes).unwrap();
    assert_eq!(
        manifest.operators()[NODE].inline_metadata["state_layout"],
        serde_json::json!(3)
    );
    assert_eq!(
        manifest.operators()[NODE].inline_metadata["state_accounting"],
        serde_json::json!(3)
    );
    assert_eq!(manifest.operators()[NODE].segments.len(), 4);
    assert_eq!(
        manifest.operators()[NODE].inline_metadata["rows"],
        serde_json::json!(2)
    );
    assert_eq!(manifest.sources()["events"].ended, terminal);
    manifest
}

fn assert_capture_output(probe: &Probe, source: &str, schema: SchemaRef) {
    let output = probe.writes.lock();
    assert_eq!(output.len(), 1);
    assert_eq!(output[0].metadata(), &expected_normalized_metadata(source));
    assert_eq!(output[0].table_payload().unwrap().schema(), &schema);
    let expected = RecordBatch::try_new(
        schema,
        vec![
            Arc::new(StringArray::from(vec!["a"])),
            Arc::new(Int64Array::from(vec![7])),
        ],
    )
    .unwrap();
    assert_eq!(output[0].table_payload().unwrap().batches(), &[expected]);
    drop(output);
}

#[derive(Default)]
struct ExecutionWitness {
    entered: bool,
    heartbeat: bool,
    callback_exited: bool,
}

#[derive(Default)]
struct ControlWitness {
    dropped_start: bool,
    released: bool,
}

#[derive(Default)]
struct LedgerWitness {
    paid: bool,
    metadata_ok: bool,
    values_ok: bool,
    rows: u64,
    bytes: u64,
    arrays: Vec<Weak<dyn Array>>,
    backing: Vec<Weak<MemoryReservation>>,
}

#[derive(Default)]
struct Witness {
    execution: ExecutionWitness,
    control: ControlWitness,
    ledger: LedgerWitness,
}

#[derive(Default)]
struct Gate {
    state: std::sync::Mutex<Witness>,
    changed: Condvar,
    async_entered: Notify,
}

impl Gate {
    fn release(&self) {
        self.state.lock().unwrap().control.released = true;
        self.changed.notify_all();
    }
    fn hold(&self, witness: Witness) {
        let mut state = self.state.lock().unwrap();
        let released = state.control.released;
        *state = witness;
        state.control.released = released;
        drop(state);
        self.changed.notify_all();
        self.async_entered.notify_one();
        let mut state = self.state.lock().unwrap();
        while !state.control.released {
            state = self.changed.wait(state).unwrap();
        }
        state.execution.callback_exited = true;
    }
    fn wait_for(&self, condition: impl Fn(&Witness) -> bool) -> bool {
        let deadline = StdInstant::now() + DEADLINE;
        let mut state = self.state.lock().unwrap();
        while !condition(&state) {
            let Some(remaining) = deadline.checked_duration_since(StdInstant::now()) else {
                return false;
            };
            let (next, timeout) = self.changed.wait_timeout(state, remaining).unwrap();
            state = next;
            if timeout.timed_out() && !condition(&state) {
                return false;
            }
        }
        true
    }
}

struct Controller {
    gate: Arc<Gate>,
    thread: Option<std::thread::JoinHandle<bool>>,
}

impl Controller {
    fn join(mut self) -> bool {
        self.thread.take().unwrap().join().unwrap()
    }
}

impl Drop for Controller {
    fn drop(&mut self) {
        self.gate.release();
        if let Some(thread) = self.thread.take() {
            let _ = thread.join();
        }
    }
}

fn candidate_witness(
    source: &str,
    metadata: &BatchMetadata,
    records: &[RecordBatch],
    charge: (u64, u64),
    reserved: usize,
    backing: Vec<Weak<MemoryReservation>>,
) -> Witness {
    let expected_record = RecordBatch::try_new(
        Arc::new(Schema::new(vec![
            Field::new("key_0", DataType::Utf8, false),
            Field::new("state_0_0", DataType::Int64, true),
        ])),
        vec![
            Arc::new(StringArray::from(vec!["a"])),
            Arc::new(Int64Array::from(vec![7])),
        ],
    )
    .unwrap();
    Witness {
        execution: ExecutionWitness {
            entered: true,
            ..ExecutionWitness::default()
        },
        control: ControlWitness::default(),
        ledger: LedgerWitness {
            paid: reserved > 0,
            rows: charge.0,
            bytes: charge.1,
            metadata_ok: metadata == &expected_normalized_metadata(source),
            values_ok: records == [expected_record],
            arrays: records
                .iter()
                .flat_map(|record| record.columns().iter().map(Arc::downgrade))
                .collect(),
            backing,
        },
    }
}

fn assert_released(gate: &Gate, expected_bytes: u64) {
    let state = gate.state.lock().unwrap();
    assert!(state.execution.entered && state.execution.callback_exited && state.ledger.paid);
    assert_eq!(state.ledger.rows, 2);
    assert_eq!(state.ledger.bytes, expected_bytes);
    assert!(state.ledger.metadata_ok && state.ledger.values_ok);
    assert_eq!(state.ledger.arrays.len(), 2);
    assert!(!state.ledger.backing.is_empty());
    assert!(
        state
            .ledger
            .arrays
            .iter()
            .all(|weak| weak.upgrade().is_none())
    );
    assert!(
        state
            .ledger
            .backing
            .iter()
            .all(|weak| weak.upgrade().is_none())
    );
}

fn recovery_refund_probe(
    operator: &mut SqlOperator,
    saved: &OperatorStateSnapshot,
) -> MemoryReservation {
    let fee = operator.reserve_recovery_envelope(saved).unwrap();
    let refund_probe = fee.new_empty();
    drop(fee);
    refund_probe
}

fn callback_backing_alive(state: &Witness) -> bool {
    !state.execution.callback_exited
        && state.ledger.paid
        && !state.ledger.backing.is_empty()
        && state
            .ledger
            .backing
            .iter()
            .all(|weak| weak.upgrade().is_some())
        && state
            .ledger
            .arrays
            .iter()
            .all(|weak| weak.upgrade().is_some())
}

fn callback_arrays_alive(state: &Witness) -> bool {
    state.ledger.paid
        && !state.execution.callback_exited
        && !state.ledger.arrays.is_empty()
        && !state.ledger.backing.is_empty()
        && state
            .ledger
            .arrays
            .iter()
            .all(|weak| weak.upgrade().is_some())
        && state
            .ledger
            .backing
            .iter()
            .all(|weak| weak.upgrade().is_some())
}

fn start_untouched(probe: &Probe) -> bool {
    probe.source_opens.load(Ordering::SeqCst) == 0
        && probe.source_reads.load(Ordering::SeqCst) == 0
        && probe.sink_opens.load(Ordering::SeqCst) == 0
        && !probe.start_completed.load(Ordering::SeqCst)
}

fn terminal_unopened(probe: &Probe) -> bool {
    probe.source_opens.load(Ordering::SeqCst) == 0
        && probe.source_reads.load(Ordering::SeqCst) == 0
        && probe.sink_opens.load(Ordering::SeqCst) == 0
        && probe.sink_recovers.load(Ordering::SeqCst) == 0
        && probe.sink_commits.load(Ordering::SeqCst) == 0
}

fn cancelled_start_controller(
    gate: &Arc<Gate>,
    core: Arc<JobCore>,
    probe: Arc<Probe>,
    start: super::super::StartObserver,
) -> Controller {
    let control_core = core;
    let control_probe = probe;
    let control_gate = gate.clone();
    Controller {
        gate: gate.clone(),
        thread: Some(std::thread::spawn(move || {
            let entered = control_gate.wait_for(|state| state.execution.entered);
            drop(start);
            control_gate.state.lock().unwrap().control.dropped_start = true;
            control_gate.changed.notify_all();
            let registered = control_core
                .runtime_status
                .lock()
                .tasks
                .snapshot()
                .values()
                .any(|status| status.task_name == format!("operator:{NODE}"));
            let pending = control_core.state.lock().outcome.is_none();
            let candidate_alive = {
                let state = control_gate.state.lock().unwrap();
                callback_backing_alive(&state)
            };
            let untouched = start_untouched(&control_probe);
            control_gate.release();
            entered && registered && pending && candidate_alive && untouched
        })),
    }
}

#[tokio::test(flavor = "current_thread")]
async fn compact_managed_native3_restore_yields_before_entry_ack() {
    let directory = tempfile::tempdir().unwrap();
    let source = "managed-heartbeat-source";
    let captured = capture(directory.path(), source, false).await;
    let expected_bytes = captured.operators()[NODE].inline_metadata["bytes"]
        .as_u64()
        .unwrap();
    let gate = Arc::new(Gate::default());
    let operator = sql().await;
    let guard = operator.on_prepared_restore_for_test(&expected_normalized_metadata(source), {
        let gate = gate.clone();
        move |candidate| {
            let witness = candidate_witness(
                source,
                &candidate.metadata,
                &candidate.records,
                (candidate.rows, candidate.bytes),
                candidate.reserved,
                candidate.backing,
            );
            drop(candidate.records);
            gate.hold(witness);
            Ok(())
        }
    });
    let probe = Arc::new(Probe::default());
    let control_gate = gate.clone();
    let control_probe = probe.clone();
    let controller = Controller {
        gate: gate.clone(),
        thread: Some(std::thread::spawn(move || {
            let entered = control_gate.wait_for(|state| state.execution.entered);
            let heartbeat = control_gate.wait_for(|state| state.execution.heartbeat);
            let untouched = control_probe.source_opens.load(Ordering::SeqCst) == 0
                && control_probe.source_reads.load(Ordering::SeqCst) == 0
                && control_probe.sink_opens.load(Ordering::SeqCst) == 0
                && !control_probe.start_completed.load(Ordering::SeqCst);
            control_gate.release();
            entered && heartbeat && untouched
        })),
    };
    let beat_gate = gate.clone();
    let heartbeat = tokio::spawn(async move {
        beat_gate.async_entered.notified().await;
        beat_gate.state.lock().unwrap().execution.heartbeat = true;
        beat_gate.changed.notify_all();
    });
    let mut runner = ContinuousRunner::new();
    let start = runner.start_checkpointed(
        spec(source, false, &probe).await,
        checkpoint(directory.path()),
    );
    let core = start.core.as_ref().unwrap().clone();
    let launched = start.await;
    probe
        .start_completed
        .store(launched.is_ok(), Ordering::SeqCst);
    if let Ok(job) = launched {
        let _ = job.cancel().await;
    }
    runner.shutdown().await.unwrap();
    heartbeat.abort();
    let _ = heartbeat.await;
    let progressed = controller.join();
    assert_eq!(runner.registry_counts(), (0, 0));
    assert!(core.runtime_status.lock().tasks.snapshot().is_empty());
    assert_released(&gate, expected_bytes);
    assert_eq!(guard.installs(), 1);
    assert!(
        progressed,
        "supported paid SQL recovery blocked the current-thread heartbeat before entry ack"
    );
}

#[tokio::test(flavor = "current_thread")]
async fn compact_managed_native3_cancel_keeps_candidate_until_join_without_install() {
    let directory = tempfile::tempdir().unwrap();
    let source = "managed-cancel-source";
    let captured = capture(directory.path(), source, false).await;
    let expected_bytes = captured.operators()[NODE].inline_metadata["bytes"]
        .as_u64()
        .unwrap();
    let gate = Arc::new(Gate::default());
    let operator = sql().await;
    let guard = operator.on_prepared_restore_for_test(&expected_normalized_metadata(source), {
        let gate = gate.clone();
        move |candidate| {
            let witness = candidate_witness(
                source,
                &candidate.metadata,
                &candidate.records,
                (candidate.rows, candidate.bytes),
                candidate.reserved,
                candidate.backing,
            );
            drop(candidate.records);
            gate.hold(witness);
            Ok(())
        }
    });
    let probe = Arc::new(Probe::default());
    let mut runner = ContinuousRunner::new();
    let start = runner.start_checkpointed(
        spec(source, false, &probe).await,
        checkpoint(directory.path()),
    );
    let core = start.core.as_ref().unwrap().clone();
    let controller = cancelled_start_controller(&gate, core.clone(), probe.clone(), start);
    let joined = tokio::time::timeout(DEADLINE + DEADLINE, async {
        gate.async_entered.notified().await;
        runner.shutdown().await
    })
    .await;
    if joined.is_err() {
        gate.release();
        runner.shutdown().await.unwrap();
    }
    let held_until_release = controller.join();
    let failure = super::super::StartObserver::observe(core.clone())
        .await
        .unwrap_err();
    assert!(joined.unwrap().is_ok());
    assert!(held_until_release);
    assert_eq!(runner.registry_counts(), (0, 0));
    assert!(core.runtime_status.lock().tasks.snapshot().is_empty());
    assert!(matches!(
        failure.primary.error,
        CalcFlowError::Cancelled { .. }
    ));
    assert_eq!(probe.source_opens.load(Ordering::SeqCst), 0);
    assert_eq!(probe.source_reads.load(Ordering::SeqCst), 0);
    assert_eq!(probe.sink_opens.load(Ordering::SeqCst), 0);
    assert!(gate.state.lock().unwrap().control.dropped_start);
    assert_released(&gate, expected_bytes);
    assert_eq!(
        guard.installs(),
        0,
        "late cancellation installed the paid SQL candidate"
    );
}

async fn snapshot(root: &Path, manifest: &crate::CheckpointManifest) -> OperatorStateSnapshot {
    let backend = LocalStateBackend::new(root.join("state")).await.unwrap();
    let key =
        StateLineageKey::new(manifest.pipeline_name(), manifest.pipeline_fingerprint()).unwrap();
    let lineage = backend.open_lineage(&key).await.unwrap();
    let mut segments = BTreeMap::new();
    for handle in &manifest.operators()[NODE].segments {
        let bytes = lineage.load_segment(handle).await.unwrap();
        assert_eq!(bytes.len() as u64, handle.byte_len());
        assert_eq!(hex::encode(Sha256::digest(&bytes)), handle.sha256());
        segments.insert(handle.segment_id().into(), StateSegment::new(bytes));
    }
    let raw = OperatorStateSnapshot {
        inline_metadata: manifest.operators()[NODE].inline_metadata.clone(),
        segments,
    };
    let specification = spec("terminal-reader", true, &Arc::new(Probe::default())).await;
    let parts = specification
        .plan
        .into_runtime_parts(specification.edge_budget)
        .unwrap();
    let node = parts
        .nodes
        .iter()
        .find(|node| node.operator_id.as_str() == NODE)
        .unwrap();
    assert!(!node.operator.requires_output_frontier_state());
    let mut decoded = node
        .checkpoint_capability
        .decode_snapshot(NODE, raw.clone())
        .unwrap();
    assert!(
        decoded
            .inline_metadata
            .remove(crate::pipeline::OUTPUT_FRONTIER_METADATA_KEY_V1)
            .is_none()
    );
    let encoded = node
        .checkpoint_capability
        .encode_snapshot(NODE, decoded.clone())
        .unwrap();
    assert_eq!(encoded.inline_metadata, raw.inline_metadata);
    assert_eq!(decoded.segments.len(), raw.segments.len());
    for (id, segment) in &decoded.segments {
        assert_eq!(segment.bytes(), raw.segments[id].bytes());
    }
    decoded
}

#[derive(Clone, Copy)]
enum Damage {
    Codec,
    FullSchema,
}

async fn damaged_manifest(
    root: &Path,
    original: &crate::CheckpointManifest,
    damage: Damage,
) -> crate::CheckpointManifest {
    let (segment_id, bytes) = match damage {
        Damage::Codec => ("group-state", b"not-an-arrow-native-state-file".to_vec()),
        Damage::FullSchema => {
            let original = logical_schema();
            let mut fields: Vec<_> = original
                .fields()
                .iter()
                .map(|field| field.as_ref().clone())
                .collect();
            fields[2] = fields[2]
                .clone()
                .with_metadata([("altered-unused".into(), "true".into())].into());
            let schema = Schema::new_with_metadata(fields, original.metadata().clone());
            let mut bytes = vec![];
            FileWriter::try_new(&mut bytes, &schema)
                .unwrap()
                .finish()
                .unwrap();
            ("logical-schema", bytes)
        }
    };
    let mut operators = original.operators().clone();
    let entry = operators.get_mut(NODE).unwrap();
    let control_handle = entry
        .segments
        .iter()
        .find(|handle| handle.segment_id() == "control")
        .unwrap();
    let control_bytes = tokio::fs::read(root.join("state").join(control_handle.relative_path()))
        .await
        .unwrap();
    assert_eq!(
        hex::encode(Sha256::digest(&control_bytes)),
        control_handle.sha256()
    );
    let mut control: serde_json::Value = serde_json::from_slice(&control_bytes).unwrap();
    assert_eq!(control["state_layout"], serde_json::json!(3));
    assert_eq!(control["state_accounting"], serde_json::json!(3));
    let digest = rewrite_state_segment(root, &mut entry.segments, segment_id, &bytes).await;
    match damage {
        Damage::Codec => control["segments"]["group_state"] = serde_json::json!(digest),
        Damage::FullSchema => {
            control["segments"]["logical_schema"] = serde_json::json!(digest);
            control["identity"]["logical_schema_sha256"] = serde_json::json!(digest);
        }
    }
    let control_bytes = serde_json::to_vec(&control).unwrap();
    let control_digest =
        rewrite_state_segment(root, &mut entry.segments, "control", &control_bytes).await;
    entry
        .inline_metadata
        .insert("control_sha256".into(), serde_json::json!(control_digest));
    let damaged = crate::CheckpointManifest::new(CheckpointManifestFields {
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
    .unwrap();
    let encoded = damaged.canonical_bytes().unwrap();
    assert_eq!(
        crate::CheckpointManifest::from_bytes(&encoded).unwrap(),
        damaged
    );
    tokio::fs::write(
        root.join("manifests/manifest-00000000000000000001.json"),
        encoded,
    )
    .await
    .unwrap();
    damaged
}

async fn rewrite_state_segment(
    root: &Path,
    handles: &mut [StateHandle],
    segment_id: &str,
    bytes: &[u8],
) -> String {
    let handle = handles
        .iter_mut()
        .find(|handle| handle.segment_id() == segment_id)
        .unwrap();
    let digest = hex::encode(Sha256::digest(bytes));
    tokio::fs::write(root.join("state").join(handle.relative_path()), bytes)
        .await
        .unwrap();
    *handle = StateHandle::new(
        handle.operator_id(),
        handle.epoch(),
        handle.segment_id(),
        handle.relative_path(),
        bytes.len() as u64,
        &digest,
    )
    .unwrap();
    digest
}

async fn terminal_attempt(root: &Path, source: &str) -> (bool, Arc<Probe>) {
    let probe = Arc::new(Probe::default());
    let mut runner = ContinuousRunner::new();
    let result = runner
        .start_checkpointed(spec(source, true, &probe).await, checkpoint(root))
        .await;
    let failed = match result {
        Ok(job) => {
            let _ = job.wait().await;
            drop(job);
            false
        }
        Err(_) => true,
    };
    runner.shutdown().await.unwrap();
    assert_eq!(runner.registry_counts(), (0, 0));
    assert_eq!(probe.source_opens.load(Ordering::SeqCst), 0);
    assert_eq!(probe.source_reads.load(Ordering::SeqCst), 0);
    (failed, probe)
}

fn assert_terminal_damage(damage: Damage, direct: &CalcFlowError) {
    if matches!(damage, Damage::Codec) {
        assert!(
            matches!(direct, CalcFlowError::Format { message } if message == "SQL state segment has no Arrow IPC file magic")
        );
    }
    if matches!(damage, Damage::FullSchema) {
        assert!(
            matches!(direct, CalcFlowError::Format { message } if message.contains("declared input"))
        );
    }
}

#[tokio::test(flavor = "current_thread")]
async fn compact_terminal_native3_sql_validates_native_state_before_sink_recovery() {
    let mut refusals = vec![];
    for damage in [Damage::Codec, Damage::FullSchema] {
        let directory = tempfile::tempdir().unwrap();
        let source = match damage {
            Damage::Codec => "terminal-codec-source",
            Damage::FullSchema => "terminal-schema-source",
        };
        let valid = capture(directory.path(), source, true).await;
        let valid_snapshot = snapshot(directory.path(), &valid).await;
        StreamOperator::restore(&mut sql().await, &valid_snapshot).unwrap();
        let invalid = damaged_manifest(directory.path(), &valid, damage).await;
        let invalid_snapshot = snapshot(directory.path(), &invalid).await;
        let direct = StreamOperator::restore(&mut sql().await, &invalid_snapshot).unwrap_err();
        assert_terminal_damage(damage, &direct);
        let (failed, probe) = terminal_attempt(directory.path(), source).await;
        refusals.push(
            failed
                && probe.sink_opens.load(Ordering::SeqCst) == 0
                && probe.sink_recovers.load(Ordering::SeqCst) == 0
                && probe.sink_commits.load(Ordering::SeqCst) == 0,
        );
    }
    assert_eq!(
        refusals,
        [true, true],
        "strict terminal native3 SQL skipped native validation before sink recovery"
    );
}

#[tokio::test(flavor = "current_thread")]
async fn compact_valid_terminal_native3_sql_recovers_transactional_sink() {
    let directory = tempfile::tempdir().unwrap();
    let source = "terminal-valid-source";
    let manifest = capture(directory.path(), source, true).await;
    StreamOperator::restore(
        &mut sql().await,
        &snapshot(directory.path(), &manifest).await,
    )
    .unwrap();
    let (failed, probe) = terminal_attempt(directory.path(), source).await;
    assert!(!failed);
    assert_eq!(probe.sink_opens.load(Ordering::SeqCst), 1);
    assert_eq!(probe.sink_recovers.load(Ordering::SeqCst), 1);
    assert_eq!(probe.sink_commits.load(Ordering::SeqCst), 1);
    assert_eq!(probe.sink_closes.load(Ordering::SeqCst), 1);
}

#[tokio::test(flavor = "current_thread")]
async fn compact_terminal_sql_drop_retains_paid_candidate_until_native_join() {
    let directory = tempfile::tempdir().unwrap();
    let source = "terminal-drop-held-source";
    let manifest = capture(directory.path(), source, true).await;
    let expected_bytes = manifest.operators()[NODE].inline_metadata["bytes"]
        .as_u64()
        .unwrap();
    let gate = Arc::new(Gate::default());
    let operator = sql().await;
    let registration =
        operator.on_prepared_restore_for_test(&expected_normalized_metadata(source), {
            let gate = gate.clone();
            move |candidate| {
                let witness = candidate_witness(
                    source,
                    &candidate.metadata,
                    &candidate.records,
                    (candidate.rows, candidate.bytes),
                    candidate.reserved,
                    candidate.backing,
                );
                drop(candidate.records);
                gate.hold(witness);
                Ok(())
            }
        });
    let probe = Arc::new(Probe::default());
    let mut runner = ContinuousRunner::new();
    let start = runner.start_checkpointed(
        spec(source, true, &probe).await,
        checkpoint(directory.path()),
    );
    let core = start.core.as_ref().unwrap().clone();
    let control_core = core.clone();
    let control_probe = probe.clone();
    let control_gate = gate.clone();
    let controller = Controller {
        gate: gate.clone(),
        thread: Some(std::thread::spawn(move || {
            let entered = control_gate.wait_for(|state| state.execution.entered);
            drop(start);
            let pending = control_core.state.lock().outcome.is_none();
            let loaned = control_core.sql_recovery.live_counts() == (1, 1);
            let no_supervisor = control_core
                .runtime_status
                .lock()
                .tasks
                .snapshot()
                .is_empty();
            let alive = {
                let state = control_gate.state.lock().unwrap();
                callback_arrays_alive(&state)
            };
            let unopened = terminal_unopened(&control_probe);
            control_gate.release();
            entered && pending && loaned && no_supervisor && alive && unopened
        })),
    };
    let observed = super::super::StartObserver::observe(core.clone()).await;
    if let Ok(job) = &observed {
        let _ = job.cancel().await;
    }
    drop(observed);
    runner.shutdown().await.unwrap();
    let retained_until_join = controller.join();
    assert_eq!(runner.registry_counts(), (0, 0));
    assert!(core.runtime_status.lock().tasks.snapshot().is_empty());
    assert_eq!(core.sql_recovery.live_counts(), (0, 0));
    assert_eq!(registration.installs(), 0);
    assert_eq!(probe.source_opens.load(Ordering::SeqCst), 0);
    assert_eq!(probe.source_reads.load(Ordering::SeqCst), 0);
    assert_eq!(probe.sink_recovers.load(Ordering::SeqCst), 0);
    assert_eq!(probe.sink_commits.load(Ordering::SeqCst), 0);
    assert_released(&gate, expected_bytes);
    assert!(retained_until_join);
}

#[tokio::test(flavor = "current_thread")]
async fn compact_sql_candidate_panic_keeps_original_registered_task_identity() {
    let directory = tempfile::tempdir().unwrap();
    let source = "managed-kernel-panic-source";
    let manifest = capture(directory.path(), source, false).await;
    let expected_bytes = manifest.operators()[NODE].inline_metadata["bytes"]
        .as_u64()
        .unwrap();
    let gate = Arc::new(Gate::default());
    let actual_id = Arc::new(AtomicU64::new(u64::MAX));
    let core_slot = Arc::new(Mutex::new(None::<Arc<JobCore>>));
    let operator = sql().await;
    let registration =
        operator.on_prepared_restore_for_test(&expected_normalized_metadata(source), {
            let core_slot = core_slot.clone();
            let actual_id = actual_id.clone();
            let gate = gate.clone();
            move |candidate| {
                let core = core_slot.lock().as_ref().unwrap().clone();
                let task_id = core
                    .runtime_status
                    .lock()
                    .tasks
                    .snapshot()
                    .into_iter()
                    .find(|(_, status)| status.task_name == format!("operator:{NODE}"))
                    .unwrap()
                    .0;
                actual_id.store(task_id.as_u64(), Ordering::SeqCst);
                let witness = candidate_witness(
                    source,
                    &candidate.metadata,
                    &candidate.records,
                    (candidate.rows, candidate.bytes),
                    candidate.reserved,
                    candidate.backing,
                );
                drop(candidate.records);
                gate.release();
                gate.hold(witness);
                panic!("SQL candidate kernel panic");
            }
        });
    let probe = Arc::new(Probe::default());
    let mut runner = ContinuousRunner::new();
    let start = runner.start_checkpointed(
        spec(source, false, &probe).await,
        checkpoint(directory.path()),
    );
    let core = start.core.as_ref().unwrap().clone();
    *core_slot.lock() = Some(core.clone());
    let result = start.await;
    let failure = match result {
        Ok(job) => {
            let _ = job.cancel().await;
            None
        }
        Err(failure) => Some(failure),
    };
    runner.shutdown().await.unwrap();
    core_slot.lock().take();
    assert_eq!(runner.registry_counts(), (0, 0));
    assert_eq!(core.sql_recovery.live_counts(), (0, 0));
    assert_released(&gate, expected_bytes);
    assert_eq!(registration.installs(), 0);
    assert_eq!(probe.source_opens.load(Ordering::SeqCst), 0);
    assert_eq!(probe.source_reads.load(Ordering::SeqCst), 0);
    assert_eq!(probe.sink_recovers.load(Ordering::SeqCst), 0);
    assert_eq!(
        core.sql_recovery.panic_identity(),
        Some(actual_id.load(Ordering::SeqCst))
    );
    let primary = failure.unwrap().primary;
    assert_eq!(primary.origin, RuntimeFailureOrigin::Preflight);
    assert!(matches!(
        &primary.error,
        CalcFlowError::Internal { message } if message == "managed checkpoint recovery failed"
    ));
}

async fn captured_sql_recovery(
    directory: tempfile::TempDir,
    source: &str,
) -> (
    tempfile::TempDir,
    crate::CheckpointManifest,
    OperatorStateSnapshot,
    u64,
) {
    let manifest = capture(directory.path(), source, false).await;
    let saved = snapshot(directory.path(), &manifest).await;
    let bytes = manifest.operators()[NODE].inline_metadata["bytes"]
        .as_u64()
        .unwrap();
    (directory, manifest, saved, bytes)
}

fn observe_predecessor_settled(gate: &Gate, settled: &AtomicBool) {
    let state = gate.state.lock().unwrap();
    settled.store(
        state.execution.callback_exited
            && state
                .ledger
                .arrays
                .iter()
                .all(|weak| weak.upgrade().is_none())
            && state
                .ledger
                .backing
                .iter()
                .all(|weak| weak.upgrade().is_none()),
        Ordering::SeqCst,
    );
    drop(state);
}

fn assert_aborted_driver_panic(runner: &ContinuousRunner, core: &JobCore, actual_id: &AtomicU64) {
    let diagnostics = runner.diagnostics();
    let cleanup = diagnostics
        .records
        .iter()
        .find(|record| record.launch_id == core.launch_id)
        .unwrap();
    let panics = cleanup
        .cleanup_failures
        .iter()
        .filter(|failure| {
            matches!(failure.error,
        CalcFlowError::TaskPanicked { task_id, .. } if task_id == actual_id.load(Ordering::SeqCst))
        })
        .count();
    assert_eq!(panics, 1);
    assert!(!cleanup.failures_truncated);
}

fn aborted_driver_controller(control_gate: Arc<Gate>, control_core: Arc<JobCore>) -> Controller {
    Controller {
        gate: control_gate.clone(),
        thread: Some(std::thread::spawn(move || {
            let entered = control_gate.wait_for(|state| state.execution.entered);
            let returned = control_gate.wait_for(|state| state.control.dropped_start);
            let home_owns_join = control_core.sql_recovery.live_counts().0 == 1
                && control_core.sql_recovery.returned_loans() == 1;
            let pending = control_core.state.lock().outcome.is_none();
            let alive = {
                let state = control_gate.state.lock().unwrap();
                callback_backing_alive(&state)
            };
            control_gate.release();
            entered && returned && home_owns_join && pending && alive
        })),
    }
}

#[tokio::test(flavor = "current_thread")]
async fn compact_sql_aborted_driver_returns_live_join_to_job_home() {
    let directory = tempfile::tempdir().unwrap();
    let source = "managed-aborted-driver-source";
    let manifest = capture(directory.path(), source, false).await;
    let expected_bytes = manifest.operators()[NODE].inline_metadata["bytes"]
        .as_u64()
        .unwrap();
    let gate = Arc::new(Gate::default());
    let operator = sql().await;
    let registration =
        operator.on_prepared_restore_for_test(&expected_normalized_metadata(source), {
            let gate = gate.clone();
            move |candidate| {
                let witness = candidate_witness(
                    source,
                    &candidate.metadata,
                    &candidate.records,
                    (candidate.rows, candidate.bytes),
                    candidate.reserved,
                    candidate.backing,
                );
                drop(candidate.records);
                gate.hold(witness);
                panic!("late SQL panic after aborted driver");
            }
        });
    let probe = Arc::new(Probe::default());
    let mut runner = ContinuousRunner::new();
    let start = runner.start_checkpointed(
        spec(source, false, &probe).await,
        checkpoint(directory.path()),
    );
    let core = start.core.as_ref().unwrap().clone();
    let controller = aborted_driver_controller(gate.clone(), core.clone());
    let actual_id = Arc::new(AtomicU64::new(u64::MAX));
    let abort_id = actual_id.clone();
    let abort_core = core.clone();
    let abort_gate = gate.clone();
    let aborter = tokio::spawn(async move {
        abort_gate.async_entered.notified().await;
        let task_id = abort_core
            .runtime_status
            .lock()
            .tasks
            .snapshot()
            .into_iter()
            .find(|(_, status)| status.task_name == format!("operator:{NODE}"))
            .unwrap()
            .0;
        abort_id.store(task_id.as_u64(), Ordering::SeqCst);
        abort_core.prepare_driver_report(super::super::DriverReport::aborted(
            abort_core.launch_id,
            "prepared before native settlement",
        ));
        abort_core.driver_abort.lock().as_ref().unwrap().abort();
        let returned = tokio::time::timeout(DEADLINE, async {
            while abort_core.sql_recovery.returned_loans() != 1 {
                tokio::task::yield_now().await;
            }
        })
        .await
        .is_ok();
        abort_gate.state.lock().unwrap().control.dropped_start = returned;
        abort_gate.changed.notify_all();
    });
    let result = start.await;
    if let Ok(job) = &result {
        let _ = job.cancel().await;
    }
    let failure = result.err();
    runner.shutdown().await.unwrap();
    let _ = aborter.await;
    let retained_until_join = controller.join();
    assert_eq!(runner.registry_counts(), (0, 0));
    assert_eq!(core.sql_recovery.live_counts(), (0, 0));
    assert!(core.runtime_status.lock().tasks.snapshot().is_empty());
    assert_eq!(registration.installs(), 0);
    assert_eq!(probe.source_opens.load(Ordering::SeqCst), 0);
    assert_eq!(probe.source_reads.load(Ordering::SeqCst), 0);
    assert_eq!(probe.sink_recovers.load(Ordering::SeqCst), 0);
    assert_released(&gate, expected_bytes);
    assert!(retained_until_join);
    assert!(matches!(failure.unwrap().primary.error,
        CalcFlowError::Internal { ref message } if message.contains("prepared before native settlement")));
    assert_aborted_driver_panic(&runner, &core, &actual_id);
}

#[tokio::test(flavor = "current_thread")]
async fn compact_sql_request_fee_refusal_returns_actual_operator_before_launch() {
    use crate::runtime::streaming::sql_recovery_work::{
        JobSqlRecoveryOwner, SqlRecoveryContext, SqlRestoreIdentity, SqlRestoreRequest,
    };
    let directory = tempfile::tempdir().unwrap();
    let source = "managed-request-fee-source";
    let manifest = capture(directory.path(), source, false).await;
    let saved = snapshot(directory.path(), &manifest).await;
    let mut operator = sql().await;
    let pressure = operator.reserve_recovery_envelope(&saved).unwrap();
    pressure.try_grow((1 << 30) - pressure.size()).unwrap();
    assert!(operator.stream_runtime_initialized());
    let owner = JobSqlRecoveryOwner::new();
    owner.configure(1).unwrap();
    let context = StreamJobContext::new(
        701,
        "request-fee-control",
        JsonMap::new(),
        None,
        CancellationToken::new(),
    );
    let submitted = owner.submit(SqlRestoreRequest {
        operator,
        snapshot: saved.clone(),
        identity: SqlRestoreIdentity {
            node_id: NODE.into(),
            node_order: 0,
            task_id: None,
        },
        context: SqlRecoveryContext::from(&context),
        launch_cancel: CancellationToken::new(),
    });
    let refused = submitted.result.err();
    let mut original = submitted.operator.unwrap();
    assert!(
        matches!(refused, Some(CalcFlowError::DataFusion { node_id: Some(ref id), .. }) if id == NODE)
    );
    assert!(original.stream_runtime_initialized());
    assert_eq!(owner.live_counts(), (0, 0));
    assert_eq!(pressure.size(), 1 << 30);
    drop(pressure);
    let submitted = owner.submit(SqlRestoreRequest {
        operator: original,
        snapshot: saved,
        identity: SqlRestoreIdentity {
            node_id: NODE.into(),
            node_order: 0,
            task_id: None,
        },
        context: SqlRecoveryContext::from(&context),
        launch_cancel: CancellationToken::new(),
    });
    let mut completion = submitted.result.unwrap().join().await.unwrap();
    let current = completion.check_current();
    original = completion.operator;
    let prepared = current
        .and(completion.prepared)
        .unwrap()
        .into_restore()
        .unwrap();
    original.install_restore(prepared);
    let restored = StreamOperator::checkpoint(&mut original, Epoch::INITIAL).unwrap();
    assert_eq!(restored.inline_metadata["rows"], serde_json::json!(2));
    assert_eq!(
        restored.inline_metadata["bytes"],
        manifest.operators()[NODE].inline_metadata["bytes"]
    );
    assert_eq!(
        restored.inline_metadata["state_layout"],
        serde_json::json!(3)
    );
    assert_eq!(owner.live_counts(), (0, 0));
    assert!(owner.drain().await.is_empty());
}

#[tokio::test(flavor = "current_thread")]
async fn compact_sql_dropped_queued_request_does_not_block_next_join() {
    use crate::runtime::streaming::sql_recovery_work::{
        JobSqlRecoveryOwner, SqlRecoveryContext, SqlRestoreIdentity, SqlRestoreRequest,
    };
    let directory = tempfile::tempdir().unwrap();
    let source = "queued-request-drop-source";
    let manifest = capture(directory.path(), source, false).await;
    let saved = snapshot(directory.path(), &manifest).await;
    let expected_bytes = manifest.operators()[NODE].inline_metadata["bytes"]
        .as_u64()
        .unwrap();
    let mut first = sql().await;
    let fee = first.reserve_recovery_envelope(&saved).unwrap();
    let refund_probe = fee.new_empty();
    drop(fee);
    let gate = Arc::new(Gate::default());
    gate.release();
    let second = sql().await;
    let registration =
        second.on_prepared_restore_for_test(&expected_normalized_metadata(source), {
            let gate = gate.clone();
            move |candidate| {
                let witness = candidate_witness(
                    source,
                    &candidate.metadata,
                    &candidate.records,
                    (candidate.rows, candidate.bytes),
                    candidate.reserved,
                    candidate.backing,
                );
                drop(candidate.records);
                gate.hold(witness);
                Ok(())
            }
        });
    let owner = JobSqlRecoveryOwner::new();
    owner.configure(2).unwrap();
    let context = StreamJobContext::new(
        702,
        "queued-request-drop-control",
        JsonMap::new(),
        None,
        CancellationToken::new(),
    );
    let first = owner
        .submit(SqlRestoreRequest {
            operator: first,
            snapshot: saved.clone(),
            identity: SqlRestoreIdentity {
                node_id: NODE.into(),
                node_order: 0,
                task_id: None,
            },
            context: SqlRecoveryContext::from(&context),
            launch_cancel: CancellationToken::new(),
        })
        .result
        .unwrap();
    let second = owner
        .submit(SqlRestoreRequest {
            operator: second,
            snapshot: saved,
            identity: SqlRestoreIdentity {
                node_id: NODE.into(),
                node_order: 1,
                task_id: None,
            },
            context: SqlRecoveryContext::from(&context),
            launch_cancel: CancellationToken::new(),
        })
        .result
        .unwrap();
    let queued_fee_paid = refund_probe.try_grow(1 << 30).is_err();
    refund_probe.free();
    drop(first);
    let result = tokio::time::timeout(DEADLINE, second.join()).await;
    let completed = matches!(&result, Ok(Ok(completion)) if completion.prepared.is_ok());
    drop(result);
    let queued_fee_refunded = refund_probe.try_grow(1 << 30).is_ok();
    refund_probe.free();
    let retired_before_cleanup = owner.live_counts() == (0, 0);
    let cleanup = owner.drain().await;
    gate.release();
    assert_eq!(owner.live_counts(), (0, 0));
    assert!(cleanup.is_empty());
    assert!(queued_fee_paid);
    assert!(
        completed,
        "later SQL request did not finish after queued predecessor Drop"
    );
    assert!(queued_fee_refunded && retired_before_cleanup);
    assert_eq!(registration.installs(), 0);
    assert_released(&gate, expected_bytes);
}

fn returned_active_owner() -> (
    crate::runtime::streaming::sql_recovery_work::JobSqlRecoveryOwner,
    StreamJobContext,
) {
    use crate::runtime::streaming::sql_recovery_work::JobSqlRecoveryOwner;
    let owner = JobSqlRecoveryOwner::new();
    owner.configure(2).unwrap();
    let context = StreamJobContext::new(
        703,
        "returned-active-join-control",
        JsonMap::new(),
        None,
        CancellationToken::new(),
    );
    (owner, context)
}

fn returned_active_controller(
    control_owner: crate::runtime::streaming::sql_recovery_work::JobSqlRecoveryOwner,
    control_gate: Arc<Gate>,
) -> (Controller, Arc<Notify>) {
    let controller_ready = Arc::new(Notify::new());
    let control_ready = controller_ready.clone();
    let controller = Controller {
        gate: control_gate.clone(),
        thread: Some(std::thread::spawn(move || {
            let entered = control_gate.wait_for(|state| state.execution.entered);
            let dropped = control_gate.wait_for(|state| state.control.dropped_start);
            let home_owns_join =
                control_owner.live_counts() == (2, 0) && control_owner.returned_loans() == 1;
            let state = control_gate.state.lock().unwrap();
            let paid_alive = callback_arrays_alive(&state);
            drop(state);
            control_ready.notify_one();
            control_gate.release();
            entered && dropped && home_owns_join && paid_alive
        })),
    };
    (controller, controller_ready)
}

fn submit_sql_restore(
    owner: &crate::runtime::streaming::sql_recovery_work::JobSqlRecoveryOwner,
    operator: SqlOperator,
    saved: OperatorStateSnapshot,
    context: &StreamJobContext,
    node_order: usize,
    launch_cancel: CancellationToken,
) -> crate::runtime::streaming::sql_recovery_work::SqlRecoveryTicket {
    use crate::runtime::streaming::sql_recovery_work::{
        SqlRecoveryContext, SqlRestoreIdentity, SqlRestoreRequest,
    };
    owner
        .submit(SqlRestoreRequest {
            operator,
            snapshot: saved,
            identity: SqlRestoreIdentity {
                node_id: NODE.into(),
                node_order,
                task_id: None,
            },
            context: SqlRecoveryContext::from(context),
            launch_cancel,
        })
        .result
        .unwrap()
}

#[tokio::test(flavor = "current_thread")]
async fn compact_sql_returned_active_join_does_not_block_next_request() {
    let first_directory = tempfile::tempdir().unwrap();
    let first_source = "returned-active-first-source";
    let (_first_directory, _first_manifest, first_saved, first_bytes) =
        captured_sql_recovery(first_directory, first_source).await;
    let second_directory = tempfile::tempdir().unwrap();
    let second_source = "returned-active-second-source";
    let (_second_directory, _second_manifest, second_saved, second_bytes) =
        captured_sql_recovery(second_directory, second_source).await;
    let mut first = sql().await;
    let refund_probe = recovery_refund_probe(&mut first, &first_saved);
    let first_gate = Arc::new(Gate::default());
    let first_registration =
        first.on_prepared_restore_for_test(&expected_normalized_metadata(first_source), {
            let gate = first_gate.clone();
            move |candidate| {
                let witness = candidate_witness(
                    first_source,
                    &candidate.metadata,
                    &candidate.records,
                    (candidate.rows, candidate.bytes),
                    candidate.reserved,
                    candidate.backing,
                );
                drop(candidate.records);
                gate.hold(witness);
                Ok(())
            }
        });
    let second = sql().await;
    let second_gate = Arc::new(Gate::default());
    second_gate.release();
    let first_settled_before_second = Arc::new(AtomicBool::new(false));
    let second_registration =
        second.on_prepared_restore_for_test(&expected_normalized_metadata(second_source), {
            let first_gate = first_gate.clone();
            let second_gate = second_gate.clone();
            let settled = first_settled_before_second.clone();
            move |candidate| {
                observe_predecessor_settled(&first_gate, &settled);
                let witness = candidate_witness(
                    second_source,
                    &candidate.metadata,
                    &candidate.records,
                    (candidate.rows, candidate.bytes),
                    candidate.reserved,
                    candidate.backing,
                );
                drop(candidate.records);
                second_gate.hold(witness);
                Ok(())
            }
        });
    let (owner, context) = returned_active_owner();
    let launch = CancellationToken::new();
    let first = submit_sql_restore(&owner, first, first_saved, &context, 0, launch);
    let launch = CancellationToken::new();
    let second = submit_sql_restore(&owner, second, second_saved, &context, 1, launch);
    let (controller, controller_ready) =
        returned_active_controller(owner.clone(), first_gate.clone());
    let mut first_join = Box::pin(first.join());
    let entered = tokio::select! {
        () = first_gate.async_entered.notified() => true,
        result = &mut first_join => { drop(result); false },
        () = tokio::time::sleep(DEADLINE) => false,
    };
    let held_fee_paid = refund_probe.try_grow(1 << 30).is_err();
    refund_probe.free();
    drop(first_join);
    first_gate.state.lock().unwrap().control.dropped_start = true;
    first_gate.changed.notify_all();
    let controller_observed_return = tokio::time::timeout(DEADLINE, controller_ready.notified())
        .await
        .is_ok();
    let result = tokio::time::timeout(DEADLINE, second.join()).await;
    let completed = matches!(&result, Ok(Ok(completion)) if completion.prepared.is_ok());
    drop(result);
    let returned_fee_refunded = refund_probe.try_grow(1 << 30).is_ok();
    refund_probe.free();
    let retired_before_cleanup = owner.live_counts() == (0, 0);
    first_gate.release();
    let cleanup = owner.drain().await;
    let held_until_return = controller.join();
    assert_eq!(owner.live_counts(), (0, 0));
    assert!(cleanup.is_empty());
    assert!(entered && held_fee_paid && held_until_return && controller_observed_return);
    assert!(
        completed,
        "later SQL request did not join its abandoned active predecessor"
    );
    assert!(returned_fee_refunded && retired_before_cleanup);
    assert!(first_settled_before_second.load(Ordering::SeqCst));
    assert_eq!(first_registration.installs(), 0);
    assert_eq!(second_registration.installs(), 0);
    assert_released(&first_gate, first_bytes);
    assert_released(&second_gate, second_bytes);
}
