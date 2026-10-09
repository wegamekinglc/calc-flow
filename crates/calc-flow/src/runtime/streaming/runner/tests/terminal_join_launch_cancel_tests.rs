use super::*;
use datafusion::arrow::{
    array::TimestampMicrosecondArray,
    datatypes::{DataType, Field, Schema, TimeUnit},
};
use datafusion::execution::memory_pool::{MemoryPool, MemoryReservation};
use std::sync::{Condvar, Mutex as BlockingMutex, Weak};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum PauseAt {
    LocalRead,
    NativeReader,
}

#[derive(Default)]
struct Gate {
    entered: Notify,
    release: BlockingMutex<bool>,
    changed: Condvar,
    expired: AtomicBool,
}

impl Gate {
    fn pause(&self) {
        self.entered.notify_one();
        let held = self.release.lock().unwrap();
        let (_released, timeout) = self
            .changed
            .wait_timeout_while(held, StdDuration::from_secs(10), |released| !*released)
            .unwrap();
        self.expired.store(timeout.timed_out(), Ordering::SeqCst);
    }

    fn release(&self) {
        *self.release.lock().unwrap() = true;
        self.changed.notify_all();
    }
}

struct ReleaseOnDrop(Arc<Gate>);

impl Drop for ReleaseOnDrop {
    fn drop(&mut self) {
        self.0.release();
    }
}

struct Probe {
    pause_at: PauseAt,
    gate: Arc<Gate>,
    wire: Mutex<Vec<Weak<MemoryReservation>>>,
    wire_paid: AtomicBool,
    readers: Mutex<Vec<(bool, bool)>>,
}

impl Probe {
    fn new(pause_at: PauseAt) -> Arc<Self> {
        Arc::new(Self {
            pause_at,
            gate: Arc::new(Gate::default()),
            wire: Mutex::new(Vec::new()),
            wire_paid: AtomicBool::new(true),
            readers: Mutex::new(Vec::new()),
        })
    }

    fn read(&self, bytes: &[u8], capacity: usize, credit: &Arc<MemoryReservation>, peak: usize) {
        let paid = bytes.starts_with(b"CFJDLT1\0") && credit.size() >= capacity + peak;
        self.wire_paid.fetch_and(paid, Ordering::SeqCst);
        let first = {
            let mut wire = self.wire.lock();
            wire.push(Arc::downgrade(credit));
            wire.len() == 1
        };
        if first && self.pause_at == PauseAt::LocalRead {
            self.gate.pause();
        }
    }

    fn reader(&self, credit: Option<&MemoryReservation>, copied: bool, owned: bool) {
        if copied {
            return;
        }
        let first = {
            let mut readers = self.readers.lock();
            readers.push((
                credit.is_some_and(|credit| credit.size() > 0),
                owned && std::thread::current().name() == Some("calc-flow-gather"),
            ));
            readers.len() == 1
        };
        if first && self.pause_at == PauseAt::NativeReader {
            self.gate.pause();
        }
    }
}

#[derive(Default)]
struct Fixture {
    log: Arc<Mutex<Vec<String>>>,
    records: Arc<Mutex<Vec<RecordBatch>>>,
    source_opened: Arc<AtomicUsize>,
    sink_closed: Arc<AtomicUsize>,
}

struct Source {
    timestamp: i64,
    delivered: bool,
    watermark: bool,
    released: Arc<AtomicBool>,
    opened: Arc<AtomicUsize>,
}

#[async_trait]
impl StreamSource for Source {
    async fn open(&mut self, _cursor: Option<Cursor>) -> Result<()> {
        self.opened.fetch_add(1, Ordering::SeqCst);
        Ok(())
    }

    async fn next(&mut self) -> Result<Option<SourceEvent>> {
        if !self.delivered {
            self.delivered = true;
            return Ok(Some(SourceEvent::Data {
                batch: row(self.timestamp),
                cursor: Cursor::unbound(1_u64.to_be_bytes().to_vec(), JsonMap::new()).unwrap(),
            }));
        }
        if self.watermark {
            return Ok(None);
        }
        while !self.released.load(Ordering::SeqCst) {
            tokio::time::sleep(StdDuration::from_millis(5)).await;
        }
        self.watermark = true;
        Ok(Some(SourceEvent::Watermark(EventTime::from_micros(120))))
    }

    async fn close(&mut self) -> Result<()> {
        Ok(())
    }

    fn capabilities(&self) -> SourceCapabilities {
        SourceCapabilities {
            replayable: true,
            max_batch_rows: 1,
            max_batch_bytes: 1 << 20,
        }
    }
}

struct Sink {
    inner: PeriodicCheckpointSink,
    records: Arc<Mutex<Vec<RecordBatch>>>,
}

#[async_trait]
impl TransactionalStreamSink for Sink {
    async fn open(&mut self) -> Result<()> {
        self.inner.open().await
    }
    async fn begin_epoch(&mut self, epoch: crate::Epoch) -> Result<()> {
        self.inner.begin_epoch(epoch).await
    }
    async fn write(&mut self, batch: &Batch) -> Result<()> {
        self.records
            .lock()
            .extend(batch.table_payload()?.batches().iter().cloned());
        self.inner.write(batch).await
    }
    async fn pre_commit(&mut self, epoch: crate::Epoch) -> Result<JsonMap> {
        self.inner.pre_commit(epoch).await
    }
    async fn commit(&mut self, epoch: crate::Epoch, state: &JsonMap) -> Result<()> {
        self.inner.commit(epoch, state).await
    }
    async fn abort(&mut self, epoch: crate::Epoch, state: Option<&JsonMap>) -> Result<()> {
        self.inner.abort(epoch, state).await
    }
    async fn recover(&mut self, manifest: &crate::CheckpointManifest) -> Result<()> {
        self.inner.recover(manifest).await
    }
    async fn close(&mut self) -> Result<()> {
        self.inner.close().await
    }
}

fn schema() -> Arc<Schema> {
    Arc::new(Schema::new(vec![
        Field::new("key", DataType::Int64, false),
        Field::new(
            "ts",
            DataType::Timestamp(TimeUnit::Microsecond, None),
            false,
        ),
    ]))
}

fn row(timestamp: i64) -> Batch {
    Batch::table(
        vec![
            RecordBatch::try_new(
                schema(),
                vec![
                    Arc::new(Int64Array::from(vec![7])),
                    Arc::new(TimestampMicrosecondArray::from(vec![timestamp])),
                ],
            )
            .unwrap(),
        ],
        BatchMetadata::default(),
    )
    .unwrap()
}

fn plan(probe: Option<&Arc<Probe>>) -> (crate::StreamExecutionPlan, Arc<dyn MemoryPool>) {
    plan_with_producer(probe, false)
}

fn v1_plan() -> (crate::StreamExecutionPlan, Arc<dyn MemoryPool>) {
    plan_with_producer(None, true)
}

fn plan_with_producer(
    probe: Option<&Arc<Probe>>,
    v1_producer: bool,
) -> (crate::StreamExecutionPlan, Arc<dyn MemoryPool>) {
    let mut join = StreamJoinOperator::new(
        "match",
        schema(),
        schema(),
        StreamJoinSpec::inner(
            ["key"],
            ["key"],
            "ts",
            "ts",
            JoinTimeBounds::new(StdDuration::ZERO, StdDuration::from_micros(10)).unwrap(),
            JoinStateLimits::new(100_000, 134_217_728, 1_000_000).unwrap(),
        )
        .unwrap(),
    )
    .unwrap();
    if v1_producer {
        join.set_checkpoint_v1_test_producer();
    }
    let pool = join.checkpoint_preload_test_pool().unwrap();
    if let Some(probe) = probe {
        let probe = Arc::clone(probe);
        join.set_checkpoint_decoded_row_test_hook(Arc::new(move |credit, _, copied, owned| {
            probe.reader(credit, copied, owned);
        }));
    }
    let plan = PipelineBuilder::new("terminal-join-launch-cancel")
        .unwrap()
        .add_node("match", Box::new(join))
        .unwrap()
        .compile_stream(
            &UdfRegistry::new().snapshot(),
            &StreamRequirements::default(),
        )
        .unwrap();
    (plan, pool)
}

fn spec(
    plan: crate::StreamExecutionPlan,
    fixture: &Fixture,
    released: &Arc<AtomicBool>,
) -> ContinuousJobSpec {
    ContinuousJobSpec {
        context: StreamJobContext::new(
            92,
            plan.fingerprint(),
            JsonMap::new(),
            None,
            CancellationToken::new(),
        ),
        plan,
        sources: [("left", 95), ("right", 100)]
            .into_iter()
            .map(|(id, timestamp)| NamedSourceBinding {
                binding_id: id.into(),
                binding: SourceBinding::new(
                    Box::new(Source {
                        timestamp,
                        delivered: false,
                        watermark: false,
                        released: Arc::clone(released),
                        opened: Arc::clone(&fixture.source_opened),
                    }),
                    None,
                    0,
                )
                .unwrap(),
            })
            .collect(),
        sinks: vec![NamedSinkBinding {
            output_id: "output".into(),
            sink_id: "pairs".into(),
            binding: OrdinarySinkBinding::new_transactional(Box::new(Sink {
                inner: PeriodicCheckpointSink {
                    log: Arc::clone(&fixture.log),
                    closed: Arc::clone(&fixture.sink_closed),
                },
                records: Arc::clone(&fixture.records),
            })),
        }],
        edge_budget: EdgeBudget {
            max_rows: 1,
            max_bytes: 1 << 20,
        },
        delivery_mode: M2DeliveryMode::ProcessLocalOrdered,
        static_inputs: crate::static_input::PreparedStaticInputs::default(),
    }
}

fn checkpoint(root: &Path, probe: Option<&Arc<Probe>>) -> CheckpointRuntimeSpec {
    let checkpoint = CheckpointRuntimeSpec::managed(
        ManagedCheckpointRuntime::new(root).unwrap(),
        StreamRuntimeConfig {
            checkpoint_interval: StdDuration::from_secs(3_600),
            checkpoint_timeout: StdDuration::from_secs(10),
            ..StreamRuntimeConfig::default()
        },
    )
    .unwrap();
    match probe {
        None => checkpoint,
        Some(probe) => {
            let probe = Arc::clone(probe);
            checkpoint.with_join_preload_read_hook(Arc::new(
                move |bytes, capacity, credit, peak| {
                    probe.read(bytes, capacity, credit, peak);
                    Ok(())
                },
            ))
        }
    }
}

async fn natural_terminal_history(root: &Path) {
    let fixture = Fixture::default();
    let released = Arc::new(AtomicBool::new(false));
    let (plan, pool) = v1_plan();
    let mut runner = ContinuousRunner::new();
    let job = runner
        .start_checkpointed(spec(plan, &fixture, &released), checkpoint(root, None))
        .await
        .unwrap();
    wait_for_join_emission(&job, 1).await;
    job.trigger_checkpoint().await.unwrap();
    released.store(true, Ordering::SeqCst);
    let outcome = tokio::time::timeout(StdDuration::from_secs(30), job.wait())
        .await
        .unwrap();
    assert_eq!(outcome.state, ContinuousJobState::Completed);
    assert_eq!(outcome.cause, TerminalCause::NaturalEnd);
    drop(job);
    runner.shutdown().await.unwrap();
    assert_eq!(pool.reserved(), 0);
    let records = fixture.records.lock();
    assert_eq!(records.iter().map(RecordBatch::num_rows).sum::<usize>(), 1);
    let record = &records[0];
    let value = |name: &str| record.column_by_name(name).unwrap().as_any();
    assert_eq!(
        value("left__key")
            .downcast_ref::<Int64Array>()
            .unwrap()
            .value(0),
        7
    );
    assert_eq!(
        value("right__key")
            .downcast_ref::<Int64Array>()
            .unwrap()
            .value(0),
        7
    );
    assert_eq!(
        value("left__ts")
            .downcast_ref::<TimestampMicrosecondArray>()
            .unwrap()
            .value(0),
        95
    );
    assert_eq!(
        value("right__ts")
            .downcast_ref::<TimestampMicrosecondArray>()
            .unwrap()
            .value(0),
        100
    );
}

struct Observation {
    pause_at: PauseAt,
    probe: Arc<Probe>,
    forwarded: bool,
    drain_pending: bool,
    pre_release_reserved: usize,
    post_drain_reserved: usize,
    owner: DriverOwnership,
    statuses: usize,
    source_opened: usize,
    sink_log: Vec<String>,
}

async fn cancel_start(root: &Path, pause_at: PauseAt) -> Observation {
    let probe = Probe::new(pause_at);
    let release_guard = ReleaseOnDrop(Arc::clone(&probe.gate));
    let fixture = Fixture::default();
    let released = Arc::new(AtomicBool::new(true));
    let (plan, pool) = plan(Some(&probe));
    let spec = spec(plan, &fixture, &released);
    let job_cancel = spec.context.cancellation().clone();
    let mut runner = ContinuousRunner::new();
    let start = runner.start_checkpointed(spec, checkpoint(root, Some(&probe)));
    let core = Arc::clone(start.core.as_ref().unwrap());
    tokio::time::timeout(StdDuration::from_secs(10), probe.gate.entered.notified())
        .await
        .expect("a real paid restore operation must reach the gate");
    assert!(!core.launch_cancel.is_cancelled());
    assert!(!job_cancel.is_cancelled());
    drop(start);
    assert!(core.launch_cancel.is_cancelled());
    let mut shutdown = Box::pin(runner.shutdown());
    let drain_pending = matches!(futures::poll!(shutdown.as_mut()), Poll::Pending);
    let forwarded = tokio::time::timeout(StdDuration::from_secs(1), job_cancel.cancelled())
        .await
        .is_ok();
    let pre_release_reserved = pool.reserved();
    probe.gate.release();
    tokio::time::timeout(StdDuration::from_secs(30), shutdown)
        .await
        .unwrap()
        .unwrap();
    let statuses = core.runtime_status.lock().nodes.len();
    let owner = core.state.lock().owner;
    drop(core);
    drop(release_guard);
    let sink_log = fixture.log.lock().clone();
    Observation {
        pause_at,
        probe,
        forwarded,
        drain_pending,
        pre_release_reserved,
        post_drain_reserved: pool.reserved(),
        owner,
        statuses,
        source_opened: fixture.source_opened.load(Ordering::SeqCst),
        sink_log,
    }
}

fn print_observation(observed: &Observation) {
    let probe = &observed.probe;
    let readers = probe.readers.lock();
    eprintln!(
        "launch {:?}: forwarded={}, wire={}, readers={readers:?}, drain_pending={}, reserved={}/{}, owner={:?}, statuses={}, source_opened={}, sink={:?}",
        observed.pause_at,
        observed.forwarded,
        probe.wire.lock().len(),
        observed.drain_pending,
        observed.pre_release_reserved,
        observed.post_drain_reserved,
        observed.owner,
        observed.statuses,
        observed.source_opened,
        observed.sink_log
    );
}

fn assert_cancelled(observed: &Observation, expected_readers: usize) {
    let probe = &observed.probe;
    let readers = probe.readers.lock();
    assert!(!probe.gate.expired.load(Ordering::SeqCst));
    assert!(observed.drain_pending);
    assert!(observed.pre_release_reserved > 0);
    assert_eq!(observed.post_drain_reserved, 0);
    assert_eq!(observed.owner, DriverOwnership::Terminal);
    assert!(probe.wire_paid.load(Ordering::SeqCst));
    assert!(!probe.wire.lock().is_empty());
    assert!(
        probe
            .wire
            .lock()
            .iter()
            .all(|credit| credit.upgrade().is_none())
    );
    assert!(readers.iter().all(|(paid, native)| *paid && *native));
    assert!(
        observed.forwarded,
        "launch cancellation must reach the real job token"
    );
    assert_eq!(
        readers.len(),
        expected_readers,
        "no native or Original replay after launch stop"
    );
    assert_eq!(
        observed.statuses, 0,
        "cancelled recovery must not publish partial Join status"
    );
    assert_eq!(observed.source_opened, 0);
    assert!(
        !observed
            .sink_log
            .iter()
            .any(|entry| entry.starts_with("sink-recover:"))
    );
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn test_terminal_join_launch_cancel_stops_paid_load_and_native_restore_before_publish() {
    let directory = tempfile::tempdir().unwrap();
    let root = directory.path().join("managed");
    natural_terminal_history(&root).await;
    let local = cancel_start(&root, PauseAt::LocalRead).await;
    let native = cancel_start(&root, PauseAt::NativeReader).await;
    print_observation(&local);
    print_observation(&native);
    assert_cancelled(&local, 0);
    assert_cancelled(&native, 1);
}
