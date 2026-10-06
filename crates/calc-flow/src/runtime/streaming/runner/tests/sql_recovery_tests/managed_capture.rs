use std::{any::Any, sync::Weak, thread::ThreadId};

use datafusion::execution::memory_pool::{MemoryPool, MemoryReservation};

use super::*;

#[derive(Clone, Copy, Default, Eq, PartialEq)]
enum Phase {
    #[default]
    Waiting,
    Held,
    Exited,
}

#[derive(Default)]
struct CaptureWitness {
    phase: Phase,
    heartbeats: usize,
    thread: Option<ThreadId>,
    operator: Option<Weak<()>>,
    capture: Option<Weak<dyn Any + Send + Sync>>,
    fee: Option<Weak<MemoryReservation>>,
    pool: Option<Arc<dyn MemoryPool>>,
    reserved: usize,
    layout: u64,
    segments: Vec<String>,
    metadata_matches: bool,
}

#[derive(Default)]
struct CaptureGate {
    state: std::sync::Mutex<CaptureWitness>,
    changed: Condvar,
    entered: Notify,
    release: AtomicBool,
}

impl CaptureGate {
    fn hold(&self, mut witness: CaptureWitness) {
        witness.phase = Phase::Held;
        *self.state.lock().unwrap() = witness;
        self.entered.notify_one();
        self.changed.notify_all();
        let mut state = self.state.lock().unwrap();
        while !self.release.load(Ordering::SeqCst) {
            state = self.changed.wait(state).unwrap();
        }
        state.phase = Phase::Exited;
        self.changed.notify_all();
    }

    fn release(&self) {
        let _state = self
            .state
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        self.release.store(true, Ordering::SeqCst);
        self.changed.notify_all();
    }

    fn wait_for(&self, condition: impl Fn(&CaptureWitness) -> bool) -> bool {
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

struct HoldReport {
    checks: BTreeMap<&'static str, bool>,
    heartbeats: usize,
    acks: usize,
    pre_commits: usize,
    commits: usize,
    exits: usize,
}

struct CaptureController {
    gate: Arc<CaptureGate>,
    thread: Option<std::thread::JoinHandle<HoldReport>>,
}

impl CaptureController {
    fn join(mut self) -> HoldReport {
        self.thread.take().unwrap().join().unwrap()
    }
}

impl Drop for CaptureController {
    fn drop(&mut self) {
        self.gate.release();
        if let Some(thread) = self.thread.take() {
            let _ = thread.join();
        }
    }
}

struct ReleaseOnDrop(Arc<CaptureGate>);

impl Drop for ReleaseOnDrop {
    fn drop(&mut self) {
        self.0.release();
    }
}

#[derive(Clone, Copy, Eq, PartialEq)]
enum Stop {
    None,
    Cancel,
    Drop,
}

fn capture_checkpoint(root: &Path) -> CheckpointRuntimeSpec {
    CheckpointRuntimeSpec::managed(
        ManagedCheckpointRuntime::new(root).unwrap(),
        StreamRuntimeConfig {
            checkpoint_interval: StdDuration::from_secs(3_600),
            checkpoint_timeout: DEADLINE + DEADLINE + DEADLINE,
            ..StreamRuntimeConfig::default()
        },
    )
    .unwrap()
}

async fn ready(probe: &Probe) {
    tokio::time::timeout(DEADLINE, async {
        while !probe.tail.load(Ordering::SeqCst) || probe.writes.lock().is_empty() {
            tokio::task::yield_now().await;
        }
    })
    .await
    .unwrap();
}

fn assert_released_capture(gate: &CaptureGate) {
    let witness = gate.state.lock().unwrap();
    assert!(witness.phase == Phase::Exited);
    assert_eq!(witness.layout, 3);
    assert_eq!(
        witness.segments,
        ["batch-metadata", "control", "group-state", "logical-schema"]
    );
    assert!(witness.metadata_matches);
    assert!(witness.reserved > 0);
    assert!(witness.operator.as_ref().unwrap().upgrade().is_none());
    assert!(witness.capture.as_ref().unwrap().upgrade().is_none());
    assert!(witness.fee.as_ref().unwrap().upgrade().is_none());
    assert_eq!(witness.pool.as_ref().unwrap().reserved(), 0);
}

async fn current_manifest(root: &Path) -> crate::CheckpointManifest {
    let bytes = tokio::fs::read(root.join("manifests/manifest-00000000000000000001.json"))
        .await
        .unwrap();
    let manifest = crate::CheckpointManifest::from_bytes(&bytes).unwrap();
    let entry = &manifest.operators()[NODE];
    assert_eq!(entry.inline_metadata["state_layout"], serde_json::json!(3));
    assert_eq!(
        entry.inline_metadata["state_accounting"],
        serde_json::json!(3)
    );
    assert_eq!(entry.inline_metadata["rows"], serde_json::json!(2));
    assert_eq!(entry.segments.len(), 4);
    manifest
}

async fn assert_current_restore(root: &Path, manifest: &crate::CheckpointManifest, source: &str) {
    let captured = snapshot(root, manifest).await;
    let mut restored = sql().await;
    restored.restore(&captured).unwrap();
    let job = StreamJobContext::new(
        709,
        "capture-oracle",
        JsonMap::new(),
        None,
        CancellationToken::new(),
    );
    let context = StreamOperatorContext::new(&job, NODE, None);
    let empty = Batch::table(
        vec![RecordBatch::new_empty(logical_schema())],
        expected_normalized_metadata(source),
    )
    .unwrap();
    let mut collector = crate::EdgeCollector::new(restored.output_ports().to_vec());
    restored
        .process_data("events", empty, &context, &mut collector)
        .await
        .unwrap();
    let output = collector.drain("output");
    assert_eq!(output.len(), 1);
    let probe = Probe::default();
    probe
        .writes
        .lock()
        .push(output[0].as_data().unwrap().clone());
    assert_capture_output(
        &probe,
        source,
        sql().await.output_ports()[0].schema().unwrap().clone(),
    );
}

fn held_observations(
    gate: &CaptureGate,
    core: &JobCore,
    probe: &Probe,
    counts: &[Arc<AtomicUsize>; 3],
    events: [bool; 3],
    task_id: Option<u64>,
    stop: Stop,
) -> HoldReport {
    let witness = gate.state.lock().unwrap();
    let [entered, progressed, stopped] = events;
    HoldReport {
        checks: BTreeMap::from([
            ("entered", entered),
            ("heartbeat while native held", progressed),
            ("actual context stop processed", stopped),
            (
                "original operator alive",
                witness.operator.as_ref().unwrap().upgrade().is_some(),
            ),
            (
                "actual candidate alive",
                witness.capture.as_ref().unwrap().upgrade().is_some(),
            ),
            (
                "candidate fee alive",
                witness.fee.as_ref().unwrap().upgrade().is_some(),
            ),
            (
                "actual pool fee paid",
                witness.pool.as_ref().unwrap().reserved() >= witness.reserved,
            ),
            ("job outcome pending", core.state.lock().outcome.is_none()),
            (
                "original task identity retained by native owner",
                task_id.is_some_and(|id| core.sql_recovery.active_task_for_test(NODE) == Some(id)),
            ),
            (
                "running actor task registered",
                stop != Stop::None
                    || core
                        .runtime_status
                        .lock()
                        .tasks
                        .snapshot()
                        .keys()
                        .any(|id| Some(id.as_u64()) == task_id),
            ),
            (
                "no candidate installed",
                counts[0].load(Ordering::SeqCst) == 0,
            ),
        ]),
        heartbeats: witness.heartbeats,
        acks: counts[1].load(Ordering::SeqCst),
        pre_commits: probe.sink_pre_commits.load(Ordering::SeqCst),
        commits: probe.sink_commits.load(Ordering::SeqCst),
        exits: counts[2].load(Ordering::SeqCst),
    }
}

fn controller(
    gate: Arc<CaptureGate>,
    core: Arc<JobCore>,
    probe: Arc<Probe>,
    stop: Stop,
    owned_job: Option<super::super::super::ContinuousJob>,
    cancellation: CancellationToken,
    counts: [Arc<AtomicUsize>; 3],
) -> CaptureController {
    let control_gate = gate.clone();
    CaptureController {
        gate,
        thread: Some(std::thread::spawn(move || {
            let _release = ReleaseOnDrop(control_gate.clone());
            let entered = control_gate.wait_for(|state| state.phase == Phase::Held);
            let progressed = control_gate.wait_for(|state| state.heartbeats > 0);
            let task_id =
                core.runtime_status
                    .lock()
                    .tasks
                    .snapshot()
                    .iter()
                    .find_map(|(&id, task)| {
                        (task.task_name == format!("operator:{NODE}")).then_some(id.as_u64())
                    });
            match stop {
                Stop::Cancel => core.request_cancel(false),
                Stop::Drop => drop(owned_job),
                Stop::None => {}
            }
            let stopped =
                stop == Stop::None || control_gate.wait_for(|_| cancellation.is_cancelled());
            let report = held_observations(
                &control_gate,
                &core,
                &probe,
                &counts,
                [entered, progressed, stopped],
                task_id,
                stop,
            );
            control_gate.release();
            report
        })),
    }
}

fn heartbeat(gate: Arc<CaptureGate>) -> tokio::task::JoinHandle<()> {
    tokio::spawn(async move {
        gate.entered.notified().await;
        let mut ticks = tokio::time::interval(StdDuration::from_millis(1));
        loop {
            ticks.tick().await;
            let mut witness = gate.state.lock().unwrap();
            if witness.phase == Phase::Exited {
                break;
            }
            if witness.phase == Phase::Held {
                witness.heartbeats += 1;
                gate.changed.notify_all();
            }
        }
    })
}

async fn assert_capture_result(
    result: Result<Epoch>,
    root: &Path,
    source: &str,
    stop: Stop,
    core: &JobCore,
    counts: &[Arc<AtomicUsize>; 3],
) {
    if stop == Stop::None {
        assert_eq!(result.unwrap(), Epoch::INITIAL);
        let manifest = current_manifest(root).await;
        assert!(!manifest.sources()["events"].ended);
        assert_current_restore(root, &manifest, source).await;
        assert_eq!(counts[1].load(Ordering::SeqCst), 1);
    } else {
        assert!(result.is_err());
        assert_eq!(
            core.state.lock().outcome.as_ref().unwrap().state,
            ContinuousJobState::Cancelled
        );
        assert_eq!(
            counts[0].load(Ordering::SeqCst),
            0,
            "stopped candidate installed"
        );
        assert_eq!(counts[1].load(Ordering::SeqCst), 0);
        assert!(
            tokio::fs::metadata(root.join("manifests/manifest-00000000000000000001.json"))
                .await
                .is_err()
        );
    }
}

fn assert_held_capture_report(report: &HoldReport, gate: &CaptureGate, executor_thread: ThreadId) {
    assert_eq!(
        (
            report.acks,
            report.pre_commits,
            report.commits,
            report.exits
        ),
        (0, 0, 0, 0)
    );
    assert!(
        report.checks.values().all(|value| *value),
        "held observations: {:?}",
        report.checks
    );
    assert!(report.heartbeats > 0);
    assert_ne!(gate.state.lock().unwrap().thread.unwrap(), executor_thread);
}

async fn held_capture(source: &'static str, stop: Stop) {
    let directory = tempfile::tempdir().unwrap();
    let gate = Arc::new(CaptureGate::default());
    let declaration = sql().await;
    let guard = Arc::new(declaration.on_prepared_checkpoint_for_test(
        &expected_normalized_metadata(source),
        {
            let gate = gate.clone();
            move |candidate| {
                let witness = CaptureWitness {
                    thread: Some(candidate.thread),
                    operator: Some(candidate.operator),
                    capture: Some(candidate.capture),
                    fee: Some(candidate.fee),
                    pool: Some(candidate.pool),
                    reserved: candidate.reserved,
                    layout: candidate.layout,
                    segments: candidate.segments,
                    metadata_matches: candidate.metadata == expected_normalized_metadata(source),
                    ..CaptureWitness::default()
                };
                drop(candidate.metadata);
                gate.hold(witness);
                Ok(())
            }
        },
    ));
    let probe = Arc::new(Probe::default());
    let specification = spec(source, false, &probe).await;
    let cancellation = specification.context.cancellation().clone();
    let mut runner = ContinuousRunner::new();
    let mut job = Some(
        runner
            .start_checkpointed(specification, capture_checkpoint(directory.path()))
            .await
            .unwrap(),
    );
    ready(&probe).await;
    let core = job.as_ref().unwrap().core.clone();
    let coordinator = core.manual_checkpoint.lock().clone().unwrap();
    let owned_job = if stop == Stop::Drop { job.take() } else { None };
    let controller = controller(
        gate.clone(),
        core.clone(),
        probe.clone(),
        stop,
        owned_job,
        cancellation,
        guard.counts(),
    );
    let executor_thread = std::thread::current().id();
    let heartbeat = heartbeat(gate.clone());
    let requested = coordinator.request_manual().await.unwrap();
    let result = requested.await;
    if let Some(job) = job.take() {
        let _ = job.cancel().await;
        drop(job);
    }
    runner.shutdown().await.unwrap();
    heartbeat.abort();
    let _ = heartbeat.await;
    let report = controller.join();
    assert_eq!(runner.registry_counts(), (0, 0));
    assert!(core.runtime_status.lock().tasks.snapshot().is_empty());
    assert_capture_result(
        result,
        directory.path(),
        source,
        stop,
        &core,
        &guard.counts(),
    )
    .await;
    assert!(guard.exits() > 0);
    if stop != Stop::None {
        assert_eq!(guard.installs(), 0);
        assert_eq!(guard.acks(), 0);
    }
    assert_capture_output(
        &probe,
        source,
        declaration.output_ports()[0].schema().unwrap().clone(),
    );
    probe.writes.lock().clear();
    assert_released_capture(&gate);
    assert_held_capture_report(&report, &gate, executor_thread);
}

#[tokio::test(flavor = "current_thread")]
async fn current_native3_capture_yields_before_ack_or_barrier() {
    held_capture("current-capture-heartbeat", Stop::None).await;
}

#[tokio::test(flavor = "current_thread")]
async fn current_native3_capture_cancel_keeps_original_and_fee_until_join() {
    held_capture("current-capture-cancel", Stop::Cancel).await;
}

#[tokio::test(flavor = "current_thread")]
async fn current_native3_capture_job_drop_keeps_original_and_fee_until_join() {
    held_capture("current-capture-drop", Stop::Drop).await;
}

async fn successful_capture(source: &str, terminal: bool) {
    let directory = tempfile::tempdir().unwrap();
    let probe = Arc::new(Probe::default());
    let mut runner = ContinuousRunner::new();
    let job = runner
        .start_checkpointed(
            spec(source, terminal, &probe).await,
            capture_checkpoint(directory.path()),
        )
        .await
        .unwrap();
    if terminal {
        assert_eq!(job.wait().await.state, ContinuousJobState::Completed);
    } else {
        ready(&probe).await;
        assert_eq!(job.trigger_checkpoint().await.unwrap(), Epoch::INITIAL);
        let _ = job.cancel().await;
    }
    drop(job);
    runner.shutdown().await.unwrap();
    assert_eq!(runner.registry_counts(), (0, 0));
    let manifest = current_manifest(directory.path()).await;
    assert_eq!(manifest.sources()["events"].ended, terminal);
    assert_current_restore(directory.path(), &manifest, source).await;
    assert_capture_output(
        &probe,
        source,
        sql().await.output_ports()[0].schema().unwrap().clone(),
    );
    assert_eq!(probe.sink_pre_commits.load(Ordering::SeqCst), 1);
    assert_eq!(probe.sink_commits.load(Ordering::SeqCst), 1);
    assert_eq!(probe.source_closes.load(Ordering::SeqCst), 1);
    assert_eq!(probe.sink_closes.load(Ordering::SeqCst), 1);
}

#[tokio::test(flavor = "current_thread")]
async fn current_native3_capture_ordinary_success_control() {
    successful_capture("current-capture-ordinary", false).await;
}

#[tokio::test(flavor = "current_thread")]
async fn current_native3_capture_terminal_success_control() {
    successful_capture("current-capture-terminal", true).await;
}
