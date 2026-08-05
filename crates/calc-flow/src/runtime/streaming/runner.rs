use std::{
    collections::{BTreeMap, BTreeSet, VecDeque},
    future::Future,
    panic::AssertUnwindSafe,
    pin::Pin,
    sync::{
        Arc,
        atomic::{AtomicBool, AtomicU64, Ordering},
    },
    task::{Context, Poll},
    time::Duration,
};

use chrono::Utc;
use futures::FutureExt;
use parking_lot::Mutex;
use tokio::{
    sync::{Notify, mpsc, watch},
    task::{JoinHandle, JoinSet},
};

use super::{
    ChannelMetrics, EdgeReceiver, EdgeSender,
    channel::edge_channel_with_metrics,
    job::{
        ContinuousJobSpec, OrdinarySinkBinding, ValidatedContinuousJob, ValidatedOrdinarySink,
        preflight_job,
    },
    metrics::{M2MetricsSnapshot, MetricsRecorder, sink_metric_id},
    operator_task::{
        OperatorIngress, OperatorProgress, OperatorProgressSnapshot, OperatorTaskInputs,
        spawn_operator_task,
    },
    sink_task::{SinkFailurePhase, SinkProgress, SinkTaskInputs, spawn_sink_task},
    source_task::{
        SourceBinding, SourceProgress, SourceProgressSnapshot,
        spawn_source_tasks_gated_with_metrics,
    },
    supervisor::{
        SupervisionReport, TaskId, TaskRegistry, TaskStatus, TaskSupervisor, TerminalArbiter,
        TerminalDecision, panic_message,
    },
};
use crate::pipeline::{
    RuntimeSinkRoute, RuntimeSourceRoute, RuntimeStreamNode, StreamRuntimePlanParts,
};
use crate::{CalcFlowError, CancellationToken};

#[derive(Clone, Debug, Eq, Ord, PartialEq, PartialOrd)]
pub(crate) enum FailureOrigin {
    Preflight,
    OperatorEntry {
        node_id: String,
    },
    SourceOpen {
        binding_id: String,
    },
    SinkOpen {
        output_id: String,
        sink_id: String,
    },
    SourceClose {
        binding_id: String,
    },
    SinkClose {
        output_id: String,
        sink_id: String,
    },
    SinkWrite {
        output_id: String,
        sink_id: String,
    },
    SinkIngress {
        output_id: String,
        edge_id: String,
    },
    Task {
        task_id: TaskId,
        task_name: String,
    },
    Metrics {
        component_id: String,
        counter: &'static str,
    },
}

#[derive(Debug)]
pub(crate) struct RuntimeFailure {
    pub(crate) origin: FailureOrigin,
    pub(crate) error: CalcFlowError,
}

#[derive(Clone, Debug)]
pub(crate) struct StartFailure {
    pub(crate) primary: Arc<RuntimeFailure>,
    pub(crate) diagnostic_id: Option<u64>,
}

pub(crate) type StartResult<T> = Result<T, StartFailure>;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) enum ContinuousJobState {
    Running,
    Draining,
    Completed,
    Cancelled,
    Failed,
    RecoveryRequired,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub(crate) enum TerminalCause {
    NaturalEnd,
    GracefulShutdown,
    ExplicitCancel,
    DeadlineExceeded,
    TaskFailure { primary_task_id: TaskId },
}

#[derive(Clone, Debug)]
pub(crate) struct ContinuousJobOutcome {
    pub(crate) state: ContinuousJobState,
    pub(crate) cause: TerminalCause,
    pub(crate) errors: Vec<Arc<RuntimeFailure>>,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) enum DriverOwnership {
    CoreOwned,
    Driving,
    ReaperOwned,
    Terminal,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) enum LaunchDeliveryState {
    Provisional,
    ReadyUnclaimed,
    Claimed,
    CancelRequested,
    Failed,
}

#[derive(Clone, Copy, Debug, Eq, Ord, PartialEq, PartialOrd)]
pub(crate) struct LaunchId(u64);

struct JobCoreState {
    owner: DriverOwnership,
    launch_delivery: LaunchDeliveryState,
    state: ContinuousJobState,
    selected_cause: Option<TerminalCause>,
    outcome: Option<Arc<ContinuousJobOutcome>>,
    start_failure: Option<StartFailure>,
}

struct JobCore {
    launch_id: LaunchId,
    job_id: u64,
    state: Mutex<JobCoreState>,
    terminal_arbiter: TerminalArbiter,
    changed: Notify,
    launch_cancel: CancellationToken,
    runner_commands: mpsc::UnboundedSender<RunnerCommand>,
    metrics: MetricsRecorder,
    runtime_status: Mutex<RuntimeStatus>,
    #[cfg(test)]
    terminal_commit_seam: Mutex<Option<TerminalCommitTestSeam>>,
}

#[cfg(test)]
struct TerminalCommitTestSeam {
    reached: tokio::sync::oneshot::Sender<()>,
    release: tokio::sync::oneshot::Receiver<()>,
}

#[derive(Default)]
struct RuntimeStatus {
    tasks: TaskRegistry,
    sources: BTreeMap<String, SourceProgress>,
    nodes: BTreeMap<String, OperatorProgress>,
    sinks: BTreeMap<String, SinkProgress>,
    sink_outputs: BTreeMap<String, String>,
}

impl JobCore {
    fn new(
        launch_id: LaunchId,
        job_id: u64,
        runner_commands: mpsc::UnboundedSender<RunnerCommand>,
        metrics: MetricsRecorder,
        sink_outputs: BTreeMap<String, String>,
    ) -> Self {
        Self {
            launch_id,
            job_id,
            state: Mutex::new(JobCoreState {
                owner: DriverOwnership::CoreOwned,
                launch_delivery: LaunchDeliveryState::Provisional,
                state: ContinuousJobState::Running,
                selected_cause: None,
                outcome: None,
                start_failure: None,
            }),
            terminal_arbiter: TerminalArbiter::default(),
            changed: Notify::new(),
            launch_cancel: CancellationToken::new(),
            runner_commands,
            metrics,
            runtime_status: Mutex::new(RuntimeStatus {
                sink_outputs,
                ..RuntimeStatus::default()
            }),
            #[cfg(test)]
            terminal_commit_seam: Mutex::new(None),
        }
    }

    fn request_cancel(&self, reaper_owned: bool) {
        self.terminal_arbiter.request_explicit_cancel();
        let cancel_launch = {
            let mut state = self.state.lock();
            if state.owner != DriverOwnership::Terminal {
                if reaper_owned {
                    state.owner = DriverOwnership::ReaperOwned;
                }
                if state.launch_delivery != LaunchDeliveryState::Claimed && reaper_owned {
                    state.launch_delivery = LaunchDeliveryState::CancelRequested;
                }
            }
            state.launch_delivery != LaunchDeliveryState::Claimed
        };
        if cancel_launch {
            self.launch_cancel.cancel();
        }
        let _ = self
            .runner_commands
            .send(RunnerCommand::Wake(self.launch_id));
        self.changed.notify_waiters();
    }

    /// Synchronously transfers a non-terminal job to runner-drop reaper
    /// ownership. The caller holds the runner registry lock, which is the
    /// linearization point shared with terminal publication.
    fn request_runner_drop_cancel(&self) -> bool {
        self.terminal_arbiter.request_explicit_cancel();
        let mut state = self.state.lock();
        if state.owner == DriverOwnership::Terminal {
            return false;
        }
        state.owner = DriverOwnership::ReaperOwned;
        let cancel_launch = state.launch_delivery != LaunchDeliveryState::Claimed;
        if cancel_launch {
            state.launch_delivery = LaunchDeliveryState::CancelRequested;
        }
        drop(state);
        if cancel_launch {
            self.launch_cancel.cancel();
        }
        let _ = self.metrics.record_abandoned_runner_drop();
        let _ = self
            .runner_commands
            .send(RunnerCommand::Wake(self.launch_id));
        self.changed.notify_waiters();
        true
    }

    fn request_shutdown(&self) {
        self.terminal_arbiter.request_graceful_shutdown();
        let mut state = self.state.lock();
        if state.state == ContinuousJobState::Running {
            state.state = ContinuousJobState::Draining;
        }
        drop(state);
        let _ = self
            .runner_commands
            .send(RunnerCommand::Wake(self.launch_id));
        self.changed.notify_waiters();
    }

    fn request_deadline(&self) {
        self.terminal_arbiter.request_deadline();
        let _ = self
            .runner_commands
            .send(RunnerCommand::Wake(self.launch_id));
        self.changed.notify_waiters();
    }

    #[cfg(test)]
    fn install_terminal_commit_seam(
        &self,
    ) -> (
        tokio::sync::oneshot::Receiver<()>,
        tokio::sync::oneshot::Sender<()>,
    ) {
        let (reached_tx, reached_rx) = tokio::sync::oneshot::channel();
        let (release_tx, release_rx) = tokio::sync::oneshot::channel();
        let previous = self
            .terminal_commit_seam
            .lock()
            .replace(TerminalCommitTestSeam {
                reached: reached_tx,
                release: release_rx,
            });
        assert!(
            previous.is_none(),
            "only one terminal commit seam may be active"
        );
        (reached_rx, release_tx)
    }

    #[cfg(test)]
    async fn pause_before_terminal_commit(&self) {
        let seam = self.terminal_commit_seam.lock().take();
        if let Some(seam) = seam {
            let _ = seam.reached.send(());
            let _ = seam.release.await;
        }
    }
}

pub(crate) struct ContinuousJob {
    core: Arc<JobCore>,
    _ownership: JobOwnershipToken,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub(crate) struct SourceStatus {
    pub(crate) replayable: bool,
    pub(crate) latest_observed_order: Option<Vec<u8>>,
    pub(crate) durable_order: Option<Vec<u8>>,
    pub(crate) next_sequence: Option<u64>,
    pub(crate) ended: bool,
}

impl From<SourceProgressSnapshot> for SourceStatus {
    fn from(progress: SourceProgressSnapshot) -> Self {
        Self {
            replayable: progress.replayable,
            latest_observed_order: progress
                .latest_observed_cursor
                .map(|cursor| cursor.order().to_vec()),
            durable_order: progress
                .durable_cursor
                .map(|cursor| cursor.order().to_vec()),
            next_sequence: progress.next_sequence,
            ended: progress.ended,
        }
    }
}

#[derive(Clone, Debug, Default, Eq, PartialEq)]
pub(crate) struct OperatorStatus {
    pub(crate) input_batches: u64,
    pub(crate) fully_fanned_out_batches: u64,
    pub(crate) datafusion_runtime_created: bool,
    pub(crate) on_end_calls: u64,
    pub(crate) ended: bool,
}

impl From<OperatorProgressSnapshot> for OperatorStatus {
    fn from(progress: OperatorProgressSnapshot) -> Self {
        Self {
            input_batches: progress.input_batches,
            fully_fanned_out_batches: progress.fully_fanned_out_batches,
            datafusion_runtime_created: progress.datafusion_runtime_created,
            on_end_calls: progress.on_end_calls,
            ended: progress.ended,
        }
    }
}

#[derive(Clone, Debug, Default, Eq, PartialEq)]
pub(crate) struct SinkStatus {
    pub(crate) delivered_batches: u64,
    pub(crate) delivered_rows: u64,
    pub(crate) delivered_bytes: u64,
    pub(crate) ended: bool,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub(crate) struct ContinuousJobStatus {
    pub(crate) job_id: u64,
    pub(crate) state: ContinuousJobState,
    pub(crate) terminal_cause: Option<TerminalCause>,
    pub(crate) tasks: BTreeMap<TaskId, TaskStatus>,
    pub(crate) edges: BTreeMap<String, ChannelMetrics>,
    pub(crate) sources: BTreeMap<String, SourceStatus>,
    pub(crate) nodes: BTreeMap<String, OperatorStatus>,
    pub(crate) sinks: BTreeMap<String, SinkStatus>,
    pub(crate) metrics: M2MetricsSnapshot,
}

impl std::fmt::Debug for ContinuousJob {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter
            .debug_struct("ContinuousJob")
            .field("job_id", &self.core.job_id)
            .field("state", &self.core.state.lock().state)
            .finish_non_exhaustive()
    }
}

struct JobOwnershipToken {
    core: Arc<JobCore>,
}

impl Drop for JobOwnershipToken {
    fn drop(&mut self) {
        self.core.request_cancel(true);
    }
}

impl ContinuousJob {
    pub(crate) fn id(&self) -> u64 {
        self.core.job_id
    }

    pub(crate) fn status(&self) -> ContinuousJobStatus {
        let (state, terminal_cause) = {
            let state = self.core.state.lock();
            (state.state, state.selected_cause.clone())
        };
        let metrics = self.core.metrics.snapshot();
        let runtime = self.core.runtime_status.lock();
        let tasks = runtime.tasks.snapshot();
        let sources = runtime
            .sources
            .iter()
            .map(|(id, progress)| (id.clone(), progress.snapshot().into()))
            .collect();
        let nodes = runtime
            .nodes
            .iter()
            .map(|(id, progress)| (id.clone(), progress.snapshot().into()))
            .collect();
        let sinks = metrics
            .sinks
            .iter()
            .map(|(metric_id, sink_metrics)| {
                let ended = runtime
                    .sink_outputs
                    .get(metric_id)
                    .and_then(|output_id| runtime.sinks.get(output_id))
                    .is_some_and(|progress| progress.snapshot().ended);
                (
                    metric_id.clone(),
                    SinkStatus {
                        delivered_batches: sink_metrics.delivered_batches,
                        delivered_rows: sink_metrics.delivered_rows,
                        delivered_bytes: sink_metrics.delivered_bytes,
                        ended,
                    },
                )
            })
            .collect();
        let edges = metrics
            .edges
            .iter()
            .map(|(id, edge)| (id.clone(), edge.channel.clone()))
            .collect();
        ContinuousJobStatus {
            job_id: self.core.job_id,
            state,
            terminal_cause,
            tasks,
            edges,
            sources,
            nodes,
            sinks,
            metrics,
        }
    }

    pub(crate) fn state(&self) -> ContinuousJobState {
        self.core.state.lock().state
    }

    pub(crate) fn driver_owner(&self) -> DriverOwnership {
        self.core.state.lock().owner
    }

    pub(crate) fn wait(&self) -> OutcomeObserver {
        OutcomeObserver::new(Arc::clone(&self.core))
    }

    pub(crate) fn shutdown(&self) -> OutcomeObserver {
        self.core.request_shutdown();
        OutcomeObserver::new(Arc::clone(&self.core))
    }

    pub(crate) fn cancel(&self) -> OutcomeObserver {
        self.core.request_cancel(false);
        OutcomeObserver::new(Arc::clone(&self.core))
    }
}

pub(crate) struct OutcomeObserver {
    inner: Pin<Box<dyn Future<Output = Arc<ContinuousJobOutcome>> + Send>>,
}

impl OutcomeObserver {
    fn new(core: Arc<JobCore>) -> Self {
        Self {
            inner: Box::pin(async move {
                loop {
                    let notified = core.changed.notified();
                    if let Some(outcome) = core.state.lock().outcome.clone() {
                        return outcome;
                    }
                    notified.await;
                }
            }),
        }
    }
}

impl Future for OutcomeObserver {
    type Output = Arc<ContinuousJobOutcome>;

    fn poll(mut self: Pin<&mut Self>, context: &mut Context<'_>) -> Poll<Self::Output> {
        self.inner.as_mut().poll(context)
    }
}

pub(crate) struct StartObserver {
    inner: Pin<Box<dyn Future<Output = StartResult<ContinuousJob>> + Send>>,
    core: Option<Arc<JobCore>>,
    delivered: Arc<AtomicBool>,
}

impl StartObserver {
    fn ready(result: StartResult<ContinuousJob>) -> Self {
        Self {
            inner: Box::pin(std::future::ready(result)),
            core: None,
            delivered: Arc::new(AtomicBool::new(true)),
        }
    }

    fn observe(core: Arc<JobCore>) -> Self {
        let delivered = Arc::new(AtomicBool::new(false));
        let delivered_in_future = Arc::clone(&delivered);
        let observed_core = Arc::clone(&core);
        let inner = Box::pin(async move {
            loop {
                let notified = observed_core.changed.notified();
                let action = {
                    let mut state = observed_core.state.lock();
                    match state.launch_delivery {
                        LaunchDeliveryState::ReadyUnclaimed => {
                            state.launch_delivery = LaunchDeliveryState::Claimed;
                            state.owner = DriverOwnership::Driving;
                            Some(Ok(()))
                        }
                        LaunchDeliveryState::Failed => Some(Err(state
                            .start_failure
                            .clone()
                            .expect("failed launch has error"))),
                        LaunchDeliveryState::CancelRequested
                            if state.owner == DriverOwnership::Terminal =>
                        {
                            Some(Err(cancelled_start_failure(observed_core.job_id)))
                        }
                        _ => None,
                    }
                };
                match action {
                    Some(Ok(())) => {
                        delivered_in_future.store(true, Ordering::Release);
                        observed_core.changed.notify_waiters();
                        return Ok(ContinuousJob {
                            core: Arc::clone(&observed_core),
                            _ownership: JobOwnershipToken {
                                core: Arc::clone(&observed_core),
                            },
                        });
                    }
                    Some(Err(error)) => return Err(error),
                    None => notified.await,
                }
            }
        });
        Self {
            inner,
            core: Some(core),
            delivered,
        }
    }
}

impl Future for StartObserver {
    type Output = StartResult<ContinuousJob>;

    fn poll(mut self: Pin<&mut Self>, context: &mut Context<'_>) -> Poll<Self::Output> {
        self.inner.as_mut().poll(context)
    }
}

impl Drop for StartObserver {
    fn drop(&mut self) {
        if !self.delivered.load(Ordering::Acquire)
            && let Some(core) = &self.core
        {
            core.request_cancel(true);
        }
    }
}

fn cancelled_start_failure(job_id: u64) -> StartFailure {
    StartFailure {
        primary: Arc::new(RuntimeFailure {
            origin: FailureOrigin::Preflight,
            error: CalcFlowError::Cancelled {
                run_id: job_id.to_string(),
            },
        }),
        diagnostic_id: None,
    }
}

#[derive(Clone, Debug)]
pub(crate) struct RunnerDiagnosticRecord {
    pub(crate) id: u64,
    pub(crate) launch_id: LaunchId,
    pub(crate) cleanup_failures: Vec<Arc<RuntimeFailure>>,
    pub(crate) failures_truncated: bool,
}

#[derive(Clone, Debug)]
pub(crate) struct RunnerDiagnosticsSnapshot {
    pub(crate) records: Vec<Arc<RunnerDiagnosticRecord>>,
    pub(crate) truncated_records: u64,
    pub(crate) diagnostics_overflowed: bool,
}

#[derive(Default)]
struct RunnerDiagnosticsState {
    records: VecDeque<Arc<RunnerDiagnosticRecord>>,
    next_id: u64,
    truncated_records: u64,
    diagnostics_overflowed: bool,
}

#[derive(Default)]
struct RunnerDiagnostics(Mutex<RunnerDiagnosticsState>);

impl RunnerDiagnostics {
    fn record(&self, launch_id: LaunchId, mut failures: Vec<Arc<RuntimeFailure>>) -> Option<u64> {
        if failures.is_empty() {
            return None;
        }
        failures.sort_by(|left, right| left.origin.cmp(&right.origin));
        let failures_truncated = failures.len() > 64;
        failures.truncate(64);
        let mut state = self.0.lock();
        let id = state.next_id;
        match state.next_id.checked_add(1) {
            Some(next) => state.next_id = next,
            None => state.diagnostics_overflowed = true,
        }
        if state.records.len() == 64 {
            state.records.pop_front();
            state.truncated_records = state.truncated_records.saturating_add(1);
        }
        state.records.push_back(Arc::new(RunnerDiagnosticRecord {
            id,
            launch_id,
            cleanup_failures: failures,
            failures_truncated,
        }));
        Some(id)
    }

    fn snapshot(&self) -> RunnerDiagnosticsSnapshot {
        let state = self.0.lock();
        RunnerDiagnosticsSnapshot {
            records: state.records.iter().cloned().collect(),
            truncated_records: state.truncated_records,
            diagnostics_overflowed: state.diagnostics_overflowed,
        }
    }
}

struct RunnerRegistryState {
    provisional: Option<LaunchId>,
    live_jobs: BTreeMap<LaunchId, Arc<JobCore>>,
    reaper_jobs: BTreeSet<LaunchId>,
    pending_start: Option<LaunchId>,
    shutting_down: bool,
}

struct RunnerCore {
    commands: mpsc::UnboundedSender<RunnerCommand>,
    root_cancel: CancellationToken,
    registry: Mutex<RunnerRegistryState>,
    driver: Mutex<Option<JoinHandle<()>>>,
    diagnostics: RunnerDiagnostics,
    next_launch_id: AtomicU64,
    closed: AtomicBool,
    changed: Notify,
    #[cfg(test)]
    abandonment_warnings: AtomicU64,
}

const ABANDONED_RUNNER_WARNING: &str =
    "continuous runner dropped before shutdown completed; cancellation requested";

enum RunnerCommand {
    Start {
        launch_id: LaunchId,
        core: Arc<JobCore>,
        job: Box<ValidatedContinuousJob>,
    },
    Wake(LaunchId),
    Shutdown,
}

pub(crate) struct ContinuousRunner {
    core: Arc<RunnerCore>,
}

impl ContinuousRunner {
    pub(crate) fn new() -> Self {
        let (commands, receiver) = mpsc::unbounded_channel();
        let core = Arc::new(RunnerCore {
            commands,
            root_cancel: CancellationToken::new(),
            registry: Mutex::new(RunnerRegistryState {
                provisional: None,
                live_jobs: BTreeMap::new(),
                reaper_jobs: BTreeSet::new(),
                pending_start: None,
                shutting_down: false,
            }),
            driver: Mutex::new(None),
            diagnostics: RunnerDiagnostics::default(),
            next_launch_id: AtomicU64::new(0),
            closed: AtomicBool::new(false),
            changed: Notify::new(),
            #[cfg(test)]
            abandonment_warnings: AtomicU64::new(0),
        });
        let driver = tokio::spawn(runner_lifecycle(Arc::clone(&core), receiver));
        *core.driver.lock() = Some(driver);
        Self { core }
    }

    pub(crate) fn diagnostics(&self) -> RunnerDiagnosticsSnapshot {
        self.core.diagnostics.snapshot()
    }

    #[cfg(test)]
    pub(crate) fn registry_counts(&self) -> (usize, usize) {
        let registry = self.core.registry.lock();
        (registry.live_jobs.len(), registry.reaper_jobs.len())
    }

    pub(crate) fn start(&self, spec: ContinuousJobSpec) -> StartObserver {
        let validated = match preflight_job(spec) {
            Ok(validated) => validated,
            Err(error) => {
                return StartObserver::ready(Err(StartFailure {
                    primary: Arc::new(RuntimeFailure {
                        origin: FailureOrigin::Preflight,
                        error,
                    }),
                    diagnostic_id: None,
                }));
            }
        };
        let Ok(launch_id) =
            self.core
                .next_launch_id
                .fetch_update(Ordering::AcqRel, Ordering::Acquire, |value| {
                    value.checked_add(1)
                })
        else {
            return StartObserver::ready(Err(StartFailure {
                primary: Arc::new(RuntimeFailure {
                    origin: FailureOrigin::Preflight,
                    error: CalcFlowError::Internal {
                        message: "streaming launch ID space is exhausted".into(),
                    },
                }),
                diagnostic_id: None,
            }));
        };
        let launch_id = LaunchId(launch_id);
        let job_id = validated.context.job_id();
        let (metrics, sink_outputs) = metrics_for_job(&validated);
        let core = Arc::new(JobCore::new(
            launch_id,
            job_id,
            self.core.commands.clone(),
            metrics,
            sink_outputs,
        ));
        {
            let mut registry = self.core.registry.lock();
            if registry.shutting_down {
                return conflict_observer("runner is shutting down");
            }
            let has_non_reaper_live = registry
                .live_jobs
                .values()
                .any(|job| job.state.lock().owner != DriverOwnership::ReaperOwned);
            let has_non_reaper_provisional = registry.provisional.is_some_and(|launch_id| {
                registry
                    .live_jobs
                    .get(&launch_id)
                    .is_some_and(|job| job.state.lock().owner != DriverOwnership::ReaperOwned)
            });
            if has_non_reaper_provisional || has_non_reaper_live || registry.pending_start.is_some()
            {
                return conflict_observer("active");
            }
            if registry.live_jobs.is_empty() && registry.reaper_jobs.is_empty() {
                registry.provisional = Some(launch_id);
            } else {
                registry.pending_start = Some(launch_id);
            }
            registry.live_jobs.insert(launch_id, Arc::clone(&core));
        }
        if self
            .core
            .commands
            .send(RunnerCommand::Start {
                launch_id,
                core: Arc::clone(&core),
                job: Box::new(validated),
            })
            .is_err()
        {
            return StartObserver::ready(Err(StartFailure {
                primary: Arc::new(RuntimeFailure {
                    origin: FailureOrigin::Preflight,
                    error: CalcFlowError::Internal {
                        message: "streaming runner lifecycle driver is unavailable".into(),
                    },
                }),
                diagnostic_id: None,
            }));
        }
        StartObserver::observe(core)
    }

    pub(crate) fn shutdown(&mut self) -> RunnerShutdownObserver {
        {
            let mut registry = self.core.registry.lock();
            if !registry.shutting_down {
                registry.shutting_down = true;
                let _ = self.core.commands.send(RunnerCommand::Shutdown);
            }
        }
        RunnerShutdownObserver::new(Arc::clone(&self.core))
    }
}

fn metrics_for_job(job: &ValidatedContinuousJob) -> (MetricsRecorder, BTreeMap<String, String>) {
    let sink_outputs = job
        .sinks
        .iter()
        .flat_map(|(output_id, sinks)| {
            sinks
                .iter()
                .map(move |sink| (sink_metric_id(output_id, &sink.sink_id), output_id.clone()))
        })
        .collect::<BTreeMap<_, _>>();
    let metrics = MetricsRecorder::new(
        job.plan
            .edges
            .iter()
            .map(|(edge_id, edge)| (edge_id.clone(), edge.budget)),
        job.plan.source_routes.keys().cloned(),
        job.plan.nodes.iter().map(|node| node.node_id.clone()),
        sink_outputs.keys().cloned(),
    );
    (metrics, sink_outputs)
}

impl Drop for ContinuousRunner {
    fn drop(&mut self) {
        if !self.core.closed.load(Ordering::Acquire) {
            warn_abandoned_runner_drop(&self.core);
            let mut registry = self.core.registry.lock();
            self.core.root_cancel.cancel();
            registry.shutting_down = true;
            let live_jobs = registry
                .live_jobs
                .iter()
                .map(|(launch_id, job)| (*launch_id, Arc::clone(job)))
                .collect::<Vec<_>>();
            for (launch_id, job) in live_jobs {
                if job.request_runner_drop_cancel() {
                    registry.reaper_jobs.insert(launch_id);
                }
            }
            drop(registry);
            let _ = self.core.commands.send(RunnerCommand::Shutdown);
        }
    }
}

fn warn_abandoned_runner_drop(core: &RunnerCore) {
    tracing::warn!(
        target: "calc_flow::runtime::streaming",
        event = "abandoned_runner_drop",
        "continuous runner dropped before shutdown completed; cancellation requested"
    );
    #[cfg(test)]
    core.abandonment_warnings.fetch_add(1, Ordering::SeqCst);
    #[cfg(not(test))]
    let _ = core;
}

fn conflict_observer(key: &str) -> StartObserver {
    StartObserver::ready(Err(StartFailure {
        primary: Arc::new(RuntimeFailure {
            origin: FailureOrigin::Preflight,
            error: CalcFlowError::Conflict {
                resource: "streaming job".into(),
                key: key.into(),
            },
        }),
        diagnostic_id: None,
    }))
}

pub(crate) struct RunnerShutdownObserver {
    core: Arc<RunnerCore>,
    closed: Pin<Box<dyn Future<Output = ()> + Send>>,
    closed_observed: bool,
}

impl RunnerShutdownObserver {
    fn new(core: Arc<RunnerCore>) -> Self {
        let observed_core = Arc::clone(&core);
        Self {
            core,
            closed: Box::pin(async move {
                loop {
                    let notified = observed_core.changed.notified();
                    if observed_core.closed.load(Ordering::Acquire) {
                        return;
                    }
                    notified.await;
                }
            }),
            closed_observed: false,
        }
    }
}

impl Future for RunnerShutdownObserver {
    type Output = crate::Result<()>;

    fn poll(self: Pin<&mut Self>, context: &mut Context<'_>) -> Poll<Self::Output> {
        let this = self.get_mut();
        if !this.closed_observed {
            if this.closed.as_mut().poll(context).is_pending() {
                return Poll::Pending;
            }
            this.closed_observed = true;
        }
        let mut driver = this.core.driver.lock();
        let Some(lifecycle) = driver.as_mut() else {
            return Poll::Ready(Ok(()));
        };
        match Pin::new(lifecycle).poll(context) {
            Poll::Pending => Poll::Pending,
            Poll::Ready(joined) => {
                driver.take();
                Poll::Ready(joined.map_err(|error| CalcFlowError::Internal {
                    message: format!("runner lifecycle driver join failed: {error}"),
                }))
            }
        }
    }
}

struct PendingStart {
    launch_id: LaunchId,
    core: Arc<JobCore>,
    job: Box<ValidatedContinuousJob>,
}

async fn runner_lifecycle(
    runner_core: Arc<RunnerCore>,
    mut commands: mpsc::UnboundedReceiver<RunnerCommand>,
) {
    let mut drivers = JoinSet::new();
    let mut active: Option<LaunchId> = None;
    let mut pending: Option<PendingStart> = None;
    let mut shutting_down = false;
    loop {
        if shutting_down
            && active.is_none()
            && pending.is_none()
            && runner_core.registry.lock().live_jobs.is_empty()
        {
            break;
        }
        tokio::select! {
            () = runner_core.root_cancel.cancelled(), if !shutting_down => {
                shutting_down = true;
                begin_lifecycle_shutdown(&runner_core, &mut pending);
            }
            command = commands.recv() => match command {
                Some(RunnerCommand::Start { launch_id, core: job_core, job }) if active.is_none() && !shutting_down => {
                    active = Some(launch_id);
                    job_core.state.lock().owner = DriverOwnership::Driving;
                    drivers.spawn(run_job_driver(launch_id, Arc::clone(&job_core), *job));
                }
                Some(RunnerCommand::Start { launch_id, core: job_core, job }) if !shutting_down => {
                    pending = Some(PendingStart { launch_id, core: job_core, job });
                }
                Some(RunnerCommand::Start { core: job_core, .. }) => {
                    job_core.request_cancel(true);
                    publish_abandoned_without_driver(&runner_core, &job_core);
                }
                Some(RunnerCommand::Wake(launch_id)) => {
                    let mut registry = runner_core.registry.lock();
                    if registry
                        .live_jobs
                        .get(&launch_id)
                        .is_some_and(|job| job.state.lock().owner == DriverOwnership::ReaperOwned)
                    {
                        registry.reaper_jobs.insert(launch_id);
                    }
                }
                Some(RunnerCommand::Shutdown) | None => {
                    shutting_down = true;
                    begin_lifecycle_shutdown(&runner_core, &mut pending);
                }
            },
            joined = drivers.join_next(), if active.is_some() => {
                if let Some(joined) = joined {
                    let report = match joined {
                        Ok(report) => report,
                        Err(error) => DriverReport::aborted(
                            active.expect("an active driver owns the join"),
                            &error.to_string(),
                        ),
                    };
                    publish_driver_report(&runner_core, report);
                    active = None;
                    if !shutting_down && let Some(next) = pending.take() {
                        active = Some(next.launch_id);
                        next.core.state.lock().owner = DriverOwnership::Driving;
                        drivers.spawn(run_job_driver(next.launch_id, Arc::clone(&next.core), *next.job));
                    }
                }
            }
        }
    }
    runner_core.closed.store(true, Ordering::Release);
    runner_core.changed.notify_waiters();
}

fn begin_lifecycle_shutdown(runner: &RunnerCore, pending: &mut Option<PendingStart>) {
    let jobs = {
        let mut registry = runner.registry.lock();
        registry.shutting_down = true;
        registry.live_jobs.values().cloned().collect::<Vec<_>>()
    };
    for job in jobs {
        job.request_cancel(true);
    }
    if let Some(pending_start) = pending.take() {
        publish_abandoned_without_driver(runner, &pending_start.core);
    }
}

fn publish_abandoned_without_driver(runner: &RunnerCore, job: &Arc<JobCore>) {
    let mut registry = runner.registry.lock();
    if !registry.live_jobs.contains_key(&job.launch_id) {
        return;
    }
    let mut state = job.state.lock();
    if state.owner != DriverOwnership::Terminal {
        let mut errors = Vec::new();
        if state.owner == DriverOwnership::ReaperOwned
            && let Err(error) = job.metrics.record_reaper_join()
        {
            errors.push(Arc::new(RuntimeFailure {
                origin: FailureOrigin::Metrics {
                    component_id: "job".into(),
                    counter: "reaper_joins",
                },
                error,
            }));
        }
        state.owner = DriverOwnership::Terminal;
        state.launch_delivery = LaunchDeliveryState::CancelRequested;
        state.state = ContinuousJobState::Cancelled;
        state.selected_cause = Some(TerminalCause::ExplicitCancel);
        state.outcome = Some(Arc::new(ContinuousJobOutcome {
            state: ContinuousJobState::Cancelled,
            cause: TerminalCause::ExplicitCancel,
            errors,
        }));
    }
    registry.live_jobs.remove(&job.launch_id);
    registry.reaper_jobs.remove(&job.launch_id);
    if registry.provisional == Some(job.launch_id) {
        registry.provisional = None;
    }
    if registry.pending_start == Some(job.launch_id) {
        registry.pending_start = None;
    }
    drop(state);
    drop(registry);
    job.changed.notify_waiters();
}

enum DriverCompletion {
    StartFailed(StartFailure),
    Outcome(Arc<ContinuousJobOutcome>),
}

struct DriverReport {
    launch_id: LaunchId,
    completion: DriverCompletion,
    cleanup_failures: Vec<Arc<RuntimeFailure>>,
}

impl DriverReport {
    fn aborted(launch_id: LaunchId, message: &str) -> Self {
        let failure = Arc::new(RuntimeFailure {
            origin: FailureOrigin::Preflight,
            error: CalcFlowError::Internal {
                message: format!("job driver join failed: {message}"),
            },
        });
        Self {
            launch_id,
            completion: DriverCompletion::StartFailed(StartFailure {
                primary: failure,
                diagnostic_id: None,
            }),
            cleanup_failures: Vec::new(),
        }
    }
}

fn publish_driver_report(runner: &RunnerCore, mut report: DriverReport) {
    let mut registry = runner.registry.lock();
    let job = registry.live_jobs.get(&report.launch_id).cloned();
    let Some(job) = job else {
        return;
    };
    let diagnostic_id = runner
        .diagnostics
        .record(report.launch_id, report.cleanup_failures);
    let mut state = job.state.lock();
    let was_reaper_owned = state.owner == DriverOwnership::ReaperOwned;
    if was_reaper_owned
        && let Err(error) = job.metrics.record_reaper_join()
        && let DriverCompletion::Outcome(outcome) = &mut report.completion
    {
        Arc::make_mut(outcome).errors.push(Arc::new(RuntimeFailure {
            origin: FailureOrigin::Metrics {
                component_id: "job".into(),
                counter: "reaper_joins",
            },
            error,
        }));
    }
    state.owner = DriverOwnership::Terminal;
    match &mut report.completion {
        DriverCompletion::StartFailed(failure) => {
            failure.diagnostic_id = diagnostic_id;
            state.launch_delivery = LaunchDeliveryState::Failed;
            state.state = ContinuousJobState::Failed;
            state.start_failure = Some(failure.clone());
        }
        DriverCompletion::Outcome(outcome) => {
            state.state = outcome.state;
            state.selected_cause = Some(outcome.cause.clone());
            state.outcome = Some(Arc::clone(outcome));
        }
    }
    registry.live_jobs.remove(&report.launch_id);
    registry.reaper_jobs.remove(&report.launch_id);
    if registry.provisional == Some(report.launch_id) {
        registry.provisional = None;
    }
    if registry.pending_start == Some(report.launch_id) {
        registry.pending_start = None;
    }
    drop(state);
    drop(registry);
    job.changed.notify_waiters();
}

enum ConnectorResource {
    Source {
        binding_id: String,
        binding: SourceBinding,
    },
    Sink {
        output_id: String,
        sink_id: String,
        configured_index: usize,
        binding: OrdinarySinkBinding,
    },
}

impl ConnectorResource {
    fn open_origin(&self) -> FailureOrigin {
        match self {
            Self::Source { binding_id, .. } => FailureOrigin::SourceOpen {
                binding_id: binding_id.clone(),
            },
            Self::Sink {
                output_id, sink_id, ..
            } => FailureOrigin::SinkOpen {
                output_id: output_id.clone(),
                sink_id: sink_id.clone(),
            },
        }
    }

    fn close_origin(&self) -> FailureOrigin {
        match self {
            Self::Source { binding_id, .. } => FailureOrigin::SourceClose {
                binding_id: binding_id.clone(),
            },
            Self::Sink {
                output_id, sink_id, ..
            } => FailureOrigin::SinkClose {
                output_id: output_id.clone(),
                sink_id: sink_id.clone(),
            },
        }
    }

    async fn open(&mut self) -> crate::Result<()> {
        match self {
            Self::Source { binding, .. } => binding.open().await,
            Self::Sink { binding, .. } => binding.sink.open().await,
        }
    }

    async fn close(&mut self) -> crate::Result<()> {
        match self {
            Self::Source { binding, .. } => binding.close().await,
            Self::Sink { binding, .. } => binding.sink.close().await,
        }
    }
}

enum OpenResult {
    Opened,
    Failed(CalcFlowError),
    Cancelled,
}

struct OpenExit {
    origin: FailureOrigin,
    resource: ConnectorResource,
    result: OpenResult,
}

async fn run_job_driver(
    launch_id: LaunchId,
    core: Arc<JobCore>,
    validated: ValidatedContinuousJob,
) -> DriverReport {
    let ValidatedContinuousJob {
        context,
        plan,
        sources,
        sinks,
        delivery_mode: _,
    } = validated;
    let cancellation = context.cancellation().clone();
    let entry = run_operator_entry(plan, &context, &core, &cancellation).await;
    let mut runtime = match entry {
        Ok(entry) => entry,
        Err(EntryFailure::Failed(primary)) => {
            return DriverReport {
                launch_id,
                completion: DriverCompletion::StartFailed(StartFailure {
                    primary,
                    diagnostic_id: None,
                }),
                cleanup_failures: Vec::new(),
            };
        }
        Err(EntryFailure::Cancelled) => {
            return cancelled_driver_report(launch_id, &core.metrics);
        }
    };
    let mut resources = connector_resources(sources, sinks);
    let open_failures = open_connector_resources(&mut resources, &core.launch_cancel).await;
    if !open_failures.is_empty() {
        return finish_failed_launch(
            launch_id,
            open_failures,
            &mut resources,
            &mut runtime.supervisor,
        )
        .await;
    }
    if core.launch_cancel.is_cancelled() {
        return finish_cancelled_launch(
            launch_id,
            &mut resources,
            &mut runtime.supervisor,
            &core.metrics,
        )
        .await;
    }

    let (opened_sources, opened_sinks) = opened_connector_bindings(std::mem::take(&mut resources));
    let task_progress =
        register_boundary_tasks(&mut runtime, &context, opened_sources, opened_sinks, &core);

    core.state.lock().launch_delivery = LaunchDeliveryState::ReadyUnclaimed;
    core.changed.notify_waiters();
    if !await_handle_claim(&core).await {
        runtime.supervisor.cancel();
        let report = runtime.supervisor.join_all().await;
        return cancelled_driver_report_with_task_cleanup(
            launch_id,
            report,
            &task_progress,
            &core.metrics,
        );
    }
    if runtime.data_gate.send(true).is_err() {
        cancellation.cancel();
    }
    drive_running_job(
        launch_id,
        &core,
        context.deadline().copied(),
        cancellation,
        task_progress,
        &mut runtime.supervisor,
        &core.metrics,
    )
    .await
}

enum EntryFailure {
    Failed(Arc<RuntimeFailure>),
    Cancelled,
}

struct RegisteredRuntime {
    supervisor: TaskSupervisor,
    data_gate: watch::Sender<bool>,
    source_outputs: BTreeMap<String, Vec<EdgeSender>>,
    sink_inputs: BTreeMap<String, EdgeReceiver>,
}

struct RuntimeTaskProgress {
    sources: BTreeMap<String, SourceProgress>,
    sinks: BTreeMap<String, SinkProgress>,
}

fn create_runtime_channels(
    plan: &StreamRuntimePlanParts,
    metrics: &MetricsRecorder,
) -> crate::Result<(BTreeMap<String, EdgeSender>, BTreeMap<String, EdgeReceiver>)> {
    let mut senders = BTreeMap::new();
    let mut receivers = BTreeMap::new();
    for edge in plan.edges.values() {
        let (sender, receiver) =
            edge_channel_with_metrics(edge.stable_id.clone(), edge.budget, metrics.clone())?;
        senders.insert(edge.stable_id.clone(), sender);
        receivers.insert(edge.stable_id.clone(), receiver);
    }
    Ok((senders, receivers))
}

fn take_node_ingresses(
    node: &RuntimeStreamNode,
    receivers: &mut BTreeMap<String, EdgeReceiver>,
) -> crate::Result<BTreeMap<String, OperatorIngress>> {
    node.ingress_edges
        .iter()
        .map(|(ingress, edge_id)| {
            let receiver = receivers
                .remove(edge_id)
                .ok_or_else(|| CalcFlowError::Internal {
                    message: format!(
                        "operator {:?} ingress {:?} lost edge {:?}",
                        node.node_id, ingress, edge_id
                    ),
                })?;
            Ok((
                ingress.clone(),
                OperatorIngress::new(edge_id.clone(), receiver),
            ))
        })
        .collect()
}

fn take_node_outputs(
    node: &RuntimeStreamNode,
    senders: &mut BTreeMap<String, EdgeSender>,
) -> crate::Result<BTreeMap<String, Vec<EdgeSender>>> {
    node.output_edges
        .iter()
        .map(|(port, edge_ids)| {
            let outputs = edge_ids
                .iter()
                .map(|edge_id| {
                    senders
                        .remove(edge_id)
                        .ok_or_else(|| CalcFlowError::Internal {
                            message: format!(
                                "operator {:?} output {:?} lost edge {:?}",
                                node.node_id, port, edge_id
                            ),
                        })
                })
                .collect::<crate::Result<Vec<_>>>()?;
            Ok((port.clone(), outputs))
        })
        .collect()
}

async fn run_operator_entry(
    plan: StreamRuntimePlanParts,
    context: &super::StreamJobContext,
    core: &Arc<JobCore>,
    cancellation: &CancellationToken,
) -> Result<RegisteredRuntime, EntryFailure> {
    let mut supervisor = TaskSupervisor::new_with_terminal_arbiter(
        cancellation.clone(),
        core.terminal_arbiter.clone(),
    );
    core.runtime_status.lock().tasks = supervisor.registry();
    let (entry_tx, _) = watch::channel(false);
    let (data_tx, _) = watch::channel(false);
    let (ack_tx, mut ack_rx) = mpsc::unbounded_channel();
    let (mut senders, mut receivers) =
        create_runtime_channels(&plan, &core.metrics).map_err(preflight_entry_failure)?;
    let node_count = plan.nodes.len();
    let registration = &mut OperatorRegistration {
        context,
        core,
        entry_tx: &entry_tx,
        data_tx: &data_tx,
        ack_tx: &ack_tx,
        senders: &mut senders,
        receivers: &mut receivers,
        supervisor: &mut supervisor,
        metrics: &core.metrics,
        runtime_status: &core.runtime_status,
    };
    if let Err(failure) = register_operator_nodes(plan.nodes, registration) {
        supervisor.cancel();
        let _ = supervisor.join_all().await;
        return Err(failure);
    }
    drop(ack_tx);
    let _ = entry_tx.send(true);
    await_operator_entry(node_count, core, &mut ack_rx, &mut supervisor).await?;
    let endpoints = take_boundary_endpoints(
        plan.source_routes,
        plan.sink_routes,
        &mut senders,
        &mut receivers,
    );
    let (source_outputs, sink_inputs) = match endpoints {
        Ok(endpoints) if senders.is_empty() && receivers.is_empty() => endpoints,
        Ok(_) => {
            return fail_registered_entry(
                &mut supervisor,
                preflight_entry_failure(CalcFlowError::Internal {
                    message: "runtime topology left unowned channel endpoints".into(),
                }),
            )
            .await;
        }
        Err(failure) => return fail_registered_entry(&mut supervisor, failure).await,
    };
    Ok(RegisteredRuntime {
        supervisor,
        data_gate: data_tx,
        source_outputs,
        sink_inputs,
    })
}

struct OperatorRegistration<'a> {
    context: &'a super::StreamJobContext,
    core: &'a Arc<JobCore>,
    entry_tx: &'a watch::Sender<bool>,
    data_tx: &'a watch::Sender<bool>,
    ack_tx: &'a mpsc::UnboundedSender<super::operator_task::OperatorEntryAck>,
    senders: &'a mut BTreeMap<String, EdgeSender>,
    receivers: &'a mut BTreeMap<String, EdgeReceiver>,
    supervisor: &'a mut TaskSupervisor,
    metrics: &'a MetricsRecorder,
    runtime_status: &'a Mutex<RuntimeStatus>,
}

fn register_operator_nodes(
    nodes: Vec<RuntimeStreamNode>,
    registration: &mut OperatorRegistration<'_>,
) -> Result<(), EntryFailure> {
    for node in nodes {
        let node_id = node.node_id.clone();
        let progress = OperatorProgress::default();
        let ingresses =
            take_node_ingresses(&node, registration.receivers).map_err(preflight_entry_failure)?;
        let outputs =
            take_node_outputs(&node, registration.senders).map_err(preflight_entry_failure)?;
        let task_context = registration
            .context
            .for_node(&node_id)
            .map_err(preflight_entry_failure)?;
        spawn_operator_task(
            registration.supervisor,
            OperatorTaskInputs {
                node_id: node_id.clone(),
                operator: node.operator,
                ingresses,
                outputs,
                output_ports: node.output_ports,
                context: task_context,
                progress: progress.clone(),
                metrics: registration.metrics.clone(),
                entry_gate: registration.entry_tx.subscribe(),
                entry_ack: registration.ack_tx.clone(),
                data_gate: registration.data_tx.subscribe(),
                launch_cancel: registration.core.launch_cancel.clone(),
            },
        );
        registration
            .runtime_status
            .lock()
            .nodes
            .insert(node_id, progress);
    }
    Ok(())
}

async fn await_operator_entry(
    node_count: usize,
    core: &Arc<JobCore>,
    ack_rx: &mut mpsc::UnboundedReceiver<super::operator_task::OperatorEntryAck>,
    supervisor: &mut TaskSupervisor,
) -> Result<(), EntryFailure> {
    let mut acknowledgements = Vec::with_capacity(node_count);
    for _ in 0..node_count {
        let acknowledgement = tokio::select! {
            biased;
            () = core.launch_cancel.cancelled() => None,
            acknowledgement = ack_rx.recv() => acknowledgement,
        };
        let Some(acknowledgement) = acknowledgement else {
            break;
        };
        acknowledgements.push(acknowledgement);
    }
    let entry_failure =
        if acknowledgements.len() != node_count && !core.launch_cancel.is_cancelled() {
            Some(Arc::new(RuntimeFailure {
                origin: FailureOrigin::Preflight,
                error: CalcFlowError::Internal {
                    message: "operator entry acknowledgement channel closed early".into(),
                },
            }))
        } else {
            acknowledgements.sort_by(|left, right| left.node_id.cmp(&right.node_id));
            acknowledgements.into_iter().find_map(|acknowledgement| {
                acknowledgement.result.err().map(|error| {
                    Arc::new(RuntimeFailure {
                        origin: FailureOrigin::OperatorEntry {
                            node_id: acknowledgement.node_id,
                        },
                        error,
                    })
                })
            })
        };
    if let Some(primary) = entry_failure {
        return fail_registered_entry(supervisor, EntryFailure::Failed(primary)).await;
    }
    if core.launch_cancel.is_cancelled() {
        return fail_registered_entry(supervisor, EntryFailure::Cancelled).await;
    }
    Ok(())
}

type BoundaryEndpoints = (
    BTreeMap<String, Vec<EdgeSender>>,
    BTreeMap<String, EdgeReceiver>,
);

fn take_boundary_endpoints(
    source_routes: BTreeMap<String, RuntimeSourceRoute>,
    sink_routes: BTreeMap<String, RuntimeSinkRoute>,
    senders: &mut BTreeMap<String, EdgeSender>,
    receivers: &mut BTreeMap<String, EdgeReceiver>,
) -> Result<BoundaryEndpoints, EntryFailure> {
    let source_outputs = source_routes
        .into_iter()
        .map(|(binding_id, route)| {
            let sender = senders.remove(&route.edge_id).ok_or_else(|| {
                EntryFailure::Failed(Arc::new(RuntimeFailure {
                    origin: FailureOrigin::Preflight,
                    error: CalcFlowError::Internal {
                        message: format!(
                            "source route {:?} lost edge {:?}",
                            binding_id, route.edge_id
                        ),
                    },
                }))
            })?;
            Ok((binding_id, vec![sender]))
        })
        .collect::<Result<BTreeMap<_, _>, EntryFailure>>()?;
    let sink_inputs = sink_routes
        .into_iter()
        .map(|(output_id, route)| {
            let receiver = receivers.remove(&route.edge_id).ok_or_else(|| {
                EntryFailure::Failed(Arc::new(RuntimeFailure {
                    origin: FailureOrigin::Preflight,
                    error: CalcFlowError::Internal {
                        message: format!(
                            "sink route {:?} lost edge {:?}",
                            output_id, route.edge_id
                        ),
                    },
                }))
            })?;
            Ok((output_id, receiver))
        })
        .collect::<Result<BTreeMap<_, _>, EntryFailure>>()?;
    Ok((source_outputs, sink_inputs))
}

fn preflight_entry_failure(error: CalcFlowError) -> EntryFailure {
    EntryFailure::Failed(Arc::new(RuntimeFailure {
        origin: FailureOrigin::Preflight,
        error,
    }))
}

async fn fail_registered_entry<T>(
    supervisor: &mut TaskSupervisor,
    failure: EntryFailure,
) -> Result<T, EntryFailure> {
    supervisor.cancel();
    let _ = supervisor.join_all().await;
    Err(failure)
}

fn connector_resources(
    sources: BTreeMap<String, SourceBinding>,
    sinks: BTreeMap<String, Vec<ValidatedOrdinarySink>>,
) -> Vec<ConnectorResource> {
    let mut resources = Vec::new();
    for (binding_id, binding) in sources {
        resources.push(ConnectorResource::Source {
            binding_id,
            binding,
        });
    }
    for (output_id, bindings) in sinks {
        for (configured_index, sink) in bindings.into_iter().enumerate() {
            resources.push(ConnectorResource::Sink {
                output_id: output_id.clone(),
                sink_id: sink.sink_id,
                configured_index,
                binding: sink.binding,
            });
        }
    }
    resources
}

fn opened_connector_bindings(
    resources: Vec<ConnectorResource>,
) -> (
    BTreeMap<String, SourceBinding>,
    BTreeMap<String, Vec<ValidatedOrdinarySink>>,
) {
    let mut sources = BTreeMap::new();
    let mut sinks = BTreeMap::<String, Vec<(usize, ValidatedOrdinarySink)>>::new();
    for resource in resources {
        match resource {
            ConnectorResource::Source {
                binding_id,
                binding,
            } => {
                sources.insert(binding_id, binding);
            }
            ConnectorResource::Sink {
                output_id,
                sink_id,
                configured_index,
                binding,
            } => sinks
                .entry(output_id)
                .or_default()
                .push((configured_index, ValidatedOrdinarySink { sink_id, binding })),
        }
    }
    let sinks = sinks
        .into_iter()
        .map(|(output_id, mut bindings)| {
            bindings.sort_by_key(|(index, _)| *index);
            (
                output_id,
                bindings.into_iter().map(|(_, binding)| binding).collect(),
            )
        })
        .collect();
    (sources, sinks)
}

fn register_boundary_tasks(
    runtime: &mut RegisteredRuntime,
    context: &super::StreamJobContext,
    sources: BTreeMap<String, SourceBinding>,
    sinks: BTreeMap<String, Vec<ValidatedOrdinarySink>>,
    core: &Arc<JobCore>,
) -> RuntimeTaskProgress {
    let mut source_progress = BTreeMap::new();
    for (binding_id, binding) in sources {
        let outputs = runtime
            .source_outputs
            .remove(&binding_id)
            .expect("preflight projected every validated source route");
        let progress = spawn_source_tasks_gated_with_metrics(
            &mut runtime.supervisor,
            context,
            &binding_id,
            binding,
            outputs,
            runtime.data_gate.subscribe(),
            core.launch_cancel.clone(),
            core.metrics.clone(),
        )
        .expect("preflight validated every source task scope and first-hop budget");
        source_progress.insert(binding_id, progress);
    }
    let mut sink_progress = BTreeMap::new();
    for (output_id, bindings) in sinks {
        let input = runtime
            .sink_inputs
            .remove(&output_id)
            .expect("preflight projected every validated sink route");
        let progress = SinkProgress::default();
        spawn_sink_task(
            &mut runtime.supervisor,
            SinkTaskInputs {
                output_id: output_id.clone(),
                sinks: bindings,
                input,
                context: context
                    .for_sink(&output_id)
                    .expect("preflight validated every sink task scope"),
                progress: progress.clone(),
                metrics: core.metrics.clone(),
                data_gate: runtime.data_gate.subscribe(),
                launch_cancel: core.launch_cancel.clone(),
            },
        );
        sink_progress.insert(output_id, progress);
    }
    debug_assert!(runtime.source_outputs.is_empty());
    debug_assert!(runtime.sink_inputs.is_empty());
    {
        let mut status = core.runtime_status.lock();
        status.sources = source_progress.clone();
        status.sinks = sink_progress.clone();
    }
    RuntimeTaskProgress {
        sources: source_progress,
        sinks: sink_progress,
    }
}

async fn open_connector_resources(
    resources: &mut Vec<ConnectorResource>,
    cancellation: &CancellationToken,
) -> Vec<Arc<RuntimeFailure>> {
    let mut open_units = JoinSet::new();
    for (task_id, resource) in std::mem::take(resources).into_iter().enumerate() {
        spawn_open_unit(
            &mut open_units,
            task_id as u64,
            resource,
            cancellation.clone(),
        );
    }
    let mut open_failures = Vec::new();
    while let Some(joined) = open_units.join_next().await {
        match joined {
            Ok(exit) => {
                if let OpenResult::Failed(error) = exit.result {
                    open_failures.push(Arc::new(RuntimeFailure {
                        origin: exit.origin,
                        error,
                    }));
                    cancellation.cancel();
                }
                resources.push(exit.resource);
            }
            Err(error) => {
                open_failures.push(Arc::new(RuntimeFailure {
                    origin: FailureOrigin::Preflight,
                    error: CalcFlowError::Internal {
                        message: format!("connector open unit join failed: {error}"),
                    },
                }));
                cancellation.cancel();
            }
        }
    }
    open_failures
}

async fn await_handle_claim(core: &Arc<JobCore>) -> bool {
    loop {
        let notified = core.changed.notified();
        let delivery = core.state.lock().launch_delivery;
        if delivery == LaunchDeliveryState::Claimed || core.launch_cancel.is_cancelled() {
            return delivery == LaunchDeliveryState::Claimed && !core.launch_cancel.is_cancelled();
        }
        notified.await;
    }
}

async fn drive_running_job(
    launch_id: LaunchId,
    core: &Arc<JobCore>,
    deadline: Option<chrono::DateTime<Utc>>,
    cancellation: CancellationToken,
    progress: RuntimeTaskProgress,
    supervisor: &mut TaskSupervisor,
    metrics: &MetricsRecorder,
) -> DriverReport {
    let deadline_for_wait = deadline;
    let deadline_wait = async move {
        match deadline_for_wait {
            Some(deadline) => {
                let delay = deadline
                    .signed_duration_since(Utc::now())
                    .to_std()
                    .unwrap_or(Duration::ZERO);
                tokio::time::sleep(delay).await;
            }
            None => std::future::pending().await,
        }
    };
    tokio::pin!(deadline_wait);
    let join = supervisor.join_all();
    tokio::pin!(join);
    let mut committed_terminal = None;
    let mut graceful_requested = false;
    let mut deadline_fired = false;
    loop {
        if !deadline_fired && deadline.is_some_and(|deadline| Utc::now() >= deadline) {
            deadline_fired = true;
            core.request_deadline();
        }
        if committed_terminal.is_none() {
            #[cfg(test)]
            core.pause_before_terminal_commit().await;
            let observation = core.terminal_arbiter.observe_and_commit(&cancellation);
            committed_terminal = observation.terminal.map(|decision| match decision {
                TerminalDecision::TaskFailure(primary_task_id) => {
                    TerminalCause::TaskFailure { primary_task_id }
                }
                TerminalDecision::ExplicitCancel => TerminalCause::ExplicitCancel,
                TerminalDecision::DeadlineExceeded => TerminalCause::DeadlineExceeded,
            });
            if let Some(terminal) = &committed_terminal {
                core.state.lock().selected_cause = Some(terminal.clone());
            }
            if committed_terminal.is_none() && !graceful_requested && observation.graceful_shutdown
            {
                graceful_requested = true;
                core.state.lock().selected_cause = Some(TerminalCause::GracefulShutdown);
                for source in progress.sources.values() {
                    source.request_drain();
                }
            }
        }
        tokio::select! {
            biased;
            report = &mut join => {
                return finish_running_report(
                    launch_id,
                    committed_terminal,
                    graceful_requested,
                    report,
                    &progress,
                    metrics,
                );
            }
            () = core.changed.notified() => {}
            () = &mut deadline_wait, if !deadline_fired => {
                deadline_fired = true;
                core.request_deadline();
            },
        }
    }
}

fn finish_running_report(
    launch_id: LaunchId,
    committed_terminal: Option<TerminalCause>,
    graceful_requested: bool,
    report: SupervisionReport,
    progress: &RuntimeTaskProgress,
    metrics: &MetricsRecorder,
) -> DriverReport {
    let primary_task_id = report
        .primary_errors()
        .first()
        .map(|failure| failure.task_id);
    let mut errors = runtime_failures(report, &progress.sources, &progress.sinks);
    let cause = committed_terminal.unwrap_or(match primary_task_id {
        Some(primary_task_id) => TerminalCause::TaskFailure { primary_task_id },
        None if graceful_requested => TerminalCause::GracefulShutdown,
        None => TerminalCause::NaturalEnd,
    });
    let state = match cause {
        TerminalCause::NaturalEnd | TerminalCause::GracefulShutdown => {
            ContinuousJobState::Completed
        }
        TerminalCause::ExplicitCancel | TerminalCause::DeadlineExceeded => {
            ContinuousJobState::Cancelled
        }
        TerminalCause::TaskFailure { .. } => errors
            .first()
            .map_or(ContinuousJobState::Failed, |failure| {
                classify_failure_state(failure)
            }),
    };
    metrics.record_terminal(state, cause.clone());
    if let Some(overflow) = metrics.account_terminal_errors_once(&errors) {
        errors.push(overflow);
    }
    DriverReport {
        launch_id,
        completion: DriverCompletion::Outcome(Arc::new(ContinuousJobOutcome {
            state,
            cause,
            errors,
        })),
        cleanup_failures: Vec::new(),
    }
}

fn classify_failure_state(failure: &RuntimeFailure) -> ContinuousJobState {
    let recoverable_origin = matches!(
        &failure.origin,
        FailureOrigin::SourceOpen { .. }
            | FailureOrigin::SourceClose { .. }
            | FailureOrigin::SinkOpen { .. }
            | FailureOrigin::SinkClose { .. }
            | FailureOrigin::SinkWrite { .. }
    ) || matches!(
        &failure.origin,
        FailureOrigin::Task { task_name, .. } if task_name.starts_with("source:")
    );
    let recoverable_error = matches!(
        &failure.error,
        CalcFlowError::Io { .. }
            | CalcFlowError::ExternalProvider { .. }
            | CalcFlowError::RecoveryRequired { .. }
    );
    if recoverable_origin && recoverable_error {
        ContinuousJobState::RecoveryRequired
    } else {
        ContinuousJobState::Failed
    }
}

fn runtime_failures(
    report: SupervisionReport,
    sources: &BTreeMap<String, SourceProgress>,
    sinks: &BTreeMap<String, SinkProgress>,
) -> Vec<Arc<RuntimeFailure>> {
    let mut failures = Vec::new();
    let mut consumed_sources = BTreeSet::new();
    let mut close_only_pumps = BTreeSet::new();
    let mut consumed_sinks = BTreeSet::new();
    for failure in report.errors {
        if let Some((binding_id, unit)) = source_task_identity(&failure.task_name) {
            let binding_id = binding_id.to_owned();
            let is_pump = unit == "pump";
            if is_pump && close_only_pumps.contains(&binding_id) {
                continue;
            }
            if let Some(progress) = sources.get(&binding_id)
                && consumed_sources.insert(binding_id.clone())
            {
                let close = progress.take_close_failures();
                let preserve_task_error = !is_pump || close.pump_operation_failed;
                if !close.pump_operation_failed && !close.errors.is_empty() {
                    close_only_pumps.insert(binding_id.clone());
                }
                if preserve_task_error {
                    failures.push(task_runtime_failure(failure));
                }
                failures.extend(close.errors.into_iter().map(|error| {
                    Arc::new(RuntimeFailure {
                        origin: FailureOrigin::SourceClose {
                            binding_id: binding_id.clone(),
                        },
                        error,
                    })
                }));
                continue;
            }
        }
        if let Some(output_id) = failure.task_name.strip_prefix("sink:")
            && let Some(progress) = sinks.get(output_id)
        {
            let records = progress.take_failures();
            if !records.is_empty() {
                consumed_sinks.insert(output_id.to_owned());
                failures.extend(records.into_iter().map(sink_runtime_failure));
                continue;
            }
        }
        failures.push(task_runtime_failure(failure));
    }
    for (output_id, progress) in sinks {
        if !consumed_sinks.contains(output_id) {
            failures.extend(
                progress
                    .take_failures()
                    .into_iter()
                    .map(sink_runtime_failure),
            );
        }
    }
    failures
}

fn source_task_identity(task_name: &str) -> Option<(&str, &str)> {
    let (binding_id, unit) = task_name.strip_prefix("source:")?.rsplit_once(':')?;
    matches!(unit, "pump" | "task").then_some((binding_id, unit))
}

fn task_runtime_failure(failure: super::supervisor::TaskFailure) -> Arc<RuntimeFailure> {
    Arc::new(RuntimeFailure {
        origin: FailureOrigin::Task {
            task_id: failure.task_id,
            task_name: failure.task_name,
        },
        error: failure.error,
    })
}

fn sink_runtime_failure(failure: super::sink_task::SinkTaskFailure) -> Arc<RuntimeFailure> {
    let origin = match failure.phase {
        SinkFailurePhase::Close => FailureOrigin::SinkClose {
            output_id: failure.output_id,
            sink_id: failure.sink_id,
        },
        SinkFailurePhase::Write if matches!(&failure.error, CalcFlowError::EdgeClosed { .. }) => {
            let edge_id = match &failure.error {
                CalcFlowError::EdgeClosed { edge } => edge.clone(),
                _ => unreachable!("guard above fixes the error variant"),
            };
            FailureOrigin::SinkIngress {
                output_id: failure.output_id,
                edge_id,
            }
        }
        SinkFailurePhase::Write => FailureOrigin::SinkWrite {
            output_id: failure.output_id,
            sink_id: failure.sink_id,
        },
    };
    Arc::new(RuntimeFailure {
        origin,
        error: failure.error,
    })
}

async fn finish_failed_launch(
    launch_id: LaunchId,
    mut failures: Vec<Arc<RuntimeFailure>>,
    resources: &mut [ConnectorResource],
    supervisor: &mut TaskSupervisor,
) -> DriverReport {
    let cleanup_failures = close_resources(resources).await;
    supervisor.cancel();
    let _ = supervisor.join_all().await;
    failures.sort_by(|left, right| left.origin.cmp(&right.origin));
    DriverReport {
        launch_id,
        completion: DriverCompletion::StartFailed(StartFailure {
            primary: failures.remove(0),
            diagnostic_id: None,
        }),
        cleanup_failures,
    }
}

async fn finish_cancelled_launch(
    launch_id: LaunchId,
    resources: &mut [ConnectorResource],
    supervisor: &mut TaskSupervisor,
    metrics: &MetricsRecorder,
) -> DriverReport {
    let cleanup_failures = close_resources(resources).await;
    supervisor.cancel();
    let _ = supervisor.join_all().await;
    DriverReport {
        cleanup_failures,
        ..cancelled_driver_report(launch_id, metrics)
    }
}

fn spawn_open_unit(
    units: &mut JoinSet<OpenExit>,
    task_id: u64,
    mut resource: ConnectorResource,
    cancellation: CancellationToken,
) {
    units.spawn(async move {
        let origin = resource.open_origin();
        let result = tokio::select! {
            biased;
            () = cancellation.cancelled() => OpenResult::Cancelled,
            result = AssertUnwindSafe(resource.open()).catch_unwind() => match result {
                Ok(Ok(())) => OpenResult::Opened,
                Ok(Err(error)) => OpenResult::Failed(error),
                Err(payload) => OpenResult::Failed(CalcFlowError::TaskPanicked {
                    task_id,
                    message: panic_message(payload.as_ref()),
                }),
            },
        };
        OpenExit {
            origin,
            resource,
            result,
        }
    });
}

async fn close_resources(resources: &mut [ConnectorResource]) -> Vec<Arc<RuntimeFailure>> {
    resources.sort_by_key(ConnectorResource::close_origin);
    let mut failures = Vec::new();
    for resource in resources {
        let origin = resource.close_origin();
        if let Err(error) = resource.close().await {
            failures.push(Arc::new(RuntimeFailure { origin, error }));
        }
    }
    failures
}

fn cancelled_driver_report(launch_id: LaunchId, metrics: &MetricsRecorder) -> DriverReport {
    metrics.record_terminal(ContinuousJobState::Cancelled, TerminalCause::ExplicitCancel);
    DriverReport {
        launch_id,
        completion: DriverCompletion::Outcome(Arc::new(ContinuousJobOutcome {
            state: ContinuousJobState::Cancelled,
            cause: TerminalCause::ExplicitCancel,
            errors: Vec::new(),
        })),
        cleanup_failures: Vec::new(),
    }
}

fn cancelled_driver_report_with_task_cleanup(
    launch_id: LaunchId,
    report: SupervisionReport,
    progress: &RuntimeTaskProgress,
    metrics: &MetricsRecorder,
) -> DriverReport {
    let mut errors = runtime_failures(report, &progress.sources, &progress.sinks);
    metrics.record_terminal(ContinuousJobState::Cancelled, TerminalCause::ExplicitCancel);
    if let Some(overflow) = metrics.account_terminal_errors_once(&errors) {
        errors.push(overflow);
    }
    DriverReport {
        launch_id,
        completion: DriverCompletion::Outcome(Arc::new(ContinuousJobOutcome {
            state: ContinuousJobState::Cancelled,
            cause: TerminalCause::ExplicitCancel,
            errors,
        })),
        cleanup_failures: Vec::new(),
    }
}

#[cfg(test)]
mod tests {
    use std::{
        collections::{BTreeMap, BTreeSet, VecDeque},
        future::Future as _,
        sync::{
            Arc,
            atomic::{AtomicBool, AtomicU64, AtomicUsize, Ordering},
        },
        task::Poll,
        time::Duration as StdDuration,
    };

    use async_trait::async_trait;
    use datafusion::arrow::{array::Int64Array, record_batch::RecordBatch};
    use parking_lot::Mutex;
    use tokio::sync::{Notify, Semaphore, mpsc};

    use super::{
        ABANDONED_RUNNER_WARNING, ContinuousJobState, ContinuousRunner, DriverCompletion,
        DriverOwnership, FailureOrigin as RuntimeFailureOrigin, JobCore, LaunchId, RunnerCore,
        RunnerDiagnostics, RunnerRegistryState, RunnerShutdownObserver, RuntimeFailure,
        RuntimeTaskProgress, TerminalCause, classify_failure_state, finish_running_report,
    };
    use crate::{
        Batch, BatchKind, BatchMetadata, CalcFlowError, CancellationToken, Edge, EdgeBudget,
        EventTime, ExpressionOperator, JsonMap, OperatorMetadata, PipelineBuilder, Port,
        PortEndpoint, Result, StreamCollector, StreamJobContext, StreamOperator,
        StreamOperatorContext, StreamRequirements, UdfRegistry, UnionOperator,
        runtime::streaming::{
            job::{
                ContinuousJobSpec, M2DeliveryMode, M2SinkDelivery, NamedSinkBinding,
                NamedSourceBinding, OrdinarySinkBinding, OrdinaryStreamSink,
            },
            metrics::MetricsRecorder,
            source_task::{Cursor, SourceBinding, SourceCapabilities, SourceEvent, StreamSource},
            supervisor::{SupervisionReport, TaskId},
        },
    };

    struct ResetOperator {
        inputs: [Port; 1],
        outputs: [Port; 1],
        resets: Arc<AtomicUsize>,
        fail_reset: bool,
        panic_reset: bool,
    }

    struct EntryDataProbeOperator {
        inputs: [Port; 1],
        outputs: [Port; 1],
        resets: Arc<AtomicUsize>,
        processed: Arc<AtomicUsize>,
    }

    struct OrderedEntryFailureOperator {
        inputs: [Port; 1],
        outputs: [Port; 1],
        node_id: &'static str,
        later_node_returned: Arc<AtomicBool>,
    }

    impl OperatorMetadata for OrderedEntryFailureOperator {
        fn name(&self) -> &'static str {
            "ordered-entry-failure"
        }

        fn input_ports(&self) -> &[Port] {
            &self.inputs
        }

        fn output_ports(&self) -> &[Port] {
            &self.outputs
        }

        fn configuration(&self) -> JsonMap {
            JsonMap::new()
        }
    }

    #[async_trait]
    impl StreamOperator for OrderedEntryFailureOperator {
        async fn process_data(
            &mut self,
            _ingress: &str,
            _batch: Batch,
            _context: &StreamOperatorContext<'_>,
            _output: &mut dyn StreamCollector,
        ) -> Result<()> {
            unreachable!("entry failure prevents data execution")
        }

        async fn on_watermark(
            &mut self,
            _watermark: EventTime,
            _context: &StreamOperatorContext<'_>,
            _output: &mut dyn StreamCollector,
        ) -> Result<()> {
            Ok(())
        }

        async fn on_end(
            &mut self,
            _context: &StreamOperatorContext<'_>,
            _output: &mut dyn StreamCollector,
        ) -> Result<()> {
            Ok(())
        }

        fn reset(&mut self) -> Result<()> {
            if self.node_id == "a" {
                while !self.later_node_returned.load(Ordering::SeqCst) {
                    std::thread::yield_now();
                }
            } else {
                self.later_node_returned.store(true, Ordering::SeqCst);
            }
            Err(CalcFlowError::Operator {
                node_id: self.node_id.into(),
                message: format!("{} reset failed", self.node_id),
            })
        }
    }

    struct JobIdentityProbeOperator {
        inputs: [Port; 1],
        outputs: [Port; 1],
        observed_job_id: Arc<Mutex<Option<u64>>>,
    }

    struct ActiveCancelledOperator {
        inputs: [Port; 1],
        outputs: [Port; 1],
    }

    impl OperatorMetadata for ActiveCancelledOperator {
        fn name(&self) -> &'static str {
            "active-cancelled"
        }

        fn input_ports(&self) -> &[Port] {
            &self.inputs
        }

        fn output_ports(&self) -> &[Port] {
            &self.outputs
        }

        fn configuration(&self) -> JsonMap {
            JsonMap::new()
        }
    }

    #[async_trait]
    impl StreamOperator for ActiveCancelledOperator {
        async fn process_data(
            &mut self,
            _ingress: &str,
            _batch: Batch,
            _context: &StreamOperatorContext<'_>,
            _output: &mut dyn StreamCollector,
        ) -> Result<()> {
            Err(CalcFlowError::Cancelled {
                run_id: "operator-active-cancelled".into(),
            })
        }

        async fn on_watermark(
            &mut self,
            _watermark: EventTime,
            _context: &StreamOperatorContext<'_>,
            _output: &mut dyn StreamCollector,
        ) -> Result<()> {
            Ok(())
        }

        async fn on_end(
            &mut self,
            _context: &StreamOperatorContext<'_>,
            _output: &mut dyn StreamCollector,
        ) -> Result<()> {
            Ok(())
        }
    }

    impl OperatorMetadata for JobIdentityProbeOperator {
        fn name(&self) -> &'static str {
            "job-identity-probe"
        }

        fn input_ports(&self) -> &[Port] {
            &self.inputs
        }

        fn output_ports(&self) -> &[Port] {
            &self.outputs
        }

        fn configuration(&self) -> JsonMap {
            JsonMap::new()
        }
    }

    #[async_trait]
    impl StreamOperator for JobIdentityProbeOperator {
        async fn process_data(
            &mut self,
            _ingress: &str,
            batch: Batch,
            context: &StreamOperatorContext<'_>,
            output: &mut dyn StreamCollector,
        ) -> Result<()> {
            *self.observed_job_id.lock() = Some(context.job().job_id());
            output.emit("output", batch).await
        }

        async fn on_watermark(
            &mut self,
            _watermark: EventTime,
            _context: &StreamOperatorContext<'_>,
            _output: &mut dyn StreamCollector,
        ) -> Result<()> {
            Ok(())
        }

        async fn on_end(
            &mut self,
            _context: &StreamOperatorContext<'_>,
            _output: &mut dyn StreamCollector,
        ) -> Result<()> {
            Ok(())
        }
    }

    impl OperatorMetadata for EntryDataProbeOperator {
        fn name(&self) -> &'static str {
            "entry-data-probe"
        }

        fn input_ports(&self) -> &[Port] {
            &self.inputs
        }

        fn output_ports(&self) -> &[Port] {
            &self.outputs
        }

        fn configuration(&self) -> JsonMap {
            JsonMap::new()
        }
    }

    #[async_trait]
    impl StreamOperator for EntryDataProbeOperator {
        async fn process_data(
            &mut self,
            _ingress: &str,
            batch: Batch,
            _context: &StreamOperatorContext<'_>,
            output: &mut dyn StreamCollector,
        ) -> Result<()> {
            self.processed.fetch_add(1, Ordering::SeqCst);
            output.emit("output", batch).await
        }

        async fn on_watermark(
            &mut self,
            _watermark: EventTime,
            _context: &StreamOperatorContext<'_>,
            _output: &mut dyn StreamCollector,
        ) -> Result<()> {
            Ok(())
        }

        async fn on_end(
            &mut self,
            _context: &StreamOperatorContext<'_>,
            _output: &mut dyn StreamCollector,
        ) -> Result<()> {
            Ok(())
        }

        fn reset(&mut self) -> Result<()> {
            self.resets.fetch_add(1, Ordering::SeqCst);
            Ok(())
        }
    }

    impl OperatorMetadata for ResetOperator {
        fn name(&self) -> &'static str {
            "reset-probe"
        }

        fn input_ports(&self) -> &[Port] {
            &self.inputs
        }

        fn output_ports(&self) -> &[Port] {
            &self.outputs
        }

        fn configuration(&self) -> JsonMap {
            BTreeMap::new()
        }
    }

    #[async_trait]
    impl StreamOperator for ResetOperator {
        async fn process_data(
            &mut self,
            _ingress: &str,
            _batch: Batch,
            _context: &StreamOperatorContext<'_>,
            _output: &mut dyn StreamCollector,
        ) -> Result<()> {
            Err(CalcFlowError::Operator {
                node_id: "node".into(),
                message: "data failure".into(),
            })
        }

        async fn on_watermark(
            &mut self,
            _watermark: EventTime,
            _context: &StreamOperatorContext<'_>,
            _output: &mut dyn StreamCollector,
        ) -> Result<()> {
            Ok(())
        }

        async fn on_end(
            &mut self,
            _context: &StreamOperatorContext<'_>,
            _output: &mut dyn StreamCollector,
        ) -> Result<()> {
            Ok(())
        }

        fn reset(&mut self) -> Result<()> {
            self.resets.fetch_add(1, Ordering::SeqCst);
            assert!(!self.panic_reset, "operator reset panicked");
            if self.fail_reset {
                Err(CalcFlowError::Operator {
                    node_id: "node".into(),
                    message: "reset failed".into(),
                })
            } else {
                Ok(())
            }
        }
    }

    #[derive(Clone, Default)]
    struct LifecycleProbe {
        opened: Arc<AtomicUsize>,
        open_completed: Arc<AtomicUsize>,
        closed: Arc<AtomicUsize>,
        open_started: Arc<Notify>,
        open_release: Arc<Notify>,
        close_started: Arc<Notify>,
        close_release: Arc<Notify>,
        block_open: Arc<AtomicBool>,
        block_close: Arc<AtomicBool>,
        fail_open: Arc<AtomicBool>,
        panic_open: Arc<AtomicBool>,
        panic_next: Arc<AtomicBool>,
        panic_write: Arc<AtomicBool>,
        fail_close: Arc<AtomicBool>,
        panic_close: Arc<AtomicBool>,
        finite: Arc<AtomicBool>,
    }

    struct ProbeSource(LifecycleProbe);

    #[async_trait]
    impl StreamSource for ProbeSource {
        async fn open(&mut self, _cursor: Option<Cursor>) -> Result<()> {
            self.0.opened.fetch_add(1, Ordering::SeqCst);
            self.0.open_started.notify_waiters();
            if self.0.block_open.load(Ordering::SeqCst) {
                self.0.open_release.notified().await;
            }
            assert!(
                !self.0.panic_open.load(Ordering::SeqCst),
                "source open panicked"
            );
            if self.0.fail_open.load(Ordering::SeqCst) {
                Err(CalcFlowError::Internal {
                    message: "source open failed".into(),
                })
            } else {
                self.0.open_completed.fetch_add(1, Ordering::SeqCst);
                Ok(())
            }
        }

        async fn next(&mut self) -> Result<Option<SourceEvent>> {
            assert!(
                !self.0.panic_next.load(Ordering::SeqCst),
                "source next panicked"
            );
            if self.0.finite.load(Ordering::SeqCst) {
                Ok(None)
            } else {
                std::future::pending().await
            }
        }

        async fn close(&mut self) -> Result<()> {
            self.0.closed.fetch_add(1, Ordering::SeqCst);
            self.0.close_started.notify_waiters();
            if self.0.block_close.load(Ordering::SeqCst) {
                self.0.close_release.notified().await;
            }
            if self.0.fail_close.load(Ordering::SeqCst) {
                Err(CalcFlowError::Internal {
                    message: "source close failed".into(),
                })
            } else {
                Ok(())
            }
        }

        fn capabilities(&self) -> SourceCapabilities {
            SourceCapabilities {
                replayable: true,
                max_batch_rows: 1,
                max_batch_bytes: 1,
            }
        }
    }

    struct ProbeSink(LifecycleProbe);

    #[async_trait]
    impl OrdinaryStreamSink for ProbeSink {
        fn delivery_capability(&self) -> M2SinkDelivery {
            M2SinkDelivery::ProcessLocalOrdered
        }

        async fn open(&mut self) -> Result<()> {
            self.0.opened.fetch_add(1, Ordering::SeqCst);
            self.0.open_started.notify_waiters();
            if self.0.block_open.load(Ordering::SeqCst) {
                self.0.open_release.notified().await;
            }
            assert!(
                !self.0.panic_open.load(Ordering::SeqCst),
                "sink open panicked"
            );
            if self.0.fail_open.load(Ordering::SeqCst) {
                Err(CalcFlowError::Internal {
                    message: "sink open failed".into(),
                })
            } else {
                self.0.open_completed.fetch_add(1, Ordering::SeqCst);
                Ok(())
            }
        }

        async fn write(&mut self, _batch: &Batch) -> Result<()> {
            assert!(
                !self.0.panic_write.load(Ordering::SeqCst),
                "sink write panicked"
            );
            Ok(())
        }

        async fn close(&mut self) -> Result<()> {
            self.0.closed.fetch_add(1, Ordering::SeqCst);
            self.0.close_started.notify_waiters();
            if self.0.block_close.load(Ordering::SeqCst) {
                self.0.close_release.notified().await;
            }
            assert!(
                !self.0.panic_close.load(Ordering::SeqCst),
                "sink close panicked"
            );
            if self.0.fail_close.load(Ordering::SeqCst) {
                Err(CalcFlowError::Internal {
                    message: "sink close failed".into(),
                })
            } else {
                Ok(())
            }
        }
    }

    struct FiniteSource {
        events: VecDeque<SourceEvent>,
        closed: Arc<AtomicUsize>,
    }

    struct ResumeProbeSource {
        opened_with: Arc<Mutex<Vec<Option<Cursor>>>>,
        closed: Arc<AtomicUsize>,
    }

    #[async_trait]
    impl StreamSource for ResumeProbeSource {
        async fn open(&mut self, cursor: Option<Cursor>) -> Result<()> {
            self.opened_with.lock().push(cursor);
            Ok(())
        }

        async fn next(&mut self) -> Result<Option<SourceEvent>> {
            Ok(None)
        }

        async fn close(&mut self) -> Result<()> {
            self.closed.fetch_add(1, Ordering::SeqCst);
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

    struct GatedDataSource {
        release: Arc<Notify>,
        delivered: bool,
        closed: Arc<AtomicUsize>,
    }

    #[async_trait]
    impl StreamSource for GatedDataSource {
        async fn open(&mut self, _cursor: Option<Cursor>) -> Result<()> {
            Ok(())
        }

        async fn next(&mut self) -> Result<Option<SourceEvent>> {
            if self.delivered {
                return std::future::pending().await;
            }
            self.release.notified().await;
            self.delivered = true;
            Ok(Some(SourceEvent::Data {
                batch: one_row(1),
                cursor: Cursor::new(vec![1], JsonMap::new()).unwrap(),
            }))
        }

        async fn close(&mut self) -> Result<()> {
            self.closed.fetch_add(1, Ordering::SeqCst);
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

    #[async_trait]
    impl StreamSource for FiniteSource {
        async fn open(&mut self, _cursor: Option<Cursor>) -> Result<()> {
            Ok(())
        }

        async fn next(&mut self) -> Result<Option<SourceEvent>> {
            Ok(self.events.pop_front())
        }

        async fn close(&mut self) -> Result<()> {
            self.closed.fetch_add(1, Ordering::SeqCst);
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

    struct OrderedRecordingSink {
        id: String,
        writes: Arc<Mutex<Vec<(String, String, u64)>>>,
        closed: Arc<AtomicUsize>,
    }

    struct CountingPendingSource {
        events: VecDeque<SourceEvent>,
        polls: Arc<AtomicUsize>,
        closed: Arc<AtomicUsize>,
    }

    #[derive(Clone, Copy)]
    enum RunningSourceFailure {
        Next,
        Cursor,
    }

    struct PrimaryAndCloseFailingSource {
        failure: RunningSourceFailure,
        next_call: usize,
        closed: Arc<AtomicUsize>,
    }

    #[async_trait]
    impl StreamSource for PrimaryAndCloseFailingSource {
        async fn open(&mut self, _cursor: Option<Cursor>) -> Result<()> {
            Ok(())
        }

        async fn next(&mut self) -> Result<Option<SourceEvent>> {
            let call = self.next_call;
            self.next_call += 1;
            match (self.failure, call) {
                (RunningSourceFailure::Next, _) => Err(CalcFlowError::Internal {
                    message: "source-next-primary".into(),
                }),
                (RunningSourceFailure::Cursor, 0 | 1) => Ok(Some(SourceEvent::Data {
                    batch: one_row(i64::try_from(call).unwrap()),
                    cursor: Cursor::new(vec![1], JsonMap::new()).unwrap(),
                })),
                (RunningSourceFailure::Cursor, _) => std::future::pending().await,
            }
        }

        async fn close(&mut self) -> Result<()> {
            self.closed.fetch_add(1, Ordering::SeqCst);
            Err(CalcFlowError::Internal {
                message: "source-close-secondary".into(),
            })
        }

        fn capabilities(&self) -> SourceCapabilities {
            SourceCapabilities {
                replayable: true,
                max_batch_rows: 1,
                max_batch_bytes: 1 << 20,
            }
        }
    }

    #[async_trait]
    impl StreamSource for CountingPendingSource {
        async fn open(&mut self, _cursor: Option<Cursor>) -> Result<()> {
            Ok(())
        }

        async fn next(&mut self) -> Result<Option<SourceEvent>> {
            self.polls.fetch_add(1, Ordering::SeqCst);
            match self.events.pop_front() {
                Some(event) => Ok(Some(event)),
                None => std::future::pending().await,
            }
        }

        async fn close(&mut self) -> Result<()> {
            self.closed.fetch_add(1, Ordering::SeqCst);
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

    struct GatedSink {
        started: Arc<AtomicBool>,
        gate: Arc<Notify>,
        writes: Arc<Mutex<Vec<(String, u64)>>>,
        closed: Arc<AtomicUsize>,
    }

    struct DeadlinePendingOperator {
        inputs: [Port; 1],
        outputs: [Port; 1],
        entered: Arc<AtomicBool>,
    }

    impl OperatorMetadata for DeadlinePendingOperator {
        fn name(&self) -> &'static str {
            "deadline-pending"
        }

        fn input_ports(&self) -> &[Port] {
            &self.inputs
        }

        fn output_ports(&self) -> &[Port] {
            &self.outputs
        }

        fn configuration(&self) -> JsonMap {
            JsonMap::new()
        }
    }

    #[async_trait]
    impl StreamOperator for DeadlinePendingOperator {
        async fn process_data(
            &mut self,
            _ingress: &str,
            _batch: Batch,
            _context: &StreamOperatorContext<'_>,
            _output: &mut dyn StreamCollector,
        ) -> Result<()> {
            self.entered.store(true, Ordering::SeqCst);
            std::future::pending().await
        }

        async fn on_watermark(
            &mut self,
            _watermark: EventTime,
            _context: &StreamOperatorContext<'_>,
            _output: &mut dyn StreamCollector,
        ) -> Result<()> {
            Ok(())
        }

        async fn on_end(
            &mut self,
            _context: &StreamOperatorContext<'_>,
            _output: &mut dyn StreamCollector,
        ) -> Result<()> {
            Ok(())
        }
    }

    #[derive(Clone, Copy, Debug, Eq, Ord, PartialEq, PartialOrd)]
    enum StressGate {
        LeftData0,
        LeftData1,
        LeftEof,
        RightData0,
        RightData1,
        RightEof,
        Edge0,
        Edge1,
        Edge2,
        Edge3,
        SinkA0,
        SinkA1,
        SinkA2,
        SinkA3,
        SinkB0,
        SinkB1,
        SinkB2,
        SinkB3,
        Natural,
        Drain,
        Cancel,
    }

    fn stress_schedule(seed: u64) -> Vec<StressGate> {
        let mut gates = vec![
            StressGate::LeftData0,
            StressGate::LeftData1,
            StressGate::LeftEof,
            StressGate::RightData0,
            StressGate::RightData1,
            StressGate::RightEof,
            StressGate::Edge0,
            StressGate::Edge1,
            StressGate::Edge2,
            StressGate::Edge3,
            StressGate::SinkA0,
            StressGate::SinkA1,
            StressGate::SinkA2,
            StressGate::SinkA3,
            StressGate::SinkB0,
            StressGate::SinkB1,
            StressGate::SinkB2,
            StressGate::SinkB3,
            match seed % 3 {
                0 => StressGate::Natural,
                1 => StressGate::Drain,
                _ => StressGate::Cancel,
            },
        ];
        let mut state = seed.wrapping_add(0x9e37_79b9_7f4a_7c15);
        for index in (1..gates.len()).rev() {
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            let swap = usize::try_from(state % u64::try_from(index + 1).unwrap()).unwrap();
            gates.swap(index, swap);
        }
        if seed % 3 != 0 {
            let terminal = gates
                .iter()
                .position(|gate| matches!(gate, StressGate::Drain | StressGate::Cancel))
                .unwrap();
            let first_eof = gates
                .iter()
                .position(|gate| matches!(gate, StressGate::LeftEof | StressGate::RightEof))
                .unwrap();
            if terminal > first_eof {
                gates.swap(terminal, first_eof);
            }
        }
        gates
    }

    struct StressSource {
        events: VecDeque<(Arc<Semaphore>, Option<SourceEvent>)>,
        closed: Arc<AtomicUsize>,
    }

    #[async_trait]
    impl StreamSource for StressSource {
        async fn open(&mut self, _cursor: Option<Cursor>) -> Result<()> {
            Ok(())
        }

        async fn next(&mut self) -> Result<Option<SourceEvent>> {
            let Some((gate, event)) = self.events.pop_front() else {
                return std::future::pending().await;
            };
            let permit = gate.acquire().await.map_err(|_| CalcFlowError::Internal {
                message: "stress source gate closed".into(),
            })?;
            permit.forget();
            Ok(event)
        }

        async fn close(&mut self) -> Result<()> {
            self.closed.fetch_add(1, Ordering::SeqCst);
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

    struct StressForwardOperator {
        inputs: [Port; 1],
        outputs: [Port; 1],
        gates: Option<Arc<Mutex<VecDeque<Arc<Semaphore>>>>>,
    }

    impl StressForwardOperator {
        fn new(gates: Option<Vec<Arc<Semaphore>>>) -> Self {
            Self {
                inputs: [Port::new("input", BatchKind::Table, true, None).unwrap()],
                outputs: [Port::new("output", BatchKind::Table, true, None).unwrap()],
                gates: gates.map(|gates| Arc::new(Mutex::new(gates.into()))),
            }
        }
    }

    impl OperatorMetadata for StressForwardOperator {
        fn name(&self) -> &'static str {
            "stress-forward"
        }

        fn input_ports(&self) -> &[Port] {
            &self.inputs
        }

        fn output_ports(&self) -> &[Port] {
            &self.outputs
        }

        fn configuration(&self) -> JsonMap {
            JsonMap::new()
        }
    }

    #[async_trait]
    impl StreamOperator for StressForwardOperator {
        async fn process_data(
            &mut self,
            _ingress: &str,
            batch: Batch,
            _context: &StreamOperatorContext<'_>,
            output: &mut dyn StreamCollector,
        ) -> Result<()> {
            let gate = self
                .gates
                .as_ref()
                .and_then(|gates| gates.lock().pop_front());
            if let Some(gate) = gate {
                let permit = gate.acquire().await.map_err(|_| CalcFlowError::Internal {
                    message: "stress edge gate closed".into(),
                })?;
                permit.forget();
            }
            output.emit("output", batch).await
        }

        async fn on_watermark(
            &mut self,
            _watermark: EventTime,
            _context: &StreamOperatorContext<'_>,
            _output: &mut dyn StreamCollector,
        ) -> Result<()> {
            Ok(())
        }

        async fn on_end(
            &mut self,
            _context: &StreamOperatorContext<'_>,
            _output: &mut dyn StreamCollector,
        ) -> Result<()> {
            Ok(())
        }
    }

    struct StressSink {
        gates: VecDeque<Arc<Semaphore>>,
        writes: Arc<Mutex<Vec<(String, u64)>>>,
        closed: Arc<AtomicUsize>,
    }

    #[async_trait]
    impl OrdinaryStreamSink for StressSink {
        async fn open(&mut self) -> Result<()> {
            Ok(())
        }

        async fn write(&mut self, batch: &Batch) -> Result<()> {
            let gate = self
                .gates
                .pop_front()
                .ok_or_else(|| CalcFlowError::Internal {
                    message: "stress sink exhausted its gates".into(),
                })?;
            let permit = gate.acquire().await.map_err(|_| CalcFlowError::Internal {
                message: "stress sink gate closed".into(),
            })?;
            permit.forget();
            self.writes.lock().push((
                batch.metadata().source().into(),
                batch.metadata().sequence(),
            ));
            Ok(())
        }

        async fn close(&mut self) -> Result<()> {
            self.closed.fetch_add(1, Ordering::SeqCst);
            Ok(())
        }
    }

    #[async_trait]
    impl OrdinaryStreamSink for GatedSink {
        async fn open(&mut self) -> Result<()> {
            Ok(())
        }

        async fn write(&mut self, batch: &Batch) -> Result<()> {
            self.started.store(true, Ordering::SeqCst);
            self.gate.notified().await;
            self.writes.lock().push((
                batch.metadata().source().into(),
                batch.metadata().sequence(),
            ));
            Ok(())
        }

        async fn close(&mut self) -> Result<()> {
            self.closed.fetch_add(1, Ordering::SeqCst);
            Ok(())
        }
    }

    #[async_trait]
    impl OrdinaryStreamSink for OrderedRecordingSink {
        async fn open(&mut self) -> Result<()> {
            Ok(())
        }

        async fn write(&mut self, batch: &Batch) -> Result<()> {
            self.writes.lock().push((
                self.id.clone(),
                batch.metadata().source().into(),
                batch.metadata().sequence(),
            ));
            Ok(())
        }

        async fn close(&mut self) -> Result<()> {
            self.closed.fetch_add(1, Ordering::SeqCst);
            Ok(())
        }
    }

    fn one_row(value: i64) -> Batch {
        let record = RecordBatch::try_from_iter(vec![(
            "value",
            Arc::new(Int64Array::from(vec![value])) as _,
        )])
        .unwrap();
        Batch::table(vec![record], BatchMetadata::default()).unwrap()
    }

    fn finite_binding(values: &[i64], closed: &Arc<AtomicUsize>) -> SourceBinding {
        let events = values
            .iter()
            .enumerate()
            .map(|(index, value)| SourceEvent::Data {
                batch: one_row(*value),
                cursor: Cursor::new(vec![u8::try_from(index + 1).unwrap()], JsonMap::new())
                    .unwrap(),
            })
            .collect();
        SourceBinding::new(
            Box::new(FiniteSource {
                events,
                closed: Arc::clone(closed),
            }),
            None,
            0,
        )
        .unwrap()
    }

    fn counting_pending_binding(
        count: usize,
        polls: &Arc<AtomicUsize>,
        closed: &Arc<AtomicUsize>,
    ) -> SourceBinding {
        let events = (0..count)
            .map(|index| SourceEvent::Data {
                batch: one_row(i64::try_from(index).unwrap()),
                cursor: Cursor::new(
                    u64::try_from(index + 1).unwrap().to_be_bytes().to_vec(),
                    JsonMap::new(),
                )
                .unwrap(),
            })
            .collect();
        SourceBinding::new(
            Box::new(CountingPendingSource {
                events,
                polls: Arc::clone(polls),
                closed: Arc::clone(closed),
            }),
            None,
            0,
        )
        .unwrap()
    }

    fn union_plan() -> crate::StreamExecutionPlan {
        let union = UnionOperator::new(
            "merge",
            vec![
                Port::new("left", BatchKind::Table, true, None).unwrap(),
                Port::new("right", BatchKind::Table, true, None).unwrap(),
            ],
        )
        .unwrap();
        PipelineBuilder::new("union-runtime")
            .unwrap()
            .add_node("merge", Box::new(union))
            .unwrap()
            .compile_stream(
                &UdfRegistry::new().snapshot(),
                &StreamRequirements::default(),
            )
            .unwrap()
    }

    fn unary_expression_plan() -> crate::StreamExecutionPlan {
        let expression =
            ExpressionOperator::new("calc", "plus_one = value + 1", Vec::new(), None, Vec::new())
                .unwrap();
        PipelineBuilder::new("unary-runtime")
            .unwrap()
            .add_node("calc", Box::new(expression))
            .unwrap()
            .compile_stream(
                &UdfRegistry::new().snapshot(),
                &StreamRequirements::default(),
            )
            .unwrap()
    }

    fn two_entry_probe_plan(
        resets: &Arc<AtomicUsize>,
        processed: &Arc<AtomicUsize>,
    ) -> crate::StreamExecutionPlan {
        let probe = |resets: &Arc<AtomicUsize>, processed: &Arc<AtomicUsize>| {
            Box::new(EntryDataProbeOperator {
                inputs: [Port::new("input", BatchKind::Table, true, None).unwrap()],
                outputs: [Port::new("output", BatchKind::Table, false, None).unwrap()],
                resets: Arc::clone(resets),
                processed: Arc::clone(processed),
            }) as Box<dyn StreamOperator>
        };
        PipelineBuilder::new("entry-data-gates")
            .unwrap()
            .add_node("first", probe(resets, processed))
            .unwrap()
            .add_node("second", probe(resets, processed))
            .unwrap()
            .connect(Edge::new(
                PortEndpoint::new("first", "output").unwrap(),
                PortEndpoint::new("second", "input").unwrap(),
            ))
            .unwrap()
            .compile_stream(
                &UdfRegistry::new().snapshot(),
                &StreamRequirements::default(),
            )
            .unwrap()
    }

    fn deadline_pending_plan(entered: Arc<AtomicBool>) -> crate::StreamExecutionPlan {
        let operator = DeadlinePendingOperator {
            inputs: [Port::new("input", BatchKind::Table, true, None).unwrap()],
            outputs: [Port::new("output", BatchKind::Table, true, None).unwrap()],
            entered,
        };
        PipelineBuilder::new("deadline-pending")
            .unwrap()
            .add_node("pending", Box::new(operator) as Box<dyn StreamOperator>)
            .unwrap()
            .compile_stream(
                &UdfRegistry::new().snapshot(),
                &StreamRequirements::default(),
            )
            .unwrap()
    }

    fn spec(
        fail_reset: bool,
        resets: Arc<AtomicUsize>,
        source: LifecycleProbe,
        sink: LifecycleProbe,
    ) -> ContinuousJobSpec {
        reset_spec(fail_reset, false, resets, source, sink)
    }

    fn reset_spec(
        fail_reset: bool,
        panic_reset: bool,
        resets: Arc<AtomicUsize>,
        source: LifecycleProbe,
        sink: LifecycleProbe,
    ) -> ContinuousJobSpec {
        let operator = ResetOperator {
            inputs: [Port::new("input", BatchKind::Table, true, None).unwrap()],
            outputs: [Port::new("output", BatchKind::Table, true, None).unwrap()],
            resets,
            fail_reset,
            panic_reset,
        };
        let plan = PipelineBuilder::new("launch")
            .unwrap()
            .add_node("node", Box::new(operator) as Box<dyn StreamOperator>)
            .unwrap()
            .compile_stream(
                &UdfRegistry::new().snapshot(),
                &StreamRequirements::default(),
            )
            .unwrap();
        ContinuousJobSpec {
            context: StreamJobContext::new(
                9,
                plan.fingerprint(),
                JsonMap::new(),
                None,
                CancellationToken::new(),
            ),
            plan,
            sources: vec![NamedSourceBinding {
                binding_id: "input".into(),
                binding: SourceBinding::new(Box::new(ProbeSource(source)), None, 0).unwrap(),
            }],
            sinks: vec![NamedSinkBinding {
                output_id: "output".into(),
                sink_id: "sink".into(),
                binding: OrdinarySinkBinding::new(Box::new(ProbeSink(sink))),
            }],
            edge_budget: EdgeBudget {
                max_rows: 1,
                max_bytes: 1,
            },
            delivery_mode: M2DeliveryMode::ProcessLocalOrdered,
        }
    }

    fn forward_spec(
        job_id: u64,
        source: SourceBinding,
        sinks: Vec<NamedSinkBinding>,
    ) -> ContinuousJobSpec {
        let plan = PipelineBuilder::new("runtime-panic-cleanup")
            .unwrap()
            .add_node(
                "node",
                Box::new(StressForwardOperator::new(None)) as Box<dyn StreamOperator>,
            )
            .unwrap()
            .compile_stream(
                &UdfRegistry::new().snapshot(),
                &StreamRequirements::default(),
            )
            .unwrap();
        ContinuousJobSpec {
            context: StreamJobContext::new(
                job_id,
                plan.fingerprint(),
                JsonMap::new(),
                None,
                CancellationToken::new(),
            ),
            plan,
            sources: vec![NamedSourceBinding {
                binding_id: "input".into(),
                binding: source,
            }],
            sinks,
            edge_budget: EdgeBudget {
                max_rows: 1,
                max_bytes: 1 << 20,
            },
            delivery_mode: M2DeliveryMode::ProcessLocalOrdered,
        }
    }

    fn named_probe_sink(sink_id: &str, probe: LifecycleProbe) -> NamedSinkBinding {
        NamedSinkBinding {
            output_id: "output".into(),
            sink_id: sink_id.into(),
            binding: OrdinarySinkBinding::new(Box::new(ProbeSink(probe))),
        }
    }

    fn stress_plan(gates: &BTreeMap<StressGate, Arc<Semaphore>>) -> crate::StreamExecutionPlan {
        let union = UnionOperator::new(
            "merge",
            vec![
                Port::new("left", BatchKind::Table, true, None).unwrap(),
                Port::new("right", BatchKind::Table, true, None).unwrap(),
            ],
        )
        .unwrap();
        let unary_gates = [
            StressGate::Edge0,
            StressGate::Edge1,
            StressGate::Edge2,
            StressGate::Edge3,
        ]
        .map(|gate| Arc::clone(&gates[&gate]))
        .to_vec();
        PipelineBuilder::new("seeded-stress")
            .unwrap()
            .add_node("merge", Box::new(union))
            .unwrap()
            .add_node(
                "unary",
                Box::new(StressForwardOperator::new(Some(unary_gates))) as Box<dyn StreamOperator>,
            )
            .unwrap()
            .add_node(
                "branch_a",
                Box::new(StressForwardOperator::new(None)) as Box<dyn StreamOperator>,
            )
            .unwrap()
            .add_node(
                "branch_b",
                Box::new(StressForwardOperator::new(None)) as Box<dyn StreamOperator>,
            )
            .unwrap()
            .connect(Edge::new(
                PortEndpoint::new("merge", "output").unwrap(),
                PortEndpoint::new("unary", "input").unwrap(),
            ))
            .unwrap()
            .connect(Edge::new(
                PortEndpoint::new("unary", "output").unwrap(),
                PortEndpoint::new("branch_a", "input").unwrap(),
            ))
            .unwrap()
            .connect(Edge::new(
                PortEndpoint::new("unary", "output").unwrap(),
                PortEndpoint::new("branch_b", "input").unwrap(),
            ))
            .unwrap()
            .compile_stream(
                &UdfRegistry::new().snapshot(),
                &StreamRequirements::default(),
            )
            .unwrap()
    }

    fn stress_source_binding(
        gates: &BTreeMap<StressGate, Arc<Semaphore>>,
        data_gates: [StressGate; 2],
        eof_gate: StressGate,
        values: [i64; 2],
        closed: &Arc<AtomicUsize>,
    ) -> SourceBinding {
        let events = data_gates
            .into_iter()
            .zip(values)
            .enumerate()
            .map(|(index, (gate, value))| {
                (
                    Arc::clone(&gates[&gate]),
                    Some(SourceEvent::Data {
                        batch: one_row(value),
                        cursor: Cursor::new(vec![u8::try_from(index + 1).unwrap()], JsonMap::new())
                            .unwrap(),
                    }),
                )
            })
            .chain(std::iter::once((Arc::clone(&gates[&eof_gate]), None)))
            .collect();
        SourceBinding::new(
            Box::new(StressSource {
                events,
                closed: Arc::clone(closed),
            }),
            None,
            0,
        )
        .unwrap()
    }

    fn assert_source_fifo(seed: u64, writes: &[(String, u64)]) {
        let unique = writes.iter().cloned().collect::<BTreeSet<_>>();
        assert_eq!(unique.len(), writes.len(), "duplicate at seed {seed}");
        for source in ["left", "right"] {
            let sequence = writes
                .iter()
                .filter(|(observed_source, _)| observed_source == source)
                .map(|(_, sequence)| *sequence)
                .collect::<Vec<_>>();
            assert_eq!(
                sequence,
                (0..u64::try_from(sequence.len()).unwrap()).collect::<Vec<_>>(),
                "per-source FIFO failed at seed {seed} for {source}"
            );
        }
    }

    #[test]
    fn seeded_gate_generator_is_pure_and_permutates_every_named_gate() {
        for seed in 0..100 {
            let first = stress_schedule(seed);
            let second = stress_schedule(seed);
            assert_eq!(first, second, "non-deterministic schedule at seed {seed}");
            assert_eq!(first.len(), 19, "wrong gate count at seed {seed}");
            assert_eq!(
                first.iter().copied().collect::<BTreeSet<_>>().len(),
                19,
                "duplicate gate at seed {seed}"
            );
            assert_eq!(
                first
                    .iter()
                    .filter(|gate| {
                        matches!(
                            gate,
                            StressGate::Natural | StressGate::Drain | StressGate::Cancel
                        )
                    })
                    .count(),
                1,
                "wrong terminal gate count at seed {seed}"
            );
        }
    }

    #[allow(
        clippy::similar_names,
        clippy::too_many_lines,
        reason = "the stress scenario keeps paired branch state and all seeded invariants together"
    )]
    #[tokio::test(start_paused = true)]
    async fn seeded_paused_time_stress_runs_one_hundred_full_graph_schedules() {
        const DATA_GATES: [StressGate; 18] = [
            StressGate::LeftData0,
            StressGate::LeftData1,
            StressGate::LeftEof,
            StressGate::RightData0,
            StressGate::RightData1,
            StressGate::RightEof,
            StressGate::Edge0,
            StressGate::Edge1,
            StressGate::Edge2,
            StressGate::Edge3,
            StressGate::SinkA0,
            StressGate::SinkA1,
            StressGate::SinkA2,
            StressGate::SinkA3,
            StressGate::SinkB0,
            StressGate::SinkB1,
            StressGate::SinkB2,
            StressGate::SinkB3,
        ];
        for seed in 0..100 {
            let gates = DATA_GATES
                .into_iter()
                .map(|gate| (gate, Arc::new(Semaphore::new(0))))
                .collect::<BTreeMap<_, _>>();
            let plan = stress_plan(&gates);
            let source_closed = Arc::new(AtomicUsize::new(0));
            let sink_closed = Arc::new(AtomicUsize::new(0));
            let sink_a_writes = Arc::new(Mutex::new(Vec::new()));
            let sink_b_writes = Arc::new(Mutex::new(Vec::new()));
            let sink_a_gates = [
                StressGate::SinkA0,
                StressGate::SinkA1,
                StressGate::SinkA2,
                StressGate::SinkA3,
            ]
            .map(|gate| Arc::clone(&gates[&gate]));
            let sink_b_gates = [
                StressGate::SinkB0,
                StressGate::SinkB1,
                StressGate::SinkB2,
                StressGate::SinkB3,
            ]
            .map(|gate| Arc::clone(&gates[&gate]));
            let spec = ContinuousJobSpec {
                context: StreamJobContext::new(
                    10_000 + seed,
                    plan.fingerprint(),
                    JsonMap::new(),
                    None,
                    CancellationToken::new(),
                ),
                plan,
                sources: vec![
                    NamedSourceBinding {
                        binding_id: "left".into(),
                        binding: stress_source_binding(
                            &gates,
                            [StressGate::LeftData0, StressGate::LeftData1],
                            StressGate::LeftEof,
                            [1, 2],
                            &source_closed,
                        ),
                    },
                    NamedSourceBinding {
                        binding_id: "right".into(),
                        binding: stress_source_binding(
                            &gates,
                            [StressGate::RightData0, StressGate::RightData1],
                            StressGate::RightEof,
                            [10, 20],
                            &source_closed,
                        ),
                    },
                ],
                sinks: vec![
                    NamedSinkBinding {
                        output_id: "branch_a.output".into(),
                        sink_id: "slow-a".into(),
                        binding: OrdinarySinkBinding::new(Box::new(StressSink {
                            gates: sink_a_gates.into(),
                            writes: Arc::clone(&sink_a_writes),
                            closed: Arc::clone(&sink_closed),
                        })),
                    },
                    NamedSinkBinding {
                        output_id: "branch_b.output".into(),
                        sink_id: "slow-b".into(),
                        binding: OrdinarySinkBinding::new(Box::new(StressSink {
                            gates: sink_b_gates.into(),
                            writes: Arc::clone(&sink_b_writes),
                            closed: Arc::clone(&sink_closed),
                        })),
                    },
                ],
                edge_budget: EdgeBudget {
                    max_rows: 1,
                    max_bytes: 1 << 20,
                },
                delivery_mode: M2DeliveryMode::ProcessLocalOrdered,
            };
            let mut runner = ContinuousRunner::new();
            let job = runner
                .start(spec)
                .await
                .unwrap_or_else(|failure| panic!("start failed at seed {seed}: {failure:?}"));
            let schedule = stress_schedule(seed);
            let mut terminal = None;
            for gate in schedule {
                match gate {
                    StressGate::Natural => {}
                    StressGate::Drain => terminal = Some(job.shutdown()),
                    StressGate::Cancel => terminal = Some(job.cancel()),
                    gate => gates[&gate].add_permits(1),
                }
                tokio::task::yield_now().await;
                let status = job.status();
                assert!(status.tasks.len() <= 10, "task growth at seed {seed}");
                assert!(
                    status
                        .edges
                        .values()
                        .all(|edge| { edge.charged_rows <= 1 && edge.charged_bytes <= (1 << 20) }),
                    "edge budget breach at seed {seed}: {:?}",
                    status.edges
                );
            }
            let outcome = match terminal {
                Some(observer) => observer.await,
                None => job.wait().await,
            };
            let status = job.status();
            assert!(status.tasks.is_empty(), "task leak at seed {seed}");
            assert!(
                status.edges.values().all(|edge| {
                    edge.queue_depth == 0 && edge.charged_rows == 0 && edge.charged_bytes == 0
                }),
                "queue leak at seed {seed}: {:?}",
                status.edges
            );
            assert!(
                status.edges.values().all(|edge| {
                    edge.high_water_rows <= 1 && edge.high_water_bytes <= (1 << 20)
                }),
                "high-water budget breach at seed {seed}: {:?}",
                status.edges
            );
            let a = sink_a_writes.lock().clone();
            let b = sink_b_writes.lock().clone();
            assert_source_fifo(seed, &a);
            assert_source_fifo(seed, &b);
            match seed % 3 {
                0 => {
                    assert_eq!(outcome.cause, TerminalCause::NaturalEnd, "seed {seed}");
                    assert_eq!(a.len(), 4, "loss at seed {seed}");
                    assert_eq!(b.len(), 4, "loss at seed {seed}");
                    assert_eq!(a, b, "fan-out divergence at seed {seed}");
                    assert!(
                        status
                            .metrics
                            .edges
                            .values()
                            .all(|edge| { edge.input_batches == edge.output_batches }),
                        "edge loss at seed {seed}: {:?}",
                        status.metrics.edges
                    );
                }
                1 => {
                    assert_eq!(
                        outcome.cause,
                        TerminalCause::GracefulShutdown,
                        "seed {seed}"
                    );
                    assert_eq!(a, b, "graceful fan-out divergence at seed {seed}");
                }
                _ => assert_eq!(outcome.cause, TerminalCause::ExplicitCancel, "seed {seed}"),
            }
            assert_eq!(source_closed.load(Ordering::SeqCst), 2, "seed {seed}");
            assert_eq!(sink_closed.load(Ordering::SeqCst), 2, "seed {seed}");
            assert_eq!(runner.registry_counts(), (0, 0), "seed {seed}");
            drop(job);
            runner.shutdown().await.unwrap();
            assert_eq!(runner.registry_counts(), (0, 0), "seed {seed}");
        }
    }

    #[tokio::test]
    async fn operator_entry_failure_joins_before_any_connector_lifecycle() {
        let resets = Arc::new(AtomicUsize::new(0));
        let source = LifecycleProbe::default();
        let sink = LifecycleProbe::default();
        let mut runner = ContinuousRunner::new();

        let failure = runner
            .start(spec(
                true,
                Arc::clone(&resets),
                source.clone(),
                sink.clone(),
            ))
            .await
            .unwrap_err();

        assert!(matches!(
            failure.primary.error,
            CalcFlowError::Operator { .. }
        ));
        assert_eq!(resets.load(Ordering::SeqCst), 1);
        assert_eq!(source.opened.load(Ordering::SeqCst), 0);
        assert_eq!(source.closed.load(Ordering::SeqCst), 0);
        assert_eq!(sink.opened.load(Ordering::SeqCst), 0);
        assert_eq!(sink.closed.load(Ordering::SeqCst), 0);
        runner.shutdown().await.unwrap();
    }

    #[tokio::test]
    async fn operator_reset_panic_is_typed_before_any_connector_lifecycle() {
        let resets = Arc::new(AtomicUsize::new(0));
        let source = LifecycleProbe::default();
        let sink = LifecycleProbe::default();
        let mut runner = ContinuousRunner::new();

        let failure = runner
            .start(reset_spec(
                false,
                true,
                Arc::clone(&resets),
                source.clone(),
                sink.clone(),
            ))
            .await
            .unwrap_err();

        assert!(matches!(
            failure.primary.origin,
            super::FailureOrigin::OperatorEntry { ref node_id } if node_id == "node"
        ));
        assert!(matches!(
            failure.primary.error,
            CalcFlowError::TaskPanicked { task_id: 0, ref message }
                if message == "operator reset panicked"
        ));
        assert_eq!(resets.load(Ordering::SeqCst), 1);
        assert_eq!(source.opened.load(Ordering::SeqCst), 0);
        assert_eq!(source.closed.load(Ordering::SeqCst), 0);
        assert_eq!(sink.opened.load(Ordering::SeqCst), 0);
        assert_eq!(sink.closed.load(Ordering::SeqCst), 0);
        runner.shutdown().await.unwrap();
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 4)]
    async fn operator_entry_primary_is_sorted_by_node_id_not_ack_arrival() {
        let later_node_returned = Arc::new(AtomicBool::new(false));
        let operator = |node_id: &'static str| {
            Box::new(OrderedEntryFailureOperator {
                inputs: [Port::new("input", BatchKind::Table, true, None).unwrap()],
                outputs: [Port::new("output", BatchKind::Table, true, None).unwrap()],
                node_id,
                later_node_returned: Arc::clone(&later_node_returned),
            }) as Box<dyn StreamOperator>
        };
        let plan = PipelineBuilder::new("stable-entry-failure")
            .unwrap()
            .add_node("a", operator("a"))
            .unwrap()
            .add_node("z", operator("z"))
            .unwrap()
            .connect(Edge::new(
                PortEndpoint::new("a", "output").unwrap(),
                PortEndpoint::new("z", "input").unwrap(),
            ))
            .unwrap()
            .compile_stream(
                &UdfRegistry::new().snapshot(),
                &StreamRequirements::default(),
            )
            .unwrap();
        let source = LifecycleProbe::default();
        let sink = LifecycleProbe::default();
        let job_spec = ContinuousJobSpec {
            context: StreamJobContext::new(
                93,
                plan.fingerprint(),
                JsonMap::new(),
                None,
                CancellationToken::new(),
            ),
            plan,
            sources: vec![NamedSourceBinding {
                binding_id: "input".into(),
                binding: SourceBinding::new(Box::new(ProbeSource(source.clone())), None, 0)
                    .unwrap(),
            }],
            sinks: vec![NamedSinkBinding {
                output_id: "output".into(),
                sink_id: "sink".into(),
                binding: OrdinarySinkBinding::new(Box::new(ProbeSink(sink.clone()))),
            }],
            edge_budget: EdgeBudget::default(),
            delivery_mode: M2DeliveryMode::ProcessLocalOrdered,
        };
        let mut runner = ContinuousRunner::new();

        let failure = runner.start(job_spec).await.unwrap_err();

        assert!(later_node_returned.load(Ordering::SeqCst));
        assert!(matches!(
            failure.primary.origin,
            super::FailureOrigin::OperatorEntry { ref node_id } if node_id == "a"
        ));
        assert!(matches!(
            failure.primary.error,
            CalcFlowError::Operator { ref node_id, ref message }
                if node_id == "a" && message == "a reset failed"
        ));
        assert_eq!(source.opened.load(Ordering::SeqCst), 0);
        assert_eq!(sink.opened.load(Ordering::SeqCst), 0);
        runner.shutdown().await.unwrap();
    }

    #[tokio::test]
    async fn all_operators_enter_before_open_and_data_waits_for_handle_claim() {
        let resets = Arc::new(AtomicUsize::new(0));
        let processed = Arc::new(AtomicUsize::new(0));
        let plan = two_entry_probe_plan(&resets, &processed);
        let polls = Arc::new(AtomicUsize::new(0));
        let source_closed = Arc::new(AtomicUsize::new(0));
        let sink = LifecycleProbe::default();
        sink.block_open.store(true, Ordering::SeqCst);
        let job_spec = ContinuousJobSpec {
            context: StreamJobContext::new(
                79,
                plan.fingerprint(),
                JsonMap::new(),
                None,
                CancellationToken::new(),
            ),
            plan,
            sources: vec![NamedSourceBinding {
                binding_id: "input".into(),
                binding: counting_pending_binding(1, &polls, &source_closed),
            }],
            sinks: vec![NamedSinkBinding {
                output_id: "output".into(),
                sink_id: "blocked".into(),
                binding: OrdinarySinkBinding::new(Box::new(ProbeSink(sink.clone()))),
            }],
            edge_budget: EdgeBudget {
                max_rows: 1,
                max_bytes: 1 << 20,
            },
            delivery_mode: M2DeliveryMode::ProcessLocalOrdered,
        };
        let mut runner = ContinuousRunner::new();
        let observer = runner.start(job_spec);
        let core = Arc::clone(observer.core.as_ref().unwrap());
        let mut observer = Box::pin(observer);
        assert!(matches!(futures::poll!(observer.as_mut()), Poll::Pending));

        for _ in 0..100 {
            if resets.load(Ordering::SeqCst) == 2 && sink.opened.load(Ordering::SeqCst) == 1 {
                break;
            }
            tokio::task::yield_now().await;
        }
        assert_eq!(resets.load(Ordering::SeqCst), 2);
        assert_eq!(core.runtime_status.lock().tasks.snapshot().len(), 2);
        assert_eq!(
            core.state.lock().launch_delivery,
            super::LaunchDeliveryState::Provisional,
            "synchronous reset has no observer suspension state; entry and open remain provisional"
        );
        assert_eq!(polls.load(Ordering::SeqCst), 0);
        assert_eq!(processed.load(Ordering::SeqCst), 0);

        sink.open_release.notify_waiters();
        for _ in 0..100 {
            if core.state.lock().launch_delivery == super::LaunchDeliveryState::ReadyUnclaimed {
                break;
            }
            tokio::task::yield_now().await;
        }
        assert_eq!(
            core.state.lock().launch_delivery,
            super::LaunchDeliveryState::ReadyUnclaimed
        );
        assert_eq!(core.runtime_status.lock().tasks.snapshot().len(), 5);
        assert_eq!(polls.load(Ordering::SeqCst), 0);
        assert_eq!(processed.load(Ordering::SeqCst), 0);

        let Poll::Ready(Ok(job)) = futures::poll!(observer.as_mut()) else {
            panic!("ready-unclaimed start must deliver its handle in one poll");
        };
        drop(observer);
        assert_eq!(
            core.state.lock().launch_delivery,
            super::LaunchDeliveryState::Claimed
        );
        assert!(!core.launch_cancel.is_cancelled());
        assert_eq!(polls.load(Ordering::SeqCst), 0);
        assert_eq!(processed.load(Ordering::SeqCst), 0);
        for _ in 0..100 {
            if processed.load(Ordering::SeqCst) == 2 {
                break;
            }
            tokio::task::yield_now().await;
        }
        assert_eq!(processed.load(Ordering::SeqCst), 2);
        assert!(polls.load(Ordering::SeqCst) >= 2);

        let cancelled = job.cancel();
        assert!(
            !job.core.launch_cancel.is_cancelled(),
            "claimed jobs submit their cause before the driver cancels running tasks"
        );
        let outcome = cancelled.await;
        assert_eq!(outcome.cause, TerminalCause::ExplicitCancel);
        drop(job);
        runner.shutdown().await.unwrap();
    }

    #[test]
    fn explicit_cancel_wins_over_deadline_at_the_single_arbiter_commit() {
        let (commands, _commands_rx) = mpsc::unbounded_channel();
        let core = JobCore::new(
            LaunchId(0),
            91,
            commands,
            MetricsRecorder::default(),
            BTreeMap::new(),
        );
        core.terminal_arbiter.request_deadline();
        core.terminal_arbiter.request_explicit_cancel();
        let cancellation = CancellationToken::new();
        let observation = core.terminal_arbiter.observe_and_commit(&cancellation);

        assert_eq!(
            observation.terminal,
            Some(super::super::supervisor::TerminalDecision::ExplicitCancel)
        );
        assert!(cancellation.is_cancelled());
    }

    #[tokio::test]
    async fn explicit_recorded_before_linearized_deadline_commit_wins() {
        let mut runner = ContinuousRunner::new();
        let job = runner
            .start(spec(
                false,
                Arc::new(AtomicUsize::new(0)),
                LifecycleProbe::default(),
                LifecycleProbe::default(),
            ))
            .await
            .unwrap();
        let (commit_reached, release_commit) = job.core.install_terminal_commit_seam();

        job.core.request_deadline();
        commit_reached.await.unwrap();
        let cancelled = job.cancel();
        release_commit.send(()).unwrap();
        let outcome = cancelled.await;

        assert_eq!(outcome.state, ContinuousJobState::Cancelled);
        assert_eq!(outcome.cause, TerminalCause::ExplicitCancel);
        drop(job);
        runner.shutdown().await.unwrap();
    }

    #[test]
    fn primary_task_failure_wins_over_cancel_and_deadline_same_round() {
        let primary_task_id = TaskId::new(7);
        let report = SupervisionReport {
            primary_error_count: 1,
            errors: vec![super::super::supervisor::TaskFailure {
                task_id: primary_task_id,
                task_name: "operator:node".into(),
                error: CalcFlowError::Operator {
                    node_id: "node".into(),
                    message: "same-round failure".into(),
                },
            }],
        };
        let progress = RuntimeTaskProgress {
            sources: BTreeMap::new(),
            sinks: BTreeMap::new(),
        };

        let report = finish_running_report(
            LaunchId(0),
            None,
            false,
            report,
            &progress,
            &MetricsRecorder::default(),
        );

        let DriverCompletion::Outcome(outcome) = report.completion else {
            panic!("running driver must publish an outcome");
        };
        assert_eq!(
            outcome.cause,
            TerminalCause::TaskFailure { primary_task_id }
        );
        assert_eq!(outcome.state, ContinuousJobState::Failed);
    }

    #[tokio::test]
    async fn active_operator_cancelled_error_is_a_failed_task_failure() {
        let operator = ActiveCancelledOperator {
            inputs: [Port::new("input", BatchKind::Table, true, None).unwrap()],
            outputs: [Port::new("output", BatchKind::Table, true, None).unwrap()],
        };
        let plan = PipelineBuilder::new("active-cancelled")
            .unwrap()
            .add_node("node", Box::new(operator) as Box<dyn StreamOperator>)
            .unwrap()
            .compile_stream(
                &UdfRegistry::new().snapshot(),
                &StreamRequirements::default(),
            )
            .unwrap();
        let source_closed = Arc::new(AtomicUsize::new(0));
        let sink = LifecycleProbe::default();
        let job_spec = ContinuousJobSpec {
            context: StreamJobContext::new(
                95,
                plan.fingerprint(),
                JsonMap::new(),
                None,
                CancellationToken::new(),
            ),
            plan,
            sources: vec![NamedSourceBinding {
                binding_id: "input".into(),
                binding: finite_binding(&[1], &source_closed),
            }],
            sinks: vec![NamedSinkBinding {
                output_id: "output".into(),
                sink_id: "sink".into(),
                binding: OrdinarySinkBinding::new(Box::new(ProbeSink(sink.clone()))),
            }],
            edge_budget: EdgeBudget {
                max_rows: 1,
                max_bytes: 1 << 20,
            },
            delivery_mode: M2DeliveryMode::ProcessLocalOrdered,
        };
        let mut runner = ContinuousRunner::new();
        let job = runner.start(job_spec).await.unwrap();

        let outcome = job.wait().await;

        assert_eq!(outcome.state, ContinuousJobState::Failed);
        assert_eq!(
            outcome.cause,
            TerminalCause::TaskFailure {
                primary_task_id: TaskId::new(0)
            }
        );
        assert!(matches!(
            outcome.errors[0].error,
            CalcFlowError::Cancelled { ref run_id } if run_id == "operator-active-cancelled"
        ));
        assert_eq!(source_closed.load(Ordering::SeqCst), 1);
        assert_eq!(sink.closed.load(Ordering::SeqCst), 1);
        drop(job);
        runner.shutdown().await.unwrap();
    }

    #[test]
    fn recovery_classification_requires_both_allowed_origin_and_recoverable_error() {
        let classify = |origin, error| classify_failure_state(&RuntimeFailure { origin, error });
        let io_error = || CalcFlowError::Io {
            path: "connector".into(),
            source: std::io::Error::other("recoverable"),
        };
        let source_task = || RuntimeFailureOrigin::Task {
            task_id: TaskId::new(2),
            task_name: "source:left:pump".into(),
        };
        let operator_task = || RuntimeFailureOrigin::Task {
            task_id: TaskId::new(0),
            task_name: "operator:node".into(),
        };

        for origin in [
            RuntimeFailureOrigin::SourceOpen {
                binding_id: "left".into(),
            },
            RuntimeFailureOrigin::SourceClose {
                binding_id: "left".into(),
            },
            RuntimeFailureOrigin::SinkOpen {
                output_id: "output".into(),
                sink_id: "sink".into(),
            },
            RuntimeFailureOrigin::SinkClose {
                output_id: "output".into(),
                sink_id: "sink".into(),
            },
            RuntimeFailureOrigin::SinkWrite {
                output_id: "output".into(),
                sink_id: "sink".into(),
            },
            source_task(),
        ] {
            assert_eq!(
                classify(origin, io_error()),
                ContinuousJobState::RecoveryRequired
            );
        }

        for origin in [
            RuntimeFailureOrigin::Preflight,
            RuntimeFailureOrigin::OperatorEntry {
                node_id: "node".into(),
            },
            RuntimeFailureOrigin::SinkIngress {
                output_id: "output".into(),
                edge_id: "edge".into(),
            },
            operator_task(),
            RuntimeFailureOrigin::Metrics {
                component_id: "job".into(),
                counter: "errors",
            },
        ] {
            assert_eq!(classify(origin, io_error()), ContinuousJobState::Failed);
        }
        assert_eq!(
            classify(
                source_task(),
                CalcFlowError::Cancelled {
                    run_id: "active-source".into(),
                }
            ),
            ContinuousJobState::Failed
        );
        assert_eq!(
            classify(
                RuntimeFailureOrigin::SourceOpen {
                    binding_id: "left".into(),
                },
                CalcFlowError::Internal {
                    message: "not recoverable".into(),
                }
            ),
            ContinuousJobState::Failed
        );
    }

    #[tokio::test]
    async fn terminal_observers_are_idempotent_and_dropped_wait_does_not_cancel() {
        let mut runner = ContinuousRunner::new();
        let source = LifecycleProbe::default();
        let sink = LifecycleProbe::default();
        let job = runner
            .start(spec(
                false,
                Arc::new(AtomicUsize::new(0)),
                source.clone(),
                sink.clone(),
            ))
            .await
            .unwrap();
        let mut wait = Box::pin(job.wait());
        futures::future::poll_fn(|context| match wait.as_mut().poll(context) {
            Poll::Pending => Poll::Ready(()),
            Poll::Ready(_) => panic!("running job completed before a terminal cause"),
        })
        .await;
        drop(wait);

        let cancelled = job.cancel().await;
        let observed = job.wait().await;

        assert!(Arc::ptr_eq(&cancelled, &observed));
        assert_eq!(cancelled.state, ContinuousJobState::Cancelled);
        assert_eq!(cancelled.cause, TerminalCause::ExplicitCancel);
        assert_eq!(job.driver_owner(), DriverOwnership::Terminal);
        assert_eq!(source.closed.load(Ordering::SeqCst), 1);
        assert_eq!(sink.closed.load(Ordering::SeqCst), 1);
        drop(job);
        runner.shutdown().await.unwrap();
    }

    #[tokio::test]
    async fn dropped_cancel_and_shutdown_observers_preserve_driver_and_join_ownership() {
        for (label, cancel, drop_during_close) in [
            ("cancel-driver", true, false),
            ("shutdown-driver", false, false),
            ("cancel-join", true, true),
            ("shutdown-join", false, true),
        ] {
            let mut runner = ContinuousRunner::new();
            let source = LifecycleProbe::default();
            source
                .block_close
                .store(drop_during_close, Ordering::SeqCst);
            let sink = LifecycleProbe::default();
            let job = runner
                .start(spec(
                    false,
                    Arc::new(AtomicUsize::new(0)),
                    source.clone(),
                    sink.clone(),
                ))
                .await
                .unwrap();
            let mut observer = Box::pin(if cancel { job.cancel() } else { job.shutdown() });

            assert!(
                matches!(futures::poll!(observer.as_mut()), Poll::Pending),
                "terminal observer completed before the driver linearization point: {label}"
            );
            if drop_during_close {
                for _ in 0..100 {
                    if source.closed.load(Ordering::SeqCst) == 1 {
                        break;
                    }
                    tokio::task::yield_now().await;
                }
                assert_eq!(source.closed.load(Ordering::SeqCst), 1, "{label}");
                assert!(
                    matches!(futures::poll!(observer.as_mut()), Poll::Pending),
                    "terminal observer completed before connector join: {label}"
                );
            }

            drop(observer);
            assert_eq!(runner.registry_counts().0, 1, "registry lost at {label}");
            if drop_during_close {
                source.close_release.notify_waiters();
            }
            let outcome = job.wait().await;
            assert_eq!(
                outcome.cause,
                if cancel {
                    TerminalCause::ExplicitCancel
                } else {
                    TerminalCause::GracefulShutdown
                },
                "{label}"
            );
            assert!(job.status().tasks.is_empty(), "{label}");
            assert_eq!(source.closed.load(Ordering::SeqCst), 1, "{label}");
            assert_eq!(sink.closed.load(Ordering::SeqCst), 1, "{label}");
            drop(job);
            runner.shutdown().await.unwrap();
            assert_eq!(runner.registry_counts(), (0, 0), "{label}");
        }
    }

    #[tokio::test]
    async fn dropped_runner_shutdown_observer_leaves_join_handle_for_retry() {
        let release = Arc::new(Notify::new());
        let completed = Arc::new(AtomicBool::new(false));
        let release_in_driver = Arc::clone(&release);
        let completed_in_driver = Arc::clone(&completed);
        let driver = tokio::spawn(async move {
            release_in_driver.notified().await;
            completed_in_driver.store(true, Ordering::SeqCst);
        });
        let (commands, _commands_rx) = mpsc::unbounded_channel();
        let core = Arc::new(RunnerCore {
            commands,
            root_cancel: CancellationToken::new(),
            registry: Mutex::new(RunnerRegistryState {
                provisional: None,
                live_jobs: BTreeMap::new(),
                reaper_jobs: BTreeSet::new(),
                pending_start: None,
                shutting_down: true,
            }),
            driver: Mutex::new(Some(driver)),
            diagnostics: RunnerDiagnostics::default(),
            next_launch_id: AtomicU64::new(0),
            closed: AtomicBool::new(true),
            changed: Notify::new(),
            abandonment_warnings: AtomicU64::new(0),
        });
        let mut first = Box::pin(RunnerShutdownObserver::new(Arc::clone(&core)));

        assert!(matches!(futures::poll!(first.as_mut()), Poll::Pending));
        drop(first);

        assert!(
            core.driver.lock().is_some(),
            "a dropped shutdown observer detached the core-owned lifecycle handle"
        );
        release.notify_one();
        RunnerShutdownObserver::new(Arc::clone(&core))
            .await
            .unwrap();
        assert!(completed.load(Ordering::SeqCst));
        assert!(core.driver.lock().is_none());
    }

    #[tokio::test]
    async fn dropped_real_runner_shutdown_observer_is_joined_by_retry() {
        let mut runner = ContinuousRunner::new();
        let source = LifecycleProbe::default();
        source.block_close.store(true, Ordering::SeqCst);
        let sink = LifecycleProbe::default();
        let job = runner
            .start(spec(
                false,
                Arc::new(AtomicUsize::new(0)),
                source.clone(),
                sink.clone(),
            ))
            .await
            .unwrap();
        let core = Arc::clone(&runner.core);
        let mut shutdown = Box::pin(runner.shutdown());

        assert!(matches!(futures::poll!(shutdown.as_mut()), Poll::Pending));
        drop(shutdown);
        for _ in 0..100 {
            if source.closed.load(Ordering::SeqCst) == 1 {
                break;
            }
            tokio::task::yield_now().await;
        }
        assert_eq!(source.closed.load(Ordering::SeqCst), 1);
        assert!(core.driver.lock().is_some());

        source.close_release.notify_waiters();
        runner.shutdown().await.unwrap();
        assert!(core.driver.lock().is_none());
        assert_eq!(job.wait().await.cause, TerminalCause::ExplicitCancel);
        assert_eq!(source.closed.load(Ordering::SeqCst), 1);
        assert_eq!(sink.closed.load(Ordering::SeqCst), 1);
    }

    #[tokio::test]
    async fn dropping_start_during_open_closes_once_and_next_start_waits_for_reaper() {
        let mut runner = ContinuousRunner::new();
        let blocked_source = LifecycleProbe::default();
        blocked_source.block_open.store(true, Ordering::SeqCst);
        let blocked_sink = LifecycleProbe::default();
        let opened = blocked_source.open_started.notified();
        tokio::pin!(opened);
        let mut start = Box::pin(runner.start(spec(
            false,
            Arc::new(AtomicUsize::new(0)),
            blocked_source.clone(),
            blocked_sink.clone(),
        )));
        tokio::select! {
            result = &mut start => panic!("start completed before the open gate: {result:?}"),
            () = &mut opened => {}
        }
        drop(start);

        let source = LifecycleProbe::default();
        let sink = LifecycleProbe::default();
        let next = runner
            .start(spec(false, Arc::new(AtomicUsize::new(0)), source, sink))
            .await
            .unwrap();

        assert_eq!(blocked_source.opened.load(Ordering::SeqCst), 1);
        assert_eq!(blocked_source.closed.load(Ordering::SeqCst), 1);
        assert_eq!(blocked_sink.closed.load(Ordering::SeqCst), 1);
        drop(next);
        runner.shutdown().await.unwrap();
    }

    #[tokio::test]
    async fn dropping_start_immediately_after_provisional_registration_is_reaped() {
        let mut runner = ContinuousRunner::new();
        let source = LifecycleProbe::default();
        let sink = LifecycleProbe::default();
        let observer = runner.start(spec(
            false,
            Arc::new(AtomicUsize::new(0)),
            source.clone(),
            sink.clone(),
        ));
        let provisional = Arc::clone(observer.core.as_ref().unwrap());
        assert_eq!(
            provisional.state.lock().launch_delivery,
            super::LaunchDeliveryState::Provisional
        );

        drop(observer);

        assert!(provisional.launch_cancel.is_cancelled());
        assert_eq!(provisional.state.lock().owner, DriverOwnership::ReaperOwned);
        let next = runner
            .start(spec(
                false,
                Arc::new(AtomicUsize::new(0)),
                LifecycleProbe::default(),
                LifecycleProbe::default(),
            ))
            .await
            .unwrap();
        assert_eq!(provisional.state.lock().owner, DriverOwnership::Terminal);
        assert_eq!(source.opened.load(Ordering::SeqCst), 0);
        assert_eq!(sink.opened.load(Ordering::SeqCst), 0);
        drop(next);
        runner.shutdown().await.unwrap();
    }

    #[tokio::test]
    async fn dropping_start_after_one_of_two_sources_opens_reaps_every_begun_connector() {
        let plan = union_plan();
        let left = LifecycleProbe::default();
        let right = LifecycleProbe::default();
        right.block_open.store(true, Ordering::SeqCst);
        let sink = LifecycleProbe::default();
        let job_spec = ContinuousJobSpec {
            context: StreamJobContext::new(
                80,
                plan.fingerprint(),
                JsonMap::new(),
                None,
                CancellationToken::new(),
            ),
            plan,
            sources: vec![
                NamedSourceBinding {
                    binding_id: "left".into(),
                    binding: SourceBinding::new(Box::new(ProbeSource(left.clone())), None, 0)
                        .unwrap(),
                },
                NamedSourceBinding {
                    binding_id: "right".into(),
                    binding: SourceBinding::new(Box::new(ProbeSource(right.clone())), None, 0)
                        .unwrap(),
                },
            ],
            sinks: vec![NamedSinkBinding {
                output_id: "output".into(),
                sink_id: "sink".into(),
                binding: OrdinarySinkBinding::new(Box::new(ProbeSink(sink.clone()))),
            }],
            edge_budget: EdgeBudget::default(),
            delivery_mode: M2DeliveryMode::ProcessLocalOrdered,
        };
        let mut runner = ContinuousRunner::new();
        let observer = runner.start(job_spec);
        let provisional = Arc::clone(observer.core.as_ref().unwrap());
        let mut observer = Box::pin(observer);
        assert!(matches!(futures::poll!(observer.as_mut()), Poll::Pending));
        for _ in 0..100 {
            if left.open_completed.load(Ordering::SeqCst) == 1
                && right.opened.load(Ordering::SeqCst) == 1
            {
                break;
            }
            tokio::task::yield_now().await;
        }
        assert_eq!(left.open_completed.load(Ordering::SeqCst), 1);
        assert_eq!(right.opened.load(Ordering::SeqCst), 1);
        assert_eq!(right.open_completed.load(Ordering::SeqCst), 0);
        assert_eq!(
            provisional.state.lock().launch_delivery,
            super::LaunchDeliveryState::Provisional
        );

        drop(observer);

        let next = runner
            .start(spec(
                false,
                Arc::new(AtomicUsize::new(0)),
                LifecycleProbe::default(),
                LifecycleProbe::default(),
            ))
            .await
            .unwrap();
        assert_eq!(provisional.state.lock().owner, DriverOwnership::Terminal);
        assert_eq!(left.closed.load(Ordering::SeqCst), 1);
        assert_eq!(right.closed.load(Ordering::SeqCst), 1);
        assert_eq!(sink.closed.load(Ordering::SeqCst), 1);
        drop(next);
        runner.shutdown().await.unwrap();
    }

    #[tokio::test]
    async fn dropping_ready_unclaimed_start_never_releases_the_data_gate() {
        let plan = unary_expression_plan();
        let polls = Arc::new(AtomicUsize::new(0));
        let source_closed = Arc::new(AtomicUsize::new(0));
        let sink = LifecycleProbe::default();
        let job_spec = ContinuousJobSpec {
            context: StreamJobContext::new(
                89,
                plan.fingerprint(),
                JsonMap::new(),
                None,
                CancellationToken::new(),
            ),
            plan,
            sources: vec![NamedSourceBinding {
                binding_id: "input".into(),
                binding: counting_pending_binding(1, &polls, &source_closed),
            }],
            sinks: vec![NamedSinkBinding {
                output_id: "output".into(),
                sink_id: "sink".into(),
                binding: OrdinarySinkBinding::new(Box::new(ProbeSink(sink.clone()))),
            }],
            edge_budget: EdgeBudget::default(),
            delivery_mode: M2DeliveryMode::ProcessLocalOrdered,
        };
        let mut runner = ContinuousRunner::new();
        let observer = runner.start(job_spec);
        let provisional = Arc::clone(observer.core.as_ref().unwrap());
        for _ in 0..100 {
            if provisional.state.lock().launch_delivery
                == super::LaunchDeliveryState::ReadyUnclaimed
            {
                break;
            }
            tokio::task::yield_now().await;
        }
        assert_eq!(
            provisional.state.lock().launch_delivery,
            super::LaunchDeliveryState::ReadyUnclaimed
        );
        assert_eq!(polls.load(Ordering::SeqCst), 0);

        drop(observer);

        let next = runner
            .start(spec(
                false,
                Arc::new(AtomicUsize::new(0)),
                LifecycleProbe::default(),
                LifecycleProbe::default(),
            ))
            .await
            .unwrap();
        assert_eq!(polls.load(Ordering::SeqCst), 0);
        assert_eq!(source_closed.load(Ordering::SeqCst), 1);
        assert_eq!(sink.closed.load(Ordering::SeqCst), 1);
        assert_eq!(provisional.state.lock().owner, DriverOwnership::Terminal);
        drop(next);
        runner.shutdown().await.unwrap();
    }

    #[tokio::test]
    async fn connector_open_failure_closes_every_begun_resource_once() {
        let mut runner = ContinuousRunner::new();
        let source = LifecycleProbe::default();
        source.fail_open.store(true, Ordering::SeqCst);
        let sink = LifecycleProbe::default();

        let failure = runner
            .start(spec(
                false,
                Arc::new(AtomicUsize::new(0)),
                source.clone(),
                sink.clone(),
            ))
            .await
            .unwrap_err();

        assert!(matches!(
            failure.primary.origin,
            super::FailureOrigin::SourceOpen { .. }
        ));
        assert_eq!(source.opened.load(Ordering::SeqCst), 1);
        assert_eq!(source.closed.load(Ordering::SeqCst), 1);
        assert_eq!(sink.opened.load(Ordering::SeqCst), 1);
        assert_eq!(sink.closed.load(Ordering::SeqCst), 1);
        runner.shutdown().await.unwrap();
    }

    #[tokio::test]
    async fn connector_open_panic_is_typed_and_closes_every_begun_resource_once() {
        let mut runner = ContinuousRunner::new();
        let source = LifecycleProbe::default();
        source.panic_open.store(true, Ordering::SeqCst);
        let sink = LifecycleProbe::default();

        let failure = runner
            .start(spec(
                false,
                Arc::new(AtomicUsize::new(0)),
                source.clone(),
                sink.clone(),
            ))
            .await
            .unwrap_err();

        assert!(matches!(
            failure.primary.origin,
            super::FailureOrigin::SourceOpen { ref binding_id } if binding_id == "input"
        ));
        assert!(matches!(
            failure.primary.error,
            CalcFlowError::TaskPanicked { task_id: 0, ref message }
                if message == "source open panicked"
        ));
        assert_eq!(source.opened.load(Ordering::SeqCst), 1);
        assert_eq!(source.closed.load(Ordering::SeqCst), 1);
        assert_eq!(sink.opened.load(Ordering::SeqCst), 1);
        assert_eq!(sink.closed.load(Ordering::SeqCst), 1);
        runner.shutdown().await.unwrap();
    }

    #[tokio::test]
    async fn source_next_panic_is_typed_and_source_and_sink_close_once() {
        let mut runner = ContinuousRunner::new();
        let source = LifecycleProbe::default();
        source.panic_next.store(true, Ordering::SeqCst);
        let sink = LifecycleProbe::default();
        let job = runner
            .start(spec(
                false,
                Arc::new(AtomicUsize::new(0)),
                source.clone(),
                sink.clone(),
            ))
            .await
            .unwrap();

        let outcome = job.wait().await;

        assert_eq!(outcome.state, ContinuousJobState::Failed);
        assert_eq!(
            outcome.cause,
            TerminalCause::TaskFailure {
                primary_task_id: TaskId::new(1)
            }
        );
        assert!(matches!(
            outcome.errors[0].error,
            CalcFlowError::TaskPanicked { task_id: 1, ref message }
                if message == "source next panicked"
        ));
        assert_eq!(source.closed.load(Ordering::SeqCst), 1);
        assert_eq!(sink.closed.load(Ordering::SeqCst), 1);
        drop(job);
        runner.shutdown().await.unwrap();
    }

    #[tokio::test]
    async fn sink_write_panic_is_typed_and_all_connectors_close_once() {
        let source_closed = Arc::new(AtomicUsize::new(0));
        let panic_sink = LifecycleProbe::default();
        panic_sink.panic_write.store(true, Ordering::SeqCst);
        let sibling_sink = LifecycleProbe::default();
        let job_spec = forward_spec(
            96,
            finite_binding(&[1], &source_closed),
            vec![
                named_probe_sink("a-panic", panic_sink.clone()),
                named_probe_sink("b-sibling", sibling_sink.clone()),
            ],
        );
        let mut runner = ContinuousRunner::new();
        let job = runner.start(job_spec).await.unwrap();

        let outcome = job.wait().await;

        assert_eq!(outcome.state, ContinuousJobState::Failed);
        assert_eq!(
            outcome.cause,
            TerminalCause::TaskFailure {
                primary_task_id: TaskId::new(3)
            }
        );
        assert!(matches!(
            outcome.errors[0].origin,
            super::FailureOrigin::SinkWrite { ref sink_id, .. } if sink_id == "a-panic"
        ));
        assert!(matches!(
            outcome.errors[0].error,
            CalcFlowError::TaskPanicked { task_id: 3, ref message }
                if message == "sink write panicked"
        ));
        assert_eq!(source_closed.load(Ordering::SeqCst), 1);
        assert_eq!(panic_sink.closed.load(Ordering::SeqCst), 1);
        assert_eq!(sibling_sink.closed.load(Ordering::SeqCst), 1);
        drop(job);
        runner.shutdown().await.unwrap();
    }

    #[tokio::test]
    async fn sink_close_panic_is_typed_and_does_not_skip_later_sink_close() {
        let source_closed = Arc::new(AtomicUsize::new(0));
        let panic_sink = LifecycleProbe::default();
        panic_sink.panic_close.store(true, Ordering::SeqCst);
        let sibling_sink = LifecycleProbe::default();
        let job_spec = forward_spec(
            97,
            finite_binding(&[1], &source_closed),
            vec![
                named_probe_sink("a-panic", panic_sink.clone()),
                named_probe_sink("b-sibling", sibling_sink.clone()),
            ],
        );
        let mut runner = ContinuousRunner::new();
        let job = runner.start(job_spec).await.unwrap();

        let outcome = job.wait().await;

        assert_eq!(outcome.state, ContinuousJobState::Failed);
        assert_eq!(
            outcome.cause,
            TerminalCause::TaskFailure {
                primary_task_id: TaskId::new(3)
            }
        );
        assert!(matches!(
            outcome.errors[0].origin,
            super::FailureOrigin::SinkClose { ref sink_id, .. } if sink_id == "a-panic"
        ));
        assert!(matches!(
            outcome.errors[0].error,
            CalcFlowError::TaskPanicked { task_id: 3, ref message }
                if message == "sink close panicked"
        ));
        assert_eq!(source_closed.load(Ordering::SeqCst), 1);
        assert_eq!(panic_sink.closed.load(Ordering::SeqCst), 1);
        assert_eq!(sibling_sink.closed.load(Ordering::SeqCst), 1);
        drop(job);
        runner.shutdown().await.unwrap();
    }

    #[tokio::test]
    async fn sink_open_panic_is_typed_and_closes_every_begun_resource_once() {
        let mut runner = ContinuousRunner::new();
        let source = LifecycleProbe::default();
        let sink = LifecycleProbe::default();
        sink.panic_open.store(true, Ordering::SeqCst);

        let failure = runner
            .start(spec(
                false,
                Arc::new(AtomicUsize::new(0)),
                source.clone(),
                sink.clone(),
            ))
            .await
            .unwrap_err();

        assert!(matches!(
            failure.primary.origin,
            super::FailureOrigin::SinkOpen { ref output_id, ref sink_id }
                if output_id == "output" && sink_id == "sink"
        ));
        assert!(matches!(
            failure.primary.error,
            CalcFlowError::TaskPanicked { task_id: 1, ref message }
                if message == "sink open panicked"
        ));
        assert_eq!(source.opened.load(Ordering::SeqCst), 1);
        assert_eq!(source.closed.load(Ordering::SeqCst), 1);
        assert_eq!(sink.opened.load(Ordering::SeqCst), 1);
        assert_eq!(sink.closed.load(Ordering::SeqCst), 1);
        runner.shutdown().await.unwrap();
    }

    #[tokio::test]
    async fn source_open_primary_keeps_close_failure_in_stable_diagnostics() {
        let mut runner = ContinuousRunner::new();
        let source = LifecycleProbe::default();
        source.fail_open.store(true, Ordering::SeqCst);
        source.fail_close.store(true, Ordering::SeqCst);

        let failure = runner
            .start(spec(
                false,
                Arc::new(AtomicUsize::new(0)),
                source.clone(),
                LifecycleProbe::default(),
            ))
            .await
            .unwrap_err();

        assert!(matches!(
            failure.primary.origin,
            super::FailureOrigin::SourceOpen { ref binding_id } if binding_id == "input"
        ));
        assert!(matches!(
            failure.primary.error,
            CalcFlowError::Internal { ref message } if message == "source open failed"
        ));
        let diagnostic_id = failure
            .diagnostic_id
            .expect("source close failure must be retained in diagnostics");
        let diagnostics = runner.diagnostics();
        let record = diagnostics
            .records
            .iter()
            .find(|record| record.id == diagnostic_id)
            .unwrap();
        assert_eq!(record.cleanup_failures.len(), 1);
        assert!(matches!(
            record.cleanup_failures[0].origin,
            super::FailureOrigin::SourceClose { ref binding_id } if binding_id == "input"
        ));
        assert!(matches!(
            record.cleanup_failures[0].error,
            CalcFlowError::Internal { ref message } if message == "source close failed"
        ));
        assert_eq!(source.closed.load(Ordering::SeqCst), 1);
        runner.shutdown().await.unwrap();
    }

    #[tokio::test]
    async fn running_source_primary_errors_keep_close_as_stable_secondary() {
        for (job_id, failure, expected_task, expected_primary) in [
            (
                86,
                RunningSourceFailure::Next,
                "source:input:pump",
                "source-next-primary",
            ),
            (
                87,
                RunningSourceFailure::Cursor,
                "source:input:task",
                "sources.input.cursor",
            ),
        ] {
            let plan = unary_expression_plan();
            let source_closed = Arc::new(AtomicUsize::new(0));
            let sink_closed = Arc::new(AtomicUsize::new(0));
            let spec = ContinuousJobSpec {
                context: StreamJobContext::new(
                    job_id,
                    plan.fingerprint(),
                    JsonMap::new(),
                    None,
                    CancellationToken::new(),
                ),
                plan,
                sources: vec![NamedSourceBinding {
                    binding_id: "input".into(),
                    binding: SourceBinding::new(
                        Box::new(PrimaryAndCloseFailingSource {
                            failure,
                            next_call: 0,
                            closed: Arc::clone(&source_closed),
                        }),
                        None,
                        0,
                    )
                    .unwrap(),
                }],
                sinks: vec![NamedSinkBinding {
                    output_id: "output".into(),
                    sink_id: "recording".into(),
                    binding: OrdinarySinkBinding::new(Box::new(OrderedRecordingSink {
                        id: "recording".into(),
                        writes: Arc::new(Mutex::new(Vec::new())),
                        closed: Arc::clone(&sink_closed),
                    })),
                }],
                edge_budget: EdgeBudget {
                    max_rows: 1,
                    max_bytes: 1 << 20,
                },
                delivery_mode: M2DeliveryMode::ProcessLocalOrdered,
            };
            let mut runner = ContinuousRunner::new();
            let job = runner.start(spec).await.unwrap();
            let outcome = job.wait().await;

            assert_eq!(outcome.state, ContinuousJobState::Failed);
            assert!(matches!(outcome.cause, TerminalCause::TaskFailure { .. }));
            assert_eq!(outcome.errors.len(), 2, "errors: {:?}", outcome.errors);
            assert!(matches!(
                outcome.errors[0].origin,
                super::FailureOrigin::Task { ref task_name, .. } if task_name == expected_task
            ));
            match &outcome.errors[0].error {
                CalcFlowError::Internal { message } => assert_eq!(message, expected_primary),
                CalcFlowError::InvalidArgument { field, .. } => {
                    assert_eq!(field, expected_primary);
                }
                error => panic!("unexpected source primary: {error:?}"),
            }
            assert!(matches!(
                outcome.errors[1].origin,
                super::FailureOrigin::SourceClose { ref binding_id } if binding_id == "input"
            ));
            assert!(matches!(
                outcome.errors[1].error,
                CalcFlowError::Internal { ref message } if message == "source-close-secondary"
            ));
            assert_eq!(source_closed.load(Ordering::SeqCst), 1);
            assert_eq!(sink_closed.load(Ordering::SeqCst), 1);
            drop(job);
            runner.shutdown().await.unwrap();
        }
    }

    #[tokio::test]
    async fn graceful_shutdown_and_deadline_have_explicit_distinct_causes() {
        let mut runner = ContinuousRunner::new();
        let job = runner
            .start(spec(
                false,
                Arc::new(AtomicUsize::new(0)),
                LifecycleProbe::default(),
                LifecycleProbe::default(),
            ))
            .await
            .unwrap();

        let drained = job.shutdown().await;

        assert_eq!(drained.state, ContinuousJobState::Completed);
        assert_eq!(drained.cause, TerminalCause::GracefulShutdown);
        drop(job);

        let mut deadline_spec = spec(
            false,
            Arc::new(AtomicUsize::new(0)),
            LifecycleProbe::default(),
            LifecycleProbe::default(),
        );
        deadline_spec.context = StreamJobContext::new(
            10,
            deadline_spec.plan.fingerprint(),
            JsonMap::new(),
            Some(chrono::Utc::now() - chrono::Duration::milliseconds(1)),
            CancellationToken::new(),
        );
        let deadline_job = runner.start(deadline_spec).await.unwrap();
        let deadline = deadline_job.wait().await;

        assert_eq!(deadline.state, ContinuousJobState::Cancelled);
        assert_eq!(deadline.cause, TerminalCause::DeadlineExceeded);
        assert!(deadline.errors.is_empty());
        drop(deadline_job);
        runner.shutdown().await.unwrap();
    }

    #[tokio::test]
    async fn explicit_cancel_keeps_source_and_sink_close_errors_secondary() {
        let mut runner = ContinuousRunner::new();
        let source = LifecycleProbe::default();
        source.fail_close.store(true, Ordering::SeqCst);
        let sink = LifecycleProbe::default();
        sink.fail_close.store(true, Ordering::SeqCst);
        let job = runner
            .start(spec(
                false,
                Arc::new(AtomicUsize::new(0)),
                source.clone(),
                sink.clone(),
            ))
            .await
            .unwrap();

        let outcome = job.cancel().await;

        assert_eq!(outcome.state, ContinuousJobState::Cancelled);
        assert_eq!(outcome.cause, TerminalCause::ExplicitCancel);
        assert_eq!(outcome.errors.len(), 2);
        assert!(matches!(
            outcome.errors[0].origin,
            super::FailureOrigin::SourceClose { ref binding_id } if binding_id == "input"
        ));
        assert!(matches!(
            outcome.errors[1].origin,
            super::FailureOrigin::SinkClose { ref output_id, ref sink_id }
                if output_id == "output" && sink_id == "sink"
        ));
        assert_eq!(source.closed.load(Ordering::SeqCst), 1);
        assert_eq!(sink.closed.load(Ordering::SeqCst), 1);
        drop(job);
        runner.shutdown().await.unwrap();
    }

    #[tokio::test]
    async fn deadline_keeps_source_and_sink_close_errors_secondary() {
        let mut runner = ContinuousRunner::new();
        let source = LifecycleProbe::default();
        source.fail_close.store(true, Ordering::SeqCst);
        let sink = LifecycleProbe::default();
        sink.fail_close.store(true, Ordering::SeqCst);
        let mut deadline_spec = spec(
            false,
            Arc::new(AtomicUsize::new(0)),
            source.clone(),
            sink.clone(),
        );
        deadline_spec.context = StreamJobContext::new(
            10,
            deadline_spec.plan.fingerprint(),
            JsonMap::new(),
            Some(chrono::Utc::now() - chrono::Duration::milliseconds(1)),
            CancellationToken::new(),
        );
        let job = runner.start(deadline_spec).await.unwrap();

        let outcome = job.wait().await;

        assert_eq!(outcome.state, ContinuousJobState::Cancelled);
        assert_eq!(outcome.cause, TerminalCause::DeadlineExceeded);
        assert_eq!(outcome.errors.len(), 2);
        assert!(matches!(
            outcome.errors[0].origin,
            super::FailureOrigin::SourceClose { ref binding_id } if binding_id == "input"
        ));
        assert!(matches!(
            outcome.errors[1].origin,
            super::FailureOrigin::SinkClose { ref output_id, ref sink_id }
                if output_id == "output" && sink_id == "sink"
        ));
        assert_eq!(source.closed.load(Ordering::SeqCst), 1);
        assert_eq!(sink.closed.load(Ordering::SeqCst), 1);
        drop(job);
        runner.shutdown().await.unwrap();
    }

    #[tokio::test]
    async fn committed_deadline_is_immutable_while_connector_close_is_blocked() {
        let mut runner = ContinuousRunner::new();
        let source = LifecycleProbe::default();
        source.block_close.store(true, Ordering::SeqCst);
        let sink = LifecycleProbe::default();
        sink.block_close.store(true, Ordering::SeqCst);
        let mut deadline_spec = spec(
            false,
            Arc::new(AtomicUsize::new(0)),
            source.clone(),
            sink.clone(),
        );
        deadline_spec.context = StreamJobContext::new(
            94,
            deadline_spec.plan.fingerprint(),
            JsonMap::new(),
            Some(chrono::Utc::now() - chrono::Duration::milliseconds(1)),
            CancellationToken::new(),
        );
        let job = runner.start(deadline_spec).await.unwrap();

        while job.status().terminal_cause != Some(TerminalCause::DeadlineExceeded)
            || source.closed.load(Ordering::SeqCst) != 1
            || sink.closed.load(Ordering::SeqCst) != 1
        {
            tokio::task::yield_now().await;
        }
        let cancelled = job.cancel();
        assert_eq!(
            job.status().terminal_cause,
            Some(TerminalCause::DeadlineExceeded)
        );

        source.close_release.notify_one();
        sink.close_release.notify_one();
        let outcome = cancelled.await;

        assert_eq!(outcome.state, ContinuousJobState::Cancelled);
        assert_eq!(outcome.cause, TerminalCause::DeadlineExceeded);
        assert_eq!(source.closed.load(Ordering::SeqCst), 1);
        assert_eq!(sink.closed.load(Ordering::SeqCst), 1);
        drop(job);
        runner.shutdown().await.unwrap();
    }

    #[tokio::test]
    async fn context_task_handle_status_and_diagnostics_share_the_context_job_id() {
        let expected_job_id = 42_424;
        let observed_job_id = Arc::new(Mutex::new(None));
        let operator = JobIdentityProbeOperator {
            inputs: [Port::new("input", BatchKind::Table, true, None).unwrap()],
            outputs: [Port::new("output", BatchKind::Table, false, None).unwrap()],
            observed_job_id: Arc::clone(&observed_job_id),
        };
        let plan = PipelineBuilder::new("job-identity")
            .unwrap()
            .add_node("node", Box::new(operator) as Box<dyn StreamOperator>)
            .unwrap()
            .compile_stream(
                &UdfRegistry::new().snapshot(),
                &StreamRequirements::default(),
            )
            .unwrap();
        let source_closed = Arc::new(AtomicUsize::new(0));
        let sink = LifecycleProbe::default();
        let identity_spec = ContinuousJobSpec {
            context: StreamJobContext::new(
                expected_job_id,
                plan.fingerprint(),
                JsonMap::new(),
                None,
                CancellationToken::new(),
            ),
            plan,
            sources: vec![NamedSourceBinding {
                binding_id: "input".into(),
                binding: finite_binding(&[1], &source_closed),
            }],
            sinks: vec![NamedSinkBinding {
                output_id: "output".into(),
                sink_id: "sink".into(),
                binding: OrdinarySinkBinding::new(Box::new(ProbeSink(sink.clone()))),
            }],
            edge_budget: EdgeBudget {
                max_rows: 1,
                max_bytes: 1 << 20,
            },
            delivery_mode: M2DeliveryMode::ProcessLocalOrdered,
        };
        let mut runner = ContinuousRunner::new();
        let job = runner.start(identity_spec).await.unwrap();

        let outcome = job.wait().await;

        assert_eq!(outcome.state, ContinuousJobState::Completed);
        assert_eq!(job.id(), expected_job_id);
        assert_eq!(job.status().job_id, expected_job_id);
        assert_eq!(*observed_job_id.lock(), Some(expected_job_id));
        drop(job);

        let failed_job_id = expected_job_id + 1;
        let source = LifecycleProbe::default();
        source.fail_open.store(true, Ordering::SeqCst);
        source.fail_close.store(true, Ordering::SeqCst);
        let mut failed_spec = spec(
            false,
            Arc::new(AtomicUsize::new(0)),
            source,
            LifecycleProbe::default(),
        );
        failed_spec.context = StreamJobContext::new(
            failed_job_id,
            failed_spec.plan.fingerprint(),
            JsonMap::new(),
            None,
            CancellationToken::new(),
        );
        let failed = runner.start(failed_spec);
        let failed_launch_id = failed.core.as_ref().unwrap().launch_id;
        let failure = failed.await.unwrap_err();
        let diagnostic_id = failure.diagnostic_id.unwrap();
        let diagnostics = runner.diagnostics();
        let diagnostic = diagnostics
            .records
            .iter()
            .find(|record| record.id == diagnostic_id)
            .unwrap();
        assert_eq!(diagnostic.launch_id, failed_launch_id);
        runner.shutdown().await.unwrap();
    }

    #[tokio::test]
    async fn reused_context_job_id_has_distinct_private_diagnostic_launch_ids() {
        let context_job_id = 42_425;
        let mut runner = ContinuousRunner::new();
        let mut observed = Vec::new();
        for _ in 0..2 {
            let source = LifecycleProbe::default();
            source.fail_open.store(true, Ordering::SeqCst);
            source.fail_close.store(true, Ordering::SeqCst);
            let mut failed_spec = spec(
                false,
                Arc::new(AtomicUsize::new(0)),
                source,
                LifecycleProbe::default(),
            );
            failed_spec.context = StreamJobContext::new(
                context_job_id,
                failed_spec.plan.fingerprint(),
                JsonMap::new(),
                None,
                CancellationToken::new(),
            );
            let start = runner.start(failed_spec);
            let launch_id = start.core.as_ref().unwrap().launch_id;
            let failure = start.await.unwrap_err();
            observed.push((launch_id, failure.diagnostic_id.unwrap()));
        }

        assert_ne!(observed[0].0, observed[1].0);
        let diagnostics = runner.diagnostics();
        let diagnostic_launch_ids = observed
            .iter()
            .map(|(launch_id, diagnostic_id)| {
                let record = diagnostics
                    .records
                    .iter()
                    .find(|record| record.id == *diagnostic_id)
                    .unwrap();
                assert_eq!(record.launch_id, *launch_id);
                record.launch_id
            })
            .collect::<Vec<_>>();
        assert_eq!(diagnostic_launch_ids, [observed[0].0, observed[1].0]);
        runner.shutdown().await.unwrap();
    }

    #[tokio::test]
    async fn dropped_job_transfers_driver_and_next_start_reaps_before_launch() {
        let mut runner = ContinuousRunner::new();
        let first_source = LifecycleProbe::default();
        let first = runner
            .start(spec(
                false,
                Arc::new(AtomicUsize::new(0)),
                first_source.clone(),
                LifecycleProbe::default(),
            ))
            .await
            .unwrap();
        assert_eq!(first.driver_owner(), DriverOwnership::Driving);

        drop(first);
        let second = runner
            .start(spec(
                false,
                Arc::new(AtomicUsize::new(0)),
                LifecycleProbe::default(),
                LifecycleProbe::default(),
            ))
            .await
            .unwrap();

        assert_eq!(first_source.closed.load(Ordering::SeqCst), 1);
        assert_eq!(second.state(), ContinuousJobState::Running);
        drop(second);
        runner.shutdown().await.unwrap();
        assert_eq!(runner.registry_counts(), (0, 0));
    }

    #[tokio::test]
    async fn dropped_job_transfers_driver_and_runner_shutdown_reaps() {
        let mut runner = ContinuousRunner::new();
        let source = LifecycleProbe::default();
        let sink = LifecycleProbe::default();
        let job = runner
            .start(spec(
                false,
                Arc::new(AtomicUsize::new(0)),
                source.clone(),
                sink.clone(),
            ))
            .await
            .unwrap();
        let core = Arc::clone(&job.core);

        drop(job);

        assert_eq!(core.state.lock().owner, DriverOwnership::ReaperOwned);
        runner.shutdown().await.unwrap();
        assert_eq!(runner.registry_counts(), (0, 0));
        assert_eq!(core.state.lock().owner, DriverOwnership::Terminal);
        assert_eq!(core.metrics.snapshot().job.reaper_joins, 1);
        assert_eq!(source.closed.load(Ordering::SeqCst), 1);
        assert_eq!(sink.closed.load(Ordering::SeqCst), 1);
    }

    #[tokio::test]
    async fn runner_drop_records_abandonment_and_requests_reaper_cancellation() {
        let runner = ContinuousRunner::new();
        let runner_core = Arc::clone(&runner.core);
        let job = runner
            .start(spec(
                false,
                Arc::new(AtomicUsize::new(0)),
                LifecycleProbe::default(),
                LifecycleProbe::default(),
            ))
            .await
            .unwrap();
        assert_eq!(job.status().metrics.job.abandoned_runner_drops, 0);
        assert!(!job.core.launch_cancel.is_cancelled());

        drop(runner);

        assert_eq!(
            ABANDONED_RUNNER_WARNING,
            "continuous runner dropped before shutdown completed; cancellation requested"
        );
        assert_eq!(runner_core.abandonment_warnings.load(Ordering::SeqCst), 1);
        assert_eq!(job.status().metrics.job.abandoned_runner_drops, 1);
        assert!(job.core.terminal_arbiter.explicit_cancel_requested());
        assert_eq!(job.core.state.lock().owner, DriverOwnership::ReaperOwned);
        assert!(
            !job.core.launch_cancel.is_cancelled(),
            "claimed jobs are cancelled only after the driver arbitrates the terminal cause"
        );
    }

    #[tokio::test]
    async fn runner_drop_reaper_publishes_completion_for_a_live_job() {
        let runner = ContinuousRunner::new();
        let source = LifecycleProbe::default();
        let sink = LifecycleProbe::default();
        let job = runner
            .start(spec(
                false,
                Arc::new(AtomicUsize::new(0)),
                source.clone(),
                sink.clone(),
            ))
            .await
            .unwrap();

        drop(runner);

        let mut wait = Box::pin(job.wait());
        let mut completed = None;
        for _ in 0..100 {
            if let Poll::Ready(outcome) = futures::poll!(wait.as_mut()) {
                completed = Some(outcome);
                break;
            }
            tokio::task::yield_now().await;
        }
        let outcome = completed.expect("runner Drop reaper did not publish the live job outcome");
        assert_eq!(outcome.state, ContinuousJobState::Cancelled);
        assert_eq!(outcome.cause, TerminalCause::ExplicitCancel);
        assert_eq!(job.driver_owner(), DriverOwnership::Terminal);
        assert!(job.status().tasks.is_empty());
        assert_eq!(job.status().metrics.job.abandoned_runner_drops, 1);
        assert_eq!(job.status().metrics.job.reaper_joins, 1);
        assert_eq!(source.closed.load(Ordering::SeqCst), 1);
        assert_eq!(sink.closed.load(Ordering::SeqCst), 1);
    }

    #[tokio::test]
    async fn runner_drop_cancels_and_reaps_a_provisional_open() {
        let runner = ContinuousRunner::new();
        let source = LifecycleProbe::default();
        source.block_open.store(true, Ordering::SeqCst);
        let sink = LifecycleProbe::default();
        let observer = runner.start(spec(
            false,
            Arc::new(AtomicUsize::new(0)),
            source.clone(),
            sink.clone(),
        ));
        let provisional = Arc::clone(observer.core.as_ref().unwrap());
        let mut observer = Box::pin(observer);
        assert!(matches!(futures::poll!(observer.as_mut()), Poll::Pending));
        for _ in 0..100 {
            if source.opened.load(Ordering::SeqCst) == 1 {
                break;
            }
            tokio::task::yield_now().await;
        }
        assert_eq!(source.opened.load(Ordering::SeqCst), 1);

        drop(runner);

        assert!(provisional.launch_cancel.is_cancelled());
        assert_eq!(provisional.state.lock().owner, DriverOwnership::ReaperOwned);
        assert_eq!(provisional.metrics.snapshot().job.abandoned_runner_drops, 1);
        let mut completed = None;
        for _ in 0..100 {
            if let Poll::Ready(result) = futures::poll!(observer.as_mut()) {
                completed = Some(result);
                break;
            }
            tokio::task::yield_now().await;
        }
        let failure = completed
            .expect("runner Drop reaper did not publish provisional launch cancellation")
            .unwrap_err();
        assert!(matches!(
            failure.primary.error,
            CalcFlowError::Cancelled { .. }
        ));
        assert_eq!(provisional.state.lock().owner, DriverOwnership::Terminal);
        assert_eq!(source.closed.load(Ordering::SeqCst), 1);
        assert_eq!(sink.closed.load(Ordering::SeqCst), 1);
    }

    #[tokio::test]
    async fn job_drop_then_runner_drop_transfers_and_publishes_once() {
        let runner = ContinuousRunner::new();
        let source = LifecycleProbe::default();
        let sink = LifecycleProbe::default();
        let job = runner
            .start(spec(
                false,
                Arc::new(AtomicUsize::new(0)),
                source.clone(),
                sink.clone(),
            ))
            .await
            .unwrap();
        let core = Arc::clone(&job.core);
        let mut wait = Box::pin(job.wait());

        drop(job);
        assert_eq!(core.state.lock().owner, DriverOwnership::ReaperOwned);
        drop(runner);

        let mut completed = None;
        for _ in 0..100 {
            if let Poll::Ready(outcome) = futures::poll!(wait.as_mut()) {
                completed = Some(outcome);
                break;
            }
            tokio::task::yield_now().await;
        }
        let outcome = completed.expect("combined Drop reaper did not publish the job outcome");
        assert_eq!(outcome.state, ContinuousJobState::Cancelled);
        assert_eq!(outcome.cause, TerminalCause::ExplicitCancel);
        assert_eq!(core.state.lock().owner, DriverOwnership::Terminal);
        assert_eq!(core.metrics.snapshot().job.abandoned_runner_drops, 1);
        assert_eq!(core.metrics.snapshot().job.reaper_joins, 1);
        assert_eq!(source.closed.load(Ordering::SeqCst), 1);
        assert_eq!(sink.closed.load(Ordering::SeqCst), 1);
    }

    #[tokio::test]
    async fn explicit_cancel_wins_over_an_unobserved_graceful_request() {
        let mut runner = ContinuousRunner::new();
        let job = runner
            .start(spec(
                false,
                Arc::new(AtomicUsize::new(0)),
                LifecycleProbe::default(),
                LifecycleProbe::default(),
            ))
            .await
            .unwrap();

        let graceful = job.shutdown();
        let cancelled = job.cancel().await;
        drop(graceful);

        assert_eq!(cancelled.state, ContinuousJobState::Cancelled);
        assert_eq!(cancelled.cause, TerminalCause::ExplicitCancel);
        drop(job);
        runner.shutdown().await.unwrap();
    }

    #[tokio::test(start_paused = true)]
    async fn deadline_preempts_graceful_drain_with_pending_operator_and_source() {
        let entered = Arc::new(AtomicBool::new(false));
        let plan = deadline_pending_plan(Arc::clone(&entered));
        let polls = Arc::new(AtomicUsize::new(0));
        let source_closed = Arc::new(AtomicUsize::new(0));
        let sink_closed = Arc::new(AtomicUsize::new(0));
        let writes = Arc::new(Mutex::new(Vec::new()));
        let spec = ContinuousJobSpec {
            context: StreamJobContext::new(
                84,
                plan.fingerprint(),
                JsonMap::new(),
                Some(chrono::Utc::now() + chrono::Duration::seconds(5)),
                CancellationToken::new(),
            ),
            plan,
            sources: vec![NamedSourceBinding {
                binding_id: "input".into(),
                binding: counting_pending_binding(1, &polls, &source_closed),
            }],
            sinks: vec![NamedSinkBinding {
                output_id: "output".into(),
                sink_id: "recording".into(),
                binding: OrdinarySinkBinding::new(Box::new(OrderedRecordingSink {
                    id: "recording".into(),
                    writes: Arc::clone(&writes),
                    closed: Arc::clone(&sink_closed),
                })),
            }],
            edge_budget: EdgeBudget {
                max_rows: 1,
                max_bytes: 1 << 20,
            },
            delivery_mode: M2DeliveryMode::ProcessLocalOrdered,
        };
        let mut runner = ContinuousRunner::new();
        let job = runner.start(spec).await.unwrap();
        for _ in 0..100 {
            if entered.load(Ordering::SeqCst) && polls.load(Ordering::SeqCst) >= 2 {
                break;
            }
            tokio::task::yield_now().await;
        }
        assert!(entered.load(Ordering::SeqCst));
        assert!(polls.load(Ordering::SeqCst) >= 2);

        let shutdown = job.shutdown();
        assert_eq!(job.state(), ContinuousJobState::Draining);
        tokio::time::advance(StdDuration::from_secs(6)).await;
        let outcome = tokio::time::timeout(StdDuration::from_secs(1), shutdown)
            .await
            .expect("deadline must interrupt a graceful drain blocked in an operator handler");

        assert_eq!(outcome.state, ContinuousJobState::Cancelled);
        assert_eq!(outcome.cause, TerminalCause::DeadlineExceeded);
        assert!(writes.lock().is_empty());
        assert_eq!(source_closed.load(Ordering::SeqCst), 1);
        assert_eq!(sink_closed.load(Ordering::SeqCst), 1);
        drop(job);
        runner.shutdown().await.unwrap();
    }

    #[tokio::test(start_paused = true)]
    async fn deadline_preempts_graceful_drain_with_pending_sink_and_source() {
        let plan = unary_expression_plan();
        let polls = Arc::new(AtomicUsize::new(0));
        let source_closed = Arc::new(AtomicUsize::new(0));
        let sink_closed = Arc::new(AtomicUsize::new(0));
        let sink_started = Arc::new(AtomicBool::new(false));
        let sink_gate = Arc::new(Notify::new());
        let writes = Arc::new(Mutex::new(Vec::new()));
        let spec = ContinuousJobSpec {
            context: StreamJobContext::new(
                85,
                plan.fingerprint(),
                JsonMap::new(),
                Some(chrono::Utc::now() + chrono::Duration::seconds(5)),
                CancellationToken::new(),
            ),
            plan,
            sources: vec![NamedSourceBinding {
                binding_id: "input".into(),
                binding: counting_pending_binding(1, &polls, &source_closed),
            }],
            sinks: vec![NamedSinkBinding {
                output_id: "output".into(),
                sink_id: "gated".into(),
                binding: OrdinarySinkBinding::new(Box::new(GatedSink {
                    started: Arc::clone(&sink_started),
                    gate: sink_gate,
                    writes: Arc::clone(&writes),
                    closed: Arc::clone(&sink_closed),
                })),
            }],
            edge_budget: EdgeBudget {
                max_rows: 1,
                max_bytes: 1 << 20,
            },
            delivery_mode: M2DeliveryMode::ProcessLocalOrdered,
        };
        let mut runner = ContinuousRunner::new();
        let job = runner.start(spec).await.unwrap();
        for _ in 0..100 {
            if sink_started.load(Ordering::SeqCst) && polls.load(Ordering::SeqCst) >= 2 {
                break;
            }
            tokio::task::yield_now().await;
        }
        assert!(sink_started.load(Ordering::SeqCst));
        assert!(polls.load(Ordering::SeqCst) >= 2);

        let shutdown = job.shutdown();
        assert_eq!(job.state(), ContinuousJobState::Draining);
        tokio::time::advance(StdDuration::from_secs(6)).await;
        let outcome = tokio::time::timeout(StdDuration::from_secs(1), shutdown)
            .await
            .expect("deadline must interrupt a graceful drain blocked in a sink write");

        assert_eq!(outcome.state, ContinuousJobState::Cancelled);
        assert_eq!(outcome.cause, TerminalCause::DeadlineExceeded);
        assert!(writes.lock().is_empty());
        assert_eq!(source_closed.load(Ordering::SeqCst), 1);
        assert_eq!(sink_closed.load(Ordering::SeqCst), 1);
        drop(job);
        runner.shutdown().await.unwrap();
    }

    #[tokio::test]
    async fn finite_source_eof_drives_the_registered_graph_to_natural_completion() {
        let mut runner = ContinuousRunner::new();
        let source = LifecycleProbe::default();
        source.finite.store(true, Ordering::SeqCst);
        let job = runner
            .start(spec(
                false,
                Arc::new(AtomicUsize::new(0)),
                source,
                LifecycleProbe::default(),
            ))
            .await
            .unwrap();

        let outcome = tokio::time::timeout(std::time::Duration::from_millis(50), job.wait())
            .await
            .expect("finite source graph did not converge after explicit EOF");

        assert_eq!(outcome.state, ContinuousJobState::Completed);
        assert_eq!(outcome.cause, TerminalCause::NaturalEnd);
        drop(job);
        runner.shutdown().await.unwrap();
    }

    #[allow(
        clippy::too_many_lines,
        reason = "the end-to-end scenario keeps setup, lifecycle, and status assertions together"
    )]
    #[tokio::test]
    async fn two_sources_union_expression_and_ordered_sinks_complete_end_to_end() {
        let union = UnionOperator::new(
            "merge",
            vec![
                Port::new("left", BatchKind::Table, true, None).unwrap(),
                Port::new("right", BatchKind::Table, true, None).unwrap(),
            ],
        )
        .unwrap();
        let expression =
            ExpressionOperator::new("calc", "plus_one = value + 1", Vec::new(), None, Vec::new())
                .unwrap();
        let plan = PipelineBuilder::new("e2e")
            .unwrap()
            .add_node("merge", Box::new(union))
            .unwrap()
            .add_node("calc", Box::new(expression))
            .unwrap()
            .connect(Edge::new(
                PortEndpoint::new("merge", "output").unwrap(),
                PortEndpoint::new("calc", "input").unwrap(),
            ))
            .unwrap()
            .compile_stream(
                &UdfRegistry::new().snapshot(),
                &StreamRequirements::default(),
            )
            .unwrap();
        let left_closed = Arc::new(AtomicUsize::new(0));
        let right_closed = Arc::new(AtomicUsize::new(0));
        let sink_closed = Arc::new(AtomicUsize::new(0));
        let writes = Arc::new(Mutex::new(Vec::new()));
        let spec = ContinuousJobSpec {
            context: StreamJobContext::new(
                77,
                plan.fingerprint(),
                JsonMap::new(),
                None,
                CancellationToken::new(),
            ),
            plan,
            sources: vec![
                NamedSourceBinding {
                    binding_id: "left".into(),
                    binding: finite_binding(&[1, 2], &left_closed),
                },
                NamedSourceBinding {
                    binding_id: "right".into(),
                    binding: finite_binding(&[10, 20], &right_closed),
                },
            ],
            sinks: ["first", "second"]
                .into_iter()
                .map(|sink_id| NamedSinkBinding {
                    output_id: "output".into(),
                    sink_id: sink_id.into(),
                    binding: OrdinarySinkBinding::new(Box::new(OrderedRecordingSink {
                        id: sink_id.into(),
                        writes: Arc::clone(&writes),
                        closed: Arc::clone(&sink_closed),
                    })),
                })
                .collect(),
            edge_budget: EdgeBudget {
                max_rows: 1,
                max_bytes: 1 << 20,
            },
            delivery_mode: M2DeliveryMode::ProcessLocalOrdered,
        };
        let mut runner = ContinuousRunner::new();
        let job = runner.start(spec).await.unwrap();

        let outcome = job.wait().await;

        assert_eq!(outcome.state, ContinuousJobState::Completed);
        assert_eq!(outcome.cause, TerminalCause::NaturalEnd);
        assert!(outcome.errors.is_empty());
        let status = job.status();
        assert_eq!(status.job_id, job.id());
        assert_eq!(status.state, ContinuousJobState::Completed);
        assert_eq!(status.terminal_cause, Some(TerminalCause::NaturalEnd));
        assert!(status.tasks.is_empty());
        assert_eq!(status.edges.len(), 4);
        assert!(status.edges.values().all(|edge| {
            edge.queue_depth == 0 && edge.charged_rows == 0 && edge.charged_bytes == 0
        }));
        assert!(
            status
                .edges
                .values()
                .all(|edge| { edge.high_water_rows <= 1 && edge.high_water_bytes <= (1 << 20) })
        );
        for (edge_id, batches) in [
            ("source/6c656674/6d65726765/6c656674", 2),
            ("source/7269676874/6d65726765/7269676874", 2),
            ("merge.output->calc.input", 4),
            ("sink/63616c63/6f7574707574/6f7574707574", 4),
        ] {
            let edge = &status.metrics.edges[edge_id];
            assert_eq!(edge.input_batches, batches, "enqueue boundary {edge_id}");
            assert_eq!(edge.output_batches, batches, "dequeue boundary {edge_id}");
        }
        for source_id in ["left", "right"] {
            assert_eq!(
                status.sources[source_id].latest_observed_order,
                Some(vec![2])
            );
            assert_eq!(status.sources[source_id].durable_order, None);
            assert_eq!(status.sources[source_id].next_sequence, Some(2));
            assert!(status.sources[source_id].ended);
            assert_eq!(status.metrics.sources[source_id].poll_count, 3);
            assert_eq!(status.metrics.sources[source_id].data_batches, 2);
            assert_eq!(
                status.metrics.sources[source_id].fully_fanned_out_batches,
                2
            );
        }
        for node_id in ["merge", "calc"] {
            assert_eq!(status.nodes[node_id].input_batches, 4);
            assert_eq!(status.nodes[node_id].fully_fanned_out_batches, 4);
            assert!(status.nodes[node_id].ended);
            assert_eq!(status.metrics.nodes[node_id].input_batches, 4);
            assert_eq!(status.metrics.nodes[node_id].fully_fanned_out_batches, 4);
        }
        for sink_id in ["first", "second"] {
            let metric_id = super::sink_metric_id("output", sink_id);
            assert_eq!(status.sinks[&metric_id].delivered_batches, 4);
            assert!(status.sinks[&metric_id].ended);
            assert_eq!(status.metrics.sinks[&metric_id].delivered_batches, 4);
        }
        assert_eq!(
            status.metrics.job.terminal_state,
            Some(ContinuousJobState::Completed)
        );
        assert_eq!(
            status.metrics.job.terminal_cause,
            Some(TerminalCause::NaturalEnd)
        );
        assert!(!format!("{status:?}").contains("secret-canary"));
        let writes = writes.lock().clone();
        assert_eq!(writes.len(), 8);
        for pair in writes.chunks_exact(2) {
            assert_eq!(pair[0].0, "first");
            assert_eq!(pair[1].0, "second");
            assert_eq!((&pair[0].1, pair[0].2), (&pair[1].1, pair[1].2));
        }
        for source in ["left", "right"] {
            let sequence = writes
                .iter()
                .filter(|(sink, observed_source, _)| sink == "first" && observed_source == source)
                .map(|(_, _, sequence)| *sequence)
                .collect::<Vec<_>>();
            assert_eq!(sequence, [0, 1]);
        }
        assert_eq!(left_closed.load(Ordering::SeqCst), 1);
        assert_eq!(right_closed.load(Ordering::SeqCst), 1);
        assert_eq!(sink_closed.load(Ordering::SeqCst), 2);
        assert_eq!(runner.registry_counts(), (0, 0));
        drop(job);
        runner.shutdown().await.unwrap();
    }

    #[tokio::test]
    async fn two_sources_reopen_once_at_their_distinct_resume_cursors() {
        let plan = union_plan();
        let left_resume = Cursor::new(vec![1], JsonMap::new()).unwrap();
        let right_resume = Cursor::new(vec![2], JsonMap::new()).unwrap();
        let left_opened = Arc::new(Mutex::new(Vec::new()));
        let right_opened = Arc::new(Mutex::new(Vec::new()));
        let source_closed = Arc::new(AtomicUsize::new(0));
        let sink = LifecycleProbe::default();
        let spec = ContinuousJobSpec {
            context: StreamJobContext::new(
                92,
                plan.fingerprint(),
                JsonMap::new(),
                None,
                CancellationToken::new(),
            ),
            plan,
            sources: vec![
                NamedSourceBinding {
                    binding_id: "left".into(),
                    binding: SourceBinding::new(
                        Box::new(ResumeProbeSource {
                            opened_with: Arc::clone(&left_opened),
                            closed: Arc::clone(&source_closed),
                        }),
                        Some(left_resume.clone()),
                        3,
                    )
                    .unwrap(),
                },
                NamedSourceBinding {
                    binding_id: "right".into(),
                    binding: SourceBinding::new(
                        Box::new(ResumeProbeSource {
                            opened_with: Arc::clone(&right_opened),
                            closed: Arc::clone(&source_closed),
                        }),
                        Some(right_resume.clone()),
                        7,
                    )
                    .unwrap(),
                },
            ],
            sinks: vec![NamedSinkBinding {
                output_id: "output".into(),
                sink_id: "sink".into(),
                binding: OrdinarySinkBinding::new(Box::new(ProbeSink(sink.clone()))),
            }],
            edge_budget: EdgeBudget::default(),
            delivery_mode: M2DeliveryMode::ProcessLocalOrdered,
        };
        let mut runner = ContinuousRunner::new();
        let job = runner.start(spec).await.unwrap();

        let outcome = job.wait().await;

        assert_eq!(outcome.cause, TerminalCause::NaturalEnd);
        assert_eq!(&*left_opened.lock(), &[Some(left_resume)]);
        assert_eq!(&*right_opened.lock(), &[Some(right_resume)]);
        assert_eq!(source_closed.load(Ordering::SeqCst), 2);
        assert_eq!(sink.closed.load(Ordering::SeqCst), 1);
        drop(job);
        runner.shutdown().await.unwrap();
    }

    #[tokio::test]
    async fn slow_sink_backpressures_both_sources_after_bounded_prefetch() {
        let plan = union_plan();
        let left_polls = Arc::new(AtomicUsize::new(0));
        let right_polls = Arc::new(AtomicUsize::new(0));
        let left_closed = Arc::new(AtomicUsize::new(0));
        let right_closed = Arc::new(AtomicUsize::new(0));
        let sink_closed = Arc::new(AtomicUsize::new(0));
        let sink_started = Arc::new(AtomicBool::new(false));
        let sink_gate = Arc::new(Notify::new());
        let writes = Arc::new(Mutex::new(Vec::new()));
        let spec = ContinuousJobSpec {
            context: StreamJobContext::new(
                81,
                plan.fingerprint(),
                JsonMap::new(),
                None,
                CancellationToken::new(),
            ),
            plan,
            sources: vec![
                NamedSourceBinding {
                    binding_id: "left".into(),
                    binding: counting_pending_binding(100, &left_polls, &left_closed),
                },
                NamedSourceBinding {
                    binding_id: "right".into(),
                    binding: counting_pending_binding(100, &right_polls, &right_closed),
                },
            ],
            sinks: vec![NamedSinkBinding {
                output_id: "output".into(),
                sink_id: "slow".into(),
                binding: OrdinarySinkBinding::new(Box::new(GatedSink {
                    started: Arc::clone(&sink_started),
                    gate: Arc::clone(&sink_gate),
                    writes,
                    closed: Arc::clone(&sink_closed),
                })),
            }],
            edge_budget: EdgeBudget {
                max_rows: 1,
                max_bytes: 1 << 20,
            },
            delivery_mode: M2DeliveryMode::ProcessLocalOrdered,
        };
        let mut runner = ContinuousRunner::new();
        let job = runner.start(spec).await.unwrap();
        while !sink_started.load(Ordering::SeqCst) {
            tokio::task::yield_now().await;
        }
        for _ in 0..200 {
            tokio::task::yield_now().await;
        }
        let stopped = (
            left_polls.load(Ordering::SeqCst),
            right_polls.load(Ordering::SeqCst),
        );
        for _ in 0..200 {
            tokio::task::yield_now().await;
        }
        assert_eq!(
            stopped,
            (
                left_polls.load(Ordering::SeqCst),
                right_polls.load(Ordering::SeqCst),
            )
        );
        assert!(
            stopped.0 < 100 && stopped.1 < 100,
            "polls did not stop: {stopped:?}"
        );

        let outcome = job.cancel().await;
        assert_eq!(outcome.cause, TerminalCause::ExplicitCancel);
        assert_eq!(left_closed.load(Ordering::SeqCst), 1);
        assert_eq!(right_closed.load(Ordering::SeqCst), 1);
        assert_eq!(sink_closed.load(Ordering::SeqCst), 1);
        drop(job);
        runner.shutdown().await.unwrap();
    }

    #[tokio::test]
    async fn terminal_metrics_overflow_keeps_operator_primary_and_publishes_once() {
        let resets = Arc::new(AtomicUsize::new(0));
        let operator = ResetOperator {
            inputs: [Port::new("input", BatchKind::Table, true, None).unwrap()],
            outputs: [Port::new("output", BatchKind::Table, true, None).unwrap()],
            resets,
            fail_reset: false,
            panic_reset: false,
        };
        let plan = PipelineBuilder::new("metrics-overflow")
            .unwrap()
            .add_node("node", Box::new(operator) as Box<dyn StreamOperator>)
            .unwrap()
            .compile_stream(
                &UdfRegistry::new().snapshot(),
                &StreamRequirements::default(),
            )
            .unwrap();
        let release = Arc::new(Notify::new());
        let source_closed = Arc::new(AtomicUsize::new(0));
        let spec = ContinuousJobSpec {
            context: StreamJobContext::new(
                91,
                plan.fingerprint(),
                JsonMap::new(),
                None,
                CancellationToken::new(),
            ),
            plan,
            sources: vec![NamedSourceBinding {
                binding_id: "input".into(),
                binding: SourceBinding::new(
                    Box::new(GatedDataSource {
                        release: Arc::clone(&release),
                        delivered: false,
                        closed: Arc::clone(&source_closed),
                    }),
                    None,
                    0,
                )
                .unwrap(),
            }],
            sinks: vec![NamedSinkBinding {
                output_id: "output".into(),
                sink_id: "sink".into(),
                binding: OrdinarySinkBinding::new(Box::new(ProbeSink(LifecycleProbe::default()))),
            }],
            edge_budget: EdgeBudget {
                max_rows: 1,
                max_bytes: 1 << 20,
            },
            delivery_mode: M2DeliveryMode::ProcessLocalOrdered,
        };
        let mut runner = ContinuousRunner::new();
        let job = runner.start(spec).await.unwrap();
        job.core.metrics.preset_job_task_errors_for_test(u64::MAX);
        release.notify_one();

        let outcome = job.wait().await;

        assert_eq!(outcome.state, ContinuousJobState::Failed);
        assert_eq!(
            outcome
                .errors
                .iter()
                .filter(|failure| matches!(failure.origin, super::FailureOrigin::Metrics { .. }))
                .count(),
            1,
            "errors: {:?}",
            outcome.errors
        );
        assert!(matches!(
            outcome.errors[0].error,
            CalcFlowError::Operator { ref message, .. } if message == "data failure"
        ));
        assert!(matches!(
            outcome.errors.last().unwrap().origin,
            super::FailureOrigin::Metrics {
                counter: "task_errors",
                ..
            }
        ));
        let status = job.status();
        assert_eq!(status.metrics.job.task_errors, u64::MAX);
        assert!(status.metrics.job.metrics_overflowed);
        assert!(status.tasks.is_empty());
        assert!(status.edges.values().all(|edge| edge.queue_depth == 0));
        assert_eq!(source_closed.load(Ordering::SeqCst), 1);
        drop(job);
        runner.shutdown().await.unwrap();
    }

    #[tokio::test]
    async fn graceful_drains_the_accepted_prefix_while_cancel_makes_no_drain_promise() {
        async fn run_case(
            graceful: bool,
        ) -> (Arc<super::ContinuousJobOutcome>, Vec<(String, u64)>) {
            let plan = unary_expression_plan();
            let polls = Arc::new(AtomicUsize::new(0));
            let source_closed = Arc::new(AtomicUsize::new(0));
            let sink_closed = Arc::new(AtomicUsize::new(0));
            let sink_started = Arc::new(AtomicBool::new(false));
            let sink_gate = Arc::new(Notify::new());
            let writes = Arc::new(Mutex::new(Vec::new()));
            let spec = ContinuousJobSpec {
                context: StreamJobContext::new(
                    if graceful { 82 } else { 83 },
                    plan.fingerprint(),
                    JsonMap::new(),
                    None,
                    CancellationToken::new(),
                ),
                plan,
                sources: vec![NamedSourceBinding {
                    binding_id: "input".into(),
                    binding: counting_pending_binding(1, &polls, &source_closed),
                }],
                sinks: vec![NamedSinkBinding {
                    output_id: "output".into(),
                    sink_id: "gated".into(),
                    binding: OrdinarySinkBinding::new(Box::new(GatedSink {
                        started: Arc::clone(&sink_started),
                        gate: Arc::clone(&sink_gate),
                        writes: Arc::clone(&writes),
                        closed: Arc::clone(&sink_closed),
                    })),
                }],
                edge_budget: EdgeBudget {
                    max_rows: 1,
                    max_bytes: 1 << 20,
                },
                delivery_mode: M2DeliveryMode::ProcessLocalOrdered,
            };
            let mut runner = ContinuousRunner::new();
            let job = runner.start(spec).await.unwrap();
            while !sink_started.load(Ordering::SeqCst) {
                tokio::task::yield_now().await;
            }
            let outcome = if graceful {
                let observer = job.shutdown();
                sink_gate.notify_one();
                observer.await
            } else {
                job.cancel().await
            };
            assert_eq!(source_closed.load(Ordering::SeqCst), 1);
            assert_eq!(sink_closed.load(Ordering::SeqCst), 1);
            drop(job);
            runner.shutdown().await.unwrap();
            let observed = writes.lock().clone();
            (outcome, observed)
        }

        let (graceful, drained) = run_case(true).await;
        assert_eq!(graceful.state, ContinuousJobState::Completed);
        assert_eq!(graceful.cause, TerminalCause::GracefulShutdown);
        assert_eq!(drained, [("input".into(), 0)]);

        let (cancelled, not_drained) = run_case(false).await;
        assert_eq!(cancelled.state, ContinuousJobState::Cancelled);
        assert_eq!(cancelled.cause, TerminalCause::ExplicitCancel);
        assert!(not_drained.is_empty());
    }
}
