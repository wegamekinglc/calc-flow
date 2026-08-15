use std::{
    collections::{BTreeMap, BTreeSet, VecDeque},
    future::Future,
    panic::AssertUnwindSafe,
    path::PathBuf,
    pin::Pin,
    sync::{
        Arc,
        atomic::{AtomicBool, AtomicU64, Ordering},
    },
    task::{Context, Poll},
    time::Duration,
};

use chrono::Utc;
use futures::{FutureExt, future::try_join_all};
use parking_lot::Mutex;
use serde::Serialize;
use sha2::{Digest as _, Sha256};
use tokio::{
    sync::{Notify, mpsc, watch},
    task::{JoinHandle, JoinSet},
};

use super::{
    ChannelMetrics, EdgeReceiver, EdgeSender,
    channel::edge_channel_with_metrics,
    checkpoint::{
        ManagedCheckpointRuntime, OpenedManagedCheckpointRuntime,
        coordinator::{
            CheckpointAck, CheckpointCoordinatorHandle, CheckpointEvent, CheckpointPhase,
            CheckpointRequest, ManualCheckpointFailure, ManualCheckpointFailureCategory,
            ParticipantSet, spawn_checkpoint_coordinator,
        },
    },
    job::{
        ContinuousJobSpec, OrdinarySinkBinding, OwningContinuousJob, StableSinkId,
        ValidatedContinuousJob, ValidatedOrdinarySink, preflight_job,
    },
    metrics::{M2MetricsSnapshot, MetricsRecorder, MetricsTimer, sink_metric_id},
    operator_task::{
        OperatorCheckpointAck, OperatorCheckpointCommand, OperatorCheckpointPort, OperatorIngress,
        OperatorProgress, OperatorProgressSnapshot, OperatorRestoreState, OperatorTaskInputs,
        OperatorTerminalPort, spawn_operator_task,
    },
    progress::{
        DurableProgressRestore, DurableSourceCut, LiveProgressCoordinator, LiveProgressEvidence,
        LiveProgressStatusHandle, restore_durable_progress, spawn_live_progress_task,
        types::LogicalInstant,
    },
    projection::{JobStatus, StatusProjection},
    sink_task::{
        SinkCheckpointAck, SinkCheckpointCommand, SinkCheckpointPort, SinkEpochOwner,
        SinkFailurePhase, SinkFinalizeAck, SinkProgress, SinkTaskInputs, spawn_sink_task,
    },
    source_task::{
        SourceBinding, SourceProgress, SourceProgressSnapshot,
        spawn_source_tasks_gated_with_live_progress,
    },
    supervisor::{
        SupervisionReport, TaskId, TaskRegistry, TaskStatus, TaskSupervisor, TerminalArbiter,
        TerminalDecision, panic_message,
    },
};
use crate::pipeline::{
    OperatorCheckpointCapability, RuntimeSinkRoute, RuntimeSourceRoute, RuntimeStreamNode,
    StreamRuntimePlanParts,
};
#[cfg(test)]
use crate::state::ManifestTransactionFaultPoint;
use crate::{
    CalcFlowError, CancellationToken, CheckpointManifest, CheckpointManifestFields, Epoch,
    OperatorManifestEntry, RecoveryStatus, SinkManifestEntry, SourceManifestEntry, StateBackend,
    StateLineageKey, StreamRuntimeConfig,
    state::{
        ManifestPublication, ManifestTransaction, PreparedEpochManifest, PreparedManifestIdentity,
        SelectedManifest,
    },
};

const CONNECTOR_CLOSE_TIMEOUT: Duration = Duration::from_secs(5);

#[cfg(test)]
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) enum CheckpointFaultPoint {
    SourceAdmission,
    SourceCut,
    PartialAlignment,
    StateStage,
    SinkPreCommit,
    ManifestWrite,
    ManifestRename,
    ManifestParentSync,
    PartialSinkCommit,
    CompletedCommit,
    Retention,
    Compaction,
}

#[cfg(test)]
impl CheckpointFaultPoint {
    pub(crate) const ALL: [Self; 12] = [
        Self::SourceAdmission,
        Self::SourceCut,
        Self::PartialAlignment,
        Self::StateStage,
        Self::SinkPreCommit,
        Self::ManifestWrite,
        Self::ManifestRename,
        Self::ManifestParentSync,
        Self::PartialSinkCommit,
        Self::CompletedCommit,
        Self::Retention,
        Self::Compaction,
    ];
}

#[cfg(test)]
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) enum CheckpointFaultMode {
    Io,
    Panic,
    Cancel,
    Restart,
}

#[cfg(test)]
impl CheckpointFaultMode {
    pub(crate) const ALL: [Self; 4] = [Self::Io, Self::Panic, Self::Cancel, Self::Restart];
}

#[cfg(test)]
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
struct CheckpointFault {
    point: CheckpointFaultPoint,
    mode: CheckpointFaultMode,
}

#[cfg(test)]
#[derive(Default)]
struct CheckpointFaultState {
    armed: Option<CheckpointFault>,
    trigger_count: usize,
    cancellation_trigger_count: usize,
}

#[cfg(test)]
#[derive(Clone, Default)]
pub(crate) struct CheckpointFaultInjector(Arc<Mutex<CheckpointFaultState>>);

#[cfg(test)]
impl CheckpointFaultInjector {
    fn armed(point: CheckpointFaultPoint, mode: CheckpointFaultMode) -> Self {
        Self(Arc::new(Mutex::new(CheckpointFaultState {
            armed: Some(CheckpointFault { point, mode }),
            trigger_count: 0,
            cancellation_trigger_count: 0,
        })))
    }

    fn trigger(
        &self,
        point: CheckpointFaultPoint,
        cancellation: &CancellationToken,
    ) -> crate::Result<()> {
        let fault = {
            let mut state = self.0.lock();
            match state.armed {
                Some(fault) if fault.point == point => {
                    state.trigger_count += 1;
                    if fault.mode == CheckpointFaultMode::Cancel {
                        state.cancellation_trigger_count += 1;
                    }
                    state.armed.take()
                }
                _ => None,
            }
        };
        let Some(fault) = fault else {
            return Ok(());
        };
        match fault.mode {
            CheckpointFaultMode::Io => Err(CalcFlowError::Io {
                path: format!("/fault-injection/{point:?}/credential-canary"),
                source: std::io::Error::other("injected checkpoint I/O fault"),
            }),
            CheckpointFaultMode::Panic => {
                panic!("injected checkpoint panic at {point:?}")
            }
            CheckpointFaultMode::Cancel => {
                cancellation.cancel();
                Ok(())
            }
            CheckpointFaultMode::Restart => Err(CalcFlowError::Internal {
                message: format!("injected checkpoint restart at {point:?}"),
            }),
        }
    }

    #[cfg(unix)]
    fn is_armed(&self, point: CheckpointFaultPoint, mode: CheckpointFaultMode) -> bool {
        self.0
            .lock()
            .armed
            .is_some_and(|fault| fault.point == point && fault.mode == mode)
    }

    pub(crate) fn trigger_count(&self) -> usize {
        self.0.lock().trigger_count
    }

    pub(crate) fn cancellation_trigger_count(&self) -> usize {
        self.0.lock().cancellation_trigger_count
    }
}

#[cfg(test)]
#[derive(Clone, Default)]
pub(crate) struct CheckpointStartedTestGate {
    entered: Arc<AtomicBool>,
    entered_changed: Arc<Notify>,
    released: Arc<AtomicBool>,
    release_changed: Arc<Notify>,
}

#[cfg(test)]
impl CheckpointStartedTestGate {
    pub(crate) fn has_entered(&self) -> bool {
        self.entered.load(Ordering::Acquire)
    }

    pub(crate) async fn wait_until_entered(&self) {
        while !self.entered.load(Ordering::Acquire) {
            let changed = self.entered_changed.notified();
            if self.entered.load(Ordering::Acquire) {
                break;
            }
            changed.await;
        }
    }

    pub(crate) fn release(&self) {
        self.released.store(true, Ordering::Release);
        self.release_changed.notify_waiters();
    }

    async fn pause(&self) {
        self.entered.store(true, Ordering::Release);
        self.entered_changed.notify_waiters();
        while !self.released.load(Ordering::Acquire) {
            let changed = self.release_changed.notified();
            if self.released.load(Ordering::Acquire) {
                break;
            }
            changed.await;
        }
    }
}

pub(crate) struct CheckpointRuntimeSpec {
    storage: CheckpointRuntimeStorage,
    config: StreamRuntimeConfig,
    #[cfg(test)]
    faults: CheckpointFaultInjector,
    #[cfg(test)]
    started_gate: Option<CheckpointStartedTestGate>,
}

enum CheckpointRuntimeStorage {
    LegacyParts {
        state_backend: Arc<dyn StateBackend>,
        manifest_root: PathBuf,
    },
    Managed(ManagedCheckpointRuntime),
    #[cfg(test)]
    ManagedTestParts {
        state_backend: Arc<dyn StateBackend>,
        manifest_root: PathBuf,
    },
}

impl CheckpointRuntimeSpec {
    pub(crate) fn new(
        state_backend: Arc<dyn StateBackend>,
        manifest_root: impl Into<PathBuf>,
        config: StreamRuntimeConfig,
    ) -> crate::Result<Self> {
        validate_checkpoint_config(&config)?;
        Ok(Self {
            storage: CheckpointRuntimeStorage::LegacyParts {
                state_backend,
                manifest_root: manifest_root.into(),
            },
            config,
            #[cfg(test)]
            faults: CheckpointFaultInjector::default(),
            #[cfg(test)]
            started_gate: None,
        })
    }

    fn managed(
        storage: ManagedCheckpointRuntime,
        config: StreamRuntimeConfig,
    ) -> crate::Result<Self> {
        validate_checkpoint_config(&config)?;
        Ok(Self {
            storage: CheckpointRuntimeStorage::Managed(storage),
            config,
            #[cfg(test)]
            faults: CheckpointFaultInjector::default(),
            #[cfg(test)]
            started_gate: None,
        })
    }

    #[cfg(test)]
    fn managed_test_parts(
        state_backend: Arc<dyn StateBackend>,
        manifest_root: impl Into<PathBuf>,
        config: StreamRuntimeConfig,
    ) -> crate::Result<Self> {
        validate_checkpoint_config(&config)?;
        Ok(Self {
            storage: CheckpointRuntimeStorage::ManagedTestParts {
                state_backend,
                manifest_root: manifest_root.into(),
            },
            config,
            faults: CheckpointFaultInjector::default(),
            started_gate: None,
        })
    }

    #[cfg(test)]
    pub(crate) fn with_fault(
        mut self,
        point: CheckpointFaultPoint,
        mode: CheckpointFaultMode,
    ) -> Self {
        self.faults = CheckpointFaultInjector::armed(point, mode);
        self
    }

    #[cfg(test)]
    pub(crate) fn with_fault_probe(
        mut self,
        point: CheckpointFaultPoint,
        mode: CheckpointFaultMode,
    ) -> (Self, CheckpointFaultInjector) {
        let faults = CheckpointFaultInjector::armed(point, mode);
        self.faults = faults.clone();
        (self, faults)
    }

    #[cfg(test)]
    pub(crate) fn with_started_gate(mut self, gate: CheckpointStartedTestGate) -> Self {
        self.started_gate = Some(gate);
        self
    }
}

fn validate_checkpoint_config(config: &StreamRuntimeConfig) -> crate::Result<()> {
    config.validate()?;
    if config.retained_epochs == 0 {
        return Err(CalcFlowError::InvalidArgument {
            field: "retained_epochs".into(),
            message: "must be positive".into(),
        });
    }
    Ok(())
}

struct ValidatedCheckpointRuntime {
    spec: CheckpointRuntimeSpec,
    identity: PreparedManifestIdentity,
}

struct OpenedCheckpointRuntime {
    transaction: Arc<ManifestTransaction>,
    _managed_storage: Option<OpenedManagedCheckpointRuntime>,
    identity: PreparedManifestIdentity,
    config: StreamRuntimeConfig,
    selected: Option<SelectedManifest>,
    next_epoch: Epoch,
    status: CheckpointStatusHandle,
    startup_orphans_removed: usize,
    managed: bool,
    #[cfg(test)]
    faults: CheckpointFaultInjector,
    #[cfg(test)]
    started_gate: Option<CheckpointStartedTestGate>,
}

impl OpenedCheckpointRuntime {
    #[cfg(test)]
    fn inject_fault(
        &self,
        point: CheckpointFaultPoint,
        cancellation: &CancellationToken,
    ) -> crate::Result<bool> {
        let trigger_count = self.faults.trigger_count();
        self.faults.trigger(point, cancellation)?;
        Ok(self.faults.trigger_count() != trigger_count && cancellation.is_cancelled())
    }

    #[cfg(test)]
    async fn pause_after_started(&self) {
        if let Some(gate) = &self.started_gate {
            gate.pause().await;
        }
    }
}

#[derive(Clone, Debug, Eq, Ord, PartialEq, PartialOrd)]
pub(crate) enum FailureOrigin {
    Preflight,
    RunnerLifecycle,
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
    SinkCheckpoint {
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
    pub(crate) cleanup_failures: Vec<Arc<RuntimeFailure>>,
}

pub(crate) fn runner_shutdown_failure(error: CalcFlowError) -> Arc<RuntimeFailure> {
    Arc::new(RuntimeFailure {
        origin: FailureOrigin::RunnerLifecycle,
        error,
    })
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
    pipeline_name: String,
    state: Mutex<JobCoreState>,
    terminal_arbiter: TerminalArbiter,
    changed: Notify,
    launch_cancel: CancellationToken,
    runner_commands: mpsc::UnboundedSender<RunnerCommand>,
    metrics: MetricsRecorder,
    status_projection: StatusProjection,
    runtime_status: Mutex<RuntimeStatus>,
    checkpoint_enabled: bool,
    manual_checkpoint: Mutex<Option<CheckpointCoordinatorHandle>>,
    operation_cancel_requested: AtomicBool,
    #[cfg(test)]
    terminal_commit_seam: Mutex<Option<TerminalCommitTestSeam>>,
    #[cfg(test)]
    launch_probe: Option<Arc<TestLaunchProbe>>,
}

#[cfg(test)]
struct TerminalCommitTestSeam {
    reached: tokio::sync::oneshot::Sender<()>,
    release: tokio::sync::oneshot::Receiver<()>,
}

#[cfg(test)]
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum TestLaunchCheckpoint {
    AfterOperatorEntry,
    LivePublished,
}

#[cfg(test)]
struct TestLaunchProbe {
    checkpoint: TestLaunchCheckpoint,
    reached: AtomicBool,
    released: AtomicBool,
    changed: Notify,
}

#[cfg(test)]
impl TestLaunchProbe {
    fn new(checkpoint: TestLaunchCheckpoint) -> Self {
        Self {
            checkpoint,
            reached: AtomicBool::new(false),
            released: AtomicBool::new(false),
            changed: Notify::new(),
        }
    }

    async fn pause_at(&self, checkpoint: TestLaunchCheckpoint) {
        if self.checkpoint != checkpoint {
            return;
        }
        self.reached.store(true, Ordering::Release);
        self.changed.notify_waiters();
        loop {
            let notified = self.changed.notified();
            if self.released.load(Ordering::Acquire) {
                return;
            }
            notified.await;
        }
    }

    async fn wait_until_reached(&self) {
        loop {
            let notified = self.changed.notified();
            if self.reached.load(Ordering::Acquire) {
                return;
            }
            notified.await;
        }
    }

    fn release(&self) {
        self.released.store(true, Ordering::Release);
        self.changed.notify_waiters();
    }
}

#[derive(Default)]
struct RuntimeStatus {
    tasks: TaskRegistry,
    sources: BTreeMap<String, SourceProgress>,
    nodes: BTreeMap<String, OperatorProgress>,
    sinks: BTreeMap<String, SinkProgress>,
    sink_outputs: BTreeMap<String, String>,
    progress: Option<LiveProgressStatusHandle>,
    checkpoint: Option<CheckpointStatusHandle>,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) enum CheckpointFailureCategory {
    Timeout,
    Protocol,
    Io,
    Maintenance,
    Runtime,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub(crate) struct CheckpointStatus {
    pub(crate) current_epoch: Option<Epoch>,
    pub(crate) phase: Option<CheckpointPhase>,
    pub(crate) terminal: bool,
    pub(crate) source_acknowledgements: usize,
    pub(crate) operator_acknowledgements: usize,
    pub(crate) sink_precommit_acknowledgements: usize,
    pub(crate) sink_commit_acknowledgements: usize,
    pub(crate) expected_sources: usize,
    pub(crate) expected_operators: usize,
    pub(crate) expected_sinks: usize,
    pub(crate) elapsed: Option<Duration>,
    pub(crate) last_completed_epoch: Option<Epoch>,
    pub(crate) installed_unknown_epoch: Option<Epoch>,
    pub(crate) failure_category: Option<CheckpointFailureCategory>,
    pub(crate) runtime_config_changed: bool,
}

struct CheckpointStatusState {
    snapshot: CheckpointStatus,
    started: Option<tokio::time::Instant>,
}

#[derive(Clone)]
struct CheckpointStatusHandle(Arc<Mutex<CheckpointStatusState>>);

impl CheckpointStatusHandle {
    fn new(identity: &PreparedManifestIdentity, selected: Option<&SelectedManifest>) -> Self {
        Self(Arc::new(Mutex::new(CheckpointStatusState {
            snapshot: CheckpointStatus {
                current_epoch: None,
                phase: None,
                terminal: false,
                source_acknowledgements: 0,
                operator_acknowledgements: 0,
                sink_precommit_acknowledgements: 0,
                sink_commit_acknowledgements: 0,
                expected_sources: identity.source_ids.len(),
                expected_operators: identity.operator_ids.len(),
                expected_sinks: identity.sink_ids.len(),
                elapsed: None,
                last_completed_epoch: selected.map(|selected| selected.manifest.epoch()),
                installed_unknown_epoch: None,
                failure_category: None,
                runtime_config_changed: selected
                    .is_some_and(|selected| selected.validation.runtime_config_changed),
            },
            started: None,
        })))
    }

    fn snapshot(&self) -> CheckpointStatus {
        let state = self.0.lock();
        let mut snapshot = state.snapshot.clone();
        snapshot.elapsed = state.started.map(|started| started.elapsed());
        snapshot
    }

    fn set_expected(&self, sources: usize, operators: usize, sinks: usize) {
        let mut state = self.0.lock();
        state.snapshot.expected_sources = sources;
        state.snapshot.expected_operators = operators;
        state.snapshot.expected_sinks = sinks;
    }

    fn start(&self, epoch: Epoch, terminal: bool) {
        let mut state = self.0.lock();
        state.snapshot.current_epoch = Some(epoch);
        state.snapshot.phase = Some(CheckpointPhase::Requested);
        state.snapshot.terminal = terminal;
        state.snapshot.source_acknowledgements = 0;
        state.snapshot.operator_acknowledgements = 0;
        state.snapshot.sink_precommit_acknowledgements = 0;
        state.snapshot.sink_commit_acknowledgements = 0;
        state.snapshot.installed_unknown_epoch = None;
        state.snapshot.failure_category = None;
        state.started = Some(tokio::time::Instant::now());
    }

    fn promote_terminal(&self, epoch: Epoch) -> crate::Result<()> {
        let mut state = self.0.lock();
        if state.snapshot.current_epoch != Some(epoch) {
            return Err(checkpoint_protocol_error(
                epoch,
                "terminal promotion does not match the active checkpoint",
            ));
        }
        state.snapshot.terminal = true;
        Ok(())
    }

    fn advance(&self, epoch: Epoch, phase: CheckpointPhase) {
        let mut state = self.0.lock();
        if state.snapshot.current_epoch == Some(epoch) {
            state.snapshot.phase = Some(phase);
        }
    }

    fn acknowledge_sources(&self, epoch: Epoch, count: usize) {
        self.acknowledge(epoch, |status| {
            status.source_acknowledgements = count;
        });
    }

    fn acknowledge_operators(&self, epoch: Epoch, count: usize) {
        self.acknowledge(epoch, |status| {
            status.operator_acknowledgements = count;
        });
    }

    fn acknowledge_sink_precommits(&self, epoch: Epoch, count: usize) {
        self.acknowledge(epoch, |status| {
            status.sink_precommit_acknowledgements = count;
        });
    }

    fn acknowledge_sink_commits(&self, epoch: Epoch, count: usize) {
        self.acknowledge(epoch, |status| {
            status.sink_commit_acknowledgements = count;
        });
    }

    fn acknowledge(&self, epoch: Epoch, update: impl FnOnce(&mut CheckpointStatus)) {
        let mut state = self.0.lock();
        if state.snapshot.current_epoch == Some(epoch) {
            update(&mut state.snapshot);
        }
    }

    fn sinks_committed(&self, epoch: Epoch) {
        let mut state = self.0.lock();
        if state.snapshot.current_epoch == Some(epoch) {
            state.snapshot.phase = Some(CheckpointPhase::SinksCommitted);
            state.snapshot.last_completed_epoch = Some(epoch);
            state.snapshot.installed_unknown_epoch = None;
        }
    }

    fn installed_unknown(&self, epoch: Epoch) {
        let mut state = self.0.lock();
        if state.snapshot.current_epoch == Some(epoch) {
            state.snapshot.installed_unknown_epoch = Some(epoch);
            state.snapshot.failure_category = Some(CheckpointFailureCategory::Runtime);
        }
    }

    fn complete(&self, epoch: Epoch) {
        let mut state = self.0.lock();
        if state.snapshot.current_epoch == Some(epoch) {
            state.snapshot.current_epoch = None;
            state.snapshot.phase = None;
            state.snapshot.terminal = false;
            state.snapshot.source_acknowledgements = 0;
            state.snapshot.operator_acknowledgements = 0;
            state.snapshot.sink_precommit_acknowledgements = 0;
            state.snapshot.sink_commit_acknowledgements = 0;
            state.snapshot.elapsed = None;
            state.started = None;
        }
    }

    fn fail(&self, category: CheckpointFailureCategory) {
        let mut state = self.0.lock();
        state.snapshot.failure_category = Some(category);
    }

    fn fail_if_unset(&self, category: CheckpointFailureCategory) {
        let mut state = self.0.lock();
        if state.snapshot.failure_category.is_none() {
            state.snapshot.failure_category = Some(category);
        }
    }

    fn cancel(&self) {
        let mut state = self.0.lock();
        state.snapshot.current_epoch = None;
        state.snapshot.phase = None;
        state.snapshot.terminal = false;
        state.snapshot.source_acknowledgements = 0;
        state.snapshot.operator_acknowledgements = 0;
        state.snapshot.sink_precommit_acknowledgements = 0;
        state.snapshot.sink_commit_acknowledgements = 0;
        state.snapshot.elapsed = None;
        state.started = None;
    }
}

impl JobCore {
    fn new(
        launch_id: LaunchId,
        job_id: u64,
        runner_commands: mpsc::UnboundedSender<RunnerCommand>,
        metrics: MetricsRecorder,
        status_projection: StatusProjection,
        checkpoint_enabled: bool,
        pipeline_name: String,
    ) -> Self {
        let sink_outputs = status_projection.sink_outputs();
        Self {
            launch_id,
            job_id,
            pipeline_name,
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
            status_projection,
            runtime_status: Mutex::new(RuntimeStatus {
                sink_outputs,
                ..RuntimeStatus::default()
            }),
            checkpoint_enabled,
            manual_checkpoint: Mutex::new(None),
            operation_cancel_requested: AtomicBool::new(false),
            #[cfg(test)]
            terminal_commit_seam: Mutex::new(None),
            #[cfg(test)]
            launch_probe: None,
        }
    }

    fn request_cancel(&self, reaper_owned: bool) {
        self.operation_cancel_requested
            .store(true, Ordering::Release);
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
        self.operation_cancel_requested
            .store(true, Ordering::Release);
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
        self.operation_cancel_requested
            .store(true, Ordering::Release);
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
    pub(crate) late_rows: u64,
    pub(crate) affected_batches: u64,
    pub(crate) max_lateness_micros: Option<u64>,
    pub(crate) null_event_time_rows: u64,
    pub(crate) null_event_time_batches: u64,
}

impl From<OperatorProgressSnapshot> for OperatorStatus {
    fn from(progress: OperatorProgressSnapshot) -> Self {
        Self {
            input_batches: progress.input_batches,
            fully_fanned_out_batches: progress.fully_fanned_out_batches,
            datafusion_runtime_created: progress.datafusion_runtime_created,
            on_end_calls: progress.on_end_calls,
            ended: progress.ended,
            late_rows: progress.late_rows,
            affected_batches: progress.affected_batches,
            max_lateness_micros: progress.max_lateness_micros,
            null_event_time_rows: progress.null_event_time_rows,
            null_event_time_batches: progress.null_event_time_batches,
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
    pub(crate) progress: Option<LiveProgressEvidence>,
    pub(crate) checkpoint: Option<CheckpointStatus>,
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
        let progress = runtime
            .progress
            .as_ref()
            .map(LiveProgressStatusHandle::snapshot);
        let checkpoint = runtime
            .checkpoint
            .as_ref()
            .map(CheckpointStatusHandle::snapshot);
        ContinuousJobStatus {
            job_id: self.core.job_id,
            state,
            terminal_cause,
            tasks,
            edges,
            sources,
            nodes,
            sinks,
            progress,
            checkpoint,
            metrics,
        }
    }

    pub(crate) fn public_status(&self) -> JobStatus {
        self.core.status_projection.project(&self.status())
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

    pub(crate) async fn trigger_checkpoint(&self) -> crate::Result<Epoch> {
        if !self.core.checkpoint_enabled {
            return Err(CalcFlowError::InvalidArgument {
                field: "runtime.checkpoint".into(),
                message: "manual checkpoints require a checkpoint runtime".into(),
            });
        }
        let coordinator = loop {
            let changed = self.core.changed.notified();
            if let Some(coordinator) = self.core.manual_checkpoint.lock().clone() {
                break coordinator;
            }
            if let Some(outcome) = self.core.state.lock().outcome.clone() {
                return Err(manual_checkpoint_terminal_error(
                    self.core.job_id,
                    &self.core.pipeline_name,
                    &outcome,
                ));
            }
            changed.await;
        };
        let result = coordinator.request_manual().await?.await;
        if matches!(&result, Err(CalcFlowError::Cancelled { .. })) {
            let outcome = self.wait().await;
            if !matches!(
                outcome.cause,
                TerminalCause::ExplicitCancel | TerminalCause::DeadlineExceeded
            ) {
                let status = self.status();
                if let Some(error) = super::projection::project_manual_terminal_outcome(
                    self.core.job_id,
                    &outcome,
                    status.checkpoint.as_ref(),
                ) {
                    return Err(CalcFlowError::Streaming(error));
                }
            }
        }
        result
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

fn manual_checkpoint_terminal_error(
    job_id: u64,
    pipeline_name: &str,
    outcome: &ContinuousJobOutcome,
) -> CalcFlowError {
    match outcome.state {
        ContinuousJobState::Cancelled => CalcFlowError::Cancelled {
            run_id: format!("streaming-job-{job_id}"),
        },
        ContinuousJobState::RecoveryRequired => CalcFlowError::RecoveryRequired {
            pipeline_name: pipeline_name.into(),
            message: "job terminated while the manual checkpoint was pending".into(),
        },
        _ => CalcFlowError::Internal {
            message: format!(
                "streaming job {job_id} terminated before the manual checkpoint completed"
            ),
        },
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
        cleanup_failures: Vec::new(),
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
    stop_after_first_job: bool,
    registry: Mutex<RunnerRegistryState>,
    driver: Mutex<Option<JoinHandle<()>>>,
    diagnostics: RunnerDiagnostics,
    next_launch_id: AtomicU64,
    closed: AtomicBool,
    changed: Notify,
    #[cfg(test)]
    abandonment_warnings: AtomicU64,
    #[cfg(test)]
    next_launch_probe: Mutex<Option<Arc<TestLaunchProbe>>>,
    #[cfg(test)]
    panic_lifecycle_after_shutdown: AtomicBool,
}

const ABANDONED_RUNNER_WARNING: &str =
    "continuous runner dropped before shutdown completed; cancellation requested";

enum RunnerCommand {
    Start {
        launch_id: LaunchId,
        core: Arc<JobCore>,
        job: Box<ValidatedContinuousJob>,
        checkpoint: Option<Box<ValidatedCheckpointRuntime>>,
    },
    Wake(LaunchId),
    Shutdown,
}

pub(crate) struct ContinuousRunner {
    core: Arc<RunnerCore>,
}

#[cfg(test)]
#[derive(Clone)]
pub(crate) struct RunnerLifecycleProbe {
    core: Arc<RunnerCore>,
}

#[cfg(test)]
impl RunnerLifecycleProbe {
    pub(crate) fn registry_counts(&self) -> (usize, usize) {
        let registry = self.core.registry.lock();
        (registry.live_jobs.len(), registry.reaper_jobs.len())
    }

    pub(crate) fn is_finished(&self) -> bool {
        self.core.closed.load(Ordering::Acquire) && self.core.driver.lock().is_none()
    }

    pub(crate) async fn join(&self) -> crate::Result<()> {
        RunnerShutdownObserver::new(Arc::clone(&self.core)).await
    }
}

/// Crate-private one-shot ownership boundary used by the public continuous facade.
pub(crate) struct OneShotContinuousRunner {
    runner: ContinuousRunner,
}

pub(crate) struct OneShotStartObserver {
    inner: Pin<Box<dyn Future<Output = StartResult<OwningContinuousJob>> + Send>>,
}

impl OneShotContinuousRunner {
    pub(crate) fn new() -> Self {
        Self {
            runner: ContinuousRunner::new_one_shot(),
        }
    }

    pub(crate) fn start(self, spec: ContinuousJobSpec) -> OneShotStartObserver {
        let runner = self.runner;
        let start = runner.start(spec);
        OneShotStartObserver::new(runner, start)
    }

    pub(crate) fn start_checkpointed(
        self,
        spec: ContinuousJobSpec,
        checkpoint: ManagedCheckpointRuntime,
    ) -> OneShotStartObserver {
        self.start_checkpointed_with_config(spec, checkpoint, StreamRuntimeConfig::default())
    }

    pub(crate) fn start_checkpointed_with_config(
        self,
        spec: ContinuousJobSpec,
        checkpoint: ManagedCheckpointRuntime,
        config: StreamRuntimeConfig,
    ) -> OneShotStartObserver {
        let runner = self.runner;
        let start = match CheckpointRuntimeSpec::managed(checkpoint, config) {
            Ok(checkpoint) => runner.start_checkpointed(spec, checkpoint),
            Err(error) => preflight_error_observer(error),
        };
        OneShotStartObserver::new(runner, start)
    }

    #[cfg(test)]
    pub(crate) fn start_checkpointed_with_config_and_fault(
        self,
        spec: ContinuousJobSpec,
        checkpoint: ManagedCheckpointRuntime,
        config: StreamRuntimeConfig,
        point: CheckpointFaultPoint,
        mode: CheckpointFaultMode,
    ) -> OneShotStartObserver {
        let runner = self.runner;
        let start = match CheckpointRuntimeSpec::managed(checkpoint, config) {
            Ok(checkpoint) => runner.start_checkpointed(spec, checkpoint.with_fault(point, mode)),
            Err(error) => preflight_error_observer(error),
        };
        OneShotStartObserver::new(runner, start)
    }

    #[cfg(test)]
    pub(crate) fn start_checkpointed_with_config_and_fault_probe(
        self,
        spec: ContinuousJobSpec,
        checkpoint: ManagedCheckpointRuntime,
        config: StreamRuntimeConfig,
        point: CheckpointFaultPoint,
        mode: CheckpointFaultMode,
    ) -> (OneShotStartObserver, CheckpointFaultInjector) {
        let runner = self.runner;
        let (start, probe) = match CheckpointRuntimeSpec::managed(checkpoint, config) {
            Ok(checkpoint) => {
                let (checkpoint, probe) = checkpoint.with_fault_probe(point, mode);
                (runner.start_checkpointed(spec, checkpoint), probe)
            }
            Err(error) => (
                preflight_error_observer(error),
                CheckpointFaultInjector::armed(point, mode),
            ),
        };
        (OneShotStartObserver::new(runner, start), probe)
    }

    #[cfg(test)]
    fn panic_lifecycle_after_shutdown_for_test(&self) {
        self.runner
            .core
            .panic_lifecycle_after_shutdown
            .store(true, Ordering::Release);
    }
}

impl OneShotStartObserver {
    fn new(mut runner: ContinuousRunner, start: StartObserver) -> Self {
        Self {
            inner: Box::pin(async move {
                match start.await {
                    Ok(job) => Ok(OwningContinuousJob::new(job, runner)),
                    Err(mut failure) => {
                        if let Err(error) = runner.shutdown().await {
                            failure
                                .cleanup_failures
                                .push(runner_shutdown_failure(error));
                        }
                        Err(failure)
                    }
                }
            }),
        }
    }
}

impl Future for OneShotStartObserver {
    type Output = StartResult<OwningContinuousJob>;

    fn poll(mut self: Pin<&mut Self>, context: &mut Context<'_>) -> Poll<Self::Output> {
        self.inner.as_mut().poll(context)
    }
}

impl ContinuousRunner {
    pub(crate) fn new() -> Self {
        Self::new_with_stop_after_first_job(false)
    }

    fn new_one_shot() -> Self {
        Self::new_with_stop_after_first_job(true)
    }

    fn new_with_stop_after_first_job(stop_after_first_job: bool) -> Self {
        let (commands, receiver) = mpsc::unbounded_channel();
        let core = Arc::new(RunnerCore {
            commands,
            root_cancel: CancellationToken::new(),
            stop_after_first_job,
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
            #[cfg(test)]
            next_launch_probe: Mutex::new(None),
            #[cfg(test)]
            panic_lifecycle_after_shutdown: AtomicBool::new(false),
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

    #[cfg(test)]
    pub(crate) fn lifecycle_probe(&self) -> RunnerLifecycleProbe {
        RunnerLifecycleProbe {
            core: Arc::clone(&self.core),
        }
    }

    pub(crate) fn start(&self, spec: ContinuousJobSpec) -> StartObserver {
        self.start_internal(spec, None)
    }

    pub(crate) fn start_checkpointed(
        &self,
        spec: ContinuousJobSpec,
        checkpoint: CheckpointRuntimeSpec,
    ) -> StartObserver {
        self.start_internal(spec, Some(checkpoint))
    }

    #[allow(
        clippy::too_many_lines,
        reason = "preflight, registration, and ownership publication form one synchronous launch transaction"
    )]
    fn start_internal(
        &self,
        spec: ContinuousJobSpec,
        checkpoint: Option<CheckpointRuntimeSpec>,
    ) -> StartObserver {
        #[cfg(test)]
        let launch_probe = self.core.next_launch_probe.lock().take();
        let runtime_config_hash = match checkpoint.as_ref() {
            Some(checkpoint) => match spec.plan.runtime_config_hash(&checkpoint.config) {
                Ok(hash) => Some(hash),
                Err(error) => return preflight_error_observer(error),
            },
            None => None,
        };
        let validated = match preflight_job(spec) {
            Ok(validated) => validated,
            Err(error) => {
                return StartObserver::ready(Err(StartFailure {
                    primary: Arc::new(RuntimeFailure {
                        origin: FailureOrigin::Preflight,
                        error,
                    }),
                    diagnostic_id: None,
                    cleanup_failures: Vec::new(),
                }));
            }
        };
        if checkpoint.is_none()
            && let Some(output_id) =
                validated
                    .plan
                    .requirements
                    .delivery
                    .iter()
                    .find_map(|(output_id, guarantee)| {
                        (*guarantee == crate::DeliveryGuarantee::ExactlyOnce)
                            .then_some(output_id.as_str())
                    })
        {
            return preflight_error_observer(CalcFlowError::InvalidArgument {
                field: format!("requirements.delivery.{output_id}"),
                message: "exactly-once delivery requires a checkpoint runtime".into(),
            });
        }
        if checkpoint.is_some()
            && let Err(error) = validate_checkpoint_operator_capabilities(&validated.plan)
        {
            return preflight_error_observer(error);
        }
        let checkpoint = match checkpoint {
            Some(spec) => {
                let identity = match checkpoint_identity(
                    &validated,
                    runtime_config_hash.expect("checkpoint configuration was hashed"),
                ) {
                    Ok(identity) => identity,
                    Err(error) => return preflight_error_observer(error),
                };
                Some(Box::new(ValidatedCheckpointRuntime { spec, identity }))
            }
            None => None,
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
                cleanup_failures: Vec::new(),
            }));
        };
        let launch_id = LaunchId(launch_id);
        let job_id = validated.context.job_id();
        let status_projection = StatusProjection::new(&validated);
        let metrics = metrics_for_job(&validated);
        let core = JobCore::new(
            launch_id,
            job_id,
            self.core.commands.clone(),
            metrics,
            status_projection,
            checkpoint.is_some(),
            validated.plan.name.clone(),
        );
        #[cfg(test)]
        let core = JobCore {
            launch_probe,
            ..core
        };
        let core = Arc::new(core);
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
                checkpoint,
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
                cleanup_failures: Vec::new(),
            }));
        }
        StartObserver::observe(core)
    }

    #[cfg(test)]
    fn start_with_test_launch_probe(
        &self,
        spec: ContinuousJobSpec,
        probe: Arc<TestLaunchProbe>,
    ) -> StartObserver {
        let previous = self.core.next_launch_probe.lock().replace(probe);
        assert!(
            previous.is_none(),
            "test launch probe slot was already occupied"
        );
        self.start(spec)
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

fn validate_checkpoint_operator_capabilities(plan: &StreamRuntimePlanParts) -> crate::Result<()> {
    for node in &plan.nodes {
        let operator_id = node.operator_id.as_str();
        match node.checkpoint_capability {
            OperatorCheckpointCapability::Stateless => {}
            OperatorCheckpointCapability::CheckpointedStateful { state_version }
                if state_version > 0 => {}
            OperatorCheckpointCapability::CheckpointedStateful { .. } => {
                return Err(CalcFlowError::InvalidArgument {
                    field: format!("operators.{operator_id}.checkpoint_capability"),
                    message: "checkpoint state version must be greater than zero".into(),
                });
            }
            OperatorCheckpointCapability::Unproven => {
                return Err(CalcFlowError::InvalidArgument {
                    field: format!("operators.{operator_id}.checkpoint_capability"),
                    message: "operator checkpoint capability is unproven".into(),
                });
            }
        }
    }
    Ok(())
}

fn metrics_for_job(job: &ValidatedContinuousJob) -> MetricsRecorder {
    let sink_metric_ids = job
        .sinks
        .iter()
        .flat_map(|(output_id, sinks)| {
            sinks
                .iter()
                .map(move |sink| sink_metric_id(output_id, sink.sink_id.as_str()))
        })
        .collect::<Vec<_>>();
    MetricsRecorder::new(
        job.plan
            .edges
            .iter()
            .map(|(edge_id, edge)| (edge_id.clone(), edge.budget)),
        job.plan.source_routes.keys().cloned(),
        job.plan.nodes.iter().map(|node| node.node_id.clone()),
        sink_metric_ids,
    )
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
        cleanup_failures: Vec::new(),
    }))
}

fn preflight_error_observer(error: CalcFlowError) -> StartObserver {
    StartObserver::ready(Err(StartFailure {
        primary: Arc::new(RuntimeFailure {
            origin: FailureOrigin::Preflight,
            error,
        }),
        diagnostic_id: None,
        cleanup_failures: Vec::new(),
    }))
}

fn checkpoint_identity(
    job: &ValidatedContinuousJob,
    runtime_config_hash: String,
) -> crate::Result<PreparedManifestIdentity> {
    let mut sink_outputs = BTreeMap::<String, String>::new();
    for (output_id, sinks) in &job.sinks {
        for sink in sinks {
            if let Some(previous_output) =
                sink_outputs.insert(sink.sink_id.to_string(), output_id.clone())
            {
                return Err(CalcFlowError::InvalidArgument {
                    field: format!("sinks.{}", sink.sink_id),
                    message: format!(
                        "sink ID is bound to more than one output: {previous_output:?} and {output_id:?}"
                    ),
                });
            }
        }
    }
    Ok(PreparedManifestIdentity {
        pipeline_name: job.plan.name.clone(),
        pipeline_fingerprint: job.plan.fingerprint.clone(),
        runtime_config_hash,
        source_ids: job.plan.source_routes.keys().cloned().collect(),
        operator_ids: job
            .plan
            .nodes
            .iter()
            .map(|node| node.operator_id.as_str().to_owned())
            .collect(),
        sink_ids: sink_outputs.into_keys().collect(),
    })
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
    checkpoint: Option<Box<ValidatedCheckpointRuntime>>,
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
                Some(RunnerCommand::Start { launch_id, core: job_core, job, checkpoint }) if active.is_none() && !shutting_down => {
                    active = Some(launch_id);
                    job_core.state.lock().owner = DriverOwnership::Driving;
                    drivers.spawn(run_job_driver(launch_id, Arc::clone(&job_core), *job, checkpoint.map(|checkpoint| *checkpoint)));
                }
                Some(RunnerCommand::Start { launch_id, core: job_core, job, checkpoint }) if !shutting_down => {
                    pending = Some(PendingStart { launch_id, core: job_core, job, checkpoint });
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
                    if runner_core.stop_after_first_job {
                        shutting_down = true;
                        begin_lifecycle_shutdown(&runner_core, &mut pending);
                    } else if !shutting_down && let Some(next) = pending.take() {
                        active = Some(next.launch_id);
                        next.core.state.lock().owner = DriverOwnership::Driving;
                        drivers.spawn(run_job_driver(
                            next.launch_id,
                            Arc::clone(&next.core),
                            *next.job,
                            next.checkpoint.map(|checkpoint| *checkpoint),
                        ));
                    }
                }
            }
        }
    }
    runner_core.closed.store(true, Ordering::Release);
    runner_core.changed.notify_waiters();
    #[cfg(test)]
    assert!(
        !runner_core
            .panic_lifecycle_after_shutdown
            .load(Ordering::Acquire),
        "injected runner lifecycle shutdown panic"
    );
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
    remove_job_registration(&mut registry, job.launch_id);
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
                cleanup_failures: Vec::new(),
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
    remove_job_registration(&mut registry, report.launch_id);
    drop(state);
    drop(registry);
    job.changed.notify_waiters();
}

fn remove_job_registration(registry: &mut RunnerRegistryState, launch_id: LaunchId) {
    registry.live_jobs.remove(&launch_id);
    registry.reaper_jobs.remove(&launch_id);
    if registry.provisional == Some(launch_id) {
        registry.provisional = None;
    }
    if registry.pending_start == Some(launch_id) {
        registry.pending_start = None;
    }
}

enum ConnectorResource {
    Source {
        binding_id: String,
        binding: Box<SourceBinding>,
    },
    Sink {
        output_id: String,
        sink_id: StableSinkId,
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
                sink_id: sink_id.to_string(),
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
                sink_id: sink_id.to_string(),
            },
        }
    }

    async fn open(&mut self) -> crate::Result<()> {
        match self {
            Self::Source { binding, .. } => binding.open().await,
            Self::Sink { binding, .. } => binding.open().await,
        }
    }

    async fn close(&mut self) -> crate::Result<()> {
        match self {
            Self::Source { binding, .. } => binding.close().await,
            Self::Sink { binding, .. } => binding.close().await,
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

#[allow(
    clippy::too_many_lines,
    reason = "gated recovery and the existing launch lifecycle share one fail-closed ownership path"
)]
async fn run_job_driver(
    launch_id: LaunchId,
    core: Arc<JobCore>,
    validated: ValidatedContinuousJob,
    checkpoint: Option<ValidatedCheckpointRuntime>,
) -> DriverReport {
    let ValidatedContinuousJob {
        context,
        plan,
        mut sources,
        sinks,
        progress: prepared_progress,
        delivery_mode: _,
        delivery_proofs: _,
    } = validated;
    let cancellation = context.cancellation().clone();
    let checkpoint = match checkpoint {
        Some(checkpoint) => match open_checkpoint_runtime(checkpoint, &cancellation).await {
            Ok(checkpoint) => {
                if let Err(error) = core
                    .metrics
                    .record_checkpoint_orphan_cleanup(checkpoint.startup_orphans_removed)
                {
                    return checkpoint_start_failure(launch_id, error);
                }
                Some(checkpoint)
            }
            Err(error) => {
                return DriverReport {
                    launch_id,
                    completion: DriverCompletion::StartFailed(StartFailure {
                        primary: Arc::new(RuntimeFailure {
                            origin: FailureOrigin::Preflight,
                            error,
                        }),
                        diagnostic_id: None,
                        cleanup_failures: Vec::new(),
                    }),
                    cleanup_failures: Vec::new(),
                };
            }
        },
        None => None,
    };
    if let Some(checkpoint) = checkpoint.as_ref() {
        core.runtime_status.lock().checkpoint = Some(checkpoint.status.clone());
    }
    if let Some(checkpoint) = checkpoint.as_ref()
        && let Some(selected) = checkpoint.selected.as_ref()
    {
        if let Err(error) = validate_manifest_operator_capabilities(&selected.manifest, &plan) {
            return checkpoint_start_failure(launch_id, error);
        }
        match manifest_is_terminal(&selected.manifest, &plan) {
            Ok(true) => {
                drop(sources);
                return recover_terminal_manifest(
                    launch_id,
                    &core,
                    &checkpoint.transaction,
                    &checkpoint.identity,
                    selected,
                    sinks,
                    checkpoint.managed,
                )
                .await;
            }
            Ok(false) => {}
            Err(error) => {
                return checkpoint_start_failure(
                    launch_id,
                    sanitize_managed_recovery_error(error, checkpoint.managed),
                );
            }
        }
    }
    let recovery_timer = checkpoint
        .as_ref()
        .and_then(|checkpoint| checkpoint.selected.as_ref())
        .map(|_| core.metrics.timer());
    let (operator_restores, durable_progress) = match checkpoint.as_ref() {
        Some(checkpoint) => {
            match prepare_checkpoint_recovery(
                checkpoint,
                &plan,
                &prepared_progress,
                &mut sources,
                &cancellation,
            )
            .await
            {
                Ok(restored) => restored,
                Err(error) => {
                    return checkpoint_start_failure(
                        launch_id,
                        sanitize_managed_recovery_error(error, checkpoint.managed),
                    );
                }
            }
        }
        None => (BTreeMap::new(), None),
    };
    let mut checkpoint_channels = checkpoint.as_ref().map(|checkpoint| {
        LiveCheckpointChannels::new(
            &plan,
            &sources,
            &sinks,
            Arc::clone(&checkpoint.transaction),
            #[cfg(test)]
            checkpoint.faults.clone(),
            #[cfg(test)]
            cancellation.clone(),
        )
    });
    let operator_checkpoint = checkpoint_channels
        .as_ref()
        .map(|channels| channels.operator.clone());
    let entry = run_operator_entry(
        plan,
        &context,
        &core,
        &cancellation,
        operator_restores,
        operator_checkpoint,
    )
    .await;
    let mut runtime = match entry {
        Ok(entry) => entry,
        Err(EntryFailure::Failed(primary)) => {
            let managed_recovery = checkpoint
                .as_ref()
                .is_some_and(|checkpoint| checkpoint.managed && checkpoint.selected.is_some());
            return DriverReport {
                launch_id,
                completion: DriverCompletion::StartFailed(StartFailure {
                    primary: sanitize_managed_recovery_failure(primary, managed_recovery),
                    diagnostic_id: None,
                    cleanup_failures: Vec::new(),
                }),
                cleanup_failures: Vec::new(),
            };
        }
        Err(EntryFailure::Cancelled) => {
            return cancelled_driver_report(launch_id, &core.metrics);
        }
    };
    if let Some(report) = cancel_after_operator_entry(launch_id, &core, &mut runtime).await {
        return report;
    }
    let mut resources = connector_resources(sources, sinks);
    let open_failures = if checkpoint.is_some() {
        open_checkpoint_connector_resources(&mut resources, &core.launch_cancel).await
    } else {
        open_connector_resources(&mut resources, &core.launch_cancel).await
    };
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
    let mut opened_sinks = opened_sinks;
    if let Some(checkpoint) = checkpoint.as_ref()
        && let Some(selected) = &checkpoint.selected
    {
        let recovery = recover_opened_sinks(&mut opened_sinks, &selected.manifest).await;
        let recovery = match recovery {
            Ok(()) => {
                let sink_retries = opened_sinks.values().map(Vec::len).sum();
                recovery_timer
                    .as_ref()
                    .expect("selected checkpoint created a restore timer")
                    .elapsed("checkpoint", "restore_duration")
                    .and_then(|elapsed| {
                        core.metrics
                            .record_checkpoint_restore(elapsed, sink_retries)
                    })
            }
            Err(error) => Err(sanitize_managed_recovery_error(error, checkpoint.managed)),
        };
        if let Err(error) = recovery {
            let mut resources = connector_resources(opened_sources, opened_sinks);
            return finish_failed_launch(
                launch_id,
                vec![Arc::new(RuntimeFailure {
                    origin: FailureOrigin::Preflight,
                    error,
                })],
                &mut resources,
                &mut runtime.supervisor,
            )
            .await;
        }
    }
    let task_progress = register_boundary_tasks(
        &mut runtime,
        &context,
        opened_sources,
        opened_sinks,
        prepared_progress,
        durable_progress.as_ref(),
        checkpoint,
        checkpoint_channels.take(),
        &core,
    );

    core.state.lock().launch_delivery = LaunchDeliveryState::ReadyUnclaimed;
    core.changed.notify_waiters();
    #[cfg(test)]
    if let Some(probe) = &core.launch_probe {
        probe.pause_at(TestLaunchCheckpoint::LivePublished).await;
    }
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

fn manifest_is_terminal(
    manifest: &CheckpointManifest,
    plan: &StreamRuntimePlanParts,
) -> crate::Result<bool> {
    let sources_terminal = manifest.sources().values().all(|source| source.ended);
    let mut operators_terminal = true;
    for node in &plan.nodes {
        let operator_id = node.operator_id.as_str();
        let entry = manifest
            .operators()
            .get(operator_id)
            .expect("selected manifest operator IDs were validated");
        let expected = node.ingress_edges.keys().cloned().collect::<BTreeSet<_>>();
        let actual = entry.progress.keys().cloned().collect::<BTreeSet<_>>();
        if actual != expected {
            return Err(CalcFlowError::CheckpointMismatch {
                message: format!(
                    "terminal recovery operator {operator_id:?} ingress IDs do not match the prepared plan"
                ),
            });
        }
        operators_terminal &= entry
            .progress
            .values()
            .all(|progress| matches!(progress.state, crate::ManifestIngressState::Ended));
    }
    if sources_terminal != operators_terminal {
        return Err(CalcFlowError::CheckpointMismatch {
            message: "terminal recovery source and operator end states disagree".into(),
        });
    }
    Ok(sources_terminal)
}

async fn recover_terminal_manifest(
    launch_id: LaunchId,
    core: &Arc<JobCore>,
    transaction: &Arc<ManifestTransaction>,
    identity: &PreparedManifestIdentity,
    selected: &SelectedManifest,
    sinks: BTreeMap<String, Vec<ValidatedOrdinarySink>>,
    managed: bool,
) -> DriverReport {
    let recovery_timer = core.metrics.timer();
    let mut resources = connector_resources(BTreeMap::new(), sinks);
    let open_failures = open_connector_resources(&mut resources, &core.launch_cancel).await;
    if !open_failures.is_empty() {
        let mut supervisor = TaskSupervisor::new(core.launch_cancel.clone());
        return finish_failed_launch(launch_id, open_failures, &mut resources, &mut supervisor)
            .await;
    }
    let (_, mut opened_sinks) = opened_connector_bindings(std::mem::take(&mut resources));
    if let Err(error) = recover_opened_sinks(&mut opened_sinks, &selected.manifest).await {
        let mut resources = connector_resources(BTreeMap::new(), opened_sinks);
        let mut supervisor = TaskSupervisor::new(core.launch_cancel.clone());
        return finish_failed_launch(
            launch_id,
            vec![Arc::new(RuntimeFailure {
                origin: FailureOrigin::Preflight,
                error: sanitize_managed_recovery_error(error, managed),
            })],
            &mut resources,
            &mut supervisor,
        )
        .await;
    }
    let sink_retries = opened_sinks.values().map(Vec::len).sum();
    let mut resources = connector_resources(BTreeMap::new(), opened_sinks);
    let close_failures = close_resources(&mut resources).await;
    if let Some(primary) = close_failures.first().cloned() {
        return DriverReport {
            launch_id,
            completion: DriverCompletion::StartFailed(StartFailure {
                primary,
                diagnostic_id: None,
                cleanup_failures: Vec::new(),
            }),
            cleanup_failures: close_failures.into_iter().skip(1).collect(),
        };
    }
    let retention = match transaction
        .retain_cancellable(identity, None, &core.launch_cancel)
        .await
    {
        Ok(report) => report,
        Err(error) => {
            return checkpoint_start_failure(
                launch_id,
                sanitize_managed_recovery_error(error, managed),
            );
        }
    };
    let restore_metrics = recovery_timer
        .elapsed("checkpoint", "restore_duration")
        .and_then(|elapsed| {
            core.metrics
                .record_checkpoint_restore(elapsed, sink_retries)
        })
        .and_then(|()| {
            core.metrics
                .record_checkpoint_orphan_cleanup(retention.removed_orphan_segments)
        });
    if let Err(error) = restore_metrics {
        return checkpoint_start_failure(launch_id, error);
    }
    core.state.lock().launch_delivery = LaunchDeliveryState::ReadyUnclaimed;
    core.changed.notify_waiters();
    if !await_handle_claim(core).await {
        return cancelled_driver_report(launch_id, &core.metrics);
    }
    let cause = TerminalCause::NaturalEnd;
    core.metrics
        .record_terminal(ContinuousJobState::Completed, cause.clone());
    DriverReport {
        launch_id,
        completion: DriverCompletion::Outcome(Arc::new(ContinuousJobOutcome {
            state: ContinuousJobState::Completed,
            cause,
            errors: Vec::new(),
        })),
        cleanup_failures: Vec::new(),
    }
}

fn validate_manifest_operator_capabilities(
    manifest: &CheckpointManifest,
    plan: &StreamRuntimePlanParts,
) -> crate::Result<()> {
    for node in &plan.nodes {
        let operator_id = node.operator_id.as_str();
        let entry = manifest.operators().get(operator_id).ok_or_else(|| {
            CalcFlowError::CheckpointMismatch {
                message: format!("checkpoint is missing operator {operator_id:?}"),
            }
        })?;
        let segments = if entry.segments.is_empty() {
            BTreeMap::new()
        } else {
            BTreeMap::from([("state".into(), Vec::new())])
        };
        node.checkpoint_capability.decode_snapshot(
            operator_id,
            crate::OperatorStateSnapshot {
                inline_metadata: entry.inline_metadata.clone(),
                segments,
            },
        )?;
    }
    Ok(())
}

fn checkpoint_start_failure(launch_id: LaunchId, error: CalcFlowError) -> DriverReport {
    DriverReport {
        launch_id,
        completion: DriverCompletion::StartFailed(StartFailure {
            primary: Arc::new(RuntimeFailure {
                origin: FailureOrigin::Preflight,
                error,
            }),
            diagnostic_id: None,
            cleanup_failures: Vec::new(),
        }),
        cleanup_failures: Vec::new(),
    }
}

async fn prepare_checkpoint_recovery(
    checkpoint: &OpenedCheckpointRuntime,
    plan: &StreamRuntimePlanParts,
    prepared_progress: &super::progress::PreparedStreamJob,
    sources: &mut BTreeMap<String, SourceBinding>,
    cancellation: &CancellationToken,
) -> crate::Result<(
    BTreeMap<String, OperatorRestoreState>,
    Option<DurableProgressRestore>,
)> {
    let Some(selected) = &checkpoint.selected else {
        return Ok((BTreeMap::new(), None));
    };
    let durable = restore_durable_progress(
        prepared_progress,
        selected.manifest.sources(),
        LogicalInstant::ZERO,
    )?;
    for (source_id, restored) in &durable.sources {
        sources
            .get_mut(source_id)
            .expect("selected manifest source IDs were validated")
            .restore(source_id, restored)?;
    }
    for source_id in durable
        .sources
        .iter()
        .filter_map(|(source_id, restored)| restored.ended.then_some(source_id))
    {
        sources
            .remove(source_id)
            .expect("restored ended source was validated before connector ownership");
    }
    let checkpoint_capabilities = plan
        .nodes
        .iter()
        .map(|node| (node.operator_id.as_str(), node.checkpoint_capability))
        .collect::<BTreeMap<_, _>>();
    let mut operators = BTreeMap::new();
    for (operator_id, entry) in selected.manifest.operators() {
        let snapshot = checkpoint
            .transaction
            .load_operator_state_cancellable(
                operator_id,
                selected.manifest.epoch(),
                entry,
                cancellation,
            )
            .await?;
        let snapshot = checkpoint_capabilities
            .get(operator_id.as_str())
            .ok_or_else(|| CalcFlowError::CheckpointMismatch {
                message: format!(
                    "checkpoint operator {operator_id:?} is absent from the prepared plan"
                ),
            })?
            .decode_snapshot(operator_id, snapshot)?;
        operators.insert(
            operator_id.clone(),
            OperatorRestoreState {
                snapshot,
                progress: entry.progress.clone(),
                next_epoch: checkpoint.next_epoch,
            },
        );
    }
    Ok((operators, Some(durable)))
}

fn restored_ended_source_cuts(
    checkpoint: &OpenedCheckpointRuntime,
) -> crate::Result<BTreeMap<super::progress::BindingIdentity, DurableSourceCut>> {
    checkpoint
        .selected
        .as_ref()
        .into_iter()
        .flat_map(|selected| selected.manifest.sources())
        .filter(|(_, entry)| entry.ended)
        .map(|(source_id, entry)| {
            Ok((
                super::progress::BindingIdentity::new(source_id.as_str())?,
                DurableSourceCut {
                    cursor: entry.cursor.clone(),
                    next_sequence: entry.sequence,
                    ended: true,
                },
            ))
        })
        .collect()
}

fn add_restored_ended_source_cuts(
    checkpoint: &OpenedCheckpointRuntime,
    cuts: &mut BTreeMap<super::progress::BindingIdentity, DurableSourceCut>,
) -> crate::Result<()> {
    for (binding, cut) in restored_ended_source_cuts(checkpoint)? {
        if cuts.insert(binding.clone(), cut).is_some() {
            return Err(CalcFlowError::CheckpointMismatch {
                message: format!(
                    "restored ended source {:?} also produced a live checkpoint cut",
                    binding.as_str()
                ),
            });
        }
    }
    Ok(())
}

async fn recover_opened_sinks(
    sinks: &mut BTreeMap<String, Vec<ValidatedOrdinarySink>>,
    manifest: &CheckpointManifest,
) -> crate::Result<()> {
    for output_sinks in sinks.values_mut() {
        super::sink_task::recover_transactional_sinks(output_sinks, manifest).await?;
    }
    Ok(())
}

async fn open_checkpoint_runtime(
    checkpoint: ValidatedCheckpointRuntime,
    cancellation: &CancellationToken,
) -> crate::Result<OpenedCheckpointRuntime> {
    let ValidatedCheckpointRuntime { spec, identity } = checkpoint;
    let (state_backend, manifest_root, managed_storage, managed) = match spec.storage {
        CheckpointRuntimeStorage::LegacyParts {
            state_backend,
            manifest_root,
        } => (state_backend, manifest_root, None, false),
        CheckpointRuntimeStorage::Managed(storage) => {
            let opened = storage.open(cancellation).await?;
            let state_backend: Arc<dyn StateBackend> = opened.state_backend();
            let manifest_root = opened.manifest_root().to_owned();
            (state_backend, manifest_root, Some(opened), true)
        }
        #[cfg(test)]
        CheckpointRuntimeStorage::ManagedTestParts {
            state_backend,
            manifest_root,
        } => (state_backend, manifest_root, None, true),
    };
    let key = StateLineageKey::new(&identity.pipeline_name, &identity.pipeline_fingerprint)?;
    let lineage = settle_checkpoint_operation(
        cancellation,
        "lineage-open",
        state_backend.open_lineage(&key),
    )
    .await
    .map_err(|error| sanitize_managed_preflight_error(error, managed, false))?;
    let transaction = ManifestTransaction::open_cancellable(
        Arc::from(lineage),
        &key,
        &manifest_root,
        spec.config.retained_epochs,
        cancellation,
    )
    .await
    .map_err(|error| sanitize_managed_preflight_error(error, managed, false))?;
    #[cfg(all(test, unix))]
    let transaction = configure_test_manifest_transaction(transaction, &spec.faults);
    let selected = transaction
        .select_latest_cancellable(&identity, cancellation)
        .await
        .map_err(|error| sanitize_managed_preflight_error(error, managed, true))?;
    let startup_orphans_removed = transaction
        .retain_cancellable(
            &identity,
            selected.as_ref().map(|selected| &selected.manifest),
            cancellation,
        )
        .await
        .map_err(|error| sanitize_managed_preflight_error(error, managed, true))?
        .removed_orphan_segments;
    #[cfg(test)]
    let transaction = {
        let faults = spec.faults.clone();
        let fault_cancellation = cancellation.clone();
        transaction.with_fault_hook(Arc::new(move |point| {
            let point = match point {
                ManifestTransactionFaultPoint::StateStage => CheckpointFaultPoint::StateStage,
                ManifestTransactionFaultPoint::ManifestWrite => CheckpointFaultPoint::ManifestWrite,
                ManifestTransactionFaultPoint::ManifestRename => {
                    CheckpointFaultPoint::ManifestRename
                }
                ManifestTransactionFaultPoint::ManifestParentSync => {
                    CheckpointFaultPoint::ManifestParentSync
                }
                ManifestTransactionFaultPoint::Compaction => CheckpointFaultPoint::Compaction,
            };
            faults.trigger(point, &fault_cancellation)?;
            if fault_cancellation.is_cancelled() {
                return Err(CalcFlowError::Cancelled {
                    run_id: format!("checkpoint:{point:?}"),
                });
            }
            Ok(())
        }))
    };
    let transaction = Arc::new(transaction);
    let next_epoch = selected
        .as_ref()
        .map_or(Epoch::INITIAL, |selected| selected.next_epoch);
    let status = CheckpointStatusHandle::new(&identity, selected.as_ref());
    Ok(OpenedCheckpointRuntime {
        transaction,
        _managed_storage: managed_storage,
        identity,
        config: spec.config,
        selected,
        next_epoch,
        status,
        startup_orphans_removed,
        managed,
        #[cfg(test)]
        faults: spec.faults,
        #[cfg(test)]
        started_gate: spec.started_gate,
    })
}

#[cfg(all(test, unix))]
fn configure_test_manifest_transaction(
    transaction: ManifestTransaction,
    faults: &CheckpointFaultInjector,
) -> ManifestTransaction {
    if faults.is_armed(
        CheckpointFaultPoint::ManifestParentSync,
        CheckpointFaultMode::Io,
    ) {
        transaction.with_real_parent_sync_failure_for_test()
    } else {
        transaction
    }
}

fn sanitize_managed_preflight_error(
    error: CalcFlowError,
    managed: bool,
    manifest_candidate: bool,
) -> CalcFlowError {
    if !managed {
        return error;
    }
    match error {
        CalcFlowError::Conflict { .. } | CalcFlowError::PlanLeased { .. } => {
            return CalcFlowError::Conflict {
                resource: "managed checkpoint directory".into(),
                key: "active".into(),
            };
        }
        CalcFlowError::Cancelled { .. } => {
            return CalcFlowError::Cancelled {
                run_id: "managed-checkpoint-open".into(),
            };
        }
        _ => {}
    }
    if manifest_candidate {
        return CalcFlowError::CheckpointMismatch {
            message: "checkpoint lineage contains an invalid manifest candidate".into(),
        };
    }
    CalcFlowError::Internal {
        message: "managed checkpoint storage initialization failed".into(),
    }
}

fn sanitize_managed_recovery_error(error: CalcFlowError, managed: bool) -> CalcFlowError {
    if !managed {
        return error;
    }
    safe_managed_recovery_error(&error)
}

fn safe_managed_recovery_error(error: &CalcFlowError) -> CalcFlowError {
    match error {
        CalcFlowError::Cancelled { .. } => CalcFlowError::Cancelled {
            run_id: "managed-checkpoint-recovery".into(),
        },
        _ => CalcFlowError::Internal {
            message: "managed checkpoint recovery failed".into(),
        },
    }
}

fn sanitize_managed_recovery_failure(
    failure: Arc<RuntimeFailure>,
    managed: bool,
) -> Arc<RuntimeFailure> {
    if !managed {
        return failure;
    }
    let error = safe_managed_recovery_error(&failure.error);
    Arc::new(RuntimeFailure {
        origin: FailureOrigin::Preflight,
        error,
    })
}

async fn settle_checkpoint_operation<T>(
    cancellation: &CancellationToken,
    operation: &str,
    future: impl Future<Output = crate::Result<T>>,
) -> crate::Result<T> {
    if cancellation.is_cancelled() {
        return Err(checkpoint_cancellation_error(operation));
    }
    tokio::pin!(future);
    tokio::select! {
        biased;
        () = cancellation.cancelled() => {
            let _ = future.await;
            Err(checkpoint_cancellation_error(operation))
        }
        result = &mut future => result,
    }
}

fn checkpoint_cancellation_error(operation: &str) -> CalcFlowError {
    CalcFlowError::Cancelled {
        run_id: format!("checkpoint:{operation}"),
    }
}

async fn cancel_after_operator_entry(
    launch_id: LaunchId,
    core: &JobCore,
    runtime: &mut RegisteredRuntime,
) -> Option<DriverReport> {
    #[cfg(test)]
    if let Some(probe) = &core.launch_probe {
        probe
            .pause_at(TestLaunchCheckpoint::AfterOperatorEntry)
            .await;
    }
    if !core.launch_cancel.is_cancelled() {
        return None;
    }
    runtime.supervisor.cancel();
    let report = runtime.supervisor.join_all().await;
    Some(cancelled_driver_report_with_task_cleanup(
        launch_id,
        report,
        &RuntimeTaskProgress {
            sources: BTreeMap::new(),
            sinks: BTreeMap::new(),
        },
        &core.metrics,
    ))
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

#[derive(Clone)]
struct OperatorCheckpointRegistration {
    acknowledgements: mpsc::Sender<OperatorCheckpointAck>,
    transaction: Arc<ManifestTransaction>,
    terminal_ready: mpsc::Sender<String>,
    terminal_commands: Arc<Mutex<BTreeMap<String, mpsc::Receiver<OperatorCheckpointCommand>>>>,
    #[cfg(test)]
    faults: CheckpointFaultInjector,
    #[cfg(test)]
    fault_cancellation: CancellationToken,
}

impl OperatorCheckpointRegistration {
    fn port(&self, node_id: &str) -> OperatorCheckpointPort {
        OperatorCheckpointPort {
            acknowledgements: self.acknowledgements.clone(),
            transaction: Some(Arc::clone(&self.transaction)),
            terminal: Some(OperatorTerminalPort {
                ready: self.terminal_ready.clone(),
                commands: self
                    .terminal_commands
                    .lock()
                    .remove(node_id)
                    .expect("checkpoint wiring covers every validated operator"),
            }),
            #[cfg(test)]
            alignment_fault: Some({
                let faults = self.faults.clone();
                let cancellation = self.fault_cancellation.clone();
                Arc::new(move || {
                    faults.trigger(CheckpointFaultPoint::PartialAlignment, &cancellation)
                })
            }),
        }
    }
}

struct LiveCheckpointChannels {
    operator: OperatorCheckpointRegistration,
    operator_acknowledgements: mpsc::Receiver<OperatorCheckpointAck>,
    operator_terminal_ready: mpsc::Receiver<String>,
    operator_commands: BTreeMap<String, mpsc::Sender<OperatorCheckpointCommand>>,
    sink_acknowledgement_sender: mpsc::Sender<SinkCheckpointAck>,
    sink_acknowledgements: mpsc::Receiver<SinkCheckpointAck>,
    sink_finalization_sender: Option<mpsc::Sender<SinkFinalizeAck>>,
    sink_finalizations: mpsc::Receiver<SinkFinalizeAck>,
    sink_terminal_ready_sender: mpsc::Sender<String>,
    sink_terminal_ready: mpsc::Receiver<String>,
    sink_commands: BTreeMap<String, mpsc::Sender<SinkCheckpointCommand>>,
    sink_command_receivers: BTreeMap<String, mpsc::Receiver<SinkCheckpointCommand>>,
}

impl LiveCheckpointChannels {
    fn new(
        plan: &StreamRuntimePlanParts,
        sources: &BTreeMap<String, SourceBinding>,
        sinks: &BTreeMap<String, Vec<ValidatedOrdinarySink>>,
        transaction: Arc<ManifestTransaction>,
        #[cfg(test)] faults: CheckpointFaultInjector,
        #[cfg(test)] fault_cancellation: CancellationToken,
    ) -> Self {
        let participant_count = plan
            .nodes
            .len()
            .saturating_add(sources.len())
            .saturating_add(sinks.len());
        let capacity = participant_count.max(4);
        let (operator_tx, operator_rx) = mpsc::channel(capacity);
        let (operator_terminal_tx, operator_terminal_rx) = mpsc::channel(capacity);
        let (sink_tx, sink_rx) = mpsc::channel(capacity);
        let (finalization_tx, finalization_rx) = mpsc::channel(capacity);
        let (sink_terminal_tx, sink_terminal_rx) = mpsc::channel(capacity);
        let mut operator_commands = BTreeMap::new();
        let mut operator_command_receivers = BTreeMap::new();
        for node in &plan.nodes {
            let (sender, receiver) = mpsc::channel(capacity);
            let operator_id = node.operator_id.as_str().to_owned();
            operator_commands.insert(operator_id.clone(), sender);
            operator_command_receivers.insert(operator_id, receiver);
        }
        let mut sink_commands = BTreeMap::new();
        let mut sink_command_receivers = BTreeMap::new();
        for output_id in sinks.keys() {
            let (sender, receiver) = mpsc::channel(capacity);
            sink_commands.insert(output_id.clone(), sender);
            sink_command_receivers.insert(output_id.clone(), receiver);
        }
        Self {
            operator: OperatorCheckpointRegistration {
                acknowledgements: operator_tx,
                transaction,
                terminal_ready: operator_terminal_tx,
                terminal_commands: Arc::new(Mutex::new(operator_command_receivers)),
                #[cfg(test)]
                faults,
                #[cfg(test)]
                fault_cancellation,
            },
            operator_acknowledgements: operator_rx,
            operator_terminal_ready: operator_terminal_rx,
            operator_commands,
            sink_acknowledgement_sender: sink_tx,
            sink_acknowledgements: sink_rx,
            sink_finalization_sender: Some(finalization_tx),
            sink_finalizations: finalization_rx,
            sink_terminal_ready_sender: sink_terminal_tx,
            sink_terminal_ready: sink_terminal_rx,
            sink_commands,
            sink_command_receivers,
        }
    }

    fn take_sink_port(&mut self, output_id: &str, initial_epoch: Epoch) -> SinkCheckpointPort {
        let finalizations = if self.sink_command_receivers.len() == 1 {
            self.sink_finalization_sender
                .take()
                .expect("checkpoint wiring retains its finalization sender")
        } else {
            self.sink_finalization_sender
                .as_ref()
                .expect("checkpoint wiring retains its finalization sender")
                .clone()
        };
        SinkCheckpointPort {
            initial_epoch,
            acknowledgements: self.sink_acknowledgement_sender.clone(),
            commands: self
                .sink_command_receivers
                .remove(output_id)
                .expect("checkpoint wiring covers every validated sink output"),
            finalizations,
            terminal_ready: Some(self.sink_terminal_ready_sender.clone()),
        }
    }
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
    mut restores: BTreeMap<String, OperatorRestoreState>,
    checkpoint: Option<OperatorCheckpointRegistration>,
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
        restores: &mut restores,
        checkpoint: checkpoint.as_ref(),
    };
    if let Err(failure) = register_operator_nodes(plan.nodes, registration) {
        supervisor.cancel();
        let _ = supervisor.join_all().await;
        return Err(failure);
    }
    if !restores.is_empty() {
        return fail_registered_entry(
            &mut supervisor,
            preflight_entry_failure(CalcFlowError::CheckpointMismatch {
                message: "checkpoint restore contains an unknown operator".into(),
            }),
        )
        .await;
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
    restores: &'a mut BTreeMap<String, OperatorRestoreState>,
    checkpoint: Option<&'a OperatorCheckpointRegistration>,
}

fn register_operator_nodes(
    nodes: Vec<RuntimeStreamNode>,
    registration: &mut OperatorRegistration<'_>,
) -> Result<(), EntryFailure> {
    for node in nodes {
        let node_id = node.operator_id.as_str().to_owned();
        debug_assert_eq!(node.node_id, node_id);
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
                checkpoint_capability: node.checkpoint_capability,
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
                checkpoint: registration
                    .checkpoint
                    .map(|checkpoint| checkpoint.port(&node_id)),
                restore: registration.restores.remove(&node_id),
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
            binding: Box::new(binding),
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
                sources.insert(binding_id, *binding);
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

#[allow(
    clippy::too_many_arguments,
    clippy::too_many_lines,
    reason = "boundary registration wires the validated sources, sinks, progress, and checkpoint owner"
)]
fn register_boundary_tasks(
    runtime: &mut RegisteredRuntime,
    context: &super::StreamJobContext,
    sources: BTreeMap<String, SourceBinding>,
    sinks: BTreeMap<String, Vec<ValidatedOrdinarySink>>,
    prepared_progress: super::progress::PreparedStreamJob,
    durable_progress: Option<&DurableProgressRestore>,
    checkpoint: Option<OpenedCheckpointRuntime>,
    mut checkpoint_channels: Option<LiveCheckpointChannels>,
    core: &Arc<JobCore>,
) -> RuntimeTaskProgress {
    let prepared_progress = Arc::new(prepared_progress);
    let source_outputs = std::mem::take(&mut runtime.source_outputs);
    let live_progress = match durable_progress {
        Some(restored) => LiveProgressCoordinator::new_restored(
            &prepared_progress,
            source_outputs,
            context.cancellation().clone(),
            restored,
        ),
        None => LiveProgressCoordinator::new(
            &prepared_progress,
            source_outputs,
            context.cancellation().clone(),
        ),
    }
    .expect("preflight projected every prepared progress source route");
    let progress_status = live_progress.status_handle();
    let mut source_progress = BTreeMap::new();
    for (binding_id, binding) in sources {
        let progress = spawn_source_tasks_gated_with_live_progress(
            &mut runtime.supervisor,
            context,
            &binding_id,
            binding,
            runtime.data_gate.subscribe(),
            core.launch_cancel.clone(),
            core.metrics.clone(),
            live_progress.clone(),
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
        let sink_checkpoint = checkpoint.as_ref().map(|checkpoint| {
            checkpoint_channels
                .as_mut()
                .expect("checkpoint runtime retains its bounded task channels")
                .take_sink_port(&output_id, checkpoint.next_epoch)
        });
        #[cfg(test)]
        let sink_commit_fault = checkpoint.as_ref().map(|checkpoint| {
            let faults = checkpoint.faults.clone();
            let cancellation = context.cancellation().clone();
            Arc::new(move || {
                let trigger_count = faults.trigger_count();
                faults.trigger(CheckpointFaultPoint::PartialSinkCommit, &cancellation)?;
                if faults.trigger_count() != trigger_count && cancellation.is_cancelled() {
                    Err(checkpoint_cancellation_error("partial-sink-commit"))
                } else {
                    Ok(())
                }
            }) as super::sink_task::SinkCommitFaultHook
        });
        spawn_sink_task(
            &mut runtime.supervisor,
            SinkTaskInputs {
                output_id: output_id.clone(),
                pipeline_name: checkpoint
                    .as_ref()
                    .map(|checkpoint| checkpoint.identity.pipeline_name.clone()),
                sinks: bindings,
                input,
                context: context
                    .for_sink(&output_id)
                    .expect("preflight validated every sink task scope"),
                progress: progress.clone(),
                metrics: core.metrics.clone(),
                data_gate: runtime.data_gate.subscribe(),
                launch_cancel: core.launch_cancel.clone(),
                checkpoint: sink_checkpoint,
                epoch_owner: SinkEpochOwner::default(),
                #[cfg(test)]
                sink_commit_fault,
            },
        );
        sink_progress.insert(output_id, progress);
    }
    match (checkpoint, checkpoint_channels) {
        (Some(checkpoint), Some(channels)) => {
            let task_inputs = LiveCheckpointTaskInputs {
                checkpoint,
                channels,
                live_progress: live_progress.clone(),
                sources: source_progress.clone(),
                sinks: sink_progress.clone(),
                cancellation: context.cancellation().clone(),
                metrics: core.metrics.clone(),
                core: Arc::clone(core),
            };
            runtime
                .supervisor
                .spawn("checkpoint", run_live_checkpoint_task(task_inputs));
        }
        (None, None) => {}
        _ => unreachable!("checkpoint runtime and channels are created together"),
    }
    spawn_live_progress_task(
        &mut runtime.supervisor,
        live_progress,
        context.cancellation().clone(),
    );
    debug_assert!(runtime.source_outputs.is_empty());
    debug_assert!(runtime.sink_inputs.is_empty());
    {
        let mut status = core.runtime_status.lock();
        status.sources = source_progress.clone();
        status.sinks = sink_progress.clone();
        status.progress = Some(progress_status);
    }
    RuntimeTaskProgress {
        sources: source_progress,
        sinks: sink_progress,
    }
}

struct LiveCheckpointTaskInputs {
    checkpoint: OpenedCheckpointRuntime,
    channels: LiveCheckpointChannels,
    live_progress: LiveProgressCoordinator,
    sources: BTreeMap<String, SourceProgress>,
    sinks: BTreeMap<String, SinkProgress>,
    cancellation: CancellationToken,
    metrics: MetricsRecorder,
    core: Arc<JobCore>,
}

struct ManualCheckpointRegistration {
    core: Arc<JobCore>,
    coordinator: CheckpointCoordinatorHandle,
}

impl ManualCheckpointRegistration {
    fn install(core: Arc<JobCore>, coordinator: CheckpointCoordinatorHandle) -> Self {
        let previous = core.manual_checkpoint.lock().replace(coordinator.clone());
        debug_assert!(previous.is_none());
        core.changed.notify_waiters();
        Self { core, coordinator }
    }
}

impl Drop for ManualCheckpointRegistration {
    fn drop(&mut self) {
        self.core.manual_checkpoint.lock().take();
        self.core.changed.notify_waiters();
        self.coordinator.terminate(ManualCheckpointFailure::Failed {
            category: ManualCheckpointFailureCategory::Internal,
            epoch: None,
            phase: None,
        });
    }
}

#[derive(Default)]
struct EpochManifestAssembly {
    epoch: Option<Epoch>,
    terminal: bool,
    manifest_durable: bool,
    manifest_installed_unknown: bool,
    deferred_publication_error: Option<CalcFlowError>,
    sources: BTreeMap<String, SourceManifestEntry>,
    operators: BTreeMap<String, OperatorManifestEntry>,
    sink_outputs: BTreeMap<String, BTreeMap<String, SinkManifestEntry>>,
    finalized_sink_outputs: BTreeSet<String>,
    timed_phase: Option<CheckpointPhase>,
    phase_timer: Option<MetricsTimer>,
    checkpoint_timer: Option<MetricsTimer>,
}

impl EpochManifestAssembly {
    fn start(&mut self, epoch: Epoch, terminal: bool) -> crate::Result<()> {
        if self.epoch.replace(epoch).is_some() {
            return Err(checkpoint_protocol_error(
                epoch,
                "started while another manifest assembly was active",
            ));
        }
        self.sources.clear();
        self.operators.clear();
        self.sink_outputs.clear();
        self.finalized_sink_outputs.clear();
        self.terminal = terminal;
        self.manifest_durable = false;
        self.manifest_installed_unknown = false;
        self.deferred_publication_error = None;
        Ok(())
    }

    fn expect_epoch(&self, epoch: Epoch) -> crate::Result<()> {
        if self.epoch == Some(epoch) {
            Ok(())
        } else {
            Err(checkpoint_protocol_error(
                epoch,
                "acknowledgement does not match the active manifest assembly",
            ))
        }
    }

    fn promote_terminal(&mut self, epoch: Epoch) -> crate::Result<()> {
        self.expect_epoch(epoch)?;
        self.terminal = true;
        Ok(())
    }

    fn complete(&mut self, epoch: Epoch) -> crate::Result<()> {
        self.expect_epoch(epoch)?;
        self.epoch = None;
        self.manifest_durable = false;
        self.manifest_installed_unknown = false;
        Ok(())
    }

    fn start_metrics(&mut self, metrics: &MetricsRecorder, terminal: bool) -> crate::Result<()> {
        metrics.record_checkpoint_requested(terminal)?;
        self.timed_phase = Some(CheckpointPhase::Requested);
        self.phase_timer = Some(metrics.timer());
        self.checkpoint_timer = Some(metrics.timer());
        Ok(())
    }

    fn advance_metrics(
        &mut self,
        metrics: &MetricsRecorder,
        next: CheckpointPhase,
    ) -> crate::Result<()> {
        if let (Some(phase), Some(timer)) = (self.timed_phase, self.phase_timer.as_ref()) {
            metrics
                .record_checkpoint_phase(phase, timer.elapsed("checkpoint", "phase_duration")?)?;
        }
        self.timed_phase = Some(next);
        self.phase_timer = Some(metrics.timer());
        Ok(())
    }

    fn complete_metrics(&mut self, metrics: &MetricsRecorder, terminal: bool) -> crate::Result<()> {
        self.advance_metrics(metrics, CheckpointPhase::SinksCommitted)?;
        let elapsed = self
            .checkpoint_timer
            .as_ref()
            .ok_or_else(|| CalcFlowError::Internal {
                message: "checkpoint completion omitted its metrics timer".into(),
            })?
            .elapsed("checkpoint", "total_duration")?;
        metrics.record_checkpoint_completed(terminal, elapsed)?;
        self.timed_phase = None;
        self.phase_timer = None;
        self.checkpoint_timer = None;
        Ok(())
    }

    fn fail_metrics(&mut self, metrics: &MetricsRecorder, terminal: bool) -> crate::Result<()> {
        if let (Some(phase), Some(timer)) = (self.timed_phase, self.phase_timer.as_ref()) {
            metrics
                .record_checkpoint_phase(phase, timer.elapsed("checkpoint", "phase_duration")?)?;
        }
        metrics.record_checkpoint_failed(terminal)?;
        self.timed_phase = None;
        self.phase_timer = None;
        self.checkpoint_timer = None;
        Ok(())
    }

    fn cancel_metrics(&mut self) {
        self.timed_phase = None;
        self.phase_timer = None;
        self.checkpoint_timer = None;
    }
}

#[allow(
    clippy::too_many_lines,
    reason = "the checkpoint task is the single owner of epoch events, acknowledgements, and manifest publication"
)]
async fn run_live_checkpoint_task(inputs: LiveCheckpointTaskInputs) -> crate::Result<()> {
    let LiveCheckpointTaskInputs {
        checkpoint,
        mut channels,
        live_progress,
        sources,
        sinks,
        cancellation,
        metrics,
        core,
    } = inputs;
    let participants = ParticipantSet {
        sources: checkpoint.identity.source_ids.clone(),
        operators: checkpoint.identity.operator_ids.clone(),
        sinks: channels.sink_commands.keys().cloned().collect(),
    };
    let expected_operators = participants.operators.clone();
    let expected_sinks = participants.sinks.clone();
    checkpoint.status.set_expected(
        participants.sources.len(),
        participants.operators.len(),
        participants.sinks.len(),
    );
    let capacity = participants
        .sources
        .len()
        .saturating_add(participants.operators.len())
        .saturating_add(participants.sinks.len())
        .max(4);
    let coordinator_cancellation = CancellationToken::new();
    let (coordinator, mut events, coordinator_task) = spawn_checkpoint_coordinator(
        participants,
        checkpoint.next_epoch,
        capacity,
        checkpoint.config.checkpoint_timeout,
        coordinator_cancellation.clone(),
    )?;
    let _manual_checkpoint =
        ManualCheckpointRegistration::install(Arc::clone(&core), coordinator.clone());
    let mut interval = tokio::time::interval(checkpoint.config.checkpoint_interval);
    interval.set_missed_tick_behavior(tokio::time::MissedTickBehavior::Delay);
    interval.tick().await;
    let mut request_active = false;
    let mut terminal_request_active = false;
    let mut assembly = EpochManifestAssembly::default();
    let mut terminal_source_cuts = None;
    let mut terminal_operators = BTreeSet::new();
    let mut terminal_sinks = BTreeSet::new();
    let terminal_sources = {
        let terminal_sources = sources.clone();
        let terminal_cancellation = cancellation.clone();
        async move { wait_for_terminal_source_cuts(&terminal_sources, &terminal_cancellation).await }
    };
    tokio::pin!(terminal_sources);
    let mut terminal_sources_observed = false;
    let result = loop {
        tokio::select! {
            biased;
            () = cancellation.cancelled(), if !assembly.manifest_durable => break Ok(()),
            event = events.recv() => {
                let Some(event) = event else {
                    break Err(CalcFlowError::Internal {
                        message: "checkpoint coordinator event channel closed".into(),
                    });
                };
                match handle_checkpoint_event(
                    event,
                    &coordinator,
                    &checkpoint,
                    &live_progress,
                    &sources,
                    &channels.sink_commands,
                    &channels.operator_commands,
                    &cancellation,
                    &metrics,
                    &mut assembly,
                    &mut request_active,
                    &mut terminal_request_active,
                    &mut terminal_source_cuts,
                ).await {
                    Ok(true) => break Ok(()),
                    Ok(false) => {
                        if cancellation.is_cancelled() {
                            continue;
                        }
                        if let Err(error) = maybe_request_terminal_checkpoint(
                            &coordinator,
                            &expected_operators,
                            &expected_sinks,
                            &terminal_operators,
                            &terminal_sinks,
                            terminal_source_cuts.is_some(),
                            &mut request_active,
                            &mut terminal_request_active,
                        ).await {
                            break Err(error);
                        }
                    }
                    Err(error) => break Err(error),
                }
            }
            terminal = &mut terminal_sources,
                if !terminal_sources_observed && !assembly.manifest_durable => {
                match terminal {
                    Ok(mut cuts) => {
                        if let Err(error) = add_restored_ended_source_cuts(&checkpoint, &mut cuts) {
                            break Err(error);
                        }
                        terminal_source_cuts = Some(cuts);
                        terminal_sources_observed = true;
                    }
                    Err(error) => break Err(error),
                }
                if let Err(error) = maybe_request_terminal_checkpoint(
                    &coordinator,
                    &expected_operators,
                    &expected_sinks,
                    &terminal_operators,
                    &terminal_sinks,
                    terminal_source_cuts.is_some(),
                    &mut request_active,
                    &mut terminal_request_active,
                ).await {
                    break Err(error);
                }
            }
            ready = channels.operator_terminal_ready.recv(), if !assembly.manifest_durable => {
                let Some(ready) = ready else {
                    break Err(checkpoint_channel_closed("operator terminal readiness"));
                };
                if !expected_operators.contains(&ready) {
                    break Err(CalcFlowError::CheckpointMismatch {
                        message: format!("foreign terminal operator {ready:?}"),
                    });
                }
                terminal_operators.insert(ready);
                if let Err(error) = maybe_request_terminal_checkpoint(
                    &coordinator,
                    &expected_operators,
                    &expected_sinks,
                    &terminal_operators,
                    &terminal_sinks,
                    terminal_source_cuts.is_some(),
                    &mut request_active,
                    &mut terminal_request_active,
                ).await {
                    break Err(error);
                }
            }
            ready = channels.sink_terminal_ready.recv(), if !assembly.manifest_durable => {
                let Some(ready) = ready else {
                    break Err(checkpoint_channel_closed("sink terminal readiness"));
                };
                if !expected_sinks.contains(&ready) {
                    break Err(CalcFlowError::CheckpointMismatch {
                        message: format!("foreign terminal sink output {ready:?}"),
                    });
                }
                terminal_sinks.insert(ready);
                if let Err(error) = maybe_request_terminal_checkpoint(
                    &coordinator,
                    &expected_operators,
                    &expected_sinks,
                    &terminal_operators,
                    &terminal_sinks,
                    terminal_source_cuts.is_some(),
                    &mut request_active,
                    &mut terminal_request_active,
                ).await {
                    break Err(error);
                }
            }
            acknowledgement = channels.operator_acknowledgements.recv(),
                if !assembly.manifest_durable => {
                let Some(acknowledgement) = acknowledgement else {
                    break Err(checkpoint_channel_closed("operator acknowledgements"));
                };
                if let Err(error) = accept_operator_ack(
                    acknowledgement,
                    &coordinator,
                    &mut assembly,
                    &checkpoint.status,
                ).await {
                    break Err(error);
                }
            }
            acknowledgement = channels.sink_acknowledgements.recv(),
                if !assembly.manifest_durable => {
                let Some(acknowledgement) = acknowledgement else {
                    break Err(checkpoint_channel_closed("sink acknowledgements"));
                };
                if let Err(error) = accept_sink_ack(
                    acknowledgement,
                    &coordinator,
                    &mut assembly,
                    &checkpoint.status,
                ).await {
                    break Err(error);
                }
                #[cfg(test)]
                if let Err(error) =
                    checkpoint.inject_fault(CheckpointFaultPoint::SinkPreCommit, &cancellation)
                {
                    break Err(error);
                }
            }
            finalization = channels.sink_finalizations.recv(),
                if assembly.finalized_sink_outputs != expected_sinks => {
                let Some(finalization) = finalization else {
                    if cancellation.is_cancelled() {
                        break Ok(());
                    }
                    break Err(checkpoint_channel_closed("sink finalizations"));
                };
                if let Err(error) = accept_sink_finalization(
                    finalization,
                    &coordinator,
                    &mut assembly,
                    &checkpoint.status,
                ).await {
                    break Err(error);
                }
            }
            _ = interval.tick(), if !request_active && !terminal_sources_observed => {
                if let Err(error) = coordinator.request(CheckpointRequest::Periodic).await {
                    break Err(error);
                }
                request_active = true;
            }
        }
    };
    let publication_unknown = assembly.manifest_installed_unknown;
    let sink_commit_incomplete =
        assembly.manifest_durable && assembly.finalized_sink_outputs != expected_sinks;
    let sink_commit_failure = sink_commit_incomplete
        .then(|| {
            sinks
                .values()
                .find_map(SinkProgress::checkpoint_failure_sink_id)
        })
        .flatten();
    let result = match result {
        _ if publication_unknown => Err(CalcFlowError::RecoveryRequired {
            pipeline_name: checkpoint.identity.pipeline_name.clone(),
            message: format!(
                "checkpoint epoch {} was installed but publication durability is unknown",
                assembly
                    .epoch
                    .expect("indeterminate publication retains its active epoch")
                    .as_u64()
            ),
        }),
        Err(error) if sink_commit_incomplete => Err(CalcFlowError::RecoveryRequired {
            pipeline_name: checkpoint.identity.pipeline_name.clone(),
            message: format!(
                "checkpoint manifest is durable but sink commit did not complete: {error}"
            ),
        }),
        Ok(()) if sink_commit_incomplete => Err(CalcFlowError::RecoveryRequired {
            pipeline_name: checkpoint.identity.pipeline_name.clone(),
            message: "checkpoint manifest is durable but sink commit completion was not observed"
                .into(),
        }),
        result => result,
    };
    let manual_failure = if let (Some(sink_id), Some(epoch)) = (sink_commit_failure, assembly.epoch)
    {
        ManualCheckpointFailure::SinkCommit { sink_id, epoch }
    } else {
        match &result {
            _ if core.operation_cancel_requested.load(Ordering::Acquire)
                && !assembly.manifest_durable
                && !publication_unknown =>
            {
                ManualCheckpointFailure::Cancelled
            }
            Err(CalcFlowError::RecoveryRequired {
                pipeline_name,
                message,
            }) => ManualCheckpointFailure::RecoveryRequired {
                pipeline_name: pipeline_name.clone(),
                message: message.clone(),
            },
            Err(error) => ManualCheckpointFailure::Failed {
                category: if matches!(error, CalcFlowError::Io { .. }) {
                    ManualCheckpointFailureCategory::Io
                } else {
                    ManualCheckpointFailureCategory::Internal
                },
                epoch: assembly.epoch,
                phase: checkpoint.status.snapshot().phase,
            },
            Ok(()) if cancellation.is_cancelled() => ManualCheckpointFailure::Cancelled,
            Ok(()) => ManualCheckpointFailure::Failed {
                category: ManualCheckpointFailureCategory::Internal,
                epoch: assembly.epoch,
                phase: checkpoint.status.snapshot().phase,
            },
        }
    };
    match &manual_failure {
        ManualCheckpointFailure::Failed { category, .. } => {
            let category = match category {
                ManualCheckpointFailureCategory::Timeout => CheckpointFailureCategory::Timeout,
                ManualCheckpointFailureCategory::Protocol => CheckpointFailureCategory::Protocol,
                ManualCheckpointFailureCategory::Io => CheckpointFailureCategory::Io,
                ManualCheckpointFailureCategory::Internal => CheckpointFailureCategory::Runtime,
            };
            checkpoint.status.fail_if_unset(category);
        }
        ManualCheckpointFailure::Cancelled
            if core.operation_cancel_requested.load(Ordering::Acquire) =>
        {
            checkpoint.status.cancel();
        }
        ManualCheckpointFailure::Cancelled => {
            checkpoint
                .status
                .fail_if_unset(CheckpointFailureCategory::Runtime);
        }
        ManualCheckpointFailure::RecoveryRequired { .. }
        | ManualCheckpointFailure::SinkCommit { .. } => {}
    }
    coordinator.terminate(manual_failure);
    drop(coordinator);
    let result = match (result, coordinator_task.await) {
        (Err(error), _) => Err(error),
        (Ok(()), Ok(result)) => result,
        (Ok(()), Err(error)) => Err(CalcFlowError::Internal {
            message: format!("checkpoint coordinator task join failed: {error}"),
        }),
    };
    if result.is_err() {
        assembly.fail_metrics(&metrics, assembly.terminal)?;
    } else if cancellation.is_cancelled() {
        assembly.cancel_metrics();
    }
    result
}

#[allow(
    clippy::too_many_arguments,
    clippy::too_many_lines,
    reason = "one event transition consumes the coordinator-owned checkpoint dependencies"
)]
async fn handle_checkpoint_event(
    event: CheckpointEvent,
    coordinator: &CheckpointCoordinatorHandle,
    checkpoint: &OpenedCheckpointRuntime,
    live_progress: &LiveProgressCoordinator,
    sources: &BTreeMap<String, SourceProgress>,
    sink_commands: &BTreeMap<String, mpsc::Sender<SinkCheckpointCommand>>,
    operator_commands: &BTreeMap<String, mpsc::Sender<OperatorCheckpointCommand>>,
    cancellation: &CancellationToken,
    metrics: &MetricsRecorder,
    assembly: &mut EpochManifestAssembly,
    request_active: &mut bool,
    terminal_request_active: &mut bool,
    terminal_source_cuts: &mut Option<BTreeMap<super::progress::BindingIdentity, DurableSourceCut>>,
) -> crate::Result<bool> {
    match event {
        CheckpointEvent::Started(epoch) => {
            checkpoint.status.start(epoch, *terminal_request_active);
            assembly.start(epoch, *terminal_request_active)?;
            assembly.start_metrics(metrics, *terminal_request_active)?;
            #[cfg(test)]
            if checkpoint.inject_fault(CheckpointFaultPoint::SourceAdmission, cancellation)? {
                return Ok(false);
            }
            #[cfg(test)]
            checkpoint.pause_after_started().await;
            let durable = if *terminal_request_active {
                let cuts = terminal_source_cuts.take().ok_or_else(|| {
                    checkpoint_protocol_error(epoch, "terminal source cuts are missing")
                })?;
                let durable = live_progress
                    .terminal_checkpoint_cut(epoch, &cuts, cancellation)
                    .await?;
                notify_terminal_checkpoint(operator_commands, sink_commands, epoch).await?;
                durable
            } else {
                let mut durable_cuts =
                    pause_checkpoint_sources(sources, epoch, cancellation).await?;
                if let Err(error) = add_restored_ended_source_cuts(checkpoint, &mut durable_cuts) {
                    abort_checkpoint_sources(sources, epoch);
                    return Err(error);
                }
                let promote_terminal = source_cuts_are_terminal(&durable_cuts);
                let durable_result = if promote_terminal {
                    let promotion = checkpoint
                        .status
                        .promote_terminal(epoch)
                        .and_then(|()| assembly.promote_terminal(epoch))
                        .and_then(|()| metrics.record_checkpoint_promoted_terminal());
                    if let Err(error) = promotion {
                        abort_checkpoint_sources(sources, epoch);
                        return Err(error);
                    }
                    *terminal_request_active = true;
                    match live_progress
                        .terminal_checkpoint_cut(epoch, &durable_cuts, cancellation)
                        .await
                    {
                        Ok(durable) => {
                            notify_terminal_checkpoint(operator_commands, sink_commands, epoch)
                                .await
                                .map(|()| durable)
                        }
                        Err(error) => Err(error),
                    }
                } else {
                    live_progress
                        .checkpoint_cut(epoch, &durable_cuts, cancellation)
                        .await
                };
                let durable = match durable_result {
                    Ok(durable) => durable,
                    Err(error) => {
                        abort_checkpoint_sources(sources, epoch);
                        return Err(error);
                    }
                };
                for source in sources.values() {
                    source.commit_checkpoint(epoch)?;
                }
                durable
            };
            assembly.sources = durable;
            #[cfg(test)]
            if checkpoint.inject_fault(CheckpointFaultPoint::SourceCut, cancellation)? {
                return Ok(false);
            }
            for (source_id, entry) in &assembly.sources {
                coordinator
                    .ack(CheckpointAck::source(
                        source_id,
                        epoch,
                        &checkpoint_digest(entry)?,
                    ))
                    .await?;
            }
            checkpoint
                .status
                .acknowledge_sources(epoch, assembly.sources.len());
        }
        CheckpointEvent::ReadyToPublish(epoch) => {
            publish_epoch_manifest(
                checkpoint,
                coordinator,
                sink_commands,
                cancellation,
                metrics,
                assembly,
                epoch,
            )
            .await?;
        }
        CheckpointEvent::Completed(epoch) => {
            #[cfg(test)]
            if checkpoint.inject_fault(CheckpointFaultPoint::CompletedCommit, cancellation)? {
                assembly.manifest_durable = false;
                return Ok(false);
            }
            if let Some(error) = assembly.deferred_publication_error.take() {
                return Err(error);
            }
            let terminal = assembly.terminal;
            assembly.complete(epoch)?;
            assembly.complete_metrics(metrics, terminal)?;
            retain_completed_epoch(checkpoint, cancellation, metrics, epoch).await?;
            *request_active = false;
            *terminal_request_active = false;
            if terminal {
                return Ok(true);
            }
        }
        CheckpointEvent::Failed(epoch, phase) => {
            checkpoint.status.fail(if phase == "timeout" {
                CheckpointFailureCategory::Timeout
            } else {
                CheckpointFailureCategory::Protocol
            });
            return Err(checkpoint_protocol_error(
                epoch,
                &format!("coordinator failed during {phase}"),
            ));
        }
        CheckpointEvent::PhaseAdvanced(epoch, phase) => {
            checkpoint.status.advance(epoch, phase);
            assembly.advance_metrics(metrics, phase)?;
        }
    }
    Ok(false)
}

fn source_cuts_are_terminal(
    cuts: &BTreeMap<super::progress::BindingIdentity, DurableSourceCut>,
) -> bool {
    !cuts.is_empty() && cuts.values().all(|cut| cut.ended)
}

async fn notify_terminal_checkpoint(
    operator_commands: &BTreeMap<String, mpsc::Sender<OperatorCheckpointCommand>>,
    sink_commands: &BTreeMap<String, mpsc::Sender<SinkCheckpointCommand>>,
    epoch: Epoch,
) -> crate::Result<()> {
    for command in operator_commands.values() {
        command
            .send(OperatorCheckpointCommand::Terminal(epoch))
            .await
            .map_err(|_| checkpoint_channel_closed("operator commands"))?;
    }
    for command in sink_commands.values() {
        command
            .send(SinkCheckpointCommand::Terminal(epoch))
            .await
            .map_err(|_| checkpoint_channel_closed("sink commands"))?;
    }
    Ok(())
}

async fn publish_epoch_manifest(
    checkpoint: &OpenedCheckpointRuntime,
    coordinator: &CheckpointCoordinatorHandle,
    sink_commands: &BTreeMap<String, mpsc::Sender<SinkCheckpointCommand>>,
    cancellation: &CancellationToken,
    metrics: &MetricsRecorder,
    assembly: &mut EpochManifestAssembly,
    epoch: Epoch,
) -> crate::Result<()> {
    assembly.expect_epoch(epoch)?;
    checkpoint
        .status
        .advance(epoch, CheckpointPhase::SinksPrecommitted);
    assembly.advance_metrics(metrics, CheckpointPhase::SinksPrecommitted)?;
    let manifest = build_epoch_manifest(checkpoint, assembly, epoch)?;
    let (state_bytes, manifest_bytes) = checkpoint_manifest_sizes(&manifest)?;
    metrics.record_checkpoint_manifest(state_bytes, manifest_bytes)?;
    let publication = checkpoint
        .transaction
        .publish_cancellable(
            PreparedEpochManifest {
                manifest,
                staged_segments: BTreeMap::new(),
            },
            cancellation,
        )
        .await;
    let publication = match publication {
        Ok(publication) => publication,
        Err(_) if cancellation.is_cancelled() => return Ok(()),
        Err(error) => return Err(error),
    };
    match publication {
        ManifestPublication::Durable => {
            assembly.manifest_durable = true;
            settle_durable_manifest(coordinator, sink_commands, epoch, assembly.terminal).await
        }
        ManifestPublication::Installed {
            parent_synced,
            error,
        } => {
            if parent_synced {
                assembly.manifest_durable = true;
                if !cancellation.is_cancelled() {
                    assembly.deferred_publication_error = Some(error);
                }
                settle_durable_manifest(coordinator, sink_commands, epoch, assembly.terminal)
                    .await?;
            } else {
                assembly.manifest_installed_unknown = true;
                checkpoint.status.installed_unknown(epoch);
                notify_sink_preserve(sink_commands, epoch).await?;
                return if cancellation.is_cancelled() {
                    Ok(())
                } else {
                    Err(error)
                };
            }
            Ok(())
        }
    }
}

async fn settle_durable_manifest(
    coordinator: &CheckpointCoordinatorHandle,
    sink_commands: &BTreeMap<String, mpsc::Sender<SinkCheckpointCommand>>,
    epoch: Epoch,
    terminal: bool,
) -> crate::Result<()> {
    notify_sink_manifest_durable(sink_commands, epoch, terminal).await?;
    coordinator.manifest_durable(epoch).await
}

async fn retain_completed_epoch(
    checkpoint: &OpenedCheckpointRuntime,
    cancellation: &CancellationToken,
    metrics: &MetricsRecorder,
    epoch: Epoch,
) -> crate::Result<()> {
    checkpoint.status.sinks_committed(epoch);
    #[cfg(test)]
    if checkpoint.inject_fault(CheckpointFaultPoint::Retention, cancellation)? {
        return Ok(());
    }
    let retained = checkpoint
        .transaction
        .retain_cancellable(&checkpoint.identity, None, cancellation)
        .await;
    if cancellation.is_cancelled() {
        return Ok(());
    }
    let report = match retained {
        Ok(report) => report,
        Err(error) => {
            checkpoint
                .status
                .fail(CheckpointFailureCategory::Maintenance);
            return Err(error);
        }
    };
    #[cfg(not(test))]
    let _ = cancellation;
    metrics.record_checkpoint_orphan_cleanup(report.removed_orphan_segments)?;
    checkpoint.status.complete(epoch);
    Ok(())
}

async fn notify_sink_manifest_durable(
    sink_commands: &BTreeMap<String, mpsc::Sender<SinkCheckpointCommand>>,
    epoch: Epoch,
    terminal: bool,
) -> crate::Result<()> {
    let mut first_error = None;
    for sender in sink_commands.values() {
        let command = if terminal {
            SinkCheckpointCommand::TerminalManifestDurable(epoch)
        } else {
            SinkCheckpointCommand::ManifestDurable(epoch)
        };
        if sender.send(command).await.is_err() && first_error.is_none() {
            first_error = Some(checkpoint_channel_closed("sink commands"));
        }
    }
    match first_error {
        Some(error) => Err(error),
        None => Ok(()),
    }
}

async fn notify_sink_preserve(
    sink_commands: &BTreeMap<String, mpsc::Sender<SinkCheckpointCommand>>,
    epoch: Epoch,
) -> crate::Result<()> {
    let mut first_error = None;
    for sender in sink_commands.values() {
        let sent = sender.send(SinkCheckpointCommand::Preserve(epoch)).await;
        if sent.is_err() && first_error.is_none() {
            first_error = Some(checkpoint_channel_closed("sink commands"));
        }
    }
    match first_error {
        Some(error) => Err(error),
        None => Ok(()),
    }
}

async fn wait_for_terminal_source_cuts(
    sources: &BTreeMap<String, SourceProgress>,
    cancellation: &CancellationToken,
) -> crate::Result<BTreeMap<super::progress::BindingIdentity, DurableSourceCut>> {
    let cuts = try_join_all(sources.iter().map(|(source_id, progress)| async move {
        let cut = progress.wait_for_terminal_cut(cancellation).await?;
        Ok::<_, CalcFlowError>((
            super::progress::BindingIdentity::new(source_id.as_str())?,
            cut.durable(source_id)?,
        ))
    }))
    .await?;
    Ok(cuts.into_iter().collect())
}

#[allow(
    clippy::too_many_arguments,
    reason = "terminal admission compares every prepared participant set atomically"
)]
async fn maybe_request_terminal_checkpoint(
    coordinator: &CheckpointCoordinatorHandle,
    expected_operators: &BTreeSet<String>,
    expected_sinks: &BTreeSet<String>,
    ready_operators: &BTreeSet<String>,
    ready_sinks: &BTreeSet<String>,
    sources_ready: bool,
    request_active: &mut bool,
    terminal_request_active: &mut bool,
) -> crate::Result<()> {
    if sources_ready
        && ready_operators == expected_operators
        && ready_sinks == expected_sinks
        && !*request_active
    {
        coordinator.request(CheckpointRequest::Terminal).await?;
        *request_active = true;
        *terminal_request_active = true;
    }
    Ok(())
}

async fn pause_checkpoint_sources(
    sources: &BTreeMap<String, SourceProgress>,
    epoch: Epoch,
    cancellation: &CancellationToken,
) -> crate::Result<BTreeMap<super::progress::BindingIdentity, DurableSourceCut>> {
    let cuts = try_join_all(sources.iter().map(|(source_id, progress)| async move {
        let cut = progress.barrier(epoch, cancellation).await?;
        Ok::<_, CalcFlowError>((
            super::progress::BindingIdentity::new(source_id.as_str())?,
            cut.durable(source_id)?,
        ))
    }))
    .await;
    match cuts {
        Ok(cuts) => Ok(cuts.into_iter().collect()),
        Err(error) => {
            abort_checkpoint_sources(sources, epoch);
            Err(error)
        }
    }
}

fn abort_checkpoint_sources(sources: &BTreeMap<String, SourceProgress>, epoch: Epoch) {
    for source in sources.values() {
        let _ = source.abort_checkpoint(epoch);
    }
}

async fn accept_operator_ack(
    acknowledgement: OperatorCheckpointAck,
    coordinator: &CheckpointCoordinatorHandle,
    assembly: &mut EpochManifestAssembly,
    status: &CheckpointStatusHandle,
) -> crate::Result<()> {
    assembly.expect_epoch(acknowledgement.epoch)?;
    insert_identical(
        &mut assembly.operators,
        &acknowledgement.node_id,
        acknowledgement.state.clone(),
        acknowledgement.epoch,
        "operator",
    )?;
    coordinator
        .ack(CheckpointAck::operator(
            &acknowledgement.node_id,
            acknowledgement.epoch,
            &checkpoint_digest(&acknowledgement.state)?,
        ))
        .await?;
    status.acknowledge_operators(acknowledgement.epoch, assembly.operators.len());
    Ok(())
}

async fn accept_sink_ack(
    acknowledgement: SinkCheckpointAck,
    coordinator: &CheckpointCoordinatorHandle,
    assembly: &mut EpochManifestAssembly,
    status: &CheckpointStatusHandle,
) -> crate::Result<()> {
    assembly.expect_epoch(acknowledgement.epoch)?;
    insert_identical(
        &mut assembly.sink_outputs,
        &acknowledgement.output_id,
        acknowledgement.sinks.clone(),
        acknowledgement.epoch,
        "sink output",
    )?;
    coordinator
        .ack(CheckpointAck::sink_precommit(
            &acknowledgement.output_id,
            acknowledgement.epoch,
            &checkpoint_digest(&acknowledgement.sinks)?,
        ))
        .await?;
    status.acknowledge_sink_precommits(acknowledgement.epoch, assembly.sink_outputs.len());
    Ok(())
}

async fn accept_sink_finalization(
    finalization: SinkFinalizeAck,
    coordinator: &CheckpointCoordinatorHandle,
    assembly: &mut EpochManifestAssembly,
    status: &CheckpointStatusHandle,
) -> crate::Result<()> {
    assembly.expect_epoch(finalization.epoch)?;
    assembly
        .finalized_sink_outputs
        .insert(finalization.output_id.clone());
    coordinator
        .ack(CheckpointAck::sink_commit(
            &finalization.output_id,
            finalization.epoch,
        ))
        .await?;
    status.acknowledge_sink_commits(finalization.epoch, assembly.finalized_sink_outputs.len());
    Ok(())
}

fn insert_identical<T: Clone + Eq>(
    entries: &mut BTreeMap<String, T>,
    id: &str,
    value: T,
    epoch: Epoch,
    kind: &str,
) -> crate::Result<()> {
    match entries.get(id) {
        Some(previous) if previous == &value => Ok(()),
        Some(_) => Err(checkpoint_protocol_error(
            epoch,
            &format!("conflicting duplicate {kind} acknowledgement for {id:?}"),
        )),
        None => {
            entries.insert(id.into(), value);
            Ok(())
        }
    }
}

fn build_epoch_manifest(
    checkpoint: &OpenedCheckpointRuntime,
    assembly: &EpochManifestAssembly,
    epoch: Epoch,
) -> crate::Result<CheckpointManifest> {
    let mut sinks = BTreeMap::new();
    for output_sinks in assembly.sink_outputs.values() {
        for (sink_id, entry) in output_sinks {
            if sinks.insert(sink_id.clone(), entry.clone()).is_some() {
                return Err(checkpoint_protocol_error(
                    epoch,
                    &format!("sink ID {sink_id:?} is bound to more than one output"),
                ));
            }
        }
    }
    if assembly.sources.keys().cloned().collect::<BTreeSet<_>>() != checkpoint.identity.source_ids
        || assembly.operators.keys().cloned().collect::<BTreeSet<_>>()
            != checkpoint.identity.operator_ids
        || sinks.keys().cloned().collect::<BTreeSet<_>>() != checkpoint.identity.sink_ids
    {
        return Err(checkpoint_protocol_error(
            epoch,
            "manifest participant IDs do not match the prepared job",
        ));
    }
    CheckpointManifest::new(CheckpointManifestFields {
        pipeline_name: checkpoint.identity.pipeline_name.clone(),
        pipeline_fingerprint: checkpoint.identity.pipeline_fingerprint.clone(),
        runtime_config_hash: checkpoint.identity.runtime_config_hash.clone(),
        epoch,
        created_at: Utc::now(),
        recovery_status: RecoveryStatus::Final,
        sources: assembly.sources.clone(),
        operators: assembly.operators.clone(),
        sinks,
    })
}

fn checkpoint_manifest_sizes(manifest: &CheckpointManifest) -> crate::Result<(u64, u64)> {
    let state_bytes = manifest
        .operators()
        .values()
        .flat_map(|operator| operator.segments.iter())
        .try_fold(0_u64, |total, handle| {
            total
                .checked_add(handle.byte_len())
                .ok_or_else(|| CalcFlowError::InvalidArgument {
                    field: "runtime.metrics.checkpoint.state_bytes".into(),
                    message: "counter overflow".into(),
                })
        })?;
    let manifest_bytes = u64::try_from(manifest.canonical_bytes()?.len()).map_err(|_| {
        CalcFlowError::InvalidArgument {
            field: "runtime.metrics.checkpoint.manifest_bytes".into(),
            message: "counter overflow".into(),
        }
    })?;
    Ok((state_bytes, manifest_bytes))
}

fn checkpoint_digest(value: &impl Serialize) -> crate::Result<String> {
    let value = serde_json::to_value(value).map_err(|error| CalcFlowError::Format {
        message: error.to_string(),
    })?;
    let canonical = crate::canonical_json(&value)?;
    Ok(hex::encode(Sha256::digest(canonical.as_bytes())))
}

fn checkpoint_protocol_error(epoch: Epoch, message: &str) -> CalcFlowError {
    CalcFlowError::Internal {
        message: format!("checkpoint epoch {}: {message}", epoch.as_u64()),
    }
}

fn checkpoint_channel_closed(channel: &str) -> CalcFlowError {
    CalcFlowError::Internal {
        message: format!("checkpoint {channel} channel closed"),
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

async fn open_checkpoint_connector_resources(
    resources: &mut Vec<ConnectorResource>,
    cancellation: &CancellationToken,
) -> Vec<Arc<RuntimeFailure>> {
    let (mut sources, mut sinks): (Vec<_>, Vec<_>) = std::mem::take(resources)
        .into_iter()
        .partition(|resource| matches!(resource, ConnectorResource::Source { .. }));
    let failures = open_connector_resources(&mut sources, cancellation).await;
    resources.extend(sources);
    if !failures.is_empty() || cancellation.is_cancelled() {
        return failures;
    }
    let failures = open_connector_resources(&mut sinks, cancellation).await;
    resources.extend(sinks);
    failures
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
            () = cancellation.cancelled(), if committed_terminal.is_none() => {}
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
    let cause = match primary_task_id {
        Some(primary_task_id) => TerminalCause::TaskFailure { primary_task_id },
        None => committed_terminal.unwrap_or(if graceful_requested {
            TerminalCause::GracefulShutdown
        } else {
            TerminalCause::NaturalEnd
        }),
    };
    let recovery_required = errors
        .iter()
        .any(|failure| matches!(&failure.error, CalcFlowError::RecoveryRequired { .. }));
    let state = match cause {
        _ if recovery_required => ContinuousJobState::RecoveryRequired,
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
            | FailureOrigin::SinkCheckpoint { .. }
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
        SinkFailurePhase::Checkpoint => FailureOrigin::SinkCheckpoint {
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
            cleanup_failures: Vec::new(),
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
    for (task_id, resource) in resources.iter_mut().enumerate() {
        let origin = resource.close_origin();
        if let Some(error) = close_resource(task_id as u64, resource).await {
            failures.push(Arc::new(RuntimeFailure { origin, error }));
        }
    }
    failures
}

async fn close_resource(task_id: u64, resource: &mut ConnectorResource) -> Option<CalcFlowError> {
    let close = AssertUnwindSafe(resource.close()).catch_unwind();
    match tokio::time::timeout(CONNECTOR_CLOSE_TIMEOUT, close).await {
        Ok(Ok(result)) => result.err(),
        Ok(Err(payload)) => Some(CalcFlowError::TaskPanicked {
            task_id,
            message: panic_message(payload.as_ref()),
        }),
        Err(_) => Some(CalcFlowError::Internal {
            message: "connector close exceeded private teardown bound of 5 seconds".into(),
        }),
    }
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
        path::Path,
        sync::{
            Arc,
            atomic::{AtomicBool, AtomicU64, AtomicUsize, Ordering},
        },
        task::Poll,
        time::Duration as StdDuration,
    };

    use async_trait::async_trait;
    use chrono::TimeZone;
    use datafusion::arrow::{array::Int64Array, record_batch::RecordBatch};
    use parking_lot::Mutex;
    use sha2::{Digest as _, Sha256};
    use tokio::sync::{Notify, Semaphore, mpsc};

    use super::{
        ABANDONED_RUNNER_WARNING, CheckpointCoordinatorHandle, CheckpointFailureCategory,
        CheckpointPhase, CheckpointRuntimeSpec, ContinuousJobState, ContinuousRunner,
        DriverCompletion, DriverOwnership, FailureOrigin as RuntimeFailureOrigin, JobCore,
        LaunchId, OneShotContinuousRunner, OneShotStartObserver, RunnerCore, RunnerDiagnostics,
        RunnerRegistryState, RunnerShutdownObserver, RuntimeFailure, RuntimeTaskProgress,
        TerminalCause, classify_failure_state, finish_running_report,
        maybe_request_terminal_checkpoint, notify_sink_manifest_durable,
        sanitize_managed_preflight_error, settle_durable_manifest, source_cuts_are_terminal,
    };
    use crate::{
        Batch, BatchKind, BatchMetadata, CalcFlowError, CancellationToken,
        CheckpointManifestFields, CursorManifestEntry, Edge, EdgeBudget, EventTime,
        ExpressionOperator, JsonMap, LocalStateBackend, ManifestIngressState,
        OperatorIngressManifestEntry, OperatorManifestEntry, OperatorMetadata, PipelineBuilder,
        Port, PortEndpoint, RecoveryStatus, Result, SinkDeliveryManifest, SinkManifestEntry,
        SourceManifestEntry, SourceWatermarkManifestState, StateBackend, StateHandle,
        StateLineageBackend, StateLineageKey, StreamCollector, StreamJobContext, StreamOperator,
        StreamOperatorContext, StreamRequirements, StreamRuntimeConfig, UdfRegistry, UnionOperator,
        runtime::streaming::{
            checkpoint::ManagedCheckpointRuntime,
            checkpoint::coordinator::{
                CheckpointAck, CheckpointEvent, CheckpointPhase as CoordinatorPhase,
                CheckpointRequest, ParticipantSet, spawn_checkpoint_coordinator,
            },
            job::{
                ContinuousJobSpec, M2DeliveryMode, NamedSinkBinding, NamedSourceBinding,
                OrdinarySinkBinding, OrdinaryStreamSink, TransactionalStreamSink,
                ValidatedOrdinarySink,
            },
            metrics::MetricsRecorder,
            progress::DurableSourceCut,
            sink_task::SinkCheckpointCommand,
            source_task::{Cursor, SourceBinding, SourceCapabilities, SourceEvent, StreamSource},
            supervisor::{SupervisionReport, TaskId},
        },
    };

    #[test]
    fn one_shot_runner_start_has_a_consuming_signature() {
        let start: fn(OneShotContinuousRunner, ContinuousJobSpec) -> OneShotStartObserver =
            OneShotContinuousRunner::start;

        let _ = start;
    }

    #[test]
    fn one_shot_runner_reuse_ui_is_a_move_error() {
        let manifest_dir = Path::new(env!("CARGO_MANIFEST_DIR"));
        let fixture = manifest_dir.join("tests/ui/one_shot_runner_reuse.rs");
        let output_dir = manifest_dir.join("../../target/ui-tests");
        std::fs::create_dir_all(&output_dir).unwrap();
        let rustc = std::env::var_os("RUSTC").unwrap_or_else(|| "rustc".into());
        let output = std::process::Command::new(rustc)
            .arg("--edition=2024")
            .arg("--crate-name=one_shot_runner_reuse")
            .arg(&fixture)
            .arg("--out-dir")
            .arg(output_dir)
            .output()
            .unwrap();
        let stderr = String::from_utf8_lossy(&output.stderr);

        assert!(!output.status.success(), "fixture unexpectedly compiled");
        assert!(stderr.contains("error[E0382]"), "{stderr}");
    }

    #[test]
    fn one_shot_checkpointed_start_has_a_consuming_signature() {
        let start: fn(
            OneShotContinuousRunner,
            ContinuousJobSpec,
            ManagedCheckpointRuntime,
        ) -> OneShotStartObserver = OneShotContinuousRunner::start_checkpointed;

        let _ = start;
    }

    #[test]
    fn managed_manifest_preflight_preserves_cancellation() {
        let error = sanitize_managed_preflight_error(
            CalcFlowError::Cancelled {
                run_id: "credential-secret-run".into(),
            },
            true,
            true,
        );

        assert!(matches!(
            error,
            CalcFlowError::Cancelled { ref run_id } if run_id == "managed-checkpoint-open"
        ));
    }

    #[tokio::test]
    async fn managed_checkpoint_identity_mismatch_is_redacted_before_lifecycle_work() {
        let directory = tempfile::tempdir().unwrap();
        let root = directory.path().join("credential-secret-checkpoint-root");
        let opened = ManagedCheckpointRuntime::new(&root)
            .unwrap()
            .open(&CancellationToken::new())
            .await
            .unwrap();
        let source = LifecycleProbe::default();
        let sink = LifecycleProbe::default();
        let resets = Arc::new(AtomicUsize::new(0));
        let job_spec = spec(false, Arc::clone(&resets), source.clone(), sink.clone());
        let config = StreamRuntimeConfig::default();
        let manifest = crate::CheckpointManifest::new(CheckpointManifestFields {
            pipeline_name: "credential-secret-foreign-job".into(),
            pipeline_fingerprint: job_spec.plan.fingerprint().into(),
            runtime_config_hash: job_spec.plan.runtime_config_hash(&config).unwrap(),
            epoch: crate::Epoch::INITIAL,
            created_at: chrono::Utc.with_ymd_and_hms(2026, 8, 12, 8, 0, 0).unwrap(),
            recovery_status: RecoveryStatus::Final,
            sources: BTreeMap::from([(
                "input".into(),
                SourceManifestEntry {
                    cursor: None,
                    identity_hash:
                        "0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef".into(),
                    sequence: 0,
                    ended: false,
                    watermark_policy: SourceWatermarkManifestState::Disabled { idle: false },
                },
            )]),
            operators: BTreeMap::from([(
                "node".into(),
                OperatorManifestEntry {
                    progress: BTreeMap::from([(
                        "input".into(),
                        OperatorIngressManifestEntry {
                            state: ManifestIngressState::Active,
                            watermark: None,
                        },
                    )]),
                    inline_metadata: BTreeMap::new(),
                    segments: Vec::new(),
                },
            )]),
            sinks: BTreeMap::from([(
                "sink".into(),
                SinkManifestEntry {
                    delivery: SinkDeliveryManifest::Ordinary,
                    pre_commit: None,
                },
            )]),
        })
        .unwrap();
        std::fs::write(
            opened
                .manifest_root_for_test()
                .join("manifest-00000000000000000001.json"),
            manifest.canonical_bytes().unwrap(),
        )
        .unwrap();
        drop(opened);

        let failure = OneShotContinuousRunner::new()
            .start_checkpointed(job_spec, ManagedCheckpointRuntime::new(&root).unwrap())
            .await
            .unwrap_err();

        for rendered in [format!("{failure:?}"), format!("{failure:#?}")] {
            assert!(!rendered.contains("credential-secret"));
            assert!(!rendered.contains(&root.display().to_string()));
        }
        assert!(matches!(
            failure.primary.error,
            CalcFlowError::CheckpointMismatch { .. }
        ));
        assert_eq!(resets.load(Ordering::SeqCst), 0);
        assert_eq!(source.opened.load(Ordering::SeqCst), 0);
        assert_eq!(sink.opened.load(Ordering::SeqCst), 0);
    }

    #[tokio::test]
    #[allow(
        clippy::too_many_lines,
        reason = "the redaction canaries and lifecycle assertions form one recovery scenario"
    )]
    async fn managed_checkpoint_missing_state_is_redacted_before_lifecycle_work() {
        let directory = tempfile::tempdir().unwrap();
        let root = directory.path().join("credential-secret-checkpoint-root");
        let source = LifecycleProbe::default();
        let sink = LifecycleProbe::default();
        let resets = Arc::new(AtomicUsize::new(0));
        let job_spec = spec(false, Arc::clone(&resets), source.clone(), sink.clone());
        let config = StreamRuntimeConfig::default();
        let prepared = crate::runtime::streaming::progress::prepare_stream_job(
            job_spec.plan.fingerprint(),
            &[crate::runtime::streaming::progress::SourceBindingSpec {
                descriptor: crate::runtime::streaming::progress::SourceDescriptor::new(
                    crate::runtime::streaming::progress::BindingIdentity::new("input").unwrap(),
                    crate::runtime::streaming::progress::DeclaredSchema::DynamicOrUnknown,
                    crate::runtime::streaming::progress::NativeWatermarkCapability::EmitsNative,
                    crate::runtime::streaming::progress::ReplayPositioningCapability::ExactPauseReportAndSeek,
                    None,
                )
                .with_delivery_and_bounds(true, 1, 1),
                watermark_policy:
                    crate::runtime::streaming::progress::WatermarkPolicy::SourceProvided,
            }],
            crate::runtime::streaming::progress::StreamProgressRuntimeConfig::default(),
        )
        .unwrap();
        let identity_hash = prepared.bindings[0].identity_hash();
        let key = StateLineageKey::new(job_spec.plan.name(), job_spec.plan.fingerprint()).unwrap();
        let missing_segment = local_state_handle(
            &key,
            "node",
            crate::Epoch::INITIAL,
            "credential-secret-segment",
            b"credential-secret-state",
        );
        let checksum = missing_segment.sha256().to_owned();
        let manifest = crate::CheckpointManifest::new(CheckpointManifestFields {
            pipeline_name: job_spec.plan.name().into(),
            pipeline_fingerprint: job_spec.plan.fingerprint().into(),
            runtime_config_hash: job_spec.plan.runtime_config_hash(&config).unwrap(),
            epoch: crate::Epoch::INITIAL,
            created_at: chrono::Utc.with_ymd_and_hms(2026, 8, 12, 8, 0, 0).unwrap(),
            recovery_status: RecoveryStatus::Final,
            sources: BTreeMap::from([(
                "input".into(),
                SourceManifestEntry {
                    cursor: Some(CursorManifestEntry {
                        order: "09".into(),
                        payload: BTreeMap::from([(
                            "credential-secret-cursor".into(),
                            serde_json::json!("credential-secret-payload"),
                        )]),
                    }),
                    identity_hash: identity_hash.clone(),
                    sequence: 1,
                    ended: false,
                    watermark_policy: SourceWatermarkManifestState::SourceProvided {
                        last_emitted_micros: None,
                        idle: false,
                    },
                },
            )]),
            operators: BTreeMap::from([(
                "node".into(),
                OperatorManifestEntry {
                    progress: BTreeMap::from([(
                        "input".into(),
                        OperatorIngressManifestEntry {
                            state: ManifestIngressState::Active,
                            watermark: None,
                        },
                    )]),
                    inline_metadata:
                        crate::pipeline::OperatorCheckpointCapability::CheckpointedStateful {
                            state_version: 1,
                        }
                        .encode_snapshot("node", crate::OperatorStateSnapshot::default())
                        .unwrap()
                        .inline_metadata,
                    segments: vec![missing_segment],
                },
            )]),
            sinks: BTreeMap::from([(
                "sink".into(),
                SinkManifestEntry {
                    delivery: SinkDeliveryManifest::Ordinary,
                    pre_commit: None,
                },
            )]),
        })
        .unwrap();
        let backend = LocalStateBackend::new(root.join("state")).await.unwrap();
        let lineage = backend.open_lineage(&key).await.unwrap();
        lineage
            .stage_segment(
                &manifest.operators()["node"].segments[0],
                b"credential-secret-state",
            )
            .await
            .unwrap();
        lineage
            .validate_segment(&manifest.operators()["node"].segments[0])
            .await
            .unwrap();
        lineage
            .publish_segment(&manifest.operators()["node"].segments[0])
            .await
            .unwrap();
        let transaction = crate::state::ManifestTransaction::open(
            Arc::from(lineage),
            &key,
            root.join("manifests"),
            config.retained_epochs,
        )
        .await
        .unwrap();
        transaction
            .publish(crate::state::PreparedEpochManifest {
                manifest,
                staged_segments: BTreeMap::new(),
            })
            .await
            .unwrap();
        drop(transaction);

        let failure_path = format!(
            "{}/credential-secret-segment/{checksum}/{identity_hash}/credential-secret-cursor/credential-secret-pre-commit",
            root.display()
        );
        let load_count = Arc::new(AtomicUsize::new(0));
        let checkpoint = CheckpointRuntimeSpec::managed_test_parts(
            Arc::new(FailAfterValidationBackend {
                inner: backend,
                load_count: Arc::clone(&load_count),
                failure_path: failure_path.into(),
            }),
            root.join("manifests"),
            config,
        )
        .unwrap();
        let mut runner = ContinuousRunner::new();
        let failure = runner
            .start_checkpointed(job_spec, checkpoint)
            .await
            .unwrap_err();
        runner.shutdown().await.unwrap();

        assert!(
            matches!(
                failure.primary.error,
                CalcFlowError::Internal { ref message }
                    if message == "managed checkpoint recovery failed"
            ),
            "unexpected failure: {failure:#?}"
        );
        let canaries = [
            "credential-secret",
            root.to_str().unwrap(),
            checksum.as_str(),
            identity_hash.as_str(),
        ];
        let mut current: &(dyn std::error::Error + 'static) = &failure.primary.error;
        loop {
            let rendered = format!("{current} {current:?}");
            for canary in canaries {
                assert!(!rendered.contains(canary), "leaked {canary:?}: {rendered}");
            }
            let Some(source) = current.source() else {
                break;
            };
            current = source;
        }
        assert_eq!(resets.load(Ordering::SeqCst), 0);
        assert_eq!(source.opened.load(Ordering::SeqCst), 0);
        assert_eq!(sink.opened.load(Ordering::SeqCst), 0);
        assert_eq!(load_count.load(Ordering::SeqCst), 3);
    }

    #[tokio::test]
    async fn owning_job_status_is_allowlisted_stably_ordered_and_observe_only() {
        let alpha_sink = LifecycleProbe::default();
        let zeta_sink = LifecycleProbe::default();
        let mut job_spec = spec(
            false,
            Arc::new(AtomicUsize::new(0)),
            LifecycleProbe::default(),
            LifecycleProbe::default(),
        );
        job_spec.sinks = vec![
            NamedSinkBinding {
                output_id: "output".into(),
                sink_id: "zeta".into(),
                binding: OrdinarySinkBinding::new(Box::new(ProbeSink(zeta_sink))),
            },
            NamedSinkBinding {
                output_id: "output".into(),
                sink_id: "alpha".into(),
                binding: OrdinarySinkBinding::new(Box::new(ProbeSink(alpha_sink))),
            },
        ];
        let job = OneShotContinuousRunner::new()
            .start(job_spec)
            .await
            .unwrap();

        let first = job.status();
        let repeated = job.status();

        assert_eq!(first, repeated);
        assert_eq!(
            first.state,
            crate::runtime::streaming::projection::JobState::Running
        );
        assert_eq!(first.job_id, job.id());
        assert_eq!(
            first.delivery["output"].requested,
            crate::DeliveryGuarantee::AtLeastOnce
        );
        assert_eq!(
            first.delivery["output"].effective,
            crate::DeliveryGuarantee::AtLeastOnce
        );
        assert_eq!(
            first.sources["input"].replay_positioning,
            crate::continuous::ReplayPositioning::ExactPauseReportAndSeek
        );
        assert_eq!(first.sources["input"].max_batch_rows, 1);
        assert_eq!(first.sources["input"].max_batch_bytes, 1);
        assert_eq!(first.edges.values().next().unwrap().envelope_limit, 1);
        assert_eq!(first.edges.values().next().unwrap().row_limit, 1);
        assert_eq!(first.edges.values().next().unwrap().byte_limit, 1);
        assert_eq!(
            first.sinks.keys().cloned().collect::<Vec<_>>(),
            vec!["alpha".to_owned(), "zeta".to_owned()]
        );

        let encoded = serde_json::to_value(&first).unwrap();
        let keys = encoded
            .as_object()
            .unwrap()
            .keys()
            .map(String::as_str)
            .collect::<BTreeSet<_>>();
        assert_eq!(
            keys,
            BTreeSet::from([
                "checkpoint",
                "delivery",
                "edges",
                "job_id",
                "metrics_overflowed",
                "operators",
                "sinks",
                "sources",
                "state",
                "task_count",
                "task_errors",
                "terminal_cause",
            ])
        );
        let encoded = encoded.to_string();
        for forbidden in [
            "cursor",
            "payload",
            "tasks",
            "progress",
            "abandoned_runner_drops",
            "latest_observed_order",
            "durable_order",
        ] {
            assert!(!encoded.contains(forbidden), "leaked field {forbidden:?}");
        }

        let outcome = job.cancel().await;
        assert_eq!(outcome.state, ContinuousJobState::Cancelled);
        let terminal = job.status();
        assert_eq!(
            terminal.state,
            crate::runtime::streaming::projection::JobState::Cancelled
        );
        assert_eq!(
            terminal.terminal_cause,
            Some(crate::runtime::streaming::projection::TerminalCause::ExplicitCancel)
        );
    }

    #[tokio::test]
    async fn owning_job_status_remains_safe_during_a_concurrent_lifecycle_transition() {
        const CURSOR_SENTINEL: &str = "private-cursor-order-redaction-sentinel";
        const PAYLOAD_SENTINEL: &str = "private-connector-payload-redaction-sentinel";
        const PATH_SENTINEL: &str = "/srv/private/checkpoints/customer-42";

        let source = LifecycleProbe::default();
        let cursor = Cursor::new(
            "input",
            CURSOR_SENTINEL.as_bytes().to_vec(),
            BTreeMap::from([
                ("connection".into(), serde_json::json!(PAYLOAD_SENTINEL)),
                ("path".into(), serde_json::json!(PATH_SENTINEL)),
            ]),
        )
        .unwrap();
        let mut job_spec = spec(
            false,
            Arc::new(AtomicUsize::new(0)),
            source.clone(),
            LifecycleProbe::default(),
        );
        job_spec.sources[0].binding =
            SourceBinding::new(Box::new(ProbeSource(source)), Some(cursor), 19).unwrap();
        let job = OneShotContinuousRunner::new()
            .start(job_spec)
            .await
            .unwrap();

        let observe = async {
            loop {
                let status = job.status();
                assert!(status.delivery.keys().is_sorted());
                assert!(status.edges.keys().is_sorted());
                assert!(status.sources.keys().is_sorted());
                assert!(status.operators.keys().is_sorted());
                assert!(status.sinks.keys().is_sorted());
                match status.state {
                    crate::runtime::streaming::projection::JobState::Running
                    | crate::runtime::streaming::projection::JobState::Draining => {
                        assert_eq!(status.terminal_cause, None);
                    }
                    crate::runtime::streaming::projection::JobState::Completed
                    | crate::runtime::streaming::projection::JobState::Cancelled
                    | crate::runtime::streaming::projection::JobState::Failed
                    | crate::runtime::streaming::projection::JobState::RecoveryRequired => {
                        assert!(status.terminal_cause.is_some());
                        assert_eq!(status.task_count, 0);
                    }
                }
                let encoded = serde_json::to_string(&status).unwrap();
                for sentinel in [CURSOR_SENTINEL, PAYLOAD_SENTINEL, PATH_SENTINEL] {
                    assert!(!encoded.contains(sentinel), "leaked sentinel {sentinel:?}");
                }
                if status.state == crate::runtime::streaming::projection::JobState::Cancelled {
                    break;
                }
                tokio::task::yield_now().await;
            }
        };
        let cancel = job.cancel();

        let ((), outcome) = tokio::join!(observe, cancel);

        assert_eq!(outcome.state, ContinuousJobState::Cancelled);
        assert_eq!(job.status().sources["input"].next_sequence, Some(19));
    }

    #[tokio::test]
    async fn owning_job_waiters_observe_one_terminal_without_owning_cancellation() {
        let source = LifecycleProbe::default();
        let sink = LifecycleProbe::default();
        let job = OneShotContinuousRunner::new()
            .start(spec(
                false,
                Arc::new(AtomicUsize::new(0)),
                source.clone(),
                sink.clone(),
            ))
            .await
            .unwrap();
        let runner = job.runner_probe_for_test();

        let mut dropped_waiter = Box::pin(job.wait());
        assert!(matches!(
            futures::poll!(dropped_waiter.as_mut()),
            Poll::Pending
        ));
        drop(dropped_waiter);
        assert_eq!(job.state(), ContinuousJobState::Running);

        let mut dropped_shutdown = Box::pin(job.shutdown());
        assert!(matches!(
            futures::poll!(dropped_shutdown.as_mut()),
            Poll::Pending
        ));
        drop(dropped_shutdown);
        assert_eq!(job.state(), ContinuousJobState::Draining);

        let first = job.cancel();
        let second = job.cancel();
        let (first, second) = tokio::join!(first, second);

        assert!(Arc::ptr_eq(&first, &second));
        assert_eq!(first.state, ContinuousJobState::Cancelled);
        assert_eq!(first.cause, TerminalCause::ExplicitCancel);
        assert_eq!(source.closed.load(Ordering::SeqCst), 1);
        assert_eq!(sink.closed.load(Ordering::SeqCst), 1);
        assert!(job.owner_settled_for_test());
        assert_eq!(runner.registry_counts(), (0, 0));
        assert!(runner.is_finished());
    }

    #[tokio::test]
    async fn one_shot_start_failure_waits_for_begun_resource_cleanup() {
        let source = LifecycleProbe::default();
        source.fail_open.store(true, Ordering::SeqCst);
        let sink = LifecycleProbe::default();

        let failure = OneShotContinuousRunner::new()
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
        assert!(failure.cleanup_failures.is_empty());
    }

    #[tokio::test]
    async fn one_shot_start_failure_surfaces_runner_lifecycle_join_failure() {
        let source = LifecycleProbe::default();
        source.fail_open.store(true, Ordering::SeqCst);
        let sink = LifecycleProbe::default();
        let runner = OneShotContinuousRunner::new();
        runner.panic_lifecycle_after_shutdown_for_test();

        let failure = runner
            .start(spec(false, Arc::new(AtomicUsize::new(0)), source, sink))
            .await
            .unwrap_err();

        assert!(matches!(
            failure.cleanup_failures.as_slice(),
            [failure]
                if matches!(failure.origin, super::FailureOrigin::RunnerLifecycle)
                    && matches!(failure.error, CalcFlowError::Internal { .. })
        ));
    }

    #[tokio::test]
    async fn owning_job_drop_cancels_while_an_existing_waiter_only_observes() {
        let source = LifecycleProbe::default();
        source.block_close.store(true, Ordering::SeqCst);
        let sink = LifecycleProbe::default();
        let job = OneShotContinuousRunner::new()
            .start(spec(
                false,
                Arc::new(AtomicUsize::new(0)),
                source.clone(),
                sink.clone(),
            ))
            .await
            .unwrap();
        let runner = job.runner_probe_for_test();
        let mut waiter = Box::pin(job.wait());
        let close_started = source.close_started.notified();

        drop(job);

        close_started.await;
        assert!(futures::poll!(waiter.as_mut()).is_pending());
        assert_eq!(runner.registry_counts().0, 1);
        assert!(!runner.is_finished());

        source.close_release.notify_waiters();
        let outcome = waiter.await;
        assert_eq!(outcome.state, ContinuousJobState::Cancelled);
        assert_eq!(outcome.cause, TerminalCause::ExplicitCancel);
        assert_eq!(source.closed.load(Ordering::SeqCst), 1);
        assert_eq!(sink.closed.load(Ordering::SeqCst), 1);
        assert_eq!(runner.registry_counts(), (0, 0));
        assert!(runner.is_finished());
    }

    #[tokio::test]
    async fn owning_job_natural_terminal_reaps_runner_without_a_waiter() {
        let source = LifecycleProbe::default();
        source.finite.store(true, Ordering::SeqCst);
        let sink = LifecycleProbe::default();
        let job = OneShotContinuousRunner::new()
            .start(spec(
                false,
                Arc::new(AtomicUsize::new(0)),
                source.clone(),
                sink.clone(),
            ))
            .await
            .unwrap();
        let runner = job.runner_probe_for_test();

        runner.join().await.unwrap();

        assert_eq!(job.state(), ContinuousJobState::Completed);
        assert_eq!(source.closed.load(Ordering::SeqCst), 1);
        assert_eq!(sink.closed.load(Ordering::SeqCst), 1);
        assert_eq!(runner.registry_counts(), (0, 0));
        assert!(runner.is_finished());
        let outcome = job.wait().await;
        assert_eq!(outcome.state, ContinuousJobState::Completed);
        assert_eq!(outcome.cause, TerminalCause::NaturalEnd);
        assert!(job.owner_settled_for_test());
    }

    #[tokio::test]
    async fn owning_job_contains_task_panic_before_publishing_terminal() {
        let source = LifecycleProbe::default();
        source.panic_next.store(true, Ordering::SeqCst);
        let sink = LifecycleProbe::default();
        let job = OneShotContinuousRunner::new()
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
        assert!(matches!(
            outcome.errors[0].error,
            CalcFlowError::TaskPanicked { ref message, .. }
                if message == "source next panicked"
        ));
        assert_eq!(source.closed.load(Ordering::SeqCst), 1);
        assert_eq!(sink.closed.load(Ordering::SeqCst), 1);
        assert!(job.owner_settled_for_test());
    }

    #[tokio::test]
    async fn owning_job_surfaces_runner_lifecycle_join_failure() {
        let source = LifecycleProbe::default();
        let sink = LifecycleProbe::default();
        let runner = OneShotContinuousRunner::new();
        runner.panic_lifecycle_after_shutdown_for_test();
        let job = runner
            .start(spec(false, Arc::new(AtomicUsize::new(0)), source, sink))
            .await
            .unwrap();
        let runner = job.runner_probe_for_test();

        let outcome = job.cancel().await;

        assert!(matches!(
            outcome.errors.last(),
            Some(failure)
                if matches!(failure.origin, super::FailureOrigin::RunnerLifecycle)
                    && matches!(failure.error, CalcFlowError::Internal { .. })
        ));
        assert_eq!(runner.registry_counts(), (0, 0));
        assert!(runner.is_finished());
    }

    #[tokio::test]
    async fn checkpointed_owning_job_releases_transaction_and_lineage_lease() {
        let directory = tempfile::tempdir().unwrap();
        let plan = PipelineBuilder::new("one-shot-checkpoint-lease")
            .unwrap()
            .add_checkpoint_capable_node(
                "node",
                Box::new(StressForwardOperator::new(None)) as Box<dyn StreamOperator>,
            )
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
        let source_polls = Arc::new(AtomicUsize::new(0));
        let source_closed = Arc::new(AtomicUsize::new(0));
        let sink_opened = Arc::new(AtomicUsize::new(0));
        let sink_closed = Arc::new(AtomicUsize::new(0));
        let spec = ContinuousJobSpec {
            context: StreamJobContext::new(
                9_901,
                plan.fingerprint(),
                JsonMap::new(),
                None,
                CancellationToken::new(),
            ),
            plan,
            sources: vec![NamedSourceBinding {
                binding_id: "input".into(),
                binding: counting_pending_binding(0, &source_polls, &source_closed),
            }],
            sinks: vec![NamedSinkBinding {
                output_id: "output".into(),
                sink_id: "sink".into(),
                binding: OrdinarySinkBinding::new_transactional(Box::new(CheckpointProbeSink {
                    opened: Arc::clone(&sink_opened),
                    closed: Arc::clone(&sink_closed),
                })),
            }],
            edge_budget: EdgeBudget {
                max_rows: 1,
                max_bytes: 1 << 20,
            },
            delivery_mode: M2DeliveryMode::ProcessLocalOrdered,
        };
        let checkpoint = ManagedCheckpointRuntime::new(directory.path()).unwrap();
        let job = OneShotContinuousRunner::new()
            .start_checkpointed(spec, checkpoint)
            .await
            .unwrap();
        wait_for_counter(&source_polls, 1).await;

        let outcome = job.cancel().await;

        assert_eq!(outcome.state, ContinuousJobState::Cancelled);
        assert_eq!(source_closed.load(Ordering::SeqCst), 1);
        assert_eq!(sink_opened.load(Ordering::SeqCst), 1);
        assert_eq!(sink_closed.load(Ordering::SeqCst), 1);
        ManagedCheckpointRuntime::new(directory.path())
            .unwrap()
            .open(&CancellationToken::new())
            .await
            .unwrap();
    }

    struct ResetOperator {
        inputs: [Port; 1],
        outputs: [Port; 1],
        resets: Arc<AtomicUsize>,
        fail_reset: bool,
        panic_reset: bool,
    }

    struct BlockingEntryOperator {
        inputs: [Port; 1],
        outputs: [Port; 1],
        entered: Arc<AtomicBool>,
        release: Arc<AtomicBool>,
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

    impl OperatorMetadata for BlockingEntryOperator {
        fn name(&self) -> &'static str {
            "blocking-entry"
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
    impl StreamOperator for BlockingEntryOperator {
        async fn process_data(
            &mut self,
            _ingress: &str,
            batch: Batch,
            _context: &StreamOperatorContext<'_>,
            output: &mut dyn StreamCollector,
        ) -> Result<()> {
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
            self.entered.store(true, Ordering::SeqCst);
            while !self.release.load(Ordering::SeqCst) {
                std::thread::yield_now();
            }
            Ok(())
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
                cursor: Cursor::unbound(vec![1], JsonMap::new()).unwrap(),
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

    #[derive(Default)]
    struct MixedDeliveryTransactionalState {
        committed_epochs: BTreeSet<u64>,
        visible: Vec<(String, u64)>,
    }

    #[derive(Default)]
    struct MixedDeliveryProbes {
        source_closed: Arc<AtomicUsize>,
        transactional_closed: Arc<AtomicUsize>,
        ordinary_closed: Arc<AtomicUsize>,
        transactional: Arc<Mutex<MixedDeliveryTransactionalState>>,
        ordinary_writes: Arc<Mutex<Vec<(String, String, u64)>>>,
    }

    struct MixedDeliveryTransactionalSink {
        pending: Vec<(String, u64)>,
        state: Arc<Mutex<MixedDeliveryTransactionalState>>,
        closed: Arc<AtomicUsize>,
    }

    struct CountingPendingSource {
        events: VecDeque<SourceEvent>,
        polls: Arc<AtomicUsize>,
        closed: Arc<AtomicUsize>,
    }

    struct CheckpointProbeSink {
        opened: Arc<AtomicUsize>,
        closed: Arc<AtomicUsize>,
    }

    #[derive(Clone)]
    struct FailOnceRetentionBackend {
        inner: LocalStateBackend,
        failure_armed: Arc<AtomicBool>,
    }

    struct FailOnceRetentionLineage {
        inner: Box<dyn StateLineageBackend>,
        failure_armed: Arc<AtomicBool>,
    }

    #[derive(Clone)]
    struct FailAfterValidationBackend {
        inner: LocalStateBackend,
        load_count: Arc<AtomicUsize>,
        failure_path: Arc<str>,
    }

    struct FailAfterValidationLineage {
        inner: Box<dyn StateLineageBackend>,
        load_count: Arc<AtomicUsize>,
        failure_path: Arc<str>,
    }

    #[async_trait]
    impl StateBackend for FailOnceRetentionBackend {
        async fn open_lineage(
            &self,
            key: &StateLineageKey,
        ) -> Result<Box<dyn StateLineageBackend>> {
            Ok(Box::new(FailOnceRetentionLineage {
                inner: self.inner.open_lineage(key).await?,
                failure_armed: Arc::clone(&self.failure_armed),
            }))
        }
    }

    #[async_trait]
    impl StateBackend for FailAfterValidationBackend {
        async fn open_lineage(
            &self,
            key: &StateLineageKey,
        ) -> Result<Box<dyn StateLineageBackend>> {
            Ok(Box::new(FailAfterValidationLineage {
                inner: self.inner.open_lineage(key).await?,
                load_count: Arc::clone(&self.load_count),
                failure_path: Arc::clone(&self.failure_path),
            }))
        }
    }

    #[async_trait]
    impl StateLineageBackend for FailOnceRetentionLineage {
        fn identity_hash(&self) -> &str {
            self.inner.identity_hash()
        }

        async fn stage_segment(&self, handle: &StateHandle, bytes: &[u8]) -> Result<()> {
            self.inner.stage_segment(handle, bytes).await
        }

        async fn validate_segment(&self, handle: &StateHandle) -> Result<()> {
            self.inner.validate_segment(handle).await
        }

        async fn publish_segment(&self, handle: &StateHandle) -> Result<()> {
            self.inner.publish_segment(handle).await
        }

        async fn load_segment(&self, handle: &StateHandle) -> Result<Vec<u8>> {
            self.inner.load_segment(handle).await
        }

        async fn collect_orphans(&self, retained: &[StateHandle]) -> Result<usize> {
            if self.failure_armed.swap(false, Ordering::SeqCst) {
                return Err(CalcFlowError::Internal {
                    message: "injected retention failure".into(),
                });
            }
            self.inner.collect_orphans(retained).await
        }
    }

    #[async_trait]
    impl StateLineageBackend for FailAfterValidationLineage {
        fn identity_hash(&self) -> &str {
            self.inner.identity_hash()
        }

        async fn stage_segment(&self, handle: &StateHandle, bytes: &[u8]) -> Result<()> {
            self.inner.stage_segment(handle, bytes).await
        }

        async fn validate_segment(&self, handle: &StateHandle) -> Result<()> {
            self.inner.validate_segment(handle).await
        }

        async fn publish_segment(&self, handle: &StateHandle) -> Result<()> {
            self.inner.publish_segment(handle).await
        }

        async fn load_segment(&self, handle: &StateHandle) -> Result<Vec<u8>> {
            if self.load_count.fetch_add(1, Ordering::SeqCst) >= 2 {
                return Err(CalcFlowError::Io {
                    path: self.failure_path.to_string(),
                    source: std::io::Error::new(
                        std::io::ErrorKind::NotFound,
                        "credential-secret-I/O-source",
                    ),
                });
            }
            self.inner.load_segment(handle).await
        }

        async fn collect_orphans(&self, retained: &[StateHandle]) -> Result<usize> {
            self.inner.collect_orphans(retained).await
        }
    }

    struct RecoveryProbeOperator {
        inputs: [Port; 1],
        outputs: [Port; 1],
        log: Arc<Mutex<Vec<String>>>,
    }

    impl OperatorMetadata for RecoveryProbeOperator {
        fn name(&self) -> &'static str {
            "recovery-probe"
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
    impl StreamOperator for RecoveryProbeOperator {
        async fn process_data(
            &mut self,
            _ingress: &str,
            _batch: Batch,
            _context: &StreamOperatorContext<'_>,
            _output: &mut dyn StreamCollector,
        ) -> Result<()> {
            Ok(())
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
            self.log.lock().push("operator-reset".into());
            Ok(())
        }

        fn restore(&mut self, snapshot: &crate::OperatorStateSnapshot) -> Result<()> {
            assert_eq!(
                snapshot.inline_metadata["restored"],
                serde_json::json!(true)
            );
            self.log.lock().push("operator-restore".into());
            Ok(())
        }
    }

    struct RecoveryProbeSource {
        log: Arc<Mutex<Vec<String>>>,
        closed: Arc<AtomicUsize>,
    }

    struct MixedRestoreSource {
        opens: Arc<AtomicUsize>,
        seeks: Arc<AtomicUsize>,
        polls: Arc<AtomicUsize>,
        closed: Arc<AtomicUsize>,
    }

    #[async_trait]
    impl StreamSource for MixedRestoreSource {
        async fn open(&mut self, cursor: Option<Cursor>) -> Result<()> {
            self.opens.fetch_add(1, Ordering::SeqCst);
            if cursor.is_some() {
                self.seeks.fetch_add(1, Ordering::SeqCst);
            }
            Ok(())
        }

        async fn next(&mut self) -> Result<Option<SourceEvent>> {
            self.polls.fetch_add(1, Ordering::SeqCst);
            std::future::pending().await
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

        fn native_watermark_capability(
            &self,
        ) -> crate::runtime::streaming::progress::NativeWatermarkCapability {
            crate::runtime::streaming::progress::NativeWatermarkCapability::NeverEmits
        }
    }

    #[async_trait]
    impl StreamSource for RecoveryProbeSource {
        async fn open(&mut self, cursor: Option<Cursor>) -> Result<()> {
            let order = cursor
                .as_ref()
                .map_or_else(|| "none".into(), |cursor| hex::encode(cursor.order()));
            self.log.lock().push(format!("source-open:{order}"));
            Ok(())
        }

        async fn next(&mut self) -> Result<Option<SourceEvent>> {
            std::future::pending().await
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

        fn native_watermark_capability(
            &self,
        ) -> crate::runtime::streaming::progress::NativeWatermarkCapability {
            crate::runtime::streaming::progress::NativeWatermarkCapability::NeverEmits
        }
    }

    struct RecoveryProbeSink {
        log: Arc<Mutex<Vec<String>>>,
        closed: Arc<AtomicUsize>,
    }

    struct PeriodicCheckpointSink {
        log: Arc<Mutex<Vec<String>>>,
        closed: Arc<AtomicUsize>,
    }

    struct FailOnceCommitSink {
        fail_commit: Arc<AtomicBool>,
        log: Arc<Mutex<Vec<String>>>,
        closed: Arc<AtomicUsize>,
    }

    struct BlockingCommitSink {
        commit_entered: Arc<AtomicBool>,
        commit_changed: Arc<Notify>,
        commit_release: Arc<Semaphore>,
        log: Arc<Mutex<Vec<String>>>,
        closed: Arc<AtomicUsize>,
    }

    #[async_trait]
    impl TransactionalStreamSink for PeriodicCheckpointSink {
        async fn open(&mut self) -> Result<()> {
            self.log.lock().push("sink-open".into());
            Ok(())
        }

        async fn begin_epoch(&mut self, epoch: crate::Epoch) -> Result<()> {
            self.log
                .lock()
                .push(format!("sink-begin:{}", epoch.as_u64()));
            Ok(())
        }

        async fn write(&mut self, _batch: &Batch) -> Result<()> {
            Ok(())
        }

        async fn pre_commit(&mut self, epoch: crate::Epoch) -> Result<JsonMap> {
            self.log
                .lock()
                .push(format!("sink-precommit:{}", epoch.as_u64()));
            Ok(BTreeMap::from([(
                "epoch".into(),
                serde_json::json!(epoch.as_u64()),
            )]))
        }

        async fn commit(&mut self, epoch: crate::Epoch, _state: &JsonMap) -> Result<()> {
            self.log
                .lock()
                .push(format!("sink-commit:{}", epoch.as_u64()));
            Ok(())
        }

        async fn abort(&mut self, epoch: crate::Epoch, _state: Option<&JsonMap>) -> Result<()> {
            self.log
                .lock()
                .push(format!("sink-abort:{}", epoch.as_u64()));
            Ok(())
        }

        async fn recover(&mut self, manifest: &crate::CheckpointManifest) -> Result<()> {
            self.log
                .lock()
                .push(format!("sink-recover:{}", manifest.epoch().as_u64()));
            Ok(())
        }

        async fn close(&mut self) -> Result<()> {
            self.closed.fetch_add(1, Ordering::SeqCst);
            self.log.lock().push("sink-close".into());
            Ok(())
        }
    }

    #[async_trait]
    impl TransactionalStreamSink for FailOnceCommitSink {
        async fn open(&mut self) -> Result<()> {
            self.log.lock().push("sink-open".into());
            Ok(())
        }

        async fn begin_epoch(&mut self, epoch: crate::Epoch) -> Result<()> {
            self.log
                .lock()
                .push(format!("sink-begin:{}", epoch.as_u64()));
            Ok(())
        }

        async fn write(&mut self, _batch: &Batch) -> Result<()> {
            Ok(())
        }

        async fn pre_commit(&mut self, epoch: crate::Epoch) -> Result<JsonMap> {
            self.log
                .lock()
                .push(format!("sink-precommit:{}", epoch.as_u64()));
            Ok(BTreeMap::from([(
                "epoch".into(),
                serde_json::json!(epoch.as_u64()),
            )]))
        }

        async fn commit(&mut self, epoch: crate::Epoch, _state: &JsonMap) -> Result<()> {
            self.log
                .lock()
                .push(format!("sink-commit:{}", epoch.as_u64()));
            if self.fail_commit.swap(false, Ordering::SeqCst) {
                return Err(CalcFlowError::Internal {
                    message: "injected post-manifest commit failure".into(),
                });
            }
            Ok(())
        }

        async fn abort(&mut self, epoch: crate::Epoch, _state: Option<&JsonMap>) -> Result<()> {
            self.log
                .lock()
                .push(format!("sink-abort:{}", epoch.as_u64()));
            Ok(())
        }

        async fn recover(&mut self, manifest: &crate::CheckpointManifest) -> Result<()> {
            self.log
                .lock()
                .push(format!("sink-recover:{}", manifest.epoch().as_u64()));
            Ok(())
        }

        async fn close(&mut self) -> Result<()> {
            self.closed.fetch_add(1, Ordering::SeqCst);
            self.log.lock().push("sink-close".into());
            Ok(())
        }
    }

    #[async_trait]
    impl TransactionalStreamSink for BlockingCommitSink {
        async fn open(&mut self) -> Result<()> {
            self.log.lock().push("sink-open".into());
            Ok(())
        }

        async fn begin_epoch(&mut self, epoch: crate::Epoch) -> Result<()> {
            self.log
                .lock()
                .push(format!("sink-begin:{}", epoch.as_u64()));
            Ok(())
        }

        async fn write(&mut self, _batch: &Batch) -> Result<()> {
            Ok(())
        }

        async fn pre_commit(&mut self, epoch: crate::Epoch) -> Result<JsonMap> {
            self.log
                .lock()
                .push(format!("sink-precommit:{}", epoch.as_u64()));
            Ok(BTreeMap::from([(
                "epoch".into(),
                serde_json::json!(epoch.as_u64()),
            )]))
        }

        async fn commit(&mut self, epoch: crate::Epoch, _state: &JsonMap) -> Result<()> {
            self.commit_entered.store(true, Ordering::Release);
            self.commit_changed.notify_waiters();
            self.commit_release
                .acquire()
                .await
                .expect("test commit gate remains open")
                .forget();
            self.log
                .lock()
                .push(format!("sink-commit:{}", epoch.as_u64()));
            Ok(())
        }

        async fn abort(&mut self, epoch: crate::Epoch, _state: Option<&JsonMap>) -> Result<()> {
            self.log
                .lock()
                .push(format!("sink-abort:{}", epoch.as_u64()));
            Ok(())
        }

        async fn recover(&mut self, manifest: &crate::CheckpointManifest) -> Result<()> {
            self.log
                .lock()
                .push(format!("sink-recover:{}", manifest.epoch().as_u64()));
            Ok(())
        }

        async fn close(&mut self) -> Result<()> {
            self.closed.fetch_add(1, Ordering::SeqCst);
            self.log.lock().push("sink-close".into());
            Ok(())
        }
    }

    #[async_trait]
    impl TransactionalStreamSink for RecoveryProbeSink {
        async fn open(&mut self) -> Result<()> {
            self.log.lock().push("sink-open".into());
            Ok(())
        }

        async fn begin_epoch(&mut self, epoch: crate::Epoch) -> Result<()> {
            self.log
                .lock()
                .push(format!("sink-begin:{}", epoch.as_u64()));
            Ok(())
        }

        async fn write(&mut self, _batch: &Batch) -> Result<()> {
            Ok(())
        }

        async fn pre_commit(&mut self, _epoch: crate::Epoch) -> Result<JsonMap> {
            Ok(JsonMap::new())
        }

        async fn commit(&mut self, _epoch: crate::Epoch, _state: &JsonMap) -> Result<()> {
            Ok(())
        }

        async fn abort(&mut self, _epoch: crate::Epoch, _state: Option<&JsonMap>) -> Result<()> {
            Ok(())
        }

        async fn recover(&mut self, manifest: &crate::CheckpointManifest) -> Result<()> {
            self.log
                .lock()
                .push(format!("sink-recover:{}", manifest.epoch().as_u64()));
            Ok(())
        }

        async fn close(&mut self) -> Result<()> {
            self.closed.fetch_add(1, Ordering::SeqCst);
            Ok(())
        }
    }

    #[async_trait]
    impl TransactionalStreamSink for CheckpointProbeSink {
        async fn open(&mut self) -> Result<()> {
            self.opened.fetch_add(1, Ordering::SeqCst);
            Ok(())
        }

        async fn begin_epoch(&mut self, _epoch: crate::Epoch) -> Result<()> {
            Ok(())
        }

        async fn write(&mut self, _batch: &Batch) -> Result<()> {
            Ok(())
        }

        async fn pre_commit(&mut self, _epoch: crate::Epoch) -> Result<JsonMap> {
            Ok(JsonMap::new())
        }

        async fn commit(&mut self, _epoch: crate::Epoch, _state: &JsonMap) -> Result<()> {
            Ok(())
        }

        async fn abort(&mut self, _epoch: crate::Epoch, _state: Option<&JsonMap>) -> Result<()> {
            Ok(())
        }

        async fn recover(&mut self, _manifest: &crate::CheckpointManifest) -> Result<()> {
            Ok(())
        }

        async fn close(&mut self) -> Result<()> {
            self.closed.fetch_add(1, Ordering::SeqCst);
            Ok(())
        }
    }

    struct ZeroCostLifecycleSource {
        events: VecDeque<SourceEvent>,
        polls: Arc<AtomicUsize>,
        poll_calls: mpsc::UnboundedSender<usize>,
        eof_observed: Arc<AtomicUsize>,
        closed: Arc<AtomicUsize>,
        end: ZeroCostSourceEnd,
    }

    #[derive(Clone, Copy)]
    enum ZeroCostSourceEnd {
        Eof,
        Pending,
        Error,
    }

    struct ZeroCostLifecycleSink {
        gate: Arc<Semaphore>,
        writes: Arc<Mutex<Vec<u64>>>,
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
                    cursor: Cursor::unbound(vec![1], JsonMap::new()).unwrap(),
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

    #[async_trait]
    impl StreamSource for ZeroCostLifecycleSource {
        async fn open(&mut self, _cursor: Option<Cursor>) -> Result<()> {
            Ok(())
        }

        async fn next(&mut self) -> Result<Option<SourceEvent>> {
            let poll = self.polls.fetch_add(1, Ordering::SeqCst) + 1;
            if let Some(event) = self.events.pop_front() {
                let _ = self.poll_calls.send(poll);
                return Ok(Some(event));
            }
            match std::mem::replace(&mut self.end, ZeroCostSourceEnd::Pending) {
                ZeroCostSourceEnd::Eof => {
                    self.eof_observed.fetch_add(1, Ordering::SeqCst);
                    let _ = self.poll_calls.send(poll);
                    Ok(None)
                }
                ZeroCostSourceEnd::Error => {
                    let _ = self.poll_calls.send(poll);
                    Err(CalcFlowError::Internal {
                        message: "zero-cost lifecycle source failed".into(),
                    })
                }
                ZeroCostSourceEnd::Pending => {
                    let _ = self.poll_calls.send(poll);
                    std::future::pending().await
                }
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

    #[async_trait]
    impl OrdinaryStreamSink for ZeroCostLifecycleSink {
        async fn open(&mut self) -> Result<()> {
            Ok(())
        }

        async fn write(&mut self, batch: &Batch) -> Result<()> {
            let permit = self
                .gate
                .acquire()
                .await
                .map_err(|_| CalcFlowError::Internal {
                    message: "zero-cost lifecycle sink gate closed".into(),
                })?;
            permit.forget();
            assert_eq!(batch.num_rows(), 0);
            assert_eq!(batch.estimated_bytes()?, 0);
            self.writes.lock().push(batch.metadata().sequence());
            Ok(())
        }

        async fn close(&mut self) -> Result<()> {
            self.closed.fetch_add(1, Ordering::SeqCst);
            Ok(())
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
        fail_watermark: Option<Arc<AtomicBool>>,
    }

    impl StressForwardOperator {
        fn new(gates: Option<Vec<Arc<Semaphore>>>) -> Self {
            Self {
                inputs: [Port::new("input", BatchKind::Table, true, None).unwrap()],
                outputs: [Port::new("output", BatchKind::Table, true, None).unwrap()],
                gates: gates.map(|gates| Arc::new(Mutex::new(gates.into()))),
                fail_watermark: None,
            }
        }

        fn failing_on_watermark(flag: Arc<AtomicBool>) -> Self {
            Self {
                fail_watermark: Some(flag),
                ..Self::new(None)
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
            if self
                .fail_watermark
                .as_ref()
                .is_some_and(|flag| flag.load(Ordering::SeqCst))
            {
                return Err(CalcFlowError::Operator {
                    node_id: "node".into(),
                    message: "zero-cost lifecycle receiver failed".into(),
                });
            }
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
        zero_cost_writes: Arc<AtomicUsize>,
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
            if batch.num_rows() == 0 && batch.estimated_bytes()? == 0 {
                self.zero_cost_writes.fetch_add(1, Ordering::SeqCst);
            }
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

    fn mixed_delivery_records(state: &JsonMap) -> Result<Vec<(String, u64)>> {
        serde_json::from_value(state.get("records").cloned().ok_or_else(|| {
            CalcFlowError::CheckpointMismatch {
                message: "mixed-delivery pre-commit records are missing".into(),
            }
        })?)
        .map_err(|error| CalcFlowError::CheckpointMismatch {
            message: format!("mixed-delivery pre-commit records are invalid: {error}"),
        })
    }

    #[async_trait]
    impl TransactionalStreamSink for MixedDeliveryTransactionalSink {
        async fn open(&mut self) -> Result<()> {
            Ok(())
        }

        async fn begin_epoch(&mut self, _epoch: crate::Epoch) -> Result<()> {
            self.pending.clear();
            Ok(())
        }

        async fn write(&mut self, batch: &Batch) -> Result<()> {
            self.pending.push((
                batch.metadata().source().into(),
                batch.metadata().sequence(),
            ));
            Ok(())
        }

        async fn pre_commit(&mut self, _epoch: crate::Epoch) -> Result<JsonMap> {
            Ok(BTreeMap::from([(
                "records".into(),
                serde_json::json!(self.pending),
            )]))
        }

        async fn commit(&mut self, epoch: crate::Epoch, state: &JsonMap) -> Result<()> {
            let records = mixed_delivery_records(state)?;
            let mut durable = self.state.lock();
            if durable.committed_epochs.insert(epoch.as_u64()) {
                durable.visible.extend(records);
            }
            Ok(())
        }

        async fn abort(&mut self, _epoch: crate::Epoch, _state: Option<&JsonMap>) -> Result<()> {
            self.pending.clear();
            Ok(())
        }

        async fn recover(&mut self, manifest: &crate::CheckpointManifest) -> Result<()> {
            let state = manifest
                .sinks()
                .get("transactional")
                .and_then(|entry| entry.pre_commit.clone())
                .ok_or_else(|| CalcFlowError::CheckpointMismatch {
                    message: "mixed-delivery transactional recovery state is missing".into(),
                })?;
            self.commit(manifest.epoch(), &state).await
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

    fn zero_row() -> Batch {
        let record = RecordBatch::try_from_iter(vec![(
            "value",
            Arc::new(Int64Array::from(Vec::<i64>::new())) as _,
        )])
        .unwrap();
        Batch::table(vec![record], BatchMetadata::default()).unwrap()
    }

    fn mixed_delivery_fault_spec(job_id: u64, probes: &MixedDeliveryProbes) -> ContinuousJobSpec {
        let plan = PipelineBuilder::new("checkpoint-mixed-delivery-fault")
            .unwrap()
            .add_checkpoint_capable_node(
                "root",
                Box::new(StressForwardOperator::new(None)) as Box<dyn StreamOperator>,
            )
            .unwrap()
            .add_checkpoint_capable_node(
                "exact",
                Box::new(StressForwardOperator::new(None)) as Box<dyn StreamOperator>,
            )
            .unwrap()
            .add_checkpoint_capable_node(
                "ordinary",
                Box::new(StressForwardOperator::new(None)) as Box<dyn StreamOperator>,
            )
            .unwrap()
            .connect(Edge::new(
                PortEndpoint::new("root", "output").unwrap(),
                PortEndpoint::new("exact", "input").unwrap(),
            ))
            .unwrap()
            .connect(Edge::new(
                PortEndpoint::new("root", "output").unwrap(),
                PortEndpoint::new("ordinary", "input").unwrap(),
            ))
            .unwrap()
            .compile_stream(
                &UdfRegistry::new().snapshot(),
                &StreamRequirements {
                    delivery: BTreeMap::from([(
                        "exact.output".into(),
                        crate::DeliveryGuarantee::ExactlyOnce,
                    )]),
                },
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
                binding: finite_binding(&[1], &probes.source_closed),
            }],
            sinks: vec![
                NamedSinkBinding {
                    output_id: "exact.output".into(),
                    sink_id: "transactional".into(),
                    binding: OrdinarySinkBinding::new_transactional(Box::new(
                        MixedDeliveryTransactionalSink {
                            pending: Vec::new(),
                            state: Arc::clone(&probes.transactional),
                            closed: Arc::clone(&probes.transactional_closed),
                        },
                    )),
                },
                NamedSinkBinding {
                    output_id: "ordinary.output".into(),
                    sink_id: "ordinary".into(),
                    binding: OrdinarySinkBinding::new(Box::new(OrderedRecordingSink {
                        id: "ordinary".into(),
                        writes: Arc::clone(&probes.ordinary_writes),
                        closed: Arc::clone(&probes.ordinary_closed),
                    })),
                },
            ],
            edge_budget: EdgeBudget {
                max_rows: 1,
                max_bytes: 1 << 20,
            },
            delivery_mode: M2DeliveryMode::ProcessLocalOrdered,
        }
    }

    fn assert_terminal_checkpoint_resources_released(job: &super::ContinuousJob) {
        let status = job.status();
        assert!(status.tasks.is_empty());
        assert!(status.edges.values().all(|edge| {
            edge.queue_depth == 0 && edge.charged_rows == 0 && edge.charged_bytes == 0
        }));
    }

    fn finite_binding(values: &[i64], closed: &Arc<AtomicUsize>) -> SourceBinding {
        let events = values
            .iter()
            .enumerate()
            .map(|(index, value)| SourceEvent::Data {
                batch: one_row(*value),
                cursor: Cursor::unbound(vec![u8::try_from(index + 1).unwrap()], JsonMap::new())
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
                cursor: Cursor::unbound(
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

    fn pending_checkpoint_spec(
        pipeline_name: &str,
        job_id: u64,
        source_polls: &Arc<AtomicUsize>,
        source_closed: &Arc<AtomicUsize>,
        sink: Box<dyn TransactionalStreamSink>,
    ) -> ContinuousJobSpec {
        let plan = PipelineBuilder::new(pipeline_name)
            .unwrap()
            .add_checkpoint_capable_node(
                "node",
                Box::new(StressForwardOperator::new(None)) as Box<dyn StreamOperator>,
            )
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
                job_id,
                plan.fingerprint(),
                JsonMap::new(),
                None,
                CancellationToken::new(),
            ),
            plan,
            sources: vec![NamedSourceBinding {
                binding_id: "input".into(),
                binding: counting_pending_binding(0, source_polls, source_closed),
            }],
            sinks: vec![NamedSinkBinding {
                output_id: "output".into(),
                sink_id: "sink".into(),
                binding: OrdinarySinkBinding::new_transactional(sink),
            }],
            edge_budget: EdgeBudget {
                max_rows: 1,
                max_bytes: 1 << 20,
            },
            delivery_mode: M2DeliveryMode::ProcessLocalOrdered,
        }
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

    #[test]
    fn checkpoint_admission_covers_all_operator_capability_classes() {
        let operator = || {
            UnionOperator::new(
                "merge",
                vec![
                    Port::new("left", BatchKind::Table, true, None).unwrap(),
                    Port::new("right", BatchKind::Table, true, None).unwrap(),
                ],
            )
            .unwrap()
        };
        let compile = |builder: PipelineBuilder| {
            builder
                .compile_stream(
                    &UdfRegistry::new().snapshot(),
                    &StreamRequirements::default(),
                )
                .unwrap()
                .into_runtime_parts(EdgeBudget::default())
                .unwrap()
        };
        let stateless = compile(
            PipelineBuilder::new("stateless")
                .unwrap()
                .add_node("merge", Box::new(operator()))
                .unwrap(),
        );
        let checkpointed = compile(
            PipelineBuilder::new("checkpointed")
                .unwrap()
                .add_checkpoint_capable_node(
                    "merge",
                    Box::new(operator()) as Box<dyn StreamOperator>,
                )
                .unwrap(),
        );
        let unproven = compile(
            PipelineBuilder::new("unproven")
                .unwrap()
                .add_node("merge", Box::new(operator()) as Box<dyn StreamOperator>)
                .unwrap(),
        );

        assert!(super::validate_checkpoint_operator_capabilities(&stateless).is_ok());
        assert!(super::validate_checkpoint_operator_capabilities(&checkpointed).is_ok());
        let error = super::validate_checkpoint_operator_capabilities(&unproven).unwrap_err();
        assert!(matches!(
            error,
            CalcFlowError::InvalidArgument { ref field, ref message }
                if field == "operators.merge.checkpoint_capability"
                    && message.contains("unproven")
        ));
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
            .add_checkpoint_capable_node("node", Box::new(operator) as Box<dyn StreamOperator>)
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

    fn blocking_entry_spec(
        entered: &Arc<AtomicBool>,
        release: &Arc<AtomicBool>,
        source: LifecycleProbe,
        sink: LifecycleProbe,
    ) -> ContinuousJobSpec {
        let operator = BlockingEntryOperator {
            inputs: [Port::new("input", BatchKind::Table, true, None).unwrap()],
            outputs: [Port::new("output", BatchKind::Table, true, None).unwrap()],
            entered: Arc::clone(entered),
            release: Arc::clone(release),
        };
        let plan = PipelineBuilder::new("blocking-entry")
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

    async fn wait_for_operator_entry(entered: &AtomicBool) -> bool {
        tokio::time::timeout(std::time::Duration::from_secs(5), async {
            while !entered.load(Ordering::SeqCst) {
                tokio::time::sleep(std::time::Duration::from_millis(1)).await;
            }
        })
        .await
        .is_ok()
    }

    async fn wait_for_counter(counter: &AtomicUsize, expected: usize) {
        for _ in 0..100 {
            if counter.load(Ordering::SeqCst) == expected {
                return;
            }
            tokio::task::yield_now().await;
        }
        assert_eq!(counter.load(Ordering::SeqCst), expected);
    }

    fn local_state_handle(
        key: &StateLineageKey,
        operator_id: &str,
        epoch: crate::Epoch,
        segment_id: &str,
        bytes: &[u8],
    ) -> StateHandle {
        let digest = |value: &[u8]| hex::encode(Sha256::digest(value));
        let lineage_hash =
            digest(format!("{}\0{}", key.pipeline_name(), key.pipeline_fingerprint()).as_bytes());
        let operator_hash = digest(operator_id.as_bytes());
        let segment_hash = digest(segment_id.as_bytes());
        let relative_path = format!(
            "committed/{lineage_hash}/{operator_hash}/{}-{segment_hash}.segment",
            epoch.as_u64()
        );
        StateHandle::new(
            operator_id,
            epoch,
            segment_id,
            &relative_path,
            u64::try_from(bytes.len()).unwrap(),
            &digest(bytes),
        )
        .unwrap()
    }

    fn forward_spec(
        job_id: u64,
        source: SourceBinding,
        sinks: Vec<NamedSinkBinding>,
    ) -> ContinuousJobSpec {
        forward_spec_with_operator(
            job_id,
            source,
            sinks,
            Box::new(StressForwardOperator::new(None)),
        )
    }

    fn forward_spec_with_operator(
        job_id: u64,
        source: SourceBinding,
        sinks: Vec<NamedSinkBinding>,
        operator: Box<dyn StreamOperator>,
    ) -> ContinuousJobSpec {
        let plan = PipelineBuilder::new("runtime-panic-cleanup")
            .unwrap()
            .add_node("node", operator)
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

    fn stress_plan(
        gates: &BTreeMap<StressGate, Arc<Semaphore>>,
        zero_cost_gate: &Arc<Semaphore>,
        zero_cost_batches: usize,
    ) -> crate::StreamExecutionPlan {
        let union = UnionOperator::new(
            "merge",
            vec![
                Port::new("left", BatchKind::Table, true, None).unwrap(),
                Port::new("right", BatchKind::Table, true, None).unwrap(),
            ],
        )
        .unwrap();
        let unary_gates = std::iter::repeat_with(|| Arc::clone(zero_cost_gate))
            .take(zero_cost_batches)
            .chain(
                [
                    StressGate::Edge0,
                    StressGate::Edge1,
                    StressGate::Edge2,
                    StressGate::Edge3,
                ]
                .map(|gate| Arc::clone(&gates[&gate])),
            )
            .collect();
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
        zero_cost_phase: Option<(&Arc<Semaphore>, usize)>,
        data_gates: [StressGate; 2],
        eof_gate: StressGate,
        values: [i64; 2],
        closed: &Arc<AtomicUsize>,
    ) -> SourceBinding {
        let zero_cost_count = zero_cost_phase.map_or(0, |(_, count)| count);
        let zero_cost_events = zero_cost_phase.into_iter().flat_map(|(gate, count)| {
            (0..count).map(move |index| {
                (
                    Arc::clone(gate),
                    Some(SourceEvent::Data {
                        batch: zero_row(),
                        cursor: Cursor::unbound(
                            vec![u8::try_from(index + 1).unwrap()],
                            JsonMap::new(),
                        )
                        .unwrap(),
                    }),
                )
            })
        });
        let events =
            zero_cost_events
                .chain(data_gates.into_iter().zip(values).enumerate().map(
                    |(index, (gate, value))| {
                        (
                            Arc::clone(&gates[&gate]),
                            Some(SourceEvent::Data {
                                batch: one_row(value),
                                cursor: Cursor::unbound(
                                    vec![u8::try_from(zero_cost_count + index + 1).unwrap()],
                                    JsonMap::new(),
                                )
                                .unwrap(),
                            }),
                        )
                    },
                ))
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

    async fn wait_for_stress_writes(
        primary_writes: &Mutex<Vec<(String, u64)>>,
        replica_writes: &Mutex<Vec<(String, u64)>>,
        expected: usize,
    ) {
        for _ in 0..100 {
            if primary_writes.lock().len() >= expected && replica_writes.lock().len() >= expected {
                return;
            }
            tokio::task::yield_now().await;
        }
    }

    fn assert_zero_cost_active_edges(job: &super::ContinuousJob) {
        let status = job.status();
        assert!(status.edges.values().all(|edge| {
            edge.queue_depth <= 1 && edge.charged_rows == 0 && edge.charged_bytes == 0
        }));
    }

    async fn run_zero_cost_stress_phase(
        job: &super::ContinuousJob,
        gates: [&Semaphore; 4],
        writes: [&Mutex<Vec<(String, u64)>>; 2],
        batches: usize,
        seed: u64,
    ) {
        let [
            input_gate,
            transform_gate,
            primary_sink_gate,
            replica_sink_gate,
        ] = gates;
        let [primary_writes, replica_writes] = writes;
        input_gate.add_permits(batches);
        transform_gate.add_permits(batches);
        for delivered in 1..=batches {
            primary_sink_gate.add_permits(1);
            replica_sink_gate.add_permits(1);
            wait_for_stress_writes(primary_writes, replica_writes, delivered).await;
            assert_eq!(primary_writes.lock().len(), delivered, "seed {seed}");
            assert_eq!(replica_writes.lock().len(), delivered, "seed {seed}");
            assert_zero_cost_active_edges(job);
        }
    }

    #[tokio::test]
    async fn durable_notification_attempts_every_sink_after_one_channel_closes() {
        let (closed_sender, closed_receiver) = mpsc::channel(1);
        drop(closed_receiver);
        let (open_sender, mut open_receiver) = mpsc::channel(1);
        let senders = BTreeMap::from([
            ("a-closed".into(), closed_sender),
            ("b-open".into(), open_sender),
        ]);

        let error = notify_sink_manifest_durable(&senders, crate::Epoch::INITIAL, false)
            .await
            .unwrap_err();

        assert!(matches!(error, CalcFlowError::Internal { .. }));
        assert!(matches!(
            open_receiver.recv().await,
            Some(SinkCheckpointCommand::ManifestDurable(
                crate::Epoch::INITIAL
            ))
        ));
    }

    fn durable_notification_manifest() -> crate::CheckpointManifest {
        crate::CheckpointManifest::new(CheckpointManifestFields {
            pipeline_name: "durable-notification-order".into(),
            pipeline_fingerprint:
                "0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef".into(),
            runtime_config_hash: "abcdef0123456789abcdef0123456789abcdef0123456789abcdef0123456789"
                .into(),
            epoch: crate::Epoch::INITIAL,
            created_at: chrono::Utc.with_ymd_and_hms(2026, 8, 10, 8, 0, 0).unwrap(),
            recovery_status: RecoveryStatus::Final,
            sources: BTreeMap::new(),
            operators: BTreeMap::new(),
            sinks: BTreeMap::from([(
                "sink".into(),
                SinkManifestEntry {
                    delivery: SinkDeliveryManifest::Transactional,
                    pre_commit: Some(BTreeMap::from([(
                        "prepared".into(),
                        serde_json::json!(true),
                    )])),
                },
            )]),
        })
        .unwrap()
    }

    async fn install_durable_notification_manifest(root: &Path) -> crate::CheckpointManifest {
        let manifest = durable_notification_manifest();
        let backend = LocalStateBackend::new(root.join("state")).await.unwrap();
        let key = StateLineageKey::new(manifest.pipeline_name(), manifest.pipeline_fingerprint())
            .unwrap();
        let lineage = backend.open_lineage(&key).await.unwrap();
        let transaction = crate::state::ManifestTransaction::open(
            Arc::from(lineage),
            &key,
            root.join("manifests"),
            2,
        )
        .await
        .unwrap();
        transaction
            .publish(crate::state::PreparedEpochManifest {
                manifest: manifest.clone(),
                staged_segments: BTreeMap::new(),
            })
            .await
            .unwrap();
        assert!(
            root.join("manifests/manifest-00000000000000000001.json")
                .exists()
        );
        manifest
    }

    async fn stopped_manifest_coordinator(timed_out: bool) -> CheckpointCoordinatorHandle {
        let cancellation = CancellationToken::new();
        let timeout = if timed_out {
            StdDuration::from_millis(1)
        } else {
            StdDuration::from_secs(1)
        };
        let (coordinator, mut events, task) = spawn_checkpoint_coordinator(
            ParticipantSet {
                sources: BTreeSet::from(["source".into()]),
                operators: BTreeSet::from(["operator".into()]),
                sinks: BTreeSet::from(["output".into()]),
            },
            crate::Epoch::INITIAL,
            4,
            timeout,
            cancellation.clone(),
        )
        .unwrap();
        if timed_out {
            coordinator
                .request(CheckpointRequest::Periodic)
                .await
                .unwrap();
            assert!(matches!(
                events.recv().await,
                Some(CheckpointEvent::Started(crate::Epoch::INITIAL))
            ));
            task.await.unwrap().unwrap_err();
        } else {
            cancellation.cancel();
            task.await.unwrap().unwrap();
        }
        coordinator
    }

    async fn assert_durable_notification_precedes_failed_bookkeeping(
        coordinator: &CheckpointCoordinatorHandle,
    ) {
        let (sink_sender, mut sink_receiver) = mpsc::channel(1);
        let error = settle_durable_manifest(
            coordinator,
            &BTreeMap::from([("output".into(), sink_sender)]),
            crate::Epoch::INITIAL,
            false,
        )
        .await
        .unwrap_err();
        assert!(matches!(error, CalcFlowError::Internal { .. }));
        assert!(matches!(
            sink_receiver.try_recv(),
            Ok(SinkCheckpointCommand::ManifestDurable(
                crate::Epoch::INITIAL
            ))
        ));
    }

    #[tokio::test]
    async fn installed_manifest_notifies_sinks_before_failed_bookkeeping_and_recovers_forward() {
        let directory = tempfile::tempdir().unwrap();
        let manifest = install_durable_notification_manifest(directory.path()).await;
        for timed_out in [false, true] {
            let coordinator = stopped_manifest_coordinator(timed_out).await;
            assert_durable_notification_precedes_failed_bookkeeping(&coordinator).await;
        }

        let log = Arc::new(Mutex::new(Vec::new()));
        let mut sinks = vec![ValidatedOrdinarySink {
            sink_id: "sink".into(),
            binding: OrdinarySinkBinding::new_transactional(Box::new(RecoveryProbeSink {
                log: Arc::clone(&log),
                closed: Arc::new(AtomicUsize::new(0)),
            })),
        }];
        crate::runtime::streaming::sink_task::recover_transactional_sinks(&mut sinks, &manifest)
            .await
            .unwrap();
        assert_eq!(&*log.lock(), &["sink-recover:1"]);
    }

    #[tokio::test]
    async fn checkpoint_connector_open_waits_for_every_source_before_opening_sinks() {
        let source = LifecycleProbe::default();
        source.block_open.store(true, Ordering::SeqCst);
        let sink = LifecycleProbe::default();
        let mut resources = super::connector_resources(
            BTreeMap::from([(
                "input".into(),
                SourceBinding::new(Box::new(ProbeSource(source.clone())), None, 0).unwrap(),
            )]),
            BTreeMap::from([(
                "output".into(),
                vec![ValidatedOrdinarySink {
                    sink_id: "sink".into(),
                    binding: OrdinarySinkBinding::new(Box::new(ProbeSink(sink.clone()))),
                }],
            )]),
        );
        let cancellation = CancellationToken::new();
        let source_opened = source.open_started.notified();
        tokio::pin!(source_opened);
        {
            let opening = super::open_checkpoint_connector_resources(&mut resources, &cancellation);
            tokio::pin!(opening);

            tokio::select! {
                failures = &mut opening => panic!("checkpoint opens completed early: {failures:?}"),
                () = &mut source_opened => {}
            }
            assert_eq!(sink.opened.load(Ordering::SeqCst), 0);

            source.open_release.notify_waiters();
            assert!(opening.as_mut().await.is_empty());
        }
        assert_eq!(sink.opened.load(Ordering::SeqCst), 1);
        assert!(super::close_resources(&mut resources).await.is_empty());
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
    #[tokio::test]
    async fn seeded_paused_time_stress_runs_one_hundred_full_graph_schedules() {
        const MESSAGE_SLOT_LIMIT: usize = 1;
        const ZERO_COST_SLOT_MULTIPLIER: usize = 10;
        const ZERO_COST_PHASE_BATCHES: usize = MESSAGE_SLOT_LIMIT * ZERO_COST_SLOT_MULTIPLIER;
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
            run_seeded_zero_cost_shape_phase(seed, MESSAGE_SLOT_LIMIT, ZERO_COST_SLOT_MULTIPLIER)
                .await;
            let gates = DATA_GATES
                .into_iter()
                .map(|gate| (gate, Arc::new(Semaphore::new(0))))
                .collect::<BTreeMap<_, _>>();
            let zero_cost_source_gate = Arc::new(Semaphore::new(0));
            let zero_cost_operator_gate = Arc::new(Semaphore::new(0));
            let zero_cost_sink_a_gate = Arc::new(Semaphore::new(0));
            let zero_cost_sink_b_gate = Arc::new(Semaphore::new(0));
            let plan = stress_plan(&gates, &zero_cost_operator_gate, ZERO_COST_PHASE_BATCHES);
            let source_closed = Arc::new(AtomicUsize::new(0));
            let sink_closed = Arc::new(AtomicUsize::new(0));
            let sink_a_writes = Arc::new(Mutex::new(Vec::new()));
            let sink_b_writes = Arc::new(Mutex::new(Vec::new()));
            let sink_a_zero_cost_writes = Arc::new(AtomicUsize::new(0));
            let sink_b_zero_cost_writes = Arc::new(AtomicUsize::new(0));
            let sink_a_gates = std::iter::repeat_with(|| Arc::clone(&zero_cost_sink_a_gate))
                .take(ZERO_COST_PHASE_BATCHES)
                .chain(
                    [
                        StressGate::SinkA0,
                        StressGate::SinkA1,
                        StressGate::SinkA2,
                        StressGate::SinkA3,
                    ]
                    .map(|gate| Arc::clone(&gates[&gate])),
                );
            let sink_b_gates = std::iter::repeat_with(|| Arc::clone(&zero_cost_sink_b_gate))
                .take(ZERO_COST_PHASE_BATCHES)
                .chain(
                    [
                        StressGate::SinkB0,
                        StressGate::SinkB1,
                        StressGate::SinkB2,
                        StressGate::SinkB3,
                    ]
                    .map(|gate| Arc::clone(&gates[&gate])),
                );
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
                            Some((&zero_cost_source_gate, ZERO_COST_PHASE_BATCHES)),
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
                            None,
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
                            gates: sink_a_gates.collect(),
                            writes: Arc::clone(&sink_a_writes),
                            zero_cost_writes: Arc::clone(&sink_a_zero_cost_writes),
                            closed: Arc::clone(&sink_closed),
                        })),
                    },
                    NamedSinkBinding {
                        output_id: "branch_b.output".into(),
                        sink_id: "slow-b".into(),
                        binding: OrdinarySinkBinding::new(Box::new(StressSink {
                            gates: sink_b_gates.collect(),
                            writes: Arc::clone(&sink_b_writes),
                            zero_cost_writes: Arc::clone(&sink_b_zero_cost_writes),
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
            run_zero_cost_stress_phase(
                &job,
                [
                    &zero_cost_source_gate,
                    &zero_cost_operator_gate,
                    &zero_cost_sink_a_gate,
                    &zero_cost_sink_b_gate,
                ],
                [&sink_a_writes, &sink_b_writes],
                ZERO_COST_PHASE_BATCHES,
                seed,
            )
            .await;
            assert_eq!(
                sink_a_zero_cost_writes.load(Ordering::SeqCst),
                ZERO_COST_PHASE_BATCHES,
                "seed {seed}"
            );
            assert_eq!(
                sink_b_zero_cost_writes.load(Ordering::SeqCst),
                ZERO_COST_PHASE_BATCHES,
                "seed {seed}"
            );
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
                assert!(status.tasks.len() <= 11, "task growth at seed {seed}");
                assert!(
                    status.edges.values().all(|edge| {
                        edge.queue_depth <= 1
                            && edge.charged_rows <= 1
                            && edge.charged_bytes <= (1 << 20)
                    }),
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
                    edge.high_water_depth <= 1
                        && edge.high_water_rows <= 1
                        && edge.high_water_bytes <= (1 << 20)
                }),
                "high-water budget breach at seed {seed}: {:?}",
                status.edges
            );
            let a = sink_a_writes.lock().clone();
            let b = sink_b_writes.lock().clone();
            assert!(
                a.len() >= ZERO_COST_PHASE_BATCHES && b.len() >= ZERO_COST_PHASE_BATCHES,
                "zero-cost phase did not span ten slot-limit multiples at seed {seed}"
            );
            assert_source_fifo(seed, &a);
            assert_source_fifo(seed, &b);
            match seed % 3 {
                0 => {
                    assert_eq!(outcome.cause, TerminalCause::NaturalEnd, "seed {seed}");
                    assert_eq!(a.len(), ZERO_COST_PHASE_BATCHES + 4, "loss at seed {seed}");
                    assert_eq!(b.len(), ZERO_COST_PHASE_BATCHES + 4, "loss at seed {seed}");
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

    #[derive(Clone, Copy, Debug)]
    enum ZeroCostTermination {
        Graceful,
        ReceiverClose,
        Cancel,
        Error,
    }

    fn zero_cost_lifecycle_events(cycles: usize) -> VecDeque<SourceEvent> {
        (0..cycles)
            .flat_map(|index| {
                let watermark = i64::try_from(index + 1).unwrap();
                let cursor = u64::try_from(index + 1).unwrap().to_be_bytes().to_vec();
                [
                    SourceEvent::Data {
                        batch: zero_row(),
                        cursor: Cursor::unbound(cursor, JsonMap::new()).unwrap(),
                    },
                    SourceEvent::Idle,
                    SourceEvent::Watermark(EventTime::from_micros(watermark)),
                ]
            })
            .collect()
    }

    fn assert_zero_cost_shape_multiplier(
        events: &VecDeque<SourceEvent>,
        message_slot_limit: usize,
        multiplier: usize,
    ) {
        let required = message_slot_limit * multiplier;
        let mut idle = 0;
        let mut watermarks = Vec::new();
        let mut empty_data = 0;
        for event in events {
            match event {
                SourceEvent::Idle => idle += 1,
                SourceEvent::Watermark(watermark) => watermarks.push(watermark.as_micros()),
                SourceEvent::Data { batch, .. } => {
                    assert_eq!(batch.num_rows(), 0);
                    assert_eq!(batch.estimated_bytes().unwrap(), 0);
                    empty_data += 1;
                }
            }
        }
        assert_eq!(idle, required);
        assert_eq!(watermarks.len(), required);
        assert!(watermarks.windows(2).all(|pair| pair[0] < pair[1]));
        assert_eq!(empty_data, required);
    }

    async fn wait_for_single_stress_writes(writes: &Mutex<Vec<(String, u64)>>, expected: usize) {
        for _ in 0..1_000 {
            if writes.lock().len() >= expected {
                return;
            }
            tokio::task::yield_now().await;
        }
        panic!("seeded zero-cost sink did not reach {expected} writes")
    }

    async fn run_seeded_zero_cost_shape_phase(
        seed: u64,
        message_slot_limit: usize,
        multiplier: usize,
    ) {
        let shape_count = message_slot_limit * multiplier;
        let shape_events = zero_cost_lifecycle_events(shape_count);
        assert_zero_cost_shape_multiplier(&shape_events, message_slot_limit, multiplier);

        let source_gate = Arc::new(Semaphore::new(0));
        let sink_gate = Arc::new(Semaphore::new(0));
        let source_closed = Arc::new(AtomicUsize::new(0));
        let sink_closed = Arc::new(AtomicUsize::new(0));
        let writes = Arc::new(Mutex::new(Vec::new()));
        let zero_cost_writes = Arc::new(AtomicUsize::new(0));
        let mut events = shape_events
            .into_iter()
            .map(|event| (Arc::clone(&source_gate), Some(event)))
            .collect::<VecDeque<_>>();
        events.push_back((Arc::clone(&source_gate), None));
        let source = SourceBinding::new(
            Box::new(StressSource {
                events,
                closed: Arc::clone(&source_closed),
            }),
            None,
            0,
        )
        .unwrap();
        let sinks = vec![NamedSinkBinding {
            output_id: "output".into(),
            sink_id: "seeded-zero-cost".into(),
            binding: OrdinarySinkBinding::new(Box::new(StressSink {
                gates: std::iter::repeat_with(|| Arc::clone(&sink_gate))
                    .take(shape_count)
                    .collect(),
                writes: Arc::clone(&writes),
                zero_cost_writes: Arc::clone(&zero_cost_writes),
                closed: Arc::clone(&sink_closed),
            })),
        }];
        let mut runner = ContinuousRunner::new();
        let job = runner
            .start(forward_spec(30_000 + seed, source, sinks))
            .await
            .unwrap_or_else(|failure| {
                panic!("zero-cost unary start failed at seed {seed}: {failure:?}")
            });

        source_gate.add_permits(shape_count * 3 + 1);
        for delivered in 1..=shape_count {
            sink_gate.add_permits(1);
            wait_for_single_stress_writes(&writes, delivered).await;
            assert_eq!(writes.lock().len(), delivered, "seed {seed}");
            assert_zero_cost_active_edges(&job);
        }
        let outcome = job.wait().await;
        assert_eq!(outcome.state, ContinuousJobState::Completed, "seed {seed}");
        assert_eq!(outcome.cause, TerminalCause::NaturalEnd, "seed {seed}");
        assert_eq!(
            zero_cost_writes.load(Ordering::SeqCst),
            shape_count,
            "seed {seed}"
        );
        let status = job.status();
        assert!(status.tasks.is_empty(), "unary task leak at seed {seed}");
        assert!(status.edges.values().all(|edge| {
            edge.queue_depth == 0
                && edge.charged_rows == 0
                && edge.charged_bytes == 0
                && edge.high_water_depth <= message_slot_limit
                && edge.high_water_rows == 0
                && edge.high_water_bytes == 0
        }));
        assert!(
            status
                .edges
                .values()
                .all(|edge| edge.high_water_depth == message_slot_limit)
        );
        assert!(status.edges.values().any(|edge| edge.blocked_sends > 0));
        assert_eq!(source_closed.load(Ordering::SeqCst), 1, "seed {seed}");
        assert_eq!(sink_closed.load(Ordering::SeqCst), 1, "seed {seed}");
        assert_eq!(runner.registry_counts(), (0, 0), "seed {seed}");
        drop(job);
        runner.shutdown().await.unwrap();
        assert_eq!(runner.registry_counts(), (0, 0), "seed {seed}");
    }

    async fn expect_poll_calls(
        poll_calls: &mut mpsc::UnboundedReceiver<usize>,
        expected: impl IntoIterator<Item = usize>,
    ) {
        for expected in expected {
            let observed = tokio::time::timeout(StdDuration::from_secs(5), poll_calls.recv())
                .await
                .expect("zero-cost source did not make the permitted next poll")
                .expect("zero-cost source poll recorder closed early");
            assert_eq!(observed, expected);
        }
    }

    async fn wait_for_source_boundary_backpressure(job: &super::ContinuousJob) {
        for _ in 0..1_000 {
            let status = job.status();
            if status.edges.iter().any(|(edge_id, edge)| {
                edge_id.starts_with("source/")
                    && edge.queue_depth == status.metrics.edges[edge_id].message_slot_limit
                    && edge.blocked_sends > 0
            }) {
                return;
            }
            tokio::task::yield_now().await;
        }
        panic!("unary source boundary did not reach exact slot backpressure")
    }

    async fn assert_no_additional_poll(poll_calls: &mut mpsc::UnboundedReceiver<usize>) {
        let mut next_poll = Box::pin(poll_calls.recv());
        assert!(matches!(futures::poll!(next_poll.as_mut()), Poll::Pending));
    }

    fn assert_exact_poll_and_prefetch_bound(
        job: &super::ContinuousJob,
        polls: &AtomicUsize,
        expected: usize,
    ) {
        assert_eq!(polls.load(Ordering::SeqCst), expected);
        let status = job.status();
        assert_eq!(
            status.metrics.sources["input"].poll_count,
            u64::try_from(expected).unwrap()
        );
        assert!(status.edges.iter().all(|(edge_id, edge)| {
            edge.queue_depth <= status.metrics.edges[edge_id].message_slot_limit
                && edge.charged_rows == 0
                && edge.charged_bytes == 0
        }));
    }

    async fn wait_for_write_count(writes: &Mutex<Vec<u64>>, expected: usize) {
        for _ in 0..1_000 {
            if writes.lock().len() >= expected {
                return;
            }
            tokio::task::yield_now().await;
        }
        panic!("zero-cost sink did not reach {expected} writes")
    }

    async fn finish_zero_cost_lifecycle(
        job: &super::ContinuousJob,
        sink_gate: &Semaphore,
        termination: ZeroCostTermination,
        receiver_failure: &AtomicBool,
        release_permits: usize,
    ) -> Arc<super::ContinuousJobOutcome> {
        match termination {
            ZeroCostTermination::Graceful => {
                let observer = job.shutdown();
                sink_gate.add_permits(1);
                observer.await
            }
            ZeroCostTermination::ReceiverClose => {
                receiver_failure.store(true, Ordering::SeqCst);
                sink_gate.add_permits(release_permits);
                job.wait().await
            }
            ZeroCostTermination::Cancel => job.cancel().await,
            ZeroCostTermination::Error => {
                sink_gate.add_permits(release_permits);
                job.wait().await
            }
        }
    }

    fn assert_zero_cost_terminal(
        termination: ZeroCostTermination,
        outcome: &super::ContinuousJobOutcome,
    ) {
        match termination {
            ZeroCostTermination::Graceful => {
                assert_eq!(outcome.state, ContinuousJobState::Completed);
                assert_eq!(outcome.cause, TerminalCause::GracefulShutdown);
            }
            ZeroCostTermination::Cancel => {
                assert_eq!(outcome.state, ContinuousJobState::Cancelled);
                assert_eq!(outcome.cause, TerminalCause::ExplicitCancel);
            }
            ZeroCostTermination::ReceiverClose => {
                assert_eq!(outcome.state, ContinuousJobState::Failed);
                assert!(matches!(outcome.cause, TerminalCause::TaskFailure { .. }));
                assert!(matches!(
                    outcome.errors.first().map(|failure| (&failure.origin, &failure.error)),
                    Some((
                        super::FailureOrigin::Task { task_name, .. },
                        CalcFlowError::Operator { node_id, message }
                    )) if task_name == "operator:node"
                        && node_id == "node"
                        && message == "zero-cost lifecycle receiver failed"
                ));
            }
            ZeroCostTermination::Error => {
                assert_eq!(outcome.state, ContinuousJobState::Failed);
                assert!(matches!(outcome.cause, TerminalCause::TaskFailure { .. }));
                assert!(matches!(
                    outcome
                        .errors
                        .first()
                        .map(|failure| (&failure.origin, &failure.error)),
                    Some((
                        super::FailureOrigin::Task { task_name, .. },
                        CalcFlowError::Internal { message }
                    )) if task_name == "source:input:pump"
                        && message == "zero-cost lifecycle source failed"
                ));
            }
        }
    }

    fn assert_zero_cost_convergence(job: &super::ContinuousJob, termination: ZeroCostTermination) {
        let status = job.status();
        assert!(status.tasks.is_empty(), "task leak: {termination:?}");
        assert!(status.edges.values().all(|edge| {
            edge.queue_depth == 0
                && edge.charged_rows == 0
                && edge.charged_bytes == 0
                && edge.high_water_depth <= 1
                && edge.high_water_rows == 0
                && edge.high_water_bytes == 0
        }));
        assert!(
            status.edges.values().any(|edge| edge.blocked_sends > 0),
            "zero-cost messages never exercised backpressure: {termination:?}"
        );
    }

    async fn run_zero_cost_lifecycle_case(termination: ZeroCostTermination) {
        const CYCLES: usize = 10;
        const CREDIT_RETURNS: usize = 3;
        const INITIAL_POLL_BOUND: usize = 6;
        const POLLS_PER_CREDIT: usize = 3;

        let polls = Arc::new(AtomicUsize::new(0));
        let (poll_call_tx, mut poll_calls) = mpsc::unbounded_channel();
        let eof_observed = Arc::new(AtomicUsize::new(0));
        let source_closed = Arc::new(AtomicUsize::new(0));
        let sink_closed = Arc::new(AtomicUsize::new(0));
        let sink_gate = Arc::new(Semaphore::new(0));
        let writes = Arc::new(Mutex::new(Vec::new()));
        let receiver_failure = Arc::new(AtomicBool::new(false));
        let source = SourceBinding::new(
            Box::new(ZeroCostLifecycleSource {
                events: zero_cost_lifecycle_events(CYCLES),
                polls: Arc::clone(&polls),
                poll_calls: poll_call_tx,
                eof_observed: Arc::clone(&eof_observed),
                closed: Arc::clone(&source_closed),
                end: match termination {
                    ZeroCostTermination::Graceful => ZeroCostSourceEnd::Eof,
                    ZeroCostTermination::Error => ZeroCostSourceEnd::Error,
                    ZeroCostTermination::ReceiverClose | ZeroCostTermination::Cancel => {
                        ZeroCostSourceEnd::Pending
                    }
                },
            }),
            None,
            0,
        )
        .unwrap();
        let sinks = vec![NamedSinkBinding {
            output_id: "output".into(),
            sink_id: "zero-cost-sink".into(),
            binding: OrdinarySinkBinding::new(Box::new(ZeroCostLifecycleSink {
                gate: Arc::clone(&sink_gate),
                writes: Arc::clone(&writes),
                closed: Arc::clone(&sink_closed),
            })),
        }];
        let mut runner = ContinuousRunner::new();
        let operator: Box<dyn StreamOperator> =
            if matches!(termination, ZeroCostTermination::ReceiverClose) {
                Box::new(StressForwardOperator::failing_on_watermark(Arc::clone(
                    &receiver_failure,
                )))
            } else {
                Box::new(StressForwardOperator::new(None))
            };
        let job = runner
            .start(forward_spec_with_operator(20_000, source, sinks, operator))
            .await
            .unwrap();

        expect_poll_calls(&mut poll_calls, 1..=INITIAL_POLL_BOUND).await;
        wait_for_source_boundary_backpressure(&job).await;
        assert_no_additional_poll(&mut poll_calls).await;
        assert_exact_poll_and_prefetch_bound(&job, &polls, INITIAL_POLL_BOUND);
        let mut expected_polls = INITIAL_POLL_BOUND;
        for delivered in 1..=CREDIT_RETURNS {
            sink_gate.add_permits(1);
            let next_expected = expected_polls + POLLS_PER_CREDIT;
            expect_poll_calls(&mut poll_calls, expected_polls + 1..=next_expected).await;
            wait_for_write_count(&writes, delivered).await;
            wait_for_source_boundary_backpressure(&job).await;
            assert_no_additional_poll(&mut poll_calls).await;
            assert_exact_poll_and_prefetch_bound(&job, &polls, next_expected);
            expected_polls = next_expected;
        }

        if matches!(termination, ZeroCostTermination::Graceful) {
            sink_gate.add_permits(CYCLES - CREDIT_RETURNS - 1);
            expect_poll_calls(&mut poll_calls, expected_polls + 1..=CYCLES * 3 + 1).await;
            wait_for_write_count(&writes, CYCLES - 1).await;
            assert_eq!(eof_observed.load(Ordering::SeqCst), 1);
            assert_eq!(job.status().state, ContinuousJobState::Running);
        }

        let outcome =
            finish_zero_cost_lifecycle(&job, &sink_gate, termination, &receiver_failure, CYCLES)
                .await;
        assert_zero_cost_terminal(termination, &outcome);
        assert_zero_cost_convergence(&job, termination);
        let observed = writes.lock().clone();
        assert!(observed.len() >= CREDIT_RETURNS);
        if matches!(termination, ZeroCostTermination::Graceful) {
            assert_eq!(observed.len(), CYCLES);
            assert_eq!(polls.load(Ordering::SeqCst), CYCLES * 3 + 1);
            assert!(job.status().sources["input"].ended);
        }
        assert_eq!(
            observed,
            (0..u64::try_from(observed.len()).unwrap()).collect::<Vec<_>>()
        );
        assert_eq!(source_closed.load(Ordering::SeqCst), 1);
        assert_eq!(sink_closed.load(Ordering::SeqCst), 1);
        assert_eq!(runner.registry_counts(), (0, 0));
        drop(job);
        runner.shutdown().await.unwrap();
        assert_eq!(runner.registry_counts(), (0, 0));
    }

    #[tokio::test]
    async fn real_unary_zero_cost_lifecycle_matrix_returns_credits_and_converges() {
        for termination in [
            ZeroCostTermination::Graceful,
            ZeroCostTermination::ReceiverClose,
            ZeroCostTermination::Cancel,
            ZeroCostTermination::Error,
        ] {
            run_zero_cost_lifecycle_case(termination).await;
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
        assert_eq!(core.runtime_status.lock().tasks.snapshot().len(), 6);
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
            super::StatusProjection::default(),
            false,
            "test".into(),
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
            stop_after_first_job: false,
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
            next_launch_probe: Mutex::new(None),
            panic_lifecycle_after_shutdown: AtomicBool::new(false),
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

    #[tokio::test(start_paused = true)]
    async fn dropped_start_pending_close_expires_and_reaper_allows_next_start() {
        let source = LifecycleProbe::default();
        source.block_open.store(true, Ordering::SeqCst);
        source.block_close.store(true, Ordering::SeqCst);
        let sink = LifecycleProbe::default();
        let mut runner = ContinuousRunner::new();
        let observer = runner.start(spec(
            false,
            Arc::new(AtomicUsize::new(0)),
            source.clone(),
            sink.clone(),
        ));
        let provisional = Arc::clone(observer.core.as_ref().unwrap());
        wait_for_counter(&source.opened, 1).await;
        wait_for_counter(&sink.open_completed, 1).await;

        drop(observer);

        assert_eq!(provisional.state.lock().owner, DriverOwnership::ReaperOwned);
        wait_for_counter(&source.closed, 1).await;
        let mut next = Box::pin(runner.start(spec(
            false,
            Arc::new(AtomicUsize::new(0)),
            LifecycleProbe::default(),
            LifecycleProbe::default(),
        )));
        assert!(matches!(futures::poll!(next.as_mut()), Poll::Pending));

        tokio::time::advance(StdDuration::from_secs(5)).await;
        let next = tokio::time::timeout(StdDuration::from_secs(1), next)
            .await
            .expect("a bounded cancelled-launch close must release the reaper")
            .unwrap();

        assert_eq!(provisional.state.lock().owner, DriverOwnership::Terminal);
        assert_eq!(source.closed.load(Ordering::SeqCst), 1);
        assert_eq!(sink.closed.load(Ordering::SeqCst), 1);
        let diagnostic = runner
            .diagnostics()
            .records
            .into_iter()
            .find(|record| record.launch_id == provisional.launch_id)
            .expect("dropped start cleanup must retain its timeout diagnostic");
        assert_eq!(diagnostic.cleanup_failures.len(), 1);
        assert!(matches!(
            &diagnostic.cleanup_failures[0].origin,
            super::FailureOrigin::SourceClose { binding_id } if binding_id == "input"
        ));
        assert!(matches!(
            &diagnostic.cleanup_failures[0].error,
            CalcFlowError::Internal { message }
                if message == "connector close exceeded private teardown bound of 5 seconds"
        ));
        {
            let runtime = provisional.runtime_status.lock();
            assert!(runtime.tasks.snapshot().is_empty());
        }
        assert!(provisional.metrics.snapshot().edges.values().all(|edge| {
            edge.channel.queue_depth == 0
                && edge.channel.charged_rows == 0
                && edge.channel.charged_bytes == 0
        }));

        drop(next);
        runner.shutdown().await.unwrap();
        assert_eq!(runner.registry_counts(), (0, 0));
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn dropping_start_during_operator_entry_reaps_before_connector_lifecycle() {
        let entered = Arc::new(AtomicBool::new(false));
        let release = Arc::new(AtomicBool::new(false));
        let source = LifecycleProbe::default();
        let sink = LifecycleProbe::default();
        let mut runner = ContinuousRunner::new();
        let observer = runner.start(blocking_entry_spec(
            &entered,
            &release,
            source.clone(),
            sink.clone(),
        ));
        let provisional = Arc::clone(observer.core.as_ref().unwrap());
        let entry_observed = wait_for_operator_entry(&entered).await;
        if !entry_observed {
            release.store(true, Ordering::SeqCst);
        }
        assert!(entry_observed, "operator entry did not begin");

        drop(observer);
        assert!(provisional.launch_cancel.is_cancelled());
        assert_eq!(provisional.state.lock().owner, DriverOwnership::ReaperOwned);
        release.store(true, Ordering::SeqCst);

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
        assert_eq!(source.closed.load(Ordering::SeqCst), 0);
        assert_eq!(sink.opened.load(Ordering::SeqCst), 0);
        assert_eq!(sink.closed.load(Ordering::SeqCst), 0);
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
    async fn dropping_start_after_operator_entry_before_live_publication_is_reaped() {
        let resets = Arc::new(AtomicUsize::new(0));
        let source = LifecycleProbe::default();
        let sink = LifecycleProbe::default();
        let probe = Arc::new(super::TestLaunchProbe::new(
            super::TestLaunchCheckpoint::AfterOperatorEntry,
        ));
        let mut runner = ContinuousRunner::new();
        let observer = runner.start_with_test_launch_probe(
            spec(false, Arc::clone(&resets), source.clone(), sink.clone()),
            Arc::clone(&probe),
        );
        let provisional = Arc::clone(observer.core.as_ref().unwrap());

        probe.wait_until_reached().await;
        assert_eq!(resets.load(Ordering::SeqCst), 1);
        assert_eq!(
            provisional.state.lock().launch_delivery,
            super::LaunchDeliveryState::Provisional
        );
        assert_eq!(source.opened.load(Ordering::SeqCst), 0);
        assert_eq!(sink.opened.load(Ordering::SeqCst), 0);

        drop(observer);
        assert!(provisional.launch_cancel.is_cancelled());
        assert_eq!(provisional.state.lock().owner, DriverOwnership::ReaperOwned);
        probe.release();

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
        assert_eq!(source.closed.load(Ordering::SeqCst), 0);
        assert_eq!(sink.closed.load(Ordering::SeqCst), 0);
        drop(next);
        runner.shutdown().await.unwrap();
        assert_eq!(runner.registry_counts(), (0, 0));
    }

    #[tokio::test]
    async fn dropping_start_after_live_publication_before_handle_delivery_is_reaped() {
        let plan = unary_expression_plan();
        let polls = Arc::new(AtomicUsize::new(0));
        let source_closed = Arc::new(AtomicUsize::new(0));
        let sink = LifecycleProbe::default();
        let job_spec = ContinuousJobSpec {
            context: StreamJobContext::new(
                90,
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
        let probe = Arc::new(super::TestLaunchProbe::new(
            super::TestLaunchCheckpoint::LivePublished,
        ));
        let observer = runner.start_with_test_launch_probe(job_spec, Arc::clone(&probe));
        let provisional = Arc::clone(observer.core.as_ref().unwrap());
        probe.wait_until_reached().await;
        assert_eq!(
            provisional.state.lock().launch_delivery,
            super::LaunchDeliveryState::ReadyUnclaimed
        );
        assert_eq!(sink.open_completed.load(Ordering::SeqCst), 1);

        drop(observer);
        assert!(provisional.launch_cancel.is_cancelled());
        assert_eq!(provisional.state.lock().owner, DriverOwnership::ReaperOwned);
        assert_eq!(polls.load(Ordering::SeqCst), 0);
        probe.release();

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
        assert_eq!(source_closed.load(Ordering::SeqCst), 1);
        assert_eq!(sink.closed.load(Ordering::SeqCst), 1);
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

    #[tokio::test(start_paused = true)]
    async fn failed_launch_bounds_close_and_keeps_stable_cleanup_diagnostics() {
        let source = LifecycleProbe::default();
        source.block_open.store(true, Ordering::SeqCst);
        source.fail_open.store(true, Ordering::SeqCst);
        source.block_close.store(true, Ordering::SeqCst);
        let panic_sink = LifecycleProbe::default();
        panic_sink.panic_close.store(true, Ordering::SeqCst);
        let error_sink = LifecycleProbe::default();
        error_sink.fail_close.store(true, Ordering::SeqCst);
        let later_sink = LifecycleProbe::default();
        let job_spec = forward_spec(
            101,
            SourceBinding::new(Box::new(ProbeSource(source.clone())), None, 0).unwrap(),
            vec![
                named_probe_sink("a-panic", panic_sink.clone()),
                named_probe_sink("b-error", error_sink.clone()),
                named_probe_sink("c-later", later_sink.clone()),
            ],
        );
        let mut runner = ContinuousRunner::new();
        let observer = runner.start(job_spec);
        let provisional = Arc::clone(observer.core.as_ref().unwrap());
        wait_for_counter(&source.opened, 1).await;
        wait_for_counter(&panic_sink.open_completed, 1).await;
        wait_for_counter(&error_sink.open_completed, 1).await;
        wait_for_counter(&later_sink.open_completed, 1).await;
        source.open_release.notify_waiters();
        wait_for_counter(&source.closed, 1).await;
        assert_eq!(panic_sink.closed.load(Ordering::SeqCst), 0);
        let shutdown = runner.shutdown();

        tokio::time::advance(StdDuration::from_secs(5)).await;
        let failure = tokio::time::timeout(StdDuration::from_secs(1), observer)
            .await
            .expect("failed launch must outlive a permanently pending close")
            .unwrap_err();
        shutdown.await.unwrap();

        assert!(matches!(
            &failure.primary.origin,
            super::FailureOrigin::SourceOpen { binding_id } if binding_id == "input"
        ));
        assert!(matches!(
            &failure.primary.error,
            CalcFlowError::Internal { message } if message == "source open failed"
        ));
        let diagnostic_id = failure
            .diagnostic_id
            .expect("bounded close failures must be retained as secondaries");
        let diagnostics = runner.diagnostics();
        let cleanup = &diagnostics
            .records
            .iter()
            .find(|record| record.id == diagnostic_id)
            .unwrap()
            .cleanup_failures;
        assert_eq!(cleanup.len(), 3);
        assert!(matches!(
            (&cleanup[0].origin, &cleanup[0].error),
            (
                super::FailureOrigin::SourceClose { binding_id },
                CalcFlowError::Internal { message }
            ) if binding_id == "input"
                && message == "connector close exceeded private teardown bound of 5 seconds"
        ));
        assert!(matches!(
            (&cleanup[1].origin, &cleanup[1].error),
            (
                super::FailureOrigin::SinkClose { output_id, sink_id },
                CalcFlowError::TaskPanicked { task_id: 1, message }
            ) if output_id == "output" && sink_id == "a-panic"
                && message == "sink close panicked"
        ));
        assert!(matches!(
            (&cleanup[2].origin, &cleanup[2].error),
            (
                super::FailureOrigin::SinkClose { output_id, sink_id },
                CalcFlowError::Internal { message }
            ) if output_id == "output" && sink_id == "b-error"
                && message == "sink close failed"
        ));
        assert_eq!(source.closed.load(Ordering::SeqCst), 1);
        assert_eq!(panic_sink.closed.load(Ordering::SeqCst), 1);
        assert_eq!(error_sink.closed.load(Ordering::SeqCst), 1);
        assert_eq!(later_sink.closed.load(Ordering::SeqCst), 1);
        assert_eq!(provisional.state.lock().owner, DriverOwnership::Terminal);
        assert!(
            provisional
                .runtime_status
                .lock()
                .tasks
                .snapshot()
                .is_empty()
        );
        assert!(provisional.metrics.snapshot().edges.values().all(|edge| {
            edge.channel.queue_depth == 0
                && edge.channel.charged_rows == 0
                && edge.channel.charged_bytes == 0
        }));
        assert_eq!(runner.registry_counts(), (0, 0));
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
        let left_resume = Cursor::new("left", vec![1], JsonMap::new()).unwrap();
        let right_resume = Cursor::new("right", vec![2], JsonMap::new()).unwrap();
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

    #[tokio::test]
    async fn checkpointed_runner_opens_an_empty_lineage_only_after_preflight() {
        let directory = tempfile::tempdir().unwrap();
        let backend = Arc::new(
            LocalStateBackend::new(directory.path().join("state"))
                .await
                .unwrap(),
        );
        let plan = PipelineBuilder::new("checkpoint-empty")
            .unwrap()
            .add_checkpoint_capable_node(
                "node",
                Box::new(StressForwardOperator::new(None)) as Box<dyn StreamOperator>,
            )
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
        let source_polls = Arc::new(AtomicUsize::new(0));
        let source_closed = Arc::new(AtomicUsize::new(0));
        let sink_opened = Arc::new(AtomicUsize::new(0));
        let sink_closed = Arc::new(AtomicUsize::new(0));
        let spec = ContinuousJobSpec {
            context: StreamJobContext::new(
                901,
                plan.fingerprint(),
                JsonMap::new(),
                None,
                CancellationToken::new(),
            ),
            plan,
            sources: vec![NamedSourceBinding {
                binding_id: "input".into(),
                binding: counting_pending_binding(0, &source_polls, &source_closed),
            }],
            sinks: vec![NamedSinkBinding {
                output_id: "output".into(),
                sink_id: "sink".into(),
                binding: OrdinarySinkBinding::new_transactional(Box::new(CheckpointProbeSink {
                    opened: Arc::clone(&sink_opened),
                    closed: Arc::clone(&sink_closed),
                })),
            }],
            edge_budget: EdgeBudget {
                max_rows: 1,
                max_bytes: 1 << 20,
            },
            delivery_mode: M2DeliveryMode::ProcessLocalOrdered,
        };
        let checkpoint = CheckpointRuntimeSpec::new(
            backend.clone(),
            directory.path().join("manifests"),
            StreamRuntimeConfig::default(),
        )
        .unwrap();
        let mut runner = ContinuousRunner::new();

        let job = runner.start_checkpointed(spec, checkpoint).await.unwrap();

        assert_eq!(sink_opened.load(Ordering::SeqCst), 1);
        wait_for_counter(&source_polls, 1).await;
        let outcome = job.cancel().await;
        assert_eq!(outcome.state, ContinuousJobState::Cancelled);
        drop(job);
        runner.shutdown().await.unwrap();
        assert_eq!(source_closed.load(Ordering::SeqCst), 1);
        assert_eq!(sink_closed.load(Ordering::SeqCst), 1);
        assert!(directory.path().join("manifests").is_dir());
    }

    #[tokio::test]
    #[allow(
        clippy::too_many_lines,
        reason = "the no-manifest recovery oracle owns the orphan, retry, and lifecycle assertions"
    )]
    async fn checkpointed_runner_cleans_pre_manifest_state_before_retry() {
        let directory = tempfile::tempdir().unwrap();
        let backend = Arc::new(
            LocalStateBackend::new(directory.path().join("state"))
                .await
                .unwrap(),
        );
        let plan = PipelineBuilder::new("checkpoint-pre-manifest-cleanup")
            .unwrap()
            .add_checkpoint_capable_node(
                "node",
                Box::new(StressForwardOperator::new(None)) as Box<dyn StreamOperator>,
            )
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
        let lineage_key = StateLineageKey::new(plan.name(), plan.fingerprint()).unwrap();
        let orphan_bytes = b"published-before-manifest";
        let abandoned_bytes = b"staged-before-manifest";
        let retry_bytes = b"retry-at-same-coordinate";
        let orphan = local_state_handle(
            &lineage_key,
            "node",
            crate::Epoch::INITIAL,
            "published",
            orphan_bytes,
        );
        let orphan_retry = local_state_handle(
            &lineage_key,
            "node",
            crate::Epoch::INITIAL,
            "published",
            retry_bytes,
        );
        let abandoned = local_state_handle(
            &lineage_key,
            "node",
            crate::Epoch::INITIAL,
            "staged",
            abandoned_bytes,
        );
        let abandoned_retry = local_state_handle(
            &lineage_key,
            "node",
            crate::Epoch::INITIAL,
            "staged",
            retry_bytes,
        );
        {
            let lineage = backend.open_lineage(&lineage_key).await.unwrap();
            lineage.stage_segment(&orphan, orphan_bytes).await.unwrap();
            lineage.validate_segment(&orphan).await.unwrap();
            lineage.publish_segment(&orphan).await.unwrap();
            lineage
                .stage_segment(&abandoned, abandoned_bytes)
                .await
                .unwrap();
        }
        let source_polls = Arc::new(AtomicUsize::new(0));
        let source_closed = Arc::new(AtomicUsize::new(0));
        let sink_opened = Arc::new(AtomicUsize::new(0));
        let sink_closed = Arc::new(AtomicUsize::new(0));
        let spec = ContinuousJobSpec {
            context: StreamJobContext::new(
                902,
                plan.fingerprint(),
                JsonMap::new(),
                None,
                CancellationToken::new(),
            ),
            plan,
            sources: vec![NamedSourceBinding {
                binding_id: "input".into(),
                binding: counting_pending_binding(0, &source_polls, &source_closed),
            }],
            sinks: vec![NamedSinkBinding {
                output_id: "output".into(),
                sink_id: "sink".into(),
                binding: OrdinarySinkBinding::new_transactional(Box::new(CheckpointProbeSink {
                    opened: Arc::clone(&sink_opened),
                    closed: Arc::clone(&sink_closed),
                })),
            }],
            edge_budget: EdgeBudget {
                max_rows: 1,
                max_bytes: 1 << 20,
            },
            delivery_mode: M2DeliveryMode::ProcessLocalOrdered,
        };
        let checkpoint = CheckpointRuntimeSpec::new(
            backend.clone(),
            directory.path().join("manifests"),
            StreamRuntimeConfig {
                checkpoint_interval: StdDuration::from_secs(3_600),
                ..StreamRuntimeConfig::default()
            },
        )
        .unwrap();
        let mut runner = ContinuousRunner::new();

        let job = runner.start_checkpointed(spec, checkpoint).await.unwrap();

        wait_for_counter(&source_polls, 1).await;
        assert_eq!(job.status().metrics.checkpoints.orphan_segments_removed, 1);
        assert_eq!(job.status().checkpoint.unwrap().last_completed_epoch, None);
        let outcome = job.cancel().await;
        assert_eq!(outcome.state, ContinuousJobState::Cancelled);
        drop(job);
        runner.shutdown().await.unwrap();
        assert_eq!(runner.registry_counts(), (0, 0));
        assert_eq!(source_closed.load(Ordering::SeqCst), 1);
        assert_eq!(sink_opened.load(Ordering::SeqCst), 1);
        assert_eq!(sink_closed.load(Ordering::SeqCst), 1);

        let lineage = backend.open_lineage(&lineage_key).await.unwrap();
        assert!(matches!(
            lineage.load_segment(&orphan).await,
            Err(CalcFlowError::NotFound { .. })
        ));
        lineage
            .stage_segment(&orphan_retry, retry_bytes)
            .await
            .unwrap();
        lineage.validate_segment(&orphan_retry).await.unwrap();
        lineage.publish_segment(&orphan_retry).await.unwrap();
        lineage
            .stage_segment(&abandoned_retry, retry_bytes)
            .await
            .unwrap();
        lineage.validate_segment(&abandoned_retry).await.unwrap();
        lineage.publish_segment(&abandoned_retry).await.unwrap();
        assert_eq!(
            lineage.load_segment(&orphan_retry).await.unwrap(),
            retry_bytes
        );
        assert_eq!(
            lineage.load_segment(&abandoned_retry).await.unwrap(),
            retry_bytes
        );
    }

    #[tokio::test]
    async fn exactly_once_start_without_checkpoint_fails_before_lifecycle_work() {
        let plan = PipelineBuilder::new("checkpoint-required")
            .unwrap()
            .add_checkpoint_capable_node(
                "node",
                Box::new(StressForwardOperator::new(None)) as Box<dyn StreamOperator>,
            )
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
        let source_polls = Arc::new(AtomicUsize::new(0));
        let source_closed = Arc::new(AtomicUsize::new(0));
        let sink_opened = Arc::new(AtomicUsize::new(0));
        let sink_closed = Arc::new(AtomicUsize::new(0));
        let spec = ContinuousJobSpec {
            context: StreamJobContext::new(
                906,
                plan.fingerprint(),
                JsonMap::new(),
                None,
                CancellationToken::new(),
            ),
            plan,
            sources: vec![NamedSourceBinding {
                binding_id: "input".into(),
                binding: counting_pending_binding(0, &source_polls, &source_closed),
            }],
            sinks: vec![NamedSinkBinding {
                output_id: "output".into(),
                sink_id: "sink".into(),
                binding: OrdinarySinkBinding::new_transactional(Box::new(CheckpointProbeSink {
                    opened: Arc::clone(&sink_opened),
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

        let failure = match runner.start(spec).await {
            Ok(_) => panic!("exactly-once start unexpectedly ran without checkpoint ownership"),
            Err(failure) => failure,
        };

        assert!(matches!(
            &failure.primary.error,
            CalcFlowError::InvalidArgument { field, message }
                if field == "requirements.delivery.output"
                    && message.contains("checkpoint runtime")
        ));
        assert_eq!(source_polls.load(Ordering::SeqCst), 0);
        assert_eq!(source_closed.load(Ordering::SeqCst), 0);
        assert_eq!(sink_opened.load(Ordering::SeqCst), 0);
        assert_eq!(sink_closed.load(Ordering::SeqCst), 0);
        runner.shutdown().await.unwrap();
    }

    #[tokio::test]
    #[allow(
        clippy::too_many_lines,
        reason = "the collision preflight assertion owns both output bindings and lifecycle probes"
    )]
    async fn checkpointed_start_rejects_cross_output_sink_id_collisions_before_open() {
        let directory = tempfile::tempdir().unwrap();
        let backend = Arc::new(
            LocalStateBackend::new(directory.path().join("state"))
                .await
                .unwrap(),
        );
        let plan = PipelineBuilder::new("checkpoint-sink-collision")
            .unwrap()
            .add_checkpoint_capable_node(
                "root",
                Box::new(StressForwardOperator::new(None)) as Box<dyn StreamOperator>,
            )
            .unwrap()
            .add_checkpoint_capable_node(
                "branch_a",
                Box::new(StressForwardOperator::new(None)) as Box<dyn StreamOperator>,
            )
            .unwrap()
            .add_checkpoint_capable_node(
                "branch_b",
                Box::new(StressForwardOperator::new(None)) as Box<dyn StreamOperator>,
            )
            .unwrap()
            .connect(Edge::new(
                PortEndpoint::new("root", "output").unwrap(),
                PortEndpoint::new("branch_a", "input").unwrap(),
            ))
            .unwrap()
            .connect(Edge::new(
                PortEndpoint::new("root", "output").unwrap(),
                PortEndpoint::new("branch_b", "input").unwrap(),
            ))
            .unwrap()
            .compile_stream(
                &UdfRegistry::new().snapshot(),
                &StreamRequirements {
                    delivery: BTreeMap::from([
                        (
                            "branch_a.output".into(),
                            crate::DeliveryGuarantee::ExactlyOnce,
                        ),
                        (
                            "branch_b.output".into(),
                            crate::DeliveryGuarantee::ExactlyOnce,
                        ),
                    ]),
                },
            )
            .unwrap();
        let source_polls = Arc::new(AtomicUsize::new(0));
        let source_closed = Arc::new(AtomicUsize::new(0));
        let sink_opened = Arc::new(AtomicUsize::new(0));
        let sink_closed = Arc::new(AtomicUsize::new(0));
        let spec = ContinuousJobSpec {
            context: StreamJobContext::new(
                907,
                plan.fingerprint(),
                JsonMap::new(),
                None,
                CancellationToken::new(),
            ),
            plan,
            sources: vec![NamedSourceBinding {
                binding_id: "input".into(),
                binding: counting_pending_binding(0, &source_polls, &source_closed),
            }],
            sinks: ["branch_a.output", "branch_b.output"]
                .into_iter()
                .map(|output_id| NamedSinkBinding {
                    output_id: output_id.into(),
                    sink_id: "duplicate".into(),
                    binding: OrdinarySinkBinding::new_transactional(Box::new(
                        CheckpointProbeSink {
                            opened: Arc::clone(&sink_opened),
                            closed: Arc::clone(&sink_closed),
                        },
                    )),
                })
                .collect(),
            edge_budget: EdgeBudget {
                max_rows: 1,
                max_bytes: 1 << 20,
            },
            delivery_mode: M2DeliveryMode::ProcessLocalOrdered,
        };
        let checkpoint = CheckpointRuntimeSpec::new(
            backend,
            directory.path().join("manifests"),
            StreamRuntimeConfig::default(),
        )
        .unwrap();
        let mut runner = ContinuousRunner::new();

        let failure = match runner.start_checkpointed(spec, checkpoint).await {
            Ok(_) => panic!("duplicate sink IDs unexpectedly entered lifecycle work"),
            Err(failure) => failure,
        };

        assert!(
            matches!(
                &failure.primary.error,
                CalcFlowError::InvalidArgument { field, message }
                    if field == "sinks.duplicate" && message.contains("more than one output")
            ),
            "unexpected preflight error: {:?}",
            failure.primary.error
        );
        assert_eq!(source_polls.load(Ordering::SeqCst), 0);
        assert_eq!(source_closed.load(Ordering::SeqCst), 0);
        assert_eq!(sink_opened.load(Ordering::SeqCst), 0);
        assert_eq!(sink_closed.load(Ordering::SeqCst), 0);
        assert!(!directory.path().join("manifests").exists());
        runner.shutdown().await.unwrap();
    }

    #[tokio::test(start_paused = true)]
    #[allow(
        clippy::too_many_lines,
        reason = "the periodic lifecycle and bounded status snapshot are one integration contract"
    )]
    async fn checkpointed_runner_periodically_publishes_before_sink_commit() {
        let directory = tempfile::tempdir().unwrap();
        let backend = Arc::new(
            LocalStateBackend::new(directory.path().join("state"))
                .await
                .unwrap(),
        );
        let plan = PipelineBuilder::new("checkpoint-periodic")
            .unwrap()
            .add_checkpoint_capable_node(
                "node",
                Box::new(StressForwardOperator::new(None)) as Box<dyn StreamOperator>,
            )
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
        let source_polls = Arc::new(AtomicUsize::new(0));
        let source_closed = Arc::new(AtomicUsize::new(0));
        let sink_closed = Arc::new(AtomicUsize::new(0));
        let log = Arc::new(Mutex::new(Vec::new()));
        let spec = ContinuousJobSpec {
            context: StreamJobContext::new(
                903,
                plan.fingerprint(),
                JsonMap::new(),
                None,
                CancellationToken::new(),
            ),
            plan,
            sources: vec![NamedSourceBinding {
                binding_id: "input".into(),
                binding: counting_pending_binding(0, &source_polls, &source_closed),
            }],
            sinks: vec![NamedSinkBinding {
                output_id: "output".into(),
                sink_id: "sink".into(),
                binding: OrdinarySinkBinding::new_transactional(Box::new(PeriodicCheckpointSink {
                    log: Arc::clone(&log),
                    closed: Arc::clone(&sink_closed),
                })),
            }],
            edge_budget: EdgeBudget {
                max_rows: 1,
                max_bytes: 1 << 20,
            },
            delivery_mode: M2DeliveryMode::ProcessLocalOrdered,
        };
        let checkpoint = CheckpointRuntimeSpec::new(
            backend,
            directory.path().join("manifests"),
            StreamRuntimeConfig {
                checkpoint_interval: StdDuration::from_millis(10),
                checkpoint_timeout: StdDuration::from_secs(10),
                ..StreamRuntimeConfig::default()
            },
        )
        .unwrap();
        let mut runner = ContinuousRunner::new();
        let job = runner.start_checkpointed(spec, checkpoint).await.unwrap();
        wait_for_counter(&source_polls, 1).await;

        tokio::time::timeout(StdDuration::from_secs(5), async {
            loop {
                if log.lock().iter().any(|entry| entry == "sink-begin:2") {
                    break;
                }
                tokio::time::sleep(StdDuration::from_millis(1)).await;
            }
        })
        .await
        .expect("periodic checkpoint should complete");

        assert_eq!(
            &*log.lock(),
            &[
                "sink-open",
                "sink-begin:1",
                "sink-precommit:1",
                "sink-commit:1",
                "sink-begin:2",
            ]
        );
        let checkpoint_status = job
            .status()
            .checkpoint
            .expect("checkpointed jobs expose bounded checkpoint status");
        assert_eq!(checkpoint_status.current_epoch, None);
        assert_eq!(
            checkpoint_status.last_completed_epoch,
            Some(crate::Epoch::INITIAL)
        );
        assert_eq!(checkpoint_status.failure_category, None);
        assert!(!checkpoint_status.runtime_config_changed);
        assert_eq!(checkpoint_status.expected_sources, 1);
        assert_eq!(checkpoint_status.expected_operators, 1);
        assert_eq!(checkpoint_status.expected_sinks, 1);
        let checkpoint_metrics = job.status().metrics.checkpoints;
        assert_eq!(checkpoint_metrics.requested, 1);
        assert_eq!(checkpoint_metrics.completed, 1);
        assert_eq!(checkpoint_metrics.failed, 0);
        assert_eq!(checkpoint_metrics.terminal_completed, 0);
        assert!(checkpoint_metrics.manifest_bytes > 0);
        assert!(
            checkpoint_metrics
                .phase_duration
                .contains_key(&CheckpointPhase::ManifestDurable)
        );
        let outcome = job.cancel().await;
        assert_eq!(outcome.state, ContinuousJobState::Cancelled);
        drop(job);
        runner.shutdown().await.unwrap();
        assert_eq!(source_closed.load(Ordering::SeqCst), 1);
        assert_eq!(sink_closed.load(Ordering::SeqCst), 1);
    }

    #[tokio::test]
    async fn manual_checkpoint_returns_only_after_durable_manifest_and_sink_commit() {
        let directory = tempfile::tempdir().unwrap();
        let backend = Arc::new(
            LocalStateBackend::new(directory.path().join("state"))
                .await
                .unwrap(),
        );
        let plan = PipelineBuilder::new("checkpoint-manual")
            .unwrap()
            .add_checkpoint_capable_node(
                "node",
                Box::new(StressForwardOperator::new(None)) as Box<dyn StreamOperator>,
            )
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
        let source_polls = Arc::new(AtomicUsize::new(0));
        let source_closed = Arc::new(AtomicUsize::new(0));
        let sink_closed = Arc::new(AtomicUsize::new(0));
        let log = Arc::new(Mutex::new(Vec::new()));
        let spec = ContinuousJobSpec {
            context: StreamJobContext::new(
                913,
                plan.fingerprint(),
                JsonMap::new(),
                None,
                CancellationToken::new(),
            ),
            plan,
            sources: vec![NamedSourceBinding {
                binding_id: "input".into(),
                binding: counting_pending_binding(0, &source_polls, &source_closed),
            }],
            sinks: vec![NamedSinkBinding {
                output_id: "output".into(),
                sink_id: "sink".into(),
                binding: OrdinarySinkBinding::new_transactional(Box::new(PeriodicCheckpointSink {
                    log: Arc::clone(&log),
                    closed: Arc::clone(&sink_closed),
                })),
            }],
            edge_budget: EdgeBudget {
                max_rows: 1,
                max_bytes: 1 << 20,
            },
            delivery_mode: M2DeliveryMode::ProcessLocalOrdered,
        };
        let checkpoint = CheckpointRuntimeSpec::new(
            backend,
            directory.path().join("manifests"),
            StreamRuntimeConfig {
                checkpoint_interval: StdDuration::from_secs(3_600),
                checkpoint_timeout: StdDuration::from_secs(10),
                ..StreamRuntimeConfig::default()
            },
        )
        .unwrap();
        let mut runner = ContinuousRunner::new();
        let job = runner.start_checkpointed(spec, checkpoint).await.unwrap();
        wait_for_counter(&source_polls, 1).await;

        let epoch = tokio::time::timeout(StdDuration::from_secs(5), job.trigger_checkpoint())
            .await
            .expect("manual checkpoint should not hang")
            .unwrap();

        assert_eq!(epoch, crate::Epoch::INITIAL);
        assert!(
            directory
                .path()
                .join("manifests/manifest-00000000000000000001.json")
                .is_file()
        );
        assert_eq!(
            &*log.lock(),
            &[
                "sink-open",
                "sink-begin:1",
                "sink-precommit:1",
                "sink-commit:1",
                "sink-begin:2",
            ]
        );

        assert_eq!(job.cancel().await.state, ContinuousJobState::Cancelled);
        drop(job);
        runner.shutdown().await.unwrap();
        assert_eq!(source_closed.load(Ordering::SeqCst), 1);
        assert_eq!(sink_closed.load(Ordering::SeqCst), 1);
    }

    #[tokio::test]
    #[allow(
        clippy::too_many_lines,
        reason = "post-manifest recovery and the next manual epoch are one restart contract"
    )]
    async fn post_manifest_manual_commit_failure_requires_recovery_and_continues_epoch() {
        let directory = tempfile::tempdir().unwrap();
        let manifest_root = directory.path().join("manifests");
        let fail_commit = Arc::new(AtomicBool::new(true));
        let log = Arc::new(Mutex::new(Vec::new()));
        let plan = || {
            PipelineBuilder::new("checkpoint-manual-recovery")
                .unwrap()
                .add_checkpoint_capable_node(
                    "node",
                    Box::new(StressForwardOperator::new(None)) as Box<dyn StreamOperator>,
                )
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
                .unwrap()
        };
        let checkpoint = || {
            CheckpointRuntimeSpec::managed(
                ManagedCheckpointRuntime::new(directory.path()).unwrap(),
                StreamRuntimeConfig {
                    checkpoint_interval: StdDuration::from_secs(3_600),
                    checkpoint_timeout: StdDuration::from_secs(10),
                    ..StreamRuntimeConfig::default()
                },
            )
            .unwrap()
        };
        let spec = |job_id, source_polls: &Arc<AtomicUsize>, source_closed, sink_closed| {
            let plan = plan();
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
                    binding: counting_pending_binding(0, source_polls, &source_closed),
                }],
                sinks: vec![NamedSinkBinding {
                    output_id: "output".into(),
                    sink_id: "sink".into(),
                    binding: OrdinarySinkBinding::new_transactional(Box::new(FailOnceCommitSink {
                        fail_commit: Arc::clone(&fail_commit),
                        log: Arc::clone(&log),
                        closed: sink_closed,
                    })),
                }],
                edge_budget: EdgeBudget {
                    max_rows: 1,
                    max_bytes: 1 << 20,
                },
                delivery_mode: M2DeliveryMode::ProcessLocalOrdered,
            }
        };
        let first_source_polls = Arc::new(AtomicUsize::new(0));
        let first_source_closed = Arc::new(AtomicUsize::new(0));
        let first_sink_closed = Arc::new(AtomicUsize::new(0));
        let mut first_runner = ContinuousRunner::new();
        let first_job = first_runner
            .start_checkpointed(
                spec(
                    914,
                    &first_source_polls,
                    Arc::clone(&first_source_closed),
                    Arc::clone(&first_sink_closed),
                ),
                checkpoint(),
            )
            .await
            .unwrap();
        wait_for_counter(&first_source_polls, 1).await;

        let (manual, failed) = tokio::time::timeout(StdDuration::from_secs(5), async {
            tokio::join!(first_job.trigger_checkpoint(), first_job.wait())
        })
        .await
        .expect("post-manifest commit failure should converge");

        let CalcFlowError::Streaming(manual_error) = manual.unwrap_err() else {
            panic!("manual commit failure must use the safe streaming boundary");
        };
        assert_eq!(
            manual_error.category(),
            crate::runtime::streaming::projection::StreamingErrorCategory::Connector
        );
        assert_eq!(manual_error.epoch(), Some(crate::Epoch::INITIAL));
        assert_eq!(
            manual_error.checkpoint_phase(),
            Some(crate::runtime::streaming::projection::CheckpointPhase::ManifestDurable)
        );
        assert_eq!(
            manual_error.component_kind(),
            Some(crate::runtime::streaming::projection::ComponentKind::Sink)
        );
        assert_eq!(manual_error.component_id(), Some("sink"));
        assert_eq!(failed.state, ContinuousJobState::RecoveryRequired);
        assert!(
            manifest_root
                .join("manifest-00000000000000000001.json")
                .is_file()
        );
        assert!(!log.lock().iter().any(|entry| entry == "sink-abort:1"));
        drop(first_job);
        first_runner.shutdown().await.unwrap();
        assert_eq!(first_runner.registry_counts(), (0, 0));
        assert_eq!(first_source_closed.load(Ordering::SeqCst), 1);
        assert_eq!(first_sink_closed.load(Ordering::SeqCst), 1);

        let restart_source_polls = Arc::new(AtomicUsize::new(0));
        let restart_source_closed = Arc::new(AtomicUsize::new(0));
        let restart_sink_closed = Arc::new(AtomicUsize::new(0));
        let mut restart_runner = ContinuousRunner::new();
        let restart_job = restart_runner
            .start_checkpointed(
                spec(
                    915,
                    &restart_source_polls,
                    Arc::clone(&restart_source_closed),
                    Arc::clone(&restart_sink_closed),
                ),
                checkpoint(),
            )
            .await
            .unwrap();
        wait_for_counter(&restart_source_polls, 1).await;

        let restarted_epoch =
            tokio::time::timeout(StdDuration::from_secs(5), restart_job.trigger_checkpoint())
                .await
                .expect("restart manual checkpoint should not hang")
                .unwrap();

        assert_eq!(restarted_epoch, crate::Epoch::new(2).unwrap());
        assert!(log.lock().iter().any(|entry| entry == "sink-recover:1"));
        assert!(log.lock().iter().any(|entry| entry == "sink-commit:2"));
        assert_eq!(
            restart_job.cancel().await.state,
            ContinuousJobState::Cancelled
        );
        drop(restart_job);
        restart_runner.shutdown().await.unwrap();
        assert_eq!(restart_runner.registry_counts(), (0, 0));
        assert_eq!(restart_source_closed.load(Ordering::SeqCst), 1);
        assert_eq!(restart_sink_closed.load(Ordering::SeqCst), 1);
    }

    #[tokio::test]
    async fn cancel_fails_inflight_and_queued_manual_requests_without_leaks() {
        let directory = tempfile::tempdir().unwrap();
        let backend = Arc::new(
            LocalStateBackend::new(directory.path().join("state"))
                .await
                .unwrap(),
        );
        let source_polls = Arc::new(AtomicUsize::new(0));
        let source_closed = Arc::new(AtomicUsize::new(0));
        let sink_closed = Arc::new(AtomicUsize::new(0));
        let gate = super::CheckpointStartedTestGate::default();
        let checkpoint = CheckpointRuntimeSpec::new(
            backend,
            directory.path().join("manifests"),
            StreamRuntimeConfig {
                checkpoint_interval: StdDuration::from_secs(3_600),
                checkpoint_timeout: StdDuration::from_secs(10),
                ..StreamRuntimeConfig::default()
            },
        )
        .unwrap()
        .with_started_gate(gate.clone());
        let spec = pending_checkpoint_spec(
            "checkpoint-manual-cancel",
            916,
            &source_polls,
            &source_closed,
            Box::new(CheckpointProbeSink {
                opened: Arc::new(AtomicUsize::new(0)),
                closed: Arc::clone(&sink_closed),
            }),
        );
        let mut runner = ContinuousRunner::new();
        let job = Arc::new(runner.start_checkpointed(spec, checkpoint).await.unwrap());
        wait_for_counter(&source_polls, 1).await;
        let inflight = tokio::spawn({
            let job = Arc::clone(&job);
            async move { job.trigger_checkpoint().await }
        });
        gate.wait_until_entered().await;
        let coordinator = job
            .core
            .manual_checkpoint
            .lock()
            .clone()
            .expect("checkpoint task registered its manual queue");
        let queued = coordinator.request_manual().await.unwrap();

        let cancelled = job.cancel();
        gate.release();
        let outcome = cancelled.await;
        let inflight = inflight.await.unwrap();

        assert!(matches!(inflight, Err(CalcFlowError::Cancelled { .. })));
        assert!(matches!(queued.await, Err(CalcFlowError::Cancelled { .. })));
        assert_eq!(outcome.state, ContinuousJobState::Cancelled);
        assert_terminal_checkpoint_resources_released(&job);
        drop(job);
        runner.shutdown().await.unwrap();
        assert_eq!(runner.registry_counts(), (0, 0));
        assert_eq!(source_closed.load(Ordering::SeqCst), 1);
        assert_eq!(sink_closed.load(Ordering::SeqCst), 1);
    }

    #[tokio::test]
    async fn dropped_manual_future_keeps_accepted_request_in_fifo() {
        let directory = tempfile::tempdir().unwrap();
        let backend = Arc::new(
            LocalStateBackend::new(directory.path().join("state"))
                .await
                .unwrap(),
        );
        let source_polls = Arc::new(AtomicUsize::new(0));
        let source_closed = Arc::new(AtomicUsize::new(0));
        let sink_closed = Arc::new(AtomicUsize::new(0));
        let log = Arc::new(Mutex::new(Vec::new()));
        let gate = super::CheckpointStartedTestGate::default();
        let spec = pending_checkpoint_spec(
            "checkpoint-manual-drop",
            920,
            &source_polls,
            &source_closed,
            Box::new(PeriodicCheckpointSink {
                log: Arc::clone(&log),
                closed: Arc::clone(&sink_closed),
            }),
        );
        let checkpoint = CheckpointRuntimeSpec::new(
            backend,
            directory.path().join("manifests"),
            StreamRuntimeConfig {
                checkpoint_interval: StdDuration::from_secs(3_600),
                checkpoint_timeout: StdDuration::from_secs(10),
                ..StreamRuntimeConfig::default()
            },
        )
        .unwrap()
        .with_started_gate(gate.clone());
        let mut runner = ContinuousRunner::new();
        let job = Arc::new(runner.start_checkpointed(spec, checkpoint).await.unwrap());
        wait_for_counter(&source_polls, 1).await;
        let dropped = tokio::spawn({
            let job = Arc::clone(&job);
            async move { job.trigger_checkpoint().await }
        });
        gate.wait_until_entered().await;

        dropped.abort();
        assert!(dropped.await.unwrap_err().is_cancelled());
        gate.release();
        tokio::time::timeout(StdDuration::from_secs(5), async {
            loop {
                if log.lock().iter().any(|entry| entry == "sink-begin:2") {
                    break;
                }
                tokio::task::yield_now().await;
            }
        })
        .await
        .expect("dropped waiter must not cancel its accepted checkpoint");

        assert_eq!(
            job.trigger_checkpoint().await.unwrap(),
            crate::Epoch::new(2).unwrap()
        );
        assert!(log.lock().iter().any(|entry| entry == "sink-commit:1"));
        assert!(log.lock().iter().any(|entry| entry == "sink-commit:2"));
        assert_eq!(job.cancel().await.state, ContinuousJobState::Cancelled);
        assert_terminal_checkpoint_resources_released(&job);
        drop(job);
        runner.shutdown().await.unwrap();
        assert_eq!(runner.registry_counts(), (0, 0));
        assert_eq!(source_closed.load(Ordering::SeqCst), 1);
        assert_eq!(sink_closed.load(Ordering::SeqCst), 1);
    }

    #[tokio::test]
    async fn cancel_after_manifest_durability_finishes_commit_before_manual_success() {
        let directory = tempfile::tempdir().unwrap();
        let backend = Arc::new(
            LocalStateBackend::new(directory.path().join("state"))
                .await
                .unwrap(),
        );
        let source_polls = Arc::new(AtomicUsize::new(0));
        let source_closed = Arc::new(AtomicUsize::new(0));
        let sink_closed = Arc::new(AtomicUsize::new(0));
        let commit_entered = Arc::new(AtomicBool::new(false));
        let commit_changed = Arc::new(Notify::new());
        let commit_release = Arc::new(Semaphore::new(0));
        let log = Arc::new(Mutex::new(Vec::new()));
        let spec = pending_checkpoint_spec(
            "checkpoint-manual-durable-cancel",
            917,
            &source_polls,
            &source_closed,
            Box::new(BlockingCommitSink {
                commit_entered: Arc::clone(&commit_entered),
                commit_changed: Arc::clone(&commit_changed),
                commit_release: Arc::clone(&commit_release),
                log: Arc::clone(&log),
                closed: Arc::clone(&sink_closed),
            }),
        );
        let checkpoint = CheckpointRuntimeSpec::new(
            backend,
            directory.path().join("manifests"),
            StreamRuntimeConfig {
                checkpoint_interval: StdDuration::from_secs(3_600),
                checkpoint_timeout: StdDuration::from_secs(10),
                ..StreamRuntimeConfig::default()
            },
        )
        .unwrap();
        let mut runner = ContinuousRunner::new();
        let job = Arc::new(runner.start_checkpointed(spec, checkpoint).await.unwrap());
        wait_for_counter(&source_polls, 1).await;
        let manual = tokio::spawn({
            let job = Arc::clone(&job);
            async move { job.trigger_checkpoint().await }
        });
        while !commit_entered.load(Ordering::Acquire) {
            let changed = commit_changed.notified();
            if commit_entered.load(Ordering::Acquire) {
                break;
            }
            changed.await;
        }

        let cancelled = job.cancel();
        loop {
            if job.core.state.lock().selected_cause == Some(TerminalCause::ExplicitCancel) {
                break;
            }
            tokio::task::yield_now().await;
        }
        commit_release.add_permits(1);
        let (manual, outcome) = tokio::join!(manual, cancelled);

        assert_eq!(manual.unwrap().unwrap(), crate::Epoch::INITIAL);
        assert_eq!(outcome.state, ContinuousJobState::Cancelled);
        assert!(log.lock().iter().any(|entry| entry == "sink-commit:1"));
        assert!(!log.lock().iter().any(|entry| entry == "sink-abort:1"));
        assert_terminal_checkpoint_resources_released(&job);
        drop(job);
        runner.shutdown().await.unwrap();
        assert_eq!(runner.registry_counts(), (0, 0));
        assert_eq!(source_closed.load(Ordering::SeqCst), 1);
        assert_eq!(sink_closed.load(Ordering::SeqCst), 1);
    }

    #[tokio::test]
    async fn shutdown_drains_inflight_and_queued_manual_requests_before_completion() {
        let directory = tempfile::tempdir().unwrap();
        let backend = Arc::new(
            LocalStateBackend::new(directory.path().join("state"))
                .await
                .unwrap(),
        );
        let source_polls = Arc::new(AtomicUsize::new(0));
        let source_closed = Arc::new(AtomicUsize::new(0));
        let sink_closed = Arc::new(AtomicUsize::new(0));
        let gate = super::CheckpointStartedTestGate::default();
        let spec = pending_checkpoint_spec(
            "checkpoint-manual-shutdown",
            918,
            &source_polls,
            &source_closed,
            Box::new(CheckpointProbeSink {
                opened: Arc::new(AtomicUsize::new(0)),
                closed: Arc::clone(&sink_closed),
            }),
        );
        let checkpoint = CheckpointRuntimeSpec::new(
            backend,
            directory.path().join("manifests"),
            StreamRuntimeConfig {
                checkpoint_interval: StdDuration::from_secs(3_600),
                checkpoint_timeout: StdDuration::from_secs(10),
                ..StreamRuntimeConfig::default()
            },
        )
        .unwrap()
        .with_started_gate(gate.clone());
        let mut runner = ContinuousRunner::new();
        let job = Arc::new(runner.start_checkpointed(spec, checkpoint).await.unwrap());
        wait_for_counter(&source_polls, 1).await;
        let inflight = tokio::spawn({
            let job = Arc::clone(&job);
            async move { job.trigger_checkpoint().await }
        });
        gate.wait_until_entered().await;
        let coordinator = job
            .core
            .manual_checkpoint
            .lock()
            .clone()
            .expect("checkpoint task registered its manual queue");
        let queued = coordinator.request_manual().await.unwrap();

        let shutdown = job.shutdown();
        gate.release();
        let (inflight, queued, outcome) = tokio::join!(inflight, queued, shutdown);

        assert_eq!(inflight.unwrap().unwrap(), crate::Epoch::INITIAL);
        assert_eq!(queued.unwrap(), crate::Epoch::new(2).unwrap());
        assert_eq!(outcome.state, ContinuousJobState::Completed);
        assert_eq!(outcome.cause, TerminalCause::GracefulShutdown);
        assert!(
            directory
                .path()
                .join("manifests/manifest-00000000000000000001.json")
                .is_file()
        );
        assert!(
            directory
                .path()
                .join("manifests/manifest-00000000000000000002.json")
                .is_file()
        );
        assert_terminal_checkpoint_resources_released(&job);
        drop(job);
        runner.shutdown().await.unwrap();
        assert_eq!(runner.registry_counts(), (0, 0));
        assert_eq!(source_closed.load(Ordering::SeqCst), 1);
        assert_eq!(sink_closed.load(Ordering::SeqCst), 1);
    }

    #[tokio::test(start_paused = true)]
    async fn manual_request_queued_during_periodic_checkpoint_receives_next_epoch() {
        let directory = tempfile::tempdir().unwrap();
        let backend = Arc::new(
            LocalStateBackend::new(directory.path().join("state"))
                .await
                .unwrap(),
        );
        let source_polls = Arc::new(AtomicUsize::new(0));
        let source_closed = Arc::new(AtomicUsize::new(0));
        let sink_closed = Arc::new(AtomicUsize::new(0));
        let log = Arc::new(Mutex::new(Vec::new()));
        let gate = super::CheckpointStartedTestGate::default();
        let spec = pending_checkpoint_spec(
            "checkpoint-periodic-manual-race",
            919,
            &source_polls,
            &source_closed,
            Box::new(PeriodicCheckpointSink {
                log: Arc::clone(&log),
                closed: Arc::clone(&sink_closed),
            }),
        );
        let checkpoint = CheckpointRuntimeSpec::new(
            backend,
            directory.path().join("manifests"),
            StreamRuntimeConfig {
                checkpoint_interval: StdDuration::from_millis(10),
                checkpoint_timeout: StdDuration::from_secs(10),
                ..StreamRuntimeConfig::default()
            },
        )
        .unwrap()
        .with_started_gate(gate.clone());
        let mut runner = ContinuousRunner::new();
        let job = runner.start_checkpointed(spec, checkpoint).await.unwrap();
        wait_for_counter(&source_polls, 1).await;

        tokio::time::advance(StdDuration::from_millis(10)).await;
        gate.wait_until_entered().await;
        let coordinator = job
            .core
            .manual_checkpoint
            .lock()
            .clone()
            .expect("checkpoint task registered its manual queue");
        let manual = coordinator.request_manual().await.unwrap();
        gate.release();

        assert_eq!(manual.await.unwrap(), crate::Epoch::new(2).unwrap());
        assert!(log.lock().iter().any(|entry| entry == "sink-commit:1"));
        assert!(log.lock().iter().any(|entry| entry == "sink-commit:2"));
        assert_eq!(job.cancel().await.state, ContinuousJobState::Cancelled);
        drop(job);
        runner.shutdown().await.unwrap();
        assert_eq!(runner.registry_counts(), (0, 0));
        assert_eq!(source_closed.load(Ordering::SeqCst), 1);
        assert_eq!(sink_closed.load(Ordering::SeqCst), 1);
    }

    #[tokio::test]
    async fn manual_checkpoint_faults_before_durability_fail_without_manifest_or_leaks() {
        for (name, point) in [
            ("barrier", super::CheckpointFaultPoint::SourceCut),
            ("precommit", super::CheckpointFaultPoint::SinkPreCommit),
            ("manifest", super::CheckpointFaultPoint::ManifestWrite),
        ] {
            let directory = tempfile::tempdir().unwrap();
            let backend = Arc::new(
                LocalStateBackend::new(directory.path().join("state"))
                    .await
                    .unwrap(),
            );
            let source_polls = Arc::new(AtomicUsize::new(0));
            let source_closed = Arc::new(AtomicUsize::new(0));
            let sink_closed = Arc::new(AtomicUsize::new(0));
            let log = Arc::new(Mutex::new(Vec::new()));
            let spec = pending_checkpoint_spec(
                &format!("checkpoint-manual-{name}-fault"),
                930,
                &source_polls,
                &source_closed,
                Box::new(PeriodicCheckpointSink {
                    log: Arc::clone(&log),
                    closed: Arc::clone(&sink_closed),
                }),
            );
            let checkpoint = CheckpointRuntimeSpec::new(
                backend,
                directory.path().join("manifests"),
                StreamRuntimeConfig {
                    checkpoint_interval: StdDuration::from_secs(3_600),
                    checkpoint_timeout: StdDuration::from_secs(10),
                    ..StreamRuntimeConfig::default()
                },
            )
            .unwrap()
            .with_fault(point, super::CheckpointFaultMode::Io);
            let mut runner = ContinuousRunner::new();
            let job = runner.start_checkpointed(spec, checkpoint).await.unwrap();
            wait_for_counter(&source_polls, 1).await;

            let (manual, outcome) = tokio::time::timeout(StdDuration::from_secs(5), async {
                tokio::join!(job.trigger_checkpoint(), job.wait())
            })
            .await
            .unwrap_or_else(|error| panic!("{name} fault did not converge: {error}"));

            let CalcFlowError::Streaming(error) = manual.unwrap_err() else {
                panic!("{name} manual failure must use the streaming boundary");
            };
            assert_eq!(
                error.category(),
                crate::runtime::streaming::projection::StreamingErrorCategory::Io,
                "{name}"
            );
            assert_eq!(error.epoch(), Some(crate::Epoch::INITIAL), "{name}");
            assert_eq!(outcome.state, ContinuousJobState::Failed, "{name}");
            assert!(
                !directory
                    .path()
                    .join("manifests/manifest-00000000000000000001.json")
                    .exists(),
                "{name} fault published a manifest"
            );
            drop(job);
            runner.shutdown().await.unwrap();
            assert_eq!(runner.registry_counts(), (0, 0), "{name}");
            assert_eq!(source_closed.load(Ordering::SeqCst), 1, "{name}");
            assert_eq!(sink_closed.load(Ordering::SeqCst), 1, "{name}");
        }
    }

    #[tokio::test]
    async fn installed_manifest_with_unknown_durability_requires_recovery() {
        let directory = tempfile::tempdir().unwrap();
        let backend = Arc::new(
            LocalStateBackend::new(directory.path().join("state"))
                .await
                .unwrap(),
        );
        let source_polls = Arc::new(AtomicUsize::new(0));
        let source_closed = Arc::new(AtomicUsize::new(0));
        let sink_closed = Arc::new(AtomicUsize::new(0));
        let sink_log = Arc::new(Mutex::new(Vec::new()));
        let spec = pending_checkpoint_spec(
            "checkpoint-manifest-installed-unknown",
            931,
            &source_polls,
            &source_closed,
            Box::new(PeriodicCheckpointSink {
                log: Arc::clone(&sink_log),
                closed: Arc::clone(&sink_closed),
            }),
        );
        let checkpoint = CheckpointRuntimeSpec::new(
            backend,
            directory.path().join("manifests"),
            StreamRuntimeConfig {
                checkpoint_interval: StdDuration::from_secs(3_600),
                checkpoint_timeout: StdDuration::from_secs(10),
                ..StreamRuntimeConfig::default()
            },
        )
        .unwrap()
        .with_fault(
            super::CheckpointFaultPoint::ManifestRename,
            super::CheckpointFaultMode::Io,
        );
        let mut runner = ContinuousRunner::new();
        let job = runner.start_checkpointed(spec, checkpoint).await.unwrap();
        wait_for_counter(&source_polls, 1).await;

        let (manual, outcome) = tokio::time::timeout(StdDuration::from_secs(5), async {
            tokio::join!(job.trigger_checkpoint(), job.wait())
        })
        .await
        .expect("indeterminate publication should converge");

        assert!(matches!(
            manual,
            Err(CalcFlowError::RecoveryRequired { .. })
        ));
        assert_eq!(outcome.state, ContinuousJobState::RecoveryRequired);
        let status = job.public_status();
        assert_eq!(
            status.state,
            crate::runtime::streaming::projection::JobState::RecoveryRequired
        );
        assert_eq!(
            status.checkpoint.installed_unknown_epoch,
            Some(crate::Epoch::INITIAL)
        );
        assert_eq!(status.checkpoint.last_completed_epoch, None);
        assert_eq!(
            status.checkpoint.phase,
            Some(crate::runtime::streaming::projection::CheckpointPhase::ManifestInstalled)
        );
        assert_eq!(
            status.checkpoint.failure_category,
            Some(
                crate::runtime::streaming::projection::StreamingErrorCategory::CheckpointPublicationUnknown
            )
        );
        let internal_status = job.status();
        let public_outcome = crate::runtime::streaming::projection::project_job_outcome(
            job.id(),
            &outcome,
            internal_status.checkpoint.as_ref(),
            None,
        );
        assert_eq!(public_outcome.errors.len(), 1);
        assert_eq!(
            public_outcome.errors[0].category(),
            crate::runtime::streaming::projection::StreamingErrorCategory::CheckpointPublicationUnknown
        );
        assert_eq!(
            public_outcome.errors[0].epoch(),
            Some(crate::Epoch::INITIAL)
        );
        assert!(!sink_log.lock().iter().any(|entry| entry == "sink-commit:1"));
        assert!(!sink_log.lock().iter().any(|entry| entry == "sink-abort:1"));
        drop(job);
        runner.shutdown().await.unwrap();
        assert_eq!(source_closed.load(Ordering::SeqCst), 1);
        assert_eq!(sink_closed.load(Ordering::SeqCst), 1);
    }

    #[tokio::test]
    async fn installed_unknown_publication_wins_over_concurrent_cancellation() {
        let directory = tempfile::tempdir().unwrap();
        let backend = Arc::new(
            LocalStateBackend::new(directory.path().join("state"))
                .await
                .unwrap(),
        );
        let source_polls = Arc::new(AtomicUsize::new(0));
        let source_closed = Arc::new(AtomicUsize::new(0));
        let sink_closed = Arc::new(AtomicUsize::new(0));
        let sink_log = Arc::new(Mutex::new(Vec::new()));
        let spec = pending_checkpoint_spec(
            "checkpoint-manifest-installed-cancelled",
            932,
            &source_polls,
            &source_closed,
            Box::new(PeriodicCheckpointSink {
                log: Arc::clone(&sink_log),
                closed: Arc::clone(&sink_closed),
            }),
        );
        let checkpoint = CheckpointRuntimeSpec::new(
            backend,
            directory.path().join("manifests"),
            StreamRuntimeConfig {
                checkpoint_interval: StdDuration::from_secs(3_600),
                checkpoint_timeout: StdDuration::from_secs(10),
                ..StreamRuntimeConfig::default()
            },
        )
        .unwrap()
        .with_fault(
            super::CheckpointFaultPoint::ManifestRename,
            super::CheckpointFaultMode::Cancel,
        );
        let mut runner = ContinuousRunner::new();
        let job = runner.start_checkpointed(spec, checkpoint).await.unwrap();
        wait_for_counter(&source_polls, 1).await;

        let (manual, outcome) = tokio::time::timeout(StdDuration::from_secs(5), async {
            tokio::join!(job.trigger_checkpoint(), job.wait())
        })
        .await
        .expect("publication/cancellation overlap should converge");

        assert!(matches!(
            manual,
            Err(CalcFlowError::RecoveryRequired { .. })
        ));
        assert_eq!(outcome.state, ContinuousJobState::RecoveryRequired);
        let status = job.public_status();
        assert_eq!(
            status.checkpoint.installed_unknown_epoch,
            Some(crate::Epoch::INITIAL)
        );
        assert_eq!(status.checkpoint.last_completed_epoch, None);
        assert!(!sink_log.lock().iter().any(|entry| entry == "sink-commit:1"));
        assert!(!sink_log.lock().iter().any(|entry| entry == "sink-abort:1"));
        drop(job);
        runner.shutdown().await.unwrap();
        assert_eq!(source_closed.load(Ordering::SeqCst), 1);
        assert_eq!(sink_closed.load(Ordering::SeqCst), 1);
    }

    #[tokio::test]
    #[allow(
        clippy::too_many_lines,
        reason = "the mixed-delivery runtime proof owns both disjoint output lifecycles"
    )]
    async fn checkpointed_runner_supports_disjoint_exactly_once_and_ordinary_outputs() {
        let directory = tempfile::tempdir().unwrap();
        let backend = Arc::new(
            LocalStateBackend::new(directory.path().join("state"))
                .await
                .unwrap(),
        );
        let plan = PipelineBuilder::new("checkpoint-mixed-delivery")
            .unwrap()
            .add_checkpoint_capable_node(
                "root",
                Box::new(StressForwardOperator::new(None)) as Box<dyn StreamOperator>,
            )
            .unwrap()
            .add_checkpoint_capable_node(
                "exact",
                Box::new(StressForwardOperator::new(None)) as Box<dyn StreamOperator>,
            )
            .unwrap()
            .add_checkpoint_capable_node(
                "ordinary",
                Box::new(StressForwardOperator::new(None)) as Box<dyn StreamOperator>,
            )
            .unwrap()
            .connect(Edge::new(
                PortEndpoint::new("root", "output").unwrap(),
                PortEndpoint::new("exact", "input").unwrap(),
            ))
            .unwrap()
            .connect(Edge::new(
                PortEndpoint::new("root", "output").unwrap(),
                PortEndpoint::new("ordinary", "input").unwrap(),
            ))
            .unwrap()
            .compile_stream(
                &UdfRegistry::new().snapshot(),
                &StreamRequirements {
                    delivery: BTreeMap::from([(
                        "exact.output".into(),
                        crate::DeliveryGuarantee::ExactlyOnce,
                    )]),
                },
            )
            .unwrap();
        let source_closed = Arc::new(AtomicUsize::new(0));
        let transactional_closed = Arc::new(AtomicUsize::new(0));
        let transactional_log = Arc::new(Mutex::new(Vec::new()));
        let ordinary_closed = Arc::new(AtomicUsize::new(0));
        let ordinary_writes = Arc::new(Mutex::new(Vec::new()));
        let spec = ContinuousJobSpec {
            context: StreamJobContext::new(
                908,
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
            sinks: vec![
                NamedSinkBinding {
                    output_id: "exact.output".into(),
                    sink_id: "transactional".into(),
                    binding: OrdinarySinkBinding::new_transactional(Box::new(
                        PeriodicCheckpointSink {
                            log: Arc::clone(&transactional_log),
                            closed: Arc::clone(&transactional_closed),
                        },
                    )),
                },
                NamedSinkBinding {
                    output_id: "ordinary.output".into(),
                    sink_id: "ordinary".into(),
                    binding: OrdinarySinkBinding::new(Box::new(OrderedRecordingSink {
                        id: "ordinary".into(),
                        writes: Arc::clone(&ordinary_writes),
                        closed: Arc::clone(&ordinary_closed),
                    })),
                },
            ],
            edge_budget: EdgeBudget {
                max_rows: 1,
                max_bytes: 1 << 20,
            },
            delivery_mode: M2DeliveryMode::ProcessLocalOrdered,
        };
        let manifest_root = directory.path().join("manifests");
        let checkpoint = CheckpointRuntimeSpec::new(
            backend,
            &manifest_root,
            StreamRuntimeConfig {
                checkpoint_interval: StdDuration::from_secs(3_600),
                ..StreamRuntimeConfig::default()
            },
        )
        .unwrap();
        let mut runner = ContinuousRunner::new();

        let job = runner.start_checkpointed(spec, checkpoint).await.unwrap();
        let outcome = tokio::time::timeout(StdDuration::from_secs(5), job.wait())
            .await
            .expect("mixed-delivery checkpointed job hung");

        assert_eq!(outcome.state, ContinuousJobState::Completed, "{outcome:?}");
        assert_eq!(ordinary_writes.lock().len(), 1);
        assert_eq!(source_closed.load(Ordering::SeqCst), 1);
        assert_eq!(transactional_closed.load(Ordering::SeqCst), 1);
        assert_eq!(ordinary_closed.load(Ordering::SeqCst), 1);
        assert!(
            transactional_log
                .lock()
                .iter()
                .any(|entry| entry == "sink-commit:1")
        );
        let manifest = crate::CheckpointManifest::from_bytes(
            &tokio::fs::read(manifest_root.join("manifest-00000000000000000001.json"))
                .await
                .unwrap(),
        )
        .unwrap();
        assert_eq!(
            manifest.sinks()["transactional"].delivery,
            SinkDeliveryManifest::Transactional
        );
        assert_eq!(
            manifest.sinks()["ordinary"].delivery,
            SinkDeliveryManifest::Ordinary
        );
        assert!(manifest.sinks()["ordinary"].pre_commit.is_none());
        drop(job);
        runner.shutdown().await.unwrap();
    }

    #[tokio::test]
    #[allow(
        clippy::too_many_lines,
        reason = "the mixed-delivery restart proof owns both job generations and their lifecycle assertions"
    )]
    async fn mixed_delivery_restart_preserves_exactly_once_and_exposes_ordinary_replay() {
        let directory = tempfile::tempdir().unwrap();
        let backend = Arc::new(
            LocalStateBackend::new(directory.path().join("state"))
                .await
                .unwrap(),
        );
        let manifest_root = directory.path().join("manifests");
        let probes = MixedDeliveryProbes::default();
        let checkpoint = || {
            CheckpointRuntimeSpec::new(
                backend.clone(),
                &manifest_root,
                StreamRuntimeConfig {
                    checkpoint_interval: StdDuration::from_secs(3_600),
                    ..StreamRuntimeConfig::default()
                },
            )
            .unwrap()
        };

        let mut first_runner = ContinuousRunner::new();
        let first_job = first_runner
            .start_checkpointed(
                mixed_delivery_fault_spec(910, &probes),
                checkpoint().with_fault(
                    super::CheckpointFaultPoint::ManifestWrite,
                    super::CheckpointFaultMode::Restart,
                ),
            )
            .await
            .unwrap();
        let first_outcome = tokio::time::timeout(StdDuration::from_secs(5), first_job.wait())
            .await
            .expect("mixed-delivery manifest-write fault hung");

        assert_eq!(first_outcome.state, ContinuousJobState::Failed);
        assert!(first_outcome.errors.iter().any(|failure| {
            matches!(
                &failure.error,
                CalcFlowError::Internal { message }
                    if message == "injected checkpoint restart at ManifestWrite"
            )
        }));
        assert_eq!(
            &*probes.ordinary_writes.lock(),
            &[("ordinary".into(), "input".into(), 0)]
        );
        assert!(probes.transactional.lock().visible.is_empty());
        assert!(
            !manifest_root
                .join("manifest-00000000000000000001.json")
                .exists()
        );
        assert_terminal_checkpoint_resources_released(&first_job);
        drop(first_job);
        first_runner.shutdown().await.unwrap();
        assert_eq!(first_runner.registry_counts(), (0, 0));
        assert_eq!(probes.source_closed.load(Ordering::SeqCst), 1);
        assert_eq!(probes.transactional_closed.load(Ordering::SeqCst), 1);
        assert_eq!(probes.ordinary_closed.load(Ordering::SeqCst), 1);

        let mut restart_runner = ContinuousRunner::new();
        let restart_job = restart_runner
            .start_checkpointed(mixed_delivery_fault_spec(911, &probes), checkpoint())
            .await
            .unwrap();
        let restart_outcome = tokio::time::timeout(StdDuration::from_secs(5), restart_job.wait())
            .await
            .expect("mixed-delivery restart hung");

        assert_eq!(
            restart_outcome.state,
            ContinuousJobState::Completed,
            "{restart_outcome:?}"
        );
        let ordinary_at_least_once_boundary = ("ordinary".into(), "input".into(), 0);
        assert_eq!(
            &*probes.ordinary_writes.lock(),
            &[
                ordinary_at_least_once_boundary.clone(),
                ordinary_at_least_once_boundary,
            ]
        );
        assert_eq!(&probes.transactional.lock().visible, &[("input".into(), 0)]);
        assert_eq!(probes.source_closed.load(Ordering::SeqCst), 2);
        assert_eq!(probes.transactional_closed.load(Ordering::SeqCst), 2);
        assert_eq!(probes.ordinary_closed.load(Ordering::SeqCst), 2);
        assert_terminal_checkpoint_resources_released(&restart_job);
        drop(restart_job);
        restart_runner.shutdown().await.unwrap();
        assert_eq!(restart_runner.registry_counts(), (0, 0));
    }

    #[tokio::test]
    #[allow(
        clippy::too_many_lines,
        reason = "the checkpoint ordering scenario is clearest as one end-to-end test"
    )]
    async fn terminal_checkpoint_waits_until_the_periodic_epoch_completes() {
        let participants = ParticipantSet {
            sources: BTreeSet::from(["source".into()]),
            operators: BTreeSet::from(["operator".into()]),
            sinks: BTreeSet::from(["sink".into()]),
        };
        let cancellation = CancellationToken::new();
        let (coordinator, mut events, task) = spawn_checkpoint_coordinator(
            participants.clone(),
            crate::Epoch::INITIAL,
            4,
            StdDuration::from_secs(30),
            cancellation.clone(),
        )
        .unwrap();
        coordinator
            .request(CheckpointRequest::Periodic)
            .await
            .unwrap();
        let mut request_active = true;
        let mut terminal_request_active = false;
        assert_eq!(
            events.recv().await.unwrap(),
            CheckpointEvent::Started(crate::Epoch::INITIAL)
        );

        maybe_request_terminal_checkpoint(
            &coordinator,
            &participants.operators,
            &participants.sinks,
            &participants.operators,
            &participants.sinks,
            true,
            &mut request_active,
            &mut terminal_request_active,
        )
        .await
        .unwrap();
        assert!(request_active);
        assert!(!terminal_request_active);

        coordinator
            .ack(CheckpointAck::source(
                "source",
                crate::Epoch::INITIAL,
                "source-state",
            ))
            .await
            .unwrap();
        assert_eq!(
            events.recv().await.unwrap(),
            CheckpointEvent::PhaseAdvanced(crate::Epoch::INITIAL, CoordinatorPhase::SourcesCut)
        );
        coordinator
            .ack(CheckpointAck::operator(
                "operator",
                crate::Epoch::INITIAL,
                "operator-state",
            ))
            .await
            .unwrap();
        assert_eq!(
            events.recv().await.unwrap(),
            CheckpointEvent::PhaseAdvanced(
                crate::Epoch::INITIAL,
                CoordinatorPhase::OperatorsSnapshotted,
            )
        );
        coordinator
            .ack(CheckpointAck::sink_precommit(
                "sink",
                crate::Epoch::INITIAL,
                "sink-state",
            ))
            .await
            .unwrap();
        assert_eq!(
            events.recv().await.unwrap(),
            CheckpointEvent::ReadyToPublish(crate::Epoch::INITIAL)
        );
        coordinator
            .manifest_durable(crate::Epoch::INITIAL)
            .await
            .unwrap();
        assert_eq!(
            events.recv().await.unwrap(),
            CheckpointEvent::PhaseAdvanced(
                crate::Epoch::INITIAL,
                CoordinatorPhase::ManifestDurable,
            )
        );
        coordinator
            .ack(CheckpointAck::sink_commit("sink", crate::Epoch::INITIAL))
            .await
            .unwrap();
        assert_eq!(
            events.recv().await.unwrap(),
            CheckpointEvent::Completed(crate::Epoch::INITIAL)
        );
        tokio::task::yield_now().await;
        assert!(matches!(
            events.try_recv(),
            Err(mpsc::error::TryRecvError::Empty)
        ));

        request_active = false;
        maybe_request_terminal_checkpoint(
            &coordinator,
            &participants.operators,
            &participants.sinks,
            &participants.operators,
            &participants.sinks,
            true,
            &mut request_active,
            &mut terminal_request_active,
        )
        .await
        .unwrap();
        assert!(request_active);
        assert!(terminal_request_active);
        assert_eq!(
            events.recv().await.unwrap(),
            CheckpointEvent::Started(crate::Epoch::INITIAL.next().unwrap())
        );

        cancellation.cancel();
        task.await.unwrap().unwrap();
    }

    #[test]
    fn periodic_cut_after_all_sources_end_is_terminal() {
        let cut = |ended| DurableSourceCut {
            cursor: None,
            next_sequence: 1,
            ended,
        };
        let all_ended = BTreeMap::from([
            (
                crate::runtime::streaming::progress::BindingIdentity::new("left").unwrap(),
                cut(true),
            ),
            (
                crate::runtime::streaming::progress::BindingIdentity::new("right").unwrap(),
                cut(true),
            ),
        ]);
        let one_live = BTreeMap::from([
            (
                crate::runtime::streaming::progress::BindingIdentity::new("left").unwrap(),
                cut(true),
            ),
            (
                crate::runtime::streaming::progress::BindingIdentity::new("right").unwrap(),
                cut(false),
            ),
        ]);

        assert!(source_cuts_are_terminal(&all_ended));
        assert!(!source_cuts_are_terminal(&one_live));
        assert!(!source_cuts_are_terminal(&BTreeMap::new()));
    }

    #[tokio::test]
    #[allow(
        clippy::too_many_lines,
        reason = "terminal publication and restart short-circuit are one end-to-end recovery contract"
    )]
    async fn checkpointed_runner_commits_a_terminal_epoch_without_a_post_end_barrier() {
        let directory = tempfile::tempdir().unwrap();
        let terminal_plan = || {
            PipelineBuilder::new("checkpoint-terminal")
                .unwrap()
                .add_checkpoint_capable_node(
                    "node",
                    Box::new(StressForwardOperator::new(None)) as Box<dyn StreamOperator>,
                )
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
                .unwrap()
        };
        let plan = terminal_plan();
        let source_closed = Arc::new(AtomicUsize::new(0));
        let sink_closed = Arc::new(AtomicUsize::new(0));
        let log = Arc::new(Mutex::new(Vec::new()));
        let spec = ContinuousJobSpec {
            context: StreamJobContext::new(
                904,
                plan.fingerprint(),
                JsonMap::new(),
                None,
                CancellationToken::new(),
            ),
            plan,
            sources: vec![NamedSourceBinding {
                binding_id: "input".into(),
                binding: finite_binding(&[7], &source_closed),
            }],
            sinks: vec![NamedSinkBinding {
                output_id: "output".into(),
                sink_id: "sink".into(),
                binding: OrdinarySinkBinding::new_transactional(Box::new(PeriodicCheckpointSink {
                    log: Arc::clone(&log),
                    closed: Arc::clone(&sink_closed),
                })),
            }],
            edge_budget: EdgeBudget {
                max_rows: 1,
                max_bytes: 1 << 20,
            },
            delivery_mode: M2DeliveryMode::ProcessLocalOrdered,
        };
        let checkpoint = CheckpointRuntimeSpec::managed(
            ManagedCheckpointRuntime::new(directory.path()).unwrap(),
            StreamRuntimeConfig {
                checkpoint_interval: StdDuration::from_secs(3_600),
                checkpoint_timeout: StdDuration::from_secs(10),
                ..StreamRuntimeConfig::default()
            },
        )
        .unwrap();
        let mut runner = ContinuousRunner::new();
        let job = runner.start_checkpointed(spec, checkpoint).await.unwrap();

        let outcome = match tokio::time::timeout(StdDuration::from_millis(100), job.wait()).await {
            Ok(outcome) => outcome,
            Err(error) => {
                let _ = job.cancel().await;
                drop(job);
                runner.shutdown().await.unwrap();
                panic!("terminal checkpoint did not complete: {error}");
            }
        };

        assert_eq!(outcome.state, ContinuousJobState::Completed);
        assert_eq!(outcome.cause, TerminalCause::NaturalEnd);
        drop(job);
        runner.shutdown().await.unwrap();
        assert_eq!(
            &*log.lock(),
            &[
                "sink-open",
                "sink-begin:1",
                "sink-precommit:1",
                "sink-commit:1",
                "sink-close",
            ]
        );
        assert_eq!(source_closed.load(Ordering::SeqCst), 1);
        assert_eq!(sink_closed.load(Ordering::SeqCst), 1);

        log.lock().clear();
        let source_polls = Arc::new(AtomicUsize::new(0));
        let restored_source_closed = Arc::new(AtomicUsize::new(0));
        let restored_sink_closed = Arc::new(AtomicUsize::new(0));
        let plan = terminal_plan();
        let restored_spec = ContinuousJobSpec {
            context: StreamJobContext::new(
                905,
                plan.fingerprint(),
                JsonMap::new(),
                None,
                CancellationToken::new(),
            ),
            plan,
            sources: vec![NamedSourceBinding {
                binding_id: "input".into(),
                binding: counting_pending_binding(0, &source_polls, &restored_source_closed),
            }],
            sinks: vec![NamedSinkBinding {
                output_id: "output".into(),
                sink_id: "sink".into(),
                binding: OrdinarySinkBinding::new_transactional(Box::new(PeriodicCheckpointSink {
                    log: Arc::clone(&log),
                    closed: Arc::clone(&restored_sink_closed),
                })),
            }],
            edge_budget: EdgeBudget {
                max_rows: 1,
                max_bytes: 1 << 20,
            },
            delivery_mode: M2DeliveryMode::ProcessLocalOrdered,
        };
        let restored_checkpoint = CheckpointRuntimeSpec::managed(
            ManagedCheckpointRuntime::new(directory.path()).unwrap(),
            StreamRuntimeConfig {
                checkpoint_interval: StdDuration::from_secs(3_600),
                checkpoint_timeout: StdDuration::from_secs(10),
                ..StreamRuntimeConfig::default()
            },
        )
        .unwrap();
        let mut restored_runner = ContinuousRunner::new();
        let restored_job = restored_runner
            .start_checkpointed(restored_spec, restored_checkpoint)
            .await
            .unwrap();
        let restored = tokio::time::timeout(StdDuration::from_millis(100), restored_job.wait())
            .await
            .expect("terminal manifest recovery should short-circuit");
        assert_eq!(restored.state, ContinuousJobState::Completed);
        drop(restored_job);
        restored_runner.shutdown().await.unwrap();
        assert_eq!(source_polls.load(Ordering::SeqCst), 0);
        assert_eq!(restored_source_closed.load(Ordering::SeqCst), 0);
        assert_eq!(restored_sink_closed.load(Ordering::SeqCst), 1);
        assert_eq!(&*log.lock(), &["sink-open", "sink-recover:1", "sink-close"]);
    }

    #[tokio::test]
    #[allow(
        clippy::too_many_lines,
        reason = "the retention failure and restart are one durable-manifest recovery contract"
    )]
    async fn retention_failure_after_terminal_commit_recovers_the_durable_epoch() {
        let directory = tempfile::tempdir().unwrap();
        let retention_failure_armed = Arc::new(AtomicBool::new(false));
        let backend = Arc::new(FailOnceRetentionBackend {
            inner: LocalStateBackend::new(directory.path().join("state"))
                .await
                .unwrap(),
            failure_armed: Arc::clone(&retention_failure_armed),
        });
        let terminal_plan = || {
            PipelineBuilder::new("checkpoint-retention-fault")
                .unwrap()
                .add_checkpoint_capable_node(
                    "node",
                    Box::new(StressForwardOperator::new(None)) as Box<dyn StreamOperator>,
                )
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
                .unwrap()
        };
        let checkpoint = || {
            CheckpointRuntimeSpec::new(
                backend.clone(),
                directory.path().join("manifests"),
                StreamRuntimeConfig {
                    checkpoint_interval: StdDuration::from_secs(3_600),
                    checkpoint_timeout: StdDuration::from_secs(10),
                    ..StreamRuntimeConfig::default()
                },
            )
            .unwrap()
        };
        let log = Arc::new(Mutex::new(Vec::new()));
        let source_closed = Arc::new(AtomicUsize::new(0));
        let sink_closed = Arc::new(AtomicUsize::new(0));
        let source_gate = Arc::new(Semaphore::new(0));
        let plan = terminal_plan();
        let spec = ContinuousJobSpec {
            context: StreamJobContext::new(
                906,
                plan.fingerprint(),
                JsonMap::new(),
                None,
                CancellationToken::new(),
            ),
            plan,
            sources: vec![NamedSourceBinding {
                binding_id: "input".into(),
                binding: SourceBinding::new(
                    Box::new(StressSource {
                        events: VecDeque::from([
                            (
                                Arc::clone(&source_gate),
                                Some(SourceEvent::Data {
                                    batch: one_row(7),
                                    cursor: Cursor::new("input", vec![1], JsonMap::new()).unwrap(),
                                }),
                            ),
                            (Arc::clone(&source_gate), None),
                        ]),
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
                binding: OrdinarySinkBinding::new_transactional(Box::new(PeriodicCheckpointSink {
                    log: Arc::clone(&log),
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
        let job = runner.start_checkpointed(spec, checkpoint()).await.unwrap();
        retention_failure_armed.store(true, Ordering::SeqCst);
        source_gate.add_permits(2);

        let failed = tokio::time::timeout(StdDuration::from_secs(1), job.wait())
            .await
            .expect("retention fault should terminate the job");

        assert_eq!(failed.state, ContinuousJobState::Failed);
        assert!(failed.errors.iter().any(|failure| {
            matches!(
                &failure.error,
                CalcFlowError::Internal { message } if message == "injected retention failure"
            )
        }));
        let failed_checkpoint = job.status().checkpoint.unwrap();
        assert_eq!(failed_checkpoint.current_epoch, Some(crate::Epoch::INITIAL));
        assert_eq!(
            failed_checkpoint.phase,
            Some(CheckpointPhase::SinksCommitted)
        );
        assert_eq!(
            failed_checkpoint.last_completed_epoch,
            Some(crate::Epoch::INITIAL)
        );
        assert_eq!(
            failed_checkpoint.failure_category,
            Some(CheckpointFailureCategory::Maintenance)
        );
        assert!(failed_checkpoint.elapsed.is_some());
        assert_eq!(failed_checkpoint.source_acknowledgements, 1);
        assert_eq!(failed_checkpoint.operator_acknowledgements, 1);
        assert_eq!(failed_checkpoint.sink_precommit_acknowledgements, 1);
        assert_eq!(failed_checkpoint.sink_commit_acknowledgements, 1);
        let failed_metrics = job.status().metrics.checkpoints;
        assert_eq!(failed_metrics.requested, 1);
        assert_eq!(failed_metrics.completed, 1);
        assert_eq!(failed_metrics.failed, 1);
        assert_eq!(failed_metrics.terminal_requested, 1);
        assert_eq!(failed_metrics.terminal_completed, 1);
        assert_eq!(failed_metrics.terminal_failed, 1);
        drop(job);
        runner.shutdown().await.unwrap();
        assert!(log.lock().contains(&"sink-commit:1".into()));
        assert!(!log.lock().iter().any(|entry| entry.contains("abort")));
        assert_eq!(source_closed.load(Ordering::SeqCst), 1);
        assert_eq!(sink_closed.load(Ordering::SeqCst), 1);

        log.lock().clear();
        let source_polls = Arc::new(AtomicUsize::new(0));
        let restored_source_closed = Arc::new(AtomicUsize::new(0));
        let restored_sink_closed = Arc::new(AtomicUsize::new(0));
        let plan = terminal_plan();
        let restored_spec = ContinuousJobSpec {
            context: StreamJobContext::new(
                907,
                plan.fingerprint(),
                JsonMap::new(),
                None,
                CancellationToken::new(),
            ),
            plan,
            sources: vec![NamedSourceBinding {
                binding_id: "input".into(),
                binding: counting_pending_binding(0, &source_polls, &restored_source_closed),
            }],
            sinks: vec![NamedSinkBinding {
                output_id: "output".into(),
                sink_id: "sink".into(),
                binding: OrdinarySinkBinding::new_transactional(Box::new(PeriodicCheckpointSink {
                    log: Arc::clone(&log),
                    closed: Arc::clone(&restored_sink_closed),
                })),
            }],
            edge_budget: EdgeBudget {
                max_rows: 1,
                max_bytes: 1 << 20,
            },
            delivery_mode: M2DeliveryMode::ProcessLocalOrdered,
        };
        let mut restored_runner = ContinuousRunner::new();
        let restored_job = restored_runner
            .start_checkpointed(restored_spec, checkpoint())
            .await
            .unwrap();
        let restored = tokio::time::timeout(StdDuration::from_millis(100), restored_job.wait())
            .await
            .expect("durable terminal epoch should recover after retention failure");
        assert_eq!(restored.state, ContinuousJobState::Completed);
        let restored_metrics = restored_job.status().metrics.checkpoints;
        assert_eq!(restored_metrics.requested, 0);
        assert_eq!(restored_metrics.sink_commit_retries, 1);
        drop(restored_job);
        restored_runner.shutdown().await.unwrap();
        assert_eq!(source_polls.load(Ordering::SeqCst), 0);
        assert_eq!(restored_source_closed.load(Ordering::SeqCst), 0);
        assert_eq!(restored_sink_closed.load(Ordering::SeqCst), 1);
        assert_eq!(&*log.lock(), &["sink-open", "sink-recover:1", "sink-close"]);
    }

    #[tokio::test]
    #[allow(
        clippy::too_many_lines,
        reason = "the recovery lifecycle is asserted as one ordered integration scenario"
    )]
    async fn checkpointed_runner_restores_operator_source_and_sink_before_polling() {
        let directory = tempfile::tempdir().unwrap();
        let backend = Arc::new(
            LocalStateBackend::new(directory.path().join("state"))
                .await
                .unwrap(),
        );
        let log = Arc::new(Mutex::new(Vec::new()));
        let operator = RecoveryProbeOperator {
            inputs: [Port::new("input", BatchKind::Table, true, None).unwrap()],
            outputs: [Port::new("output", BatchKind::Table, true, None).unwrap()],
            log: Arc::clone(&log),
        };
        let requirements = StreamRequirements {
            delivery: BTreeMap::from([("output".into(), crate::DeliveryGuarantee::ExactlyOnce)]),
        };
        let plan = PipelineBuilder::new("checkpoint-recovery")
            .unwrap()
            .add_checkpoint_capable_node("node", Box::new(operator) as Box<dyn StreamOperator>)
            .unwrap()
            .compile_stream(&UdfRegistry::new().snapshot(), &requirements)
            .unwrap();
        let config = StreamRuntimeConfig::default();
        let prepared = crate::runtime::streaming::progress::prepare_stream_job(
            plan.fingerprint(),
            &[crate::runtime::streaming::progress::SourceBindingSpec {
                descriptor: crate::runtime::streaming::progress::SourceDescriptor::new(
                    crate::runtime::streaming::progress::BindingIdentity::new("input").unwrap(),
                    crate::runtime::streaming::progress::DeclaredSchema::DynamicOrUnknown,
                    crate::runtime::streaming::progress::NativeWatermarkCapability::NeverEmits,
                    crate::runtime::streaming::progress::ReplayPositioningCapability::ExactPauseReportAndSeek,
                    None,
                )
                .with_delivery_and_bounds(true, 1, 1 << 20),
                watermark_policy: crate::runtime::streaming::progress::WatermarkPolicy::Disabled {
                    idle_timeout: None,
                },
            }],
            crate::runtime::streaming::progress::StreamProgressRuntimeConfig::default(),
        )
        .unwrap();
        let manifest = crate::CheckpointManifest::new(CheckpointManifestFields {
            pipeline_name: plan.name().into(),
            pipeline_fingerprint: plan.fingerprint().into(),
            runtime_config_hash: plan.runtime_config_hash(&config).unwrap(),
            epoch: crate::Epoch::INITIAL,
            created_at: chrono::Utc.with_ymd_and_hms(2026, 8, 9, 9, 0, 0).unwrap(),
            recovery_status: RecoveryStatus::Final,
            sources: BTreeMap::from([(
                "input".into(),
                SourceManifestEntry {
                    cursor: Some(CursorManifestEntry {
                        order: "09".into(),
                        payload: BTreeMap::from([("offset".into(), serde_json::json!(9))]),
                    }),
                    identity_hash: prepared.bindings[0].identity_hash(),
                    sequence: 12,
                    ended: false,
                    watermark_policy: SourceWatermarkManifestState::Disabled { idle: true },
                },
            )]),
            operators: BTreeMap::from([(
                "node".into(),
                OperatorManifestEntry {
                    progress: BTreeMap::from([(
                        "input".into(),
                        OperatorIngressManifestEntry {
                            state: ManifestIngressState::Active,
                            watermark: None,
                        },
                    )]),
                    inline_metadata:
                        crate::pipeline::OperatorCheckpointCapability::CheckpointedStateful {
                            state_version: 1,
                        }
                        .encode_snapshot(
                            "node",
                            crate::OperatorStateSnapshot {
                                inline_metadata: BTreeMap::from([(
                                    "restored".into(),
                                    serde_json::json!(true),
                                )]),
                                segments: BTreeMap::new(),
                            },
                        )
                        .unwrap()
                        .inline_metadata,
                    segments: Vec::new(),
                },
            )]),
            sinks: BTreeMap::from([(
                "sink".into(),
                SinkManifestEntry {
                    delivery: SinkDeliveryManifest::Transactional,
                    pre_commit: Some(JsonMap::new()),
                },
            )]),
        })
        .unwrap();
        let key = StateLineageKey::new(plan.name(), plan.fingerprint()).unwrap();
        let lineage = backend.open_lineage(&key).await.unwrap();
        let transaction = crate::state::ManifestTransaction::open(
            Arc::from(lineage),
            &key,
            directory.path().join("manifests"),
            config.retained_epochs,
        )
        .await
        .unwrap();
        transaction
            .publish(crate::state::PreparedEpochManifest {
                manifest,
                staged_segments: BTreeMap::new(),
            })
            .await
            .unwrap();
        drop(transaction);

        let source_closed = Arc::new(AtomicUsize::new(0));
        let sink_closed = Arc::new(AtomicUsize::new(0));
        let spec = ContinuousJobSpec {
            context: StreamJobContext::new(
                902,
                plan.fingerprint(),
                JsonMap::new(),
                None,
                CancellationToken::new(),
            ),
            plan,
            sources: vec![NamedSourceBinding {
                binding_id: "input".into(),
                binding: SourceBinding::new(
                    Box::new(RecoveryProbeSource {
                        log: Arc::clone(&log),
                        closed: Arc::clone(&source_closed),
                    }),
                    None,
                    0,
                )
                .unwrap()
                .with_watermark_policy(
                    crate::runtime::streaming::progress::WatermarkPolicy::Disabled {
                        idle_timeout: None,
                    },
                ),
            }],
            sinks: vec![NamedSinkBinding {
                output_id: "output".into(),
                sink_id: "sink".into(),
                binding: OrdinarySinkBinding::new_transactional(Box::new(RecoveryProbeSink {
                    log: Arc::clone(&log),
                    closed: Arc::clone(&sink_closed),
                })),
            }],
            edge_budget: EdgeBudget {
                max_rows: 1,
                max_bytes: 1 << 20,
            },
            delivery_mode: M2DeliveryMode::ProcessLocalOrdered,
        };
        let checkpoint = CheckpointRuntimeSpec::managed(
            ManagedCheckpointRuntime::new(directory.path()).unwrap(),
            config,
        )
        .unwrap();
        let mut runner = ContinuousRunner::new();

        let job = runner.start_checkpointed(spec, checkpoint).await.unwrap();

        assert_eq!(
            &*log.lock(),
            &[
                "operator-reset",
                "operator-restore",
                "source-open:09",
                "sink-open",
                "sink-recover:1",
            ]
        );
        assert_eq!(job.status().sources["input"].next_sequence, Some(12));
        let progress = job.status().progress.unwrap();
        assert!(matches!(
            progress.current.bindings
                [&crate::runtime::streaming::progress::BindingIdentity::new("input").unwrap()]
                .activity,
            crate::runtime::streaming::progress::aggregate::IngressActivity::Idle {
                watermark: None
            }
        ));
        assert_eq!(progress.current.counters.trace_records, 0);
        let outcome = job.cancel().await;
        assert_eq!(outcome.state, ContinuousJobState::Cancelled);
        drop(job);
        runner.shutdown().await.unwrap();
        assert_eq!(source_closed.load(Ordering::SeqCst), 1);
        assert_eq!(sink_closed.load(Ordering::SeqCst), 1);
    }

    #[tokio::test(start_paused = true)]
    #[allow(
        clippy::too_many_lines,
        reason = "mixed ended/live restore owns the manifest, connector probes, and periodic cut"
    )]
    async fn restored_ended_source_participates_without_open_seek_poll_or_barrier() {
        let directory = tempfile::tempdir().unwrap();
        let backend = Arc::new(
            LocalStateBackend::new(directory.path().join("state"))
                .await
                .unwrap(),
        );
        let union = UnionOperator::new(
            "merge",
            vec![
                Port::new("ended", BatchKind::Table, true, None).unwrap(),
                Port::new("live", BatchKind::Table, true, None).unwrap(),
            ],
        )
        .unwrap();
        let plan = PipelineBuilder::new("checkpoint-mixed-source-restore")
            .unwrap()
            .add_node("merge", Box::new(union))
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
        let config = StreamRuntimeConfig {
            checkpoint_interval: StdDuration::from_millis(10),
            checkpoint_timeout: StdDuration::from_secs(10),
            ..StreamRuntimeConfig::default()
        };
        let source_spec = |binding_id: &str| {
            crate::runtime::streaming::progress::SourceBindingSpec {
                descriptor: crate::runtime::streaming::progress::SourceDescriptor::new(
                    crate::runtime::streaming::progress::BindingIdentity::new(binding_id).unwrap(),
                    crate::runtime::streaming::progress::DeclaredSchema::DynamicOrUnknown,
                    crate::runtime::streaming::progress::NativeWatermarkCapability::NeverEmits,
                    crate::runtime::streaming::progress::ReplayPositioningCapability::ExactPauseReportAndSeek,
                    None,
                )
                .with_delivery_and_bounds(true, 1, 1 << 20),
                watermark_policy:
                    crate::runtime::streaming::progress::WatermarkPolicy::Disabled {
                        idle_timeout: None,
                    },
            }
        };
        let prepared = crate::runtime::streaming::progress::prepare_stream_job(
            plan.fingerprint(),
            &[source_spec("ended"), source_spec("live")],
            crate::runtime::streaming::progress::StreamProgressRuntimeConfig::default(),
        )
        .unwrap();
        let identity_hash = |binding_id: &str| {
            prepared
                .bindings
                .iter()
                .find(|binding| binding.identity.as_str() == binding_id)
                .unwrap()
                .identity_hash()
        };
        let manifest = crate::CheckpointManifest::new(CheckpointManifestFields {
            pipeline_name: plan.name().into(),
            pipeline_fingerprint: plan.fingerprint().into(),
            runtime_config_hash: plan.runtime_config_hash(&config).unwrap(),
            epoch: crate::Epoch::INITIAL,
            created_at: chrono::Utc.with_ymd_and_hms(2026, 8, 9, 9, 30, 0).unwrap(),
            recovery_status: RecoveryStatus::Final,
            sources: BTreeMap::from([
                (
                    "ended".into(),
                    SourceManifestEntry {
                        cursor: Some(CursorManifestEntry {
                            order: "01".into(),
                            payload: BTreeMap::new(),
                        }),
                        identity_hash: identity_hash("ended"),
                        sequence: 1,
                        ended: true,
                        watermark_policy: SourceWatermarkManifestState::Disabled { idle: false },
                    },
                ),
                (
                    "live".into(),
                    SourceManifestEntry {
                        cursor: Some(CursorManifestEntry {
                            order: "09".into(),
                            payload: BTreeMap::new(),
                        }),
                        identity_hash: identity_hash("live"),
                        sequence: 12,
                        ended: false,
                        watermark_policy: SourceWatermarkManifestState::Disabled { idle: false },
                    },
                ),
            ]),
            operators: BTreeMap::from([(
                "merge".into(),
                OperatorManifestEntry {
                    progress: BTreeMap::from([
                        (
                            "ended".into(),
                            OperatorIngressManifestEntry {
                                state: ManifestIngressState::Ended,
                                watermark: None,
                            },
                        ),
                        (
                            "live".into(),
                            OperatorIngressManifestEntry {
                                state: ManifestIngressState::Active,
                                watermark: None,
                            },
                        ),
                    ]),
                    inline_metadata: BTreeMap::new(),
                    segments: Vec::new(),
                },
            )]),
            sinks: BTreeMap::from([(
                "sink".into(),
                SinkManifestEntry {
                    delivery: SinkDeliveryManifest::Transactional,
                    pre_commit: Some(JsonMap::new()),
                },
            )]),
        })
        .unwrap();
        let key = StateLineageKey::new(plan.name(), plan.fingerprint()).unwrap();
        let lineage = backend.open_lineage(&key).await.unwrap();
        let transaction = crate::state::ManifestTransaction::open(
            Arc::from(lineage),
            &key,
            directory.path().join("manifests"),
            config.retained_epochs,
        )
        .await
        .unwrap();
        transaction
            .publish(crate::state::PreparedEpochManifest {
                manifest,
                staged_segments: BTreeMap::new(),
            })
            .await
            .unwrap();
        drop(transaction);

        let ended_opens = Arc::new(AtomicUsize::new(0));
        let ended_seeks = Arc::new(AtomicUsize::new(0));
        let ended_polls = Arc::new(AtomicUsize::new(0));
        let ended_closed = Arc::new(AtomicUsize::new(0));
        let live_opens = Arc::new(AtomicUsize::new(0));
        let live_seeks = Arc::new(AtomicUsize::new(0));
        let live_polls = Arc::new(AtomicUsize::new(0));
        let live_closed = Arc::new(AtomicUsize::new(0));
        let sink_closed = Arc::new(AtomicUsize::new(0));
        let sink_log = Arc::new(Mutex::new(Vec::new()));
        let source_binding = |opens: &Arc<AtomicUsize>,
                              seeks: &Arc<AtomicUsize>,
                              polls: &Arc<AtomicUsize>,
                              closed: &Arc<AtomicUsize>| {
            SourceBinding::new(
                Box::new(MixedRestoreSource {
                    opens: Arc::clone(opens),
                    seeks: Arc::clone(seeks),
                    polls: Arc::clone(polls),
                    closed: Arc::clone(closed),
                }),
                None,
                0,
            )
            .unwrap()
            .with_watermark_policy(
                crate::runtime::streaming::progress::WatermarkPolicy::Disabled {
                    idle_timeout: None,
                },
            )
        };
        let spec = ContinuousJobSpec {
            context: StreamJobContext::new(
                909,
                plan.fingerprint(),
                JsonMap::new(),
                None,
                CancellationToken::new(),
            ),
            plan,
            sources: vec![
                NamedSourceBinding {
                    binding_id: "ended".into(),
                    binding: source_binding(
                        &ended_opens,
                        &ended_seeks,
                        &ended_polls,
                        &ended_closed,
                    ),
                },
                NamedSourceBinding {
                    binding_id: "live".into(),
                    binding: source_binding(&live_opens, &live_seeks, &live_polls, &live_closed),
                },
            ],
            sinks: vec![NamedSinkBinding {
                output_id: "output".into(),
                sink_id: "sink".into(),
                binding: OrdinarySinkBinding::new_transactional(Box::new(PeriodicCheckpointSink {
                    log: Arc::clone(&sink_log),
                    closed: Arc::clone(&sink_closed),
                })),
            }],
            edge_budget: EdgeBudget {
                max_rows: 1,
                max_bytes: 1 << 20,
            },
            delivery_mode: M2DeliveryMode::ProcessLocalOrdered,
        };
        let checkpoint =
            CheckpointRuntimeSpec::new(backend, directory.path().join("manifests"), config)
                .unwrap();
        let mut runner = ContinuousRunner::new();

        let job = runner.start_checkpointed(spec, checkpoint).await.unwrap();
        wait_for_counter(&live_polls, 1).await;
        tokio::time::advance(StdDuration::from_millis(20)).await;
        tokio::time::timeout(StdDuration::from_secs(5), async {
            loop {
                if sink_log.lock().iter().any(|entry| entry == "sink-begin:3") {
                    break;
                }
                tokio::task::yield_now().await;
            }
        })
        .await
        .expect("mixed-source restored checkpoint did not complete");

        assert_eq!(ended_opens.load(Ordering::SeqCst), 0);
        assert_eq!(ended_seeks.load(Ordering::SeqCst), 0);
        assert_eq!(ended_polls.load(Ordering::SeqCst), 0);
        assert_eq!(ended_closed.load(Ordering::SeqCst), 0);
        assert_eq!(live_opens.load(Ordering::SeqCst), 1);
        assert_eq!(live_seeks.load(Ordering::SeqCst), 1);
        assert_eq!(job.status().sources["live"].next_sequence, Some(12));
        assert_eq!(
            job.status().checkpoint.unwrap().last_completed_epoch,
            Some(crate::Epoch::new(2).unwrap())
        );
        let outcome = job.cancel().await;
        assert_eq!(outcome.state, ContinuousJobState::Cancelled);
        drop(job);
        runner.shutdown().await.unwrap();
        assert_eq!(live_closed.load(Ordering::SeqCst), 1);
        assert_eq!(sink_closed.load(Ordering::SeqCst), 1);
    }

    #[test]
    fn checkpoint_fault_matrix_enumerates_every_durable_boundary_and_mode() {
        use super::{CheckpointFaultInjector, CheckpointFaultMode, CheckpointFaultPoint};

        assert_eq!(
            CheckpointFaultPoint::ALL,
            [
                CheckpointFaultPoint::SourceAdmission,
                CheckpointFaultPoint::SourceCut,
                CheckpointFaultPoint::PartialAlignment,
                CheckpointFaultPoint::StateStage,
                CheckpointFaultPoint::SinkPreCommit,
                CheckpointFaultPoint::ManifestWrite,
                CheckpointFaultPoint::ManifestRename,
                CheckpointFaultPoint::ManifestParentSync,
                CheckpointFaultPoint::PartialSinkCommit,
                CheckpointFaultPoint::CompletedCommit,
                CheckpointFaultPoint::Retention,
                CheckpointFaultPoint::Compaction,
            ]
        );
        assert_eq!(
            CheckpointFaultMode::ALL,
            [
                CheckpointFaultMode::Io,
                CheckpointFaultMode::Panic,
                CheckpointFaultMode::Cancel,
                CheckpointFaultMode::Restart,
            ]
        );

        let cancellation = CancellationToken::new();
        let injector = CheckpointFaultInjector::armed(
            CheckpointFaultPoint::SourceCut,
            CheckpointFaultMode::Io,
        );
        assert!(
            injector
                .trigger(CheckpointFaultPoint::SourceAdmission, &cancellation)
                .is_ok()
        );
        assert!(
            injector
                .trigger(CheckpointFaultPoint::SourceCut, &cancellation)
                .is_err()
        );
        assert!(
            injector
                .trigger(CheckpointFaultPoint::SourceCut, &cancellation)
                .is_ok()
        );
        assert_eq!(injector.trigger_count(), 1);

        let cancellation = CancellationToken::new();
        let injector = CheckpointFaultInjector::armed(
            CheckpointFaultPoint::SourceCut,
            CheckpointFaultMode::Cancel,
        );
        assert!(
            injector
                .trigger(CheckpointFaultPoint::SourceCut, &cancellation)
                .is_ok()
        );
        assert!(cancellation.is_cancelled());
        assert_eq!(injector.trigger_count(), 1);
    }
}
