mod asof;
mod checkpoint_task;
pub(super) mod operator_fusion;
mod sql_recovery;
mod supervision;

#[cfg(test)]
use checkpoint_task::{
    DurableSettlementPhase, DurableSettlementRequest, maybe_request_terminal_checkpoint,
    notify_sink_abort, notify_sink_manifest_durable, settle_durable_manifest,
    source_cuts_are_terminal,
};
use checkpoint_task::{LiveCheckpointTaskInputs, run_live_checkpoint_task};
use supervision::{SupervisionHome, SupervisorLoan};
#[cfg(test)]
mod entity_work_tests;
#[cfg(test)]
mod supervision_loan_tests;

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
use futures::{FutureExt, future::try_join_all};
use parking_lot::Mutex;
use serde::Serialize;
use sha2::{Digest as _, Sha256};
use tokio::{
    sync::{Notify, mpsc, watch},
    task::JoinSet,
};

use super::{
    ChannelMetrics, EdgeReceiver, EdgeSender,
    channel::edge_channel_with_metrics,
    checkpoint::{
        ManagedCheckpointRuntime,
        coordinator::{
            CheckpointAck, CheckpointCoordinatorHandle, CheckpointEvent, CheckpointPhase,
            CheckpointRequest, ManualCheckpointFailure, ManualCheckpointFailureCategory,
            ParticipantSet, spawn_checkpoint_coordinator,
        },
    },
    entity_work::JobEntityWorkOwner,
    job::{
        ContinuousJobSpec, OrdinarySinkBinding, OwningContinuousJob, StableSinkId,
        ValidatedContinuousJob, ValidatedOrdinarySink, preflight_job,
    },
    local_edge::{LocalEdgeOwner, OperatorEdgeReceiver, OperatorEdgeSender, local_edge},
    metrics::{M2MetricsSnapshot, MetricsRecorder, MetricsTimer, sink_metric_id},
    operator_task::{
        OperatorCheckpointAck, OperatorCheckpointCommand, OperatorCheckpointPort, OperatorIngress,
        OperatorProgress, OperatorProgressSnapshot, OperatorRestoreState, OperatorTaskInputs,
        OperatorTerminalPort, prepare_operator_task_group,
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
    sql_recovery_work::{JobSqlRecoveryOwner, SqlRecoveryClient},
    supervisor::{
        SupervisionReport, TaskFailure, TaskId, TaskRegistry, TaskStatus, TaskSupervisor,
        terminal::{TerminalArbiter, TerminalDecision},
    },
};
use crate::operator::rolling_metrics::RollingMetricsStore;
use crate::pipeline::{
    OperatorCheckpointCapability, RuntimeSinkRoute, RuntimeSourceRoute, RuntimeStreamNode,
    StreamRuntimePlanParts,
};
#[cfg(test)]
use crate::state::ManifestTransactionFaultPoint;
use crate::{
    CalcFlowError, CancellationToken, CheckpointManifest, CheckpointManifestFields, Epoch,
    EventTime, OperatorManifestEntry, RecoveryStatus, SinkManifestEntry, SourceManifestEntry,
    StateBackend, StateLineageKey, StreamRuntimeConfig,
    state::{
        ManifestPublication, ManifestTransaction, PreparedEpochManifest, PreparedManifestIdentity,
        SelectedManifest,
    },
};

pub(crate) use super::failure::{
    ContinuousJobOutcome, ContinuousJobState, DriverOwnership, FailureOrigin, LaunchDeliveryState,
    LaunchId, RuntimeFailure, StartFailure, StartResult, TerminalCause, runner_shutdown_failure,
};
use super::failure::{classify_failure_state, panic_message};

#[cfg(all(test, unix))]
pub(crate) use super::test_seams::configure_test_manifest_transaction;
#[cfg(test)]
pub(crate) use super::test_seams::{
    CheckpointFaultInjector, CheckpointFaultMode, CheckpointFaultPoint, CheckpointStartedTestGate,
    TerminalCommitTestSeam, TestLaunchCheckpoint, TestLaunchProbe,
};

const CONNECTOR_OPEN_TIMEOUT: Duration = Duration::from_secs(30);
const CONNECTOR_OPEN_SETTLE_TIMEOUT: Duration = Duration::from_secs(35);
const CONNECTOR_CLOSE_TIMEOUT: Duration = Duration::from_secs(5);
#[cfg(test)]
use super::registry::ABANDONED_RUNNER_WARNING;
use super::registry::{RunnerCommand, RunnerCore, RunnerRegistryState};

pub(crate) use super::checkpoint_runtime::CheckpointRuntimeSpec;
use super::checkpoint_runtime::{
    CheckpointRuntimeStorage, OpenedCheckpointRuntime, ValidatedCheckpointRuntime,
};

use super::checkpoint_status::checkpoint_protocol_error;
pub(crate) use super::checkpoint_status::{
    CheckpointFailureCategory, CheckpointStatus, CheckpointStatusHandle,
};

struct JobCoreState {
    owner: DriverOwnership,
    launch_delivery: LaunchDeliveryState,
    state: ContinuousJobState,
    selected_cause: Option<TerminalCause>,
    outcome: Option<Arc<ContinuousJobOutcome>>,
    start_failure: Option<StartFailure>,
}

pub(super) struct JobCore {
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
    entity_work: JobEntityWorkOwner,
    sql_recovery: JobSqlRecoveryOwner,
    gather_work: super::gather_work::JobGatherOwner,
    asof_loads: asof::LoadOwner,
    supervision: SupervisionHome,
    #[cfg(test)]
    owned_lane_launches: Arc<AtomicU64>,
    #[cfg(test)]
    driver_abort: Mutex<Option<tokio::task::AbortHandle>>,
    #[cfg(test)]
    panic_after_prepared_report: AtomicBool,
    #[cfg(test)]
    report_publication_gate: Mutex<Option<Arc<DriverReportGate>>>,
    #[cfg(test)]
    terminal_commit_seam: Mutex<Option<TerminalCommitTestSeam>>,
    #[cfg(test)]
    launch_probe: Option<Arc<TestLaunchProbe>>,
}

#[derive(Default)]
struct RuntimeStatus {
    tasks: TaskRegistry,
    sources: BTreeMap<String, SourceProgress>,
    nodes: BTreeMap<String, OperatorProgress>,
    rolling_metrics: BTreeMap<String, RollingMetricsStore>,
    sinks: BTreeMap<String, SinkProgress>,
    sink_outputs: BTreeMap<String, String>,
    progress: Option<LiveProgressStatusHandle>,
    checkpoint: Option<CheckpointStatusHandle>,
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
        #[cfg(test)]
        let owned_lane_launches = Arc::new(AtomicU64::new(0));
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
            entity_work: JobEntityWorkOwner::new(
                job_id,
                #[cfg(test)]
                owned_lane_launches.clone(),
            ),
            sql_recovery: JobSqlRecoveryOwner::new(),
            gather_work: super::gather_work::JobGatherOwner::new(job_id.to_string().into()),
            asof_loads: asof::LoadOwner::default(),
            supervision: SupervisionHome::default(),
            #[cfg(test)]
            owned_lane_launches,
            #[cfg(test)]
            driver_abort: Mutex::new(None),
            #[cfg(test)]
            panic_after_prepared_report: AtomicBool::new(false),
            #[cfg(test)]
            report_publication_gate: Mutex::new(None),
            #[cfg(test)]
            terminal_commit_seam: Mutex::new(None),
            #[cfg(test)]
            launch_probe: None,
        }
    }

    fn prepare_driver_report(&self, report: DriverReport) -> LaunchId {
        self.supervision.prepare(report);
        #[cfg(test)]
        assert!(
            !self.panic_after_prepared_report.load(Ordering::SeqCst),
            "injected panic after owned report preparation"
        );
        self.launch_id
    }

    #[cfg(test)]
    fn abort_driver_for_test(&self) {
        self.driver_abort
            .lock()
            .as_ref()
            .expect("registered job driver")
            .abort();
    }

    #[cfg(test)]
    fn panic_after_prepared_report_for_test(&self) {
        self.panic_after_prepared_report
            .store(true, Ordering::SeqCst);
    }

    #[cfg(test)]
    fn pause_report_publication_for_test(&self) -> Arc<DriverReportGate> {
        let gate = Arc::new(DriverReportGate::default());
        *self.report_publication_gate.lock() = Some(gate.clone());
        gate
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

#[derive(Clone, Debug, Default, Eq, PartialEq)]
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
    pub(crate) stream_join: Option<crate::StreamJoinStatus>,
    pub(crate) stream_asof_join: Option<crate::StreamAsofJoinStatus>,
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
            stream_join: progress.stream_join,
            stream_asof_join: progress.stream_asof_join,
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

    pub(crate) fn rolling_metrics(&self) -> BTreeMap<String, crate::RollingMetrics> {
        self.core
            .runtime_status
            .lock()
            .rolling_metrics
            .iter()
            .map(|(id, store)| (id.clone(), store.snapshot()))
            .collect()
    }

    /// Collects the payload-free status of every Join node, keyed by node ID.
    pub(crate) fn stream_join_status(&self) -> BTreeMap<String, crate::StreamJoinStatus> {
        let runtime = self.core.runtime_status.lock();
        runtime
            .nodes
            .iter()
            .filter_map(|(id, progress)| {
                progress
                    .snapshot()
                    .stream_join
                    .map(|status| (id.clone(), status))
            })
            .collect()
    }

    /// Collects independent ASOF diagnostics without changing inner Join status.
    pub(crate) fn stream_asof_join_status(&self) -> BTreeMap<String, crate::StreamAsofJoinStatus> {
        self.core
            .runtime_status
            .lock()
            .nodes
            .iter()
            .filter_map(|(id, progress)| {
                progress
                    .snapshot()
                    .stream_asof_join
                    .map(|status| (id.clone(), status))
            })
            .collect()
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
pub(super) struct RunnerDiagnostics(Mutex<RunnerDiagnosticsState>);

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

    pub(crate) fn cleanup_observer(&self) -> RunnerShutdownObserver {
        RunnerShutdownObserver::new(Arc::clone(&self.runner.core))
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
        mut spec: ContinuousJobSpec,
        checkpoint: Option<CheckpointRuntimeSpec>,
    ) -> StartObserver {
        #[cfg(test)]
        let launch_probe = self.core.next_launch_probe.lock().take();
        if let Some(budget) = checkpoint
            .as_ref()
            .and_then(|checkpoint| checkpoint.config.sql_state_budget)
        {
            if let Err(error) = spec.plan.set_sql_state_budget(Some(budget)) {
                return preflight_error_observer(error);
            }
        }
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
        let launch_id = LaunchId::new(launch_id);
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
        static_inputs: job.static_inputs.digests.clone(),
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
                    let driver = drivers.spawn(run_job_driver(launch_id, Arc::clone(&job_core), *job, checkpoint.map(|checkpoint| *checkpoint)));
                    #[cfg(test)]
                    { *job_core.driver_abort.lock() = Some(driver.clone()); }
                    drop(driver);
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
                    let launch_id = active.expect("an active driver owns the join");
                    let job_core = runner_core.registry.lock().live_jobs[&launch_id].clone();
                    let join_error = joined.err().map(|error| error.to_string());
                    let report = settle_driver_report(&job_core, join_error.as_deref()).await;
                    publish_driver_report(&runner_core, report);
                    active = None;
                    if runner_core.stop_after_first_job {
                        shutting_down = true;
                        begin_lifecycle_shutdown(&runner_core, &mut pending);
                    } else if !shutting_down && let Some(next) = pending.take() {
                        active = Some(next.launch_id);
                        next.core.state.lock().owner = DriverOwnership::Driving;
                        let driver = drivers.spawn(run_job_driver(
                            next.launch_id,
                            Arc::clone(&next.core),
                            *next.job,
                            next.checkpoint.map(|checkpoint| *checkpoint),
                        ));
                        #[cfg(test)]
                        { *next.core.driver_abort.lock() = Some(driver.clone()); }
                        drop(driver);
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

#[cfg(test)]
#[derive(Default)]
struct DriverReportGate {
    entered: Notify,
    release: Notify,
}

async fn settle_driver_report(core: &Arc<JobCore>, join_error: Option<&str>) -> DriverReport {
    if let Some(error) = core.asof_loads.close_and_drain().await {
        core.supervision
            .append_cleanup(vec![Arc::new(RuntimeFailure {
                origin: FailureOrigin::Preflight,
                error,
            })]);
    }
    if let Some(mut loan) = core.supervision.take(
        core.entity_work.clone(),
        core.sql_recovery.clone(),
        core.gather_work.clone(),
    ) {
        let report = loan.join_all().await;
        if !core.supervision.has_report() {
            core.supervision
                .prepare(unprepared_driver_report(core, report, join_error));
        }
        drop(loan);
    }
    core.sql_recovery.close_admission();
    let sql_failures = core.sql_recovery.drain().await;
    core.supervision.append_cleanup(sql_failures);
    if !core.supervision.has_report() {
        core.entity_work.close_admission();
        let mut secondary = core.entity_work.drain().await;
        secondary.extend(core.gather_work.close_and_drain().await);
        let mut failed = unprepared_driver_report(
            core,
            SupervisionReport {
                primary_error_count: 0,
                errors: Vec::new(),
            },
            join_error,
        );
        failed
            .cleanup_failures
            .extend(secondary.into_iter().map(task_runtime_failure));
        core.supervision.prepare(failed);
    }
    core.gather_work.close_admission();
    let gather_secondary = core.gather_work.close_and_drain().await;
    if !gather_secondary.is_empty() {
        core.supervision.append_secondary(gather_secondary);
    }
    #[cfg(test)]
    {
        let gate = core.report_publication_gate.lock().clone();
        if let Some(gate) = gate {
            gate.entered.notify_one();
            gate.release.notified().await;
        }
    }
    core.supervision.clear_joined();
    core.supervision
        .take_report()
        .expect("the lifecycle publishes one owned report")
}

fn unprepared_driver_report(
    core: &JobCore,
    report: SupervisionReport,
    join_error: Option<&str>,
) -> DriverReport {
    let progress = {
        let status = core.runtime_status.lock();
        RuntimeTaskProgress {
            sources: status.sources.clone(),
            sinks: status.sinks.clone(),
        }
    };
    let selected = core.state.lock().selected_cause.clone();
    if report.primary_error_count > 0
        || matches!(
            selected,
            Some(TerminalCause::ExplicitCancel | TerminalCause::DeadlineExceeded)
        )
    {
        return finish_running_report(
            core.launch_id,
            selected,
            false,
            report,
            &progress,
            &core.metrics,
        );
    }
    let claimed = {
        let mut state = core.state.lock();
        if state.launch_delivery == LaunchDeliveryState::Claimed {
            true
        } else {
            // Seal handle delivery under the same lock used by StartObserver.
            state.launch_delivery = LaunchDeliveryState::Finalizing;
            false
        }
    };
    if claimed {
        failed_running_driver_report(
            core,
            report,
            &progress,
            join_error.unwrap_or("missing prepared driver report"),
        )
    } else {
        let mut failed = DriverReport::aborted(
            core.launch_id,
            join_error.unwrap_or("missing prepared driver report"),
        );
        failed.cleanup_failures.extend(runtime_failures(
            report,
            &progress.sources,
            &progress.sinks,
        ));
        failed
    }
}

fn failed_running_driver_report(
    core: &JobCore,
    report: SupervisionReport,
    progress: &RuntimeTaskProgress,
    message: &str,
) -> DriverReport {
    let mut errors = vec![Arc::new(RuntimeFailure {
        origin: FailureOrigin::RunnerLifecycle,
        error: CalcFlowError::Internal {
            message: format!("job driver join failed: {message}"),
        },
    })];
    errors.extend(runtime_failures(report, &progress.sources, &progress.sinks));
    let state = if errors
        .iter()
        .any(|failure| matches!(failure.error, CalcFlowError::RecoveryRequired { .. }))
    {
        ContinuousJobState::RecoveryRequired
    } else {
        ContinuousJobState::Failed
    };
    let cause = TerminalCause::RunnerFailure;
    core.metrics.record_terminal(state, cause.clone());
    if let Some(overflow) = core.metrics.account_terminal_errors_once(&errors) {
        errors.push(overflow);
    }
    DriverReport {
        launch_id: core.launch_id,
        completion: DriverCompletion::Outcome(Arc::new(ContinuousJobOutcome {
            state,
            cause,
            errors,
        })),
        cleanup_failures: Vec::new(),
    }
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

    async fn settle_open(&mut self) -> crate::Result<()> {
        match self {
            Self::Source { .. } => Ok(()),
            Self::Sink { binding, .. } => binding.settle_open().await,
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
) -> LaunchId {
    let ValidatedContinuousJob {
        context,
        mut plan,
        mut sources,
        sinks,
        progress: prepared_progress,
        delivery_mode: _,
        delivery_proofs: _,
        static_inputs: _,
    } = validated;
    if let Err(error) = core.sql_recovery.configure(
        plan.nodes
            .iter()
            .filter(|node| {
                matches!(
                    node.operator,
                    crate::pipeline::CompiledStreamOperator::Sql(_)
                )
            })
            .count(),
    ) {
        return core.prepare_driver_report(checkpoint_start_failure(launch_id, error));
    }
    let context = context.with_gather_owner(core.gather_work.clone());
    core.runtime_status.lock().rolling_metrics = plan
        .nodes
        .iter()
        .filter(|node| {
            matches!(
                &node.operator,
                crate::pipeline::CompiledStreamOperator::Rolling(_)
            )
        })
        .map(|node| {
            (
                node.operator_id.as_str().to_owned(),
                RollingMetricsStore::default(),
            )
        })
        .collect();
    let cancellation = context.cancellation().clone();
    let checkpoint = match checkpoint {
        Some(checkpoint) => match open_checkpoint_runtime(checkpoint, &cancellation).await {
            Ok(checkpoint) => {
                if let Err(error) = core
                    .metrics
                    .record_checkpoint_orphan_cleanup(checkpoint.startup_orphans_removed)
                {
                    return core.prepare_driver_report(checkpoint_start_failure(launch_id, error));
                }
                Some(checkpoint)
            }
            Err(error) => {
                return core.prepare_driver_report(DriverReport {
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
                });
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
            return core.prepare_driver_report(checkpoint_start_failure(launch_id, error));
        }
        match manifest_is_terminal(&selected.manifest, &plan) {
            Ok(true) => {
                let mut sql_frames = sql_recovery::TerminalSqlFrames::default();
                match sql_recovery::restore_terminal(
                    &mut plan,
                    &mut sql_frames,
                    checkpoint,
                    &context,
                    &core.sql_recovery,
                    &core.launch_cancel,
                )
                .await
                {
                    Ok(nodes) => core.runtime_status.lock().nodes.extend(nodes),
                    Err(error) => {
                        return core.prepare_driver_report(checkpoint_start_failure(
                            launch_id,
                            sanitize_managed_recovery_error(error, checkpoint.managed),
                        ));
                    }
                }
                let restored = asof::restore_terminal(
                    &mut plan,
                    checkpoint,
                    &prepared_progress,
                    &core.asof_loads,
                    &cancellation,
                    &context,
                )
                .await;
                match restored {
                    Ok(nodes) => core.runtime_status.lock().nodes.extend(nodes),
                    Err(error) => {
                        return core.prepare_driver_report(checkpoint_start_failure(
                            launch_id,
                            sanitize_managed_recovery_error(error, checkpoint.managed),
                        ));
                    }
                }
                drop(sources);
                return core.prepare_driver_report(
                    recover_terminal_manifest(
                        launch_id,
                        &core,
                        &checkpoint.transaction,
                        &checkpoint.identity,
                        selected,
                        sinks,
                        checkpoint.managed,
                    )
                    .await,
                );
            }
            Ok(false) => {}
            Err(error) => {
                return core.prepare_driver_report(checkpoint_start_failure(
                    launch_id,
                    sanitize_managed_recovery_error(error, checkpoint.managed),
                ));
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
                &core.asof_loads,
                &cancellation,
            )
            .await
            {
                Ok(restored) => restored,
                Err(error) => {
                    return core.prepare_driver_report(checkpoint_start_failure(
                        launch_id,
                        sanitize_managed_recovery_error(error, checkpoint.managed),
                    ));
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
            return core.prepare_driver_report(DriverReport {
                launch_id,
                completion: DriverCompletion::StartFailed(StartFailure {
                    primary: sanitize_managed_recovery_failure(primary, managed_recovery),
                    diagnostic_id: None,
                    cleanup_failures: Vec::new(),
                }),
                cleanup_failures: Vec::new(),
            });
        }
        Err(EntryFailure::Cancelled) => {
            return core.prepare_driver_report(cancelled_driver_report(launch_id, &core.metrics));
        }
    };
    if let Some(report) = cancel_after_operator_entry(launch_id, &core, &mut runtime).await {
        return core.prepare_driver_report(report);
    }
    let mut resources = connector_resources(sources, sinks);
    let open_failures = if checkpoint.is_some() {
        open_checkpoint_connector_resources(&mut resources, &core.launch_cancel).await
    } else {
        open_connector_resources(&mut resources, &core.launch_cancel).await
    };
    if !open_failures.is_empty() {
        return core.prepare_driver_report(
            finish_failed_launch(
                launch_id,
                open_failures,
                &mut resources,
                &mut runtime.supervisor,
            )
            .await,
        );
    }
    if core.launch_cancel.is_cancelled() {
        return core.prepare_driver_report(
            finish_cancelled_launch(
                launch_id,
                &mut resources,
                &mut runtime.supervisor,
                &core.metrics,
            )
            .await,
        );
    }

    let (opened_sources, opened_sinks) = opened_connector_bindings(std::mem::take(&mut resources));
    let mut opened_sinks = opened_sinks;
    if let Some(checkpoint) = checkpoint.as_ref()
        && let Some(selected) = &checkpoint.selected
    {
        let recovery = recover_opened_sinks(
            &mut opened_sinks,
            &selected.manifest,
            &checkpoint.transaction,
            &core.launch_cancel,
        )
        .await;
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
            return core.prepare_driver_report(
                finish_failed_launch(
                    launch_id,
                    vec![Arc::new(RuntimeFailure {
                        origin: FailureOrigin::Preflight,
                        error,
                    })],
                    &mut resources,
                    &mut runtime.supervisor,
                )
                .await,
            );
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
        return core.prepare_driver_report(cancelled_driver_report_with_task_cleanup(
            launch_id,
            report,
            &task_progress,
            &core.metrics,
        ));
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
    if let Err(error) = recover_opened_sinks(
        &mut opened_sinks,
        &selected.manifest,
        transaction,
        &core.launch_cancel,
    )
    .await
    {
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
            BTreeMap::from([("state".into(), crate::StateSegment::new(Vec::new()))])
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
    loads: &asof::LoadOwner,
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
        .map(|node| {
            (
                node.operator_id.as_str(),
                (
                    &node.operator,
                    node.checkpoint_capability,
                    node.operator.requires_output_frontier_state(),
                ),
            )
        })
        .collect::<BTreeMap<_, _>>();
    let mut operators = BTreeMap::new();
    for (operator_id, entry) in selected.manifest.operators() {
        let &(operator, checkpoint_capability, requires_output_frontier) = checkpoint_capabilities
            .get(operator_id.as_str())
            .ok_or_else(|| CalcFlowError::CheckpointMismatch {
                message: format!(
                    "checkpoint operator {operator_id:?} is absent from the prepared plan"
                ),
            })?;
        let snapshot = asof::load_snapshot(
            &checkpoint.transaction,
            loads,
            operator,
            operator_id,
            entry,
            cancellation,
        )
        .await?;
        let mut snapshot = checkpoint_capability.decode_snapshot(operator_id, snapshot)?;
        let restored_output_frontier = snapshot
            .inline_metadata
            .remove(crate::pipeline::OUTPUT_FRONTIER_METADATA_KEY_V1);
        let output_frontier = match (requires_output_frontier, restored_output_frontier) {
            (true, Some(serde_json::Value::Null)) | (false, None) => None,
            (true, Some(value)) => {
                Some(value.as_i64().map(EventTime::from_micros).ok_or_else(|| {
                    CalcFlowError::CheckpointMismatch {
                        message: format!(
                            "checkpoint operator {operator_id:?} has an invalid output frontier"
                        ),
                    }
                })?)
            }
            (true, None) => {
                return Err(CalcFlowError::CheckpointMismatch {
                    message: format!(
                        "checkpoint operator {operator_id:?} is missing its output frontier"
                    ),
                });
            }
            (false, Some(_)) => {
                return Err(CalcFlowError::CheckpointMismatch {
                    message: format!(
                        "checkpoint operator {operator_id:?} unexpectedly contains an output frontier"
                    ),
                });
            }
        };
        operators.insert(
            operator_id.clone(),
            OperatorRestoreState {
                snapshot,
                progress: entry.progress.clone(),
                output_frontier,
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
    transaction: &ManifestTransaction,
    cancellation: &CancellationToken,
) -> crate::Result<()> {
    for output_sinks in sinks.values_mut() {
        super::sink_task::recover_transactional_sinks_with_state(
            output_sinks,
            manifest,
            transaction,
            cancellation,
        )
        .await?;
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

fn sanitize_managed_preflight_error(
    error: CalcFlowError,
    managed: bool,
    manifest_candidate: bool,
) -> CalcFlowError {
    if !managed {
        return error;
    }
    if let CalcFlowError::CheckpointMismatch { message } = &error {
        if message.starts_with("static_inputs.") {
            return error;
        }
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
    supervisor: SupervisorLoan,
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
    acks: mpsc::Sender<OperatorCheckpointAck>,
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
            acks: self.acks.clone(),
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
    operator_acks: mpsc::Receiver<OperatorCheckpointAck>,
    operator_terminal_ready: mpsc::Receiver<String>,
    operator_commands: BTreeMap<String, mpsc::Sender<OperatorCheckpointCommand>>,
    sink_ack_sender: mpsc::Sender<SinkCheckpointAck>,
    sink_acks: mpsc::Receiver<SinkCheckpointAck>,
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
                acks: operator_tx,
                transaction,
                terminal_ready: operator_terminal_tx,
                terminal_commands: Arc::new(Mutex::new(operator_command_receivers)),
                #[cfg(test)]
                faults,
                #[cfg(test)]
                fault_cancellation,
            },
            operator_acks: operator_rx,
            operator_terminal_ready: operator_terminal_rx,
            operator_commands,
            sink_ack_sender: sink_tx,
            sink_acks: sink_rx,
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
            acks: self.sink_ack_sender.clone(),
            commands: self
                .sink_command_receivers
                .remove(output_id)
                .expect("checkpoint wiring covers every validated sink output"),
            finalizations,
            terminal_ready: Some(self.sink_terminal_ready_sender.clone()),
            transaction: Some(Arc::clone(&self.operator.transaction)),
        }
    }
}

#[cfg(test)]
type OperatorEndpoints = (
    BTreeMap<String, OperatorEdgeSender>,
    BTreeMap<String, OperatorEdgeReceiver>,
);

#[cfg(test)]
fn create_runtime_channels(
    plan: &StreamRuntimePlanParts,
    metrics: &MetricsRecorder,
) -> crate::Result<OperatorEndpoints> {
    let channels = create_runtime_edges(plan, &BTreeMap::new(), metrics)?;
    Ok((channels.senders, channels.receivers))
}

struct RuntimeChannels {
    senders: BTreeMap<String, OperatorEdgeSender>,
    receivers: BTreeMap<String, OperatorEdgeReceiver>,
    local_owners: BTreeMap<String, Vec<LocalEdgeOwner>>,
}

fn create_runtime_edges(
    plan: &StreamRuntimePlanParts,
    layout: &BTreeMap<String, operator_fusion::FusionProof>,
    metrics: &MetricsRecorder,
) -> crate::Result<RuntimeChannels> {
    let local_edges: BTreeMap<_, _> = layout
        .iter()
        .flat_map(|(head, proof)| proof.edges().iter().map(move |edge| (edge.as_str(), head)))
        .collect();
    let mut channels = RuntimeChannels {
        senders: BTreeMap::new(),
        receivers: BTreeMap::new(),
        local_owners: BTreeMap::new(),
    };
    for (edge_id, edge) in &plan.edges {
        let (sender, receiver) = if let Some(head) = local_edges.get(edge_id.as_str()) {
            let (sender, receiver, owner) =
                local_edge(edge_id.clone(), edge.budget, metrics.clone())?;
            channels
                .local_owners
                .entry((*head).clone())
                .or_default()
                .push(owner);
            (
                OperatorEdgeSender::Local(sender),
                OperatorEdgeReceiver::Local(receiver),
            )
        } else {
            let (sender, receiver) =
                edge_channel_with_metrics(edge_id.clone(), edge.budget, metrics.clone())?;
            (sender.into(), receiver.into())
        };
        channels.senders.insert(edge_id.clone(), sender);
        channels.receivers.insert(edge_id.clone(), receiver);
    }
    Ok(channels)
}

fn take_node_ingresses(
    node: &RuntimeStreamNode,
    receivers: &mut BTreeMap<String, OperatorEdgeReceiver>,
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
    senders: &mut BTreeMap<String, OperatorEdgeSender>,
) -> crate::Result<BTreeMap<String, Vec<OperatorEdgeSender>>> {
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
    let supervisor = TaskSupervisor::new_with_terminal_arbiter(
        cancellation.clone(),
        core.terminal_arbiter.clone(),
    );
    let mut supervisor = core.supervision.install(
        supervisor,
        core.entity_work.clone(),
        core.sql_recovery.clone(),
        core.gather_work.clone(),
    );
    core.runtime_status.lock().tasks = supervisor.registry();
    let (entry_tx, _) = watch::channel(false);
    let (data_tx, _) = watch::channel(false);
    let (ack_tx, mut ack_rx) = mpsc::unbounded_channel();
    let layout = operator_fusion::plan_fusion(&plan);
    let RuntimeChannels {
        mut senders,
        mut receivers,
        mut local_owners,
    } = create_runtime_edges(&plan, &layout, &core.metrics).map_err(preflight_entry_failure)?;
    let node_count = plan.nodes.len();
    let registration = &mut OperatorRegistration {
        next_node_order: 0,
        context,
        core,
        entry_tx: &entry_tx,
        data_tx: &data_tx,
        ack_tx: &ack_tx,
        senders: &mut senders,
        receivers: &mut receivers,
        local_owners: &mut local_owners,
        supervisor: &mut supervisor,
        metrics: &core.metrics,
        runtime_status: &core.runtime_status,
        restores: &mut restores,
        checkpoint: checkpoint.as_ref(),
    };
    if let Err(failure) = register_operator_nodes(plan.nodes, layout, registration) {
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
    next_node_order: usize,
    context: &'a super::StreamJobContext,
    core: &'a Arc<JobCore>,
    entry_tx: &'a watch::Sender<bool>,
    data_tx: &'a watch::Sender<bool>,
    ack_tx: &'a mpsc::UnboundedSender<super::operator_task::OperatorEntryAck>,
    senders: &'a mut BTreeMap<String, OperatorEdgeSender>,
    receivers: &'a mut BTreeMap<String, OperatorEdgeReceiver>,
    local_owners: &'a mut BTreeMap<String, Vec<LocalEdgeOwner>>,
    supervisor: &'a mut TaskSupervisor,
    metrics: &'a MetricsRecorder,
    runtime_status: &'a Mutex<RuntimeStatus>,
    restores: &'a mut BTreeMap<String, OperatorRestoreState>,
    checkpoint: Option<&'a OperatorCheckpointRegistration>,
}

fn register_operator_nodes(
    nodes: Vec<RuntimeStreamNode>,
    mut layout: BTreeMap<String, operator_fusion::FusionProof>,
    registration: &mut OperatorRegistration<'_>,
) -> Result<(), EntryFailure> {
    let mut prepared = BTreeMap::new();
    let mut original_order = Vec::new();
    for (ordinal, node) in nodes.into_iter().enumerate() {
        let inputs = prepare_operator_task(node, registration)?;
        let token = registration
            .supervisor
            .reserve_logical(format!("operator:{}", inputs.node_id));
        original_order.push(inputs.node_id.clone());
        prepared.insert(inputs.node_id.clone(), (ordinal, inputs, token));
    }
    let mut groups = Vec::new();
    for node_id in original_order {
        if !prepared.contains_key(&node_id) {
            continue;
        }
        let proof = layout.remove(&node_id);
        let members = if let Some(proof) = &proof {
            proof
                .members()
                .iter()
                .map(|(id, ordinal)| {
                    let (actual_ordinal, inputs, token) = prepared.remove(id).ok_or_else(|| {
                        preflight_entry_failure(CalcFlowError::Internal {
                            message: format!("fusion member {id:?} is missing or already claimed"),
                        })
                    })?;
                    if *ordinal != actual_ordinal {
                        return Err(preflight_entry_failure(CalcFlowError::Internal {
                            message: format!(
                                "fusion member {id:?} changed its original registration ordinal"
                            ),
                        }));
                    }
                    Ok((inputs, token))
                })
                .collect::<Result<Vec<_>, EntryFailure>>()?
        } else {
            let (_, inputs, token) = prepared
                .remove(&node_id)
                .expect("unclaimed original member exists");
            vec![(inputs, token)]
        };
        let owners = registration
            .local_owners
            .remove(&node_id)
            .unwrap_or_default();
        groups.push(
            prepare_operator_task_group(members, proof, owners).map_err(preflight_entry_failure)?,
        );
    }
    if !prepared.is_empty() || !layout.is_empty() || !registration.local_owners.is_empty() {
        return Err(preflight_entry_failure(CalcFlowError::Internal {
            message: "fusion layout left unowned members or local edges".into(),
        }));
    }
    for group in groups {
        registration.supervisor.spawn_prepared_group(group);
    }
    Ok(())
}

fn prepare_operator_task(
    node: RuntimeStreamNode,
    registration: &mut OperatorRegistration<'_>,
) -> Result<OperatorTaskInputs, EntryFailure> {
    let node_id = node.operator_id.as_str().to_owned();
    debug_assert_eq!(node.node_id, node_id);
    let progress = OperatorProgress::with_optional_rolling_metrics(
        registration
            .runtime_status
            .lock()
            .rolling_metrics
            .get(&node_id)
            .cloned(),
    );
    let ingresses =
        take_node_ingresses(&node, registration.receivers).map_err(preflight_entry_failure)?;
    let outputs =
        take_node_outputs(&node, registration.senders).map_err(preflight_entry_failure)?;
    let task_context = registration
        .context
        .for_node(&node_id)
        .map_err(preflight_entry_failure)?;
    let node_order = registration.next_node_order;
    registration.next_node_order += 1;
    let inputs = OperatorTaskInputs {
        sql_recovery: matches!(
            &node.operator,
            crate::pipeline::CompiledStreamOperator::Sql(_)
        )
        .then(|| SqlRecoveryClient {
            owner: registration.core.sql_recovery.clone(),
            node_order,
            task_id: None,
        }),
        late_output_ports: node.late_output_ports,
        entity_work: matches!(
            &node.operator,
            crate::pipeline::CompiledStreamOperator::Rolling(_)
        )
        .then(|| {
            registration
                .core
                .entity_work
                .unbound_client(format!("operator:{node_id}").into())
        }),
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
    };
    registration
        .runtime_status
        .lock()
        .nodes
        .insert(node_id, progress);
    Ok(inputs)
}

async fn await_operator_entry(
    node_count: usize,
    core: &Arc<JobCore>,
    ack_rx: &mut mpsc::UnboundedReceiver<super::operator_task::OperatorEntryAck>,
    supervisor: &mut TaskSupervisor,
) -> Result<(), EntryFailure> {
    let mut acks = Vec::with_capacity(node_count);
    for _ in 0..node_count {
        let ack = tokio::select! {
            biased;
            () = core.launch_cancel.cancelled() => None,
            ack = ack_rx.recv() => ack,
        };
        let Some(ack) = ack else {
            break;
        };
        acks.push(ack);
    }
    let entry_failure = if acks.len() != node_count && !core.launch_cancel.is_cancelled() {
        Some(Arc::new(RuntimeFailure {
            origin: FailureOrigin::Preflight,
            error: CalcFlowError::Internal {
                message: "operator entry ack channel closed early".into(),
            },
        }))
    } else {
        acks.sort_by(|left, right| left.node_id.cmp(&right.node_id));
        acks.into_iter().find_map(|ack| {
            ack.result.err().map(|error| {
                Arc::new(RuntimeFailure {
                    origin: FailureOrigin::OperatorEntry {
                        node_id: ack.node_id,
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
    senders: &mut BTreeMap<String, OperatorEdgeSender>,
    receivers: &mut BTreeMap<String, OperatorEdgeReceiver>,
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
            Ok((
                binding_id,
                vec![sender.into_physical().map_err(preflight_entry_failure)?],
            ))
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
            Ok((
                output_id,
                receiver.into_physical().map_err(preflight_entry_failure)?,
            ))
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
                lifecycle_timeout: checkpoint
                    .as_ref()
                    .map_or(CONNECTOR_CLOSE_TIMEOUT, |checkpoint| {
                        checkpoint.config.checkpoint_timeout
                    }),
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

// Existing supervision flow retains its cancellation and cleanup ordering.
// #lizard forgives
async fn drive_running_job(
    launch_id: LaunchId,
    core: &Arc<JobCore>,
    deadline: Option<chrono::DateTime<Utc>>,
    cancellation: CancellationToken,
    progress: RuntimeTaskProgress,
    supervisor: &mut SupervisorLoan,
    metrics: &MetricsRecorder,
) -> LaunchId {
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
                return core.prepare_driver_report(finish_running_report(
                    launch_id,
                    committed_terminal,
                    graceful_requested,
                    report,
                    &progress,
                    metrics,
                ));
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
        TerminalCause::TaskFailure { .. } | TerminalCause::RunnerFailure => errors
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

// Existing failure aggregation retains deterministic source and sink ordering.
// #lizard forgives
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

fn task_runtime_failure(failure: TaskFailure) -> Arc<RuntimeFailure> {
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
        let mut result = tokio::select! {
            biased;
            () = cancellation.cancelled() => OpenResult::Cancelled,
            result = tokio::time::timeout(
                CONNECTOR_OPEN_TIMEOUT,
                AssertUnwindSafe(resource.open()).catch_unwind(),
            ) => match result {
                Ok(Ok(Ok(()))) => OpenResult::Opened,
                Ok(Ok(Err(error))) => OpenResult::Failed(error),
                Ok(Err(payload)) => OpenResult::Failed(CalcFlowError::TaskPanicked {
                    task_id,
                    message: panic_message(payload.as_ref()),
                }),
                Err(_) => OpenResult::Failed(CalcFlowError::Internal {
                    message: "connector open exceeded private 30-second bound".into(),
                }),
            },
        };
        if !matches!(&result, OpenResult::Opened) {
            // A connector may have started bounded native work before its open
            // future was cancelled. Keep lineage ownership until that work ends.
            match tokio::time::timeout(CONNECTOR_OPEN_SETTLE_TIMEOUT, resource.settle_open()).await
            {
                Ok(Err(error)) if matches!(&result, OpenResult::Cancelled) => {
                    result = OpenResult::Failed(error);
                }
                Ok(_) => {}
                Err(_) => {
                    result = OpenResult::Failed(CalcFlowError::Internal {
                        message: "connector open settlement exceeded private 35-second bound"
                            .into(),
                    });
                }
            }
        }
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
mod tests;
