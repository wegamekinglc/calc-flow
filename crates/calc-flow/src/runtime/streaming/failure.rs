//! The streaming runner's failure taxonomy.
//!
//! Failure origins, the runtime/start failure envelopes, job lifecycle
//! states, and the recovery classification every supervision path shares.
//! Moved verbatim from `runner.rs`; `runner` re-exports each name so
//! existing import paths are unchanged.

use std::sync::Arc;

use crate::CalcFlowError;

use super::supervisor::TaskId;

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

impl LaunchId {
    pub(crate) fn new(value: u64) -> Self {
        Self(value)
    }
}

pub(super) fn classify_failure_state(failure: &RuntimeFailure) -> ContinuousJobState {
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
