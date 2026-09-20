//! The streaming runner's failure taxonomy.
//!
//! Failure origins, the runtime/start failure envelopes, job lifecycle
//! states, and the recovery classification every supervision path shares.
//! Moved verbatim from `runner.rs`; `runner` re-exports each name so
//! existing import paths are unchanged. `panic_message`, the bounded
//! panic-payload message every task-panic error shares, lives here as well.

use std::{any::Any, sync::Arc};

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
    RunnerFailure,
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
    Finalizing,
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

pub(crate) fn panic_message(payload: &(dyn Any + Send)) -> String {
    const MAX_PANIC_BYTES: usize = 1_024;
    const ELLIPSIS: &str = "…";

    let message = if let Some(message) = payload.downcast_ref::<&str>() {
        *message
    } else if let Some(message) = payload.downcast_ref::<String>() {
        message.as_str()
    } else {
        return "non-string panic payload".into();
    };
    if message.len() <= MAX_PANIC_BYTES {
        return message.to_owned();
    }

    let mut prefix_end = MAX_PANIC_BYTES - ELLIPSIS.len();
    while !message.is_char_boundary(prefix_end) {
        prefix_end -= 1;
    }
    format!("{}{}", &message[..prefix_end], ELLIPSIS)
}

#[cfg(test)]
mod tests {
    use std::any::Any;

    use super::panic_message;

    #[test]
    fn panic_payload_is_utf8_safe_and_bounded_to_1024_bytes() {
        let long_ascii = "a".repeat(1_100);
        let bounded_ascii = panic_message(&long_ascii);
        assert_eq!(bounded_ascii.len(), 1_024);
        assert!(bounded_ascii.ends_with('…'));
        assert_eq!(&bounded_ascii[..1_021], "a".repeat(1_021));

        let split_at_limit = format!("{}{}", "a".repeat(1_020), "😀".repeat(2));
        let bounded_multibyte = panic_message(&split_at_limit);
        assert!(bounded_multibyte.is_char_boundary(bounded_multibyte.len()));
        assert_eq!(bounded_multibyte.len(), 1_023);
        assert!(bounded_multibyte.ends_with('…'));
        assert_eq!(&bounded_multibyte[..1_020], "a".repeat(1_020));

        let short = String::from("short panic");
        assert_eq!(panic_message(&short), short);
        assert_eq!(
            panic_message(&(7_u64) as &(dyn Any + Send)),
            "non-string panic payload"
        );
    }
}
