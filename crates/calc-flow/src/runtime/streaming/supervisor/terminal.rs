//! Job-level terminal-decision arbitration shared by the runner and the
//! supervisor. The runner constructs one arbiter per job and injects it into
//! the supervisor, so terminal requests, task-failure announcements, and the
//! driver's immutable commit share one lock.

use std::{collections::BTreeSet, sync::Arc};

use parking_lot::Mutex;

use super::TaskId;
use crate::CancellationToken;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) enum TerminalDecision {
    TaskFailure(TaskId),
    ExplicitCancel,
    DeadlineExceeded,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) struct TerminalObservation {
    pub(crate) terminal: Option<TerminalDecision>,
    pub(crate) graceful_shutdown: bool,
}

#[derive(Default)]
struct TerminalArbiterState {
    primary_failures: BTreeSet<TaskId>,
    explicit_cancel: bool,
    deadline_exceeded: bool,
    graceful_shutdown: bool,
    committed: Option<TerminalDecision>,
}

/// One lock shared by terminal requests, task-failure announcement, and the
/// immutable driver commit. No caller may hold a job-state lock while entering
/// this arbiter.
#[derive(Clone, Default)]
pub(crate) struct TerminalArbiter(Arc<Mutex<TerminalArbiterState>>);

impl TerminalArbiter {
    pub(crate) fn request_explicit_cancel(&self) -> bool {
        let mut state = self.0.lock();
        if state.committed.is_some() {
            return false;
        }
        state.explicit_cancel = true;
        true
    }

    pub(crate) fn request_deadline(&self) -> bool {
        let mut state = self.0.lock();
        if state.committed.is_some() {
            return false;
        }
        state.deadline_exceeded = true;
        true
    }

    pub(crate) fn request_graceful_shutdown(&self) -> bool {
        let mut state = self.0.lock();
        if state.committed.is_some() {
            return false;
        }
        state.graceful_shutdown = true;
        true
    }

    /// Applies task-failure > explicit > deadline and cancels workers before
    /// releasing the same lock that makes the terminal decision immutable.
    pub(crate) fn observe_and_commit(
        &self,
        cancellation: &CancellationToken,
    ) -> TerminalObservation {
        let mut state = self.0.lock();
        if state.committed.is_none() {
            state.committed = state
                .primary_failures
                .first()
                .copied()
                .map(TerminalDecision::TaskFailure)
                .or_else(|| {
                    state
                        .explicit_cancel
                        .then_some(TerminalDecision::ExplicitCancel)
                })
                .or_else(|| {
                    cancellation
                        .is_cancelled()
                        .then_some(TerminalDecision::ExplicitCancel)
                })
                .or_else(|| {
                    state
                        .deadline_exceeded
                        .then_some(TerminalDecision::DeadlineExceeded)
                });
            if state.committed.is_some() {
                cancellation.cancel();
            }
        }
        TerminalObservation {
            terminal: state.committed,
            graceful_shutdown: state.graceful_shutdown,
        }
    }

    pub(super) fn record_task_failure(&self, task_id: TaskId, cancellation: &CancellationToken) {
        let mut state = self.0.lock();
        if state.committed.is_none() && !cancellation.is_cancelled() {
            state.primary_failures.insert(task_id);
        }
    }

    pub(super) fn record_task_failure_and_cancel(
        &self,
        task_id: TaskId,
        cancellation: &CancellationToken,
    ) {
        let mut state = self.0.lock();
        if state.committed.is_none() && !cancellation.is_cancelled() {
            state.primary_failures.insert(task_id);
        }
        cancellation.cancel();
    }

    pub(super) fn primary_failures(&self) -> BTreeSet<TaskId> {
        self.0.lock().primary_failures.clone()
    }

    #[cfg(test)]
    pub(crate) fn explicit_cancel_requested(&self) -> bool {
        self.0.lock().explicit_cancel
    }
}

#[cfg(test)]
mod tests {
    use super::{TaskId, TerminalArbiter, TerminalDecision};
    use crate::CancellationToken;

    #[test]
    fn terminal_arbiter_prioritizes_one_locked_snapshot_and_keeps_graceful_nonterminal() {
        let cancellation = CancellationToken::new();
        let graceful_only = TerminalArbiter::default();
        assert!(graceful_only.request_graceful_shutdown());
        let observation = graceful_only.observe_and_commit(&cancellation);
        assert_eq!(observation.terminal, None);
        assert!(observation.graceful_shutdown);
        assert!(!cancellation.is_cancelled());

        let arbiter = TerminalArbiter::default();
        assert!(arbiter.request_graceful_shutdown());
        assert!(arbiter.request_deadline());
        assert!(arbiter.request_explicit_cancel());
        arbiter.record_task_failure(TaskId::new(7), &cancellation);
        arbiter.record_task_failure(TaskId::new(3), &cancellation);

        let observation = arbiter.observe_and_commit(&cancellation);
        assert_eq!(
            observation.terminal,
            Some(TerminalDecision::TaskFailure(TaskId::new(3)))
        );
        assert!(observation.graceful_shutdown);
        assert!(cancellation.is_cancelled());
    }

    #[test]
    fn terminal_arbiter_classifies_external_token_cancellation_as_explicit() {
        let cancellation = CancellationToken::new();
        cancellation.cancel();

        let observation = TerminalArbiter::default().observe_and_commit(&cancellation);

        assert_eq!(observation.terminal, Some(TerminalDecision::ExplicitCancel));
        assert!(!observation.graceful_shutdown);
    }
}
