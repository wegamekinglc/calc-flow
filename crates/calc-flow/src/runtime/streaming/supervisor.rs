use std::{
    any::Any,
    collections::{BTreeMap, BTreeSet, HashMap},
    future::Future,
    panic::AssertUnwindSafe,
    sync::Arc,
};

use futures::FutureExt;
use parking_lot::Mutex;
use tokio::{sync::oneshot, task::JoinSet};

use crate::{CalcFlowError, CancellationToken, Result};

/// Stable identity assigned in supervisor registration order (spec D5.1).
#[derive(Clone, Copy, Debug, Eq, Ord, PartialEq, PartialOrd)]
pub(crate) struct TaskId(u64);

impl TaskId {
    pub(crate) const fn new(value: u64) -> Self {
        Self(value)
    }

    pub(crate) const fn as_u64(self) -> u64 {
        self.0
    }
}

/// One failed supervised task, retained with its stable identity.
#[derive(Debug)]
pub(crate) struct TaskFailure {
    pub(crate) task_id: TaskId,
    pub(crate) task_name: String,
    pub(crate) error: CalcFlowError,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub(crate) struct TaskStatus {
    pub(crate) task_name: String,
}

#[derive(Clone, Default)]
pub(crate) struct TaskRegistry(Arc<Mutex<BTreeMap<TaskId, TaskStatus>>>);

impl TaskRegistry {
    pub(crate) fn snapshot(&self) -> BTreeMap<TaskId, TaskStatus> {
        self.0.lock().clone()
    }

    fn insert(&self, task_id: TaskId, task_name: String) {
        self.0.lock().insert(task_id, TaskStatus { task_name });
    }

    fn remove(&self, task_id: TaskId) -> Option<TaskStatus> {
        self.0.lock().remove(&task_id)
    }

    fn len(&self) -> usize {
        self.0.lock().len()
    }
}

/// Fully joined terminal report from a supervisor registry.
#[derive(Debug, Default)]
pub(crate) struct SupervisionReport {
    /// Prefix of `errors` observed before convergence cancellation began.
    pub(crate) primary_error_count: usize,
    pub(crate) errors: Vec<TaskFailure>,
}

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

    fn record_task_failure(&self, task_id: TaskId, cancellation: &CancellationToken) {
        let mut state = self.0.lock();
        if state.committed.is_none() && !cancellation.is_cancelled() {
            state.primary_failures.insert(task_id);
        }
    }

    fn record_task_failure_and_cancel(&self, task_id: TaskId, cancellation: &CancellationToken) {
        let mut state = self.0.lock();
        if state.committed.is_none() && !cancellation.is_cancelled() {
            state.primary_failures.insert(task_id);
        }
        cancellation.cancel();
    }

    fn primary_failures(&self) -> BTreeSet<TaskId> {
        self.0.lock().primary_failures.clone()
    }

    #[cfg(test)]
    pub(crate) fn explicit_cancel_requested(&self) -> bool {
        self.0.lock().explicit_cancel
    }
}

impl SupervisionReport {
    pub(crate) fn primary_errors(&self) -> &[TaskFailure] {
        &self.errors[..self.primary_error_count]
    }
}

struct TaskExit {
    task_id: TaskId,
    task_name: String,
    result: Result<()>,
}

/// Lets a task record its failure trigger before teardown work completes.
#[derive(Clone)]
pub(crate) struct TaskFailureSignal {
    task_id: TaskId,
    cancellation: CancellationToken,
    terminal_arbiter: TerminalArbiter,
}

impl TaskFailureSignal {
    pub(crate) const fn task_id(&self) -> TaskId {
        self.task_id
    }

    fn record_before_cancellation(&self) {
        self.terminal_arbiter
            .record_task_failure(self.task_id, &self.cancellation);
    }

    /// Records this task as a failure trigger, then starts convergence.
    pub(crate) fn cancel_siblings(&self) {
        self.terminal_arbiter
            .record_task_failure_and_cancel(self.task_id, &self.cancellation);
    }

    async fn converge_after_failure(&self) {
        self.record_before_cancellation();
        if !self.cancellation.is_cancelled() {
            tokio::task::yield_now().await;
            self.cancel_siblings();
        }
    }
}

/// Owns every Tokio task of one internal continuous job.
///
/// User futures are start-gated until their stable task ID is present in the
/// registry. Errors observed before cancellation form the primary scheduling
/// round; every later convergence error remains secondary (D5/S8.4).
pub(crate) struct TaskSupervisor {
    cancellation: CancellationToken,
    tasks: JoinSet<TaskExit>,
    stable_ids: HashMap<tokio::task::Id, TaskId>,
    registry: TaskRegistry,
    joined_errors: Vec<TaskFailure>,
    terminal_arbiter: TerminalArbiter,
    next_task_id: u64,
}

impl TaskSupervisor {
    pub(crate) fn new(cancellation: CancellationToken) -> Self {
        Self::new_with_terminal_arbiter(cancellation, TerminalArbiter::default())
    }

    pub(crate) fn new_with_terminal_arbiter(
        cancellation: CancellationToken,
        terminal_arbiter: TerminalArbiter,
    ) -> Self {
        Self {
            cancellation,
            tasks: JoinSet::new(),
            stable_ids: HashMap::new(),
            registry: TaskRegistry::default(),
            joined_errors: Vec::new(),
            terminal_arbiter,
            next_task_id: 0,
        }
    }

    /// Registers and starts one owned task.
    pub(crate) fn spawn<F>(&mut self, name: impl Into<String>, future: F) -> TaskId
    where
        F: Future<Output = Result<()>> + Send + 'static,
    {
        self.spawn_with_failure_signal(name, |_| future)
    }

    /// Registers a task that may announce failure before its teardown ends.
    pub(crate) fn spawn_with_failure_signal<F, Fut>(
        &mut self,
        name: impl Into<String>,
        make_future: F,
    ) -> TaskId
    where
        F: FnOnce(TaskFailureSignal) -> Fut,
        Fut: Future<Output = Result<()>> + Send + 'static,
    {
        let task_id = TaskId::new(self.next_task_id);
        self.next_task_id = self
            .next_task_id
            .checked_add(1)
            .expect("a streaming job cannot register u64::MAX tasks");
        let task_name = name.into();
        let task_name_in_task = task_name.clone();
        let failure_signal = TaskFailureSignal {
            task_id,
            cancellation: self.cancellation.clone(),
            terminal_arbiter: self.terminal_arbiter.clone(),
        };
        let future = make_future(failure_signal.clone());
        let (start_tx, start_rx) = oneshot::channel();
        self.registry.insert(task_id, task_name);
        let abort_handle = self.tasks.spawn(async move {
            let result = match start_rx.await {
                Ok(()) => match AssertUnwindSafe(future).catch_unwind().await {
                    Ok(result) => result,
                    Err(payload) => Err(CalcFlowError::TaskPanicked {
                        task_id: task_id.as_u64(),
                        message: panic_message(payload.as_ref()),
                    }),
                },
                Err(_) => Err(CalcFlowError::Internal {
                    message: format!("task {task_name_in_task:?} start gate was dropped"),
                }),
            };
            if result.is_err() {
                failure_signal.converge_after_failure().await;
            }
            TaskExit {
                task_id,
                task_name: task_name_in_task,
                result,
            }
        });
        self.stable_ids.insert(abort_handle.id(), task_id);
        start_tx
            .send(())
            .expect("the newly registered task still owns its start gate");
        task_id
    }

    pub(crate) fn cancel(&self) {
        self.cancellation.cancel();
    }

    pub(crate) fn task_count(&self) -> usize {
        self.registry.len()
    }

    pub(crate) fn registry(&self) -> TaskRegistry {
        self.registry.clone()
    }

    /// Joins every registered task, cancelling siblings after the first
    /// failed task becomes observable.
    pub(crate) async fn join_all(&mut self) -> SupervisionReport {
        while let Some(joined) = self.tasks.join_next_with_id().await {
            let exit = match joined {
                Ok((tokio_id, exit)) => {
                    self.stable_ids.remove(&tokio_id);
                    exit
                }
                Err(error) => {
                    let task_id = self
                        .stable_ids
                        .remove(&error.id())
                        .unwrap_or(TaskId::new(u64::MAX));
                    let task_name = self.registry.remove(task_id).map_or_else(
                        || "unknown supervised task".into(),
                        |status| status.task_name,
                    );
                    let failure = TaskFailure {
                        task_id,
                        task_name,
                        error: CalcFlowError::Internal {
                            message: format!("supervised task join failed: {error}"),
                        },
                    };
                    let failure_signal = TaskFailureSignal {
                        task_id,
                        cancellation: self.cancellation.clone(),
                        terminal_arbiter: self.terminal_arbiter.clone(),
                    };
                    failure_signal.record_before_cancellation();
                    failure_signal.cancel_siblings();
                    self.joined_errors.push(failure);
                    continue;
                }
            };
            self.registry.remove(exit.task_id);
            if let Err(error) = exit.result {
                self.joined_errors.push(TaskFailure {
                    task_id: exit.task_id,
                    task_name: exit.task_name,
                    error,
                });
            }
        }
        let primary_failures = self.terminal_arbiter.primary_failures();
        self.joined_errors.sort_by_key(|failure| {
            (
                !primary_failures.contains(&failure.task_id),
                failure.task_id,
            )
        });
        let primary_error_count = self
            .joined_errors
            .iter()
            .take_while(|failure| primary_failures.contains(&failure.task_id))
            .count();
        SupervisionReport {
            primary_error_count,
            errors: std::mem::take(&mut self.joined_errors),
        }
    }
}

impl Drop for TaskSupervisor {
    fn drop(&mut self) {
        self.cancellation.cancel();
        self.tasks.abort_all();
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
    use std::sync::{
        Arc,
        atomic::{AtomicBool, Ordering},
    };

    use tokio::sync::{Barrier, oneshot};

    use super::{TaskId, TaskSupervisor, TerminalArbiter, TerminalDecision, panic_message};
    use crate::{CalcFlowError, CancellationToken};

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

    #[tokio::test]
    async fn first_failure_cancels_and_joins_a_sibling() {
        let cancellation = CancellationToken::new();
        let sibling_finished = Arc::new(AtomicBool::new(false));
        let (started_tx, started_rx) = oneshot::channel();
        let mut supervisor = TaskSupervisor::new(cancellation.clone());

        let failed_id = supervisor.spawn("failing", async move {
            let _ = started_tx.send(());
            Err(CalcFlowError::Internal {
                message: "source failed".into(),
            })
        });
        let sibling_finished_in_task = Arc::clone(&sibling_finished);
        supervisor.spawn("sibling", async move {
            cancellation.cancelled().await;
            sibling_finished_in_task.store(true, Ordering::SeqCst);
            Ok(())
        });

        started_rx.await.unwrap();
        let report = supervisor.join_all().await;

        assert_eq!(failed_id, TaskId::new(0));
        assert_eq!(report.errors.len(), 1);
        assert_eq!(report.errors[0].task_id, failed_id);
        assert_eq!(report.errors[0].task_name, "failing");
        assert!(matches!(
            report.errors[0].error,
            CalcFlowError::Internal { ref message } if message == "source failed"
        ));
        assert!(sibling_finished.load(Ordering::SeqCst));
        assert_eq!(supervisor.task_count(), 0);
    }

    #[tokio::test]
    async fn task_failure_requests_cancellation_before_join_is_polled() {
        let cancellation = CancellationToken::new();
        let mut supervisor = TaskSupervisor::new(cancellation.clone());
        let (failure_started_tx, failure_started_rx) = oneshot::channel();
        supervisor.spawn("failing", async move {
            failure_started_tx.send(()).unwrap();
            Err(CalcFlowError::Internal {
                message: "source failed".into(),
            })
        });

        failure_started_rx.await.unwrap();
        tokio::task::yield_now().await;

        assert!(
            cancellation.is_cancelled(),
            "a failed task must start sibling convergence without waiting for join_all"
        );
        let report = supervisor.join_all().await;
        assert_eq!(report.errors.len(), 1);
    }

    #[tokio::test]
    async fn simultaneous_failures_are_returned_in_stable_task_order() {
        for repetition in 0..100 {
            let release = Arc::new(Barrier::new(3));
            let mut supervisor = TaskSupervisor::new(CancellationToken::new());
            for message in ["zero", "one"] {
                let release_in_task = Arc::clone(&release);
                supervisor.spawn(message, async move {
                    release_in_task.wait().await;
                    Err(CalcFlowError::Internal {
                        message: message.into(),
                    })
                });
            }

            release.wait().await;
            let report = supervisor.join_all().await;

            assert_eq!(
                report
                    .errors
                    .iter()
                    .map(|failure| failure.task_id)
                    .collect::<Vec<_>>(),
                [TaskId::new(0), TaskId::new(1)],
                "unstable failure order at repetition {repetition}"
            );
        }
    }

    #[tokio::test]
    async fn cancellation_convergence_error_does_not_replace_the_primary_trigger() {
        let cancellation = CancellationToken::new();
        let mut supervisor = TaskSupervisor::new(cancellation.clone());
        let convergence_id = supervisor.spawn("convergence", async move {
            cancellation.cancelled().await;
            Err(CalcFlowError::EdgeClosed {
                edge: "runtime-closed-secondary".into(),
            })
        });
        let (release_primary_tx, release_primary_rx) = oneshot::channel();
        let primary_id = supervisor.spawn("primary", async move {
            release_primary_rx.await.unwrap();
            Err(CalcFlowError::Internal {
                message: "primary source failure".into(),
            })
        });

        release_primary_tx.send(()).unwrap();
        let report = supervisor.join_all().await;

        assert_eq!(convergence_id, TaskId::new(0));
        assert_eq!(primary_id, TaskId::new(1));
        assert_eq!(report.errors.len(), 2);
        assert_eq!(report.errors[0].task_id, primary_id);
        assert!(matches!(
            &report.errors[0].error,
            CalcFlowError::Internal { message } if message == "primary source failure"
        ));
        assert_eq!(report.errors[1].task_id, convergence_id);
        assert!(matches!(
            report.errors[1].error,
            CalcFlowError::EdgeClosed { ref edge } if edge == "runtime-closed-secondary"
        ));
    }

    #[tokio::test]
    async fn panic_becomes_a_typed_error_with_stable_task_identity() {
        let mut supervisor = TaskSupervisor::new(CancellationToken::new());
        let task_id = supervisor.spawn("panicking-source", async move {
            panic!("connector invariant");
            #[allow(unreachable_code)]
            Ok(())
        });

        let report = supervisor.join_all().await;

        assert_eq!(task_id, TaskId::new(0));
        assert_eq!(report.errors.len(), 1);
        assert!(matches!(
            report.errors[0].error,
            CalcFlowError::TaskPanicked { task_id: 0, ref message }
                if message == "connector invariant"
        ));
        assert_eq!(supervisor.task_count(), 0);
    }

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
