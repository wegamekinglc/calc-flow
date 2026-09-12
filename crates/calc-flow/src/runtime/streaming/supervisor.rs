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

mod ready_pair;

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
pub(crate) struct TaskRegistry(
    Arc<Mutex<BTreeMap<TaskId, TaskStatus>>>,
    #[cfg(test)] Arc<Mutex<Vec<tokio::task::AbortHandle>>>,
);

impl TaskRegistry {
    #[cfg(test)]
    pub(crate) fn abort_all_for_test(&self) {
        for handle in self.1.lock().iter() {
            handle.abort();
        }
    }

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

/// Resources that must outlive failure publication and sibling convergence.
pub(crate) struct RetainedTaskResult<R> {
    pub(crate) result: Result<()>,
    pub(crate) retained: R,
}

pub(crate) trait TaskOutput: Send + 'static {
    type Retained: Send + 'static;

    fn into_parts(self) -> (Result<()>, Self::Retained);
}

impl TaskOutput for Result<()> {
    type Retained = ();

    fn into_parts(self) -> (Result<()>, ()) {
        (self, ())
    }
}

impl<R: Send + 'static> TaskOutput for RetainedTaskResult<R> {
    type Retained = R;

    fn into_parts(self) -> (Result<()>, R) {
        (self.result, self.retained)
    }
}

struct TaskRegistration {
    task_id: TaskId,
    task_name: String,
    failure_signal: TaskFailureSignal,
    settled: Arc<Mutex<BTreeMap<TaskId, TaskExit>>>,
}

pub(crate) struct PreparedPair<F, G> {
    first_task: TaskRegistration,
    second_task: TaskRegistration,
    first: F,
    second: G,
    readiness: bool,
    #[cfg(test)]
    observer: Option<ready_pair::Observer>,
}

impl<F, G> PreparedPair<F, G>
where
    F: Future + Send,
    F::Output: TaskOutput,
    G: Future + Send,
    G::Output: TaskOutput,
{
    pub(crate) fn ids(&self) -> [TaskId; 2] {
        [self.first_task.task_id, self.second_task.task_id]
    }

    pub(crate) fn with_readiness(mut self) -> Self {
        self.readiness = true;
        self
    }

    #[cfg(test)]
    fn observe(mut self, observer: ready_pair::Observer) -> Self {
        self.observer = Some(observer);
        self
    }

    pub(crate) async fn run(self, started: bool) -> Vec<TaskId> {
        let first = self.first_task.run(started, self.first);
        let second = self.second_task.run(started, self.second);
        let [first, second] = if self.readiness {
            ready_pair::run(
                first,
                second,
                #[cfg(test)]
                self.observer,
            )
            .await
        } else {
            let (second, first) = tokio::join!(biased; second, first);
            [first, second]
        };
        vec![first, second]
    }
}

impl TaskRegistration {
    async fn run<Fut>(self, started: bool, future: Fut) -> TaskId
    where
        Fut: Future + Send,
        Fut::Output: TaskOutput,
    {
        let completion = if started {
            contain_task_panic(self.task_id, future).await
        } else {
            Err(CalcFlowError::Internal {
                message: format!("task {:?} start gate was dropped", self.task_name),
            })
        };
        let (result, retained) = match completion {
            Ok(output) => {
                let (result, retained) = output.into_parts();
                (result, Some(retained))
            }
            Err(error) => (Err(error), None),
        };
        let failed = result.is_err();
        if failed {
            self.failure_signal.record_before_cancellation();
        }
        let exit = TaskExit {
            task_id: self.task_id,
            task_name: self.task_name,
            result,
        };
        // A sibling can keep a shared driver pending after this member settles.
        // Publish before convergence can yield, so aborting that driver cannot
        // replace an already completed result with a spurious join failure.
        self.settled.lock().insert(self.task_id, exit);
        if failed {
            self.failure_signal.converge_after_failure().await;
        }
        drop(retained);
        self.task_id
    }
}

pub(crate) async fn contain_task_panic<F: Future>(task_id: TaskId, future: F) -> Result<F::Output> {
    AssertUnwindSafe(future)
        .catch_unwind()
        .await
        .map_err(|payload| CalcFlowError::TaskPanicked {
            task_id: task_id.as_u64(),
            message: panic_message(payload.as_ref()),
        })
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
    tasks: JoinSet<Vec<TaskId>>,
    settled: Arc<Mutex<BTreeMap<TaskId, TaskExit>>>,
    stable_ids: HashMap<tokio::task::Id, Vec<TaskId>>,
    registry: TaskRegistry,
    joined_errors: Vec<TaskFailure>,
    terminal_arbiter: TerminalArbiter,
    next_task_id: u64,
    #[cfg(test)]
    ready_pair_spawns: usize,
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
            settled: Arc::new(Mutex::new(BTreeMap::new())),
            stable_ids: HashMap::new(),
            registry: TaskRegistry::default(),
            joined_errors: Vec::new(),
            terminal_arbiter,
            next_task_id: 0,
            #[cfg(test)]
            ready_pair_spawns: 0,
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
        Fut: Future + Send + 'static,
        Fut::Output: TaskOutput,
    {
        let task = self.reserve_task(name);
        let task_id = task.task_id;
        let future = make_future(task.failure_signal.clone());
        let (start_tx, start_rx) = oneshot::channel();
        self.registry.insert(task_id, task.task_name.clone());
        let abort_handle = self
            .tasks
            .spawn(async move { vec![task.run(start_rx.await.is_ok(), future).await] });
        self.stable_ids.insert(abort_handle.id(), vec![task_id]);
        #[cfg(test)]
        self.registry.1.lock().push(abort_handle.clone());
        start_tx
            .send(())
            .expect("the newly registered task still owns its start gate");
        task_id
    }

    pub(crate) fn cancel(&self) {
        self.cancellation.cancel();
    }

    #[cfg(test)]
    pub(crate) fn spawn_pair_with_failure_signals<F, Fut, G, Gut>(
        &mut self,
        first_name: &str,
        first: F,
        second_name: &str,
        second: G,
    ) -> [TaskId; 2]
    where
        F: FnOnce(TaskFailureSignal) -> Fut,
        Fut: Future<Output = Result<()>> + Send + 'static,
        G: FnOnce(TaskFailureSignal) -> Gut,
        Gut: Future<Output = Result<()>> + Send + 'static,
    {
        let pair = self.prepare_pair_with_failure_signals(first_name, first, second_name, second);
        self.spawn_prepared_pair(pair)
    }

    pub(crate) fn prepare_pair_with_failure_signals<F, Fut, G, Gut>(
        &mut self,
        first_name: &str,
        first: F,
        second_name: &str,
        second: G,
    ) -> PreparedPair<Fut, Gut>
    where
        F: FnOnce(TaskFailureSignal) -> Fut,
        Fut: Future + Send + 'static,
        Fut::Output: TaskOutput,
        G: FnOnce(TaskFailureSignal) -> Gut,
        Gut: Future + Send + 'static,
        Gut::Output: TaskOutput,
    {
        let first_task = self.reserve_task(first_name);
        let second_task = self.reserve_task(second_name);
        let first = first(first_task.failure_signal.clone());
        let second = second(second_task.failure_signal.clone());
        let ids = [first_task.task_id, second_task.task_id];
        self.registry.insert(ids[0], first_task.task_name.clone());
        self.registry.insert(ids[1], second_task.task_name.clone());
        PreparedPair {
            first_task,
            second_task,
            first,
            second,
            readiness: false,
            #[cfg(test)]
            observer: None,
        }
    }

    pub(crate) fn spawn_prepared_pair<F, G>(&mut self, pair: PreparedPair<F, G>) -> [TaskId; 2]
    where
        F: Future + Send + 'static,
        F::Output: TaskOutput,
        G: Future + Send + 'static,
        G::Output: TaskOutput,
    {
        let ids = pair.ids();
        #[cfg(test)]
        {
            self.ready_pair_spawns += usize::from(pair.readiness);
        }
        let (start_tx, start_rx) = oneshot::channel();
        let abort_handle = self
            .tasks
            .spawn(async move { pair.run(start_rx.await.is_ok()).await });
        self.stable_ids.insert(abort_handle.id(), ids.to_vec());
        #[cfg(test)]
        self.registry.1.lock().push(abort_handle.clone());
        start_tx
            .send(())
            .expect("the newly registered pair still owns its start gate");
        ids
    }

    fn reserve_task(&mut self, name: impl Into<String>) -> TaskRegistration {
        let task_id = TaskId::new(self.next_task_id);
        self.next_task_id = self
            .next_task_id
            .checked_add(1)
            .expect("a streaming job cannot register u64::MAX tasks");
        TaskRegistration {
            task_id,
            task_name: name.into(),
            failure_signal: TaskFailureSignal {
                task_id,
                cancellation: self.cancellation.clone(),
                terminal_arbiter: self.terminal_arbiter.clone(),
            },
            settled: Arc::clone(&self.settled),
        }
    }

    pub(crate) fn task_count(&self) -> usize {
        self.registry.len()
    }

    #[cfg(test)]
    pub(crate) fn physical_driver_count(&self) -> usize {
        self.tasks.len()
    }

    #[cfg(test)]
    pub(crate) fn ready_pair_spawn_count(&self) -> usize {
        self.ready_pair_spawns
    }

    pub(crate) fn registry(&self) -> TaskRegistry {
        self.registry.clone()
    }

    /// Joins every registered task, cancelling siblings after the first
    /// failed task becomes observable.
    pub(crate) async fn join_all(&mut self) -> SupervisionReport {
        self.settle_tasks().await;
        self.take_report()
    }

    pub(crate) fn cancel_and_abort(&mut self) {
        self.cancel();
        self.tasks.abort_all();
    }

    pub(crate) async fn settle_tasks(&mut self) {
        while let Some(joined) = self.tasks.join_next_with_id().await {
            let exits = match joined {
                Ok((tokio_id, exits)) => {
                    self.stable_ids.remove(&tokio_id);
                    exits
                }
                Err(error) => {
                    self.record_join_failure(&error);
                    continue;
                }
            };
            for task_id in exits {
                let exit = self
                    .settled
                    .lock()
                    .remove(&task_id)
                    .expect("a joined task publishes its exit before returning");
                self.record_settled(exit);
            }
        }
    }

    pub(crate) fn take_report(&mut self) -> SupervisionReport {
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

    fn record_join_failure(&mut self, error: &tokio::task::JoinError) {
        let task_ids = self
            .stable_ids
            .remove(&error.id())
            .unwrap_or_else(|| vec![TaskId::new(u64::MAX)]);
        for task_id in task_ids {
            let settled = self.settled.lock().remove(&task_id);
            if let Some(exit) = settled {
                self.record_settled(exit);
                continue;
            }
            let task_name = self.registry.remove(task_id).map_or_else(
                || "unknown supervised task".into(),
                |status| status.task_name,
            );
            let failure_signal = TaskFailureSignal {
                task_id,
                cancellation: self.cancellation.clone(),
                terminal_arbiter: self.terminal_arbiter.clone(),
            };
            failure_signal.record_before_cancellation();
            failure_signal.cancel_siblings();
            self.joined_errors.push(TaskFailure {
                task_id,
                task_name,
                error: CalcFlowError::Internal {
                    message: format!("supervised task join failed: {error}"),
                },
            });
        }
    }

    fn record_settled(&mut self, exit: TaskExit) {
        self.registry.remove(exit.task_id);
        if let Err(error) = exit.result {
            self.joined_errors.push(TaskFailure {
                task_id: exit.task_id,
                task_name: exit.task_name,
                error,
            });
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

    #[tokio::test]
    async fn fused_pair_keeps_logical_identities_in_one_physical_task() {
        let mut supervisor = TaskSupervisor::new(CancellationToken::new());
        let barrier = Arc::new(Barrier::new(2));
        let first_barrier = Arc::clone(&barrier);
        let ids = supervisor.spawn_pair_with_failure_signals(
            "operator:rolling",
            move |signal| async move {
                assert_eq!(signal.task_id(), TaskId::new(0));
                first_barrier.wait().await;
                Ok(())
            },
            "operator:projection",
            move |signal| async move {
                assert_eq!(signal.task_id(), TaskId::new(1));
                barrier.wait().await;
                Ok(())
            },
        );
        assert_eq!(ids, [TaskId::new(0), TaskId::new(1)]);
        let registry = supervisor.registry();
        assert_eq!(registry.snapshot()[&ids[0]].task_name, "operator:rolling");
        assert_eq!(
            registry.snapshot()[&ids[1]].task_name,
            "operator:projection"
        );
        assert_eq!(supervisor.task_count(), 2);
        assert_eq!(supervisor.physical_driver_count(), 1);
        assert!(supervisor.join_all().await.errors.is_empty());
        assert!(registry.snapshot().is_empty());
    }

    #[tokio::test]
    async fn fused_panic_cancels_waiting_member_with_original_failure_identity() {
        let cancellation = CancellationToken::new();
        let mut supervisor = TaskSupervisor::new(cancellation.clone());
        let waiting_finished = Arc::new(AtomicBool::new(false));
        let finished = Arc::clone(&waiting_finished);
        supervisor.spawn_pair_with_failure_signals(
            "operator:rolling",
            move |_| async move {
                cancellation.cancelled().await;
                finished.store(true, Ordering::SeqCst);
                Ok(())
            },
            "operator:projection",
            |_| async {
                panic!("fused projection failure");
                #[allow(unreachable_code)]
                Ok(())
            },
        );
        let report = supervisor.join_all().await;
        assert_eq!(report.errors.len(), 1);
        assert_eq!(report.errors[0].task_name, "operator:projection");
        assert_eq!(report.errors[0].task_id, TaskId::new(1));
        assert!(matches!(
            &report.errors[0].error,
            CalcFlowError::TaskPanicked { task_id: 1, message }
                if message == "fused projection failure"
        ));
        assert!(waiting_finished.load(Ordering::SeqCst));
        assert_eq!(supervisor.task_count(), 0);
    }

    #[tokio::test]
    async fn fused_simultaneous_failures_keep_both_primary_identities() {
        let mut supervisor = TaskSupervisor::new(CancellationToken::new());
        supervisor.spawn_pair_with_failure_signals(
            "operator:rolling",
            |_| async {
                Err(CalcFlowError::Internal {
                    message: "first".into(),
                })
            },
            "operator:projection",
            |_| async {
                Err(CalcFlowError::Internal {
                    message: "second".into(),
                })
            },
        );
        let report = supervisor.join_all().await;
        assert_eq!(report.primary_errors().len(), 2);
        assert_eq!(report.errors[0].task_id, TaskId::new(0));
        assert_eq!(report.errors[1].task_id, TaskId::new(1));
        assert_eq!(supervisor.task_count(), 0);
    }

    #[tokio::test]
    async fn aborting_fused_driver_drops_both_futures_and_clears_registry() {
        struct MarkDropped(Arc<AtomicBool>);
        impl Drop for MarkDropped {
            fn drop(&mut self) {
                self.0.store(true, Ordering::SeqCst);
            }
        }
        let first = Arc::new(AtomicBool::new(false));
        let second = Arc::new(AtomicBool::new(false));
        let first_owned = MarkDropped(Arc::clone(&first));
        let second_owned = MarkDropped(Arc::clone(&second));
        let mut supervisor = TaskSupervisor::new(CancellationToken::new());
        supervisor.spawn_pair_with_failure_signals(
            "operator:rolling",
            move |_| async move {
                let _owned = first_owned;
                std::future::pending().await
            },
            "operator:projection",
            move |_| async move {
                let _owned = second_owned;
                std::future::pending().await
            },
        );
        supervisor.tasks.abort_all();
        let report = supervisor.join_all().await;
        assert_eq!(report.errors.len(), 2);
        assert!(first.load(Ordering::SeqCst));
        assert!(second.load(Ordering::SeqCst));
        assert_eq!(supervisor.task_count(), 0);
        assert!(supervisor.stable_ids.is_empty());
    }

    #[tokio::test]
    async fn aborting_fused_driver_preserves_an_already_settled_member() {
        for outcome in ["ok", "error", "panic"] {
            let mut supervisor = TaskSupervisor::new(CancellationToken::new());
            let (entered_tx, entered_rx) = oneshot::channel();
            supervisor.spawn_pair_with_failure_signals(
                "operator:rolling",
                move |_| async move {
                    entered_tx.send(()).unwrap();
                    match outcome {
                        "ok" => Ok(()),
                        "error" => Err(CalcFlowError::Internal {
                            message: "original rolling failure".into(),
                        }),
                        _ => panic!("original rolling panic"),
                    }
                },
                "operator:projection",
                |_| std::future::pending(),
            );
            entered_rx.await.unwrap();
            supervisor.tasks.abort_all();
            let report = supervisor.join_all().await;
            let completed_error = report
                .errors
                .iter()
                .find(|error| error.task_id == TaskId::new(0));
            match outcome {
                "ok" => assert!(
                    completed_error.is_none(),
                    "completed success became a join failure"
                ),
                "error" => assert!(matches!(
                    &completed_error.unwrap().error,
                    CalcFlowError::Internal { message } if message == "original rolling failure"
                )),
                _ => assert!(matches!(
                    &completed_error.unwrap().error,
                    CalcFlowError::TaskPanicked { task_id: 0, message } if message == "original rolling panic"
                )),
            }
            assert_eq!(report.errors.last().unwrap().task_id, TaskId::new(1));
            assert!(supervisor.settled.lock().is_empty());
            assert_eq!(supervisor.task_count(), 0);
        }
    }

    #[tokio::test]
    async fn aborting_driver_preserves_published_retained_failure() {
        let mut supervisor = TaskSupervisor::new(CancellationToken::new());
        let (mut sender, receiver) =
            crate::edge_channel("retained-edge", crate::EdgeBudget::default()).unwrap();
        let lifetime = Arc::new(());
        let weak = Arc::downgrade(&lifetime);
        let (entered, ready) = oneshot::channel();
        let pair = supervisor.prepare_pair_with_failure_signals(
            "waiting",
            |_| std::future::pending::<crate::Result<()>>(),
            "failed",
            |_| async move {
                entered.send(()).unwrap();
                super::RetainedTaskResult {
                    result: Err(CalcFlowError::OperatorReason {
                        node_id: "asof".into(),
                        reason_code: crate::StreamingFailureReason::AsofDuplicateIdentity,
                        message: "original downstream failure".into(),
                    }),
                    retained: (receiver, lifetime),
                }
            },
        );
        supervisor.spawn_prepared_pair(pair);
        ready.await.unwrap();
        supervisor.cancel_and_abort();
        let report = supervisor.join_all().await;
        assert_eq!(supervisor.task_count(), 0);
        assert!(weak.upgrade().is_none());
        assert_eq!(report.primary_errors().len(), 1, "{report:?}");
        assert_eq!(report.errors[0].task_id, TaskId::new(1));
        assert!(matches!(
            report.errors[0].error,
            CalcFlowError::OperatorReason {
                reason_code: crate::StreamingFailureReason::AsofDuplicateIdentity,
                ..
            }
        ));
        assert_eq!(report.errors[1].task_id, TaskId::new(0));
        assert!(matches!(
            sender.send(crate::StreamMessage::end_of_input()).await,
            Err(CalcFlowError::EdgeClosed { .. })
        ));
    }

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
