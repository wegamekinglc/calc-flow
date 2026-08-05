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

/// Fully joined terminal report from a supervisor registry.
#[derive(Debug, Default)]
pub(crate) struct SupervisionReport {
    pub(crate) errors: Vec<TaskFailure>,
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
    primary_failures: Arc<Mutex<BTreeSet<TaskId>>>,
}

impl TaskFailureSignal {
    fn record_before_cancellation(&self) {
        let mut primary_failures = self.primary_failures.lock();
        if !self.cancellation.is_cancelled() {
            primary_failures.insert(self.task_id);
        }
    }

    /// Records this task as a failure trigger, then starts convergence.
    pub(crate) fn cancel_siblings(&self) {
        let mut primary_failures = self.primary_failures.lock();
        if !self.cancellation.is_cancelled() {
            primary_failures.insert(self.task_id);
        }
        self.cancellation.cancel();
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
    registry: BTreeMap<TaskId, String>,
    joined_errors: Vec<TaskFailure>,
    primary_failures: Arc<Mutex<BTreeSet<TaskId>>>,
    next_task_id: u64,
}

impl TaskSupervisor {
    pub(crate) fn new(cancellation: CancellationToken) -> Self {
        Self {
            cancellation,
            tasks: JoinSet::new(),
            stable_ids: HashMap::new(),
            registry: BTreeMap::new(),
            joined_errors: Vec::new(),
            primary_failures: Arc::new(Mutex::new(BTreeSet::new())),
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
            primary_failures: Arc::clone(&self.primary_failures),
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
                    let task_name = self
                        .registry
                        .remove(&task_id)
                        .unwrap_or_else(|| "unknown supervised task".into());
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
                        primary_failures: Arc::clone(&self.primary_failures),
                    };
                    failure_signal.record_before_cancellation();
                    failure_signal.cancel_siblings();
                    self.joined_errors.push(failure);
                    continue;
                }
            };
            self.registry.remove(&exit.task_id);
            if let Err(error) = exit.result {
                self.joined_errors.push(TaskFailure {
                    task_id: exit.task_id,
                    task_name: exit.task_name,
                    error,
                });
            }
        }
        let primary_failures = std::mem::take(&mut *self.primary_failures.lock());
        self.joined_errors.sort_by_key(|failure| {
            (
                !primary_failures.contains(&failure.task_id),
                failure.task_id,
            )
        });
        SupervisionReport {
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

fn panic_message(payload: &(dyn Any + Send)) -> String {
    if let Some(message) = payload.downcast_ref::<&str>() {
        (*message).to_owned()
    } else if let Some(message) = payload.downcast_ref::<String>() {
        message.clone()
    } else {
        "non-string panic payload".into()
    }
}

#[cfg(test)]
mod tests {
    use std::sync::{
        Arc,
        atomic::{AtomicBool, Ordering},
    };

    use tokio::sync::{Barrier, oneshot};

    use super::{TaskId, TaskSupervisor};
    use crate::{CalcFlowError, CancellationToken};

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
            [TaskId::new(0), TaskId::new(1)]
        );
    }

    #[tokio::test]
    async fn cancellation_convergence_error_does_not_replace_the_primary_trigger() {
        let cancellation = CancellationToken::new();
        let mut supervisor = TaskSupervisor::new(cancellation.clone());
        let convergence_id = supervisor.spawn("convergence", async move {
            cancellation.cancelled().await;
            Err(CalcFlowError::Internal {
                message: "failed while converging cancellation".into(),
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
}
