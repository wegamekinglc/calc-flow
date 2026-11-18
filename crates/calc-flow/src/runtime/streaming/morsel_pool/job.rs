use std::sync::{Arc, Mutex};

use futures::future::BoxFuture;

#[cfg(test)]
use std::sync::atomic::{AtomicU64, AtomicUsize, Ordering};

use super::{MorselClient, MorselPool, MorselPoolConfig, MorselTicket, lock};
use crate::runtime::streaming::supervisor::{TaskFailure, TaskId};
use crate::{CalcFlowError, Result};

const MAX_JOB_MORSEL_ROWS: usize = 64 * 1024 * 1024;
const MAX_JOB_MORSEL_BYTES: usize = 4 << 30;

#[derive(Clone)]
pub(crate) struct JobMorselOwner {
    inner: Arc<JobMorselInner>,
}

struct JobMorselInner {
    run_id: String,
    config: MorselPoolConfig,
    state: Mutex<JobMorselState>,
    #[cfg(test)]
    worker_limit: AtomicUsize,
    #[cfg(test)]
    launches: Arc<AtomicU64>,
    #[cfg(test)]
    hooks: Mutex<MorselTestHooks>,
}

struct JobMorselState {
    pool: Option<MorselPool>,
    abandoned: Vec<BoxFuture<'static, Vec<TaskFailure>>>,
    #[cfg(test)]
    client: Option<MorselClient>,
    closed: bool,
}

#[cfg(test)]
#[derive(Clone, Default)]
pub(crate) struct MorselTestHooks {
    pub(crate) before_run: Option<Arc<dyn Fn(usize) + Send + Sync>>,
    pub(crate) after_run: Option<Arc<dyn Fn(usize) + Send + Sync>>,
    pub(crate) panic_before_second_launch: bool,
}

impl JobMorselOwner {
    pub(crate) fn new(job_id: u64, workers: usize, #[cfg(test)] launches: Arc<AtomicU64>) -> Self {
        let workers = workers.max(2);
        Self {
            inner: Arc::new(JobMorselInner {
                run_id: job_id.to_string(),
                config: MorselPoolConfig {
                    workers,
                    max_morsels: workers.saturating_mul(2),
                    max_rows: MAX_JOB_MORSEL_ROWS,
                    max_bytes: MAX_JOB_MORSEL_BYTES,
                    inline_rows: 0,
                },
                state: Mutex::new(JobMorselState {
                    pool: None,
                    abandoned: Vec::new(),
                    #[cfg(test)]
                    client: None,
                    closed: false,
                }),
                #[cfg(test)]
                worker_limit: AtomicUsize::new(0),
                #[cfg(test)]
                launches,
                #[cfg(test)]
                hooks: Mutex::new(MorselTestHooks::default()),
            }),
        }
    }

    pub(crate) fn client(&self) -> Result<MorselClient> {
        let mut state = lock(&self.inner.state);
        if state.closed {
            return Err(CalcFlowError::Cancelled {
                run_id: self.inner.run_id.clone(),
            });
        }
        if state.pool.is_none() {
            state.pool = Some(MorselPool::new(&self.inner.run_id, self.inner.config)?);
        }
        let client = state
            .pool
            .as_ref()
            .expect("the job pool was initialized")
            .client();
        #[cfg(test)]
        {
            state.client = Some(client.clone());
        }
        Ok(client)
    }

    pub(crate) fn admits(&self, rows: usize, bytes: usize) -> bool {
        rows <= self.inner.config.max_rows && bytes <= self.inner.config.max_bytes
    }

    pub(crate) fn retain_abandoned<T, F>(
        &self,
        tickets: Vec<MorselTicket<T>>,
        task_id: TaskId,
        task_name: String,
        classify: F,
    ) where
        T: Send + 'static,
        F: Fn(Result<T>) -> Option<CalcFlowError> + Send + Sync + 'static,
    {
        if tickets.is_empty() {
            return;
        }
        let abandoned = async move {
            let mut failures = Vec::new();
            for ticket in tickets {
                if let Some(error) = classify(ticket.join_async().await) {
                    failures.push(TaskFailure {
                        indicator_diagnostic: None,
                        task_id,
                        task_name: task_name.clone(),
                        error,
                    });
                }
            }
            failures
        };
        lock(&self.inner.state).abandoned.push(Box::pin(abandoned));
    }

    pub(crate) fn retain_results<T, F>(
        &self,
        results: Vec<Result<T>>,
        task_id: TaskId,
        task_name: &str,
        classify: F,
    ) where
        F: Fn(Result<T>) -> Option<CalcFlowError> + Send + Sync + 'static,
    {
        if results.is_empty() {
            return;
        }
        let failures = results
            .into_iter()
            .filter_map(classify)
            .map(|error| TaskFailure {
                indicator_diagnostic: None,
                task_id,
                task_name: task_name.to_owned(),
                error,
            })
            .collect::<Vec<_>>();
        if failures.is_empty() {
            return;
        }
        lock(&self.inner.state)
            .abandoned
            .push(Box::pin(async move { failures }));
    }

    pub(crate) async fn drain_failures(&self) -> Vec<TaskFailure> {
        let abandoned = {
            let mut state = lock(&self.inner.state);
            std::mem::take(&mut state.abandoned)
        };
        let mut failures = Vec::new();
        for abandoned in abandoned {
            failures.extend(abandoned.await);
        }
        failures
    }

    pub(crate) fn parallelism(&self, available: usize) -> usize {
        #[cfg(test)]
        {
            let configured = self.inner.worker_limit.load(Ordering::SeqCst);
            if configured > 0 {
                return configured;
            }
        }
        available.min(self.inner.config.workers)
    }

    pub(crate) async fn close_and_drain(&self) -> Result<()> {
        let pool = {
            let mut state = lock(&self.inner.state);
            state.closed = true;
            state.pool.take()
        };
        let Some(mut pool) = pool else {
            return Ok(());
        };
        tokio::task::spawn_blocking(move || pool.close_and_join())
            .await
            .map_err(|error| CalcFlowError::Internal {
                message: format!("morsel pool teardown join failed: {error}"),
            })?
    }

    #[cfg(test)]
    pub(crate) fn set_worker_limit_for_test(&self, workers: usize) {
        assert!(workers > 0);
        self.inner.worker_limit.store(workers, Ordering::SeqCst);
    }

    #[cfg(test)]
    pub(crate) fn set_hooks(&self, hooks: MorselTestHooks) {
        *lock(&self.inner.hooks) = hooks;
    }

    #[cfg(test)]
    pub(crate) fn hooks(&self) -> MorselTestHooks {
        lock(&self.inner.hooks).clone()
    }

    #[cfg(test)]
    pub(crate) fn record_launch(&self) {
        self.inner.launches.fetch_add(1, Ordering::SeqCst);
    }

    #[cfg(test)]
    pub(crate) fn active_bytes(&self) -> usize {
        lock(&self.inner.state)
            .client
            .as_ref()
            .map_or(0, |client| client.usage().bytes)
    }
}

impl Drop for JobMorselInner {
    fn drop(&mut self) {
        let pool = lock(&self.state).pool.take();
        if let Some(mut pool) = pool {
            let _ = pool.close_and_join();
        }
    }
}
