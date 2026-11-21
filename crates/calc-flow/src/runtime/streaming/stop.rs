use std::{
    sync::{
        Arc,
        atomic::{AtomicBool, Ordering},
    },
    time::Duration,
};

use chrono::{DateTime, Utc};

use crate::{CalcFlowError, CancellationToken, Result};

#[derive(Clone)]
pub(crate) struct GatherStop {
    #[cfg(all(test, feature = "benchmark-internals"))]
    trace: crate::benchmark_internals::JoinCpuTraceOwner,
    #[cfg(all(test, feature = "benchmark-internals"))]
    operator: Arc<str>,
    run_id: Arc<str>,
    cancellation: CancellationToken,
    deadline: Option<DateTime<Utc>>,
    #[cfg(test)]
    deadline_clock: Option<tokio::sync::watch::Receiver<DateTime<Utc>>>,
    abandoned: Arc<AtomicBool>,
    #[cfg(test)]
    check_hook: Option<Arc<dyn Fn(bool) + Send + Sync>>,
}

impl GatherStop {
    pub(crate) fn from_job(job: &super::StreamJobContext) -> Self {
        Self {
            #[cfg(all(test, feature = "benchmark-internals"))]
            trace: job.join_cpu_trace().clone(),
            #[cfg(all(test, feature = "benchmark-internals"))]
            operator: "".into(),
            run_id: job.job_id().to_string().into(),
            cancellation: job.cancellation().clone(),
            deadline: job.deadline().copied(),
            #[cfg(test)]
            deadline_clock: job.deadline_clock_for_test(),
            abandoned: Arc::new(AtomicBool::new(false)),
            #[cfg(test)]
            check_hook: None,
        }
    }

    pub(crate) fn from_run(run: &crate::RunContext) -> Self {
        Self {
            #[cfg(all(test, feature = "benchmark-internals"))]
            trace: crate::benchmark_internals::JoinCpuTraceOwner::default(),
            #[cfg(all(test, feature = "benchmark-internals"))]
            operator: "".into(),
            run_id: run.run_id().into(),
            cancellation: run.cancellation().clone(),
            deadline: run.deadline().copied(),
            #[cfg(test)]
            deadline_clock: None,
            abandoned: Arc::new(AtomicBool::new(false)),
            #[cfg(test)]
            check_hook: None,
        }
    }

    #[cfg(all(test, feature = "benchmark-internals"))]
    pub(crate) fn join_cpu_trace(&self) -> &crate::benchmark_internals::JoinCpuTraceOwner {
        &self.trace
    }

    #[cfg(all(test, feature = "benchmark-internals"))]
    pub(crate) fn join_cpu_operator(&self) -> &str {
        &self.operator
    }

    #[cfg(all(test, feature = "benchmark-internals"))]
    pub(crate) fn with_join_cpu_operator(mut self, operator: Arc<str>) -> Self {
        self.operator = operator;
        self
    }

    #[cfg(test)]
    pub(crate) fn run_id(&self) -> &str {
        &self.run_id
    }

    #[cfg(test)]
    pub(crate) fn set_deadline_for_test(&mut self, deadline: Option<DateTime<Utc>>) {
        self.deadline = deadline;
    }

    pub(crate) fn check(&self) -> Result<()> {
        #[cfg(test)]
        if let Some(hook) = &self.check_hook {
            hook(false);
        }
        if self.abandoned.load(Ordering::Acquire)
            || self.cancellation.is_cancelled()
            || self.deadline.is_some_and(|deadline| {
                #[cfg(test)]
                if let Some(clock) = &self.deadline_clock {
                    return *clock.borrow() >= deadline;
                }
                Utc::now() >= deadline
            })
        {
            return Err(cancelled(&self.run_id));
        }
        Ok(())
    }

    /// Polls only the attempt's shared stop flag. The ticket owner must drive
    /// cancellation/deadline waiting and drop the ticket on stop; admission and
    /// result settlement must still use `check`.
    #[cfg(test)]
    pub(crate) fn check_abandoned(&self) -> Result<()> {
        #[cfg(test)]
        if let Some(hook) = &self.check_hook {
            hook(true);
        }
        if self.abandoned.load(Ordering::Acquire) {
            return Err(cancelled(&self.run_id));
        }
        Ok(())
    }

    #[cfg(test)]
    pub(crate) fn abandon(&self) {
        self.abandoned.store(true, Ordering::Release);
    }

    #[cfg(test)]
    pub(crate) fn with_check_hook(mut self, hook: Arc<dyn Fn(bool) + Send + Sync>) -> Self {
        self.check_hook = Some(hook);
        self
    }

    #[cfg(test)]
    pub(crate) fn abandon_for_test(&self) {
        self.abandon();
    }

    pub(crate) async fn wait(&self) {
        let deadline = deadline_wait(self.deadline);
        #[cfg(test)]
        let deadline = async {
            if let (Some(mut clock), Some(expires)) = (self.deadline_clock.clone(), self.deadline) {
                clock
                    .wait_for(|now| *now >= expires)
                    .await
                    .expect("deadline clock outlives its wait");
            } else {
                deadline.await;
            }
        };
        tokio::select! {
            () = self.cancellation.cancelled() => {},
            () = deadline => {},
        }
    }
}

async fn deadline_wait(deadline: Option<DateTime<Utc>>) {
    match deadline {
        Some(deadline) => {
            let delay = (deadline - Utc::now()).to_std().unwrap_or(Duration::ZERO);
            tokio::time::sleep(delay).await;
        }
        None => std::future::pending().await,
    }
}

fn cancelled(run_id: &str) -> CalcFlowError {
    CalcFlowError::Cancelled {
        run_id: run_id.into(),
    }
}
