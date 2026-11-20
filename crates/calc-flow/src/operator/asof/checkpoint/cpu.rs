use super::super::StreamAsofJoinOperator;
use crate::{
    CalcFlowError, Result,
    runtime::streaming::{
        gather_work::{GatherStop, TaskId},
        morsel_pool::{JobMorselOwner, MorselCost, MorselLease, MorselTicket},
    },
};
use datafusion::execution::memory_pool::MemoryReservation;

struct RestoreWork {
    operator: Box<StreamAsofJoinOperator>,
    snapshot: crate::OperatorStateSnapshot,
    progress: crate::IngressProgressSnapshot,
    frontier: Option<crate::EventTime>,
    _credit: MemoryReservation,
}

impl RestoreWork {
    fn run(mut self, lease: &MorselLease) -> Result<Box<StreamAsofJoinOperator>> {
        self.operator.restore_with_progress_checked(
            &self.snapshot,
            &self.progress,
            self.frontier,
            &|| lease.check(),
        )?;
        lease.check()?;
        Ok(self.operator)
    }
}

struct CancelOnDrop<T: Send + 'static> {
    ticket: Option<MorselTicket<T>>,
    owner: JobMorselOwner,
    task: Option<TaskId>,
    task_name: String,
}

impl<T: Send + 'static> CancelOnDrop<T> {
    fn disarm(&mut self) {
        self.ticket = None;
    }
}

impl<T: Send + 'static> Drop for CancelOnDrop<T> {
    fn drop(&mut self) {
        let Some(ticket) = self.ticket.take() else {
            return;
        };
        ticket.cancel();
        if let Some(task) = self.task {
            self.owner.retain_abandoned(
                vec![ticket],
                task,
                std::mem::take(&mut self.task_name),
                |result| match result {
                    Ok(_) | Err(CalcFlowError::Cancelled { .. }) => None,
                    Err(error) => Some(error),
                },
            );
        }
    }
}

async fn execute_managed<T, O, F>(
    job: &crate::StreamJobContext,
    name: &str,
    stage: &str,
    task: Option<TaskId>,
    input: T,
    cost: MorselCost,
    operation: F,
) -> Result<O>
where
    T: Send + 'static,
    O: Send + 'static,
    F: FnOnce(T, &MorselLease) -> Result<O> + Send + 'static,
{
    let owner = job.morsel_owner().ok_or_else(|| CalcFlowError::Internal {
        message: format!("ASOF {name} managed checkpoint has no job morsel pool"),
    })?;
    let client = owner.client()?;
    let stop = GatherStop::from_job(job);
    let submission = client.submit_async(input, cost, operation);
    tokio::pin!(submission);
    let ticket = tokio::select! {
        biased;
        () = stop.wait() => {
            return Err(CalcFlowError::Cancelled { run_id: job.job_id().to_string() });
        }
        result = &mut submission => result.map_err(|error| match error {
            CalcFlowError::InvalidArgument { ref field, .. }
                if matches!(field.as_str(), "morsel.rows" | "morsel.bytes" | "morsel.count") =>
            {
                crate::operator::asof::reason(
                    name,
                    crate::StreamingFailureReason::AsofWorkspaceLimitExceeded,
                    &format!("ASOF {stage} work admission exceeds max_state_bytes"),
                )
            }
            error => error,
        })?,
    };
    let mut cancel = CancelOnDrop {
        ticket: Some(ticket.clone()),
        owner: owner.clone(),
        task,
        task_name: name.to_owned(),
    };
    let output = tokio::select! {
        biased;
        () = stop.wait() => {
            return Err(CalcFlowError::Cancelled { run_id: job.job_id().to_string() });
        }
        result = ticket.join_async() => result,
    };
    cancel.disarm();
    stop.check()?;
    output
}

impl StreamAsofJoinOperator {
    fn cpu_work_credit(&self, work_bytes: usize) -> Result<MemoryReservation> {
        self.reserve_workspace(super::super::checked(
            &self.name,
            1024 + work_bytes as u64,
            self.name.len() as u64 * 2,
        )?)
    }

    pub(crate) async fn restore_managed(
        self: Box<Self>,
        snapshot: crate::OperatorStateSnapshot,
        progress: crate::IngressProgressSnapshot,
        frontier: Option<crate::EventTime>,
        job: &crate::StreamJobContext,
        task: Option<TaskId>,
    ) -> Result<Box<Self>> {
        if snapshot.inline_metadata.contains_key("source_replay") {
            return self
                .restore_source_replay(snapshot, progress, frontier, job)
                .await;
        }
        self.restore_native_managed(snapshot, progress, frontier, job, task)
            .await
    }

    pub(in crate::operator::asof) async fn restore_native_managed(
        self: Box<Self>,
        snapshot: crate::OperatorStateSnapshot,
        progress: crate::IngressProgressSnapshot,
        frontier: Option<crate::EventTime>,
        job: &crate::StreamJobContext,
        task: Option<TaskId>,
    ) -> Result<Box<Self>> {
        job.check_cancelled()?;
        let credit = self.cpu_work_credit(size_of::<RestoreWork>())?;
        let name = self.name.clone();
        let cost = MorselCost {
            rows: 1,
            scratch_bytes: credit.size(),
            output_bytes: 0,
        };
        let work = RestoreWork {
            operator: self,
            snapshot,
            progress,
            frontier,
            _credit: credit,
        };
        execute_managed(job, &name, "restore", task, work, cost, RestoreWork::run).await
    }
}

struct CaptureWork {
    operator: Box<StreamAsofJoinOperator>,
    epoch: crate::Epoch,
    _credit: MemoryReservation,
}

impl CaptureWork {
    fn run(
        mut self,
        lease: &MorselLease,
    ) -> Result<(Box<StreamAsofJoinOperator>, crate::OperatorStateSnapshot)> {
        lease.check()?;
        let snapshot = self.operator.capture(self.epoch)?;
        lease.check()?;
        Ok((self.operator, snapshot))
    }
}

impl StreamAsofJoinOperator {
    pub(crate) async fn capture_managed(
        self: Box<Self>,
        epoch: crate::Epoch,
        job: &crate::StreamJobContext,
        task: Option<TaskId>,
    ) -> Result<(Box<Self>, crate::OperatorStateSnapshot)> {
        job.check_cancelled()?;
        let credit = self.cpu_work_credit(size_of::<CaptureWork>())?;
        let name = self.name.clone();
        let cost = MorselCost {
            rows: 1,
            scratch_bytes: credit.size(),
            output_bytes: 0,
        };
        let work = CaptureWork {
            operator: self,
            epoch,
            _credit: credit,
        };
        execute_managed(job, &name, "capture", task, work, cost, CaptureWork::run).await
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{CancellationToken, JsonMap, runtime::streaming::morsel_pool::JobMorselOwner};
    use std::sync::{Arc, atomic::AtomicU64};

    #[tokio::test]
    async fn abandoned_managed_work_retains_its_task_failure() {
        let morsels = JobMorselOwner::new(92, 2, Arc::new(AtomicU64::new(0)));
        let job = crate::StreamJobContext::new(
            92,
            "asof",
            JsonMap::new(),
            None,
            CancellationToken::new(),
        )
        .with_morsel_owner(morsels.clone());
        let task_id = TaskId::new(17);
        let (entered_tx, mut entered_rx) = tokio::sync::mpsc::unbounded_channel();
        let (release_tx, mut release_rx) = tokio::sync::mpsc::unbounded_channel();
        let running = tokio::spawn(async move {
            execute_managed(
                &job,
                "asof",
                "capture",
                Some(task_id),
                (),
                MorselCost {
                    rows: 1,
                    scratch_bytes: 64,
                    output_bytes: 0,
                },
                move |(), _lease| {
                    entered_tx.send(()).unwrap();
                    release_rx.blocking_recv().unwrap();
                    Err::<(), _>(CalcFlowError::Internal {
                        message: "abandoned ASOF capture failed".into(),
                    })
                },
            )
            .await
        });
        entered_rx.recv().await.unwrap();

        running.abort();
        let _ = running.await;
        release_tx.send(()).unwrap();
        morsels.close_and_drain().await.unwrap();
        let failures = morsels.drain_failures().await;
        assert_eq!(failures.len(), 1);
        assert_eq!(failures[0].task_id, task_id);
        assert!(matches!(
            failures[0].error,
            CalcFlowError::Internal { ref message } if message == "abandoned ASOF capture failed"
        ));
    }
}
