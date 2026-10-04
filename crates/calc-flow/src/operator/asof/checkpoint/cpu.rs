use super::super::{StreamAsofJoinOperator, state::PayloadBatch};
use crate::{
    Result,
    runtime::streaming::gather_work::{
        AdmissionFailure, GatherOperatorId, GatherStop, OwnedCpuWork,
    },
};
use datafusion::execution::memory_pool::MemoryReservation;
use std::sync::Arc;

pub(super) struct PayloadWork {
    pub pending: Vec<Arc<PayloadBatch>>,
    pub workspace: MemoryReservation,
    pub limit: usize,
    pub name: String,
}

impl OwnedCpuWork for PayloadWork {
    type Output = ();

    fn run(self, stop: &GatherStop) -> Result<()> {
        debug_assert!(self.workspace.size() > 0);
        for batch in &self.pending {
            stop.check()?;
            batch.ensure_encoded(self.limit, &self.name)?;
        }
        stop.check()
    }
}

struct RestoreWork {
    operator: Box<StreamAsofJoinOperator>,
    snapshot: crate::OperatorStateSnapshot,
    progress: crate::IngressProgressSnapshot,
    frontier: Option<crate::EventTime>,
}

impl OwnedCpuWork for RestoreWork {
    type Output = Box<StreamAsofJoinOperator>;

    fn run(mut self, stop: &GatherStop) -> Result<Self::Output> {
        self.operator.restore_with_progress_checked(
            &self.snapshot,
            &self.progress,
            self.frontier,
            &|| stop.check(),
        )?;
        stop.check()?;
        Ok(self.operator)
    }
}

impl StreamAsofJoinOperator {
    pub(crate) async fn restore_managed(
        self: Box<Self>,
        snapshot: crate::OperatorStateSnapshot,
        progress: crate::IngressProgressSnapshot,
        frontier: Option<crate::EventTime>,
        job: &crate::StreamJobContext,
        task: Option<crate::runtime::streaming::gather_work::TaskId>,
    ) -> Result<Box<Self>> {
        if snapshot.inline_metadata.contains_key("source_replay") {
            return self
                .restore_source_replay(snapshot, progress, frontier, job)
                .await;
        }
        job.check_cancelled()?;
        let credit = self.reserve_workspace(crate::operator::asof::checked(
            &self.name,
            1024 + size_of::<RestoreWork>() as u64,
            self.name.len() as u64 * 2,
        )?)?;
        let name = self.name.clone();
        let operator = GatherOperatorId::new(Arc::from(name.as_str())).with_task(task);
        let scope = job.gather_owner().client(operator).scope()?;
        let work = RestoreWork {
            operator: self,
            snapshot,
            progress,
            frontier,
        };
        let ticket = scope
            .submit_work(work, credit, GatherStop::from_job(job))
            .await
            .map_err(|failure| match failure {
                AdmissionFailure::Budget { .. } => crate::operator::asof::reason(
                    &name,
                    crate::StreamingFailureReason::AsofWorkspaceLimitExceeded,
                    "ASOF restore work admission exceeds max_state_bytes",
                ),
                AdmissionFailure::Runtime(error) => error,
            })?;
        let output = ticket.finish().await?;
        job.check_cancelled()?;
        Ok(output.value)
    }
}

struct CaptureWork {
    operator: Box<StreamAsofJoinOperator>,
    epoch: crate::Epoch,
}

impl OwnedCpuWork for CaptureWork {
    type Output = (Box<StreamAsofJoinOperator>, crate::OperatorStateSnapshot);

    fn run(mut self, stop: &GatherStop) -> Result<Self::Output> {
        stop.check()?;
        let snapshot = self.operator.capture(self.epoch)?;
        stop.check()?;
        Ok((self.operator, snapshot))
    }
}

impl StreamAsofJoinOperator {
    pub(crate) async fn capture_managed(
        self: Box<Self>,
        epoch: crate::Epoch,
        job: &crate::StreamJobContext,
        task: Option<crate::runtime::streaming::gather_work::TaskId>,
    ) -> Result<(Box<Self>, crate::OperatorStateSnapshot)> {
        job.check_cancelled()?;
        let credit = self.reserve_workspace(crate::operator::asof::checked(
            &self.name,
            1024 + size_of::<CaptureWork>() as u64,
            self.name.len() as u64 * 2,
        )?)?;
        let name = self.name.clone();
        let operator = GatherOperatorId::new(Arc::from(name.as_str())).with_task(task);
        let scope = job.gather_owner().client(operator).scope()?;
        let work = CaptureWork {
            operator: self,
            epoch,
        };
        let ticket = scope
            .submit_work(work, credit, GatherStop::from_job(job))
            .await
            .map_err(|failure| match failure {
                AdmissionFailure::Budget { .. } => crate::operator::asof::reason(
                    &name,
                    crate::StreamingFailureReason::AsofWorkspaceLimitExceeded,
                    "ASOF capture work admission exceeds max_state_bytes",
                ),
                AdmissionFailure::Runtime(error) => error,
            })?;
        let output = ticket.finish().await?;
        job.check_cancelled()?;
        Ok(output.value)
    }
}
