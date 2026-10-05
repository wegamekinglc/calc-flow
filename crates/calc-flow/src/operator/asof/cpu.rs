use super::StreamAsofJoinOperator;
use crate::{
    Result, StreamOperatorContext,
    runtime::streaming::gather_work::{
        AdmissionFailure, GatherOperatorId, GatherScope, GatherStop, OwnedCpuWork, ParallelCpuWork,
        WorkOutput,
    },
};
use datafusion::execution::memory_pool::MemoryReservation;
use std::sync::Arc;

impl StreamAsofJoinOperator {
    pub(super) async fn run_cpu_work<W: OwnedCpuWork>(
        &self,
        work: W,
        context: &StreamOperatorContext<'_>,
    ) -> Result<W::Output> {
        context.check_cancelled()?;
        let credit = self.cpu_work_credit(size_of::<W>())?;
        let operator = GatherOperatorId::new(Arc::from(self.name.as_str()));
        let scope = context.gather_client(operator).scope()?;
        let ticket = scope
            .submit_work(work, credit, GatherStop::from_job(context.job()))
            .await
            .map_err(|failure| match failure {
                AdmissionFailure::Budget { .. } => super::reason(
                    &self.name,
                    crate::StreamingFailureReason::AsofWorkspaceLimitExceeded,
                    "ASOF CPU work admission exceeds max_state_bytes",
                ),
                AdmissionFailure::Runtime(error) => error,
            })?;
        let output = ticket.finish().await?;
        context.check_cancelled()?;
        Ok(output.value)
    }

    pub(super) fn cpu_work_credit(&self, work_bytes: usize) -> Result<MemoryReservation> {
        self.reserve_workspace(super::checked(
            &self.name,
            1024 + work_bytes as u64,
            self.name.len() as u64 * 2,
        )?)
    }
}

pub(super) async fn finish_parallel_work<W: ParallelCpuWork>(
    work: W,
    credit: MemoryReservation,
    scope: &GatherScope,
    context: &StreamOperatorContext<'_>,
) -> Result<Option<WorkOutput<Vec<W::Output>>>> {
    let ticket = match scope
        .submit_parallel_work(Arc::new(work), credit, GatherStop::from_job(context.job()))
        .await
    {
        Ok(ticket) => ticket,
        Err(AdmissionFailure::Budget { .. }) => return Ok(None),
        Err(AdmissionFailure::Runtime(error)) => return Err(error),
    };
    let output = ticket.finish().await?;
    context.check_cancelled()?;
    Ok(Some(output))
}
