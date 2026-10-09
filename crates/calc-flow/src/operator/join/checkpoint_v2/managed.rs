use super::super::{OperatorStateSnapshot, StreamJoinOperator};
use super::{candidate::Prepared, construction};
use crate::runtime::streaming::gather_work::{
    AdmissionFailure, GatherStop, ObservedTicket, TaskId,
};
use crate::{CalcFlowError, Result, StreamJobContext};

impl StreamJoinOperator {
    pub(crate) async fn restore_managed_snapshot(
        &mut self,
        snapshot: &mut OperatorStateSnapshot,
        job: &StreamJobContext,
        task: Option<TaskId>,
    ) -> Result<()> {
        job.check_cancelled()?;
        if snapshot
            .inline_metadata
            .get("layout_version")
            .and_then(serde_json::Value::as_u64)
            != Some(2)
        {
            return self.restore_managed_metadata(snapshot, job, task).await;
        }
        self.restore_managed_v2(snapshot, job, task).await
    }

    async fn restore_managed_v2(
        &mut self,
        snapshot: &mut OperatorStateSnapshot,
        job: &StreamJobContext,
        task: Option<TaskId>,
    ) -> Result<()> {
        self.runtime.runtime()?;
        let mut construction = match construction::prepare(self, snapshot, job, task).await {
            Ok(construction) => construction,
            Err(error) => {
                job.check_cancelled()?;
                return Err(admission_error(error));
            }
        };
        let admission = construction
            .scope
            .submit_observed_work(
                construction.work.take().expect("prepared V2 work"),
                construction.control.new_empty(),
                construction.stop.clone(),
                construction.retirement.take().expect("observed V2 input"),
                &mut self.compaction_cleanup,
            )
            .await;
        let outcome = match admission {
            Ok(ticket) => self.finish_v2_ticket(ticket, job).await,
            Err(error) => Err(admission_error(error)),
        };
        job.check_cancelled()?;
        self.wait_v2_cleanup(&construction.stop, job).await?;
        outcome
    }

    async fn finish_v2_ticket(
        &mut self,
        ticket: ObservedTicket<Prepared>,
        job: &StreamJobContext,
    ) -> Result<()> {
        let output = ticket.finish().await?;
        output.install(|prepared| self.install_v2(prepared, &|| job.check_cancelled()))
    }

    async fn wait_v2_cleanup(&mut self, stop: &GatherStop, job: &StreamJobContext) -> Result<()> {
        if let Some(cleanup) = &self.compaction_cleanup {
            cleanup.wait_job(stop, job).await?;
        }
        self.compaction_cleanup = None;
        Ok(())
    }
}

fn admission_error(error: AdmissionFailure) -> CalcFlowError {
    match error {
        AdmissionFailure::Budget {
            source: datafusion::common::DataFusionError::ResourcesExhausted(message),
            ..
        } => CalcFlowError::DataFusion {
            node_id: None,
            message,
        },
        error => error.into(),
    }
}
