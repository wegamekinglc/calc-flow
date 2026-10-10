use super::super::{PreparedJoinBatch, StreamJoinOperator};
use super::{
    bounds,
    funding::OutputFunding,
    inputs::Inputs,
    worker::{self, Fragment, FragmentWork},
};
use crate::runtime::streaming::gather_work::{
    AdmissionFailure, GatherOperatorId, GatherStop, ObservedTicket,
};
use crate::{Result, StreamJobContext, StreamOperatorContext};
use datafusion::arrow::record_batch::RecordBatch;
use datafusion::execution::memory_pool::MemoryReservation;
use std::{ops::Range, sync::Arc};

impl StreamJoinOperator {
    pub(in crate::operator::join) async fn owned_output_chunk(
        &mut self,
        prepared: &PreparedJoinBatch,
        range: Range<usize>,
        context: &StreamOperatorContext<'_>,
    ) -> Result<Option<RecordBatch>> {
        let Some(work) = self.prepare_output_work(prepared, range, context).await? else {
            return Ok(None);
        };
        let schema = Arc::clone(&work.schema);
        let fragments = self.run_fragment_work(work, context).await;
        self.await_compaction_release(context).await?;
        let Some(fragments) = fragments? else {
            return Ok(None);
        };
        context.check_cancelled()?;
        worker::into_record(fragments, schema, &self.name).map(Some)
    }

    async fn prepare_output_work(
        &mut self,
        prepared: &PreparedJoinBatch,
        range: Range<usize>,
        context: &StreamOperatorContext<'_>,
    ) -> Result<Option<Arc<FragmentWork>>> {
        let Some(bounds) =
            bounds::certify(&self.output_materializer(prepared), range.clone(), context).await?
        else {
            context.check_cancelled()?;
            return Ok(None);
        };
        let Some((workspace, funding)) = self.reserve_output_work(bounds)? else {
            context.check_cancelled()?;
            return Ok(None);
        };
        let schema = Arc::clone(self.output_ports[0].schema().expect("compiled Join output"));
        let inputs = self
            .capture_output_inputs(prepared, range, workspace, context)
            .await?;
        Ok(Some(Arc::new(FragmentWork {
            inputs,
            schema,
            funding,
            name: Arc::from(self.name.as_str()),
            #[cfg(test)]
            hook: self.materialize_unit_test_hook.clone(),
            #[cfg(test)]
            take_hook: self.materialize_take_test_hook.clone(),
        })))
    }

    fn reserve_output_work(
        &mut self,
        bounds: bounds::Certificate,
    ) -> Result<Option<(Arc<MemoryReservation>, Arc<OutputFunding>)>> {
        let Some(workspace) = self.optional_credit(bounds.workspace)? else {
            return Ok(None);
        };
        let Some(output) = self.optional_credit(bounds.output)? else {
            return Ok(None);
        };
        Ok(Some((
            Arc::new(workspace),
            Arc::new(OutputFunding { _credit: output }),
        )))
    }

    async fn capture_output_inputs(
        &mut self,
        prepared: &PreparedJoinBatch,
        range: Range<usize>,
        workspace: Arc<MemoryReservation>,
        context: &StreamOperatorContext<'_>,
    ) -> Result<Inputs> {
        let (released, receiver) = tokio::sync::oneshot::channel();
        self.compaction_release = Some(receiver);
        self.probe_control = Some(Arc::clone(&workspace));
        Inputs::capture(
            &self.output_materializer(prepared),
            range,
            workspace,
            released,
            context,
        )
        .await
    }

    async fn run_fragment_work(
        &mut self,
        work: Arc<FragmentWork>,
        context: &StreamOperatorContext<'_>,
    ) -> Result<Option<Vec<Fragment>>> {
        let scope = context
            .gather_client(GatherOperatorId::new(Arc::clone(&work.name)))
            .scope()?;
        let credit = self
            .probe_control
            .as_ref()
            .expect("paid output snapshot")
            .new_empty();
        let retirement = context.job().gather_owner().retain_retirement()?;
        let ticket = scope
            .submit_observed_parallel_work(
                work,
                credit,
                GatherStop::from_job(context.job()),
                retirement,
                &mut self.compaction_cleanup,
            )
            .await;
        finish_submission(ticket, context).await
    }
}

async fn finish_submission<T: Send + 'static>(
    ticket: std::result::Result<ObservedTicket<T>, AdmissionFailure>,
    context: &StreamOperatorContext<'_>,
) -> Result<Option<T>> {
    let ticket = match ticket {
        Ok(ticket) => ticket,
        Err(AdmissionFailure::Budget { .. }) => {
            context.check_cancelled()?;
            return Ok(None);
        }
        Err(AdmissionFailure::Runtime(error)) => {
            context.check_cancelled()?;
            return Err(error);
        }
    };
    let output = tokio::select! {
        biased;
        () = materialization_stop(context.job()) => { context.check_cancelled()?; unreachable!("materialization stopped") },
        result = ticket.finish() => result,
    };
    context.check_cancelled()?;
    let mut value = None;
    output?.install(|output| {
        value = Some(output);
        Ok(())
    })?;
    Ok(value)
}

async fn materialization_stop(job: &StreamJobContext) {
    let deadline = async {
        match job.deadline() {
            Some(deadline) => {
                tokio::time::sleep(
                    (*deadline - chrono::Utc::now())
                        .to_std()
                        .unwrap_or_default(),
                )
                .await;
            }
            None => std::future::pending().await,
        }
    };
    tokio::select! {
        () = job.cancellation().cancelled() => {},
        () = deadline => {},
    }
}
