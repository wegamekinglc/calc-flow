use super::super::{
    AdmittedRow, MatchedPair, PreparedMatches, SidePlan, StreamJoinOperator, enforce_match_limit,
    native_lookup::{NativeKeys, eligible, scratch_error},
};
use super::{
    MIN_PROBE_ROWS, control,
    inputs::ProbeInputs,
    worker::{CountWork, FillWork, PairOutput},
};
use crate::runtime::streaming::gather_work::{
    AdmissionFailure, GatherOperatorId, GatherStop, ObservedTicket, ParallelCpuWork,
};
use crate::{Result, StreamJobContext, StreamOperatorContext};
use parking_lot::Mutex;
use std::sync::Arc;

enum Capture {
    Declined(Vec<AdmittedRow>),
    Ready(Arc<ProbeInputs>),
}

impl StreamJoinOperator {
    pub(in crate::operator::join) async fn owned_native_matches(
        &mut self,
        plan: &SidePlan,
        admitted: Vec<AdmittedRow>,
        context: &StreamOperatorContext<'_>,
    ) -> (Option<Result<PreparedMatches>>, Vec<AdmittedRow>) {
        let inputs = match self.capture_probe(plan, admitted) {
            Ok(Capture::Declined(admitted)) => return (None, admitted),
            Ok(Capture::Ready(inputs)) => inputs,
            Err(error) => return (Some(Err(error)), Vec::new()),
        };
        let result = self.dispatch_native_probe(&inputs, context).await;
        let recovered = self.recover_probe(inputs, context).await;
        let (admitted, keys) = match recovered {
            Ok(recovered) => recovered,
            Err(error) => return (Some(Err(error)), Vec::new()),
        };
        let result = result.map(|output| {
            output.map(|output| PreparedMatches {
                pairs: output.pairs,
                keys: Some(keys),
                credit: Some(output.credit),
            })
        });
        match result {
            Ok(None) => (None, admitted),
            Ok(Some(prepared)) => (Some(Ok(prepared)), admitted),
            Err(error) => (Some(Err(error)), admitted),
        }
    }

    fn capture_probe(&mut self, plan: &SidePlan, admitted: Vec<AdmittedRow>) -> Result<Capture> {
        if !self.probe_ready(plan, admitted.len())? {
            return Ok(Capture::Declined(admitted));
        }
        let Some(keys) = self.native_probe_keys(plan, &admitted)? else {
            return Ok(Capture::Declined(admitted));
        };
        let bytes = control::input_bytes(keys.keys.len(), &self.name)?;
        let Some(credit) = self.optional_credit(bytes)? else {
            return Ok(Capture::Declined(admitted));
        };
        let control = Arc::new(credit);
        let opposite = if plan.incoming_is_left {
            &self.state.right
        } else {
            &self.state.left
        };
        let index = opposite.1.as_ref().expect("captured native index");
        let slots = keys.keys.iter().map(|key| index.key_id(key)).collect();
        let (released, receiver) = tokio::sync::oneshot::channel();
        self.compaction_release = Some(receiver);
        self.probe_control = Some(Arc::clone(&control));
        Ok(Capture::Ready(Arc::new(ProbeInputs {
            admitted,
            keys: Some(keys),
            slots,
            opposite: Some(Arc::clone(&opposite.0)),
            index: Some(Arc::clone(index)),
            containers: self.v2_containers.clone(),
            bounds: self.spec.bounds,
            incoming_is_left: plan.incoming_is_left,
            control: Some(control),
            released: Some(released),
            #[cfg(test)]
            hook: self.probe_test_hook.take(),
        })))
    }

    fn probe_ready(&mut self, plan: &SidePlan, rows: usize) -> Result<bool> {
        if rows < MIN_PROBE_ROWS || !eligible(&self.compiled, self.input_schema(0)) {
            return Ok(false);
        }
        self.ensure_native_index(!plan.incoming_is_left)
    }

    async fn dispatch_native_probe(
        &mut self,
        inputs: &Arc<ProbeInputs>,
        context: &StreamOperatorContext<'_>,
    ) -> Result<Option<PairOutput>> {
        let work = Arc::new(CountWork {
            inputs: Arc::clone(inputs),
            limit: self.spec.limits.max_matches_per_input_batch,
        });
        let Some(count) = self.run_probe_unit(work, context).await? else {
            return Ok(None);
        };
        enforce_match_limit(
            count,
            &mut self.state.metrics.match_limit_failures,
            self.spec.limits.max_matches_per_input_batch,
            &self.name,
        )?;
        let bytes = count
            .checked_mul(size_of::<MatchedPair>() + 256)
            .ok_or_else(|| scratch_error(&self.name))?;
        let Some(credit) = self.optional_credit(bytes)? else {
            return Ok(None);
        };
        self.run_probe_unit(
            Arc::new(FillWork {
                inputs: Arc::clone(inputs),
                count,
                credit: Mutex::new(Some(credit)),
            }),
            context,
        )
        .await
    }

    async fn run_probe_unit<W: ParallelCpuWork>(
        &mut self,
        work: Arc<W>,
        context: &StreamOperatorContext<'_>,
    ) -> Result<Option<W::Output>> {
        let scope = context
            .gather_client(GatherOperatorId::new(Arc::from(self.name.as_str())))
            .scope()?;
        let credit = self
            .probe_control
            .as_ref()
            .expect("prepaid probe controls")
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
        let result = match ticket {
            Ok(ticket) => finish_probe_ticket(ticket, context).await.map(Some),
            Err(AdmissionFailure::Budget { .. }) => Ok(None),
            Err(AdmissionFailure::Runtime(error)) => Err(error),
        };
        context.check_cancelled()?;
        self.await_probe_attempt(context).await?;
        result
    }

    async fn await_probe_attempt(&mut self, context: &StreamOperatorContext<'_>) -> Result<()> {
        if let Some(cleanup) = &self.compaction_cleanup {
            cleanup.wait(&GatherStop::from_job(context.job())).await?;
        }
        self.compaction_cleanup = None;
        context.check_cancelled()
    }

    async fn recover_probe(
        &mut self,
        inputs: Arc<ProbeInputs>,
        context: &StreamOperatorContext<'_>,
    ) -> Result<(Vec<AdmittedRow>, NativeKeys)> {
        context.check_cancelled()?;
        self.await_probe_attempt(context).await?;
        let mut inputs =
            Arc::try_unwrap(inputs).unwrap_or_else(|_| panic!("settled native probe inputs"));
        let admitted = std::mem::take(&mut inputs.admitted);
        let keys = inputs.keys.take().expect("owned native keys");
        drop(inputs);
        self.await_compaction_release(context).await?;
        Ok((admitted, keys))
    }
}

async fn finish_probe_ticket<T: Send + 'static>(
    ticket: ObservedTicket<Vec<T>>,
    context: &StreamOperatorContext<'_>,
) -> Result<T> {
    let output = tokio::select! {
        biased;
        () = probe_stop(context.job()) => {
            context.check_cancelled()?;
            unreachable!("probe stop requires cancellation or deadline");
        }
        output = ticket.finish() => output,
    };
    context.check_cancelled()?;
    let mut value = None;
    output?.install(|mut values| {
        debug_assert_eq!(values.len(), 1);
        value = values.pop();
        Ok(())
    })?;
    Ok(value.expect("one native probe unit"))
}

async fn probe_stop(job: &StreamJobContext) {
    let deadline = async {
        match job.deadline() {
            Some(deadline) => {
                let delay = (*deadline - chrono::Utc::now())
                    .to_std()
                    .unwrap_or_default();
                tokio::time::sleep(delay).await;
            }
            None => std::future::pending().await,
        }
    };
    tokio::select! {
        () = job.cancellation().cancelled() => {},
        () = deadline => {},
    }
}
