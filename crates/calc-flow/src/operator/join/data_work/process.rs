use super::super::{
    AdmittedBatch, MatchedPair, PreparedMatches, SidePlan, StreamJoinOperator, enforce_match_limit,
    native_lookup::{NativeKeys, eligible, scratch_error},
};
use super::{
    MIN_PROBE_ROWS, ProbeAttempt, control,
    inputs::ProbeInputs,
    pairs::{self, FillFunding, PairOutput},
    partition::Counts,
    worker::{CountWork, FillWork},
};
use crate::runtime::streaming::gather_work::{
    AdmissionFailure, GatherOperatorId, GatherStop, JobGatherOwner, ObservedTicket, ParallelCpuWork,
};
use crate::{Result, StreamJobContext, StreamOperatorContext};
use parking_lot::Mutex;
use std::sync::Arc;

#[cfg(test)]
#[path = "../tests/probe_stop_tests.rs"]
mod probe_stop_tests;

#[cfg(test)]
#[path = "../tests/native_gather_window_tests.rs"]
mod native_gather_window_tests;

#[cfg(test)]
#[path = "../tests/batch_parallel_output_tests.rs"]
mod batch_parallel_output_tests;

struct Capture {
    declined: Option<(AdmittedBatch, Option<NativeKeys>)>,
    ready: Option<Arc<ProbeInputs>>,
}

enum ProbeOutput {
    Pairs(PairOutput),
    Counted(
        Counts,
        datafusion::execution::memory_pool::MemoryReservation,
    ),
}
impl Capture {
    fn declined(admitted: AdmittedBatch, keys: Option<NativeKeys>) -> Self {
        Self {
            declined: Some((admitted, keys)),
            ready: None,
        }
    }
    fn ready(inputs: Arc<ProbeInputs>) -> Self {
        Self {
            declined: None,
            ready: Some(inputs),
        }
    }
}

impl StreamJoinOperator {
    pub(in crate::operator::join) async fn owned_native_matches(
        &mut self,
        plan: &SidePlan,
        admitted: &mut AdmittedBatch,
        keys: Option<NativeKeys>,
        context: &StreamOperatorContext<'_>,
    ) -> ProbeAttempt {
        let inputs = {
            let captured = match self.capture_probe(
                plan,
                std::mem::take(admitted),
                keys,
                context.job().gather_owner(),
            ) {
                Ok(captured) => captured,
                Err(error) => return ProbeAttempt::Complete(Err(error)),
            };
            if let Some((recovered, keys)) = captured.declined {
                *admitted = recovered;
                return ProbeAttempt::Serial(keys);
            }
            captured.ready.expect("captured probe inputs")
        };
        let result = match self.dispatch_native_probe(&inputs, context).await {
            Ok(Some(ProbeOutput::Counted(counts, credit))) => {
                return ProbeAttempt::Complete(
                    self.defer_probe(inputs, admitted, counts, credit, context)
                        .await,
                );
            }
            other => other,
        };
        let recovered = self.recover_probe(inputs, admitted, context).await;
        let keys = match recovered {
            Ok(recovered) => recovered,
            Err(error) => return ProbeAttempt::Complete(Err(error)),
        };
        match result {
            Ok(None) => ProbeAttempt::Serial(Some(keys)),
            Ok(Some(ProbeOutput::Pairs(output))) => ProbeAttempt::Complete(Ok(PreparedMatches {
                pairs: output.pairs,
                keys: Some(keys),
                credit: Some(output.credit),
                deferred: None,
            })),
            Ok(Some(ProbeOutput::Counted(..))) => unreachable!("counted probe deferred above"),
            Err(error) => ProbeAttempt::Complete(Err(error)),
        }
    }

    fn capture_probe(
        &mut self,
        plan: &SidePlan,
        admitted: AdmittedBatch,
        keys: Option<NativeKeys>,
        owner: &JobGatherOwner,
    ) -> Result<Capture> {
        if !self.probe_ready(plan, admitted.len())? {
            return Ok(Capture::declined(admitted, keys));
        }
        let Some(keys) = keys else {
            return Ok(Capture::declined(admitted, None));
        };
        self.capture_keyed_probe(plan, admitted, keys, owner)
    }

    fn capture_keyed_probe(
        &mut self,
        plan: &SidePlan,
        mut admitted: AdmittedBatch,
        keys: NativeKeys,
        owner: &JobGatherOwner,
    ) -> Result<Capture> {
        let bytes = control::input_bytes(keys.keys.len(), admitted.len(), &self.name)?;
        let Some(credit) = self.optional_credit(bytes)? else {
            return Ok(Capture::declined(admitted, Some(keys)));
        };
        if !admitted.detach(self)? {
            return Ok(Capture::declined(admitted, Some(keys)));
        }
        let control = Arc::new(credit);
        let opposite = if plan.incoming_is_left {
            &self.state.right
        } else {
            &self.state.left
        };
        let index = opposite.1.as_ref().expect("captured native index");
        let slots = keys
            .keys
            .iter()
            .map(|key| key.as_ref().and_then(|key| index.key_id(key)))
            .collect::<Vec<_>>();
        let estimated_visits = (0..admitted.len())
            .filter_map(|position| slots[keys.id(position) as usize])
            .fold(0usize, |total, id| {
                total.saturating_add(index.run_len_by_id(id))
            });
        if estimated_visits < super::probe_min_visits()
            && !(super::morsel::materialization_worthwhile(self, admitted.len(), estimated_visits)
                && owner.parallelism() > 1)
        {
            return Ok(Capture::declined(admitted, Some(keys)));
        }
        let (released, receiver) = tokio::sync::oneshot::channel();
        self.compaction_release = Some(receiver);
        self.probe_control = Some(Arc::clone(&control));
        Ok(Capture::ready(Arc::new(ProbeInputs {
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
            #[cfg(test)]
            unit_hook: self.probe_unit_test_hook.take(),
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
    ) -> Result<Option<ProbeOutput>> {
        let limit = self.spec.limits.max_matches_per_input_batch;
        let work = Arc::new(CountWork {
            inputs: Arc::clone(inputs),
            limit,
        });
        let Some(counts) = self
            .run_probe_units(
                work,
                context,
                move |values| Counts::ordered(values, limit),
                #[cfg(test)]
                false,
            )
            .await?
        else {
            return Ok(None);
        };
        enforce_match_limit(
            counts.total,
            &mut self.state.metrics.match_limit_failures,
            self.spec.limits.max_matches_per_input_batch,
            &self.name,
        )?;
        if let Some(credit) = self.deferred_credit(counts, inputs.admitted.len())? {
            return Ok(Some(ProbeOutput::Counted(counts, credit)));
        }
        self.fill_probe(inputs, counts, context)
            .await
            .map(|output| output.map(ProbeOutput::Pairs))
    }

    fn deferred_credit(
        &mut self,
        counts: Counts,
        rows: usize,
    ) -> Result<Option<datafusion::execution::memory_pool::MemoryReservation>> {
        if counts.total == 0 || !super::morsel::eligible(self, rows) {
            return Ok(None);
        }
        let pinned = self
            .probe_control
            .as_ref()
            .expect("owned count controls")
            .size();
        let Some(bytes) = super::morsel::pair_credit_bytes(counts.total, pinned) else {
            return Ok(None);
        };
        self.optional_credit(bytes)
    }

    pub(super) async fn fill_probe(
        &mut self,
        inputs: &Arc<ProbeInputs>,
        counts: Counts,
        context: &StreamOperatorContext<'_>,
    ) -> Result<Option<PairOutput>> {
        let Some((funding, credit)) = self.admit_fill(counts)? else {
            return Ok(None);
        };
        self.run_probe_units(
            Arc::new(FillWork {
                inputs: Arc::clone(inputs),
                counts,
                funding,
            }),
            context,
            move |fragments| pairs::ordered_output(fragments, credit, counts.total),
            #[cfg(test)]
            true,
        )
        .await
    }

    fn admit_fill(
        &mut self,
        counts: Counts,
    ) -> Result<
        Option<(
            FillFunding,
            Option<datafusion::execution::memory_pool::MemoryReservation>,
        )>,
    > {
        let bytes = counts
            .total
            .checked_mul(size_of::<MatchedPair>() + 256)
            .ok_or_else(|| scratch_error(&self.name))?;
        let Some(credit) = self.optional_credit(bytes)? else {
            return Ok(None);
        };
        if counts.units == 1 {
            return Ok(Some((FillFunding::Single(Mutex::new(Some(credit))), None)));
        }
        let bytes = counts
            .total
            .checked_mul(3 * size_of::<usize>())
            .ok_or_else(|| scratch_error(&self.name))?;
        let Some(fragments) = self.optional_credit(bytes)? else {
            return Ok(None);
        };
        Ok(Some((
            FillFunding::Fragments(Arc::new(fragments)),
            Some(credit),
        )))
    }

    pub(super) async fn run_probe_units<W: ParallelCpuWork, T>(
        &mut self,
        work: Arc<W>,
        context: &StreamOperatorContext<'_>,
        install: impl FnOnce(Vec<W::Output>) -> Result<T>,
        #[cfg(test)] completed: bool,
    ) -> Result<Option<T>> {
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
        #[cfg(test)]
        let gate = completed
            .then(|| self.probe_result_test_gate.take())
            .flatten();
        let result = match ticket {
            Ok(ticket) => finish_probe_ticket(
                ticket,
                context,
                install,
                #[cfg(test)]
                gate.as_deref(),
            )
            .await
            .map(Some),
            Err(AdmissionFailure::Budget { .. }) => Ok(None),
            Err(AdmissionFailure::Runtime(error)) => Err(error),
        };
        context.check_cancelled()?;
        self.await_probe_attempt(context).await?;
        result
    }

    pub(super) async fn await_probe_attempt(
        &mut self,
        context: &StreamOperatorContext<'_>,
    ) -> Result<()> {
        if let Some(cleanup) = &self.compaction_cleanup {
            cleanup.wait(&GatherStop::from_job(context.job())).await?;
        }
        self.compaction_cleanup = None;
        context.check_cancelled()
    }

    pub(super) async fn recover_probe(
        &mut self,
        inputs: Arc<ProbeInputs>,
        admitted: &mut AdmittedBatch,
        context: &StreamOperatorContext<'_>,
    ) -> Result<NativeKeys> {
        context.check_cancelled()?;
        self.await_probe_attempt(context).await?;
        let mut inputs =
            Arc::try_unwrap(inputs).unwrap_or_else(|_| panic!("settled native probe inputs"));
        *admitted = std::mem::take(&mut inputs.admitted);
        let keys = inputs.keys.take().expect("owned native keys");
        drop(inputs);
        self.await_compaction_release(context).await?;
        Ok(keys)
    }
}

async fn finish_probe_ticket<T: Send + 'static, U>(
    ticket: ObservedTicket<Vec<T>>,
    context: &StreamOperatorContext<'_>,
    install: impl FnOnce(Vec<T>) -> Result<U>,
    #[cfg(test)] gate: Option<&super::ResultTestGate>,
) -> Result<U> {
    // Grouped workers poll abandonment only. Keep this stop waiter active until
    // settlement so cancellation/deadline drops the ticket and sets that flag.
    let output = tokio::select! {
        biased;
        () = probe_stop(context.job()) => {
            context.check_cancelled()?;
            unreachable!("probe stop requires cancellation or deadline");
        }
        output = ticket.finish() => output,
    };
    context.check_cancelled()?;
    let output = output?;
    #[cfg(test)]
    await_result_gate(gate, context).await?;
    let mut value = None;
    output.install(|values| {
        value = Some(install(values)?);
        Ok(())
    })?;
    Ok(value.expect("installed native probe units"))
}

async fn probe_stop(job: &StreamJobContext) {
    let deadline = async {
        #[cfg(test)]
        if let (Some(mut clock), Some(expires)) = (job.deadline_clock_for_test(), job.deadline()) {
            clock
                .wait_for(|now| now >= expires)
                .await
                .expect("deadline clock outlives its wait");
            return;
        }
        match job.deadline() {
            Some(deadline) => {
                while chrono::Utc::now() < *deadline {
                    let delay = (*deadline - chrono::Utc::now())
                        .to_std()
                        .unwrap_or_default();
                    tokio::time::sleep(delay).await;
                }
            }
            None => std::future::pending().await,
        }
    };
    tokio::select! {
        () = job.cancellation().cancelled() => {},
        () = deadline => {},
    }
}

#[cfg(test)]
async fn await_result_gate(
    gate: Option<&super::ResultTestGate>,
    context: &StreamOperatorContext<'_>,
) -> Result<()> {
    if let Some(gate) = gate {
        gate.entered.notify_one();
        tokio::select! {
            () = gate.release.notified() => {},
            () = probe_stop(context.job()) => {},
        }
    }
    context.check_cancelled()
}
