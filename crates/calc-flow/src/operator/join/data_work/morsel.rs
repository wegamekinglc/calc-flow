//! Count first; defer owned probe/gather until mandatory state admission succeeds.

use std::{collections::BTreeMap, sync::Arc};

use datafusion::{
    arrow::{datatypes::SchemaRef, record_batch::RecordBatch},
    execution::memory_pool::MemoryReservation,
};

use super::super::{
    AdmittedBatch, PreparedJoinBatch, PreparedMatches, StreamJoinOperator, matches::Selection,
    materialization::JoinOutput, native_lookup::NativeKeys, operator_error, output_gather,
};
use super::{inputs::ProbeInputs, partition::Counts, worker};
use crate::runtime::streaming::gather_work::{GatherStop, ParallelCpuWork};
use crate::{Batch, BatchMetadata, EdgeBudget, Result, StreamCollector, StreamOperatorContext};

mod fallback;
mod funding;
pub(super) use funding::pair_credit_bytes;

pub(in crate::operator::join) struct DeferredProbe {
    inputs: ProbeInputs,
    counts: Counts,
    fallback: MemoryReservation,
}

impl DeferredProbe {
    pub(in crate::operator::join) fn attach_keys(&mut self, keys: Option<NativeKeys>) {
        self.inputs.keys = keys;
    }
}

pub(super) fn eligible(operator: &StreamJoinOperator, rows: usize) -> bool {
    #[cfg(test)]
    if DISABLED.get() {
        return false;
    }
    rows >= 64_000
        && operator.output_ports[0].schema().is_some_and(|schema| {
            !schema.fields().is_empty()
                && schema
                    .fields()
                    .iter()
                    .all(|field| output_gather::flat_owned_type(field.data_type()))
        })
}

#[cfg(test)]
thread_local! {
    pub(super) static DISABLED: std::cell::Cell<bool> = const { std::cell::Cell::new(false) };
}

#[cfg(test)]
pub(in crate::operator::join) struct RestoreMode(bool);

#[cfg(test)]
impl Drop for RestoreMode {
    fn drop(&mut self) {
        DISABLED.set(self.0);
    }
}

#[cfg(test)]
pub(in crate::operator::join) fn configure_for_test(disabled: bool) -> RestoreMode {
    RestoreMode(DISABLED.replace(disabled))
}

struct GatherWork {
    inputs: Arc<ProbeInputs>,
    counts: Counts,
    schema: SchemaRef,
    projection: Option<Arc<[usize]>>,
    charges: Vec<u64>,
    budget: EdgeBudget,
    name: String,
    scratch: MemoryReservation,
}

pub(in crate::operator::join) struct GatheredOutput(Vec<FundedMorsel>);

struct FundedMorsel {
    // Array backing is destroyed before its funding.
    records: Vec<RecordBatch>,
    _credit: MemoryReservation,
}

impl ParallelCpuWork for GatherWork {
    type Output = Option<FundedMorsel>;

    fn unit_count(&self) -> usize {
        self.counts.units
    }

    fn run(&self, ordinal: usize, stop: &GatherStop) -> Result<Self::Output> {
        #[cfg(test)]
        self.inputs.observe(super::ProbePhase::Fill);
        #[cfg(test)]
        self.inputs
            .observe_unit(super::ProbePhase::Fill, ordinal, false);
        let pairs = worker::fill_range(&self.inputs, self.counts, ordinal, stop)?;
        let result = self.gather(&pairs, stop)?;
        stop.check()?;
        #[cfg(test)]
        self.inputs
            .observe_unit(super::ProbePhase::Fill, ordinal, true);
        Ok(result)
    }
}

impl GatherWork {
    fn gather(&self, pairs: &Selection, stop: &GatherStop) -> Result<Option<FundedMorsel>> {
        let output = JoinOutput {
            schema: &self.schema,
            projection: self.projection.as_ref(),
            admitted: &self.inputs.admitted,
            opposite: self.inputs.opposite.as_deref().expect("owned morsel state"),
            matched: pairs,
            incoming_is_left: self.inputs.incoming_is_left,
            operator_id: &self.name,
            admitted_charges: Some(&self.charges),
        };
        if !funding::owned_payloads(&output, stop)? {
            return Ok(None);
        }
        let ranges = output.ranges(self.budget)?;
        stop.check()?;
        let Some(credit) = self.admit_output(&output, ranges.len())? else {
            return Ok(None);
        };
        let records = self.collect_records(&output, ranges, stop)?;
        Ok(Some(FundedMorsel {
            records,
            _credit: credit,
        }))
    }

    fn admit_output(
        &self,
        output: &JoinOutput<'_>,
        records: usize,
    ) -> Result<Option<MemoryReservation>> {
        let Some(bytes) = funding::output_bytes(output, records)? else {
            return Ok(None);
        };
        let credit = self.scratch.new_empty();
        Ok(credit.try_grow(bytes).is_ok().then_some(credit))
    }

    fn collect_records(
        &self,
        output: &JoinOutput<'_>,
        ranges: Vec<std::ops::Range<usize>>,
        stop: &GatherStop,
    ) -> Result<Vec<RecordBatch>> {
        let mut records = Vec::with_capacity(ranges.len());
        for range in ranges {
            stop.check_abandoned()?;
            let record = output.materialize(range)?;
            // Validate every physical chunk before any worker output escapes.
            let message = Batch::table(vec![record.clone()], BatchMetadata::default())?;
            if message.estimated_bytes()? > self.budget.max_bytes {
                return Err(operator_error(
                    &self.name,
                    "validated Join chunk exceeded its byte budget",
                ));
            }
            records.push(record);
        }
        Ok(records)
    }
}

impl StreamJoinOperator {
    pub(super) async fn defer_probe(
        &mut self,
        inputs: Arc<ProbeInputs>,
        admitted: &mut AdmittedBatch,
        counts: Counts,
        fallback: MemoryReservation,
        context: &StreamOperatorContext<'_>,
    ) -> Result<PreparedMatches> {
        self.await_probe_attempt(context).await?;
        let mut inputs = Arc::try_unwrap(inputs).unwrap_or_else(|_| panic!("settled count inputs"));
        *admitted = std::mem::take(&mut inputs.admitted);
        let keys = inputs.keys.take();
        if let Some(released) = inputs.released.take() {
            let _ = released.send(());
        }
        self.await_compaction_release(context).await?;
        Ok(PreparedMatches {
            pairs: Vec::new().into(),
            keys,
            credit: None,
            deferred: Some(Box::new(DeferredProbe {
                inputs,
                counts,
                fallback,
            })),
        })
    }

    pub(in crate::operator::join) async fn finish_morsels(
        &mut self,
        prepared: &mut PreparedJoinBatch,
        context: &StreamOperatorContext<'_>,
    ) -> Result<()> {
        let Some(deferred) = prepared.deferred.take() else {
            return Ok(());
        };
        self.await_compaction_release(context).await?;
        let DeferredProbe {
            mut inputs,
            counts,
            fallback,
        } = *deferred;
        inputs.admitted = std::mem::take(&mut prepared.admitted);
        let (released, receiver) = tokio::sync::oneshot::channel();
        inputs.released = Some(released);
        self.compaction_release = Some(receiver);
        self.probe_control.clone_from(&inputs.control);
        let inputs = Arc::new(inputs);
        let gathered = self
            .gather_morsels(
                &inputs,
                counts,
                prepared.admitted_charges.as_deref(),
                context,
            )
            .await;
        let result = match gathered {
            Ok(Some(output)) => {
                prepared.gathered = Some(output);
                Ok(())
            }
            Ok(None) => {
                self.fallback_morsels(&inputs, counts, fallback, prepared, context)
                    .await
            }
            Err(error) => Err(error),
        };
        self.recover_probe(inputs, &mut prepared.admitted, context)
            .await?;
        result
    }

    async fn gather_morsels(
        &mut self,
        inputs: &Arc<ProbeInputs>,
        counts: Counts,
        charges: Option<&[u64]>,
        context: &StreamOperatorContext<'_>,
    ) -> Result<Option<GatheredOutput>> {
        let Some(charges) = charges else {
            return Ok(None);
        };
        let schema = Arc::clone(self.output_ports[0].schema().expect("compiled Join output"));
        let Some(bytes) =
            funding::scratch_bytes(counts.total, inputs.admitted.len(), schema.fields().len())
        else {
            return Ok(None);
        };
        let scratch = self
            .runtime
            .runtime()?
            .incremental_reservation("stream-join-morsel");
        if scratch.try_grow(bytes).is_err() {
            return Ok(None);
        }
        let work = Arc::new(GatherWork {
            inputs: Arc::clone(inputs),
            counts,
            schema,
            projection: self.output_columns.clone(),
            charges: charges.to_vec(),
            budget: context.output_budget(),
            name: self.name.clone(),
            scratch,
        });
        self.run_probe_units(
            work,
            context,
            |values| {
                Ok(values
                    .into_iter()
                    .collect::<Option<Vec<_>>>()
                    .map(GatheredOutput))
            },
            #[cfg(test)]
            true,
        )
        .await
        .map(Option::flatten)
    }
}

impl GatheredOutput {
    pub(in crate::operator::join) fn rows(&self) -> usize {
        self.0
            .iter()
            .flat_map(|morsel| &morsel.records)
            .map(RecordBatch::num_rows)
            .sum()
    }

    pub(in crate::operator::join) async fn emit(
        &self,
        name: &str,
        context: &StreamOperatorContext<'_>,
        output: &mut dyn StreamCollector,
        sequence: &mut u64,
    ) -> Result<()> {
        let records = self.0.iter().flat_map(|morsel| &morsel.records);
        let count = records.clone().count();
        crate::operator::output_chunk::validate_output_sequence_range(name, *sequence, count)?;
        for record in records {
            context.check_cancelled()?;
            let metadata = BatchMetadata::new(name, *sequence, BTreeMap::new())?;
            output
                .emit("output", Batch::table(vec![record.clone()], metadata)?)
                .await?;
            *sequence += 1;
        }
        Ok(())
    }
}
