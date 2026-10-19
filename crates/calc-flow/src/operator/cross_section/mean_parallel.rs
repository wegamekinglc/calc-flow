#[cfg(test)]
pub(super) type TestHook = Arc<dyn Fn(usize, bool) + Send + Sync>;

use super::{
    Array, ArrayRef, CompiledEvaluation, CrossSectionOperator, Float64Array, Float64Builder,
    RecordBatch, Result, SchemaRef, StatisticAccumulator, StreamOperatorContext, StringArray,
    TimestampMicrosecondArray, UInt64Array, internal_error, operator_error,
};
use crate::operator::StreamRuntimeState;
use crate::runtime::streaming::gather_work::{
    AdmissionFailure, AttemptCleanup, GatherOperatorId, GatherStop, ParallelCpuWork,
};
use datafusion::execution::memory_pool::MemoryReservation;
use std::{ops::Deref, ops::Range, sync::Arc};

mod funding;

const MIN_ROWS: usize = 262_144;
const ROWS_PER_UNIT: usize = 65_536;
const QUANTUM_ROWS: usize = 4_096;

pub(super) struct MeanParallelState {
    // Operator-local optional-work credit; StateBudget remains a logical limit.
    runtime: StreamRuntimeState,
    // Reset/restore replace pending state without releasing running work early.
    cleanup: Option<AttemptCleanup>,
}

impl MeanParallelState {
    pub(super) const fn new() -> Self {
        Self {
            runtime: StreamRuntimeState::new(),
            cleanup: None,
        }
    }
}

#[derive(Clone)]
pub(super) struct PendingMean {
    record: RecordBatch,
    boundaries: Option<Arc<Boundaries>>,
}

impl Deref for PendingMean {
    type Target = RecordBatch;

    fn deref(&self) -> &Self::Target {
        &self.record
    }
}

struct Boundaries {
    offsets: Vec<usize>,
    _credit: MemoryReservation,
}

pub(super) struct PreparedMean {
    pub(super) record: RecordBatch,
    pub(super) credit: Arc<MemoryReservation>,
}

struct GroupWork {
    values: ArrayRef,
    boundaries: Arc<Boundaries>,
    ranges: Vec<Range<usize>>,
    min_samples: u64,
    credit: Arc<MemoryReservation>,
    #[cfg(test)]
    hook: Option<TestHook>,
}

// Data drops before the shared lease, including results abandoned by the owner.
struct GroupMeans {
    array: Float64Array,
    _credit: Arc<MemoryReservation>,
}

impl CrossSectionOperator {
    pub(super) async fn await_mean_attempt(
        &mut self,
        context: &StreamOperatorContext<'_>,
    ) -> Result<()> {
        if let Some(cleanup) = &self.mean_parallel.cleanup {
            cleanup
                .wait_job(&GatherStop::from_job(context.job()), context.job())
                .await?;
        }
        self.mean_parallel.cleanup = None;
        context.check_cancelled()
    }

    pub(super) async fn prove_pending_mean(
        &mut self,
        record: RecordBatch,
        context: &StreamOperatorContext<'_>,
    ) -> Result<Option<PendingMean>> {
        let Some((times, entities, sequences)) = ordered_columns(&record, self) else {
            return Ok(None);
        };
        if context
            .input_watermark()
            .is_some_and(|watermark| times.value(0) <= watermark.as_micros())
        {
            return Ok(None);
        }
        let eligible =
            record.num_rows() >= MIN_ROWS && context.job().gather_owner().parallelism() > 1;
        let mut attempted = false;
        let mut boundaries = None;
        let entity = |row| entities.is_valid(row).then(|| entities.value(row));
        for index in 1..record.num_rows() {
            owner_quantum(index, context).await?;
            let previous = (
                times.value(index - 1),
                entity(index - 1),
                sequences.value(index - 1),
            );
            let current = (times.value(index), entity(index), sequences.value(index));
            if previous >= current {
                return Ok(None);
            }
            if previous.0 != current.0 && eligible {
                if !attempted {
                    attempted = true;
                    boundaries = self.reserve_mean_boundaries(record.num_rows())?;
                }
                if let Some(boundaries) = &mut boundaries {
                    boundaries.offsets.push(index);
                }
            }
        }
        if let Some(boundaries) = &mut boundaries {
            boundaries.offsets.push(record.num_rows());
        }
        context.check_cancelled()?;
        Ok(Some(PendingMean {
            record,
            boundaries: boundaries.map(Arc::new),
        }))
    }

    pub(super) async fn parallel_mean_record(
        &mut self,
        pending: &PendingMean,
        context: &StreamOperatorContext<'_>,
    ) -> Result<Option<PreparedMean>> {
        let Some(boundaries) = &pending.boundaries else {
            return Ok(None);
        };
        let units = context
            .job()
            .gather_owner()
            .parallelism()
            .min(pending.num_rows() / ROWS_PER_UNIT)
            .min(boundaries.offsets.len() - 1);
        if units < 2 {
            return Ok(None);
        }
        let output = &self.compiled.outputs[0];
        let CompiledEvaluation::Mean { min_samples } = output.evaluation else {
            return Err(internal_error("parallel mean received a different output"));
        };
        let values = pending.column(output.input_index);
        let Some(bytes) = funding::work_bytes(pending, values, units) else {
            return Ok(None);
        };
        let Some(credit) = self.mean_credit("cross-section-mean-work", bytes)? else {
            return Ok(None);
        };
        let ranges = group_ranges(&boundaries.offsets, units);
        if ranges.len() < 2 {
            return Ok(None);
        }
        let work = Arc::new(GroupWork {
            values: values.clone(),
            boundaries: boundaries.clone(),
            ranges,
            min_samples,
            credit: Arc::new(credit),
            #[cfg(test)]
            hook: self.mean_test_hook.clone(),
        });
        let Some(results) = self.run_mean_work(work.clone(), context).await? else {
            return Ok(None);
        };
        let schema = self.output_ports[0]
            .schema()
            .expect("cross-section output always has an exact schema");
        let prepared = assemble(pending, &work, results, schema, context).await?;
        let Some(bytes) = funding::chunk_bytes(&prepared.record, context).await? else {
            return Ok(None);
        };
        if prepared.credit.try_grow(bytes).is_err() {
            return Ok(None);
        }
        Ok(Some(prepared))
    }

    async fn run_mean_work(
        &mut self,
        work: Arc<GroupWork>,
        context: &StreamOperatorContext<'_>,
    ) -> Result<Option<Vec<GroupMeans>>> {
        let scope = context
            .gather_client(GatherOperatorId::new(Arc::from(self.name.as_str())))
            .scope()?;
        let credit = work.credit.new_empty();
        let retirement = context.job().gather_owner().retain_retirement()?;
        let ticket = scope
            .submit_observed_parallel_work(
                work,
                credit,
                GatherStop::from_job(context.job()),
                retirement,
                &mut self.mean_parallel.cleanup,
            )
            .await;
        let ticket = match ticket {
            Ok(ticket) => ticket,
            Err(AdmissionFailure::Budget { .. }) => {
                self.await_mean_attempt(context).await?;
                return Ok(None);
            }
            Err(AdmissionFailure::Runtime(error)) => return Err(error),
        };
        let output = tokio::select! {
            biased;
            () = stopped(context.job()) => {
                context.check_cancelled()?;
                unreachable!("mean stop requires cancellation or deadline");
            }
            output = ticket.finish() => output?,
        };
        context.check_cancelled()?;
        let mut results = None;
        output.install(|values| {
            results = Some(values);
            Ok(())
        })?;
        Ok(results)
    }
}

fn ordered_columns<'a>(
    record: &'a RecordBatch,
    operator: &CrossSectionOperator,
) -> Option<(
    &'a TimestampMicrosecondArray,
    &'a StringArray,
    &'a UInt64Array,
)> {
    let times = record
        .column(operator.compiled.event_time_index)
        .as_any()
        .downcast_ref::<TimestampMicrosecondArray>()?;
    let entities = record
        .column(operator.compiled.entity_columns[0].index)
        .as_any()
        .downcast_ref::<StringArray>()?;
    let sequences = record
        .column(operator.compiled.sequence_columns[0].index)
        .as_any()
        .downcast_ref::<UInt64Array>()?;
    (times.null_count() == 0 && sequences.null_count() == 0).then_some((times, entities, sequences))
}

fn group_ranges(offsets: &[usize], units: usize) -> Vec<Range<usize>> {
    let rows = *offsets.last().expect("nonempty mean boundaries");
    let groups = offsets.len() - 1;
    let mut ranges = Vec::with_capacity(units);
    let mut start = 0;
    for unit in 1..units {
        // The remainder adds at most one row per preceding partition.
        let row = rows / units * unit + (rows % units).min(unit);
        let end = offsets.partition_point(|&offset| offset < row).min(groups);
        if start < end {
            ranges.push(start..end);
            start = end;
        }
    }
    if start < groups {
        ranges.push(start..groups);
    }
    ranges
}

impl ParallelCpuWork for GroupWork {
    type Output = GroupMeans;

    fn unit_count(&self) -> usize {
        self.ranges.len()
    }

    fn run(&self, ordinal: usize, stop: &GatherStop) -> Result<GroupMeans> {
        stop.check()?;
        #[cfg(test)]
        if let Some(hook) = &self.hook {
            hook(ordinal, false);
        }
        let values = self
            .values
            .as_any()
            .downcast_ref::<Float64Array>()
            .expect("columnar mean has validated Float64 values");
        let range = &self.ranges[ordinal];
        let mut means = Float64Builder::with_capacity(range.len());
        for group in range.clone() {
            let mut accumulator = StatisticAccumulator::default();
            for row in self.boundaries.offsets[group]..self.boundaries.offsets[group + 1] {
                if row % QUANTUM_ROWS == 0 {
                    stop.check_abandoned()?;
                }
                if values.is_valid(row) && !values.value(row).is_nan() {
                    accumulator.add(values.value(row));
                }
            }
            means.append_option(
                (accumulator.count >= self.min_samples).then(|| accumulator.classified_mean()),
            );
        }
        stop.check()?;
        #[cfg(test)]
        if let Some(hook) = &self.hook {
            hook(ordinal, true);
        }
        Ok(GroupMeans {
            array: means.finish(),
            _credit: self.credit.clone(),
        })
    }
}

async fn assemble(
    pending: &PendingMean,
    work: &GroupWork,
    results: Vec<GroupMeans>,
    schema: &SchemaRef,
    context: &StreamOperatorContext<'_>,
) -> Result<PreparedMean> {
    let mut means = Float64Builder::with_capacity(pending.num_rows());
    let mut row = 0;
    for (result, range) in results.iter().zip(&work.ranges) {
        for (local, group) in range.clone().enumerate() {
            let mean = result
                .array
                .is_valid(local)
                .then(|| result.array.value(local));
            while row < work.boundaries.offsets[group + 1] {
                owner_quantum(row, context).await?;
                means.append_option(mean);
                row += 1;
            }
        }
    }
    if row != pending.num_rows() {
        return Err(internal_error(
            "parallel mean did not cover every input row",
        ));
    }
    let columns = pending
        .columns()
        .iter()
        .cloned()
        .chain(std::iter::once(Arc::new(means.finish()) as ArrayRef))
        .collect();
    let record = RecordBatch::try_new(schema.clone(), columns)
        .map_err(|error| operator_error(context.operator_id(), &error.to_string()))?;
    context.check_cancelled()?;
    Ok(PreparedMean {
        record,
        credit: work.credit.clone(),
    })
}

async fn owner_quantum(row: usize, context: &StreamOperatorContext<'_>) -> Result<()> {
    if row.is_multiple_of(QUANTUM_ROWS) {
        context.check_cancelled()?;
        tokio::task::yield_now().await;
        context.check_cancelled()?;
    }
    Ok(())
}

async fn stopped(job: &crate::StreamJobContext) {
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
