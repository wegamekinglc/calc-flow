use super::super::{
    MatchedPair, PayloadRef, canonical_column, columnar::RowPayload, materialization::JoinOutput,
    operator_error,
};
use crate::runtime::streaming::gather_work::{
    AttemptCleanup, GatherOperatorId, GatherStop, ObservedTicket, ParallelCpuWork,
};
use crate::{DataFusionRuntime, Result, StreamJobContext, StreamOperatorContext};
use datafusion::arrow::{array::ArrayRef, datatypes::SchemaRef, record_batch::RecordBatch};
use datafusion::execution::memory_pool::MemoryReservation;
use std::{ops::Range, sync::Arc};

mod funding;
mod side;

use side::OwnedSide;

const MIN_ROWS: usize = 64_000;
const MIN_COLUMNS: usize = 6;

struct Columns {
    left: OwnedSide,
    right: OwnedSide,
    schema: SchemaRef,
    projection: Option<Arc<[usize]>>,
    name: Arc<str>,
    #[cfg(test)]
    hook: Option<super::test_support::Hook>,
    input_credit: MemoryReservation,
    output_credit: Arc<MemoryReservation>,
}

pub(in crate::operator::join) struct ColumnResult {
    array: ArrayRef,
    _credit: Arc<MemoryReservation>,
}

impl ParallelCpuWork for Columns {
    type Output = ColumnResult;

    fn unit_count(&self) -> usize {
        self.schema.fields().len()
    }

    fn run(&self, ordinal: usize, stop: &GatherStop) -> Result<ColumnResult> {
        stop.check()?;
        #[cfg(test)]
        let _observation = super::test_support::observe(self.hook.clone(), ordinal);
        let left_width = self.left.width();
        let index = self
            .projection
            .as_ref()
            .map_or(ordinal, |columns| columns[ordinal]);
        let (side, index) = if index < left_width {
            (&self.left, index)
        } else {
            (&self.right, index - left_width)
        };
        let array = side.column(index, &self.input_credit, &self.name)?;
        stop.check()?;
        Ok(ColumnResult {
            array: canonical_column(array, self.schema.field(ordinal).data_type()),
            _credit: Arc::clone(&self.output_credit),
        })
    }
}

fn profitable(rows: usize, columns: usize) -> bool {
    #[cfg(test)]
    if super::test_support::force() {
        return true;
    }
    rows >= MIN_ROWS && columns >= MIN_COLUMNS
}

fn records<'a>(output: &'a JoinOutput<'_>, pair: &MatchedPair) -> (PayloadRef<'a>, PayloadRef<'a>) {
    let incoming = output.admitted.row(pair.pos).record;
    let opposite = (&output.opposite[pair.opposite_index].record).into();
    if output.incoming_is_left {
        (incoming, opposite)
    } else {
        (opposite, incoming)
    }
}

fn supports_capture(output: &JoinOutput<'_>, range: &Range<usize>) -> bool {
    if output.schema.fields().is_empty()
        || !profitable(range.len(), output.schema.fields().len())
        || !output
            .schema
            .fields()
            .iter()
            .all(|field| super::independently_compact(field.data_type()))
        || output.admitted_charges.is_none()
    {
        return false;
    }
    let matched = &output.matched[range.clone()];
    if matched.is_empty() || !matched.iter().all(|pair| detached_pair(output, pair)) {
        return false;
    }
    true
}

fn detached_pair(output: &JoinOutput<'_>, pair: &MatchedPair) -> bool {
    let (left, right) = records(output, pair);
    left.shared_columns().is_some()
        && right.shared_columns().is_some()
        && left.owner().is_some()
        && right.owner().is_some()
}

async fn capture(
    output: &JoinOutput<'_>,
    range: Range<usize>,
    runtime: &DataFusionRuntime,
    context: &StreamOperatorContext<'_>,
) -> Result<Option<Arc<Columns>>> {
    if !supports_capture(output, &range) {
        return Ok(None);
    }
    let matched = &output.matched[range];
    let input_credit = runtime.incremental_reservation("stream-join-gather");
    let output_credit = input_credit.new_empty();
    let input_bytes = funding::input_bytes(matched.len(), output.schema, output.operator_id)?;
    let output_bytes = funding::output_bytes(output, matched)?;
    if input_credit.try_grow(input_bytes).is_err() || output_credit.try_grow(output_bytes).is_err()
    {
        return Ok(None);
    }
    let left =
        OwnedSide::capture(matched.iter().map(|pair| records(output, pair).0), context).await?;
    let right =
        OwnedSide::capture(matched.iter().map(|pair| records(output, pair).1), context).await?;
    Ok(Some(Arc::new(Columns {
        left,
        right,
        schema: Arc::clone(output.schema),
        projection: output.projection.cloned(),
        name: Arc::from(output.operator_id),
        #[cfg(test)]
        hook: super::test_support::hook(),
        input_credit,
        output_credit: Arc::new(output_credit),
    })))
}

pub(in crate::operator::join) async fn materialize(
    output: &JoinOutput<'_>,
    range: Range<usize>,
    runtime: &DataFusionRuntime,
    context: &StreamOperatorContext<'_>,
    cleanup: &mut Option<AttemptCleanup>,
) -> Result<Option<(RecordBatch, Vec<ColumnResult>)>> {
    let Some(work) = capture(output, range, runtime, context).await? else {
        return Ok(None);
    };
    let credit = work.input_credit.new_empty();
    let scope = context
        .gather_client(GatherOperatorId::new(Arc::from(output.operator_id)))
        .scope()?;
    let stop = GatherStop::from_job(context.job());
    let retirement = context.job().gather_owner().retain_retirement()?;
    let ticket = scope
        .submit_observed_parallel_work(work, credit, stop.clone(), retirement, cleanup)
        .await?;
    let columns = finish_gather(ticket, &stop, context, cleanup).await?;
    let record = RecordBatch::try_new(
        Arc::clone(output.schema),
        columns
            .iter()
            .map(|column| Arc::clone(&column.array))
            .collect(),
    )
    .map_err(|error| {
        operator_error(
            output.operator_id,
            &format!("output projection failed: {error}"),
        )
    })?;
    Ok(Some((record, columns)))
}

async fn finish_gather(
    ticket: ObservedTicket<Vec<ColumnResult>>,
    stop: &GatherStop,
    context: &StreamOperatorContext<'_>,
    cleanup: &mut Option<AttemptCleanup>,
) -> Result<Vec<ColumnResult>> {
    let gathered = tokio::select! {
        biased;
        () = cancelled(context.job()) => {
            context.check_cancelled()?;
            unreachable!("gather stop requires cancellation or deadline");
        }
        gathered = ticket.finish() => gathered?,
    };
    context.check_cancelled()?;
    let mut columns = Vec::new();
    gathered.install(|values| {
        columns = values;
        Ok(())
    })?;
    if let Some(attempt) = cleanup.as_ref() {
        attempt.wait(stop).await?;
    }
    *cleanup = None;
    context.check_cancelled()?;
    Ok(columns)
}

pub(in crate::operator::join) async fn cancelled(job: &StreamJobContext) {
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
