use super::super::{
    MatchedPair, canonical_column, columnar::RowPayload, materialization::JoinOutput,
    operator_error,
};
use crate::runtime::streaming::gather_work::{
    AttemptCleanup, GatherOperatorId, GatherStop, ParallelCpuWork,
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
        let (side, index) = if ordinal < left_width {
            (&self.left, ordinal)
        } else {
            (&self.right, ordinal - left_width)
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

fn records<'a>(output: &'a JoinOutput<'_>, pair: &MatchedPair) -> (&'a RowPayload, &'a RowPayload) {
    let incoming = &output.admitted[pair.pos].record;
    let opposite = &output.opposite[pair.opposite_index].record;
    if output.incoming_is_left {
        (incoming, opposite)
    } else {
        (opposite, incoming)
    }
}

async fn capture(
    output: &JoinOutput<'_>,
    range: Range<usize>,
    runtime: &DataFusionRuntime,
    context: &StreamOperatorContext<'_>,
) -> Result<Option<Arc<Columns>>> {
    if !profitable(range.len(), output.schema.fields().len())
        || !output
            .schema
            .fields()
            .iter()
            .all(|field| super::independently_compact(field.data_type()))
        || output.admitted_charges.is_none()
    {
        return Ok(None);
    }
    let matched = &output.matched[range];
    if matched.is_empty()
        || !matched.iter().all(|pair| {
            let (left, right) = records(output, pair);
            left.shared_columns().is_some() && right.shared_columns().is_some()
        })
    {
        return Ok(None);
    }
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
        attempt.wait(&stop).await?;
    }
    *cleanup = None;
    context.check_cancelled()?;
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

async fn cancelled(job: &StreamJobContext) {
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
