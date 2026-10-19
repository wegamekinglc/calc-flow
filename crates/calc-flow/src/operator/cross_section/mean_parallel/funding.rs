use super::{
    ArrayRef, Boundaries, CrossSectionOperator, GroupMeans, GroupWork, MemoryReservation,
    PendingMean, PreparedMean, Result,
};
use crate::operator::row_cost::{RowCosts, fixed_width};
use crate::runtime::streaming::gather_work::cleanup_control_bytes;

impl CrossSectionOperator {
    pub(super) fn reserve_mean_boundaries(&mut self, rows: usize) -> Result<Option<Boundaries>> {
        let Some(capacity) = rows.checked_add(1) else {
            return Ok(None);
        };
        let Some(bytes) = capacity
            .checked_mul(size_of::<usize>())
            .and_then(|bytes| bytes.checked_add(size_of::<Boundaries>() + 128))
        else {
            return Ok(None);
        };
        let Some(credit) = self.mean_credit("cross-section-mean-boundaries", bytes)? else {
            return Ok(None);
        };
        let mut offsets = Vec::with_capacity(capacity);
        offsets.push(0);
        Ok(Some(Boundaries {
            offsets,
            _credit: credit,
        }))
    }

    pub(super) fn mean_credit(
        &mut self,
        consumer: &str,
        bytes: usize,
    ) -> Result<Option<MemoryReservation>> {
        #[cfg(test)]
        if let Some(pool) = self.mean_test_pool.take() {
            self.mean_parallel.runtime.initialize_benchmark_pool(pool)?;
        }
        let credit = self
            .mean_parallel
            .runtime
            .runtime()?
            .incremental_reservation(&format!("{}:{consumer}", self.name));
        Ok(credit.try_grow(bytes).ok().map(|()| credit))
    }
}

pub(super) fn work_bytes(pending: &PendingMean, values: &ArrayRef, units: usize) -> Option<usize> {
    if pending
        .columns()
        .iter()
        .any(|column| fixed_width(column.data_type()).is_none())
    {
        return None;
    }
    let groups = pending.boundaries.as_ref()?.offsets.len() - 1;
    // Pay full shared slice backing, both builder capacities, array/Arc controls,
    // ordinal collection and cleanup before any background ownership/allocation.
    let source = values.get_array_memory_size().checked_add(4096)?;
    let results = groups
        .checked_mul(18)?
        .checked_add(units.checked_mul(4096 + size_of::<GroupMeans>() * 2)?)?;
    let final_output = pending
        .num_rows()
        .checked_mul(18)?
        .checked_add(pending.num_columns().checked_add(1)?.checked_mul(4096)?)?;
    let controls = units
        .checked_mul(size_of::<std::ops::Range<usize>>())?
        .checked_add(size_of::<GroupWork>() + size_of::<PreparedMean>())?
        .checked_add(cleanup_control_bytes::<Vec<GroupMeans>>())?
        .checked_add(4096)?;
    source
        .checked_add(results)?
        .checked_add(final_output)?
        .checked_add(controls)
}

pub(super) async fn chunk_bytes(
    record: &super::RecordBatch,
    context: &super::StreamOperatorContext<'_>,
) -> Result<Option<usize>> {
    let mut backing = 0usize;
    for (ordinal, column) in record.columns().iter().enumerate() {
        super::owner_quantum(ordinal, context).await?;
        let Some(total) = backing.checked_add(column.get_array_memory_size()) else {
            return Ok(None);
        };
        backing = total;
    }
    let budget = context.output_budget();
    let rows = record.num_rows();
    let row_charge = RowCosts::try_total(record)?;
    let upper = rows
        .div_ceil(8)
        .checked_mul(record.num_columns())
        .and_then(|validity| row_charge?.checked_add(validity));
    let whole = rows <= budget.max_rows && backing <= budget.max_bytes;
    let chunks = if whole {
        1
    } else if upper.is_some_and(|bound| bound <= budget.max_bytes) {
        // Flat totals use lengths, offsets and stored null counts, without a
        // row scan. Full-record validity also bounds every candidate range.
        rows.div_ceil(budget.max_rows)
    } else {
        // A reservation upper bound, not RSS: a byte-limited range may be one row.
        rows
    };
    let Some(headers) = record
        .num_columns()
        .checked_mul(4096)
        .and_then(|bytes| bytes.checked_add(context.operator_id().len()))
        .and_then(|bytes| bytes.checked_add(2048))
        .and_then(|bytes| bytes.checked_mul(2))
        .and_then(|bytes| bytes.checked_mul(chunks))
    else {
        return Ok(None);
    };
    // RowCosts, geometric range growth, validity inventories and Batch slots
    // can overlap while the original Float64 builder and group results live.
    Ok(if whole { 0 } else { rows }
        .checked_mul(size_of::<usize>())
        .and_then(|scratch| scratch.checked_add(headers)))
}
