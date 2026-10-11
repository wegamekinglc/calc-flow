use std::sync::Arc;

use datafusion::arrow::{
    array::{Array, ArrayRef},
    compute::concat,
    datatypes::SchemaRef,
    record_batch::RecordBatch,
};
use datafusion::execution::memory_pool::MemoryReservation;

use super::super::super::StoredRow;
use super::super::{
    budget,
    ipc::{
        accounting::{self, add, product, sum},
        concat::{bulk::admit, slice_controls},
    },
    payload::Funding,
};
use super::buffer::error;
use crate::Result;

const SELECTION_RUNS_PER_CHUNK: usize = 1_024;

pub(super) struct CompactBatch {
    pub(super) record: RecordBatch,
    _funding: Arc<Funding>,
}

#[cfg(test)]
impl CompactBatch {
    pub(super) fn funding_owner(&self) -> std::sync::Weak<Funding> {
        let Self {
            _funding: funding, ..
        } = self;
        Arc::downgrade(funding)
    }
}

pub(super) fn compact(
    rows: &[&StoredRow],
    schema: SchemaRef,
    workspace: &MemoryReservation,
    check: &dyn Fn() -> Result<()>,
) -> Result<CompactBatch> {
    let credit = workspace.new_empty();
    accounting::reserve(&credit, budget::payload_seed()?)?;
    let funding = Funding::new([Arc::clone(&schema), Arc::clone(&schema)], credit, None);
    accounting::reserve(
        workspace,
        accounting::vector_peak::<ArrayRef>(schema.fields().len())?,
    )?;
    let columns = (0..schema.fields().len())
        .map(|index| compact_column(rows, index, workspace, &funding, check))
        .collect::<Result<Vec<_>>>()?;
    let record = RecordBatch::try_new(schema, columns)
        .map_err(|_| error("V2 checkpoint compacted schema differs"))?;
    Ok(CompactBatch {
        record,
        _funding: funding,
    })
}

fn compact_column(
    rows: &[&StoredRow],
    index: usize,
    workspace: &MemoryReservation,
    funding: &Arc<Funding>,
    check: &dyn Fn() -> Result<()>,
) -> Result<ArrayRef> {
    let (first, mut consumed) = concatenate_column(rows, index, workspace, funding, check)?;
    if consumed == rows.len() {
        return Ok(first);
    }
    let scratch = workspace.new_empty();
    let count = add(
        1,
        (rows.len() - consumed).div_ceil(SELECTION_RUNS_PER_CHUNK),
    )?;
    admit_selections(count, &scratch)?;
    let mut chunks = Vec::with_capacity(count);
    chunks.push(first);
    while consumed < rows.len() {
        let (chunk, count) =
            concatenate_column(&rows[consumed..], index, workspace, funding, check)?;
        chunks.push(chunk);
        consumed = add(consumed, count)?;
    }
    let arrays = chunks.iter().map(Arc::as_ref).collect::<Vec<_>>();
    concatenate(&arrays, &scratch, funding, check)
}

fn concatenate_column(
    rows: &[&StoredRow],
    index: usize,
    workspace: &MemoryReservation,
    funding: &Arc<Funding>,
    check: &dyn Fn() -> Result<()>,
) -> Result<(ArrayRef, usize)> {
    let scratch = workspace.new_empty();
    let count = admit_selections(rows.len().min(SELECTION_RUNS_PER_CHUNK), &scratch)?;
    let mut selected = Vec::with_capacity(count);
    let mut consumed = 0;
    while consumed < rows.len() && selected.len() < SELECTION_RUNS_PER_CHUNK {
        let (column, count) = selected_column(&rows[consumed..], index, &scratch, check)?;
        selected.push(column);
        consumed = add(consumed, count)?;
    }
    if selected.len() == 1 {
        let first = &selected[0];
        admit_selection(first.as_ref(), &scratch, check)?;
        selected.push(first.slice(0, 0));
    }
    let arrays = selected.iter().map(Arc::as_ref).collect::<Vec<_>>();
    Ok((concatenate(&arrays, &scratch, funding, check)?, consumed))
}

fn admit_selections(rows: usize, workspace: &MemoryReservation) -> Result<usize> {
    let count = add(rows, usize::from(rows == 1))?;
    accounting::reserve(
        workspace,
        sum(&[
            accounting::vector_peak::<ArrayRef>(count)?,
            accounting::vector_peak::<&dyn Array>(count)?,
        ])?,
    )?;
    Ok(count)
}

fn selected_column(
    rows: &[&StoredRow],
    index: usize,
    workspace: &MemoryReservation,
    check: &dyn Fn() -> Result<()>,
) -> Result<(ArrayRef, usize)> {
    check()?;
    let first = rows[0];
    let source = first.record.column(index);
    let offset = first.record.offset();
    let mut count = 1;
    for row in &rows[1..] {
        check()?;
        if !Arc::ptr_eq(source, row.record.column(index))
            || offset.checked_add(count) != Some(row.record.offset())
        {
            break;
        }
        count = add(count, 1)?;
    }
    admit_selection(source.as_ref(), workspace, check)?;
    Ok((source.slice(offset, count), count))
}

fn admit_selection(
    source: &dyn Array,
    workspace: &MemoryReservation,
    check: &dyn Fn() -> Result<()>,
) -> Result<()> {
    let nodes = accounting::shape_nodes(source.data_type())?;
    let controls = source_controls(source)?;
    accounting::reserve(
        workspace,
        sum(&[
            accounting::array_controls(nodes, product(nodes, 3)?)?,
            product(controls, 3)?,
            slice_controls(source, check)?,
        ])?,
    )
}

fn source_controls(source: &dyn Array) -> Result<usize> {
    source
        .get_array_memory_size()
        .checked_sub(source.get_buffer_memory_size())
        .ok_or_else(|| error("V2 checkpoint slice control size differs"))
}

fn concatenate(
    arrays: &[&dyn Array],
    workspace: &MemoryReservation,
    funding: &Arc<Funding>,
    check: &dyn Fn() -> Result<()>,
) -> Result<ArrayRef> {
    admit(arrays, workspace, funding, check)?;
    check()?;
    let result = concat(arrays).map_err(|_| error("V2 checkpoint payload compaction failed"))?;
    check()?;
    Ok(result)
}

#[cfg(test)]
#[path = "compact_tests.rs"]
mod tests;
