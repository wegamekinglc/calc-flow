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
        concat::bulk::admit,
    },
    payload::Funding,
};
use super::buffer::error;
use crate::Result;

pub(super) struct CompactBatch {
    pub(super) record: RecordBatch,
    _funding: Arc<Funding>,
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
    let count = admit_selections(rows.len(), workspace)?;
    let mut selected = Vec::with_capacity(count);
    for row in rows {
        selected.push(selected_column(row, index, workspace, check)?);
    }
    if rows.len() == 1 {
        let first = &selected[0];
        admit_selection(first.as_ref(), workspace)?;
        selected.push(first.slice(0, 0));
    }
    let arrays = selected
        .iter()
        .map(|array| array.as_ref())
        .collect::<Vec<_>>();
    concatenate(&arrays, workspace, funding, check)
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
    row: &StoredRow,
    index: usize,
    workspace: &MemoryReservation,
    check: &dyn Fn() -> Result<()>,
) -> Result<ArrayRef> {
    check()?;
    admit_selection(row.record.column(index).as_ref(), workspace)?;
    Ok(row.record.column_view(index))
}

fn admit_selection(source: &dyn Array, workspace: &MemoryReservation) -> Result<()> {
    let nodes = accounting::shape_nodes(source.data_type())?;
    accounting::reserve(
        workspace,
        sum(&[
            accounting::array_controls(nodes, product(nodes, 3)?)?,
            product(source.get_array_memory_size(), 3)?,
        ])?,
    )
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
