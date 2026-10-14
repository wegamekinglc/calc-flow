use std::sync::Arc;

use datafusion::arrow::{array::ArrayRef, datatypes::SchemaRef, record_batch::RecordBatch};
use datafusion::execution::memory_pool::MemoryReservation;

use super::super::{
    ipc::accounting::{self, reserve, sum},
    projected_copy,
};
use super::{Decoder, StoredRow};
use crate::Result;
use crate::operator::join::{
    columnar::{RowPayload, SelectedRow},
    state_projection::Layout,
};
use crate::runtime::streaming::gather_work::RetirementGuard;

impl Decoder<'_, '_> {
    pub(super) fn migrate_legacy(&mut self, rows: &mut [Vec<StoredRow>; 2]) -> Result<bool> {
        let Some(layout) = self.config.projection.filter(|_| !self.projected) else {
            return Ok(false);
        };
        let retirement = self.shared_retirement()?;
        migrate_rows(
            rows,
            layout,
            &self.workspace,
            retirement.as_ref(),
            self.check,
        )?;
        Ok(true)
    }

    pub(super) fn shared_retirement(&mut self) -> Result<Option<Arc<RetirementGuard>>> {
        reserve(&self.workspace, accounting::arc::<RetirementGuard>()?)?;
        Ok(self.retirements.pop().map(Arc::new))
    }
}

pub(super) fn migrate_rows(
    rows: &mut [Vec<StoredRow>; 2],
    layout: &Layout,
    workspace: &MemoryReservation,
    retirement: Option<&Arc<RetirementGuard>>,
    check: &dyn Fn() -> Result<()>,
) -> Result<()> {
    for (side, rows) in rows.iter_mut().enumerate() {
        migrate_side(rows, side, layout, workspace, retirement, check)?;
    }
    check()
}

fn migrate_side(
    rows: &mut [StoredRow],
    side: usize,
    layout: &Layout,
    workspace: &MemoryReservation,
    retirement: Option<&Arc<RetirementGuard>>,
    check: &dyn Fn() -> Result<()>,
) -> Result<()> {
    let mut start = 0;
    while start < rows.len() {
        check()?;
        let end = parent_end(rows, start, check)?;
        migrate_group(
            &mut rows[start..end],
            side,
            layout,
            workspace,
            retirement.cloned(),
            check,
        )?;
        start = end;
    }
    Ok(())
}

fn parent_end(rows: &[StoredRow], start: usize, check: &dyn Fn() -> Result<()>) -> Result<usize> {
    let columns = rows[start].record.columns();
    for (index, row) in rows.iter().enumerate().skip(start) {
        check()?;
        if !std::ptr::eq(columns, row.record.columns()) {
            return Ok(index);
        }
    }
    Ok(rows.len())
}

fn migrate_group(
    rows: &mut [StoredRow],
    side: usize,
    layout: &Layout,
    workspace: &MemoryReservation,
    retirement: Option<Arc<RetirementGuard>>,
    check: &dyn Fn() -> Result<()>,
) -> Result<()> {
    let input_credit = workspace.new_empty();
    reserve(
        &input_credit,
        group_controls(rows.len(), &layout.schemas[side])?,
    )?;
    let selected = selection(rows, check)?;
    let source = selected_record(&rows[0].record, side, layout)?;
    let owner = projected_copy::copy_shared(
        &source,
        &selected,
        layout.schemas.clone(),
        &input_credit,
        retirement,
        check,
    )?;
    for (index, row) in rows.iter_mut().enumerate() {
        check()?;
        row.record = RowPayload::RestoredV2 {
            chunk: Arc::clone(&owner),
            row: index,
        };
    }
    Ok(())
}

fn group_controls(rows: usize, schema: &SchemaRef) -> Result<usize> {
    sum(&[
        accounting::vector_peak::<SelectedRow>(rows)?,
        accounting::vector_peak::<ArrayRef>(schema.fields().len())?,
        size_of::<RecordBatch>(),
    ])
}

fn selection(rows: &[StoredRow], check: &dyn Fn() -> Result<()>) -> Result<Vec<SelectedRow>> {
    rows.iter()
        .map(|row| {
            check()?;
            Ok(SelectedRow {
                source: row.record.offset(),
                row_id: row.row_id,
                time: row.event_time,
                retain: true,
            })
        })
        .collect()
}

fn selected_record(source: &RowPayload, side: usize, layout: &Layout) -> Result<RecordBatch> {
    let columns = source.columns();
    RecordBatch::try_new(
        Arc::clone(&layout.schemas[side]),
        layout.columns[side]
            .iter()
            .map(|&index| Arc::clone(&columns[index]))
            .collect(),
    )
    .map_err(|_| super::super::geometry::invalid("legacy projected record reconstruction failed"))
}
