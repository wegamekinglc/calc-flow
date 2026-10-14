use std::sync::Arc;

use super::super::{
    AdmittedRow, StoredRow, StreamJoinOperator,
    columnar::{CopySelection, Quantum, RowPayload, SelectedRow},
    operator_error,
};
use crate::{Result, StreamOperatorContext};
use datafusion::arrow::record_batch::RecordBatch;

impl StreamJoinOperator {
    pub(in crate::operator::join) async fn project_admitted(
        &mut self,
        admitted: &mut [AdmittedRow],
        retained: &mut [StoredRow],
        side: usize,
        context: &StreamOperatorContext<'_>,
    ) -> Result<()> {
        if self.payload_projection.is_none() {
            return Ok(());
        }
        let mut quantum = Quantum::default();
        let mut start = 0;
        while start < admitted.len() {
            let end = parent_end(admitted, start, context, &mut quantum).await?;
            self.project_group(&mut admitted[start..end], side, context, &mut quantum)
                .await?;
            start = end;
        }
        let mut selected = retained.iter_mut();
        for source in admitted {
            quantum.step(context, 8, 0).await?;
            if source.retain {
                let row = selected
                    .next()
                    .expect("retained rows preserve admission order");
                debug_assert_eq!(row.row_id, source.row_id);
                row.record = source.record.clone();
            }
        }
        context.check_cancelled()
    }

    async fn project_group(
        &mut self,
        rows: &mut [AdmittedRow],
        side: usize,
        context: &StreamOperatorContext<'_>,
        quantum: &mut Quantum,
    ) -> Result<()> {
        let selected = self.select_projected_rows(rows, context, quantum).await?;
        let controls = self.projected_column_credit(side)?;
        let record = self.project_payload(&rows[0].record, side)?;
        let payload = if let Some(chunk) = self
            .owned_payload_columns(record.columns(), side, &selected.rows, context, quantum)
            .await?
        {
            RowPayload::Shared { chunk, row: 0 }
        } else {
            let chunk = self
                .copy_projected_payload(record, selected, controls, context)
                .await?;
            RowPayload::RestoredV2 { chunk, row: 0 }
        };
        assign_payload(rows, &payload, context, quantum).await
    }

    async fn select_projected_rows(
        &mut self,
        rows: &[AdmittedRow],
        context: &StreamOperatorContext<'_>,
        quantum: &mut Quantum,
    ) -> Result<CopySelection> {
        let mut selected = CopySelection::reserve(self, rows.len())?.ok_or_else(|| {
            operator_error(&self.name, "retained selection credit admission failed")
        })?;
        for row in rows {
            quantum.step(context, 8, 0).await?;
            selected.rows.push(SelectedRow {
                source: row.record.offset(),
                row_id: row.row_id,
                time: row.event_time,
                retain: row.retain,
            });
        }
        Ok(selected)
    }

    fn projected_column_credit(
        &mut self,
        side: usize,
    ) -> Result<datafusion::execution::memory_pool::MemoryReservation> {
        let width = self.retained_schema(side).fields().len();
        let bytes = width
            .checked_mul(size_of::<datafusion::arrow::array::ArrayRef>())
            .and_then(|bytes| bytes.checked_add(size_of::<RecordBatch>()))
            .ok_or_else(|| super::super::charge_overflow(&self.name))?;
        self.optional_credit(bytes)?
            .ok_or_else(|| operator_error(&self.name, "retained column credit admission failed"))
    }

    fn project_payload(&self, source: &RowPayload, side: usize) -> Result<RecordBatch> {
        let layout = self
            .payload_projection
            .as_ref()
            .expect("projected admission");
        let columns = source
            .shared_columns()
            .expect("admission preserves full parent");
        RecordBatch::try_new(
            Arc::clone(&layout.schemas[side]),
            layout.columns[side]
                .iter()
                .map(|&index| Arc::clone(&columns[index]))
                .collect(),
        )
        .map_err(|error| {
            operator_error(&self.name, &format!("retained projection failed: {error}"))
        })
    }
}

async fn assign_payload(
    rows: &mut [AdmittedRow],
    payload: &RowPayload,
    context: &StreamOperatorContext<'_>,
    quantum: &mut Quantum,
) -> Result<()> {
    for (index, row) in rows.iter_mut().enumerate() {
        quantum.step(context, 8, 0).await?;
        row.record = match payload {
            RowPayload::Shared { chunk, .. } => RowPayload::Shared {
                chunk: Arc::clone(chunk),
                row: index,
            },
            RowPayload::RestoredV2 { chunk, .. } => RowPayload::RestoredV2 {
                chunk: Arc::clone(chunk),
                row: index,
            },
            _ => unreachable!("projected admission owns a funded selected payload"),
        };
    }
    Ok(())
}

async fn parent_end(
    rows: &[AdmittedRow],
    start: usize,
    context: &StreamOperatorContext<'_>,
    quantum: &mut Quantum,
) -> Result<usize> {
    let columns = rows[start]
        .record
        .shared_columns()
        .expect("full admission parent");
    for (index, row) in rows.iter().enumerate().skip(start) {
        quantum.step(context, 8, 0).await?;
        if row
            .record
            .shared_columns()
            .is_none_or(|other| !std::ptr::eq(columns, other))
        {
            return Ok(index);
        }
    }
    Ok(rows.len())
}
