use std::sync::Arc;

use super::super::{
    AdmittedBatch, AdmittedRows, StoredRow, StreamJoinOperator,
    columnar::{OwnedSelection, Quantum, RowPayload, SelectedRows},
    operator_error,
};
use crate::{Result, StreamOperatorContext};
use datafusion::arrow::record_batch::RecordBatch;

impl StreamJoinOperator {
    pub(in crate::operator::join) async fn project_admitted(
        &mut self,
        admitted: &mut AdmittedBatch,
        retained: &mut [StoredRow],
        side: usize,
        context: &StreamOperatorContext<'_>,
    ) -> Result<()> {
        if self.payload_projection.is_none() {
            return Ok(());
        }
        self.project_admitted_payloads(admitted, side, context)
            .await?;
        let mut quantum = Quantum::default();
        let mut selected = retained.iter_mut();
        for group in (&*admitted as &dyn AdmittedRows).groups() {
            for start in (0..group.len()).step_by(64) {
                let end = (start + 64).min(group.len());
                let mut bits = group.retained_bits(start, end - start);
                while bits != 0 {
                    let index = start + bits.trailing_zeros() as usize;
                    bits &= bits - 1;
                    quantum.step(context, 8, 0).await?;
                    let source = group.row(index);
                    let row = selected
                        .next()
                        .expect("retained rows preserve admission order");
                    debug_assert_eq!(row.row_id, source.row_id);
                    row.record = source.record.owned();
                }
            }
        }
        context.check_cancelled()
    }

    pub(in crate::operator::join) async fn project_admitted_payloads(
        &mut self,
        admitted: &mut AdmittedBatch,
        side: usize,
        context: &StreamOperatorContext<'_>,
    ) -> Result<()> {
        if self.payload_projection.is_none() {
            return Ok(());
        }
        let mut quantum = Quantum::default();
        let records = admitted.record_count();
        for index in 0..records {
            let range = admitted.group_range(index);
            if range.is_empty() {
                continue;
            }
            let group = super::super::admitted::AdmittedGroup::new(&*admitted, range);
            if group.any_retained() {
                let payload = self
                    .project_group(group, side, context, &mut quantum)
                    .await?;
                admitted.replace(index, payload);
            } else {
                // No row of this group enters state, so the projected parent
                // can share the incoming buffers; the gather still reads its
                // physical rows through the view.
                let record = self.project_payload(group.row(0).record, side)?;
                quantum.step(context, 1, 0).await?;
                admitted.replace_view(index, Arc::new(record));
            }
        }
        context.check_cancelled()
    }

    async fn project_group(
        &mut self,
        rows: super::super::admitted::AdmittedGroup<'_>,
        side: usize,
        context: &StreamOperatorContext<'_>,
        quantum: &mut Quantum,
    ) -> Result<RowPayload> {
        let controls = self.projected_column_credit(side)?;
        let record = self.project_payload(rows.row(0).record, side)?;
        let selection = SelectedRows::Admitted(rows);
        if let Some(chunk) = self
            .owned_payload_columns(record.columns(), side, selection, context, quantum)
            .await?
        {
            return Ok(RowPayload::Shared { chunk, row: 0 });
        }
        let selected = OwnedSelection::from_group(self, rows, context, quantum).await?;
        let chunk = self
            .copy_projected_payload(record, selected, controls, context)
            .await?;
        Ok(RowPayload::RestoredV2 { chunk, row: 0 })
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

    fn project_payload(
        &self,
        source: super::super::PayloadRef<'_>,
        side: usize,
    ) -> Result<RecordBatch> {
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
