use super::{StreamAsofJoinOperator, checked, reason, state::State};
use crate::{Batch, Result, StreamingFailureReason};
use datafusion::{
    arrow::record_batch::RecordBatch,
    execution::memory_pool::{MemoryConsumer, MemoryReservation},
};

impl StreamAsofJoinOperator {
    pub(super) fn reserve_workspace(&self, bytes: u64) -> Result<MemoryReservation> {
        let bytes = usize::try_from(bytes).map_err(|_| {
            reason(
                &self.name,
                StreamingFailureReason::AsofWorkspaceLimitExceeded,
                "ASOF workspace exceeds address domain",
            )
        })?;
        let reservation = MemoryConsumer::new("asof-owned-workspace").register(&self.runtime.pool);
        reservation.try_grow(bytes).map_err(|_| {
            reason(
                &self.name,
                StreamingFailureReason::AsofWorkspaceLimitExceeded,
                "ASOF aggregate workspace exceeds max_state_bytes",
            )
        })?;
        Ok(reservation)
    }

    pub(super) fn input_workspace(
        &self,
        batch: &Batch,
        input: super::admission::ValidatedInput,
    ) -> Result<MemoryReservation> {
        let mut bytes = 0;
        let side = if input.index == 0 {
            self.spec.left()
        } else {
            self.spec.right()
        };
        for record in batch.table_payload()?.batches() {
            for row in 0..record.num_rows() {
                if input
                    .watermark
                    .is_some_and(|wm| super::admission::times(record, side).value(row) < wm)
                {
                    continue;
                }
                let row_charge = row_workspace(record, row, &self.name)?;
                bytes = checked(&self.name, bytes, row_charge)?;
            }
        }
        self.reserve_workspace(bytes)
    }

    pub(super) fn identity_workspace(
        &self,
        batch: &Batch,
        input: super::admission::ValidatedInput,
    ) -> Result<MemoryReservation> {
        let side = if input.index == 0 {
            self.spec.left()
        } else {
            self.spec.right()
        };
        let mut bytes = 0;
        for record in batch.table_payload()?.batches() {
            for row in 0..record.num_rows() {
                if input
                    .watermark
                    .is_some_and(|wm| super::admission::times(record, side).value(row) < wm)
                {
                    continue;
                }
                bytes = checked(&self.name, bytes, 2048)?;
                for name in side.keys().iter().chain(side.sequence_by()) {
                    let column =
                        record.column(record.schema().index_of(name).expect("validated schema"));
                    let logical = column
                        .to_data()
                        .slice(row, 1)
                        .get_slice_memory_size()
                        .map_err(|error| super::arrow_error(&error))?
                        as u64;
                    bytes = checked(
                        &self.name,
                        bytes,
                        logical.checked_mul(4).ok_or_else(|| {
                            reason(
                                &self.name,
                                StreamingFailureReason::AsofCounterOverflow,
                                "ASOF identity workspace arithmetic overflowed",
                            )
                        })?,
                    )?;
                }
            }
        }
        self.reserve_workspace(bytes)
    }

    pub(super) fn state_workspace(&self, state: &State) -> Result<MemoryReservation> {
        let rows = state.inventory(None, &self.name)?.identities;
        let bytes = rows
            .checked_mul(384)
            .and_then(|value| value.checked_add(state.right.len() as u64 * 64))
            .ok_or_else(|| {
                reason(
                    &self.name,
                    StreamingFailureReason::AsofCounterOverflow,
                    "ASOF state clone workspace arithmetic overflowed",
                )
            })?;
        self.reserve_workspace(bytes)
    }
}

fn row_workspace(batch: &RecordBatch, row: usize, name: &str) -> Result<u64> {
    let schema = batch.schema();
    let mut bytes = 4096;
    for (key, value) in schema.metadata() {
        bytes = checked(name, bytes, key.len() as u64)?;
        bytes = checked(name, bytes, value.len() as u64 + 256)?;
    }
    for field in schema.fields() {
        bytes = checked(name, bytes, field.size() as u64 + 256)?;
    }
    for column in batch.columns() {
        let data = column.to_data();
        let logical = data
            .slice(row, 1)
            .get_slice_memory_size()
            .map_err(|error| super::arrow_error(&error))?;
        bytes = checked(name, bytes, logical as u64)?;
        bytes = checked(name, bytes, 256)?;
    }
    bytes.checked_mul(8).ok_or_else(|| {
        reason(
            name,
            StreamingFailureReason::AsofCounterOverflow,
            "ASOF workspace arithmetic overflowed",
        )
    })
}
