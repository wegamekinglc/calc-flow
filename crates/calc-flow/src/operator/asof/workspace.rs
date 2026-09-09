use super::{StreamAsofJoinOperator, checked, reason, state::State};
use crate::{Batch, Result, StreamingFailureReason};
use datafusion::{
    arrow::{array::ArrayRef, datatypes::Schema, record_batch::RecordBatch},
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
        self.reserve_input_rows(batch, input, |record, row| {
            row_workspace(record, row, &self.name)
        })
    }

    pub(super) fn identity_workspace(
        &self,
        batch: &Batch,
        input: super::admission::ValidatedInput,
    ) -> Result<MemoryReservation> {
        self.reserve_input_rows(batch, input, |record, row| {
            identity_row_workspace(record, row, input.side(&self.spec), &self.name)
        })
    }

    fn reserve_input_rows(
        &self,
        batch: &Batch,
        input: super::admission::ValidatedInput,
        charge: impl Fn(&RecordBatch, usize) -> Result<u64>,
    ) -> Result<MemoryReservation> {
        let side = input.side(&self.spec);
        let mut bytes = 0;
        for record in batch.table_payload()?.batches() {
            for row in 0..record.num_rows() {
                if input.is_late(super::admission::times(record, side).value(row)) {
                    continue;
                }
                bytes = checked(&self.name, bytes, charge(record, row)?)?;
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

fn identity_row_workspace(
    record: &RecordBatch,
    row: usize,
    side: &super::AsofJoinSide,
    name: &str,
) -> Result<u64> {
    let mut bytes = 2048;
    for field in side.keys().iter().chain(side.sequence_by()) {
        let column = record.column(record.schema().index_of(field).expect("validated schema"));
        let logical = column_workspace(column, row)?;
        let charge = logical.checked_mul(4).ok_or_else(|| {
            reason(
                name,
                StreamingFailureReason::AsofCounterOverflow,
                "ASOF identity workspace arithmetic overflowed",
            )
        })?;
        bytes = checked(name, bytes, charge)?;
    }
    Ok(bytes)
}

fn row_workspace(batch: &RecordBatch, row: usize, name: &str) -> Result<u64> {
    let mut bytes = schema_workspace(&batch.schema(), name)?;
    for column in batch.columns() {
        bytes = checked(name, bytes, column_workspace(column, row)?)?;
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

fn schema_workspace(schema: &Schema, name: &str) -> Result<u64> {
    let mut bytes = 4096;
    for (key, value) in schema.metadata() {
        bytes = checked(name, bytes, key.len() as u64)?;
        bytes = checked(name, bytes, value.len() as u64 + 256)?;
    }
    for field in schema.fields() {
        bytes = checked(name, bytes, field.size() as u64 + 256)?;
    }
    Ok(bytes)
}

fn column_workspace(column: &ArrayRef, row: usize) -> Result<u64> {
    column
        .to_data()
        .slice(row, 1)
        .get_slice_memory_size()
        .map(|bytes| bytes as u64)
        .map_err(|error| super::arrow_error(&error))
}
