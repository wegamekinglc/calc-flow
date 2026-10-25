use std::{io::Write, sync::Arc};

use datafusion::{
    arrow::{datatypes::Schema, record_batch::RecordBatch},
    execution::memory_pool::MemoryReservation,
};

use super::{StateSegment, incremental, sql_ipc_buffer_bytes, sql_state_error};
use crate::{Batch, Result};

pub(super) struct SqlInputSegment {
    pub segment: StateSegment,
    _reservation: Arc<MemoryReservation>,
}

impl SqlInputSegment {
    pub(super) fn restored(
        segment: StateSegment,
        reservation: Arc<MemoryReservation>,
    ) -> Arc<Self> {
        Arc::new(Self {
            segment: segment.with_owner(reservation.clone()),
            _reservation: reservation,
        })
    }
}

pub(super) fn encode(
    batch: &Batch,
    reservation: MemoryReservation,
    check_cancelled: impl FnMut() -> Result<()>,
) -> Result<Arc<SqlInputSegment>> {
    encode_with_owner(batch, reservation, None, check_cancelled)
}

pub(super) fn encode_with_owner(
    batch: &Batch,
    reservation: MemoryReservation,
    owner: Option<MemoryReservation>,
    mut check_cancelled: impl FnMut() -> Result<()>,
) -> Result<Arc<SqlInputSegment>> {
    let scratch = scratch_bytes(batch)?;
    let mut buffer = PaidBuffer {
        bytes: Vec::new(),
        reservation: &reservation,
        scratch,
        owner: owner.as_ref(),
    };
    let mut started = false;
    super::encode_sql_state_into(batch, &mut buffer, || {
        check_cancelled()?;
        if !started {
            incremental::ensure_reservation(&reservation, scratch, "SQL checkpoint IPC")?;
            started = true;
        }
        Ok(())
    })?;
    let bytes = buffer.bytes;
    let reservation = settle_owner(bytes.capacity(), reservation, owner)?;
    let reservation = Arc::new(reservation);
    Ok(Arc::new(SqlInputSegment {
        segment: StateSegment::new(bytes).with_owner(reservation.clone()),
        _reservation: reservation,
    }))
}

fn settle_owner(
    capacity: usize,
    reservation: MemoryReservation,
    owner: Option<MemoryReservation>,
) -> Result<MemoryReservation> {
    Ok(if let Some(owner) = owner {
        settle_reservation(capacity, &owner)?;
        owner
    } else {
        settle_reservation(capacity, &reservation)?;
        reservation
    })
}

fn settle_reservation(capacity: usize, reservation: &MemoryReservation) -> Result<()> {
    let retained = capacity
        .checked_add(256)
        .ok_or_else(|| sql_state_error("SQL IPC retained charge overflowed"))?;
    if reservation.size() > retained {
        reservation.shrink(reservation.size() - retained);
    }
    Ok(())
}

fn scratch_bytes(batch: &Batch) -> Result<usize> {
    let table = batch.table_payload()?;
    let schema = schema_bytes(table.schema())?;
    let fields = table
        .schema()
        .fields()
        .iter()
        .try_fold(0usize, |bytes, field| bytes.checked_add(field.size()))
        .ok_or_else(|| sql_state_error("SQL IPC schema scratch overflowed"))?;
    let largest = largest_batch_bytes(table.batches())?;
    let width = fields
        .div_ceil(size_of::<datafusion::arrow::datatypes::DataType>())
        .max(1);
    let footer = table
        .batches()
        .len()
        .checked_mul(width)
        .ok_or_else(|| sql_state_error("SQL IPC footer scratch overflowed"))?;
    incremental::checked_bytes(
        8192,
        [(largest, 4), (schema, 16), (footer, 512)],
        "SQL checkpoint IPC",
    )
}

fn largest_batch_bytes(records: &[RecordBatch]) -> Result<usize> {
    records.iter().try_fold(0usize, |largest, record| {
        let bytes = record.columns().iter().try_fold(0usize, |bytes, array| {
            bytes
                .checked_add(sql_ipc_buffer_bytes(&array.to_data())?)
                .ok_or_else(|| sql_state_error("SQL IPC scratch overflowed"))
        })?;
        Ok::<_, crate::CalcFlowError>(largest.max(bytes))
    })
}

pub(super) fn schema_bytes(schema: &Schema) -> Result<usize> {
    let slots = incremental::checked_bytes(
        size_of::<Schema>(),
        [(
            schema.metadata().capacity(),
            2 * size_of::<(String, String)>(),
        )],
        "SQL IPC schema",
    )?;
    let metadata = schema
        .metadata()
        .iter()
        .try_fold(slots, |bytes, (key, value)| {
            bytes
                .checked_add(key.capacity())?
                .checked_add(value.capacity())
        })
        .ok_or_else(|| sql_state_error("SQL IPC schema metadata charge overflowed"))?;
    schema
        .fields()
        .iter()
        .try_fold(metadata, |bytes, field| bytes.checked_add(field.size()))
        .ok_or_else(|| sql_state_error("SQL IPC schema field charge overflowed"))
}

struct PaidBuffer<'a> {
    bytes: Vec<u8>,
    reservation: &'a MemoryReservation,
    scratch: usize,
    owner: Option<&'a MemoryReservation>,
}

impl PaidBuffer<'_> {
    fn reserve_growth(&self, length: usize) -> std::io::Result<usize> {
        let capacity = length
            .max(256)
            .checked_next_power_of_two()
            .ok_or_else(|| std::io::Error::other("SQL IPC capacity overflowed"))?;
        let owned = self.buffer_charge(capacity)?;
        let charge = self.scratch_charge(owned)?;
        self.reserve_owner(owned)?;
        incremental::ensure_reservation(self.reservation, charge, "SQL checkpoint IPC")
            .map_err(|error| std::io::Error::other(error.to_string()))?;
        Ok(capacity)
    }

    fn buffer_charge(&self, capacity: usize) -> std::io::Result<usize> {
        capacity
            .checked_add(self.bytes.capacity())
            .and_then(|n| n.checked_add(256))
            .ok_or_else(|| std::io::Error::other("SQL IPC buffer charge overflowed"))
    }

    fn scratch_charge(&self, owned: usize) -> std::io::Result<usize> {
        // The ownership lease already pays the engine pool for the wire buffer.
        self.scratch
            .checked_add(if self.owner.is_some() { 0 } else { owned })
            .ok_or_else(|| std::io::Error::other("SQL IPC scratch charge overflowed"))
    }

    fn reserve_owner(&self, owned: usize) -> std::io::Result<()> {
        if let Some(owner) = self.owner {
            incremental::ensure_reservation(owner, owned, "SQL compact checkpoint")
                .map_err(|e| std::io::Error::other(e.to_string()))?;
        }
        Ok(())
    }

    fn grow(&mut self, length: usize) -> std::io::Result<()> {
        let capacity = self.reserve_growth(length)?;
        self.bytes
            .try_reserve_exact(capacity - self.bytes.len())
            .map_err(|error| std::io::Error::other(error.to_string()))?;
        if self.bytes.capacity() > capacity {
            return Err(std::io::Error::other(
                "SQL IPC allocation exceeded prepaid capacity",
            ));
        }
        let settled = self.scratch + capacity + 256;
        if self.reservation.size() > settled {
            self.reservation.shrink(self.reservation.size() - settled);
        }
        Ok(())
    }
}

impl Write for PaidBuffer<'_> {
    fn write(&mut self, input: &[u8]) -> std::io::Result<usize> {
        let length = self
            .bytes
            .len()
            .checked_add(input.len())
            .ok_or_else(|| std::io::Error::other("SQL IPC byte count overflowed"))?;
        if length > self.bytes.capacity() {
            self.grow(length)?;
        }
        self.bytes.extend_from_slice(input);
        Ok(input.len())
    }

    fn flush(&mut self) -> std::io::Result<()> {
        Ok(())
    }
}
