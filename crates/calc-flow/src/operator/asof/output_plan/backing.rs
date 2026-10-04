use super::{checked, grow_workspace, overflow};
use crate::Result;
use datafusion::{
    arrow::{
        buffer::{Buffer, NullBuffer},
        record_batch::RecordBatch,
    },
    execution::memory_pool::MemoryReservation,
};

pub(super) fn source_bytes(
    record: &RecordBatch,
    workspace: &MemoryReservation,
    name: &str,
) -> Result<u64> {
    let (capacity, scratch_bytes) = scratch_bound(record.num_columns(), name)?;
    let mut scratch = workspace.new_empty();
    grow_workspace(&mut scratch, scratch_bytes, name)?;
    let mut owners = Vec::with_capacity(capacity);
    for column in record.columns() {
        let data = column.to_data();
        for buffer in data
            .buffers()
            .iter()
            .chain(data.nulls().map(NullBuffer::buffer))
        {
            add_owner(&mut owners, buffer, name)?;
        }
    }
    owners
        .into_iter()
        .try_fold(0, |total, (_, capacity)| checked(name, total, capacity))
}

fn scratch_bound(columns: usize, name: &str) -> Result<(usize, u64)> {
    let capacity = columns.checked_mul(3).ok_or_else(|| overflow(name))?;
    let bytes = (columns as u64)
        .checked_mul(1024)
        .and_then(|bytes| bytes.checked_add(1024))
        .ok_or_else(|| overflow(name))?;
    Ok((capacity, bytes))
}

fn add_owner(owners: &mut Vec<(usize, u64)>, buffer: &Buffer, name: &str) -> Result<()> {
    let extent = buffer
        .ptr_offset()
        .checked_add(buffer.len())
        .ok_or_else(|| overflow(name))?;
    let capacity = buffer.capacity().max(extent) as u64;
    let base = buffer.data_ptr().as_ptr() as usize;
    if let Some((_, current)) = owners.iter_mut().find(|(owner, _)| *owner == base) {
        *current = (*current).max(capacity);
    } else {
        owners.push((base, capacity));
    }
    Ok(())
}
