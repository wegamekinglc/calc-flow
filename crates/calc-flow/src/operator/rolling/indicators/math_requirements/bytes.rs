use std::mem::{align_of, size_of};

use crate::{CalcFlowError, Result};

pub(in super::super) fn add(left: u64, right: u64) -> Result<u64> {
    left.checked_add(right).ok_or_else(invalid)
}
pub(in super::super) fn product(count: usize, width: usize) -> Result<u64> {
    size(count.checked_mul(width).ok_or_else(invalid)?)
}
pub(in super::super) fn size(value: usize) -> Result<u64> {
    u64::try_from(value).map_err(|_| invalid())
}
pub(in super::super) fn slots<T>(count: usize) -> Result<u64> {
    product(count, size_of::<T>())
}
pub(in super::super) fn total<const N: usize>(values: [Result<u64>; N]) -> Result<u64> {
    values
        .into_iter()
        .try_fold(0, |sum, value| add(sum, value?))
}
pub(in super::super) fn repeat(value: Result<u64>, count: usize) -> Result<u64> {
    value?.checked_mul(size(count)?).ok_or_else(invalid)
}
pub(in super::super) fn growing<T>(count: usize) -> Result<u64> {
    if count == 0 || size_of::<T>() == 0 {
        return Ok(0);
    }
    // Vec/VecDeque doubles an existing allocation. A cloned deque may begin
    // at an arbitrary exact length, so next_power_of_two(count) is insufficient.
    // New capacity is at most max(2*count, MIN_NON_ZERO_CAP); old <= new/2.
    let minimum = match size_of::<T>() {
        1 => 8,
        2..=1024 => 4,
        _ => 1,
    };
    let capacity = count.checked_mul(2).ok_or_else(invalid)?.max(minimum);
    slots::<T>(capacity.checked_add(capacity / 2).ok_or_else(invalid)?)
}
pub(in super::super) fn hash_table(count: usize) -> Result<u64> {
    if count == 0 {
        return Ok(0);
    }
    // Pinned std/hashbrown reserves a power-of-two table at <=7/8 occupancy.
    // Include the old table during growth or tombstone rehash, and Group width.
    let buckets = hash_buckets(count)?;
    let allocation = add(product(buckets, size_of::<(u64, usize)>() + 1)?, 16)?;
    let aligned = add(allocation, size(align_of::<(u64, usize)>())?)?;
    add(aligned, aligned)
}
fn hash_buckets(count: usize) -> Result<usize> {
    let needed = count
        .checked_mul(8)
        .and_then(|value| value.checked_add(6))
        .ok_or_else(invalid)?
        / 7;
    Ok(needed
        .checked_next_power_of_two()
        .ok_or_else(invalid)?
        .max(4))
}
pub(in super::super) fn tree_nodes<K, V>(count: usize) -> Result<u64> {
    // Pinned std BTreeMap: B=6, 11 key/value slots, 12 child edges. Charge
    // one internal-sized node per entry (every live node contains an entry).
    let slots = product(11, size_of::<K>() + size_of::<V>())?;
    let header = size(size_of::<usize>() * 14 + size_of::<u16>() * 2)?;
    let padding = size(align_of::<K>() + align_of::<V>() + align_of::<usize>())?;
    let node = add(add(slots, header)?, padding)?;
    size(count)?.checked_mul(node).ok_or_else(invalid)
}
pub(in super::super) fn invalid() -> CalcFlowError {
    CalcFlowError::Format {
        message: "rolling numerical workspace size is invalid or overflowed".into(),
    }
}

#[cfg(test)]
#[path = "bytes_tests.rs"]
mod tests;
