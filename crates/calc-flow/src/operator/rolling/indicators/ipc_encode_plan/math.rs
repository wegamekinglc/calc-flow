use crate::{CalcFlowError, Result};

pub(super) fn invalid() -> CalcFlowError {
    CalcFlowError::Format {
        message: "rolling IPC encoding allocation size is invalid".into(),
    }
}
pub(super) fn add(a: u64, b: u64) -> Result<u64> {
    a.checked_add(b).ok_or_else(invalid)
}
pub(super) fn mul(a: u64, b: u64) -> Result<u64> {
    a.checked_mul(b).ok_or_else(invalid)
}
pub(super) fn size(n: usize) -> Result<u64> {
    u64::try_from(n).map_err(|_| invalid())
}
pub(super) fn aligned(n: u64) -> Result<u64> {
    mul(add(n, 63)? / 64, 64)
}
pub(super) fn vector(n: u64, width: u64) -> Result<u64> {
    add(mul(n, width)?, 11)
}
pub(super) fn table(slots: u64) -> Result<u64> {
    add(mul(slots, 17)?, 24)
}
pub(super) fn string(bytes: usize) -> Result<u64> {
    add(size(bytes)?, 16)
}

// Vec doubling plus the old allocation retained during a moving growth.
pub(super) fn growing(n: u64, width: u64) -> Result<u64> {
    if n == 0 {
        return Ok(0);
    }
    let floor = if width == 1 { 8 } else { 4 };
    mul(mul(n.max(floor), width)?, 3)
}

// The pinned DefaultAllocator doubles its length; Vec starts with at least eight bytes.
pub(super) fn builder(bytes: u64, tables: u64) -> Result<u64> {
    if bytes > u64::try_from(i32::MAX).map_err(|_| invalid())? {
        return Err(invalid());
    }
    let storage = growing(bytes.max(8), 1)?;
    let vtables = growing(tables, 4)?;
    add(add(storage, vtables)?, growing(8, 8)?)
}

pub(super) fn hash_bytes(capacity: usize, width: usize) -> Result<u64> {
    if capacity == 0 {
        return Ok(0);
    }
    let buckets = hash_slots(capacity)?;
    sum([mul(buckets, add(size(width)?, 1)?), Ok(64)])
}
fn hash_slots(capacity: usize) -> Result<u64> {
    let wanted = add(mul(size(capacity)?, 8)?, 6)? / 7;
    wanted
        .checked_next_power_of_two()
        .map(|n| n.max(4))
        .ok_or_else(invalid)
        .and_then(|n| usize::try_from(n).map(|_| n).map_err(|_| invalid()))
}

pub(super) fn sum<const N: usize>(values: [Result<u64>; N]) -> Result<u64> {
    values
        .into_iter()
        .try_fold(0, |total, value| add(total, value?))
}
pub(super) fn elements<T>(count: u64) -> Result<u64> {
    mul(count, size(size_of::<T>())?)
}
pub(super) fn vec_elements<T>(count: u64) -> Result<u64> {
    growing(count, size(size_of::<T>())?)
}
pub(super) fn vec_slots<T>(count: usize) -> Result<u64> {
    vec_elements::<T>(size(count)?)
}
