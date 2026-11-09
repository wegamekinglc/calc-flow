use std::mem::{align_of, size_of};

use super::{Result, invalid};

pub(super) fn add(left: u64, right: u64) -> Result<u64> {
    left.checked_add(right)
        .ok_or_else(|| invalid("allocation overflow"))
}
pub(super) fn mul(left: u64, right: u64) -> Result<u64> {
    left.checked_mul(right)
        .ok_or_else(|| invalid("allocation overflow"))
}
pub(super) fn size(value: usize) -> Result<u64> {
    u64::try_from(value).map_err(|_| invalid("allocation size"))
}
pub(super) fn count(value: i64) -> Result<u64> {
    u64::try_from(value).map_err(|_| invalid("negative length"))
}
pub(super) fn bytes<T>(count: u64) -> Result<u64> {
    mul(count, size(size_of::<T>())?)
}
pub(super) fn arc<T>() -> Result<u64> {
    add(
        bytes::<T>(1)?,
        size(2 * size_of::<usize>() + align_of::<T>())?,
    )
}
pub(super) fn slice_arc<T>(len: u64) -> Result<u64> {
    add(
        bytes::<T>(len)?,
        size(2 * size_of::<usize>() + align_of::<T>())?,
    )
}
pub(super) fn grow<T>(len: u64) -> Result<u64> {
    if len == 0 {
        return Ok(0);
    }
    let minimum = if size_of::<T>() == 1 { 8 } else { 4 };
    mul(bytes::<T>(len.max(minimum))?, 3)
}
pub(super) fn aligned(bytes: u64) -> Result<u64> {
    Ok(add(bytes, 63)? / 64 * 64)
}
pub(super) fn buffer(bytes: u64) -> Result<u64> {
    // Pinned Bytes: pointer + len + largest Deallocation variant, then Arc header.
    add(aligned(bytes)?, arc::<[usize; 6]>()?)
}
pub(super) fn bitmap(len: u64) -> Result<u64> {
    buffer(add(len, 7)? / 8)
}
pub(super) fn hash<T>(len: u64) -> Result<u64> {
    if len == 0 {
        return Ok(0);
    }
    let buckets = add(mul(len, 8)?, 6)? / 7;
    let buckets = buckets
        .max(4)
        .checked_next_power_of_two()
        .ok_or_else(|| invalid("hash capacity"))?;
    add(bytes::<T>(buckets)?, add(buckets, 32)?)
}
pub(super) fn sum<const N: usize>(values: [Result<u64>; N]) -> Result<u64> {
    values
        .into_iter()
        .try_fold(0, |total, value| add(total, value?))
}
pub(super) fn scale(value: Result<u64>, factor: u64) -> Result<u64> {
    mul(value?, factor)
}
pub(super) fn index(value: u64) -> Result<usize> {
    usize::try_from(value).map_err(|_| invalid("allocation index"))
}
