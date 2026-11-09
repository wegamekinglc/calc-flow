use super::{Result, invalid};
use std::{alloc::Layout, sync::atomic::AtomicUsize};

pub(super) fn add(a: u64, b: u64) -> Result<u64> {
    a.checked_add(b).ok_or_else(|| invalid("size overflow"))
}
pub(super) fn mul(a: u64, b: u64) -> Result<u64> {
    a.checked_mul(b).ok_or_else(|| invalid("size overflow"))
}
pub(super) fn size(n: usize) -> Result<u64> {
    u64::try_from(n).map_err(|_| invalid("size"))
}
pub(super) fn bytes<T>(n: u64) -> Result<u64> {
    mul(n, size(size_of::<T>())?)
}
pub(super) fn sum<const N: usize>(values: [Result<u64>; N]) -> Result<u64> {
    values
        .into_iter()
        .try_fold(0, |total, value| add(total, value?))
}
pub(super) fn scale(value: Result<u64>, n: u64) -> Result<u64> {
    mul(value?, n)
}
pub(super) fn align(n: u64) -> Result<u64> {
    mul(add(n, 63)? / 64, 64)
}
pub(super) fn bitmap(n: u64) -> Result<u64> {
    align(add(n, 7)? / 8)
}
pub(super) fn grow(n: u64, width: u64) -> Result<u64> {
    if n == 0 {
        return Ok(0);
    }
    mul(mul(n.max(if width == 1 { 8 } else { 4 }), width)?, 3)
}
pub(super) fn retained(n: u64, width: u64) -> Result<u64> {
    if n == 0 {
        return Ok(0);
    }
    mul(mul(n.max(if width == 1 { 8 } else { 4 }), width)?, 2)
}
pub(super) fn arc<T>() -> Result<u64> {
    let (layout, _) = Layout::new::<[AtomicUsize; 2]>()
        .extend(Layout::new::<T>())
        .map_err(|_| invalid("array layout"))?;
    size(layout.pad_to_align().size())
}
pub(super) fn arc_slice<T>(n: usize) -> Result<u64> {
    let items = Layout::array::<T>(n).map_err(|_| invalid("array slice layout"))?;
    let (layout, _) = Layout::new::<[AtomicUsize; 2]>()
        .extend(items)
        .map_err(|_| invalid("array slice layout"))?;
    size(layout.pad_to_align().size())
}
pub(super) fn buffer_headers(n: u64) -> Result<u64> {
    bytes::<[usize; 9]>(n)
}

pub(super) fn scalar_capacity(rows: usize) -> Result<()> {
    Layout::array::<datafusion::common::ScalarValue>(rows)
        .map(|_| ())
        .map_err(|_| invalid("scalar vector capacity"))
}

pub(super) fn vector<T>(n: usize) -> Result<u64> {
    grow(size(n)?, size(size_of::<T>())?)
}

pub(super) fn physical_width(
    dtype: &datafusion::arrow::datatypes::DataType,
) -> Result<Option<u64>> {
    if let datafusion::arrow::datatypes::DataType::FixedSizeBinary(width) = dtype {
        return u64::try_from(*width)
            .map(Some)
            .map_err(|_| invalid("fixed binary width"));
    }
    dtype.primitive_width().map(size).transpose()
}

pub(super) fn scalar_view_length(len: u64) -> Result<()> {
    if len >= u64::from(u32::MAX) {
        return Err(invalid("fresh view block length"));
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    #[test]
    fn test_fresh_scalar_view_block_length_obeys_pinned_finish_limit() {
        let maximum = u64::from(u32::MAX);
        assert!(super::scalar_view_length(0).is_ok());
        assert!(super::scalar_view_length(maximum - 1).is_ok());
        assert!(super::scalar_view_length(maximum).is_err());
        assert!(super::scalar_view_length(maximum + 1).is_err());
    }
}
