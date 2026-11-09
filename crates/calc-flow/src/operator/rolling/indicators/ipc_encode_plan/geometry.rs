use super::{
    Check,
    math::{add, aligned, growing, invalid, size},
};
use crate::Result;
use datafusion::arrow::{array::ArrayData, datatypes::DataType};

#[derive(Clone, Copy)]
pub(super) struct Window {
    pub(super) offset: usize,
    pub(super) len: usize,
}
impl Window {
    pub(super) fn whole(data: &ArrayData) -> Self {
        Self {
            offset: data.offset(),
            len: data.len(),
        }
    }
    pub(super) fn slice(data: &ArrayData, offset: usize, len: usize) -> Result<Self> {
        if offset.checked_add(len).ok_or_else(invalid)? > data.len() {
            return Err(invalid());
        }
        Ok(Self {
            offset: data.offset().checked_add(offset).ok_or_else(invalid)?,
            len,
        })
    }
}
pub(super) struct Range {
    pub(super) start: usize,
    pub(super) len: usize,
}
#[derive(Clone, Copy)]
pub(super) struct RunSlice {
    pub(super) first: usize,
    pub(super) count: usize,
    pub(super) width: usize,
}

pub(super) fn bits(len: usize) -> Result<u64> {
    Ok(add(size(len)?, 7)? / 8)
}
pub(super) fn bit_copy(len: usize) -> Result<u64> {
    growing(add(size(len)?, 63)? / 64, 8)
}
pub(super) fn bitmap_copy(data: &ArrayData, window: Window) -> Result<Option<u64>> {
    let Some(nulls) = data.nulls() else {
        return aligned(bits(window.len)?).map(Some);
    };
    let shift = window
        .offset
        .checked_sub(data.offset())
        .ok_or_else(invalid)?;
    let start = nulls.offset().checked_add(shift).ok_or_else(invalid)?;
    if start.is_multiple_of(8) {
        Ok(None)
    } else {
        bit_copy(window.len).map(Some)
    }
}
pub(super) fn range(data: &ArrayData, window: Window, width: usize) -> Result<Range> {
    if window.len == 0 {
        return Ok(Range { start: 0, len: 0 });
    }
    let start = offset(data, window.offset, width)?;
    let index = window.offset.checked_add(window.len).ok_or_else(invalid)?;
    let end = offset(data, index, width)?;
    Ok(Range {
        start,
        len: end.checked_sub(start).ok_or_else(invalid)?,
    })
}
pub(super) fn child_window(
    parent: &ArrayData,
    child: &ArrayData,
    window: Window,
) -> Result<Window> {
    if matches!(parent.data_type(), DataType::Struct(_)) {
        let shift = window
            .offset
            .checked_sub(parent.offset())
            .ok_or_else(invalid)?;
        Window::slice(child, shift, window.len)
    } else {
        Ok(Window::whole(child))
    }
}
pub(super) fn view_child(child: &ArrayData, window: Window) -> Result<Window> {
    if window.len == 0 {
        Window::slice(child, 0, 0)
    } else {
        Ok(Window::whole(child))
    }
}
pub(super) fn fixed_width(dt: &DataType) -> Option<usize> {
    match dt {
        DataType::Dictionary(key, _) => key.primitive_width(),
        DataType::FixedSizeBinary(width) => usize::try_from(*width).ok(),
        _ => dt.primitive_width(),
    }
}
pub(super) fn run_slice(
    data: &ArrayData,
    window: Window,
    check: Check<'_>,
) -> Result<Option<RunSlice>> {
    let ends = data.child_data().first().ok_or_else(invalid)?;
    let width = fixed_width(ends.data_type()).ok_or_else(invalid)?;
    let last = last_end(ends, width)?;
    if window.offset == 0 && last == window.len {
        return Ok(None);
    }
    run_bounds(ends, window, width, check).map(Some)
}
fn last_end(data: &ArrayData, width: usize) -> Result<usize> {
    if data.is_empty() {
        return Ok(0);
    }
    let index = data
        .offset()
        .checked_add(data.len() - 1)
        .ok_or_else(invalid)?;
    offset(data, index, width)
}
fn run_bounds(
    data: &ArrayData,
    window: Window,
    width: usize,
    check: Check<'_>,
) -> Result<RunSlice> {
    let first = first_run(data, window, width, check)?;
    let last = last_run(data, window, width, check)?;
    let count = last
        .checked_sub(first)
        .and_then(|n| n.checked_add(1))
        .ok_or_else(invalid)?;
    Ok(RunSlice {
        first,
        count,
        width,
    })
}
fn first_run(data: &ArrayData, window: Window, width: usize, check: Check<'_>) -> Result<usize> {
    if window.offset == 0 || window.len == 0 {
        return Ok(0);
    }
    physical(data, window.offset, width, check)
}
fn last_run(data: &ArrayData, window: Window, width: usize, check: Check<'_>) -> Result<usize> {
    if window.len == 0 {
        return Ok(0);
    }
    let target = window
        .offset
        .checked_add(window.len - 1)
        .ok_or_else(invalid)?;
    physical(data, target, width, check)
}
fn buffer_value(data: &ArrayData, index: usize, width: usize) -> Result<&[u8]> {
    let buffer = data.buffers().first().ok_or_else(invalid)?;
    let start = index.checked_mul(width).ok_or_else(invalid)?;
    let end = start.checked_add(width).ok_or_else(invalid)?;
    buffer.as_slice().get(start..end).ok_or_else(invalid)
}
fn bytes<const N: usize>(value: &[u8]) -> Result<[u8; N]> {
    value.try_into().map_err(|_| invalid())
}
fn offset(data: &ArrayData, index: usize, width: usize) -> Result<usize> {
    let raw = buffer_value(data, index, width)?;
    let value = match width {
        2 => bytes(raw).map(i16::from_ne_bytes).map(i64::from),
        4 => bytes(raw).map(i32::from_ne_bytes).map(i64::from),
        8 => bytes(raw).map(i64::from_ne_bytes),
        _ => return Err(invalid()),
    }?;
    usize::try_from(value).map_err(|_| invalid())
}
fn physical(data: &ArrayData, target: usize, width: usize, check: Check<'_>) -> Result<usize> {
    let (mut left, mut right) = (0, data.len());
    while left < right {
        check()?;
        let middle = left + (right - left) / 2;
        let index = data.offset().checked_add(middle).ok_or_else(invalid)?;
        if offset(data, index, width)? <= target {
            left = middle + 1;
        } else {
            right = middle;
        }
    }
    if left == data.len() {
        return Err(invalid());
    }
    Ok(left)
}
