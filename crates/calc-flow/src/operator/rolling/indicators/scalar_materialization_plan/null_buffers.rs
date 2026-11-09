use super::super::cost::{bits, buffer as constructor_buffer, product};
use super::super::{Result, add, invalid, size};
use datafusion::arrow::datatypes::{DataType as D, UnionMode};

pub(super) fn nullable(dtype: &D) -> bool {
    !matches!(dtype, D::Null | D::Union(..) | D::RunEndEncoded(..))
}
pub(super) fn raw(dtype: &D, len: usize) -> Result<BufferCost> {
    let values = buffers(dtype, len)?;
    if len > 0 && nullable(dtype) {
        values.combine(allocated(bits(len))?)
    } else {
        Ok(values)
    }
}
fn buffers(dtype: &D, len: usize) -> Result<BufferCost> {
    match dtype {
        D::Null | D::Struct(_) | D::FixedSizeList(..) | D::RunEndEncoded(..) => {
            Ok(BufferCost::default())
        }
        D::Boolean => allocated(bits(len)),
        D::Utf8 | D::Binary => variable(len, 4),
        D::LargeUtf8 | D::LargeBinary => variable(len, 8),
        _ => lists(dtype, len),
    }
}
fn lists(dtype: &D, len: usize) -> Result<BufferCost> {
    match dtype {
        D::List(_) | D::Map(..) => allocated(offset_bytes(len, 4)?),
        D::LargeList(_) => allocated(offset_bytes(len, 8)?),
        D::ListView(_) => repeated(len, 4),
        D::LargeListView(_) => repeated(len, 8),
        _ => encoded(dtype, len),
    }
}
fn encoded(dtype: &D, len: usize) -> Result<BufferCost> {
    match dtype {
        D::Utf8View | D::BinaryView => allocated(product(len, 16)?),
        D::Union(_, UnionMode::Sparse) => allocated(len),
        D::Union(_, UnionMode::Dense) => pair_cost(len, product(len, 4)?),
        D::Dictionary(key, _) => primitive(key, len),
        _ => primitive(dtype, len),
    }
}
fn primitive(dtype: &D, len: usize) -> Result<BufferCost> {
    allocated(product(len, width(dtype)?)?)
}
fn repeated(len: usize, width: usize) -> Result<BufferCost> {
    let bytes = product(len, width)?;
    pair_cost(bytes, bytes)
}
pub(super) fn pair(left: usize, right: usize) -> Result<u64> {
    pair_cost(left, right).map(|cost| cost.constructor_owned)
}
fn pair_cost(left: usize, right: usize) -> Result<BufferCost> {
    allocated(left)?.combine(allocated(right)?)
}
fn offset_bytes(len: usize, width: usize) -> Result<usize> {
    product(len.checked_add(1).ok_or_else(invalid)?, width)
}
fn width(dtype: &D) -> Result<usize> {
    if let D::FixedSizeBinary(size) = dtype {
        return usize::try_from(*size).map_err(|_| invalid());
    }
    dtype.primitive_width().ok_or_else(invalid)
}

fn variable(len: usize, width: usize) -> Result<BufferCost> {
    pair_cost(offset_bytes(len, width)?, 0)
}

#[derive(Clone, Copy, Default)]
pub(super) struct BufferCost {
    pub(super) logical_owned: u64,
    pub(super) constructor_owned: u64,
}
impl BufferCost {
    fn combine(self, other: Self) -> Result<Self> {
        Ok(Self {
            logical_owned: add(self.logical_owned, other.logical_owned)?,
            constructor_owned: add(self.constructor_owned, other.constructor_owned)?,
        })
    }
}
fn allocated(bytes: usize) -> Result<BufferCost> {
    Ok(BufferCost {
        logical_owned: size(bytes)?,
        constructor_owned: constructor_buffer(bytes)?,
    })
}
pub(super) fn run_ends(len: usize) -> Result<BufferCost> {
    // Pinned new_null uses from_slice_ref for its single nonempty run end.
    allocated(if len == 0 { 0 } else { 64 })
}
