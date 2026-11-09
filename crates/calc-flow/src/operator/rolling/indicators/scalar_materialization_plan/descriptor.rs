use super::cost::{arc, arc_slice, times};
use super::{Check, Result, add, invalid};
use datafusion::arrow::{
    array::{
        ArrayData, ArrayRef, BooleanArray, BufferSpec, DictionaryArray, FixedSizeBinaryArray,
        FixedSizeListArray, Int8Array, LargeListArray, LargeListViewArray, LargeStringArray,
        ListArray, ListViewArray, MapArray, NullArray, RunArray, StringArray, StringViewArray,
        StructArray, UnionArray,
    },
    buffer::Buffer,
    datatypes::DataType as D,
};

pub(super) fn dtype(dtype: &D, check: Check<'_>) -> Result<u64> {
    check()?;
    if let D::Dictionary(key, value) = dtype {
        let children = add(self::dtype(key, check)?, self::dtype(value, check)?)?;
        return add(times::<D>(2)?, children);
    }
    Ok(0)
}
pub(super) fn owner(dtype: &D) -> Result<u64> {
    match dtype {
        D::List(_) => arc::<ListArray>(),
        D::LargeList(_) => arc::<LargeListArray>(),
        D::ListView(_) => arc::<ListViewArray>(),
        D::LargeListView(_) => arc::<LargeListViewArray>(),
        D::FixedSizeList(..) => arc::<FixedSizeListArray>(),
        D::Struct(_) => arc::<StructArray>(),
        _ => other_owner(dtype),
    }
}
fn other_owner(dtype: &D) -> Result<u64> {
    match dtype {
        D::Null => arc::<NullArray>(),
        D::Boolean => arc::<BooleanArray>(),
        D::Map(..) => arc::<MapArray>(),
        D::Union(..) => arc::<UnionArray>(),
        D::Dictionary(..) => arc::<DictionaryArray<datafusion::arrow::datatypes::Int8Type>>(),
        D::RunEndEncoded(..) => arc::<RunArray<datafusion::arrow::datatypes::Int16Type>>(),
        _ => byte_owner(dtype),
    }
}
fn byte_owner(dtype: &D) -> Result<u64> {
    match dtype {
        D::Utf8 | D::Binary => arc::<StringArray>(),
        D::LargeUtf8 | D::LargeBinary => arc::<LargeStringArray>(),
        D::Utf8View | D::BinaryView => arc::<StringViewArray>(),
        D::FixedSizeBinary(_) => arc::<FixedSizeBinaryArray>(),
        dtype if dtype.is_primitive() => arc::<Int8Array>(),
        _ => Err(invalid()),
    }
}
pub(super) fn buffer_count(dtype: &D) -> usize {
    match dtype {
        D::Null | D::Struct(_) | D::FixedSizeList(..) | D::RunEndEncoded(..) => 0,
        D::Utf8
        | D::LargeUtf8
        | D::Binary
        | D::LargeBinary
        | D::ListView(_)
        | D::LargeListView(_)
        | D::Union(_, datafusion::arrow::datatypes::UnionMode::Dense) => 2,
        _ => 1,
    }
}
pub(super) fn vector(dtype: &D, children: usize) -> Result<u64> {
    match dtype {
        D::Struct(_) => Ok(times::<ArrayData>(children)?.max(times::<ArrayRef>(children)?)),
        D::Union(fields, _) => {
            let slots = fields.iter().try_fold(1, next_slot)?;
            times::<Option<ArrayRef>>(slots)
        }
        _ => Ok(0),
    }
}
pub(super) fn views(buffers: usize) -> Result<u64> {
    arc_slice::<Buffer>(buffers)
}
pub(super) fn metadata(
    dtype: &D,
    buffers: usize,
    children: usize,
    check: Check<'_>,
) -> Result<u64> {
    let vectors = add(times::<Buffer>(buffers)?, times::<ArrayData>(children)?)?;
    add(vectors, self::dtype(dtype, check)?)
}
pub(super) fn layout(dtype: &D) -> Result<u64> {
    let count = if matches!(dtype, D::Utf8View | D::BinaryView) {
        1
    } else {
        buffer_count(dtype)
    };
    times::<BufferSpec>(count)
}

fn next_slot(
    slots: usize,
    (id, _): (i8, &datafusion::arrow::datatypes::FieldRef),
) -> Result<usize> {
    let id = usize::try_from(id).map_err(|_| invalid())?;
    Ok(slots.max(id.checked_add(1).ok_or_else(invalid)?))
}
pub(super) fn owned(dtype: &D, check: Check<'_>) -> Result<u64> {
    add(owner(dtype)?, self::dtype(dtype, check)?)
}

pub(super) fn view_to_data_capacity(buffers: usize) -> Result<usize> {
    // Pinned byte-view to_data clones N buffers, then inserts its views buffer at zero.
    Ok(buffers.checked_mul(2).ok_or_else(invalid)?.max(4))
}
pub(super) fn view_conversion_extra(buffers: usize) -> Result<u64> {
    let data = buffers.checked_sub(1).ok_or_else(invalid)?;
    let capacity = view_to_data_capacity(data)?;
    let extra = capacity.checked_sub(buffers).ok_or_else(invalid)?;
    times::<Buffer>(extra)
}
