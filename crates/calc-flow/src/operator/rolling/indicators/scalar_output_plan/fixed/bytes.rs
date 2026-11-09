use super::super::{
    invalid,
    math::{add, mul, size},
};
use super::{Cell, Check, Geometry, Result, array, buffer};
use datafusion::arrow::{
    array::GenericByteArray,
    datatypes::{
        ArrowNativeType, BinaryType, ByteArrayType, DataType, LargeBinaryType, LargeUtf8Type,
        Utf8Type,
    },
};

pub(super) fn initial(dtype: &DataType, slots: u64) -> Result<Geometry> {
    let width = offset_width(dtype)?;
    geometry(slots, slots, 0, width)
}
pub(super) fn complete(
    dtype: &DataType,
    rows: usize,
    repair: bool,
    cell: &mut Cell<'_, '_>,
    base: Geometry,
    check: Check<'_>,
) -> Result<Geometry> {
    let DataType::FixedSizeList(field, width) = dtype else {
        return Err(invalid("fixed list datatype"));
    };
    let width = u64::try_from(*width).map_err(|_| invalid("fixed list width"))?;
    match field.data_type() {
        DataType::Utf8 => scan::<Utf8Type>(rows, repair, cell, base, width, check),
        DataType::LargeUtf8 => scan::<LargeUtf8Type>(rows, repair, cell, base, width, check),
        DataType::Binary => scan::<BinaryType>(rows, repair, cell, base, width, check),
        DataType::LargeBinary => scan::<LargeBinaryType>(rows, repair, cell, base, width, check),
        _ => Ok(base),
    }
}
fn scan<T: ByteArrayType>(
    rows: usize,
    repair: bool,
    cell: &mut Cell<'_, '_>,
    base: Geometry,
    width: u64,
    check: Check<'_>,
) -> Result<Geometry> {
    let mut items = 0;
    let mut data = 0;
    for row in 0..rows {
        check()?;
        let (len, bytes) = input::<T>(repair, cell(row), width)?;
        items = add(items, len)?;
        data = add(data, bytes)?;
    }
    geometry(base.slots, items, data, size(size_of::<T::Offset>())?)
}
fn input<T: ByteArrayType>(
    repair: bool,
    value: Option<&datafusion::common::ScalarValue>,
    width: u64,
) -> Result<(u64, u64)> {
    let Some(array) = array(value)? else {
        return Ok((width, 0));
    };
    if repair && array.is_null(0) {
        return Ok((width, 0));
    }
    let child = array
        .values()
        .as_any()
        .downcast_ref::<GenericByteArray<T>>()
        .ok_or_else(|| invalid("fixed byte child"))?;
    lengths(child)
}
fn lengths<T: ByteArrayType>(array: &GenericByteArray<T>) -> Result<(u64, u64)> {
    let offsets = array.value_offsets();
    let first = offsets[0].as_usize();
    let last = offsets[array.len()].as_usize();
    Ok((
        size(array.len())?,
        size(
            last.checked_sub(first)
                .ok_or_else(|| invalid("fixed byte range"))?,
        )?,
    ))
}
fn geometry(slots: u64, items: u64, data: u64, width: u64) -> Result<Geometry> {
    let offsets = buffer(mul(add(items, 1)?, width)?)?;
    let values = buffer(data)?;
    Ok(Geometry {
        slots,
        values: add(offsets, values)?,
        child_capacity: items,
        buffers: 4,
        descriptors: 0,
        workspace: 0,
        builders: 2,
    })
}
fn offset_width(dtype: &DataType) -> Result<u64> {
    match dtype {
        DataType::Utf8 | DataType::Binary => Ok(4),
        DataType::LargeUtf8 | DataType::LargeBinary => Ok(8),
        _ => Err(invalid("fixed fallback datatype")),
    }
}
use datafusion::arrow::array::Array;
