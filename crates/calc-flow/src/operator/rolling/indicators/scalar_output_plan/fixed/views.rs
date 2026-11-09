use super::super::{
    invalid,
    math::{add, arc_slice, mul, size, vector},
};
use super::{Cell, Check, Geometry, Result, array, buffer};
use datafusion::arrow::{
    array::{Array, GenericByteViewArray},
    buffer::Buffer,
    datatypes::{BinaryViewType, ByteViewType, DataType, StringViewType},
};

pub(super) fn initial(slots: u64) -> Result<Geometry> {
    Ok(Geometry {
        slots,
        values: buffer(mul(slots, 16)?)?,
        child_capacity: slots,
        buffers: 3,
        descriptors: 0,
        workspace: 0,
        builders: 2,
    })
}
pub(super) fn complete(
    dtype: &DataType,
    rows: usize,
    repair: bool,
    cell: &mut Cell<'_, '_>,
    base: Geometry,
    check: Check<'_>,
) -> Result<Geometry> {
    let DataType::FixedSizeList(field, _) = dtype else {
        return Err(invalid("fixed list datatype"));
    };
    match field.data_type() {
        DataType::Utf8View => scan::<StringViewType>(rows, repair, cell, base, check),
        DataType::BinaryView => scan::<BinaryViewType>(rows, repair, cell, base, check),
        _ => Ok(base),
    }
}
fn scan<T: ByteViewType>(
    rows: usize,
    repair: bool,
    cell: &mut Cell<'_, '_>,
    base: Geometry,
    check: Check<'_>,
) -> Result<Geometry> {
    let mut blocks = 0;
    let mut offset = 0;
    for row in 0..rows {
        check()?;
        let count = input::<T>(repair, cell(row))?;
        offset = advance(offset, count)?;
        blocks = add(blocks, count)?;
    }
    descriptor_geometry(base, blocks)
}
fn input<T: ByteViewType>(
    repair: bool,
    value: Option<&datafusion::common::ScalarValue>,
) -> Result<u64> {
    let Some(array) = array(value)? else {
        return Ok(0);
    };
    if repair && array.is_null(0) {
        return Ok(0);
    }
    let values = array
        .values()
        .as_any()
        .downcast_ref::<GenericByteViewArray<T>>()
        .ok_or_else(|| invalid("fixed view child"))?;
    size(values.data_buffers().len())
}
pub(super) fn advance(offset: u32, count: u64) -> Result<u32> {
    let count =
        u32::try_from(count & u64::from(u32::MAX)).map_err(|_| invalid("view buffer count"))?;
    offset
        .checked_add(count)
        .ok_or_else(|| invalid("view buffer index offset"))
}
fn descriptor_geometry(base: Geometry, blocks: u64) -> Result<Geometry> {
    let blocks = usize::try_from(blocks).map_err(|_| invalid("view buffer count"))?;
    let descriptors = arc_slice::<Buffer>(blocks)?;
    let workspace = vector::<Buffer>(
        blocks
            .checked_add(1)
            .ok_or_else(|| invalid("view buffer count"))?,
    )?;
    Ok(Geometry {
        descriptors,
        workspace,
        ..base
    })
}

#[cfg(test)]
mod tests {
    #[test]
    fn test_variadic_offset_matches_the_pinned_cast_and_checked_add() {
        assert!(super::advance(u32::MAX, 1).is_err());
        assert_eq!(super::advance(0, u64::from(u32::MAX)).unwrap(), u32::MAX);
        assert_eq!(super::advance(0, u64::from(u32::MAX) + 1).unwrap(), 0);
    }
}
