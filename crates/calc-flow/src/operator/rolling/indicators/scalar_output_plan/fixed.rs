use super::{
    Check, OutputPlan, Result, invalid,
    math::{
        add, align, bitmap, buffer_headers, bytes, mul, physical_width, scale, size, sum, vector,
    },
    workspace,
};
use datafusion::{
    arrow::{
        array::{Array, ArrayData, ArrayRef, FixedSizeListArray, MutableArrayData},
        datatypes::DataType,
    },
    common::ScalarValue,
};
#[path = "fixed/bytes.rs"]
mod bytes;
#[path = "fixed/mutable.rs"]
mod mutable;
pub(super) use mutable::{buffers::Growing, scalar_union};
#[path = "fixed/views.rs"]
mod views;
type Cell<'a, 'b> = dyn FnMut(usize) -> Option<&'a ScalarValue> + 'b;
#[derive(Clone, Copy)]
struct Geometry {
    slots: u64,
    values: u64,
    child_capacity: u64,
    buffers: u64,
    descriptors: u64,
    workspace: u64,
    builders: u64,
}

pub(super) fn plan(
    dtype: &DataType,
    rows: usize,
    cell: &mut Cell<'_, '_>,
    check: Check<'_>,
) -> Result<Option<OutputPlan>> {
    let Some(geometry) = shape(dtype, rows, check)? else {
        return Ok(None);
    };
    build(dtype, rows, geometry, cell, check).map(Some)
}
fn shape(dtype: &DataType, rows: usize, check: Check<'_>) -> Result<Option<Geometry>> {
    let DataType::FixedSizeList(field, width) = dtype else {
        return Ok(None);
    };
    if rows < 2 {
        return Ok(None);
    }
    geometry(field.data_type(), *width, rows, check).map(Some)
}
fn build(
    dtype: &DataType,
    rows: usize,
    geometry: Geometry,
    cell: &mut Cell<'_, '_>,
    check: Check<'_>,
) -> Result<OutputPlan> {
    let repair = effective(dtype, rows, cell, check)?;
    let repaired = inputs(dtype, rows, repair, cell, check)?;
    let geometry = complete_geometry(dtype, rows, repair, cell, geometry, check)?;
    let output = output(dtype, rows, geometry, check)?;
    let scratch = scratch(dtype, rows, geometry, repaired, check)?;
    Ok(OutputPlan {
        output_bytes: output,
        scratch_bytes: scratch,
    })
}
fn complete_geometry(
    dtype: &DataType,
    rows: usize,
    repair: bool,
    cell: &mut Cell<'_, '_>,
    geometry: Geometry,
    check: Check<'_>,
) -> Result<Geometry> {
    let geometry = bytes::complete(dtype, rows, repair, cell, geometry, check)?;
    let geometry = views::complete(dtype, rows, repair, cell, geometry, check)?;
    mutable::complete(dtype, rows, repair, cell, geometry, check)
}
fn output(dtype: &DataType, rows: usize, geometry: Geometry, check: Check<'_>) -> Result<u64> {
    let empty = super::super::scalar_materialization_plan::null_array_cost(dtype, 0, check)?;
    sum([
        Ok(empty.logical_owned),
        Ok(geometry.descriptors),
        Ok(geometry.values),
        bitmap(size(rows)?),
        bitmap(geometry.child_capacity),
    ])
}

fn geometry(dtype: &DataType, width: i32, rows: usize, check: Check<'_>) -> Result<Geometry> {
    let width = u64::try_from(width).map_err(|_| invalid("fixed list width"))?;
    let slots = mul(size(rows)?, width)?;
    child_geometry(dtype, slots, check)
}
fn child_geometry(dtype: &DataType, slots: u64, _check: Check<'_>) -> Result<Geometry> {
    if mutable::supports(dtype) {
        return Ok(mutable::initial(slots));
    }
    match dtype {
        DataType::Utf8View | DataType::BinaryView => views::initial(slots),
        DataType::Null => Ok(Geometry {
            slots,
            values: 0,
            child_capacity: slots,
            buffers: 2,
            descriptors: 0,
            workspace: 0,
            builders: 2,
        }),
        DataType::Boolean => Ok(Geometry {
            slots,
            values: buffer(add(slots, 7)? / 8)?,
            child_capacity: slots,
            buffers: 3,
            descriptors: 0,
            workspace: 0,
            builders: 2,
        }),
        _ => primitive_geometry(dtype, slots),
    }
}
fn primitive_geometry(dtype: &DataType, slots: u64) -> Result<Geometry> {
    let Some(unit) = physical_width(dtype)? else {
        return bytes::initial(dtype, slots);
    };
    let values = buffer(mul(slots, unit)?)?;
    Ok(Geometry {
        slots,
        values,
        child_capacity: slots,
        buffers: 3,
        descriptors: 0,
        workspace: 0,
        builders: 2,
    })
}

fn buffer(bytes: u64) -> Result<u64> {
    let aligned = align(bytes)?;
    let n = usize::try_from(aligned).map_err(|_| invalid("buffer layout"))?;
    std::alloc::Layout::from_size_align(n, datafusion::arrow::alloc::ALIGNMENT)
        .map_err(|_| invalid("buffer layout"))?;
    Ok(aligned)
}
fn effective(
    dtype: &DataType,
    rows: usize,
    cell: &mut Cell<'_, '_>,
    check: Check<'_>,
) -> Result<bool> {
    for row in 0..rows {
        check()?;
        if let Some(array) = array(cell(row))?
            && !array.is_null(0)
        {
            if array.data_type() != dtype {
                return Err(invalid("fixed list inferred datatype"));
            }
            return Ok(true);
        }
    }
    Ok(false)
}
fn array(value: Option<&ScalarValue>) -> Result<Option<&FixedSizeListArray>> {
    match value {
        Some(ScalarValue::FixedSizeList(array)) if array.len() == 1 => Ok(Some(array.as_ref())),
        None => Ok(None),
        _ => Err(invalid("fixed list scalar shape")),
    }
}
fn inputs(
    dtype: &DataType,
    rows: usize,
    repair: bool,
    cell: &mut Cell<'_, '_>,
    check: Check<'_>,
) -> Result<u64> {
    let mut total = 0;
    for row in 0..rows {
        check()?;
        total = add(total, input(dtype, repair, cell(row), check)?)?;
    }
    Ok(total)
}
fn input(
    dtype: &DataType,
    repair: bool,
    value: Option<&ScalarValue>,
    check: Check<'_>,
) -> Result<u64> {
    let array = array(value)?;
    if repair && array.is_none_or(|array| array.is_null(0)) {
        return replacement(dtype, check);
    }
    match array {
        Some(array) if array.data_type() == dtype => workspace::array_data_workspace(array, check),
        None => super::super::scalar_materialization_plan::null_array_requirement(dtype, 0, check),
        _ => Err(invalid("fixed list inferred datatype")),
    }
}
fn replacement(dtype: &DataType, check: Check<'_>) -> Result<u64> {
    let replacement = super::super::scalar_materialization_plan::null_array_cost(dtype, 1, check)?;
    let meta = super::super::scalar_materialization_plan::null_array_cost(dtype, 0, check)?;
    add(replacement.construction_peak, meta.construction_peak)
}
fn scratch(
    dtype: &DataType,
    rows: usize,
    geometry: Geometry,
    inputs: u64,
    check: Check<'_>,
) -> Result<u64> {
    let fixed = super::super::scalar_materialization_plan::null_array_cost(dtype, 0, check)?;
    let metadata = input_vectors(rows)?;
    let builders = mul(builder(rows)?, geometry.builders)?;
    sum([
        Ok(inputs),
        Ok(geometry.workspace),
        Ok(fixed.construction_peak),
        Ok(metadata),
        Ok(builders),
        Ok(geometry.values),
        scale(bitmap(geometry.child_capacity), 2),
        scale(bitmap(size(rows)?), 2),
        buffer_headers(geometry.buffers),
    ])
}
fn input_vectors(rows: usize) -> Result<u64> {
    sum([
        bytes::<ArrayData>(size(rows)?),
        vector::<ArrayRef>(rows),
        vector::<&dyn Array>(rows),
    ])
}
fn builder(rows: usize) -> Result<u64> {
    let vectors = sum([
        vector::<&ArrayData>(rows),
        scale(vector::<Box<dyn Fn()>>(rows), 2),
    ])?;
    // Dense-union values capture five words; null-bit closures capture three.
    let captures = bytes::<([usize; 5], [usize; 3])>(size(rows)?)?;
    sum([
        Ok(vectors),
        Ok(captures),
        bytes::<usize>(1),
        bytes::<MutableArrayData<'_>>(1),
    ])
}

pub(super) fn advance_view_offset(offset: u32, count: u64) -> Result<u32> {
    views::advance(offset, count)
}

pub(super) fn normal(
    dtype: &DataType,
    count: usize,
    rows: u64,
    at: &mut super::concatenated::At<'_>,
    check: Check<'_>,
) -> Result<super::concatenated::Cost> {
    let geometry = mutable::normal(dtype, count, rows, at, check)?;
    let inputs = normal_metadata(dtype, count, at, check)?;
    Ok(super::concatenated::Cost {
        owned: add(geometry.values, geometry.descriptors)?,
        temporary: scratch(dtype, count, geometry, inputs, check)?,
    })
}
fn normal_metadata(
    dtype: &DataType,
    count: usize,
    at: &mut super::concatenated::At<'_>,
    check: Check<'_>,
) -> Result<u64> {
    let mut total = 0;
    for index in 0..count {
        check()?;
        total = add(total, piece_metadata(dtype, at(index)?, check)?)?;
    }
    Ok(total)
}
fn piece_metadata(
    dtype: &DataType,
    piece: super::concatenated::Piece<'_>,
    check: Check<'_>,
) -> Result<u64> {
    match piece.array {
        Some(array) => add(
            workspace::array_data_workspace(array, check)?,
            super::single::sliced(array, check)?,
        ),
        None => super::super::scalar_materialization_plan::null_array_requirement(
            dtype, piece.len, check,
        ),
    }
}

#[cfg(test)]
mod tests {
    #[test]
    fn test_packed_buffer_uses_the_pinned_arrow_allocation_alignment() {
        let alignment = datafusion::arrow::alloc::ALIGNMENT;
        if alignment <= 64 {
            return;
        }
        let bytes = usize::try_from(isize::MAX).unwrap() - 63;
        assert!(std::alloc::Layout::from_size_align(bytes, 64).is_ok());
        assert!(std::alloc::Layout::from_size_align(bytes, alignment).is_err());
        assert!(super::buffer(u64::try_from(bytes).unwrap()).is_err());
    }
}
