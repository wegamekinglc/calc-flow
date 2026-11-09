use super::super::math::{add, bitmap, buffer_headers, retained, size, sum, vector};
use super::{Check, Result, SingleCtorCost, invalid, null_array, plan, shape};
use datafusion::{
    arrow::{
        array::{ArrayData, ArrayRef},
        datatypes::{DataType, FieldRef, UnionFields, UnionMode},
    },
    common::ScalarValue,
};

pub(super) fn dictionary(
    expected: &DataType,
    key: &DataType,
    value: &ScalarValue,
    check: Check<'_>,
) -> Result<SingleCtorCost> {
    let child = plan(dictionary_child(expected, key)?, value, check)?;
    // The factory's empty values are extra allowance; the actual scalar values have length one.
    child
        .combine(null_array(expected, 1, check)?)?
        .combine(key_mask(value)?)
}

fn dictionary_child<'a>(expected: &'a DataType, key: &DataType) -> Result<&'a DataType> {
    let DataType::Dictionary(actual_key, child) = expected else {
        return Err(invalid("single dictionary type"));
    };
    if actual_key.as_ref() != key || !key.is_dictionary_key_type() {
        return Err(invalid("single dictionary key"));
    }
    Ok(child)
}

fn key_mask(value: &ScalarValue) -> Result<SingleCtorCost> {
    // PrimitiveArray's Option iterator allocates a 64-rounded validity buffer, unlike new_null.
    let mask = bitmap(1)?;
    Ok(SingleCtorCost {
        logical_owned: if value.is_null() { mask } else { 0 },
        construction_peak: add(mask, buffer_headers(1)?)?,
    })
}

pub(super) fn run(
    expected: &DataType,
    ends: &FieldRef,
    field: &FieldRef,
    value: &ScalarValue,
    check: Check<'_>,
) -> Result<SingleCtorCost> {
    validate_run_fields(expected, ends, field)?;
    let child = plan(field.data_type(), value, check)?;
    let rebuilt = shape::rebuild(field.data_type(), value, check)?;
    let root = run_storage(expected, end_width(ends.data_type())?, check)?;
    child.combine(rebuilt)?.combine(root)
}

fn validate_run_fields(expected: &DataType, ends: &FieldRef, field: &FieldRef) -> Result<()> {
    if !matches!(expected, DataType::RunEndEncoded(a, b) if a == ends && b == field) {
        return Err(invalid("single run fields"));
    }
    Ok(())
}

fn run_storage(expected: &DataType, width: u64, check: Check<'_>) -> Result<SingleCtorCost> {
    let root = null_array(expected, 0, check)?;
    let data = retained(1, width)?;
    Ok(SingleCtorCost {
        logical_owned: add(root.logical_owned, data)?,
        construction_peak: sum([Ok(root.construction_peak), Ok(data), run_vectors(width)])?,
    })
}

fn run_vectors(width: u64) -> Result<u64> {
    sum([
        vector::<ArrayData>(2),
        vector::<datafusion::arrow::buffer::Buffer>(1),
        buffer_headers(1),
        Ok(width),
    ])
}

fn end_width(dtype: &DataType) -> Result<u64> {
    match dtype {
        DataType::Int16 => Ok(2),
        DataType::Int32 => Ok(4),
        DataType::Int64 => Ok(8),
        _ => Err(invalid("single run index")),
    }
}

pub(super) fn union(
    expected: &DataType,
    value: Option<&(i8, Box<ScalarValue>)>,
    fields: &UnionFields,
    mode: UnionMode,
    check: Check<'_>,
) -> Result<SingleCtorCost> {
    if !matches!(expected, DataType::Union(actual_fields, actual_mode) if actual_fields == fields && *actual_mode == mode)
    {
        return Err(invalid("single union fields"));
    }
    let Some((id, value)) = value else {
        return null_array(expected, 1, check);
    };
    if !fields.iter().any(|(key, _)| key == *id) {
        return Err(invalid("single union type ID"));
    }
    union_children(expected, *id, value, fields, mode, check)
}

fn union_children(
    expected: &DataType,
    id: i8,
    value: &ScalarValue,
    fields: &UnionFields,
    mode: UnionMode,
    check: Check<'_>,
) -> Result<SingleCtorCost> {
    let mut children = SingleCtorCost::default();
    for (key, field) in fields.iter() {
        check()?;
        let cost = union_child(field.data_type(), value, key == id, mode, check)?;
        children = children.combine(cost)?;
    }
    children.combine(union_storage(expected, fields, mode, check)?)
}

fn union_child(
    dtype: &DataType,
    value: &ScalarValue,
    active: bool,
    mode: UnionMode,
    check: Check<'_>,
) -> Result<SingleCtorCost> {
    if active {
        return plan(dtype, value, check)?.combine(shape::rebuild(dtype, value, check)?);
    }
    null_array(dtype, usize::from(mode == UnionMode::Sparse), check)
}

fn union_storage(
    expected: &DataType,
    fields: &UnionFields,
    mode: UnionMode,
    check: Check<'_>,
) -> Result<SingleCtorCost> {
    let root = null_array(expected, 0, check)?;
    let data = add(1, if mode == UnionMode::Dense { 4 } else { 0 })?;
    let scratch = union_vectors(fields)?;
    Ok(SingleCtorCost {
        logical_owned: add(root.logical_owned, data)?,
        construction_peak: sum([Ok(root.construction_peak), Ok(data), Ok(scratch)])?,
    })
}

fn union_vectors(fields: &UnionFields) -> Result<u64> {
    let mut slots = 0;
    for (id, _) in fields.iter() {
        let index = usize::try_from(id).map_err(|_| invalid("single union type ID"))?;
        slots = slots.max(
            index
                .checked_add(1)
                .ok_or_else(|| invalid("single union slots"))?,
        );
    }
    sum([
        vector::<ArrayRef>(fields.len()),
        vector::<ArrayData>(fields.len()),
        super::super::math::bytes::<i32>(size(slots)?),
        buffer_headers(2),
    ])
}
