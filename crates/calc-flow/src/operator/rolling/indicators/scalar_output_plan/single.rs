use super::{
    Check, OutputPlan, Result, invalid,
    math::{add, bytes, grow, size, sum},
};
use datafusion::{
    arrow::{
        array::{Array, ArrayRef, FixedSizeListArray, StructArray, UnionArray},
        datatypes::{DataType, UnionMode},
    },
    common::ScalarValue,
};

pub(super) fn array(value: &ScalarValue) -> Option<&dyn Array> {
    match value {
        ScalarValue::List(array) => Some(array.as_ref()),
        ScalarValue::LargeList(array) => Some(array.as_ref()),
        ScalarValue::ListView(array) => Some(array.as_ref()),
        ScalarValue::LargeListView(array) => Some(array.as_ref()),
        ScalarValue::FixedSizeList(array) => Some(array.as_ref()),
        _ => composite(value),
    }
}
fn composite(value: &ScalarValue) -> Option<&dyn Array> {
    match value {
        ScalarValue::Struct(array) => Some(array.as_ref()),
        ScalarValue::Map(array) => Some(array.as_ref()),
        _ => None,
    }
}
pub(super) fn plan(dtype: &DataType, array: &dyn Array, check: Check<'_>) -> Result<OutputPlan> {
    check()?;
    if array.len() != 1 || array.data_type() != dtype {
        return Err(invalid("nested scalar shape"));
    }
    let output = sliced(array, check)?;
    let scratch = temporary_vectors()?;
    Ok(OutputPlan {
        output_bytes: output,
        scratch_bytes: scratch,
    })
}
fn temporary_vectors() -> Result<u64> {
    sum([
        bytes::<ScalarValue>(1),
        grow(1, size(size_of::<ArrayRef>())?),
        grow(1, size(size_of::<&dyn Array>())?),
    ])
}

pub(super) fn sliced(array: &dyn Array, check: Check<'_>) -> Result<u64> {
    check()?;
    let own = super::super::scalar_materialization_plan::descriptor_bytes(array, check)?;
    add(own, sliced_children(array, check)?)
}
fn sliced_children(array: &dyn Array, check: Check<'_>) -> Result<u64> {
    match array.data_type() {
        DataType::Struct(_) => structure(typed::<StructArray>(array)?, check),
        DataType::FixedSizeList(..) => fixed(array, check),
        DataType::Union(_, UnionMode::Sparse) => union(typed::<UnionArray>(array)?, check),
        _ => Ok(0),
    }
}
fn fixed(array: &dyn Array, check: Check<'_>) -> Result<u64> {
    sliced(typed::<FixedSizeListArray>(array)?.values().as_ref(), check)
}

fn structure(array: &StructArray, check: Check<'_>) -> Result<u64> {
    let mut cost = 0;
    for child in array.columns() {
        cost = add(cost, sliced(child.as_ref(), check)?)?;
    }
    Ok(cost)
}
fn union(array: &UnionArray, check: Check<'_>) -> Result<u64> {
    let DataType::Union(fields, _) = array.data_type() else {
        return Err(invalid("union type"));
    };
    let mut cost = 0;
    for (id, _) in fields.iter() {
        cost = add(cost, sliced(array.child(id).as_ref(), check)?)?;
    }
    Ok(cost)
}
fn typed<T: Array + 'static>(array: &dyn Array) -> Result<&T> {
    array
        .as_any()
        .downcast_ref()
        .ok_or_else(|| invalid("array representation"))
}
