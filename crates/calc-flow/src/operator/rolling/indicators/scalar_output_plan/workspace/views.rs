use super::super::{
    invalid,
    math::{add, vector},
};
use super::{Check, Result};
use datafusion::arrow::{
    array::{
        Array, BinaryViewArray, FixedSizeListArray, LargeListArray, LargeListViewArray, ListArray,
        ListViewArray, MapArray, StringViewArray, StructArray, UnionArray,
        downcast_dictionary_array, downcast_run_array,
    },
    buffer::Buffer,
    datatypes::DataType,
};

pub(super) fn metadata(array: &dyn Array, check: Check<'_>) -> Result<u64> {
    let fixed = super::super::super::scalar_materialization_plan::null_array_cost(
        array.data_type(),
        0,
        check,
    )?;
    add(fixed.construction_peak, walk(array, check)?)
}
fn walk(array: &dyn Array, check: Check<'_>) -> Result<u64> {
    check()?;
    match array.data_type() {
        DataType::Utf8View => buffers(typed::<StringViewArray>(array)?.data_buffers().len()),
        DataType::BinaryView => buffers(typed::<BinaryViewArray>(array)?.data_buffers().len()),
        _ => composite(array, check),
    }
}
fn buffers(count: usize) -> Result<u64> {
    vector::<Buffer>(
        count
            .checked_add(1)
            .ok_or_else(|| invalid("view buffer count"))?,
    )
}

fn composite(array: &dyn Array, check: Check<'_>) -> Result<u64> {
    match array.data_type() {
        DataType::Struct(_) => struct_array(array, check),
        DataType::Map(..) => map(array, check),
        DataType::Union(..) => union_array(array, check),
        DataType::Dictionary(..) => dictionary(array, check),
        DataType::RunEndEncoded(..) => run(array, check),
        _ => lists(array, check),
    }
}
fn lists(array: &dyn Array, check: Check<'_>) -> Result<u64> {
    let child = match array.data_type() {
        DataType::List(_) => typed::<ListArray>(array)?.values(),
        DataType::LargeList(_) => typed::<LargeListArray>(array)?.values(),
        DataType::FixedSizeList(..) => typed::<FixedSizeListArray>(array)?.values(),
        _ => return list_views(array, check),
    };
    walk(child.as_ref(), check)
}
fn list_views(array: &dyn Array, check: Check<'_>) -> Result<u64> {
    let child = match array.data_type() {
        DataType::ListView(_) => typed::<ListViewArray>(array)?.values(),
        DataType::LargeListView(_) => typed::<LargeListViewArray>(array)?.values(),
        _ => return Ok(0),
    };
    walk(child.as_ref(), check)
}
fn struct_array(array: &dyn Array, check: Check<'_>) -> Result<u64> {
    structure(typed::<StructArray>(array)?, check)
}
fn map(array: &dyn Array, check: Check<'_>) -> Result<u64> {
    walk(typed::<MapArray>(array)?.entries(), check)
}
fn union_array(array: &dyn Array, check: Check<'_>) -> Result<u64> {
    union(typed::<UnionArray>(array)?, check)
}

fn structure(array: &StructArray, check: Check<'_>) -> Result<u64> {
    let mut total = 0;
    for child in array.columns() {
        total = add(total, walk(child.as_ref(), check)?)?;
    }
    Ok(total)
}
fn union(array: &UnionArray, check: Check<'_>) -> Result<u64> {
    let DataType::Union(fields, _) = array.data_type() else {
        return Err(invalid("union type"));
    };
    let mut total = 0;
    for (id, _) in fields.iter() {
        total = add(total, walk(array.child(id).as_ref(), check)?)?;
    }
    Ok(total)
}
fn dictionary(array: &dyn Array, check: Check<'_>) -> Result<u64> {
    downcast_dictionary_array!(array=>walk(array.values().as_ref(),check),_=>Err(invalid("dictionary key type")))
}
fn run(array: &dyn Array, check: Check<'_>) -> Result<u64> {
    downcast_run_array!(array=>walk(array.values().as_ref(),check),_=>Err(invalid("run index type")))
}
fn typed<T: Array + 'static>(array: &dyn Array) -> Result<&T> {
    array
        .as_any()
        .downcast_ref()
        .ok_or_else(|| invalid("array representation"))
}

pub(super) fn descriptor_tree(array: &dyn Array, check: Check<'_>) -> Result<u64> {
    let fixed = super::super::super::scalar_materialization_plan::null_array_cost(
        array.data_type(),
        0,
        check,
    )?;
    add(fixed.logical_owned, walk(array, check)?)
}
