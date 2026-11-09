use super::{
    Check, Result, capacity,
    math::{add, mul},
    single,
};
use datafusion::{
    arrow::datatypes::{DataType, Schema},
    common::ScalarValue,
};
#[path = "workspace/indices.rs"]
mod indices;
#[path = "workspace/views.rs"]
mod views;
use indices::plan as indices;

pub(crate) fn plan<'a>(
    schema: &Schema,
    rows: usize,
    cell: &mut dyn FnMut(usize, usize) -> Option<&'a ScalarValue>,
    check: Check<'_>,
) -> Result<u64> {
    check()?;
    capacity(schema, rows)?;
    let mut peak = 0;
    for (index, field) in schema.fields().iter().enumerate() {
        let cost = field_cost(field.data_type(), rows, &mut |row| cell(index, row), check)?;
        peak = peak.max(cost);
    }
    Ok(peak)
}
fn field_cost<'a>(
    dtype: &DataType,
    rows: usize,
    cell: &mut dyn FnMut(usize) -> Option<&'a ScalarValue>,
    check: Check<'_>,
) -> Result<u64> {
    if needs_comparison(dtype) || indices(dtype, rows, check)? != 0 {
        return column(dtype, rows, cell, check);
    }
    Ok(0)
}

pub(super) fn column<'a>(
    dtype: &DataType,
    rows: usize,
    cell: &mut dyn FnMut(usize) -> Option<&'a ScalarValue>,
    check: Check<'_>,
) -> Result<u64> {
    if !needs_comparison(dtype) {
        return indices(dtype, rows, check);
    }
    let comparison = comparison_workspace(dtype, rows, cell, check)?;
    let existing = add(comparison, indices(dtype, rows, check)?)?;
    add(
        existing,
        super::normal::dictionary::planning_workspace(dtype, rows, cell, check)?,
    )
}
fn comparison_workspace<'a>(
    dtype: &DataType,
    rows: usize,
    cell: &mut dyn FnMut(usize) -> Option<&'a ScalarValue>,
    check: Check<'_>,
) -> Result<u64> {
    let mut largest = 0;
    for row in 0..rows {
        check()?;
        let cost = cell_cost(dtype, cell(row), check)?;
        largest = largest.max(cost);
    }
    mul(largest, 2)
}

fn cell_cost(dtype: &DataType, value: Option<&ScalarValue>, check: Check<'_>) -> Result<u64> {
    match value {
        Some(value) => scalar_cost(dtype, value, check),
        None => hole(dtype, check),
    }
}
fn scalar_cost(dtype: &DataType, value: &ScalarValue, check: Check<'_>) -> Result<u64> {
    let metadata = needs_dictionary_metadata(dtype);
    let active = scalar(value, metadata, check)?;
    if metadata && fresh_union(value) {
        let empty = super::super::scalar_materialization_plan::null_array_cost(dtype, 0, check)?;
        return Ok(active.max(empty.construction_peak));
    }
    Ok(active)
}
fn fresh_union(value: &ScalarValue) -> bool {
    match value {
        ScalarValue::Union(..) => true,
        ScalarValue::Dictionary(_, value) | ScalarValue::RunEndEncoded(_, _, value) => {
            fresh_union(value)
        }
        _ => false,
    }
}

fn scalar(value: &ScalarValue, active: bool, check: Check<'_>) -> Result<u64> {
    check()?;
    if let Some(array) = single::array(value) {
        return if active || needs_dictionary_metadata(array.data_type()) {
            array_data_workspace(array, check)
        } else {
            Ok(0)
        };
    }
    match value {
        ScalarValue::RunEndEncoded(_, _, value) => scalar(value, true, check),
        ScalarValue::Dictionary(_, value) | ScalarValue::Union(Some((_, value)), _, _) => {
            scalar(value, active, check)
        }
        _ => Ok(0),
    }
}
fn hole(dtype: &DataType, check: Check<'_>) -> Result<u64> {
    let scalar = super::super::scalar_materialization_plan::null_requirement(dtype, check)?;
    let arrays = super::super::scalar_materialization_plan::null_array_cost(dtype, 0, check)?;
    add(scalar, arrays.construction_peak)
}
fn needs_comparison(dtype: &DataType) -> bool {
    matches!(dtype, DataType::RunEndEncoded(..)) || needs_dictionary_metadata(dtype)
}
fn needs_dictionary_metadata(dtype: &DataType) -> bool {
    match dtype {
        DataType::Dictionary(..) => true,
        DataType::RunEndEncoded(_, value) => needs_dictionary_metadata(value.data_type()),
        DataType::Struct(fields) => fields
            .iter()
            .any(|field| needs_dictionary_metadata(field.data_type())),
        DataType::Union(fields, _) => fields
            .iter()
            .any(|(_, field)| needs_dictionary_metadata(field.data_type())),
        _ => nested_comparison(dtype),
    }
}
fn nested_comparison(dtype: &DataType) -> bool {
    match dtype {
        DataType::List(field)
        | DataType::LargeList(field)
        | DataType::ListView(field)
        | DataType::LargeListView(field)
        | DataType::FixedSizeList(field, _)
        | DataType::Map(field, _) => needs_dictionary_metadata(field.data_type()),
        _ => false,
    }
}

/// Bounds one standard canonical Arrow owner's `to_data` metadata, without copying its payload.
pub(crate) fn array_data_workspace(
    array: &dyn datafusion::arrow::array::Array,
    check: Check<'_>,
) -> Result<u64> {
    views::metadata(array, check)
}

pub(super) fn descriptor_tree(
    array: &dyn datafusion::arrow::array::Array,
    check: Check<'_>,
) -> Result<u64> {
    views::descriptor_tree(array, check)
}
