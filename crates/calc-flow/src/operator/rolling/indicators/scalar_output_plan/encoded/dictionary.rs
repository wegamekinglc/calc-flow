use super::super::math::{bitmap, buffer_headers, bytes, grow, mul, scale, size, sum};
use super::{Cell, Check, OutputPlan, Result, column, invalid, physical};
use datafusion::{arrow::datatypes::DataType, common::ScalarValue};

pub(super) fn plan(
    dtype: &DataType,
    key: &DataType,
    value: &DataType,
    rows: usize,
    cell: &mut Cell<'_, '_>,
    check: Check<'_>,
) -> Result<OutputPlan> {
    let width = key_width(key)?;
    validate(key, value, rows, cell, check)?;
    // `unwrap` was validated; no fallback scalar or owned callback is manufactured.
    let mut borrowed = |row| unwrap(key, cell(row)).expect("validated stable dictionary cell");
    let child = column(value, rows, &mut borrowed, check)?;
    finish(dtype, rows, width, child, check)
}
fn finish(
    dtype: &DataType,
    rows: usize,
    width: u64,
    child: OutputPlan,
    check: Check<'_>,
) -> Result<OutputPlan> {
    let descriptor =
        super::super::super::scalar_materialization_plan::null_array_cost(dtype, 0, check)?;
    let buffer = key_buffers(rows, width)?;
    Ok(OutputPlan {
        output_bytes: sum([
            Ok(child.output_bytes),
            Ok(descriptor.logical_owned),
            Ok(buffer),
        ])?,
        scratch_bytes: sum([
            Ok(child.scratch_bytes),
            Ok(descriptor.construction_peak),
            clone_workspace(rows),
            key_workspace(rows, width),
            Ok(buffer),
        ])?,
    })
}
fn unwrap<'a>(key: &DataType, value: Option<&'a ScalarValue>) -> Result<Option<&'a ScalarValue>> {
    match value {
        Some(ScalarValue::Dictionary(actual, value)) if actual.as_ref() == key => {
            Ok(Some(value.as_ref()))
        }
        None => Ok(None),
        _ => Err(invalid("dictionary scalar datatype")),
    }
}
fn validate(
    key: &DataType,
    value: &DataType,
    rows: usize,
    cell: &mut Cell<'_, '_>,
    check: Check<'_>,
) -> Result<()> {
    for row in 0..rows {
        check()?;
        position(key, value, unwrap(key, cell(row))?, row)?;
    }
    Ok(())
}
fn position(
    key: &DataType,
    value: &DataType,
    cell: Option<&ScalarValue>,
    row: usize,
) -> Result<()> {
    if physical::valid(value, cell)? {
        super::super::producer::key_index(key, size(row)?)?;
    }
    Ok(())
}
fn key_width(key: &DataType) -> Result<u64> {
    super::super::producer::key_index(key, 0)?;
    size(
        key.primitive_width()
            .ok_or_else(|| invalid("dictionary key width"))?,
    )
}
fn key_buffers(rows: usize, width: u64) -> Result<u64> {
    sum([mul(size(rows)?, width), scale(bitmap(size(rows)?), 2)])
}
fn key_workspace(rows: usize, width: u64) -> Result<u64> {
    sum([grow(size(rows)?, mul(width, 2)?), buffer_headers(2)])
}
fn clone_workspace(rows: usize) -> Result<u64> {
    sum([
        scale(bytes::<ScalarValue>(size(rows)?), 2),
        bytes::<DataType>(size(rows)?),
    ])
}
