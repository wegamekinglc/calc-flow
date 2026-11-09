use super::super::math::{buffer_headers, grow, retained, size, sum, vector};
use super::{Cell, Check, OutputPlan, Result, column, invalid};
use datafusion::{
    arrow::datatypes::{DataType, FieldRef},
    common::ScalarValue,
};
use std::borrow::Cow;

pub(super) fn plan(
    dtype: &DataType,
    rows: usize,
    cell: &mut Cell<'_, '_>,
    check: Check<'_>,
) -> Result<OutputPlan> {
    let DataType::RunEndEncoded(ends, value) = dtype else {
        return Err(invalid("run datatype"));
    };
    let width = end_width(ends.data_type(), rows)?;
    let indices = select(ends, value, rows, cell, check)?;
    let mut selected =
        |row| unpack(ends, value, cell(indices[row])).expect("validated stable Run cells");
    let child = column(value.data_type(), indices.len(), &mut selected, check)?;
    finish(dtype, rows, indices.len(), width, child, cell, check)
}
fn select(
    ends: &FieldRef,
    value: &FieldRef,
    rows: usize,
    cell: &mut Cell<'_, '_>,
    check: Check<'_>,
) -> Result<Vec<usize>> {
    let mut indices = Vec::with_capacity(rows);
    for row in 0..rows {
        check()?;
        let next = unpack(ends, value, cell(row))?;
        if differs(
            value.data_type(),
            indices.last().copied(),
            next,
            ends,
            value,
            cell,
            check,
        )? {
            indices.push(row);
        }
    }
    Ok(indices)
}
fn differs(
    dtype: &DataType,
    current: Option<usize>,
    next: Option<&ScalarValue>,
    ends: &FieldRef,
    value: &FieldRef,
    cell: &mut Cell<'_, '_>,
    check: Check<'_>,
) -> Result<bool> {
    let Some(current) = current else {
        return Ok(true);
    };
    let previous = unpack(ends, value, cell(current))?;
    let equal = {
        let left = comparable(dtype, previous, check)?;
        let right = comparable(dtype, next, check)?;
        left == right
    };
    check()?;
    Ok(!equal)
}
fn comparable<'a>(
    dtype: &DataType,
    value: Option<&'a ScalarValue>,
    check: Check<'_>,
) -> Result<Cow<'a, ScalarValue>> {
    check()?;
    match value {
        Some(value) => Ok(Cow::Borrowed(value)),
        None => super::super::super::materialize::canonical::null_scalar(
            dtype,
            "rolling-output-plan",
            check,
        )
        .map(Cow::Owned),
    }
}
fn unpack<'a>(
    ends: &FieldRef,
    value: &FieldRef,
    scalar: Option<&'a ScalarValue>,
) -> Result<Option<&'a ScalarValue>> {
    match scalar {
        Some(ScalarValue::RunEndEncoded(actual_ends, actual_value, scalar))
            if actual_ends == ends && actual_value == value =>
        {
            Ok(Some(scalar.as_ref()))
        }
        None => Ok(None),
        _ => Err(invalid("run scalar fields")),
    }
}
fn end_width(dtype: &DataType, rows: usize) -> Result<u64> {
    if !matches!(dtype, DataType::Int16 | DataType::Int32 | DataType::Int64) {
        return Err(invalid("run index datatype"));
    }
    super::super::producer::key_index(dtype, size(rows)?)?;
    size(
        dtype
            .primitive_width()
            .ok_or_else(|| invalid("run index width"))?,
    )
}
fn finish(
    dtype: &DataType,
    rows: usize,
    runs: usize,
    width: u64,
    child: OutputPlan,
    cell: &mut Cell<'_, '_>,
    check: Check<'_>,
) -> Result<OutputPlan> {
    let empty = super::super::super::scalar_materialization_plan::null_array_cost(dtype, 0, check)?;
    let (ends, vectors) = storage(runs, width)?;
    let metadata = super::super::workspace::column(dtype, rows, cell, check)?;
    Ok(OutputPlan {
        output_bytes: sum([Ok(child.output_bytes), Ok(empty.logical_owned), Ok(ends)])?,
        scratch_bytes: sum([
            Ok(child.scratch_bytes),
            Ok(empty.construction_peak),
            Ok(vectors),
            Ok(metadata),
        ])?,
    })
}

fn storage(runs: usize, width: u64) -> Result<(u64, u64)> {
    let ends = retained(size(runs)?, width)?;
    let vectors = sum([
        grow(size(runs)?, width),
        vector::<ScalarValue>(runs),
        buffer_headers(1),
    ])?;
    Ok((ends, vectors))
}
