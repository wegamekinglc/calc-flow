use crate::{CalcFlowError, Result};
use datafusion::{
    arrow::{array::ArrayRef, compute::concat, datatypes::DataType},
    common::ScalarValue,
};

#[path = "producer/dictionary.rs"]
mod dictionary;
#[path = "producer/fallback.rs"]
mod fallback;
#[path = "producer/normal.rs"]
mod normal;
#[path = "producer/run.rs"]
mod run;
#[path = "producer/union.rs"]
mod union;
#[path = "producer/views.rs"]
mod views;
pub(super) fn view_index(preceding: u64, index: u32) -> Result<u32> {
    views::reference_index(preceding, index)
}

type Check<'a> = &'a mut dyn FnMut() -> Result<()>;

pub(crate) fn encode_column<'a>(
    dtype: &DataType,
    rows: usize,
    cell: &mut dyn FnMut(usize) -> Option<&'a ScalarValue>,
    check: Check<'_>,
) -> Result<ArrayRef> {
    check()?;
    if rows == 0 {
        return null_array(dtype, 0, check);
    }
    let mut values = Vec::with_capacity(rows);
    for row in 0..rows {
        check()?;
        values.push(match cell(row) {
            Some(value) => value.clone(),
            None => {
                super::super::materialize::canonical::null_scalar(dtype, "rolling-output", check)?
            }
        });
    }
    column(values, check)
}

fn column(values: Vec<ScalarValue>, check: Check<'_>) -> Result<ArrayRef> {
    check()?;
    let dtype = values
        .first()
        .ok_or_else(|| invalid("empty scalar column"))?
        .data_type();
    match &dtype {
        DataType::Dictionary(key, _) => dictionary::column(values, key, check),
        DataType::RunEndEncoded(ends, value) => run::column(values, ends, value, check),
        DataType::Union(..) => union::column(values, check),
        DataType::FixedSizeList(..) => fixed(values, check),
        _ => ordinary(&dtype, values, check),
    }
}
fn ordinary(dtype: &DataType, values: Vec<ScalarValue>, check: Check<'_>) -> Result<ArrayRef> {
    scalar_views(dtype, &values, check)?;
    super::normal::column(dtype, &values, check)?;
    if normal::needed(dtype, check)? {
        return normal_column(values, check);
    }
    ScalarValue::iter_to_array(values).map_err(|_| invalid("scalar column"))
}

pub(super) fn single(value: &ScalarValue, check: Check<'_>) -> Result<ArrayRef> {
    check()?;
    scalar_view(value)?;
    match value {
        ScalarValue::Union(value, fields, mode) => {
            union::single(value.as_ref(), fields, *mode, check)
        }
        ScalarValue::Dictionary(key, value) => dictionary::single(key, value, check),
        ScalarValue::RunEndEncoded(ends, field, value) => run::single(ends, field, value, check),
        ScalarValue::Null => null_array(&DataType::Null, 1, check),
        _ => value.to_array().map_err(|_| invalid("scalar child")),
    }
}
fn null_array(dtype: &DataType, len: usize, check: Check<'_>) -> Result<ArrayRef> {
    super::super::materialize::canonical::null_array(dtype, len, "rolling-output", check)
}
fn concatenate(arrays: &[ArrayRef], check: Check<'_>) -> Result<ArrayRef> {
    fallback::validate(arrays, check)?;
    let refs = arrays.iter().map(AsRef::as_ref).collect::<Vec<_>>();
    concat(&refs).map_err(|_| invalid("scalar array concatenation"))
}
fn fixed(values: Vec<ScalarValue>, check: Check<'_>) -> Result<ArrayRef> {
    let mut arrays = Vec::with_capacity(values.len());
    for value in values {
        arrays.push(single(&value, check)?);
    }
    repair_fixed(&mut arrays, check)?;
    concatenate(&arrays, check)
}
fn repair_fixed(arrays: &mut [ArrayRef], check: Check<'_>) -> Result<()> {
    let first = first_nonnull(arrays, check)?;
    if let Some(dtype @ DataType::FixedSizeList(..)) = first {
        for array in arrays {
            check()?;
            if array.is_null(0) {
                *array = null_array(&dtype, 1, check)?;
            }
        }
    }
    Ok(())
}
fn first_nonnull(arrays: &[ArrayRef], check: Check<'_>) -> Result<Option<DataType>> {
    for array in arrays {
        check()?;
        if !array.is_null(0) {
            return Ok(Some(array.data_type().clone()));
        }
    }
    Ok(None)
}

fn invalid(reason: &str) -> CalcFlowError {
    CalcFlowError::Format {
        message: format!("rolling scalar output {reason} is invalid"),
    }
}

pub(super) fn key_span(key: &DataType, offset: u64, len: u64) -> Result<()> {
    fallback::span(key, offset, len)
}

pub(super) fn dense_length(rows: u64) -> Result<()> {
    views::dense_length(rows)
}

#[cfg(test)]
#[path = "producer_tests.rs"]
mod tests;

pub(super) fn key_index(key: &DataType, index: u64) -> Result<()> {
    fallback::span(key, index, 1)
}

fn normal_column(values: Vec<ScalarValue>, check: Check<'_>) -> Result<ArrayRef> {
    let mut arrays = Vec::with_capacity(values.len());
    for value in values {
        arrays.push(single(&value, check)?);
    }
    let refs = arrays.iter().map(AsRef::as_ref).collect::<Vec<_>>();
    normal::concat(&refs, check)
}

pub(super) fn normal_fallback(
    dtype: &DataType,
    count: usize,
    at: &mut super::normal::At<'_>,
    check: Check<'_>,
) -> Result<()> {
    fallback::normal(dtype, count, at, check)
}

pub(super) fn key_limit(key: &DataType) -> Result<u64> {
    fallback::key_limit(key)
}

fn scalar_views(dtype: &DataType, values: &[ScalarValue], check: Check<'_>) -> Result<()> {
    if !matches!(dtype, DataType::Utf8View | DataType::BinaryView) {
        return Ok(());
    }
    let mut blocks = super::leaf::View::default();
    for value in values {
        check()?;
        blocks.append(scalar_view_len(value)?)?;
    }
    Ok(())
}
fn scalar_view(value: &ScalarValue) -> Result<()> {
    super::math::scalar_view_length(scalar_view_len(value)?)
}
fn scalar_view_len(value: &ScalarValue) -> Result<u64> {
    let len = match value {
        ScalarValue::Utf8View(value) => value.as_ref().map_or(0, String::len),
        ScalarValue::BinaryView(value) => value.as_ref().map_or(0, Vec::len),
        _ => 0,
    };
    super::math::size(len)
}
