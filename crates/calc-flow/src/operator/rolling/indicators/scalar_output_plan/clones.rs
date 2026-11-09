use super::{
    Check, Result,
    math::{add, bytes, size, sum},
    single,
};
use datafusion::{arrow::datatypes::DataType, common::ScalarValue};

pub(super) fn column<'a>(
    dtype: &DataType,
    rows: usize,
    cell: &mut dyn FnMut(usize) -> Option<&'a ScalarValue>,
    check: Check<'_>,
) -> Result<u64> {
    let mut total = bytes::<ScalarValue>(size(rows)?)?;
    for row in 0..rows {
        check()?;
        let owned = cell_cost(dtype, cell(row), check)?;
        total = add(total, owned)?;
    }
    Ok(total)
}
fn cell_cost(dtype: &DataType, value: Option<&ScalarValue>, check: Check<'_>) -> Result<u64> {
    match value {
        Some(value) => scalar(value, check),
        None => super::super::scalar_materialization_plan::null_requirement(dtype, check),
    }
}
fn scalar(value: &ScalarValue, check: Check<'_>) -> Result<u64> {
    check()?;
    if single::array(value).is_some() {
        return Ok(0);
    }
    match value {
        ScalarValue::Utf8(value) | ScalarValue::LargeUtf8(value) | ScalarValue::Utf8View(value) => {
            size(value.as_ref().map_or(0, String::len))
        }
        ScalarValue::RunEndEncoded(_, _, value) | ScalarValue::Union(Some((_, value)), _, _) => {
            boxed(value, check)
        }
        ScalarValue::Dictionary(key, value) => sum([
            bytes::<DataType>(1),
            datatype(key, check),
            boxed(value, check),
        ]),
        _ => binary(value),
    }
}
fn binary(value: &ScalarValue) -> Result<u64> {
    match value {
        ScalarValue::Binary(value)
        | ScalarValue::LargeBinary(value)
        | ScalarValue::BinaryView(value)
        | ScalarValue::FixedSizeBinary(_, value) => size(value.as_ref().map_or(0, Vec::len)),
        _ => Ok(0),
    }
}
fn boxed(value: &ScalarValue, check: Check<'_>) -> Result<u64> {
    add(bytes::<ScalarValue>(1)?, scalar(value, check)?)
}
pub(super) fn datatype(dtype: &DataType, check: Check<'_>) -> Result<u64> {
    check()?;
    if let DataType::Dictionary(key, value) = dtype {
        return sum([
            bytes::<DataType>(2),
            datatype(key, check),
            datatype(value, check),
        ]);
    }
    Ok(0)
}
