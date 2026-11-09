use super::{Check, Result, add, bytes, canonical, cost::times, encoded, invalid};
use datafusion::{
    arrow::{array::Array, datatypes::DataType as D},
    common::ScalarValue,
};

pub(super) fn cell(array: &dyn Array, row: usize, check: Check<'_>) -> Result<u64> {
    check()?;
    if canonical::direct(array, row, check)?.is_some() {
        return Ok(0);
    }
    if matches!(
        array.data_type(),
        D::Dictionary(..) | D::RunEndEncoded(..) | D::Union(..)
    ) {
        return encoded::clone_cell(array, row, check).map(|cost| cost.owned);
    }
    plain(array, row)
}
fn plain(array: &dyn Array, row: usize) -> Result<u64> {
    if array.is_null(row) {
        return Ok(0);
    }
    if let Some(bytes) = bytes::cell(array, row)? {
        return Ok(bytes);
    }
    if array.data_type().is_primitive() || matches!(array.data_type(), D::Null | D::Boolean) {
        return Ok(0);
    }
    Err(invalid())
}
pub(super) fn null(dtype: &D, check: Check<'_>) -> Result<u64> {
    super::null::scalar(dtype, check)?;
    null_heap(dtype, check)
}
fn null_heap(dtype: &D, check: Check<'_>) -> Result<u64> {
    check()?;
    match dtype {
        D::Dictionary(key, value) => null_dictionary(key, value, check),
        D::RunEndEncoded(_, field) => null_run(field.data_type(), check),
        _ => Ok(0),
    }
}
fn null_dictionary(key: &D, value: &D, check: Check<'_>) -> Result<u64> {
    if !key.is_integer() {
        return Err(invalid());
    }
    add(
        encoded::dictionary_box(key, check)?.owned,
        null_heap(value, check)?,
    )
}
fn null_run(value: &D, check: Check<'_>) -> Result<u64> {
    add(times::<ScalarValue>(1)?, null_heap(value, check)?)
}
