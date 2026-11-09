use datafusion::arrow::{
    array::{Array, BinaryArray, LargeBinaryArray, LargeStringArray, StringArray, StructArray},
    datatypes::DataType,
    record_batch::RecordBatch,
};

use super::{
    Check, Result,
    bytes::{add, invalid, product, size, total},
};

/// An allocation-free upper bound for the existing whole-record estimator.
/// Unknown nested layouts fall back to the conservative split construction quote.
pub(super) fn record(record: &RecordBatch, check: Check<'_>) -> Result<Option<u64>> {
    let mut total = 0;
    for column in record.columns() {
        check()?;
        let Some(bytes) = array(column.as_ref(), check)? else {
            return Ok(None);
        };
        total = add(total, bytes)?;
    }
    Ok(Some(total))
}
fn array(array: &dyn Array, check: Check<'_>) -> Result<Option<u64>> {
    check()?;
    let Some(own) = own(array, check)? else {
        return Ok(None);
    };
    let nulls = array.nulls().map_or(0, |nulls| nulls.len().div_ceil(8));
    add(own, size(nulls)?).map(Some)
}
fn own(array: &dyn Array, check: Check<'_>) -> Result<Option<u64>> {
    if let Some(width) = array.data_type().primitive_width() {
        return product(array.len(), width).map(Some);
    }
    match array.data_type() {
        DataType::Null => Ok(Some(0)),
        DataType::Boolean => size(array.len().div_ceil(8)).map(Some),
        DataType::FixedSizeBinary(width) => fixed(array.len(), *width),
        DataType::Struct(_) => structure(typed::<StructArray>(array)?, check),
        _ => variable(array),
    }
}
fn fixed(rows: usize, width: i32) -> Result<Option<u64>> {
    product(rows, usize::try_from(width).map_err(|_| invalid())?).map(Some)
}
fn variable(array: &dyn Array) -> Result<Option<u64>> {
    match array.data_type() {
        DataType::Utf8 | DataType::Binary => narrow(array).map(Some),
        DataType::LargeUtf8 | DataType::LargeBinary => wide(array).map(Some),
        _ => Ok(None),
    }
}
fn narrow(array: &dyn Array) -> Result<u64> {
    if array.data_type() == &DataType::Utf8 {
        let value = typed::<StringArray>(array)?;
        return byte_storage(value.value_offsets().len(), value.value_data().len(), 4);
    }
    let value = typed::<BinaryArray>(array)?;
    byte_storage(value.value_offsets().len(), value.value_data().len(), 4)
}
fn wide(array: &dyn Array) -> Result<u64> {
    if array.data_type() == &DataType::LargeUtf8 {
        let value = typed::<LargeStringArray>(array)?;
        return byte_storage(value.value_offsets().len(), value.value_data().len(), 8);
    }
    let value = typed::<LargeBinaryArray>(array)?;
    byte_storage(value.value_offsets().len(), value.value_data().len(), 8)
}
fn byte_storage(offsets: usize, values: usize, width: usize) -> Result<u64> {
    total([product(offsets, width), size(values)])
}
fn structure(array: &StructArray, check: Check<'_>) -> Result<Option<u64>> {
    let mut total = 0;
    for child in array.columns() {
        let Some(bytes) = self::array(child.as_ref(), check)? else {
            return Ok(None);
        };
        total = add(total, bytes)?;
    }
    Ok(Some(total))
}
fn typed<T: Array + 'static>(array: &dyn Array) -> Result<&T> {
    array.as_any().downcast_ref().ok_or_else(invalid)
}
