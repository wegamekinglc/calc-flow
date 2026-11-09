use super::{Result, invalid};
use datafusion::{arrow::datatypes::DataType, common::ScalarValue};

pub(super) fn valid(dtype: &DataType, value: Option<&ScalarValue>) -> Result<bool> {
    let Some(value) = value else {
        return null_valid(dtype);
    };
    if let Some(array) = super::super::single::array(value) {
        if array.len() != 1 {
            return Err(invalid("nested scalar shape"));
        }
        return Ok(array.is_valid(0));
    }
    match value {
        ScalarValue::Null | ScalarValue::RunEndEncoded(..) | ScalarValue::Union(..) => Ok(true),
        ScalarValue::Dictionary(_, value) => dictionary(dtype, Some(value)),
        _ => Ok(!value.is_null()),
    }
}
fn dictionary(dtype: &DataType, value: Option<&ScalarValue>) -> Result<bool> {
    let DataType::Dictionary(_, dtype) = dtype else {
        return Err(invalid("dictionary value datatype"));
    };
    valid(dtype, value)
}
fn null_valid(dtype: &DataType) -> Result<bool> {
    match dtype {
        DataType::Null | DataType::RunEndEncoded(..) | DataType::Union(..) => Ok(true),
        DataType::Dictionary(_, value) => null_valid(value),
        _ => Ok(false),
    }
}
