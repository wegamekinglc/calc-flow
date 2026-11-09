use super::{invalid, typed};
use crate::Result;
use datafusion::{
    arrow::{
        array::{
            Array, BinaryArray, BinaryViewArray, BooleanArray, FixedSizeBinaryArray,
            LargeBinaryArray, LargeStringArray, NullArray, PrimitiveArray, StringArray,
            StringViewArray,
        },
        datatypes::DataType,
    },
    common::ScalarValue,
};
macro_rules! primitive {
    ($ty:ty, $array:ident, $row:ident, $node:ident) => {
        value::<PrimitiveArray<$ty>>($array, $row, $node)
    };
}
pub(super) fn read_as(
    array: &dyn Array,
    row: usize,
    expected: &DataType,
    node: &str,
) -> Result<ScalarValue> {
    let scalar = read(array, row, node)?;
    let DataType::Timestamp(_, timezone) = expected else {
        return Ok(scalar);
    };
    Ok(rebind_timezone(scalar, timezone.as_ref()))
}
fn rebind_timezone(scalar: ScalarValue, timezone: Option<&std::sync::Arc<str>>) -> ScalarValue {
    match scalar {
        ScalarValue::TimestampSecond(value, _) => {
            ScalarValue::TimestampSecond(value, timezone.cloned())
        }
        ScalarValue::TimestampMillisecond(value, _) => {
            ScalarValue::TimestampMillisecond(value, timezone.cloned())
        }
        ScalarValue::TimestampMicrosecond(value, _) => {
            ScalarValue::TimestampMicrosecond(value, timezone.cloned())
        }
        ScalarValue::TimestampNanosecond(value, _) => {
            ScalarValue::TimestampNanosecond(value, timezone.cloned())
        }
        other => other,
    }
}
fn read(array: &dyn Array, row: usize, node: &str) -> Result<ScalarValue> {
    datafusion::arrow::array::downcast_primitive! {
        array.data_type() => (primitive, array, row, node),
        _ => bytes(array, row, node)
    }
}
fn bytes(array: &dyn Array, row: usize, node: &str) -> Result<ScalarValue> {
    match array.data_type() {
        DataType::Utf8 => value::<StringArray>(array, row, node),
        DataType::LargeUtf8 => value::<LargeStringArray>(array, row, node),
        DataType::Utf8View => value::<StringViewArray>(array, row, node),
        DataType::Binary => value::<BinaryArray>(array, row, node),
        _ => other(array, row, node),
    }
}
fn other(array: &dyn Array, row: usize, node: &str) -> Result<ScalarValue> {
    match array.data_type() {
        DataType::LargeBinary => value::<LargeBinaryArray>(array, row, node),
        DataType::BinaryView => value::<BinaryViewArray>(array, row, node),
        DataType::FixedSizeBinary(_) => value::<FixedSizeBinaryArray>(array, row, node),
        DataType::Boolean => value::<BooleanArray>(array, row, node),
        DataType::Null => value::<NullArray>(array, row, node),
        _ => Err(invalid(node)),
    }
}
fn value<T: Array + 'static>(array: &dyn Array, row: usize, node: &str) -> Result<ScalarValue> {
    ScalarValue::try_from_array(typed::<T>(array, node)?, row).map_err(|_| invalid(node))
}
