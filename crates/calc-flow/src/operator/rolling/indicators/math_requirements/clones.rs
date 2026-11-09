use datafusion::{arrow::datatypes::DataType, common::ScalarValue};

use super::bytes::{add, invalid, size, slots};
use crate::Result;

type Check<'a> = &'a mut dyn FnMut() -> Result<()>;

/// Additional heap from `ScalarValue::clone`, excluding its inline slot and
/// shared Arrow arrays, child fields and timezones. String/Vec clones use len.
pub(super) fn scalar_heap(value: &ScalarValue, check: Check<'_>) -> Result<u64> {
    check()?;
    match value {
        ScalarValue::Utf8(value) | ScalarValue::LargeUtf8(value) | ScalarValue::Utf8View(value) => {
            size(value.as_ref().map_or(0, String::len))
        }
        ScalarValue::Binary(value)
        | ScalarValue::LargeBinary(value)
        | ScalarValue::BinaryView(value)
        | ScalarValue::FixedSizeBinary(_, value) => size(value.as_ref().map_or(0, Vec::len)),
        _ => boxed(value, check),
    }
}
fn boxed(value: &ScalarValue, check: Check<'_>) -> Result<u64> {
    match value {
        ScalarValue::Dictionary(key, value) => {
            // Arrow dictionary keys are integer types and have no nested heap.
            if !key.is_integer() {
                return Err(invalid());
            }
            add(slots::<DataType>(1)?, boxed_value(value, check)?)
        }
        ScalarValue::RunEndEncoded(_, _, value) | ScalarValue::Union(Some((_, value)), _, _) => {
            boxed_value(value, check)
        }
        _ => Ok(0),
    }
}
fn boxed_value(value: &ScalarValue, check: Check<'_>) -> Result<u64> {
    add(slots::<ScalarValue>(1)?, scalar_heap(value, check)?)
}

#[cfg(test)]
#[path = "clones_tests.rs"]
mod tests;
