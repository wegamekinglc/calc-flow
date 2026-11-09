use super::{Check, invalid, null_scalar, read_scalar_as, typed};
use crate::Result;
use datafusion::{
    arrow::{
        array::{Array, DictionaryArray, RunArray, UnionArray},
        datatypes::{
            ArrowDictionaryKeyType, DataType, Int8Type, Int16Type, Int32Type, Int64Type,
            RunEndIndexType, UInt8Type, UInt16Type, UInt32Type, UInt64Type,
        },
    },
    common::ScalarValue,
};
use std::sync::Arc;

pub(super) fn read(
    array: &dyn Array,
    row: usize,
    expected: &DataType,
    node: &str,
    check: Check<'_>,
) -> Result<ScalarValue> {
    match expected {
        DataType::Dictionary(key, _) => dictionary(array, key, row, expected, node, check),
        DataType::RunEndEncoded(ends, _) => {
            run(array, ends.data_type(), row, expected, node, check)
        }
        DataType::Union(..) => union(array, row, expected, node, check),
        _ => Err(invalid(node)),
    }
}
fn dictionary(
    array: &dyn Array,
    key: &DataType,
    row: usize,
    expected: &DataType,
    node: &str,
    check: Check<'_>,
) -> Result<ScalarValue> {
    match key {
        DataType::Int8 => dict::<Int8Type>(array, row, expected, node, check),
        DataType::Int16 => dict::<Int16Type>(array, row, expected, node, check),
        DataType::Int32 => dict::<Int32Type>(array, row, expected, node, check),
        DataType::Int64 => dict::<Int64Type>(array, row, expected, node, check),
        _ => unsigned(array, key, row, expected, node, check),
    }
}
fn unsigned(
    array: &dyn Array,
    key: &DataType,
    row: usize,
    expected: &DataType,
    node: &str,
    check: Check<'_>,
) -> Result<ScalarValue> {
    match key {
        DataType::UInt8 => dict::<UInt8Type>(array, row, expected, node, check),
        DataType::UInt16 => dict::<UInt16Type>(array, row, expected, node, check),
        DataType::UInt32 => dict::<UInt32Type>(array, row, expected, node, check),
        DataType::UInt64 => dict::<UInt64Type>(array, row, expected, node, check),
        _ => Err(invalid(node)),
    }
}
fn dict<K: ArrowDictionaryKeyType>(
    array: &dyn Array,
    row: usize,
    expected: &DataType,
    node: &str,
    check: Check<'_>,
) -> Result<ScalarValue> {
    let array = typed::<DictionaryArray<K>>(array, node)?;
    let DataType::Dictionary(_, value_type) = expected else {
        return Err(invalid(node));
    };
    let value = if let Some(index) = array.key(row) {
        read_scalar_as(array.values().as_ref(), index, value_type, node, check)?
    } else {
        null_scalar(value_type, node, check)?
    };
    Ok(ScalarValue::Dictionary(
        Box::new(K::DATA_TYPE),
        Box::new(value),
    ))
}
fn run(
    array: &dyn Array,
    ends: &DataType,
    row: usize,
    expected: &DataType,
    node: &str,
    check: Check<'_>,
) -> Result<ScalarValue> {
    match ends {
        DataType::Int16 => run_value::<Int16Type>(array, row, expected, node, check),
        DataType::Int32 => run_value::<Int32Type>(array, row, expected, node, check),
        DataType::Int64 => run_value::<Int64Type>(array, row, expected, node, check),
        _ => Err(invalid(node)),
    }
}
fn run_value<R: RunEndIndexType>(
    array: &dyn Array,
    row: usize,
    expected: &DataType,
    node: &str,
    check: Check<'_>,
) -> Result<ScalarValue> {
    let array = typed::<RunArray<R>>(array, node)?;
    let DataType::RunEndEncoded(ends, field) = expected else {
        return Err(invalid(node));
    };
    let value = read_scalar_as(
        array.values().as_ref(),
        array.get_physical_index(row),
        field.data_type(),
        node,
        check,
    )?;
    Ok(ScalarValue::RunEndEncoded(
        Arc::clone(ends),
        Arc::clone(field),
        Box::new(value),
    ))
}
fn union(
    array: &dyn Array,
    row: usize,
    expected: &DataType,
    node: &str,
    check: Check<'_>,
) -> Result<ScalarValue> {
    let array = typed::<UnionArray>(array, node)?;
    let DataType::Union(fields, mode) = expected else {
        return Err(invalid(node));
    };
    let id = array.type_id(row);
    let field = fields
        .iter()
        .find_map(|(field_id, field)| (field_id == id).then_some(field))
        .ok_or_else(|| invalid(node))?;
    let value = read_scalar_as(
        array.child(id).as_ref(),
        array.value_offset(row),
        field.data_type(),
        node,
        check,
    )?;
    Ok(ScalarValue::Union(
        Some((id, Box::new(value))),
        fields.clone(),
        *mode,
    ))
}
