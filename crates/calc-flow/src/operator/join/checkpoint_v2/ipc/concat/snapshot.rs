use datafusion::arrow::{
    array::{
        Array, BinaryViewArray, DictionaryArray, FixedSizeListArray, GenericListArray,
        GenericListViewArray, MapArray, RunArray, StringViewArray, StructArray, UnionArray,
    },
    buffer::Buffer,
    datatypes::{
        ArrowDictionaryKeyType, DataType, Int8Type, Int16Type, Int32Type, Int64Type,
        RunEndIndexType, UInt8Type, UInt16Type, UInt32Type, UInt64Type,
    },
};

use super::super::accounting::{add, data_type_bytes, product};
use crate::Result;

pub(super) fn bytes(array: &dyn Array, check: &dyn Fn() -> Result<()>) -> Result<usize> {
    check()?;
    let controls = data_type_bytes(array.data_type())?;
    let own = view_buffers(array)?;
    add(add(controls, own)?, children(array, check)?)
}

fn view_buffers(array: &dyn Array) -> Result<usize> {
    let buffers = match array.data_type() {
        DataType::Utf8View => typed::<StringViewArray>(array)?.data_buffers().len(),
        DataType::BinaryView => typed::<BinaryViewArray>(array)?.data_buffers().len(),
        _ => return Ok(0),
    };
    // to_data clones an exact-capacity Vec, then inserts its views buffer.
    product(product(add(buffers, 1)?.max(4), 3)?, size_of::<Buffer>())
}

fn children(array: &dyn Array, check: &dyn Fn() -> Result<()>) -> Result<usize> {
    if let Some(child) = unary(array)? {
        return bytes(child, check);
    }
    match array.data_type() {
        DataType::Struct(_) => typed::<StructArray>(array)?
            .columns()
            .iter()
            .try_fold(0, |total, child| add(total, bytes(child.as_ref(), check)?)),
        DataType::Union(fields, _) => {
            let array = typed::<UnionArray>(array)?;
            fields.iter().try_fold(0, |total, (id, _)| {
                add(total, bytes(array.child(id).as_ref(), check)?)
            })
        }
        DataType::Dictionary(key, _) => dictionary(array, key, check),
        DataType::RunEndEncoded(run_ends, _) => run(array, run_ends.data_type(), check),
        _ => Ok(0),
    }
}

fn unary(array: &dyn Array) -> Result<Option<&dyn Array>> {
    Ok(match array.data_type() {
        DataType::List(_) => Some(typed::<GenericListArray<i32>>(array)?.values().as_ref()),
        DataType::LargeList(_) => Some(typed::<GenericListArray<i64>>(array)?.values().as_ref()),
        DataType::ListView(_) => Some(typed::<GenericListViewArray<i32>>(array)?.values().as_ref()),
        DataType::LargeListView(_) => {
            Some(typed::<GenericListViewArray<i64>>(array)?.values().as_ref())
        }
        DataType::FixedSizeList(_, _) => {
            Some(typed::<FixedSizeListArray>(array)?.values().as_ref())
        }
        DataType::Map(_, _) => Some(typed::<MapArray>(array)?.entries()),
        _ => None,
    })
}

fn dictionary(array: &dyn Array, key: &DataType, check: &dyn Fn() -> Result<()>) -> Result<usize> {
    if let Some(bytes) = signed_dictionary(array, key, check)? {
        return Ok(bytes);
    }
    match key {
        DataType::UInt8 => dictionary_values::<UInt8Type>(array, check),
        DataType::UInt16 => dictionary_values::<UInt16Type>(array, check),
        DataType::UInt32 => dictionary_values::<UInt32Type>(array, check),
        DataType::UInt64 => dictionary_values::<UInt64Type>(array, check),
        _ => Err(super::super::super::geometry::invalid(
            "V2 dictionary snapshot key type is invalid",
        )),
    }
}

fn signed_dictionary(
    array: &dyn Array,
    key: &DataType,
    check: &dyn Fn() -> Result<()>,
) -> Result<Option<usize>> {
    let bytes = match key {
        DataType::Int8 => dictionary_values::<Int8Type>(array, check),
        DataType::Int16 => dictionary_values::<Int16Type>(array, check),
        DataType::Int32 => dictionary_values::<Int32Type>(array, check),
        DataType::Int64 => dictionary_values::<Int64Type>(array, check),
        _ => return Ok(None),
    }?;
    Ok(Some(bytes))
}

fn dictionary_values<K: ArrowDictionaryKeyType>(
    array: &dyn Array,
    check: &dyn Fn() -> Result<()>,
) -> Result<usize> {
    bytes(typed::<DictionaryArray<K>>(array)?.values().as_ref(), check)
}

fn run(array: &dyn Array, key: &DataType, check: &dyn Fn() -> Result<()>) -> Result<usize> {
    match key {
        DataType::Int16 => run_values::<Int16Type>(array, check),
        DataType::Int32 => run_values::<Int32Type>(array, check),
        DataType::Int64 => run_values::<Int64Type>(array, check),
        _ => Err(super::super::super::geometry::invalid(
            "V2 run-end snapshot type is invalid",
        )),
    }
}

fn run_values<K: RunEndIndexType>(
    array: &dyn Array,
    check: &dyn Fn() -> Result<()>,
) -> Result<usize> {
    bytes(typed::<RunArray<K>>(array)?.values().as_ref(), check)
}

fn typed<T: Array + 'static>(array: &dyn Array) -> Result<&T> {
    array
        .as_any()
        .downcast_ref::<T>()
        .ok_or_else(|| super::super::super::geometry::invalid("V2 snapshot array type mismatch"))
}
