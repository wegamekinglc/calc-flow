use super::{Check, data, invalid, typed};
use crate::Result;
use datafusion::arrow::{
    array::{Array, ArrayData, DictionaryArray, RunArray, UnionArray},
    datatypes::{
        ArrowDictionaryKeyType, DataType, Int8Type, Int16Type, Int32Type, Int64Type,
        RunEndIndexType, UInt8Type, UInt16Type, UInt32Type, UInt64Type, UnionMode,
    },
};

pub(super) fn encoded(
    array: &dyn Array,
    offset: usize,
    len: usize,
    expected_dtype: &DataType,
    node: &str,
    check: Check<'_>,
) -> Result<ArrayData> {
    match expected_dtype {
        DataType::Dictionary(key, _) => {
            dictionary(array, key, offset, len, expected_dtype, node, check)
        }
        DataType::RunEndEncoded(ends, _) => run(
            array,
            ends.data_type(),
            offset,
            len,
            expected_dtype,
            node,
            check,
        ),
        DataType::Union(_, mode) => union(array, *mode, offset, len, expected_dtype, node, check),
        _ => Err(invalid(node)),
    }
}
fn dictionary(
    array: &dyn Array,
    key: &DataType,
    offset: usize,
    len: usize,
    expected_dtype: &DataType,
    node: &str,
    check: Check<'_>,
) -> Result<ArrayData> {
    match key {
        DataType::Int8 => {
            dictionary_data::<Int8Type>(array, offset, len, expected_dtype, node, check)
        }
        DataType::Int16 => {
            dictionary_data::<Int16Type>(array, offset, len, expected_dtype, node, check)
        }
        DataType::Int32 => {
            dictionary_data::<Int32Type>(array, offset, len, expected_dtype, node, check)
        }
        DataType::Int64 => {
            dictionary_data::<Int64Type>(array, offset, len, expected_dtype, node, check)
        }
        _ => unsigned_dictionary(array, key, offset, len, expected_dtype, node, check),
    }
}
fn unsigned_dictionary(
    array: &dyn Array,
    key: &DataType,
    offset: usize,
    len: usize,
    expected_dtype: &DataType,
    node: &str,
    check: Check<'_>,
) -> Result<ArrayData> {
    match key {
        DataType::UInt8 => {
            dictionary_data::<UInt8Type>(array, offset, len, expected_dtype, node, check)
        }
        DataType::UInt16 => {
            dictionary_data::<UInt16Type>(array, offset, len, expected_dtype, node, check)
        }
        DataType::UInt32 => {
            dictionary_data::<UInt32Type>(array, offset, len, expected_dtype, node, check)
        }
        DataType::UInt64 => {
            dictionary_data::<UInt64Type>(array, offset, len, expected_dtype, node, check)
        }
        _ => Err(invalid(node)),
    }
}
fn dictionary_data<K: ArrowDictionaryKeyType>(
    array: &dyn Array,
    offset: usize,
    len: usize,
    expected_dtype: &DataType,
    node: &str,
    check: Check<'_>,
) -> Result<ArrayData> {
    let array = typed::<DictionaryArray<K>>(array, node)?;
    let values = data(
        array.values().as_ref(),
        0,
        array.values().len(),
        dictionary_value(expected_dtype, node)?,
        node,
        check,
    )?;
    ArrayData::builder(expected_dtype.clone())
        .len(len)
        .offset(offset)
        .nulls(array.keys().nulls().map(|mask| mask.slice(offset, len)))
        .buffers(vec![array.keys().values().inner().clone()])
        .child_data(vec![values])
        .build()
        .map_err(|_| invalid(node))
}
fn run(
    array: &dyn Array,
    key: &DataType,
    offset: usize,
    len: usize,
    expected_dtype: &DataType,
    node: &str,
    check: Check<'_>,
) -> Result<ArrayData> {
    match key {
        DataType::Int16 => run_data::<Int16Type>(array, offset, len, expected_dtype, node, check),
        DataType::Int32 => run_data::<Int32Type>(array, offset, len, expected_dtype, node, check),
        DataType::Int64 => run_data::<Int64Type>(array, offset, len, expected_dtype, node, check),
        _ => Err(invalid(node)),
    }
}
fn run_data<R: RunEndIndexType>(
    array: &dyn Array,
    offset: usize,
    len: usize,
    expected_dtype: &DataType,
    node: &str,
    check: Check<'_>,
) -> Result<ArrayData> {
    let array = typed::<RunArray<R>>(array, node)?;
    let ends = array.run_ends().inner();
    let ends = ArrayData::builder(R::DATA_TYPE)
        .len(ends.len())
        .buffers(vec![ends.inner().clone()])
        .build()
        .map_err(|_| invalid(node))?;
    let values = data(
        array.values().as_ref(),
        0,
        array.values().len(),
        run_value(expected_dtype, node)?,
        node,
        check,
    )?;
    let offset = array
        .offset()
        .checked_add(offset)
        .ok_or_else(|| invalid(node))?;
    ArrayData::builder(expected_dtype.clone())
        .len(len)
        .offset(offset)
        .child_data(vec![ends, values])
        .build()
        .map_err(|_| invalid(node))
}
fn union(
    array: &dyn Array,
    mode: UnionMode,
    offset: usize,
    len: usize,
    expected_dtype: &DataType,
    node: &str,
    check: Check<'_>,
) -> Result<ArrayData> {
    let array = typed::<UnionArray>(array, node)?;
    let DataType::Union(fields, _) = expected_dtype else {
        return Err(invalid(node));
    };
    let mut children = Vec::with_capacity(fields.len());
    for (id, field) in fields.iter() {
        let child = array.child(id);
        let (start, length) = if mode == UnionMode::Sparse {
            (offset, len)
        } else {
            (0, child.len())
        };
        children.push(data(
            child.as_ref(),
            start,
            length,
            field.data_type(),
            node,
            check,
        )?);
    }
    let mut buffers = vec![array.type_ids().slice(offset, len).inner().clone()];
    if let Some(offsets) = array.offsets() {
        buffers.push(offsets.slice(offset, len).inner().clone());
    }
    ArrayData::builder(expected_dtype.clone())
        .len(len)
        .buffers(buffers)
        .child_data(children)
        .build()
        .map_err(|_| invalid(node))
}

fn dictionary_value<'a>(dtype: &'a DataType, node: &str) -> Result<&'a DataType> {
    let DataType::Dictionary(_, value) = dtype else {
        return Err(invalid(node));
    };
    Ok(value)
}
fn run_value<'a>(dtype: &'a DataType, node: &str) -> Result<&'a DataType> {
    let DataType::RunEndEncoded(_, value) = dtype else {
        return Err(invalid(node));
    };
    Ok(value.data_type())
}
