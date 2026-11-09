use super::super::{Check, invalid, typed};
use super::{Graph, finish};
use crate::Result;
use datafusion::arrow::{
    array::{Array, DictionaryArray, RunArray},
    datatypes::{
        ArrowDictionaryKeyType, DataType, Int8Type, Int16Type, Int32Type, Int64Type,
        RunEndIndexType, UInt8Type, UInt16Type, UInt32Type, UInt64Type,
    },
};

pub(super) fn graph(array: &dyn Array, check: Check<'_>) -> Result<Graph> {
    match array.data_type() {
        DataType::Dictionary(key, _) => dictionary(array, key, check),
        DataType::RunEndEncoded(ends, _) => run(array, ends.data_type(), check),
        _ => Err(invalid()),
    }
}
fn dictionary(array: &dyn Array, key: &DataType, check: Check<'_>) -> Result<Graph> {
    match key {
        DataType::Int8 => dict::<Int8Type>(array, check),
        DataType::Int16 => dict::<Int16Type>(array, check),
        DataType::Int32 => dict::<Int32Type>(array, check),
        DataType::Int64 => dict::<Int64Type>(array, check),
        _ => unsigned(array, key, check),
    }
}
fn unsigned(array: &dyn Array, key: &DataType, check: Check<'_>) -> Result<Graph> {
    match key {
        DataType::UInt8 => dict::<UInt8Type>(array, check),
        DataType::UInt16 => dict::<UInt16Type>(array, check),
        DataType::UInt32 => dict::<UInt32Type>(array, check),
        DataType::UInt64 => dict::<UInt64Type>(array, check),
        _ => Err(invalid()),
    }
}
fn dict<K: ArrowDictionaryKeyType>(array: &dyn Array, check: Check<'_>) -> Result<Graph> {
    let array = typed::<DictionaryArray<K>>(array)?;
    let child = super::graph(array.values().as_ref(), 0, array.values().len(), check)?;
    finish(array.data_type(), 1, 1, child, check)
}
fn run(array: &dyn Array, ends: &DataType, check: Check<'_>) -> Result<Graph> {
    match ends {
        DataType::Int16 => run_values::<Int16Type>(array, check),
        DataType::Int32 => run_values::<Int32Type>(array, check),
        DataType::Int64 => run_values::<Int64Type>(array, check),
        _ => Err(invalid()),
    }
}
fn run_values<R: RunEndIndexType>(array: &dyn Array, check: Check<'_>) -> Result<Graph> {
    let array = typed::<RunArray<R>>(array)?;
    let mut child = super::graph(array.values().as_ref(), 0, array.values().len(), check)?;
    let ends = super::super::descriptor::metadata(&R::DATA_TYPE, 1, 0, check)?;
    child.metadata = super::super::add(child.metadata, ends)?;
    // Validation clones the run-end ArrayData, including its one-buffer vector.
    child.cost = child.cost.extra(super::super::add(ends, ends)?)?;
    finish(array.data_type(), 0, 2, child, check)
}
