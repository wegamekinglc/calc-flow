use super::cost::{Cost, times};
use super::descriptor;
use super::{Check, Result, invalid, typed};
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

#[derive(Clone, Copy)]
enum Mode {
    Materialize,
    Clone,
}
impl Mode {
    fn child(self, array: &dyn Array, row: usize, check: Check<'_>) -> Result<Cost> {
        match self {
            Self::Materialize => super::cell(array, row, check),
            Self::Clone => super::cloned::cell(array, row, check).map(Cost::owned),
        }
    }
    fn missing(self, dtype: &DataType, check: Check<'_>) -> Result<Cost> {
        match self {
            Self::Materialize => super::null::scalar(dtype, check),
            Self::Clone => super::cloned::null(dtype, check).map(Cost::owned),
        }
    }
}
pub(super) fn cell(array: &dyn Array, row: usize, check: Check<'_>) -> Result<Cost> {
    selected(array, row, Mode::Materialize, check)
}
pub(super) fn clone_cell(array: &dyn Array, row: usize, check: Check<'_>) -> Result<Cost> {
    selected(array, row, Mode::Clone, check)
}
fn selected(array: &dyn Array, row: usize, mode: Mode, check: Check<'_>) -> Result<Cost> {
    check()?;
    match array.data_type() {
        DataType::Dictionary(key, _) => dictionary(array, key, row, mode, check),
        DataType::RunEndEncoded(ends, _) => run(array, ends.data_type(), row, mode, check),
        DataType::Union(..) => union(array, row, mode, check),
        _ => Err(invalid()),
    }
}
fn dictionary(
    array: &dyn Array,
    key: &DataType,
    row: usize,
    mode: Mode,
    check: Check<'_>,
) -> Result<Cost> {
    match key {
        DataType::Int8 => dict::<Int8Type>(array, row, mode, check),
        DataType::Int16 => dict::<Int16Type>(array, row, mode, check),
        DataType::Int32 => dict::<Int32Type>(array, row, mode, check),
        DataType::Int64 => dict::<Int64Type>(array, row, mode, check),
        _ => unsigned(array, key, row, mode, check),
    }
}
fn unsigned(
    array: &dyn Array,
    key: &DataType,
    row: usize,
    mode: Mode,
    check: Check<'_>,
) -> Result<Cost> {
    match key {
        DataType::UInt8 => dict::<UInt8Type>(array, row, mode, check),
        DataType::UInt16 => dict::<UInt16Type>(array, row, mode, check),
        DataType::UInt32 => dict::<UInt32Type>(array, row, mode, check),
        DataType::UInt64 => dict::<UInt64Type>(array, row, mode, check),
        _ => Err(invalid()),
    }
}
fn dict<K: ArrowDictionaryKeyType>(
    array: &dyn Array,
    row: usize,
    mode: Mode,
    check: Check<'_>,
) -> Result<Cost> {
    let array = typed::<DictionaryArray<K>>(array)?;
    let inner = if let Some(index) = array.key(row) {
        mode.child(array.values().as_ref(), index, check)?
    } else {
        mode.missing(array.values().data_type(), check)?
    };
    inner.combine(dictionary_box(&K::DATA_TYPE, check)?)
}
pub(super) fn dictionary_box(key: &DataType, check: Check<'_>) -> Result<Cost> {
    let index = super::add(times::<DataType>(1)?, descriptor::dtype(key, check)?)?;
    Ok(Cost::owned(super::add(index, times::<ScalarValue>(1)?)?))
}
fn run(
    array: &dyn Array,
    ends: &DataType,
    row: usize,
    mode: Mode,
    check: Check<'_>,
) -> Result<Cost> {
    match ends {
        DataType::Int16 => run_value::<Int16Type>(array, row, mode, check),
        DataType::Int32 => run_value::<Int32Type>(array, row, mode, check),
        DataType::Int64 => run_value::<Int64Type>(array, row, mode, check),
        _ => Err(invalid()),
    }
}
fn run_value<R: RunEndIndexType>(
    array: &dyn Array,
    row: usize,
    mode: Mode,
    check: Check<'_>,
) -> Result<Cost> {
    let array = typed::<RunArray<R>>(array)?;
    let inner = mode.child(
        array.values().as_ref(),
        array.get_physical_index(row),
        check,
    )?;
    inner.combine(Cost::owned(times::<ScalarValue>(1)?))
}
fn union(array: &dyn Array, row: usize, mode: Mode, check: Check<'_>) -> Result<Cost> {
    let array = typed::<UnionArray>(array)?;
    let inner = mode.child(
        array.child(array.type_id(row)).as_ref(),
        array.value_offset(row),
        check,
    )?;
    inner.combine(Cost::owned(times::<ScalarValue>(1)?))
}
