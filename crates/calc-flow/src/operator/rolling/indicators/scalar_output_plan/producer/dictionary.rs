use super::{Check, Result, invalid};
use datafusion::{
    arrow::{
        array::{ArrayRef, DictionaryArray, PrimitiveArray},
        datatypes::{
            ArrowDictionaryKeyType, DataType, Int8Type, Int16Type, Int32Type, Int64Type, UInt8Type,
            UInt16Type, UInt32Type, UInt64Type,
        },
    },
    common::{ScalarValue, scalar::dict_from_values},
};
use std::sync::Arc;

#[derive(Clone, Copy)]
enum Keys {
    Sequence,
    Single(bool),
}
pub(super) fn column(
    values: Vec<ScalarValue>,
    key: &DataType,
    check: Check<'_>,
) -> Result<ArrayRef> {
    let mut inner = Vec::with_capacity(values.len());
    for value in values {
        check()?;
        match value {
            ScalarValue::Dictionary(actual, value) if actual.as_ref() == key => inner.push(*value),
            _ => return Err(invalid("dictionary scalar type")),
        }
    }
    wrap(super::column(inner, check)?, key, Keys::Sequence)
}
pub(super) fn single(key: &DataType, value: &ScalarValue, check: Check<'_>) -> Result<ArrayRef> {
    wrap(
        super::single(value, check)?,
        key,
        Keys::Single(value.is_null()),
    )
}
fn wrap(values: ArrayRef, key: &DataType, keys: Keys) -> Result<ArrayRef> {
    match key {
        DataType::Int8 => typed::<Int8Type>(values, keys),
        DataType::Int16 => typed::<Int16Type>(values, keys),
        DataType::Int32 => typed::<Int32Type>(values, keys),
        DataType::Int64 => typed::<Int64Type>(values, keys),
        _ => unsigned(values, key, keys),
    }
}
fn unsigned(values: ArrayRef, key: &DataType, keys: Keys) -> Result<ArrayRef> {
    match key {
        DataType::UInt8 => typed::<UInt8Type>(values, keys),
        DataType::UInt16 => typed::<UInt16Type>(values, keys),
        DataType::UInt32 => typed::<UInt32Type>(values, keys),
        DataType::UInt64 => typed::<UInt64Type>(values, keys),
        _ => Err(invalid("dictionary key type")),
    }
}
fn typed<K: ArrowDictionaryKeyType>(values: ArrayRef, keys: Keys) -> Result<ArrayRef> {
    let Keys::Single(is_null) = keys else {
        return dict_from_values::<K>(values).map_err(|_| invalid("dictionary key index"));
    };
    let key = (!is_null).then(K::default_value);
    let keys = std::iter::once(key).collect::<PrimitiveArray<K>>();
    DictionaryArray::<K>::try_new(keys, values)
        .map(|array| Arc::new(array) as ArrayRef)
        .map_err(|_| invalid("dictionary child"))
}
