//! Typed string admission: encode distinct keys once per batch.

use super::{EncodedColumns, encode_row_columns};
use crate::{CalcFlowError, Result};
use datafusion::arrow::{
    array::{
        Array, ArrayRef, GenericStringArray, LargeStringArray, OffsetSizeTrait, StringArray,
        UInt32Array,
    },
    compute::take,
};
use hashbrown::HashTable;

pub(super) fn encode<R>(
    column: &ArrayRef,
    reserve: impl FnOnce(u64) -> Result<R>,
) -> Result<Option<EncodedColumns>> {
    if column.null_count() > 0 {
        return Ok(None);
    }
    if let Some(column) = column.as_any().downcast_ref::<StringArray>() {
        return dictionary(column, reserve).map(Some);
    }
    if let Some(column) = column.as_any().downcast_ref::<LargeStringArray>() {
        return dictionary(column, reserve).map(Some);
    }
    Ok(None)
}

fn dictionary<O: OffsetSizeTrait, R>(
    column: &GenericStringArray<O>,
    reserve: impl FnOnce(u64) -> Result<R>,
) -> Result<EncodedColumns> {
    let seed = ahash::RandomState::new();
    let mut index = HashTable::<u32>::with_capacity(column.len());
    let mut unique = Vec::<u32>::new();
    let mut ids = Vec::with_capacity(column.len());
    for row in 0..column.len() {
        let value = column.value(row);
        let hash = seed.hash_one(value);
        let id = if let Some(id) = index.find(hash, |id| {
            column.value(unique[*id as usize] as usize) == value
        }) {
            *id
        } else {
            let id = checked_id(unique.len())?;
            unique.push(checked_id(row)?);
            index.insert_unique(hash, id, |id| {
                seed.hash_one(column.value(unique[*id as usize] as usize))
            });
            id
        };
        ids.push(id);
    }
    // Arrow 58.3 take_bytes allocates the sum of selected value lengths once.
    // Hold that additional buffer reservation while RowConverter's canonical
    // rows are live; fixed admission headroom already covers ids and offsets.
    encode_distinct(column, unique, ids, reserve)
}

fn encode_distinct<O: OffsetSizeTrait, R>(
    column: &GenericStringArray<O>,
    unique: Vec<u32>,
    ids: Vec<u32>,
    reserve: impl FnOnce(u64) -> Result<R>,
) -> Result<EncodedColumns> {
    let value_bytes = unique
        .iter()
        .map(|row| column.value(*row as usize).len() as u64)
        .sum();
    let _workspace = reserve(value_bytes)?;
    let values = take(column, &UInt32Array::from(unique), None)
        .map_err(|error| super::super::arrow_error(&error))?;
    let (EncodedColumns::Rows(rows) | EncodedColumns::Binary(rows)) =
        encode_row_columns(&[values])?
    else {
        unreachable!("string dictionary uses canonical Arrow rows")
    };
    Ok(EncodedColumns::StringKeys { rows, ids })
}

fn checked_id(value: usize) -> Result<u32> {
    u32::try_from(value).map_err(|_| CalcFlowError::Format {
        message: "ASOF typed string dictionary exceeds the u32 row domain".into(),
    })
}
