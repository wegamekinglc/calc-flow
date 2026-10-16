use std::collections::{BTreeMap, btree_map::Entry};

use datafusion::arrow::{
    array::{DictionaryArray, PrimitiveArray, new_empty_array},
    datatypes::{
        ArrowDictionaryKeyType, ArrowNativeType, Int8Type, Int16Type, Int32Type, Int64Type,
        UInt8Type, UInt16Type, UInt32Type, UInt64Type,
    },
};

use super::super::StrideStop;
use super::{
    Arc, Array, ArrayData, ArrayRef, DataType, Funding, MemoryReservation, Result, accounting, add,
    invalid, product, reserve, sum,
};
use crate::operator::join::{checkpoint_v2::history, columnar::SelectedRow};

#[cfg(test)]
pub(super) mod tests;

pub(super) fn copy(
    data: &ArrayData,
    key: &DataType,
    workspace: &MemoryReservation,
    funding: &Arc<Funding>,
    check: &dyn Fn() -> Result<()>,
) -> Result<ArrayData> {
    match key {
        DataType::Int8 => typed::<Int8Type>(data, workspace, funding, check),
        DataType::Int16 => typed::<Int16Type>(data, workspace, funding, check),
        DataType::Int32 => typed::<Int32Type>(data, workspace, funding, check),
        DataType::Int64 => typed::<Int64Type>(data, workspace, funding, check),
        _ => unsigned(data, key, workspace, funding, check),
    }
}

fn unsigned(
    data: &ArrayData,
    key: &DataType,
    workspace: &MemoryReservation,
    funding: &Arc<Funding>,
    check: &dyn Fn() -> Result<()>,
) -> Result<ArrayData> {
    match key {
        DataType::UInt8 => typed::<UInt8Type>(data, workspace, funding, check),
        DataType::UInt16 => typed::<UInt16Type>(data, workspace, funding, check),
        DataType::UInt32 => typed::<UInt32Type>(data, workspace, funding, check),
        DataType::UInt64 => typed::<UInt64Type>(data, workspace, funding, check),
        _ => Err(invalid("retained dictionary key type is invalid")),
    }
}

fn typed<K: ArrowDictionaryKeyType>(
    data: &ArrayData,
    workspace: &MemoryReservation,
    funding: &Arc<Funding>,
    check: &dyn Fn() -> Result<()>,
) -> Result<ArrayData> {
    admit::<K>(data.len(), workspace, funding)?;
    let dictionary = DictionaryArray::<K>::from(data.clone());
    let mut mapping = BTreeMap::new();
    let mut selected = Vec::with_capacity(data.len());
    let mut keys = Vec::with_capacity(data.len());
    let mut stop = StrideStop::new(check);
    for key in dictionary.keys() {
        stop.visit()?;
        keys.push(
            key.map(|key| remap::<K>(key, &mut mapping, &mut selected))
                .transpose()?,
        );
    }
    let values = selected_values(dictionary.values(), &selected, workspace, funding, check)?;
    let keys = keys.into_iter().collect::<PrimitiveArray<K>>();
    check()?;
    DictionaryArray::<K>::try_new(keys, values)
        .map(|array| array.to_data())
        .map_err(|_| invalid("retained dictionary reconstruction failed"))
}

fn admit<K: ArrowDictionaryKeyType>(
    rows: usize,
    workspace: &MemoryReservation,
    funding: &Arc<Funding>,
) -> Result<()> {
    reserve(workspace, workspace_bytes::<K>(rows)?)?;
    funding.grow(add(
        accounting::vector_peak::<K::Native>(rows)?,
        validity_bytes(rows)?,
    )?)
}

fn workspace_bytes<K: ArrowDictionaryKeyType>(rows: usize) -> Result<usize> {
    sum(&[
        history::tree::<usize, K::Native>(rows)
            .ok_or_else(|| invalid("retained dictionary mapping credit overflow"))?,
        accounting::vector_peak::<SelectedRow>(rows)?,
        accounting::vector_peak::<Option<K::Native>>(rows)?,
        accounting::array_controls(2, 3)?,
    ])
}

fn validity_bytes(rows: usize) -> Result<usize> {
    let bytes = add(rows, 7)? / 8;
    // BooleanBufferBuilder allocates in 64-byte blocks, including short bitmaps.
    product(product(add(bytes, 63)? / 64, 64)?, 3)
}

fn remap<K: ArrowDictionaryKeyType>(
    key: K::Native,
    mapping: &mut BTreeMap<usize, K::Native>,
    selected: &mut Vec<SelectedRow>,
) -> Result<K::Native> {
    match mapping.entry(key.as_usize()) {
        Entry::Occupied(entry) => Ok(*entry.get()),
        Entry::Vacant(entry) => {
            let index = K::Native::from_usize(selected.len())
                .ok_or_else(|| invalid("retained dictionary key range overflow"))?;
            selected.push(SelectedRow {
                source: key.as_usize(),
                row_id: 0,
                time: crate::EventTime::from_micros(0),
            });
            Ok(*entry.insert(index))
        }
    }
}

fn selected_values(
    values: &ArrayRef,
    selected: &[SelectedRow],
    workspace: &MemoryReservation,
    funding: &Arc<Funding>,
    check: &dyn Fn() -> Result<()>,
) -> Result<ArrayRef> {
    if selected.is_empty() {
        return empty_values(values.data_type(), workspace, funding, check);
    }
    super::super::copy_column(values, selected.into(), workspace, funding, check)
}

fn empty_values(
    data_type: &DataType,
    workspace: &MemoryReservation,
    funding: &Arc<Funding>,
    check: &dyn Fn() -> Result<()>,
) -> Result<ArrayRef> {
    let (controls, backing) = empty_credit(data_type)?;
    reserve(workspace, controls)?;
    funding.grow(backing)?;
    check()?;
    Ok(new_empty_array(data_type))
}

fn empty_credit(data_type: &DataType) -> Result<(usize, usize)> {
    let nodes = accounting::shape_nodes(data_type)?;
    let buffers = product(nodes, 3)?;
    let controls = accounting::array_controls(nodes, buffers)?;
    // Empty Utf8/List/Map offsets still allocate one aligned buffer per node.
    let offsets = product(nodes, 64 * 3)?;
    Ok((controls, add(controls, offsets)?))
}
