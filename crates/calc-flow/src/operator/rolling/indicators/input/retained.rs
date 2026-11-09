use datafusion::arrow::{
    array::{Array, ArrayRef, DictionaryArray, RunArray, UnionArray},
    datatypes::{
        ArrowDictionaryKeyType, DataType, Int16Type, Int32Type, Int64Type, RunEndIndexType,
    },
};

use super::super::footprint::Footprint;
use super::{Check, Result, TableBatch, overflow};

pub(in super::super) fn plan(
    table: &TableBatch,
    node: &str,
    check: Check<'_>,
) -> Result<Footprint> {
    let mut footprint = Footprint::default();
    for record in table.batches() {
        for column in record.columns() {
            column_plan(&mut footprint, column, node, check)?;
        }
    }
    Ok(footprint)
}
pub(super) fn plan_selected<'a>(
    rows: impl IntoIterator<Item = (&'a [ArrayRef], usize)>,
    node: &str,
    check: Check<'_>,
) -> Result<Footprint> {
    let mut footprint = Footprint::default();
    for (columns, row) in rows {
        for column in columns {
            check()?;
            selected(&mut footprint, column, row, node, check)?;
        }
    }
    Ok(footprint)
}

fn column_plan(
    footprint: &mut Footprint,
    array: &ArrayRef,
    node: &str,
    check: Check<'_>,
) -> Result<()> {
    check()?;
    if array.is_empty() {
        return Ok(());
    }
    if nested(array.data_type()) {
        return footprint.backing(array, node, check);
    }
    for row in 0..array.len() {
        check()?;
        selected(footprint, array, row, node, check)?;
    }
    Ok(())
}
fn nested(dtype: &DataType) -> bool {
    matches!(
        dtype,
        DataType::List(_)
            | DataType::LargeList(_)
            | DataType::FixedSizeList(..)
            | DataType::ListView(_)
            | DataType::LargeListView(_)
            | DataType::Struct(_)
            | DataType::Map(..)
    )
}
fn selected(
    footprint: &mut Footprint,
    array: &ArrayRef,
    row: usize,
    node: &str,
    check: Check<'_>,
) -> Result<()> {
    check()?;
    if nested(array.data_type()) {
        return footprint.backing(array, node, check);
    }
    match array.data_type() {
        DataType::Dictionary(key, _) => dictionary(footprint, array, key, row, node, check),
        DataType::RunEndEncoded(ends, _) => {
            run(footprint, array, ends.data_type(), row, node, check)
        }
        DataType::Union(..) => union(footprint, array, row, node, check),
        _ => Ok(()),
    }
}
macro_rules! dictionary {
    ($ty:ty, $footprint:ident, $array:ident, $row:ident, $node:ident, $check:ident) => {
        dict::<$ty>($footprint, $array, $row, $node, $check)
    };
}
fn dictionary(
    footprint: &mut Footprint,
    array: &ArrayRef,
    key: &DataType,
    row: usize,
    node: &str,
    check: Check<'_>,
) -> Result<()> {
    datafusion::arrow::array::downcast_integer! {
        key => (dictionary, footprint, array, row, node, check),
        _ => Err(overflow(node))
    }
}
fn dict<K: ArrowDictionaryKeyType>(
    footprint: &mut Footprint,
    array: &ArrayRef,
    row: usize,
    node: &str,
    check: Check<'_>,
) -> Result<()> {
    let array = typed::<DictionaryArray<K>>(array.as_ref(), node)?;
    if let Some(index) = array.key(row) {
        selected(footprint, array.values(), index, node, check)?;
    }
    Ok(())
}
fn run(
    footprint: &mut Footprint,
    array: &ArrayRef,
    ends: &DataType,
    row: usize,
    node: &str,
    check: Check<'_>,
) -> Result<()> {
    match ends {
        DataType::Int16 => run_value::<Int16Type>(footprint, array, row, node, check),
        DataType::Int32 => run_value::<Int32Type>(footprint, array, row, node, check),
        DataType::Int64 => run_value::<Int64Type>(footprint, array, row, node, check),
        _ => Err(overflow(node)),
    }
}
fn run_value<R: RunEndIndexType>(
    footprint: &mut Footprint,
    array: &ArrayRef,
    row: usize,
    node: &str,
    check: Check<'_>,
) -> Result<()> {
    let array = typed::<RunArray<R>>(array.as_ref(), node)?;
    selected(
        footprint,
        array.values(),
        array.get_physical_index(row),
        node,
        check,
    )
}
fn union(
    footprint: &mut Footprint,
    array: &ArrayRef,
    row: usize,
    node: &str,
    check: Check<'_>,
) -> Result<()> {
    let array = typed::<UnionArray>(array.as_ref(), node)?;
    selected(
        footprint,
        array.child(array.type_id(row)),
        array.value_offset(row),
        node,
        check,
    )
}
fn typed<'a, T: Array + 'static>(array: &'a dyn Array, node: &str) -> Result<&'a T> {
    array.as_any().downcast_ref().ok_or_else(|| overflow(node))
}
