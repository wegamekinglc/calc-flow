use super::super::{single, workspace};
use super::{At, Check, Cost, Piece, Result, invalid, node};
use datafusion::arrow::{
    array::{Array, RunArray, downcast_run_array},
    datatypes::{DataType, RunEndIndexType},
};
#[path = "run/storage.rs"]
mod storage;

pub(super) fn plan(
    dtype: &DataType,
    count: usize,
    rows: u64,
    at: &mut At<'_>,
    check: Check<'_>,
) -> Result<Cost> {
    let DataType::RunEndEncoded(ends, values) = dtype else {
        return Err(invalid("normal run datatype"));
    };
    let width = end_width(ends.data_type(), rows)?;
    let active = active(count, at, check)?;
    let mut slices = |index| physical(at(active[index])?);
    let values = node(values.data_type(), active.len(), &mut slices, check)?;
    let own = storage::plan(width, active.len(), &mut slices, check)?;
    values.combine(own)
}
fn end_width(dtype: &DataType, rows: u64) -> Result<u64> {
    let width = match dtype {
        DataType::Int16 => 2,
        DataType::Int32 => 4,
        DataType::Int64 => 8,
        _ => return Err(invalid("normal run end type")),
    };
    super::super::producer::key_index(dtype, rows)?;
    Ok(width)
}
fn active(count: usize, at: &mut At<'_>, check: Check<'_>) -> Result<Vec<usize>> {
    let mut active = Vec::with_capacity(count);
    for index in 0..count {
        check()?;
        if at(index)?.len != 0 {
            active.push(index);
        }
    }
    if active.is_empty() {
        return Err(invalid("empty normal run concat"));
    }
    Ok(active)
}
pub(in crate::operator::rolling::indicators::scalar_output_plan) fn physical(
    piece: Piece<'_>,
) -> Result<Piece<'_>> {
    let Some(array) = piece.array else {
        return Ok(Piece::null(usize::from(piece.len != 0)));
    };
    downcast_run_array!(array=>typed_physical(array,piece.start,piece.len),_=>Err(invalid("normal run array")))
}
fn typed_physical<R: RunEndIndexType>(
    array: &RunArray<R>,
    start: usize,
    len: usize,
) -> Result<Piece<'_>> {
    let end = start
        .checked_add(len)
        .filter(|end| *end <= array.len())
        .ok_or_else(|| invalid("normal run range"))?;
    let first = array.get_physical_index(start);
    let last = array.get_physical_index(end - 1);
    Ok(Piece {
        array: Some(array.values().as_ref()),
        start: first,
        len: last - first + 1,
    })
}

pub(super) fn single(dtype: &DataType, piece: Piece<'_>, check: Check<'_>) -> Result<Cost> {
    if let Some(array) = piece.array {
        return Ok(Cost {
            owned: single::sliced(array, check)?,
            temporary: workspace::array_data_workspace(array, check)?,
        });
    }
    let factory =
        super::super::super::scalar_materialization_plan::null_array_cost(dtype, piece.len, check)?;
    Ok(Cost {
        owned: factory.logical_owned,
        temporary: factory.construction_peak,
    })
}
