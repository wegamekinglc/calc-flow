use super::{
    Check, OutputPlan, Result, invalid,
    math::{add, bitmap, buffer_headers, bytes, grow, scale, size, sum},
    single,
};
use datafusion::{
    arrow::{
        array::{Array, ArrayRef},
        datatypes::DataType,
    },
    common::ScalarValue,
};
#[path = "concatenated/leaves.rs"]
mod leaves;
#[path = "concatenated/lists.rs"]
mod lists;
#[path = "concatenated/run.rs"]
mod run;
pub(super) use lists::{full as view_values, selected as list_piece};
pub(super) use run::physical as run_piece;
pub(super) use structures::{column as struct_piece, selected as map_piece};
#[path = "concatenated/structures.rs"]
mod structures;

#[derive(Clone, Copy)]
pub(super) struct Piece<'a> {
    pub(super) array: Option<&'a dyn Array>,
    pub(super) start: usize,
    pub(super) len: usize,
}
impl Piece<'_> {
    pub(super) fn null(len: usize) -> Self {
        Self {
            array: None,
            start: 0,
            len,
        }
    }
}
pub(super) type At<'a> = dyn FnMut(usize) -> Result<Piece<'a>> + 'a;
#[derive(Clone, Copy, Default)]
pub(super) struct Cost {
    pub(super) owned: u64,
    pub(super) temporary: u64,
}
impl Cost {
    fn combine(self, other: Self) -> Result<Self> {
        Ok(Self {
            owned: add(self.owned, other.owned)?,
            temporary: add(self.temporary, other.temporary)?,
        })
    }
}
pub(super) fn plan<'a>(
    dtype: &DataType,
    rows: usize,
    cell: &mut dyn FnMut(usize) -> Option<&'a ScalarValue>,
    check: Check<'_>,
) -> Result<Option<OutputPlan>> {
    if rows < 2 || !array_backed(dtype) {
        return Ok(None);
    }
    build(dtype, rows, cell, check).map(Some)
}
fn build<'a>(
    dtype: &DataType,
    rows: usize,
    cell: &mut dyn FnMut(usize) -> Option<&'a ScalarValue>,
    check: Check<'_>,
) -> Result<OutputPlan> {
    let mut at = |index| root(dtype, cell(index));
    let cost = node(dtype, rows, &mut at, check)?;
    let empty = super::super::scalar_materialization_plan::null_array_cost(dtype, 0, check)?;
    Ok(OutputPlan {
        output_bytes: add(empty.logical_owned, cost.owned)?,
        scratch_bytes: add(
            empty.construction_peak,
            root_workspace(rows, cost.temporary)?,
        )?,
    })
}
fn root_workspace(rows: usize, temporary: u64) -> Result<u64> {
    let count = size(rows)?;
    sum([
        Ok(temporary),
        bytes::<ScalarValue>(count),
        grow(count, size(size_of::<ArrayRef>())?),
        grow(count, size(size_of::<&dyn Array>())?),
    ])
}
fn root<'a>(dtype: &DataType, value: Option<&'a ScalarValue>) -> Result<Piece<'a>> {
    let Some(value) = value else {
        return Ok(Piece::null(1));
    };
    let array = single::array(value).ok_or_else(|| invalid("nested scalar"))?;
    if array.len() != 1 || array.data_type() != dtype {
        return Err(invalid("nested scalar shape"));
    }
    Ok(Piece {
        array: Some(array),
        start: 0,
        len: 1,
    })
}
fn array_backed(dtype: &DataType) -> bool {
    match dtype {
        DataType::List(_)
        | DataType::LargeList(_)
        | DataType::ListView(_)
        | DataType::LargeListView(_)
        | DataType::FixedSizeList(..) => true,
        _ => matches!(dtype, DataType::Struct(_) | DataType::Map(..)),
    }
}
pub(super) fn node(
    dtype: &DataType,
    count: usize,
    at: &mut At<'_>,
    check: Check<'_>,
) -> Result<Cost> {
    check()?;
    if count == 1 {
        return run::single(dtype, at(0)?, check);
    }
    copied_node(dtype, count, at, check)
}
fn copied_node(dtype: &DataType, count: usize, at: &mut At<'_>, check: Check<'_>) -> Result<Cost> {
    let (rows, nulls) = geometry(count, at, check)?;
    let data = if let Some(cost) = leaves::plan(dtype, count, rows, at, check)? {
        cost
    } else {
        lists::plan(dtype, count, rows, at, check)?
    };
    let nullable = needs_mask(dtype, nulls);
    data.combine(workspace(rows, count, nullable)?)
}
fn needs_mask(dtype: &DataType, nulls: bool) -> bool {
    nulls
        && !matches!(
            dtype,
            DataType::Null | DataType::RunEndEncoded(..) | DataType::Union(..)
        )
}
fn workspace(rows: u64, count: usize, nulls: bool) -> Result<Cost> {
    let mask = if nulls { scale(bitmap(rows), 2)? } else { 0 };
    let vectors = grow(size(count)?, size(size_of::<&dyn Array>())?)?;
    Ok(Cost {
        owned: mask,
        temporary: sum([Ok(mask), Ok(vectors), buffer_headers(3), bitmap(rows)])?,
    })
}

fn geometry(count: usize, at: &mut At<'_>, check: Check<'_>) -> Result<(u64, bool)> {
    let mut rows = 0;
    let mut nulls = false;
    for index in 0..count {
        check()?;
        let piece = at(index)?;
        rows = add(rows, size(piece.len)?)?;
        nulls |= piece
            .array
            .map_or(piece.len != 0, |array| array.null_count() != 0);
    }
    Ok((rows, nulls))
}
fn typed<T: Array + 'static>(value: &dyn Array) -> Result<&T> {
    value
        .as_any()
        .downcast_ref()
        .ok_or_else(|| invalid("array representation"))
}
fn slice(array: &dyn Array, start: usize, len: usize) -> Result<Piece<'_>> {
    if start.checked_add(len).is_none_or(|end| end > array.len()) {
        return Err(invalid("nested slice range"));
    }
    Ok(Piece {
        array: Some(array),
        start,
        len,
    })
}
