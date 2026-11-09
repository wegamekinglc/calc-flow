use super::super::math::{add, retained, vector};
use super::{At, Check, Cost, Piece, Result, invalid, node, slice, typed};
use datafusion::arrow::{
    array::{MapArray, StructArray},
    datatypes::{DataType, Fields},
};

pub(super) fn structure(
    fields: &Fields,
    count: usize,
    at: &mut At<'_>,
    check: Check<'_>,
) -> Result<Cost> {
    let mut total = Cost::default();
    for (index, field) in fields.iter().enumerate() {
        let mut child = |ordinal| column(at(ordinal)?, index);
        total = total.combine(node(field.data_type(), count, &mut child, check)?)?;
    }
    let vector = vector::<datafusion::arrow::array::ArrayRef>(fields.len())?;
    total.combine(Cost {
        owned: 0,
        temporary: vector,
    })
}
pub(in crate::operator::rolling::indicators::scalar_output_plan) fn column(
    piece: Piece<'_>,
    index: usize,
) -> Result<Piece<'_>> {
    let Some(array) = piece.array else {
        return Ok(Piece::null(piece.len));
    };
    let array = typed::<StructArray>(array)?;
    slice(array.column(index).as_ref(), piece.start, piece.len)
}
pub(super) fn map(
    dtype: &DataType,
    count: usize,
    rows: u64,
    at: &mut At<'_>,
    check: Check<'_>,
) -> Result<Cost> {
    super::super::normal::map_length(count, at, check)?;
    let mut child = |index| selected(at(index)?);
    let total = node(dtype, count, &mut child, check)?;
    let mut local = offsets(rows, count)?;
    local.temporary = add(local.temporary, slice_workspace(dtype, count, at, check)?)?;
    total.combine(local)
}
fn offsets(rows: u64, count: usize) -> Result<Cost> {
    let buffer = retained(add(rows, 1)?, 4)?;
    Ok(Cost {
        owned: buffer,
        temporary: add(buffer, vector::<datafusion::arrow::array::ArrayRef>(count)?)?,
    })
}

fn slice_workspace(
    dtype: &DataType,
    count: usize,
    at: &mut At<'_>,
    check: Check<'_>,
) -> Result<u64> {
    let mut total = 0;
    for index in 0..count {
        check()?;
        total = add(total, slice_cost(dtype, selected(at(index)?)?, check)?)?;
    }
    Ok(total)
}
fn slice_cost(dtype: &DataType, piece: Piece<'_>, check: Check<'_>) -> Result<u64> {
    match piece.array {
        Some(array) => super::super::single::sliced(array, check),
        None => super::super::super::scalar_materialization_plan::null_array_requirement(
            dtype, 0, check,
        ),
    }
}
pub(in crate::operator::rolling::indicators::scalar_output_plan) fn selected(
    piece: Piece<'_>,
) -> Result<Piece<'_>> {
    let Some(array) = piece.array else {
        return Ok(Piece::null(0));
    };
    let array = typed::<MapArray>(array)?;
    let offsets = array.value_offsets();
    let start = usize::try_from(offsets[piece.start]).map_err(|_| invalid("map offset"))?;
    let end =
        usize::try_from(offsets[piece.start + piece.len]).map_err(|_| invalid("map offset"))?;
    slice(
        array.entries(),
        start,
        end.checked_sub(start).ok_or_else(|| invalid("map range"))?,
    )
}
