use super::super::super::{
    concatenated::{self, Piece},
    invalid, single,
};
use super::{At, Check, Result, geometry, source};
use datafusion::{arrow::datatypes::DataType, common::ScalarValue};
type Select = for<'a> fn(Piece<'a>) -> Result<Piece<'a>>;

pub(in crate::operator::rolling::indicators::scalar_output_plan) fn plan<'a>(
    dtype: &DataType,
    rows: usize,
    cell: &mut dyn FnMut(usize) -> Option<&'a ScalarValue>,
    check: Check<'_>,
) -> Result<u64> {
    let dtype = leaf_type(dtype);
    if rows < 2 || !array_backed(dtype) {
        return Ok(0);
    }
    let mut at = |index| root(dtype, cell(index)).map(Some);
    walk(dtype, rows, &mut at, check)
}
fn leaf_type(dtype: &DataType) -> &DataType {
    match dtype {
        DataType::Dictionary(_, value) => leaf_type(value),
        DataType::RunEndEncoded(_, value) => leaf_type(value.data_type()),
        _ => dtype,
    }
}
fn leaf_value(value: &ScalarValue) -> &ScalarValue {
    match value {
        ScalarValue::Dictionary(_, value) | ScalarValue::RunEndEncoded(_, _, value) => {
            leaf_value(value)
        }
        _ => value,
    }
}
fn array_backed(dtype: &DataType) -> bool {
    match dtype {
        DataType::List(_)
        | DataType::LargeList(_)
        | DataType::ListView(_)
        | DataType::LargeListView(_) => true,
        _ => matches!(dtype, DataType::Map(..) | DataType::Struct(_)),
    }
}
fn root<'a>(dtype: &DataType, value: Option<&'a ScalarValue>) -> Result<Piece<'a>> {
    let Some(value) = value else {
        return Ok(Piece::null(1));
    };
    let array = single::array(leaf_value(value))
        .ok_or_else(|| invalid("normal dictionary workspace scalar"))?;
    if array.len() != 1 || array.data_type() != dtype {
        return Err(invalid("normal dictionary workspace shape"));
    }
    Ok(Piece {
        array: Some(array),
        start: 0,
        len: 1,
    })
}
fn walk(dtype: &DataType, count: usize, at: &mut At<'_>, check: Check<'_>) -> Result<u64> {
    check()?;
    if super::super::constructors(count, at, check)? < 2 {
        return Ok(0);
    }
    match dtype {
        DataType::Dictionary(_, value) => dictionary(value, count, at, check),
        DataType::Struct(fields) => structure(fields, count, at, check),
        DataType::RunEndEncoded(_, values) => run(values.data_type(), count, at, check),
        _ => lists(dtype, count, at, check),
    }
}
fn dictionary(value: &DataType, count: usize, at: &mut At<'_>, check: Check<'_>) -> Result<u64> {
    if !source::mergeable(value) {
        return Ok(0);
    }
    let facts = geometry::facts(count, at, check)?;
    geometry::simulation(facts.references, facts.mask)
}
fn structure(
    fields: &datafusion::arrow::datatypes::Fields,
    count: usize,
    at: &mut At<'_>,
    check: Check<'_>,
) -> Result<u64> {
    let mut peak = 0;
    for (index, field) in fields.iter().enumerate() {
        let mut child = |source| {
            at(source)?
                .map(|piece| concatenated::struct_piece(piece, index))
                .transpose()
        };
        peak = peak.max(walk(field.data_type(), count, &mut child, check)?);
    }
    Ok(peak)
}
fn run(dtype: &DataType, count: usize, at: &mut At<'_>, check: Check<'_>) -> Result<u64> {
    let mut child = |index| {
        at(index)?
            .filter(|piece| piece.len != 0)
            .map(concatenated::run_piece)
            .transpose()
    };
    walk(dtype, count, &mut child, check)
}
fn lists(dtype: &DataType, count: usize, at: &mut At<'_>, check: Check<'_>) -> Result<u64> {
    match dtype {
        DataType::List(field) => child(
            field.data_type(),
            count,
            at,
            concatenated::list_piece::<i32>,
            check,
        ),
        DataType::LargeList(field) => child(
            field.data_type(),
            count,
            at,
            concatenated::list_piece::<i64>,
            check,
        ),
        DataType::ListView(field) => child(
            field.data_type(),
            count,
            at,
            concatenated::view_values::<i32>,
            check,
        ),
        DataType::LargeListView(field) => child(
            field.data_type(),
            count,
            at,
            concatenated::view_values::<i64>,
            check,
        ),
        DataType::Map(field, _) => {
            child(field.data_type(), count, at, concatenated::map_piece, check)
        }
        _ => Ok(0),
    }
}
fn child(
    dtype: &DataType,
    count: usize,
    at: &mut At<'_>,
    select: Select,
    check: Check<'_>,
) -> Result<u64> {
    let mut child = |index| at(index)?.map(select).transpose();
    walk(dtype, count, &mut child, check)
}
