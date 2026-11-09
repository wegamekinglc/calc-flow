use super::super::{
    Check, Result,
    math::{add, bytes, size},
};
use datafusion::arrow::datatypes::DataType;

pub(super) fn plan(dtype: &DataType, rows: usize, check: Check<'_>) -> Result<u64> {
    check()?;
    match dtype {
        DataType::RunEndEncoded(_, value) => add(
            bytes::<usize>(size(rows)?)?,
            plan(value.data_type(), rows, check)?,
        ),
        DataType::Dictionary(_, value) => plan(value, rows, check),
        _ => normal(dtype, rows, check),
    }
}
fn normal(dtype: &DataType, rows: usize, check: Check<'_>) -> Result<u64> {
    check()?;
    match dtype {
        DataType::RunEndEncoded(_, value) => add(
            bytes::<usize>(size(rows)?)?,
            normal(value.data_type(), rows, check)?,
        ),
        DataType::Struct(fields) => structure(fields, rows, check),
        _ => lists(dtype, rows, check),
    }
}
fn structure(
    fields: &datafusion::arrow::datatypes::Fields,
    rows: usize,
    check: Check<'_>,
) -> Result<u64> {
    let mut largest = 0;
    for field in fields {
        largest = largest.max(normal(field.data_type(), rows, check)?);
    }
    Ok(largest)
}
fn lists(dtype: &DataType, rows: usize, check: Check<'_>) -> Result<u64> {
    match dtype {
        DataType::List(field)
        | DataType::LargeList(field)
        | DataType::ListView(field)
        | DataType::LargeListView(field)
        | DataType::Map(field, _) => normal(field.data_type(), rows, check),
        _ => Ok(0),
    }
}
