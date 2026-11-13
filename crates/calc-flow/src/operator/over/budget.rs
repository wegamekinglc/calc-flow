use super::{AnalyticExpr, Declaration, spec::AnalyticKind};
use crate::{Result, operator::distinct::owned};
use datafusion::arrow::{
    array::{LargeStringArray, StringArray},
    datatypes::DataType,
};

pub(crate) fn declaration_bound(declaration: &Declaration) -> Result<usize> {
    let order = declaration.spec.order();
    let mut names = order
        .partition_by()
        .iter()
        .chain(order.sequence_by())
        .map(String::len)
        .chain(order.order_by().iter().map(|k| k.column().len()))
        .chain(
            declaration
                .spec
                .window()
                .map(|w| w.event_time_column().len()),
        );
    let size = names.try_fold(4096, |n, len| {
        owned::add(n, owned::add(256, owned::mul(len, 32)?)?)
    })?;
    declaration.expressions.iter().try_fold(size, |n, e| {
        let column = match &e.0 {
            AnalyticKind::RowNumber { .. } => 0,
            AnalyticKind::Lag { column, .. } | AnalyticKind::Sum { column, .. } => column.len(),
        };
        owned::add(
            n,
            owned::add(1024, owned::mul(owned::add(column, e.name().len())?, 32)?)?,
        )
    })
}
pub(in crate::operator) fn workspace_bound(
    history: &owned::OwnedBatch,
    declaration: &Declaration,
) -> Result<usize> {
    let n = history.batch.num_rows();
    let partitions = partition_bound(n, declaration.spec.order().partition_by().len())?;
    let provisional_input = provisional_input_bound(history)?;
    let output_schema = owned::add(declaration_bound(declaration)?, provisional_input)?;
    declaration
        .expressions
        .iter()
        .try_fold(owned::add(partitions, output_schema)?, |bound, e| {
            // Generated arrays coexist with the separately reserved compact final copy.
            owned::add(bound, owned::add(expression_bound(history, e)?, 512)?)
        })
}
fn partition_bound(rows: usize, columns: usize) -> Result<usize> {
    owned::add(owned::mul(rows, 8)?, owned::mul(columns, 256)?)
}
fn provisional_input_bound(history: &owned::OwnedBatch) -> Result<usize> {
    let table = history.batch.table_payload()?;
    // Separate owners and their lifetimes are listed in the OVER budget ledger.
    let schema_owners = owned::mul(owned::schema_quote(table.schema())?, 4)?;
    owned::add(owned::backing(&history.batch)?, schema_owners)
}
fn expression_bound(history: &owned::OwnedBatch, expression: &AnalyticExpr) -> Result<usize> {
    match &expression.0 {
        AnalyticKind::RowNumber { .. } | AnalyticKind::Sum { .. } => {
            array_bound(history.batch.num_rows(), 8, 0)
        }
        AnalyticKind::Lag { column, .. } => lag_bound(history, column),
    }
}
fn lag_bound(history: &owned::OwnedBatch, column: &str) -> Result<usize> {
    let n = history.batch.num_rows();
    let table = history.batch.table_payload()?;
    let field = table
        .schema()
        .field_with_name(column)
        .map_err(owned::error)?;
    let dtype = field.data_type();
    let values = array_bound(
        n,
        scalar_width(dtype),
        string_bytes(history, column, dtype)?,
    )?;
    let indices = array_bound(n, 4, 0)?;
    owned::add(owned::add(values, indices)?, string_gather_bound(n, dtype)?)
}
fn scalar_width(dtype: &DataType) -> usize {
    match dtype {
        DataType::Boolean | DataType::Int8 | DataType::UInt8 => 1,
        DataType::Int16 | DataType::UInt16 => 2,
        DataType::Int32 | DataType::UInt32 | DataType::Date32 | DataType::Utf8 => 4,
        _ => 8,
    }
}
fn string_gather_bound(n: usize, dtype: &DataType) -> Result<usize> {
    if matches!(dtype, DataType::Utf8 | DataType::LargeUtf8) {
        owned::add(
            owned::align(owned::mul(n, size_of::<(usize, usize)>())?)?,
            256,
        )
    } else {
        Ok(0)
    }
}
fn array_bound(n: usize, width: usize, strings: usize) -> Result<usize> {
    let values = values_bound(n, width)?;
    let validity = owned::align(owned::add(n, 7)? / 8)?;
    owned::add(
        owned::add(values, validity)?,
        owned::add(owned::align(strings)?, 512)?,
    )
}
fn values_bound(n: usize, width: usize) -> Result<usize> {
    owned::align(owned::mul(owned::add(n, 1)?, width)?)
}
fn string_bytes(history: &owned::OwnedBatch, column: &str, dtype: &DataType) -> Result<usize> {
    history
        .batch
        .table_payload()?
        .batches()
        .iter()
        .try_fold(0, |bound, r| {
            let array = r.column_by_name(column).unwrap();
            let bytes = match dtype {
                DataType::Utf8 => array
                    .as_any()
                    .downcast_ref::<StringArray>()
                    .unwrap()
                    .value_data()
                    .len(),
                DataType::LargeUtf8 => array
                    .as_any()
                    .downcast_ref::<LargeStringArray>()
                    .unwrap()
                    .value_data()
                    .len(),
                _ => 0,
            };
            owned::add(bound, bytes)
        })
}
