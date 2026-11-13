use super::{Declaration, spec::AnalyticKind};
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
    let table = history.batch.table_payload()?;
    let partitions = owned::add(
        owned::mul(n, 8)?,
        owned::mul(declaration.spec.order().partition_by().len(), 256)?,
    )?;
    // Separate owners and their lifetimes are listed in the OVER budget ledger.
    let schema_owners = owned::mul(owned::schema_quote(table.schema())?, 4)?;
    let provisional_input = owned::add(owned::backing(&history.batch)?, schema_owners)?;
    let output_schema = owned::add(declaration_bound(declaration)?, provisional_input)?;
    declaration
        .expressions
        .iter()
        .try_fold(owned::add(partitions, output_schema)?, |bound, e| {
            let buffers = match &e.0 {
                AnalyticKind::RowNumber { .. } | AnalyticKind::Sum { .. } => array_bound(n, 8, 0)?,
                AnalyticKind::Lag { column, .. } => {
                    let field = table
                        .schema()
                        .field_with_name(column)
                        .map_err(owned::error)?;
                    let bytes = string_bytes(history, column, field.data_type())?;
                    let width = match field.data_type() {
                        DataType::Boolean | DataType::Int8 | DataType::UInt8 => 1,
                        DataType::Int16 | DataType::UInt16 => 2,
                        DataType::Int32 | DataType::UInt32 | DataType::Date32 | DataType::Utf8 => 4,
                        _ => 8,
                    };
                    let gather =
                        if matches!(field.data_type(), DataType::Utf8 | DataType::LargeUtf8) {
                            owned::add(
                                owned::align(owned::mul(n, size_of::<(usize, usize)>())?)?,
                                256,
                            )?
                        } else {
                            0
                        };
                    owned::add(
                        owned::add(array_bound(n, width, bytes)?, array_bound(n, 4, 0)?)?,
                        gather,
                    )?
                }
            };
            // A live generated array and its compact final copy coexist. The copy's
            // own reservation is acquired by normalize before allocation.
            owned::add(bound, owned::add(buffers, 512)?)
        })
}
fn array_bound(n: usize, width: usize, strings: usize) -> Result<usize> {
    let values = owned::align(owned::mul(owned::add(n, 1)?, width)?)?;
    let validity = owned::align(owned::add(n, 7)? / 8)?;
    owned::add(
        owned::add(values, validity)?,
        owned::add(owned::align(strings)?, 512)?,
    )
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
