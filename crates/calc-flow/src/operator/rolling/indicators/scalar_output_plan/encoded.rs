use super::{Check, OutputPlan, Result, column, invalid};
use datafusion::{arrow::datatypes::DataType, common::ScalarValue};
#[path = "encoded/dictionary.rs"]
mod dictionary;
#[path = "encoded/physical.rs"]
mod physical;
#[path = "encoded/run.rs"]
mod run;
type Cell<'a, 'b> = dyn FnMut(usize) -> Option<&'a ScalarValue> + 'b;
pub(super) fn plan(
    dtype: &DataType,
    rows: usize,
    cell: &mut Cell<'_, '_>,
    check: Check<'_>,
) -> Result<Option<OutputPlan>> {
    match dtype {
        DataType::Dictionary(key, value) => {
            dictionary::plan(dtype, key, value, rows, cell, check).map(Some)
        }
        DataType::RunEndEncoded(..) => run::plan(dtype, rows, cell, check).map(Some),
        DataType::Union(..) => super::fixed::scalar_union(dtype, rows, cell, check).map(Some),
        _ => Ok(None),
    }
}
