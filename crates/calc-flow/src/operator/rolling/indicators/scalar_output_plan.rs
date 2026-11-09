//! Checked new Arrow column output and construction bounds.

use crate::{CalcFlowError, Result};
use datafusion::{arrow::datatypes::Schema, common::ScalarValue};
#[path = "scalar_output_plan/clones.rs"]
mod clones;
#[path = "scalar_output_plan/concatenated.rs"]
mod concatenated;
#[path = "scalar_output_plan/encoded.rs"]
mod encoded;
#[path = "scalar_output_plan/workspace.rs"]
mod workspace;
pub(crate) use workspace::{array_data_workspace, plan as planning_workspace};
#[path = "scalar_output_plan/fixed.rs"]
mod fixed;
#[path = "scalar_output_plan/holes.rs"]
mod holes;
#[path = "scalar_output_plan/leaf.rs"]
mod leaf;
#[path = "scalar_output_plan/math.rs"]
mod math;
#[path = "scalar_output_plan/normal.rs"]
mod normal;
#[path = "scalar_output_plan/producer.rs"]
mod producer;
#[path = "scalar_output_plan/single.rs"]
mod single;
#[path = "scalar_output_plan/single_ctor.rs"]
mod single_ctor;
pub(crate) use producer::encode_column;
type Check<'a> = &'a mut dyn FnMut() -> Result<()>;
pub(super) fn slice_descriptor_requirement(
    array: &dyn datafusion::arrow::array::Array,
    check: Check<'_>,
) -> Result<u64> {
    single::sliced(array, check)
}
#[derive(Clone, Copy, Debug, Default)]
pub(crate) struct OutputPlan {
    pub(crate) output_bytes: u64,
    pub(crate) scratch_bytes: u64,
}
/// The caller reserves `planning_workspace` before this traversal and releases it after return.
pub(crate) fn plan<'a>(
    schema: &Schema,
    rows: usize,
    cell: &mut dyn FnMut(usize, usize) -> Option<&'a ScalarValue>,
    check: Check<'_>,
) -> Result<OutputPlan> {
    check()?;
    capacity(schema, rows)?;
    let mut cost = OutputPlan::default();
    for (col, field) in schema.fields().iter().enumerate() {
        let part = charged_column(field.data_type(), rows, &mut |row| cell(col, row), check)?;
        cost.output_bytes = math::add(cost.output_bytes, part.output_bytes)?;
        cost.scratch_bytes = cost.scratch_bytes.max(part.scratch_bytes);
    }
    Ok(cost)
}
fn charged_column<'a>(
    dtype: &datafusion::arrow::datatypes::DataType,
    rows: usize,
    cell: &mut dyn FnMut(usize) -> Option<&'a ScalarValue>,
    check: Check<'_>,
) -> Result<OutputPlan> {
    let mut cost = column(dtype, rows, cell, check)?;
    cost.scratch_bytes = math::add(
        cost.scratch_bytes,
        clones::column(dtype, rows, cell, check)?,
    )?;
    Ok(cost)
}
fn column<'a>(
    dtype: &datafusion::arrow::datatypes::DataType,
    rows: usize,
    cell: &mut dyn FnMut(usize) -> Option<&'a ScalarValue>,
    check: Check<'_>,
) -> Result<OutputPlan> {
    if let Some(cost) = holes::plan(dtype, rows, cell, check)? {
        return Ok(cost);
    }
    if let Some(cost) = typed_column(dtype, rows, cell, check)? {
        return Ok(cost);
    }
    leaf::plan(dtype, rows, cell, check)
}

fn typed_column<'a>(
    dtype: &datafusion::arrow::datatypes::DataType,
    rows: usize,
    cell: &mut dyn FnMut(usize) -> Option<&'a ScalarValue>,
    check: Check<'_>,
) -> Result<Option<OutputPlan>> {
    if let Some(cost) = fixed::plan(dtype, rows, cell, check)? {
        return Ok(Some(cost));
    }
    if let Some(cost) = array_column(dtype, rows, cell, check)? {
        return Ok(Some(cost));
    }
    encoded::plan(dtype, rows, cell, check)
}
fn array_column<'a>(
    dtype: &datafusion::arrow::datatypes::DataType,
    rows: usize,
    cell: &mut dyn FnMut(usize) -> Option<&'a ScalarValue>,
    check: Check<'_>,
) -> Result<Option<OutputPlan>> {
    if rows == 1
        && let Some(array) = cell(0).and_then(single::array)
    {
        return single::plan(dtype, array, check).map(Some);
    }
    concatenated::plan(dtype, rows, cell, check)
}
fn capacity(schema: &Schema, rows: usize) -> Result<()> {
    if schema.fields().is_empty() {
        return Ok(());
    }
    math::scalar_capacity(rows)
}

fn invalid(reason: &str) -> CalcFlowError {
    CalcFlowError::Format {
        message: format!("rolling scalar output allocation {reason} is invalid"),
    }
}
#[cfg(test)]
#[path = "scalar_output_plan/tests.rs"]
mod tests;
