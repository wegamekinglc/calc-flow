//! Checked extra owned heap for the recursive borrowed scalar materializer.

use crate::{CalcFlowError, Result};
use datafusion::arrow::{
    array::{Array, ArrayRef},
    datatypes::DataType,
};

#[path = "scalar_materialization_plan/bytes.rs"]
mod bytes;
#[path = "scalar_materialization_plan/cloned.rs"]
mod cloned;
#[path = "scalar_materialization_plan/cost.rs"]
mod cost;
#[path = "scalar_materialization_plan/descriptor.rs"]
mod descriptor;

#[path = "scalar_materialization_plan/canonical.rs"]
mod canonical;

#[path = "scalar_materialization_plan/encoded.rs"]
mod encoded;

#[path = "scalar_materialization_plan/null.rs"]
mod null;

type Check<'a> = &'a mut dyn FnMut() -> Result<()>;

pub(super) fn plan(columns: &[ArrayRef], check: Check<'_>) -> Result<u64> {
    check()?;
    let mut total = cost::Cost::default();
    for column in columns {
        total = total.sequence(column_cost(column.as_ref(), check)?)?;
    }
    check()?;
    add(total.owned, total.temporary)
}
pub(super) fn plan_selected<'a>(
    rows: impl IntoIterator<Item = (&'a [ArrayRef], usize)>,
    check: Check<'_>,
) -> Result<u64> {
    check()?;
    let mut total = cost::Cost::default();
    for (columns, row) in rows {
        total = total.sequence(selected_row(columns, row, check)?)?;
    }
    check()?;
    add(total.owned, total.temporary)
}
fn selected_row(columns: &[ArrayRef], row: usize, check: Check<'_>) -> Result<cost::Cost> {
    let mut total = cost::Cost::default();
    for column in columns {
        check()?;
        if row >= column.len() {
            return Err(invalid());
        }
        total = total.sequence(cell(column.as_ref(), row, check)?)?;
    }
    Ok(total)
}
fn column_cost(array: &dyn Array, check: Check<'_>) -> Result<cost::Cost> {
    let mut total = cost::Cost::default();
    for row in 0..array.len() {
        check()?;
        total = total.sequence(cell(array, row, check)?)?;
    }
    Ok(total)
}
fn cell(array: &dyn Array, row: usize, check: Check<'_>) -> Result<cost::Cost> {
    if let Some(cost) = canonical::direct(array, row, check)? {
        return Ok(cost);
    }
    if matches!(
        array.data_type(),
        DataType::Dictionary(..) | DataType::RunEndEncoded(..) | DataType::Union(..)
    ) {
        return encoded::cell(array, row, check);
    }
    plain_cell(array, row)
}
fn plain_cell(array: &dyn Array, row: usize) -> Result<cost::Cost> {
    if array.is_null(row) {
        return Ok(cost::Cost::default());
    }
    if let Some(bytes) = bytes::cell(array, row)? {
        return Ok(cost::Cost::owned(bytes));
    }
    if array.data_type().is_primitive()
        || matches!(array.data_type(), DataType::Null | DataType::Boolean)
    {
        return Ok(cost::Cost::default());
    }
    Err(invalid())
}
/// Descriptor bound for canonical/IPC standard owners, excluding child owners and data backing.
pub(super) fn descriptor_bytes(array: &dyn Array, check: Check<'_>) -> Result<u64> {
    check()?;
    let own = descriptor::owned(array.data_type(), check)?;
    add(
        add(own, descriptor_vector(array)?)?,
        descriptor_views(array)?,
    )
}
fn descriptor_vector(array: &dyn Array) -> Result<u64> {
    match array.data_type() {
        DataType::Struct(fields) => descriptor::vector(array.data_type(), fields.len()),
        DataType::Map(..) => cost::times::<datafusion::arrow::array::ArrayData>(2),
        _ => descriptor::vector(array.data_type(), 0),
    }
}
fn descriptor_views(array: &dyn Array) -> Result<u64> {
    use datafusion::arrow::array::{BinaryViewArray, StringViewArray};
    let len = match array.data_type() {
        DataType::Utf8View => typed::<StringViewArray>(array)?.data_buffers().len(),
        DataType::BinaryView => typed::<BinaryViewArray>(array)?.data_buffers().len(),
        _ => return Ok(0),
    };
    descriptor::views(len)
}
fn typed<T: Array + 'static>(array: &dyn Array) -> Result<&T> {
    array.as_any().downcast_ref().ok_or_else(invalid)
}
fn add(left: u64, right: u64) -> Result<u64> {
    left.checked_add(right).ok_or_else(invalid)
}
fn size(value: usize) -> Result<u64> {
    u64::try_from(value).map_err(|_| invalid())
}
fn invalid() -> CalcFlowError {
    CalcFlowError::Format {
        message: "rolling scalar materialization representation or size is invalid".into(),
    }
}

#[cfg(test)]
#[path = "scalar_materialization_plan/tests.rs"]
mod tests;

pub(super) fn null_requirement(dtype: &DataType, check: Check<'_>) -> Result<u64> {
    let cost = null::scalar(dtype, check)?;
    add(cost.owned, cost.temporary)
}

pub(super) fn null_array_requirement(
    dtype: &DataType,
    len: usize,
    check: Check<'_>,
) -> Result<u64> {
    null_array_cost(dtype, len, check).map(|cost| cost.construction_peak)
}

pub(super) struct NullArrayCost {
    pub(super) logical_owned: u64,
    pub(super) construction_peak: u64,
}
pub(super) fn null_array_cost(
    dtype: &DataType,
    len: usize,
    check: Check<'_>,
) -> Result<NullArrayCost> {
    null::array_profile(dtype, len, check)
}

#[cfg(test)]
#[path = "scalar_materialization_plan/profile_tests.rs"]
mod profile_tests;

pub(super) fn datatype_clone_requirement(dtype: &DataType, check: Check<'_>) -> Result<u64> {
    descriptor::dtype(dtype, check)
}

/// Pinned `ArrayData` layout-vector allocation for one node, without allocation.
pub(super) fn datatype_layout_requirement(dtype: &DataType) -> Result<u64> {
    descriptor::layout(dtype)
}

#[cfg(test)]
#[path = "scalar_materialization_plan/decode_tests.rs"]
mod decode_tests;

/// New owned heap and maximum constructor temporary for the selected scalar.
pub(super) fn scalar_requirement(
    array: &dyn Array,
    row: usize,
    check: Check<'_>,
) -> Result<(u64, u64)> {
    scalar_range(array, row, check)?;
    let cost = cell(array, row, check)?;
    Ok((cost.owned, cost.temporary))
}
/// Extra heap from cloning that scalar, excluding its caller-funded inline slot.
pub(super) fn scalar_clone_requirement(
    array: &dyn Array,
    row: usize,
    check: Check<'_>,
) -> Result<u64> {
    scalar_range(array, row, check)?;
    cloned::cell(array, row, check)
}
fn scalar_range(array: &dyn Array, row: usize, check: Check<'_>) -> Result<()> {
    check()?;
    if row >= array.len() {
        return Err(invalid());
    }
    Ok(())
}
