use super::{
    Check, OutputPlan, Result,
    math::{bytes, grow, size, sum},
};
use datafusion::{arrow::datatypes::DataType, common::ScalarValue};

pub(super) fn plan<'a>(
    dtype: &DataType,
    rows: usize,
    cell: &mut dyn FnMut(usize) -> Option<&'a ScalarValue>,
    check: Check<'_>,
) -> Result<Option<OutputPlan>> {
    if rows == 0 {
        return empty(dtype, check).map(Some);
    }
    if rows == 1 && array_backed(dtype) && cell(0).is_none() {
        return single(dtype, check).map(Some);
    }
    Ok(None)
}
fn empty(dtype: &DataType, check: Check<'_>) -> Result<OutputPlan> {
    let cost = super::super::scalar_materialization_plan::null_array_cost(dtype, 0, check)?;
    Ok(OutputPlan {
        output_bytes: cost.logical_owned,
        scratch_bytes: cost.construction_peak,
    })
}
fn single(dtype: &DataType, check: Check<'_>) -> Result<OutputPlan> {
    let cost = super::super::scalar_materialization_plan::null_array_cost(dtype, 1, check)?;
    let scalar = super::super::scalar_materialization_plan::null_requirement(dtype, check)?;
    Ok(OutputPlan {
        output_bytes: cost.logical_owned,
        scratch_bytes: sum([
            Ok(cost.construction_peak),
            Ok(scalar),
            bytes::<ScalarValue>(1),
            grow(1, size(size_of::<datafusion::arrow::array::ArrayRef>())?),
            grow(1, size(size_of::<&dyn datafusion::arrow::array::Array>())?),
        ])?,
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
