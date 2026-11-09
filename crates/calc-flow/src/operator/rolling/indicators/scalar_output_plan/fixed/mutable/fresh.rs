use super::super::Cell;
use super::{Check, Geometry, Result, node, sources::Fresh};
use crate::operator::rolling::indicators::{
    scalar_materialization_plan, scalar_output_plan as output,
};
use datafusion::{arrow::datatypes::DataType, common::ScalarValue};
use output::{
    OutputPlan, clones,
    math::{add, size, sum},
    single, single_ctor, workspace,
};

pub(in crate::operator::rolling::indicators::scalar_output_plan) fn plan(
    dtype: &DataType,
    rows: usize,
    cell: &mut Cell<'_, '_>,
    check: Check<'_>,
) -> Result<OutputPlan> {
    let inputs = inputs(dtype, rows, cell, check)?;
    let mut sources = Fresh { count: rows, cell };
    let cost = node(dtype, size(rows)?, &mut sources, check)?;
    let geometry = Geometry {
        slots: size(rows)?,
        values: cost.output,
        child_capacity: size(rows)?,
        buffers: 0,
        descriptors: cost.descriptors,
        workspace: cost.scratch,
        builders: cost.builders,
    };
    finish(dtype, rows, geometry, inputs, check)
}
fn finish(
    dtype: &DataType,
    rows: usize,
    geometry: Geometry,
    inputs: (u64, u64),
    check: Check<'_>,
) -> Result<OutputPlan> {
    let empty = scalar_materialization_plan::null_array_cost(dtype, 0, check)?;
    Ok(OutputPlan {
        output_bytes: sum([
            Ok(inputs.0),
            Ok(geometry.values),
            Ok(geometry.descriptors),
            Ok(empty.logical_owned),
        ])?,
        scratch_bytes: output::fixed::scratch(dtype, rows, geometry, inputs.1, check)?,
    })
}

fn inputs(
    dtype: &DataType,
    rows: usize,
    cell: &mut Cell<'_, '_>,
    check: Check<'_>,
) -> Result<(u64, u64)> {
    let mut owned = 0;
    let mut peak = 0;
    for row in 0..rows {
        check()?;
        let value = cell(row);
        let cost = constructor(dtype, value, check)?;
        let output = cost.logical_owned;
        let temporary = cost.construction_peak;
        owned = add(owned, output)?;
        peak = sum([Ok(peak), Ok(temporary), metadata(dtype, value, check)])?;
    }
    Ok((owned, peak))
}
fn constructor(
    dtype: &DataType,
    value: Option<&ScalarValue>,
    check: Check<'_>,
) -> Result<single_ctor::SingleCtorCost> {
    if let Some(value) = value {
        return single_ctor::plan(dtype, value, check);
    }
    let cost = scalar_materialization_plan::null_array_cost(dtype, 1, check)?;
    Ok(single_ctor::SingleCtorCost {
        logical_owned: cost.logical_owned,
        construction_peak: cost.construction_peak,
    })
}

fn metadata(dtype: &DataType, value: Option<&ScalarValue>, check: Check<'_>) -> Result<u64> {
    let fixed = scalar_materialization_plan::null_array_cost(dtype, 0, check)?;
    sum([
        Ok(fixed.construction_peak),
        clones::datatype(dtype, check),
        borrowed_metadata(value, check),
    ])
}
fn borrowed_metadata(value: Option<&ScalarValue>, check: Check<'_>) -> Result<u64> {
    check()?;
    let Some(value) = value else {
        return Ok(0);
    };
    if let Some(array) = single::array(value) {
        return workspace::array_data_workspace(array, check);
    }
    match value {
        ScalarValue::Dictionary(_, value)
        | ScalarValue::RunEndEncoded(_, _, value)
        | ScalarValue::Union(Some((_, value)), _, _) => borrowed_metadata(Some(value), check),
        _ => Ok(0),
    }
}
