//! New logical storage and total construction peak for borrowed `producer::single`.

use super::{Check, Result, invalid};
use datafusion::{arrow::datatypes::DataType, common::ScalarValue};

#[path = "single_ctor/encoded.rs"]
mod encoded;
#[path = "single_ctor/shape.rs"]
mod shape;

#[derive(Clone, Copy, Debug, Default)]
pub(super) struct SingleCtorCost {
    pub(super) logical_owned: u64,
    pub(super) construction_peak: u64,
}

/// Excludes the borrowed scalar, column vectors and later concatenation.
pub(super) fn plan(
    expected: &DataType,
    value: &ScalarValue,
    check: Check<'_>,
) -> Result<SingleCtorCost> {
    check()?;
    let cost = if let Some(array) = super::single::array(value) {
        if array.len() != 1 || array.data_type() != expected {
            return Err(invalid("single nested scalar shape"));
        }
        SingleCtorCost::default()
    } else {
        dispatch(expected, value, check)?
    };
    check()?;
    Ok(cost)
}

fn dispatch(expected: &DataType, value: &ScalarValue, check: Check<'_>) -> Result<SingleCtorCost> {
    match value {
        ScalarValue::Dictionary(key, child) => encoded::dictionary(expected, key, child, check),
        ScalarValue::RunEndEncoded(ends, field, child) => {
            encoded::run(expected, ends, field, child, check)
        }
        ScalarValue::Union(value, fields, mode) => {
            encoded::union(expected, value.as_ref(), fields, *mode, check)
        }
        _ => leaf(expected, value, check),
    }
}

fn leaf(expected: &DataType, value: &ScalarValue, check: Check<'_>) -> Result<SingleCtorCost> {
    if let ScalarValue::FixedSizeBinary(width, Some(bytes)) = value
        && usize::try_from(*width).ok() != Some(bytes.len())
    {
        return Err(invalid("single fixed binary width"));
    }
    let cost = super::leaf::plan(expected, 1, &mut |_| Some(value), check)?;
    Ok(SingleCtorCost {
        logical_owned: cost.output_bytes,
        construction_peak: super::math::add(cost.output_bytes, cost.scratch_bytes)?,
    })
}

fn null_array(expected: &DataType, rows: usize, check: Check<'_>) -> Result<SingleCtorCost> {
    let cost = super::super::scalar_materialization_plan::null_array_cost(expected, rows, check)?;
    Ok(SingleCtorCost {
        logical_owned: cost.logical_owned,
        construction_peak: cost.construction_peak,
    })
}

impl SingleCtorCost {
    fn combine(self, other: Self) -> Result<Self> {
        Ok(Self {
            logical_owned: super::math::add(self.logical_owned, other.logical_owned)?,
            construction_peak: super::math::add(self.construction_peak, other.construction_peak)?,
        })
    }
}

#[cfg(test)]
#[path = "single_ctor/tests.rs"]
mod tests;
