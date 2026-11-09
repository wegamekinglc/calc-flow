use super::super::math::{add, arc_slice, vector};
use super::{Check, Result, SingleCtorCost, invalid, null_array};
use datafusion::{
    arrow::{buffer::Buffer, datatypes::DataType},
    common::ScalarValue,
};

/// Payload stays shared during `to_data`/`make_array`; descriptors are rebuilt.
pub(super) fn rebuild(
    expected: &DataType,
    value: &ScalarValue,
    check: Check<'_>,
) -> Result<SingleCtorCost> {
    check()?;
    if let Some(array) = super::super::single::array(value) {
        return array_shape(array, check);
    }
    let base = null_array(expected, 0, check)?;
    base.combine(dynamic(expected, value, check)?)
}

fn array_shape(
    array: &dyn datafusion::arrow::array::Array,
    check: Check<'_>,
) -> Result<SingleCtorCost> {
    let descriptors = super::super::workspace::descriptor_tree(array, check)?;
    let metadata = super::super::workspace::array_data_workspace(array, check)?;
    Ok(SingleCtorCost {
        logical_owned: descriptors,
        construction_peak: add(descriptors, metadata)?,
    })
}

fn dynamic(expected: &DataType, value: &ScalarValue, check: Check<'_>) -> Result<SingleCtorCost> {
    match (expected, value) {
        (DataType::Dictionary(_, dtype), ScalarValue::Dictionary(_, child)) => {
            rebuild(dtype, child, check)
        }
        (DataType::RunEndEncoded(_, field), ScalarValue::RunEndEncoded(_, _, child)) => {
            rebuild(field.data_type(), child, check)
        }
        (DataType::Union(fields, _), ScalarValue::Union(Some((id, child)), _, _)) => {
            let field = fields
                .iter()
                .find(|(key, _)| key == id)
                .ok_or_else(|| invalid("single union type ID"))?
                .1;
            rebuild(field.data_type(), child, check)
        }
        (DataType::Utf8View, ScalarValue::Utf8View(Some(value))) => views(value.len()),
        (DataType::BinaryView, ScalarValue::BinaryView(Some(value))) => views(value.len()),
        _ => Ok(SingleCtorCost::default()),
    }
}

fn views(len: usize) -> Result<SingleCtorCost> {
    let blocks = usize::from(len > 12);
    let descriptors = arc_slice::<Buffer>(blocks)?;
    Ok(SingleCtorCost {
        logical_owned: descriptors,
        construction_peak: add(descriptors, vector::<Buffer>(blocks + 1)?)?,
    })
}
