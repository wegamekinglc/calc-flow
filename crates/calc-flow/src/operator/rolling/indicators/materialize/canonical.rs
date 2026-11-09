//! Canonical typed metadata preserves shared backing without third-party slice calls.

use crate::{CalcFlowError, Result};
use datafusion::{
    arrow::{
        array::{
            Array, ArrayData, FixedSizeListArray, GenericListArray, GenericListViewArray,
            LargeListArray, LargeListViewArray, ListArray, ListViewArray, MapArray,
            OffsetSizeTrait, StructArray,
        },
        datatypes::DataType,
    },
    common::ScalarValue,
};
use std::sync::Arc;

#[path = "canonical/encoded.rs"]
mod encoded;

#[path = "canonical/leaf.rs"]
mod leaf;

#[path = "canonical/expected.rs"]
mod expected;

type Check<'a> = &'a mut dyn FnMut() -> Result<()>;

#[cfg(test)]
pub(in super::super) fn nested_scalar(
    array: &dyn Array,
    row: usize,
    node: &str,
    check: Check<'_>,
) -> Result<Option<ScalarValue>> {
    nested_scalar_as(array, row, array.data_type(), node, check)
}

fn scalar(
    array: &dyn Array,
    row: usize,
    expected_dtype: &DataType,
    node: &str,
    check: Check<'_>,
) -> Result<Option<ScalarValue>> {
    match expected_dtype {
        DataType::List(_) => boxed(
            list::<i32>(array, row, 1, expected_dtype, node, check).map(ListArray::from),
            ScalarValue::List,
        ),
        DataType::LargeList(_) => boxed(
            list::<i64>(array, row, 1, expected_dtype, node, check).map(LargeListArray::from),
            ScalarValue::LargeList,
        ),
        DataType::FixedSizeList(..) => boxed(
            fixed_list(array, row, 1, expected_dtype, node, check).map(FixedSizeListArray::from),
            ScalarValue::FixedSizeList,
        ),
        _ => composite(array, row, expected_dtype, node, check),
    }
}
fn boxed<T>(value: Result<T>, wrap: fn(Arc<T>) -> ScalarValue) -> Result<Option<ScalarValue>> {
    value.map(|value| Some(wrap(Arc::new(value))))
}
fn data(
    array: &dyn Array,
    offset: usize,
    len: usize,
    expected_dtype: &DataType,
    node: &str,
    check: Check<'_>,
) -> Result<ArrayData> {
    check()?;
    range(array, offset, len, node)?;
    if let Some(data) = leaf::data(array, offset, len, expected_dtype, node) {
        return data;
    }
    match expected_dtype {
        DataType::List(_) => list::<i32>(array, offset, len, expected_dtype, node, check),
        DataType::LargeList(_) => list::<i64>(array, offset, len, expected_dtype, node, check),
        DataType::FixedSizeList(..) => fixed_list(array, offset, len, expected_dtype, node, check),
        _ => composite_data(array, offset, len, expected_dtype, node, check),
    }
}
fn composite(
    array: &dyn Array,
    row: usize,
    expected_dtype: &DataType,
    node: &str,
    check: Check<'_>,
) -> Result<Option<ScalarValue>> {
    match expected_dtype {
        DataType::ListView(_) => boxed(
            list_view::<i32>(array, row, 1, expected_dtype, node, check).map(ListViewArray::from),
            ScalarValue::ListView,
        ),
        DataType::LargeListView(_) => boxed(
            list_view::<i64>(array, row, 1, expected_dtype, node, check)
                .map(LargeListViewArray::from),
            ScalarValue::LargeListView,
        ),
        DataType::Struct(_) => boxed(
            struct_data(array, row, 1, expected_dtype, node, check).map(StructArray::from),
            ScalarValue::Struct,
        ),
        DataType::Map(..) => boxed(
            map_data(array, row, 1, expected_dtype, node, check).map(MapArray::from),
            ScalarValue::Map,
        ),
        _ => Ok(None),
    }
}
fn composite_data(
    array: &dyn Array,
    offset: usize,
    len: usize,
    expected_dtype: &DataType,
    node: &str,
    check: Check<'_>,
) -> Result<ArrayData> {
    match expected_dtype {
        DataType::ListView(_) => list_view::<i32>(array, offset, len, expected_dtype, node, check),
        DataType::LargeListView(_) => {
            list_view::<i64>(array, offset, len, expected_dtype, node, check)
        }
        DataType::Struct(_) => struct_data(array, offset, len, expected_dtype, node, check),
        DataType::Map(..) => map_data(array, offset, len, expected_dtype, node, check),
        _ => encoded::encoded(array, offset, len, expected_dtype, node, check),
    }
}

fn list<O: OffsetSizeTrait>(
    array: &dyn Array,
    offset: usize,
    len: usize,
    expected_dtype: &DataType,
    node: &str,
    check: Check<'_>,
) -> Result<ArrayData> {
    let array = typed::<GenericListArray<O>>(array, node)?;
    let values = data(
        array.values().as_ref(),
        0,
        array.values().len(),
        child_dtype(expected_dtype, node)?,
        node,
        check,
    )?;
    let builder = ArrayData::builder(expected_dtype.clone())
        .len(len)
        .offset(offset)
        .nulls(array.nulls().map(|mask| mask.slice(offset, len)))
        .buffers(vec![array.offsets().inner().inner().clone()])
        .child_data(vec![values]);
    builder.build().map_err(|_| invalid(node))
}
fn fixed_list(
    array: &dyn Array,
    offset: usize,
    len: usize,
    expected_dtype: &DataType,
    node: &str,
    check: Check<'_>,
) -> Result<ArrayData> {
    let array = typed::<FixedSizeListArray>(array, node)?;
    let width = usize::try_from(array.value_length()).map_err(|_| invalid(node))?;
    let start = offset.checked_mul(width).ok_or_else(|| invalid(node))?;
    let size = len.checked_mul(width).ok_or_else(|| invalid(node))?;
    let values = data(
        array.values().as_ref(),
        start,
        size,
        child_dtype(expected_dtype, node)?,
        node,
        check,
    )?;
    let builder = ArrayData::builder(expected_dtype.clone())
        .len(len)
        .nulls(array.nulls().map(|mask| mask.slice(offset, len)))
        .child_data(vec![values]);
    builder.build().map_err(|_| invalid(node))
}
fn range(array: &dyn Array, offset: usize, len: usize, node: &str) -> Result<()> {
    if offset.checked_add(len).is_none_or(|end| end > array.len()) {
        return Err(invalid(node));
    }
    Ok(())
}
fn typed<'a, T: Array + 'static>(array: &'a dyn Array, node: &str) -> Result<&'a T> {
    array.as_any().downcast_ref().ok_or_else(|| invalid(node))
}
fn invalid(node: &str) -> CalcFlowError {
    CalcFlowError::Operator {
        node_id: node.into(),
        message: "rolling canonical scalar metadata or representation is invalid".into(),
    }
}

#[cfg(test)]
#[path = "canonical/tests.rs"]
mod tests;

fn list_view<O: OffsetSizeTrait>(
    array: &dyn Array,
    offset: usize,
    len: usize,
    expected_dtype: &DataType,
    node: &str,
    check: Check<'_>,
) -> Result<ArrayData> {
    let array = typed::<GenericListViewArray<O>>(array, node)?;
    let values = data(
        array.values().as_ref(),
        0,
        array.values().len(),
        child_dtype(expected_dtype, node)?,
        node,
        check,
    )?;
    ArrayData::builder(expected_dtype.clone())
        .len(len)
        .offset(offset)
        .nulls(array.nulls().map(|mask| mask.slice(offset, len)))
        .buffers(vec![
            array.offsets().inner().clone(),
            array.sizes().inner().clone(),
        ])
        .child_data(vec![values])
        .build()
        .map_err(|_| invalid(node))
}
fn struct_data(
    array: &dyn Array,
    offset: usize,
    len: usize,
    expected_dtype: &DataType,
    node: &str,
    check: Check<'_>,
) -> Result<ArrayData> {
    let array = typed::<StructArray>(array, node)?;
    let mut children = Vec::with_capacity(array.columns().len());
    let fields = struct_fields(expected_dtype, node)?;
    for (child, field) in array.columns().iter().zip(fields.iter()) {
        children.push(data(
            child.as_ref(),
            offset,
            len,
            field.data_type(),
            node,
            check,
        )?);
    }
    ArrayData::builder(expected_dtype.clone())
        .len(len)
        .nulls(array.nulls().map(|mask| mask.slice(offset, len)))
        .child_data(children)
        .build()
        .map_err(|_| invalid(node))
}
fn map_data(
    array: &dyn Array,
    offset: usize,
    len: usize,
    expected_dtype: &DataType,
    node: &str,
    check: Check<'_>,
) -> Result<ArrayData> {
    let array = typed::<MapArray>(array, node)?;
    let entries = data(
        array.entries(),
        0,
        array.entries().len(),
        child_dtype(expected_dtype, node)?,
        node,
        check,
    )?;
    ArrayData::builder(expected_dtype.clone())
        .len(len)
        .offset(offset)
        .nulls(array.nulls().map(|mask| mask.slice(offset, len)))
        .buffers(vec![array.offsets().inner().inner().clone()])
        .child_data(vec![entries])
        .build()
        .map_err(|_| invalid(node))
}

#[path = "canonical/null.rs"]
mod null;

pub(in super::super) fn requires_null_fallback(dtype: &DataType, check: Check<'_>) -> Result<bool> {
    null::needs(dtype, check)
}
pub(in super::super) fn null_scalar(
    dtype: &DataType,
    node: &str,
    check: Check<'_>,
) -> Result<ScalarValue> {
    check()?;
    super::super::scalar_materialization_plan::null_requirement(dtype, check)?;
    let scalar = null::scalar(dtype, node, check)?;
    check()?;
    Ok(scalar)
}

pub(in super::super) fn null_array(
    dtype: &DataType,
    len: usize,
    node: &str,
    check: Check<'_>,
) -> Result<datafusion::arrow::array::ArrayRef> {
    check()?;
    super::super::scalar_materialization_plan::null_array_requirement(dtype, len, check)?;
    let array = null::array(dtype, len, node, check)?;
    check()?;
    Ok(array)
}

#[cfg(test)]
#[path = "canonical/rebinding_tests.rs"]
mod rebinding_tests;

pub(in super::super) fn nested_scalar_as(
    array: &dyn Array,
    row: usize,
    expected_dtype: &DataType,
    node: &str,
    check: Check<'_>,
) -> Result<Option<ScalarValue>> {
    check()?;
    range(array, row, 1, node)?;
    expected::validate(array.data_type(), expected_dtype, node, check)?;
    let value = scalar(array, row, expected_dtype, node, check)?;
    check()?;
    Ok(value)
}

fn child_dtype<'a>(dtype: &'a DataType, node: &str) -> Result<&'a DataType> {
    match dtype {
        DataType::List(f)
        | DataType::LargeList(f)
        | DataType::FixedSizeList(f, _)
        | DataType::ListView(f)
        | DataType::LargeListView(f)
        | DataType::Map(f, _) => Ok(f.data_type()),
        _ => Err(invalid(node)),
    }
}
fn struct_fields<'a>(
    dtype: &'a DataType,
    node: &str,
) -> Result<&'a datafusion::arrow::datatypes::Fields> {
    let DataType::Struct(fields) = dtype else {
        return Err(invalid(node));
    };
    Ok(fields)
}
