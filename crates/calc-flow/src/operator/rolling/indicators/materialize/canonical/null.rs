use super::{Check, boxed, invalid};
use crate::Result;
use datafusion::{
    arrow::{
        array::{
            ArrayData, ArrayRef, FixedSizeListArray, LargeListArray, LargeListViewArray, ListArray,
            ListViewArray, MapArray, StructArray, make_array,
        },
        datatypes::{DataType as D, Fields, UnionFields},
    },
    common::ScalarValue,
};
use std::sync::Arc;

#[path = "null_data.rs"]
mod data;

pub(super) fn needs(dtype: &D, check: Check<'_>) -> Result<bool> {
    check()?;
    match dtype {
        D::List(f)
        | D::LargeList(f)
        | D::FixedSizeList(f, _)
        | D::ListView(f)
        | D::LargeListView(f)
        | D::Map(f, _)
        | D::RunEndEncoded(_, f) => needs(f.data_type(), check),
        D::Dictionary(_, v) => needs(v, check),
        D::Struct(fields) => fields_need(fields, check),
        D::Union(fields, _) => union_needs(fields, check),
        _ => Ok(false),
    }
}
fn fields_need(fields: &Fields, check: Check<'_>) -> Result<bool> {
    for field in fields {
        if needs(field.data_type(), check)? {
            return Ok(true);
        }
    }
    Ok(false)
}
fn union_needs(fields: &UnionFields, check: Check<'_>) -> Result<bool> {
    if fields.is_empty() {
        return Ok(true);
    }
    for (_, field) in fields.iter() {
        if needs(field.data_type(), check)? {
            return Ok(true);
        }
    }
    Ok(false)
}
pub(super) fn scalar(dtype: &D, node: &str, check: Check<'_>) -> Result<ScalarValue> {
    if !needs(dtype, check)? {
        return ScalarValue::try_new_null(dtype).map_err(|_| invalid(node));
    }
    encoded_scalar(dtype, node, check)
}
fn encoded_scalar(dtype: &D, node: &str, check: Check<'_>) -> Result<ScalarValue> {
    match dtype {
        D::Dictionary(key, value) => Ok(ScalarValue::Dictionary(
            key.clone(),
            Box::new(scalar(value, node, check)?),
        )),
        D::RunEndEncoded(ends, value) => Ok(ScalarValue::RunEndEncoded(
            Arc::clone(ends),
            Arc::clone(value),
            Box::new(scalar(value.data_type(), node, check)?),
        )),
        D::Union(fields, mode) => Ok(ScalarValue::Union(None, fields.clone(), *mode)),
        _ => nested(dtype, node, check),
    }
}
fn nested(dtype: &D, node: &str, check: Check<'_>) -> Result<ScalarValue> {
    let data = data::null_data(dtype, 1, node, check)?;
    let value = match dtype {
        D::List(_) => boxed(Ok(ListArray::from(data)), ScalarValue::List),
        D::LargeList(_) => boxed(Ok(LargeListArray::from(data)), ScalarValue::LargeList),
        D::FixedSizeList(..) => boxed(
            Ok(FixedSizeListArray::from(data)),
            ScalarValue::FixedSizeList,
        ),
        _ => composite(dtype, data, node),
    }?;
    value.ok_or_else(|| invalid(node))
}
fn composite(dtype: &D, data: ArrayData, node: &str) -> Result<Option<ScalarValue>> {
    match dtype {
        D::ListView(_) => boxed(Ok(ListViewArray::from(data)), ScalarValue::ListView),
        D::LargeListView(_) => boxed(
            Ok(LargeListViewArray::from(data)),
            ScalarValue::LargeListView,
        ),
        D::Struct(_) => boxed(Ok(StructArray::from(data)), ScalarValue::Struct),
        D::Map(..) => boxed(Ok(MapArray::from(data)), ScalarValue::Map),
        _ => Err(invalid(node)),
    }
}

pub(super) fn array(dtype: &D, len: usize, node: &str, check: Check<'_>) -> Result<ArrayRef> {
    data::null_data(dtype, len, node, check).map(make_array)
}
