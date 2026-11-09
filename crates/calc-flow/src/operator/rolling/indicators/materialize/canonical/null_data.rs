use super::{Check, invalid, needs};
use crate::Result;
use datafusion::arrow::{
    array::{ArrayData, ArrayDataBuilder},
    buffer::{Buffer, MutableBuffer, NullBuffer},
    datatypes::{DataType as D, Fields},
};

#[path = "null_encoded.rs"]
mod encoded;

pub(super) fn null_data(dtype: &D, len: usize, node: &str, check: Check<'_>) -> Result<ArrayData> {
    check()?;
    if !needs(dtype, check)? {
        return Ok(ArrayData::new_null(dtype, len));
    }
    let builder = ArrayData::builder(dtype.clone()).len(len);
    match dtype {
        D::List(f) | D::LargeList(f) | D::ListView(f) | D::LargeListView(f) | D::Map(f, _) => {
            list(builder, dtype, f.data_type(), len, node, check)
        }
        D::FixedSizeList(f, width) => fixed(builder, f.data_type(), *width, len, node, check),
        D::Struct(fields) => structure(builder, fields, len, node, check),
        _ => encoded::build(builder, dtype, len, node, check),
    }
}
fn list(
    builder: ArrayDataBuilder,
    dtype: &D,
    value: &D,
    len: usize,
    node: &str,
    check: Check<'_>,
) -> Result<ArrayData> {
    let values = null_data(value, 0, node, check)?;
    let buffers = list_buffers(dtype, len, node)?;
    build(builder.buffers(buffers).child_data(vec![values]), len, node)
}
fn fixed(
    builder: ArrayDataBuilder,
    value: &D,
    width: i32,
    len: usize,
    node: &str,
    check: Check<'_>,
) -> Result<ArrayData> {
    let width = usize::try_from(width).map_err(|_| invalid(node))?;
    let child_len = len.checked_mul(width).ok_or_else(|| invalid(node))?;
    let values = null_data(value, child_len, node, check)?;
    build(builder.child_data(vec![values]), len, node)
}
fn structure(
    builder: ArrayDataBuilder,
    fields: &Fields,
    len: usize,
    node: &str,
    check: Check<'_>,
) -> Result<ArrayData> {
    let mut children = Vec::with_capacity(fields.len());
    for field in fields {
        children.push(null_data(field.data_type(), len, node, check)?);
    }
    build(builder.child_data(children), len, node)
}
fn list_buffers(dtype: &D, len: usize, node: &str) -> Result<Vec<Buffer>> {
    match dtype {
        D::List(_) | D::Map(..) => Ok(vec![zero(offsets(len, 4, node)?)]),
        D::LargeList(_) => Ok(vec![zero(offsets(len, 8, node)?)]),
        D::ListView(_) => pair(len, 4, node),
        D::LargeListView(_) => pair(len, 8, node),
        _ => Err(invalid(node)),
    }
}
fn offsets(len: usize, width: usize, node: &str) -> Result<usize> {
    len.checked_add(1)
        .and_then(|n| n.checked_mul(width))
        .ok_or_else(|| invalid(node))
}
fn pair(len: usize, width: usize, node: &str) -> Result<Vec<Buffer>> {
    let bytes = len.checked_mul(width).ok_or_else(|| invalid(node))?;
    Ok(vec![zero(bytes), zero(bytes)])
}
fn zero(bytes: usize) -> Buffer {
    MutableBuffer::from_len_zeroed(bytes).into()
}
fn build(builder: ArrayDataBuilder, len: usize, node: &str) -> Result<ArrayData> {
    builder
        .nulls(Some(NullBuffer::new_null(len)))
        .build()
        .map_err(|_| invalid(node))
}
