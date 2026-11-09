use super::{Check, invalid, null_data, zero};
use crate::Result;
use datafusion::arrow::{
    array::ArrayDataBuilder,
    buffer::Buffer,
    datatypes::{DataType as D, UnionFields, UnionMode},
};

pub(super) fn build(
    builder: ArrayDataBuilder,
    dtype: &D,
    len: usize,
    node: &str,
    check: Check<'_>,
) -> Result<datafusion::arrow::array::ArrayData> {
    match dtype {
        D::Dictionary(key, value) => dictionary(builder, key, value, len, node, check),
        D::RunEndEncoded(ends, value) => run(
            builder,
            ends.data_type(),
            value.data_type(),
            len,
            node,
            check,
        ),
        D::Union(fields, mode) => union(builder, fields, *mode, len, node, check),
        _ => Err(invalid(node)),
    }
}
fn dictionary(
    builder: ArrayDataBuilder,
    key: &D,
    value: &D,
    len: usize,
    node: &str,
    check: Check<'_>,
) -> Result<datafusion::arrow::array::ArrayData> {
    let width = key.primitive_width().ok_or_else(|| invalid(node))?;
    let bytes = len.checked_mul(width).ok_or_else(|| invalid(node))?;
    let child = null_data(value, 0, node, check)?;
    super::build(
        builder.buffers(vec![zero(bytes)]).child_data(vec![child]),
        len,
        node,
    )
}
fn run(
    builder: ArrayDataBuilder,
    ends: &D,
    value: &D,
    len: usize,
    node: &str,
    check: Check<'_>,
) -> Result<datafusion::arrow::array::ArrayData> {
    let ends = run_ends(ends, len, node)?;
    let value = null_data(value, usize::from(len > 0), node, check)?;
    builder
        .child_data(vec![ends, value])
        .build()
        .map_err(|_| invalid(node))
}
fn run_ends(dtype: &D, len: usize, node: &str) -> Result<datafusion::arrow::array::ArrayData> {
    if len == 0 {
        return Ok(datafusion::arrow::array::ArrayData::new_empty(dtype));
    }
    let buffer = run_buffer(dtype, len, node)?;
    ArrayDataBuilder::new(dtype.clone())
        .len(1)
        .buffers(vec![buffer])
        .build()
        .map_err(|_| invalid(node))
}
fn run_buffer(dtype: &D, len: usize, node: &str) -> Result<Buffer> {
    let buffer = match dtype {
        D::Int16 => Buffer::from_slice_ref([i16::try_from(len).map_err(|_| invalid(node))?]),
        D::Int32 => Buffer::from_slice_ref([i32::try_from(len).map_err(|_| invalid(node))?]),
        D::Int64 => Buffer::from_slice_ref([i64::try_from(len).map_err(|_| invalid(node))?]),
        _ => return Err(invalid(node)),
    };
    Ok(buffer)
}
fn union(
    builder: ArrayDataBuilder,
    fields: &UnionFields,
    mode: UnionMode,
    len: usize,
    node: &str,
    check: Check<'_>,
) -> Result<datafusion::arrow::array::ArrayData> {
    let buffers = union_buffers(fields, mode, len, node)?;
    let mut children = Vec::with_capacity(fields.len());
    for (index, (_, field)) in fields.iter().enumerate() {
        let length = if mode == UnionMode::Sparse || index == 0 {
            len
        } else {
            0
        };
        children.push(null_data(field.data_type(), length, node, check)?);
    }
    builder
        .buffers(buffers)
        .child_data(children)
        .build()
        .map_err(|_| invalid(node))
}
fn union_buffers(
    fields: &UnionFields,
    mode: UnionMode,
    len: usize,
    node: &str,
) -> Result<Vec<Buffer>> {
    let id = match fields.iter().next() {
        Some((id, _)) => id,
        None if len == 0 => 0,
        None => return Err(invalid(node)),
    };
    let ids = std::iter::repeat_n(id, len).collect::<Buffer>();
    if mode == UnionMode::Dense {
        Ok(vec![ids, offsets(len, node)?])
    } else {
        Ok(vec![ids])
    }
}

fn offsets(len: usize, node: &str) -> Result<Buffer> {
    let end = i32::try_from(len).map_err(|_| invalid(node))?;
    Ok((0..end).collect::<Buffer>())
}
