use super::cost::{Cost, buffer, product, times};
use super::descriptor;
use super::{Check, Result, add, invalid, typed};
use datafusion::arrow::{
    array::{
        Array, BinaryViewArray, FixedSizeListArray, GenericListArray, GenericListViewArray,
        MapArray, OffsetSizeTrait, StringViewArray, StructArray, UnionArray,
    },
    datatypes::{DataType as D, UnionMode},
};

#[path = "canonical_encoded.rs"]
mod encoded;

#[derive(Clone, Copy, Default)]
pub(super) struct Graph {
    pub cost: Cost,
    pub metadata: u64,
    pub validation: u64,
}
impl Graph {
    fn combine(self, other: Self) -> Result<Self> {
        Ok(Self {
            cost: self.cost.combine(other.cost)?,
            metadata: add(self.metadata, other.metadata)?,
            validation: add(self.validation, other.validation)?,
        })
    }
}
pub(super) fn direct(array: &dyn Array, row: usize, check: Check<'_>) -> Result<Option<Cost>> {
    let dtype = array.data_type();
    if !matches!(
        dtype,
        D::List(_)
            | D::LargeList(_)
            | D::FixedSizeList(..)
            | D::ListView(_)
            | D::LargeListView(_)
            | D::Struct(_)
            | D::Map(..)
    ) {
        return Ok(None);
    }
    Ok(Some(graph(array, row, 1, check)?.cost))
}
pub(super) fn graph(
    array: &dyn Array,
    offset: usize,
    len: usize,
    check: Check<'_>,
) -> Result<Graph> {
    check()?;
    if offset.checked_add(len).is_none_or(|end| end > array.len()) {
        return Err(invalid());
    }
    match array.data_type() {
        D::List(_) => list::<i32>(array, offset, len, check),
        D::LargeList(_) => list::<i64>(array, offset, len, check),
        D::ListView(_) => list_view::<i32>(array, offset, len, check),
        D::LargeListView(_) => list_view::<i64>(array, offset, len, check),
        _ => nested(array, offset, len, check),
    }
}
fn nested(array: &dyn Array, offset: usize, len: usize, check: Check<'_>) -> Result<Graph> {
    match array.data_type() {
        D::FixedSizeList(..) => fixed(array, offset, len, check),
        D::Struct(_) => structure(array, offset, len, check),
        D::Map(..) => map(array, len, check),
        D::Union(..) => union(array, offset, len, check),
        D::Dictionary(..) | D::RunEndEncoded(..) => encoded::graph(array, check),
        _ => leaf(array, check),
    }
}
fn list<O: OffsetSizeTrait>(
    array: &dyn Array,
    _offset: usize,
    _len: usize,
    check: Check<'_>,
) -> Result<Graph> {
    let array = typed::<GenericListArray<O>>(array)?;
    let child = graph(array.values().as_ref(), 0, array.values().len(), check)?;
    finish(array.data_type(), 1, 1, child, check)
}
fn list_view<O: OffsetSizeTrait>(
    array: &dyn Array,
    _offset: usize,
    _len: usize,
    check: Check<'_>,
) -> Result<Graph> {
    let array = typed::<GenericListViewArray<O>>(array)?;
    let child = graph(array.values().as_ref(), 0, array.values().len(), check)?;
    finish(array.data_type(), 2, 1, child, check)
}
fn fixed(array: &dyn Array, offset: usize, len: usize, check: Check<'_>) -> Result<Graph> {
    let array = typed::<FixedSizeListArray>(array)?;
    let child = fixed_child(array, offset, len, check)?;
    let scratch = fixed_validation(array, len)?;
    let extra = add(child.metadata, scratch)?;
    let mut result = finish(array.data_type(), 0, 1, child, check)?;
    result.cost = result.cost.extra(extra)?;
    Ok(result)
}
fn fixed_child(
    array: &FixedSizeListArray,
    offset: usize,
    len: usize,
    check: Check<'_>,
) -> Result<Graph> {
    let width = usize::try_from(array.value_length()).map_err(|_| invalid())?;
    graph(
        array.values().as_ref(),
        product(offset, width)?,
        product(len, width)?,
        check,
    )
}
fn fixed_validation(array: &FixedSizeListArray, len: usize) -> Result<u64> {
    let D::FixedSizeList(field, _) = array.data_type() else {
        return Err(invalid());
    };
    if field.is_nullable() || array.nulls().is_none() {
        return Ok(0);
    }
    let width = usize::try_from(array.value_length()).map_err(|_| invalid())?;
    buffer(super::cost::bits(product(len, width)?))
}
fn structure(array: &dyn Array, offset: usize, len: usize, check: Check<'_>) -> Result<Graph> {
    let array = typed::<StructArray>(array)?;
    let mut children = Graph::default();
    for child in array.columns() {
        children = children.combine(graph(child.as_ref(), offset, len, check)?)?;
    }
    finish(array.data_type(), 0, array.columns().len(), children, check)
}
fn map(array: &dyn Array, _len: usize, check: Check<'_>) -> Result<Graph> {
    let array = typed::<MapArray>(array)?;
    let mut child = graph(array.entries(), 0, array.entries().len(), check)?;
    child.cost.owned = child
        .cost
        .owned
        .checked_sub(descriptor::owner(array.entries().data_type())?)
        .ok_or_else(invalid)?;
    finish(array.data_type(), 1, 1, child, check)
}
fn union(array: &dyn Array, offset: usize, len: usize, check: Check<'_>) -> Result<Graph> {
    let array = typed::<UnionArray>(array)?;
    let D::Union(fields, mode) = array.data_type() else {
        return Err(invalid());
    };
    let children = union_children(array, fields, *mode, offset, len, check)?;
    let mut result = finish(
        array.data_type(),
        descriptor::buffer_count(array.data_type()),
        fields.len(),
        children,
        check,
    )?;
    result.cost = result.cost.extra(times::<Option<usize>>(128)?)?;
    Ok(result)
}
fn union_children(
    array: &UnionArray,
    fields: &datafusion::arrow::datatypes::UnionFields,
    mode: UnionMode,
    offset: usize,
    len: usize,
    check: Check<'_>,
) -> Result<Graph> {
    let mut children = Graph::default();
    for (id, _) in fields.iter() {
        let child = array.child(id);
        let (offset, len) = if mode == UnionMode::Sparse {
            (offset, len)
        } else {
            (0, child.len())
        };
        children = children.combine(graph(child.as_ref(), offset, len, check)?)?;
    }
    Ok(children)
}
fn leaf(array: &dyn Array, check: Check<'_>) -> Result<Graph> {
    let dtype = array.data_type();
    let buffers = view_buffers(array)?.unwrap_or_else(|| descriptor::buffer_count(dtype));
    let mut result = finish(dtype, buffers, 0, Graph::default(), check)?;
    // Typed to_data followed by ArrayData::slice owns two descriptor copies at its peak.
    result.cost = result.cost.extra(result.metadata)?;
    if matches!(dtype, D::Utf8View | D::BinaryView) {
        return finish_view(result, buffers);
    }
    Ok(result)
}
fn view_buffers(array: &dyn Array) -> Result<Option<usize>> {
    let count = match array.data_type() {
        D::Utf8View => typed::<StringViewArray>(array)?.data_buffers().len(),
        D::BinaryView => typed::<BinaryViewArray>(array)?.data_buffers().len(),
        _ => return Ok(None),
    };
    Ok(Some(count.checked_add(1).ok_or_else(invalid)?))
}
fn finish(
    dtype: &D,
    buffers: usize,
    children: usize,
    child: Graph,
    check: Check<'_>,
) -> Result<Graph> {
    let metadata = metadata(dtype, buffers, children, child.metadata, check)?;
    let local = local_cost(dtype, children, check)?;
    let validation = validation(dtype, child.validation)?;
    let cost = child
        .cost
        .combine(local)?
        .extra(add(metadata, validation)?)?;
    Ok(Graph {
        cost,
        metadata,
        validation,
    })
}
fn local_cost(dtype: &D, children: usize, check: Check<'_>) -> Result<Cost> {
    Ok(Cost::owned(add(
        descriptor::owned(dtype, check)?,
        descriptor::vector(dtype, children)?,
    )?))
}

fn validation(dtype: &D, child: u64) -> Result<u64> {
    add(descriptor::layout(dtype)?, child)
}

fn metadata(
    dtype: &D,
    buffers: usize,
    children: usize,
    child: u64,
    check: Check<'_>,
) -> Result<u64> {
    add(
        descriptor::metadata(dtype, buffers, children, check)?,
        child,
    )
}

fn finish_view(mut graph: Graph, buffers: usize) -> Result<Graph> {
    let data = buffers.checked_sub(1).ok_or_else(invalid)?;
    graph.cost.owned = add(graph.cost.owned, descriptor::views(data)?)?;
    graph.cost = graph
        .cost
        .extra(descriptor::view_conversion_extra(buffers)?)?;
    Ok(graph)
}
