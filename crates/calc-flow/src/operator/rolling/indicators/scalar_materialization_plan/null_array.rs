use super::super::{Check, Result, add, invalid};
use super::super::{
    cost::{Cost, product, times},
    descriptor,
};
use super::buffers;
use datafusion::arrow::{
    array::ArrayData,
    datatypes::{DataType as D, Fields, UnionFields, UnionMode},
};

#[derive(Clone, Copy, Default)]
pub(super) struct Fresh {
    pub cost: Cost,
    pub(super) logical_owned: u64,
    metadata: u64,
    validation: u64,
}
impl Fresh {
    fn combine(self, other: Self) -> Result<Self> {
        Ok(Self {
            cost: self.cost.combine(other.cost)?,
            logical_owned: add(self.logical_owned, other.logical_owned)?,
            metadata: add(self.metadata, other.metadata)?,
            validation: add(self.validation, other.validation)?,
        })
    }
}
pub(super) fn array(dtype: &D, len: usize, check: Check<'_>) -> Result<Fresh> {
    check()?;
    match dtype {
        D::List(field) | D::LargeList(field) | D::ListView(field) | D::LargeListView(field) => {
            finish(dtype, len, 1, array(field.data_type(), 0, check)?, check)
        }
        D::FixedSizeList(field, width) => fixed(dtype, field.data_type(), *width, len, check),
        D::Struct(fields) => structure(dtype, fields, len, check),
        _ => encoded(dtype, len, check),
    }
}
fn fixed(dtype: &D, value: &D, width: i32, len: usize, check: Check<'_>) -> Result<Fresh> {
    let width = usize::try_from(width).map_err(|_| invalid())?;
    let child = array(value, product(len, width)?, check)?;
    let extra = child.metadata;
    let mut output = finish(dtype, len, 1, child, check)?;
    output.cost = output.cost.extra(extra)?;
    Ok(output)
}
fn structure(dtype: &D, fields: &Fields, len: usize, check: Check<'_>) -> Result<Fresh> {
    let mut children = Fresh::default();
    for field in fields {
        children = children.combine(array(field.data_type(), len, check)?)?;
    }
    finish(dtype, len, fields.len(), children, check)
}
fn encoded(dtype: &D, len: usize, check: Check<'_>) -> Result<Fresh> {
    match dtype {
        D::Map(field, _) => map(dtype, field.data_type(), len, check),
        D::Dictionary(_, value) => finish(dtype, len, 1, array(value, 0, check)?, check),
        D::RunEndEncoded(ends, value) => {
            run(dtype, ends.data_type(), value.data_type(), len, check)
        }
        D::Union(fields, mode) => union(dtype, fields, *mode, len, check),
        _ => finish(dtype, len, 0, Fresh::default(), check),
    }
}
fn map(dtype: &D, entries: &D, len: usize, check: Check<'_>) -> Result<Fresh> {
    let mut child = array(entries, 0, check)?;
    let owner = descriptor::owner(entries)?;
    child.cost.owned = child.cost.owned.checked_sub(owner).ok_or_else(invalid)?;
    child.logical_owned = child.logical_owned.checked_sub(owner).ok_or_else(invalid)?;
    finish(dtype, len, 1, child, check)
}
fn union(
    dtype: &D,
    fields: &UnionFields,
    mode: UnionMode,
    len: usize,
    check: Check<'_>,
) -> Result<Fresh> {
    if fields.is_empty() && len > 0 {
        return Err(invalid());
    }
    let children = union_children(fields, mode, len, check)?;
    finish(dtype, len, fields.len(), children, check)
}
fn union_children(
    fields: &UnionFields,
    mode: UnionMode,
    len: usize,
    check: Check<'_>,
) -> Result<Fresh> {
    let mut children = Fresh::default();
    for (index, (_, field)) in fields.iter().enumerate() {
        let size = if mode == UnionMode::Sparse || index == 0 {
            len
        } else {
            0
        };
        children = children.combine(array(field.data_type(), size, check)?)?;
    }
    Ok(children)
}
fn run(dtype: &D, ends: &D, value: &D, len: usize, check: Check<'_>) -> Result<Fresh> {
    let mut child = run_child(ends, value, len, check)?;
    let meta = descriptor::metadata(ends, 1, 0, check)?;
    let raw = buffers::run_ends(len)?;
    child.cost = append_run_storage(child.cost, raw.constructor_owned, meta, len)?;
    child = append_end_metadata(child, ends, meta)?;
    child.logical_owned = add(child.logical_owned, raw.logical_owned)?;
    finish(dtype, len, 2, child, check)
}
fn finish(dtype: &D, len: usize, children: usize, child: Fresh, check: Check<'_>) -> Result<Fresh> {
    let metadata = metadata(dtype, children, child.metadata, check)?;
    let local = local(dtype, len, check)?;
    let validation = add(child.validation, validation(dtype, len)?)?;
    child.with_local(local, metadata, validation)
}

fn metadata(dtype: &D, children: usize, child: u64, check: Check<'_>) -> Result<u64> {
    add(
        child,
        descriptor::metadata(dtype, descriptor::buffer_count(dtype), children, check)?,
    )
}
#[derive(Clone, Copy)]
struct Local {
    cost: Cost,
    logical_owned: u64,
}
fn local(dtype: &D, len: usize, check: Check<'_>) -> Result<Local> {
    let mut local = local_storage(dtype, len, check)?;
    local.cost = local.cost.extra(empty_mask(dtype, len)?)?;
    Ok(local)
}
fn local_storage(dtype: &D, len: usize, check: Check<'_>) -> Result<Local> {
    let own = add(descriptor::owned(dtype, check)?, refs(dtype)?)?;
    Local::new(own, buffers::raw(dtype, len)?, view(dtype)?)
}
impl Local {
    fn new(own: u64, raw: buffers::BufferCost, view: Cost) -> Result<Self> {
        let logical_owned = add(add(own, raw.logical_owned)?, view.owned)?;
        let cost = Cost::owned(add(own, raw.constructor_owned)?).combine(view)?;
        Ok(Self {
            cost,
            logical_owned,
        })
    }
}
impl Fresh {
    fn with_local(self, local: Local, metadata: u64, validation: u64) -> Result<Self> {
        Ok(Self {
            cost: self.cost.combine(local.cost)?.extra(metadata)?,
            logical_owned: add(self.logical_owned, local.logical_owned)?,
            metadata,
            validation,
        })
    }
    pub(super) fn validated_cost(self) -> Result<Cost> {
        self.cost.extra(self.validation)
    }
}
fn view(dtype: &D) -> Result<Cost> {
    if matches!(dtype, D::Utf8View | D::BinaryView) {
        return Ok(Cost::owned(descriptor::views(0)?));
    }
    Ok(Cost::default())
}
fn empty_mask(dtype: &D, len: usize) -> Result<u64> {
    if len == 0 && buffers::nullable(dtype) {
        return super::super::cost::buffer(0);
    }
    Ok(0)
}
pub(super) fn refs(dtype: &D) -> Result<u64> {
    match dtype {
        D::Struct(fields) => descriptor::vector(dtype, fields.len()),
        D::Map(..) => times::<ArrayData>(2),
        D::Union(fields, _) => descriptor::vector(dtype, fields.len()),
        _ => Ok(0),
    }
}

fn append_run_storage(cost: Cost, raw: u64, meta: u64, len: usize) -> Result<Cost> {
    cost.combine(Cost::owned(raw))?
        .extra(end_temporary(meta, len)?)
}
fn end_temporary(meta: u64, len: usize) -> Result<u64> {
    if len == 0 {
        return add(meta, super::super::cost::buffer(0)?);
    }
    Ok(meta)
}
fn append_end_metadata(mut child: Fresh, ends: &D, meta: u64) -> Result<Fresh> {
    child.metadata = add(child.metadata, meta)?;
    child.validation = add(child.validation, descriptor::layout(ends)?)?;
    Ok(child)
}

fn run_child(ends: &D, value: &D, len: usize, check: Check<'_>) -> Result<Fresh> {
    validate_run_length(ends, len)?;
    array(value, usize::from(len > 0), check)
}
fn validate_run_length(ends: &D, len: usize) -> Result<()> {
    let fits = match ends {
        D::Int16 => i16::try_from(len).is_ok(),
        D::Int32 => i32::try_from(len).is_ok(),
        D::Int64 => i64::try_from(len).is_ok(),
        _ => false,
    };
    if fits { Ok(()) } else { Err(invalid()) }
}

pub(super) fn controlled(dtype: &D, len: usize, check: Check<'_>) -> Result<Cost> {
    array(dtype, len, check)?.validated_cost()
}
fn validation(dtype: &D, len: usize) -> Result<u64> {
    let layout = descriptor::layout(dtype)?;
    let extra = validation_extra(dtype, len)?;
    add(layout, extra)
}
fn validation_extra(dtype: &D, len: usize) -> Result<u64> {
    let bytes = match dtype {
        D::Union(..) => times::<Option<usize>>(128)?,
        D::FixedSizeList(field, width) if !field.is_nullable() && len > 0 => {
            expanded_mask(*width, len)?
        }
        _ => 0,
    };
    Ok(bytes)
}
fn expanded_mask(width: i32, len: usize) -> Result<u64> {
    let width = usize::try_from(width).map_err(|_| invalid())?;
    super::super::cost::buffer(super::super::cost::bits(product(len, width)?))
}
