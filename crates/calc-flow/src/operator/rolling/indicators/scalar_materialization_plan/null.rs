use super::cost::{Cost, buffer, times};
use super::descriptor;
use super::{Check, Result, add, invalid};
use datafusion::{arrow::datatypes::DataType as D, common::ScalarValue};

#[path = "null_array.rs"]
mod arrays;
#[path = "null_buffers.rs"]
mod buffers;

pub(super) fn scalar(dtype: &D, check: Check<'_>) -> Result<Cost> {
    check()?;
    match dtype {
        D::Dictionary(key, value) => scalar(value, check)?.combine(dictionary_box(key, check)?),
        D::RunEndEncoded(_, value) => run_scalar(value.data_type(), check),
        D::Union(..) => Ok(Cost::default()),
        _ => nested(dtype, check),
    }
}
fn dictionary_box(key: &D, check: Check<'_>) -> Result<Cost> {
    let boxes = add(times::<D>(1)?, descriptor::dtype(key, check)?)?;
    Ok(Cost::owned(add(boxes, times::<ScalarValue>(1)?)?))
}
fn nested(dtype: &D, check: Check<'_>) -> Result<Cost> {
    if super::super::materialize::canonical::requires_null_fallback(dtype, check)? {
        return arrays::controlled(dtype, 1, check);
    }
    match dtype {
        D::List(field) | D::LargeList(field) => list(dtype, field.data_type(), check),
        D::FixedSizeList(field, width) => fixed(dtype, field.data_type(), *width, check),
        D::ListView(field) | D::LargeListView(field) => list_view(dtype, field.data_type(), check),
        _ => composite(dtype, check),
    }
}
fn composite(dtype: &D, check: Check<'_>) -> Result<Cost> {
    match dtype {
        D::Struct(_) | D::Map(..) => {
            let fresh = arrays::array(dtype, 1, check)?;
            let clone = add(descriptor::owner(dtype)?, arrays::refs(dtype)?)?;
            fresh.cost.extra(clone)
        }
        _ => Ok(Cost::default()),
    }
}
fn list(dtype: &D, child: &D, check: Check<'_>) -> Result<Cost> {
    let width = if matches!(dtype, D::List(_)) { 4 } else { 8 };
    let own = with_mask(dtype, buffer(width * 2)?)?;
    arrays::array(child, 0, check)?.cost.combine(own)
}
fn fixed(dtype: &D, child: &D, width: i32, check: Check<'_>) -> Result<Cost> {
    let len = usize::try_from(width).map_err(|_| invalid())?;
    arrays::array(child, len, check)?
        .cost
        .combine(with_mask(dtype, 0)?)
}
fn list_view(dtype: &D, child: &D, check: Check<'_>) -> Result<Cost> {
    let width = if matches!(dtype, D::ListView(_)) {
        4
    } else {
        8
    };
    let storage = buffers::pair(width, width)?;
    arrays::array(child, 0, check)?
        .cost
        .combine(with_mask(dtype, storage)?)
}
fn with_mask(dtype: &D, bytes: u64) -> Result<Cost> {
    let raw = add(bytes, buffer(1)?)?;
    Ok(Cost::owned(add(descriptor::owner(dtype)?, raw)?))
}

fn run_scalar(value: &D, check: Check<'_>) -> Result<Cost> {
    scalar(value, check)?.combine(Cost::owned(times::<ScalarValue>(1)?))
}

pub(super) fn array_profile(
    dtype: &D,
    len: usize,
    check: Check<'_>,
) -> Result<super::NullArrayCost> {
    let fallback = super::super::materialize::canonical::requires_null_fallback(dtype, check)?;
    let fresh = arrays::array(dtype, len, check)?;
    let cost = if fallback {
        fresh.validated_cost()?
    } else {
        fresh.cost
    };
    Ok(super::NullArrayCost {
        logical_owned: fresh.logical_owned,
        construction_peak: add(cost.owned, cost.temporary)?,
    })
}
