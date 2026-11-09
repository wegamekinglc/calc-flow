//! Controlled copying of already validated layout-five inline metadata.

use super::super::super::math_requirements::bytes;
use crate::{CalcFlowError, JsonMap, Result};
use serde_json::{Map, Value};
use std::{
    alloc::Layout,
    collections::BTreeMap,
    mem::{align_of, size_of},
};

type Check<'a> = &'a mut dyn FnMut() -> Result<()>;

pub(in super::super::super) fn requirement(fields: &JsonMap, check: Check<'_>) -> Result<u64> {
    check()?;
    let containers = bytes::tree_nodes::<String, Value>(fields.len())?;
    add(containers, pairs(fields.iter(), check)?)
}
pub(in super::super::super) fn segment_requirement(id: Option<&str>) -> Result<u64> {
    let Some(id) = id else {
        return Ok(0);
    };
    add(
        bytes::tree_nodes::<String, crate::StateSegment>(1)?,
        text(id)?,
    )
}
fn pairs<'a>(
    mut fields: impl Iterator<Item = (&'a String, &'a Value)>,
    check: Check<'_>,
) -> Result<u64> {
    fields.try_fold(0, |sum, (key, value)| {
        check()?;
        add(add(sum, text(key)?)?, value_requirement(value, check)?)
    })
}
fn value_requirement(value: &Value, check: Check<'_>) -> Result<u64> {
    check()?;
    match value {
        Value::String(value) => text(value),
        Value::Array(values) => array_requirement(values, check),
        Value::Object(fields) => object_requirement(fields, check),
        // The pinned serde_json feature set stores numbers inline.
        Value::Null | Value::Bool(_) | Value::Number(_) => Ok(0),
    }
}
fn array_requirement(values: &[Value], check: Check<'_>) -> Result<u64> {
    let slots = allocation(values.len(), Layout::new::<Value>())?;
    values.iter().try_fold(slots, |sum, value| {
        add(sum, value_requirement(value, check)?)
    })
}
fn object_requirement(fields: &Map<String, Value>, check: Check<'_>) -> Result<u64> {
    let containers = object_containers(fields.len())?;
    add(containers, pairs(fields.iter(), check)?)
}
fn object_containers(count: usize) -> Result<u64> {
    if count == 0 {
        return Ok(0);
    }
    // Pinned indexmap 2.14 Bucket has hash/owned key/value; pad each field conservatively.
    let size = size_of::<usize>()
        + size_of::<String>()
        + size_of::<Value>()
        + align_of::<usize>()
        + align_of::<String>()
        + align_of::<Value>();
    let alignment = align_of::<usize>()
        .max(align_of::<String>())
        .max(align_of::<Value>());
    let bucket = Layout::from_size_align(size, alignment)
        .map_err(|_| invalid())?
        .pad_to_align();
    add(allocation(count, bucket)?, indices(count)?)
}
fn indices(count: usize) -> Result<u64> {
    let buckets = count
        .checked_mul(8)
        .and_then(|n| n.checked_add(6))
        .ok_or_else(invalid)?
        / 7;
    let buckets = buckets
        .checked_next_power_of_two()
        .ok_or_else(invalid)?
        .max(4);
    let payload = buckets
        .checked_mul(size_of::<usize>() + 1)
        .and_then(|n| n.checked_add(16))
        .ok_or_else(invalid)?;
    // Hashbrown's control group is at most 16 bytes on the supported 32/64-bit targets.
    let layout = Layout::from_size_align(payload, 16).map_err(|_| invalid())?;
    length(layout.size())
}
fn allocation(count: usize, element: Layout) -> Result<u64> {
    let size = count.checked_mul(element.size()).ok_or_else(invalid)?;
    let layout = Layout::from_size_align(size, element.align()).map_err(|_| invalid())?;
    length(layout.size())
}
fn text(value: &str) -> Result<u64> {
    allocation(value.len(), Layout::new::<u8>())
}
fn length(size: usize) -> Result<u64> {
    u64::try_from(size).map_err(|_| invalid())
}
fn add(left: u64, right: u64) -> Result<u64> {
    left.checked_add(right).ok_or_else(invalid)
}
fn invalid() -> CalcFlowError {
    CalcFlowError::CheckpointMismatch {
        message: "rolling state.inline_metadata allocation size is invalid".into(),
    }
}

pub(in super::super::super) fn copy(fields: &JsonMap, check: Check<'_>) -> Result<JsonMap> {
    check()?;
    let mut result = BTreeMap::new();
    for (key, value) in fields {
        check()?;
        result.insert(key.clone(), copy_value(value, check)?);
    }
    check()?;
    Ok(result)
}
fn copy_value(value: &Value, check: Check<'_>) -> Result<Value> {
    check()?;
    match value {
        Value::Array(values) => copy_array(values, check).map(Value::Array),
        Value::Object(fields) => copy_object(fields, check).map(Value::Object),
        other => Ok(other.clone()),
    }
}
fn copy_array(values: &[Value], check: Check<'_>) -> Result<Vec<Value>> {
    let mut result = Vec::with_capacity(values.len());
    for value in values {
        result.push(copy_value(value, check)?);
    }
    Ok(result)
}
fn copy_object(fields: &Map<String, Value>, check: Check<'_>) -> Result<Map<String, Value>> {
    let mut result = Map::with_capacity(fields.len());
    for (key, value) in fields {
        check()?;
        result.insert(key.clone(), copy_value(value, check)?);
    }
    Ok(result)
}

#[cfg(test)]
#[path = "inline_tests.rs"]
mod tests;
