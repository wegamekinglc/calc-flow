use super::super::StreamJoinSpec;
use super::{Construction, Decision, MetadataWork, profile};
use crate::OperatorStateSnapshot;
use datafusion::execution::memory_pool::{MemoryConsumer, MemoryPool, MemoryReservation};
use serde_json::Value;
use std::{
    mem::{align_of, size_of},
    sync::Arc,
};

const ARC_HEADER: usize = 2 * size_of::<usize>();

pub(super) fn required(
    snapshot: &OperatorStateSnapshot,
    spec: &StreamJoinSpec,
    name: &str,
) -> Option<usize> {
    let json = map_bytes(&snapshot.inline_metadata)?;
    let typed = typed_spec(spec)?;
    checked_sum(&[
        json.checked_mul(2)?,
        parser_collect()?,
        typed.checked_mul(2)?,
        validation_controls()?,
        construction_controls(name)?,
    ])
}

fn checked_sum(parts: &[usize]) -> Option<usize> {
    parts
        .iter()
        .try_fold(0usize, |sum, next| sum.checked_add(*next))
}

fn parser_collect() -> Option<usize> {
    tree::<String, Value>(8)?.checked_add(8 * size_of::<(String, Value)>())
}

fn validation_controls() -> Option<usize> {
    tree::<&String, ()>(1)?.checked_add(4 * size_of::<&String>())
}

fn typed_spec(spec: &StreamJoinSpec) -> Option<usize> {
    let text = profile::expected_text_bytes(spec)?;
    // Fields, inner and chosen-prefix overlap, including both default copies.
    text.checked_mul(3)?
        .checked_add(18)?
        .checked_add(10 * size_of::<String>())
}

pub(in crate::operator::join) fn caller_controls(name: &str) -> Option<usize> {
    checked_sum(&[
        registration_controls()?,
        arc_bytes(name.len())?,
        arc_bytes(0)?,
        stop_controls()?,
        size_of::<super::SubmissionControl>(),
    ])
}

fn stop_controls() -> Option<usize> {
    // A u64 job ID displays in at most 20 bytes; old/new String growth <=80.
    checked_sum(&[80, arc_bytes(20)?, arc_bytes(size_of::<bool>())?])
}

pub(in crate::operator::join) fn registration_controls() -> Option<usize> {
    let registration = size_of::<MemoryConsumer>() + size_of::<Arc<dyn MemoryPool>>() + ARC_HEADER;
    let label = "sql-incremental:stream-join-metadata".len();
    registration.checked_add(3 * label)
}

fn arc_bytes(bytes: usize) -> Option<usize> {
    let alignment = align_of::<usize>();
    ARC_HEADER
        .checked_add(bytes)?
        .checked_add(alignment - 1)?
        .checked_div(alignment)?
        .checked_mul(alignment)
}

fn construction_controls(name: &str) -> Option<usize> {
    checked_sum(&[
        name.len(),
        caller_controls(name)?,
        size_of::<Construction>(),
        size_of::<MetadataWork>(),
        size_of::<Decision>(),
        size_of::<MemoryReservation>(),
    ])
}

fn map_bytes(values: &crate::JsonMap) -> Option<usize> {
    values
        .iter()
        .try_fold(tree::<String, Value>(values.len())?, |sum, (key, value)| {
            sum.checked_add(key.len())?.checked_add(value_bytes(value)?)
        })
}

fn value_bytes(value: &Value) -> Option<usize> {
    match value {
        Value::String(text) => Some(text.len()),
        Value::Array(values) => values.iter().try_fold(
            values.len().checked_mul(size_of::<Value>())?,
            |sum, value| sum.checked_add(value_bytes(value)?),
        ),
        Value::Object(values) => values
            .iter()
            .try_fold(tree::<String, Value>(values.len())?, |sum, (key, value)| {
                sum.checked_add(key.len())?.checked_add(value_bytes(value)?)
            }),
        _ => Some(0),
    }
}

fn tree<K, V>(count: usize) -> Option<usize> {
    count.checked_add(1)?.checked_mul(node_bytes::<K, V>()?)
}

fn node_bytes<K, V>() -> Option<usize> {
    let alignment = align_of::<K>()
        .max(align_of::<V>())
        .max(align_of::<usize>());
    let fields = size_of::<usize>() + 2 * size_of::<u16>() + 12 * size_of::<usize>();
    let slots = 11usize.checked_mul(size_of::<K>().checked_add(size_of::<V>())?)?;
    let padded = checked_sum(&[fields, slots, 6 * (alignment - 1)])?;
    padded
        .checked_add(alignment - 1)?
        .checked_div(alignment)?
        .checked_mul(alignment)
}
