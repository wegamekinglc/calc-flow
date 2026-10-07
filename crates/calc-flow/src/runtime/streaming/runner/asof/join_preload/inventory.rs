use std::{
    mem::{align_of, size_of},
    sync::Arc,
};

use datafusion::execution::memory_pool::{MemoryConsumer, MemoryPool, MemoryReservation};
use serde_json::Value;

use crate::{OperatorManifestEntry, OperatorStateSnapshot, StateHandle, StateSegment};

const ARC_HEADER: usize = 2 * size_of::<usize>();
const TREE_SLOTS: usize = 11;
const TREE_EDGES: usize = 12;

pub(super) fn request_bytes(id: &str, entry: &OperatorManifestEntry) -> Option<usize> {
    let handles = entry.segments.len().checked_mul(size_of::<StateHandle>())?;
    let metadata = metadata_bytes(&entry.inline_metadata)?.checked_mul(2)?;
    let initial = id
        .len()
        .checked_add(handles)?
        .checked_add(metadata)?
        .checked_add(reservation_bytes())?;
    entry.segments.iter().try_fold(initial, |bytes, handle| {
        bytes.checked_add(segment_bytes(handle)?)
    })
}

fn segment_bytes(handle: &StateHandle) -> Option<usize> {
    let strings = [
        handle.operator_id(),
        handle.segment_id(),
        handle.relative_path(),
        handle.sha256(),
    ];
    let cloned = strings
        .iter()
        .try_fold(0usize, |bytes, value| bytes.checked_add(value.len()))?;
    let controls = segment_controls(handle)?;
    usize::try_from(handle.byte_len())
        .ok()?
        .checked_add(cloned)?
        .checked_add(controls)
}

fn segment_controls(handle: &StateHandle) -> Option<usize> {
    let strings = handle
        .segment_id()
        .len()
        .checked_add(handle.sha256().len())?;
    tree_node::<String, StateSegment>()?
        .checked_add(strings)?
        .checked_add(size_of::<Vec<u8>>() + ARC_HEADER)
}

fn metadata_bytes(values: &crate::JsonMap) -> Option<usize> {
    let nodes = values
        .len()
        .checked_add(1)?
        .checked_mul(tree_node::<String, Value>()?)?;
    values.iter().try_fold(nodes, |bytes, (key, value)| {
        bytes
            .checked_add(key.len())?
            .checked_add(value_bytes(value)?)
    })
}

fn value_bytes(value: &Value) -> Option<usize> {
    match value {
        Value::String(value) => Some(value.len()),
        Value::Array(values) => values.iter().try_fold(
            values.len().checked_mul(size_of::<Value>())?,
            |bytes, value| bytes.checked_add(value_bytes(value)?),
        ),
        Value::Object(values) => object_bytes(values),
        _ => Some(0),
    }
}

fn object_bytes(values: &serde_json::Map<String, Value>) -> Option<usize> {
    let nodes = values
        .len()
        .checked_add(1)?
        .checked_mul(tree_node::<String, Value>()?)?;
    values.iter().try_fold(nodes, |bytes, (key, value)| {
        bytes
            .checked_add(key.len())?
            .checked_add(value_bytes(value)?)
    })
}

// Rust 1.88 BTree nodes contain 11 slots and up to 12 child pointers; each node is nonempty.
fn tree_node<K, V>() -> Option<usize> {
    let alignment = align_of::<K>()
        .max(align_of::<V>())
        .max(align_of::<usize>());
    let header = size_of::<usize>() + 2 * size_of::<u16>();
    let slots = TREE_SLOTS.checked_mul(size_of::<K>().checked_add(size_of::<V>())?)?;
    let edges = TREE_EDGES * size_of::<usize>();
    // Five leaf fields plus the child array; bound padding regardless of Rust field ordering.
    let padding = 6 * (alignment - 1);
    round(
        header
            .checked_add(slots)?
            .checked_add(edges)?
            .checked_add(padding)?,
        alignment,
    )
}

fn reservation_bytes() -> usize {
    let name = "sql-incremental:stream-join-preload".len();
    size_of::<MemoryConsumer>()
        + size_of::<Arc<dyn MemoryPool>>()
        + ARC_HEADER
        + 3 * name
        + size_of::<MemoryReservation>()
        + ARC_HEADER
}

pub(super) fn future_bytes<F>() -> Option<usize> {
    // Tokio 1.52 Cell: Header, Arc scheduler, task ID, Stage union and Trailer; 256 is its maximum alignment.
    let header = 3 * size_of::<usize>() + size_of::<u64>();
    let trailer = 2 * size_of::<usize>()
        + size_of::<Option<std::task::Waker>>()
        + size_of::<Option<Arc<dyn Fn() + Send + Sync>>>();
    let stage = size_of::<super::FundedLoad<F>>()
        .checked_add(size_of::<
            Result<crate::Result<OperatorStateSnapshot>, tokio::task::JoinError>,
        >())?
        .checked_add(size_of::<usize>())?;
    let task = header
        .checked_add(size_of::<usize>() + size_of::<u64>())?
        .checked_add(stage)?
        .checked_add(trailer)?;
    size_of::<F>().checked_add(round(task, 256)?)
}

fn round(bytes: usize, alignment: usize) -> Option<usize> {
    bytes
        .checked_add(alignment - 1)
        .map(|bytes| bytes / alignment * alignment)
}
