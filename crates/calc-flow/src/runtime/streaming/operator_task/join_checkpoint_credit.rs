use std::{collections::BTreeMap, mem::align_of, sync::Arc};

use datafusion::execution::memory_pool::MemoryReservation;
use serde_json::Value;

use super::BindingOrdinal;
use crate::{
    CalcFlowError, JsonMap, OperatorIngressManifestEntry, OperatorManifestEntry,
    OperatorStateSnapshot, Result, StateHandle,
};

const STATE_VERSION_KEY: &str = "__calc_flow_operator_state_version";
const FRONTIER_KEY: &str = crate::pipeline::OUTPUT_FRONTIER_METADATA_KEY_V1;
const MAX_PATH: usize = "committed/".len() + 64 + 1 + 64 + 1 + 20 + 1 + 64 + ".segment".len();

pub(in crate::runtime::streaming) fn admit_envelope(
    credit: &MemoryReservation,
    node_id: &str,
    snapshot: &OperatorStateSnapshot,
    ingresses: &BTreeMap<String, BindingOrdinal>,
) -> Result<()> {
    let progress = progress_bytes(ingresses.keys().map(String::as_str), ingresses.len())?;
    let handles = snapshot.segments.keys().try_fold(
        product(snapshot.segments.len(), size_of::<StateHandle>())?,
        |bytes, segment| add(bytes, handle_strings(node_id, segment, MAX_PATH, 64)?),
    )?;
    // Existing writer metadata/segments retain their independent capture credit.
    // These are the allocations added by capability/frontier and the ACK envelope.
    let bytes = envelope_bytes(node_id, snapshot.segments.len(), progress, handles)?;
    reserve(credit, bytes)
}

fn envelope_bytes(node_id: &str, count: usize, progress: usize, handles: usize) -> Result<usize> {
    sum(&[
        node_id.len(),
        progress,
        product(2, tree_node::<String, Value>()?)?,
        STATE_VERSION_KEY.len(),
        FRONTIER_KEY.len(),
        handles,
        working_pins(count, handles)?,
    ])
}

pub(in crate::runtime::streaming) fn admit_ack_clone(
    credit: &MemoryReservation,
    node_id: &str,
    state: &OperatorManifestEntry,
) -> Result<()> {
    let bytes = sum(&[
        entry_clone_bytes(state)?,
        assembly_bytes(node_id)?,
        digest_bytes(state)?,
    ])?;
    reserve(credit, bytes)
}

fn entry_clone_bytes(state: &OperatorManifestEntry) -> Result<usize> {
    sum(&[
        metadata_geometry(&state.inline_metadata)?.owned,
        progress_bytes(
            state.progress.keys().map(String::as_str),
            state.progress.len(),
        )?,
        handles_bytes(&state.segments)?,
    ])
}

fn assembly_bytes(node_id: &str) -> Result<usize> {
    // accept_operator_ack keeps the ACK alive while cloning into assembly.
    // Separate operator, capture-guard and working-pin maps clone node IDs.
    sum(&[
        tree_node::<String, OperatorManifestEntry>()?,
        tree_node::<String, Arc<MemoryReservation>>()?,
        tree_node::<String, Arc<crate::state::WorkingStatePins>>()?,
        product(4, node_id.len())?,
        // CheckpointAck::operator owns another participant ID and digest String.
        64,
    ])
}

fn progress_bytes<'a>(mut keys: impl Iterator<Item = &'a str>, count: usize) -> Result<usize> {
    keys.try_fold(
        tree::<String, OperatorIngressManifestEntry>(count)?,
        |bytes, key| add(bytes, key.len()),
    )
}

fn handles_bytes(handles: &[StateHandle]) -> Result<usize> {
    handles.iter().try_fold(
        product(handles.len(), size_of::<StateHandle>())?,
        |bytes, handle| {
            add(
                bytes,
                handle_strings(
                    handle.operator_id(),
                    handle.segment_id(),
                    handle.relative_path().len(),
                    handle.sha256().len(),
                )?,
            )
        },
    )
}

fn handle_strings(operator: &str, segment: &str, path: usize, digest: usize) -> Result<usize> {
    sum(&[operator.len(), segment.len(), path, digest])
}

fn working_pins(count: usize, handles: usize) -> Result<usize> {
    if count == 0 {
        return Ok(0);
    }
    // WorkingStatePins::acquire owns one Vec copy and one session.working map copy.
    sum(&[
        handles,
        handles
            .checked_sub(product(count, size_of::<StateHandle>())?)
            .ok_or_else(overflow)?,
        tree::<StateHandle, usize>(count)?,
        size_of::<crate::state::WorkingStatePins>() + 2 * size_of::<usize>(),
    ])
}

fn digest_bytes(state: &OperatorManifestEntry) -> Result<usize> {
    let geometry = manifest_geometry(state)?;
    // checkpoint_digest: to_value and canonical_json's sorted clone overlap.
    // sort_value also collects object entries through an intermediate Vec.
    sum(&[
        product(2, geometry.owned)?,
        geometry.collectors,
        bulk_peak::<u8>(geometry.wire, 128)?,
        bulk_peak::<(&Value, usize)>(geometry.nodes, 4)?,
        64,
    ])
}

#[derive(Clone, Copy)]
struct Geometry {
    owned: usize,
    wire: usize,
    nodes: usize,
    collectors: usize,
}

impl Geometry {
    fn combine(self, other: Self) -> Result<Self> {
        Ok(Self {
            owned: add(self.owned, other.owned)?,
            wire: add(self.wire, other.wire)?,
            nodes: add(self.nodes, other.nodes)?,
            collectors: add(self.collectors, other.collectors)?,
        })
    }
}

fn scalar() -> Geometry {
    // serde_json numbers use at most 24 bytes; 32 also covers null/bool.
    Geometry {
        owned: 0,
        wire: 32,
        nodes: 1,
        collectors: 0,
    }
}

fn text(text: &str) -> Result<Geometry> {
    Ok(Geometry {
        owned: text.len(),
        // JSON escaping uses at most six ASCII bytes per source UTF-8 byte.
        wire: add(product(text.len(), 6)?, 2)?,
        nodes: 1,
        collectors: 0,
    })
}

fn array(count: usize) -> Result<Geometry> {
    Ok(Geometry {
        owned: product(count, size_of::<Value>())?,
        wire: add(count, 2)?,
        nodes: 1,
        collectors: 0,
    })
}

fn object(count: usize) -> Result<Geometry> {
    Ok(Geometry {
        owned: tree::<String, Value>(count)?,
        wire: add(product(count, 2)?, 2)?,
        nodes: 1,
        collectors: bulk_peak::<(String, Value)>(count, 4)?,
    })
}

fn field(geometry: Geometry, key: &str, value: Geometry) -> Result<Geometry> {
    geometry.combine(text(key)?)?.combine(value)
}

fn metadata_geometry(metadata: &JsonMap) -> Result<Geometry> {
    metadata
        .iter()
        .try_fold(object(metadata.len())?, |geometry, (key, value)| {
            field(geometry, key, value_geometry(value)?)
        })
}

fn value_geometry(value: &Value) -> Result<Geometry> {
    match value {
        Value::String(value) => text(value),
        Value::Array(values) => values
            .iter()
            .try_fold(array(values.len())?, |geometry, value| {
                geometry.combine(value_geometry(value)?)
            }),
        Value::Object(values) => values
            .iter()
            .try_fold(object(values.len())?, |geometry, (key, value)| {
                field(geometry, key, value_geometry(value)?)
            }),
        _ => Ok(scalar()),
    }
}

fn manifest_geometry(state: &OperatorManifestEntry) -> Result<Geometry> {
    let progress = progress_geometry(state)?;
    let handles = state
        .segments
        .iter()
        .try_fold(array(state.segments.len())?, |geometry, handle| {
            geometry.combine(handle_geometry(handle)?)
        })?;
    fields(&[
        ("progress", progress),
        (
            "inline_metadata",
            metadata_geometry(&state.inline_metadata)?,
        ),
        ("segments", handles),
    ])
}

fn progress_geometry(state: &OperatorManifestEntry) -> Result<Geometry> {
    state
        .progress
        .keys()
        .try_fold(object(state.progress.len())?, |geometry, key| {
            let entry = fields(&[("state", text("active")?), ("watermark", scalar())])?;
            field(geometry, key, entry)
        })
}

fn handle_geometry(handle: &StateHandle) -> Result<Geometry> {
    fields(&[
        ("operator_id", text(handle.operator_id())?),
        ("epoch", scalar()),
        ("segment_id", text(handle.segment_id())?),
        ("relative_path", text(handle.relative_path())?),
        ("byte_len", scalar()),
        ("sha256", text(handle.sha256())?),
    ])
}

fn fields(values: &[(&str, Geometry)]) -> Result<Geometry> {
    values
        .iter()
        .try_fold(object(values.len())?, |geometry, (key, value)| {
            field(geometry, key, *value)
        })
}

fn tree<K, V>(count: usize) -> Result<usize> {
    product(add(count, 1)?, tree_node::<K, V>()?)
}

fn tree_node<K, V>() -> Result<usize> {
    let alignment = align_of::<K>()
        .max(align_of::<V>())
        .max(align_of::<usize>());
    // Pinned Rust BTree nodes: 11 slots, 12 edges, six field boundaries.
    let fields = size_of::<usize>() + 2 * size_of::<u16>() + 12 * size_of::<usize>();
    let slots = product(11, add(size_of::<K>(), size_of::<V>())?)?;
    let padded = sum(&[fields, slots, product(6, alignment - 1)?, alignment - 1])?;
    product(padded / alignment, alignment)
}

fn bulk_peak<T>(length: usize, initial: usize) -> Result<usize> {
    if length == 0 || size_of::<T>() == 0 {
        return Ok(0);
    }
    // Bulk reserve may choose a non-power-of-two capacity. At a growth point,
    // old capacity < new required length L, and new capacity <= 2L.
    product(product(length, 3)?.max(initial), size_of::<T>())
}

fn sum(values: &[usize]) -> Result<usize> {
    values.iter().try_fold(0, |total, value| add(total, *value))
}

fn add(left: usize, right: usize) -> Result<usize> {
    left.checked_add(right).ok_or_else(overflow)
}

fn product(left: usize, right: usize) -> Result<usize> {
    left.checked_mul(right).ok_or_else(overflow)
}

fn reserve(credit: &MemoryReservation, bytes: usize) -> Result<()> {
    credit.try_grow(bytes).map_err(|_| CalcFlowError::Internal {
        message: "Join checkpoint envelope credit admission failed".into(),
    })
}

fn overflow() -> CalcFlowError {
    CalcFlowError::Internal {
        message: "Join checkpoint envelope charge overflow".into(),
    }
}
