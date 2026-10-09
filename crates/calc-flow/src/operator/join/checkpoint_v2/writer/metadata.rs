use datafusion::execution::memory_pool::MemoryReservation;
use serde::Serialize;
use serde_json::Value;

use super::super::super::{JoinMetrics, StreamJoinOperator, StreamJoinSpec, StreamJoinType};
use super::super::{
    budget,
    ipc::accounting::{add, product, sum, vector_peak},
};
use super::{
    buffer::{ensure, error},
    encode::PayloadEntry,
};
use crate::{Epoch, JsonMap, Result};

#[derive(Serialize)]
pub(super) struct DeltaEntry {
    pub(super) epoch: u64,
    pub(super) sides: Vec<&'static str>,
}

#[derive(Serialize)]
struct Metadata<'a> {
    layout_version: u32,
    spec: &'a StreamJoinSpec,
    next_left_row_id: u64,
    next_right_row_id: u64,
    next_output_sequence: u64,
    metrics: &'a JoinMetrics,
    ended: bool,
    epoch: u64,
    v2_inventory: Inventory<'a>,
}

#[derive(Serialize)]
struct Inventory<'a> {
    codec_version: u32,
    base_epoch: u64,
    deltas: &'a [DeltaEntry],
    payloads: &'a [PayloadEntry],
}

const METADATA_FIELDS: &[&str] = &[
    "layout_version",
    "spec",
    "next_left_row_id",
    "next_right_row_id",
    "next_output_sequence",
    "metrics",
    "ended",
    "epoch",
    "v2_inventory",
];
const SPEC_FIELDS: &[&str] = &[
    "join_type",
    "left_keys",
    "right_keys",
    "left_event_time",
    "right_event_time",
    "bounds",
    "limits",
    "left_prefix",
    "right_prefix",
];
const METRICS_FIELDS: &[&str] = &[
    "left",
    "right",
    "emitted_match_rows",
    "state_limit_failures",
    "match_limit_failures",
];
const SIDE_FIELDS: &[&str] = &[
    "retained_rows",
    "retained_bytes",
    "evicted_rows",
    "late_rows",
    "late_affected_batches",
    "max_lateness_micros",
    "null_event_time_rows",
    "null_key_rows",
];

pub(super) fn encode(
    operator: &StreamJoinOperator,
    epoch: Epoch,
    base_epoch: u64,
    deltas: &[DeltaEntry],
    payloads: &[PayloadEntry],
    credit: &MemoryReservation,
) -> Result<JsonMap> {
    let bytes = metadata_bytes(&operator.spec, deltas, payloads)?;
    ensure(credit, add(credit.size(), bytes)?)?;
    let metadata = Metadata {
        layout_version: 2,
        spec: &operator.spec,
        next_left_row_id: operator.state.next_left_row_id,
        next_right_row_id: operator.state.next_right_row_id,
        next_output_sequence: operator.state.next_output_sequence,
        metrics: &operator.state.metrics,
        ended: operator.state.ended,
        epoch: epoch.as_u64(),
        v2_inventory: Inventory {
            codec_version: 2,
            base_epoch,
            deltas,
            payloads,
        },
    };
    let Value::Object(metadata) = serde_json::to_value(metadata)
        .map_err(|_| error("V2 checkpoint metadata encoding failed"))?
    else {
        unreachable!("Join checkpoint metadata is an object")
    };
    Ok(metadata.into_iter().collect())
}

fn metadata_bytes(
    spec: &StreamJoinSpec,
    deltas: &[DeltaEntry],
    payloads: &[PayloadEntry],
) -> Result<usize> {
    sum(&[
        root_bytes()?,
        spec_bytes(spec)?,
        metrics_bytes()?,
        inventory_bytes(deltas, payloads)?,
    ])
}

fn root_bytes() -> Result<usize> {
    // The serde root, collector Vec and final JsonMap can overlap; values move.
    sum(&[
        object_bytes(METADATA_FIELDS)?,
        vector_peak::<(String, Value)>(METADATA_FIELDS.len())?,
        budget::tree::<String, Value>(METADATA_FIELDS.len())?,
    ])
}

fn metrics_bytes() -> Result<usize> {
    sum(&[
        object_bytes(METRICS_FIELDS)?,
        product(2, object_bytes(SIDE_FIELDS)?)?,
    ])
}

fn spec_bytes(spec: &StreamJoinSpec) -> Result<usize> {
    let join_type = match spec.join_type() {
        StreamJoinType::Inner => "inner",
    };
    sum(&[
        object_bytes(SPEC_FIELDS)?,
        join_type.len(),
        strings_bytes(spec.left_keys())?,
        strings_bytes(spec.right_keys())?,
        spec.left_event_time().len(),
        spec.right_event_time().len(),
        spec.left_prefix().len(),
        spec.right_prefix().len(),
        object_bytes(&["before_micros", "after_micros"])?,
        object_bytes(&[
            "max_state_rows_per_side",
            "max_state_bytes_per_side",
            "max_matches_per_input_batch",
        ])?,
    ])
}

fn strings_bytes(values: &[String]) -> Result<usize> {
    values
        .iter()
        .try_fold(array_bytes(values.len())?, |bytes, value| {
            add(bytes, value.len())
        })
}

fn inventory_bytes(deltas: &[DeltaEntry], payloads: &[PayloadEntry]) -> Result<usize> {
    let bytes = deltas.iter().try_fold(
        sum(&[
            object_bytes(&["codec_version", "base_epoch", "deltas", "payloads"])?,
            array_bytes(deltas.len())?,
            array_bytes(payloads.len())?,
        ])?,
        |bytes, delta| add(bytes, delta_bytes(delta)?),
    )?;
    payloads.iter().try_fold(bytes, |bytes, payload| {
        sum(&[
            bytes,
            object_bytes(&["side", "sha256", "rows", "bytes"])?,
            payload.side.len(),
            payload.sha256.len(),
        ])
    })
}

fn delta_bytes(delta: &DeltaEntry) -> Result<usize> {
    let sides = delta
        .sides
        .iter()
        .try_fold(array_bytes(delta.sides.len())?, |bytes, side| {
            add(bytes, side.len())
        })?;
    add(object_bytes(&["epoch", "sides"])?, sides)
}

fn array_bytes(length: usize) -> Result<usize> {
    // serde_json 1.0.150 SerializeVec uses with_capacity(Some(slice.len())).
    product(length, size_of::<Value>())
}

fn object_bytes(fields: &[&str]) -> Result<usize> {
    fields.iter().try_fold(
        budget::tree::<String, Value>(fields.len())?,
        |bytes, field| add(bytes, field.len()),
    )
}
