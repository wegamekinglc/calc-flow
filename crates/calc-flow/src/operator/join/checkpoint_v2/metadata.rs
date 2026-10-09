use serde::Deserialize;
use serde_json::{Map, Value};

use super::super::{JoinMetrics, StreamJoinSpec, metadata_validation::ValidatedMetadata};
use super::frame::InvalidFrame;
use super::inventory::{exact_keys, integer, object};
use crate::json::JsonMap;

type Result<T> = std::result::Result<T, InvalidFrame>;

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct Metadata {
    layout_version: u32,
    spec: StreamJoinSpec,
    next_left_row_id: u64,
    next_right_row_id: u64,
    next_output_sequence: u64,
    metrics: JoinMetrics,
    ended: bool,
    epoch: u64,
    v2_inventory: Value,
}

pub(super) fn decode(metadata: JsonMap, expected: &StreamJoinSpec) -> Result<ValidatedMetadata> {
    validate(&metadata)?;
    let metadata: Metadata = serde_json::from_value(Value::Object(metadata.into_iter().collect()))
        .map_err(|_| InvalidFrame("V2 metadata is invalid"))?;
    if metadata.layout_version != 2 || metadata.spec != *expected || metadata.epoch == 0 {
        return Err(InvalidFrame(
            "V2 checkpoint specification or epoch is incompatible",
        ));
    }
    drop(metadata.v2_inventory);
    Ok(ValidatedMetadata {
        next_left_row_id: metadata.next_left_row_id,
        next_right_row_id: metadata.next_right_row_id,
        next_output_sequence: metadata.next_output_sequence,
        metrics: metadata.metrics,
        ended: metadata.ended,
        epoch: metadata.epoch,
    })
}

pub(super) fn validate(metadata: &JsonMap) -> Result<()> {
    for key in [
        "next_left_row_id",
        "next_right_row_id",
        "next_output_sequence",
    ] {
        integer(metadata.get(key))?;
    }
    if metadata.get("ended").and_then(Value::as_bool).is_none() {
        return Err(InvalidFrame("ended must be a boolean"));
    }
    validate_spec(object(metadata.get("spec"), SPEC_KEYS)?)?;
    validate_metrics(object(metadata.get("metrics"), METRIC_KEYS)?)
}

fn validate_spec(spec: &Map<String, Value>) -> Result<()> {
    if spec.get("join_type").and_then(Value::as_str) != Some("inner") {
        return Err(InvalidFrame("join_type must be inner"));
    }
    for key in ["left_keys", "right_keys"] {
        validate_key_columns(spec.get(key))?;
    }
    validate_string_fields(spec)?;
    validate_integers(object(spec.get("bounds"), BOUNDS_KEYS)?, BOUNDS_KEYS)?;
    validate_integers(object(spec.get("limits"), LIMIT_KEYS)?, LIMIT_KEYS)
}

fn validate_string_fields(spec: &Map<String, Value>) -> Result<()> {
    for key in [
        "left_event_time",
        "right_event_time",
        "left_prefix",
        "right_prefix",
    ] {
        if spec.get(key).and_then(Value::as_str).is_none() {
            return Err(InvalidFrame(
                "event-time columns and prefixes must be strings",
            ));
        }
    }
    Ok(())
}

fn validate_key_columns(value: Option<&Value>) -> Result<()> {
    let keys = value
        .and_then(Value::as_array)
        .ok_or(InvalidFrame("Join keys must be arrays"))?;
    if keys.iter().any(|value| !value.is_string()) {
        return Err(InvalidFrame("Join key columns must be strings"));
    }
    Ok(())
}

fn validate_metrics(metrics: &Map<String, Value>) -> Result<()> {
    validate_side(object(metrics.get("left"), SIDE_METRIC_KEYS)?)?;
    validate_side(object(metrics.get("right"), SIDE_METRIC_KEYS)?)?;
    validate_integers(
        metrics,
        &[
            "emitted_match_rows",
            "state_limit_failures",
            "match_limit_failures",
        ],
    )
}

fn validate_side(side: &Map<String, Value>) -> Result<()> {
    exact_keys(side.keys().map(String::as_str), SIDE_METRIC_KEYS)?;
    for key in SIDE_METRIC_KEYS {
        if *key == "max_lateness_micros" && side.get(*key).is_some_and(Value::is_null) {
            continue;
        }
        integer(side.get(*key))?;
    }
    Ok(())
}

fn validate_integers(object: &Map<String, Value>, keys: &[&str]) -> Result<()> {
    for key in keys {
        integer(object.get(*key))?;
    }
    Ok(())
}

const SPEC_KEYS: &[&str] = &[
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
const BOUNDS_KEYS: &[&str] = &["before_micros", "after_micros"];
const LIMIT_KEYS: &[&str] = &[
    "max_state_rows_per_side",
    "max_state_bytes_per_side",
    "max_matches_per_input_batch",
];
const METRIC_KEYS: &[&str] = &[
    "left",
    "right",
    "emitted_match_rows",
    "state_limit_failures",
    "match_limit_failures",
];
const SIDE_METRIC_KEYS: &[&str] = &[
    "retained_rows",
    "retained_bytes",
    "evicted_rows",
    "late_rows",
    "late_affected_batches",
    "max_lateness_micros",
    "null_event_time_rows",
    "null_key_rows",
];
