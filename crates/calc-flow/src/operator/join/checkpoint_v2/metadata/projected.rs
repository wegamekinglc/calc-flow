use serde::Deserialize;
use serde_json::Value;

use super::super::super::state_projection::checkpoint::{CHARGE_MODE, Descriptor};
use super::{
    InvalidFrame, JoinMetrics, JsonMap, Result, StreamJoinSpec, ValidatedMetadata, exact_keys,
    integer,
};

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
    charge_mode: String,
    retained_payloads: Descriptor,
}

const ROOT_KEYS: &[&str] = &[
    "layout_version",
    "spec",
    "next_left_row_id",
    "next_right_row_id",
    "next_output_sequence",
    "metrics",
    "ended",
    "epoch",
    "v2_inventory",
    "charge_mode",
    "retained_payloads",
];

pub(in crate::operator::join::checkpoint_v2) fn validate_root(metadata: &JsonMap) -> Result<()> {
    exact_keys(metadata.keys().map(String::as_str), ROOT_KEYS)?;
    super::validate(metadata)?;
    if integer(metadata.get("layout_version"))? != 3
        || metadata.get("charge_mode").and_then(Value::as_str) != Some(CHARGE_MODE)
    {
        return Err(InvalidFrame(
            "projected Join layout or charge mode is invalid",
        ));
    }
    Ok(())
}

pub(in crate::operator::join::checkpoint_v2) fn decode(
    metadata: JsonMap,
    spec: &StreamJoinSpec,
    expected: Option<&Descriptor>,
) -> Result<ValidatedMetadata> {
    let expected = expected.ok_or(InvalidFrame(
        "projected Join requires the matching retained layout",
    ))?;
    validate_root(&metadata)?;
    let metadata: Metadata = serde_json::from_value(Value::Object(metadata.into_iter().collect()))
        .map_err(|_| InvalidFrame("projected Join metadata is invalid"))?;
    validate_expected(&metadata, spec, expected)?;
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

fn validate_expected(
    metadata: &Metadata,
    spec: &StreamJoinSpec,
    expected: &Descriptor,
) -> Result<()> {
    if metadata.layout_version != 3
        || metadata.spec != *spec
        || metadata.epoch == 0
        || metadata.charge_mode != CHARGE_MODE
        || metadata.retained_payloads != *expected
    {
        return Err(InvalidFrame(
            "projected Join specification, epoch or retained descriptor differs",
        ));
    }
    Ok(())
}
