use crate::{CalcFlowError, JsonMap, Result};
use serde_json::Value;

pub(super) fn validate_shape(metadata: &JsonMap) -> Result<()> {
    if metadata.iter().all(|(key, value)| match key.as_str() {
        "kind" | "row_encoding" | "fingerprint" => {
            value.as_str().is_some_and(|text| text.len() <= 64)
        }
        "state_version"
        | "layout_version"
        | "accounting_version"
        | "epoch"
        | "next_output_sequence" => value.as_u64().is_some(),
        "terminal" => value.is_boolean(),
        "metrics" => metrics(value),
        _ => false,
    }) {
        return Ok(());
    }
    Err(CalcFlowError::CheckpointMismatch {
        message: "ASOF snapshot metadata has an invalid fixed shape".into(),
    })
}
fn metrics(value: &Value) -> bool {
    value.as_object().is_some_and(|values| {
        values.iter().all(|(key, value)| match key.as_str() {
            "left" | "right" => side(value),
            "output_watermark_micros" => value.is_null(),
            "pending_left_rows"
            | "retained_right_rows"
            | "identity_only_rows"
            | "state_rows"
            | "state_bytes"
            | "emitted_left_rows"
            | "matched_rows"
            | "unmatched_rows"
            | "evicted_right_rows"
            | "state_limit_failures"
            | "workspace_limit_failures"
            | "output_limit_failures" => value.as_u64().is_some(),
            _ => false,
        })
    })
}
fn side(value: &Value) -> bool {
    value.as_object().is_some_and(|values| {
        values.iter().all(|(key, value)| match key.as_str() {
            "accepted_rows" | "late_rows" | "duplicate_rows" => value.as_u64().is_some(),
            "watermark_micros" => value.is_null(),
            "idle" | "ended" => value.is_boolean(),
            _ => false,
        })
    })
}
