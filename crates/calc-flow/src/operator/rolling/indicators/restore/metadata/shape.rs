use crate::{CalcFlowError, JsonMap, Result};
use serde_json::{Map, Value};
use std::alloc::Layout;

type Check<'a> = &'a mut dyn FnMut() -> Result<()>;
#[derive(Clone, Copy, Default)]
pub(super) struct Facts {
    pub(super) strings: u64,
    pub(super) handle_strings: u64,
    pub(super) descriptor_strings: u64,
    pub(super) descriptors: usize,
    pub(super) segment_ids: u64,
    pub(super) path_components: usize,
}
pub(super) fn plan(fields: &JsonMap, check: Check<'_>) -> Result<Facts> {
    plan_entries(fields.iter(), check)
}
pub(super) fn plan_entries<'a>(
    fields: impl Iterator<Item = (&'a String, &'a Value)>,
    check: Check<'_>,
) -> Result<Facts> {
    check()?;
    let mut facts = Facts::default();
    for (key, value) in fields {
        check()?;
        root(key, value, &mut facts, check)?;
    }
    check()?;
    Ok(facts)
}
fn root(key: &str, value: &Value, facts: &mut Facts, check: Check<'_>) -> Result<()> {
    match key {
        "configuration_hash" | "state_schema_fingerprint" => text(value, facts),
        "kernel_fingerprint" | "numerical_profile" | "pipeline_fingerprint" | "operator_id" => {
            optional_text(value, facts)
        }
        "state_layout_version" => u32_value(value),
        "epoch" | "next_output_sequence" => u64_value(value),
        "last_input_watermark" => optional_i64(value),
        _ => root_container(key, value, facts, check),
    }
}
fn root_container(key: &str, value: &Value, facts: &mut Facts, check: Check<'_>) -> Result<()> {
    match key {
        "ended" => value.as_bool().map(|_| ()).ok_or_else(invalid),
        "metrics" => metrics(value, check),
        "segment_inventory" => inventory(value, facts, check),
        "late_output" => late(value, check),
        _ => Err(invalid()),
    }
}
fn text(value: &Value, facts: &mut Facts) -> Result<()> {
    let text = value.as_str().ok_or_else(invalid)?;
    Layout::array::<u8>(text.len()).map_err(|_| invalid())?;
    facts.strings = facts
        .strings
        .checked_add(u64::try_from(text.len()).map_err(|_| invalid())?)
        .ok_or_else(invalid)?;
    Ok(())
}
fn optional_text(value: &Value, facts: &mut Facts) -> Result<()> {
    if value.is_null() {
        Ok(())
    } else {
        text(value, facts)
    }
}
fn u64_value(value: &Value) -> Result<()> {
    value.as_u64().map(|_| ()).ok_or_else(invalid)
}
fn u32_value(value: &Value) -> Result<()> {
    let value = value.as_u64().ok_or_else(invalid)?;
    u32::try_from(value).map(|_| ()).map_err(|_| invalid())
}
fn optional_i64(value: &Value) -> Result<()> {
    if value.is_null() {
        Ok(())
    } else {
        value.as_i64().map(|_| ()).ok_or_else(invalid)
    }
}
fn optional_u64(value: &Value) -> Result<()> {
    if value.is_null() {
        Ok(())
    } else {
        u64_value(value)
    }
}
fn object(value: &Value) -> Result<&Map<String, Value>> {
    value.as_object().ok_or_else(invalid)
}
fn metrics(value: &Value, check: Check<'_>) -> Result<()> {
    for (key, value) in object(value)? {
        check()?;
        metric_field(key, value)?;
    }
    Ok(())
}
fn metric_field(key: &str, value: &Value) -> Result<()> {
    match key {
        "late_rows" | "affected_batches" | "null_event_time_rows" | "null_event_time_batches" => {
            u64_value(value)
        }
        "max_lateness_micros" => optional_u64(value),
        _ => Err(invalid()),
    }
}
fn late(value: &Value, check: Check<'_>) -> Result<()> {
    for (key, value) in object(value)? {
        check()?;
        late_field(key, value)?;
    }
    Ok(())
}
fn late_field(key: &str, value: &Value) -> Result<()> {
    match key {
        "version" | "schema_version" => u32_value(value),
        "next_sequence" => u64_value(value),
        _ => Err(invalid()),
    }
}

fn inventory(value: &Value, facts: &mut Facts, check: Check<'_>) -> Result<()> {
    let items = value.as_array().ok_or_else(invalid)?;
    if items.len() > 1 {
        return Err(invalid());
    }
    facts.descriptors = items.len();
    for item in items {
        descriptor(item, facts, check)?;
    }
    Ok(())
}
fn descriptor(value: &Value, facts: &mut Facts, check: Check<'_>) -> Result<()> {
    let before = facts.strings;
    for (key, value) in object(value)? {
        check()?;
        descriptor_field(key, value, facts, check)?;
    }
    facts.descriptor_strings = facts.strings.checked_sub(before).ok_or_else(invalid)?;
    Ok(())
}
fn descriptor_field(key: &str, value: &Value, facts: &mut Facts, check: Check<'_>) -> Result<()> {
    match key {
        "kind" => {
            if value.as_str() == Some("base") {
                Ok(())
            } else {
                Err(invalid())
            }
        }
        "state_layout_version" => u32_value(value),
        "schema_fingerprint" => text(value, facts),
        "handle" => handle(value, facts, check),
        _ => Err(invalid()),
    }
}
fn handle(value: &Value, facts: &mut Facts, check: Check<'_>) -> Result<()> {
    let before = facts.strings;
    for (key, value) in object(value)? {
        check()?;
        handle_field(key, value, facts, check)?;
    }
    facts.handle_strings = facts.strings.checked_sub(before).ok_or_else(invalid)?;
    Ok(())
}
fn handle_field(key: &str, value: &Value, facts: &mut Facts, check: Check<'_>) -> Result<()> {
    match key {
        "operator_id" | "sha256" => text(value, facts),
        "relative_path" => path(value, facts, check),
        "segment_id" => segment_id(value, facts),
        "epoch" | "byte_len" => u64_value(value),
        _ => Err(invalid()),
    }
}
fn path(value: &Value, facts: &mut Facts, check: Check<'_>) -> Result<()> {
    let path = value.as_str().ok_or_else(invalid)?;
    let mut count = 1_usize;
    for chunk in path.as_bytes().chunks(4096) {
        check()?;
        count = chunk.iter().try_fold(count, |count, byte| {
            count
                .checked_add(usize::from(*byte == b'/'))
                .ok_or_else(invalid)
        })?;
    }
    facts.path_components = count;
    text(value, facts)
}

fn segment_id(value: &Value, facts: &mut Facts) -> Result<()> {
    facts.segment_ids =
        u64::try_from(value.as_str().ok_or_else(invalid)?.len()).map_err(|_| invalid())?;
    text(value, facts)
}

fn invalid() -> CalcFlowError {
    CalcFlowError::CheckpointMismatch {
        message: "rolling state.metadata has an invalid field or type".into(),
    }
}
#[cfg(test)]
#[path = "shape_tests.rs"]
mod tests;
