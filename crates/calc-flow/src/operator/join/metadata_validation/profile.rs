use super::super::StreamJoinSpec;
use crate::OperatorStateSnapshot;
use serde_json::Value;

const ROOT: &[&str] = &[
    "layout_version",
    "spec",
    "next_left_row_id",
    "next_right_row_id",
    "next_output_sequence",
    "metrics",
    "ended",
    "epoch",
];
const METRICS: &[&str] = &[
    "left",
    "right",
    "emitted_match_rows",
    "state_limit_failures",
    "match_limit_failures",
];
const SIDE: &[&str] = &[
    "retained_rows",
    "retained_bytes",
    "evicted_rows",
    "late_rows",
    "late_affected_batches",
    "max_lateness_micros",
    "null_event_time_rows",
    "null_key_rows",
];
const SPEC: &[&str] = &[
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
const BOUNDS: &[&str] = &["before_micros", "after_micros"];
const LIMITS: &[&str] = &[
    "max_state_rows_per_side",
    "max_state_bytes_per_side",
    "max_matches_per_input_batch",
];

pub(super) fn eligible(
    snapshot: &OperatorStateSnapshot,
    expected: &StreamJoinSpec,
    name: &str,
) -> Option<()> {
    bounded_expected(expected, name)?;
    let values = &snapshot.inline_metadata;
    scalar_metadata(values)?;
    expected_spec(&values["spec"], expected)?;
    metrics(&values["metrics"])?;
    values
        .iter()
        .all(|(key, value)| copy_unit(key, value).is_some())
        .then_some(())
}

fn bounded_expected(expected: &StreamJoinSpec, name: &str) -> Option<()> {
    (expected.left_keys.len() == 1 && expected.right_keys.len() == 1).then_some(())?;
    (name.len() <= 1024).then_some(())?;
    (expected_text_bytes(expected)? <= 1024).then_some(())
}

fn scalar_metadata(values: &crate::JsonMap) -> Option<()> {
    (values.len() == ROOT.len()).then_some(())?;
    ROOT.iter()
        .all(|key| values.contains_key(*key))
        .then_some(())?;
    (values["layout_version"].as_u64() == Some(1)).then_some(())?;
    for key in [
        "next_left_row_id",
        "next_right_row_id",
        "next_output_sequence",
        "epoch",
    ] {
        values[key].as_u64()?;
    }
    values["ended"].as_bool().map(|_| ())
}

pub(super) fn expected_text_bytes(spec: &StreamJoinSpec) -> Option<usize> {
    [
        spec.left_keys[0].as_str(),
        spec.right_keys[0].as_str(),
        spec.left_event_time.as_str(),
        spec.right_event_time.as_str(),
        spec.left_prefix.as_str(),
        spec.right_prefix.as_str(),
    ]
    .iter()
    .try_fold(0usize, |sum, text| sum.checked_add(text.len()))
}

fn object<'a>(value: &'a Value, allowed: &[&str]) -> Option<&'a serde_json::Map<String, Value>> {
    let object = value.as_object()?;
    object
        .keys()
        .all(|key| allowed.contains(&key.as_str()))
        .then_some(object)
}

fn expected_spec(value: &Value, expected: &StreamJoinSpec) -> Option<()> {
    let fields = object(value, SPEC)?;
    join_type(fields)?;
    spec_keys(fields, expected)?;
    spec_times(fields, expected)?;
    prefixes(fields, expected)?;
    spec_policy(fields, expected)
}

fn join_type(fields: &serde_json::Map<String, Value>) -> Option<()> {
    (fields.get("join_type")?.as_str()? == "inner").then_some(())
}

fn spec_keys(fields: &serde_json::Map<String, Value>, expected: &StreamJoinSpec) -> Option<()> {
    for (field, keys) in [
        ("left_keys", &expected.left_keys),
        ("right_keys", &expected.right_keys),
    ] {
        key_list(fields.get(field)?, keys)?;
    }
    Some(())
}

fn spec_times(fields: &serde_json::Map<String, Value>, expected: &StreamJoinSpec) -> Option<()> {
    for (field, name) in [
        ("left_event_time", &expected.left_event_time),
        ("right_event_time", &expected.right_event_time),
    ] {
        (fields.get(field)?.as_str()? == name).then_some(())?;
    }
    Some(())
}

fn spec_policy(fields: &serde_json::Map<String, Value>, expected: &StreamJoinSpec) -> Option<()> {
    scalar_pair(
        fields.get("bounds")?,
        BOUNDS,
        &[expected.bounds.before_micros, expected.bounds.after_micros],
    )?;
    scalar_pair(
        fields.get("limits")?,
        LIMITS,
        &[
            expected.limits.max_state_rows_per_side,
            expected.limits.max_state_bytes_per_side,
            expected.limits.max_matches_per_input_batch,
        ],
    )
}

fn key_list(value: &Value, keys: &[String]) -> Option<()> {
    let values = value.as_array()?;
    (values.len() == keys.len()).then_some(())?;
    values
        .iter()
        .zip(keys)
        .all(|(value, key)| value.as_str() == Some(key.as_str()))
        .then_some(())
}

fn prefixes(fields: &serde_json::Map<String, Value>, expected: &StreamJoinSpec) -> Option<()> {
    for (field, name, default) in [
        ("left_prefix", &expected.left_prefix, "left"),
        ("right_prefix", &expected.right_prefix, "right"),
    ] {
        let selected = fields.get(field).map_or(Some(default), Value::as_str)?;
        (selected == name).then_some(())?;
    }
    Some(())
}

fn scalar_pair(value: &Value, fields: &[&str], expected: &[u64]) -> Option<()> {
    let values = object(value, fields)?;
    (values.len() == fields.len()).then_some(())?;
    fields
        .iter()
        .zip(expected)
        .all(|(field, expected)| values[*field].as_u64() == Some(*expected))
        .then_some(())
}

fn metrics(value: &Value) -> Option<()> {
    let values = object(value, METRICS)?;
    (values.len() == METRICS.len()).then_some(())?;
    metric_counters(values)?;
    metric_sides(values)
}

fn metric_counters(values: &serde_json::Map<String, Value>) -> Option<()> {
    for key in [
        "emitted_match_rows",
        "state_limit_failures",
        "match_limit_failures",
    ] {
        values.get(key)?.as_u64()?;
    }
    Some(())
}

fn metric_sides(values: &serde_json::Map<String, Value>) -> Option<()> {
    for side in ["left", "right"] {
        side_metrics(values.get(side)?)?;
    }
    Some(())
}

fn side_metrics(value: &Value) -> Option<()> {
    let values = object(value, SIDE)?;
    for key in SIDE.iter().filter(|key| **key != "max_lateness_micros") {
        values.get(*key)?.as_u64()?;
    }
    if let Some(lateness) = values.get("max_lateness_micros") {
        if !lateness.is_null() {
            lateness.as_u64()?;
        }
    }
    Some(())
}

pub(super) fn copy_unit(key: &str, value: &Value) -> Option<()> {
    let (headers, bytes) = value_shape(value)?;
    let controls = headers
        .checked_add(1)?
        .checked_mul(size_of::<Value>().max(size_of::<String>()))?;
    (headers < 64).then_some(())?;
    (bytes.checked_add(key.len())?.checked_add(controls)? <= 4096).then_some(())
}

fn value_shape(value: &Value) -> Option<(usize, usize)> {
    match value {
        Value::String(text) => Some((1, text.len())),
        Value::Array(values) => array_shape(values),
        Value::Object(values) => object_shape(values),
        _ => Some((1, 0)),
    }
}

fn array_shape(values: &[Value]) -> Option<(usize, usize)> {
    values.iter().try_fold((1usize, 0usize), add_value_shape)
}

fn object_shape(values: &serde_json::Map<String, Value>) -> Option<(usize, usize)> {
    values
        .iter()
        .try_fold((1usize, 0usize), |shape, (key, value)| {
            add_field_shape(shape, key, value)
        })
}

fn add_value_shape((headers, bytes): (usize, usize), value: &Value) -> Option<(usize, usize)> {
    let (next, copied) = value_shape(value)?;
    Some((headers.checked_add(next)?, bytes.checked_add(copied)?))
}

fn add_field_shape(
    (headers, bytes): (usize, usize),
    key: &str,
    value: &Value,
) -> Option<(usize, usize)> {
    let (next, copied) = value_shape(value)?;
    Some((
        headers.checked_add(next)?.checked_add(1)?,
        bytes.checked_add(copied)?.checked_add(key.len())?,
    ))
}
