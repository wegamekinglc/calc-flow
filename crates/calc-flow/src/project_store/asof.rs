//! Field-level validation for the independent ASOF wire declaration.

use std::collections::BTreeSet;

use serde_json::{Map, Value};

use crate::ValidationIssue;

use super::raw_issue;

const SAFE_MAX: u64 = 9_007_199_254_740_991;

pub(super) fn collect_node_issues(operator: &Value, index: usize) -> Vec<ValidationIssue> {
    let mut issues = Vec::new();
    validate_operator(
        operator,
        &format!("graph.nodes[{index}].operator"),
        &mut issues,
    );
    issues
}

fn validate_operator(operator: &Value, base: &str, issues: &mut Vec<ValidationIssue>) {
    check_object(Some(operator), base, &["kind", "spec"], issues);
    let base = format!("{base}.spec");
    let Some(spec) = check_object(
        operator.get("spec"),
        &base,
        &["left", "right", "tolerance_micros", "limits", "late_policy"],
        issues,
    ) else {
        return;
    };
    for side in ["left", "right"] {
        validate_side(spec.get(side), &format!("{base}.{side}"), issues);
    }
    check_integer(
        spec.get("tolerance_micros"),
        &format!("{base}.tolerance_micros"),
        0,
        issues,
    );
    validate_limits(spec.get("limits"), &format!("{base}.limits"), issues);
    validate_policy(spec, &base, issues);
    validate_side_pairs(spec, &base, issues);
}

fn check_object<'a>(
    value: Option<&'a Value>,
    path: &str,
    fields: &[&str],
    issues: &mut Vec<ValidationIssue>,
) -> Option<&'a Map<String, Value>> {
    let value = require(value, path, issues)?;
    let Some(object) = value.as_object() else {
        issues.push(raw_issue(path.into(), "invalid_type", "must be an object"));
        return None;
    };
    for field in object.keys().filter(|key| !fields.contains(&key.as_str())) {
        issues.push(raw_issue(
            format!("{path}.{field}"),
            "unknown_field",
            "unknown ASOF field",
        ));
    }
    Some(object)
}

fn require<'a>(
    value: Option<&'a Value>,
    path: &str,
    issues: &mut Vec<ValidationIssue>,
) -> Option<&'a Value> {
    if value.is_none() {
        issues.push(raw_issue(
            path.into(),
            "missing_field",
            "required ASOF field is missing",
        ));
    }
    value
}

fn validate_side(value: Option<&Value>, path: &str, issues: &mut Vec<ValidationIssue>) {
    let Some(side) = check_object(
        value,
        path,
        &["keys", "event_time", "sequence_by", "prefix"],
        issues,
    ) else {
        return;
    };
    check_names(
        side.get("keys"),
        &format!("{path}.keys"),
        "invalid_asof_keys",
        issues,
    );
    check_names(
        side.get("sequence_by"),
        &format!("{path}.sequence_by"),
        "invalid_asof_sequence",
        issues,
    );
    check_name(
        side.get("event_time"),
        &format!("{path}.event_time"),
        issues,
    );
    if let Some(prefix) = check_name(side.get("prefix"), &format!("{path}.prefix"), issues)
        && !crate::operator::is_portable_identifier(prefix)
    {
        issues.push(raw_issue(
            format!("{path}.prefix"),
            "invalid_output_prefix",
            "must be a non-empty ASCII identifier",
        ));
    }
}

fn check_name<'a>(
    value: Option<&'a Value>,
    path: &str,
    issues: &mut Vec<ValidationIssue>,
) -> Option<&'a str> {
    let value = require(value, path, issues)?;
    if let Some(value) = value.as_str().filter(|value| !value.is_empty()) {
        Some(value)
    } else {
        issues.push(raw_issue(
            path.into(),
            "invalid_type",
            "must be a non-empty string",
        ));
        None
    }
}

fn check_names(value: Option<&Value>, path: &str, code: &str, issues: &mut Vec<ValidationIssue>) {
    let Some(value) = require(value, path, issues) else {
        return;
    };
    let Some(names) = value.as_array() else {
        issues.push(raw_issue(
            path.into(),
            "invalid_type",
            "must be an array of column names",
        ));
        return;
    };
    let mut unique = BTreeSet::new();
    let valid = !names.is_empty()
        && names.iter().all(|name| {
            name.as_str()
                .is_some_and(|name| !name.is_empty() && unique.insert(name))
        });
    if !valid {
        issues.push(raw_issue(
            path.into(),
            code,
            "must contain at least one unique non-empty column name",
        ));
    }
}

fn check_integer(
    value: Option<&Value>,
    path: &str,
    minimum: u64,
    issues: &mut Vec<ValidationIssue>,
) {
    let Some(value) = require(value, path, issues) else {
        return;
    };
    let code = if !(value.is_i64() || value.is_u64()) {
        "invalid_type"
    } else if value
        .as_u64()
        .is_some_and(|number| (minimum..=SAFE_MAX).contains(&number))
    {
        return;
    } else {
        "out_of_range"
    };
    let field = path.rsplit('.').next().unwrap_or(path);
    issues.push(raw_issue(
        path.into(),
        code,
        format!("{field} must be an integer in {minimum}..={SAFE_MAX}"),
    ));
}

fn validate_limits(value: Option<&Value>, path: &str, issues: &mut Vec<ValidationIssue>) {
    let fields = ["max_state_rows", "max_state_bytes"];
    if let Some(limits) = check_object(value, path, &fields, issues) {
        for field in fields {
            check_integer(limits.get(field), &format!("{path}.{field}"), 1, issues);
        }
    }
}

fn validate_policy(spec: &Map<String, Value>, base: &str, issues: &mut Vec<ValidationIssue>) {
    let path = format!("{base}.late_policy");
    let Some(policy) = require(spec.get("late_policy"), &path, issues) else {
        return;
    };
    if !matches!(policy.as_str(), Some("error" | "drop")) {
        issues.push(raw_issue(
            format!("{base}.late_policy"),
            "invalid_asof_late_policy",
            "late_policy must be error or drop",
        ));
    }
}

fn validate_side_pairs(spec: &Map<String, Value>, base: &str, issues: &mut Vec<ValidationIssue>) {
    let (Some(left), Some(right)) = (spec.get("left"), spec.get("right")) else {
        return;
    };
    if let (Some(left), Some(right)) = (
        left.get("keys").and_then(Value::as_array),
        right.get("keys").and_then(Value::as_array),
    ) && left.len() != right.len()
    {
        issues.push(raw_issue(
            format!("{base}.right.keys"),
            "invalid_asof_keys",
            "left and right keys must have equal length",
        ));
    }
    if let (Some(left), Some(right)) = (
        left.get("prefix").and_then(Value::as_str),
        right.get("prefix").and_then(Value::as_str),
    ) && left == right
    {
        issues.push(raw_issue(
            format!("{base}.right.prefix"),
            "invalid_output_prefix",
            "left and right prefixes must differ",
        ));
    }
}
