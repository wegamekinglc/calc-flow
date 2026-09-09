use calc_flow::{CalcFlowError, ProjectSpec, export_project_json, import_project_json};
use serde_json::{Value, json};

fn spec() -> Value {
    json!({
        "left": {
            "keys": ["symbol"], "event_time": "ts",
            "sequence_by": ["seq"], "prefix": "trade"
        },
        "right": {
            "keys": ["symbol"], "event_time": "ts",
            "sequence_by": ["seq"], "prefix": "quote"
        },
        "tolerance_micros": 10,
        "late_policy": "error",
        "limits": {"max_state_rows": 100_000, "max_state_bytes": 67_108_864}
    })
}

fn document(spec: &Value) -> Value {
    let schema = json!([
        {"name": "symbol", "data_type": "string", "nullable": false},
        {"name": "ts", "data_type": "timestamp[us, UTC]", "nullable": false},
        {"name": "seq", "data_type": "int64", "nullable": false}
    ]);
    json!({
        "format_version": 3, "id": "asof", "name": "ASOF",
        "runtime": {"mode": "stream", "options": {}},
        "graph": {"name": "asof", "nodes": [{
            "id": "match",
            "input_ports": [
                {"name": "left", "kind": "table", "required": true, "schema": schema},
                {"name": "right", "kind": "table", "required": true, "schema": schema}
            ],
            "output_ports": [],
            "operator": {"kind": "stream_asof_join", "spec": spec}
        }]}
    })
}

fn import(spec: &Value) -> Result<ProjectSpec, CalcFlowError> {
    import_project_json(document(spec).to_string().as_bytes())
}

fn issues(spec: &Value) -> Vec<(String, String)> {
    match import(spec) {
        Err(CalcFlowError::ProjectValidation { issues }) => issues
            .into_iter()
            .map(|issue| (issue.path, issue.code))
            .collect(),
        result => panic!("expected structured validation issues, got {result:?}"),
    }
}

#[test]
fn asof_project_round_trip_preserves_the_strict_independent_spec() {
    let original = spec();
    let project = import(&original).expect("ASOF project imports");
    let exported: Value = serde_json::from_str(&export_project_json(&project).unwrap()).unwrap();
    assert_eq!(exported["graph"]["nodes"][0]["operator"]["spec"], original);
    assert_eq!(exported["format_version"], 3);
}

#[test]
fn malformed_tolerance_preserves_its_path_and_type_or_range_reason() {
    for (value, code) in [
        (json!(true), "invalid_type"),
        (json!(1.5), "invalid_type"),
        (json!("10"), "invalid_type"),
        (json!(-1), "out_of_range"),
        (json!(9_007_199_254_740_992_u64), "out_of_range"),
    ] {
        let mut input = spec();
        input["tolerance_micros"] = value;
        assert_eq!(
            issues(&input),
            [(
                "graph.nodes[0].operator.spec.tolerance_micros".into(),
                code.into()
            )]
        );
    }
}

#[test]
fn limits_are_required_positive_safe_integers() {
    for field in ["max_state_rows", "max_state_bytes"] {
        let mut input = spec();
        input["limits"][field] = json!(0);
        assert_eq!(
            issues(&input),
            [(
                format!("graph.nodes[0].operator.spec.limits.{field}"),
                "out_of_range".into()
            )]
        );
    }
}

#[test]
fn unknown_side_fields_are_rejected_at_the_nested_path() {
    let mut input = spec();
    input["right"]["direction"] = json!("nearest");
    assert_eq!(
        issues(&input),
        [(
            "graph.nodes[0].operator.spec.right.direction".into(),
            "unknown_field".into()
        )]
    );
}

#[test]
fn every_asof_object_is_strict_and_identity_fields_are_required() {
    for path in ["", "/left", "/right", "/limits"] {
        let mut input = spec();
        input.pointer_mut(path).unwrap()["unknown"] = json!(true);
        let location = path.replace('/', ".");
        assert_eq!(
            issues(&input),
            [(
                format!("graph.nodes[0].operator.spec{location}.unknown"),
                "unknown_field".into()
            )]
        );
    }
    for side in ["left", "right"] {
        for field in ["keys", "event_time", "sequence_by", "prefix"] {
            let mut input = spec();
            input[side].as_object_mut().unwrap().remove(field);
            assert_eq!(
                issues(&input),
                [(
                    format!("graph.nodes[0].operator.spec.{side}.{field}"),
                    "missing_field".into()
                )]
            );
        }
    }
}

#[test]
fn raw_late_policy_is_required_even_though_constructors_default_to_error() {
    let mut input = spec();
    input.as_object_mut().unwrap().remove("late_policy");
    assert_eq!(
        issues(&input),
        [(
            "graph.nodes[0].operator.spec.late_policy".into(),
            "missing_field".into()
        )]
    );
}

#[test]
fn direct_serde_also_requires_the_materialized_late_policy() {
    let mut input = spec();
    input.as_object_mut().unwrap().remove("late_policy");
    let result = serde_json::from_value::<calc_flow::StreamAsofJoinSpec>(input);
    assert!(
        result.is_err(),
        "wire decoding must not invent a late policy"
    );
}
