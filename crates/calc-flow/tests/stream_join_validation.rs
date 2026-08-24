use std::{sync::Arc, time::Duration};

use calc_flow::{
    ArrowFieldSpec, Batch, BatchKind, BatchMetadata, CalcFlowError, DataFusionConfig,
    JoinStateLimits, JoinTimeBounds, NodeSpec, OperatorSpec, PortSpec, ProjectSpec,
    ProviderRegistry, RuntimeSpec, StateConfig, StreamJoinOperator, StreamJoinSpec,
    StreamRunOptions, UdfRegistry, import_project_json, validate_project,
};
use datafusion::arrow::{
    array::{StringArray, TimestampMicrosecondArray},
    datatypes::{DataType, Field, Schema, TimeUnit},
    record_batch::RecordBatch,
};
use serde_json::{Value, json};

const SAFE_MAX: i64 = 9_007_199_254_740_991;

fn join_document(spec: &Value) -> Value {
    let mut node = json!({
        "id": "match",
        "input_ports": [
            {
                "name": "left",
                "kind": "table",
                "required": true,
                "schema": [
                    {"name": "account_id", "data_type": "int64", "nullable": false},
                    {"name": "authorized_at", "data_type": "timestamp[us]", "nullable": false}
                ]
            },
            {
                "name": "right",
                "kind": "table",
                "required": true,
                "schema": [
                    {"name": "account_id", "data_type": "int64", "nullable": false},
                    {"name": "paid_at", "data_type": "timestamp[us]", "nullable": false}
                ]
            }
        ],
        "operator": {"kind": "stream_join", "spec": spec}
    });
    node["output_ports"] = json!([]);
    json!({
        "format_version": 3,
        "id": "payments",
        "name": "Payments",
        "runtime": {"mode": "stream", "options": {}},
        "graph": {"name": "payments", "nodes": [node]}
    })
}

fn valid_spec() -> Value {
    json!({
        "join_type": "inner",
        "left_keys": ["account_id"],
        "right_keys": ["account_id"],
        "left_event_time": "authorized_at",
        "right_event_time": "paid_at",
        "bounds": {"before_micros": 300_000_000, "after_micros": 30_000_000},
        "limits": {
            "max_state_rows_per_side": 100_000,
            "max_state_bytes_per_side": 134_217_728,
            "max_matches_per_input_batch": 1_000_000
        },
        "left_prefix": "authorization",
        "right_prefix": "payment"
    })
}

fn import(spec: &Value) -> Result<ProjectSpec, CalcFlowError> {
    import_project_json(join_document(spec).to_string().as_bytes())
}

fn raw_issues(spec: &Value) -> Vec<(String, String, String)> {
    match import(spec) {
        Err(CalcFlowError::ProjectValidation { issues }) => issues
            .into_iter()
            .map(|issue| (issue.path, issue.code, issue.message))
            .collect(),
        Err(other) => panic!("expected ProjectValidation, got {other:?}"),
        Ok(project) => panic!("expected rejection, imported {project:?}"),
    }
}

#[test]
fn clean_raw_join_at_the_safe_integer_ceiling_round_trips() {
    let mut spec = valid_spec();
    spec["bounds"]["before_micros"] = json!(SAFE_MAX);
    spec["bounds"]["after_micros"] = json!(SAFE_MAX);
    spec["limits"]["max_state_rows_per_side"] = json!(SAFE_MAX);
    spec["limits"]["max_state_bytes_per_side"] = json!(SAFE_MAX);
    spec["limits"]["max_matches_per_input_batch"] = json!(SAFE_MAX);
    let project = import(&spec).expect("ceiling values import");
    let OperatorSpec::StreamJoin { spec } = &project.graph.nodes[0].operator else {
        panic!("stream_join operator parses");
    };
    assert_eq!(spec.bounds().before_micros(), SAFE_MAX as u64);
    assert_eq!(spec.limits().max_matches_per_input_batch(), SAFE_MAX as u64);
}

#[test]
fn missing_prefixes_default_only_after_a_clean_raw_pass() {
    let mut spec = valid_spec();
    spec.as_object_mut().unwrap().remove("left_prefix");
    spec.as_object_mut().unwrap().remove("right_prefix");
    let project = import(&spec).expect("clean raw join defaults prefixes");
    let OperatorSpec::StreamJoin { spec } = &project.graph.nodes[0].operator else {
        panic!("stream_join operator parses");
    };
    assert_eq!(spec.left_prefix(), "left");
    assert_eq!(spec.right_prefix(), "right");
}

#[test]
fn unsupported_join_type_scalar_reports_the_stable_issue() {
    for join_type in [Value::Null, json!("outer"), json!(5), json!(true)] {
        let mut spec = valid_spec();
        spec["join_type"] = join_type.clone();
        assert_eq!(
            raw_issues(&spec),
            vec![(
                "graph.nodes[0].operator.spec.join_type".into(),
                "unsupported_join_type".into(),
                "join_type must be the string inner".into(),
            )],
            "join_type = {join_type}"
        );
    }
    let mut spec = valid_spec();
    spec.as_object_mut().unwrap().remove("join_type");
    assert_eq!(
        raw_issues(&spec),
        vec![(
            "graph.nodes[0].operator.spec.join_type".into(),
            "unsupported_join_type".into(),
            "join_type must be the string inner".into(),
        )]
    );
}

#[test]
fn malformed_bounds_report_the_stable_field_issue() {
    for bad in [
        json!(-1),
        json!(1.5),
        json!("5"),
        json!(true),
        json!(null),
        json!(9_007_199_254_740_992i64),
    ] {
        let mut spec = valid_spec();
        spec["bounds"]["before_micros"] = bad.clone();
        assert_eq!(
            raw_issues(&spec),
            vec![(
                "graph.nodes[0].operator.spec.bounds.before_micros".into(),
                "invalid_time_bound".into(),
                "before_micros must be an integer microsecond count in 0..=9007199254740991".into(),
            )],
            "before_micros = {bad}"
        );
    }
}

#[test]
fn malformed_limits_report_the_stable_field_issue() {
    for bad in [
        json!(0),
        json!(-2),
        json!(2.5),
        json!("7"),
        json!(null),
        json!(9_007_199_254_740_992i64),
    ] {
        let mut spec = valid_spec();
        spec["limits"]["max_state_bytes_per_side"] = bad.clone();
        assert_eq!(
            raw_issues(&spec),
            vec![(
                "graph.nodes[0].operator.spec.limits.max_state_bytes_per_side".into(),
                "invalid_join_limit".into(),
                "max_state_bytes_per_side must be an integer in 1..=9007199254740991".into(),
            )],
            "max_state_bytes_per_side = {bad}"
        );
    }
}

#[test]
fn missing_bounds_and_limits_report_each_child_field_in_table_order() {
    let mut spec = valid_spec();
    spec.as_object_mut().unwrap().remove("bounds");
    spec.as_object_mut().unwrap().remove("limits");
    assert_eq!(
        raw_issues(&spec),
        vec![
            (
                "graph.nodes[0].operator.spec.bounds.before_micros".into(),
                "invalid_time_bound".into(),
                "before_micros must be an integer microsecond count in 0..=9007199254740991".into(),
            ),
            (
                "graph.nodes[0].operator.spec.bounds.after_micros".into(),
                "invalid_time_bound".into(),
                "after_micros must be an integer microsecond count in 0..=9007199254740991".into(),
            ),
            (
                "graph.nodes[0].operator.spec.limits.max_state_rows_per_side".into(),
                "invalid_join_limit".into(),
                "max_state_rows_per_side must be an integer in 1..=9007199254740991".into(),
            ),
            (
                "graph.nodes[0].operator.spec.limits.max_state_bytes_per_side".into(),
                "invalid_join_limit".into(),
                "max_state_bytes_per_side must be an integer in 1..=9007199254740991".into(),
            ),
            (
                "graph.nodes[0].operator.spec.limits.max_matches_per_input_batch".into(),
                "invalid_join_limit".into(),
                "max_matches_per_input_batch must be an integer in 1..=9007199254740991".into(),
            ),
        ]
    );
}

#[test]
fn prefix_issues_report_the_stable_codes_in_table_order() {
    let mut empty_left = valid_spec();
    empty_left["left_prefix"] = json!("");
    assert_eq!(
        raw_issues(&empty_left),
        vec![(
            "graph.nodes[0].operator.spec.left_prefix".into(),
            "invalid_output_prefix".into(),
            "left_prefix must be a non-empty portable identifier".into(),
        )]
    );

    let mut numeric_right = valid_spec();
    numeric_right["right_prefix"] = json!("9pay");
    assert_eq!(
        raw_issues(&numeric_right),
        vec![(
            "graph.nodes[0].operator.spec.right_prefix".into(),
            "invalid_output_prefix".into(),
            "right_prefix must be a non-empty portable identifier".into(),
        )]
    );

    let mut duplicate = valid_spec();
    duplicate["left_prefix"] = json!("same");
    duplicate["right_prefix"] = json!("same");
    assert_eq!(
        raw_issues(&duplicate),
        vec![(
            "graph.nodes[0].operator.spec.right_prefix".into(),
            "invalid_output_prefix".into(),
            "right_prefix must differ from left_prefix and produce no output-field collision"
                .into(),
        )]
    );
}

#[test]
fn issues_accumulate_in_node_and_field_order() {
    let mut spec = valid_spec();
    spec["join_type"] = json!("left");
    spec["bounds"]["before_micros"] = json!(-1);
    spec["limits"]["max_state_rows_per_side"] = json!(0);
    spec["left_prefix"] = json!("");
    assert_eq!(
        raw_issues(&spec)
            .into_iter()
            .map(|(path, code, _)| (path, code))
            .collect::<Vec<_>>(),
        vec![
            (
                "graph.nodes[0].operator.spec.join_type".into(),
                "unsupported_join_type".into(),
            ),
            (
                "graph.nodes[0].operator.spec.bounds.before_micros".into(),
                "invalid_time_bound".into(),
            ),
            (
                "graph.nodes[0].operator.spec.limits.max_state_rows_per_side".into(),
                "invalid_join_limit".into(),
            ),
            (
                "graph.nodes[0].operator.spec.left_prefix".into(),
                "invalid_output_prefix".into(),
            ),
        ]
    );
}

#[test]
fn unknown_fields_and_wrong_containers_remain_format_errors() {
    let mut unknown = valid_spec();
    unknown["unexpected_field"] = json!(1);
    match import(&unknown) {
        Err(CalcFlowError::Format { .. }) => {}
        Err(other) => panic!("unknown fields stay format errors, got {other:?}"),
        Ok(_) => panic!("unknown fields stay format errors"),
    }

    let mut object_scalar = valid_spec();
    object_scalar["bounds"]["before_micros"] = json!({});
    match import(&object_scalar) {
        Err(CalcFlowError::Format { .. }) => {}
        Err(other) => panic!("object scalar stays a format error, got {other:?}"),
        Ok(_) => panic!("object scalar stays a format error"),
    }

    let document = join_document(&Value::Null);
    let mut document = document;
    document["graph"]["nodes"][0]["operator"]["spec"] = json!([1, 2]);
    match import_project_json(document.to_string().as_bytes()) {
        Err(CalcFlowError::Format { .. }) => {}
        Err(other) => panic!("array spec stays a format error, got {other:?}"),
        Ok(_) => panic!("array spec stays a format error"),
    }
}

fn typed_join_project(
    spec: StreamJoinSpec,
    left_schema: Vec<ArrowFieldSpec>,
    right_schema: Vec<ArrowFieldSpec>,
) -> ProjectSpec {
    ProjectSpec {
        format_version: 3,
        id: "payments".into(),
        name: "Payments".into(),
        description: String::new(),
        runtime: RuntimeSpec::Stream(StreamRunOptions::default()),
        graph: calc_flow::PipelineSpec {
            name: "payments".into(),
            nodes: vec![NodeSpec {
                id: "match".into(),
                operator: OperatorSpec::StreamJoin { spec },
                input_ports: vec![
                    PortSpec {
                        name: "left".into(),
                        kind: BatchKind::Table,
                        required: true,
                        schema: left_schema,
                    },
                    PortSpec {
                        name: "right".into(),
                        kind: BatchKind::Table,
                        required: true,
                        schema: right_schema,
                    },
                ],
                output_ports: Vec::new(),
                position: None,
            }],
            edges: Vec::new(),
            datafusion: DataFusionConfig::default(),
        },
        data_sources: Vec::new(),
        sources: Vec::new(),
        sinks: Vec::new(),
        state: StateConfig::default(),
    }
}

fn typed_spec() -> StreamJoinSpec {
    StreamJoinSpec::inner(
        ["account_id"],
        ["account_id"],
        "authorized_at",
        "paid_at",
        JoinTimeBounds::new(
            Duration::from_micros(300_000_000),
            Duration::from_micros(30_000_000),
        )
        .unwrap(),
        JoinStateLimits::new(100_000, 134_217_728, 1_000_000).unwrap(),
    )
    .unwrap()
    .with_prefixes("authorization", "payment")
    .unwrap()
}

fn left_fields() -> Vec<ArrowFieldSpec> {
    vec![
        ArrowFieldSpec {
            name: "account_id".into(),
            data_type: "int64".into(),
            nullable: false,
        },
        ArrowFieldSpec {
            name: "authorized_at".into(),
            data_type: "timestamp[us]".into(),
            nullable: false,
        },
    ]
}

fn right_fields() -> Vec<ArrowFieldSpec> {
    vec![
        ArrowFieldSpec {
            name: "account_id".into(),
            data_type: "int64".into(),
            nullable: false,
        },
        ArrowFieldSpec {
            name: "paid_at".into(),
            data_type: "timestamp[us]".into(),
            nullable: false,
        },
    ]
}

fn semantic_issues(project: &ProjectSpec) -> Vec<(String, String, String)> {
    let (providers, udfs) = (ProviderRegistry::default(), UdfRegistry::new().snapshot());
    validate_project(project, &providers, &udfs)
        .issues
        .into_iter()
        .filter(|issue| issue.path.contains("operator.spec"))
        .map(|issue| (issue.path, issue.code, issue.message))
        .collect()
}

#[test]
fn typed_clean_join_project_validates_without_join_issues() {
    let issues = semantic_issues(&typed_join_project(
        typed_spec(),
        left_fields(),
        right_fields(),
    ));
    assert!(
        !issues
            .iter()
            .any(|(path, _, _)| path.contains("operator.spec")),
        "clean join reports no spec issues: {issues:?}"
    );
}

#[test]
fn raw_empty_or_duplicate_keys_report_the_stable_issues() {
    let mut empty = valid_spec();
    empty["left_keys"] = json!([]);
    empty["right_keys"] = json!([]);
    assert_eq!(
        raw_issues(&empty),
        vec![(
            "graph.nodes[0].operator.spec.left_keys".into(),
            "invalid_join_keys".into(),
            "left_keys must contain at least one unique column name".into(),
        )]
    );

    let mut duplicated = valid_spec();
    duplicated["left_keys"] = json!(["account_id", "account_id"]);
    duplicated["right_keys"] = json!(["account_id", "account_id"]);
    assert_eq!(
        raw_issues(&duplicated),
        vec![(
            "graph.nodes[0].operator.spec.left_keys".into(),
            "invalid_join_keys".into(),
            "left_keys must contain at least one unique column name".into(),
        )]
    );

    let mut mismatched = valid_spec();
    mismatched["left_keys"] = json!(["account_id", "region"]);
    mismatched["right_keys"] = json!(["account_id"]);
    assert_eq!(
        raw_issues(&mismatched),
        vec![(
            "graph.nodes[0].operator.spec.right_keys".into(),
            "invalid_join_keys".into(),
            "right_keys must contain exactly 2 unique column names to match left_keys".into(),
        )]
    );
}

#[test]
fn typed_missing_key_column_reports_the_stable_path_and_code() {
    let spec = StreamJoinSpec::inner(
        ["account_idx"],
        ["account_id"],
        "authorized_at",
        "paid_at",
        JoinTimeBounds::new(Duration::ZERO, Duration::ZERO).unwrap(),
        JoinStateLimits::new(1, 1, 1).unwrap(),
    )
    .unwrap()
    .with_prefixes("authorization", "payment")
    .unwrap();
    let issues = semantic_issues(&typed_join_project(spec, left_fields(), right_fields()));
    assert_eq!(
        issues,
        vec![(
            "graph.nodes[0].operator.spec.left_keys[0]".into(),
            "invalid_join_keys".into(),
            "left_keys[0] names missing or ambiguous column account_idx".into(),
        )],
        "{issues:?}"
    );
}

#[test]
fn typed_incompatible_key_types_report_the_stable_code() {
    let mut right = right_fields();
    right[0].data_type = "string".into();
    let spec = StreamJoinSpec::inner(
        ["account_id"],
        ["account_id"],
        "authorized_at",
        "paid_at",
        JoinTimeBounds::new(Duration::ZERO, Duration::ZERO).unwrap(),
        JoinStateLimits::new(1, 1, 1).unwrap(),
    )
    .unwrap()
    .with_prefixes("authorization", "payment")
    .unwrap();
    let issues = semantic_issues(&typed_join_project(spec, left_fields(), right));
    assert_eq!(
        issues,
        vec![(
            "graph.nodes[0].operator.spec.left_keys[0]".into(),
            "incompatible_key_type".into(),
            "join key pair 0 requires identical supported Arrow types; left is Int64 and right is Utf8"
                .into(),
        )],
        "{issues:?}"
    );
}

#[test]
fn typed_event_time_issues_report_the_stable_codes() {
    let missing = StreamJoinSpec::inner(
        ["account_id"],
        ["account_id"],
        "mauthorized_at",
        "paid_at",
        JoinTimeBounds::new(Duration::ZERO, Duration::ZERO).unwrap(),
        JoinStateLimits::new(1, 1, 1).unwrap(),
    )
    .unwrap()
    .with_prefixes("authorization", "payment")
    .unwrap();
    let issues = semantic_issues(&typed_join_project(missing, left_fields(), right_fields()));
    assert_eq!(
        issues,
        vec![(
            "graph.nodes[0].operator.spec.left_event_time".into(),
            "invalid_event_time".into(),
            "left_event_time names missing or ambiguous column mauthorized_at".into(),
        )],
        "{issues:?}"
    );

    let mut left = left_fields();
    left[1].data_type = "int64".into();
    let non_timestamp = StreamJoinSpec::inner(
        ["account_id"],
        ["account_id"],
        "authorized_at",
        "paid_at",
        JoinTimeBounds::new(Duration::ZERO, Duration::ZERO).unwrap(),
        JoinStateLimits::new(1, 1, 1).unwrap(),
    )
    .unwrap()
    .with_prefixes("authorization", "payment")
    .unwrap();
    let issues = semantic_issues(&typed_join_project(non_timestamp, left, right_fields()));
    assert_eq!(
        issues,
        vec![(
            "graph.nodes[0].operator.spec.left_event_time".into(),
            "invalid_event_time".into(),
            "left_event_time must be a timezone-naive or UTC Arrow timestamp; found Int64".into(),
        )],
        "{issues:?}"
    );
}

/// Feeds both Join ingresses and returns the emitted output messages.
async fn drive_join(
    left_keys: &[&str],
    right_keys: &[&str],
) -> calc_flow::Result<(usize, StreamJoinOperator)> {
    use calc_flow::{
        CancellationToken, EdgeCollector, JsonMap, OperatorMetadata, StreamJobContext,
        StreamOperator, StreamOperatorContext,
    };

    let schema = Arc::new(Schema::new(vec![
        Field::new("key", DataType::Utf8, false),
        Field::new(
            "ts",
            DataType::Timestamp(TimeUnit::Microsecond, None),
            false,
        ),
    ]));
    let mut operator = StreamJoinOperator::new(
        "match",
        Arc::clone(&schema),
        schema,
        StreamJoinSpec::inner(
            ["key"],
            ["key"],
            "ts",
            "ts",
            JoinTimeBounds::new(Duration::from_secs(300), Duration::from_secs(60)).unwrap(),
            JoinStateLimits::new(100_000, 134_217_728, 1_000_000).unwrap(),
        )
        .unwrap(),
    )
    .unwrap();
    let job = StreamJobContext::new(
        1,
        "fingerprint",
        JsonMap::new(),
        None,
        CancellationToken::new(),
    );
    let context = StreamOperatorContext::new(&job, "match", None);
    let mut collector = EdgeCollector::new(operator.output_ports().to_vec());
    let batch = |keys: &[&str]| {
        let record = RecordBatch::try_new(
            Arc::new(Schema::new(vec![
                Field::new("key", DataType::Utf8, false),
                Field::new(
                    "ts",
                    DataType::Timestamp(TimeUnit::Microsecond, None),
                    false,
                ),
            ])),
            vec![
                Arc::new(StringArray::from(keys.to_vec())),
                Arc::new(TimestampMicrosecondArray::from(vec![
                    1_000_000_i64;
                    keys.len()
                ])),
            ],
        )
        .unwrap();
        Batch::table(vec![record], BatchMetadata::default()).unwrap()
    };
    operator
        .process_data("left", batch(left_keys), &context, &mut collector)
        .await?;
    operator
        .process_data("right", batch(right_keys), &context, &mut collector)
        .await?;
    let outputs = collector.drain("output").len();
    Ok((outputs, operator))
}

#[tokio::test]
async fn zero_match_batches_do_not_fail_the_join_and_emit_nothing() {
    // AC2/AC7 steady-state path: a non-empty batch probing non-empty opposite
    // state with zero key-equality matches must succeed with zero output.
    let (outputs, operator) = drive_join(&["a"], &["b"]).await.unwrap();
    assert_eq!(outputs, 0);
    let status = operator.status();
    assert_eq!(status.left.retained_rows, 1);
    assert_eq!(status.right.retained_rows, 1);
    assert_eq!(status.emitted_match_rows, 0);
    assert_eq!(status.state_limit_failures, 0);
    assert_eq!(status.match_limit_failures, 0);
}

#[tokio::test]
async fn matching_keys_still_emit_after_the_zero_match_path_exists() {
    let (outputs, _operator) = drive_join(&["a"], &["a"]).await.unwrap();
    assert_eq!(outputs, 1);
}
