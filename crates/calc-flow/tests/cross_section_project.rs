//! Project-v3 integration for the cross-section operator: strict document
//! acceptance, validation, batch execution, stream compilation, and
//! canonical fingerprints (SCE-00 D8, SCE-09).

use std::{collections::BTreeMap, sync::Arc};

use calc_flow::{
    ExecutionOptions, ProjectSpec, ProviderRegistry, StreamRequirements, UdfRegistry,
    compile_project, compile_stream_project_graph, validate_project,
};
use datafusion::arrow::{
    array::{ArrayRef, Float64Array, StringArray, TimestampMicrosecondArray, UInt64Array},
    datatypes::{DataType, Field, Schema, TimeUnit},
    record_batch::RecordBatch,
};
use serde_json::json;

fn cross_section_node_json() -> serde_json::Value {
    json!({
        "id": "cross_section_features",
        "operator": {
            "kind": "cross_section",
            "spec": {
                "configuration_version": 1,
                "state_layout_version": 1,
                "event_time": "ts",
                "entity_by": ["symbol"],
                "partition_by": ["industry"],
                "sequence_by": ["sequence"],
                "grouping": {"kind": "exact_time"},
                "outputs": [
                    {
                        "kind": "rank",
                        "primitive_version": 1,
                        "input": "momentum_20",
                        "output": "momentum_rank",
                        "direction": "ascending",
                        "tie_method": "average",
                        "null_placement": "exclude",
                        "min_samples": 1
                    },
                    {
                        "kind": "zscore",
                        "primitive_version": 1,
                        "input": "momentum_20",
                        "output": "alpha",
                        "min_samples": 1,
                        "ddof": 0
                    },
                    {
                        "kind": "winsorize",
                        "primitive_version": 1,
                        "input": "momentum_20",
                        "output": "momentum_winsorized",
                        "min_samples": 1,
                        "lower": 0.1,
                        "upper": 0.9
                    },
                    {
                        "kind": "top",
                        "primitive_version": 1,
                        "input": "momentum_20",
                        "output": "is_top",
                        "count": 2,
                        "include_ties": true,
                        "min_samples": 1
                    },
                    {
                        "kind": "bottom",
                        "primitive_version": 1,
                        "input": "momentum_20",
                        "output": "is_bottom",
                        "count": 2,
                        "include_ties": false,
                        "min_samples": 1
                    },
                    {
                        "kind": "mean_fill",
                        "primitive_version": 1,
                        "input": "momentum_20",
                        "output": "momentum_filled",
                        "min_samples": 1
                    }
                ],
                "allowed_lateness_micros": 0,
                "late_policy": {"kind": "error", "scope": "envelope"},
                "value_policy": "nan_exclude_preserve_v1"
            }
        },
        "input_ports": [
            {
                "name": "input",
                "kind": "table",
                "required": true,
                "schema": [
                    {"name": "ts", "data_type": "timestamp[us, UTC]", "nullable": false},
                    {"name": "symbol", "data_type": "string", "nullable": false},
                    {"name": "industry", "data_type": "string", "nullable": true},
                    {"name": "sequence", "data_type": "uint64", "nullable": false},
                    {"name": "momentum_20", "data_type": "float64", "nullable": true}
                ]
            }
        ]
    })
}

fn project_json(runtime: &serde_json::Value, node: &serde_json::Value) -> serde_json::Value {
    let mut project = json!({
        "format_version": 3,
        "id": "cross_section_project",
        "name": "cross_section_project",
        "runtime": runtime,
        "graph": {
            "name": "cross_section_project",
            "nodes": [node]
        }
    });
    if runtime["mode"] == "batch" {
        project["data_sources"] = json!([{
            "id": "fixture",
            "input": "input",
            "format": "json",
            "data": []
        }]);
    }
    project
}

fn registries() -> (ProviderRegistry, calc_flow::UdfRegistrySnapshot) {
    (ProviderRegistry::default(), UdfRegistry::new().snapshot())
}

fn input_batch() -> calc_flow::Batch {
    let schema = Arc::new(Schema::new(vec![
        Field::new(
            "ts",
            DataType::Timestamp(TimeUnit::Microsecond, Some(Arc::from("UTC"))),
            false,
        ),
        Field::new("symbol", DataType::Utf8, false),
        Field::new("industry", DataType::Utf8, true),
        Field::new("sequence", DataType::UInt64, false),
        Field::new("momentum_20", DataType::Float64, true),
    ]));
    let record = RecordBatch::try_new(
        schema,
        vec![
            Arc::new(TimestampMicrosecondArray::from(vec![100_i64, 100, 100]).with_timezone("UTC"))
                as ArrayRef,
            Arc::new(StringArray::from(vec!["a", "b", "c"])),
            Arc::new(StringArray::from(vec![
                Some("tech"),
                Some("tech"),
                Some("tech"),
            ])),
            Arc::new(UInt64Array::from(vec![1_u64, 2, 3])),
            Arc::new(Float64Array::from(vec![Some(2.0), Some(2.0), Some(1.0)])),
        ],
    )
    .unwrap();
    calc_flow::Batch::table(vec![record], calc_flow::BatchMetadata::default()).unwrap()
}

#[test]
fn cross_section_project_document_validates_and_executes_in_batch_mode() {
    let project: ProjectSpec = serde_json::from_value(project_json(
        &json!({"mode": "batch", "options": {}}),
        &cross_section_node_json(),
    ))
    .unwrap();
    let (providers, udfs) = registries();
    let report = validate_project(&project, &providers, &udfs);
    assert!(report.valid, "validation issues: {:?}", report.issues);
    assert!(report.fingerprint.is_some());

    let plan = compile_project(&project, &providers, &udfs).unwrap();
    let outputs = tokio::runtime::Builder::new_current_thread()
        .enable_all()
        .build()
        .unwrap()
        .block_on(plan.execute(
            BTreeMap::from([("input".into(), input_batch())]),
            ExecutionOptions::default(),
        ))
        .unwrap();
    let output = outputs.outputs["output"].clone();
    let record = output.table_payload().unwrap().batches()[0].clone();
    let ranks = record
        .column_by_name("momentum_rank")
        .unwrap()
        .as_any()
        .downcast_ref::<Float64Array>()
        .unwrap();
    assert_eq!(
        ranks.iter().collect::<Vec<_>>(),
        vec![Some(2.5), Some(2.5), Some(1.0)]
    );
}

#[test]
fn cross_section_project_document_compiles_in_stream_mode() {
    let project: ProjectSpec = serde_json::from_value(project_json(
        &json!({"mode": "stream", "options": {}}),
        &cross_section_node_json(),
    ))
    .unwrap();
    let (providers, udfs) = registries();
    let plan =
        compile_stream_project_graph(&project, &providers, &udfs, &StreamRequirements::default())
            .unwrap();
    assert_eq!(plan.topological_order(), vec!["cross_section_features"]);
}

#[test]
fn cross_section_project_canonicalizes_to_a_stable_fingerprint() {
    let (providers, udfs) = registries();
    let canonical: ProjectSpec = serde_json::from_value(project_json(
        &json!({"mode": "batch", "options": {}}),
        &cross_section_node_json(),
    ))
    .unwrap();
    let mut reordered_node = cross_section_node_json();
    reordered_node["position"] = json!({"x": 1.5, "y": 2.5});
    let reordered: ProjectSpec = serde_json::from_value(project_json(
        &json!({"mode": "batch", "options": {}}),
        &reordered_node,
    ))
    .unwrap();
    let first = validate_project(&canonical, &providers, &udfs);
    let second = validate_project(&reordered, &providers, &udfs);
    assert_eq!(first.fingerprint, second.fingerprint);
}

#[test]
fn bucketed_grouping_project_document_validates() {
    let mut node = cross_section_node_json();
    node["operator"]["spec"]["grouping"] =
        json!({"kind": "fixed_bucket", "width_micros": 60_000_000});
    let project: ProjectSpec = serde_json::from_value(project_json(
        &json!({"mode": "batch", "options": {}}),
        &node,
    ))
    .unwrap();
    let (providers, udfs) = registries();
    let report = validate_project(&project, &providers, &udfs);
    assert!(report.valid, "validation issues: {:?}", report.issues);
}

#[test]
fn cross_section_spec_round_trips_the_frozen_canonical_json() {
    let node = cross_section_node_json();
    let spec = &node["operator"]["spec"];
    let document: ProjectSpec = serde_json::from_value(project_json(
        &json!({"mode": "batch", "options": {}}),
        &node,
    ))
    .unwrap();
    let canonical = serde_json::to_value(&document).unwrap();
    let serialized_spec = &canonical["graph"]["nodes"][0]["operator"]["spec"];
    assert_eq!(serialized_spec, spec);
}

#[test]
fn non_applicable_ordering_fields_on_statistics_are_rejected_at_parse() {
    let mut node = cross_section_node_json();
    node["operator"]["spec"]["outputs"][1]["direction"] = json!("ascending");
    let error = serde_json::from_value::<ProjectSpec>(project_json(
        &json!({"mode": "batch", "options": {}}),
        &node,
    ));
    assert!(error.is_err(), "zscore accepted a direction field");
}

#[test]
fn cross_section_project_rejects_a_missing_exact_schema_at_validation() {
    let mut node = cross_section_node_json();
    node["input_ports"][0]["schema"] = json!([]);
    let project: ProjectSpec = serde_json::from_value(project_json(
        &json!({"mode": "batch", "options": {}}),
        &node,
    ))
    .unwrap();
    let (providers, udfs) = registries();
    let report = validate_project(&project, &providers, &udfs);
    assert!(!report.valid);
    assert!(
        report
            .issues
            .iter()
            .any(|issue| { issue.code == "invalid_ports" && issue.path.contains("input_ports") }),
        "expected an invalid_ports issue: {:?}",
        report.issues
    );
}

#[test]
fn cross_section_project_reports_operator_construction_failure_as_a_validation_issue() {
    let mut node = cross_section_node_json();
    node["operator"]["spec"]["entity_by"] = json!([]);
    let project: ProjectSpec = serde_json::from_value(project_json(
        &json!({"mode": "batch", "options": {}}),
        &node,
    ))
    .unwrap();
    let (providers, udfs) = registries();
    let report = validate_project(&project, &providers, &udfs);
    assert!(!report.valid);
    assert!(
        report
            .issues
            .iter()
            .any(|issue| issue.code == "invalid_operator" && issue.path.contains("operator")),
        "expected an invalid_operator issue: {:?}",
        report.issues
    );
}

fn side_output_node_json() -> serde_json::Value {
    let mut node = cross_section_node_json();
    node["operator"]["spec"]["late_policy"] = json!({
        "kind": "side_output", "metrics_version": 1, "schema_version": 1
    });
    node
}

#[test]
fn test_side_output_project_derives_two_external_slots() {
    let project: ProjectSpec = serde_json::from_value(project_json(
        &json!({"mode": "stream", "options": {}}),
        &side_output_node_json(),
    ))
    .unwrap();
    let (providers, udfs) = registries();
    let plan =
        compile_stream_project_graph(&project, &providers, &udfs, &StreamRequirements::default())
            .unwrap();
    assert_eq!(plan.sink_binding_ids(), ["late", "output"]);
}

fn explicit_side_output_node_json() -> serde_json::Value {
    let mut node = side_output_node_json();
    let input = node["input_ports"][0]["schema"].as_array().unwrap().clone();
    let mut normal = input.clone();
    for output in node["operator"]["spec"]["outputs"].as_array().unwrap() {
        let data_type = match output["kind"].as_str().unwrap() {
            "delta" => "int64",
            "top" | "bottom" => "bool",
            _ => "float64",
        };
        normal.push(json!({"name": output["output"], "data_type": data_type, "nullable": true}));
    }
    let mut late = input;
    for (name, data_type) in [
        ("_cf_late_node", "string"),
        ("_cf_late_input_port", "string"),
        ("_cf_late_event_time_micros", "int64"),
        ("_cf_late_closing_time_micros", "int64"),
        ("_cf_late_watermark_micros", "int64"),
        ("_cf_late_reason", "string"),
        ("_cf_late_source", "string"),
        ("_cf_late_sequence", "uint64"),
        ("_cf_late_row_index", "uint64"),
    ] {
        late.push(json!({"name": name, "data_type": data_type, "nullable": false}));
    }
    node["output_ports"] = json!([
        {"name": "output", "kind": "table", "required": true, "schema": normal},
        {"name": "late", "kind": "table", "required": true, "schema": late},
    ]);
    node
}

fn compile_side_node(
    node: &serde_json::Value,
) -> calc_flow::Result<calc_flow::StreamExecutionPlan> {
    let project: ProjectSpec = serde_json::from_value(project_json(
        &json!({"mode": "stream", "options": {}}),
        node,
    ))
    .unwrap();
    let (providers, udfs) = registries();
    compile_stream_project_graph(&project, &providers, &udfs, &StreamRequirements::default())
}

#[test]
fn test_side_output_project_checks_explicit_derived_ports() {
    let node = explicit_side_output_node_json();
    let plan = compile_side_node(&node).unwrap();
    assert_eq!(plan.sink_binding_ids(), ["late", "output"]);
    for mutation in [
        "missing", "extra", "name", "schema", "required", "kind", "disabled", "order",
    ] {
        let mut invalid = node.clone();
        match mutation {
            "missing" => {
                invalid["output_ports"].as_array_mut().unwrap().pop();
            }
            "extra" => {
                let extra = json!({"name": "fake", "kind": "table", "required": true});
                invalid["output_ports"].as_array_mut().unwrap().push(extra);
            }
            "name" => invalid["output_ports"][1]["name"] = json!("fake"),
            "schema" => invalid["output_ports"][1]["schema"][0]["nullable"] = json!(true),
            "required" => invalid["output_ports"][1]["required"] = json!(false),
            "kind" => invalid["output_ports"][1]["kind"] = json!("array"),
            "disabled" => {
                invalid["operator"]["spec"]["late_policy"] =
                    json!({"kind": "drop", "metrics_version": 1});
            }
            "order" => invalid["output_ports"].as_array_mut().unwrap().swap(0, 1),
            _ => unreachable!(),
        }
        let error = compile_side_node(&invalid).unwrap_err().to_string();
        assert!(
            error.contains("graph.nodes[0].output_ports"),
            "{mutation}: {error}"
        );
        assert!(
            error.contains("schema_mismatch")
                || error.contains("invalid_ports")
                || (mutation == "kind" && error.contains("array_schema")),
            "{mutation}: {error}"
        );
    }
}

#[test]
fn test_side_output_project_reports_version_and_reserved_field_paths() {
    for field in ["metrics_version", "schema_version"] {
        let mut node = side_output_node_json();
        node["operator"]["spec"]["late_policy"][field] = json!(2);
        let error = compile_side_node(&node).unwrap_err().to_string();
        assert!(
            error.contains(&format!("graph.nodes[0].operator.spec.late_policy.{field}")),
            "{error}"
        );
        assert!(error.contains("unsupported_version"), "{error}");
    }
    let mut node = side_output_node_json();
    node["input_ports"][0]["schema"]
        .as_array_mut()
        .unwrap()
        .push(json!({
            "name": "_cf_late_future", "data_type": "string", "nullable": true
        }));
    let error = compile_side_node(&node).unwrap_err().to_string();
    assert!(
        error.contains("graph.nodes[0].input_ports[0].schema[5].name"),
        "{error}"
    );
    assert!(
        error.contains("reserved_field") && error.contains("_cf_late_future"),
        "{error}"
    );
}

#[test]
fn test_side_output_project_rejects_batch_mode() {
    let project: ProjectSpec = serde_json::from_value(project_json(
        &json!({"mode": "batch", "options": {}}),
        &side_output_node_json(),
    ))
    .unwrap();
    let (providers, udfs) = registries();
    let error = compile_project(&project, &providers, &udfs)
        .unwrap_err()
        .to_string();
    assert!(
        error.contains("graph.nodes[0].operator.spec.late_policy"),
        "{error}"
    );
    assert!(error.contains("unsupported_mode"), "{error}");
}

#[test]
fn test_legacy_late_policy_preserves_canonical_json_fingerprint_and_bindings() {
    use sha2::{Digest, Sha256};
    let (providers, udfs) = registries();
    for (policy, fingerprint, canonical_hash) in [
        (
            json!({"kind": "error", "scope": "envelope"}),
            "0e782975434f7863aefda44c8525d10f8c90dbad64630ae09fca3b272a4ac401",
            "048c66f1a8fc649cb6d308962057d52de9b816f6ca2027efbf92bcdadcfc779f",
        ),
        (
            json!({"kind": "drop", "metrics_version": 1}),
            "a3f89d5c8e624eb71f7c7805faccfd3a4270777083f80254f0bbfdc7ac4feaa6",
            "1cd09e62bf0bb697ce82713c973d0a8cf1571d43e03790c7801ca74462004508",
        ),
    ] {
        let mut node = cross_section_node_json();
        node["operator"]["spec"]["late_policy"] = policy;
        let project: ProjectSpec = serde_json::from_value(project_json(
            &json!({"mode": "stream", "options": {}}),
            &node,
        ))
        .unwrap();
        let canonical =
            calc_flow::canonical_json(&serde_json::to_value(&project).unwrap()).unwrap();
        assert_eq!(
            hex::encode(Sha256::digest(canonical.as_bytes())),
            canonical_hash
        );
        let plan = compile_stream_project_graph(
            &project,
            &providers,
            &udfs,
            &StreamRequirements::default(),
        )
        .unwrap();
        assert_eq!(plan.fingerprint(), fingerprint);
        assert_eq!(plan.source_binding_ids(), ["input"]);
        assert_eq!(plan.sink_binding_ids(), ["output"]);
    }
}
