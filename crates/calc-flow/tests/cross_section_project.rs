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
