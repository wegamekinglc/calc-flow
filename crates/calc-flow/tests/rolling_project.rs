use std::{collections::BTreeMap, sync::Arc};

use calc_flow::{
    ExecutionOptions, ProjectSpec, ProviderRegistry, StreamRequirements, UdfRegistry,
    compile_project, compile_stream_project_graph, validate_project,
};
use datafusion::arrow::{
    array::{
        ArrayRef, Float64Array, Int64Array, StringArray, TimestampMicrosecondArray, UInt64Array,
    },
    datatypes::{DataType, Field, Schema, TimeUnit},
    record_batch::RecordBatch,
};
use serde_json::json;

fn rolling_node_json() -> serde_json::Value {
    json!({
        "id": "rolling_features",
        "operator": {
            "kind": "rolling",
            "spec": {
                "configuration_version": 1,
                "state_layout_version": 1,
                "partition_by": ["symbol"],
                "event_time": "ts",
                "sequence_by": ["sequence"],
                "outputs": [
                    {
                        "kind": "lag",
                        "primitive_version": 1,
                        "input": "price",
                        "output": "price_lag_1",
                        "periods": 1
                    },
                    {
                        "kind": "delta",
                        "primitive_version": 1,
                        "input": "volume",
                        "output": "volume_delta_1",
                        "periods": 1
                    }
                ],
                "allowed_lateness_micros": 0,
                "late_policy": {"kind": "error", "scope": "envelope"},
                "value_policy": "stateful_numeric_v1"
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
                    {"name": "sequence", "data_type": "uint64", "nullable": false},
                    {"name": "price", "data_type": "float64", "nullable": true},
                    {"name": "volume", "data_type": "int64", "nullable": true}
                ]
            }
        ]
    })
}

fn project_json(runtime: &serde_json::Value, node: &serde_json::Value) -> serde_json::Value {
    let mut project = json!({
        "format_version": 3,
        "id": "rolling_project",
        "name": "rolling_project",
        "runtime": runtime,
        "graph": {
            "name": "rolling_project",
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
    let providers = ProviderRegistry::default();
    let udfs = UdfRegistry::new().snapshot();
    (providers, udfs)
}

fn input_batch() -> calc_flow::Batch {
    let schema = Arc::new(Schema::new(vec![
        Field::new(
            "ts",
            DataType::Timestamp(TimeUnit::Microsecond, Some(Arc::from("UTC"))),
            false,
        ),
        Field::new("symbol", DataType::Utf8, false),
        Field::new("sequence", DataType::UInt64, false),
        Field::new("price", DataType::Float64, true),
        Field::new("volume", DataType::Int64, true),
    ]));
    let record = RecordBatch::try_new(
        schema,
        vec![
            Arc::new(TimestampMicrosecondArray::from(vec![20_i64, 10, 11]).with_timezone("UTC"))
                as ArrayRef,
            Arc::new(StringArray::from(vec!["a", "a", "a"])),
            Arc::new(UInt64Array::from(vec![2_u64, 1, 3])),
            Arc::new(Float64Array::from(vec![Some(2.0), Some(1.0), Some(3.0)])),
            Arc::new(Int64Array::from(vec![Some(20), Some(10), Some(30)])),
        ],
    )
    .unwrap();
    calc_flow::Batch::table(vec![record], calc_flow::BatchMetadata::default()).unwrap()
}

#[test]
fn rolling_project_document_validates_and_compiles_in_batch_mode() {
    let project: ProjectSpec = serde_json::from_value(project_json(
        &json!({"mode": "batch", "options": {}}),
        &rolling_node_json(),
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
    let lags = record
        .column_by_name("price_lag_1")
        .unwrap()
        .as_any()
        .downcast_ref::<Float64Array>()
        .unwrap();
    assert_eq!(
        lags.iter().collect::<Vec<_>>(),
        vec![None, Some(1.0), Some(3.0)]
    );
}

#[test]
fn rolling_project_document_compiles_in_stream_mode() {
    let project: ProjectSpec = serde_json::from_value(project_json(
        &json!({"mode": "stream", "options": {}}),
        &rolling_node_json(),
    ))
    .unwrap();
    let (providers, udfs) = registries();
    let plan =
        compile_stream_project_graph(&project, &providers, &udfs, &StreamRequirements::default())
            .unwrap();
    assert_eq!(plan.topological_order(), vec!["rolling_features"]);
}

#[test]
fn rolling_project_canonicalizes_to_a_stable_fingerprint() {
    let (providers, udfs) = registries();
    let canonical: ProjectSpec = serde_json::from_value(project_json(
        &json!({"mode": "batch", "options": {}}),
        &rolling_node_json(),
    ))
    .unwrap();
    let reordered_text = serde_json::to_string(&project_json(
        &json!({"mode": "batch", "options": {}}),
        &rolling_node_json(),
    ))
    .unwrap();
    let mut reordered: serde_json::Map<String, serde_json::Value> =
        serde_json::from_str(&reordered_text).unwrap();
    reordered.sort_keys();
    let reversed: ProjectSpec = serde_json::from_value(serde_json::Value::Object(
        reordered.into_iter().rev().collect(),
    ))
    .unwrap();
    let first = validate_project(&canonical, &providers, &udfs);
    let second = validate_project(&reversed, &providers, &udfs);
    assert!(first.valid && second.valid);
    assert_eq!(first.fingerprint, second.fingerprint);
}

#[test]
fn rolling_project_rejects_unknown_and_missing_spec_fields() {
    let mut node = rolling_node_json();
    node["operator"]["spec"]["unexpected"] = json!(true);
    assert!(
        serde_json::from_value::<ProjectSpec>(project_json(
            &json!({"mode": "batch", "options": {}}),
            &node
        ))
        .is_err()
    );

    let mut node = rolling_node_json();
    node["operator"]["spec"]
        .as_object_mut()
        .unwrap()
        .remove("late_policy");
    assert!(
        serde_json::from_value::<ProjectSpec>(project_json(
            &json!({"mode": "batch", "options": {}}),
            &node
        ))
        .is_err()
    );
}

fn aggregate_outputs_json() -> serde_json::Value {
    json!([
        {
            "kind": "count",
            "primitive_version": 1,
            "input": "price",
            "output": "price_count_2",
            "frame": {"kind": "rows", "size": 2},
            "min_periods": 1
        },
        {
            "kind": "sum",
            "primitive_version": 1,
            "input": "price",
            "output": "price_sum_2",
            "frame": {"kind": "rows", "size": 2},
            "min_periods": 1
        },
        {
            "kind": "mean",
            "primitive_version": 1,
            "input": "price",
            "output": "price_mean_2",
            "frame": {"kind": "rows", "size": 2},
            "min_periods": 1
        },
        {
            "kind": "variance",
            "primitive_version": 1,
            "input": "price",
            "output": "price_var_2",
            "frame": {"kind": "rows", "size": 2},
            "min_periods": 1,
            "ddof": 1
        },
        {
            "kind": "stddev",
            "primitive_version": 1,
            "input": "price",
            "output": "price_std_2",
            "frame": {"kind": "rows", "size": 2},
            "min_periods": 1,
            "ddof": 1
        },
        {
            "kind": "sum",
            "primitive_version": 1,
            "input": "volume",
            "output": "volume_sum_2",
            "frame": {"kind": "rows", "size": 2},
            "min_periods": 1
        }
    ])
}

fn assert_aggregate_record(record: &RecordBatch) {
    let floats = |name: &str| -> Vec<Option<f64>> {
        record
            .column_by_name(name)
            .unwrap()
            .as_any()
            .downcast_ref::<Float64Array>()
            .unwrap()
            .iter()
            .collect()
    };
    let counts = record
        .column_by_name("price_count_2")
        .unwrap()
        .as_any()
        .downcast_ref::<UInt64Array>()
        .unwrap()
        .iter()
        .collect::<Vec<_>>();
    assert_eq!(counts, vec![Some(1), Some(2), Some(2)]);
    assert_eq!(floats("price_sum_2"), vec![Some(1.0), Some(4.0), Some(5.0)]);
    assert_eq!(
        floats("price_mean_2"),
        vec![Some(1.0), Some(2.0), Some(2.5)]
    );
    assert_eq!(floats("price_var_2"), vec![None, Some(2.0), Some(0.5)]);
    let stddev_values = floats("price_std_2");
    assert_eq!(stddev_values[0], None);
    assert!((stddev_values[1].unwrap() - 2.0_f64.sqrt()).abs() < 1e-15);
    assert!((stddev_values[2].unwrap() - 0.5_f64.sqrt()).abs() < 1e-15);
    let volume_sums = record
        .column_by_name("volume_sum_2")
        .unwrap()
        .as_any()
        .downcast_ref::<Int64Array>()
        .unwrap()
        .iter()
        .collect::<Vec<_>>();
    assert_eq!(volume_sums, vec![Some(10), Some(40), Some(50)]);
}

#[test]
fn rolling_project_accepts_aggregate_output_kinds() {
    let mut node = rolling_node_json();
    node["operator"]["spec"]["outputs"] = aggregate_outputs_json();
    let project: ProjectSpec = serde_json::from_value(project_json(
        &json!({"mode": "batch", "options": {}}),
        &node,
    ))
    .unwrap();
    let (providers, udfs) = registries();
    let report = validate_project(&project, &providers, &udfs);
    assert!(report.valid, "validation issues: {:?}", report.issues);

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
    assert_aggregate_record(&record);
}

#[test]
fn rolling_project_still_rejects_unsupported_output_kinds() {
    // SCE-08 delivered min/max/covariance/correlation; unknown catalog
    // kinds stay rejected.
    for kind in ["std", "ewma", "skew"] {
        let mut node = rolling_node_json();
        node["operator"]["spec"]["outputs"][0] = json!({
            "kind": kind,
            "primitive_version": 1,
            "input": "price",
            "output": "price_unsupported",
            "frame": {"kind": "rows", "size": 20},
            "min_periods": 1
        });
        assert!(
            serde_json::from_value::<ProjectSpec>(project_json(
                &json!({"mode": "batch", "options": {}}),
                &node
            ))
            .is_err(),
            "project accepted unsupported kind {kind}"
        );
    }
}

#[test]
fn rolling_project_requires_the_exact_input_port_contract() {
    let mut node = rolling_node_json();
    node["input_ports"][0]["name"] = json!("upstream");
    let project: ProjectSpec = serde_json::from_value(project_json(
        &json!({"mode": "batch", "options": {}}),
        &node,
    ))
    .unwrap();
    let (providers, udfs) = registries();
    let report = validate_project(&project, &providers, &udfs);
    assert!(!report.valid);

    let mut node = rolling_node_json();
    node["input_ports"][0]["schema"] = json!([]);
    let project: ProjectSpec = serde_json::from_value(project_json(
        &json!({"mode": "batch", "options": {}}),
        &node,
    ))
    .unwrap();
    let report = validate_project(&project, &providers, &udfs);
    assert!(!report.valid);
}

#[test]
fn rolling_project_derives_output_ports_when_omitted() {
    let project: ProjectSpec = serde_json::from_value(project_json(
        &json!({"mode": "batch", "options": {}}),
        &rolling_node_json(),
    ))
    .unwrap();
    let (providers, udfs) = registries();
    let report = validate_project(&project, &providers, &udfs);
    assert!(report.valid, "validation issues: {:?}", report.issues);
}

#[test]
fn rolling_project_rejects_multiple_input_ports_at_compile() {
    let mut node = rolling_node_json();
    node["input_ports"] = json!([
        node["input_ports"][0].clone(),
        {"name": "side", "kind": "table", "required": true, "schema": []}
    ]);
    let project: ProjectSpec = serde_json::from_value(project_json(
        &json!({"mode": "batch", "options": {}}),
        &node,
    ))
    .unwrap();
    let (providers, udfs) = registries();
    assert!(compile_project(&project, &providers, &udfs).is_err());
}

#[test]
fn rolling_project_rejects_wrong_input_port_name_at_compile() {
    let mut node = rolling_node_json();
    node["input_ports"][0]["name"] = json!("upstream");
    let project: ProjectSpec = serde_json::from_value(project_json(
        &json!({"mode": "batch", "options": {}}),
        &node,
    ))
    .unwrap();
    let (providers, udfs) = registries();
    assert!(compile_project(&project, &providers, &udfs).is_err());
}

#[test]
fn rolling_project_rejects_a_missing_exact_schema_at_compile() {
    let mut node = rolling_node_json();
    node["input_ports"][0]["schema"] = json!([]);
    let project: ProjectSpec = serde_json::from_value(project_json(
        &json!({"mode": "batch", "options": {}}),
        &node,
    ))
    .unwrap();
    let (providers, udfs) = registries();
    assert!(compile_project(&project, &providers, &udfs).is_err());
}

#[test]
fn rolling_project_reports_operator_construction_failure_as_a_validation_issue() {
    let mut node = rolling_node_json();
    node["operator"]["spec"]["outputs"][0]["periods"] = json!(0);
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
