use std::sync::Arc;

use calc_flow::{CalcFlowError, RollingNumericalProfile, RollingSpec};
use datafusion::arrow::datatypes::{DataType, Field, Schema, TimeUnit};
use serde_json::{Value, json};

fn input_schema() -> Schema {
    Schema::new(vec![
        Field::new(
            "ts",
            DataType::Timestamp(TimeUnit::Microsecond, Some(Arc::from("UTC"))),
            false,
        ),
        Field::new("symbol", DataType::Utf8, false),
        Field::new("sequence", DataType::UInt64, false),
        Field::new("price", DataType::Float64, true),
        Field::new("volume", DataType::Int64, true),
        Field::new("label", DataType::Utf8, true),
    ])
}

fn valid_spec_json() -> Value {
    json!({
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
    })
}

fn valid_spec() -> RollingSpec {
    serde_json::from_value(valid_spec_json()).unwrap()
}

fn with_field(schema: &Schema, index: usize, replacement: &Field) -> Schema {
    let fields = schema
        .fields()
        .iter()
        .enumerate()
        .map(|(position, field)| {
            if position == index {
                replacement.clone().into()
            } else {
                field.clone()
            }
        })
        .collect::<Vec<_>>();
    Schema::new(fields)
}

#[test]
fn canonical_lag_delta_spec_round_trips_the_frozen_json() {
    let spec: RollingSpec = serde_json::from_value(valid_spec_json()).unwrap();
    assert_eq!(spec.numerical_profile, RollingNumericalProfile::StableV1);
    assert_eq!(serde_json::to_value(&spec).unwrap(), valid_spec_json());
}

#[test]
fn drop_late_policy_uses_the_exact_frozen_shape() {
    let mut document = valid_spec_json();
    document["late_policy"] = json!({"kind": "drop", "metrics_version": 1});
    let spec: RollingSpec = serde_json::from_value(document.clone()).unwrap();
    assert_eq!(serde_json::to_value(&spec).unwrap(), document);
}

#[test]
fn unknown_spec_field_is_rejected() {
    let mut document = valid_spec_json();
    document["unexpected"] = json!(true);
    assert!(serde_json::from_value::<RollingSpec>(document).is_err());
}

#[test]
fn missing_semantic_field_is_rejected() {
    let mut document = valid_spec_json();
    document.as_object_mut().unwrap().remove("value_policy");
    assert!(serde_json::from_value::<RollingSpec>(document).is_err());
}

#[test]
fn unsupported_output_kind_is_rejected() {
    for kind in ["std", "median", "skew"] {
        let mut document = valid_spec_json();
        document["outputs"][0] = json!({
            "kind": kind,
            "primitive_version": 1,
            "input": "price",
            "output": "price_unsupported",
            "frame": {"kind": "rows", "size": 20},
            "min_periods": 1
        });
        assert!(
            serde_json::from_value::<RollingSpec>(document).is_err(),
            "unsupported kind {kind} was accepted"
        );
    }
}

#[test]
fn ewma_requires_layout_two_and_has_float64_output() {
    let mut document = valid_spec_json();
    document["state_layout_version"] = json!(2);
    document["outputs"] = json!([{
        "kind": "ewma",
        "primitive_version": 1,
        "input": "volume",
        "output": "volume_ema",
        "span": 3,
        "min_periods": 7
    }]);
    let spec: RollingSpec = serde_json::from_value(document).unwrap();
    let schema = spec.validate(&input_schema()).unwrap();
    assert_eq!(
        schema.field_with_name("volume_ema").unwrap().data_type(),
        &DataType::Float64
    );

    let mut layout_one = serde_json::to_value(spec).unwrap();
    layout_one["state_layout_version"] = json!(1);
    let error = serde_json::from_value::<RollingSpec>(layout_one)
        .unwrap()
        .validate(&input_schema())
        .unwrap_err();
    assert!(
        matches!(
            error,
            CalcFlowError::InvalidArgument { ref field, .. }
                if field == "rolling.state_layout_version"
        ),
        "unexpected error: {error}"
    );
}

#[test]
fn lag_output_rejects_aggregate_only_fields() {
    for field in ["frame", "min_periods", "ddof"] {
        let mut document = valid_spec_json();
        document["outputs"][0][field] = json!(1);
        assert!(
            serde_json::from_value::<RollingSpec>(document).is_err(),
            "lag accepted aggregate-only field {field}"
        );
    }
}

#[test]
fn unknown_value_policy_is_rejected() {
    let mut document = valid_spec_json();
    document["value_policy"] = json!("lenient");
    assert!(serde_json::from_value::<RollingSpec>(document).is_err());
}

#[test]
fn error_late_policy_rejects_metrics_version_and_drop_rejects_scope() {
    let mut document = valid_spec_json();
    document["late_policy"] = json!({"kind": "error", "scope": "envelope", "metrics_version": 1});
    assert!(serde_json::from_value::<RollingSpec>(document).is_err());
    let mut document = valid_spec_json();
    document["late_policy"] = json!({"kind": "drop", "metrics_version": 1, "scope": "envelope"});
    assert!(serde_json::from_value::<RollingSpec>(document).is_err());
}

#[test]
fn valid_spec_derives_the_output_schema() {
    let output_schema = valid_spec().validate(&input_schema()).unwrap();
    let expected = Schema::new(vec![
        Field::new(
            "ts",
            DataType::Timestamp(TimeUnit::Microsecond, Some(Arc::from("UTC"))),
            false,
        ),
        Field::new("symbol", DataType::Utf8, false),
        Field::new("sequence", DataType::UInt64, false),
        Field::new("price", DataType::Float64, true),
        Field::new("volume", DataType::Int64, true),
        Field::new("label", DataType::Utf8, true),
        Field::new("price_lag_1", DataType::Float64, true),
        Field::new("volume_delta_1", DataType::Int64, true),
    ]);
    assert_eq!(output_schema.as_ref(), &expected);
}

#[test]
fn unsupported_configuration_version_is_rejected() {
    let mut spec = valid_spec();
    spec.configuration_version = 2;
    let error = spec.validate(&input_schema()).unwrap_err();
    assert!(
        matches!(
            error,
            CalcFlowError::InvalidArgument { ref field, .. }
                if field == "rolling.configuration_version"
        ),
        "unexpected error: {error}"
    );
}

#[test]
fn unsupported_state_layout_version_is_rejected() {
    let mut spec = valid_spec();
    spec.state_layout_version = 0;
    let error = spec.validate(&input_schema()).unwrap_err();
    assert!(
        matches!(
            error,
            CalcFlowError::InvalidArgument { ref field, .. }
                if field == "rolling.state_layout_version"
        ),
        "unexpected error: {error}"
    );
}

#[test]
fn empty_partition_by_is_rejected() {
    let mut spec = valid_spec();
    spec.partition_by = Vec::new();
    let error = spec.validate(&input_schema()).unwrap_err();
    assert!(
        matches!(
            error,
            CalcFlowError::InvalidArgument { ref field, .. }
                if field == "rolling.partition_by"
        ),
        "unexpected error: {error}"
    );
}

#[test]
fn duplicate_partition_column_is_rejected() {
    let mut spec = valid_spec();
    spec.partition_by = vec!["symbol".into(), "symbol".into()];
    let error = spec.validate(&input_schema()).unwrap_err();
    assert!(
        matches!(
            error,
            CalcFlowError::InvalidArgument { ref field, .. }
                if field == "rolling.partition_by[1]"
        ),
        "unexpected error: {error}"
    );
}

#[test]
fn missing_partition_column_is_rejected() {
    let mut spec = valid_spec();
    spec.partition_by = vec!["industry".into()];
    let error = spec.validate(&input_schema()).unwrap_err();
    assert!(
        matches!(error, CalcFlowError::Compile { .. }),
        "unexpected error: {error}"
    );
}

#[test]
fn unsupported_partition_column_type_is_rejected() {
    let schema = with_field(
        &input_schema(),
        1,
        &Field::new("symbol", DataType::LargeBinary, false),
    );
    let error = valid_spec().validate(&schema).unwrap_err();
    assert!(
        matches!(error, CalcFlowError::Compile { .. }),
        "unexpected error: {error}"
    );
}

#[test]
fn missing_event_time_column_is_rejected() {
    let mut spec = valid_spec();
    spec.event_time = "event_ts".into();
    let error = spec.validate(&input_schema()).unwrap_err();
    assert!(
        matches!(error, CalcFlowError::Compile { .. }),
        "unexpected error: {error}"
    );
}

#[test]
fn nullable_event_time_is_rejected() {
    let schema = with_field(
        &input_schema(),
        0,
        &Field::new(
            "ts",
            DataType::Timestamp(TimeUnit::Microsecond, Some(Arc::from("UTC"))),
            true,
        ),
    );
    let error = valid_spec().validate(&schema).unwrap_err();
    assert!(
        matches!(error, CalcFlowError::Compile { .. }),
        "unexpected error: {error}"
    );
}

#[test]
fn non_utc_or_coarse_event_time_is_rejected() {
    for data_type in [
        DataType::Timestamp(TimeUnit::Microsecond, None),
        DataType::Timestamp(TimeUnit::Millisecond, Some(Arc::from("UTC"))),
        DataType::Timestamp(TimeUnit::Microsecond, Some(Arc::from("Asia/Shanghai"))),
        DataType::Int64,
    ] {
        let schema = with_field(
            &input_schema(),
            0,
            &Field::new("ts", data_type.clone(), false),
        );
        let error = valid_spec().validate(&schema).unwrap_err();
        assert!(
            matches!(error, CalcFlowError::Compile { .. }),
            "event-time type {data_type} was accepted"
        );
    }
}

#[test]
fn empty_sequence_by_is_rejected() {
    let mut spec = valid_spec();
    spec.sequence_by = Vec::new();
    let error = spec.validate(&input_schema()).unwrap_err();
    assert!(
        matches!(
            error,
            CalcFlowError::InvalidArgument { ref field, .. }
                if field == "rolling.sequence_by"
        ),
        "unexpected error: {error}"
    );
}

#[test]
fn nullable_sequence_column_is_rejected() {
    let schema = with_field(
        &input_schema(),
        2,
        &Field::new("sequence", DataType::UInt64, true),
    );
    let error = valid_spec().validate(&schema).unwrap_err();
    assert!(
        matches!(error, CalcFlowError::Compile { .. }),
        "unexpected error: {error}"
    );
}

#[test]
fn floating_sequence_column_is_rejected() {
    let schema = with_field(
        &input_schema(),
        2,
        &Field::new("sequence", DataType::Float64, false),
    );
    let error = valid_spec().validate(&schema).unwrap_err();
    assert!(
        matches!(error, CalcFlowError::Compile { .. }),
        "unexpected error: {error}"
    );
}

#[test]
fn empty_outputs_are_rejected() {
    let mut spec = valid_spec();
    spec.outputs = Vec::new();
    let error = spec.validate(&input_schema()).unwrap_err();
    assert!(
        matches!(
            error,
            CalcFlowError::InvalidArgument { ref field, .. }
                if field == "rolling.outputs"
        ),
        "unexpected error: {error}"
    );
}

#[test]
fn zero_periods_is_rejected() {
    for index in [0, 1] {
        let mut document = valid_spec_json();
        document["outputs"][index]["periods"] = json!(0);
        let spec: RollingSpec = serde_json::from_value(document).unwrap();
        let error = spec.validate(&input_schema()).unwrap_err();
        assert!(
            matches!(
                error,
                CalcFlowError::InvalidArgument { ref field, .. }
                    if field == &format!("rolling.outputs[{index}].periods")
            ),
            "unexpected error: {error}"
        );
    }
}

#[test]
fn unsupported_primitive_version_is_rejected() {
    let mut document = valid_spec_json();
    document["outputs"][0]["primitive_version"] = json!(2);
    let spec: RollingSpec = serde_json::from_value(document).unwrap();
    let error = spec.validate(&input_schema()).unwrap_err();
    assert!(
        matches!(
            error,
            CalcFlowError::InvalidArgument { ref field, .. }
                if field == "rolling.outputs[0].primitive_version"
        ),
        "unexpected error: {error}"
    );
}

#[test]
fn missing_output_input_column_is_rejected() {
    let mut document = valid_spec_json();
    document["outputs"][0]["input"] = json!("close");
    let spec: RollingSpec = serde_json::from_value(document).unwrap();
    let error = spec.validate(&input_schema()).unwrap_err();
    assert!(
        matches!(error, CalcFlowError::Compile { .. }),
        "unexpected error: {error}"
    );
}

#[test]
fn delta_on_non_numeric_column_is_rejected() {
    let mut document = valid_spec_json();
    document["outputs"][1]["input"] = json!("label");
    let spec: RollingSpec = serde_json::from_value(document).unwrap();
    let error = spec.validate(&input_schema()).unwrap_err();
    assert!(
        matches!(error, CalcFlowError::Compile { .. }),
        "unexpected error: {error}"
    );
}

#[test]
fn lag_preserves_any_input_type() {
    let mut document = valid_spec_json();
    document["outputs"][0]["input"] = json!("label");
    document["outputs"][0]["output"] = json!("label_lag_1");
    let spec: RollingSpec = serde_json::from_value(document).unwrap();
    let output_schema = spec.validate(&input_schema()).unwrap();
    assert_eq!(
        output_schema.field_with_name("label_lag_1").unwrap(),
        &Field::new("label_lag_1", DataType::Utf8, true)
    );
}

#[test]
fn duplicate_output_name_is_rejected() {
    let mut document = valid_spec_json();
    document["outputs"][1]["output"] = json!("price_lag_1");
    let spec: RollingSpec = serde_json::from_value(document).unwrap();
    let error = spec.validate(&input_schema()).unwrap_err();
    assert!(
        matches!(
            error,
            CalcFlowError::InvalidArgument { ref field, .. }
                if field == "rolling.outputs[1].output"
        ),
        "unexpected error: {error}"
    );
}

#[test]
fn output_name_colliding_with_an_input_field_is_rejected() {
    let mut document = valid_spec_json();
    document["outputs"][1]["output"] = json!("volume");
    let spec: RollingSpec = serde_json::from_value(document).unwrap();
    let error = spec.validate(&input_schema()).unwrap_err();
    assert!(
        matches!(
            error,
            CalcFlowError::InvalidArgument { ref field, .. }
                if field == "rolling.outputs[1].output"
        ),
        "unexpected error: {error}"
    );
}

#[test]
fn drop_metrics_version_must_be_one() {
    let mut document = valid_spec_json();
    document["late_policy"] = json!({"kind": "drop", "metrics_version": 2});
    let spec: RollingSpec = serde_json::from_value(document).unwrap();
    let error = spec.validate(&input_schema()).unwrap_err();
    assert!(
        matches!(
            error,
            CalcFlowError::InvalidArgument { ref field, .. }
                if field == "rolling.late_policy.metrics_version"
        ),
        "unexpected error: {error}"
    );
}

mod operator_contract {
    use super::*;
    use calc_flow::{
        Batch, BatchKind, BatchMetadata, BatchOperator, BatchOperatorContext, JsonMap,
        LateErrorScope, LatePolicySpec, OperatorMetadata, Port, RollingFrameSpec, RollingOperator,
        RollingOutputSpec,
    };
    use datafusion::arrow::record_batch::RecordBatch;
    use std::collections::BTreeMap;

    fn aggregate_spec_json(outputs: Value) -> Value {
        let mut document = valid_spec_json();
        document["outputs"] = outputs;
        document
    }

    fn aggregate_output(kind: &str, input: &str, output: &str, size: u64) -> Value {
        json!({
            "kind": kind,
            "primitive_version": 1,
            "input": input,
            "output": output,
            "frame": {"kind": "rows", "size": size},
            "min_periods": 1
        })
    }

    fn ddof_output(kind: &str, input: &str, output: &str, size: u64, ddof: u64) -> Value {
        let mut declaration = aggregate_output(kind, input, output, size);
        declaration["ddof"] = json!(ddof);
        declaration
    }

    fn duration_output(kind: &str, input: &str, output: &str, micros: u64) -> Value {
        json!({
            "kind": kind,
            "primitive_version": 1,
            "input": input,
            "output": output,
            "frame": {"kind": "duration", "micros": micros},
            "min_periods": 1
        })
    }

    fn pair_output(
        kind: &str,
        left: &str,
        right: &str,
        output: &str,
        frame: Value,
        ddof: u64,
    ) -> Value {
        let mut declaration = json!({
            "kind": kind,
            "primitive_version": 1,
            "left": left,
            "right": right,
            "output": output,
            "min_periods": 1,
            "ddof": ddof
        });
        declaration["frame"] = frame;
        declaration
    }

    fn aggregate_spec(outputs: Value) -> RollingSpec {
        serde_json::from_value(aggregate_spec_json(outputs)).unwrap()
    }

    #[test]
    fn test_side_output_validates_versions() {
        for field in ["metrics_version", "schema_version"] {
            for version in [0, 2] {
                let mut policy = serde_json::json!({"kind": "side_output", "metrics_version": 1, "schema_version": 1});
                policy[field] = serde_json::json!(version);
                let mut spec = valid_spec();
                spec.late_policy = serde_json::from_value(policy).unwrap();
                let error = spec.validate(&input_schema()).unwrap_err();
                assert!(
                    matches!(error, CalcFlowError::InvalidArgument { field: ref name, .. } if name.ends_with(field)),
                    "{error}"
                );
                assert!(RollingOperator::new("features", Arc::new(input_schema()), spec).is_err());
            }
            let mut policy = serde_json::json!({"kind": "side_output", "metrics_version": 1, "schema_version": 1});
            policy.as_object_mut().unwrap().remove(field);
            assert!(serde_json::from_value::<LatePolicySpec>(policy).is_err());
        }
    }

    #[test]
    fn test_side_output_validates_reserved_prefix() {
        for name in ["_cf_late_reason", "_cf_late_future"] {
            let mut fields = input_schema().fields().to_vec();
            fields.push(Arc::new(Field::new(name, DataType::Utf8, true)));
            let input = Arc::new(Schema::new(fields));
            for policy in [
                LatePolicySpec::Error {
                    scope: LateErrorScope::Envelope,
                },
                LatePolicySpec::Drop { metrics_version: 1 },
            ] {
                let mut spec = valid_spec();
                spec.late_policy = policy;
                assert!(spec.validate(&input).is_ok());
                assert_eq!(
                    RollingOperator::new("features", input.clone(), spec)
                        .unwrap()
                        .output_ports()
                        .len(),
                    1
                );
            }
            let mut spec = valid_spec();
            spec.late_policy = LatePolicySpec::SideOutput {
                metrics_version: 1,
                schema_version: 1,
            };
            let error = spec.validate(&input).unwrap_err().to_string();
            assert!(
                error.contains(name) && error.contains("reserved"),
                "{error}"
            );
            assert!(RollingOperator::new("features", input, spec).is_err());
        }
    }

    #[test]
    fn test_side_output_derives_exact_ports_and_preserves_arrow_metadata() {
        let input = input_schema();
        let fields = input
            .fields()
            .iter()
            .map(|field| {
                field.as_ref().clone().with_metadata(
                    [("field_key".into(), "field_value".into())]
                        .into_iter()
                        .collect(),
                )
            })
            .collect::<Vec<_>>();
        let input = Arc::new(Schema::new_with_metadata(
            fields,
            [("schema_key".into(), "schema_value".into())]
                .into_iter()
                .collect(),
        ));
        let mut spec = valid_spec();
        let normal = spec.validate(&input).unwrap();
        spec.late_policy = serde_json::from_value(serde_json::json!({
            "kind": "side_output", "metrics_version": 1, "schema_version": 1
        }))
        .unwrap();
        assert_eq!(spec.validate(&input).unwrap(), normal);
        let operator = RollingOperator::new("features", input.clone(), spec).unwrap();
        let ports = operator.output_ports();
        assert_eq!(
            ports.iter().map(Port::name).collect::<Vec<_>>(),
            ["output", "late"]
        );
        assert!(
            ports
                .iter()
                .all(|port| port.required() && port.kind() == BatchKind::Table)
        );
        assert_eq!(ports[0].schema(), Some(&normal));
        let late = ports[1].schema().unwrap();
        assert_eq!(late.metadata(), input.metadata());
        assert_eq!(
            &late.fields()[..input.fields().len()],
            input.fields().as_ref()
        );
        let diagnostics = [
            ("_cf_late_node", DataType::Utf8),
            ("_cf_late_input_port", DataType::Utf8),
            ("_cf_late_event_time_micros", DataType::Int64),
            ("_cf_late_closing_time_micros", DataType::Int64),
            ("_cf_late_watermark_micros", DataType::Int64),
            ("_cf_late_reason", DataType::Utf8),
            ("_cf_late_source", DataType::Utf8),
            ("_cf_late_sequence", DataType::UInt64),
            ("_cf_late_row_index", DataType::UInt64),
        ];
        assert_eq!(
            late.fields().len(),
            input.fields().len() + diagnostics.len()
        );
        for (field, (name, data_type)) in late.fields()[input.fields().len()..]
            .iter()
            .zip(diagnostics)
        {
            assert_eq!(field.as_ref(), &Field::new(name, data_type, false));
        }
    }

    #[test]
    fn aggregate_outputs_round_trip_the_frozen_json() {
        let document = aggregate_spec_json(json!([
            aggregate_output("count", "price", "price_count_20", 20),
            aggregate_output("sum", "volume", "volume_sum_20", 20),
            aggregate_output("mean", "price", "price_mean_20", 20),
            ddof_output("variance", "price", "price_var_20", 20, 1),
            ddof_output("stddev", "price", "price_std_20", 20, 0),
        ]));
        let spec: RollingSpec = serde_json::from_value(document.clone()).unwrap();
        assert_eq!(serde_json::to_value(&spec).unwrap(), document);
    }

    #[test]
    fn duration_frames_round_trip_the_frozen_json() {
        let document = aggregate_spec_json(json!([
            duration_output("mean", "price", "price_mean_60s", 60_000_000),
            duration_output("count", "label", "label_count_60s", 60_000_000),
            pair_output(
                "correlation",
                "price",
                "volume",
                "price_volume_corr_60s",
                json!({"kind": "duration", "micros": 60_000_000}),
                1,
            ),
        ]));
        let spec: RollingSpec = serde_json::from_value(document.clone()).unwrap();
        assert_eq!(serde_json::to_value(&spec).unwrap(), document);
    }

    #[test]
    fn zero_duration_micros_is_rejected() {
        let spec = aggregate_spec(json!([duration_output("mean", "price", "m", 0)]));
        let error = spec.validate(&input_schema()).unwrap_err();
        assert!(
            matches!(
                error,
                CalcFlowError::InvalidArgument { ref field, .. }
                    if field == "rolling.outputs[0].frame.micros"
            ),
            "unexpected error: {error}"
        );
    }

    #[test]
    fn duration_frames_allow_min_periods_above_any_row_count() {
        // A duration frame has no row-count ceiling; only row frames cap
        // min_periods at their size (SCE-00 D5).
        let mut declaration = duration_output("mean", "price", "m", 60_000_000);
        declaration["min_periods"] = json!(10_000);
        let spec = aggregate_spec(json!([declaration]));
        assert!(spec.validate(&input_schema()).is_ok());
    }

    #[test]
    fn duration_frames_still_reject_zero_min_periods() {
        let mut declaration = duration_output("mean", "price", "m", 60_000_000);
        declaration["min_periods"] = json!(0);
        let spec = aggregate_spec(json!([declaration]));
        let error = spec.validate(&input_schema()).unwrap_err();
        assert!(
            matches!(
                error,
                CalcFlowError::InvalidArgument { ref field, .. }
                    if field == "rolling.outputs[0].min_periods"
            ),
            "unexpected error: {error}"
        );
    }

    #[test]
    fn duration_frames_reject_unknown_frame_fields() {
        let mut declaration = duration_output("mean", "price", "m", 60_000_000);
        declaration["frame"]["size"] = json!(5);
        let document = aggregate_spec_json(json!([declaration]));
        assert!(serde_json::from_value::<RollingSpec>(document).is_err());
    }

    #[test]
    fn extrema_outputs_round_trip_the_frozen_json() {
        let document = aggregate_spec_json(json!([
            aggregate_output("min", "price", "price_min_20", 20),
            aggregate_output("max", "price", "price_max_20", 20),
            duration_output("max", "volume", "volume_max_60s", 60_000_000),
        ]));
        let spec: RollingSpec = serde_json::from_value(document.clone()).unwrap();
        assert_eq!(serde_json::to_value(&spec).unwrap(), document);
    }

    #[test]
    fn extrema_outputs_reject_ddof() {
        for kind in ["min", "max"] {
            let declaration = ddof_output(kind, "price", "price_extrema", 20, 1);
            let document = aggregate_spec_json(json!([declaration]));
            assert!(
                serde_json::from_value::<RollingSpec>(document).is_err(),
                "{kind} with ddof was accepted"
            );
        }
    }

    #[test]
    fn pair_outputs_round_trip_the_frozen_json() {
        let document = aggregate_spec_json(json!([
            pair_output(
                "covariance",
                "price",
                "volume",
                "price_volume_cov_20",
                json!({"kind": "rows", "size": 20}),
                1,
            ),
            pair_output(
                "correlation",
                "price",
                "volume",
                "price_volume_corr_20",
                json!({"kind": "rows", "size": 20}),
                0,
            ),
        ]));
        let spec: RollingSpec = serde_json::from_value(document.clone()).unwrap();
        assert_eq!(serde_json::to_value(&spec).unwrap(), document);
    }

    #[test]
    fn pair_outputs_reject_missing_ddof_left_or_right() {
        let mut missing_ddof = pair_output(
            "covariance",
            "price",
            "volume",
            "price_volume_cov",
            json!({"kind": "rows", "size": 20}),
            1,
        );
        missing_ddof.as_object_mut().unwrap().remove("ddof");
        assert!(
            serde_json::from_value::<RollingSpec>(aggregate_spec_json(json!([missing_ddof])))
                .is_err()
        );
        for field in ["left", "right"] {
            let mut missing_operand = pair_output(
                "correlation",
                "price",
                "volume",
                "price_volume_corr",
                json!({"kind": "rows", "size": 20}),
                1,
            );
            missing_operand.as_object_mut().unwrap().remove(field);
            assert!(
                serde_json::from_value::<RollingSpec>(aggregate_spec_json(json!([
                    missing_operand
                ])))
                .is_err(),
                "pair output without {field} was accepted"
            );
        }
    }

    #[test]
    fn pair_outputs_reject_the_single_input_field() {
        let mut declaration = pair_output(
            "covariance",
            "price",
            "volume",
            "price_volume_cov",
            json!({"kind": "rows", "size": 20}),
            1,
        );
        declaration["input"] = json!("price");
        let document = aggregate_spec_json(json!([declaration]));
        assert!(serde_json::from_value::<RollingSpec>(document).is_err());
    }

    #[test]
    fn single_input_outputs_reject_left_and_right() {
        for kind in ["mean", "min", "max"] {
            let mut declaration = aggregate_output(kind, "price", "price_agg", 20);
            declaration["left"] = json!("price");
            let document = aggregate_spec_json(json!([declaration]));
            assert!(
                serde_json::from_value::<RollingSpec>(document).is_err(),
                "{kind} with left was accepted"
            );
        }
    }

    #[test]
    fn pair_outputs_reject_an_empty_right_column() {
        let spec = aggregate_spec(json!([pair_output(
            "covariance",
            "price",
            "",
            "price_volume_cov",
            json!({"kind": "rows", "size": 20}),
            1,
        )]));
        let error = spec.validate(&input_schema()).unwrap_err();
        assert!(
            matches!(
                error,
                CalcFlowError::InvalidArgument { ref field, .. }
                    if field == "rolling.outputs[0].right"
            ),
            "unexpected error: {error}"
        );
    }

    #[test]
    fn extrema_over_a_column_without_total_order_are_rejected() {
        let schema = with_field(
            &input_schema(),
            5,
            &Field::new(
                "label",
                DataType::Timestamp(TimeUnit::Nanosecond, None),
                true,
            ),
        );
        for kind in ["min", "max"] {
            let spec = aggregate_spec(json!([aggregate_output(
                kind,
                "label",
                "label_extrema_20",
                20
            )]));
            let error = spec.validate(&schema).unwrap_err();
            let expected = format!("rolling {kind} does not support column");
            assert!(
                matches!(
                    error,
                    CalcFlowError::Compile { ref message } if message.contains(&expected)
                ),
                "unexpected error for {kind}: {error}"
            );
        }
    }

    #[test]
    fn pair_output_ddof_above_one_is_rejected() {
        let spec = aggregate_spec(json!([pair_output(
            "correlation",
            "price",
            "volume",
            "price_volume_corr",
            json!({"kind": "rows", "size": 20}),
            2,
        )]));
        let error = spec.validate(&input_schema()).unwrap_err();
        assert!(
            matches!(
                error,
                CalcFlowError::InvalidArgument { ref field, .. }
                    if field == "rolling.outputs[0].ddof"
            ),
            "unexpected error: {error}"
        );
    }

    #[test]
    fn pair_outputs_reject_non_numeric_operands() {
        for left in ["label", "price"] {
            let right = if left == "label" { "price" } else { "label" };
            let spec = aggregate_spec(json!([pair_output(
                "covariance",
                left,
                right,
                "pair_stat",
                json!({"kind": "rows", "size": 20}),
                1,
            )]));
            let error = spec.validate(&input_schema()).unwrap_err();
            assert!(
                matches!(error, CalcFlowError::Compile { .. }),
                "unexpected error: {error}"
            );
        }
    }

    #[test]
    fn aggregate_outputs_reject_lag_only_fields() {
        let mut declaration = aggregate_output("mean", "price", "price_mean", 20);
        declaration["periods"] = json!(1);
        let document = aggregate_spec_json(json!([declaration]));
        assert!(serde_json::from_value::<RollingSpec>(document).is_err());
    }

    #[test]
    fn statistical_outputs_reject_missing_ddof() {
        for kind in ["variance", "stddev"] {
            let declaration = aggregate_output(kind, "price", "price_stat", 20);
            let document = aggregate_spec_json(json!([declaration]));
            assert!(
                serde_json::from_value::<RollingSpec>(document).is_err(),
                "{kind} without ddof was accepted"
            );
        }
    }

    #[test]
    fn non_statistical_aggregates_reject_ddof() {
        for kind in ["count", "sum", "mean"] {
            let declaration = ddof_output(kind, "price", "price_agg", 20, 1);
            let document = aggregate_spec_json(json!([declaration]));
            assert!(
                serde_json::from_value::<RollingSpec>(document).is_err(),
                "{kind} with ddof was accepted"
            );
        }
    }

    #[test]
    fn aggregate_output_schema_uses_the_frozen_type_table() {
        let spec = aggregate_spec(json!([
            aggregate_output("count", "price", "price_count", 20),
            aggregate_output("count", "label", "label_count", 20),
            aggregate_output("sum", "volume", "volume_sum", 20),
            aggregate_output("sum", "price", "price_sum", 20),
            aggregate_output("mean", "volume", "volume_mean", 20),
            ddof_output("variance", "price", "price_var", 20, 1),
            ddof_output("stddev", "volume", "volume_std", 20, 0),
        ]));
        let output_schema = spec.validate(&input_schema()).unwrap();
        let derived = &output_schema.fields()[input_schema().fields().len()..];
        let expected = [
            ("price_count", DataType::UInt64),
            ("label_count", DataType::UInt64),
            ("volume_sum", DataType::Int64),
            ("price_sum", DataType::Float64),
            ("volume_mean", DataType::Float64),
            ("price_var", DataType::Float64),
            ("volume_std", DataType::Float64),
        ];
        assert_eq!(derived.len(), expected.len());
        for (field, (name, data_type)) in derived.iter().zip(expected) {
            assert_eq!(field.name(), name);
            assert_eq!(field.data_type(), &data_type);
            assert!(field.is_nullable());
        }
    }

    #[test]
    fn extrema_and_pair_output_schema_uses_the_frozen_type_table() {
        let spec = aggregate_spec(json!([
            aggregate_output("min", "price", "price_min", 20),
            aggregate_output("max", "volume", "volume_max", 20),
            aggregate_output("max", "label", "label_max", 20),
            pair_output(
                "covariance",
                "price",
                "volume",
                "price_volume_cov",
                json!({"kind": "rows", "size": 20}),
                1,
            ),
            pair_output(
                "correlation",
                "price",
                "volume",
                "price_volume_corr",
                json!({"kind": "duration", "micros": 60_000_000}),
                1,
            ),
        ]));
        let output_schema = spec.validate(&input_schema()).unwrap();
        let derived = &output_schema.fields()[input_schema().fields().len()..];
        let expected = [
            ("price_min", DataType::Float64),
            ("volume_max", DataType::Int64),
            ("label_max", DataType::Utf8),
            ("price_volume_cov", DataType::Float64),
            ("price_volume_corr", DataType::Float64),
        ];
        assert_eq!(derived.len(), expected.len());
        for (field, (name, data_type)) in derived.iter().zip(expected) {
            assert_eq!(field.name(), name);
            assert_eq!(field.data_type(), &data_type);
            assert!(field.is_nullable());
        }
    }

    #[test]
    fn zero_frame_size_is_rejected() {
        let mut spec = aggregate_spec(json!([aggregate_output("mean", "price", "m", 20)]));
        let RollingOutputSpec::Mean { frame, .. } = &mut spec.outputs[0] else {
            panic!("expected a mean output");
        };
        let RollingFrameSpec::Rows { size } = frame else {
            panic!("expected a rows frame");
        };
        *size = 0;
        let error = spec.validate(&input_schema()).unwrap_err();
        assert!(
            matches!(
                error,
                CalcFlowError::InvalidArgument { ref field, .. }
                    if field == "rolling.outputs[0].frame.size"
            ),
            "unexpected error: {error}"
        );
    }

    #[test]
    fn zero_min_periods_is_rejected() {
        let mut spec = aggregate_spec(json!([aggregate_output("mean", "price", "m", 20)]));
        let RollingOutputSpec::Mean { min_periods, .. } = &mut spec.outputs[0] else {
            panic!("expected a mean output");
        };
        *min_periods = 0;
        let error = spec.validate(&input_schema()).unwrap_err();
        assert!(
            matches!(
                error,
                CalcFlowError::InvalidArgument { ref field, .. }
                    if field == "rolling.outputs[0].min_periods"
            ),
            "unexpected error: {error}"
        );
    }

    #[test]
    fn min_periods_above_the_frame_size_is_rejected() {
        let mut declaration = aggregate_output("mean", "price", "m", 3);
        declaration["min_periods"] = json!(4);
        let spec = aggregate_spec(json!([declaration]));
        let error = spec.validate(&input_schema()).unwrap_err();
        assert!(
            matches!(
                error,
                CalcFlowError::InvalidArgument { ref field, .. }
                    if field == "rolling.outputs[0].min_periods"
            ),
            "unexpected error: {error}"
        );
    }

    #[test]
    fn ddof_above_one_is_rejected() {
        let mut spec = aggregate_spec(json!([ddof_output("variance", "price", "v", 20, 1)]));
        let RollingOutputSpec::Variance { ddof, .. } = &mut spec.outputs[0] else {
            panic!("expected a variance output");
        };
        *ddof = 2;
        let error = spec.validate(&input_schema()).unwrap_err();
        assert!(
            matches!(
                error,
                CalcFlowError::InvalidArgument { ref field, .. }
                    if field == "rolling.outputs[0].ddof"
            ),
            "unexpected error: {error}"
        );
    }

    #[test]
    fn sum_mean_variance_and_stddev_reject_non_numeric_inputs() {
        for declaration in [
            aggregate_output("sum", "label", "label_sum", 20),
            aggregate_output("mean", "label", "label_mean", 20),
            ddof_output("variance", "label", "label_var", 20, 1),
            ddof_output("stddev", "label", "label_std", 20, 1),
        ] {
            let spec = aggregate_spec(json!([declaration]));
            let error = spec.validate(&input_schema()).unwrap_err();
            assert!(
                matches!(error, CalcFlowError::Compile { .. }),
                "unexpected error: {error}"
            );
        }
    }

    #[test]
    fn count_accepts_non_numeric_inputs() {
        let spec = aggregate_spec(json!([aggregate_output("count", "label", "n", 20)]));
        assert!(spec.validate(&input_schema()).is_ok());
    }

    #[test]
    fn operator_exposes_exact_ports_and_frozen_configuration() {
        let operator =
            RollingOperator::new("rolling_features", Arc::new(input_schema()), valid_spec())
                .unwrap();
        assert_eq!(operator.name(), "rolling_features");
        let [input] = operator.input_ports() else {
            panic!("rolling exposes one input port");
        };
        assert_eq!(input.name(), "input");
        assert!(input.required());
        assert_eq!(input.schema().unwrap().as_ref(), &input_schema());
        let [output] = operator.output_ports() else {
            panic!("rolling exposes one output port");
        };
        assert_eq!(output.name(), "output");
        assert!(output.required());
        assert_eq!(
            output.schema().unwrap().as_ref(),
            valid_spec().validate(&input_schema()).unwrap().as_ref()
        );
        assert_eq!(
            serde_json::to_value(operator.configuration()).unwrap(),
            json!({
                "kind": "rolling",
                "spec": valid_spec_json(),
            })
        );
    }

    #[test]
    fn spec_getter_returns_the_validated_declaration_and_debug_stays_non_exhaustive() {
        let spec = valid_spec();
        let operator =
            RollingOperator::new("rolling_features", Arc::new(input_schema()), spec.clone())
                .unwrap();
        assert_eq!(operator.spec(), &spec);
        let rendered = format!("{operator:?}");
        assert!(rendered.contains("RollingOperator"));
        assert!(rendered.contains("rolling_features"));
    }

    #[tokio::test]
    async fn test_side_output_direct_batch_execution_is_rejected() {
        let mut spec = valid_spec();
        spec.late_policy = LatePolicySpec::SideOutput {
            metrics_version: 1,
            schema_version: 1,
        };
        let input = Arc::new(input_schema());
        let mut operator = RollingOperator::new("features", input.clone(), spec).unwrap();
        let batch = Batch::table(
            vec![RecordBatch::new_empty(input)],
            BatchMetadata::default(),
        )
        .unwrap();
        let run =
            calc_flow::RunContext::new(JsonMap::new(), None, calc_flow::CancellationToken::new())
                .unwrap();
        let error = operator
            .process(
                &BTreeMap::from([("input".into(), batch)]),
                &BatchOperatorContext { run: &run },
            )
            .await
            .unwrap_err();
        assert!(error.to_string().contains("unsupported_mode"), "{error}");
    }
}
