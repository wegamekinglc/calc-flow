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
