mod late;

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
        crate::RunContext::new(JsonMap::new(), None, crate::CancellationToken::new()).unwrap();
    let error = operator
        .process(
            &BTreeMap::from([("input".into(), batch)]),
            &BatchOperatorContext { run: &run },
        )
        .await
        .unwrap_err();
    assert!(error.to_string().contains("unsupported_mode"), "{error}");
}

#[tokio::test]
async fn test_side_output_direct_execution_completes_both_ports() {
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
    let job = crate::StreamJobContext::new(
        1,
        "0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef",
        JsonMap::new(),
        None,
        crate::CancellationToken::new(),
    );
    let context = StreamOperatorContext::new(&job, "features", None);
    let mut output = crate::EdgeCollector::new(operator.output_ports().to_vec());
    operator
        .process_data("input", batch, &context, &mut output)
        .await
        .unwrap();
    operator
        .on_watermark(EventTime::from_micros(1), &context, &mut output)
        .await
        .unwrap();
    operator.on_end(&context, &mut output).await.unwrap();
    assert!(output.drain("output").is_empty());
    assert!(output.drain("late").is_empty());
    assert!(operator.state.ended);
    assert_eq!(
        operator.state.last_input_watermark,
        Some(EventTime::from_micros(1))
    );
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
        let mut policy =
            serde_json::json!({"kind": "side_output", "metrics_version": 1, "schema_version": 1});
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
use datafusion::arrow::array::Array;
use datafusion::arrow::datatypes::{DataType, Field, Schema, TimeUnit};
use serde_json::{Value, json};

use super::*;
use crate::{CalcFlowError, OperatorMetadata};

const TEST_FINGERPRINT: &str = "0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef";

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

// ------------------------------------------------------------------
// Strict serialized model
// ------------------------------------------------------------------

#[test]
fn canonical_lag_delta_spec_round_trips_the_frozen_json() {
    let spec: RollingSpec = serde_json::from_value(valid_spec_json()).unwrap();
    assert_eq!(spec.numerical_profile, RollingNumericalProfile::StableV1);
    assert_eq!(serde_json::to_value(&spec).unwrap(), valid_spec_json());
}

#[test]
fn stable_v2_profile_is_explicit_and_fingerprinted() {
    let mut document = valid_spec_json();
    document["numerical_profile"] = json!("stable_v2");
    let preview: RollingSpec = serde_json::from_value(document.clone()).unwrap();
    let stable = valid_spec();

    assert_eq!(
        preview.numerical_profile,
        RollingNumericalProfile::StableV2Preview
    );
    assert_eq!(serde_json::to_value(&preview).unwrap(), document);
    assert_ne!(
        compile_spec(&stable, &input_schema())
            .unwrap()
            .kernel_plan
            .fingerprint(),
        compile_spec(&preview, &input_schema())
            .unwrap()
            .kernel_plan
            .fingerprint()
    );
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

// ------------------------------------------------------------------
// Declaration and schema validation
// ------------------------------------------------------------------

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

// ------------------------------------------------------------------
// Aggregate declarations (SCE-07; SCE-00 D3, contract section 5.2; D5)
// ------------------------------------------------------------------

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

fn mean_leaf(input: &str, size: u64) -> Value {
    json!({
        "kind": "mean",
        "primitive_version": 1,
        "input": input,
        "frame": {"kind": "rows", "size": size},
        "min_periods": 1
    })
}

fn difference_output(left: &Value, right: &Value, output: &str) -> Value {
    json!({
        "kind": "difference",
        "primitive_version": 1,
        "left": left,
        "right": right,
        "output": output
    })
}

fn aggregate_spec(outputs: Value) -> RollingSpec {
    serde_json::from_value(aggregate_spec_json(outputs)).unwrap()
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
        serde_json::from_value::<RollingSpec>(aggregate_spec_json(json!([missing_ddof]))).is_err()
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
            serde_json::from_value::<RollingSpec>(aggregate_spec_json(json!([missing_operand])))
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

// ------------------------------------------------------------------
// Shared lag/delta kernel
// ------------------------------------------------------------------

fn ts_scalar(value: i64) -> ScalarValue {
    ScalarValue::TimestampMicrosecond(Some(value), Some(Arc::from("UTC")))
}

fn full_row(event_time: i64, symbol: &str, sequence: u64, rest: Vec<ScalarValue>) -> BufferedRow {
    let mut values = vec![
        ts_scalar(event_time),
        ScalarValue::Utf8(Some(symbol.into())),
        ScalarValue::UInt64(Some(sequence)),
    ];
    values.extend(rest);
    while values.len() < 6 {
        values.push(match values.len() {
            3 => ScalarValue::Float64(None),
            4 => ScalarValue::Int64(None),
            _ => ScalarValue::Utf8(None),
        });
    }
    BufferedRow::new(
        vec![Some(KeyValue::String(symbol.into()))],
        vec![KeyValue::Unsigned(sequence)],
        event_time,
        values,
    )
}

fn kernel_schema() -> Schema {
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

fn kernel_spec(outputs: Value) -> RollingSpec {
    let mut document = valid_spec_json();
    document["partition_by"] = json!(["symbol"]);
    document["event_time"] = json!("ts");
    document["sequence_by"] = json!(["sequence"]);
    document["outputs"] = outputs;
    serde_json::from_value(document).unwrap()
}

fn exponential_kernel_spec(outputs: Value) -> RollingSpec {
    let mut document = valid_spec_json();
    document["state_layout_version"] = json!(2);
    document["partition_by"] = json!(["symbol"]);
    document["event_time"] = json!("ts");
    document["sequence_by"] = json!(["sequence"]);
    document["outputs"] = outputs;
    serde_json::from_value(document).unwrap()
}

fn ewma_price(span: u64, min_periods: u64, output: &str) -> Value {
    json!({
        "kind": "ewma",
        "primitive_version": 1,
        "input": "price",
        "output": output,
        "span": span,
        "min_periods": min_periods
    })
}

fn lag_price(periods: u64) -> Value {
    json!({
        "kind": "lag",
        "primitive_version": 1,
        "input": "price",
        "output": "price_lag",
        "periods": periods
    })
}

fn delta_price(periods: u64) -> Value {
    json!({
        "kind": "delta",
        "primitive_version": 1,
        "input": "price",
        "output": "price_delta",
        "periods": periods
    })
}

fn delta_volume(periods: u64) -> Value {
    json!({
        "kind": "delta",
        "primitive_version": 1,
        "input": "volume",
        "output": "volume_delta",
        "periods": periods
    })
}

fn compute(
    spec: &RollingSpec,
    histories: &RollingHistories,
    rows: &[BufferedRow],
) -> Result<ComputedOutputs> {
    let compiled = compile_spec(spec, &kernel_schema())?;
    compute_output_columns(rows, histories, &compiled, "rolling")
}

fn float_column(outputs: &ComputedOutputs, index: usize) -> Vec<Option<f64>> {
    outputs.columns[index]
        .as_any()
        .downcast_ref::<Float64Array>()
        .unwrap()
        .iter()
        .collect()
}

#[test]
fn stable_v2_rebases_large_offset_variance_with_shifted_sums() {
    let mut spec = kernel_spec(json!([ddof_output(
        "variance",
        "price",
        "price_variance",
        64,
        0
    )]));
    spec.numerical_profile = RollingNumericalProfile::StableV2Preview;
    let prices = (0..64)
        .map(|index| 1.0e12 + f64::from(index % 9) / 10.0)
        .collect::<Vec<_>>();
    let rows = prices
        .iter()
        .enumerate()
        .map(|(index, price)| {
            let sequence = u64::try_from(index + 1).unwrap();
            full_row(
                i64::try_from(index + 1).unwrap(),
                "a",
                sequence,
                vec![ScalarValue::Float64(Some(*price))],
            )
        })
        .collect::<Vec<_>>();

    let outputs = compute(&spec, &RollingHistories::default(), &rows).unwrap();
    let actual = float_column(&outputs, 0)[63].unwrap();
    let shift = prices[0];
    let shifted = prices.iter().map(|value| value - shift).collect::<Vec<_>>();
    let sum = shifted.iter().sum::<f64>();
    let expected =
        shifted.iter().map(|value| value * value).sum::<f64>() / 64.0 - (sum / 64.0).powi(2);

    assert_eq!(actual.to_bits(), expected.to_bits());
    assert_eq!(outputs.touched[0].1.transition_count, 64);
}

fn signed_column(outputs: &ComputedOutputs, index: usize) -> Vec<Option<i64>> {
    outputs.columns[index]
        .as_any()
        .downcast_ref::<datafusion::arrow::array::Int64Array>()
        .unwrap()
        .iter()
        .collect()
}

#[test]
fn ewma_uses_first_sample_seeding_and_ignores_null_and_nan() {
    let spec = exponential_kernel_spec(json!([ewma_price(3, 2, "ema")]));
    let rows = vec![
        full_row(1, "a", 1, vec![ScalarValue::Float64(Some(10.0))]),
        full_row(2, "a", 2, vec![ScalarValue::Float64(None)]),
        full_row(3, "a", 3, vec![ScalarValue::Float64(Some(14.0))]),
        full_row(4, "a", 4, vec![ScalarValue::Float64(Some(f64::NAN))]),
        full_row(5, "a", 5, vec![ScalarValue::Float64(Some(18.0))]),
        full_row(6, "a", 6, vec![ScalarValue::Float64(Some(10.0))]),
    ];
    let outputs = compute(&spec, &RollingHistories::default(), &rows).unwrap();
    assert_eq!(
        float_column(&outputs, 0),
        vec![None, None, Some(12.0), Some(12.0), Some(15.0), Some(12.5)]
    );
}

#[test]
fn ewma_shares_state_and_is_segmentation_invariant() {
    let spec = exponential_kernel_spec(json!([
        ewma_price(3, 1, "ema_ready"),
        ewma_price(3, 3, "ema_warm")
    ]));
    let compiled = compile_spec(&spec, &kernel_schema()).unwrap();
    assert_eq!(compiled.window_groups.len(), 1);
    let rows = vec![
        full_row(1, "a", 1, vec![ScalarValue::Float64(Some(10.0))]),
        full_row(2, "a", 2, vec![ScalarValue::Float64(Some(14.0))]),
        full_row(3, "a", 3, vec![ScalarValue::Float64(Some(18.0))]),
        full_row(4, "a", 4, vec![ScalarValue::Float64(Some(10.0))]),
    ];
    let all =
        compute_output_columns(&rows, &RollingHistories::default(), &compiled, "rolling").unwrap();
    let first = compute_output_columns(
        &rows[..2],
        &RollingHistories::default(),
        &compiled,
        "rolling",
    )
    .unwrap();
    let mut histories = RollingHistories::default();
    histories.apply(first.touched.clone());
    let second = compute_output_columns(&rows[2..], &histories, &compiled, "rolling").unwrap();
    for column in 0..2 {
        let segmented = float_column(&first, column)
            .into_iter()
            .chain(float_column(&second, column))
            .collect::<Vec<_>>();
        assert_eq!(segmented, float_column(&all, column));
    }
}

#[test]
fn ewma_layout_two_restores_without_retained_history() {
    let spec = exponential_kernel_spec(json!([ewma_price(3, 1, "ema")]));
    let compiled = compile_spec(&spec, &kernel_schema()).unwrap();
    let rows = vec![
        full_row(1, "a", 1, vec![ScalarValue::Float64(Some(10.0))]),
        full_row(2, "a", 2, vec![ScalarValue::Float64(Some(14.0))]),
    ];
    let first =
        compute_output_columns(&rows, &RollingHistories::default(), &compiled, "rolling").unwrap();
    let mut histories = RollingHistories::default();
    histories.apply(first.touched);
    assert!(
        histories
            .by_entity
            .values()
            .all(|state| state.rows.is_empty())
    );
    let mut legacy = compiled.clone();
    legacy.state_layout_version = ROLLING_EWMA_STATE_LAYOUT_VERSION;
    legacy.state_schema_fingerprint = compiled.legacy_state_schema_fingerprint.clone();
    let bytes = encode_state_segment_legacy(
        &histories,
        &BTreeMap::new(),
        &kernel_schema(),
        &legacy,
        TEST_FINGERPRINT,
        "rolling",
    )
    .unwrap();
    let metadata = RollingSnapshotMetadata {
        late_output: None,
        state_layout_version: 2,
        configuration_hash: compiled.configuration_hash.clone(),
        state_schema_fingerprint: compiled.legacy_state_schema_fingerprint.clone(),
        kernel_fingerprint: None,
        numerical_profile: None,
        epoch: Epoch::new(1).unwrap(),
        pipeline_fingerprint: Some(TEST_FINGERPRINT.into()),
        operator_id: Some("rolling".into()),
        last_input_watermark: None,
        next_output_sequence: 0,
        ended: false,
        metrics: LateMetricDelta::default(),
        segment_inventory: Vec::new(),
    };
    let restored = decode_state_segment(&bytes, &kernel_schema(), &compiled, &metadata).unwrap();
    let continuation = vec![full_row(3, "a", 3, vec![ScalarValue::Float64(Some(18.0))])];
    let expected = compute_output_columns(&continuation, &histories, &compiled, "rolling").unwrap();
    let output_schema = Arc::new(output_schema(&kernel_schema(), &compiled.outputs));
    let (actual, state, _) = build_typed_stream_output(
        &continuation,
        &restored.histories,
        None,
        &compiled,
        &output_schema,
        "rolling",
        None,
    )
    .unwrap()
    .unwrap();
    let actual = actual
        .column(kernel_schema().fields().len())
        .as_any()
        .downcast_ref::<Float64Array>()
        .unwrap()
        .iter()
        .collect::<Vec<_>>();
    assert_eq!(actual, float_column(&expected, 0));
    assert_eq!(actual, vec![Some(15.0)]);
    assert!(state.is_some());
}

#[test]
fn layout_one_history_bootstraps_the_typed_transition() {
    let spec = kernel_spec(json!([aggregate_output("mean", "price", "price_mean", 3)]));
    let compiled = compile_spec(&spec, &kernel_schema()).unwrap();
    let first_rows = vec![
        full_row(1, "a", 1, vec![ScalarValue::Float64(Some(1.0))]),
        full_row(2, "a", 2, vec![ScalarValue::Float64(Some(3.0))]),
    ];
    let first = compute_output_columns(
        &first_rows,
        &RollingHistories::default(),
        &compiled,
        "rolling",
    )
    .unwrap();
    let mut histories = RollingHistories::default();
    histories.apply(first.touched);
    let mut legacy = compiled.clone();
    legacy.state_layout_version = ROLLING_STATE_LAYOUT_VERSION;
    legacy.state_schema_fingerprint = compiled.legacy_state_schema_fingerprint.clone();
    let bytes = encode_state_segment_legacy(
        &histories,
        &BTreeMap::new(),
        &kernel_schema(),
        &legacy,
        TEST_FINGERPRINT,
        "rolling",
    )
    .unwrap();
    let metadata = RollingSnapshotMetadata {
        late_output: None,
        state_layout_version: ROLLING_STATE_LAYOUT_VERSION,
        configuration_hash: compiled.configuration_hash.clone(),
        state_schema_fingerprint: compiled.legacy_state_schema_fingerprint.clone(),
        kernel_fingerprint: None,
        numerical_profile: None,
        epoch: Epoch::new(1).unwrap(),
        pipeline_fingerprint: Some(TEST_FINGERPRINT.into()),
        operator_id: Some("rolling".into()),
        last_input_watermark: None,
        next_output_sequence: 0,
        ended: false,
        metrics: LateMetricDelta::default(),
        segment_inventory: Vec::new(),
    };
    let restored = decode_state_segment(&bytes, &kernel_schema(), &compiled, &metadata).unwrap();
    let continuation = vec![full_row(3, "a", 3, vec![ScalarValue::Float64(Some(5.0))])];
    let expected = compute_output_columns(&continuation, &histories, &compiled, "rolling").unwrap();
    let output_schema = Arc::new(output_schema(&kernel_schema(), &compiled.outputs));
    let (actual, state, _) = build_typed_stream_output(
        &continuation,
        &restored.histories,
        None,
        &compiled,
        &output_schema,
        "rolling",
        None,
    )
    .unwrap()
    .unwrap();
    let actual = actual
        .column(kernel_schema().fields().len())
        .as_any()
        .downcast_ref::<Float64Array>()
        .unwrap()
        .iter()
        .collect::<Vec<_>>();
    assert_eq!(actual, float_column(&expected, 0));
    assert_eq!(actual, vec![Some(3.0)]);
    assert!(state.is_some());
}

#[test]
fn ewma_restore_rejects_noncanonical_accumulator_rows() {
    let spec = exponential_kernel_spec(json!([ewma_price(3, 1, "ema")]));
    let compiled = compile_spec(&spec, &kernel_schema()).unwrap();
    let entity = vec![Some(KeyValue::String("a".into()))];
    let values = ewma_entity_values(&entity, &kernel_schema(), &compiled).unwrap();

    let mut decoded = DecodedRollingState::default();
    let mut previous = None;
    let error = decode_ewma_state_row(
        None,
        &values,
        Some((0, 0, 10.0)),
        &mut decoded,
        &compiled,
        &mut previous,
    )
    .unwrap_err();
    assert!(matches!(error, CalcFlowError::Format { .. }));

    let mut populated = values.clone();
    populated[3] = ScalarValue::Float64(Some(10.0));
    let error = decode_ewma_state_row(
        None,
        &populated,
        Some((0, 1, 10.0)),
        &mut DecodedRollingState::default(),
        &compiled,
        &mut None,
    )
    .unwrap_err();
    assert!(matches!(error, CalcFlowError::Format { .. }));

    let error = decode_ewma_state_row(
        Some(0),
        &values,
        Some((0, 1, 10.0)),
        &mut DecodedRollingState::default(),
        &compiled,
        &mut None,
    )
    .unwrap_err();
    assert!(matches!(error, CalcFlowError::Format { .. }));
}

#[test]
fn lag_references_the_previous_row_within_each_entity() {
    let spec = kernel_spec(json!([lag_price(1)]));
    let rows = vec![
        full_row(1, "a", 1, vec![ScalarValue::Float64(Some(1.0))]),
        full_row(1, "b", 1, vec![ScalarValue::Float64(Some(10.0))]),
        full_row(2, "a", 2, vec![ScalarValue::Float64(Some(2.0))]),
        full_row(2, "b", 2, vec![ScalarValue::Float64(Some(20.0))]),
        full_row(3, "a", 3, vec![ScalarValue::Float64(Some(3.0))]),
    ];
    let outputs = compute(&spec, &RollingHistories::default(), &rows).unwrap();
    assert_eq!(
        float_column(&outputs, 0),
        vec![None, None, Some(1.0), Some(10.0), Some(2.0)]
    );
}

#[test]
fn lag_periods_span_the_shared_history_across_segmentation() {
    let spec = kernel_spec(json!([lag_price(2)]));
    let mut histories = RollingHistories::default();
    let first = vec![
        full_row(1, "a", 1, vec![ScalarValue::Float64(Some(1.0))]),
        full_row(2, "a", 2, vec![ScalarValue::Float64(Some(2.0))]),
    ];
    let outputs = compute(&spec, &histories, &first).unwrap();
    assert_eq!(float_column(&outputs, 0), vec![None, None]);
    histories.apply(outputs.touched);

    let second = vec![
        full_row(3, "a", 3, vec![ScalarValue::Float64(Some(3.0))]),
        full_row(4, "a", 4, vec![ScalarValue::Float64(Some(4.0))]),
        full_row(5, "a", 5, vec![ScalarValue::Float64(Some(5.0))]),
    ];
    let outputs = compute(&spec, &histories, &second).unwrap();
    assert_eq!(
        float_column(&outputs, 0),
        vec![Some(1.0), Some(2.0), Some(3.0)]
    );
}

#[test]
fn lag_preserves_null_and_nan_at_the_referenced_position() {
    let spec = kernel_spec(json!([lag_price(1)]));
    let rows = vec![
        full_row(1, "a", 1, vec![ScalarValue::Float64(None)]),
        full_row(2, "a", 2, vec![ScalarValue::Float64(Some(f64::NAN))]),
        full_row(3, "a", 3, vec![ScalarValue::Float64(Some(3.0))]),
        full_row(4, "a", 4, vec![ScalarValue::Float64(Some(4.0))]),
    ];
    let outputs = compute(&spec, &RollingHistories::default(), &rows).unwrap();
    let values = outputs.columns[0]
        .as_any()
        .downcast_ref::<Float64Array>()
        .unwrap();
    assert!(values.is_null(0));
    assert!(values.is_null(1));
    assert!(values.value(2).is_nan());
    assert_eq!(values.value(3).to_bits(), 3.0_f64.to_bits());
}

#[test]
fn lag_works_for_non_numeric_columns() {
    let spec = kernel_spec(json!([{
        "kind": "lag",
        "primitive_version": 1,
        "input": "label",
        "output": "label_lag",
        "periods": 1
    }]));
    let rows = vec![
        full_row(
            1,
            "a",
            1,
            vec![
                ScalarValue::Float64(None),
                ScalarValue::Int64(None),
                ScalarValue::Utf8(Some("x".into())),
            ],
        ),
        full_row(
            2,
            "a",
            2,
            vec![
                ScalarValue::Float64(None),
                ScalarValue::Int64(None),
                ScalarValue::Utf8(Some("y".into())),
            ],
        ),
    ];
    let outputs = compute(&spec, &RollingHistories::default(), &rows).unwrap();
    let values = outputs.columns[0]
        .as_any()
        .downcast_ref::<datafusion::arrow::array::StringArray>()
        .unwrap();
    assert!(values.is_null(0));
    assert_eq!(values.value(1), "x");
}

#[test]
fn delta_subtracts_the_referenced_value_with_checked_integer_math() {
    let spec = kernel_spec(json!([delta_volume(1)]));
    let rows = vec![
        full_row(
            1,
            "a",
            1,
            vec![ScalarValue::Float64(None), ScalarValue::Int64(Some(7))],
        ),
        full_row(
            2,
            "a",
            2,
            vec![ScalarValue::Float64(None), ScalarValue::Int64(Some(10))],
        ),
        full_row(
            3,
            "a",
            3,
            vec![ScalarValue::Float64(None), ScalarValue::Int64(Some(4))],
        ),
    ];
    let outputs = compute(&spec, &RollingHistories::default(), &rows).unwrap();
    assert_eq!(signed_column(&outputs, 0), vec![None, Some(3), Some(-6)]);
}

#[test]
fn delta_integer_overflow_is_a_data_error() {
    let spec = kernel_spec(json!([delta_volume(1)]));
    let rows = vec![
        full_row(
            1,
            "a",
            1,
            vec![ScalarValue::Float64(None), ScalarValue::Int64(Some(-1))],
        ),
        full_row(
            2,
            "a",
            2,
            vec![
                ScalarValue::Float64(None),
                ScalarValue::Int64(Some(i64::MAX)),
            ],
        ),
    ];
    let error = compute(&spec, &RollingHistories::default(), &rows).unwrap_err();
    assert!(
        matches!(error, CalcFlowError::Operator { ref node_id, .. } if node_id == "rolling"),
        "unexpected error: {error}"
    );
}

#[test]
fn delta_preserves_null_and_propagates_nan() {
    let spec = kernel_spec(json!([delta_price(1)]));
    let rows = vec![
        full_row(1, "a", 1, vec![ScalarValue::Float64(None)]),
        full_row(2, "a", 2, vec![ScalarValue::Float64(Some(1.5))]),
        full_row(3, "a", 3, vec![ScalarValue::Float64(Some(f64::NAN))]),
        full_row(4, "a", 4, vec![ScalarValue::Float64(Some(2.5))]),
        full_row(5, "a", 5, vec![ScalarValue::Float64(Some(f64::INFINITY))]),
        full_row(6, "a", 6, vec![ScalarValue::Float64(Some(f64::INFINITY))]),
    ];
    let outputs = compute(&spec, &RollingHistories::default(), &rows).unwrap();
    let values = outputs.columns[0]
        .as_any()
        .downcast_ref::<Float64Array>()
        .unwrap();
    assert!(values.is_null(0));
    assert!(values.is_null(1));
    assert!(values.value(2).is_nan());
    assert!(values.value(3).is_nan());
    assert_eq!(values.value(4).to_bits(), f64::INFINITY.to_bits());
    assert!(values.value(5).is_nan());
}

#[test]
fn delta_unsigned_underflow_is_a_data_error() {
    let spec = kernel_spec(json!([{
        "kind": "delta",
        "primitive_version": 1,
        "input": "sequence",
        "output": "sequence_delta",
        "periods": 1
    }]));
    let rows = vec![full_row(1, "a", 10, vec![]), full_row(2, "a", 3, vec![])];
    let error = compute(&spec, &RollingHistories::default(), &rows).unwrap_err();
    assert!(
        matches!(error, CalcFlowError::Operator { .. }),
        "unexpected error: {error}"
    );
}

#[test]
fn history_is_truncated_to_the_maximum_declared_periods() {
    let spec = kernel_spec(json!([lag_price(2), lag_price(1)]));
    let mut histories = RollingHistories::default();
    for batch in 0..3_u32 {
        let rows = (0..4_u32)
            .map(|index| {
                let sequence = batch * 4 + index + 1;
                full_row(
                    i64::from(sequence),
                    "a",
                    u64::from(sequence),
                    vec![ScalarValue::Float64(Some(f64::from(sequence)))],
                )
            })
            .collect::<Vec<_>>();
        let outputs = compute(&spec, &histories, &rows).unwrap();
        histories.apply(outputs.touched);
    }
    for state in histories.by_entity.values() {
        assert!(state.rows.len() <= 2);
    }
}

#[test]
fn failed_delta_leaves_histories_untouched() {
    let spec = kernel_spec(json!([delta_volume(1)]));
    let histories = RollingHistories::default();
    let rows = vec![
        full_row(
            1,
            "a",
            1,
            vec![ScalarValue::Float64(None), ScalarValue::Int64(Some(-1))],
        ),
        full_row(
            2,
            "a",
            2,
            vec![
                ScalarValue::Float64(None),
                ScalarValue::Int64(Some(i64::MAX)),
            ],
        ),
    ];
    assert!(compute(&spec, &histories, &rows).is_err());
    assert!(histories.by_entity.is_empty());
}

// ------------------------------------------------------------------
// Shared aggregate kernel (SCE-07)
// ------------------------------------------------------------------

fn price_rows(prices: &[Option<f64>]) -> Vec<BufferedRow> {
    prices
        .iter()
        .enumerate()
        .map(|(index, price)| {
            let sequence = u64::try_from(index + 1).unwrap();
            full_row(
                i64::try_from(index + 1).unwrap(),
                "a",
                sequence,
                vec![ScalarValue::Float64(*price)],
            )
        })
        .collect()
}

fn unsigned_column(outputs: &ComputedOutputs, index: usize) -> Vec<Option<u64>> {
    outputs.columns[index]
        .as_any()
        .downcast_ref::<UInt64Array>()
        .unwrap()
        .iter()
        .collect()
}

fn float64_fast_record(rows: &[(i64, &str, u64, Option<f64>)]) -> RecordBatch {
    use datafusion::arrow::array::{Int64Array, StringArray, TimestampMicrosecondArray};

    let timestamps = rows
        .iter()
        .map(|(event_time, ..)| Some(*event_time))
        .collect::<TimestampMicrosecondArray>()
        .with_timezone("UTC");
    let symbols = StringArray::from_iter_values(rows.iter().map(|(_, symbol, ..)| *symbol));
    let sequences = UInt64Array::from_iter_values(rows.iter().map(|(_, _, sequence, _)| *sequence));
    let prices = rows
        .iter()
        .map(|(_, _, _, price)| *price)
        .collect::<Float64Array>();
    let volume = Int64Array::new_null(rows.len());
    let label = StringArray::new_null(rows.len());
    RecordBatch::try_new(
        Arc::new(kernel_schema()),
        vec![
            Arc::new(timestamps),
            Arc::new(symbols),
            Arc::new(sequences),
            Arc::new(prices),
            Arc::new(volume),
            Arc::new(label),
        ],
    )
    .unwrap()
}

fn float64_pair_schema() -> Schema {
    with_field(
        &kernel_schema(),
        4,
        &Field::new("volume", DataType::Float64, true),
    )
}

type Float64PairRow<'a> = (i64, &'a str, u64, Option<f64>, Option<f64>);

fn float64_pair_record(rows: &[Float64PairRow<'_>]) -> RecordBatch {
    use datafusion::arrow::array::{StringArray, TimestampMicrosecondArray};

    let timestamps = rows
        .iter()
        .map(|(event_time, ..)| Some(*event_time))
        .collect::<TimestampMicrosecondArray>()
        .with_timezone("UTC");
    RecordBatch::try_new(
        Arc::new(float64_pair_schema()),
        vec![
            Arc::new(timestamps),
            Arc::new(StringArray::from_iter_values(
                rows.iter().map(|(_, symbol, ..)| *symbol),
            )),
            Arc::new(UInt64Array::from_iter_values(
                rows.iter().map(|(_, _, sequence, ..)| *sequence),
            )),
            Arc::new(
                rows.iter()
                    .map(|(_, _, _, price, _)| *price)
                    .collect::<Float64Array>(),
            ),
            Arc::new(
                rows.iter()
                    .map(|(_, _, _, _, volume)| *volume)
                    .collect::<Float64Array>(),
            ),
            Arc::new(StringArray::new_null(rows.len())),
        ],
    )
    .unwrap()
}

fn int64_fast_record(values: &[Option<i64>]) -> RecordBatch {
    use datafusion::arrow::array::{Int64Array, StringArray, TimestampMicrosecondArray};

    let len = values.len();
    RecordBatch::try_new(
        Arc::new(kernel_schema()),
        vec![
            Arc::new(
                (1..=i64::try_from(len).unwrap())
                    .map(Some)
                    .collect::<TimestampMicrosecondArray>()
                    .with_timezone("UTC"),
            ),
            Arc::new(StringArray::from_iter_values(std::iter::repeat_n("a", len))),
            Arc::new(UInt64Array::from_iter_values(
                1..=u64::try_from(len).unwrap(),
            )),
            Arc::new(Float64Array::new_null(len)),
            Arc::new(values.iter().copied().collect::<Int64Array>()),
            Arc::new(StringArray::new_null(len)),
        ],
    )
    .unwrap()
}

fn primitive_volume_record(values: ArrayRef) -> (Schema, RecordBatch) {
    use datafusion::arrow::array::{StringArray, TimestampMicrosecondArray};

    let len = values.len();
    let schema = with_field(
        &kernel_schema(),
        4,
        &Field::new("volume", values.data_type().clone(), true),
    );
    let batch = RecordBatch::try_new(
        Arc::new(schema.clone()),
        vec![
            Arc::new(
                (1..=i64::try_from(len).unwrap())
                    .map(Some)
                    .collect::<TimestampMicrosecondArray>()
                    .with_timezone("UTC"),
            ),
            Arc::new(StringArray::from_iter_values(std::iter::repeat_n("a", len))),
            Arc::new(UInt64Array::from_iter_values(
                1..=u64::try_from(len).unwrap(),
            )),
            Arc::new(Float64Array::new_null(len)),
            values,
            Arc::new(StringArray::new_null(len)),
        ],
    )
    .unwrap();
    (schema, batch)
}

fn assert_typed_matches_general(spec: &RollingSpec, schema: &Schema, input: &RecordBatch) {
    let compiled = compile_spec(spec, schema).unwrap();
    let fast = compiled
        .kernel_plan
        .open_and_fill(input, "rolling")
        .unwrap()
        .unwrap();
    let rows = (0..input.num_rows())
        .map(|row_index| read_buffered_row(input, row_index, &compiled, "rolling").unwrap())
        .collect::<Vec<_>>();
    let general =
        compute_output_columns(&rows, &RollingHistories::default(), &compiled, "rolling").unwrap();

    assert_eq!(fast.columns.len(), general.columns.len());
    for (fast, general) in fast.columns.iter().zip(&general.columns) {
        assert_eq!(fast.to_data(), general.to_data());
    }
}

#[test]
fn ordered_float64_plan_matches_general_aggregate_kernel_without_row_materialization() {
    let spec = kernel_spec(json!([
        aggregate_output("count", "price", "price_count", 2),
        aggregate_output("sum", "price", "price_sum", 2),
        aggregate_output("mean", "price", "price_mean", 2),
        ddof_output("variance", "price", "price_var", 2, 1),
        ddof_output("stddev", "price", "price_std", 2, 0),
    ]));
    let compiled = compile_spec(&spec, &kernel_schema()).unwrap();
    assert_eq!(
        compiled.kernel_plan.selection(),
        KernelSelection::OrderedPrimitive
    );
    let input = float64_fast_record(&[
        (1, "a", 1, Some(1.0)),
        (1, "b", 1, Some(10.0)),
        (2, "a", 2, None),
        (2, "b", 2, Some(20.0)),
        (3, "a", 3, Some(f64::NAN)),
        (3, "b", 3, Some(30.0)),
        (4, "a", 4, Some(4.0)),
    ]);
    let fast = compiled
        .kernel_plan
        .open_and_fill(&input, "rolling")
        .unwrap()
        .unwrap();
    let rows = (0..input.num_rows())
        .map(|row_index| read_buffered_row(&input, row_index, &compiled, "rolling").unwrap())
        .collect::<Vec<_>>();
    let general =
        compute_output_columns(&rows, &RollingHistories::default(), &compiled, "rolling").unwrap();

    assert_eq!(fast.columns.len(), general.columns.len());
    for (fast, general) in fast.columns.iter().zip(&general.columns) {
        assert_eq!(fast.as_ref(), general.as_ref());
    }
    assert_eq!(fast.metrics.order_proof_rows, input.num_rows());
    assert_eq!(fast.metrics.entities, 2);
    assert_eq!(fast.metrics.scalar_value_conversions, 0);
    assert_eq!(fast.metrics.sort_count, 0);
}

#[test]
fn typed_update_and_fill_matches_one_shot_fill_across_micro_batches() {
    use datafusion::arrow::compute::concat;

    let spec = kernel_spec(json!([
        aggregate_output("count", "price", "price_count", 3),
        aggregate_output("mean", "price", "price_mean", 3),
        ddof_output("variance", "price", "price_var", 3, 1),
    ]));
    let compiled = compile_spec(&spec, &kernel_schema()).unwrap();
    let input = float64_fast_record(&[
        (1, "a", 1, Some(1.0)),
        (1, "b", 1, Some(10.0)),
        (2, "a", 2, None),
        (2, "b", 2, Some(20.0)),
        (3, "a", 3, Some(3.0)),
        (3, "b", 3, Some(30.0)),
    ]);
    let one_shot = compiled
        .kernel_plan
        .open_and_fill(&input, "rolling")
        .unwrap()
        .unwrap();
    let first = compiled
        .kernel_plan
        .open_and_fill(&input.slice(0, 4), "rolling")
        .unwrap()
        .unwrap();
    let second = compiled
        .kernel_plan
        .update_and_fill(&first.state, &input.slice(4, 2), "rolling")
        .unwrap()
        .unwrap();

    for ((expected, prefix), suffix) in one_shot
        .columns
        .iter()
        .zip(&first.columns)
        .zip(&second.columns)
    {
        let combined = concat(&[prefix.as_ref(), suffix.as_ref()]).unwrap();
        assert_eq!(combined.as_ref(), expected.as_ref());
    }
    assert_eq!(second.metrics.entities, 2);
    assert_eq!(second.metrics.scalar_value_conversions, 0);
}

fn single_entity_retained_prices(operator: &RollingOperator) -> &Float64Array {
    operator
        .state
        .histories
        .by_entity
        .values()
        .next()
        .unwrap()
        .columnar
        .records
        .front()
        .unwrap()
        .column(3)
        .as_any()
        .downcast_ref::<Float64Array>()
        .unwrap()
}

#[tokio::test]
async fn ordered_stream_finalization_reuses_input_columns_and_keeps_incremental_state() {
    use crate::{CancellationToken, StreamJobContext};

    let spec = kernel_spec(json!([aggregate_output("mean", "price", "mean", 2)]));
    let mut operator = RollingOperator::new("rolling", Arc::new(kernel_schema()), spec).unwrap();
    let job = StreamJobContext::new(
        7,
        "fingerprint",
        JsonMap::new(),
        None,
        CancellationToken::new(),
    );
    let mut collector = crate::EdgeCollector::new(operator.output_ports().to_vec());
    let first = float64_fast_record(&[(1, "a", 1, Some(1.0)), (2, "a", 2, Some(3.0))]);
    let context = StreamOperatorContext::new(&job, "rolling", None);
    operator
        .process_data(
            "input",
            Batch::table(vec![first.clone()], BatchMetadata::default()).unwrap(),
            &context,
            &mut collector,
        )
        .await
        .unwrap();
    assert!(collector.drain("output").is_empty());
    operator
        .on_watermark(EventTime::from_micros(2), &context, &mut collector)
        .await
        .unwrap();
    let emitted = collector.drain("output");
    let record = &emitted[0]
        .as_data()
        .unwrap()
        .table_payload()
        .unwrap()
        .batches()[0];
    for column in 0..first.num_columns() {
        assert_eq!(
            record.column(column).to_data().buffers()[0].as_ptr(),
            first.column(column).to_data().buffers()[0].as_ptr(),
        );
    }
    assert_eq!(
        record
            .column(first.num_columns())
            .as_any()
            .downcast_ref::<Float64Array>()
            .unwrap()
            .values()
            .as_ref(),
        &[1.0, 2.0],
    );
    let second = float64_fast_record(&[(3, "a", 3, Some(5.0))]);
    let retained_row = single_entity_retained_prices(&operator).values()[1..].as_ptr();
    let context = StreamOperatorContext::new(&job, "rolling", Some(EventTime::from_micros(2)));
    operator
        .process_data(
            "input",
            Batch::table(vec![second], BatchMetadata::default()).unwrap(),
            &context,
            &mut collector,
        )
        .await
        .unwrap();
    operator
        .on_watermark(EventTime::from_micros(3), &context, &mut collector)
        .await
        .unwrap();
    let emitted = collector.drain("output");
    let record = &emitted[0]
        .as_data()
        .unwrap()
        .table_payload()
        .unwrap()
        .batches()[0];
    assert_eq!(
        record
            .column(first.num_columns())
            .as_any()
            .downcast_ref::<Float64Array>()
            .unwrap()
            .values()
            .as_ref(),
        &[4.0]
    );
    assert_eq!(
        single_entity_retained_prices(&operator).values().as_ptr(),
        retained_row,
        "an unchanged retained row must not be cloned on append"
    );
}

#[test]
fn warm_stream_finalization_allocations_depend_on_touched_entities() {
    use crate::{CancellationToken, StreamJobContext};

    fn allocations(entities: usize) -> u64 {
        let spec = kernel_spec(json!([aggregate_output("mean", "price", "mean", 20)]));
        let mut operator =
            RollingOperator::new("rolling", Arc::new(kernel_schema()), spec).unwrap();
        let job = StreamJobContext::new(
            7,
            TEST_FINGERPRINT,
            JsonMap::new(),
            None,
            CancellationToken::new(),
        );
        let symbols = (0..entities)
            .map(|index| format!("e{index:06}"))
            .collect::<Vec<_>>();
        let rows = symbols
            .iter()
            .map(|symbol| (1, symbol.as_str(), 1, Some(1.0)))
            .collect::<Vec<_>>();
        let input = float64_fast_record(&rows);
        let context = StreamOperatorContext::new(&job, "rolling", None);
        let mut collector = crate::EdgeCollector::new(operator.output_ports().to_vec());
        futures::executor::block_on(operator.process_data(
            "input",
            Batch::table(vec![input], BatchMetadata::default()).unwrap(),
            &context,
            &mut collector,
        ))
        .unwrap();
        futures::executor::block_on(operator.on_watermark(
            EventTime::from_micros(1),
            &context,
            &mut collector,
        ))
        .unwrap();
        collector.drain("output");
        let input = float64_fast_record(&[(2, symbols[0].as_str(), 2, Some(3.0))]);
        let context = StreamOperatorContext::new(&job, "rolling", Some(EventTime::from_micros(1)));
        futures::executor::block_on(operator.process_data(
            "input",
            Batch::table(vec![input], BatchMetadata::default()).unwrap(),
            &context,
            &mut collector,
        ))
        .unwrap();
        allocation_counter::measure(|| {
            futures::executor::block_on(operator.on_watermark(
                EventTime::from_micros(2),
                &context,
                &mut collector,
            ))
            .unwrap();
        })
        .count_total
    }
    let small = allocations(4);
    let large = allocations(4096);
    assert!(
        large <= small + 100,
        "one touched entity: small={small}, large={large}"
    );
}

#[test]
fn ordered_tail_history_allocations_are_columnar() {
    use crate::{CancellationToken, StreamJobContext};

    fn allocations(window: u64) -> u64 {
        let spec = kernel_spec(json!([aggregate_output("mean", "price", "mean", window)]));
        let mut operator =
            RollingOperator::new("rolling", Arc::new(kernel_schema()), spec).unwrap();
        let job = StreamJobContext::new(
            7,
            TEST_FINGERPRINT,
            JsonMap::new(),
            None,
            CancellationToken::new(),
        );
        let context = StreamOperatorContext::new(&job, "rolling", None);
        let rows = (1..=8192_u64)
            .map(|index| (i64::try_from(index).unwrap(), "a", index, Some(1.0)))
            .collect::<Vec<_>>();
        let input = float64_fast_record(&rows);
        let mut collector = crate::EdgeCollector::new(operator.output_ports().to_vec());
        futures::executor::block_on(operator.process_data(
            "input",
            Batch::table(vec![input], BatchMetadata::default()).unwrap(),
            &context,
            &mut collector,
        ))
        .unwrap();
        allocation_counter::measure(|| {
            futures::executor::block_on(operator.on_watermark(
                EventTime::from_micros(8192),
                &context,
                &mut collector,
            ))
            .unwrap();
        })
        .count_total
    }

    let small = allocations(32);
    let large = allocations(4096);
    println!("retained history allocations: tail32={small}, tail4096={large}");
    assert!(
        large <= small + 100,
        "one entity, 8192 input rows: tail32={small}, tail4096={large}"
    );
}

#[test]
fn sparse_fixed_width_appends_amortize_retained_buffer_copies() {
    use crate::{CancellationToken, StreamJobContext};

    let first = fixed_width_history_record(1, 1024);
    let spec = kernel_spec(json!([aggregate_output("mean", "price", "mean", 1024)]));
    let mut operator = RollingOperator::new("rolling", first.schema(), spec).unwrap();
    let job = StreamJobContext::new(
        7,
        TEST_FINGERPRINT,
        JsonMap::new(),
        None,
        CancellationToken::new(),
    );
    let mut collector = crate::EdgeCollector::new(operator.output_ports().to_vec());
    let context = StreamOperatorContext::new(&job, "rolling", None);
    futures::executor::block_on(operator.process_data(
        "input",
        Batch::table(vec![first], BatchMetadata::default()).unwrap(),
        &context,
        &mut collector,
    ))
    .unwrap();
    futures::executor::block_on(operator.on_watermark(
        EventTime::from_micros(1024),
        &context,
        &mut collector,
    ))
    .unwrap();
    collector.drain("output");
    let initial = single_entity_retained_prices(&operator).to_data();
    let initial_bytes = single_entity_retained_prices(&operator).get_buffer_memory_size();
    let mut backing = initial.buffers()[0].data_ptr();
    drop(initial);
    let mut copies = 0;
    let measured = allocation_counter::measure(|| {
        for time in 1025..=1792 {
            let context = StreamOperatorContext::new(
                &job,
                "rolling",
                Some(EventTime::from_micros(i64::from(time - 1))),
            );
            let input = Batch::table(
                vec![fixed_width_history_record(time, 1)],
                BatchMetadata::default(),
            )
            .unwrap();
            futures::executor::block_on(operator.process_data(
                "input",
                input,
                &context,
                &mut collector,
            ))
            .unwrap();
            futures::executor::block_on(operator.on_watermark(
                EventTime::from_micros(i64::from(time)),
                &context,
                &mut collector,
            ))
            .unwrap();
            collector.drain("output");
            let current = single_entity_retained_prices(&operator).to_data();
            copies += usize::from(current.buffers()[0].data_ptr() != backing);
            backing = current.buffers()[0].data_ptr();
        }
    });
    assert!(
        copies <= 8,
        "768 sparse appends copied the retained prefix {copies} times"
    );
    let history = operator.state.histories.by_entity.values().next().unwrap();
    assert_eq!(
        history
            .columnar
            .records
            .iter()
            .map(RecordBatch::num_rows)
            .sum::<usize>(),
        1024
    );
    assert_eq!(history.transition_count, 1792);
    assert!(single_entity_retained_prices(&operator).get_buffer_memory_size() <= initial_bytes / 2);
    println!(
        "768 sparse appends: prefix copies={copies}, allocations={}, allocated bytes={}",
        measured.count_total, measured.bytes_total
    );
}

fn fixed_width_history_record(first: u32, rows: u32) -> RecordBatch {
    use datafusion::arrow::array::TimestampMicrosecondArray;

    let times = (first..first + rows)
        .map(|time| Some(i64::from(time)))
        .collect::<TimestampMicrosecondArray>()
        .with_timezone("UTC");
    let symbols = UInt64Array::from_iter_values((0..rows).map(|_| 1));
    let sequences = UInt64Array::from_iter_values((first..first + rows).map(u64::from));
    let prices = Float64Array::from_iter_values((0..rows).map(|_| 1.0));
    let schema = Schema::new(vec![
        Field::new("ts", times.data_type().clone(), false),
        Field::new("symbol", DataType::UInt64, false),
        Field::new("sequence", DataType::UInt64, false),
        Field::new("price", DataType::Float64, false),
    ]);
    RecordBatch::try_new(
        Arc::new(schema),
        vec![
            Arc::new(times),
            Arc::new(symbols),
            Arc::new(sequences),
            Arc::new(prices),
        ],
    )
    .unwrap()
}

#[tokio::test]
async fn native_rolling_observations_report_work_without_materializing_histories() {
    use crate::operator::rolling_metrics::{RollingCallback, RollingMetricsStore};
    use crate::{CancellationToken, StreamJobContext};

    let spec = kernel_spec(json!([aggregate_output("mean", "price", "mean", 20)]));
    let mut operator = RollingOperator::new("rolling", Arc::new(kernel_schema()), spec).unwrap();
    let cancellation = CancellationToken::new();
    let job = StreamJobContext::new(
        7,
        TEST_FINGERPRINT,
        JsonMap::new(),
        None,
        cancellation.clone(),
    );
    let store = RollingMetricsStore::default();
    let mut collector = crate::EdgeCollector::new(operator.output_ports().to_vec());
    let mut watermark = None;
    for rows in [
        vec![(1, "a", 1, Some(1.0)), (2, "b", 2, Some(2.0))],
        vec![(3, "a", 3, Some(3.0)), (4, "b", 4, Some(4.0))],
    ] {
        let input = float64_fast_record(&rows);
        let callback = store.begin(RollingCallback::Data, cancellation.clone());
        let context = StreamOperatorContext::new(&job, "rolling", watermark)
            .with_rolling_metrics(callback.recorder());
        let result = operator
            .process_data(
                "input",
                Batch::table(vec![input], BatchMetadata::default()).unwrap(),
                &context,
                &mut collector,
            )
            .await;
        callback.complete(&result);
        result.unwrap();
        let callback = store.begin(RollingCallback::Watermark, cancellation.clone());
        let context = StreamOperatorContext::new(&job, "rolling", watermark)
            .with_rolling_metrics(callback.recorder());
        let next = EventTime::from_micros(rows.last().unwrap().0);
        let result = operator.on_watermark(next, &context, &mut collector).await;
        callback.complete(&result);
        result.unwrap();
        watermark = Some(next);
        collector.drain("output");
    }
    let observations = store.snapshot();
    assert_eq!(observations.data.order_proof_rows, 4);
    assert_eq!(observations.watermark.resolved_rows, 4);
    assert_eq!(observations.watermark.touched_entities, 4);
    assert_eq!(observations.watermark.copied_entities, 2);
    assert_eq!(observations.watermark.numeric_rows, 4);
    assert_eq!(observations.watermark.history_rows_materialized, 0);
    assert_eq!(observations.watermark.scalar_value_conversions, 4);
    assert_eq!(observations.watermark.output_rows_prepared, 4);
    assert_eq!(observations.watermark.output_chunks_prepared, 2);
}

#[tokio::test]
async fn late_rejection_does_not_report_unperformed_order_proof() {
    use crate::operator::rolling_metrics::{RollingCallback, RollingMetricsStore};
    use crate::{CancellationToken, StreamJobContext};

    let spec = kernel_spec(json!([aggregate_output("mean", "price", "mean", 2)]));
    let mut operator = RollingOperator::new("rolling", Arc::new(kernel_schema()), spec).unwrap();
    let cancellation = CancellationToken::new();
    let job = StreamJobContext::new(
        7,
        TEST_FINGERPRINT,
        JsonMap::new(),
        None,
        cancellation.clone(),
    );
    let store = RollingMetricsStore::default();
    let callback = store.begin(RollingCallback::Data, cancellation);
    let context = StreamOperatorContext::new(&job, "rolling", Some(EventTime::from_micros(2)))
        .with_rolling_metrics(callback.recorder());
    let mut collector = crate::EdgeCollector::new(operator.output_ports().to_vec());
    let input = float64_fast_record(&[(1, "a", 1, Some(1.0))]);
    let result = operator
        .process_data(
            "input",
            Batch::table(vec![input], BatchMetadata::default()).unwrap(),
            &context,
            &mut collector,
        )
        .await;
    callback.complete(&result);
    assert!(result.unwrap_err().to_string().contains("late_row"));
    assert!(operator.state.buffer.is_empty());
    let metrics = store.snapshot().data;
    assert_eq!(metrics.failed, 1);
    assert_eq!(
        metrics.order_proof_rows, 0,
        "lateness rejects the row before ordered encoding or duplicate lookup"
    );
}

#[tokio::test]
async fn all_late_drop_skips_duplicate_checks_and_reports_no_order_proof() {
    assert_drop_late_observations(&[1, 1, 2], &[]).await;
}

#[tokio::test]
async fn mixed_late_drop_counts_only_nonlate_duplicate_checks() {
    assert_drop_late_observations(&[3, 1, 1, 2, 4], &[3, 4]).await;
}

async fn assert_drop_late_observations(times: &[i64], accepted_times: &[i64]) {
    let (mut operator, job, store, mut collector) = late_observation_fixture(true);
    observe_late_data(&mut operator, &job, &store, times, &mut collector)
        .await
        .unwrap();
    let metrics = store.snapshot().data;
    assert_eq!(metrics.succeeded, 1);
    assert_eq!(metrics.failed, 0);
    assert_eq!(
        metrics.order_proof_rows,
        u64::try_from(accepted_times.len()).unwrap()
    );
    assert_eq!(operator.state.buffer.len(), accepted_times.len());
    assert_eq!(operator.state.metrics.late_rows, 3);
    assert_eq!(operator.state.metrics.affected_batches, 1);
    assert_eq!(operator.state.metrics.max_lateness_micros, Some(1));
    assert!(collector.drain("output").is_empty());
    let context = StreamOperatorContext::new(&job, "rolling", Some(EventTime::from_micros(2)));
    operator.on_end(&context, &mut collector).await.unwrap();
    let emitted = collector.drain("output");
    let actual_times = emitted
        .iter()
        .flat_map(|message| {
            message
                .as_data()
                .unwrap()
                .table_payload()
                .unwrap()
                .batches()
                .iter()
        })
        .flat_map(|record| {
            record
                .column(0)
                .as_any()
                .downcast_ref::<datafusion::arrow::array::TimestampMicrosecondArray>()
                .unwrap()
                .values()
                .iter()
                .copied()
        })
        .collect::<Vec<_>>();
    assert_eq!(actual_times, accepted_times);
}

#[tokio::test]
async fn mixed_late_error_reports_attempted_prefix_and_rejects_the_envelope() {
    assert_rejected_late_envelope(
            false,
            &[3, 1, 1, 4],
            "late_row: envelope rejected at row_index=1; event_time_micros=1, closed_at_watermark_micros=2",
            1,
        )
        .await;
}

#[tokio::test]
async fn duplicate_after_late_drop_preserves_envelope_metrics_and_state() {
    assert_rejected_late_envelope(true, &[1, 3, 3, 4], "duplicate row identity", 2).await;
}

async fn assert_rejected_late_envelope(
    drop_late: bool,
    times: &[i64],
    expected_error: &str,
    expected_order_proofs: u64,
) {
    let (mut operator, job, store, mut collector) = late_observation_fixture(drop_late);
    let error = observe_late_data(&mut operator, &job, &store, times, &mut collector)
        .await
        .unwrap_err();
    assert!(
        matches!(error, CalcFlowError::Operator { ref node_id, ref message } if node_id == "rolling" && message.contains(expected_error)),
        "unexpected error: {error}"
    );
    let metrics = store.snapshot().data;
    assert_eq!(metrics.failed, 1);
    assert_eq!(metrics.succeeded, 0);
    assert_eq!(metrics.order_proof_rows, expected_order_proofs);
    assert_eq!(operator.state.metrics, LateMetricDelta::default());
    assert!(operator.state.buffer.is_empty());
    assert!(operator.state.ordered.is_empty());
    assert!(operator.state.histories.by_entity.is_empty());
    assert!(operator.state.typed_kernel_state.is_none());
    assert!(operator.state.operator_id.is_none());
    assert!(operator.state.pipeline_fingerprint.is_none());
    assert_eq!(operator.state.next_output_sequence, 0);
    assert!(collector.drain("output").is_empty());
    observe_late_data(&mut operator, &job, &store, &[3, 4], &mut collector)
        .await
        .unwrap();
    let context = StreamOperatorContext::new(&job, "rolling", Some(EventTime::from_micros(2)));
    operator.on_end(&context, &mut collector).await.unwrap();
    let emitted = collector.drain("output");
    assert_eq!(emitted.len(), 1);
    let output = emitted[0].as_data().unwrap();
    assert_eq!(output.metadata().sequence(), 0);
    let means = output.table_payload().unwrap().batches()[0]
        .column(6)
        .as_any()
        .downcast_ref::<Float64Array>()
        .unwrap();
    assert_eq!(means.values().as_ref(), &[3.0, 3.5]);
    assert_eq!(operator.state.metrics, LateMetricDelta::default());
}

fn late_observation_fixture(
    drop_late: bool,
) -> (
    RollingOperator,
    crate::StreamJobContext,
    crate::operator::rolling_metrics::RollingMetricsStore,
    crate::EdgeCollector,
) {
    use crate::operator::rolling_metrics::RollingMetricsStore;
    use crate::{CancellationToken, StreamJobContext};

    let mut spec = kernel_spec(json!([aggregate_output("mean", "price", "mean", 2)]));
    if drop_late {
        spec.late_policy = LatePolicySpec::Drop { metrics_version: 1 };
    }
    let operator = RollingOperator::new("rolling", Arc::new(kernel_schema()), spec).unwrap();
    let collector = crate::EdgeCollector::new(operator.output_ports().to_vec());
    let job = StreamJobContext::new(
        7,
        TEST_FINGERPRINT,
        JsonMap::new(),
        None,
        CancellationToken::new(),
    );
    (operator, job, RollingMetricsStore::default(), collector)
}

async fn observe_late_data(
    operator: &mut RollingOperator,
    job: &crate::StreamJobContext,
    store: &crate::operator::rolling_metrics::RollingMetricsStore,
    times: &[i64],
    collector: &mut crate::EdgeCollector,
) -> Result<()> {
    use crate::CancellationToken;
    use crate::operator::rolling_metrics::RollingCallback;

    let rows = times
        .iter()
        .map(|&time| {
            (
                time,
                "a",
                u64::try_from(time).unwrap(),
                Some(f64::from(i32::try_from(time).unwrap())),
            )
        })
        .collect::<Vec<_>>();
    let input = Batch::table(vec![float64_fast_record(&rows)], BatchMetadata::default()).unwrap();
    let callback = store.begin(RollingCallback::Data, CancellationToken::new());
    let context = StreamOperatorContext::new(job, "rolling", Some(EventTime::from_micros(2)))
        .with_rolling_metrics(callback.recorder());
    let result = operator
        .process_data("input", input, &context, collector)
        .await;
    callback.complete(&result);
    result
}

fn parallel_rollback_record(start: usize, count: usize) -> RecordBatch {
    let names = (0..64)
        .map(|entity| format!("e{entity:04}"))
        .collect::<Vec<_>>();
    float64_fast_record(
        &(start..start + count)
            .map(|row| {
                (
                    i64::try_from(row).unwrap(),
                    names[row % 64].as_str(),
                    u64::try_from(row).unwrap(),
                    Some(f64::from(u32::try_from(row % 101).unwrap())),
                )
            })
            .collect::<Vec<_>>(),
    )
}

#[tokio::test]
async fn failed_parallel_ordered_emission_preserves_committed_snapshot() {
    use std::sync::atomic::{AtomicU64, Ordering as AtomicOrdering};

    use crate::runtime::streaming::entity_work::JobEntityWorkOwner;
    use crate::{CancellationToken, StreamJobContext};

    struct Reject;
    #[async_trait]
    impl StreamCollector for Reject {
        async fn emit(&mut self, _port: &str, _batch: Batch) -> Result<()> {
            Err(internal_error("expected parallel collector rejection"))
        }
    }

    let spec = kernel_spec(json!([
        aggregate_output("mean", "price", "fast", 5),
        aggregate_output("mean", "price", "slow", 20)
    ]));
    let mut operator = RollingOperator::new("rolling", Arc::new(kernel_schema()), spec).unwrap();
    let job = StreamJobContext::new(
        7,
        TEST_FINGERPRINT,
        JsonMap::new(),
        None,
        CancellationToken::new(),
    );
    let context = StreamOperatorContext::new(&job, "rolling", None);
    let mut collector = crate::EdgeCollector::new(operator.output_ports().to_vec());
    operator
        .process_data(
            "input",
            Batch::table(
                vec![parallel_rollback_record(0, 1_280)],
                BatchMetadata::default(),
            )
            .unwrap(),
            &context,
            &mut collector,
        )
        .await
        .unwrap();
    operator
        .on_watermark(EventTime::from_micros(1_279), &context, &mut collector)
        .await
        .unwrap();
    collector.drain("output");
    assert!(operator.state.typed_kernel_state.is_some());

    // Materialize the existing history once; compare the same epoch's
    // complete encoded snapshot without advancing checkpoint state again.
    let epoch = Epoch::new(1).unwrap();
    let snapshot = operator.checkpoint(epoch).unwrap();
    let committed = |operator: &RollingOperator| {
        (
            operator.encode_state(epoch).unwrap(),
            format!("{:?}", operator.state.typed_kernel_state),
            (
                operator.state.last_input_watermark,
                operator.state.next_output_sequence,
                operator.state.ended,
                operator.state.metrics,
                operator.state.pipeline_fingerprint.clone(),
                operator.state.operator_id.clone(),
                operator.state.last_checkpoint_epoch,
            ),
            retained_buffer_addresses(operator),
        )
    };
    let before = committed(&operator);
    let (segment_id, bytes) = before.0.as_ref().unwrap();
    assert_eq!(snapshot.segments[segment_id].bytes(), bytes);

    let launches = Arc::new(AtomicU64::new(0));
    let owner = JobEntityWorkOwner::new(7, Arc::clone(&launches));
    let task = owner.test_client(3, "operator:rolling".into());
    let scope = task.callback_scope();
    let parallel_context =
        StreamOperatorContext::new(&job, "rolling", Some(EventTime::from_micros(1_279)))
            .with_entity_work(task.context_client().unwrap());
    let result = operator
        .emit_ordered(
            vec![parallel_rollback_record(1_280, 64_000)],
            &parallel_context,
            &mut Reject,
        )
        .await;
    scope.settle_abandoned().await;
    drop(parallel_context);
    drop(scope);
    owner.close_admission();
    assert!(owner.drain().await.is_empty());

    assert!(
        result
            .unwrap_err()
            .to_string()
            .contains("expected parallel collector rejection")
    );
    assert_eq!(launches.load(AtomicOrdering::SeqCst), 2);
    assert_eq!(committed(&operator), before);
    assert!(operator.state.buffer.is_empty());
    assert!(operator.state.ordered.is_empty());
    assert!(collector.drain("output").is_empty());
}

#[tokio::test]
async fn failed_ordered_emission_preserves_both_history_and_typed_state() {
    assert_rejected_compaction_preserves_state(false).await;
}

#[tokio::test]
async fn cancelled_ordered_emission_preserves_both_history_and_typed_state() {
    assert_rejected_compaction_preserves_state(true).await;
}

async fn assert_rejected_compaction_preserves_state(cancelled: bool) {
    use crate::CancellationToken;
    use crate::operator::rolling_metrics::{RollingCallback, RollingMetricsStore};

    struct Reject(bool);
    #[async_trait]
    impl StreamCollector for Reject {
        async fn emit(&mut self, _port: &str, _batch: Batch) -> Result<()> {
            if self.0 {
                Err(CalcFlowError::Cancelled { run_id: "7".into() })
            } else {
                Err(internal_error("expected collector rejection"))
            }
        }
    }

    let (mut operator, job, mut collector) = rolling_with_large_retained_label().await;
    let prior = format!("{:?}", operator.state.typed_kernel_state);
    let prior_buffers = retained_buffer_addresses(&operator);
    let sequence = operator.state.next_output_sequence;
    let next = Batch::table(
        vec![float64_fast_record(&[(3, "a", 3, Some(5.0))])],
        BatchMetadata::default(),
    )
    .unwrap();
    let context = StreamOperatorContext::new(&job, "rolling", Some(EventTime::from_micros(2)));
    operator
        .process_data("input", next.clone(), &context, &mut collector)
        .await
        .unwrap();
    let store = RollingMetricsStore::default();
    let callback = store.begin(RollingCallback::Watermark, CancellationToken::new());
    let observed_context = context.with_rolling_metrics(callback.recorder());
    let result = operator
        .on_watermark(
            EventTime::from_micros(3),
            &observed_context,
            &mut Reject(cancelled),
        )
        .await;
    callback.complete(&result);
    let error = result.unwrap_err();
    if cancelled {
        assert!(matches!(error, CalcFlowError::Cancelled { run_id } if run_id == "7"));
    } else {
        assert!(error.to_string().contains("expected collector rejection"));
    }
    assert_eq!(format!("{:?}", operator.state.typed_kernel_state), prior);
    assert_eq!(operator.state.next_output_sequence, sequence);
    assert_eq!(
        operator.state.last_input_watermark,
        Some(EventTime::from_micros(2))
    );
    assert_eq!(retained_buffer_addresses(&operator), prior_buffers);
    let history = operator.state.histories.by_entity.values().next().unwrap();
    assert_eq!(history.transition_count, 2);
    assert!(
        history.columnar.records[0]
            .column(5)
            .get_buffer_memory_size()
            >= 8 * 1024 * 1024
    );
    assert_eq!(
        single_entity_retained_prices(&operator).values().as_ref(),
        &[1.0, 3.0]
    );
    let metrics = store.snapshot().watermark;
    assert_eq!(metrics.failed, u64::from(!cancelled));
    assert_eq!(metrics.cancelled, u64::from(cancelled));
    assert_eq!(metrics.numeric_rows, 1);
    assert_eq!(metrics.output_rows_prepared, 1);
    assert_eq!(metrics.output_chunks_prepared, 1);
    assert!(collector.drain("output").is_empty());
    assert_retry_releases_evicted_label(&mut operator, &job, &mut collector, next, sequence).await;
}

async fn rolling_with_large_retained_label() -> (
    RollingOperator,
    crate::StreamJobContext,
    crate::EdgeCollector,
) {
    use crate::{CancellationToken, StreamJobContext};

    let spec = kernel_spec(json!([aggregate_output("mean", "price", "mean", 2)]));
    let mut operator = RollingOperator::new("rolling", Arc::new(kernel_schema()), spec).unwrap();
    let job = StreamJobContext::new(
        7,
        TEST_FINGERPRINT,
        JsonMap::new(),
        None,
        CancellationToken::new(),
    );
    let context = StreamOperatorContext::new(&job, "rolling", None);
    let mut collector = crate::EdgeCollector::new(operator.output_ports().to_vec());
    let first = with_large_first_label(&float64_fast_record(&[
        (1, "a", 1, Some(1.0)),
        (2, "a", 2, Some(3.0)),
    ]));
    operator
        .process_data(
            "input",
            Batch::table(vec![first], BatchMetadata::default()).unwrap(),
            &context,
            &mut collector,
        )
        .await
        .unwrap();
    operator
        .on_watermark(EventTime::from_micros(2), &context, &mut collector)
        .await
        .unwrap();
    collector.drain("output");
    (operator, job, collector)
}

fn retained_buffer_addresses(operator: &RollingOperator) -> Vec<usize> {
    operator
        .state
        .histories
        .by_entity
        .values()
        .flat_map(|history| &history.columnar.records)
        .flat_map(RecordBatch::columns)
        .flat_map(|column| {
            let data = column.to_data();
            data.buffers()
                .iter()
                .map(|buffer| buffer.as_ptr() as usize)
                .chain(data.nulls().map(|nulls| nulls.buffer().as_ptr() as usize))
                .collect::<Vec<_>>()
        })
        .collect()
}

async fn assert_retry_releases_evicted_label(
    operator: &mut RollingOperator,
    job: &crate::StreamJobContext,
    collector: &mut crate::EdgeCollector,
    next: Batch,
    sequence: u64,
) {
    let context = StreamOperatorContext::new(job, "rolling", Some(EventTime::from_micros(2)));
    operator
        .process_data("input", next, &context, collector)
        .await
        .unwrap();
    operator
        .on_watermark(EventTime::from_micros(3), &context, collector)
        .await
        .unwrap();
    let emitted = collector.drain("output");
    assert_eq!(emitted.len(), 1);
    let output = emitted[0].as_data().unwrap();
    assert_eq!(output.metadata().sequence(), sequence);
    let means = output.table_payload().unwrap().batches()[0]
        .column(6)
        .as_any()
        .downcast_ref::<Float64Array>()
        .unwrap();
    assert_eq!(means.values().as_ref(), &[4.0]);
    assert_eq!(operator.state.next_output_sequence, sequence + 1);
    let history = operator.state.histories.by_entity.values().next().unwrap();
    assert_eq!(history.transition_count, 3);
    let retained = history
        .columnar
        .records
        .iter()
        .flat_map(|record| {
            record
                .column(3)
                .as_any()
                .downcast_ref::<Float64Array>()
                .unwrap()
                .values()
                .iter()
                .copied()
        })
        .collect::<Vec<_>>();
    assert_eq!(retained, [3.0, 5.0]);
    assert!(
        history
            .columnar
            .records
            .iter()
            .map(|record| record.column(5).get_buffer_memory_size())
            .sum::<usize>()
            < 1024
    );
}

#[tokio::test]
async fn general_rolling_observations_distinguish_routing_from_numeric_work() {
    use crate::operator::rolling_metrics::{RollingCallback, RollingMetricsStore};
    use crate::{CancellationToken, StreamJobContext};

    let mut operator =
        RollingOperator::new("rolling", Arc::new(input_schema()), valid_spec()).unwrap();
    let cancellation = CancellationToken::new();
    let job = StreamJobContext::new(
        7,
        TEST_FINGERPRINT,
        JsonMap::new(),
        None,
        cancellation.clone(),
    );
    let store = RollingMetricsStore::default();
    let mut collector = crate::EdgeCollector::new(operator.output_ports().to_vec());
    for index in 1..=2_u32 {
        let row = full_row(
            i64::from(index),
            "a",
            u64::from(index),
            vec![ScalarValue::Float64(Some(f64::from(index)))],
        );
        let callback = store.begin(RollingCallback::Watermark, cancellation.clone());
        let context = StreamOperatorContext::new(&job, "rolling", None)
            .with_rolling_metrics(callback.recorder());
        let result = operator
            .emit_rows(vec![row], &context, &mut collector)
            .await;
        callback.complete(&result);
        result.unwrap();
    }
    let metrics = store.snapshot().watermark;
    assert_eq!(metrics.resolved_rows, 2);
    assert_eq!(metrics.touched_entities, 2);
    assert_eq!(metrics.copied_entities, 1);
    assert_eq!(metrics.numeric_rows, 2);
    assert_eq!(metrics.output_rows_prepared, 2);
}

#[tokio::test]
async fn rolling_scan_retains_history_across_checkpoint() {
    use crate::{CancellationToken, StreamJobContext};

    for checkpoint in [false, true] {
        let spec = kernel_spec(json!([{
            "kind": "unique_count",
            "primitive_version": 1,
            "input": "price",
            "output": "distinct_price",
            "frame": {"kind": "rows", "size": 3},
            "min_periods": 1
        }]));
        let mut operator =
            RollingOperator::new("rolling", Arc::new(kernel_schema()), spec.clone()).unwrap();
        let job = StreamJobContext::new(
            7,
            TEST_FINGERPRINT,
            JsonMap::new(),
            None,
            CancellationToken::new(),
        );
        let mut collector = crate::EdgeCollector::new(operator.output_ports().to_vec());
        let context = StreamOperatorContext::new(&job, "rolling", None);
        let first = float64_fast_record(&[
            (1, "a", 1, Some(1.0)),
            (2, "a", 2, Some(2.0)),
            (3, "a", 3, Some(3.0)),
        ]);
        operator
            .process_data(
                "input",
                Batch::table(vec![first], BatchMetadata::default()).unwrap(),
                &context,
                &mut collector,
            )
            .await
            .unwrap();
        operator
            .on_watermark(EventTime::from_micros(3), &context, &mut collector)
            .await
            .unwrap();
        collector.drain("output");

        if checkpoint {
            let snapshot = operator.checkpoint(Epoch::new(1).unwrap()).unwrap();
            let mut restored =
                RollingOperator::new("rolling", Arc::new(kernel_schema()), spec).unwrap();
            StreamOperator::restore(&mut restored, &snapshot).unwrap();
            operator = restored;
        }

        let context = StreamOperatorContext::new(&job, "rolling", Some(EventTime::from_micros(3)));
        let next = float64_fast_record(&[(4, "a", 4, Some(3.0))]);
        operator
            .process_data(
                "input",
                Batch::table(vec![next], BatchMetadata::default()).unwrap(),
                &context,
                &mut collector,
            )
            .await
            .unwrap();
        operator
            .on_watermark(EventTime::from_micros(4), &context, &mut collector)
            .await
            .unwrap();
        let emitted = collector.drain("output");
        let record = &emitted[0]
            .as_data()
            .unwrap()
            .table_payload()
            .unwrap()
            .batches()[0];
        assert_eq!(record.num_rows(), 1);
        let index = record.schema().index_of("distinct_price").unwrap();
        let counts = record
            .column(index)
            .as_any()
            .downcast_ref::<UInt64Array>()
            .unwrap();
        assert_eq!(counts.value(0), 2, "checkpoint={checkpoint}");
    }
}

#[test]
fn rolling_scans_bound_comparisons_independently_of_window_width() {
    let spec = kernel_spec(json!([
        {
            "kind": "argmax",
            "primitive_version": 1,
            "input": "price",
            "output": "maximum_age",
            "frame": {"kind": "rows", "size": 256},
            "min_periods": 1
        },
        {
            "kind": "unique_count",
            "primitive_version": 1,
            "input": "price",
            "output": "distinct",
            "frame": {"kind": "rows", "size": 64},
            "min_periods": 1
        },
        {
            "kind": "argmin",
            "primitive_version": 1,
            "input": "price",
            "output": "minimum_age",
            "frame": {"kind": "rows", "size": 64},
            "min_periods": 1
        }
    ]));
    let input_rows = (0..1024_u64)
        .map(|index| {
            (
                i64::try_from(index + 1).unwrap(),
                "a",
                index + 1,
                Some(f64::from(u32::try_from(index % 127).unwrap())),
            )
        })
        .collect::<Vec<_>>();
    let input = float64_fast_record(&input_rows);
    let compiled = compile_spec(&spec, &kernel_schema()).unwrap();
    let rows = (0..input.num_rows())
        .map(|index| read_buffered_row(&input, index, &compiled, "rolling").unwrap())
        .collect::<Vec<_>>();
    SCAN_COMPARE_COUNT.with(|count| count.set(0));
    let result =
        compute_output_columns(&rows, &RollingHistories::default(), &compiled, "rolling").unwrap();
    let comparisons = SCAN_COMPARE_COUNT.with(std::cell::Cell::get);
    assert!(comparisons < 8 * rows.len(), "{comparisons} comparisons");
    let ages = result.columns[0]
        .as_any()
        .downcast_ref::<UInt64Array>()
        .unwrap();
    let distinct = result.columns[1]
        .as_any()
        .downcast_ref::<UInt64Array>()
        .unwrap();
    assert_eq!(ages.value(0), 0);
    assert_eq!(ages.value(126), 0);
    assert_eq!(ages.value(255), 129);
    assert_eq!(distinct.value(63), 64);
    assert_eq!(distinct.value(1023), 64);
}

#[test]
fn numeric_scans_compile_to_ordered_columnar_kernel() {
    let spec = kernel_spec(json!([
        {
            "kind": "argmax",
            "primitive_version": 1,
            "input": "price",
            "output": "maximum_age",
            "frame": {"kind": "rows", "size": 64},
            "min_periods": 1
        },
        {
            "kind": "unique_count",
            "primitive_version": 1,
            "input": "price",
            "output": "distinct",
            "frame": {"kind": "rows", "size": 64},
            "min_periods": 1
        },
        {
            "kind": "argmin",
            "primitive_version": 1,
            "input": "price",
            "output": "minimum_age",
            "frame": {"kind": "rows", "size": 64},
            "min_periods": 1
        }
    ]));
    let compiled = compile_spec(&spec, &kernel_schema()).unwrap();
    assert_eq!(
        compiled.kernel_plan.selection(),
        KernelSelection::OrderedPrimitive
    );
    let input = float64_fast_record(&[
        (1, "a", 1, Some(0.0)),
        (2, "a", 2, Some(-0.0)),
        (3, "a", 3, Some(f64::NAN)),
        (4, "a", 4, Some(1.0)),
        (5, "a", 5, Some(1.0)),
        (6, "a", 6, None),
        (7, "a", 7, Some(2.0)),
    ]);
    assert_typed_matches_general(&spec, &kernel_schema(), &input);
}

#[test]
fn incremental_scan_memory_estimate_covers_allocated_capacity() {
    let mut extrema = IncrementalScanState::new(ScanKind::Argmax, 64, 1).unwrap();
    extrema.advance_float(0, Some(1.0));
    let IncrementalScanState::Extrema {
        recent_valid,
        candidates,
        ..
    } = &extrema
    else {
        unreachable!()
    };
    let extrema_minimum = recent_valid.capacity() * size_of::<bool>()
        + candidates.capacity() * size_of::<(usize, f64)>();
    assert!(extrema.estimated_bytes() >= extrema_minimum);

    let mut unique = IncrementalScanState::new(ScanKind::UniqueCount, 64, 1).unwrap();
    unique.advance_float(0, Some(1.0));
    let IncrementalScanState::UniqueFloat64 { recent, counts, .. } = &unique else {
        unreachable!()
    };
    let unique_minimum = recent.capacity() * size_of::<Option<u64>>()
        + counts.capacity() * size_of::<(u64, usize)>();
    assert!(unique.estimated_bytes() >= unique_minimum);
}

#[test]
fn incremental_scans_keep_ties_zero_signs_and_invalid_rows_across_batches() {
    let spec = kernel_spec(json!([
        {
            "kind": "argmax",
            "primitive_version": 1,
            "input": "price",
            "output": "maximum_age",
            "frame": {"kind": "rows", "size": 4},
            "min_periods": 1
        },
        {
            "kind": "unique_count",
            "primitive_version": 1,
            "input": "price",
            "output": "distinct",
            "frame": {"kind": "rows", "size": 4},
            "min_periods": 1
        }
    ]));
    let compiled = compile_spec(&spec, &kernel_schema()).unwrap();
    let input = float64_fast_record(&[
        (1, "a", 1, Some(0.0)),
        (2, "a", 2, Some(-0.0)),
        (3, "a", 3, Some(f64::NAN)),
        (4, "a", 4, Some(1.0)),
        (5, "a", 5, Some(1.0)),
        (6, "a", 6, None),
        (7, "a", 7, Some(2.0)),
        (8, "a", 8, Some(2.0)),
    ]);
    let rows = (0..input.num_rows())
        .map(|index| read_buffered_row(&input, index, &compiled, "rolling").unwrap())
        .collect::<Vec<_>>();
    let first = compute_output_columns(
        &rows[..4],
        &RollingHistories::default(),
        &compiled,
        "rolling",
    )
    .unwrap();
    let mut histories = RollingHistories::default();
    histories.apply(first.touched);
    let second = compute_output_columns(&rows[4..], &histories, &compiled, "rolling").unwrap();
    let values = |column: usize| {
        first.columns[column]
            .as_any()
            .downcast_ref::<UInt64Array>()
            .unwrap()
            .iter()
            .chain(
                second.columns[column]
                    .as_any()
                    .downcast_ref::<UInt64Array>()
                    .unwrap()
                    .iter(),
            )
            .collect::<Vec<_>>()
    };
    assert_eq!(
        values(0),
        vec![
            Some(0),
            Some(1),
            Some(2),
            Some(0),
            Some(1),
            Some(2),
            Some(0),
            Some(1)
        ]
    );
    assert_eq!(
        values(1),
        vec![
            Some(1),
            Some(1),
            Some(1),
            Some(2),
            Some(2),
            Some(1),
            Some(2),
            Some(2)
        ]
    );
}

#[tokio::test]
async fn columnar_history_compacts_input_and_survives_checkpoint_or_fallback() {
    use crate::{CancellationToken, StreamJobContext};

    for checkpoint in [false, true] {
        let spec = kernel_spec(json!([
            aggregate_output("mean", "price", "mean2", 2),
            aggregate_output("mean", "price", "mean5", 5)
        ]));
        let mut operator =
            RollingOperator::new("rolling", Arc::new(kernel_schema()), spec.clone()).unwrap();
        let job = StreamJobContext::new(
            7,
            TEST_FINGERPRINT,
            JsonMap::new(),
            None,
            CancellationToken::new(),
        );
        let rows = (1..=128_u32)
            .map(|index| {
                (
                    i64::from(index),
                    if index % 2 == 0 { "a" } else { "b" },
                    u64::from(index),
                    Some(f64::from(index)),
                )
            })
            .collect::<Vec<_>>();
        let input = float64_fast_record(&rows);
        let context = StreamOperatorContext::new(&job, "rolling", None);
        let mut collector = crate::EdgeCollector::new(operator.output_ports().to_vec());
        operator
            .process_data(
                "input",
                Batch::table(vec![input.clone()], BatchMetadata::default()).unwrap(),
                &context,
                &mut collector,
            )
            .await
            .unwrap();
        operator
            .on_watermark(EventTime::from_micros(128), &context, &mut collector)
            .await
            .unwrap();
        collector.drain("output");
        for history in operator.state.histories.by_entity.values() {
            assert!(history.rows.is_empty());
            assert_eq!(history.transition_count, 64);
            let retained = history.columnar.records.front().unwrap();
            assert_eq!(retained.num_rows(), 5);
            assert!(
                retained.column(3).get_buffer_memory_size()
                    < input.column(3).get_buffer_memory_size() / 4
            );
        }
        if checkpoint {
            let snapshot = operator.checkpoint(Epoch::new(1).unwrap()).unwrap();
            for history in operator.state.histories.by_entity.values() {
                assert!(history.columnar.records.is_empty());
                assert_eq!(history.rows.len(), 5);
            }
            let mut restored =
                RollingOperator::new("rolling", Arc::new(kernel_schema()), spec).unwrap();
            StreamOperator::restore(&mut restored, &snapshot).unwrap();
            operator = restored;
        }
        let next =
            float64_fast_record(&[(132, "a", 132, Some(132.0)), (130, "a", 130, Some(130.0))]);
        let context =
            StreamOperatorContext::new(&job, "rolling", Some(EventTime::from_micros(128)));
        operator
            .process_data(
                "input",
                Batch::table(vec![next], BatchMetadata::default()).unwrap(),
                &context,
                &mut collector,
            )
            .await
            .unwrap();
        operator
            .on_watermark(EventTime::from_micros(132), &context, &mut collector)
            .await
            .unwrap();
        let emitted = collector.drain("output");
        let record = &emitted[0]
            .as_data()
            .unwrap()
            .table_payload()
            .unwrap()
            .batches()[0];
        let means = record
            .column(7)
            .as_any()
            .downcast_ref::<Float64Array>()
            .unwrap();
        assert_eq!(means.values().as_ref(), &[126.0, 128.0]);
        assert_eq!(operator.state.histories.by_entity.len(), 2);
    }
}

#[test]
fn whole_record_budget_proof_avoids_per_row_scratch() {
    let rows = (1..=8192_u64)
        .map(|index| (i64::try_from(index).unwrap(), "中文", index, None))
        .collect::<Vec<_>>();
    let input = float64_fast_record(&rows);
    let measured = allocation_counter::measure(|| {
        let chunks = chunk_output_record(
            &input,
            "rolling",
            0,
            crate::EdgeBudget::new(8192, crate::EdgeBudget::MAX_BYTES).unwrap(),
        )
        .unwrap();
        assert_eq!(chunks.len(), 1);
        assert_eq!(chunks[0].num_rows(), input.num_rows());
    });
    println!(
        "whole-record budget allocated bytes: {}",
        measured.bytes_total
    );
    assert!(
        measured.bytes_total < 16_384,
        "whole-record budget preparation allocated {} bytes",
        measured.bytes_total
    );
}

#[tokio::test]
async fn columnar_history_releases_evicted_large_variable_width_values() {
    assert_releases_evicted_large_label(2).await;
}

#[tokio::test]
async fn columnar_history_compaction_accounts_for_bytes_not_only_rows() {
    assert_releases_evicted_large_label(20).await;
}

fn with_large_first_label(record: &RecordBatch) -> RecordBatch {
    use datafusion::arrow::array::StringArray;

    let mut columns = record.columns().to_vec();
    let labels = std::iter::once("x".repeat(8 * 1024 * 1024))
        .chain((1..record.num_rows()).map(|_| "y".to_owned()))
        .collect::<Vec<_>>();
    columns[5] = Arc::new(StringArray::from(labels));
    RecordBatch::try_new(record.schema(), columns).unwrap()
}

async fn assert_releases_evicted_large_label(window: u32) {
    use crate::{CancellationToken, StreamJobContext};

    let spec = kernel_spec(json!([aggregate_output(
        "mean",
        "price",
        "mean",
        u64::from(window)
    )]));
    let mut operator = RollingOperator::new("rolling", Arc::new(kernel_schema()), spec).unwrap();
    let job = StreamJobContext::new(
        7,
        TEST_FINGERPRINT,
        JsonMap::new(),
        None,
        CancellationToken::new(),
    );
    let mut collector = crate::EdgeCollector::new(operator.output_ports().to_vec());
    let rows = (1..=window)
        .map(|index| {
            (
                i64::from(index),
                "a",
                u64::from(index),
                Some(f64::from(index)),
            )
        })
        .collect::<Vec<_>>();
    let first = with_large_first_label(&float64_fast_record(&rows));
    let context = StreamOperatorContext::new(&job, "rolling", None);
    operator
        .process_data(
            "input",
            Batch::table(vec![first], BatchMetadata::default()).unwrap(),
            &context,
            &mut collector,
        )
        .await
        .unwrap();
    operator
        .on_watermark(
            EventTime::from_micros(i64::from(window)),
            &context,
            &mut collector,
        )
        .await
        .unwrap();
    collector.drain("output");

    let next = window + 1;
    let second = float64_fast_record(&[(i64::from(next), "a", u64::from(next), Some(5.0))]);
    let context = StreamOperatorContext::new(
        &job,
        "rolling",
        Some(EventTime::from_micros(i64::from(window))),
    );
    operator
        .process_data(
            "input",
            Batch::table(vec![second], BatchMetadata::default()).unwrap(),
            &context,
            &mut collector,
        )
        .await
        .unwrap();
    operator
        .on_watermark(
            EventTime::from_micros(i64::from(next)),
            &context,
            &mut collector,
        )
        .await
        .unwrap();
    collector.drain("output");
    let history = operator.state.histories.by_entity.values().next().unwrap();
    let retained_label_bytes: usize = history
        .columnar
        .records
        .iter()
        .map(|record| record.column(5).get_buffer_memory_size())
        .sum();
    assert_eq!(
        history
            .columnar
            .records
            .iter()
            .map(RecordBatch::num_rows)
            .sum::<usize>(),
        usize::try_from(window).unwrap()
    );
    assert!(
        retained_label_bytes < 1024,
        "{window} live short/null labels retain {retained_label_bytes} bytes after the 8 MiB label leaves the window"
    );
}

#[test]
fn whole_record_budget_proof_preserves_exact_limits_and_sequence_errors() {
    let input = float64_fast_record(&[(1, "中文", 1, Some(1.0)), (2, "中文", 2, None)]);
    // The shared chunker's whole-record fast path is priced by the same
    // envelope estimator the channel enforces.
    let bytes = Batch::table(vec![input.clone()], BatchMetadata::default())
        .unwrap()
        .estimated_bytes()
        .unwrap();
    let exact = chunk_output_record(
        &input,
        "rolling",
        u64::MAX - 1,
        crate::EdgeBudget::new(2, bytes).unwrap(),
    )
    .unwrap();
    assert_eq!(exact.len(), 1);
    let split = chunk_output_record(
        &input,
        "rolling",
        0,
        crate::EdgeBudget::new(2, bytes - 1).unwrap(),
    )
    .unwrap();
    assert_eq!(split.len(), 2);
    let row = input.slice(1, 1);
    let bytes = Batch::table(vec![row.clone()], BatchMetadata::default())
        .unwrap()
        .estimated_bytes()
        .unwrap();
    let error = chunk_output_record(
        &row,
        "rolling",
        0,
        crate::EdgeBudget::new(1, bytes - 1).unwrap(),
    )
    .unwrap_err();
    assert!(
        matches!(error, CalcFlowError::InvalidArgument { field, .. } if field == "message.bytes")
    );
    let budget = crate::EdgeBudget::new(2, crate::EdgeBudget::MAX_BYTES).unwrap();
    assert!(
        chunk_output_record(&input, "rolling", u64::MAX, budget)
            .unwrap_err()
            .to_string()
            .contains("output sequence overflowed before emission")
    );
    assert!(
        chunk_output_record(&input.slice(0, 0), "rolling", u64::MAX, budget)
            .unwrap()
            .is_empty()
    );
}

#[test]
fn ordered_float64_plan_falls_back_for_unsorted_input() {
    let spec = kernel_spec(json!([aggregate_output("mean", "price", "price_mean", 2)]));
    let compiled = compile_spec(&spec, &kernel_schema()).unwrap();
    let input = float64_fast_record(&[(2, "a", 2, Some(2.0)), (1, "a", 1, Some(1.0))]);

    assert!(
        compiled
            .kernel_plan
            .open_and_fill(&input, "rolling")
            .unwrap()
            .is_none()
    );
}

#[test]
fn ordered_float64_plan_preserves_infinity_and_overflow_classification() {
    let spec = kernel_spec(json!([
        aggregate_output("sum", "price", "price_sum", 2),
        aggregate_output("mean", "price", "price_mean", 2),
        ddof_output("variance", "price", "price_var", 2, 1),
        ddof_output("stddev", "price", "price_std", 2, 0),
    ]));
    let compiled = compile_spec(&spec, &kernel_schema()).unwrap();
    let input = float64_fast_record(&[
        (1, "a", 1, Some(f64::INFINITY)),
        (2, "a", 2, Some(f64::NEG_INFINITY)),
        (3, "a", 3, Some(1.0)),
        (4, "a", 4, Some(f64::MAX)),
        (5, "a", 5, Some(f64::MAX)),
    ]);
    let fast = compiled
        .kernel_plan
        .open_and_fill(&input, "rolling")
        .unwrap()
        .unwrap();
    let rows = (0..input.num_rows())
        .map(|row_index| read_buffered_row(&input, row_index, &compiled, "rolling").unwrap())
        .collect::<Vec<_>>();
    let general =
        compute_output_columns(&rows, &RollingHistories::default(), &compiled, "rolling").unwrap();

    for (fast, general) in fast.columns.iter().zip(&general.columns) {
        assert_eq!(fast.to_data(), general.to_data());
    }
}

#[test]
fn duration_float64_plan_matches_the_general_kernel() {
    let spec = aggregate_spec(json!([duration_output("mean", "price", "price_mean", 10)]));
    let compiled = compile_spec(&spec, &kernel_schema()).unwrap();
    let input = float64_fast_record(&[
        (1, "a", 1, Some(1.0)),
        (5, "a", 2, Some(5.0)),
        (12, "a", 3, Some(12.0)),
        (12, "b", 1, Some(20.0)),
        (14, "a", 4, Some(14.0)),
    ]);
    let fast = compiled
        .kernel_plan
        .open_and_fill(&input, "rolling")
        .unwrap()
        .unwrap();
    let rows = (0..input.num_rows())
        .map(|row_index| read_buffered_row(&input, row_index, &compiled, "rolling").unwrap())
        .collect::<Vec<_>>();
    let general =
        compute_output_columns(&rows, &RollingHistories::default(), &compiled, "rolling").unwrap();

    assert_eq!(
        compiled.kernel_plan.selection(),
        KernelSelection::OrderedPrimitive
    );
    assert_eq!(fast.columns[0].as_ref(), general.columns[0].as_ref());
}

#[test]
fn float64_extrema_and_pair_groups_match_the_general_kernel() {
    let spec = aggregate_spec(json!([
        duration_output("mean", "price", "price_mean", 10),
        duration_output("min", "price", "price_min", 10),
        duration_output("max", "price", "price_max", 10),
        pair_output(
            "covariance",
            "price",
            "volume",
            "price_volume_cov",
            json!({"kind": "duration", "micros": 10}),
            1
        ),
        pair_output(
            "correlation",
            "price",
            "volume",
            "price_volume_corr",
            json!({"kind": "rows", "size": 3}),
            1
        ),
    ]));
    let schema = float64_pair_schema();
    let compiled = compile_spec(&spec, &schema).unwrap();
    let input = float64_pair_record(&[
        (1, "a", 1, Some(-0.0), Some(2.0)),
        (5, "a", 2, Some(5.0), Some(4.0)),
        (12, "a", 3, Some(3.0), Some(8.0)),
        (12, "b", 1, Some(20.0), Some(1.0)),
        (14, "a", 4, Some(7.0), Some(16.0)),
        (15, "a", 5, Some(f64::NAN), Some(32.0)),
    ]);
    let fast = compiled
        .kernel_plan
        .open_and_fill(&input, "rolling")
        .unwrap()
        .unwrap();
    let rows = (0..input.num_rows())
        .map(|row_index| read_buffered_row(&input, row_index, &compiled, "rolling").unwrap())
        .collect::<Vec<_>>();
    let general =
        compute_output_columns(&rows, &RollingHistories::default(), &compiled, "rolling").unwrap();

    assert_eq!(fast.columns.len(), general.columns.len());
    for (fast, general) in fast.columns.iter().zip(&general.columns) {
        assert_eq!(fast.to_data(), general.to_data());
    }
}

#[test]
fn int64_numeric_groups_keep_exact_sums_and_match_the_general_kernel() {
    let spec = kernel_spec(json!([
        aggregate_output("count", "volume", "volume_count", 2),
        aggregate_output("sum", "volume", "volume_sum", 2),
        aggregate_output("mean", "volume", "volume_mean", 2),
        ddof_output("variance", "volume", "volume_var", 2, 1),
    ]));
    let compiled = compile_spec(&spec, &kernel_schema()).unwrap();
    let input = int64_fast_record(&[Some(i64::MAX), Some(-1), None, Some(2)]);
    let fast = compiled
        .kernel_plan
        .open_and_fill(&input, "rolling")
        .unwrap()
        .unwrap();
    let rows = (0..input.num_rows())
        .map(|row_index| read_buffered_row(&input, row_index, &compiled, "rolling").unwrap())
        .collect::<Vec<_>>();
    let general =
        compute_output_columns(&rows, &RollingHistories::default(), &compiled, "rolling").unwrap();

    assert_eq!(fast.columns.len(), general.columns.len());
    for (fast, general) in fast.columns.iter().zip(&general.columns) {
        assert_eq!(fast.to_data(), general.to_data());
    }
}

#[test]
fn uint64_numeric_groups_keep_exact_sums_and_match_the_general_kernel() {
    let spec = kernel_spec(json!([
        aggregate_output("count", "sequence", "sequence_count", 3),
        aggregate_output("sum", "sequence", "sequence_sum", 3),
        aggregate_output("mean", "sequence", "sequence_mean", 3),
    ]));
    let compiled = compile_spec(&spec, &kernel_schema()).unwrap();
    let input = float64_fast_record(&[
        (1, "a", 1, None),
        (2, "a", 2, None),
        (3, "a", 3, None),
        (4, "a", 4, None),
    ]);
    let fast = compiled
        .kernel_plan
        .open_and_fill(&input, "rolling")
        .unwrap()
        .unwrap();
    let rows = (0..input.num_rows())
        .map(|row_index| read_buffered_row(&input, row_index, &compiled, "rolling").unwrap())
        .collect::<Vec<_>>();
    let general =
        compute_output_columns(&rows, &RollingHistories::default(), &compiled, "rolling").unwrap();

    for (fast, general) in fast.columns.iter().zip(&general.columns) {
        assert_eq!(fast.to_data(), general.to_data());
    }
}

#[test]
fn primitive_numeric_and_extrema_types_preserve_the_frozen_output_schema() {
    use datafusion::arrow::array::{Float32Array, Int8Array, UInt16Array};

    let spec = kernel_spec(json!([
        aggregate_output("count", "volume", "volume_count", 3),
        aggregate_output("sum", "volume", "volume_sum", 3),
        aggregate_output("mean", "volume", "volume_mean", 3),
        aggregate_output("min", "volume", "volume_min", 3),
        aggregate_output("max", "volume", "volume_max", 3),
    ]));
    let cases = [
        primitive_volume_record(Arc::new(Int8Array::from(vec![
            Some(-128),
            Some(127),
            None,
            Some(-1),
        ]))),
        primitive_volume_record(Arc::new(UInt16Array::from(vec![
            Some(65_535),
            Some(1),
            None,
            Some(4),
        ]))),
        primitive_volume_record(Arc::new(Float32Array::from(vec![
            Some(-0.0),
            Some(5.5),
            Some(f32::NAN),
            Some(-2.25),
        ]))),
    ];

    for (schema, input) in cases {
        assert_typed_matches_general(&spec, &schema, &input);
        let compiled = compile_spec(&spec, &schema).unwrap();
        assert_eq!(
            &compiled.outputs[3].output_type,
            schema.field(4).data_type(),
        );
        assert_eq!(
            &compiled.outputs[4].output_type,
            schema.field(4).data_type(),
        );
    }
}

#[test]
fn typed_ewma_shares_one_recurrence_and_matches_the_general_kernel() {
    let spec = exponential_kernel_spec(json!([
        ewma_price(3, 1, "ema_ready"),
        ewma_price(3, 3, "ema_warm"),
    ]));
    let input = float64_fast_record(&[
        (1, "a", 1, Some(1.0)),
        (1, "b", 1, Some(10.0)),
        (2, "a", 2, None),
        (3, "a", 3, Some(3.0)),
        (4, "a", 4, Some(f64::NAN)),
        (5, "a", 5, Some(5.0)),
    ]);

    assert_typed_matches_general(&spec, &kernel_schema(), &input);
    let compiled = compile_spec(&spec, &kernel_schema()).unwrap();
    assert_eq!(compiled.window_groups.len(), 1);
    assert_eq!(
        compiled.kernel_plan.selection(),
        KernelSelection::OrderedPrimitive
    );
}

#[test]
fn cumulative_mean_uses_typed_transition_with_general_numeric_parity() {
    let spec = exponential_kernel_spec(json!([{
        "kind": "cumulative_mean",
        "primitive_version": 1,
        "input": "price",
        "output": "average",
        "min_periods": 1
    }]));
    let input = float64_fast_record(&[
        (1, "a", 1, Some(1.0)),
        (1, "b", 1, Some(10.0)),
        (2, "a", 2, None),
        (3, "a", 3, Some(3.0)),
        (4, "a", 4, Some(f64::NAN)),
        (5, "a", 5, Some(5.0)),
    ]);

    assert_typed_matches_general(&spec, &kernel_schema(), &input);
    let compiled = compile_spec(&spec, &kernel_schema()).unwrap();
    assert_eq!(
        compiled.kernel_plan.selection(),
        KernelSelection::OrderedPrimitive
    );
}

#[test]
fn cumulative_mean_accepts_ordered_columnar_stream_input() {
    let spec = exponential_kernel_spec(json!([{
        "kind": "cumulative_mean",
        "primitive_version": 1,
        "input": "price",
        "output": "average",
        "min_periods": 1
    }]));
    let mut operator = RollingOperator::new("rolling", Arc::new(kernel_schema()), spec).unwrap();
    let input = float64_fast_record(&[(1, "a", 1, Some(10.0)), (2, "a", 2, Some(14.0))]);
    let batch = Batch::table(vec![input], BatchMetadata::default()).unwrap();
    assert!(
        operator
            .try_buffer_ordered(batch.table_payload().unwrap(), None, None)
            .unwrap()
    );
    assert!(!operator.state.ordered.is_empty());
}

#[tokio::test]
async fn ordered_cumulative_mean_restores_seeds_for_the_correct_entity() {
    use crate::{CancellationToken, StreamJobContext};

    let spec = exponential_kernel_spec(json!([{
        "kind": "cumulative_mean",
        "primitive_version": 1,
        "input": "price",
        "output": "average",
        "min_periods": 1
    }]));
    let mut operator =
        RollingOperator::new("rolling", Arc::new(kernel_schema()), spec.clone()).unwrap();
    let job = StreamJobContext::new(
        7,
        TEST_FINGERPRINT,
        JsonMap::new(),
        None,
        CancellationToken::new(),
    );
    let mut collector = crate::EdgeCollector::new(operator.output_ports().to_vec());
    let drain_means = |collector: &mut crate::EdgeCollector| {
        let output = collector.drain("output");
        output[0]
            .as_data()
            .unwrap()
            .table_payload()
            .unwrap()
            .batches()[0]
            .column(kernel_schema().fields().len())
            .as_any()
            .downcast_ref::<Float64Array>()
            .unwrap()
            .values()
            .as_ref()
            .to_vec()
    };
    let context = StreamOperatorContext::new(&job, "rolling", None);
    let first = float64_fast_record(&[(1, "a", 1, Some(10.0)), (1, "b", 1, Some(100.0))]);
    operator
        .process_data(
            "input",
            Batch::table(vec![first], BatchMetadata::default()).unwrap(),
            &context,
            &mut collector,
        )
        .await
        .unwrap();
    operator
        .on_watermark(EventTime::from_micros(1), &context, &mut collector)
        .await
        .unwrap();
    assert_eq!(drain_means(&mut collector), [10.0, 100.0]);

    let context = StreamOperatorContext::new(&job, "rolling", Some(EventTime::from_micros(1)));
    let second = float64_fast_record(&[(2, "b", 2, Some(200.0))]);
    operator
        .process_data(
            "input",
            Batch::table(vec![second], BatchMetadata::default()).unwrap(),
            &context,
            &mut collector,
        )
        .await
        .unwrap();
    operator
        .on_watermark(EventTime::from_micros(2), &context, &mut collector)
        .await
        .unwrap();
    assert_eq!(drain_means(&mut collector), [150.0]);

    let snapshot = operator.checkpoint(Epoch::new(1).unwrap()).unwrap();
    let mut restored = RollingOperator::new("rolling", Arc::new(kernel_schema()), spec).unwrap();
    StreamOperator::restore(&mut restored, &snapshot).unwrap();
    let context = StreamOperatorContext::new(&job, "rolling", Some(EventTime::from_micros(2)));
    let next = float64_fast_record(&[(3, "a", 3, Some(30.0)), (3, "b", 3, Some(300.0))]);
    restored
        .process_data(
            "input",
            Batch::table(vec![next], BatchMetadata::default()).unwrap(),
            &context,
            &mut collector,
        )
        .await
        .unwrap();
    restored
        .on_watermark(EventTime::from_micros(3), &context, &mut collector)
        .await
        .unwrap();
    assert_eq!(drain_means(&mut collector), [20.0, 200.0]);
}

#[test]
fn typed_exponential_means_resume_from_columnar_checkpoints() {
    for spec in [
        exponential_kernel_spec(json!([ewma_price(3, 1, "ema")])),
        exponential_kernel_spec(json!([{
            "kind": "cumulative_mean",
            "primitive_version": 1,
            "input": "price",
            "output": "average",
            "min_periods": 1
        }])),
    ] {
        let compiled = compile_spec(&spec, &kernel_schema()).unwrap();
        let output_schema = Arc::new(output_schema(&kernel_schema(), &compiled.outputs));
        let rows = vec![
            full_row(1, "a", 1, vec![ScalarValue::Float64(Some(10.0))]),
            full_row(2, "a", 2, vec![ScalarValue::Float64(Some(14.0))]),
            full_row(3, "a", 3, vec![ScalarValue::Float64(Some(18.0))]),
            full_row(4, "a", 4, vec![ScalarValue::Float64(Some(10.0))]),
        ];
        let expected =
            compute_output_columns(&rows, &RollingHistories::default(), &compiled, "rolling")
                .unwrap();

        let (first, _, touched) = build_typed_stream_output(
            &rows[..2],
            &RollingHistories::default(),
            None,
            &compiled,
            &output_schema,
            "rolling",
            None,
        )
        .unwrap()
        .unwrap();
        let mut histories = RollingHistories::default();
        histories.apply(touched);
        let bytes = encode_state_segment(
            &histories,
            &BTreeMap::new(),
            &kernel_schema(),
            &compiled,
            TEST_FINGERPRINT,
            "rolling",
        )
        .unwrap();
        let metadata = RollingSnapshotMetadata {
            late_output: None,
            state_layout_version: ROLLING_COLUMNAR_STATE_LAYOUT_VERSION,
            configuration_hash: compiled.configuration_hash.clone(),
            state_schema_fingerprint: compiled.state_schema_fingerprint.clone(),
            kernel_fingerprint: Some(compiled.kernel_plan.fingerprint().to_owned()),
            numerical_profile: Some(compiled.kernel_plan.numerical_profile().to_owned()),
            epoch: Epoch::new(1).unwrap(),
            pipeline_fingerprint: Some(TEST_FINGERPRINT.into()),
            operator_id: Some("rolling".into()),
            last_input_watermark: None,
            next_output_sequence: 0,
            ended: false,
            metrics: LateMetricDelta::default(),
            segment_inventory: Vec::new(),
        };
        let restored =
            decode_state_segment(&bytes, &kernel_schema(), &compiled, &metadata).unwrap();
        assert!(
            restored
                .histories
                .by_entity
                .values()
                .all(|state| state.rows.is_empty())
        );
        let (second, _, _) = build_typed_stream_output(
            &rows[2..],
            &restored.histories,
            None,
            &compiled,
            &output_schema,
            "rolling",
            None,
        )
        .unwrap()
        .unwrap();

        let actual = first
            .column(kernel_schema().fields().len())
            .as_any()
            .downcast_ref::<Float64Array>()
            .unwrap()
            .iter()
            .chain(
                second
                    .column(kernel_schema().fields().len())
                    .as_any()
                    .downcast_ref::<Float64Array>()
                    .unwrap()
                    .iter(),
            )
            .collect::<Vec<_>>();
        assert_eq!(actual, float_column(&expected, 0));
    }
}

#[test]
fn stable_v2_transition_count_round_trips_in_columnar_state() {
    let mut spec = kernel_spec(json!([aggregate_output("mean", "price", "price_mean", 64)]));
    spec.numerical_profile = RollingNumericalProfile::StableV2Preview;
    let compiled = compile_spec(&spec, &kernel_schema()).unwrap();
    let rows = (0..64)
        .map(|index| {
            let sequence = u64::try_from(index + 1).unwrap();
            full_row(
                i64::from(index + 1),
                "a",
                sequence,
                vec![ScalarValue::Float64(Some(1.0e12 + f64::from(index)))],
            )
        })
        .collect::<Vec<_>>();
    let outputs =
        compute_output_columns(&rows, &RollingHistories::default(), &compiled, "rolling").unwrap();
    let mut histories = RollingHistories::default();
    histories.apply(outputs.touched);
    let bytes = encode_state_segment(
        &histories,
        &BTreeMap::new(),
        &kernel_schema(),
        &compiled,
        TEST_FINGERPRINT,
        "rolling",
    )
    .unwrap();
    let metadata = RollingSnapshotMetadata {
        late_output: None,
        state_layout_version: ROLLING_COLUMNAR_STATE_LAYOUT_VERSION,
        configuration_hash: compiled.configuration_hash.clone(),
        state_schema_fingerprint: compiled.state_schema_fingerprint.clone(),
        kernel_fingerprint: Some(compiled.kernel_plan.fingerprint().to_owned()),
        numerical_profile: Some("stable_v2".into()),
        epoch: Epoch::new(1).unwrap(),
        pipeline_fingerprint: Some(TEST_FINGERPRINT.into()),
        operator_id: Some("rolling".into()),
        last_input_watermark: None,
        next_output_sequence: 0,
        ended: false,
        metrics: LateMetricDelta::default(),
        segment_inventory: Vec::new(),
    };

    let restored = decode_state_segment(&bytes, &kernel_schema(), &compiled, &metadata).unwrap();
    assert_eq!(
        restored
            .histories
            .by_entity
            .values()
            .next()
            .unwrap()
            .transition_count,
        64
    );
}

#[test]
fn fused_dual_mean_writes_only_the_final_difference_column() {
    let spec = kernel_spec(json!([difference_output(
        &mean_leaf("price", 2),
        &mean_leaf("price", 4),
        "mean_spread"
    )]));
    let compiled = compile_spec(&spec, &kernel_schema()).unwrap();
    let input = float64_fast_record(&[
        (1, "a", 1, Some(1.0)),
        (2, "a", 2, Some(3.0)),
        (3, "a", 3, Some(5.0)),
        (4, "a", 4, Some(9.0)),
    ]);
    let fast = compiled
        .kernel_plan
        .open_and_fill(&input, "rolling")
        .unwrap()
        .unwrap();
    let rows = (0..input.num_rows())
        .map(|row_index| read_buffered_row(&input, row_index, &compiled, "rolling").unwrap())
        .collect::<Vec<_>>();
    let general =
        compute_output_columns(&rows, &RollingHistories::default(), &compiled, "rolling").unwrap();

    assert_eq!(compiled.outputs.len(), 1);
    assert_eq!(compiled.window_groups.len(), 2);
    assert_eq!(fast.columns.len(), 1);
    assert_eq!(fast.columns[0].to_data(), general.columns[0].to_data());
    assert_eq!(
        fast.columns[0]
            .as_any()
            .downcast_ref::<Float64Array>()
            .unwrap()
            .iter()
            .collect::<Vec<_>>(),
        vec![Some(0.0), Some(0.0), Some(1.0), Some(2.5)]
    );
}

#[test]
fn count_sum_and_mean_slide_over_each_entity_window() {
    let spec = kernel_spec(json!([
        aggregate_output("count", "price", "price_count", 2),
        aggregate_output("sum", "price", "price_sum", 2),
        aggregate_output("mean", "price", "price_mean", 2),
    ]));
    let rows = vec![
        full_row(1, "a", 1, vec![ScalarValue::Float64(Some(1.0))]),
        full_row(1, "b", 1, vec![ScalarValue::Float64(Some(10.0))]),
        full_row(2, "a", 2, vec![ScalarValue::Float64(Some(2.0))]),
        full_row(2, "b", 2, vec![ScalarValue::Float64(Some(20.0))]),
        full_row(3, "a", 3, vec![ScalarValue::Float64(Some(3.0))]),
        full_row(4, "a", 4, vec![ScalarValue::Float64(Some(4.0))]),
    ];
    let outputs = compute(&spec, &RollingHistories::default(), &rows).unwrap();
    assert_eq!(
        unsigned_column(&outputs, 0),
        vec![Some(1), Some(1), Some(2), Some(2), Some(2), Some(2)]
    );
    assert_eq!(
        float_column(&outputs, 1),
        vec![
            Some(1.0),
            Some(10.0),
            Some(3.0),
            Some(30.0),
            Some(5.0),
            Some(7.0)
        ]
    );
    assert_eq!(
        float_column(&outputs, 2),
        vec![
            Some(1.0),
            Some(10.0),
            Some(1.5),
            Some(15.0),
            Some(2.5),
            Some(3.5)
        ]
    );
}

#[test]
fn null_and_nan_samples_are_excluded_but_rows_still_emit() {
    let spec = kernel_spec(json!([
        aggregate_output("count", "price", "price_count", 2),
        aggregate_output("sum", "price", "price_sum", 2),
        aggregate_output("mean", "price", "price_mean", 2),
    ]));
    let rows = price_rows(&[Some(1.0), None, Some(f64::NAN), Some(4.0)]);
    let outputs = compute(&spec, &RollingHistories::default(), &rows).unwrap();
    assert_eq!(
        unsigned_column(&outputs, 0),
        vec![Some(1), Some(1), None, Some(1)]
    );
    assert_eq!(
        float_column(&outputs, 1),
        vec![Some(1.0), Some(1.0), None, Some(4.0)]
    );
    let means = float_column(&outputs, 2);
    assert_eq!(means[0], Some(1.0));
    assert_eq!(means[1], Some(1.0));
    assert_eq!(means[2], None);
    assert_eq!(means[3], Some(4.0));
}

#[test]
fn min_periods_counts_valid_samples_not_rows() {
    let mut declaration = aggregate_output("mean", "price", "price_mean", 3);
    declaration["min_periods"] = json!(2);
    let spec = kernel_spec(json!([declaration]));
    let rows = price_rows(&[Some(1.0), None, Some(3.0), Some(4.0)]);
    let outputs = compute(&spec, &RollingHistories::default(), &rows).unwrap();
    assert_eq!(
        float_column(&outputs, 0),
        vec![None, None, Some(2.0), Some(3.5)]
    );
}

#[test]
fn variance_and_stddev_follow_the_ddof_divisor() {
    let spec = kernel_spec(json!([
        ddof_output("variance", "price", "price_var_1", 2, 1),
        ddof_output("variance", "price", "price_var_0", 2, 0),
        ddof_output("stddev", "price", "price_std_1", 2, 1),
        ddof_output("stddev", "price", "price_std_0", 2, 0),
    ]));
    let rows = price_rows(&[Some(1.0), Some(2.0), Some(3.0), Some(4.0)]);
    let outputs = compute(&spec, &RollingHistories::default(), &rows).unwrap();
    let sample = float_column(&outputs, 0);
    assert_eq!(sample[0], None);
    assert_eq!(sample[1..], [Some(0.5), Some(0.5), Some(0.5)]);
    assert_eq!(
        float_column(&outputs, 1),
        vec![Some(0.0), Some(0.25), Some(0.25), Some(0.25)]
    );
    let std_sample = float_column(&outputs, 2);
    assert_eq!(std_sample[0], None);
    for value in &std_sample[1..] {
        assert!((value.unwrap() - 0.5_f64.sqrt()).abs() < 1e-15);
    }
    let std_population = float_column(&outputs, 3);
    assert_eq!(std_population[0], Some(0.0));
    for value in &std_population[1..] {
        assert!((value.unwrap() - 0.5).abs() < 1e-15);
    }
}

#[test]
fn integer_sum_is_exact_and_checked() {
    let spec = kernel_spec(json!([aggregate_output("sum", "volume", "volume_sum", 2)]));
    let rows = [10_i64, 20, 30]
        .into_iter()
        .enumerate()
        .map(|(index, volume)| {
            let sequence = u64::try_from(index + 1).unwrap();
            full_row(
                i64::try_from(index + 1).unwrap(),
                "a",
                sequence,
                vec![ScalarValue::Float64(None), ScalarValue::Int64(Some(volume))],
            )
        })
        .collect::<Vec<_>>();
    let outputs = compute(&spec, &RollingHistories::default(), &rows).unwrap();
    assert_eq!(
        signed_column(&outputs, 0),
        vec![Some(10), Some(30), Some(50)]
    );
}

#[test]
fn integer_sum_overflow_is_a_data_error() {
    let spec = kernel_spec(json!([aggregate_output("sum", "volume", "volume_sum", 2)]));
    let rows = vec![
        full_row(
            1,
            "a",
            1,
            vec![
                ScalarValue::Float64(None),
                ScalarValue::Int64(Some(i64::MAX - 1)),
            ],
        ),
        full_row(
            2,
            "a",
            2,
            vec![ScalarValue::Float64(None), ScalarValue::Int64(Some(2))],
        ),
    ];
    let error = compute(&spec, &RollingHistories::default(), &rows).unwrap_err();
    assert!(
        matches!(error, CalcFlowError::Operator { ref message, .. } if message.contains("sum")),
        "unexpected error: {error}"
    );
}

#[test]
fn integer_slide_transient_overflow_returns_representable_sums() {
    // Window sums [MAX], [MAX,-1], [-1,5] are all representable, but the
    // add-before-remove slide transient MAX-1+5 overflows narrow i64.
    let spec = kernel_spec(json!([aggregate_output("sum", "volume", "volume_sum", 2)]));
    let rows = [i64::MAX, -1, 5]
        .into_iter()
        .enumerate()
        .map(|(index, volume)| {
            full_row(
                i64::try_from(index + 1).unwrap(),
                "a",
                u64::try_from(index + 1).unwrap(),
                vec![ScalarValue::Float64(None), ScalarValue::Int64(Some(volume))],
            )
        })
        .collect::<Vec<_>>();
    let outputs = compute(&spec, &RollingHistories::default(), &rows).unwrap();
    assert_eq!(
        signed_column(&outputs, 0),
        vec![Some(i64::MAX), Some(i64::MAX - 1), Some(4)]
    );
}

#[test]
fn unsigned_slide_transient_overflow_returns_representable_sums() {
    let schema = numeric_schema(DataType::UInt64);
    let spec = numeric_spec(json!([aggregate_output("sum", "value", "value_sum", 2)]));
    let compiled = compile_spec(&spec, &schema).unwrap();
    let rows = [u64::MAX, 0, 5]
        .into_iter()
        .enumerate()
        .map(|(index, value)| {
            numeric_row(
                i64::try_from(index + 1).unwrap(),
                u64::try_from(index + 1).unwrap(),
                ScalarValue::UInt64(Some(value)),
            )
        })
        .collect::<Vec<_>>();
    let outputs =
        compute_output_columns(&rows, &RollingHistories::default(), &compiled, "rolling").unwrap();
    assert_eq!(
        unsigned_column(&outputs, 0),
        vec![Some(u64::MAX), Some(u64::MAX), Some(5)]
    );
}

#[test]
fn integer_transient_slide_matches_the_rebuild_fold() {
    let spec = kernel_spec(json!([aggregate_output("sum", "volume", "volume_sum", 2)]));
    let rows = [i64::MAX, -1, 5, 2]
        .into_iter()
        .enumerate()
        .map(|(index, volume)| {
            full_row(
                i64::try_from(index + 1).unwrap(),
                "a",
                u64::try_from(index + 1).unwrap(),
                vec![ScalarValue::Float64(None), ScalarValue::Int64(Some(volume))],
            )
        })
        .collect::<Vec<_>>();
    let one_shot = compute(&spec, &RollingHistories::default(), &rows).unwrap();
    let mut histories = RollingHistories::default();
    let mut segmented: Vec<Option<i64>> = Vec::new();
    for chunk in rows.chunks(3) {
        let outputs = compute(&spec, &histories, chunk).unwrap();
        let values = signed_column(&outputs, 0);
        histories.apply(outputs.touched);
        segmented.extend(values);
    }
    assert_eq!(segmented, signed_column(&one_shot, 0));
}

fn numeric_schema(data_type: DataType) -> Schema {
    Schema::new(vec![
        Field::new(
            "ts",
            DataType::Timestamp(TimeUnit::Microsecond, Some(Arc::from("UTC"))),
            false,
        ),
        Field::new("symbol", DataType::Utf8, false),
        Field::new("sequence", DataType::UInt64, false),
        Field::new("value", data_type, true),
    ])
}

fn numeric_row(event_time: i64, sequence: u64, value: ScalarValue) -> BufferedRow {
    BufferedRow::new(
        vec![Some(KeyValue::String("a".into()))],
        vec![KeyValue::Unsigned(sequence)],
        event_time,
        vec![
            ts_scalar(event_time),
            ScalarValue::Utf8(Some("a".into())),
            ScalarValue::UInt64(Some(sequence)),
            value,
        ],
    )
}

fn numeric_spec(outputs: Value) -> RollingSpec {
    let mut document = aggregate_spec_json(outputs);
    document["partition_by"] = json!(["symbol"]);
    document["event_time"] = json!("ts");
    document["sequence_by"] = json!(["sequence"]);
    serde_json::from_value(document).unwrap()
}

#[test]
fn unsigned_sum_stays_exact_and_checked() {
    let schema = numeric_schema(DataType::UInt64);
    let spec = numeric_spec(json!([aggregate_output("sum", "value", "value_sum", 2)]));
    let compiled = compile_spec(&spec, &schema).unwrap();
    let rows = [5_u64, 7, 9]
        .into_iter()
        .enumerate()
        .map(|(index, value)| {
            let sequence = u64::try_from(index + 1).unwrap();
            numeric_row(
                i64::try_from(index + 1).unwrap(),
                sequence,
                ScalarValue::UInt64(Some(value)),
            )
        })
        .collect::<Vec<_>>();
    let outputs =
        compute_output_columns(&rows, &RollingHistories::default(), &compiled, "rolling").unwrap();
    assert_eq!(
        unsigned_column(&outputs, 0),
        vec![Some(5), Some(12), Some(16)]
    );
    let overflow = vec![
        numeric_row(1, 1, ScalarValue::UInt64(Some(u64::MAX))),
        numeric_row(2, 2, ScalarValue::UInt64(Some(1))),
    ];
    let error = compute_output_columns(
        &overflow,
        &RollingHistories::default(),
        &compiled,
        "rolling",
    )
    .unwrap_err();
    assert!(
        matches!(error, CalcFlowError::Operator { ref message, .. } if message.contains("sum")),
        "unexpected error: {error}"
    );
}

#[test]
fn float32_samples_widen_to_float64_outputs() {
    let schema = numeric_schema(DataType::Float32);
    let spec = numeric_spec(json!([
        aggregate_output("sum", "value", "value_sum", 2),
        aggregate_output("mean", "value", "value_mean", 2),
    ]));
    let compiled = compile_spec(&spec, &schema).unwrap();
    let rows = [1.5_f32, 2.5, 4.0]
        .into_iter()
        .enumerate()
        .map(|(index, value)| {
            let sequence = u64::try_from(index + 1).unwrap();
            numeric_row(
                i64::try_from(index + 1).unwrap(),
                sequence,
                ScalarValue::Float32(Some(value)),
            )
        })
        .collect::<Vec<_>>();
    let outputs =
        compute_output_columns(&rows, &RollingHistories::default(), &compiled, "rolling").unwrap();
    assert_eq!(
        float_column(&outputs, 0),
        vec![Some(1.5), Some(4.0), Some(6.5)]
    );
    assert_eq!(
        float_column(&outputs, 1),
        vec![Some(1.5), Some(2.0), Some(3.25)]
    );
}

#[test]
fn infinities_follow_ieee_and_undefined_results_are_nan_not_null() {
    let spec = kernel_spec(json!([
        aggregate_output("sum", "price", "price_sum", 2),
        ddof_output("variance", "price", "price_var", 2, 1),
    ]));
    let rows = price_rows(&[Some(1.0), Some(f64::INFINITY), Some(3.0), Some(4.0)]);
    let outputs = compute(&spec, &RollingHistories::default(), &rows).unwrap();
    let sums = float_column(&outputs, 0);
    assert_eq!(sums[0], Some(1.0));
    assert_eq!(sums[1], Some(f64::INFINITY));
    assert_eq!(sums[2], Some(f64::INFINITY));
    assert_eq!(sums[3], Some(7.0));
    let variances = float_column(&outputs, 1);
    assert_eq!(variances[0], None);
    assert!(variances[1].unwrap().is_nan());
    assert!(variances[2].unwrap().is_nan());
    assert_eq!(variances[3], Some(0.5));
}

// ------------------------------------------------------------------
// Frozen ±inf readout semantics (SCE-07 defect 1 ruling, A1-A6)
// ------------------------------------------------------------------

/// Rows for one entity with explicit per-row price values.
fn entity_prices(symbol: &str, prices: &[Option<f64>]) -> Vec<BufferedRow> {
    prices
        .iter()
        .enumerate()
        .map(|(index, price)| {
            let sequence = u64::try_from(index + 1).unwrap();
            full_row(
                i64::try_from(index + 1).unwrap(),
                symbol,
                sequence,
                vec![ScalarValue::Float64(*price)],
            )
        })
        .collect()
}

#[test]
fn a1_mean_classification_is_independent_of_infinity_position() {
    let spec = kernel_spec(json!([aggregate_output("mean", "price", "price_mean", 3)]));
    let mut rows = Vec::new();
    for (symbol, prices) in [
        ("a", [f64::INFINITY, 1.0, 2.0]),
        ("b", [1.0, f64::INFINITY, 2.0]),
        ("c", [1.0, 2.0, f64::INFINITY]),
        ("d", [f64::NEG_INFINITY, 1.0, 2.0]),
        ("e", [1.0, f64::NEG_INFINITY, 2.0]),
        ("f", [1.0, 2.0, f64::NEG_INFINITY]),
    ] {
        rows.extend(entity_prices(symbol, &prices.map(Some)));
    }
    let outputs = compute(&spec, &RollingHistories::default(), &rows).unwrap();
    let means = float_column(&outputs, 0);
    for index in [2_usize, 5, 8] {
        assert_eq!(
            means[index],
            Some(f64::INFINITY),
            "positive infinity multiset at row {index}"
        );
    }
    for index in [11_usize, 14, 17] {
        assert_eq!(
            means[index],
            Some(f64::NEG_INFINITY),
            "negative infinity multiset at row {index}"
        );
    }
}

#[test]
fn a2_mixed_sign_infinities_yield_nan_in_any_order() {
    let spec = kernel_spec(json!([aggregate_output("mean", "price", "price_mean", 3)]));
    let mut rows = Vec::new();
    rows.extend(entity_prices(
        "a",
        &[Some(f64::INFINITY), Some(f64::NEG_INFINITY)],
    ));
    rows.extend(entity_prices(
        "b",
        &[Some(f64::NEG_INFINITY), Some(f64::INFINITY)],
    ));
    rows.extend(entity_prices(
        "c",
        &[
            Some(f64::INFINITY),
            Some(f64::INFINITY),
            Some(f64::NEG_INFINITY),
        ],
    ));
    let outputs = compute(&spec, &RollingHistories::default(), &rows).unwrap();
    let means = float_column(&outputs, 0);
    for index in [1_usize, 3, 6] {
        assert!(
            means[index].is_some_and(f64::is_nan),
            "mixed-sign window at row {index} must be NaN"
        );
    }
}

#[test]
fn a3_departed_infinities_leave_no_residue_across_slide_and_refold() {
    let spec = kernel_spec(json!([aggregate_output("mean", "price", "price_mean", 2)]));
    let rows = entity_prices(
        "a",
        &[
            Some(1.0),
            Some(f64::INFINITY),
            Some(3.0),
            Some(4.0),
            Some(5.0),
        ],
    );
    let one_shot = compute(&spec, &RollingHistories::default(), &rows).unwrap();
    assert_eq!(
        float_column(&one_shot, 0),
        vec![
            Some(1.0),
            Some(f64::INFINITY),
            Some(f64::INFINITY),
            Some(3.5),
            Some(4.5)
        ]
    );

    let mut histories = RollingHistories::default();
    let mut segmented: Vec<Option<f64>> = Vec::new();
    let compiled = compile_spec(&spec, &kernel_schema()).unwrap();
    for chunk in rows.chunks(3) {
        let outputs = compute(&spec, &histories, chunk).unwrap();
        let values = float_column(&outputs, 0);
        histories.apply(outputs.touched);
        // Mirror a checkpoint restore: rebuild every window from the
        // retained rows instead of carrying the live accumulators.
        rebuild_windows(&mut histories, &compiled, "rolling").unwrap();
        segmented.extend(values);
    }
    assert_eq!(segmented, float_column(&one_shot, 0));
}

#[test]
fn negative_infinity_departure_restores_finite_window_outputs() {
    let spec = kernel_spec(json!([aggregate_output("mean", "price", "price_mean", 2)]));
    let rows = entity_prices("a", &[Some(f64::NEG_INFINITY), Some(2.0), Some(5.0)]);
    let outputs = compute(&spec, &RollingHistories::default(), &rows).unwrap();

    assert_eq!(
        float_column(&outputs, 0),
        vec![Some(f64::NEG_INFINITY), Some(f64::NEG_INFINITY), Some(3.5)]
    );
}

#[test]
fn a4_near_overflow_finite_window_keeps_the_west_mean() {
    let spec = kernel_spec(json!([aggregate_output("mean", "price", "price_mean", 2)]));
    let rows = entity_prices("a", &[Some(1e308), Some(1e308)]);
    let outputs = compute(&spec, &RollingHistories::default(), &rows).unwrap();
    let means = float_column(&outputs, 0);
    let value = means[1].expect("finite window mean must be non-null");
    assert!(
        value.is_finite(),
        "naive sum/count readout would overflow to infinity"
    );
    assert!((value - 1e308).abs() <= 1e308 * 1e-15);
}

#[test]
fn a5_variance_and_stddev_with_inf_are_nan_after_the_null_gates() {
    let spec = kernel_spec(json!([
        ddof_output("variance", "price", "price_var_1", 2, 1),
        ddof_output("stddev", "price", "price_std_0", 2, 0),
    ]));
    let rows = entity_prices("a", &[Some(f64::INFINITY), Some(2.0), Some(5.0), Some(6.0)]);
    let outputs = compute(&spec, &RollingHistories::default(), &rows).unwrap();
    let variances = float_column(&outputs, 0);
    // ddof=1 with one valid sample: divisor zero wins over the NaN class.
    assert_eq!(variances[0], None);
    // Two valid samples with an inf present: NaN, not null and not inf.
    assert!(variances[1].unwrap().is_nan());
    // The inf has left the window: back to finite values (the removal
    // step carries West drift well inside the frozen D13 tolerance).
    assert!((variances[2].unwrap() - 4.5).abs() <= 4.5 * 1e-10);
    assert!((variances[3].unwrap() - 0.5).abs() <= 1e-12);
    let stddevs = float_column(&outputs, 1);
    // ddof=0 passes the divisor gate with one sample: NaN classification.
    assert!(stddevs[0].unwrap().is_nan());
    assert!(stddevs[1].unwrap().is_nan());
    assert!((stddevs[2].unwrap() - 1.5_f64).abs() <= 1e-12);
    assert!((stddevs[3].unwrap() - 0.5_f64).abs() <= 1e-12);
}

#[test]
fn a6_infinities_count_toward_valid_samples_and_min_periods() {
    let spec = kernel_spec(json!([
        aggregate_output("count", "price", "price_count", 3),
        aggregate_output("mean", "price", "price_mean", 3),
    ]));
    let rows = entity_prices("a", &[Some(f64::INFINITY), Some(f64::NAN), None]);
    let outputs = compute(&spec, &RollingHistories::default(), &rows).unwrap();
    // NaN and null are excluded, so the valid count stays at one; the
    // infinity alone still satisfies min_periods=1.
    assert_eq!(
        unsigned_column(&outputs, 0),
        vec![Some(1), Some(1), Some(1)]
    );
    let means = float_column(&outputs, 1);
    assert_eq!(means[0], Some(f64::INFINITY));
    assert_eq!(means[1], Some(f64::INFINITY));
    assert_eq!(means[2], Some(f64::INFINITY));
}

#[test]
fn compatible_outputs_share_one_window_state() {
    let spec = kernel_spec(json!([
        aggregate_output("count", "price", "price_count", 2),
        aggregate_output("sum", "price", "price_sum", 2),
        aggregate_output("mean", "price", "price_mean", 2),
        ddof_output("variance", "price", "price_var", 2, 1),
        ddof_output("stddev", "price", "price_std", 2, 1),
        aggregate_output("mean", "price", "price_mean_3", 3),
    ]));
    let compiled = compile_spec(&spec, &kernel_schema()).unwrap();
    assert_eq!(compiled.window_groups.len(), 2);
    let rows = price_rows(&[Some(1.0), Some(2.0), Some(3.0), Some(4.0)]);
    let outputs = compute(&spec, &RollingHistories::default(), &rows).unwrap();
    let variances = float_column(&outputs, 3);
    let stddevs = float_column(&outputs, 4);
    let means_3 = float_column(&outputs, 5);
    for (variance, stddev) in variances.iter().zip(&stddevs) {
        match (variance, stddev) {
            (Some(variance), Some(stddev)) => {
                assert!((stddev * stddev - variance).abs() < 1e-12);
            }
            (None, None) => {}
            other => panic!("variance/stddev nullness diverged: {other:?}"),
        }
    }
    assert_eq!(means_3, vec![Some(1.0), Some(1.5), Some(2.0), Some(3.0)]);
}

#[test]
fn frame_size_extends_history_retention_beyond_lag_periods() {
    let spec = kernel_spec(json!([
        lag_price(1),
        aggregate_output("mean", "price", "price_mean", 3),
    ]));
    let mut histories = RollingHistories::default();
    for batch in 0..3_u32 {
        let rows = (0..4_u32)
            .map(|index| {
                let sequence = batch * 4 + index + 1;
                full_row(
                    i64::from(sequence),
                    "a",
                    u64::from(sequence),
                    vec![ScalarValue::Float64(Some(f64::from(sequence)))],
                )
            })
            .collect::<Vec<_>>();
        let outputs = compute(&spec, &histories, &rows).unwrap();
        histories.apply(outputs.touched);
    }
    for state in histories.by_entity.values() {
        assert!(state.rows.len() <= 3);
        assert_eq!(state.rows.len(), 3);
    }
}

#[test]
fn aggregate_windows_survive_segmentation() {
    let spec = kernel_spec(json!([
        aggregate_output("sum", "price", "price_sum", 2),
        ddof_output("variance", "price", "price_var", 2, 1),
    ]));
    let all_rows = price_rows(&[Some(1.0), Some(2.0), Some(3.0), Some(4.0), Some(5.0)]);
    let one_shot = compute(&spec, &RollingHistories::default(), &all_rows).unwrap();
    let mut histories = RollingHistories::default();
    let mut segmented: Vec<Vec<Option<f64>>> = vec![Vec::new(); 2];
    for chunk in all_rows.chunks(2) {
        let outputs = compute(&spec, &histories, chunk).unwrap();
        histories.apply(outputs.touched);
        for (index, column) in outputs.columns.iter().enumerate() {
            let values = column
                .as_any()
                .downcast_ref::<Float64Array>()
                .unwrap()
                .iter()
                .collect::<Vec<_>>();
            segmented[index].extend(values);
        }
    }
    for (index, column) in one_shot.columns.iter().enumerate() {
        let expected = column
            .as_any()
            .downcast_ref::<Float64Array>()
            .unwrap()
            .iter()
            .collect::<Vec<_>>();
        assert_eq!(segmented[index], expected);
    }
}

#[test]
fn failed_aggregate_leaves_histories_untouched() {
    let spec = kernel_spec(json!([aggregate_output("sum", "volume", "volume_sum", 2)]));
    let histories = RollingHistories::default();
    let rows = vec![
        full_row(
            1,
            "a",
            1,
            vec![
                ScalarValue::Float64(None),
                ScalarValue::Int64(Some(i64::MAX)),
            ],
        ),
        full_row(
            2,
            "a",
            2,
            vec![ScalarValue::Float64(None), ScalarValue::Int64(Some(1))],
        ),
    ];
    assert!(compute(&spec, &histories, &rows).is_err());
    assert!(histories.by_entity.is_empty());
}

// ------------------------------------------------------------------
// Operator construction and metadata
// ------------------------------------------------------------------

#[test]
fn operator_exposes_exact_ports_and_frozen_configuration() {
    let operator =
        RollingOperator::new("rolling_features", Arc::new(input_schema()), valid_spec()).unwrap();
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
        RollingOperator::new("rolling_features", Arc::new(input_schema()), spec.clone()).unwrap();
    assert_eq!(operator.spec(), &spec);
    let rendered = format!("{operator:?}");
    assert!(rendered.contains("RollingOperator"));
    assert!(rendered.contains("rolling_features"));
}

#[tokio::test]
async fn emission_chunks_by_edge_budget_and_oversize_rows_fail() {
    use crate::{CancellationToken, EdgeBudget, IngressProgressSnapshot, StreamJobContext};

    struct NoopLateMetrics;
    impl crate::operator::LateMetricSink for NoopLateMetrics {
        fn record(&self, _delta: LateMetricDelta) -> Result<()> {
            Ok(())
        }
    }

    fn matrix_record(times: Vec<i64>, prices: Vec<Option<f64>>) -> RecordBatch {
        let len = times.len();
        RecordBatch::try_new(
            Arc::new(input_schema()),
            vec![
                Arc::new(
                    datafusion::arrow::array::TimestampMicrosecondArray::from(times)
                        .with_timezone("UTC"),
                ) as ArrayRef,
                Arc::new(datafusion::arrow::array::StringArray::from(vec!["a"; len])) as ArrayRef,
                Arc::new(UInt64Array::from((1..=len as u64).collect::<Vec<_>>())),
                Arc::new(Float64Array::from(prices)),
                Arc::new(datafusion::arrow::array::Int64Array::from(
                    (1..=len as u64)
                        .map(|v| Some(i64::try_from(v).unwrap() * 10))
                        .collect::<Vec<_>>(),
                )),
                Arc::new(datafusion::arrow::array::StringArray::from(vec!["x"; len])),
            ],
        )
        .unwrap()
    }

    let job = StreamJobContext::new(
        7,
        "0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef",
        JsonMap::new(),
        None,
        CancellationToken::new(),
    );
    let budget = EdgeBudget::new(2, EdgeBudget::MAX_BYTES).unwrap();
    let context = StreamOperatorContext::for_task(
        &job,
        "rolling",
        None,
        IngressProgressSnapshot::default(),
        budget,
        Arc::new(NoopLateMetrics),
    );
    let mut operator =
        RollingOperator::new("rolling", Arc::new(input_schema()), valid_spec()).unwrap();
    let mut collector = crate::EdgeCollector::new(operator.output_ports().to_vec());
    let record = matrix_record(vec![10, 11, 12], vec![Some(1.0), Some(2.0), Some(3.0)]);
    let batch = Batch::table(vec![record], BatchMetadata::default()).unwrap();
    operator
        .process_data("input", batch, &context, &mut collector)
        .await
        .unwrap();
    operator
        .on_watermark(EventTime::from_micros(20), &context, &mut collector)
        .await
        .unwrap();
    let emitted = collector.drain("output");
    assert_eq!(emitted.len(), 2);
    assert_eq!(emitted[0].as_data().unwrap().metadata().sequence(), 0);
    assert_eq!(emitted[1].as_data().unwrap().metadata().sequence(), 1);

    let tiny = EdgeBudget::new(10, 1).unwrap();
    let context = StreamOperatorContext::for_task(
        &job,
        "rolling",
        None,
        IngressProgressSnapshot::default(),
        tiny,
        Arc::new(NoopLateMetrics),
    );
    let mut operator =
        RollingOperator::new("rolling", Arc::new(input_schema()), valid_spec()).unwrap();
    let mut collector = crate::EdgeCollector::new(operator.output_ports().to_vec());
    let record = matrix_record(vec![10], vec![Some(1.0)]);
    let batch = Batch::table(vec![record], BatchMetadata::default()).unwrap();
    operator
        .process_data("input", batch, &context, &mut collector)
        .await
        .unwrap();
    let error = operator
        .on_watermark(EventTime::from_micros(20), &context, &mut collector)
        .await
        .unwrap_err();
    assert!(matches!(
            error,
            CalcFlowError::InvalidArgument { ref field, .. } if field == "message.bytes"
    ));
}
