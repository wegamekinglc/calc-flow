use std::sync::Arc;

use crate::{CalcFlowError, OperatorMetadata, RollingOperator, RollingSpec};
use datafusion::arrow::datatypes::{DataType, Field, Schema, TimeUnit};
use serde_json::json;

use super::SafeIndicatorDiagnostic;

pub(crate) fn operator(indicator: bool) -> RollingOperator {
    let output = if indicator {
        json!({"kind":"atr","primitive_version":1,"high":"high","low":"low","close":"close","output":"atr","period":1})
    } else {
        json!({"kind":"ewma","primitive_version":1,"input":"high","output":"ema","span":2,"min_periods":1})
    };
    let spec: RollingSpec = serde_json::from_value(json!({
        "configuration_version":1,"state_layout_version":if indicator {5} else {2},
        "partition_by":["entity"],"event_time":"ts","sequence_by":["seq"],
        "outputs":[output],"allowed_lateness_micros":0,
        "late_policy":{"kind":"error","scope":"envelope"},"value_policy":"stateful_numeric_v1"
    }))
    .unwrap();
    let schema = Arc::new(Schema::new(vec![
        Field::new(
            "ts",
            DataType::Timestamp(TimeUnit::Microsecond, Some("UTC".into())),
            false,
        ),
        Field::new("entity", DataType::Utf8, false),
        Field::new("seq", DataType::UInt64, false),
        Field::new("high", DataType::Float64, true),
        Field::new("low", DataType::Float64, true),
        Field::new("close", DataType::Float64, true),
    ]));
    RollingOperator::new("features", schema, spec).unwrap()
}

fn numerical_error(field: &str, reason: &str) -> CalcFlowError {
    CalcFlowError::InvalidArgument {
        field: field.into(),
        message: format!("rolling node \"features\", ordered position 0: {reason}"),
    }
}

#[tokio::test]
async fn test_actual_indicator_domain_error_keeps_field_position_without_samples() {
    use crate::{
        Batch, BatchMetadata, CancellationToken, EdgeCollector, JsonMap, OperatorMetadata,
        StreamJobContext, StreamOperator, StreamOperatorContext,
    };
    use datafusion::arrow::{
        array::{Float64Array, StringArray, TimestampMicrosecondArray, UInt64Array},
        record_batch::RecordBatch,
    };
    let mut operator = operator(true);
    let record = RecordBatch::try_from_iter(vec![
        (
            "ts",
            Arc::new(TimestampMicrosecondArray::from(vec![1]).with_timezone("UTC")) as _,
        ),
        (
            "entity",
            Arc::new(StringArray::from(vec!["secret-entity-value"])) as _,
        ),
        ("seq", Arc::new(UInt64Array::from(vec![1])) as _),
        ("high", Arc::new(Float64Array::from(vec![4321.0])) as _),
        ("low", Arc::new(Float64Array::from(vec![5432.0])) as _),
        ("close", Arc::new(Float64Array::from(vec![5000.0])) as _),
    ])
    .unwrap();
    let record = RecordBatch::try_new(
        Arc::clone(operator.input_ports()[0].schema().unwrap()),
        record.columns().to_vec(),
    )
    .unwrap();
    let batch = Batch::table(vec![record], BatchMetadata::default()).unwrap();
    let job = StreamJobContext::new(
        1,
        "a".repeat(64),
        JsonMap::new(),
        None,
        CancellationToken::new(),
    );
    let context = StreamOperatorContext::new(&job, "features", None);
    let mut output = EdgeCollector::new(operator.output_ports().to_vec());
    let error = operator
        .process_data("input", batch, &context, &mut output)
        .await
        .unwrap_err();
    let diagnostic = SafeIndicatorDiagnostic::from_rolling("features", &operator, &error)
        .unwrap_or_else(|| panic!("actual built-in error: {error:?}"));
    let message = diagnostic.message();
    assert!(message.contains("high"));
    assert!(message.contains("input row 0; ordered position not assigned"));
    assert!(!message.contains("ordered position 0"));
    assert!(message.contains("greater than or equal"));
    for value in ["secret-entity-value", "4321", "5432", "5000"] {
        assert!(!message.contains(value));
    }
    assert!(output.drain("output").is_empty());
}

#[test]
fn test_legacy_and_unrecognized_error_details_do_not_get_indicator_evidence() {
    let error = numerical_error("high", "must be greater than or equal to low");
    assert!(SafeIndicatorDiagnostic::from_rolling("features", &operator(false), &error).is_none());
    let legacy = operator(false);
    let mut old_only = legacy.spec().clone();
    old_only.state_layout_version = 5;
    let old_only = RollingOperator::new(
        "features",
        Arc::clone(legacy.input_ports()[0].schema().unwrap()),
        old_only,
    )
    .unwrap();
    assert!(SafeIndicatorDiagnostic::from_rolling("features", &old_only, &error).is_none());
    let actual = operator(true);
    for error in [
        numerical_error("entity", "RSI/ATR requires finite non-missing samples"),
        numerical_error("high", "secret-payload-123"),
        CalcFlowError::Operator { node_id: "wrong-node".into(), message: "high: rolling node \"features\", ordered position 0: must be greater than or equal to low".into() },
    ] { assert!(SafeIndicatorDiagnostic::from_rolling("features", &actual, &error).is_none()); }
}

#[test]
fn test_safe_numerical_reason_and_operator_wrapping_share_the_same_evidence() {
    let actual = operator(true);
    let direct = numerical_error(
        "high",
        "ATR high - low is not representable as finite float64",
    );
    let wrapped = CalcFlowError::Operator { node_id: "features".into(), message: "high: rolling node \"features\", ordered position 0: ATR high - low is not representable as finite float64".into() };
    let direct = SafeIndicatorDiagnostic::from_rolling("features", &actual, &direct).unwrap();
    let wrapped = SafeIndicatorDiagnostic::from_rolling("features", &actual, &wrapped).unwrap();
    assert_eq!(direct, wrapped);
    assert_eq!(direct.node_id(), "features");
    assert!(
        direct
            .message()
            .contains("not representable as finite float64")
    );
}

#[test]
fn test_noncanonical_positions_and_secret_suffixes_are_not_diagnostic_evidence() {
    let actual = operator(true);
    for message in [
        "rolling node \"features\", ordered position +0: must be greater than or equal to low",
        "rolling node \"features\", ordered position 00: must be greater than or equal to low",
        "rolling node \"features\", ordered position 0: must be greater than or equal to low secret-value",
        "rolling node \"features\", pending row 0; ordered position not assigned: must be greater than or equal to low",
        "rolling node \"features\", accepted row 0; ordered position not assigned: must be greater than or equal to low",
    ] {
        let error = CalcFlowError::InvalidArgument {
            field: "high".into(),
            message: message.into(),
        };
        assert!(SafeIndicatorDiagnostic::from_rolling("features", &actual, &error).is_none());
    }
}

#[test]
fn test_prebuffer_input_row_is_kept_without_inventing_an_ordered_position() {
    let error = CalcFlowError::InvalidArgument {
        field: "high".into(),
        message: r#"rolling node "features", input row 19; ordered position not assigned: must be greater than or equal to low"#.into(),
    };
    let diagnostic =
        SafeIndicatorDiagnostic::from_rolling("features", &operator(true), &error).unwrap();
    assert!(
        diagnostic
            .message()
            .contains("input row 19; ordered position not assigned")
    );
    assert!(!diagnostic.message().contains("ordered position 19"));
}

fn layout_six_operator(mixed: bool) -> RollingOperator {
    let base = operator(true);
    let mut spec = base.spec().clone();
    spec.state_layout_version = crate::ROLLING_MACD_STATE_LAYOUT_VERSION;
    if !mixed {
        spec.outputs.clear();
    }
    spec.outputs.push(
        serde_json::from_value(json!({
            "kind":"macd_component", "primitive_version":1,
            "input":"close", "output":"line", "component":"line"
        }))
        .unwrap(),
    );
    RollingOperator::new(
        "features",
        Arc::clone(base.input_ports()[0].schema().unwrap()),
        spec,
    )
    .unwrap()
}

#[test]
fn test_layout_six_keeps_mixed_atr_input_coordinate_evidence() {
    let error = CalcFlowError::InvalidArgument {
        field: "high".into(),
        message: r#"rolling node "features", input row 2; ordered position not assigned: must be greater than or equal to low"#.into(),
    };
    let diagnostic =
        SafeIndicatorDiagnostic::from_rolling("features", &layout_six_operator(true), &error)
            .expect("mixed layout-six ATR diagnostic");
    assert!(diagnostic.message().contains("high"));
    assert!(
        diagnostic
            .message()
            .contains("input row 2; ordered position not assigned")
    );
}

#[test]
fn test_layout_six_keeps_macd_position_counter_evidence() {
    let error = numerical_error(
        "input.rows",
        "processed position count would exceed u64::MAX",
    );
    let diagnostic =
        SafeIndicatorDiagnostic::from_rolling("features", &layout_six_operator(false), &error)
            .expect("native MACD position counter diagnostic");
    assert!(diagnostic.message().contains("input.rows"));
    assert!(diagnostic.message().contains("ordered position 0"));
    assert!(diagnostic.message().contains("u64::MAX"));
}

#[test]
fn test_layout_six_does_not_trust_unknown_macd_error_details() {
    let operator = layout_six_operator(false);
    for error in [
        numerical_error("close", "secret-value"),
        numerical_error("input.rows", "secret-value"),
        numerical_error(
            "input.rows",
            "processed position count would exceed u64::MAX secret",
        ),
    ] {
        assert!(SafeIndicatorDiagnostic::from_rolling("features", &operator, &error).is_none());
    }
}
