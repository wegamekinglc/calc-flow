//! Indicator side output retains the existing lateness and delivery policy.

use std::sync::Arc;

use async_trait::async_trait;
use calc_flow::{
    Batch, BatchMetadata, CalcFlowError, CancellationToken, EdgeCollector, EventTime, JsonMap,
    OperatorMetadata, Result, RollingOperator, RollingSpec, StreamCollector, StreamJobContext,
    StreamOperator, StreamOperatorContext,
};
use datafusion::arrow::{
    array::{ArrayRef, Float64Array, StringArray, TimestampMicrosecondArray, UInt64Array},
    datatypes::{DataType, Field, Schema, TimeUnit},
    record_batch::RecordBatch,
};
use serde_json::json;

fn schema() -> Arc<Schema> {
    Arc::new(Schema::new(vec![
        Field::new("entity", DataType::Utf8, false),
        Field::new(
            "ts",
            DataType::Timestamp(TimeUnit::Microsecond, Some("UTC".into())),
            false,
        ),
        Field::new("seq", DataType::UInt64, false),
        Field::new("price", DataType::Float64, false),
    ]))
}
fn input(rows: &[(u32, f64)]) -> Batch {
    let columns: Vec<ArrayRef> = vec![
        Arc::new(StringArray::from(vec!["a"; rows.len()])),
        Arc::new(
            TimestampMicrosecondArray::from(
                rows.iter()
                    .map(|(time, _)| i64::from(*time))
                    .collect::<Vec<_>>(),
            )
            .with_timezone("UTC"),
        ),
        Arc::new(UInt64Array::from(
            rows.iter()
                .map(|(time, _)| u64::from(*time))
                .collect::<Vec<_>>(),
        )),
        Arc::new(Float64Array::from(
            rows.iter().map(|(_, value)| *value).collect::<Vec<_>>(),
        )),
    ];
    Batch::table(
        vec![RecordBatch::try_new(schema(), columns).unwrap()],
        BatchMetadata::default(),
    )
    .unwrap()
}
fn operator() -> RollingOperator {
    let spec: RollingSpec = serde_json::from_value(json!({
        "configuration_version":1,"state_layout_version":5,
        "partition_by":["entity"],"event_time":"ts","sequence_by":["seq"],
        "outputs":[{"kind":"rsi","primitive_version":1,"input":"price","output":"rsi","period":2}],
        "allowed_lateness_micros":0,
        "late_policy":{"kind":"side_output","metrics_version":1,"schema_version":1},
        "value_policy":"stateful_numeric_v1"
    }))
    .unwrap();
    RollingOperator::new("features", schema(), spec).unwrap()
}
fn values(batch: &Batch, name: &str) -> Vec<Option<f64>> {
    batch
        .table_payload()
        .unwrap()
        .batches()
        .iter()
        .flat_map(|record| {
            record
                .column_by_name(name)
                .unwrap()
                .as_any()
                .downcast_ref::<Float64Array>()
                .unwrap()
                .iter()
        })
        .collect()
}

#[tokio::test]
async fn test_indicator_side_output_routes_infinite_late_price_without_advancing_state() {
    let mut operator = operator();
    let job = StreamJobContext::new(
        21,
        "a".repeat(64),
        JsonMap::new(),
        None,
        CancellationToken::new(),
    );
    let initial = StreamOperatorContext::new(&job, "features", None);
    let mut output = EdgeCollector::new(operator.output_ports().to_vec());
    operator
        .process_data(
            "input",
            input(&[(10, 1.0), (11, 2.0)]),
            &initial,
            &mut output,
        )
        .await
        .unwrap();
    operator
        .on_watermark(EventTime::from_micros(11), &initial, &mut output)
        .await
        .unwrap();
    assert_eq!(output.drain("output").len(), 1);
    let context = StreamOperatorContext::new(&job, "features", Some(EventTime::from_micros(11)));
    operator
        .process_data(
            "input",
            input(&[(5, f64::INFINITY), (12, 1.0)]),
            &context,
            &mut output,
        )
        .await
        .unwrap();
    let late = output.drain("late");
    assert_eq!(late.len(), 1);
    let late = late[0].as_data().unwrap();
    assert_eq!(late.metadata().sequence(), 0);
    assert_eq!(values(late, "price"), [Some(f64::INFINITY)]);
    operator
        .on_watermark(EventTime::from_micros(12), &context, &mut output)
        .await
        .unwrap();
    let main = output.drain("output");
    assert_eq!(main.len(), 1);
    assert_eq!(values(main[0].as_data().unwrap(), "rsi"), [Some(50.0)]);
}

struct RejectLate;
#[async_trait]
impl StreamCollector for RejectLate {
    async fn emit(&mut self, port: &str, _: Batch) -> Result<()> {
        assert_eq!(port, "late");
        Err(CalcFlowError::Operator {
            node_id: "collector".into(),
            message: "late emission rejected".into(),
        })
    }
}

#[tokio::test]
async fn test_indicator_late_emitter_failure_poison_is_cleared_only_by_reset() {
    let mut operator = operator();
    let job = StreamJobContext::new(
        22,
        "a".repeat(64),
        JsonMap::new(),
        None,
        CancellationToken::new(),
    );
    let context = StreamOperatorContext::new(&job, "features", Some(EventTime::from_micros(11)));
    let error = operator
        .process_data("input", input(&[(5, 0.0)]), &context, &mut RejectLate)
        .await
        .unwrap_err();
    assert!(
        error.to_string().contains("late emission rejected"),
        "{error}"
    );
    let mut output = EdgeCollector::new(operator.output_ports().to_vec());
    assert!(
        operator
            .process_data("input", input(&[(12, 1.0)]), &context, &mut output)
            .await
            .is_err()
    );
    assert!(output.drain("output").is_empty());
    assert!(output.drain("late").is_empty());
    StreamOperator::reset(&mut operator).unwrap();
    operator
        .process_data("input", input(&[(12, 1.0)]), &context, &mut output)
        .await
        .unwrap();
    operator.on_end(&context, &mut output).await.unwrap();
    let main = output.drain("output");
    assert_eq!(main.len(), 1);
    assert_eq!(values(main[0].as_data().unwrap(), "rsi"), [None]);
}

#[tokio::test]
async fn test_indicator_side_output_preparation_error_neither_emits_nor_poisons() {
    let mut operator = operator();
    let job = StreamJobContext::new(
        23,
        "a".repeat(64),
        JsonMap::new(),
        None,
        CancellationToken::new(),
    );
    let context = StreamOperatorContext::new(&job, "features", Some(EventTime::from_micros(11)));
    let mut output = EdgeCollector::new(operator.output_ports().to_vec());
    let error = operator
        .process_data(
            "input",
            input(&[(5, 0.0), (12, f64::INFINITY)]),
            &context,
            &mut output,
        )
        .await
        .unwrap_err();
    assert!(error.to_string().contains("finite non-missing"), "{error}");
    assert!(output.drain("late").is_empty());
    operator
        .process_data(
            "input",
            input(&[(5, 0.0), (12, 1.0)]),
            &context,
            &mut output,
        )
        .await
        .unwrap();
    assert_eq!(output.drain("late").len(), 1);
    operator.on_end(&context, &mut output).await.unwrap();
    let main = output.drain("output");
    assert_eq!(main.len(), 1);
    assert_eq!(values(main[0].as_data().unwrap(), "rsi"), [None]);
}
