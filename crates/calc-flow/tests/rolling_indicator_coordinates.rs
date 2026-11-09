//! Admission rows and finalized positions have distinct diagnostic coordinates.

use std::{collections::BTreeMap, sync::Arc};

use calc_flow::{
    Batch, BatchMetadata, BatchOperator, BatchOperatorContext, CalcFlowError, CancellationToken,
    EdgeCollector, EventTime, JsonMap, OperatorMetadata, RollingOperator, RollingSpec, RunContext,
    StreamJobContext, StreamOperator, StreamOperatorContext,
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
        Field::new("high", DataType::Float64, false),
        Field::new("low", DataType::Float64, false),
        Field::new("close", DataType::Float64, true),
    ]))
}

fn operator() -> RollingOperator {
    let spec: RollingSpec = serde_json::from_value(json!({
        "configuration_version":1,"state_layout_version":5,
        "partition_by":["entity"],"event_time":"ts","sequence_by":["seq"],
        "outputs":[
            {"kind":"atr","primitive_version":1,"high":"high","low":"low","close":"close","output":"atr","period":1},
            {"kind":"rsi","primitive_version":1,"input":"close","output":"rsi","period":1}
        ],
        "allowed_lateness_micros":0,"late_policy":{"kind":"error","scope":"envelope"},
        "value_policy":"stateful_numeric_v1"
    })).unwrap();
    RollingOperator::new("features", schema(), spec).unwrap()
}

fn input(times: &[i64], high: &[f64], low: &[f64], close: &[Option<f64>]) -> Batch {
    let columns: Vec<ArrayRef> = vec![
        Arc::new(StringArray::from(vec!["private-entity"; times.len()])),
        Arc::new(TimestampMicrosecondArray::from(times.to_vec()).with_timezone("UTC")),
        Arc::new(UInt64Array::from(
            times
                .iter()
                .map(|time| u64::try_from(*time).unwrap())
                .collect::<Vec<_>>(),
        )),
        Arc::new(Float64Array::from(high.to_vec())),
        Arc::new(Float64Array::from(low.to_vec())),
        Arc::new(Float64Array::from(close.to_vec())),
    ];
    Batch::table(
        vec![RecordBatch::try_new(schema(), columns).unwrap()],
        BatchMetadata::default(),
    )
    .unwrap()
}

fn assert_input_row(error: &CalcFlowError, row: usize) {
    let CalcFlowError::InvalidArgument { field, message } = error else {
        panic!("{error}");
    };
    assert_eq!(field, "high");
    assert_eq!(message, &input_message(row));
    assert!(!message.contains("private-entity"));
}

fn input_message(row: usize) -> String {
    format!(
        "rolling node \"features\", input row {row}; ordered position not assigned: must be greater than or equal to low"
    )
}

#[tokio::test]
async fn test_finite_reverse_envelope_reports_input_row_before_ordering() {
    let mut operator = operator();
    let batch = input(&[2, 1], &[1.0, 3.0], &[2.0, 1.0], &[None, Some(2.0)]);
    let original = batch.table_payload().unwrap().batches()[0].clone();
    let inputs = BTreeMap::from([("input".into(), batch)]);
    let run = RunContext::new(JsonMap::new(), None, CancellationToken::new()).unwrap();
    let error = BatchOperator::process(&mut operator, &inputs, &BatchOperatorContext { run: &run })
        .await
        .unwrap_err();
    let CalcFlowError::Operator { node_id, message } = error else {
        panic!("{error}");
    };
    assert_eq!(node_id, "features");
    assert_eq!(message, format!("high: {}", input_message(0)));
    assert_eq!(
        &original,
        &inputs["input"].table_payload().unwrap().batches()[0]
    );
}

fn job() -> StreamJobContext {
    StreamJobContext::new(
        42,
        "a".repeat(64),
        JsonMap::new(),
        None,
        CancellationToken::new(),
    )
}

#[tokio::test]
async fn test_later_stream_batch_reports_its_input_row_without_accepting_bad_rows() {
    let mut operator = operator();
    let job = job();
    let initial = StreamOperatorContext::new(&job, "features", None);
    let mut collector = EdgeCollector::new(operator.output_ports().to_vec());
    operator
        .process_data(
            "input",
            input(&[1, 2], &[3.0, 4.0], &[1.0, 2.0], &[Some(2.0), Some(3.0)]),
            &initial,
            &mut collector,
        )
        .await
        .unwrap();
    operator
        .on_watermark(EventTime::from_micros(3), &initial, &mut collector)
        .await
        .unwrap();
    assert!(!collector.drain("output").is_empty());
    let context = StreamOperatorContext::new(&job, "features", Some(EventTime::from_micros(3)));
    let error = operator
        .process_data(
            "input",
            input(&[5, 4], &[1.0, 5.0], &[2.0, 3.0], &[None, Some(4.0)]),
            &context,
            &mut collector,
        )
        .await
        .unwrap_err();
    assert_input_row(&error, 0);
    assert!(collector.drain("output").is_empty());
    operator
        .process_data(
            "input",
            input(&[4], &[5.0], &[3.0], &[Some(4.0)]),
            &context,
            &mut collector,
        )
        .await
        .unwrap();
    operator
        .on_watermark(EventTime::from_micros(6), &context, &mut collector)
        .await
        .unwrap();
    let messages = collector.drain("output");
    let record = &messages[0]
        .as_data()
        .unwrap()
        .table_payload()
        .unwrap()
        .batches()[0];
    assert_eq!(record.num_rows(), 1);
    assert_eq!(
        record
            .column_by_name("rsi")
            .unwrap()
            .as_any()
            .downcast_ref::<Float64Array>()
            .unwrap()
            .value(0)
            .to_bits(),
        100.0_f64.to_bits()
    );
}

#[tokio::test]
async fn test_finalized_recurrence_keeps_the_entity_ordered_position() {
    let mut operator = operator();
    let job = job();
    let context = StreamOperatorContext::new(&job, "features", None);
    let mut collector = EdgeCollector::new(operator.output_ports().to_vec());
    operator
        .process_data(
            "input",
            input(
                &[2, 1],
                &[1.0; 2],
                &[-1.0; 2],
                &[Some(-f64::MAX), Some(f64::MAX)],
            ),
            &context,
            &mut collector,
        )
        .await
        .unwrap();
    for _ in 0..2 {
        let error = operator
            .on_watermark(EventTime::from_micros(3), &context, &mut collector)
            .await
            .unwrap_err();
        let CalcFlowError::InvalidArgument { field, message } = error else {
            panic!("{error}");
        };
        assert_eq!(field, "close");
        assert_eq!(
            message,
            "rolling node \"features\", ordered position 1: RSI price difference from previous price is not representable as finite float64"
        );
        assert!(!message.contains("private-entity"));
        assert!(collector.drain("output").is_empty());
    }
}
