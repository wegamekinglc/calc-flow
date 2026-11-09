//! Dropped rows do not materialize unused caller payloads.

use std::sync::Arc;

use calc_flow::{
    Batch, BatchMetadata, CancellationToken, EdgeCollector, EventTime, JsonMap, OperatorMetadata,
    RollingOperator, RollingSpec, StateBudget, StreamJobContext, StreamOperator,
    StreamOperatorContext,
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
        Field::new("unused", DataType::Utf8, false),
    ]))
}
fn input(time: u32, price: f64, payload: &str) -> Batch {
    let columns: Vec<ArrayRef> = vec![
        Arc::new(StringArray::from(vec!["a"])),
        Arc::new(TimestampMicrosecondArray::from(vec![i64::from(time)]).with_timezone("UTC")),
        Arc::new(UInt64Array::from(vec![u64::from(time)])),
        Arc::new(Float64Array::from(vec![price])),
        Arc::new(StringArray::from(vec![payload])),
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
        "allowed_lateness_micros":0,"late_policy":{"kind":"drop","metrics_version":1},
        "value_policy":"stateful_numeric_v1"
    }))
    .unwrap();
    let mut operator = RollingOperator::new("features", schema(), spec).unwrap();
    operator
        .set_state_budget(StateBudget::new(32, 64 << 10).unwrap())
        .unwrap();
    operator
}

#[tokio::test]
async fn test_late_drop_ignores_large_unused_payload_before_state_admission() {
    let mut operator = operator();
    let job = StreamJobContext::new(
        17,
        "a".repeat(64),
        JsonMap::new(),
        None,
        CancellationToken::new(),
    );
    let context = StreamOperatorContext::new(&job, "features", None);
    let mut output = EdgeCollector::new(operator.output_ports().to_vec());
    for (time, price) in [(10, 1.0), (11, 2.0)] {
        operator
            .process_data("input", input(time, price, "small"), &context, &mut output)
            .await
            .unwrap();
    }
    operator
        .on_watermark(EventTime::from_micros(11), &context, &mut output)
        .await
        .unwrap();
    assert_eq!(output.drain("output").len(), 1);

    let context = StreamOperatorContext::new(&job, "features", Some(EventTime::from_micros(11)));

    // The Arrow allocation already belongs to the caller, and this row is
    // discarded by its identity before the unused payload is materialized.
    let dropped = input(5, 0.0, &"x".repeat(1 << 20));
    operator
        .process_data("input", dropped, &context, &mut output)
        .await
        .unwrap();
    assert!(output.drain("output").is_empty());
    operator
        .process_data("input", input(12, 1.0, "small"), &context, &mut output)
        .await
        .unwrap();
    operator
        .on_watermark(EventTime::from_micros(12), &context, &mut output)
        .await
        .unwrap();
    let messages = output.drain("output");
    assert_eq!(messages.len(), 1);
    let record = &messages[0]
        .as_data()
        .unwrap()
        .table_payload()
        .unwrap()
        .batches()[0];
    let rsi = record
        .column_by_name("rsi")
        .unwrap()
        .as_any()
        .downcast_ref::<Float64Array>()
        .unwrap();
    assert_eq!(rsi.iter().collect::<Vec<_>>(), [Some(50.0)]);
}
