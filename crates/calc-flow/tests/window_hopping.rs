use std::{sync::Arc, time::Duration};

use calc_flow::{
    AggregateFunction, Batch, BatchMetadata, CancellationToken, EdgeCollector, EventTime, JsonMap,
    OperatorMetadata, StreamJobContext, StreamOperator, StreamOperatorContext,
    WindowAggregateOperator, WindowGeometry, WindowSpec,
};
use datafusion::arrow::{
    array::{ArrayRef, Int64Array, TimestampMicrosecondArray, UInt64Array},
    datatypes::{DataType, Field, Schema, TimeUnit},
    record_batch::RecordBatch,
};

fn schema() -> Arc<Schema> {
    Arc::new(Schema::new(vec![
        Field::new(
            "event_time",
            DataType::Timestamp(TimeUnit::Microsecond, None),
            false,
        ),
        Field::new("value", DataType::Int64, false),
    ]))
}

fn batch(event_times: Vec<i64>, values: Vec<i64>) -> Batch {
    let record = RecordBatch::try_new(
        schema(),
        vec![
            Arc::new(TimestampMicrosecondArray::from(event_times)) as ArrayRef,
            Arc::new(Int64Array::from(values)),
        ],
    )
    .unwrap();
    Batch::table(vec![record], BatchMetadata::default()).unwrap()
}

fn job() -> StreamJobContext {
    StreamJobContext::new(
        1,
        "fingerprint",
        JsonMap::new(),
        None,
        CancellationToken::new(),
    )
}

fn hopping_operator() -> WindowAggregateOperator {
    let spec = WindowSpec::hopping(
        "event_time",
        Duration::from_micros(60),
        Duration::from_micros(10),
    )
    .unwrap()
    .aggregate(AggregateFunction::Count, "value", "count_value")
    .unwrap();
    WindowAggregateOperator::new("window", schema(), spec).unwrap()
}

#[tokio::test]
async fn hopping_row_receives_exact_overlap_in_oldest_start_order() {
    let mut operator = hopping_operator();
    let job = job();
    let context = StreamOperatorContext::new(&job, "window", None);
    let mut collector = EdgeCollector::new(operator.output_ports().to_vec());
    operator
        .process_data("input", batch(vec![15], vec![1]), &context, &mut collector)
        .await
        .unwrap();
    operator.on_end(&context, &mut collector).await.unwrap();

    let output = collector.drain("output");
    let record = &output[0]
        .as_data()
        .unwrap()
        .table_payload()
        .unwrap()
        .batches()[0];
    assert_eq!(record.num_rows(), 6);
    assert_eq!(
        record
            .column_by_name("window_start")
            .unwrap()
            .as_any()
            .downcast_ref::<TimestampMicrosecondArray>()
            .unwrap()
            .values(),
        &[-40, -30, -20, -10, 0, 10]
    );
    assert_eq!(
        record
            .column_by_name("window_end")
            .unwrap()
            .as_any()
            .downcast_ref::<TimestampMicrosecondArray>()
            .unwrap()
            .values(),
        &[20, 30, 40, 50, 60, 70]
    );
    assert_eq!(
        record
            .column_by_name("count_value")
            .unwrap()
            .as_any()
            .downcast_ref::<UInt64Array>()
            .unwrap()
            .values(),
        &[1, 1, 1, 1, 1, 1]
    );
}

#[tokio::test]
async fn hopping_row_drops_only_closed_assignments_against_input_watermark() {
    let mut operator = hopping_operator();
    let job = job();
    let context = StreamOperatorContext::new(&job, "window", Some(EventTime::from_micros(30)));
    let mut collector = EdgeCollector::new(operator.output_ports().to_vec());
    operator
        .process_data("input", batch(vec![15], vec![1]), &context, &mut collector)
        .await
        .unwrap();
    operator.on_end(&context, &mut collector).await.unwrap();

    let output = collector.drain("output");
    let record = &output[0]
        .as_data()
        .unwrap()
        .table_payload()
        .unwrap()
        .batches()[0];
    assert_eq!(
        record
            .column_by_name("window_start")
            .unwrap()
            .as_any()
            .downcast_ref::<TimestampMicrosecondArray>()
            .unwrap()
            .values(),
        &[-20, -10, 0, 10]
    );
    assert_eq!(
        record
            .column_by_name("window_end")
            .unwrap()
            .as_any()
            .downcast_ref::<TimestampMicrosecondArray>()
            .unwrap()
            .values(),
        &[40, 50, 60, 70]
    );
}

#[tokio::test]
async fn lateness_is_classified_by_window_end_not_row_timestamp() {
    const MINUTE: i64 = 60 * 1_000_000;
    let spec = WindowSpec::tumbling("event_time", Duration::from_secs(60 * 60))
        .unwrap()
        .aggregate(AggregateFunction::Count, "value", "count_value")
        .unwrap();
    let job = job();

    let mut accepted = WindowAggregateOperator::new("accepted", schema(), spec.clone()).unwrap();
    let accepted_context =
        StreamOperatorContext::new(&job, "accepted", Some(EventTime::from_micros(30 * MINUTE)));
    let mut accepted_output = EdgeCollector::new(accepted.output_ports().to_vec());
    accepted
        .process_data(
            "input",
            batch(vec![15 * MINUTE], vec![1]),
            &accepted_context,
            &mut accepted_output,
        )
        .await
        .unwrap();
    accepted
        .on_end(&accepted_context, &mut accepted_output)
        .await
        .unwrap();
    assert_eq!(accepted_output.drain("output").len(), 1);

    let mut dropped = WindowAggregateOperator::new("dropped", schema(), spec).unwrap();
    let dropped_context =
        StreamOperatorContext::new(&job, "dropped", Some(EventTime::from_micros(65 * MINUTE)));
    let mut dropped_output = EdgeCollector::new(dropped.output_ports().to_vec());
    dropped
        .process_data(
            "input",
            batch(vec![15 * MINUTE], vec![1]),
            &dropped_context,
            &mut dropped_output,
        )
        .await
        .unwrap();
    dropped
        .on_end(&dropped_context, &mut dropped_output)
        .await
        .unwrap();
    assert!(dropped_output.drain("output").is_empty());
}

#[tokio::test]
async fn assignment_overflow_aborts_every_earlier_update_in_the_batch() {
    let size_micros = u64::try_from(i64::MAX).unwrap();
    let spec = WindowSpec {
        event_time_column: "event_time".into(),
        group_by: Vec::new(),
        geometry: WindowGeometry::Tumbling { size_micros },
        aggregates: vec![calc_flow::AggregateSpec {
            function: AggregateFunction::Sum,
            column: "value".into(),
            output: "sum_value".into(),
        }],
    };
    let mut operator = WindowAggregateOperator::new("window", schema(), spec).unwrap();
    let job = job();
    let context = StreamOperatorContext::new(&job, "window", None);
    let mut collector = EdgeCollector::new(operator.output_ports().to_vec());

    assert!(
        operator
            .process_data(
                "input",
                batch(vec![1, i64::MAX], vec![5, 7]),
                &context,
                &mut collector,
            )
            .await
            .is_err()
    );
    operator.on_end(&context, &mut collector).await.unwrap();
    assert!(collector.drain("output").is_empty());
}
