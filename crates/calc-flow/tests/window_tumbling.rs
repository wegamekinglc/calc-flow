use std::{sync::Arc, time::Duration};

use calc_flow::{
    AggregateFunction, Batch, BatchMetadata, CancellationToken, EdgeCollector, EventTime, JsonMap,
    OperatorMetadata, StreamJobContext, StreamOperator, StreamOperatorContext,
    WindowAggregateOperator, WindowSpec,
};
use datafusion::arrow::{
    array::{
        Array, ArrayRef, Float64Array, Int64Array, StringArray, TimestampMicrosecondArray,
        UInt64Array,
    },
    datatypes::{DataType, Field, Schema, TimeUnit},
    record_batch::RecordBatch,
};

fn input_schema() -> Arc<Schema> {
    Arc::new(Schema::new(vec![
        Field::new(
            "event_time",
            DataType::Timestamp(TimeUnit::Microsecond, None),
            true,
        ),
        Field::new("group", DataType::Utf8, true),
        Field::new("value", DataType::Int64, true),
    ]))
}

fn input_batch(
    event_times: Vec<Option<i64>>,
    groups: Vec<Option<&str>>,
    values: Vec<Option<i64>>,
) -> Batch {
    let record = RecordBatch::try_new(
        input_schema(),
        vec![
            Arc::new(TimestampMicrosecondArray::from(event_times)) as ArrayRef,
            Arc::new(StringArray::from(groups)),
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

fn operator() -> WindowAggregateOperator {
    let spec = WindowSpec::tumbling("event_time", Duration::from_secs(60))
        .unwrap()
        .group_by(["group"])
        .unwrap()
        .aggregate(AggregateFunction::Sum, "value", "sum_value")
        .unwrap();
    WindowAggregateOperator::new("window", input_schema(), spec).unwrap()
}

#[tokio::test]
async fn tumbling_windows_use_euclidean_boundaries_and_emit_once_in_key_order() {
    let mut operator = operator();
    let job = job();
    let context = StreamOperatorContext::new(&job, "window-node", None);
    let mut collector = EdgeCollector::new(operator.output_ports().to_vec());
    operator
        .process_data(
            "input",
            input_batch(
                vec![Some(-1), Some(0), Some(59_999_999), Some(60_000_000)],
                vec![Some("b"), Some("a"), Some("a"), Some("a")],
                vec![Some(1), Some(2), Some(3), Some(4)],
            ),
            &context,
            &mut collector,
        )
        .await
        .unwrap();

    operator
        .on_watermark(EventTime::from_micros(60_000_000), &context, &mut collector)
        .await
        .unwrap();
    let closed = collector.drain("output");
    assert_eq!(closed.len(), 1);
    let closed = closed[0].as_data().unwrap();
    assert_eq!(closed.metadata().source(), "window-node");
    assert_eq!(closed.metadata().sequence(), 0);
    assert!(closed.metadata().attributes().is_empty());
    let record = &closed.table_payload().unwrap().batches()[0];
    assert_eq!(
        record
            .column_by_name("window_start")
            .unwrap()
            .as_any()
            .downcast_ref::<TimestampMicrosecondArray>()
            .unwrap()
            .values(),
        &[-60_000_000, 0]
    );
    assert_eq!(
        record
            .column_by_name("window_end")
            .unwrap()
            .as_any()
            .downcast_ref::<TimestampMicrosecondArray>()
            .unwrap()
            .values(),
        &[0, 60_000_000]
    );
    assert_eq!(
        record
            .column_by_name("group")
            .unwrap()
            .as_any()
            .downcast_ref::<StringArray>()
            .unwrap()
            .iter()
            .collect::<Vec<_>>(),
        [Some("b"), Some("a")]
    );
    assert_eq!(
        record
            .column_by_name("sum_value")
            .unwrap()
            .as_any()
            .downcast_ref::<Int64Array>()
            .unwrap()
            .values(),
        &[1, 5]
    );

    operator.on_end(&context, &mut collector).await.unwrap();
    let final_window = collector.drain("output");
    assert_eq!(final_window.len(), 1);
    let final_window = final_window[0].as_data().unwrap();
    assert_eq!(final_window.metadata().sequence(), 1);
    let record = &final_window.table_payload().unwrap().batches()[0];
    assert_eq!(
        record
            .column_by_name("window_start")
            .unwrap()
            .as_any()
            .downcast_ref::<TimestampMicrosecondArray>()
            .unwrap()
            .value(0),
        60_000_000
    );
    assert_eq!(
        record
            .column_by_name("sum_value")
            .unwrap()
            .as_any()
            .downcast_ref::<Int64Array>()
            .unwrap()
            .value(0),
        4
    );
    operator.on_end(&context, &mut collector).await.unwrap();
    assert!(collector.drain("output").is_empty());
}

#[tokio::test]
async fn watermark_gaps_do_not_materialize_empty_windows() {
    let mut operator = operator();
    let job = job();
    let context = StreamOperatorContext::new(&job, "window-node", None);
    let mut collector = EdgeCollector::new(operator.output_ports().to_vec());
    operator
        .process_data(
            "input",
            input_batch(vec![Some(1)], vec![Some("a")], vec![Some(2)]),
            &context,
            &mut collector,
        )
        .await
        .unwrap();

    operator
        .on_watermark(
            EventTime::from_micros(5 * 60_000_000),
            &context,
            &mut collector,
        )
        .await
        .unwrap();
    let output = collector.drain("output");
    assert_eq!(output.len(), 1);
    assert_eq!(
        output[0]
            .as_data()
            .unwrap()
            .table_payload()
            .unwrap()
            .batches()[0]
            .num_rows(),
        1
    );

    operator.on_end(&context, &mut collector).await.unwrap();
    assert!(collector.drain("output").is_empty());
}

#[tokio::test]
async fn all_null_aggregate_inputs_emit_count_zero_and_nullable_results() {
    let spec = WindowSpec::tumbling("event_time", Duration::from_secs(60))
        .unwrap()
        .group_by(["group"])
        .unwrap()
        .aggregate(AggregateFunction::Count, "value", "count_value")
        .unwrap()
        .aggregate(AggregateFunction::Sum, "value", "sum_value")
        .unwrap()
        .aggregate(AggregateFunction::Avg, "value", "avg_value")
        .unwrap();
    let mut operator = WindowAggregateOperator::new("window", input_schema(), spec).unwrap();
    let job = job();
    let context = StreamOperatorContext::new(&job, "window", None);
    let mut collector = EdgeCollector::new(operator.output_ports().to_vec());
    operator
        .process_data(
            "input",
            input_batch(
                vec![Some(0), Some(1), Some(2)],
                vec![None, None, Some("a")],
                vec![None, Some(2), None],
            ),
            &context,
            &mut collector,
        )
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
    let groups = record
        .column_by_name("group")
        .unwrap()
        .as_any()
        .downcast_ref::<StringArray>()
        .unwrap();
    assert!(groups.is_null(0));
    assert_eq!(groups.value(1), "a");
    assert_eq!(
        record
            .column_by_name("count_value")
            .unwrap()
            .as_any()
            .downcast_ref::<UInt64Array>()
            .unwrap()
            .values(),
        &[1, 0]
    );
    let sums = record
        .column_by_name("sum_value")
        .unwrap()
        .as_any()
        .downcast_ref::<Int64Array>()
        .unwrap();
    assert_eq!(sums.value(0), 2);
    assert!(sums.is_null(1));
    let averages = record
        .column_by_name("avg_value")
        .unwrap()
        .as_any()
        .downcast_ref::<Float64Array>()
        .unwrap();
    assert_eq!(averages.value(0).to_bits(), 2.0_f64.to_bits());
    assert!(averages.is_null(1));
}

#[tokio::test]
async fn aggregate_overflow_aborts_the_whole_input_batch_transaction() {
    let mut operator = operator();
    let job = job();
    let context = StreamOperatorContext::new(&job, "window", None);
    let mut collector = EdgeCollector::new(operator.output_ports().to_vec());

    assert!(
        operator
            .process_data(
                "input",
                input_batch(
                    vec![Some(0), Some(1)],
                    vec![Some("a"), Some("a")],
                    vec![Some(i64::MAX), Some(1)],
                ),
                &context,
                &mut collector,
            )
            .await
            .is_err()
    );
    assert!(collector.drain("output").is_empty());

    operator
        .process_data(
            "input",
            input_batch(vec![Some(2)], vec![Some("a")], vec![Some(7)]),
            &context,
            &mut collector,
        )
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
            .column_by_name("sum_value")
            .unwrap()
            .as_any()
            .downcast_ref::<Int64Array>()
            .unwrap()
            .value(0),
        7
    );
}

#[tokio::test]
async fn oversized_group_key_fails_without_installing_partial_state() {
    let mut operator = operator();
    let job = job();
    let context = StreamOperatorContext::new(&job, "window", None);
    let mut collector = EdgeCollector::new(operator.output_ports().to_vec());
    let oversized = "x".repeat(65_534);

    assert!(
        operator
            .process_data(
                "input",
                input_batch(
                    vec![Some(0)],
                    vec![Some(oversized.as_str())],
                    vec![Some(99)],
                ),
                &context,
                &mut collector,
            )
            .await
            .is_err()
    );
    operator
        .process_data(
            "input",
            input_batch(vec![Some(1)], vec![Some("small")], vec![Some(3)]),
            &context,
            &mut collector,
        )
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
    assert_eq!(record.num_rows(), 1);
    assert_eq!(
        record
            .column_by_name("group")
            .unwrap()
            .as_any()
            .downcast_ref::<StringArray>()
            .unwrap()
            .value(0),
        "small"
    );
}
