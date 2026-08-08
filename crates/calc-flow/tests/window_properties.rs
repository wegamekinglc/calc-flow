use std::{sync::Arc, time::Duration};

use calc_flow::{
    AggregateFunction, Batch, BatchMetadata, CancellationToken, EdgeCollector, JsonMap,
    OperatorMetadata, StreamJobContext, StreamOperator, StreamOperatorContext,
    WindowAggregateOperator, WindowSpec,
};
use datafusion::arrow::{
    array::{ArrayRef, Float64Array, TimestampMicrosecondArray, UInt64Array},
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
        Field::new("group", DataType::Float64, false),
        Field::new("value", DataType::Float64, false),
    ]))
}

fn batch(groups: Vec<f64>, values: Vec<f64>) -> Batch {
    let event_times = vec![0; groups.len()];
    let record = RecordBatch::try_new(
        schema(),
        vec![
            Arc::new(TimestampMicrosecondArray::from(event_times)) as ArrayRef,
            Arc::new(Float64Array::from(groups)),
            Arc::new(Float64Array::from(values)),
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

#[tokio::test]
async fn float_group_identity_distinguishes_signed_zero_and_nan_payloads() {
    let spec = WindowSpec::tumbling("event_time", Duration::from_secs(1))
        .unwrap()
        .group_by(["group"])
        .unwrap()
        .aggregate(AggregateFunction::Count, "value", "count_value")
        .unwrap();
    let mut operator = WindowAggregateOperator::new("window", schema(), spec).unwrap();
    let job = job();
    let context = StreamOperatorContext::new(&job, "window", None);
    let mut collector = EdgeCollector::new(operator.output_ports().to_vec());
    let negative_zero = f64::from_bits(0x8000_0000_0000_0000);
    let positive_zero = 0.0_f64;
    let first_nan = f64::from_bits(0x7ff8_0000_0000_0001);
    let second_nan = f64::from_bits(0x7ff8_0000_0000_0002);
    operator
        .process_data(
            "input",
            batch(
                vec![
                    negative_zero,
                    positive_zero,
                    first_nan,
                    second_nan,
                    first_nan,
                ],
                vec![1.0; 5],
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
    let group_bits = record
        .column_by_name("group")
        .unwrap()
        .as_any()
        .downcast_ref::<Float64Array>()
        .unwrap()
        .values()
        .iter()
        .map(|value| value.to_bits())
        .collect::<Vec<_>>();
    assert_eq!(
        group_bits,
        vec![
            negative_zero.to_bits(),
            positive_zero.to_bits(),
            first_nan.to_bits(),
            second_nan.to_bits(),
        ]
    );
    assert_eq!(
        record
            .column_by_name("count_value")
            .unwrap()
            .as_any()
            .downcast_ref::<UInt64Array>()
            .unwrap()
            .values(),
        &[1, 1, 2, 1]
    );
}

#[tokio::test]
async fn float_aggregates_canonicalize_nan_and_preserve_selected_scalar_bits() {
    let spec = WindowSpec::tumbling("event_time", Duration::from_secs(1))
        .unwrap()
        .aggregate(AggregateFunction::Sum, "value", "sum_value")
        .unwrap()
        .aggregate(AggregateFunction::Avg, "value", "avg_value")
        .unwrap()
        .aggregate(AggregateFunction::Min, "value", "min_value")
        .unwrap()
        .aggregate(AggregateFunction::Max, "value", "max_value")
        .unwrap();
    let mut operator = WindowAggregateOperator::new("window", schema(), spec).unwrap();
    let job = job();
    let context = StreamOperatorContext::new(&job, "window", None);
    let mut collector = EdgeCollector::new(operator.output_ports().to_vec());
    let payload_nan = f64::from_bits(0x7ff8_0000_0000_0042);
    operator
        .process_data(
            "input",
            batch(vec![0.0], vec![payload_nan]),
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
    let bits = |name: &str| {
        record
            .column_by_name(name)
            .unwrap()
            .as_any()
            .downcast_ref::<Float64Array>()
            .unwrap()
            .value(0)
            .to_bits()
    };
    assert_eq!(bits("sum_value"), 0x7ff8_0000_0000_0000);
    assert_eq!(bits("avg_value"), 0x7ff8_0000_0000_0000);
    assert_eq!(bits("min_value"), payload_nan.to_bits());
    assert_eq!(bits("max_value"), payload_nan.to_bits());
}

#[tokio::test]
async fn ordered_rows_produce_identical_float_bits_across_input_batch_partitions() {
    async fn run(partitions: &[&[f64]]) -> Vec<u64> {
        let spec = WindowSpec::tumbling("event_time", Duration::from_secs(1))
            .unwrap()
            .aggregate(AggregateFunction::Sum, "value", "sum_value")
            .unwrap()
            .aggregate(AggregateFunction::Avg, "value", "avg_value")
            .unwrap();
        let mut operator = WindowAggregateOperator::new("window", schema(), spec).unwrap();
        let job = job();
        let context = StreamOperatorContext::new(&job, "window", None);
        let mut collector = EdgeCollector::new(operator.output_ports().to_vec());
        for values in partitions {
            operator
                .process_data(
                    "input",
                    batch(vec![0.0; values.len()], values.to_vec()),
                    &context,
                    &mut collector,
                )
                .await
                .unwrap();
        }
        operator.on_end(&context, &mut collector).await.unwrap();
        let output = collector.drain("output");
        let record = &output[0]
            .as_data()
            .unwrap()
            .table_payload()
            .unwrap()
            .batches()[0];
        ["sum_value", "avg_value"]
            .into_iter()
            .map(|name| {
                record
                    .column_by_name(name)
                    .unwrap()
                    .as_any()
                    .downcast_ref::<Float64Array>()
                    .unwrap()
                    .value(0)
                    .to_bits()
            })
            .collect()
    }

    let values = [1.0, 1.0e100, -1.0e100, -0.0, 3.0];
    assert_eq!(
        run(&[&values]).await,
        run(&[&values[..2], &values[2..4], &values[4..]]).await
    );
}
