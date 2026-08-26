use std::sync::Arc;

use calc_flow::{
    Batch, BatchMetadata, CancellationToken, EdgeCollector, EventTime, JsonMap, OperatorMetadata,
    RollingOperator, RollingSpec, StreamJobContext, StreamOperator, StreamOperatorContext,
};
use datafusion::arrow::{
    array::{
        Array, ArrayRef, Float64Array, Int64Array, StringArray, TimestampMicrosecondArray,
        UInt64Array,
    },
    datatypes::{DataType, Field, Schema, TimeUnit},
    record_batch::RecordBatch,
};

const FINGERPRINT: &str = "0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef";

fn input_schema() -> Arc<Schema> {
    Arc::new(Schema::new(vec![
        Field::new(
            "ts",
            DataType::Timestamp(TimeUnit::Microsecond, Some(Arc::from("UTC"))),
            false,
        ),
        Field::new("symbol", DataType::Utf8, false),
        Field::new("sequence", DataType::UInt64, false),
        Field::new("price", DataType::Float64, true),
        Field::new("volume", DataType::Int64, true),
    ]))
}

fn input_batch(rows: &[(i64, &str, u64, Option<f64>, Option<i64>)]) -> Batch {
    let record = RecordBatch::try_new(
        input_schema(),
        vec![
            Arc::new(
                TimestampMicrosecondArray::from(rows.iter().map(|row| row.0).collect::<Vec<_>>())
                    .with_timezone("UTC"),
            ) as ArrayRef,
            Arc::new(StringArray::from(
                rows.iter().map(|row| row.1).collect::<Vec<_>>(),
            )),
            Arc::new(UInt64Array::from(
                rows.iter().map(|row| row.2).collect::<Vec<_>>(),
            )),
            Arc::new(Float64Array::from(
                rows.iter().map(|row| row.3).collect::<Vec<_>>(),
            )),
            Arc::new(Int64Array::from(
                rows.iter().map(|row| row.4).collect::<Vec<_>>(),
            )),
        ],
    )
    .unwrap();
    Batch::table(vec![record], BatchMetadata::default()).unwrap()
}

fn spec(late_policy: serde_json::Value, allowed_lateness_micros: u64) -> RollingSpec {
    serde_json::from_value(serde_json::json!({
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
        "allowed_lateness_micros": allowed_lateness_micros,
        "late_policy": late_policy,
        "value_policy": "stateful_numeric_v1"
    }))
    .unwrap()
}

fn error_operator() -> RollingOperator {
    RollingOperator::new(
        "rolling",
        input_schema(),
        spec(serde_json::json!({"kind": "error", "scope": "envelope"}), 0),
    )
    .unwrap()
}

fn drop_operator(lateness: u64) -> RollingOperator {
    RollingOperator::new(
        "rolling",
        input_schema(),
        spec(
            serde_json::json!({"kind": "drop", "metrics_version": 1}),
            lateness,
        ),
    )
    .unwrap()
}

fn job() -> StreamJobContext {
    StreamJobContext::new(
        1,
        FINGERPRINT,
        JsonMap::new(),
        None,
        CancellationToken::new(),
    )
}

fn context<'a>(job: &'a StreamJobContext, watermark: Option<i64>) -> StreamOperatorContext<'a> {
    StreamOperatorContext::new(job, "rolling", watermark.map(EventTime::from_micros))
}

fn collector() -> EdgeCollector {
    EdgeCollector::new(error_operator().output_ports().to_vec())
}

#[derive(Default)]
struct Observed {
    event_times: Vec<i64>,
    symbols: Vec<String>,
    sequences: Vec<u64>,
    lags: Vec<Option<f64>>,
    deltas: Vec<Option<i64>>,
    batch_sequences: Vec<u64>,
}

fn drain(collector: &mut EdgeCollector, observed: &mut Observed) {
    for message in collector.drain("output") {
        let batch = message.as_data().unwrap();
        observed.batch_sequences.push(batch.metadata().sequence());
        for record in batch.table_payload().unwrap().batches() {
            let event_times = record
                .column_by_name("ts")
                .unwrap()
                .as_any()
                .downcast_ref::<TimestampMicrosecondArray>()
                .unwrap();
            let symbols = record
                .column_by_name("symbol")
                .unwrap()
                .as_any()
                .downcast_ref::<StringArray>()
                .unwrap();
            let sequences = record
                .column_by_name("sequence")
                .unwrap()
                .as_any()
                .downcast_ref::<UInt64Array>()
                .unwrap();
            let lags = record
                .column_by_name("price_lag_1")
                .unwrap()
                .as_any()
                .downcast_ref::<Float64Array>()
                .unwrap();
            let deltas = record
                .column_by_name("volume_delta_1")
                .unwrap()
                .as_any()
                .downcast_ref::<Int64Array>()
                .unwrap();
            for index in 0..record.num_rows() {
                observed.event_times.push(event_times.value(index));
                observed.symbols.push(symbols.value(index).to_owned());
                observed.sequences.push(sequences.value(index));
                observed.lags.push(lags.iter().nth(index).unwrap());
                observed.deltas.push(deltas.iter().nth(index).unwrap());
            }
        }
    }
}

#[tokio::test]
async fn accepted_rows_emit_in_canonical_order_at_the_closing_watermark() {
    let job = job();
    let mut operator = error_operator();
    let mut collector = collector();
    let mut observed = Observed::default();

    operator
        .process_data(
            "input",
            input_batch(&[
                (15, "b", 1, Some(10.0), Some(100)),
                (10, "a", 1, Some(1.0), Some(10)),
            ]),
            &context(&job, None),
            &mut collector,
        )
        .await
        .unwrap();
    drain(&mut collector, &mut observed);
    assert!(observed.event_times.is_empty());

    operator
        .on_watermark(
            EventTime::from_micros(20),
            &context(&job, Some(20)),
            &mut collector,
        )
        .await
        .unwrap();
    drain(&mut collector, &mut observed);
    assert_eq!(observed.event_times, vec![10, 15]);
    assert_eq!(observed.symbols, vec!["a", "b"]);
    assert_eq!(observed.lags, vec![None, None]);
    assert_eq!(observed.batch_sequences, vec![0]);

    operator
        .on_watermark(
            EventTime::from_micros(30),
            &context(&job, Some(30)),
            &mut collector,
        )
        .await
        .unwrap();
    drain(&mut collector, &mut observed);
    assert_eq!(observed.event_times, vec![10, 15]);
}

#[tokio::test]
async fn out_of_order_rows_within_lateness_are_buffered_and_emitted_in_order() {
    let job = job();
    let mut operator = drop_operator(5);
    let mut collector = collector();
    let mut observed = Observed::default();

    operator
        .process_data(
            "input",
            input_batch(&[(20, "a", 1, Some(2.0), Some(20))]),
            &context(&job, None),
            &mut collector,
        )
        .await
        .unwrap();
    operator
        .process_data(
            "input",
            input_batch(&[(10, "a", 1, Some(1.0), Some(10))]),
            &context(&job, None),
            &mut collector,
        )
        .await
        .unwrap();

    operator
        .on_watermark(
            EventTime::from_micros(15),
            &context(&job, Some(15)),
            &mut collector,
        )
        .await
        .unwrap();
    drain(&mut collector, &mut observed);
    assert_eq!(observed.event_times, vec![10]);

    operator
        .on_watermark(
            EventTime::from_micros(25),
            &context(&job, Some(25)),
            &mut collector,
        )
        .await
        .unwrap();
    drain(&mut collector, &mut observed);
    assert_eq!(observed.event_times, vec![10, 20]);
    assert_eq!(observed.lags, vec![None, Some(1.0)]);
    assert_eq!(observed.deltas, vec![None, Some(10)]);
}

#[tokio::test]
async fn late_rows_are_dropped_under_the_drop_policy() {
    let job = job();
    let mut operator = drop_operator(5);
    let mut collector = collector();
    let mut observed = Observed::default();

    operator
        .process_data(
            "input",
            input_batch(&[
                (115, "a", 1, Some(1.0), Some(10)),
                (114, "a", 2, Some(2.0), Some(20)),
                (117, "a", 3, Some(3.0), Some(30)),
            ]),
            &context(&job, Some(120)),
            &mut collector,
        )
        .await
        .unwrap();
    operator
        .on_end(&context(&job, Some(120)), &mut collector)
        .await
        .unwrap();
    drain(&mut collector, &mut observed);
    assert_eq!(observed.event_times, vec![117]);
    assert_eq!(observed.sequences, vec![3]);
}

#[tokio::test]
async fn late_rows_under_error_policy_reject_the_envelope_transactionally() {
    let job = job();
    let mut operator = error_operator();
    let mut collector = collector();

    let error = operator
        .process_data(
            "input",
            input_batch(&[(5, "a", 1, Some(1.0), Some(10))]),
            &context(&job, Some(10)),
            &mut collector,
        )
        .await
        .unwrap_err();
    assert!(
        error.to_string().contains(
            "rolling: late_row: envelope rejected at row_index=0; event_time_micros=5, closed_at_watermark_micros=10"
        ),
        "unexpected error: {error}"
    );

    let mut observed = Observed::default();
    operator
        .process_data(
            "input",
            input_batch(&[(11, "a", 1, Some(1.0), Some(10))]),
            &context(&job, Some(10)),
            &mut collector,
        )
        .await
        .unwrap();
    operator
        .on_watermark(
            EventTime::from_micros(20),
            &context(&job, Some(20)),
            &mut collector,
        )
        .await
        .unwrap();
    drain(&mut collector, &mut observed);
    assert_eq!(observed.event_times, vec![11]);
}

#[tokio::test]
async fn duplicate_identity_within_one_envelope_is_a_data_error() {
    let job = job();
    let mut operator = error_operator();
    let mut collector = collector();
    let error = operator
        .process_data(
            "input",
            input_batch(&[
                (10, "a", 1, Some(1.0), Some(10)),
                (10, "a", 1, Some(2.0), Some(20)),
            ]),
            &context(&job, None),
            &mut collector,
        )
        .await
        .unwrap_err();
    assert!(
        matches!(
            error,
            calc_flow::CalcFlowError::Operator { ref node_id, .. } if node_id == "rolling"
        ),
        "unexpected error: {error}"
    );
}

#[tokio::test]
async fn duplicate_identity_against_a_buffered_row_is_a_data_error() {
    let job = job();
    let mut operator = error_operator();
    let mut collector = collector();
    operator
        .process_data(
            "input",
            input_batch(&[(10, "a", 1, Some(1.0), Some(10))]),
            &context(&job, None),
            &mut collector,
        )
        .await
        .unwrap();
    let error = operator
        .process_data(
            "input",
            input_batch(&[(10, "a", 1, Some(2.0), Some(20))]),
            &context(&job, None),
            &mut collector,
        )
        .await
        .unwrap_err();
    assert!(
        matches!(
            error,
            calc_flow::CalcFlowError::Operator { ref node_id, .. } if node_id == "rolling"
        ),
        "unexpected error: {error}"
    );
}

#[tokio::test]
async fn on_end_flushes_every_buffered_row_once_in_canonical_order() {
    let job = job();
    let mut operator = error_operator();
    let mut collector = collector();
    let mut observed = Observed::default();

    operator
        .process_data(
            "input",
            input_batch(&[
                (20, "a", 1, Some(2.0), Some(20)),
                (10, "a", 1, Some(1.0), Some(10)),
            ]),
            &context(&job, None),
            &mut collector,
        )
        .await
        .unwrap();
    operator
        .on_end(&context(&job, None), &mut collector)
        .await
        .unwrap();
    operator
        .on_end(&context(&job, None), &mut collector)
        .await
        .unwrap();
    drain(&mut collector, &mut observed);
    assert_eq!(observed.event_times, vec![10, 20]);
    assert_eq!(observed.lags, vec![None, Some(1.0)]);

    let error = operator
        .process_data(
            "input",
            input_batch(&[(30, "a", 1, Some(3.0), Some(30))]),
            &context(&job, None),
            &mut collector,
        )
        .await
        .unwrap_err();
    assert!(
        matches!(error, calc_flow::CalcFlowError::Operator { .. }),
        "unexpected error: {error}"
    );
}

#[tokio::test]
async fn watermark_must_advance_strictly() {
    let job = job();
    let mut operator = error_operator();
    let mut collector = collector();
    operator
        .on_watermark(
            EventTime::from_micros(20),
            &context(&job, Some(20)),
            &mut collector,
        )
        .await
        .unwrap();
    let error = operator
        .on_watermark(
            EventTime::from_micros(20),
            &context(&job, Some(20)),
            &mut collector,
        )
        .await
        .unwrap_err();
    assert!(
        matches!(error, calc_flow::CalcFlowError::Operator { .. }),
        "unexpected error: {error}"
    );
}

#[tokio::test]
async fn micro_batch_segmentation_matches_one_envelope_output() {
    let job = job();
    let rows = vec![
        (10, "a", 1, Some(1.0), Some(10)),
        (10, "b", 1, Some(5.0), Some(50)),
        (11, "a", 2, Some(2.0), Some(20)),
        (12, "b", 2, Some(6.0), Some(60)),
        (12, "a", 3, Some(3.0), Some(30)),
    ];
    let mut whole = error_operator();
    let mut whole_collector = collector();
    let mut whole_observed = Observed::default();
    whole
        .process_data(
            "input",
            input_batch(&rows),
            &context(&job, None),
            &mut whole_collector,
        )
        .await
        .unwrap();
    whole
        .on_watermark(
            EventTime::from_micros(100),
            &context(&job, Some(100)),
            &mut whole_collector,
        )
        .await
        .unwrap();
    drain(&mut whole_collector, &mut whole_observed);

    let mut segmented = error_operator();
    let mut segmented_collector = collector();
    let mut segmented_observed = Observed::default();
    for chunk in rows.chunks(2) {
        segmented
            .process_data(
                "input",
                input_batch(chunk),
                &context(&job, None),
                &mut segmented_collector,
            )
            .await
            .unwrap();
    }
    segmented
        .on_watermark(
            EventTime::from_micros(100),
            &context(&job, Some(100)),
            &mut segmented_collector,
        )
        .await
        .unwrap();
    drain(&mut segmented_collector, &mut segmented_observed);

    assert_eq!(whole_observed.event_times, segmented_observed.event_times);
    assert_eq!(whole_observed.symbols, segmented_observed.symbols);
    assert_eq!(whole_observed.sequences, segmented_observed.sequences);
    assert_eq!(whole_observed.lags, segmented_observed.lags);
    assert_eq!(whole_observed.deltas, segmented_observed.deltas);
}

#[tokio::test]
async fn empty_envelope_is_a_noop() {
    let job = job();
    let mut operator = error_operator();
    let mut collector = collector();
    operator
        .process_data(
            "input",
            input_batch(&[]),
            &context(&job, None),
            &mut collector,
        )
        .await
        .unwrap();
    operator
        .on_watermark(
            EventTime::from_micros(10),
            &context(&job, Some(10)),
            &mut collector,
        )
        .await
        .unwrap();
    let mut observed = Observed::default();
    drain(&mut collector, &mut observed);
    assert!(observed.event_times.is_empty());
}
