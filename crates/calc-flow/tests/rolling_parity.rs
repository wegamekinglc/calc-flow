//! Randomized batch versus final-stream parity for the SCE-08 rolling
//! catalog (duration frames, extrema, covariance, correlation): the same
//! rows under different arrival orders, segmentations, and watermark
//! placements produce the same canonical output rows within the frozen D13
//! tolerances, with min/max compared exactly and NaN/inf classifications
//! compared bitwise.

use std::sync::Arc;

use calc_flow::{
    Batch, BatchMetadata, BatchOperator, CancellationToken, EdgeCollector, EventTime, JsonMap,
    OperatorMetadata, RollingOperator, RollingSpec, StreamJobContext, StreamOperator,
    StreamOperatorContext,
};
use datafusion::arrow::{
    array::{
        Array, ArrayRef, Float64Array, Int64Array, StringArray, TimestampMicrosecondArray,
        UInt64Array,
    },
    datatypes::{DataType, Field, Schema, TimeUnit},
    record_batch::RecordBatch,
};
use proptest::{collection::vec, prelude::*};

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

/// One generated row: event time, entity, sequence, price, volume.
type InputRow = (i64, String, u64, Option<f64>, Option<i64>);

fn input_batch(rows: &[InputRow]) -> Batch {
    let record = RecordBatch::try_new(
        input_schema(),
        vec![
            Arc::new(
                TimestampMicrosecondArray::from(rows.iter().map(|row| row.0).collect::<Vec<_>>())
                    .with_timezone("UTC"),
            ) as ArrayRef,
            Arc::new(StringArray::from(
                rows.iter().map(|row| row.1.as_str()).collect::<Vec<_>>(),
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

fn parity_spec(duration_micros: u64, row_frame: u64) -> RollingSpec {
    serde_json::from_value(serde_json::json!({
        "configuration_version": 1,
        "state_layout_version": 1,
        "partition_by": ["symbol"],
        "event_time": "ts",
        "sequence_by": ["sequence"],
        "outputs": [
            {
                "kind": "mean",
                "primitive_version": 1,
                "input": "price",
                "output": "price_mean",
                "frame": {"kind": "duration", "micros": duration_micros},
                "min_periods": 1
            },
            {
                "kind": "max",
                "primitive_version": 1,
                "input": "price",
                "output": "price_max",
                "frame": {"kind": "duration", "micros": duration_micros},
                "min_periods": 1
            },
            {
                "kind": "min",
                "primitive_version": 1,
                "input": "volume",
                "output": "volume_min",
                "frame": {"kind": "rows", "size": row_frame},
                "min_periods": 1
            },
            {
                "kind": "covariance",
                "primitive_version": 1,
                "left": "price",
                "right": "volume",
                "output": "price_volume_cov",
                "frame": {"kind": "duration", "micros": duration_micros},
                "min_periods": 2,
                "ddof": 1
            },
            {
                "kind": "correlation",
                "primitive_version": 1,
                "left": "price",
                "right": "volume",
                "output": "price_volume_corr",
                "frame": {"kind": "rows", "size": row_frame},
                "min_periods": 1,
                "ddof": 1
            }
        ],
        "allowed_lateness_micros": 5,
        "late_policy": {"kind": "error", "scope": "envelope"},
        "value_policy": "stateful_numeric_v1"
    }))
    .unwrap()
}

#[derive(Clone, Default, PartialEq, Debug)]
struct ObservedRow {
    event_time: i64,
    symbol: String,
    sequence: u64,
    mean: Option<f64>,
    max: Option<f64>,
    volume_min: Option<i64>,
    covariance: Option<f64>,
    correlation: Option<f64>,
}

fn drain_rows(collector: &mut EdgeCollector, rows: &mut Vec<ObservedRow>) {
    for message in collector.drain("output") {
        let batch = message.as_data().unwrap();
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
            let means = record
                .column_by_name("price_mean")
                .unwrap()
                .as_any()
                .downcast_ref::<Float64Array>()
                .unwrap();
            let maxes = record
                .column_by_name("price_max")
                .unwrap()
                .as_any()
                .downcast_ref::<Float64Array>()
                .unwrap();
            let volume_mins = record
                .column_by_name("volume_min")
                .unwrap()
                .as_any()
                .downcast_ref::<Int64Array>()
                .unwrap();
            let covariances = record
                .column_by_name("price_volume_cov")
                .unwrap()
                .as_any()
                .downcast_ref::<Float64Array>()
                .unwrap();
            let correlations = record
                .column_by_name("price_volume_corr")
                .unwrap()
                .as_any()
                .downcast_ref::<Float64Array>()
                .unwrap();
            for index in 0..record.num_rows() {
                rows.push(ObservedRow {
                    event_time: event_times.value(index),
                    symbol: symbols.value(index).to_owned(),
                    sequence: sequences.value(index),
                    mean: means.iter().nth(index).unwrap(),
                    max: maxes.iter().nth(index).unwrap(),
                    volume_min: volume_mins.iter().nth(index).unwrap(),
                    covariance: covariances.iter().nth(index).unwrap(),
                    correlation: correlations.iter().nth(index).unwrap(),
                });
            }
        }
    }
}

async fn run_batch(rows: &[InputRow], duration_micros: u64, row_frame: u64) -> Vec<ObservedRow> {
    let mut operator = RollingOperator::new(
        "rolling",
        input_schema(),
        parity_spec(duration_micros, row_frame),
    )
    .unwrap();
    let run = calc_flow::RunContext::new(
        std::collections::BTreeMap::new(),
        None,
        CancellationToken::new(),
    )
    .unwrap();
    let context = calc_flow::BatchOperatorContext { run: &run };
    let outputs = operator
        .process(
            &std::collections::BTreeMap::from([("input".into(), input_batch(rows))]),
            &context,
        )
        .await
        .unwrap();
    observed_from_batch(&outputs["output"])
}

fn observed_from_batch(batch: &Batch) -> Vec<ObservedRow> {
    let mut rows = Vec::new();
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
        let means = record
            .column_by_name("price_mean")
            .unwrap()
            .as_any()
            .downcast_ref::<Float64Array>()
            .unwrap();
        let maxes = record
            .column_by_name("price_max")
            .unwrap()
            .as_any()
            .downcast_ref::<Float64Array>()
            .unwrap();
        let volume_mins = record
            .column_by_name("volume_min")
            .unwrap()
            .as_any()
            .downcast_ref::<Int64Array>()
            .unwrap();
        let covariances = record
            .column_by_name("price_volume_cov")
            .unwrap()
            .as_any()
            .downcast_ref::<Float64Array>()
            .unwrap();
        let correlations = record
            .column_by_name("price_volume_corr")
            .unwrap()
            .as_any()
            .downcast_ref::<Float64Array>()
            .unwrap();
        for index in 0..record.num_rows() {
            rows.push(ObservedRow {
                event_time: event_times.value(index),
                symbol: symbols.value(index).to_owned(),
                sequence: sequences.value(index),
                mean: means.iter().nth(index).unwrap(),
                max: maxes.iter().nth(index).unwrap(),
                volume_min: volume_mins.iter().nth(index).unwrap(),
                covariance: covariances.iter().nth(index).unwrap(),
                correlation: correlations.iter().nth(index).unwrap(),
            });
        }
    }
    rows
}

/// Drives the stream lifecycle over shuffled, segmented envelopes. Each
/// envelope is delivered at a watermark that stays below every undelivered
/// row's closing coordinate, so bounded out-of-order arrival reorders
/// without classifying anything too late; the final watermark flushes.
async fn run_stream(
    envelopes: &[Vec<InputRow>],
    duration_micros: u64,
    row_frame: u64,
) -> Vec<ObservedRow> {
    let job = StreamJobContext::new(
        1,
        FINGERPRINT,
        JsonMap::new(),
        None,
        CancellationToken::new(),
    );
    let mut operator = RollingOperator::new(
        "rolling",
        input_schema(),
        parity_spec(duration_micros, row_frame),
    )
    .unwrap();
    let mut collector = EdgeCollector::new(operator.output_ports().to_vec());
    let mut observed = Vec::new();
    let mut last_watermark: Option<i64> = None;
    let lateness = 5_i64;
    for (index, envelope) in envelopes.iter().enumerate() {
        let context =
            StreamOperatorContext::new(&job, "rolling", last_watermark.map(EventTime::from_micros));
        operator
            .process_data("input", input_batch(envelope), &context, &mut collector)
            .await
            .unwrap();
        // Close at most the rows every later envelope can still reorder
        // around: one below the earliest undelivered closing coordinate. The
        // final envelope leaves everything open for the closing watermark.
        let Some(undelivered_minimum) = envelopes[index + 1..]
            .iter()
            .flat_map(|rows| rows.iter().map(|row| row.0))
            .min()
        else {
            drain_rows(&mut collector, &mut observed);
            continue;
        };
        let candidate = undelivered_minimum + lateness - 1;
        if last_watermark.is_none_or(|previous| candidate > previous) {
            let context = StreamOperatorContext::new(
                &job,
                "rolling",
                Some(EventTime::from_micros(candidate)),
            );
            operator
                .on_watermark(EventTime::from_micros(candidate), &context, &mut collector)
                .await
                .unwrap();
            last_watermark = Some(candidate);
        }
        drain_rows(&mut collector, &mut observed);
    }
    let final_watermark = i64::MAX / 2;
    let context = StreamOperatorContext::new(
        &job,
        "rolling",
        Some(EventTime::from_micros(final_watermark)),
    );
    operator
        .on_watermark(
            EventTime::from_micros(final_watermark),
            &context,
            &mut collector,
        )
        .await
        .unwrap();
    let context = StreamOperatorContext::new(&job, "rolling", None);
    operator.on_end(&context, &mut collector).await.unwrap();
    drain_rows(&mut collector, &mut observed);
    observed
}

/// D13 tolerances: rolling sum/mean 1e-12; variance/stddev/covariance/
/// correlation 1e-10 relative with 1e-12 absolute; min/max exact.
fn assert_close(actual: Option<f64>, expected: Option<f64>, what: &str, rtol: f64) {
    match (actual, expected) {
        (None, None) => {}
        (Some(actual), Some(expected)) => {
            if actual.is_nan() || expected.is_nan() {
                assert!(actual.is_nan() && expected.is_nan(), "{what} NaN mismatch");
            } else if actual.is_infinite() || expected.is_infinite() {
                // Infinities are exact classifications: equal sign or fail.
                assert!(
                    actual.total_cmp(&expected) == std::cmp::Ordering::Equal,
                    "{what} infinity classification: {actual} vs {expected}"
                );
            } else {
                let tolerance = rtol * expected.abs().max(1.0) + 1e-12;
                assert!(
                    (actual - expected).abs() <= tolerance,
                    "{what} drifted: {actual} vs {expected}"
                );
            }
        }
        (actual, expected) => panic!("{what} nullness {actual:?} vs {expected:?}"),
    }
}

fn assert_rows_match(actual: &[ObservedRow], expected: &[ObservedRow]) {
    assert_eq!(actual.len(), expected.len(), "row count diverged");
    for (index, (left, right)) in actual.iter().zip(expected).enumerate() {
        assert_eq!(left.event_time, right.event_time, "row {index} identity");
        assert_eq!(left.symbol, right.symbol, "row {index} identity");
        assert_eq!(left.sequence, right.sequence, "row {index} identity");
        // min/max are exact; the D13 mean tolerance is 1e-12 and the pair
        // tolerance is 1e-10.
        assert_eq!(left.volume_min, right.volume_min, "row {index} min");
        assert_close(left.mean, right.mean, "row {index} mean", 1e-12);
        assert_close(left.max, right.max, "row {index} max", 1e-12);
        assert_close(
            left.covariance,
            right.covariance,
            "row {index} covariance",
            1e-10,
        );
        assert_close(
            left.correlation,
            right.correlation,
            "row {index} correlation",
            1e-10,
        );
    }
}

fn price_value(seed: u32) -> Option<f64> {
    match seed % 10 {
        0 | 1 => None,
        2 => Some(f64::NAN),
        3 => Some(f64::INFINITY),
        4 => Some(f64::NEG_INFINITY),
        other => Some(f64::from(other) * 1.5 - 6.0),
    }
}

fn volume_value(seed: u32) -> Option<i64> {
    match seed % 8 {
        0 => None,
        other => Some(i64::from(other) * 11 - 44),
    }
}

proptest! {
    #![proptest_config(ProptestConfig {
        cases: 32,
        failure_persistence: None,
        ..ProptestConfig::default()
    })]
    #[test]
    fn batch_and_final_stream_agree_across_segmentation_watermarks_and_disorder(
        row_count in 4_usize..28,
        duration_micros in 3_u64..24,
        row_frame in 1_u64..6,
        time_bits in vec(any::<u32>(), 4..28),
        entity_bits in vec(any::<u32>(), 4..28),
        price_bits in vec(any::<u32>(), 4..28),
        volume_bits in vec(any::<u32>(), 4..28),
        arrival_bits in vec(any::<u32>(), 4..28),
        chunk_bits in vec(any::<u32>(), 4..28),
    ) {
        let bits_len = time_bits
            .len()
            .min(entity_bits.len())
            .min(price_bits.len())
            .min(volume_bits.len())
            .min(arrival_bits.len())
            .min(chunk_bits.len());
        proptest::prop_assume!(row_count <= bits_len);
        let rows: Vec<InputRow> = (0..row_count)
            .map(|index| {
                let time = i64::from(time_bits[index] % 20);
                let symbol = match entity_bits[index] % 3 {
                    0 => "a".to_owned(),
                    1 => "b".to_owned(),
                    _ => "c".to_owned(),
                };
                (
                    time,
                    symbol,
                    u64::try_from(index).unwrap() + 1,
                    price_value(price_bits[index]),
                    volume_value(volume_bits[index]),
                )
            })
            .collect();
        // The batch input order is the arrival order: a deterministic
        // shuffle of the canonical rows.
        let mut arrival: Vec<usize> = (0..row_count).collect();
        for index in (1..arrival.len()).rev() {
            let swap = (arrival_bits[index % arrival_bits.len()] as usize) % (index + 1);
            arrival.swap(index, swap);
        }
        let shuffled: Vec<InputRow> = arrival.iter().map(|&index| rows[index].clone()).collect();

        // Envelopes of one to four rows.
        let mut envelopes: Vec<Vec<InputRow>> = Vec::new();
        let mut cursor = 0;
        while cursor < shuffled.len() {
            let width = 1 + (chunk_bits[cursor % chunk_bits.len()] as usize) % 4;
            let end = (cursor + width).min(shuffled.len());
            envelopes.push(shuffled[cursor..end].to_vec());
            cursor = end;
        }

        let runtime = tokio::runtime::Builder::new_current_thread()
            .enable_all()
            .build()
            .unwrap();
        let batch_rows = runtime.block_on(run_batch(&shuffled, duration_micros, row_frame));
        let stream_rows = runtime.block_on(run_stream(&envelopes, duration_micros, row_frame));
        assert_rows_match(&stream_rows, &batch_rows);
    }
}
