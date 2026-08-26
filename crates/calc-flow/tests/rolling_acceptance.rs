//! Independent acceptance vectors for the rolling operator, authored by
//! cf-tester during the SCE-06 independent verification of PR #198 and
//! absorbed into the shipped suite: independently chosen fixtures,
//! segmentation and recovery boundaries, corruption cases, and non-goal
//! rejection vectors (NaN compared bitwise).

use std::{collections::BTreeMap, sync::Arc};

use calc_flow::{
    Batch, BatchMetadata, CancellationToken, EdgeCollector, Epoch, EventTime, JsonMap,
    OperatorMetadata, OperatorStateSnapshot, RollingOperator, RollingSpec, StreamCollector,
    StreamJobContext, StreamOperator, StreamOperatorContext,
};
use datafusion::arrow::{
    array::{
        Array, ArrayRef, Float64Array, Int64Array, StringArray, TimestampMicrosecondArray,
        UInt64Array,
    },
    datatypes::{DataType, Field, Schema, TimeUnit},
    record_batch::RecordBatch,
};

const FINGERPRINT: &str = "fedcba9876543210fedcba9876543210fedcba9876543210fedcba9876543210";

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

fn probe_spec(lag_periods: u64, delta_periods: u64, lateness: u64, drop: bool) -> RollingSpec {
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
                "output": "price_lag",
                "periods": lag_periods
            },
            {
                "kind": "delta",
                "primitive_version": 1,
                "input": "volume",
                "output": "volume_delta",
                "periods": delta_periods
            }
        ],
        "allowed_lateness_micros": lateness,
        "late_policy": if drop {
            serde_json::json!({"kind": "drop", "metrics_version": 1})
        } else {
            serde_json::json!({"kind": "error", "scope": "envelope"})
        },
        "value_policy": "stateful_numeric_v1"
    }))
    .unwrap()
}

fn make_operator(spec: RollingSpec) -> RollingOperator {
    RollingOperator::new("probe_rolling", input_schema(), spec).unwrap()
}

fn job() -> StreamJobContext {
    StreamJobContext::new(
        7,
        FINGERPRINT,
        JsonMap::new(),
        None,
        CancellationToken::new(),
    )
}

fn context<'a>(job: &'a StreamJobContext, watermark: Option<i64>) -> StreamOperatorContext<'a> {
    StreamOperatorContext::new(job, "probe_rolling", watermark.map(EventTime::from_micros))
}

fn collector_for(operator: &RollingOperator) -> EdgeCollector {
    EdgeCollector::new(operator.output_ports().to_vec())
}

/// Rows as (ts, symbol, seq, lag bits, delta); lag compared bitwise so NaN
/// preservation is exact.
#[derive(Debug, PartialEq)]
struct ProbeRow(i64, String, u64, Option<u64>, Option<i64>);

fn drain_rows(collector: &mut EdgeCollector, rows: &mut Vec<ProbeRow>) {
    for message in collector.drain("output") {
        let batch = message.as_data().unwrap();
        for record in batch.table_payload().unwrap().batches() {
            let ts = record
                .column_by_name("ts")
                .unwrap()
                .as_any()
                .downcast_ref::<TimestampMicrosecondArray>()
                .unwrap();
            let symbol = record
                .column_by_name("symbol")
                .unwrap()
                .as_any()
                .downcast_ref::<StringArray>()
                .unwrap();
            let sequence = record
                .column_by_name("sequence")
                .unwrap()
                .as_any()
                .downcast_ref::<UInt64Array>()
                .unwrap();
            let lag = record
                .column_by_name("price_lag")
                .unwrap()
                .as_any()
                .downcast_ref::<Float64Array>()
                .unwrap();
            let delta = record
                .column_by_name("volume_delta")
                .unwrap()
                .as_any()
                .downcast_ref::<Int64Array>()
                .unwrap();
            for index in 0..record.num_rows() {
                rows.push(ProbeRow(
                    ts.value(index),
                    symbol.value(index).to_owned(),
                    sequence.value(index),
                    lag.iter().nth(index).unwrap().map(f64::to_bits),
                    delta.iter().nth(index).unwrap(),
                ));
            }
        }
    }
}

/// Independent fixture: interleaved entities, duplicate timestamps within and
/// across entities, a null price, a null volume, and a NaN price.
fn probe_rows() -> Vec<(i64, &'static str, u64, Option<f64>, Option<i64>)> {
    vec![
        (12, "a", 3, Some(3.0), Some(120)),
        (10, "a", 1, Some(1.0), Some(100)),
        (15, "b", 4, Some(14.0), Some(800)),
        (11, "a", 2, None, Some(110)),
        (10, "b", 1, Some(10.0), Some(1000)),
        (14, "a", 5, Some(f64::NAN), Some(140)),
        (12, "a", 4, Some(3.5), Some(130)),
        (13, "b", 3, Some(13.0), None),
        (11, "b", 2, Some(11.0), Some(900)),
        (16, "a", 6, Some(6.0), Some(150)),
        (17, "a", 7, Some(7.0), Some(160)),
        (18, "a", 8, Some(8.0), Some(170)),
    ]
}

/// Independently derived expectation for lag(price, 3) / delta(volume, 2) over
/// `probe_rows` in canonical order.
fn expected_rows() -> Vec<ProbeRow> {
    let nan = f64::NAN.to_bits();
    vec![
        ProbeRow(10, "a".into(), 1, None, None),
        ProbeRow(10, "b".into(), 1, None, None),
        ProbeRow(11, "a".into(), 2, None, None),
        ProbeRow(11, "b".into(), 2, None, None),
        ProbeRow(12, "a".into(), 3, None, Some(20)),
        ProbeRow(12, "a".into(), 4, Some(1.0_f64.to_bits()), Some(20)),
        ProbeRow(13, "b".into(), 3, None, None),
        ProbeRow(14, "a".into(), 5, None, Some(20)),
        ProbeRow(15, "b".into(), 4, Some(10.0_f64.to_bits()), Some(-100)),
        ProbeRow(16, "a".into(), 6, Some(3.0_f64.to_bits()), Some(20)),
        ProbeRow(17, "a".into(), 7, Some(3.5_f64.to_bits()), Some(20)),
        ProbeRow(18, "a".into(), 8, Some(nan), Some(20)),
    ]
}

async fn batch_reference(rows: &[(i64, &str, u64, Option<f64>, Option<i64>)]) -> Vec<ProbeRow> {
    let plan = calc_flow::PipelineBuilder::new("cf tester probe equivalence")
        .unwrap()
        .add_node("probe_rolling", make_operator(probe_spec(3, 2, 0, false)))
        .unwrap()
        .compile_batch(&calc_flow::UdfRegistry::new().snapshot())
        .unwrap();
    let result = plan
        .execute(
            BTreeMap::from([("input".into(), input_batch(rows))]),
            calc_flow::ExecutionOptions::default(),
        )
        .await
        .unwrap();
    let mut collector = collector_for(&make_operator(probe_spec(3, 2, 0, false)));
    collector
        .emit("output", result.outputs["output"].clone())
        .await
        .unwrap();
    let mut observed = Vec::new();
    drain_rows(&mut collector, &mut observed);
    observed
}

async fn stream_run(
    segments: &[&[(i64, &str, u64, Option<f64>, Option<i64>)]],
    checkpoint_after: Option<usize>,
) -> Vec<ProbeRow> {
    let job = job();
    let mut observed = Vec::new();
    let mut head = make_operator(probe_spec(3, 2, 0, false));
    let mut head_collector = collector_for(&head);
    let cut = checkpoint_after.unwrap_or(usize::MAX);
    let mut driven = 0_usize;
    for segment in segments.iter().take(cut.min(segments.len())) {
        head.process_data(
            "input",
            input_batch(segment),
            &context(&job, None),
            &mut head_collector,
        )
        .await
        .unwrap();
        driven += 1;
    }
    drain_rows(&mut head_collector, &mut observed);
    let mut current = head;
    let mut collector = head_collector;
    if cut < segments.len() {
        let snapshot = current.checkpoint(Epoch::new(3).unwrap()).unwrap();
        let mut recovered = make_operator(probe_spec(3, 2, 0, false));
        recovered.restore(&snapshot).unwrap();
        let mut recovered_collector = collector_for(&recovered);
        for segment in segments.iter().skip(driven) {
            recovered
                .process_data(
                    "input",
                    input_batch(segment),
                    &context(&job, None),
                    &mut recovered_collector,
                )
                .await
                .unwrap();
        }
        current = recovered;
        collector = recovered_collector;
    }
    current
        .on_watermark(
            EventTime::from_micros(9_999),
            &context(&job, Some(9_999)),
            &mut collector,
        )
        .await
        .unwrap();
    current
        .on_end(&context(&job, Some(9_999)), &mut collector)
        .await
        .unwrap();
    drain_rows(&mut collector, &mut observed);
    observed
}

#[tokio::test]
async fn probe_batch_matches_independent_expectation() {
    let observed = batch_reference(&probe_rows()).await;
    assert_eq!(observed, expected_rows());
}

#[tokio::test]
async fn probe_stream_equals_batch_under_probe_segmentation_and_recovery() {
    let rows = probe_rows();
    let expected = batch_reference(&rows).await;

    let whole = stream_run(&[&rows], None).await;
    assert_eq!(whole, expected, "single-envelope stream drifted");

    let segmented = stream_run(&[&rows[..5], &rows[5..9], &rows[9..]], None).await;
    assert_eq!(segmented, expected, "segmented stream drifted");

    let recovered_early = stream_run(&[&rows[..5], &rows[5..9], &rows[9..]], Some(1)).await;
    assert_eq!(
        recovered_early, expected,
        "recovery after one envelope drifted"
    );

    let recovered_late = stream_run(&[&rows[..5], &rows[5..9], &rows[9..]], Some(2)).await;
    assert_eq!(
        recovered_late, expected,
        "recovery after two envelopes drifted"
    );
}

async fn replay_d7_second_envelope(snapshot: &OperatorStateSnapshot) -> OperatorStateSnapshot {
    let job = job();
    let mut recovered = make_operator(probe_spec(1, 1, 5, true));
    recovered.restore(snapshot).unwrap();
    let mut recovered_collector = collector_for(&recovered);
    recovered
        .process_data(
            "input",
            input_batch(&[
                (118, "a", 4, Some(4.0), Some(40)),
                (120, "a", 5, Some(5.0), Some(50)),
            ]),
            &context(&job, Some(123)),
            &mut recovered_collector,
        )
        .await
        .unwrap();
    recovered.checkpoint(Epoch::new(2).unwrap()).unwrap()
}

#[tokio::test]
async fn probe_d7_frozen_vector_recomputes_and_replay_does_not_double_count() {
    let job = job();
    let mut operator = make_operator(probe_spec(1, 1, 5, true));
    let mut collector = collector_for(&operator);
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
    let first = operator.checkpoint(Epoch::new(1).unwrap()).unwrap();
    assert_eq!(
        first.inline_metadata["metrics"],
        serde_json::json!({
            "late_rows": 2,
            "affected_batches": 1,
            "max_lateness_micros": 6,
            "null_event_time_rows": 0,
            "null_event_time_batches": 0
        }),
        "first half of the frozen D7 vector drifted"
    );

    let second = replay_d7_second_envelope(&first).await;
    assert_eq!(
        second.inline_metadata["metrics"],
        serde_json::json!({
            "late_rows": 3,
            "affected_batches": 2,
            "max_lateness_micros": 6,
            "null_event_time_rows": 0,
            "null_event_time_batches": 0
        }),
        "second half of the frozen D7 vector drifted"
    );

    let replayed = replay_d7_second_envelope(&first).await;
    assert_eq!(
        replayed.inline_metadata["metrics"], second.inline_metadata["metrics"],
        "replaying the post-crash envelope double-counted the D7 metrics"
    );
}

#[tokio::test]
async fn probe_lateness_boundary_is_inclusive_and_undefined_watermark_accepts() {
    let job = job();
    let mut operator = make_operator(probe_spec(1, 1, 5, true));
    let mut collector = collector_for(&operator);
    operator
        .process_data(
            "input",
            input_batch(&[
                (115, "a", 1, Some(1.0), Some(10)),
                (116, "a", 2, Some(2.0), Some(20)),
            ]),
            &context(&job, Some(120)),
            &mut collector,
        )
        .await
        .unwrap();
    let snapshot = operator.checkpoint(Epoch::new(1).unwrap()).unwrap();
    assert_eq!(
        snapshot.inline_metadata["metrics"]["late_rows"],
        serde_json::json!(1),
        "closing coordinate t + L == W must be late, t + L == W + 1 must not"
    );

    let mut fresh = make_operator(probe_spec(1, 1, 5, true));
    let mut fresh_collector = collector_for(&fresh);
    fresh
        .process_data(
            "input",
            input_batch(&[(1, "a", 1, Some(9.0), Some(90))]),
            &context(&job, None),
            &mut fresh_collector,
        )
        .await
        .unwrap();
    let undefined = fresh.checkpoint(Epoch::new(1).unwrap()).unwrap();
    assert_eq!(
        undefined.inline_metadata["metrics"]["late_rows"],
        serde_json::json!(0),
        "an undefined watermark must not classify any row as late"
    );
}

#[tokio::test]
async fn probe_redelivery_after_emission_is_dropped_without_duplicate_output() {
    let job = job();
    let mut operator = make_operator(probe_spec(1, 1, 5, true));
    let mut collector = collector_for(&operator);
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
        .on_watermark(
            EventTime::from_micros(30),
            &context(&job, Some(30)),
            &mut collector,
        )
        .await
        .unwrap();
    let mut observed = Vec::new();
    drain_rows(&mut collector, &mut observed);
    assert_eq!(observed.len(), 1, "first delivery should emit exactly once");

    operator
        .process_data(
            "input",
            input_batch(&[(20, "a", 1, Some(2.0), Some(20))]),
            &context(&job, Some(30)),
            &mut collector,
        )
        .await
        .unwrap();
    operator
        .on_end(&context(&job, Some(30)), &mut collector)
        .await
        .unwrap();
    drain_rows(&mut collector, &mut observed);
    assert_eq!(
        observed.len(),
        1,
        "at-least-once redelivery after emission duplicated output"
    );
}

#[tokio::test]
async fn probe_restore_rejects_corrupt_segment_and_leaves_state_untouched() {
    let job = job();
    let mut operator = make_operator(probe_spec(1, 1, 0, false));
    let mut collector = collector_for(&operator);
    operator
        .process_data(
            "input",
            input_batch(&probe_rows()[..3]),
            &context(&job, None),
            &mut collector,
        )
        .await
        .unwrap();
    let snapshot = operator.checkpoint(Epoch::new(1).unwrap()).unwrap();

    let mut corrupted = snapshot.clone();
    for bytes in corrupted.segments.values_mut() {
        let middle = bytes.len() / 2;
        bytes[middle] ^= 0xFF;
    }
    let mut target = make_operator(probe_spec(1, 1, 0, false));
    assert!(
        target.restore(&corrupted).is_err(),
        "a length-preserving payload corruption must fail the checksum"
    );

    target.restore(&snapshot).unwrap();
    let mut target_collector = collector_for(&target);
    target
        .on_end(&context(&job, None), &mut target_collector)
        .await
        .unwrap();
    let mut observed = Vec::new();
    drain_rows(&mut target_collector, &mut observed);
    assert_eq!(observed.len(), 3, "valid restore after corruption mismatch");
}

#[tokio::test]
async fn probe_restore_rejects_a_different_configuration() {
    let job = job();
    let mut source = make_operator(probe_spec(1, 1, 0, false));
    let mut collector = collector_for(&source);
    source
        .process_data(
            "input",
            input_batch(&probe_rows()[..1]),
            &context(&job, None),
            &mut collector,
        )
        .await
        .unwrap();
    let snapshot = source.checkpoint(Epoch::new(1).unwrap()).unwrap();

    let mut different_periods = make_operator(probe_spec(2, 1, 0, false));
    assert!(
        different_periods.restore(&snapshot).is_err(),
        "restore accepted a snapshot from a different periods configuration"
    );
}

#[tokio::test]
async fn probe_empty_snapshot_restore_resets_and_epochs_advance_strictly() {
    let job = job();
    let mut operator = make_operator(probe_spec(1, 1, 0, false));
    let mut collector = collector_for(&operator);
    operator
        .process_data(
            "input",
            input_batch(&probe_rows()[..2]),
            &context(&job, None),
            &mut collector,
        )
        .await
        .unwrap();

    let empty = OperatorStateSnapshot {
        inline_metadata: JsonMap::new(),
        segments: BTreeMap::new(),
    };
    operator.restore(&empty).unwrap();
    operator
        .on_end(&context(&job, None), &mut collector)
        .await
        .unwrap();
    let mut observed = Vec::new();
    drain_rows(&mut collector, &mut observed);
    assert!(
        observed.is_empty(),
        "restore of an empty snapshot must reset buffered rows"
    );

    operator.checkpoint(Epoch::new(2).unwrap()).unwrap();
    assert!(
        operator.checkpoint(Epoch::new(2).unwrap()).is_err(),
        "a non-advancing checkpoint epoch must be rejected"
    );
    assert!(
        operator.checkpoint(Epoch::new(1).unwrap()).is_err(),
        "a regressing checkpoint epoch must be rejected"
    );
}

#[tokio::test]
async fn probe_null_event_time_data_fails_loudly_in_batch_mode() {
    let malformed = RecordBatch::try_new(
        input_schema(),
        vec![
            Arc::new(TimestampMicrosecondArray::from(vec![Some(10_i64), None]).with_timezone("UTC"))
                as ArrayRef,
            Arc::new(StringArray::from(vec!["a", "a"])),
            Arc::new(UInt64Array::from(vec![1_u64, 2])),
            Arc::new(Float64Array::from(vec![Some(1.0), Some(2.0)])),
            Arc::new(Int64Array::from(vec![Some(10), Some(20)])),
        ],
    );
    assert!(
        malformed.is_err(),
        "null event-time data must be rejected loudly at batch construction"
    );
}

#[test]
fn probe_ambiguous_input_column_is_rejected() {
    let schema = Arc::new(Schema::new(vec![
        Field::new(
            "ts",
            DataType::Timestamp(TimeUnit::Microsecond, Some(Arc::from("UTC"))),
            false,
        ),
        Field::new("symbol", DataType::Utf8, false),
        Field::new("sequence", DataType::UInt64, false),
        Field::new("price", DataType::Float64, true),
        Field::new("price", DataType::Float64, true),
        Field::new("volume", DataType::Int64, true),
    ]));
    let error =
        RollingOperator::new("probe_rolling", schema, probe_spec(1, 1, 0, false)).unwrap_err();
    assert!(
        matches!(error, calc_flow::CalcFlowError::Compile { .. }),
        "ambiguous input columns must be rejected, got: {error}"
    );
}

#[test]
fn probe_non_goal_output_kinds_and_frames_are_rejected() {
    let base = serde_json::json!({
        "configuration_version": 1,
        "state_layout_version": 1,
        "partition_by": ["symbol"],
        "event_time": "ts",
        "sequence_by": ["sequence"],
        "allowed_lateness_micros": 0,
        "late_policy": {"kind": "error", "scope": "envelope"},
        "value_policy": "stateful_numeric_v1"
    });
    for kind in ["count", "sum", "mean", "variance", "std", "stddev"] {
        let mut document = base.clone();
        document["outputs"] = serde_json::json!([{
            "kind": kind,
            "primitive_version": 1,
            "input": "price",
            "output": "price_aggregate",
            "frame": {"kind": "rows", "size": 5},
            "min_periods": 1
        }]);
        assert!(
            serde_json::from_value::<RollingSpec>(document).is_err(),
            "aggregate catalog kind {kind} must stay rejected"
        );
    }
    let mut duration = base.clone();
    duration["outputs"] = serde_json::json!([{
        "kind": "lag",
        "primitive_version": 1,
        "input": "price",
        "output": "price_lag",
        "periods": 1,
        "frame": {"kind": "duration", "micros": 1_000}
    }]);
    assert!(
        serde_json::from_value::<RollingSpec>(duration).is_err(),
        "duration frames must stay rejected on lag/delta outputs"
    );
    let mut cross_section = base.clone();
    cross_section["outputs"] = serde_json::json!([{
        "kind": "cross_section",
        "primitive_version": 1,
        "input": "price",
        "output": "price_cs",
        "periods": 1
    }]);
    assert!(
        serde_json::from_value::<RollingSpec>(cross_section).is_err(),
        "cross_section output kinds must stay rejected"
    );
}
