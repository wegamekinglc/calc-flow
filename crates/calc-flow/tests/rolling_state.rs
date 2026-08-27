use std::{collections::BTreeMap, sync::Arc};

use calc_flow::{
    Batch, BatchMetadata, CancellationToken, EdgeCollector, Epoch, EventTime, JsonMap,
    LocalStateBackend, OperatorMetadata, OperatorStateSnapshot, RollingOperator, RollingSpec,
    StateBackend, StateHandle, StateLineageKey, StateSegment, StreamCollector, StreamJobContext,
    StreamOperator, StreamOperatorContext,
};
use datafusion::arrow::{
    array::{
        Array, ArrayRef, Float64Array, Int64Array, StringArray, TimestampMicrosecondArray,
        UInt8Array, UInt64Array,
    },
    datatypes::{DataType, Field, Schema, TimeUnit},
    record_batch::RecordBatch,
};
use sha2::{Digest, Sha256};

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

type InputRow = (i64, &'static str, u64, Option<f64>, Option<i64>);

fn input_batch(rows: &[InputRow]) -> Batch {
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

fn spec(late_policy: &serde_json::Value, allowed_lateness_micros: u64) -> RollingSpec {
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
        spec(
            &serde_json::json!({"kind": "error", "scope": "envelope"}),
            0,
        ),
    )
    .unwrap()
}

fn drop_operator(lateness: u64) -> RollingOperator {
    RollingOperator::new(
        "rolling",
        input_schema(),
        spec(
            &serde_json::json!({"kind": "drop", "metrics_version": 1}),
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

fn context(job: &StreamJobContext, watermark: Option<i64>) -> StreamOperatorContext<'_> {
    StreamOperatorContext::new(job, "rolling", watermark.map(EventTime::from_micros))
}

fn new_collector() -> EdgeCollector {
    EdgeCollector::new(error_operator().output_ports().to_vec())
}

type ObservedRow = (i64, String, u64, Option<f64>, Option<i64>);

#[derive(Clone, Default, PartialEq, Debug)]
struct Observed {
    rows: Vec<ObservedRow>,
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
                observed.rows.push((
                    event_times.value(index),
                    symbols.value(index).to_owned(),
                    sequences.value(index),
                    lags.iter().nth(index).unwrap(),
                    deltas.iter().nth(index).unwrap(),
                ));
            }
        }
    }
}

fn fixture_rows() -> Vec<InputRow> {
    vec![
        (10, "a", 1, Some(1.0), Some(10)),
        (10, "b", 1, Some(5.0), Some(50)),
        (11, "a", 2, Some(2.0), Some(20)),
        (12, "b", 2, Some(6.0), Some(60)),
        (12, "a", 3, Some(3.0), Some(30)),
        (13, "b", 3, Some(7.0), Some(70)),
        (14, "a", 4, Some(4.0), Some(40)),
    ]
}

async fn drive_segmented(
    operator: &mut RollingOperator,
    chunks: &[&[InputRow]],
    job: &StreamJobContext,
    collector: &mut EdgeCollector,
) {
    for chunk in chunks {
        operator
            .process_data("input", input_batch(chunk), &context(job, None), collector)
            .await
            .unwrap();
    }
}

async fn finish(
    operator: &mut RollingOperator,
    job: &StreamJobContext,
    collector: &mut EdgeCollector,
    observed: &mut Observed,
) {
    operator
        .on_watermark(
            EventTime::from_micros(1_000),
            &context(job, Some(1_000)),
            collector,
        )
        .await
        .unwrap();
    operator
        .on_end(&context(job, Some(1_000)), collector)
        .await
        .unwrap();
    drain(collector, observed);
}

#[tokio::test]
async fn checkpoint_restore_continues_without_duplicate_or_missing_rows() {
    let job = job();
    let rows = fixture_rows();
    let chunks: Vec<&[_]> = rows.chunks(3).collect();

    let mut reference = error_operator();
    let mut reference_collector = new_collector();
    let mut reference_observed = Observed::default();
    drive_segmented(&mut reference, &chunks, &job, &mut reference_collector).await;
    finish(
        &mut reference,
        &job,
        &mut reference_collector,
        &mut reference_observed,
    )
    .await;

    let mut restarted = error_operator();
    let mut recovered = error_operator();
    let mut restarted_collector = new_collector();
    let restarted_observed = Observed::default();
    drive_segmented(&mut restarted, &chunks[..2], &job, &mut restarted_collector).await;
    let snapshot = restarted.checkpoint(Epoch::new(1).unwrap()).unwrap();
    recovered.restore(&snapshot).unwrap();
    let mut recovered_collector = new_collector();
    let mut recovered_observed = restarted_observed.clone();
    drive_segmented(&mut recovered, &chunks[2..], &job, &mut recovered_collector).await;
    finish(
        &mut recovered,
        &job,
        &mut recovered_collector,
        &mut recovered_observed,
    )
    .await;

    assert_eq!(reference_observed, recovered_observed);
}

#[tokio::test]
async fn restored_frontier_sequence_and_metrics_continue() {
    let job = job();
    let mut operator = error_operator();
    let mut collector = new_collector();
    let mut observed = Observed::default();

    operator
        .process_data(
            "input",
            input_batch(&fixture_rows()[..4]),
            &context(&job, None),
            &mut collector,
        )
        .await
        .unwrap();
    operator
        .on_watermark(
            EventTime::from_micros(11),
            &context(&job, Some(11)),
            &mut collector,
        )
        .await
        .unwrap();
    drain(&mut collector, &mut observed);
    let snapshot = operator.checkpoint(Epoch::new(1).unwrap()).unwrap();

    let mut recovered = error_operator();
    recovered.restore(&snapshot).unwrap();
    let error = recovered
        .on_watermark(
            EventTime::from_micros(11),
            &context(&job, Some(11)),
            &mut collector,
        )
        .await
        .unwrap_err();
    assert!(
        matches!(error, calc_flow::CalcFlowError::Operator { .. }),
        "unexpected error: {error}"
    );

    let mut recovered_collector = new_collector();
    finish(
        &mut recovered,
        &job,
        &mut recovered_collector,
        &mut observed,
    )
    .await;
    assert_eq!(
        observed.rows,
        vec![
            (10, "a".into(), 1, None, None),
            (10, "b".into(), 1, None, None),
            (11, "a".into(), 2, Some(1.0), Some(10)),
            (12, "b".into(), 2, Some(5.0), Some(10)),
        ]
    );
    assert_eq!(observed.batch_sequences, vec![0, 1]);
}

#[tokio::test]
async fn every_segmentation_and_recovery_boundary_restores_the_same_final_rows() {
    let job = job();
    let rows = fixture_rows();

    let mut reference = error_operator();
    let mut reference_collector = new_collector();
    let mut expected = Observed::default();
    reference
        .process_data(
            "input",
            input_batch(&rows),
            &context(&job, None),
            &mut reference_collector,
        )
        .await
        .unwrap();
    finish(
        &mut reference,
        &job,
        &mut reference_collector,
        &mut expected,
    )
    .await;

    for boundary in 1..rows.len() {
        let mut head = error_operator();
        let mut head_collector = new_collector();
        let mut observed = Observed::default();
        head.process_data(
            "input",
            input_batch(&rows[..boundary]),
            &context(&job, None),
            &mut head_collector,
        )
        .await
        .unwrap();
        let snapshot = head.checkpoint(Epoch::new(1).unwrap()).unwrap();

        let mut recovered = error_operator();
        recovered.restore(&snapshot).unwrap();
        let mut recovered_collector = new_collector();
        recovered
            .process_data(
                "input",
                input_batch(&rows[boundary..]),
                &context(&job, None),
                &mut recovered_collector,
            )
            .await
            .unwrap();
        finish(
            &mut recovered,
            &job,
            &mut recovered_collector,
            &mut observed,
        )
        .await;
        assert_eq!(observed.rows, expected.rows, "boundary {boundary} drifted");
    }
}

#[tokio::test]
async fn batch_and_final_stream_produce_the_same_ordered_rows_after_recovery() {
    let job = job();
    let rows = fixture_rows();

    let plan = calc_flow::PipelineBuilder::new("rolling equivalence")
        .unwrap()
        .add_node("rolling", error_operator())
        .unwrap()
        .compile_batch(&calc_flow::UdfRegistry::new().snapshot())
        .unwrap();
    let batch_result = plan
        .execute(
            BTreeMap::from([("input".into(), input_batch(&rows))]),
            calc_flow::ExecutionOptions::default(),
        )
        .await
        .unwrap();
    let mut batch_observed = Observed::default();
    let mut batch_collector = new_collector();
    batch_collector
        .emit("output", batch_result.outputs["output"].clone())
        .await
        .unwrap();
    drain(&mut batch_collector, &mut batch_observed);

    let mut head = error_operator();
    let mut head_collector = new_collector();
    let mut stream_observed = Observed::default();
    head.process_data(
        "input",
        input_batch(&rows[..3]),
        &context(&job, None),
        &mut head_collector,
    )
    .await
    .unwrap();
    let snapshot = head.checkpoint(Epoch::new(1).unwrap()).unwrap();
    let mut recovered = error_operator();
    recovered.restore(&snapshot).unwrap();
    let mut recovered_collector = new_collector();
    recovered
        .process_data(
            "input",
            input_batch(&rows[3..]),
            &context(&job, None),
            &mut recovered_collector,
        )
        .await
        .unwrap();
    finish(
        &mut recovered,
        &job,
        &mut recovered_collector,
        &mut stream_observed,
    )
    .await;

    assert_eq!(batch_observed.rows, stream_observed.rows);
}

#[tokio::test]
async fn invalid_restore_has_no_side_effect_and_repeated_restore_is_idempotent() {
    let job = job();
    let mut operator = error_operator();
    let mut collector = new_collector();
    operator
        .process_data(
            "input",
            input_batch(&fixture_rows()[..2]),
            &context(&job, None),
            &mut collector,
        )
        .await
        .unwrap();
    let snapshot = operator.checkpoint(Epoch::new(1).unwrap()).unwrap();

    let mut tampered = snapshot.clone();
    let first_segment = tampered.segments.keys().next().cloned();
    if let Some(segment_id) = first_segment {
        let segment = tampered.segments.get_mut(&segment_id).unwrap();
        let mut bytes = segment.bytes().to_vec();
        bytes.push(0xAA);
        *segment = StateSegment::new(bytes);
    }
    let mut target = error_operator();
    if !tampered.segments.is_empty() {
        assert!(target.restore(&tampered).is_err());
    }
    target.restore(&snapshot).unwrap();
    target.restore(&snapshot).unwrap();

    let mut recovered_collector = new_collector();
    let mut observed = Observed::default();
    finish(&mut target, &job, &mut recovered_collector, &mut observed).await;
    assert_eq!(
        observed.rows,
        vec![
            (10, "a".into(), 1, None, None),
            (10, "b".into(), 1, None, None),
        ]
    );
}

#[tokio::test]
async fn restore_rejects_layout_configuration_and_segment_mismatches() {
    let job = job();
    let mut operator = error_operator();
    let mut collector = new_collector();
    operator
        .process_data(
            "input",
            input_batch(&fixture_rows()[..1]),
            &context(&job, None),
            &mut collector,
        )
        .await
        .unwrap();
    let snapshot = operator.checkpoint(Epoch::new(1).unwrap()).unwrap();

    let mut wrong_layout = snapshot.clone();
    wrong_layout
        .inline_metadata
        .insert("state_layout_version".into(), serde_json::json!(99));
    assert!(error_operator().restore(&wrong_layout).is_err());

    let mut wrong_configuration = snapshot.clone();
    wrong_configuration.inline_metadata.insert(
        "configuration_hash".into(),
        serde_json::json!("00".repeat(32)),
    );
    assert!(error_operator().restore(&wrong_configuration).is_err());

    let mut missing_segment = snapshot.clone();
    missing_segment.segments.clear();
    if !snapshot.segments.is_empty() {
        assert!(error_operator().restore(&missing_segment).is_err());
    }

    let different_spec = spec(
        &serde_json::json!({"kind": "drop", "metrics_version": 1}),
        5,
    );
    let mut different_operator =
        RollingOperator::new("rolling", input_schema(), different_spec).unwrap();
    assert!(different_operator.restore(&snapshot).is_err());
}

#[tokio::test]
async fn late_metrics_survive_checkpoint_restore_with_the_frozen_d7_vector() {
    let job = job();
    let mut operator = drop_operator(5);
    let mut collector = new_collector();

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
        })
    );

    let mut recovered = drop_operator(5);
    recovered.restore(&first).unwrap();
    recovered
        .process_data(
            "input",
            input_batch(&[
                (118, "a", 4, Some(4.0), Some(40)),
                (120, "a", 5, Some(5.0), Some(50)),
            ]),
            &context(&job, Some(123)),
            &mut collector,
        )
        .await
        .unwrap();
    let second = recovered.checkpoint(Epoch::new(2).unwrap()).unwrap();
    assert_eq!(
        second.inline_metadata["metrics"],
        serde_json::json!({
            "late_rows": 3,
            "affected_batches": 2,
            "max_lateness_micros": 6,
            "null_event_time_rows": 0,
            "null_event_time_batches": 0
        })
    );
}

#[tokio::test]
async fn reset_clears_every_state_component() {
    let job = job();
    let mut operator = error_operator();
    let mut collector = new_collector();
    operator
        .process_data(
            "input",
            input_batch(&fixture_rows()[..2]),
            &context(&job, None),
            &mut collector,
        )
        .await
        .unwrap();
    operator
        .on_watermark(
            EventTime::from_micros(50),
            &context(&job, Some(50)),
            &mut collector,
        )
        .await
        .unwrap();
    operator.reset().unwrap();

    let mut observed = Observed::default();
    operator
        .on_watermark(
            EventTime::from_micros(10),
            &context(&job, Some(10)),
            &mut collector,
        )
        .await
        .unwrap();
    operator
        .process_data(
            "input",
            input_batch(&fixture_rows()[..1]),
            &context(&job, None),
            &mut collector,
        )
        .await
        .unwrap();
    finish(&mut operator, &job, &mut collector, &mut observed).await;
    assert_eq!(
        observed.rows,
        vec![
            (10, "a".into(), 1, None, None),
            (10, "b".into(), 1, None, None),
            (10, "a".into(), 1, None, None),
        ]
    );
    assert_eq!(observed.batch_sequences, vec![0, 0]);
}

#[tokio::test]
async fn terminal_restore_does_not_repeat_final_rows() {
    let job = job();
    let mut operator = error_operator();
    let mut collector = new_collector();
    let mut observed = Observed::default();
    operator
        .process_data(
            "input",
            input_batch(&fixture_rows()[..2]),
            &context(&job, None),
            &mut collector,
        )
        .await
        .unwrap();
    finish(&mut operator, &job, &mut collector, &mut observed).await;
    let terminal = operator.checkpoint(Epoch::new(1).unwrap()).unwrap();
    assert_eq!(observed.rows.len(), 2);

    let mut recovered = error_operator();
    recovered.restore(&terminal).unwrap();
    let mut recovered_collector = new_collector();
    let mut recovered_observed = Observed::default();
    recovered
        .on_watermark(
            EventTime::from_micros(1_001),
            &context(&job, Some(1_001)),
            &mut recovered_collector,
        )
        .await
        .unwrap();
    recovered
        .on_end(&context(&job, Some(1_001)), &mut recovered_collector)
        .await
        .unwrap();
    drain(&mut recovered_collector, &mut recovered_observed);
    assert!(recovered_observed.rows.is_empty());
}

#[tokio::test]
async fn snapshot_segments_round_trip_through_the_validating_local_backend() {
    let job = job();
    let mut operator = error_operator();
    let mut collector = new_collector();
    operator
        .process_data(
            "input",
            input_batch(&fixture_rows()[..3]),
            &context(&job, None),
            &mut collector,
        )
        .await
        .unwrap();
    let snapshot = operator.checkpoint(Epoch::new(1).unwrap()).unwrap();
    assert!(!snapshot.segments.is_empty());

    let root = tempfile::TempDir::new().unwrap();
    let backend = LocalStateBackend::new(root.path()).await.unwrap();
    let key = StateLineageKey::new("rolling_equivalence", FINGERPRINT).unwrap();
    let lineage = backend.open_lineage(&key).await.unwrap();

    let digest = |bytes: &[u8]| hex::encode(Sha256::digest(bytes));
    let lineage_hash =
        digest(format!("{}\0{}", key.pipeline_name(), key.pipeline_fingerprint()).as_bytes());
    let mut restored_segments = BTreeMap::new();
    for (segment_id, segment) in &snapshot.segments {
        let bytes = segment.bytes();
        let handle = StateHandle::new(
            "rolling",
            Epoch::new(1).unwrap(),
            segment_id,
            &format!(
                "committed/{}/{}/{}-{}.arrow",
                lineage_hash,
                digest(b"rolling"),
                1,
                digest(segment_id.as_bytes())
            ),
            u64::try_from(bytes.len()).unwrap(),
            &digest(bytes),
        )
        .unwrap();
        lineage.stage_segment(&handle, bytes).await.unwrap();
        lineage.validate_segment(&handle).await.unwrap();
        lineage.publish_segment(&handle).await.unwrap();
        restored_segments.insert(
            segment_id.clone(),
            StateSegment::new(lineage.load_segment(&handle).await.unwrap()),
        );
    }

    let restored = OperatorStateSnapshot {
        inline_metadata: snapshot.inline_metadata.clone(),
        segments: restored_segments,
    };
    let mut recovered = error_operator();
    recovered.restore(&restored).unwrap();
    let mut recovered_collector = new_collector();
    let mut observed = Observed::default();
    finish(
        &mut recovered,
        &job,
        &mut recovered_collector,
        &mut observed,
    )
    .await;
    assert_eq!(
        observed.rows,
        vec![
            (10, "a".into(), 1, None, None),
            (10, "b".into(), 1, None, None),
            (11, "a".into(), 2, Some(1.0), Some(10)),
        ]
    );
}

// ---------------------------------------------------------------------------
// P1: identity, frontier, extreme-value, and corruption-matrix coverage.
// ---------------------------------------------------------------------------

fn different_job() -> StreamJobContext {
    StreamJobContext::new(
        2,
        "ffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffff",
        JsonMap::new(),
        None,
        CancellationToken::new(),
    )
}

#[tokio::test]
async fn context_identity_mismatch_is_rejected() {
    let job = job();
    let mut operator = error_operator();
    let mut collector = new_collector();
    operator
        .process_data(
            "input",
            input_batch(&fixture_rows()[..1]),
            &context(&job, None),
            &mut collector,
        )
        .await
        .unwrap();

    let other = different_job();
    let error = operator
        .on_watermark(
            EventTime::from_micros(100),
            &context(&other, Some(100)),
            &mut collector,
        )
        .await
        .unwrap_err();
    assert!(
        matches!(error, calc_flow::CalcFlowError::Operator { .. }),
        "unexpected error: {error}"
    );

    let wrong_operator_context = StreamOperatorContext::new(&job, "other_rolling", None);
    let error = operator
        .process_data(
            "input",
            input_batch(&fixture_rows()[..1]),
            &wrong_operator_context,
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
async fn zero_row_checkpoint_after_watermark_restores_empty_state() {
    let job = job();
    let mut operator = error_operator();
    let mut collector = new_collector();
    operator
        .on_watermark(
            EventTime::from_micros(50),
            &context(&job, Some(50)),
            &mut collector,
        )
        .await
        .unwrap();
    let snapshot = operator.checkpoint(Epoch::new(1).unwrap()).unwrap();
    assert!(snapshot.segments.is_empty());

    let mut recovered = error_operator();
    recovered.restore(&snapshot).unwrap();
    let error = recovered
        .on_watermark(
            EventTime::from_micros(50),
            &context(&job, Some(50)),
            &mut collector,
        )
        .await
        .unwrap_err();
    assert!(
        matches!(error, calc_flow::CalcFlowError::Operator { .. }),
        "the restored watermark frontier must reject a non-advancing watermark: {error}"
    );
    recovered
        .on_watermark(
            EventTime::from_micros(51),
            &context(&job, Some(51)),
            &mut collector,
        )
        .await
        .unwrap();
}

fn periods_two_spec() -> RollingSpec {
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
                "periods": 2
            },
            {
                "kind": "delta",
                "primitive_version": 1,
                "input": "volume",
                "output": "volume_delta_1",
                "periods": 2
            }
        ],
        "allowed_lateness_micros": 0,
        "late_policy": {"kind": "error", "scope": "envelope"},
        "value_policy": "stateful_numeric_v1"
    }))
    .unwrap()
}

#[tokio::test]
async fn periods_two_history_survives_emission_checkpoint_and_restore() {
    let job = job();
    let spec = periods_two_spec();
    let mut operator = RollingOperator::new("rolling", input_schema(), spec.clone()).unwrap();
    let mut collector = new_collector();
    let mut observed = Observed::default();
    operator
        .process_data(
            "input",
            input_batch(&fixture_rows()[..4]),
            &context(&job, None),
            &mut collector,
        )
        .await
        .unwrap();
    operator
        .on_watermark(
            EventTime::from_micros(12),
            &context(&job, Some(12)),
            &mut collector,
        )
        .await
        .unwrap();
    drain(&mut collector, &mut observed);
    let snapshot = operator.checkpoint(Epoch::new(1).unwrap()).unwrap();

    let mut recovered = RollingOperator::new("rolling", input_schema(), spec).unwrap();
    recovered.restore(&snapshot).unwrap();
    let mut recovered_collector = new_collector();
    recovered
        .process_data(
            "input",
            input_batch(&fixture_rows()[4..6]),
            &context(&job, None),
            &mut recovered_collector,
        )
        .await
        .unwrap();
    finish(
        &mut recovered,
        &job,
        &mut recovered_collector,
        &mut observed,
    )
    .await;
    assert_eq!(
        observed.rows,
        vec![
            (10, "a".into(), 1, None, None),
            (10, "b".into(), 1, None, None),
            (11, "a".into(), 2, None, None),
            (12, "b".into(), 2, None, None),
            (12, "a".into(), 3, Some(1.0), Some(20)),
            (13, "b".into(), 3, Some(5.0), Some(20)),
        ]
    );
}

#[tokio::test]
async fn extreme_lateness_values_are_checked_not_wrapped() {
    let job = job();
    let mut maximal = drop_operator(u64::MAX);
    let mut collector = new_collector();
    let error = maximal
        .process_data(
            "input",
            input_batch(&fixture_rows()[..1]),
            &context(&job, Some(100)),
            &mut collector,
        )
        .await
        .unwrap_err();
    assert!(
        matches!(error, calc_flow::CalcFlowError::Operator { .. }),
        "u64::MAX lateness must be checked: {error}"
    );

    let mut operator = drop_operator(1);
    let error = operator
        .process_data(
            "input",
            input_batch(&[(i64::MAX, "a", 1, Some(1.0), Some(10))]),
            &context(&job, Some(i64::MAX)),
            &mut collector,
        )
        .await
        .unwrap_err();
    assert!(
        matches!(error, calc_flow::CalcFlowError::Operator { .. }),
        "finality overflow must be checked: {error}"
    );

    let mut operator = drop_operator(1);
    operator
        .process_data(
            "input",
            input_batch(&[(i64::MAX, "a", 1, Some(1.0), Some(10))]),
            &context(&job, None),
            &mut collector,
        )
        .await
        .unwrap();
    let error = operator
        .on_watermark(
            EventTime::from_micros(i64::MAX - 1),
            &context(&job, Some(i64::MAX - 1)),
            &mut collector,
        )
        .await
        .unwrap_err();
    assert!(
        matches!(error, calc_flow::CalcFlowError::Operator { .. }),
        "closing-coordinate overflow at emission must be checked: {error}"
    );
}

#[tokio::test]
async fn malformed_inline_metadata_is_rejected_without_side_effects() {
    let job = job();
    let mut operator = error_operator();
    let mut collector = new_collector();
    operator
        .process_data(
            "input",
            input_batch(&fixture_rows()[..1]),
            &context(&job, None),
            &mut collector,
        )
        .await
        .unwrap();
    let snapshot = operator.checkpoint(Epoch::new(1).unwrap()).unwrap();

    let mut malformed = snapshot.clone();
    malformed
        .inline_metadata
        .insert("epoch".into(), serde_json::json!("not-a-number"));
    let mut target = error_operator();
    assert!(target.restore(&malformed).is_err());

    target.restore(&snapshot).unwrap();
    let mut recovered_collector = new_collector();
    let mut observed = Observed::default();
    finish(&mut target, &job, &mut recovered_collector, &mut observed).await;
    assert_eq!(observed.rows, vec![(10, "a".into(), 1, None, None)]);
}

fn reencode_segment(
    snapshot: &OperatorStateSnapshot,
    mutate: impl FnOnce(&Schema, RecordBatch) -> (Schema, RecordBatch),
) -> OperatorStateSnapshot {
    use datafusion::arrow::ipc::{reader::FileReader, writer::FileWriter};

    let (segment_id, segment) = snapshot.segments.iter().next().unwrap();
    let mut reader = FileReader::try_new(std::io::Cursor::new(segment.bytes()), None).unwrap();
    let schema = reader.schema().as_ref().clone();
    let record = reader.next().unwrap().unwrap();
    let (new_schema, new_record) = mutate(&schema, record);
    let mut bytes = Vec::new();
    {
        let mut writer = FileWriter::try_new(&mut bytes, &new_schema).unwrap();
        writer.write(&new_record).unwrap();
        writer.finish().unwrap();
    }
    OperatorStateSnapshot {
        inline_metadata: snapshot.inline_metadata.clone(),
        segments: BTreeMap::from([(segment_id.clone(), StateSegment::new(bytes))]),
    }
}

async fn snapshot_with_history_and_buffer() -> OperatorStateSnapshot {
    let job = job();
    let mut operator = error_operator();
    let mut collector = new_collector();
    operator
        .process_data(
            "input",
            input_batch(&fixture_rows()[..2]),
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
    operator
        .process_data(
            "input",
            input_batch(&fixture_rows()[2..4]),
            &context(&job, None),
            &mut collector,
        )
        .await
        .unwrap();
    operator.checkpoint(Epoch::new(1).unwrap()).unwrap()
}

#[tokio::test]
async fn segment_with_wrong_column_count_is_rejected() {
    let snapshot = snapshot_with_history_and_buffer().await;
    let corrupted = reencode_segment(&snapshot, |schema, record| {
        let fields: Vec<_> = schema.fields().iter().cloned().collect();
        let new_schema = Schema::new_with_metadata(
            fields[..fields.len() - 1].to_vec(),
            schema.metadata().clone(),
        );
        let columns: Vec<_> = record
            .columns()
            .iter()
            .take(record.num_columns() - 1)
            .cloned()
            .collect();
        let new_record = RecordBatch::try_new(Arc::new(new_schema.clone()), columns).unwrap();
        (new_schema, new_record)
    });
    assert!(error_operator().restore(&corrupted).is_err());
}

#[tokio::test]
async fn segment_with_out_of_order_rows_is_rejected() {
    let snapshot = snapshot_with_history_and_buffer().await;
    let corrupted = reencode_segment(&snapshot, |schema, record| {
        use datafusion::arrow::compute::concat_batches;
        let head = record.slice(0, 1);
        let tail = record.slice(1, record.num_rows() - 1);
        let reversed = concat_batches(&record.schema(), [&tail, &head]).unwrap();
        (schema.clone(), reversed)
    });
    assert!(error_operator().restore(&corrupted).is_err());
}

#[tokio::test]
async fn segment_with_non_contiguous_history_positions_is_rejected() {
    let snapshot = snapshot_with_history_and_buffer().await;
    let corrupted = reencode_segment(&snapshot, |schema, record| {
        let mut columns = record.columns().to_vec();
        let positions: Vec<Option<u64>> = (0..record.num_rows())
            .map(|index| if index == 0 { Some(5) } else { None })
            .collect();
        columns[1] = Arc::new(UInt64Array::from(positions));
        let new_record = RecordBatch::try_new(record.schema(), columns).unwrap();
        (schema.clone(), new_record)
    });
    assert!(error_operator().restore(&corrupted).is_err());
}

#[tokio::test]
async fn segment_with_an_unknown_row_kind_is_rejected() {
    let snapshot = snapshot_with_history_and_buffer().await;
    let corrupted = reencode_segment(&snapshot, |schema, record| {
        let mut columns = record.columns().to_vec();
        columns[0] = Arc::new(UInt8Array::from(vec![7_u8; record.num_rows()]));
        let new_record = RecordBatch::try_new(record.schema(), columns).unwrap();
        (schema.clone(), new_record)
    });
    assert!(error_operator().restore(&corrupted).is_err());
}

#[tokio::test]
async fn inventory_with_a_future_epoch_is_rejected() {
    let snapshot = snapshot_with_history_and_buffer().await;
    let mut corrupted = snapshot.clone();
    let inventory = &snapshot.inline_metadata["segment_inventory"][0];
    corrupted.inline_metadata.insert(
        "segment_inventory".into(),
        serde_json::json!([{
            "kind": "base",
            "state_layout_version": 1,
            "schema_fingerprint": inventory["schema_fingerprint"].clone(),
            "handle": {
                "operator_id": "rolling",
                "epoch": 99,
                "segment_id": inventory["handle"]["segment_id"].clone(),
                "relative_path": inventory["handle"]["relative_path"].clone(),
                "byte_len": inventory["handle"]["byte_len"].clone(),
                "sha256": inventory["handle"]["sha256"].clone(),
            }
        }]),
    );
    assert!(error_operator().restore(&corrupted).is_err());
}

#[tokio::test]
async fn segments_without_identity_metadata_are_rejected() {
    let snapshot = snapshot_with_history_and_buffer().await;
    let mut corrupted = snapshot.clone();
    corrupted
        .inline_metadata
        .insert("pipeline_fingerprint".into(), serde_json::Value::Null);
    corrupted
        .inline_metadata
        .insert("operator_id".into(), serde_json::Value::Null);
    assert!(error_operator().restore(&corrupted).is_err());
}

// ---------------------------------------------------------------------
// SCE-07: aggregate state, cancellation, and checkpoint coverage
// ---------------------------------------------------------------------

fn aggregate_operator() -> RollingOperator {
    RollingOperator::new(
        "rolling",
        input_schema(),
        serde_json::from_value(serde_json::json!({
            "configuration_version": 1,
            "state_layout_version": 1,
            "partition_by": ["symbol"],
            "event_time": "ts",
            "sequence_by": ["sequence"],
            "outputs": [
                {
                    "kind": "count",
                    "primitive_version": 1,
                    "input": "price",
                    "output": "price_count_2",
                    "frame": {"kind": "rows", "size": 2},
                    "min_periods": 1
                },
                {
                    "kind": "sum",
                    "primitive_version": 1,
                    "input": "price",
                    "output": "price_sum_2",
                    "frame": {"kind": "rows", "size": 2},
                    "min_periods": 1
                },
                {
                    "kind": "mean",
                    "primitive_version": 1,
                    "input": "price",
                    "output": "price_mean_2",
                    "frame": {"kind": "rows", "size": 2},
                    "min_periods": 1
                },
                {
                    "kind": "variance",
                    "primitive_version": 1,
                    "input": "price",
                    "output": "price_var_2",
                    "frame": {"kind": "rows", "size": 2},
                    "min_periods": 1,
                    "ddof": 1
                },
                {
                    "kind": "sum",
                    "primitive_version": 1,
                    "input": "volume",
                    "output": "volume_sum_2",
                    "frame": {"kind": "rows", "size": 2},
                    "min_periods": 1
                }
            ],
            "allowed_lateness_micros": 0,
            "late_policy": {"kind": "error", "scope": "envelope"},
            "value_policy": "stateful_numeric_v1"
        }))
        .unwrap(),
    )
    .unwrap()
}

type ObservedAggregateRow = (
    i64,
    String,
    u64,
    Option<u64>,
    Option<f64>,
    Option<f64>,
    Option<f64>,
    Option<i64>,
);

fn drain_aggregates(collector: &mut EdgeCollector, rows: &mut Vec<ObservedAggregateRow>) {
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
            let counts = record
                .column_by_name("price_count_2")
                .unwrap()
                .as_any()
                .downcast_ref::<UInt64Array>()
                .unwrap();
            let sums = record
                .column_by_name("price_sum_2")
                .unwrap()
                .as_any()
                .downcast_ref::<Float64Array>()
                .unwrap();
            let means = record
                .column_by_name("price_mean_2")
                .unwrap()
                .as_any()
                .downcast_ref::<Float64Array>()
                .unwrap();
            let variances = record
                .column_by_name("price_var_2")
                .unwrap()
                .as_any()
                .downcast_ref::<Float64Array>()
                .unwrap();
            let volume_sums = record
                .column_by_name("volume_sum_2")
                .unwrap()
                .as_any()
                .downcast_ref::<Int64Array>()
                .unwrap();
            for index in 0..record.num_rows() {
                rows.push((
                    event_times.value(index),
                    symbols.value(index).to_owned(),
                    sequences.value(index),
                    counts.iter().nth(index).unwrap(),
                    sums.iter().nth(index).unwrap(),
                    means.iter().nth(index).unwrap(),
                    variances.iter().nth(index).unwrap(),
                    volume_sums.iter().nth(index).unwrap(),
                ));
            }
        }
    }
}

async fn drive_aggregates(
    operator: &mut RollingOperator,
    chunks: &[&[InputRow]],
    job: &StreamJobContext,
    collector: &mut EdgeCollector,
    rows: &mut Vec<ObservedAggregateRow>,
) {
    for chunk in chunks {
        operator
            .process_data("input", input_batch(chunk), &context(job, None), collector)
            .await
            .unwrap();
        let closing = chunk.iter().map(|row| row.0).max().unwrap_or(0);
        operator
            .on_watermark(
                EventTime::from_micros(closing),
                &context(job, Some(closing)),
                collector,
            )
            .await
            .unwrap();
        drain_aggregates(collector, rows);
    }
}

#[tokio::test]
async fn aggregate_windows_continue_across_checkpoint_and_restore() {
    let job = job();
    let rows = fixture_rows();
    let chunks: Vec<&[_]> = rows.chunks(3).collect();

    let mut reference = aggregate_operator();
    let mut reference_collector = EdgeCollector::new(reference.output_ports().to_vec());
    let mut reference_rows = Vec::new();
    drive_aggregates(
        &mut reference,
        &chunks,
        &job,
        &mut reference_collector,
        &mut reference_rows,
    )
    .await;

    let mut restarted = aggregate_operator();
    let mut restarted_collector = EdgeCollector::new(restarted.output_ports().to_vec());
    let mut restarted_rows = Vec::new();
    drive_aggregates(
        &mut restarted,
        &chunks[..2],
        &job,
        &mut restarted_collector,
        &mut restarted_rows,
    )
    .await;
    let snapshot = restarted.checkpoint(Epoch::new(1).unwrap()).unwrap();

    let mut recovered = aggregate_operator();
    recovered.restore(&snapshot).unwrap();
    let mut recovered_collector = EdgeCollector::new(recovered.output_ports().to_vec());
    drive_aggregates(
        &mut recovered,
        &chunks[2..],
        &job,
        &mut recovered_collector,
        &mut restarted_rows,
    )
    .await;

    assert_eq!(reference_rows, restarted_rows);
    assert_eq!(reference_rows.len(), rows.len());
    let last = reference_rows.last().unwrap();
    assert_eq!(last.0, 14);
    assert_eq!(last.3, Some(2));
    assert_eq!(last.4, Some(7.0));
    assert_eq!(last.5, Some(3.5));
    assert_eq!(last.6, Some(0.5));
    assert_eq!(last.7, Some(70));
}

#[tokio::test]
async fn cancelled_emission_preserves_buffered_state_until_retry() {
    let token = CancellationToken::new();
    let job = StreamJobContext::new(1, FINGERPRINT, JsonMap::new(), None, token.clone());
    let mut operator = aggregate_operator();
    let mut collector = EdgeCollector::new(operator.output_ports().to_vec());
    operator
        .process_data(
            "input",
            input_batch(&fixture_rows()[..2]),
            &context(&job, None),
            &mut collector,
        )
        .await
        .unwrap();

    token.cancel();
    let error = operator
        .on_watermark(
            EventTime::from_micros(20),
            &context(&job, Some(20)),
            &mut collector,
        )
        .await
        .unwrap_err();
    assert!(
        matches!(error, calc_flow::CalcFlowError::Cancelled { .. }),
        "unexpected error: {error}"
    );
    let mut observed = Vec::new();
    drain_aggregates(&mut collector, &mut observed);
    assert!(observed.is_empty());

    let retry_job = StreamJobContext::new(
        1,
        FINGERPRINT,
        JsonMap::new(),
        None,
        CancellationToken::new(),
    );
    operator
        .on_watermark(
            EventTime::from_micros(20),
            &context(&retry_job, Some(20)),
            &mut collector,
        )
        .await
        .unwrap();
    drain_aggregates(&mut collector, &mut observed);
    assert_eq!(observed.len(), 2);
    assert_eq!(observed[0].3, Some(1));
    assert_eq!(observed[0].4, Some(1.0));
}
