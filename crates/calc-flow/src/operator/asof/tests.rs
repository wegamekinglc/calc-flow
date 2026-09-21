use super::*;
use crate::{
    BatchMetadata, CancellationToken, EdgeBudget, EdgeCollector, Epoch, IngressProgress,
    IngressState, StreamJobContext,
};
use datafusion::arrow::{
    array::{Float64Array, Int64Array, StringArray, TimestampMicrosecondArray, UInt64Array},
    datatypes::{DataType, Field, Schema, TimeUnit},
    record_batch::RecordBatch,
};
use std::{collections::BTreeMap, sync::Arc, time::Duration};

struct LateMetrics;
impl crate::operator::stream::LateMetricSink for LateMetrics {
    fn record(&self, _delta: crate::operator::stream::LateMetricDelta) -> Result<()> {
        Ok(())
    }
}

struct RetainedPayloadCollector {
    payload: Arc<Vec<u8>>,
    owners: Vec<usize>,
}

#[async_trait]
impl StreamCollector for RetainedPayloadCollector {
    async fn emit(&mut self, _port: &str, _batch: Batch) -> Result<()> {
        self.owners.push(Arc::strong_count(&self.payload));
        Ok(())
    }
}

#[tokio::test]
async fn finalization_without_eviction_does_not_clone_retained_state() {
    let (mut op, left, right) = prefix_fixture();
    let job = StreamJobContext::new(1, "asof", JsonMap::new(), None, CancellationToken::new());
    let cx = StreamOperatorContext::new(&job, "asof", None)
        .with_test_output_budget(EdgeBudget::new(1, 1 << 20).unwrap());
    let mut preload = EdgeCollector::new(op.output_ports().to_vec());
    op.process_data("right", right, &cx, &mut preload)
        .await
        .unwrap();
    op.process_data("left", left, &cx, &mut preload)
        .await
        .unwrap();
    let payload = op
        .state
        .right
        .values()
        .next()
        .unwrap()
        .values()
        .next()
        .unwrap()
        .as_ref()
        .unwrap()
        .bytes_arc();
    let owners = Arc::strong_count(&payload);
    let mut output = RetainedPayloadCollector {
        payload,
        owners: Vec::new(),
    };
    op.on_watermark(EventTime::from_micros(103), &cx, &mut output)
        .await
        .unwrap();
    assert_eq!(
        output.owners, [owners; 3],
        "each chunk must borrow unchanged right state"
    );
    assert_eq!(op.status.matched_rows, 3);
    assert_eq!(op.status.pending_left_rows, 0);
    let expected = op.prepare_checkpoint(&op.state, &cx).await.unwrap();
    assert_eq!(
        op.prepared, expected.segment,
        "checkpoint bytes stay canonical"
    );
}
struct CancelPrefixCollector {
    cancel: CancellationToken,
    accepted: Vec<Batch>,
}

#[async_trait]
impl StreamCollector for CancelPrefixCollector {
    async fn emit(&mut self, _port: &str, batch: Batch) -> Result<()> {
        if self.accepted.is_empty() {
            self.accepted.push(batch);
            Ok(())
        } else {
            self.cancel.cancel();
            std::future::pending().await
        }
    }
}

#[tokio::test]
async fn cancelled_prefix_has_canonical_checkpoint_and_resumes_exactly() {
    let (mut op, left, right) = prefix_fixture();
    let cancel = CancellationToken::new();
    let job = StreamJobContext::new(1, "asof", JsonMap::new(), None, cancel.clone());
    let cx = StreamOperatorContext::new(&job, "asof", None)
        .with_test_output_budget(EdgeBudget::new(1, 1 << 20).unwrap());
    let mut preload = EdgeCollector::new(op.output_ports().to_vec());
    op.process_data("right", right, &cx, &mut preload)
        .await
        .unwrap();
    op.process_data("left", left, &cx, &mut preload)
        .await
        .unwrap();
    let before = op.capture(Epoch::INITIAL).unwrap();
    let mut stopped = CancelPrefixCollector {
        cancel,
        accepted: Vec::new(),
    };
    let result = op
        .on_watermark(EventTime::from_micros(103), &cx, &mut stopped)
        .await;
    assert!(matches!(result, Err(CalcFlowError::Cancelled { .. })));
    assert_eq!(op.status.pending_left_rows, 2);
    assert_eq!(op.status.matched_rows, 1);
    assert_eq!(op.next_output_sequence, 1);
    assert_eq!(op.runtime.pool.reserved(), 0);
    let snapshot = op.capture(Epoch::INITIAL).unwrap();
    let fresh_job =
        StreamJobContext::new(2, "asof", JsonMap::new(), None, CancellationToken::new());
    let resumed_cx = StreamOperatorContext::new(&fresh_job, "asof", None)
        .with_test_output_budget(EdgeBudget::new(1, 1 << 20).unwrap());
    let canonical = op.prepare_checkpoint(&op.state, &resumed_cx).await.unwrap();
    assert_eq!(op.prepared, canonical.segment);
    let (mut restored, _, _) = prefix_fixture();
    restored.restore(&before).unwrap();
    assert_eq!(
        restored.status.pending_left_rows, 3,
        "older shared snapshot is unchanged"
    );
    restored.restore(&snapshot).unwrap();
    let repeated = restored.capture(Epoch::INITIAL).unwrap();
    assert!(Arc::ptr_eq(
        &snapshot.segments["asof-state-v1"].bytes_arc(),
        &repeated.segments["asof-state-v1"].bytes_arc()
    ));
    restored.restore(&repeated).unwrap();
    let mut remaining = EdgeCollector::new(restored.output_ports().to_vec());
    restored
        .on_watermark(EventTime::from_micros(103), &resumed_cx, &mut remaining)
        .await
        .unwrap();
    let mut batches = stopped.accepted;
    batches.extend(
        remaining
            .drain("output")
            .into_iter()
            .map(|message| message.as_data().unwrap().clone()),
    );
    for (index, batch) in batches.iter().enumerate() {
        assert_eq!(batch.metadata().sequence(), index as u64);
        let record = &batch.table_payload().unwrap().batches()[0];
        let left = record
            .column(2)
            .as_any()
            .downcast_ref::<Int64Array>()
            .unwrap();
        let right = record
            .column(5)
            .as_any()
            .downcast_ref::<Int64Array>()
            .unwrap();
        assert_eq!(left.value(0), i64::try_from(index).unwrap() + 1);
        assert_eq!(right.value(0), 1);
    }
    assert_eq!(batches.len(), 3);
    assert_eq!(restored.status.emitted_left_rows, 3);
    assert_eq!(restored.status.pending_left_rows, 0);
}

fn prefix_fixture() -> (StreamAsofJoinOperator, Batch, Batch) {
    let (template, right) = fixture();
    let schema = template.schemas[0].clone();
    let spec = StreamAsofJoinSpec::new(
        template.spec.left().clone(),
        template.spec.right().clone(),
        Duration::from_micros(10),
        template.spec.limits(),
    )
    .unwrap();
    let op = StreamAsofJoinOperator::new("asof", schema.clone(), schema.clone(), spec).unwrap();
    let left = Batch::table(
        vec![
            RecordBatch::try_new(
                schema,
                vec![
                    Arc::new(StringArray::from(vec!["A"; 3])),
                    Arc::new(
                        TimestampMicrosecondArray::from(vec![100, 101, 102]).with_timezone("UTC"),
                    ),
                    Arc::new(Int64Array::from(vec![1, 2, 3])),
                ],
            )
            .unwrap(),
        ],
        BatchMetadata::default(),
    )
    .unwrap();
    (op, left, right)
}

fn fixture() -> (StreamAsofJoinOperator, Batch) {
    let schema = Arc::new(Schema::new(vec![
        Field::new("key", DataType::Utf8, false),
        Field::new(
            "time",
            DataType::Timestamp(TimeUnit::Microsecond, Some("UTC".into())),
            false,
        ),
        Field::new("seq", DataType::Int64, false),
    ]));
    let side = |prefix: &str| {
        AsofJoinSide::new(
            vec!["key".into()],
            "time".into(),
            vec!["seq".into()],
            prefix.into(),
        )
        .unwrap()
    };
    let spec = StreamAsofJoinSpec::new(
        side("left"),
        side("right"),
        Duration::ZERO,
        AsofStateLimits::new(100, 1_048_576).unwrap(),
    )
    .unwrap();
    let operator =
        StreamAsofJoinOperator::new("asof", schema.clone(), schema.clone(), spec).unwrap();
    let batch = Batch::table(
        vec![
            RecordBatch::try_new(
                schema,
                vec![
                    Arc::new(StringArray::from(vec!["A"])),
                    Arc::new(TimestampMicrosecondArray::from(vec![100]).with_timezone("UTC")),
                    Arc::new(Int64Array::from(vec![1])),
                ],
            )
            .unwrap(),
        ],
        BatchMetadata::default(),
    )
    .unwrap();
    (operator, batch)
}

#[tokio::test]
async fn one_row_output_byte_limit_preserves_pending_state_and_releases_workspace() {
    let (mut op, batch) = fixture();
    let job = StreamJobContext::new(1, "asof", JsonMap::new(), None, CancellationToken::new());
    let cx = StreamOperatorContext::for_task(
        &job,
        "asof",
        None,
        IngressProgressSnapshot::default(),
        EdgeBudget {
            max_bytes: 1,
            ..EdgeBudget::default()
        },
        Arc::new(LateMetrics),
    );
    let mut output = EdgeCollector::new(op.output_ports().to_vec());
    op.process_data("left", batch, &cx, &mut output)
        .await
        .unwrap();
    let before = op.capture(Epoch::INITIAL).unwrap();
    assert_eq!(op.runtime.pool.reserved(), 0);
    let error = op
        .on_watermark(EventTime::from_micros(101), &cx, &mut output)
        .await
        .unwrap_err();
    assert!(matches!(
        error,
        CalcFlowError::OperatorReason {
            reason_code: StreamingFailureReason::AsofOutputLimitExceeded,
            ..
        }
    ));
    assert_eq!(op.status.pending_left_rows, 1);
    assert_eq!(op.status.output_limit_failures, 1);
    assert_eq!(
        op.capture(Epoch::INITIAL).unwrap().segments,
        before.segments
    );
    assert!(output.drain("output").is_empty());
    assert_eq!(op.runtime.pool.reserved(), 0);
    op.reset().unwrap();
    assert_eq!(op.runtime.pool.reserved(), 0);
}

#[tokio::test]
async fn ten_thousand_row_admission_fits_the_recommended_workspace_limits() {
    // DAL-287 repro shape: four-column facts admitted under the limits
    // docs/asof-join-guide.md recommends. Admission must charge close to the
    // actual encoded bytes, not a per-row schema tax.
    let schema = Arc::new(Schema::new(vec![
        Field::new(
            "event_time",
            DataType::Timestamp(TimeUnit::Microsecond, Some("UTC".into())),
            false,
        ),
        Field::new("sequence", DataType::UInt64, false),
        Field::new("symbol", DataType::Utf8, false),
        Field::new("price", DataType::Float64, false),
    ]));
    let side = |prefix: &str| {
        AsofJoinSide::new(
            vec!["symbol".into()],
            "event_time".into(),
            vec!["sequence".into()],
            prefix.into(),
        )
        .unwrap()
    };
    let spec = StreamAsofJoinSpec::new(
        side("left"),
        side("right"),
        Duration::from_secs(10_000_000),
        AsofStateLimits::new(100_000, 64 * 1024 * 1024).unwrap(),
    )
    .unwrap();
    let mut operator =
        StreamAsofJoinOperator::new("asof", schema.clone(), schema.clone(), spec).unwrap();
    let rows = 10_000_u64;
    let indexes = 0..i32::try_from(rows).unwrap();
    let record = RecordBatch::try_new(
        schema,
        vec![
            Arc::new(
                TimestampMicrosecondArray::from_iter_values(
                    indexes.clone().map(|row| 1_000_000 + i64::from(row)),
                )
                .with_timezone("UTC"),
            ),
            Arc::new(UInt64Array::from_iter_values(0..rows)),
            Arc::new(StringArray::from_iter_values(
                indexes.clone().map(|row| format!("S{:03}", row % 64)),
            )),
            Arc::new(Float64Array::from_iter_values(
                indexes.map(|row| 100.0 + f64::from(row % 257)),
            )),
        ],
    )
    .unwrap();
    let batch = Batch::table(vec![record], BatchMetadata::default()).unwrap();
    let job = StreamJobContext::new(1, "asof", JsonMap::new(), None, CancellationToken::new());
    let cx = StreamOperatorContext::new(&job, "asof", None);
    let mut output = EdgeCollector::new(operator.output_ports().to_vec());

    operator
        .process_data("left", batch, &cx, &mut output)
        .await
        .unwrap();

    assert_eq!(operator.status.left.accepted_rows, rows);
    assert_eq!(operator.status.pending_left_rows, rows);
    assert_eq!(operator.runtime.pool.reserved(), 0);
}

#[tokio::test]
async fn watermark_tick_without_eviction_reuses_the_prepared_segment() {
    // A watermark tick that neither admits, removes nor evicts anything must
    // not re-encode state: the previously installed segment stays installed.
    let (mut op, _) = fixture();
    let schema = Arc::new(Schema::new(vec![
        Field::new("key", DataType::Utf8, false),
        Field::new(
            "time",
            DataType::Timestamp(TimeUnit::Microsecond, Some("UTC".into())),
            false,
        ),
        Field::new("seq", DataType::Int64, false),
    ]));
    let right = Batch::table(
        vec![
            RecordBatch::try_new(
                schema,
                vec![
                    Arc::new(StringArray::from(vec!["A", "A"])),
                    Arc::new(TimestampMicrosecondArray::from(vec![100, 200]).with_timezone("UTC")),
                    Arc::new(Int64Array::from(vec![1, 2])),
                ],
            )
            .unwrap(),
        ],
        BatchMetadata::default(),
    )
    .unwrap();
    let job = StreamJobContext::new(1, "asof", JsonMap::new(), None, CancellationToken::new());
    let mut output = EdgeCollector::new(op.output_ports().to_vec());
    let progress = |watermark: i64| {
        IngressProgressSnapshot::new(BTreeMap::from([
            (
                "left".into(),
                IngressProgress::new(
                    IngressState::Active,
                    Some(EventTime::from_micros(watermark)),
                ),
            ),
            (
                "right".into(),
                IngressProgress::new(
                    IngressState::Active,
                    Some(EventTime::from_micros(watermark)),
                ),
            ),
        ]))
    };
    let idle = StreamOperatorContext::new(&job, "asof", None);
    op.process_data("right", right, &idle, &mut output)
        .await
        .unwrap();
    let admitted = op.capture(Epoch::INITIAL).unwrap();

    // The first sweep at 150 evicts the row at 100: state changes and is
    // re-encoded into a fresh segment allocation.
    let cx = StreamOperatorContext::with_ingress_progress(&job, "asof", None, progress(150));
    op.on_watermark(EventTime::from_micros(150), &cx, &mut output)
        .await
        .unwrap();
    let swept = op.capture(Epoch::INITIAL).unwrap();
    assert_eq!(op.status().evicted_right_rows, 1);
    assert!(!Arc::ptr_eq(
        &admitted.segments["asof-state-v1"].bytes_arc(),
        &swept.segments["asof-state-v1"].bytes_arc()
    ));

    // An identical tick changes nothing: status is untouched and the prepared
    // segment allocation is reused instead of re-encoded.
    let status = op.status();
    op.on_watermark(EventTime::from_micros(150), &cx, &mut output)
        .await
        .unwrap();
    let repeated = op.capture(Epoch::INITIAL).unwrap();
    assert_eq!(op.status(), status);
    assert!(Arc::ptr_eq(
        &swept.segments["asof-state-v1"].bytes_arc(),
        &repeated.segments["asof-state-v1"].bytes_arc()
    ));

    // Watermark progress below the next evictable row also reuses it.
    let cx = StreamOperatorContext::with_ingress_progress(&job, "asof", None, progress(199));
    op.on_watermark(EventTime::from_micros(199), &cx, &mut output)
        .await
        .unwrap();
    let advanced = op.capture(Epoch::INITIAL).unwrap();
    assert!(Arc::ptr_eq(
        &swept.segments["asof-state-v1"].bytes_arc(),
        &advanced.segments["asof-state-v1"].bytes_arc()
    ));
    assert_eq!(
        op.status().output_watermark_micros,
        Some(EventTime::from_micros(198))
    );
}

#[tokio::test]
async fn restored_logical_counter_overflow_fails_before_admission_or_emit() {
    let (mut op, batch) = fixture();
    let mut snapshot = op.capture(Epoch::INITIAL).unwrap();
    let metrics = snapshot.inline_metadata.get_mut("metrics").unwrap();
    metrics["right"]["accepted_rows"] = u64::MAX.into();
    metrics["evicted_right_rows"] = u64::MAX.into();
    op.restore(&snapshot).unwrap();
    let before = op.status();
    let job = StreamJobContext::new(1, "asof", JsonMap::new(), None, CancellationToken::new());
    let cx = StreamOperatorContext::new(&job, "asof", None);
    let mut output = EdgeCollector::new(op.output_ports().to_vec());
    let error = op
        .process_data("right", batch, &cx, &mut output)
        .await
        .unwrap_err();
    assert!(matches!(
        error,
        CalcFlowError::OperatorReason {
            reason_code: StreamingFailureReason::AsofCounterOverflow,
            ..
        }
    ));
    assert_eq!(op.status(), before);
    assert!(output.drain("output").is_empty());
    assert_eq!(op.runtime.pool.reserved(), 0);
}
