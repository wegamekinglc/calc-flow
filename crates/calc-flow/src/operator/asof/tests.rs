use super::*;
use crate::{
    BatchMetadata, CancellationToken, EdgeBudget, EdgeCollector, Epoch, IngressProgress,
    IngressState, StreamJobContext,
};
use checkpoint::PreparedSegment;
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
    payload: Arc<state::PayloadBatch>,
    owners: Vec<usize>,
}

#[tokio::test]
async fn admission_prepares_index_only_when_checkpoint_is_captured() {
    let (mut op, input) = fixture();
    let job = StreamJobContext::new(1, "asof", JsonMap::new(), None, CancellationToken::new());
    let cx = StreamOperatorContext::new(&job, "asof", None);
    let mut output = EdgeCollector::new(op.output_ports().to_vec());
    op.process_data("right", input, &cx, &mut output)
        .await
        .unwrap();
    assert!(op.prepared.is_none(), "admission must defer index encoding");
    let charged = op.status.state_bytes;
    let snapshot = op.capture(Epoch::INITIAL).unwrap();
    assert!(snapshot.segments.contains_key("asof-index-v2"));
    assert_eq!(op.status.state_bytes, charged);
    assert!(op.prepared.is_some());
}

#[tokio::test]
async fn async_checkpoint_preparation_keeps_capture_on_shared_bytes() {
    let (mut op, input) = fixture();
    let job = StreamJobContext::new(1, "asof", JsonMap::new(), None, CancellationToken::new());
    let cx = StreamOperatorContext::new(&job, "asof", None);
    let mut output = EdgeCollector::new(op.output_ports().to_vec());
    op.process_data("right", input, &cx, &mut output)
        .await
        .unwrap();
    op.prepare_checkpoint_async(&cx).await.unwrap();
    let prepared = op.prepared.as_ref().unwrap().canonical().bytes_arc();
    let snapshot = op.checkpoint(Epoch::INITIAL).unwrap();
    assert!(Arc::ptr_eq(
        &prepared,
        &snapshot.segments["asof-index-v2"].bytes_arc()
    ));
}

#[tokio::test]
async fn cancelled_checkpoint_preparation_preserves_deferred_state_for_retry() {
    let (mut op, input) = fixture();
    let job = StreamJobContext::new(1, "asof", JsonMap::new(), None, CancellationToken::new());
    let cx = StreamOperatorContext::new(&job, "asof", None);
    let mut output = EdgeCollector::new(op.output_ports().to_vec());
    op.process_data("right", input, &cx, &mut output)
        .await
        .unwrap();
    let deferred = op.deferred_index_len;
    let charged = op.status.state_bytes;
    let cancelled = CancellationToken::new();
    cancelled.cancel();
    let cancelled_job = StreamJobContext::new(2, "asof", JsonMap::new(), None, cancelled);
    let cancelled_cx = StreamOperatorContext::new(&cancelled_job, "asof", None);
    assert!(matches!(
        op.prepare_checkpoint_async(&cancelled_cx).await,
        Err(CalcFlowError::Cancelled { .. })
    ));
    assert_eq!(op.deferred_index_len, deferred);
    assert_eq!(op.status.state_bytes, charged);
    assert!(op.prepared.is_none());
    op.prepare_checkpoint_async(&cx).await.unwrap();
    let snapshot = op.capture(Epoch::INITIAL).unwrap();
    let (mut restored, _) = fixture();
    restored.restore(&snapshot).unwrap();
    assert_eq!(restored.status.state_bytes, charged);
    assert_eq!(restored.status.retained_right_rows, 1);
}

#[tokio::test]
async fn deferred_index_length_tracks_existing_and_new_right_buckets() {
    let (mut op, first) = fixture();
    let schema = op.schemas[1].clone();
    let second = Batch::table(
        vec![
            RecordBatch::try_new(
                schema,
                vec![
                    Arc::new(StringArray::from(vec!["A", "B"])),
                    Arc::new(TimestampMicrosecondArray::from(vec![101, 99]).with_timezone("UTC")),
                    Arc::new(Int64Array::from(vec![2, 1])),
                ],
            )
            .unwrap(),
        ],
        BatchMetadata::default(),
    )
    .unwrap();
    let job = StreamJobContext::new(1, "asof", JsonMap::new(), None, CancellationToken::new());
    let cx = StreamOperatorContext::new(&job, "asof", None);
    let mut output = EdgeCollector::new(op.output_ports().to_vec());
    op.process_data("right", first, &cx, &mut output)
        .await
        .unwrap();
    op.process_data("right", second, &cx, &mut output)
        .await
        .unwrap();
    assert_eq!(
        op.deferred_index_len,
        Some(checkpoint::encoded_length(&op.state, &op.name).unwrap())
    );
    assert!(op.prepared.is_none());
    let before = op.status.state_bytes;
    let snapshot = op.capture(Epoch::INITIAL).unwrap();
    assert_eq!(op.status.state_bytes, before);
    let (mut restored, _) = fixture();
    restored.restore(&snapshot).unwrap();
    assert_eq!(restored.status.state_bytes, before);
}

#[tokio::test]
async fn finalizable_prefix_can_use_full_edge_row_budget() {
    let (template, _) = fixture();
    let schema = template.schemas[0].clone();
    let spec = StreamAsofJoinSpec::new(
        template.spec.left().clone(),
        template.spec.right().clone(),
        Duration::ZERO,
        AsofStateLimits::new(10_000, 64 << 20).unwrap(),
    )
    .unwrap();
    let mut op = StreamAsofJoinOperator::new("asof", schema.clone(), schema.clone(), spec).unwrap();
    let rows = 5_000;
    let left = Batch::table(
        vec![
            RecordBatch::try_new(
                schema,
                vec![
                    Arc::new(StringArray::from(vec!["A"; rows])),
                    Arc::new(TimestampMicrosecondArray::from(vec![100; rows]).with_timezone("UTC")),
                    Arc::new(Int64Array::from_iter_values(
                        0..i64::try_from(rows).unwrap(),
                    )),
                ],
            )
            .unwrap(),
        ],
        BatchMetadata::default(),
    )
    .unwrap();
    let job = StreamJobContext::new(1, "asof", JsonMap::new(), None, CancellationToken::new());
    let cx = StreamOperatorContext::new(&job, "asof", None)
        .with_test_output_budget(EdgeBudget::new(8_192, 64 << 20).unwrap());
    let mut output = EdgeCollector::new(op.output_ports().to_vec());
    op.process_data("left", left, &cx, &mut output)
        .await
        .unwrap();
    op.on_watermark(EventTime::from_micros(101), &cx, &mut output)
        .await
        .unwrap();
    let messages = output.drain("output");
    assert_eq!(messages.len(), 1);
    assert_eq!(
        messages[0]
            .as_data()
            .unwrap()
            .table_payload()
            .unwrap()
            .batches()[0]
            .num_rows(),
        rows
    );
    assert!(
        !crate::pipeline::CompiledStreamOperator::StreamAsofJoin(Box::new(op))
            .datafusion_runtime_initialized()
    );
}

#[tokio::test]
async fn full_admission_of_a_tiny_slice_compacts_large_backing_buffers() {
    let (mut op, _) = fixture();
    let schema = op.schemas[1].clone();
    let large = "x".repeat(4 * 1024 * 1024);
    let record = RecordBatch::try_new(
        schema,
        vec![
            Arc::new(StringArray::from(vec![large.as_str(), "A", large.as_str()])),
            Arc::new(TimestampMicrosecondArray::from(vec![99, 100, 101]).with_timezone("UTC")),
            Arc::new(Int64Array::from(vec![1, 2, 3])),
        ],
    )
    .unwrap();
    let input = Batch::table(vec![record.slice(1, 1)], BatchMetadata::default()).unwrap();
    let job = StreamJobContext::new(1, "asof", JsonMap::new(), None, CancellationToken::new());
    let cx = StreamOperatorContext::new(&job, "asof", None);
    let mut output = EdgeCollector::new(op.output_ports().to_vec());
    op.process_data("right", input, &cx, &mut output)
        .await
        .unwrap();
    let retained = op
        .state
        .right
        .values()
        .next()
        .unwrap()
        .values()
        .next()
        .unwrap();
    let bytes = retained
        .as_ref()
        .unwrap()
        .batch
        .record
        .column(0)
        .to_data()
        .get_buffer_memory_size();
    assert!(bytes < 16_384, "tiny row retained {bytes} backing bytes");
}

#[tokio::test]
async fn full_admission_does_not_retain_unbilled_small_slice_tail() {
    let (mut op, _) = fixture();
    let schema = op.schemas[1].clone();
    let record = RecordBatch::try_new(
        schema,
        vec![
            Arc::new(StringArray::from(vec!["A", &"x".repeat(3_000)])),
            Arc::new(TimestampMicrosecondArray::from(vec![99, 100]).with_timezone("UTC")),
            Arc::new(Int64Array::from(vec![1, 2])),
        ],
    )
    .unwrap();
    let input = Batch::table(vec![record.slice(0, 1)], BatchMetadata::default()).unwrap();
    let job = StreamJobContext::new(1, "asof", JsonMap::new(), None, CancellationToken::new());
    let cx = StreamOperatorContext::new(&job, "asof", None);
    let mut output = EdgeCollector::new(op.output_ports().to_vec());
    op.process_data("right", input, &cx, &mut output)
        .await
        .unwrap();
    let retained = op
        .state
        .right
        .values()
        .next()
        .unwrap()
        .values()
        .next()
        .unwrap()
        .as_ref()
        .unwrap();
    let data = retained.batch.record.column(0).to_data();
    assert!(
        data.get_buffer_memory_size() < 64,
        "admitted slice retained backing bytes outside its charged payload"
    );
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
        .batch
        .clone();
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
    let captured = op.capture(Epoch::INITIAL).unwrap();
    assert_eq!(
        captured.segments.get("asof-index-v2").cloned(),
        expected.segment.map(|segment| segment.canonical()),
        "checkpoint bytes stay canonical"
    );
}
#[tokio::test]
async fn finalized_prefixes_share_committed_checkpoint_bytes_until_capture() {
    let (mut op, left, right) = prefix_fixture();
    let job = StreamJobContext::new(1, "asof", JsonMap::new(), None, CancellationToken::new());
    let cx = StreamOperatorContext::new(&job, "asof", None)
        .with_test_output_budget(EdgeBudget::new(1, 1 << 20).unwrap());
    let mut collector = EdgeCollector::new(op.output_ports().to_vec());
    op.process_data("right", right, &cx, &mut collector)
        .await
        .unwrap();
    op.process_data("left", left, &cx, &mut collector)
        .await
        .unwrap();
    let committed = op.capture(Epoch::INITIAL).unwrap().segments["asof-index-v2"].bytes_arc();
    let owners = Arc::strong_count(&committed);
    op.on_watermark(EventTime::from_micros(103), &cx, &mut collector)
        .await
        .unwrap();
    assert_eq!(collector.drain("output").len(), 3);
    assert_eq!(
        Arc::strong_count(&committed),
        owners,
        "each finalized prefix borrows the committed bytes instead of copying them"
    );
    let expected = op
        .prepare_checkpoint(&op.state, &cx)
        .await
        .unwrap()
        .segment
        .map(|segment| segment.canonical());
    assert!(op.prepared.as_ref().unwrap().is_drained());
    let retained_index_bytes =
        op.status.state_bytes - op.state.inventory(None, &op.name).unwrap().bytes;
    assert_eq!(retained_index_bytes, committed.capacity() as u64 + 64);
    op.prepare_checkpoint_async(&cx).await.unwrap();
    assert!(!op.prepared.as_ref().unwrap().is_drained());
    assert_eq!(
        op.status.state_bytes,
        op.state
            .inventory(op.prepared.as_ref(), &op.name)
            .unwrap()
            .bytes
    );
    let captured = op.capture(Epoch::INITIAL).unwrap().segments["asof-index-v2"].clone();
    assert_eq!(Some(captured.clone()), expected, "capture stays canonical");
    assert_eq!(
        Arc::strong_count(&committed),
        1,
        "async preparation releases the drained committed bytes"
    );
    let repeated = op.capture(Epoch::INITIAL).unwrap().segments["asof-index-v2"].bytes_arc();
    assert!(
        Arc::ptr_eq(&captured.bytes_arc(), &repeated),
        "later captures share the materialized bytes"
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
    assert_eq!(
        op.prepared.as_ref().map(PreparedSegment::canonical),
        canonical.segment.map(|segment| segment.canonical())
    );
    let (mut restored, _, _) = prefix_fixture();
    restored.restore(&before).unwrap();
    assert_eq!(
        restored.status.pending_left_rows, 3,
        "older shared snapshot is unchanged"
    );
    restored.restore(&snapshot).unwrap();
    let repeated = restored.capture(Epoch::INITIAL).unwrap();
    assert!(Arc::ptr_eq(
        &snapshot.segments["asof-index-v2"].bytes_arc(),
        &repeated.segments["asof-index-v2"].bytes_arc()
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

#[tokio::test]
async fn evictable_right_state_survives_a_cancelled_output_prefix() {
    let (template, left, right) = prefix_fixture();
    let spec = StreamAsofJoinSpec::new(
        template.spec.left().clone(),
        template.spec.right().clone(),
        Duration::ZERO,
        template.spec.limits(),
    )
    .unwrap();
    let schema = template.schemas[0].clone();
    let mut op =
        StreamAsofJoinOperator::new("asof", schema.clone(), schema.clone(), spec.clone()).unwrap();
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
    let progress = IngressProgressSnapshot::new(BTreeMap::from([
        (
            "left".into(),
            IngressProgress::new(IngressState::Active, Some(EventTime::from_micros(103))),
        ),
        (
            "right".into(),
            IngressProgress::new(IngressState::Active, Some(EventTime::from_micros(103))),
        ),
    ]));
    let tick_cx =
        StreamOperatorContext::with_ingress_progress(&job, "asof", None, progress.clone())
            .with_test_output_budget(EdgeBudget::new(1, 1 << 20).unwrap());
    let mut stopped = CancelPrefixCollector {
        cancel,
        accepted: Vec::new(),
    };
    assert!(matches!(
        op.on_watermark(EventTime::from_micros(103), &tick_cx, &mut stopped)
            .await,
        Err(CalcFlowError::Cancelled { .. })
    ));
    assert_eq!(op.status.pending_left_rows, 2);
    assert_eq!(op.status.retained_right_rows, 1);
    assert!(op.prepared.is_none());
    assert!(op.deferred_index_len.is_some());
    let snapshot = op.capture(Epoch::INITIAL).unwrap();
    let mut restored = StreamAsofJoinOperator::new("asof", schema.clone(), schema, spec).unwrap();
    restored.restore(&snapshot).unwrap();
    let resumed_job =
        StreamJobContext::new(2, "asof", JsonMap::new(), None, CancellationToken::new());
    let resumed_cx =
        StreamOperatorContext::with_ingress_progress(&resumed_job, "asof", None, progress)
            .with_test_output_budget(EdgeBudget::new(1, 1 << 20).unwrap());
    let mut remaining = EdgeCollector::new(restored.output_ports().to_vec());
    restored
        .on_watermark(EventTime::from_micros(103), &resumed_cx, &mut remaining)
        .await
        .unwrap();
    assert_eq!(remaining.drain("output").len(), 2);
    assert_eq!(restored.status.retained_right_rows, 0);
    assert_eq!(restored.status.evicted_right_rows, 1);
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

#[tokio::test]
async fn finalized_prefix_keeps_index_deferred_until_capture() {
    let (mut op, left, right) = prefix_fixture();
    let job = StreamJobContext::new(1, "asof", JsonMap::new(), None, CancellationToken::new());
    let cx = StreamOperatorContext::new(&job, "asof", None);
    let mut output = EdgeCollector::new(op.output_ports().to_vec());
    op.process_data("right", right, &cx, &mut output)
        .await
        .unwrap();
    op.process_data("left", left, &cx, &mut output)
        .await
        .unwrap();
    op.on_watermark(EventTime::from_micros(103), &cx, &mut output)
        .await
        .unwrap();
    assert_eq!(op.status.emitted_left_rows, 3);
    assert!(op.prepared.is_none(), "output must not serialize the index");
    assert_eq!(
        op.deferred_index_len,
        Some(checkpoint::encoded_length(&op.state, &op.name).unwrap())
    );
    let snapshot = op.capture(Epoch::INITIAL).unwrap();
    assert!(snapshot.segments.contains_key("asof-index-v2"));
}

#[tokio::test]
async fn finalizes_large_ready_prefix_in_one_bounded_batch() {
    let (template, right) = fixture();
    let schema = template.schemas[0].clone();
    let spec = StreamAsofJoinSpec::new(
        template.spec.left().clone(),
        template.spec.right().clone(),
        Duration::ZERO,
        AsofStateLimits::new(1_000, 64 * 1024 * 1024).unwrap(),
    )
    .unwrap();
    let mut op = StreamAsofJoinOperator::new("asof", schema.clone(), schema.clone(), spec).unwrap();
    let rows = 300;
    let left = Batch::table(
        vec![
            RecordBatch::try_new(
                schema,
                vec![
                    Arc::new(StringArray::from(vec!["A"; rows])),
                    Arc::new(TimestampMicrosecondArray::from(vec![100; rows]).with_timezone("UTC")),
                    Arc::new(Int64Array::from_iter_values(
                        0..i64::try_from(rows).unwrap(),
                    )),
                ],
            )
            .unwrap(),
        ],
        BatchMetadata::default(),
    )
    .unwrap();
    let job = StreamJobContext::new(1, "asof", JsonMap::new(), None, CancellationToken::new());
    let cx = StreamOperatorContext::new(&job, "asof", None)
        .with_test_output_budget(EdgeBudget::new(512, 8 << 20).unwrap());
    let mut collector = EdgeCollector::new(op.output_ports().to_vec());
    op.process_data("right", right, &cx, &mut collector)
        .await
        .unwrap();
    op.process_data("left", left, &cx, &mut collector)
        .await
        .unwrap();
    op.on_watermark(EventTime::from_micros(101), &cx, &mut collector)
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
            .batches()
            .iter()
            .map(RecordBatch::num_rows)
            .sum::<usize>(),
        rows
    );
    assert_eq!(op.status.matched_rows, rows as u64);
    assert_eq!(op.status.pending_left_rows, 0);
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
    let operator = admit_repro_rows(10_000).await;
    assert!(operator.status.state_bytes < 16 * 1024 * 1024);
}

#[tokio::test]
async fn sixty_thousand_row_batch_admission_uses_columnar_workspace() {
    let operator = admit_repro_rows(60_000).await;
    assert!(operator.status.state_bytes < 64 * 1024 * 1024);
}

async fn admit_repro_rows(rows: u64) -> StreamAsofJoinOperator {
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
    operator
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
        &admitted.segments["asof-index-v2"].bytes_arc(),
        &swept.segments["asof-index-v2"].bytes_arc()
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
        &swept.segments["asof-index-v2"].bytes_arc(),
        &repeated.segments["asof-index-v2"].bytes_arc()
    ));

    // Watermark progress below the next evictable row also reuses it.
    let cx = StreamOperatorContext::with_ingress_progress(&job, "asof", None, progress(199));
    op.on_watermark(EventTime::from_micros(199), &cx, &mut output)
        .await
        .unwrap();
    let advanced = op.capture(Epoch::INITIAL).unwrap();
    assert!(Arc::ptr_eq(
        &swept.segments["asof-index-v2"].bytes_arc(),
        &advanced.segments["asof-index-v2"].bytes_arc()
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
