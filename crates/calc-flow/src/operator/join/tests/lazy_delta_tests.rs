use super::*;

/// Admission under checkpointing performs no per-row delta bookkeeping: the
/// lazy watermark design records nothing until a capture derives the upserts
/// from live state (spec FR1/FR2).
#[tokio::test]
async fn eviction_of_never_captured_rows_records_no_tombstones() {
    let mut operator =
        StreamJoinOperator::new("match", left_schema(), right_schema(), spec()).unwrap();
    let job_context = job();
    let context = StreamOperatorContext::new(&job_context, "match", None);
    let mut collector = EdgeCollector::new(operator.output_ports().to_vec());
    operator
        .process_data(
            "left",
            left_batch((0..3).collect()),
            &context,
            &mut collector,
        )
        .await
        .unwrap();
    reset_join_work();
    let progress = progress_context(
        &job_context,
        (IngressState::Active, None),
        (IngressState::Active, Some(60_000_001)),
    );
    operator
        .on_ingress_progress("right", &progress)
        .await
        .unwrap();
    assert_eq!(operator.status().left.retained_rows, 2);
    // The evicted row was never captured, so it coalesces away instead of
    // appending a tombstone record.
    assert_eq!(join_work().tombstone_appends, 0);
    assert_eq!(operator.state.deltas.dirty[0], 2);
    assert!(operator.state.deltas.tombstones[0].is_empty());
}

/// The fresh-preparation branch of `install_v2_writer_base` must not drain
/// the dirty window: the prepared base may go stale before its capture, and
/// the first capture re-encodes every live row (spec FR6). Admitting batch B
/// after the preparation invalidates the cut, so the capture must derive a
/// delta covering both A and B.
#[tokio::test]
async fn fresh_preparation_freeze_keeps_all_rows_derivable() {
    let mut operator =
        StreamJoinOperator::new("match", left_schema(), right_schema(), spec()).unwrap();
    let job_context = job();
    let context = StreamOperatorContext::new(&job_context, "match", None);
    let mut collector = EdgeCollector::new(operator.output_ports().to_vec());
    operator
        .process_data("left", left_batch(vec![0, 1, 2]), &context, &mut collector)
        .await
        .unwrap();
    operator.prepare_checkpoint_async(&context).await.unwrap();
    // The fresh install must not advance the watermark (spec FR6): batch A
    // still has to be derivable when the prepared cut goes stale.
    assert_eq!(operator.state.deltas.watermark[0], 0);
    assert_eq!(operator.state.deltas.dirty[0], 3);
    operator
        .process_data("left", left_batch(vec![3, 4]), &context, &mut collector)
        .await
        .unwrap();
    let snapshot = operator.checkpoint(Epoch::INITIAL).unwrap();
    let delta = snapshot
        .segments
        .get("left-delta-1")
        .expect("stale cut falls back to a full dirty delta");
    assert!(
        delta.bytes().len() > 40,
        "the dirty delta carries the payload rows"
    );
    let mut restored =
        StreamJoinOperator::new("match", left_schema(), right_schema(), spec()).unwrap();
    restored.restore(&snapshot).unwrap();
    assert_eq!(restored.status().left.retained_rows, 5);
    // The capture itself drains: watermark at the frontier, nothing dirty.
    assert_eq!(operator.state.deltas.watermark[0], 5);
    assert_eq!(operator.state.deltas.dirty[0], 0);
    assert!(operator.state.deltas.tombstones[0].is_empty());
    assert_eq!(restored.state.deltas.watermark[0], 5);
    assert_eq!(restored.state.deltas.dirty[0], 0);
}

/// Evicting a row whose `row_id` lies below the capture watermark records
/// exactly one tombstone; rows at or above it coalesce away (spec FR2), and
/// the capture emits the tombstone while advancing the watermark (spec FR5).
#[tokio::test]
async fn captured_row_eviction_appends_one_tombstone_then_drains() {
    let mut operator =
        StreamJoinOperator::new("match", left_schema(), right_schema(), spec()).unwrap();
    let job_context = job();
    let context = StreamOperatorContext::new(&job_context, "match", None);
    let mut collector = EdgeCollector::new(operator.output_ports().to_vec());
    operator
        .process_data("left", left_batch(vec![0, 1, 2]), &context, &mut collector)
        .await
        .unwrap();
    let first = operator.checkpoint(Epoch::INITIAL).unwrap();
    assert_eq!(operator.state.deltas.watermark[0], 3);
    let progress = progress_context(
        &job_context,
        (IngressState::Active, None),
        (IngressState::Active, Some(60_000_001)),
    );
    reset_join_work();
    operator
        .on_ingress_progress("right", &progress)
        .await
        .unwrap();
    assert_eq!(join_work().tombstone_appends, 1);
    assert_eq!(operator.state.deltas.tombstones[0].len(), 1);
    assert_eq!(operator.state.deltas.tombstones[0].row_ids[0], 0);
    let second = operator.checkpoint(Epoch::new(2).unwrap()).unwrap();
    assert!(
        second.segments.contains_key("left-delta-2"),
        "the tombstone must reach the delta wire"
    );
    assert!(operator.state.deltas.tombstones[0].is_empty());
    assert_eq!(operator.state.deltas.watermark[0], 3);
    let mut restored =
        StreamJoinOperator::new("match", left_schema(), right_schema(), spec()).unwrap();
    restored.restore(&second).unwrap();
    assert_eq!(restored.status().left.retained_rows, 2);
    assert_eq!(restored.status().left.evicted_rows, 1);
    drop(first);
}

/// A matching fresh preparation captures from the prepared base and drains
/// the watermark at the capture (spec FR5's successful-capture drain point).
#[tokio::test]
async fn matching_fresh_preparation_captures_prepared_base_and_drains() {
    let mut operator =
        StreamJoinOperator::new("match", left_schema(), right_schema(), spec()).unwrap();
    let job_context = job();
    let context = StreamOperatorContext::new(&job_context, "match", None);
    let mut collector = EdgeCollector::new(operator.output_ports().to_vec());
    operator
        .process_data("left", left_batch(vec![0, 1]), &context, &mut collector)
        .await
        .unwrap();
    operator.prepare_checkpoint_async(&context).await.unwrap();
    let snapshot = operator.checkpoint(Epoch::INITIAL).unwrap();
    assert!(
        snapshot.segments.contains_key("left-base"),
        "matching preparation captures its prepared base"
    );
    assert_eq!(operator.state.deltas.watermark[0], 2);
    assert_eq!(operator.state.deltas.dirty[0], 0);
}

/// Terminal `on_end` under checkpointing records tombstones for live rows
/// below the watermark in bulk; rows at or above it leave no trace (spec
/// FR2), exactly mirroring the old coalescing flush.
#[tokio::test]
async fn on_end_records_below_watermark_tombstones_only() {
    let mut operator =
        StreamJoinOperator::new("match", left_schema(), right_schema(), spec()).unwrap();
    let job_context = job();
    let context = StreamOperatorContext::new(&job_context, "match", None);
    let mut collector = EdgeCollector::new(operator.output_ports().to_vec());
    operator
        .process_data("left", left_batch(vec![0, 1, 2]), &context, &mut collector)
        .await
        .unwrap();
    operator.checkpoint(Epoch::INITIAL).unwrap();
    operator
        .process_data("left", left_batch(vec![3, 4]), &context, &mut collector)
        .await
        .unwrap();
    reset_join_work();
    operator.on_end(&context, &mut collector).await.unwrap();
    // Rows 0-2 were captured and die as tombstones; rows 3-4 coalesce away.
    assert_eq!(join_work().tombstone_appends, 3);
    assert_eq!(operator.state.deltas.tombstones[0].len(), 3);
    assert_eq!(operator.state.deltas.tombstones[0].row_ids, vec![0, 1, 2]);
    assert_eq!(operator.state.deltas.dirty[0], 0);
}

/// Whole-side expiry under checkpointing keeps the `evict_all` fast path and
/// appends below-watermark tombstones in bulk (spec FR14).
#[tokio::test]
async fn whole_side_expiry_under_checkpointing_appends_tombstones_in_bulk() {
    let mut operator =
        StreamJoinOperator::new("match", left_schema(), right_schema(), spec()).unwrap();
    let job_context = job();
    let context = StreamOperatorContext::new(&job_context, "match", None);
    let mut collector = EdgeCollector::new(operator.output_ports().to_vec());
    operator
        .process_data(
            "left",
            left_batch((0..65).collect()),
            &context,
            &mut collector,
        )
        .await
        .unwrap();
    operator
        .process_data("right", right_batch(vec![0]), &context, &mut collector)
        .await
        .unwrap();
    operator.checkpoint(Epoch::INITIAL).unwrap();
    reset_join_work();
    let progress = progress_context(
        &job_context,
        (IngressState::Active, None),
        (IngressState::Active, Some(60_000_365)),
    );
    operator
        .on_ingress_progress("right", &progress)
        .await
        .unwrap();
    let work = join_work();
    assert!(operator.state.left.is_empty());
    assert_eq!(operator.state.deltas.tombstones[0].len(), 65);
    assert_eq!(work.tombstone_appends, 65);
    assert_eq!(
        work.retained_visits, 0,
        "complete expiry must bypass heap pops"
    );
    let snapshot = operator.checkpoint(Epoch::new(2).unwrap()).unwrap();
    assert!(
        snapshot.segments.contains_key("left-delta-2"),
        "the bulk tombstones reach the delta wire"
    );
    let mut restored =
        StreamJoinOperator::new("match", left_schema(), right_schema(), spec()).unwrap();
    restored.restore(&snapshot).unwrap();
    assert_eq!(restored.status().left.retained_rows, 0);
    assert_eq!(restored.status().left.evicted_rows, 65);
}

/// Row ids are monotone-with-gaps (spec FR15): a null-time row consumes an
/// id without being retained, so the run-range tombstone records must carry
/// exactly the admitted ids and never an id-range enumeration.
#[tokio::test]
async fn tombstones_round_trip_their_own_gap_carrying_id_lists() {
    let schema = Arc::new(Schema::new(vec![
        Field::new("account_id", DataType::Int64, false),
        Field::new(
            "authorized_at",
            DataType::Timestamp(TimeUnit::Microsecond, Some("UTC".into())),
            true,
        ),
        Field::new("amount", DataType::Int64, true),
    ]));
    let mut operator =
        StreamJoinOperator::new("match", Arc::clone(&schema), right_schema(), spec()).unwrap();
    let job_context = job();
    let context = StreamOperatorContext::new(&job_context, "match", None);
    let mut collector = EdgeCollector::new(operator.output_ports().to_vec());
    // ids 0 and 2 stay live with times 0 and 1; id 1 dies on its null time.
    let batch = Batch::table(
        vec![
            RecordBatch::try_new(
                Arc::clone(&schema),
                vec![
                    Arc::new(Int64Array::from(vec![7; 3])),
                    Arc::new(
                        TimestampMicrosecondArray::from(vec![Some(0_i64), None, Some(1_i64)])
                            .with_timezone("UTC"),
                    ),
                    Arc::new(Int64Array::from(vec![42; 3])),
                ],
            )
            .unwrap(),
        ],
        BatchMetadata::default(),
    )
    .unwrap();
    operator
        .process_data("left", batch, &context, &mut collector)
        .await
        .unwrap();
    assert_eq!(operator.status().left.retained_rows, 2);
    assert_eq!(operator.state.next_left_row_id, 3);
    operator.checkpoint(Epoch::INITIAL).unwrap();
    let progress = progress_context(
        &job_context,
        (IngressState::Active, None),
        (IngressState::Active, Some(60_000_002)),
    );
    operator
        .on_ingress_progress("right", &progress)
        .await
        .unwrap();
    assert_eq!(operator.state.deltas.tombstones[0].row_ids, vec![0, 2]);
    let snapshot = operator.checkpoint(Epoch::new(2).unwrap()).unwrap();
    let mut restored =
        StreamJoinOperator::new("match", Arc::clone(&schema), right_schema(), spec()).unwrap();
    restored.restore(&snapshot).unwrap();
    assert_eq!(restored.status().left.retained_rows, 0);
    assert_eq!(restored.status().left.evicted_rows, 2);
    assert_eq!(restored.state.next_left_row_id, 3);
}
