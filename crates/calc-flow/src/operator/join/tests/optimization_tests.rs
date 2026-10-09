use super::*;

fn with_keys(batch: &Batch, keys: Vec<i64>) -> Batch {
    let record = &batch.table_payload().unwrap().batches()[0];
    let mut columns = record.columns().to_vec();
    columns[0] = Arc::new(Int64Array::from(keys));
    let record = RecordBatch::try_new(record.schema(), columns).unwrap();
    Batch::table(vec![record], BatchMetadata::default()).unwrap()
}

#[tokio::test]
async fn test_native_probe_resolves_each_distinct_key_once_before_both_window_passes() {
    tokio::spawn(assert_native_probe_resolves_each_distinct_key_once_before_both_window_passes())
        .await
        .unwrap();
    tokio::spawn(assert_repeated_probe_keeps_only_one_canonical_owner_per_distinct_key())
        .await
        .unwrap();
}

async fn assert_native_probe_resolves_each_distinct_key_once_before_both_window_passes() {
    let mut operator =
        StreamJoinOperator::new("match", left_schema(), right_schema(), spec()).unwrap();
    let job = job();
    let context = StreamOperatorContext::new(&job, "match", None);
    let left = with_keys(&left_batch(vec![1, 0]), vec![7, 8]);
    let prepared = operator
        .prepare_batch("left", &left, &context)
        .await
        .unwrap();
    operator.commit_prepared("left", prepared).unwrap();
    reset_join_work();
    let right = with_keys(&right_batch(vec![0; 6]), vec![7, 9, 7, 8, 9, 8]);
    let prepared = operator
        .prepare_batch("right", &right, &context)
        .await
        .unwrap();
    assert_eq!(
        prepared
            .output
            .iter()
            .map(|pair| (pair.pos, pair.opposite_index))
            .collect::<Vec<_>>(),
        [(0, 0), (2, 0), (3, 1), (5, 1)]
    );
    assert_eq!(join_work().key_encodings, 3);
    assert_eq!(join_work().sql_probe_table_builds, 0);
    assert_eq!(join_work().native_key_lookups, 3);
}

async fn assert_repeated_probe_keeps_only_one_canonical_owner_per_distinct_key() {
    let mut operator =
        StreamJoinOperator::new("match", left_schema(), right_schema(), spec()).unwrap();
    let job = job();
    let context = StreamOperatorContext::new(&job, "match", None);
    let source = with_keys(&right_batch(vec![0; 6]), vec![7, 9, 7, 8, 9, 8]);
    let plan = operator.begin_batch("right", &source).unwrap();
    let mut bundle = operator.admission_bundle(&plan);
    operator
        .admit_record(
            &source.table_payload().unwrap().batches()[0],
            &plan,
            "right",
            &context,
            &mut bundle,
            0,
        )
        .await
        .unwrap();
    let keys = operator
        .native_probe_keys(&plan, &bundle.admitted)
        .unwrap()
        .unwrap();
    assert_eq!(
        Arc::strong_count(&keys.keys[0]),
        1,
        "repeated probe positions must borrow one canonical owner without Arc cloning"
    );
    assert_eq!(keys.keys.len(), 3);
    assert_probe_encodes_contiguous_frames_without_per_row_type_resolution().await;
}

async fn assert_probe_encodes_contiguous_frames_without_per_row_type_resolution() {
    let mut operator =
        StreamJoinOperator::new("match", left_schema(), right_schema(), spec()).unwrap();
    operator.set_stream_resources(
        DataFusionConfig {
            target_partitions: 2,
            ..DataFusionConfig::default()
        },
        UdfRegistrySnapshot::default(),
    );
    let job = job();
    let context = StreamOperatorContext::new(&job, "match", None);
    let source = with_keys(&right_batch(vec![0; 129]), vec![7; 129]);
    let plan = operator.begin_batch("right", &source).unwrap();
    let mut bundle = operator.admission_bundle(&plan);
    operator
        .admit_record(
            &source.table_payload().unwrap().batches()[0],
            &plan,
            "right",
            &context,
            &mut bundle,
            0,
        )
        .await
        .unwrap();
    reset_join_work();
    let mut keys = None;
    let allocations = allocation_counter::measure(|| {
        keys = operator.native_probe_keys(&plan, &bundle.admitted).unwrap();
    });
    let keys = keys.unwrap();
    assert!(
        allocations.bytes_max - u64::try_from(allocations.bytes_current).unwrap() <= 1024,
        "repeated keys reuse one row buffer without retaining N frames and offsets: {allocations:?}"
    );
    assert_eq!(join_work().borrowed_key_hashes, 0);
    assert_eq!(join_work().borrowed_key_equalities, 0);
    assert_eq!(join_work().arena_frames, 129);
    assert_eq!(join_work().key_encodings, 1);
    assert!(join_work().key_type_resolutions <= 2);
    assert_eq!(keys.keys.len(), 1);
    assert_eq!(Arc::strong_count(&keys.keys[0]), 1);
}

fn generic_records() -> Vec<RecordBatch> {
    [65, 65]
        .into_iter()
        .map(|rows| {
            RecordBatch::try_new(
                left_schema(),
                vec![
                    Arc::new(Int64Array::from(vec![7; rows])),
                    Arc::new(TimestampMicrosecondArray::from(vec![0; rows]).with_timezone("UTC")),
                    Arc::new(Int64Array::from(vec![None::<i64>; rows])),
                ],
            )
            .unwrap()
        })
        .collect()
}

#[tokio::test]
async fn test_all_admitted_generic_blocks_charge_only_remaining_metadata_work() {
    tokio::spawn(assert_all_admitted_generic_blocks_charge_only_remaining_metadata_work())
        .await
        .unwrap();
    tokio::spawn(assert_generic_id_range_overflow_keeps_scalar_row_precedence())
        .await
        .unwrap();
}

async fn assert_all_admitted_generic_blocks_charge_only_remaining_metadata_work() {
    let mut operator =
        StreamJoinOperator::new("match", left_schema(), right_schema(), spec()).unwrap();
    operator.set_stream_resources(
        DataFusionConfig {
            target_partitions: 2,
            ..DataFusionConfig::default()
        },
        UdfRegistrySnapshot::default(),
    );
    let job = job();
    let context = StreamOperatorContext::new(&job, "match", None);
    let source = Batch::table(generic_records(), BatchMetadata::default()).unwrap();
    let plan = operator.begin_batch("left", &source).unwrap();
    let mut bundle = operator.admission_bundle(&plan);
    reset_join_work();
    let mut source_base = 0;
    for record in source.table_payload().unwrap().batches() {
        operator
            .admit_record(record, &plan, "left", &context, &mut bundle, source_base)
            .await
            .unwrap();
        source_base += record.num_rows();
    }
    assert_eq!(bundle.next_row_id, 130);
    assert_eq!(bundle.admitted_source_rows, (0..130).collect::<Vec<_>>());
    for (position, row) in bundle.admitted.iter().enumerate() {
        assert_eq!(usize::try_from(row.row_id).unwrap(), position);
        assert_eq!(row.record.offset(), position % 65);
        assert_eq!(row.event_time, EventTime::from_micros(0));
        assert!(row.retain);
    }
    let work = join_work();
    assert_eq!(
        work.quantum_boundaries, 2,
        "bulk time copy, constant temporal mask and one metadata append visit"
    );
    assert_eq!(
        work.scalar_admissions, 0,
        "proven blocks bypass scalar reserve/classify"
    );
    assert_eq!(work.generic_fast_rows, 130);
    assert_eq!(work.admission_grants, 6);
    assert!(
        work.quantum_steps < 32,
        "grant avoids one Quantum future per row"
    );
}

#[tokio::test]
async fn test_generic_grant_manual_poll_cancellation_never_commits() {
    assert_quantum_does_not_force_peer_before_budget_exhaustion().await;
    assert_generic_peer_cancellation_never_commits().await;
    assert_generic_small_records_share_one_bounded_admission_quantum().await;
    assert_quantum_checks_cancel_before_and_after_cooperation().await;
    assert_quantum_checks_deadline_before_and_after_cooperation().await;
}

async fn assert_generic_peer_cancellation_never_commits() {
    tokio::spawn(generic_peer_cancellation_never_commits())
        .await
        .unwrap();
}

async fn generic_peer_cancellation_never_commits() {
    let mut operator =
        StreamJoinOperator::new("match", left_schema(), right_schema(), spec()).unwrap();
    operator.set_stream_resources(
        DataFusionConfig {
            target_partitions: 2,
            ..DataFusionConfig::default()
        },
        UdfRegistrySnapshot::default(),
    );
    let job = job();
    let context = StreamOperatorContext::new(&job, "match", None);
    let records = vec![generic_records()[0].clone(); 512];
    let source = Batch::table(records, BatchMetadata::default()).unwrap();
    let mut collector = EdgeCollector::new(operator.output_ports().to_vec());
    reset_join_work();
    let cancel = job.cancellation().clone();
    let peer = tokio::spawn(async move { cancel.cancel() });
    assert!(matches!(
        operator
            .process_data("left", source, &context, &mut collector)
            .await,
        Err(CalcFlowError::Cancelled { .. })
    ));
    peer.await.unwrap();
    let work = join_work();
    assert!(work.generic_fast_rows > 0);
    assert!(work.generic_fast_rows <= 128 * 64);
    assert!(work.quantum_boundaries > 0);
    assert_eq!(operator.state.next_left_row_id, 0);
    assert!(operator.state.left.is_empty());
    assert!(collector.drain("output").is_empty());
}

async fn assert_quantum_does_not_force_peer_before_budget_exhaustion() {
    tokio::spawn(async {
        use std::sync::atomic::{AtomicBool, Ordering};

        let job = job();
        let context = StreamOperatorContext::new(&job, "match", None);
        let ran = Arc::new(AtomicBool::new(false));
        let seen = Arc::clone(&ran);
        let peer = tokio::spawn(async move { seen.store(true, Ordering::Release) });
        let mut quantum = columnar::Quantum::default();
        quantum.step(&context, 64, 0).await.unwrap();
        quantum.step(&context, 1, 0).await.unwrap();
        assert!(
            !ran.load(Ordering::Acquire),
            "one work boundary must not force a ready peer before Tokio's cooperative budget is exhausted"
        );
        peer.await.unwrap();
    })
    .await
    .unwrap();
}

async fn assert_generic_small_records_share_one_bounded_admission_quantum() {
    tokio::spawn(generic_small_records_share_one_bounded_admission_quantum())
        .await
        .unwrap();
}

async fn generic_small_records_share_one_bounded_admission_quantum() {
    use std::sync::atomic::{AtomicBool, Ordering};

    let mut operator =
        StreamJoinOperator::new("match", left_schema(), right_schema(), spec()).unwrap();
    operator.set_stream_resources(
        DataFusionConfig {
            target_partitions: 2,
            ..DataFusionConfig::default()
        },
        UdfRegistrySnapshot::default(),
    );
    let record = generic_records()[0].slice(0, 1);
    let job = job();
    let context = StreamOperatorContext::new(&job, "match", None);
    let source = Batch::table(vec![record.clone(); 130], BatchMetadata::default()).unwrap();
    let plan = operator.begin_batch("left", &source).unwrap();
    let mut bundle = operator.admission_bundle(&plan);
    let ran = Arc::new(AtomicBool::new(false));
    let seen = Arc::clone(&ran);
    let peer = tokio::spawn(async move { seen.store(true, Ordering::Release) });
    reset_join_work();
    for source_base in 0..130 {
        operator
            .admit_record(&record, &plan, "left", &context, &mut bundle, source_base)
            .await
            .unwrap();
    }
    let work = join_work();
    assert_eq!(work.generic_fast_rows, 130);
    assert_eq!(work.quantum_boundaries, (130 - 1) / 7);
    let rows_per_quantum = work.generic_fast_rows.div_ceil(work.quantum_boundaries + 1);
    let consumed = rows_per_quantum * (512 + 16);
    assert!(consumed <= 4096);
    assert!(consumed + 512 > 4096);
    assert_eq!(bundle.next_row_id, 130);
    assert_eq!(bundle.admitted_source_rows, (0..130).collect::<Vec<_>>());
    assert!(!ran.load(Ordering::Acquire));
    peer.await.unwrap();
    assert_eq!(operator.state.next_left_row_id, 0);
    assert!(operator.state.left.is_empty());
}

async fn exhaust_cooperative_budget() {
    while tokio::task::coop::has_budget_remaining() {
        tokio::task::consume_budget().await;
    }
}

async fn assert_quantum_checks_cancel_before_and_after_cooperation() {
    tokio::spawn(async {
        for cancelled_before in [true, false] {
            let job = job();
            let context = StreamOperatorContext::new(&job, "match", None);
            let mut quantum = columnar::Quantum::default();
            quantum.step(&context, 64, 4096).await.unwrap();
            exhaust_cooperative_budget().await;
            let mut boundary = Box::pin(quantum.step(&context, 1, 1));
            if cancelled_before {
                job.cancellation().cancel();
                assert!(matches!(
                    futures::poll!(boundary.as_mut()),
                    std::task::Poll::Ready(Err(CalcFlowError::Cancelled { .. }))
                ));
            } else {
                assert!(futures::poll!(boundary.as_mut()).is_pending());
                job.cancellation().cancel();
                assert!(matches!(
                    boundary.await,
                    Err(CalcFlowError::Cancelled { .. })
                ));
            }
        }
    })
    .await
    .unwrap();
}

async fn assert_quantum_checks_deadline_before_and_after_cooperation() {
    tokio::spawn(async {
        let deadline = chrono::Utc::now() - chrono::Duration::seconds(1);
        let job = StreamJobContext::new(
            1,
            "expired",
            JsonMap::new(),
            Some(deadline),
            CancellationToken::new(),
        );
        let context = StreamOperatorContext::new(&job, "match", None);
        let mut quantum = columnar::Quantum::default();
        quantum.step(&context, 64, 4096).await.unwrap();
        exhaust_cooperative_budget().await;
        assert!(matches!(
            futures::poll!(Box::pin(quantum.step(&context, 1, 1))),
            std::task::Poll::Ready(Err(CalcFlowError::Cancelled { .. }))
        ));

        let deadline = chrono::Utc::now() + chrono::Duration::seconds(1);
        let job = StreamJobContext::new(
            1,
            "expires-while-cooperating",
            JsonMap::new(),
            Some(deadline),
            CancellationToken::new(),
        );
        let context = StreamOperatorContext::new(&job, "match", None);
        let mut quantum = columnar::Quantum::default();
        quantum.step(&context, 64, 4096).await.unwrap();
        let mut boundary = Box::pin(quantum.step(&context, 1, 1));
        assert!(futures::poll!(boundary.as_mut()).is_pending());
        tokio::time::sleep(Duration::from_millis(1010)).await;
        assert!(matches!(
            boundary.await,
            Err(CalcFlowError::Cancelled { .. })
        ));
    })
    .await
    .unwrap();
}

async fn assert_generic_id_range_overflow_keeps_scalar_row_precedence() {
    let mut operator =
        StreamJoinOperator::new("match", left_schema(), right_schema(), spec()).unwrap();
    operator.set_stream_resources(
        DataFusionConfig {
            target_partitions: 2,
            ..DataFusionConfig::default()
        },
        UdfRegistrySnapshot::default(),
    );
    operator.state.next_left_row_id = u64::MAX - 1;
    let job = job();
    let context = StreamOperatorContext::new(&job, "match", None);
    let source = left_batch(vec![0, 0]);
    let plan = operator.begin_batch("left", &source).unwrap();
    let mut bundle = operator.admission_bundle(&plan);
    reset_join_work();
    let result = operator
        .admit_record(
            &source.table_payload().unwrap().batches()[0],
            &plan,
            "left",
            &context,
            &mut bundle,
            0,
        )
        .await;
    assert!(matches!(
        result,
        Err(CalcFlowError::OperatorReason {
            reason_code: crate::StreamingFailureReason::JoinCounterOverflow,
            ..
        })
    ));
    assert_eq!(bundle.next_row_id, u64::MAX);
    assert_eq!(bundle.admitted.len(), 1);
    assert_eq!(bundle.admitted[0].row_id, u64::MAX - 1);
    assert_eq!(join_work().scalar_admissions, 2);
    assert_eq!(join_work().generic_fast_rows, 0);
    assert_eq!(operator.state.next_left_row_id, u64::MAX - 1);
}
