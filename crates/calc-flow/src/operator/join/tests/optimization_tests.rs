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

#[tokio::test]
async fn test_repeated_probe_keeps_only_one_canonical_owner_per_distinct_key() {
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
        work.quantum_yields, 2,
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
    let mut collector = EdgeCollector::new(operator.output_ports().to_vec());
    reset_join_work();
    let mut process = Box::pin(operator.process_data("left", source, &context, &mut collector));
    assert!(futures::poll!(process.as_mut()).is_pending());
    assert!(futures::poll!(process.as_mut()).is_pending());
    assert_eq!(join_work().generic_fast_rows, 122);
    job.cancellation().cancel();
    assert!(matches!(
        process.await,
        Err(CalcFlowError::Cancelled { .. })
    ));
    assert_eq!(operator.state.next_left_row_id, 0);
    assert!(operator.state.left.is_empty());
    assert!(collector.drain("output").is_empty());
    assert_generic_small_records_share_one_bounded_admission_quantum().await;
}

async fn assert_generic_small_records_share_one_bounded_admission_quantum() {
    let mut operator =
        StreamJoinOperator::new("match", left_schema(), right_schema(), spec()).unwrap();
    operator.set_stream_resources(
        DataFusionConfig {
            target_partitions: 2,
            ..DataFusionConfig::default()
        },
        UdfRegistrySnapshot::default(),
    );
    let records = (0..130).map(|_| generic_records()[0].slice(0, 1)).collect();
    let source = Batch::table(records, BatchMetadata::default()).unwrap();
    let job = job();
    let context = StreamOperatorContext::new(&job, "match", None);
    let mut collector = EdgeCollector::new(operator.output_ports().to_vec());
    reset_join_work();
    let mut process = Box::pin(operator.process_data("left", source, &context, &mut collector));
    assert!(futures::poll!(process.as_mut()).is_pending());
    let work = join_work();
    assert_eq!(work.generic_fast_rows, 7);
    assert_eq!(work.quantum_yields, 1);
    let consumed = work.generic_fast_rows * (512 + 16);
    assert!(consumed <= 4096);
    assert!(
        consumed + 512 > 4096,
        "next record's time-copy charge must yield"
    );
    job.cancellation().cancel();
    assert!(matches!(
        process.await,
        Err(CalcFlowError::Cancelled { .. })
    ));
    assert_eq!(operator.state.next_left_row_id, 0);
    assert!(operator.state.left.is_empty());
    assert!(collector.drain("output").is_empty());
}

#[tokio::test]
async fn test_generic_id_range_overflow_keeps_scalar_row_precedence() {
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
