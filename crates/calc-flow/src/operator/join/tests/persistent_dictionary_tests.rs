use super::*;
use datafusion::arrow::array::{
    Date32Array, Date64Array, Int8Array, Int16Array, Int32Array, LargeStringArray, UInt8Array,
    UInt16Array, UInt32Array,
};

#[tokio::test]
async fn test_persistent_dictionary_native_index_uses_identity_only() {
    let mut operator =
        StreamJoinOperator::new("match", left_schema(), right_schema(), spec()).unwrap();
    let job = job();
    let context = StreamOperatorContext::new(&job, "match", None);
    let mut output = EdgeCollector::new(operator.output_ports().to_vec());
    operator
        .process_data("right", right_batch(vec![0; 3]), &context, &mut output)
        .await
        .unwrap();
    operator
        .process_data("left", left_batch(vec![0]), &context, &mut output)
        .await
        .unwrap();
    output.drain("output");
    reset_join_work();
    operator
        .process_data("left", left_batch(vec![0; 2]), &context, &mut output)
        .await
        .unwrap();
    assert_eq!(operator.status().emitted_match_rows, 9);
    assert_eq!(join_work().key_encodings, 0);
    assert_eq!(
        join_work().native_content_hashes,
        0,
        "native count/fill/append must use shared identity, with no second content interner"
    );
}

#[tokio::test]
async fn test_persistent_dictionary_probe_only_unknown_allocates_no_directory() {
    let mut operator =
        StreamJoinOperator::new("match", left_schema(), right_schema(), spec()).unwrap();
    let job = job();
    let context = StreamOperatorContext::with_ingress_progress(
        &job,
        "match",
        None,
        IngressProgressSnapshot::new(BTreeMap::from([
            (
                "left".into(),
                IngressProgress::new(IngressState::Active, None),
            ),
            (
                "right".into(),
                IngressProgress::new(IngressState::Ended, None),
            ),
        ])),
    );
    let mut output = EdgeCollector::new(operator.output_ports().to_vec());
    reset_join_work();
    operator
        .process_data("left", left_batch(vec![0; 2]), &context, &mut output)
        .await
        .unwrap();
    assert_eq!(operator.state.left.len(), 0);
    assert_eq!(join_work().key_encodings, 0);
    assert_eq!(
        operator.key_dictionary.inventory(),
        (0, 0, 0, false),
        "unknown nonretained values must not create persistent capacity or namespace"
    );
}

#[tokio::test]
async fn test_persistent_dictionary_references_follow_both_side_eviction() {
    let mut operator =
        StreamJoinOperator::new("match", left_schema(), right_schema(), spec()).unwrap();
    let job = job();
    let context = StreamOperatorContext::new(&job, "match", None);
    let mut output = EdgeCollector::new(operator.output_ports().to_vec());
    operator
        .process_data("right", right_batch(vec![0; 2]), &context, &mut output)
        .await
        .unwrap();
    operator
        .process_data("left", left_batch(vec![0]), &context, &mut output)
        .await
        .unwrap();
    assert_eq!(
        (
            operator.key_dictionary.inventory().0,
            operator.key_dictionary.inventory().1
        ),
        (1, 3)
    );
    let ended = IngressProgress::new(IngressState::Ended, None);
    operator.evict_progress("left", ended, false).unwrap();
    assert_eq!(operator.state.right.len(), 0);
    assert_eq!(
        operator.key_dictionary.inventory().1,
        1,
        "right-side expiry must retire its exact references"
    );
    operator.evict_progress("right", ended, false).unwrap();
    assert_eq!(operator.key_dictionary.inventory().0, 0);
    assert_eq!(operator.key_dictionary.inventory().1, 0);
}

#[tokio::test]
async fn test_persistent_dictionary_recycles_directory_after_unique_evictions() {
    let mut operator =
        StreamJoinOperator::new("match", left_schema(), right_schema(), spec()).unwrap();
    let job = job();
    let context = StreamOperatorContext::new(&job, "match", None);
    let mut output = EdgeCollector::new(operator.output_ports().to_vec());
    for key in 0..20 {
        let record = keyed_record(true, vec![key], vec![0]);
        operator
            .process_data(
                "left",
                Batch::table(vec![record], BatchMetadata::default()).unwrap(),
                &context,
                &mut output,
            )
            .await
            .unwrap();
        operator
            .evict_progress(
                "right",
                IngressProgress::new(IngressState::Ended, None),
                false,
            )
            .unwrap();
    }
    assert_eq!(operator.key_dictionary.inventory().0, 0);
    assert!(
        operator.key_dictionary.inventory().2 <= 2,
        "directory capacity must follow concurrent live/leased identities, not all arrivals"
    );
}

#[tokio::test]
async fn test_persistent_dictionary_expired_unique_owner_retires_independently() {
    let mut operator =
        StreamJoinOperator::new("match", left_schema(), right_schema(), spec()).unwrap();
    let job = job().with_checkpointing(false);
    let context = StreamOperatorContext::new(&job, "match", None);
    let mut output = EdgeCollector::new(operator.output_ports().to_vec());
    let right = keyed_record(false, vec![7, 8], vec![0; 2]);
    operator
        .process_data(
            "right",
            Batch::table(vec![right], BatchMetadata::default()).unwrap(),
            &context,
            &mut output,
        )
        .await
        .unwrap();
    operator
        .process_data("left", left_batch(vec![0]), &context, &mut output)
        .await
        .unwrap();
    let expired = Arc::downgrade(&operator.state.right[1].encoded_key);
    let pool = operator
        .runtime
        .runtime()
        .unwrap()
        .incremental_memory_pool();
    let before = pool.reserved();
    operator
        .evict_progress(
            "left",
            IngressProgress::new(IngressState::Ended, None),
            false,
        )
        .unwrap();
    assert!(
        expired.upgrade().is_none(),
        "an unrelated surviving key must not pin an expired canonical owner or its credit"
    );
    assert!(pool.reserved() < before);
    assert_eq!(operator.key_dictionary.inventory().1, 1);
    drop((operator, output, context));
    assert!(job.gather_owner().close_and_drain().await.is_empty());
    drop(job);
    assert_eq!(pool.reserved(), 0);
}

#[tokio::test]
async fn test_persistent_dictionary_history_bytes_do_not_lease_recycled_namespace() {
    let mut operator =
        StreamJoinOperator::new("match", left_schema(), right_schema(), spec()).unwrap();
    let job = job();
    let context = StreamOperatorContext::new(&job, "match", None);
    let mut output = EdgeCollector::new(operator.output_ports().to_vec());
    operator
        .process_data("left", left_batch(vec![0]), &context, &mut output)
        .await
        .unwrap();
    let canonical = Arc::clone(&operator.state.left[0].encoded_key);
    let lease = operator.key_dictionary.lease(&canonical).unwrap();
    let old_address = lease.address();
    let old_bytes = canonical.to_vec();
    let before = operator.checkpoint(Epoch::INITIAL).unwrap();
    operator
        .evict_progress(
            "right",
            IngressProgress::new(
                IngressState::Active,
                Some(EventTime::from_micros(1_000_000_000)),
            ),
            true,
        )
        .unwrap();
    assert_eq!(operator.key_dictionary.inventory().0, 0);
    operator
        .process_data(
            "left",
            left_batch(vec![1_000_000_000]),
            &context,
            &mut output,
        )
        .await
        .unwrap();
    let current = operator
        .key_dictionary
        .lease(&operator.state.left[0].encoded_key)
        .unwrap();
    assert_ne!(
        current.address(),
        old_address,
        "a live old lease must never alias a new lifetime"
    );
    assert!(
        operator.key_dictionary.lease(&canonical).is_none(),
        "byte-only history addresses cannot resolve in the current namespace"
    );
    let pool = operator
        .runtime
        .runtime()
        .unwrap()
        .incremental_memory_pool();
    let leased = pool.reserved();
    drop(lease);
    let retired_controls = crate::operator::key_identity::NamespaceLease::control_bytes().unwrap()
        + 1024
        + key_dictionary::Lease::control_bytes().unwrap();
    assert!(
        pool.reserved() <= leased - retired_controls,
        "history bytes must not retain namespace or entry-lease control credit"
    );
    assert_eq!(canonical.as_slice(), old_bytes);
    let after = operator.checkpoint(Epoch::new(2).unwrap()).unwrap();
    let mut restored =
        StreamJoinOperator::new("match", left_schema(), right_schema(), spec()).unwrap();
    restored.restore(&after).unwrap();
    assert_eq!(restored.state.left.len(), 1);
    assert_eq!(restored.state.left[0].event_time.as_micros(), 1_000_000_000);
    assert_eq!(restored.state.left[0].encoded_key.as_slice(), old_bytes);
    drop((
        operator, restored, current, canonical, before, after, context, output,
    ));
    assert!(job.gather_owner().close_and_drain().await.is_empty());
    drop(job);
    assert_eq!(pool.reserved(), 0);
}

#[tokio::test]
async fn test_persistent_dictionary_binds_columns_once_per_record() {
    let small = Box::pin(known_column_dispatch(13)).await;
    let large = Box::pin(known_column_dispatch(26)).await;
    assert_eq!(
        small, large,
        "type binding must depend on records, not selected row count"
    );
}

async fn known_column_dispatch(rows: usize) -> usize {
    let mut operator =
        StreamJoinOperator::new("match", left_schema(), right_schema(), spec()).unwrap();
    let job = job();
    let context = StreamOperatorContext::new(&job, "match", None);
    let mut output = EdgeCollector::new(operator.output_ports().to_vec());
    operator
        .process_data("right", right_batch(vec![0]), &context, &mut output)
        .await
        .unwrap();
    let records = (0..2)
        .map(|_| keyed_record(true, vec![7; rows], vec![0; rows]))
        .collect();
    reset_join_work();
    operator
        .process_data(
            "left",
            Batch::table(records, BatchMetadata::default()).unwrap(),
            &context,
            &mut output,
        )
        .await
        .unwrap();
    assert_eq!(join_work().key_encodings, 0);
    assert_eq!(operator.status().emitted_match_rows, (2 * rows) as u64);
    join_work().key_type_resolutions
}

#[tokio::test]
async fn test_persistent_dictionary_all_v1_tags_reuse_exact_frames() {
    for key in all_tag_keys() {
        Box::pin(assert_typed_key_reuse(key)).await;
    }
}

fn all_tag_keys() -> Vec<ArrayRef> {
    let mut keys: Vec<ArrayRef> = vec![
        Arc::new(BooleanArray::from(vec![true, false, false, true])),
        Arc::new(Int8Array::from(vec![i8::MIN, i8::MAX, i8::MAX, i8::MIN])),
        Arc::new(Int16Array::from(vec![
            i16::MIN,
            i16::MAX,
            i16::MAX,
            i16::MIN,
        ])),
        Arc::new(Int32Array::from(vec![
            i32::MIN,
            i32::MAX,
            i32::MAX,
            i32::MIN,
        ])),
        Arc::new(Int64Array::from(vec![
            i64::MIN,
            i64::MAX,
            i64::MAX,
            i64::MIN,
        ])),
        Arc::new(UInt8Array::from(vec![0, u8::MAX, u8::MAX, 0])),
        Arc::new(UInt16Array::from(vec![0, u16::MAX, u16::MAX, 0])),
        Arc::new(UInt32Array::from(vec![0, u32::MAX, u32::MAX, 0])),
        Arc::new(UInt64Array::from(vec![0, u64::MAX, u64::MAX, 0])),
        Arc::new(StringArray::from(vec!["", "é\0", "é\0", ""])),
        Arc::new(LargeStringArray::from(vec!["", "é\0", "é\0", ""])),
        Arc::new(Date32Array::from(vec![-1, 1, 1, -1])),
        Arc::new(Date64Array::from(vec![
            -86_400_000,
            86_400_000,
            86_400_000,
            -86_400_000,
        ])),
    ];
    for timezone in [None, Some("UTC"), Some("Europe/Paris")] {
        macro_rules! timestamp {
            ($array:ty) => {
                keys.push(Arc::new(
                    <$array>::from(vec![-1, 1, 1, -1]).with_timezone_opt(timezone),
                ))
            };
        }
        timestamp!(TimestampSecondArray);
        timestamp!(TimestampMillisecondArray);
        timestamp!(TimestampMicrosecondArray);
        timestamp!(TimestampNanosecondArray);
    }
    keys
}

async fn assert_typed_key_reuse(key: ArrayRef) {
    let schema = |source: SchemaRef| {
        Arc::new(Schema::new(
            source
                .fields()
                .iter()
                .enumerate()
                .map(|(index, field)| {
                    if index == 0 {
                        field
                            .as_ref()
                            .clone()
                            .with_data_type(key.data_type().clone())
                    } else {
                        field.as_ref().clone()
                    }
                })
                .collect::<Vec<_>>(),
        ))
    };
    let left = schema(left_schema());
    let right = schema(right_schema());
    let record = |is_left| {
        let mut record = keyed_record(is_left, vec![7; 4], vec![0; 4]);
        let mut columns = record.columns().to_vec();
        columns[0] = Arc::clone(&key);
        record = RecordBatch::try_new(
            if is_left {
                Arc::clone(&left)
            } else {
                Arc::clone(&right)
            },
            columns,
        )
        .unwrap();
        record
    };
    let expected = (0..4)
        .map(|row| encode_join_key_columns_v1(&[Arc::clone(&key)], row, &[0]).unwrap())
        .collect::<Vec<_>>();
    let mut operator =
        StreamJoinOperator::new("match", Arc::clone(&left), Arc::clone(&right), spec()).unwrap();
    let job = job();
    let context = StreamOperatorContext::new(&job, "match", None);
    let mut output = EdgeCollector::new(operator.output_ports().to_vec());
    reset_join_work();
    operator
        .process_data(
            "right",
            Batch::table(vec![record(false)], BatchMetadata::default()).unwrap(),
            &context,
            &mut output,
        )
        .await
        .unwrap();
    operator
        .process_data(
            "left",
            Batch::table(vec![record(true)], BatchMetadata::default()).unwrap(),
            &context,
            &mut output,
        )
        .await
        .unwrap();
    assert_eq!(
        join_work().key_encodings,
        2,
        "{} must use distinct retained canonical frames",
        key.data_type()
    );
    assert_eq!(
        join_work().sql_probe_table_builds,
        0,
        "ordinary accepted types use native identity lookup"
    );
    for (row, bytes) in operator.state.left.iter().zip(expected) {
        assert_eq!(row.encoded_key.as_slice(), bytes);
    }
    assert!(Arc::ptr_eq(
        &operator.state.left[0].encoded_key,
        &operator.state.right[0].encoded_key
    ));
    assert_eq!(operator.status().emitted_match_rows, 8);
}

fn keyed_record(left: bool, keys: Vec<i64>, times: Vec<i64>) -> RecordBatch {
    let source = if left {
        left_batch(times)
    } else {
        right_batch(times)
    };
    let record = &source.table_payload().unwrap().batches()[0];
    let mut columns = record.columns().to_vec();
    columns[0] = Arc::new(Int64Array::from(keys));
    RecordBatch::try_new(record.schema(), columns).unwrap()
}

#[tokio::test]
async fn test_persistent_dictionary_reuses_cross_batch_key_frames() {
    Box::pin(assert_cross_batch_keys(true)).await;
}

#[tokio::test]
async fn test_persistent_dictionary_shares_cross_side_canonical_owner() {
    Box::pin(assert_cross_batch_keys(false)).await;
}

async fn assert_cross_batch_keys(check_frames: bool) {
    let mut operator =
        StreamJoinOperator::new("match", left_schema(), right_schema(), spec()).unwrap();
    let job = job();
    let context = StreamOperatorContext::new(&job, "match", None);
    let mut output = EdgeCollector::new(operator.output_ports().to_vec());
    let right = keyed_record(false, vec![7, 7, 8], vec![3, 1, 2]);
    let left = vec![
        keyed_record(true, vec![7, 7], vec![2, 0]),
        keyed_record(true, vec![8], vec![1]),
    ];
    let fees = [(0, 0), (0, 1), (1, 0)].map(|(record, physical)| {
        state_row_charge(&left[record], physical, &[0], "match").unwrap()
    });
    reset_join_work();
    operator
        .process_data(
            "right",
            Batch::table(vec![right.clone()], BatchMetadata::default()).unwrap(),
            &context,
            &mut output,
        )
        .await
        .unwrap();
    assert_eq!(join_work().key_encodings, 2);
    let canonical = Arc::clone(&operator.state.right[0].encoded_key);
    operator
        .process_data(
            "left",
            Batch::table(left.clone(), BatchMetadata::default()).unwrap(),
            &context,
            &mut output,
        )
        .await
        .unwrap();
    assert_cross_batch_output(&mut output);
    for (stored, fee) in operator.state.left.iter().zip(fees) {
        assert_eq!(stored.charge, fee);
    }
    assert_eq!(
        (
            operator.state.next_left_row_id,
            operator.state.next_right_row_id
        ),
        (3, 3)
    );
    assert_eq!(operator.status().emitted_match_rows, 5);
    if check_frames {
        assert_eq!(
            join_work().key_encodings,
            2,
            "known keys on the opposite side must reuse their persistent canonical frames"
        );
    } else {
        assert!(
            Arc::ptr_eq(&canonical, &operator.state.left[0].encoded_key),
            "equal retained keys on both sides must alias one persistent canonical owner"
        );
    }
}

fn assert_cross_batch_output(output: &mut EdgeCollector) {
    let messages = output.drain("output");
    assert_eq!(messages.len(), 1);
    let record = &messages[0]
        .as_data()
        .unwrap()
        .table_payload()
        .unwrap()
        .batches()[0];
    assert_eq!(record.num_rows(), 5);
    let times = |index| {
        record
            .column(index)
            .as_any()
            .downcast_ref::<TimestampMicrosecondArray>()
            .unwrap()
            .values()
            .to_vec()
    };
    assert_eq!(times(1), [2, 2, 0, 0, 1]);
    assert_eq!(times(4), [1, 3, 1, 3, 2]);
    assert_eq!(
        record.column(2).to_data(),
        Int64Array::from(vec![42; 5]).to_data()
    );
    assert_eq!(
        record.column(5).to_data(),
        StringArray::from(vec!["paid"; 5]).to_data()
    );
}

#[tokio::test]
async fn test_persistent_dictionary_known_batch_does_not_snapshot_unrelated_identities() {
    for distinct in [1, 128] {
        let mut declaration = spec();
        declaration.limits = JoinStateLimits::new(1024, 16 << 20, 4096).unwrap();
        let mut operator =
            StreamJoinOperator::new("match", left_schema(), right_schema(), declaration).unwrap();
        let job = job();
        let context = StreamOperatorContext::new(&job, "match", None);
        let mut output = EdgeCollector::new(operator.output_ports().to_vec());
        let record = keyed_record(
            true,
            (0..distinct).collect(),
            vec![0; usize::try_from(distinct).unwrap()],
        );
        operator
            .process_data(
                "left",
                Batch::table(vec![record], BatchMetadata::default()).unwrap(),
                &context,
                &mut output,
            )
            .await
            .unwrap();
        let unrelated = Arc::clone(&operator.state.left.last().unwrap().encoded_key);
        let owners = Arc::strong_count(&unrelated);
        let source = Batch::table(
            vec![keyed_record(false, vec![0], vec![0])],
            BatchMetadata::default(),
        )
        .unwrap();
        let plan = operator.begin_batch("right", &source).unwrap();
        let mut bundle = operator.admission_bundle(&plan, &source).unwrap();
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
        let keys = operator
            .native_probe_keys(&plan, &bundle.admitted)
            .unwrap()
            .unwrap();
        assert_eq!(
            join_work().dictionary_snapshot_visits,
            0,
            "ordinary known-key admission must not copy the retained directory ({distinct} identities)"
        );
        assert_eq!(
            keys.keys.len(),
            1,
            "the batch owns only its matched identity"
        );
        if distinct > 1 {
            assert_eq!(
                Arc::strong_count(&unrelated),
                owners,
                "unrelated canonical byte owners must not be cloned into an ordinary batch"
            );
        }
        drop((keys, bundle, operator, output, context));
        assert!(job.gather_owner().close_and_drain().await.is_empty());
    }
}

#[tokio::test]
async fn test_persistent_dictionary_failed_growth_keeps_committed_backing() {
    let mut declaration = spec();
    declaration.limits = JoinStateLimits::new(5, 16 << 20, 4096).unwrap();
    let mut operator =
        StreamJoinOperator::new("match", left_schema(), right_schema(), declaration).unwrap();
    let job = job();
    let context = StreamOperatorContext::new(&job, "match", None);
    let mut output = EdgeCollector::new(operator.output_ports().to_vec());
    let input = |keys: Vec<i64>| {
        Batch::table(
            vec![keyed_record(true, keys.clone(), vec![0; keys.len()])],
            BatchMetadata::default(),
        )
        .unwrap()
    };
    operator
        .process_data("left", input(vec![0, 1, 2, 3]), &context, &mut output)
        .await
        .unwrap();
    let before = operator.key_dictionary.inventory();
    let original = Arc::clone(&operator.state.left[0].encoded_key);
    let error = operator
        .process_data("left", input(vec![4, 5]), &context, &mut output)
        .await
        .unwrap_err();
    assert!(error.to_string().contains("retained state limit exceeded"));
    assert_eq!(operator.state.left.len(), 4);
    assert!(Arc::ptr_eq(&original, &operator.state.left[0].encoded_key));
    assert_eq!(
        operator.key_dictionary.inventory(),
        before,
        "failed growth must discard its staged backing and retain the previous directory"
    );
    assert!(output.drain("output").is_empty());
    drop((operator, output, context));
    assert!(job.gather_owner().close_and_drain().await.is_empty());
}

#[tokio::test]
async fn test_persistent_dictionary_reclaims_retired_slots_with_a_live_hot_key() {
    let mut operator =
        StreamJoinOperator::new("match", left_schema(), right_schema(), spec()).unwrap();
    let job = job().with_checkpointing(false);
    let context = StreamOperatorContext::new(&job, "match", None);
    let mut output = EdgeCollector::new(operator.output_ports().to_vec());
    let hot = Batch::table(
        vec![keyed_record(true, vec![7], vec![1_000_000_000])],
        BatchMetadata::default(),
    )
    .unwrap();
    operator
        .process_data("left", hot, &context, &mut output)
        .await
        .unwrap();
    for key in 8..28 {
        let cold = Batch::table(
            vec![keyed_record(true, vec![key], vec![0])],
            BatchMetadata::default(),
        )
        .unwrap();
        operator
            .process_data("left", cold, &context, &mut output)
            .await
            .unwrap();
        let held = operator
            .key_dictionary
            .lease(&operator.state.left[1].encoded_key)
            .unwrap();
        operator
            .evict_progress(
                "right",
                IngressProgress::new(
                    IngressState::Active,
                    Some(EventTime::from_micros(500_000_000)),
                ),
                false,
            )
            .unwrap();
        assert_eq!(operator.state.left.len(), 1);
        drop(held);
    }
    assert_eq!(operator.key_dictionary.inventory().1, 1);
    assert!(
        operator.key_dictionary.inventory().2 <= 4,
        "released retired leases must free slots even while a hot identity keeps the namespace alive"
    );
    drop((operator, output, context));
    assert!(job.gather_owner().close_and_drain().await.is_empty());
}
