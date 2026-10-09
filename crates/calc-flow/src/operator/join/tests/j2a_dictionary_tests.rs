use super::*;
use datafusion::execution::memory_pool::{GreedyMemoryPool, MemoryConsumer, MemoryPool};

#[tokio::test]
async fn test_first_retained_batch_interns_keys_and_shares_dirty_owners() {
    let mut operator =
        StreamJoinOperator::new("match", left_schema(), right_schema(), spec()).unwrap();
    let job = job();
    let context = StreamOperatorContext::new(&job, "match", None);
    reset_join_work();
    let prepared = operator
        .prepare_batch("left", &left_batch(vec![3, 0, 2, 1]), &context)
        .await
        .unwrap();
    assert_eq!(join_work().key_encodings, 1);
    assert_eq!(join_work().sql_probe_table_builds, 0);
    assert!(prepared.output.is_empty());
    for row in &prepared.retained {
        assert!(Arc::ptr_eq(
            &row.encoded_key,
            &prepared.retained[0].encoded_key
        ));
    }
    operator.commit_prepared("left", prepared).unwrap();
    let retained = &operator.state.left[0].encoded_key;
    for op in operator.state.deltas.pending.iter() {
        let PendingOp::Upsert { encoded_key, .. } = op else {
            panic!("first batch only inserts rows");
        };
        assert!(Arc::ptr_eq(retained, encoded_key));
    }
    let pool = operator
        .runtime
        .runtime()
        .unwrap()
        .incremental_memory_pool();
    drop(operator);
    drop(context);
    assert!(job.gather_owner().close_and_drain().await.is_empty());
    drop(job);
    assert_eq!(pool.reserved(), 0);
}

#[tokio::test]
async fn test_native_count_does_not_visit_materialized_pairs() {
    let mut operator =
        StreamJoinOperator::new("match", left_schema(), right_schema(), spec()).unwrap();
    let job = job();
    let context = StreamOperatorContext::new(&job, "match", None);
    let prepared = operator
        .prepare_batch("left", &left_batch((0..100).collect()), &context)
        .await
        .unwrap();
    operator.commit_prepared("left", prepared).unwrap();
    reset_join_work();
    let prepared = operator
        .prepare_batch("right", &right_batch(vec![0]), &context)
        .await
        .unwrap();
    assert_eq!(prepared.output.len(), 100);
    assert_eq!(
        join_work().native_range_visits,
        100,
        "counting must not enumerate pairs a second time"
    );
    assert!(
        join_work().native_boundary_visits <= 32,
        "two logarithmic boundaries for count and materialization"
    );
}

fn source_batch(owned: bool) -> (SchemaRef, Batch) {
    let original = left_batch((0..65).collect());
    let record = &original.table_payload().unwrap().batches()[0];
    if owned {
        return (left_schema(), original);
    }
    let schema = Arc::new(Schema::new(vec![
        left_schema().field(0).clone(),
        left_schema().field(1).clone(),
        Field::new("amount", DataType::Binary, true),
    ]));
    let record = RecordBatch::try_new(
        Arc::clone(&schema),
        vec![
            Arc::clone(record.column(0)),
            Arc::clone(record.column(1)),
            Arc::new(BinaryArray::from_vec(vec![&[1_u8][..]; 65])),
        ],
    )
    .unwrap();
    (
        schema,
        Batch::table(vec![record], BatchMetadata::default()).unwrap(),
    )
}

#[tokio::test]
async fn test_batched_masks_drive_owned_and_generic_admission() {
    admission_masks::tests::assert_microsecond_copy_skips_scalar_normalization();
    admission_masks::tests::assert_constant_temporal_masks_skip_scalar_rows();
    admission_masks::tests::assert_vectorized_masks_equal_scalar_units_and_finite_boundaries();
    for owned in [false, true] {
        let (schema, batch) = source_batch(owned);
        let mut operator =
            StreamJoinOperator::new("match", schema, right_schema(), spec()).unwrap();
        let job = job();
        let context = StreamOperatorContext::new(&job, "match", None);
        reset_join_work();
        let prepared = operator
            .prepare_batch("left", &batch, &context)
            .await
            .unwrap();
        assert_eq!(prepared.admitted.len(), 65);
        assert_eq!(join_work().admission_mask_blocks, 2);
        assert_eq!(prepared.next_row_id, 65);
        assert_eq!(prepared.admitted[1].record.offset(), 1);
    }
}

#[test]
fn test_dictionary_collision_runs_and_reused_ids_preserve_sorted_dense_rows() {
    assert_dictionary_collision_runs_and_reused_ids_preserve_sorted_dense_rows();
    assert_hot_run_expiry_and_refill_move_only_linear_entries();
    assert_borrowed_hash_and_collision_equality_use_exact_canonical_v1_bytes();
    assert_probe_interner_disambiguates_colliding_composite_v1_keys();
    assert_dictionary_owner_follows_live_rows_when_either_batch_expires_first();
}

fn assert_dictionary_collision_runs_and_reused_ids_preserve_sorted_dense_rows() {
    let pool: Arc<dyn MemoryPool> = Arc::new(GreedyMemoryPool::new(1 << 20));
    let credit = MemoryConsumer::new("collision-index").register(&pool);
    credit
        .try_grow(native_lookup::NativeIndex::build_charge(4).unwrap())
        .unwrap();
    let empty = RecordBatch::new_empty(Arc::new(Schema::empty()));
    let key = |value: u8| Arc::new(vec![value].into());
    let rows = [(1, 9, 3), (2, 0, 0), (1, 5, 2), (1, 5, 1)]
        .into_iter()
        .map(|(value, time, row_id)| StoredRow {
            record: empty.clone().into(),
            event_time: EventTime::from_micros(time),
            row_id,
            charge: 0,
            encoded_key: key(value),
        })
        .collect::<Vec<_>>();
    let mut index = native_lookup::NativeIndex::new_colliding(&rows, credit);
    let range = (EventTime::from_micros(5), EventTime::from_micros(9));
    assert_eq!(index.range(&key(1), range).collect::<Vec<_>>(), [3, 2, 0]);
    assert_eq!(index.range(&key(2), range).count(), 0);
    let dense = [&rows[0], &rows[3], &rows[2]];
    index.remove(&rows[1], Some((&rows[3], 1)), |position| {
        &dense[position].encoded_key
    });
    let next = StoredRow {
        encoded_key: key(3),
        row_id: 4,
        ..rows[1].clone()
    };
    let append = index.reserve(1).unwrap();
    index.append(3, &[next]);
    append.commit();
    assert_eq!(index.range(&key(1), range).collect::<Vec<_>>(), [1, 2, 0]);
    assert_eq!(
        index
            .range(
                &key(2),
                (EventTime::from_micros(0), EventTime::from_micros(9))
            )
            .count(),
        0
    );
    assert_eq!(
        index
            .range(
                &key(3),
                (EventTime::from_micros(0), EventTime::from_micros(9))
            )
            .collect::<Vec<_>>(),
        [3]
    );
    drop(index);
    assert_eq!(pool.reserved(), 0);
}

fn remove_identity(
    index: &mut native_lookup::NativeIndex,
    dense: &mut Vec<&StoredRow>,
    row_id: u64,
) {
    let position = dense.iter().position(|row| row.row_id == row_id).unwrap();
    let row = dense.swap_remove(position);
    let moved = dense.get(position).map(|row| (*row, position));
    index.remove(row, moved, |position| &dense[position].encoded_key);
}

fn assert_hot_run_expiry_and_refill_move_only_linear_entries() {
    let pool: Arc<dyn MemoryPool> = Arc::new(GreedyMemoryPool::new(1 << 20));
    let credit = MemoryConsumer::new("hot-run-index").register(&pool);
    credit
        .try_grow(native_lookup::NativeIndex::build_charge(1_024).unwrap())
        .unwrap();
    let empty = RecordBatch::new_empty(Arc::new(Schema::empty()));
    let key = Arc::new(vec![7].into());
    let rows = (0..1_536)
        .map(|row_id| StoredRow {
            record: empty.clone().into(),
            event_time: EventTime::from_micros(row_id),
            row_id: u64::try_from(row_id).unwrap(),
            charge: 0,
            encoded_key: Arc::clone(&key),
        })
        .collect::<Vec<_>>();
    let mut index = native_lookup::NativeIndex::new(&rows[..1_024], credit);
    let mut dense = rows[..1_024].iter().collect::<Vec<_>>();
    reset_join_work();
    for identity in 0..512 {
        remove_identity(&mut index, &mut dense, identity);
    }
    let append = index.reserve(512).unwrap();
    index.append(dense.len(), &rows[1_024..]);
    dense.extend(&rows[1_024..]);
    append.commit();
    let identities = index
        .range(
            &key,
            (EventTime::from_micros(0), EventTime::from_micros(2_000)),
        )
        .map(|position| dense[position].row_id)
        .collect::<Vec<_>>();
    assert_eq!(identities, (512..1_536).collect::<Vec<_>>());
    for identity in 512..1_536 {
        remove_identity(&mut index, &mut dense, identity);
    }
    assert!(
        join_work().native_shifted_entries <= rows.len(),
        "prefix expiry and geometric compaction must be linear; moved={}",
        join_work().native_shifted_entries
    );
    assert_eq!(pool.reserved(), 1_024);
    drop(index);
    assert_eq!(pool.reserved(), 0);
}

fn assert_borrowed_hash_and_collision_equality_use_exact_canonical_v1_bytes() {
    let mut types = vec![
        DataType::Boolean,
        DataType::Int16,
        DataType::Int32,
        DataType::Int64,
        DataType::UInt8,
        DataType::UInt16,
        DataType::UInt32,
        DataType::UInt64,
        DataType::Utf8,
        DataType::LargeUtf8,
    ];
    for unit in [
        TimeUnit::Second,
        TimeUnit::Millisecond,
        TimeUnit::Microsecond,
        TimeUnit::Nanosecond,
    ] {
        for timezone in [None, Some("UTC"), Some("Europe/Paris")] {
            types.push(DataType::Timestamp(unit, timezone.map(Into::into)));
        }
    }
    let hasher = borrowed_key::KeyHashState::default();
    for data_type in types {
        let columns = [
            datafusion::arrow::array::new_null_array(&data_type, 1),
            Arc::new(StringArray::from(vec!["é\0value"])) as ArrayRef,
        ];
        for indices in [&[0][..], &[0, 1][..], &[1, 0][..]] {
            let canonical = encode_join_key_columns_v1(&columns, 0, indices).unwrap();
            let borrowed = borrowed_key::BorrowedKey {
                columns: &columns,
                row: 0,
                indices,
            };
            assert_eq!(
                borrowed.hash(&hasher).unwrap(),
                borrowed_key::framed_hash(&hasher, &canonical),
                "{data_type:?}, {indices:?}"
            );
            assert!(borrowed.equals(&canonical));
            let mut collision = canonical.clone();
            collision[0] ^= 1;
            assert!(!borrowed.equals(&collision));
            collision = canonical.clone();
            collision.push(0);
            assert!(!borrowed.equals(&collision));
            assert!(!borrowed.equals(&canonical[..canonical.len() - 1]));
        }
    }
}

#[tokio::test]
async fn test_wide_composite_masks_keep_quantum_work_bounded_in_both_paths() {
    for owned in [false, true] {
        let keys = (0..61)
            .map(|column| format!("key_{column}"))
            .collect::<Vec<_>>();
        let mut fields = keys
            .iter()
            .map(|name| Field::new(name, DataType::Int64, false))
            .collect::<Vec<_>>();
        fields.push(left_schema().field(1).clone());
        fields.push(Field::new(
            "amount",
            if owned {
                DataType::Int64
            } else {
                DataType::Binary
            },
            false,
        ));
        let left = Arc::new(Schema::new(fields));
        let mut right_fields = keys
            .iter()
            .map(|name| Field::new(name, DataType::Int64, false))
            .collect::<Vec<_>>();
        right_fields.extend([
            right_schema().field(1).clone(),
            right_schema().field(2).clone(),
        ]);
        let right = Arc::new(Schema::new(right_fields));
        let mut columns: Vec<ArrayRef> = (0..61)
            .map(|column| Arc::new(Int64Array::from(vec![i64::from(column)])) as ArrayRef)
            .collect();
        columns.push(Arc::new(
            TimestampMicrosecondArray::from(vec![0]).with_timezone("UTC"),
        ));
        columns.push(if owned {
            Arc::new(Int64Array::from(vec![1]))
        } else {
            Arc::new(BinaryArray::from_vec(vec![&[1][..]]))
        });
        let batch = Batch::table(
            vec![RecordBatch::try_new(Arc::clone(&left), columns).unwrap()],
            BatchMetadata::default(),
        )
        .unwrap();
        let mut declaration = spec();
        declaration.left_keys = keys.clone();
        declaration.right_keys = keys;
        let mut operator = StreamJoinOperator::new("match", left, right, declaration).unwrap();
        let job = job();
        let context = StreamOperatorContext::new(&job, "match", None);
        let prepared = operator
            .prepare_batch("left", &batch, &context)
            .await
            .unwrap();
        assert_eq!(prepared.admitted.len(), 1);
        assert_eq!(prepared.retained.len(), 1);
    }
}

fn assert_probe_interner_disambiguates_colliding_composite_v1_keys() {
    let schema = Arc::new(Schema::new(vec![
        Field::new("first", DataType::Utf8, false),
        Field::new("second", DataType::Utf8, false),
    ]));
    let admitted = [
        ("a", "bc"),
        ("ab", "c"),
        ("a", "bc"),
        ("é\0", "値"),
        ("ab", "c"),
    ]
    .into_iter()
    .enumerate()
    .map(|(position, (first, second))| AdmittedRow {
        record: RecordBatch::try_new(
            Arc::clone(&schema),
            vec![
                Arc::new(StringArray::from(vec![first])),
                Arc::new(StringArray::from(vec![second])),
            ],
        )
        .unwrap()
        .into(),
        event_time: EventTime::from_micros(0),
        row_id: position as u64,
        retain: true,
    })
    .collect::<Vec<_>>();
    let pool: Arc<dyn MemoryPool> = Arc::new(GreedyMemoryPool::new(1 << 20));
    let credit = MemoryConsumer::new("collision-interner").register(&pool);
    credit.try_grow(65_536).unwrap();
    reset_join_work();
    let keys = native_lookup::colliding_probe_keys(&admitted, &[0, 1], credit).unwrap();
    assert_eq!(join_work().key_encodings, 3);
    assert!(Arc::ptr_eq(keys.key(0), keys.key(2)));
    assert!(Arc::ptr_eq(keys.key(1), keys.key(4)));
    for (first, second) in [(0, 1), (0, 3), (1, 3)] {
        assert!(!Arc::ptr_eq(keys.key(first), keys.key(second)));
    }
    for (row, key) in admitted.iter().zip(keys.row_keys()) {
        assert_eq!(
            key.as_slice(),
            encode_join_key_columns_v1(row.record.columns(), row.record.offset(), &[0, 1]).unwrap()
        );
    }
    drop(keys);
    assert_eq!(pool.reserved(), 0);
}

fn assert_dictionary_owner_follows_live_rows_when_either_batch_expires_first() {
    for times in [[0, 10], [10, 0]] {
        let pool: Arc<dyn MemoryPool> = Arc::new(GreedyMemoryPool::new(1 << 20));
        let columns = [Arc::new(Int64Array::from(vec![7])) as ArrayRef];
        let record = RecordBatch::new_empty(Arc::new(Schema::empty()));
        let stored = times
            .into_iter()
            .enumerate()
            .map(|(identity, time)| {
                let credit = MemoryConsumer::new("batch-key").register(&pool);
                credit.try_grow(1_024).unwrap();
                StoredRow {
                    record: record.clone().into(),
                    event_time: EventTime::from_micros(time),
                    row_id: identity as u64,
                    charge: 0,
                    encoded_key: columnar::funded_key(&columns, 0, &[0], Arc::new(credit)).unwrap(),
                }
            })
            .collect::<Vec<_>>();
        let credit = MemoryConsumer::new("owner-index").register(&pool);
        credit
            .try_grow(native_lookup::NativeIndex::build_charge(stored.len()).unwrap())
            .unwrap();
        let index = native_lookup::NativeIndex::new(&stored, credit);
        let mut rows = RetainedRows(
            Arc::new(stored),
            Some(index),
            columnar::SparseQueue::default(),
        );
        let expired = usize::from(times[1] < times[0]);
        drop(rows.swap_remove(expired));
        let index = rows.1.as_ref().unwrap();
        assert_eq!(
            pool.reserved(),
            index.funded_bytes() + 1_024,
            "dictionary must not retain the expired batch's pooled key credit; times={times:?}"
        );
        drop(rows);
        assert_eq!(pool.reserved(), 0);
    }
}
