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
    let expiration_append = operator
        .state
        .left_expirations
        .prepare_append(
            &prepared.retained,
            &prepared.chunks.indices,
            &context,
            false,
        )
        .await
        .unwrap();
    operator
        .commit_prepared("left", prepared, true, expiration_append)
        .unwrap();
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
    assert_eq!(join_work().key_encodings, 1);
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
    let keys = operator
        .native_probe_keys(&plan, &bundle.admitted)
        .unwrap()
        .unwrap();
    assert_eq!(
        Arc::strong_count(keys.keys[0].as_ref().unwrap()),
        2,
        "one dictionary lease and one distinct probe owner must serve every repeated position"
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
    assert_eq!(join_work().arena_frames, 0);
    assert_eq!(join_work().key_encodings, 1);
    assert!(join_work().key_type_resolutions <= 2);
    assert_eq!(keys.keys.len(), 1);
    assert_eq!(Arc::strong_count(keys.keys[0].as_ref().unwrap()), 2);
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
async fn test_batch_admission_does_not_stage_owned_rows() {
    Box::pin(assert_sparse_generic_sql_has_no_owned_admission_rows()).await;
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
    let mut bundle = operator.admission_bundle(&plan, &source).unwrap();
    reset_join_work();
    let mut source_base = 0;
    for record in source.table_payload().unwrap().batches() {
        operator
            .admit_record(record, &plan, "left", &context, &mut bundle, source_base)
            .await
            .unwrap();
        source_base += record.num_rows();
    }
    assert_eq!(bundle.admitted.len(), 130);
    assert_eq!(bundle.next_row_id, 130);
    assert_eq!(bundle.admitted.record_count(), 2);
    assert_eq!(
        bundle.admitted.sparse_index_capacity(),
        0,
        "dense records must not allocate identity indices"
    );
    assert_eq!(bundle.admitted.sparse_index_len(), 0);
    assert_eq!(join_work().sparse_index_writes, 0);
    for row in &bundle.admitted {
        assert!(
            row.record.owner_count() <= 1,
            "one payload owner per record segment"
        );
    }
    let mut visited = 0;
    let allocated = allocation_counter::measure(|| {
        for row in &bundle.admitted {
            assert!(row.record.owner_count() <= 1);
            visited += usize::from(row.retain);
        }
    });
    assert_eq!(visited, 130);
    assert_eq!(
        allocated.count_total, 0,
        "borrowed iteration must not allocate row descriptors"
    );
    assert_eq!(
        join_work().admitted_row_records,
        0,
        "incoming admission must own record segments rather than one descriptor per admitted row"
    );
}

async fn assert_sparse_generic_sql_has_no_owned_admission_rows() {
    let left = generic_sql_schema(&left_schema());
    let right = generic_sql_schema(&right_schema());
    let record = generic_sql_record;
    let (mut operator, pool) = generic_sql_operator(&left, &right);
    let job = job();
    let context = StreamOperatorContext::new(&job, "match", None);
    let opposite = Batch::table(
        vec![record(Arc::clone(&right), vec![Some(7)], vec![Some(0)])],
        BatchMetadata::default(),
    )
    .unwrap();
    commit_generic_opposite(&mut operator, &opposite, &context).await;
    let source = Batch::table(
        vec![
            record(
                Arc::clone(&left),
                vec![Some(7), None, Some(7)],
                vec![Some(0), Some(0), None],
            ),
            record(
                Arc::clone(&left),
                vec![None, Some(7), Some(7)],
                vec![Some(0), Some(0), Some(0)],
            ),
        ],
        BatchMetadata::default(),
    )
    .unwrap();
    reset_join_work();
    let prepared = operator
        .prepare_batch("left", &source, &context)
        .await
        .unwrap();
    assert_eq!(
        join_work().sql_probe_table_builds,
        1,
        "real native grant refusal must execute generic SQL for accepted Int8 keys"
    );
    assert!(pool.refusals.load(std::sync::atomic::Ordering::SeqCst) > 0);
    assert_eq!(
        prepared
            .output
            .iter()
            .map(|pair| (pair.pos, pair.opposite_index))
            .collect::<Vec<_>>(),
        [(0, 0), (1, 0), (2, 0)]
    );
    assert_eq!(
        prepared
            .admitted
            .iter()
            .map(|row| row.row_id)
            .collect::<Vec<_>>(),
        [0, 4, 5]
    );
    assert_eq!(prepared.metrics.null_event_time_rows, 1);
    assert_eq!(prepared.metrics.null_key_rows, 2);
    assert_eq!(prepared.next_row_id, 6);
    assert_generic_materialized(&operator, &prepared, 3);
    for (row, (record, physical)) in prepared.retained.iter().zip([(0, 0), (1, 1), (1, 2)]) {
        let original = &source.table_payload().unwrap().batches()[record];
        assert_eq!(
            row.charge,
            state_row_charge(original, physical, &[0], "match").unwrap()
        );
    }
    assert_eq!(
        join_work().admitted_row_records,
        0,
        "sparse generic SQL must use the same segment ownership path"
    );
    let expiration = operator
        .state
        .left_expirations
        .prepare_append(
            &prepared.retained,
            &prepared.chunks.indices,
            &context,
            false,
        )
        .await
        .unwrap();
    operator
        .commit_prepared("left", prepared, true, expiration)
        .unwrap();
    Box::pin(assert_generic_post_restore(
        &mut operator,
        (left, right),
        &context,
    ))
    .await;
}

async fn commit_generic_opposite(
    operator: &mut StreamJoinOperator,
    opposite: &Batch,
    context: &StreamOperatorContext<'_>,
) {
    let prepared = operator
        .prepare_batch("right", opposite, context)
        .await
        .unwrap();
    let expiration = operator
        .state
        .right_expirations
        .prepare_append(&prepared.retained, &prepared.chunks.indices, context, false)
        .await
        .unwrap();
    operator
        .commit_prepared("right", prepared, true, expiration)
        .unwrap();
}

fn generic_sql_operator(
    left: &SchemaRef,
    right: &SchemaRef,
) -> (
    StreamJoinOperator,
    Arc<persistent_dictionary_refusal_tests::DictionaryPool>,
) {
    let mut operator =
        StreamJoinOperator::new("match", Arc::clone(left), Arc::clone(right), spec()).unwrap();
    let pool = Arc::new(persistent_dictionary_refusal_tests::DictionaryPool::default());
    operator
        .runtime
        .initialize_benchmark_pool(pool.clone())
        .unwrap();
    pool.mode.store(3, std::sync::atomic::Ordering::SeqCst);
    (operator, pool)
}

fn generic_sql_schema(schema: &SchemaRef) -> SchemaRef {
    Arc::new(Schema::new(
        schema
            .fields()
            .iter()
            .enumerate()
            .map(|(index, field)| {
                if index == 0 {
                    field
                        .as_ref()
                        .clone()
                        .with_data_type(DataType::Int8)
                        .with_nullable(true)
                } else {
                    field.as_ref().clone().with_nullable(true)
                }
            })
            .collect::<Vec<_>>(),
    ))
}

fn generic_sql_record(
    schema: SchemaRef,
    mut keys: Vec<Option<i8>>,
    mut times: Vec<Option<i64>>,
) -> RecordBatch {
    use datafusion::arrow::array::Int8Array;
    let rows = keys.len();
    keys.insert(0, Some(99));
    keys.push(Some(99));
    times.insert(0, Some(999));
    times.push(Some(999));
    let columns: Vec<ArrayRef> = vec![
        Arc::new(Int8Array::from(keys)),
        Arc::new(TimestampMicrosecondArray::from(times).with_timezone("UTC")),
        datafusion::arrow::array::new_null_array(schema.field(2).data_type(), rows + 2),
    ];
    RecordBatch::try_new(schema, columns)
        .unwrap()
        .slice(1, rows)
}

async fn assert_generic_post_restore(
    operator: &mut StreamJoinOperator,
    schemas: (SchemaRef, SchemaRef),
    context: &StreamOperatorContext<'_>,
) {
    let (left, right) = schemas;
    let record = generic_sql_record;
    operator.checkpoint_v1_test_producer = true;
    operator.prepare_checkpoint_async(context).await.unwrap();
    let snapshot = operator.checkpoint(Epoch::INITIAL).unwrap();
    assert_eq!(snapshot.inline_metadata["layout_version"], 1);
    let mut restored = StreamJoinOperator::new("match", Arc::clone(&left), right, spec()).unwrap();
    restored.restore(&snapshot).unwrap();
    let follow = Batch::table(
        vec![record(left, vec![None, Some(7)], vec![Some(0); 2])],
        BatchMetadata::default(),
    )
    .unwrap();
    let prepared = restored
        .prepare_batch("left", &follow, context)
        .await
        .unwrap();
    assert_eq!(prepared.admitted.row(0).row_id, 7);
    assert_eq!(matches::Matches::pair(&prepared.output, 0).pos, 0);
    assert_eq!(prepared.next_row_id, 8);
    assert_eq!(prepared.metrics.null_key_rows, 3);
    assert_generic_materialized(&restored, &prepared, 1);
}

fn assert_generic_materialized(
    operator: &StreamJoinOperator,
    prepared: &PreparedJoinBatch,
    rows: usize,
) {
    use datafusion::arrow::array::Int8Array;
    let schema = operator.output_ports()[0].schema().unwrap();
    let actual = materialize_output_record(
        schema,
        &prepared.admitted,
        &operator.state.right,
        &prepared.output,
        true,
        "match",
    )
    .unwrap();
    let expected = RecordBatch::try_new(
        Arc::clone(schema),
        vec![
            Arc::new(Int8Array::from(vec![7; rows])) as ArrayRef,
            Arc::new(TimestampMicrosecondArray::from(vec![0; rows]).with_timezone("UTC")),
            datafusion::arrow::array::new_null_array(&DataType::Int64, rows),
            Arc::new(Int8Array::from(vec![7; rows])),
            Arc::new(TimestampMicrosecondArray::from(vec![0; rows]).with_timezone("UTC")),
            datafusion::arrow::array::new_null_array(&DataType::Utf8, rows),
        ],
    )
    .unwrap();
    assert_eq!(actual, expected);
}

#[tokio::test]
async fn test_batch_admission_workspace_preflight_precedes_semantic_rows() {
    use datafusion::execution::memory_pool::{GreedyMemoryPool, MemoryPool};
    for (records, rows, spills) in [(1, 65, true), (5, 1, true), (1, 64, false), (4, 1, false)] {
        let mut operator =
            StreamJoinOperator::new("match", left_schema(), right_schema(), spec()).unwrap();
        let pool = Arc::new(GreedyMemoryPool::new(0));
        operator.runtime.runtime = Some(
            crate::DataFusionRuntime::with_benchmark_pool(
                DataFusionConfig::default(),
                pool.clone(),
            )
            .unwrap(),
        );
        operator.state.next_left_row_id = u64::MAX;
        let source = Batch::table(
            (0..records)
                .map(|_| left_batch(vec![0; rows]).table_payload().unwrap().batches()[0].clone())
                .collect(),
            BatchMetadata::default(),
        )
        .unwrap();
        let job = job();
        let context = StreamOperatorContext::new(&job, "match", None);
        let before = operator.status();
        let error = operator
            .prepare_batch("left", &source, &context)
            .await
            .err()
            .expect("workspace or row ID refusal");
        let message = error.to_string();
        assert_eq!(
            message.contains("admission workspace"),
            spills,
            "{records} records × {rows} rows: {message}"
        );
        if !spills {
            assert!(
                message.contains("row_id"),
                "inline metadata requires no heap headroom: {message}"
            );
        }
        assert_eq!(operator.status(), before);
        assert_eq!(pool.reserved(), 0);
    }
}

#[tokio::test]
async fn test_batch_admission_workspace_capacity_boundary() {
    use datafusion::execution::memory_pool::{GreedyMemoryPool, MemoryPool};
    for (records, rows) in [(1, 65), (5, 1)] {
        let source = Batch::table(
            (0..records)
                .map(|_| left_batch(vec![0; rows]).table_payload().unwrap().batches()[0].clone())
                .collect(),
            BatchMetadata::default(),
        )
        .unwrap();
        let mut funded =
            StreamJoinOperator::new("match", left_schema(), right_schema(), spec()).unwrap();
        let plan = funded.begin_batch("left", &source).unwrap();
        let bundle = funded.admission_bundle(&plan, &source).unwrap();
        let fee = bundle.admitted.workspace_bytes();
        assert!(
            fee >= records * rows * (size_of::<usize>() + size_of::<i64>() + size_of::<u64>())
                + (records * rows).div_ceil(64) * size_of::<u64>()
        );
        drop(bundle);
        for capacity in [fee - 1, fee] {
            let pool: Arc<dyn MemoryPool> = Arc::new(GreedyMemoryPool::new(capacity));
            let mut operator =
                StreamJoinOperator::new("match", left_schema(), right_schema(), spec()).unwrap();
            operator
                .runtime
                .initialize_benchmark_pool(Arc::clone(&pool))
                .unwrap();
            let before = operator.status();
            let result = operator.admission_bundle(&plan, &source);
            assert_eq!(result.is_ok(), capacity == fee);
            if let Ok(bundle) = result {
                assert_eq!(pool.reserved(), fee);
                assert_eq!(bundle.admitted.sparse_index_capacity(), 0);
                assert_eq!(
                    bundle.admitted.retention_metadata_bytes(),
                    (records * rows).div_ceil(64) * 8
                );
                drop(bundle);
            }
            assert_eq!(operator.status(), before);
            assert_eq!(pool.reserved(), 0);
        }
    }
}

#[tokio::test]
async fn test_batch_admission_mixed_records_copy_and_borrow() {
    let first = left_batch(vec![0, 1, 2]).table_payload().unwrap().batches()[0].clone();
    let second = generic_records()[0].slice(0, 3);
    let source = Batch::table(vec![first, second], BatchMetadata::default()).unwrap();
    let mut operator =
        StreamJoinOperator::new("match", left_schema(), right_schema(), spec()).unwrap();
    let job = job();
    let context = StreamOperatorContext::new(&job, "match", None);
    let plan = operator.begin_batch("left", &source).unwrap();
    let mut bundle = operator.admission_bundle(&plan, &source).unwrap();
    let records = source.table_payload().unwrap().batches();
    operator
        .admit_record(&records[0], &plan, "left", &context, &mut bundle, 0)
        .await
        .unwrap();
    assert!(
        matches!(
            bundle.admitted.row(0).record.owner(),
            Some(columnar::RowPayload::Shared { .. })
        ),
        "first record must actually copy"
    );
    operator
        .admit_record(&records[1], &plan, "left", &context, &mut bundle, 3)
        .await
        .unwrap();
    assert!(
        bundle.admitted.row(3).record.owner().is_none(),
        "nullable second payload must actually stay borrowed"
    );
    assert_eq!(
        bundle
            .admitted
            .iter()
            .map(|row| row.row_id)
            .collect::<Vec<_>>(),
        [0, 1, 2, 3, 4, 5]
    );
    assert_eq!(bundle.next_row_id, 6);
    bundle.admitted.prepare_retained_parents();
    let retained = retained_rows(
        &bundle.admitted,
        &plan.key_indices,
        "match",
        None,
        &operator.key_dictionary,
        &context,
        &mut operator.key_work,
    )
    .await
    .unwrap();
    for (index, row) in retained.iter().enumerate() {
        let original = &records[index / 3];
        let key =
            encode_join_key_columns_v1(original.columns(), index % 3, &plan.key_indices).unwrap();
        assert_eq!(
            row.charge,
            state_columns_charge_with_key(original.columns(), index % 3, key.len(), "match")
                .unwrap()
        );
    }
}

#[tokio::test]
async fn test_batch_admission_mixed_pending_records_keep_table_after_replacement() {
    let source = Batch::table(generic_records(), BatchMetadata::default()).unwrap();
    let records = source.table_payload().unwrap().batches();
    let mut operator =
        StreamJoinOperator::new("match", left_schema(), right_schema(), spec()).unwrap();
    let job = job();
    let context = StreamOperatorContext::new(&job, "match", None);
    let plan = operator.begin_batch("left", &source).unwrap();
    let mut bundle = operator.admission_bundle(&plan, &source).unwrap();
    operator
        .admit_record(&records[0], &plan, "left", &context, &mut bundle, 0)
        .await
        .unwrap();
    bundle.admitted.replace(
        0,
        columnar::RowPayload::Rowed {
            parent: Arc::new(records[0].clone()),
            row: 0,
        },
    );
    operator
        .admit_record(&records[1], &plan, "left", &context, &mut bundle, 65)
        .await
        .unwrap();
    assert_eq!(bundle.admitted.row(65).record.offset(), 0);
    assert!(Arc::ptr_eq(
        bundle.admitted.row(65).record.column(2),
        records[1].column(2)
    ));
    assert_eq!(bundle.admitted.row(129).row_id, 129);
}

#[tokio::test]
async fn test_batch_admission_last_dropped_record_releases_temporary_table() {
    let source = Batch::table(
        vec![
            left_batch(vec![1]).table_payload().unwrap().batches()[0].clone(),
            left_batch(vec![-1]).table_payload().unwrap().batches()[0].clone(),
        ],
        BatchMetadata::default(),
    )
    .unwrap();
    let omitted = Arc::downgrade(source.table_payload().unwrap().batches()[1].column(2));
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
                IngressProgress::new(IngressState::Active, Some(EventTime::from_micros(0))),
            ),
            (
                "right".into(),
                IngressProgress::new(IngressState::Active, None),
            ),
        ])),
    );
    let plan = operator.begin_batch("left", &source).unwrap();
    let mut bundle = operator.admission_bundle(&plan, &source).unwrap();
    for (index, record) in source.table_payload().unwrap().batches().iter().enumerate() {
        operator
            .admit_record(record, &plan, "left", &context, &mut bundle, index)
            .await
            .unwrap();
    }
    assert_eq!(bundle.admitted.len(), 1);
    assert!(matches!(
        bundle.admitted.row(0).record.owner(),
        Some(columnar::RowPayload::Shared { .. })
    ));
    assert_eq!(bundle.metrics.late_rows, 1);
    assert_eq!(bundle.next_row_id, 2);
    drop(source);
    assert!(
        omitted.upgrade().is_none(),
        "the completed batch must release its dropped suffix immediately"
    );
}

#[tokio::test]
async fn test_batch_admission_retention_uses_funded_packed_words() {
    for (records, rows, bytes) in [(1, 64, 8), (1, 65, 16), (2, 65, 24), (4, 1, 8), (5, 1, 8)] {
        let source = Batch::table(
            (0..records)
                .map(|_| left_batch(vec![0; rows]).table_payload().unwrap().batches()[0].clone())
                .collect(),
            BatchMetadata::default(),
        )
        .unwrap();
        let mut operator =
            StreamJoinOperator::new("match", left_schema(), right_schema(), spec()).unwrap();
        let plan = operator.begin_batch("left", &source).unwrap();
        let bundle = operator.admission_bundle(&plan, &source).unwrap();
        assert_eq!(
            bundle.admitted.retention_metadata_bytes(),
            bytes,
            "actual retained-bit storage capacity"
        );
        let inline = records <= 4 && records * rows <= 64;
        assert_eq!(bundle.admitted.workspace_bytes() == 0, inline);
        let pool = operator
            .runtime
            .runtime()
            .unwrap()
            .incremental_memory_pool();
        assert_eq!(pool.reserved(), bundle.admitted.workspace_bytes());
        drop(bundle);
        assert_eq!(pool.reserved(), 0);
    }
}

#[tokio::test]
async fn test_batch_admission_sparse_promotion_keeps_prefix_and_coordinates() {
    let mut fields = left_schema()
        .fields()
        .iter()
        .map(|field| field.as_ref().clone())
        .collect::<Vec<_>>();
    fields[0] = fields[0].clone().with_nullable(true);
    let schema = Arc::new(Schema::new(fields));
    let record = |rows: usize, gap: Option<usize>| {
        RecordBatch::try_new(
            Arc::clone(&schema),
            vec![
                Arc::new(Int64Array::from(
                    (0..rows)
                        .map(|row| (Some(row) != gap).then_some(7))
                        .collect::<Vec<_>>(),
                )),
                Arc::new(TimestampMicrosecondArray::from(vec![0; rows]).with_timezone("UTC")),
                Arc::new(Int64Array::from(vec![None::<i64>; rows])),
            ],
        )
        .unwrap()
    };
    let source = Batch::table(
        vec![record(66, Some(64)), record(65, None)],
        BatchMetadata::default(),
    )
    .unwrap();
    let mut operator = StreamJoinOperator::new("match", schema, right_schema(), spec()).unwrap();
    let job = job();
    let context = StreamOperatorContext::new(&job, "match", None);
    let plan = operator.begin_batch("left", &source).unwrap();
    let mut bundle = operator.admission_bundle(&plan, &source).unwrap();
    let mut base = 0;
    reset_join_work();
    for record in source.table_payload().unwrap().batches() {
        operator
            .admit_record(record, &plan, "left", &context, &mut bundle, base)
            .await
            .unwrap();
        base += record.num_rows();
    }
    assert_eq!(
        bundle.admitted.sparse_index_len(),
        65,
        "only the sparse segment stores physical indices"
    );
    assert_eq!(join_work().sparse_prefix_promoted, 64);
    assert_eq!(join_work().sparse_index_writes, 65);
    assert_eq!(bundle.admitted.row(64).row_id, 65);
    assert_eq!(bundle.admitted.row(64).record.offset(), 65);
    assert_eq!(bundle.admitted.row(65).row_id, 66);
    assert_eq!(bundle.admitted.row(65).record.offset(), 0);
    assert_eq!(bundle.admitted.row(129).row_id, 130);
    assert_eq!(bundle.next_row_id, 131);
    let expected = (0..131).filter(|&row| row != 64).collect::<Vec<u64>>();
    assert_eq!(
        admitted_charge_cache(Some((0..131).collect()), &bundle.admitted).unwrap(),
        expected
    );
    let pool = operator
        .runtime
        .runtime()
        .unwrap()
        .incremental_memory_pool();
    assert_eq!(pool.reserved(), bundle.admitted.workspace_bytes());
    drop(bundle);
    assert_eq!(pool.reserved(), 0);
}

#[tokio::test]
async fn test_batch_admission_sparse_promotion_cancel_and_abandon() {
    assert_sparse_promotion_stopped(false).await;
    assert_sparse_promotion_stopped(true).await;
}

async fn assert_sparse_promotion_stopped(cancel: bool) {
    const PREFIX: usize = 16_384;
    let mut fields = left_schema()
        .fields()
        .iter()
        .map(|field| field.as_ref().clone())
        .collect::<Vec<_>>();
    fields[0] = fields[0].clone().with_nullable(true);
    let schema = Arc::new(Schema::new(fields));
    let record = RecordBatch::try_new(
        Arc::clone(&schema),
        vec![
            Arc::new(Int64Array::from(
                (0..PREFIX + 2)
                    .map(|row| (row != PREFIX).then_some(7))
                    .collect::<Vec<_>>(),
            )),
            Arc::new(TimestampMicrosecondArray::from(vec![0; PREFIX + 2]).with_timezone("UTC")),
            Arc::new(Int64Array::from(vec![None::<i64>; PREFIX + 2])),
        ],
    )
    .unwrap();
    let source = Batch::table(vec![record.clone()], BatchMetadata::default()).unwrap();
    let mut operator = StreamJoinOperator::new("match", schema, right_schema(), spec()).unwrap();
    let before = operator.status();
    let job = job();
    let context = StreamOperatorContext::new(&job, "match", None);
    let plan = operator.begin_batch("left", &source).unwrap();
    let mut bundle = operator.admission_bundle(&plan, &source).unwrap();
    let entered = Arc::new(tokio::sync::Notify::new());
    let _observation = admitted::observe_sparse_promotion(Arc::clone(&entered));
    reset_join_work();
    let mut admission =
        Box::pin(operator.admit_record(&record, &plan, "left", &context, &mut bundle, 0));
    tokio::select! {
        biased;
        () = entered.notified() => {},
        result = &mut admission => panic!("admission finished before funded sparse promotion yielded: {result:?}"),
    }
    let promoted = join_work().sparse_prefix_promoted;
    assert!(
        promoted > 0 && promoted < PREFIX,
        "promotion must yield with a genuine partial prefix: {promoted}"
    );
    if cancel {
        job.cancellation().cancel();
        assert!(matches!(
            admission.as_mut().await,
            Err(CalcFlowError::Cancelled { .. })
        ));
    }
    drop(admission);
    assert_eq!(bundle.admitted.len(), PREFIX);
    assert_eq!(
        bundle.admitted.sparse_index_len(),
        0,
        "a dropped or cancelled promotion rolls back its uninstalled prefix"
    );
    assert_eq!(bundle.admitted.row(PREFIX - 1).row_id, (PREFIX - 1) as u64);
    assert_eq!(bundle.admitted.row(PREFIX - 1).record.offset(), PREFIX - 1);
    assert_eq!(operator.status(), before);
    assert_eq!(operator.state.next_left_row_id, 0);
    assert_eq!(source.table_payload().unwrap().batches()[0], record);
    let pool = operator
        .runtime
        .runtime()
        .unwrap()
        .incremental_memory_pool();
    assert_eq!(pool.reserved(), bundle.admitted.workspace_bytes());
    drop(bundle);
    assert_eq!(pool.reserved(), 0);
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
    let mut bundle = operator.admission_bundle(&plan, &source).unwrap();
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
    assert_eq!(
        (0..bundle.admitted.len())
            .map(|index| bundle.admitted.source_index(index))
            .collect::<Vec<_>>(),
        (0..130).collect::<Vec<_>>()
    );
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
    let mut bundle = operator.admission_bundle(&plan, &source).unwrap();
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
    assert_eq!(
        (0..bundle.admitted.len())
            .map(|index| bundle.admitted.source_index(index))
            .collect::<Vec<_>>(),
        (0..130).collect::<Vec<_>>()
    );
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
    let mut bundle = operator.admission_bundle(&plan, &source).unwrap();
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
    assert_eq!(bundle.admitted.row(0).row_id, u64::MAX - 1);
    assert_eq!(join_work().scalar_admissions, 2);
    assert_eq!(join_work().generic_fast_rows, 0);
    assert_eq!(operator.state.next_left_row_id, u64::MAX - 1);
}
