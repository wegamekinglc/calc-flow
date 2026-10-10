use super::*;
use datafusion::arrow::array::{
    Int16Array, Int32Array, LargeStringArray, UInt8Array, UInt16Array, UInt32Array,
};

pub(super) fn index_funding(operator: &StreamJoinOperator) -> usize {
    [&operator.state.left, &operator.state.right]
        .into_iter()
        .map(|rows| {
            rows.1.as_ref().map_or(0, |index| {
                let expected = index.resident_bytes();
                assert_eq!(index.row_count(), rows.len());
                assert_eq!(
                    index.funded_bytes(),
                    expected,
                    "all resident native index capacities must remain funded"
                );
                expected
            })
        })
        .sum()
}

pub(super) fn payload_funding<'a>(
    records: impl Iterator<Item = &'a columnar::RowPayload>,
) -> usize {
    records
        .filter_map(columnar::RowPayload::funded_owner)
        .collect::<BTreeMap<_, _>>()
        .values()
        .sum()
}

pub(super) fn state_funding(operator: &StreamJoinOperator) -> usize {
    let rows = operator.state.left.iter().chain(&*operator.state.right);
    let pending = operator
        .state
        .deltas
        .pending
        .iter()
        .filter_map(|op| match op {
            PendingOp::Upsert { record, .. } => Some(record),
            PendingOp::Tombstone { .. } => None,
        });
    let payloads = payload_funding(rows.clone().map(|row| &row.record).chain(pending));
    let pending_keys = operator.state.deltas.pending.iter().map(|op| match op {
        PendingOp::Upsert { encoded_key, .. } | PendingOp::Tombstone { encoded_key, .. } => {
            encoded_key
        }
    });
    let keys = rows
        .map(|row| &row.encoded_key)
        .chain(pending_keys)
        .filter_map(|key| key.funded_owner())
        .collect::<BTreeMap<_, _>>();
    index_funding(operator) + payloads + keys.values().sum::<usize>()
}

pub(super) fn assert_resident_and_gather_funding(
    pool: &dyn datafusion::execution::memory_pool::MemoryPool,
    job: &StreamJobContext,
    resident: usize,
) -> usize {
    let (home, generation, attempt) = job.gather_owner().funding();
    assert_eq!(attempt, 0, "completed attempts must refund their workspace");
    assert_eq!(pool.reserved(), resident + home + generation);
    eprintln!("actual funding: resident={resident}, home={home}, generation={generation}");
    pool.reserved()
}

fn key_cases() -> Vec<ArrayRef> {
    let mut arrays: Vec<ArrayRef> = vec![
        Arc::new(BooleanArray::from(vec![true, false, false, true])),
        Arc::new(Int16Array::from(vec![
            i16::MAX,
            i16::MIN,
            i16::MIN,
            i16::MAX,
        ])),
        Arc::new(Int32Array::from(vec![
            i32::MAX,
            i32::MIN,
            i32::MIN,
            i32::MAX,
        ])),
        Arc::new(Int64Array::from(vec![
            i64::MAX,
            i64::MIN,
            i64::MIN,
            i64::MAX,
        ])),
        Arc::new(UInt8Array::from(vec![u8::MAX, 0, 0, u8::MAX])),
        Arc::new(UInt16Array::from(vec![u16::MAX, 0, 0, u16::MAX])),
        Arc::new(UInt32Array::from(vec![u32::MAX, 0, 0, u32::MAX])),
        Arc::new(UInt64Array::from(vec![u64::MAX, 0, 0, u64::MAX])),
        Arc::new(StringArray::from(vec!["é", "", "", "é"])),
        Arc::new(LargeStringArray::from(vec!["é", "", "", "é"])),
    ];
    for unit in [
        TimeUnit::Second,
        TimeUnit::Millisecond,
        TimeUnit::Microsecond,
        TimeUnit::Nanosecond,
    ] {
        for timezone in [None, Some("UTC"), Some("Europe/Paris")] {
            arrays.push(timestamp_keys(unit, timezone));
        }
    }
    arrays
}

fn timestamp_keys(unit: TimeUnit, timezone: Option<&str>) -> ArrayRef {
    macro_rules! keys {
        ($array:ty) => {
            Arc::new(<$array>::from(vec![1_000, 1_001, 1_001, 1_000]).with_timezone_opt(timezone))
                as ArrayRef
        };
    }
    match unit {
        TimeUnit::Second => keys!(TimestampSecondArray),
        TimeUnit::Millisecond => keys!(TimestampMillisecondArray),
        TimeUnit::Microsecond => keys!(TimestampMicrosecondArray),
        TimeUnit::Nanosecond => keys!(TimestampNanosecondArray),
    }
}

fn typed_schema(schema: &SchemaRef, key_type: &DataType) -> SchemaRef {
    Arc::new(Schema::new(
        schema
            .fields()
            .iter()
            .enumerate()
            .map(|(index, field)| {
                if index == 0 {
                    field.as_ref().clone().with_data_type(key_type.clone())
                } else {
                    field.as_ref().clone()
                }
            })
            .collect::<Vec<_>>(),
    ))
}

fn typed_batch(batch: &Batch, schema: &SchemaRef, key: &ArrayRef) -> Batch {
    let mut columns = batch.table_payload().unwrap().batches()[0]
        .columns()
        .to_vec();
    columns[0] = Arc::clone(key);
    let record = RecordBatch::try_new(Arc::clone(schema), columns).unwrap();
    Batch::table(vec![record], BatchMetadata::default()).unwrap()
}

async fn admitted_probe(
    operator: &mut StreamJoinOperator,
    ingress: &str,
    batch: &Batch,
    context: &StreamOperatorContext<'_>,
) -> (SidePlan, Vec<AdmittedRow>) {
    let plan = operator.begin_batch(ingress, batch).unwrap();
    let mut bundle = operator.admission_bundle(&plan);
    operator
        .admit_record(
            &batch.table_payload().unwrap().batches()[0],
            &plan,
            ingress,
            context,
            &mut bundle,
            0,
        )
        .await
        .unwrap();
    bundle.finish(&operator.name).unwrap();
    (plan, bundle.admitted)
}

async fn assert_reference(
    operator: &mut StreamJoinOperator,
    ingress: &str,
    batch: &Batch,
    context: &StreamOperatorContext<'_>,
) {
    let (plan, admitted) = admitted_probe(operator, ingress, batch, context).await;
    let (reference, admitted) = operator.legacy_matches(&plan, admitted, context).await;
    let reference = reference.unwrap();
    reset_join_work();
    let native = operator
        .native_matches(&plan, &admitted)
        .unwrap()
        .expect("eligible native key");
    assert_eq!(join_work().sql_probe_table_builds, 0);
    let identities = |pairs: &[MatchedPair]| {
        pairs
            .iter()
            .map(|pair| (pair.pos, pair.opposite_index))
            .collect::<Vec<_>>()
    };
    assert_eq!(identities(&native.pairs), identities(&reference));
    let opposite = if plan.incoming_is_left {
        &operator.state.right
    } else {
        &operator.state.left
    };
    let materialize = |pairs: &[MatchedPair]| {
        materialize_output_record(
            operator.output_ports()[0].schema().unwrap(),
            &admitted,
            opposite,
            pairs,
            plan.incoming_is_left,
            "match",
        )
        .unwrap()
    };
    assert_eq!(materialize(&native.pairs), materialize(&reference));
}

async fn compare_key_case(key: &ArrayRef, incoming_is_left: bool) {
    let left = typed_schema(&left_schema(), key.data_type());
    let right = typed_schema(&right_schema(), key.data_type());
    let mut operator =
        StreamJoinOperator::new("match", left.clone(), right.clone(), spec()).unwrap();
    let left_batch = typed_batch(&left_batch(vec![3, 0, 0, 2]), &left, key);
    let right_batch = typed_batch(&right_batch(vec![2, 1, 1, 0]), &right, key);
    let job = job();
    let context = StreamOperatorContext::new(&job, "match", None);
    let mut collector = EdgeCollector::new(operator.output_ports().to_vec());
    let (first, first_batch, probe, probe_batch) = if incoming_is_left {
        ("right", right_batch, "left", left_batch)
    } else {
        ("left", left_batch, "right", right_batch)
    };
    operator
        .process_data(first, first_batch, &context, &mut collector)
        .await
        .unwrap();
    assert_reference(&mut operator, probe, &probe_batch, &context).await;
}

#[tokio::test]
async fn test_native_key_types_match_sql_order_in_both_directions() {
    for key in key_cases() {
        for incoming_is_left in [false, true] {
            compare_key_case(&key, incoming_is_left).await;
        }
    }
    assert_native_arena_all_key_types_preserve_exact_v1_frames().await;
}

fn composite_batch(schema: &SchemaRef, first: &[&str], second: &[&str]) -> Batch {
    let record = RecordBatch::try_new(
        Arc::clone(schema),
        vec![
            Arc::new(StringArray::from(first.to_vec())),
            Arc::new(TimestampMicrosecondArray::from(vec![3, 0, 0, 2]).with_timezone("UTC")),
            Arc::new(StringArray::from(second.to_vec())),
        ],
    )
    .unwrap();
    Batch::table(vec![record], BatchMetadata::default()).unwrap()
}

#[tokio::test]
async fn test_composite_native_framing_distinguishes_equal_concatenations() {
    let left = typed_schema(&left_schema(), &DataType::Utf8);
    let left = Arc::new(Schema::new(
        left.fields()
            .iter()
            .enumerate()
            .map(|(index, field)| {
                if index == 2 {
                    field.as_ref().clone().with_data_type(DataType::Utf8)
                } else {
                    field.as_ref().clone()
                }
            })
            .collect::<Vec<_>>(),
    ));
    let right = typed_schema(&right_schema(), &DataType::Utf8);
    let mut declaration = spec();
    declaration.left_keys.push("amount".to_owned());
    declaration.right_keys.push("status".to_owned());
    let mut operator =
        StreamJoinOperator::new("match", left.clone(), right.clone(), declaration).unwrap();
    let job = job();
    let context = StreamOperatorContext::new(&job, "match", None);
    let mut collector = EdgeCollector::new(operator.output_ports().to_vec());
    operator
        .process_data(
            "left",
            composite_batch(&left, &["ab", "a", "é", ""], &["c", "bc", "", "é"]),
            &context,
            &mut collector,
        )
        .await
        .unwrap();
    let probe = composite_batch(&right, &["a", "ab", "", "é"], &["bc", "c", "é", ""]);
    assert_reference(&mut operator, "right", &probe, &context).await;
    let (plan, admitted) = admitted_probe(&mut operator, "right", &probe, &context).await;
    assert_eq!(
        operator
            .native_matches(&plan, &admitted)
            .unwrap()
            .unwrap()
            .pairs
            .len(),
        4
    );
}

#[tokio::test]
async fn test_native_index_is_ready_before_the_first_probe() {
    let mut operator =
        StreamJoinOperator::new("match", left_schema(), right_schema(), spec()).unwrap();
    let job = job();
    let context = StreamOperatorContext::new(&job, "match", None);
    let mut collector = EdgeCollector::new(operator.output_ports().to_vec());
    operator
        .process_data(
            "left",
            left_batch(vec![3, 0, 0, 2]),
            &context,
            &mut collector,
        )
        .await
        .unwrap();
    assert!(
        operator.state.left.1.is_some(),
        "fresh eligible retention must prepare its index before the first query"
    );
    let pool = operator
        .runtime
        .runtime()
        .unwrap()
        .incremental_memory_pool();
    assert!(index_funding(&operator) > 1_024);
    assert_eq!(pool.reserved(), state_funding(&operator));
    drop(operator);
    assert_eq!(pool.reserved(), 0);
}

#[tokio::test]
async fn test_native_probe_interns_each_distinct_repeated_key_once() {
    let mut operator =
        StreamJoinOperator::new("match", left_schema(), right_schema(), spec()).unwrap();
    let job = job();
    let context = StreamOperatorContext::new(&job, "match", None);
    let mut collector = EdgeCollector::new(operator.output_ports().to_vec());
    operator
        .process_data("left", left_batch(vec![0, 1]), &context, &mut collector)
        .await
        .unwrap();
    let source = keyed_right_batch(&[7, 8, 7, 7, 8, 7], &[2, 2, 1, 0, 1, 2]);
    let (plan, admitted) = admitted_probe(&mut operator, "right", &source, &context).await;
    reset_join_work();
    let native = operator.native_matches(&plan, &admitted).unwrap().unwrap();
    assert_eq!(join_work().sql_probe_table_builds, 0);
    assert_eq!(join_work().key_encodings, 2);
    assert_eq!(native.pairs.len(), 8);
    for position in [2, 3, 5] {
        assert!(Arc::ptr_eq(native.keys.key(0), native.keys.key(position)));
    }
    assert!(Arc::ptr_eq(native.keys.key(1), native.keys.key(4)));
    assert!(!Arc::ptr_eq(native.keys.key(0), native.keys.key(1)));
    let pool = operator
        .runtime
        .runtime()
        .unwrap()
        .incremental_memory_pool();
    drop((native, admitted, operator));
    drop(context);
    assert!(job.gather_owner().close_and_drain().await.is_empty());
    drop(job);
    assert_eq!(pool.reserved(), 0);
}

#[tokio::test]
async fn test_native_probe_key_allocations_and_aliases_remain_funded() {
    key_arena::tests::assert_empty_arena_sizing_for_wide_schema_allocates_nothing();
    let mut cases: Vec<ArrayRef> = vec![
        Arc::new(UInt8Array::from(vec![0])),
        Arc::new(StringArray::from(vec![""])),
        Arc::new(UInt8Array::from(vec![7; 129])),
        Arc::new(UInt8Array::from((0..=128).collect::<Vec<u8>>())),
        Arc::new(StringArray::from(vec![""; 129])),
        Arc::new(StringArray::from(
            (0..129).map(|row| format!("key-{row}")).collect::<Vec<_>>(),
        )),
        Arc::new(StringArray::from(vec!["", "a", "", "b", "a"])),
        Arc::new(StringArray::from(
            (0..8)
                .map(|row| format!("{row}{}", "é\0".repeat(1366)))
                .collect::<Vec<_>>(),
        )),
    ];
    for rows in [2, 3, 4, 8, 9, 17, 33, 65] {
        cases.push(Arc::new(UInt8Array::from((0..rows).collect::<Vec<u8>>())));
    }
    cases.extend(key_cases());
    for values in cases {
        assert_native_probe_key_funding(values).await;
    }
    assert_native_arena_composite_frames_use_shared_rowed_and_legacy_offsets().await;
}

async fn assert_native_arena_all_key_types_preserve_exact_v1_frames() {
    for values in key_cases() {
        let left = typed_schema(&left_schema(), values.data_type());
        let right = typed_schema(&right_schema(), values.data_type());
        let source = typed_batch(&right_batch(vec![0; values.len()]), &right, &values);
        let mut operator = StreamJoinOperator::new("match", left, right, spec()).unwrap();
        let job = job();
        let context = StreamOperatorContext::new(&job, "match", None);
        let (plan, admitted) = admitted_probe(&mut operator, "right", &source, &context).await;
        let expected = admitted
            .iter()
            .map(|row| {
                encode_join_key_columns_v1(
                    row.record.columns(),
                    row.record.offset(),
                    &plan.key_indices,
                )
                .unwrap()
            })
            .collect::<Vec<_>>();
        assert_eq!(
            key_arena::ArenaLayout::measure(&admitted, &plan.key_indices)
                .unwrap()
                .unwrap()
                .charge,
            native_lookup::probe_key_charge(&admitted, &plan.key_indices).unwrap(),
        );
        reset_join_work();
        let keys = operator
            .native_probe_keys(&plan, &admitted)
            .unwrap()
            .unwrap();
        assert_eq!(
            keys.row_keys().map(|key| key.to_vec()).collect::<Vec<_>>(),
            expected
        );
        assert_eq!(
            join_work().arena_frames,
            values.len(),
            "{}",
            values.data_type()
        );
        assert_eq!(join_work().borrowed_key_hashes, 0);
        assert_eq!(join_work().borrowed_key_equalities, 0);
        assert!(join_work().key_type_resolutions <= 2);
        let pool = operator
            .runtime
            .runtime()
            .unwrap()
            .incremental_memory_pool();
        drop((keys, admitted, operator));
        drop(context);
        assert!(job.gather_owner().close_and_drain().await.is_empty());
        drop(job);
        assert_eq!(pool.reserved(), 0);
    }
}

async fn assert_native_arena_composite_frames_use_shared_rowed_and_legacy_offsets() {
    let left = typed_schema(&left_schema(), &DataType::Utf8);
    let right = typed_schema(&right_schema(), &DataType::Utf8);
    let record = RecordBatch::try_new(
        Arc::clone(&right),
        vec![
            Arc::new(StringArray::from(vec![
                "unused", "a", "ab", "a", "\0", "unused",
            ])),
            Arc::new(TimestampMicrosecondArray::from(vec![0; 6]).with_timezone("UTC")),
            Arc::new(StringArray::from(vec![
                "unused", "bc", "c", "bc", "é", "unused",
            ])),
        ],
    )
    .unwrap()
    .slice(1, 4);
    let source = Batch::table(vec![record.clone()], BatchMetadata::default()).unwrap();
    let mut operator = StreamJoinOperator::new("match", left, right, spec()).unwrap();
    let job = job();
    let context = StreamOperatorContext::new(&job, "match", None);
    let (mut plan, mut admitted) = admitted_probe(&mut operator, "right", &source, &context).await;
    assert!(
        admitted
            .iter()
            .all(|row| matches!(row.record, columnar::RowPayload::Shared { .. }))
    );
    plan.key_indices = vec![0, 2];
    admitted.push(AdmittedRow {
        record: columnar::RowPayload::Rowed {
            parent: Arc::new(record.slice(1, 2)),
            row: 1,
        },
        event_time: EventTime::from_micros(0),
        row_id: 4,
        retain: false,
    });
    admitted.push(AdmittedRow {
        record: record.slice(3, 1).into(),
        event_time: EventTime::from_micros(0),
        row_id: 5,
        retain: false,
    });
    let expected = admitted
        .iter()
        .map(|row| {
            encode_join_key_columns_v1(row.record.columns(), row.record.offset(), &plan.key_indices)
                .unwrap()
        })
        .collect::<Vec<_>>();
    reset_join_work();
    let layout = key_arena::ArenaLayout::measure(&admitted, &plan.key_indices)
        .unwrap()
        .unwrap();
    assert!(layout.fits(admitted.len(), plan.key_indices.len()));
    let mut keys = None;
    let measured = allocation_counter::measure(|| {
        keys = operator.native_probe_keys(&plan, &admitted).unwrap();
    });
    let keys = keys.unwrap();
    assert!(
        measured.bytes_max
            <= u64::try_from(layout.peak(admitted.len(), plan.key_indices.len()).unwrap()).unwrap()
    );
    assert_eq!(
        keys.row_keys().map(|key| key.to_vec()).collect::<Vec<_>>(),
        expected
    );
    assert_eq!(keys.keys.len(), 3);
    assert!(Arc::ptr_eq(keys.key(0), keys.key(2)));
    assert!(Arc::ptr_eq(keys.key(0), keys.key(4)));
    assert!(Arc::ptr_eq(keys.key(3), keys.key(5)));
    assert_eq!(join_work().arena_frames, admitted.len());
    assert_eq!(join_work().borrowed_key_hashes, 0);
    let pool = operator
        .runtime
        .runtime()
        .unwrap()
        .incremental_memory_pool();
    drop((keys, admitted, operator));
    drop(context);
    assert!(job.gather_owner().close_and_drain().await.is_empty());
    drop(job);
    assert_eq!(pool.reserved(), 0);
}

async fn assert_native_probe_key_funding(values: ArrayRef) {
    let left = typed_schema(&left_schema(), values.data_type());
    let right = typed_schema(&right_schema(), values.data_type());
    let source = typed_batch(&right_batch(vec![0; values.len()]), &right, &values);
    let mut operator = StreamJoinOperator::new("match", left, right, spec()).unwrap();
    let job = job();
    let context = StreamOperatorContext::new(&job, "match", None);
    let (plan, admitted) = admitted_probe(&mut operator, "right", &source, &context).await;
    let pool = operator
        .runtime
        .runtime()
        .unwrap()
        .incremental_memory_pool();
    let baseline = pool.reserved();
    let mut layout = None;
    let sizing = allocation_counter::measure(|| {
        layout = key_arena::ArenaLayout::measure(&admitted, &plan.key_indices).unwrap();
    });
    assert_eq!(
        sizing.bytes_max, 0,
        "sizing precedes optional credit admission"
    );
    let layout = layout.unwrap();
    assert_eq!(
        layout.charge,
        native_lookup::probe_key_charge(&admitted, &plan.key_indices).unwrap()
    );
    let mut keys = None;
    let measured = allocation_counter::measure(|| {
        keys = operator.native_probe_keys(&plan, &admitted).unwrap();
    });
    let keys = keys.expect("eligible key fixtures must take the native path");
    let paid = pool.reserved() - baseline;
    assert_eq!(
        paid, layout.charge,
        "native admission keeps the exact previous P threshold"
    );
    if layout.fits(admitted.len(), plan.key_indices.len()) {
        assert!(
            measured.bytes_max
                <= u64::try_from(layout.peak(admitted.len(), plan.key_indices.len()).unwrap())
                    .unwrap()
        );
    }
    let mut live = measured.bytes_current;
    assert!(live > 0);
    assert!(
        measured.bytes_max <= u64::try_from(paid).unwrap(),
        "type={}, rows={}, peak={}, funding={paid}",
        values.data_type(),
        values.len(),
        measured.bytes_max
    );
    assert!(usize::try_from(live).unwrap() <= paid);
    let first_alias = Arc::clone(keys.key(0));
    let last_alias = Arc::clone(keys.key(admitted.len() - 1));
    assert_eq!(first_alias.funded_owner().unwrap().1, paid);
    assert_eq!(last_alias.funded_owner().unwrap().1, paid);
    let released = allocation_counter::measure(|| drop(keys));
    live += released.bytes_current;
    assert!(live > 0);
    assert!(usize::try_from(live).unwrap() <= paid);
    assert_eq!(pool.reserved(), baseline + paid);
    let released = allocation_counter::measure(|| drop((first_alias, last_alias)));
    assert_eq!(live + released.bytes_current, 0);
    assert_eq!(pool.reserved(), baseline);
    drop((admitted, operator));
    drop(context);
    assert!(job.gather_owner().close_and_drain().await.is_empty());
    drop(job);
    assert_eq!(pool.reserved(), 0);
}

#[test]
fn test_native_index_updates_out_of_order_append_and_dense_eviction() {
    let service = crate::runtime::streaming::gather_work::TestService::new(2, 1).unwrap();
    let runtime = tokio::runtime::Builder::new_current_thread()
        .enable_all()
        .build()
        .unwrap();
    runtime.block_on(assert_native_index_updates_out_of_order_append_and_dense_eviction());
    runtime.block_on(ordered_probe_tests::assert_bounded_order_and_global_limit(
        &service,
    ));
    drop(runtime);
    service.shutdown();
}

async fn assert_native_index_updates_out_of_order_append_and_dense_eviction() {
    let mut operator =
        StreamJoinOperator::new("match", left_schema(), right_schema(), spec()).unwrap();
    let job = job();
    let context = StreamOperatorContext::new(&job, "match", None);
    let mut collector = EdgeCollector::new(operator.output_ports().to_vec());
    operator
        .process_data(
            "left",
            left_batch(vec![3, 0, 4, 1, 5, 2]),
            &context,
            &mut collector,
        )
        .await
        .unwrap();
    assert_reference(&mut operator, "right", &right_batch(vec![1, 2]), &context).await;
    operator
        .process_data("left", left_batch(vec![1, 6, -1]), &context, &mut collector)
        .await
        .unwrap();
    assert_reference(&mut operator, "right", &right_batch(vec![2, 1]), &context).await;
    let progress = progress_context(
        &job,
        (IngressState::Active, None),
        (IngressState::Active, Some(60_000_001)),
    );
    operator
        .on_ingress_progress("right", &progress)
        .await
        .unwrap();
    assert_eq!(operator.state.left.len(), 7);
    assert_reference(
        &mut operator,
        "right",
        &right_batch(vec![60_000_001]),
        &progress,
    )
    .await;
}

#[test]
fn test_native_index_budget_denial_keeps_state_and_refunds_append() {
    checkpoint_compaction_tests::isolated_checkpoint_test(|service, runtime| {
        runtime.block_on(native_index_budget_denial(service));
    });
}

async fn native_index_budget_denial(service: &crate::runtime::streaming::gather_work::TestService) {
    let mut operator =
        StreamJoinOperator::new("match", left_schema(), right_schema(), spec()).unwrap();
    let job = job().with_gather_owner(service.owner("native-index-refund".into()));
    let context = StreamOperatorContext::new(&job, "match", None);
    let mut collector = EdgeCollector::new(operator.output_ports().to_vec());
    operator
        .process_data("left", left_batch(vec![0, 1, 2]), &context, &mut collector)
        .await
        .unwrap();
    let captured = operator.checkpoint_v1(Epoch::INITIAL).unwrap();
    operator.restore(&captured).unwrap();
    let (plan, admitted) =
        admitted_probe(&mut operator, "right", &right_batch(vec![1]), &context).await;
    let admitted_paid = payload_funding(admitted.iter().map(|row| &row.record));
    let before = operator.status();
    let runtime = operator.runtime.runtime().unwrap();
    let pool = runtime.incremental_memory_pool();
    let pressure = runtime.incremental_reservation("native-index-pressure");
    pressure.try_grow((1 << 30) - pool.reserved()).unwrap();
    assert!(operator.native_matches(&plan, &admitted).unwrap().is_none());
    reset_join_work();
    let (matches, admitted) = operator.evaluate_matches(&plan, admitted, &context).await;
    let failure = matches.err().unwrap();
    assert!(matches!(failure, CalcFlowError::DataFusion { .. }));
    assert_eq!(join_work().sql_probe_table_builds, 1);
    assert!(operator.state.left.1.is_none());
    assert_eq!(operator.status(), before);
    assert_eq!(pool.reserved(), 1 << 30);
    drop(pressure);
    assert_resident_and_gather_funding(pool.as_ref(), &job, admitted_paid);
    let native = operator.native_matches(&plan, &admitted).unwrap().unwrap();
    drop(native);
    assert!(index_funding(&operator) > 1_024);
    let paid = assert_resident_and_gather_funding(
        pool.as_ref(),
        &job,
        state_funding(&operator) + admitted_paid,
    );
    let index_paid = index_funding(&operator);
    let append = operator.reserve_native_append(true, 5).unwrap().unwrap();
    let prepared = operator.state.left.1.as_ref().unwrap().funded_bytes();
    assert!(prepared > index_paid);
    assert_eq!(pool.reserved(), paid + prepared - index_paid);
    drop(append);
    assert_eq!(pool.reserved(), paid);
    operator.reset().unwrap();
    assert_resident_and_gather_funding(pool.as_ref(), &job, admitted_paid);
    drop(admitted);
    assert_resident_and_gather_funding(pool.as_ref(), &job, 0);
    drop(context);
    drop(operator);
    assert!(job.gather_owner().close_and_drain().await.is_empty());
    let (home, generation, attempt) = job.gather_owner().funding();
    assert_eq!((generation, attempt), (0, 0));
    assert_eq!(pool.reserved(), home);
    drop(job);
    assert_eq!(pool.reserved(), 0);
}

#[test]
fn test_empty_native_result_funds_key_vector_until_actual_drop() {
    checkpoint_compaction_tests::isolated_checkpoint_test(|service, runtime| {
        runtime.block_on(owned_probe_tests::assert_owned_process_data(service));
        runtime.block_on(empty_native_result_credit(service));
    });
}

async fn empty_native_result_credit(service: &crate::runtime::streaming::gather_work::TestService) {
    let mut operator =
        StreamJoinOperator::new("match", left_schema(), right_schema(), spec()).unwrap();
    let job = job().with_gather_owner(service.owner("native-key-vector-refund".into()));
    let context = StreamOperatorContext::new(&job, "match", None);
    let mut collector = EdgeCollector::new(operator.output_ports().to_vec());
    operator
        .process_data("left", left_batch(vec![0, 1, 2]), &context, &mut collector)
        .await
        .unwrap();
    let (plan, admitted) = admitted_probe(
        &mut operator,
        "right",
        &keyed_right_batch(&[99, 99, 100, 99], &[1, 1, 1, 1]),
        &context,
    )
    .await;
    reset_join_work();
    let mut native = operator.native_matches(&plan, &admitted).unwrap().unwrap();
    assert!(native.pairs.is_empty());
    assert_eq!(join_work().key_encodings, 2);
    assert_eq!(join_work().sql_probe_table_builds, 0);
    assert!(Arc::ptr_eq(native.keys.key(0), native.keys.key(1)));
    assert!(Arc::ptr_eq(native.keys.key(0), native.keys.key(3)));
    assert!(!Arc::ptr_eq(native.keys.key(0), native.keys.key(2)));
    let pool = operator
        .runtime
        .runtime()
        .unwrap()
        .incremental_memory_pool();
    let paid = pool.reserved();
    let key_capacity = native.keys.keys.capacity();
    let admitted_paid = payload_funding(admitted.iter().map(|row| &row.record));
    let state_paid = state_funding(&operator);
    let (home, generation, attempt) = job.gather_owner().funding();
    assert_eq!(attempt, 0);
    assert!(paid > state_paid + admitted_paid + home + generation);
    native.keys.keys.clear();
    assert_eq!(native.keys.keys.capacity(), key_capacity);
    assert_eq!(
        pool.reserved(),
        paid,
        "live vector backing remains funded after its key owners drop"
    );
    drop(native);
    assert_resident_and_gather_funding(pool.as_ref(), &job, state_paid + admitted_paid);
    operator.reset().unwrap();
    assert_resident_and_gather_funding(pool.as_ref(), &job, admitted_paid);
    drop(admitted);
    assert_resident_and_gather_funding(pool.as_ref(), &job, 0);
    drop(context);
    drop(operator);
    assert!(job.gather_owner().close_and_drain().await.is_empty());
    let (home, generation, attempt) = job.gather_owner().funding();
    assert_eq!((generation, attempt), (0, 0));
    assert_eq!(pool.reserved(), home);
    drop(job);
    assert_eq!(pool.reserved(), 0);
}

#[test]
fn test_native_fallback_refunds_unused_index_before_legacy_sql() {
    checkpoint_compaction_tests::isolated_checkpoint_test(|service, runtime| {
        runtime.block_on(owned_probe_tests::assert_refused_process_data(service));
        runtime.block_on(native_fallback_index_refund(service));
    });
}

async fn native_fallback_index_refund(
    service: &crate::runtime::streaming::gather_work::TestService,
) {
    let mut operator =
        StreamJoinOperator::new("match", left_schema(), right_schema(), spec()).unwrap();
    let job = job().with_gather_owner(service.owner("native-fallback-index".into()));
    let context = StreamOperatorContext::new(&job, "match", None);
    let mut collector = EdgeCollector::new(operator.output_ports().to_vec());
    operator
        .process_data("left", left_batch(vec![0, 1, 2]), &context, &mut collector)
        .await
        .unwrap();
    let captured = operator.checkpoint_v1(Epoch::INITIAL).unwrap();
    operator.restore(&captured).unwrap();
    let (plan, admitted) =
        admitted_probe(&mut operator, "right", &right_batch(vec![1]), &context).await;
    let before = operator.status();
    let runtime = operator.runtime.runtime().unwrap();
    let pool = runtime.incremental_memory_pool();
    let headroom = native_lookup::NativeIndex::build_charge(operator.state.left.len()).unwrap();
    let pressure = runtime.incremental_reservation("native-probe-refusal");
    pressure
        .try_grow((1 << 30) - pool.reserved() - headroom)
        .unwrap();
    reset_join_work();
    let (reference, admitted) = operator.legacy_matches(&plan, admitted, &context).await;
    let reference = reference.expect("original SQL must accept the identical pressure budget");
    let reference_builds = join_work().sql_probe_table_builds;
    assert!(reference_builds > 0);
    operator.discard_paid_key_cache(&plan);
    assert!(operator.state.left.1.is_none());
    assert_eq!(pool.reserved(), (1 << 30) - headroom);
    reset_join_work();
    let (actual, admitted) = operator.evaluate_matches(&plan, admitted, &context).await;
    let actual = actual.expect("declined native scratch must refund unused index before SQL");
    let pairs = |rows: &[MatchedPair]| {
        rows.iter()
            .map(|pair| (pair.pos, pair.opposite_index))
            .collect::<Vec<_>>()
    };
    assert_eq!(pairs(&actual.pairs), pairs(&reference));
    assert_eq!(join_work().sql_probe_table_builds, reference_builds);
    assert!(operator.state.left.1.is_none());
    assert_eq!(operator.status(), before);
    drop(actual);
    operator.discard_paid_key_cache(&plan);
    assert_eq!(pool.reserved(), (1 << 30) - headroom);
    drop(pressure);
    drop(admitted);
    drop(context);
    drop(operator);
    assert!(job.gather_owner().close_and_drain().await.is_empty());
    drop(job);
    assert_eq!(pool.reserved(), 0);
}
