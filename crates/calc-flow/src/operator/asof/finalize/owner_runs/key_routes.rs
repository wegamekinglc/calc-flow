//! Chunk-local match routing must agree with binary search and rowwise ownership.

use super::*;

fn install_right(operator: &mut StreamAsofJoinOperator) {
    let rows = operator
        .state
        .left
        .iter()
        .map(|((time, key, sequence), row)| ((*time, key.clone(), sequence.into_owned()), row))
        .collect::<Vec<_>>();
    let mut buckets = BTreeMap::new();
    for (index, ((time, key, sequence), left)) in rows.into_iter().enumerate() {
        let bucket = buckets.entry(key).or_insert_with(state::RightBucket::new);
        if index % 4 != 0 {
            let left_batch = operator.state.batches.view(left).batch;
            let owner = Arc::new(state::PayloadBatch {
                key: (1, left_batch.key.1),
                record: left_batch.record.clone(),
                encoded: OnceLock::new(),
                encoded_charge_bytes: 0,
                body_bytes: 0,
            });
            let right = operator.state.attach(&state::RowPayload {
                batch: owner,
                row: left.row as usize,
            });
            bucket.insert((time, sequence.clone()), Some(right));
        }
        if index % 7 == 0 {
            bucket.insert((time + 1, sequence), None);
        }
    }
    for (key, bucket) in buckets {
        operator.state.right.insert(key, bucket);
    }
    operator.state.rebuild_encoding_owners();
}

async fn assert_cut_matches_binary_search(operator: &StreamAsofJoinOperator, count: usize) {
    let job = StreamJobContext::new(1, "asof", JsonMap::new(), None, CancellationToken::new());
    let context = StreamOperatorContext::new(&job, "asof", None);
    let pool: Arc<dyn MemoryPool> = Arc::new(GreedyMemoryPool::new(1 << 24));
    let mut credit = MemoryConsumer::new("routed-output-test").register(&pool);
    let mut reference_credit = MemoryConsumer::new("binary-output-test").register(&pool);
    let mut plan = OutputPlanBuilder::new(count, None, &mut credit, "asof").unwrap();
    let mut reference = OutputPlanBuilder::new(count, None, &mut reference_credit, "asof").unwrap();
    let prefix = monotonic_candidate_rows(operator, count, &context, &mut plan, &mut credit)
        .await
        .unwrap();
    let reference_prefix = binary_search_candidate_rows(
        operator,
        count,
        &context,
        &mut reference,
        &mut reference_credit,
    )
    .await
    .unwrap();
    assert_prefix(&prefix, &rowwise_prefix(operator, count));
    assert_prefix(&prefix, &reference_prefix);
    assert_projected_inventory_and_journal(operator, &prefix, &reference_prefix);
    let plan = plan
        .finish(operator.physical_schema(1), &mut credit, "asof")
        .unwrap();
    let reference = reference
        .finish(operator.physical_schema(1), &mut reference_credit, "asof")
        .unwrap();
    assert_eq!(plan.len, count);
    assert_eq!(plan.matched, reference.matched);
    if count == operator.state.left.len() {
        assert!(plan.matched > 0 && plan.matched < count as u64);
    }
    assert_eq!(plan.raw_bytes, reference.raw_bytes);
    for (actual, expected) in [
        (&plan.left, &reference.left),
        (&plan.right, &reference.right),
    ] {
        assert_eq!(actual.positions, expected.positions);
        assert_eq!(actual.has_nulls, expected.has_nulls);
        assert_eq!(actual.batches, expected.batches);
        assert_eq!(
            actual
                .spans
                .iter()
                .map(|s| (s.source, s.start, s.end))
                .collect::<Vec<_>>(),
            expected
                .spans
                .iter()
                .map(|s| (s.source, s.start, s.end))
                .collect::<Vec<_>>()
        );
    }
    assert_eq!(credit.size(), reference_credit.size());
}

fn keyed_operator(keys: usize, overlap: bool) -> StreamAsofJoinOperator {
    let count = 520;
    let record = RecordBatch::try_new(
        Arc::new(Schema::new(vec![
            Field::new("key", DataType::Utf8, false),
            Field::new(
                "time",
                DataType::Timestamp(TimeUnit::Microsecond, Some("UTC".into())),
                false,
            ),
            Field::new("seq", DataType::Utf8, false),
        ])),
        vec![
            Arc::new(StringArray::from_iter_values((0..count).map(|i| {
                if keys == 1 {
                    "one shared output key".into()
                } else if i == 0 {
                    "retired key".into()
                } else {
                    format!("shared output key {}", (i - 1) % (keys - 1))
                }
            }))),
            Arc::new(
                TimestampMicrosecondArray::from_iter_values((0..count).map(|i| {
                    i64::try_from(if overlap {
                        (i % (count / 2)) * 2 + i / (count / 2)
                    } else {
                        i
                    })
                    .unwrap()
                }))
                .with_timezone("UTC"),
            ),
            Arc::new(StringArray::from_iter_values(
                (0..count).map(|i| format!("canonical sequence {i:04}")),
            )),
        ],
    )
    .unwrap();
    let mut operator = operator_for(
        &record,
        &["key".into()],
        &["seq".into()],
        &[count / 2..count, 0..count / 2],
    );
    operator.spec = StreamAsofJoinSpec::new(
        operator.spec.left().clone(),
        operator.spec.right().clone(),
        Duration::from_micros(3),
        AsofStateLimits::new(100_000, 100_000_000).unwrap(),
    )
    .unwrap();
    install_right(&mut operator);
    operator
}

async fn routed_and_fallback_cuts_preserve_matches_after_partial_consumption() {
    for keys in [1, 63, 64, 65] {
        for overlap in [false, true] {
            let mut operator = keyed_operator(keys, overlap);
            for consumed in [0, 1, 17] {
                operator.state.commit_left_prefix(consumed);
                let remaining = operator.state.left.len();
                for selected in [1, 17, remaining] {
                    assert_cut_matches_binary_search(&operator, selected).await;
                }
            }
        }
    }
}

async fn legacy_output_keeps_generic_matching() {
    let mut operator = keyed_operator(3, false);
    let rows = operator
        .state
        .left
        .iter()
        .map(|((time, key, sequence), row)| ((*time, key.clone(), sequence.into_owned()), row))
        .collect::<Vec<_>>();
    operator.state.left = state::LeftState::default();
    for (order, row) in rows {
        operator.state.left.insert(order, row);
    }
    assert_cut_matches_binary_search(&operator, operator.state.left.len()).await;
}

async fn generic_cursors_catch_up_between_routed_runs() {
    let times = (0..520).map(|i| i * 2).chain([401, 403]);
    let record = RecordBatch::try_new(
        Arc::new(Schema::new(vec![
            Field::new("key", DataType::Utf8, false),
            Field::new(
                "time",
                DataType::Timestamp(TimeUnit::Microsecond, Some("UTC".into())),
                false,
            ),
            Field::new("seq", DataType::Utf8, false),
        ])),
        vec![
            Arc::new(StringArray::from(vec!["one shared output key"; 522])),
            Arc::new(TimestampMicrosecondArray::from_iter_values(times).with_timezone("UTC")),
            Arc::new(StringArray::from_iter_values(
                (0..522).map(|i| format!("sequence {i:04}")),
            )),
        ],
    )
    .unwrap();
    let mut operator = operator_for(
        &record,
        &["key".into()],
        &["seq".into()],
        &[520..522, 0..520],
    );
    install_right(&mut operator);
    let routes = operator
        .state
        .left
        .output_runs()
        .map(|run| (run.len(), run.keyed_rows(run.len()).is_some()))
        .collect::<Vec<_>>();
    assert_eq!(
        routes,
        vec![(201, true), (1, false), (1, false), (1, false), (318, true)]
    );
    assert_cut_matches_binary_search(&operator, 522).await;
}

async fn monotonic_matching_hashes_chunk_keys_instead_of_each_row() {
    let count = 1_024;
    let key = Arc::new(StringArray::from(vec!["shared output key"; count])) as ArrayRef;
    let sequence = Arc::new(Int64Array::from_iter_values(
        0..i64::try_from(count).unwrap(),
    )) as ArrayRef;
    let mut operator = operator(key, sequence, std::slice::from_ref(&(0..count)));
    let key = operator
        .state
        .left
        .output_iter()
        .next()
        .unwrap()
        .0
        .1
        .clone();
    operator.state.right.insert(key, state::RightBucket::new());
    let expected = rowwise_prefix(&operator, count);
    let job = StreamJobContext::new(1, "asof", JsonMap::new(), None, CancellationToken::new());
    let context = StreamOperatorContext::new(&job, "asof", None);
    let pool: Arc<dyn MemoryPool> = Arc::new(GreedyMemoryPool::new(1 << 24));
    let mut credit = MemoryConsumer::new("output-key-routing-test").register(&pool);
    let mut plan = OutputPlanBuilder::new(count, None, &mut credit, "asof").unwrap();
    state::take_encoding_hashes();
    let prefix = monotonic_candidate_rows(&operator, count, &context, &mut plan, &mut credit)
        .await
        .unwrap();
    let hashes = state::take_encoding_hashes();
    assert_prefix(&prefix, &expected);
    let plan = plan
        .finish(operator.physical_schema(1), &mut credit, "asof")
        .unwrap();
    assert_eq!(plan.len, count);
    assert_eq!(plan.matched, 0);
    assert_eq!(plan.right.positions, vec![(0, 0); count]);
    assert!(hashes <= 24, "one-key output prefix hashed {hashes} times");
}

pub(super) async fn check_matching_routes() {
    monotonic_matching_hashes_chunk_keys_instead_of_each_row().await;
    routed_and_fallback_cuts_preserve_matches_after_partial_consumption().await;
    legacy_output_keeps_generic_matching().await;
    generic_cursors_catch_up_between_routed_runs().await;
}
