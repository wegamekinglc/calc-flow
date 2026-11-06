use super::*;
use datafusion::arrow::array::Array;
use datafusion::execution::memory_pool::{MemoryPool, UnboundedMemoryPool};

#[tokio::test]
async fn test_grouped_candidate_ranks_merge_without_integer_slot_lookups() {
    let mut observed = Vec::new();
    for single in [true, false] {
        let fixture = allocation_fixture(single).await;
        let (credit, workspace, mut candidates) = fixture
            .plan
            .input_candidates(GROUPS, "candidate-ranks")
            .unwrap();
        let batch = reversed_input(fixture.plan.schema.clone());
        fixture
            .plan
            .update_chunk(
                batch.table_payload().unwrap().batches().first().unwrap(),
                &mut candidates,
                &credit,
                workspace,
                "candidate-ranks",
            )
            .unwrap();
        let mut partial = candidates.partial.take().unwrap();
        assert_eq!(partial.slots, (0..GROUPS).rev().collect::<Vec<_>>());
        let job = StreamJobContext::new(
            975,
            "candidate-ranks",
            JsonMap::new(),
            None,
            CancellationToken::new(),
        );
        let context = StreamOperatorContext::new(&job, "candidate-ranks", None);
        partial
            .merge(
                &mut candidates.groups,
                &fixture.plan.groups,
                &context,
                "candidate-ranks",
            )
            .await
            .unwrap();
        assert!(candidates.groups.values().all(|candidate| {
            candidate.group.aggregates.result(0).as_ref() == &ScalarValue::Int64(Some(2))
        }));
        observed.push(partial.candidate_slot_lookups);
    }
    eprintln!(
        "Candidate merge integer-slot lookups: inline={}, mixed={}",
        observed[0], observed[1]
    );
    assert_eq!(
        observed[1], GROUPS,
        "generic positive control must use its slot map"
    );
    assert_eq!(
        observed[0], 0,
        "dense partial ranks must select inline candidates directly"
    );
}

fn reversed_input(schema: SchemaRef) -> Batch {
    let record = RecordBatch::try_new(
        schema,
        vec![
            Arc::new(StringArray::from_iter_values(
                (0..GROUPS).rev().map(|group| format!("account-{group}")),
            )),
            Arc::new(Int64Array::from(vec![Some(1); GROUPS])),
        ],
    )
    .unwrap();
    Batch::table(vec![record], crate::BatchMetadata::default()).unwrap()
}

struct CostFixture {
    runtime: DataFusionRuntime,
    plan: IncrementalSql,
    pool: Arc<crate::state_memory_pool::OwnedStatePool>,
    input: Batch,
}

async fn cost_fixture() -> CostFixture {
    let pool = Arc::new(crate::state_memory_pool::OwnedStatePool::new(
        Arc::new(UnboundedMemoryPool::default()),
        "candidate probe",
        "candidate probe limit",
    ));
    let runtime =
        DataFusionRuntime::with_benchmark_pool(DataFusionConfig::default(), pool.clone()).unwrap();
    let seed = allocation_input();
    let query = crate::expression::parse_select_query(
        "SELECT key, SUM(value) AS total FROM events GROUP BY key",
    )
    .unwrap();
    let mut plan = IncrementalSql::plan(
        &runtime,
        &query,
        "events",
        seed.table_payload().unwrap().schema().clone(),
        "candidate-cost",
    )
    .await
    .unwrap()
    .unwrap();
    let job = StreamJobContext::new(
        976,
        "candidate-cost",
        JsonMap::new(),
        None,
        CancellationToken::new(),
    );
    let context = StreamOperatorContext::new(&job, "candidate-cost", None);
    let transaction = plan
        .update(&seed, &context, "candidate-cost")
        .await
        .unwrap();
    plan.commit(transaction);
    let input = cost_input(plan.schema.clone());
    CostFixture {
        runtime,
        plan,
        pool,
        input,
    }
}

fn cost_input(schema: SchemaRef) -> Batch {
    let keys = (0..=CHUNK_ROWS).map(|row| {
        let rank = row % GROUPS;
        let group = if rank.is_multiple_of(2) {
            GROUPS - rank - 1
        } else {
            GROUPS + rank
        };
        format!("account-{group}")
    });
    let record = RecordBatch::try_new(
        schema,
        vec![
            Arc::new(StringArray::from_iter_values(keys)),
            Arc::new(Int64Array::from(vec![Some(1); CHUNK_ROWS + 1])),
        ],
    )
    .unwrap();
    Batch::table(vec![record], crate::BatchMetadata::default()).unwrap()
}

fn legacy_candidates(candidates: &mut InputCandidates) {
    candidates.groups = CandidateGroups::new(false);
}

fn cost_expected() -> Vec<(String, i64)> {
    let mut totals = vec![0_i64; 2 * GROUPS];
    totals[..GROUPS].fill(1);
    for row in 0..=CHUNK_ROWS {
        let rank = row % GROUPS;
        let group = if rank.is_multiple_of(2) {
            GROUPS - rank - 1
        } else {
            GROUPS + rank
        };
        totals[group] += 1;
    }
    (0..GROUPS)
        .chain((GROUPS + 1..2 * GROUPS).step_by(2))
        .map(|group| (format!("account-{group}"), totals[group]))
        .collect()
}

fn assert_cost_output(records: &[RecordBatch], schema: &SchemaRef, expected: &[(String, i64)]) {
    let mut index = 0;
    for record in records {
        assert_eq!(record.schema(), *schema);
        let keys = record
            .column(0)
            .as_any()
            .downcast_ref::<StringArray>()
            .unwrap();
        let totals = record
            .column(1)
            .as_any()
            .downcast_ref::<Int64Array>()
            .unwrap();
        for row in 0..record.num_rows() {
            assert_eq!(keys.value(row), expected[index].0);
            assert!(!totals.is_null(row));
            assert_eq!(totals.value(row), expected[index].1);
            index += 1;
        }
    }
    assert_eq!(index, expected.len());
}

async fn cost_stage(
    fixture: &mut CostFixture,
    legacy: bool,
    context: &StreamOperatorContext<'_>,
) -> (MemoryReservation, InputCandidates) {
    let (credit, workspace, mut candidates) = fixture
        .plan
        .input_candidates(fixture.input.num_rows(), "candidate-cost")
        .unwrap();
    if legacy {
        legacy_candidates(&mut candidates);
    }
    fixture
        .plan
        .update_records(
            fixture.input.table_payload().unwrap().batches(),
            &mut candidates,
            (&credit, workspace),
            context,
            "candidate-cost",
        )
        .await
        .unwrap();
    fixture
        .plan
        .finish_candidates(&mut candidates, context, "candidate-cost")
        .await
        .unwrap();
    (credit, candidates)
}

fn cost_case(
    executor: &tokio::runtime::Runtime,
    mut fixture: CostFixture,
    legacy: bool,
) -> (
    allocation_counter::AllocationInfo,
    allocation_counter::AllocationInfo,
) {
    let job = StreamJobContext::new(
        977,
        "candidate-cost",
        JsonMap::new(),
        None,
        CancellationToken::new(),
    );
    let context = StreamOperatorContext::new(&job, "candidate-cost", None);
    let expected = cost_expected();
    let output_schema = fixture.plan.output_schema.clone();
    let baseline = fixture.pool.reserved();
    fixture.pool.take_peak();
    let mut retained = allocation_counter::AllocationInfo::default();
    let mut staged = None;
    let mut prepare_bytes = 0;
    let lifetime = allocation_counter::measure(|| {
        retained = allocation_counter::measure(|| {
            staged = Some(executor.block_on(cost_stage(&mut fixture, legacy, &context)));
        });
        let (credit, candidates) = staged.take().unwrap();
        let prepare =
            fixture
                .plan
                .prepare_transaction(candidates, &credit, &context, "candidate-cost");
        prepare_bytes = size_of_val(&prepare);
        let prepared = executor.block_on(prepare).unwrap();
        assert_cost_output(&prepared.0, &output_schema, &expected);
        drop((prepared, credit));
    });
    let paid = fixture.pool.take_peak() - baseline;
    eprintln!(
        "Candidate layout: Candidate={} map={} store={} prepare_future={prepare_bytes}",
        size_of::<Candidate>(),
        size_of::<CandidateMap>(),
        size_of::<CandidateGroups>()
    );
    eprintln!(
        "Full candidate transaction legacy={legacy}: retained={retained:?}, lifecycle={lifetime:?}, prepaid_peak={paid}"
    );
    assert!(lifetime.bytes_max <= u64::try_from(paid).unwrap());
    assert_eq!(fixture.pool.reserved(), baseline);
    drop(fixture.plan);
    assert_eq!(fixture.runtime.incremental_memory_pool().reserved(), 0);
    (retained, lifetime)
}

#[test]
fn test_grouped_candidate_ranks_full_owned_transaction_peak_and_total_bytes() {
    let executor = tokio::runtime::Builder::new_current_thread()
        .enable_all()
        .build()
        .unwrap();
    let ranked = executor.block_on(cost_fixture());
    let legacy = executor.block_on(cost_fixture());
    let (ranked_held, ranked_lifetime) = cost_case(&executor, ranked, false);
    let (legacy_held, legacy_lifetime) = cost_case(&executor, legacy, true);
    assert!(ranked_held.bytes_current < legacy_held.bytes_current);
    assert!(ranked_lifetime.bytes_total < legacy_lifetime.bytes_total);
    assert!(ranked_lifetime.bytes_max < legacy_lifetime.bytes_max);
}

#[tokio::test]
async fn test_grouped_candidate_ranks_single_growth_trace_fits_unchanged_container_credit() {
    use std::mem::MaybeUninit;

    let fixture = allocation_fixture(true).await;
    let mut groups = CandidateGroups::new(true);
    let mut partial = fixture
        .plan
        .partial_groups("candidate-growth")
        .unwrap()
        .unwrap();
    let mut legacy: HashMap<usize, MaybeUninit<Candidate>, RandomState> =
        HashMap::with_hasher(RandomState::new());
    assert!(size_of::<CandidateGroups>() <= 4096);
    for count in 1..=1025 {
        let old = groups.ranked_layout();
        let old_slots = partial.slots.capacity();
        let previous = &fixture.plan.groups[(count - 1) % GROUPS];
        let candidate = fixture
            .plan
            .candidate(Some(previous), &previous.key, None, "candidate-growth")
            .unwrap();
        partial.add_slot(2 * count, "candidate-growth").unwrap();
        groups
            .insert(2 * count, candidate, "candidate-growth")
            .unwrap();
        legacy.insert(2 * count, MaybeUninit::uninit());
        let actual = groups.ranked_layout();
        assert_eq!(actual.0, actual.2, "payload and map capacity at {count}");
        let peak = candidate_groups::growth_peak(count, old.0, "candidate-growth").unwrap();
        let paid = count * (4 * size_of::<(usize, Candidate)>() + size_of::<[usize; 8]>());
        let actual_peak = growth_overlap(old.1, actual.1)
            + growth_overlap(old.2, actual.2) * size_of::<Candidate>()
            + partial.slots.capacity() * size_of::<usize>();
        assert!(actual_peak <= peak, "actual container growth at {count}");
        assert!(peak <= paid, "unchanged candidate-only credit at {count}");
        let slot_peak = growth_overlap(old_slots, partial.slots.capacity()) * size_of::<usize>();
        assert!(slot_peak <= partial.reservation.size());
        if old.0 != actual.0 || [257, 513, 1025].contains(&count) {
            eprintln!(
                "Candidate rank growth: count={count} map={}/{} vec={}/{} partial_slots={old_slots}/{} candidate_peak={actual_peak} bounded_peak={peak} candidate_credit={paid} slot_peak={slot_peak} partial_credit={} retained_ranked={} retained_legacy={}",
                old.0,
                actual.0,
                old.2,
                actual.2,
                partial.slots.capacity(),
                partial.reservation.size(),
                actual.1
                    + actual.2 * size_of::<Candidate>()
                    + partial.slots.capacity() * size_of::<usize>(),
                legacy.allocation_size()
            );
        }
    }
    let slots = std::mem::take(&mut partial.slots);
    groups.take_slots(slots);
    drop(partial);
    assert_eq!(groups.ranked_layout().3, 2048);
    let owned = groups.into_iter().map(|(slot, _)| slot).collect::<Vec<_>>();
    assert_eq!(owned, (1..=1025).map(|count| 2 * count).collect::<Vec<_>>());
    drop((fixture.partial, fixture.plan));
    assert_eq!(fixture.runtime.incremental_memory_pool().reserved(), 0);
}

fn growth_overlap(old: usize, current: usize) -> usize {
    current + if current == old { 0 } else { old }
}

fn native_input(unsigned: bool, update: bool) -> Batch {
    let keys = if update {
        vec![Some(2), Some(9), None, Some(7), Some(9)]
    } else {
        vec![Some(7), None, Some(2)]
    };
    let values: ArrayRef = if unsigned {
        Arc::new(UInt64Array::from(if update {
            vec![Some(u64::MAX), None, Some(3), Some(1), Some(2)]
        } else {
            vec![Some(u64::MAX), None, Some(1)]
        }))
    } else {
        Arc::new(Int64Array::from(if update {
            vec![Some(i64::MIN), None, Some(3), Some(1), Some(2)]
        } else {
            vec![Some(i64::MAX), None, Some(-1)]
        }))
    };
    let record = RecordBatch::try_from_iter([
        ("key", Arc::new(Int64Array::from(keys)) as ArrayRef),
        ("value", values),
    ])
    .unwrap();
    Batch::table(vec![record], crate::BatchMetadata::default()).unwrap()
}

async fn native_ranked_candidates(
    plan: &IncrementalSql,
    input: &Batch,
    context: &StreamOperatorContext<'_>,
) -> (MemoryReservation, InputCandidates) {
    let (credit, workspace, mut candidates) = plan
        .input_candidates(input.num_rows(), "native-ranks")
        .unwrap();
    assert!(
        candidates.native.is_some(),
        "default primitive route must use native keys"
    );
    assert!(matches!(candidates.groups, CandidateGroups::Ranked(_)));
    plan.update_records(
        input.table_payload().unwrap().batches(),
        &mut candidates,
        (&credit, workspace),
        context,
        "native-ranks",
    )
    .await
    .unwrap();
    let mut native = candidates.native.take().unwrap();
    plan.native_candidates(&mut native, &mut candidates, context, "native-ranks")
        .await
        .unwrap();
    let mut partial = candidates.partial.take().unwrap();
    assert_eq!(partial.slots, [2, 3, 1, 0]);
    partial
        .merge(
            &mut candidates.groups,
            &plan.groups,
            context,
            "native-ranks",
        )
        .await
        .unwrap();
    assert_eq!(partial.candidate_slot_lookups, 0);
    candidates
        .groups
        .take_slots(std::mem::take(&mut partial.slots));
    (credit, candidates)
}

fn assert_native_output(records: &[RecordBatch], unsigned: bool) {
    let record = &records[0];
    assert_eq!(record.num_rows(), 4);
    let keys = record
        .column(0)
        .as_any()
        .downcast_ref::<Int64Array>()
        .unwrap();
    assert_eq!(
        keys,
        &Int64Array::from(vec![Some(7), None, Some(2), Some(9)])
    );
    let expected: ArrayRef = if unsigned {
        Arc::new(UInt64Array::from(vec![Some(0), Some(3), Some(0), Some(2)]))
    } else {
        Arc::new(Int64Array::from(vec![
            Some(i64::MIN),
            Some(3),
            Some(i64::MAX),
            Some(2),
        ]))
    };
    assert_eq!(record.column(1), &expected);
}

#[tokio::test]
async fn test_grouped_candidate_ranks_native_default_signed_unsigned_null_wrapping_and_order() {
    for unsigned in [false, true] {
        let input = native_input(unsigned, false);
        let runtime = DataFusionRuntime::new(DataFusionConfig::default()).unwrap();
        let query = crate::expression::parse_select_query(
            "SELECT key, SUM(value) AS total FROM events GROUP BY key",
        )
        .unwrap();
        let mut plan = IncrementalSql::plan(
            &runtime,
            &query,
            "events",
            input.table_payload().unwrap().schema().clone(),
            "native-ranks",
        )
        .await
        .unwrap()
        .unwrap();
        assert!(plan.compact_budget.is_none());
        assert!(plan.inline_sum.is_some());
        let job = StreamJobContext::new(
            978,
            "native-ranks",
            JsonMap::new(),
            None,
            CancellationToken::new(),
        );
        let context = StreamOperatorContext::new(&job, "native-ranks", None);
        let seed = plan.update(&input, &context, "native-ranks").await.unwrap();
        let held = seed.records.clone();
        plan.commit(seed);
        let baseline = runtime.compact_state_bytes();
        let input = native_input(unsigned, true);
        let (credit, candidates) = native_ranked_candidates(&plan, &input, &context).await;
        let prepared = plan
            .prepare_transaction(candidates, &credit, &context, "native-ranks")
            .await
            .unwrap();
        assert_native_output(&prepared.0, unsigned);
        drop((prepared, credit));
        assert_eq!(runtime.compact_state_bytes(), baseline);
        let transaction = plan.update(&input, &context, "native-ranks").await.unwrap();
        assert_native_output(&transaction.records, unsigned);
        plan.commit(transaction);
        assert_eq!(held[0].num_rows(), 3);
        assert!(held[0].column(1).is_null(1));
        drop((held, plan));
        assert_eq!(runtime.incremental_memory_pool().reserved(), 0);
    }
}

#[tokio::test]
async fn test_grouped_candidate_ranks_empty_native_selection_releases_partial_slots() {
    let record = RecordBatch::try_from_iter([
        (
            "key",
            Arc::new(Int64Array::from(vec![Some(7), None])) as ArrayRef,
        ),
        (
            "value",
            Arc::new(Int64Array::from(vec![Some(-1), None])) as ArrayRef,
        ),
    ])
    .unwrap();
    let runtime = DataFusionRuntime::new(DataFusionConfig::default()).unwrap();
    let query = crate::expression::parse_select_query(
        "SELECT key, SUM(value) AS total FROM events WHERE value > 0 GROUP BY key",
    )
    .unwrap();
    let mut plan = IncrementalSql::plan(&runtime, &query, "events", record.schema(), "empty-ranks")
        .await
        .unwrap()
        .unwrap();
    let input = Batch::table(vec![record], crate::BatchMetadata::default()).unwrap();
    let baseline = runtime.compact_state_bytes();
    let job = StreamJobContext::new(
        979,
        "empty-ranks",
        JsonMap::new(),
        None,
        CancellationToken::new(),
    );
    let context = StreamOperatorContext::new(&job, "empty-ranks", None);
    let (credit, workspace, mut candidates) = plan
        .input_candidates(input.num_rows(), "empty-ranks")
        .unwrap();
    assert!(candidates.native.is_some());
    assert!(plan.compact_budget.is_none());
    plan.update_records(
        input.table_payload().unwrap().batches(),
        &mut candidates,
        (&credit, workspace),
        &context,
        "empty-ranks",
    )
    .await
    .unwrap();
    assert_eq!(candidates.native.as_ref().unwrap().groups.len(), 0);
    let partial = candidates.partial.as_ref().unwrap();
    assert_eq!(partial.slots.len(), 0);
    assert_eq!(
        partial.slots.capacity(),
        4,
        "exercise the real zero-group native reserve"
    );
    plan.finish_candidates(&mut candidates, &context, "empty-ranks")
        .await
        .unwrap();
    assert_eq!(candidates.groups.len(), 0);
    assert_eq!(
        candidates.groups.ranked_layout().3,
        0,
        "zero candidates cannot pay transferred slot storage"
    );
    let prepared = plan
        .prepare_transaction(candidates, &credit, &context, "empty-ranks")
        .await
        .unwrap();
    assert_eq!(prepared.0[0].num_rows(), 0);
    drop((prepared, credit));
    assert_eq!(runtime.compact_state_bytes(), baseline);
    drop(plan);
    assert_eq!(runtime.incremental_memory_pool().reserved(), 0);
}

#[tokio::test]
async fn test_grouped_candidate_ranks_cancellation_after_admission_refunds_without_installing() {
    let fixture = allocation_fixture(true).await;
    let before = fixture
        .plan
        .export_native_state("rank-cancel", || Ok(()))
        .unwrap();
    let empty = CandidateGroups::new(false);
    let output = fixture
        .plan
        .output_chunk(0, GROUPS, &empty, "rank-cancel")
        .unwrap();
    let baseline = fixture.runtime.compact_state_bytes();
    let job = StreamJobContext::new(
        980,
        "rank-cancel",
        JsonMap::new(),
        None,
        CancellationToken::new(),
    );
    let context = StreamOperatorContext::new(&job, "rank-cancel", None);
    let (credit, workspace, mut candidates) = fixture
        .plan
        .input_candidates(GROUPS, "rank-cancel")
        .unwrap();
    let input = reversed_input(fixture.plan.schema.clone());
    fixture
        .plan
        .update_records(
            input.table_payload().unwrap().batches(),
            &mut candidates,
            (&credit, workspace),
            &context,
            "rank-cancel",
        )
        .await
        .unwrap();
    assert_eq!(candidates.groups.len(), GROUPS);
    assert!(matches!(candidates.groups, CandidateGroups::Ranked(_)));
    let result = {
        let finish = fixture
            .plan
            .finish_candidates(&mut candidates, &context, "rank-cancel");
        tokio::pin!(finish);
        assert!(futures::poll!(finish.as_mut()).is_pending());
        job.cancellation().cancel();
        finish.await
    };
    assert!(matches!(result, Err(CalcFlowError::Cancelled { .. })));
    drop((candidates, credit));
    assert_eq!(fixture.runtime.compact_state_bytes(), baseline);
    let after = fixture
        .plan
        .export_native_state("rank-cancel", || Ok(()))
        .unwrap();
    assert_eq!(before.records(), after.records());
    assert_eq!(
        output,
        fixture
            .plan
            .output_chunk(0, GROUPS, &empty, "rank-cancel")
            .unwrap()
    );
    drop((before, after, fixture.partial, fixture.plan));
    assert_eq!(fixture.runtime.incremental_memory_pool().reserved(), 0);
}

#[tokio::test]
async fn test_grouped_candidate_ranks_full_update_exact_and_one_byte_short_custom_credit() {
    let mut fixture = cost_fixture().await;
    let job = StreamJobContext::new(
        981,
        "rank-credit",
        JsonMap::new(),
        None,
        CancellationToken::new(),
    );
    let context = StreamOperatorContext::new(&job, "rank-credit", None);
    let before = fixture
        .plan
        .export_native_state("rank-credit", || Ok(()))
        .unwrap();
    let baseline = fixture.pool.reserved();
    fixture.pool.take_peak();
    let probe = fixture
        .plan
        .update(&fixture.input, &context, "rank-credit")
        .await
        .unwrap();
    let peak = fixture.pool.take_peak();
    assert_cost_output(
        &probe.records,
        &fixture.plan.output_schema,
        &cost_expected(),
    );
    drop(probe);
    assert_eq!(fixture.pool.reserved(), baseline);
    fixture
        .pool
        .set_limit(Some(u64::try_from(peak - 1).unwrap()))
        .unwrap();
    let error = fixture
        .plan
        .update(&fixture.input, &context, "rank-credit")
        .await
        .err()
        .unwrap();
    assert!(error.to_string().contains("candidate probe"), "{error}");
    assert_eq!(fixture.pool.reserved(), baseline);
    let after = fixture
        .plan
        .export_native_state("rank-credit", || Ok(()))
        .unwrap();
    assert_eq!(before.records(), after.records());
    drop(after);
    fixture
        .pool
        .set_limit(Some(u64::try_from(peak).unwrap()))
        .unwrap();
    let retry = fixture
        .plan
        .update(&fixture.input, &context, "rank-credit")
        .await
        .unwrap();
    assert_cost_output(
        &retry.records,
        &fixture.plan.output_schema,
        &cost_expected(),
    );
    fixture.plan.commit(retry);
    drop((before, fixture.plan));
    assert_eq!(fixture.pool.reserved(), 0);
    assert_eq!(fixture.runtime.incremental_memory_pool().reserved(), 0);
}

#[test]
fn test_grouped_candidate_ranks_checked_capacity_rejects_overflow() {
    let error = candidate_groups::growth_peak(usize::MAX, 0, "rank-overflow").unwrap_err();
    assert!(
        error
            .to_string()
            .contains("candidate table capacity overflowed")
    );
    assert!(
        matches!(error, CalcFlowError::DataFusion { node_id: Some(name), .. } if name == "rank-overflow")
    );
}
