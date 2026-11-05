use super::*;
use crate::{CancellationToken, DataFusionConfig, JsonMap, StreamJobContext};
use datafusion::arrow::array::{Int64Array, StringArray};

const GROUPS: usize = 256;

#[path = "grouped_inline_semantic_tests.rs"]
mod semantics;
#[path = "grouped_inline_storage_tests.rs"]
mod storage;

struct AllocationFixture {
    runtime: DataFusionRuntime,
    plan: IncrementalSql,
    partial: PartialGroups,
    states: Vec<Vec<ArrayRef>>,
}

fn allocation_input() -> Batch {
    let record = RecordBatch::try_from_iter(vec![
        (
            "key",
            Arc::new(StringArray::from_iter_values(
                (0..GROUPS).map(|group| format!("account-{group}")),
            )) as ArrayRef,
        ),
        (
            "value",
            Arc::new(Int64Array::from(vec![Some(1); GROUPS])) as ArrayRef,
        ),
    ])
    .unwrap();
    Batch::table(vec![record], crate::BatchMetadata::default()).unwrap()
}

async fn allocation_fixture(single: bool) -> AllocationFixture {
    let input = allocation_input();
    let runtime = DataFusionRuntime::new(DataFusionConfig::default()).unwrap();
    let sql = if single {
        "SELECT key, SUM(value) AS total FROM events GROUP BY key"
    } else {
        "SELECT key, SUM(value) AS total, COUNT(value) AS valid FROM events GROUP BY key"
    };
    let query = crate::expression::parse_select_query(sql).unwrap();
    let mut plan = IncrementalSql::plan(
        &runtime,
        &query,
        "events",
        input.table_payload().unwrap().schema().clone(),
        "inline-allocation",
    )
    .await
    .unwrap()
    .unwrap();
    let job = StreamJobContext::new(
        971,
        "inline-allocation",
        JsonMap::new(),
        None,
        CancellationToken::new(),
    );
    let context = StreamOperatorContext::new(&job, "inline-allocation", None);
    let initial = plan
        .update(&input, &context, "inline-allocation")
        .await
        .unwrap();
    plan.commit(initial);
    let partial = plan.partial_groups("inline-allocation").unwrap().unwrap();
    let mut states = vec![vec![
        Arc::new(Int64Array::from(vec![Some(2); GROUPS])) as ArrayRef
    ]];
    if !single {
        states.push(vec![
            Arc::new(Int64Array::from(vec![Some(1); GROUPS])) as ArrayRef
        ]);
    }
    AllocationFixture {
        runtime,
        plan,
        partial,
        states,
    }
}

fn updated_candidate(fixture: &AllocationFixture, previous: &Group, rank: usize) -> Candidate {
    let mut candidate = fixture
        .plan
        .candidate(Some(previous), &previous.key, None, "inline-allocation")
        .unwrap();
    fixture
        .partial
        .merge_rank(&mut candidate, &fixture.states, rank, "inline-allocation")
        .unwrap();
    fixture
        .plan
        .finalize_candidate(&mut candidate, "inline-allocation")
        .unwrap();
    candidate
}

async fn candidate_allocation_case(single: bool) -> allocation_counter::AllocationInfo {
    let fixture = allocation_fixture(single).await;
    let mut retained = Vec::with_capacity(GROUPS);
    let allocations = allocation_counter::measure(|| {
        for (rank, previous) in fixture.plan.groups.iter().enumerate() {
            retained.push(updated_candidate(&fixture, previous, rank));
        }
    });
    assert_eq!(retained.len(), GROUPS);
    drop((retained, fixture.partial, fixture.plan));
    assert_eq!(fixture.runtime.incremental_memory_pool().reserved(), 0);
    allocations
}

#[tokio::test]
async fn test_grouped_inline_single_sum_avoids_owned_state_and_result_heap_vectors() {
    let single = candidate_allocation_case(true).await;
    let mixed = candidate_allocation_case(false).await;
    eprintln!("Grouped state owned heap: single={single:?}, mixed={mixed:?}");
    let maximum = i64::try_from(2 * GROUPS).unwrap();
    assert!(
        mixed.count_current > maximum,
        "mixed aggregates must retain their generic owned state: {mixed:?}"
    );
    assert!(
        single.count_current <= maximum,
        "one integer SUM must avoid per-group state/result heap vectors at the actual candidate/finalize boundary: {single:?}"
    );
    assert!(
        single.count_total <= u64::try_from(7 * GROUPS).unwrap(),
        "inline storage must not add transient candidate/finalize allocations over the measured legacy boundary: {single:?}"
    );
}

#[tokio::test]
async fn test_grouped_inline_sum_has_no_finalizer_vector() {
    let fixture = allocation_fixture(true).await;
    let candidate = updated_candidate(&fixture, &fixture.plan.groups[0], 0);
    assert!(
        candidate.accumulators.is_empty(),
        "inline SUM must not allocate a placeholder finalizer vector"
    );
    drop((candidate, fixture.partial, fixture.plan));
    assert_eq!(fixture.runtime.incremental_memory_pool().reserved(), 0);
}

#[tokio::test]
async fn test_grouped_inline_sum_skips_empty_finalizer_pass() {
    let fixture = allocation_fixture(true).await;
    let candidate = updated_candidate(&fixture, &fixture.plan.groups[0], 0);
    let mut candidates = CandidateMap::default();
    candidates.insert(0, candidate);
    let job = StreamJobContext::new(
        972,
        "inline-allocation",
        JsonMap::new(),
        None,
        CancellationToken::new(),
    );
    let context = StreamOperatorContext::new(&job, "inline-allocation", None);
    let before = fixture
        .plan
        .finalized_groups
        .load(std::sync::atomic::Ordering::SeqCst);
    fixture
        .plan
        .finalize_candidates(&mut candidates, &context, "inline-allocation")
        .await
        .unwrap();
    assert_eq!(
        fixture
            .plan
            .finalized_groups
            .load(std::sync::atomic::Ordering::SeqCst),
        before,
        "inline SUM must skip the empty per-candidate finalizer pass"
    );
    drop((candidates, fixture.partial, fixture.plan));
    assert_eq!(fixture.runtime.incremental_memory_pool().reserved(), 0);
}
