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
    AllocationFixture { runtime, plan }
}

#[tokio::test]
async fn test_grouped_inline_single_sum_updates_typed_state_without_candidates() {
    let mut fixture = allocation_fixture(true).await;
    assert!(fixture.plan.groups.is_empty());
    assert!(fixture.plan.index.is_empty());
    let before = fixture
        .plan
        .finalized_groups
        .load(std::sync::atomic::Ordering::SeqCst);
    let job = StreamJobContext::new(
        972,
        "inline-allocation",
        JsonMap::new(),
        None,
        CancellationToken::new(),
    );
    let context = StreamOperatorContext::new(&job, "inline-allocation", None);
    let transaction = fixture
        .plan
        .update(&allocation_input(), &context, "inline-allocation")
        .await
        .unwrap();
    assert_eq!(
        fixture
            .plan
            .finalized_groups
            .load(std::sync::atomic::Ordering::SeqCst),
        before,
        "keyed SUM must not enter the generic per-candidate finalizer"
    );
    for record in &transaction.records {
        let totals = record
            .column_by_name("total")
            .unwrap()
            .as_any()
            .downcast_ref::<Int64Array>()
            .unwrap();
        assert!(totals.iter().all(|value| value == Some(2)));
    }
    fixture.plan.commit(transaction);
    assert!(fixture.plan.groups.is_empty());
    drop(fixture.plan);
    assert_eq!(fixture.runtime.incremental_memory_pool().reserved(), 0);
}

#[tokio::test]
async fn test_grouped_mixed_sum_uses_typed_aggregate_state() {
    let mut fixture = allocation_fixture(false).await;
    assert!(fixture.plan.keyed_sum.is_none());
    assert!(fixture.plan.keyed_state.is_some());
    assert!(fixture.plan.groups.is_empty());
    let job = StreamJobContext::new(
        973,
        "inline-allocation",
        JsonMap::new(),
        None,
        CancellationToken::new(),
    );
    let context = StreamOperatorContext::new(&job, "inline-allocation", None);
    let transaction = fixture
        .plan
        .update(&allocation_input(), &context, "inline-allocation")
        .await
        .unwrap();
    for record in &transaction.records {
        let totals = record
            .column_by_name("total")
            .unwrap()
            .as_any()
            .downcast_ref::<Int64Array>()
            .unwrap();
        let counts = record
            .column_by_name("valid")
            .unwrap()
            .as_any()
            .downcast_ref::<Int64Array>()
            .unwrap();
        assert!(totals.iter().all(|value| value == Some(2)));
        assert!(counts.iter().all(|value| value == Some(2)));
    }
    fixture.plan.commit(transaction);
    assert!(fixture.plan.groups.is_empty());
    drop(fixture.plan);
    assert_eq!(fixture.runtime.incremental_memory_pool().reserved(), 0);
}
