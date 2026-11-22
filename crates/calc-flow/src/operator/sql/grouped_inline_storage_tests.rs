use super::*;
use crate::{CancellationToken, JsonMap, StreamJobContext};

#[test]
fn test_grouped_inline_compact_layout_preserves_generic_container_sizes() {
    type LegacyStorage = (Vec<Vec<ScalarValue>>, Vec<ScalarValue>);
    assert!(size_of::<GroupAggregates>() <= size_of::<LegacyStorage>());
    assert!(
        size_of::<IncrementalSql>() <= 4096,
        "keyed SUM must retain the existing plan structural floor"
    );
}

#[tokio::test]
async fn test_grouped_inline_new_key_budget_refusal_is_atomic() {
    let mut fixture = allocation_fixture(true).await;
    let before = fixture
        .plan
        .export_native_state("inline-budget", || Ok(()))
        .unwrap();
    let baseline = fixture.runtime.compact_state_bytes();
    let groups = fixture.plan.keyed_sum.as_ref().unwrap().len();
    fixture
        .runtime
        .set_compact_state_limit(Some(u64::try_from(baseline).unwrap()), "inline-budget")
        .unwrap();
    let record = RecordBatch::try_from_iter(vec![
        (
            "key",
            Arc::new(StringArray::from(vec![Some("new-account")])) as ArrayRef,
        ),
        (
            "value",
            Arc::new(Int64Array::from(vec![Some(9)])) as ArrayRef,
        ),
    ])
    .unwrap();
    let input = Batch::table(vec![record], crate::BatchMetadata::default()).unwrap();
    let job = StreamJobContext::new(
        973,
        "inline-budget",
        JsonMap::new(),
        None,
        CancellationToken::new(),
    );
    let context = StreamOperatorContext::new(&job, "inline-budget", None);
    let error = fixture.plan.update(&input, &context, "inline-budget").await;
    assert!(error.is_err());
    assert_eq!(fixture.runtime.compact_state_bytes(), baseline);
    assert_eq!(fixture.plan.keyed_sum.as_ref().unwrap().len(), groups);
    let after = fixture
        .plan
        .export_native_state("inline-budget", || Ok(()))
        .unwrap();
    assert_eq!(before.records(), after.records());
    fixture
        .runtime
        .set_compact_state_limit(None, "inline-budget")
        .unwrap();
    drop((before, after, fixture.plan));
    assert_eq!(fixture.runtime.incremental_memory_pool().reserved(), 0);
    assert_eq!(fixture.runtime.compact_state_bytes(), 0);
}
