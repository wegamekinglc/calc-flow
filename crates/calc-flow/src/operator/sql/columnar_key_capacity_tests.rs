use super::*;
use crate::{CancellationToken, JsonMap, StreamJobContext};
use datafusion::execution::memory_pool::{GreedyMemoryPool, MemoryConsumer, MemoryPool};

#[test]
fn test_columnar_selector_uses_exact_budget_and_half_input_boundary() {
    let default_bytes = StateBudget::default().max_bytes();
    let budget = Some(StateBudget::new(8, default_bytes).unwrap());
    assert_eq!(selected_groups(0, budget), None);
    assert_eq!(selected_groups(15, budget), None);
    assert_eq!(selected_groups(16, budget), Some(8));
    assert_eq!(selected_groups(17, budget), Some(8));
    assert_eq!(selected_groups(32768, None), None);
    assert_eq!(
        selected_groups(32768, Some(StateBudget::new(8, default_bytes - 1).unwrap())),
        None
    );
    assert_eq!(selected_groups(usize::MAX, budget), Some(8));
}

#[test]
fn test_columnar_envelope_covers_overshoot_and_rejects_invalid_layouts() {
    let envelope = Envelope::new(32768, 256, [4096, 8192], "capacity").unwrap();
    assert_eq!(envelope.groups, 256 + CHUNK_ROWS);
    assert_eq!(envelope.rows, CHUNK_ROWS);
    assert!(envelope.charge > 4 * 8192);
    assert!(Envelope::new(usize::MAX, usize::MAX, [0, 0], "capacity").is_none());
    let maximum = usize::try_from(i32::MAX).unwrap();
    assert!(Envelope::new(32, 8, [maximum, 0], "capacity").is_some());
    assert!(Envelope::new(32, 8, [maximum + 1, 0], "capacity").is_none());
    assert!(round64(usize::MAX).is_none());
    assert!(workspace_bytes(usize::MAX, CHUNK_ROWS, [0, 0], "capacity").is_none());
}

#[tokio::test]
async fn test_columnar_prework_refusal_refunds_and_census_cancellation_is_bounded() {
    let records = [RecordBatch::try_from_iter(vec![
        (
            "left",
            Arc::new(StringArray::from(vec![Some("capacity-left"); 32])) as ArrayRef,
        ),
        (
            "right",
            Arc::new(StringArray::from(vec![Some("capacity-right"); 32])) as ArrayRef,
        ),
    ])
    .unwrap()];
    let bytes = [
        utf8_span(records[0].column(0)).unwrap(),
        utf8_span(records[0].column(1)).unwrap(),
    ];
    let envelope = Envelope::new(32, 8, bytes, "capacity").unwrap();
    let pool: Arc<dyn MemoryPool> = Arc::new(GreedyMemoryPool::new(envelope.charge - 1));
    let reservation = MemoryConsumer::new("columnar-refusal").register(&pool);
    let job = StreamJobContext::new(
        977,
        "capacity",
        JsonMap::new(),
        None,
        CancellationToken::new(),
    );
    let context = StreamOperatorContext::new(&job, "capacity", None);
    let plan = Plan { columns: [0, 1] };
    assert!(
        plan.prepare(
            &records,
            32,
            Some(StateBudget::new(8, StateBudget::default().max_bytes()).unwrap()),
            reservation,
            &context,
            "capacity"
        )
        .await
        .unwrap()
        .is_none()
    );
    assert_eq!(pool.reserved(), 0);
    cancel_census(plan, records[0].clone()).await;
}

async fn cancel_census(plan: Plan, record: RecordBatch) {
    let token = CancellationToken::new();
    let job = StreamJobContext::new(978, "census", JsonMap::new(), None, token.clone());
    let context = StreamOperatorContext::new(&job, "census", None);
    let mut records = vec![record.slice(0, 0); 1024];
    records.push(record);
    let cancel = tokio::spawn(async move {
        tokio::task::yield_now().await;
        token.cancel();
    });
    let result = tokio::time::timeout(
        std::time::Duration::from_secs(5),
        plan.census(&records, &context),
    )
    .await
    .unwrap();
    cancel.await.unwrap();
    assert!(matches!(
        result,
        Err(crate::CalcFlowError::Cancelled { .. })
    ));
}
