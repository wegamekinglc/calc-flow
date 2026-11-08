use super::*;
use crate::{EdgeCollector, Epoch, OperatorMetadata, SqlOperator, StreamCollector, StreamOperator};

#[path = "c1_cache_budget_tests.rs"]
mod budgets;

fn job() -> StreamJobContext {
    StreamJobContext::new(975, NAME, JsonMap::new(), None, CancellationToken::new())
}

async fn query_plan(runtime: &DataFusionRuntime, batch: &Batch, sql: &str) -> IncrementalSql {
    IncrementalSql::plan(
        runtime,
        &crate::expression::parse_select_query(sql).unwrap(),
        "events",
        batch.table_payload().unwrap().schema().clone(),
        NAME,
    )
    .await
    .unwrap()
    .unwrap()
}

#[tokio::test]
async fn test_c1_tail_growth_and_empty_append_keep_sealed_aggregate_chunks() {
    let job = job();
    let context = StreamOperatorContext::new(&job, NAME, None);
    let mut fixture = fixture(&context).await;
    let extra = (Some("new-account".into()), "new-stock".into());
    let transaction = fixture
        .plan
        .update(&input(std::slice::from_ref(&extra), 4), &context, NAME)
        .await
        .unwrap();
    assert!(Arc::ptr_eq(
        fixture.original[0].column(2),
        transaction.records[0].column(2)
    ));
    assert!(!Arc::ptr_eq(
        fixture.original[1].column(2),
        transaction.records[1].column(2)
    ));
    assert_eq!(transaction.records[1].num_rows(), GROUPS - CHUNK_ROWS + 1);
    fixture.expected.insert(extra.clone(), 4);
    fixture.keys.push(extra);
    assert_eq!(totals(&transaction.records), fixture.expected);
    let grown = transaction.records.clone();
    fixture.plan.commit(transaction);
    let empty = fixture
        .plan
        .update(&input(&[], 0), &context, NAME)
        .await
        .unwrap();
    for (before, after) in grown.iter().zip(&empty.records) {
        assert!(Arc::ptr_eq(before.column(2), after.column(2)));
    }
    assert_eq!(
        empty
            .records
            .iter()
            .flat_map(record_keys)
            .collect::<Vec<_>>(),
        fixture.keys
    );
    assert_eq!(totals(&empty.records), fixture.expected);
    fixture.plan.commit(empty);
    assert_eq!(state_totals(&fixture.plan), fixture.expected);
    assert_eq!(totals(&fixture.original), fixture.initial);
}

#[tokio::test]
async fn test_c1_generic_aggregate_readouts_share_clean_chunks() {
    let runtime = DataFusionRuntime::new(DataFusionConfig::default()).unwrap();
    let keys = warm_keys();
    let warm = input(&keys, 1);
    let query = "SELECT account, stock, SUM(value), COUNT(value), MIN(value), MAX(value), BIT_OR(value), BIT_AND(value), BIT_XOR(value) FROM events GROUP BY account, stock";
    let mut plan = query_plan(&runtime, &warm, query).await;
    let job = job();
    let context = StreamOperatorContext::new(&job, NAME, None);
    let transaction = plan.update(&warm, &context, NAME).await.unwrap();
    let original = transaction.records.clone();
    plan.commit(transaction);
    let transaction = plan
        .update(&input(&keys[9000..], 7), &context, NAME)
        .await
        .unwrap();
    for column in 2..9 {
        assert!(Arc::ptr_eq(
            original[0].column(column),
            transaction.records[0].column(column)
        ));
        assert!(!Arc::ptr_eq(
            original[1].column(column),
            transaction.records[1].column(column)
        ));
    }
    assert_generic_results(&transaction.records, &keys);
    for record in original {
        for column in 2..9 {
            let array = record
                .column(column)
                .as_any()
                .downcast_ref::<Int64Array>()
                .unwrap();
            assert!(array.values().iter().all(|&value| value == 1));
        }
    }
}

fn assert_generic_results(records: &[RecordBatch], keys: &[Key]) {
    assert_eq!(
        records.iter().flat_map(record_keys).collect::<Vec<_>>(),
        keys
    );
    for (chunk, record) in records.iter().enumerate() {
        for row in 0..record.num_rows() {
            let changed = chunk * CHUNK_ROWS + row >= 9000;
            let expected = if changed {
                [8, 2, 1, 7, 7, 1, 6]
            } else {
                [1; 7]
            };
            for (column, expected) in (2..9).zip(expected) {
                assert_eq!(
                    ScalarValue::try_from_array(record.column(column), row).unwrap(),
                    ScalarValue::Int64(Some(expected))
                );
            }
        }
    }
}

#[tokio::test]
async fn test_c1_having_projection_order_and_limit_reexecute_over_cached_raw_columns() {
    let runtime = DataFusionRuntime::new(DataFusionConfig::default()).unwrap();
    let keys = warm_keys();
    let warm = input(&keys, 1);
    let query = "SELECT account, stock, SUM(value) * 2 AS doubled FROM events GROUP BY account, stock HAVING SUM(value) > 1 ORDER BY doubled DESC, stock, account LIMIT 5";
    let mut plan = query_plan(&runtime, &warm, query).await;
    let job = job();
    let context = StreamOperatorContext::new(&job, NAME, None);
    let transaction = plan.update(&warm, &context, NAME).await.unwrap();
    assert_eq!(
        transaction
            .records
            .iter()
            .map(RecordBatch::num_rows)
            .sum::<usize>(),
        0
    );
    plan.commit(transaction);
    let transaction = plan
        .update(&input(&keys[9000..], 7), &context, NAME)
        .await
        .unwrap();
    assert_eq!(
        transaction
            .records
            .iter()
            .flat_map(record_keys)
            .collect::<Vec<_>>(),
        (90..95)
            .map(|account| (Some(format!("account-{account:03}")), "stock-000".into()))
            .collect::<Vec<_>>()
    );
    assert!(
        totals(&transaction.records)
            .values()
            .all(|&value| value == 16)
    );
    plan.commit(transaction);
    let cached_tail = plan.snapshot_aggregates.get(1).unwrap()[0].clone();
    let transaction = plan
        .update(&input(&keys[..1], 9), &context, NAME)
        .await
        .unwrap();
    assert!(Arc::ptr_eq(
        &cached_tail,
        &transaction
            .snapshot_aggregates
            .as_ref()
            .unwrap()
            .get(1)
            .unwrap()[0]
    ));
    assert_eq!(
        transaction
            .records
            .iter()
            .map(RecordBatch::num_rows)
            .sum::<usize>(),
        5
    );
    assert_eq!(record_keys(&transaction.records[0])[0], keys[0]);
    assert_eq!(
        ScalarValue::try_from_array(transaction.records[0].column(2), 0).unwrap(),
        ScalarValue::Int64(Some(20))
    );
}

struct RejectOutput;

#[async_trait::async_trait]
impl StreamCollector for RejectOutput {
    async fn emit(&mut self, _port: &str, _batch: Batch) -> Result<()> {
        Err(df_error(NAME, "injected C1 emit failure"))
    }
}

struct PendingOutput(Option<tokio::sync::oneshot::Sender<()>>);

#[async_trait::async_trait]
impl StreamCollector for PendingOutput {
    async fn emit(&mut self, _port: &str, _batch: Batch) -> Result<()> {
        self.0.take().unwrap().send(()).unwrap();
        std::future::pending().await
    }
}

async fn warm_operator(context: &StreamOperatorContext<'_>) -> (SqlOperator, Vec<RecordBatch>) {
    let mut operator = SqlOperator::new(NAME, SQL, vec!["events".into()], vec![]).unwrap();
    let mut collector = EdgeCollector::new(operator.output_ports().to_vec());
    operator
        .process_data("events", input(&warm_keys(), 1), context, &mut collector)
        .await
        .unwrap();
    let messages = collector.drain("output");
    let records = messages[0]
        .as_data()
        .unwrap()
        .table_payload()
        .unwrap()
        .batches()
        .to_vec();
    (operator, records)
}

#[tokio::test]
async fn test_c1_failed_and_dropped_emits_preserve_installed_cache_and_counters() {
    let job = job();
    let context = StreamOperatorContext::new(&job, NAME, None);
    let (mut operator, original) = warm_operator(&context).await;
    let ledger = operator.compact.as_ref().unwrap().ledger;
    let keys = warm_keys();
    let incoming = input(&keys[9000..], 7);
    let error = operator
        .process_data("events", incoming.clone(), &context, &mut RejectOutput)
        .await
        .unwrap_err();
    assert!(error.to_string().contains("injected C1 emit failure"));
    assert_eq!(operator.compact.as_ref().unwrap().ledger, ledger);
    let (entered, started) = tokio::sync::oneshot::channel();
    let mut pending = PendingOutput(Some(entered));
    {
        let process = operator.process_data("events", incoming, &context, &mut pending);
        tokio::pin!(process);
        tokio::select! {
            result = &mut process => panic!("emit unexpectedly finished: {result:?}"),
            result = started => result.unwrap(),
        }
    }
    assert_eq!(operator.compact.as_ref().unwrap().ledger, ledger);
    assert_eq!(operator.incremental_work, (GROUPS + 2 * APPEND_ROWS, 1));
    assert_eq!(
        state_totals(operator.incremental.as_ref().unwrap()),
        totals(&original)
    );
    let mut collector = EdgeCollector::new(operator.output_ports().to_vec());
    let accepted = input(&keys[..1], 3);
    let accepted_bytes = u64::try_from(accepted.estimated_bytes().unwrap()).unwrap();
    operator
        .process_data("events", accepted, &context, &mut collector)
        .await
        .unwrap();
    let messages = collector.drain("output");
    let records = messages[0]
        .as_data()
        .unwrap()
        .table_payload()
        .unwrap()
        .batches();
    assert!(Arc::ptr_eq(original[1].column(2), records[1].column(2)));
    let mut expected = totals(&original);
    add_values(&mut expected, &keys[..1], 3);
    assert_snapshot(records, &keys, &expected);
    assert_eq!(operator.incremental_work, (GROUPS + 2 * APPEND_ROWS + 1, 1));
    assert_eq!(
        operator.compact.as_ref().unwrap().ledger.rows,
        ledger.rows + 1
    );
    assert_eq!(
        operator.compact.as_ref().unwrap().ledger.bytes,
        ledger.bytes + accepted_bytes
    );
}

#[tokio::test]
async fn test_c1_restore_and_reset_start_with_cold_aggregate_cache() {
    let job = job();
    let context = StreamOperatorContext::new(&job, NAME, None);
    let (mut original_operator, original) = warm_operator(&context).await;
    let snapshot = original_operator.checkpoint(Epoch::INITIAL).unwrap();
    let mut restored = original_operator.clone();
    restored.restore(&snapshot).unwrap();
    assert!(
        restored
            .incremental
            .as_ref()
            .unwrap()
            .snapshot_aggregates
            .get(0)
            .is_none()
    );
    let keys = warm_keys();
    let mut collector = EdgeCollector::new(restored.output_ports().to_vec());
    restored
        .process_data("events", input(&keys[9000..], 2), &context, &mut collector)
        .await
        .unwrap();
    let messages = collector.drain("output");
    let records = messages[0]
        .as_data()
        .unwrap()
        .table_payload()
        .unwrap()
        .batches();
    assert!(!Arc::ptr_eq(original[0].column(2), records[0].column(2)));
    let mut expected = totals(&original);
    add_values(&mut expected, &keys[9000..], 2);
    assert_snapshot(records, &keys, &expected);
    restored.reset().unwrap();
    assert!(restored.incremental.is_none());
    restored
        .process_data("events", input(&keys[..1], 9), &context, &mut collector)
        .await
        .unwrap();
    let messages = collector.drain("output");
    assert_eq!(
        totals(
            messages[0]
                .as_data()
                .unwrap()
                .table_payload()
                .unwrap()
                .batches()
        ),
        Totals::from([(keys[0].clone(), 9)])
    );
    assert!(totals(&original).values().all(|&value| value == 1));
}

#[tokio::test]
async fn test_c1_volatile_projection_keeps_existing_retained_execution() {
    let query = "SELECT account, stock, SUM(value) AS total, random() AS sample FROM events GROUP BY account, stock";
    let mut operator = SqlOperator::new(NAME, query, vec!["events".into()], vec![]).unwrap();
    let job = job();
    let context = StreamOperatorContext::new(&job, NAME, None);
    let mut collector = EdgeCollector::new(operator.output_ports().to_vec());
    let keys = warm_keys();
    let mut snapshots = Vec::new();
    for value in [1, 2] {
        operator
            .process_data(
                "events",
                input(&keys[..10], value),
                &context,
                &mut collector,
            )
            .await
            .unwrap();
        let messages = collector.drain("output");
        snapshots.push(
            messages[0]
                .as_data()
                .unwrap()
                .table_payload()
                .unwrap()
                .batches()[0]
                .column(3)
                .clone(),
        );
        assert!(operator.incremental.is_none());
    }
    assert_ne!(snapshots[0].to_data(), snapshots[1].to_data());
}

#[tokio::test]
async fn test_c1_cancelled_update_preserves_installed_aggregate_cache() {
    use std::sync::atomic::Ordering;
    let runtime = DataFusionRuntime::new(DataFusionConfig::default()).unwrap();
    let keys = warm_keys();
    let warm = input(&keys, 1);
    let query = "SELECT account, stock, SUM(value) AS total, COUNT(*) AS rows FROM events GROUP BY account, stock";
    let mut plan = query_plan(&runtime, &warm, query).await;
    let job = job();
    let context = StreamOperatorContext::new(&job, NAME, None);
    let transaction = plan.update(&warm, &context, NAME).await.unwrap();
    let original = transaction.records.clone();
    plan.commit(transaction);
    plan.finalized_groups.store(0, Ordering::SeqCst);
    let finalized = plan.finalized_groups.clone();
    let cancellation = job.cancellation().clone();
    let canceller = tokio::spawn(async move {
        while finalized.load(Ordering::SeqCst) == 0 {
            tokio::task::yield_now().await;
        }
        cancellation.cancel();
    });
    let error = plan
        .update(&input(&keys[9000..], 7), &context, NAME)
        .await
        .err()
        .unwrap();
    canceller.await.unwrap();
    assert!(matches!(error, CalcFlowError::Cancelled { .. }));
    assert_eq!(state_totals(&plan), totals(&original));
    for (ordinal, record) in original.iter().enumerate() {
        assert!(Arc::ptr_eq(
            record.column(2),
            &plan.snapshot_aggregates.get(ordinal).unwrap()[0]
        ));
    }
}

#[tokio::test]
async fn test_c1_changed_aggregate_schema_invalidates_cached_chunks() {
    let job = job();
    let context = StreamOperatorContext::new(&job, NAME, None);
    let mut fixture = fixture(&context).await;
    fixture.plan.aggregate_schema = Arc::new(fixture.plan.aggregate_schema.as_ref().clone());
    let transaction = fixture
        .plan
        .update(&input(&[], 0), &context, NAME)
        .await
        .unwrap();
    for (old, current) in fixture.original.iter().zip(&transaction.records) {
        assert!(!Arc::ptr_eq(old.column(2), current.column(2)));
    }
    assert_snapshot(&transaction.records, &fixture.keys, &fixture.initial);
}
