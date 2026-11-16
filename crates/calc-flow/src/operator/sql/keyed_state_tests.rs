use super::*;
use crate::{BatchMetadata, CancellationToken, DataFusionConfig, JsonMap, StreamJobContext};
use datafusion::arrow::array::StringArray;
use std::collections::BTreeMap;

const SQL: &str = "SELECT key, SUM(value) AS total, AVG(value) AS mean, MIN(value) AS lo, \
                   MAX(value) AS hi, COUNT(*) AS rows FROM events GROUP BY key";

fn input(keys: &[Option<&str>], values: &[Option<i64>]) -> Batch {
    let record = RecordBatch::try_from_iter(vec![
        (
            "key",
            Arc::new(StringArray::from(keys.to_vec())) as ArrayRef,
        ),
        (
            "value",
            Arc::new(Int64Array::from(values.to_vec())) as ArrayRef,
        ),
    ])
    .unwrap();
    Batch::table(vec![record], BatchMetadata::default()).unwrap()
}

fn rows(records: &[RecordBatch]) -> Vec<Vec<ScalarValue>> {
    let mut rows = records
        .iter()
        .flat_map(|record| {
            (0..record.num_rows()).map(|row| {
                record
                    .columns()
                    .iter()
                    .map(|array| ScalarValue::try_from_array(array, row).unwrap())
                    .collect::<Vec<_>>()
            })
        })
        .collect::<Vec<_>>();
    rows.sort_by_key(|row| format!("{row:?}"));
    rows
}

async fn plan_for(runtime: &DataFusionRuntime, batch: &Batch) -> IncrementalSql {
    let query = crate::expression::parse_select_query(SQL).unwrap();
    let schema = batch.table_payload().unwrap().schema().clone();
    IncrementalSql::plan(runtime, &query, "events", schema, "keyed-state")
        .await
        .unwrap()
        .unwrap()
}

fn context() -> StreamJobContext {
    StreamJobContext::new(
        992,
        "keyed-state",
        JsonMap::new(),
        None,
        CancellationToken::new(),
    )
}

async fn oracle(runtime: &DataFusionRuntime, history: &[RecordBatch]) -> Vec<Vec<ScalarValue>> {
    let expected = runtime
        .sql(
            SQL,
            &BTreeMap::from([(
                "events".into(),
                Batch::table(history.to_vec(), BatchMetadata::default()).unwrap(),
            )]),
            Some("keyed-state-oracle"),
        )
        .await
        .unwrap();
    rows(expected.table_payload().unwrap().batches())
}

#[tokio::test]
async fn test_keyed_mixed_aggregates_match_prefix_rollback_and_restore() {
    let runtime = DataFusionRuntime::new(DataFusionConfig::default()).unwrap();
    let first = input(
        &[Some("a"), Some("b"), None, Some("a"), Some("b")],
        &[Some(3), None, Some(4), Some(5), Some(7)],
    );
    let second = input(
        &[Some("a"), Some("c"), None, Some("b")],
        &[Some(8), Some(9), None, Some(11)],
    );
    let empty = Batch::table(
        vec![first.table_payload().unwrap().batches()[0].slice(0, 0)],
        BatchMetadata::default(),
    )
    .unwrap();
    let mut plan = plan_for(&runtime, &first).await;
    assert!(plan.keyed_state.is_some());
    assert!(plan.groups.is_empty());
    assert!(plan.index.is_empty());
    let job = context();
    let context = StreamOperatorContext::new(&job, "keyed-state", None);
    let mut history = first.table_payload().unwrap().batches().to_vec();
    let transaction = plan.update(&first, &context, "keyed-state").await.unwrap();
    assert_eq!(rows(&transaction.records), oracle(&runtime, &history).await);
    plan.commit(transaction);
    let before = plan.export_native_state("keyed-state", || Ok(())).unwrap();

    let discarded = plan.update(&second, &context, "keyed-state").await.unwrap();
    drop(discarded);
    let after = plan.export_native_state("keyed-state", || Ok(())).unwrap();
    assert_eq!(before.records(), after.records());

    history.extend(second.table_payload().unwrap().batches().to_vec());
    let transaction = plan.update(&second, &context, "keyed-state").await.unwrap();
    assert_eq!(rows(&transaction.records), oracle(&runtime, &history).await);
    plan.commit(transaction);
    let exported = plan.export_native_state("keyed-state", || Ok(())).unwrap();
    let mut restored = plan_for(&runtime, &first)
        .await
        .import_native_state(
            exported.records(),
            u64::try_from(first.num_rows() + second.num_rows()).unwrap(),
            true,
            || Ok(()),
            "keyed-state",
        )
        .unwrap();
    let reexported = restored
        .export_native_state("keyed-state", || Ok(()))
        .unwrap();
    assert_eq!(exported.records(), reexported.records());

    history.extend(empty.table_payload().unwrap().batches().to_vec());
    let left = plan.update(&empty, &context, "keyed-state").await.unwrap();
    let right = restored
        .update(&empty, &context, "keyed-state")
        .await
        .unwrap();
    assert_eq!(rows(&left.records), rows(&right.records));
    assert_eq!(rows(&right.records), oracle(&runtime, &history).await);
    plan.commit(left);
    restored.commit(right);
    drop((before, after, exported, reexported, plan, restored));
    assert_eq!(runtime.incremental_memory_pool().reserved(), 0);
    assert_eq!(runtime.compact_state_bytes(), 0);
}
