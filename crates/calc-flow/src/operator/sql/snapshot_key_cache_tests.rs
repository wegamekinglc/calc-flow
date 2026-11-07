//! Snapshot key-column reuse tests: sealed output chunks share their key
//! arrays across snapshots, and only a chunk whose row count changed is
//! rebuilt. Uses `Arc::ptr_eq` on the emitted records, so the property is
//! observable through the public snapshot contract.

use super::*;
use crate::{CancellationToken, DataFusionConfig, JsonMap, StreamJobContext};
use datafusion::arrow::array::{Array, Int64Array, StringArray};

const NAME: &str = "key-cache";

/// `CHUNK_ROWS` groups plus a partial tail in the second chunk.
const SEALED_ROWS: usize = 8192;
const TAIL_ROWS: usize = 5;

fn keyed_input(keys: &[String], value: i64) -> Batch {
    let record = RecordBatch::try_from_iter(vec![
        (
            "key",
            Arc::new(StringArray::from(
                keys.iter()
                    .map(|key| Some(key.as_str()))
                    .collect::<Vec<_>>(),
            )) as ArrayRef,
        ),
        (
            "value",
            Arc::new(Int64Array::from(vec![Some(value); keys.len()])) as ArrayRef,
        ),
    ])
    .unwrap();
    Batch::table(vec![record], crate::BatchMetadata::default()).unwrap()
}

async fn plan_for(input: &Batch) -> IncrementalSql {
    let query = crate::expression::parse_select_query(
        "SELECT key, SUM(value) AS total FROM events GROUP BY key",
    )
    .unwrap();
    let runtime = DataFusionRuntime::new(DataFusionConfig::default()).unwrap();
    let schema = input.table_payload().unwrap().schema().clone();
    IncrementalSql::plan(&runtime, &query, "events", schema, NAME)
        .await
        .unwrap()
        .unwrap()
}

fn context_for() -> StreamJobContext {
    StreamJobContext::new(972, NAME, JsonMap::new(), None, CancellationToken::new())
}

fn key_column(record: &RecordBatch) -> &ArrayRef {
    record.columns().first().expect("key column")
}

fn assert_shared_key_columns(previous: &[RecordBatch], current: &[RecordBatch], label: &str) {
    assert_eq!(previous.len(), current.len(), "{label} chunk count");
    for (chunk, (before, after)) in previous.iter().zip(current).enumerate() {
        assert_eq!(before.num_rows(), after.num_rows(), "{label} chunk {chunk}");
        assert!(
            Arc::ptr_eq(key_column(before), key_column(after)),
            "{label} chunk {chunk} key column must be the shared snapshot array"
        );
    }
}

fn totals(records: &[RecordBatch]) -> std::collections::BTreeMap<String, i64> {
    let mut totals = std::collections::BTreeMap::new();
    for record in records {
        let keys = record
            .column(0)
            .as_any()
            .downcast_ref::<StringArray>()
            .expect("utf8 keys");
        let values = record
            .column(1)
            .as_any()
            .downcast_ref::<Int64Array>()
            .expect("int64 totals");
        for row in 0..record.num_rows() {
            if keys.is_null(row) {
                continue;
            }
            totals.insert(keys.value(row).to_owned(), values.value(row));
        }
    }
    totals
}

#[tokio::test]
async fn snapshots_reuse_sealed_chunk_key_columns() {
    let initial_keys = (0..SEALED_ROWS + TAIL_ROWS)
        .map(|index| format!("k{index:05}"))
        .collect::<Vec<_>>();
    let input = keyed_input(&initial_keys, 1);
    let mut plan = plan_for(&input).await;
    let job = context_for();
    let context = StreamOperatorContext::new(&job, NAME, None);

    let first = plan.update(&input, &context, NAME).await.unwrap();
    let first_records = first.records.clone();
    assert_eq!(
        first_records
            .iter()
            .map(RecordBatch::num_rows)
            .collect::<Vec<_>>(),
        vec![SEALED_ROWS, TAIL_ROWS],
        "the fixture must produce one sealed and one partial chunk"
    );
    plan.commit(first);

    let touched = vec![
        initial_keys[0].clone(),
        initial_keys[5].clone(),
        initial_keys[SEALED_ROWS + TAIL_ROWS - 1].clone(),
    ];
    let second = plan
        .update(&keyed_input(&touched, 10), &context, NAME)
        .await
        .unwrap();
    let second_records = second.records.clone();
    assert_shared_key_columns(&first_records, &second_records, "unchanged group set");
    plan.commit(second);

    let grown = plan
        .update(
            &keyed_input(&[format!("k{:05}", SEALED_ROWS + TAIL_ROWS)], 1),
            &context,
            NAME,
        )
        .await
        .unwrap();
    let grown_records = grown.records.clone();
    assert_eq!(grown_records[1].num_rows(), TAIL_ROWS + 1);
    assert!(
        Arc::ptr_eq(key_column(&first_records[0]), key_column(&grown_records[0])),
        "the sealed chunk stays shared when only the tail grows"
    );
    assert!(
        !Arc::ptr_eq(
            key_column(&second_records[1]),
            key_column(&grown_records[1])
        ),
        "the grown tail chunk is rebuilt"
    );
    plan.commit(grown);

    let fourth = plan
        .update(
            &keyed_input(&[initial_keys[1].clone()], 100),
            &context,
            NAME,
        )
        .await
        .unwrap();
    let fourth_records = fourth.records.clone();
    assert_shared_key_columns(&grown_records, &fourth_records, "after tail growth");
    plan.commit(fourth);

    let mut expected = (0..=(SEALED_ROWS + TAIL_ROWS))
        .map(|index| (format!("k{index:05}"), 1))
        .collect::<std::collections::BTreeMap<_, _>>();
    expected.insert(initial_keys[0].clone(), 11);
    expected.insert(initial_keys[1].clone(), 101);
    expected.insert(initial_keys[5].clone(), 11);
    expected.insert(initial_keys[SEALED_ROWS + TAIL_ROWS - 1].clone(), 11);
    assert_eq!(totals(&fourth_records), expected);
}
