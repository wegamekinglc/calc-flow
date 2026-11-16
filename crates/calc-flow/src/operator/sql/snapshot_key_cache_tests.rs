//! Complete snapshot-chunk reuse tests: dirty 8,192-group chunks rebuild
//! their projected columns, while clean chunks share every array across
//! snapshots. Cache installation is commit-scoped.

use super::*;
use crate::{CancellationToken, DataFusionConfig, JsonMap, StreamJobContext};
use datafusion::arrow::array::{Array, Int64Array, StringArray};

const NAME: &str = "key-cache";
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

fn total_column(record: &RecordBatch) -> &ArrayRef {
    record.columns().get(1).expect("SUM column")
}

fn assert_shared_chunk(before: &RecordBatch, after: &RecordBatch, label: &str) {
    assert_eq!(before.num_rows(), after.num_rows(), "{label} row count");
    for (ordinal, (before, after)) in before.columns().iter().zip(after.columns()).enumerate() {
        assert!(
            Arc::ptr_eq(before, after),
            "{label} column {ordinal} must be shared"
        );
    }
}

fn assert_rebuilt_chunk(before: &RecordBatch, after: &RecordBatch, label: &str) {
    assert!(
        !Arc::ptr_eq(key_column(before), key_column(after)),
        "{label} key column must be rebuilt"
    );
    assert!(
        !Arc::ptr_eq(total_column(before), total_column(after)),
        "{label} SUM column must be rebuilt"
    );
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
async fn snapshots_reuse_clean_chunks_and_rebuild_dirty_chunks() {
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

    let touched = vec![initial_keys[0].clone(), initial_keys[5].clone()];
    let second = plan
        .update(&keyed_input(&touched, 10), &context, NAME)
        .await
        .unwrap();
    let second_records = second.records.clone();
    assert_rebuilt_chunk(
        &first_records[0],
        &second_records[0],
        "touched sealed chunk",
    );
    assert_shared_chunk(&first_records[1], &second_records[1], "clean tail");
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
    assert_shared_chunk(&second_records[0], &grown_records[0], "clean sealed chunk");
    assert_rebuilt_chunk(&second_records[1], &grown_records[1], "grown tail");
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
    assert_rebuilt_chunk(
        &grown_records[0],
        &fourth_records[0],
        "second sealed update",
    );
    assert_shared_chunk(&grown_records[1], &fourth_records[1], "clean grown tail");
    plan.commit(fourth);

    let empty = plan
        .update(&keyed_input(&[], 0), &context, NAME)
        .await
        .unwrap();
    let empty_records = empty.records.clone();
    for (chunk, (before, after)) in fourth_records.iter().zip(&empty_records).enumerate() {
        assert_shared_chunk(before, after, &format!("empty input chunk {chunk}"));
    }
    plan.commit(empty);

    let unchanged = plan
        .update(&keyed_input(&[initial_keys[0].clone()], 0), &context, NAME)
        .await
        .unwrap();
    let unchanged_records = unchanged.records.clone();
    for (chunk, (before, after)) in empty_records.iter().zip(&unchanged_records).enumerate() {
        assert_shared_chunk(before, after, &format!("unchanged value chunk {chunk}"));
    }
    plan.commit(unchanged);

    let mut expected = (0..=SEALED_ROWS + TAIL_ROWS)
        .map(|index| (format!("k{index:05}"), 1))
        .collect::<std::collections::BTreeMap<_, _>>();
    expected.insert(initial_keys[0].clone(), 11);
    expected.insert(initial_keys[1].clone(), 101);
    expected.insert(initial_keys[5].clone(), 11);
    assert_eq!(totals(&unchanged_records), expected);
}

#[tokio::test]
async fn discarded_growth_does_not_install_tail_columns() {
    let initial_keys = (0..SEALED_ROWS + TAIL_ROWS)
        .map(|index| format!("k{index:05}"))
        .collect::<Vec<_>>();
    let input = keyed_input(&initial_keys, 1);
    let mut plan = plan_for(&input).await;
    let job = context_for();
    let context = StreamOperatorContext::new(&job, NAME, None);
    let first = plan.update(&input, &context, NAME).await.unwrap();
    let first_records = first.records.clone();
    plan.commit(first);

    let discarded = plan
        .update(
            &keyed_input(&[format!("k{:05}", SEALED_ROWS + TAIL_ROWS)], 1),
            &context,
            NAME,
        )
        .await
        .unwrap();
    let discarded_records = discarded.records.clone();
    assert_eq!(discarded_records[1].num_rows(), TAIL_ROWS + 1);
    assert_shared_chunk(
        &first_records[0],
        &discarded_records[0],
        "clean discarded chunk",
    );
    assert_rebuilt_chunk(
        &first_records[1],
        &discarded_records[1],
        "discarded tail growth",
    );
    drop(discarded);

    let retried = plan
        .update(&keyed_input(&[initial_keys[0].clone()], 10), &context, NAME)
        .await
        .unwrap();
    let retried_records = retried.records.clone();
    assert_rebuilt_chunk(&first_records[0], &retried_records[0], "retry update");
    assert_shared_chunk(
        &first_records[1],
        &retried_records[1],
        "committed tail after rollback",
    );
    let mut expected = (0..SEALED_ROWS + TAIL_ROWS)
        .map(|index| (format!("k{index:05}"), 1))
        .collect::<std::collections::BTreeMap<_, _>>();
    expected.insert(initial_keys[0].clone(), 11);
    assert_eq!(totals(&retried_records), expected);
    plan.commit(retried);
}
