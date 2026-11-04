use super::*;
use crate::{CancellationToken, DataFusionConfig, JsonMap, StreamJobContext};
use datafusion::arrow::array::{Int64Array, StringArray};
use std::collections::BTreeMap;

fn phase_values(unsigned: bool, phase: usize) -> Vec<Option<u64>> {
    match phase {
        0 => vec![
            Some(u64::MAX >> usize::from(!unsigned)),
            None,
            None,
            Some(5),
        ],
        1 => vec![Some(1), Some(7), None, Some(99), Some(99)],
        2 => (0..=CHUNK_ROWS)
            .map(|index| (index % 3 != 0).then_some(2))
            .collect(),
        5 => (0..=CHUNK_ROWS)
            .map(|index| (index % 4 != 0).then_some(3))
            .collect(),
        3 => vec![],
        _ => vec![None, Some(2), None, Some(99)],
    }
}

fn value_array(unsigned: bool, values: Vec<Option<u64>>) -> ArrayRef {
    if unsigned {
        Arc::new(datafusion::arrow::array::UInt64Array::from(values))
    } else {
        Arc::new(Int64Array::from(
            values
                .into_iter()
                .map(|value| value.map(|value| i64::try_from(value).unwrap()))
                .collect::<Vec<_>>(),
        ))
    }
}

fn key_array(native_key: bool, keys: &[i64]) -> ArrayRef {
    if native_key {
        Arc::new(Int64Array::from(keys.to_vec()))
    } else {
        Arc::new(StringArray::from(
            keys.iter()
                .map(|key| (*key != 2).then(|| format!("account-{key}")))
                .collect::<Vec<_>>(),
        ))
    }
}

fn enabled_array(keys: &[i64], phase: usize) -> BooleanArray {
    BooleanArray::from(
        keys.iter()
            .map(|key| match phase {
                2 => *key % 5 != 0,
                5 => *key % 7 != 0,
                _ => *key < 3 || phase == 0,
            })
            .collect::<Vec<_>>(),
    )
}

fn record(unsigned: bool, native_key: bool, phase: usize) -> RecordBatch {
    let values = phase_values(unsigned, phase);
    let keys = (0..values.len())
        .map(|index| i64::try_from(index).unwrap())
        .collect::<Vec<_>>();
    let mut columns = vec![("key", key_array(native_key, &keys))];
    if !native_key {
        columns.push((
            "stock",
            Arc::new(StringArray::from(vec![Some("stock"); keys.len()])) as ArrayRef,
        ));
    }
    columns.extend([
        ("value", value_array(unsigned, values)),
        ("enabled", Arc::new(enabled_array(&keys, phase)) as ArrayRef),
    ]);
    let schema = Arc::new(Schema::new(
        columns
            .iter()
            .map(|(name, array)| Field::new(*name, array.data_type().clone(), true))
            .collect::<Vec<_>>(),
    ));
    RecordBatch::try_new(
        schema,
        columns.into_iter().map(|(_, array)| array).collect(),
    )
    .unwrap()
}

fn rows(records: &[RecordBatch]) -> Vec<Vec<String>> {
    let mut rows = records
        .iter()
        .flat_map(|record| {
            (0..record.num_rows()).map(|row| {
                record
                    .columns()
                    .iter()
                    .map(|column| {
                        format!("{:?}", ScalarValue::try_from_array(column, row).unwrap())
                    })
                    .collect::<Vec<_>>()
            })
        })
        .collect::<Vec<_>>();
    rows.sort();
    rows
}

#[tokio::test]
async fn test_grouped_integer_sum_uses_column_merge_with_exact_prefix_results() {
    for unsigned in [false, true] {
        for native_key in [false, true] {
            let runtime = DataFusionRuntime::new(DataFusionConfig::default()).unwrap();
            let group = if native_key { "key" } else { "key, stock" };
            let sql = format!(
                "SELECT {group}, SUM(value) FILTER (WHERE enabled) AS total, COUNT(value) AS valid FROM events GROUP BY {group}"
            );
            let query = crate::expression::parse_select_query(&sql).unwrap();
            let first = record(unsigned, native_key, 0);
            let mut plan =
                IncrementalSql::plan(&runtime, &query, "events", first.schema(), "integer")
                    .await
                    .unwrap()
                    .unwrap();
            let job =
                StreamJobContext::new(1, "integer", JsonMap::new(), None, CancellationToken::new());
            let context = StreamOperatorContext::new(&job, "integer", None);
            let mut prefix = Vec::new();
            for phase in 0..6 {
                let record = record(unsigned, native_key, phase);
                prefix.push(record.clone());
                let batch = Batch::table(vec![record], crate::BatchMetadata::default()).unwrap();
                if phase == 5 {
                    let discarded = plan.update(&batch, &context, "integer").await.unwrap();
                    drop(discarded);
                }
                let transaction = plan.update(&batch, &context, "integer").await.unwrap();
                let expected = runtime
                    .sql(
                        &sql,
                        &BTreeMap::from([(
                            "events".into(),
                            Batch::table(prefix.clone(), crate::BatchMetadata::default()).unwrap(),
                        )]),
                        Some("oracle"),
                    )
                    .await
                    .unwrap();
                assert_eq!(
                    rows(&transaction.records),
                    rows(expected.table_payload().unwrap().batches())
                );
                plan.commit(transaction);
            }
            let accumulators = plan
                .candidate_accumulators(plan.groups.first(), "integer")
                .unwrap();
            for (expression, accumulator) in plan.aggregates.iter().zip(accumulators) {
                assert_eq!(
                    accumulator.is_none(),
                    expression.fun().name() == "sum",
                    "integer SUM must merge saved states in columns instead of constructing one scalar accumulator per group"
                );
            }
            drop(plan);
            assert_eq!(runtime.incremental_memory_pool().reserved(), 0);
        }
    }
}

#[tokio::test]
async fn test_grouped_integer_sum_pending_history_does_not_retain_scalar_buffers() {
    for native_key in [false, true] {
        let runtime = DataFusionRuntime::new(DataFusionConfig::default()).unwrap();
        let group = if native_key { "key" } else { "key, stock" };
        let sql = format!(
            "SELECT {group}, SUM(value) AS integer_sum, SUM(floating) AS floating_sum FROM events GROUP BY {group}"
        );
        let query = crate::expression::parse_select_query(&sql).unwrap();
        let first = with_floating_column(&record(false, native_key, 4));
        let mut plan = IncrementalSql::plan(&runtime, &query, "events", first.schema(), "history")
            .await
            .unwrap()
            .unwrap();
        let job =
            StreamJobContext::new(1, "history", JsonMap::new(), None, CancellationToken::new());
        let context = StreamOperatorContext::new(&job, "history", None);
        let initial = Batch::table(vec![first.clone()], crate::BatchMetadata::default()).unwrap();
        let transaction = plan.update(&initial, &context, "history").await.unwrap();
        plan.commit(transaction);
        assert!(plan.sequential.is_some());

        let mut integer_history_bytes = 0;
        let mut floating_history_bytes = 0;
        for previous in &plan.groups {
            let candidate = plan
                .candidate(Some(previous), &previous.key, None, "history")
                .unwrap();
            for (expression, state) in plan.aggregates.iter().zip(&candidate.group.states) {
                let bytes = state.capacity() * size_of::<ScalarValue>();
                if grouped_sum::integer_sum(expression) {
                    integer_history_bytes += bytes;
                } else {
                    floating_history_bytes += bytes;
                }
            }
        }
        assert!(
            floating_history_bytes > 0,
            "sequential float state still needs owned candidate history"
        );
        assert_eq!(
            integer_history_bytes, 0,
            "pending integer SUM candidates must not allocate copied historical scalar buffers"
        );

        let next = with_floating_column(&record(false, native_key, 1));
        let next_batch = Batch::table(vec![next.clone()], crate::BatchMetadata::default()).unwrap();
        let discarded = plan.update(&next_batch, &context, "history").await.unwrap();
        drop(discarded);
        let transaction = plan.update(&next_batch, &context, "history").await.unwrap();
        let expected = runtime
            .sql(
                &sql,
                &BTreeMap::from([(
                    "events".into(),
                    Batch::table(vec![first, next], crate::BatchMetadata::default()).unwrap(),
                )]),
                Some("history-oracle"),
            )
            .await
            .unwrap();
        assert_eq!(
            rows(&transaction.records),
            rows(expected.table_payload().unwrap().batches())
        );
        plan.commit(transaction);
        drop(plan);
        assert_eq!(runtime.incremental_memory_pool().reserved(), 0);
    }
}

fn with_floating_column(record: &RecordBatch) -> RecordBatch {
    let schema = record.schema();
    let values = record.column(schema.index_of("value").unwrap());
    let floating = datafusion::arrow::compute::cast(values, &DataType::Float64).unwrap();
    let mut fields = schema.fields().to_vec();
    fields.push(Arc::new(Field::new("floating", DataType::Float64, true)));
    let mut columns = record.columns().to_vec();
    columns.push(floating);
    RecordBatch::try_new(Arc::new(Schema::new(fields)), columns).unwrap()
}

fn fixture_cell(record: &RecordBatch, column: &str, row: usize) -> ScalarValue {
    ScalarValue::try_from_array(record.column_by_name(column).unwrap(), row).unwrap()
}

#[test]
fn test_grouped_integer_fixture_preserves_phase_boundaries_and_masks() {
    let signed = record(false, false, 0);
    let unsigned = record(true, true, 0);
    assert_eq!(signed.num_rows(), 4);
    assert_eq!(
        fixture_cell(&signed, "value", 0),
        ScalarValue::Int64(Some(i64::MAX))
    );
    assert_eq!(
        fixture_cell(&unsigned, "value", 0),
        ScalarValue::UInt64(Some(u64::MAX))
    );
    assert_eq!(fixture_cell(&signed, "key", 2), ScalarValue::Utf8(None));
    assert_eq!(
        fixture_cell(&unsigned, "key", 2),
        ScalarValue::Int64(Some(2))
    );
    assert_eq!(
        fixture_cell(&signed, "enabled", 3),
        ScalarValue::Boolean(Some(true))
    );
    let next = record(false, false, 1);
    assert_eq!(next.num_rows(), 5);
    assert_eq!(fixture_cell(&next, "value", 1), ScalarValue::Int64(Some(7)));
    assert_eq!(
        fixture_cell(&next, "enabled", 3),
        ScalarValue::Boolean(Some(false))
    );
    let dense = record(false, false, 2);
    assert_eq!(dense.num_rows(), CHUNK_ROWS + 1);
    assert_eq!(fixture_cell(&dense, "value", 0), ScalarValue::Int64(None));
    assert_eq!(
        fixture_cell(&dense, "value", CHUNK_ROWS),
        ScalarValue::Int64(Some(2))
    );
    assert_eq!(
        fixture_cell(&dense, "enabled", 0),
        ScalarValue::Boolean(Some(false))
    );
    assert_eq!(
        fixture_cell(&dense, "stock", CHUNK_ROWS),
        ScalarValue::Utf8(Some("stock".into()))
    );
    let empty = record(false, false, 3);
    assert_eq!(empty.num_rows(), 0);
    assert!(
        empty
            .schema()
            .fields()
            .iter()
            .all(|field| field.is_nullable())
    );
    let sparse = record(false, false, 4);
    assert_eq!(sparse.num_rows(), 4);
    assert_eq!(fixture_cell(&sparse, "value", 0), ScalarValue::Int64(None));
    assert_eq!(
        fixture_cell(&sparse, "value", 1),
        ScalarValue::Int64(Some(2))
    );
    let updated = record(false, false, 5);
    assert_eq!(updated.num_rows(), CHUNK_ROWS + 1);
    assert_eq!(
        fixture_cell(&updated, "value", 1),
        ScalarValue::Int64(Some(3))
    );
    assert_eq!(
        fixture_cell(&updated, "value", CHUNK_ROWS),
        ScalarValue::Int64(None)
    );
    assert_eq!(
        fixture_cell(&updated, "enabled", 0),
        ScalarValue::Boolean(Some(false))
    );
}
