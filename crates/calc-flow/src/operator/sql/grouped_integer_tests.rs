use super::*;
use crate::{CancellationToken, DataFusionConfig, JsonMap, StreamJobContext};
use datafusion::arrow::array::{Int64Array, StringArray};
use std::collections::BTreeMap;

fn record(unsigned: bool, native_key: bool, phase: usize) -> RecordBatch {
    let values = match phase {
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
    };
    let keys = (0..values.len())
        .map(|index| i64::try_from(index).unwrap())
        .collect::<Vec<_>>();
    let values: ArrayRef = if unsigned {
        Arc::new(datafusion::arrow::array::UInt64Array::from(values))
    } else {
        Arc::new(Int64Array::from(
            values
                .into_iter()
                .map(|value| value.map(|value| i64::try_from(value).unwrap()))
                .collect::<Vec<_>>(),
        ))
    };
    let key: ArrayRef = if native_key {
        Arc::new(Int64Array::from(keys.clone()))
    } else {
        Arc::new(StringArray::from(
            keys.iter()
                .map(|key| (*key != 2).then(|| format!("account-{key}")))
                .collect::<Vec<_>>(),
        ))
    };
    let enabled = keys
        .iter()
        .map(|key| match phase {
            2 => *key % 5 != 0,
            5 => *key % 7 != 0,
            _ => *key < 3 || phase == 0,
        })
        .collect::<Vec<_>>();
    let mut columns = vec![("key", key)];
    if !native_key {
        columns.push((
            "stock",
            Arc::new(StringArray::from(vec![Some("stock"); keys.len()])) as ArrayRef,
        ));
    }
    columns.extend([
        ("value", values),
        ("enabled", Arc::new(BooleanArray::from(enabled)) as ArrayRef),
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
