use super::super::incremental::IncrementalSql;
use crate::{
    Batch, BatchMetadata, CancellationToken, DataFusionConfig, DataFusionRuntime, JsonMap,
    StreamJobContext, StreamOperatorContext,
};
use datafusion::arrow::{
    array::{ArrayRef, BooleanArray, Int64Array},
    datatypes::{DataType, Field, Schema, TimeUnit},
    record_batch::RecordBatch,
};
use datafusion::common::ScalarValue;
use std::{collections::BTreeMap, sync::Arc};

const GROUPED: &str = "SELECT key, MIN(value) FILTER (WHERE selected) AS lo, \
                       MAX(value) FILTER (WHERE selected) AS hi FROM events WHERE keep \
                       GROUP BY key HAVING COUNT(*) > 0";
const GLOBAL: &str = "SELECT MIN(value) FILTER (WHERE selected) AS lo, \
                      MAX(value) FILTER (WHERE selected) AS hi FROM events WHERE keep \
                      HAVING COUNT(*) > 0";

fn timestamp(unit: TimeUnit, value: i64, timezone: &Arc<str>) -> ScalarValue {
    let timezone = Some(timezone.clone());
    match unit {
        TimeUnit::Second => ScalarValue::TimestampSecond(Some(value), timezone),
        TimeUnit::Millisecond => ScalarValue::TimestampMillisecond(Some(value), timezone),
        TimeUnit::Microsecond => ScalarValue::TimestampMicrosecond(Some(value), timezone),
        TimeUnit::Nanosecond => ScalarValue::TimestampNanosecond(Some(value), timezone),
    }
}

fn duration(unit: TimeUnit, value: i64) -> ScalarValue {
    match unit {
        TimeUnit::Second => ScalarValue::DurationSecond(Some(value)),
        TimeUnit::Millisecond => ScalarValue::DurationMillisecond(Some(value)),
        TimeUnit::Microsecond => ScalarValue::DurationMicrosecond(Some(value)),
        TimeUnit::Nanosecond => ScalarValue::DurationNanosecond(Some(value)),
    }
}

fn series(data_type: &DataType) -> Vec<ScalarValue> {
    let timezone: Arc<str> = Arc::from("UTC");
    (0..4)
        .map(|value| series_value(data_type, value, &timezone))
        .collect()
}

fn series_value(data_type: &DataType, value: i32, timezone: &Arc<str>) -> ScalarValue {
    match data_type {
        DataType::Boolean => ScalarValue::Boolean(Some(value % 2 == 0)),
        DataType::Date32 => ScalarValue::Date32(Some(value)),
        DataType::Date64 => ScalarValue::Date64(Some(i64::from(value))),
        _ => temporal_value(data_type, i64::from(value), timezone),
    }
}

fn temporal_value(data_type: &DataType, value: i64, timezone: &Arc<str>) -> ScalarValue {
    match data_type {
        DataType::Time32(TimeUnit::Second) => {
            ScalarValue::Time32Second(Some(i32::try_from(value).unwrap()))
        }
        DataType::Time32(TimeUnit::Millisecond) => {
            ScalarValue::Time32Millisecond(Some(i32::try_from(value).unwrap()))
        }
        DataType::Time64(TimeUnit::Microsecond) => ScalarValue::Time64Microsecond(Some(value)),
        DataType::Time64(TimeUnit::Nanosecond) => ScalarValue::Time64Nanosecond(Some(value)),
        DataType::Timestamp(unit, _) => timestamp(*unit, value, timezone),
        DataType::Duration(unit) => duration(*unit, value),
        _ => unreachable!("temporal extrema fixture"),
    }
}

fn temporal_types() -> Vec<DataType> {
    vec![
        DataType::Boolean,
        DataType::Date32,
        DataType::Date64,
        DataType::Time32(TimeUnit::Second),
        DataType::Time32(TimeUnit::Millisecond),
        DataType::Time64(TimeUnit::Microsecond),
        DataType::Time64(TimeUnit::Nanosecond),
        DataType::Timestamp(TimeUnit::Second, Some(Arc::from("UTC"))),
        DataType::Timestamp(TimeUnit::Millisecond, Some(Arc::from("UTC"))),
        DataType::Timestamp(TimeUnit::Microsecond, Some(Arc::from("UTC"))),
        DataType::Timestamp(TimeUnit::Nanosecond, Some(Arc::from("UTC"))),
        DataType::Duration(TimeUnit::Second),
        DataType::Duration(TimeUnit::Millisecond),
        DataType::Duration(TimeUnit::Microsecond),
        DataType::Duration(TimeUnit::Nanosecond),
    ]
}

fn input(data_type: &DataType, phase: usize) -> RecordBatch {
    let values = series(data_type);
    let (keys, values, selected, keep): (Vec<i64>, Vec<_>, Vec<_>, Vec<_>) = match phase {
        0 => (
            vec![1, 1, 1, 2, 3],
            vec![
                values[2].clone(),
                ScalarValue::try_from(data_type).unwrap(),
                values[1].clone(),
                values[3].clone(),
                ScalarValue::try_from(data_type).unwrap(),
            ],
            vec![true, true, false, true, false],
            vec![true, true, true, false, true],
        ),
        1 => (
            vec![1, 2, 3],
            vec![values[1].clone(), values[3].clone(), values[0].clone()],
            vec![true, true, false],
            vec![true, true, true],
        ),
        _ => unreachable!("temporal extrema phase"),
    };
    let value = ScalarValue::iter_to_array(values).unwrap();
    let schema = Arc::new(Schema::new(vec![
        Field::new("key", DataType::Int64, false),
        Field::new("value", data_type.clone(), true),
        Field::new("selected", DataType::Boolean, false),
        Field::new("keep", DataType::Boolean, false),
    ]));
    RecordBatch::try_new(
        schema,
        vec![
            Arc::new(Int64Array::from(keys)) as ArrayRef,
            value,
            Arc::new(BooleanArray::from(selected)) as ArrayRef,
            Arc::new(BooleanArray::from(keep)) as ArrayRef,
        ],
    )
    .unwrap()
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

async fn plan(runtime: &DataFusionRuntime, sql: &str, schema: &Arc<Schema>) -> IncrementalSql {
    let query = crate::expression::parse_select_query(sql).unwrap();
    IncrementalSql::plan(
        runtime,
        &query,
        "events",
        schema.clone(),
        "temporal-extrema",
    )
    .await
    .unwrap()
    .expect("temporal MIN/MAX must remain compact")
}

fn job() -> StreamJobContext {
    StreamJobContext::new(
        991,
        "temporal-extrema",
        JsonMap::new(),
        None,
        CancellationToken::new(),
    )
}

async fn oracle(
    runtime: &DataFusionRuntime,
    sql: &str,
    history: &[RecordBatch],
) -> Vec<Vec<ScalarValue>> {
    let input = Batch::table(history.to_vec(), BatchMetadata::default()).unwrap();
    let expected = runtime
        .sql(
            sql,
            &BTreeMap::from([("events".into(), input)]),
            Some("temporal-oracle"),
        )
        .await
        .unwrap();
    rows(expected.table_payload().unwrap().batches())
}

#[tokio::test]
async fn test_temporal_extrema_match_datafusion_global_and_grouped() {
    for data_type in temporal_types() {
        for sql in [GROUPED, GLOBAL] {
            let runtime = DataFusionRuntime::new(DataFusionConfig::default()).unwrap();
            let first = input(&data_type, 0);
            let mut plan = plan(&runtime, sql, &first.schema()).await;
            let job = job();
            let context = StreamOperatorContext::new(&job, "temporal-extrema", None);
            let mut history = Vec::new();
            for phase in 0..3 {
                let record = if phase == 2 {
                    first.slice(0, 0)
                } else {
                    input(&data_type, phase)
                };
                history.push(record.clone());
                let batch = Batch::table(vec![record], BatchMetadata::default()).unwrap();
                let transaction = plan
                    .update(&batch, &context, "temporal-extrema")
                    .await
                    .unwrap();
                assert_eq!(
                    rows(&transaction.records),
                    oracle(&runtime, sql, &history).await,
                    "{sql} over {data_type:?}, phase {phase}"
                );
                plan.commit(transaction);
            }
            drop(plan);
            assert_eq!(runtime.incremental_memory_pool().reserved(), 0);
        }
    }
}

#[tokio::test]
async fn test_temporal_extrema_restore_and_discarded_update_are_atomic() {
    for data_type in [
        DataType::Boolean,
        DataType::Date32,
        DataType::Timestamp(TimeUnit::Microsecond, Some(Arc::from("UTC"))),
        DataType::Duration(TimeUnit::Nanosecond),
    ] {
        let runtime = DataFusionRuntime::new(DataFusionConfig::default()).unwrap();
        let first = input(&data_type, 0);
        let second = input(&data_type, 1);
        let mut original = plan(&runtime, GROUPED, &first.schema()).await;
        let job = job();
        let context = StreamOperatorContext::new(&job, "temporal-extrema", None);
        let initial = Batch::table(vec![first.clone()], BatchMetadata::default()).unwrap();
        let transaction = original
            .update(&initial, &context, "temporal-extrema")
            .await
            .unwrap();
        original.commit(transaction);
        let before = original
            .export_native_state("temporal-extrema", || Ok(()))
            .unwrap();
        let next = Batch::table(vec![second.clone()], BatchMetadata::default()).unwrap();
        let discarded = original
            .update(&next, &context, "temporal-extrema")
            .await
            .unwrap();
        drop(discarded);
        let after = original
            .export_native_state("temporal-extrema", || Ok(()))
            .unwrap();
        assert_eq!(before.records(), after.records());

        let transaction = original
            .update(&next, &context, "temporal-extrema")
            .await
            .unwrap();
        original.commit(transaction);
        let exported = original
            .export_native_state("temporal-extrema", || Ok(()))
            .unwrap();
        let mut restored = plan(&runtime, GROUPED, &first.schema())
            .await
            .import_native_state(
                exported.records(),
                u64::try_from(first.num_rows() + second.num_rows()).unwrap(),
                true,
                || Ok(()),
                "temporal-extrema",
            )
            .unwrap();
        let reexported = restored
            .export_native_state("temporal-extrema", || Ok(()))
            .unwrap();
        assert_eq!(exported.records(), reexported.records());
        let empty = Batch::table(vec![first.slice(0, 0)], BatchMetadata::default()).unwrap();
        let left = original
            .update(&empty, &context, "temporal-extrema")
            .await
            .unwrap();
        let right = restored
            .update(&empty, &context, "temporal-extrema")
            .await
            .unwrap();
        assert_eq!(left.records, right.records);
        original.commit(left);
        restored.commit(right);
        drop((before, after, exported, reexported, original, restored));
        assert_eq!(runtime.incremental_memory_pool().reserved(), 0);
    }
}
