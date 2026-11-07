//! Tests for delegated delta aggregation: admission, mini-batch shape, and
//! content equivalence with the serial kernel.

use super::*;
use crate::{BatchMetadata, CancellationToken, DataFusionConfig, JsonMap, StreamJobContext};
use datafusion::arrow::array::{Int64Array, StringArray};
use datafusion::arrow::datatypes::{DataType, Field, Schema};

const ROWS: usize = 150_000;
const GROUPS: usize = 3_000;

fn input() -> Batch {
    let keys = (0..ROWS)
        .map(|row| format!("k{:05}", row % GROUPS))
        .collect::<Vec<_>>();
    let values = (0..ROWS)
        .map(|row| i64::try_from(row % 97 + 1).unwrap())
        .collect::<Vec<_>>();
    let schema = Arc::new(Schema::new(vec![
        Field::new("side", DataType::Utf8, true),
        Field::new("key", DataType::Utf8, true),
        Field::new("value", DataType::Int64, true),
    ]));
    let sides = (0..ROWS)
        .map(|row| (row % 2 == 0).then_some("buy"))
        .collect::<Vec<_>>();
    let record = RecordBatch::try_new(
        schema,
        vec![
            Arc::new(StringArray::from(sides)) as ArrayRef,
            Arc::new(StringArray::from(keys)) as ArrayRef,
            Arc::new(Int64Array::from(values)) as ArrayRef,
        ],
    )
    .unwrap();
    Batch::table(vec![record], BatchMetadata::default()).unwrap()
}

fn runtime(partitions: usize) -> DataFusionRuntime {
    DataFusionRuntime::new(DataFusionConfig {
        target_partitions: partitions,
        min_rows_per_partition: 1,
        ..DataFusionConfig::default()
    })
    .unwrap()
}

async fn plan(engine: &DataFusionRuntime, batch: &Batch) -> IncrementalSql {
    let query = crate::expression::parse_select_query(
        "SELECT key, SUM(value) AS total FROM events GROUP BY key",
    )
    .unwrap();
    IncrementalSql::plan(
        engine,
        &query,
        "events",
        batch.table_payload().unwrap().schema().clone(),
        "delegation",
    )
    .await
    .unwrap()
    .unwrap()
}

fn context() -> StreamJobContext {
    StreamJobContext::new(
        974,
        "delegation",
        JsonMap::new(),
        None,
        CancellationToken::new(),
    )
}

fn totals(records: &[RecordBatch]) -> std::collections::BTreeMap<String, i64> {
    let mut totals = std::collections::BTreeMap::new();
    for record in records {
        let keys = record
            .column(0)
            .as_any()
            .downcast_ref::<StringArray>()
            .unwrap();
        let values = record
            .column(1)
            .as_any()
            .downcast_ref::<Int64Array>()
            .unwrap();
        for row in 0..record.num_rows() {
            totals.insert(keys.value(row).to_owned(), values.value(row));
        }
    }
    totals
}

#[tokio::test]
async fn large_batches_delegate_and_match_the_serial_kernel() {
    let engine = runtime(4);
    let input = input();
    let job = context();
    let operator_context = StreamOperatorContext::new(&job, "delegation", None);

    let mut delegated = plan(&engine, &input).await;
    let Some(mini) = grouped_delegation::pre_aggregate(
        &delegated,
        &engine,
        &input,
        &operator_context,
        "delegation",
    )
    .await
    .unwrap() else {
        panic!("a 150k-row fixed-p4 inline-SUM batch must delegate");
    };
    assert_eq!(mini.num_rows(), GROUPS);
    assert_eq!(
        mini.table_payload().unwrap().schema(),
        input.table_payload().unwrap().schema(),
        "the mini-batch carries the exact input schema"
    );
    let first = delegated
        .update(&mini, &operator_context, "delegation")
        .await
        .unwrap();
    let delegated_records = first.records.clone();
    delegated.commit(first);

    let mut serial = plan(&engine, &input).await;
    serial.disable_delegation_for_test();
    let first = serial
        .update(&input, &operator_context, "delegation")
        .await
        .unwrap();
    let serial_records = first.records.clone();
    serial.commit(first);

    assert_eq!(totals(&delegated_records), totals(&serial_records));
    let mut expected = std::collections::BTreeMap::new();
    for row in 0..ROWS {
        *expected
            .entry(format!("k{:05}", row % GROUPS))
            .or_insert(0_i64) += i64::try_from(row % 97 + 1).unwrap();
    }
    assert_eq!(totals(&delegated_records), expected);
}

#[tokio::test]
async fn small_batches_and_single_partitions_stay_serial() {
    let engine = runtime(4);
    let job = context();
    let operator_context = StreamOperatorContext::new(&job, "delegation", None);
    let input = input();

    let keys = (0..1_000).map(|row| format!("k{row}")).collect::<Vec<_>>();
    let schema = Arc::new(Schema::new(vec![
        Field::new("key", DataType::Utf8, true),
        Field::new("value", DataType::Int64, true),
    ]));
    let record = RecordBatch::try_new(
        schema,
        vec![
            Arc::new(StringArray::from(keys)) as ArrayRef,
            Arc::new(Int64Array::from(vec![1_i64; 1_000])) as ArrayRef,
        ],
    )
    .unwrap();
    let small_batch = Batch::table(vec![record], BatchMetadata::default()).unwrap();
    let small = plan(&engine, &small_batch).await;
    assert!(
        grouped_delegation::pre_aggregate(
            &small,
            &engine,
            &small_batch,
            &operator_context,
            "delegation"
        )
        .await
        .unwrap()
        .is_none(),
        "batches below the delegation floor stay serial"
    );

    let single = runtime(1);
    let pinned = plan(&single, &input).await;
    assert!(
        grouped_delegation::pre_aggregate(
            &pinned,
            &single,
            &input,
            &operator_context,
            "delegation"
        )
        .await
        .unwrap()
        .is_none(),
        "a single effective partition stays serial"
    );

    let mut disabled = plan(&engine, &input).await;
    disabled.disable_delegation_for_test();
    assert!(
        grouped_delegation::pre_aggregate(
            &disabled,
            &engine,
            &input,
            &operator_context,
            "delegation"
        )
        .await
        .unwrap()
        .is_none(),
        "the test switch keeps the serial kernel"
    );
}
