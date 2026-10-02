use std::sync::Arc;

use datafusion::arrow::{
    array::{Array, Int64Array},
    datatypes::DataType,
};

use super::*;
use crate::{CancellationToken, EdgeCollector, StreamJobContext};

#[path = "decimal_tests.rs"]
mod decimal_tests;

#[path = "retained_tests.rs"]
mod retained_tests;

#[path = "retained_planning_tests.rs"]
mod retained_planning_tests;

fn batch(values: &[i64]) -> Batch {
    Batch::table(
        vec![
            RecordBatch::try_from_iter(vec![(
                "value",
                Arc::new(Int64Array::from(values.to_vec())) as Arc<dyn Array>,
            )])
            .unwrap(),
        ],
        BatchMetadata::default(),
    )
    .unwrap()
}

#[tokio::test]
async fn test_sql_incremental_updates_only_new_rows_and_reuses_plan() {
    let query = "SELECT COUNT(*) AS rows, SUM(value) AS total, MIN(value) AS lo, MAX(value) AS hi FROM events";
    let mut operator = SqlOperator::new("totals", query, vec!["events".into()], vec![]).unwrap();
    let runtime = DataFusionRuntime::new(DataFusionConfig::default()).unwrap();
    let job = StreamJobContext::new(
        1,
        "incremental",
        JsonMap::new(),
        None,
        CancellationToken::new(),
    );
    let context = StreamOperatorContext::new(&job, "totals", None);
    let mut collector = EdgeCollector::new(operator.output_ports().to_vec());
    let mut prefix = Vec::new();
    for values in [&[1, 2][..], &[3][..]] {
        prefix.extend_from_slice(values);
        let expected = runtime
            .sql(
                query,
                &BTreeMap::from([("events".into(), batch(&prefix))]),
                Some("oracle"),
            )
            .await
            .unwrap();
        operator
            .process_data("events", batch(values), &context, &mut collector)
            .await
            .unwrap();
        let output = collector.drain("output");
        let actual = output[0].as_data().unwrap().table_payload().unwrap();
        assert_eq!(actual.schema(), expected.table_payload().unwrap().schema());
        assert_eq!(
            actual.batches(),
            expected.table_payload().unwrap().batches()
        );
    }
    assert_eq!(
        operator.incremental_work,
        (3, 1),
        "eligible aggregate must scan new rows only and plan once"
    );
}

fn grouped_batch(keys: &[Option<&str>], values: &[Option<i64>]) -> Batch {
    Batch::table(
        vec![
            RecordBatch::try_new(
                Arc::new(datafusion::arrow::datatypes::Schema::new(vec![
                    datafusion::arrow::datatypes::Field::new("key", DataType::Utf8, true),
                    datafusion::arrow::datatypes::Field::new("value", DataType::Int64, true),
                ])),
                vec![
                    Arc::new(datafusion::arrow::array::StringArray::from(keys.to_vec()))
                        as Arc<dyn Array>,
                    Arc::new(Int64Array::from(values.to_vec())) as Arc<dyn Array>,
                ],
            )
            .unwrap(),
        ],
        BatchMetadata::default(),
    )
    .unwrap()
}

fn rows(batch: &Batch) -> Vec<Vec<datafusion::common::ScalarValue>> {
    let mut rows = batch
        .table_payload()
        .unwrap()
        .batches()
        .iter()
        .flat_map(|record| {
            (0..record.num_rows())
                .map(|row| {
                    record
                        .columns()
                        .iter()
                        .map(|column| {
                            datafusion::common::ScalarValue::try_from_array(column, row).unwrap()
                        })
                        .collect::<Vec<_>>()
                })
                .collect::<Vec<_>>()
        })
        .collect::<Vec<_>>();
    rows.sort_by(|left, right| left.partial_cmp(right).unwrap());
    rows
}

#[tokio::test]
async fn test_sql_incremental_grouped_prefixes_touch_only_new_input() {
    let query = "SELECT MAX(value) AS hi, key AS name, COUNT(value) AS valid, SUM(value) AS total, MIN(value) AS lo, COUNT(*) AS rows, SUM(value) AS again FROM events GROUP BY key";
    let mut operator = SqlOperator::new("totals", query, vec!["events".into()], vec![]).unwrap();
    let runtime = DataFusionRuntime::new(DataFusionConfig::default()).unwrap();
    let job = StreamJobContext::new(1, "grouped", JsonMap::new(), None, CancellationToken::new());
    let context = StreamOperatorContext::new(&job, "totals", None);
    let mut collector = EdgeCollector::new(operator.output_ports().to_vec());
    let mut keys = Vec::new();
    let mut values = Vec::new();
    for (incoming_keys, incoming_values) in [
        (
            vec![Some("a"), Some("b"), None],
            vec![Some(1), None, Some(9)],
        ),
        (
            vec![Some("a"), Some("c"), None],
            vec![Some(2), Some(4), None],
        ),
        (vec![Some("a")], vec![Some(3)]),
    ] {
        keys.extend_from_slice(&incoming_keys);
        values.extend_from_slice(&incoming_values);
        let expected = runtime
            .sql(
                query,
                &BTreeMap::from([("events".into(), grouped_batch(&keys, &values))]),
                Some("oracle"),
            )
            .await
            .unwrap();
        operator
            .process_data(
                "events",
                grouped_batch(&incoming_keys, &incoming_values),
                &context,
                &mut collector,
            )
            .await
            .unwrap();
        let output = collector.drain("output");
        let actual = output[0].as_data().unwrap();
        assert_eq!(
            actual.table_payload().unwrap().schema(),
            expected.table_payload().unwrap().schema()
        );
        assert_eq!(rows(actual), rows(&expected));
    }
    assert_eq!(operator.incremental_work, (7, 1));
}

#[tokio::test]
async fn test_sql_incremental_restore_rebuilds_without_extra_output() {
    let query = "SELECT key, COUNT(*) AS rows, SUM(value) AS total FROM events GROUP BY key";
    let mut initial = SqlOperator::new("totals", query, vec!["events".into()], vec![]).unwrap();
    let job = StreamJobContext::new(1, "restore", JsonMap::new(), None, CancellationToken::new());
    let context = StreamOperatorContext::new(&job, "totals", None);
    let mut collector = EdgeCollector::new(initial.output_ports().to_vec());
    initial
        .process_data(
            "events",
            grouped_batch(&[Some("a"), Some("b")], &[Some(1), Some(4)]),
            &context,
            &mut collector,
        )
        .await
        .unwrap();
    collector.drain("output");
    let snapshot = initial.checkpoint(Epoch::INITIAL).unwrap();
    let mut restored = initial.clone();
    StreamOperator::restore(&mut restored, &snapshot).unwrap();
    restored
        .process_data(
            "events",
            grouped_batch(&[Some("a"), Some("c")], &[Some(2), Some(8)]),
            &context,
            &mut collector,
        )
        .await
        .unwrap();
    let output = collector.drain("output");
    assert_eq!(output.len(), 1);
    let runtime = DataFusionRuntime::new(DataFusionConfig::default()).unwrap();
    let expected = runtime
        .sql(
            query,
            &BTreeMap::from([(
                "events".into(),
                grouped_batch(
                    &[Some("a"), Some("b"), Some("a"), Some("c")],
                    &[Some(1), Some(4), Some(2), Some(8)],
                ),
            )]),
            None,
        )
        .await
        .unwrap();
    assert_eq!(rows(output[0].as_data().unwrap()), rows(&expected));
    assert_eq!(restored.incremental_work, (4, 1));
    restored
        .process_data(
            "events",
            grouped_batch(&[Some("a")], &[Some(3)]),
            &context,
            &mut collector,
        )
        .await
        .unwrap();
    assert_eq!(restored.incremental_work, (5, 1));
}

#[tokio::test]
async fn test_sql_incremental_retention_does_not_copy_historical_handles() {
    let mut operator = SqlOperator::new(
        "totals",
        "SELECT SUM(value) AS total FROM events",
        vec!["events".into()],
        vec![],
    )
    .unwrap();
    let job = StreamJobContext::new(1, "append", JsonMap::new(), None, CancellationToken::new());
    let context = StreamOperatorContext::new(&job, "totals", None);
    let mut collector = EdgeCollector::new(operator.output_ports().to_vec());
    for value in 0..20 {
        operator
            .process_data("events", batch(&[value]), &context, &mut collector)
            .await
            .unwrap();
        collector.drain("output");
    }
    assert_eq!(
        operator
            .retained_handles_copied
            .load(std::sync::atomic::Ordering::SeqCst),
        0
    );
    let snapshot = operator.checkpoint(Epoch::INITIAL).unwrap();
    assert_eq!(
        decode_sql_state(snapshot.segments["input"].bytes())
            .unwrap()
            .num_rows(),
        20
    );
}

async fn assert_prefix_oracle(query: &str, incoming: &[Batch], eligible: bool) -> SqlOperator {
    let mut operator = SqlOperator::new("totals", query, vec!["events".into()], vec![]).unwrap();
    let runtime = DataFusionRuntime::new(DataFusionConfig::default()).unwrap();
    let job = StreamJobContext::new(1, "prefix", JsonMap::new(), None, CancellationToken::new());
    let context = StreamOperatorContext::new(&job, "totals", None);
    let mut collector = EdgeCollector::new(operator.output_ports().to_vec());
    let mut records = Vec::new();
    for batch in incoming {
        if batch.num_rows() != 0 || records.is_empty() {
            records.extend_from_slice(batch.table_payload().unwrap().batches());
        }
        let cumulative = Batch::table(records.clone(), batch.metadata().clone()).unwrap();
        let expected = runtime
            .sql(
                query,
                &BTreeMap::from([("events".into(), cumulative)]),
                None,
            )
            .await
            .unwrap();
        operator
            .process_data("events", batch.clone(), &context, &mut collector)
            .await
            .unwrap();
        let output = collector.drain("output");
        assert_eq!(output.len(), 1);
        let actual = output[0].as_data().unwrap();
        assert_eq!(
            actual.table_payload().unwrap().schema(),
            expected.table_payload().unwrap().schema(),
            "{query}"
        );
        assert_eq!(rows(actual), rows(&expected), "{query}");
        assert_eq!(actual.metadata().source(), batch.metadata().source());
        assert_eq!(actual.metadata().sequence(), batch.metadata().sequence());
        assert_eq!(
            actual.metadata().attributes(),
            batch.metadata().attributes()
        );
    }
    assert_eq!(operator.incremental.is_some(), eligible, "{query}");
    operator
}

#[tokio::test]
async fn test_sql_incremental_integer_promotions_and_wrapping_prefix_oracles() {
    use datafusion::arrow::array::{
        Int8Array, Int16Array, Int32Array, UInt8Array, UInt16Array, UInt32Array, UInt64Array,
    };
    let arrays: Vec<Arc<dyn Array>> = vec![
        Arc::new(Int8Array::from(vec![
            Some(i8::MAX),
            None,
            Some(1),
            Some(i8::MIN),
        ])),
        Arc::new(Int16Array::from(vec![
            Some(i16::MAX),
            None,
            Some(1),
            Some(i16::MIN),
        ])),
        Arc::new(Int32Array::from(vec![
            Some(i32::MAX),
            None,
            Some(1),
            Some(i32::MIN),
        ])),
        Arc::new(Int64Array::from(vec![
            Some(i64::MAX),
            None,
            Some(1),
            Some(i64::MIN),
        ])),
        Arc::new(UInt8Array::from(vec![
            Some(u8::MAX),
            None,
            Some(1),
            Some(u8::MIN),
        ])),
        Arc::new(UInt16Array::from(vec![
            Some(u16::MAX),
            None,
            Some(1),
            Some(u16::MIN),
        ])),
        Arc::new(UInt32Array::from(vec![
            Some(u32::MAX),
            None,
            Some(1),
            Some(u32::MIN),
        ])),
        Arc::new(UInt64Array::from(vec![
            Some(u64::MAX),
            None,
            Some(1),
            Some(u64::MIN),
        ])),
    ];
    for array in arrays {
        let schema = Arc::new(datafusion::arrow::datatypes::Schema::new(vec![
            datafusion::arrow::datatypes::Field::new("value", array.data_type().clone(), true),
        ]));
        let incoming = [
            Batch::table(
                vec![RecordBatch::try_new(schema.clone(), vec![array.slice(0, 2)]).unwrap()],
                BatchMetadata::default(),
            )
            .unwrap(),
            Batch::table(
                vec![RecordBatch::try_new(schema.clone(), vec![array.slice(2, 2)]).unwrap()],
                BatchMetadata::default(),
            )
            .unwrap(),
            Batch::table(
                vec![RecordBatch::new_empty(schema)],
                BatchMetadata::default(),
            )
            .unwrap(),
        ];
        assert_prefix_oracle("SELECT COUNT(*) AS rows, COUNT(value) AS valid, SUM(value) AS total, MIN(value) AS lo, MAX(value) AS hi FROM events", &incoming, true).await;
    }
}

#[tokio::test]
async fn test_sql_incremental_composite_keys_and_subset_projection() {
    use datafusion::arrow::{
        array::{BooleanArray, LargeStringArray},
        datatypes::{DataType, Field, Schema},
    };
    let schema = Arc::new(Schema::new(vec![
        Field::new("flag", DataType::Boolean, true),
        Field::new("name", DataType::LargeUtf8, true),
        Field::new("value", DataType::Int64, true),
    ]));
    let batch = |flags: Vec<Option<bool>>, names: Vec<Option<&str>>, values: Vec<Option<i64>>| {
        Batch::table(
            vec![
                RecordBatch::try_new(
                    schema.clone(),
                    vec![
                        Arc::new(BooleanArray::from(flags)),
                        Arc::new(LargeStringArray::from(names)),
                        Arc::new(Int64Array::from(values)),
                    ],
                )
                .unwrap(),
            ],
            BatchMetadata::default(),
        )
        .unwrap()
    };
    assert_prefix_oracle(
        "SELECT name AS label, MAX(value) AS hi, SUM(value) AS total, SUM(value) AS duplicate_total, COUNT(name) AS names, COUNT(flag) AS flags FROM events GROUP BY flag, name",
        &[batch(vec![Some(true), Some(false), None], vec![Some("same"), Some("same"), None], vec![Some(1), Some(2), None]), batch(vec![Some(false), None, Some(true)], vec![Some("same"), None, Some("new")], vec![Some(4), Some(5), Some(6)])], true,
    ).await;
}

#[tokio::test]
async fn test_sql_incremental_empty_other_schema_preserves_values_and_incoming_metadata() {
    use datafusion::arrow::{
        array::Float64Array,
        datatypes::{DataType, Field, Schema},
    };
    let different = Arc::new(Schema::new_with_metadata(
        vec![Field::new("other", DataType::Float64, false)],
        std::collections::HashMap::from([("different".into(), "schema".into())]),
    ));
    let empty = Batch::table(
        vec![
            RecordBatch::try_new(
                different,
                vec![Arc::new(Float64Array::from(Vec::<f64>::new()))],
            )
            .unwrap(),
        ],
        BatchMetadata::new(
            "new-empty",
            42,
            JsonMap::from([("marker".into(), json!("last"))]),
        )
        .unwrap(),
    )
    .unwrap();
    assert_prefix_oracle(
        "SELECT COUNT(*) AS rows, SUM(value) AS total FROM events",
        &[batch(&[1, 2]), empty],
        true,
    )
    .await;
}

#[tokio::test]
async fn test_sql_incremental_unsupported_queries_use_whole_query_fallback() {
    for query in [
        "SELECT SUM(value) AS total, AVG(value) AS mean FROM events",
        "SELECT COUNT(DISTINCT value) AS unique_values FROM events",
        "SELECT SUM(value + 1) AS computed FROM events",
        "SELECT SUM(CAST(value AS BIGINT)) AS cast_total FROM events",
        "SELECT SUM(value) AS total FROM events WHERE value > 1",
        "SELECT value, SUM(value) AS total FROM events GROUP BY value HAVING SUM(value) > 1",
        "SELECT value, COUNT(*) AS rows FROM events GROUP BY value ORDER BY value DESC LIMIT 2",
        "WITH source AS (SELECT value FROM events) SELECT SUM(value) AS total FROM source",
        "SELECT SUM(value) FILTER (WHERE value > 1) AS total FROM events",
        "SELECT SUM(value) OVER () AS total FROM events GROUP BY value",
    ] {
        assert_prefix_oracle(query, &[batch(&[1, 2]), batch(&[2, 3])], false).await;
    }
}

#[tokio::test]
async fn test_sql_incremental_float_decimal_queries_preserve_cumulative_engine() {
    use datafusion::arrow::array::{Decimal128Array, Float64Array};
    for (array, eligible) in [
        (
            Arc::new(Float64Array::from(vec![1e16, 1.0, -1e16, 3.0])) as Arc<dyn Array>,
            false,
        ),
        (
            Arc::new(
                Decimal128Array::from(vec![Some(123), None, Some(456), Some(-100)])
                    .with_precision_and_scale(12, 2)
                    .unwrap(),
            ) as Arc<dyn Array>,
            true,
        ),
    ] {
        let schema = Arc::new(datafusion::arrow::datatypes::Schema::new(vec![
            datafusion::arrow::datatypes::Field::new("value", array.data_type().clone(), true),
        ]));
        let incoming = [
            Batch::table(
                vec![RecordBatch::try_new(schema.clone(), vec![array.slice(0, 2)]).unwrap()],
                BatchMetadata::default(),
            )
            .unwrap(),
            Batch::table(
                vec![RecordBatch::try_new(schema, vec![array.slice(2, 2)]).unwrap()],
                BatchMetadata::default(),
            )
            .unwrap(),
        ];
        assert_prefix_oracle(
            "SELECT SUM(value) AS total, AVG(value) AS mean FROM events",
            &incoming,
            eligible,
        )
        .await;
    }
}

#[tokio::test]
async fn test_sql_incremental_unique_groups_fit_bounded_pool() {
    for native in [false, true] {
        let grouped_batch = |keys: &[Option<&str>], values: &[Option<i64>]| {
            if native {
                integer_grouped_fixture(keys, values)
            } else {
                grouped_batch(keys, values)
            }
        };
        let mut operator = SqlOperator::new("totals", "SELECT key, COUNT(*) AS rows, COUNT(value) AS valid, SUM(value) AS total, MIN(value) AS lo, MAX(value) AS hi FROM events GROUP BY key", vec!["events".into()], vec![]).unwrap();
        let pressure = operator
            .stream_state
            .runtime()
            .unwrap()
            .incremental_reservation("pressure");
        pressure.try_grow((1 << 30) - (96 << 20)).unwrap();
        let job = StreamJobContext::new(1, "pool", JsonMap::new(), None, CancellationToken::new());
        let context = StreamOperatorContext::new(&job, "totals", None);
        let mut collector = EdgeCollector::new(operator.output_ports().to_vec());
        let names = (0..10_000)
            .map(|index| format!("key-{index}"))
            .collect::<Vec<_>>();
        let keys = names
            .iter()
            .map(|name| Some(name.as_str()))
            .collect::<Vec<_>>();
        operator
            .process_data(
                "events",
                grouped_batch(&keys, &vec![Some(1); keys.len()]),
                &context,
                &mut collector,
            )
            .await
            .unwrap();
        assert_eq!(
            collector.drain("output")[0].as_data().unwrap().num_rows(),
            10_000
        );
    }
}

#[test]
fn test_sql_incremental_memory_charges_reject_overflow() {
    assert!(
        matches!(incremental::checked_bytes(0, [(usize::MAX, 2)], "overflow"), Err(CalcFlowError::DataFusion { node_id: Some(name), .. }) if name == "overflow")
    );
    assert!(incremental::checked_bytes(usize::MAX, [(1, 1)], "overflow").is_err());
    assert_eq!(
        incremental::checked_bytes(4, [(8, 3), (2, 5)], "charge").unwrap(),
        38
    );
}

#[tokio::test]
async fn test_sql_incremental_count_zero_columns_uses_bounded_workspace() {
    use datafusion::arrow::record_batch::RecordBatchOptions;
    let schema = Arc::new(datafusion::arrow::datatypes::Schema::empty());
    let input = Batch::table(
        vec![
            RecordBatch::try_new_with_options(
                schema,
                vec![],
                &RecordBatchOptions::new().with_row_count(Some(100_000)),
            )
            .unwrap(),
        ],
        BatchMetadata::default(),
    )
    .unwrap();
    let mut operator = SqlOperator::new(
        "totals",
        "SELECT COUNT(*) AS rows FROM events",
        vec!["events".into()],
        vec![],
    )
    .unwrap();
    let pressure = operator
        .stream_state
        .runtime()
        .unwrap()
        .incremental_reservation("pressure");
    pressure.try_grow((1 << 30) - (1 << 20)).unwrap();
    let job = StreamJobContext::new(
        1,
        "zero-column",
        JsonMap::new(),
        None,
        CancellationToken::new(),
    );
    let context = StreamOperatorContext::new(&job, "totals", None);
    let mut collector = EdgeCollector::new(operator.output_ports().to_vec());
    operator
        .process_data("events", input, &context, &mut collector)
        .await
        .unwrap();
    assert_eq!(
        rows(collector.drain("output")[0].as_data().unwrap()),
        vec![vec![datafusion::common::ScalarValue::Int64(Some(100_000))]]
    );
}

#[tokio::test]
async fn test_sql_incremental_count_unused_payload_does_not_charge_workspace() {
    use datafusion::arrow::array::StringArray;
    let value = "x".repeat(1024);
    let input = Batch::table(
        vec![
            RecordBatch::try_from_iter(vec![(
                "unused",
                Arc::new(StringArray::from(vec![value.as_str(); 10_000])) as Arc<dyn Array>,
            )])
            .unwrap(),
        ],
        BatchMetadata::default(),
    )
    .unwrap();
    let mut operator = SqlOperator::new(
        "totals",
        "SELECT COUNT(*) AS rows FROM events",
        vec!["events".into()],
        vec![],
    )
    .unwrap();
    let pressure = operator
        .stream_state
        .runtime()
        .unwrap()
        .incremental_reservation("pressure");
    pressure.try_grow((1 << 30) - (1 << 20)).unwrap();
    let job = StreamJobContext::new(1, "unused", JsonMap::new(), None, CancellationToken::new());
    let context = StreamOperatorContext::new(&job, "totals", None);
    let mut collector = EdgeCollector::new(operator.output_ports().to_vec());
    operator
        .process_data("events", input, &context, &mut collector)
        .await
        .unwrap();
    assert_eq!(
        rows(collector.drain("output")[0].as_data().unwrap()),
        vec![vec![datafusion::common::ScalarValue::Int64(Some(10_000))]]
    );
}

#[tokio::test]
async fn test_sql_incremental_ignored_wide_empty_schema_does_not_reserve_handles() {
    use datafusion::arrow::datatypes::{DataType, Field, Schema};
    let mut operator = SqlOperator::new(
        "totals",
        "SELECT SUM(value) AS total FROM events",
        vec!["events".into()],
        vec![],
    )
    .unwrap();
    let job = StreamJobContext::new(
        1,
        "wide-empty",
        JsonMap::new(),
        None,
        CancellationToken::new(),
    );
    let context = StreamOperatorContext::new(&job, "totals", None);
    let mut collector = EdgeCollector::new(operator.output_ports().to_vec());
    operator
        .process_data("events", batch(&[1, 2]), &context, &mut collector)
        .await
        .unwrap();
    collector.drain("output");
    let before = operator.checkpoint(Epoch::INITIAL).unwrap();
    let pressure = operator
        .stream_state
        .runtime()
        .unwrap()
        .incremental_reservation("pressure");
    pressure.try_grow((1 << 30) - (1 << 20)).unwrap();
    let schema = Arc::new(Schema::new(
        (0..100_000)
            .map(|index| Field::new(format!("ignored-{index}"), DataType::Int64, false))
            .collect::<Vec<_>>(),
    ));
    let array = Arc::new(Int64Array::from(Vec::<i64>::new())) as Arc<dyn Array>;
    let empty = Batch::table(
        vec![RecordBatch::try_new(schema, vec![array; 100_000]).unwrap()],
        BatchMetadata::new("wide-empty", 9, JsonMap::new()).unwrap(),
    )
    .unwrap();
    operator
        .process_data("events", empty, &context, &mut collector)
        .await
        .unwrap();
    let output = collector.drain("output");
    assert_eq!(
        rows(output[0].as_data().unwrap()),
        vec![vec![datafusion::common::ScalarValue::Int64(Some(3))]]
    );
    assert_eq!(
        output[0].as_data().unwrap().metadata().source(),
        "wide-empty"
    );
    let after = operator.checkpoint(Epoch::INITIAL).unwrap();
    assert_eq!(before.inline_metadata, after.inline_metadata);
    assert!(Arc::ptr_eq(
        &before.segments["input"].bytes_arc(),
        &after.segments["input"].bytes_arc()
    ));
}

#[tokio::test]
async fn test_sql_incremental_pending_empty_record_copies_are_reserved() {
    let record = RecordBatch::try_from_iter(vec![(
        "value",
        Arc::new(Int64Array::from(Vec::<i64>::new())) as Arc<dyn Array>,
    )])
    .unwrap();
    let input = Batch::table(vec![record; 12_000], BatchMetadata::default()).unwrap();
    let mut operator = SqlOperator::new(
        "totals",
        "SELECT SUM(value) AS total FROM events",
        vec!["events".into()],
        vec![],
    )
    .unwrap();
    let pressure = operator
        .stream_state
        .runtime()
        .unwrap()
        .incremental_reservation("pressure");
    pressure.try_grow((1 << 30) - (1 << 20)).unwrap();
    let job = StreamJobContext::new(
        1,
        "empty-records",
        JsonMap::new(),
        None,
        CancellationToken::new(),
    );
    let context = StreamOperatorContext::new(&job, "totals", None);
    let mut collector = EdgeCollector::new(operator.output_ports().to_vec());
    assert!(matches!(
        operator
            .process_data("events", input.clone(), &context, &mut collector)
            .await,
        Err(CalcFlowError::DataFusion { .. })
    ));
    assert!(operator.retained.is_none());
    assert!(collector.drain("output").is_empty());
    drop(pressure);
    operator
        .process_data("events", input, &context, &mut collector)
        .await
        .unwrap();
    assert_eq!(
        rows(collector.drain("output")[0].as_data().unwrap()),
        vec![vec![datafusion::common::ScalarValue::Int64(None)]]
    );
}

struct RejectOutput;

#[async_trait]
impl StreamCollector for RejectOutput {
    async fn emit(&mut self, _port: &str, _batch: Batch) -> Result<()> {
        Err(CalcFlowError::Operator {
            node_id: "reject".into(),
            message: "injected emit failure".into(),
        })
    }
}

struct PendingOutput {
    entered: Option<tokio::sync::oneshot::Sender<()>>,
}

#[async_trait]
impl StreamCollector for PendingOutput {
    async fn emit(&mut self, _port: &str, _batch: Batch) -> Result<()> {
        self.entered.take().unwrap().send(()).unwrap();
        std::future::pending().await
    }
}

#[tokio::test]
async fn test_sql_incremental_rejected_new_groups_release_candidates_and_retry_once() {
    for native in [false, true] {
        let grouped_batch = |keys: &[Option<&str>], values: &[Option<i64>]| {
            if native {
                integer_grouped_fixture(keys, values)
            } else {
                grouped_batch(keys, values)
            }
        };
        let query = "SELECT key, COUNT(*) AS rows, SUM(value) AS total FROM events GROUP BY key";
        let mut operator =
            SqlOperator::new("totals", query, vec!["events".into()], vec![]).unwrap();
        let job =
            StreamJobContext::new(1, "reject", JsonMap::new(), None, CancellationToken::new());
        let context = StreamOperatorContext::new(&job, "totals", None);
        let mut collector = EdgeCollector::new(operator.output_ports().to_vec());
        operator
            .process_data(
                "events",
                grouped_batch(&[Some("a"), Some("b")], &[Some(1), Some(4)]),
                &context,
                &mut collector,
            )
            .await
            .unwrap();
        collector.drain("output");
        let before = operator.checkpoint(Epoch::INITIAL).unwrap();
        let names = (0..1000)
            .map(|index| format!("new-{index}"))
            .collect::<Vec<_>>();
        let mut keys = vec![Some("a")];
        keys.extend(names.iter().map(|name| Some(name.as_str())));
        let incoming = grouped_batch(&keys, &vec![Some(2); keys.len()]);
        let mut capacity_charge = None;
        for _ in 0..8 {
            assert!(
                matches!(operator.process_data("events", incoming.clone(), &context, &mut RejectOutput).await, Err(CalcFlowError::Operator { node_id, .. }) if node_id == "reject")
            );
            let current = operator.incremental.as_ref().unwrap().capacity_charge();
            if let Some(previous) = capacity_charge {
                assert_eq!(current, previous);
            }
            capacity_charge = Some(current);
            let after = operator.checkpoint(Epoch::INITIAL).unwrap();
            assert_eq!(before.inline_metadata, after.inline_metadata);
            assert!(Arc::ptr_eq(
                &before.segments["input"].bytes_arc(),
                &after.segments["input"].bytes_arc()
            ));
            let pressure = operator
                .stream_state
                .runtime()
                .unwrap()
                .incremental_reservation("probe");
            pressure.try_grow((1 << 30) - (1 << 20)).unwrap();
        }
        operator
            .process_data("events", incoming, &context, &mut collector)
            .await
            .unwrap();
        let mut all_keys = vec![Some("a"), Some("b")];
        all_keys.extend_from_slice(&keys);
        let mut values = vec![Some(1), Some(4)];
        values.extend(vec![Some(2); keys.len()]);
        let expected = DataFusionRuntime::new(DataFusionConfig::default())
            .unwrap()
            .sql(
                query,
                &BTreeMap::from([("events".into(), grouped_batch(&all_keys, &values))]),
                None,
            )
            .await
            .unwrap();
        assert_eq!(
            rows(collector.drain("output")[0].as_data().unwrap()),
            rows(&expected)
        );
    }
}

#[tokio::test]
async fn test_sql_incremental_dropped_emit_preserves_state_and_retry_once() {
    for native in [false, true] {
        let grouped_batch = |keys: &[Option<&str>], values: &[Option<i64>]| {
            if native {
                integer_grouped_fixture(keys, values)
            } else {
                grouped_batch(keys, values)
            }
        };
        let query = "SELECT key, COUNT(*) AS rows, SUM(value) AS total FROM events GROUP BY key";
        for initially_empty in [true, false] {
            let mut operator =
                SqlOperator::new("totals", query, vec!["events".into()], vec![]).unwrap();
            let job = StreamJobContext::new(
                1,
                "drop-emit",
                JsonMap::new(),
                None,
                CancellationToken::new(),
            );
            let context = StreamOperatorContext::new(&job, "totals", None);
            let mut collector = EdgeCollector::new(operator.output_ports().to_vec());
            if !initially_empty {
                operator
                    .process_data(
                        "events",
                        grouped_batch(&[Some("a"), Some("b")], &[Some(1), Some(4)]),
                        &context,
                        &mut collector,
                    )
                    .await
                    .unwrap();
                collector.drain("output");
            }
            let before = operator.checkpoint(Epoch::INITIAL).unwrap();
            let incoming = grouped_batch(&[Some("a"), Some("c")], &[Some(2), Some(8)]);
            let (entered, started) = tokio::sync::oneshot::channel();
            let mut pending = PendingOutput {
                entered: Some(entered),
            };
            {
                let process =
                    operator.process_data("events", incoming.clone(), &context, &mut pending);
                tokio::pin!(process);
                tokio::select! {
                    result = &mut process => panic!("emit finished unexpectedly: {result:?}"),
                    result = started => result.unwrap(),
                }
            }
            let after = operator.checkpoint(Epoch::INITIAL).unwrap();
            assert_eq!(before.inline_metadata, after.inline_metadata);
            if initially_empty {
                assert!(after.segments.is_empty());
            } else {
                assert!(Arc::ptr_eq(
                    &before.segments["input"].bytes_arc(),
                    &after.segments["input"].bytes_arc()
                ));
            }
            operator
                .process_data("events", incoming, &context, &mut collector)
                .await
                .unwrap();
            let expected_input = if initially_empty {
                grouped_batch(&[Some("a"), Some("c")], &[Some(2), Some(8)])
            } else {
                grouped_batch(
                    &[Some("a"), Some("b"), Some("a"), Some("c")],
                    &[Some(1), Some(4), Some(2), Some(8)],
                )
            };
            let expected = DataFusionRuntime::new(DataFusionConfig::default())
                .unwrap()
                .sql(
                    query,
                    &BTreeMap::from([("events".into(), expected_input)]),
                    None,
                )
                .await
                .unwrap();
            assert_eq!(
                rows(collector.drain("output")[0].as_data().unwrap()),
                rows(&expected)
            );
        }
    }
}

#[tokio::test]
async fn test_sql_incremental_cancel_stops_group_finalization_and_rolls_back() {
    for native in [false, true] {
        let grouped_batch = |keys: &[Option<&str>], values: &[Option<i64>]| {
            if native {
                integer_grouped_fixture(keys, values)
            } else {
                grouped_batch(keys, values)
            }
        };
        let mut operator = SqlOperator::new(
            "totals",
            "SELECT key, SUM(value) AS total FROM events GROUP BY key",
            vec!["events".into()],
            vec![],
        )
        .unwrap();
        let job = StreamJobContext::new(
            1,
            "cancel-finalize",
            JsonMap::new(),
            None,
            CancellationToken::new(),
        );
        let context = StreamOperatorContext::new(&job, "totals", None);
        let mut collector = EdgeCollector::new(operator.output_ports().to_vec());
        let names = (0..if native { 100_000 } else { 1000 })
            .map(|index| format!("group-{}", if native { index % 10_000 } else { index }))
            .collect::<Vec<_>>();
        let keys = names
            .iter()
            .map(|name| Some(name.as_str()))
            .collect::<Vec<_>>();
        let incoming = grouped_batch(&keys, &vec![Some(1); keys.len()]);
        operator
            .process_data("events", incoming.clone(), &context, &mut collector)
            .await
            .unwrap();
        collector.drain("output");
        let before = operator.checkpoint(Epoch::INITIAL).unwrap();
        let finalized = operator
            .incremental
            .as_ref()
            .unwrap()
            .finalized_groups
            .clone();
        finalized.store(0, std::sync::atomic::Ordering::SeqCst);
        let observed = finalized.clone();
        let cancellation = job.cancellation().clone();
        let cancel = tokio::spawn(async move {
            while observed.load(std::sync::atomic::Ordering::SeqCst) == 0 {
                tokio::task::yield_now().await;
            }
            cancellation.cancel();
        });
        let result = operator
            .process_data("events", incoming, &context, &mut collector)
            .await;
        cancel.await.unwrap();
        assert!(matches!(result, Err(CalcFlowError::Cancelled { .. })));
        assert!(
            finalized.load(std::sync::atomic::Ordering::SeqCst) <= 512,
            "cancel must make progress during candidate finalization"
        );
        assert!(collector.drain("output").is_empty());
        let after = operator.checkpoint(Epoch::INITIAL).unwrap();
        assert_eq!(before.inline_metadata, after.inline_metadata);
        assert!(Arc::ptr_eq(
            &before.segments["input"].bytes_arc(),
            &after.segments["input"].bytes_arc()
        ));
    }
}

pub(super) static MATERIALIZE_CLONES: std::sync::atomic::AtomicUsize =
    std::sync::atomic::AtomicUsize::new(0);
pub(super) fn after_record_copies(metadata: &BatchMetadata, count: usize) {
    if metadata.source() == "materialize-cancel" {
        MATERIALIZE_CLONES.fetch_add(count, std::sync::atomic::Ordering::SeqCst);
    }
}

#[tokio::test]
async fn test_sql_incremental_restore_reserves_empty_record_handles_atomically() {
    let query = "SELECT COUNT(*) AS rows FROM events";
    let record = RecordBatch::try_from_iter(vec![(
        "value",
        Arc::new(Int64Array::from(Vec::<i64>::new())) as Arc<dyn Array>,
    )])
    .unwrap();
    let mut source = SqlOperator::new("totals", query, vec!["events".into()], vec![]).unwrap();
    let job = StreamJobContext::new(
        1,
        "restore-pool",
        JsonMap::new(),
        None,
        CancellationToken::new(),
    );
    let context = StreamOperatorContext::new(&job, "totals", None);
    let mut collector = EdgeCollector::new(source.output_ports().to_vec());
    source
        .process_data(
            "events",
            Batch::table(vec![record; 12_000], BatchMetadata::default()).unwrap(),
            &context,
            &mut collector,
        )
        .await
        .unwrap();
    let snapshot = source.checkpoint(Epoch::INITIAL).unwrap();
    assert_eq!(snapshot.inline_metadata["bytes"], serde_json::json!(0));
    let mut restored = SqlOperator::new("totals", query, vec!["events".into()], vec![]).unwrap();
    let pressure = restored
        .stream_state
        .runtime()
        .unwrap()
        .incremental_reservation("pressure");
    pressure.try_grow((1 << 30) - (1 << 20)).unwrap();
    assert!(matches!(
        StreamOperator::restore(&mut restored, &snapshot),
        Err(CalcFlowError::DataFusion { .. })
    ));
    assert!(restored.retained.is_none());
    drop(pressure);
    StreamOperator::restore(&mut restored, &snapshot).unwrap();
    assert_eq!(restored.retained.as_ref().unwrap().records.len(), 12_000);
}

#[tokio::test]
async fn test_sql_incremental_prepare_stops_copying_records_when_cancelled() {
    use std::sync::atomic::Ordering;
    MATERIALIZE_CLONES.store(0, Ordering::SeqCst);
    let record = RecordBatch::try_from_iter(vec![(
        "value",
        Arc::new(Int64Array::from(Vec::<i64>::new())) as Arc<dyn Array>,
    )])
    .unwrap();
    let input = Batch::table(
        vec![record; 100_000],
        BatchMetadata::new("materialize-cancel", 1, JsonMap::new()).unwrap(),
    )
    .unwrap();
    let mut operator = SqlOperator::new(
        "totals",
        "SELECT COUNT(*) AS rows FROM events",
        vec!["events".into()],
        vec![],
    )
    .unwrap();
    let job = StreamJobContext::new(
        1,
        "copy-cancel",
        JsonMap::new(),
        None,
        CancellationToken::new(),
    );
    let context = StreamOperatorContext::new(&job, "totals", None);
    let mut collector = EdgeCollector::new(operator.output_ports().to_vec());
    operator
        .process_data("events", input, &context, &mut collector)
        .await
        .unwrap();
    let cancellation = job.cancellation().clone();
    let canceller = tokio::spawn(async move {
        while MATERIALIZE_CLONES.load(Ordering::SeqCst) == 0 {
            tokio::task::yield_now().await;
        }
        cancellation.cancel();
    });
    assert!(matches!(
        operator.prepare_checkpoint_async(&context).await,
        Err(CalcFlowError::Cancelled { .. })
    ));
    canceller.await.unwrap();
    assert!(
        MATERIALIZE_CLONES.load(Ordering::SeqCst) <= 16_384,
        "copy must stop at a bounded record chunk"
    );
    assert!(operator.retained.as_ref().unwrap().segment.is_none());
}

fn bulk_group_input() -> Batch {
    let values = (0..100_000_i64)
        .map(|value| if value % 17 == 0 { None } else { Some(value) })
        .collect::<Vec<_>>();
    let record = RecordBatch::try_from_iter(vec![
        (
            "key",
            Arc::new(Int64Array::from(
                (0..100_000_i64).map(|value| value % 65).collect::<Vec<_>>(),
            )) as Arc<dyn Array>,
        ),
        (
            "value",
            Arc::new(Int64Array::from(values)) as Arc<dyn Array>,
        ),
    ])
    .unwrap();
    Batch::table(vec![record], BatchMetadata::default()).unwrap()
}

fn integer_grouped_fixture(keys: &[Option<&str>], values: &[Option<i64>]) -> Batch {
    let keys = keys
        .iter()
        .map(|key| {
            key.map(|key| match key {
                "a" => 0,
                "b" => 1,
                "c" => 2,
                key => key.rsplit_once('-').unwrap().1.parse::<i64>().unwrap() + 3,
            })
        })
        .collect::<Vec<_>>();
    Batch::table(
        vec![
            RecordBatch::try_from_iter(vec![
                ("key", Arc::new(Int64Array::from(keys)) as Arc<dyn Array>),
                (
                    "value",
                    Arc::new(Int64Array::from(values.to_vec())) as Arc<dyn Array>,
                ),
            ])
            .unwrap(),
        ],
        BatchMetadata::default(),
    )
    .unwrap()
}

fn assert_bulk_group_work(operator: &SqlOperator, groups: usize) {
    let incremental = operator.incremental.as_ref().unwrap();
    assert_eq!(
        incremental
            .partial_groups
            .load(std::sync::atomic::Ordering::SeqCst),
        groups
    );
    assert_eq!(
        incremental
            .historical_key_lookups
            .load(std::sync::atomic::Ordering::SeqCst),
        groups,
        "committed index must be probed only on each key's first transaction touch"
    );
}

#[tokio::test]
async fn test_sql_incremental_grouped_bulk_update_uses_compact_partial_groups() {
    let input = bulk_group_input();
    let query = "SELECT key, COUNT(*) AS rows, SUM(value) AS total, MIN(value) AS lo, MAX(value) AS hi FROM events GROUP BY key";
    let expected = DataFusionRuntime::new(DataFusionConfig::default())
        .unwrap()
        .sql(
            query,
            &BTreeMap::from([("events".into(), input.clone())]),
            None,
        )
        .await
        .unwrap();
    let mut operator = SqlOperator::new("totals", query, vec!["events".into()], vec![]).unwrap();
    let job = StreamJobContext::new(
        1,
        "bulk-grouped",
        JsonMap::new(),
        None,
        CancellationToken::new(),
    );
    let context = StreamOperatorContext::new(&job, "totals", None);
    let mut collector = EdgeCollector::new(operator.output_ports().to_vec());
    operator
        .process_data("events", input.clone(), &context, &mut collector)
        .await
        .unwrap();
    let produced = collector.drain("output");
    let actual = produced[0].as_data().unwrap();
    assert_eq!(
        actual.table_payload().unwrap().schema(),
        expected.table_payload().unwrap().schema()
    );
    assert_eq!(rows(actual), rows(&expected));
    assert_bulk_group_work(&operator, 65);
    assert_eq!(
        operator
            .incremental
            .as_ref()
            .unwrap()
            .encoded_rows
            .load(std::sync::atomic::Ordering::SeqCst),
        65,
        "only distinct native keys should be encoded"
    );
    let incremental = operator.incremental.as_ref().unwrap();
    incremental
        .encoded_rows
        .store(0, std::sync::atomic::Ordering::SeqCst);
    incremental
        .partial_groups
        .store(0, std::sync::atomic::Ordering::SeqCst);
    incremental
        .historical_key_lookups
        .store(0, std::sync::atomic::Ordering::SeqCst);
    let record = RecordBatch::try_new(
        input.table_payload().unwrap().schema().clone(),
        vec![
            Arc::new(Int64Array::from(vec![64])) as Arc<dyn Array>,
            Arc::new(Int64Array::from(vec![Some(9)])) as Arc<dyn Array>,
        ],
    )
    .unwrap();
    let second = Batch::table(vec![record.clone()], BatchMetadata::default()).unwrap();
    operator
        .process_data("events", second, &context, &mut collector)
        .await
        .unwrap();
    let mut records = input.table_payload().unwrap().batches().to_vec();
    records.push(record);
    let expected = DataFusionRuntime::new(DataFusionConfig::default())
        .unwrap()
        .sql(
            query,
            &BTreeMap::from([(
                "events".into(),
                Batch::table(records, BatchMetadata::default()).unwrap(),
            )]),
            None,
        )
        .await
        .unwrap();
    let produced = collector.drain("output");
    let actual = produced[0].as_data().unwrap();
    assert_eq!(
        actual.table_payload().unwrap().schema(),
        expected.table_payload().unwrap().schema()
    );
    assert_eq!(rows(actual), rows(&expected));
    assert_bulk_group_work(&operator, 1);
    assert_eq!(
        operator
            .incremental
            .as_ref()
            .unwrap()
            .encoded_rows
            .load(std::sync::atomic::Ordering::SeqCst),
        1
    );
}

#[tokio::test]
async fn test_sql_incremental_single_primitive_key_prefix_matches_datafusion() {
    let data_types = [
        DataType::Int8,
        DataType::Int16,
        DataType::Int32,
        DataType::Int64,
        DataType::UInt8,
        DataType::UInt16,
        DataType::UInt32,
        DataType::UInt64,
        DataType::Boolean,
    ];
    for data_type in data_types {
        assert_primitive_key_prefix(data_type).await;
    }
}

fn primitive_key_record(values: Vec<datafusion::common::ScalarValue>) -> RecordBatch {
    use datafusion::arrow::datatypes::{Field, Schema};
    let row_count = values.len();
    let keys = datafusion::common::ScalarValue::iter_to_array(values).unwrap();
    RecordBatch::try_new(
        Arc::new(Schema::new(vec![
            Field::new("value", DataType::Int64, true),
            Field::new("key", keys.data_type().clone(), true),
        ])),
        vec![
            Arc::new(
                (0..row_count)
                    .map(|row| (row != 1).then_some(i64::try_from(row).unwrap()))
                    .collect::<Int64Array>(),
            ),
            keys,
        ],
    )
    .unwrap()
}

async fn assert_primitive_key_prefix(data_type: DataType) {
    use datafusion::common::ScalarValue;
    let query = "SELECT key, COUNT(*) AS rows, SUM(value) AS total, MIN(value) AS lo, MAX(value) AS hi FROM events GROUP BY key";

    let null = ScalarValue::try_from(&data_type).unwrap();
    let (low, high) = if data_type == DataType::Boolean {
        (
            ScalarValue::Boolean(Some(false)),
            ScalarValue::Boolean(Some(true)),
        )
    } else {
        (
            ScalarValue::Int64(Some(0)).cast_to(&data_type).unwrap(),
            ScalarValue::Int64(Some(1)).cast_to(&data_type).unwrap(),
        )
    };
    let runtime = DataFusionRuntime::new(DataFusionConfig::default()).unwrap();
    let mut operator = SqlOperator::new("totals", query, vec!["events".into()], vec![]).unwrap();
    let job = StreamJobContext::new(
        1,
        "primitive-key",
        JsonMap::new(),
        None,
        CancellationToken::new(),
    );
    let context = StreamOperatorContext::new(&job, "totals", None);
    let mut collector = EdgeCollector::new(operator.output_ports().to_vec());
    let mut prefix = Vec::new();
    for values in [
        vec![null.clone(), low.clone(), low.clone(), high.clone()],
        vec![high, null, low],
    ] {
        if let Some(incremental) = operator.incremental.as_ref() {
            incremental
                .encoded_rows
                .store(0, std::sync::atomic::Ordering::SeqCst);
        }
        let record = primitive_key_record(values);
        prefix.push(record.clone());
        operator
            .process_data(
                "events",
                Batch::table(vec![record], BatchMetadata::default()).unwrap(),
                &context,
                &mut collector,
            )
            .await
            .unwrap();
        let expected = runtime
            .sql(
                query,
                &BTreeMap::from([(
                    "events".into(),
                    Batch::table(prefix.clone(), BatchMetadata::default()).unwrap(),
                )]),
                None,
            )
            .await
            .unwrap();
        let produced = collector.drain("output");
        let actual = produced[0].as_data().unwrap();
        assert_eq!(
            actual.table_payload().unwrap().schema(),
            expected.table_payload().unwrap().schema()
        );
        assert_eq!(rows(actual), rows(&expected));
        assert_eq!(
            operator
                .incremental
                .as_ref()
                .unwrap()
                .encoded_rows
                .load(std::sync::atomic::Ordering::SeqCst),
            3
        );
    }
    assert_restored_primitive_prefix(
        &mut operator,
        &runtime,
        query,
        &prefix,
        &context,
        &mut collector,
    )
    .await;
}

async fn assert_restored_primitive_prefix(
    operator: &mut SqlOperator,
    runtime: &DataFusionRuntime,
    query: &str,
    prefix: &[RecordBatch],
    context: &StreamOperatorContext<'_>,
    collector: &mut EdgeCollector,
) {
    let before = operator.checkpoint(Epoch::INITIAL).unwrap();
    let mut restored = operator.clone();
    StreamOperator::restore(&mut restored, &before).unwrap();
    let empty = Batch::table(
        vec![RecordBatch::new_empty(prefix[0].schema())],
        BatchMetadata::default(),
    )
    .unwrap();
    restored
        .process_data("events", empty, context, collector)
        .await
        .unwrap();
    let output = collector.drain("output");
    assert_eq!(output.len(), 1);
    let expected = runtime
        .sql(
            query,
            &BTreeMap::from([(
                "events".into(),
                Batch::table(prefix.to_vec(), BatchMetadata::default()).unwrap(),
            )]),
            None,
        )
        .await
        .unwrap();
    let actual = output[0].as_data().unwrap();
    assert_eq!(
        actual.table_payload().unwrap().schema(),
        expected.table_payload().unwrap().schema()
    );
    assert_eq!(rows(actual), rows(&expected));
    let after = restored.checkpoint(Epoch::INITIAL).unwrap();
    assert_eq!(before.inline_metadata, after.inline_metadata);
    assert_eq!(
        before.segments.keys().collect::<Vec<_>>(),
        after.segments.keys().collect::<Vec<_>>()
    );
    for (name, segment) in &before.segments {
        assert_eq!(segment.bytes(), after.segments[name].bytes());
    }
}
