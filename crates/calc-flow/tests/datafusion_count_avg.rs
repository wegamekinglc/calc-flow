use std::{collections::BTreeMap, sync::Arc};

use calc_flow::{Batch, BatchMetadata, DataFusionConfig, DataFusionRuntime};
use datafusion::arrow::{
    array::{Array, Float64Array, Int64Array, StringArray, TimestampMicrosecondArray, UInt64Array},
    compute::concat_batches,
    datatypes::{DataType, Field, Schema, TimeUnit},
    record_batch::RecordBatch,
};

fn prices(values: Vec<Option<f64>>, nullable: bool) -> Batch {
    let rows = values.len();
    let schema = Arc::new(Schema::new(vec![
        Field::new(
            "event_time",
            DataType::Timestamp(TimeUnit::Microsecond, None),
            false,
        ),
        Field::new("sequence", DataType::UInt64, false),
        Field::new("symbol", DataType::Utf8, false),
        Field::new("price", DataType::Float64, nullable),
    ]));
    let record = RecordBatch::try_new(
        schema,
        vec![
            Arc::new(TimestampMicrosecondArray::from_iter_values(
                (0..rows).map(|index| i64::try_from(index).unwrap()),
            )),
            Arc::new(UInt64Array::from_iter_values(
                (0..rows).map(|index| u64::try_from(index).unwrap()),
            )),
            Arc::new(StringArray::from(vec!["entity"; rows])),
            Arc::new(Float64Array::from(values)),
        ],
    )
    .unwrap();
    Batch::table(vec![record], BatchMetadata::default()).unwrap()
}

fn sql(dual: bool) -> String {
    let expression = if dual {
        "AVG(price) OVER fast - AVG(price) OVER slow"
    } else {
        "AVG(price) OVER slow"
    };
    format!(
        "SELECT event_time, sequence, symbol, price, \
         COUNT(price) OVER slow AS count20, \
         CASE WHEN COUNT(price) OVER slow = 20 THEN {expression} END AS value \
         FROM input WINDOW slow AS (PARTITION BY symbol ORDER BY event_time, sequence \
         ROWS BETWEEN 19 PRECEDING AND CURRENT ROW), \
         fast AS (PARTITION BY symbol ORDER BY event_time, sequence \
         ROWS BETWEEN 4 PRECEDING AND CURRENT ROW)"
    )
}

async fn rewritten_and_fallback(input: Batch, dual: bool) -> RecordBatch {
    let input_before = input.table_payload().unwrap().batches().to_vec();
    let tables = BTreeMap::from([("input".to_owned(), input)]);
    let query = sql(dual);
    let runtime = DataFusionRuntime::new(DataFusionConfig {
        batch_size: 7,
        ..DataFusionConfig::default()
    })
    .unwrap();
    let fallback = DataFusionRuntime::new(DataFusionConfig {
        batch_size: 7,
        enable_rolling_rewrite: false,
        ..DataFusionConfig::default()
    })
    .unwrap();
    let expected = fallback.sql(&query, &tables, None).await.unwrap();
    let actual = runtime.sql(&query, &tables, None).await.unwrap();
    let metrics = runtime.metrics();
    assert_eq!(
        metrics[0].rolling_rewritten_windows,
        if dual { 3 } else { 2 }
    );
    assert!(metrics[0].rolling_fallback_reasons.is_empty());
    assert!(metrics[0].physical_plan.contains("route=sorted_partitions"));
    let actual = actual.table_payload().unwrap();
    let expected = expected.table_payload().unwrap();
    assert_eq!(actual.schema(), expected.schema());
    let actual = concat_batches(actual.schema(), actual.batches()).unwrap();
    let expected = concat_batches(expected.schema(), expected.batches()).unwrap();
    assert_eq!(actual, expected);
    assert_eq!(
        tables["input"].table_payload().unwrap().batches(),
        input_before
    );
    let schema = actual.schema();
    let field = schema.field_with_name("count20").unwrap();
    assert_eq!(field.data_type(), &DataType::Int64);
    assert!(!field.is_nullable());
    actual
}

#[tokio::test]
async fn full_window_count_avg_rewrites_nullable_and_required_price() {
    for nullable in [true, false] {
        for dual in [false, true] {
            let actual = rewritten_and_fallback(
                prices(
                    (1..=43).map(|value| Some(f64::from(value))).collect(),
                    nullable,
                ),
                dual,
            )
            .await;
            let values = actual.column_by_name("value").unwrap();
            assert_eq!(values.null_count(), 19);
            let counts = actual
                .column_by_name("count20")
                .unwrap()
                .as_any()
                .downcast_ref::<Int64Array>()
                .unwrap();
            assert_eq!(counts.value(0), 1);
            assert_eq!(counts.value(19), 20);
        }
    }
}

#[tokio::test]
async fn full_window_count_avg_keeps_ten_row_outputs_null() {
    for dual in [false, true] {
        let actual = rewritten_and_fallback(prices(vec![Some(3.0); 10], false), dual).await;
        assert_eq!(actual.column_by_name("value").unwrap().null_count(), 10);
    }
}

#[tokio::test]
async fn full_window_count_avg_preserves_sql_extremes_and_non_null_nan_counts() {
    let cases = [
        [vec![Some(1e308); 20], vec![Some(1.0); 25]].concat(),
        [vec![Some(f64::NAN)], vec![Some(1.0); 44]].concat(),
        [vec![Some(f64::INFINITY)], vec![Some(1.0); 44]].concat(),
        [vec![Some(f64::NEG_INFINITY)], vec![Some(1.0); 44]].concat(),
        [vec![None; 25], vec![Some(2.0); 20], vec![None; 25]].concat(),
        [
            vec![Some(1e16), Some(1.0), Some(-1e16)],
            vec![Some(3.0); 42],
        ]
        .concat(),
    ];
    for values in cases {
        for dual in [false, true] {
            rewritten_and_fallback(prices(values.clone(), true), dual).await;
        }
    }
}

#[tokio::test]
async fn count_avg_rewrite_keeps_unproved_combinations_on_datafusion() {
    let cases = [
        "COUNT(*) OVER slow, AVG(price) OVER slow",
        "COUNT(sequence) OVER slow, AVG(price) OVER slow",
        "COUNT(price) OVER slow, AVG(price) OVER fast",
        "COUNT(DISTINCT price) OVER slow, AVG(price) OVER slow",
        "COUNT(price) FILTER (WHERE price > 0) OVER slow, AVG(price) OVER slow",
        "COUNT(CAST(price AS INT)) OVER slow, AVG(price) OVER slow",
        "COUNT(price + 1) OVER slow, AVG(price) OVER slow",
        "COUNT(price) OVER slow",
        "SUM(price) OVER slow, COUNT(price) OVER slow, AVG(price) OVER slow",
    ];
    let tables = BTreeMap::from([(
        "input".to_owned(),
        prices(
            [vec![None, Some(f64::NAN)], vec![Some(2.0); 23]].concat(),
            true,
        ),
    )]);
    for expression in cases {
        let query = format!(
            "SELECT {expression} FROM input \
             WINDOW slow AS (PARTITION BY symbol ORDER BY event_time, sequence \
             ROWS BETWEEN 19 PRECEDING AND CURRENT ROW), \
             fast AS (PARTITION BY symbol ORDER BY event_time, sequence \
             ROWS BETWEEN 4 PRECEDING AND CURRENT ROW)"
        );
        let runtime = DataFusionRuntime::new(DataFusionConfig::default()).unwrap();
        let fallback = DataFusionRuntime::new(DataFusionConfig {
            enable_rolling_rewrite: false,
            ..DataFusionConfig::default()
        })
        .unwrap();
        let actual = runtime.sql(&query, &tables, None).await;
        let expected = fallback.sql(&query, &tables, None).await;
        let (actual, expected) = match (actual, expected) {
            (Ok(actual), Ok(expected)) => (actual, expected),
            (Err(actual), Err(expected)) => {
                assert_eq!(actual.to_string(), expected.to_string());
                continue;
            }
            other => panic!("rewrite and fallback outcomes differ: {other:?}"),
        };
        let actual = actual.table_payload().unwrap();
        let expected = expected.table_payload().unwrap();
        assert_eq!(actual.schema(), expected.schema());
        assert_eq!(
            concat_batches(actual.schema(), actual.batches()).unwrap(),
            concat_batches(expected.schema(), expected.batches()).unwrap()
        );
        let metrics = runtime.metrics();
        assert_eq!(metrics[0].rolling_rewritten_windows, 0, "{expression}");
        assert!(
            !metrics[0].rolling_fallback_reasons.is_empty(),
            "{expression}"
        );
        assert!(
            !metrics[0].physical_plan.contains("CalcFlowRollingExec"),
            "{expression}"
        );
    }
}

#[tokio::test]
async fn full_window_count_avg_isolates_nullable_multi_keys_and_executions() {
    let original = prices(
        [
            vec![Some(f64::NAN); 35],
            vec![Some(2.0); 35],
            vec![Some(7.0); 35],
        ]
        .concat(),
        true,
    );
    let record = original.table_payload().unwrap().batches()[0].clone();
    let mut fields = record.schema().fields().to_vec();
    fields[2] = Arc::new(Field::new("symbol", DataType::Utf8, true));
    fields.push(Arc::new(Field::new("venue", DataType::Int64, true)));
    let schema = Arc::new(Schema::new(fields));
    let mut columns = record.columns().to_vec();
    columns[2] = Arc::new(StringArray::from(
        [vec![Some("a"); 70], vec![None; 35]].concat(),
    ));
    columns.push(Arc::new(Int64Array::from(
        [vec![None; 35], vec![Some(1); 70]].concat(),
    )));
    let record = RecordBatch::try_new(Arc::clone(&schema), columns).unwrap();
    let input = Batch::table(
        vec![
            record.slice(0, 40),
            RecordBatch::new_empty(schema),
            record.slice(40, 65),
        ],
        BatchMetadata::default(),
    )
    .unwrap();
    let tables = BTreeMap::from([("input".to_owned(), input)]);
    let query = format!(
        "{} ORDER BY symbol NULLS FIRST, venue NULLS FIRST, event_time, sequence",
        sql(true).replace("PARTITION BY symbol", "PARTITION BY symbol, venue"),
    );
    let config = DataFusionConfig {
        batch_size: 3,
        target_partitions: 4,
        min_rows_per_partition: 1,
        ..DataFusionConfig::default()
    };
    let runtime = DataFusionRuntime::new(config).unwrap();
    let fallback = DataFusionRuntime::new(DataFusionConfig {
        enable_rolling_rewrite: false,
        ..config
    })
    .unwrap();
    let expected = fallback.sql(&query, &tables, None).await.unwrap();
    let expected = expected.table_payload().unwrap();
    let expected = concat_batches(expected.schema(), expected.batches()).unwrap();
    for _ in 0..2 {
        let actual = runtime.sql(&query, &tables, None).await.unwrap();
        let actual = actual.table_payload().unwrap();
        let actual = concat_batches(actual.schema(), actual.batches()).unwrap();
        assert_eq!(actual, expected);
        assert_eq!(actual.column_by_name("value").unwrap().null_count(), 3 * 19);
    }
    for metric in runtime.metrics() {
        assert_eq!(metric.rolling_rewritten_windows, 3);
        assert_eq!(metric.window_partition_count, 4);
        assert!(metric.rolling_fallback_reasons.is_empty());
    }
}

#[tokio::test]
async fn unsupported_window_options_report_the_matching_aggregate() {
    let tables = BTreeMap::from([("input".to_owned(), prices(vec![Some(2.0); 25], false))]);
    let cases = [
        (
            "COUNT(DISTINCT price) OVER slow",
            "count_filter_distinct_or_null_treatment_is_not_supported",
        ),
        (
            "COUNT(price) FILTER (WHERE price > 0) OVER slow",
            "count_filter_distinct_or_null_treatment_is_not_supported",
        ),
        (
            "AVG(price) FILTER (WHERE price > 0) OVER slow",
            "avg_filter_distinct_or_null_treatment_is_not_supported",
        ),
    ];
    for (expression, reason) in cases {
        let query = format!(
            "SELECT {expression}, AVG(price) OVER slow FROM input \
             WINDOW slow AS (PARTITION BY symbol ORDER BY event_time, sequence \
             ROWS BETWEEN 19 PRECEDING AND CURRENT ROW)"
        );
        let runtime = DataFusionRuntime::new(DataFusionConfig::default()).unwrap();
        let fallback = DataFusionRuntime::new(DataFusionConfig {
            enable_rolling_rewrite: false,
            ..DataFusionConfig::default()
        })
        .unwrap();
        let actual = runtime.sql(&query, &tables, None).await.unwrap();
        let expected = fallback.sql(&query, &tables, None).await.unwrap();
        let metric = &runtime.metrics()[0];
        assert_eq!(metric.rolling_rewritten_windows, 0, "{expression}");
        assert_eq!(metric.rolling_fallback_reasons, [reason], "{expression}");
        let actual = actual.table_payload().unwrap();
        let expected = expected.table_payload().unwrap();
        assert_eq!(
            concat_batches(actual.schema(), actual.batches()).unwrap(),
            concat_batches(expected.schema(), expected.batches()).unwrap()
        );
    }
}

#[tokio::test]
async fn count_query_preserves_sql_mean_in_separate_window_stages() {
    let tables = BTreeMap::from([("input".to_owned(), prices(vec![Some(1e308); 25], false))]);
    let query = "SELECT COUNT(price) OVER slow AS count20, AVG(price) OVER slow AS mean20, \
        AVG(price) OVER other_order AS other_mean20 FROM input \
        WINDOW slow AS (PARTITION BY symbol ORDER BY event_time, sequence \
        ROWS BETWEEN 19 PRECEDING AND CURRENT ROW), \
        other_order AS (PARTITION BY symbol ORDER BY event_time, price \
        ROWS BETWEEN 19 PRECEDING AND CURRENT ROW)";
    let runtime = DataFusionRuntime::new(DataFusionConfig::default()).unwrap();
    let fallback = DataFusionRuntime::new(DataFusionConfig {
        enable_rolling_rewrite: false,
        ..DataFusionConfig::default()
    })
    .unwrap();
    let actual = runtime.sql(query, &tables, None).await.unwrap();
    let expected = fallback.sql(query, &tables, None).await.unwrap();
    assert_eq!(runtime.metrics()[0].rolling_rewritten_windows, 3);
    let actual = actual.table_payload().unwrap();
    let expected = expected.table_payload().unwrap();
    assert_eq!(
        concat_batches(actual.schema(), actual.batches()).unwrap(),
        concat_batches(expected.schema(), expected.batches()).unwrap()
    );
}

#[tokio::test]
async fn unsupported_count_stage_preserves_whole_query_fallback() {
    let tables = BTreeMap::from([("input".to_owned(), prices(vec![Some(1e308); 25], false))]);
    let query = "SELECT COUNT(sequence) OVER slow AS count20, \
        AVG(price) OVER other_order AS mean20 FROM input \
        WINDOW slow AS (PARTITION BY symbol ORDER BY event_time, sequence \
        ROWS BETWEEN 19 PRECEDING AND CURRENT ROW), \
        other_order AS (PARTITION BY symbol ORDER BY event_time, price \
        ROWS BETWEEN 19 PRECEDING AND CURRENT ROW)";
    let runtime = DataFusionRuntime::new(DataFusionConfig::default()).unwrap();
    let fallback = DataFusionRuntime::new(DataFusionConfig {
        enable_rolling_rewrite: false,
        ..DataFusionConfig::default()
    })
    .unwrap();
    let actual = runtime.sql(query, &tables, None).await.unwrap();
    let expected = fallback.sql(query, &tables, None).await.unwrap();
    assert_eq!(runtime.metrics()[0].rolling_rewritten_windows, 0);
    assert!(
        !runtime.metrics()[0]
            .physical_plan
            .contains("CalcFlowRollingExec")
    );
    let actual = actual.table_payload().unwrap();
    let expected = expected.table_payload().unwrap();
    assert_eq!(
        concat_batches(actual.schema(), actual.batches()).unwrap(),
        concat_batches(expected.schema(), expected.batches()).unwrap()
    );
}
