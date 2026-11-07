use std::{collections::BTreeMap, sync::Arc};

use datafusion::{
    arrow::{
        array::{ArrayRef, Float64Array, StringArray, TimestampMicrosecondArray, UInt64Array},
        compute::concat_batches,
        datatypes::{DataType, Field, Schema, TimeUnit},
        record_batch::RecordBatch,
    },
    execution::memory_pool::{GreedyMemoryPool, MemoryPool},
};

use crate::{Batch, BatchMetadata, DataFusionConfig, DataFusionRuntime};

fn input() -> Batch {
    let schema = Arc::new(Schema::new(vec![
        Field::new(
            "event_time",
            DataType::Timestamp(TimeUnit::Microsecond, None),
            false,
        ),
        Field::new("sequence", DataType::UInt64, false),
        Field::new("symbol", DataType::Utf8, false),
        Field::new("price", DataType::Float64, false),
    ]));
    let columns: Vec<ArrayRef> = vec![
        Arc::new(TimestampMicrosecondArray::from_iter_values(
            (1..=30).chain(1..=30),
        )),
        Arc::new(UInt64Array::from_iter_values((1..=30).chain(1..=30))),
        Arc::new(StringArray::from_iter_values(
            std::iter::repeat_n("a", 30).chain(std::iter::repeat_n("b", 30)),
        )),
        Arc::new(Float64Array::from_iter_values(
            (1..=30).chain(101..=130).map(f64::from),
        )),
    ];
    let record = RecordBatch::try_new(schema, columns).unwrap();
    Batch::table(
        vec![record.slice(0, 17), record.slice(17, 43)],
        BatchMetadata::default(),
    )
    .unwrap()
}

#[tokio::test]
async fn p3_actual_sql_two_avg_frames_use_one_sorted_node_and_keep_datafusion_limits() {
    let tables = BTreeMap::from([("input".to_owned(), input())]);
    let query = "SELECT symbol, event_time, sequence, AVG(price) OVER (PARTITION BY symbol ORDER BY event_time, sequence ROWS BETWEEN 4 PRECEDING AND CURRENT ROW) AS fast, AVG(price) OVER (PARTITION BY symbol ORDER BY event_time, sequence ROWS BETWEEN 19 PRECEDING AND CURRENT ROW) AS slow FROM input ORDER BY symbol, event_time, sequence";
    let config = DataFusionConfig {
        batch_size: 3,
        target_partitions: 1,
        ..DataFusionConfig::default()
    };
    let reference = DataFusionRuntime::new(DataFusionConfig {
        enable_rolling_rewrite: false,
        ..config
    })
    .unwrap();
    let expected = reference.sql(query, &tables, None).await.unwrap();
    let runtime = DataFusionRuntime::new(config).unwrap();
    let actual = runtime.sql(query, &tables, None).await.unwrap();
    let metric = runtime.metrics().pop().unwrap();
    assert_eq!(metric.rolling_rewritten_windows, 2);
    assert!(
        metric
            .physical_plan
            .contains("CalcFlowRollingExec: windows=2, route=sorted_partitions"),
        "{}",
        metric.physical_plan
    );
    let actual = actual.table_payload().unwrap();
    assert!(actual.batches().len() > 1);
    let actual = concat_batches(actual.schema(), actual.batches()).unwrap();
    let expected = expected.table_payload().unwrap();
    let expected = concat_batches(expected.schema(), expected.batches()).unwrap();
    assert_eq!(
        actual.project(&[0, 1, 2]).unwrap(),
        expected.project(&[0, 1, 2]).unwrap()
    );
    for index in [3, 4] {
        let actual = actual
            .column(index)
            .as_any()
            .downcast_ref::<Float64Array>()
            .unwrap();
        let expected = expected
            .column(index)
            .as_any()
            .downcast_ref::<Float64Array>()
            .unwrap();
        assert_eq!(
            actual
                .iter()
                .map(|value| value.map(f64::to_bits))
                .collect::<Vec<_>>(),
            expected
                .iter()
                .map(|value| value.map(f64::to_bits))
                .collect::<Vec<_>>()
        );
    }
    let pool = Arc::new(GreedyMemoryPool::new(0));
    let bounded = DataFusionRuntime::with_benchmark_pool(config, pool.clone()).unwrap();
    assert!(bounded.sql(query, &tables, None).await.is_err());
    assert_eq!(pool.reserved(), 0);
}
