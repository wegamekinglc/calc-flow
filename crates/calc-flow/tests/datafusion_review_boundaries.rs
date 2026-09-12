use std::{collections::BTreeMap, sync::Arc};

use calc_flow::{Batch, BatchMetadata, DataFusionConfig, DataFusionRuntime};
use datafusion::arrow::{
    array::{
        Array, Float64Array, Int32Builder, MapBuilder, StringBuilder, TimestampMicrosecondArray,
    },
    compute::concat_batches,
    datatypes::{DataType, Field, Schema, TimeUnit},
    record_batch::RecordBatch,
};

#[tokio::test]
async fn count_avg_map_partition_preserves_datafusion_fallback() {
    let mut maps = MapBuilder::new(None, StringBuilder::new(), Int32Builder::new());
    for _ in 0..3 {
        maps.keys().append_value("entity");
        maps.values().append_value(1);
        maps.append(true).unwrap();
    }
    let maps = maps.finish();
    let schema = Arc::new(Schema::new(vec![
        Field::new(
            "event_time",
            DataType::Timestamp(TimeUnit::Microsecond, None),
            false,
        ),
        Field::new("entity", maps.data_type().clone(), false),
        Field::new("price", DataType::Float64, true),
    ]));
    let batch = RecordBatch::try_new(
        schema,
        vec![
            Arc::new(TimestampMicrosecondArray::from(vec![1, 2, 3])),
            Arc::new(maps),
            Arc::new(Float64Array::from(vec![1.0, 3.0, 5.0])),
        ],
    )
    .unwrap();
    let tables = BTreeMap::from([(
        "input".into(),
        Batch::table(vec![batch], BatchMetadata::default()).unwrap(),
    )]);
    let sql = "SELECT COUNT(price) OVER w AS n, AVG(price) OVER w AS mean FROM input WINDOW w AS (PARTITION BY entity ORDER BY event_time ROWS BETWEEN 1 PRECEDING AND CURRENT ROW)";
    let fallback = DataFusionRuntime::new(DataFusionConfig {
        enable_rolling_rewrite: false,
        ..DataFusionConfig::default()
    })
    .unwrap();
    let expected = fallback.sql(sql, &tables, None).await.unwrap();
    let optimized = DataFusionRuntime::new(DataFusionConfig::default()).unwrap();
    let actual = optimized.sql(sql, &tables, None).await.unwrap();
    let expected = expected.table_payload().unwrap();
    let actual = actual.table_payload().unwrap();
    assert_eq!(
        concat_batches(expected.schema(), expected.batches()).unwrap(),
        concat_batches(actual.schema(), actual.batches()).unwrap()
    );
}
