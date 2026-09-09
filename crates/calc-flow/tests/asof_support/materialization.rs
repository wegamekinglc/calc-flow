//! Typed ordering and bounded candidate materialization contracts.

use std::sync::Arc;

use datafusion::arrow::{
    array::{ArrayRef, Int64Array, StringArray, UInt64Array},
    datatypes::{DataType, Field, Schema},
    record_batch::RecordBatch,
    row::{RowConverter, SortField},
};
use datafusion::execution::{
    disk_manager::{DiskManagerBuilder, DiskManagerMode},
    memory_pool::{GreedyMemoryPool, MemoryConsumer, MemoryPool},
    runtime_env::RuntimeEnvBuilder,
};
use datafusion::prelude::{SessionConfig, SessionContext};

#[test]
fn typed_sequence_encoding_orders_signed_values_and_utf8() {
    let converter = RowConverter::new(vec![
        SortField::new(DataType::Int64),
        SortField::new(DataType::Utf8),
    ])
    .unwrap();
    let arrays: Vec<ArrayRef> = vec![
        Arc::new(Int64Array::from(vec![9, -1, 7, 9, i64::MIN, i64::MAX])),
        Arc::new(StringArray::from(vec!["a", "z", "z", "é", "z", "a"])),
    ];
    let rows = converter.convert_columns(&arrays).unwrap();
    let mut order: Vec<usize> = (0..arrays[0].len()).collect();
    order.sort_by(|&left, &right| rows.row(left).cmp(&rows.row(right)));
    assert_eq!(order, [4, 1, 2, 0, 3, 5]);
}

#[tokio::test]
async fn bounded_candidate_join_preserves_unmatched_and_reused_right_rows() {
    let pool = Arc::new(GreedyMemoryPool::new(8 * 1024 * 1024));
    let runtime = RuntimeEnvBuilder::new()
        .with_memory_pool(pool.clone())
        .with_disk_manager_builder(
            DiskManagerBuilder::default().with_mode(DiskManagerMode::Disabled),
        )
        .build_arc()
        .unwrap();
    let context =
        SessionContext::new_with_config_rt(SessionConfig::new().with_target_partitions(1), runtime);
    let left_schema = Arc::new(Schema::new(vec![
        Field::new("ordinal", DataType::UInt64, false),
        Field::new("key", DataType::Utf8, false),
    ]));
    let right_schema = Arc::new(Schema::new(vec![
        Field::new("ordinal", DataType::UInt64, false),
        Field::new("key", DataType::Utf8, false),
        Field::new("value", DataType::Int64, false),
    ]));
    let left = RecordBatch::try_new(
        left_schema,
        vec![
            Arc::new(UInt64Array::from(vec![2, 0, 1, 3])),
            Arc::new(StringArray::from(vec!["A", "A", "A", "B"])),
        ],
    )
    .unwrap();
    let right = RecordBatch::try_new(
        right_schema,
        vec![
            Arc::new(UInt64Array::from(vec![0, 2, 3])),
            Arc::new(StringArray::from(vec!["A", "A", "A"])),
            Arc::new(Int64Array::from(vec![100, 100, 999])),
        ],
    )
    .unwrap();
    context.register_batch("l", left).unwrap();
    context.register_batch("r", right).unwrap();
    let output = context
        .sql(
            "SELECT l.ordinal, l.key, r.value FROM l LEFT JOIN r \
             ON l.ordinal = r.ordinal AND l.key = r.key ORDER BY l.ordinal",
        )
        .await
        .unwrap()
        .collect()
        .await
        .unwrap();
    assert_eq!(output.iter().map(RecordBatch::num_rows).sum::<usize>(), 4);
    let values: Vec<_> = output
        .iter()
        .flat_map(|batch| {
            batch
                .column(2)
                .as_any()
                .downcast_ref::<Int64Array>()
                .unwrap()
                .iter()
        })
        .collect();
    assert_eq!(values, [Some(100), None, Some(100), None]);
    assert!(output[0].schema().field(2).is_nullable());
    assert_eq!(pool.reserved(), 0);
}

#[test]
fn work_pool_fails_before_exceeding_its_reservation() {
    let pool: Arc<dyn MemoryPool> = Arc::new(GreedyMemoryPool::new(64));
    let reservation = MemoryConsumer::new("asof-workspace").register(&pool);
    reservation.try_grow(64).unwrap();
    assert!(reservation.try_grow(1).is_err());
    assert_eq!(pool.reserved(), 64);
    drop(reservation);
    assert_eq!(pool.reserved(), 0);
}
