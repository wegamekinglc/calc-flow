use super::*;
use crate::operator::asof::state::{PayloadBatch, RowPayload};
use datafusion::{
    arrow::{
        array::{Int64Array, TimestampSecondArray},
        datatypes::{DataType, Field},
    },
    execution::memory_pool::{GreedyMemoryPool, MemoryConsumer, MemoryPool},
};
use std::sync::OnceLock;

fn payload(metadata: HashMap<String, String>) -> RowPayload {
    let schema = Arc::new(
        Schema::new(vec![Field::new("value", DataType::Int64, false)]).with_metadata(metadata),
    );
    let record = RecordBatch::try_new(schema, vec![Arc::new(Int64Array::from(vec![1]))]).unwrap();
    RowPayload {
        batch: Arc::new(PayloadBatch {
            key: (0, 0),
            record: Arc::new(record),
            body_bytes: 0,
            encoded_charge_bytes: 0,
            encoded: OnceLock::new(),
        }),
        row: 0,
    }
}

fn assert_metadata_capacity_is_funded_or_released(row: RowPayload, minimum: usize) {
    let schema = Arc::downgrade(&row.batch.record.schema());
    let pool: Arc<dyn MemoryPool> = Arc::new(GreedyMemoryPool::new(32 << 20));
    let mut credit = MemoryConsumer::new("metadata-test").register(&pool);
    let mut builder = OutputPlanBuilder::new(1, None, &mut credit, "asof").unwrap();
    builder.push(row.view(), None, &mut credit, "asof").unwrap();
    let plan = builder
        .finish(&row.batch.record.schema(), &mut credit, "asof")
        .unwrap();
    drop(row);
    assert!(
        schema.upgrade().is_none() || credit.size() >= minimum,
        "retained metadata has {} bytes of credit, needs {minimum}",
        credit.size()
    );
    drop((plan, credit));
    assert_eq!(pool.reserved(), 0);
}

#[test]
fn schema_metadata_empty_hash_capacity_is_funded_or_released() {
    let metadata = HashMap::with_capacity(65_536);
    let minimum = metadata.capacity() * size_of::<(String, String)>();
    assert_metadata_capacity_is_funded_or_released(payload(metadata), minimum);
}

#[test]
fn schema_metadata_string_capacity_is_funded_or_released() {
    let mut key = String::with_capacity(1 << 20);
    let mut value = String::with_capacity(1 << 20);
    key.push('k');
    value.push('v');
    let minimum = key.capacity() + value.capacity();
    assert_metadata_capacity_is_funded_or_released(payload(HashMap::from([(key, value)])), minimum);
}

#[test]
fn schema_metadata_tight_pool_cannot_retain_unfunded_source() {
    let row = payload(HashMap::with_capacity(65_536));
    let before = Arc::strong_count(&row.batch.record);
    let pool: Arc<dyn MemoryPool> = Arc::new(GreedyMemoryPool::new(128 << 10));
    let mut credit = MemoryConsumer::new("metadata-test").register(&pool);
    let mut builder = OutputPlanBuilder::new(1, None, &mut credit, "asof").unwrap();
    let initial = credit.size();
    let result = builder.push(row.view(), None, &mut credit, "asof");
    match result {
        Err(crate::CalcFlowError::OperatorReason {
            reason_code: StreamingFailureReason::AsofWorkspaceLimitExceeded,
            ..
        }) => assert_eq!(credit.size(), initial),
        Ok(()) => assert!(credit.size() < 128 << 10),
        other => panic!("unexpected output preparation result: {other:?}"),
    }
    assert_eq!(Arc::strong_count(&row.batch.record), before);
    drop((builder, credit));
    assert_eq!(pool.reserved(), 0);
}

#[test]
fn output_plan_reserves_timestamp_timezone_backing() {
    let mut row = payload(HashMap::new());
    let timezone: Arc<str> = "x".repeat(1 << 20).into();
    let column: ArrayRef = Arc::new(TimestampSecondArray::from(vec![1]).with_timezone(timezone));
    let schema = Arc::new(Schema::new(vec![Field::new(
        "time",
        column.data_type().clone(),
        false,
    )]));
    row.batch = Arc::new(PayloadBatch {
        key: (0, 1),
        record: Arc::new(RecordBatch::try_new(schema, vec![column]).unwrap()),
        body_bytes: 0,
        encoded_charge_bytes: 0,
        encoded: OnceLock::new(),
    });
    let pool: Arc<dyn MemoryPool> = Arc::new(GreedyMemoryPool::new(32 << 20));
    let mut credit = MemoryConsumer::new("type-test").register(&pool);
    let mut builder = OutputPlanBuilder::new(1, None, &mut credit, "asof").unwrap();
    builder.push(row.view(), None, &mut credit, "asof").unwrap();
    assert!(credit.size() >= 1 << 20);
    drop((row, builder, credit));
    assert_eq!(pool.reserved(), 0);
}
