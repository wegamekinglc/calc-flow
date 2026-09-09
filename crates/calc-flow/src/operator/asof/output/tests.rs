use super::*;
use crate::{AsofJoinSide, AsofStateLimits, StreamingFailureReason};
use datafusion::{
    arrow::array::{Int64Array, StringArray, TimestampMicrosecondArray},
    arrow::datatypes::TimeUnit,
    execution::memory_pool::MemoryConsumer,
};
use std::time::Duration;

fn fixture() -> (StreamAsofJoinSpec, [SchemaRef; 3], StateSegment) {
    let schema = Arc::new(Schema::new(vec![
        Field::new("key", DataType::Utf8, false),
        Field::new(
            "time",
            DataType::Timestamp(TimeUnit::Microsecond, Some("UTC".into())),
            false,
        ),
        Field::new("seq", DataType::Int64, false),
    ]));
    let side = |prefix: &str| {
        AsofJoinSide::new(
            vec!["key".into()],
            "time".into(),
            vec!["seq".into()],
            prefix.into(),
        )
        .unwrap()
    };
    let spec = StreamAsofJoinSpec::new(
        side("left"),
        side("right"),
        Duration::ZERO,
        AsofStateLimits::new(100, 1_048_576).unwrap(),
    )
    .unwrap();
    let output = super::super::schema::output_schema(&spec, &schema, &schema).unwrap();
    let row = RecordBatch::try_new(
        schema.clone(),
        vec![
            Arc::new(StringArray::from(vec!["A"])),
            Arc::new(TimestampMicrosecondArray::from(vec![100]).with_timezone("UTC")),
            Arc::new(Int64Array::from(vec![1])),
        ],
    )
    .unwrap();
    let bytes = StateSegment::new(codec::encode_batch(&row, 1_048_576).unwrap());
    (spec, [schema.clone(), schema, output], bytes)
}

#[tokio::test]
async fn cancelled_materialization_deregisters_candidates_before_reusing_runtime() {
    let (spec, schemas, bytes) = fixture();
    let digests = [
        codec::schema_digest(&schemas[0]).unwrap(),
        codec::schema_digest(&schemas[1]).unwrap(),
    ];
    let mut runtime = OutputRuntime::new(1_048_576);
    let context = runtime.context().unwrap().clone();
    let pool = runtime.pool.clone();
    let rows = [(&bytes, Some(&bytes))];
    let workspace = MemoryConsumer::new("test-host-workspace").register(&pool);
    workspace.try_grow(1_024).unwrap();
    let mut future = Box::pin(async {
        let _workspace = workspace;
        runtime
            .materialize(&rows, &spec, &schemas, &digests, "asof")
            .await
    });
    assert!(futures::poll!(future.as_mut()).is_pending());
    assert!(context.table_exist("asof_left").unwrap());
    assert!(context.table_exist("asof_right").unwrap());
    drop(future);
    assert!(!context.table_exist("asof_left").unwrap());
    assert!(!context.table_exist("asof_right").unwrap());
    assert_eq!(pool.reserved(), 0);
    let result = runtime
        .materialize(&rows, &spec, &schemas, &digests, "asof")
        .await
        .unwrap();
    assert_eq!(result.table_payload().unwrap().batches()[0].num_rows(), 1);
    assert!(!context.table_exist("asof_left").unwrap());
    assert!(!context.table_exist("asof_right").unwrap());
    assert_eq!(pool.reserved(), 0);
}

#[tokio::test]
async fn failed_datafusion_materialization_deregisters_candidates_and_releases_pool() {
    let (spec, schemas, bytes) = fixture();
    let digests = [
        codec::schema_digest(&schemas[0]).unwrap(),
        codec::schema_digest(&schemas[1]).unwrap(),
    ];
    let mut runtime = OutputRuntime::new(1);
    let context = runtime.context().unwrap().clone();
    let result = runtime
        .materialize(&[(&bytes, Some(&bytes))], &spec, &schemas, &digests, "asof")
        .await;
    assert!(matches!(
        result,
        Err(crate::CalcFlowError::OperatorReason {
            reason_code: StreamingFailureReason::AsofWorkspaceLimitExceeded,
            ..
        })
    ));
    assert!(!context.table_exist("asof_left").unwrap());
    assert!(!context.table_exist("asof_right").unwrap());
    assert_eq!(runtime.pool.reserved(), 0);
}
