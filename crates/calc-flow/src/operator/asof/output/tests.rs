use super::super::{
    codec,
    state::{PayloadBatch, RowPayload},
};
use super::*;
use crate::{AsofJoinSide, AsofStateLimits, StateSegment, StreamAsofJoinSpec};
use datafusion::execution::memory_pool::MemoryConsumer;
use datafusion::{
    arrow::array::{Int64Array, StringArray, TimestampMicrosecondArray},
    arrow::datatypes::{DataType, Field, Schema, TimeUnit},
};
use std::sync::atomic::{AtomicBool, Ordering};
use std::time::Duration;

fn fixture() -> (StreamAsofJoinSpec, [SchemaRef; 3], RowPayload) {
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
    let bytes = StateSegment::new(codec::encode_batch(&row, 1_048_576, &mut Vec::new()).unwrap());
    let payload = RowPayload {
        batch: Arc::new(PayloadBatch {
            key: (0, 0),
            record: Arc::new(row),
            body_bytes: codec::payload_body_bytes(bytes.bytes()).unwrap(),
            encoded_charge_bytes: bytes.bytes().len() as u64,
            encoded: std::sync::OnceLock::from(bytes),
        }),
        row: 0,
    };
    (spec, [schema.clone(), schema, output], payload)
}

#[test]
fn direct_materialization_preserves_order_and_missing_right_rows() {
    let (_, schemas, first) = fixture();
    let next = RecordBatch::try_new(
        schemas[0].clone(),
        vec![
            Arc::new(StringArray::from(vec!["A", "B"])),
            Arc::new(TimestampMicrosecondArray::from(vec![101, 102]).with_timezone("UTC")),
            Arc::new(Int64Array::from(vec![2, 3])),
        ],
    )
    .unwrap();
    let bytes = StateSegment::new(codec::encode_batch(&next, 1_048_576, &mut Vec::new()).unwrap());
    let batch = Arc::new(PayloadBatch {
        key: (0, 1),
        record: Arc::new(next),
        body_bytes: codec::payload_body_bytes(bytes.bytes()).unwrap(),
        encoded_charge_bytes: bytes.bytes().len() as u64,
        encoded: std::sync::OnceLock::from(bytes),
    });
    let second = RowPayload {
        batch: batch.clone(),
        row: 0,
    };
    let third = RowPayload { batch, row: 1 };
    let result = materialize_rows(
        &[
            (third.view(), Some(first.view())),
            (first.view(), None),
            (second.view(), Some(third.view())),
        ],
        &schemas[2],
    )
    .unwrap();
    let table = result.table_payload().unwrap();
    let output = &table.batches()[0];
    let left_seq = output
        .column(2)
        .as_any()
        .downcast_ref::<Int64Array>()
        .unwrap();
    let right_seq = output
        .column(5)
        .as_any()
        .downcast_ref::<Int64Array>()
        .unwrap();
    assert_eq!(left_seq.values().as_ref(), &[3, 1, 2]);
    assert_eq!(
        right_seq.iter().collect::<Vec<_>>(),
        vec![Some(1), None, Some(3)]
    );
}

#[tokio::test]
async fn cancelled_materialization_keeps_runtime_reusable() {
    let (_, schemas, bytes) = fixture();
    let mut runtime = OutputRuntime::new(1_048_576);
    let pool = runtime.pool.clone();
    let rows = [(bytes.view(), Some(bytes.view()))];
    let reservation = MemoryConsumer::new("test-output").register(&runtime.pool);
    let mut future = Box::pin(runtime.materialize(&rows, &schemas[2], reservation, || Ok(())));
    assert!(futures::poll!(future.as_mut()).is_pending());
    drop(future);
    assert_eq!(pool.reserved(), 0);
    let reservation = MemoryConsumer::new("test-output").register(&runtime.pool);
    let (result, _reservation) = runtime
        .materialize(&rows, &schemas[2], reservation, || Ok(()))
        .await
        .unwrap();
    assert_eq!(result.table_payload().unwrap().batches()[0].num_rows(), 1);
    assert_eq!(pool.reserved(), 0);
}

#[tokio::test(flavor = "current_thread")]
async fn large_materialization_leaves_executor_available_for_timers() {
    let (_, schemas, row) = fixture();
    let mut runtime = OutputRuntime::new(1_048_576);
    let rows = (0..64_000)
        .map(|index| (row.view(), (index % 2 == 0).then_some(row.view())))
        .collect::<Vec<_>>();
    let timer_fired = Arc::new(AtomicBool::new(false));
    let signal = timer_fired.clone();
    let timer = tokio::spawn(async move {
        tokio::time::sleep(Duration::from_millis(1)).await;
        signal.store(true, Ordering::SeqCst);
    });
    let reservation = MemoryConsumer::new("test-output").register(&runtime.pool);
    let (result, _reservation) = runtime
        .materialize(&rows, &schemas[2], reservation, || Ok(()))
        .await
        .unwrap();
    assert_eq!(
        result.table_payload().unwrap().batches()[0].num_rows(),
        64_000
    );
    assert!(
        timer_fired.load(Ordering::SeqCst),
        "Arrow output gathering blocked the Tokio executor"
    );
    timer.await.unwrap();
}

#[tokio::test(flavor = "current_thread")]
async fn cancellation_during_manifest_capture_releases_its_workspace() {
    let (_, schemas, row) = fixture();
    let mut runtime = OutputRuntime::new(1_048_576);
    let pool = Arc::clone(&runtime.pool);
    let cancellation = crate::CancellationToken::new();
    let job =
        crate::StreamJobContext::new(1, "asof", crate::JsonMap::new(), None, cancellation.clone());
    let context = crate::StreamOperatorContext::new(&job, "asof", None);
    let calls = std::sync::atomic::AtomicUsize::new(0);
    let rows = vec![(row.view(), Some(row.view())); 2_048];
    let reservation = MemoryConsumer::new("test-output").register(&runtime.pool);
    reservation.try_grow(4_096).unwrap();
    let result = runtime
        .materialize(&rows, &schemas[2], reservation, || {
            if calls.fetch_add(1, Ordering::SeqCst) == 1 {
                cancellation.cancel();
            }
            context.check_cancelled()
        })
        .await;
    assert!(matches!(
        result,
        Err(crate::CalcFlowError::Cancelled { .. })
    ));
    assert_eq!(pool.reserved(), 0);
}

#[tokio::test(flavor = "current_thread")]
async fn materialization_worker_owns_unique_batches_without_row_payload_clones() {
    let (_, schemas, row) = fixture();
    let mut runtime = OutputRuntime::new(1_048_576);
    let (started_tx, started_rx) = std::sync::mpsc::channel();
    let gate = Arc::new(std::sync::Barrier::new(2));
    runtime.worker_gate = Some((started_tx, gate.clone()));
    let rows = vec![(row.view(), Some(row.view())); 128];
    let before = Arc::strong_count(&row.batch);
    let reservation = MemoryConsumer::new("test-output").register(&runtime.pool);
    let mut future = Box::pin(runtime.materialize(&rows, &schemas[2], reservation, || Ok(())));
    assert!(futures::poll!(future.as_mut()).is_pending());
    assert!(futures::poll!(future.as_mut()).is_pending());
    started_rx.recv_timeout(Duration::from_secs(1)).unwrap();
    let during = Arc::strong_count(&row.batch);
    drop(future);
    gate.wait();
    assert_eq!(during, before, "worker cloned a payload owner per row");
}

#[tokio::test(flavor = "current_thread")]
async fn dropped_materialization_keeps_worker_memory_reserved_until_exit() {
    let (_, schemas, row) = fixture();
    let mut runtime = OutputRuntime::new(1_048_576);
    let pool = runtime.pool.clone();
    let (started_tx, started_rx) = std::sync::mpsc::channel();
    let gate = Arc::new(std::sync::Barrier::new(2));
    runtime.worker_gate = Some((started_tx, gate.clone()));
    let reservation = MemoryConsumer::new("test-output").register(&runtime.pool);
    reservation.try_grow(4_096).unwrap();
    let rows = [(row.view(), Some(row.view()))];
    let mut future = Box::pin(runtime.materialize(&rows, &schemas[2], reservation, || Ok(())));
    assert!(futures::poll!(future.as_mut()).is_pending());
    assert!(futures::poll!(future.as_mut()).is_pending());
    started_rx.recv_timeout(Duration::from_secs(1)).unwrap();
    drop(future);
    assert_eq!(pool.reserved(), 4_096);
    gate.wait();
    tokio::time::timeout(Duration::from_secs(1), async {
        while pool.reserved() != 0 {
            tokio::task::yield_now().await;
        }
    })
    .await
    .unwrap();
}
