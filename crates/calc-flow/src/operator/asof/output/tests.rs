use super::super::{
    codec,
    output_plan::OutputPlanBuilder,
    state::{PayloadBatch, RowPayload},
};
use super::*;
use crate::runtime::streaming::gather_work::{GatherScope, GatherTicket};
use crate::{AsofJoinSide, AsofStateLimits, StateSegment, StreamAsofJoinSpec};
use datafusion::execution::memory_pool::MemoryConsumer;
use datafusion::{
    arrow::array::{Int64Array, StringArray, TimestampMicrosecondArray},
    arrow::datatypes::{DataType, Field, Schema, TimeUnit},
};
use std::sync::atomic::{AtomicBool, Ordering};
use std::time::Duration;

mod all_shared;

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

#[test]
fn complete_left_span_shares_its_arrow_values_buffer() {
    let (_, schemas, row) = fixture();
    let result = materialize_rows(&[(row.view(), None)], &schemas[2]).unwrap();
    let source = row.batch.record.column(2).to_data();
    let output = result.table_payload().unwrap().batches()[0]
        .column(2)
        .to_data();
    assert_eq!(source.buffers()[0].as_ptr(), output.buffers()[0].as_ptr());
}

#[tokio::test]
async fn output_plan_empty_projection_preserves_row_count() {
    let (_, _, row) = fixture();
    let schema = Arc::new(Schema::empty());
    let rows = [
        (row.view(), Some(row.view())),
        (row.view(), None),
        (row.view(), Some(row.view())),
    ];
    let mut runtime = OutputRuntime::new(1_048_576, "asof");
    runtime.set_output_projection(Vec::new());
    let reservation = MemoryConsumer::new("test-output").register(&runtime.pool);
    let (result, reservation) = runtime
        .materialize(&rows, &schema, reservation, || Ok(()))
        .await
        .unwrap();
    let record = &result.table_payload().unwrap().batches()[0];
    assert_eq!(record.num_columns(), 0);
    assert_eq!(record.num_rows(), 3);
    drop((result, reservation));
    assert_eq!(runtime.pool.reserved(), 0);
}

#[test]
fn complete_left_span_with_larger_backing_still_copies() {
    let (_, schemas, mut row) = fixture();
    let record = RecordBatch::try_new(
        schemas[0].clone(),
        vec![
            row.batch.record.column(0).clone(),
            row.batch.record.column(1).clone(),
            Arc::new(Int64Array::from(vec![7; 1_024]).slice(3, 1)),
        ],
    )
    .unwrap();
    row.batch = Arc::new(PayloadBatch {
        key: (0, 1),
        record: Arc::new(record),
        body_bytes: 0,
        encoded_charge_bytes: 0,
        encoded: std::sync::OnceLock::new(),
    });
    let result = materialize_rows(&[(row.view(), None)], &schemas[2]).unwrap();
    let source = row.batch.record.column(2).to_data();
    let output = result.table_payload().unwrap().batches()[0]
        .column(2)
        .to_data();
    assert_ne!(source.buffers()[0].as_ptr(), output.buffers()[0].as_ptr());
    assert!(output.get_buffer_memory_size() < source.get_buffer_memory_size());
}

#[tokio::test]
async fn cancelled_materialization_keeps_runtime_reusable() {
    let (_, schemas, bytes) = fixture();
    let mut runtime = OutputRuntime::new(1_048_576, "asof");
    let pool = runtime.pool.clone();
    let rows = [(bytes.view(), Some(bytes.view()))];
    let reservation = MemoryConsumer::new("test-output").register(&runtime.pool);
    let mut future = Box::pin(runtime.materialize(&rows, &schemas[2], reservation, || Ok(())));
    assert!(futures::poll!(future.as_mut()).is_pending());
    drop(future);
    assert_eq!(pool.reserved(), 0);
    let reservation = MemoryConsumer::new("test-output").register(&runtime.pool);
    let (result, reservation) = runtime
        .materialize(&rows, &schemas[2], reservation, || Ok(()))
        .await
        .unwrap();
    assert_eq!(result.table_payload().unwrap().batches()[0].num_rows(), 1);
    assert!(pool.reserved() > 0);
    drop((result, reservation));
    assert_eq!(pool.reserved(), 0);
}

#[tokio::test(flavor = "current_thread")]
async fn large_materialization_leaves_executor_available_for_timers() {
    let (_, schemas, row) = fixture();
    let mut runtime = OutputRuntime::new(1_048_576, "asof");
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
    let mut runtime = OutputRuntime::new(1_048_576, "asof");
    let pool = Arc::clone(&runtime.pool);
    let cancellation = crate::CancellationToken::new();
    let job =
        crate::StreamJobContext::new(1, "asof", crate::JsonMap::new(), None, cancellation.clone());
    let context = StreamOperatorContext::new(&job, "asof", None);
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
    let mut runtime = OutputRuntime::new(1_048_576, "asof");
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
    let mut runtime = OutputRuntime::new(1_048_576, "asof");
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
    let paid = pool.reserved();
    drop(future);
    let retained = pool.reserved();
    gate.wait();
    tokio::time::timeout(Duration::from_secs(1), async {
        while pool.reserved() != 0 {
            tokio::task::yield_now().await;
        }
    })
    .await
    .unwrap();
    assert!(paid >= 4_096);
    assert_eq!(retained, paid);
}

#[tokio::test(flavor = "current_thread")]
async fn output_plan_worker_retains_source_backing_and_credit_after_observer_drop() {
    let job = crate::StreamJobContext::new(
        0,
        "asof",
        crate::JsonMap::new(),
        None,
        crate::CancellationToken::new(),
    );
    let context = StreamOperatorContext::new(&job, "asof", None);
    let (_, schemas, mut row) = fixture();
    let record = RecordBatch::try_new(
        schemas[0].clone(),
        vec![
            row.batch.record.column(0).clone(),
            row.batch.record.column(1).clone(),
            Arc::new(Int64Array::from(vec![7; 1_024]).slice(3, 1)),
        ],
    )
    .unwrap();
    row.batch = Arc::new(PayloadBatch {
        key: (0, 1),
        record: Arc::new(record),
        body_bytes: 0,
        encoded_charge_bytes: 0,
        encoded: std::sync::OnceLock::new(),
    });
    let weak = Arc::downgrade(row.batch.record.column(2));
    let backing = row.batch.record.get_array_memory_size();
    let mut runtime = OutputRuntime::new(1_048_576, "asof");
    let pool = runtime.pool.clone();
    let mut workspace = MemoryConsumer::new("test-output").register(&pool);
    let mut builder = OutputPlanBuilder::new(1, None, &mut workspace, "asof").unwrap();
    builder
        .push(row.view(), None, &mut workspace, "asof")
        .unwrap();
    let plan = builder.finish(&schemas[1], &mut workspace, "asof").unwrap();
    let charge = workspace.size();
    assert!(charge >= backing);
    drop(row);
    let (started_tx, started_rx) = std::sync::mpsc::channel();
    let gate = Arc::new(std::sync::Barrier::new(2));
    runtime.worker_gate = Some((started_tx, gate.clone()));
    let mut future =
        Box::pin(runtime.materialize_plan(plan, &schemas[2], workspace, "asof", &context));
    assert!(futures::poll!(future.as_mut()).is_pending());
    assert!(futures::poll!(future.as_mut()).is_pending());
    started_rx.recv_timeout(Duration::from_secs(1)).unwrap();
    let paid = pool.reserved();
    drop(future);
    let retained = pool.reserved();
    let array_retained = weak.upgrade().is_some();
    gate.wait();
    job.gather_owner().close_and_drain().await;
    drop(job);
    tokio::time::timeout(Duration::from_secs(1), async {
        while pool.reserved() != 0 {
            tokio::task::yield_now().await;
        }
    })
    .await
    .unwrap();
    assert!(paid >= charge);
    assert_eq!(retained, paid);
    assert!(array_retained);
    assert!(weak.upgrade().is_none());
}

#[tokio::test(flavor = "current_thread")]
async fn output_plan_worker_holds_arrays_without_source_or_output_schema_metadata() {
    let job = crate::StreamJobContext::new(
        0,
        "asof",
        crate::JsonMap::new(),
        None,
        crate::CancellationToken::new(),
    );
    let context = StreamOperatorContext::new(&job, "asof", None);
    let (_, schemas, mut row) = fixture();
    let schema = Arc::new(
        Schema::new(row.batch.record.schema().fields().clone())
            .with_metadata(HashMap::with_capacity(65_536)),
    );
    let source_schema = Arc::downgrade(&schema);
    row.batch = Arc::new(PayloadBatch {
        key: (0, 1),
        record: Arc::new(
            RecordBatch::try_new(schema, row.batch.record.columns().to_vec()).unwrap(),
        ),
        body_bytes: 0,
        encoded_charge_bytes: 0,
        encoded: std::sync::OnceLock::new(),
    });
    let schema = Arc::new(
        schemas[2]
            .as_ref()
            .clone()
            .with_metadata(HashMap::with_capacity(65_536)),
    );
    let output_schema = Arc::downgrade(&schema);
    let array = Arc::downgrade(row.batch.record.column(2));
    let mut runtime = OutputRuntime::new(128 << 10, "asof");
    let pool = runtime.pool.clone();
    let mut workspace = MemoryConsumer::new("test-output").register(&pool);
    let mut builder = OutputPlanBuilder::new(1, None, &mut workspace, "asof").unwrap();
    builder
        .push(row.view(), None, &mut workspace, "asof")
        .unwrap();
    let plan = builder.finish(&schemas[1], &mut workspace, "asof").unwrap();
    drop((row, schemas));
    assert!(source_schema.upgrade().is_none());
    let (started_tx, started_rx) = std::sync::mpsc::channel();
    let gate = Arc::new(std::sync::Barrier::new(2));
    runtime.worker_gate = Some((started_tx, gate.clone()));
    let mut future = Box::pin(runtime.materialize_plan(plan, &schema, workspace, "asof", &context));
    assert!(futures::poll!(future.as_mut()).is_pending());
    assert!(futures::poll!(future.as_mut()).is_pending());
    started_rx.recv_timeout(Duration::from_secs(1)).unwrap();
    let paid = pool.reserved();
    drop(future);
    drop(schema);
    let schema_released = output_schema.upgrade().is_none();
    let array_retained = array.upgrade().is_some();
    gate.wait();
    job.gather_owner().close_and_drain().await;
    drop(job);
    tokio::time::timeout(Duration::from_secs(1), async {
        while pool.reserved() != 0 {
            tokio::task::yield_now().await;
        }
    })
    .await
    .unwrap();
    assert!(schema_released);
    assert!(array_retained);
    assert!(paid > 0);
    assert!(array.upgrade().is_none());
}

#[tokio::test]
async fn output_plan_unmatched_type_heap_is_prepaid_before_worker_launch() {
    let job = crate::StreamJobContext::new(
        0,
        "asof",
        crate::JsonMap::new(),
        None,
        crate::CancellationToken::new(),
    );
    let context = StreamOperatorContext::new(&job, "asof", None);
    let (_, schemas, row) = fixture();
    let data_type = DataType::Timestamp(TimeUnit::Second, Some("x".repeat(1 << 20).into()));
    let right = Schema::new(vec![Field::new("time", data_type.clone(), true)]);
    let schema = Arc::new(Schema::new(vec![Field::new(
        "right__time",
        data_type,
        true,
    )]));
    let mut runtime = OutputRuntime::new(128 << 10, "asof");
    runtime.set_output_projection(vec![3]);
    let pool = runtime.pool.clone();
    let mut workspace = MemoryConsumer::new("type-test").register(&pool);
    let selected = [Vec::new(), vec![0]];
    let mut builder = OutputPlanBuilder::new(1, Some(&selected), &mut workspace, "asof").unwrap();
    builder
        .push(row.view(), None, &mut workspace, "asof")
        .unwrap();
    let plan = builder.finish(&right, &mut workspace, "asof").unwrap();
    let result = runtime
        .materialize_plan(plan, &schema, workspace, "asof", &context)
        .await;
    assert!(matches!(
        result,
        Err(crate::CalcFlowError::OperatorReason {
            reason_code: crate::StreamingFailureReason::AsofWorkspaceLimitExceeded,
            ..
        })
    ));
    assert_eq!(pool.reserved(), 0);
    drop(schemas);
}

struct ParallelColumnGate {
    released: parking_lot::Mutex<bool>,
    changed: parking_lot::Condvar,
}

impl ParallelColumnGate {
    fn release(&self) {
        *self.released.lock() = true;
        self.changed.notify_all();
    }

    fn wait(&self) {
        let mut released = self.released.lock();
        while !*released {
            self.changed.wait(&mut released);
        }
    }
}

struct ParallelColumnRelease(Arc<ParallelColumnGate>);

impl Drop for ParallelColumnRelease {
    fn drop(&mut self) {
        self.0.release();
    }
}

struct GatedMaterialization {
    input: MaterializationInput,
    workers: Option<usize>,
    entered: std::sync::mpsc::Sender<(usize, std::thread::ThreadId)>,
    gate: Arc<ParallelColumnGate>,
}

impl OwnedGatherPlan for GatedMaterialization {
    fn column_count(&self) -> usize {
        self.input.column_count()
    }

    fn parallelism(&self) -> usize {
        self.workers
            .unwrap_or_else(|| self.column_count().clamp(1, 8))
    }

    fn gather(&self, ordinal: usize, stop: &GatherStop) -> Result<ArrayRef> {
        stop.check()?;
        let _ = self.entered.send((ordinal, std::thread::current().id()));
        self.gate.wait();
        self.input.gather(ordinal, stop)
    }
}

fn parallel_column_input(count: i64) -> (MaterializationInput, std::sync::Weak<dyn Array>) {
    let columns: Vec<ArrayRef> = vec![
        Arc::new(Int64Array::from_iter_values(0..count)),
        Arc::new(Int64Array::from_iter_values((0..count).map(|value| -value))),
    ];
    let source = Arc::downgrade(&columns[0]);
    let rows = usize::try_from(count).unwrap();
    let plan = OutputPlan {
        left: OutputSide {
            batches: vec![columns],
            positions: (0..rows).rev().map(|row| (0, row)).collect(),
            spans: (0..rows)
                .rev()
                .map(|row| Span {
                    source: 0,
                    start: row,
                    end: row + 1,
                })
                .collect(),
            has_nulls: false,
        },
        right: OutputSide {
            batches: Vec::new(),
            positions: Vec::new(),
            spans: Vec::new(),
            has_nulls: false,
        },
        len: rows,
        matched: 0,
        raw_bytes: 0,
    };
    let input = MaterializationInput {
        rows: plan,
        requests: vec![
            ColumnRequest {
                index: 0,
                data_type: DataType::Int64,
            },
            ColumnRequest {
                index: 1,
                data_type: DataType::Int64,
            },
        ],
        worker_gate: None,
        worker_probe: None,
    };
    (input, source)
}

#[test]
fn output_plan_two_copied_columns_enter_distinct_native_workers_before_release() {
    use crate::runtime::streaming::gather_work::TestService;

    let service = TestService::new(2, 1).unwrap();
    let runtime = tokio::runtime::Builder::new_current_thread()
        .enable_all()
        .build()
        .unwrap();
    let job = crate::StreamJobContext::new(
        61,
        "asof",
        crate::JsonMap::new(),
        None,
        crate::CancellationToken::new(),
    )
    .with_gather_owner(service.owner("61".into()));
    let pool: Arc<dyn MemoryPool> = Arc::new(GreedyMemoryPool::new(67_108_864));
    let reservation = MemoryConsumer::new("parallel-asof-columns").register(&pool);
    reservation.try_grow(33_554_432).unwrap();
    let count = 100_000;
    let rows = usize::try_from(count).unwrap();
    let (input, source) = parallel_column_input(count);
    let gate = Arc::new(ParallelColumnGate {
        released: parking_lot::Mutex::new(false),
        changed: parking_lot::Condvar::new(),
    });
    let _release = ParallelColumnRelease(gate.clone());
    let (entered_tx, entered_rx) = std::sync::mpsc::channel();
    let owned = Arc::new(GatedMaterialization {
        input,
        workers: None,
        entered: entered_tx,
        gate: gate.clone(),
    });
    let context = StreamOperatorContext::new(&job, "asof", None);
    let scope = context
        .gather_client(GatherOperatorId::new("operator:asof".into()))
        .scope()
        .unwrap();
    let ticket = runtime
        .block_on(scope.submit(owned, reservation, GatherStop::from_job(&job)))
        .unwrap();
    let first = entered_rx.recv_timeout(Duration::from_secs(3)).ok();
    let second = entered_rx.recv_timeout(Duration::from_secs(3)).ok();
    let retained = source.upgrade().is_some() && pool.reserved() >= 33_554_432;
    gate.release();
    let output = runtime.block_on(ticket.finish()).unwrap();
    let ordered = output.value.iter().enumerate().all(|(ordinal, array)| {
        let values = array.as_any().downcast_ref::<Int64Array>().unwrap();
        values.len() == rows
            && values.values().iter().copied().eq((0..count)
                .rev()
                .map(|value| if ordinal == 0 { value } else { -value }))
    });
    drop(output);
    runtime.block_on(job.gather_owner().close_and_drain());
    let joined = service.joined_workers();
    drop(scope);
    drop(job);
    drop(runtime);
    service.shutdown();
    assert_eq!(pool.reserved(), 0);
    assert!(source.upgrade().is_none());
    assert!(retained && ordered);
    assert!(
        matches!((first, second), (Some((a, thread_a)), Some((b, thread_b))) if a != b && thread_a != thread_b)
    );
    assert_eq!(joined, 2);
}

struct ParallelAttemptProbe {
    ticket: GatherTicket,
    source: std::sync::Weak<dyn Array>,
    gate: Arc<ParallelColumnGate>,
    entered: std::sync::mpsc::Receiver<(usize, std::thread::ThreadId)>,
}

impl ParallelAttemptProbe {
    fn submit(
        runtime: &tokio::runtime::Runtime,
        scope: &GatherScope,
        job: &crate::StreamJobContext,
        pool: &Arc<dyn MemoryPool>,
        workers: usize,
    ) -> Self {
        let reservation = MemoryConsumer::new("reuse-parallel-columns").register(pool);
        reservation.try_grow(33_554_432).unwrap();
        let (mut input, source) = parallel_column_input(100_000);
        for index in 2..8 {
            input.rows.left.batches[0].push(Arc::new(Int64Array::from_iter_values(
                (0..100_000).map(|value| if index % 2 == 0 { value } else { -value }),
            )));
            input.requests.push(ColumnRequest {
                index,
                data_type: DataType::Int64,
            });
        }
        let gate = Arc::new(ParallelColumnGate {
            released: parking_lot::Mutex::new(false),
            changed: parking_lot::Condvar::new(),
        });
        let (entered_tx, entered) = std::sync::mpsc::channel();
        let owned = Arc::new(GatedMaterialization {
            input,
            workers: Some(workers),
            entered: entered_tx,
            gate: gate.clone(),
        });
        let ticket = runtime
            .block_on(scope.submit(owned, reservation, GatherStop::from_job(job)))
            .unwrap();
        Self {
            ticket,
            source,
            gate,
            entered,
        }
    }

    fn finish(self, runtime: &tokio::runtime::Runtime) -> bool {
        self.gate.release();
        let output = runtime.block_on(self.ticket.finish()).unwrap();
        let ordered = output.value.len() == 8
            && output.value.iter().enumerate().all(|(ordinal, array)| {
                let values = array.as_any().downcast_ref::<Int64Array>().unwrap();
                values.len() == 100_000
                    && values.values().iter().copied().eq((0..100_000)
                        .rev()
                        .map(|value| if ordinal % 2 == 0 { value } else { -value }))
            });
        drop(output);
        ordered && self.source.upgrade().is_none()
    }
}

#[test]
fn output_plan_reused_eight_worker_pool_obeys_two_worker_claim_limit() {
    use crate::runtime::streaming::gather_work::TestService;

    let service = TestService::new(8, 1).unwrap();
    let runtime = tokio::runtime::Builder::new_current_thread()
        .enable_all()
        .build()
        .unwrap();
    let job = crate::StreamJobContext::new(
        62,
        "asof",
        crate::JsonMap::new(),
        None,
        crate::CancellationToken::new(),
    )
    .with_gather_owner(service.owner("62".into()));
    let pool: Arc<dyn MemoryPool> = Arc::new(GreedyMemoryPool::new(67_108_864));
    let context = StreamOperatorContext::new(&job, "asof", None);
    let scope = context
        .gather_client(GatherOperatorId::new("operator:asof".into()))
        .scope()
        .unwrap();
    let warm = ParallelAttemptProbe::submit(&runtime, &scope, &job, &pool, 8);
    let release_warm = ParallelColumnRelease(warm.gate.clone());
    let mut warm_threads = std::collections::HashSet::new();
    for _ in 0..8 {
        let Ok((_, thread)) = warm.entered.recv_timeout(Duration::from_secs(3)) else {
            break;
        };
        warm_threads.insert(thread);
    }
    let warm_correct = warm.finish(&runtime);
    drop(release_warm);
    let limited = ParallelAttemptProbe::submit(&runtime, &scope, &job, &pool, 2);
    let release_limited = ParallelColumnRelease(limited.gate.clone());
    let first = limited.entered.recv_timeout(Duration::from_secs(3)).ok();
    let second = limited.entered.recv_timeout(Duration::from_secs(3)).ok();
    let extra = limited
        .entered
        .recv_timeout(Duration::from_millis(100))
        .ok();
    let retained = limited.source.upgrade().is_some() && pool.reserved() >= 33_554_432;
    let limited_correct = limited.finish(&runtime);
    drop(release_limited);
    runtime.block_on(job.gather_owner().close_and_drain());
    let joined = service.joined_workers();
    drop(scope);
    drop(job);
    drop(runtime);
    service.shutdown();
    assert_eq!(pool.reserved(), 0);
    assert!(warm_correct && limited_correct && retained);
    assert_eq!(warm_threads.len(), 8);
    assert_eq!(joined, 8);
    assert!(matches!((first, second), (Some((a, _)), Some((b, _))) if a != b));
    assert!(
        extra.is_none(),
        "a reused pool exceeded the current attempt's two-worker limit"
    );
}

struct NarrowCopyProbe {
    input: MaterializationInput,
    entered: std::sync::mpsc::Sender<(usize, std::ops::Range<usize>, std::thread::ThreadId)>,
    gate: Arc<ParallelColumnGate>,
}

impl OwnedGatherPlan for NarrowCopyProbe {
    fn column_count(&self) -> usize {
        self.input.column_count()
    }

    fn parallelism(&self) -> usize {
        2
    }

    fn row_gather(&self) -> Result<Option<RowGather>> {
        self.input.row_gather()
    }

    fn shared_column(&self, ordinal: usize) -> Result<Option<ArrayRef>> {
        self.input.shared_column(ordinal)
    }

    fn gather_range(
        &self,
        ordinal: usize,
        range: std::ops::Range<usize>,
        stop: &GatherStop,
    ) -> Result<ArrayRef> {
        let _ = self
            .entered
            .send((ordinal, range.clone(), std::thread::current().id()));
        self.gate.wait();
        self.input.gather_range(ordinal, range, stop)
    }

    fn gather(&self, ordinal: usize, stop: &GatherStop) -> Result<ArrayRef> {
        if ordinal == 1 {
            let _ =
                self.entered
                    .send((ordinal, 0..self.input.rows.len, std::thread::current().id()));
            self.gate.wait();
        }
        self.input.gather(ordinal, stop)
    }
}

fn narrow_shared_left_and_copied_right() -> (
    MaterializationInput,
    std::sync::Weak<dyn Array>,
    std::sync::Weak<dyn Array>,
    usize,
) {
    let rows = 100_000;
    let left: ArrayRef = Arc::new(Int64Array::from_iter_values(0..100_000));
    let right: ArrayRef = Arc::new(datafusion::arrow::array::Float64Array::from(
        (0..1_000)
            .map(|value| (value % 11 != 0).then_some(f64::from(value) * 1.25))
            .collect::<Vec<_>>(),
    ));
    let left_weak = Arc::downgrade(&left);
    let right_weak = Arc::downgrade(&right);
    let pointer = left.to_data().buffers()[0].as_ptr() as usize;
    let input = MaterializationInput {
        rows: OutputPlan {
            left: OutputSide {
                batches: vec![vec![left]],
                positions: (0..rows).map(|row| (0, row)).collect(),
                spans: vec![Span {
                    source: 0,
                    start: 0,
                    end: rows,
                }],
                has_nulls: false,
            },
            right: OutputSide {
                batches: vec![vec![right]],
                positions: (0..rows)
                    .map(|row| {
                        if row % 7 == 0 {
                            (0, 0)
                        } else {
                            (1, (rows - row - 1) % 1_000)
                        }
                    })
                    .collect(),
                spans: Vec::new(),
                has_nulls: true,
            },
            len: rows,
            matched: (0..rows).filter(|row| row % 7 != 0).count() as u64,
            raw_bytes: 0,
        },
        requests: vec![
            ColumnRequest {
                index: 0,
                data_type: DataType::Int64,
            },
            ColumnRequest {
                index: 1,
                data_type: DataType::Float64,
            },
        ],
        worker_gate: None,
        worker_probe: None,
    };
    (input, left_weak, right_weak, pointer)
}

fn narrow_values_and_shared_buffer(columns: &[ArrayRef], pointer: usize) -> bool {
    let left = columns[0].as_any().downcast_ref::<Int64Array>().unwrap();
    let right = columns[1]
        .as_any()
        .downcast_ref::<datafusion::arrow::array::Float64Array>()
        .unwrap();
    left.to_data().buffers()[0].as_ptr() as usize == pointer
        && left.values().iter().copied().eq(0..100_000)
        && right.iter().eq((0..100_000).map(|row| {
            let value = (100_000 - row - 1) % 1_000;
            (row % 7 != 0 && value % 11 != 0).then_some(f64::from(value) * 1.25)
        }))
}

#[test]
fn output_plan_one_copied_column_enters_two_row_workers_while_left_stays_shared() {
    use crate::runtime::streaming::gather_work::TestService;

    let service = TestService::new(2, 1).unwrap();
    let runtime = tokio::runtime::Builder::new_current_thread()
        .enable_all()
        .build()
        .unwrap();
    let job = crate::StreamJobContext::new(
        63,
        "asof",
        crate::JsonMap::new(),
        None,
        crate::CancellationToken::new(),
    )
    .with_gather_owner(service.owner("63".into()));
    let pool: Arc<dyn MemoryPool> = Arc::new(GreedyMemoryPool::new(67_108_864));
    let reservation = MemoryConsumer::new("narrow-row-morsels").register(&pool);
    reservation.try_grow(33_554_432).unwrap();
    let (input, left, right, pointer) = narrow_shared_left_and_copied_right();
    let gate = Arc::new(ParallelColumnGate {
        released: parking_lot::Mutex::new(false),
        changed: parking_lot::Condvar::new(),
    });
    let release = ParallelColumnRelease(gate.clone());
    let (entered_tx, entered) = std::sync::mpsc::channel();
    let context = StreamOperatorContext::new(&job, "asof", None);
    let scope = context
        .gather_client(GatherOperatorId::new("operator:asof".into()))
        .scope()
        .unwrap();
    let input = Arc::new(NarrowCopyProbe {
        input,
        entered: entered_tx,
        gate: gate.clone(),
    });
    let ticket = runtime
        .block_on(scope.submit(input, reservation, GatherStop::from_job(&job)))
        .unwrap();
    let first = entered.recv_timeout(Duration::from_secs(3)).ok();
    let second = entered.recv_timeout(Duration::from_secs(3)).ok();
    let paid =
        left.upgrade().is_some() && right.upgrade().is_some() && pool.reserved() >= 33_554_432;
    gate.release();
    let output = runtime.block_on(ticket.finish()).unwrap();
    let exact = narrow_values_and_shared_buffer(&output.value, pointer);
    let input_lifecycle = left.upgrade().is_some() && right.upgrade().is_none();
    drop(output);
    drop(release);
    runtime.block_on(job.gather_owner().close_and_drain());
    let joined = service.joined_workers();
    drop(scope);
    drop(job);
    drop(runtime);
    service.shutdown();
    assert_eq!(pool.reserved(), 0);
    assert!(left.upgrade().is_none() && right.upgrade().is_none());
    assert!(paid && exact && input_lifecycle);
    assert_eq!(joined, 2);
    assert!(
        matches!((first, second), (Some((1, a, ta)), Some((1, b, tb)))
        if ta != tb && !a.is_empty() && !b.is_empty()
                && a.end <= 100_000 && b.end <= 100_000
                && ((a.start == 0 && a.end == b.start && b.end == 100_000)
                    || (b.start == 0 && b.end == a.start && a.end == 100_000)))
    );
}
