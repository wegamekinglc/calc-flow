use super::*;
use crate::{
    CancellationToken, Epoch, JsonMap, OperatorStateSnapshot, StreamJobContext, StreamOperator,
    operator::join::{
        JoinStateLimits, JoinTimeBounds, StreamJoinOperator, StreamJoinSpec, encode_join_key_v1,
        encode_side, state_row_charge,
    },
    runtime::streaming::gather_work::TestService,
};
use datafusion::arrow::{
    array::{Int64Array, TimestampMicrosecondArray},
    datatypes::{DataType, Field, Schema, TimeUnit},
    record_batch::RecordBatch,
};
use datafusion::execution::memory_pool::MemoryPool;
use std::{
    future::Future,
    sync::atomic::{AtomicUsize, Ordering},
    task::{Context, Poll, Waker},
    time::Duration,
};

fn operator() -> StreamJoinOperator {
    let schema = Arc::new(Schema::new(vec![
        Field::new("key", DataType::Int64, false),
        Field::new(
            "time",
            DataType::Timestamp(TimeUnit::Microsecond, None),
            false,
        ),
    ]));
    let spec = StreamJoinSpec::inner(
        ["key"],
        ["key"],
        "time",
        "time",
        JoinTimeBounds::new(Duration::ZERO, Duration::ZERO).unwrap(),
        JoinStateLimits::new(100, 100_000, 100).unwrap(),
    )
    .unwrap();
    let mut output = StreamJoinOperator::new("match", Arc::clone(&schema), schema, spec).unwrap();
    output.prepare_checkpoint_preload_runtime().unwrap();
    output
}

fn job(service: &TestService, cancellation: CancellationToken) -> StreamJobContext {
    StreamJobContext::new(1, "fingerprint", JsonMap::new(), None, cancellation)
        .with_gather_owner(service.owner("segments-test".into()))
}

fn pool(operator: &StreamJoinOperator) -> Arc<dyn MemoryPool> {
    operator
        .runtime
        .runtime
        .as_ref()
        .unwrap()
        .incremental_memory_pool()
}

fn row(operator: &StreamJoinOperator, id: u64) -> StoredRow {
    let record = RecordBatch::try_new(
        Arc::clone(operator.input_schema(0)),
        vec![
            Arc::new(Int64Array::from(vec![7])),
            Arc::new(TimestampMicrosecondArray::from(vec![10])),
        ],
    )
    .unwrap();
    let charge = state_row_charge(
        &record,
        0,
        &operator.compiled.left_key_indices,
        &operator.name,
    )
    .unwrap();
    let encoded_key = encode_join_key_v1(&record, 0, &operator.compiled.left_key_indices).unwrap();
    StoredRow {
        record: record.into(),
        event_time: crate::EventTime::from_micros(10),
        row_id: id,
        charge,
        encoded_key: Arc::new(encoded_key.into()),
    }
}

fn snapshot(operator: &mut StreamJoinOperator) -> OperatorStateSnapshot {
    let left = (0..4).map(|id| row(operator, id)).collect::<Vec<_>>();
    let right = vec![row(operator, 0)];
    operator.state.metrics.left.retained_rows = 4;
    operator.state.metrics.left.retained_bytes = left.iter().map(|row| row.charge).sum();
    operator.state.metrics.right.retained_rows = 1;
    operator.state.metrics.right.retained_bytes = right[0].charge;
    operator.state.left = left.into();
    operator.state.right = right.into();
    operator.state.next_left_row_id = 4;
    operator.state.next_right_row_id = 1;
    let mut snapshot = operator.checkpoint_v1(Epoch::new(7).unwrap()).unwrap();
    for (side, rows) in [
        ("left", &operator.state.left),
        ("right", &operator.state.right),
    ] {
        snapshot.segments.insert(
            format!("{side}-base"),
            StateSegment::new(
                encode_side(
                    if side == "left" { &rows[..3] } else { rows },
                    &operator.name,
                    side,
                    &|| Ok(()),
                )
                .unwrap(),
            ),
        );
    }
    snapshot.segments.insert(
        "left-delta-9".into(),
        StateSegment::new(delta(&[upsert(&operator.state.left[3])])),
    );
    snapshot
}

fn delta(ops: &[crate::operator::join::PendingOp]) -> Vec<u8> {
    let mut bytes = crate::operator::join::JOIN_DELTA_MAGIC.to_vec();
    bytes.extend_from_slice(&u64::try_from(ops.len()).unwrap().to_le_bytes());
    let mut encoder = crate::operator::join::row_ipc::RowIpcEncoder::default();
    for op in ops {
        crate::operator::join::encode_delta_op(&mut bytes, op, &mut encoder, "match").unwrap();
    }
    bytes
}

fn upsert(row: &StoredRow) -> crate::operator::join::PendingOp {
    crate::operator::join::PendingOp::Upsert {
        side: crate::operator::join::JoinSide::Left,
        row_id: row.row_id,
        event_time: row.event_time,
        encoded_key: row.encoded_key.clone(),
        record: row.record.clone(),
        charge: row.charge,
    }
}

fn construction(
    operator: &StreamJoinOperator,
    snapshot: &OperatorStateSnapshot,
    job: &StreamJobContext,
) -> RestoreSegmentsConstruction {
    poll_copy(operator.segments_construction(snapshot, job))
        .unwrap()
        .unwrap()
}

fn work(construction: &mut RestoreSegmentsConstruction) -> RestoreSegmentsWork {
    RestoreSegmentsWork {
        schema: Some(construction.schema_work(None, None)),
        input: construction.input.take(),
        workspace: construction.workspace.take(),
        hook: None,
        schema_hook: None,
    }
}

fn poll_copy<F: Future>(future: F) -> F::Output {
    let mut future = std::pin::pin!(future);
    let mut context = Context::from_waker(Waker::noop());
    loop {
        if let Poll::Ready(result) = future.as_mut().poll(&mut context) {
            return result;
        }
    }
}

fn prepared_rows(decision: &RestoreSegmentsDecision) -> usize {
    let RestoreSegmentsDecision::Prepared(prepared) = decision else {
        panic!("paid base rows")
    };
    prepared.left.len() + prepared.right.len()
}

async fn submit(
    operator: &mut StreamJoinOperator,
    construction: &mut RestoreSegmentsConstruction,
    job: &StreamJobContext,
) -> ObservedTicket<RestoreSegmentsDecision> {
    construction.schema.metadata.control.scope = Some(
        job.gather_owner()
            .client(GatherOperatorId::new(Arc::from("match")))
            .scope()
            .unwrap(),
    );
    construction
        .submit(
            &mut operator.compaction_cleanup,
            None,
            None,
            operator.decoded_row_test_hook.clone(),
        )
        .await
        .unwrap()
}

fn assert_home_credit(job: &StreamJobContext, pool: &Arc<dyn MemoryPool>) {
    let (home, generation, attempt) = job.gather_owner().funding();
    assert_eq!((generation, attempt), (0, 0));
    assert!(home > 0);
    assert_eq!(pool.reserved(), home);
}

async fn check_cancel_during_decode(service: &TestService) -> usize {
    let snapshot = snapshot(&mut operator());
    let mut target = operator();
    let cancellation = CancellationToken::new();
    let job = job(service, cancellation.clone());
    let pool = pool(&target);
    let readers = Arc::new(AtomicUsize::new(0));
    let observed = Arc::clone(&readers);
    target.decoded_row_test_hook = Some(Arc::new(move |credit, _, decoded, owned| {
        if !decoded && observed.fetch_add(1, Ordering::Relaxed) == 0 {
            assert_eq!(std::thread::current().name(), Some("calc-flow-gather"));
            assert!(owned);
            assert!(credit.unwrap().size() > 0);
            cancellation.cancel();
        }
    }));
    let mut construction = construction(&target, &snapshot, &job);
    assert!(construction.copy(&target, &snapshot, &job).await.unwrap());
    let ticket = submit(&mut target, &mut construction, &job).await;
    let result = ticket.finish().await;
    assert!(matches!(
        result,
        Err(crate::CalcFlowError::Cancelled { .. })
    ));
    assert_eq!(target.status().left.retained_rows, 0);
    assert_eq!(target.state.last_checkpoint_epoch, None);
    assert!(target.compaction_cleanup.is_some());
    drop(result);
    drop(construction);
    drop(target);
    job.gather_owner().close_and_drain().await;
    assert_home_credit(&job, &pool);
    drop(job);
    assert_eq!(pool.reserved(), 0);
    readers.load(Ordering::Relaxed)
}

#[test]
fn test_restore_segments_last_buffer_and_abandoned_worker_keep_real_credit() {
    let service = TestService::new(1, 1).unwrap();
    let runtime = tokio::runtime::Builder::new_current_thread()
        .enable_all()
        .build()
        .unwrap();
    runtime.block_on(check_last_buffer(&service));
    runtime.block_on(check_abandoned_worker(&service));
    drop(runtime);
    service.shutdown();
}

async fn check_last_buffer(service: &TestService) {
    let snapshot = snapshot(&mut operator());
    let mut target = operator();
    let pool = pool(&target);
    let job = job(service, CancellationToken::new());
    let mut construction = construction(&target, &snapshot, &job);
    assert!(construction.copy(&target, &snapshot, &job).await.unwrap());
    let ticket = submit(&mut target, &mut construction, &job).await;
    let output = ticket.finish().await.unwrap();
    let mut escaped = None;
    output
        .install(|decision| {
            let RestoreSegmentsDecision::Prepared(prepared) = decision else {
                panic!("prepared")
            };
            let record = &prepared
                .left
                .iter()
                .find(|row| row.row_id == 3)
                .unwrap()
                .record;
            let column = record.columns()[0].to_data();
            escaped = Some((
                column.buffers()[0].clone(),
                Arc::downgrade(record.schema_ref()),
                record.funded_owner().unwrap().1,
            ));
            Ok(())
        })
        .unwrap();
    let stop = construction
        .schema
        .metadata
        .control
        .stop
        .as_ref()
        .unwrap()
        .clone();
    target.wait_metadata_cleanup(&stop, &job).await.unwrap();
    drop(construction);
    drop(target);
    let (buffer, schema, paid) = escaped.unwrap();
    let (home, generation, attempt) = job.gather_owner().funding();
    assert_eq!(attempt, 0);
    assert_eq!(pool.reserved(), home + generation + paid);
    assert!(schema.upgrade().is_some());
    assert_eq!(i64::from_ne_bytes(buffer.as_slice().try_into().unwrap()), 7);
    {
        let mut drain = std::pin::pin!(job.gather_owner().close_and_drain());
        assert!(std::future::poll_fn(|cx| Poll::Ready(drain.as_mut().poll(cx).is_pending())).await);
        assert!(pool.reserved() >= paid);
        drop(buffer);
        drain.await;
    }
    assert!(schema.upgrade().is_none());
    assert_home_credit(&job, &pool);
    drop(job);
    assert_eq!(pool.reserved(), 0);
}

async fn check_abandoned_worker(service: &TestService) {
    let snapshot = snapshot(&mut operator());
    let mut target = operator();
    let job = job(service, CancellationToken::new());
    let pool = pool(&target);
    let (entered, started) = tokio::sync::oneshot::channel();
    let (release, wait) = std::sync::mpsc::channel();
    let entered = std::sync::Mutex::new(Some(entered));
    let wait = std::sync::Mutex::new(wait);
    target.decoded_row_test_hook = Some(Arc::new(move |credit, _, decoded, owned| {
        if !decoded {
            assert!(owned);
            if let Some(entered) = entered.lock().unwrap().take() {
                entered.send(credit.unwrap().size()).unwrap();
                wait.lock()
                    .unwrap()
                    .recv_timeout(Duration::from_secs(10))
                    .unwrap();
            }
        }
    }));
    let mut construction = construction(&target, &snapshot, &job);
    assert!(construction.copy(&target, &snapshot, &job).await.unwrap());
    let ticket = submit(&mut target, &mut construction, &job).await;
    let paid = started.await.unwrap();
    let stop = construction
        .schema
        .metadata
        .control
        .stop
        .as_ref()
        .unwrap()
        .clone();
    drop(ticket);
    drop(construction);
    {
        let mut cleanup = std::pin::pin!(
            target
                .compaction_cleanup
                .as_ref()
                .unwrap()
                .wait_job(&stop, &job)
        );
        assert!(
            std::future::poll_fn(|cx| Poll::Ready(cleanup.as_mut().poll(cx).is_pending())).await
        );
        assert!(pool.reserved() >= paid);
        release.send(()).unwrap();
        cleanup.await.unwrap();
    }
    assert_eq!(target.status().left.retained_rows, 0);
    job.gather_owner().close_and_drain().await;
    assert_home_credit(&job, &pool);
    drop(job);
    assert_eq!(pool.reserved(), 0);
}

fn old_route_probe(target: &mut StreamJoinOperator) -> Arc<AtomicUsize> {
    let parses = Arc::new(AtomicUsize::new(0));
    let observed_parses = Arc::clone(&parses);
    target.metadata_test_hook = Some(Arc::new(move |credit, parsing| {
        if parsing {
            assert_eq!(std::thread::current().name(), Some("calc-flow-gather"));
            observed_parses.fetch_add(1, Ordering::Relaxed);
        } else {
            assert!(credit.unwrap().size() > 0);
        }
    }));
    target.decoded_row_test_hook = Some(Arc::new(|credit, _, decoded, owned| {
        assert!(credit.is_none());
        assert!(!owned);
        assert!(decoded || std::thread::current().name() != Some("calc-flow-gather"));
    }));
    parses
}

async fn finish_refusal(
    target: &mut StreamJoinOperator,
    snapshot: &OperatorStateSnapshot,
    job: StreamJobContext,
    parses: &AtomicUsize,
) {
    let pool = pool(target);
    target
        .restore_managed_metadata(snapshot, &job, None)
        .await
        .unwrap();
    assert_eq!(parses.load(Ordering::Relaxed), 1);
    assert_eq!(target.status().left.retained_rows, 4);
    assert_eq!(target.status().right.retained_rows, 1);
    assert!(target.compaction_cleanup.is_none());
    job.gather_owner().close_and_drain().await;
    drop(job);
    assert_eq!(pool.reserved(), 0);
}

async fn check_resident_refusal(service: &TestService) {
    use datafusion::execution::memory_pool::{MemoryConsumer, MemoryLimit};
    let snapshot = snapshot(&mut operator());
    let mut target = operator();
    let pool = pool(&target);
    let original =
        super::super::super::inventory::required(&snapshot, &target.spec, &target.name).unwrap();
    let (schema_input, schema_output) =
        super::super::inventory::required([target.input_schema(0), target.input_schema(1)])
            .unwrap();
    let job = job(service, CancellationToken::new());
    let geometry = poll_copy(frame::scan(&snapshot, &job)).unwrap().unwrap();
    let bounds = inventory::required(
        &geometry,
        frame::delta_count(&snapshot, &geometry).unwrap(),
        [target.input_schema(0), target.input_schema(1)],
        [
            &target.compiled.left_key_indices,
            &target.compiled.right_key_indices,
        ],
        &target.name,
    )
    .unwrap();
    assert!(bounds.input + bounds.workspace + 4 * bounds.resident[0] + bounds.resident[1] > 65_536);
    let MemoryLimit::Finite(limit) = pool.memory_limit() else {
        panic!("configured finite pool")
    };
    let headroom = original + schema_input + schema_output + 65_536;
    let pressure = MemoryConsumer::new("restore-bases-refusal").register(&pool);
    pressure.try_grow(limit - headroom).unwrap();
    let parses = old_route_probe(&mut target);
    target
        .restore_managed_metadata(&snapshot, &job, None)
        .await
        .unwrap();
    assert_eq!(parses.load(Ordering::Relaxed), 1);
    assert_eq!(target.status().left.retained_rows, 4);
    assert!(target.compaction_cleanup.is_none());
    let (home, generation, attempt) = job.gather_owner().funding();
    assert_eq!(attempt, 0);
    assert_eq!(pool.reserved(), pressure.size() + home + generation);
    job.gather_owner().close_and_drain().await;
    drop(job);
    assert_eq!(pool.reserved(), pressure.size());
    drop(pressure);
    assert_eq!(pool.reserved(), 0);
}

async fn check_attempt_refusal(service: &TestService) {
    use crate::runtime::streaming::gather_work::admission_probe::{AdmissionProbe, AdmissionStage};
    use datafusion::execution::memory_pool::MemoryLimit;
    let snapshot = snapshot(&mut operator());
    let mut target = operator();
    let pool = pool(&target);
    let job = job(service, CancellationToken::new());
    let MemoryLimit::Finite(limit) = pool.memory_limit() else {
        panic!("configured finite pool")
    };
    let probe = AdmissionProbe::install(
        job.gather_owner(),
        AdmissionStage::Attempt,
        Arc::clone(&pool),
        limit,
    );
    let parses = old_route_probe(&mut target);
    finish_refusal(&mut target, &snapshot, job, &parses).await;
    let event = probe.take_event().unwrap();
    assert_eq!(event.stage, AdmissionStage::Attempt);
    assert_eq!(event.available + 1, event.fee);
}

async fn check_final_cancel(service: &TestService) {
    let snapshot = snapshot(&mut operator());
    let mut target = operator();
    let cancellation = CancellationToken::new();
    let job = job(service, cancellation.clone());
    let pool = pool(&target);
    let mut construction = construction(&target, &snapshot, &job);
    assert!(construction.copy(&target, &snapshot, &job).await.unwrap());
    let ticket = submit(&mut target, &mut construction, &job).await;
    let output = ticket.finish().await.unwrap();
    cancellation.cancel();
    let error = output
        .install(|decision| target.install_segments_decision(decision, &snapshot, &job))
        .unwrap_err();
    assert!(matches!(error, crate::CalcFlowError::Cancelled { .. }));
    assert_eq!(target.status().left.retained_rows, 0);
    assert_eq!(target.state.last_checkpoint_epoch, None);
    assert!(target.compaction_cleanup.is_some());
    drop(construction);
    job.gather_owner().close_and_drain().await;
    assert_home_credit(&job, &pool);
    drop(job);
    assert_eq!(pool.reserved(), 0);
}

#[test]
fn test_restore_segments_refusal_and_cancellation_preserve_original_routes() {
    let service = TestService::new(1, 1).unwrap();
    let runtime = tokio::runtime::Builder::new_current_thread()
        .enable_all()
        .build()
        .unwrap();
    runtime.block_on(check_resident_refusal(&service));
    runtime.block_on(check_attempt_refusal(&service));
    assert_eq!(runtime.block_on(check_cancel_during_decode(&service)), 1);
    runtime.block_on(check_final_cancel(&service));
    drop(runtime);
    service.shutdown();
}

fn ordered_snapshot() -> OperatorStateSnapshot {
    use crate::operator::join::{JoinSide, PendingOp};
    let mut source = operator();
    let previous = snapshot(&mut source);
    let removed = source.state.left.remove(3);
    source.state.metrics.left.retained_rows = 3;
    source.state.metrics.left.retained_bytes -= removed.charge;
    let ops = [
        PendingOp::Tombstone {
            side: JoinSide::Left,
            row_id: removed.row_id,
            event_time: removed.event_time,
            encoded_key: removed.encoded_key,
        },
        PendingOp::Tombstone {
            side: JoinSide::Left,
            row_id: 1,
            event_time: crate::EventTime::from_micros(10),
            encoded_key: Arc::new(vec![42; 2_048].into()),
        },
    ];
    let mut snapshot = source.checkpoint_v1(Epoch::new(11).unwrap()).unwrap();
    snapshot.segments = previous.segments;
    snapshot
        .segments
        .insert("left-delta-10".into(), StateSegment::new(delta(&ops)));
    snapshot
}

#[test]
fn test_restore_segments_fold_keeps_numeric_order_wire_and_original_errors() {
    let service = TestService::new(1, 1).unwrap();
    let runtime = tokio::runtime::Builder::new_current_thread()
        .enable_all()
        .build()
        .unwrap();
    runtime.block_on(check_order_and_wire(&service));
    runtime.block_on(check_duplicate_before_tag(&service));
    drop(runtime);
    service.shutdown();
}

async fn check_order_and_wire(service: &TestService) {
    let snapshot = ordered_snapshot();
    let original_metadata = snapshot.inline_metadata.clone();
    let original_segments = snapshot.segments.clone();
    let mut target = operator();
    let pool = pool(&target);
    let job = job(service, CancellationToken::new());
    let geometry = frame::scan(&snapshot, &job).await.unwrap().unwrap();
    assert_eq!(geometry.rows, [4, 1]);
    assert_eq!(geometry.segments, 4);
    assert_eq!(geometry.longest_key, 2_048);
    target
        .restore_managed_metadata(&snapshot, &job, None)
        .await
        .unwrap();
    assert_eq!(
        target
            .state
            .left
            .iter()
            .map(|row| row.row_id)
            .collect::<Vec<_>>(),
        [0, 1, 2]
    );
    assert_eq!(target.status().left.retained_rows, 3);
    assert_eq!(target.status().right.retained_rows, 1);
    let mut original = operator();
    original.restore(&snapshot).unwrap();
    let actual = target.checkpoint_v1(Epoch::new(12).unwrap()).unwrap();
    let expected = original.checkpoint_v1(Epoch::new(12).unwrap()).unwrap();
    assert_eq!(actual.inline_metadata, expected.inline_metadata);
    assert_eq!(actual.segments, expected.segments);
    assert_eq!(snapshot.inline_metadata, original_metadata);
    assert_eq!(snapshot.segments, original_segments);
    drop(actual);
    drop(target);
    job.gather_owner().close_and_drain().await;
    assert_home_credit(&job, &pool);
    drop(job);
    assert_eq!(pool.reserved(), 0);
}

async fn check_duplicate_before_tag(service: &TestService) {
    let mut snapshot = snapshot(&mut operator());
    let duplicated = row(&operator(), 3);
    let mut bytes = delta(&[upsert(&duplicated), upsert(&duplicated)]);
    let mut cursor = frame::Cursor::new(&bytes, true).unwrap();
    let range = cursor.next().unwrap().ipc.unwrap();
    bytes[range.end] = 255;
    snapshot
        .segments
        .insert("left-delta-9".into(), StateSegment::new(bytes));
    let mut target = operator();
    let pool = pool(&target);
    let job = job(service, CancellationToken::new());
    assert!(
        target
            .segments_construction(&snapshot, &job)
            .await
            .unwrap()
            .is_none()
    );
    let original = operator().restore(&snapshot).unwrap_err();
    let error = target
        .restore_managed_metadata(&snapshot, &job, None)
        .await
        .unwrap_err();
    assert_eq!(error.to_string(), original.to_string());
    assert!(error.to_string().contains("repeats one row identity"));
    assert_eq!(target.status().left.retained_rows, 0);
    assert_eq!(target.state.last_checkpoint_epoch, None);
    drop(target);
    job.gather_owner().close_and_drain().await;
    assert_home_credit(&job, &pool);
    drop(job);
    assert_eq!(pool.reserved(), 0);
}

#[test]
fn test_restore_segments_allocations_cover_cumulative_rows_keys_and_full_tail() {
    let service = TestService::new(1, 1).unwrap();
    let runtime = tokio::runtime::Builder::new_current_thread()
        .enable_all()
        .build()
        .unwrap();
    let entered = runtime.enter();
    let job = job(&service, CancellationToken::new());
    check_cumulative_allocations(&job);
    check_second_next(&job);
    drop(job);
    drop(entered);
    drop(runtime);
    service.shutdown();
}

fn check_cumulative_allocations(job: &StreamJobContext) {
    let snapshot = ordered_snapshot();
    let mut target = operator();
    let pool = pool(&target);
    let mut construction = construction(&target, &snapshot, job);
    let funded = pool.reserved();
    let measured = allocation_counter::measure(|| {
        assert!(poll_copy(construction.copy(&target, &snapshot, job)).unwrap());
        let decision = work(&mut construction)
            .run(construction.schema.metadata.control.stop.as_ref().unwrap())
            .unwrap();
        assert_eq!(prepared_rows(&decision), 4);
        target
            .install_segments_decision(decision, &snapshot, job)
            .unwrap();
    });
    assert!(
        measured.bytes_total <= u64::try_from(funded).unwrap(),
        "{measured:?}; funded={funded}"
    );
    assert_eq!(target.status().left.retained_rows, 3);
    assert_eq!(target.status().right.retained_rows, 1);
    drop(construction);
    drop(target);
    assert_eq!(pool.reserved(), 0);
    println!(
        "cumulative5/final4 rawkey2048 and full-tail requested allocations {measured:?}; independently paid={funded}"
    );
}

fn extra_batch_snapshot() -> OperatorStateSnapshot {
    use datafusion::arrow::ipc::writer::StreamWriter;
    let mut snapshot = snapshot(&mut operator());
    let record = row(&operator(), 3).record;
    let record = record.view();
    let mut writer = StreamWriter::try_new(Vec::new(), record.schema_ref()).unwrap();
    writer.write(&record).unwrap();
    writer.write(&record).unwrap();
    writer.finish().unwrap();
    let ipc = writer.into_inner().unwrap();
    let previous = snapshot.segments["left-delta-9"].bytes();
    let mut cursor = frame::Cursor::new(previous, true).unwrap();
    let range = cursor.next().unwrap().ipc.unwrap();
    let mut bytes = previous[..range.start - 8].to_vec();
    bytes.extend_from_slice(&u64::try_from(ipc.len()).unwrap().to_le_bytes());
    bytes.extend_from_slice(&ipc);
    snapshot
        .segments
        .insert("left-delta-9".into(), StateSegment::new(bytes));
    snapshot
}

fn check_second_next(job: &StreamJobContext) {
    let snapshot = extra_batch_snapshot();
    let mut target = operator();
    let pool = pool(&target);
    let mut construction = construction(&target, &snapshot, job);
    assert!(poll_copy(construction.copy(&target, &snapshot, job)).unwrap());
    let funded = pool.reserved();
    let mut decision = None;
    let measured = allocation_counter::measure(|| {
        decision = Some(
            work(&mut construction)
                .run(construction.schema.metadata.control.stop.as_ref().unwrap())
                .unwrap(),
        );
    });
    assert!(matches!(
        decision,
        Some(RestoreSegmentsDecision::Original(_))
    ));
    assert!(
        measured.bytes_total <= u64::try_from(funded).unwrap(),
        "{measured:?}; funded={funded}"
    );
    let error = target
        .install_segments_decision(decision.unwrap(), &snapshot, job)
        .unwrap_err();
    let original = operator().restore(&snapshot).unwrap_err();
    assert_eq!(error.to_string(), original.to_string());
    assert!(error.to_string().contains("extra record batches"));
    assert_eq!(target.status().left.retained_rows, 0);
    drop(construction);
    assert_eq!(pool.reserved(), 0);
    println!(
        "delta second-next requested allocations {measured:?}; independently paid={funded}; negative net excludes preexisting owners, total conservatively bounds new peak"
    );
}
