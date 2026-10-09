use super::*;

#[path = "float_tests.rs"]
mod float_tests;

#[path = "boolean_tests.rs"]
mod boolean_tests;

#[path = "nullable_float_tests.rs"]
mod nullable_float_tests;
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
        .with_gather_owner(service.owner("bases-test".into()))
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
    let left = (0..3).map(|id| row(operator, id)).collect::<Vec<_>>();
    let right = vec![row(operator, 0)];
    operator.state.metrics.left.retained_rows = 3;
    operator.state.metrics.left.retained_bytes = left.iter().map(|row| row.charge).sum();
    operator.state.metrics.right.retained_rows = 1;
    operator.state.metrics.right.retained_bytes = right[0].charge;
    operator.state.left = left.into();
    operator.state.right = right.into();
    operator.state.next_left_row_id = 3;
    operator.state.next_right_row_id = 1;
    let mut snapshot = operator.checkpoint(Epoch::new(7).unwrap()).unwrap();
    for (side, rows) in [
        ("left", &operator.state.left),
        ("right", &operator.state.right),
    ] {
        snapshot.segments.insert(
            format!("{side}-base"),
            StateSegment::new(encode_side(rows, &operator.name, side, &|| Ok(())).unwrap()),
        );
    }
    snapshot
}

fn construction(
    operator: &StreamJoinOperator,
    snapshot: &OperatorStateSnapshot,
    job: &StreamJobContext,
) -> RestoreBasesConstruction {
    operator.bases_construction(snapshot, job).unwrap().unwrap()
}

fn work(construction: &mut RestoreBasesConstruction) -> RestoreBasesWork {
    RestoreBasesWork {
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

fn prepared_rows(decision: &RestoreBasesDecision) -> usize {
    let RestoreBasesDecision::Prepared(prepared) = decision else {
        panic!("paid base rows")
    };
    prepared.left.len() + prepared.right.len()
}

#[test]
fn test_restore_bases_peak_covers_both_sides_and_original_second_next_error() {
    let service = TestService::new(1, 1).unwrap();
    let runtime = tokio::runtime::Builder::new_current_thread()
        .enable_all()
        .build()
        .unwrap();
    let entered = runtime.enter();
    let mut source = operator();
    let snapshot = snapshot(&mut source);
    let job = job(&service, CancellationToken::new());
    let mut target = operator();
    let pool = pool(&target);
    let mut construction = construction(&target, &snapshot, &job);
    let funded = pool.reserved();
    let measured = allocation_counter::measure(|| {
        assert!(poll_copy(construction.copy(&target, &snapshot, &job)).unwrap());
        let decision = work(&mut construction)
            .run(construction.schema.metadata.control.stop.as_ref().unwrap())
            .unwrap();
        assert_eq!(prepared_rows(&decision), 4);
        target
            .install_bases_decision(decision, &snapshot, &job)
            .unwrap();
    });
    assert!(
        measured.bytes_max <= u64::try_from(funded).unwrap(),
        "{measured:?}; funded={funded}"
    );
    assert!(measured.bytes_total <= u64::try_from(funded).unwrap());
    println!("both-side copy/decode requested peak {measured:?}; independent funded={funded}");
    assert_eq!(target.status().left.retained_rows, 3);
    assert_eq!(target.status().right.retained_rows, 1);
    drop(construction);
    drop(target);
    assert_eq!(pool.reserved(), 0);
    check_second_next(&snapshot, &job);
    check_resident_constructor(&job);
    drop(entered);
    drop(runtime);
    service.shutdown();
}

fn check_resident_constructor(job: &StreamJobContext) {
    use datafusion::execution::memory_pool::MemoryConsumer;
    let target = operator();
    let pool = pool(&target);
    let record = row(&target, 0).record;
    let record = record.view();
    let paid = columnar::restored::required(
        target.input_schema(0),
        super::super::super::inventory::registration_controls().unwrap(),
    )
    .unwrap();
    let credit = MemoryConsumer::new("restored-resident-control").register(&pool);
    credit.try_grow(paid).unwrap();
    let lease = columnar::restored::ResidentLease::new(
        credit,
        job.gather_owner().retain_retirement().unwrap(),
    );
    let mut payload = None;
    let measured = allocation_counter::measure(|| {
        payload = Some(columnar::restored::copy_row(
            &record,
            target.input_schema(0),
            0,
            crate::EventTime::from_micros(10),
            lease,
        ));
    });
    assert!(
        measured.bytes_max <= u64::try_from(paid).unwrap(),
        "{measured:?}; resident={paid}"
    );
    assert_eq!(pool.reserved(), paid);
    drop(payload);
    assert_eq!(pool.reserved(), 0);
    println!("standalone fresh resident requested peak {measured:?}; paid={paid}");
}

fn extra_batch_snapshot(snapshot: &OperatorStateSnapshot) -> OperatorStateSnapshot {
    use datafusion::arrow::ipc::writer::StreamWriter;
    let target = operator();
    let record = row(&target, 0).record;
    let record = record.view();
    let mut writer = StreamWriter::try_new(Vec::new(), record.schema_ref()).unwrap();
    writer.write(&record).unwrap();
    writer.write(&record).unwrap();
    writer.finish().unwrap();
    let ipc = writer.into_inner().unwrap();
    let previous = snapshot.segments["left-base"].bytes();
    let old_length =
        usize::try_from(u64::from_le_bytes(previous[40..48].try_into().unwrap())).unwrap();
    let mut bytes = previous[..40].to_vec();
    bytes.extend_from_slice(&u64::try_from(ipc.len()).unwrap().to_le_bytes());
    bytes.extend_from_slice(&ipc);
    bytes.extend_from_slice(&previous[48 + old_length..]);
    let mut snapshot = snapshot.clone();
    snapshot
        .segments
        .insert("left-base".into(), StateSegment::new(bytes));
    snapshot
}

fn check_second_next(snapshot: &OperatorStateSnapshot, job: &StreamJobContext) {
    let snapshot = extra_batch_snapshot(snapshot);
    let mut target = operator();
    let pool = pool(&target);
    let mut construction = construction(&target, &snapshot, job);
    assert!(poll_copy(construction.copy(&target, &snapshot, job)).unwrap());
    let mut decision = None;
    let funded = pool.reserved();
    let measured = allocation_counter::measure(|| {
        decision = Some(
            work(&mut construction)
                .run(construction.schema.metadata.control.stop.as_ref().unwrap())
                .unwrap(),
        );
    });
    assert!(matches!(decision, Some(RestoreBasesDecision::Original(_))));
    assert!(
        measured.bytes_max <= u64::try_from(funded).unwrap(),
        "{measured:?}"
    );
    assert!(measured.bytes_total <= u64::try_from(funded).unwrap());
    let error = target
        .install_bases_decision(decision.unwrap(), &snapshot, job)
        .unwrap_err();
    let original = operator().restore(&snapshot).unwrap_err();
    assert_eq!(error.to_string(), original.to_string());
    assert!(error.to_string().contains("extra record batches"));
    assert_eq!(target.status().left.retained_rows, 0);
    drop(construction);
    assert_eq!(pool.reserved(), 0);
    println!("second-next requested peak {measured:?}; independent funded={funded}");
}

async fn submit(
    operator: &mut StreamJoinOperator,
    construction: &mut RestoreBasesConstruction,
    job: &StreamJobContext,
) -> ObservedTicket<RestoreBasesDecision> {
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

#[test]
fn test_restore_bases_cancellation_during_decode_stops_before_the_next_row() {
    let service = TestService::new(1, 1).unwrap();
    let runtime = tokio::runtime::Builder::new_current_thread()
        .enable_all()
        .build()
        .unwrap();
    let readers = runtime.block_on(check_cancel_during_decode(&service));
    drop(runtime);
    service.shutdown();
    assert_eq!(readers, 1, "cancelled decode must not read later rows");
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
fn test_restore_bases_last_buffer_and_abandoned_worker_keep_real_credit() {
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
            let RestoreBasesDecision::Prepared(prepared) = decision else {
                panic!("prepared")
            };
            let record = &prepared.left[0].record;
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

#[test]
fn test_restore_bases_refusal_keeps_original_schema_work_and_fee_route() {
    let service = TestService::new(1, 1).unwrap();
    let runtime = tokio::runtime::Builder::new_current_thread()
        .enable_all()
        .build()
        .unwrap();
    runtime.block_on(check_resident_refusal(&service));
    runtime.block_on(check_attempt_refusal(&service));
    drop(runtime);
    service.shutdown();
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
    assert_eq!(target.status().left.retained_rows, 3);
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
    let bounds = inventory::required(
        &input::geometry(&snapshot).unwrap(),
        [target.input_schema(0), target.input_schema(1)],
        [
            &target.compiled.left_key_indices,
            &target.compiled.right_key_indices,
        ],
        &target.name,
    )
    .unwrap();
    assert!(bounds.input + bounds.workspace + 3 * bounds.resident[0] + bounds.resident[1] > 65_536);
    let MemoryLimit::Finite(limit) = pool.memory_limit() else {
        panic!("configured finite pool")
    };
    let headroom = original + schema_input + schema_output + 65_536;
    let pressure = MemoryConsumer::new("restore-bases-refusal").register(&pool);
    pressure.try_grow(limit - headroom).unwrap();
    let job = job(service, CancellationToken::new());
    let parses = old_route_probe(&mut target);
    target
        .restore_managed_metadata(&snapshot, &job, None)
        .await
        .unwrap();
    assert_eq!(parses.load(Ordering::Relaxed), 1);
    assert_eq!(target.status().left.retained_rows, 3);
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

#[test]
fn test_restore_bases_final_cancel_and_post_install_home_close_preserve_state() {
    let service = TestService::new(1, 1).unwrap();
    let runtime = tokio::runtime::Builder::new_current_thread()
        .enable_all()
        .build()
        .unwrap();
    runtime.block_on(check_final_cancel(&service));
    runtime.block_on(check_post_install_close(&service));
    drop(runtime);
    service.shutdown();
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
        .install(|decision| target.install_bases_decision(decision, &snapshot, &job))
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

struct RegistrationBlocker {
    entered: tokio::sync::oneshot::Sender<()>,
    release: std::sync::mpsc::Receiver<()>,
}

impl OwnedCpuWork for RegistrationBlocker {
    type Output = ();
    fn control_bytes(&self) -> Result<usize> {
        Ok(size_of::<Self>())
    }
    fn run(self, _: &GatherStop) -> Result<()> {
        self.entered.send(()).unwrap();
        self.release.recv_timeout(Duration::from_secs(10)).unwrap();
        Ok(())
    }
}

fn healthy_close_probe(
    target: &mut StreamJoinOperator,
    job: &StreamJobContext,
) -> Arc<AtomicUsize> {
    let parses = Arc::new(AtomicUsize::new(0));
    let observed_parses = Arc::clone(&parses);
    let owner = job.gather_owner().clone();
    let pool = pool(target);
    target.metadata_test_hook = Some(Arc::new(move |_, parsing| {
        if parsing {
            let (home, generation, attempt) = owner.funding();
            assert_eq!((generation, attempt), (0, 0));
            assert_eq!(
                pool.reserved(),
                home + super::super::super::inventory::caller_controls("match").unwrap()
            );
            observed_parses.fetch_add(1, Ordering::Relaxed);
        }
    }));
    parses
}

async fn check_post_install_close(service: &TestService) {
    use datafusion::execution::memory_pool::{GreedyMemoryPool, MemoryConsumer};
    let snapshot = snapshot(&mut operator());
    let mut target = operator();
    let job = job(service, CancellationToken::new());
    let pool = pool(&target);
    let blocker_job = self::job(service, CancellationToken::new());
    let blocker_pool: Arc<dyn MemoryPool> = Arc::new(GreedyMemoryPool::new(1_048_576));
    let scope = blocker_job
        .gather_owner()
        .client(GatherOperatorId::new("blocker".into()))
        .scope()
        .unwrap();
    let credit = MemoryConsumer::new("restore-bases-registration-blocker").register(&blocker_pool);
    credit.try_grow(4096).unwrap();
    let (entered, started) = tokio::sync::oneshot::channel();
    let (release, wait) = std::sync::mpsc::channel();
    let blocker = scope
        .submit_work(
            RegistrationBlocker {
                entered,
                release: wait,
            },
            credit,
            GatherStop::from_job(&blocker_job),
        )
        .await
        .unwrap();
    started.await.unwrap();
    let parses = healthy_close_probe(&mut target, &job);
    let mut restore = Box::pin(target.restore_managed_metadata(&snapshot, &job, None));
    for _ in 0..128 {
        assert!(futures::poll!(restore.as_mut()).is_pending());
        if service.waiting_requests() == 1 {
            break;
        }
    }
    let (_, generation, attempt) = job.gather_owner().funding();
    assert_eq!(generation, 0);
    assert!(attempt > 0);
    assert_eq!(service.waiting_requests(), 1);
    job.check_cancelled().unwrap();
    job.gather_owner().close_admission();
    release.send(()).unwrap();
    drop(blocker.finish().await.unwrap());
    blocker_job.gather_owner().close_and_drain().await;
    restore.await.unwrap();
    assert_eq!(parses.load(Ordering::Relaxed), 1);
    assert_eq!(target.status().left.retained_rows, 3);
    assert_eq!(target.status().right.retained_rows, 1);
    assert_eq!(target.state.last_checkpoint_epoch, Epoch::new(7));
    job.gather_owner().close_and_drain().await;
    assert_home_credit(&job, &pool);
    assert_home_credit(&blocker_job, &blocker_pool);
    drop(target);
    drop(job);
    drop(scope);
    drop(blocker_job);
    assert_eq!(pool.reserved(), 0);
    assert_eq!(blocker_pool.reserved(), 0);
}
