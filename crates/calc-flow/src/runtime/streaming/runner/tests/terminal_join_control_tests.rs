use super::super::{OperatorRestoreState, restored_output_frontier_value};
use super::*;
use crate::{
    EdgeCollector, Epoch, IngressProgress, IngressProgressSnapshot, IngressState,
    OperatorStateSnapshot,
    pipeline::CompiledStreamOperator,
    runtime::streaming::{
        gather_work::{
            TestService,
            admission_probe::{AdmissionProbe, AdmissionStage},
        },
        operator_task::{OperatorProgress, restore_terminal_join},
    },
};
use datafusion::{
    arrow::{
        array::TimestampMicrosecondArray,
        datatypes::{DataType, Field, Schema, TimeUnit},
    },
    execution::memory_pool::{MemoryLimit, MemoryPool},
};

fn schema() -> Arc<Schema> {
    Arc::new(Schema::new(vec![
        Field::new("key", DataType::Int64, false),
        Field::new(
            "ts",
            DataType::Timestamp(TimeUnit::Microsecond, None),
            false,
        ),
    ]))
}

fn join() -> StreamJoinOperator {
    let schema = schema();
    StreamJoinOperator::new(
        "match",
        Arc::clone(&schema),
        schema,
        StreamJoinSpec::inner(
            ["key"],
            ["key"],
            "ts",
            "ts",
            JoinTimeBounds::new(StdDuration::ZERO, StdDuration::from_micros(10)).unwrap(),
            JoinStateLimits::new(100, 100_000, 100).unwrap(),
        )
        .unwrap(),
    )
    .unwrap()
}

fn row(time: i64) -> Batch {
    let record = RecordBatch::try_new(
        schema(),
        vec![
            Arc::new(Int64Array::from(vec![7])),
            Arc::new(TimestampMicrosecondArray::from(vec![time])),
        ],
    )
    .unwrap();
    Batch::table(vec![record], BatchMetadata::default()).unwrap()
}

fn plain_job() -> StreamJobContext {
    StreamJobContext::new(
        41,
        "fingerprint",
        JsonMap::new(),
        None,
        CancellationToken::new(),
    )
}

async fn terminal_snapshot() -> OperatorStateSnapshot {
    let mut source = join();
    source.set_checkpoint_v1_test_producer();
    let job = plain_job();
    let context = StreamOperatorContext::new(&job, "match", None);
    let mut output = EdgeCollector::new(source.output_ports().to_vec());
    source
        .process_data("left", row(95), &context, &mut output)
        .await
        .unwrap();
    source
        .process_data("right", row(100), &context, &mut output)
        .await
        .unwrap();
    let dirty = source.checkpoint(Epoch::INITIAL).unwrap();
    assert_eq!(dirty.segments.len(), 2);
    source.on_end(&context, &mut output).await.unwrap();
    let terminal = source.checkpoint(Epoch::new(2).unwrap()).unwrap();
    assert_eq!(terminal.segments.len(), 4);
    for (name, tag) in [
        ("left-delta-1", 1_u8),
        ("right-delta-1", 1),
        ("left-delta-2", 2),
        ("right-delta-2", 2),
    ] {
        let bytes = terminal.segments[name].bytes();
        assert!(bytes.starts_with(b"CFJDLT1\0"));
        assert_eq!(u64::from_le_bytes(bytes[8..16].try_into().unwrap()), 1);
        assert_eq!(bytes[16], tag);
    }
    terminal
}

fn restore(snapshot: OperatorStateSnapshot) -> OperatorRestoreState {
    OperatorRestoreState {
        snapshot,
        progress: ["left", "right"]
            .into_iter()
            .map(|id| {
                (
                    id.into(),
                    OperatorIngressManifestEntry {
                        state: ManifestIngressState::Ended,
                        watermark: Some(EventTime::from_micros(120)),
                    },
                )
            })
            .collect(),
        output_frontier: Some(EventTime::from_micros(110)),
        next_epoch: Epoch::new(3).unwrap(),
    }
}

fn owned_target(
    service: &TestService,
) -> (
    CompiledStreamOperator,
    Arc<dyn MemoryPool>,
    StreamJobContext,
) {
    let mut target = join();
    let pool = target.checkpoint_preload_test_pool().unwrap();
    let job = plain_job().with_gather_owner(service.owner("terminal-controls".into()));
    (
        CompiledStreamOperator::StreamJoin(Box::new(target)),
        pool,
        job,
    )
}

async fn call(
    operator: &mut CompiledStreamOperator,
    restore: &OperatorRestoreState,
    job: &StreamJobContext,
) -> Result<OperatorProgress> {
    let ingresses = ["left".to_owned(), "right".to_owned()];
    restore_terminal_join(operator, &ingresses, restore, job).await
}

async fn drain(
    operator: CompiledStreamOperator,
    job: StreamJobContext,
    pool: &Arc<dyn MemoryPool>,
) {
    drop(operator);
    job.gather_owner().close_and_drain().await;
    let (home, generation, attempt) = job.gather_owner().funding();
    assert_eq!((generation, attempt), (0, 0));
    assert_eq!(pool.reserved(), home);
    drop(job);
    assert_eq!(pool.reserved(), 0);
}

fn assert_unchanged(snapshot: &OperatorStateSnapshot, original: &OperatorStateSnapshot) {
    assert_eq!(snapshot.inline_metadata, original.inline_metadata);
    assert_eq!(snapshot.segments, original.segments);
}

async fn reject(
    service: &TestService,
    restore: &OperatorRestoreState,
    expected: &str,
    parses: usize,
) {
    let (mut operator, pool, job) = owned_target(service);
    let count = Arc::new(AtomicUsize::new(0));
    let observed = Arc::clone(&count);
    let CompiledStreamOperator::StreamJoin(target) = &mut operator else {
        unreachable!()
    };
    target.set_checkpoint_metadata_test_hook(Arc::new(move |_, parsing| {
        if parsing {
            observed.fetch_add(1, Ordering::SeqCst);
        }
    }));
    let original = restore.snapshot.clone();
    let result = call(&mut operator, restore, &job).await;
    let Err(error) = result else {
        panic!("terminal corruption returned publishable progress")
    };
    assert!(error.to_string().contains(expected), "{error}");
    assert_eq!(count.load(Ordering::SeqCst), parses);
    assert_unchanged(&restore.snapshot, &original);
    drop(error);
    drain(operator, job, &pool).await;
}

fn assert_frontier_contract() {
    assert!(
        restored_output_frontier_value("match", true, None)
            .unwrap_err()
            .to_string()
            .contains("missing its output frontier")
    );
    assert!(
        restored_output_frontier_value("match", true, Some(serde_json::json!("bad")))
            .unwrap_err()
            .to_string()
            .contains("invalid output frontier")
    );
    assert_eq!(
        restored_output_frontier_value("match", true, Some(serde_json::Value::Null)).unwrap(),
        None
    );
    assert_eq!(
        restored_output_frontier_value("match", true, Some(serde_json::json!(110))).unwrap(),
        Some(EventTime::from_micros(110))
    );
}

#[test]
fn test_terminal_join_native_state_ingress_priority_and_frontier_contract() {
    let service = TestService::new(1, 1).unwrap();
    let runtime = tokio::runtime::Builder::new_current_thread()
        .enable_all()
        .build()
        .unwrap();
    runtime.block_on(check_native_and_priority(&service));
    assert_frontier_contract();
    drop(runtime);
    service.shutdown();
}

async fn check_native_and_priority(service: &TestService) {
    let valid = restore(terminal_snapshot().await);
    let mut empty_nonended = restore(valid.snapshot.clone());
    empty_nonended
        .snapshot
        .inline_metadata
        .insert("ended".into(), serde_json::json!(false));
    assert_eq!(empty_nonended.snapshot.segments, valid.snapshot.segments);
    let mut initial = join();
    initial.restore(&empty_nonended.snapshot).unwrap();
    let native = initial.status();
    assert!(!native.left.ended && !native.right.ended);
    assert_eq!(
        (native.left.retained_rows, native.right.retained_rows),
        (0, 0)
    );
    let ended = IngressProgressSnapshot::new(
        ["left", "right"]
            .into_iter()
            .map(|id| {
                (
                    id.into(),
                    IngressProgress::new(IngressState::Ended, Some(EventTime::from_micros(120))),
                )
            })
            .collect(),
    );
    let overlay = initial.status().with_ingress_progress(&ended);
    assert!(overlay.left.ended && overlay.right.ended);
    println!("A phase: successfully decoded native non-ended/overlay mismatch");
    reject(
        service,
        &empty_nonended,
        "non-ended or retained native state",
        1,
    )
    .await;
    println!("A phase: valid terminal native state");
    let (mut operator, pool, job) = owned_target(service);
    let progress = call(&mut operator, &valid, &job).await.unwrap().snapshot();
    assert!(progress.ended);
    let status = progress.stream_join.unwrap();
    assert!(status.left.ended && status.right.ended);
    assert_eq!(
        (status.left.retained_rows, status.right.retained_rows),
        (0, 0)
    );
    assert_eq!(status.emitted_match_rows, 1);
    drain(operator, job, &pool).await;
    let mut malformed = valid;
    malformed
        .snapshot
        .inline_metadata
        .insert("layout_version".into(), serde_json::json!("bad"));
    let expected = join().restore(&malformed.snapshot).unwrap_err().to_string();
    let mut bad_ids = restore(malformed.snapshot.clone());
    let right = bad_ids.progress.remove("right").unwrap();
    bad_ids.progress.insert("alien".into(), right);
    println!("A phase: ingress IDs before malformed metadata");
    reject(service, &bad_ids, "ingress set does not match", 0).await;
    let mut active = restore(malformed.snapshot.clone());
    active.progress.get_mut("left").unwrap().state = ManifestIngressState::Active;
    println!("A phase: non-ended ingress before malformed metadata");
    reject(service, &active, "requires ended ingresses", 0).await;
    println!("A phase: malformed Original diagnostic; no successful metadata decode");
    reject(service, &malformed, &expected, 0).await;
    println!(
        "terminal native/overlay, ingress-before-metadata, Original diagnostic and typed frontier controls passed"
    );
}

fn observe_readers(target: &mut StreamJoinOperator) -> Arc<AtomicUsize> {
    let readers = Arc::new(AtomicUsize::new(0));
    let observed = Arc::clone(&readers);
    target.set_checkpoint_decoded_row_test_hook(Arc::new(move |credit, _, copied, owned| {
        if !copied {
            observed.fetch_add(1, Ordering::SeqCst);
            assert!(credit.is_none());
            assert!(!owned);
        }
    }));
    readers
}

#[test]
fn test_terminal_join_real_attempt_refusal_cancel_and_drain_refund() {
    let service = TestService::new(1, 1).unwrap();
    let runtime = tokio::runtime::Builder::new_current_thread()
        .enable_all()
        .build()
        .unwrap();
    runtime.block_on(check_refusal_and_cancel(&service));
    drop(runtime);
    service.shutdown();
}

async fn check_refusal_and_cancel(service: &TestService) {
    let restore = restore(terminal_snapshot().await);
    let original = restore.snapshot.clone();
    check_refusal(service, &restore).await;
    check_cancel(service, &restore).await;
    assert_unchanged(&restore.snapshot, &original);
    println!(
        "terminal actual Attempt fee-1 fallback and first-native-reader cancellation drained to pool0"
    );
}

async fn check_refusal(service: &TestService, restore: &OperatorRestoreState) {
    let (mut operator, pool, job) = owned_target(service);
    let MemoryLimit::Finite(limit) = pool.memory_limit() else {
        panic!("finite pool")
    };
    let probe = AdmissionProbe::install(
        job.gather_owner(),
        AdmissionStage::Attempt,
        Arc::clone(&pool),
        limit,
    );
    let CompiledStreamOperator::StreamJoin(target) = &mut operator else {
        unreachable!()
    };
    let readers = observe_readers(target);
    let progress = call(&mut operator, restore, &job).await.unwrap().snapshot();
    let event = probe.take_event().unwrap();
    assert_eq!(event.stage, AdmissionStage::Attempt);
    assert!(event.fee > 0);
    assert_eq!(event.available + 1, event.fee);
    assert_eq!(event.task, None);
    assert_eq!(readers.load(Ordering::SeqCst), 2);
    assert!(progress.ended);
    let status = progress.stream_join.unwrap();
    assert_eq!(
        (status.left.retained_rows, status.right.retained_rows),
        (0, 0)
    );
    assert!(status.left.ended && status.right.ended);
    assert_eq!(status.emitted_match_rows, 1);
    drop(probe);
    drain(operator, job, &pool).await;
}

async fn check_cancel(service: &TestService, restore: &OperatorRestoreState) {
    let (mut operator, pool, job) = owned_target(service);
    let readers = Arc::new(AtomicUsize::new(0));
    let observed = Arc::clone(&readers);
    let cancellation = job.cancellation().clone();
    let CompiledStreamOperator::StreamJoin(target) = &mut operator else {
        unreachable!()
    };
    target.set_checkpoint_decoded_row_test_hook(Arc::new(move |credit, _, copied, owned| {
        if !copied && observed.fetch_add(1, Ordering::SeqCst) == 0 {
            assert!(owned);
            assert_eq!(std::thread::current().name(), Some("calc-flow-gather"));
            assert!(credit.unwrap().size() > 0);
            cancellation.cancel();
        }
    }));
    let result = call(&mut operator, restore, &job).await;
    assert!(matches!(result, Err(CalcFlowError::Cancelled { .. })));
    drop(result);
    assert_eq!(readers.load(Ordering::SeqCst), 1);
    let status = operator.stream_join_status().unwrap();
    assert_eq!(
        (status.left.retained_rows, status.right.retained_rows),
        (0, 0)
    );
    assert_eq!(status.emitted_match_rows, 0);
    assert!(!status.left.ended && !status.right.ended);
    drain(operator, job, &pool).await;
}
