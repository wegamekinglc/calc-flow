use super::*;
use crate::{
    CancellationToken, Epoch, JsonMap, StreamOperator,
    operator::join::{JoinStateLimits, JoinTimeBounds, StreamJoinSpec},
    runtime::streaming::gather_work::TestService,
};
use datafusion::arrow::datatypes::{DataType, Field, TimeUnit};
use std::{
    future::Future,
    sync::atomic::{AtomicUsize, Ordering},
    task::{Context, Poll, Wake, Waker},
    time::Duration,
};

fn operator() -> StreamJoinOperator {
    let mut name = String::with_capacity(4_096);
    name.push_str("memo");
    let mut history = std::collections::HashMap::with_capacity(4_096);
    history.insert("old".into(), "value".into());
    history.clear();
    let schema = Arc::new(Schema::new_with_metadata(
        vec![
            Field::new("key", DataType::Int64, false),
            Field::new(
                "time",
                DataType::Timestamp(TimeUnit::Microsecond, Some("UTC".into())),
                false,
            ),
            Field::new(
                name,
                DataType::Timestamp(TimeUnit::Second, Some("".into())),
                true,
            )
            .with_metadata(history.clone()),
        ],
        history,
    ));
    let spec = StreamJoinSpec::inner(
        ["key"],
        ["key"],
        "time",
        "time",
        JoinTimeBounds::new(Duration::ZERO, Duration::ZERO).unwrap(),
        JoinStateLimits::new(10, 100_000, 10).unwrap(),
    )
    .unwrap();
    let mut operator = StreamJoinOperator::new("match", Arc::clone(&schema), schema, spec).unwrap();
    operator.prepare_checkpoint_preload_runtime().unwrap();
    operator
}

fn job(service: &TestService) -> StreamJobContext {
    StreamJobContext::new(
        1,
        "fingerprint",
        JsonMap::new(),
        None,
        CancellationToken::new(),
    )
    .with_gather_owner(service.owner("schema-test".into()))
}

fn snapshot(operator: &mut StreamJoinOperator) -> OperatorStateSnapshot {
    let mut snapshot = operator.checkpoint_v1(Epoch::new(7).unwrap()).unwrap();
    for side in ["left", "right"] {
        snapshot.segments.insert(
            format!("{side}-base"),
            crate::StateSegment::new(
                super::super::super::encode_side(&[], &operator.name, side, &|| Ok(())).unwrap(),
            ),
        );
    }
    snapshot
}

fn construction(
    operator: &StreamJoinOperator,
    snapshot: &OperatorStateSnapshot,
    job: &StreamJobContext,
) -> SchemaConstruction {
    let metadata = operator
        .metadata_construction(snapshot, job)
        .unwrap()
        .unwrap();
    SchemaConstruction::new(operator, metadata, job)
        .unwrap()
        .unwrap()
}

struct CopyWake(AtomicUsize);

impl Wake for CopyWake {
    fn wake(self: Arc<Self>) {
        self.0.fetch_add(1, Ordering::Relaxed);
    }
    fn wake_by_ref(self: &Arc<Self>) {
        self.0.fetch_add(1, Ordering::Relaxed);
    }
}

fn poll_to_completion<F: Future>(future: F, waker: &Waker) -> F::Output {
    let mut future = std::pin::pin!(future);
    let mut context = Context::from_waker(waker);
    loop {
        if let Poll::Ready(output) = future.as_mut().poll(&mut context) {
            return output;
        }
    }
}

#[test]
fn test_schema_constructor_peak_and_partial_drop_are_actually_funded() {
    let service = TestService::new(1, 1).unwrap();
    let runtime = tokio::runtime::Builder::new_current_thread()
        .enable_all()
        .build()
        .unwrap();
    let entered = runtime.enter();
    let mut operator = operator();
    let snapshot = snapshot(&mut operator);
    let job = job(&service);
    let pool = operator
        .runtime
        .runtime
        .as_ref()
        .unwrap()
        .incremental_memory_pool();
    let mut construction = construction(&operator, &snapshot, &job);
    let paid_input = construction.input_credit.as_ref().unwrap().size();
    let DescriptorFunding {
        _credit: credit, ..
    } = construction.funding.as_ref().unwrap().as_ref();
    let paid_output = credit.size();
    let mut schemas = None;
    let wake = Arc::new(CopyWake(AtomicUsize::new(0)));
    let waker = Waker::from(Arc::clone(&wake));
    let measured = allocation_counter::measure(|| {
        poll_to_completion(
            plan::copy(
                &mut construction.plans,
                [operator.input_schema(0), operator.input_schema(1)],
                &job,
            ),
            &waker,
        )
        .unwrap();
        let stop = construction.metadata.control.stop.as_ref().unwrap();
        schemas = Some(OwnedExpectedSchemas {
            left: plan::build(std::mem::take(&mut construction.plans[0]), stop).unwrap(),
            right: plan::build(std::mem::take(&mut construction.plans[1]), stop).unwrap(),
            _funding: Arc::clone(construction.funding.as_ref().unwrap()),
        });
    });
    assert!(
        measured.bytes_max <= u64::try_from(paid_input + paid_output).unwrap(),
        "{measured:?}"
    );
    assert!(wake.0.load(Ordering::Relaxed) > 0);
    println!("constructor allocation {measured:?}; input={paid_input} output={paid_output}");
    let schemas = schemas.unwrap();
    assert_eq!(schemas.schema(0), operator.input_schema(0).as_ref());
    assert_eq!(schemas.schema(1), operator.input_schema(1).as_ref());
    assert!(!Arc::ptr_eq(&schemas.left, operator.input_schema(0)));
    let DataType::Timestamp(_, Some(copied)) = schemas.schema(0).field(1).data_type() else {
        panic!("timezone");
    };
    let DataType::Timestamp(_, Some(original)) = operator.input_schema(0).field(1).data_type()
    else {
        panic!("timezone");
    };
    assert!(!Arc::ptr_eq(copied, original));
    assert_eq!(
        schemas.schema(0).field(2).data_type(),
        &DataType::Timestamp(TimeUnit::Second, Some("".into()))
    );
    drop(construction);
    assert_eq!(pool.reserved(), paid_output);
    drop(schemas);
    assert_eq!(pool.reserved(), 0);
    check_partial_drop(&operator, &snapshot, &job, &pool);
    drop(entered);
    runtime.shutdown_timeout(Duration::from_secs(1));
    service.shutdown();
}

fn check_partial_drop(
    operator: &StreamJoinOperator,
    snapshot: &OperatorStateSnapshot,
    job: &StreamJobContext,
    pool: &Arc<dyn datafusion::execution::memory_pool::MemoryPool>,
) {
    let mut construction = construction(operator, snapshot, job);
    let paid = pool.reserved();
    let waker = Waker::noop();
    let mut context = Context::from_waker(waker);
    {
        let mut copy = std::pin::pin!(plan::copy(
            &mut construction.plans,
            [operator.input_schema(0), operator.input_schema(1)],
            job
        ));
        assert!(copy.as_mut().poll(&mut context).is_pending());
        assert!(copy.as_mut().poll(&mut context).is_pending());
        assert_eq!(pool.reserved(), paid);
    }
    // The borrowed future leaves its owned partial carrier paid until the carrier drops.
    drop(construction);
    assert_eq!(pool.reserved(), 0);
}

#[test]
fn test_schema_output_and_abandoned_attempt_keep_real_cleanup_owned() {
    let service = TestService::new(1, 1).unwrap();
    let runtime = tokio::runtime::Builder::new_current_thread()
        .enable_all()
        .build()
        .unwrap();
    runtime.block_on(check_escaped_output(&service));
    runtime.block_on(check_abandoned_attempt(&service));
    drop(runtime);
    service.shutdown();
}

async fn submit(
    operator: &mut StreamJoinOperator,
    construction: &mut SchemaConstruction,
    job: &StreamJobContext,
) -> ObservedTicket<SchemaDecision> {
    construction.metadata.control.scope = Some(
        job.gather_owner()
            .client(GatherOperatorId::new(Arc::from("match")))
            .scope()
            .unwrap(),
    );
    construction
        .submit(
            &mut operator.compaction_cleanup,
            None,
            operator.schema_test_hook.clone(),
        )
        .await
        .unwrap()
}

async fn check_escaped_output(service: &TestService) {
    let mut operator = operator();
    let snapshot = snapshot(&mut operator);
    let job = job(service);
    let pool = operator
        .runtime
        .runtime
        .as_ref()
        .unwrap()
        .incremental_memory_pool();
    let mut construction = construction(&operator, &snapshot, &job);
    construction.copy(&operator, &snapshot, &job).await.unwrap();
    let ticket = submit(&mut operator, &mut construction, &job).await;
    let output = ticket.finish().await.unwrap();
    let mut escaped = None;
    output
        .install(|decision| {
            escaped = Some(decision.unwrap().schemas.clone());
            Ok(())
        })
        .unwrap();
    let stop = construction.metadata.control.stop.as_ref().unwrap();
    operator
        .compaction_cleanup
        .as_ref()
        .unwrap()
        .wait_job(stop, &job)
        .await
        .unwrap();
    let escaped = escaped.unwrap();
    let schema = Arc::downgrade(&escaped.left);
    let OwnedExpectedSchemas {
        _funding: resident, ..
    } = &escaped;
    let funding = Arc::downgrade(resident);
    let paid = escaped.credit().size();
    drop(construction);
    let (home, generation, attempt) = job.gather_owner().funding();
    assert_eq!(attempt, 0);
    assert!(pool.reserved() >= home + generation + paid);
    {
        let mut drain = std::pin::pin!(job.gather_owner().close_and_drain());
        assert!(std::future::poll_fn(|cx| Poll::Ready(drain.as_mut().poll(cx).is_pending())).await);
        assert!(schema.upgrade().is_some());
        assert!(funding.upgrade().is_some());
        assert!(pool.reserved() >= paid);
        drop(escaped);
        drain.await;
    }
    assert!(schema.upgrade().is_none());
    assert!(funding.upgrade().is_none());
    assert_home_then_final_refund(&job, &pool);
    drop(job);
    assert_eq!(pool.reserved(), 0);
}

async fn check_abandoned_attempt(service: &TestService) {
    let mut operator = operator();
    let snapshot = snapshot(&mut operator);
    let job = job(service);
    let pool = operator
        .runtime
        .runtime
        .as_ref()
        .unwrap()
        .incremental_memory_pool();
    let (entered, started) = tokio::sync::oneshot::channel();
    let (release, wait) = std::sync::mpsc::channel();
    let entered = std::sync::Mutex::new(Some(entered));
    let wait = std::sync::Mutex::new(wait);
    operator.schema_test_hook = Some(Arc::new(move |credit, constructing| {
        if constructing {
            entered
                .lock()
                .unwrap()
                .take()
                .unwrap()
                .send(credit.unwrap().size())
                .unwrap();
            wait.lock()
                .unwrap()
                .recv_timeout(Duration::from_secs(10))
                .unwrap();
        }
    }));
    let mut construction = construction(&operator, &snapshot, &job);
    let funding = Arc::downgrade(construction.funding.as_ref().unwrap());
    construction.copy(&operator, &snapshot, &job).await.unwrap();
    let ticket = submit(&mut operator, &mut construction, &job).await;
    let paid = started.await.unwrap();
    drop(ticket);
    let stop = construction.metadata.control.stop.as_ref().unwrap().clone();
    drop(construction);
    {
        let mut cleanup = std::pin::pin!(
            operator
                .compaction_cleanup
                .as_ref()
                .unwrap()
                .wait_job(&stop, &job)
        );
        assert!(
            std::future::poll_fn(|cx| Poll::Ready(cleanup.as_mut().poll(cx).is_pending())).await
        );
        assert!(funding.upgrade().is_some());
        assert!(pool.reserved() >= paid);
        release.send(()).unwrap();
        cleanup.await.unwrap();
    }
    assert!(funding.upgrade().is_none());
    job.gather_owner().close_and_drain().await;
    assert_home_then_final_refund(&job, &pool);
    drop(job);
    assert_eq!(pool.reserved(), 0);
}

fn assert_home_then_final_refund(
    job: &StreamJobContext,
    pool: &Arc<dyn datafusion::execution::memory_pool::MemoryPool>,
) {
    let (home, generation, attempt) = job.gather_owner().funding();
    assert_eq!((generation, attempt), (0, 0));
    assert!(home > 0);
    assert_eq!(pool.reserved(), home);
}

#[test]
fn test_schema_refusal_preserves_exact_metadata_only_admission() {
    let service = TestService::new(1, 1).unwrap();
    let runtime = tokio::runtime::Builder::new_current_thread()
        .enable_all()
        .build()
        .unwrap();
    runtime.block_on(check_credit_refusal(&service, false));
    runtime.block_on(check_credit_refusal(&service, true));
    runtime.block_on(check_attempt_refusal(&service));
    drop(runtime);
    service.shutdown();
}

fn wide_operator() -> StreamJoinOperator {
    let original = operator();
    let mut fields = Vec::with_capacity(plan::MAX_FIELDS);
    fields.extend(original.input_schema(0).fields()[..2].iter().cloned());
    for index in 2..plan::MAX_FIELDS {
        let name = format!("p{index:02}{}", "x".repeat(plan::MAX_TEXT_BYTES - 3));
        fields.push(Arc::new(Field::new(name, DataType::Utf8, true)));
    }
    let schema = Arc::new(Schema::new(fields));
    let mut output =
        StreamJoinOperator::new("match", Arc::clone(&schema), schema, original.spec.clone())
            .unwrap();
    output.prepare_checkpoint_preload_runtime().unwrap();
    output
}

fn native_parse_probe(operator: &mut StreamJoinOperator) -> (Arc<AtomicUsize>, Arc<AtomicUsize>) {
    let copies = Arc::new(AtomicUsize::new(0));
    let parses = Arc::new(AtomicUsize::new(0));
    let observed_copies = Arc::clone(&copies);
    let observed_parses = Arc::clone(&parses);
    let caller = std::thread::current().id();
    operator.metadata_test_hook = Some(Arc::new(move |credit, parsing| {
        if parsing {
            assert_ne!(
                std::thread::current().id(),
                caller,
                "old metadata optimization must remain native"
            );
            observed_parses.fetch_add(1, Ordering::Relaxed);
        } else {
            assert!(credit.expect("independent original metadata credit").size() > 0);
            observed_copies.fetch_add(1, Ordering::Relaxed);
        }
    }));
    (copies, parses)
}

async fn check_credit_refusal(service: &TestService, output_denied: bool) {
    use datafusion::execution::memory_pool::{MemoryConsumer, MemoryLimit};
    let mut operator = wide_operator();
    let snapshot = snapshot(&mut operator);
    let job = job(service);
    let pool = operator
        .runtime
        .runtime
        .as_ref()
        .unwrap()
        .incremental_memory_pool();
    let original =
        super::super::inventory::required(&snapshot, &operator.spec, &operator.name).unwrap();
    let (input, output) =
        inventory::required([operator.input_schema(0), operator.input_schema(1)]).unwrap();
    let headroom = if output_denied {
        original + input + output - 1
    } else {
        original + 65_536
    };
    assert!(
        input > 65_536,
        "actual descriptor input exceeds this old-worker budget"
    );
    let MemoryLimit::Finite(limit) = pool.memory_limit() else {
        panic!("finite configured pool");
    };
    let pressure = MemoryConsumer::new("schema-refusal-control").register(&pool);
    pressure.try_grow(limit - headroom).unwrap();
    let (copies, parses) = native_parse_probe(&mut operator);
    operator
        .restore_managed_metadata(&snapshot, &job, None)
        .await
        .unwrap();
    assert_eq!(copies.load(Ordering::Relaxed), 1);
    assert_eq!(parses.load(Ordering::Relaxed), 1);
    assert_eq!(operator.state.last_checkpoint_epoch, Epoch::new(7));
    assert!(operator.compaction_cleanup.is_none());
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
    let original = operator();
    let mut fields = original.input_schema(0).fields().to_vec();
    fields.push(Arc::new(Field::new("payload", DataType::LargeUtf8, true)));
    let schema = Arc::new(Schema::new(fields));
    let mut operator =
        StreamJoinOperator::new("match", Arc::clone(&schema), schema, original.spec.clone())
            .unwrap();
    operator.prepare_checkpoint_preload_runtime().unwrap();
    let snapshot = snapshot(&mut operator);
    let job = job(service);
    assert!(
        crate::operator::join::columnar::restored::required(
            operator.input_schema(0),
            super::super::inventory::registration_controls().unwrap(),
        )
        .is_none()
    );
    assert!(
        !operator
            .try_restore_owned_utf8(&snapshot, &job, None)
            .await
            .unwrap()
    );
    let pool = operator
        .runtime
        .runtime
        .as_ref()
        .unwrap()
        .incremental_memory_pool();
    let MemoryLimit::Finite(limit) = pool.memory_limit() else {
        panic!("finite configured pool");
    };
    let probe = AdmissionProbe::install(
        job.gather_owner(),
        AdmissionStage::Attempt,
        Arc::clone(&pool),
        limit,
    );
    let (copies, parses) = native_parse_probe(&mut operator);
    let constructed = Arc::new(AtomicUsize::new(0));
    let observed = Arc::clone(&constructed);
    operator.schema_test_hook = Some(Arc::new(move |_, constructing| {
        if constructing {
            observed.fetch_add(1, Ordering::Relaxed);
        }
    }));
    operator
        .restore_managed_metadata(&snapshot, &job, None)
        .await
        .unwrap();
    let denial = probe.take_event().unwrap();
    assert_eq!(denial.stage, AdmissionStage::Attempt);
    assert_eq!(denial.available + 1, denial.fee);
    assert_eq!(copies.load(Ordering::Relaxed), 2);
    assert_eq!(parses.load(Ordering::Relaxed), 1);
    assert_eq!(constructed.load(Ordering::Relaxed), 0);
    assert_eq!(operator.state.last_checkpoint_epoch, Epoch::new(7));
    assert!(operator.compaction_cleanup.is_none());
    job.gather_owner().close_and_drain().await;
    drop(job);
    assert_eq!(pool.reserved(), 0);
}

#[test]
fn test_schema_comparison_preserves_original_v1_reader_and_caller_schema() {
    let service = TestService::new(1, 1).unwrap();
    let runtime = tokio::runtime::Builder::new_current_thread()
        .enable_all()
        .build()
        .unwrap();
    runtime.block_on(check_empty_timezone_reader_error(&service));
    runtime.block_on(check_v1_reader(&service, false));
    runtime.block_on(check_v1_reader(&service, true));
    drop(runtime);
    service.shutdown();
}

fn declared_operator(schema: SchemaRef, spec: &StreamJoinSpec) -> StreamJoinOperator {
    let mut operator =
        StreamJoinOperator::new("match", Arc::clone(&schema), schema, spec.clone()).unwrap();
    operator.prepare_checkpoint_preload_runtime().unwrap();
    operator
}

fn retained_snapshot(operator: &mut StreamJoinOperator) -> OperatorStateSnapshot {
    use crate::operator::join::{StoredRow, encode_join_key_v1, state_row_charge};
    use datafusion::arrow::{
        array::{Int64Array, TimestampMicrosecondArray, TimestampSecondArray},
        record_batch::RecordBatch,
    };
    let record = RecordBatch::try_new(
        Arc::clone(operator.input_schema(0)),
        vec![
            Arc::new(Int64Array::from(vec![7])),
            Arc::new(TimestampMicrosecondArray::from(vec![10]).with_timezone("UTC")),
            Arc::new(match operator.input_schema(0).field(2).data_type() {
                DataType::Timestamp(_, Some(timezone)) => {
                    TimestampSecondArray::from(vec![20]).with_timezone(Arc::clone(timezone))
                }
                _ => TimestampSecondArray::from(vec![20]),
            }),
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
    let encoded_key = Arc::new(
        encode_join_key_v1(&record, 0, &operator.compiled.left_key_indices)
            .unwrap()
            .into(),
    );
    let row = StoredRow {
        record: record.into(),
        event_time: crate::EventTime::from_micros(10),
        row_id: 0,
        charge,
        encoded_key,
    };
    operator.state.left = vec![row.clone()].into();
    operator.state.right = vec![row].into();
    operator.state.next_left_row_id = 1;
    operator.state.next_right_row_id = 1;
    for metrics in [
        &mut operator.state.metrics.left,
        &mut operator.state.metrics.right,
    ] {
        metrics.retained_rows = 1;
        metrics.retained_bytes = charge;
    }
    let mut snapshot = operator.checkpoint_v1(Epoch::new(7).unwrap()).unwrap();
    for (side, rows) in [
        ("left", &operator.state.left),
        ("right", &operator.state.right),
    ] {
        snapshot.segments.insert(
            format!("{side}-base"),
            crate::StateSegment::new(
                super::super::super::encode_side(rows, &operator.name, side, &|| Ok(())).unwrap(),
            ),
        );
    }
    snapshot
}

async fn check_v1_reader(service: &TestService, unsupported_metadata: bool) {
    let prototype = operator();
    let mut fields = prototype.input_schema(0).fields()[..2].to_vec();
    fields.push(Arc::new(Field::new(
        "memo",
        DataType::Timestamp(TimeUnit::Second, None),
        true,
    )));
    let declared = Schema::new(fields);
    let schema = if unsupported_metadata {
        Arc::new(
            declared
                .clone()
                .with_metadata(std::collections::HashMap::from([(
                    "caller".into(),
                    "must survive".into(),
                )])),
        )
    } else {
        Arc::new(declared)
    };
    let mut source = declared_operator(Arc::clone(&schema), &prototype.spec);
    let snapshot = retained_snapshot(&mut source);
    let before = schema.as_ref().clone();
    let mut actual = declared_operator(Arc::clone(&schema), &prototype.spec);
    let job = job(service);
    let compares = Arc::new(AtomicUsize::new(0));
    let observed = Arc::clone(&compares);
    actual.schema_test_hook = Some(Arc::new(move |credit, constructing| {
        if !constructing {
            assert_eq!(credit.is_none(), unsupported_metadata);
            observed.fetch_add(1, Ordering::Relaxed);
        }
    }));
    actual
        .restore_managed_metadata(&snapshot, &job, None)
        .await
        .unwrap();
    assert_eq!(compares.load(Ordering::Relaxed), 2);
    assert_eq!(schema.as_ref(), &before);
    assert_eq!(actual.status().left.retained_rows, 1);
    assert_eq!(actual.status().right.retained_rows, 1);
    let mut original = declared_operator(Arc::clone(&schema), &prototype.spec);
    original.restore(&snapshot).unwrap();
    let epoch = Epoch::new(8).unwrap();
    let expected = original.checkpoint_v1(epoch).unwrap();
    let checkpoint = actual.checkpoint_v1(epoch).unwrap();
    assert_eq!(checkpoint.inline_metadata, expected.inline_metadata);
    assert_eq!(
        checkpoint.segments.keys().collect::<Vec<_>>(),
        expected.segments.keys().collect::<Vec<_>>()
    );
    for (name, segment) in &checkpoint.segments {
        assert_eq!(segment.bytes(), expected.segments[name].bytes());
    }
    let mut invalid = snapshot.clone();
    invalid
        .segments
        .insert("left-base".into(), crate::StateSegment::new(vec![0]));
    let mut legacy = declared_operator(Arc::clone(&schema), &prototype.spec);
    let old_error = legacy.restore(&invalid).unwrap_err();
    actual.schema_test_hook = None;
    let error = actual
        .restore_managed_metadata(&invalid, &job, None)
        .await
        .unwrap_err();
    assert_eq!(error.to_string(), old_error.to_string());
    assert_eq!(
        actual.status().left.retained_rows,
        1,
        "failed recovery keeps the previous state"
    );
    let pool = actual
        .runtime
        .runtime
        .as_ref()
        .unwrap()
        .incremental_memory_pool();
    drop(actual);
    job.gather_owner().close_and_drain().await;
    drop(job);
    assert_eq!(pool.reserved(), 0);
}

async fn check_empty_timezone_reader_error(service: &TestService) {
    let mut source = operator();
    let snapshot = retained_snapshot(&mut source);
    let mut legacy = operator();
    let original = legacy.restore(&snapshot).unwrap_err();
    let mut managed = operator();
    let job = job(service);
    let error = managed
        .restore_managed_metadata(&snapshot, &job, None)
        .await
        .unwrap_err();
    assert_eq!(error.to_string(), original.to_string());
    assert!(error.to_string().contains("IPC schema is incompatible"));
    let pool = managed
        .runtime
        .runtime
        .as_ref()
        .unwrap()
        .incremental_memory_pool();
    job.gather_owner().close_and_drain().await;
    drop(job);
    assert_eq!(pool.reserved(), 0);
}
