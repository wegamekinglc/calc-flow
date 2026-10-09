use super::*;
use crate::{
    CancellationToken, Epoch, JsonMap, StreamOperator,
    operator::join::{JoinStateLimits, JoinTimeBounds},
    runtime::streaming::gather_work::TestService,
};
use datafusion::arrow::datatypes::{DataType, Field, Schema, TimeUnit};
use std::{
    future::Future,
    sync::atomic::{AtomicUsize, Ordering},
    task::{Context, Poll, Wake, Waker},
    time::Duration,
};

fn operator_fixture() -> StreamJoinOperator {
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
        JoinStateLimits::new(10, 100_000, 10).unwrap(),
    )
    .unwrap();
    StreamJoinOperator::new("match", Arc::clone(&schema), schema, spec).unwrap()
}

fn job(service: &TestService) -> StreamJobContext {
    StreamJobContext::new(
        1,
        "fingerprint",
        JsonMap::new(),
        None,
        CancellationToken::new(),
    )
    .with_gather_owner(service.owner("metadata-test".into()))
}

fn initialized_operator() -> StreamJoinOperator {
    let mut operator = operator_fixture();
    operator.prepare_checkpoint_preload_runtime().unwrap();
    operator
}

fn valid_snapshot(operator: &mut StreamJoinOperator) -> OperatorStateSnapshot {
    let mut snapshot = operator.checkpoint_v1(Epoch::new(7).unwrap()).unwrap();
    for side in ["left", "right"] {
        snapshot.segments.insert(
            format!("{side}-base"),
            crate::StateSegment::new(
                super::super::encode_side(&[], &operator.name, side, &|| Ok(())).unwrap(),
            ),
        );
    }
    snapshot
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

fn measure_copy<F: Future>(future: F) -> (F::Output, allocation_counter::AllocationInfo) {
    let mut future = std::pin::pin!(future);
    let wake = Arc::new(CopyWake(AtomicUsize::new(0)));
    let waker = Waker::from(Arc::clone(&wake));
    let mut context = Context::from_waker(&waker);
    let mut output = None;
    let measured = allocation_counter::measure(|| {
        loop {
            if let Poll::Ready(value) = future.as_mut().poll(&mut context) {
                output = Some(value);
                break;
            }
        }
    });
    assert!(wake.0.load(Ordering::Relaxed) > 0);
    (output.unwrap(), measured)
}

#[test]
fn test_metadata_copy_and_original_parser_peak_are_actually_funded() {
    let service = TestService::new(1, 1).unwrap();
    let runtime = tokio::runtime::Builder::new_current_thread()
        .enable_all()
        .build()
        .unwrap();
    let entered = runtime.enter();
    let mut operator = initialized_operator();
    let mut snapshot = valid_snapshot(&mut operator);
    let fields = snapshot
        .inline_metadata
        .get_mut("spec")
        .unwrap()
        .as_object_mut()
        .unwrap();
    fields.remove("left_prefix");
    fields.remove("right_prefix");
    let original = snapshot.inline_metadata.clone();
    let job = job(&service);
    let pool = operator
        .runtime
        .runtime
        .as_ref()
        .unwrap()
        .incremental_memory_pool();
    let mut construction = operator
        .metadata_construction(&snapshot, &job)
        .unwrap()
        .unwrap();
    let SubmissionControl {
        _credit: caller_credit,
        ..
    } = &construction.control;
    let paid = construction.credit.as_ref().unwrap().size() + caller_credit.size();
    assert_eq!(pool.reserved(), paid);
    let (copied, copy) = measure_copy(construction.copy(&operator, &snapshot, &job));
    copied.unwrap();
    assert!(copy.bytes_max <= u64::try_from(paid).unwrap());
    let work = MetadataWork {
        snapshot: std::mem::take(&mut construction.snapshot),
        expected: construction.expected.take().unwrap(),
        name: std::mem::take(&mut construction.name),
        hook: None,
    };
    let mut result = None;
    let parsed = allocation_counter::measure(|| {
        result = Some(
            work.run(construction.control.stop.as_ref().unwrap())
                .unwrap(),
        );
    });
    eprintln!("metadata copy={copy:?}, original parser={parsed:?}, paid={paid}");
    assert!(u64::try_from(copy.bytes_current).unwrap() + parsed.bytes_max <= paid as u64);
    assert_eq!(copy.bytes_current + parsed.bytes_current, 0);
    assert!(result.unwrap().is_some());
    assert_eq!(snapshot.inline_metadata, original);
    drop(construction);
    assert_eq!(pool.reserved(), 0);
    drop(entered);
    drop(runtime);
    service.shutdown();
}

#[test]
fn test_partial_metadata_copy_drop_keeps_credit_and_managed_drain_owned() {
    let service = TestService::new(1, 1).unwrap();
    let runtime = tokio::runtime::Builder::new_current_thread()
        .enable_all()
        .build()
        .unwrap();
    runtime.block_on(async {
        let mut operator = initialized_operator();
        let snapshot = valid_snapshot(&mut operator);
        let job = job(&service);
        let pool = operator
            .runtime
            .runtime
            .as_ref()
            .unwrap()
            .incremental_memory_pool();
        let mut construction = operator
            .metadata_construction(&snapshot, &job)
            .unwrap()
            .unwrap();
        let paid = pool.reserved();
        let mut copy = Box::pin(construction.copy(&operator, &snapshot, &job));
        assert!(std::future::poll_fn(|cx| Poll::Ready(copy.as_mut().poll(cx).is_pending())).await);
        assert!(std::future::poll_fn(|cx| Poll::Ready(copy.as_mut().poll(cx).is_pending())).await);
        assert_eq!(pool.reserved(), paid);
        drop(copy);
        assert!(!construction.name.is_empty());
        let mut drain = Box::pin(job.gather_owner().close_and_drain());
        assert!(std::future::poll_fn(|cx| Poll::Ready(drain.as_mut().poll(cx).is_pending())).await);
        assert_eq!(pool.reserved(), paid);
        drop(construction);
        assert!(drain.await.is_empty());
        assert_eq!(pool.reserved(), 0);
    });
    drop(runtime);
    service.shutdown();
}

#[test]
fn test_metadata_worker_cancel_does_not_install_or_refund_live_input() {
    let service = TestService::new(1, 1).unwrap();
    let runtime = tokio::runtime::Builder::new_current_thread()
        .enable_all()
        .build()
        .unwrap();
    let mut operator = initialized_operator();
    let snapshot = valid_snapshot(&mut operator);
    operator.state.last_checkpoint_epoch = None;
    let job = job(&service);
    let pool = operator
        .runtime
        .runtime
        .as_ref()
        .unwrap()
        .incremental_memory_pool();
    let (entered, mut started) = tokio::sync::oneshot::channel();
    let (release, wait) = std::sync::mpsc::channel();
    let entered = std::sync::Mutex::new(Some(entered));
    let wait = std::sync::Mutex::new(wait);
    let actual_pool = Arc::clone(&pool);
    operator.metadata_test_hook = Some(Arc::new(move |credit, parsing| {
        if parsing {
            entered
                .lock()
                .unwrap()
                .take()
                .unwrap()
                .send(actual_pool.reserved())
                .unwrap();
            wait.lock()
                .unwrap()
                .recv_timeout(Duration::from_secs(10))
                .unwrap();
        } else {
            assert!(credit.unwrap().size() > 0);
        }
    }));
    runtime.block_on(async {
        let mut restore = Box::pin(operator.restore_managed_metadata(&snapshot, &job, None));
        let paid = tokio::select! {
            paid = &mut started => paid.unwrap(),
            result = &mut restore => panic!("restore returned before worker gate: {result:?}"),
        };
        let (home, generation, attempt) = job.gather_owner().funding();
        assert!(attempt > 0);
        assert!(paid >= home + generation + attempt);
        job.cancellation().cancel();
        drop(restore);
        assert!(operator.compaction_cleanup.is_some());
        assert_eq!(operator.state.last_checkpoint_epoch, None);
        assert!(pool.reserved() >= home + generation + attempt);
        drop(operator);
        let mut drain = Box::pin(job.gather_owner().close_and_drain());
        assert!(std::future::poll_fn(|cx| Poll::Ready(drain.as_mut().poll(cx).is_pending())).await);
        release.send(()).unwrap();
        assert!(drain.await.is_empty());
        let (home, generation, attempt) = job.gather_owner().funding();
        assert_eq!(attempt, 0);
        assert_eq!(pool.reserved(), home + generation);
    });
    drop(runtime);
    drop(job);
    service.shutdown();
    assert_eq!(pool.reserved(), 0);
}

#[test]
fn test_uncertified_nested_metrics_and_original_errors_use_legacy_exactly() {
    let service = TestService::new(1, 1).unwrap();
    let runtime = tokio::runtime::Builder::new_current_thread()
        .enable_all()
        .build()
        .unwrap();
    runtime.block_on(async {
        let mut source = operator_fixture();
        let mut snapshot = valid_snapshot(&mut source);
        snapshot.inline_metadata.get_mut("metrics").unwrap()["left"]["legacy_extra"] =
            serde_json::json!({"arbitrary": ["still accepted"]});
        let original = snapshot.inline_metadata.clone();
        let mut expected = operator_fixture();
        expected.restore(&snapshot).unwrap();
        let mut actual = initialized_operator();
        let copies = Arc::new(AtomicUsize::new(0));
        let observed = Arc::clone(&copies);
        actual.metadata_test_hook = Some(Arc::new(move |credit, parsing| {
            assert!(credit.is_none(), "uncertified input must not be copied");
            if parsing {
                observed.fetch_add(1, Ordering::Relaxed);
            }
        }));
        actual
            .restore_managed_metadata(&snapshot, &job(&service), None)
            .await
            .unwrap();
        assert_eq!(actual.status(), expected.status());
        assert_eq!(copies.load(Ordering::Relaxed), 1);
        assert_eq!(snapshot.inline_metadata, original);
        snapshot
            .inline_metadata
            .insert("layout_version".into(), serde_json::json!(99));
        let expected_error = expected.restore(&snapshot).unwrap_err().to_string();
        let actual_error = actual
            .restore_managed_metadata(&snapshot, &job(&service), None)
            .await
            .unwrap_err()
            .to_string();
        assert_eq!(actual_error, expected_error);
    });
    drop(runtime);
    service.shutdown();
}

#[test]
fn test_metadata_refusal_refunds_before_original_restore() {
    use datafusion::execution::memory_pool::{MemoryConsumer, MemoryLimit};
    let service = TestService::new(1, 1).unwrap();
    let runtime = tokio::runtime::Builder::new_current_thread()
        .enable_all()
        .build()
        .unwrap();
    let mut operator = initialized_operator();
    let snapshot = valid_snapshot(&mut operator);
    let original = snapshot.inline_metadata.clone();
    let pool = operator
        .runtime
        .runtime
        .as_ref()
        .unwrap()
        .incremental_memory_pool();
    let MemoryLimit::Finite(limit) = pool.memory_limit() else {
        panic!("configured finite pool");
    };
    let pressure = MemoryConsumer::new("metadata-budget-control").register(&pool);
    pressure.try_grow(limit).unwrap();
    let job = job(&service);
    runtime
        .block_on(operator.restore_managed_metadata(&snapshot, &job, None))
        .unwrap();
    assert_eq!(pool.reserved(), pressure.size());
    assert_eq!(operator.state.last_checkpoint_epoch, Epoch::new(7));
    let required = inventory::required(&snapshot, &operator.spec, &operator.name).unwrap();
    pressure.shrink(required);
    let copied = Arc::new(AtomicUsize::new(0));
    let observed = Arc::clone(&copied);
    operator.metadata_test_hook = Some(Arc::new(move |credit, parsing| {
        if !parsing && credit.is_some() {
            observed.fetch_add(1, Ordering::Relaxed);
        }
    }));
    runtime
        .block_on(operator.restore_managed_metadata(&snapshot, &job, None))
        .unwrap();
    assert_eq!(copied.load(Ordering::Relaxed), 1);
    assert!(operator.compaction_cleanup.is_none());
    assert_eq!(job.gather_owner().funding(), (0, 0, 0));
    assert_eq!(pool.reserved(), pressure.size());
    assert_eq!(snapshot.inline_metadata, original);
    drop(pressure);
    assert_eq!(pool.reserved(), 0);
    drop(runtime);
    service.shutdown();
}

#[test]
fn test_healthy_closed_metadata_admission_uses_original_restore() {
    let service = TestService::new(1, 1).unwrap();
    let runtime = tokio::runtime::Builder::new_current_thread()
        .enable_all()
        .build()
        .unwrap();
    let mut operator = initialized_operator();
    let snapshot = valid_snapshot(&mut operator);
    let job = job(&service);
    let owner = job.gather_owner().clone();
    operator.metadata_test_hook = Some(Arc::new(move |credit, parsing| {
        if !parsing && credit.is_some() {
            owner.close_admission();
        }
    }));
    runtime
        .block_on(operator.restore_managed_metadata(&snapshot, &job, None))
        .unwrap();
    assert_eq!(operator.state.last_checkpoint_epoch, Epoch::new(7));
    assert_eq!(
        operator
            .runtime
            .runtime
            .as_ref()
            .unwrap()
            .incremental_memory_pool()
            .reserved(),
        0
    );
    assert!(
        runtime
            .block_on(job.gather_owner().close_and_drain())
            .is_empty()
    );
    drop(runtime);
    service.shutdown();
}

struct RegistrationBlocker {
    entered: tokio::sync::oneshot::Sender<()>,
    release: std::sync::mpsc::Receiver<()>,
}

impl OwnedCpuWork for RegistrationBlocker {
    type Output = ();

    fn run(self, _: &GatherStop) -> Result<()> {
        self.entered.send(()).unwrap();
        self.release.recv_timeout(Duration::from_secs(10)).unwrap();
        Ok(())
    }
}

fn post_install_refund_probe(
    operator: &mut StreamJoinOperator,
    job: &StreamJobContext,
) -> (Arc<AtomicUsize>, Arc<AtomicUsize>) {
    let copies = Arc::new(AtomicUsize::new(0));
    let parses = Arc::new(AtomicUsize::new(0));
    let observed_copies = Arc::clone(&copies);
    let observed_parses = Arc::clone(&parses);
    let pool = operator
        .runtime
        .runtime
        .as_ref()
        .unwrap()
        .incremental_memory_pool();
    let owner = job.gather_owner().clone();
    operator.metadata_test_hook = Some(Arc::new(move |credit, parsing| {
        if parsing {
            let (home, generation, attempt) = owner.funding();
            assert_eq!((generation, attempt), (0, 0));
            assert_eq!(
                pool.reserved(),
                home + inventory::caller_controls("match").unwrap()
            );
            observed_parses.fetch_add(1, Ordering::Relaxed);
        } else if let Some(credit) = credit {
            assert!(credit.size() > 0);
            observed_copies.fetch_add(1, Ordering::Relaxed);
        }
    }));
    (copies, parses)
}

#[test]
fn test_healthy_home_close_after_metadata_attempt_install_refunds_then_restores() {
    use datafusion::execution::memory_pool::{GreedyMemoryPool, MemoryConsumer, MemoryPool};
    let service = TestService::new(1, 1).unwrap();
    let runtime = tokio::runtime::Builder::new_current_thread()
        .enable_all()
        .build()
        .unwrap();
    let blocker_pool: Arc<dyn MemoryPool> = Arc::new(GreedyMemoryPool::new(1_048_576));
    let mut operator = initialized_operator();
    let snapshot = valid_snapshot(&mut operator);
    operator.state.last_checkpoint_epoch = None;
    let original = snapshot.inline_metadata.clone();
    let blocker_job = job(&service);
    let job = job(&service);
    let pool = operator
        .runtime
        .runtime
        .as_ref()
        .unwrap()
        .incremental_memory_pool();
    let (copies, parses) = post_install_refund_probe(&mut operator, &job);
    let result = runtime.block_on(async {
        let blocker_scope = blocker_job
            .gather_owner()
            .client(GatherOperatorId::new("blocker".into()))
            .scope()
            .unwrap();
        let credit = MemoryConsumer::new("metadata-registration-blocker").register(&blocker_pool);
        credit.try_grow(4_096).unwrap();
        let (entered, started) = tokio::sync::oneshot::channel();
        let (release, wait) = std::sync::mpsc::channel();
        let blocker = blocker_scope
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
        let mut restore = Box::pin(operator.restore_managed_metadata(&snapshot, &job, None));
        for _ in 0..24 {
            assert!(futures::poll!(restore.as_mut()).is_pending());
            if service.waiting_requests() == 1 {
                break;
            }
        }
        let (home, generation, attempt) = job.gather_owner().funding();
        assert!(
            attempt > 0,
            "metadata attempt must be installed before closing its home"
        );
        assert_eq!(
            generation, 0,
            "metadata must still await native registration"
        );
        assert_eq!(service.waiting_requests(), 1);
        assert!(pool.reserved() > home + attempt);
        job.check_cancelled().unwrap();
        job.gather_owner().close_admission();
        release.send(()).unwrap();
        drop(blocker.finish().await.unwrap());
        assert!(
            blocker_job
                .gather_owner()
                .close_and_drain()
                .await
                .is_empty()
        );
        let result = restore.await;
        job.check_cancelled().unwrap();
        assert!(job.gather_owner().close_and_drain().await.is_empty());
        result
    });
    eprintln!(
        "post-install healthy close restore={result:?}, copy={}, parse={}",
        copies.load(Ordering::Relaxed),
        parses.load(Ordering::Relaxed)
    );
    result.unwrap();
    assert!(operator.compaction_cleanup.is_none());
    assert_eq!(operator.state.last_checkpoint_epoch, Epoch::new(7));
    assert_eq!(copies.load(Ordering::Relaxed), 1);
    assert_eq!(parses.load(Ordering::Relaxed), 1);
    assert_eq!(snapshot.inline_metadata, original);
    let (home, generation, attempt) = blocker_job.gather_owner().funding();
    assert_eq!(attempt, 0);
    assert_eq!(blocker_pool.reserved(), home + generation);
    drop(operator);
    drop(job);
    drop(blocker_job);
    drop(runtime);
    service.shutdown();
    assert_eq!(pool.reserved(), 0);
    assert_eq!(blocker_pool.reserved(), 0);
}

#[test]
fn test_last_stop_check_prevents_restore_assignment() {
    let service = TestService::new(1, 1).unwrap();
    let mut operator = initialized_operator();
    let snapshot = valid_snapshot(&mut operator);
    operator.state.last_checkpoint_epoch = None;
    let metadata = operator.parse_restore_metadata(&snapshot).unwrap();
    let job = job(&service);
    let error = operator
        .install_restored_metadata(&snapshot, metadata, &|| {
            job.cancellation().cancel();
            job.check_cancelled()
        })
        .unwrap_err();
    assert!(matches!(error, crate::CalcFlowError::Cancelled { .. }));
    assert_eq!(operator.state.last_checkpoint_epoch, None);
    assert_eq!(operator.status().left.retained_rows, 0);
    service.shutdown();
}

#[test]
fn test_metadata_output_refund_keeps_late_caller_controls_funded() {
    let service = TestService::new(1, 1).unwrap();
    let runtime = tokio::runtime::Builder::new_current_thread()
        .enable_all()
        .build()
        .unwrap();
    let mut operator = initialized_operator();
    let snapshot = valid_snapshot(&mut operator);
    let job = job(&service);
    let pool = operator
        .runtime
        .runtime
        .as_ref()
        .unwrap()
        .incremental_memory_pool();
    runtime.block_on(async {
        let mut construction = operator
            .metadata_construction(&snapshot, &job)
            .unwrap()
            .unwrap();
        construction.copy(&operator, &snapshot, &job).await.unwrap();
        construction.control.scope = Some(
            job.gather_owner()
                .client(GatherOperatorId::new(Arc::from(operator.name.as_str())))
                .scope()
                .unwrap(),
        );
        let SubmissionControl {
            _credit: caller_credit,
            ..
        } = &construction.control;
        let caller_paid = caller_credit.size();
        assert_eq!(
            caller_paid,
            inventory::caller_controls(&operator.name).unwrap()
        );
        let ticket = construction
            .submit(&mut operator.compaction_cleanup, None)
            .await
            .unwrap();
        let output = ticket.finish().await.unwrap();
        output
            .install(|decision| {
                let Some(metadata) = decision else {
                    panic!("certified metadata");
                };
                operator.install_restored_metadata(&snapshot, metadata, &|| job.check_cancelled())
            })
            .unwrap();
        operator
            .wait_metadata_cleanup(construction.control.stop.as_ref().unwrap(), &job)
            .await
            .unwrap();
        let (home, generation, attempt) = job.gather_owner().funding();
        assert_eq!(attempt, 0);
        assert_eq!(pool.reserved(), caller_paid + home + generation);
        assert!(construction.control.scope.is_some());
        assert!(construction.control.stop.is_some());
        drop(construction);
        assert_eq!(pool.reserved(), home + generation);
        assert!(job.gather_owner().close_and_drain().await.is_empty());
    });
    drop(operator);
    drop(job);
    drop(runtime);
    service.shutdown();
    assert_eq!(pool.reserved(), 0);
}
