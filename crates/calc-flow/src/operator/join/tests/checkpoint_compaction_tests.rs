use super::*;

#[tokio::test]
async fn base_compaction_runs_only_in_checkpoint_preparation() {
    let mut operator =
        StreamJoinOperator::new("match", left_schema(), right_schema(), spec()).unwrap();
    let job_context = job();
    let context = StreamOperatorContext::new(&job_context, "match", None);
    let mut collector = EdgeCollector::new(operator.output_ports().to_vec());
    for epoch in 1..=4 {
        operator
            .process_data("left", left_batch(vec![epoch]), &context, &mut collector)
            .await
            .unwrap();
        operator
            .checkpoint(Epoch::new(u64::try_from(epoch).unwrap()).unwrap())
            .unwrap();
    }
    assert!(operator.state.deltas.needs_compaction);
    operator
        .process_data("left", left_batch(vec![5]), &context, &mut collector)
        .await
        .unwrap();
    assert!(
        operator.state.deltas.needs_compaction,
        "data handler must not compact the base"
    );
    let progress = progress_context(
        &job_context,
        (IngressState::Active, None),
        (IngressState::Active, None),
    );
    operator
        .on_ingress_progress("right", &progress)
        .await
        .unwrap();
    assert!(
        operator.state.deltas.needs_compaction,
        "progress handler must not compact the base"
    );
    operator.prepare_checkpoint_async(&context).await.unwrap();
    assert!(!operator.state.deltas.needs_compaction);
    let snapshot = operator.checkpoint(Epoch::new(5).unwrap()).unwrap();
    assert!(snapshot.segments.contains_key("left-base"));
    assert_eq!(snapshot.segments.len(), 2);
    let mut restored =
        StreamJoinOperator::new("match", left_schema(), right_schema(), spec()).unwrap();
    restored.restore(&snapshot).unwrap();
    assert_eq!(restored.status(), operator.status());
}

#[tokio::test]
async fn checkpoint_without_an_intervening_handler_keeps_v1_restore_and_continuation() {
    let mut operator =
        StreamJoinOperator::new("match", left_schema(), right_schema(), spec()).unwrap();
    let job_context = job();
    let context = StreamOperatorContext::new(&job_context, "match", None);
    let mut collector = EdgeCollector::new(operator.output_ports().to_vec());
    let mut previous = OperatorStateSnapshot::default();
    for epoch in 1..=4 {
        operator
            .process_data("left", left_batch(vec![epoch]), &context, &mut collector)
            .await
            .unwrap();
        previous = operator
            .checkpoint(Epoch::new(u64::try_from(epoch).unwrap()).unwrap())
            .unwrap();
    }
    let direct = operator.checkpoint(Epoch::new(5).unwrap()).unwrap();
    assert_eq!(
        direct.segments, previous.segments,
        "direct capture carries prepared V1 deltas"
    );
    let before = operator.status();
    operator.prepare_checkpoint_async(&context).await.unwrap();
    let prepared = operator.checkpoint(Epoch::new(6).unwrap()).unwrap();
    assert_eq!(prepared.inline_metadata["layout_version"], 1);
    assert_eq!(prepared.segments.len(), 2);
    for segment in prepared.segments.values() {
        assert_eq!(&segment.bytes()[..8], JOIN_STATE_MAGIC);
    }
    let mut restored =
        StreamJoinOperator::new("match", left_schema(), right_schema(), spec()).unwrap();
    restored.restore(&prepared).unwrap();
    assert_eq!(restored.status(), before);
    restored
        .process_data("right", right_batch(vec![0]), &context, &mut collector)
        .await
        .unwrap();
    assert_eq!(restored.status().emitted_match_rows, 4);
    assert_eq!(
        collector.drain("output")[0]
            .as_data()
            .unwrap()
            .metadata()
            .sequence(),
        0
    );
}

#[tokio::test]
async fn cancelled_checkpoint_preparation_leaves_dirty_changes_and_previous_capture_usable() {
    let mut operator =
        StreamJoinOperator::new("match", left_schema(), right_schema(), spec()).unwrap();
    let job_context = job();
    let context = StreamOperatorContext::new(&job_context, "match", None);
    let mut collector = EdgeCollector::new(operator.output_ports().to_vec());
    let mut previous = OperatorStateSnapshot::default();
    for epoch in 1..=4 {
        operator
            .process_data("left", left_batch(vec![epoch]), &context, &mut collector)
            .await
            .unwrap();
        previous = operator
            .checkpoint(Epoch::new(u64::try_from(epoch).unwrap()).unwrap())
            .unwrap();
    }
    operator
        .process_data("left", left_batch(vec![5]), &context, &mut collector)
        .await
        .unwrap();
    let cancelled = job();
    cancelled.cancellation().cancel();
    assert!(
        operator
            .prepare_checkpoint_async(&StreamOperatorContext::new(&cancelled, "match", None))
            .await
            .is_err()
    );
    let mut restored =
        StreamJoinOperator::new("match", left_schema(), right_schema(), spec()).unwrap();
    restored.restore(&previous).unwrap();
    assert_eq!(restored.status().left.retained_rows, 4);
    operator.prepare_checkpoint_async(&context).await.unwrap();
    let snapshot = operator.checkpoint(Epoch::new(5).unwrap()).unwrap();
    restored.restore(&snapshot).unwrap();
    assert_eq!(restored.status().left.retained_rows, 5);
}

async fn checkpoint_worker_fixture(
    job: &StreamJobContext,
) -> (StreamJoinOperator, OperatorStateSnapshot, EdgeCollector) {
    let mut operator =
        StreamJoinOperator::new("match", left_schema(), right_schema(), spec()).unwrap();
    let context = StreamOperatorContext::new(job, "match", None);
    let mut collector = EdgeCollector::new(operator.output_ports().to_vec());
    let mut previous = OperatorStateSnapshot::default();
    for epoch in 1..=4 {
        operator
            .process_data("left", left_batch(vec![epoch]), &context, &mut collector)
            .await
            .unwrap();
        previous = operator
            .checkpoint(Epoch::new(u64::try_from(epoch).unwrap()).unwrap())
            .unwrap();
    }
    (operator, previous, collector)
}

struct CheckpointGateHarness {
    gate: checkpoint_compaction::TestGate,
    started: tokio::sync::oneshot::Receiver<(usize, std::thread::ThreadId)>,
    release: std::sync::mpsc::Sender<()>,
}

fn checkpoint_gate(failed: bool) -> CheckpointGateHarness {
    let (entered, started) = tokio::sync::oneshot::channel();
    let (release, wait) = std::sync::mpsc::channel();
    CheckpointGateHarness {
        gate: checkpoint_compaction::TestGate {
            entered,
            wait,
            failed,
        },
        started,
        release,
    }
}

fn isolated_checkpoint_test(
    check: impl FnOnce(&crate::runtime::streaming::gather_work::TestService, &tokio::runtime::Runtime),
) {
    let service = crate::runtime::streaming::gather_work::TestService::new(1, 1).unwrap();
    let runtime = tokio::runtime::Builder::new_current_thread()
        .enable_all()
        .build()
        .unwrap();
    check(&service, &runtime);
    drop(runtime);
    service.shutdown();
}

async fn abandoned_checkpoint_replacement(
    service: &crate::runtime::streaming::gather_work::TestService,
    restore: bool,
) {
    let job = job().with_gather_owner(service.owner("checkpoint-replacement".into()));
    let context = StreamOperatorContext::new(&job, "match", None);
    let (mut operator, previous, mut collector) = checkpoint_worker_fixture(&job).await;
    let pool = operator
        .runtime
        .runtime()
        .unwrap()
        .incremental_memory_pool();
    let weak = Arc::downgrade(operator.state.left[0].record.column(0));
    let original = operator.state.left.as_ptr() as usize;
    let CheckpointGateHarness {
        gate,
        started,
        release,
    } = checkpoint_gate(false);
    operator.checkpoint_gate = Some(std::sync::Mutex::new(gate));
    let mut prepare = Box::pin(operator.prepare_checkpoint_async(&context));
    assert!(
        futures::poll!(prepare.as_mut()).is_pending(),
        "preparation must use the owned worker"
    );
    let (worker_pointer, worker_thread) = started.await.unwrap();
    assert_eq!(
        worker_pointer, original,
        "preparation must share the retained vector without cloning it"
    );
    assert_ne!(worker_thread, std::thread::current().id());
    drop(prepare);
    if restore {
        let mut replacement =
            StreamJoinOperator::new("match", left_schema(), right_schema(), spec()).unwrap();
        replacement
            .process_data("left", left_batch(vec![90]), &context, &mut collector)
            .await
            .unwrap();
        let snapshot = replacement.checkpoint(Epoch::INITIAL).unwrap();
        operator.restore(&snapshot).unwrap();
    } else {
        operator.reset().unwrap();
    }
    assert!(
        weak.upgrade().is_some(),
        "reset/restore must retain the old worker input until drain"
    );
    let (_, _, attempt) = job.gather_owner().funding();
    assert!(attempt >= 4 * 124, "input credit must survive replacement");
    let rows_before = u64::from(restore);
    let batch = left_batch(if restore { vec![] } else { vec![90] });
    let mut mutation =
        Box::pin(operator.process_data("left", batch.clone(), &context, &mut collector));
    let waited = futures::poll!(mutation.as_mut()).is_pending();
    drop(mutation);
    assert_eq!(operator.status().left.retained_rows, rows_before);
    release.send(()).unwrap();
    operator
        .process_data("left", batch, &context, &mut collector)
        .await
        .unwrap();
    assert!(job.gather_owner().close_and_drain().await.is_empty());
    assert!(weak.upgrade().is_none());
    drop(context);
    drop(job);
    assert_eq!(pool.reserved(), 0);
    assert!(
        waited,
        "replacement mutation must await the old input retirement"
    );
    assert_eq!(
        operator.status().left.retained_rows,
        1,
        "old worker must not install stale state"
    );
    let epoch = if restore { 2 } else { 1 };
    let snapshot = operator.checkpoint(Epoch::new(epoch).unwrap()).unwrap();
    let mut restored =
        StreamJoinOperator::new("match", left_schema(), right_schema(), spec()).unwrap();
    restored.restore(&snapshot).unwrap();
    assert_eq!(restored.status().left.retained_rows, 1);
    restored.restore(&previous).unwrap();
    assert_eq!(restored.status().left.retained_rows, 4);
}

#[test]
fn dropped_checkpoint_worker_survives_reset_without_stale_installation() {
    isolated_checkpoint_test(|service, runtime| {
        runtime.block_on(abandoned_checkpoint_replacement(service, false));
    });
}

#[test]
fn dropped_checkpoint_worker_survives_restore_without_stale_installation() {
    isolated_checkpoint_test(|service, runtime| {
        runtime.block_on(abandoned_checkpoint_replacement(service, true));
    });
}

async fn abandoned_checkpoint_mutation(
    service: &crate::runtime::streaming::gather_work::TestService,
) {
    let job = job().with_gather_owner(service.owner("checkpoint-mutation".into()));
    let context = StreamOperatorContext::new(&job, "match", None);
    let (mut operator, previous, mut collector) = checkpoint_worker_fixture(&job).await;
    let pool = operator
        .runtime
        .runtime()
        .unwrap()
        .incremental_memory_pool();
    let CheckpointGateHarness {
        gate,
        started,
        release,
    } = checkpoint_gate(false);
    operator.checkpoint_gate = Some(std::sync::Mutex::new(gate));
    let mut prepare = Box::pin(operator.prepare_checkpoint_async(&context));
    assert!(futures::poll!(prepare.as_mut()).is_pending());
    started.await.unwrap();
    drop(prepare);
    let direct = operator.checkpoint(Epoch::new(5).unwrap()).unwrap();
    assert_eq!(direct.segments, previous.segments);
    let mut directly_restored =
        StreamJoinOperator::new("match", left_schema(), right_schema(), spec()).unwrap();
    directly_restored.restore(&direct).unwrap();
    assert_eq!(directly_restored.status(), operator.status());
    let mut mutation =
        Box::pin(operator.process_data("left", left_batch(vec![5]), &context, &mut collector));
    assert!(
        futures::poll!(mutation.as_mut()).is_pending(),
        "mutation must await the old immutable input owner"
    );
    drop(mutation);
    assert_eq!(operator.status().left.retained_rows, 4);
    let (_, _, attempt) = job.gather_owner().funding();
    assert!(
        attempt >= 4 * 124,
        "input credit must survive observer drops"
    );
    release.send(()).unwrap();
    operator
        .process_data("left", left_batch(vec![5]), &context, &mut collector)
        .await
        .unwrap();
    assert_eq!(operator.status().left.retained_rows, 5);
    assert!(
        operator.state.deltas.needs_compaction,
        "handler must not replace the abandoned preparation"
    );
    let snapshot = operator.checkpoint(Epoch::new(6).unwrap()).unwrap();
    let mut restored =
        StreamJoinOperator::new("match", left_schema(), right_schema(), spec()).unwrap();
    restored.restore(&snapshot).unwrap();
    assert_eq!(restored.status().left.retained_rows, 5);
    assert!(job.gather_owner().close_and_drain().await.is_empty());
    drop(context);
    drop(job);
    assert_eq!(pool.reserved(), 0);
}

#[test]
fn dropped_checkpoint_and_mutation_futures_preserve_input_ownership() {
    isolated_checkpoint_test(|service, runtime| {
        runtime.block_on(abandoned_checkpoint_mutation(service));
    });
}

async fn stopped_compaction_wait(
    service: &crate::runtime::streaming::gather_work::TestService,
    deadline: bool,
) {
    let job = job().with_gather_owner(service.owner("checkpoint-stop".into()));
    let context = StreamOperatorContext::new(&job, "match", None);
    let (mut operator, previous, mut collector) = checkpoint_worker_fixture(&job).await;
    let CheckpointGateHarness {
        gate,
        started,
        release,
    } = checkpoint_gate(false);
    operator.checkpoint_gate = Some(std::sync::Mutex::new(gate));
    let mut prepare = Box::pin(operator.prepare_checkpoint_async(&context));
    assert!(futures::poll!(prepare.as_mut()).is_pending());
    started.await.unwrap();
    drop(prepare);
    let before = operator.status();
    let cancellation = CancellationToken::new();
    let waiting_job = StreamJobContext::new(
        3,
        "checkpoint-wait",
        JsonMap::new(),
        deadline.then(|| chrono::Utc::now() + chrono::Duration::milliseconds(40)),
        cancellation.clone(),
    );
    let waiting = StreamOperatorContext::new(&waiting_job, "match", None);
    let mut mutation =
        Box::pin(operator.process_data("left", left_batch(vec![5]), &waiting, &mut collector));
    assert!(futures::poll!(mutation.as_mut()).is_pending());
    if !deadline {
        cancellation.cancel();
    }
    let result = tokio::time::timeout(Duration::from_secs(2), mutation.as_mut()).await;
    drop(mutation);
    let after = operator.status();
    let (_, _, paid_attempt) = job.gather_owner().funding();
    release.send(()).unwrap();
    assert!(job.gather_owner().close_and_drain().await.is_empty());
    assert!(
        matches!(result, Ok(Err(CalcFlowError::Cancelled { .. }))),
        "retirement wait must observe stop/deadline: {result:?}"
    );
    assert!(paid_attempt >= 4 * 124);
    assert_eq!(after, before);
    let direct = operator.checkpoint(Epoch::new(5).unwrap()).unwrap();
    assert_eq!(direct.segments, previous.segments);
}

#[test]
fn compaction_retirement_wait_observes_cancellation_and_deadline() {
    isolated_checkpoint_test(|service, runtime| {
        runtime.block_on(async {
            stopped_compaction_wait(service, false).await;
            stopped_compaction_wait(service, true).await;
        });
    });
}

#[tokio::test]
async fn test_progress_without_retirement_preserves_cancel_and_deadline_checks() {
    let healthy = job();
    let (mut operator, _, _) = checkpoint_worker_fixture(&healthy).await;
    let before = operator.status();
    for expired in [false, true] {
        let cancellation = CancellationToken::new();
        let stopped = StreamJobContext::new(
            3,
            "progress-stop",
            JsonMap::new(),
            expired.then(|| chrono::Utc::now() - chrono::Duration::seconds(1)),
            cancellation.clone(),
        );
        if !expired {
            cancellation.cancel();
        }
        let progress = progress_context(
            &stopped,
            (IngressState::Active, None),
            (IngressState::Active, Some(60_000_005)),
        );
        assert!(operator.compaction_release.is_none());
        assert!(operator.compaction_cleanup.is_none());
        assert!(matches!(
            operator.on_ingress_progress("right", &progress).await,
            Err(CalcFlowError::Cancelled { .. })
        ));
        assert_eq!(operator.status(), before);
    }
}

struct ProgressRefundFixture {
    operator: StreamJoinOperator,
    job: StreamJobContext,
    refund: std::sync::mpsc::Sender<()>,
    pool: Arc<dyn datafusion::execution::memory_pool::MemoryPool>,
}

async fn progress_refund_fixture(
    service: &crate::runtime::streaming::gather_work::TestService,
) -> ProgressRefundFixture {
    let job = job().with_gather_owner(service.owner("progress-credit-retirement".into()));
    let context = StreamOperatorContext::new(&job, "match", None);
    let (mut operator, _, _) = checkpoint_worker_fixture(&job).await;
    let pool = operator
        .runtime
        .runtime()
        .unwrap()
        .incremental_memory_pool();
    let CheckpointGateHarness {
        gate,
        started,
        release,
    } = checkpoint_gate(false);
    let (retired, owners_gone) = tokio::sync::oneshot::channel();
    let (refund, wait) = std::sync::mpsc::channel();
    operator.checkpoint_gate = Some(std::sync::Mutex::new(gate));
    operator.checkpoint_retirement_gate = Some(std::sync::Mutex::new(
        checkpoint_compaction::TestRetirementGate {
            entered: retired,
            wait,
        },
    ));
    let mut prepare = Box::pin(operator.prepare_checkpoint_async(&context));
    assert!(futures::poll!(prepare.as_mut()).is_pending());
    started.await.unwrap();
    drop(prepare);
    release.send(()).unwrap();
    owners_gone.await.unwrap();
    operator.compaction_release.take().unwrap().await.unwrap();
    assert_eq!(Arc::strong_count(&operator.state.left.0), 1);
    assert!(operator.compaction_release.is_none());
    assert!(operator.compaction_cleanup.is_some());
    ProgressRefundFixture {
        operator,
        job,
        refund,
        pool,
    }
}

async fn progress_cleanup_wait(
    service: &crate::runtime::streaming::gather_work::TestService,
    stop: u8,
) {
    let ProgressRefundFixture {
        mut operator,
        job,
        refund,
        pool,
    } = progress_refund_fixture(service).await;
    let before = operator.status();
    let cancellation = CancellationToken::new();
    let waiting = StreamJobContext::new(
        3,
        "progress-refund-wait",
        JsonMap::new(),
        (stop == 2).then(|| chrono::Utc::now() + chrono::Duration::milliseconds(40)),
        cancellation.clone(),
    );
    let progress = progress_context(
        &waiting,
        (IngressState::Active, None),
        (IngressState::Active, Some(60_000_005)),
    );
    let (home, generation, attempt) = job.gather_owner().funding();
    assert!(attempt >= 4 * 124);
    assert_eq!(pool.reserved(), home + generation + attempt);
    let mut mutation = Box::pin(operator.on_ingress_progress("right", &progress));
    assert!(futures::poll!(mutation.as_mut()).is_pending());
    if stop == 0 {
        drop(mutation);
        assert_eq!(operator.status(), before);
        assert!(operator.compaction_cleanup.is_some());
        refund.send(()).unwrap();
        assert!(job.gather_owner().close_and_drain().await.is_empty());
        operator
            .on_ingress_progress("right", &progress)
            .await
            .unwrap();
        assert_eq!(operator.status().left.retained_rows, 0);
        assert!(operator.compaction_cleanup.is_none());
    } else {
        if stop == 1 {
            cancellation.cancel();
        }
        let result = tokio::time::timeout(Duration::from_secs(2), mutation.as_mut()).await;
        drop(mutation);
        assert!(matches!(result, Ok(Err(CalcFlowError::Cancelled { .. }))));
        assert_eq!(operator.status(), before);
        assert!(operator.compaction_cleanup.is_some());
        assert_eq!(job.gather_owner().funding().2, attempt);
        assert_eq!(pool.reserved(), home + generation + attempt);
        refund.send(()).unwrap();
        assert!(job.gather_owner().close_and_drain().await.is_empty());
    }
    drop(job);
    assert_eq!(pool.reserved(), 0);
}

#[test]
fn test_progress_cleanup_only_waits_for_actual_refund_and_observes_stop() {
    isolated_checkpoint_test(|service, runtime| {
        runtime.block_on(async {
            for stop in 0..=2 {
                progress_cleanup_wait(service, stop).await;
            }
        });
    });
}

#[test]
fn managed_close_drains_abandoned_compaction_after_operator_drop() {
    isolated_checkpoint_test(|service, runtime| {
        runtime.block_on(async {
            let job = job().with_gather_owner(service.owner("checkpoint-close".into()));
            let context = StreamOperatorContext::new(&job, "match", None);
            let (mut operator, _, _) = checkpoint_worker_fixture(&job).await;
            let pool = operator
                .runtime
                .runtime()
                .unwrap()
                .incremental_memory_pool();
            let weak = Arc::downgrade(operator.state.left[0].record.column(0));
            let CheckpointGateHarness {
                gate,
                started,
                release,
            } = checkpoint_gate(false);
            operator.checkpoint_gate = Some(std::sync::Mutex::new(gate));
            let mut prepare = Box::pin(operator.prepare_checkpoint_async(&context));
            assert!(futures::poll!(prepare.as_mut()).is_pending());
            started.await.unwrap();
            drop(prepare);
            drop(operator);
            let mut drain = Box::pin(job.gather_owner().close_and_drain());
            assert!(futures::poll!(drain.as_mut()).is_pending());
            drop(drain);
            let mut drain = Box::pin(job.gather_owner().close_and_drain());
            assert!(futures::poll!(drain.as_mut()).is_pending());
            assert!(weak.upgrade().is_some());
            let (_, _, attempt) = job.gather_owner().funding();
            assert!(attempt >= 4 * 124);
            assert!(pool.reserved() >= attempt);
            release.send(()).unwrap();
            assert!(drain.await.is_empty());
            assert!(weak.upgrade().is_none());
            drop(context);
            drop(job);
            assert_eq!(pool.reserved(), 0);
        });
    });
}

#[tokio::test]
async fn unpolled_compaction_preparation_preserves_dirty_state_and_funding() {
    let job = job();
    let context = StreamOperatorContext::new(&job, "match", None);
    let (mut operator, previous, mut collector) = checkpoint_worker_fixture(&job).await;
    operator
        .process_data("left", left_batch(vec![5]), &context, &mut collector)
        .await
        .unwrap();
    let before = operator.status();
    let pool = operator
        .runtime
        .runtime()
        .unwrap()
        .incremental_memory_pool();
    let reserved = pool.reserved();
    drop(Box::pin(operator.prepare_checkpoint_async(&context)));
    assert_eq!(pool.reserved(), reserved);
    assert_eq!(operator.status(), before);
    let snapshot = operator.checkpoint(Epoch::new(5).unwrap()).unwrap();
    let mut restored =
        StreamJoinOperator::new("match", left_schema(), right_schema(), spec()).unwrap();
    restored.restore(&previous).unwrap();
    assert_eq!(restored.status().left.retained_rows, 4);
    restored.restore(&snapshot).unwrap();
    assert_eq!(restored.status(), before);
}

async fn failed_checkpoint_preparation(
    service: &crate::runtime::streaming::gather_work::TestService,
) {
    let job = job().with_gather_owner(service.owner("checkpoint-failure".into()));
    let context = StreamOperatorContext::new(&job, "match", None);
    let (mut operator, previous, mut collector) = checkpoint_worker_fixture(&job).await;
    operator
        .process_data("left", left_batch(vec![5]), &context, &mut collector)
        .await
        .unwrap();
    let before = operator.status();
    let pool = operator
        .runtime
        .runtime()
        .unwrap()
        .incremental_memory_pool();
    let CheckpointGateHarness {
        gate,
        started,
        release,
    } = checkpoint_gate(true);
    operator.checkpoint_gate = Some(std::sync::Mutex::new(gate));
    let mut prepare = Box::pin(operator.prepare_checkpoint_async(&context));
    assert!(futures::poll!(prepare.as_mut()).is_pending());
    started.await.unwrap();
    release.send(()).unwrap();
    assert!(prepare.await.is_err());
    assert_eq!(operator.status(), before);
    let (home, generation, attempt) = job.gather_owner().funding();
    assert_eq!(attempt, 0);
    assert_eq!(pool.reserved(), home + generation);
    let snapshot = operator.checkpoint(Epoch::new(5).unwrap()).unwrap();
    let mut restored =
        StreamJoinOperator::new("match", left_schema(), right_schema(), spec()).unwrap();
    restored.restore(&previous).unwrap();
    assert_eq!(restored.status().left.retained_rows, 4);
    restored.restore(&snapshot).unwrap();
    assert_eq!(restored.status(), before);
    operator.prepare_checkpoint_async(&context).await.unwrap();
    assert!(!operator.state.deltas.needs_compaction);
    assert!(job.gather_owner().close_and_drain().await.is_empty());
    drop(context);
    drop(job);
    assert_eq!(pool.reserved(), 0);
}

#[test]
fn released_snapshot_waits_for_actual_attempt_credit_before_retry() {
    isolated_checkpoint_test(|service, runtime| {
        runtime.block_on(async {
            for ticket_observed in [false, true] {
                checkpoint_credit_retirement(service, ticket_observed).await;
            }
        });
    });
}

async fn checkpoint_credit_retirement(
    service: &crate::runtime::streaming::gather_work::TestService,
    ticket_observed: bool,
) {
    let job = job().with_gather_owner(service.owner("checkpoint-credit-retirement".into()));
    let context = StreamOperatorContext::new(&job, "match", None);
    let (mut operator, _, _) = checkpoint_worker_fixture(&job).await;
    let pool = operator
        .runtime
        .runtime()
        .unwrap()
        .incremental_memory_pool();
    let pressure = operator
        .runtime
        .runtime()
        .unwrap()
        .incremental_reservation("held-credit");
    let CheckpointGateHarness {
        gate,
        started,
        release,
    } = checkpoint_gate(false);
    let (retired, owners_gone) = tokio::sync::oneshot::channel();
    let (refund, wait) = std::sync::mpsc::channel();
    operator.checkpoint_gate = Some(std::sync::Mutex::new(gate));
    operator.checkpoint_retirement_gate = Some(std::sync::Mutex::new(
        checkpoint_compaction::TestRetirementGate {
            entered: retired,
            wait,
        },
    ));
    let mut prepare = Box::pin(operator.prepare_checkpoint_async(&context));
    assert!(futures::poll!(prepare.as_mut()).is_pending());
    started.await.unwrap();
    if ticket_observed {
        assert!(futures::poll!(prepare.as_mut()).is_pending());
    }
    drop(prepare);
    release.send(()).unwrap();
    owners_gone.await.unwrap();
    assert_eq!(Arc::strong_count(&operator.state.left.0), 1);
    let (_, _, attempt) = job.gather_owner().funding();
    assert!(attempt >= 4 * 124);
    pressure.try_grow((1 << 30) - pool.reserved()).unwrap();
    let mut retry = Box::pin(operator.prepare_checkpoint_async(&context));
    let first_poll = futures::poll!(retry.as_mut());
    let waited = first_poll.is_pending();
    drop(retry);
    assert_eq!(pool.reserved(), 1 << 30);
    refund.send(()).unwrap();
    job.gather_owner().close_and_drain().await;
    assert!(
        waited,
        "retry reused credit before native retirement: {first_poll:?}"
    );
    drop(pressure);
    drop(context);
    drop(job);
    assert_eq!(pool.reserved(), 0);
}

#[test]
fn failed_checkpoint_worker_preserves_dirty_changes_and_prior_capture() {
    isolated_checkpoint_test(|service, runtime| {
        runtime.block_on(failed_checkpoint_preparation(service));
    });
}

#[test]
fn checkpoint_observer_is_published_before_native_capacity_wait() {
    isolated_checkpoint_test(|service, runtime| {
        runtime.block_on(async {
            let first_job =
                job().with_gather_owner(service.owner("checkpoint-capacity-first".into()));
            let first_context = StreamOperatorContext::new(&first_job, "match", None);
            let (mut first, _, _) = checkpoint_worker_fixture(&first_job).await;
            let second_job =
                job().with_gather_owner(service.owner("checkpoint-capacity-second".into()));
            let second_context = StreamOperatorContext::new(&second_job, "match", None);
            let (mut second, _, mut collector) = checkpoint_worker_fixture(&second_job).await;
            let pool = second.runtime.runtime().unwrap().incremental_memory_pool();
            let pressure = second
                .runtime
                .runtime()
                .unwrap()
                .incremental_reservation("queued-retirement-pressure");
            let weak = Arc::downgrade(second.state.left[0].record.column(0));
            let (retired, owners_gone) = tokio::sync::oneshot::channel();
            let (refund, wait) = std::sync::mpsc::channel();
            second.checkpoint_retirement_gate = Some(std::sync::Mutex::new(
                checkpoint_compaction::TestRetirementGate {
                    entered: retired,
                    wait,
                },
            ));
            let CheckpointGateHarness {
                gate,
                started,
                release,
            } = checkpoint_gate(false);
            first.checkpoint_gate = Some(std::sync::Mutex::new(gate));
            let mut active = Box::pin(first.prepare_checkpoint_async(&first_context));
            assert!(futures::poll!(active.as_mut()).is_pending());
            started.await.unwrap();
            let mut queued = Box::pin(second.prepare_checkpoint_async(&second_context));
            assert!(futures::poll!(queued.as_mut()).is_pending());
            assert_eq!(
                second_job.gather_owner().funding().2,
                4 * (124 + 2 * size_of::<&StoredRow>())
                    + checkpoint_compaction::expected_attempt_control_bytes("match"),
                "the actual attempt must fund snapshot, scratch and opt-in cleanup control",
            );
            assert!(weak.upgrade().is_some());
            let owner = second_job.gather_owner().clone();
            let retirement = std::thread::spawn(move || owner.close_admission());
            owners_gone.await.unwrap();
            drop(queued);
            let tracked = second.compaction_cleanup.is_some();
            pressure.try_grow((1 << 30) - pool.reserved()).unwrap();
            let mut retry = Box::pin(second.prepare_checkpoint_async(&second_context));
            let first_poll = futures::poll!(retry.as_mut());
            let waited = first_poll.is_pending();
            drop(retry);
            assert_eq!(pool.reserved(), 1 << 30);
            refund.send(()).unwrap();
            retirement.join().unwrap();
            drop(pressure);
            drop(active);
            release.send(()).unwrap();
            assert!(first_job.gather_owner().close_and_drain().await.is_empty());
            second
                .process_data("left", left_batch(vec![5]), &second_context, &mut collector)
                .await
                .unwrap();
            assert!(second_job.gather_owner().close_and_drain().await.is_empty());
            assert!(
                waited,
                "pre-ticket retry reused credit before native retirement: {first_poll:?}"
            );
            assert!(
                tracked,
                "paid preparation awaiting native capacity must publish its cleanup observer"
            );
            assert_eq!(second.status().left.retained_rows, 5);
        });
    });
}

async fn checkpoint_credit_failure(spare: usize) {
    let job = job();
    let context = StreamOperatorContext::new(&job, "match", None);
    let (mut operator, previous, mut collector) = checkpoint_worker_fixture(&job).await;
    operator
        .process_data("left", left_batch(vec![5]), &context, &mut collector)
        .await
        .unwrap();
    let before = operator.status();
    let runtime = operator.runtime.runtime().unwrap();
    let pool = runtime.incremental_memory_pool();
    let pressure = runtime.incremental_reservation("checkpoint-pressure");
    pressure
        .try_grow((1 << 30) - pool.reserved() - spare)
        .unwrap();
    assert!(matches!(
        operator.prepare_checkpoint_async(&context).await,
        Err(CalcFlowError::DataFusion { .. })
    ));
    assert_eq!(operator.status(), before);
    operator
        .process_data("left", left_batch(vec![]), &context, &mut collector)
        .await
        .unwrap();
    let snapshot = operator.checkpoint(Epoch::new(5).unwrap()).unwrap();
    let mut restored =
        StreamJoinOperator::new("match", left_schema(), right_schema(), spec()).unwrap();
    restored.restore(&previous).unwrap();
    assert_eq!(restored.status().left.retained_rows, 4);
    restored.restore(&snapshot).unwrap();
    assert_eq!(restored.status(), before);
    drop(pressure);
    operator.prepare_checkpoint_async(&context).await.unwrap();
    assert!(!operator.state.deltas.needs_compaction);
    assert!(job.gather_owner().close_and_drain().await.is_empty());
    drop(context);
    drop(job);
    assert_eq!(pool.reserved(), 0);
}

#[tokio::test]
async fn checkpoint_worker_admission_failure_preserves_v1_dirty_changes() {
    checkpoint_credit_failure(0).await;
    checkpoint_credit_failure(5 * (124 + 2 * size_of::<&StoredRow>())).await;
}

#[tokio::test]
async fn checkpoint_encoding_preserves_v1_logical_limits_without_an_encoded_segment_cap() {
    let mut tight = spec();
    tight.limits = JoinStateLimits::new(100, 4 * 124, 1_000).unwrap();
    let mut operator =
        StreamJoinOperator::new("match", left_schema(), right_schema(), tight.clone()).unwrap();
    let job = job();
    let context = StreamOperatorContext::new(&job, "match", None);
    let mut collector = EdgeCollector::new(operator.output_ports().to_vec());
    for epoch in 1..=4 {
        operator
            .process_data("left", left_batch(vec![epoch]), &context, &mut collector)
            .await
            .unwrap();
        operator
            .checkpoint(Epoch::new(u64::try_from(epoch).unwrap()).unwrap())
            .unwrap();
    }
    assert_eq!(operator.status().left.retained_bytes, 496);
    operator.prepare_checkpoint_async(&context).await.unwrap();
    let snapshot = operator.checkpoint(Epoch::new(5).unwrap()).unwrap();
    assert_eq!(snapshot.segments.len(), 2);
    assert!(snapshot.segments["left-base"].bytes().len() > 496);
    let mut restored =
        StreamJoinOperator::new("match", left_schema(), right_schema(), tight).unwrap();
    restored.restore(&snapshot).unwrap();
    assert_eq!(restored.status(), operator.status());
}
