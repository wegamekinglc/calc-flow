use super::*;
use crate::runtime::streaming::sql_recovery_work::{
    JobSqlRecoveryOwner, OwnedSqlRestoreCompletion,
};
use std::{future::Future, pin::Pin, task::Wake};

#[derive(Clone, Copy)]
enum QueuedStop {
    Context,
    Launch,
    Deadline,
    Unchanged,
}

#[derive(Default)]
struct QueuedStopWake(AtomicUsize);

impl Wake for QueuedStopWake {
    fn wake(self: Arc<Self>) {
        self.0.fetch_add(1, Ordering::SeqCst);
    }

    fn wake_by_ref(self: &Arc<Self>) {
        self.0.fetch_add(1, Ordering::SeqCst);
    }
}

struct ReleaseHeldQueuedGate(Arc<Gate>);

impl Drop for ReleaseHeldQueuedGate {
    fn drop(&mut self) {
        self.0.release();
    }
}

fn active_queued_candidate_is_paid(gate: &Gate) -> bool {
    let state = gate.state.lock().unwrap();
    state.execution.entered
        && state.ledger.paid
        && !state.control.released
        && !state.execution.callback_exited
        && !state.ledger.arrays.is_empty()
        && !state.ledger.backing.is_empty()
        && state
            .ledger
            .arrays
            .iter()
            .all(|weak| weak.upgrade().is_some())
        && state
            .ledger
            .backing
            .iter()
            .all(|weak| weak.upgrade().is_some())
}

fn queued_sources(stop: QueuedStop) -> (&'static str, &'static str) {
    match stop {
        QueuedStop::Context => (
            "queued-context-active-source",
            "queued-context-waiter-source",
        ),
        QueuedStop::Launch => ("queued-launch-active-source", "queued-launch-waiter-source"),
        QueuedStop::Deadline => (
            "queued-deadline-active-source",
            "queued-deadline-waiter-source",
        ),
        QueuedStop::Unchanged => (
            "queued-control-active-source",
            "queued-control-waiter-source",
        ),
    }
}

struct QueuedSignals {
    queued_cancel: CancellationToken,
    launch_cancel: CancellationToken,
    deadline: Option<chrono::DateTime<chrono::Utc>>,
}

struct RefundProbes<'a> {
    active: &'a MemoryReservation,
    queued: &'a MemoryReservation,
}

struct FeeObservation {
    queued: bool,
    active: bool,
}

struct InitialQueueObservation {
    initially_pending: bool,
    unexpired_at_first_poll: bool,
    fees: FeeObservation,
    before_counts: (usize, usize),
}

struct QueueResultObservation {
    queued_cancelled: bool,
    still_pending: bool,
    queued_fee_refunded: bool,
}

struct StopObservation {
    event_occurred: bool,
    active_paid_before_cleanup: bool,
    stop_wakes: usize,
    result: QueueResultObservation,
    after_counts: (usize, usize),
}

struct DrainObservation {
    drain_waited_for_held_active: bool,
    active_paid_during_drain: bool,
    first_stopped: bool,
    held_counts_during_drain: (usize, usize),
    fees: FeeObservation,
    cleanup: Vec<Arc<RuntimeFailure>>,
}

async fn observe_queued_stop(
    stop: QueuedStop,
    mut queued_join: Pin<&mut impl Future<Output = Result<OwnedSqlRestoreCompletion>>>,
    owner: &JobSqlRecoveryOwner,
    active_gate: &Gate,
    refunds: &RefundProbes<'_>,
    signals: &QueuedSignals,
) -> (InitialQueueObservation, StopObservation) {
    let QueuedSignals {
        queued_cancel,
        launch_cancel,
        deadline,
    } = signals;
    let deadline = *deadline;
    let wake = Arc::new(QueuedStopWake::default());
    let waker = std::task::Waker::from(wake.clone());
    let mut task_context = std::task::Context::from_waker(&waker);
    let initial_poll = queued_join.as_mut().poll(&mut task_context);
    let initially_pending = initial_poll.is_pending();
    drop(initial_poll);
    let unexpired_at_first_poll = deadline.is_none_or(|deadline| chrono::Utc::now() < deadline);
    let queued_fee_paid = refunds.queued.try_grow(1 << 30).is_err();
    refunds.queued.free();
    let active_fee_paid = refunds.active.try_grow(1 << 30).is_err();
    refunds.active.free();
    let before_counts = owner.live_counts();
    wake.0.store(0, Ordering::SeqCst);
    match stop {
        QueuedStop::Context => queued_cancel.cancel(),
        QueuedStop::Launch => launch_cancel.cancel(),
        QueuedStop::Deadline => {
            let remaining = deadline
                .unwrap()
                .signed_duration_since(chrono::Utc::now())
                .to_std()
                .unwrap_or_default();
            tokio::time::sleep(remaining + StdDuration::from_millis(20)).await;
        }
        QueuedStop::Unchanged => {}
    }
    tokio::task::yield_now().await;
    let event_occurred = match stop {
        QueuedStop::Context => queued_cancel.is_cancelled(),
        QueuedStop::Launch => launch_cancel.is_cancelled(),
        QueuedStop::Deadline => chrono::Utc::now() >= deadline.unwrap(),
        QueuedStop::Unchanged => true,
    };
    let stop_wakes = wake.0.load(Ordering::SeqCst);
    let post_stop_poll = queued_join.as_mut().poll(&mut task_context);
    let (queued_cancelled, still_pending) = match post_stop_poll {
        Poll::Ready(result) => {
            let cancelled = matches!(&result,
                Err(CalcFlowError::Cancelled { run_id }) if run_id == "705");
            drop(result);
            (cancelled, false)
        }
        Poll::Pending => (false, true),
    };
    let queued_fee_refunded = refunds.queued.try_grow(1 << 30).is_ok();
    refunds.queued.free();
    let after_counts = owner.live_counts();
    let active_paid_before_cleanup =
        active_queued_candidate_is_paid(active_gate) && refunds.active.try_grow(1 << 30).is_err();
    refunds.active.free();
    (
        InitialQueueObservation {
            initially_pending,
            unexpired_at_first_poll,
            fees: FeeObservation {
                queued: queued_fee_paid,
                active: active_fee_paid,
            },
            before_counts,
        },
        StopObservation {
            event_occurred,
            stop_wakes,
            result: QueueResultObservation {
                queued_cancelled,
                still_pending,
                queued_fee_refunded,
            },
            after_counts,
            active_paid_before_cleanup,
        },
    )
}

async fn settle_held_predecessor(
    owner: &JobSqlRecoveryOwner,
    active_gate: &Gate,
    first_join: &mut Option<impl Future<Output = Result<OwnedSqlRestoreCompletion>>>,
    refunds: &RefundProbes<'_>,
) -> DrainObservation {
    owner.close_admission();
    let mut drain = Box::pin(owner.drain());
    let drain_poll = futures::poll!(drain.as_mut());
    let drain_waited_for_held_active = drain_poll.is_pending();
    drop(drain_poll);
    let active_paid_during_drain =
        active_queued_candidate_is_paid(active_gate) && refunds.active.try_grow(1 << 30).is_err();
    refunds.active.free();
    let held_counts_during_drain = owner.live_counts();
    active_gate.release();
    let first_result = first_join.take().unwrap().await;
    let first_stopped = matches!(&first_result,
        Ok(completion) if matches!(&completion.prepared,
            Err(CalcFlowError::Cancelled { run_id }) if run_id == "704"));
    drop(first_result);
    let cleanup = drain.await;
    let first_fee_refunded = refunds.active.try_grow(1 << 30).is_ok();
    refunds.active.free();
    let all_queued_fee_refunded = refunds.queued.try_grow(1 << 30).is_ok();
    refunds.queued.free();
    DrainObservation {
        drain_waited_for_held_active,
        active_paid_during_drain,
        first_stopped,
        held_counts_during_drain,
        fees: FeeObservation {
            active: first_fee_refunded,
            queued: all_queued_fee_refunded,
        },
        cleanup,
    }
}

struct QueuedOracles<'a> {
    owner: &'a JobSqlRecoveryOwner,
    active_gate: &'a Gate,
    expected_bytes: u64,
    installs: [usize; 2],
    queued_starts: usize,
    queued_starts_before_cleanup: usize,
}

fn assert_queued_observations(
    stop: QueuedStop,
    oracle: &QueuedOracles<'_>,
    initial: &InitialQueueObservation,
    stopped: &StopObservation,
    drain: &DrainObservation,
) {
    assert_eq!(oracle.owner.live_counts(), (0, 0));
    assert!(drain.cleanup.is_empty());
    assert!(drain.fees.active && drain.fees.queued);
    assert_released(oracle.active_gate, oracle.expected_bytes);
    assert_eq!(oracle.installs[0], 0);
    assert_eq!(oracle.installs[1], 0);
    assert_eq!(oracle.queued_starts, 0);
    assert!(drain.first_stopped);
    assert!(initial.initially_pending && initial.unexpired_at_first_poll);
    assert!(initial.fees.queued && initial.fees.active);
    assert_eq!(initial.before_counts, (2, 1));
    assert!(stopped.event_occurred);
    assert!(stopped.active_paid_before_cleanup && drain.active_paid_during_drain);
    assert!(drain.drain_waited_for_held_active);
    assert_eq!(drain.held_counts_during_drain, (1, 1));
    assert_eq!(oracle.queued_starts_before_cleanup, 0);
    assert_queue_result(stop, stopped);
}

fn assert_queue_result(stop: QueuedStop, stopped: &StopObservation) {
    if matches!(stop, QueuedStop::Unchanged) {
        assert_eq!(stopped.stop_wakes, 0);
        assert!(
            stopped.result.still_pending
                && !stopped.result.queued_cancelled
                && !stopped.result.queued_fee_refunded
        );
        assert_eq!(stopped.after_counts, (2, 1));
    } else {
        assert!(
            stopped.stop_wakes > 0,
            "queued stop event did not wake its registered waiter"
        );
        assert!(
            stopped.result.queued_cancelled && !stopped.result.still_pending,
            "queued stop waited for held active native work"
        );
        assert!(
            stopped.result.queued_fee_refunded,
            "queued envelope fee was held behind active native work"
        );
        assert_eq!(stopped.after_counts, (1, 1));
    }
}

fn held_predecessor_context() -> StreamJobContext {
    StreamJobContext::new(
        704,
        "held-predecessor-stop-control",
        JsonMap::new(),
        None,
        CancellationToken::new(),
    )
}

fn queued_context(stop: QueuedStop) -> (StreamJobContext, QueuedSignals) {
    let queued_cancel = CancellationToken::new();
    let launch_cancel = CancellationToken::new();
    let deadline = matches!(stop, QueuedStop::Deadline)
        .then(|| chrono::Utc::now() + chrono::Duration::milliseconds(100));
    let queued_context = StreamJobContext::new(
        705,
        "queued-independent-stop-control",
        JsonMap::new(),
        deadline,
        queued_cancel.clone(),
    );
    (
        queued_context,
        QueuedSignals {
            queued_cancel,
            launch_cancel,
            deadline,
        },
    )
}

async fn queued_operators(
    saved: &OperatorStateSnapshot,
    queued_saved: &OperatorStateSnapshot,
) -> (
    (SqlOperator, MemoryReservation),
    (SqlOperator, MemoryReservation),
) {
    let mut first = sql().await;
    let first_refund_probe = recovery_refund_probe(&mut first, saved);
    let mut queued = sql().await;
    let queued_refund_probe = recovery_refund_probe(&mut queued, queued_saved);
    ((first, first_refund_probe), (queued, queued_refund_probe))
}

async fn assert_queued_stop_with_held_predecessor(stop: QueuedStop) {
    let (source, queued_source) = queued_sources(stop);
    let (directory, manifest, saved, expected_bytes) =
        captured_sql_recovery(tempfile::tempdir().unwrap(), source).await;
    let (queued_directory, queued_manifest, queued_saved, _) =
        captured_sql_recovery(tempfile::tempdir().unwrap(), queued_source).await;
    let ((first, first_refund_probe), (queued, queued_refund_probe)) =
        queued_operators(&saved, &queued_saved).await;
    let active_gate = Arc::new(Gate::default());
    let release_on_drop = ReleaseHeldQueuedGate(active_gate.clone());
    let first_registration =
        first.on_prepared_restore_for_test(&expected_normalized_metadata(source), {
            let gate = active_gate.clone();
            move |candidate| {
                let witness = candidate_witness(
                    source,
                    &candidate.metadata,
                    &candidate.records,
                    (candidate.rows, candidate.bytes),
                    candidate.reserved,
                    candidate.backing,
                );
                drop(candidate.records);
                gate.hold(witness);
                Ok(())
            }
        });
    let queued_native_starts = Arc::new(AtomicUsize::new(0));
    let queued_registration =
        queued.on_prepared_restore_for_test(&expected_normalized_metadata(queued_source), {
            let starts = queued_native_starts.clone();
            move |_candidate| {
                starts.fetch_add(1, Ordering::SeqCst);
                Ok(())
            }
        });
    let owner = JobSqlRecoveryOwner::new();
    owner.configure(2).unwrap();
    let first_context = held_predecessor_context();
    let launch = CancellationToken::new();
    let first = submit_sql_restore(&owner, first, saved.clone(), &first_context, 0, launch);
    let mut first_join = Box::pin(first.join());
    let entered = tokio::select! {
        () = active_gate.async_entered.notified() => true,
        result = &mut first_join => { drop(result); false },
        () = tokio::time::sleep(DEADLINE) => false,
    };
    if !entered {
        active_gate.release();
        drop(first_join);
        let cleanup = owner.drain().await;
        assert_eq!(owner.live_counts(), (0, 0));
        assert!(cleanup.is_empty());
        panic!("real active SQL recovery did not reach the held candidate gate");
    }
    let (queued_context, signals) = queued_context(stop);
    let queued = submit_sql_restore(
        &owner,
        queued,
        queued_saved,
        &queued_context,
        1,
        signals.launch_cancel.clone(),
    );
    let refunds = RefundProbes {
        active: &first_refund_probe,
        queued: &queued_refund_probe,
    };
    let mut queued_join = Box::pin(queued.join());
    let (initial, stopped) = observe_queued_stop(
        stop,
        queued_join.as_mut(),
        &owner,
        &active_gate,
        &refunds,
        &signals,
    )
    .await;
    let queued_starts_before_cleanup = queued_native_starts.load(Ordering::SeqCst);
    drop(queued_join);
    let mut first_join = Some(first_join);
    let drain = settle_held_predecessor(&owner, &active_gate, &mut first_join, &refunds).await;
    drop(release_on_drop);
    let installs = [
        first_registration.installs(),
        queued_registration.installs(),
    ];
    let queued_starts = queued_native_starts.load(Ordering::SeqCst);
    let oracle = QueuedOracles {
        owner: &owner,
        active_gate: &active_gate,
        expected_bytes,
        installs,
        queued_starts,
        queued_starts_before_cleanup,
    };
    assert_queued_observations(stop, &oracle, &initial, &stopped, &drain);
    drop((directory, manifest, queued_directory, queued_manifest));
}

#[tokio::test(flavor = "current_thread")]
async fn compact_sql_queued_context_cancel_wakes_and_refunds_before_active_exit() {
    assert_queued_stop_with_held_predecessor(QueuedStop::Context).await;
}

#[tokio::test(flavor = "current_thread")]
async fn compact_sql_queued_launch_cancel_wakes_and_refunds_before_active_exit() {
    assert_queued_stop_with_held_predecessor(QueuedStop::Launch).await;
}

#[tokio::test(flavor = "current_thread")]
async fn compact_sql_queued_deadline_wakes_and_refunds_before_active_exit() {
    assert_queued_stop_with_held_predecessor(QueuedStop::Deadline).await;
}

#[tokio::test(flavor = "current_thread")]
async fn compact_sql_unstopped_queue_retains_fee_and_active_work_until_drain() {
    assert_queued_stop_with_held_predecessor(QueuedStop::Unchanged).await;
}
