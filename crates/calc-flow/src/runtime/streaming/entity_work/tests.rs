use super::*;
use crate::operator::rolling::entity_parallel_test_pair;
use std::sync::{
    Condvar,
    atomic::{AtomicU64, Ordering},
};

struct ReleaseGate(Arc<Gate>);

impl Drop for ReleaseGate {
    fn drop(&mut self) {
        self.0.release();
    }
}

#[derive(Default)]
struct Gate {
    entered: AtomicU64,
    released: std::sync::Mutex<bool>,
    changed: Condvar,
}

impl Gate {
    fn block(&self) {
        self.entered.fetch_add(1, Ordering::SeqCst);
        let mut released = self.released.lock().unwrap();
        while !*released {
            released = self.changed.wait(released).unwrap();
        }
    }

    async fn wait_for(&self, count: u64) {
        tokio::time::timeout(std::time::Duration::from_secs(5), async {
            while self.entered.load(Ordering::SeqCst) < count {
                tokio::task::yield_now().await;
            }
        })
        .await
        .unwrap();
    }

    fn release(&self) {
        *self.released.lock().unwrap() = true;
        self.changed.notify_all();
    }
}

impl Drop for Gate {
    fn drop(&mut self) {
        self.release();
    }
}

fn owner() -> JobEntityWorkOwner {
    JobEntityWorkOwner::new(7, Arc::new(AtomicU64::new(0)))
}

fn reserved(scope: &CallbackWorkScope, scratch: ScratchPlan) -> PairLaunchGuard {
    let ReservePair::Reserved(reserved) = scope.context_client().try_reserve(scratch) else {
        panic!("idle live owner must reserve one pair");
    };
    reserved
}

#[tokio::test]
async fn real_pair_retains_one_slot_through_join_until_merge_releases_scratch() {
    let owner = owner();
    let task = owner.client(TaskId::new(3), "operator:rolling".into());
    let scope = task.callback_scope();
    let (seed, requests, scratch) = entity_parallel_test_pair();
    let mut ticket = reserved(&scope, scratch).start(requests, None);
    let joined = ticket.join().await;
    let next_scope = task.callback_scope();
    assert!(matches!(
        next_scope.context_client().try_reserve(scratch),
        ReservePair::Serial(SerialReason::Busy)
    ));
    assert_eq!(owner.active_bytes(), scratch.accounted_bytes().unwrap());
    let update = joined.merge(seed).unwrap();
    drop(update);
    assert_eq!(owner.active_bytes(), 0);
    assert!(matches!(
        next_scope.context_client().try_reserve(scratch),
        ReservePair::Reserved(_)
    ));
    drop(next_scope);
    drop(scope);
    owner.close_admission();
    assert!(owner.drain().await.is_empty());
}

#[tokio::test]
async fn lazy_routing_drops_with_joined_payload_while_seed_remains_owned() {
    let owner = owner();
    let task = owner.client(TaskId::new(3), "operator:rolling".into());
    let scope = task.callback_scope();
    let (seed, requests, scratch) = entity_parallel_test_pair();
    let input = requests[0].input_weak_for_test();
    let mut ticket = reserved(&scope, scratch).start(requests, None);
    let joined = ticket.join().await;
    assert!(
        input.upgrade().is_some(),
        "joined results still need original routing"
    );
    assert_eq!(owner.active_bytes(), scratch.accounted_bytes().unwrap());
    drop(joined);
    assert!(
        input.upgrade().is_none(),
        "seed cannot retain routing after permit release"
    );
    assert_eq!(owner.active_bytes(), 0);
    std::hint::black_box(&seed);
    drop(seed);
    drop(input);
    owner.close_admission();
    assert!(owner.drain().await.is_empty());
}

#[tokio::test]
async fn abandoned_ticket_cancels_but_drain_waits_for_real_handles_without_blocking_tokio() {
    let owner = owner();
    let gate = Arc::new(Gate::default());
    let _release = ReleaseGate(gate.clone());
    owner.set_hooks(TestHooks {
        before_run: Some({
            let gate = gate.clone();
            Arc::new(move |_| gate.block())
        }),
        ..TestHooks::default()
    });
    let scope = owner
        .client(TaskId::new(3), "operator:rolling".into())
        .callback_scope();
    let (_, requests, scratch) = entity_parallel_test_pair();
    let ticket = reserved(&scope, scratch).start(requests, None);
    gate.wait_for(2).await;
    drop(ticket);
    assert!(owner.work_cancelled());
    let settle = scope.settle_abandoned();
    tokio::pin!(settle);
    assert!(futures::poll!(&mut settle).is_pending());
    tokio::task::yield_now().await;
    assert!(owner.active_bytes() > 0);
    gate.release();
    settle.await;
    assert_eq!(owner.active_bytes(), 0);
    assert_eq!(owner.launch_count(), 2);
}

#[tokio::test]
async fn partial_launch_panic_keeps_the_first_handle_until_scope_settlement() {
    let owner = owner();
    let store = crate::operator::rolling_metrics::RollingMetricsStore::default();
    let callback = store.begin(
        crate::operator::rolling_metrics::RollingCallback::Watermark,
        crate::CancellationToken::new(),
    );
    let gate = Arc::new(Gate::default());
    let _release = ReleaseGate(gate.clone());
    owner.set_hooks(TestHooks {
        before_run: Some({
            let gate = gate.clone();
            Arc::new(move |_| gate.block())
        }),
        panic_before_second_launch: true,
        ..TestHooks::default()
    });
    let scope = owner
        .client(TaskId::new(9), "operator:rolling".into())
        .callback_scope();
    let (_, requests, scratch) = entity_parallel_test_pair();
    let launched = catch_unwind(AssertUnwindSafe(|| {
        reserved(&scope, scratch).start(requests, Some(callback.recorder()))
    }));
    assert!(launched.is_err());
    drop(callback);
    gate.wait_for(1).await;
    assert_eq!(owner.launch_count(), 1);
    assert_eq!(store.snapshot().watermark.interrupted, 0);
    let settle = scope.settle_abandoned();
    tokio::pin!(settle);
    assert!(futures::poll!(&mut settle).is_pending());
    gate.release();
    settle.await;
    assert_eq!(owner.active_bytes(), 0);
    assert_eq!(store.snapshot().watermark.interrupted, 1);
    assert_eq!(store.snapshot().watermark.failed, 0);
    assert_eq!(store.snapshot().watermark.numeric_rows, 0);
    scope.settle_abandoned().await;
    assert_eq!(store.snapshot().watermark.interrupted, 1);
}

#[tokio::test]
async fn abandoned_real_lane_panic_is_secondary_and_settles_actual_work_once() {
    let owner = owner();
    owner.set_hooks(TestHooks {
        after_run: Some(Arc::new(|lane| {
            assert!(lane != 1, "late numeric lane panic");
        })),
        ..TestHooks::default()
    });
    let store = crate::operator::rolling_metrics::RollingMetricsStore::default();
    let callback = store.begin(
        crate::operator::rolling_metrics::RollingCallback::Watermark,
        crate::CancellationToken::new(),
    );
    let scope = owner
        .client(TaskId::new(11), "operator:rolling".into())
        .callback_scope();
    let (_, requests, scratch) = entity_parallel_test_pair();
    let ticket = reserved(&scope, scratch).start(requests, Some(callback.recorder()));
    // Settle after the genuine numeric workers finish, then abandon their observer.
    owner.wait_for_workers().await;
    drop(ticket);
    drop(callback);
    assert_eq!(store.snapshot().watermark.interrupted, 0);
    scope.settle_abandoned().await;
    let observation = store.snapshot();
    assert_eq!(observation.watermark.interrupted, 1);
    assert_eq!(observation.watermark.failed, 0);
    assert_eq!(observation.watermark.numeric_rows, 64_000);
    owner.close_admission();
    let diagnostics = owner.drain().await;
    assert_eq!(diagnostics.len(), 1);
    assert_eq!(diagnostics[0].task_id, TaskId::new(11));
    assert_eq!(diagnostics[0].task_name, "operator:rolling");
    assert!(matches!(
        diagnostics[0].error,
        CalcFlowError::TaskPanicked { task_id: 11, .. }
    ));
    assert!(owner.drain().await.is_empty());
}

#[tokio::test]
async fn closed_scope_and_exhausted_generation_never_launch_or_retain_clients() {
    let owner = owner();
    let weak = Arc::downgrade(&owner.0);
    let task = owner.client(TaskId::new(1), "operator:rolling".into());
    let scope = task.callback_scope();
    let client = scope.context_client();
    let (_, requests, scratch) = entity_parallel_test_pair();
    drop(requests);
    owner.set_generation(u64::MAX);
    assert!(matches!(
        client.try_reserve(scratch),
        ReservePair::Serial(SerialReason::GenerationExhausted)
    ));
    drop(scope);
    assert!(matches!(client.try_reserve(scratch), ReservePair::Stopped));
    owner.close_admission();
    assert!(owner.drain().await.is_empty());
    drop(owner);
    assert!(weak.upgrade().is_none());
    assert!(matches!(client.try_reserve(scratch), ReservePair::Stopped));
}

#[tokio::test]
async fn null_free_merge_unwind_releases_joined_payload_and_real_permit() {
    let owner = owner();
    let task = owner.client(TaskId::new(3), "operator:rolling".into());
    let scope = task.callback_scope();
    let (mut seed, requests, scratch) = entity_parallel_test_pair();
    seed.panic_before_output_finish_for_test();
    let input = requests[0].input_weak_for_test();
    let mut ticket = reserved(&scope, scratch).start(requests, None);
    let joined = ticket.join().await;
    assert_eq!(owner.active_bytes(), scratch.accounted_bytes().unwrap());
    let panic = catch_unwind(AssertUnwindSafe(|| joined.merge(seed)))
        .err()
        .unwrap();
    assert_eq!(*panic.downcast::<u32>().unwrap(), 673);
    assert!(input.upgrade().is_none());
    assert_eq!(owner.active_bytes(), 0);
    let next_scope = task.callback_scope();
    assert!(matches!(
        next_scope.context_client().try_reserve(scratch),
        ReservePair::Reserved(_)
    ));
    drop(next_scope);
    drop(scope);
    owner.close_admission();
    assert!(owner.drain().await.is_empty());
}
