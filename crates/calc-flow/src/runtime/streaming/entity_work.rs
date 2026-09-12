//! One job-owned numeric pair; observers never own its blocking join handles.

use std::{
    future::{Future, poll_fn},
    panic::{AssertUnwindSafe, catch_unwind},
    pin::Pin,
    sync::{
        Arc, Weak,
        atomic::{AtomicBool, Ordering},
    },
    task::{Context, Poll},
};

use parking_lot::Mutex;
use tokio::{sync::Notify, task::JoinHandle};

use super::supervisor::{TaskFailure, TaskId, panic_message};
use crate::{
    CalcFlowError, Result,
    operator::{
        rolling::{
            ActualNumericWork, LaneProgress, LocatedPanic, NumericJoinSeed, NumericLaneExit,
            NumericLaneOutcome, NumericLaneRequest, NumericLaneStop, ScratchPlan,
            StreamKernelUpdate, merge_numeric_results, run_numeric_lane,
        },
        rolling_metrics::{NumericWorkLease, RollingMetricsRecorder},
    },
};

#[cfg(test)]
mod tests;

#[derive(Clone)]
pub(crate) struct JobEntityWorkOwner(Arc<OwnerCell>);

struct OwnerCell {
    run_id: Arc<str>,
    state: Mutex<OwnerState>,
    changed: Notify,
    #[cfg(test)]
    launches: Arc<std::sync::atomic::AtomicU64>,
    #[cfg(test)]
    hooks: Mutex<TestHooks>,
}

struct OwnerState {
    open: bool,
    generation: u64,
    slot: Slot,
    diagnostics: [Option<TaskFailure>; 2],
}

enum Slot {
    Empty,
    Active(Box<PairRecord>),
    Borrowed { id: u64, _bytes: usize },
}

struct PairRecord {
    id: u64,
    bytes: usize,
    task_id: TaskId,
    task_name: Arc<str>,
    cancellation: Arc<AtomicBool>,
    abandoned: bool,
    launching: bool,
    requests: [Option<NumericLaneRequest>; 2],
    handles: [Option<JoinHandle<NumericLaneExit>>; 2],
    exits: [Option<NumericLaneExit>; 2],
    lease: Option<NumericWorkLease>,
}

impl PairRecord {
    fn poll_handles(&mut self, cx: &mut Context<'_>) -> bool {
        let mut pending = false;
        for lane in 0..2 {
            if let Some(handle) = &mut self.handles[lane] {
                match Pin::new(handle).poll(cx) {
                    Poll::Pending => pending = true,
                    Poll::Ready(result) => {
                        self.handles[lane] = None;
                        self.exits[lane] = Some(result.unwrap_or_else(failed_lane_join));
                    }
                }
            }
        }
        pending
    }

    fn settle_work(&mut self, exits: &[NumericLaneExit; 2]) {
        let rows = exits[0]
            .work
            .numeric_rows
            .checked_add(exits[1].work.numeric_rows);
        if let Some(lease) = self.lease.take() {
            lease.settle(
                rows.unwrap_or(0),
                rows.is_none() || exits.iter().any(|exit| exit.work.overflowed),
            );
        }
    }
}

fn failed_lane_join(error: tokio::task::JoinError) -> NumericLaneExit {
    NumericLaneExit {
        outcome: if error.is_panic() {
            NumericLaneOutcome::Panicked(LocatedPanic {
                site: None,
                payload: error.into_panic(),
            })
        } else {
            NumericLaneOutcome::Cancelled
        },
        work: ActualNumericWork::default(),
    }
}

impl JobEntityWorkOwner {
    pub(crate) fn new(
        job_id: u64,
        #[cfg(test)] launches: Arc<std::sync::atomic::AtomicU64>,
    ) -> Self {
        Self(Arc::new(OwnerCell {
            run_id: job_id.to_string().into(),
            state: Mutex::new(OwnerState {
                open: true,
                generation: 0,
                slot: Slot::Empty,
                diagnostics: [None, None],
            }),
            changed: Notify::new(),
            #[cfg(test)]
            launches,
            #[cfg(test)]
            hooks: Mutex::new(TestHooks::default()),
        }))
    }

    pub(crate) fn unbound_client(&self, task_name: Arc<str>) -> TaskEntityWorkClient {
        TaskEntityWorkClient {
            owner: Arc::downgrade(&self.0),
            task_id: None,
            task_name,
            scope: Mutex::new(Weak::new()),
        }
    }

    #[cfg(test)]
    fn client(&self, task_id: TaskId, task_name: Arc<str>) -> TaskEntityWorkClient {
        let mut client = self.unbound_client(task_name);
        client.activate(task_id);
        client
    }

    #[cfg(test)]
    pub(crate) fn test_client(&self, task_id: u64, task_name: Arc<str>) -> TaskEntityWorkClient {
        self.client(TaskId::new(task_id), task_name)
    }

    pub(crate) fn close_admission(&self) {
        let mut state = self.0.state.lock();
        state.open = false;
        if let Slot::Active(pair) = &state.slot {
            pair.cancellation.store(true, Ordering::SeqCst);
        }
        self.0.changed.notify_waiters();
    }

    pub(crate) async fn drain(&self) -> Vec<TaskFailure> {
        loop {
            let changed = self.0.changed.notified();
            tokio::pin!(changed);
            changed.as_mut().enable();
            let id = {
                let state = self.0.state.lock();
                match &state.slot {
                    Slot::Empty => break,
                    Slot::Active(pair) if pair.abandoned => Some(pair.id),
                    Slot::Active(_) | Slot::Borrowed { .. } => None,
                }
            };
            if let Some(id) = id {
                let joined = poll_fn(|cx| self.0.poll_pair(id, true, cx)).await;
                drop(joined);
            } else {
                changed.await;
            }
        }
        self.0
            .state
            .lock()
            .diagnostics
            .iter_mut()
            .filter_map(Option::take)
            .collect()
    }
}

pub(crate) struct TaskEntityWorkClient {
    owner: Weak<OwnerCell>,
    task_id: Option<TaskId>,
    task_name: Arc<str>,
    scope: Mutex<Weak<ScopeState>>,
}

impl TaskEntityWorkClient {
    pub(crate) fn activate(&mut self, task_id: TaskId) {
        self.task_id = Some(task_id);
    }

    pub(crate) fn callback_scope(&self) -> CallbackWorkScope {
        let scope = Arc::new(ScopeState {
            owner: self.owner.clone(),
            task_id: self
                .task_id
                .expect("managed data gate activates the logical task"),
            task_name: self.task_name.clone(),
            work: Mutex::new(None),
            closed: AtomicBool::new(false),
        });
        *self.scope.lock() = Arc::downgrade(&scope);
        CallbackWorkScope(scope)
    }

    pub(crate) fn context_client(&self) -> Option<ScopedEntityWorkClient> {
        self.scope.lock().upgrade().map(ScopedEntityWorkClient)
    }
}

struct ScopeState {
    owner: Weak<OwnerCell>,
    task_id: TaskId,
    task_name: Arc<str>,
    work: Mutex<Option<u64>>,
    closed: AtomicBool,
}

pub(crate) struct CallbackWorkScope(Arc<ScopeState>);

impl CallbackWorkScope {
    #[cfg(test)]
    pub(crate) fn context_client(&self) -> ScopedEntityWorkClient {
        ScopedEntityWorkClient(self.0.clone())
    }

    pub(crate) async fn settle_abandoned(&self) {
        let id = *self.0.work.lock();
        if let Some(id) = id
            && let Some(owner) = self.0.owner.upgrade()
        {
            let joined = poll_fn(|cx| owner.poll_pair(id, true, cx)).await;
            drop(joined);
        }
    }
}

impl Drop for CallbackWorkScope {
    fn drop(&mut self) {
        self.0.closed.store(true, Ordering::SeqCst);
        if let Some(id) = *self.0.work.lock()
            && let Some(owner) = self.0.owner.upgrade()
        {
            owner.abandon(id);
        }
    }
}

#[derive(Clone)]
pub(crate) struct ScopedEntityWorkClient(Arc<ScopeState>);

#[derive(Debug, Eq, PartialEq)]
pub(crate) enum SerialReason {
    Busy,
    GenerationExhausted,
    AlreadyUsed,
    ScratchLimit,
    Unrepresentable,
}

pub(crate) enum ReservePair {
    Reserved(PairLaunchGuard),
    Serial(SerialReason),
    Stopped,
}

impl ScopedEntityWorkClient {
    pub(crate) fn try_reserve(&self, scratch: ScratchPlan) -> ReservePair {
        let Some(owner) = self.0.owner.upgrade() else {
            return ReservePair::Stopped;
        };
        let mut state = owner.state.lock();
        if !state.open || self.0.closed.load(Ordering::SeqCst) {
            return ReservePair::Stopped;
        }
        if !matches!(state.slot, Slot::Empty) {
            return ReservePair::Serial(SerialReason::Busy);
        }
        let mut used = self.0.work.lock();
        if used.is_some() {
            return ReservePair::Serial(SerialReason::AlreadyUsed);
        }
        let Some(bytes) = scratch.accounted_bytes() else {
            return ReservePair::Serial(SerialReason::Unrepresentable);
        };
        if bytes > ScratchPlan::CAP_BYTES {
            return ReservePair::Serial(SerialReason::ScratchLimit);
        }
        let Some(id) = state.generation.checked_add(1) else {
            return ReservePair::Serial(SerialReason::GenerationExhausted);
        };
        state.generation = id;
        state.slot = Slot::Active(Box::new(PairRecord {
            id,
            bytes,
            task_id: self.0.task_id,
            task_name: self.0.task_name.clone(),
            cancellation: Arc::new(AtomicBool::new(false)),
            abandoned: false,
            launching: true,
            requests: [None, None],
            handles: [None, None],
            exits: [None, None],
            lease: None,
        }));
        *used = Some(id);
        ReservePair::Reserved(PairLaunchGuard {
            owner: Arc::downgrade(&owner),
            id,
            active: true,
        })
    }
}

pub(crate) struct PairLaunchGuard {
    owner: Weak<OwnerCell>,
    id: u64,
    active: bool,
}

impl PairLaunchGuard {
    pub(crate) fn start(
        mut self,
        requests: [NumericLaneRequest; 2],
        recorder: Option<RollingMetricsRecorder>,
    ) -> PairTicket {
        let owner = self
            .owner
            .upgrade()
            .expect("the managed job owns the launch");
        let mut state = owner.state.lock();
        let Slot::Active(pair) = &mut state.slot else {
            unreachable!("reserved pair remains owned");
        };
        assert_eq!(pair.id, self.id);
        pair.requests = requests.map(Some);
        pair.lease = recorder.and_then(|recorder| recorder.numeric_lease(self.id));
        for lane in 0..2 {
            #[cfg(test)]
            assert!(
                lane != 1 || !owner.hooks.lock().panic_before_second_launch,
                "injected partial numeric launch"
            );
            let request = pair.requests[lane].take().expect("one request per lane");
            let cancellation = Arc::clone(&pair.cancellation);
            #[cfg(test)]
            let hooks = owner.hooks.lock().clone();
            let handle = tokio::task::spawn_blocking(move || {
                let mut progress = LaneProgress::default();
                let result = catch_unwind(AssertUnwindSafe(|| {
                    #[cfg(test)]
                    if let Some(hook) = &hooks.before_run {
                        hook(lane);
                    }
                    let result = run_numeric_lane(request, &cancellation, &mut progress);
                    #[cfg(test)]
                    if let Some(hook) = &hooks.after_run {
                        hook(lane);
                    }
                    result
                }));
                let outcome = match result {
                    Ok(Ok(result)) => NumericLaneOutcome::Complete(result),
                    Ok(Err(NumericLaneStop::Ordinary(error))) => {
                        NumericLaneOutcome::OrdinaryError(error)
                    }
                    Ok(Err(NumericLaneStop::Cancelled)) => NumericLaneOutcome::Cancelled,
                    Err(payload) => NumericLaneOutcome::Panicked(LocatedPanic {
                        site: progress.active_site,
                        payload,
                    }),
                };
                NumericLaneExit {
                    outcome,
                    work: progress.work,
                }
            });
            // The first returned handle is registered before another launch.
            pair.handles[lane] = Some(handle);
            #[cfg(test)]
            owner.launches.fetch_add(1, Ordering::SeqCst);
        }
        pair.launching = false;
        self.active = false;
        PairTicket {
            owner: self.owner.clone(),
            id: self.id,
            active: true,
        }
    }
}

impl Drop for PairLaunchGuard {
    fn drop(&mut self) {
        if self.active
            && let Some(owner) = self.owner.upgrade()
        {
            owner.abandon(self.id);
        }
    }
}

pub(crate) struct PairTicket {
    owner: Weak<OwnerCell>,
    id: u64,
    active: bool,
}

impl PairTicket {
    pub(crate) async fn join(&mut self) -> JoinedNumericPair {
        let owner = self
            .owner
            .upgrade()
            .expect("the managed job owns the numeric pair");
        let joined = poll_fn(|cx| owner.poll_pair(self.id, false, cx))
            .await
            .expect("the active ticket is the unique candidate observer");
        self.active = false;
        joined
    }
}

impl Drop for PairTicket {
    fn drop(&mut self) {
        if self.active
            && let Some(owner) = self.owner.upgrade()
        {
            owner.abandon(self.id);
        }
    }
}

impl OwnerCell {
    fn abandon(&self, id: u64) {
        let mut state = self.state.lock();
        if let Slot::Active(pair) = &mut state.slot
            && pair.id == id
        {
            pair.abandoned = true;
            pair.launching = false;
            pair.cancellation.store(true, Ordering::SeqCst);
        }
        self.changed.notify_waiters();
    }

    fn poll_pair(
        self: &Arc<Self>,
        id: u64,
        abandoned_only: bool,
        cx: &mut Context<'_>,
    ) -> Poll<Option<JoinedNumericPair>> {
        let mut state = self.state.lock();
        let Slot::Active(pair) = &mut state.slot else {
            return Poll::Ready(None);
        };
        if pair.id != id || (abandoned_only && !pair.abandoned) {
            return Poll::Ready(None);
        }
        if pair.launching {
            return Poll::Pending;
        }
        if pair.poll_handles(cx) {
            return Poll::Pending;
        }
        let bytes = pair.bytes;
        let Slot::Active(mut pair) =
            std::mem::replace(&mut state.slot, Slot::Borrowed { id, _bytes: bytes })
        else {
            unreachable!();
        };
        let exits = pair.exits.each_mut().map(|exit| {
            exit.take().unwrap_or(NumericLaneExit {
                outcome: NumericLaneOutcome::Cancelled,
                work: ActualNumericWork::default(),
            })
        });
        pair.settle_work(&exits);
        Poll::Ready(Some(JoinedNumericPair {
            owner: Arc::downgrade(self),
            id,
            task_id: pair.task_id,
            task_name: pair.task_name,
            run_id: self.run_id.clone(),
            exits: Some(exits),
            requests: pair.requests,
        }))
    }

    fn retain_diagnostic(&self, failure: TaskFailure) {
        let mut state = self.state.lock();
        if let Some(slot) = state.diagnostics.iter_mut().find(|slot| slot.is_none()) {
            *slot = Some(failure);
        }
    }

    fn release_borrowed(&self, id: u64) {
        let mut state = self.state.lock();
        if matches!(state.slot, Slot::Borrowed { id: current, .. } if current == id) {
            state.slot = Slot::Empty;
        }
        self.changed.notify_waiters();
    }
}

pub(crate) struct JoinedNumericPair {
    owner: Weak<OwnerCell>,
    id: u64,
    task_id: TaskId,
    task_name: Arc<str>,
    run_id: Arc<str>,
    exits: Option<[NumericLaneExit; 2]>,
    requests: [Option<NumericLaneRequest>; 2],
}

impl JoinedNumericPair {
    pub(crate) fn merge(mut self, seed: NumericJoinSeed) -> Result<StreamKernelUpdate> {
        merge_numeric_results(
            seed,
            self.exits.as_mut().expect("unique pair merge"),
            &self.run_id,
        )
    }

    fn retain_exit_diagnostic(&self, outcome: NumericLaneOutcome, owner: Option<&OwnerCell>) {
        let error = match outcome {
            NumericLaneOutcome::OrdinaryError(error) => Some(error.error),
            NumericLaneOutcome::Panicked(panic) => Some(CalcFlowError::TaskPanicked {
                task_id: self.task_id.as_u64(),
                message: panic_message(panic.payload.as_ref()),
            }),
            NumericLaneOutcome::Complete(_)
            | NumericLaneOutcome::Cancelled
            | NumericLaneOutcome::Consumed => None,
        };
        if let Some(error) = error
            && let Some(owner) = owner
        {
            owner.retain_diagnostic(TaskFailure {
                task_id: self.task_id,
                task_name: self.task_name.to_string(),
                error,
            });
        }
    }
}

impl Drop for JoinedNumericPair {
    fn drop(&mut self) {
        let owner = self.owner.upgrade();
        if let Some(exits) = self.exits.take() {
            for exit in exits {
                self.retain_exit_diagnostic(exit.outcome, owner.as_deref());
            }
        }
        for request in &mut self.requests {
            drop(request.take());
        }
        // Drop every remaining lane payload before making its scratch slot reusable.
        if let Some(owner) = owner {
            owner.release_borrowed(self.id);
        }
    }
}

#[cfg(test)]
#[derive(Clone, Default)]
pub(crate) struct TestHooks {
    pub(crate) before_run: Option<Arc<dyn Fn(usize) + Send + Sync>>,
    pub(crate) after_run: Option<Arc<dyn Fn(usize) + Send + Sync>>,
    pub(crate) panic_before_second_launch: bool,
}

#[cfg(test)]
impl JobEntityWorkOwner {
    pub(crate) fn set_hooks(&self, hooks: TestHooks) {
        *self.0.hooks.lock() = hooks;
    }
    pub(crate) fn active_bytes(&self) -> usize {
        match &self.0.state.lock().slot {
            Slot::Empty => 0,
            Slot::Active(pair) => pair.bytes,
            Slot::Borrowed { _bytes: bytes, .. } => *bytes,
        }
    }
    fn launch_count(&self) -> u64 {
        self.0.launches.load(Ordering::SeqCst)
    }
    fn work_cancelled(&self) -> bool {
        matches!(&self.0.state.lock().slot, Slot::Active(pair) if pair.cancellation.load(Ordering::SeqCst))
    }
    fn set_generation(&self, generation: u64) {
        self.0.state.lock().generation = generation;
    }
    async fn wait_for_workers(&self) {
        tokio::time::timeout(std::time::Duration::from_secs(5), async {
            loop {
                let finished = {
                    let state = self.0.state.lock();
                    match &state.slot {
                        Slot::Active(pair) => {
                            pair.handles.iter().flatten().all(JoinHandle::is_finished)
                        }
                        _ => true,
                    }
                };
                if finished {
                    return;
                }
                tokio::task::yield_now().await;
            }
        })
        .await
        .unwrap();
    }
}
