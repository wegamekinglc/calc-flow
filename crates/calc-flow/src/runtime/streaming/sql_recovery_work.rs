use std::{
    collections::BTreeMap,
    panic::{AssertUnwindSafe, catch_unwind},
    sync::{Arc, Weak},
};

use chrono::{DateTime, Utc};
use datafusion::execution::memory_pool::MemoryReservation;
use parking_lot::Mutex;
use tokio::{
    sync::Notify,
    task::{JoinError, JoinHandle},
};

use super::{
    context::StreamJobContext,
    failure::{FailureOrigin, RuntimeFailure, panic_message},
    supervisor::TaskId,
};
use crate::{
    CalcFlowError, CancellationToken, OperatorStateSnapshot, Result, SqlOperator,
    operator::{PreparedSqlCheckpoint, PreparedSqlRestore},
};

#[derive(Clone)]
pub(crate) struct JobSqlRecoveryOwner(Arc<Home>);

struct Home {
    state: Mutex<State>,
    changed: Notify,
    stop: CancellationToken,
    #[cfg(test)]
    join_returns: std::sync::atomic::AtomicUsize,
    #[cfg(test)]
    panic_identity: Mutex<Option<u64>>,
}

#[derive(Default)]
struct State {
    configured: bool,
    capacity: usize,
    next: u64,
    active: Option<u64>,
    attempts: BTreeMap<u64, Attempt>,
    cleanup: Vec<(usize, u64, Arc<RuntimeFailure>)>,
    overflow: u64,
}

#[derive(Clone)]
pub(crate) struct SqlRecoveryClient {
    pub(crate) owner: JobSqlRecoveryOwner,
    pub(crate) node_order: usize,
    pub(crate) task_id: Option<TaskId>,
}

impl SqlRecoveryClient {
    pub(crate) fn activate(&mut self, task_id: TaskId) {
        self.task_id = Some(task_id);
    }
}

#[derive(Clone)]
pub(crate) struct SqlRecoveryContext {
    job_id: u64,
    deadline: Option<DateTime<Utc>>,
    cancellation: CancellationToken,
}

impl From<&StreamJobContext> for SqlRecoveryContext {
    fn from(context: &StreamJobContext) -> Self {
        Self {
            job_id: context.job_id(),
            deadline: context.deadline().copied(),
            cancellation: context.cancellation().clone(),
        }
    }
}

pub(crate) struct SqlRecoverySubmission {
    pub(crate) result: Result<SqlRecoveryTicket>,
    pub(crate) operator: Option<SqlOperator>,
}

impl SqlRecoverySubmission {
    fn refused(operator: SqlOperator, error: CalcFlowError) -> Self {
        Self {
            result: Err(error),
            operator: Some(operator),
        }
    }
}

pub(crate) struct SqlRestoreIdentity {
    pub(crate) node_id: String,
    pub(crate) node_order: usize,
    pub(crate) task_id: Option<TaskId>,
}

pub(crate) struct SqlRestoreRequest {
    pub(crate) operator: SqlOperator,
    pub(crate) snapshot: OperatorStateSnapshot,
    pub(crate) identity: SqlRestoreIdentity,
    pub(crate) context: SqlRecoveryContext,
    pub(crate) launch_cancel: CancellationToken,
}

pub(crate) struct SqlCaptureRequest {
    pub(crate) operator: SqlOperator,
    pub(crate) identity: SqlRestoreIdentity,
    pub(crate) context: SqlRecoveryContext,
    pub(crate) launch_cancel: CancellationToken,
    pub(crate) reservation: MemoryReservation,
}

enum SqlWorkOperation {
    Restore(OperatorStateSnapshot),
    Capture,
}

struct SqlWorkRequest {
    operator: SqlOperator,
    operation: SqlWorkOperation,
    identity: SqlRestoreIdentity,
    context: SqlRecoveryContext,
    launch_cancel: CancellationToken,
}

pub(crate) enum PreparedSqlWork {
    Restore(Box<PreparedSqlRestore>),
    Capture(PreparedSqlCheckpoint),
}

impl PreparedSqlWork {
    pub(crate) fn into_restore(self) -> Result<PreparedSqlRestore> {
        match self {
            Self::Restore(prepared) => Ok(*prepared),
            Self::Capture(_) => Err(wrong_work_operation()),
        }
    }

    pub(crate) fn into_checkpoint(self) -> Result<PreparedSqlCheckpoint> {
        match self {
            Self::Capture(prepared) => Ok(prepared),
            Self::Restore(_) => Err(wrong_work_operation()),
        }
    }
}

fn wrong_work_operation() -> CalcFlowError {
    CalcFlowError::Internal {
        message: "SQL work returned a different checkpoint operation".into(),
    }
}

struct Work {
    home: Weak<Home>,
    id: u64,
    operator: SqlOperator,
    operation: SqlWorkOperation,
    identity: SqlRestoreIdentity,
    context: SqlRecoveryContext,
    launch_cancel: CancellationToken,
    stop: CancellationToken,
    home_stop: CancellationToken,
    reservation: MemoryReservation,
}

struct Attempt {
    identity: SqlRestoreIdentity,
    stop: CancellationToken,
    work: Option<Work>,
    handle: Option<JoinHandle<OwnedSqlRestoreCompletion>>,
    loaned: bool,
    abandoned: bool,
}

pub(crate) struct OwnedSqlRestoreCompletion {
    home: Weak<Home>,
    id: u64,
    identity: SqlRestoreIdentity,
    pub(crate) operator: SqlOperator,
    pub(crate) prepared: Result<PreparedSqlWork>,
    context: SqlRecoveryContext,
    launch_cancel: CancellationToken,
    stop: CancellationToken,
    home_stop: CancellationToken,
    _reservation: MemoryReservation,
}

impl OwnedSqlRestoreCompletion {
    pub(crate) fn check_current(&mut self) -> Result<()> {
        let current = check_stop(
            &self.context,
            &self.launch_cancel,
            &self.stop,
            &self.home_stop,
        );
        if current.is_err() {
            let previous = std::mem::replace(&mut self.prepared, Err(cancelled(&self.context)));
            if let Err(error) = previous
                && !matches!(error, CalcFlowError::Cancelled { .. })
                && let Some(home) = self.home.upgrade()
            {
                record_cleanup(&home, &self.identity, self.id, error);
            }
        }
        current
    }
}

struct QueuedSqlStop {
    context: SqlRecoveryContext,
    launch: CancellationToken,
    attempt: CancellationToken,
    home: CancellationToken,
}

impl QueuedSqlStop {
    async fn cancelled(&self) {
        tokio::select! {
            () = self.context.cancellation.cancelled() => {},
            () = self.launch.cancelled() => {},
            () = self.attempt.cancelled() => {},
            () = self.home.cancelled() => {},
            () = wait_deadline(self.context.deadline) => {},
        }
    }
}

async fn wait_deadline(deadline: Option<DateTime<Utc>>) {
    let Some(deadline) = deadline else {
        return std::future::pending().await;
    };
    loop {
        let Ok(delay) = (deadline - Utc::now()).to_std() else {
            return;
        };
        if delay.is_zero() {
            return;
        }
        tokio::time::sleep(delay).await;
    }
}

impl JobSqlRecoveryOwner {
    pub(crate) fn new() -> Self {
        Self(Arc::new(Home {
            state: Mutex::new(State::default()),
            changed: Notify::new(),
            stop: CancellationToken::new(),
            #[cfg(test)]
            join_returns: std::sync::atomic::AtomicUsize::new(0),
            #[cfg(test)]
            panic_identity: Mutex::new(None),
        }))
    }

    pub(crate) fn configure(&self, capacity: usize) -> Result<()> {
        let mut state = self.0.state.lock();
        if state.configured {
            return Err(CalcFlowError::Internal {
                message: "SQL recovery owner was configured twice".into(),
            });
        }
        state.capacity = capacity;
        state.configured = true;
        Ok(())
    }

    pub(crate) fn submit(&self, mut request: SqlRestoreRequest) -> SqlRecoverySubmission {
        let reservation = match request
            .operator
            .reserve_recovery_envelope(&request.snapshot)
        {
            Ok(reservation) => reservation,
            Err(error) => return SqlRecoverySubmission::refused(request.operator, error),
        };
        self.submit_paid(
            SqlWorkRequest {
                operator: request.operator,
                operation: SqlWorkOperation::Restore(request.snapshot),
                identity: request.identity,
                context: request.context,
                launch_cancel: request.launch_cancel,
            },
            reservation,
        )
    }

    pub(crate) fn submit_capture(&self, request: SqlCaptureRequest) -> SqlRecoverySubmission {
        self.submit_paid(
            SqlWorkRequest {
                operator: request.operator,
                operation: SqlWorkOperation::Capture,
                identity: request.identity,
                context: request.context,
                launch_cancel: request.launch_cancel,
            },
            request.reservation,
        )
    }

    fn submit_paid(
        &self,
        request: SqlWorkRequest,
        reservation: MemoryReservation,
    ) -> SqlRecoverySubmission {
        let run_id = request.context.job_id;
        let mut state = self.0.state.lock();
        if self.0.stop.is_cancelled() || state.attempts.len() >= state.capacity {
            return SqlRecoverySubmission::refused(request.operator, cancelled(&request.context));
        }
        let Some(id) = state.next.checked_add(1) else {
            return SqlRecoverySubmission::refused(
                request.operator,
                CalcFlowError::Internal {
                    message: "SQL recovery attempt sequence overflowed".into(),
                },
            );
        };
        state.next = id;
        let stop = CancellationToken::new();
        let identity = SqlRestoreIdentity {
            node_id: request.identity.node_id.clone(),
            node_order: request.identity.node_order,
            task_id: request.identity.task_id,
        };
        state.attempts.insert(
            id,
            Attempt {
                identity,
                stop: stop.clone(),
                work: Some(Work {
                    home: Arc::downgrade(&self.0),
                    id,
                    operator: request.operator,
                    operation: request.operation,
                    identity: request.identity,
                    context: request.context,
                    launch_cancel: request.launch_cancel,
                    stop: stop.clone(),
                    home_stop: self.0.stop.clone(),
                    reservation,
                }),
                handle: None,
                loaned: false,
                abandoned: false,
            },
        );
        drop(state);
        self.0.changed.notify_waiters();
        SqlRecoverySubmission {
            result: Ok(SqlRecoveryTicket {
                owner: self.clone(),
                id,
                run_id,
                completed: false,
            }),
            operator: None,
        }
    }

    #[cfg(test)]
    pub(crate) fn returned_loans(&self) -> usize {
        self.0
            .join_returns
            .load(std::sync::atomic::Ordering::SeqCst)
    }

    #[cfg(test)]
    pub(crate) fn panic_identity(&self) -> Option<u64> {
        *self.0.panic_identity.lock()
    }

    #[cfg(test)]
    pub(crate) fn live_counts(&self) -> (usize, usize) {
        let state = self.0.state.lock();
        (
            state.attempts.len(),
            state
                .attempts
                .values()
                .filter(|attempt| attempt.loaned)
                .count(),
        )
    }

    #[cfg(test)]
    pub(crate) fn active_task_for_test(&self, node_id: &str) -> Option<u64> {
        let state = self.0.state.lock();
        let attempt = state.attempts.get(&state.active?)?;
        if attempt.identity.node_id != node_id
            || attempt.work.is_some()
            || !(attempt.loaned ^ attempt.handle.is_some())
        {
            return None;
        }
        attempt.identity.task_id.map(TaskId::as_u64)
    }

    pub(crate) fn close_admission(&self) {
        self.0.stop.cancel();
        for attempt in self.0.state.lock().attempts.values() {
            attempt.stop.cancel();
        }
        self.0.changed.notify_waiters();
    }

    fn queued_stop(&self, id: u64) -> Option<QueuedSqlStop> {
        let state = self.0.state.lock();
        let work = state.attempts.get(&id)?.work.as_ref()?;
        Some(QueuedSqlStop {
            context: work.context.clone(),
            launch: work.launch_cancel.clone(),
            attempt: work.stop.clone(),
            home: work.home_stop.clone(),
        })
    }

    fn retire_stopped_queue(&self, id: u64) -> bool {
        let retired = {
            let mut state = self.0.state.lock();
            let stopped = state
                .attempts
                .get(&id)
                .and_then(|attempt| attempt.work.as_ref())
                .is_some_and(|work| {
                    check_stop(
                        &work.context,
                        &work.launch_cancel,
                        &work.stop,
                        &work.home_stop,
                    )
                    .is_err()
                });
            stopped.then(|| state.attempts.remove(&id)).flatten()
        };
        let removed = retired.is_some();
        drop(retired);
        if removed {
            self.0.changed.notify_waiters();
        }
        removed
    }

    fn claim(&self, id: u64) -> Result<Option<SqlRecoveryJoinLoan>> {
        let mut state = self.0.state.lock();
        let mut start_gate = None;
        if state.active.is_none() {
            let Some(&next) = state.attempts.keys().next() else {
                return Ok(None);
            };
            if next != id {
                return Ok(None);
            }
            let Some(attempt) = state.attempts.get_mut(&id) else {
                return Ok(None);
            };
            if let Some(work) = &attempt.work
                && let Err(error) = check_stop(
                    &work.context,
                    &work.launch_cancel,
                    &work.stop,
                    &work.home_stop,
                )
            {
                let retired = state.attempts.remove(&id);
                drop(state);
                drop(retired);
                self.0.changed.notify_waiters();
                return Err(error);
            }
            let Some(work) = attempt.work.take() else {
                return Ok(None);
            };
            let (start, gate) = std::sync::mpsc::sync_channel(1);
            let task_id = attempt.identity.task_id;
            let node_id = attempt.identity.node_id.clone();
            let launched = launch_work(work, gate);
            let handle = match launched {
                Ok(handle) => handle,
                Err(payload) => {
                    state.attempts.remove(&id);
                    drop(state);
                    self.0.changed.notify_waiters();
                    return Err(panicked(task_id, &node_id, payload.as_ref()));
                }
            };
            attempt.handle = Some(handle);
            state.active = Some(id);
            start_gate = Some(start);
        }
        if state.active != Some(id) {
            return Ok(None);
        }
        let Some(attempt) = state.attempts.get_mut(&id) else {
            return Ok(None);
        };
        let Some(handle) = attempt.handle.take() else {
            return Ok(None);
        };
        attempt.loaned = true;
        let loan = SqlRecoveryJoinLoan {
            owner: self.clone(),
            id,
            handle: Some(handle),
        };
        drop(state);
        if let Some(start) = start_gate {
            let _ = start.send(());
        }
        Ok(Some(loan))
    }

    fn retire_abandoned_queue(&self) -> bool {
        let retired = {
            let mut state = self.0.state.lock();
            let id = state.attempts.iter().find_map(|(&id, attempt)| {
                (attempt.abandoned && attempt.work.is_some()).then_some(id)
            });
            id.and_then(|id| state.attempts.remove(&id))
        };
        if retired.is_none() {
            return false;
        }
        drop(retired);
        self.0.changed.notify_waiters();
        true
    }

    fn claim_abandoned(&self, waiter: u64) -> Option<SqlRecoveryJoinLoan> {
        let mut state = self.0.state.lock();
        let id = state.active.filter(|&id| id != waiter)?;
        let attempt = state.attempts.get_mut(&id)?;
        if !attempt.abandoned || attempt.loaned {
            return None;
        }
        let handle = attempt.handle.take()?;
        attempt.loaned = true;
        Some(SqlRecoveryJoinLoan {
            owner: self.clone(),
            id,
            handle: Some(handle),
        })
    }

    fn check_waiting(&self, id: u64, run_id: u64) -> Result<()> {
        if !self.0.state.lock().attempts.contains_key(&id) {
            return Err(CalcFlowError::Cancelled {
                run_id: run_id.to_string(),
            });
        }
        if self.retire_stopped_queue(id) {
            return Err(CalcFlowError::Cancelled {
                run_id: run_id.to_string(),
            });
        }
        Ok(())
    }

    async fn join(&self, id: u64, run_id: u64) -> Result<OwnedSqlRestoreCompletion> {
        loop {
            let changed = self.0.changed.notified();
            tokio::pin!(changed);
            changed.as_mut().enable();
            if self.retire_abandoned_queue() {
                continue;
            }
            self.check_waiting(id, run_id)?;
            let queued_stop = self.queued_stop(id);
            if let Some(loan) = self.claim_abandoned(id) {
                match &queued_stop {
                    Some(stop) => tokio::select! {
                        () = loan.discard() => {},
                        () = stop.cancelled() => {},
                    },
                    None => loan.discard().await,
                }
                continue;
            }
            if let Some(loan) = self.claim(id)? {
                return loan.join().await;
            }
            match &queued_stop {
                Some(stop) => tokio::select! {
                    () = changed => {},
                    () = stop.cancelled() => {},
                },
                None => changed.await,
            }
        }
    }

    pub(crate) async fn drain(&self) -> Vec<Arc<RuntimeFailure>> {
        self.close_admission();
        loop {
            let changed = self.0.changed.notified();
            tokio::pin!(changed);
            changed.as_mut().enable();
            let queued = {
                let mut state = self.0.state.lock();
                let id = state
                    .attempts
                    .iter()
                    .find_map(|(&id, attempt)| attempt.work.is_some().then_some(id));
                id.and_then(|id| state.attempts.remove(&id))
            };
            if let Some(queued) = queued {
                drop(queued);
                self.0.changed.notify_waiters();
                continue;
            }
            let Some(id) = self.0.state.lock().active else {
                break;
            };
            if let Ok(Some(loan)) = self.claim(id) {
                loan.discard().await;
                continue;
            }
            changed.await;
        }
        let mut state = self.0.state.lock();
        let mut failures = std::mem::take(&mut state.cleanup);
        let overflow = std::mem::take(&mut state.overflow);
        drop(state);
        if overflow > 0 {
            failures.push((
                usize::MAX,
                u64::MAX,
                Arc::new(RuntimeFailure {
                    origin: FailureOrigin::RunnerLifecycle,
                    error: CalcFlowError::Internal {
                        message: format!("{overflow} additional SQL recovery cleanup failures"),
                    },
                }),
            ));
        }
        failures.sort_by_key(|(order, id, _)| (*order, *id));
        failures
            .into_iter()
            .map(|(_, _, failure)| failure)
            .collect()
    }
}

pub(crate) struct SqlRecoveryTicket {
    owner: JobSqlRecoveryOwner,
    id: u64,
    run_id: u64,
    completed: bool,
}

impl SqlRecoveryTicket {
    pub(crate) async fn join(mut self) -> Result<OwnedSqlRestoreCompletion> {
        let completion = self.owner.join(self.id, self.run_id).await;
        self.completed = true;
        completion
    }
}

impl Drop for SqlRecoveryTicket {
    fn drop(&mut self) {
        if !self.completed {
            let mut state = self.owner.0.state.lock();
            if let Some(attempt) = state.attempts.get_mut(&self.id) {
                attempt.abandoned = true;
                attempt.stop.cancel();
            }
            self.owner.0.changed.notify_waiters();
        }
    }
}

struct SqlRecoveryJoinLoan {
    owner: JobSqlRecoveryOwner,
    id: u64,
    handle: Option<JoinHandle<OwnedSqlRestoreCompletion>>,
}

impl SqlRecoveryJoinLoan {
    fn finish(&self) -> Attempt {
        let mut state = self.owner.0.state.lock();
        let attempt = state
            .attempts
            .remove(&self.id)
            .expect("registered SQL attempt");
        assert_eq!(state.active.take(), Some(self.id));
        drop(state);
        self.owner.0.changed.notify_waiters();
        attempt
    }

    async fn discard(mut self) {
        let result = self.handle.as_mut().expect("owned SQL join").await;
        self.handle = None;
        match result {
            Ok(mut completion) => {
                completion.stop.cancel();
                let _ = completion.check_current();
                drop(completion);
            }
            Err(error) => self.record_join_error(error),
        }
        drop(self.finish());
    }

    fn record_join_error(&self, error: JoinError) {
        let identity = {
            let state = self.owner.0.state.lock();
            let attempt = state
                .attempts
                .get(&self.id)
                .expect("registered SQL attempt");
            SqlRestoreIdentity {
                node_id: attempt.identity.node_id.clone(),
                node_order: attempt.identity.node_order,
                task_id: attempt.identity.task_id,
            }
        };
        let origin = cleanup_origin(&identity);
        let error = native_join_error(error, &identity);
        let mut state = self.owner.0.state.lock();
        push_cleanup(&mut state, identity.node_order, self.id, origin, error);
    }

    async fn join(mut self) -> Result<OwnedSqlRestoreCompletion> {
        let result = self.handle.as_mut().expect("owned SQL join").await;
        self.handle = None;
        let attempt = self.finish();
        match result {
            Ok(mut completion) => {
                if attempt.abandoned {
                    completion.stop.cancel();
                }
                let _ = completion.check_current();
                Ok(completion)
            }
            Err(error) => Err(native_join_error(error, &attempt.identity)),
        }
    }
}

impl Drop for SqlRecoveryJoinLoan {
    fn drop(&mut self) {
        if let Some(handle) = self.handle.take() {
            let mut state = self.owner.0.state.lock();
            let attempt = state.attempts.get_mut(&self.id).expect("live SQL attempt");
            attempt.stop.cancel();
            attempt.abandoned = true;
            attempt.loaned = false;
            assert!(attempt.handle.replace(handle).is_none());
            #[cfg(test)]
            self.owner
                .0
                .join_returns
                .fetch_add(1, std::sync::atomic::Ordering::SeqCst);
            drop(state);
            self.owner.0.changed.notify_waiters();
        } else if self.owner.0.state.lock().attempts.contains_key(&self.id) {
            drop(self.finish());
        }
    }
}

fn launch_work(
    work: Work,
    gate: std::sync::mpsc::Receiver<()>,
) -> std::thread::Result<JoinHandle<OwnedSqlRestoreCompletion>> {
    catch_unwind(AssertUnwindSafe(|| {
        tokio::task::spawn_blocking(move || {
            let _ = gate.recv();
            run(work)
        })
    }))
}

fn prepare_work(work: &mut Work) -> Result<PreparedSqlWork> {
    let check = || {
        check_stop(
            &work.context,
            &work.launch_cancel,
            &work.stop,
            &work.home_stop,
        )
    };
    check()?;
    let prepared = match &work.operation {
        SqlWorkOperation::Restore(snapshot) => {
            PreparedSqlWork::Restore(Box::new(work.operator.prepare_restore(snapshot, &check)?))
        }
        SqlWorkOperation::Capture => {
            PreparedSqlWork::Capture(work.operator.prepare_checkpoint_work(&check)?)
        }
    };
    check()?;
    Ok(prepared)
}

fn run(mut work: Work) -> OwnedSqlRestoreCompletion {
    let prepared = match catch_unwind(AssertUnwindSafe(|| prepare_work(&mut work))) {
        Ok(result) => result,
        Err(payload) => Err(panicked(
            work.identity.task_id,
            &work.identity.node_id,
            payload.as_ref(),
        )),
    };
    #[cfg(test)]
    if let Err(CalcFlowError::TaskPanicked { task_id, .. }) = &prepared
        && let Some(home) = work.home.upgrade()
    {
        *home.panic_identity.lock() = Some(*task_id);
    }
    OwnedSqlRestoreCompletion {
        home: work.home,
        id: work.id,
        identity: work.identity,
        operator: work.operator,
        prepared,
        context: work.context,
        launch_cancel: work.launch_cancel,
        stop: work.stop,
        home_stop: work.home_stop,
        _reservation: work.reservation,
    }
}

fn record_cleanup(home: &Home, identity: &SqlRestoreIdentity, id: u64, error: CalcFlowError) {
    let mut state = home.state.lock();
    push_cleanup(
        &mut state,
        identity.node_order,
        id,
        cleanup_origin(identity),
        error,
    );
}

fn cleanup_origin(identity: &SqlRestoreIdentity) -> FailureOrigin {
    identity
        .task_id
        .map_or(FailureOrigin::Preflight, |task_id| FailureOrigin::Task {
            task_id,
            task_name: format!("operator:{}", identity.node_id),
        })
}

fn push_cleanup(
    state: &mut State,
    order: usize,
    id: u64,
    origin: FailureOrigin,
    error: CalcFlowError,
) {
    if state.cleanup.len() < state.capacity {
        state
            .cleanup
            .push((order, id, Arc::new(RuntimeFailure { origin, error })));
    } else {
        state.overflow = state.overflow.saturating_add(1);
    }
}

fn native_join_error(error: JoinError, identity: &SqlRestoreIdentity) -> CalcFlowError {
    if error.is_panic() {
        panicked(
            identity.task_id,
            &identity.node_id,
            error.into_panic().as_ref(),
        )
    } else {
        CalcFlowError::Internal {
            message: format!(
                "SQL recovery node {:?} native join failed: {error}",
                identity.node_id
            ),
        }
    }
}

fn panicked(
    task_id: Option<TaskId>,
    node_id: &str,
    payload: &(dyn std::any::Any + Send),
) -> CalcFlowError {
    match task_id {
        Some(task_id) => CalcFlowError::TaskPanicked {
            task_id: task_id.as_u64(),
            message: panic_message(payload),
        },
        None => CalcFlowError::Internal {
            message: format!(
                "terminal SQL recovery node {node_id:?} panicked: {}",
                panic_message(payload)
            ),
        },
    }
}

fn cancelled(context: &SqlRecoveryContext) -> CalcFlowError {
    CalcFlowError::Cancelled {
        run_id: context.job_id.to_string(),
    }
}

fn check_stop(
    context: &SqlRecoveryContext,
    launch: &CancellationToken,
    attempt: &CancellationToken,
    home: &CancellationToken,
) -> Result<()> {
    if context.cancellation.is_cancelled()
        || context
            .deadline
            .is_some_and(|deadline| Utc::now() >= deadline)
    {
        return Err(cancelled(context));
    }
    if launch.is_cancelled() || attempt.is_cancelled() || home.is_cancelled() {
        return Err(cancelled(context));
    }
    Ok(())
}
