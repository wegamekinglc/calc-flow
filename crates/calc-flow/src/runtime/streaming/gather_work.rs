use std::{
    any::{Any, TypeId},
    marker::PhantomData,
    panic::{AssertUnwindSafe, catch_unwind},
    sync::{
        Arc, Weak,
        atomic::{AtomicBool, Ordering},
    },
    time::Duration,
};

use chrono::{DateTime, Utc};
use datafusion::{
    arrow::array::ArrayRef, common::DataFusionError, execution::memory_pool::MemoryReservation,
};
use parking_lot::{Condvar, Mutex};
use tokio::sync::Notify;

use super::supervisor::TaskFailure;
pub(crate) use super::supervisor::TaskId;
use crate::{CalcFlowError, CancellationToken, Result};

fn launch_registration(registered: process::Registration, workers: usize) -> Result<()> {
    catch_unwind(AssertUnwindSafe(|| registered.launch(workers))).unwrap_or_else(|payload| {
        Err(internal(&format!(
            "native worker launch panic: {}",
            worker::panic_summary(payload.as_ref())
        )))
    })
}

const DIAGNOSTIC_LIMIT: usize = 8;
const DIAGNOSTIC_CAPACITY: usize = DIAGNOSTIC_LIMIT + 1;
const DIAGNOSTIC_NAME_BYTES: usize = 256;
const DIAGNOSTIC_MESSAGE_BYTES: usize = 1_024;
const HOME_CONTROL_BYTES: usize = 16_384;

const _: () = assert!(
    size_of::<GatherHome>()
        + DIAGNOSTIC_CAPACITY * size_of::<TaskFailure>()
        + DIAGNOSTIC_LIMIT
            * (DIAGNOSTIC_NAME_BYTES + DIAGNOSTIC_MESSAGE_BYTES + "ASOF gather: ".len())
        + 256
        <= HOME_CONTROL_BYTES
);

#[cfg(test)]
pub(crate) mod admission_probe;
mod columns;
mod parallel;
pub(crate) use parallel::ParallelCpuWork;
mod process;
mod row_gather;
#[cfg(test)]
pub(crate) use process::TestService;
pub(crate) use row_gather::RowGather;
#[cfg(test)]
mod tests;
mod worker;

pub(crate) trait GatherPlan: Send + Sync {
    fn column_count(&self) -> usize;
    fn gather(&self, ordinal: usize, stop: &GatherStop) -> Result<ArrayRef>;

    fn parallelism(&self) -> usize {
        self.column_count().clamp(1, 8)
    }

    fn row_gather(&self) -> Result<Option<RowGather>> {
        Ok(None)
    }

    fn shared_column(&self, _ordinal: usize) -> Result<Option<ArrayRef>> {
        Ok(None)
    }

    fn gather_range(
        &self,
        _ordinal: usize,
        _range: std::ops::Range<usize>,
        _stop: &GatherStop,
    ) -> Result<ArrayRef> {
        Err(internal("row range gather unsupported"))
    }
}

pub(crate) trait OwnedCpuWork: Send + 'static {
    type Output: Send + 'static;

    fn control_bytes(&self) -> Result<usize> {
        Ok(0)
    }

    fn run(self, stop: &GatherStop) -> Result<Self::Output>;
}

#[derive(Debug)]
pub(crate) enum AdmissionFailure {
    Budget {
        stage: &'static str,
        source: DataFusionError,
    },
    Runtime(CalcFlowError),
}

type AdmissionResult<T> = std::result::Result<T, AdmissionFailure>;

impl From<CalcFlowError> for AdmissionFailure {
    fn from(error: CalcFlowError) -> Self {
        Self::Runtime(error)
    }
}

impl From<AdmissionFailure> for CalcFlowError {
    fn from(failure: AdmissionFailure) -> Self {
        match failure {
            AdmissionFailure::Budget { stage, source } => {
                internal(&format!("{stage} credit admission: {source}"))
            }
            AdmissionFailure::Runtime(error) => error,
        }
    }
}

type ErasedOutput = Box<dyn Any + Send>;

trait ErasedWork: Send {
    fn run(self: Box<Self>, stop: &GatherStop) -> Result<ErasedOutput>;
}

trait WorkPackage: Send + 'static {
    type Output: Send + 'static;

    fn control_bytes(&self) -> Result<usize>;
    fn into_work(self) -> Result<ReadyWork>;
}

struct WorkAdapter<W>(W);

impl<W: OwnedCpuWork> WorkPackage for WorkAdapter<W> {
    type Output = W::Output;

    fn control_bytes(&self) -> Result<usize> {
        self.0.control_bytes()
    }

    fn into_work(self) -> Result<ReadyWork> {
        Ok(ReadyWork::single(Box::new(self)))
    }
}

impl<W: OwnedCpuWork> ErasedWork for WorkAdapter<W> {
    fn run(self: Box<Self>, stop: &GatherStop) -> Result<ErasedOutput> {
        let Self(work) = *self;
        Ok(Box::new(work.run(stop)?))
    }
}

struct ColumnsWork(Arc<dyn GatherPlan>);

impl WorkPackage for ColumnsWork {
    type Output = Vec<ArrayRef>;

    fn control_bytes(&self) -> Result<usize> {
        columns::Columns::control_bytes(self.0.as_ref())
    }

    fn into_work(self) -> Result<ReadyWork> {
        Ok(ReadyWork::Columns(columns::Columns::new(self.0)?))
    }
}

impl OwnedCpuWork for ColumnsWork {
    type Output = Vec<ArrayRef>;

    fn control_bytes(&self) -> Result<usize> {
        <Self as WorkPackage>::control_bytes(self)
    }

    fn run(self, stop: &GatherStop) -> Result<Vec<ArrayRef>> {
        let mut columns = Vec::with_capacity(self.0.column_count());
        for ordinal in 0..self.0.column_count() {
            stop.check()?;
            columns.push(self.0.gather(ordinal, stop)?);
        }
        Ok(columns)
    }
}

enum ReadyWork {
    Single {
        work: Option<Box<dyn ErasedWork>>,
        outcome: Option<Result<ErasedOutput>>,
        running: bool,
    },
    Columns(columns::Columns),
    Parallel(parallel::Units),
}

impl ReadyWork {
    fn single(work: Box<dyn ErasedWork>) -> Self {
        Self::Single {
            work: Some(work),
            outcome: None,
            running: false,
        }
    }

    fn units(&self) -> usize {
        match self {
            Self::Single { .. } => 1,
            Self::Columns(columns) => columns.count(),
            Self::Parallel(units) => units.count(),
        }
    }

    fn workers(&self) -> usize {
        match self {
            Self::Single { .. } => 1,
            Self::Columns(columns) => columns.workers(),
            Self::Parallel(units) => units.workers(),
        }
    }

    fn active(&self) -> usize {
        match self {
            Self::Single { running, .. } => usize::from(*running),
            Self::Columns(columns) => columns.active(),
            Self::Parallel(units) => units.active(),
        }
    }

    fn claim(&mut self, ordinal: usize) -> Option<ClaimedWork> {
        match self {
            Self::Single { work, running, .. } if ordinal == 0 => {
                let work = work.take()?;
                *running = true;
                Some(ClaimedWork::Single(work))
            }
            Self::Single { .. } => None,
            Self::Columns(columns) => columns.claim(ordinal),
            Self::Parallel(units) => units.claim(ordinal),
        }
    }

    fn complete(&mut self, ordinal: usize, value: Result<ErasedOutput>) -> bool {
        match self {
            Self::Single {
                outcome, running, ..
            } => {
                *outcome = Some(value);
                *running = false;
                true
            }
            Self::Columns(columns) => columns.complete(ordinal, value),
            Self::Parallel(units) => units.complete(ordinal, value),
        }
    }

    fn cancel_pending(&mut self, run_id: &str) {
        match self {
            Self::Columns(columns) => columns.cancel_pending(run_id),
            Self::Parallel(units) => units.cancel_pending(run_id),
            Self::Single { .. } => {}
        }
    }

    fn finish(
        self,
        home: &GatherHome,
        operator: &GatherOperatorId,
        stop: &GatherStop,
    ) -> Result<ErasedOutput> {
        match self {
            Self::Single { outcome, .. } => {
                outcome.ok_or_else(|| internal("missing unit outcome"))?
            }
            Self::Columns(columns) => columns.finish(home, operator, stop),
            Self::Parallel(units) => units.finish(home, operator),
        }
    }
}

enum ClaimedWork {
    Single(Box<dyn ErasedWork>),
    Column {
        plan: Arc<dyn GatherPlan>,
        ordinal: usize,
        range: Option<std::ops::Range<usize>>,
        siblings: Arc<AtomicBool>,
    },
}

impl ClaimedWork {
    fn run(self, stop: &GatherStop) -> Result<ErasedOutput> {
        match self {
            Self::Single(work) => work.run(stop),
            Self::Column {
                plan,
                ordinal,
                range,
                siblings,
            } => {
                stop.check()?;
                if siblings.load(Ordering::Acquire) {
                    return Err(cancelled(&stop.run_id));
                }
                let value = match range {
                    Some(range) => plan.gather_range(ordinal, range, stop),
                    None => plan.gather(ordinal, stop),
                };
                drop(plan);
                Ok(Box::new(value?))
            }
        }
    }
}

#[derive(Clone)]
pub(crate) struct GatherOperatorId {
    token: Arc<()>,
    name: Arc<str>,
    task: Option<TaskId>,
}

impl GatherOperatorId {
    pub(crate) fn new(name: Arc<str>) -> Self {
        Self {
            token: Arc::new(()),
            name,
            task: None,
        }
    }

    pub(crate) fn with_task(mut self, task: Option<TaskId>) -> Self {
        self.task = task;
        self
    }
}

#[derive(Clone)]
pub(crate) struct GatherStop {
    run_id: Arc<str>,
    cancellation: CancellationToken,
    deadline: Option<DateTime<Utc>>,
    abandoned: Arc<AtomicBool>,
}

impl GatherStop {
    pub(crate) fn from_job(job: &super::StreamJobContext) -> Self {
        Self {
            run_id: job.job_id().to_string().into(),
            cancellation: job.cancellation().clone(),
            deadline: job.deadline().copied(),
            abandoned: Arc::new(AtomicBool::new(false)),
        }
    }

    pub(crate) fn check(&self) -> Result<()> {
        if self.abandoned.load(Ordering::Acquire)
            || self.cancellation.is_cancelled()
            || self.deadline.is_some_and(|deadline| Utc::now() >= deadline)
        {
            return Err(cancelled(&self.run_id));
        }
        Ok(())
    }

    fn abandon(&self) {
        self.abandoned.store(true, Ordering::Release);
    }

    async fn wait(&self) {
        tokio::select! {
            () = self.cancellation.cancelled() => {},
            () = deadline_wait(self.deadline) => {},
        }
    }
}

async fn deadline_wait(deadline: Option<DateTime<Utc>>) {
    match deadline {
        Some(deadline) => {
            let delay = (deadline - Utc::now()).to_std().unwrap_or(Duration::ZERO);
            tokio::time::sleep(delay).await;
        }
        None => std::future::pending().await,
    }
}

#[derive(Clone)]
pub(crate) struct JobGatherOwner(Arc<ExternalLease>);

struct ExternalLease {
    home: Arc<GatherHome>,
}

impl Drop for ExternalLease {
    fn drop(&mut self) {
        self.home.close();
    }
}

pub(crate) struct GatherClient {
    home: Weak<GatherHome>,
    operator: GatherOperatorId,
}

pub(crate) struct GatherScope {
    home: Arc<GatherHome>,
    id: u64,
    operator: GatherOperatorId,
}

pub(crate) type GatherTicket = WorkTicket<Vec<ArrayRef>>;

pub(crate) struct WorkTicket<T: Send + 'static> {
    home: Arc<GatherHome>,
    generation: u64,
    attempt: u64,
    active: bool,
    output: PhantomData<fn() -> T>,
}

pub(crate) struct WorkOutput<T> {
    pub(crate) value: T,
    pub(crate) credit: MemoryReservation,
}

struct WorkerIdentity {
    active: Option<(u64, u64, usize)>,
    token: Arc<()>,
    task_id: TaskId,
    task_name: String,
}

struct GatherHome {
    run_id: Arc<str>,
    state: Mutex<HomeState>,
    changed: Notify,
    native_changed: Condvar,
}

struct HomeState {
    open: bool,
    generation: u64,
    next_scope: u64,
    next_attempt: u64,
    pool: Option<Weak<process::NativePool>>,
    phase: PoolPhase,
    queue: std::collections::VecDeque<Request>,
    slot: Slot,
    control: Option<MemoryReservation>,
    failures: Vec<TaskFailure>,
    omitted_failures: Option<u64>,
    omitted_task: Option<TaskId>,
    service: Option<Weak<process::ProcessService>>,
    #[cfg(test)]
    admission_probe: Option<Arc<admission_probe::AdmissionProbe>>,
}

#[derive(Clone, Copy, Eq, PartialEq)]
enum PoolPhase {
    Absent,
    Initializing,
    Ready,
    Retiring,
}

enum Slot {
    Empty,
    Active(Attempt),
    Parked(Attempt),
    Dropping(u64),
}

struct Attempt {
    id: u64,
    generation: u64,
    scope: u64,
    operator: GatherOperatorId,
    work: Option<ReadyWork>,
    next_unit: usize,
    output_type: TypeId,
    outcome: Option<Result<ErasedOutput>>,
    stop: GatherStop,
    running: bool,
    settled: bool,
    abandoned: bool,
    credit: MemoryReservation,
}

#[derive(Clone, Copy)]
struct Request {
    generation: u64,
    attempt: u64,
    ordinal: usize,
}

struct SlotRelease<'a> {
    home: &'a GatherHome,
    attempt: u64,
}

impl Drop for SlotRelease<'_> {
    fn drop(&mut self) {
        self.home.release_slot(self.attempt);
    }
}

struct SubmissionGuard {
    home: Arc<GatherHome>,
    attempt: u64,
    armed: bool,
}

impl Drop for SubmissionGuard {
    fn drop(&mut self) {
        if self.armed {
            self.home.abandon(self.attempt);
        }
    }
}

impl JobGatherOwner {
    pub(crate) fn new(run_id: Arc<str>) -> Self {
        Self(Arc::new(ExternalLease {
            home: Arc::new(GatherHome::new(run_id)),
        }))
    }

    pub(crate) fn client(&self, operator: GatherOperatorId) -> GatherClient {
        GatherClient {
            home: Arc::downgrade(&self.0.home),
            operator,
        }
    }

    pub(crate) fn close_admission(&self) {
        self.0.home.close();
    }

    pub(crate) async fn close_and_drain(&self) -> Vec<TaskFailure> {
        self.close_admission();
        self.0.home.wait_closed().await;
        self.0.home.state.lock().take_failures()
    }
}

impl GatherClient {
    pub(crate) fn scope(&self) -> Result<GatherScope> {
        let home = self
            .home
            .upgrade()
            .ok_or_else(|| cancelled("ASOF gather"))?;
        let id = {
            let mut state = home.state.lock();
            if !state.open {
                return Err(cancelled(&home.run_id));
            }
            state.next_scope = increment(state.next_scope, "scope")?;
            state.next_scope
        };
        Ok(GatherScope {
            home,
            id,
            operator: self.operator.clone(),
        })
    }
}

impl Drop for GatherScope {
    fn drop(&mut self) {
        let attempt = {
            let state = self.home.state.lock();
            match &state.slot {
                Slot::Active(attempt) | Slot::Parked(attempt)
                    if attempt.scope == self.id
                        && Arc::ptr_eq(&attempt.operator.token, &self.operator.token) =>
                {
                    Some(attempt.id)
                }
                _ => None,
            }
        };
        if let Some(attempt) = attempt {
            self.home.abandon(attempt);
        }
    }
}

impl GatherScope {
    pub(crate) async fn submit_parallel_work<W: ParallelCpuWork>(
        &self,
        work: Arc<W>,
        credit: MemoryReservation,
        stop: GatherStop,
    ) -> AdmissionResult<WorkTicket<Vec<W::Output>>> {
        self.submit_package(parallel::Package(work), credit, stop)
            .await
    }

    pub(crate) async fn submit(
        &self,
        plan: Arc<dyn GatherPlan>,
        credit: MemoryReservation,
        stop: GatherStop,
    ) -> AdmissionResult<GatherTicket> {
        if plan.column_count() <= 1 || plan.parallelism() <= 1 {
            self.submit_work(ColumnsWork(plan), credit, stop).await
        } else {
            self.submit_package(ColumnsWork(plan), credit, stop).await
        }
    }

    pub(crate) async fn submit_work<W: OwnedCpuWork>(
        &self,
        work: W,
        credit: MemoryReservation,
        stop: GatherStop,
    ) -> AdmissionResult<WorkTicket<W::Output>> {
        self.submit_package(WorkAdapter(work), credit, stop).await
    }

    async fn submit_package<P: WorkPackage>(
        &self,
        work: P,
        credit: MemoryReservation,
        stop: GatherStop,
    ) -> AdmissionResult<WorkTicket<P::Output>> {
        let attempt = self
            .home
            .install_attempt(self, work, credit, stop.clone())
            .await?;
        let mut guard = SubmissionGuard {
            home: self.home.clone(),
            attempt,
            armed: true,
        };
        let workers = self.home.attempt_workers(attempt)?;
        let generation = self.home.ensure_pool(&stop, workers).await?;
        self.home.dispatch(attempt, generation)?;
        guard.armed = false;
        Ok(WorkTicket {
            home: self.home.clone(),
            generation,
            attempt,
            active: true,
            output: PhantomData,
        })
    }
}

impl<T: Send + 'static> Drop for WorkTicket<T> {
    fn drop(&mut self) {
        if self.active {
            self.home.abandon(self.attempt);
        }
    }
}

impl<T: Send + 'static> WorkTicket<T> {
    pub(crate) async fn finish(mut self) -> Result<WorkOutput<T>> {
        self.home
            .wait_attempt(self.generation, self.attempt)
            .await?;
        let record = self.home.take_attempt(self.generation, self.attempt)?;
        self.active = false;
        self.home.finish_owned(record)
    }
}

fn finish_record<T: Send + 'static>(mut record: Attempt) -> Result<WorkOutput<T>> {
    if record.output_type != TypeId::of::<T>() {
        return Err(internal("typed output identity mismatch"));
    }
    let value = record
        .outcome
        .take()
        .ok_or_else(|| internal("missing settled output"))??;
    record.stop.check()?;
    let value = *value
        .downcast::<T>()
        .map_err(|_| internal("typed output identity mismatch"))?;
    let credit = record.credit.split(record.credit.size());
    Ok(WorkOutput { value, credit })
}

impl GatherHome {
    fn new(run_id: Arc<str>) -> Self {
        Self {
            run_id,
            state: Mutex::new(HomeState {
                open: true,
                generation: 0,
                next_scope: 0,
                next_attempt: 0,
                pool: None,
                phase: PoolPhase::Absent,
                queue: std::collections::VecDeque::new(),
                slot: Slot::Empty,
                control: None,
                failures: Vec::new(),
                omitted_failures: Some(0),
                omitted_task: None,
                service: None,
                #[cfg(test)]
                admission_probe: None,
            }),
            changed: Notify::new(),
            native_changed: Condvar::new(),
        }
    }

    async fn install_attempt<W: WorkPackage>(
        &self,
        scope: &GatherScope,
        work: W,
        credit: MemoryReservation,
        stop: GatherStop,
    ) -> AdmissionResult<u64> {
        let mut work = Some(work);
        loop {
            let changed = self.changed.notified();
            tokio::pin!(changed);
            changed.as_mut().enable();
            stop.check()?;
            if let Some(id) = self.try_install(scope, &mut work, &credit, stop.clone())? {
                return Ok(id);
            }
            tokio::select! {
                () = changed => {},
                () = stop.wait() => return Err(cancelled(&self.run_id).into()),
            }
        }
    }

    fn try_install<W: WorkPackage>(
        &self,
        scope: &GatherScope,
        work: &mut Option<W>,
        credit: &MemoryReservation,
        stop: GatherStop,
    ) -> AdmissionResult<Option<u64>> {
        let mut state = self.state.lock();
        if !state.open {
            return Err(cancelled(&self.run_id).into());
        }
        self.retire_if_pressured(&mut state);
        if state.phase == PoolPhase::Retiring || !matches!(state.slot, Slot::Empty) {
            return Ok(None);
        }
        let next_attempt = increment(state.next_attempt, "attempt")?;
        let funded = fund_work(
            work,
            credit,
            scope.operator.name.len(),
            state.control.is_none(),
            #[cfg(test)]
            state.admission_probe.as_ref(),
            #[cfg(test)]
            &scope.operator,
        )?;
        if let Some(home_credit) = funded.home_credit {
            state.control = Some(home_credit);
            state.failures = Vec::with_capacity(DIAGNOSTIC_CAPACITY);
        }
        state.next_attempt = next_attempt;
        let id = state.next_attempt;
        state.slot = Slot::Active(Attempt {
            id,
            generation: 0,
            scope: scope.id,
            operator: scope.operator.clone(),
            work: Some(funded.work),
            next_unit: 0,
            output_type: TypeId::of::<W::Output>(),
            outcome: None,
            stop,
            running: false,
            settled: false,
            abandoned: false,
            credit: funded.credit,
        });
        Ok(Some(id))
    }

    fn attempt_workers(&self, id: u64) -> Result<usize> {
        let state = self.state.lock();
        let Slot::Active(record) = &state.slot else {
            return Err(internal("missing initializing attempt"));
        };
        if record.id != id {
            return Err(internal("stale initializing attempt"));
        }
        record
            .work
            .as_ref()
            .map(ReadyWork::workers)
            .ok_or_else(|| internal("missing initializing work"))
    }

    async fn ensure_pool(
        self: &Arc<Self>,
        stop: &GatherStop,
        workers: usize,
    ) -> AdmissionResult<u64> {
        self.wait_retirement(stop).await?;
        let service = self.execution_service()?;
        let registered = match self.ready_pool()? {
            Some(pool) => service.grow(pool)?,
            None => service.register(self.clone(), stop).await?,
        };
        let generation = registered.generation;
        let launched = launch_registration(registered, workers);
        if let Err(error) = launched {
            self.wait_generation_cleanup(generation).await;
            return Err(error.into());
        }
        Ok(generation)
    }

    async fn wait_generation_cleanup(&self, generation: u64) {
        loop {
            let changed = self.changed.notified();
            tokio::pin!(changed);
            changed.as_mut().enable();
            {
                let state = self.state.lock();
                if state.generation != generation || state.phase == PoolPhase::Absent {
                    return;
                }
            }
            changed.await;
        }
    }

    async fn wait_retirement(&self, stop: &GatherStop) -> Result<()> {
        loop {
            let changed = self.changed.notified();
            tokio::pin!(changed);
            changed.as_mut().enable();
            stop.check()?;
            if !self.retiring_generation()? {
                return Ok(());
            }
            tokio::select! {
                () = changed => {},
                () = stop.wait() => return Err(cancelled(&self.run_id)),
            }
        }
    }

    fn retiring_generation(&self) -> Result<bool> {
        let state = self.state.lock();
        if !state.open {
            return Err(cancelled(&self.run_id));
        }
        Ok(state.phase == PoolPhase::Retiring)
    }

    fn execution_service(&self) -> Result<Arc<process::ProcessService>> {
        if let Some(service) = &self.state.lock().service {
            return service
                .upgrade()
                .ok_or_else(|| internal("isolated process service closed"));
        }
        process::service()
    }

    fn ready_pool(&self) -> Result<Option<Arc<process::NativePool>>> {
        let state = self.state.lock();
        if !state.open {
            return Err(cancelled(&self.run_id));
        }
        Ok(state.pool.as_ref().and_then(Weak::upgrade))
    }

    fn dispatch(&self, attempt: u64, generation: u64) -> Result<()> {
        let mut state = self.state.lock();
        if !state.open || matches!(state.phase, PoolPhase::Initializing | PoolPhase::Retiring) {
            return Err(cancelled(&self.run_id));
        }
        let (Slot::Active(record) | Slot::Parked(record)) = &mut state.slot else {
            return Err(internal("lost registered attempt"));
        };
        if record.id != attempt || record.abandoned {
            return Err(cancelled(&self.run_id));
        }
        record.generation = generation;
        let units = record.work.as_ref().map_or(0, ReadyWork::units).min(8);
        record.next_unit = units;
        if !state.queue.is_empty() {
            return Err(internal("paid request ring occupied"));
        }
        for ordinal in 0..units {
            state.queue.push_back(Request {
                generation,
                attempt,
                ordinal,
            });
        }
        self.native_changed.notify_all();
        Ok(())
    }

    fn close(&self) {
        let attempt = {
            let mut state = self.state.lock();
            state.open = false;
            match &state.slot {
                Slot::Active(record) | Slot::Parked(record) => Some(record.id),
                _ => None,
            }
        };
        if let Some(attempt) = attempt {
            self.abandon(attempt);
        }
        self.native_changed.notify_all();
        self.changed.notify_waiters();
        process::wake_existing();
    }

    fn abandon(&self, id: u64) {
        let abandoned = {
            let mut state = self.state.lock();
            let (Slot::Active(record) | Slot::Parked(record)) = &mut state.slot else {
                return;
            };
            if record.id != id {
                return;
            }
            record.abandoned = true;
            record.stop.abandon();
            if let Some(work) = record.work.as_mut() {
                work.cancel_pending(&self.run_id);
                record.next_unit = work.units();
            }
            let running = record.running;
            state.queue.retain(|request| request.attempt != id);
            if running {
                None
            } else {
                Some(detach(&mut state, id))
            }
        };
        if let Some(record) = abandoned {
            self.destroy_attempt(record);
        }
        self.native_changed.notify_all();
        self.changed.notify_waiters();
    }

    fn destroy_attempt(&self, record: Attempt) {
        let _release = SlotRelease {
            home: self,
            attempt: record.id,
        };
        let operator = record.operator.clone();
        self.retain_failure(&record);
        if let Err(payload) = catch_unwind(AssertUnwindSafe(|| drop(record))) {
            self.retain_cleanup_panic(&operator, payload.as_ref());
        }
    }

    fn finish_owned<T: Send + 'static>(&self, record: Attempt) -> Result<WorkOutput<T>> {
        let _release = SlotRelease {
            home: self,
            attempt: record.id,
        };
        let operator = record.operator.clone();
        catch_unwind(AssertUnwindSafe(|| finish_record(record)))
            .map_err(|payload| self.retain_cleanup_panic(&operator, payload.as_ref()))?
    }

    fn release_slot(&self, id: u64) {
        let mut state = self.state.lock();
        if matches!(state.slot, Slot::Dropping(current) if current == id) {
            state.slot = Slot::Empty;
            self.retire_if_pressured(&mut state);
        }
        self.changed.notify_waiters();
    }

    async fn wait_attempt(&self, generation: u64, id: u64) -> Result<()> {
        loop {
            let changed = self.changed.notified();
            tokio::pin!(changed);
            changed.as_mut().enable();
            {
                let state = self.state.lock();
                let (Slot::Active(record) | Slot::Parked(record)) = &state.slot else {
                    return Err(cancelled(&self.run_id));
                };
                if record.id != id || record.generation != generation {
                    return Err(internal("stale ticket"));
                }
                if record.settled {
                    return Ok(());
                }
            }
            changed.await;
        }
    }

    fn take_attempt(&self, generation: u64, id: u64) -> Result<Attempt> {
        let mut state = self.state.lock();
        if !matches!(&state.slot, Slot::Active(record) | Slot::Parked(record) if record.id == id && record.generation == generation && record.settled)
        {
            return Err(internal("unsettled or stale ticket"));
        }
        Ok(detach(&mut state, id))
    }

    async fn wait_closed(&self) {
        loop {
            let changed = self.changed.notified();
            tokio::pin!(changed);
            changed.as_mut().enable();
            {
                let state = self.state.lock();
                if state.phase == PoolPhase::Absent && matches!(state.slot, Slot::Empty) {
                    return;
                }
            }
            changed.await;
        }
    }
}

fn detach(state: &mut HomeState, id: u64) -> Attempt {
    state.queue.retain(|request| request.attempt != id);
    let (Slot::Active(record) | Slot::Parked(record)) =
        std::mem::replace(&mut state.slot, Slot::Dropping(id))
    else {
        unreachable!("detach a registered attempt")
    };
    record
}

fn increment(value: u64, kind: &str) -> Result<u64> {
    value
        .checked_add(1)
        .ok_or_else(|| internal(&format!("{kind} identity overflow")))
}

fn cancelled(run_id: &str) -> CalcFlowError {
    CalcFlowError::Cancelled {
        run_id: run_id.into(),
    }
}

fn internal(message: &str) -> CalcFlowError {
    CalcFlowError::Internal {
        message: format!("ASOF gather: {message}"),
    }
}

#[cfg(test)]
impl JobGatherOwner {
    pub(crate) fn funding(&self) -> (usize, usize, usize) {
        let state = self.0.home.state.lock();
        let home = state.control.as_ref().map_or(0, MemoryReservation::size);
        let generation = state
            .pool
            .as_ref()
            .and_then(Weak::upgrade)
            .map_or(0, |pool| pool.credit_size());
        let attempt = match &state.slot {
            Slot::Active(record) | Slot::Parked(record) => record.credit.size(),
            _ => 0,
        };
        (home, generation, attempt)
    }
}

struct FundedWork {
    work: ReadyWork,
    home_credit: Option<MemoryReservation>,
    credit: MemoryReservation,
}

fn fund_work<W: WorkPackage>(
    work: &mut Option<W>,
    credit: &MemoryReservation,
    name_bytes: usize,
    needs_control: bool,
    #[cfg(test)] probe: Option<&Arc<admission_probe::AdmissionProbe>>,
    #[cfg(test)] operator: &GatherOperatorId,
) -> AdmissionResult<FundedWork> {
    let (home_fee, total_fee) = pending_work_fees(work.as_ref(), name_bytes, needs_control)?;
    #[cfg(test)]
    let _pressure = admission_probe::before(
        probe,
        admission_probe::AdmissionStage::Attempt,
        total_fee,
        operator,
    )?;
    credit
        .try_grow(total_fee)
        .map_err(|source| AdmissionFailure::Budget {
            stage: "attempt",
            source,
        })?;
    let owned = work.take().ok_or_else(|| internal("missing owned work"))?;
    let work = owned.into_work()?;
    let home_credit = (home_fee > 0).then(|| credit.split(home_fee));
    Ok(FundedWork {
        work,
        home_credit,
        credit: credit.split(credit.size()),
    })
}

fn pending_work_fees<W: WorkPackage>(
    work: Option<&W>,
    name_bytes: usize,
    needs_control: bool,
) -> Result<(usize, usize)> {
    let work = work.ok_or_else(|| internal("missing owned work"))?;
    execution_fees::<W>(work.control_bytes()?, name_bytes, needs_control)
}

fn home_fee(needs_control: bool) -> usize {
    if needs_control { HOME_CONTROL_BYTES } else { 0 }
}

fn execution_fees<W: WorkPackage>(
    control_bytes: usize,
    name_bytes: usize,
    needs_control: bool,
) -> Result<(usize, usize)> {
    let home = home_fee(needs_control);
    let total = control_bytes
        .checked_add(size_of::<W>())
        .and_then(|bytes| bytes.checked_add(size_of::<W::Output>()))
        .and_then(|bytes| bytes.checked_add(8_192))
        .and_then(|bytes| bytes.checked_add(name_bytes))
        .and_then(|bytes| bytes.checked_add(64))
        .and_then(|bytes| bytes.checked_add(home))
        .ok_or_else(|| internal("complete execution credit overflow"))?;
    Ok((home, total))
}

#[cfg(test)]
impl<T: Send + 'static> WorkTicket<T> {
    pub(crate) fn generation(&self) -> u64 {
        self.generation
    }

    pub(crate) async fn wait_settled(&self) -> Result<()> {
        self.home.wait_attempt(self.generation, self.attempt).await
    }
}
