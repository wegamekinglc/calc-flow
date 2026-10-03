use std::{
    future::poll_fn,
    sync::{
        Arc, Weak,
        atomic::{AtomicUsize, Ordering},
    },
    task::Poll,
    thread::{self, JoinHandle},
    time::Duration,
};

use datafusion::execution::memory_pool::MemoryReservation;
use parking_lot::{Condvar, Mutex};
use tokio::sync::{OwnedSemaphorePermit, Semaphore};

use super::{
    AdmissionFailure, AdmissionResult, GatherHome, GatherStop, PoolPhase, Slot, increment, internal,
};
use crate::Result;

const STACK_BYTES: usize = 512 * 1_024;
const INFRASTRUCTURE_LIMIT: usize = 32 * 1_024 * 1_024;
static INFRASTRUCTURE_BYTES: AtomicUsize = AtomicUsize::new(0);
static PROCESS: Mutex<Option<ProcessControl>> = Mutex::new(None);

struct ProcessControl {
    service: Arc<ProcessService>,
    _joiner: JoinHandle<()>,
    _credit: InfrastructureCredit,
}

pub(super) struct ProcessService {
    slots: Mutex<[Option<PoolRecord>; 256]>,
    wake: Condvar,
    workers: Arc<Semaphore>,
    registrations: Arc<Semaphore>,
    pressure: AtomicUsize,
    retire_cursor: AtomicUsize,
    #[cfg(test)]
    shutdown: std::sync::atomic::AtomicBool,
    #[cfg(test)]
    joined_workers: AtomicUsize,
    #[cfg(test)]
    native_exit_panic: std::sync::atomic::AtomicBool,
    #[cfg(test)]
    invalid_next_worker_name: std::sync::atomic::AtomicBool,
}

struct PoolRecord {
    pool: Arc<NativePool>,
    handles: [Option<WorkerHandle>; 8],
    _registration: OwnedSemaphorePermit,
}

struct WorkerHandle {
    handle: JoinHandle<()>,
    identity: Arc<Mutex<Option<super::WorkerIdentity>>>,
    permit: OwnedSemaphorePermit,
    credit: InfrastructureCredit,
}

pub(super) struct NativePool {
    pub(super) generation: u64,
    pub(super) home: Arc<GatherHome>,
    service: Weak<ProcessService>,
    _credit: MemoryReservation,
}

pub(super) struct Registration {
    service: Arc<ProcessService>,
    pool: Arc<NativePool>,
    slot: usize,
    permit: Option<OwnedSemaphorePermit>,
    armed: bool,
    pub(super) generation: u64,
}

struct InfrastructureCredit {
    bytes: usize,
}

impl InfrastructureCredit {
    fn reserve(bytes: usize) -> Result<Self> {
        INFRASTRUCTURE_BYTES
            .fetch_update(Ordering::AcqRel, Ordering::Acquire, |current| {
                current
                    .checked_add(bytes)
                    .filter(|total| *total <= INFRASTRUCTURE_LIMIT)
            })
            .map_err(|_| internal("bounded process infrastructure exhausted"))?;
        Ok(Self { bytes })
    }
}

impl Drop for InfrastructureCredit {
    fn drop(&mut self) {
        INFRASTRUCTURE_BYTES.fetch_sub(self.bytes, Ordering::AcqRel);
    }
}

pub(super) fn service() -> Result<Arc<ProcessService>> {
    let mut process = PROCESS.lock();
    if let Some(control) = &*process {
        return Ok(control.service.clone());
    }
    let workers = thread::available_parallelism()
        .map_or(1, usize::from)
        .min(32);
    let control = start_service(workers, 256)?;
    let service = control.service.clone();
    *process = Some(control);
    Ok(service)
}

fn start_service(workers: usize, registrations: usize) -> Result<ProcessControl> {
    let credit = InfrastructureCredit::reserve(STACK_BYTES + 1_048_576)?;
    let service = Arc::new(ProcessService {
        slots: Mutex::new(std::array::from_fn(|_| None)),
        wake: Condvar::new(),
        workers: Arc::new(Semaphore::new(workers)),
        registrations: Arc::new(Semaphore::new(registrations)),
        pressure: AtomicUsize::new(0),
        retire_cursor: AtomicUsize::new(0),
        #[cfg(test)]
        shutdown: std::sync::atomic::AtomicBool::new(false),
        #[cfg(test)]
        joined_workers: AtomicUsize::new(0),
        #[cfg(test)]
        native_exit_panic: std::sync::atomic::AtomicBool::new(false),
        #[cfg(test)]
        invalid_next_worker_name: std::sync::atomic::AtomicBool::new(false),
    });
    let registry = service.clone();
    let joiner = thread::Builder::new()
        .name("calc-flow-gather-joiner".into())
        .stack_size(STACK_BYTES)
        .spawn(move || registry.join_loop())
        .map_err(|error| internal(&format!("native joiner launch: {error}")))?;
    Ok(ProcessControl {
        service,
        _joiner: joiner,
        _credit: credit,
    })
}

pub(super) fn wake_existing() {
    if let Some(control) = &*PROCESS.lock() {
        control.service.wake.notify_all();
    }
}

impl ProcessService {
    pub(super) async fn register(
        self: &Arc<Self>,
        home: Arc<GatherHome>,
        stop: &GatherStop,
    ) -> AdmissionResult<Registration> {
        let registration = acquire(self, self.registrations.clone(), stop).await?;
        let permit = acquire(self, self.workers.clone(), stop).await?;
        stop.check()?;
        let pool = Arc::new(self.prepare_pool(home)?);
        let slot = {
            let mut slots = self.slots.lock();
            let slot = slots
                .iter()
                .position(Option::is_none)
                .ok_or_else(|| internal("paid registry exhausted"))?;
            slots[slot] = Some(PoolRecord {
                pool: pool.clone(),
                handles: std::array::from_fn(|_| None),
                _registration: registration,
            });
            slot
        };
        let generation = pool.generation;
        Ok(Registration {
            service: self.clone(),
            pool,
            slot,
            permit: Some(permit),
            armed: true,
            generation,
        })
    }

    pub(super) fn grow(self: &Arc<Self>, pool: Arc<NativePool>) -> Result<Registration> {
        let slot = self
            .slots
            .lock()
            .iter()
            .position(|record| {
                record
                    .as_ref()
                    .is_some_and(|record| Arc::ptr_eq(&record.pool, &pool))
            })
            .ok_or_else(|| internal("lost reusable native pool"))?;
        let generation = pool.generation;
        Ok(Registration {
            service: self.clone(),
            pool,
            slot,
            permit: None,
            armed: true,
            generation,
        })
    }

    fn prepare_pool(self: &Arc<Self>, home: Arc<GatherHome>) -> AdmissionResult<NativePool> {
        let (generation, credit) = {
            let mut state = home.state.lock();
            if !state.open {
                return Err(super::cancelled(&home.run_id).into());
            }
            let Slot::Active(attempt) = &state.slot else {
                return Err(internal("missing initializing attempt").into());
            };
            #[cfg(test)]
            let _pressure = super::admission_probe::before(
                state.admission_probe.as_ref(),
                super::admission_probe::AdmissionStage::Generation,
                16_384,
                &attempt.operator,
            )?;
            attempt
                .credit
                .try_grow(16_384)
                .map_err(|source| AdmissionFailure::Budget {
                    stage: "generation",
                    source,
                })?;
            let credit = attempt.credit.split(16_384);
            state.generation = increment(state.generation, "pool generation")?;
            state.phase = PoolPhase::Initializing;
            state.service = Some(Arc::downgrade(self));
            state.queue = std::collections::VecDeque::with_capacity(8);
            (state.generation, credit)
        };
        Ok(NativePool {
            generation,
            home,
            service: Arc::downgrade(self),
            _credit: credit,
        })
    }

    fn join_loop(&self) {
        loop {
            self.retire_idle();
            for index in 0..256 {
                self.join_finished(index);
            }
            let mut slots = self.slots.lock();
            #[cfg(test)]
            if self.shutdown.load(Ordering::Acquire) && slots.iter().all(Option::is_none) {
                return;
            }
            self.wake.wait_for(&mut slots, Duration::from_millis(10));
        }
    }

    fn retire_idle(&self) {
        if self.pressure.load(Ordering::Acquire) == 0 {
            return;
        }
        let start = self.retire_cursor.fetch_add(1, Ordering::AcqRel) % 256;
        for offset in 0..256 {
            let candidate = {
                let slots = self.slots.lock();
                slots[(start + offset) % 256]
                    .as_ref()
                    .map(|record| (record.pool.home.clone(), record.pool.generation))
            };
            if let Some((home, generation)) = candidate
                && home.retire_generation(generation)
            {
                return;
            }
        }
    }

    fn join_finished(&self, index: usize) {
        let finished = {
            let mut slots = self.slots.lock();
            let Some(record) = slots[index].as_mut() else {
                return;
            };
            let slot = record.handles.iter_mut().find(|slot| {
                slot.as_ref()
                    .is_some_and(|worker| worker.handle.is_finished())
            });
            slot.and_then(Option::take)
                .map(|worker| (worker, record.pool.home.clone()))
        };
        if let Some((worker, home)) = finished {
            let WorkerHandle {
                handle,
                identity,
                permit,
                credit,
            } = worker;
            if let Err(payload) = handle.join() {
                home.retain_join_panic(identity.lock().as_ref(), payload.as_ref());
            }
            drop(identity);
            #[cfg(test)]
            self.joined_workers.fetch_add(1, Ordering::AcqRel);
            drop(permit);
            drop(credit);
        }
        self.remove_closed(index);
    }

    fn remove_closed(&self, index: usize) {
        let pool = {
            let slots = self.slots.lock();
            let Some(record) = &slots[index] else {
                return;
            };
            if record.handles.iter().any(Option::is_some) {
                return;
            }
            record.pool.clone()
        };
        if !pool.home.closed_generation(pool.generation) {
            return;
        }
        let home = pool.home.clone();
        let generation = pool.generation;
        home.clear_ring(generation);
        drop(pool);
        let record = self.slots.lock()[index].take();
        drop(record);
        home.finish_generation(generation);
    }
}

struct CapacityWaiter<'a> {
    service: &'a ProcessService,
    armed: bool,
}

impl CapacityWaiter<'_> {
    fn arm(&mut self) -> Result<()> {
        if !self.armed {
            self.service
                .pressure
                .fetch_update(Ordering::AcqRel, Ordering::Acquire, |current| {
                    current.checked_add(1)
                })
                .map_err(|_| internal("capacity waiter overflow"))?;
            self.armed = true;
            self.service.wake.notify_all();
        }
        Ok(())
    }
}

impl Drop for CapacityWaiter<'_> {
    fn drop(&mut self) {
        if self.armed {
            self.service.pressure.fetch_sub(1, Ordering::AcqRel);
            self.service.wake.notify_all();
        }
    }
}

async fn acquire(
    service: &ProcessService,
    semaphore: Arc<Semaphore>,
    stop: &GatherStop,
) -> Result<OwnedSemaphorePermit> {
    let permit = semaphore.acquire_owned();
    tokio::pin!(permit);
    let mut waiter = CapacityWaiter {
        service,
        armed: false,
    };
    let queued = poll_fn(|context| match permit.as_mut().poll(context) {
        Poll::Ready(result) => Poll::Ready(result.map_err(|_| internal("process capacity closed"))),
        Poll::Pending => {
            if let Err(error) = waiter.arm() {
                return Poll::Ready(Err(error));
            }
            Poll::Pending
        }
    });
    tokio::select! {
        biased;
        () = stop.wait() => Err(super::cancelled(&stop.run_id)),
        permit = queued => permit,
    }
}

impl Registration {
    pub(super) fn launch(mut self, workers: usize) -> Result<()> {
        for _ in 0..workers.clamp(1, 8) {
            let mut slots = self.service.slots.lock();
            let record = slots[self.slot]
                .as_mut()
                .ok_or_else(|| internal("lost pre-registered pool"))?;
            if record.handles.iter().filter(|slot| slot.is_some()).count() >= workers {
                break;
            }
            let permit = if let Some(permit) = self.permit.take() {
                permit
            } else {
                if self.service.pressure.load(Ordering::Acquire) > 0 {
                    break;
                }
                let Ok(permit) = self.service.workers.clone().try_acquire_owned() else {
                    break;
                };
                permit
            };
            let ordinal = record
                .handles
                .iter()
                .position(Option::is_none)
                .ok_or_else(|| internal("bounded native worker slots exhausted"))?;
            record.handles[ordinal] = Some(launch_worker(&self.pool, &self.service, permit)?);
        }
        {
            let mut state = self.pool.home.state.lock();
            state.pool = Some(Arc::downgrade(&self.pool));
            state.phase = PoolPhase::Ready;
        }
        self.armed = false;
        self.pool.home.native_changed.notify_all();
        Ok(())
    }
}

fn launch_worker(
    pool: &Arc<NativePool>,
    service: &ProcessService,
    permit: OwnedSemaphorePermit,
) -> Result<WorkerHandle> {
    let credit = InfrastructureCredit::reserve(STACK_BYTES + 131_072)?;
    let operator = {
        let state = pool.home.state.lock();
        match &state.slot {
            Slot::Active(attempt) | Slot::Parked(attempt) => Some(attempt.operator.clone()),
            _ => None,
        }
    };
    let identity = Arc::new(Mutex::new(
        operator.as_ref().map(super::worker::worker_identity),
    ));
    let worker_identity = identity.clone();
    let pool = pool.clone();
    let name = worker_name(service);
    let handle = thread::Builder::new()
        .name(name.into())
        .stack_size(STACK_BYTES)
        .spawn(move || {
            super::worker::run(&pool, &worker_identity);
            #[cfg(test)]
            if let Some(service) = pool.service.upgrade() {
                assert!(
                    !service.native_exit_panic.swap(false, Ordering::AcqRel),
                    "native worker exit panic"
                );
            }
        })
        .map_err(|error| internal(&format!("native worker launch: {error}")))?;
    Ok(WorkerHandle {
        handle,
        identity,
        permit,
        credit,
    })
}

#[cfg(test)]
fn worker_name(service: &ProcessService) -> &'static str {
    if service
        .invalid_next_worker_name
        .swap(false, Ordering::AcqRel)
    {
        return "calc-flow-gather\0invalid";
    }
    "calc-flow-gather"
}

#[cfg(not(test))]
fn worker_name(_: &ProcessService) -> &'static str {
    "calc-flow-gather"
}

impl Drop for Registration {
    fn drop(&mut self) {
        if self.armed {
            self.pool.home.state.lock().phase = PoolPhase::Retiring;
            self.pool.home.native_changed.notify_all();
            self.service.wake.notify_all();
        }
    }
}

impl NativePool {
    #[cfg(test)]
    pub(super) fn credit_size(&self) -> usize {
        let Self {
            _credit: credit, ..
        } = self;
        credit.size()
    }

    pub(super) fn wake_joiner(&self) {
        if let Some(service) = self.service.upgrade() {
            service.wake.notify_all();
        }
    }
}

impl GatherHome {
    fn retire_generation(&self, generation: u64) -> bool {
        let mut state = self.state.lock();
        if !mark_retiring(&mut state, generation) {
            return false;
        }
        drop(state);
        self.native_changed.notify_all();
        true
    }

    pub(super) fn retire_if_pressured(&self, state: &mut super::HomeState) {
        let Some(service) = state.service.as_ref().and_then(Weak::upgrade) else {
            return;
        };
        let generation = state.generation;
        if service.pressure.load(Ordering::Acquire) > 0 && mark_retiring(state, generation) {
            self.native_changed.notify_all();
            service.wake.notify_all();
        }
    }

    fn closed_generation(&self, generation: u64) -> bool {
        let state = self.state.lock();
        state.generation == generation
            && (!state.open || state.phase == PoolPhase::Retiring)
            && state.phase != PoolPhase::Initializing
    }

    fn clear_ring(&self, generation: u64) {
        let ring = {
            let mut state = self.state.lock();
            if state.generation != generation {
                return;
            }
            std::mem::take(&mut state.queue)
        };
        drop(ring);
    }

    fn finish_generation(&self, generation: u64) {
        let mut state = self.state.lock();
        if state.generation != generation {
            return;
        }
        state.pool = None;
        state.phase = PoolPhase::Absent;
        self.changed.notify_waiters();
    }
}

fn mark_retiring(state: &mut super::HomeState, generation: u64) -> bool {
    if !retirement_ready(state, generation) || !settled_slot(&state.slot) {
        return false;
    }
    let slot = std::mem::replace(&mut state.slot, Slot::Empty);
    state.slot = match slot {
        Slot::Active(record) => Slot::Parked(record),
        other => other,
    };
    state.phase = PoolPhase::Retiring;
    true
}

fn retirement_ready(state: &super::HomeState, generation: u64) -> bool {
    state.generation == generation
        && state.open
        && state.phase == PoolPhase::Ready
        && state.queue.is_empty()
}

fn settled_slot(slot: &Slot) -> bool {
    match slot {
        Slot::Empty => true,
        Slot::Active(record) | Slot::Parked(record) => record.settled && !record.running,
        Slot::Dropping(_) => false,
    }
}

#[cfg(test)]
pub(crate) struct TestService {
    control: Option<ProcessControl>,
}

#[cfg(test)]
impl TestService {
    pub(crate) fn new(workers: usize, registrations: usize) -> Result<Self> {
        Ok(Self {
            control: Some(start_service(workers, registrations)?),
        })
    }

    pub(crate) fn owner(&self, run_id: Arc<str>) -> super::JobGatherOwner {
        let owner = super::JobGatherOwner::new(run_id);
        owner.0.home.state.lock().service =
            Some(Arc::downgrade(&self.control.as_ref().unwrap().service));
        owner
    }

    pub(crate) fn joined_workers(&self) -> usize {
        self.control
            .as_ref()
            .unwrap()
            .service
            .joined_workers
            .load(Ordering::Acquire)
    }

    pub(crate) fn panic_next_worker_exit(&self) {
        self.control
            .as_ref()
            .unwrap()
            .service
            .native_exit_panic
            .store(true, Ordering::Release);
    }

    pub(crate) fn waiting_requests(&self) -> usize {
        self.control
            .as_ref()
            .unwrap()
            .service
            .pressure
            .load(Ordering::Acquire)
    }

    pub(crate) fn available_capacity(&self) -> (usize, usize, usize) {
        let service = &self.control.as_ref().unwrap().service;
        (
            service.workers.available_permits(),
            service.registrations.available_permits(),
            service
                .slots
                .lock()
                .iter()
                .filter_map(Option::as_ref)
                .count(),
        )
    }

    pub(crate) fn invalidate_next_worker_name(&self) {
        self.control
            .as_ref()
            .unwrap()
            .service
            .invalid_next_worker_name
            .store(true, Ordering::Release);
    }

    pub(crate) fn shutdown(mut self) {
        if let Some(control) = self.control.take() {
            shutdown_on_native_thread(control);
        }
    }
}

#[cfg(test)]
impl Drop for TestService {
    fn drop(&mut self) {
        if let Some(control) = self.control.take() {
            shutdown_on_native_thread(control);
        }
    }
}

#[cfg(test)]
fn shutdown_on_native_thread(control: ProcessControl) {
    assert!(
        tokio::runtime::Handle::try_current().is_err(),
        "isolated service control runs outside Tokio"
    );
    let credit = InfrastructureCredit::reserve(STACK_BYTES + 131_072).unwrap();
    let controller = thread::Builder::new()
        .name("calc-flow-gather-test-control".into())
        .stack_size(STACK_BYTES)
        .spawn(move || {
            let ProcessControl {
                service,
                _joiner: joiner,
                _credit: credit,
            } = control;
            let homes: Vec<_> = service
                .slots
                .lock()
                .iter()
                .filter_map(|slot| slot.as_ref().map(|record| record.pool.home.clone()))
                .collect();
            for home in homes {
                home.close();
            }
            service.shutdown.store(true, Ordering::Release);
            service.wake.notify_all();
            let result = joiner.join();
            drop(credit);
            result
        })
        .unwrap();
    controller.join().unwrap().unwrap();
    drop(credit);
}
