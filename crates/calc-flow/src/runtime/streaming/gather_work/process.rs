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
    AdmissionFailure, AdmissionResult, GatherHome, GatherStop, PendingHomeOwner, PoolPhase, Slot,
    increment, internal,
};
use crate::Result;

const STACK_BYTES: usize = 512 * 1_024;
static PROCESS: Mutex<Option<ProcessControl>> = Mutex::new(None);

struct ProcessControl {
    service: Arc<ProcessService>,
    _joiner: JoinHandle<()>,
}

pub(super) struct ProcessService {
    pub(super) worker_limit: usize,
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
    // Release dynamic controls before the pool refunds their generation credit.
    handles: Box<[Option<WorkerHandle>]>,
    pool: Arc<NativePool>,
    _registration: OwnedSemaphorePermit,
}

struct WorkerHandle {
    handle: JoinHandle<()>,
    identity: Arc<Mutex<Option<super::WorkerIdentity>>>,
    permit: OwnedSemaphorePermit,
}

pub(super) struct NativePool {
    pub(super) generation: u64,
    // Refund generation credit before publishing the final Home release.
    _credit: MemoryReservation,
    pub(super) home: PendingHomeOwner,
    service: Weak<ProcessService>,
}

pub(super) struct Registration {
    service: Arc<ProcessService>,
    pool: Arc<NativePool>,
    slot: usize,
    permit: Option<OwnedSemaphorePermit>,
    armed: bool,
    pub(super) generation: u64,
}

pub(super) fn worker_capacity(available: usize) -> usize {
    available.max(1)
}

pub(super) fn available_workers() -> usize {
    worker_capacity(thread::available_parallelism().map_or(1, usize::from))
}

fn ring_capacity(workers: usize) -> Result<usize> {
    workers
        .checked_next_power_of_two()
        .ok_or_else(|| internal("native request ring capacity overflow"))
}

pub(super) fn generation_bytes(workers: usize) -> Result<usize> {
    // Identity replacement briefly retains both old and new diagnostic names.
    // The Arc header and alignment allowance cover the owned identity mutex.
    let per_worker = size_of::<Option<WorkerHandle>>()
        + size_of::<Mutex<Option<super::WorkerIdentity>>>()
        + 4 * size_of::<AtomicUsize>()
        + 2 * super::DIAGNOSTIC_NAME_BYTES;
    let handles = workers.checked_mul(per_worker);
    let ring = ring_capacity(workers)?.checked_mul(size_of::<super::Request>());
    handles
        .zip(ring)
        .and_then(|(handles, ring)| handles.checked_add(ring))
        .and_then(|bytes| bytes.checked_add(16_384))
        .ok_or_else(|| internal("native generation control credit overflow"))
}

pub(super) fn service() -> Result<Arc<ProcessService>> {
    let mut process = PROCESS.lock();
    if let Some(control) = &*process {
        return Ok(control.service.clone());
    }
    let workers = available_workers();
    let control = start_service(workers, 256)?;
    let service = control.service.clone();
    *process = Some(control);
    Ok(service)
}

fn start_service(workers: usize, registrations: usize) -> Result<ProcessControl> {
    let workers = worker_capacity(workers);
    let service = Arc::new(ProcessService {
        worker_limit: workers,
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
        let (pool, slot) = {
            let mut slots = self.slots.lock();
            let slot = slots
                .iter()
                .position(Option::is_none)
                .ok_or_else(|| internal("paid registry exhausted"))?;
            let pool = Arc::new(self.prepare_pool(home)?);
            slots[slot] = Some(PoolRecord {
                pool: pool.clone(),
                handles: std::iter::repeat_with(|| None)
                    .take(self.worker_limit)
                    .collect::<Vec<_>>()
                    .into_boxed_slice(),
                _registration: registration,
            });
            (pool, slot)
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
        let bytes = generation_bytes(self.worker_limit)?;
        let capacity = ring_capacity(self.worker_limit)?;
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
                bytes,
                &attempt.operator,
            )?;
            attempt
                .credit
                .try_grow(bytes)
                .map_err(|source| AdmissionFailure::Budget {
                    stage: "generation",
                    source,
                })?;
            let credit = attempt.credit.split(bytes);
            let queue = std::collections::VecDeque::with_capacity(capacity);
            if queue.capacity() > capacity {
                return Err(internal("native request ring exceeded prepaid capacity").into());
            }
            let generation = increment(state.generation, "pool generation")?;
            // Track both the final NativePool owner and the joiner's Home reference.
            home.retain_releases(2)?;
            state.generation = generation;
            state.phase = PoolPhase::Initializing;
            state.service = Some(Arc::downgrade(self));
            state.queue = queue;
            (state.generation, credit)
        };
        Ok(NativePool {
            generation,
            _credit: credit,
            home: PendingHomeOwner(Some(home)),
            service: Arc::downgrade(self),
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
            } = worker;
            if let Err(payload) = handle.join() {
                home.retain_join_panic(identity.lock().as_ref(), payload.as_ref());
            }
            drop(identity);
            #[cfg(test)]
            self.joined_workers.fetch_add(1, Ordering::AcqRel);
            drop(permit);
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
        let workers = workers.max(1).min(self.service.worker_limit);
        for _ in 0..workers {
            let mut slots = self.service.slots.lock();
            let record = slots[self.slot]
                .as_mut()
                .ok_or_else(|| internal("lost pre-registered pool"))?;
            if record.handles.iter().filter(|slot| slot.is_some()).count() >= workers {
                break;
            }
            let Some(permit) = launch_permit(&mut self.permit, &self.service) else {
                break;
            };
            let ordinal = record
                .handles
                .iter()
                .position(Option::is_none)
                .ok_or_else(|| internal("bounded native worker slots exhausted"))?;
            record.handles[ordinal] = Some(launch_worker(&self.pool, &self.service, permit)?);
        }
        {
            let state = self.pool.home.state.lock();
            if !state.open {
                return Err(super::cancelled(&self.pool.home.run_id));
            }
        }
        {
            let slots = self.service.slots.lock();
            let ready = slots[self.slot].as_ref().is_some_and(|record| {
                record
                    .handles
                    .iter()
                    .flatten()
                    .any(|worker| !worker.handle.is_finished())
            });
            if !ready {
                if !self.pool.home.state.lock().open {
                    return Err(super::cancelled(&self.pool.home.run_id));
                }
                return Err(internal("native pool has no runnable worker"));
            }
        }
        {
            let mut state = self.pool.home.state.lock();
            if !state.open {
                return Err(super::cancelled(&self.pool.home.run_id));
            }
            state.pool = Some(Arc::downgrade(&self.pool));
            state.phase = PoolPhase::Ready;
        }
        self.armed = false;
        self.pool.home.native_changed.notify_all();
        Ok(())
    }
}

fn launch_permit(
    permit: &mut Option<OwnedSemaphorePermit>,
    service: &ProcessService,
) -> Option<OwnedSemaphorePermit> {
    if let Some(permit) = permit.take() {
        return Some(permit);
    }
    if service.pressure.load(Ordering::Acquire) > 0 {
        return None;
    }
    service.workers.clone().try_acquire_owned().ok()
}

fn launch_worker(
    pool: &Arc<NativePool>,
    service: &ProcessService,
    permit: OwnedSemaphorePermit,
) -> Result<WorkerHandle> {
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

    fn finish_generation(self: Arc<Self>, generation: u64) {
        let mut state = self.state.lock();
        let current = state.generation == generation;
        if current {
            state.pool = None;
            state.phase = PoolPhase::Absent;
            self.changed.notify_waiters();
        }
        #[cfg(test)]
        let release_probe = if current {
            state.generation_release_probe.take()
        } else {
            None
        };
        drop(state);
        #[cfg(test)]
        if let Some(probe) = release_probe {
            probe();
        }
        self.release_owner();
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
    let controller = thread::Builder::new()
        .name("calc-flow-gather-test-control".into())
        .stack_size(STACK_BYTES)
        .spawn(move || {
            let ProcessControl {
                service,
                _joiner: joiner,
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
            joiner.join()
        })
        .unwrap();
    controller.join().unwrap().unwrap();
}
