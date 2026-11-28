//! Fixed, bounded compute-only morsel execution for one job.

use std::{
    any::Any,
    collections::VecDeque,
    panic::{AssertUnwindSafe, catch_unwind},
    sync::{
        Arc, Condvar, Mutex, MutexGuard, Weak,
        atomic::{AtomicBool, AtomicU64, Ordering},
    },
    thread::{self, JoinHandle},
};

use tokio::sync::Notify;

use crate::{CalcFlowError, Result};

mod job;
#[cfg(test)]
pub(crate) use job::MorselTestHooks;
pub(crate) use job::{JobMorselOwner, ManagedMorsel, execute_managed};

static NEXT_POOL_ID: AtomicU64 = AtomicU64::new(1);
thread_local! {
    static CURRENT_POOL: std::cell::Cell<u64> = const { std::cell::Cell::new(0) };
}

/// Fixed compute-pool limits and the inline row threshold.
#[derive(Clone, Copy, Debug)]
pub(crate) struct MorselPoolConfig {
    pub workers: usize,
    pub max_morsels: usize,
    pub max_rows: usize,
    pub max_bytes: usize,
    pub inline_rows: usize,
}

/// Logical input, scratch and output charges for one morsel.
#[derive(Clone, Copy, Debug, Default)]
pub(crate) struct MorselCost {
    pub rows: usize,
    pub scratch_bytes: usize,
    pub output_bytes: usize,
}

impl MorselCost {
    fn bytes(self) -> Result<usize> {
        self.scratch_bytes
            .checked_add(self.output_bytes)
            .ok_or_else(|| invalid("morsel.bytes", "morsel byte charge overflowed"))
    }
}

/// Current admitted work, including queued and running morsels.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub(crate) struct MorselUsage {
    pub morsels: usize,
    pub rows: usize,
    pub bytes: usize,
}

struct PoolState {
    queue: VecDeque<Box<dyn Job>>,
    usage: MorselUsage,
    closed: bool,
}

struct Inner {
    id: u64,
    name: String,
    config: MorselPoolConfig,
    state: Mutex<PoolState>,
    ready: Condvar,
    settled: Condvar,
    settled_async: Notify,
    // Running leases poll this flag without locking.
    cancelled: Arc<AtomicBool>,
    next_ticket: AtomicU64,
}

/// Cooperative cancellation visible to a running morsel.
pub(crate) struct MorselLease {
    pool_cancelled: Arc<AtomicBool>,
    ticket_cancelled: Arc<AtomicBool>,
    run_id: String,
}

impl MorselLease {
    pub(crate) fn check(&self) -> Result<()> {
        if self.pool_cancelled.load(Ordering::Acquire)
            || self.ticket_cancelled.load(Ordering::Acquire)
        {
            return Err(CalcFlowError::Cancelled {
                run_id: self.run_id.clone(),
            });
        }
        Ok(())
    }
}

/// A cancellation handle that also wakes blocked workers and admission.
#[derive(Clone)]
pub(crate) struct MorselCancellation {
    cancelled: Arc<AtomicBool>,
    pool: Weak<Inner>,
}

impl MorselCancellation {
    pub(crate) fn cancel(&self) {
        self.cancelled.store(true, Ordering::Release);
        if let Some(pool) = self.pool.upgrade() {
            // Let waiters finish checking cancellation and enter their wait
            // before sending a notification that a condition variable cannot save.
            drop(lock(&pool.state));
            pool.ready.notify_all();
            pool.settled.notify_all();
            pool.settled_async.notify_waiters();
        }
    }
}

struct TicketState<T> {
    result: Mutex<Option<Result<T>>>,
    ready: Condvar,
    ready_async: Notify,
    refunded: AtomicBool,
    #[cfg(test)]
    consumed: AtomicBool,
}

/// A typed, settled-once result and cancellation handle.
pub(crate) struct MorselTicket<T> {
    id: u64,
    pool: Weak<Inner>,
    state: Arc<TicketState<T>>,
    cancelled: Arc<AtomicBool>,
    cost: MorselCost,
}

impl<T> Clone for MorselTicket<T> {
    fn clone(&self) -> Self {
        Self {
            id: self.id,
            pool: self.pool.clone(),
            state: Arc::clone(&self.state),
            cancelled: Arc::clone(&self.cancelled),
            cost: self.cost,
        }
    }
}

impl<T> MorselTicket<T> {
    /// Cancels queued work immediately and signals a running lease.
    pub(crate) fn cancel(&self) {
        self.cancelled.store(true, Ordering::Release);
        if let Some(pool) = self.pool.upgrade() {
            pool.cancel_ticket(self.id);
        }
    }

    /// Waits for the owned result. Worker-local joins fail instead of deadlocking.
    pub(crate) fn join(self) -> Result<T> {
        self.reject_nested_join()?;
        let mut result = lock(&self.state.result);
        while result.is_none() {
            result = self
                .state
                .ready
                .wait(result)
                .unwrap_or_else(|error| panic!("morsel ticket lock poisoned: {error}"));
        }
        let settled = result
            .take()
            .expect("morsel ticket result is present after settlement");
        #[cfg(test)]
        self.state.consumed.store(true, Ordering::Release);
        settled
    }

    /// Awaits the owned result without blocking the async caller.
    pub(crate) async fn join_async(self) -> Result<T> {
        self.reject_nested_join()?;
        loop {
            let notified = self.state.ready_async.notified();
            tokio::pin!(notified);
            notified.as_mut().enable();
            if let Some(result) = lock(&self.state.result).take() {
                #[cfg(test)]
                self.state.consumed.store(true, Ordering::Release);
                return result;
            }
            notified.await;
        }
    }

    #[cfg(test)]
    pub(crate) fn is_consumed(&self) -> bool {
        self.state.consumed.load(Ordering::Acquire)
    }

    fn reject_nested_join(&self) -> Result<()> {
        if self.pool.upgrade().is_some() && CURRENT_POOL.with(std::cell::Cell::get) != 0 {
            return Err(invalid(
                "morsel_pool.nested",
                "a morsel cannot join work owned by its own pool",
            ));
        }
        Ok(())
    }

    fn settle(&self, result: Result<T>, pool: &Inner) {
        let mut slot = lock(&self.state.result);
        if slot.is_some() {
            return;
        }
        *slot = Some(result);
        self.state.ready.notify_all();
        self.state.ready_async.notify_waiters();
        drop(slot);
        if !self.state.refunded.swap(true, Ordering::AcqRel) {
            pool.refund(self.cost);
        }
    }
}

trait Job: Send {
    fn execute(self: Box<Self>, pool: &Inner);
    fn cancel(self: Box<Self>, pool: &Inner);
    fn id(&self) -> u64;
}

struct Work<T, O, F> {
    ticket: MorselTicket<O>,
    input: Option<T>,
    operation: Option<F>,
}

impl<T, O, F> Work<T, O, F>
where
    T: Send + 'static,
    O: Send + 'static,
    F: FnOnce(T, &MorselLease) -> Result<O> + Send + 'static,
{
    fn execute_inner(mut self, pool: &Inner) -> Result<O> {
        let input = self.input.take().expect("morsel input is owned once");
        let operation = self
            .operation
            .take()
            .expect("morsel operation is owned once");
        let lease = MorselLease {
            pool_cancelled: Arc::clone(&pool.cancelled),
            ticket_cancelled: Arc::clone(&self.ticket.cancelled),
            run_id: pool.name.clone(),
        };
        lease.check()?;
        operation(input, &lease)
    }
}

impl<T, O, F> Job for Work<T, O, F>
where
    T: Send + 'static,
    O: Send + 'static,
    F: FnOnce(T, &MorselLease) -> Result<O> + Send + 'static,
{
    fn execute(self: Box<Self>, pool: &Inner) {
        let ticket = self.ticket.clone();
        let result = catch_unwind(AssertUnwindSafe(|| self.execute_inner(pool)))
            .map_err(|payload| CalcFlowError::TaskPanicked {
                task_id: ticket.id,
                message: panic_message(payload.as_ref()),
            })
            .and_then(|result| result);
        let failed = matches!(
            &result,
            Err(error) if !matches!(error, CalcFlowError::Cancelled { .. })
        );
        ticket.settle(result, pool);
        if failed {
            pool.fail();
        }
    }

    fn cancel(self: Box<Self>, pool: &Inner) {
        let ticket = self.ticket.clone();
        match catch_unwind(AssertUnwindSafe(|| drop(self))) {
            Ok(()) => ticket.settle(
                Err(CalcFlowError::Cancelled {
                    run_id: pool.name.clone(),
                }),
                pool,
            ),
            Err(payload) => {
                ticket.settle(
                    Err(CalcFlowError::TaskPanicked {
                        task_id: ticket.id,
                        message: panic_message(payload.as_ref()),
                    }),
                    pool,
                );
                pool.fail();
            }
        }
    }

    fn id(&self) -> u64 {
        self.ticket.id
    }
}

/// A cloneable submission handle for one job-scoped pool.
pub(crate) struct MorselClient {
    inner: Arc<Inner>,
}

impl Clone for MorselClient {
    fn clone(&self) -> Self {
        Self {
            inner: Arc::clone(&self.inner),
        }
    }
}

impl MorselClient {
    #[cfg(test)]
    pub(crate) fn usage(&self) -> MorselUsage {
        lock(&self.inner.state).usage
    }

    pub(crate) fn submit<T, O, F>(
        &self,
        input: T,
        cost: MorselCost,
        operation: F,
    ) -> Result<MorselTicket<O>>
    where
        T: Send + 'static,
        O: Send + 'static,
        F: FnOnce(T, &MorselLease) -> Result<O> + Send + 'static,
    {
        Inner::reject_nested()?;
        self.inner.validate_cost(cost)?;
        let id = self.inner.admit(cost)?;
        self.submit_admitted(input, cost, operation, id)
    }

    pub(crate) async fn submit_async<T, O, F>(
        &self,
        input: T,
        cost: MorselCost,
        operation: F,
    ) -> Result<MorselTicket<O>>
    where
        T: Send + 'static,
        O: Send + 'static,
        F: FnOnce(T, &MorselLease) -> Result<O> + Send + 'static,
    {
        Inner::reject_nested()?;
        self.inner.validate_cost(cost)?;
        let id = self.inner.admit_async(cost).await?;
        self.submit_admitted(input, cost, operation, id)
    }

    fn submit_admitted<T, O, F>(
        &self,
        input: T,
        cost: MorselCost,
        operation: F,
        id: u64,
    ) -> Result<MorselTicket<O>>
    where
        T: Send + 'static,
        O: Send + 'static,
        F: FnOnce(T, &MorselLease) -> Result<O> + Send + 'static,
    {
        let ticket = MorselTicket {
            id,
            pool: Arc::downgrade(&self.inner),
            state: Arc::new(TicketState {
                result: Mutex::new(None),
                ready: Condvar::new(),
                ready_async: Notify::new(),
                refunded: AtomicBool::new(false),
                #[cfg(test)]
                consumed: AtomicBool::new(false),
            }),
            cancelled: Arc::new(AtomicBool::new(false)),
            cost,
        };
        let work = Work {
            ticket: ticket.clone(),
            input: Some(input),
            operation: Some(operation),
        };
        self.inner.enqueue(work)?;
        Ok(ticket)
    }

    pub(crate) fn execute<T, O, F>(&self, input: T, cost: MorselCost, operation: F) -> Result<O>
    where
        T: Send + 'static,
        O: Send + 'static,
        F: FnOnce(T, &MorselLease) -> Result<O> + Send + 'static,
    {
        Inner::reject_nested()?;
        self.inner.validate_cost(cost)?;
        self.inner.check_open()?;
        if cost.rows <= self.inner.config.inline_rows {
            let lease = MorselLease {
                pool_cancelled: Arc::clone(&self.inner.cancelled),
                ticket_cancelled: Arc::new(AtomicBool::new(false)),
                run_id: self.inner.name.clone(),
            };
            let result = catch_unwind(AssertUnwindSafe(|| operation(input, &lease)))
                .map_err(|payload| CalcFlowError::TaskPanicked {
                    task_id: self.inner.id,
                    message: panic_message(payload.as_ref()),
                })
                .and_then(|result| result);
            if result.is_err() {
                self.inner.fail();
            }
            return result;
        }
        self.submit(input, cost, operation)?.join()
    }
}

/// Owns the fixed workers and settles every outstanding ticket on teardown.
pub(crate) struct MorselPool {
    inner: Arc<Inner>,
    workers: Vec<JoinHandle<()>>,
    closed: bool,
    join_error: Option<String>,
}

impl MorselPool {
    pub(crate) fn new(name: &str, config: MorselPoolConfig) -> Result<Self> {
        validate_config(config)?;
        let inner = Arc::new(Inner {
            id: NEXT_POOL_ID.fetch_add(1, Ordering::Relaxed),
            name: name.to_owned(),
            config,
            state: Mutex::new(PoolState {
                queue: VecDeque::new(),
                usage: MorselUsage::default(),
                closed: false,
            }),
            ready: Condvar::new(),
            settled: Condvar::new(),
            settled_async: Notify::new(),
            cancelled: Arc::new(AtomicBool::new(false)),
            next_ticket: AtomicU64::new(1),
        });
        let mut workers: Vec<JoinHandle<()>> = Vec::with_capacity(config.workers);
        for index in 0..config.workers {
            let worker_inner = Arc::clone(&inner);
            let handle = match thread::Builder::new()
                .name(format!("{name}-morsel-{index}"))
                .spawn(move || worker_loop(worker_inner))
            {
                Ok(handle) => handle,
                Err(error) => {
                    inner.cancelled.store(true, Ordering::Release);
                    lock(&inner.state).closed = true;
                    inner.ready.notify_all();
                    inner.settled.notify_all();
                    for worker in workers {
                        let _ = worker.join();
                    }
                    return Err(CalcFlowError::Internal {
                        message: format!("morsel worker could not start: {error}"),
                    });
                }
            };
            workers.push(handle);
        }
        Ok(Self {
            inner,
            workers,
            closed: false,
            join_error: None,
        })
    }

    pub(crate) fn client(&self) -> MorselClient {
        MorselClient {
            inner: Arc::clone(&self.inner),
        }
    }

    pub(crate) fn cancellation(&self) -> MorselCancellation {
        MorselCancellation {
            cancelled: Arc::clone(&self.inner.cancelled),
            pool: Arc::downgrade(&self.inner),
        }
    }

    pub(crate) fn usage(&self) -> MorselUsage {
        lock(&self.inner.state).usage
    }

    pub(crate) fn close_and_join(&mut self) -> Result<()> {
        if self.closed {
            return self.join_error.as_ref().map_or_else(
                || Ok(()),
                |message| {
                    Err(CalcFlowError::TaskPanicked {
                        task_id: self.inner.id,
                        message: message.clone(),
                    })
                },
            );
        }
        self.closed = true;
        self.inner.cancelled.store(true, Ordering::Release);
        {
            lock(&self.inner.state).closed = true;
        }
        self.inner.cancel_queued();
        self.inner.ready.notify_all();
        self.inner.settled.notify_all();
        self.inner.settled_async.notify_waiters();
        let mut first_error = None;
        for worker in self.workers.drain(..) {
            if let Err(payload) = worker.join() {
                first_error.get_or_insert_with(|| panic_message(payload.as_ref()));
            }
        }
        if let Some(message) = first_error {
            self.join_error = Some(message.clone());
            return Err(CalcFlowError::TaskPanicked {
                task_id: self.inner.id,
                message,
            });
        }
        Ok(())
    }
}

impl Drop for MorselPool {
    fn drop(&mut self) {
        let _ = self.close_and_join();
    }
}

impl Inner {
    fn validate_cost(&self, cost: MorselCost) -> Result<()> {
        if cost.rows > self.config.max_rows {
            return Err(invalid("morsel.rows", "morsel exceeds the row limit"));
        }
        if cost.bytes()? > self.config.max_bytes {
            return Err(invalid("morsel.bytes", "morsel exceeds the byte limit"));
        }
        Ok(())
    }

    fn reject_nested() -> Result<()> {
        if CURRENT_POOL.with(std::cell::Cell::get) != 0 {
            return Err(invalid(
                "morsel_pool.nested",
                "nested morsel submission is not allowed",
            ));
        }
        Ok(())
    }

    fn check_open(&self) -> Result<()> {
        if lock(&self.state).closed || self.cancelled.load(Ordering::Acquire) {
            return Err(CalcFlowError::Cancelled {
                run_id: self.name.clone(),
            });
        }
        Ok(())
    }

    fn admit(&self, cost: MorselCost) -> Result<u64> {
        let bytes = cost.bytes()?;
        let mut state = lock(&self.state);
        loop {
            if let Some(id) = self.admit_if_available(&mut state, cost, bytes)? {
                return Ok(id);
            }
            state = self
                .settled
                .wait(state)
                .unwrap_or_else(|error| panic!("morsel admission lock poisoned: {error}"));
        }
    }

    async fn admit_async(&self, cost: MorselCost) -> Result<u64> {
        let bytes = cost.bytes()?;
        loop {
            let notified = self.settled_async.notified();
            tokio::pin!(notified);
            notified.as_mut().enable();
            let admitted = {
                let mut state = lock(&self.state);
                self.admit_if_available(&mut state, cost, bytes)?
            };
            if let Some(id) = admitted {
                return Ok(id);
            }
            notified.await;
        }
    }

    fn admit_if_available(
        &self,
        state: &mut PoolState,
        cost: MorselCost,
        bytes: usize,
    ) -> Result<Option<u64>> {
        if state.closed || self.cancelled.load(Ordering::Acquire) {
            return Err(CalcFlowError::Cancelled {
                run_id: self.name.clone(),
            });
        }
        let next = MorselUsage {
            morsels: state
                .usage
                .morsels
                .checked_add(1)
                .ok_or_else(|| invalid("morsel.count", "morsel count overflowed"))?,
            rows: state
                .usage
                .rows
                .checked_add(cost.rows)
                .ok_or_else(|| invalid("morsel.rows", "morsel row charge overflowed"))?,
            bytes: state
                .usage
                .bytes
                .checked_add(bytes)
                .ok_or_else(|| invalid("morsel.bytes", "morsel byte charge overflowed"))?,
        };
        if next.morsels <= self.config.max_morsels
            && next.rows <= self.config.max_rows
            && next.bytes <= self.config.max_bytes
        {
            state.usage = next;
            return Ok(Some(self.next_ticket.fetch_add(1, Ordering::Relaxed)));
        }
        Ok(None)
    }

    fn enqueue<T, O, F>(&self, work: Work<T, O, F>) -> Result<()>
    where
        T: Send + 'static,
        O: Send + 'static,
        F: FnOnce(T, &MorselLease) -> Result<O> + Send + 'static,
    {
        let mut state = lock(&self.state);
        if state.closed || self.cancelled.load(Ordering::Acquire) {
            drop(state);
            work.ticket.settle(
                Err(CalcFlowError::Cancelled {
                    run_id: self.name.clone(),
                }),
                self,
            );
            return Err(CalcFlowError::Cancelled {
                run_id: self.name.clone(),
            });
        }
        state.queue.push_back(Box::new(work));
        self.ready.notify_one();
        Ok(())
    }

    fn next_job(&self) -> Option<Box<dyn Job>> {
        let mut state = lock(&self.state);
        loop {
            if self.cancelled.load(Ordering::Acquire) {
                if let Some(job) = state.queue.pop_front() {
                    drop(state);
                    job.cancel(self);
                    state = lock(&self.state);
                    continue;
                }
                return None;
            }
            if let Some(job) = state.queue.pop_front() {
                return Some(job);
            }
            if state.closed {
                return None;
            }
            state = self
                .ready
                .wait(state)
                .unwrap_or_else(|error| panic!("morsel queue lock poisoned: {error}"));
        }
    }

    fn refund(&self, cost: MorselCost) {
        let mut state = lock(&self.state);
        state.usage.morsels = state.usage.morsels.saturating_sub(1);
        state.usage.rows = state.usage.rows.saturating_sub(cost.rows);
        state.usage.bytes = state.usage.bytes.saturating_sub(cost.bytes().unwrap_or(0));
        self.settled.notify_all();
        self.settled_async.notify_waiters();
    }

    fn cancel_ticket(&self, id: u64) {
        let job = {
            let mut state = lock(&self.state);
            let index = state.queue.iter().position(|job| job.id() == id);
            index.and_then(|index| state.queue.remove(index))
        };
        if let Some(job) = job {
            job.cancel(self);
        }
    }

    fn cancel_queued(&self) {
        loop {
            let job = lock(&self.state).queue.pop_front();
            let Some(job) = job else {
                break;
            };
            job.cancel(self);
        }
    }

    fn fail(&self) {
        self.cancelled.store(true, Ordering::Release);
        self.cancel_queued();
        self.ready.notify_all();
        self.settled.notify_all();
        self.settled_async.notify_waiters();
    }
}

#[allow(
    clippy::needless_pass_by_value,
    reason = "the spawned worker owns its pool Arc for its full lifetime"
)]
fn worker_loop(inner: Arc<Inner>) {
    while let Some(job) = inner.next_job() {
        CURRENT_POOL.with(|current| current.set(inner.id));
        job.execute(&inner);
        CURRENT_POOL.with(|current| current.set(0));
    }
    CURRENT_POOL.with(|current| current.set(0));
}

fn validate_config(config: MorselPoolConfig) -> Result<()> {
    for (field, value) in [
        ("morsel_pool.workers", config.workers),
        ("morsel_pool.max_morsels", config.max_morsels),
        ("morsel_pool.max_rows", config.max_rows),
        ("morsel_pool.max_bytes", config.max_bytes),
    ] {
        if value == 0 {
            return Err(invalid(field, "morsel pool limits must be positive"));
        }
    }
    Ok(())
}

fn invalid(field: &str, message: &str) -> CalcFlowError {
    CalcFlowError::InvalidArgument {
        field: field.into(),
        message: message.into(),
    }
}

fn lock<T>(mutex: &Mutex<T>) -> MutexGuard<'_, T> {
    mutex
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner)
}

fn panic_message(payload: &dyn Any) -> String {
    if let Some(message) = payload.downcast_ref::<&str>() {
        return (*message).to_owned();
    }
    if let Some(message) = payload.downcast_ref::<String>() {
        return message.clone();
    }
    "non-string morsel panic payload".into()
}

#[cfg(test)]
mod tests;
