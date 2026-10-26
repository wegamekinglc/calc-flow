//! A node-local ownership quota layered over the engine's reservation pool.
//! Every state lease (including candidates and cached checkpoints) pays before
//! allocation. Dropping a failed candidate returns its charge automatically.
use datafusion::{
    error::{DataFusionError, Result as DFResult},
    execution::memory_pool::{MemoryConsumer, MemoryLimit, MemoryPool, MemoryReservation},
};
use parking_lot::Mutex;
use std::{fmt, sync::Arc};

#[derive(Debug)]
pub(crate) struct OwnedStatePool {
    parent: Arc<dyn MemoryPool>,
    name: &'static str,
    limit_error: &'static str,
    state: Mutex<(usize, usize)>,
    #[cfg(test)]
    peak: std::sync::atomic::AtomicUsize,
}

impl OwnedStatePool {
    pub(crate) fn new(
        parent: Arc<dyn MemoryPool>,
        name: &'static str,
        limit_error: &'static str,
    ) -> Self {
        Self {
            parent,
            name,
            limit_error,
            state: Mutex::new((0, usize::MAX)),
            #[cfg(test)]
            peak: std::sync::atomic::AtomicUsize::new(0),
        }
    }

    #[cfg(test)]
    pub(crate) fn take_peak(&self) -> usize {
        self.peak
            .swap(self.reserved(), std::sync::atomic::Ordering::SeqCst)
    }

    pub(crate) fn set_limit(&self, limit: Option<u64>) -> DFResult<()> {
        let limit = limit.map_or(usize::MAX, |n| usize::try_from(n).unwrap_or(usize::MAX));
        let mut state = self.state.lock();
        if state.0 > limit {
            return Err(DataFusionError::ResourcesExhausted(self.limit_error.into()));
        }
        state.1 = limit;
        Ok(())
    }
}

impl fmt::Display for OwnedStatePool {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.name)
    }
}

impl MemoryPool for OwnedStatePool {
    fn name(&self) -> &'static str {
        self.name
    }
    fn register(&self, consumer: &MemoryConsumer) {
        self.parent.register(consumer);
    }
    fn unregister(&self, consumer: &MemoryConsumer) {
        self.parent.unregister(consumer);
    }
    fn grow(&self, reservation: &MemoryReservation, additional: usize) {
        let mut state = self.state.lock();
        self.parent.grow(reservation, additional);
        state.0 += additional;
    }
    fn shrink(&self, reservation: &MemoryReservation, shrink: usize) {
        let mut state = self.state.lock();
        self.parent.shrink(reservation, shrink);
        state.0 -= shrink;
    }
    fn try_grow(&self, reservation: &MemoryReservation, additional: usize) -> DFResult<()> {
        let mut state = self.state.lock();
        let next = state
            .0
            .checked_add(additional)
            .filter(|n| *n <= state.1)
            .ok_or_else(|| {
                DataFusionError::ResourcesExhausted(format!(
                    "{} exceeded at {}: {} + {additional} bytes > {}",
                    self.name,
                    reservation.consumer().name(),
                    state.0,
                    state.1
                ))
            })?;
        self.parent.try_grow(reservation, additional)?;
        state.0 = next;
        #[cfg(test)]
        self.peak
            .fetch_max(next, std::sync::atomic::Ordering::SeqCst);
        Ok(())
    }
    fn reserved(&self) -> usize {
        self.state.lock().0
    }
    fn memory_limit(&self) -> MemoryLimit {
        MemoryLimit::Finite(self.state.lock().1)
    }
}
