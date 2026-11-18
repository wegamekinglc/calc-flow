use std::sync::{
    Arc,
    atomic::{AtomicU64, Ordering},
};

use super::{Result, mismatch};

#[derive(Clone, Debug)]
pub(crate) struct V4InflightBudget {
    inner: Arc<BudgetInner>,
}

#[derive(Debug)]
struct BudgetInner {
    limit: u64,
    reserved: AtomicU64,
}

impl Default for V4InflightBudget {
    fn default() -> Self {
        Self::new(1 << 30)
    }
}

impl V4InflightBudget {
    pub(crate) fn new(limit: u64) -> Self {
        Self {
            inner: Arc::new(BudgetInner {
                limit,
                reserved: AtomicU64::new(0),
            }),
        }
    }

    pub(crate) fn reserve(&self, bytes: u64) -> Result<V4CaptureReservation> {
        let mut current = self.inner.reserved.load(Ordering::Acquire);
        loop {
            let Some(next) = current.checked_add(bytes) else {
                return Err(budget_error(self.inner.limit));
            };
            if next > self.inner.limit {
                return Err(budget_error(self.inner.limit));
            }
            match self.inner.reserved.compare_exchange_weak(
                current,
                next,
                Ordering::AcqRel,
                Ordering::Acquire,
            ) {
                Ok(_) => {
                    return Ok(V4CaptureReservation {
                        inner: Arc::new(ReservationInner {
                            budget: Arc::clone(&self.inner),
                            bytes,
                        }),
                    });
                }
                Err(observed) => current = observed,
            }
        }
    }
}

fn budget_error(limit: u64) -> crate::CalcFlowError {
    mismatch(format!(
        "checkpoint capture exceeds the {limit}-byte in-flight reservation"
    ))
}

#[derive(Clone, Debug)]
pub(crate) struct V4CaptureReservation {
    #[allow(dead_code, reason = "the owned Arc releases the reservation on drop")]
    inner: Arc<ReservationInner>,
}

#[derive(Debug)]
struct ReservationInner {
    budget: Arc<BudgetInner>,
    bytes: u64,
}

impl Drop for ReservationInner {
    fn drop(&mut self) {
        self.budget.reserved.fetch_sub(self.bytes, Ordering::AcqRel);
    }
}

impl V4CaptureReservation {
    #[cfg(test)]
    pub(crate) fn bytes(&self) -> u64 {
        self.inner.bytes
    }
}

#[cfg(test)]
mod tests {
    use super::V4InflightBudget;

    #[test]
    fn shared_reservations_release_the_epoch_budget_once() {
        let budget = V4InflightBudget::new(8);
        let first = budget.reserve(6).unwrap();
        assert!(budget.reserve(3).is_err());
        let shared = first.clone();
        drop(first);
        assert!(budget.reserve(3).is_err());
        drop(shared);
        assert_eq!(budget.reserve(8).unwrap().bytes(), 8);
    }
}
