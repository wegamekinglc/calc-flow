use super::super::native_lookup::scratch_error;
use crate::Result;
use datafusion::execution::memory_pool::MemoryReservation;
use std::ops::Range;
use std::sync::Arc;

#[derive(Clone, Copy)]
pub(super) struct Partition {
    rows: usize,
    units: usize,
}

impl Partition {
    pub(super) fn new(rows: usize, capacity: usize) -> Self {
        Self {
            rows,
            units: capacity.max(1).min((rows / super::MIN_PROBE_ROWS).max(1)),
        }
    }

    pub(super) fn units(self) -> usize {
        self.units
    }

    pub(super) fn range(self, ordinal: usize) -> Range<usize> {
        debug_assert!(ordinal < self.units);
        let width = self.rows / self.units;
        let remainder = self.rows % self.units;
        let start = ordinal * width + ordinal.min(remainder);
        start..start + width + usize::from(ordinal < remainder)
    }
}

pub(super) fn sentinel(limit: u64) -> Option<usize> {
    usize::try_from(limit).ok()?.checked_add(1)
}

pub(super) fn add_count(total: usize, value: usize, limit: u64) -> Result<usize> {
    match sentinel(limit) {
        Some(cap) => Ok(total.saturating_add(value).min(cap)),
        None => total
            .checked_add(value)
            .ok_or_else(|| scratch_error("join")),
    }
}

#[derive(Clone)]
pub(super) struct Counts {
    pub(super) rows: Arc<CountRows>,
    pub(super) units: usize,
    pub(super) total: usize,
}

pub(super) struct CountRows {
    values: Vec<usize>,
    // Dynamic count backing must drop before its prepaid input-control lease.
    _credit: Arc<MemoryReservation>,
}

impl std::ops::Deref for CountRows {
    type Target = [usize];

    fn deref(&self) -> &Self::Target {
        &self.values
    }
}

impl Counts {
    pub(super) fn ordered(
        values: Vec<usize>,
        limit: u64,
        credit: Arc<MemoryReservation>,
    ) -> Result<Self> {
        debug_assert!(!values.is_empty());
        let total = values
            .iter()
            .try_fold(0, |total, &value| add_count(total, value, limit))?;
        Ok(Self {
            units: values.len(),
            total,
            rows: Arc::new(CountRows {
                values,
                _credit: credit,
            }),
        })
    }
}
