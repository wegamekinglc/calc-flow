use super::super::{
    JoinTimeBounds,
    native_lookup::{NativeIndex, WindowBounds, scratch_error},
};
use super::inputs::ProbeInputs;
use crate::Result;
use datafusion::execution::memory_pool::MemoryReservation;
use std::ops::Range;
use std::sync::{Arc, Weak};

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
    values: Vec<CountFragment>,
    // Dynamic count backing must drop before its prepaid input-control lease.
    _credit: Arc<MemoryReservation>,
}

pub(super) struct CountFragment {
    pub(super) count: usize,
    pub(super) windows: Option<WindowCache>,
}

pub(super) struct WindowCache {
    pub(super) values: Vec<WindowBounds>,
    control: Weak<MemoryReservation>,
    index: Weak<NativeIndex>,
    range: Range<usize>,
    bounds: JoinTimeBounds,
    incoming_is_left: bool,
    // Bounds and weak allocation guards are destroyed before their lease.
    _credit: MemoryReservation,
}

impl WindowCache {
    pub(super) fn admit(inputs: &ProbeInputs, ordinal: usize) -> Result<Option<Self>> {
        let range = inputs.partition.range(ordinal);
        let guards = super::control::window_guards_bytes()?;
        let Some(bytes) = range
            .len()
            .checked_mul(size_of::<WindowBounds>())
            .and_then(|bytes| bytes.checked_add(guards))
        else {
            return Ok(None);
        };
        let control = inputs.control.as_ref().expect("prepaid count controls");
        let credit = control.new_empty();
        if credit.try_grow(bytes).is_err() {
            return Ok(None);
        }
        Ok(Some(Self {
            values: Vec::with_capacity(range.len()),
            control: Arc::downgrade(control),
            index: Arc::downgrade(inputs.index.as_ref().expect("owned native index")),
            range,
            bounds: inputs.bounds,
            incoming_is_left: inputs.incoming_is_left,
            _credit: credit,
        }))
    }

    fn get(&self, inputs: &ProbeInputs, ordinal: usize) -> Option<&[WindowBounds]> {
        let range = inputs.partition.range(ordinal);
        (self.range == range
            && self.values.len() == range.len()
            && self.bounds == inputs.bounds
            && self.incoming_is_left == inputs.incoming_is_left
            && inputs
                .control
                .as_ref()
                .is_some_and(|control| self.control.as_ptr() == Arc::as_ptr(control))
            && inputs
                .index
                .as_ref()
                .is_some_and(|index| self.index.as_ptr() == Arc::as_ptr(index)))
        .then_some(self.values.as_slice())
    }
}

impl Counts {
    pub(super) fn ordered(
        values: Vec<CountFragment>,
        limit: u64,
        credit: Arc<MemoryReservation>,
    ) -> Result<Self> {
        debug_assert!(!values.is_empty());
        let total = values
            .iter()
            .try_fold(0, |total, value| add_count(total, value.count, limit))?;
        Ok(Self {
            units: values.len(),
            total,
            rows: Arc::new(CountRows {
                values,
                _credit: credit,
            }),
        })
    }

    pub(super) fn count(&self, ordinal: usize) -> usize {
        self.rows.values[ordinal].count
    }

    pub(super) fn windows(&self, inputs: &ProbeInputs, ordinal: usize) -> Option<&[WindowBounds]> {
        self.rows.values[ordinal]
            .windows
            .as_ref()?
            .get(inputs, ordinal)
    }

    pub(super) fn discard_windows(&mut self) -> bool {
        let rows = Arc::get_mut(&mut self.rows).expect("settled count results are uniquely owned");
        let mut discarded = false;
        for fragment in &mut rows.values {
            discarded |= fragment.windows.take().is_some();
        }
        discarded
    }
}
