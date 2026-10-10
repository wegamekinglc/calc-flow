use super::super::native_lookup::scratch_error;
use crate::Result;
use std::ops::Range;

const MAX_UNITS: usize = 8;

pub(super) fn units(rows: usize) -> usize {
    (rows / super::MIN_PROBE_ROWS).clamp(1, MAX_UNITS)
}

pub(super) fn range(rows: usize, ordinal: usize) -> Range<usize> {
    let units = units(rows);
    debug_assert!(ordinal < units);
    let width = rows / units;
    let remainder = rows % units;
    let start = ordinal * width + ordinal.min(remainder);
    start..start + width + usize::from(ordinal < remainder)
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

#[derive(Clone, Copy)]
pub(super) struct Counts {
    pub(super) rows: [usize; MAX_UNITS],
    pub(super) units: usize,
    pub(super) total: usize,
}

impl Counts {
    pub(super) fn ordered(values: Vec<usize>, limit: u64) -> Result<Self> {
        debug_assert!(!values.is_empty() && values.len() <= MAX_UNITS);
        let mut counts = Self {
            rows: [0; MAX_UNITS],
            units: values.len(),
            total: 0,
        };
        for (ordinal, value) in values.into_iter().enumerate() {
            counts.rows[ordinal] = value;
            counts.total = add_count(counts.total, value, limit)?;
        }
        Ok(counts)
    }
}
