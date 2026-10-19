use super::super::MatchedPair;
use super::inputs::ProbeInputs;
use super::{
    pairs::{FillFunding, PairFragment},
    partition::{self, CountFragment, Counts, WindowCache},
};
use crate::{
    Result,
    runtime::streaming::gather_work::{GatherStop, ParallelCpuWork},
};
use std::sync::Arc;

const STOP_GROUP_ITEMS: usize = 4_096;

/// A position pays for its first match (or miss); additional matches consume
/// the same budget so neither long runs nor repeated short runs delay stopping.
struct ProbeQuantum<'a> {
    stop: &'a GatherStop,
    remaining: usize,
}

impl<'a> ProbeQuantum<'a> {
    fn new(stop: &'a GatherStop) -> Self {
        Self { stop, remaining: 0 }
    }

    fn step(&mut self) -> Result<()> {
        if self.remaining == 0 {
            self.stop.check_abandoned()?;
            self.remaining = STOP_GROUP_ITEMS;
        }
        self.remaining -= 1;
        Ok(())
    }
}

pub(super) struct CountWork {
    pub(super) inputs: Arc<ProbeInputs>,
    pub(super) limit: u64,
}

impl ParallelCpuWork for CountWork {
    type Output = CountFragment;

    fn unit_count(&self) -> usize {
        self.inputs.partition.units()
    }

    fn run(&self, ordinal: usize, stop: &GatherStop) -> Result<CountFragment> {
        #[cfg(feature = "benchmark-internals")]
        let _phase = stop.join_cpu_trace().worker_stage(
            stop.join_cpu_operator(),
            "count_worker",
            self.inputs.partition.range(ordinal).len(),
        );
        #[cfg(test)]
        self.inputs.observe(super::ProbePhase::Count);
        #[cfg(test)]
        self.inputs
            .observe_unit(super::ProbePhase::Count, ordinal, false);
        let count = self.count_range(ordinal, stop);
        stop.check()?;
        #[cfg(test)]
        self.inputs
            .observe_unit(super::ProbePhase::Count, ordinal, true);
        count
    }
}

impl CountWork {
    fn count_range(&self, ordinal: usize, stop: &GatherStop) -> Result<CountFragment> {
        let mut count = 0usize;
        let sentinel = partition::sentinel(self.limit);
        let mut quantum = ProbeQuantum::new(stop);
        let mut windows = WindowCache::admit(&self.inputs, ordinal)?;
        let mut cursors = self.inputs.cursors();
        for position in self.inputs.partition.range(ordinal) {
            quantum.step()?;
            let bounds = self.inputs.window(position, &mut cursors);
            if let Some(windows) = &mut windows {
                windows.values.push(bounds);
            }
            count = partition::add_count(count, bounds.matches(), self.limit)?;
            if Some(count) == sentinel {
                windows = None;
                break;
            }
        }
        Ok(CountFragment { count, windows })
    }
}

pub(super) struct FillWork {
    pub(super) inputs: Arc<ProbeInputs>,
    pub(super) counts: Counts,
    pub(super) funding: FillFunding,
}

impl ParallelCpuWork for FillWork {
    type Output = PairFragment;

    fn unit_count(&self) -> usize {
        self.counts.units
    }

    fn run(&self, ordinal: usize, stop: &GatherStop) -> Result<PairFragment> {
        #[cfg(feature = "benchmark-internals")]
        let _phase = stop.join_cpu_trace().worker_stage(
            stop.join_cpu_operator(),
            "fill_worker",
            self.inputs.partition.range(ordinal).len(),
        );
        #[cfg(test)]
        self.inputs.observe(super::ProbePhase::Fill);
        #[cfg(test)]
        self.inputs
            .observe_unit(super::ProbePhase::Fill, ordinal, false);
        stop.check()?;
        let credit = self.funding.unit();
        let pairs = fill_range(&self.inputs, &self.counts, ordinal, stop)?;
        stop.check()?;
        #[cfg(test)]
        self.inputs
            .observe_unit(super::ProbePhase::Fill, ordinal, true);
        Ok(PairFragment { pairs, credit })
    }
}

pub(super) fn fill_range(
    inputs: &ProbeInputs,
    counts: &Counts,
    ordinal: usize,
    stop: &GatherStop,
) -> Result<super::super::matches::Selection> {
    let range = inputs.partition.range(ordinal);
    let mut pairs =
        super::super::matches::Selection::with_capacity(counts.count(ordinal), range.len());
    let mut quantum = ProbeQuantum::new(stop);
    let windows = counts.windows(inputs, ordinal);
    let mut cursors = windows.is_none().then(|| inputs.cursors());
    for (offset, position) in range.enumerate() {
        quantum.step()?;
        let bounds = match windows {
            Some(windows) => windows[offset],
            None => inputs.window(position, cursors.as_mut().expect("uncached window cursors")),
        };
        for (offset, opposite_index) in inputs.matches_bounds(position, bounds).enumerate() {
            if offset > 0 {
                quantum.step()?;
            }
            pairs.push(MatchedPair {
                pos: position,
                opposite_index,
            });
        }
    }
    debug_assert_eq!(pairs.len(), counts.count(ordinal));
    Ok(pairs)
}

/// Refused optional acceleration retains the already admitted serial pair lease.
/// This inline fallback owns full stop checks; no ticket waiter drives it.
pub(super) fn fill_all(
    inputs: &ProbeInputs,
    counts: &Counts,
    stop: &GatherStop,
) -> Result<super::super::matches::Selection> {
    let mut pairs =
        super::super::matches::Selection::with_capacity(counts.total, inputs.admitted.len());
    let mut work = 0usize;
    let mut cursors = None;
    for ordinal in 0..counts.units {
        let windows = counts.windows(inputs, ordinal);
        for (offset, pos) in inputs.partition.range(ordinal).enumerate() {
            if work.is_multiple_of(STOP_GROUP_ITEMS) {
                stop.check()?;
            }
            work += 1;
            let bounds = match windows {
                Some(windows) => windows[offset],
                None => inputs.window(pos, cursors.get_or_insert_with(|| inputs.cursors())),
            };
            for opposite_index in inputs.matches_bounds(pos, bounds) {
                if work.is_multiple_of(STOP_GROUP_ITEMS) {
                    stop.check()?;
                }
                work += 1;
                pairs.push(MatchedPair {
                    pos,
                    opposite_index,
                });
            }
        }
    }
    stop.check()?;
    debug_assert_eq!(pairs.len(), counts.total);
    Ok(pairs)
}
