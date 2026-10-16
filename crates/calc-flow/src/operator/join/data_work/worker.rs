use super::super::MatchedPair;
use super::inputs::ProbeInputs;
use super::{
    pairs::{FillFunding, PairFragment},
    partition::{self, Counts},
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
    type Output = usize;

    fn unit_count(&self) -> usize {
        partition::units(self.inputs.admitted.len())
    }

    fn run(&self, ordinal: usize, stop: &GatherStop) -> Result<usize> {
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
    fn count_range(&self, ordinal: usize, stop: &GatherStop) -> Result<usize> {
        let mut count = 0usize;
        let sentinel = partition::sentinel(self.limit);
        let mut quantum = ProbeQuantum::new(stop);
        for position in partition::range(self.inputs.admitted.len(), ordinal) {
            quantum.step()?;
            count = partition::add_count(count, self.inputs.count(position), self.limit)?;
            if Some(count) == sentinel {
                break;
            }
        }
        Ok(count)
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
        #[cfg(test)]
        self.inputs.observe(super::ProbePhase::Fill);
        #[cfg(test)]
        self.inputs
            .observe_unit(super::ProbePhase::Fill, ordinal, false);
        stop.check()?;
        let credit = self.funding.unit();
        let pairs = fill_range(&self.inputs, self.counts, ordinal, stop)?;
        stop.check()?;
        #[cfg(test)]
        self.inputs
            .observe_unit(super::ProbePhase::Fill, ordinal, true);
        Ok(PairFragment { pairs, credit })
    }
}

pub(super) fn fill_range(
    inputs: &ProbeInputs,
    counts: Counts,
    ordinal: usize,
    stop: &GatherStop,
) -> Result<super::super::matches::Selection> {
    let range = partition::range(inputs.admitted.len(), ordinal);
    let mut pairs =
        super::super::matches::Selection::with_capacity(counts.rows[ordinal], range.len());
    let mut quantum = ProbeQuantum::new(stop);
    for position in range {
        quantum.step()?;
        for (offset, opposite_index) in inputs.matches(position).enumerate() {
            if offset > 0 {
                quantum.step()?;
            }
            pairs.push(MatchedPair {
                pos: position,
                opposite_index,
            });
        }
    }
    debug_assert_eq!(pairs.len(), counts.rows[ordinal]);
    Ok(pairs)
}

/// Refused optional acceleration retains the already admitted serial pair lease.
/// This inline fallback owns full stop checks; no ticket waiter drives it.
pub(super) fn fill_all(
    inputs: &ProbeInputs,
    count: usize,
    stop: &GatherStop,
) -> Result<super::super::matches::Selection> {
    let mut pairs = super::super::matches::Selection::with_capacity(count, inputs.admitted.len());
    let mut work = 0usize;
    for pos in 0..inputs.admitted.len() {
        if work % STOP_GROUP_ITEMS == 0 {
            stop.check()?;
        }
        work += 1;
        for opposite_index in inputs.matches(pos) {
            if work % STOP_GROUP_ITEMS == 0 {
                stop.check()?;
            }
            work += 1;
            pairs.push(MatchedPair {
                pos,
                opposite_index,
            });
        }
    }
    stop.check()?;
    debug_assert_eq!(pairs.len(), count);
    Ok(pairs)
}
