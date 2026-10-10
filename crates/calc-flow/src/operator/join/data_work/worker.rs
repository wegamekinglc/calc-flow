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
        for position in partition::range(self.inputs.admitted.len(), ordinal) {
            stop.check()?;
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
        let mut pairs = Vec::with_capacity(self.counts.rows[ordinal]);
        for position in partition::range(self.inputs.admitted.len(), ordinal) {
            stop.check()?;
            for (ordinal, opposite_index) in self.inputs.matches(position).enumerate() {
                if ordinal.is_multiple_of(256) {
                    stop.check()?;
                }
                pairs.push(MatchedPair {
                    pos: position,
                    opposite_index,
                });
            }
        }
        debug_assert_eq!(pairs.len(), self.counts.rows[ordinal]);
        stop.check()?;
        #[cfg(test)]
        self.inputs
            .observe_unit(super::ProbePhase::Fill, ordinal, true);
        Ok(PairFragment { pairs, credit })
    }
}
