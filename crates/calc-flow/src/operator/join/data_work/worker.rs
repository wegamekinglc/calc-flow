use super::super::{MatchedPair, native_lookup::scratch_error};
use super::inputs::ProbeInputs;
use crate::{
    Result,
    runtime::streaming::gather_work::{GatherStop, ParallelCpuWork},
};
use datafusion::execution::memory_pool::MemoryReservation;
use parking_lot::Mutex;
use std::sync::Arc;

pub(super) struct PairOutput {
    pub(super) pairs: Vec<MatchedPair>,
    pub(super) credit: MemoryReservation,
}

pub(super) struct CountWork {
    pub(super) inputs: Arc<ProbeInputs>,
    pub(super) limit: u64,
}

impl ParallelCpuWork for CountWork {
    type Output = usize;

    fn unit_count(&self) -> usize {
        1
    }

    fn run(&self, ordinal: usize, stop: &GatherStop) -> Result<usize> {
        debug_assert_eq!(ordinal, 0);
        #[cfg(test)]
        self.inputs.observe(super::ProbePhase::Count);
        let limit = usize::try_from(self.limit).unwrap_or(usize::MAX);
        let mut count = 0usize;
        for position in 0..self.inputs.admitted.len() {
            stop.check()?;
            let remaining = limit.saturating_sub(count).saturating_add(1);
            count = count
                .checked_add(self.inputs.count(position).min(remaining))
                .ok_or_else(|| scratch_error("join"))?;
            if count > limit {
                break;
            }
        }
        Ok(count)
    }
}

pub(super) struct FillWork {
    pub(super) inputs: Arc<ProbeInputs>,
    pub(super) count: usize,
    pub(super) credit: Mutex<Option<MemoryReservation>>,
}

impl ParallelCpuWork for FillWork {
    type Output = PairOutput;

    fn unit_count(&self) -> usize {
        1
    }

    fn run(&self, ordinal: usize, stop: &GatherStop) -> Result<PairOutput> {
        debug_assert_eq!(ordinal, 0);
        #[cfg(test)]
        self.inputs.observe(super::ProbePhase::Fill);
        stop.check()?;
        let credit = self
            .credit
            .lock()
            .take()
            .expect("single fill unit owns pair funding");
        let mut pairs = Vec::with_capacity(self.count);
        for position in 0..self.inputs.admitted.len() {
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
        debug_assert_eq!(pairs.len(), self.count);
        Ok(PairOutput { pairs, credit })
    }
}
