use datafusion::arrow::datatypes::SchemaRef;
use datafusion::execution::memory_pool::MemoryReservation;

use super::super::{OperatorStateSnapshot, StreamJoinSpec};
use super::budget;
use super::candidate::{Config, Decoder, Prepared};
use super::ipc::accounting::{add, product, sum, vector_peak};
use crate::Result;
use crate::runtime::streaming::gather_work::{
    GatherStop, OwnedCpuWork, RetirementGuard, cleanup_control_bytes,
};

pub(super) struct OwnedConfig {
    pub(super) name: String,
    pub(super) spec: StreamJoinSpec,
    pub(super) schemas: [SchemaRef; 2],
    pub(super) keys: [Vec<usize>; 2],
    pub(super) time_indices: [usize; 2],
    #[cfg(test)]
    pub(super) metadata_hook: Option<super::super::MetadataTestHook>,
    #[cfg(test)]
    pub(super) decoded_hook: Option<super::super::DecodedRowTestHook>,
}

impl OwnedConfig {
    fn borrowed(&self) -> Config<'_> {
        Config {
            name: &self.name,
            spec: &self.spec,
            schemas: self.schemas.clone(),
            keys: [&self.keys[0], &self.keys[1]],
            time_indices: self.time_indices,
            #[cfg(test)]
            metadata_hook: self.metadata_hook.as_ref(),
            #[cfg(test)]
            decoded_hook: self.decoded_hook.as_ref(),
            #[cfg(test)]
            owned_work: true,
        }
    }
}

// Input data dies before its credit. The retirement Vec has independent credit
// through its deallocation, and reports retirement after the input refund.
pub(super) struct RestoreWork {
    pub(super) snapshot: OperatorStateSnapshot,
    pub(super) config: OwnedConfig,
    pub(super) input_credit: MemoryReservation,
    pub(super) retirements: Vec<RetirementGuard>,
    pub(super) _retirement_credit: MemoryReservation,
}

impl OwnedCpuWork for RestoreWork {
    type Output = Prepared;

    fn control_bytes(&self) -> Result<usize> {
        sum(&[
            size_of::<Self>(),
            cleanup_control_bytes::<Prepared>(),
            budget::registration_bytes()?,
        ])
    }

    fn run(mut self, stop: &GatherStop) -> Result<Prepared> {
        stop.check()?;
        let config = self.config.borrowed();
        let check = || stop.check();
        let outcome = Decoder::new(
            &self.snapshot,
            &config,
            &self.input_credit,
            std::mem::take(&mut self.retirements),
            &check,
        )
        .and_then(Decoder::decode);
        stop.check()?;
        outcome
    }
}

// Covers only new config copies and a push-built retirement Vec. The caller
// retains the transferred snapshot's paid owner and pays submission controls.
pub(super) fn initial_input_bytes(
    name: &str,
    spec: &StreamJoinSpec,
    keys: [&[usize]; 2],
    segment_count: usize,
) -> Result<usize> {
    let key_rows = add(keys[0].len(), keys[1].len())?;
    sum(&[
        size_of::<RestoreWork>(),
        name.len(),
        budget::spec_bytes(spec)?,
        product(key_rows, size_of::<usize>())?,
        retirement_bytes(segment_count)?,
    ])
}

pub(super) fn retirement_bytes(segment_count: usize) -> Result<usize> {
    vector_peak::<RetirementGuard>(add(segment_count, 1)?)
}
