use super::{StateOwner, empty};
use crate::operator::rolling::{
    DecodedRollingState, RollingOperator, RollingSnapshotMetadata, RollingStreamState,
    budget::{self, StateCharge},
    state_codec,
};
use crate::{CalcFlowError, OperatorStateSnapshot, Result, StreamOperator};

impl RollingOperator {
    pub(in crate::operator::rolling) fn restore_legacy(
        &mut self,
        snapshot: &OperatorStateSnapshot,
    ) -> Result<()> {
        if empty(snapshot) {
            return StreamOperator::reset(self);
        }
        self.restore_legacy_nonempty(snapshot)
    }
    fn restore_legacy_nonempty(&mut self, snapshot: &OperatorStateSnapshot) -> Result<()> {
        let metadata = state_codec::parse_snapshot_metadata(snapshot)?;
        state_codec::validate_snapshot_metadata(&metadata, &self.compiled, snapshot)?;
        let next_late = crate::operator::late_output::LateOutputSnapshot::restore_sequence(
            self.spec.late_policy,
            metadata.late_output.as_ref(),
        )?;
        let mut restored = self
            .decode_state(&metadata, snapshot)
            .map_err(|error| self.legacy_restore_error(error))?;
        let charge = self.legacy_restore_charge(&mut restored)?;
        self.state = StateOwner::new(
            legacy_state(restored, metadata, next_late, charge),
            self.indicator_resources.is_some(),
        );
        Ok(())
    }
    fn legacy_restore_error(&self, error: CalcFlowError) -> CalcFlowError {
        if super::super::is_layout(self.compiled.state_layout_version) {
            state_codec::layout_five_restore_error(error, &self.name)
        } else {
            error
        }
    }
    fn legacy_restore_charge(&self, restored: &mut DecodedRollingState) -> Result<StateCharge> {
        let buffered = budget::buffered_charge(restored.buffer.values(), &self.name)?;
        let (histories, cache) = restored.histories.prepare_charge_cache(&self.name)?;
        let charge = buffered.checked_add(histories, &self.name)?;
        if !self.state_budget.allows(charge.rows, charge.bytes) {
            return Err(crate::operator::checkpoint::checkpoint_mismatch(
                "restored rolling state exceeds the configured state budget",
            ));
        }
        restored.histories.install_charge_cache(cache);
        Ok(charge)
    }
}
fn legacy_state(
    restored: DecodedRollingState,
    metadata: RollingSnapshotMetadata,
    next_late: u64,
    charge: StateCharge,
) -> RollingStreamState {
    RollingStreamState {
        buffer: restored.buffer,
        ordered: crate::operator::rolling::ordered_stream::OrderedStreamBuffer::default(),
        histories: restored.histories,
        last_input_watermark: metadata.last_input_watermark,
        next_output_sequence: metadata.next_output_sequence,
        next_late_output_sequence: next_late,
        late_output_failed: false,
        ended: metadata.ended,
        metrics: metadata.metrics,
        pipeline_fingerprint: metadata.pipeline_fingerprint,
        operator_id: metadata.operator_id,
        last_checkpoint_epoch: Some(metadata.epoch),
        typed_kernel_state: None,
        charge,
    }
}
