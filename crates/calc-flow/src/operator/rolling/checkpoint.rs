//! Retained-only checkpoint capture for legacy rolling outputs.

use super::{
    BTreeMap, Epoch, RollingOperator, RollingSnapshotMetadata, SegmentDescriptor, StateInventory,
    Value, budget, checkpoint_mismatch, format_error, internal_error,
};
use crate::{OperatorStateSnapshot, Result, StateSegment};

type Segments = BTreeMap<String, StateSegment>;

impl RollingOperator {
    pub(super) fn checkpoint_legacy(&mut self, epoch: Epoch) -> Result<OperatorStateSnapshot> {
        if self.state.deferred_history {
            return Err(checkpoint_mismatch(
                "rolling checkpointing is disabled for the deferred-history stream state",
            ));
        }
        self.validate_legacy_epoch(epoch)?;
        self.materialize_legacy_capture()?;
        self.recount_legacy_capture()?;
        let (descriptor, segments) = self.capture_legacy_segment(epoch)?;
        let snapshot = self.legacy_snapshot(epoch, descriptor, segments)?;
        self.state.owned_mut()?.last_checkpoint_epoch = Some(epoch);
        Ok(snapshot)
    }

    fn validate_legacy_epoch(&self, epoch: Epoch) -> Result<()> {
        if self
            .state
            .last_checkpoint_epoch
            .is_some_and(|previous| epoch <= previous)
        {
            return Err(checkpoint_mismatch(
                "rolling checkpoint epoch did not advance strictly",
            ));
        }
        Ok(())
    }

    fn materialize_legacy_capture(&mut self) -> Result<()> {
        self.materialize_ordered_buffer(None)?;
        self.state
            .owned_mut()?
            .histories
            .materialize_columnar(&self.compiled, &self.name, None)
    }

    fn recount_legacy_capture(&mut self) -> Result<()> {
        let buffered = budget::buffered_charge(self.state.buffer.values(), &self.name)?;
        let ordered = budget::ordered_buffer_charge(
            self.state.ordered.records(),
            &self.compiled,
            &self.name,
        )?
        .ok_or_else(|| internal_error("rolling ordered state has unsupported state charge"))?;
        let other = buffered.checked_add(ordered, &self.name)?;
        let (histories, cache) = self.state.histories.prepare_charge_cache(&self.name)?;
        let charge = other.checked_add(histories, &self.name)?;
        self.check_state_budget(charge, &self.name)?;
        let state = self.state.owned_mut()?;
        state.histories.install_charge_cache(cache);
        state.charge = charge;
        Ok(())
    }

    fn capture_legacy_segment(
        &self,
        epoch: Epoch,
    ) -> Result<(Option<SegmentDescriptor>, Segments)> {
        let Some((id, bytes)) = self.encode_state(epoch)? else {
            return Ok((None, BTreeMap::new()));
        };
        let segment = StateSegment::new(bytes);
        let descriptor = self.snapshot_segment_descriptor(epoch, &id, &segment)?;
        Ok((Some(descriptor), BTreeMap::from([(id, segment)])))
    }

    fn legacy_snapshot(
        &self,
        epoch: Epoch,
        descriptor: Option<SegmentDescriptor>,
        segments: Segments,
    ) -> Result<OperatorStateSnapshot> {
        let inventory = StateInventory::new(descriptor.into_iter().collect())
            .map_err(|error| checkpoint_mismatch(error.to_string()))?;
        let metadata = self.legacy_capture_metadata(epoch, &inventory);
        let Value::Object(inline) =
            serde_json::to_value(metadata).map_err(|error| format_error(&error))?
        else {
            return Err(internal_error(
                "rolling snapshot metadata did not serialize as an object",
            ));
        };
        Ok(OperatorStateSnapshot {
            inline_metadata: inline.into_iter().collect(),
            segments,
        })
    }

    fn legacy_capture_metadata(
        &self,
        epoch: Epoch,
        inventory: &StateInventory,
    ) -> RollingSnapshotMetadata {
        RollingSnapshotMetadata {
            state_layout_version: self.compiled.state_layout_version,
            configuration_hash: self.compiled.configuration_hash.clone(),
            state_schema_fingerprint: self.compiled.state_schema_fingerprint.clone(),
            kernel_fingerprint: Some(self.compiled.kernel_plan.fingerprint().to_owned()),
            numerical_profile: Some(self.compiled.kernel_plan.numerical_profile().to_owned()),
            epoch,
            pipeline_fingerprint: self.state.pipeline_fingerprint.clone(),
            operator_id: self.state.operator_id.clone(),
            last_input_watermark: self.state.last_input_watermark,
            next_output_sequence: self.state.next_output_sequence,
            ended: self.state.ended,
            metrics: self.state.metrics,
            segment_inventory: inventory.segments().to_vec(),
            late_output: super::super::late_output::LateOutputSnapshot::new(
                self.spec.late_policy,
                self.state.next_late_output_sequence,
            ),
        }
    }
}
