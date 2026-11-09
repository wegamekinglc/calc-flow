use std::collections::BTreeMap;

use super::{Capture, RollingOperator};
use crate::operator::rolling::{RollingSnapshotMetadata, checkpoint_mismatch};
use crate::state::{SegmentDescriptor, SegmentKind, StateInventory};
use crate::{Epoch, OperatorStateSnapshot, Result, StateHandle};

impl RollingOperator {
    pub(in super::super::super) fn capture_prepared(
        &mut self,
        epoch: Epoch,
    ) -> Result<OperatorStateSnapshot> {
        let _guard = self.try_native_checkpoint_guard(self.checkpoint_identity())?;
        self.validate_capture_epoch(epoch)?;
        let source = self.indicator_state_packet().expect("indicator packet");
        let capture = self.indicator_capture.as_ref().filter(|capture| {
            capture.state.ptr_eq(&std::sync::Arc::downgrade(&source))
        }).ok_or_else(|| checkpoint_mismatch("state.preparation: prepare_checkpoint_async is required after the latest state change"))?;
        let snapshot = self.capture_snapshot(epoch, capture)?;
        self.indicator_checkpoint_epoch = Some(epoch);
        Ok(snapshot)
    }

    fn validate_capture_epoch(&self, epoch: Epoch) -> Result<()> {
        if self
            .indicator_checkpoint_epoch
            .is_some_and(|previous| epoch <= previous)
        {
            return Err(checkpoint_mismatch(
                "rolling checkpoint epoch did not advance strictly",
            ));
        }
        Ok(())
    }

    fn capture_snapshot(&self, epoch: Epoch, capture: &Capture) -> Result<OperatorStateSnapshot> {
        let descriptor = capture.descriptor(epoch, &self.compiled.state_schema_fingerprint)?;
        let inventory = StateInventory::new(descriptor.into_iter().collect())?;
        let metadata = self.capture_metadata(epoch, capture, &inventory);
        let serde_json::Value::Object(inline) = serde_json::to_value(metadata)
            .map_err(|_| checkpoint_mismatch("rolling state.metadata could not be encoded"))?
        else {
            return Err(checkpoint_mismatch(
                "rolling state.metadata is not an object",
            ));
        };
        let segments = capture
            .segment
            .as_ref()
            .map(|segment| (segment_id(epoch), segment.clone()))
            .into_iter()
            .collect::<BTreeMap<_, _>>();
        Ok(OperatorStateSnapshot {
            inline_metadata: inline.into_iter().collect(),
            segments,
        })
    }

    fn capture_metadata(
        &self,
        epoch: Epoch,
        capture: &Capture,
        inventory: &StateInventory,
    ) -> RollingSnapshotMetadata {
        RollingSnapshotMetadata {
            state_layout_version: self.compiled.state_layout_version,
            configuration_hash: self.compiled.configuration_hash.clone(),
            state_schema_fingerprint: self.compiled.state_schema_fingerprint.clone(),
            kernel_fingerprint: Some(self.compiled.kernel_plan.fingerprint().into()),
            numerical_profile: Some(self.compiled.kernel_plan.numerical_profile().into()),
            epoch,
            pipeline_fingerprint: Some(capture.graph.clone()),
            operator_id: Some(capture.node.as_str().to_owned()),
            last_input_watermark: self.state.last_input_watermark,
            next_output_sequence: self.state.next_output_sequence,
            ended: self.state.ended,
            metrics: self.state.metrics,
            segment_inventory: inventory.segments().to_vec(),
            late_output: crate::operator::late_output::LateOutputSnapshot::new(
                self.spec.late_policy,
                self.state.next_late_output_sequence,
            ),
        }
    }
}

impl Capture {
    fn descriptor(&self, epoch: Epoch, schema: &str) -> Result<Option<SegmentDescriptor>> {
        let Some(segment) = &self.segment else {
            return Ok(None);
        };
        let id = segment_id(epoch);
        let path = format!(
            "committed/{}/{:020}-{id}.arrow",
            self.node.as_str(),
            epoch.as_u64()
        );
        let length = u64::try_from(segment.bytes().len())
            .map_err(|_| checkpoint_mismatch("rolling state.encoded_bytes exceeds u64"))?;
        Ok(Some(SegmentDescriptor {
            kind: SegmentKind::Base,
            state_layout_version: 5,
            schema_fingerprint: schema.into(),
            handle: StateHandle::new(&self.node, epoch, &id, &path, length, segment.sha256())?,
        }))
    }
}
fn segment_id(epoch: Epoch) -> String {
    format!("base-{:020}-00000000", epoch.as_u64())
}
