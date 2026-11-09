//! Intrinsic admission for the new rolling checkpoint format.

use std::{collections::BTreeMap, sync::Arc};

use super::super::{
    CompiledRollingSpec, LatePolicySpec, ROLLING_INDICATOR_STATE_LAYOUT_VERSION, RollingOperator,
    RollingSnapshotMetadata, state_codec,
};
use super::{Resources, resources::Credit};
use crate::{
    CalcFlowError, JsonMap, OperatorManifestEntry, OperatorStateSnapshot, Result, StateBudget,
    StateHandle, StateSegment,
    array::RollingWorkers,
    pipeline::OperatorCheckpointCapability,
    runtime::streaming::StreamJobControl,
    state::{SegmentDescriptor, SegmentKind, StateInventory},
};

mod preloaded;

pub(crate) struct IndicatorRollingRecord {
    node: String,
    graph: Arc<str>,
    compiled: Arc<CompiledRollingSpec>,
    policy: LatePolicySpec,
    budget: StateBudget,
    resources: Option<Resources>,
    workers: Arc<RollingWorkers>,
    capability: OperatorCheckpointCapability,
    control: Arc<StreamJobControl>,
}

pub(crate) struct RollingIoReservation {
    _credit: Option<Credit>,
}

struct AdmittedMetadata {
    value: RollingSnapshotMetadata,
    _credit: Option<Credit>,
}
impl std::ops::Deref for AdmittedMetadata {
    type Target = RollingSnapshotMetadata;
    fn deref(&self) -> &Self::Target {
        &self.value
    }
}

impl RollingOperator {
    pub(crate) fn native_checkpoint_record(
        &self,
        node: &str,
        graph: Arc<str>,
        capability: OperatorCheckpointCapability,
        control: Arc<StreamJobControl>,
    ) -> Option<IndicatorRollingRecord> {
        (self.compiled.state_layout_version == ROLLING_INDICATOR_STATE_LAYOUT_VERSION).then(|| {
            IndicatorRollingRecord {
                node: node.to_owned(),
                graph,
                compiled: Arc::clone(&self.compiled),
                policy: self.spec.late_policy,
                budget: self.state_budget,
                resources: self.indicator_resources.clone(),
                workers: self.native_checkpoint_workers(),
                capability,
                control,
            }
        })
    }
}

impl IndicatorRollingRecord {
    pub(crate) fn node(&self) -> &str {
        &self.node
    }
    pub(crate) fn control(&self) -> &Arc<StreamJobControl> {
        &self.control
    }
    pub(crate) fn workers(&self) -> &Arc<RollingWorkers> {
        &self.workers
    }

    pub(crate) fn worker_identity(
        &self,
    ) -> Result<Option<Arc<crate::array::RollingWorkerIdentity>>> {
        self.resources
            .as_ref()
            .map(|resources| super::work_identity::reserve(resources, &self.node, 0, self.budget))
            .transpose()
    }

    pub(crate) fn mismatch(&self, field: &str, message: &str) -> CalcFlowError {
        CalcFlowError::CheckpointMismatch {
            message: format!("rolling operator {:?} state.{field}: {message}", self.node),
        }
    }

    fn metadata(&self, inline: &JsonMap) -> Result<AdmittedMetadata> {
        if let Some(resources) = &self.resources {
            return self.controlled_metadata(inline, resources);
        }
        let mut decoded = self.capability.decode_snapshot(
            &self.node,
            OperatorStateSnapshot {
                inline_metadata: inline.clone(),
                segments: BTreeMap::new(),
            },
        )?;
        decoded
            .inline_metadata
            .remove(crate::pipeline::OUTPUT_FRONTIER_METADATA_KEY_V1);
        let metadata = state_codec::parse_snapshot_metadata(&decoded)
            .map_err(|error| self.metadata_error(error))?;
        self.validate_metadata(&metadata)?;
        Ok(AdmittedMetadata {
            value: metadata,
            _credit: None,
        })
    }

    fn controlled_metadata(
        &self,
        inline: &JsonMap,
        resources: &Resources,
    ) -> Result<AdmittedMetadata> {
        let quote = self.metadata_quote(inline)?;
        let credit = self.metadata_credit(resources, quote.peak)?;
        let metadata = super::restore::metadata::parse_entries(Self::metadata_fields(inline))?;
        self.validate_metadata(&metadata)?;
        self.control.check_cancelled()?;
        Ok(AdmittedMetadata {
            value: metadata,
            _credit: Some(credit),
        })
    }

    fn metadata_fields(
        inline: &JsonMap,
    ) -> impl Iterator<Item = (&String, &serde_json::Value)> + Clone {
        OperatorCheckpointCapability::restored_metadata_fields(inline)
    }

    fn metadata_quote(&self, inline: &JsonMap) -> Result<super::restore::metadata::Quote> {
        self.control.check_cancelled()?;
        self.capability
            .validate_snapshot_fields(&self.node, inline, true)?;
        self.validate_frontier(inline)?;
        super::restore::metadata::quote_entries(Self::metadata_fields(inline), &mut || {
            self.control.check_cancelled()
        })
    }

    fn validate_frontier(&self, inline: &JsonMap) -> Result<()> {
        if inline.contains_key(crate::pipeline::OUTPUT_FRONTIER_METADATA_KEY_V1) {
            return Err(self.mismatch("metadata", "unexpected runtime output frontier"));
        }
        Ok(())
    }

    fn metadata_credit(&self, resources: &Resources, bytes: u64) -> Result<Credit> {
        resources
            .reserve(0, bytes, self.budget, &self.node)
            .map_err(|error| match error {
                CalcFlowError::Operator { .. } => {
                    self.mismatch("metadata", "exceeds the configured budget")
                }
                error => error,
            })
    }

    fn validate_metadata(&self, metadata: &RollingSnapshotMetadata) -> Result<()> {
        state_codec::validate_metadata_header(metadata, &self.compiled)?;
        self.validate_identity(metadata)?;
        super::super::super::late_output::LateOutputSnapshot::restore_sequence(
            self.policy,
            metadata.late_output.as_ref(),
        )?;
        self.inventory(metadata)
    }

    fn metadata_error(&self, error: CalcFlowError) -> CalcFlowError {
        match error {
            CalcFlowError::Format { .. } => self.mismatch("metadata", "invalid inline metadata"),
            error => error,
        }
    }

    fn validate_identity(&self, metadata: &RollingSnapshotMetadata) -> Result<()> {
        if metadata.state_layout_version != ROLLING_INDICATOR_STATE_LAYOUT_VERSION
            || metadata.pipeline_fingerprint.as_deref() != Some(self.graph.as_ref())
            || metadata.operator_id.as_deref() != Some(self.node.as_str())
        {
            return Err(self.mismatch("identity", "layout, graph or operator differs"));
        }
        Ok(())
    }

    fn inventory(&self, metadata: &RollingSnapshotMetadata) -> Result<()> {
        let inventory = StateInventory::new(metadata.segment_inventory.clone())?;
        if inventory.segments().len() > 1 {
            return Err(self.mismatch("segments", "requires zero or one base segment"));
        }
        self.validate_empty(metadata, &inventory)?;
        for descriptor in inventory.segments() {
            self.descriptor(metadata, descriptor)?;
        }
        Ok(())
    }

    fn validate_empty(
        &self,
        metadata: &RollingSnapshotMetadata,
        inventory: &StateInventory,
    ) -> Result<()> {
        if self.resources.is_some()
            && inventory.segments().is_empty()
            && metadata.next_output_sequence > 0
        {
            return Err(self.mismatch(
                "segments",
                "emitted indicator rows require retained entity state",
            ));
        }
        Ok(())
    }

    fn descriptor(
        &self,
        metadata: &RollingSnapshotMetadata,
        descriptor: &SegmentDescriptor,
    ) -> Result<()> {
        if descriptor.kind != SegmentKind::Base
            || descriptor.state_layout_version != ROLLING_INDICATOR_STATE_LAYOUT_VERSION
            || descriptor.schema_fingerprint != self.compiled.state_schema_fingerprint
            || descriptor.handle.operator_id() != self.node
            || descriptor.handle.epoch() > metadata.epoch
        {
            return Err(self.mismatch(
                "segments",
                "base descriptor identity, layout or epoch differs",
            ));
        }
        let expected = format!("base-{:020}-00000000", descriptor.handle.epoch().as_u64());
        if descriptor.handle.segment_id() != expected {
            return Err(self.mismatch("segments", "base segment ID differs from its epoch"));
        }
        Ok(())
    }

    pub(crate) fn handle<'a>(
        &self,
        entry: &'a OperatorManifestEntry,
    ) -> Result<Option<&'a StateHandle>> {
        let metadata = self.metadata(&entry.inline_metadata)?;
        if entry.segments.len() != metadata.segment_inventory.len() {
            return Err(self.mismatch("segments", "manifest and inline inventories differ"));
        }
        let Some(handle) = entry.segments.first() else {
            return Ok(None);
        };
        let declared = &metadata.segment_inventory[0].handle;
        self.validate_handle(handle, declared)?;
        Ok(Some(handle))
    }

    fn validate_handle(&self, handle: &StateHandle, declared: &StateHandle) -> Result<()> {
        handle.validate_owner(&self.node)?;
        // Managed storage remaps the physical path, preserving the logical segment.
        if handle.segment_id() != declared.segment_id()
            || handle.epoch() != declared.epoch()
            || handle.byte_len() != declared.byte_len()
            || handle.sha256() != declared.sha256()
        {
            return Err(self.mismatch("segments", "manifest and inline base handles differ"));
        }
        usize::try_from(handle.byte_len())
            .map(|_| ())
            .map_err(|_| self.mismatch("encoded_bytes", "exceeds usize"))
    }

    pub(crate) fn admit_entry(&self, entry: &OperatorManifestEntry) -> Result<()> {
        if let Some(handle) = self.handle(entry)? {
            self.preflight_encoded(handle.byte_len())?;
        }
        Ok(())
    }

    pub(crate) fn admit_retained(&self, entry: &OperatorManifestEntry) -> Result<()> {
        // Maintenance verifies files in chunks; it does not retain encoded input.
        self.handle(entry).map(|_| ())
    }

    fn preflight_encoded(&self, length: u64) -> Result<()> {
        if let Some(resources) = &self.resources {
            let bytes = length
                .checked_add(64)
                .and_then(|size| size.checked_add(Self::io_scratch()))
                .ok_or_else(|| self.mismatch("encoded_bytes", "byte count overflowed u64"))?;
            let _credit = resources
                .reserve(0, bytes, self.budget, &self.node)
                .map_err(|error| match error {
                    CalcFlowError::Operator { message, .. } => {
                        self.mismatch("peak_bytes", &message)
                    }
                    error => error,
                })?;
        }
        Ok(())
    }

    pub(crate) fn admit_snapshot(&self, snapshot: &OperatorStateSnapshot) -> Result<()> {
        let metadata = self.metadata(&snapshot.inline_metadata)?;
        let expected: Vec<_> = metadata
            .segment_inventory
            .iter()
            .map(|item| item.handle.segment_id())
            .collect();
        let actual: Vec<_> = snapshot.segments.keys().map(String::as_str).collect();
        if expected != actual {
            return Err(self.mismatch("segments", "captured inventory differs"));
        }
        for descriptor in &metadata.segment_inventory {
            let segment = &snapshot.segments[descriptor.handle.segment_id()];
            if u64::try_from(segment.bytes().len()).ok() != Some(descriptor.handle.byte_len())
                || segment.sha256() != descriptor.handle.sha256()
            {
                return Err(
                    self.mismatch("segments", "captured bytes differ from their base handle")
                );
            }
        }
        Ok(())
    }

    pub(crate) fn retain_for_io(&self, segment: &StateSegment) -> Result<StateSegment> {
        match &self.resources {
            Some(resources) => resources.retain(segment.clone(), 0, self.budget, &self.node),
            None => Ok(segment.clone()),
        }
    }

    pub(crate) fn reserve_io(&self) -> Result<RollingIoReservation> {
        let credit = self
            .resources
            .as_ref()
            .map(|resources| resources.reserve(0, Self::io_scratch(), self.budget, &self.node))
            .transpose()?;
        Ok(RollingIoReservation { _credit: credit })
    }

    fn io_scratch() -> u64 {
        // The certified local verifier uses one 4096-byte chunk and fixed SHA state.
        u64::try_from(4096 + 64 + StateSegment::controlled_scratch_bytes())
            .expect("fixed IO scratch fits u64")
    }
}
