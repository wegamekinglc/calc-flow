use std::{
    any::{Any, TypeId},
    collections::BTreeMap,
    sync::Arc,
};

use super::{NativeStreamOperator, checkpoint, options};
use crate::{
    CalcFlowError, CheckpointManifest, NodeOperator, OperatorManifestEntry, OperatorStateSnapshot,
    Result, StateHandle, StreamJobContext, StreamOperator,
    array::sequence::{
        capture::Captures,
        codec::{self, RestoreLimits},
    },
    operator::rolling::IndicatorRollingRecord,
    pipeline::{CompiledStreamOperator, OperatorCheckpointCapability, StreamRuntimePlanParts},
    runtime::streaming::StreamJobControl,
};

mod io;
pub(crate) use io::NativeIoCompletion;

#[derive(Debug)]
pub(crate) struct NativeSequenceFailure {
    node: String,
    error: Box<CalcFlowError>,
}
impl NativeSequenceFailure {
    pub(crate) fn into_parts(self) -> (String, CalcFlowError) {
        (self.node, *self.error)
    }
}

pub(crate) struct NativeSequenceInventory {
    graph: String,
    records: BTreeMap<String, Arc<NativeCheckpointRecord>>,
}
pub(crate) enum NativeCheckpointRecord {
    Sequence(NativeSequenceRecord),
    IndicatorRolling(IndicatorRollingRecord),
}
pub(crate) struct NativeSequenceRecord {
    node: String,
    configured: Arc<options::Options>,
    workers: Arc<super::super::worker::Workers>,
    captures: Captures,
    initial: usize,
    capability: OperatorCheckpointCapability,
    control: Arc<StreamJobControl>,
}

fn concrete(operator: &dyn StreamOperator) -> Option<&NativeStreamOperator> {
    if Any::type_id(operator) != TypeId::of::<NativeStreamOperator>() {
        return None;
    }
    let operator: &dyn Any = operator;
    operator.downcast_ref()
}
pub(crate) fn is_native_sequence_operator(operator: &NodeOperator) -> bool {
    match operator {
        NodeOperator::Stream(operator) => concrete(operator.as_ref()).is_some(),
        _ => false,
    }
}
pub(super) fn compiled(operator: &CompiledStreamOperator) -> Option<&NativeStreamOperator> {
    match operator {
        CompiledStreamOperator::External(operator) => concrete(operator.as_ref()),
        _ => None,
    }
}
impl NativeSequenceInventory {
    pub(crate) fn for_plan(
        plan: &StreamRuntimePlanParts,
        job: &StreamJobContext,
    ) -> Option<Arc<Self>> {
        let control = Arc::new(job.native_control());
        let graph: Arc<str> = plan.fingerprint.as_str().into();
        let mut records = BTreeMap::new();
        for node in &plan.nodes {
            let id = node.operator_id.as_str().to_owned();
            if let Some(operator) = compiled(&node.operator) {
                records.insert(
                    id.clone(),
                    Arc::new(NativeCheckpointRecord::Sequence(NativeSequenceRecord {
                        node: id,
                        configured: Arc::clone(&operator.configured),
                        workers: Arc::clone(&operator.workers),
                        captures: operator.captures.clone(),
                        initial: operator.state.state_bytes(),
                        capability: node.checkpoint_capability,
                        control: Arc::clone(&control),
                    })),
                );
            } else if let CompiledStreamOperator::Rolling(operator) = &node.operator
                && let Some(record) = operator.native_checkpoint_record(
                    &id,
                    Arc::clone(&graph),
                    node.checkpoint_capability,
                    Arc::clone(&control),
                )
            {
                records.insert(
                    id,
                    Arc::new(NativeCheckpointRecord::IndicatorRolling(record)),
                );
            }
        }
        (!records.is_empty()).then(|| {
            Arc::new(Self {
                graph: plan.fingerprint.clone(),
                records,
            })
        })
    }
    pub(crate) fn first_failure(&self, error: CalcFlowError) -> NativeSequenceFailure {
        self.records
            .values()
            .next()
            .expect("native inventory is nonempty")
            .failure(error)
    }
    pub(crate) fn mismatch(&self, field: &str, message: &str) -> CalcFlowError {
        self.records
            .values()
            .next()
            .expect("native inventory is nonempty")
            .mismatch(field, message)
    }
    pub(crate) fn cancelled(&self) -> CalcFlowError {
        let record = self
            .records
            .values()
            .next()
            .expect("native inventory is nonempty");
        CalcFlowError::Cancelled {
            run_id: record.control().job_id().to_string(),
        }
    }
    pub(crate) fn record(&self, node: &str) -> Option<&Arc<NativeCheckpointRecord>> {
        self.records.get(node)
    }
    pub(crate) fn contains(&self, node: &str) -> bool {
        self.records.contains_key(node)
    }
    pub(crate) fn check(&self) -> Result<()> {
        self.records
            .values()
            .next()
            .expect("native inventory is nonempty")
            .control()
            .check_cancelled()
    }
    pub(crate) fn admit_manifest(
        &self,
        manifest: &CheckpointManifest,
    ) -> std::result::Result<(), NativeSequenceFailure> {
        self.admit_manifest_mode(manifest, true)
    }
    pub(crate) fn admit_retained_manifest(
        &self,
        manifest: &CheckpointManifest,
    ) -> std::result::Result<(), NativeSequenceFailure> {
        self.admit_manifest_mode(manifest, false)
    }
    fn admit_manifest_mode(
        &self,
        manifest: &CheckpointManifest,
        recovery: bool,
    ) -> std::result::Result<(), NativeSequenceFailure> {
        for (node, record) in &self.records {
            let entry = manifest.operators().get(node).ok_or_else(|| {
                record.failure(record.mismatch("segments", "missing native operator entry"))
            })?;
            (if recovery {
                record.admit_entry(entry)
            } else {
                record.admit_retained(entry)
            })
            .map_err(|error| record.failure(error))?;
        }
        if manifest.pipeline_fingerprint() != self.graph {
            let record = self
                .records
                .values()
                .next()
                .expect("native inventory is nonempty");
            return Err(record.failure(record.mismatch("identity", "graph fingerprint differs")));
        }
        Ok(())
    }
}

impl NativeCheckpointRecord {
    pub(crate) fn mismatch(&self, field: &str, message: &str) -> CalcFlowError {
        match self {
            Self::Sequence(_) => codec::mismatch(field, message),
            Self::IndicatorRolling(record) => record.mismatch(field, message),
        }
    }
    pub(crate) fn failure(&self, error: CalcFlowError) -> NativeSequenceFailure {
        match self {
            Self::Sequence(record) => record.failure(error),
            Self::IndicatorRolling(record) => NativeSequenceFailure {
                node: record.node().to_owned(),
                error: Box::new(error),
            },
        }
    }
    pub(super) fn control(&self) -> &Arc<StreamJobControl> {
        match self {
            Self::Sequence(record) => &record.control,
            Self::IndicatorRolling(record) => record.control(),
        }
    }
    pub(crate) fn admit_entry(&self, entry: &OperatorManifestEntry) -> Result<()> {
        match self {
            Self::Sequence(record) => record.admit_entry(entry),
            Self::IndicatorRolling(record) => record.admit_entry(entry),
        }
    }
    pub(crate) fn admit_retained(&self, entry: &OperatorManifestEntry) -> Result<()> {
        match self {
            Self::Sequence(record) => record.admit_retained(entry),
            Self::IndicatorRolling(record) => record.admit_retained(entry),
        }
    }
    pub(crate) fn admit_snapshot(&self, snapshot: &OperatorStateSnapshot) -> Result<()> {
        match self {
            Self::Sequence(record) => record.admit_snapshot(snapshot),
            Self::IndicatorRolling(record) => record.admit_snapshot(snapshot),
        }
    }
    pub(crate) fn handles<'a>(
        &self,
        entry: &'a OperatorManifestEntry,
    ) -> Result<&'a [StateHandle]> {
        match self {
            Self::Sequence(record) => {
                record.handle(entry)?;
            }
            Self::IndicatorRolling(record) => {
                record.handle(entry)?;
            }
        }
        Ok(&entry.segments)
    }
}
impl NativeSequenceRecord {
    pub(crate) fn failure(&self, error: CalcFlowError) -> NativeSequenceFailure {
        NativeSequenceFailure {
            node: self.node.clone(),
            error: Box::new(error),
        }
    }
    pub(crate) fn admit_entry(&self, entry: &OperatorManifestEntry) -> Result<()> {
        let metadata = self
            .capability
            .decode_snapshot(
                &self.node,
                OperatorStateSnapshot {
                    inline_metadata: entry.inline_metadata.clone(),
                    segments: BTreeMap::new(),
                },
            )?
            .inline_metadata;
        self.metadata(metadata)?;
        let handle = self.handle(entry)?;
        self.preflight(handle.byte_len())
    }
    fn metadata(&self, mut metadata: crate::JsonMap) -> Result<()> {
        metadata.remove(crate::pipeline::OUTPUT_FRONTIER_METADATA_KEY_V1);
        if metadata != checkpoint::metadata(&self.configured.spec) {
            return Err(codec::mismatch(
                "metadata",
                "identity, layout, numeric semantics or fingerprint differs",
            ));
        }
        Ok(())
    }
    pub(crate) fn admit_snapshot(&self, snapshot: &OperatorStateSnapshot) -> Result<()> {
        let metadata = self
            .capability
            .decode_snapshot(&self.node, snapshot.clone())?
            .inline_metadata;
        self.metadata(metadata)?;
        if snapshot.segments.len() != 1 || !snapshot.segments.contains_key("state") {
            return Err(codec::mismatch(
                "segments",
                "requires exactly one segment named state",
            ));
        }
        codec::validate_encoded_storage(
            &self.configured.spec,
            snapshot.segments["state"].bytes().len(),
            self.configured.state,
        )
    }
    pub(crate) fn admit_retained(&self, entry: &OperatorManifestEntry) -> Result<()> {
        let metadata = self
            .capability
            .decode_snapshot(
                &self.node,
                OperatorStateSnapshot {
                    inline_metadata: entry.inline_metadata.clone(),
                    segments: BTreeMap::new(),
                },
            )?
            .inline_metadata;
        self.metadata(metadata)?;
        let length = usize::try_from(self.handle(entry)?.byte_len())
            .map_err(|_| codec::mismatch("encoded_bytes", "exceeds usize"))?;
        codec::validate_encoded_storage(&self.configured.spec, length, self.configured.state)
    }
    pub(crate) fn handle<'a>(&self, entry: &'a OperatorManifestEntry) -> Result<&'a StateHandle> {
        if entry.segments.len() != 1 || entry.segments[0].segment_id() != "state" {
            return Err(codec::mismatch(
                "segments",
                "requires exactly one segment named state",
            ));
        }
        let handle = &entry.segments[0];
        handle.validate_owner(&self.node)?;
        Ok(handle)
    }
    fn preflight(&self, length: u64) -> Result<()> {
        let length = usize::try_from(length)
            .map_err(|_| codec::mismatch("encoded_bytes", "exceeds usize"))?;
        let encoded = length
            .checked_add(64)
            .ok_or_else(|| codec::mismatch("encoded_bytes", "byte count overflowed usize"))?;
        let limits = RestoreLimits {
            state: self.configured.state,
            execution: &self.configured.execution,
            old: self.initial,
            captures: self.captures.bytes(),
            encoded,
        };
        codec::admit(
            &self.configured.execution,
            &[
                self.initial,
                limits.captures,
                encoded,
                crate::StateSegment::controlled_scratch_bytes(),
            ],
        )?;
        codec::preflight_encoded(&self.configured.spec, length, &limits)
    }
}
