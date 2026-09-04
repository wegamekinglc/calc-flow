//! Shared checkpoint-state validation helpers for stateful operators.
//!
//! Rolling, window, and cross-section operators serialize the same
//! checkpoint envelope: snapshot metadata, a segment inventory, and the
//! durable segment map. The error constructors and the segment-inventory
//! contract checks they repeat live here once. Messages are prefixed with
//! the operator name so restore failures keep naming their surface.

use crate::{
    CalcFlowError, Epoch, OperatorStateSnapshot, Result,
    state::{SegmentDescriptor, StateInventory},
};

/// Construct the standard checkpoint-mismatch error.
pub(crate) fn checkpoint_mismatch(message: impl Into<String>) -> CalcFlowError {
    CalcFlowError::CheckpointMismatch {
        message: message.into(),
    }
}

/// Construct the standard state-format error.
pub(crate) fn state_format(message: impl Into<String>) -> CalcFlowError {
    CalcFlowError::Format {
        message: message.into(),
    }
}

/// Construct the standard compile error.
pub(crate) fn compile_error(message: impl Into<String>) -> CalcFlowError {
    CalcFlowError::Compile {
        message: message.into(),
    }
}

/// Construct the standard internal invariant error.
pub(crate) fn internal_error(message: impl Into<String>) -> CalcFlowError {
    CalcFlowError::Internal {
        message: message.into(),
    }
}

/// The operator-agnostic part of a checkpoint snapshot's metadata.
#[derive(Debug)]
pub(crate) struct SnapshotContract<'a> {
    /// Operator name used to prefix restore failure messages.
    pub(crate) name: &'static str,
    /// The verified state layout version carried by the snapshot metadata.
    pub(crate) state_layout_version: u32,
    /// The verified schema fingerprint carried by the snapshot metadata.
    pub(crate) schema_fingerprint: &'a str,
    /// The epoch the snapshot was taken at.
    pub(crate) epoch: Epoch,
    /// The pipeline identity the snapshot declares, when present.
    pub(crate) pipeline_fingerprint: Option<&'a str>,
    /// The operator identity the snapshot declares, when present.
    pub(crate) operator_id: Option<&'a str>,
    /// The serialized segment inventory descriptors.
    pub(crate) segment_inventory: Vec<SegmentDescriptor>,
}

/// Validate the segment inventory and identity fields shared by every
/// stateful operator snapshot, failing closed through
/// [`CheckpointMismatch`][crate::CalcFlowError::CheckpointMismatch].
pub(crate) fn validate_inventory(
    contract: &SnapshotContract<'_>,
    snapshot: &OperatorStateSnapshot,
) -> Result<StateInventory> {
    let name = contract.name;
    let inventory = StateInventory::new(contract.segment_inventory.clone())
        .map_err(|error| checkpoint_mismatch(error.to_string()))?;
    for descriptor in inventory.segments() {
        if descriptor.state_layout_version != contract.state_layout_version
            || descriptor.schema_fingerprint != contract.schema_fingerprint
        {
            return Err(checkpoint_mismatch(format!(
                "{name} segment inventory layout or schema does not match the compiled operator"
            )));
        }
        if descriptor.handle.epoch() > contract.epoch {
            return Err(checkpoint_mismatch(format!(
                "{name} segment inventory contains a future epoch"
            )));
        }
        if contract.operator_id != Some(descriptor.handle.operator_id()) {
            return Err(checkpoint_mismatch(format!(
                "{name} segment inventory operator does not match snapshot metadata"
            )));
        }
    }
    let expected_ids = inventory
        .segments()
        .iter()
        .map(|descriptor| descriptor.handle.segment_id().to_owned())
        .collect::<Vec<_>>();
    let actual_ids = snapshot.segments.keys().cloned().collect::<Vec<_>>();
    if expected_ids != actual_ids {
        return Err(checkpoint_mismatch(format!(
            "{name} snapshot segment IDs are missing, extra, duplicated, or non-canonical"
        )));
    }
    if !snapshot.segments.is_empty()
        && (contract.pipeline_fingerprint.is_none() || contract.operator_id.is_none())
    {
        return Err(checkpoint_mismatch(format!(
            "{name} segments require pipeline and operator identity metadata"
        )));
    }
    if let Some(fingerprint) = contract.pipeline_fingerprint
        && (fingerprint.len() != 64
            || !fingerprint
                .bytes()
                .all(|byte| byte.is_ascii_digit() || (b'a'..=b'f').contains(&byte)))
    {
        return Err(checkpoint_mismatch(format!(
            "{name} pipeline fingerprint is not lowercase SHA-256"
        )));
    }
    if contract
        .operator_id
        .is_some_and(|operator_id| operator_id.is_empty() || operator_id.contains('\0'))
    {
        return Err(checkpoint_mismatch(format!(
            "{name} operator ID is empty or contains NUL"
        )));
    }
    Ok(inventory)
}
