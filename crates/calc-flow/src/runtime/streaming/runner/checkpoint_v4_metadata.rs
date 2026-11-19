use std::collections::BTreeSet;

use crate::pipeline::StreamRuntimePlanParts;
use crate::state::v4::V4Manifest;
use crate::{CalcFlowError, ManifestIngressState, Result};

pub(super) fn validate_capabilities(
    manifest: &V4Manifest,
    plan: &StreamRuntimePlanParts,
) -> Result<()> {
    for node in &plan.nodes {
        let operator_id = node.operator_id.as_str();
        let entry = manifest.operators.get(operator_id).ok_or_else(|| {
            CalcFlowError::CheckpointMismatch {
                message: format!("checkpoint is missing operator {operator_id:?}"),
            }
        })?;
        let Some((metadata, no_blocks)) = entry.opaque_metadata() else {
            continue;
        };
        node.checkpoint_capability
            .validate_snapshot_fields(operator_id, metadata, no_blocks)?;
    }
    Ok(())
}

pub(super) fn prepare_join_preloads(
    plan: &mut StreamRuntimePlanParts,
    manifest: &V4Manifest,
) -> Result<()> {
    for node in &mut plan.nodes {
        let has_blocks = matches!(
            manifest.operators.get(node.operator_id.as_str()),
            Some(crate::state::v4::OperatorV4::Opaque { blocks, .. }) if !blocks.is_empty()
        );
        if has_blocks
            && let crate::pipeline::CompiledStreamOperator::StreamJoin(operator) =
                &mut node.operator
        {
            operator.prepare_checkpoint_preload_runtime()?;
        }
    }
    Ok(())
}

pub(super) fn is_terminal(manifest: &V4Manifest, plan: &StreamRuntimePlanParts) -> Result<bool> {
    let sources_terminal = manifest.sources.values().all(|source| source.ended);
    let mut operators_terminal = true;
    for node in &plan.nodes {
        let operator_id = node.operator_id.as_str();
        let entry = manifest
            .operators
            .get(operator_id)
            .expect("selected v4 manifest operator IDs were validated");
        let expected = node.ingress_edges.keys().cloned().collect::<BTreeSet<_>>();
        let actual = entry.progress().keys().cloned().collect::<BTreeSet<_>>();
        if actual != expected {
            return Err(CalcFlowError::CheckpointMismatch {
                message: format!(
                    "terminal recovery operator {operator_id:?} ingress IDs do not match the prepared plan"
                ),
            });
        }
        operators_terminal &= entry
            .progress()
            .values()
            .all(|progress| matches!(progress.state, ManifestIngressState::Ended));
    }
    if sources_terminal != operators_terminal {
        return Err(CalcFlowError::CheckpointMismatch {
            message: "terminal recovery source and operator end states disagree".into(),
        });
    }
    Ok(sources_terminal)
}
