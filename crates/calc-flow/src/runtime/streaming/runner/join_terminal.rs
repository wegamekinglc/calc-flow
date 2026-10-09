use std::collections::BTreeMap;

use super::{
    OpenedCheckpointRuntime, OperatorProgress, OperatorRestoreState, asof,
    restored_output_frontier_value,
};
use crate::{
    CancellationToken, Result, StreamJobContext,
    pipeline::{CompiledStreamOperator, RuntimeStreamNode, StreamRuntimePlanParts},
    runtime::streaming::{
        operator_task::restore_terminal_join,
        progress::{PreparedStreamJob, restore_durable_progress, types::LogicalInstant},
    },
};

pub(super) async fn restore_terminal(
    plan: &mut StreamRuntimePlanParts,
    checkpoint: &OpenedCheckpointRuntime,
    prepared: &PreparedStreamJob,
    loads: &asof::LoadOwner,
    cancellation: &CancellationToken,
    job: &StreamJobContext,
) -> Result<BTreeMap<String, OperatorProgress>> {
    let mut statuses = BTreeMap::new();
    if !prepare_terminal(plan, checkpoint, prepared)? {
        return Ok(statuses);
    }
    for node in &mut plan.nodes {
        if !is_join(&node.operator) {
            continue;
        }
        let progress = restore_node(node, checkpoint, loads, cancellation, job).await?;
        statuses.insert(node.node_id.clone(), progress);
    }
    job.check_cancelled()?;
    Ok(statuses)
}

async fn restore_node(
    node: &mut RuntimeStreamNode,
    checkpoint: &OpenedCheckpointRuntime,
    loads: &asof::LoadOwner,
    cancellation: &CancellationToken,
    job: &StreamJobContext,
) -> Result<OperatorProgress> {
    job.check_cancelled()?;
    let restore = load_restore(node, checkpoint, loads, cancellation).await?;
    restore_terminal_join(&mut node.operator, node.ingress_edges.keys(), &restore, job).await
}

fn prepare_terminal(
    plan: &mut StreamRuntimePlanParts,
    checkpoint: &OpenedCheckpointRuntime,
    prepared: &PreparedStreamJob,
) -> Result<bool> {
    if !plan.nodes.iter().any(|node| is_join(&node.operator)) {
        return Ok(false);
    }
    let selected = checkpoint
        .selected
        .as_ref()
        .expect("terminal manifest was selected");
    restore_durable_progress(prepared, selected.manifest.sources(), LogicalInstant::ZERO)?;
    if checkpoint.join_preload_reader.is_some() {
        asof::prepare_join_preloads(plan, selected.manifest.operators())?;
    }
    Ok(true)
}

async fn load_restore(
    node: &RuntimeStreamNode,
    checkpoint: &OpenedCheckpointRuntime,
    loads: &asof::LoadOwner,
    cancellation: &CancellationToken,
) -> Result<OperatorRestoreState> {
    let id = node.operator_id.as_str();
    let selected = checkpoint
        .selected
        .as_ref()
        .expect("terminal manifest was selected");
    let entry = &selected.manifest.operators()[id];
    let snapshot = asof::load_snapshot_with_join(
        &checkpoint.transaction,
        loads,
        &node.operator,
        id,
        entry,
        cancellation,
        checkpoint.join_preload_reader.as_ref(),
    )
    .await?;
    let mut snapshot = node.checkpoint_capability.decode_snapshot(id, snapshot)?;
    let frontier = snapshot
        .inline_metadata
        .remove(crate::pipeline::OUTPUT_FRONTIER_METADATA_KEY_V1);
    Ok(OperatorRestoreState {
        snapshot,
        progress: entry.progress.clone(),
        output_frontier: restored_output_frontier_value(id, true, frontier)?,
        next_epoch: checkpoint.next_epoch,
    })
}

fn is_join(operator: &CompiledStreamOperator) -> bool {
    matches!(operator, CompiledStreamOperator::StreamJoin(_))
}
