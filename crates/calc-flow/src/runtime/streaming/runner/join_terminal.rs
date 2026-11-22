use std::sync::Arc;

use std::collections::BTreeMap;

use super::{
    OpenedCheckpointRuntime, OperatorProgress, OperatorRestoreState, asof,
    restored_output_frontier_value,
};
use crate::{
    CancellationToken, Result, StreamJobContext,
    pipeline::{CompiledStreamOperator, RuntimeStreamNode, StreamRuntimePlanParts},
    runtime::streaming::{
        operator_task::restore_terminal_owned_join,
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
    let mut restore = load_restore(node, checkpoint, loads, cancellation).await?;
    restore_terminal_owned_join(
        &mut node.operator,
        node.ingress_edges.keys(),
        &mut restore,
        job,
    )
    .await
}

fn prepare_terminal(
    plan: &mut StreamRuntimePlanParts,
    checkpoint: &OpenedCheckpointRuntime,
    prepared: &PreparedStreamJob,
) -> Result<bool> {
    if !plan.nodes.iter().any(|node| is_join(&node.operator)) {
        return Ok(false);
    }
    if let Some(selected) = &checkpoint.selected_v4 {
        super::super::progress::restore_durable_progress_v4(
            prepared,
            &selected.manifest.sources,
            LogicalInstant::ZERO,
        )?;
        super::checkpoint_v4_metadata::prepare_join_preloads(plan, &selected.manifest)?;
        return Ok(true);
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
    if let Some(selected) = &checkpoint.selected_v4 {
        let entry = &selected.manifest.operators[id];
        let (raw, prepaid) = if let Some(reader) = checkpoint
            .join_preload_reader
            .as_ref()
            .filter(|_| matches!(entry, crate::state::v4::OperatorV4::Opaque { .. }))
        {
            let CompiledStreamOperator::StreamJoin(operator) = &node.operator else {
                unreachable!("Join terminal restore requires a Join operator")
            };
            (
                asof::join_preload::load_snapshot_v4(
                    loads,
                    operator,
                    id,
                    entry,
                    &checkpoint.manifest_root,
                    cancellation,
                    reader,
                )
                .await?,
                true,
            )
        } else {
            let mut snapshots =
                super::checkpoint_v4::load_snapshots(checkpoint, &selected.manifest).await?;
            (
                snapshots
                    .remove(id)
                    .ok_or_else(|| crate::CalcFlowError::CheckpointMismatch {
                        message: format!("terminal v4 Join {id:?} has no snapshot"),
                    })?,
                false,
            )
        };
        let mut snapshot = if prepaid {
            crate::state::PreloadedSnapshot::uncredited(raw)
        } else {
            super::admit_v4_snapshot(node, entry, raw)?
        }
        .map(|snapshot| node.checkpoint_capability.decode_snapshot(id, snapshot))?;
        let metadata = snapshot
            .snapshot_mut()
            .inline_metadata
            .remove(crate::pipeline::OUTPUT_FRONTIER_METADATA_KEY_V1);
        let expected = entry
            .output_frontier()
            .map_or(serde_json::Value::Null, |value| {
                serde_json::Value::from(value.as_micros())
            });
        if metadata != Some(expected) {
            return Err(crate::CalcFlowError::CheckpointMismatch {
                message: format!("terminal v4 Join {id:?} frontier metadata differs"),
            });
        }
        return Ok(OperatorRestoreState {
            snapshot: Arc::new(snapshot),
            progress: entry.progress().clone(),
            output_frontier: entry.output_frontier(),
            next_epoch: checkpoint.next_epoch,
        });
    }
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
        snapshot: Arc::new(crate::state::PreloadedSnapshot::uncredited(snapshot)),
        progress: entry.progress.clone(),
        output_frontier: restored_output_frontier_value(id, true, frontier)?,
        next_epoch: checkpoint.next_epoch,
    })
}

fn is_join(operator: &CompiledStreamOperator) -> bool {
    matches!(operator, CompiledStreamOperator::StreamJoin(_))
}
