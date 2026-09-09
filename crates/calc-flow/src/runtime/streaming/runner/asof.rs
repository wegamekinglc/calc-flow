//! Extra terminal recovery validation for the independent ASOF state contract.

use std::collections::BTreeMap;

use super::{OpenedCheckpointRuntime, OperatorProgress, OperatorRestoreState};
use crate::{
    CalcFlowError, CancellationToken, EventTime, OperatorStateSnapshot, Result,
    pipeline::{CompiledStreamOperator, RuntimeStreamNode, StreamRuntimePlanParts},
    runtime::streaming::{
        operator_task::restore_terminal_asof,
        progress::{PreparedStreamJob, restore_durable_progress, types::LogicalInstant},
    },
};

pub(super) async fn restore_terminal(
    plan: &mut StreamRuntimePlanParts,
    checkpoint: &OpenedCheckpointRuntime,
    prepared: &PreparedStreamJob,
    cancellation: &CancellationToken,
) -> Result<BTreeMap<String, OperatorProgress>> {
    let mut statuses = BTreeMap::new();
    if !plan.nodes.iter().any(|node| is_asof(&node.operator)) {
        return Ok(statuses);
    }
    let selected = checkpoint
        .selected
        .as_ref()
        .expect("terminal manifest was selected");
    restore_durable_progress(prepared, selected.manifest.sources(), LogicalInstant::ZERO)?;
    for node in &mut plan.nodes {
        if !is_asof(&node.operator) {
            continue;
        }
        let id = node.operator_id.as_str();
        let entry = &selected.manifest.operators()[id];
        let snapshot = checkpoint
            .transaction
            .load_operator_state_cancellable(id, entry, cancellation)
            .await?;
        let (snapshot, output_frontier) = decode_terminal_snapshot(node, snapshot)?;
        let restore = OperatorRestoreState {
            snapshot,
            progress: entry.progress.clone(),
            output_frontier,
            next_epoch: checkpoint.next_epoch,
        };
        let progress =
            restore_terminal_asof(&mut node.operator, node.ingress_edges.keys(), &restore)?;
        statuses.insert(node.node_id.clone(), progress);
    }
    Ok(statuses)
}

fn is_asof(operator: &CompiledStreamOperator) -> bool {
    matches!(operator, CompiledStreamOperator::StreamAsofJoin(_))
}

fn decode_terminal_snapshot(
    node: &RuntimeStreamNode,
    snapshot: OperatorStateSnapshot,
) -> Result<(OperatorStateSnapshot, Option<EventTime>)> {
    let id = node.operator_id.as_str();
    let mut snapshot = node.checkpoint_capability.decode_snapshot(id, snapshot)?;
    let frontier = take_output_frontier(&mut snapshot, id)?;
    Ok((snapshot, frontier))
}

fn take_output_frontier(
    snapshot: &mut OperatorStateSnapshot,
    id: &str,
) -> Result<Option<EventTime>> {
    let value = snapshot
        .inline_metadata
        .remove(crate::pipeline::OUTPUT_FRONTIER_METADATA_KEY_V1);
    match value {
        Some(serde_json::Value::Null) => Ok(None),
        Some(value) if value.as_i64().is_some() => Ok(value.as_i64().map(EventTime::from_micros)),
        _ => Err(CalcFlowError::CheckpointMismatch {
            message: format!("checkpoint operator {id:?} has a missing or invalid output frontier"),
        }),
    }
}
