//! Extra terminal recovery validation for the independent ASOF state contract.

use std::{
    collections::{BTreeMap, BTreeSet},
    sync::Arc,
};

use datafusion::execution::memory_pool::MemoryReservation;

mod join_preload;
mod load_work;
pub(super) use load_work::LoadOwner;

use super::{OpenedCheckpointRuntime, OperatorProgress, OperatorRestoreState};
use crate::{
    CalcFlowError, CancellationToken, EventTime, OperatorManifestEntry, OperatorStateSnapshot,
    Result,
    pipeline::{CompiledStreamOperator, RuntimeStreamNode, StreamRuntimePlanParts},
    runtime::streaming::{
        operator_task::restore_terminal_asof,
        progress::{PreparedStreamJob, restore_durable_progress, types::LogicalInstant},
    },
    state::ManifestTransaction,
};

pub(super) async fn load_snapshot(
    transaction: &Arc<ManifestTransaction>,
    loads: &LoadOwner,
    operator: &CompiledStreamOperator,
    operator_id: &str,
    entry: &OperatorManifestEntry,
    cancellation: &CancellationToken,
) -> Result<OperatorStateSnapshot> {
    if cancellation.is_cancelled() {
        return Err(CalcFlowError::Cancelled {
            run_id: "checkpoint:state-load-lock".into(),
        });
    }
    let reservation = preload_owner(operator, operator_id, entry)?;
    let Some(owner) = reservation else {
        return transaction
            .load_operator_state_cancellable(operator_id, entry, cancellation)
            .await;
    };
    let transaction = transaction.clone();
    let operator_id = operator_id.to_owned();
    let entry = OperatorManifestEntry {
        progress: BTreeMap::new(),
        inline_metadata: entry.inline_metadata.clone(),
        segments: entry.segments.clone(),
    };
    let cancellation = cancellation.clone();
    loads
        .load(async move {
            let mut snapshot = transaction
                .load_operator_state_cancellable(&operator_id, &entry, &cancellation)
                .await?;
            snapshot.segments = snapshot
                .segments
                .into_iter()
                .map(|(id, segment)| (id, segment.with_owner(owner.clone())))
                .collect();
            Ok(snapshot)
        })
        .await
}

pub(super) async fn load_snapshot_with_join(
    transaction: &Arc<ManifestTransaction>,
    loads: &LoadOwner,
    operator: &CompiledStreamOperator,
    operator_id: &str,
    entry: &OperatorManifestEntry,
    cancellation: &CancellationToken,
    reader: Option<&Arc<crate::state::LocalStateLineageBackend>>,
) -> Result<OperatorStateSnapshot> {
    if cancellation.is_cancelled() {
        return Err(CalcFlowError::Cancelled {
            run_id: "checkpoint:state-load-lock".into(),
        });
    }
    if let (CompiledStreamOperator::StreamJoin(operator), Some(reader)) = (operator, reader) {
        return join_preload::load_snapshot(
            transaction,
            loads,
            operator,
            operator_id,
            entry,
            cancellation,
            reader,
        )
        .await;
    }
    load_snapshot(
        transaction,
        loads,
        operator,
        operator_id,
        entry,
        cancellation,
    )
    .await
}

pub(super) fn prepare_join_preloads(
    plan: &mut StreamRuntimePlanParts,
    entries: &BTreeMap<String, OperatorManifestEntry>,
) -> Result<()> {
    for node in &mut plan.nodes {
        prepare_join_preload(&mut node.operator, entries.get(node.operator_id.as_str()))?;
    }
    Ok(())
}

fn prepare_join_preload(
    operator: &mut CompiledStreamOperator,
    entry: Option<&OperatorManifestEntry>,
) -> Result<()> {
    if entry.is_some_and(|entry| !entry.segments.is_empty())
        && let CompiledStreamOperator::StreamJoin(operator) = operator
    {
        operator.prepare_checkpoint_preload_runtime()?;
    }
    Ok(())
}

fn preload_owner(
    operator: &CompiledStreamOperator,
    operator_id: &str,
    entry: &OperatorManifestEntry,
) -> Result<Option<Arc<MemoryReservation>>> {
    let CompiledStreamOperator::StreamAsofJoin(operator) = operator else {
        return Ok(None);
    };
    let mut segment_ids = BTreeSet::new();
    for handle in &entry.segments {
        handle.validate_owner(operator_id)?;
        if !segment_ids.insert(handle.segment_id()) {
            return Err(CalcFlowError::CheckpointMismatch {
                message: format!(
                    "operator {operator_id:?} repeats state segment {:?}",
                    handle.segment_id()
                ),
            });
        }
    }
    if entry.segments.is_empty() {
        return Ok(None);
    }
    let request_bytes = request_bytes(entry).ok_or_else(|| CalcFlowError::CheckpointMismatch {
        message: format!("checkpoint operator {operator_id:?} has oversized load metadata"),
    })?;
    operator
        .reserve_checkpoint_preload(
            entry
                .segments
                .iter()
                .map(crate::StateHandle::byte_len)
                .chain(std::iter::once(request_bytes)),
        )
        .map(|reservation| Some(Arc::new(reservation)))
}

fn request_bytes(entry: &OperatorManifestEntry) -> Option<u64> {
    let metadata = metadata_request_bytes(entry)?;
    entry.segments.iter().try_fold(metadata, |total, handle| {
        let strings = [
            handle.operator_id(),
            handle.segment_id(),
            handle.relative_path(),
            handle.sha256(),
        ];
        strings
            .iter()
            .try_fold(total.checked_add(256)?, |bytes, value| {
                bytes.checked_add(u64::try_from(value.len()).ok()?)
            })
    })
}

fn metadata_request_bytes(entry: &OperatorManifestEntry) -> Option<u64> {
    entry
        .inline_metadata
        .iter()
        .try_fold(1024_u64, |total, (key, value)| {
            total
                .checked_add(u64::try_from(key.len()).ok()?.checked_mul(2)?)?
                .checked_add(value_bytes(value, 0)?)?
                .checked_add(128)
        })
}

fn value_bytes(value: &serde_json::Value, depth: usize) -> Option<u64> {
    if depth >= 128 {
        return None;
    }
    match value {
        serde_json::Value::String(value) => {
            128_u64.checked_add(u64::try_from(value.len()).ok()?.checked_mul(2)?)
        }
        serde_json::Value::Array(values) => values.iter().try_fold(128_u64, |total, value| {
            total.checked_add(value_bytes(value, depth + 1)?)
        }),
        serde_json::Value::Object(values) => object_value_bytes(values, depth),
        _ => Some(128),
    }
}

fn object_value_bytes(
    values: &serde_json::Map<String, serde_json::Value>,
    depth: usize,
) -> Option<u64> {
    values.iter().try_fold(128_u64, |total, (key, value)| {
        total
            .checked_add(128)?
            .checked_add(u64::try_from(key.len()).ok()?.checked_mul(2)?)?
            .checked_add(value_bytes(value, depth + 1)?)
    })
}

pub(super) async fn restore_terminal(
    plan: &mut StreamRuntimePlanParts,
    checkpoint: &OpenedCheckpointRuntime,
    prepared: &PreparedStreamJob,
    loads: &LoadOwner,
    cancellation: &CancellationToken,
    job: &crate::StreamJobContext,
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
        let snapshot = load_snapshot(
            &checkpoint.transaction,
            loads,
            &node.operator,
            id,
            entry,
            cancellation,
        )
        .await?;
        let (snapshot, output_frontier) = decode_terminal_snapshot(node, snapshot)?;
        let mut restore = OperatorRestoreState {
            snapshot,
            progress: entry.progress.clone(),
            output_frontier,
            next_epoch: checkpoint.next_epoch,
        };
        let progress = restore_terminal_asof(
            &mut node.operator,
            node.ingress_edges.keys(),
            &mut restore,
            job,
        )
        .await?;
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

#[cfg(test)]
mod preload_tests;

#[cfg(test)]
mod join_preload_tests;
