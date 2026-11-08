use std::{
    collections::{BTreeMap, BTreeSet},
    sync::Arc,
};

use super::{
    DriverCompletion, DriverReport, FailureOrigin, LaunchId, OpenedCheckpointRuntime,
    OperatorRestoreState, RuntimeFailure, StartFailure, checkpoint_start_failure,
    sanitize_managed_recovery_error,
};
use crate::{
    CalcFlowError, OperatorStateSnapshot, StreamJobContext, StreamOperatorContext,
    array::{NativeSequenceFailure, NativeSequenceInventory},
    pipeline::StreamRuntimePlanParts,
    state::{LocalStateLineageBackend, ManifestTransaction, NativeRecoveryError},
};

pub(super) fn attach(
    transaction: ManifestTransaction,
    inventory: Option<Arc<NativeSequenceInventory>>,
    reader: Option<&Arc<LocalStateLineageBackend>>,
) -> Result<ManifestTransaction, NativeRecoveryError> {
    let Some(inventory) = inventory else {
        return Ok(transaction);
    };
    let reader = reader.ok_or_else(|| {
        NativeRecoveryError::Native(inventory.first_failure(CalcFlowError::CheckpointMismatch {
            message:
                "native:array_sequence@1 state.reader: requires a certified local reader".into(),
        }))
    })?;
    transaction
        .with_native_sequences(inventory, Arc::clone(reader))
        .map_err(NativeRecoveryError::Native)
}

pub(super) fn failure(
    launch_id: LaunchId,
    error: NativeRecoveryError,
    managed: bool,
) -> DriverReport {
    match error {
        NativeRecoveryError::Legacy(error) => {
            checkpoint_start_failure(launch_id, sanitize_managed_recovery_error(error, managed))
        }
        NativeRecoveryError::Native(failure) => native_failure(launch_id, failure),
    }
}
pub(super) fn native_failure(launch_id: LaunchId, failure: NativeSequenceFailure) -> DriverReport {
    let (node_id, error) = failure.into_parts();
    DriverReport {
        launch_id,
        completion: DriverCompletion::StartFailed(StartFailure {
            primary: Arc::new(RuntimeFailure {
                origin: FailureOrigin::NativeSequenceRecovery { node_id },
                error,
            }),
            diagnostic_id: None,
            cleanup_failures: Vec::new(),
        }),
        cleanup_failures: Vec::new(),
    }
}
pub(super) fn sanitize_preflight(error: NativeRecoveryError, managed: bool) -> NativeRecoveryError {
    match error {
        NativeRecoveryError::Legacy(error) => NativeRecoveryError::Legacy(
            super::sanitize_managed_preflight_error(error, managed, true),
        ),
        native @ NativeRecoveryError::Native(_) => native,
    }
}

pub(super) async fn load(
    plan: &StreamRuntimePlanParts,
    checkpoint: Option<&OpenedCheckpointRuntime>,
    job: &StreamJobContext,
) -> Result<BTreeMap<String, OperatorStateSnapshot>, NativeSequenceFailure> {
    let mut snapshots = BTreeMap::new();
    let Some(checkpoint) = checkpoint else {
        return Ok(snapshots);
    };
    let Some(inventory) = &checkpoint.native_sequences else {
        return Ok(snapshots);
    };
    let Some(selected) = &checkpoint.selected else {
        return Ok(snapshots);
    };
    for node in &plan.nodes {
        let id = node.operator_id.as_str();
        if inventory.contains(id) {
            let snapshot = inventory
                .load_snapshot(
                    &node.operator,
                    id,
                    &selected.manifest.operators()[id],
                    &checkpoint.transaction,
                    job,
                )
                .await?;
            snapshots.insert(id.to_owned(), snapshot);
        }
    }
    Ok(snapshots)
}

pub(super) async fn restore_terminal(
    plan: &mut StreamRuntimePlanParts,
    checkpoint: &OpenedCheckpointRuntime,
    mut snapshots: BTreeMap<String, OperatorStateSnapshot>,
    job: &StreamJobContext,
) -> Result<(), NativeSequenceFailure> {
    for node in &mut plan.nodes {
        let id = node.operator_id.as_str();
        if let Some(snapshot) = snapshots.remove(id) {
            let record = checkpoint
                .native_sequences
                .as_ref()
                .and_then(|inventory| inventory.record(id))
                .expect("native snapshot carries intrinsic admission");
            let restore = async {
                let selected = checkpoint
                    .selected
                    .as_ref()
                    .expect("recovery selected a manifest");
                let entry = &selected.manifest.operators()[id];
                super::super::operator_task::validate_sql_restore_progress(
                    node.ingress_edges.keys(),
                    &entry.progress,
                )?;
                let mut snapshot = node.checkpoint_capability.decode_snapshot(id, snapshot)?;
                let frontier = snapshot
                    .inline_metadata
                    .remove(crate::pipeline::OUTPUT_FRONTIER_METADATA_KEY_V1);
                super::restored_output_frontier_value(id, false, frontier)?;
                node.operator
                    .restore_async(&snapshot, &StreamOperatorContext::new(job, id, None))
                    .await
            };
            restore.await.map_err(|error| record.failure(error))?;
        }
    }
    Ok(())
}

pub(super) fn entry_nodes(
    plan: &StreamRuntimePlanParts,
    restores: &BTreeMap<String, OperatorRestoreState>,
    job: &StreamJobContext,
) -> BTreeSet<String> {
    let Some(inventory) = NativeSequenceInventory::for_plan(plan, job) else {
        return BTreeSet::new();
    };
    plan.nodes
        .iter()
        .filter_map(|node| {
            let id = node.operator_id.as_str();
            (inventory.contains(id) && restores.contains_key(id)).then(|| id.to_owned())
        })
        .collect()
}
