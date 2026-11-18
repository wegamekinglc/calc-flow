//! Live runner persistence for opaque checkpoint-v4 participant state.

use std::{
    collections::{BTreeMap, BTreeSet},
    path::PathBuf,
};

use chrono::Utc;
use sha2::{Digest, Sha256};

use super::super::checkpoint_runtime::OpenedCheckpointRuntime;
use crate::state::v4::{
    JobKind, OpaqueObject, OperatorV4, SinkV4, SourceHistoryV4, SourceV4, V4Handle, V4Manifest,
    V4OpaqueAdapter, V4Store, manifest_fingerprint,
};
use crate::state::{OperatorManifestEntry, SinkManifestEntry, SourceManifestEntry};
use crate::{CalcFlowError, EventTime, OperatorStateSnapshot, Result};

#[derive(Clone, Copy)]
pub(super) struct AssemblyView<'a> {
    pub sources: &'a BTreeMap<String, SourceManifestEntry>,
    pub operator_snapshots: &'a BTreeMap<String, OperatorStateSnapshot>,
    pub operator_progress: &'a BTreeMap<String, OperatorManifestEntry>,
    pub operator_frontiers: &'a BTreeMap<String, Option<EventTime>>,
    pub sink_outputs: &'a BTreeMap<String, BTreeMap<String, SinkManifestEntry>>,
    pub sink_captures: &'a BTreeMap<String, BTreeMap<String, Vec<u8>>>,
}

fn store_root(checkpoint: &OpenedCheckpointRuntime) -> PathBuf {
    checkpoint.manifest_root.join("lineage")
}

pub(super) async fn load_snapshots(
    checkpoint: &OpenedCheckpointRuntime,
    manifest: &V4Manifest,
) -> Result<BTreeMap<String, OperatorStateSnapshot>> {
    let root = store_root(checkpoint);
    let manifest = manifest.clone();
    tokio::task::spawn_blocking(move || {
        let mut store = V4Store::new(root);
        manifest
            .operators
            .keys()
            .map(|operator_id| {
                V4OpaqueAdapter::restore(&mut store, &manifest, operator_id)
                    .map(|snapshot| (operator_id.clone(), snapshot))
            })
            .collect()
    })
    .await
    .map_err(|error| CalcFlowError::Internal {
        message: format!("checkpoint-v4 restore worker failed: {error}"),
    })?
}

pub(in crate::runtime::streaming) async fn load_sink_segments(
    checkpoint: &OpenedCheckpointRuntime,
    entry: &SinkV4,
) -> Result<BTreeMap<String, Vec<u8>>> {
    let root = store_root(checkpoint);
    let handles = entry.blocks.clone();
    tokio::task::spawn_blocking(move || {
        let mut store = V4Store::new(root);
        handles
            .iter()
            .map(|handle| {
                store
                    .read_opaque(handle)
                    .map(|bytes| (handle.object_id.clone(), bytes))
            })
            .collect()
    })
    .await
    .map_err(|error| CalcFlowError::Internal {
        message: format!("checkpoint-v4 sink restore worker failed: {error}"),
    })?
}

pub(super) async fn persist_epoch(
    checkpoint: &OpenedCheckpointRuntime,
    view: AssemblyView<'_>,
    epoch: u64,
    cancellation: &crate::CancellationToken,
) -> Result<PathBuf> {
    #[cfg(not(test))]
    let _ = cancellation;
    let mut store = V4Store::new(store_root(checkpoint));
    let sources = persist_sources(checkpoint, &mut store, view.sources, epoch).await?;
    let operators = persist_operators(&mut store, view, epoch).await?;
    let sinks = persist_sinks(&mut store, view, epoch).await?;
    let mut manifest = V4Manifest {
        format_version: 4,
        pipeline_name: checkpoint.identity.pipeline_name.clone(),
        pipeline_fingerprint: checkpoint.identity.pipeline_fingerprint.clone(),
        runtime_config_hash: checkpoint.identity.runtime_config_hash.clone(),
        epoch,
        created_at: Utc::now(),
        recovery_status: "final".into(),
        job_kind: JobKind::Runner,
        recovery_fingerprint: checkpoint.identity.pipeline_fingerprint.clone(),
        sources,
        operators,
        sinks,
        static_inputs: checkpoint.identity.static_inputs.clone(),
        manifest_fingerprint: String::new(),
    };
    manifest.manifest_fingerprint = manifest_fingerprint(&manifest)?;
    validate_participants(checkpoint, &manifest, view)?;
    #[cfg(all(test, unix))]
    let real_parent_sync_failure = checkpoint.faults.is_armed(
        super::CheckpointFaultPoint::ManifestParentSync,
        super::CheckpointFaultMode::Io,
    );
    #[cfg(all(test, unix))]
    if real_parent_sync_failure {
        store.fail_parent_sync_for_test(checkpoint.faults.parent_sync_os_failure_probe());
    }
    let path = match store.publish_manifest(&manifest) {
        Ok(path) => path,
        #[cfg(all(test, unix))]
        Err(_) if real_parent_sync_failure => {
            let _ = checkpoint.inject_fault(
                super::CheckpointFaultPoint::ManifestParentSync,
                cancellation,
            );
            return Err(CalcFlowError::RecoveryRequired {
                pipeline_name: checkpoint.identity.pipeline_name.clone(),
                message: "v4 manifest parent sync failed after installation".into(),
            });
        }
        Err(error) => return Err(error),
    };
    #[cfg(test)]
    if !matches!(
        checkpoint.inject_fault(super::CheckpointFaultPoint::ManifestRename, cancellation),
        Ok(false)
    ) {
        return Err(CalcFlowError::RecoveryRequired {
            pipeline_name: checkpoint.identity.pipeline_name.clone(),
            message: "v4 manifest rename outcome is unknown".into(),
        });
    }
    #[cfg(test)]
    if !matches!(
        checkpoint.inject_fault(
            super::CheckpointFaultPoint::ManifestParentSync,
            cancellation
        ),
        Ok(false)
    ) {
        return Err(CalcFlowError::RecoveryRequired {
            pipeline_name: checkpoint.identity.pipeline_name.clone(),
            message: "v4 manifest parent sync outcome is unknown".into(),
        });
    }
    Ok(path)
}

async fn persist_sources(
    checkpoint: &OpenedCheckpointRuntime,
    store: &mut V4Store,
    sources: &BTreeMap<String, SourceManifestEntry>,
    epoch: u64,
) -> Result<BTreeMap<String, SourceV4>> {
    let mut persisted = BTreeMap::new();
    for (source_id, entry) in sources {
        let history = entry
            .history
            .as_ref()
            .map(|history| persist_history(checkpoint, store, source_id, history, epoch));
        let history = match history {
            Some(future) => Some(future.await?),
            None => None,
        };
        persisted.insert(
            source_id.clone(),
            SourceV4 {
                cursor: entry.cursor.clone(),
                identity_hash: entry.identity_hash.clone(),
                next_sequence: entry.sequence,
                ended: entry.ended,
                watermark_policy: entry.watermark_policy.clone(),
                history,
            },
        );
    }
    Ok(persisted)
}

async fn persist_history(
    checkpoint: &OpenedCheckpointRuntime,
    store: &mut V4Store,
    source_id: &str,
    history: &crate::state::SourceHistoryManifestEntry,
    epoch: u64,
) -> Result<SourceHistoryV4> {
    let mut handles = Vec::with_capacity(history.segments.len());
    for handle in &history.segments {
        let bytes = checkpoint.transaction.load_source_history(handle).await?;
        handles.push(write_bytes(store, source_id, handle.segment_id(), bytes, epoch).await?);
    }
    Ok(SourceHistoryV4 {
        format_version: history.format_version,
        contract: history.contract.clone(),
        inline_metadata: history.inline_metadata.clone(),
        segments: handles,
    })
}

async fn persist_operators(
    store: &mut V4Store,
    view: AssemblyView<'_>,
    epoch: u64,
) -> Result<BTreeMap<String, OperatorV4>> {
    let mut operators = BTreeMap::new();
    for (node_id, snapshot) in view.operator_snapshots {
        let adapter = V4OpaqueAdapter::from_snapshot(node_id, snapshot);
        let (mut operator, _) = adapter.persist(store, epoch).await?;
        if let OperatorV4::Opaque {
            progress,
            output_frontier,
            ..
        } = &mut operator
        {
            *progress = view.operator_progress[node_id].progress.clone();
            *output_frontier = view.operator_frontiers.get(node_id).copied().flatten();
        }
        operators.insert(node_id.clone(), operator);
    }
    Ok(operators)
}

async fn persist_sinks(
    store: &mut V4Store,
    view: AssemblyView<'_>,
    epoch: u64,
) -> Result<BTreeMap<String, SinkV4>> {
    let mut sinks = BTreeMap::new();
    for entries in view.sink_outputs.values() {
        for (sink_id, entry) in entries {
            let blocks = match view.sink_captures.get(sink_id) {
                Some(segments) => {
                    let mut blocks = Vec::with_capacity(segments.len());
                    for (segment_id, bytes) in segments {
                        blocks.push(
                            write_bytes(store, sink_id, segment_id, bytes.clone(), epoch).await?,
                        );
                    }
                    blocks
                }
                None => Vec::new(),
            };
            if sinks
                .insert(
                    sink_id.clone(),
                    SinkV4 {
                        delivery: entry.delivery.clone(),
                        pre_commit: entry.pre_commit.clone(),
                        blocks,
                    },
                )
                .is_some()
            {
                return Err(CalcFlowError::CheckpointMismatch {
                    message: format!("sink {sink_id:?} is bound to multiple outputs"),
                });
            }
        }
    }
    Ok(sinks)
}

async fn write_bytes(
    store: &mut V4Store,
    owner: &str,
    object_id: &str,
    bytes: Vec<u8>,
    epoch: u64,
) -> Result<V4Handle> {
    let sha256 = hex::encode(Sha256::digest(&bytes));
    let object = OpaqueObject {
        owner: owner.into(),
        object_id: object_id.into(),
        generation: epoch,
        bytes,
        sha256,
    };
    let mut encoded = store.write_opaque(epoch, vec![object]).await?;
    Ok(encoded.remove(0).handle)
}

fn validate_participants(
    checkpoint: &OpenedCheckpointRuntime,
    manifest: &V4Manifest,
    view: AssemblyView<'_>,
) -> Result<()> {
    let expected_sinks = view
        .sink_outputs
        .values()
        .flat_map(|entries| entries.keys().cloned())
        .collect::<BTreeSet<_>>();
    if manifest.sources.keys().cloned().collect::<BTreeSet<_>>() != checkpoint.identity.source_ids
        || manifest.operators.keys().cloned().collect::<BTreeSet<_>>()
            != checkpoint.identity.operator_ids
        || expected_sinks != checkpoint.identity.sink_ids
    {
        return Err(CalcFlowError::CheckpointMismatch {
            message: "v4 manifest participants differ from the prepared job".into(),
        });
    }
    Ok(())
}
