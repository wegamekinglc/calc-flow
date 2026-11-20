//! Live runner persistence for opaque checkpoint-v4 participant state.

use std::{
    collections::{BTreeMap, BTreeSet},
    path::PathBuf,
};

use chrono::Utc;
use sha2::{Digest, Sha256};

use super::super::checkpoint_runtime::OpenedCheckpointRuntime;
use crate::runtime::streaming::morsel_pool::{
    ManagedMorsel, MorselCost, MorselLease, execute_managed,
};
use crate::state::v4::{
    EncodedObject, JobKind, NewObject, ObjectPayload, OpaqueObject, OperatorV4, SinkV4, SourceV4,
    V4Handle, V4Manifest, V4OpaqueAdapter, V4Store, encode_object, encode_opaque,
    manifest_fingerprint,
    native::{NativeCapture, V4ObjectStore, load_native, persist_native},
};
use crate::state::{OperatorManifestEntry, SinkManifestEntry, SourceManifestEntry};
use crate::{CalcFlowError, EventTime, OperatorStateSnapshot, Result, StreamJobContext};

#[derive(Clone, Copy)]
pub(super) struct AssemblyView<'a> {
    pub sources: &'a BTreeMap<String, SourceManifestEntry>,
    pub operator_snapshots: &'a BTreeMap<String, OperatorStateSnapshot>,
    pub native_captures: &'a BTreeMap<String, NativeCapture>,
    pub operator_progress: &'a BTreeMap<String, OperatorManifestEntry>,
    pub operator_frontiers: &'a BTreeMap<String, Option<EventTime>>,
    pub sink_outputs: &'a BTreeMap<String, BTreeMap<String, SinkManifestEntry>>,
    pub sink_captures: &'a BTreeMap<String, BTreeMap<String, Vec<u8>>>,
}

struct OpaqueEncodingWork {
    objects: Vec<OpaqueObject>,
    root: PathBuf,
    epoch: u64,
}

impl OpaqueEncodingWork {
    fn run(self, lease: &MorselLease) -> Result<Vec<EncodedObject>> {
        self.objects
            .into_iter()
            .map(|object| {
                lease.check()?;
                encode_opaque(&object, &self.root, self.epoch)
            })
            .collect()
    }
}

async fn encode_objects(
    job: &StreamJobContext,
    owner: &str,
    objects: Vec<OpaqueObject>,
    root: PathBuf,
    epoch: u64,
) -> Result<Vec<EncodedObject>> {
    let cost = MorselCost {
        rows: objects.len(),
        scratch_bytes: 0,
        output_bytes: objects.iter().map(|object| object.bytes.len()).sum(),
    };
    execute_managed(
        job,
        owner,
        None,
        ManagedMorsel::new(
            OpaqueEncodingWork {
                objects,
                root,
                epoch,
            },
            cost,
            OpaqueEncodingWork::run,
        ),
        |error| error,
    )
    .await
}

struct NativeEncodingWork {
    objects: Vec<NewObject>,
    root: PathBuf,
    epoch: u64,
}

impl NativeEncodingWork {
    fn run(self, lease: &MorselLease) -> Result<Vec<EncodedObject>> {
        self.objects
            .into_iter()
            .map(|object| {
                lease.check()?;
                encode_object(&object, &self.root, self.epoch)
            })
            .collect()
    }
}

struct ManagedNativeStore<'a> {
    store: &'a mut V4Store,
    job: &'a StreamJobContext,
}

impl<'a> ManagedNativeStore<'a> {
    fn new(store: &'a mut V4Store, job: &'a StreamJobContext) -> Self {
        Self { store, job }
    }
}

#[async_trait::async_trait]
impl V4ObjectStore for ManagedNativeStore<'_> {
    async fn write_objects(
        &mut self,
        epoch: u64,
        objects: Vec<NewObject>,
    ) -> Result<Vec<EncodedObject>> {
        let cost = MorselCost {
            rows: objects.len(),
            scratch_bytes: 0,
            output_bytes: 0,
        };
        let encoded = execute_managed(
            self.job,
            "native-v4-objects",
            None,
            ManagedMorsel::new(
                NativeEncodingWork {
                    objects,
                    root: self.store.root().to_owned(),
                    epoch,
                },
                cost,
                NativeEncodingWork::run,
            ),
            |error| error,
        )
        .await?;
        self.store.write_encoded_objects(encoded).await
    }

    fn record_carried(&mut self, count: usize) {
        self.store.record_carried(count);
    }

    fn read_epoch_manifest(&self, epoch: u64) -> Result<Option<V4Manifest>> {
        self.store.read_epoch_manifest(epoch)
    }

    fn read_object(&mut self, handle: &V4Handle) -> Result<ObjectPayload> {
        self.store.read_object(handle)
    }
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
                match &manifest.operators[operator_id] {
                    OperatorV4::Table { .. } => load_native(&mut store, &manifest, operator_id),
                    OperatorV4::Opaque { .. } => {
                        V4OpaqueAdapter::restore(&mut store, &manifest, operator_id)
                    }
                }
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
    job: &StreamJobContext,
    view: AssemblyView<'_>,
    epoch: u64,
    cancellation: &crate::CancellationToken,
) -> Result<PathBuf> {
    #[cfg(not(test))]
    let _ = cancellation;
    let mut store = V4Store::new(store_root(checkpoint));
    let sources = persist_sources(view.sources);
    let operators = persist_operators(&mut store, job, view, epoch).await?;
    let sinks = persist_sinks(&mut store, job, view, epoch).await?;
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
    let publication = tokio::task::spawn_blocking(move || store.publish_manifest(&manifest))
        .await
        .map_err(|error| CalcFlowError::Internal {
            message: format!("checkpoint-v4 manifest worker failed: {error}"),
        })?;
    let path = match publication {
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

fn persist_sources(sources: &BTreeMap<String, SourceManifestEntry>) -> BTreeMap<String, SourceV4> {
    sources
        .iter()
        .map(|(source_id, entry)| {
            (
                source_id.clone(),
                SourceV4 {
                    cursor: entry.cursor.clone(),
                    identity_hash: entry.identity_hash.clone(),
                    next_sequence: entry.sequence,
                    ended: entry.ended,
                    watermark_policy: entry.watermark_policy.clone(),
                    history: entry.history.clone(),
                },
            )
        })
        .collect()
}

async fn persist_operators(
    store: &mut V4Store,
    job: &StreamJobContext,
    view: AssemblyView<'_>,
    epoch: u64,
) -> Result<BTreeMap<String, OperatorV4>> {
    let mut operators = BTreeMap::new();
    for (node_id, snapshot) in view.operator_snapshots {
        let progress = view.operator_progress[node_id].progress.clone();
        let output_frontier = view.operator_frontiers.get(node_id).copied().flatten();
        let operator = if let Some(capture) = view.native_captures.get(node_id) {
            let mut managed = ManagedNativeStore::new(store, job);
            persist_native(
                &mut managed,
                node_id,
                capture.clone(),
                epoch,
                progress,
                output_frontier,
            )
            .await?
        } else {
            let adapter = V4OpaqueAdapter::from_snapshot(node_id, snapshot);
            let encoded = encode_objects(
                job,
                node_id,
                adapter.blocks_for_encoding(),
                store.root().to_owned(),
                epoch,
            )
            .await?;
            let (mut operator, _) = adapter.persist_preencoded(store, encoded).await?;
            if let OperatorV4::Opaque {
                progress: current,
                output_frontier: frontier,
                ..
            } = &mut operator
            {
                *current = progress;
                *frontier = output_frontier;
            }
            operator
        };
        operators.insert(node_id.clone(), operator);
    }
    Ok(operators)
}

async fn persist_sinks(
    store: &mut V4Store,
    job: &StreamJobContext,
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
                            write_bytes(store, job, sink_id, segment_id, bytes.clone(), epoch)
                                .await?,
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
    job: &StreamJobContext,
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
    let encoded = encode_objects(job, owner, vec![object], store.root().to_owned(), epoch).await?;
    let mut encoded = store.write_encoded_opaque(encoded).await?;
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

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn opaque_write_requires_an_open_job_morsel_pool() {
        let directory = tempfile::tempdir().unwrap();
        let mut store = V4Store::new(directory.path().join("lineage"));
        let job = StreamJobContext::new(
            951,
            "v4-morsel-write",
            crate::JsonMap::new(),
            None,
            crate::CancellationToken::new(),
        );
        job.managed_morsel_owner().close_and_drain().await.unwrap();

        let result = write_bytes(&mut store, &job, "sink", "segment", b"payload".to_vec(), 7).await;
        assert!(matches!(result, Err(CalcFlowError::Cancelled { .. })));

        let job = StreamJobContext::new(
            952,
            "v4-morsel-write",
            crate::JsonMap::new(),
            None,
            crate::CancellationToken::new(),
        );
        let handle = write_bytes(&mut store, &job, "sink", "segment", b"payload".to_vec(), 7)
            .await
            .unwrap();
        assert_eq!(store.read_opaque(&handle).unwrap(), b"payload");
        assert_eq!(store.stats().opaque_blocks, 1);
    }

    #[tokio::test]
    async fn native_write_requires_an_open_job_morsel_pool() {
        use crate::state::v4::native::{NativePage, NativePageSet, NativePartition};

        let directory = tempfile::tempdir().unwrap();
        let mut store = V4Store::new(directory.path().join("lineage"));
        let job = StreamJobContext::new(
            953,
            "v4-native-morsel-write",
            crate::JsonMap::new(),
            None,
            crate::CancellationToken::new(),
        );
        job.managed_morsel_owner().close_and_drain().await.unwrap();
        let capture = || {
            NativeCapture::new(
                crate::JsonMap::new(),
                vec![NativePartition {
                    key_group: 0,
                    pages: NativePageSet::full(vec![NativePage::json(
                        "control",
                        serde_json::json!({"rows": 1}),
                        0,
                        0..0,
                    )])
                    .unwrap(),
                }],
            )
            .unwrap()
        };
        let result = {
            let mut managed = ManagedNativeStore::new(&mut store, &job);
            persist_native(&mut managed, "join", capture(), 1, BTreeMap::new(), None).await
        };
        assert!(matches!(result, Err(CalcFlowError::Cancelled { .. })));

        let job = StreamJobContext::new(
            954,
            "v4-native-morsel-write",
            crate::JsonMap::new(),
            None,
            crate::CancellationToken::new(),
        );
        let mut managed = ManagedNativeStore::new(&mut store, &job);
        let operator = persist_native(&mut managed, "join", capture(), 1, BTreeMap::new(), None)
            .await
            .unwrap();
        let OperatorV4::Table { partitions, .. } = operator else {
            panic!("native write must produce table state")
        };
        assert_eq!(partitions.len(), 1);
        assert!(managed.store.stats().content_hashes > 0);
    }
}
