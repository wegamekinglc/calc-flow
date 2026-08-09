#![allow(
    dead_code,
    reason = "M5 manifest transactions are wired into private checkpoint coordination incrementally"
)]

use std::{
    any::Any,
    collections::{BTreeMap, BTreeSet},
    fs::File,
    future::Future,
    io::Write as _,
    panic::{AssertUnwindSafe, catch_unwind},
    path::{Path, PathBuf},
    sync::Arc,
};

use sha2::{Digest as _, Sha256};
use tokio::sync::Mutex;

use super::{
    CheckpointManifest, ManifestExpectation, StateHandle, StateLineageBackend, StateLineageKey,
};
use crate::{
    CalcFlowError, CancellationToken, Epoch, JsonMap, OperatorManifestEntry, OperatorStateSnapshot,
    Result,
};

const MAX_MANIFEST_ENTRIES: usize = 4_096;

#[cfg(test)]
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) enum ManifestTransactionFaultPoint {
    StateStage,
    ManifestWrite,
    ManifestRename,
    ManifestParentSync,
    Compaction,
}

#[cfg(test)]
type ManifestTransactionFaultHook =
    Arc<dyn Fn(ManifestTransactionFaultPoint) -> Result<()> + Send + Sync>;

#[cfg(test)]
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum ManifestOperationPoint {
    Open,
    Stage,
    Load,
    Select,
    Publish,
    Retain,
}

#[cfg(test)]
type ManifestOperationHook = Arc<dyn Fn(ManifestOperationPoint) -> Result<()> + Send + Sync>;

#[derive(Clone)]
pub(crate) struct PreparedManifestIdentity {
    pub(crate) pipeline_name: String,
    pub(crate) pipeline_fingerprint: String,
    pub(crate) runtime_config_hash: String,
    pub(crate) source_ids: BTreeSet<String>,
    pub(crate) operator_ids: BTreeSet<String>,
    pub(crate) sink_ids: BTreeSet<String>,
}

impl PreparedManifestIdentity {
    fn expectation(&self, epoch: Epoch) -> ManifestExpectation<'_> {
        ManifestExpectation {
            pipeline_name: &self.pipeline_name,
            pipeline_fingerprint: &self.pipeline_fingerprint,
            runtime_config_hash: &self.runtime_config_hash,
            epoch,
            source_ids: &self.source_ids,
            operator_ids: &self.operator_ids,
            sink_ids: &self.sink_ids,
        }
    }
}

pub(crate) struct PreparedEpochManifest {
    pub(crate) manifest: CheckpointManifest,
    pub(crate) staged_segments: BTreeMap<StateHandle, Vec<u8>>,
}

#[derive(Debug)]
pub(crate) enum ManifestPublication {
    Durable,
    Installed {
        parent_synced: bool,
        error: CalcFlowError,
    },
}

pub(crate) struct StagedOperatorState {
    pub(crate) inline_metadata: JsonMap,
    pub(crate) segments: Vec<StateHandle>,
}

pub(crate) struct ManifestValidation {
    pub(crate) runtime_config_changed: bool,
}

pub(crate) struct SelectedManifest {
    pub(crate) manifest: CheckpointManifest,
    pub(crate) validation: ManifestValidation,
    pub(crate) next_epoch: Epoch,
}

pub(crate) struct RetentionReport {
    pub(crate) retained_manifests: usize,
    pub(crate) removed_manifests: usize,
    pub(crate) removed_orphan_segments: usize,
}

pub(crate) struct ManifestTransaction {
    lineage: Arc<dyn StateLineageBackend>,
    lineage_hash: String,
    manifest_root: PathBuf,
    retained_epochs: usize,
    operation: Mutex<()>,
    #[cfg(test)]
    fault_hook: Option<ManifestTransactionFaultHook>,
    #[cfg(test)]
    operation_hook: Option<ManifestOperationHook>,
}

impl ManifestTransaction {
    pub(crate) async fn open(
        lineage: Arc<dyn StateLineageBackend>,
        lineage_key: &StateLineageKey,
        manifest_root: impl AsRef<Path>,
        retained_epochs: usize,
    ) -> Result<Self> {
        Self::open_cancellable(
            lineage,
            lineage_key,
            manifest_root,
            retained_epochs,
            &CancellationToken::new(),
        )
        .await
    }

    pub(crate) async fn open_cancellable(
        lineage: Arc<dyn StateLineageBackend>,
        lineage_key: &StateLineageKey,
        manifest_root: impl AsRef<Path>,
        retained_epochs: usize,
        cancellation: &CancellationToken,
    ) -> Result<Self> {
        Self::open_cancellable_inner(
            lineage,
            lineage_key,
            manifest_root,
            retained_epochs,
            cancellation,
            #[cfg(test)]
            None,
        )
        .await
    }

    #[cfg(test)]
    async fn open_cancellable_with_operation_hook(
        lineage: Arc<dyn StateLineageBackend>,
        lineage_key: &StateLineageKey,
        manifest_root: impl AsRef<Path>,
        retained_epochs: usize,
        cancellation: &CancellationToken,
        operation_hook: ManifestOperationHook,
    ) -> Result<Self> {
        Self::open_cancellable_inner(
            lineage,
            lineage_key,
            manifest_root,
            retained_epochs,
            cancellation,
            Some(operation_hook),
        )
        .await
    }

    async fn open_cancellable_inner(
        lineage: Arc<dyn StateLineageBackend>,
        lineage_key: &StateLineageKey,
        manifest_root: impl AsRef<Path>,
        retained_epochs: usize,
        cancellation: &CancellationToken,
        #[cfg(test)] operation_hook: Option<ManifestOperationHook>,
    ) -> Result<Self> {
        validate_retained_epochs(retained_epochs)?;
        let requested = manifest_root.as_ref().to_owned();
        create_manifest_root(&requested, cancellation).await?;
        #[cfg(test)]
        if let Some(hook) = operation_hook {
            owner_settled(
                cancellation,
                "manifest-open-hook",
                worker(move || hook(ManifestOperationPoint::Open)),
            )
            .await?;
        }
        let manifest_root = validate_and_canonicalize_root(requested, cancellation).await?;
        Ok(Self {
            lineage,
            lineage_hash: digest(
                format!(
                    "{}\0{}",
                    lineage_key.pipeline_name(),
                    lineage_key.pipeline_fingerprint()
                )
                .as_bytes(),
            ),
            manifest_root,
            retained_epochs,
            operation: Mutex::new(()),
            #[cfg(test)]
            fault_hook: None,
            #[cfg(test)]
            operation_hook: None,
        })
    }

    #[cfg(test)]
    pub(crate) fn with_fault_hook(mut self, hook: ManifestTransactionFaultHook) -> Self {
        self.fault_hook = Some(hook);
        self
    }

    #[cfg(test)]
    fn with_operation_hook(mut self, hook: ManifestOperationHook) -> Self {
        self.operation_hook = Some(hook);
        self
    }

    #[cfg(test)]
    fn inject_fault(&self, point: ManifestTransactionFaultPoint) -> Result<()> {
        self.fault_hook.as_ref().map_or(Ok(()), |hook| hook(point))
    }
    #[cfg(test)]
    async fn settle_operation_hook(
        &self,
        point: ManifestOperationPoint,
        cancellation: &CancellationToken,
    ) -> Result<()> {
        let Some(hook) = self.operation_hook.clone() else {
            return Ok(());
        };
        owner_settled(
            cancellation,
            "manifest-operation-hook",
            worker(move || hook(point)),
        )
        .await
    }

    pub(crate) async fn publish(
        &self,
        prepared: PreparedEpochManifest,
    ) -> Result<ManifestPublication> {
        self.publish_cancellable(prepared, &CancellationToken::new())
            .await
    }

    pub(crate) async fn publish_cancellable(
        &self,
        prepared: PreparedEpochManifest,
        cancellation: &CancellationToken,
    ) -> Result<ManifestPublication> {
        let _guard = owner_settled(cancellation, "manifest-publish-lock", async {
            Ok(self.operation.lock().await)
        })
        .await?;
        #[cfg(test)]
        self.settle_operation_hook(ManifestOperationPoint::Publish, cancellation)
            .await?;
        self.stage_prepared_segments(&prepared.staged_segments, cancellation)
            .await?;
        self.validate_prepared_segments(&prepared.staged_segments, cancellation)
            .await?;
        self.publish_prepared_segments(&prepared.staged_segments, cancellation)
            .await?;
        owner_settled(
            cancellation,
            "manifest-publish-segment-read",
            validate_manifest_segments(self.lineage.as_ref(), &prepared.manifest),
        )
        .await?;
        let epoch = prepared.manifest.epoch();
        let manifest = prepared.manifest;
        let bytes = owner_settled(
            cancellation,
            "manifest-encode",
            worker(move || manifest.canonical_bytes()),
        )
        .await?;
        let root = self.manifest_root.clone();
        #[cfg(test)]
        let fault_hook = self.fault_hook.clone();
        owner_settled_publication(
            cancellation,
            worker(move || {
                publish_manifest(
                    &root,
                    epoch,
                    &bytes,
                    #[cfg(test)]
                    fault_hook.as_ref(),
                )
            }),
        )
        .await
    }

    async fn stage_prepared_segments(
        &self,
        segments: &BTreeMap<StateHandle, Vec<u8>>,
        cancellation: &CancellationToken,
    ) -> Result<()> {
        for (handle, bytes) in segments {
            owner_settled(
                cancellation,
                "manifest-publish-stage",
                self.lineage.stage_segment(handle, bytes),
            )
            .await?;
        }
        Ok(())
    }

    async fn validate_prepared_segments(
        &self,
        segments: &BTreeMap<StateHandle, Vec<u8>>,
        cancellation: &CancellationToken,
    ) -> Result<()> {
        for handle in segments.keys() {
            owner_settled(
                cancellation,
                "manifest-publish-validate",
                self.lineage.validate_segment(handle),
            )
            .await?;
        }
        Ok(())
    }

    async fn publish_prepared_segments(
        &self,
        segments: &BTreeMap<StateHandle, Vec<u8>>,
        cancellation: &CancellationToken,
    ) -> Result<()> {
        for handle in segments.keys() {
            owner_settled(
                cancellation,
                "manifest-publish-segment",
                self.lineage.publish_segment(handle),
            )
            .await?;
        }
        Ok(())
    }

    pub(crate) async fn stage_operator_state(
        &self,
        operator_id: &str,
        epoch: Epoch,
        snapshot: OperatorStateSnapshot,
    ) -> Result<StagedOperatorState> {
        self.stage_operator_state_cancellable(
            operator_id,
            epoch,
            snapshot,
            &CancellationToken::new(),
        )
        .await
    }

    pub(crate) async fn stage_operator_state_cancellable(
        &self,
        operator_id: &str,
        epoch: Epoch,
        snapshot: OperatorStateSnapshot,
        cancellation: &CancellationToken,
    ) -> Result<StagedOperatorState> {
        let _guard = owner_settled(cancellation, "state-stage-lock", async {
            Ok(self.operation.lock().await)
        })
        .await?;
        #[cfg(test)]
        self.settle_operation_hook(ManifestOperationPoint::Stage, cancellation)
            .await?;
        let operator_hash = digest(operator_id);
        let mut staged = Vec::with_capacity(snapshot.segments.len());
        let mut unpublished = Vec::with_capacity(snapshot.segments.len());
        for (segment_id, bytes) in snapshot.segments {
            let (handle, needs_publication) = self
                .stage_operator_segment(
                    operator_id,
                    epoch,
                    &operator_hash,
                    segment_id,
                    bytes,
                    cancellation,
                )
                .await?;
            if needs_publication {
                unpublished.push(handle.clone());
            }
            staged.push(handle);
        }
        self.validate_staged_segments(&unpublished, cancellation)
            .await?;
        self.publish_staged_segments(&unpublished, cancellation)
            .await?;
        #[cfg(test)]
        if !staged.is_empty() {
            self.inject_fault(ManifestTransactionFaultPoint::StateStage)?;
        }
        Ok(StagedOperatorState {
            inline_metadata: snapshot.inline_metadata,
            segments: staged,
        })
    }

    async fn stage_operator_segment(
        &self,
        operator_id: &str,
        epoch: Epoch,
        operator_hash: &str,
        segment_id: String,
        bytes: Vec<u8>,
        cancellation: &CancellationToken,
    ) -> Result<(StateHandle, bool)> {
        let segment_hash = digest(&segment_id);
        let relative_path = format!(
            "committed/{}/{operator_hash}/{}-{segment_hash}.segment",
            self.lineage_hash,
            epoch.as_u64()
        );
        let handle = StateHandle::new(
            operator_id,
            epoch,
            &segment_id,
            &relative_path,
            u64::try_from(bytes.len()).map_err(|_| CalcFlowError::InvalidArgument {
                field: format!("operators.{operator_id}.segments.{segment_id}"),
                message: "segment byte length does not fit u64".into(),
            })?,
            &digest(&bytes),
        )?;
        match owner_settled(
            cancellation,
            "state-stage-existing-read",
            self.lineage.load_segment(&handle),
        )
        .await
        {
            Ok(committed) if committed == bytes => Ok((handle, false)),
            Ok(_) => Err(CalcFlowError::CheckpointMismatch {
                message: format!(
                    "operator {operator_id:?} committed segment {segment_id:?} changed bytes"
                ),
            }),
            Err(CalcFlowError::NotFound { .. }) => {
                owner_settled(
                    cancellation,
                    "state-stage-write",
                    self.lineage.stage_segment(&handle, &bytes),
                )
                .await?;
                Ok((handle, true))
            }
            Err(error) => Err(error),
        }
    }

    async fn validate_staged_segments(
        &self,
        handles: &[StateHandle],
        cancellation: &CancellationToken,
    ) -> Result<()> {
        for handle in handles {
            owner_settled(
                cancellation,
                "state-stage-validate",
                self.lineage.validate_segment(handle),
            )
            .await?;
        }
        Ok(())
    }

    async fn publish_staged_segments(
        &self,
        handles: &[StateHandle],
        cancellation: &CancellationToken,
    ) -> Result<()> {
        for handle in handles {
            owner_settled(
                cancellation,
                "state-stage-publish",
                self.lineage.publish_segment(handle),
            )
            .await?;
        }
        Ok(())
    }

    pub(crate) async fn load_operator_state(
        &self,
        operator_id: &str,
        epoch: Epoch,
        entry: &OperatorManifestEntry,
    ) -> Result<OperatorStateSnapshot> {
        self.load_operator_state_cancellable(operator_id, epoch, entry, &CancellationToken::new())
            .await
    }

    pub(crate) async fn load_operator_state_cancellable(
        &self,
        operator_id: &str,
        epoch: Epoch,
        entry: &OperatorManifestEntry,
        cancellation: &CancellationToken,
    ) -> Result<OperatorStateSnapshot> {
        let _guard = owner_settled(cancellation, "state-load-lock", async {
            Ok(self.operation.lock().await)
        })
        .await?;
        #[cfg(test)]
        self.settle_operation_hook(ManifestOperationPoint::Load, cancellation)
            .await?;
        let mut segment_ids = BTreeSet::new();
        for handle in &entry.segments {
            handle.validate_for(operator_id, epoch)?;
            if !segment_ids.insert(handle.segment_id()) {
                return Err(CalcFlowError::CheckpointMismatch {
                    message: format!(
                        "operator {operator_id:?} repeats state segment {:?}",
                        handle.segment_id()
                    ),
                });
            }
        }
        let mut segments = BTreeMap::new();
        for handle in &entry.segments {
            let bytes = owner_settled(
                cancellation,
                "state-load-segment",
                self.lineage.load_segment(handle),
            )
            .await?;
            segments.insert(handle.segment_id().into(), bytes);
        }
        Ok(OperatorStateSnapshot {
            inline_metadata: entry.inline_metadata.clone(),
            segments,
        })
    }

    pub(crate) async fn select_latest(
        &self,
        identity: &PreparedManifestIdentity,
    ) -> Result<Option<SelectedManifest>> {
        self.select_latest_cancellable(identity, &CancellationToken::new())
            .await
    }

    pub(crate) async fn select_latest_cancellable(
        &self,
        identity: &PreparedManifestIdentity,
        cancellation: &CancellationToken,
    ) -> Result<Option<SelectedManifest>> {
        let _guard = owner_settled(cancellation, "manifest-select-lock", async {
            Ok(self.operation.lock().await)
        })
        .await?;
        #[cfg(test)]
        self.settle_operation_hook(ManifestOperationPoint::Select, cancellation)
            .await?;
        let root = self.manifest_root.clone();
        let candidates = manifest_candidates(root, "manifest-select-list", cancellation).await?;
        let mut latest = None;
        for candidate in candidates {
            let manifest = self
                .load_candidate_manifest(&candidate, identity, "manifest-select", cancellation)
                .await?;
            let expectation = identity.expectation(candidate.epoch);
            latest = Some(SelectedManifest {
                validation: ManifestValidation {
                    runtime_config_changed: manifest.runtime_config_changed(&expectation),
                },
                next_epoch: candidate.epoch.next()?,
                manifest,
            });
        }
        Ok(latest)
    }

    pub(crate) async fn retain(
        &self,
        identity: &PreparedManifestIdentity,
        in_flight: Option<&CheckpointManifest>,
    ) -> Result<RetentionReport> {
        self.retain_cancellable(identity, in_flight, &CancellationToken::new())
            .await
    }

    pub(crate) async fn retain_cancellable(
        &self,
        identity: &PreparedManifestIdentity,
        in_flight: Option<&CheckpointManifest>,
        cancellation: &CancellationToken,
    ) -> Result<RetentionReport> {
        let _guard = owner_settled(cancellation, "manifest-retain-lock", async {
            Ok(self.operation.lock().await)
        })
        .await?;
        #[cfg(test)]
        self.settle_operation_hook(ManifestOperationPoint::Retain, cancellation)
            .await?;
        let root = self.manifest_root.clone();
        let candidates = manifest_candidates(root, "manifest-retain-list", cancellation).await?;
        let mut manifests = Vec::with_capacity(candidates.len());
        for candidate in candidates {
            let manifest = self
                .load_candidate_manifest(&candidate, identity, "manifest-retain", cancellation)
                .await?;
            manifests.push((candidate, manifest));
        }

        let removed_manifests = manifests.len().saturating_sub(self.retained_epochs);
        let retained = &manifests[removed_manifests..];
        let mut retained_handles = BTreeSet::new();
        for (_, manifest) in retained {
            collect_manifest_handles(manifest, &mut retained_handles);
        }
        if let Some(manifest) = in_flight {
            collect_manifest_handles(manifest, &mut retained_handles);
        }
        let removals = manifests[..removed_manifests]
            .iter()
            .map(|(candidate, _)| candidate.path.clone())
            .collect::<Vec<_>>();
        let root = self.manifest_root.clone();
        owner_settled(
            cancellation,
            "manifest-retain-remove",
            worker(move || remove_manifest_files(&root, &removals)),
        )
        .await?;
        let retained_handles = retained_handles.into_iter().collect::<Vec<_>>();
        #[cfg(test)]
        self.inject_fault(ManifestTransactionFaultPoint::Compaction)?;
        let removed_orphan_segments = owner_settled(
            cancellation,
            "manifest-retain-orphans",
            self.lineage.collect_orphans(&retained_handles),
        )
        .await?;
        Ok(RetentionReport {
            retained_manifests: retained.len(),
            removed_manifests,
            removed_orphan_segments,
        })
    }

    async fn load_candidate_manifest(
        &self,
        candidate: &ManifestCandidate,
        identity: &PreparedManifestIdentity,
        operation: &str,
        cancellation: &CancellationToken,
    ) -> Result<CheckpointManifest> {
        let bytes = owner_settled(cancellation, &format!("{operation}-read"), async {
            tokio::fs::read(&candidate.path)
                .await
                .map_err(|source| io_error(&candidate.path, source))
        })
        .await?;
        let manifest = owner_settled(
            cancellation,
            &format!("{operation}-decode"),
            worker(move || CheckpointManifest::from_bytes(&bytes)),
        )
        .await?;
        validate_candidate_epoch(candidate, &manifest)?;
        manifest.validate(&identity.expectation(candidate.epoch))?;
        owner_settled(
            cancellation,
            &format!("{operation}-segments"),
            validate_manifest_segments(self.lineage.as_ref(), &manifest),
        )
        .await?;
        Ok(manifest)
    }

    #[cfg(test)]
    fn manifest_path(&self, epoch: Epoch) -> PathBuf {
        manifest_path(&self.manifest_root, epoch)
    }
}

fn validate_retained_epochs(retained_epochs: usize) -> Result<()> {
    if retained_epochs == 0 {
        Err(CalcFlowError::InvalidArgument {
            field: "retained_epochs".into(),
            message: "must be positive".into(),
        })
    } else {
        Ok(())
    }
}

async fn create_manifest_root(requested: &Path, cancellation: &CancellationToken) -> Result<()> {
    owner_settled(cancellation, "manifest-open-create", async {
        tokio::fs::create_dir_all(requested)
            .await
            .map_err(|source| io_error(requested, source))
    })
    .await
}

async fn validate_and_canonicalize_root(
    requested: PathBuf,
    cancellation: &CancellationToken,
) -> Result<PathBuf> {
    let checked = requested.clone();
    owner_settled(
        cancellation,
        "manifest-open-validate",
        worker(move || validate_directory(&checked)),
    )
    .await?;
    owner_settled(cancellation, "manifest-open-canonicalize", async {
        tokio::fs::canonicalize(&requested)
            .await
            .map_err(|source| io_error(&requested, source))
    })
    .await
}

async fn manifest_candidates(
    root: PathBuf,
    operation: &str,
    cancellation: &CancellationToken,
) -> Result<Vec<ManifestCandidate>> {
    owner_settled(
        cancellation,
        operation,
        worker(move || list_manifest_candidates(&root)),
    )
    .await
}

fn validate_candidate_epoch(
    candidate: &ManifestCandidate,
    manifest: &CheckpointManifest,
) -> Result<()> {
    if manifest.epoch() == candidate.epoch {
        Ok(())
    } else {
        Err(CalcFlowError::CheckpointMismatch {
            message: format!(
                "manifest file epoch {} does not match document epoch {}",
                candidate.epoch.as_u64(),
                manifest.epoch().as_u64()
            ),
        })
    }
}

fn digest(bytes: impl AsRef<[u8]>) -> String {
    hex::encode(Sha256::digest(bytes.as_ref()))
}

struct ManifestCandidate {
    epoch: Epoch,
    path: PathBuf,
}

async fn validate_manifest_segments(
    lineage: &dyn StateLineageBackend,
    manifest: &CheckpointManifest,
) -> Result<()> {
    for operator in manifest.operators().values() {
        for handle in &operator.segments {
            lineage.load_segment(handle).await?;
        }
    }
    Ok(())
}

fn collect_manifest_handles(manifest: &CheckpointManifest, retained: &mut BTreeSet<StateHandle>) {
    for operator in manifest.operators().values() {
        retained.extend(operator.segments.iter().cloned());
    }
}

fn list_manifest_candidates(root: &Path) -> Result<Vec<ManifestCandidate>> {
    validate_directory(root)?;
    let mut candidates = Vec::new();
    let mut temporaries = Vec::new();
    let mut entry_count = 0_usize;
    for entry in std::fs::read_dir(root).map_err(|source| io_error(root, source))? {
        let entry = entry.map_err(|source| io_error(root, source))?;
        entry_count = bounded_manifest_entry_count(entry_count)?;
        match classify_manifest_entry(&entry)? {
            ManifestDirectoryEntry::Candidate(candidate) => candidates.push(candidate),
            ManifestDirectoryEntry::Temporary(path) => temporaries.push(path),
        }
    }
    remove_manifest_temporaries(root, &temporaries)?;
    candidates.sort_by_key(|candidate| candidate.epoch);
    Ok(candidates)
}

enum ManifestDirectoryEntry {
    Candidate(ManifestCandidate),
    Temporary(PathBuf),
}

fn bounded_manifest_entry_count(current: usize) -> Result<usize> {
    let count = current
        .checked_add(1)
        .ok_or_else(|| format_error("manifest directory entry count overflowed usize".into()))?;
    if count > MAX_MANIFEST_ENTRIES {
        Err(format_error(format!(
            "manifest directory entry count exceeds {MAX_MANIFEST_ENTRIES}"
        )))
    } else {
        Ok(count)
    }
}

fn classify_manifest_entry(entry: &std::fs::DirEntry) -> Result<ManifestDirectoryEntry> {
    let path = entry.path();
    let metadata = std::fs::symlink_metadata(&path).map_err(|source| io_error(&path, source))?;
    if metadata.file_type().is_symlink() || !metadata.is_file() {
        return Err(format_error(format!(
            "manifest entry {} is not a regular file",
            path.display()
        )));
    }
    let name = entry
        .file_name()
        .into_string()
        .map_err(|_| format_error("manifest file name is not UTF-8".into()))?;
    if name.starts_with(".tmp") {
        Ok(ManifestDirectoryEntry::Temporary(path))
    } else {
        Ok(ManifestDirectoryEntry::Candidate(ManifestCandidate {
            epoch: parse_manifest_name(&name)?,
            path,
        }))
    }
}

fn parse_manifest_name(name: &str) -> Result<Epoch> {
    let value = name
        .strip_prefix("manifest-")
        .and_then(|value| value.strip_suffix(".json"))
        .filter(|value| value.len() == 20 && value.bytes().all(|byte| byte.is_ascii_digit()))
        .and_then(|value| value.parse::<u64>().ok())
        .and_then(Epoch::new);
    value.ok_or_else(|| format_error(format!("unexpected manifest file {name:?}")))
}

fn manifest_path(root: &Path, epoch: Epoch) -> PathBuf {
    root.join(format!("manifest-{:020}.json", epoch.as_u64()))
}

fn publish_manifest(
    root: &Path,
    epoch: Epoch,
    bytes: &[u8],
    #[cfg(test)] fault_hook: Option<&ManifestTransactionFaultHook>,
) -> Result<ManifestPublication> {
    validate_directory(root)?;
    let destination = manifest_path(root, epoch);
    if let Some(publication) = existing_manifest_publication(&destination, epoch, bytes)? {
        return Ok(publication);
    }
    let mut parent_synced = false;
    let publication = catch_unwind(AssertUnwindSafe(|| {
        install_manifest(
            root,
            &destination,
            bytes,
            &mut parent_synced,
            #[cfg(test)]
            fault_hook,
        )
    }));
    match publication {
        Ok(Ok(())) => Ok(ManifestPublication::Durable),
        Ok(Err(error)) => classify_failed_publication(&destination, bytes, parent_synced, error),
        Err(payload) => classify_failed_publication(
            &destination,
            bytes,
            parent_synced,
            CalcFlowError::Internal {
                message: format!(
                    "manifest filesystem worker panicked: {}",
                    panic_payload_message(payload.as_ref())
                ),
            },
        ),
    }
}

fn existing_manifest_publication(
    destination: &Path,
    epoch: Epoch,
    bytes: &[u8],
) -> Result<Option<ManifestPublication>> {
    let Ok(metadata) = std::fs::symlink_metadata(destination) else {
        return Ok(None);
    };
    if metadata.file_type().is_symlink() || !metadata.is_file() {
        return Err(format_error(format!(
            "manifest entry {} is not a regular file",
            destination.display()
        )));
    }
    let existing = std::fs::read(destination).map_err(|source| io_error(destination, source))?;
    if existing == bytes {
        Ok(Some(ManifestPublication::Durable))
    } else {
        Err(CalcFlowError::Conflict {
            resource: "checkpoint manifest".into(),
            key: epoch.as_u64().to_string(),
        })
    }
}

fn install_manifest(
    root: &Path,
    destination: &Path,
    bytes: &[u8],
    parent_synced: &mut bool,
    #[cfg(test)] fault_hook: Option<&ManifestTransactionFaultHook>,
) -> Result<()> {
    let mut temporary =
        tempfile::NamedTempFile::new_in(root).map_err(|source| io_error(root, source))?;
    temporary
        .write_all(bytes)
        .map_err(|source| io_error(temporary.path(), source))?;
    temporary
        .flush()
        .map_err(|source| io_error(temporary.path(), source))?;
    temporary
        .as_file()
        .sync_all()
        .map_err(|source| io_error(temporary.path(), source))?;
    #[cfg(test)]
    if let Some(hook) = fault_hook {
        hook(ManifestTransactionFaultPoint::ManifestWrite)?;
    }
    temporary
        .persist_noclobber(destination)
        .map_err(|error| io_error(destination, error.error))?;
    #[cfg(test)]
    if let Some(hook) = fault_hook {
        hook(ManifestTransactionFaultPoint::ManifestRename)?;
    }
    sync_directory(root)?;
    *parent_synced = true;
    #[cfg(test)]
    if let Some(hook) = fault_hook {
        hook(ManifestTransactionFaultPoint::ManifestParentSync)?;
    }
    Ok(())
}

fn classify_failed_publication(
    destination: &Path,
    expected_bytes: &[u8],
    parent_synced: bool,
    error: CalcFlowError,
) -> Result<ManifestPublication> {
    match std::fs::read(destination) {
        Ok(actual_bytes) if actual_bytes == expected_bytes => Ok(ManifestPublication::Installed {
            parent_synced,
            error,
        }),
        Ok(_) => Err(CalcFlowError::Conflict {
            resource: "checkpoint manifest".into(),
            key: destination.display().to_string(),
        }),
        Err(source) if source.kind() == std::io::ErrorKind::NotFound => Err(error),
        Err(source) => Err(io_error(destination, source)),
    }
}

fn panic_payload_message(payload: &(dyn Any + Send)) -> &str {
    payload
        .downcast_ref::<&str>()
        .copied()
        .or_else(|| payload.downcast_ref::<String>().map(String::as_str))
        .unwrap_or("non-string panic payload")
}

fn remove_manifest_files(root: &Path, removals: &[PathBuf]) -> Result<()> {
    validate_directory(root)?;
    for path in removals {
        remove_manifest_file(root, path)?;
    }
    if !removals.is_empty() {
        sync_directory(root)?;
    }
    Ok(())
}

fn remove_manifest_file(root: &Path, path: &Path) -> Result<()> {
    if path.parent() != Some(root) {
        return Err(format_error(
            "manifest removal escaped the managed root".into(),
        ));
    }
    validate_manifest_regular_file(path, "manifest entry")?;
    parse_manifest_name(
        path.file_name()
            .and_then(|name| name.to_str())
            .ok_or_else(|| format_error("manifest file name is not UTF-8".into()))?,
    )?;
    std::fs::remove_file(path).map_err(|source| io_error(path, source))
}

fn remove_manifest_temporaries(root: &Path, temporaries: &[PathBuf]) -> Result<()> {
    validate_directory(root)?;
    for path in temporaries {
        remove_manifest_temporary(root, path)?;
    }
    if !temporaries.is_empty() {
        sync_directory(root)?;
    }
    Ok(())
}

fn remove_manifest_temporary(root: &Path, path: &Path) -> Result<()> {
    if path.parent() != Some(root) {
        return Err(format_error(
            "manifest temporary removal escaped the managed root".into(),
        ));
    }
    validate_manifest_regular_file(path, "manifest temporary")?;
    let name = path
        .file_name()
        .and_then(|name| name.to_str())
        .ok_or_else(|| format_error("manifest temporary name is not UTF-8".into()))?;
    if !name.starts_with(".tmp") {
        return Err(format_error(format!(
            "unexpected manifest temporary file {name:?}"
        )));
    }
    std::fs::remove_file(path).map_err(|source| io_error(path, source))
}

fn validate_manifest_regular_file(path: &Path, label: &str) -> Result<()> {
    let metadata = std::fs::symlink_metadata(path).map_err(|source| io_error(path, source))?;
    if metadata.file_type().is_symlink() || !metadata.is_file() {
        Err(format_error(format!(
            "{label} {} is not a regular file",
            path.display()
        )))
    } else {
        Ok(())
    }
}

fn validate_directory(path: &Path) -> Result<()> {
    let metadata = std::fs::symlink_metadata(path).map_err(|source| io_error(path, source))?;
    if metadata.file_type().is_symlink() || !metadata.is_dir() {
        Err(format_error(format!(
            "manifest root {} is not a directory",
            path.display()
        )))
    } else {
        Ok(())
    }
}

#[cfg(unix)]
fn sync_directory(directory: &Path) -> Result<()> {
    File::open(directory)
        .and_then(|file| file.sync_all())
        .map_err(|source| io_error(directory, source))
}

#[cfg(not(unix))]
fn sync_directory(_directory: &Path) -> Result<()> {
    Ok(())
}

async fn worker<T: Send + 'static>(
    operation: impl FnOnce() -> Result<T> + Send + 'static,
) -> Result<T> {
    tokio::task::spawn_blocking(operation)
        .await
        .map_err(|error| CalcFlowError::Internal {
            message: format!("manifest filesystem worker failed: {error}"),
        })?
}

async fn owner_settled<T>(
    cancellation: &CancellationToken,
    operation: &str,
    future: impl Future<Output = Result<T>>,
) -> Result<T> {
    if cancellation.is_cancelled() {
        return Err(cancellation_error(operation));
    }
    tokio::pin!(future);
    tokio::select! {
        biased;
        () = cancellation.cancelled() => {
            let _ = future.await;
            Err(cancellation_error(operation))
        }
        result = &mut future => result,
    }
}

async fn owner_settled_publication(
    cancellation: &CancellationToken,
    future: impl Future<Output = Result<ManifestPublication>>,
) -> Result<ManifestPublication> {
    if cancellation.is_cancelled() {
        return Err(cancellation_error("manifest-publish"));
    }
    tokio::pin!(future);
    tokio::select! {
        biased;
        () = cancellation.cancelled() => match future.await {
            Ok(publication) => Ok(publication),
            Err(_) => Err(cancellation_error("manifest-publish")),
        },
        result = &mut future => result,
    }
}

fn cancellation_error(operation: &str) -> CalcFlowError {
    CalcFlowError::Cancelled {
        run_id: format!("checkpoint:{operation}"),
    }
}

fn format_error(message: String) -> CalcFlowError {
    CalcFlowError::Format { message }
}

fn io_error(path: &Path, source: std::io::Error) -> CalcFlowError {
    CalcFlowError::Io {
        path: path.display().to_string(),
        source,
    }
}

#[cfg(test)]
mod tests {
    use std::{
        collections::{BTreeMap, BTreeSet},
        sync::{
            Arc,
            atomic::{AtomicUsize, Ordering},
        },
    };

    use chrono::{TimeZone, Utc};
    use tempfile::TempDir;

    use super::{
        ManifestOperationPoint, ManifestPublication, ManifestTransaction,
        ManifestTransactionFaultPoint, PreparedEpochManifest, PreparedManifestIdentity,
    };
    use crate::{
        CalcFlowError, CheckpointManifest, CheckpointManifestFields, Epoch, LocalStateBackend,
        OperatorManifestEntry, OperatorStateSnapshot, RecoveryStatus, StateBackend, StateHandle,
        StateLineageBackend, StateLineageKey,
    };

    const PIPELINE_FINGERPRINT: &str =
        "0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef";
    const RUNTIME_CONFIG_HASH: &str =
        "abcdef0123456789abcdef0123456789abcdef0123456789abcdef0123456789";

    fn manifest(epoch: Epoch) -> CheckpointManifest {
        CheckpointManifest::new(CheckpointManifestFields {
            pipeline_name: "orders".into(),
            pipeline_fingerprint: PIPELINE_FINGERPRINT.into(),
            runtime_config_hash: RUNTIME_CONFIG_HASH.into(),
            epoch,
            created_at: Utc.with_ymd_and_hms(2026, 8, 9, 8, 0, 0).unwrap(),
            recovery_status: RecoveryStatus::Final,
            sources: BTreeMap::new(),
            operators: BTreeMap::new(),
            sinks: BTreeMap::new(),
        })
        .unwrap()
    }

    fn identity() -> PreparedManifestIdentity {
        PreparedManifestIdentity {
            pipeline_name: "orders".into(),
            pipeline_fingerprint: PIPELINE_FINGERPRINT.into(),
            runtime_config_hash: RUNTIME_CONFIG_HASH.into(),
            source_ids: BTreeSet::default(),
            operator_ids: BTreeSet::default(),
            sink_ids: BTreeSet::default(),
        }
    }

    fn manifest_with_operator_segments(
        epoch: Epoch,
        segments: Vec<StateHandle>,
    ) -> CheckpointManifest {
        CheckpointManifest::new(CheckpointManifestFields {
            pipeline_name: "orders".into(),
            pipeline_fingerprint: PIPELINE_FINGERPRINT.into(),
            runtime_config_hash: RUNTIME_CONFIG_HASH.into(),
            epoch,
            created_at: Utc.with_ymd_and_hms(2026, 8, 9, 8, 0, 0).unwrap(),
            recovery_status: RecoveryStatus::Final,
            sources: BTreeMap::new(),
            operators: BTreeMap::from([(
                "window".into(),
                OperatorManifestEntry {
                    progress: BTreeMap::new(),
                    inline_metadata: BTreeMap::new(),
                    segments,
                },
            )]),
            sinks: BTreeMap::new(),
        })
        .unwrap()
    }

    fn identity_with_window() -> PreparedManifestIdentity {
        let mut prepared = identity();
        prepared.operator_ids.insert("window".into());
        prepared
    }

    fn fault_hook(
        selected: ManifestTransactionFaultPoint,
        triggers: Arc<AtomicUsize>,
    ) -> Arc<dyn Fn(ManifestTransactionFaultPoint) -> crate::Result<()> + Send + Sync> {
        Arc::new(move |point| {
            if point == selected {
                triggers.fetch_add(1, Ordering::SeqCst);
                return Err(CalcFlowError::Internal {
                    message: format!("transaction fault at {point:?}"),
                });
            }
            Ok(())
        })
    }

    #[tokio::test]
    async fn production_transaction_publishes_and_selects_one_complete_manifest() {
        let directory = TempDir::new().unwrap();
        let backend = LocalStateBackend::new(directory.path().join("state"))
            .await
            .unwrap();
        let key = StateLineageKey::new("orders", PIPELINE_FINGERPRINT).unwrap();
        let lineage: Arc<dyn StateLineageBackend> =
            Arc::from(backend.open_lineage(&key).await.unwrap());
        let transaction =
            ManifestTransaction::open(lineage, &key, directory.path().join("manifests"), 2)
                .await
                .unwrap();

        transaction
            .publish(PreparedEpochManifest {
                manifest: manifest(Epoch::INITIAL),
                staged_segments: BTreeMap::new(),
            })
            .await
            .unwrap();
        let selected = transaction
            .select_latest(&identity())
            .await
            .unwrap()
            .unwrap();

        assert_eq!(selected.manifest.epoch(), Epoch::INITIAL);
        assert!(!selected.validation.runtime_config_changed);
        assert_eq!(selected.next_epoch, Epoch::INITIAL.next().unwrap());

        let mut changed = identity();
        changed.runtime_config_hash =
            "fedcba9876543210fedcba9876543210fedcba9876543210fedcba9876543210".into();
        let selected = transaction.select_latest(&changed).await.unwrap().unwrap();
        assert!(selected.validation.runtime_config_changed);
    }

    #[tokio::test]
    async fn corrupt_higher_manifest_fails_instead_of_falling_back() {
        let directory = TempDir::new().unwrap();
        let backend = LocalStateBackend::new(directory.path().join("state"))
            .await
            .unwrap();
        let key = StateLineageKey::new("orders", PIPELINE_FINGERPRINT).unwrap();
        let lineage: Arc<dyn StateLineageBackend> =
            Arc::from(backend.open_lineage(&key).await.unwrap());
        let transaction =
            ManifestTransaction::open(lineage, &key, directory.path().join("manifests"), 2)
                .await
                .unwrap();
        transaction
            .publish(PreparedEpochManifest {
                manifest: manifest(Epoch::INITIAL),
                staged_segments: BTreeMap::new(),
            })
            .await
            .unwrap();
        let higher = Epoch::INITIAL.next().unwrap();
        tokio::fs::write(transaction.manifest_path(higher), b"{not-json")
            .await
            .unwrap();

        let Err(error) = transaction.select_latest(&identity()).await else {
            panic!("corrupt higher manifest unexpectedly selected recovery");
        };

        assert!(matches!(error, CalcFlowError::Format { .. }));
    }

    #[tokio::test]
    async fn temporary_higher_manifest_never_selects_recovery() {
        let directory = TempDir::new().unwrap();
        let backend = LocalStateBackend::new(directory.path().join("state"))
            .await
            .unwrap();
        let key = StateLineageKey::new("orders", PIPELINE_FINGERPRINT).unwrap();
        let lineage: Arc<dyn StateLineageBackend> =
            Arc::from(backend.open_lineage(&key).await.unwrap());
        let manifest_root = directory.path().join("manifests");
        let transaction = ManifestTransaction::open(lineage, &key, &manifest_root, 2)
            .await
            .unwrap();
        transaction
            .publish(PreparedEpochManifest {
                manifest: manifest(Epoch::INITIAL),
                staged_segments: BTreeMap::new(),
            })
            .await
            .unwrap();
        let temporary = manifest_root.join(".tmp-higher-manifest");
        tokio::fs::write(&temporary, b"{not-json").await.unwrap();
        drop(transaction);
        let transaction = ManifestTransaction::open(
            Arc::from(backend.open_lineage(&key).await.unwrap()),
            &key,
            &manifest_root,
            2,
        )
        .await
        .unwrap();

        let selected = transaction
            .select_latest(&identity())
            .await
            .unwrap()
            .unwrap();

        assert_eq!(selected.manifest.epoch(), Epoch::INITIAL);
        assert!(!temporary.exists());
    }

    #[tokio::test]
    async fn manifest_temporary_directory_fails_closed_without_cleanup() {
        let directory = TempDir::new().unwrap();
        let backend = LocalStateBackend::new(directory.path().join("state"))
            .await
            .unwrap();
        let key = StateLineageKey::new("orders", PIPELINE_FINGERPRINT).unwrap();
        let lineage: Arc<dyn StateLineageBackend> =
            Arc::from(backend.open_lineage(&key).await.unwrap());
        let manifest_root = directory.path().join("manifests");
        let transaction = ManifestTransaction::open(lineage, &key, &manifest_root, 2)
            .await
            .unwrap();
        let temporary = manifest_root.join(".tmp-directory");
        tokio::fs::create_dir(&temporary).await.unwrap();

        let Err(error) = transaction.select_latest(&identity()).await else {
            panic!("temporary directory unexpectedly passed manifest scan");
        };

        assert!(matches!(error, CalcFlowError::Format { .. }));
        assert!(temporary.is_dir());
    }

    #[cfg(unix)]
    #[tokio::test]
    async fn manifest_temporary_symlink_fails_closed_without_cleanup() {
        let directory = TempDir::new().unwrap();
        let backend = LocalStateBackend::new(directory.path().join("state"))
            .await
            .unwrap();
        let key = StateLineageKey::new("orders", PIPELINE_FINGERPRINT).unwrap();
        let lineage: Arc<dyn StateLineageBackend> =
            Arc::from(backend.open_lineage(&key).await.unwrap());
        let manifest_root = directory.path().join("manifests");
        let transaction = ManifestTransaction::open(lineage, &key, &manifest_root, 2)
            .await
            .unwrap();
        let temporary = manifest_root.join(".tmp-symlink");
        std::os::unix::fs::symlink(directory.path(), &temporary).unwrap();

        let Err(error) = transaction.select_latest(&identity()).await else {
            panic!("temporary symlink unexpectedly passed manifest scan");
        };

        assert!(matches!(error, CalcFlowError::Format { .. }));
        assert!(
            std::fs::symlink_metadata(temporary)
                .unwrap()
                .file_type()
                .is_symlink()
        );
    }

    #[tokio::test]
    async fn retention_removes_abandoned_regular_manifest_temporary_files() {
        let directory = TempDir::new().unwrap();
        let backend = LocalStateBackend::new(directory.path().join("state"))
            .await
            .unwrap();
        let key = StateLineageKey::new("orders", PIPELINE_FINGERPRINT).unwrap();
        let lineage: Arc<dyn StateLineageBackend> =
            Arc::from(backend.open_lineage(&key).await.unwrap());
        let manifest_root = directory.path().join("manifests");
        let transaction = ManifestTransaction::open(lineage, &key, &manifest_root, 2)
            .await
            .unwrap();
        transaction
            .publish(PreparedEpochManifest {
                manifest: manifest(Epoch::INITIAL),
                staged_segments: BTreeMap::new(),
            })
            .await
            .unwrap();
        let temporary = manifest_root.join(".tmp-abandoned");
        tokio::fs::write(&temporary, b"partial").await.unwrap();

        let report = transaction.retain(&identity(), None).await.unwrap();

        assert_eq!(report.retained_manifests, 1);
        assert!(!temporary.exists());
    }

    #[tokio::test]
    async fn sparse_candidates_select_latest_complete_epoch_across_missing_higher_gap() {
        let directory = TempDir::new().unwrap();
        let backend = LocalStateBackend::new(directory.path().join("state"))
            .await
            .unwrap();
        let key = StateLineageKey::new("orders", PIPELINE_FINGERPRINT).unwrap();
        let lineage: Arc<dyn StateLineageBackend> =
            Arc::from(backend.open_lineage(&key).await.unwrap());
        let manifest_root = directory.path().join("manifests");
        let transaction = ManifestTransaction::open(lineage, &key, &manifest_root, 2)
            .await
            .unwrap();
        let third = Epoch::new(3).unwrap();
        let fourth = Epoch::new(4).unwrap();
        for epoch in [Epoch::INITIAL, third] {
            transaction
                .publish(PreparedEpochManifest {
                    manifest: manifest(epoch),
                    staged_segments: BTreeMap::new(),
                })
                .await
                .unwrap();
        }
        tokio::fs::write(
            manifest_root.join(".tmp-manifest-00000000000000000004.json"),
            b"partially published higher epoch",
        )
        .await
        .unwrap();

        let selected = transaction
            .select_latest(&identity())
            .await
            .unwrap()
            .unwrap();

        assert_eq!(selected.manifest.epoch(), third);
        assert_eq!(selected.next_epoch, fourth);
        assert!(!transaction.manifest_path(Epoch::new(2).unwrap()).exists());
        assert!(!transaction.manifest_path(fourth).exists());
    }

    #[tokio::test]
    async fn latest_max_epoch_rejects_next_epoch_overflow() {
        let directory = TempDir::new().unwrap();
        let backend = LocalStateBackend::new(directory.path().join("state"))
            .await
            .unwrap();
        let key = StateLineageKey::new("orders", PIPELINE_FINGERPRINT).unwrap();
        let lineage: Arc<dyn StateLineageBackend> =
            Arc::from(backend.open_lineage(&key).await.unwrap());
        let transaction =
            ManifestTransaction::open(lineage, &key, directory.path().join("manifests"), 2)
                .await
                .unwrap();
        transaction
            .publish(PreparedEpochManifest {
                manifest: manifest(Epoch::new(u64::MAX).unwrap()),
                staged_segments: BTreeMap::new(),
            })
            .await
            .unwrap();

        let Err(error) = transaction.select_latest(&identity()).await else {
            panic!("maximum epoch unexpectedly allocated a successor");
        };

        assert!(matches!(
            error,
            CalcFlowError::Internal { ref message } if message == "epoch counter exhausted"
        ));
    }

    #[tokio::test]
    async fn operator_snapshot_stages_validates_and_publishes_canonical_handles() {
        let directory = TempDir::new().unwrap();
        let backend = LocalStateBackend::new(directory.path().join("state"))
            .await
            .unwrap();
        let key = StateLineageKey::new("orders", PIPELINE_FINGERPRINT).unwrap();
        let lineage: Arc<dyn StateLineageBackend> =
            Arc::from(backend.open_lineage(&key).await.unwrap());
        let transaction = ManifestTransaction::open(
            Arc::clone(&lineage),
            &key,
            directory.path().join("manifests"),
            2,
        )
        .await
        .unwrap();
        let bytes = b"operator-delta".to_vec();
        let snapshot = OperatorStateSnapshot {
            inline_metadata: BTreeMap::from([("layout".into(), serde_json::json!(1))]),
            segments: BTreeMap::from([("delta-0001".into(), bytes.clone())]),
        };

        let staged = transaction
            .stage_operator_state("window", Epoch::INITIAL, snapshot)
            .await
            .unwrap();

        assert_eq!(staged.inline_metadata["layout"], serde_json::json!(1));
        assert_eq!(staged.segments.len(), 1);
        staged.segments[0]
            .validate_for("window", Epoch::INITIAL)
            .unwrap();
        assert_eq!(
            lineage.load_segment(&staged.segments[0]).await.unwrap(),
            bytes
        );
    }

    #[tokio::test]
    async fn state_stage_fault_occurs_after_segment_publication() {
        let directory = TempDir::new().unwrap();
        let backend = LocalStateBackend::new(directory.path().join("state"))
            .await
            .unwrap();
        let key = StateLineageKey::new("orders", PIPELINE_FINGERPRINT).unwrap();
        let lineage: Arc<dyn StateLineageBackend> =
            Arc::from(backend.open_lineage(&key).await.unwrap());
        let triggers = Arc::new(AtomicUsize::new(0));
        let transaction = ManifestTransaction::open(
            Arc::clone(&lineage),
            &key,
            directory.path().join("manifests"),
            2,
        )
        .await
        .unwrap()
        .with_fault_hook(fault_hook(
            ManifestTransactionFaultPoint::StateStage,
            Arc::clone(&triggers),
        ));

        let Err(error) = transaction
            .stage_operator_state(
                "window",
                Epoch::INITIAL,
                OperatorStateSnapshot {
                    inline_metadata: BTreeMap::new(),
                    segments: BTreeMap::from([("delta".into(), b"state".to_vec())]),
                },
            )
            .await
        else {
            panic!("state-stage fault unexpectedly returned staged state");
        };

        assert!(error.to_string().contains("StateStage"));
        assert_eq!(triggers.load(Ordering::SeqCst), 1);
        assert_eq!(lineage.collect_orphans(&[]).await.unwrap(), 1);
    }

    #[tokio::test]
    async fn state_stage_fault_skips_empty_snapshots_and_hits_first_dirty_segment() {
        let directory = TempDir::new().unwrap();
        let backend = LocalStateBackend::new(directory.path().join("state"))
            .await
            .unwrap();
        let key = StateLineageKey::new("orders", PIPELINE_FINGERPRINT).unwrap();
        let lineage: Arc<dyn StateLineageBackend> =
            Arc::from(backend.open_lineage(&key).await.unwrap());
        let triggers = Arc::new(AtomicUsize::new(0));
        let transaction = ManifestTransaction::open(
            Arc::clone(&lineage),
            &key,
            directory.path().join("manifests"),
            2,
        )
        .await
        .unwrap()
        .with_fault_hook(fault_hook(
            ManifestTransactionFaultPoint::StateStage,
            Arc::clone(&triggers),
        ));

        let empty = transaction
            .stage_operator_state(
                "merge",
                Epoch::INITIAL,
                OperatorStateSnapshot {
                    inline_metadata: BTreeMap::new(),
                    segments: BTreeMap::new(),
                },
            )
            .await
            .unwrap();

        assert!(empty.segments.is_empty());
        assert_eq!(triggers.load(Ordering::SeqCst), 0);

        let Err(error) = transaction
            .stage_operator_state(
                "window",
                Epoch::INITIAL,
                OperatorStateSnapshot {
                    inline_metadata: BTreeMap::new(),
                    segments: BTreeMap::from([("delta".into(), b"state".to_vec())]),
                },
            )
            .await
        else {
            panic!("dirty state-stage fault unexpectedly returned staged state");
        };

        assert!(error.to_string().contains("StateStage"));
        assert_eq!(triggers.load(Ordering::SeqCst), 1);
        assert_eq!(lineage.collect_orphans(&[]).await.unwrap(), 1);
    }

    #[tokio::test]
    async fn manifest_write_fault_precedes_the_atomic_rename() {
        let directory = TempDir::new().unwrap();
        let backend = LocalStateBackend::new(directory.path().join("state"))
            .await
            .unwrap();
        let key = StateLineageKey::new("orders", PIPELINE_FINGERPRINT).unwrap();
        let lineage: Arc<dyn StateLineageBackend> =
            Arc::from(backend.open_lineage(&key).await.unwrap());
        let triggers = Arc::new(AtomicUsize::new(0));
        let transaction =
            ManifestTransaction::open(lineage, &key, directory.path().join("manifests"), 2)
                .await
                .unwrap()
                .with_fault_hook(fault_hook(
                    ManifestTransactionFaultPoint::ManifestWrite,
                    Arc::clone(&triggers),
                ));

        let error = transaction
            .publish(PreparedEpochManifest {
                manifest: manifest(Epoch::INITIAL),
                staged_segments: BTreeMap::new(),
            })
            .await
            .unwrap_err();

        assert!(error.to_string().contains("ManifestWrite"));
        assert_eq!(triggers.load(Ordering::SeqCst), 1);
        assert!(!transaction.manifest_path(Epoch::INITIAL).exists());
        assert!(
            transaction
                .select_latest(&identity())
                .await
                .unwrap()
                .is_none()
        );
    }

    #[tokio::test]
    async fn manifest_rename_fault_reports_an_installed_selectable_epoch() {
        let directory = TempDir::new().unwrap();
        let backend = LocalStateBackend::new(directory.path().join("state"))
            .await
            .unwrap();
        let key = StateLineageKey::new("orders", PIPELINE_FINGERPRINT).unwrap();
        let lineage: Arc<dyn StateLineageBackend> =
            Arc::from(backend.open_lineage(&key).await.unwrap());
        let triggers = Arc::new(AtomicUsize::new(0));
        let transaction =
            ManifestTransaction::open(lineage, &key, directory.path().join("manifests"), 2)
                .await
                .unwrap()
                .with_fault_hook(fault_hook(
                    ManifestTransactionFaultPoint::ManifestRename,
                    Arc::clone(&triggers),
                ));

        let publication = transaction
            .publish(PreparedEpochManifest {
                manifest: manifest(Epoch::INITIAL),
                staged_segments: BTreeMap::new(),
            })
            .await
            .unwrap();

        assert!(matches!(
            publication,
            ManifestPublication::Installed {
                parent_synced: false,
                ref error,
            } if error.to_string().contains("ManifestRename")
        ));
        assert_eq!(triggers.load(Ordering::SeqCst), 1);
        assert_eq!(
            transaction
                .select_latest(&identity())
                .await
                .unwrap()
                .unwrap()
                .manifest
                .epoch(),
            Epoch::INITIAL
        );
    }

    #[tokio::test]
    async fn parent_sync_fault_reports_an_installed_durable_epoch() {
        let directory = TempDir::new().unwrap();
        let backend = LocalStateBackend::new(directory.path().join("state"))
            .await
            .unwrap();
        let key = StateLineageKey::new("orders", PIPELINE_FINGERPRINT).unwrap();
        let lineage: Arc<dyn StateLineageBackend> =
            Arc::from(backend.open_lineage(&key).await.unwrap());
        let triggers = Arc::new(AtomicUsize::new(0));
        let transaction =
            ManifestTransaction::open(lineage, &key, directory.path().join("manifests"), 2)
                .await
                .unwrap()
                .with_fault_hook(fault_hook(
                    ManifestTransactionFaultPoint::ManifestParentSync,
                    Arc::clone(&triggers),
                ));

        let publication = transaction
            .publish(PreparedEpochManifest {
                manifest: manifest(Epoch::INITIAL),
                staged_segments: BTreeMap::new(),
            })
            .await
            .unwrap();

        assert!(matches!(
            publication,
            ManifestPublication::Installed {
                parent_synced: true,
                ref error,
            } if error.to_string().contains("ManifestParentSync")
        ));
        assert_eq!(triggers.load(Ordering::SeqCst), 1);
        assert_eq!(
            transaction
                .select_latest(&identity())
                .await
                .unwrap()
                .unwrap()
                .manifest
                .epoch(),
            Epoch::INITIAL
        );
    }

    #[tokio::test]
    async fn cancellation_waits_for_manifest_worker_settlement_before_returning() {
        let directory = TempDir::new().unwrap();
        let backend = LocalStateBackend::new(directory.path().join("state"))
            .await
            .unwrap();
        let key = StateLineageKey::new("orders", PIPELINE_FINGERPRINT).unwrap();
        let lineage: Arc<dyn StateLineageBackend> =
            Arc::from(backend.open_lineage(&key).await.unwrap());
        let (entered_tx, entered_rx) = std::sync::mpsc::sync_channel(1);
        let (release_tx, release_rx) = std::sync::mpsc::sync_channel(1);
        let release_rx = Arc::new(std::sync::Mutex::new(release_rx));
        let cancellation = crate::CancellationToken::new();
        let hook_cancellation = cancellation.clone();
        let transaction = Arc::new(
            ManifestTransaction::open(lineage, &key, directory.path().join("manifests"), 2)
                .await
                .unwrap()
                .with_fault_hook(Arc::new(move |point| {
                    if point == ManifestTransactionFaultPoint::ManifestWrite {
                        entered_tx.send(()).unwrap();
                        release_rx.lock().unwrap().recv().unwrap();
                        if hook_cancellation.is_cancelled() {
                            return Err(CalcFlowError::Cancelled {
                                run_id: "checkpoint:test-worker".into(),
                            });
                        }
                    }
                    Ok(())
                })),
        );
        let publish = {
            let transaction = Arc::clone(&transaction);
            let cancellation = cancellation.clone();
            tokio::spawn(async move {
                transaction
                    .publish_cancellable(
                        PreparedEpochManifest {
                            manifest: manifest(Epoch::INITIAL),
                            staged_segments: BTreeMap::new(),
                        },
                        &cancellation,
                    )
                    .await
            })
        };
        tokio::task::spawn_blocking(move || entered_rx.recv().unwrap())
            .await
            .unwrap();

        cancellation.cancel();
        tokio::task::yield_now().await;
        assert!(!publish.is_finished());
        release_tx.send(()).unwrap();
        let result = publish.await.unwrap();

        assert!(matches!(result, Err(CalcFlowError::Cancelled { .. })));
        assert!(!transaction.manifest_path(Epoch::INITIAL).exists());
        assert!(
            transaction
                .select_latest(&identity())
                .await
                .unwrap()
                .is_none()
        );
    }

    async fn run_cancellable_manifest_operation(
        transaction: Arc<ManifestTransaction>,
        point: ManifestOperationPoint,
        cancellation: crate::CancellationToken,
    ) -> crate::Result<()> {
        match point {
            ManifestOperationPoint::Open => unreachable!("open is tested separately"),
            ManifestOperationPoint::Stage => transaction
                .stage_operator_state_cancellable(
                    "window",
                    Epoch::INITIAL,
                    OperatorStateSnapshot {
                        inline_metadata: BTreeMap::new(),
                        segments: BTreeMap::new(),
                    },
                    &cancellation,
                )
                .await
                .map(|_| ()),
            ManifestOperationPoint::Load => transaction
                .load_operator_state_cancellable(
                    "window",
                    Epoch::INITIAL,
                    &OperatorManifestEntry {
                        progress: BTreeMap::new(),
                        inline_metadata: BTreeMap::new(),
                        segments: Vec::new(),
                    },
                    &cancellation,
                )
                .await
                .map(|_| ()),
            ManifestOperationPoint::Select => transaction
                .select_latest_cancellable(&identity(), &cancellation)
                .await
                .map(|_| ()),
            ManifestOperationPoint::Publish => transaction
                .publish_cancellable(
                    PreparedEpochManifest {
                        manifest: manifest(Epoch::INITIAL),
                        staged_segments: BTreeMap::new(),
                    },
                    &cancellation,
                )
                .await
                .map(|_| ()),
            ManifestOperationPoint::Retain => transaction
                .retain_cancellable(&identity(), None, &cancellation)
                .await
                .map(|_| ()),
        }
    }

    #[tokio::test]
    async fn cancellation_settles_every_owned_transaction_operation_before_returning() {
        for point in [
            ManifestOperationPoint::Stage,
            ManifestOperationPoint::Load,
            ManifestOperationPoint::Select,
            ManifestOperationPoint::Publish,
            ManifestOperationPoint::Retain,
        ] {
            let directory = TempDir::new().unwrap();
            let backend = LocalStateBackend::new(directory.path().join("state"))
                .await
                .unwrap();
            let key = StateLineageKey::new("orders", PIPELINE_FINGERPRINT).unwrap();
            let lineage: Arc<dyn StateLineageBackend> =
                Arc::from(backend.open_lineage(&key).await.unwrap());
            let (entered_tx, entered_rx) = std::sync::mpsc::sync_channel(1);
            let (release_tx, release_rx) = std::sync::mpsc::sync_channel(1);
            let release_rx = Arc::new(std::sync::Mutex::new(release_rx));
            let triggers = Arc::new(AtomicUsize::new(0));
            let hook_triggers = Arc::clone(&triggers);
            let transaction = Arc::new(
                ManifestTransaction::open(lineage, &key, directory.path().join("manifests"), 2)
                    .await
                    .unwrap()
                    .with_operation_hook(Arc::new(move |observed| {
                        if observed == point && hook_triggers.fetch_add(1, Ordering::SeqCst) == 0 {
                            entered_tx.send(()).unwrap();
                            release_rx.lock().unwrap().recv().unwrap();
                        }
                        Ok(())
                    })),
            );
            let cancellation = crate::CancellationToken::new();
            let operation = tokio::spawn(run_cancellable_manifest_operation(
                Arc::clone(&transaction),
                point,
                cancellation.clone(),
            ));
            tokio::task::spawn_blocking(move || entered_rx.recv().unwrap())
                .await
                .unwrap();

            cancellation.cancel();
            tokio::task::yield_now().await;
            assert!(
                !operation.is_finished(),
                "{point:?} returned before settlement"
            );
            release_tx.send(()).unwrap();
            let result = operation.await.unwrap();

            assert!(
                matches!(result, Err(CalcFlowError::Cancelled { .. })),
                "{point:?} returned {result:?}"
            );
            assert_eq!(triggers.load(Ordering::SeqCst), 1);
            assert!(
                transaction
                    .select_latest(&identity())
                    .await
                    .unwrap()
                    .is_none()
            );
        }
    }

    #[tokio::test]
    async fn cancellation_waits_for_manifest_open_worker_before_returning() {
        let directory = TempDir::new().unwrap();
        let backend = LocalStateBackend::new(directory.path().join("state"))
            .await
            .unwrap();
        let key = StateLineageKey::new("orders", PIPELINE_FINGERPRINT).unwrap();
        let lineage: Arc<dyn StateLineageBackend> =
            Arc::from(backend.open_lineage(&key).await.unwrap());
        let (entered_tx, entered_rx) = std::sync::mpsc::sync_channel(1);
        let (release_tx, release_rx) = std::sync::mpsc::sync_channel(1);
        let release_rx = Arc::new(std::sync::Mutex::new(release_rx));
        let cancellation = crate::CancellationToken::new();
        let manifest_root = directory.path().join("manifests");
        let open = {
            let cancellation = cancellation.clone();
            let key = key.clone();
            let manifest_root = manifest_root.clone();
            tokio::spawn(async move {
                ManifestTransaction::open_cancellable_with_operation_hook(
                    lineage,
                    &key,
                    manifest_root,
                    2,
                    &cancellation,
                    Arc::new(move |point| {
                        assert_eq!(point, ManifestOperationPoint::Open);
                        entered_tx.send(()).unwrap();
                        release_rx.lock().unwrap().recv().unwrap();
                        Ok(())
                    }),
                )
                .await
            })
        };
        tokio::task::spawn_blocking(move || entered_rx.recv().unwrap())
            .await
            .unwrap();

        cancellation.cancel();
        tokio::task::yield_now().await;
        assert!(!open.is_finished());
        release_tx.send(()).unwrap();
        let result = open.await.unwrap();

        assert!(matches!(result, Err(CalcFlowError::Cancelled { .. })));
        assert_eq!(std::fs::read_dir(&manifest_root).unwrap().count(), 0);
        let reopened = ManifestTransaction::open(
            Arc::from(backend.open_lineage(&key).await.unwrap()),
            &key,
            &manifest_root,
            2,
        )
        .await
        .unwrap();
        assert!(reopened.select_latest(&identity()).await.unwrap().is_none());
    }

    #[tokio::test]
    async fn operator_restore_loads_validated_segments_by_stable_segment_id() {
        let directory = TempDir::new().unwrap();
        let backend = LocalStateBackend::new(directory.path().join("state"))
            .await
            .unwrap();
        let key = StateLineageKey::new("orders", PIPELINE_FINGERPRINT).unwrap();
        let lineage: Arc<dyn StateLineageBackend> =
            Arc::from(backend.open_lineage(&key).await.unwrap());
        let transaction =
            ManifestTransaction::open(lineage, &key, directory.path().join("manifests"), 2)
                .await
                .unwrap();
        let epoch = Epoch::INITIAL;
        let staged = transaction
            .stage_operator_state(
                "window",
                epoch,
                OperatorStateSnapshot {
                    inline_metadata: BTreeMap::from([("layout".into(), serde_json::json!(1))]),
                    segments: BTreeMap::from([
                        ("base".into(), b"base-state".to_vec()),
                        ("delta".into(), b"delta-state".to_vec()),
                    ]),
                },
            )
            .await
            .unwrap();
        let entry = OperatorManifestEntry {
            progress: BTreeMap::new(),
            inline_metadata: staged.inline_metadata,
            segments: staged.segments,
        };

        let restored = transaction
            .load_operator_state("window", epoch, &entry)
            .await
            .unwrap();

        assert_eq!(restored.inline_metadata["layout"], serde_json::json!(1));
        assert_eq!(
            restored.segments,
            BTreeMap::from([
                ("base".into(), b"base-state".to_vec()),
                ("delta".into(), b"delta-state".to_vec()),
            ])
        );
    }

    #[tokio::test]
    async fn retention_removes_old_manifests_after_preserving_reachable_epochs() {
        let directory = TempDir::new().unwrap();
        let backend = LocalStateBackend::new(directory.path().join("state"))
            .await
            .unwrap();
        let key = StateLineageKey::new("orders", PIPELINE_FINGERPRINT).unwrap();
        let lineage: Arc<dyn StateLineageBackend> =
            Arc::from(backend.open_lineage(&key).await.unwrap());
        let transaction =
            ManifestTransaction::open(lineage, &key, directory.path().join("manifests"), 2)
                .await
                .unwrap();
        let second = Epoch::INITIAL.next().unwrap();
        let third = second.next().unwrap();
        for epoch in [Epoch::INITIAL, second, third] {
            transaction
                .publish(PreparedEpochManifest {
                    manifest: manifest(epoch),
                    staged_segments: BTreeMap::new(),
                })
                .await
                .unwrap();
        }

        let report = transaction.retain(&identity(), None).await.unwrap();

        assert_eq!(report.retained_manifests, 2);
        assert_eq!(report.removed_manifests, 1);
        assert_eq!(report.removed_orphan_segments, 0);
        assert!(!transaction.manifest_path(Epoch::INITIAL).exists());
        assert!(transaction.manifest_path(second).exists());
        assert!(transaction.manifest_path(third).exists());
    }

    #[tokio::test]
    async fn retention_preserves_segments_reachable_only_from_in_flight_manifest() {
        let directory = TempDir::new().unwrap();
        let backend = LocalStateBackend::new(directory.path().join("state"))
            .await
            .unwrap();
        let key = StateLineageKey::new("orders", PIPELINE_FINGERPRINT).unwrap();
        let lineage: Arc<dyn StateLineageBackend> =
            Arc::from(backend.open_lineage(&key).await.unwrap());
        let transaction = ManifestTransaction::open(
            Arc::clone(&lineage),
            &key,
            directory.path().join("manifests"),
            1,
        )
        .await
        .unwrap();
        let first = transaction
            .stage_operator_state(
                "window",
                Epoch::INITIAL,
                OperatorStateSnapshot {
                    inline_metadata: BTreeMap::new(),
                    segments: BTreeMap::from([("base".into(), b"first".to_vec())]),
                },
            )
            .await
            .unwrap()
            .segments
            .remove(0);
        transaction
            .publish(PreparedEpochManifest {
                manifest: manifest_with_operator_segments(Epoch::INITIAL, vec![first.clone()]),
                staged_segments: BTreeMap::new(),
            })
            .await
            .unwrap();
        let second_epoch = Epoch::INITIAL.next().unwrap();
        let second = transaction
            .stage_operator_state(
                "window",
                second_epoch,
                OperatorStateSnapshot {
                    inline_metadata: BTreeMap::new(),
                    segments: BTreeMap::from([("delta".into(), b"second".to_vec())]),
                },
            )
            .await
            .unwrap()
            .segments
            .remove(0);
        transaction
            .publish(PreparedEpochManifest {
                manifest: manifest_with_operator_segments(second_epoch, vec![second.clone()]),
                staged_segments: BTreeMap::new(),
            })
            .await
            .unwrap();
        let in_flight =
            manifest_with_operator_segments(second_epoch.next().unwrap(), vec![first.clone()]);

        let report = transaction
            .retain(&identity_with_window(), Some(&in_flight))
            .await
            .unwrap();

        assert_eq!(report.removed_manifests, 1);
        assert_eq!(report.removed_orphan_segments, 0);
        assert_eq!(lineage.load_segment(&first).await.unwrap(), b"first");
        assert_eq!(lineage.load_segment(&second).await.unwrap(), b"second");
    }

    #[tokio::test]
    async fn compaction_fault_occurs_after_manifest_retention_before_orphan_collection() {
        let directory = TempDir::new().unwrap();
        let backend = LocalStateBackend::new(directory.path().join("state"))
            .await
            .unwrap();
        let key = StateLineageKey::new("orders", PIPELINE_FINGERPRINT).unwrap();
        let lineage: Arc<dyn StateLineageBackend> =
            Arc::from(backend.open_lineage(&key).await.unwrap());
        let triggers = Arc::new(AtomicUsize::new(0));
        let transaction = ManifestTransaction::open(
            Arc::clone(&lineage),
            &key,
            directory.path().join("manifests"),
            1,
        )
        .await
        .unwrap()
        .with_fault_hook(fault_hook(
            ManifestTransactionFaultPoint::Compaction,
            Arc::clone(&triggers),
        ));
        let first = transaction
            .stage_operator_state(
                "window",
                Epoch::INITIAL,
                OperatorStateSnapshot {
                    inline_metadata: BTreeMap::new(),
                    segments: BTreeMap::from([("base".into(), b"first".to_vec())]),
                },
            )
            .await
            .unwrap()
            .segments
            .remove(0);
        transaction
            .publish(PreparedEpochManifest {
                manifest: manifest_with_operator_segments(Epoch::INITIAL, vec![first.clone()]),
                staged_segments: BTreeMap::new(),
            })
            .await
            .unwrap();
        let second = Epoch::INITIAL.next().unwrap();
        transaction
            .publish(PreparedEpochManifest {
                manifest: manifest_with_operator_segments(second, Vec::new()),
                staged_segments: BTreeMap::new(),
            })
            .await
            .unwrap();

        let Err(error) = transaction.retain(&identity_with_window(), None).await else {
            panic!("compaction fault unexpectedly returned a retention report");
        };

        assert!(error.to_string().contains("Compaction"));
        assert_eq!(triggers.load(Ordering::SeqCst), 1);
        assert!(!transaction.manifest_path(Epoch::INITIAL).exists());
        assert!(transaction.manifest_path(second).exists());
        assert_eq!(lineage.load_segment(&first).await.unwrap(), b"first");
    }
}
