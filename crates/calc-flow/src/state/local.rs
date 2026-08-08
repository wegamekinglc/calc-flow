use std::{
    collections::BTreeSet,
    fs::{File, OpenOptions},
    io::{Read, Write},
    path::{Path, PathBuf},
    sync::Arc,
};

#[cfg(test)]
use std::collections::BTreeMap;

use async_trait::async_trait;
use fs2::FileExt;
use parking_lot::Mutex as SyncMutex;
use sha2::{Digest, Sha256};
use tokio::sync::Mutex;

use super::{StateBackend, StateHandle, StateLineageBackend, StateLineageKey};
use crate::{CalcFlowError, Epoch, Result};

#[cfg(test)]
use super::CheckpointManifest;

/// Local immutable-segment state backend rooted in one managed directory.
#[derive(Clone, Debug)]
pub struct LocalStateBackend {
    root: Arc<ManagedStateRoot>,
}

#[derive(Debug)]
struct ManagedStateRoot {
    path: PathBuf,
    identity_hash: String,
    process_leases: SyncMutex<BTreeSet<String>>,
}

impl LocalStateBackend {
    /// Creates and canonicalizes one managed local-state root.
    ///
    /// # Errors
    ///
    /// Returns an I/O or format error when the root or one of its fixed
    /// managed subtrees cannot be safely created.
    pub async fn new(root: impl AsRef<Path>) -> Result<Self> {
        let requested = root.as_ref().to_owned();
        tokio::fs::create_dir_all(&requested)
            .await
            .map_err(|source| io_error(&requested, source))?;
        let path = tokio::fs::canonicalize(&requested)
            .await
            .map_err(|source| io_error(&requested, source))?;
        let prepared_path = path.clone();
        worker(move || {
            validate_directory(&prepared_path)?;
            for component in ["locks", "staging", "committed"] {
                ensure_child_directory(&prepared_path, component)?;
            }
            Ok(())
        })
        .await?;
        let identity_hash = digest(path.to_string_lossy().as_bytes());
        Ok(Self {
            root: Arc::new(ManagedStateRoot {
                path,
                identity_hash,
                process_leases: SyncMutex::new(BTreeSet::new()),
            }),
        })
    }
}

#[async_trait]
impl StateBackend for LocalStateBackend {
    async fn open_lineage(&self, key: &StateLineageKey) -> Result<Box<dyn StateLineageBackend>> {
        let lineage_hash = lineage_hash(key);
        {
            let mut leases = self.root.process_leases.lock();
            if !leases.insert(lineage_hash.clone()) {
                return Err(lease_conflict(&lineage_hash));
            }
        }

        let root = Arc::clone(&self.root);
        let lock_hash = lineage_hash.clone();
        let lock_result = worker(move || open_lock_file(&root.path, &lock_hash)).await;
        let lock_file = match lock_result {
            Ok(lock_file) => lock_file,
            Err(error) => {
                self.root.process_leases.lock().remove(&lineage_hash);
                return Err(error);
            }
        };
        Ok(Box::new(LocalStateLineageBackend {
            root: Arc::clone(&self.root),
            lineage_hash,
            lock_file,
            publication: Mutex::new(()),
            validated: SyncMutex::new(BTreeSet::new()),
        }))
    }
}

struct LocalStateLineageBackend {
    root: Arc<ManagedStateRoot>,
    lineage_hash: String,
    lock_file: File,
    publication: Mutex<()>,
    validated: SyncMutex<BTreeSet<StateHandle>>,
}

impl Drop for LocalStateLineageBackend {
    fn drop(&mut self) {
        let _ = FileExt::unlock(&self.lock_file);
        self.root.process_leases.lock().remove(&self.lineage_hash);
    }
}

#[async_trait]
impl StateLineageBackend for LocalStateLineageBackend {
    fn identity_hash(&self) -> &str {
        &self.root.identity_hash
    }

    async fn stage_segment(&self, handle: &StateHandle, bytes: &[u8]) -> Result<()> {
        let paths = self.managed_paths(handle)?;
        let handle = handle.clone();
        let bytes = bytes.to_vec();
        let _guard = self.publication.lock().await;
        let staged_handle = handle.clone();
        worker(move || stage_file(&paths, &staged_handle, &bytes)).await?;
        self.validated.lock().remove(&handle);
        Ok(())
    }

    async fn validate_segment(&self, handle: &StateHandle) -> Result<()> {
        let paths = self.managed_paths(handle)?;
        let handle = handle.clone();
        let _guard = self.publication.lock().await;
        let validated_handle = handle.clone();
        worker(move || read_validated_file(&paths.staging, &validated_handle).map(|_| ())).await?;
        self.validated.lock().insert(handle);
        Ok(())
    }

    async fn publish_segment(&self, handle: &StateHandle) -> Result<()> {
        let paths = self.managed_paths(handle)?;
        let handle = handle.clone();
        let _guard = self.publication.lock().await;

        let committed = paths.committed.clone();
        let committed_handle = handle.clone();
        if worker(move || committed_file_matches(&committed, &committed_handle)).await? {
            return Ok(());
        }
        if !self.validated.lock().contains(&handle) {
            return Err(CalcFlowError::Conflict {
                resource: "validated state segment".into(),
                key: handle.segment_id.clone(),
            });
        }

        let published_handle = handle.clone();
        worker(move || publish_file(&paths, &published_handle)).await?;
        self.validated.lock().remove(&handle);
        Ok(())
    }

    async fn load_segment(&self, handle: &StateHandle) -> Result<Vec<u8>> {
        let paths = self.managed_paths(handle)?;
        let handle = handle.clone();
        let _guard = self.publication.lock().await;
        worker(move || read_validated_file(&paths.committed, &handle)).await
    }

    async fn collect_orphans(&self, retained: &[StateHandle]) -> Result<usize> {
        let mut retained_paths = BTreeSet::new();
        let mut latest_epoch = None;
        for handle in retained {
            self.managed_paths(handle)?;
            retained_paths.insert(handle.relative_path.clone());
            latest_epoch =
                Some(latest_epoch.map_or(handle.epoch, |epoch: Epoch| epoch.max(handle.epoch)));
        }
        let root = self.root.path.clone();
        let lineage_hash = self.lineage_hash.clone();
        let _guard = self.publication.lock().await;
        worker(move || collect_unreachable(&root, &lineage_hash, &retained_paths, latest_epoch))
            .await
    }
}

impl LocalStateLineageBackend {
    fn managed_paths(&self, handle: &StateHandle) -> Result<ManagedSegmentPaths> {
        handle.validate_for(&handle.operator_id, handle.epoch)?;
        let operator_hash = digest(handle.operator_id.as_bytes());
        let segment_hash = digest(handle.segment_id.as_bytes());
        let stem = format!("{}-{segment_hash}", handle.epoch.as_u64());
        let arrow = format!(
            "committed/{}/{operator_hash}/{stem}.arrow",
            self.lineage_hash
        );
        let opaque = format!(
            "committed/{}/{operator_hash}/{stem}.segment",
            self.lineage_hash
        );
        if handle.relative_path != arrow && handle.relative_path != opaque {
            return Err(CalcFlowError::InvalidArgument {
                field: "state_handle.relative_path".into(),
                message: "does not match the backend-managed hashed segment path".into(),
            });
        }
        let staging_parent = self
            .root
            .path
            .join("staging")
            .join(&self.lineage_hash)
            .join(handle.epoch.as_u64().to_string())
            .join(&operator_hash);
        let committed_parent = self
            .root
            .path
            .join("committed")
            .join(&self.lineage_hash)
            .join(operator_hash);
        Ok(ManagedSegmentPaths {
            staging: staging_parent.join(format!("{segment_hash}.tmp")),
            committed: self.root.path.join(&handle.relative_path),
            staging_parent,
            committed_parent,
        })
    }
}

#[cfg(test)]
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) enum CommitFaultPoint {
    AfterLease,
    AfterFirstSegmentStage,
    AfterAllSegmentsDurable,
    AfterSegmentValidation,
    AfterFirstSegmentPublication,
    AfterCommittedSynchronization,
    AfterManifestValidation,
    AfterManifestDurableWrite,
    AfterManifestRename,
    AfterManifestPublication,
}

#[cfg(test)]
pub(crate) async fn commit_manifest_for_test(
    state: &dyn StateLineageBackend,
    manifest_root: &Path,
    manifest: &CheckpointManifest,
    staged: &BTreeMap<StateHandle, Vec<u8>>,
    fault: Option<CommitFaultPoint>,
) -> Result<()> {
    inject_fault(fault, CommitFaultPoint::AfterLease)?;

    for (index, (handle, bytes)) in staged.iter().enumerate() {
        state.stage_segment(handle, bytes).await?;
        if index == 0 {
            inject_fault(fault, CommitFaultPoint::AfterFirstSegmentStage)?;
        }
    }
    inject_fault(fault, CommitFaultPoint::AfterAllSegmentsDurable)?;

    for handle in staged.keys() {
        state.validate_segment(handle).await?;
    }
    inject_fault(fault, CommitFaultPoint::AfterSegmentValidation)?;

    for (index, handle) in staged.keys().enumerate() {
        state.publish_segment(handle).await?;
        if index == 0 {
            inject_fault(fault, CommitFaultPoint::AfterFirstSegmentPublication)?;
        }
    }
    inject_fault(fault, CommitFaultPoint::AfterCommittedSynchronization)?;

    for operator in manifest.operators().values() {
        for handle in &operator.segments {
            state.load_segment(handle).await?;
        }
    }
    let manifest = manifest.clone();
    let manifest_bytes = worker(move || manifest.canonical_bytes()).await?;
    inject_fault(fault, CommitFaultPoint::AfterManifestValidation)?;

    let manifest_root = prepare_manifest_root(manifest_root).await?;
    let temporary_root = manifest_root.clone();
    let temporary =
        worker(move || write_manifest_temporary(&temporary_root, &manifest_bytes)).await?;
    inject_fault(fault, CommitFaultPoint::AfterManifestDurableWrite)?;

    let destination = manifest_root.join("manifest.json");
    worker(move || rename_manifest_temporary(temporary, &destination)).await?;
    inject_fault(fault, CommitFaultPoint::AfterManifestRename)?;
    worker(move || sync_directory(&manifest_root)).await?;
    inject_fault(fault, CommitFaultPoint::AfterManifestPublication)?;
    Ok(())
}

#[cfg(test)]
fn inject_fault(found: Option<CommitFaultPoint>, expected: CommitFaultPoint) -> Result<()> {
    if found == Some(expected) {
        Err(CalcFlowError::Internal {
            message: format!("injected manifest commit fault at {expected:?}"),
        })
    } else {
        Ok(())
    }
}

#[cfg(test)]
async fn prepare_manifest_root(requested: &Path) -> Result<PathBuf> {
    tokio::fs::create_dir_all(requested)
        .await
        .map_err(|source| io_error(requested, source))?;
    let root = tokio::fs::canonicalize(requested)
        .await
        .map_err(|source| io_error(requested, source))?;
    let validated_root = root.clone();
    worker(move || validate_directory(&validated_root)).await?;
    Ok(root)
}

#[cfg(test)]
fn write_manifest_temporary(root: &Path, bytes: &[u8]) -> Result<tempfile::NamedTempFile> {
    validate_directory(root)?;
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
    Ok(temporary)
}

#[cfg(test)]
fn rename_manifest_temporary(temporary: tempfile::NamedTempFile, destination: &Path) -> Result<()> {
    temporary
        .persist(destination)
        .map_err(|error| io_error(destination, error.error))?;
    Ok(())
}

struct ManagedSegmentPaths {
    staging: PathBuf,
    committed: PathBuf,
    staging_parent: PathBuf,
    committed_parent: PathBuf,
}

fn open_lock_file(root: &Path, lineage_hash: &str) -> Result<File> {
    validate_directory(root)?;
    let lock_directory = ensure_child_directory(root, "locks")?;
    let path = lock_directory.join(format!("{lineage_hash}.lock"));
    if let Ok(metadata) = std::fs::symlink_metadata(&path) {
        if metadata.file_type().is_symlink() || !metadata.is_file() {
            return Err(format_error(format!(
                "state lock {} is not a regular file",
                path.display()
            )));
        }
    }
    let file = OpenOptions::new()
        .create(true)
        .read(true)
        .write(true)
        .truncate(false)
        .open(&path)
        .map_err(|source| io_error(&path, source))?;
    FileExt::try_lock_exclusive(&file).map_err(|error| {
        if error.kind() == std::io::ErrorKind::WouldBlock {
            lease_conflict(lineage_hash)
        } else {
            io_error(&path, error)
        }
    })?;
    Ok(file)
}

fn stage_file(paths: &ManagedSegmentPaths, handle: &StateHandle, bytes: &[u8]) -> Result<()> {
    validate_expected_bytes(handle, bytes)?;
    prepare_segment_directories(paths)?;
    match std::fs::symlink_metadata(&paths.staging) {
        Ok(_) => return read_validated_file(&paths.staging, handle).map(|_| ()),
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => {}
        Err(source) => return Err(io_error(&paths.staging, source)),
    }
    if paths.committed.exists() {
        return Err(CalcFlowError::Conflict {
            resource: "committed state segment".into(),
            key: handle.segment_id.clone(),
        });
    }

    let mut temporary = tempfile::NamedTempFile::new_in(&paths.staging_parent)
        .map_err(|source| io_error(&paths.staging_parent, source))?;
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
    temporary
        .persist_noclobber(&paths.staging)
        .map_err(|error| io_error(&paths.staging, error.error))?;
    sync_directory(&paths.staging_parent)
}

fn publish_file(paths: &ManagedSegmentPaths, handle: &StateHandle) -> Result<()> {
    prepare_segment_directories(paths)?;
    read_validated_file(&paths.staging, handle)?;
    std::fs::rename(&paths.staging, &paths.committed)
        .map_err(|source| io_error(&paths.committed, source))?;
    sync_directory(&paths.committed_parent)
}

fn prepare_segment_directories(paths: &ManagedSegmentPaths) -> Result<()> {
    let root = paths
        .staging_parent
        .ancestors()
        .nth(4)
        .ok_or_else(|| format_error("managed staging path has no root".into()))?;
    validate_directory(root)?;
    let staging = ensure_child_directory(root, "staging")?;
    let lineage = paths
        .staging_parent
        .ancestors()
        .nth(2)
        .and_then(Path::file_name)
        .and_then(|name| name.to_str())
        .ok_or_else(|| format_error("managed staging lineage is invalid".into()))?;
    let epoch = paths
        .staging_parent
        .parent()
        .and_then(Path::file_name)
        .and_then(|name| name.to_str())
        .ok_or_else(|| format_error("managed staging epoch is invalid".into()))?;
    let operator = paths
        .staging_parent
        .file_name()
        .and_then(|name| name.to_str())
        .ok_or_else(|| format_error("managed staging operator is invalid".into()))?;
    let staging_lineage = ensure_child_directory(&staging, lineage)?;
    let staging_epoch = ensure_child_directory(&staging_lineage, epoch)?;
    ensure_child_directory(&staging_epoch, operator)?;

    let committed = ensure_child_directory(root, "committed")?;
    let committed_lineage = ensure_child_directory(&committed, lineage)?;
    ensure_child_directory(&committed_lineage, operator)?;
    Ok(())
}

fn read_validated_file(path: &Path, handle: &StateHandle) -> Result<Vec<u8>> {
    let metadata = match std::fs::symlink_metadata(path) {
        Ok(metadata) => metadata,
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => {
            return Err(CalcFlowError::NotFound {
                resource: "state segment".into(),
                key: handle.segment_id.clone(),
            });
        }
        Err(source) => return Err(io_error(path, source)),
    };
    if metadata.file_type().is_symlink() || !metadata.is_file() {
        return Err(format_error(format!(
            "state segment {} is not a regular file",
            path.display()
        )));
    }
    if metadata.len() != handle.byte_len {
        return Err(segment_mismatch(handle, "byte length"));
    }
    let capacity = usize::try_from(metadata.len()).map_err(|_| CalcFlowError::Internal {
        message: "state segment length does not fit usize".into(),
    })?;
    let mut file = File::open(path).map_err(|source| io_error(path, source))?;
    let mut bytes = Vec::with_capacity(capacity);
    file.read_to_end(&mut bytes)
        .map_err(|source| io_error(path, source))?;
    validate_expected_bytes(handle, &bytes)?;
    Ok(bytes)
}

fn committed_file_matches(path: &Path, handle: &StateHandle) -> Result<bool> {
    match std::fs::symlink_metadata(path) {
        Ok(_) => read_validated_file(path, handle).map(|_| true),
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => Ok(false),
        Err(source) => Err(io_error(path, source)),
    }
}

fn validate_expected_bytes(handle: &StateHandle, bytes: &[u8]) -> Result<()> {
    if u64::try_from(bytes.len()).ok() != Some(handle.byte_len) {
        return Err(segment_mismatch(handle, "byte length"));
    }
    if digest(bytes) != handle.sha256 {
        return Err(segment_mismatch(handle, "SHA-256"));
    }
    Ok(())
}

fn collect_unreachable(
    root: &Path,
    lineage_hash: &str,
    retained_paths: &BTreeSet<String>,
    latest_epoch: Option<Epoch>,
) -> Result<usize> {
    validate_directory(root)?;
    let lineage_root = root.join("committed").join(lineage_hash);
    let lineage_metadata = match std::fs::symlink_metadata(&lineage_root) {
        Ok(metadata) => metadata,
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => return Ok(0),
        Err(source) => return Err(io_error(&lineage_root, source)),
    };
    if lineage_metadata.file_type().is_symlink() || !lineage_metadata.is_dir() {
        return Err(format_error(format!(
            "committed lineage {} is not a directory",
            lineage_root.display()
        )));
    }

    let mut deletions = Vec::new();
    let mut sync_directories = BTreeSet::new();
    for operator_entry in
        std::fs::read_dir(&lineage_root).map_err(|source| io_error(&lineage_root, source))?
    {
        let operator_entry = operator_entry.map_err(|source| io_error(&lineage_root, source))?;
        let operator_path = operator_entry.path();
        let operator_name = portable_hash_name(&operator_entry.file_name())?;
        let metadata = std::fs::symlink_metadata(&operator_path)
            .map_err(|source| io_error(&operator_path, source))?;
        if metadata.file_type().is_symlink() || !metadata.is_dir() {
            return Err(format_error(format!(
                "committed operator entry {} is not a directory",
                operator_path.display()
            )));
        }
        for segment_entry in
            std::fs::read_dir(&operator_path).map_err(|source| io_error(&operator_path, source))?
        {
            let segment_entry = segment_entry.map_err(|source| io_error(&operator_path, source))?;
            let segment_path = segment_entry.path();
            let metadata = std::fs::symlink_metadata(&segment_path)
                .map_err(|source| io_error(&segment_path, source))?;
            if metadata.file_type().is_symlink() || !metadata.is_file() {
                return Err(format_error(format!(
                    "committed segment entry {} is not a regular file",
                    segment_path.display()
                )));
            }
            let file_name = segment_entry
                .file_name()
                .into_string()
                .map_err(|_| format_error("committed segment name is not UTF-8".into()))?;
            let epoch = parse_segment_file_name(&file_name)?;
            let relative = format!("committed/{lineage_hash}/{operator_name}/{file_name}");
            let is_newer = latest_epoch.is_none_or(|latest| epoch > latest);
            if !retained_paths.contains(&relative) && !is_newer {
                deletions.push(segment_path);
                sync_directories.insert(operator_path.clone());
            }
        }
    }

    for path in &deletions {
        std::fs::remove_file(path).map_err(|source| io_error(path, source))?;
    }
    for directory in sync_directories {
        sync_directory(&directory)?;
    }
    if !deletions.is_empty() {
        sync_directory(&lineage_root)?;
    }
    Ok(deletions.len())
}

fn parse_segment_file_name(file_name: &str) -> Result<Epoch> {
    let (epoch, tail) = file_name
        .split_once('-')
        .ok_or_else(|| format_error(format!("unexpected committed file {file_name:?}")))?;
    let hash = tail
        .strip_suffix(".arrow")
        .or_else(|| tail.strip_suffix(".segment"))
        .ok_or_else(|| format_error(format!("unexpected committed file {file_name:?}")))?;
    if !is_sha256(hash) {
        return Err(format_error(format!(
            "unexpected committed file {file_name:?}"
        )));
    }
    epoch
        .parse::<u64>()
        .ok()
        .and_then(Epoch::new)
        .ok_or_else(|| format_error(format!("unexpected committed file {file_name:?}")))
}

fn portable_hash_name(name: &std::ffi::OsStr) -> Result<String> {
    let name = name
        .to_str()
        .filter(|name| is_sha256(name))
        .ok_or_else(|| format_error("unexpected committed operator directory".into()))?;
    Ok(name.into())
}

fn ensure_child_directory(parent: &Path, component: &str) -> Result<PathBuf> {
    validate_directory(parent)?;
    let path = parent.join(component);
    match std::fs::symlink_metadata(&path) {
        Ok(metadata) => {
            if metadata.file_type().is_symlink() || !metadata.is_dir() {
                return Err(format_error(format!(
                    "managed state entry {} is not a directory",
                    path.display()
                )));
            }
        }
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => {
            std::fs::create_dir(&path).map_err(|source| io_error(&path, source))?;
            sync_directory(parent)?;
        }
        Err(source) => return Err(io_error(&path, source)),
    }
    Ok(path)
}

fn validate_directory(path: &Path) -> Result<()> {
    let metadata = std::fs::symlink_metadata(path).map_err(|source| io_error(path, source))?;
    if metadata.file_type().is_symlink() || !metadata.is_dir() {
        Err(format_error(format!(
            "managed state root {} is not a directory",
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

fn lineage_hash(key: &StateLineageKey) -> String {
    digest(format!("{}\0{}", key.pipeline_name(), key.pipeline_fingerprint()).as_bytes())
}

fn digest(bytes: impl AsRef<[u8]>) -> String {
    hex::encode(Sha256::digest(bytes.as_ref()))
}

fn is_sha256(value: &str) -> bool {
    value.len() == 64
        && value
            .bytes()
            .all(|byte| byte.is_ascii_digit() || (b'a'..=b'f').contains(&byte))
}

fn lease_conflict(lineage_hash: &str) -> CalcFlowError {
    CalcFlowError::Conflict {
        resource: "state lineage".into(),
        key: lineage_hash.into(),
    }
}

fn segment_mismatch(handle: &StateHandle, coordinate: &str) -> CalcFlowError {
    CalcFlowError::CheckpointMismatch {
        message: format!(
            "state segment {:?} {coordinate} does not match its handle",
            handle.segment_id
        ),
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

async fn worker<T: Send + 'static>(
    operation: impl FnOnce() -> Result<T> + Send + 'static,
) -> Result<T> {
    tokio::task::spawn_blocking(operation)
        .await
        .map_err(|error| CalcFlowError::Internal {
            message: format!("state filesystem worker failed: {error}"),
        })?
}

#[cfg(test)]
mod tests {
    use std::collections::BTreeMap;

    use chrono::{TimeZone, Utc};
    use tempfile::TempDir;

    use super::{
        CommitFaultPoint, LocalStateBackend, commit_manifest_for_test, digest, lineage_hash,
    };
    use crate::{
        CheckpointManifest, CheckpointManifestFields, Epoch, OperatorManifestEntry, RecoveryStatus,
        StateBackend, StateHandle, StateLineageKey,
    };

    const PIPELINE_FINGERPRINT: &str =
        "0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef";
    const RUNTIME_CONFIG_HASH: &str =
        "abcdef0123456789abcdef0123456789abcdef0123456789abcdef0123456789";

    const FAULT_POINTS: [CommitFaultPoint; 10] = [
        CommitFaultPoint::AfterLease,
        CommitFaultPoint::AfterFirstSegmentStage,
        CommitFaultPoint::AfterAllSegmentsDurable,
        CommitFaultPoint::AfterSegmentValidation,
        CommitFaultPoint::AfterFirstSegmentPublication,
        CommitFaultPoint::AfterCommittedSynchronization,
        CommitFaultPoint::AfterManifestValidation,
        CommitFaultPoint::AfterManifestDurableWrite,
        CommitFaultPoint::AfterManifestRename,
        CommitFaultPoint::AfterManifestPublication,
    ];

    fn state_handle(
        key: &StateLineageKey,
        epoch: Epoch,
        segment_id: &str,
        bytes: &[u8],
    ) -> StateHandle {
        StateHandle::new(
            "window",
            epoch,
            segment_id,
            &format!(
                "committed/{}/{}/{}-{}.arrow",
                lineage_hash(key),
                digest("window"),
                epoch.as_u64(),
                digest(segment_id)
            ),
            u64::try_from(bytes.len()).unwrap(),
            &digest(bytes),
        )
        .unwrap()
    }

    fn manifest(epoch: Epoch, mut segments: Vec<StateHandle>) -> CheckpointManifest {
        segments.sort();
        CheckpointManifest::new(CheckpointManifestFields {
            pipeline_name: "orders".into(),
            pipeline_fingerprint: PIPELINE_FINGERPRINT.into(),
            runtime_config_hash: RUNTIME_CONFIG_HASH.into(),
            epoch,
            created_at: Utc.with_ymd_and_hms(2026, 8, 8, 8, 0, 0).unwrap(),
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

    #[tokio::test]
    async fn every_commit_fault_selects_only_the_previous_or_complete_manifest() {
        for fault in FAULT_POINTS {
            let directory = TempDir::new().unwrap();
            let state_root = directory.path().join("state");
            let manifest_root = directory.path().join("manifests");
            let key = StateLineageKey::new("orders", PIPELINE_FINGERPRINT).unwrap();
            let backend = LocalStateBackend::new(&state_root).await.unwrap();
            let lineage = backend.open_lineage(&key).await.unwrap();

            let previous_bytes = b"previous-base".to_vec();
            let previous_handle = state_handle(&key, Epoch::INITIAL, "base-0001", &previous_bytes);
            let previous = manifest(Epoch::INITIAL, vec![previous_handle.clone()]);
            commit_manifest_for_test(
                lineage.as_ref(),
                &manifest_root,
                &previous,
                &BTreeMap::from([(previous_handle.clone(), previous_bytes)]),
                None,
            )
            .await
            .unwrap();

            let next_epoch = Epoch::INITIAL.next().unwrap();
            let first_bytes = b"delta-first".to_vec();
            let first_handle = state_handle(&key, next_epoch, "delta-0001", &first_bytes);
            let second_bytes = b"delta-second".to_vec();
            let second_handle = state_handle(&key, next_epoch, "delta-0002", &second_bytes);
            let candidate = manifest(
                next_epoch,
                vec![previous_handle, first_handle.clone(), second_handle.clone()],
            );
            let staged =
                BTreeMap::from([(first_handle, first_bytes), (second_handle, second_bytes)]);

            assert!(
                commit_manifest_for_test(
                    lineage.as_ref(),
                    &manifest_root,
                    &candidate,
                    &staged,
                    Some(fault),
                )
                .await
                .is_err(),
                "fault {fault:?} did not interrupt publication"
            );
            drop(lineage);
            drop(backend);

            let selected_bytes = tokio::fs::read(manifest_root.join("manifest.json"))
                .await
                .unwrap();
            let selected = CheckpointManifest::from_bytes(&selected_bytes).unwrap();
            assert!(
                selected == previous || selected == candidate,
                "fault {fault:?} exposed a partial manifest"
            );
            if matches!(
                fault,
                CommitFaultPoint::AfterManifestRename | CommitFaultPoint::AfterManifestPublication
            ) {
                assert_eq!(selected, candidate, "fault {fault:?}");
            } else {
                assert_eq!(selected, previous, "fault {fault:?}");
            }

            let restarted = LocalStateBackend::new(&state_root).await.unwrap();
            let restarted_lineage = restarted.open_lineage(&key).await.unwrap();
            for operator in selected.operators().values() {
                for handle in &operator.segments {
                    restarted_lineage.load_segment(handle).await.unwrap();
                }
            }
        }
    }
}
