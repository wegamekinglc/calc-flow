use std::{
    collections::BTreeSet,
    fmt,
    fs::{File, OpenOptions},
    io::Write as _,
    path::{Path, PathBuf},
    sync::{
        Arc, LazyLock,
        atomic::{AtomicU64, Ordering},
    },
};

use fs2::FileExt;
use parking_lot::Mutex;

use crate::{CalcFlowError, CancellationToken, LocalStateBackend};

const STATE_CHILD: &str = "state";
const MANIFEST_CHILD: &str = "manifests";
const PREFLIGHT_PROBE_PREFIX: &str = ".tmp-checkpoint-preflight";

static PROCESS_LEASES: LazyLock<Mutex<BTreeSet<PathBuf>>> =
    LazyLock::new(|| Mutex::new(BTreeSet::new()));
static NEXT_PROBE_ID: AtomicU64 = AtomicU64::new(0);

/// Crate-private single-root checkpoint owner prepared for the A6 public cutover.
pub(crate) struct ManagedCheckpointRuntime {
    managed_root: PathBuf,
}

impl ManagedCheckpointRuntime {
    /// Captures one lexical managed root without performing filesystem I/O.
    pub(crate) fn new(managed_root: impl Into<PathBuf>) -> crate::Result<Self> {
        let managed_root = managed_root.into();
        if managed_root.as_os_str().is_empty() {
            return Err(CalcFlowError::InvalidArgument {
                field: "managed_checkpoint_root".into(),
                message: "must not be empty".into(),
            });
        }
        Ok(Self { managed_root })
    }

    /// Opens the fixed local namespace and retains its complete-root lease.
    pub(crate) async fn open(
        self,
        cancellation: &CancellationToken,
    ) -> crate::Result<OpenedManagedCheckpointRuntime> {
        ensure_open_active(cancellation)?;
        open_managed_storage(self.managed_root, cancellation).await
    }
}

async fn open_managed_storage(
    requested: PathBuf,
    cancellation: &CancellationToken,
) -> crate::Result<OpenedManagedCheckpointRuntime> {
    let prepared = prepare_managed_namespace(&requested).await?;
    let state_backend = LocalStateBackend::new(&prepared.state_root)
        .await
        .map_err(|_| initialization_error())?;
    ensure_open_active(cancellation)?;
    Ok(OpenedManagedCheckpointRuntime {
        state_backend: Arc::new(state_backend),
        manifest_root: prepared.manifest_root,
        _lease: prepared.lease,
    })
}

struct PreparedManagedNamespace {
    state_root: PathBuf,
    manifest_root: PathBuf,
    lease: ManagedRootLease,
}

async fn prepare_managed_namespace(requested: &Path) -> crate::Result<PreparedManagedNamespace> {
    let canonical_root = prepare_managed_root(requested).await?;
    let lease = acquire_managed_root_lease(canonical_root.clone()).await?;
    let state_root = prepare_fixed_child(&canonical_root, STATE_CHILD).await?;
    let manifest_root = prepare_fixed_child(&canonical_root, MANIFEST_CHILD).await?;
    preflight_directory(&state_root).await?;
    preflight_directory(&manifest_root).await?;
    Ok(PreparedManagedNamespace {
        state_root,
        manifest_root,
        lease,
    })
}

fn ensure_open_active(cancellation: &CancellationToken) -> crate::Result<()> {
    if cancellation.is_cancelled() {
        Err(cancelled_open())
    } else {
        Ok(())
    }
}

impl fmt::Debug for ManagedCheckpointRuntime {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("ManagedCheckpointRuntime")
            .field("managed_root", &"<redacted>")
            .finish()
    }
}

/// Prepared engine-owned storage retained for the lifetime of one checkpoint runtime.
pub(crate) struct OpenedManagedCheckpointRuntime {
    state_backend: Arc<LocalStateBackend>,
    manifest_root: PathBuf,
    _lease: ManagedRootLease,
}

impl OpenedManagedCheckpointRuntime {
    pub(crate) fn state_backend(&self) -> Arc<LocalStateBackend> {
        Arc::clone(&self.state_backend)
    }

    pub(crate) fn manifest_root(&self) -> &Path {
        &self.manifest_root
    }

    #[cfg(test)]
    pub(crate) fn state_root_for_test(&self) -> PathBuf {
        self.manifest_root
            .parent()
            .expect("fixed manifest child has the managed root as parent")
            .join(STATE_CHILD)
    }

    #[cfg(test)]
    pub(crate) fn manifest_root_for_test(&self) -> PathBuf {
        self.manifest_root.clone()
    }
}

impl fmt::Debug for OpenedManagedCheckpointRuntime {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("OpenedManagedCheckpointRuntime")
            .field("managed_root", &"<redacted>")
            .finish_non_exhaustive()
    }
}

struct ManagedRootLease {
    canonical_root: PathBuf,
    files: Vec<File>,
}

impl fmt::Debug for ManagedRootLease {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("ManagedRootLease")
            .field("canonical_root", &"<redacted>")
            .finish_non_exhaustive()
    }
}

impl Drop for ManagedRootLease {
    fn drop(&mut self) {
        for file in &self.files {
            let _ = FileExt::unlock(file);
        }
        PROCESS_LEASES.lock().remove(&self.canonical_root);
    }
}

async fn prepare_managed_root(requested: &Path) -> crate::Result<PathBuf> {
    let requested = requested.to_owned();
    tokio::task::spawn_blocking(move || {
        std::fs::create_dir_all(&requested).map_err(|_| initialization_error())?;
        let canonical = std::fs::canonicalize(&requested).map_err(|_| initialization_error())?;
        validate_directory(&canonical)?;
        Ok(canonical)
    })
    .await
    .map_err(|_| initialization_error())?
}

async fn prepare_fixed_child(root: &Path, child: &str) -> crate::Result<PathBuf> {
    let root = root.to_owned();
    let child = child.to_owned();
    tokio::task::spawn_blocking(move || prepare_fixed_child_blocking(&root, &child))
        .await
        .map_err(|_| initialization_error())?
}

fn prepare_fixed_child_blocking(root: &Path, child: &str) -> crate::Result<PathBuf> {
    let requested = root.join(child);
    ensure_fixed_child_directory(&requested)?;
    let canonical = std::fs::canonicalize(&requested).map_err(|_| initialization_error())?;
    validate_fixed_child_parent(root, canonical)
}

fn ensure_fixed_child_directory(requested: &Path) -> crate::Result<()> {
    match std::fs::symlink_metadata(requested) {
        Ok(metadata) => validate_fixed_child_metadata(&metadata),
        Err(error) => create_missing_fixed_child(requested, &error),
    }
}

fn validate_fixed_child_metadata(metadata: &std::fs::Metadata) -> crate::Result<()> {
    if metadata.file_type().is_symlink() || !metadata.is_dir() {
        Err(initialization_error())
    } else {
        Ok(())
    }
}

fn create_missing_fixed_child(requested: &Path, error: &std::io::Error) -> crate::Result<()> {
    if error.kind() != std::io::ErrorKind::NotFound {
        return Err(initialization_error());
    }
    std::fs::create_dir(requested).map_err(|_| initialization_error())
}

fn validate_fixed_child_parent(root: &Path, canonical: PathBuf) -> crate::Result<PathBuf> {
    if canonical.parent() == Some(root) {
        Ok(canonical)
    } else {
        Err(initialization_error())
    }
}

async fn preflight_directory(root: &Path) -> crate::Result<()> {
    let root = root.to_owned();
    tokio::task::spawn_blocking(move || {
        validate_directory(&root)?;
        std::fs::read_dir(&root).map_err(|_| initialization_error())?;
        let probe_id = NEXT_PROBE_ID.fetch_add(1, Ordering::Relaxed);
        let probe = root.join(format!(
            "{PREFLIGHT_PROBE_PREFIX}-{}-{probe_id}",
            std::process::id()
        ));
        let mut file = OpenOptions::new()
            .write(true)
            .create_new(true)
            .open(&probe)
            .map_err(|_| initialization_error())?;
        let result = file
            .write_all(b"calc-flow checkpoint preflight")
            .and_then(|()| file.sync_all())
            .map_err(|_| initialization_error());
        drop(file);
        let removed = std::fs::remove_file(&probe).map_err(|_| initialization_error());
        result.and(removed)
    })
    .await
    .map_err(|_| initialization_error())?
}

async fn acquire_managed_root_lease(root: PathBuf) -> crate::Result<ManagedRootLease> {
    tokio::task::spawn_blocking(move || acquire_managed_root_lease_blocking(root))
        .await
        .map_err(|_| initialization_error())?
}

fn acquire_managed_root_lease_blocking(root: PathBuf) -> crate::Result<ManagedRootLease> {
    register_process_lease(&root)?;
    let files = acquire_directory_locks(&root).inspect_err(|_| {
        unregister_process_lease(&root);
    })?;
    Ok(ManagedRootLease {
        canonical_root: root,
        files,
    })
}

fn register_process_lease(root: &Path) -> crate::Result<()> {
    let mut process_leases = PROCESS_LEASES.lock();
    if process_leases
        .iter()
        .any(|leased| paths_overlap(leased, root))
    {
        return Err(lease_conflict());
    }
    process_leases.insert(root.to_owned());
    Ok(())
}

fn unregister_process_lease(root: &Path) {
    PROCESS_LEASES.lock().remove(root);
}

fn acquire_directory_locks(root: &Path) -> crate::Result<Vec<File>> {
    let ancestors = root.ancestors().collect::<Vec<_>>();
    let mut files = Vec::with_capacity(ancestors.len());
    for ancestor in ancestors.iter().rev() {
        let file = open_directory_for_lock(ancestor).map_err(|_| initialization_error())?;
        try_lock_directory(&file, *ancestor == root)?;
        files.push(file);
    }
    Ok(files)
}

#[cfg(not(windows))]
fn open_directory_for_lock(path: &Path) -> std::io::Result<File> {
    File::open(path)
}

#[cfg(windows)]
fn open_directory_for_lock(path: &Path) -> std::io::Result<File> {
    use std::os::windows::fs::OpenOptionsExt as _;

    const FILE_FLAG_BACKUP_SEMANTICS: u32 = 0x0200_0000;
    OpenOptions::new()
        .read(true)
        .custom_flags(FILE_FLAG_BACKUP_SEMANTICS)
        .open(path)
}

fn try_lock_directory(file: &File, exclusive: bool) -> crate::Result<()> {
    let result = if exclusive {
        FileExt::try_lock_exclusive(file)
    } else {
        FileExt::try_lock_shared(file)
    };
    match result {
        Ok(()) => Ok(()),
        Err(error) if error.kind() == std::io::ErrorKind::WouldBlock => Err(lease_conflict()),
        Err(_) => Err(initialization_error()),
    }
}

fn paths_overlap(left: &Path, right: &Path) -> bool {
    left == right || left.starts_with(right) || right.starts_with(left)
}

fn validate_directory(path: &Path) -> crate::Result<()> {
    let metadata = std::fs::symlink_metadata(path).map_err(|_| initialization_error())?;
    if metadata.file_type().is_symlink() || !metadata.is_dir() {
        Err(initialization_error())
    } else {
        Ok(())
    }
}

fn initialization_error() -> CalcFlowError {
    CalcFlowError::Internal {
        message: "managed checkpoint storage initialization failed".into(),
    }
}

fn lease_conflict() -> CalcFlowError {
    CalcFlowError::Conflict {
        resource: "managed checkpoint directory".into(),
        key: "active".into(),
    }
}

fn cancelled_open() -> CalcFlowError {
    CalcFlowError::Cancelled {
        run_id: "managed-checkpoint-open".into(),
    }
}
