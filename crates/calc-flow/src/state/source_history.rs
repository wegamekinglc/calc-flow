mod budget;
mod files;
mod work;

pub use budget::SourceHistoryLimits;
use budget::{BufferCredit, Ledger, StorageCredit, limit_error};
use work::WorkOwner;
#[cfg(test)]
mod tests;

use super::SourceHistoryManifestEntry;
use super::v4::{OpaqueObject, SourceHistorySegmentHandle, V4Store, native::V4ObjectStore};
use crate::{CalcFlowError, Epoch, JsonMap, Result, StateSegment};
use parking_lot::Mutex;
use sha2::{Digest as _, Sha256};
use std::{
    collections::BTreeMap,
    io::Read as _,
    path::{Path, PathBuf},
    sync::Arc,
};

/// Immutable-history protocol and its per-source resource limits.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct SourceHistorySpec {
    /// Connector-defined current history protocol.
    pub contract: String,
    /// Raw storage and buffer limits.
    pub limits: SourceHistoryLimits,
}

impl SourceHistorySpec {
    /// Constructs a validated history declaration.
    ///
    /// # Errors
    /// Returns an invalid-argument error for a nonportable contract or invalid limits.
    pub fn new(contract: impl Into<String>, limits: SourceHistoryLimits) -> Result<Self> {
        let value = Self {
            contract: contract.into(),
            limits,
        };
        value.validate()?;
        Ok(value)
    }

    pub(crate) fn validate(&self) -> Result<()> {
        crate::json::validate_portable_identifier("source_history.contract", &self.contract)?;
        self.limits.validate()
    }
}

struct Segment {
    handle: SourceHistorySegmentHandle,
    _storage: StorageCredit,
}

#[derive(Default)]
struct History {
    segments: BTreeMap<String, Arc<Segment>>,
    sealed: Option<SourceHistoryManifestEntry>,
}

struct Inner {
    lineage: PathBuf,
    source_id: String,
    epoch: Epoch,
    spec: SourceHistorySpec,
    ledger: Arc<Ledger>,
    history: Arc<Mutex<History>>,
    work: Arc<WorkOwner>,
    objects: Option<Arc<tokio::sync::Mutex<Box<dyn V4ObjectStore>>>>,
}

/// Runtime-owned managed storage for one immutable source history.
///
/// Sources receive this context before `open`. Operations are serialized;
/// the runtime owns and drains their tasks when the source closes.
pub struct SourceHistoryContext(Arc<Inner>);

/// Checksum-verified raw bytes with their buffer credit and retention owner.
pub struct SourceHistoryBytes {
    bytes: Vec<u8>,
    _buffer: Arc<BufferCredit>,
    _segment: Arc<Segment>,
}

impl SourceHistoryBytes {
    /// Borrows the immutable raw input.
    pub fn bytes(&self) -> &[u8] {
        &self.bytes
    }
}

impl SourceHistoryContext {
    pub(crate) fn new(
        lineage: impl Into<PathBuf>,
        source_id: &str,
        epoch: Epoch,
        spec: SourceHistorySpec,
        restored: Option<SourceHistoryManifestEntry>,
    ) -> Result<Self> {
        let lineage = lineage.into();
        if lineage.as_os_str().is_empty() {
            return Err(CalcFlowError::InvalidArgument {
                field: "source_history.lineage".into(),
                message: "must not be empty".into(),
            });
        }
        spec.validate()?;
        let ledger = Ledger::new(spec.limits);
        let history = match restored {
            Some(restored) => restored_history(source_id, &spec, &ledger, restored)?,
            None => History::default(),
        };
        Ok(Self(Arc::new(Inner {
            lineage,
            source_id: source_id.into(),
            epoch,
            spec,
            ledger,
            history: Arc::new(Mutex::new(history)),
            work: Arc::new(WorkOwner::default()),
            objects: None,
        })))
    }

    pub(crate) fn with_object_store(mut self, store: Box<dyn V4ObjectStore>) -> Self {
        let objects = Some(Arc::new(tokio::sync::Mutex::new(store)));
        self.0 = Arc::new(Inner {
            lineage: self.0.lineage.clone(),
            source_id: self.0.source_id.clone(),
            epoch: self.0.epoch,
            spec: self.0.spec.clone(),
            ledger: self.0.ledger.clone(),
            history: self.0.history.clone(),
            work: self.0.work.clone(),
            objects,
        });
        self
    }

    pub(crate) fn fork(&self) -> Self {
        Self(self.0.clone())
    }

    /// Returns the declared raw-history limits.
    pub fn limits(&self) -> SourceHistoryLimits {
        self.0.spec.limits
    }

    /// Returns the frozen descriptor on recovery or after sealing.
    pub fn manifest(&self) -> Option<SourceHistoryManifestEntry> {
        self.0.history.lock().sealed.clone()
    }

    /// Discovers a bounded, sorted file or flat directory of regular files.
    ///
    /// # Errors
    /// Returns an I/O or admission error for invalid entries or excess membership.
    pub async fn discover_files(&self, path: &Path, extension: &str) -> Result<Vec<PathBuf>> {
        let path = path.to_path_buf();
        let extension = extension.to_owned();
        let max_files = self.limits().max_segments;
        self.0
            .work
            .run(async move {
                tokio::task::spawn_blocking(move || files::discover(&path, &extension, max_files))
                    .await
                    .map_err(|error| CalcFlowError::Internal {
                        message: format!("history discovery failed: {error}"),
                    })?
            })
            .await
    }

    /// Archives one regular file under a stable segment identifier.
    ///
    /// # Errors
    /// Returns an I/O, limit, identity, or backend error before admitting invalid history.
    pub async fn archive_file(
        &self,
        segment_id: &str,
        path: &Path,
        max_file_bytes: u64,
    ) -> Result<SourceHistorySegmentHandle> {
        self.validate_archive_member(segment_id)?;
        let lineage = self.0.lineage.clone();
        let ledger = self.0.ledger.clone();
        let history = self.0.history.clone();
        let objects = self.0.objects.clone();
        let source_id = self.0.source_id.clone();
        let epoch = self.0.epoch;
        let segment_id = segment_id.to_owned();
        let path = path.to_path_buf();
        self.0
            .work
            .run(async move {
                if history.lock().sealed.is_some() {
                    return Err(limit_error("metadata", "history is sealed"));
                }
                let read =
                    tokio::task::spawn_blocking(move || read_file(&path, &ledger, max_file_bytes))
                        .await
                        .map_err(|error| CalcFlowError::Internal {
                            message: format!("history file read failed: {error}"),
                        })??;
                let (segment, storage) = read;
                let (handle, entry) = if let Some(store) = objects {
                    persist_archived_segment(
                        &mut **store.lock().await,
                        epoch,
                        &source_id,
                        &segment_id,
                        segment,
                        storage,
                    )
                    .await?
                } else {
                    archived_segment(&lineage, &source_id, epoch, &segment_id, segment, storage)
                        .await?
                };
                history.lock().segments.insert(segment_id, entry);
                Ok(handle)
            })
            .await
    }

    fn validate_archive_member(&self, segment_id: &str) -> Result<()> {
        crate::json::validate_portable_identifier("source_history.segment_id", segment_id)?;
        {
            let history = self.0.history.lock();
            if history.sealed.is_some() || history.segments.contains_key(segment_id) {
                return Err(limit_error(
                    "segment_id",
                    "history is sealed or the segment already exists",
                ));
            }
        }
        Ok(())
    }

    /// Seals the captured membership and connector metadata for every later checkpoint.
    ///
    /// # Errors
    /// Returns an invalid-argument error when the context was already sealed.
    pub fn seal(&self, inline_metadata: JsonMap) -> Result<()> {
        self.0.work.when_idle(|| {
            let mut history = self.0.history.lock();
            if history.sealed.is_some() {
                return Err(limit_error("metadata", "history is already sealed"));
            }
            history.sealed = Some(SourceHistoryManifestEntry {
                format_version: super::SOURCE_HISTORY_FORMAT_VERSION,
                contract: self.0.spec.contract.clone(),
                inline_metadata,
                segments: history
                    .segments
                    .values()
                    .map(|segment| segment.handle.clone())
                    .collect(),
            });
            Ok(())
        })
    }

    /// Loads a captured segment with checksum validation and prepaid raw buffers.
    ///
    /// # Errors
    /// Returns a limit, unknown-segment, checksum, or backend error.
    pub async fn load(&self, segment_id: &str) -> Result<SourceHistoryBytes> {
        let segment = self
            .0
            .history
            .lock()
            .segments
            .get(segment_id)
            .cloned()
            .ok_or_else(|| history_mismatch(&self.0.source_id, "unknown history segment"))?;
        let buffer = self.0.ledger.buffer(segment.handle.byte_len())?;
        let lineage = self.0.lineage.clone();
        self.0
            .work
            .run(async move {
                let handle = segment.handle.clone();
                let bytes =
                    tokio::task::spawn_blocking(move || V4Store::new(lineage).read_opaque(&handle))
                        .await
                        .map_err(|error| CalcFlowError::Internal {
                            message: format!("history read failed: {error}"),
                        })??;
                Ok(SourceHistoryBytes {
                    bytes,
                    _buffer: buffer,
                    _segment: segment,
                })
            })
            .await
    }

    pub(crate) async fn drain(&self) -> Vec<CalcFlowError> {
        self.0.work.drain().await
    }
}

fn read_file(
    path: &Path,
    ledger: &Arc<Ledger>,
    max_file_bytes: u64,
) -> Result<(StateSegment, StorageCredit)> {
    let length = file_length(path, max_file_bytes)?;
    let storage = ledger.storage(length)?;
    let buffer = ledger.buffer(length)?;
    let len = usize::try_from(length)
        .map_err(|_| limit_error("max_buffer_bytes", "file length exceeds address space"))?;
    let mut bytes = vec![0; len];
    read_file_bytes(path, &mut bytes)?;
    Ok((StateSegment::new(bytes).with_owner(buffer), storage))
}

pub(crate) fn history_mismatch(source_id: &str, message: &str) -> CalcFlowError {
    CalcFlowError::CheckpointMismatch {
        message: format!("source {source_id:?}: {message}"),
    }
}

fn restored_history(
    source_id: &str,
    spec: &SourceHistorySpec,
    ledger: &Arc<Ledger>,
    restored: SourceHistoryManifestEntry,
) -> Result<History> {
    validate_restored_history(source_id, spec, &restored)?;
    let mut history = History::default();
    let credits = restored_storage(ledger, &restored.segments)?;
    for (handle, storage) in restored.segments.iter().zip(credits) {
        if handle.owner != source_id {
            return Err(history_mismatch(source_id, "history segment owner changed"));
        }
        let segment = Arc::new(Segment {
            handle: handle.clone(),
            _storage: storage,
        });
        if history
            .segments
            .insert(handle.segment_id().into(), segment)
            .is_some()
        {
            return Err(history_mismatch(source_id, "duplicate history segment"));
        }
    }
    history.sealed = Some(restored);
    Ok(history)
}

fn validate_restored_history(
    source_id: &str,
    spec: &SourceHistorySpec,
    restored: &SourceHistoryManifestEntry,
) -> Result<()> {
    if restored.contract != spec.contract
        || restored.format_version != super::SOURCE_HISTORY_FORMAT_VERSION
    {
        return Err(history_mismatch(source_id, "history contract changed"));
    }
    if restored.segments.len() > spec.limits.max_segments {
        return Err(limit_error(
            "max_segments",
            "raw history exceeds its segment limit",
        ));
    }
    Ok(())
}

fn restored_storage(
    ledger: &Arc<Ledger>,
    segments: &[SourceHistorySegmentHandle],
) -> Result<Vec<StorageCredit>> {
    segments
        .iter()
        .map(|handle| {
            ledger.buffer(handle.byte_len())?;
            ledger.storage(handle.byte_len())
        })
        .collect()
}

async fn archived_segment(
    lineage: &Path,
    source_id: &str,
    epoch: Epoch,
    segment_id: &str,
    segment: StateSegment,
    storage: StorageCredit,
) -> Result<(SourceHistorySegmentHandle, Arc<Segment>)> {
    let mut store = V4Store::new(lineage);
    persist_archived_segment(&mut store, epoch, source_id, segment_id, segment, storage).await
}

async fn persist_archived_segment<S: V4ObjectStore + ?Sized>(
    store: &mut S,
    epoch: Epoch,
    source_id: &str,
    segment_id: &str,
    segment: StateSegment,
    storage: StorageCredit,
) -> Result<(SourceHistorySegmentHandle, Arc<Segment>)> {
    let bytes = segment.bytes().to_vec();
    let object = OpaqueObject {
        owner: source_id.into(),
        object_id: segment_id.into(),
        generation: epoch.as_u64(),
        sha256: hex::encode(Sha256::digest(&bytes)),
        bytes,
    };
    let mut encoded = store.write_opaque(epoch.as_u64(), vec![object]).await?;
    let handle = encoded.remove(0).handle;
    let entry = Arc::new(Segment {
        handle: handle.clone(),
        _storage: storage,
    });
    Ok((handle, entry))
}

fn file_length(path: &Path, max_file_bytes: u64) -> Result<u64> {
    let metadata = std::fs::symlink_metadata(path).map_err(|source| CalcFlowError::Io {
        path: path.display().to_string(),
        source,
    })?;
    if !metadata.is_file() || metadata.file_type().is_symlink() {
        return Err(limit_error("path", "history requires a regular file"));
    }
    if metadata.len() > max_file_bytes {
        return Err(limit_error(
            "max_file_bytes",
            "file exceeds its acquisition limit",
        ));
    }
    Ok(metadata.len())
}

fn read_file_bytes(path: &Path, bytes: &mut [u8]) -> Result<()> {
    let mut file = std::fs::File::open(path).map_err(|source| CalcFlowError::Io {
        path: path.display().to_string(),
        source,
    })?;
    file.read_exact(bytes).map_err(|source| CalcFlowError::Io {
        path: path.display().to_string(),
        source,
    })?;
    let mut extra = [0; 1];
    if file.read(&mut extra).map_err(|source| CalcFlowError::Io {
        path: path.display().to_string(),
        source,
    })? != 0
    {
        return Err(limit_error("path", "file grew during history acquisition"));
    }
    Ok(())
}
