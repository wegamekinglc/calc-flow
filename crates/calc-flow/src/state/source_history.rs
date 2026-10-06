mod budget;
mod files;
mod work;

pub use budget::SourceHistoryLimits;
use budget::{BufferCredit, Ledger, StorageCredit, limit_error};
use work::WorkOwner;
#[cfg(test)]
mod tests;

use super::{ManifestTransaction, SourceHistoryManifestEntry, StateHandle, WorkingStatePins};
use crate::{CalcFlowError, Epoch, JsonMap, OperatorStateSnapshot, Result, StateSegment};
use parking_lot::Mutex;
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
    handle: StateHandle,
    _working: Arc<WorkingStatePins>,
    _storage: StorageCredit,
}

#[derive(Default)]
struct History {
    segments: BTreeMap<String, Arc<Segment>>,
    sealed: Option<SourceHistoryManifestEntry>,
}

struct Inner {
    transaction: Arc<ManifestTransaction>,
    source_id: String,
    epoch: Epoch,
    spec: SourceHistorySpec,
    ledger: Arc<Ledger>,
    history: Arc<Mutex<History>>,
    work: Arc<WorkOwner>,
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
    pub(crate) async fn new(
        transaction: Arc<ManifestTransaction>,
        source_id: &str,
        epoch: Epoch,
        spec: SourceHistorySpec,
        restored: Option<SourceHistoryManifestEntry>,
    ) -> Result<Self> {
        spec.validate()?;
        let ledger = Ledger::new(spec.limits);
        let history = match restored {
            Some(restored) => {
                restored_history(&transaction, source_id, &spec, &ledger, restored).await?
            }
            None => History::default(),
        };
        Ok(Self(Arc::new(Inner {
            transaction,
            source_id: source_id.into(),
            epoch,
            spec,
            ledger,
            history: Arc::new(Mutex::new(history)),
            work: Arc::new(WorkOwner::default()),
        })))
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
    ) -> Result<StateHandle> {
        self.validate_archive_member(segment_id)?;
        let transaction = self.0.transaction.clone();
        let ledger = self.0.ledger.clone();
        let history = self.0.history.clone();
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
                let (handle, entry) = archived_segment(
                    &transaction,
                    &source_id,
                    epoch,
                    &segment_id,
                    segment,
                    storage,
                )
                .await?;
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
        let transaction = self.0.transaction.clone();
        self.0
            .work
            .run(async move {
                let bytes = transaction.load_source_history(&segment.handle).await?;
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

async fn restored_history(
    transaction: &ManifestTransaction,
    source_id: &str,
    spec: &SourceHistorySpec,
    ledger: &Arc<Ledger>,
    restored: SourceHistoryManifestEntry,
) -> Result<History> {
    validate_restored_history(source_id, spec, &restored)?;
    let mut history = History::default();
    let credits = restored_storage(ledger, &restored.segments)?;
    let working = transaction
        .pin_working_state(source_id, &restored.segments)
        .await?;
    for (handle, storage) in restored.segments.iter().zip(credits) {
        let segment = Arc::new(Segment {
            handle: handle.clone(),
            _working: working
                .clone()
                .ok_or_else(|| history_mismatch(source_id, "missing segment protection"))?,
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

fn restored_storage(ledger: &Arc<Ledger>, segments: &[StateHandle]) -> Result<Vec<StorageCredit>> {
    segments
        .iter()
        .map(|handle| {
            ledger.buffer(handle.byte_len())?;
            ledger.storage(handle.byte_len())
        })
        .collect()
}

async fn archived_segment(
    transaction: &ManifestTransaction,
    source_id: &str,
    epoch: Epoch,
    segment_id: &str,
    segment: StateSegment,
    storage: StorageCredit,
) -> Result<(StateHandle, Arc<Segment>)> {
    let mut staged = transaction
        .stage_operator_state(
            source_id,
            epoch,
            OperatorStateSnapshot {
                inline_metadata: BTreeMap::new(),
                segments: BTreeMap::from([(segment_id.to_owned(), segment)]),
            },
        )
        .await?;
    let handle = staged.segments.remove(0);
    let entry = Arc::new(Segment {
        handle: handle.clone(),
        _working: staged
            .working
            .ok_or_else(|| history_mismatch(source_id, "missing segment protection"))?,
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
