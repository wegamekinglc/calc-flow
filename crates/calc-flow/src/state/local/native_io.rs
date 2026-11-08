use super::{
    Arc, CalcFlowError, Digest as _, File, LocalStateLineageBackend, ManagedSegmentPaths,
    OwnedMutexGuard, Path, Read as _, Result, Sha256, StateHandle, Write as _, io_error,
    prepare_segment_directories, publication, reject_committed_file, segment_metadata,
    segment_mismatch, sync_staging_directory, validate_segment_metadata,
};

use crate::StateSegment;

type Check<'a> = &'a mut dyn FnMut() -> Result<()>;

#[cfg(test)]
#[derive(Clone, Copy, Eq, PartialEq)]
pub(crate) enum Point {
    OpaqueLoad,
    Read,
    Stage,
}
#[cfg(test)]
pub(super) type Hook = Arc<dyn Fn(Point) -> Result<()> + Send + Sync>;

#[cfg(test)]
impl LocalStateLineageBackend {
    pub(crate) fn set_native_io_hook(&self, hook: Hook) {
        *self.native_io_hook.lock() = Some(hook);
    }
    pub(super) fn observe_native_io(&self, point: Point) -> Result<()> {
        let hook = self.native_io_hook.lock().clone();
        if let Some(hook) = hook {
            hook(point)?;
        }
        Ok(())
    }
}

pub(crate) struct NativeSegmentIo {
    lineage: Arc<LocalStateLineageBackend>,
    paths: ManagedSegmentPaths,
    handle: StateHandle,
}
impl LocalStateLineageBackend {
    pub(crate) fn native_io(self: &Arc<Self>, handle: &StateHandle) -> Result<NativeSegmentIo> {
        if !self.supports_prepaid_load() {
            return Err(CalcFlowError::CheckpointMismatch {
                message: "native:array_sequence@1 state.reader: requires a certified local backend"
                    .into(),
            });
        }
        Ok(NativeSegmentIo {
            lineage: Arc::clone(self),
            paths: self.managed_paths(handle)?,
            handle: handle.clone(),
        })
    }
    pub(crate) async fn native_publication_guard(
        &self,
        mut check: impl FnMut() -> Result<()> + Send,
    ) -> Result<OwnedMutexGuard<()>> {
        let acquired = self.publication.clone().lock_owned();
        tokio::pin!(acquired);
        loop {
            check()?;
            tokio::select! {
                guard = &mut acquired => return Ok(guard),
                () = tokio::time::sleep(std::time::Duration::from_millis(10)) => {},
            }
        }
    }
}
impl NativeSegmentIo {
    pub(crate) fn verify(&self, check: Check<'_>) -> Result<()> {
        verify_file(&self.paths.committed, &self.handle, check)
    }
    pub(crate) fn stage_and_publish(&self, segment: &StateSegment, check: Check<'_>) -> Result<()> {
        check()?;
        #[cfg(test)]
        self.lineage.observe_native_io(Point::Stage)?;
        self.prepare(segment, check)?;
        self.publish(check)?;
        self.lineage.validated.lock().remove(&self.handle);
        check()
    }
    fn prepare(&self, segment: &StateSegment, check: Check<'_>) -> Result<()> {
        if !matches_file(&self.paths.committed, &self.handle, check)? {
            self.stage(segment, check)?;
            verify_file(&self.paths.staging, &self.handle, check)?;
            self.lineage.validated.lock().insert(self.handle.clone());
        }
        Ok(())
    }
    fn stage(&self, segment: &StateSegment, check: Check<'_>) -> Result<()> {
        verify_bytes(&self.handle, segment.bytes(), check)?;
        #[cfg(test)]
        let hook = self.lineage.publication_hook.lock().clone();
        prepare_segment_directories(
            &self.paths,
            &mut self.lineage.segment_directories.lock(),
            #[cfg(test)]
            hook.as_ref(),
        )?;
        if matches_file(&self.paths.staging, &self.handle, check)? {
            return Ok(());
        }
        reject_committed_file(&self.paths, &self.handle)?;
        write_staging(
            &self.paths,
            segment.bytes(),
            check,
            #[cfg(test)]
            hook.as_ref(),
        )?;
        sync_staging_directory(
            &self.paths.staging_parent,
            #[cfg(test)]
            hook.as_ref(),
        )
    }
    fn publish(&self, check: Check<'_>) -> Result<()> {
        #[cfg(test)]
        let hook = self.lineage.publication_hook.lock().clone();
        let validated = self.lineage.validated.lock().contains(&self.handle);
        publication::publish_native_file(
            &self.handle,
            &self.paths,
            validated,
            &self.lineage.segment_directories,
            check,
            #[cfg(test)]
            hook.as_ref(),
        )
    }
}

pub(super) fn verify_file(path: &Path, handle: &StateHandle, check: Check<'_>) -> Result<()> {
    check()?;
    let metadata = segment_metadata(path, handle)?;
    validate_segment_metadata(path, handle, &metadata)?;
    let mut file = File::open(path).map_err(|source| io_error(path, source))?;
    let digest = file_digest(&mut file, path, handle.byte_len(), check)?;
    file_end(&mut file, path, handle, check)?;
    verify_digest(handle, &digest)
}
pub(super) fn matches_file(path: &Path, handle: &StateHandle, check: Check<'_>) -> Result<bool> {
    check()?;
    match std::fs::symlink_metadata(path) {
        Ok(_) => verify_file(path, handle, check).map(|()| true),
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => Ok(false),
        Err(source) => Err(io_error(path, source)),
    }
}
fn file_digest(
    file: &mut File,
    path: &Path,
    mut remaining: u64,
    check: Check<'_>,
) -> Result<[u8; 32]> {
    let mut buffer = [0_u8; 4096];
    let mut hash = Sha256::new();
    while remaining != 0 {
        check()?;
        let count = usize::try_from(remaining.min(4096)).map_err(|_| segment_length_error())?;
        file.read_exact(&mut buffer[..count])
            .map_err(|source| io_error(path, source))?;
        hash.update(&buffer[..count]);
        remaining -= u64::try_from(count).map_err(|_| segment_length_error())?;
    }
    check()?;
    Ok(hash.finalize().into())
}
fn file_end(file: &mut File, path: &Path, handle: &StateHandle, check: Check<'_>) -> Result<()> {
    check()?;
    let mut extra = [0_u8; 1];
    if file
        .read(&mut extra)
        .map_err(|source| io_error(path, source))?
        != 0
    {
        return Err(segment_mismatch(handle, "byte length"));
    }
    check()
}
fn verify_bytes(handle: &StateHandle, bytes: &[u8], check: Check<'_>) -> Result<()> {
    if u64::try_from(bytes.len()).ok() != Some(handle.byte_len()) {
        return Err(segment_mismatch(handle, "byte length"));
    }
    let mut hash = Sha256::new();
    for chunk in bytes.chunks(4096) {
        check()?;
        hash.update(chunk);
    }
    check()?;
    verify_digest(handle, &hash.finalize().into())
}
fn verify_digest(handle: &StateHandle, digest: &[u8; 32]) -> Result<()> {
    let mut encoded = [0_u8; 64];
    hex::encode_to_slice(digest, &mut encoded).map_err(|_| segment_length_error())?;
    if encoded != handle.sha256().as_bytes() {
        return Err(segment_mismatch(handle, "checksum"));
    }
    Ok(())
}
fn segment_length_error() -> CalcFlowError {
    CalcFlowError::CheckpointMismatch {
        message: "native:array_sequence@1 state.bytes: size overflow".into(),
    }
}
fn write_staging(
    paths: &ManagedSegmentPaths,
    bytes: &[u8],
    check: Check<'_>,
    #[cfg(test)] hook: Option<&super::publication_tests::PublicationHook>,
) -> Result<()> {
    check()?;
    let mut temporary = tempfile::NamedTempFile::new_in(&paths.staging_parent)
        .map_err(|source| io_error(&paths.staging_parent, source))?;
    for chunk in bytes.chunks(4096) {
        check()?;
        temporary
            .write_all(chunk)
            .map_err(|source| io_error(temporary.path(), source))?;
    }
    finish_staging(
        temporary,
        paths,
        #[cfg(test)]
        hook,
    )
}
fn finish_staging(
    mut temporary: tempfile::NamedTempFile,
    paths: &ManagedSegmentPaths,
    #[cfg(test)] hook: Option<&super::publication_tests::PublicationHook>,
) -> Result<()> {
    temporary
        .flush()
        .map_err(|source| io_error(temporary.path(), source))?;
    #[cfg(test)]
    super::publication_tests::observe(
        hook,
        super::publication_tests::Operation::FileSync,
        temporary.path(),
    )?;
    temporary
        .as_file()
        .sync_all()
        .map_err(|source| io_error(temporary.path(), source))?;
    temporary
        .persist_noclobber(&paths.staging)
        .map_err(|error| io_error(&paths.staging, error.error))?;
    Ok(())
}
