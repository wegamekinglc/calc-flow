#[cfg(test)]
use super::publication_tests::{Operation, PublicationHook, observe};
use super::{
    BTreeSet, CalcFlowError, ManagedSegmentPaths, Path, PathBuf, Result, StateHandle, SyncMutex,
    committed_file_matches, format_error, io_error, managed_segment_components,
    prepare_committed_directories, prepare_staging_directories, read_validated_file,
    sync_directory, validate_directory,
};

pub(super) fn publish_files(
    entries: Vec<(StateHandle, ManagedSegmentPaths, bool)>,
    directories: &SyncMutex<SegmentDirectories>,
    #[cfg(test)] hook: Option<&PublicationHook>,
) -> Result<()> {
    let mut seen = BTreeSet::new();
    let mut publication_directories: Vec<PathBuf> = Vec::new();
    for (handle, paths, validated) in entries {
        let (root, lineage, epoch, operator) = managed_segment_components(&paths)?;
        prepare_committed_directories(
            root,
            &lineage,
            &operator,
            &mut directories.lock(),
            #[cfg(test)]
            hook,
        )?;
        if !committed_file_matches(&paths.committed, &handle)? {
            if !validated {
                return Err(CalcFlowError::Conflict {
                    resource: "validated state segment".into(),
                    key: handle.segment_id().into(),
                });
            }
            prepare_staging_directories(
                root,
                &lineage,
                &epoch,
                &operator,
                &mut directories.lock(),
                #[cfg(test)]
                hook,
            )?;
            read_validated_file(&paths.staging, &handle)?;
            #[cfg(test)]
            observe(hook, Operation::Rename, &paths.committed)?;
            std::fs::rename(&paths.staging, &paths.committed)
                .map_err(|source| io_error(&paths.committed, source))?;
        }
        if seen.insert(paths.committed_parent.clone()) {
            publication_directories.push(paths.committed_parent);
        }
    }
    for directory in publication_directories {
        #[cfg(test)]
        observe(hook, Operation::PublishSync, &directory)?;
        sync_directory(&directory)?;
    }
    Ok(())
}

#[derive(Default)]
pub(super) struct SegmentDirectories {
    pending: BTreeSet<PathBuf>,
    confirmed_committed: BTreeSet<PathBuf>,
}

pub(super) fn ensure_segment_directory(
    parent: &Path,
    component: &str,
    directories: &mut SegmentDirectories,
    confirm_existing: bool,
    #[cfg(test)] hook: Option<&PublicationHook>,
) -> Result<PathBuf> {
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
            directories.pending.insert(path.clone());
        }
        Err(source) => return Err(io_error(&path, source)),
    }
    if directories.pending.contains(&path)
        || (confirm_existing && !directories.confirmed_committed.contains(&path))
    {
        #[cfg(test)]
        observe(hook, Operation::CreationSync, parent)?;
        sync_directory(parent)?;
        directories.pending.remove(&path);
        if confirm_existing {
            directories.confirmed_committed.insert(path.clone());
        }
    }
    Ok(path)
}
