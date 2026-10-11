use super::{
    BTreeSet, CalcFlowError, LocalStateLineageBackend, ManagedSegmentPaths, Result, StateHandle,
    stage_file, sync_staging_directory, worker,
};

struct StagingEntry<'a> {
    handle: &'a StateHandle,
    bytes: &'a [u8],
    paths: ManagedSegmentPaths,
}

pub(super) async fn stage_segments(
    lineage: &LocalStateLineageBackend,
    segments: &[(&StateHandle, &[u8])],
) -> Result<()> {
    let entries = staging_entries(lineage, segments).await?;
    if entries.is_empty() {
        return Ok(());
    }
    let _guard = lineage.publication.lock().await;
    #[cfg(test)]
    let hook = lineage.publication_hook.lock().clone();
    let mut seen = BTreeSet::new();
    let mut staging_directories = Vec::new();
    for StagingEntry {
        handle,
        bytes,
        paths,
    } in entries
    {
        if seen.insert(paths.staging_parent.clone()) {
            staging_directories.push(paths.staging_parent.clone());
        }
        stage_entry(
            lineage,
            handle,
            bytes,
            paths,
            #[cfg(test)]
            hook.as_ref(),
        )
        .await?;
        lineage.validated.lock().remove(handle);
    }
    finish_staging(
        staging_directories,
        #[cfg(test)]
        hook,
    )
    .await
}

async fn finish_staging(
    staging_directories: Vec<super::PathBuf>,
    #[cfg(test)] hook: Option<super::publication_tests::PublicationHook>,
) -> Result<()> {
    worker(move || {
        for directory in staging_directories {
            sync_staging_directory(
                &directory,
                #[cfg(test)]
                hook.as_ref(),
            )?;
        }
        Ok(())
    })
    .await
}

async fn stage_entry(
    lineage: &LocalStateLineageBackend,
    handle: &StateHandle,
    bytes: &[u8],
    paths: ManagedSegmentPaths,
    #[cfg(test)] hook: Option<&super::publication_tests::PublicationHook>,
) -> Result<()> {
    let owned_handle = handle.clone();
    let bytes = bytes.to_vec();
    let directories = lineage.segment_directories.clone();
    #[cfg(test)]
    let file_hook = hook.cloned();
    worker(move || {
        stage_file(
            &paths,
            &owned_handle,
            &bytes,
            &directories,
            #[cfg(test)]
            file_hook.as_ref(),
        )
    })
    .await
}

async fn staging_entries<'a>(
    lineage: &LocalStateLineageBackend,
    segments: &[(&'a StateHandle, &'a [u8])],
) -> Result<Vec<StagingEntry<'a>>> {
    let unique = super::super::backend::staging_segments(segments).await?;
    let mut staging_paths = BTreeSet::new();
    let mut entries = Vec::with_capacity(unique.len());
    for (handle, bytes) in unique {
        let paths = lineage.managed_paths(handle)?;
        if !staging_paths.insert(paths.staging.clone()) {
            return Err(CalcFlowError::InvalidArgument {
                field: "state_segments".into(),
                message: format!("conflicting state segments at {}", paths.staging.display()),
            });
        }
        entries.push(StagingEntry {
            handle,
            bytes,
            paths,
        });
    }
    Ok(entries)
}
