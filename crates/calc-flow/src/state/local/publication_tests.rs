use std::{path::Path, sync::Arc};

use super::*;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) enum Operation {
    Rename,
    PublishSync,
    CreationSync,
    FileSync,
    StagingSync,
}

pub(crate) type PublicationHook =
    Arc<dyn Fn(Operation, &Path) -> std::io::Result<()> + Send + Sync>;

pub(crate) fn observe(
    hook: Option<&PublicationHook>,
    operation: Operation,
    path: &Path,
) -> Result<()> {
    if let Some(hook) = hook {
        hook(operation, path).map_err(|source| io_error(path, source))?;
    }
    Ok(())
}

#[tokio::test]
async fn single_publication_retry_must_repeat_failed_directory_sync() {
    let directory = tempfile::tempdir().unwrap();
    let backend = LocalStateBackend::new(directory.path()).await.unwrap();
    let key = StateLineageKey::new("publication", &digest("pipeline")).unwrap();
    let lineage = backend.open_local_lineage(&key).await.unwrap();
    let handle = tests::state_handle(&key, Epoch::INITIAL, "one", b"payload");
    lineage.stage_segment(&handle, b"payload").await.unwrap();
    lineage.validate_segment(&handle).await.unwrap();
    *lineage.publication_hook.lock() = Some(Arc::new(|operation, _| {
        if operation == Operation::PublishSync {
            Err(std::io::Error::other("injected sync failure"))
        } else {
            Ok(())
        }
    }));
    assert!(lineage.publish_segment(&handle).await.is_err());
    assert_eq!(lineage.load_segment(&handle).await.unwrap(), b"payload");
    assert!(
        lineage.publish_segment(&handle).await.is_err(),
        "visible bytes do not prove directory durability"
    );
    *lineage.publication_hook.lock() = None;
    lineage.publish_segment(&handle).await.unwrap();
}

pub(crate) fn set_hook(lineage: &LocalStateLineageBackend, hook: Option<PublicationHook>) {
    *lineage.publication_hook.lock() = hook;
}

#[tokio::test]
async fn failed_segment_directory_creation_sync_is_retried() {
    let directory = tempfile::tempdir().unwrap();
    let backend = LocalStateBackend::new(directory.path()).await.unwrap();
    let key = StateLineageKey::new("publication", &digest("pipeline")).unwrap();
    let lineage = backend.open_local_lineage(&key).await.unwrap();
    let handle = tests::state_handle(&key, Epoch::INITIAL, "one", b"payload");
    let failed_parent = directory.path().join("committed");
    set_hook(
        &lineage,
        Some(Arc::new(move |operation, path| {
            if operation == Operation::CreationSync && path == failed_parent {
                Err(std::io::Error::other("injected creation sync failure"))
            } else {
                Ok(())
            }
        })),
    );
    assert!(lineage.stage_segment(&handle, b"payload").await.is_err());
    assert!(
        lineage.stage_segment(&handle, b"payload").await.is_err(),
        "existing directory does not confirm a failed ancestor sync"
    );
    let hook = lineage.publication_hook.lock().clone();
    drop(lineage);
    let lineage = backend.open_local_lineage(&key).await.unwrap();
    set_hook(&lineage, hook);
    assert!(
        lineage.stage_segment(&handle, b"payload").await.is_err(),
        "fresh sessions must confirm committed ancestors too"
    );
    set_hook(&lineage, None);
    lineage.stage_segment(&handle, b"payload").await.unwrap();
    lineage.validate_segment(&handle).await.unwrap();
    lineage.publish_segment(&handle).await.unwrap();
}

pub(crate) async fn publication_lock(
    lineage: &LocalStateLineageBackend,
) -> tokio::sync::MutexGuard<'_, ()> {
    lineage.publication.lock().await
}

fn owner_handle(key: &StateLineageKey, owner: &str, segment: &str) -> StateHandle {
    StateHandle::new(
        owner,
        Epoch::INITIAL,
        segment,
        &format!(
            "committed/{}/{}/1-{}.segment",
            lineage_hash(key),
            digest(owner),
            digest(segment)
        ),
        7,
        &digest("payload"),
    )
    .unwrap()
}

#[tokio::test]
async fn batch_syncs_each_directory_once_after_file_syncs_and_renames() {
    let directory = tempfile::tempdir().unwrap();
    let backend = LocalStateBackend::new(directory.path()).await.unwrap();
    let key = StateLineageKey::new("publication", &digest("pipeline")).unwrap();
    let lineage = backend.open_local_lineage(&key).await.unwrap();
    let handles = [
        owner_handle(&key, "left", "a"),
        owner_handle(&key, "left", "b"),
        owner_handle(&key, "right", "c"),
    ];
    let operations = Arc::new(SyncMutex::new(Vec::new()));
    let observed = operations.clone();
    set_hook(
        &lineage,
        Some(Arc::new(move |operation, path| {
            observed.lock().push((operation, path.to_owned()));
            Ok(())
        })),
    );
    for handle in &handles {
        lineage.stage_segment(handle, b"payload").await.unwrap();
        lineage.validate_segment(handle).await.unwrap();
    }
    let staged = operations.lock().clone();
    assert_eq!(
        staged
            .iter()
            .filter(|(op, _)| *op == Operation::FileSync)
            .count(),
        3
    );
    assert_eq!(
        staged
            .iter()
            .filter(|(op, _)| *op == Operation::StagingSync)
            .count(),
        3
    );
    operations.lock().clear();
    let input = [
        handles[0].clone(),
        handles[1].clone(),
        handles[0].clone(),
        handles[2].clone(),
    ];
    lineage.publish_segments(&input).await.unwrap();
    let published = operations.lock().clone();
    assert_eq!(
        published.iter().map(|(op, _)| *op).collect::<Vec<_>>(),
        [
            Operation::Rename,
            Operation::Rename,
            Operation::Rename,
            Operation::PublishSync,
            Operation::PublishSync
        ]
    );
    assert_ne!(published[3].1, published[4].1);
    drop(lineage);
    let reopened = backend.open_local_lineage(&key).await.unwrap();
    for handle in &handles {
        assert_eq!(reopened.load_segment(handle).await.unwrap(), b"payload");
    }
    reopened.publish_segments(&handles).await.unwrap();
    assert!(
        !directory
            .path()
            .join("staging")
            .join(lineage_hash(&key))
            .exists(),
        "visible confirmation needs no staging subtree"
    );
}

#[tokio::test]
async fn batch_partial_rename_and_persistent_sync_failure_retries() {
    let directory = tempfile::tempdir().unwrap();
    let backend = LocalStateBackend::new(directory.path()).await.unwrap();
    let key = StateLineageKey::new("publication", &digest("pipeline")).unwrap();
    let lineage = backend.open_local_lineage(&key).await.unwrap();
    let handles = [
        owner_handle(&key, "left", "a"),
        owner_handle(&key, "left", "b"),
    ];
    for handle in &handles {
        lineage.stage_segment(handle, b"payload").await.unwrap();
        lineage.validate_segment(handle).await.unwrap();
    }
    let second = directory.path().join(handles[1].relative_path());
    set_hook(
        &lineage,
        Some(Arc::new(move |operation, path| {
            if operation == Operation::Rename && path == second {
                Err(std::io::Error::other("second rename failed"))
            } else {
                Ok(())
            }
        })),
    );
    assert!(lineage.publish_segments(&handles).await.is_err());
    assert_eq!(lineage.load_segment(&handles[0]).await.unwrap(), b"payload");
    assert!(matches!(
        lineage.load_segment(&handles[1]).await,
        Err(CalcFlowError::NotFound { .. })
    ));
    let attempts = Arc::new(SyncMutex::new(0));
    let observed = attempts.clone();
    set_hook(
        &lineage,
        Some(Arc::new(move |operation, _| {
            if operation == Operation::PublishSync {
                *observed.lock() += 1;
                Err(std::io::Error::other("persistent directory sync failure"))
            } else {
                Ok(())
            }
        })),
    );
    for _ in 0..2 {
        assert!(lineage.publish_segments(&handles).await.is_err());
    }
    assert_eq!(*attempts.lock(), 2);
    for handle in &handles {
        assert_eq!(lineage.load_segment(handle).await.unwrap(), b"payload");
    }
    set_hook(&lineage, None);
    lineage.publish_segments(&handles).await.unwrap();
}

#[tokio::test]
async fn second_directory_failure_reconfirms_both_directories() {
    let directory = tempfile::tempdir().unwrap();
    let backend = LocalStateBackend::new(directory.path()).await.unwrap();
    let key = StateLineageKey::new("publication", &digest("pipeline")).unwrap();
    let lineage = backend.open_local_lineage(&key).await.unwrap();
    let handles = [
        owner_handle(&key, "left", "a"),
        owner_handle(&key, "right", "b"),
    ];
    for handle in &handles {
        lineage.stage_segment(handle, b"payload").await.unwrap();
        lineage.validate_segment(handle).await.unwrap();
    }
    let second = directory
        .path()
        .join(handles[1].relative_path())
        .parent()
        .unwrap()
        .to_owned();
    let attempts = Arc::new(SyncMutex::new(Vec::new()));
    let observed = attempts.clone();
    set_hook(
        &lineage,
        Some(Arc::new(move |operation, path| {
            if operation == Operation::PublishSync {
                observed.lock().push(path.to_owned());
                if path == second {
                    return Err(std::io::Error::other("second directory sync failed"));
                }
            }
            Ok(())
        })),
    );
    for _ in 0..2 {
        assert!(lineage.publish_segments(&handles).await.is_err());
    }
    let recorded = attempts.lock().clone();
    assert_eq!(recorded.len(), 4);
    assert_eq!(recorded[..2], recorded[2..]);
    set_hook(&lineage, None);
    lineage.publish_segments(&handles).await.unwrap();
}
