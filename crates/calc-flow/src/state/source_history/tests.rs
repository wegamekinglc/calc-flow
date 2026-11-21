use super::*;
use futures::FutureExt;

fn context(root: &Path, limits: SourceHistoryLimits) -> SourceHistoryContext {
    SourceHistoryContext::new(
        root.join("manifests/lineage"),
        "source",
        Epoch::INITIAL,
        SourceHistorySpec::new("test_files_v1", limits).unwrap(),
        None,
    )
    .unwrap()
}

#[tokio::test]
async fn archive_buffers_refund_and_read_owner_protects_bytes() {
    let root = tempfile::tempdir().unwrap();
    let path = root.path().join("data.json");
    std::fs::write(&path, b"AB").unwrap();
    let context = context(root.path(), SourceHistoryLimits::new(1, 2, 4).unwrap());
    context.archive_file("file-0", &path, 2).await.unwrap();
    context.seal(BTreeMap::new()).unwrap();
    let bytes = context.load("file-0").await.unwrap();
    assert_eq!(bytes.bytes(), b"AB");
    assert!(
        matches!(context.load("file-0").await, Err(CalcFlowError::InvalidArgument { field, .. }) if field == "source_history.max_buffer_bytes")
    );
    drop(bytes);
    let bytes = context.load("file-0").await.unwrap();
    assert!(context.drain().await.is_empty());
    drop(context);
    assert_eq!(bytes.bytes(), b"AB");
    drop(bytes);
}

#[tokio::test]
async fn failed_acquisition_refunds_storage_and_segment_limits() {
    let root = tempfile::tempdir().unwrap();
    let path = root.path().join("data.json");
    std::fs::write(&path, b"AB").unwrap();
    let context = context(root.path(), SourceHistoryLimits::new(1, 4, 4).unwrap());
    assert!(
        matches!(context.archive_file("file-0", &path, 1).await, Err(CalcFlowError::InvalidArgument { field, .. }) if field == "source_history.max_file_bytes")
    );
    context.archive_file("file-0", &path, 2).await.unwrap();
    assert!(
        matches!(context.archive_file("file-1", &path, 2).await, Err(CalcFlowError::InvalidArgument { field, .. }) if field == "source_history.max_segments")
    );
    assert!(
        matches!(context.archive_file("file-0", &path, 2).await, Err(CalcFlowError::InvalidArgument { field, .. }) if field == "source_history.segment_id")
    );
    context.seal(BTreeMap::new()).unwrap();
    assert_eq!(context.manifest().unwrap().segments.len(), 1);
    assert!(context.archive_file("file-2", &path, 2).await.is_err());
    assert!(context.drain().await.is_empty());
}

#[tokio::test]
async fn discovery_and_total_history_bytes_are_bounded() {
    let root = tempfile::tempdir().unwrap();
    let input = root.path().join("input");
    std::fs::create_dir(&input).unwrap();
    std::fs::write(input.join("02.json"), b"CD").unwrap();
    std::fs::write(input.join("01.json"), b"AB").unwrap();
    let one = context(root.path(), SourceHistoryLimits::new(1, 4, 4).unwrap());
    assert!(
        matches!(one.discover_files(&input, "json").await, Err(CalcFlowError::InvalidArgument { field, .. }) if field == "source_history.max_segments")
    );
    assert!(one.drain().await.is_empty());
    let two = context(root.path(), SourceHistoryLimits::new(2, 3, 4).unwrap());
    let paths = two.discover_files(&input, "json").await.unwrap();
    assert_eq!(paths, [input.join("01.json"), input.join("02.json")]);
    two.archive_file("file-0", &paths[0], 2).await.unwrap();
    assert!(
        matches!(two.archive_file("file-1", &paths[1], 2).await, Err(CalcFlowError::InvalidArgument { field, .. }) if field == "source_history.max_bytes")
    );
    assert!(two.drain().await.is_empty());
}

#[tokio::test]
async fn native_v4_history_restore_preserves_handle_and_bytes_after_source_delete() {
    let root = tempfile::tempdir().unwrap();
    let lineage = root.path().join("manifests/lineage");
    let input = root.path().join("input.json");
    std::fs::write(&input, b"AB").unwrap();
    let limits = SourceHistoryLimits::new(1, 2, 4).unwrap();
    let original = SourceHistoryContext::new(
        &lineage,
        "source",
        Epoch::INITIAL,
        SourceHistorySpec::new("test_files_v1", limits).unwrap(),
        None,
    )
    .unwrap();
    let handle = original.archive_file("file-0", &input, 2).await.unwrap();
    original.seal(BTreeMap::new()).unwrap();
    let saved = original.manifest().unwrap();
    drop(original);
    std::fs::remove_file(input).unwrap();

    let restored = SourceHistoryContext::new(
        &lineage,
        "source",
        Epoch::INITIAL,
        SourceHistorySpec::new("test_files_v1", SourceHistoryLimits::new(1, 2, 4).unwrap())
            .unwrap(),
        Some(saved.clone()),
    )
    .unwrap();
    assert_eq!(restored.manifest(), Some(saved));
    assert_eq!(restored.manifest().unwrap().segments[0], handle);
    assert_eq!(restored.load("file-0").await.unwrap().bytes(), b"AB");
}

#[tokio::test]
async fn restored_history_ignores_origins_and_rejects_corrupt_bytes() {
    let root = tempfile::tempdir().unwrap();
    let path = root.path().join("data.json");
    std::fs::write(&path, b"AB").unwrap();
    let original = context(root.path(), SourceHistoryLimits::new(1, 2, 4).unwrap());
    let handle = original.archive_file("file-0", &path, 2).await.unwrap();
    original.seal(BTreeMap::new()).unwrap();
    let saved = original.manifest().unwrap();
    assert!(original.drain().await.is_empty());
    drop(original);
    std::fs::remove_file(&path).unwrap();
    let restored = SourceHistoryContext::new(
        root.path().join("manifests/lineage"),
        "source",
        Epoch::INITIAL.next().unwrap(),
        SourceHistorySpec::new("test_files_v1", SourceHistoryLimits::new(1, 2, 4).unwrap())
            .unwrap(),
        Some(saved.clone()),
    )
    .unwrap();
    assert_eq!(restored.manifest(), Some(saved));
    assert_eq!(restored.load("file-0").await.unwrap().bytes(), b"AB");
    std::fs::write(
        root.path().join("manifests").join(&handle.relative_path),
        b"CD",
    )
    .unwrap();
    assert!(matches!(
        restored.load("file-0").await,
        Err(CalcFlowError::CheckpointMismatch { .. })
    ));
    assert!(restored.load("unknown").await.is_err());
    assert!(restored.drain().await.is_empty());
}

#[tokio::test]
async fn abandoned_history_operation_drains_actual_native_work() {
    let root = tempfile::tempdir().unwrap();
    let context = context(root.path(), SourceHistoryLimits::new(1, 2, 4).unwrap());
    let credit = context.0.ledger.buffer(2).unwrap();
    let (started, entered) = tokio::sync::oneshot::channel();
    let (release, blocked) = std::sync::mpsc::channel();
    let mut request = Box::pin(context.0.work.run(async move {
        tokio::task::spawn_blocking(move || {
            started.send(()).unwrap();
            blocked.recv().unwrap();
            drop(credit);
        })
        .await
        .unwrap();
        Ok(())
    }));
    tokio::select! {
        result = &mut request => panic!("native work ended before release: {result:?}"),
        () = async { entered.await.unwrap(); } => {}
    }
    drop(request);
    let mut drain = Box::pin(context.drain());
    assert!(drain.as_mut().now_or_never().is_none());
    assert!(context.0.ledger.buffer(1).is_err());
    release.send(()).unwrap();
    assert!(drain.await.is_empty());
    assert!(context.0.ledger.buffer(2).is_ok());
    assert!(context.discover_files(root.path(), "json").await.is_err());
}
