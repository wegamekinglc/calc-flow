use calc_flow::{
    CalcFlowError, Epoch, LocalStateBackend, StateBackend, StateHandle, StateLineageBackend,
    StateLineageKey,
};
use sha2::{Digest, Sha256};
use tempfile::TempDir;

const PIPELINE_FINGERPRINT: &str =
    "0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef";

fn digest(bytes: impl AsRef<[u8]>) -> String {
    hex::encode(Sha256::digest(bytes.as_ref()))
}

fn lineage_key(name: &str) -> StateLineageKey {
    StateLineageKey::new(name, PIPELINE_FINGERPRINT).unwrap()
}

fn lineage_hash(key: &StateLineageKey) -> String {
    digest(format!(
        "{}\0{}",
        key.pipeline_name(),
        key.pipeline_fingerprint()
    ))
}

fn handle(
    key: &StateLineageKey,
    operator_id: &str,
    epoch: Epoch,
    segment_id: &str,
    bytes: &[u8],
) -> StateHandle {
    let relative_path = format!(
        "committed/{}/{}/{}-{}.arrow",
        lineage_hash(key),
        digest(operator_id),
        epoch.as_u64(),
        digest(segment_id)
    );
    StateHandle::new(
        operator_id,
        epoch,
        segment_id,
        &relative_path,
        u64::try_from(bytes.len()).unwrap(),
        &digest(bytes),
    )
    .unwrap()
}

async fn publish(lineage: &dyn StateLineageBackend, handle: &StateHandle, bytes: &[u8]) {
    lineage.stage_segment(handle, bytes).await.unwrap();
    lineage.validate_segment(handle).await.unwrap();
    lineage.publish_segment(handle).await.unwrap();
}

#[tokio::test]
async fn local_backend_owns_one_exclusive_session_per_lineage() {
    let directory = TempDir::new().unwrap();
    let backend = LocalStateBackend::new(directory.path()).await.unwrap();
    let key = lineage_key("orders");

    let first = backend.open_lineage(&key).await.unwrap();
    assert!(matches!(
        backend.open_lineage(&key).await,
        Err(CalcFlowError::Conflict { .. })
    ));
    let second_backend = LocalStateBackend::new(directory.path()).await.unwrap();
    assert!(matches!(
        second_backend.open_lineage(&key).await,
        Err(CalcFlowError::Conflict { .. })
    ));

    let other = backend
        .open_lineage(&lineage_key("other-orders"))
        .await
        .unwrap();
    drop(other);
    drop(first);

    backend.open_lineage(&key).await.unwrap();
}

#[tokio::test]
async fn local_segment_state_machine_validates_before_publish_and_load() {
    let directory = TempDir::new().unwrap();
    let backend = LocalStateBackend::new(directory.path()).await.unwrap();
    let key = lineage_key("orders");
    let lineage = backend.open_lineage(&key).await.unwrap();
    let bytes = b"window-state";
    let handle = handle(&key, "window", Epoch::INITIAL, "delta-0001", bytes);

    assert!(lineage.load_segment(&handle).await.is_err());
    assert!(lineage.publish_segment(&handle).await.is_err());
    assert!(lineage.stage_segment(&handle, b"wrong").await.is_err());

    lineage.stage_segment(&handle, bytes).await.unwrap();
    lineage.stage_segment(&handle, bytes).await.unwrap();
    assert!(lineage.load_segment(&handle).await.is_err());
    assert!(lineage.publish_segment(&handle).await.is_err());

    lineage.validate_segment(&handle).await.unwrap();
    lineage.publish_segment(&handle).await.unwrap();
    lineage.publish_segment(&handle).await.unwrap();
    assert_eq!(lineage.load_segment(&handle).await.unwrap(), bytes);
}

#[tokio::test]
async fn local_load_rechecks_length_and_checksum_before_returning_bytes() {
    let directory = TempDir::new().unwrap();
    let backend = LocalStateBackend::new(directory.path()).await.unwrap();
    let key = lineage_key("orders");
    let lineage = backend.open_lineage(&key).await.unwrap();
    let bytes = b"window-state";
    let handle = handle(&key, "window", Epoch::INITIAL, "delta-0001", bytes);
    publish(lineage.as_ref(), &handle, bytes).await;

    let committed = directory.path().join(handle.relative_path());
    tokio::fs::write(&committed, b"corrupt").await.unwrap();
    assert!(matches!(
        lineage.load_segment(&handle).await,
        Err(CalcFlowError::CheckpointMismatch { .. })
    ));
    tokio::fs::write(&committed, b"tamper-state").await.unwrap();
    assert!(matches!(
        lineage.load_segment(&handle).await,
        Err(CalcFlowError::CheckpointMismatch { .. })
    ));
}

#[tokio::test]
async fn state_larger_than_ten_mib_round_trips_outside_the_manifest() {
    let directory = TempDir::new().unwrap();
    let backend = LocalStateBackend::new(directory.path()).await.unwrap();
    let key = lineage_key("orders");
    let lineage = backend.open_lineage(&key).await.unwrap();
    let bytes = vec![0x5a; 10 * 1024 * 1024 + 1];
    let handle = handle(&key, "window", Epoch::INITIAL, "large", &bytes);

    publish(lineage.as_ref(), &handle, &bytes).await;
    assert_eq!(lineage.load_segment(&handle).await.unwrap(), bytes);
    assert!(!handle.relative_path().contains("orders"));
    assert!(!handle.relative_path().contains("window"));
    assert!(!handle.relative_path().contains("large"));
}

#[tokio::test]
async fn collection_removes_every_unreachable_segment_and_preserves_retained_state() {
    let directory = TempDir::new().unwrap();
    let backend = LocalStateBackend::new(directory.path()).await.unwrap();
    let key = lineage_key("orders");
    let lineage = backend.open_lineage(&key).await.unwrap();
    let retained = handle(&key, "window", Epoch::INITIAL, "retained", b"retained");
    let orphan = handle(&key, "window", Epoch::INITIAL, "orphan", b"orphan");
    let newer = handle(
        &key,
        "window",
        Epoch::INITIAL.next().unwrap(),
        "newer",
        b"newer",
    );

    publish(lineage.as_ref(), &retained, b"retained").await;
    publish(lineage.as_ref(), &orphan, b"orphan").await;
    publish(lineage.as_ref(), &newer, b"newer").await;

    assert_eq!(
        lineage.collect_orphans(&[retained.clone()]).await.unwrap(),
        2
    );
    assert_eq!(lineage.load_segment(&retained).await.unwrap(), b"retained");
    assert!(matches!(
        lineage.load_segment(&newer).await,
        Err(CalcFlowError::NotFound { .. })
    ));
    assert!(matches!(
        lineage.load_segment(&orphan).await,
        Err(CalcFlowError::NotFound { .. })
    ));
}

#[cfg(unix)]
#[tokio::test]
async fn collection_stops_on_a_symbolic_link_without_deleting_valid_state() {
    use std::os::unix::fs::symlink;

    let directory = TempDir::new().unwrap();
    let backend = LocalStateBackend::new(directory.path()).await.unwrap();
    let key = lineage_key("orders");
    let lineage = backend.open_lineage(&key).await.unwrap();
    let retained = handle(&key, "window", Epoch::INITIAL, "retained", b"retained");
    publish(lineage.as_ref(), &retained, b"retained").await;

    let lineage_root = directory.path().join("committed").join(lineage_hash(&key));
    symlink(directory.path(), lineage_root.join("unexpected-link")).unwrap();

    assert!(lineage.collect_orphans(&[retained.clone()]).await.is_err());
    assert_eq!(lineage.load_segment(&retained).await.unwrap(), b"retained");
}

#[tokio::test]
async fn local_batch_rejects_all_conflicts_and_unmanaged_paths_before_publication() {
    assert_staging_preflight_precedes_all_writes().await;
    let directory = TempDir::new().unwrap();
    let backend = LocalStateBackend::new(directory.path()).await.unwrap();
    let key = lineage_key("orders");
    let lineage = backend.open_lineage(&key).await.unwrap();
    let first = handle(&key, "window", Epoch::INITIAL, "first", b"valid");
    lineage.stage_segment(&first, b"valid").await.unwrap();
    lineage.validate_segment(&first).await.unwrap();
    let conflict = StateHandle::new(
        first.operator_id(),
        first.epoch(),
        first.segment_id(),
        first.relative_path(),
        5,
        &digest("other"),
    )
    .unwrap();
    let unmanaged = StateHandle::new(
        "window",
        Epoch::INITIAL,
        "other",
        "committed/unmanaged",
        5,
        &digest("other"),
    )
    .unwrap();
    for invalid in [conflict, unmanaged] {
        assert!(matches!(
            lineage.publish_segments(&[first.clone(), invalid]).await,
            Err(CalcFlowError::InvalidArgument { .. })
        ));
        assert!(matches!(
            lineage.load_segment(&first).await,
            Err(CalcFlowError::NotFound { .. })
        ));
    }
    lineage.publish_segments(&[]).await.unwrap();
    lineage
        .publish_segments(&[first.clone(), first.clone()])
        .await
        .unwrap();
    assert_eq!(lineage.load_segment(&first).await.unwrap(), b"valid");
}

async fn assert_staging_preflight_precedes_all_writes() {
    let directory = TempDir::new().unwrap();
    let backend = LocalStateBackend::new(directory.path()).await.unwrap();
    let key = lineage_key("staging");
    let lineage = backend.open_lineage(&key).await.unwrap();
    let first = handle(&key, "window", Epoch::INITIAL, "first", b"valid");
    let alias = StateHandle::new(
        first.operator_id(),
        first.epoch(),
        first.segment_id(),
        &first.relative_path().replace(".arrow", ".segment"),
        first.byte_len(),
        first.sha256(),
    )
    .unwrap();
    let unmanaged = StateHandle::new(
        "window",
        Epoch::INITIAL,
        "other",
        "committed/unmanaged",
        5,
        &digest("valid"),
    )
    .unwrap();
    let conflict = StateHandle::new(
        first.operator_id(),
        first.epoch(),
        first.segment_id(),
        first.relative_path(),
        first.byte_len(),
        &digest("other"),
    )
    .unwrap();
    for invalid in [&alias, &unmanaged, &conflict] {
        assert!(matches!(
            lineage
                .stage_segments(&[(&first, b"valid"), (invalid, b"valid")])
                .await,
            Err(CalcFlowError::InvalidArgument { .. })
        ));
        assert!(
            !directory
                .path()
                .join("staging")
                .join(lineage_hash(&key))
                .exists()
        );
    }
    assert!(matches!(
        lineage
            .stage_segments(&[(&first, b"valid"), (&first, b"other")])
            .await,
        Err(CalcFlowError::InvalidArgument { .. })
    ));
    assert!(
        !directory
            .path()
            .join("staging")
            .join(lineage_hash(&key))
            .exists()
    );
    lineage.stage_segments(&[]).await.unwrap();
    for bytes in [b"shorter".as_slice(), b"other"] {
        assert!(matches!(
            lineage.stage_segments(&[(&first, bytes)]).await,
            Err(CalcFlowError::CheckpointMismatch { .. })
        ));
    }
    assert!(
        !directory
            .path()
            .join("staging")
            .join(lineage_hash(&key))
            .exists()
    );
    lineage
        .stage_segments(&[(&first, b"valid"), (&first, b"valid")])
        .await
        .unwrap();
    lineage.validate_segment(&first).await.unwrap();
    lineage.publish_segment(&first).await.unwrap();
    assert_eq!(lineage.load_segment(&first).await.unwrap(), b"valid");
}

mod public_backend_contract {
    use super::*;

    pub(super) fn state_handle(
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

    #[tokio::test]
    async fn local_segment_state_machine_checks_visibility_bytes_and_large_payloads() {
        let directory = TempDir::new().unwrap();
        let key = StateLineageKey::new("orders", PIPELINE_FINGERPRINT).unwrap();
        let backend = LocalStateBackend::new(directory.path()).await.unwrap();
        let lineage = backend.open_lineage(&key).await.unwrap();
        let bytes = vec![7_u8; 10 * 1024 * 1024 + 1];
        let handle = state_handle(&key, Epoch::INITIAL, "large-base", &bytes);

        assert!(matches!(
            lineage.load_segment(&handle).await,
            Err(CalcFlowError::NotFound { .. })
        ));
        lineage.stage_segment(&handle, &bytes).await.unwrap();
        assert!(matches!(
            lineage.load_segment(&handle).await,
            Err(CalcFlowError::NotFound { .. })
        ));
        assert!(matches!(
            lineage.publish_segment(&handle).await,
            Err(CalcFlowError::Conflict { .. })
        ));
        lineage.validate_segment(&handle).await.unwrap();
        lineage.publish_segment(&handle).await.unwrap();
        assert_eq!(lineage.load_segment(&handle).await.unwrap(), bytes);

        std::fs::write(directory.path().join(handle.relative_path()), b"truncated").unwrap();
        assert!(matches!(
            lineage.load_segment(&handle).await,
            Err(CalcFlowError::CheckpointMismatch { .. })
        ));
    }

    #[tokio::test]
    async fn orphan_collection_removes_every_unreachable_committed_segment() {
        let directory = TempDir::new().unwrap();
        let key = StateLineageKey::new("orders", PIPELINE_FINGERPRINT).unwrap();
        let backend = LocalStateBackend::new(directory.path()).await.unwrap();
        let lineage = backend.open_lineage(&key).await.unwrap();
        let mut handles = Vec::new();
        for epoch in 1..=3 {
            let bytes = format!("epoch-{epoch}").into_bytes();
            let handle = state_handle(
                &key,
                Epoch::new(epoch).unwrap(),
                &format!("delta-{epoch:04}"),
                &bytes,
            );
            lineage.stage_segment(&handle, &bytes).await.unwrap();
            lineage.validate_segment(&handle).await.unwrap();
            lineage.publish_segment(&handle).await.unwrap();
            handles.push((handle, bytes));
        }

        assert_eq!(
            lineage
                .collect_orphans(&[handles[1].0.clone()])
                .await
                .unwrap(),
            2
        );
        assert!(matches!(
            lineage.load_segment(&handles[0].0).await,
            Err(CalcFlowError::NotFound { .. })
        ));
        assert_eq!(
            lineage.load_segment(&handles[1].0).await.unwrap(),
            handles[1].1
        );
        assert!(matches!(
            lineage.load_segment(&handles[2].0).await,
            Err(CalcFlowError::NotFound { .. })
        ));
    }

    #[tokio::test]
    async fn reopening_lineage_discards_abandoned_staging_segments() {
        let directory = TempDir::new().unwrap();
        let key = StateLineageKey::new("orders", PIPELINE_FINGERPRINT).unwrap();
        let backend = LocalStateBackend::new(directory.path()).await.unwrap();
        let lineage = backend.open_lineage(&key).await.unwrap();
        let epoch = Epoch::new(3).unwrap();
        let abandoned = state_handle(&key, epoch, "delta-0003", b"abandoned");
        lineage
            .stage_segment(&abandoned, b"abandoned")
            .await
            .unwrap();
        drop(lineage);

        let restarted = backend.open_lineage(&key).await.unwrap();
        let replacement = state_handle(&key, epoch, "delta-0003", b"replacement");
        restarted
            .stage_segment(&replacement, b"replacement")
            .await
            .unwrap();
        restarted.validate_segment(&replacement).await.unwrap();
        restarted.publish_segment(&replacement).await.unwrap();
        assert_eq!(
            restarted.load_segment(&replacement).await.unwrap(),
            b"replacement"
        );
    }

    #[tokio::test]
    async fn lineage_lease_and_unexpected_managed_file_type_fail_closed() {
        let directory = TempDir::new().unwrap();
        let key = StateLineageKey::new("orders", PIPELINE_FINGERPRINT).unwrap();
        let first_backend = LocalStateBackend::new(directory.path()).await.unwrap();
        let second_backend = LocalStateBackend::new(directory.path()).await.unwrap();
        let first_lineage = first_backend.open_lineage(&key).await.unwrap();
        assert!(matches!(
            second_backend.open_lineage(&key).await,
            Err(CalcFlowError::Conflict { .. })
        ));
        drop(first_lineage);

        let lineage = second_backend.open_lineage(&key).await.unwrap();
        let lineage_root = directory.path().join("committed").join(lineage_hash(&key));
        std::fs::create_dir_all(&lineage_root).unwrap();
        std::fs::write(lineage_root.join(digest("window")), b"not-a-directory").unwrap();
        assert!(matches!(
            lineage.collect_orphans(&[]).await,
            Err(CalcFlowError::Format { .. })
        ));
    }

    #[cfg(unix)]
    #[tokio::test]
    async fn managed_symlink_fails_closed_before_orphan_collection() {
        let directory = TempDir::new().unwrap();
        let key = StateLineageKey::new("orders", PIPELINE_FINGERPRINT).unwrap();
        let backend = LocalStateBackend::new(directory.path()).await.unwrap();
        let lineage = backend.open_lineage(&key).await.unwrap();
        let lineage_root = directory.path().join("committed").join(lineage_hash(&key));
        std::fs::create_dir_all(&lineage_root).unwrap();
        std::os::unix::fs::symlink(directory.path(), lineage_root.join(digest("window"))).unwrap();

        assert!(matches!(
            lineage.collect_orphans(&[]).await,
            Err(CalcFlowError::Format { .. })
        ));
    }
}
