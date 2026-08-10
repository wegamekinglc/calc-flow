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
