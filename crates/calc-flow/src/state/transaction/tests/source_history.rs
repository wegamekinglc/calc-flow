use super::*;
use crate::{SourceHistoryManifestEntry, SourceManifestEntry, SourceWatermarkManifestState};
use serde_json::json;

async fn open(root: &Path) -> ManifestTransaction {
    let backend = LocalStateBackend::new(root.join("state")).await.unwrap();
    let key = StateLineageKey::new("orders", PIPELINE_FINGERPRINT).unwrap();
    ManifestTransaction::open(
        Arc::from(backend.open_lineage(&key).await.unwrap()),
        &key,
        root.join("manifests"),
        1,
    )
    .await
    .unwrap()
}

fn prepared_identity() -> PreparedManifestIdentity {
    let mut value = identity();
    value.source_ids.insert("prices".into());
    value
}

fn with_history(epoch: Epoch, handles: Vec<StateHandle>, ended: bool) -> CheckpointManifest {
    CheckpointManifest::new(CheckpointManifestFields {
        pipeline_name: "orders".into(),
        pipeline_fingerprint: PIPELINE_FINGERPRINT.into(),
        runtime_config_hash: RUNTIME_CONFIG_HASH.into(),
        epoch,
        created_at: Utc.with_ymd_and_hms(2026, 10, 4, 0, 0, 0).unwrap(),
        recovery_status: RecoveryStatus::Final,
        sources: BTreeMap::from([(
            "prices".into(),
            SourceManifestEntry {
                cursor: None,
                identity_hash: PIPELINE_FINGERPRINT.into(),
                sequence: 3,
                ended,
                watermark_policy: SourceWatermarkManifestState::Disabled { idle: false },
                history: Some(SourceHistoryManifestEntry {
                    format_version: crate::SOURCE_HISTORY_FORMAT_VERSION,
                    contract: "file_snapshot_v1".into(),
                    inline_metadata: BTreeMap::from([("files".into(), json!(["prices.json"]))]),
                    segments: handles,
                }),
            },
        )]),
        operators: BTreeMap::new(),
        sinks: BTreeMap::new(),
        static_inputs: BTreeMap::new(),
    })
    .unwrap()
}

async fn publish(transaction: &ManifestTransaction, manifest: CheckpointManifest) {
    assert!(matches!(
        transaction
            .publish(PreparedEpochManifest {
                manifest,
                staged_segments: BTreeMap::new(),
            })
            .await
            .unwrap(),
        ManifestPublication::Durable
    ));
}

async fn stage(transaction: &ManifestTransaction, epoch: Epoch, bytes: &[u8]) -> StateHandle {
    transaction
        .stage_operator_state(
            "prices",
            epoch,
            snapshot_with_segments(&[("file-0", bytes)]),
        )
        .await
        .unwrap()
        .segments
        .remove(0)
}

#[tokio::test]
async fn source_history_staging_owner_protects_unpublished_bytes() {
    let directory = TempDir::new().unwrap();
    let transaction = open(directory.path()).await;
    let staged = Arc::new(
        transaction
            .stage_operator_state(
                "prices",
                Epoch::INITIAL,
                snapshot_with_segments(&[("file-0", b"original")]),
            )
            .await
            .unwrap(),
    );
    let path = directory
        .path()
        .join("state")
        .join(staged.segments[0].relative_path());
    let owner = staged.clone();
    drop(staged);
    let report = transaction.retain(&identity(), None).await.unwrap();
    assert_eq!(report.removed_orphan_segments, 0);
    assert_eq!(std::fs::read(&path).unwrap(), b"original");
    drop(owner);
    let report = transaction.retain(&identity(), None).await.unwrap();
    assert_eq!(report.removed_orphan_segments, 1);
    assert!(!path.exists());
}

#[tokio::test]
async fn source_history_independent_staging_owners_share_one_protection() {
    let directory = TempDir::new().unwrap();
    let transaction = open(directory.path()).await;
    let snapshot = || snapshot_with_segments(&[("file-0", b"original")]);
    let first = transaction
        .stage_operator_state("prices", Epoch::INITIAL, snapshot())
        .await
        .unwrap();
    let epoch = Epoch::INITIAL.next().unwrap();
    let second = transaction
        .stage_operator_state("prices", epoch, snapshot())
        .await
        .unwrap();
    assert_eq!(first.segments, second.segments);
    drop(first);
    assert_eq!(
        transaction
            .retain(&identity(), None)
            .await
            .unwrap()
            .removed_orphan_segments,
        0
    );
    let old = second.segments[0].clone();
    drop(second);
    assert_eq!(
        transaction
            .retain(&identity(), None)
            .await
            .unwrap()
            .removed_orphan_segments,
        1
    );
    let recreated = transaction
        .stage_operator_state("prices", epoch.next().unwrap(), snapshot())
        .await
        .unwrap();
    assert_ne!(recreated.segments[0].relative_path(), old.relative_path());
    assert_eq!(
        transaction
            .lineage
            .load_segment(&recreated.segments[0])
            .await
            .unwrap(),
        b"original"
    );
}

#[tokio::test]
async fn source_history_selection_validates_all_retained_terminal_bytes() {
    let directory = TempDir::new().unwrap();
    let transaction = open(directory.path()).await;
    let first = stage(&transaction, Epoch::INITIAL, b"original").await;
    publish(
        &transaction,
        with_history(Epoch::INITIAL, vec![first.clone()], false),
    )
    .await;
    let second_epoch = Epoch::INITIAL.next().unwrap();
    let second = stage(&transaction, second_epoch, b"latest!!").await;
    publish(&transaction, with_history(second_epoch, vec![second], true)).await;
    drop(transaction);
    std::fs::write(
        directory.path().join("state").join(first.relative_path()),
        b"tampered",
    )
    .unwrap();
    let recovered = open(directory.path()).await;
    let result = recovered.select_latest(&prepared_identity()).await;
    assert!(
        matches!(result, Err(CalcFlowError::CheckpointMismatch { .. })),
        "cold recovery must validate an older referenced history before selecting a terminal epoch"
    );
    assert!(
        directory
            .path()
            .join("state")
            .join(first.relative_path())
            .exists()
    );
}

#[tokio::test]
async fn source_history_retention_keeps_carry_and_inflight_references() {
    let directory = TempDir::new().unwrap();
    let transaction = open(directory.path()).await;
    let first_epoch = Epoch::INITIAL;
    let second_epoch = first_epoch.next().unwrap();
    let third_epoch = second_epoch.next().unwrap();
    let fourth_epoch = third_epoch.next().unwrap();
    let first = stage(&transaction, first_epoch, b"original").await;
    for epoch in [first_epoch, second_epoch] {
        publish(
            &transaction,
            with_history(epoch, vec![first.clone()], false),
        )
        .await;
    }
    let second = stage(&transaction, third_epoch, b"latest!!").await;
    publish(
        &transaction,
        with_history(third_epoch, vec![second.clone()], true),
    )
    .await;
    let in_flight = with_history(fourth_epoch, vec![first.clone()], false);
    let report = transaction
        .retain(&prepared_identity(), Some(&in_flight))
        .await
        .unwrap();
    assert_eq!(report.removed_manifests, 2);
    assert_eq!(report.removed_orphan_segments, 0);
    for handle in [&first, &second] {
        assert!(
            directory
                .path()
                .join("state")
                .join(handle.relative_path())
                .exists()
        );
    }
    let report = transaction
        .retain(&prepared_identity(), None)
        .await
        .unwrap();
    assert_eq!(report.removed_orphan_segments, 1);
    assert!(
        !directory
            .path()
            .join("state")
            .join(first.relative_path())
            .exists()
    );
    assert!(
        directory
            .path()
            .join("state")
            .join(second.relative_path())
            .exists()
    );
    drop(transaction);
    let recovered = open(directory.path()).await;
    let selected = recovered
        .select_latest(&prepared_identity())
        .await
        .unwrap()
        .unwrap();
    assert_eq!(selected.manifest.epoch(), third_epoch);
}
