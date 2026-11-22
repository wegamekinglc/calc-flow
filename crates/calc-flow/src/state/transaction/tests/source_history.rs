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
                    segments: handles
                        .into_iter()
                        .map(|handle| crate::state::v4::SourceHistorySegmentHandle {
                            owner: handle.operator_id().into(),
                            epoch: handle.epoch().as_u64(),
                            object_id: handle.segment_id().into(),
                            relative_path: handle.relative_path().into(),
                            byte_len: handle.byte_len(),
                            sha256: handle.sha256().into(),
                        })
                        .collect(),
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
