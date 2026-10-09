use super::*;
use crate::{LocalStateBackend, OperatorStateSnapshot, StateSegment};

fn identity() -> PreparedManifestIdentity {
    PreparedManifestIdentity {
        pipeline_name: "working-state".into(),
        pipeline_fingerprint: "0123456789abcdef".repeat(4),
        runtime_config_hash: "abcdef0123456789".repeat(4),
        source_ids: BTreeSet::new(),
        operator_ids: BTreeSet::from(["operator".into()]),
        sink_ids: BTreeSet::new(),
        static_inputs: BTreeMap::new(),
    }
}

async fn stage_ack(transaction: &ManifestTransaction, epoch: Epoch) -> OperatorCheckpointAck {
    let staged = transaction
        .stage_operator_state(
            "operator",
            epoch,
            OperatorStateSnapshot {
                inline_metadata: BTreeMap::new(),
                segments: BTreeMap::from([("state".into(), StateSegment::new(b"state".to_vec()))]),
            },
        )
        .await
        .unwrap();
    OperatorCheckpointAck {
        capture_credit: None,
        node_id: "operator".into(),
        epoch,
        state: OperatorManifestEntry {
            progress: BTreeMap::new(),
            inline_metadata: staged.inline_metadata,
            segments: staged.segments,
        },
        working: staged.working,
    }
}

async fn publish_assembly(
    transaction: &ManifestTransaction,
    identity: &PreparedManifestIdentity,
    assembly: &mut EpochManifestAssembly,
    epoch: Epoch,
) {
    let manifest = CheckpointManifest::new(CheckpointManifestFields {
        pipeline_name: identity.pipeline_name.clone(),
        pipeline_fingerprint: identity.pipeline_fingerprint.clone(),
        runtime_config_hash: identity.runtime_config_hash.clone(),
        epoch,
        created_at: Utc::now(),
        recovery_status: RecoveryStatus::Final,
        sources: BTreeMap::new(),
        operators: assembly.operators.clone(),
        sinks: BTreeMap::new(),
        static_inputs: BTreeMap::new(),
    })
    .unwrap();
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
    assembly.complete(epoch).unwrap();
}

async fn open_transaction(
    root: &std::path::Path,
    identity: &PreparedManifestIdentity,
) -> ManifestTransaction {
    let backend = LocalStateBackend::new(root.join("state")).await.unwrap();
    let key =
        StateLineageKey::new(&identity.pipeline_name, &identity.pipeline_fingerprint).unwrap();
    ManifestTransaction::open(
        Arc::from(backend.open_lineage(&key).await.unwrap()),
        &key,
        root.join("manifests"),
        1,
    )
    .await
    .unwrap()
}

async fn check_ack_lifetime(published: bool) {
    let directory = tempfile::tempdir().unwrap();
    let identity = identity();
    let transaction = open_transaction(directory.path(), &identity).await;
    let epoch = Epoch::INITIAL;
    let ack = stage_ack(&transaction, epoch).await;
    let path = directory
        .path()
        .join("state")
        .join(ack.state.segments[0].relative_path());
    let (sender, mut receiver) = mpsc::channel(1);
    sender.send(ack).await.unwrap();
    assert_eq!(
        transaction
            .retain(&identity, None)
            .await
            .unwrap()
            .removed_orphan_segments,
        0
    );
    let cancellation = CancellationToken::new();
    let (coordinator, mut events, task) = spawn_checkpoint_coordinator(
        ParticipantSet {
            sources: BTreeSet::from(["source".into()]),
            operators: identity.operator_ids.clone(),
            sinks: BTreeSet::from(["sink".into()]),
        },
        epoch,
        8,
        Duration::from_secs(30),
        cancellation.clone(),
    )
    .unwrap();
    coordinator
        .request(CheckpointRequest::Periodic)
        .await
        .unwrap();
    assert_eq!(events.recv().await, Some(CheckpointEvent::Started(epoch)));
    coordinator
        .ack(CheckpointAck::source("source", epoch, "source-cut"))
        .await
        .unwrap();
    assert_eq!(
        events.recv().await,
        Some(CheckpointEvent::PhaseAdvanced(
            epoch,
            CheckpointPhase::SourcesCut
        ))
    );
    let mut assembly = EpochManifestAssembly::default();
    assembly.start(epoch, false).unwrap();
    accept_operator_ack(
        receiver.recv().await.unwrap(),
        &coordinator,
        &mut assembly,
        &CheckpointStatusHandle::new(&identity, None),
    )
    .await
    .unwrap();
    assert_eq!(
        events.recv().await,
        Some(CheckpointEvent::PhaseAdvanced(
            epoch,
            CheckpointPhase::OperatorsSnapshotted
        ))
    );
    assert_eq!(
        transaction
            .retain(&identity, None)
            .await
            .unwrap()
            .removed_orphan_segments,
        0
    );
    assert_eq!(std::fs::read(&path).unwrap(), b"state");
    if published {
        publish_assembly(&transaction, &identity, &mut assembly, epoch).await;
    }
    drop(assembly);
    assert_eq!(
        transaction
            .retain(&identity, None)
            .await
            .unwrap()
            .removed_orphan_segments,
        usize::from(!published)
    );
    assert_eq!(path.exists(), published);
    cancellation.cancel();
    task.await.unwrap().unwrap();
}

#[tokio::test]
async fn checkpoint_ack_holds_segments_until_manifest_or_assembly_release() {
    for published in [false, true] {
        check_ack_lifetime(published).await;
    }
}
