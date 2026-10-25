use super::super::super::{CheckpointFaultMode, CheckpointFaultPoint};
use super::*;

const ORDERED: &str =
    "SELECT key, LAST_VALUE(value ORDER BY value) AS total FROM events GROUP BY key";

async fn ordered() -> SqlOperator {
    let op = SqlOperator::new(NODE, ORDERED, vec!["events".into()], vec![]).unwrap();
    let output = op
        .infer_schema(&BTreeMap::from([("events".into(), logical_schema())]))
        .await
        .unwrap();
    op.with_ports(
        vec![
            Port::with_schema_ref("events", BatchKind::Table, true, Some(logical_schema()))
                .unwrap(),
        ],
        Port::with_schema_ref("output", BatchKind::Table, true, Some(output)).unwrap(),
    )
    .unwrap()
}

async fn ordered_spec(source: &str, probe: &Arc<Probe>) -> ContinuousJobSpec {
    let mut spec = spec(source, false, probe).await;
    let plan = PipelineBuilder::new("ordered-last-managed-recovery")
        .unwrap()
        .add_node(NODE, Box::new(ordered().await))
        .unwrap()
        .compile_stream(
            &UdfRegistry::new().snapshot(),
            &StreamRequirements {
                delivery: BTreeMap::from([(
                    "output".into(),
                    crate::DeliveryGuarantee::ExactlyOnce,
                )]),
            },
        )
        .unwrap();
    spec.context = StreamJobContext::new(
        441,
        plan.fingerprint(),
        JsonMap::new(),
        None,
        CancellationToken::new(),
    );
    spec.plan = plan;
    spec
}

async fn seed_checkpoint(root: &Path, source: &str) -> Vec<u8> {
    let probe = Arc::new(Probe::default());
    let mut runner = ContinuousRunner::new();
    let job = runner
        .start_checkpointed(ordered_spec(source, &probe).await, checkpoint(root))
        .await
        .unwrap();
    tokio::time::timeout(DEADLINE, async {
        while !probe.tail.load(Ordering::SeqCst) || probe.writes.lock().is_empty() {
            tokio::task::yield_now().await;
        }
    })
    .await
    .unwrap();
    assert_eq!(job.trigger_checkpoint().await.unwrap(), Epoch::INITIAL);
    let _ = job.cancel().await;
    drop(job);
    runner.shutdown().await.unwrap();
    assert_eq!(runner.registry_counts(), (0, 0));
    tokio::fs::read(root.join("manifests/manifest-00000000000000000001.json"))
        .await
        .unwrap()
}

async fn assert_saved_winner(root: &Path, saved: &[u8]) {
    let current = tokio::fs::read(root.join("manifests/manifest-00000000000000000001.json"))
        .await
        .unwrap();
    assert_eq!(current, saved);
    let manifest = crate::CheckpointManifest::from_bytes(&current).unwrap();
    assert_eq!(
        manifest.operators()[NODE].inline_metadata["state_layout"],
        serde_json::json!(5)
    );
    let mut op = ordered().await;
    op.restore(&snapshot(root, &manifest).await).unwrap();
    let context_job = StreamJobContext::new(
        441,
        "restore-check",
        JsonMap::new(),
        None,
        CancellationToken::new(),
    );
    let ctx = StreamOperatorContext::new(&context_job, NODE, None);
    let mut output = crate::EdgeCollector::new(op.output_ports().to_vec());
    let batch = Batch::table(
        vec![RecordBatch::new_empty(logical_schema())],
        BatchMetadata::default(),
    )
    .unwrap();
    op.process_data("events", batch, &ctx, &mut output)
        .await
        .unwrap();
    let emitted = output.drain("output");
    let record = &emitted[0]
        .as_data()
        .unwrap()
        .table_payload()
        .unwrap()
        .batches()[0];
    assert_eq!(record.num_rows(), 1);
    assert_eq!(
        record
            .column(1)
            .as_any()
            .downcast_ref::<Int64Array>()
            .unwrap()
            .value(0),
        5
    );
}

async fn storage_failure(
    point: CheckpointFaultPoint,
    expected_ack: usize,
    expected_precommit: usize,
) {
    let directory = tempfile::tempdir().unwrap();
    let source = format!("ordered-storage-{point:?}");
    let saved = seed_checkpoint(directory.path(), &source).await;
    let declaration = ordered().await;
    let guard = declaration
        .on_prepared_checkpoint_for_test(&expected_normalized_metadata(&source), |_| Ok(()));
    let probe = Arc::new(Probe::default());
    let mut runner = ContinuousRunner::new();
    let job = runner
        .start_checkpointed(
            ordered_spec(&source, &probe).await,
            checkpoint(directory.path()).with_fault(point, CheckpointFaultMode::Io),
        )
        .await
        .unwrap();
    assert_eq!(probe.sink_recovers.load(Ordering::SeqCst), 1);
    assert!(job.trigger_checkpoint().await.is_err());
    assert_eq!(job.wait().await.state, ContinuousJobState::Failed);
    drop(job);
    runner.shutdown().await.unwrap();
    assert_eq!(runner.registry_counts(), (0, 0));
    assert_eq!(guard.acks(), expected_ack);
    assert_eq!(
        probe.sink_pre_commits.load(Ordering::SeqCst),
        expected_precommit
    );
    // The only commit replays the previously durable epoch during sink recovery.
    assert_eq!(probe.sink_commits.load(Ordering::SeqCst), 1);
    assert_eq!(probe.source_closes.load(Ordering::SeqCst), 1);
    assert_eq!(probe.sink_closes.load(Ordering::SeqCst), 1);
    assert!(
        !directory
            .path()
            .join("manifests/manifest-00000000000000000002.json")
            .exists()
    );
    assert_saved_winner(directory.path(), &saved).await;
}

#[tokio::test]
async fn test_ordered_last_managed_state_storage_failure_preserves_manifest_without_ack() {
    storage_failure(CheckpointFaultPoint::StateStage, 0, 0).await;
}

#[tokio::test]
async fn test_ordered_last_managed_manifest_storage_failure_preserves_prior_epoch() {
    storage_failure(CheckpointFaultPoint::ManifestWrite, 1, 1).await;
}

#[tokio::test]
async fn test_ordered_last_managed_capture_prepare_failure_never_installs_or_acks() {
    let directory = tempfile::tempdir().unwrap();
    let source = "ordered-capture-prepare-failure";
    let saved = seed_checkpoint(directory.path(), source).await;
    let declaration = ordered().await;
    let guard = declaration.on_prepared_checkpoint_for_test(
        &expected_normalized_metadata(source),
        |candidate| {
            assert_eq!(candidate.layout, 5);
            assert!(candidate.fee.upgrade().is_some());
            Err(CalcFlowError::Internal {
                message: "injected layout-5 capture prepare failure".into(),
            })
        },
    );
    let probe = Arc::new(Probe::default());
    let mut runner = ContinuousRunner::new();
    let job = runner
        .start_checkpointed(
            ordered_spec(source, &probe).await,
            checkpoint(directory.path()),
        )
        .await
        .unwrap();
    assert!(job.trigger_checkpoint().await.is_err());
    assert_eq!(job.wait().await.state, ContinuousJobState::Failed);
    drop(job);
    runner.shutdown().await.unwrap();
    assert_eq!(runner.registry_counts(), (0, 0));
    assert_eq!((guard.installs(), guard.acks(), guard.exits()), (0, 0, 1));
    assert_eq!(probe.sink_pre_commits.load(Ordering::SeqCst), 0);
    assert_eq!(probe.sink_commits.load(Ordering::SeqCst), 1);
    assert_saved_winner(directory.path(), &saved).await;
}
