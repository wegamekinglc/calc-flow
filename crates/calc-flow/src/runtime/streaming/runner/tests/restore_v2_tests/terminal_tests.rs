use super::*;

struct RecoverySink {
    inner: PeriodicCheckpointSink,
    observations: Arc<Mutex<RestoreObservations>>,
    parses: Arc<AtomicUsize>,
    recovered: Arc<Mutex<Vec<(usize, usize)>>>,
}

#[async_trait]
impl TransactionalStreamSink for RecoverySink {
    async fn open(&mut self) -> Result<()> {
        self.inner.open().await
    }

    async fn begin_epoch(&mut self, epoch: Epoch) -> Result<()> {
        self.inner.begin_epoch(epoch).await
    }

    async fn write(&mut self, batch: &Batch) -> Result<()> {
        self.inner.write(batch).await
    }

    async fn pre_commit(&mut self, epoch: Epoch) -> Result<JsonMap> {
        self.inner.pre_commit(epoch).await
    }

    async fn commit(&mut self, epoch: Epoch, state: &JsonMap) -> Result<()> {
        self.inner.commit(epoch, state).await
    }

    async fn abort(&mut self, epoch: Epoch, state: Option<&JsonMap>) -> Result<()> {
        self.inner.abort(epoch, state).await
    }

    async fn recover(&mut self, manifest: &CheckpointManifest) -> Result<()> {
        self.recovered.lock().push((
            self.parses.load(Ordering::SeqCst),
            self.observations.lock().readers.len(),
        ));
        self.inner.recover(manifest).await
    }

    async fn close(&mut self) -> Result<()> {
        self.inner.close().await
    }
}

#[derive(Default)]
struct TerminalFixture {
    inner: Fixture,
    recovered: Arc<Mutex<Vec<(usize, usize)>>>,
}

impl TerminalFixture {
    fn spec(&self) -> ContinuousJobSpec {
        self.spec_with_inner(self.inner.spec())
    }

    fn seed_spec(&self) -> ContinuousJobSpec {
        self.spec_with_inner(self.inner.seed_spec())
    }

    fn spec_with_inner(&self, mut spec: ContinuousJobSpec) -> ContinuousJobSpec {
        spec.sources = vec![
            NamedSourceBinding {
                binding_id: "left".into(),
                binding: self.inner.source(&self.inner.left, 1, 95),
            },
            NamedSourceBinding {
                binding_id: "right".into(),
                binding: self.inner.source(&self.inner.right, 1, 100),
            },
        ];
        spec.sinks[0].binding = OrdinarySinkBinding::new_transactional(Box::new(RecoverySink {
            inner: PeriodicCheckpointSink {
                log: Arc::new(Mutex::new(Vec::new())),
                closed: Arc::new(AtomicUsize::new(0)),
            },
            observations: self.inner.observations.clone(),
            parses: self.inner.parses.clone(),
            recovered: self.recovered.clone(),
        }));
        spec
    }
}

async fn seed_terminal(root: &Path) -> CheckpointManifest {
    let fixture = TerminalFixture::default();
    let mut runner = ContinuousRunner::new();
    let job = runner
        .start_checkpointed(fixture.seed_spec(), fixture.inner.checkpoint(root))
        .await
        .unwrap();
    fixture.inner.left.store(1, Ordering::SeqCst);
    fixture.inner.right.store(1, Ordering::SeqCst);
    wait_retained(&job, 1, 1).await;
    wait_for_join_emission(&job, 1).await;
    assert_eq!(
        job.trigger_checkpoint().await.unwrap(),
        Epoch::new(1).unwrap()
    );
    fixture.inner.released.store(true, Ordering::SeqCst);
    let outcome = tokio::time::timeout(StdDuration::from_secs(30), job.wait())
        .await
        .unwrap();
    drop(job);
    runner.shutdown().await.unwrap();
    assert_eq!(outcome.state, ContinuousJobState::Completed, "{outcome:?}");
    assert_eq!(outcome.cause, TerminalCause::NaturalEnd);
    let bytes = tokio::fs::read(root.join("manifests/manifest-00000000000000000002.json"))
        .await
        .unwrap();
    let manifest = CheckpointManifest::from_bytes(&bytes).unwrap();
    assert_eq!(manifest.epoch(), Epoch::new(2).unwrap());
    assert!(manifest.sources().values().all(|source| source.ended));
    assert!(
        manifest.operators()["match"]
            .progress
            .values()
            .all(|input| input.state == ManifestIngressState::Ended)
    );
    manifest
}

fn removed(payload: &Payload) -> StateSegment {
    let mut bytes = header(*b"CFJDIX2\0", payload.side);
    bytes.extend_from_slice(&2_u64.to_le_bytes());
    bytes.extend_from_slice(&0_u64.to_le_bytes());
    bytes.extend_from_slice(&1_u64.to_le_bytes());
    bytes.extend_from_slice(&payload.id.to_le_bytes());
    bytes.extend_from_slice(&payload.time.to_le_bytes());
    bytes.extend_from_slice(&17_u64.to_le_bytes());
    bytes.extend_from_slice(&KEY);
    StateSegment::new(bytes)
}

fn terminal_snapshot(mut snapshot: OperatorStateSnapshot, ended: bool) -> OperatorStateSnapshot {
    assert_eq!(snapshot.inline_metadata["layout_version"], 1);
    assert_eq!(snapshot.inline_metadata["epoch"], 2);
    assert_eq!(snapshot.inline_metadata["ended"], true);
    assert_eq!(snapshot.inline_metadata["next_left_row_id"], 1);
    assert_eq!(snapshot.inline_metadata["next_right_row_id"], 1);
    assert_eq!(snapshot.inline_metadata["next_output_sequence"], 1);
    assert_eq!(snapshot.inline_metadata["metrics"]["emitted_match_rows"], 1);
    for side in ["left", "right"] {
        assert_eq!(
            snapshot.inline_metadata["metrics"][side]["retained_rows"],
            0
        );
        assert_eq!(
            snapshot.inline_metadata["metrics"][side]["retained_bytes"],
            0
        );
        assert_eq!(snapshot.inline_metadata["metrics"][side]["evicted_rows"], 1);
    }
    let payloads = [payload(0, 0, 95), payload(1, 0, 100)];
    let mut segments = BTreeMap::new();
    let mut inventory = Vec::new();
    for (side, payload) in ["left", "right"].into_iter().zip(&payloads) {
        let digest = hex::encode(payload.digest);
        segments.insert(format!("{side}-base"), index(payload, false));
        segments.insert(format!("{side}-delta-2"), removed(payload));
        segments.insert(format!("{side}-payload-{digest}"), payload.segment.clone());
        inventory.push(serde_json::json!({
            "side": side, "sha256": digest, "rows": 1,
            "bytes": payload.segment.bytes().len(),
        }));
    }
    snapshot
        .inline_metadata
        .insert("layout_version".into(), 2.into());
    snapshot
        .inline_metadata
        .insert("ended".into(), ended.into());
    snapshot.inline_metadata.insert(
        "v2_inventory".into(),
        serde_json::json!({
            "codec_version": 2, "base_epoch": 1,
            "deltas": [{ "epoch": 2, "sides": ["left", "right"] }],
            "payloads": inventory,
        }),
    );
    snapshot.segments = segments;
    assert_eq!(snapshot.segments.len(), 6);
    assert_native_state(&snapshot, ended);
    snapshot
}

fn assert_native_state(snapshot: &OperatorStateSnapshot, ended: bool) {
    let spec = serde_json::from_value(snapshot.inline_metadata["spec"].clone()).unwrap();
    let mut operator =
        StreamJoinOperator::new("match", fixed_schema(), fixed_schema(), spec).unwrap();
    let mut decoded =
        crate::pipeline::OperatorCheckpointCapability::CheckpointedStateful { state_version: 1 }
            .decode_snapshot("match", snapshot.clone())
            .unwrap();
    decoded
        .inline_metadata
        .remove(crate::pipeline::OUTPUT_FRONTIER_METADATA_KEY_V1);
    operator.restore(&decoded).unwrap();
    let status = operator.status();
    assert_eq!(
        (status.left.retained_rows, status.right.retained_rows),
        (0, 0)
    );
    assert_eq!(
        (status.left.retained_bytes, status.right.retained_bytes),
        (0, 0)
    );
    assert_eq!((status.left.ended, status.right.ended), (ended, ended));
    assert_eq!(status.emitted_match_rows, 1);
    assert_eq!(operator.validate_terminal_recovery_state().is_ok(), ended);
}

async fn publish_terminal(
    seed_root: &Path,
    selected_root: &Path,
    original: &CheckpointManifest,
    ended: bool,
) {
    let source = transaction(seed_root, original).await;
    let target = transaction(selected_root, original).await;
    let mut operators = BTreeMap::new();
    let mut working = Vec::new();
    for (id, entry) in original.operators() {
        let mut snapshot = source.load_operator_state(id, entry).await.unwrap();
        if id == "match" {
            snapshot = terminal_snapshot(snapshot, ended);
        }
        let staged = target
            .stage_operator_state(id, original.epoch(), snapshot)
            .await
            .unwrap();
        let mut entry = entry.clone();
        entry.inline_metadata = staged.inline_metadata;
        entry.segments = staged.segments;
        working.push(staged.working);
        operators.insert(id.clone(), entry);
    }
    target
        .publish(crate::state::PreparedEpochManifest {
            manifest: copied_manifest(original, operators),
            staged_segments: BTreeMap::new(),
        })
        .await
        .unwrap();
    drop(working);
}

async fn recover_terminal(
    root: &Path,
    fixture: &TerminalFixture,
) -> std::result::Result<BTreeMap<String, crate::StreamJoinStatus>, String> {
    let mut runner = ContinuousRunner::new();
    let result = match runner
        .start_checkpointed(fixture.spec(), fixture.inner.checkpoint(root))
        .await
    {
        Ok(job) => {
            let outcome = tokio::time::timeout(StdDuration::from_secs(30), job.wait()).await;
            let status = job.stream_join_status();
            drop(job);
            match outcome {
                Ok(outcome) if outcome.state == ContinuousJobState::Completed => Ok(status),
                other => Err(format!("terminal recovery did not complete: {other:?}")),
            }
        }
        Err(error) => Err(format!("{error:?}")),
    };
    runner.shutdown().await.unwrap();
    result
}

fn assert_decoded_before_publication(fixture: &TerminalFixture) {
    assert!(fixture.inner.reopened.lock().is_empty());
    assert_eq!(fixture.inner.parses.load(Ordering::SeqCst), 1);
    assert_eq!(*fixture.inner.wire_kinds.lock(), [2, 2, 2]);
    let observed = fixture.inner.observations.lock();
    assert_eq!(observed.readers.len(), 2);
    assert!(observed.readers.iter().all(|(funding, native)| {
        funding.is_some_and(|(identity, paid)| {
            paid > 0
                && Some(identity) != observed.descriptor
                && !observed.wire.contains(&identity)
                && *native
        })
    }));
    assert_eq!(observed.payloads.len(), 2);
    assert!(
        observed
            .payloads
            .iter()
            .all(|owner| owner.is_some_and(|(_, paid)| paid > 0))
    );
}

#[tokio::test]
async fn test_terminal_managed_v2_checks_native_state_before_sink_recovery() {
    let directory = tempfile::tempdir().unwrap();
    let seed = directory.path().join("seed");
    let valid = directory.path().join("valid");
    let non_ended = directory.path().join("non-ended");
    let manifest = seed_terminal(&seed).await;
    publish_terminal(&seed, &valid, &manifest, true).await;
    publish_terminal(&seed, &non_ended, &manifest, false).await;

    let fixture = TerminalFixture::default();
    let restored = recover_terminal(&valid, &fixture).await;
    let statuses = restored.expect("valid native-ended V2 history must restore before publication");
    assert_decoded_before_publication(&fixture);
    assert_eq!(*fixture.recovered.lock(), [(1, 2)]);
    let status = &statuses["match"];
    assert_eq!(
        (status.left.retained_rows, status.right.retained_rows),
        (0, 0)
    );
    assert_eq!(
        (status.left.retained_bytes, status.right.retained_bytes),
        (0, 0)
    );
    assert!(status.left.ended && status.right.ended);
    assert_eq!(status.emitted_match_rows, 1);

    let fixture = TerminalFixture::default();
    let rejected = recover_terminal(&non_ended, &fixture).await;
    assert!(
        rejected.is_err(),
        "native non-ended V2 state must be refused before overlay"
    );
    assert_decoded_before_publication(&fixture);
    assert!(fixture.recovered.lock().is_empty());
}
