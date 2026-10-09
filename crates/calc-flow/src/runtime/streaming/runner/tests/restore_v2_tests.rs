use super::*;

#[path = "restore_v2_tests/terminal_tests.rs"]
mod terminal_tests;
use crate::{CheckpointManifest, Epoch, OperatorStateSnapshot, StateSegment};
use datafusion::arrow::ipc::{
    MetadataVersion,
    writer::{IpcWriteOptions, StreamWriter},
};
use sha2::Digest as _;

const KEY: [u8; 17] = [5, 0, 0, 0, 0, 8, 0, 0, 0, 7, 0, 0, 0, 0, 0, 0, 0];

struct CursorRowsSource {
    inner: BaseRowsSource,
    reopened: Arc<Mutex<Vec<(i64, usize)>>>,
}

#[async_trait]
impl StreamSource for CursorRowsSource {
    async fn open(&mut self, cursor: Option<Cursor>) -> Result<()> {
        self.inner.open(cursor).await?;
        self.reopened
            .lock()
            .push((self.inner.timestamp, self.inner.delivered));
        Ok(())
    }

    async fn next(&mut self) -> Result<Option<SourceEvent>> {
        self.inner.next().await
    }

    async fn close(&mut self) -> Result<()> {
        self.inner.close().await
    }

    fn capabilities(&self) -> SourceCapabilities {
        self.inner.capabilities()
    }
}

#[derive(Default)]
struct Fixture {
    observations: Arc<Mutex<RestoreObservations>>,
    parses: Arc<AtomicUsize>,
    left: Arc<AtomicUsize>,
    right: Arc<AtomicUsize>,
    released: Arc<AtomicBool>,
    reopened: Arc<Mutex<Vec<(i64, usize)>>>,
    rows: Arc<Mutex<Vec<i64>>>,
    wire_kinds: Arc<Mutex<[usize; 3]>>,
}

impl Fixture {
    fn source(&self, permitted: &Arc<AtomicUsize>, count: usize, timestamp: i64) -> SourceBinding {
        SourceBinding::new(
            Box::new(CursorRowsSource {
                inner: BaseRowsSource {
                    permitted: permitted.clone(),
                    released: self.released.clone(),
                    count,
                    delivered: 0,
                    timestamp,
                    watermark_delivered: false,
                },
                reopened: self.reopened.clone(),
            }),
            None,
            0,
        )
        .unwrap()
    }

    fn spec(&self) -> ContinuousJobSpec {
        let mut spec = ac5_job_spec(restore_plan(&self.observations, &self.parses), &self.rows);
        spec.sources = vec![
            NamedSourceBinding {
                binding_id: "left".into(),
                binding: self.source(&self.left, 2, 95),
            },
            NamedSourceBinding {
                binding_id: "right".into(),
                binding: self.source(&self.right, 1, 100),
            },
        ];
        spec
    }

    fn checkpoint(&self, root: &Path) -> CheckpointRuntimeSpec {
        let kinds = self.wire_kinds.clone();
        let observations = self.observations.clone();
        let hook: super::super::super::super::checkpoint_runtime::CheckpointPrepaidReadHook =
            Arc::new(move |bytes, _, credit, _| {
                assert!(credit.size() >= bytes.len());
                kinds.lock()[wire_kind(bytes)] += 1;
                observations.lock().wire.push(Arc::as_ptr(credit) as usize);
                Ok(())
            });
        CheckpointRuntimeSpec::managed(ManagedCheckpointRuntime::new(root).unwrap(), config())
            .unwrap()
            .with_join_preload_read_hook(hook)
    }
}

fn config() -> StreamRuntimeConfig {
    StreamRuntimeConfig {
        checkpoint_interval: StdDuration::from_secs(3_600),
        checkpoint_timeout: StdDuration::from_secs(10),
        ..StreamRuntimeConfig::default()
    }
}

fn wire_kind(bytes: &[u8]) -> usize {
    if bytes.starts_with(b"CFJIDX2\0") {
        0
    } else if bytes.starts_with(b"CFJDIX2\0") {
        1
    } else {
        assert!(bytes.starts_with(b"CFJPAY2\0"));
        2
    }
}

async fn seed_cut(fixture: &Fixture, root: &Path) {
    let mut runner = ContinuousRunner::new();
    let job = runner
        .start_checkpointed(fixture.spec(), fixture.checkpoint(root))
        .await
        .unwrap();
    fixture.left.store(1, Ordering::SeqCst);
    fixture.right.store(1, Ordering::SeqCst);
    wait_retained(&job, 1, 1).await;
    wait_for_join_emission(&job, 1).await;
    job.trigger_checkpoint().await.unwrap();
    fixture.left.store(2, Ordering::SeqCst);
    wait_retained(&job, 2, 1).await;
    wait_for_join_emission(&job, 2).await;
    job.trigger_checkpoint().await.unwrap();
    assert_eq!(job.cancel().await.state, ContinuousJobState::Cancelled);
    drop(job);
    runner.shutdown().await.unwrap();
    assert!(fixture.rows.lock().is_empty());
    fixture.reopened.lock().clear();
}

fn header(magic: [u8; 8], side: u8) -> Vec<u8> {
    let mut bytes = magic.to_vec();
    bytes.extend_from_slice(&2_u32.to_le_bytes());
    bytes.extend_from_slice(&[side, 0, 0, 0]);
    bytes
}

struct Payload {
    side: u8,
    id: u64,
    time: i64,
    digest: [u8; 32],
    segment: StateSegment,
}

fn payload(side: u8, id: u64, time: i64) -> Payload {
    let batch = fixed_row(time);
    let record = &batch.table_payload().unwrap().batches()[0];
    let options = IpcWriteOptions::try_new(8, false, MetadataVersion::V5).unwrap();
    let mut ipc = Vec::new();
    let mut writer =
        StreamWriter::try_new_with_options(&mut ipc, &fixed_schema(), options).unwrap();
    writer.write(record).unwrap();
    writer.finish().unwrap();
    drop(writer);
    let mut bytes = header(*b"CFJPAY2\0", side);
    bytes.extend_from_slice(&1_u64.to_le_bytes());
    bytes.extend_from_slice(&u64::try_from(ipc.len()).unwrap().to_le_bytes());
    bytes.extend_from_slice(&id.to_le_bytes());
    bytes.extend_from_slice(&ipc);
    Payload {
        side,
        id,
        time,
        digest: Sha256::digest(&bytes).into(),
        segment: StateSegment::new(bytes),
    }
}

fn index(payload: &Payload, delta: bool) -> StateSegment {
    let mut bytes = if delta {
        let mut bytes = header(*b"CFJDIX2\0", payload.side);
        bytes.extend_from_slice(&2_u64.to_le_bytes());
        bytes
    } else {
        header(*b"CFJIDX2\0", payload.side)
    };
    bytes.extend_from_slice(&1_u64.to_le_bytes());
    bytes.extend_from_slice(&0_u64.to_le_bytes());
    bytes.extend_from_slice(&payload.id.to_le_bytes());
    bytes.extend_from_slice(&payload.time.to_le_bytes());
    bytes.extend_from_slice(&115_u64.to_le_bytes());
    bytes.extend_from_slice(&payload.digest);
    bytes.extend_from_slice(&0_u64.to_le_bytes());
    bytes.extend_from_slice(&17_u64.to_le_bytes());
    bytes.extend_from_slice(&KEY);
    StateSegment::new(bytes)
}

fn v2_snapshot(mut original: OperatorStateSnapshot) -> OperatorStateSnapshot {
    assert_seed_metadata(&original.inline_metadata);
    let left = payload(0, 0, 95);
    let added = payload(0, 1, 95);
    let right = payload(1, 0, 100);
    let mut segments = BTreeMap::from([
        ("left-base".into(), index(&left, false)),
        ("right-base".into(), index(&right, false)),
        ("left-delta-2".into(), index(&added, true)),
    ]);
    let mut payloads = [&left, &added, &right];
    payloads.sort_by_key(|payload| (payload.side, payload.digest));
    let inventory = payloads
        .iter()
        .map(|payload| {
            let side = if payload.side == 0 { "left" } else { "right" };
            let sha256 = hex::encode(payload.digest);
            segments.insert(format!("{side}-payload-{sha256}"), payload.segment.clone());
            serde_json::json!({
                "side": side, "sha256": sha256, "rows": 1,
                "bytes": payload.segment.bytes().len(),
            })
        })
        .collect::<Vec<_>>();
    original
        .inline_metadata
        .insert("layout_version".into(), 2.into());
    original.inline_metadata.insert(
        "v2_inventory".into(),
        serde_json::json!({
            "codec_version": 2, "base_epoch": 1,
            "deltas": [{ "epoch": 2, "sides": ["left"] }], "payloads": inventory,
        }),
    );
    original.segments = segments;
    assert_eq!(original.segments.len(), 6);
    original
}

fn assert_seed_metadata(metadata: &JsonMap) {
    assert_eq!(metadata["layout_version"], 1);
    assert_eq!(metadata["epoch"], 2);
    assert_eq!(metadata["next_left_row_id"], 2);
    assert_eq!(metadata["next_right_row_id"], 1);
    assert_eq!(metadata["next_output_sequence"], 2);
    assert_eq!(metadata["metrics"]["left"]["retained_rows"], 2);
    assert_eq!(metadata["metrics"]["left"]["retained_bytes"], 230);
    assert_eq!(metadata["metrics"]["right"]["retained_rows"], 1);
    assert_eq!(metadata["metrics"]["right"]["retained_bytes"], 115);
    assert_eq!(metadata["metrics"]["emitted_match_rows"], 2);
    assert_eq!(metadata["ended"], false);
}

async fn transaction(
    root: &Path,
    manifest: &CheckpointManifest,
) -> crate::state::ManifestTransaction {
    let key =
        StateLineageKey::new(manifest.pipeline_name(), manifest.pipeline_fingerprint()).unwrap();
    let backend = LocalStateBackend::new(root.join("state")).await.unwrap();
    let lineage = backend.open_lineage(&key).await.unwrap();
    crate::state::ManifestTransaction::open(
        Arc::from(lineage),
        &key,
        root.join("manifests"),
        config().retained_epochs,
    )
    .await
    .unwrap()
}

async fn publish_v2_cut(seed_root: &Path, selected_root: &Path) {
    let bytes = tokio::fs::read(seed_root.join("manifests/manifest-00000000000000000002.json"))
        .await
        .unwrap();
    let original = CheckpointManifest::from_bytes(&bytes).unwrap();
    assert_eq!(original.epoch(), Epoch::new(2).unwrap());
    assert!(
        original
            .sources()
            .values()
            .all(|source| source.history.is_none())
    );
    assert!(
        original
            .sinks()
            .values()
            .all(|sink| sink.segments.is_empty())
    );
    let source = transaction(seed_root, &original).await;
    let target = transaction(selected_root, &original).await;
    let mut operators = BTreeMap::new();
    let mut working = Vec::new();
    for (id, entry) in original.operators() {
        let mut snapshot = source.load_operator_state(id, entry).await.unwrap();
        if id == "match" {
            snapshot = v2_snapshot(snapshot);
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
    let manifest = copied_manifest(&original, operators);
    target
        .publish(crate::state::PreparedEpochManifest {
            manifest,
            staged_segments: BTreeMap::new(),
        })
        .await
        .unwrap();
    drop(working);
}

fn copied_manifest(
    original: &CheckpointManifest,
    operators: BTreeMap<String, OperatorManifestEntry>,
) -> CheckpointManifest {
    CheckpointManifest::new(CheckpointManifestFields {
        pipeline_name: original.pipeline_name().into(),
        pipeline_fingerprint: original.pipeline_fingerprint().into(),
        runtime_config_hash: original.runtime_config_hash().into(),
        epoch: original.epoch(),
        created_at: original.created_at(),
        recovery_status: original.recovery_status(),
        sources: original.sources().clone(),
        operators,
        sinks: original.sinks().clone(),
        static_inputs: original.static_inputs().clone(),
    })
    .unwrap()
}

fn assert_native_readers(fixture: &Fixture) {
    let observed = fixture.observations.lock();
    assert_eq!(observed.readers.len(), 3);
    assert!(
        observed.readers.iter().all(|(funding, native)| {
            funding.is_some_and(|(identity, paid)| {
                paid > 0
                    && Some(identity) != observed.descriptor
                    && !observed.wire.contains(&identity)
                    && *native
            })
        }),
        "the three actual V2 IPC entries must have paid workspace in owned native work: {:?}",
        observed.readers
    );
    assert_eq!(observed.payloads.len(), 3);
    assert!(
        observed
            .payloads
            .iter()
            .all(|owner| owner.is_some_and(|(_, paid)| paid > 0))
    );
}

#[tokio::test]
async fn test_managed_join_restores_selected_v2_cut_on_native_worker() {
    let directory = tempfile::tempdir().unwrap();
    let seed_root = directory.path().join("seed");
    let selected_root = directory.path().join("selected");
    let fixture = Fixture::default();
    seed_cut(&fixture, &seed_root).await;
    publish_v2_cut(&seed_root, &selected_root).await;

    let mut runner = ContinuousRunner::new();
    let job = runner
        .start_checkpointed(fixture.spec(), fixture.checkpoint(&selected_root))
        .await
        .unwrap();
    wait_retained(&job, 2, 1).await;
    let mut cursors = fixture.reopened.lock().clone();
    cursors.sort_unstable();
    assert_eq!(cursors, [(95, 2), (100, 1)]);
    let status = job.stream_join_status();
    assert_eq!(status["match"].left.retained_bytes, 230);
    assert_eq!(status["match"].right.retained_bytes, 115);
    assert_eq!(status["match"].emitted_match_rows, 2);
    assert!(!fixture.released.load(Ordering::SeqCst));
    fixture.released.store(true, Ordering::SeqCst);
    let outcome = tokio::time::timeout(StdDuration::from_secs(30), job.wait())
        .await
        .unwrap();
    assert_eq!(outcome.state, ContinuousJobState::Completed, "{outcome:?}");
    drop(job);
    runner.shutdown().await.unwrap();
    assert_eq!(fixture.rows.lock().as_slice(), [2]);
    assert_eq!(*fixture.wire_kinds.lock(), [2, 1, 3]);
    assert_eq!(fixture.parses.load(Ordering::SeqCst), 1);
    assert_native_readers(&fixture);
}
