use super::*;
use crate::{CheckpointManifest, Epoch, OperatorStateSnapshot};
use sha2::Digest as _;

struct CursorRowsSource {
    inner: BaseRowsSource,
    reopened: Arc<Mutex<Vec<(i64, usize)>>>,
    closed: Arc<AtomicUsize>,
    watermarks: Arc<Mutex<Vec<i64>>>,
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
        let event = self.inner.next().await?;
        if let Some(SourceEvent::Watermark(time)) = &event {
            self.watermarks.lock().push(time.as_micros());
        }
        Ok(event)
    }

    async fn close(&mut self) -> Result<()> {
        self.inner.close().await?;
        self.closed.fetch_add(1, Ordering::SeqCst);
        Ok(())
    }

    fn capabilities(&self) -> SourceCapabilities {
        self.inner.capabilities()
    }
}

#[derive(Default)]
struct Fixture {
    observations: Arc<Mutex<RestoreObservations>>,
    parses: Arc<AtomicUsize>,
    writers: Arc<Mutex<Vec<(usize, usize, bool)>>>,
    wire_kinds: Arc<Mutex<[usize; 3]>>,
    left: Arc<AtomicUsize>,
    right: Arc<AtomicUsize>,
    released: Arc<AtomicBool>,
    reopened: Arc<Mutex<Vec<(i64, usize)>>>,
    source_closed: Arc<AtomicUsize>,
    watermarks: Arc<Mutex<Vec<i64>>>,
    sink_closed: Arc<AtomicUsize>,
    rows: Arc<Mutex<Vec<i64>>>,
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
                closed: self.source_closed.clone(),
                watermarks: self.watermarks.clone(),
            }),
            None,
            0,
        )
        .unwrap()
    }

    fn spec(&self, restoring: bool) -> ContinuousJobSpec {
        let restore = restore_plan(&self.observations, &self.parses);
        let plan = if restoring {
            restore
        } else {
            let writer = writer_plan(&self.writers);
            assert_eq!(writer.fingerprint(), restore.fingerprint());
            writer
        };
        let mut spec = ac5_job_spec(plan, &self.rows);
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
        spec.sinks[0].binding = OrdinarySinkBinding::new(Box::new(PairCountSink {
            rows: self.rows.clone(),
            closed: self.sink_closed.clone(),
        }));
        spec
    }

    fn checkpoint(&self, root: &Path) -> CheckpointRuntimeSpec {
        let observations = self.observations.clone();
        let kinds = self.wire_kinds.clone();
        CheckpointRuntimeSpec::managed(ManagedCheckpointRuntime::new(root).unwrap(), config())
            .unwrap()
            .with_join_preload_read_hook(Arc::new(move |bytes, _, credit, _| {
                assert!(credit.size() >= bytes.len());
                kinds.lock()[wire_kind(bytes)] += 1;
                observations.lock().wire.push(Arc::as_ptr(credit) as usize);
                Ok(())
            }))
    }
}

fn config() -> StreamRuntimeConfig {
    StreamRuntimeConfig {
        checkpoint_interval: StdDuration::from_secs(3_600),
        checkpoint_timeout: StdDuration::from_secs(10),
        ..StreamRuntimeConfig::default()
    }
}

fn writer_plan(writers: &Arc<Mutex<Vec<(usize, usize, bool)>>>) -> crate::StreamExecutionPlan {
    let schema = fixed_schema();
    let mut join = StreamJoinOperator::new(
        "match",
        Arc::clone(&schema),
        schema,
        StreamJoinSpec::inner(
            ["key"],
            ["key"],
            "ts",
            "ts",
            JoinTimeBounds::new(StdDuration::ZERO, StdDuration::from_micros(10)).unwrap(),
            JoinStateLimits::new(100_000, 134_217_728, 1_000_000).unwrap(),
        )
        .unwrap(),
    )
    .unwrap();
    let observed = writers.clone();
    join.set_checkpoint_writer_test_hook(Arc::new(move |credit| {
        observed.lock().push((
            std::ptr::from_ref(credit) as usize,
            credit.size(),
            std::thread::current().name() == Some("calc-flow-gather"),
        ));
    }));
    writer_graph(join)
}

fn writer_graph(join: StreamJoinOperator) -> crate::StreamExecutionPlan {
    let output = join.output_ports()[0].schema().unwrap().clone();
    let mut window = WindowSpec::tumbling("left__ts", StdDuration::from_micros(10)).unwrap();
    window.aggregates = vec![AggregateSpec {
        function: AggregateFunction::Count,
        column: "left__ts".into(),
        output: "pairs".into(),
    }];
    let window = WindowAggregateOperator::new("agg", output, window).unwrap();
    PipelineBuilder::new("restore-bases")
        .unwrap()
        .add_node("match", Box::new(join))
        .unwrap()
        .add_node("agg", Box::new(window))
        .unwrap()
        .connect(Edge::new(
            PortEndpoint::new("match", "output").unwrap(),
            PortEndpoint::new("agg", "input").unwrap(),
        ))
        .unwrap()
        .compile_stream(
            &UdfRegistry::new().snapshot(),
            &StreamRequirements::default(),
        )
        .unwrap()
}

async fn seed_cut(fixture: &Fixture, root: &Path) {
    let mut runner = ContinuousRunner::new();
    let job = runner
        .start_checkpointed(fixture.spec(false), fixture.checkpoint(root))
        .await
        .unwrap();
    fixture.left.store(2, Ordering::SeqCst);
    fixture.right.store(1, Ordering::SeqCst);
    wait_retained(&job, 2, 1).await;
    wait_for_join_emission(&job, 2).await;
    assert_eq!(job.trigger_checkpoint().await.unwrap(), Epoch::INITIAL);
    let outcome = job.cancel().await;
    drop(job);
    runner.shutdown().await.unwrap();
    assert_eq!(outcome.state, ContinuousJobState::Cancelled);
    assert!(fixture.rows.lock().is_empty());
    assert!(fixture.watermarks.lock().is_empty());
    assert_eq!(fixture.source_closed.load(Ordering::SeqCst), 2);
    assert_eq!(fixture.sink_closed.load(Ordering::SeqCst), 1);
    assert_writer(fixture);
    fixture.reopened.lock().clear();
}

fn assert_writer(fixture: &Fixture) {
    let writers = fixture.writers.lock();
    assert_eq!(writers.len(), 1);
    assert!(
        writers.iter().all(|(_, paid, native)| *paid > 0 && *native),
        "the actual writer Work entry must own positive workspace on the native worker: {writers:?}"
    );
}

async fn load_natural_cut(root: &Path) -> (CheckpointManifest, OperatorStateSnapshot) {
    let bytes = tokio::fs::read(root.join("manifests/manifest-00000000000000000001.json"))
        .await
        .unwrap();
    let manifest = CheckpointManifest::from_bytes(&bytes).unwrap();
    let key =
        StateLineageKey::new(manifest.pipeline_name(), manifest.pipeline_fingerprint()).unwrap();
    let backend = LocalStateBackend::new(root.join("state")).await.unwrap();
    let lineage = backend.open_lineage(&key).await.unwrap();
    let transaction = crate::state::ManifestTransaction::open(
        Arc::from(lineage),
        &key,
        root.join("manifests"),
        config().retained_epochs,
    )
    .await
    .unwrap();
    let snapshot = transaction
        .load_operator_state("match", &manifest.operators()["match"])
        .await
        .unwrap();
    (manifest, snapshot)
}

fn assert_manifest(manifest: &CheckpointManifest, snapshot: &OperatorStateSnapshot) {
    assert_eq!(manifest.epoch(), Epoch::INITIAL);
    assert_eq!(manifest.sources().len(), 2);
    assert!(manifest.sources().values().all(|source| !source.ended));
    assert!(
        manifest.operators()["match"]
            .progress
            .values()
            .all(|input| {
                input.state == ManifestIngressState::Active && input.watermark.is_none()
            })
    );
    assert_metadata(&snapshot.inline_metadata);
    assert_inventory(snapshot);
}

fn assert_metadata(metadata: &JsonMap) {
    assert_eq!(metadata["layout_version"], 2);
    assert_eq!(metadata["epoch"], 1);
    assert_eq!(metadata["ended"], false);
    assert_eq!(metadata["next_left_row_id"], 2);
    assert_eq!(metadata["next_right_row_id"], 1);
    assert_eq!(metadata["next_output_sequence"], 2);
    assert_eq!(metadata["metrics"]["emitted_match_rows"], 2);
    assert_side_metrics(metadata);
}

fn assert_side_metrics(metadata: &JsonMap) {
    assert_eq!(metadata["metrics"]["left"]["retained_rows"], 2);
    assert_eq!(metadata["metrics"]["left"]["retained_bytes"], 230);
    assert_eq!(metadata["metrics"]["right"]["retained_rows"], 1);
    assert_eq!(metadata["metrics"]["right"]["retained_bytes"], 115);
}

fn assert_inventory(snapshot: &OperatorStateSnapshot) {
    let inventory = &snapshot.inline_metadata["v2_inventory"];
    assert_eq!(inventory["codec_version"], 2);
    assert_eq!(inventory["base_epoch"], 1);
    assert_eq!(inventory["deltas"], serde_json::json!([]));
    assert_eq!(snapshot.segments.len(), 4);
    assert_base(snapshot.segments["left-base"].bytes(), 0, 2);
    assert_base(snapshot.segments["right-base"].bytes(), 1, 1);
    let payloads = inventory["payloads"].as_array().unwrap();
    assert_eq!(payloads.len(), 2);
    assert_payload(snapshot, &payloads[0], "left", 0, &[0, 1]);
    assert_payload(snapshot, &payloads[1], "right", 1, &[0]);
}

fn assert_base(bytes: &[u8], side: u8, rows: u64) {
    assert_eq!(&bytes[..8], b"CFJIDX2\0");
    assert_eq!(&bytes[8..12], &2_u32.to_le_bytes());
    assert_eq!(&bytes[12..16], &[side, 0, 0, 0]);
    assert_eq!(&bytes[16..24], &rows.to_le_bytes());
    assert_eq!(&bytes[24..32], &0_u64.to_le_bytes());
}

fn assert_payload(
    snapshot: &OperatorStateSnapshot,
    entry: &serde_json::Value,
    side_name: &str,
    side: u8,
    ids: &[u64],
) {
    assert_eq!(entry["side"], side_name);
    assert_eq!(entry["rows"], u64::try_from(ids.len()).unwrap());
    let digest = entry["sha256"].as_str().unwrap();
    let bytes = snapshot.segments[&format!("{side_name}-payload-{digest}")].bytes();
    assert_eq!(digest, hex::encode(Sha256::digest(bytes)));
    assert_eq!(entry["bytes"], u64::try_from(bytes.len()).unwrap());
    assert_payload_header(bytes, side, ids);
}

fn assert_payload_header(bytes: &[u8], side: u8, ids: &[u64]) {
    assert_eq!(&bytes[..8], b"CFJPAY2\0");
    assert_eq!(&bytes[8..12], &2_u32.to_le_bytes());
    assert_eq!(&bytes[12..16], &[side, 0, 0, 0]);
    assert_eq!(
        &bytes[16..24],
        &u64::try_from(ids.len()).unwrap().to_le_bytes()
    );
    let ipc_start = 32 + ids.len() * 8;
    assert_eq!(
        &bytes[24..32],
        &u64::try_from(bytes.len() - ipc_start)
            .unwrap()
            .to_le_bytes()
    );
    for (index, id) in ids.iter().enumerate() {
        assert_eq!(&bytes[32 + index * 8..40 + index * 8], &id.to_le_bytes());
    }
    assert_eq!(&bytes[ipc_start..ipc_start + 4], &[255; 4]);
    assert_eq!(&bytes[bytes.len() - 8..], &[255, 255, 255, 255, 0, 0, 0, 0]);
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

fn assert_resumed(job: &super::super::super::ContinuousJob, fixture: &Fixture) {
    let mut cursors = fixture.reopened.lock().clone();
    cursors.sort_unstable();
    assert_eq!(cursors, [(95, 2), (100, 1)]);
    let status = job.stream_join_status();
    assert_eq!(status["match"].left.retained_rows, 2);
    assert_eq!(status["match"].right.retained_rows, 1);
    assert_eq!(status["match"].left.retained_bytes, 230);
    assert_eq!(status["match"].right.retained_bytes, 115);
    assert_eq!(status["match"].emitted_match_rows, 2);
    assert!(!fixture.released.load(Ordering::SeqCst));
    assert!(fixture.watermarks.lock().is_empty());
    assert!(fixture.rows.lock().is_empty());
}

fn assert_readers(fixture: &Fixture) {
    let observed = fixture.observations.lock();
    assert_eq!(*fixture.wire_kinds.lock(), [2, 0, 2]);
    assert_eq!(fixture.parses.load(Ordering::SeqCst), 1);
    assert_eq!(observed.readers.len(), 2);
    assert!(observed.readers.iter().all(|(credit, native)| {
        credit.is_some_and(|(identity, paid)| {
            paid > 0
                && *native
                && Some(identity) != observed.descriptor
                && !observed.wire.contains(&identity)
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

async fn resume_cut(fixture: &Fixture, root: &Path) {
    let mut runner = ContinuousRunner::new();
    let job = runner
        .start_checkpointed(fixture.spec(true), fixture.checkpoint(root))
        .await
        .unwrap();
    wait_retained(&job, 2, 1).await;
    assert_resumed(&job, fixture);
    fixture.released.store(true, Ordering::SeqCst);
    let outcome = tokio::time::timeout(StdDuration::from_secs(30), job.wait())
        .await
        .unwrap();
    drop(job);
    runner.shutdown().await.unwrap();
    assert_eq!(outcome.state, ContinuousJobState::Completed, "{outcome:?}");
    assert_eq!(outcome.cause, TerminalCause::NaturalEnd);
    assert_eq!(fixture.rows.lock().as_slice(), [2]);
    assert_eq!(fixture.watermarks.lock().as_slice(), [120, 120]);
    assert_eq!(fixture.source_closed.load(Ordering::SeqCst), 4);
    assert_eq!(fixture.sink_closed.load(Ordering::SeqCst), 2);
    assert_readers(fixture);
}

#[tokio::test]
async fn test_managed_join_writer_v2_cut_recovers_on_a_fresh_runner() {
    let directory = tempfile::tempdir().unwrap();
    let root = directory.path().join("managed");
    let fixture = Fixture::default();
    seed_cut(&fixture, &root).await;
    let (manifest, snapshot) = load_natural_cut(&root).await;
    assert_manifest(&manifest, &snapshot);
    drop(snapshot);
    drop(manifest);
    resume_cut(&fixture, &root).await;
}
