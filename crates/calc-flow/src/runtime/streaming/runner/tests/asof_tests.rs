use super::{
    ContinuousJobSpec, ContinuousJobState, ContinuousRunner, Cursor, M2DeliveryMode,
    NamedSinkBinding, NamedSourceBinding, OrdinarySinkBinding, OrdinaryStreamSink, SourceBinding,
    SourceCapabilities, SourceEvent, StreamSource, TransactionalStreamSink,
};
use crate::{
    AsofJoinSide, AsofStateLimits, Batch, BatchMetadata, CalcFlowError, CancellationToken,
    CheckpointManifest, DeliveryGuarantee, EdgeBudget, Epoch, EventTime, JsonMap, PipelineBuilder,
    Result, StreamAsofJoinOperator, StreamAsofJoinSpec, StreamJobContext, StreamRequirements,
    StreamRuntimeConfig, UdfRegistry,
    runtime::streaming::{
        checkpoint::ManagedCheckpointRuntime,
        checkpoint_runtime::CheckpointRuntimeSpec,
        progress::{NativeWatermarkCapability, WatermarkPolicy},
        test_seams::{CheckpointFaultMode, CheckpointFaultPoint},
    },
};
use async_trait::async_trait;
use datafusion::arrow::{
    array::{Array, Int64Array, StringArray, TimestampMicrosecondArray, UInt64Array},
    datatypes::{DataType, Field, Schema, SchemaRef, TimeUnit},
    record_batch::RecordBatch,
};
use parking_lot::Mutex;
use std::{
    collections::{BTreeMap, BTreeSet},
    path::Path,
    sync::{
        Arc,
        atomic::{AtomicBool, Ordering},
    },
    time::Duration,
};

fn schema() -> SchemaRef {
    Arc::new(Schema::new(vec![
        Field::new("key", DataType::Utf8, false),
        Field::new(
            "ts",
            DataType::Timestamp(TimeUnit::Microsecond, Some("UTC".into())),
            false,
        ),
        Field::new("sequence", DataType::UInt64, false),
        Field::new("value", DataType::Int64, false),
    ]))
}

fn row(time: i64, sequence: u64) -> Batch {
    Batch::table(
        vec![
            RecordBatch::try_new(
                schema(),
                vec![
                    Arc::new(StringArray::from(vec!["a"])),
                    Arc::new(TimestampMicrosecondArray::from(vec![time]).with_timezone("UTC")),
                    Arc::new(UInt64Array::from(vec![sequence])),
                    Arc::new(Int64Array::from(vec![time])),
                ],
            )
            .unwrap(),
        ],
        BatchMetadata::default(),
    )
    .unwrap()
}

fn operator() -> StreamAsofJoinOperator {
    let side = |prefix: &str| {
        AsofJoinSide::new(
            vec!["key".into()],
            "ts".into(),
            vec!["sequence".into()],
            prefix.into(),
        )
        .unwrap()
    };
    StreamAsofJoinOperator::new(
        "match",
        schema(),
        schema(),
        StreamAsofJoinSpec::new(
            side("left"),
            side("right"),
            Duration::from_micros(10),
            AsofStateLimits::new(1_000, 16 * 1024 * 1024).unwrap(),
        )
        .unwrap(),
    )
    .unwrap()
}

#[derive(Clone, Debug, Eq, Ord, PartialEq, PartialOrd, serde::Serialize, serde::Deserialize)]
struct Match {
    left_time: i64,
    left_sequence: u64,
    right_time: Option<i64>,
    right_sequence: Option<u64>,
}

fn records(batch: &Batch) -> Vec<Match> {
    batch
        .table_payload()
        .unwrap()
        .batches()
        .iter()
        .flat_map(|batch| {
            let time = |name: &str| {
                batch
                    .column_by_name(name)
                    .unwrap()
                    .as_any()
                    .downcast_ref::<TimestampMicrosecondArray>()
                    .unwrap()
            };
            let sequence = |name: &str| {
                batch
                    .column_by_name(name)
                    .unwrap()
                    .as_any()
                    .downcast_ref::<UInt64Array>()
                    .unwrap()
            };
            let left = time("left__ts");
            let right = time("right__ts");
            let left_sequence = sequence("left__sequence");
            let right_sequence = sequence("right__sequence");
            (0..batch.num_rows())
                .map(|index| Match {
                    left_time: left.value(index),
                    left_sequence: left_sequence.value(index),
                    right_time: (!right.is_null(index)).then(|| right.value(index)),
                    right_sequence: (!right_sequence.is_null(index))
                        .then(|| right_sequence.value(index)),
                })
                .collect::<Vec<_>>()
        })
        .collect()
}

#[derive(Default)]
struct VisibleState {
    records: Vec<Match>,
    committed_epochs: BTreeSet<u64>,
    writes: usize,
    opens: usize,
}

#[derive(Clone, Default)]
struct SourceProbe {
    paused: Arc<AtomicBool>,
    opens: Arc<Mutex<Vec<usize>>>,
}

struct ScriptedSource {
    times: Vec<i64>,
    next: usize,
    watermark_sent: bool,
    native_watermarks: bool,
    pause_after: Option<usize>,
    hold_open: bool,
    probe: SourceProbe,
}

#[async_trait]
impl StreamSource for ScriptedSource {
    async fn open(&mut self, cursor: Option<Cursor>) -> Result<()> {
        if let Some(cursor) = cursor {
            let encoded: [u8; 8] = cursor.order().try_into().unwrap();
            self.next = usize::try_from(u64::from_be_bytes(encoded)).unwrap();
        }
        self.probe.opens.lock().push(self.next);
        Ok(())
    }

    async fn next(&mut self) -> Result<Option<SourceEvent>> {
        if self.pause_after == Some(self.next) {
            self.probe.paused.store(true, Ordering::SeqCst);
            std::future::pending::<()>().await;
        }
        if let Some(time) = self.times.get(self.next).copied() {
            self.next += 1;
            let sequence = u64::try_from(self.next).unwrap();
            return Ok(Some(SourceEvent::Data {
                batch: row(time, sequence),
                cursor: Cursor::unbound(sequence.to_be_bytes().to_vec(), JsonMap::new()).unwrap(),
            }));
        }
        if self.native_watermarks && !self.watermark_sent {
            self.watermark_sent = true;
            return Ok(Some(SourceEvent::Watermark(EventTime::from_micros(1_000))));
        }
        if self.hold_open {
            self.probe.paused.store(true, Ordering::SeqCst);
            std::future::pending::<()>().await;
        }
        Ok(None)
    }

    async fn close(&mut self) -> Result<()> {
        Ok(())
    }

    fn native_watermark_capability(&self) -> NativeWatermarkCapability {
        if self.native_watermarks {
            NativeWatermarkCapability::EmitsNative
        } else {
            NativeWatermarkCapability::NeverEmits
        }
    }

    fn capabilities(&self) -> SourceCapabilities {
        SourceCapabilities {
            replayable: true,
            max_batch_rows: 1,
            max_batch_bytes: 1 << 20,
        }
    }
}

struct TransactionalSink {
    pending: Vec<Match>,
    state: Arc<Mutex<VisibleState>>,
}

#[async_trait]
impl TransactionalStreamSink for TransactionalSink {
    async fn open(&mut self) -> Result<()> {
        self.state.lock().opens += 1;
        Ok(())
    }
    async fn begin_epoch(&mut self, _epoch: Epoch) -> Result<()> {
        self.pending.clear();
        Ok(())
    }
    async fn write(&mut self, batch: &Batch) -> Result<()> {
        self.pending.extend(records(batch));
        self.state.lock().writes += 1;
        Ok(())
    }
    async fn pre_commit(&mut self, _epoch: Epoch) -> Result<JsonMap> {
        Ok(BTreeMap::from([(
            "records".into(),
            serde_json::to_value(&self.pending).unwrap(),
        )]))
    }
    async fn commit(&mut self, epoch: Epoch, pending: &JsonMap) -> Result<()> {
        let records: Vec<Match> = serde_json::from_value(pending["records"].clone()).unwrap();
        let mut state = self.state.lock();
        if state.committed_epochs.insert(epoch.as_u64()) {
            state.records.extend(records);
        }
        Ok(())
    }
    async fn abort(&mut self, _epoch: Epoch, _state: Option<&JsonMap>) -> Result<()> {
        self.pending.clear();
        Ok(())
    }
    async fn recover(&mut self, manifest: &CheckpointManifest) -> Result<()> {
        let state = manifest.sinks()["result"].pre_commit.as_ref().unwrap();
        self.commit(manifest.epoch(), state).await
    }
    async fn close(&mut self) -> Result<()> {
        Ok(())
    }
}

struct OrdinarySink {
    state: Arc<Mutex<VisibleState>>,
    block_write: Option<Arc<AtomicBool>>,
}

#[async_trait]
impl OrdinaryStreamSink for OrdinarySink {
    async fn open(&mut self) -> Result<()> {
        self.state.lock().opens += 1;
        Ok(())
    }
    async fn write(&mut self, batch: &Batch) -> Result<()> {
        {
            let mut state = self.state.lock();
            state.records.extend(records(batch));
            state.writes += 1;
        }
        if let Some(entered) = &self.block_write {
            entered.store(true, Ordering::SeqCst);
            std::future::pending::<()>().await;
        }
        Ok(())
    }
    async fn close(&mut self) -> Result<()> {
        Ok(())
    }
}

struct Scenario {
    transactional: bool,
    pauses: [Option<usize>; 2],
    hold_open: bool,
    probes: [SourceProbe; 2],
    state: Arc<Mutex<VisibleState>>,
    block_write: Option<Arc<AtomicBool>>,
    edge_rows: usize,
    disabled_side: Option<usize>,
}

impl Scenario {
    fn new(transactional: bool) -> Self {
        Self {
            transactional,
            pauses: [None, None],
            hold_open: false,
            probes: [SourceProbe::default(), SourceProbe::default()],
            state: Arc::default(),
            block_write: None,
            edge_rows: 16,
            disabled_side: None,
        }
    }

    fn spec(&self, job_id: u64) -> ContinuousJobSpec {
        let guarantee = if self.transactional {
            DeliveryGuarantee::ExactlyOnce
        } else {
            DeliveryGuarantee::AtLeastOnce
        };
        let plan = PipelineBuilder::new("asof-managed-recovery")
            .unwrap()
            .add_node("match", Box::new(operator()))
            .unwrap()
            .compile_stream(
                &UdfRegistry::new().snapshot(),
                &StreamRequirements {
                    delivery: BTreeMap::from([("output".into(), guarantee)]),
                },
            )
            .unwrap();
        let binding = if self.transactional {
            OrdinarySinkBinding::new_transactional(Box::new(TransactionalSink {
                pending: Vec::new(),
                state: self.state.clone(),
            }))
        } else {
            OrdinarySinkBinding::new(Box::new(OrdinarySink {
                state: self.state.clone(),
                block_write: self.block_write.clone(),
            }))
        };
        ContinuousJobSpec {
            context: StreamJobContext::new(
                job_id,
                plan.fingerprint(),
                JsonMap::new(),
                None,
                CancellationToken::new(),
            ),
            plan,
            sources: ["left", "right"]
                .into_iter()
                .enumerate()
                .map(|(index, name)| NamedSourceBinding {
                    binding_id: name.into(),
                    binding: SourceBinding::new(
                        Box::new(ScriptedSource {
                            times: if index == 0 {
                                vec![105, 205, 305, 2_005]
                            } else {
                                vec![100, 200, 300, 2_000]
                            },
                            next: 0,
                            watermark_sent: false,
                            native_watermarks: self.disabled_side != Some(index),
                            pause_after: self.pauses[index],
                            hold_open: self.hold_open,
                            probe: self.probes[index].clone(),
                        }),
                        None,
                        0,
                    )
                    .unwrap()
                    .with_watermark_policy(
                        if self.disabled_side == Some(index) {
                            WatermarkPolicy::Disabled { idle_timeout: None }
                        } else {
                            WatermarkPolicy::SourceProvided
                        },
                    ),
                })
                .collect(),
            sinks: vec![NamedSinkBinding {
                output_id: "output".into(),
                sink_id: "result".into(),
                binding,
            }],
            edge_budget: EdgeBudget {
                max_rows: self.edge_rows,
                max_bytes: 1 << 20,
            },
            delivery_mode: M2DeliveryMode::ProcessLocalOrdered,
            static_inputs: crate::static_input::PreparedStaticInputs::default(),
        }
    }
}

fn checkpoint(root: &Path) -> CheckpointRuntimeSpec {
    CheckpointRuntimeSpec::managed(
        ManagedCheckpointRuntime::new(root).unwrap(),
        StreamRuntimeConfig {
            checkpoint_interval: Duration::from_secs(3_600),
            checkpoint_timeout: Duration::from_secs(10),
            ..StreamRuntimeConfig::default()
        },
    )
    .unwrap()
}

async fn wait_until(condition: impl Fn() -> bool) {
    tokio::time::timeout(Duration::from_secs(10), async {
        while !condition() {
            tokio::task::yield_now().await;
        }
    })
    .await
    .expect("ASOF runtime condition must become observable");
}

async fn manifest(root: &Path, epoch: Epoch) -> CheckpointManifest {
    let path = root
        .join("manifests")
        .join(format!("manifest-{:020}.json", epoch.as_u64()));
    CheckpointManifest::from_bytes(&tokio::fs::read(path).await.unwrap()).unwrap()
}

fn expected() -> Vec<Match> {
    vec![
        Match {
            left_time: 105,
            left_sequence: 1,
            right_time: Some(100),
            right_sequence: Some(1),
        },
        Match {
            left_time: 205,
            left_sequence: 2,
            right_time: Some(200),
            right_sequence: Some(2),
        },
        Match {
            left_time: 305,
            left_sequence: 3,
            right_time: Some(300),
            right_sequence: Some(3),
        },
        Match {
            left_time: 2_005,
            left_sequence: 4,
            right_time: Some(2_000),
            right_sequence: Some(4),
        },
    ]
}

async fn complete(scenario: &Scenario, root: &Path, job_id: u64) {
    let mut runner = ContinuousRunner::new();
    let job = runner
        .start_checkpointed(scenario.spec(job_id), checkpoint(root))
        .await
        .unwrap();
    let outcome = tokio::time::timeout(Duration::from_secs(10), job.wait())
        .await
        .expect("ASOF recovery must finish");
    assert_eq!(outcome.state, ContinuousJobState::Completed, "{outcome:?}");
    let statuses = job.stream_asof_join_status();
    let status = &statuses["match"];
    assert_eq!(status.left.accepted_rows, 4);
    assert_eq!(status.right.accepted_rows, 4);
    assert_eq!(status.emitted_left_rows, 4);
    assert_eq!(status.matched_rows, 4);
    assert_eq!(status.unmatched_rows, 0);
    assert_eq!(status.pending_left_rows, 0);
    drop(job);
    runner.shutdown().await.unwrap();
    assert_eq!(runner.registry_counts(), (0, 0));
}

async fn publish_pending_cut(scenario: &Scenario, root: &Path, job_id: u64) -> CheckpointManifest {
    for probe in &scenario.probes {
        probe.paused.store(false, Ordering::SeqCst);
    }
    let mut runner = ContinuousRunner::new();
    let job = runner
        .start_checkpointed(scenario.spec(job_id), checkpoint(root))
        .await
        .unwrap();
    wait_until(|| {
        scenario
            .probes
            .iter()
            .all(|probe| probe.paused.load(Ordering::SeqCst))
    })
    .await;
    let epoch = tokio::time::timeout(Duration::from_secs(10), job.trigger_checkpoint())
        .await
        .unwrap()
        .unwrap();
    let captured = manifest(root, epoch).await;
    assert!(!captured.operators()["match"].segments.is_empty());
    assert!(scenario.state.lock().records.is_empty());
    assert_eq!(job.cancel().await.state, ContinuousJobState::Cancelled);
    drop(job);
    runner.shutdown().await.unwrap();
    captured
}

#[tokio::test]
async fn test_asof_managed_restore_preserves_pending_left_without_or_with_candidate() {
    for candidate in [false, true] {
        let directory = tempfile::tempdir().unwrap();
        let mut scenario = Scenario::new(true);
        scenario.pauses = [Some(1), Some(usize::from(candidate))];
        let first = publish_pending_cut(&scenario, directory.path(), 1).await;
        let second = publish_pending_cut(&scenario, directory.path(), 2).await;
        assert_eq!(first.epoch(), Epoch::INITIAL);
        assert_eq!(second.epoch(), Epoch::new(2).unwrap());
        for captured in [&first, &second] {
            assert_eq!(
                captured.sources()["left"].cursor.as_ref().unwrap().order,
                "0000000000000001"
            );
            assert_eq!(captured.sources()["right"].cursor.is_some(), candidate);
        }
        let hashes = |captured: &CheckpointManifest| {
            captured.operators()["match"]
                .segments
                .iter()
                .map(|segment| segment.sha256().to_owned())
                .collect::<Vec<_>>()
        };
        assert_eq!(hashes(&first), hashes(&second));
        scenario.pauses = [None, None];
        complete(&scenario, directory.path(), 3).await;
        assert_eq!(
            scenario.state.lock().records,
            expected(),
            "candidate={candidate}"
        );
        assert_eq!(*scenario.probes[0].opens.lock(), [0, 1, 1]);
        assert_eq!(
            *scenario.probes[1].opens.lock(),
            [0, usize::from(candidate), usize::from(candidate)]
        );
    }
}

#[tokio::test]
async fn test_asof_managed_faults_recover_exactly_once_before_and_after_publication() {
    for point in [
        CheckpointFaultPoint::StateStage,
        CheckpointFaultPoint::ManifestWrite,
        CheckpointFaultPoint::CompletedCommit,
    ] {
        let directory = tempfile::tempdir().unwrap();
        let mut scenario = Scenario::new(true);
        scenario.hold_open = true;
        let (faulted, probe) =
            checkpoint(directory.path()).with_fault_probe(point, CheckpointFaultMode::Restart);
        let mut runner = ContinuousRunner::new();
        let job = runner
            .start_checkpointed(scenario.spec(10), faulted)
            .await
            .unwrap();
        wait_until(|| scenario.state.lock().writes > 0).await;
        let (manual, outcome) = tokio::time::timeout(Duration::from_secs(10), async {
            tokio::join!(job.trigger_checkpoint(), job.wait())
        })
        .await
        .expect("injected ASOF checkpoint failure must finish");
        if point == CheckpointFaultPoint::CompletedCommit {
            if let Ok(epoch) = manual {
                assert_eq!(epoch, Epoch::INITIAL);
            }
        } else {
            assert!(manual.is_err(), "{point:?}");
        }
        assert_ne!(outcome.state, ContinuousJobState::Completed, "{point:?}");
        assert_eq!(probe.trigger_count(), 1, "{point:?}");
        drop(job);
        runner.shutdown().await.unwrap();
        let published = tokio::fs::try_exists(
            directory
                .path()
                .join("manifests/manifest-00000000000000000001.json"),
        )
        .await
        .unwrap();
        assert_eq!(
            published,
            point == CheckpointFaultPoint::CompletedCommit,
            "{point:?}"
        );
        scenario.hold_open = false;
        complete(&scenario, directory.path(), 11).await;
        assert_eq!(scenario.state.lock().records, expected(), "{point:?}");
        let opens = scenario
            .probes
            .iter()
            .map(|probe| probe.opens.lock().len())
            .collect::<Vec<_>>();
        let writes = scenario.state.lock().writes;
        complete(&scenario, directory.path(), 12).await;
        assert_eq!(
            scenario.state.lock().records,
            expected(),
            "terminal restore at {point:?}"
        );
        assert_eq!(scenario.state.lock().writes, writes);
        assert_eq!(
            scenario
                .probes
                .iter()
                .map(|probe| probe.opens.lock().len())
                .collect::<Vec<_>>(),
            opens
        );
    }
}

#[tokio::test]
async fn test_asof_ordinary_sink_can_replay_after_output_before_manifest_publication() {
    let directory = tempfile::tempdir().unwrap();
    let mut scenario = Scenario::new(false);
    scenario.hold_open = true;
    let (faulted, probe) = checkpoint(directory.path()).with_fault_probe(
        CheckpointFaultPoint::ManifestWrite,
        CheckpointFaultMode::Restart,
    );
    let mut runner = ContinuousRunner::new();
    let job = runner
        .start_checkpointed(scenario.spec(20), faulted)
        .await
        .unwrap();
    wait_until(|| scenario.state.lock().records.len() == 3).await;
    let (manual, outcome) = tokio::time::timeout(Duration::from_secs(10), async {
        tokio::join!(job.trigger_checkpoint(), job.wait())
    })
    .await
    .unwrap();
    assert!(manual.is_err());
    assert_ne!(outcome.state, ContinuousJobState::Completed);
    assert_eq!(probe.trigger_count(), 1);
    drop(job);
    runner.shutdown().await.unwrap();
    assert_eq!(scenario.state.lock().records, expected()[..3]);
    scenario.hold_open = false;
    complete(&scenario, directory.path(), 21).await;
    assert_eq!(
        scenario.state.lock().records,
        [expected()[..3].to_vec(), expected()].concat()
    );
}

#[tokio::test]
async fn test_asof_collector_backpressure_cancellation_restores_last_published_pending_cut() {
    let directory = tempfile::tempdir().unwrap();
    let mut scenario = Scenario::new(false);
    scenario.pauses = [Some(3), Some(3)];
    scenario.edge_rows = 1;
    let mut runner = ContinuousRunner::new();
    let job = runner
        .start_checkpointed(scenario.spec(30), checkpoint(directory.path()))
        .await
        .unwrap();
    wait_until(|| {
        scenario
            .probes
            .iter()
            .all(|probe| probe.paused.load(Ordering::SeqCst))
    })
    .await;
    job.trigger_checkpoint().await.unwrap();
    assert_eq!(job.cancel().await.state, ContinuousJobState::Cancelled);
    drop(job);
    runner.shutdown().await.unwrap();

    scenario.pauses = [None, None];
    let blocked = Arc::new(AtomicBool::new(false));
    scenario.block_write = Some(blocked.clone());
    let mut runner = ContinuousRunner::new();
    let job = runner
        .start_checkpointed(scenario.spec(31), checkpoint(directory.path()))
        .await
        .unwrap();
    wait_until(|| blocked.load(Ordering::SeqCst) || job.status().terminal_cause.is_some()).await;
    assert!(
        blocked.load(Ordering::SeqCst),
        "ASOF failed before reaching blocked sink: {:?}",
        job.status().terminal_cause
    );
    wait_until(|| {
        job.status()
            .edges
            .iter()
            .any(|(name, edge)| name.starts_with("sink/") && edge.blocked_sends > 0)
    })
    .await;
    assert_eq!(scenario.state.lock().records, expected()[..1]);
    assert_eq!(job.cancel().await.state, ContinuousJobState::Cancelled);
    assert!(job.status().edges.values().all(|edge| {
        edge.queue_depth == 0 && edge.charged_rows == 0 && edge.charged_bytes == 0
    }));
    drop(job);
    runner.shutdown().await.unwrap();
    assert_eq!(runner.registry_counts(), (0, 0));
    scenario.block_write = None;
    complete(&scenario, directory.path(), 32).await;
    assert_eq!(
        scenario.state.lock().records,
        [expected()[..1].to_vec(), expected()].concat()
    );
}

#[tokio::test]
async fn test_asof_direct_runner_rejects_disabled_reachable_watermark_before_source_open() {
    for (index, side) in ["left", "right"].into_iter().enumerate() {
        let mut scenario = Scenario::new(false);
        scenario.disabled_side = Some(index);
        let spec = scenario.spec(40);
        let error = match crate::runtime::streaming::job::preflight_job(spec) {
            Ok(_) => panic!("disabled {side} source must fail ASOF preflight"),
            Err(error) => error,
        };
        assert!(matches!(error, CalcFlowError::InvalidArgument { .. }));
        assert!(
            scenario
                .probes
                .iter()
                .all(|probe| probe.opens.lock().is_empty())
        );
    }
}

#[test]
fn test_asof_direct_runner_allows_disabled_source_on_an_unrelated_branch() {
    let scenario = Scenario::new(false);
    let mut spec = scenario.spec(41);
    let plan = PipelineBuilder::new("asof-with-unrelated-input")
        .unwrap()
        .add_node("match", Box::new(operator()))
        .unwrap()
        .add_node(
            "bypass",
            Box::new(
                crate::ExpressionOperator::new(
                    "bypass",
                    "copy = value",
                    Vec::new(),
                    None,
                    Vec::new(),
                )
                .unwrap(),
            ),
        )
        .unwrap()
        .compile_stream(
            &UdfRegistry::new().snapshot(),
            &StreamRequirements::default(),
        )
        .unwrap();
    let mut original = spec
        .sources
        .into_iter()
        .map(|source| (source.binding_id, source.binding))
        .collect::<BTreeMap<_, _>>();
    spec.sources = plan
        .external_inputs()
        .iter()
        .map(|(name, endpoint)| {
            let binding = if endpoint.node_id == "match" {
                original.remove(&endpoint.port).unwrap()
            } else {
                SourceBinding::new(
                    Box::new(ScriptedSource {
                        times: Vec::new(),
                        next: 0,
                        watermark_sent: false,
                        native_watermarks: false,
                        pause_after: None,
                        hold_open: false,
                        probe: SourceProbe::default(),
                    }),
                    None,
                    0,
                )
                .unwrap()
                .with_watermark_policy(WatermarkPolicy::Disabled { idle_timeout: None })
            };
            NamedSourceBinding {
                binding_id: name.clone(),
                binding,
            }
        })
        .collect();
    spec.sinks = plan
        .sink_binding_ids()
        .into_iter()
        .enumerate()
        .map(|(index, name)| NamedSinkBinding {
            output_id: name.into(),
            sink_id: format!("sink-{index}"),
            binding: OrdinarySinkBinding::new(Box::new(OrdinarySink {
                state: Arc::default(),
                block_write: None,
            })),
        })
        .collect();
    spec.context = StreamJobContext::new(
        41,
        plan.fingerprint(),
        JsonMap::new(),
        None,
        CancellationToken::new(),
    );
    spec.plan = plan;
    assert!(crate::runtime::streaming::job::preflight_job(spec).is_ok());
    assert!(
        scenario
            .probes
            .iter()
            .all(|probe| probe.opens.lock().is_empty())
    );
}

#[tokio::test]
async fn test_asof_managed_recovery_rejects_missing_and_corrupted_state_before_source_open() {
    for missing in [false, true] {
        let directory = tempfile::tempdir().unwrap();
        let mut scenario = Scenario::new(true);
        scenario.pauses = [Some(1), Some(1)];
        let captured = publish_pending_cut(&scenario, directory.path(), 50).await;
        let handle = &captured.operators()["match"].segments[0];
        let path = directory.path().join("state").join(handle.relative_path());
        if missing {
            tokio::fs::remove_file(path).await.unwrap();
        } else {
            let mut bytes = tokio::fs::read(&path).await.unwrap();
            bytes[0] ^= 1;
            tokio::fs::write(path, bytes).await.unwrap();
        }
        let opens = scenario
            .probes
            .iter()
            .map(|probe| probe.opens.lock().len())
            .collect::<Vec<_>>();
        let mut runner = ContinuousRunner::new();
        let failure = runner
            .start_checkpointed(scenario.spec(51), checkpoint(directory.path()))
            .await
            .expect_err("invalid ASOF state must fail managed recovery");
        assert!(
            matches!(failure.primary.error, CalcFlowError::CheckpointMismatch { ref message }
            if message == "checkpoint lineage contains an invalid manifest candidate"),
            "unexpected managed recovery error: {failure:?}"
        );
        runner.shutdown().await.unwrap();
        assert_eq!(runner.registry_counts(), (0, 0));
        assert_eq!(
            scenario
                .probes
                .iter()
                .map(|probe| probe.opens.lock().len())
                .collect::<Vec<_>>(),
            opens
        );
        assert!(scenario.state.lock().records.is_empty());
    }
}

#[tokio::test]
async fn test_asof_terminal_manifest_valid_checksum_still_validates_native_state_before_sink_open()
{
    let directory = tempfile::tempdir().unwrap();
    let scenario = Scenario::new(true);
    complete(&scenario, directory.path(), 60).await;
    let captured = manifest(directory.path(), Epoch::INITIAL).await;
    assert!(captured.sources().values().all(|entry| entry.ended));
    let mut operators = captured.operators().clone();
    let metadata = &mut operators.get_mut("match").unwrap().inline_metadata;
    metadata.get_mut("metrics").unwrap()["matched_rows"] = serde_json::json!(99);
    let damaged = CheckpointManifest::new(crate::CheckpointManifestFields {
        pipeline_name: captured.pipeline_name().into(),
        pipeline_fingerprint: captured.pipeline_fingerprint().into(),
        runtime_config_hash: captured.runtime_config_hash().into(),
        epoch: captured.epoch(),
        created_at: captured.created_at(),
        recovery_status: captured.recovery_status(),
        sources: captured.sources().clone(),
        operators,
        sinks: captured.sinks().clone(),
        static_inputs: captured.static_inputs().clone(),
    })
    .unwrap();
    let bytes = damaged.canonical_bytes().unwrap();
    assert!(CheckpointManifest::from_bytes(&bytes).is_ok());
    tokio::fs::write(
        directory
            .path()
            .join("manifests/manifest-00000000000000000001.json"),
        bytes,
    )
    .await
    .unwrap();
    let source_opens = scenario
        .probes
        .iter()
        .map(|probe| probe.opens.lock().len())
        .collect::<Vec<_>>();
    let sink_opens = scenario.state.lock().opens;
    let mut runner = ContinuousRunner::new();
    let error = runner
        .start_checkpointed(scenario.spec(61), checkpoint(directory.path()))
        .await
        .expect_err("terminal recovery must validate the ASOF counters before sink recovery");
    assert!(
        matches!(error.primary.error, CalcFlowError::Internal { ref message }
        if message == "managed checkpoint recovery failed"),
        "{error:?}"
    );
    runner.shutdown().await.unwrap();
    assert_eq!(runner.registry_counts(), (0, 0));
    assert_eq!(scenario.state.lock().opens, sink_opens);
    assert_eq!(scenario.state.lock().records, expected());
    assert_eq!(
        scenario
            .probes
            .iter()
            .map(|probe| probe.opens.lock().len())
            .collect::<Vec<_>>(),
        source_opens
    );
}
