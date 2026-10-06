use super::*;
use crate::runtime::streaming::runner::{OneShotContinuousRunner, TerminalCause};
use crate::{BatchKind, Edge, ExpressionOperator, OperatorMetadata, Port, PortEndpoint};
use std::sync::{Weak, atomic::AtomicUsize};

struct Probe {
    source_closes: AtomicUsize,
    sink_closes: AtomicUsize,
    close_entered: AtomicUsize,
    close_gate: tokio::sync::Semaphore,
    block_close: AtomicBool,
    block_write: AtomicBool,
    write_entered: AtomicBool,
    arrays: Mutex<Vec<Weak<dyn Array>>>,
    batches: Mutex<Vec<Batch>>,
}

impl Default for Probe {
    fn default() -> Self {
        Self {
            source_closes: AtomicUsize::new(0),
            sink_closes: AtomicUsize::new(0),
            close_entered: AtomicUsize::new(0),
            close_gate: tokio::sync::Semaphore::new(0),
            block_close: AtomicBool::new(false),
            block_write: AtomicBool::new(false),
            write_entered: AtomicBool::new(false),
            arrays: Mutex::new(Vec::new()),
            batches: Mutex::new(Vec::new()),
        }
    }
}

struct Source {
    inner: ScriptedSource,
    probe: Arc<Probe>,
    _owned: Arc<()>,
}

#[async_trait]
impl StreamSource for Source {
    async fn open(&mut self, cursor: Option<Cursor>) -> Result<()> {
        self.inner.open(cursor).await
    }
    async fn next(&mut self) -> Result<Option<SourceEvent>> {
        let event = self.inner.next().await?;
        if let Some(SourceEvent::Data { batch, .. }) = &event {
            for record in batch.table_payload()?.batches() {
                self.probe
                    .arrays
                    .lock()
                    .extend(record.columns().iter().map(Arc::downgrade));
            }
        }
        Ok(event)
    }
    async fn close(&mut self) -> Result<()> {
        self.probe.close_entered.fetch_add(1, Ordering::SeqCst);
        if self.probe.block_close.load(Ordering::SeqCst) {
            self.probe.close_gate.acquire().await.unwrap().forget();
        }
        self.inner.close().await?;
        self.probe.source_closes.fetch_add(1, Ordering::SeqCst);
        Ok(())
    }
    fn capabilities(&self) -> SourceCapabilities {
        self.inner.capabilities()
    }
    fn native_watermark_capability(&self) -> NativeWatermarkCapability {
        self.inner.native_watermark_capability()
    }
}

struct Sink {
    inner: TransactionalSink,
    probe: Arc<Probe>,
    _owned: Arc<()>,
}

#[async_trait]
impl TransactionalStreamSink for Sink {
    async fn open(&mut self) -> Result<()> {
        self.inner.open().await
    }
    async fn begin_epoch(&mut self, epoch: Epoch) -> Result<()> {
        self.inner.begin_epoch(epoch).await
    }
    async fn write(&mut self, batch: &Batch) -> Result<()> {
        self.probe.batches.lock().push(batch.clone());
        self.inner.write(batch).await?;
        self.probe.write_entered.store(true, Ordering::SeqCst);
        if self.probe.block_write.load(Ordering::SeqCst) {
            std::future::pending::<()>().await;
        }
        Ok(())
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
        self.inner.recover(manifest).await
    }
    async fn close(&mut self) -> Result<()> {
        self.inner.close().await?;
        self.probe.sink_closes.fetch_add(1, Ordering::SeqCst);
        Ok(())
    }
}

fn chain_plan() -> crate::StreamExecutionPlan {
    let head = operator();
    let output = head.output_ports()[0].schema().unwrap().clone();
    let projection = |name: &str| {
        ExpressionOperator::new(
            name,
            "",
            output
                .fields()
                .iter()
                .map(|field| field.name().clone())
                .collect(),
            None,
            vec![],
        )
        .unwrap()
        .with_ports(
            Port::with_schema_ref("input", BatchKind::Table, true, Some(output.clone())).unwrap(),
            Port::with_schema_ref("output", BatchKind::Table, true, Some(output.clone())).unwrap(),
        )
        .unwrap()
    };
    PipelineBuilder::new("asof-managed-fused-chain")
        .unwrap()
        .add_node("match", Box::new(head))
        .unwrap()
        .add_node("project1", Box::new(projection("project1")))
        .unwrap()
        .add_node("project2", Box::new(projection("project2")))
        .unwrap()
        .connect(Edge::new(
            PortEndpoint::new("match", "output").unwrap(),
            PortEndpoint::new("project1", "input").unwrap(),
        ))
        .unwrap()
        .connect(Edge::new(
            PortEndpoint::new("project1", "output").unwrap(),
            PortEndpoint::new("project2", "input").unwrap(),
        ))
        .unwrap()
        .compile_stream(
            &UdfRegistry::new().snapshot(),
            &StreamRequirements {
                delivery: BTreeMap::from([("output".into(), DeliveryGuarantee::ExactlyOnce)]),
            },
        )
        .unwrap()
}

fn chain_spec(
    scenario: &Scenario,
    job_id: u64,
    probe: &Arc<Probe>,
) -> (ContinuousJobSpec, Vec<Weak<()>>) {
    let plan = chain_plan();
    let mut spec = scenario.spec(job_id);
    spec.context = StreamJobContext::new(
        job_id,
        plan.fingerprint(),
        JsonMap::new(),
        None,
        CancellationToken::new(),
    );
    spec.plan = plan;
    let mut resources = Vec::new();
    spec.sources = ["left", "right"]
        .into_iter()
        .enumerate()
        .map(|(index, name)| {
            let owned = Arc::new(());
            resources.push(Arc::downgrade(&owned));
            NamedSourceBinding {
                binding_id: name.into(),
                binding: SourceBinding::new(
                    Box::new(Source {
                        inner: ScriptedSource {
                            times: if index == 0 {
                                vec![105, 205, 305, 2_005]
                            } else {
                                vec![100, 200, 300, 2_000]
                            },
                            next: 0,
                            watermark_sent: false,
                            native_watermarks: true,
                            pause_after: scenario.pauses[index],
                            hold_open: scenario.hold_open,
                            probe: scenario.probes[index].clone(),
                        },
                        probe: probe.clone(),
                        _owned: owned,
                    }),
                    None,
                    0,
                )
                .unwrap()
                .with_watermark_policy(WatermarkPolicy::SourceProvided),
            }
        })
        .collect();
    let owned = Arc::new(());
    resources.push(Arc::downgrade(&owned));
    spec.sinks = vec![NamedSinkBinding {
        output_id: "output".into(),
        sink_id: "result".into(),
        binding: OrdinarySinkBinding::new_transactional(Box::new(Sink {
            inner: TransactionalSink {
                pending: vec![],
                state: scenario.state.clone(),
            },
            probe: probe.clone(),
            _owned: owned,
        })),
    }];
    (spec, resources)
}

fn assert_resources(probe: &Probe, resources: &[Weak<()>]) {
    assert!(
        resources
            .iter()
            .all(|resource| resource.upgrade().is_none())
    );
    probe.batches.lock().clear();
    assert!(
        probe
            .arrays
            .lock()
            .iter()
            .all(|array| array.upgrade().is_none())
    );
}

async fn pending_checkpoint(
    scenario: &Scenario,
    root: &Path,
    job_id: u64,
    source_arrays: usize,
) -> CheckpointManifest {
    let probe = Arc::new(Probe::default());
    let (spec, resources) = chain_spec(scenario, job_id, &probe);
    let fingerprint = spec.plan.fingerprint().to_owned();
    let mut runner = ContinuousRunner::new();
    let job = runner
        .start_checkpointed(spec, checkpoint(root))
        .await
        .unwrap();
    let registry = job.core.runtime_status.lock().tasks.clone();
    let held = tokio::time::timeout(Duration::from_secs(10), async {
        while !scenario
            .probes
            .iter()
            .all(|source| source.paused.load(Ordering::SeqCst))
        {
            tokio::task::yield_now().await;
        }
    })
    .await;
    let statuses = registry.snapshot();
    let checkpointed =
        tokio::time::timeout(Duration::from_secs(10), job.trigger_checkpoint()).await;
    let captured = match &checkpointed {
        Ok(Ok(epoch)) => Some(manifest(root, *epoch).await),
        _ => None,
    };
    let outcome = job.cancel().await;
    drop(job);
    runner.shutdown().await.unwrap();
    assert!(held.is_ok());
    assert_eq!(outcome.state, ContinuousJobState::Cancelled);
    assert!(registry.snapshot().is_empty());
    assert_eq!(runner.registry_counts(), (0, 0));
    assert_resources(&probe, &resources);
    assert_eq!(probe.arrays.lock().len(), source_arrays);
    assert_eq!(probe.source_closes.load(Ordering::SeqCst), 2);
    assert_eq!(probe.sink_closes.load(Ordering::SeqCst), 1);
    for (index, name) in ["match", "project1", "project2"].into_iter().enumerate() {
        assert_eq!(
            statuses[&crate::runtime::streaming::supervisor::TaskId::new(index as u64)].task_name,
            format!("operator:{name}")
        );
    }
    for name in [
        "source:left:pump",
        "source:left:task",
        "source:right:pump",
        "source:right:task",
        "sink:output",
    ] {
        assert!(
            statuses.values().any(|task| task.task_name == name),
            "missing {name}: {statuses:?}"
        );
    }
    let captured = captured.expect("dual barriers did not publish checkpoint");
    assert_eq!(
        captured
            .operators()
            .keys()
            .map(String::as_str)
            .collect::<Vec<_>>(),
        ["match", "project1", "project2"]
    );
    assert_eq!(
        captured.sources()["left"].cursor.as_ref().unwrap().order,
        "0000000000000001"
    );
    assert_eq!(
        captured.sources()["right"].cursor.as_ref().unwrap().order,
        "0000000000000001"
    );
    assert_eq!(
        captured.operators()["match"]
            .progress
            .keys()
            .map(String::as_str)
            .collect::<Vec<_>>(),
        ["left", "right"]
    );
    assert!(!captured.operators()["match"].segments.is_empty());
    assert_eq!(chain_plan().fingerprint(), fingerprint);
    captured
}

fn assert_output(probe: &Probe, expected_schema: &SchemaRef) {
    let batches = probe.batches.lock();
    let gathered = datafusion::arrow::compute::concat_batches(
        expected_schema,
        batches
            .iter()
            .flat_map(|batch| batch.table_payload().unwrap().batches()),
    )
    .unwrap();
    let expected_batch = RecordBatch::try_new(
        expected_schema.clone(),
        vec![
            Arc::new(StringArray::from(vec!["a"; 4])),
            Arc::new(
                TimestampMicrosecondArray::from(vec![105, 205, 305, 2005]).with_timezone("UTC"),
            ),
            Arc::new(UInt64Array::from(vec![1, 2, 3, 4])),
            Arc::new(Int64Array::from(vec![105, 205, 305, 2005])),
            Arc::new(StringArray::from(vec!["a"; 4])),
            Arc::new(
                TimestampMicrosecondArray::from(vec![100, 200, 300, 2000]).with_timezone("UTC"),
            ),
            Arc::new(UInt64Array::from(vec![1, 2, 3, 4])),
            Arc::new(Int64Array::from(vec![100, 200, 300, 2000])),
        ],
    )
    .unwrap();
    assert_eq!(gathered, expected_batch);
    drop(gathered);
    let mut sequence = 0;
    for batch in batches.iter() {
        assert_eq!(batch.metadata().source(), "match");
        assert_eq!(batch.metadata().sequence(), sequence);
        sequence += batch.num_rows() as u64;
        assert!(batch.metadata().attributes().is_empty());
        assert_eq!(batch.table_payload().unwrap().schema(), expected_schema);
    }
    drop(batches);
}

#[tokio::test(flavor = "current_thread")]
async fn a13_three_member_asof_managed_dual_barriers_restore_every_logical_participant() {
    let directory = tempfile::tempdir().unwrap();
    let mut scenario = Scenario::new(true);
    scenario.pauses = [Some(1), Some(1)];
    let first = pending_checkpoint(&scenario, directory.path(), 101, 8).await;
    for source in &scenario.probes {
        source.paused.store(false, Ordering::SeqCst);
    }
    let second = pending_checkpoint(&scenario, directory.path(), 102, 0).await;
    assert_eq!(first.epoch(), Epoch::INITIAL);
    assert_eq!(second.epoch(), Epoch::INITIAL.next().unwrap());
    assert_eq!(
        first.operators()["match"]
            .segments
            .iter()
            .map(crate::StateHandle::sha256)
            .collect::<Vec<_>>(),
        second.operators()["match"]
            .segments
            .iter()
            .map(crate::StateHandle::sha256)
            .collect::<Vec<_>>()
    );
    scenario.pauses = [None, None];
    let probe = Arc::new(Probe::default());
    let (spec, resources) = chain_spec(&scenario, 103, &probe);
    let expected_schema = operator().output_ports()[0].schema().unwrap().clone();
    let mut runner = ContinuousRunner::new();
    let job = runner
        .start_checkpointed(spec, checkpoint(directory.path()))
        .await
        .unwrap();
    let outcome = tokio::time::timeout(Duration::from_secs(10), job.wait()).await;
    let metrics = job.core.metrics.snapshot();
    let delivery = job.public_status().delivery;
    let status = job.stream_asof_join_status()["match"].clone();
    if outcome.is_err() {
        let _ = job.cancel().await;
    }
    drop(job);
    runner.shutdown().await.unwrap();
    assert_eq!(outcome.unwrap().state, ContinuousJobState::Completed);
    assert_eq!(runner.registry_counts(), (0, 0));
    assert_eq!(delivery["output"].requested, DeliveryGuarantee::ExactlyOnce);
    assert_eq!(delivery["output"].effective, DeliveryGuarantee::ExactlyOnce);
    assert_eq!(
        (
            status.emitted_left_rows,
            status.matched_rows,
            status.pending_left_rows
        ),
        (4, 4, 0)
    );
    assert_eq!(scenario.state.lock().records, expected());
    assert_output(&probe, &expected_schema);
    assert!(
        metrics
            .edges
            .values()
            .all(|edge| edge.channel.queue_depth == 0
                && edge.channel.charged_rows == 0
                && edge.channel.charged_bytes == 0
                && !edge.drop_invariant_violated)
    );
    assert_resources(&probe, &resources);
    assert_eq!(probe.arrays.lock().len(), 24);
    let terminal = manifest(directory.path(), second.epoch().next().unwrap()).await;
    assert!(terminal.operators().values().all(|entry| {
        entry
            .progress
            .values()
            .all(|ingress| ingress.state == crate::ManifestIngressState::Ended)
    }));
    assert_eq!(*scenario.probes[0].opens.lock(), [0, 1, 1]);
    assert_eq!(*scenario.probes[1].opens.lock(), [0, 1, 1]);
}

#[tokio::test(flavor = "current_thread")]
async fn a13_whole_job_owner_drop_awaits_source_close_and_transactional_sink_reaper() {
    let directory = tempfile::tempdir().unwrap();
    let mut scenario = Scenario::new(true);
    scenario.hold_open = true;
    scenario.edge_rows = 1;
    let probe = Arc::new(Probe::default());
    probe.block_write.store(true, Ordering::SeqCst);
    let (spec, resources) = chain_spec(&scenario, 104, &probe);
    let job = OneShotContinuousRunner::new()
        .start_checkpointed_with_config(
            spec,
            ManagedCheckpointRuntime::new(directory.path()).unwrap(),
            StreamRuntimeConfig {
                checkpoint_interval: Duration::from_secs(3_600),
                checkpoint_timeout: Duration::from_secs(10),
                ..StreamRuntimeConfig::default()
            },
        )
        .await
        .unwrap();
    let entered = tokio::time::timeout(Duration::from_secs(10), async {
        while !probe.write_entered.load(Ordering::SeqCst) {
            tokio::task::yield_now().await;
        }
    })
    .await;
    probe.block_close.store(true, Ordering::SeqCst);
    let runner = job.runner_probe_for_test();
    let mut waiter = Box::pin(job.wait());
    drop(job);
    let closing = tokio::time::timeout(Duration::from_secs(10), async {
        while probe.close_entered.load(Ordering::SeqCst) == 0 {
            tokio::task::yield_now().await;
        }
    })
    .await;
    let pending = futures::poll!(waiter.as_mut()).is_pending();
    let finished_before_release = runner.is_finished();
    probe.close_gate.add_permits(2);
    let outcome = tokio::time::timeout(Duration::from_secs(10), waiter).await;
    runner.join().await.unwrap();
    assert!(entered.is_ok() && closing.is_ok());
    assert!(pending && !finished_before_release);
    let outcome = outcome.unwrap();
    assert_eq!(outcome.state, ContinuousJobState::Cancelled);
    assert_eq!(outcome.cause, TerminalCause::ExplicitCancel);
    assert_eq!(runner.registry_counts(), (0, 0));
    assert!(runner.is_finished());
    assert_eq!(probe.source_closes.load(Ordering::SeqCst), 2);
    assert_eq!(probe.sink_closes.load(Ordering::SeqCst), 1);
    assert!(scenario.state.lock().records.is_empty());
    assert!(!probe.arrays.lock().is_empty());
    assert_resources(&probe, &resources);
}
