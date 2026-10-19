use super::*;
use crate::{BatchKind, Edge, ExpressionOperator, OperatorMetadata, Port, PortEndpoint};
use tokio::sync::Semaphore;

struct EofSource {
    inner: Box<dyn StreamSource>,
    eof: Arc<Semaphore>,
    reached: Arc<AtomicBool>,
}

#[async_trait]
impl StreamSource for EofSource {
    async fn open(&mut self, cursor: Option<Cursor>) -> Result<()> {
        self.inner.open(cursor).await
    }

    async fn next(&mut self) -> Result<Option<SourceEvent>> {
        let event = self.inner.next().await?;
        if event.is_none() {
            self.reached.store(true, Ordering::SeqCst);
            self.eof.acquire().await.unwrap().forget();
        }
        Ok(event)
    }

    async fn close(&mut self) -> Result<()> {
        self.inner.close().await
    }

    fn native_watermark_capability(&self) -> NativeWatermarkCapability {
        self.inner.native_watermark_capability()
    }

    fn capabilities(&self) -> SourceCapabilities {
        self.inner.capabilities()
    }
}

fn projection(name: &str, schema: SchemaRef) -> ExpressionOperator {
    ExpressionOperator::new(
        name,
        "",
        schema
            .fields()
            .iter()
            .map(|field| field.name().clone())
            .collect(),
        None,
        vec![],
    )
    .unwrap()
    .with_ports(
        Port::with_schema_ref("input", BatchKind::Table, true, Some(schema.clone())).unwrap(),
        Port::with_schema_ref("output", BatchKind::Table, true, Some(schema)).unwrap(),
    )
    .unwrap()
}

fn plan() -> crate::StreamExecutionPlan {
    let join = operator();
    let output = join.output_ports()[0].schema().unwrap().clone();
    let mut builder = PipelineBuilder::new("asof-mixed-ended-checkpoint").unwrap();
    for name in ["left_input", "right_input"] {
        builder = builder
            .add_node(name, Box::new(projection(name, schema())))
            .unwrap();
    }
    builder = builder
        .add_node("match", Box::new(join))
        .unwrap()
        .add_node(
            "project_result",
            Box::new(projection("project_result", output)),
        )
        .unwrap();
    for (from, to, ingress) in [
        ("left_input", "match", "left"),
        ("right_input", "match", "right"),
        ("match", "project_result", "input"),
    ] {
        builder = builder
            .connect(Edge::new(
                PortEndpoint::new(from, "output").unwrap(),
                PortEndpoint::new(to, ingress).unwrap(),
            ))
            .unwrap();
    }
    builder
        .compile_stream(
            &UdfRegistry::new().snapshot(),
            &StreamRequirements {
                delivery: BTreeMap::from([("output".into(), DeliveryGuarantee::ExactlyOnce)]),
            },
        )
        .unwrap()
}

fn spec(
    scenario: &Scenario,
    id: u64,
    eof: &[Arc<Semaphore>; 2],
    reached: &[Arc<AtomicBool>; 2],
) -> ContinuousJobSpec {
    let mut spec = scenario.spec(id);
    spec.plan = plan();
    spec.context = StreamJobContext::new(
        id,
        spec.plan.fingerprint(),
        JsonMap::new(),
        None,
        CancellationToken::new(),
    );
    spec.sources = ["left_input", "right_input"]
        .into_iter()
        .enumerate()
        .map(|(index, name)| NamedSourceBinding {
            binding_id: format!("{name}.input"),
            binding: SourceBinding::new(
                Box::new(EofSource {
                    inner: Box::new(ScriptedSource {
                        times: if index == 0 {
                            vec![105, 205, 305, 2_005]
                        } else {
                            vec![100, 200, 300, 2_000]
                        },
                        next: 0,
                        watermark_sent: false,
                        native_watermarks: true,
                        pause_after: scenario.pauses[index],
                        hold_open: false,
                        probe: scenario.probes[index].clone(),
                    }),
                    eof: eof[index].clone(),
                    reached: reached[index].clone(),
                }),
                None,
                0,
            )
            .unwrap()
            .with_watermark_policy(WatermarkPolicy::SourceProvided),
        })
        .collect();
    spec
}

async fn request_cut(job: &super::super::super::ContinuousJob, periodic: bool) -> Epoch {
    if periodic {
        let previous = job.status().checkpoint.unwrap().last_completed_epoch;
        let coordinator = job.core.manual_checkpoint.lock().clone().unwrap();
        coordinator
            .request(
                crate::runtime::streaming::checkpoint::coordinator::CheckpointRequest::Periodic,
            )
            .await
            .unwrap();
        wait_until(|| job.status().checkpoint.unwrap().last_completed_epoch > previous).await;
        job.status()
            .checkpoint
            .unwrap()
            .last_completed_epoch
            .unwrap()
    } else {
        job.trigger_checkpoint().await.unwrap()
    }
}

async fn assert_mixed_cut(
    job: &super::super::super::ContinuousJob,
    root: &Path,
    node: &str,
    periodic: bool,
) -> Epoch {
    let result = tokio::time::timeout(Duration::from_secs(5), request_cut(job, periodic)).await;
    assert!(
        result.is_ok(),
        "ended operator {node:?} must capture the mixed cut: {:?}",
        job.status()
    );
    let epoch = result.unwrap();
    let cut = manifest(root, epoch).await;
    assert!(cut.sources()[&format!("{node}.input")].ended);
    assert_eq!(
        cut.sources().values().filter(|source| source.ended).count(),
        1
    );
    assert!(
        cut.operators()[node]
            .progress
            .values()
            .all(|entry| entry.state == crate::ManifestIngressState::Ended)
    );
    epoch
}

async fn mixed_cut(first: usize, periodic: bool, restore_ended: bool) {
    let directory = tempfile::tempdir().unwrap();
    let mut scenario = Scenario::new(true);
    scenario.pauses = [Some(1), Some(1)];
    let eof = [Arc::new(Semaphore::new(0)), Arc::new(Semaphore::new(0))];
    let reached = [
        Arc::new(AtomicBool::new(false)),
        Arc::new(AtomicBool::new(false)),
    ];
    let mut runner = ContinuousRunner::new();
    let job = runner
        .start_checkpointed(
            spec(&scenario, 2401, &eof, &reached),
            checkpoint(directory.path()),
        )
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
    let mut runner = ContinuousRunner::new();
    let mut job = runner
        .start_checkpointed(
            spec(&scenario, 2402, &eof, &reached),
            checkpoint(directory.path()),
        )
        .await
        .unwrap();
    wait_until(|| reached.iter().all(|flag| flag.load(Ordering::SeqCst))).await;
    eof[first].add_permits(1);
    let node = ["left_input", "right_input"][first];
    wait_until(|| job.status().nodes[node].ended).await;
    let epoch = assert_mixed_cut(&job, directory.path(), node, periodic).await;
    assert!(
        scenario
            .state
            .lock()
            .committed_epochs
            .contains(&epoch.as_u64())
    );
    if restore_ended {
        let opens = scenario.probes[first].opens.lock().len();
        assert_eq!(job.cancel().await.state, ContinuousJobState::Cancelled);
        drop(job);
        runner.shutdown().await.unwrap();
        runner = ContinuousRunner::new();
        reached[1 - first].store(false, Ordering::SeqCst);
        job = runner
            .start_checkpointed(
                spec(&scenario, 2403, &eof, &reached),
                checkpoint(directory.path()),
            )
            .await
            .unwrap();
        wait_until(|| job.status().nodes[node].ended && reached[1 - first].load(Ordering::SeqCst))
            .await;
        assert_eq!(scenario.probes[first].opens.lock().len(), opens);
        assert_mixed_cut(&job, directory.path(), node, periodic).await;
    }
    eof[1 - first].add_permits(1);
    assert_eq!(
        tokio::time::timeout(Duration::from_secs(5), job.wait())
            .await
            .unwrap()
            .state,
        ContinuousJobState::Completed
    );
    let terminal = manifest(
        directory.path(),
        job.status()
            .checkpoint
            .unwrap()
            .last_completed_epoch
            .unwrap(),
    )
    .await;
    assert!(terminal.sources().values().all(|entry| entry.ended));
    assert!(terminal.operators().values().all(|operator| {
        operator
            .progress
            .values()
            .all(|entry| entry.state == crate::ManifestIngressState::Ended)
    }));
    assert_eq!(scenario.state.lock().records, expected());
    drop(job);
    runner.shutdown().await.unwrap();
}

#[tokio::test]
async fn restored_checkpoint_captures_left_operator_ended_before_peer_eof() {
    mixed_cut(0, false, false).await;
}

#[tokio::test]
async fn restored_checkpoint_captures_right_operator_ended_before_peer_eof() {
    mixed_cut(1, false, false).await;
}

#[tokio::test]
async fn periodic_checkpoint_captures_left_operator_ended_before_peer_eof() {
    mixed_cut(0, true, false).await;
}

#[tokio::test]
async fn periodic_checkpoint_captures_right_operator_ended_before_peer_eof() {
    mixed_cut(1, true, false).await;
}

#[tokio::test]
async fn restored_ended_left_operator_keeps_acknowledging_until_terminal_commit() {
    mixed_cut(0, false, true).await;
}

#[tokio::test]
async fn restored_ended_right_operator_keeps_acknowledging_until_terminal_commit() {
    mixed_cut(1, false, true).await;
}
