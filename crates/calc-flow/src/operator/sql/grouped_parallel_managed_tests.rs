use super::*;
use crate::{
    BatchKind, CheckpointManifest, EdgeBudget, Epoch, PipelineBuilder, Port, SqlOperator,
    StreamRequirements, StreamRuntimeConfig, UdfRegistry,
    runtime::streaming::{
        checkpoint::ManagedCheckpointRuntime,
        job::{
            ContinuousJobSpec, M2DeliveryMode, NamedSinkBinding, NamedSourceBinding,
            OrdinarySinkBinding, OrdinaryStreamSink,
        },
        runner::{CheckpointRuntimeSpec, ContinuousJob, ContinuousJobState, ContinuousRunner},
        source_task::{Cursor, SourceBinding, SourceCapabilities, SourceEvent, StreamSource},
    },
};
use async_trait::async_trait;
use parking_lot::Mutex as ParkingMutex;
use std::path::Path;
use tokio::sync::Notify;

const NODE: &str = "managed-parallel-sum-barrier";
const ROWS: usize = 4 * CHUNK_ROWS;
const LIMIT: Duration = Duration::from_secs(10);

struct Source {
    start: Arc<Notify>,
    next: u8,
    end: u8,
    terminal: bool,
}

#[async_trait]
impl StreamSource for Source {
    async fn open(&mut self, cursor: Option<Cursor>) -> Result<()> {
        self.next = cursor.map_or(0, |cursor| cursor.order()[0]);
        Ok(())
    }

    async fn next(&mut self) -> Result<Option<SourceEvent>> {
        if self.next == self.end {
            return if self.terminal {
                Ok(None)
            } else {
                std::future::pending().await
            };
        }
        self.start.notified().await;
        self.next += 1;
        Ok(Some(SourceEvent::Data {
            batch: source(ROWS),
            cursor: Cursor::unbound(vec![self.next], JsonMap::new())?,
        }))
    }

    async fn close(&mut self) -> Result<()> {
        Ok(())
    }

    fn capabilities(&self) -> SourceCapabilities {
        SourceCapabilities {
            replayable: true,
            max_batch_rows: ROWS,
            max_batch_bytes: 8 << 20,
        }
    }
}

struct Sink(Arc<ParkingMutex<Vec<Batch>>>);

#[async_trait]
impl OrdinaryStreamSink for Sink {
    async fn open(&mut self) -> Result<()> {
        Ok(())
    }

    async fn write(&mut self, batch: &Batch) -> Result<()> {
        self.0.lock().push(batch.clone());
        Ok(())
    }

    async fn close(&mut self) -> Result<()> {
        Ok(())
    }
}

fn register() -> (Registration, mpsc::Receiver<usize>) {
    let (entered, receiver) = mpsc::channel();
    let (completed, _) = mpsc::channel();
    let probe = Arc::new(Probe {
        entered,
        completed,
        gates: std::array::from_fn(|_| Gate::default()),
        active: AtomicUsize::new(0),
        maximum: AtomicUsize::new(0),
        fault: Mutex::new(None),
        merge_calls: AtomicUsize::new(0),
    });
    PROBES.lock().unwrap().insert(NODE.into(), probe.clone());
    (
        Registration {
            name: NODE.into(),
            probe,
        },
        receiver,
    )
}

async fn plan() -> crate::StreamExecutionPlan {
    let schema = source(1).table_payload().unwrap().schema().clone();
    let operator = SqlOperator::new(
        NODE,
        "SELECT key, SUM(value) AS total FROM events GROUP BY key",
        vec!["events".into()],
        vec![],
    )
    .unwrap();
    let output = operator
        .infer_schema(&BTreeMap::from([("events".into(), schema.clone())]))
        .await
        .unwrap();
    let operator = operator
        .with_ports(
            vec![Port::with_schema_ref("events", BatchKind::Table, true, Some(schema)).unwrap()],
            Port::with_schema_ref("output", BatchKind::Table, true, Some(output)).unwrap(),
        )
        .unwrap();
    PipelineBuilder::new("managed-parallel-sum")
        .unwrap()
        .add_node(NODE, Box::new(operator))
        .unwrap()
        .compile_stream(
            &UdfRegistry::new().snapshot(),
            &StreamRequirements::default(),
        )
        .unwrap()
}

async fn spec(
    start: Arc<Notify>,
    records: Arc<ParkingMutex<Vec<Batch>>>,
    terminal: bool,
) -> ContinuousJobSpec {
    let plan = plan().await;
    ContinuousJobSpec {
        context: StreamJobContext::new(
            811,
            plan.fingerprint(),
            JsonMap::new(),
            None,
            CancellationToken::new(),
        ),
        plan,
        sources: vec![NamedSourceBinding {
            binding_id: "events".into(),
            binding: SourceBinding::new(
                Box::new(Source {
                    start,
                    next: 0,
                    end: if terminal { 2 } else { 1 },
                    terminal,
                }),
                None,
                0,
            )
            .unwrap(),
        }],
        sinks: vec![NamedSinkBinding {
            output_id: "output".into(),
            sink_id: "sink".into(),
            binding: OrdinarySinkBinding::new(Box::new(Sink(records))),
        }],
        edge_budget: EdgeBudget::new(ROWS, 8 << 20).unwrap(),
        delivery_mode: M2DeliveryMode::ProcessLocalOrdered,
        static_inputs: crate::static_input::PreparedStaticInputs::default(),
    }
}

fn checkpoint(root: &Path) -> CheckpointRuntimeSpec {
    ContinuousRunner::managed_checkpoint_for_test(
        ManagedCheckpointRuntime::new(root).unwrap(),
        StreamRuntimeConfig {
            checkpoint_interval: Duration::from_secs(3_600),
            checkpoint_timeout: LIMIT,
            ..StreamRuntimeConfig::default()
        },
    )
    .unwrap()
}

async fn entered(receiver: &mpsc::Receiver<usize>) {
    let mut ordinals = Vec::new();
    tokio::time::timeout(LIMIT, async {
        while ordinals.len() != 2 {
            if let Ok(ordinal) = receiver.try_recv() {
                ordinals.push(ordinal);
            }
            tokio::task::yield_now().await;
        }
    })
    .await
    .unwrap();
    ordinals.sort_unstable();
    assert_eq!(ordinals, [0, 1]);
}

fn attach(service: &TestService, job: &ContinuousJob, cpus: usize) {
    let owner = job.gather_owner_for_test();
    service.attach_owner(&owner);
    assert_eq!(owner.parallelism(), cpus);
}

async fn checkpoint_behind_workers(job: &ContinuousJob, registration: &Registration) -> Epoch {
    let cut = job.trigger_checkpoint();
    tokio::pin!(cut);
    tokio::time::timeout(LIMIT, async {
        loop {
            if job
                .status()
                .checkpoint
                .is_some_and(|status| status.source_acks == 1)
            {
                break;
            }
            tokio::select! {
                result = &mut cut => panic!("checkpoint overtook held SUM units: {result:?}"),
                () = tokio::task::yield_now() => {},
            }
        }
    })
    .await
    .unwrap();
    let status = job.status().checkpoint.unwrap();
    assert_eq!(status.operator_acks, 0);
    assert_eq!(status.sink_precommit_acks, 0);
    assert!(futures::poll!(&mut cut).is_pending());
    assert_eq!(registration.probe.active.load(Ordering::SeqCst), 2);
    for gate in &registration.probe.gates {
        gate.release();
    }
    tokio::time::timeout(LIMIT, &mut cut)
        .await
        .unwrap()
        .unwrap()
}

fn assert_totals(records: &ParkingMutex<Vec<Batch>>, batches: i64) {
    let records = records.lock();
    assert_eq!(records.len(), 1);
    let payload = records[0].table_payload().unwrap();
    assert_eq!(payload.batches().len(), 1);
    let output = &payload.batches()[0];
    assert_eq!(output.num_rows(), 16);
    let values = output
        .column(1)
        .as_any()
        .downcast_ref::<Int64Array>()
        .unwrap();
    assert_eq!(
        values.values().as_ref(),
        vec![i64::try_from(ROWS / 16).unwrap() * batches; 16]
    );
}

async fn capture_held_batch(
    root: &Path,
    registration: &Registration,
    receiver: &mpsc::Receiver<usize>,
    service: &TestService,
) {
    let start = Arc::new(Notify::new());
    let records = Arc::new(ParkingMutex::new(Vec::new()));
    let mut runner = ContinuousRunner::new();
    let job = runner
        .start_checkpointed(
            spec(start.clone(), records.clone(), false).await,
            checkpoint(root),
        )
        .await
        .unwrap();
    attach(service, &job, 2);
    start.notify_one();
    entered(receiver).await;
    assert!(records.lock().is_empty());
    assert_eq!(
        checkpoint_behind_workers(&job, registration).await,
        Epoch::INITIAL
    );
    assert_totals(&records, 1);
    assert_eq!(registration.probe.active.load(Ordering::SeqCst), 0);
    assert_eq!(job.cancel().await.state, ContinuousJobState::Cancelled);
    assert!(job.status().tasks.is_empty());
    drop(job);
    runner.shutdown().await.unwrap();
    assert_eq!(runner.registry_counts(), (0, 0));
    assert_eq!(service.available_capacity(), (2, 1, 0));
}

async fn assert_manifest(root: &Path) {
    let bytes = tokio::fs::read(root.join("manifests/manifest-00000000000000000001.json"))
        .await
        .unwrap();
    let manifest = CheckpointManifest::from_bytes(&bytes).unwrap();
    assert_eq!(
        manifest.sources()["events"].cursor.as_ref().unwrap().order,
        "01"
    );
    assert_eq!(
        manifest.operators()[NODE].inline_metadata["state_layout"],
        serde_json::json!(3)
    );
    assert_eq!(
        manifest.operators()[NODE].inline_metadata["rows"],
        serde_json::json!(ROWS)
    );
    assert!(!manifest.sources()["events"].ended);
}

async fn restore_one_cpu(root: &Path, receiver: &mpsc::Receiver<usize>, service: &TestService) {
    let start = Arc::new(Notify::new());
    let records = Arc::new(ParkingMutex::new(Vec::new()));
    let mut runner = ContinuousRunner::new();
    let job = runner
        .start_checkpointed(
            spec(start.clone(), records.clone(), true).await,
            checkpoint(root),
        )
        .await
        .unwrap();
    attach(service, &job, 1);
    assert_eq!(job.status().sources["events"].durable_order, Some(vec![1]));
    start.notify_one();
    let outcome = tokio::time::timeout(LIMIT, job.wait()).await.unwrap();
    assert_eq!(
        outcome.state,
        ContinuousJobState::Completed,
        "{:?}",
        outcome.errors
    );
    assert!(outcome.errors.is_empty());
    assert_totals(&records, 2);
    assert!(
        receiver.try_recv().is_err(),
        "one CPU must not launch local SUM units"
    );
    assert!(job.status().tasks.is_empty());
    drop(job);
    runner.shutdown().await.unwrap();
    assert_eq!(runner.registry_counts(), (0, 0));
    assert_eq!(service.available_capacity(), (1, 1, 0));
}

#[test]
fn test_managed_barrier_waits_for_parallel_sum_and_restores_with_one_cpu() {
    let directory = tempfile::tempdir().unwrap();
    let (registration, receiver) = register();
    let parallel = TestService::new(2, 1).unwrap();
    let serial = TestService::new(1, 1).unwrap();
    let executor = tokio::runtime::Builder::new_current_thread()
        .enable_all()
        .build()
        .unwrap();
    executor.block_on(async {
        capture_held_batch(directory.path(), &registration, &receiver, &parallel).await;
        assert_manifest(directory.path()).await;
        restore_one_cpu(directory.path(), &receiver, &serial).await;
    });
}
