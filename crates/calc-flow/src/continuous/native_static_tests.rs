use crate::{
    Batch, BatchMetadata, Cursor, NativeWatermarkCapability, ProjectSpec, ProviderRegistry,
    ReplayPositioning, Result, SinkBinding, SourceBinding, SourceCapabilities,
    SourceDeliveryCapability, SourceEvent, SourceSchema, StreamRequirements, StreamSink,
    StreamSource, StreamingRunner, UdfRegistrySnapshot,
    array::{NativeArray, NativeTableArrayFactory},
    compile_stream_project_graph,
};
use async_trait::async_trait;
use datafusion::arrow::{
    array::Float32Array,
    datatypes::{DataType, Field, Schema},
    record_batch::RecordBatch,
};
use serde_json::json;
use std::{
    collections::{BTreeMap, VecDeque},
    sync::{
        Arc, Mutex,
        atomic::{AtomicUsize, Ordering},
    },
};

struct Rows {
    batches: VecDeque<Batch>,
    opened: Arc<AtomicUsize>,
}
#[async_trait]
impl StreamSource for Rows {
    fn capabilities(&self) -> SourceCapabilities {
        SourceCapabilities {
            replay_positioning: ReplayPositioning::Unsupported,
            delivery: SourceDeliveryCapability::Lossy,
            max_batch_rows: 2,
            max_batch_bytes: 1024,
            schema: SourceSchema::DynamicOrUnknown,
            native_watermarks: NativeWatermarkCapability::NeverEmits,
        }
    }
    async fn open(&mut self, _cursor: Option<Cursor>) -> Result<()> {
        self.opened.fetch_add(1, Ordering::SeqCst);
        Ok(())
    }
    async fn next(&mut self) -> Result<Option<SourceEvent>> {
        self.batches
            .pop_front()
            .map(|batch| {
                Ok(SourceEvent::Data {
                    batch,
                    cursor: Cursor::new("observations", vec![0], BTreeMap::new())?,
                })
            })
            .transpose()
    }
    async fn close(&mut self) -> Result<()> {
        Ok(())
    }
}
struct Output(Arc<Mutex<Vec<Batch>>>);
#[async_trait]
impl StreamSink for Output {
    async fn open(&mut self) -> Result<()> {
        Ok(())
    }
    async fn write(&mut self, batch: &Batch) -> Result<()> {
        self.0.lock().unwrap().push(batch.clone());
        Ok(())
    }
    async fn close(&mut self) -> Result<()> {
        Ok(())
    }
}
fn project() -> ProjectSpec {
    let fields = json!([{"name":"a","data_type":"float32","nullable":false},{"name":"b","data_type":"float32","nullable":false}]);
    let mut output = fields.as_array().unwrap().clone();
    output.push(json!({"name":"score","data_type":"float64","nullable":true}));
    let options = json!({"ir_version":2,"inputs":[{"name":"observations","kind":"table","columns":["a","b"]},{"name":"coefficients","kind":"array","binding":"static","dtype":"float64","shape":[2,1]}],"nodes":[{"id":"x","op":"input","name":"observations"},{"id":"w","op":"input","name":"coefficients"},{"id":"y","op":"matmul","left":"x","right":"w"}],"outputs":[{"name":"result","kind":"table","node":"y","attach_to":"observations","names":["score"]}]});
    serde_json::from_value(json!({"format_version":3,"id":"native-static-stream","name":"native-static-stream","runtime":{"mode":"stream","options":{}},"static_inputs":[{"name":"coefficients","kind":"array","mutability":"static","backend":"native","dtype":"float64","shape":[2,1]}],"graph":{"name":"native-static-stream","nodes":[{"id":"calc","operator":{"kind":"external","provider":"native","name":"table_array","version":"1","options":options},"input_ports":[{"name":"observations","kind":"table","schema":fields},{"name":"coefficients","kind":"array"}],"output_ports":[{"name":"result","kind":"table","schema":output}]}]}})).unwrap()
}
fn input() -> Batch {
    let schema = Arc::new(Schema::new(vec![
        Field::new("a", DataType::Float32, false),
        Field::new("b", DataType::Float32, false),
    ]));
    let rows = RecordBatch::try_new(
        schema,
        vec![
            Arc::new(Float32Array::from(vec![1.0, 3.0])),
            Arc::new(Float32Array::from(vec![2.0, 4.0])),
        ],
    )
    .unwrap();
    Batch::table(vec![rows], BatchMetadata::default()).unwrap()
}

fn plan() -> crate::StreamExecutionPlan {
    let registry = ProviderRegistry::default();
    registry
        .register_stream(
            "native",
            "table_array",
            "1",
            Arc::new(NativeTableArrayFactory::default()),
        )
        .unwrap();
    compile_stream_project_graph(
        &project(),
        &registry,
        &UdfRegistrySnapshot::default(),
        &StreamRequirements::default(),
    )
    .unwrap()
}
fn startup(
    plan: crate::StreamExecutionPlan,
    opened: Arc<AtomicUsize>,
) -> (StreamingRunner, std::sync::Weak<NativeArray>) {
    let sources = BTreeMap::from([(
        "observations".into(),
        SourceBinding::new(Rows {
            batches: VecDeque::from([input()]),
            opened,
        })
        .with_watermark_policy(crate::WatermarkPolicy::Disabled { idle_timeout: None }),
    )]);
    let sinks = BTreeMap::from([(
        "result".into(),
        vec![SinkBinding::ordinary("sink", Output(Arc::default())).unwrap()],
    )]);
    let weights = Arc::new(NativeArray::from_f64(vec![2.0, 3.0], vec![2, 1], None).unwrap());
    let weak = Arc::downgrade(&weights);
    let statics = BTreeMap::from([(
        "coefficients".into(),
        Batch::external(weights, BatchMetadata::default()).unwrap(),
    )]);
    (
        StreamingRunner::without_checkpoints(plan, sources, sinks)
            .unwrap()
            .with_static_inputs(statics)
            .unwrap(),
        weak,
    )
}
async fn signal(receiver: tokio::sync::oneshot::Receiver<()>) {
    tokio::time::timeout(std::time::Duration::from_secs(3), receiver)
        .await
        .unwrap()
        .unwrap();
}
#[tokio::test]
async fn test_aborted_native_startup_cleanup_waits_for_running_worker_and_owned_input() {
    let plan = plan();
    let permits = plan.native_static_plan().test_permits();
    let (entered, ready) = tokio::sync::oneshot::channel();
    let (cancelled, observed) = tokio::sync::oneshot::channel();
    let (release, held) = std::sync::mpsc::channel();
    plan.native_static_plan().test_work(move |check| {
        entered.send(()).unwrap();
        let until = std::time::Instant::now() + std::time::Duration::from_secs(3);
        while std::time::Instant::now() < until {
            if let Err(error) = check() {
                cancelled.send(()).unwrap();
                held.recv_timeout(std::time::Duration::from_secs(3))
                    .unwrap();
                return Err(error);
            }
            std::thread::yield_now();
        }
        panic!("native worker did not observe waiter cancellation");
    });
    let opened = Arc::new(AtomicUsize::new(0));
    let (runner, input) = startup(plan, Arc::clone(&opened));
    let (start, cleanup) = runner.start_with_cleanup();
    let task = tokio::spawn(start);
    signal(ready).await;
    task.abort();
    assert!(task.await.unwrap_err().is_cancelled());
    signal(observed).await;
    let mut cleanup = Box::pin(cleanup);
    assert!(
        tokio::time::timeout(std::time::Duration::from_millis(10), &mut cleanup)
            .await
            .is_err(),
        "cleanup returned while native worker still owned input"
    );
    assert!(input.upgrade().is_some());
    assert_eq!(permits.available_permits(), 1);
    assert_eq!(opened.load(Ordering::SeqCst), 0);
    release.send(()).unwrap();
    tokio::time::timeout(std::time::Duration::from_secs(3), cleanup)
        .await
        .unwrap()
        .unwrap();
    assert!(input.upgrade().is_none());
    assert_eq!(permits.available_permits(), 2);
}
#[tokio::test]
async fn test_aborted_queued_native_startup_cleanup_releases_owned_input_without_worker() {
    let plan = plan();
    let permits = plan.native_static_plan().test_permits();
    let held = Arc::clone(&permits).acquire_many_owned(2).await.unwrap();
    let queued = plan.native_static_plan().test_admission();
    let called = Arc::new(AtomicUsize::new(0));
    let flag = Arc::clone(&called);
    plan.native_static_plan().test_work(move |_| {
        flag.fetch_add(1, Ordering::SeqCst);
        Ok(())
    });
    let opened = Arc::new(AtomicUsize::new(0));
    let (runner, input) = startup(plan, Arc::clone(&opened));
    let (start, cleanup) = runner.start_with_cleanup();
    let task = tokio::spawn(start);
    signal(queued).await;
    assert!(input.upgrade().is_some());
    let mut cleanup = Box::pin(cleanup);
    assert!(
        tokio::time::timeout(std::time::Duration::from_millis(10), &mut cleanup)
            .await
            .is_err(),
        "cleanup returned while queued startup still owned input"
    );
    task.abort();
    assert!(task.await.unwrap_err().is_cancelled());
    tokio::time::timeout(std::time::Duration::from_secs(3), cleanup)
        .await
        .unwrap()
        .unwrap();
    assert!(input.upgrade().is_none());
    assert_eq!(called.load(Ordering::SeqCst), 0);
    assert_eq!(opened.load(Ordering::SeqCst), 0);
    assert_eq!(permits.available_permits(), 0);
    drop(held);
    assert_eq!(permits.available_permits(), 2);
}

struct PendingRows(Arc<AtomicUsize>);
#[async_trait]
impl StreamSource for PendingRows {
    fn capabilities(&self) -> SourceCapabilities {
        SourceCapabilities {
            replay_positioning: ReplayPositioning::ExactPauseReportAndSeek,
            delivery: SourceDeliveryCapability::Lossless,
            max_batch_rows: 2,
            max_batch_bytes: 1024,
            schema: SourceSchema::DynamicOrUnknown,
            native_watermarks: NativeWatermarkCapability::NeverEmits,
        }
    }
    async fn open(&mut self, _cursor: Option<Cursor>) -> Result<()> {
        self.0.fetch_add(1, Ordering::SeqCst);
        Ok(())
    }
    async fn next(&mut self) -> Result<Option<SourceEvent>> {
        std::future::pending().await
    }
    async fn close(&mut self) -> Result<()> {
        Ok(())
    }
}
struct TransactionalOutput;
#[async_trait]
impl super::TransactionalStreamSink for TransactionalOutput {
    async fn open(&mut self) -> Result<()> {
        Ok(())
    }
    async fn begin_epoch(&mut self, _epoch: crate::Epoch) -> Result<()> {
        Ok(())
    }
    async fn write(&mut self, _batch: &Batch) -> Result<()> {
        Ok(())
    }
    async fn pre_commit(&mut self, _epoch: crate::Epoch) -> Result<crate::JsonMap> {
        Ok(BTreeMap::new())
    }
    async fn commit(&mut self, _epoch: crate::Epoch, _state: &crate::JsonMap) -> Result<()> {
        Ok(())
    }
    async fn abort(&mut self, _epoch: crate::Epoch, _state: Option<&crate::JsonMap>) -> Result<()> {
        Ok(())
    }
    async fn recover(&mut self, _recovery: &super::SinkRecovery) -> Result<()> {
        Ok(())
    }
    async fn close(&mut self) -> Result<()> {
        Ok(())
    }
}
fn checkpointed(
    root: &std::path::Path,
    opened: Arc<AtomicUsize>,
    values: Vec<f64>,
) -> StreamingRunner {
    let sources = BTreeMap::from([(
        "observations".into(),
        SourceBinding::new(PendingRows(opened))
            .with_watermark_policy(super::WatermarkPolicy::Disabled { idle_timeout: None }),
    )]);
    let sinks = BTreeMap::from([(
        "result".into(),
        vec![SinkBinding::transactional("sink", TransactionalOutput).unwrap()],
    )]);
    let weights = NativeArray::from_f64(values, vec![2, 1], None).unwrap();
    let statics = BTreeMap::from([(
        "coefficients".into(),
        Batch::external(Arc::new(weights), BatchMetadata::default()).unwrap(),
    )]);
    StreamingRunner::new(
        plan(),
        sources,
        sinks,
        super::ManagedCheckpointRuntime::new(root).unwrap(),
    )
    .unwrap()
    .with_static_inputs(statics)
    .unwrap()
}
#[tokio::test]
async fn test_native_static_recovery_checks_digest_and_descriptor_before_source_open() {
    let directory = tempfile::tempdir().unwrap();
    let opened = Arc::new(AtomicUsize::new(0));
    let job = checkpointed(directory.path(), Arc::clone(&opened), vec![2.0, 3.0])
        .start()
        .await
        .unwrap();
    assert_eq!(opened.load(Ordering::SeqCst), 1);
    let epoch = tokio::time::timeout(std::time::Duration::from_secs(10), job.trigger_checkpoint())
        .await
        .unwrap()
        .unwrap();
    let outcome = tokio::time::timeout(std::time::Duration::from_secs(10), job.shutdown())
        .await
        .unwrap();
    assert!(outcome.completed_epoch >= Some(epoch));
    let changed_opened = Arc::new(AtomicUsize::new(0));
    let error = checkpointed(
        directory.path(),
        Arc::clone(&changed_opened),
        vec![4.0, 3.0],
    )
    .start()
    .await
    .unwrap_err();
    let crate::CalcFlowError::Streaming(error) = error else {
        panic!("expected the managed checkpoint boundary")
    };
    assert_eq!(
        error.category(),
        super::StreamingErrorCategory::CheckpointMismatch
    );
    assert!(
        error
            .to_string()
            .starts_with("static_inputs.coefficients.digest:")
    );
    assert_eq!(changed_opened.load(Ordering::SeqCst), 0);
    let identical_opened = Arc::new(AtomicUsize::new(0));
    let restored = checkpointed(
        directory.path(),
        Arc::clone(&identical_opened),
        vec![2.0, 3.0],
    )
    .start()
    .await
    .unwrap();
    let outcome = tokio::time::timeout(std::time::Duration::from_secs(10), restored.wait())
        .await
        .unwrap();
    assert!(outcome.completed_epoch >= Some(epoch));
    assert_eq!(identical_opened.load(Ordering::SeqCst), 0);
}
