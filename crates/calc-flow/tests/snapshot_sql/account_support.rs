use std::{
    collections::BTreeMap,
    path::Path,
    sync::{
        Arc, Mutex,
        atomic::{AtomicUsize, Ordering},
    },
    time::Duration,
};

use calc_flow::{
    Batch, BatchKind, BatchMetadata, CalcFlowError, Cursor, Edge, EdgeBudget, JsonMap,
    ManagedCheckpointRuntime, NativeWatermarkCapability, OperatorMetadata, PipelineBuilder, Port,
    PortEndpoint, ReplayPositioning, Result, SinkBinding, SnapshotSqlOperator, SourceBinding,
    SourceCapabilities, SourceDeliveryCapability, SourceEvent, SourceSchema, SqlOperator,
    StateBudget, StreamExecutionPlan, StreamRequirements, StreamRuntimeConfig, StreamSink,
    StreamSource, StreamingJob, StreamingRunner, UdfRegistrySnapshot, WatermarkPolicy,
};
use datafusion::arrow::{
    array::{ArrayRef, Float64Array, Int64Array, StringArray, TimestampMicrosecondArray},
    compute::concat_batches,
    datatypes::SchemaRef,
    record_batch::RecordBatch,
};
use tokio::sync::{Notify, Semaphore, mpsc};

pub const POSITIONS: &str =
    "SELECT account, stock, SUM(delta) AS position FROM input GROUP BY account, stock";
pub const LATEST: &str = "SELECT stock, LAST_VALUE(latest_price ORDER BY \"timestamp\") AS latest_price FROM input GROUP BY stock";
pub const VALUES: &str = "SELECT p.account, SUM(p.position * q.latest_price) AS market_value FROM p LEFT JOIN q ON p.stock = q.stock GROUP BY p.account ORDER BY p.account";
pub const DEADLINE: Duration = Duration::from_secs(20);

fn batch(columns: Vec<(&str, ArrayRef)>) -> Batch {
    Batch::table(
        vec![RecordBatch::try_from_iter(columns).unwrap()],
        BatchMetadata::default(),
    )
    .unwrap()
}

pub fn trade(account: &str, delta: i64) -> Batch {
    batch(vec![
        ("account", Arc::new(StringArray::from(vec![account]))),
        ("stock", Arc::new(StringArray::from(vec!["AAA"]))),
        ("delta", Arc::new(Int64Array::from(vec![delta]))),
    ])
}

pub fn quote(price: f64, timestamp: i64) -> Batch {
    batch(vec![
        ("stock", Arc::new(StringArray::from(vec!["AAA"]))),
        ("latest_price", Arc::new(Float64Array::from(vec![price]))),
        (
            "timestamp",
            Arc::new(TimestampMicrosecondArray::from(vec![timestamp]).with_timezone("UTC")),
        ),
    ])
}

fn port(name: &str, schema: &SchemaRef) -> Port {
    Port::new(
        name,
        BatchKind::Table,
        true,
        Some(schema.fields().iter().map(|f| f.as_ref().clone()).collect()),
    )
    .unwrap()
}

async fn aggregate(name: &str, query: &str, input: SchemaRef) -> SqlOperator {
    let operator = SqlOperator::new(name, query, vec!["input".into()], vec![]).unwrap();
    let output = operator
        .infer_schema(&BTreeMap::from([("input".into(), input.clone())]))
        .await
        .unwrap();
    operator
        .with_ports(vec![port("input", &input)], port("output", &output))
        .unwrap()
}

pub async fn plan() -> StreamExecutionPlan {
    let positions = aggregate(
        "positions",
        POSITIONS,
        trade("alice", 0).table_payload().unwrap().schema().clone(),
    )
    .await;
    let latest = aggregate(
        "latest",
        LATEST,
        quote(0.0, 0).table_payload().unwrap().schema().clone(),
    )
    .await;
    let schemas = BTreeMap::from([
        (
            "p".into(),
            positions.output_ports()[0].schema().unwrap().clone(),
        ),
        (
            "q".into(),
            latest.output_ports()[0].schema().unwrap().clone(),
        ),
    ]);
    let output = SqlOperator::new("oracle", VALUES, vec!["p".into(), "q".into()], vec![])
        .unwrap()
        .infer_schema(&schemas)
        .await
        .unwrap();
    let values = SnapshotSqlOperator::new("values", VALUES, vec!["p".into(), "q".into()], vec![])
        .unwrap()
        .with_ports(
            schemas
                .into_iter()
                .map(|(name, schema)| port(&name, &schema))
                .collect(),
            port("output", &output),
        )
        .unwrap();
    let mut builder = PipelineBuilder::new("account-values-recovery")
        .unwrap()
        .add_node("positions", Box::new(positions))
        .unwrap()
        .add_node("latest", Box::new(latest))
        .unwrap()
        .add_node("values", values)
        .unwrap();
    for (node, alias) in [("positions", "p"), ("latest", "q")] {
        builder = builder
            .connect(Edge::new(
                PortEndpoint::new(node, "output").unwrap(),
                PortEndpoint::new("values", alias).unwrap(),
            ))
            .unwrap();
    }
    builder
        .compile_stream(
            &UdfRegistrySnapshot::default(),
            &StreamRequirements::default(),
        )
        .unwrap()
}

#[derive(Default)]
pub struct SourceProbe {
    pub opened_at: Mutex<Vec<usize>>,
    pub reads: AtomicUsize,
    pub closes: AtomicUsize,
    pub drops: AtomicUsize,
    pub ended: Notify,
}

pub struct ReplaySource {
    history: Vec<Batch>,
    position: usize,
    permits: mpsc::UnboundedReceiver<()>,
    probe: Arc<SourceProbe>,
}

pub fn source(history: Vec<Batch>) -> (mpsc::UnboundedSender<()>, ReplaySource, Arc<SourceProbe>) {
    let (tx, permits) = mpsc::unbounded_channel();
    let probe = Arc::new(SourceProbe::default());
    (
        tx,
        ReplaySource {
            history,
            position: 0,
            permits,
            probe: probe.clone(),
        },
        probe,
    )
}

#[async_trait::async_trait]
impl StreamSource for ReplaySource {
    fn capabilities(&self) -> SourceCapabilities {
        SourceCapabilities {
            replay_positioning: ReplayPositioning::ExactPauseReportAndSeek,
            delivery: SourceDeliveryCapability::Lossless,
            max_batch_rows: 1,
            max_batch_bytes: 4096,
            schema: SourceSchema::DynamicOrUnknown,
            native_watermarks: NativeWatermarkCapability::NeverEmits,
        }
    }
    async fn open(&mut self, cursor: Option<Cursor>) -> Result<()> {
        self.position = cursor.map_or(0, |cursor| {
            usize::try_from(u64::from_be_bytes(cursor.order().try_into().unwrap())).unwrap()
        });
        self.probe.opened_at.lock().unwrap().push(self.position);
        Ok(())
    }
    async fn next(&mut self) -> Result<Option<SourceEvent>> {
        if self.permits.recv().await.is_none() {
            self.probe.ended.notify_one();
            return Ok(None);
        }
        let batch = self.history[self.position].clone();
        self.position += 1;
        self.probe.reads.fetch_add(1, Ordering::SeqCst);
        Ok(Some(SourceEvent::Data {
            batch,
            cursor: Cursor::unbound(
                u64::try_from(self.position).unwrap().to_be_bytes().to_vec(),
                JsonMap::new(),
            )
            .unwrap(),
        }))
    }
    async fn close(&mut self) -> Result<()> {
        self.probe.closes.fetch_add(1, Ordering::SeqCst);
        Ok(())
    }
}

impl Drop for ReplaySource {
    fn drop(&mut self) {
        self.probe.drops.fetch_add(1, Ordering::SeqCst);
    }
}

#[derive(Default)]
pub struct SinkProbe {
    pub entered: Notify,
    pub writes: AtomicUsize,
    pub closes: AtomicUsize,
    pub drops: AtomicUsize,
}

pub struct Sink {
    pub output: mpsc::UnboundedSender<Batch>,
    pub permits: Option<Arc<Semaphore>>,
    pub fail_at: Option<usize>,
    pub probe: Arc<SinkProbe>,
}

pub fn sink_fixture() -> (Sink, mpsc::UnboundedReceiver<Batch>, Arc<SinkProbe>) {
    let (output, rx) = mpsc::unbounded_channel();
    let probe = Arc::new(SinkProbe::default());
    (
        Sink {
            output,
            permits: None,
            fail_at: None,
            probe: probe.clone(),
        },
        rx,
        probe,
    )
}

#[async_trait::async_trait]
impl StreamSink for Sink {
    async fn open(&mut self) -> Result<()> {
        Ok(())
    }
    async fn write(&mut self, batch: &Batch) -> Result<()> {
        let write = self.probe.writes.fetch_add(1, Ordering::SeqCst) + 1;
        self.probe.entered.notify_one();
        if let Some(permits) = &self.permits {
            permits.acquire().await.unwrap().forget();
        }
        if self.fail_at == Some(write) {
            return Err(CalcFlowError::Internal {
                message: "valuation sink failed".into(),
            });
        }
        self.output.send(batch.clone()).unwrap();
        Ok(())
    }
    async fn close(&mut self) -> Result<()> {
        self.probe.closes.fetch_add(1, Ordering::SeqCst);
        Ok(())
    }
}

impl Drop for Sink {
    fn drop(&mut self) {
        self.probe.drops.fetch_add(1, Ordering::SeqCst);
    }
}

pub async fn start(
    root: &Path,
    p: ReplaySource,
    q: ReplaySource,
    sink: Sink,
    rows: usize,
) -> StreamingJob {
    StreamingRunner::new(
        plan().await,
        BTreeMap::from([
            (
                "positions.input".into(),
                SourceBinding::new(p)
                    .with_watermark_policy(WatermarkPolicy::Disabled { idle_timeout: None }),
            ),
            (
                "latest.input".into(),
                SourceBinding::new(q)
                    .with_watermark_policy(WatermarkPolicy::Disabled { idle_timeout: None }),
            ),
        ]),
        BTreeMap::from([(
            "output".into(),
            vec![SinkBinding::ordinary("sink", sink).unwrap()],
        )]),
        ManagedCheckpointRuntime::new(root).unwrap(),
    )
    .unwrap()
    .with_runtime_config(StreamRuntimeConfig {
        checkpoint_interval: Duration::from_secs(3600),
        edge_budget: EdgeBudget::new(rows, 1 << 20).unwrap(),
        sql_compact_state_budget: Some(StateBudget::new(100, 1 << 20).unwrap()),
        sql_snapshot_state_budget: Some(StateBudget::new(200, 2 << 20).unwrap()),
        ..StreamRuntimeConfig::default()
    })
    .unwrap()
    .start()
    .await
    .unwrap()
}

pub fn assert_value(actual: &Batch, expected: f64) {
    let payload = actual.table_payload().unwrap();
    let table = concat_batches(payload.schema(), payload.batches()).unwrap();
    assert_eq!(table.num_rows(), 1);
    assert_eq!(
        table.column_by_name("account").unwrap().as_ref(),
        &StringArray::from(vec!["alice"])
    );
    assert_eq!(
        table.column_by_name("market_value").unwrap().as_ref(),
        &Float64Array::from(vec![expected])
    );
}

pub fn assert_released(job: &StreamingJob, p: &SourceProbe, q: &SourceProbe, sink: &SinkProbe) {
    assert_eq!(job.status().task_count, 0);
    for source in [p, q] {
        assert_eq!(source.closes.load(Ordering::SeqCst), 1);
        assert_eq!(source.drops.load(Ordering::SeqCst), 1);
    }
    assert_eq!(sink.closes.load(Ordering::SeqCst), 1);
    assert_eq!(sink.drops.load(Ordering::SeqCst), 1);
}
