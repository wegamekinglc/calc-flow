// Native companion to benchmarks/engine_stream.py, ready sources through Arrow table assembly.
use async_trait::async_trait;
use calc_flow::{
    Batch, BatchMetadata, Cursor, DeliveryGuarantee, EdgeBudget, EventTime, JobState,
    ManagedCheckpointRuntime, NativeWatermarkCapability, ProviderRegistry, ReplayPositioning,
    Result, SinkBinding, SourceBinding, SourceCapabilities, SourceDeliveryCapability, SourceEvent,
    SourceSchema, StreamRequirements, StreamRuntimeConfig, StreamSink, StreamSource,
    StreamingRunner, UdfRegistry, WatermarkPolicy, compile_stream_project_graph,
};
use datafusion::arrow::{
    array::{Array, Float64Array, StringArray, TimestampMicrosecondArray, UInt64Array},
    datatypes::{DataType, Field, Schema, SchemaRef, TimeUnit},
    record_batch::RecordBatch,
};
use serde_json::json;
use std::{
    collections::BTreeMap,
    sync::{Arc, Mutex},
    time::{Duration, Instant},
};
use tokio::sync::{mpsc, oneshot, watch};

const ROWS: usize = 10_000_000;
const CHUNK: usize = 1_000_000;
const KEYS: usize = 64;
const BASE: i64 = 1_767_225_600_000_000;

fn time(row: usize) -> i64 {
    BASE + (row / KEYS) as i64 * 1_000_000
}
fn price(row: usize) -> f64 {
    100.0 + (row % 257) as f64 / 8.0 + (row % KEYS) as f64 / 8.0
}

fn fixture() -> (SchemaRef, Vec<Batch>) {
    let schema = Arc::new(Schema::new(vec![
        Field::new(
            "event_time",
            DataType::Timestamp(TimeUnit::Microsecond, Some("UTC".into())),
            false,
        ),
        Field::new("sequence", DataType::UInt64, false),
        Field::new("symbol", DataType::Utf8, false),
        Field::new("price", DataType::Float64, false),
    ]));
    let names = (0..KEYS).map(|k| format!("S{k:03}")).collect::<Vec<_>>();
    let batches = (0..ROWS)
        .step_by(CHUNK)
        .map(|start| {
            let rows = start..start + CHUNK;
            let record = RecordBatch::try_new(
                schema.clone(),
                vec![
                    Arc::new(
                        TimestampMicrosecondArray::from_iter_values(rows.clone().map(time))
                            .with_timezone("UTC"),
                    ),
                    Arc::new(UInt64Array::from_iter_values(
                        rows.clone().map(|r| r as u64),
                    )),
                    Arc::new(StringArray::from_iter_values(
                        rows.clone().map(|r| names[r % KEYS].as_str()),
                    )),
                    Arc::new(Float64Array::from_iter_values(rows.map(price))),
                ],
            )
            .unwrap();
            Batch::table(vec![record], BatchMetadata::default()).unwrap()
        })
        .collect();
    (schema, batches)
}

struct Source {
    schema: SchemaRef,
    events: mpsc::Receiver<SourceEvent>,
    ready: Option<oneshot::Sender<()>>,
}
#[async_trait]
impl StreamSource for Source {
    fn capabilities(&self) -> SourceCapabilities {
        SourceCapabilities {
            replay_positioning: ReplayPositioning::Unsupported,
            delivery: SourceDeliveryCapability::Lossy,
            max_batch_rows: CHUNK,
            max_batch_bytes: 32 << 20,
            schema: SourceSchema::Exact(self.schema.clone()),
            native_watermarks: NativeWatermarkCapability::EmitsNative,
        }
    }
    async fn open(&mut self, cursor: Option<Cursor>) -> Result<()> {
        assert!(cursor.is_none());
        Ok(())
    }
    async fn next(&mut self) -> Result<Option<SourceEvent>> {
        if let Some(tx) = self.ready.take() {
            tx.send(()).unwrap();
        }
        Ok(self.events.recv().await)
    }
    async fn close(&mut self) -> Result<()> {
        Ok(())
    }
}

struct Sink {
    rows: watch::Sender<usize>,
    records: Arc<Mutex<Vec<RecordBatch>>>,
}
#[async_trait]
impl StreamSink for Sink {
    async fn open(&mut self) -> Result<()> {
        Ok(())
    }
    async fn write(&mut self, batch: &Batch) -> Result<()> {
        self.records
            .lock()
            .unwrap()
            .extend_from_slice(batch.table_payload()?.batches());
        self.rows.send_modify(|n| *n += batch.num_rows());
        Ok(())
    }
    async fn close(&mut self) -> Result<()> {
        Ok(())
    }
}

#[tokio::main(flavor = "multi_thread", worker_threads = 4)]
async fn main() {
    let workers = tokio::runtime::Handle::current().metrics().num_workers();
    let cpus = std::thread::available_parallelism().unwrap().get();
    assert_eq!(workers, 4);
    assert!((2..=4).contains(&cpus));
    let affinity = std::fs::read_to_string("/proc/self/status").unwrap();
    let affinity = affinity
        .lines()
        .find_map(|line| line.strip_prefix("Cpus_allowed_list:"))
        .unwrap()
        .trim();
    let (schema, batches) = fixture();
    let project = serde_json::from_str(include_str!("job-project.json")).unwrap();
    let plan = compile_stream_project_graph(
        &project,
        &ProviderRegistry::default(),
        &UdfRegistry::new().snapshot(),
        &StreamRequirements {
            delivery: BTreeMap::from([("output".into(), DeliveryGuarantee::BestEffort)]),
        },
    )
    .unwrap();
    let mut inputs = BTreeMap::new();
    let mut senders = Vec::new();
    let mut ready = Vec::new();
    for name in ["reference.input", "quotes.input"] {
        let (tx, events) = mpsc::channel(1);
        let (signal, rx) = oneshot::channel();
        inputs.insert(
            name.to_owned(),
            SourceBinding::new(Source {
                schema: schema.clone(),
                events,
                ready: Some(signal),
            })
            .with_watermark_policy(WatermarkPolicy::SourceProvided),
        );
        senders.push(tx);
        ready.push(rx);
    }
    let (rows, mut progress) = watch::channel(0);
    let records = Arc::new(Mutex::new(Vec::new()));
    let sink = SinkBinding::ordinary(
        "suite",
        Sink {
            rows,
            records: records.clone(),
        },
    )
    .unwrap();
    let scratch = tempfile::tempdir().unwrap();
    let runner = StreamingRunner::new(
        plan,
        inputs,
        BTreeMap::from([("output".into(), vec![sink])]),
        ManagedCheckpointRuntime::new(scratch.path()).unwrap(),
    )
    .unwrap()
    .with_runtime_config(StreamRuntimeConfig {
        checkpointing: true,
        checkpoint_interval: Duration::from_secs(86400),
        edge_budget: EdgeBudget::new(CHUNK, 64 << 20).unwrap(),
        ..Default::default()
    })
    .unwrap();
    let job = runner.start().await.unwrap();
    for rx in ready {
        rx.await.unwrap();
    }
    let envelope = |i: usize| {
        let end = (i + 1) * CHUNK;
        Cursor::unbound(
            (end as u64).to_be_bytes().to_vec(),
            BTreeMap::from([
                ("rows".to_owned(), json!(end)),
                ("event_index".to_owned(), json!(2 * i + 1)),
            ]),
        )
        .unwrap()
    };
    let events = batches
        .into_iter()
        .enumerate()
        .map(|(i, batch)| {
            let data = SourceEvent::Data {
                batch,
                cursor: envelope(i),
            };
            let wm = SourceEvent::Watermark(EventTime::from_micros(time((i + 1) * CHUNK - 1) + 1));
            (data, wm)
        })
        .collect::<Vec<_>>();
    let stamp = || {
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_nanos()
    };
    let epoch_start = stamp();
    let start = Instant::now();
    let publish = async {
        for (i, (data, wm)) in events.iter().enumerate() {
            for event in [data, wm] {
                for sender in &senders {
                    sender.send(event.clone()).await.unwrap();
                }
            }
            progress
                .wait_for(|rows| *rows >= (i + 1) * CHUNK)
                .await
                .unwrap();
        }
    };
    tokio::select! {
        () = publish => (),
        outcome = job.wait() => panic!("job terminated during publication: {outcome:?}"),
    }
    let batches = std::mem::take(&mut *records.lock().unwrap());
    let chunks = batches.len();
    let table = Batch::table(batches, BatchMetadata::default()).unwrap();
    let seconds = start.elapsed().as_secs_f64();
    let epoch_end = stamp();
    drop(senders);
    let outcome = job.wait().await;
    assert_eq!(outcome.state, JobState::Completed, "{outcome:?}");
    let stats = job.stream_asof_join_status();
    assert_eq!(stats.len(), 1);
    let status = stats.values().next().unwrap();
    assert_eq!(
        (
            status.left.accepted_rows,
            status.right.accepted_rows,
            status.matched_rows,
            status.evicted_right_rows,
            status.emitted_left_rows
        ),
        (
            ROWS as u64,
            ROWS as u64,
            ROWS as u64,
            ROWS as u64,
            ROWS as u64
        )
    );
    assert_eq!(
        (
            status.pending_left_rows,
            status.retained_right_rows,
            status.state_rows,
            status.unmatched_rows,
            status.workspace_limit_failures,
            status.state_limit_failures
        ),
        (0, 0, 0, 0, 0, 0)
    );
    let mut n = 0;
    for record in table.table_payload().unwrap().batches() {
        let seq = record
            .column_by_name("sequence")
            .unwrap()
            .as_any()
            .downcast_ref::<UInt64Array>()
            .unwrap();
        let value = record
            .column_by_name("value")
            .unwrap()
            .as_any()
            .downcast_ref::<Float64Array>()
            .unwrap();
        assert_eq!((seq.null_count(), value.null_count()), (0, 0));
        for i in 0..record.num_rows() {
            assert_eq!(seq.value(i), n as u64);
            assert_eq!(value.value(i), price(n));
            n += 1;
        }
    }
    assert_eq!(n, ROWS);
    println!(
        "{}",
        json!({"trace_outer":{"start":epoch_start,"end":epoch_end,"pid":std::process::id()},"seconds":seconds,"validated_all_rows":true,"output_rows":n,"keys":KEYS,
        "workers":workers,"cpus":cpus,"affinity":affinity,"emitted_batches":chunks,"closed":true,"state":status})
    );
}
