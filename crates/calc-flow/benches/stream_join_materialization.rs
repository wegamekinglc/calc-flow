//! Bounded Join materialization: wide payload, fan-out and a real slow sink.
//! The native measurement boundary excludes setup, preload and row oracles.

use async_trait::async_trait;
use calc_flow::{
    Batch, BatchMetadata, CancellationToken, EdgeBudget, EdgeCollector, EdgeSender,
    JoinStateLimits, JoinTimeBounds, JsonMap, OperatorMetadata, StreamCollector, StreamJobContext,
    StreamJoinOperator, StreamJoinSpec, StreamMessage, StreamOperator, StreamOperatorContext,
    edge_channel,
};
use datafusion::arrow::{
    array::{Int64Array, StringArray, TimestampMicrosecondArray},
    datatypes::{DataType, Field, Schema, TimeUnit},
    record_batch::RecordBatch,
};
use serde::{Deserialize, Serialize};
use serde_json::{Value, json};
use std::{
    io::{self, BufRead, Write},
    sync::{
        Arc,
        atomic::{AtomicBool, AtomicU64, Ordering},
    },
    time::{Duration, Instant},
};

fn rss() -> u64 {
    std::fs::read_to_string("/proc/self/status")
        .unwrap_or_default()
        .lines()
        .find_map(|line| {
            line.strip_prefix("VmRSS:")
                .map(|s| s.split_whitespace().next().unwrap().parse::<u64>().unwrap() * 1024)
        })
        .unwrap_or(0)
}

#[derive(Clone, Copy, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct Config {
    fan: usize,
    keys: usize,
    width: usize,
    incoming: usize,
    delay_ms: u64,
}

fn row_id(value: usize) -> i64 {
    i64::try_from(value).expect("bounded workload row id fits i64")
}

fn schema() -> Arc<Schema> {
    Arc::new(Schema::new(vec![
        Field::new("key", DataType::Int64, false),
        Field::new(
            "ts",
            DataType::Timestamp(TimeUnit::Microsecond, None),
            false,
        ),
        Field::new("id", DataType::Int64, false),
        Field::new("payload", DataType::Utf8, false),
    ]))
}

fn batch(rows: usize, keys: usize, width: usize, side: char) -> Batch {
    let payload = side.to_string().repeat(width);
    Batch::table(
        vec![
            RecordBatch::try_new(
                schema(),
                vec![
                    Arc::new(Int64Array::from_iter_values(
                        (0..rows).map(|i| row_id(i % keys)),
                    )),
                    Arc::new(TimestampMicrosecondArray::from(vec![100_000_000_i64; rows])),
                    Arc::new(Int64Array::from_iter_values((0..rows).map(row_id))),
                    Arc::new(StringArray::from_iter_values(
                        (0..rows).map(|_| payload.as_str()),
                    )),
                ],
            )
            .unwrap(),
        ],
        BatchMetadata::default(),
    )
    .unwrap()
}

struct SendingCollector {
    sender: EdgeSender,
    first_rss: u64,
    chunks: usize,
    max_bytes: usize,
}
#[async_trait]
impl StreamCollector for SendingCollector {
    async fn emit(&mut self, _port: &str, batch: Batch) -> calc_flow::Result<()> {
        if self.chunks == 0 {
            self.first_rss = rss();
        }
        self.chunks += 1;
        self.max_bytes = self.max_bytes.max(batch.estimated_bytes()?);
        self.sender.send(StreamMessage::data(batch)).await
    }
}

fn check_batch(data: &Batch, mut position: usize, fan: usize, keys: usize, width: usize) -> usize {
    for record in data.table_payload().unwrap().batches() {
        let int = |column: usize| {
            record
                .column(column)
                .as_any()
                .downcast_ref::<Int64Array>()
                .unwrap()
        };
        let text = |column: usize| {
            record
                .column(column)
                .as_any()
                .downcast_ref::<StringArray>()
                .unwrap()
        };
        for i in 0..record.num_rows() {
            let right_id = position / fan;
            let left_id = right_id % keys + position % fan * keys;
            assert_eq!(int(0).value(i), row_id(right_id % keys));
            assert_eq!(int(4).value(i), row_id(right_id % keys));
            assert_eq!(int(2).value(i), row_id(left_id));
            assert_eq!(int(6).value(i), row_id(right_id));
            for column in [1, 5] {
                let times = record
                    .column(column)
                    .as_any()
                    .downcast_ref::<TimestampMicrosecondArray>()
                    .unwrap();
                assert_eq!(times.value(i), 100_000_000);
            }
            assert_eq!(text(3).value(i).as_bytes(), vec![b'L'; width]);
            assert_eq!(text(7).value(i).as_bytes(), vec![b'R'; width]);
            position += 1;
        }
    }
    position
}

fn prepare(
    rt: &tokio::runtime::Runtime,
    config: &Config,
) -> (StreamJoinOperator, StreamJobContext, Batch) {
    let Config {
        fan,
        keys,
        width,
        incoming,
        ..
    } = *config;
    let left = batch(keys * fan, keys, width, 'L');
    let right = batch(incoming, keys, width, 'R');
    let spec = StreamJoinSpec::inner(
        ["key"],
        ["key"],
        "ts",
        "ts",
        JoinTimeBounds::new(Duration::from_secs(300), Duration::from_secs(60)).unwrap(),
        JoinStateLimits::new(1_000_000, 1 << 30, 10_000_000).unwrap(),
    )
    .unwrap();
    let mut operator = StreamJoinOperator::new("join", schema(), schema(), spec).unwrap();
    let job = StreamJobContext::new(
        1,
        "join-materialization",
        JsonMap::new(),
        None,
        CancellationToken::new(),
    );
    let context = StreamOperatorContext::new(&job, "join", None);
    let mut preload = EdgeCollector::new(operator.output_ports().to_vec());
    rt.block_on(operator.process_data("left", left, &context, &mut preload))
        .unwrap();
    assert!(preload.drain("output").is_empty());
    (operator, job, right)
}

fn sample(rt: &tokio::runtime::Runtime, config: &Config, validate: bool) -> Value {
    let Config {
        fan,
        keys,
        width,
        incoming,
        delay_ms,
    } = *config;
    let expected = incoming * fan;
    let (mut operator, job, right) = prepare(rt, config);
    let context = StreamOperatorContext::new(&job, "join", None);
    let budget = EdgeBudget::default();
    let (sender, mut receiver) = edge_channel("join-materialization", budget).unwrap();
    let mut collector = SendingCollector {
        sender,
        first_rss: 0,
        chunks: 0,
        max_bytes: 0,
    };
    let before = rss();
    let peak = Arc::new(AtomicU64::new(before));
    let stopped = Arc::new(AtomicBool::new(false));
    let sampler = {
        let peak = peak.clone();
        let stopped = stopped.clone();
        std::thread::spawn(move || {
            while !stopped.load(Ordering::Relaxed) {
                peak.fetch_max(rss(), Ordering::Relaxed);
                std::thread::sleep(Duration::from_millis(1));
            }
        })
    };
    let mut output_rows = 0;
    let mut verified = 0;
    let mut elapsed = 0.0;
    let allocation = allocation_counter::measure(|| {
        let started = Instant::now();
        rt.block_on(async {
            let producer = operator.process_data("right", right, &context, &mut collector);
            let consumer = async {
                while output_rows < expected {
                    let message = receiver.recv().await.unwrap().unwrap();
                    let data = message.as_data().unwrap();
                    output_rows += data.num_rows();
                    if validate {
                        verified = check_batch(data, verified, fan, keys, width);
                    }
                    if delay_ms != 0 {
                        tokio::time::sleep(Duration::from_millis(delay_ms)).await;
                    }
                }
                Ok::<(), calc_flow::CalcFlowError>(())
            };
            tokio::try_join!(producer, consumer).unwrap();
        });
        elapsed = started.elapsed().as_secs_f64();
    });
    peak.fetch_max(collector.first_rss, Ordering::Relaxed);
    stopped.store(true, Ordering::Relaxed);
    sampler.join().unwrap();
    assert_eq!(output_rows, expected);
    assert_eq!(
        operator.status().emitted_match_rows,
        u64::try_from(expected).unwrap()
    );
    assert!(collector.max_bytes <= budget.max_bytes);
    if validate {
        assert_eq!(verified, expected);
    }
    let metrics = collector.sender.metrics();
    json!({
        "config":config,
        "seconds":elapsed,
        "output_rows":output_rows,
        "input_rows":incoming,
        "rss_available":before > 0,
        "rss_before_bytes":before,
        "rss_peak_bytes":peak.load(Ordering::Relaxed),
        "rss_first_emit_bytes":collector.first_rss,
        "allocation_peak_bytes":allocation.bytes_max,
        "allocation_total_bytes":allocation.bytes_total,
        "allocation_count":allocation.count_total,
        "chunks":collector.chunks,
        "max_chunk_bytes":collector.max_bytes,
        "queue_high_water_bytes":metrics.high_water_bytes,
        "blocked_sends":metrics.blocked_sends,
        "blocked_seconds":metrics.blocked_duration.as_secs_f64(),
        "validated_all_rows":validate})
}

fn cancelled_input_check() {
    let rt = tokio::runtime::Builder::new_current_thread()
        .enable_all()
        .build()
        .unwrap();
    let cancel = CancellationToken::new();
    let job = StreamJobContext::new(
        1,
        "join-materialization",
        JsonMap::new(),
        None,
        cancel.clone(),
    );
    let context = StreamOperatorContext::new(&job, "join", None);
    let spec = StreamJoinSpec::inner(
        ["key"],
        ["key"],
        "ts",
        "ts",
        JoinTimeBounds::new(Duration::from_secs(300), Duration::from_secs(60)).unwrap(),
        JoinStateLimits::new(1_000_000, 1 << 30, 10_000_000).unwrap(),
    )
    .unwrap();
    let mut operator = StreamJoinOperator::new("join", schema(), schema(), spec).unwrap();
    let mut collector = EdgeCollector::new(operator.output_ports().to_vec());
    cancel.cancel();
    let result = rt.block_on(operator.process_data(
        "right",
        batch(1000, 10, 1024, 'R'),
        &context,
        &mut collector,
    ));
    assert!(matches!(
        result,
        Err(calc_flow::CalcFlowError::Cancelled { .. })
    ));
    assert!(collector.drain("output").is_empty());
    assert_eq!(operator.status().right.retained_rows, 0);
}

fn worker() {
    let config: Config =
        serde_json::from_str(&std::env::var("CALC_FLOW_JOIN_CONFIG").unwrap()).unwrap();
    let rt = tokio::runtime::Builder::new_current_thread()
        .enable_all()
        .build()
        .unwrap();
    let validation = sample(&rt, &config, true);
    println!("JOIN_READY {validation}");
    io::stdout().flush().unwrap();
    for line in io::stdin().lock().lines() {
        let line = line.unwrap();
        if line == "quit" {
            break;
        }
        assert_eq!(line, "sample");
        println!("JOIN_SAMPLE {}", sample(&rt, &config, false));
        io::stdout().flush().unwrap();
    }
}

fn main() {
    if std::env::var_os("CALC_FLOW_JOIN_CONFIG").is_some() {
        worker();
        return;
    }
    let arguments = std::env::args().collect::<Vec<_>>();
    let check_only = arguments
        .iter()
        .any(|arg| arg == "--check" || arg == "--test");
    cancelled_input_check();
    let rt = tokio::runtime::Builder::new_current_thread()
        .enable_all()
        .build()
        .unwrap();
    let workloads = [
        ("narrow_f100_fast", 128, 100, 0),
        ("wide_f10_fast", 1024, 10, 0),
        ("wide_f100_fast", 1024, 100, 0),
        ("wide_f100_slow", 1024, 100, 10),
    ];
    let cases = workloads
        .into_iter()
        .map(|(name, width, fan, delay_ms)| {
            let config = Config {
                keys: 10,
                incoming: 1000,
                width,
                fan,
                delay_ms,
            };
            let oracle = sample(&rt, &config, true);
            let samples = if check_only {
                Vec::new()
            } else {
                (0..20)
                    .map(|_| sample(&rt, &config, false))
                    .collect::<Vec<_>>()
            };
            json!({"name": name, "config": config, "oracle": oracle, "samples": samples})
        })
        .collect::<Vec<_>>();
    let report = json!({"schema": "calc-flow.join-materialization.v1", "scope": "operator-bounded-edge", "cases": cases});
    let bytes = serde_json::to_vec_pretty(&report).unwrap();
    if let Some(index) = arguments.iter().position(|arg| arg == "--output") {
        let path = std::path::Path::new(arguments.get(index + 1).expect("--output needs a path"));
        if let Some(parent) = path.parent() {
            std::fs::create_dir_all(parent).unwrap();
        }
        std::fs::write(path, bytes).unwrap();
    } else {
        println!("{}", String::from_utf8(bytes).unwrap());
    }
}
