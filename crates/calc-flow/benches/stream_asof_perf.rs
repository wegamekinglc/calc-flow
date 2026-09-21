//! ASOF pending/retained scale, skew and restore settlement measurements.
use async_trait::async_trait;
use calc_flow::{
    AsofJoinSide, AsofStateLimits, Batch, BatchMetadata, CalcFlowError, CancellationToken,
    EdgeCollector, Epoch, EventTime, IngressProgress, IngressProgressSnapshot, IngressState,
    JsonMap, OperatorMetadata, StreamAsofJoinOperator, StreamAsofJoinSpec, StreamCollector,
    StreamJobContext, StreamOperator, StreamOperatorContext,
};
use datafusion::arrow::{
    array::{Array, Int64Array, TimestampMicrosecondArray},
    datatypes::{DataType, Field, Schema, TimeUnit},
    record_batch::RecordBatch,
};
use serde::{Deserialize, Serialize};
use serde_json::{Value, json};
use std::{
    collections::BTreeMap,
    io::{self, BufRead, Write},
    sync::{
        Arc,
        atomic::{AtomicBool, AtomicU64, Ordering},
    },
    time::{Duration, Instant},
};

const LEFT_TIME: i64 = 1_000_000;
const RIGHT_TIME: i64 = LEFT_TIME - 1;
const FRONTIER: i64 = 2_000_000;

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

fn schema() -> Arc<Schema> {
    Arc::new(Schema::new(vec![
        Field::new("key", DataType::Int64, false),
        Field::new(
            "time",
            DataType::Timestamp(TimeUnit::Microsecond, Some("UTC".into())),
            false,
        ),
        Field::new("sequence", DataType::Int64, false),
        Field::new("value", DataType::Int64, false),
    ]))
}

fn row_i64(index: usize) -> i64 {
    i64::try_from(index).expect("bounded ASOF workload index fits i64")
}

fn key(index: usize, skew: bool) -> i64 {
    if !skew {
        row_i64(index % 32)
    } else if index % 10 < 9 {
        0
    } else {
        1 + row_i64(index / 10 % 31)
    }
}

fn batch(rows: usize, skew: bool, left: bool) -> Batch {
    Batch::table(
        vec![
            RecordBatch::try_new(
                schema(),
                vec![
                    Arc::new(Int64Array::from_iter_values(
                        (0..rows).map(|i| key(i, skew)),
                    )),
                    Arc::new(
                        TimestampMicrosecondArray::from_iter_values((0..rows).map(|i| {
                            if left {
                                LEFT_TIME + row_i64(i)
                            } else {
                                RIGHT_TIME
                            }
                        }))
                        .with_timezone("UTC"),
                    ),
                    Arc::new(Int64Array::from_iter_values((0..rows).map(row_i64))),
                    Arc::new(Int64Array::from_iter_values(
                        (0..rows).map(|i| row_i64(i * if left { 7 } else { 3 })),
                    )),
                ],
            )
            .unwrap(),
        ],
        BatchMetadata::default(),
    )
    .unwrap()
}

fn operator() -> StreamAsofJoinOperator {
    let side = |name: &str| {
        AsofJoinSide::new(
            vec!["key".into()],
            "time".into(),
            vec!["sequence".into()],
            name.into(),
        )
        .unwrap()
    };
    let spec = StreamAsofJoinSpec::new(
        side("left"),
        side("right"),
        Duration::from_micros(10_000_000),
        AsofStateLimits::new(100_000, 512 << 20).unwrap(),
    )
    .unwrap();
    StreamAsofJoinOperator::new("asof", schema(), schema(), spec).unwrap()
}

fn job(cancel: CancellationToken) -> StreamJobContext {
    StreamJobContext::new(1, "dal301", JsonMap::new(), None, cancel)
}

fn progress(job: &StreamJobContext, left: i64, right: i64) -> StreamOperatorContext<'_> {
    let snapshot = IngressProgressSnapshot::new(BTreeMap::from([
        (
            "left".into(),
            IngressProgress::new(IngressState::Active, Some(EventTime::from_micros(left))),
        ),
        (
            "right".into(),
            IngressProgress::new(IngressState::Active, Some(EventTime::from_micros(right))),
        ),
    ]));
    StreamOperatorContext::with_ingress_progress(
        job,
        "asof",
        Some(EventTime::from_micros(left.min(right))),
        snapshot,
    )
}

async fn seed(
    op: &mut StreamAsofJoinOperator,
    job: &StreamJobContext,
    pending: usize,
    retained: usize,
    skew: bool,
) {
    let cx = StreamOperatorContext::new(job, "asof", None);
    let mut collector = EdgeCollector::new(op.output_ports().to_vec());
    op.process_data("right", batch(retained, skew, false), &cx, &mut collector)
        .await
        .unwrap();
    op.process_data("left", batch(pending, skew, true), &cx, &mut collector)
        .await
        .unwrap();
    assert!(collector.drain("output").is_empty());
}

fn validate(data: &Batch, start: usize, rights: &BTreeMap<i64, i64>, skew: bool) -> usize {
    let mut position = start;
    assert_eq!(data.metadata().sequence(), start as u64);
    for record in data.table_payload().unwrap().batches() {
        let ints = |c: usize| {
            record
                .column(c)
                .as_any()
                .downcast_ref::<Int64Array>()
                .unwrap()
        };
        let times = |c: usize| {
            record
                .column(c)
                .as_any()
                .downcast_ref::<TimestampMicrosecondArray>()
                .unwrap()
        };
        for row in 0..record.num_rows() {
            let k = key(position, skew);
            let candidate = rights[&k];
            assert_eq!(ints(0).value(row), k);
            assert_eq!(times(1).value(row), LEFT_TIME + row_i64(position));
            assert_eq!(ints(2).value(row), row_i64(position));
            assert_eq!(ints(3).value(row), row_i64(position * 7));
            assert!(!ints(4).is_null(row));
            assert_eq!(ints(4).value(row), k);
            assert_eq!(times(5).value(row), RIGHT_TIME);
            assert_eq!(ints(6).value(row), candidate);
            assert_eq!(ints(7).value(row), candidate * 3);
            position += 1;
        }
    }
    position
}

struct Collector {
    started: Instant,
    rows: usize,
    chunks: Vec<(usize, f64)>,
    max_bytes: usize,
    check: bool,
    skew: bool,
    rights: BTreeMap<i64, i64>,
}
impl Collector {
    fn new(pending: usize, retained: usize, skew: bool, check: bool) -> Self {
        let mut rights = BTreeMap::new();
        for i in 0..retained {
            rights.insert(key(i, skew), row_i64(i));
        }
        Self {
            started: Instant::now(),
            rows: 0,
            chunks: Vec::with_capacity(pending.div_ceil(128)),
            max_bytes: 0,
            check,
            skew,
            rights,
        }
    }
}
#[async_trait]
impl StreamCollector for Collector {
    async fn emit(&mut self, _port: &str, batch: Batch) -> calc_flow::Result<()> {
        self.chunks
            .push((batch.num_rows(), self.started.elapsed().as_secs_f64()));
        self.max_bytes = self.max_bytes.max(batch.estimated_bytes()?);
        if self.check {
            let next = validate(&batch, self.rows, &self.rights, self.skew);
            assert_eq!(next, self.rows + batch.num_rows());
        }
        self.rows += batch.num_rows();
        Ok(())
    }
}

fn sample(rt: &tokio::runtime::Runtime, config: &Config, check: bool) -> Value {
    let Config {
        pending,
        retained,
        skew,
        restored,
    } = *config;
    let job = job(CancellationToken::new());
    let mut op = operator();
    let setup_start = Instant::now();
    rt.block_on(seed(&mut op, &job, pending, retained, skew));
    let admission_seconds = setup_start.elapsed().as_secs_f64();
    let snapshot = op.checkpoint(Epoch::INITIAL).unwrap();
    let checkpoint_before = snapshot
        .segments
        .values()
        .map(|s| s.bytes().len())
        .sum::<usize>();
    let restore_start = Instant::now();
    if restored {
        drop(op);
        op = operator();
        op.restore(&snapshot).unwrap();
    }
    let restore_seconds = if restored {
        Some(restore_start.elapsed().as_secs_f64())
    } else {
        None
    };
    drop(snapshot);
    let before_status = op.status();
    assert_eq!(before_status.pending_left_rows, pending as u64);
    assert_eq!(before_status.retained_right_rows, retained as u64);
    let cx = progress(&job, FRONTIER, FRONTIER);
    let mut collector = Collector::new(pending, retained, skew, check);
    let before = rss();
    let peak = Arc::new(AtomicU64::new(before));
    let stop = Arc::new(AtomicBool::new(false));
    let sampler = {
        let peak = peak.clone();
        let stop = stop.clone();
        std::thread::spawn(move || {
            while !stop.load(Ordering::Relaxed) {
                peak.fetch_max(rss(), Ordering::Relaxed);
                std::thread::sleep(Duration::from_millis(1));
            }
        })
    };
    let mut seconds = 0.0;
    let allocation = allocation_counter::measure(|| {
        collector.started = Instant::now();
        rt.block_on(op.on_watermark(EventTime::from_micros(FRONTIER), &cx, &mut collector))
            .unwrap();
        seconds = collector.started.elapsed().as_secs_f64();
    });
    stop.store(true, Ordering::Relaxed);
    sampler.join().unwrap();
    let after_status = op.status();
    assert_eq!(collector.rows, pending);
    assert_eq!(collector.chunks.len(), pending.div_ceil(128));
    assert!(collector.chunks.iter().all(|(rows, _)| *rows <= 128));
    assert_eq!(after_status.pending_left_rows, 0);
    assert_eq!(after_status.matched_rows, pending as u64);
    assert_eq!(after_status.retained_right_rows, retained as u64);
    assert_eq!(
        after_status.output_watermark_micros,
        Some(EventTime::from_micros(FRONTIER - 1))
    );
    let capture_start = Instant::now();
    let after = op.checkpoint(Epoch::INITIAL).unwrap();
    let capture_seconds = capture_start.elapsed().as_secs_f64();
    let checkpoint_after = after
        .segments
        .values()
        .map(|s| s.bytes().len())
        .sum::<usize>();
    json!({"config":config,"seconds":seconds,"output_rows":collector.rows,"chunks":collector.chunks,"max_chunk_bytes":collector.max_bytes,"rss_available":before > 0,"rss_before_bytes":before,"rss_peak_bytes":peak.load(Ordering::Relaxed),"allocation_peak_bytes":allocation.bytes_max,"allocation_total_bytes":allocation.bytes_total,"allocation_count":allocation.count_total,"before_status":before_status,"after_status":after_status,"checkpoint_before_bytes":checkpoint_before,"checkpoint_after_bytes":checkpoint_after,"admission_seconds_untimed":admission_seconds,"restore_seconds_untimed":restore_seconds,"capture_seconds_untimed":capture_seconds,"validated_all_rows":check})
}

struct CancelCollector {
    cancel: CancellationToken,
    accepted: Vec<Batch>,
}
#[async_trait]
impl StreamCollector for CancelCollector {
    async fn emit(&mut self, _port: &str, batch: Batch) -> calc_flow::Result<()> {
        if self.accepted.is_empty() {
            self.accepted.push(batch);
            Ok(())
        } else {
            self.cancel.cancel();
            std::future::pending().await
        }
    }
}

fn strict_frontiers_cancel_restore_check() {
    let rt = tokio::runtime::Builder::new_current_thread()
        .enable_all()
        .build()
        .unwrap();
    rt.block_on(async {
        let cancel = CancellationToken::new();
        let job = job(cancel.clone());
        let mut op = operator();
        seed(&mut op, &job, 300, 512, true).await;
        let mut zero = EdgeCollector::new(op.output_ports().to_vec());
        for (l, r) in [(LEFT_TIME, LEFT_TIME), (FRONTIER, LEFT_TIME)] {
            let cx = progress(&job, l, r);
            op.on_watermark(EventTime::from_micros(l.min(r)), &cx, &mut zero)
                .await
                .unwrap();
            assert!(zero.drain("output").is_empty());
            assert_eq!(op.status().pending_left_rows, 300);
        }
        let cx = progress(&job, FRONTIER, FRONTIER);
        let mut out = CancelCollector {
            cancel,
            accepted: Vec::new(),
        };
        let error = op
            .on_watermark(EventTime::from_micros(FRONTIER), &cx, &mut out)
            .await
            .unwrap_err();
        assert!(matches!(error, CalcFlowError::Cancelled { .. }));
        assert_eq!(op.status().emitted_left_rows, 128);
        assert_eq!(op.status().pending_left_rows, 172);
        let snapshot = op.checkpoint(Epoch::INITIAL).unwrap();
        let mut restored = operator();
        restored.restore(&snapshot).unwrap();
        let resumed_job = self::job(CancellationToken::new());
        let resumed_cx = progress(&resumed_job, FRONTIER, FRONTIER);
        let mut resumed = EdgeCollector::new(restored.output_ports().to_vec());
        restored
            .on_watermark(EventTime::from_micros(FRONTIER), &resumed_cx, &mut resumed)
            .await
            .unwrap();
        let rights = (0..512).map(|i| (key(i, true), row_i64(i))).collect();
        let mut pos = validate(&out.accepted[0], 0, &rights, true);
        for m in resumed.drain("output") {
            pos = validate(m.as_data().unwrap(), pos, &rights, true);
        }
        assert_eq!(pos, 300);
        assert_eq!(restored.status().emitted_left_rows, 300);
    });
}

fn worker() {
    let config: Config =
        serde_json::from_str(&std::env::var("CALC_FLOW_ASOF_CONFIG").unwrap()).unwrap();
    let rt = tokio::runtime::Builder::new_current_thread()
        .enable_all()
        .build()
        .unwrap();
    println!("ASOF_READY {}", sample(&rt, &config, true));
    io::stdout().flush().unwrap();
    for line in io::stdin().lock().lines() {
        let line = line.unwrap();
        if line == "quit" {
            break;
        }
        assert_eq!(line, "sample");
        println!("ASOF_SAMPLE {}", sample(&rt, &config, false));
        io::stdout().flush().unwrap();
    }
}

#[derive(Clone, Copy, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct Config {
    pending: usize,
    retained: usize,
    skew: bool,
    restored: bool,
}

fn workloads() -> [(&'static str, Config); 8] {
    [
        ("balanced_512", 512, 512, false, false),
        ("balanced_2048", 2048, 2048, false, false),
        ("balanced_8192", 8192, 8192, false, false),
        ("fixed128_right512", 128, 512, false, false),
        ("fixed128_right2048", 128, 2048, false, false),
        ("fixed128_right8192", 128, 8192, false, false),
        ("skew_8192", 8192, 8192, true, false),
        ("restored_skew_8192", 8192, 8192, true, true),
    ]
    .map(|(name, pending, retained, skew, restored)| {
        (
            name,
            Config {
                pending,
                retained,
                skew,
                restored,
            },
        )
    })
}

fn main() {
    if std::env::var_os("CALC_FLOW_ASOF_CONFIG").is_some() {
        worker();
        return;
    }
    let args = std::env::args().collect::<Vec<_>>();
    let check = args.iter().any(|arg| arg == "--check" || arg == "--test");
    strict_frontiers_cancel_restore_check();
    let rt = tokio::runtime::Builder::new_current_thread()
        .enable_all()
        .build()
        .unwrap();
    let cases = workloads()
        .into_iter()
        .map(|(name, config)| {
            let oracle = sample(&rt, &config, true);
            let samples = if check {
                Vec::new()
            } else {
                (0..20)
                    .map(|_| sample(&rt, &config, false))
                    .collect::<Vec<_>>()
            };
            json!({"name": name, "config": config, "oracle": oracle, "samples": samples})
        })
        .collect::<Vec<_>>();
    let report = json!({"schema": "calc-flow.asof-finalization.v1", "scope": "operator-watermark-settlement", "cases": cases});
    let bytes = serde_json::to_vec_pretty(&report).unwrap();
    if let Some(index) = args.iter().position(|arg| arg == "--output") {
        let path = std::path::Path::new(args.get(index + 1).expect("--output needs a path"));
        if let Some(parent) = path.parent() {
            std::fs::create_dir_all(parent).unwrap();
        }
        std::fs::write(path, bytes).unwrap();
    } else {
        println!("{}", String::from_utf8(bytes).unwrap());
    }
}
