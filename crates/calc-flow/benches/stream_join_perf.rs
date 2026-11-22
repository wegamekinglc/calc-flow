//! Resident controlled baseline for the bounded event-time stream join.
//!
//! The four 10k-row handlers and the first capture of 20k dirty rows retain
//! the workload and timing boundaries introduced by DAL-130. The historical
//! base-plus-delta capture, compaction and restore series depended on the
//! legacy segmented writer. Native layout component measurements have their own workload IDs.
//!
//! The original AC19 handler floors on WSL2 i9-13900HX at `main@f74f32a`
//! (workload SHA-256 `555546a7…`) were 1.12M rows/s for no match, 491k for
//! one-to-one, 75.0k for fanout10 and 135k for eviction. Each floor was
//! 80% of the platform-specific frozen 95% lower bound.

use std::collections::BTreeMap;
use std::sync::Arc;
use std::time::{Duration, Instant};

use calc_flow::{
    Batch, BatchMetadata, CancellationToken, EdgeCollector, Epoch, EventTime, IngressProgress,
    IngressProgressSnapshot, IngressState, JoinStateLimits, JoinTimeBounds, JsonMap,
    OperatorMetadata, StreamJobContext, StreamJoinOperator, StreamJoinSpec, StreamOperator,
    StreamOperatorContext,
};
use criterion::{Criterion, criterion_group};
use datafusion::arrow::array::{Int64Array, StringArray, TimestampMicrosecondArray, UInt32Array};
use datafusion::arrow::compute::take;
use datafusion::arrow::datatypes::{DataType, Field, Schema, TimeUnit};
use datafusion::arrow::record_batch::RecordBatch;
use sha2::{Digest, Sha256};

#[path = "support/native_join_components.rs"]
#[cfg(feature = "benchmark-internals")]
mod native_join_components;

const SECOND: i64 = 1_000_000;
const BASE_TS: i64 = 100 * SECOND;
const ROWS: usize = 10_000;
const FAN_KEYS: usize = 1_000;
const ROW_LIMIT: u64 = 4_000_000;
const BYTE_LIMIT: u64 = 4 * 1_024 * 1_024 * 1_024;
const MATCH_LIMIT: u64 = 100_000_000;
const BEFORE: Duration = Duration::from_secs(300);
const AFTER: Duration = Duration::from_secs(60);

fn schema() -> Arc<Schema> {
    Arc::new(Schema::new(vec![
        Field::new("account_id", DataType::Utf8, false),
        Field::new(
            "ts",
            DataType::Timestamp(TimeUnit::Microsecond, None),
            false,
        ),
        Field::new("amount", DataType::Int64, false),
    ]))
}

fn make_batch(keys: &[String], ts: i64) -> Batch {
    valued_batch(keys, ts, 7)
}

fn valued_batch(keys: &[String], ts: i64, amount: i64) -> Batch {
    let batch = RecordBatch::try_new(
        schema(),
        vec![
            Arc::new(StringArray::from(keys.to_vec())),
            Arc::new(TimestampMicrosecondArray::from(vec![ts; keys.len()])),
            Arc::new(Int64Array::from(vec![amount; keys.len()])),
        ],
    )
    .unwrap();
    Batch::table(vec![batch], BatchMetadata::default()).unwrap()
}

/// `count` unique keys sharing the given prefix.
fn unique_keys(prefix: &str, count: usize) -> Vec<String> {
    (0..count).map(|i| format!("{prefix}{i:07}")).collect()
}

/// `count` keys cycling over `distinct` values (fan-out = count/distinct).
fn fan_keys(distinct: usize, count: usize) -> Vec<String> {
    (0..count)
        .map(|i| format!("K{:07}", i % distinct))
        .collect()
}

fn spec() -> StreamJoinSpec {
    StreamJoinSpec::inner(
        ["account_id"],
        ["account_id"],
        "ts",
        "ts",
        JoinTimeBounds::new(BEFORE, AFTER).unwrap(),
        JoinStateLimits::new(ROW_LIMIT, BYTE_LIMIT, MATCH_LIMIT).unwrap(),
    )
    .unwrap()
}

fn new_operator() -> StreamJoinOperator {
    StreamJoinOperator::new("match", schema(), schema(), spec()).unwrap()
}

fn job() -> StreamJobContext {
    StreamJobContext::new(
        1,
        "fingerprint",
        JsonMap::new(),
        None,
        CancellationToken::new(),
    )
}

fn plain_ctx(job: &StreamJobContext) -> StreamOperatorContext<'_> {
    StreamOperatorContext::new(job, "match", None)
}

fn watermark_ctx(job: &StreamJobContext, micros: i64) -> StreamOperatorContext<'_> {
    let progress = IngressProgressSnapshot::new(BTreeMap::from([(
        "right".into(),
        IngressProgress::new(IngressState::Active, Some(EventTime::from_micros(micros))),
    )]));
    StreamOperatorContext::with_ingress_progress(job, "match", None, progress)
}

/// Feeds `batch` to `ingress` without timing (bench setup helper).
async fn feed(
    operator: &mut StreamJoinOperator,
    job: &StreamJobContext,
    collector: &mut EdgeCollector,
    ingress: &str,
    batch: &Batch,
) {
    let result = operator
        .process_data(ingress, batch.clone(), &plain_ctx(job), collector)
        .await;
    if let Err(error) = &result {
        eprintln!(
            "JOIN_PERF_FEED_FAIL ingress={ingress} rows={} error={error:?}",
            batch.num_rows()
        );
    }
    result.unwrap();
}

fn read_rss() -> (u64, u64) {
    let status = std::fs::read_to_string("/proc/self/status").unwrap_or_default();
    let mut rss = 0;
    let mut hwm = 0;
    for line in status.lines() {
        if let Some(rest) = line.strip_prefix("VmRSS:") {
            rss = rest
                .split_whitespace()
                .next()
                .unwrap_or("0")
                .parse()
                .unwrap_or(0);
        } else if let Some(rest) = line.strip_prefix("VmHWM:") {
            hwm = rest
                .split_whitespace()
                .next()
                .unwrap_or("0")
                .parse()
                .unwrap_or(0);
        }
    }
    (rss, hwm)
}

fn provenance() -> serde_json::Value {
    let tree = std::process::Command::new("git")
        .args(["rev-parse", "HEAD"])
        .output()
        .map(|o| String::from_utf8_lossy(&o.stdout).trim().to_string())
        .unwrap_or_default();
    let rustc = std::process::Command::new("rustc")
        .arg("-V")
        .output()
        .map(|o| String::from_utf8_lossy(&o.stdout).trim().to_string())
        .unwrap_or_default();
    let cpu = std::fs::read_to_string("/proc/cpuinfo")
        .ok()
        .and_then(|info| {
            info.lines()
                .find(|l| l.starts_with("model name"))
                .map(|l| l.split(':').nth(1).unwrap_or("").trim().to_string())
        })
        .unwrap_or_default();
    let load = std::fs::read_to_string("/proc/loadavg")
        .map(|s| s.trim().to_string())
        .unwrap_or_default();
    // Workload hash: digest of every canonical batch's keys and fixed cells.
    let mut digest = Sha256::new();
    for keys in [
        unique_keys("L", ROWS),
        unique_keys("R", ROWS),
        unique_keys("K", ROWS),
        fan_keys(FAN_KEYS, ROWS),
    ] {
        digest.update(format!("{}|{BASE_TS}|7", keys.join(",")).as_bytes());
    }
    serde_json::json!({
        "commit": tree,
        "workload_sha256": hex::encode(digest.finalize()),
        "rustc": rustc,
        "target_triple": std::env::consts::ARCH.to_string() + "-" + std::env::consts::OS,
        "cpu_model": cpu,
        "background_load": load,
        "rows_per_input_batch": ROWS,
        "bounds": {"before_secs": BEFORE.as_secs(), "after_secs": AFTER.as_secs()},
        "limits": {"rows": ROW_LIMIT, "bytes": BYTE_LIMIT, "matches": MATCH_LIMIT},
    })
}

/// Pre-built canonical workload batches shared by the scenarios.
struct ScenarioBatches {
    left_unique: Arc<Batch>,
    left_disjoint: Arc<Batch>,
    right_disjoint: Arc<Batch>,
    left_one_to_one: Arc<Batch>,
    right_one_to_one: Arc<Batch>,
    left_fan: Arc<Batch>,
    right_fan: Arc<Batch>,
}

impl ScenarioBatches {
    fn new() -> Self {
        let keys_one_to_one = Arc::new(unique_keys("K", ROWS));
        Self {
            left_unique: Arc::new(make_batch(&unique_keys("L", ROWS), BASE_TS)),
            // Truly disjoint key sets: zero equality rows, the steady-state
            // no-match shape (the zero-row probe is well-defined since PR
            // #189).
            left_disjoint: Arc::new(make_batch(&unique_keys("L", ROWS), BASE_TS)),
            right_disjoint: Arc::new(make_batch(&unique_keys("R", ROWS), BASE_TS)),
            left_one_to_one: Arc::new(make_batch(&keys_one_to_one, BASE_TS)),
            right_one_to_one: Arc::new(make_batch(&keys_one_to_one, BASE_TS)),
            left_fan: Arc::new(make_batch(&fan_keys(FAN_KEYS, ROWS), BASE_TS)),
            right_fan: Arc::new(make_batch(&fan_keys(FAN_KEYS, ROWS), BASE_TS)),
        }
    }
}

fn baseline(c: &mut Criterion) {
    assert!(
        std::env::var_os("CALC_FLOW_JOIN_PERF_FOCUSED").is_none(),
        "the focused base-plus-delta checkpoint workloads have been retired"
    );
    println!("JOIN_PERF_PROVENANCE {}", provenance());
    let runtime = tokio::runtime::Runtime::new().unwrap();
    let batches = ScenarioBatches::new();
    run_probe(&runtime, &batches);

    let mut group = c.benchmark_group("join");
    group.sample_size(30);
    group.warm_up_time(Duration::from_secs(1));
    group.measurement_time(Duration::from_secs(2));
    handler_scenarios(&mut group, &runtime, &batches);
    checkpoint_scenarios(&mut group, &runtime, &batches);
    let (rss, hwm) = read_rss();
    println!("JOIN_PERF_RSS rss_kib={rss} hwm_kib={hwm}");
    group.finish();
}

/// Compare all six output columns in ingress order, across batch boundaries.
fn check_pairs(collector: &mut EdgeCollector, left: &Batch, right: &Batch, per_input: usize) {
    assert_eq!(left.num_rows(), right.num_rows());
    let indices = UInt32Array::from_iter_values(
        (0..right.num_rows())
            .flat_map(|row| std::iter::repeat_n(u32::try_from(row).unwrap(), per_input)),
    );
    let expected = [left, right]
        .into_iter()
        .flat_map(|batch| batch.table_payload().unwrap().batches()[0].columns())
        .map(|column| take(column.as_ref(), &indices, None).unwrap())
        .collect::<Vec<_>>();
    let mut offset = 0;
    for message in collector.drain("output") {
        let table = message.as_data().unwrap().table_payload().unwrap();
        for record in table.batches() {
            assert_eq!(record.num_columns(), expected.len());
            for (actual, wanted) in record.columns().iter().zip(&expected) {
                assert_eq!(
                    actual.as_ref(),
                    wanted.slice(offset, record.num_rows()).as_ref()
                );
            }
            offset += record.num_rows();
        }
    }
    assert_eq!(offset, indices.values().len());
}

async fn check_recovery(probe: &mut StreamJoinOperator, job: &StreamJobContext) {
    let left_key = ["left-only".to_owned()];
    let right_key = ["right-only".to_owned()];
    let left = valued_batch(&left_key, BASE_TS + SECOND, 11);
    let right = valued_batch(&right_key, BASE_TS + 2 * SECOND, 23);
    let mut collector = EdgeCollector::new(probe.output_ports().to_vec());
    feed(probe, job, &mut collector, "left", &left).await;
    feed(probe, job, &mut collector, "right", &right).await;
    check_pairs(&mut collector, &left, &right, 0);

    let snapshot = probe.checkpoint(Epoch::INITIAL).unwrap();
    let mut restored = new_operator();
    restored.restore(&snapshot).unwrap();
    assert_eq!(restored.status(), probe.status());
    let next_right = valued_batch(&left_key, BASE_TS + 3 * SECOND, 31);
    feed(&mut restored, job, &mut collector, "right", &next_right).await;
    check_pairs(&mut collector, &left, &next_right, 1);
    let next_left = valued_batch(&right_key, BASE_TS + 4 * SECOND, 43);
    feed(&mut restored, job, &mut collector, "left", &next_left).await;
    check_pairs(&mut collector, &next_left, &right, 1);
}

/// Warm-up probe: prints the per-scenario state/match cardinality and RSS,
/// then runs the fail-closed harness self-checks.
fn run_probe(runtime: &tokio::runtime::Runtime, batches: &ScenarioBatches) {
    runtime.block_on(async {
        let job = job();
        let mut probe = new_operator();
        let mut collector = EdgeCollector::new(probe.output_ports().to_vec());
        feed(&mut probe, &job, &mut collector, "left", &batches.left_one_to_one).await;
        feed(&mut probe, &job, &mut collector, "right", &batches.right_one_to_one).await;
        let status = probe.status();
        let (rss, hwm) = read_rss();
        println!(
            "JOIN_PERF_PROBE one_to_one matches={} left_retained={} left_bytes={} right_retained={} rss_kib={rss} hwm_kib={hwm}",
            status.emitted_match_rows,
            status.left.retained_rows,
            status.left.retained_bytes,
            status.right.retained_rows,
        );
        check_pairs(&mut collector, &batches.left_one_to_one, &batches.right_one_to_one, 1);
        check_recovery(&mut probe, &job).await;
        let mut probe = new_operator();
        let mut collector = EdgeCollector::new(probe.output_ports().to_vec());
        feed(&mut probe, &job, &mut collector, "left", &batches.left_fan).await;
        feed(&mut probe, &job, &mut collector, "right", &batches.right_fan).await;
        let status = probe.status();
        println!(
            "JOIN_PERF_PROBE fanout10 matches={} left_retained={}",
            status.emitted_match_rows, status.left.retained_rows,
        );

        check_pairs(&mut collector, &batches.left_fan, &batches.right_fan, ROWS / FAN_KEYS);
    });
}

fn handler_scenarios(
    group: &mut criterion::BenchmarkGroup<'_, criterion::measurement::WallTime>,
    runtime: &tokio::runtime::Runtime,
    batches: &ScenarioBatches,
) {
    // Scenario: no-match — right 10k probes 10k retained left rows, 0 pairs.
    group.bench_function("handler/right_10k_no_match", |b| {
        b.to_async(runtime).iter_custom(|iters| {
            let left = Arc::clone(&batches.left_disjoint);
            let right = Arc::clone(&batches.right_disjoint);
            async move {
                let job = job();
                let mut total = Duration::ZERO;
                for _ in 0..iters {
                    let mut operator = new_operator();
                    let mut collector = EdgeCollector::new(operator.output_ports().to_vec());
                    feed(&mut operator, &job, &mut collector, "left", &left).await;
                    let start = Instant::now();
                    operator
                        .process_data("right", (*right).clone(), &plain_ctx(&job), &mut collector)
                        .await
                        .unwrap();
                    total += start.elapsed();
                }
                total
            }
        });
    });

    // Scenario: 1:1 — 10k pairs emitted against 10k retained left rows.
    group.bench_function("handler/right_10k_one_to_one", |b| {
        b.to_async(runtime).iter_custom(|iters| {
            let left = Arc::clone(&batches.left_one_to_one);
            let right = Arc::clone(&batches.right_one_to_one);
            async move {
                let job = job();
                let mut total = Duration::ZERO;
                for _ in 0..iters {
                    let mut operator = new_operator();
                    let mut collector = EdgeCollector::new(operator.output_ports().to_vec());
                    feed(&mut operator, &job, &mut collector, "left", &left).await;
                    let start = Instant::now();
                    operator
                        .process_data("right", (*right).clone(), &plain_ctx(&job), &mut collector)
                        .await
                        .unwrap();
                    total += start.elapsed();
                    collector.drain("output");
                }
                total
            }
        });
    });

    // Scenario: high fan-out — 1,000 keys × 10 rows per side → 100k pairs.
    group.bench_function("handler/right_10k_fanout10", |b| {
        b.to_async(runtime).iter_custom(|iters| {
            let left = Arc::clone(&batches.left_fan);
            let right = Arc::clone(&batches.right_fan);
            async move {
                let job = job();
                let mut total = Duration::ZERO;
                for _ in 0..iters {
                    let mut operator = new_operator();
                    let mut collector = EdgeCollector::new(operator.output_ports().to_vec());
                    feed(&mut operator, &job, &mut collector, "left", &left).await;
                    let start = Instant::now();
                    operator
                        .process_data("right", (*right).clone(), &plain_ctx(&job), &mut collector)
                        .await
                        .unwrap();
                    total += start.elapsed();
                    collector.drain("output");
                }
                total
            }
        });
    });

    // Scenario: watermark eviction — one progress call evicts 10k left rows.
    group.bench_function("handler/watermark_evict_10k", |b| {
        b.to_async(runtime).iter_custom(|iters| {
            let left = Arc::clone(&batches.left_unique);
            async move {
                let job = job();
                let mut total = Duration::ZERO;
                for _ in 0..iters {
                    let mut operator = new_operator();
                    let mut collector = EdgeCollector::new(operator.output_ports().to_vec());
                    feed(&mut operator, &job, &mut collector, "left", &left).await;
                    let start = Instant::now();
                    operator
                        .on_ingress_progress("right", &watermark_ctx(&job, BASE_TS + 61 * SECOND))
                        .await
                        .unwrap();
                    total += start.elapsed();
                }
                total
            }
        });
    });
}

fn checkpoint_scenarios(
    group: &mut criterion::BenchmarkGroup<'_, criterion::measurement::WallTime>,
    runtime: &tokio::runtime::Runtime,
    batches: &ScenarioBatches,
) {
    // Scenario: dirty checkpoint capture — 20k dirty rows, no base carried.
    group.bench_function("checkpoint/capture_dirty_20k", |b| {
        b.to_async(runtime).iter_custom(|iters| {
            let left = Arc::clone(&batches.left_one_to_one);
            let right = Arc::clone(&batches.right_one_to_one);
            async move {
                let job = job();
                let mut total = Duration::ZERO;
                for _ in 0..iters {
                    let mut operator = new_operator();
                    let mut collector = EdgeCollector::new(operator.output_ports().to_vec());
                    feed(&mut operator, &job, &mut collector, "left", &left).await;
                    feed(&mut operator, &job, &mut collector, "right", &right).await;
                    let start = Instant::now();
                    operator.checkpoint(Epoch::INITIAL).unwrap();
                    total += start.elapsed();
                }
                total
            }
        });
    });
}

criterion_group!(benches, baseline);

fn main() {
    // Select a benchmark workload; argv[0] never establishes identity or authorization.
    let arguments = std::env::args().skip(1).collect::<Vec<_>>(); // nosemgrep: args
    if arguments.first().map(String::as_str) == Some("--matched-fixture") {
        run_matched(&arguments);
    } else if arguments.first().map(String::as_str) == Some("--native-component") {
        run_native_component(&arguments);
    } else {
        benches();
        Criterion::default().configure_from_args().final_summary();
    }
}

#[cfg(feature = "benchmark-internals")]
fn run_matched(arguments: &[String]) {
    let runtime = tokio::runtime::Runtime::new().unwrap();
    let report = runtime
        .block_on(calc_flow::benchmark_internals::run_fixture_cli(arguments))
        .unwrap();
    println!("{report}");
}

#[cfg(not(feature = "benchmark-internals"))]
fn run_matched(_arguments: &[String]) {
    panic!("matched fixture mode requires the benchmark-internals feature");
}

#[cfg(not(feature = "benchmark-internals"))]
fn run_native_component(_arguments: &[String]) {
    panic!("native component mode requires the benchmark-internals feature");
}

#[cfg(feature = "benchmark-internals")]
fn run_native_component(arguments: &[String]) {
    if let Some((component, check)) = native_join_components::parse_arguments(arguments).unwrap() {
        let runtime = tokio::runtime::Runtime::new().unwrap();
        let rows = if check { 24 } else { 1_000_000 };
        let mut observation = runtime.block_on(async {
            match component {
                native_join_components::Component::CaptureRestore => {
                    native_join_components::capture_restore(rows).await
                }
                native_join_components::Component::Compaction => {
                    native_join_components::compaction(rows, if check { 4 } else { 500 }).await
                }
                native_join_components::Component::LayoutControls => {
                    native_join_components::layout_controls().await
                }
            }
        });
        observation["check_only"] = check.into();
        observation.as_object_mut().unwrap().extend(
            calc_flow::benchmark_internals::process_observations()
                .unwrap()
                .as_object()
                .unwrap()
                .clone(),
        );
        println!("{observation}");
    }
}
