//! Resident controlled baseline for the bounded event-time stream join.
//!
//! Adopted from the frozen DAL-130 AC19 evidence harness (first baseline:
//! `main@f74f32a`, WSL2 i9-13900HX, workload SHA-256 `555546a7…`). The frozen
//! acceptance gates for future candidates are throughput floors at 0.80 × the
//! frozen 95% lower bound (`no_match` ≥ 1.12M rows/s, `one_to_one` ≥ 491k,
//! `fanout10` ≥ 75.0k, `evict` ≥ 135k) and recovery ceilings at 1.20 × the frozen
//! 95% upper bound (20k restore ≤ 59.3 ms, 60k restore ≤ 175.5 ms). Absolute
//! values are platform-keyed; slope and ratio conclusions must hold across
//! runs. The zero-match honey-key workaround from the scratch harness is gone:
//! the zero-row equality probe is well-defined since PR #189, so the no-match
//! scenario probes with truly disjoint keys.
//!
//! The two `checkpoint/capture_dirty_1250_base_*` scenarios are the capture
//! independence canary (DAL-160): equal dirty cardinality at two retained
//! state scales. Their costs must track the dirty set, not the retained
//! payload — before the FR47 fix they grew linearly with total state
//! (8.8 ms @17.5k → 67.4 ms @62.5k rows).

use std::collections::BTreeMap;
use std::sync::Arc;
use std::time::{Duration, Instant};

use calc_flow::{
    Batch, BatchMetadata, CancellationToken, EdgeCollector, Epoch, EventTime, IngressProgress,
    IngressProgressSnapshot, IngressState, JoinStateLimits, JoinTimeBounds, JsonMap,
    OperatorMetadata, OperatorStateSnapshot, StreamJobContext, StreamJoinOperator, StreamJoinSpec,
    StreamOperator, StreamOperatorContext,
};
use criterion::{Criterion, criterion_group, criterion_main};
use datafusion::arrow::array::{Int64Array, StringArray, TimestampMicrosecondArray};
use datafusion::arrow::datatypes::{DataType, Field, Schema, TimeUnit};
use datafusion::arrow::record_batch::RecordBatch;
use sha2::{Digest, Sha256};

const SECOND: i64 = 1_000_000;
const BASE_TS: i64 = 100 * SECOND;
const ROWS: usize = 10_000;
const FAN_KEYS: usize = 1_000;
const ROW_LIMIT: u64 = 4_000_000;
const BYTE_LIMIT: u64 = 4 * 1_024 * 1_024 * 1_024;
const MATCH_LIMIT: u64 = 100_000_000;
const BEFORE: Duration = Duration::from_secs(300);
const AFTER: Duration = Duration::from_secs(60);
/// Segment count that arms compaction on the next checkpoint preparation.
const COMPACTION_THRESHOLD: u64 = 4;

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
    let batch = RecordBatch::try_new(
        schema(),
        vec![
            Arc::new(StringArray::from(keys.to_vec())),
            Arc::new(TimestampMicrosecondArray::from(vec![ts; keys.len()])),
            Arc::new(Int64Array::from(vec![7_i64; keys.len()])),
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

/// Operator with `total_rows` of retained state across both sides and exactly
/// `COMPACTION_THRESHOLD` carried delta segments: the next checkpoint
/// preparation compacts them. Returns the operator and the next unused epoch.
async fn armed_operator(total_rows: usize) -> (StreamJoinOperator, u64) {
    let mut operator = new_operator();
    let job = job();
    let mut collector = EdgeCollector::new(operator.output_ports().to_vec());
    let bulk = unique_keys("B", total_rows);
    let (left_bulk, right_bulk) = bulk.split_at(bulk.len() / 2);
    feed(
        &mut operator,
        &job,
        &mut collector,
        "left",
        &make_batch(left_bulk, BASE_TS),
    )
    .await;
    feed(
        &mut operator,
        &job,
        &mut collector,
        "right",
        &make_batch(right_bulk, BASE_TS),
    )
    .await;
    let mut epoch = 1_u64;
    operator.checkpoint(Epoch::new(epoch).unwrap()).unwrap();
    // Three more tiny epochs arm the compaction threshold (4 segments).
    for index in 0..COMPACTION_THRESHOLD - 1 {
        let stamp = BASE_TS + SECOND * i64::try_from(index + 1).unwrap();
        feed(
            &mut operator,
            &job,
            &mut collector,
            "left",
            &make_batch(&["ZLhot0000".into()], stamp),
        )
        .await;
        feed(
            &mut operator,
            &job,
            &mut collector,
            "right",
            &make_batch(&["ZRhot0000".into()], stamp),
        )
        .await;
        epoch += 1;
        operator.checkpoint(Epoch::new(epoch).unwrap()).unwrap();
    }
    (operator, epoch + 1)
}

/// Compacted operator carrying `total_rows - dirty_rows` of base state plus
/// `dirty_rows` of fresh dirty ops above it: one call away from a
/// base-plus-delta checkpoint. The compaction itself already ran (untimed).
async fn dirty_operator(total_rows: usize, dirty_rows: usize) -> (StreamJoinOperator, u64) {
    assert!(total_rows > dirty_rows + 8);
    let (mut operator, next_epoch) = armed_operator(total_rows - dirty_rows).await;
    let job = job();
    let mut collector = EdgeCollector::new(operator.output_ports().to_vec());
    operator
        .prepare_checkpoint_async(&plain_ctx(&job))
        .await
        .unwrap();
    let dirty = unique_keys("D", dirty_rows);
    let (left_dirty, right_dirty) = dirty.split_at(dirty.len() / 2);
    let stamp = BASE_TS + 2 * SECOND;
    feed(
        &mut operator,
        &job,
        &mut collector,
        "left",
        &make_batch(left_dirty, stamp),
    )
    .await;
    feed(
        &mut operator,
        &job,
        &mut collector,
        "right",
        &make_batch(right_dirty, stamp),
    )
    .await;
    (operator, next_epoch)
}

/// Snapshot with a compacted base of ~`total_rows` plus a small delta.
async fn full_snapshot(total_rows: usize) -> OperatorStateSnapshot {
    let (mut operator, next_epoch) = dirty_operator(total_rows, 1_000).await;
    operator
        .checkpoint(Epoch::new(next_epoch).unwrap())
        .unwrap()
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

const FOCUSED_CHECKPOINT_ENV: &str = "CALC_FLOW_JOIN_PERF_FOCUSED";
const FOCUSED_CAPTURE: &str = "join/checkpoint/capture_dirty_1250_base_17500";
const FOCUSED_PREPARE: &str = "join/checkpoint/prepare_then_left_500_compact_60k";

#[derive(Clone, Copy)]
enum FocusedCheckpoint {
    Capture,
    Prepare,
}

fn focused_checkpoint_mode() -> bool {
    match std::env::var(FOCUSED_CHECKPOINT_ENV) {
        Err(std::env::VarError::NotPresent) => return false,
        Ok(mode) => assert_eq!(mode, "checkpoint-writer-v1", "invalid focused mode"),
        Err(error) => panic!("invalid focused mode: {error}"),
    }
    true
}

fn focused_checkpoint_case(name: &str) -> FocusedCheckpoint {
    match name {
        FOCUSED_CAPTURE => FocusedCheckpoint::Capture,
        FOCUSED_PREPARE => FocusedCheckpoint::Prepare,
        _ => panic!("focused mode requires one exact approved case"),
    }
}

fn focused_checkpoint_selection() -> Option<Option<FocusedCheckpoint>> {
    if !focused_checkpoint_mode() {
        return None;
    }
    let args = std::env::args().skip(1).collect::<Vec<_>>();
    let selected = match args.as_slice() {
        [flag] if flag == "--list" => None,
        [name, exact, bench] if exact == "--exact" && bench == "--bench" => {
            Some(focused_checkpoint_case(name))
        }
        _ => panic!("focused mode accepts --list or <approved-case> --exact --bench"),
    };
    Some(selected)
}

fn focused_checkpoint(
    c: &mut Criterion,
    runtime: &tokio::runtime::Runtime,
    selected: Option<FocusedCheckpoint>,
) {
    if let Some(selected) = selected {
        runtime.block_on(focused_checkpoint_probe(selected));
    }
    let mut group = c.benchmark_group("join");
    group.sample_size(30);
    group.warm_up_time(Duration::from_secs(1));
    group.measurement_time(Duration::from_secs(2));
    capture_dirty_scenario(&mut group, runtime, 17_500);
    prepare_compact_scenario(&mut group, runtime);
    let (rss, hwm) = read_rss();
    println!("JOIN_PERF_RSS rss_kib={rss} hwm_kib={hwm}");
    group.finish();
}

async fn focused_checkpoint_probe(selected: FocusedCheckpoint) {
    match selected {
        FocusedCheckpoint::Capture => {
            let (mut operator, next_epoch) = dirty_operator(17_500, 1_250).await;
            let status = operator.status();
            assert_eq!(status.left.retained_rows, 8_753);
            assert_eq!(status.right.retained_rows, 8_753);
            assert_eq!(status.emitted_match_rows, 0);
            let snapshot = operator
                .checkpoint(Epoch::new(next_epoch).unwrap())
                .unwrap();
            verify_checkpoint_probe(
                FOCUSED_CAPTURE,
                &snapshot,
                [8_128, 8_128],
                [8_753, 8_753],
                &["left", "right"],
            );
        }
        FocusedCheckpoint::Prepare => focused_prepare_probe().await,
    }
}

async fn focused_prepare_probe() {
    let (mut operator, next_epoch) = armed_operator(60_000).await;
    let job = job();
    let mut collector = EdgeCollector::new(operator.output_ports().to_vec());
    let batch = make_batch(&unique_keys("X", 500), BASE_TS + 3 * SECOND);
    operator
        .prepare_checkpoint_async(&plain_ctx(&job))
        .await
        .unwrap();
    feed(&mut operator, &job, &mut collector, "left", &batch).await;
    let status = operator.status();
    assert_eq!(status.left.retained_rows, 30_503);
    assert_eq!(status.right.retained_rows, 30_003);
    assert_eq!(status.emitted_match_rows, 0);
    let snapshot = operator
        .checkpoint(Epoch::new(next_epoch).unwrap())
        .unwrap();
    verify_checkpoint_probe(
        FOCUSED_PREPARE,
        &snapshot,
        [30_003, 30_003],
        [30_503, 30_003],
        &["left"],
    );
}

fn verify_checkpoint_probe(
    case: &str,
    snapshot: &OperatorStateSnapshot,
    base_rows: [u64; 2],
    retained_rows: [u64; 2],
    delta_sides: &[&str],
) {
    let metadata = &snapshot.inline_metadata;
    let layout = metadata["layout_version"].as_u64().expect("integer layout");
    assert_eq!(metadata["epoch"].as_u64(), Some(5));
    assert_eq!(metadata["ended"].as_bool(), Some(false));
    assert_eq!(metadata["metrics"]["emitted_match_rows"].as_u64(), Some(0));
    for (side, name) in ["left", "right"].into_iter().enumerate() {
        assert_eq!(
            metadata["metrics"][name]["retained_rows"].as_u64(),
            Some(retained_rows[side]),
        );
        let bytes = snapshot.segments[&format!("{name}-base")].bytes();
        verify_populated_base(bytes, layout, u8::try_from(side).unwrap(), base_rows[side]);
    }
    if layout == 2 {
        let inventory = &metadata["v2_inventory"];
        assert_eq!(inventory["codec_version"].as_u64(), Some(2));
        assert_eq!(inventory["base_epoch"].as_u64(), Some(4));
        assert_eq!(
            inventory["deltas"],
            serde_json::json!([{"epoch": 5, "sides": delta_sides}]),
        );
    }
    verify_delta_names(snapshot, delta_sides);
    let base_epoch = metadata
        .get("v2_inventory")
        .map(|inventory| &inventory["base_epoch"]);
    println!(
        "JOIN_PERF_FOCUSED_PROBE {}",
        serde_json::json!({
            "case": case,
            "layout_version": layout,
            "capture_epoch": 5,
            "base_epoch": base_epoch,
            "base_rows": base_rows,
            "retained_rows": retained_rows,
            "delta_sides": delta_sides,
            "emitted_match_rows": 0,
            "populated_base_verified": true,
        }),
    );
}

fn verify_populated_base(bytes: &[u8], layout: u64, side: u8, expected_rows: u64) {
    assert!(
        expected_rows > 0,
        "focused workload must have populated bases"
    );
    match layout {
        1 => {
            assert_eq!(bytes.get(..8), Some(&b"CFJOIN1\0"[..]));
            assert_eq!(frame_u64(bytes, 8), expected_rows);
        }
        2 => {
            assert_eq!(bytes.get(..8), Some(&b"CFJIDX2\0"[..]));
            assert_eq!(bytes.get(8..12), Some(&2_u32.to_le_bytes()[..]));
            assert_eq!(bytes.get(12..16), Some(&[side, 0, 0, 0][..]));
            assert_eq!(frame_u64(bytes, 16), expected_rows);
            assert_eq!(frame_u64(bytes, 24), 0);
        }
        _ => panic!("unsupported focused checkpoint layout {layout}"),
    }
}

fn frame_u64(bytes: &[u8], offset: usize) -> u64 {
    let encoded = bytes
        .get(offset..offset + 8)
        .expect("truncated base header");
    u64::from_le_bytes(encoded.try_into().expect("eight-byte integer"))
}

fn verify_delta_names(snapshot: &OperatorStateSnapshot, sides: &[&str]) {
    let actual = snapshot
        .segments
        .keys()
        .filter(|name| name.contains("-delta-"))
        .cloned()
        .collect::<Vec<_>>();
    let expected = sides
        .iter()
        .map(|side| format!("{side}-delta-5"))
        .collect::<Vec<_>>();
    assert_eq!(
        actual, expected,
        "prepared cut must carry only the new dirty epoch"
    );
}

fn baseline(c: &mut Criterion) {
    let focused = focused_checkpoint_selection();
    println!("JOIN_PERF_PROVENANCE {}", provenance());
    let runtime = tokio::runtime::Runtime::new().unwrap();
    if let Some(selected) = focused {
        focused_checkpoint(c, &runtime, selected);
        return;
    }
    let batches = ScenarioBatches::new();
    run_probe(&runtime, &batches);

    let mut group = c.benchmark_group("join");
    group.sample_size(30);
    group.warm_up_time(Duration::from_secs(1));
    group.measurement_time(Duration::from_secs(2));
    handler_scenarios(&mut group, &runtime, &batches);
    checkpoint_scenarios(&mut group, &runtime, &batches);
    compaction_scenarios(&mut group, &runtime);
    restore_scenarios(&mut group, &runtime);
    let (rss, hwm) = read_rss();
    println!("JOIN_PERF_RSS rss_kib={rss} hwm_kib={hwm}");
    group.finish();
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
        let mut probe = new_operator();
        let mut collector = EdgeCollector::new(probe.output_ports().to_vec());
        feed(&mut probe, &job, &mut collector, "left", &batches.left_fan).await;
        feed(&mut probe, &job, &mut collector, "right", &batches.right_fan).await;
        let status = probe.status();
        println!(
            "JOIN_PERF_PROBE fanout10 matches={} left_retained={}",
            status.emitted_match_rows, status.left.retained_rows,
        );

        // Fail-closed self-checks require checkpoint preparation to compact
        // the armed operator; dirty_operator must already carry that base.
        let (mut armed, next_epoch) = armed_operator(20_000).await;
        armed
            .prepare_checkpoint_async(&plain_ctx(&job))
            .await
            .unwrap();
        let mut collector = EdgeCollector::new(armed.output_ports().to_vec());
        feed(
            &mut armed,
            &job,
            &mut collector,
            "left",
            &make_batch(&unique_keys("X", 500), BASE_TS + 3 * SECOND),
        )
        .await;
        let snapshot = armed.checkpoint(Epoch::new(next_epoch).unwrap()).unwrap();
        assert!(
            snapshot.segments.contains_key("left-base"),
            "harness self-check failed: checkpoint preparation did not compact; segments: {:?}",
            snapshot.segments.keys().collect::<Vec<_>>()
        );
        let (mut compacted, next_epoch) = dirty_operator(20_000, 1_250).await;
        let snapshot = compacted.checkpoint(Epoch::new(next_epoch).unwrap()).unwrap();
        assert!(
            snapshot.segments.contains_key("left-base"),
            "harness self-check failed: setup compaction missing; segments: {:?}",
            snapshot.segments.keys().collect::<Vec<_>>()
        );
        let bytes: u64 = snapshot
            .segments
            .values()
            .map(|segment| segment.bytes().len() as u64)
            .sum();
        println!(
            "JOIN_PERF_PROBE snapshot_20k segments={} encoded_bytes={bytes}",
            snapshot
                .segments
                .keys()
                .cloned()
                .collect::<Vec<_>>()
                .join(",")
        );
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

    // Capture independence canary (DAL-160): equal dirty cardinality (1,250
    // rows) at two retained state scales. The capture cost must track the
    // dirty set — before the FR47 fix it grew linearly with total state.
    for total_rows in [17_500_usize, 62_500] {
        capture_dirty_scenario(group, runtime, total_rows);
    }
}

fn capture_dirty_scenario(
    group: &mut criterion::BenchmarkGroup<'_, criterion::measurement::WallTime>,
    runtime: &tokio::runtime::Runtime,
    total_rows: usize,
) {
    let name = format!("checkpoint/capture_dirty_1250_base_{total_rows}");
    group.bench_function(name, |b| {
        b.to_async(runtime).iter_custom(|iters| {
            Box::pin(async move {
                let mut total = Duration::ZERO;
                for _ in 0..iters {
                    let (mut operator, next_epoch) = dirty_operator(total_rows, 1_250).await;
                    let start = Instant::now();
                    operator
                        .checkpoint(Epoch::new(next_epoch).unwrap())
                        .unwrap();
                    total += start.elapsed();
                }
                total
            })
        });
    });
}

fn prepare_compact_scenario(
    group: &mut criterion::BenchmarkGroup<'_, criterion::measurement::WallTime>,
    runtime: &tokio::runtime::Runtime,
) {
    group.bench_function("checkpoint/prepare_then_left_500_compact_60k", |b| {
        b.to_async(runtime).iter_custom(|iters| {
            Box::pin(async move {
                let job = job();
                let mut total = Duration::ZERO;
                for _ in 0..iters {
                    let (mut operator, _) = armed_operator(60_000).await;
                    let mut collector = EdgeCollector::new(operator.output_ports().to_vec());
                    let batch = Arc::new(make_batch(&unique_keys("X", 500), BASE_TS + 3 * SECOND));
                    let start = Instant::now();
                    operator
                        .prepare_checkpoint_async(&plain_ctx(&job))
                        .await
                        .unwrap();
                    operator
                        .process_data("left", (*batch).clone(), &plain_ctx(&job), &mut collector)
                        .await
                        .unwrap();
                    total += start.elapsed();
                }
                total
            })
        });
    });
}

/// Checkpoint preparation plus a 500-row handler, with an armed 60k-row
/// compaction or an already-compacted control. Both time the same lifecycle.
fn compaction_scenarios(
    group: &mut criterion::BenchmarkGroup<'_, criterion::measurement::WallTime>,
    runtime: &tokio::runtime::Runtime,
) {
    prepare_compact_scenario(group, runtime);

    group.bench_function("checkpoint/prepare_then_left_500_steady_60k", |b| {
        b.to_async(runtime).iter_custom(|iters| {
            Box::pin(async move {
                let job = job();
                let mut total = Duration::ZERO;
                for _ in 0..iters {
                    let (mut operator, _) = dirty_operator(60_000, 1_250).await;
                    let mut collector = EdgeCollector::new(operator.output_ports().to_vec());
                    let batch = Arc::new(make_batch(&unique_keys("X", 500), BASE_TS + 3 * SECOND));
                    let start = Instant::now();
                    operator
                        .prepare_checkpoint_async(&plain_ctx(&job))
                        .await
                        .unwrap();
                    operator
                        .process_data("left", (*batch).clone(), &plain_ctx(&job), &mut collector)
                        .await
                        .unwrap();
                    total += start.elapsed();
                }
                total
            })
        });
    });
}

/// Scenario: full restore — base-plus-delta snapshot at scale.
fn restore_scenarios(
    group: &mut criterion::BenchmarkGroup<'_, criterion::measurement::WallTime>,
    runtime: &tokio::runtime::Runtime,
) {
    for total_rows in [20_000_usize, 60_000] {
        let name = format!("restore/full_{total_rows}");
        group.bench_function(name, |b| {
            b.to_async(runtime).iter_custom(|iters| {
                Box::pin(async move {
                    let snapshot = full_snapshot(total_rows).await;
                    let mut total = Duration::ZERO;
                    for _ in 0..iters {
                        let mut operator = new_operator();
                        let start = Instant::now();
                        operator.restore(&snapshot).unwrap();
                        total += start.elapsed();
                    }
                    total
                })
            });
        });
    }
}

criterion_group!(benches, baseline);
criterion_main!(benches);
