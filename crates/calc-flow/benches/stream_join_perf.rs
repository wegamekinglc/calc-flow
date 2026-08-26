//! Resident controlled baseline for the bounded event-time stream join.
//!
//! Adopted from the frozen DAL-130 AC19 evidence harness (first baseline:
//! `main@f74f32a`, WSL2 i9-13900HX, workload SHA-256 `555546a7…`). The frozen
//! acceptance gates for future candidates are throughput floors at 0.80 × the
//! frozen 95% lower bound (no_match ≥ 1.12M rows/s, one_to_one ≥ 491k,
//! fanout10 ≥ 75.0k, evict ≥ 135k) and recovery ceilings at 1.20 × the frozen
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
/// Segment count that arms inline compaction on the next data handler.
const COMPACTION_THRESHOLD: u64 = 4;

fn sha256(bytes: impl AsRef<[u8]>) -> String {
    hex::encode(Sha256::digest(bytes.as_ref()))
}

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
                .trim()
                .split_whitespace()
                .next()
                .unwrap_or("0")
                .parse()
                .unwrap_or(0);
        } else if let Some(rest) = line.strip_prefix("VmHWM:") {
            hwm = rest
                .trim()
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
/// `COMPACTION_THRESHOLD` carried delta segments: the next data handler will
/// compact inline. Returns the operator and the next unused epoch.
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
    // This data handler performs the inline compaction into a canonical base.
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

#[allow(
    clippy::too_many_lines,
    reason = "the frozen baseline protocol lays its scenarios out in one linear group"
)]
fn baseline(c: &mut Criterion) {
    println!("JOIN_PERF_PROVENANCE {}", provenance());
    let runtime = tokio::runtime::Runtime::new().unwrap();

    let left_unique = Arc::new(make_batch(&unique_keys("L", ROWS), BASE_TS));
    // Truly disjoint key sets: zero equality rows, the steady-state no-match
    // shape (the zero-row probe is well-defined since PR #189).
    let left_disjoint = Arc::new(make_batch(&unique_keys("L", ROWS), BASE_TS));
    let right_disjoint = Arc::new(make_batch(&unique_keys("R", ROWS), BASE_TS));
    let keys_one_to_one = Arc::new(unique_keys("K", ROWS));
    let left_one_to_one = Arc::new(make_batch(&keys_one_to_one, BASE_TS));
    let right_one_to_one = Arc::new(make_batch(&keys_one_to_one, BASE_TS));
    let left_fan = Arc::new(make_batch(&fan_keys(FAN_KEYS, ROWS), BASE_TS));
    let right_fan = Arc::new(make_batch(&fan_keys(FAN_KEYS, ROWS), BASE_TS));

    // Warm-up probe: prints the per-scenario state/match cardinality and RSS.
    runtime.block_on(async {
        let job = job();
        let mut probe = new_operator();
        let mut collector = EdgeCollector::new(probe.output_ports().to_vec());
        feed(&mut probe, &job, &mut collector, "left", &left_one_to_one).await;
        feed(&mut probe, &job, &mut collector, "right", &right_one_to_one).await;
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
        feed(&mut probe, &job, &mut collector, "left", &left_fan).await;
        feed(&mut probe, &job, &mut collector, "right", &right_fan).await;
        let status = probe.status();
        println!(
            "JOIN_PERF_PROBE fanout10 matches={} left_retained={}",
            status.emitted_match_rows, status.left.retained_rows,
        );

        // Fail-closed harness self-checks: the armed operator must compact on
        // the next data handler, and dirty_operator must already carry a base.
        let (mut armed, next_epoch) = armed_operator(20_000).await;
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
            "harness self-check failed: inline compaction did not trigger; segments: {:?}",
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

    let mut group = c.benchmark_group("join");
    group.sample_size(30);
    group.warm_up_time(Duration::from_secs(1));
    group.measurement_time(Duration::from_secs(2));

    // Scenario: no-match — right 10k probes 10k retained left rows, 0 pairs.
    group.bench_function("handler/right_10k_no_match", |b| {
        b.to_async(&runtime).iter_custom(|iters| {
            let left = Arc::clone(&left_disjoint);
            let right = Arc::clone(&right_disjoint);
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
        })
    });

    // Scenario: 1:1 — 10k pairs emitted against 10k retained left rows.
    group.bench_function("handler/right_10k_one_to_one", |b| {
        b.to_async(&runtime).iter_custom(|iters| {
            let left = Arc::clone(&left_one_to_one);
            let right = Arc::clone(&right_one_to_one);
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
        })
    });

    // Scenario: high fan-out — 1,000 keys × 10 rows per side → 100k pairs.
    group.bench_function("handler/right_10k_fanout10", |b| {
        b.to_async(&runtime).iter_custom(|iters| {
            let left = Arc::clone(&left_fan);
            let right = Arc::clone(&right_fan);
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
        })
    });

    // Scenario: watermark eviction — one progress call evicts 10k left rows.
    group.bench_function("handler/watermark_evict_10k", |b| {
        b.to_async(&runtime).iter_custom(|iters| {
            let left = Arc::clone(&left_unique);
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
        })
    });

    // Scenario: dirty checkpoint capture — 20k dirty rows, no base carried.
    group.bench_function("checkpoint/capture_dirty_20k", |b| {
        b.to_async(&runtime).iter_custom(|iters| {
            let left = Arc::clone(&left_one_to_one);
            let right = Arc::clone(&right_one_to_one);
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
        })
    });

    // Capture independence canary (DAL-160): equal dirty cardinality (1,250
    // rows) at two retained state scales. The capture cost must track the
    // dirty set — before the FR47 fix it grew linearly with total state.
    for total_rows in [17_500_usize, 62_500] {
        let name = format!("checkpoint/capture_dirty_1250_base_{total_rows}");
        group.bench_function(name, |b| {
            b.to_async(&runtime).iter_custom(|iters| {
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
            })
        });
    }

    // Inline compaction — a 500-row data handler that compacts a 60k-row base
    // inline vs the same handler on an already-compacted base.
    group.bench_function("handler/left_500_inline_compact_60k", |b| {
        b.to_async(&runtime).iter_custom(|iters| {
            Box::pin(async move {
                let job = job();
                let mut total = Duration::ZERO;
                for _ in 0..iters {
                    let (mut operator, _) = armed_operator(60_000).await;
                    let mut collector = EdgeCollector::new(operator.output_ports().to_vec());
                    let batch = Arc::new(make_batch(&unique_keys("X", 500), BASE_TS + 3 * SECOND));
                    let start = Instant::now();
                    operator
                        .process_data("left", (*batch).clone(), &plain_ctx(&job), &mut collector)
                        .await
                        .unwrap();
                    total += start.elapsed();
                }
                total
            })
        })
    });

    group.bench_function("handler/left_500_steady_60k_base", |b| {
        b.to_async(&runtime).iter_custom(|iters| {
            Box::pin(async move {
                let job = job();
                let mut total = Duration::ZERO;
                for _ in 0..iters {
                    let (mut operator, _) = dirty_operator(60_000, 1_250).await;
                    let mut collector = EdgeCollector::new(operator.output_ports().to_vec());
                    let batch = Arc::new(make_batch(&unique_keys("X", 500), BASE_TS + 3 * SECOND));
                    let start = Instant::now();
                    operator
                        .process_data("left", (*batch).clone(), &plain_ctx(&job), &mut collector)
                        .await
                        .unwrap();
                    total += start.elapsed();
                }
                total
            })
        })
    });

    // Scenario: full restore — base-plus-delta snapshot at scale.
    for total_rows in [20_000_usize, 60_000] {
        let name = format!("restore/full_{total_rows}");
        group.bench_function(name, |b| {
            b.to_async(&runtime).iter_custom(|iters| {
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
            })
        });
    }

    let (rss, hwm) = read_rss();
    println!("JOIN_PERF_RSS rss_kib={rss} hwm_kib={hwm}");
    group.finish();
}

criterion_group!(benches, baseline);
criterion_main!(benches);
