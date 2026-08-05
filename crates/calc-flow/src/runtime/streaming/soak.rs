use std::{
    collections::BTreeMap,
    process::Command,
    sync::{
        Arc,
        atomic::{AtomicUsize, Ordering},
    },
    time::Duration,
};

use async_trait::async_trait;
use datafusion::arrow::{array::Int64Array, record_batch::RecordBatch};
use parking_lot::Mutex;
use serde_json::json;

use super::{
    StreamJobContext,
    job::{
        ContinuousJobSpec, M2DeliveryMode, NamedSinkBinding, NamedSourceBinding,
        OrdinarySinkBinding, OrdinaryStreamSink,
    },
    runner::{ContinuousJobState, ContinuousRunner, TerminalCause},
    source_task::{
        AcceptedSequenceRecorder, Cursor, SourceBinding, SourceCapabilities, SourceEvent,
        StreamSource,
    },
};
use crate::{
    Batch, BatchKind, BatchMetadata, CancellationToken, EdgeBudget, JsonMap, PipelineBuilder, Port,
    Result, StreamRequirements, UdfRegistry, UnionOperator,
};

const CADENCE: Duration = Duration::from_secs(10);
const SAMPLE_COUNT: usize = 360;
const WARMUP_SAMPLES: usize = 60;
const FIVE_MINUTE_SAMPLES: usize = 30;
const MAX_RSS_SLOPE_MIB_PER_HOUR: f64 = 1.0;
const MAX_MEDIAN_GROWTH_KIB: u64 = 8 * 1024;
const EDGE_BUDGET: EdgeBudget = EdgeBudget {
    max_rows: 64,
    max_bytes: 1 << 20,
};

struct SoakSource {
    source_id: &'static str,
    sequence: u64,
    opened: Arc<AtomicUsize>,
    closed: Arc<AtomicUsize>,
}

#[async_trait]
impl StreamSource for SoakSource {
    async fn open(&mut self, _cursor: Option<Cursor>) -> Result<()> {
        self.opened.fetch_add(1, Ordering::SeqCst);
        Ok(())
    }

    async fn next(&mut self) -> Result<Option<SourceEvent>> {
        let sequence = self.sequence;
        self.sequence = sequence
            .checked_add(1)
            .expect("the one-hour soak cannot exhaust source sequence space");
        let record = RecordBatch::try_from_iter(vec![(
            "value",
            Arc::new(Int64Array::from(vec![i64::try_from(sequence).expect(
                "the one-hour soak cannot exhaust i64 source sequence space",
            )])) as _,
        )])
        .expect("soak table schema is stable");
        let batch = Batch::table(
            vec![record],
            BatchMetadata::new(self.source_id, sequence, BTreeMap::new())?,
        )?;
        let cursor = Cursor::new(sequence.to_be_bytes().to_vec(), JsonMap::new())?;
        Ok(Some(SourceEvent::Data { batch, cursor }))
    }

    async fn close(&mut self) -> Result<()> {
        self.closed.fetch_add(1, Ordering::SeqCst);
        Ok(())
    }

    fn capabilities(&self) -> SourceCapabilities {
        SourceCapabilities {
            replayable: true,
            max_batch_rows: 1,
            max_batch_bytes: 8,
        }
    }
}

#[derive(Default)]
struct DeliveryState {
    next: BTreeMap<String, u64>,
    sequences: BTreeMap<String, Vec<u64>>,
    total: u64,
    error: Option<String>,
}

impl DeliveryState {
    fn observe(&mut self, batch: &Batch) {
        let source = batch.metadata().source();
        let sequence = batch.metadata().sequence();
        let expected = self.next.entry(source.into()).or_default();
        if sequence != *expected && self.error.is_none() {
            self.error = Some(format!(
                "source {source:?} expected sequence {expected}, observed {sequence}"
            ));
        }
        self.sequences
            .entry(source.into())
            .or_default()
            .push(sequence);
        *expected = expected
            .checked_add(1)
            .expect("the one-hour soak cannot exhaust sink sequence space");
        self.total = self
            .total
            .checked_add(1)
            .expect("the one-hour soak cannot exhaust sink delivery space");
    }
}

struct SlowSoakSink {
    deliveries: Arc<Mutex<DeliveryState>>,
    opened: Arc<AtomicUsize>,
    closed: Arc<AtomicUsize>,
}

#[async_trait]
impl OrdinaryStreamSink for SlowSoakSink {
    async fn open(&mut self) -> Result<()> {
        self.opened.fetch_add(1, Ordering::SeqCst);
        Ok(())
    }

    async fn write(&mut self, batch: &Batch) -> Result<()> {
        tokio::time::sleep(Duration::from_millis(5)).await;
        self.deliveries.lock().observe(batch);
        Ok(())
    }

    async fn close(&mut self) -> Result<()> {
        self.closed.fetch_add(1, Ordering::SeqCst);
        Ok(())
    }
}

#[derive(Clone, Copy, Debug, PartialEq)]
struct RssSample {
    elapsed_seconds: f64,
    rss_kib: u64,
}

#[derive(Clone, Copy, Debug, PartialEq)]
struct RssGate {
    slope_mib_per_hour: f64,
    first_median_kib: u64,
    final_median_kib: u64,
    passed: bool,
}

fn parse_vm_rss_kib(status: &str) -> Option<u64> {
    status.lines().find_map(|line| {
        let mut fields = line.split_whitespace();
        if fields.next()? != "VmRSS:" {
            return None;
        }
        let kib = fields.next()?.parse().ok()?;
        (fields.next()? == "kB").then_some(kib)
    })
}

fn sample_count_as_f64(count: usize) -> f64 {
    f64::from(u32::try_from(count).expect("the bounded soak sample count fits u32"))
}

fn rss_kib_as_f64(rss_kib: u64) -> f64 {
    f64::from(u32::try_from(rss_kib).expect("a process VmRSS in KiB fits u32"))
}

fn elapsed_at_sample(index: usize) -> f64 {
    sample_count_as_f64(index + 1) * CADENCE.as_secs_f64()
}

fn least_squares_mib_per_hour(samples: &[RssSample]) -> Option<f64> {
    if samples.len() < 2 {
        return None;
    }
    let count = sample_count_as_f64(samples.len());
    let mean_x = samples
        .iter()
        .map(|sample| sample.elapsed_seconds)
        .sum::<f64>()
        / count;
    let mean_y = samples
        .iter()
        .map(|sample| rss_kib_as_f64(sample.rss_kib))
        .sum::<f64>()
        / count;
    let numerator = samples
        .iter()
        .map(|sample| (sample.elapsed_seconds - mean_x) * (rss_kib_as_f64(sample.rss_kib) - mean_y))
        .sum::<f64>();
    let denominator = samples
        .iter()
        .map(|sample| (sample.elapsed_seconds - mean_x).powi(2))
        .sum::<f64>();
    (denominator > 0.0).then_some(numerator / denominator * 3_600.0 / 1_024.0)
}

fn median_kib(values: &[u64]) -> Option<u64> {
    if values.is_empty() {
        return None;
    }
    let mut sorted = values.to_vec();
    sorted.sort_unstable();
    let middle = sorted.len() / 2;
    if sorted.len() % 2 == 0 {
        Some(sorted[middle - 1].saturating_add(sorted[middle]) / 2)
    } else {
        Some(sorted[middle])
    }
}

fn rss_gate_passed(slope_mib_per_hour: f64, first_median_kib: u64, final_median_kib: u64) -> bool {
    let slope_exceeded = slope_mib_per_hour > MAX_RSS_SLOPE_MIB_PER_HOUR;
    let median_exceeded = final_median_kib > first_median_kib.saturating_add(MAX_MEDIAN_GROWTH_KIB);
    !(slope_exceeded && median_exceeded)
}

fn evaluate_rss_gate(samples: &[RssSample]) -> Option<RssGate> {
    if samples.len() < SAMPLE_COUNT {
        return None;
    }
    let post_warmup = samples.get(WARMUP_SAMPLES..)?;
    let slope_mib_per_hour = least_squares_mib_per_hour(post_warmup)?;
    let first_median_kib = median_kib(
        &post_warmup
            .get(..FIVE_MINUTE_SAMPLES)?
            .iter()
            .map(|sample| sample.rss_kib)
            .collect::<Vec<_>>(),
    )?;
    let final_median_kib = median_kib(
        &post_warmup
            .get(post_warmup.len().checked_sub(FIVE_MINUTE_SAMPLES)?..)?
            .iter()
            .map(|sample| sample.rss_kib)
            .collect::<Vec<_>>(),
    )?;
    Some(RssGate {
        slope_mib_per_hour,
        first_median_kib,
        final_median_kib,
        passed: rss_gate_passed(slope_mib_per_hour, first_median_kib, final_median_kib),
    })
}

fn command_output(program: &str, arguments: &[&str]) -> String {
    Command::new(program)
        .args(arguments)
        .output()
        .ok()
        .filter(|output| output.status.success())
        .map_or_else(
            || "unavailable".into(),
            |output| String::from_utf8_lossy(&output.stdout).trim().to_owned(),
        )
}

fn assert_edge_budgets(status: &super::runner::ContinuousJobStatus) {
    for (edge, metrics) in &status.edges {
        assert!(
            metrics.charged_rows <= EDGE_BUDGET.max_rows
                && metrics.charged_bytes <= EDGE_BUDGET.max_bytes
                && metrics.high_water_rows <= EDGE_BUDGET.max_rows
                && metrics.high_water_bytes <= EDGE_BUDGET.max_bytes,
            "edge {edge:?} exceeded its budget: {metrics:?}"
        );
    }
}

fn assert_delivery_conservation(
    accepted: &BTreeMap<String, Vec<u64>>,
    sink: &DeliveryState,
    sink_name: &str,
) {
    assert_eq!(
        delivery_conservation_issue(accepted, sink),
        None,
        "{sink_name} did not conserve the independent accepted oracle"
    );
}

fn delivery_conservation_issue(
    accepted: &BTreeMap<String, Vec<u64>>,
    sink: &DeliveryState,
) -> Option<&'static str> {
    if sink.error.is_some() {
        Some("order or duplicate failure")
    } else if sink.sequences != *accepted {
        Some("missing accepted data")
    } else if usize::try_from(sink.total).ok()
        != Some(accepted.values().map(Vec::len).sum::<usize>())
    {
        Some("accepted and delivered totals differ")
    } else {
        None
    }
}

#[test]
fn independent_accepted_oracle_exposes_commit_after_downstream_drop() {
    let accepted = AcceptedSequenceRecorder::default();
    accepted.record_committed_for_test("left", 0);
    let sink_after_injected_drop = DeliveryState::default();

    assert_eq!(
        delivery_conservation_issue(&accepted.snapshot(), &sink_after_injected_drop),
        Some("missing accepted data")
    );
}

fn soak_spec(
    source_opened: &Arc<AtomicUsize>,
    source_closed: &Arc<AtomicUsize>,
    sink_opened: &Arc<AtomicUsize>,
    sink_closed: &Arc<AtomicUsize>,
    sink_a: &Arc<Mutex<DeliveryState>>,
    sink_b: &Arc<Mutex<DeliveryState>>,
    accepted: &AcceptedSequenceRecorder,
) -> ContinuousJobSpec {
    let union = UnionOperator::new(
        "merge",
        vec![
            Port::new("left", BatchKind::Table, true, None).unwrap(),
            Port::new("right", BatchKind::Table, true, None).unwrap(),
        ],
    )
    .unwrap();
    let plan = PipelineBuilder::new("continuous-runtime-soak")
        .unwrap()
        .add_node("merge", Box::new(union))
        .unwrap()
        .compile_stream(
            &UdfRegistry::new().snapshot(),
            &StreamRequirements::default(),
        )
        .unwrap();
    let sources = ["left", "right"]
        .into_iter()
        .map(|source_id| {
            let binding = SourceBinding::new(
                Box::new(SoakSource {
                    source_id,
                    sequence: 0,
                    opened: Arc::clone(source_opened),
                    closed: Arc::clone(source_closed),
                }),
                None,
                0,
            )
            .unwrap()
            .with_accepted_sequence_recorder(accepted.clone());
            NamedSourceBinding {
                binding_id: source_id.into(),
                binding,
            }
        })
        .collect();
    let sinks = [("sink-a", sink_a), ("sink-b", sink_b)]
        .into_iter()
        .map(|(sink_id, deliveries)| NamedSinkBinding {
            output_id: "output".into(),
            sink_id: sink_id.into(),
            binding: OrdinarySinkBinding::new(Box::new(SlowSoakSink {
                deliveries: Arc::clone(deliveries),
                opened: Arc::clone(sink_opened),
                closed: Arc::clone(sink_closed),
            })),
        })
        .collect();
    ContinuousJobSpec {
        context: StreamJobContext::new(
            1,
            plan.fingerprint(),
            JsonMap::new(),
            None,
            CancellationToken::new(),
        ),
        plan,
        sources,
        sinks,
        edge_budget: EDGE_BUDGET,
        delivery_mode: M2DeliveryMode::ProcessLocalOrdered,
    }
}

#[cfg(target_os = "linux")]
#[allow(
    clippy::too_many_lines,
    reason = "the ignored soak keeps its runtime topology, samples, and convergence assertions together"
)]
async fn run_linux_soak() {
    let source_opened = Arc::new(AtomicUsize::new(0));
    let source_closed = Arc::new(AtomicUsize::new(0));
    let sink_opened = Arc::new(AtomicUsize::new(0));
    let sink_closed = Arc::new(AtomicUsize::new(0));
    let sink_a = Arc::new(Mutex::new(DeliveryState::default()));
    let sink_b = Arc::new(Mutex::new(DeliveryState::default()));
    let accepted = AcceptedSequenceRecorder::default();
    let mut runner = ContinuousRunner::new();
    let job = runner
        .start(soak_spec(
            &source_opened,
            &source_closed,
            &sink_opened,
            &sink_closed,
            &sink_a,
            &sink_b,
            &accepted,
        ))
        .await
        .expect("real continuous runtime soak must launch");

    println!(
        "{}",
        json!({
            "type": "calc_flow_stream_soak_metadata",
            "runtime_path": "ContinuousRunner/source/operator/sink/supervisor/reaper",
            "commit": command_output("git", &["rev-parse", "HEAD"]),
            "kernel": command_output("uname", &["-sr"]),
            "rustc": command_output("rustc", &["--version"]),
            "allocator": "system",
            "rss_source": "/proc/self/status:VmRSS",
            "cadence_seconds": CADENCE.as_secs(),
            "sample_count": SAMPLE_COUNT,
            "warmup_samples": WARMUP_SAMPLES,
        })
    );

    let initial_status = job.status();
    let steady_task_count = initial_status.tasks.len();
    assert!(steady_task_count > 0, "supervisor task registry is empty");
    assert_edge_budgets(&initial_status);
    let mut samples = Vec::with_capacity(SAMPLE_COUNT);
    for index in 0..SAMPLE_COUNT {
        tokio::time::sleep(CADENCE).await;
        let process_status = tokio::fs::read_to_string("/proc/self/status")
            .await
            .expect("Linux soak could not read /proc/self/status");
        let rss_kib = parse_vm_rss_kib(&process_status).expect("Linux status omitted VmRSS");
        let elapsed_seconds = elapsed_at_sample(index);
        samples.push(RssSample {
            elapsed_seconds,
            rss_kib,
        });
        let status = job.status();
        assert_edge_budgets(&status);
        assert_eq!(
            status.tasks.len(),
            steady_task_count,
            "supervisor task count changed"
        );
        let queues = status
            .edges
            .iter()
            .map(|(edge, metrics)| {
                json!({
                    "edge": edge,
                    "rows": metrics.charged_rows,
                    "bytes": metrics.charged_bytes,
                    "high_water_rows": metrics.high_water_rows,
                    "high_water_bytes": metrics.high_water_bytes,
                })
            })
            .collect::<Vec<_>>();
        println!(
            "{}",
            json!({
                "type": "calc_flow_stream_soak_sample",
                "index": index,
                "elapsed_seconds": elapsed_seconds,
                "vmrss_kib": rss_kib,
                "task_count": steady_task_count,
                "queues": queues,
            })
        );
    }

    let outcome = job.shutdown().await;
    assert_eq!(outcome.state, ContinuousJobState::Completed);
    assert_eq!(outcome.cause, TerminalCause::GracefulShutdown);
    let accepted = accepted.snapshot();
    let accepted_total = accepted.values().map(Vec::len).sum::<usize>();
    drop(job);
    runner.shutdown().await.unwrap();
    assert_eq!(source_opened.load(Ordering::SeqCst), 2);
    assert_eq!(source_closed.load(Ordering::SeqCst), 2);
    assert_eq!(sink_opened.load(Ordering::SeqCst), 2);
    assert_eq!(sink_closed.load(Ordering::SeqCst), 2);

    let sink_a = sink_a.lock();
    let sink_b = sink_b.lock();
    assert_delivery_conservation(&accepted, &sink_a, "sink A");
    assert_delivery_conservation(&accepted, &sink_b, "sink B");

    let gate = evaluate_rss_gate(&samples).expect("soak RSS sample set was incomplete");
    println!(
        "{}",
        json!({
            "type": "calc_flow_stream_soak_result",
            "samples": samples.len(),
            "slope_mib_per_hour": gate.slope_mib_per_hour,
            "first_post_warmup_five_minute_median_kib": gate.first_median_kib,
            "final_five_minute_median_kib": gate.final_median_kib,
            "passed": gate.passed,
            "accepted_batches": accepted_total,
            "sink_a_batches": sink_a.total,
            "sink_b_batches": sink_b.total,
            "missing": 0,
            "duplicate": 0,
        })
    );
    assert!(gate.passed, "RSS guard failed: {gate:?}");
}

#[tokio::test]
#[ignore = "one-hour opt-in streaming soak; set CALC_FLOW_STREAM_SOAK=1"]
async fn one_hour_two_source_slow_sink() {
    if std::env::var("CALC_FLOW_STREAM_SOAK").as_deref() != Ok("1") {
        println!(
            "{}",
            json!({
                "type": "calc_flow_stream_soak_skip",
                "reason": "opt_in_required",
                "required_environment": "CALC_FLOW_STREAM_SOAK=1",
            })
        );
        return;
    }
    #[cfg(target_os = "linux")]
    run_linux_soak().await;
    #[cfg(not(target_os = "linux"))]
    println!(
        "{}",
        json!({
            "type": "calc_flow_stream_soak_skip",
            "reason": "unsupported_platform",
            "required_platform": "linux",
        })
    );
}

#[tokio::test]
async fn real_soak_topology_smoke_converges_through_the_reaper() {
    let source_opened = Arc::new(AtomicUsize::new(0));
    let source_closed = Arc::new(AtomicUsize::new(0));
    let sink_opened = Arc::new(AtomicUsize::new(0));
    let sink_closed = Arc::new(AtomicUsize::new(0));
    let sink_a = Arc::new(Mutex::new(DeliveryState::default()));
    let sink_b = Arc::new(Mutex::new(DeliveryState::default()));
    let accepted = AcceptedSequenceRecorder::default();
    let mut runner = ContinuousRunner::new();
    let job = runner
        .start(soak_spec(
            &source_opened,
            &source_closed,
            &sink_opened,
            &sink_closed,
            &sink_a,
            &sink_b,
            &accepted,
        ))
        .await
        .unwrap();
    tokio::time::sleep(Duration::from_millis(30)).await;
    let status = job.status();
    assert!(!status.tasks.is_empty());
    assert_edge_budgets(&status);

    let outcome = job.wait();
    drop(job);
    assert_eq!(outcome.await.cause, TerminalCause::ExplicitCancel);
    runner.shutdown().await.unwrap();

    assert_eq!(source_opened.load(Ordering::SeqCst), 2);
    assert_eq!(source_closed.load(Ordering::SeqCst), 2);
    assert_eq!(sink_opened.load(Ordering::SeqCst), 2);
    assert_eq!(sink_closed.load(Ordering::SeqCst), 2);
    assert_eq!(sink_a.lock().error, None);
    assert_eq!(sink_b.lock().error, None);
}

#[tokio::test]
async fn real_soak_topology_graceful_smoke_conserves_every_accepted_sequence() {
    let source_opened = Arc::new(AtomicUsize::new(0));
    let source_closed = Arc::new(AtomicUsize::new(0));
    let sink_opened = Arc::new(AtomicUsize::new(0));
    let sink_closed = Arc::new(AtomicUsize::new(0));
    let sink_a = Arc::new(Mutex::new(DeliveryState::default()));
    let sink_b = Arc::new(Mutex::new(DeliveryState::default()));
    let accepted = AcceptedSequenceRecorder::default();
    let mut runner = ContinuousRunner::new();
    let job = runner
        .start(soak_spec(
            &source_opened,
            &source_closed,
            &sink_opened,
            &sink_closed,
            &sink_a,
            &sink_b,
            &accepted,
        ))
        .await
        .unwrap();
    tokio::time::sleep(Duration::from_millis(30)).await;

    let outcome = job.shutdown().await;

    assert_eq!(outcome.state, ContinuousJobState::Completed);
    assert_eq!(outcome.cause, TerminalCause::GracefulShutdown);
    let accepted = accepted.snapshot();
    assert_delivery_conservation(&accepted, &sink_a.lock(), "sink A");
    assert_delivery_conservation(&accepted, &sink_b.lock(), "sink B");
    assert_eq!(source_opened.load(Ordering::SeqCst), 2);
    assert_eq!(source_closed.load(Ordering::SeqCst), 2);
    assert_eq!(sink_opened.load(Ordering::SeqCst), 2);
    assert_eq!(sink_closed.load(Ordering::SeqCst), 2);
    drop(job);
    runner.shutdown().await.unwrap();
}

#[test]
fn parses_linux_vm_rss_without_accepting_other_units() {
    assert_eq!(
        parse_vm_rss_kib("Name:\tcalc\nVmRSS:\t  12345 kB\nThreads:\t4\n"),
        Some(12_345)
    );
    assert_eq!(parse_vm_rss_kib("VmRSS: 12 MB"), None);
    assert_eq!(parse_vm_rss_kib("VmSize: 12 kB"), None);
}

#[test]
fn least_squares_reports_mib_per_hour() {
    let samples = [
        RssSample {
            elapsed_seconds: 0.0,
            rss_kib: 10_000,
        },
        RssSample {
            elapsed_seconds: 1_800.0,
            rss_kib: 10_512,
        },
        RssSample {
            elapsed_seconds: 3_600.0,
            rss_kib: 11_024,
        },
    ];
    let slope = least_squares_mib_per_hour(&samples).unwrap();
    assert!((slope - 1.0).abs() < f64::EPSILON);
}

#[test]
fn median_handles_even_and_odd_windows() {
    assert_eq!(median_kib(&[9, 1, 5]), Some(5));
    assert_eq!(median_kib(&[10, 2, 6, 4]), Some(5));
    assert_eq!(median_kib(&[]), None);
}

#[test]
fn rss_gate_allows_only_slope_threshold_to_be_exceeded() {
    assert!(rss_gate_passed(
        MAX_RSS_SLOPE_MIB_PER_HOUR + 1.0,
        100_000,
        100_000,
    ));
}

#[test]
fn rss_gate_allows_only_median_threshold_to_be_exceeded() {
    assert!(rss_gate_passed(
        0.0,
        100_000,
        100_000 + MAX_MEDIAN_GROWTH_KIB + 1,
    ));
}

#[test]
fn rss_gate_fails_when_both_thresholds_are_exceeded() {
    assert!(!rss_gate_passed(
        MAX_RSS_SLOPE_MIB_PER_HOUR + 1.0,
        100_000,
        100_000 + MAX_MEDIAN_GROWTH_KIB + 1,
    ));
}

#[test]
fn evaluates_a_stable_full_sample_set() {
    let stable = (0..SAMPLE_COUNT)
        .map(|index| RssSample {
            elapsed_seconds: elapsed_at_sample(index),
            rss_kib: 100_000,
        })
        .collect::<Vec<_>>();
    assert!(evaluate_rss_gate(&stable).unwrap().passed);
}
