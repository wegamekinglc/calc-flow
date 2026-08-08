use std::{
    collections::{BTreeMap, BTreeSet},
    process::Command,
    sync::{
        Arc,
        atomic::{AtomicUsize, Ordering},
    },
    time::Duration,
};

use async_trait::async_trait;
use datafusion::arrow::{
    array::{Int64Array, StringArray, TimestampMicrosecondArray},
    datatypes::{DataType, Field, Schema, TimeUnit},
    record_batch::RecordBatch,
};
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
    AggregateFunction, Batch, BatchKind, BatchMetadata, CancellationToken, Edge, EdgeBudget, Epoch,
    EventTime, JsonMap, OperatorMetadata, OperatorStateSnapshot, PipelineBuilder, Port,
    PortEndpoint, Result, StreamCollector, StreamOperator, StreamOperatorContext,
    StreamRequirements, UdfRegistry, UnionOperator, WindowAggregateOperator, WindowSpec,
};

const CADENCE: Duration = Duration::from_secs(10);
const TARGET_DURATION: Duration = Duration::from_secs(1_200);
const SAMPLE_COUNT: usize = 120;
const WARMUP_SAMPLES: usize = 30;
const FIVE_MINUTE_SAMPLES: usize = 30;
const MAX_RSS_SLOPE_MIB_PER_HOUR: f64 = 1.0;
const MAX_MEDIAN_GROWTH_KIB: u64 = 8 * 1024;
const STANDARD_SOAK_SINK_DELAY: Duration = Duration::from_millis(500);
const SMOKE_SINK_DELAY: Duration = Duration::from_millis(5);
const SECONDS_PER_HOUR: f64 = 60.0 * 60.0;
const SOAK_COMMAND: &str = "CALC_FLOW_STREAM_SOAK=1 cargo test -p calc-flow --lib runtime::streaming::soak::twenty_minute_two_source_slow_sink -- --ignored --exact --nocapture";
const EDGE_BUDGET: EdgeBudget = EdgeBudget {
    max_rows: 64,
    max_bytes: 1 << 20,
};
const SOAK_WINDOW_MICROS: i64 = 64;
const SOAK_CHECKPOINT_BATCH_INTERVAL: u64 = 8;

fn soak_schema() -> Arc<Schema> {
    Arc::new(Schema::new(vec![
        Field::new(
            "event_time",
            DataType::Timestamp(TimeUnit::Microsecond, Some(Arc::from("UTC"))),
            false,
        ),
        Field::new("group", DataType::Utf8, false),
        Field::new("value", DataType::Int64, false),
    ]))
}

struct SoakSource {
    source_id: &'static str,
    sequence: u64,
    pending_watermark: Option<EventTime>,
    opened: Arc<AtomicUsize>,
    closed: Arc<AtomicUsize>,
}

fn soak_batch(source_id: &str, sequence: u64) -> Result<Batch> {
    let (event_times, groups, values) = if sequence % 4 == 0 {
        (Vec::new(), Vec::new(), Vec::new())
    } else {
        let value = i64::try_from(sequence)
            .expect("the twenty-minute soak cannot exhaust i64 source sequence space");
        (
            vec![value],
            vec![format!("{source_id}-{:02}", sequence % 64)],
            vec![value],
        )
    };
    let record = RecordBatch::try_new(
        soak_schema(),
        vec![
            Arc::new(TimestampMicrosecondArray::from(event_times).with_timezone("UTC")),
            Arc::new(StringArray::from(groups)),
            Arc::new(Int64Array::from(values)),
        ],
    )
    .expect("soak table schema is stable");
    Batch::table(
        vec![record],
        BatchMetadata::new(source_id, sequence, BTreeMap::new())?,
    )
}

#[async_trait]
impl StreamSource for SoakSource {
    async fn open(&mut self, _cursor: Option<Cursor>) -> Result<()> {
        self.opened.fetch_add(1, Ordering::SeqCst);
        Ok(())
    }

    async fn next(&mut self) -> Result<Option<SourceEvent>> {
        if let Some(watermark) = self.pending_watermark.take() {
            return Ok(Some(SourceEvent::Watermark(watermark)));
        }
        let sequence = self.sequence;
        self.sequence = sequence
            .checked_add(1)
            .expect("the twenty-minute soak cannot exhaust source sequence space");
        self.pending_watermark = Some(EventTime::from_micros(
            i64::try_from(self.sequence)
                .expect("the twenty-minute soak cannot exhaust event-time space"),
        ));
        let batch = soak_batch(self.source_id, sequence)?;
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
            max_batch_bytes: 256,
        }
    }
}

#[derive(Clone, Debug, Default, Eq, PartialEq)]
struct WindowSoakProbe {
    checkpoints: u64,
    compactions: u64,
    total_segment_bytes: u64,
    max_new_segments: usize,
    max_retained_segments: usize,
    live_keys: usize,
    max_live_keys: usize,
    terminal_live_keys: Option<usize>,
}

struct SoakWindowOperator {
    inner: WindowAggregateOperator,
    probe: Arc<Mutex<WindowSoakProbe>>,
    live_keys: BTreeSet<(i64, i64, String)>,
    processed_batches: u64,
    next_epoch: u64,
}

impl SoakWindowOperator {
    fn new(probe: Arc<Mutex<WindowSoakProbe>>) -> Self {
        let spec = WindowSpec::tumbling(
            "event_time",
            Duration::from_micros(
                u64::try_from(SOAK_WINDOW_MICROS).expect("the soak window is positive"),
            ),
        )
        .expect("the soak window geometry is valid")
        .group_by(["group"])
        .expect("the soak group declaration is valid")
        .aggregate(AggregateFunction::Sum, "value", "sum_value")
        .expect("the soak aggregate declaration is valid");
        Self {
            inner: WindowAggregateOperator::new("window", soak_schema(), spec)
                .expect("the soak window compiles"),
            probe,
            live_keys: BTreeSet::new(),
            processed_batches: 0,
            next_epoch: 1,
        }
    }

    fn input_keys(batch: &Batch) -> Vec<(i64, i64, String)> {
        batch
            .table_payload()
            .expect("the soak source always emits table batches")
            .batches()
            .iter()
            .flat_map(|record| {
                let event_times = record
                    .column_by_name("event_time")
                    .expect("the soak schema has event_time")
                    .as_any()
                    .downcast_ref::<TimestampMicrosecondArray>()
                    .expect("the soak event time has its declared type");
                let groups = record
                    .column_by_name("group")
                    .expect("the soak schema has group")
                    .as_any()
                    .downcast_ref::<StringArray>()
                    .expect("the soak group has its declared type");
                (0..record.num_rows()).map(move |row| {
                    let start =
                        event_times.value(row).div_euclid(SOAK_WINDOW_MICROS) * SOAK_WINDOW_MICROS;
                    (
                        start,
                        start + SOAK_WINDOW_MICROS,
                        groups.value(row).to_owned(),
                    )
                })
            })
            .collect()
    }

    fn refresh_live_key_probe(&self, terminal: bool) {
        let mut probe = self.probe.lock();
        probe.live_keys = self.live_keys.len();
        probe.max_live_keys = probe.max_live_keys.max(probe.live_keys);
        if terminal {
            probe.terminal_live_keys = Some(probe.live_keys);
        }
    }

    fn capture_prepared_state(&mut self) -> Result<()> {
        let epoch =
            Epoch::new(self.next_epoch).expect("the twenty-minute soak never captures epoch zero");
        self.next_epoch = self
            .next_epoch
            .checked_add(1)
            .expect("the twenty-minute soak cannot exhaust epochs");
        let snapshot = self.inner.checkpoint(epoch)?;
        let retained_segments = snapshot
            .inline_metadata
            .get("segment_ids")
            .and_then(serde_json::Value::as_array)
            .map_or(0, Vec::len);
        let segment_bytes = snapshot
            .segments
            .values()
            .try_fold(0_u64, |total, bytes| {
                total.checked_add(u64::try_from(bytes.len()).ok()?)
            })
            .expect("the bounded soak cannot overflow segment byte accounting");
        let mut probe = self.probe.lock();
        probe.checkpoints = probe
            .checkpoints
            .checked_add(1)
            .expect("the twenty-minute soak cannot exhaust checkpoint counts");
        probe.total_segment_bytes = probe
            .total_segment_bytes
            .checked_add(segment_bytes)
            .expect("the twenty-minute soak cannot exhaust segment byte accounting");
        probe.max_new_segments = probe.max_new_segments.max(snapshot.segments.len());
        probe.max_retained_segments = probe.max_retained_segments.max(retained_segments);
        if snapshot
            .segments
            .keys()
            .any(|segment_id| segment_id.starts_with("base-"))
        {
            probe.compactions = probe
                .compactions
                .checked_add(1)
                .expect("the twenty-minute soak cannot exhaust compaction counts");
        }
        Ok(())
    }
}

impl OperatorMetadata for SoakWindowOperator {
    fn name(&self) -> &str {
        self.inner.name()
    }

    fn input_ports(&self) -> &[Port] {
        self.inner.input_ports()
    }

    fn output_ports(&self) -> &[Port] {
        self.inner.output_ports()
    }

    fn configuration(&self) -> JsonMap {
        self.inner.configuration()
    }
}

#[async_trait]
impl StreamOperator for SoakWindowOperator {
    async fn process_data(
        &mut self,
        ingress: &str,
        batch: Batch,
        context: &StreamOperatorContext<'_>,
        output: &mut dyn StreamCollector,
    ) -> Result<()> {
        let new_keys = Self::input_keys(&batch);
        self.inner
            .process_data(ingress, batch, context, output)
            .await?;
        self.live_keys.extend(new_keys);
        self.processed_batches = self
            .processed_batches
            .checked_add(1)
            .expect("the twenty-minute soak cannot exhaust batch counts");
        if self
            .processed_batches
            .is_multiple_of(SOAK_CHECKPOINT_BATCH_INTERVAL)
        {
            self.capture_prepared_state()?;
        }
        self.refresh_live_key_probe(false);
        Ok(())
    }

    async fn on_watermark(
        &mut self,
        watermark: EventTime,
        context: &StreamOperatorContext<'_>,
        output: &mut dyn StreamCollector,
    ) -> Result<()> {
        self.inner.on_watermark(watermark, context, output).await?;
        self.live_keys
            .retain(|(_, end, _)| *end > watermark.as_micros());
        self.refresh_live_key_probe(false);
        Ok(())
    }

    async fn on_end(
        &mut self,
        context: &StreamOperatorContext<'_>,
        output: &mut dyn StreamCollector,
    ) -> Result<()> {
        self.inner.on_end(context, output).await?;
        self.live_keys.clear();
        self.capture_prepared_state()?;
        self.refresh_live_key_probe(true);
        Ok(())
    }

    fn checkpoint(&mut self, epoch: Epoch) -> Result<OperatorStateSnapshot> {
        self.inner.checkpoint(epoch)
    }

    fn restore(&mut self, snapshot: &OperatorStateSnapshot) -> Result<()> {
        self.inner.restore(snapshot)
    }

    fn reset(&mut self) -> Result<()> {
        self.live_keys.clear();
        self.processed_batches = 0;
        self.next_epoch = 1;
        self.refresh_live_key_probe(false);
        self.inner.reset()
    }
}

#[derive(Default)]
struct DeliveryState {
    next_sequence: u64,
    rows: BTreeMap<(i64, i64, String), i64>,
    total: u64,
    error: Option<String>,
}

impl DeliveryState {
    fn observe(&mut self, batch: &Batch) {
        let sequence = batch.metadata().sequence();
        if batch.metadata().source() != "window" && self.error.is_none() {
            self.error = Some("window output used an unexpected source ID".into());
        }
        if sequence != self.next_sequence && self.error.is_none() {
            self.error = Some(format!(
                "window output expected sequence {}, observed {sequence}",
                self.next_sequence
            ));
        }
        self.next_sequence = self
            .next_sequence
            .checked_add(1)
            .expect("the twenty-minute soak cannot exhaust sink sequence space");
        self.total = self
            .total
            .checked_add(1)
            .expect("the twenty-minute soak cannot exhaust sink delivery space");

        for record in batch
            .table_payload()
            .expect("the soak window emits table batches")
            .batches()
        {
            let starts = record
                .column_by_name("window_start")
                .expect("the soak output has window_start")
                .as_any()
                .downcast_ref::<TimestampMicrosecondArray>()
                .expect("the soak output start type is stable");
            let ends = record
                .column_by_name("window_end")
                .expect("the soak output has window_end")
                .as_any()
                .downcast_ref::<TimestampMicrosecondArray>()
                .expect("the soak output end type is stable");
            let groups = record
                .column_by_name("group")
                .expect("the soak output has group")
                .as_any()
                .downcast_ref::<StringArray>()
                .expect("the soak output group type is stable");
            let sums = record
                .column_by_name("sum_value")
                .expect("the soak output has sum_value")
                .as_any()
                .downcast_ref::<Int64Array>()
                .expect("the soak output sum type is stable");
            for row in 0..record.num_rows() {
                let key = (
                    starts.value(row),
                    ends.value(row),
                    groups.value(row).to_owned(),
                );
                if self.rows.insert(key, sums.value(row)).is_some() && self.error.is_none() {
                    self.error = Some("window output emitted a duplicate final row".into());
                }
            }
        }
    }
}

struct SlowSoakSink {
    deliveries: Arc<Mutex<DeliveryState>>,
    opened: Arc<AtomicUsize>,
    closed: Arc<AtomicUsize>,
    write_delay: Duration,
}

#[async_trait]
impl OrdinaryStreamSink for SlowSoakSink {
    async fn open(&mut self) -> Result<()> {
        self.opened.fetch_add(1, Ordering::SeqCst);
        Ok(())
    }

    async fn write(&mut self, batch: &Batch) -> Result<()> {
        tokio::time::sleep(self.write_delay).await;
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

async fn wait_for_sample_deadline(started: tokio::time::Instant, index: usize) -> f64 {
    let sample_number = u32::try_from(index + 1).expect("the bounded soak sample count fits u32");
    tokio::time::sleep_until(started + CADENCE * sample_number).await;
    started.elapsed().as_secs_f64()
}

fn observed_timeline_issue(samples: &[RssSample]) -> Option<&'static str> {
    if samples.len() != SAMPLE_COUNT {
        return Some("sample count differs from the soak contract");
    }
    for (index, sample) in samples.iter().enumerate() {
        if sample.elapsed_seconds < elapsed_at_sample(index) {
            return Some("sample preceded its absolute deadline");
        }
        if index > 0 && sample.elapsed_seconds <= samples[index - 1].elapsed_seconds {
            return Some("observed sample timestamps are not strictly increasing");
        }
    }
    (samples.last()?.elapsed_seconds < TARGET_DURATION.as_secs_f64())
        .then_some("observed timeline ended before the soak target")
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
    (denominator > 0.0).then_some(numerator / denominator * SECONDS_PER_HOUR / 1_024.0)
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
    if samples.len() != SAMPLE_COUNT {
        return None;
    }
    let post_warmup = samples.get(WARMUP_SAMPLES..)?;
    let slope_mib_per_hour = least_squares_mib_per_hour(post_warmup)?;
    let (first_median_kib, final_median_kib) = rss_window_medians(post_warmup)?;
    Some(RssGate {
        slope_mib_per_hour,
        first_median_kib,
        final_median_kib,
        passed: rss_gate_passed(slope_mib_per_hour, first_median_kib, final_median_kib),
    })
}

fn rss_window_medians(samples: &[RssSample]) -> Option<(u64, u64)> {
    let first = samples.get(..FIVE_MINUTE_SAMPLES)?;
    let final_start = samples.len().checked_sub(FIVE_MINUTE_SAMPLES)?;
    let final_samples = samples.get(final_start..)?;
    Some((rss_sample_median(first)?, rss_sample_median(final_samples)?))
}

fn rss_sample_median(samples: &[RssSample]) -> Option<u64> {
    median_kib(
        &samples
            .iter()
            .map(|sample| sample.rss_kib)
            .collect::<Vec<_>>(),
    )
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

fn soak_metadata(commit: &str, kernel: &str, rustc: &str) -> serde_json::Value {
    json!({
        "schema": "calc-flow.m4-soak-log.v1",
        "type": "calc_flow_stream_soak_metadata",
        "runtime_path": "ContinuousRunner/source/progress-driver/union/window/sink/supervisor/reaper",
        "commit": commit,
        "deterministic_seed": "sequential-two-source-window-v1",
        "command": SOAK_COMMAND,
        "environment": {
            "kernel": kernel,
            "rustc": rustc,
            "allocator": "system",
            "rss_source": "/proc/self/status:VmRSS",
        },
        "cadence_seconds": CADENCE.as_secs(),
        "target_duration_seconds": TARGET_DURATION.as_secs(),
        "sink_write_delay_millis": STANDARD_SOAK_SINK_DELAY.as_millis(),
        "sample_count": SAMPLE_COUNT,
        "warmup_samples": WARMUP_SAMPLES,
        "warmup_duration_seconds": (CADENCE
            * u32::try_from(WARMUP_SAMPLES).expect("the bounded soak warmup count fits u32"))
        .as_secs(),
        "boundedness_contract": {
            "message_slot_limit": "max_rows",
            "charged_rows_limit": "max_rows",
            "charged_bytes_limit": "max_bytes",
            "zero_cost_envelopes_consume_message_slots": true,
        },
        "progress_contract": {
            "source_count": 2,
            "watermark_policy": "source-provided",
            "per_binding_inbox_capacity": 64,
            "fence_selection": "all-visible",
            "completion_receipts": true,
            "full_execution_trace": true,
            "gate_cut_evidence": true,
            "settlement_latency_evidence": true,
        },
        "state_contract": {
            "window_size_micros": SOAK_WINDOW_MICROS,
            "checkpoint_batch_interval": SOAK_CHECKPOINT_BATCH_INTERVAL,
            "incremental_arrow_segments": true,
            "compaction_required": true,
            "terminal_live_keys": 0,
            "deterministic_final_aggregate": true,
        },
    })
}

fn assert_edge_budgets(status: &super::runner::ContinuousJobStatus) {
    for (edge, metrics) in &status.edges {
        let runtime = status
            .metrics
            .edges
            .get(edge)
            .expect("every soak channel has private runtime metrics");
        assert!(
            metrics.queue_depth <= runtime.message_slot_limit
                && metrics.high_water_depth <= runtime.message_slot_limit
                && metrics.charged_rows <= EDGE_BUDGET.max_rows
                && metrics.charged_bytes <= EDGE_BUDGET.max_bytes
                && metrics.high_water_rows <= EDGE_BUDGET.max_rows
                && metrics.high_water_bytes <= EDGE_BUDGET.max_bytes,
            "edge {edge:?} exceeded its budget: {metrics:?}"
        );
    }
}

fn soak_queue_sample(
    edge: &str,
    channel: &super::ChannelMetrics,
    runtime: &super::metrics::EdgeRuntimeMetrics,
) -> serde_json::Value {
    json!({
        "edge": edge,
        "message_slot_limit": runtime.message_slot_limit,
        "queue_depth": channel.queue_depth,
        "charged_rows": channel.charged_rows,
        "charged_bytes": channel.charged_bytes,
        "high_water_depth": channel.high_water_depth,
        "high_water_rows": channel.high_water_rows,
        "high_water_bytes": channel.high_water_bytes,
        "blocked_sends": channel.blocked_sends,
        "blocked_duration_micros": channel.blocked_duration.as_micros(),
    })
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

fn expected_window_rows(
    accepted: &BTreeMap<String, Vec<u64>>,
) -> BTreeMap<(i64, i64, String), i64> {
    let mut expected = BTreeMap::new();
    for (source, sequences) in accepted {
        for sequence in sequences {
            if sequence % 4 == 0 {
                continue;
            }
            let event_time = i64::try_from(*sequence)
                .expect("the twenty-minute soak cannot exhaust event-time space");
            let start = event_time.div_euclid(SOAK_WINDOW_MICROS) * SOAK_WINDOW_MICROS;
            let key = (
                start,
                start + SOAK_WINDOW_MICROS,
                format!("{source}-{:02}", sequence % 64),
            );
            expected
                .entry(key)
                .and_modify(|sum: &mut i64| {
                    *sum = sum
                        .checked_add(event_time)
                        .expect("the twenty-minute soak cannot overflow aggregate state");
                })
                .or_insert(event_time);
        }
    }
    expected
}

fn delivery_conservation_issue(
    accepted: &BTreeMap<String, Vec<u64>>,
    sink: &DeliveryState,
) -> Option<&'static str> {
    if sink.error.is_some() {
        Some("order or duplicate failure")
    } else if sink.rows != expected_window_rows(accepted) {
        Some("missing or incorrect final window state")
    } else {
        None
    }
}

fn zero_cost_batch_counts(accepted: &BTreeMap<String, Vec<u64>>) -> (usize, usize) {
    let total = accepted.values().map(Vec::len).sum();
    let zero_cost = accepted
        .values()
        .flatten()
        .filter(|sequence| **sequence % 4 == 0)
        .count();
    (zero_cost, total)
}

#[test]
fn independent_accepted_oracle_exposes_commit_after_downstream_drop() {
    let accepted = AcceptedSequenceRecorder::default();
    accepted.record_committed_for_test("left", 1);
    let sink_after_injected_drop = DeliveryState::default();

    assert_eq!(
        delivery_conservation_issue(&accepted.snapshot(), &sink_after_injected_drop),
        Some("missing or incorrect final window state")
    );
}

#[allow(
    clippy::too_many_arguments,
    reason = "the soak topology receives explicit lifecycle probes, delivery oracles, and delay"
)]
fn soak_spec(
    source_opened: &Arc<AtomicUsize>,
    source_closed: &Arc<AtomicUsize>,
    sink_opened: &Arc<AtomicUsize>,
    sink_closed: &Arc<AtomicUsize>,
    sink_a: &Arc<Mutex<DeliveryState>>,
    sink_b: &Arc<Mutex<DeliveryState>>,
    window_probe: &Arc<Mutex<WindowSoakProbe>>,
    accepted: &AcceptedSequenceRecorder,
    sink_write_delay: Duration,
) -> ContinuousJobSpec {
    let input_fields = soak_schema()
        .fields()
        .iter()
        .map(|field| field.as_ref().clone())
        .collect::<Vec<_>>();
    let union = UnionOperator::new(
        "merge",
        vec![
            Port::new("left", BatchKind::Table, true, Some(input_fields.clone())).unwrap(),
            Port::new("right", BatchKind::Table, true, Some(input_fields)).unwrap(),
        ],
    )
    .unwrap();
    let plan = PipelineBuilder::new("continuous-runtime-soak")
        .unwrap()
        .add_node("merge", Box::new(union))
        .unwrap()
        .add_node(
            "window",
            Box::new(SoakWindowOperator::new(Arc::clone(window_probe))) as Box<dyn StreamOperator>,
        )
        .unwrap()
        .connect(Edge::new(
            PortEndpoint::new("merge", "output").unwrap(),
            PortEndpoint::new("window", "input").unwrap(),
        ))
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
                    pending_watermark: None,
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
                write_delay: sink_write_delay,
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
    let window_probe = Arc::new(Mutex::new(WindowSoakProbe::default()));
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
            &window_probe,
            &accepted,
            STANDARD_SOAK_SINK_DELAY,
        ))
        .await
        .expect("real continuous runtime soak must launch");

    let commit = command_output("git", &["rev-parse", "HEAD"]);
    println!(
        "{}",
        soak_metadata(
            &commit,
            &command_output("uname", &["-sr"]),
            &command_output("rustc", &["--version"]),
        )
    );

    let initial_status = job.status();
    let steady_task_count = initial_status.tasks.len();
    assert!(steady_task_count > 0, "supervisor task registry is empty");
    assert_edge_budgets(&initial_status);
    let mut samples = Vec::with_capacity(SAMPLE_COUNT);
    let sampling_started = tokio::time::Instant::now();
    for index in 0..SAMPLE_COUNT {
        let elapsed_seconds = wait_for_sample_deadline(sampling_started, index).await;
        let process_status = tokio::fs::read_to_string("/proc/self/status")
            .await
            .expect("Linux soak could not read /proc/self/status");
        let rss_kib = parse_vm_rss_kib(&process_status).expect("Linux status omitted VmRSS");
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
                let runtime = status
                    .metrics
                    .edges
                    .get(edge)
                    .expect("every soak channel has private runtime metrics");
                soak_queue_sample(edge, metrics, runtime)
            })
            .collect::<Vec<_>>();
        let state = window_probe.lock().clone();
        println!(
            "{}",
            json!({
                "type": "calc_flow_stream_soak_sample",
                "index": index,
                "scheduled_elapsed_seconds": elapsed_at_sample(index),
                "elapsed_seconds": elapsed_seconds,
                "vmrss_kib": rss_kib,
                "task_count": steady_task_count,
                "queues": queues,
                "state": {
                    "checkpoints": state.checkpoints,
                    "compactions": state.compactions,
                    "total_segment_bytes": state.total_segment_bytes,
                    "max_new_segments": state.max_new_segments,
                    "max_retained_segments": state.max_retained_segments,
                    "live_keys": state.live_keys,
                    "max_live_keys": state.max_live_keys,
                },
            })
        );
    }
    assert_eq!(
        observed_timeline_issue(&samples),
        None,
        "soak sample timeline did not satisfy its machine-readable contract"
    );

    let outcome = job.shutdown().await;
    assert_eq!(outcome.state, ContinuousJobState::Completed);
    assert_eq!(outcome.cause, TerminalCause::GracefulShutdown);
    let terminal_status = job.status();
    let progress = terminal_status
        .progress
        .as_ref()
        .expect("M4 soak status omitted progress evidence");
    assert!(
        terminal_status.tasks.is_empty(),
        "supervised tasks did not converge"
    );
    assert!(
        terminal_status.edges.values().all(|metrics| {
            metrics.queue_depth == 0 && metrics.charged_rows == 0 && metrics.charged_bytes == 0
        }),
        "edge charges did not converge: {:?}",
        terminal_status.edges
    );
    assert_eq!(progress.current.unsettled_receipts, 0);
    assert!(progress.maximum_unsettled_receipts > 0);
    assert_eq!(progress.current.counters.immediate_rejections, 0);
    assert_eq!(progress.current.counters.transaction_error_settlements, 0);
    assert_eq!(progress.current.counters.post_end_tail_settlements, 0);
    assert_eq!(progress.current.counters.cancelled_settlements, 0);
    assert_eq!(progress.current.counters.fatal_settlements, 0);
    assert_eq!(progress.current.counters.driver_phase_failures, 0);
    assert_eq!(
        progress.current.counters.accepted_envelopes,
        progress.current.counters.settlement_attempts
    );
    assert_eq!(progress.current.terminal_gate_cuts.len(), 2);
    assert!(progress.current.counters.trace_records > 0);
    assert!(progress.current.counters.maximum_inbox_fences_per_drain > 0);
    assert!(progress.current.counters.maximum_selected_items_per_drain > 0);
    let gate_cuts = progress
        .current
        .terminal_gate_cuts
        .iter()
        .map(|(binding, close)| {
            (
                binding.as_str().to_owned(),
                json!({
                    "close_ordinal": close.close_ordinal.get(),
                    "cause": format!("{:?}", close.cause),
                    "old_generation": close.old_generation.get(),
                    "new_generation": close.new_generation.get(),
                    "closed_state": format!("{:?}", close.closed_state),
                    "next_inbox_sequence_cut": close.next_inbox_sequence_cut.get(),
                }),
            )
        })
        .collect::<BTreeMap<_, _>>();
    let saturated_edges = terminal_status
        .edges
        .iter()
        .filter(|(edge, channel)| {
            terminal_status.metrics.edges[*edge].message_slot_limit == channel.high_water_depth
        })
        .count();
    let blocked_edges = terminal_status
        .edges
        .values()
        .filter(|channel| channel.blocked_sends > 0)
        .count();
    assert!(
        saturated_edges > 0,
        "soak never saturated a message-slot limit"
    );
    assert!(
        blocked_edges > 0,
        "soak never observed producer backpressure"
    );
    let accepted = accepted.snapshot();
    let (zero_cost_batches, accepted_total) = zero_cost_batch_counts(&accepted);
    let expected_rows = expected_window_rows(&accepted);
    assert!(zero_cost_batches > 0, "soak accepted no zero-cost data");
    drop(job);
    runner.shutdown().await.unwrap();
    let registry_counts = runner.registry_counts();
    assert_eq!(registry_counts, (0, 0));
    assert_eq!(source_opened.load(Ordering::SeqCst), 2);
    assert_eq!(source_closed.load(Ordering::SeqCst), 2);
    assert_eq!(sink_opened.load(Ordering::SeqCst), 2);
    assert_eq!(sink_closed.load(Ordering::SeqCst), 2);

    let sink_a = sink_a.lock();
    let sink_b = sink_b.lock();
    assert_delivery_conservation(&accepted, &sink_a, "sink A");
    assert_delivery_conservation(&accepted, &sink_b, "sink B");
    let state = window_probe.lock().clone();
    assert!(state.checkpoints > 0, "soak captured no window state");
    assert!(state.compactions > 0, "soak exercised no state compaction");
    assert!(
        state.total_segment_bytes > 0,
        "soak produced no Arrow state segments"
    );
    assert!(state.max_live_keys > 0, "soak observed no live window keys");
    assert_eq!(state.terminal_live_keys, Some(0));
    assert!(
        state.max_retained_segments
            <= 32
                + usize::try_from(SOAK_CHECKPOINT_BATCH_INTERVAL)
                    .expect("the checkpoint interval fits usize"),
        "window state inventory grew beyond its checkpoint interval bound: {state:?}"
    );

    let gate = evaluate_rss_gate(&samples).expect("soak RSS sample set was incomplete");
    println!(
        "{}",
        json!({
            "schema": "calc-flow.m4-soak-log.v1",
            "type": "calc_flow_stream_soak_result",
            "commit": commit,
            "target_duration_seconds": TARGET_DURATION.as_secs(),
            "samples": samples.len(),
            "observed_timeline": {
                "validated": true,
                "first_elapsed_seconds": samples.first().map(|sample| sample.elapsed_seconds),
                "last_elapsed_seconds": samples.last().map(|sample| sample.elapsed_seconds),
            },
            "slope_mib_per_hour": gate.slope_mib_per_hour,
            "first_post_warmup_five_minute_median_kib": gate.first_median_kib,
            "final_five_minute_median_kib": gate.final_median_kib,
            "passed": gate.passed,
            "progress": {
                "deterministic_seed": "sequential-two-source-window-v1",
                "admission_attempts": progress.current.counters.admission_attempts,
                "accepted_envelopes": progress.current.counters.accepted_envelopes,
                "immediate_rejections": progress.current.counters.immediate_rejections,
                "drain_epochs": progress.current.counters.drain_epochs,
                "inbox_fences": progress.current.counters.inbox_fences,
                "due_timers": progress.current.counters.due_timers,
                "terminal_transitions": progress.current.counters.terminal_transitions,
                "gate_transitions": progress.current.counters.gate_transitions,
                "settlements": {
                    "total": progress.current.counters.settlement_attempts,
                    "commit_success": progress.current.counters.commit_success_settlements,
                    "transaction_error": progress
                        .current
                        .counters
                        .transaction_error_settlements,
                    "post_end_tail": progress.current.counters.post_end_tail_settlements,
                    "cancelled": progress.current.counters.cancelled_settlements,
                    "fatal": progress.current.counters.fatal_settlements,
                },
                "maximum_unsettled_receipts": progress.maximum_unsettled_receipts,
                "terminal_unsettled_receipts": progress.current.unsettled_receipts,
                "progress_emissions": progress.current.counters.progress_emissions,
                "terminal_timer_entries": progress.current.counters.timer_entries,
                "maximum_timer_entries": progress.maximum_timer_entries,
                "trace_records": progress.current.counters.trace_records,
                "maximum_trace_records": progress.maximum_trace_records,
                "maximum_inbox_fences_per_drain": progress
                    .current
                    .counters
                    .maximum_inbox_fences_per_drain,
                "maximum_selected_items_per_drain": progress
                    .current
                    .counters
                    .maximum_selected_items_per_drain,
                "maximum_due_timers_per_drain": progress
                    .current
                    .counters
                    .maximum_due_timers_per_drain,
                "maximum_settlement_latency_micros": progress
                    .maximum_settlement_latency_micros,
                "terminal_gate_cuts": gate_cuts,
                "driver_phase_failures": progress.current.counters.driver_phase_failures,
                "driver_phase": format!("{:?}", progress.current.phase),
            },
            "boundedness": {
                "all_edges_within_limits": true,
                "saturated_edges": saturated_edges,
                "blocked_edges": blocked_edges,
                "zero_cost_backpressure_exercised": zero_cost_batches > 0
                    && saturated_edges > 0
                    && blocked_edges > 0,
            },
            "conservation": {
                "accepted_batches": accepted_total,
                "accepted_zero_cost_batches": zero_cost_batches,
                "sink_a_batches": sink_a.total,
                "sink_b_batches": sink_b.total,
                "expected_window_rows": expected_rows.len(),
                "sink_a_window_rows": sink_a.rows.len(),
                "sink_b_window_rows": sink_b.rows.len(),
                "missing": 0,
                "duplicate": 0,
            },
            "state": {
                "checkpoints": state.checkpoints,
                "compactions": state.compactions,
                "total_segment_bytes": state.total_segment_bytes,
                "max_new_segments": state.max_new_segments,
                "max_retained_segments": state.max_retained_segments,
                "max_live_keys": state.max_live_keys,
                "terminal_live_keys": state.terminal_live_keys,
            },
            "convergence": {
                "steady_task_count": steady_task_count,
                "terminal_task_count": terminal_status.tasks.len(),
                "terminal_queue_depth": terminal_status
                    .edges
                    .values()
                    .map(|metrics| metrics.queue_depth)
                    .sum::<usize>(),
                "terminal_charged_rows": terminal_status
                    .edges
                    .values()
                    .map(|metrics| metrics.charged_rows)
                    .sum::<usize>(),
                "terminal_charged_bytes": terminal_status
                    .edges
                    .values()
                    .map(|metrics| metrics.charged_bytes)
                    .sum::<usize>(),
                "runner_live_jobs": registry_counts.0,
                "runner_reaper_jobs": registry_counts.1,
            },
            "lifecycle": {
                "source_opened": source_opened.load(Ordering::SeqCst),
                "source_closed": source_closed.load(Ordering::SeqCst),
                "sink_opened": sink_opened.load(Ordering::SeqCst),
                "sink_closed": sink_closed.load(Ordering::SeqCst),
            },
            "outcome": {
                "state": format!("{:?}", outcome.state),
                "cause": format!("{:?}", outcome.cause),
                "primary_error": null,
                "resource_errors": [],
            },
        })
    );
    assert!(gate.passed, "RSS guard failed: {gate:?}");
}

#[tokio::test]
#[ignore = "twenty-minute opt-in streaming soak; set CALC_FLOW_STREAM_SOAK=1"]
async fn twenty_minute_two_source_slow_sink() {
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
    let window_probe = Arc::new(Mutex::new(WindowSoakProbe::default()));
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
            &window_probe,
            &accepted,
            SMOKE_SINK_DELAY,
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
    let window_probe = Arc::new(Mutex::new(WindowSoakProbe::default()));
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
            &window_probe,
            &accepted,
            SMOKE_SINK_DELAY,
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
    let probe = window_probe.lock().clone();
    assert!(probe.checkpoints > 0);
    assert!(probe.compactions > 0);
    assert!(probe.total_segment_bytes > 0);
    assert!(probe.max_live_keys > 0);
    assert_eq!(probe.terminal_live_keys, Some(0));
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

#[test]
fn rss_gate_rejects_incomplete_or_extra_sample_sets() {
    let samples = (0..=SAMPLE_COUNT)
        .map(|index| RssSample {
            elapsed_seconds: elapsed_at_sample(index),
            rss_kib: 100_000,
        })
        .collect::<Vec<_>>();

    assert_eq!(evaluate_rss_gate(&samples[..SAMPLE_COUNT - 1]), None);
    assert_eq!(evaluate_rss_gate(&samples), None);
}

#[test]
fn twenty_minute_soak_contract_has_exact_cadence_and_sample_windows() {
    assert_eq!(CADENCE, Duration::from_secs(10));
    assert_eq!(SAMPLE_COUNT, 120);
    assert_eq!(WARMUP_SAMPLES, 30);
    assert!((elapsed_at_sample(SAMPLE_COUNT - 1) - 1_200.0).abs() < f64::EPSILON);
}

#[tokio::test(start_paused = true)]
async fn absolute_soak_deadlines_do_not_accumulate_sampling_work_delay() {
    let started = tokio::time::Instant::now();
    let mut samples = Vec::with_capacity(SAMPLE_COUNT);

    for index in 0..SAMPLE_COUNT {
        let elapsed_seconds = wait_for_sample_deadline(started, index).await;
        samples.push(RssSample {
            elapsed_seconds,
            rss_kib: 100_000,
        });
        tokio::time::advance(Duration::from_secs(1)).await;
    }

    assert_eq!(samples.len(), SAMPLE_COUNT);
    assert_eq!(observed_timeline_issue(&samples), None);
    assert!((samples.last().unwrap().elapsed_seconds - 1_200.0).abs() < f64::EPSILON);
}

#[test]
fn soak_source_uses_a_zero_cost_batch_for_every_fourth_sequence() {
    for sequence in 0..8 {
        let batch = soak_batch("left", sequence).unwrap();
        if sequence % 4 == 0 {
            assert_eq!(batch.num_rows(), 0);
            assert_eq!(batch.estimated_bytes().unwrap(), 0);
        } else {
            assert_eq!(batch.num_rows(), 1);
            assert!(batch.estimated_bytes().unwrap() > 0);
        }
    }
}

#[test]
fn soak_metadata_is_machine_readable_and_declares_the_slot_contract() {
    let metadata = soak_metadata("abc123", "Linux 1", "rustc 1.88");

    assert_eq!(metadata["schema"], "calc-flow.m4-soak-log.v1");
    assert_eq!(metadata["commit"], "abc123");
    assert_eq!(metadata["target_duration_seconds"], 1_200);
    assert_eq!(metadata["sample_count"], 120);
    assert_eq!(metadata["sink_write_delay_millis"], 500);
    assert_eq!(metadata["warmup_duration_seconds"], 300);
    assert_eq!(
        metadata["deterministic_seed"],
        "sequential-two-source-window-v1"
    );
    assert_eq!(metadata["environment"]["kernel"], "Linux 1");
    assert_eq!(metadata["environment"]["rustc"], "rustc 1.88");
    assert_eq!(
        metadata["boundedness_contract"]["message_slot_limit"],
        "max_rows"
    );
    assert_eq!(metadata["progress_contract"]["source_count"], 2);
    assert_eq!(metadata["state_contract"]["window_size_micros"], 64);
    assert_eq!(metadata["state_contract"]["checkpoint_batch_interval"], 8);
    assert_eq!(metadata["state_contract"]["terminal_live_keys"], 0);
    assert_eq!(
        metadata["progress_contract"]["fence_selection"],
        "all-visible"
    );
    assert_eq!(
        metadata["command"],
        "CALC_FLOW_STREAM_SOAK=1 cargo test -p calc-flow --lib runtime::streaming::soak::twenty_minute_two_source_slow_sink -- --ignored --exact --nocapture"
    );
}

#[test]
fn soak_queue_sample_reports_slots_payload_and_backpressure() {
    let channel = super::ChannelMetrics {
        queue_depth: 7,
        charged_rows: 3,
        charged_bytes: 24,
        high_water_depth: 8,
        high_water_rows: 4,
        high_water_bytes: 32,
        blocked_sends: 5,
        blocked_duration: Duration::from_micros(9),
    };
    let runtime = super::metrics::EdgeRuntimeMetrics {
        message_slot_limit: 8,
        channel: channel.clone(),
        ..super::metrics::EdgeRuntimeMetrics::default()
    };

    let sample = soak_queue_sample("left->merge", &channel, &runtime);

    assert_eq!(sample["message_slot_limit"], 8);
    assert_eq!(sample["queue_depth"], 7);
    assert_eq!(sample["charged_rows"], 3);
    assert_eq!(sample["charged_bytes"], 24);
    assert_eq!(sample["high_water_depth"], 8);
    assert_eq!(sample["blocked_sends"], 5);
    assert_eq!(sample["blocked_duration_micros"], 9);
}

#[test]
fn soak_zero_cost_count_is_derived_from_the_accepted_oracle() {
    let accepted = BTreeMap::from([
        ("left".to_owned(), vec![0, 1, 2, 3, 4]),
        ("right".to_owned(), vec![0, 1, 2, 3]),
    ]);

    assert_eq!(zero_cost_batch_counts(&accepted), (3, 9));
}
