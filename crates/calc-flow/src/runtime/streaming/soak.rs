use std::{
    collections::{BTreeMap, BTreeSet},
    fs::{File, OpenOptions},
    hint::black_box,
    io::Write as _,
    path::{Path, PathBuf},
    process::Command,
    sync::{
        Arc,
        atomic::{AtomicUsize, Ordering},
    },
    time::Duration,
};

use async_trait::async_trait;
use chrono::{TimeZone, Utc};
use criterion::Criterion;
use datafusion::arrow::{
    array::{Int64Array, StringArray, TimestampMicrosecondArray},
    datatypes::{DataType, Field, Schema, TimeUnit},
    record_batch::RecordBatch,
};
use parking_lot::Mutex;
use serde::{Deserialize, Serialize};
use serde_json::json;
use sha2::{Digest, Sha256};

use super::{
    StreamJobContext,
    job::{
        ContinuousJobSpec, M2DeliveryMode, NamedSinkBinding, NamedSourceBinding,
        OrdinarySinkBinding, OrdinaryStreamSink, TransactionalStreamSink,
    },
    progress::{
        BindingIdentity, DeclaredSchema, DurableSourceCut, LiveProgressCoordinator,
        NativeWatermarkCapability, ReplayPositioningCapability, SourceBindingSpec,
        SourceDescriptor, StreamProgressRuntimeConfig, WatermarkPolicy, prepare_stream_job,
    },
    runner::{
        CheckpointFaultMode, CheckpointFaultPoint, CheckpointRuntimeSpec, ContinuousJob,
        ContinuousJobState, ContinuousRunner, TerminalCause,
    },
    source_task::{
        AcceptedSequenceRecorder, Cursor, SourceBinding, SourceCapabilities, SourceEvent,
        StreamSource,
    },
};
use crate::{
    AggregateFunction, Batch, BatchKind, BatchMetadata, CalcFlowError, CancellationToken,
    CheckpointManifest as BenchmarkCheckpointManifest, CheckpointManifestFields,
    CursorManifestEntry, Edge, EdgeBudget, Epoch, EventTime, JsonMap, LocalStateBackend,
    ManifestIngressState, OperatorManifestEntry, OperatorMetadata, OperatorStateSnapshot,
    PipelineBuilder, Port, PortEndpoint, RecoveryStatus, Result, SourceWatermarkManifestState,
    StateBackend, StateLineageBackend, StateLineageKey, StreamCollector, StreamOperator,
    StreamOperatorContext, StreamRequirements, StreamRuntimeConfig, UdfRegistry, UnionOperator,
    WindowAggregateOperator, WindowSpec,
    state::{ManifestTransaction, PreparedEpochManifest, PreparedManifestIdentity},
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
// Final-only output must stay frequent enough for the slow sink to govern
// source admission; otherwise the lossless M3 trace can outrun backpressure.
const SOAK_WINDOW_MICROS: i64 = 1;
const SOAK_CHECKPOINT_BATCH_INTERVAL: u64 = 8;
const CHECKPOINT_SOAK_CADENCE: Duration = Duration::from_secs(10);
const CHECKPOINT_SOAK_SAMPLE_COUNT: usize = 120;
const CHECKPOINT_SOAK_RESTART_SAMPLES: [usize; 2] = [39, 79];
const CHECKPOINT_RESTART_WARMUP_SAMPLES: usize = 10;
const CHECKPOINT_FIRST_STEADY_START: usize =
    CHECKPOINT_SOAK_RESTART_SAMPLES[0] + 1 + CHECKPOINT_RESTART_WARMUP_SAMPLES;
const CHECKPOINT_FIRST_STEADY_END: usize = CHECKPOINT_SOAK_RESTART_SAMPLES[1] + 1;
const CHECKPOINT_FINAL_STEADY_START: usize =
    CHECKPOINT_FIRST_STEADY_END + CHECKPOINT_RESTART_WARMUP_SAMPLES;
const CHECKPOINT_RSS_COMPARISON_RANGES: [[usize; 2]; 2] = [
    [
        CHECKPOINT_FIRST_STEADY_START,
        CHECKPOINT_FIRST_STEADY_END - 1,
    ],
    [
        CHECKPOINT_FINAL_STEADY_START,
        CHECKPOINT_SOAK_SAMPLE_COUNT - 1,
    ],
];
const MAX_CHECKPOINT_SOAK_STATE_BYTES: u64 = 64 * 1_024 * 1_024;
const CHECKPOINT_SOAK_COMMAND: &str = "CALC_FLOW_M5_CHECKPOINT_SOAK=1 cargo test -p calc-flow --lib runtime::streaming::soak::twenty_minute_epoch_checkpoint_restart -- --ignored --exact --nocapture";
const CHECKPOINT_BENCHMARK_COMMAND: &str = "CARGO_TARGET_DIR=<fresh-candidate-target> CALC_FLOW_M5_CHECKPOINT_BENCHMARK=1 CALC_FLOW_M5_CHECKPOINT_BENCHMARK_RUN_ID=<unique-run-id> CALC_FLOW_M5_PRIVATE_SOURCE_COMMIT=<candidate-commit> CALC_FLOW_M5_PRIVATE_SOURCE_TREE=<candidate-tree> cargo test -p calc-flow --lib runtime::streaming::soak::private_m5_epoch_checkpoint_absolute_benchmark -- --ignored --exact --nocapture";
const M5_PRIVATE_BENCHMARK_CASES: [&str; 7] = [
    "m5/private_path/barrier_cut_fan_out",
    "m5/private_path/two_input_alignment",
    "m5/private_path/dirty_window_state_stage",
    "m5/private_path/production_manifest_publication",
    "m5/private_path/cold_restore",
    "m5/private_path/transactional_sink_commit",
    "m5/private_full_path/periodic_checkpoint_restart",
];
const CHECKPOINT_BENCHMARK_SAMPLE_SIZE: usize = 10;
const CHECKPOINT_BENCHMARK_WARM_UP: Duration = Duration::from_secs(1);
const CHECKPOINT_BENCHMARK_MEASUREMENT: Duration = Duration::from_secs(3);
const PRIVATE_BENCHMARK_SOURCE_COMMIT: Option<&str> =
    option_env!("CALC_FLOW_M5_PRIVATE_SOURCE_COMMIT");
const PRIVATE_BENCHMARK_SOURCE_TREE: Option<&str> = option_env!("CALC_FLOW_M5_PRIVATE_SOURCE_TREE");

type SourceOpenHistory = Arc<Mutex<Vec<(String, Option<Vec<u8>>)>>>;

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
        let retained_segments = snapshot_inventory_len(&snapshot)?;
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

fn snapshot_inventory_len(snapshot: &OperatorStateSnapshot) -> Result<usize> {
    snapshot
        .inline_metadata
        .get("segment_inventory")
        .and_then(serde_json::Value::as_array)
        .map(Vec::len)
        .ok_or_else(|| CalcFlowError::Internal {
            message: "window snapshot omitted its segment inventory".into(),
        })
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
        let row_count = batch
            .table_payload()
            .expect("the soak window emits table batches")
            .batches()
            .iter()
            .map(RecordBatch::num_rows)
            .sum::<usize>();
        if row_count == 0 && self.error.is_none() {
            self.error = Some("window output emitted an empty batch".into());
            return;
        }
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

fn evaluate_checkpoint_restart_rss_gate(samples: &[RssSample]) -> Option<RssGate> {
    if samples.len() != CHECKPOINT_SOAK_SAMPLE_COUNT {
        return None;
    }
    let first = samples.get(CHECKPOINT_FIRST_STEADY_START..CHECKPOINT_FIRST_STEADY_END)?;
    let final_samples = samples.get(CHECKPOINT_FINAL_STEADY_START..)?;
    if first.len() != FIVE_MINUTE_SAMPLES || final_samples.len() != FIVE_MINUTE_SAMPLES {
        return None;
    }
    let slope_mib_per_hour = least_squares_mib_per_hour(final_samples)?;
    let first_median_kib = rss_sample_median(first)?;
    let final_median_kib = rss_sample_median(final_samples)?;
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

#[test]
fn delivery_oracle_rejects_an_empty_window_output_batch() {
    let mut deliveries = DeliveryState::default();
    deliveries.observe(&soak_batch("window", 0).unwrap());

    assert_eq!(
        deliveries.error.as_deref(),
        Some("window output emitted an empty batch")
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

struct CheckpointMatrixSource {
    source_id: &'static str,
    next_sequence: u64,
    pending_watermark: bool,
    ended: bool,
    poll_delay: Duration,
    opened_with: Arc<Mutex<Vec<Option<Vec<u8>>>>>,
    closed: Arc<AtomicUsize>,
}

fn checkpoint_matrix_batch(source_id: &str) -> Result<Batch> {
    let value = if source_id == "left" { 1 } else { 10 };
    let record = RecordBatch::try_new(
        soak_schema(),
        vec![
            Arc::new(TimestampMicrosecondArray::from(vec![0]).with_timezone("UTC")),
            Arc::new(StringArray::from(vec![source_id])),
            Arc::new(Int64Array::from(vec![value])),
        ],
    )
    .map_err(|error| CalcFlowError::Internal {
        message: format!("checkpoint matrix batch construction failed: {error}"),
    })?;
    Batch::table(
        vec![record],
        BatchMetadata::new(source_id, 0, BTreeMap::new())?,
    )
}

fn checkpoint_matrix_cursor_order(source_id: &str) -> Vec<u8> {
    match source_id {
        "left" => 41_u64,
        "right" => 73_u64,
        _ => panic!("unexpected checkpoint matrix source {source_id:?}"),
    }
    .to_be_bytes()
    .to_vec()
}

#[async_trait]
impl StreamSource for CheckpointMatrixSource {
    async fn open(&mut self, cursor: Option<Cursor>) -> Result<()> {
        self.opened_with
            .lock()
            .push(cursor.as_ref().map(|cursor| cursor.order().to_vec()));
        if let Some(cursor) = cursor {
            let bytes: [u8; 8] =
                cursor
                    .order()
                    .try_into()
                    .map_err(|_| CalcFlowError::CheckpointMismatch {
                        message: "checkpoint matrix cursor order is not a u64".into(),
                    })?;
            self.next_sequence = u64::from_be_bytes(bytes).checked_add(1).ok_or_else(|| {
                CalcFlowError::CheckpointMismatch {
                    message: "checkpoint matrix cursor exhausted u64".into(),
                }
            })?;
        }
        Ok(())
    }

    async fn next(&mut self) -> Result<Option<SourceEvent>> {
        if self.next_sequence != 0 && !self.poll_delay.is_zero() {
            tokio::time::sleep(self.poll_delay).await;
        }
        if self.pending_watermark {
            self.pending_watermark = false;
            return Ok(Some(SourceEvent::Watermark(EventTime::from_micros(1))));
        }
        if self.next_sequence == 0 {
            self.next_sequence = 1;
            self.pending_watermark = true;
            return Ok(Some(SourceEvent::Data {
                batch: checkpoint_matrix_batch(self.source_id)?,
                cursor: Cursor::new(
                    checkpoint_matrix_cursor_order(self.source_id),
                    JsonMap::new(),
                )?,
            }));
        }
        self.ended = true;
        Ok(None)
    }

    async fn close(&mut self) -> Result<()> {
        self.closed.fetch_add(1, Ordering::SeqCst);
        Ok(())
    }

    fn capabilities(&self) -> SourceCapabilities {
        SourceCapabilities {
            replayable: true,
            max_batch_rows: 1,
            max_batch_bytes: 1 << 20,
        }
    }
}

struct CheckpointRestartSoakSource {
    source_id: &'static str,
    next_sequence: u64,
    pending_watermark: Option<EventTime>,
    poll_delay: Duration,
    stop: Arc<AtomicUsize>,
    opened_with: SourceOpenHistory,
    closed: Arc<AtomicUsize>,
}

fn checkpoint_restart_soak_batch(source_id: &str, sequence: u64) -> Result<Batch> {
    let event_time = i64::try_from(sequence).map_err(|_| CalcFlowError::Internal {
        message: "checkpoint soak event time exhausted i64".into(),
    })?;
    let value = event_time
        .checked_add(1)
        .ok_or_else(|| CalcFlowError::Internal {
            message: "checkpoint soak value exhausted i64".into(),
        })?;
    let record = RecordBatch::try_new(
        soak_schema(),
        vec![
            Arc::new(TimestampMicrosecondArray::from(vec![event_time]).with_timezone("UTC")),
            Arc::new(StringArray::from(vec![source_id])),
            Arc::new(Int64Array::from(vec![value])),
        ],
    )
    .map_err(|error| CalcFlowError::Internal {
        message: format!("checkpoint soak batch construction failed: {error}"),
    })?;
    Batch::table(
        vec![record],
        BatchMetadata::new(source_id, sequence, BTreeMap::new())?,
    )
}

#[async_trait]
impl StreamSource for CheckpointRestartSoakSource {
    async fn open(&mut self, cursor: Option<Cursor>) -> Result<()> {
        self.opened_with.lock().push((
            self.source_id.into(),
            cursor.as_ref().map(|cursor| cursor.order().to_vec()),
        ));
        if let Some(cursor) = cursor {
            let bytes: [u8; 8] =
                cursor
                    .order()
                    .try_into()
                    .map_err(|_| CalcFlowError::CheckpointMismatch {
                        message: "checkpoint soak cursor order is not a u64".into(),
                    })?;
            self.next_sequence = u64::from_be_bytes(bytes).checked_add(1).ok_or_else(|| {
                CalcFlowError::CheckpointMismatch {
                    message: "checkpoint soak cursor exhausted u64".into(),
                }
            })?;
        }
        Ok(())
    }

    async fn next(&mut self) -> Result<Option<SourceEvent>> {
        if let Some(watermark) = self.pending_watermark.take() {
            return Ok(Some(SourceEvent::Watermark(watermark)));
        }
        if self.stop.load(Ordering::SeqCst) != 0 {
            return Ok(None);
        }
        tokio::time::sleep(self.poll_delay).await;
        let sequence = self.next_sequence;
        self.next_sequence = sequence
            .checked_add(1)
            .ok_or_else(|| CalcFlowError::Internal {
                message: "checkpoint soak source exhausted u64".into(),
            })?;
        let watermark_micros =
            i64::try_from(self.next_sequence).map_err(|_| CalcFlowError::Internal {
                message: "checkpoint soak watermark exhausted i64".into(),
            })?;
        self.pending_watermark = Some(EventTime::from_micros(watermark_micros));
        Ok(Some(SourceEvent::Data {
            batch: checkpoint_restart_soak_batch(self.source_id, sequence)?,
            cursor: Cursor::new(sequence.to_be_bytes().to_vec(), JsonMap::new())?,
        }))
    }

    async fn close(&mut self) -> Result<()> {
        self.closed.fetch_add(1, Ordering::SeqCst);
        Ok(())
    }

    fn capabilities(&self) -> SourceCapabilities {
        SourceCapabilities {
            replayable: true,
            max_batch_rows: 1,
            max_batch_bytes: 1 << 20,
        }
    }
}

struct CheckpointMatrixForward {
    inputs: [Port; 1],
    outputs: [Port; 1],
}

impl CheckpointMatrixForward {
    fn new(fields: Vec<Field>) -> Self {
        Self {
            inputs: [Port::new("input", BatchKind::Table, true, Some(fields.clone())).unwrap()],
            outputs: [Port::new("output", BatchKind::Table, true, Some(fields)).unwrap()],
        }
    }
}

impl OperatorMetadata for CheckpointMatrixForward {
    fn name(&self) -> &'static str {
        "checkpoint-matrix-forward"
    }

    fn input_ports(&self) -> &[Port] {
        &self.inputs
    }

    fn output_ports(&self) -> &[Port] {
        &self.outputs
    }

    fn configuration(&self) -> JsonMap {
        JsonMap::new()
    }
}

#[async_trait]
impl StreamOperator for CheckpointMatrixForward {
    async fn process_data(
        &mut self,
        _ingress: &str,
        batch: Batch,
        _context: &StreamOperatorContext<'_>,
        output: &mut dyn StreamCollector,
    ) -> Result<()> {
        output.emit("output", batch).await
    }

    async fn on_watermark(
        &mut self,
        _watermark: EventTime,
        _context: &StreamOperatorContext<'_>,
        _output: &mut dyn StreamCollector,
    ) -> Result<()> {
        Ok(())
    }

    async fn on_end(
        &mut self,
        _context: &StreamOperatorContext<'_>,
        _output: &mut dyn StreamCollector,
    ) -> Result<()> {
        Ok(())
    }
}

struct CheckpointMatrixSink {
    sink_id: &'static str,
    root: PathBuf,
    epoch: Option<Epoch>,
    pending: Vec<String>,
    written_records: Arc<AtomicUsize>,
    closed: Arc<AtomicUsize>,
    write_delay: Duration,
}

impl CheckpointMatrixSink {
    fn prepared_path(&self, epoch: Epoch) -> PathBuf {
        self.root
            .join(format!("prepared-{:020}.json", epoch.as_u64()))
    }

    fn visible_path(&self) -> PathBuf {
        self.root.join("visible.json")
    }

    async fn read_visible_epochs(&self) -> Result<BTreeMap<String, Vec<String>>> {
        match tokio::fs::read(self.visible_path()).await {
            Ok(bytes) => serde_json::from_slice(&bytes).map_err(|error| CalcFlowError::Internal {
                message: format!("checkpoint matrix visible state is invalid: {error}"),
            }),
            Err(error) if error.kind() == std::io::ErrorKind::NotFound => Ok(BTreeMap::new()),
            Err(error) => Err(CalcFlowError::Internal {
                message: format!("checkpoint matrix visible state read failed: {error}"),
            }),
        }
    }

    async fn prepared_records(&self, epoch: Epoch) -> Result<Vec<String>> {
        let bytes = tokio::fs::read(self.prepared_path(epoch))
            .await
            .map_err(|error| CalcFlowError::CheckpointMismatch {
                message: format!(
                    "sink {:?} original prepared artifact for epoch {} is unavailable: {error}",
                    self.sink_id,
                    epoch.as_u64()
                ),
            })?;
        serde_json::from_slice::<JsonMap>(&bytes)
            .map_err(|error| CalcFlowError::CheckpointMismatch {
                message: format!(
                    "sink {:?} prepared artifact is invalid: {error}",
                    self.sink_id
                ),
            })?
            .get("records")
            .and_then(serde_json::Value::as_array)
            .ok_or_else(|| CalcFlowError::CheckpointMismatch {
                message: format!(
                    "sink {:?} prepared artifact records are missing",
                    self.sink_id
                ),
            })?
            .iter()
            .map(|value| {
                value
                    .as_str()
                    .map(str::to_owned)
                    .ok_or_else(|| CalcFlowError::CheckpointMismatch {
                        message: format!(
                            "sink {:?} pre-commit record is not a string",
                            self.sink_id
                        ),
                    })
            })
            .collect()
    }

    async fn commit_prepared(&self, epoch: Epoch, state: &JsonMap) -> Result<()> {
        if state
            .get("artifact_epoch")
            .and_then(serde_json::Value::as_u64)
            != Some(epoch.as_u64())
        {
            return Err(CalcFlowError::CheckpointMismatch {
                message: format!("sink {:?} prepared artifact identity changed", self.sink_id),
            });
        }
        let mut visible = self.read_visible_epochs().await?;
        let epoch_key = epoch.as_u64().to_string();
        if let std::collections::btree_map::Entry::Vacant(entry) = visible.entry(epoch_key) {
            let records = self.prepared_records(epoch).await?;
            entry.insert(records);
            let bytes = serde_json::to_vec(&visible).map_err(|error| CalcFlowError::Internal {
                message: format!("checkpoint matrix visible state encode failed: {error}"),
            })?;
            let temporary = self.root.join(".visible.tmp");
            tokio::fs::write(&temporary, bytes)
                .await
                .map_err(|error| CalcFlowError::Internal {
                    message: format!("checkpoint matrix visible state write failed: {error}"),
                })?;
            tokio::fs::rename(&temporary, self.visible_path())
                .await
                .map_err(|error| CalcFlowError::Internal {
                    message: format!("checkpoint matrix visible state rename failed: {error}"),
                })?;
        }
        match tokio::fs::remove_file(self.prepared_path(epoch)).await {
            Ok(()) => {}
            Err(error) if error.kind() == std::io::ErrorKind::NotFound => {}
            Err(error) => {
                return Err(CalcFlowError::Internal {
                    message: format!("checkpoint matrix prepared cleanup failed: {error}"),
                });
            }
        }
        Ok(())
    }
}

#[async_trait]
impl TransactionalStreamSink for CheckpointMatrixSink {
    async fn open(&mut self) -> Result<()> {
        tokio::fs::create_dir_all(&self.root)
            .await
            .map_err(|error| CalcFlowError::Internal {
                message: format!("checkpoint matrix sink open failed: {error}"),
            })
    }

    async fn begin_epoch(&mut self, epoch: Epoch) -> Result<()> {
        self.epoch = Some(epoch);
        self.pending.clear();
        Ok(())
    }

    async fn write(&mut self, batch: &Batch) -> Result<()> {
        if !self.write_delay.is_zero() {
            tokio::time::sleep(self.write_delay).await;
        }
        for record in batch
            .table_payload()
            .map_err(|error| CalcFlowError::CheckpointMismatch {
                message: format!("checkpoint matrix sink received a non-table batch: {error}"),
            })?
            .batches()
        {
            let starts = record
                .column_by_name("window_start")
                .and_then(|array| array.as_any().downcast_ref::<TimestampMicrosecondArray>())
                .ok_or_else(|| CalcFlowError::CheckpointMismatch {
                    message: "checkpoint matrix output omitted window_start".into(),
                })?;
            let ends = record
                .column_by_name("window_end")
                .and_then(|array| array.as_any().downcast_ref::<TimestampMicrosecondArray>())
                .ok_or_else(|| CalcFlowError::CheckpointMismatch {
                    message: "checkpoint matrix output omitted window_end".into(),
                })?;
            let groups = record
                .column_by_name("group")
                .and_then(|array| array.as_any().downcast_ref::<StringArray>())
                .ok_or_else(|| CalcFlowError::CheckpointMismatch {
                    message: "checkpoint matrix output omitted group".into(),
                })?;
            let sums = record
                .column_by_name("sum_value")
                .and_then(|array| array.as_any().downcast_ref::<Int64Array>())
                .ok_or_else(|| CalcFlowError::CheckpointMismatch {
                    message: "checkpoint matrix output omitted sum_value".into(),
                })?;
            for row in 0..record.num_rows() {
                self.pending.push(format!(
                    "{}|{}|{}|{}|{}",
                    self.sink_id,
                    starts.value(row),
                    ends.value(row),
                    groups.value(row),
                    sums.value(row)
                ));
            }
            self.written_records
                .fetch_add(record.num_rows(), Ordering::SeqCst);
        }
        Ok(())
    }

    async fn pre_commit(&mut self, epoch: Epoch) -> Result<JsonMap> {
        if self.epoch != Some(epoch) {
            return Err(CalcFlowError::CheckpointMismatch {
                message: format!("sink {:?} pre-committed the wrong epoch", self.sink_id),
            });
        }
        let state: JsonMap = BTreeMap::from([("records".into(), json!(self.pending))]);
        let bytes = serde_json::to_vec(&state).map_err(|error| CalcFlowError::Internal {
            message: format!("checkpoint matrix prepared state encode failed: {error}"),
        })?;
        tokio::fs::write(self.prepared_path(epoch), bytes)
            .await
            .map_err(|error| CalcFlowError::Internal {
                message: format!("checkpoint matrix prepared state write failed: {error}"),
            })?;
        Ok(BTreeMap::from([(
            "artifact_epoch".into(),
            json!(epoch.as_u64()),
        )]))
    }

    async fn commit(&mut self, epoch: Epoch, state: &JsonMap) -> Result<()> {
        self.commit_prepared(epoch, state).await
    }

    async fn abort(&mut self, epoch: Epoch, _state: Option<&JsonMap>) -> Result<()> {
        match tokio::fs::remove_file(self.prepared_path(epoch)).await {
            Ok(()) => {}
            Err(error) if error.kind() == std::io::ErrorKind::NotFound => {}
            Err(error) => {
                return Err(CalcFlowError::Internal {
                    message: format!("checkpoint matrix abort failed: {error}"),
                });
            }
        }
        self.pending.clear();
        self.epoch = None;
        Ok(())
    }

    async fn recover(&mut self, manifest: &crate::CheckpointManifest) -> Result<()> {
        let state = manifest
            .sinks()
            .get(self.sink_id)
            .and_then(|entry| entry.pre_commit.clone())
            .ok_or_else(|| CalcFlowError::CheckpointMismatch {
                message: format!("sink {:?} recovery state is missing", self.sink_id),
            })?;
        self.commit_prepared(manifest.epoch(), &state).await
    }

    async fn close(&mut self) -> Result<()> {
        self.closed.fetch_add(1, Ordering::SeqCst);
        Ok(())
    }
}

fn checkpoint_matrix_window() -> WindowAggregateOperator {
    let spec = WindowSpec::tumbling("event_time", Duration::from_micros(1))
        .unwrap()
        .group_by(["group"])
        .unwrap()
        .aggregate(AggregateFunction::Sum, "value", "sum_value")
        .unwrap();
    WindowAggregateOperator::new("window", soak_schema(), spec).unwrap()
}

#[allow(
    clippy::too_many_arguments,
    clippy::too_many_lines,
    reason = "the restart matrix passes explicit durable roots and lifecycle probes"
)]
fn checkpoint_matrix_spec(
    job_id: u64,
    sink_root: &Path,
    source_opened_with: &Arc<Mutex<Vec<Option<Vec<u8>>>>>,
    source_closed: &Arc<AtomicUsize>,
    sink_closed: &Arc<AtomicUsize>,
    sink_writes: &Arc<AtomicUsize>,
    source_poll_delay: Duration,
    cancellation: CancellationToken,
    exactly_once: bool,
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
    let window = checkpoint_matrix_window();
    let window_fields = window.output_ports()[0]
        .schema()
        .unwrap()
        .fields()
        .iter()
        .map(|field| field.as_ref().clone())
        .collect::<Vec<_>>();
    let plan = PipelineBuilder::new("checkpoint-restart-fault-matrix")
        .unwrap()
        .add_node("merge", Box::new(union))
        .unwrap()
        .add_node("window", Box::new(window))
        .unwrap()
        .add_checkpoint_capable_node(
            "branch_a",
            Box::new(CheckpointMatrixForward::new(window_fields.clone()))
                as Box<dyn StreamOperator>,
        )
        .unwrap()
        .add_checkpoint_capable_node(
            "branch_b",
            Box::new(CheckpointMatrixForward::new(window_fields)) as Box<dyn StreamOperator>,
        )
        .unwrap()
        .connect(Edge::new(
            PortEndpoint::new("merge", "output").unwrap(),
            PortEndpoint::new("window", "input").unwrap(),
        ))
        .unwrap()
        .connect(Edge::new(
            PortEndpoint::new("window", "output").unwrap(),
            PortEndpoint::new("branch_a", "input").unwrap(),
        ))
        .unwrap()
        .connect(Edge::new(
            PortEndpoint::new("window", "output").unwrap(),
            PortEndpoint::new("branch_b", "input").unwrap(),
        ))
        .unwrap()
        .compile_stream(
            &UdfRegistry::new().snapshot(),
            &if exactly_once {
                StreamRequirements {
                    delivery: BTreeMap::from([
                        (
                            "branch_a.output".into(),
                            crate::DeliveryGuarantee::ExactlyOnce,
                        ),
                        (
                            "branch_b.output".into(),
                            crate::DeliveryGuarantee::ExactlyOnce,
                        ),
                    ]),
                }
            } else {
                StreamRequirements::default()
            },
        )
        .unwrap();
    let sources = ["left", "right"]
        .into_iter()
        .map(|source_id| NamedSourceBinding {
            binding_id: source_id.into(),
            binding: SourceBinding::new(
                Box::new(CheckpointMatrixSource {
                    source_id,
                    next_sequence: 0,
                    pending_watermark: false,
                    ended: false,
                    poll_delay: source_poll_delay,
                    opened_with: Arc::clone(source_opened_with),
                    closed: Arc::clone(source_closed),
                }),
                None,
                0,
            )
            .unwrap(),
        })
        .collect();
    let sinks = [("branch_a.output", "sink-a"), ("branch_b.output", "sink-b")]
        .into_iter()
        .map(|(output_id, sink_id)| NamedSinkBinding {
            output_id: output_id.into(),
            sink_id: sink_id.into(),
            binding: OrdinarySinkBinding::new_transactional(Box::new(CheckpointMatrixSink {
                sink_id,
                root: sink_root.join(sink_id),
                epoch: None,
                pending: Vec::new(),
                written_records: Arc::clone(sink_writes),
                closed: Arc::clone(sink_closed),
                write_delay: Duration::ZERO,
            })),
        })
        .collect();
    ContinuousJobSpec {
        context: StreamJobContext::new(
            job_id,
            plan.fingerprint(),
            JsonMap::new(),
            None,
            cancellation,
        ),
        plan,
        sources,
        sinks,
        edge_budget: EdgeBudget {
            max_rows: 2,
            max_bytes: 1 << 20,
        },
        delivery_mode: M2DeliveryMode::ProcessLocalOrdered,
    }
}

#[allow(
    clippy::too_many_arguments,
    clippy::too_many_lines,
    reason = "the soak spec owns its exact graph, durable roots, and lifecycle probes"
)]
fn checkpoint_restart_soak_spec(
    job_id: u64,
    sink_root: &Path,
    stop: &Arc<AtomicUsize>,
    source_opened_with: &SourceOpenHistory,
    source_closed: &Arc<AtomicUsize>,
    sink_closed: &Arc<AtomicUsize>,
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
    let window = checkpoint_matrix_window();
    let window_fields = window.output_ports()[0]
        .schema()
        .unwrap()
        .fields()
        .iter()
        .map(|field| field.as_ref().clone())
        .collect::<Vec<_>>();
    let plan = PipelineBuilder::new("checkpoint-restart-soak")
        .unwrap()
        .add_node("merge", Box::new(union))
        .unwrap()
        .add_node("window", Box::new(window))
        .unwrap()
        .add_checkpoint_capable_node(
            "branch_a",
            Box::new(CheckpointMatrixForward::new(window_fields.clone()))
                as Box<dyn StreamOperator>,
        )
        .unwrap()
        .add_checkpoint_capable_node(
            "branch_b",
            Box::new(CheckpointMatrixForward::new(window_fields)) as Box<dyn StreamOperator>,
        )
        .unwrap()
        .connect(Edge::new(
            PortEndpoint::new("merge", "output").unwrap(),
            PortEndpoint::new("window", "input").unwrap(),
        ))
        .unwrap()
        .connect(Edge::new(
            PortEndpoint::new("window", "output").unwrap(),
            PortEndpoint::new("branch_a", "input").unwrap(),
        ))
        .unwrap()
        .connect(Edge::new(
            PortEndpoint::new("window", "output").unwrap(),
            PortEndpoint::new("branch_b", "input").unwrap(),
        ))
        .unwrap()
        .compile_stream(
            &UdfRegistry::new().snapshot(),
            &StreamRequirements {
                delivery: BTreeMap::from([
                    (
                        "branch_a.output".into(),
                        crate::DeliveryGuarantee::ExactlyOnce,
                    ),
                    (
                        "branch_b.output".into(),
                        crate::DeliveryGuarantee::ExactlyOnce,
                    ),
                ]),
            },
        )
        .unwrap();
    let sources = ["left", "right"]
        .into_iter()
        .map(|source_id| NamedSourceBinding {
            binding_id: source_id.into(),
            binding: SourceBinding::new(
                Box::new(CheckpointRestartSoakSource {
                    source_id,
                    next_sequence: 0,
                    pending_watermark: None,
                    poll_delay: Duration::from_millis(2),
                    stop: Arc::clone(stop),
                    opened_with: Arc::clone(source_opened_with),
                    closed: Arc::clone(source_closed),
                }),
                None,
                0,
            )
            .unwrap(),
        })
        .collect();
    let sink_writes = Arc::new(AtomicUsize::new(0));
    let sinks = [("branch_a.output", "sink-a"), ("branch_b.output", "sink-b")]
        .into_iter()
        .map(|(output_id, sink_id)| NamedSinkBinding {
            output_id: output_id.into(),
            sink_id: sink_id.into(),
            binding: OrdinarySinkBinding::new_transactional(Box::new(CheckpointMatrixSink {
                sink_id,
                root: sink_root.join(sink_id),
                epoch: None,
                pending: Vec::new(),
                written_records: Arc::clone(&sink_writes),
                closed: Arc::clone(sink_closed),
                write_delay: sink_write_delay,
            })),
        })
        .collect();
    ContinuousJobSpec {
        context: StreamJobContext::new(
            job_id,
            plan.fingerprint(),
            JsonMap::new(),
            None,
            CancellationToken::new(),
        ),
        plan,
        sources,
        sinks,
        edge_budget: EdgeBudget {
            max_rows: 8,
            max_bytes: 1 << 20,
        },
        delivery_mode: M2DeliveryMode::ProcessLocalOrdered,
    }
}

#[allow(
    clippy::struct_excessive_bools,
    reason = "the fault matrix report keeps independently asserted recovery dimensions"
)]
struct CheckpointFaultMatrixReport {
    selected_before_restart: Option<Epoch>,
    prepared_artifacts_after_failure: usize,
    restored_epoch: Option<Epoch>,
    restored_cursor_orders: BTreeMap<String, Vec<u8>>,
    sources_ended: bool,
    watermarks_restored: bool,
    window_state_restored: bool,
    visible_records: usize,
    duplicate_records: usize,
    missing_records: usize,
    temporary_artifacts: usize,
    terminal_tasks: usize,
    terminal_charged_edges: usize,
    cancellation_requested: bool,
    deterministic_terminal_error: bool,
}

#[derive(Debug, Eq, PartialEq)]
struct PrivateCheckpointBenchmarkReport {
    completed_jobs: usize,
    failed_jobs: usize,
    visible_records: usize,
    manifest_count: usize,
    restored_epoch: Option<Epoch>,
}

#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
struct PrivateBenchmarkProvenance {
    commit: String,
    tree: String,
    clean: bool,
    harness_hash: String,
    config_hash: String,
    executable: PathBuf,
    executable_sha256: String,
    build_identity_hash: String,
    toolchain_hash: String,
    environment_hash: String,
}

fn is_full_git_oid(value: &str) -> bool {
    value.len() == 40 && value.bytes().all(|byte| byte.is_ascii_hexdigit())
}

fn benchmark_evidence_error(message: impl Into<String>) -> CalcFlowError {
    CalcFlowError::InvalidArgument {
        field: "CALC_FLOW_M5_CHECKPOINT_BENCHMARK".into(),
        message: message.into(),
    }
}

fn sha256_bytes(bytes: &[u8]) -> String {
    hex::encode(Sha256::digest(bytes))
}

fn private_benchmark_build_identity_hash(provenance: &PrivateBenchmarkProvenance) -> String {
    sha256_bytes(
        &serde_json::to_vec(&json!([
            "calc-flow.m5-private-build-identity.v1",
            provenance.commit,
            provenance.tree,
            provenance.executable_sha256,
            provenance.toolchain_hash,
            provenance.harness_hash,
            provenance.config_hash,
            provenance.environment_hash,
        ]))
        .unwrap(),
    )
}

fn strict_command_output(program: &str, arguments: &[&str]) -> Result<String> {
    let output = Command::new(program)
        .args(arguments)
        .output()
        .map_err(|source| CalcFlowError::Io {
            path: program.into(),
            source,
        })?;
    if !output.status.success() {
        return Err(CalcFlowError::Internal {
            message: format!(
                "private benchmark command {program:?} failed: {}",
                String::from_utf8_lossy(&output.stderr).trim()
            ),
        });
    }
    Ok(String::from_utf8_lossy(&output.stdout).trim().to_owned())
}

fn private_benchmark_harness_hash() -> String {
    sha256_bytes(
        &serde_json::to_vec(&json!({
            "version": 2,
            "cases": M5_PRIVATE_BENCHMARK_CASES,
            "private_paths": [
                "LiveProgressCoordinator::checkpoint_cut",
                "OperatorBarrierAlignment",
                "ManifestTransaction::stage_operator_state",
                "ManifestTransaction::publish",
                "ManifestTransaction::select_latest/load_operator_state",
                "TransactionalStreamSink::commit",
                "ContinuousRunner::start_checkpointed/restart",
            ],
            "topology": {
                "sources": 2,
                "operators": ["union", "final_window", "two_forward_branches"],
                "sinks": 2,
            },
        }))
        .unwrap(),
    )
}

fn private_benchmark_config_hash() -> String {
    sha256_bytes(
        &serde_json::to_vec(&json!({
            "criterion": {
                "sample_size": CHECKPOINT_BENCHMARK_SAMPLE_SIZE,
                "warm_up_millis": CHECKPOINT_BENCHMARK_WARM_UP.as_millis(),
                "measurement_millis": CHECKPOINT_BENCHMARK_MEASUREMENT.as_millis(),
            },
        }))
        .unwrap(),
    )
}

fn current_private_benchmark_provenance() -> Result<PrivateBenchmarkProvenance> {
    let commit = strict_command_output("git", &["rev-parse", "HEAD"])?;
    let tree = strict_command_output("git", &["rev-parse", "HEAD^{tree}"])?;
    let clean = strict_command_output("git", &["status", "--porcelain=v1"])?.is_empty();
    let executable = std::env::current_exe()
        .and_then(std::fs::canonicalize)
        .map_err(|source| CalcFlowError::Io {
            path: "current benchmark executable".into(),
            source,
        })?;
    let executable_bytes = std::fs::read(&executable).map_err(|source| CalcFlowError::Io {
        path: executable.display().to_string(),
        source,
    })?;
    let executable_sha256 = sha256_bytes(&executable_bytes);
    let toolchain = strict_command_output("rustc", &["-vV"])?;
    let toolchain_hash = sha256_bytes(toolchain.as_bytes());
    let environment = json!({
        "os": std::env::consts::OS,
        "arch": std::env::consts::ARCH,
        "kernel": command_output("uname", &["-srvmo"]),
        "rustc_host": toolchain.lines().find(|line| line.starts_with("host: ")),
        "rustflags": std::env::var("RUSTFLAGS").unwrap_or_default(),
    });
    let environment_hash = sha256_bytes(&serde_json::to_vec(&environment).unwrap());
    let harness_hash = private_benchmark_harness_hash();
    let config_hash = private_benchmark_config_hash();
    let mut provenance = PrivateBenchmarkProvenance {
        commit,
        tree,
        clean,
        harness_hash,
        config_hash,
        executable,
        executable_sha256,
        build_identity_hash: String::new(),
        toolchain_hash,
        environment_hash,
    };
    provenance.build_identity_hash = private_benchmark_build_identity_hash(&provenance);
    Ok(provenance)
}

fn validate_private_benchmark_source_identity(
    provenance: &PrivateBenchmarkProvenance,
    embedded_commit: Option<&str>,
    embedded_tree: Option<&str>,
) -> Result<()> {
    let embedded_commit = embedded_commit.ok_or_else(|| {
        benchmark_evidence_error("private benchmark binary has no embedded source commit")
    })?;
    let embedded_tree = embedded_tree.ok_or_else(|| {
        benchmark_evidence_error("private benchmark binary has no embedded source tree")
    })?;
    if !is_full_git_oid(embedded_commit)
        || !is_full_git_oid(embedded_tree)
        || embedded_commit != provenance.commit
        || embedded_tree != provenance.tree
    {
        return Err(benchmark_evidence_error(
            "private benchmark binary source identity is stale",
        ));
    }
    Ok(())
}

fn private_benchmark_case_directory(case: &str) -> String {
    case.replace('/', "_")
}

fn private_benchmark_sample_reference(path: &Path, target_root: &Path) -> Result<String> {
    path.strip_prefix(target_root)
        .map(|relative| relative.to_string_lossy().into_owned())
        .map_err(|_| CalcFlowError::Internal {
            message: format!(
                "private benchmark sample {} escapes target root {}",
                path.display(),
                target_root.display()
            ),
        })
}

fn finite_positive_measurement(value: &serde_json::Value, field: &str) -> Result<f64> {
    let value = value
        .as_f64()
        .filter(|value| value.is_finite() && *value > 0.0)
        .ok_or_else(|| CalcFlowError::Internal {
            message: format!("private benchmark {field} must be finite and positive"),
        })?;
    Ok(value)
}

fn median(values: &mut [f64]) -> f64 {
    values.sort_by(f64::total_cmp);
    let middle = values.len() / 2;
    if values.len().is_multiple_of(2) {
        values[middle - 1].midpoint(values[middle])
    } else {
        values[middle]
    }
}

fn bootstrap_median_confidence_interval(samples: &[f64]) -> [f64; 2] {
    const RESAMPLES: usize = 4_096;
    let mut state = 0x4d35_6570_6f63_6842_u64;
    let sample_count = u64::try_from(samples.len()).unwrap();
    let mut medians = Vec::with_capacity(RESAMPLES);
    for _ in 0..RESAMPLES {
        let mut resample = Vec::with_capacity(samples.len());
        for _ in samples {
            state = state
                .wrapping_mul(6_364_136_223_846_793_005)
                .wrapping_add(1_442_695_040_888_963_407);
            let index = usize::try_from(state % sample_count).unwrap();
            resample.push(samples[index]);
        }
        medians.push(median(&mut resample));
    }
    medians.sort_by(f64::total_cmp);
    [medians[RESAMPLES / 40], medians[RESAMPLES * 39 / 40]]
}

fn private_benchmark_criterion_summary(
    estimates: &serde_json::Value,
) -> Result<(f64, [f64; 2], f64)> {
    let median =
        finite_positive_measurement(&estimates["median"]["point_estimate"], "Criterion median")?;
    let lower = finite_positive_measurement(
        &estimates["median"]["confidence_interval"]["lower_bound"],
        "Criterion lower confidence bound",
    )?;
    let upper = finite_positive_measurement(
        &estimates["median"]["confidence_interval"]["upper_bound"],
        "Criterion upper confidence bound",
    )?;
    let confidence_level = estimates["median"]["confidence_interval"]["confidence_level"]
        .as_f64()
        .filter(|value| value.to_bits() == 0.95_f64.to_bits())
        .ok_or_else(|| CalcFlowError::Internal {
            message: "private benchmark Criterion confidence level must be exactly 0.95".into(),
        })?;
    if lower > median || median > upper {
        return Err(CalcFlowError::Internal {
            message: "private benchmark Criterion confidence interval is not ordered".into(),
        });
    }
    Ok((median, [lower, upper], confidence_level))
}

fn private_benchmark_absolute_measurement(
    case: &str,
    run_label: &str,
    run_root: &Path,
    target_root: &Path,
) -> Result<serde_json::Value> {
    let case_root = run_root
        .join(private_benchmark_case_directory(case))
        .join(run_label);
    let estimates_path = case_root.join("estimates.json");
    let sample_path = case_root.join("sample.json");
    let estimates_bytes = std::fs::read(&estimates_path).map_err(|source| CalcFlowError::Io {
        path: estimates_path.display().to_string(),
        source,
    })?;
    let sample_bytes = std::fs::read(&sample_path).map_err(|source| CalcFlowError::Io {
        path: sample_path.display().to_string(),
        source,
    })?;
    let estimates: serde_json::Value =
        serde_json::from_slice(&estimates_bytes).map_err(|error| CalcFlowError::Internal {
            message: format!(
                "private benchmark estimate {} is invalid: {error}",
                estimates_path.display()
            ),
        })?;
    let (criterion_median, [criterion_lower, criterion_upper], confidence_level) =
        private_benchmark_criterion_summary(&estimates)?;
    let sample: serde_json::Value =
        serde_json::from_slice(&sample_bytes).map_err(|error| CalcFlowError::Internal {
            message: format!(
                "private benchmark sample {} is invalid: {error}",
                sample_path.display()
            ),
        })?;
    let iterations = sample["iters"]
        .as_array()
        .ok_or_else(|| CalcFlowError::Internal {
            message: "private benchmark sample iterations are missing".into(),
        })?;
    let times = sample["times"]
        .as_array()
        .ok_or_else(|| CalcFlowError::Internal {
            message: "private benchmark sample times are missing".into(),
        })?;
    if iterations.len() != CHECKPOINT_BENCHMARK_SAMPLE_SIZE || times.len() != iterations.len() {
        return Err(CalcFlowError::Internal {
            message: format!(
                "private benchmark requires exactly {CHECKPOINT_BENCHMARK_SAMPLE_SIZE} matched samples"
            ),
        });
    }
    let mut normalized = iterations
        .iter()
        .zip(times)
        .enumerate()
        .map(|(index, (iterations, elapsed))| {
            let iterations = finite_positive_measurement(
                iterations,
                &format!("sample {index} iteration count"),
            )?;
            let elapsed =
                finite_positive_measurement(elapsed, &format!("sample {index} elapsed time"))?;
            let normalized = elapsed / iterations;
            if !normalized.is_finite() || normalized <= 0.0 {
                return Err(CalcFlowError::Internal {
                    message: format!("private benchmark sample {index} is not finite"),
                });
            }
            Ok(normalized)
        })
        .collect::<Result<Vec<_>>>()?;
    let bootstrap_confidence = bootstrap_median_confidence_interval(&normalized);
    let minimum = normalized.iter().copied().min_by(f64::total_cmp).unwrap();
    let maximum = normalized.iter().copied().max_by(f64::total_cmp).unwrap();
    let raw_median = median(&mut normalized);
    if bootstrap_confidence[0] > raw_median || raw_median > bootstrap_confidence[1] {
        return Err(CalcFlowError::Internal {
            message: "private benchmark bootstrap confidence interval is not ordered".into(),
        });
    }
    Ok(json!({
        "case": case,
        "comparison": "none",
        "median_ns": raw_median,
        "median_confidence_interval_ns": bootstrap_confidence,
        "confidence_level": confidence_level,
        "sample_count": normalized.len(),
        "relative_sample_spread_percent": (maximum / minimum - 1.0) * 100.0,
        "criterion": {
            "median_ns": criterion_median,
            "median_confidence_interval_ns": [criterion_lower, criterion_upper],
        },
        "decision": "absolute_only",
        "artifacts": {
            "sample": {
                "path": private_benchmark_sample_reference(&sample_path, target_root)?,
                "sha256": sha256_bytes(&sample_bytes),
            },
            "estimates": {
                "path": private_benchmark_sample_reference(&estimates_path, target_root)?,
                "sha256": sha256_bytes(&estimates_bytes),
            },
        },
    }))
}

fn validate_private_benchmark_artifacts(
    measurement: &serde_json::Value,
    target_root: &Path,
) -> Result<()> {
    if measurement["comparison"] != "none"
        || measurement["decision"] != "absolute_only"
        || measurement["sample_count"] != CHECKPOINT_BENCHMARK_SAMPLE_SIZE
        || measurement["confidence_level"].as_f64().map(f64::to_bits) != Some(0.95_f64.to_bits())
    {
        return Err(benchmark_evidence_error(
            "private absolute measurement contract is invalid",
        ));
    }
    let median = finite_positive_measurement(&measurement["median_ns"], "reported median")?;
    let confidence = measurement["median_confidence_interval_ns"]
        .as_array()
        .filter(|confidence| confidence.len() == 2)
        .ok_or_else(|| benchmark_evidence_error("reported confidence interval is invalid"))?;
    let lower = finite_positive_measurement(&confidence[0], "reported lower confidence bound")?;
    let upper = finite_positive_measurement(&confidence[1], "reported upper confidence bound")?;
    if lower > median || median > upper {
        return Err(benchmark_evidence_error(
            "reported confidence interval is not ordered",
        ));
    }
    let artifacts = measurement["artifacts"]
        .as_object()
        .filter(|artifacts| artifacts.len() == 2)
        .ok_or_else(|| benchmark_evidence_error("private benchmark artifacts are incomplete"))?;
    let canonical_root = target_root
        .canonicalize()
        .map_err(|source| CalcFlowError::Io {
            path: target_root.display().to_string(),
            source,
        })?;
    for name in ["sample", "estimates"] {
        let artifact = artifacts
            .get(name)
            .ok_or_else(|| benchmark_evidence_error(format!("{name} artifact is missing")))?;
        let reference = artifact["path"]
            .as_str()
            .ok_or_else(|| benchmark_evidence_error(format!("{name} artifact path is missing")))?;
        let reference = Path::new(reference);
        if reference.is_absolute() {
            return Err(benchmark_evidence_error(format!(
                "{name} artifact path must be relative"
            )));
        }
        let path = canonical_root.join(reference);
        let metadata = std::fs::symlink_metadata(&path).map_err(|source| CalcFlowError::Io {
            path: path.display().to_string(),
            source,
        })?;
        if metadata.file_type().is_symlink() || !metadata.is_file() {
            return Err(benchmark_evidence_error(format!(
                "{name} artifact is not a regular file"
            )));
        }
        let canonical = path.canonicalize().map_err(|source| CalcFlowError::Io {
            path: path.display().to_string(),
            source,
        })?;
        if !canonical.starts_with(&canonical_root) {
            return Err(benchmark_evidence_error(format!(
                "{name} artifact escapes the benchmark target root"
            )));
        }
        let expected = artifact["sha256"]
            .as_str()
            .filter(|digest| {
                digest.len() == 64 && digest.bytes().all(|byte| byte.is_ascii_hexdigit())
            })
            .ok_or_else(|| benchmark_evidence_error(format!("{name} artifact hash is invalid")))?;
        let bytes = std::fs::read(&canonical).map_err(|source| CalcFlowError::Io {
            path: canonical.display().to_string(),
            source,
        })?;
        if sha256_bytes(&bytes) != expected {
            return Err(benchmark_evidence_error(format!(
                "{name} artifact hash does not match the report"
            )));
        }
    }
    Ok(())
}

fn private_checkpoint_benchmark_metadata(
    provenance: &PrivateBenchmarkProvenance,
    criterion_root: &Path,
    measurements: &[serde_json::Value],
) -> serde_json::Value {
    json!({
        "schema": "calc-flow.m5-checkpoint-absolute-benchmark.v1",
        "commit": provenance.commit,
        "mode": "candidate_absolute",
        "command": CHECKPOINT_BENCHMARK_COMMAND,
        "criterion_run_root": criterion_root,
        "provenance": provenance,
        "comparison": "none",
        "absolute_cases": M5_PRIVATE_BENCHMARK_CASES,
        "measurements": measurements,
        "overall_result": "absolute_only",
        "topology": {
            "sources": 2,
            "operators": ["union", "final_window", "two_forward_branches"],
            "sinks": 2,
            "checkpoint": "periodic durable epoch followed by deterministic process restart",
        },
    })
}

fn write_private_checkpoint_benchmark_metadata(
    target_root: &Path,
    commit: &str,
    run_id: &str,
    metadata: &serde_json::Value,
) -> Result<PathBuf> {
    if !is_full_git_oid(commit) {
        return Err(benchmark_evidence_error(
            "benchmark report name requires an exact commit SHA",
        ));
    }
    if run_id.is_empty()
        || run_id.len() > 64
        || !run_id
            .bytes()
            .all(|byte| byte.is_ascii_alphanumeric() || matches!(byte, b'-' | b'_'))
    {
        return Err(benchmark_evidence_error(
            "benchmark report run ID is invalid",
        ));
    }
    let artifact_root = target_root.join("m5-checkpoint-benchmark");
    std::fs::create_dir_all(&artifact_root).map_err(|source| CalcFlowError::Io {
        path: artifact_root.display().to_string(),
        source,
    })?;
    let path = artifact_root.join(format!("{commit}-{run_id}.json"));
    let digest_path = artifact_root.join(format!("{commit}-{run_id}.json.sha256"));
    if path.exists() || digest_path.exists() {
        return Err(CalcFlowError::Conflict {
            resource: "private benchmark report".into(),
            key: commit.into(),
        });
    }
    let bytes = serde_json::to_vec_pretty(metadata).unwrap();
    let mut file = OpenOptions::new()
        .create_new(true)
        .write(true)
        .open(&path)
        .map_err(|source| CalcFlowError::Io {
            path: path.display().to_string(),
            source,
        })?;
    file.write_all(&bytes).map_err(|source| CalcFlowError::Io {
        path: path.display().to_string(),
        source,
    })?;
    file.sync_all().map_err(|source| CalcFlowError::Io {
        path: path.display().to_string(),
        source,
    })?;
    let mut digest_file = OpenOptions::new()
        .create_new(true)
        .write(true)
        .open(&digest_path)
        .map_err(|source| CalcFlowError::Io {
            path: digest_path.display().to_string(),
            source,
        })?;
    digest_file
        .write_all(format!("{}\n", sha256_bytes(&bytes)).as_bytes())
        .map_err(|source| CalcFlowError::Io {
            path: digest_path.display().to_string(),
            source,
        })?;
    digest_file.sync_all().map_err(|source| CalcFlowError::Io {
        path: digest_path.display().to_string(),
        source,
    })?;
    File::open(&artifact_root)
        .and_then(|directory| directory.sync_all())
        .map_err(|source| CalcFlowError::Io {
            path: artifact_root.display().to_string(),
            source,
        })?;
    Ok(path)
}

fn validate_private_checkpoint_benchmark_report(path: &Path) -> Result<serde_json::Value> {
    let metadata = std::fs::symlink_metadata(path).map_err(|source| CalcFlowError::Io {
        path: path.display().to_string(),
        source,
    })?;
    if metadata.file_type().is_symlink() || !metadata.is_file() {
        return Err(benchmark_evidence_error(
            "private benchmark report is not a regular file",
        ));
    }
    let bytes = std::fs::read(path).map_err(|source| CalcFlowError::Io {
        path: path.display().to_string(),
        source,
    })?;
    let digest_path = PathBuf::from(format!("{}.sha256", path.display()));
    let expected = std::fs::read_to_string(&digest_path).map_err(|source| CalcFlowError::Io {
        path: digest_path.display().to_string(),
        source,
    })?;
    let expected = expected.trim();
    if expected.len() != 64
        || !expected.bytes().all(|byte| byte.is_ascii_hexdigit())
        || sha256_bytes(&bytes) != expected
    {
        return Err(benchmark_evidence_error(
            "private benchmark report hash does not match its sidecar",
        ));
    }
    let report: serde_json::Value = serde_json::from_slice(&bytes)
        .map_err(|error| benchmark_evidence_error(error.to_string()))?;
    if report["schema"] != "calc-flow.m5-checkpoint-absolute-benchmark.v1" {
        return Err(benchmark_evidence_error(
            "private benchmark report schema is invalid",
        ));
    }
    Ok(report)
}

fn private_checkpoint_benchmark_target_root() -> PathBuf {
    let workspace_root = Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../..")
        .canonicalize()
        .unwrap();
    std::env::var_os("CARGO_TARGET_DIR").map_or_else(
        || workspace_root.join("target"),
        |configured| {
            let configured = PathBuf::from(configured);
            if configured.is_absolute() {
                configured
            } else {
                workspace_root.join(configured)
            }
        },
    )
}

fn checkpoint_manifest_count(root: &Path) -> usize {
    match std::fs::read_dir(root) {
        Ok(entries) => entries
            .filter_map(std::result::Result::ok)
            .filter(|entry| {
                entry.file_name().to_str().is_some_and(|name| {
                    name.starts_with("manifest-")
                        && Path::new(name)
                            .extension()
                            .is_some_and(|extension| extension.eq_ignore_ascii_case("json"))
                })
            })
            .count(),
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => 0,
        Err(error) => panic!("failed to enumerate benchmark manifests: {error}"),
    }
}

const PRIVATE_BENCHMARK_PIPELINE_FINGERPRINT: &str =
    "0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef";
const PRIVATE_BENCHMARK_RUNTIME_HASH: &str =
    "abcdef0123456789abcdef0123456789abcdef0123456789abcdef0123456789";

fn private_benchmark_source(binding_id: &str) -> SourceBindingSpec {
    SourceBindingSpec {
        descriptor: SourceDescriptor::new(
            BindingIdentity::new(binding_id).unwrap(),
            DeclaredSchema::DynamicOrUnknown,
            NativeWatermarkCapability::NeverEmits,
            ReplayPositioningCapability::ExactPauseReportAndSeek,
            None,
        ),
        watermark_policy: WatermarkPolicy::Disabled { idle_timeout: None },
    }
}

#[allow(
    clippy::similar_names,
    reason = "the four symmetric fan-out endpoints retain their topology identity"
)]
async fn run_private_barrier_cut_fan_out_benchmark_case() -> usize {
    let prepared = Arc::new(
        prepare_stream_job(
            "m5-private-barrier-cut",
            &[
                private_benchmark_source("left"),
                private_benchmark_source("right"),
            ],
            StreamProgressRuntimeConfig::default(),
        )
        .unwrap(),
    );
    let budget = EdgeBudget {
        max_rows: 4,
        max_bytes: 1 << 20,
    };
    let (left_a, left_primary_receiver) = crate::edge_channel("left-a", budget).unwrap();
    let (left_b, left_secondary_receiver) = crate::edge_channel("left-b", budget).unwrap();
    let (right_a, right_primary_receiver) = crate::edge_channel("right-a", budget).unwrap();
    let (right_b, right_secondary_receiver) = crate::edge_channel("right-b", budget).unwrap();
    let cancellation = CancellationToken::new();
    let coordinator = LiveProgressCoordinator::new(
        &prepared,
        BTreeMap::from([
            ("left".into(), vec![left_a, left_b]),
            ("right".into(), vec![right_a, right_b]),
        ]),
        cancellation.clone(),
    )
    .unwrap();
    coordinator
        .checkpoint_cut(
            Epoch::INITIAL,
            &BTreeMap::from([
                (
                    BindingIdentity::new("left").unwrap(),
                    DurableSourceCut {
                        cursor: Some(CursorManifestEntry {
                            order: "01".into(),
                            payload: BTreeMap::new(),
                        }),
                        next_sequence: 0,
                        ended: false,
                    },
                ),
                (
                    BindingIdentity::new("right").unwrap(),
                    DurableSourceCut {
                        cursor: Some(CursorManifestEntry {
                            order: "02".into(),
                            payload: BTreeMap::new(),
                        }),
                        next_sequence: 0,
                        ended: false,
                    },
                ),
            ]),
            &cancellation,
        )
        .await
        .unwrap();
    let mut receivers = [
        left_primary_receiver,
        left_secondary_receiver,
        right_primary_receiver,
        right_secondary_receiver,
    ];
    let mut barriers = 0;
    for receiver in &mut receivers {
        let message = receiver.recv().await.unwrap().unwrap();
        assert_eq!(message.as_barrier(), Some(Epoch::INITIAL));
        barriers += 1;
    }
    barriers
}

fn private_benchmark_identity(operator_ids: BTreeSet<String>) -> PreparedManifestIdentity {
    PreparedManifestIdentity {
        pipeline_name: "m5-private-benchmark".into(),
        pipeline_fingerprint: PRIVATE_BENCHMARK_PIPELINE_FINGERPRINT.into(),
        runtime_config_hash: PRIVATE_BENCHMARK_RUNTIME_HASH.into(),
        source_ids: BTreeSet::new(),
        operator_ids,
        sink_ids: BTreeSet::new(),
    }
}

fn private_benchmark_manifest(
    epoch: Epoch,
    operators: BTreeMap<String, OperatorManifestEntry>,
) -> BenchmarkCheckpointManifest {
    BenchmarkCheckpointManifest::new(CheckpointManifestFields {
        pipeline_name: "m5-private-benchmark".into(),
        pipeline_fingerprint: PRIVATE_BENCHMARK_PIPELINE_FINGERPRINT.into(),
        runtime_config_hash: PRIVATE_BENCHMARK_RUNTIME_HASH.into(),
        epoch,
        created_at: Utc.with_ymd_and_hms(2026, 8, 9, 8, 0, 0).unwrap(),
        recovery_status: RecoveryStatus::Final,
        sources: BTreeMap::new(),
        operators,
        sinks: BTreeMap::new(),
    })
    .unwrap()
}

async fn private_benchmark_transaction(
    operator_ids: BTreeSet<String>,
) -> (
    tempfile::TempDir,
    ManifestTransaction,
    PreparedManifestIdentity,
) {
    let directory = tempfile::tempdir().unwrap();
    let backend = LocalStateBackend::new(directory.path().join("state"))
        .await
        .unwrap();
    let key = StateLineageKey::new(
        "m5-private-benchmark",
        PRIVATE_BENCHMARK_PIPELINE_FINGERPRINT,
    )
    .unwrap();
    let lineage: Arc<dyn StateLineageBackend> =
        Arc::from(backend.open_lineage(&key).await.unwrap());
    let transaction =
        ManifestTransaction::open(lineage, &key, directory.path().join("manifests"), 2)
            .await
            .unwrap();
    let identity = private_benchmark_identity(operator_ids);
    (directory, transaction, identity)
}

async fn run_private_dirty_window_state_stage_benchmark_case() -> usize {
    let (_directory, transaction, _identity) =
        private_benchmark_transaction(BTreeSet::from(["window".into()])).await;
    transaction
        .stage_operator_state(
            "window",
            Epoch::INITIAL,
            OperatorStateSnapshot {
                inline_metadata: BTreeMap::new(),
                segments: BTreeMap::from([("delta".into(), b"dirty-window-state".to_vec())]),
            },
        )
        .await
        .unwrap()
        .segments
        .len()
}

async fn run_private_manifest_publication_benchmark_case() -> usize {
    let (directory, transaction, _identity) = private_benchmark_transaction(BTreeSet::new()).await;
    transaction
        .publish(PreparedEpochManifest {
            manifest: private_benchmark_manifest(Epoch::INITIAL, BTreeMap::new()),
            staged_segments: BTreeMap::new(),
        })
        .await
        .unwrap();
    checkpoint_manifest_count(&directory.path().join("manifests"))
}

async fn run_private_cold_restore_benchmark_case() -> usize {
    let (_directory, transaction, identity) =
        private_benchmark_transaction(BTreeSet::from(["window".into()])).await;
    let staged = transaction
        .stage_operator_state(
            "window",
            Epoch::INITIAL,
            OperatorStateSnapshot {
                inline_metadata: BTreeMap::new(),
                segments: BTreeMap::from([("delta".into(), b"restored-window-state".to_vec())]),
            },
        )
        .await
        .unwrap();
    transaction
        .publish(PreparedEpochManifest {
            manifest: private_benchmark_manifest(
                Epoch::INITIAL,
                BTreeMap::from([(
                    "window".into(),
                    OperatorManifestEntry {
                        progress: BTreeMap::new(),
                        inline_metadata: staged.inline_metadata,
                        segments: staged.segments,
                    },
                )]),
            ),
            staged_segments: BTreeMap::new(),
        })
        .await
        .unwrap();
    let selected = transaction.select_latest(&identity).await.unwrap().unwrap();
    transaction
        .load_operator_state(
            "window",
            selected.manifest.epoch(),
            &selected.manifest.operators()["window"],
        )
        .await
        .unwrap()
        .segments
        .len()
}

async fn run_private_transactional_sink_commit_benchmark_case() -> usize {
    let directory = tempfile::tempdir().unwrap();
    let mut sink = CheckpointMatrixSink {
        sink_id: "sink-a",
        root: directory.path().join("sink-a"),
        epoch: None,
        pending: Vec::new(),
        written_records: Arc::new(AtomicUsize::new(0)),
        closed: Arc::new(AtomicUsize::new(0)),
        write_delay: Duration::ZERO,
    };
    sink.open().await.unwrap();
    sink.begin_epoch(Epoch::INITIAL).await.unwrap();
    sink.pending.push("sink-a|0|1|left|1".into());
    let state = sink.pre_commit(Epoch::INITIAL).await.unwrap();
    sink.commit(Epoch::INITIAL, &state).await.unwrap();
    sink.read_visible_epochs()
        .await
        .unwrap()
        .values()
        .map(Vec::len)
        .sum()
}

async fn run_private_checkpoint_path_benchmark_smoke() -> BTreeMap<&'static str, usize> {
    BTreeMap::from([
        (
            "m5/private_path/barrier_cut_fan_out",
            run_private_barrier_cut_fan_out_benchmark_case().await,
        ),
        (
            "m5/private_path/two_input_alignment",
            super::operator_task::benchmark_two_input_alignment().unwrap(),
        ),
        (
            "m5/private_path/dirty_window_state_stage",
            run_private_dirty_window_state_stage_benchmark_case().await,
        ),
        (
            "m5/private_path/production_manifest_publication",
            run_private_manifest_publication_benchmark_case().await,
        ),
        (
            "m5/private_path/cold_restore",
            run_private_cold_restore_benchmark_case().await,
        ),
        (
            "m5/private_path/transactional_sink_commit",
            run_private_transactional_sink_commit_benchmark_case().await,
        ),
    ])
}

async fn run_private_no_checkpoint_benchmark_case() -> PrivateCheckpointBenchmarkReport {
    let directory = tempfile::tempdir().unwrap();
    let sink_root = directory.path().join("sinks");
    let source_opens = Arc::new(Mutex::new(Vec::new()));
    let source_closed = Arc::new(AtomicUsize::new(0));
    let sink_closed = Arc::new(AtomicUsize::new(0));
    let sink_writes = Arc::new(AtomicUsize::new(0));
    let spec = checkpoint_matrix_spec(
        30_000,
        &sink_root,
        &source_opens,
        &source_closed,
        &sink_closed,
        &sink_writes,
        Duration::ZERO,
        CancellationToken::new(),
        false,
    );
    let mut runner = ContinuousRunner::new();
    let job = runner.start(spec).await.unwrap();
    let outcome = tokio::time::timeout(Duration::from_secs(5), job.wait())
        .await
        .expect("private no-checkpoint benchmark case hung");
    assert_eq!(outcome.state, ContinuousJobState::Completed);
    drop(job);
    runner.shutdown().await.unwrap();
    assert_eq!(runner.registry_counts(), (0, 0));
    assert_eq!(source_closed.load(Ordering::SeqCst), 2);
    assert_eq!(sink_closed.load(Ordering::SeqCst), 2);
    PrivateCheckpointBenchmarkReport {
        completed_jobs: 1,
        failed_jobs: 0,
        visible_records: sink_writes.load(Ordering::SeqCst),
        manifest_count: checkpoint_manifest_count(&directory.path().join("manifests")),
        restored_epoch: None,
    }
}

#[allow(
    clippy::too_many_lines,
    reason = "the private benchmark measures one complete checkpoint and cold-restart lifecycle"
)]
async fn run_private_checkpoint_restart_benchmark_case() -> PrivateCheckpointBenchmarkReport {
    let directory = tempfile::tempdir().unwrap();
    let state_root = directory.path().join("state");
    let manifest_root = directory.path().join("manifests");
    let sink_root = directory.path().join("sinks");
    let backend = Arc::new(LocalStateBackend::new(&state_root).await.unwrap());
    let sink_writes = Arc::new(AtomicUsize::new(0));

    let first_source_opens = Arc::new(Mutex::new(Vec::new()));
    let first_source_closed = Arc::new(AtomicUsize::new(0));
    let first_sink_closed = Arc::new(AtomicUsize::new(0));
    let first_spec = checkpoint_matrix_spec(
        30_001,
        &sink_root,
        &first_source_opens,
        &first_source_closed,
        &first_sink_closed,
        &sink_writes,
        Duration::from_millis(5),
        CancellationToken::new(),
        true,
    );
    let first_checkpoint = CheckpointRuntimeSpec::new(
        backend.clone(),
        &manifest_root,
        StreamRuntimeConfig {
            checkpoint_interval: Duration::from_millis(1),
            checkpoint_timeout: Duration::from_secs(5),
            retained_epochs: 2,
            ..StreamRuntimeConfig::default()
        },
    )
    .unwrap()
    .with_fault(
        CheckpointFaultPoint::CompletedCommit,
        CheckpointFaultMode::Restart,
    );
    let mut first_runner = ContinuousRunner::new();
    let first_job = first_runner
        .start_checkpointed(first_spec, first_checkpoint)
        .await
        .unwrap();
    let first_outcome = tokio::time::timeout(Duration::from_secs(5), first_job.wait())
        .await
        .expect("private checkpoint benchmark pre-restart case hung");
    assert_eq!(first_outcome.state, ContinuousJobState::Failed);
    drop(first_job);
    first_runner.shutdown().await.unwrap();
    assert_eq!(first_runner.registry_counts(), (0, 0));

    let restart_source_opens = Arc::new(Mutex::new(Vec::new()));
    let restart_source_closed = Arc::new(AtomicUsize::new(0));
    let restart_sink_closed = Arc::new(AtomicUsize::new(0));
    let restart_spec = checkpoint_matrix_spec(
        30_002,
        &sink_root,
        &restart_source_opens,
        &restart_source_closed,
        &restart_sink_closed,
        &sink_writes,
        Duration::ZERO,
        CancellationToken::new(),
        true,
    );
    let restart_checkpoint = CheckpointRuntimeSpec::new(
        backend,
        &manifest_root,
        StreamRuntimeConfig {
            checkpoint_interval: Duration::from_secs(3_600),
            checkpoint_timeout: Duration::from_secs(5),
            retained_epochs: 2,
            ..StreamRuntimeConfig::default()
        },
    )
    .unwrap();
    let mut restart_runner = ContinuousRunner::new();
    let restart_job = restart_runner
        .start_checkpointed(restart_spec, restart_checkpoint)
        .await
        .unwrap();
    let restart_outcome = tokio::time::timeout(Duration::from_secs(5), restart_job.wait())
        .await
        .expect("private checkpoint benchmark restart case hung");
    assert_eq!(restart_outcome.state, ContinuousJobState::Completed);
    let restored_epoch = restart_job
        .status()
        .checkpoint
        .and_then(|checkpoint| checkpoint.last_completed_epoch);
    drop(restart_job);
    restart_runner.shutdown().await.unwrap();
    assert_eq!(restart_runner.registry_counts(), (0, 0));
    let visible_records = read_checkpoint_matrix_visible(&sink_root)
        .await
        .values()
        .map(Vec::len)
        .sum();

    PrivateCheckpointBenchmarkReport {
        completed_jobs: 1,
        failed_jobs: 1,
        visible_records,
        manifest_count: checkpoint_manifest_count(&manifest_root),
        restored_epoch,
    }
}

async fn read_checkpoint_matrix_visible(root: &Path) -> BTreeMap<String, Vec<String>> {
    let mut visible = BTreeMap::new();
    for sink_id in ["sink-a", "sink-b"] {
        let path = root.join(sink_id).join("visible.json");
        let epochs: BTreeMap<String, Vec<String>> = serde_json::from_slice(
            &tokio::fs::read(&path)
                .await
                .unwrap_or_else(|error| panic!("failed to read {}: {error}", path.display())),
        )
        .unwrap();
        visible.insert(sink_id.into(), epochs.into_values().flatten().collect());
    }
    visible
}

#[derive(Debug, Default, Eq, PartialEq)]
struct CheckpointTemporaryArtifacts {
    state: usize,
    manifests: usize,
    sink_a: usize,
    sink_b: usize,
}

impl CheckpointTemporaryArtifacts {
    const fn total(&self) -> usize {
        self.state + self.manifests + self.sink_a + self.sink_b
    }
}

fn count_checkpoint_artifacts(root: &Path, is_durable_file: impl Fn(&Path) -> bool) -> usize {
    let mut count = 0;
    let mut directories = vec![root.to_path_buf()];
    while let Some(directory) = directories.pop() {
        let entries = match std::fs::read_dir(&directory) {
            Ok(entries) => entries,
            Err(error) if error.kind() == std::io::ErrorKind::NotFound => continue,
            Err(error) => panic!(
                "checkpoint oracle could not inspect {}: {error}",
                directory.display()
            ),
        };
        for entry in entries {
            let entry = entry.unwrap_or_else(|error| {
                panic!(
                    "checkpoint oracle could not read an entry under {}: {error}",
                    directory.display()
                )
            });
            let path = entry.path();
            let metadata = std::fs::symlink_metadata(&path).unwrap_or_else(|error| {
                panic!(
                    "checkpoint oracle could not inspect {}: {error}",
                    path.display()
                )
            });
            if metadata.is_dir() {
                directories.push(path);
            } else if !is_durable_file(&path) {
                count += 1;
            }
        }
    }
    count
}

fn is_canonical_checkpoint_manifest(path: &Path) -> bool {
    let Some(name) = path.file_name().and_then(|name| name.to_str()) else {
        return false;
    };
    let Some(epoch) = name
        .strip_prefix("manifest-")
        .and_then(|name| name.strip_suffix(".json"))
    else {
        return false;
    };
    epoch.len() == 20 && epoch.bytes().all(|byte| byte.is_ascii_digit())
}

fn inspect_checkpoint_temporary_artifacts(
    state_root: &Path,
    manifest_root: &Path,
    sink_root: &Path,
) -> CheckpointTemporaryArtifacts {
    CheckpointTemporaryArtifacts {
        state: count_checkpoint_artifacts(&state_root.join("staging"), |_| false),
        manifests: count_checkpoint_artifacts(manifest_root, is_canonical_checkpoint_manifest),
        sink_a: count_checkpoint_artifacts(&sink_root.join("sink-a"), |path| {
            path.file_name().and_then(|name| name.to_str()) == Some("visible.json")
        }),
        sink_b: count_checkpoint_artifacts(&sink_root.join("sink-b"), |path| {
            path.file_name().and_then(|name| name.to_str()) == Some("visible.json")
        }),
    }
}

fn checkpoint_matrix_temporary_artifacts(
    state_root: &Path,
    manifest_root: &Path,
    sink_root: &Path,
) -> usize {
    inspect_checkpoint_temporary_artifacts(state_root, manifest_root, sink_root).total()
}

#[allow(
    clippy::too_many_lines,
    reason = "one restart case owns the failure, durable selection, replay, and output oracle"
)]
async fn run_checkpoint_restart_fault_case(
    point: CheckpointFaultPoint,
    mode: CheckpointFaultMode,
) -> CheckpointFaultMatrixReport {
    let directory = tempfile::tempdir().unwrap();
    let state_root = directory.path().join("state");
    let manifest_root = directory.path().join("manifests");
    let sink_root = directory.path().join("sinks");
    let backend = Arc::new(LocalStateBackend::new(&state_root).await.unwrap());
    let config = StreamRuntimeConfig {
        checkpoint_interval: if point == CheckpointFaultPoint::PartialAlignment {
            Duration::from_millis(1)
        } else {
            Duration::from_secs(3_600)
        },
        checkpoint_timeout: Duration::from_secs(5),
        retained_epochs: 2,
        ..StreamRuntimeConfig::default()
    };

    let first_source_opens = Arc::new(Mutex::new(Vec::new()));
    let first_source_closed = Arc::new(AtomicUsize::new(0));
    let first_sink_closed = Arc::new(AtomicUsize::new(0));
    let first_sink_writes = Arc::new(AtomicUsize::new(0));
    let first_cancellation = CancellationToken::new();
    let first_spec = checkpoint_matrix_spec(
        20_000,
        &sink_root,
        &first_source_opens,
        &first_source_closed,
        &first_sink_closed,
        &first_sink_writes,
        if point == CheckpointFaultPoint::PartialAlignment {
            Duration::from_millis(5)
        } else {
            Duration::ZERO
        },
        first_cancellation.clone(),
        true,
    );
    let (first_checkpoint, fault_probe) =
        CheckpointRuntimeSpec::new(backend.clone(), &manifest_root, config)
            .unwrap()
            .with_fault_probe(point, mode);
    let mut first_runner = ContinuousRunner::new();
    let first_job = first_runner
        .start_checkpointed(first_spec, first_checkpoint)
        .await
        .unwrap_or_else(|failure| {
            panic!("fault case {point:?}/{mode:?} failed to start: {failure:?}")
        });
    let first_outcome = tokio::time::timeout(Duration::from_secs(5), first_job.wait())
        .await
        .unwrap_or_else(|error| panic!("fault case {point:?}/{mode:?} hung: {error}"));
    let first_error = format!("{:?}", first_outcome.errors);
    let deterministic_terminal_error = match mode {
        CheckpointFaultMode::Cancel => {
            first_outcome.state == ContinuousJobState::Cancelled
                && first_outcome.errors.is_empty()
                && first_cancellation.is_cancelled()
        }
        CheckpointFaultMode::Io => {
            first_outcome.state == ContinuousJobState::Failed
                && first_error.contains("injected checkpoint I/O fault")
        }
        CheckpointFaultMode::Panic => {
            first_outcome.state == ContinuousJobState::Failed
                && first_error.contains("injected checkpoint panic")
        }
        CheckpointFaultMode::Restart => {
            first_outcome.state == ContinuousJobState::Failed
                && first_error.contains("injected checkpoint restart")
        }
    };
    let manifest_path = manifest_root.join("manifest-00000000000000000001.json");
    let selected_before_restart = if manifest_path.exists() {
        Some(Epoch::INITIAL)
    } else {
        None
    };
    drop(first_job);
    first_runner.shutdown().await.unwrap();
    assert_eq!(first_runner.registry_counts(), (0, 0));
    assert_eq!(first_source_closed.load(Ordering::SeqCst), 2);
    assert_eq!(first_sink_closed.load(Ordering::SeqCst), 2);
    let prepared_artifacts_after_failure = ["sink-a", "sink-b"]
        .into_iter()
        .filter(|sink_id| {
            sink_root
                .join(sink_id)
                .join("prepared-00000000000000000001.json")
                .exists()
        })
        .count();

    let restart_source_opens = Arc::new(Mutex::new(Vec::new()));
    let restart_source_closed = Arc::new(AtomicUsize::new(0));
    let restart_sink_closed = Arc::new(AtomicUsize::new(0));
    let restart_sink_writes = Arc::new(AtomicUsize::new(0));
    let restart_spec = checkpoint_matrix_spec(
        20_001,
        &sink_root,
        &restart_source_opens,
        &restart_source_closed,
        &restart_sink_closed,
        &restart_sink_writes,
        Duration::ZERO,
        CancellationToken::new(),
        true,
    );
    let restart_checkpoint = CheckpointRuntimeSpec::new(
        backend,
        &manifest_root,
        StreamRuntimeConfig {
            checkpoint_interval: Duration::from_secs(3_600),
            ..config
        },
    )
    .unwrap();
    let mut restart_runner = ContinuousRunner::new();
    let restart_job = restart_runner
        .start_checkpointed(restart_spec, restart_checkpoint)
        .await
        .unwrap_or_else(|failure| {
            panic!("restart case {point:?}/{mode:?} failed to start: {failure:?}")
        });
    let restart_outcome = tokio::time::timeout(Duration::from_secs(5), restart_job.wait())
        .await
        .unwrap_or_else(|error| panic!("restart case {point:?}/{mode:?} hung: {error}"));
    assert_eq!(
        restart_outcome.state,
        ContinuousJobState::Completed,
        "restart case {point:?}/{mode:?} failed: {restart_outcome:?}"
    );
    let terminal_status = restart_job.status();
    let terminal_tasks = terminal_status.tasks.len();
    let terminal_charged_edges = terminal_status
        .edges
        .values()
        .filter(|edge| edge.queue_depth != 0 || edge.charged_rows != 0 || edge.charged_bytes != 0)
        .count();
    drop(restart_job);
    restart_runner.shutdown().await.unwrap();
    assert_eq!(restart_runner.registry_counts(), (0, 0));

    let manifest =
        crate::CheckpointManifest::from_bytes(&tokio::fs::read(&manifest_path).await.unwrap())
            .unwrap();
    let restored_cursor_orders = manifest
        .sources()
        .iter()
        .map(|(source_id, entry)| {
            (
                source_id.clone(),
                hex::decode(entry.cursor.as_ref().unwrap().order.as_bytes()).unwrap(),
            )
        })
        .collect::<BTreeMap<_, _>>();
    let sources_ended = manifest.sources().values().all(|entry| entry.ended);
    let watermarks_restored = manifest.sources().values().all(|entry| {
        matches!(
            entry.watermark_policy,
            SourceWatermarkManifestState::SourceProvided {
                last_emitted_micros: Some(watermark),
                idle: false,
            } if watermark == EventTime::from_micros(1)
        )
    });
    let window_state_restored = manifest.operators().get("window").is_some_and(|entry| {
        !entry.segments.is_empty()
            && entry
                .progress
                .values()
                .all(|ingress| ingress.state == ManifestIngressState::Ended)
    });
    let visible = read_checkpoint_matrix_visible(&sink_root).await;
    let observed = visible.values().flatten().cloned().collect::<Vec<_>>();
    let unique = observed.iter().cloned().collect::<BTreeSet<_>>();
    let expected = ["sink-a", "sink-b"]
        .into_iter()
        .flat_map(|sink_id| {
            [
                format!("{sink_id}|0|1|left|1"),
                format!("{sink_id}|0|1|right|10"),
            ]
        })
        .collect::<BTreeSet<_>>();
    let duplicate_records = observed.len().saturating_sub(unique.len());
    let missing_records = expected.difference(&unique).count();

    CheckpointFaultMatrixReport {
        selected_before_restart,
        prepared_artifacts_after_failure,
        restored_epoch: Some(manifest.epoch()),
        restored_cursor_orders,
        sources_ended,
        watermarks_restored,
        window_state_restored,
        visible_records: observed.len(),
        duplicate_records,
        missing_records,
        temporary_artifacts: checkpoint_matrix_temporary_artifacts(
            &state_root,
            &manifest_root,
            &sink_root,
        ),
        terminal_tasks,
        terminal_charged_edges,
        cancellation_requested: fault_probe.cancellation_trigger_count() == 1,
        deterministic_terminal_error,
    }
}

struct CheckpointRestartSoakReport {
    restarts: usize,
    completed_epochs: u64,
    compacted_window_epochs: usize,
    maximum_manifest_count: usize,
    maximum_state_bytes: u64,
    source_records: usize,
    output_records: usize,
    duplicate_records: usize,
    missing_records: usize,
    temporary_artifacts: usize,
    terminal_tasks: usize,
    terminal_charged_edges: usize,
    terminal_registries: (usize, usize),
}

fn checkpoint_restart_soak_metadata(commit: &str, kernel: &str, rustc: &str) -> serde_json::Value {
    json!({
        "schema": "calc-flow.m5-checkpoint-soak.v1",
        "commit": commit,
        "target_duration_seconds": 1_200,
        "sample_count": CHECKPOINT_SOAK_SAMPLE_COUNT,
        "cadence_seconds": CHECKPOINT_SOAK_CADENCE.as_secs(),
        "restart_sample_indices": CHECKPOINT_SOAK_RESTART_SAMPLES,
        "source_count": 2,
        "transactional_sink_count": 2,
        "retained_epochs": 2,
        "warmup_samples": WARMUP_SAMPLES,
        "restart_warmup_samples": CHECKPOINT_RESTART_WARMUP_SAMPLES,
        "rss_comparison_sample_ranges": CHECKPOINT_RSS_COMPARISON_RANGES,
        "rss_slope_sample_range": CHECKPOINT_RSS_COMPARISON_RANGES[1],
        "state_bytes_limit": MAX_CHECKPOINT_SOAK_STATE_BYTES,
        "deterministic_seed": "two-source-final-window-restart-v1",
        "environment": {
            "kernel": kernel,
            "rustc": rustc,
            "allocator": "system",
            "rss_source": "/proc/self/status:VmRSS",
        },
        "command": CHECKPOINT_SOAK_COMMAND,
    })
}

#[allow(
    clippy::too_many_arguments,
    reason = "each soak generation receives the same explicit durable and lifecycle state"
)]
async fn start_checkpoint_restart_generation(
    job_id: u64,
    backend: Arc<LocalStateBackend>,
    manifest_root: &Path,
    sink_root: &Path,
    stop: &Arc<AtomicUsize>,
    opened_with: &SourceOpenHistory,
    source_closed: &Arc<AtomicUsize>,
    sink_closed: &Arc<AtomicUsize>,
    config: StreamRuntimeConfig,
    sink_write_delay: Duration,
) -> (ContinuousRunner, ContinuousJob) {
    let spec = checkpoint_restart_soak_spec(
        job_id,
        sink_root,
        stop,
        opened_with,
        source_closed,
        sink_closed,
        sink_write_delay,
    );
    let checkpoint = CheckpointRuntimeSpec::new(backend, manifest_root, config).unwrap();
    let runner = ContinuousRunner::new();
    let job = runner
        .start_checkpointed(spec, checkpoint)
        .await
        .unwrap_or_else(|failure| panic!("checkpoint soak generation failed: {failure:?}"));
    (runner, job)
}

async fn wait_for_completed_checkpoints(job: &ContinuousJob, expected: u64) {
    let completed = tokio::time::timeout(Duration::from_secs(20), async {
        loop {
            let status = job.status();
            assert_eq!(
                status.state,
                ContinuousJobState::Running,
                "checkpoint soak generation terminated before {expected} checkpoints: {:?}",
                job.wait().await
            );
            if status.metrics.checkpoints.completed >= expected {
                break;
            }
            tokio::time::sleep(Duration::from_millis(1)).await;
        }
    })
    .await;
    assert!(
        completed.is_ok(),
        "checkpoint soak generation did not complete {expected} checkpoints: {:?}",
        job.status()
    );
}

async fn checkpoint_manifest_documents(root: &Path) -> Vec<crate::CheckpointManifest> {
    let mut manifests = Vec::new();
    let mut entries = tokio::fs::read_dir(root).await.unwrap();
    while let Some(entry) = entries.next_entry().await.unwrap() {
        let name = entry.file_name();
        if name.to_string_lossy().starts_with("manifest-") {
            if let Some(manifest) = read_checkpoint_manifest_document(&entry.path()).await {
                manifests.push(manifest);
            }
        }
    }
    manifests.sort_by_key(crate::CheckpointManifest::epoch);
    manifests
}

async fn read_checkpoint_manifest_document(path: &Path) -> Option<crate::CheckpointManifest> {
    let bytes = match tokio::fs::read(path).await {
        Ok(bytes) => bytes,
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => return None,
        Err(error) => panic!("checkpoint soak could not read manifest: {error}"),
    };
    Some(crate::CheckpointManifest::from_bytes(&bytes).unwrap())
}

fn manifest_has_compacted_window(manifest: &crate::CheckpointManifest) -> bool {
    manifest
        .operators()
        .get("window")
        .and_then(|entry| entry.inline_metadata.get("segment_inventory"))
        .and_then(serde_json::Value::as_array)
        .is_some_and(|inventory| {
            inventory
                .iter()
                .any(|descriptor| descriptor["kind"] == "base")
        })
}

async fn directory_regular_file_bytes(root: &Path) -> u64 {
    let mut total = 0_u64;
    let mut directories = vec![root.to_path_buf()];
    while let Some(directory) = directories.pop() {
        let mut entries = match tokio::fs::read_dir(&directory).await {
            Ok(entries) => entries,
            Err(error) if error.kind() == std::io::ErrorKind::NotFound => continue,
            Err(error) => panic!("checkpoint soak could not inspect state directory: {error}"),
        };
        while let Some(entry) = entries.next_entry().await.unwrap() {
            let metadata = match entry.metadata().await {
                Ok(metadata) => metadata,
                Err(error) if error.kind() == std::io::ErrorKind::NotFound => continue,
                Err(error) => panic!("checkpoint soak could not inspect state entry: {error}"),
            };
            if metadata.is_dir() {
                directories.push(entry.path());
            } else if metadata.is_file() {
                total = total
                    .checked_add(metadata.len())
                    .expect("checkpoint soak state byte count overflowed");
            }
        }
    }
    total
}

fn checkpoint_soak_expected_records(manifest: &crate::CheckpointManifest) -> BTreeSet<String> {
    let cursors = manifest
        .sources()
        .iter()
        .map(|(source_id, entry)| {
            let order = hex::decode(entry.cursor.as_ref().unwrap().order.as_bytes()).unwrap();
            let bytes: [u8; 8] = order.try_into().unwrap();
            (source_id.as_str(), u64::from_be_bytes(bytes))
        })
        .collect::<Vec<_>>();
    ["sink-a", "sink-b"]
        .into_iter()
        .flat_map(|sink_id| {
            cursors.iter().flat_map(move |(source_id, last)| {
                (0..=*last).map(move |sequence| {
                    format!(
                        "{sink_id}|{sequence}|{}|{source_id}|{}",
                        sequence + 1,
                        sequence + 1
                    )
                })
            })
        })
        .collect()
}

#[allow(
    clippy::too_many_arguments,
    reason = "the final oracle receives explicit durable roots and observed resource maxima"
)]
async fn finish_checkpoint_restart_soak(
    mut runner: ContinuousRunner,
    job: ContinuousJob,
    stop: &AtomicUsize,
    state_root: &Path,
    manifest_root: &Path,
    sink_root: &Path,
    maximum_manifest_count: usize,
    maximum_state_bytes: u64,
    compacted_epochs: &BTreeSet<Epoch>,
) -> CheckpointRestartSoakReport {
    stop.store(1, Ordering::SeqCst);
    let outcome = tokio::time::timeout(Duration::from_secs(10), job.wait())
        .await
        .expect("checkpoint soak terminal epoch timed out");
    assert_eq!(outcome.state, ContinuousJobState::Completed);
    assert_eq!(outcome.cause, TerminalCause::NaturalEnd);
    let terminal_status = job.status();
    let terminal_tasks = terminal_status.tasks.len();
    let terminal_charged_edges = terminal_status
        .edges
        .values()
        .filter(|edge| edge.queue_depth != 0 || edge.charged_rows != 0 || edge.charged_bytes != 0)
        .count();
    drop(job);
    runner.shutdown().await.unwrap();
    let terminal_registries = runner.registry_counts();
    let manifests = checkpoint_manifest_documents(manifest_root).await;
    let terminal = manifests.last().unwrap();
    let visible = read_checkpoint_matrix_visible(sink_root).await;
    let observed = visible.values().flatten().cloned().collect::<Vec<_>>();
    let unique = observed.iter().cloned().collect::<BTreeSet<_>>();
    let expected = checkpoint_soak_expected_records(terminal);
    let source_records = expected.len() / 2;
    CheckpointRestartSoakReport {
        restarts: 2,
        completed_epochs: terminal.epoch().as_u64(),
        compacted_window_epochs: compacted_epochs.len(),
        maximum_manifest_count: maximum_manifest_count.max(manifests.len()),
        maximum_state_bytes,
        source_records,
        output_records: observed.len(),
        duplicate_records: observed.len().saturating_sub(unique.len()),
        missing_records: expected.difference(&unique).count(),
        temporary_artifacts: checkpoint_matrix_temporary_artifacts(
            state_root,
            manifest_root,
            sink_root,
        ),
        terminal_tasks,
        terminal_charged_edges,
        terminal_registries,
    }
}

async fn cancel_checkpoint_restart_generation(mut runner: ContinuousRunner, job: ContinuousJob) {
    let outcome = job.cancel().await;
    assert_eq!(outcome.state, ContinuousJobState::Cancelled);
    assert!(job.status().tasks.is_empty());
    drop(job);
    runner.shutdown().await.unwrap();
    assert_eq!(runner.registry_counts(), (0, 0));
}

async fn run_checkpoint_restart_soak_smoke() -> CheckpointRestartSoakReport {
    let directory = tempfile::tempdir().unwrap();
    let state_root = directory.path().join("state");
    let manifest_root = directory.path().join("manifests");
    let sink_root = directory.path().join("sinks");
    let backend = Arc::new(LocalStateBackend::new(&state_root).await.unwrap());
    let stop = Arc::new(AtomicUsize::new(0));
    let opened_with = Arc::new(Mutex::new(Vec::new()));
    let source_closed = Arc::new(AtomicUsize::new(0));
    let sink_closed = Arc::new(AtomicUsize::new(0));
    let config = StreamRuntimeConfig {
        checkpoint_interval: Duration::from_millis(250),
        checkpoint_timeout: Duration::from_secs(5),
        retained_epochs: 2,
        ..StreamRuntimeConfig::default()
    };
    let mut maximum_manifest_count = 0;
    let mut maximum_state_bytes = 0;
    let mut compacted_epochs = BTreeSet::new();

    for generation in 0..2 {
        let (runner, job) = start_checkpoint_restart_generation(
            30_000 + generation,
            Arc::clone(&backend),
            &manifest_root,
            &sink_root,
            &stop,
            &opened_with,
            &source_closed,
            &sink_closed,
            config,
            Duration::ZERO,
        )
        .await;
        wait_for_completed_checkpoints(&job, 12).await;
        let manifests = checkpoint_manifest_documents(&manifest_root).await;
        maximum_manifest_count = maximum_manifest_count.max(manifests.len());
        maximum_state_bytes =
            maximum_state_bytes.max(directory_regular_file_bytes(&state_root).await);
        compacted_epochs.extend(
            manifests
                .iter()
                .filter(|manifest| manifest_has_compacted_window(manifest))
                .map(crate::CheckpointManifest::epoch),
        );
        cancel_checkpoint_restart_generation(runner, job).await;
    }

    let (runner, job) = start_checkpoint_restart_generation(
        30_002,
        backend,
        &manifest_root,
        &sink_root,
        &stop,
        &opened_with,
        &source_closed,
        &sink_closed,
        config,
        Duration::ZERO,
    )
    .await;
    wait_for_completed_checkpoints(&job, 12).await;
    let manifests = checkpoint_manifest_documents(&manifest_root).await;
    maximum_manifest_count = maximum_manifest_count.max(manifests.len());
    maximum_state_bytes = maximum_state_bytes.max(directory_regular_file_bytes(&state_root).await);
    compacted_epochs.extend(
        manifests
            .iter()
            .filter(|manifest| manifest_has_compacted_window(manifest))
            .map(crate::CheckpointManifest::epoch),
    );
    let report = finish_checkpoint_restart_soak(
        runner,
        job,
        &stop,
        &state_root,
        &manifest_root,
        &sink_root,
        maximum_manifest_count,
        maximum_state_bytes,
        &compacted_epochs,
    )
    .await;
    let opens = opened_with.lock();
    assert_eq!(opens.len(), 6);
    assert!(opens[..2].iter().all(|(_, cursor)| cursor.is_none()));
    assert!(opens[2..].iter().all(|(_, cursor)| cursor.is_some()));
    assert_eq!(source_closed.load(Ordering::SeqCst), 6);
    assert_eq!(sink_closed.load(Ordering::SeqCst), 6);
    report
}

#[cfg(target_os = "linux")]
#[allow(
    clippy::too_many_lines,
    reason = "the ignored checkpoint soak keeps sampling, restarts, and its final oracle together"
)]
async fn run_checkpoint_restart_linux_soak() {
    let directory = tempfile::tempdir().unwrap();
    let state_root = directory.path().join("state");
    let manifest_root = directory.path().join("manifests");
    let sink_root = directory.path().join("sinks");
    let backend = Arc::new(LocalStateBackend::new(&state_root).await.unwrap());
    let stop = Arc::new(AtomicUsize::new(0));
    let opened_with = Arc::new(Mutex::new(Vec::new()));
    let source_closed = Arc::new(AtomicUsize::new(0));
    let sink_closed = Arc::new(AtomicUsize::new(0));
    let config = StreamRuntimeConfig {
        checkpoint_interval: Duration::from_secs(5),
        checkpoint_timeout: Duration::from_secs(10),
        retained_epochs: 2,
        ..StreamRuntimeConfig::default()
    };
    let commit = command_output("git", &["rev-parse", "HEAD"]);
    println!(
        "{}",
        checkpoint_restart_soak_metadata(
            &commit,
            &command_output("uname", &["-sr"]),
            &command_output("rustc", &["--version"]),
        )
    );
    let (mut runner, mut job) = start_checkpoint_restart_generation(
        40_000,
        Arc::clone(&backend),
        &manifest_root,
        &sink_root,
        &stop,
        &opened_with,
        &source_closed,
        &sink_closed,
        config,
        Duration::from_millis(20),
    )
    .await;
    let initial_status = job.status();
    let steady_task_count = initial_status.tasks.len();
    assert!(
        steady_task_count > 0,
        "checkpoint soak task registry is empty"
    );
    assert_edge_budgets(&initial_status);
    let started = tokio::time::Instant::now();
    let mut rss_samples = Vec::with_capacity(CHECKPOINT_SOAK_SAMPLE_COUNT);
    let mut maximum_manifest_count = 0;
    let mut maximum_state_bytes = 0;
    let mut maximum_task_count = steady_task_count;
    let mut maximum_queue_depth = 0;
    let mut maximum_charged_rows = 0;
    let mut maximum_charged_bytes = 0;
    let mut compacted_epochs = BTreeSet::new();
    for index in 0..CHECKPOINT_SOAK_SAMPLE_COUNT {
        let elapsed_seconds = wait_for_sample_deadline(started, index).await;
        let process_status = tokio::fs::read_to_string("/proc/self/status")
            .await
            .expect("Linux checkpoint soak could not read /proc/self/status");
        let rss_kib = parse_vm_rss_kib(&process_status).expect("Linux status omitted VmRSS");
        rss_samples.push(RssSample {
            elapsed_seconds,
            rss_kib,
        });
        let manifests = checkpoint_manifest_documents(&manifest_root).await;
        assert!(
            manifests.len() <= 3,
            "checkpoint soak exceeded retained plus in-flight manifest bound"
        );
        maximum_manifest_count = maximum_manifest_count.max(manifests.len());
        let state_bytes = directory_regular_file_bytes(&state_root).await;
        assert!(
            state_bytes <= MAX_CHECKPOINT_SOAK_STATE_BYTES,
            "checkpoint soak exceeded its state byte bound"
        );
        maximum_state_bytes = maximum_state_bytes.max(state_bytes);
        compacted_epochs.extend(
            manifests
                .iter()
                .filter(|manifest| manifest_has_compacted_window(manifest))
                .map(crate::CheckpointManifest::epoch),
        );
        let status = job.status();
        assert_eq!(status.state, ContinuousJobState::Running);
        assert_eq!(status.tasks.len(), steady_task_count);
        assert_eq!(status.metrics.checkpoints.failed, 0);
        assert_edge_budgets(&status);
        maximum_task_count = maximum_task_count.max(status.tasks.len());
        maximum_queue_depth = maximum_queue_depth.max(
            status
                .edges
                .values()
                .map(|edge| edge.queue_depth)
                .max()
                .unwrap_or(0),
        );
        maximum_charged_rows = maximum_charged_rows.max(
            status
                .edges
                .values()
                .map(|edge| edge.charged_rows)
                .max()
                .unwrap_or(0),
        );
        maximum_charged_bytes = maximum_charged_bytes.max(
            status
                .edges
                .values()
                .map(|edge| edge.charged_bytes)
                .max()
                .unwrap_or(0),
        );
        let queues = status
            .edges
            .iter()
            .map(|(edge_id, edge)| {
                json!({
                    "edge_id": edge_id,
                    "queue_depth": edge.queue_depth,
                    "charged_rows": edge.charged_rows,
                    "charged_bytes": edge.charged_bytes,
                    "high_water_depth": edge.high_water_depth,
                })
            })
            .collect::<Vec<_>>();
        println!(
            "{}",
            json!({
                "type": "calc_flow_m5_checkpoint_soak_sample",
                "index": index,
                "elapsed_seconds": elapsed_seconds,
                "vmrss_kib": rss_kib,
                "task_count": status.tasks.len(),
                "queues": queues,
                "checkpoint": {
                    "last_completed_epoch": status.checkpoint.as_ref()
                        .and_then(|checkpoint| checkpoint.last_completed_epoch)
                        .map(Epoch::as_u64),
                    "completed": status.metrics.checkpoints.completed,
                    "failed": status.metrics.checkpoints.failed,
                },
                "manifest_count": manifests.len(),
                "state_bytes": state_bytes,
            })
        );
        if CHECKPOINT_SOAK_RESTART_SAMPLES.contains(&index) {
            let outcome = job.cancel().await;
            assert_eq!(outcome.state, ContinuousJobState::Cancelled);
            drop(job);
            runner.shutdown().await.unwrap();
            assert_eq!(runner.registry_counts(), (0, 0));
            let generation = CHECKPOINT_SOAK_RESTART_SAMPLES
                .iter()
                .position(|sample| *sample == index)
                .unwrap()
                + 1;
            (runner, job) = start_checkpoint_restart_generation(
                40_000 + u64::try_from(generation).unwrap(),
                Arc::clone(&backend),
                &manifest_root,
                &sink_root,
                &stop,
                &opened_with,
                &source_closed,
                &sink_closed,
                config,
                Duration::from_millis(20),
            )
            .await;
            let restarted_status = job.status();
            assert_eq!(restarted_status.tasks.len(), steady_task_count);
            assert_edge_budgets(&restarted_status);
        }
    }
    let report = finish_checkpoint_restart_soak(
        runner,
        job,
        &stop,
        &state_root,
        &manifest_root,
        &sink_root,
        maximum_manifest_count,
        maximum_state_bytes,
        &compacted_epochs,
    )
    .await;
    assert_eq!(report.restarts, 2);
    assert!(report.compacted_window_epochs > 0);
    assert!(report.maximum_manifest_count <= 3);
    assert_eq!(report.duplicate_records, 0);
    assert_eq!(report.missing_records, 0);
    assert_eq!(report.temporary_artifacts, 0);
    assert_eq!(report.terminal_tasks, 0);
    assert_eq!(report.terminal_charged_edges, 0);
    assert_eq!(report.terminal_registries, (0, 0));
    assert_eq!(observed_timeline_issue(&rss_samples), None);
    let rss_gate = evaluate_checkpoint_restart_rss_gate(&rss_samples)
        .expect("checkpoint soak RSS samples incomplete");
    assert!(
        rss_gate.passed,
        "checkpoint soak RSS guard failed: {rss_gate:?}"
    );
    println!(
        "{}",
        json!({
            "schema": "calc-flow.m5-checkpoint-soak.v1",
            "type": "calc_flow_m5_checkpoint_soak_summary",
            "commit": commit,
            "target_duration_seconds": 1_200,
            "sample_count": rss_samples.len(),
            "restart_sample_indices": CHECKPOINT_SOAK_RESTART_SAMPLES,
            "restarts": report.restarts,
            "completed_epochs": report.completed_epochs,
            "compacted_window_epochs": report.compacted_window_epochs,
            "resource_bounds": {
                "maximum_task_count": maximum_task_count,
                "maximum_queue_depth": maximum_queue_depth,
                "maximum_charged_rows": maximum_charged_rows,
                "maximum_charged_bytes": maximum_charged_bytes,
                "maximum_manifest_count": report.maximum_manifest_count,
                "maximum_state_bytes": report.maximum_state_bytes,
                "state_bytes_limit": MAX_CHECKPOINT_SOAK_STATE_BYTES,
            },
            "conservation": {
                "source_records": report.source_records,
                "output_records": report.output_records,
                "duplicate_records": report.duplicate_records,
                "missing_records": report.missing_records,
            },
            "rss": {
                "slope_mib_per_hour": rss_gate.slope_mib_per_hour,
                "comparison_sample_ranges": CHECKPOINT_RSS_COMPARISON_RANGES,
                "slope_sample_range": CHECKPOINT_RSS_COMPARISON_RANGES[1],
                "first_aligned_five_minute_median_kib": rss_gate.first_median_kib,
                "final_aligned_five_minute_median_kib": rss_gate.final_median_kib,
                "passed": rss_gate.passed,
            },
            "temporary_artifacts": report.temporary_artifacts,
            "terminal_tasks": report.terminal_tasks,
            "terminal_charged_edges": report.terminal_charged_edges,
            "terminal_registries": report.terminal_registries,
            "source_open_events": opened_with.lock().len(),
            "source_close_events": source_closed.load(Ordering::SeqCst),
            "sink_close_events": sink_closed.load(Ordering::SeqCst),
        })
    );
}

#[cfg(not(target_os = "linux"))]
async fn run_checkpoint_restart_linux_soak() {
    panic!("checkpoint restart soak evidence requires Linux /proc")
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
    assert_eq!(
        outcome.state,
        ContinuousJobState::Completed,
        "graceful soak smoke failed: {outcome:?}"
    );
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
    assert!(
        state.max_retained_segments > 0,
        "soak observed no retained window inventory"
    );
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

    assert_eq!(
        outcome.state,
        ContinuousJobState::Completed,
        "graceful soak smoke failed: {outcome:?}"
    );
    assert_eq!(outcome.cause, TerminalCause::GracefulShutdown);
    let accepted = accepted.snapshot();
    assert_delivery_conservation(&accepted, &sink_a.lock(), "sink A");
    assert_delivery_conservation(&accepted, &sink_b.lock(), "sink B");
    let probe = window_probe.lock().clone();
    assert!(probe.checkpoints > 0);
    assert!(probe.total_segment_bytes > 0);
    assert!(probe.max_live_keys > 0);
    assert!(probe.max_retained_segments > 0);
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
fn checkpoint_restart_rss_gate_compares_equivalent_lifecycle_phases() {
    let lifecycle_rss = |index: usize, growing: bool| match index {
        40..50 | 80..90 => 90_000,
        50..80 => 120_000,
        90..120 => {
            let growth = if growing {
                u64::try_from(index - 90).unwrap().saturating_mul(1_000)
            } else {
                0
            };
            120_000 + growth
        }
        _ => 100_000,
    };
    let stable = (0..SAMPLE_COUNT)
        .map(|index| RssSample {
            elapsed_seconds: elapsed_at_sample(index),
            rss_kib: lifecycle_rss(index, false),
        })
        .collect::<Vec<_>>();
    let growing = (0..SAMPLE_COUNT)
        .map(|index| RssSample {
            elapsed_seconds: elapsed_at_sample(index),
            rss_kib: lifecycle_rss(index, true),
        })
        .collect::<Vec<_>>();

    assert!(!evaluate_rss_gate(&stable).unwrap().passed);
    assert!(
        evaluate_checkpoint_restart_rss_gate(&stable)
            .unwrap()
            .passed
    );
    assert!(
        !evaluate_checkpoint_restart_rss_gate(&growing)
            .unwrap()
            .passed
    );
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
    assert_eq!(metadata["state_contract"]["window_size_micros"], 1);
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

#[tokio::test]
async fn checkpoint_restart_soak_smoke_exercises_retention_and_compaction() {
    let report = run_checkpoint_restart_soak_smoke().await;

    assert_eq!(report.restarts, 2);
    assert!(report.completed_epochs >= 12);
    assert!(report.compacted_window_epochs > 0);
    assert!(report.maximum_manifest_count <= 3);
    assert!(report.maximum_state_bytes > 0);
    assert_eq!(report.duplicate_records, 0);
    assert_eq!(report.missing_records, 0);
    assert_eq!(report.terminal_tasks, 0);
    assert_eq!(report.terminal_charged_edges, 0);
    assert_eq!(report.terminal_registries, (0, 0));
    assert_eq!(report.temporary_artifacts, 0);
}

#[test]
fn checkpoint_restart_soak_contract_is_exact_and_machine_readable() {
    let metadata = checkpoint_restart_soak_metadata("abc123", "Linux 1", "rustc 1.88");

    assert_eq!(metadata["schema"], "calc-flow.m5-checkpoint-soak.v1");
    assert_eq!(metadata["target_duration_seconds"], 1_200);
    assert_eq!(metadata["sample_count"], 120);
    assert_eq!(metadata["cadence_seconds"], 10);
    assert_eq!(metadata["restart_sample_indices"], json!([39, 79]));
    assert_eq!(metadata["source_count"], 2);
    assert_eq!(metadata["transactional_sink_count"], 2);
    assert_eq!(metadata["retained_epochs"], 2);
    assert_eq!(metadata["warmup_samples"], 30);
    assert_eq!(metadata["restart_warmup_samples"], 10);
    assert_eq!(
        metadata["rss_comparison_sample_ranges"],
        json!([[50, 79], [90, 119]])
    );
    assert_eq!(metadata["rss_slope_sample_range"], json!([90, 119]));
    assert_eq!(metadata["state_bytes_limit"], 64 * 1_024 * 1_024);
    assert_eq!(
        metadata["environment"]["rss_source"],
        "/proc/self/status:VmRSS"
    );
    assert_eq!(metadata["commit"], "abc123");
    assert_eq!(
        metadata["command"],
        "CALC_FLOW_M5_CHECKPOINT_SOAK=1 cargo test -p calc-flow --lib runtime::streaming::soak::twenty_minute_epoch_checkpoint_restart -- --ignored --exact --nocapture"
    );
}

#[tokio::test]
async fn checkpoint_manifest_scan_skips_entry_removed_after_enumeration() {
    let directory = tempfile::tempdir().unwrap();
    let path = directory.path().join("manifest-race.json");
    tokio::fs::write(&path, b"enumerated before retention")
        .await
        .unwrap();
    let mut entries = tokio::fs::read_dir(directory.path()).await.unwrap();
    let entry = entries.next_entry().await.unwrap().unwrap();
    tokio::fs::remove_file(entry.path()).await.unwrap();

    assert!(
        read_checkpoint_manifest_document(&entry.path())
            .await
            .is_none()
    );
}

#[tokio::test]
async fn private_full_path_benchmark_smoke_runs_baseline_and_restart_cases() {
    let private_paths = run_private_checkpoint_path_benchmark_smoke().await;
    assert_eq!(
        private_paths,
        BTreeMap::from([
            ("m5/private_path/barrier_cut_fan_out", 4),
            ("m5/private_path/two_input_alignment", 2),
            ("m5/private_path/dirty_window_state_stage", 1),
            ("m5/private_path/production_manifest_publication", 1),
            ("m5/private_path/cold_restore", 1),
            ("m5/private_path/transactional_sink_commit", 1),
        ])
    );

    let baseline = run_private_no_checkpoint_benchmark_case().await;
    assert_eq!(baseline.completed_jobs, 1);
    assert_eq!(baseline.visible_records, 4);
    assert_eq!(baseline.manifest_count, 0);

    let checkpoint = run_private_checkpoint_restart_benchmark_case().await;
    assert_eq!(checkpoint.completed_jobs, 1);
    assert_eq!(checkpoint.failed_jobs, 1);
    assert_eq!(checkpoint.visible_records, 4);
    assert!(checkpoint.manifest_count >= 2);
    assert!(checkpoint.restored_epoch.is_some());
}

#[test]
fn private_benchmark_errors_name_the_candidate_only_gate() {
    let CalcFlowError::InvalidArgument { field, .. } = benchmark_evidence_error("invalid") else {
        panic!("benchmark evidence error must be an invalid argument");
    };

    assert_eq!(field, "CALC_FLOW_M5_CHECKPOINT_BENCHMARK");
}

#[test]
fn private_benchmark_rejects_missing_or_stale_embedded_source_identity() {
    let candidate = PrivateBenchmarkProvenance {
        commit: "bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb".into(),
        tree: "2222222222222222222222222222222222222222".into(),
        clean: true,
        harness_hash: "1".repeat(64),
        config_hash: "2".repeat(64),
        executable: PathBuf::from("candidate-executable"),
        executable_sha256: "3".repeat(64),
        build_identity_hash: "4".repeat(64),
        toolchain_hash: "5".repeat(64),
        environment_hash: "6".repeat(64),
    };

    validate_private_benchmark_source_identity(
        &candidate,
        Some(&candidate.commit),
        Some(&candidate.tree),
    )
    .unwrap();
    assert!(
        validate_private_benchmark_source_identity(&candidate, None, Some(&candidate.tree))
            .is_err()
    );
    assert!(
        validate_private_benchmark_source_identity(
            &candidate,
            Some("aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa"),
            Some(&candidate.tree),
        )
        .is_err()
    );
}

#[test]
fn private_benchmark_metadata_reports_seven_absolute_only_m5_cases() {
    let candidate = PrivateBenchmarkProvenance {
        commit: "bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb".into(),
        tree: "2222222222222222222222222222222222222222".into(),
        clean: true,
        harness_hash: "harness-v2".into(),
        config_hash: "config-v2".into(),
        executable: PathBuf::from("candidate-executable"),
        executable_sha256: "candidate-executable".into(),
        build_identity_hash: "candidate-build".into(),
        toolchain_hash: "toolchain".into(),
        environment_hash: "environment".into(),
    };
    let measurements = M5_PRIVATE_BENCHMARK_CASES
        .into_iter()
        .map(|case| json!({"case": case, "decision": "absolute_only"}))
        .collect::<Vec<_>>();
    let metadata = private_checkpoint_benchmark_metadata(
        &candidate,
        Path::new("target/m5-checkpoint-criterion"),
        &measurements,
    );

    assert_eq!(
        metadata["schema"],
        "calc-flow.m5-checkpoint-absolute-benchmark.v1"
    );
    assert_eq!(metadata["commit"], candidate.commit);
    assert_eq!(
        metadata["absolute_cases"],
        json!([
            "m5/private_path/barrier_cut_fan_out",
            "m5/private_path/two_input_alignment",
            "m5/private_path/dirty_window_state_stage",
            "m5/private_path/production_manifest_publication",
            "m5/private_path/cold_restore",
            "m5/private_path/transactional_sink_commit",
            "m5/private_full_path/periodic_checkpoint_restart",
        ])
    );
    assert_eq!(metadata["comparison"], "none");
    assert_eq!(metadata["topology"]["sources"], 2);
    assert_eq!(metadata["topology"]["sinks"], 2);
    assert_eq!(metadata["command"], CHECKPOINT_BENCHMARK_COMMAND);
    assert!(
        metadata["command"]
            .as_str()
            .unwrap()
            .contains("CALC_FLOW_M5_PRIVATE_SOURCE_COMMIT=<candidate-commit>")
    );
    assert!(
        metadata["command"]
            .as_str()
            .unwrap()
            .contains("CALC_FLOW_M5_PRIVATE_SOURCE_TREE=<candidate-tree>")
    );
    assert_eq!(metadata["provenance"]["tree"], candidate.tree);
    assert_eq!(metadata["measurements"].as_array().unwrap().len(), 7);
    assert_eq!(metadata["overall_result"], "absolute_only");
}

#[test]
fn private_absolute_measurement_recomputes_raw_statistics_and_hashes_inputs() {
    let directory = tempfile::tempdir().unwrap();
    let target_root = directory.path();
    let run_root = target_root.join("candidate-run");
    let case_directory = "m5_private_path_barrier_cut_fan_out";
    let candidate_case = run_root.join(case_directory).join("candidate-sha");
    std::fs::create_dir_all(&candidate_case).unwrap();
    std::fs::write(
        candidate_case.join("estimates.json"),
        serde_json::to_vec(&json!({
            "median": {
                "point_estimate": 104.5,
                "confidence_interval": {
                    "confidence_level": 0.95,
                    "lower_bound": 101.0,
                    "upper_bound": 108.0,
                },
            },
        }))
        .unwrap(),
    )
    .unwrap();
    let sample_bytes = serde_json::to_vec(&json!({
        "sampling_mode": "Flat",
        "iters": vec![1.0; 10],
        "times": (100..110).map(f64::from).collect::<Vec<_>>(),
    }))
    .unwrap();
    std::fs::write(candidate_case.join("sample.json"), &sample_bytes).unwrap();

    let measurement = private_benchmark_absolute_measurement(
        "m5/private_path/barrier_cut_fan_out",
        "candidate-sha",
        &run_root,
        target_root,
    )
    .unwrap();

    assert_eq!(measurement["median_ns"], 104.5);
    assert_eq!(measurement["sample_count"], 10);
    assert_eq!(measurement["confidence_level"], 0.95);
    let confidence = measurement["median_confidence_interval_ns"]
        .as_array()
        .unwrap();
    assert!(confidence[0].as_f64().unwrap() <= 104.5);
    assert!(confidence[1].as_f64().unwrap() >= 104.5);
    assert_eq!(measurement["decision"], "absolute_only");
    assert!(
        measurement["artifacts"]["sample"]["path"]
            .as_str()
            .unwrap()
            .ends_with("candidate-sha/sample.json")
    );
    assert_eq!(
        measurement["artifacts"]["sample"]["sha256"],
        sha256_bytes(&sample_bytes)
    );
    assert!(
        measurement["artifacts"]["estimates"]["sha256"]
            .as_str()
            .is_some_and(|digest| digest.len() == 64)
    );
    validate_private_benchmark_artifacts(&measurement, target_root).unwrap();

    std::fs::write(candidate_case.join("estimates.json"), b"{}").unwrap();
    assert!(validate_private_benchmark_artifacts(&measurement, target_root).is_err());
}

#[test]
fn private_benchmark_report_path_is_validated_and_immutable() {
    let directory = tempfile::tempdir().unwrap();
    let metadata = json!({"schema": "calc-flow.m5-checkpoint-absolute-benchmark.v1"});
    assert!(
        write_private_checkpoint_benchmark_metadata(
            directory.path(),
            "../not-a-commit",
            "run-1",
            &metadata,
        )
        .is_err()
    );

    let commit = "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa";
    let path =
        write_private_checkpoint_benchmark_metadata(directory.path(), commit, "run-1", &metadata)
            .unwrap();
    let original = std::fs::read(&path).unwrap();
    assert!(
        write_private_checkpoint_benchmark_metadata(directory.path(), commit, "run-1", &json!({}),)
            .is_err()
    );
    let repeated =
        write_private_checkpoint_benchmark_metadata(directory.path(), commit, "run-2", &metadata)
            .unwrap();
    assert_ne!(path, repeated);
    assert_eq!(std::fs::read(&path).unwrap(), original);
    validate_private_checkpoint_benchmark_report(&path).unwrap();

    std::fs::write(&path, b"{}").unwrap();
    assert!(validate_private_checkpoint_benchmark_report(&path).is_err());
}

#[test]
#[ignore = "opt-in private M5 absolute benchmark; set CALC_FLOW_M5_CHECKPOINT_BENCHMARK=1"]
#[allow(
    clippy::too_many_lines,
    reason = "the opt-in benchmark keeps one immutable candidate-only evidence transaction"
)]
fn private_m5_epoch_checkpoint_absolute_benchmark() {
    assert_eq!(
        std::env::var("CALC_FLOW_M5_CHECKPOINT_BENCHMARK").as_deref(),
        Ok("1"),
        "set CALC_FLOW_M5_CHECKPOINT_BENCHMARK=1 for candidate-only absolute evidence",
    );
    let run_id = std::env::var("CALC_FLOW_M5_CHECKPOINT_BENCHMARK_RUN_ID")
        .expect("set a unique CALC_FLOW_M5_CHECKPOINT_BENCHMARK_RUN_ID");
    assert!(
        !run_id.is_empty()
            && run_id.len() <= 64
            && run_id
                .bytes()
                .all(|byte| byte.is_ascii_alphanumeric() || matches!(byte, b'-' | b'_')),
        "private benchmark run ID must contain only ASCII letters, digits, '-' or '_'"
    );
    let target_root = private_checkpoint_benchmark_target_root();
    std::fs::create_dir_all(&target_root).unwrap();
    let provenance = current_private_benchmark_provenance().unwrap();
    assert!(provenance.clean, "benchmark evidence requires a clean tree");
    validate_private_benchmark_source_identity(
        &provenance,
        PRIVATE_BENCHMARK_SOURCE_COMMIT,
        PRIVATE_BENCHMARK_SOURCE_TREE,
    )
    .unwrap();
    let criterion_root = target_root
        .join("m5-checkpoint-criterion")
        .join("runs")
        .join(format!(
            "{}-{}-{run_id}",
            provenance.commit,
            &provenance.build_identity_hash[..16]
        ));
    assert!(
        !criterion_root.exists(),
        "private benchmark run path already exists; evidence is immutable"
    );
    let mut criterion = Criterion::default()
        .sample_size(CHECKPOINT_BENCHMARK_SAMPLE_SIZE)
        .warm_up_time(CHECKPOINT_BENCHMARK_WARM_UP)
        .measurement_time(CHECKPOINT_BENCHMARK_MEASUREMENT)
        .output_directory(&criterion_root)
        .without_plots()
        .save_baseline(provenance.commit.clone());
    let runtime = tokio::runtime::Runtime::new().unwrap();

    criterion.bench_function(M5_PRIVATE_BENCHMARK_CASES[0], |bencher| {
        bencher
            .iter(|| black_box(runtime.block_on(run_private_barrier_cut_fan_out_benchmark_case())));
    });
    criterion.bench_function(M5_PRIVATE_BENCHMARK_CASES[1], |bencher| {
        bencher.iter(|| black_box(super::operator_task::benchmark_two_input_alignment().unwrap()));
    });
    criterion.bench_function(M5_PRIVATE_BENCHMARK_CASES[2], |bencher| {
        bencher.iter(|| {
            black_box(runtime.block_on(run_private_dirty_window_state_stage_benchmark_case()))
        });
    });
    criterion.bench_function(M5_PRIVATE_BENCHMARK_CASES[3], |bencher| {
        bencher.iter(|| {
            black_box(runtime.block_on(run_private_manifest_publication_benchmark_case()))
        });
    });
    criterion.bench_function(M5_PRIVATE_BENCHMARK_CASES[4], |bencher| {
        bencher.iter(|| black_box(runtime.block_on(run_private_cold_restore_benchmark_case())));
    });
    criterion.bench_function(M5_PRIVATE_BENCHMARK_CASES[5], |bencher| {
        bencher.iter(|| {
            black_box(runtime.block_on(run_private_transactional_sink_commit_benchmark_case()))
        });
    });
    criterion.bench_function(
        "m5/private_full_path/periodic_checkpoint_restart",
        |bencher| {
            bencher.iter(|| {
                black_box(runtime.block_on(run_private_checkpoint_restart_benchmark_case()))
            });
        },
    );
    criterion.final_summary();

    let measurements = M5_PRIVATE_BENCHMARK_CASES
        .into_iter()
        .map(|case| {
            let measurement = private_benchmark_absolute_measurement(
                case,
                &provenance.commit,
                &criterion_root,
                &target_root,
            )?;
            validate_private_benchmark_artifacts(&measurement, &target_root)?;
            Ok(measurement)
        })
        .collect::<Result<Vec<_>>>()
        .unwrap();
    let criterion_reference = criterion_root.strip_prefix(&target_root).unwrap();
    let metadata =
        private_checkpoint_benchmark_metadata(&provenance, criterion_reference, &measurements);
    let report_path = write_private_checkpoint_benchmark_metadata(
        &target_root,
        &provenance.commit,
        &run_id,
        &metadata,
    )
    .unwrap();
    println!("{}", report_path.display());
    println!("{metadata}");
}

#[cfg_attr(
    not(target_os = "linux"),
    ignore = "checkpoint restart soak evidence requires Linux /proc"
)]
#[tokio::test]
#[ignore = "twenty-minute opt-in M5 checkpoint restart soak; set CALC_FLOW_M5_CHECKPOINT_SOAK=1"]
async fn twenty_minute_epoch_checkpoint_restart() {
    assert_eq!(
        std::env::var("CALC_FLOW_M5_CHECKPOINT_SOAK").as_deref(),
        Ok("1"),
        "set CALC_FLOW_M5_CHECKPOINT_SOAK=1 to run the exact M5 soak"
    );
    run_checkpoint_restart_linux_soak().await;
}

#[tokio::test]
async fn checkpoint_restart_fault_matrix_preserves_exactly_once_window_output() {
    for point in CheckpointFaultPoint::ALL {
        for mode in CheckpointFaultMode::ALL {
            let report = run_checkpoint_restart_fault_case(point, mode).await;
            let durable_before_restart = matches!(
                point,
                CheckpointFaultPoint::ManifestRename
                    | CheckpointFaultPoint::ManifestParentSync
                    | CheckpointFaultPoint::PartialSinkCommit
                    | CheckpointFaultPoint::CompletedCommit
                    | CheckpointFaultPoint::Retention
                    | CheckpointFaultPoint::Compaction
            );
            assert_eq!(
                report.selected_before_restart,
                durable_before_restart.then_some(Epoch::INITIAL),
                "selected epoch mismatch at {point:?}/{mode:?}"
            );
            assert_eq!(
                report.prepared_artifacts_after_failure,
                usize::from(point == CheckpointFaultPoint::ManifestRename) * 2,
                "prepared-artifact preservation mismatch at {point:?}/{mode:?}"
            );
            assert_eq!(report.restored_epoch, Some(Epoch::INITIAL));
            assert_eq!(
                report.restored_cursor_orders,
                BTreeMap::from([
                    ("left".into(), 41_u64.to_be_bytes().to_vec()),
                    ("right".into(), 73_u64.to_be_bytes().to_vec()),
                ]),
                "restored source cursor mismatch at {point:?}/{mode:?}"
            );
            assert!(report.sources_ended);
            assert!(report.watermarks_restored);
            assert!(report.window_state_restored);
            assert_eq!(report.visible_records, 4);
            assert_eq!(report.duplicate_records, 0);
            assert_eq!(report.missing_records, 0);
            assert_eq!(report.temporary_artifacts, 0);
            assert_eq!(report.terminal_tasks, 0);
            assert_eq!(report.terminal_charged_edges, 0);
            assert_eq!(
                report.cancellation_requested,
                mode == CheckpointFaultMode::Cancel,
                "cancellation token mismatch at {point:?}/{mode:?}"
            );
            assert!(
                report.deterministic_terminal_error,
                "terminal error mismatch at {point:?}/{mode:?}"
            );
        }
    }
}

#[test]
fn temporary_artifact_oracle_covers_state_manifest_and_both_sinks() {
    let directory = tempfile::tempdir().unwrap();
    let state_root = directory.path().join("state");
    let manifest_root = directory.path().join("manifests");
    let sink_root = directory.path().join("sinks");
    let state_staging = state_root.join("staging/lineage/1/window");
    let state_committed = state_root.join("committed/lineage/window");
    std::fs::create_dir_all(&state_staging).unwrap();
    std::fs::create_dir_all(&state_committed).unwrap();
    std::fs::create_dir_all(&manifest_root).unwrap();
    std::fs::create_dir_all(sink_root.join("sink-a")).unwrap();
    std::fs::create_dir_all(sink_root.join("sink-b")).unwrap();
    std::fs::write(state_staging.join("segment.tmp"), b"temporary state").unwrap();
    std::fs::write(state_committed.join("segment.segment"), b"durable state").unwrap();
    std::fs::write(
        manifest_root.join("manifest-00000000000000000001.json"),
        b"durable manifest",
    )
    .unwrap();
    std::fs::write(manifest_root.join(".tmp-manifest"), b"temporary manifest").unwrap();
    std::fs::write(
        sink_root.join("sink-a/visible.json"),
        b"durable sink output",
    )
    .unwrap();
    std::fs::write(
        sink_root.join("sink-a/prepared-1.json"),
        b"prepared sink output",
    )
    .unwrap();
    std::fs::write(sink_root.join("sink-b/.tmp-sink"), b"temporary sink output").unwrap();

    assert_eq!(
        inspect_checkpoint_temporary_artifacts(&state_root, &manifest_root, &sink_root),
        CheckpointTemporaryArtifacts {
            state: 1,
            manifests: 1,
            sink_a: 1,
            sink_b: 1,
        }
    );
}
