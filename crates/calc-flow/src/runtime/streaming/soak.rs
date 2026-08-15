use std::{
    collections::{BTreeMap, BTreeSet},
    fs::{File, OpenOptions},
    hint::black_box,
    io::{Read as _, Write as _},
    path::{Path, PathBuf},
    process::{Command, Stdio},
    sync::{
        Arc,
        atomic::{AtomicUsize, Ordering},
    },
    time::Duration,
};

use async_trait::async_trait;
use chrono::{TimeZone, Utc};
use criterion::{BatchSize, Criterion};
use datafusion::arrow::{
    array::{Int64Array, StringArray, TimestampMicrosecondArray},
    datatypes::{DataType, Field, Schema, TimeUnit},
    record_batch::RecordBatch,
};
use parking_lot::Mutex;
use serde::{Deserialize, Serialize, de::DeserializeOwned};
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
        CheckpointFaultMode, CheckpointFaultPoint, CheckpointRuntimeSpec,
        CheckpointStartedTestGate, ContinuousJob, ContinuousJobState, ContinuousRunner,
        TerminalCause,
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
    OperatorManifestEntry, OperatorMetadata, OperatorStateSnapshot, PipelineBuilder, Port,
    PortEndpoint, RecoveryStatus, Result, SourceWatermarkManifestState, StateBackend,
    StateLineageBackend, StateLineageKey, StreamCollector, StreamOperator, StreamOperatorContext,
    StreamRequirements, StreamRuntimeConfig, UdfRegistry, UnionOperator, WindowAggregateOperator,
    WindowSpec,
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
const CHECKPOINT_SOAK_CHILD_ENV: &str = "CALC_FLOW_M5_CHECKPOINT_SOAK_CHILD_PLAN";
const CHECKPOINT_SOAK_CHILD_TEST: &str =
    "runtime::streaming::soak::checkpoint_restart_soak_generation_child_process";
const CHECKPOINT_SOAK_PROCESS_SCHEMA: &str = "calc-flow.m5-checkpoint-soak-process.v1";
const CHECKPOINT_SOAK_PLAN_SCHEMA: &str = "calc-flow.m5-checkpoint-soak-plan.v1";
const CHECKPOINT_SOAK_GENERATIONS: usize = 3;
const CHECKPOINT_SOAK_REPORT_LIMIT: u64 = 1 << 20;
const CHECKPOINT_SOAK_CADENCE_TOLERANCE: Duration = Duration::from_secs(2);
const MAX_CHECKPOINT_SOAK_RESTART_GAP: Duration = Duration::from_secs(60);
const CHECKPOINT_BENCHMARK_COMMAND: &str = "CARGO_TARGET_DIR=<fresh-candidate-target> CARGO_INCREMENTAL=0 CALC_FLOW_M5_CHECKPOINT_BENCHMARK=1 CALC_FLOW_M5_CHECKPOINT_BENCHMARK_RUN_ID=<unique-run-id> CALC_FLOW_M5_PRIVATE_SOURCE_COMMIT=<candidate-commit> CALC_FLOW_M5_PRIVATE_SOURCE_TREE=<candidate-tree> cargo test --release --locked -p calc-flow --lib runtime::streaming::soak::private_m5_epoch_checkpoint_absolute_benchmark -- --ignored --exact --nocapture";
const M5_PRIVATE_BENCHMARK_CASES: [&str; 12] = [
    "m5/private_path/barrier_cut_single_source",
    "m5/private_path/barrier_cut_two_source_fan_out",
    "m5/private_path/pass_through_two_input_alignment",
    "m5/private_path/window_two_input_alignment",
    "m5/private_path/dirty_window_state_stage",
    "m5/private_path/non_empty_manifest_publication",
    "m5/private_path/retained_delta_compacted_base_restore",
    "m5/private_path/single_transactional_sink_commit",
    "m5/private_path/multi_transactional_sink_commit",
    "m5/private_full_path/no_checkpoint",
    "m5/private_full_path/checkpoint_disabled",
    "m5/private_full_path/checkpoint_enabled",
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
        let cursor = Cursor::new(
            self.source_id,
            sequence.to_be_bytes().to_vec(),
            JsonMap::new(),
        )?;
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

#[derive(Clone, Copy, Debug, Deserialize, PartialEq, Serialize)]
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
    pause_before_eof: bool,
    poll_delay: Duration,
    opened_with: SourceOpenHistory,
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
        if self.next_sequence != 0 && self.pause_before_eof {
            return std::future::pending().await;
        }
        if self.next_sequence == 0 {
            self.next_sequence = 1;
            self.pending_watermark = true;
            return Ok(Some(SourceEvent::Data {
                batch: checkpoint_matrix_batch(self.source_id)?,
                cursor: Cursor::new(
                    self.source_id,
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
            cursor: Cursor::new(
                self.source_id,
                sequence.to_be_bytes().to_vec(),
                JsonMap::new(),
            )?,
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

    fn visible_epoch_path(&self, epoch: Epoch) -> PathBuf {
        self.root
            .join(format!("visible-{:020}.json", epoch.as_u64()))
    }

    async fn read_visible_records(&self) -> Result<Vec<String>> {
        read_checkpoint_matrix_sink_visible(&self.root).await
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
        let visible_path = self.visible_epoch_path(epoch);
        match tokio::fs::read(&visible_path).await {
            Ok(bytes) => {
                serde_json::from_slice::<Vec<String>>(&bytes).map_err(|error| {
                    CalcFlowError::Internal {
                        message: format!(
                            "checkpoint matrix visible epoch state is invalid: {error}"
                        ),
                    }
                })?;
            }
            Err(error) if error.kind() == std::io::ErrorKind::NotFound => {
                let records = self.prepared_records(epoch).await?;
                let bytes =
                    serde_json::to_vec(&records).map_err(|error| CalcFlowError::Internal {
                        message: format!("checkpoint matrix visible state encode failed: {error}"),
                    })?;
                let temporary = self
                    .root
                    .join(format!(".visible-{:020}.tmp", epoch.as_u64()));
                tokio::fs::write(&temporary, bytes).await.map_err(|error| {
                    CalcFlowError::Internal {
                        message: format!("checkpoint matrix visible state write failed: {error}"),
                    }
                })?;
                tokio::fs::rename(&temporary, &visible_path)
                    .await
                    .map_err(|error| CalcFlowError::Internal {
                        message: format!("checkpoint matrix visible state rename failed: {error}"),
                    })?;
            }
            Err(error) => {
                return Err(CalcFlowError::Internal {
                    message: format!("checkpoint matrix visible state read failed: {error}"),
                });
            }
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
    source_opened_with: &SourceOpenHistory,
    source_closed: &Arc<AtomicUsize>,
    sink_closed: &Arc<AtomicUsize>,
    sink_writes: &Arc<AtomicUsize>,
    source_poll_delay: Duration,
    pause_before_eof: bool,
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
        .compile_stream(
            &UdfRegistry::new().snapshot(),
            &if exactly_once {
                StreamRequirements {
                    delivery: BTreeMap::from([(
                        "output".into(),
                        crate::DeliveryGuarantee::ExactlyOnce,
                    )]),
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
                    pause_before_eof,
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
    let sinks = [("output", "sink-a"), ("output", "sink-b")]
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
    committed_sinks_before_restart: BTreeSet<String>,
    prepared_sinks_before_restart: BTreeSet<String>,
    restart_source_opens: Vec<(String, Option<Vec<u8>>)>,
    final_manifest_cursors: BTreeMap<String, CursorManifestEntry>,
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
            "version": 3,
            "cases": M5_PRIVATE_BENCHMARK_CASES,
            "private_paths": [
                "LiveProgressCoordinator::checkpoint_cut",
                "async OperatorBarrierAlignment pass-through/window",
                "ManifestTransaction::stage_operator_state",
                "ManifestTransaction::publish",
                "WindowAggregateOperator::restore compacted base plus deltas",
                "single/multi TransactionalStreamSink::commit",
                "ContinuousRunner no-checkpoint/terminal-only/periodic",
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
            "workload": {
                "case_count": M5_PRIVATE_BENCHMARK_CASES.len(),
                "barrier_sources": [1, 2],
                "barrier_fan_out": [1, 2],
                "restore_epochs": 35,
                "transactional_sink_counts": [1, 2],
                "full_runner_source_count": 2,
                "full_runner_sink_count": 2,
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
        "cargo_encoded_rustflags": std::env::var("CARGO_ENCODED_RUSTFLAGS")
            .unwrap_or_default()
            .split('\u{1f}')
            .map(|flag| {
                flag.split_once('=')
                    .filter(|(name, _)| *name == "--remap-path-prefix")
                    .map_or_else(|| flag.to_owned(), |(name, mapping)| {
                        let destination = mapping.rsplit_once('=').map_or(mapping, |(_, value)| value);
                        format!("{name}=<workspace>={destination}")
                    })
            })
            .collect::<Vec<_>>(),
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
            "checkpoint_modes": ["none", "terminal_only", "periodic"],
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
    sync_directory(&artifact_root)?;
    Ok(path)
}

// Directory fsync durability is a Unix idiom; plain File::open on a Windows
// directory fails with ERROR_ACCESS_DENIED, so non-Unix platforms skip the
// directory flush after the file-level sync_all calls above.
#[cfg(unix)]
fn sync_directory(path: &Path) -> Result<()> {
    File::open(path)
        .and_then(|directory| directory.sync_all())
        .map_err(|source| CalcFlowError::Io {
            path: path.display().to_string(),
            source,
        })
}

#[cfg(not(unix))]
fn sync_directory(_path: &Path) -> Result<()> {
    Ok(())
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
    let configured =
        PathBuf::from(std::env::var_os("CARGO_TARGET_DIR").unwrap_or_else(|| "target".into()));
    if configured.is_absolute() {
        configured
    } else {
        std::env::current_dir().unwrap().join(configured)
    }
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

struct PrivateBarrierCutFixture {
    coordinator: LiveProgressCoordinator,
    cuts: BTreeMap<BindingIdentity, DurableSourceCut>,
    receivers: Vec<crate::EdgeReceiver>,
    cancellation: CancellationToken,
}

fn prepare_private_barrier_cut_benchmark(
    source_ids: &[&str],
    fan_out: usize,
) -> PrivateBarrierCutFixture {
    let sources = source_ids
        .iter()
        .map(|source_id| private_benchmark_source(source_id))
        .collect::<Vec<_>>();
    let prepared = Arc::new(
        prepare_stream_job(
            "m5-private-barrier-cut",
            &sources,
            StreamProgressRuntimeConfig::default(),
        )
        .unwrap(),
    );
    let budget = EdgeBudget {
        max_rows: 4,
        max_bytes: 1 << 20,
    };
    let mut receivers = Vec::new();
    let outputs = source_ids
        .iter()
        .map(|source_id| {
            let senders = (0..fan_out)
                .map(|branch| {
                    let edge_id = format!("{source_id}-{branch}");
                    let (sender, receiver) = crate::edge_channel(edge_id, budget).unwrap();
                    receivers.push(receiver);
                    sender
                })
                .collect::<Vec<_>>();
            ((*source_id).into(), senders)
        })
        .collect();
    let cancellation = CancellationToken::new();
    let coordinator =
        LiveProgressCoordinator::new(&prepared, outputs, cancellation.clone()).unwrap();
    let cuts = source_ids
        .iter()
        .enumerate()
        .map(|(index, source_id)| {
            (
                BindingIdentity::new(*source_id).unwrap(),
                DurableSourceCut {
                    cursor: Some(CursorManifestEntry {
                        order: format!("{:02x}", index + 1),
                        payload: BTreeMap::new(),
                    }),
                    next_sequence: 0,
                    ended: false,
                },
            )
        })
        .collect();
    PrivateBarrierCutFixture {
        coordinator,
        cuts,
        receivers,
        cancellation,
    }
}

impl PrivateBarrierCutFixture {
    async fn measure(mut self) -> usize {
        self.coordinator
            .checkpoint_cut(Epoch::INITIAL, &self.cuts, &self.cancellation)
            .await
            .unwrap();
        let mut barriers = 0;
        for receiver in &mut self.receivers {
            let message = receiver.recv().await.unwrap().unwrap();
            assert_eq!(message.as_barrier(), Some(Epoch::INITIAL));
            barriers += 1;
        }
        barriers
    }
}

struct PrivateWindowAlignmentOperator {
    inputs: [Port; 2],
    outputs: [Port; 1],
    window: WindowAggregateOperator,
}

impl PrivateWindowAlignmentOperator {
    fn new() -> Self {
        let window = checkpoint_matrix_window();
        let fields = soak_schema()
            .fields()
            .iter()
            .map(|field| field.as_ref().clone())
            .collect::<Vec<_>>();
        Self {
            inputs: [
                Port::new("left", BatchKind::Table, true, Some(fields.clone())).unwrap(),
                Port::new("right", BatchKind::Table, true, Some(fields)).unwrap(),
            ],
            outputs: [window.output_ports()[0].clone()],
            window,
        }
    }
}

impl OperatorMetadata for PrivateWindowAlignmentOperator {
    fn name(&self) -> &'static str {
        "private-window-alignment"
    }

    fn input_ports(&self) -> &[Port] {
        &self.inputs
    }

    fn output_ports(&self) -> &[Port] {
        &self.outputs
    }

    fn configuration(&self) -> JsonMap {
        self.window.configuration()
    }
}

#[async_trait]
impl StreamOperator for PrivateWindowAlignmentOperator {
    async fn process_data(
        &mut self,
        _ingress: &str,
        batch: Batch,
        context: &StreamOperatorContext<'_>,
        output: &mut dyn StreamCollector,
    ) -> Result<()> {
        self.window
            .process_data("input", batch, context, output)
            .await
    }

    async fn on_watermark(
        &mut self,
        watermark: EventTime,
        context: &StreamOperatorContext<'_>,
        output: &mut dyn StreamCollector,
    ) -> Result<()> {
        self.window.on_watermark(watermark, context, output).await
    }

    async fn on_end(
        &mut self,
        context: &StreamOperatorContext<'_>,
        output: &mut dyn StreamCollector,
    ) -> Result<()> {
        self.window.on_end(context, output).await
    }

    fn checkpoint(&mut self, epoch: Epoch) -> Result<OperatorStateSnapshot> {
        self.window.checkpoint(epoch)
    }

    fn restore(&mut self, snapshot: &OperatorStateSnapshot) -> Result<()> {
        self.window.restore(snapshot)
    }
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

#[derive(Default)]
struct PrivateDiscardCollector {
    batches: usize,
}

#[async_trait]
impl StreamCollector for PrivateDiscardCollector {
    async fn emit(&mut self, _port: &str, _batch: Batch) -> Result<()> {
        self.batches += 1;
        Ok(())
    }
}

async fn private_dirty_window_snapshot(epoch: Epoch) -> OperatorStateSnapshot {
    let mut window = checkpoint_matrix_window();
    let job = StreamJobContext::new(
        31_000,
        PRIVATE_BENCHMARK_PIPELINE_FINGERPRINT,
        JsonMap::new(),
        None,
        CancellationToken::new(),
    );
    let context = StreamOperatorContext::new(&job, "window", None);
    let mut output = PrivateDiscardCollector::default();
    for (source_id, sequence) in [("left", 0), ("right", 0)] {
        window
            .process_data(
                "input",
                checkpoint_restart_soak_batch(source_id, sequence).unwrap(),
                &context,
                &mut output,
            )
            .await
            .unwrap();
    }
    let snapshot = window.checkpoint(epoch).unwrap();
    assert!(!snapshot.inline_metadata.is_empty());
    assert!(!snapshot.segments.is_empty());
    snapshot
}

fn private_benchmark_union() -> UnionOperator {
    let fields = soak_schema()
        .fields()
        .iter()
        .map(|field| field.as_ref().clone())
        .collect::<Vec<_>>();
    UnionOperator::new(
        "private-alignment-union",
        vec![
            Port::new("left", BatchKind::Table, true, Some(fields.clone())).unwrap(),
            Port::new("right", BatchKind::Table, true, Some(fields)).unwrap(),
        ],
    )
    .unwrap()
}

#[derive(Clone, Copy)]
enum PrivateAlignmentKind {
    PassThrough,
    Window,
}

struct PrivateAlignmentFixture {
    _directory: Option<tempfile::TempDir>,
    inner: super::operator_task::tests::BenchmarkAlignmentFixture,
    kind: PrivateAlignmentKind,
}

async fn prepare_private_alignment_benchmark(
    kind: PrivateAlignmentKind,
) -> PrivateAlignmentFixture {
    let batches = [
        checkpoint_restart_soak_batch("left", 0).unwrap(),
        checkpoint_restart_soak_batch("right", 0).unwrap(),
        checkpoint_restart_soak_batch("left", 1).unwrap(),
    ];
    let (directory, transaction, operator): (_, _, Box<dyn StreamOperator>) = match kind {
        PrivateAlignmentKind::PassThrough => (None, None, Box::new(private_benchmark_union())),
        PrivateAlignmentKind::Window => {
            let (directory, transaction, _identity) =
                private_benchmark_transaction(BTreeSet::from(["window".into()])).await;
            (
                Some(directory),
                Some(Arc::new(transaction)),
                Box::new(PrivateWindowAlignmentOperator::new()),
            )
        }
    };
    let inner = super::operator_task::tests::prepare_benchmark_alignment_fixture(
        operator,
        match kind {
            PrivateAlignmentKind::PassThrough => {
                crate::pipeline::OperatorCheckpointCapability::Stateless
            }
            PrivateAlignmentKind::Window => {
                crate::pipeline::OperatorCheckpointCapability::CheckpointedStateful {
                    state_version: crate::operator::WINDOW_STATE_LAYOUT_VERSION,
                }
            }
        },
        transaction,
        batches,
    )
    .await;
    PrivateAlignmentFixture {
        _directory: directory,
        inner,
        kind,
    }
}

impl PrivateAlignmentFixture {
    async fn measure(self) -> usize {
        let report = self.inner.measure().await.unwrap();
        assert_eq!(report.barriers, 1);
        match self.kind {
            PrivateAlignmentKind::PassThrough => {
                assert_eq!(report.data_before_barrier, 2);
                assert_eq!(report.state_segments, 0);
            }
            PrivateAlignmentKind::Window => {
                assert_eq!(report.data_before_barrier, 0);
                assert!(report.state_segments > 0);
            }
        }
        report.barriers + report.state_segments
    }
}

struct PrivateStateStageFixture {
    _directory: tempfile::TempDir,
    transaction: ManifestTransaction,
    snapshot: OperatorStateSnapshot,
}

async fn prepare_private_state_stage_benchmark() -> PrivateStateStageFixture {
    let (directory, transaction, _identity) =
        private_benchmark_transaction(BTreeSet::from(["window".into()])).await;
    let snapshot = private_dirty_window_snapshot(Epoch::INITIAL).await;
    PrivateStateStageFixture {
        _directory: directory,
        transaction,
        snapshot,
    }
}

impl PrivateStateStageFixture {
    async fn measure(self) -> usize {
        self.transaction
            .stage_operator_state("window", Epoch::INITIAL, self.snapshot)
            .await
            .unwrap()
            .segments
            .len()
    }
}

struct PrivateManifestPublicationFixture {
    directory: tempfile::TempDir,
    transaction: ManifestTransaction,
    prepared: PreparedEpochManifest,
}

async fn prepare_private_manifest_publication_benchmark() -> PrivateManifestPublicationFixture {
    let (directory, transaction, _identity) =
        private_benchmark_transaction(BTreeSet::from(["window".into()])).await;
    let staged = transaction
        .stage_operator_state(
            "window",
            Epoch::INITIAL,
            private_dirty_window_snapshot(Epoch::INITIAL).await,
        )
        .await
        .unwrap();
    let prepared = PreparedEpochManifest {
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
    };
    PrivateManifestPublicationFixture {
        directory,
        transaction,
        prepared,
    }
}

impl PrivateManifestPublicationFixture {
    async fn measure(self) -> usize {
        self.transaction.publish(self.prepared).await.unwrap();
        let manifest_path = self.directory.path().join("manifests");
        let bytes = std::fs::read_dir(&manifest_path)
            .unwrap()
            .filter_map(std::result::Result::ok)
            .find(|entry| is_canonical_checkpoint_manifest(&entry.path()))
            .map(|entry| std::fs::read(entry.path()).unwrap())
            .unwrap();
        assert!(bytes.len() > 2);
        checkpoint_manifest_count(&manifest_path)
    }
}

struct PrivateRestoreFixture {
    snapshot: OperatorStateSnapshot,
    window: WindowAggregateOperator,
}

async fn prepare_private_retained_restore_benchmark() -> PrivateRestoreFixture {
    let mut window = checkpoint_matrix_window();
    let job = StreamJobContext::new(
        31_001,
        PRIVATE_BENCHMARK_PIPELINE_FINGERPRINT,
        JsonMap::new(),
        None,
        CancellationToken::new(),
    );
    let context = StreamOperatorContext::new(&job, "window", None);
    let mut output = PrivateDiscardCollector::default();
    let mut retained_bytes = BTreeMap::new();
    let mut final_snapshot = None;
    for raw_epoch in 1..=35 {
        let epoch = Epoch::new(raw_epoch).unwrap();
        window
            .process_data(
                "input",
                checkpoint_restart_soak_batch("left", raw_epoch).unwrap(),
                &context,
                &mut output,
            )
            .await
            .unwrap();
        let snapshot = window.checkpoint(epoch).unwrap();
        retained_bytes.extend(snapshot.segments.clone());
        final_snapshot = Some(snapshot);
    }
    let mut snapshot = final_snapshot.unwrap();
    let retained_ids = snapshot.inline_metadata["segment_inventory"]
        .as_array()
        .unwrap()
        .iter()
        .map(|descriptor| {
            descriptor["handle"]["segment_id"]
                .as_str()
                .unwrap()
                .to_owned()
        })
        .collect::<BTreeSet<_>>();
    retained_bytes.retain(|segment_id, _| retained_ids.contains(segment_id));
    snapshot.segments = retained_bytes;
    PrivateRestoreFixture {
        snapshot,
        window: checkpoint_matrix_window(),
    }
}

impl PrivateRestoreFixture {
    fn measure(mut self) -> usize {
        let kinds = self.snapshot.inline_metadata["segment_inventory"]
            .as_array()
            .unwrap()
            .iter()
            .filter_map(|descriptor| descriptor["kind"].as_str())
            .collect::<BTreeSet<_>>();
        assert_eq!(kinds, BTreeSet::from(["base", "delta"]));
        self.window.restore(&self.snapshot).unwrap();
        self.snapshot.segments.len()
    }
}

struct PrivateSinkCommitFixture {
    _directory: tempfile::TempDir,
    sinks: Vec<(CheckpointMatrixSink, JsonMap)>,
}

async fn prepare_private_sink_commit_benchmark(sink_count: usize) -> PrivateSinkCommitFixture {
    let directory = tempfile::tempdir().unwrap();
    let mut sinks = Vec::new();
    for index in 0..sink_count {
        let sink_id = if index == 0 { "sink-a" } else { "sink-b" };
        let mut sink = CheckpointMatrixSink {
            sink_id,
            root: directory.path().join(sink_id),
            epoch: None,
            pending: Vec::new(),
            written_records: Arc::new(AtomicUsize::new(0)),
            closed: Arc::new(AtomicUsize::new(0)),
            write_delay: Duration::ZERO,
        };
        sink.open().await.unwrap();
        sink.begin_epoch(Epoch::INITIAL).await.unwrap();
        sink.pending.push(format!("{sink_id}|0|1|left|1"));
        let state = sink.pre_commit(Epoch::INITIAL).await.unwrap();
        sinks.push((sink, state));
    }
    PrivateSinkCommitFixture {
        _directory: directory,
        sinks,
    }
}

impl PrivateSinkCommitFixture {
    async fn measure(mut self) -> usize {
        let sink_count = self.sinks.len();
        for (sink, state) in &mut self.sinks {
            sink.commit(Epoch::INITIAL, state).await.unwrap();
        }
        let mut visible = 0;
        for (sink, _) in &self.sinks {
            visible += sink.read_visible_records().await.unwrap().len();
        }
        assert_eq!(visible, sink_count);
        visible
    }
}

async fn run_private_checkpoint_path_benchmark_smoke() -> BTreeMap<&'static str, usize> {
    BTreeMap::from([
        (
            M5_PRIVATE_BENCHMARK_CASES[0],
            prepare_private_barrier_cut_benchmark(&["source"], 1)
                .measure()
                .await,
        ),
        (
            M5_PRIVATE_BENCHMARK_CASES[1],
            prepare_private_barrier_cut_benchmark(&["left", "right"], 2)
                .measure()
                .await,
        ),
        (
            M5_PRIVATE_BENCHMARK_CASES[2],
            prepare_private_alignment_benchmark(PrivateAlignmentKind::PassThrough)
                .await
                .measure()
                .await,
        ),
        (
            M5_PRIVATE_BENCHMARK_CASES[3],
            prepare_private_alignment_benchmark(PrivateAlignmentKind::Window)
                .await
                .measure()
                .await,
        ),
        (
            M5_PRIVATE_BENCHMARK_CASES[4],
            prepare_private_state_stage_benchmark()
                .await
                .measure()
                .await,
        ),
        (
            M5_PRIVATE_BENCHMARK_CASES[5],
            prepare_private_manifest_publication_benchmark()
                .await
                .measure()
                .await,
        ),
        (
            M5_PRIVATE_BENCHMARK_CASES[6],
            prepare_private_retained_restore_benchmark().await.measure(),
        ),
        (
            M5_PRIVATE_BENCHMARK_CASES[7],
            prepare_private_sink_commit_benchmark(1)
                .await
                .measure()
                .await,
        ),
        (
            M5_PRIVATE_BENCHMARK_CASES[8],
            prepare_private_sink_commit_benchmark(2)
                .await
                .measure()
                .await,
        ),
    ])
}

#[derive(Clone, Copy)]
enum PrivateFullRunnerMode {
    NoCheckpoint,
    CheckpointDisabled,
    CheckpointEnabled,
}

fn private_full_runner_sink_delay(mode: PrivateFullRunnerMode) -> Duration {
    if matches!(mode, PrivateFullRunnerMode::NoCheckpoint) {
        Duration::ZERO
    } else {
        Duration::from_millis(5)
    }
}

struct PrivateFullRunnerFixture {
    directory: tempfile::TempDir,
    sink_root: PathBuf,
    source_closed: Arc<AtomicUsize>,
    sink_closed: Arc<AtomicUsize>,
    sink_writes: Arc<AtomicUsize>,
    runner: ContinuousRunner,
    spec: Option<ContinuousJobSpec>,
    checkpoint: Option<CheckpointRuntimeSpec>,
    mode: PrivateFullRunnerMode,
}

async fn prepare_private_full_runner_benchmark(
    mode: PrivateFullRunnerMode,
) -> PrivateFullRunnerFixture {
    let directory = tempfile::tempdir().unwrap();
    let sink_root = directory.path().join("sinks");
    let source_opens = Arc::new(Mutex::new(Vec::new()));
    let source_closed = Arc::new(AtomicUsize::new(0));
    let sink_closed = Arc::new(AtomicUsize::new(0));
    let sink_writes = Arc::new(AtomicUsize::new(0));
    let checkpointed = !matches!(mode, PrivateFullRunnerMode::NoCheckpoint);
    let spec = checkpoint_matrix_spec(
        30_000,
        &sink_root,
        &source_opens,
        &source_closed,
        &sink_closed,
        &sink_writes,
        private_full_runner_sink_delay(mode),
        false,
        CancellationToken::new(),
        checkpointed,
    );
    let checkpoint = if checkpointed {
        let backend = Arc::new(
            LocalStateBackend::new(directory.path().join("state"))
                .await
                .unwrap(),
        );
        Some(
            CheckpointRuntimeSpec::new(
                backend,
                directory.path().join("manifests"),
                StreamRuntimeConfig {
                    checkpoint_interval: if matches!(mode, PrivateFullRunnerMode::CheckpointEnabled)
                    {
                        Duration::from_millis(1)
                    } else {
                        Duration::from_secs(3_600)
                    },
                    checkpoint_timeout: Duration::from_secs(5),
                    retained_epochs: 2,
                    ..StreamRuntimeConfig::default()
                },
            )
            .unwrap(),
        )
    } else {
        None
    };
    PrivateFullRunnerFixture {
        directory,
        sink_root,
        source_closed,
        sink_closed,
        sink_writes,
        runner: ContinuousRunner::new(),
        spec: Some(spec),
        checkpoint,
        mode,
    }
}

impl PrivateFullRunnerFixture {
    async fn measure(mut self) -> PrivateCheckpointBenchmarkReport {
        let spec = self.spec.take().unwrap();
        let job = if let Some(checkpoint) = self.checkpoint.take() {
            self.runner
                .start_checkpointed(spec, checkpoint)
                .await
                .unwrap()
        } else {
            self.runner.start(spec).await.unwrap()
        };
        let outcome = tokio::time::timeout(Duration::from_secs(5), job.wait())
            .await
            .expect("private full-runner benchmark case hung");
        assert_eq!(outcome.state, ContinuousJobState::Completed);
        let restored_epoch = job
            .status()
            .checkpoint
            .and_then(|checkpoint| checkpoint.last_completed_epoch);
        drop(job);
        self.runner.shutdown().await.unwrap();
        assert_eq!(self.runner.registry_counts(), (0, 0));
        assert_eq!(self.source_closed.load(Ordering::SeqCst), 2);
        assert_eq!(self.sink_closed.load(Ordering::SeqCst), 2);
        let manifest_count = checkpoint_manifest_count(&self.directory.path().join("manifests"));
        let visible_records = if matches!(self.mode, PrivateFullRunnerMode::NoCheckpoint) {
            self.sink_writes.load(Ordering::SeqCst)
        } else {
            read_checkpoint_matrix_visible(&self.sink_root)
                .await
                .values()
                .map(Vec::len)
                .sum()
        };
        assert_eq!(visible_records, 4);
        match self.mode {
            PrivateFullRunnerMode::NoCheckpoint => assert_eq!(manifest_count, 0),
            PrivateFullRunnerMode::CheckpointDisabled => assert_eq!(manifest_count, 1),
            PrivateFullRunnerMode::CheckpointEnabled => assert!(manifest_count >= 2),
        }
        PrivateCheckpointBenchmarkReport {
            completed_jobs: 1,
            failed_jobs: 0,
            visible_records,
            manifest_count,
            restored_epoch,
        }
    }
}

async fn read_checkpoint_matrix_visible(root: &Path) -> BTreeMap<String, Vec<String>> {
    let mut visible = BTreeMap::new();
    for sink_id in ["sink-a", "sink-b"] {
        let sink_root = root.join(sink_id);
        visible.insert(
            sink_id.into(),
            read_checkpoint_matrix_sink_visible(&sink_root)
                .await
                .unwrap(),
        );
    }
    visible
}

async fn read_checkpoint_matrix_sink_visible(root: &Path) -> Result<Vec<String>> {
    let mut paths = Vec::new();
    let mut entries = match tokio::fs::read_dir(root).await {
        Ok(entries) => entries,
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => return Ok(Vec::new()),
        Err(error) => {
            return Err(CalcFlowError::Internal {
                message: format!("checkpoint matrix visible directory read failed: {error}"),
            });
        }
    };
    while let Some(entry) = entries
        .next_entry()
        .await
        .map_err(|error| CalcFlowError::Internal {
            message: format!("checkpoint matrix visible entry read failed: {error}"),
        })?
    {
        if is_canonical_checkpoint_sink_output(&entry.path()) {
            paths.push(entry.path());
        }
    }
    paths.sort();
    let mut records = Vec::new();
    for path in paths {
        let bytes = tokio::fs::read(&path)
            .await
            .map_err(|error| CalcFlowError::Internal {
                message: format!("checkpoint matrix visible epoch read failed: {error}"),
            })?;
        let mut epoch_records: Vec<String> =
            serde_json::from_slice(&bytes).map_err(|error| CalcFlowError::Internal {
                message: format!("checkpoint matrix visible epoch is invalid: {error}"),
            })?;
        records.append(&mut epoch_records);
    }
    Ok(records)
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

fn is_canonical_checkpoint_sink_output(path: &Path) -> bool {
    let Some(name) = path.file_name().and_then(|name| name.to_str()) else {
        return false;
    };
    let Some(epoch) = name
        .strip_prefix("visible-")
        .and_then(|name| name.strip_suffix(".json"))
    else {
        return false;
    };
    epoch.len() == 20 && epoch.bytes().all(|byte| byte.is_ascii_digit())
}

fn checkpoint_matrix_sink_has_visible(root: &Path) -> bool {
    std::fs::read_dir(root).is_ok_and(|entries| {
        entries
            .filter_map(std::result::Result::ok)
            .any(|entry| is_canonical_checkpoint_sink_output(&entry.path()))
    })
}

fn inspect_checkpoint_temporary_artifacts(
    state_root: &Path,
    manifest_root: &Path,
    sink_root: &Path,
) -> CheckpointTemporaryArtifacts {
    CheckpointTemporaryArtifacts {
        state: count_checkpoint_artifacts(&state_root.join("staging"), |_| false),
        manifests: count_checkpoint_artifacts(manifest_root, is_canonical_checkpoint_manifest),
        sink_a: count_checkpoint_artifacts(
            &sink_root.join("sink-a"),
            is_canonical_checkpoint_sink_output,
        ),
        sink_b: count_checkpoint_artifacts(
            &sink_root.join("sink-b"),
            is_canonical_checkpoint_sink_output,
        ),
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
        checkpoint_interval: Duration::from_millis(10),
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
        true,
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
    let deterministic_terminal_error = match (point, mode) {
        (CheckpointFaultPoint::ManifestRename, _) => {
            first_outcome.state == ContinuousJobState::RecoveryRequired
                && first_outcome.errors.iter().any(|failure| {
                    matches!(
                        &failure.error,
                        CalcFlowError::RecoveryRequired {
                            pipeline_name,
                            message,
                        } if pipeline_name == "checkpoint-restart-fault-matrix"
                            && message.contains("checkpoint epoch 1")
                            && message.contains("publication durability is unknown")
                    )
                })
        }
        (CheckpointFaultPoint::PartialSinkCommit, CheckpointFaultMode::Cancel) => {
            first_outcome.state == ContinuousJobState::RecoveryRequired
                && !first_outcome.errors.is_empty()
                && first_cancellation.is_cancelled()
        }
        (
            CheckpointFaultPoint::PartialSinkCommit,
            CheckpointFaultMode::Io | CheckpointFaultMode::Panic | CheckpointFaultMode::Restart,
        ) => {
            first_outcome.state == ContinuousJobState::RecoveryRequired
                && first_outcome.errors.iter().any(|failure| {
                    matches!(
                        &failure.error,
                        CalcFlowError::RecoveryRequired {
                            pipeline_name,
                            message,
                        } if pipeline_name == "checkpoint-restart-fault-matrix"
                            && message.contains("sink \"sink-a\"")
                            && message.contains("epoch 1")
                    )
                })
        }
        (_, CheckpointFaultMode::Cancel) => {
            first_outcome.state == ContinuousJobState::Cancelled
                && first_outcome.errors.is_empty()
                && first_cancellation.is_cancelled()
        }
        (_, CheckpointFaultMode::Io) => {
            first_outcome.state == ContinuousJobState::Failed
                && first_error.contains("injected checkpoint I/O fault")
        }
        (_, CheckpointFaultMode::Panic) => {
            first_outcome.state == ContinuousJobState::Failed
                && first_error.contains("injected checkpoint panic")
        }
        (_, CheckpointFaultMode::Restart) => {
            first_outcome.state == ContinuousJobState::Failed
                && first_error.contains("injected checkpoint restart")
        }
    };
    let manifest_path = manifest_root.join("manifest-00000000000000000001.json");
    let selected_before_restart_manifest = if manifest_path.exists() {
        Some(
            crate::CheckpointManifest::from_bytes(&tokio::fs::read(&manifest_path).await.unwrap())
                .unwrap(),
        )
    } else {
        None
    };
    let selected_before_restart = selected_before_restart_manifest
        .as_ref()
        .map(crate::CheckpointManifest::epoch);
    let window_state_restored = selected_before_restart_manifest
        .as_ref()
        .and_then(|manifest| manifest.operators().get("window"))
        .is_some_and(|entry| !entry.segments.is_empty());
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
    let prepared_sinks_before_restart = ["sink-a", "sink-b"]
        .into_iter()
        .filter(|sink_id| {
            sink_root
                .join(sink_id)
                .join("prepared-00000000000000000001.json")
                .exists()
        })
        .map(str::to_owned)
        .collect::<BTreeSet<_>>();
    let committed_sinks_before_restart = ["sink-a", "sink-b"]
        .into_iter()
        .filter(|sink_id| checkpoint_matrix_sink_has_visible(&sink_root.join(sink_id)))
        .map(str::to_owned)
        .collect::<BTreeSet<_>>();

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
        false,
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

    let manifests = checkpoint_manifest_documents(&manifest_root).await;
    let manifest = manifests
        .last()
        .expect("restart matrix published no final manifest");
    let final_manifest_cursors = manifest
        .sources()
        .iter()
        .map(|(source_id, entry)| {
            (
                source_id.clone(),
                entry.cursor.clone().expect("matrix source cursor is final"),
            )
        })
        .collect();
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

    let mut restart_source_open_events = restart_source_opens.lock().clone();
    restart_source_open_events.sort_by(|left, right| left.0.cmp(&right.0));
    CheckpointFaultMatrixReport {
        selected_before_restart,
        prepared_artifacts_after_failure,
        committed_sinks_before_restart,
        prepared_sinks_before_restart,
        restart_source_opens: restart_source_open_events,
        final_manifest_cursors,
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
    generation_process_ids: Vec<u32>,
    generation_exit_codes: Vec<i32>,
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

#[derive(Clone, Copy, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(rename_all = "snake_case")]
enum CheckpointSoakProcessMode {
    Smoke,
    Standard,
}

#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
struct CheckpointSoakProcessPlan {
    schema: String,
    commit: String,
    executable_sha256: String,
    run_root: PathBuf,
    parent_pid: u32,
    parent_launch_offset_micros: u64,
    generation: usize,
    sample_start: usize,
    sample_end: usize,
    checkpoint_interval_millis: u64,
    checkpoint_timeout_millis: u64,
    target_checkpoints: u64,
    sink_write_delay_millis: u64,
    retained_epochs: usize,
    final_generation: bool,
    mode: CheckpointSoakProcessMode,
    config_hash: String,
}

#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
struct CheckpointSoakProcessSample {
    index: usize,
    elapsed_micros: u64,
    rss_kib: u64,
    task_count: usize,
    maximum_queue_depth: usize,
    maximum_charged_rows: usize,
    maximum_charged_bytes: usize,
    completed_checkpoints: u64,
    failed_checkpoints: u64,
    manifest_count: usize,
    state_bytes: u64,
}

#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
struct CheckpointSoakProcessReport {
    schema: String,
    commit: String,
    executable_sha256: String,
    config_hash: String,
    run_root: PathBuf,
    parent_pid: u32,
    pid: u32,
    generation: usize,
    sample_start: usize,
    sample_end: usize,
    generation_started_micros: u64,
    generation_finished_micros: u64,
    restored_epoch: Option<u64>,
    restored_cursor_orders: BTreeMap<String, Option<String>>,
    terminal_epoch: u64,
    terminal_cursor_orders: BTreeMap<String, String>,
    restored_window_state: bool,
    restored_watermarks: bool,
    restored_progress: bool,
    samples: Vec<CheckpointSoakProcessSample>,
    compacted_epochs: Vec<u64>,
    maximum_manifest_count: usize,
    maximum_state_bytes: u64,
    completed_checkpoints: u64,
    failed_checkpoints: u64,
    source_open_events: usize,
    source_close_events: usize,
    sink_close_events: usize,
    terminal_cause: String,
    terminal_tasks: usize,
    terminal_charged_edges: usize,
    terminal_registries: (usize, usize),
    source_records: usize,
    output_records: usize,
    duplicate_records: usize,
    missing_records: usize,
    temporary_artifacts: usize,
}

#[derive(Clone, Debug, PartialEq)]
struct CheckpointSoakParentTiming {
    generation: usize,
    launch_micros: u64,
    finish_micros: u64,
}

struct CheckpointSoakProcessEvidence {
    report: CheckpointRestartSoakReport,
    processes: Vec<CheckpointSoakProcessReport>,
    parent_timings: Vec<CheckpointSoakParentTiming>,
    rss_samples: Vec<RssSample>,
}

fn checkpoint_restart_soak_metadata(commit: &str, kernel: &str, rustc: &str) -> serde_json::Value {
    json!({
        "schema": "calc-flow.m5-checkpoint-soak.v1",
        "commit": commit,
        "target_duration_seconds": 1_200,
        "sample_count": CHECKPOINT_SOAK_SAMPLE_COUNT,
        "cadence_seconds": CHECKPOINT_SOAK_CADENCE.as_secs(),
        "cadence_tolerance_seconds": CHECKPOINT_SOAK_CADENCE_TOLERANCE.as_secs(),
        "maximum_restart_gap_seconds": MAX_CHECKPOINT_SOAK_RESTART_GAP.as_secs(),
        "timing_source": "parent_std_instant_plus_child_local_instant",
        "restart_sample_indices": CHECKPOINT_SOAK_RESTART_SAMPLES,
        "restart_kind": "os_process",
        "process_generations": CHECKPOINT_SOAK_GENERATIONS,
        "process_sample_ranges": [[0, 40], [40, 80], [80, 120]],
        "child_report_schema": CHECKPOINT_SOAK_PROCESS_SCHEMA,
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

fn checkpoint_soak_process_error(message: impl Into<String>) -> CalcFlowError {
    CalcFlowError::InvalidArgument {
        field: "checkpoint_restart_soak_process".into(),
        message: message.into(),
    }
}

fn checkpoint_soak_test_executable() -> Result<PathBuf> {
    let executable = std::env::current_exe().map_err(|source| CalcFlowError::Io {
        path: "current checkpoint soak test executable".into(),
        source,
    })?;
    let executable = executable
        .canonicalize()
        .map_err(|source| CalcFlowError::Io {
            path: executable.display().to_string(),
            source,
        })?;
    let metadata = std::fs::symlink_metadata(&executable).map_err(|source| CalcFlowError::Io {
        path: executable.display().to_string(),
        source,
    })?;
    if !executable.is_absolute() || metadata.file_type().is_symlink() || !metadata.is_file() {
        return Err(checkpoint_soak_process_error(
            "test executable must resolve to an absolute regular file",
        ));
    }
    Ok(executable)
}

fn checkpoint_soak_file_sha256(path: &Path) -> Result<String> {
    let mut file = File::open(path).map_err(|source| CalcFlowError::Io {
        path: path.display().to_string(),
        source,
    })?;
    let mut hasher = Sha256::new();
    let mut buffer = [0_u8; 16 * 1_024];
    loop {
        let read = file.read(&mut buffer).map_err(|source| CalcFlowError::Io {
            path: path.display().to_string(),
            source,
        })?;
        if read == 0 {
            break;
        }
        hasher.update(&buffer[..read]);
    }
    Ok(hex::encode(hasher.finalize()))
}

fn checkpoint_soak_plan_hash(plan: &CheckpointSoakProcessPlan) -> String {
    sha256_bytes(
        &serde_json::to_vec(&json!({
            "schema": plan.schema,
            "commit": plan.commit,
            "executable_sha256": plan.executable_sha256,
            "run_root": plan.run_root,
            "parent_pid": plan.parent_pid,
            "parent_launch_offset_micros": plan.parent_launch_offset_micros,
            "generation": plan.generation,
            "sample_start": plan.sample_start,
            "sample_end": plan.sample_end,
            "checkpoint_interval_millis": plan.checkpoint_interval_millis,
            "checkpoint_timeout_millis": plan.checkpoint_timeout_millis,
            "target_checkpoints": plan.target_checkpoints,
            "sink_write_delay_millis": plan.sink_write_delay_millis,
            "retained_epochs": plan.retained_epochs,
            "final_generation": plan.final_generation,
            "mode": plan.mode,
        }))
        .unwrap(),
    )
}

fn bind_checkpoint_soak_parent_launch(
    plan: &mut CheckpointSoakProcessPlan,
    elapsed: Duration,
) -> Result<()> {
    plan.parent_launch_offset_micros = checkpoint_soak_parent_elapsed_micros(elapsed)?;
    plan.config_hash = checkpoint_soak_plan_hash(plan);
    Ok(())
}

fn checkpoint_soak_parent_elapsed_micros(elapsed: Duration) -> Result<u64> {
    u64::try_from(elapsed.as_micros()).map_err(|_| {
        checkpoint_soak_process_error("checkpoint soak parent clock exhausted u64 microseconds")
    })
}

fn checkpoint_soak_process_plans(
    run_root: &Path,
    commit: &str,
    executable_sha256: &str,
    mode: CheckpointSoakProcessMode,
) -> Vec<CheckpointSoakProcessPlan> {
    let ranges = match mode {
        CheckpointSoakProcessMode::Smoke => [0..0, 0..0, 0..0],
        CheckpointSoakProcessMode::Standard => [0..40, 40..80, 80..120],
    };
    ranges
        .into_iter()
        .enumerate()
        .map(|(generation, samples)| {
            let mut plan = CheckpointSoakProcessPlan {
                schema: CHECKPOINT_SOAK_PLAN_SCHEMA.into(),
                commit: commit.into(),
                executable_sha256: executable_sha256.into(),
                run_root: run_root.to_path_buf(),
                parent_pid: std::process::id(),
                parent_launch_offset_micros: 0,
                generation,
                sample_start: samples.start,
                sample_end: samples.end,
                checkpoint_interval_millis: match mode {
                    CheckpointSoakProcessMode::Smoke => 50,
                    CheckpointSoakProcessMode::Standard => 5_000,
                },
                checkpoint_timeout_millis: 10_000,
                target_checkpoints: match mode {
                    CheckpointSoakProcessMode::Smoke => 12,
                    CheckpointSoakProcessMode::Standard => 0,
                },
                sink_write_delay_millis: match mode {
                    CheckpointSoakProcessMode::Smoke => 0,
                    CheckpointSoakProcessMode::Standard => 20,
                },
                retained_epochs: 2,
                final_generation: generation + 1 == CHECKPOINT_SOAK_GENERATIONS,
                mode,
                config_hash: String::new(),
            };
            plan.config_hash = checkpoint_soak_plan_hash(&plan);
            plan
        })
        .collect()
}

fn checkpoint_soak_plan_path(plan: &CheckpointSoakProcessPlan) -> PathBuf {
    plan.run_root
        .join("evidence")
        .join("plans")
        .join(format!("generation-{}.json", plan.generation))
}

fn checkpoint_soak_report_path(plan: &CheckpointSoakProcessPlan) -> PathBuf {
    plan.run_root
        .join("evidence")
        .join("reports")
        .join(format!("generation-{}.json", plan.generation))
}

fn write_checkpoint_soak_document<T: Serialize>(path: &Path, value: &T) -> Result<()> {
    let parent = path.parent().ok_or_else(|| {
        checkpoint_soak_process_error("checkpoint soak document has no parent directory")
    })?;
    std::fs::create_dir_all(parent).map_err(|source| CalcFlowError::Io {
        path: parent.display().to_string(),
        source,
    })?;
    if path.exists() {
        return Err(CalcFlowError::Conflict {
            resource: "checkpoint soak evidence".into(),
            key: path.display().to_string(),
        });
    }
    let bytes = serde_json::to_vec(value).map_err(|error| CalcFlowError::Internal {
        message: format!("checkpoint soak document encoding failed: {error}"),
    })?;
    let temporary = path.with_extension(format!("json.tmp-{}", std::process::id()));
    let mut file = OpenOptions::new()
        .create_new(true)
        .write(true)
        .open(&temporary)
        .map_err(|source| CalcFlowError::Io {
            path: temporary.display().to_string(),
            source,
        })?;
    file.write_all(&bytes)
        .and_then(|()| file.sync_all())
        .map_err(|source| CalcFlowError::Io {
            path: temporary.display().to_string(),
            source,
        })?;
    std::fs::rename(&temporary, path).map_err(|source| CalcFlowError::Io {
        path: path.display().to_string(),
        source,
    })?;
    sync_checkpoint_soak_directory(parent)
}

#[cfg(unix)]
fn sync_checkpoint_soak_directory(path: &Path) -> Result<()> {
    File::open(path)
        .and_then(|directory| directory.sync_all())
        .map_err(|source| CalcFlowError::Io {
            path: path.display().to_string(),
            source,
        })
}

#[cfg(not(unix))]
fn sync_checkpoint_soak_directory(_path: &Path) -> Result<()> {
    Ok(())
}

fn read_checkpoint_soak_document<T>(path: &Path) -> Result<T>
where
    T: DeserializeOwned + Serialize,
{
    let metadata = std::fs::symlink_metadata(path).map_err(|source| CalcFlowError::Io {
        path: path.display().to_string(),
        source,
    })?;
    if metadata.file_type().is_symlink()
        || !metadata.is_file()
        || metadata.len() > CHECKPOINT_SOAK_REPORT_LIMIT
    {
        return Err(checkpoint_soak_process_error(format!(
            "checkpoint soak document {} is not a bounded regular file",
            path.display()
        )));
    }
    let bytes = std::fs::read(path).map_err(|source| CalcFlowError::Io {
        path: path.display().to_string(),
        source,
    })?;
    let value: T = serde_json::from_slice(&bytes).map_err(|error| {
        checkpoint_soak_process_error(format!(
            "checkpoint soak document {} is malformed: {error}",
            path.display()
        ))
    })?;
    let canonical = serde_json::to_vec(&value).unwrap();
    if canonical != bytes {
        let mismatch = canonical
            .iter()
            .zip(&bytes)
            .position(|(canonical, stored)| canonical != stored)
            .unwrap_or(canonical.len().min(bytes.len()));
        return Err(checkpoint_soak_process_error(format!(
            "checkpoint soak document {} is not canonical at byte {mismatch} (stored {}, canonical {})",
            path.display(),
            bytes.len(),
            canonical.len(),
        )));
    }
    Ok(value)
}

fn validate_checkpoint_soak_plan(plan: &CheckpointSoakProcessPlan) -> Result<()> {
    let expected_ranges = match plan.mode {
        CheckpointSoakProcessMode::Smoke => [0..0, 0..0, 0..0],
        CheckpointSoakProcessMode::Standard => [0..40, 40..80, 80..120],
    };
    let expected = expected_ranges.get(plan.generation).ok_or_else(|| {
        checkpoint_soak_process_error("checkpoint soak generation is out of range")
    })?;
    let root = plan
        .run_root
        .canonicalize()
        .map_err(|source| CalcFlowError::Io {
            path: plan.run_root.display().to_string(),
            source,
        })?;
    let executable = checkpoint_soak_test_executable()?;
    let current_commit = strict_command_output("git", &["rev-parse", "HEAD"])?;
    if plan.schema != CHECKPOINT_SOAK_PLAN_SCHEMA
        || plan.run_root != root
        || plan.commit != current_commit
        || plan.executable_sha256 != checkpoint_soak_file_sha256(&executable)?
        || plan.parent_pid == 0
        || plan.parent_pid == std::process::id()
        || plan.sample_start != expected.start
        || plan.sample_end != expected.end
        || plan.retained_epochs != 2
        || plan.final_generation != (plan.generation + 1 == CHECKPOINT_SOAK_GENERATIONS)
        || plan.config_hash != checkpoint_soak_plan_hash(plan)
    {
        return Err(checkpoint_soak_process_error(
            "checkpoint soak child plan identity or schedule is invalid",
        ));
    }
    Ok(())
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

fn checkpoint_soak_manifest_cursors(
    manifest: &crate::CheckpointManifest,
) -> BTreeMap<String, String> {
    manifest
        .sources()
        .iter()
        .map(|(source_id, entry)| {
            (
                source_id.clone(),
                entry.cursor.as_ref().unwrap().order.clone(),
            )
        })
        .collect()
}

fn checkpoint_soak_open_cursors(
    opened_with: &SourceOpenHistory,
) -> BTreeMap<String, Option<String>> {
    opened_with
        .lock()
        .iter()
        .map(|(source_id, cursor)| (source_id.clone(), cursor.as_ref().map(hex::encode)))
        .collect()
}

fn checkpoint_soak_restore_evidence(
    selected: Option<&crate::CheckpointManifest>,
) -> (bool, bool, bool) {
    let Some(manifest) = selected else {
        return (false, false, false);
    };
    let window = manifest.operators().get("window");
    let window_state = window.is_some_and(|entry| !entry.segments.is_empty());
    let watermarks = window.is_some_and(|entry| {
        !entry.progress.is_empty()
            && entry
                .progress
                .values()
                .all(|progress| progress.watermark.is_some())
    });
    let progress = manifest.sources().values().all(|entry| {
        entry.cursor.is_some()
            && entry.sequence > 0
            && matches!(
                entry.watermark_policy,
                SourceWatermarkManifestState::SourceProvided {
                    last_emitted_micros: Some(_),
                    ..
                }
            )
    });
    (window_state, watermarks, progress)
}

#[cfg(target_os = "linux")]
async fn checkpoint_soak_process_rss_kib() -> Result<u64> {
    let process_status = tokio::fs::read_to_string("/proc/self/status")
        .await
        .map_err(|source| CalcFlowError::Io {
            path: "/proc/self/status".into(),
            source,
        })?;
    parse_vm_rss_kib(&process_status)
        .ok_or_else(|| checkpoint_soak_process_error("Linux checkpoint soak status omitted VmRSS"))
}

#[cfg(not(target_os = "linux"))]
async fn checkpoint_soak_process_rss_kib() -> Result<u64> {
    Err(checkpoint_soak_process_error(
        "standard checkpoint soak RSS evidence requires Linux",
    ))
}

async fn checkpoint_soak_process_sample(
    job: &ContinuousJob,
    state_root: &Path,
    manifest_root: &Path,
    index: usize,
    elapsed_micros: u64,
) -> Result<CheckpointSoakProcessSample> {
    let status = job.status();
    if status.state != ContinuousJobState::Running || status.metrics.checkpoints.failed != 0 {
        return Err(checkpoint_soak_process_error(format!(
            "checkpoint soak child terminated before sample {index}"
        )));
    }
    assert_edge_budgets(&status);
    let maximum_queue_depth = status
        .edges
        .values()
        .map(|edge| edge.queue_depth)
        .max()
        .unwrap_or(0);
    let maximum_charged_rows = status
        .edges
        .values()
        .map(|edge| edge.charged_rows)
        .max()
        .unwrap_or(0);
    let maximum_charged_bytes = status
        .edges
        .values()
        .map(|edge| edge.charged_bytes)
        .max()
        .unwrap_or(0);
    Ok(CheckpointSoakProcessSample {
        index,
        elapsed_micros,
        rss_kib: checkpoint_soak_process_rss_kib().await?,
        task_count: status.tasks.len(),
        maximum_queue_depth,
        maximum_charged_rows,
        maximum_charged_bytes,
        completed_checkpoints: status.metrics.checkpoints.completed,
        failed_checkpoints: status.metrics.checkpoints.failed,
        manifest_count: checkpoint_manifest_documents(manifest_root).await.len(),
        state_bytes: directory_regular_file_bytes(state_root).await,
    })
}

struct CheckpointSoakTerminalEvidence {
    cause: &'static str,
    completed_checkpoints: u64,
    failed_checkpoints: u64,
    terminal_tasks: usize,
    terminal_charged_edges: usize,
    terminal_registries: (usize, usize),
}

async fn settle_checkpoint_soak_process(
    mut runner: ContinuousRunner,
    job: ContinuousJob,
    stop: &AtomicUsize,
    final_generation: bool,
) -> Result<CheckpointSoakTerminalEvidence> {
    let outcome = if final_generation {
        stop.store(1, Ordering::SeqCst);
        tokio::time::timeout(Duration::from_secs(20), job.wait())
            .await
            .map_err(|_| checkpoint_soak_process_error("terminal checkpoint timed out"))?
    } else {
        job.cancel().await
    };
    let expected_state = if final_generation {
        ContinuousJobState::Completed
    } else {
        ContinuousJobState::Cancelled
    };
    let expected_cause = if final_generation {
        TerminalCause::NaturalEnd
    } else {
        TerminalCause::ExplicitCancel
    };
    if outcome.state != expected_state || outcome.cause != expected_cause {
        return Err(checkpoint_soak_process_error(format!(
            "checkpoint soak child terminated unexpectedly: {outcome:?}"
        )));
    }
    let status = job.status();
    let terminal_tasks = status.tasks.len();
    let terminal_charged_edges = status
        .edges
        .values()
        .filter(|edge| edge.queue_depth != 0 || edge.charged_rows != 0 || edge.charged_bytes != 0)
        .count();
    let completed_checkpoints = status.metrics.checkpoints.completed;
    let failed_checkpoints = status.metrics.checkpoints.failed;
    drop(job);
    runner.shutdown().await?;
    let terminal_registries = runner.registry_counts();
    Ok(CheckpointSoakTerminalEvidence {
        cause: if final_generation {
            "natural_end"
        } else {
            "explicit_cancel"
        },
        completed_checkpoints,
        failed_checkpoints,
        terminal_tasks,
        terminal_charged_edges,
        terminal_registries,
    })
}

async fn checkpoint_manifest_documents(root: &Path) -> Vec<crate::CheckpointManifest> {
    let mut manifests = Vec::new();
    let mut entries = match tokio::fs::read_dir(root).await {
        Ok(entries) => entries,
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => return manifests,
        Err(error) => panic!("checkpoint soak could not scan manifests: {error}"),
    };
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

#[allow(
    clippy::too_many_lines,
    reason = "one child owns its independent runtime, durable restore, samples, and terminal report"
)]
async fn run_checkpoint_soak_child(
    plan: &CheckpointSoakProcessPlan,
) -> Result<CheckpointSoakProcessReport> {
    let generation_started = tokio::time::Instant::now();
    let generation_started_micros = plan.parent_launch_offset_micros;
    validate_checkpoint_soak_plan(plan)?;
    let state_root = plan.run_root.join("state");
    let manifest_root = plan.run_root.join("manifests");
    let sink_root = plan.run_root.join("sinks");
    let before = checkpoint_manifest_documents(&manifest_root).await;
    let selected = before.last();
    let restored_epoch = selected.map(|manifest| manifest.epoch().as_u64());
    let restored_cursor_orders = selected.map_or_else(
        || BTreeMap::from([("left".into(), None), ("right".into(), None)]),
        |manifest| {
            checkpoint_soak_manifest_cursors(manifest)
                .into_iter()
                .map(|(source_id, order)| (source_id, Some(order)))
                .collect()
        },
    );
    let (restored_window_state, restored_watermarks, restored_progress) =
        checkpoint_soak_restore_evidence(selected);
    let backend = Arc::new(LocalStateBackend::new(&state_root).await?);
    let stop = Arc::new(AtomicUsize::new(0));
    let opened_with = Arc::new(Mutex::new(Vec::new()));
    let source_closed = Arc::new(AtomicUsize::new(0));
    let sink_closed = Arc::new(AtomicUsize::new(0));
    let config = StreamRuntimeConfig {
        checkpoint_interval: Duration::from_millis(plan.checkpoint_interval_millis),
        checkpoint_timeout: Duration::from_millis(plan.checkpoint_timeout_millis),
        retained_epochs: plan.retained_epochs,
        ..StreamRuntimeConfig::default()
    };
    let (runner, job) = start_checkpoint_restart_generation(
        50_000 + u64::try_from(plan.generation).unwrap(),
        backend,
        &manifest_root,
        &sink_root,
        &stop,
        &opened_with,
        &source_closed,
        &sink_closed,
        config,
        Duration::from_millis(plan.sink_write_delay_millis),
    )
    .await;
    let initial_status = job.status();
    if initial_status.tasks.is_empty()
        || checkpoint_soak_open_cursors(&opened_with) != restored_cursor_orders
    {
        return Err(checkpoint_soak_process_error(
            "checkpoint soak child did not restore the durable source cursors",
        ));
    }
    assert_edge_budgets(&initial_status);

    let mut samples = Vec::with_capacity(plan.sample_end - plan.sample_start);
    if plan.mode == CheckpointSoakProcessMode::Smoke {
        wait_for_completed_checkpoints(&job, plan.target_checkpoints).await;
    } else {
        for (local_index, index) in (plan.sample_start..plan.sample_end).enumerate() {
            let sample_number = u32::try_from(local_index + 1).unwrap();
            tokio::time::sleep_until(generation_started + CHECKPOINT_SOAK_CADENCE * sample_number)
                .await;
            let elapsed_micros = generation_started_micros
                .checked_add(checkpoint_soak_parent_elapsed_micros(
                    generation_started.elapsed(),
                )?)
                .ok_or_else(|| {
                    checkpoint_soak_process_error("checkpoint soak sample clock overflowed")
                })?;
            samples.push(
                checkpoint_soak_process_sample(
                    &job,
                    &state_root,
                    &manifest_root,
                    index,
                    elapsed_micros,
                )
                .await?,
            );
        }
    }
    let before_terminal = job.status();
    let steady_task_count = initial_status.tasks.len();
    if before_terminal.tasks.len() != steady_task_count
        || before_terminal.metrics.checkpoints.failed != 0
    {
        return Err(checkpoint_soak_process_error(
            "checkpoint soak child task or checkpoint metrics drifted",
        ));
    }
    let terminal =
        settle_checkpoint_soak_process(runner, job, &stop, plan.final_generation).await?;
    let manifests = checkpoint_manifest_documents(&manifest_root).await;
    let selected_terminal = manifests.last().ok_or_else(|| {
        checkpoint_soak_process_error("checkpoint soak child published no terminal manifest")
    })?;
    let terminal_cursor_orders = checkpoint_soak_manifest_cursors(selected_terminal);
    let compacted_epochs = manifests
        .iter()
        .filter(|manifest| manifest_has_compacted_window(manifest))
        .map(|manifest| manifest.epoch().as_u64())
        .collect::<Vec<_>>();
    let maximum_manifest_count = samples
        .iter()
        .map(|sample| sample.manifest_count)
        .max()
        .unwrap_or(0)
        .max(manifests.len());
    let maximum_state_bytes = samples
        .iter()
        .map(|sample| sample.state_bytes)
        .max()
        .unwrap_or(0)
        .max(directory_regular_file_bytes(&state_root).await);
    let visible = read_checkpoint_matrix_visible(&sink_root).await;
    let observed = visible.values().flatten().cloned().collect::<Vec<_>>();
    let unique = observed.iter().cloned().collect::<BTreeSet<_>>();
    let expected = checkpoint_soak_expected_records(selected_terminal);
    let generation_finished_micros = generation_started_micros
        .checked_add(checkpoint_soak_parent_elapsed_micros(
            generation_started.elapsed(),
        )?)
        .ok_or_else(|| checkpoint_soak_process_error("checkpoint soak child clock overflowed"))?;
    Ok(CheckpointSoakProcessReport {
        schema: CHECKPOINT_SOAK_PROCESS_SCHEMA.into(),
        commit: plan.commit.clone(),
        executable_sha256: plan.executable_sha256.clone(),
        config_hash: plan.config_hash.clone(),
        run_root: plan.run_root.clone(),
        parent_pid: plan.parent_pid,
        pid: std::process::id(),
        generation: plan.generation,
        sample_start: plan.sample_start,
        sample_end: plan.sample_end,
        generation_started_micros,
        generation_finished_micros,
        restored_epoch,
        restored_cursor_orders,
        terminal_epoch: selected_terminal.epoch().as_u64(),
        terminal_cursor_orders,
        restored_window_state,
        restored_watermarks,
        restored_progress,
        samples,
        compacted_epochs,
        maximum_manifest_count,
        maximum_state_bytes,
        completed_checkpoints: terminal.completed_checkpoints,
        failed_checkpoints: terminal.failed_checkpoints,
        source_open_events: opened_with.lock().len(),
        source_close_events: source_closed.load(Ordering::SeqCst),
        sink_close_events: sink_closed.load(Ordering::SeqCst),
        terminal_cause: terminal.cause.into(),
        terminal_tasks: terminal.terminal_tasks,
        terminal_charged_edges: terminal.terminal_charged_edges,
        terminal_registries: terminal.terminal_registries,
        source_records: expected.len() / 2,
        output_records: observed.len(),
        duplicate_records: observed.len().saturating_sub(unique.len()),
        missing_records: expected.difference(&unique).count(),
        temporary_artifacts: checkpoint_matrix_temporary_artifacts(
            &state_root,
            &manifest_root,
            &sink_root,
        ),
    })
}

fn checkpoint_soak_cursor_order(value: &str) -> Result<u64> {
    let bytes = hex::decode(value).map_err(|error| {
        checkpoint_soak_process_error(format!("checkpoint soak cursor is not hex: {error}"))
    })?;
    let order: [u8; 8] = bytes.try_into().map_err(|_| {
        checkpoint_soak_process_error("checkpoint soak cursor is not an exact u64 order")
    })?;
    Ok(u64::from_be_bytes(order))
}

fn validate_checkpoint_soak_samples(
    plan: &CheckpointSoakProcessPlan,
    report: &CheckpointSoakProcessReport,
) -> Result<()> {
    let expected = (plan.sample_start..plan.sample_end).collect::<Vec<_>>();
    let observed = report
        .samples
        .iter()
        .map(|sample| sample.index)
        .collect::<Vec<_>>();
    if observed != expected {
        return Err(checkpoint_soak_process_error(format!(
            "checkpoint soak child sample indices are invalid: observed {observed:?}, expected {expected:?}"
        )));
    }
    let cadence_micros = u64::try_from(CHECKPOINT_SOAK_CADENCE.as_micros()).unwrap();
    let tolerance_micros = u64::try_from(CHECKPOINT_SOAK_CADENCE_TOLERANCE.as_micros()).unwrap();
    if let Some(issue) = report
        .samples
        .iter()
        .enumerate()
        .find_map(|(local_index, sample)| {
            checkpoint_soak_sample_issue(
                report,
                local_index,
                sample,
                cadence_micros,
                tolerance_micros,
            )
        })
    {
        return Err(checkpoint_soak_process_error(issue));
    }
    Ok(())
}

fn checkpoint_soak_sample_issue(
    report: &CheckpointSoakProcessReport,
    local_index: usize,
    sample: &CheckpointSoakProcessSample,
    cadence_micros: u64,
    tolerance_micros: u64,
) -> Option<String> {
    let target = cadence_micros * u64::try_from(local_index + 1).unwrap();
    let Some(observed) = sample
        .elapsed_micros
        .checked_sub(report.generation_started_micros)
    else {
        return Some(format!(
            "checkpoint soak generation {} sample {} precedes its generation",
            report.generation, sample.index
        ));
    };
    if observed < target - tolerance_micros || observed > target + tolerance_micros {
        return Some(format!(
            "checkpoint soak generation {} sample {} elapsed {observed}us outside target {target}us +/-{tolerance_micros}us",
            report.generation, sample.index
        ));
    }
    if sample.elapsed_micros > report.generation_finished_micros {
        return Some(format!(
            "checkpoint soak sample {local_index} finishes after its generation"
        ));
    }
    if sample.rss_kib == 0 {
        return Some(format!(
            "checkpoint soak sample {local_index} has no RSS evidence"
        ));
    }
    if sample.task_count == 0 {
        return Some(format!(
            "checkpoint soak sample {local_index} has no live tasks"
        ));
    }
    if sample.failed_checkpoints != 0 {
        return Some(format!(
            "checkpoint soak sample {local_index} records {} failed checkpoints",
            sample.failed_checkpoints
        ));
    }
    if sample.manifest_count > 3 {
        return Some(format!(
            "checkpoint soak sample {local_index} records {} manifests",
            sample.manifest_count
        ));
    }
    if sample.state_bytes > MAX_CHECKPOINT_SOAK_STATE_BYTES {
        return Some(format!(
            "checkpoint soak sample {local_index} records {} state bytes above {}",
            sample.state_bytes, MAX_CHECKPOINT_SOAK_STATE_BYTES
        ));
    }
    None
}

fn validate_checkpoint_soak_advance(report: &CheckpointSoakProcessReport) -> Result<()> {
    let expected_sources = BTreeSet::from(["left", "right"]);
    let restored_sources = report
        .restored_cursor_orders
        .keys()
        .map(String::as_str)
        .collect::<BTreeSet<_>>();
    let terminal_sources = report
        .terminal_cursor_orders
        .keys()
        .map(String::as_str)
        .collect::<BTreeSet<_>>();
    if restored_sources != expected_sources || terminal_sources != expected_sources {
        return Err(checkpoint_soak_process_error(
            "checkpoint soak child source identity set is invalid",
        ));
    }
    for source_id in expected_sources {
        let terminal = checkpoint_soak_cursor_order(&report.terminal_cursor_orders[source_id])?;
        match &report.restored_cursor_orders[source_id] {
            Some(restored) if terminal <= checkpoint_soak_cursor_order(restored)? => {
                return Err(checkpoint_soak_process_error(format!(
                    "checkpoint soak child did not advance source {source_id:?}"
                )));
            }
            None if report.generation != 0 || terminal == 0 => {
                return Err(checkpoint_soak_process_error(format!(
                    "checkpoint soak child has invalid initial cursor for {source_id:?}"
                )));
            }
            _ => {}
        }
    }
    if report
        .restored_epoch
        .is_some_and(|restored| report.terminal_epoch <= restored)
        || report.restored_epoch.is_none() && report.terminal_epoch == 0
    {
        return Err(checkpoint_soak_process_error(
            "checkpoint soak child did not advance the durable epoch",
        ));
    }
    Ok(())
}

fn validate_checkpoint_soak_report(
    plan: &CheckpointSoakProcessPlan,
    report: &CheckpointSoakProcessReport,
    exit_code: i32,
) -> Result<()> {
    let expected_cause = if plan.final_generation {
        "natural_end"
    } else {
        "explicit_cancel"
    };
    if exit_code != 0
        || report.schema != CHECKPOINT_SOAK_PROCESS_SCHEMA
        || report.commit != plan.commit
        || report.executable_sha256 != plan.executable_sha256
        || report.config_hash != plan.config_hash
        || report.run_root != plan.run_root
        || report.parent_pid != plan.parent_pid
        || report.pid == 0
        || report.pid == plan.parent_pid
        || report.generation != plan.generation
        || report.sample_start != plan.sample_start
        || report.sample_end != plan.sample_end
        || report.generation_started_micros != plan.parent_launch_offset_micros
        || report.generation_finished_micros <= report.generation_started_micros
        || report.terminal_cause != expected_cause
        || report.source_open_events != 2
        || report.source_close_events != 2
        || report.sink_close_events != 2
        || report.failed_checkpoints != 0
        || report.maximum_manifest_count > 3
        || report.maximum_state_bytes > MAX_CHECKPOINT_SOAK_STATE_BYTES
        || report.terminal_tasks != 0
        || report.terminal_charged_edges != 0
        || report.terminal_registries != (0, 0)
    {
        return Err(checkpoint_soak_process_error(
            "checkpoint soak child report identity, exit, or terminal bounds are invalid",
        ));
    }
    if plan.final_generation && report.temporary_artifacts != 0 {
        return Err(checkpoint_soak_process_error(format!(
            "final checkpoint soak child retained {} temporary artifacts",
            report.temporary_artifacts
        )));
    }
    if plan.generation == 0 {
        if report.restored_epoch.is_some()
            || report.restored_cursor_orders.values().any(Option::is_some)
        {
            return Err(checkpoint_soak_process_error(
                "initial checkpoint soak child unexpectedly restored in-memory continuity",
            ));
        }
    } else if report.restored_epoch.is_none()
        || report.restored_cursor_orders.values().any(Option::is_none)
        || !report.restored_window_state
        || !report.restored_watermarks
        || !report.restored_progress
    {
        return Err(checkpoint_soak_process_error(
            "restarted checkpoint soak child lacks durable restore evidence",
        ));
    }
    validate_checkpoint_soak_samples(plan, report)?;
    validate_checkpoint_soak_advance(report)
}

fn checkpoint_soak_log_excerpt(path: &Path) -> String {
    std::fs::read(path).map_or_else(
        |error| format!("unavailable: {error}"),
        |bytes| {
            let start = bytes.len().saturating_sub(8 * 1_024);
            String::from_utf8_lossy(&bytes[start..]).into_owned()
        },
    )
}

fn run_checkpoint_soak_process_blocking(
    executable: &Path,
    plan_path: &Path,
    stdout_path: &Path,
    stderr_path: &Path,
    timeout: Duration,
) -> Result<i32> {
    let stdout = OpenOptions::new()
        .create_new(true)
        .write(true)
        .open(stdout_path)
        .map_err(|source| CalcFlowError::Io {
            path: stdout_path.display().to_string(),
            source,
        })?;
    let stderr = OpenOptions::new()
        .create_new(true)
        .write(true)
        .open(stderr_path)
        .map_err(|source| CalcFlowError::Io {
            path: stderr_path.display().to_string(),
            source,
        })?;
    let mut child = Command::new(executable)
        .arg(CHECKPOINT_SOAK_CHILD_TEST)
        .arg("--exact")
        .arg("--nocapture")
        .arg("--test-threads=1")
        .env(CHECKPOINT_SOAK_CHILD_ENV, plan_path)
        .stdin(Stdio::null())
        .stdout(Stdio::from(stdout))
        .stderr(Stdio::from(stderr))
        .spawn()
        .map_err(|source| CalcFlowError::Io {
            path: executable.display().to_string(),
            source,
        })?;
    let started = std::time::Instant::now();
    loop {
        if let Some(status) = child.try_wait().map_err(|source| CalcFlowError::Io {
            path: executable.display().to_string(),
            source,
        })? {
            return status.code().ok_or_else(|| {
                checkpoint_soak_process_error("checkpoint soak child exited without a status code")
            });
        }
        if started.elapsed() >= timeout {
            child.kill().map_err(|source| CalcFlowError::Io {
                path: executable.display().to_string(),
                source,
            })?;
            let _ = child.wait();
            return Err(checkpoint_soak_process_error(format!(
                "checkpoint soak child exceeded {timeout:?}"
            )));
        }
        std::thread::sleep(Duration::from_millis(10));
    }
}

async fn spawn_checkpoint_soak_process(
    executable: &Path,
    plan: &CheckpointSoakProcessPlan,
) -> Result<(i32, CheckpointSoakProcessReport)> {
    let plan_path = checkpoint_soak_plan_path(plan);
    write_checkpoint_soak_document(&plan_path, plan)?;
    let log_root = plan.run_root.join("evidence").join("logs");
    std::fs::create_dir_all(&log_root).map_err(|source| CalcFlowError::Io {
        path: log_root.display().to_string(),
        source,
    })?;
    let stdout_path = log_root.join(format!("generation-{}.stdout", plan.generation));
    let stderr_path = log_root.join(format!("generation-{}.stderr", plan.generation));
    let timeout = match plan.mode {
        CheckpointSoakProcessMode::Smoke => Duration::from_secs(60),
        CheckpointSoakProcessMode::Standard => {
            let samples = u64::try_from(plan.sample_end - plan.sample_start).unwrap();
            CHECKPOINT_SOAK_CADENCE * u32::try_from(samples).unwrap() + Duration::from_secs(60)
        }
    };
    let executable = executable.to_path_buf();
    let blocking_plan = plan_path.clone();
    let blocking_stdout = stdout_path.clone();
    let blocking_stderr = stderr_path.clone();
    let exit_code = tokio::task::spawn_blocking(move || {
        run_checkpoint_soak_process_blocking(
            &executable,
            &blocking_plan,
            &blocking_stdout,
            &blocking_stderr,
            timeout,
        )
    })
    .await
    .map_err(|error| CalcFlowError::Internal {
        message: format!("checkpoint soak child owner task failed: {error}"),
    })??;
    if exit_code != 0 {
        return Err(checkpoint_soak_process_error(format!(
            "checkpoint soak child generation {} exited {exit_code}; stderr: {}",
            plan.generation,
            checkpoint_soak_log_excerpt(&stderr_path)
        )));
    }
    let report = read_checkpoint_soak_document(&checkpoint_soak_report_path(plan))?;
    validate_checkpoint_soak_report(plan, &report, exit_code)?;
    Ok((exit_code, report))
}

fn validate_checkpoint_soak_process_set(
    plans: &[CheckpointSoakProcessPlan],
    reports: &[CheckpointSoakProcessReport],
    exit_codes: &[i32],
    parent_timings: &[CheckpointSoakParentTiming],
) -> Result<()> {
    if plans.len() != CHECKPOINT_SOAK_GENERATIONS
        || reports.len() != plans.len()
        || exit_codes.len() != plans.len()
        || parent_timings.len() != plans.len()
    {
        return Err(checkpoint_soak_process_error(
            "checkpoint soak process set is incomplete",
        ));
    }
    let pids = reports
        .iter()
        .map(|report| report.pid)
        .collect::<BTreeSet<_>>();
    if pids.len() != CHECKPOINT_SOAK_GENERATIONS {
        return Err(checkpoint_soak_process_error(
            "checkpoint soak process IDs are not distinct",
        ));
    }
    for ((plan, report), exit_code) in plans.iter().zip(reports).zip(exit_codes) {
        validate_checkpoint_soak_report(plan, report, *exit_code)?;
    }
    validate_checkpoint_soak_timeline(plans, reports, parent_timings)?;
    for pair in reports.windows(2) {
        let previous = &pair[0];
        let restarted = &pair[1];
        let previous_cursors = previous
            .terminal_cursor_orders
            .iter()
            .map(|(source_id, order)| (source_id.clone(), Some(order.clone())))
            .collect::<BTreeMap<_, _>>();
        if restarted.restored_epoch != Some(previous.terminal_epoch)
            || restarted.restored_cursor_orders != previous_cursors
        {
            return Err(checkpoint_soak_process_error(
                "checkpoint soak child did not continue from prior filesystem evidence",
            ));
        }
    }
    let observed_samples = reports
        .iter()
        .flat_map(|report| report.samples.iter().map(|sample| sample.index))
        .collect::<Vec<_>>();
    let expected_samples = match plans[0].mode {
        CheckpointSoakProcessMode::Smoke => Vec::new(),
        CheckpointSoakProcessMode::Standard => (0..CHECKPOINT_SOAK_SAMPLE_COUNT).collect(),
    };
    if observed_samples != expected_samples {
        return Err(checkpoint_soak_process_error(
            "checkpoint soak process samples contain a gap or duplicate",
        ));
    }
    let final_report = reports.last().unwrap();
    if final_report.duplicate_records != 0 || final_report.missing_records != 0 {
        return Err(checkpoint_soak_process_error(
            "checkpoint soak transactional output is not exactly once",
        ));
    }
    Ok(())
}

fn validate_checkpoint_soak_timeline(
    plans: &[CheckpointSoakProcessPlan],
    reports: &[CheckpointSoakProcessReport],
    parent_timings: &[CheckpointSoakParentTiming],
) -> Result<()> {
    let maximum_restart_gap = u64::try_from(MAX_CHECKPOINT_SOAK_RESTART_GAP.as_micros()).unwrap();
    for ((plan, report), timing) in plans.iter().zip(reports).zip(parent_timings) {
        if timing.generation != plan.generation
            || timing.launch_micros != report.generation_started_micros
            || timing.finish_micros < report.generation_finished_micros
        {
            return Err(checkpoint_soak_process_error(
                "checkpoint soak parent timing does not bound the child report",
            ));
        }
    }
    for pair in parent_timings.windows(2) {
        let Some(gap) = pair[1].launch_micros.checked_sub(pair[0].finish_micros) else {
            return Err(checkpoint_soak_process_error(
                "checkpoint soak process restart gap is invalid",
            ));
        };
        if gap > maximum_restart_gap {
            return Err(checkpoint_soak_process_error(
                "checkpoint soak process restart gap is invalid",
            ));
        }
    }
    if plans[0].mode == CheckpointSoakProcessMode::Smoke {
        return Ok(());
    }
    let samples = reports
        .iter()
        .flat_map(|report| report.samples.iter())
        .collect::<Vec<_>>();
    let tolerance = u64::try_from(CHECKPOINT_SOAK_CADENCE_TOLERANCE.as_micros()).unwrap();
    let cadence = u64::try_from(CHECKPOINT_SOAK_CADENCE.as_micros()).unwrap();
    for pair in samples.windows(2) {
        let Some(gap) = pair[1].elapsed_micros.checked_sub(pair[0].elapsed_micros) else {
            return Err(checkpoint_soak_process_error(
                "checkpoint soak observed cadence or restart gap is invalid",
            ));
        };
        let restart_boundary = CHECKPOINT_SOAK_RESTART_SAMPLES.contains(&pair[0].index);
        let maximum = if restart_boundary {
            cadence + maximum_restart_gap
        } else {
            cadence + tolerance
        };
        if gap < cadence - tolerance || gap > maximum {
            return Err(checkpoint_soak_process_error(
                "checkpoint soak observed cadence or restart gap is invalid",
            ));
        }
    }
    let Some(first) = samples.first() else {
        return Err(checkpoint_soak_process_error(
            "standard checkpoint soak produced no real samples",
        ));
    };
    let last = samples.last().unwrap();
    let maximum_final_elapsed =
        u64::try_from(TARGET_DURATION.as_micros()).unwrap() + 2 * maximum_restart_gap + tolerance;
    if first.elapsed_micros < cadence - tolerance
        || first.elapsed_micros > cadence + tolerance
        || last.elapsed_micros < u64::try_from(TARGET_DURATION.as_micros()).unwrap()
        || last.elapsed_micros > maximum_final_elapsed
    {
        return Err(checkpoint_soak_process_error(
            "checkpoint soak observed timeline does not span the real target duration",
        ));
    }
    Ok(())
}

fn aggregate_checkpoint_soak_processes(
    reports: Vec<CheckpointSoakProcessReport>,
    exit_codes: Vec<i32>,
    parent_timings: Vec<CheckpointSoakParentTiming>,
) -> CheckpointSoakProcessEvidence {
    let final_report = reports.last().unwrap();
    let compacted_epochs = reports
        .iter()
        .flat_map(|report| report.compacted_epochs.iter().copied())
        .collect::<BTreeSet<_>>();
    let rss_samples = reports
        .iter()
        .flat_map(|report| {
            report.samples.iter().map(|sample| RssSample {
                elapsed_seconds: Duration::from_micros(sample.elapsed_micros).as_secs_f64(),
                rss_kib: sample.rss_kib,
            })
        })
        .collect();
    let report = CheckpointRestartSoakReport {
        restarts: reports.len().saturating_sub(1),
        generation_process_ids: reports.iter().map(|report| report.pid).collect(),
        generation_exit_codes: exit_codes,
        completed_epochs: final_report.terminal_epoch,
        compacted_window_epochs: compacted_epochs.len(),
        maximum_manifest_count: reports
            .iter()
            .map(|report| report.maximum_manifest_count)
            .max()
            .unwrap_or(0),
        maximum_state_bytes: reports
            .iter()
            .map(|report| report.maximum_state_bytes)
            .max()
            .unwrap_or(0),
        source_records: final_report.source_records,
        output_records: final_report.output_records,
        duplicate_records: final_report.duplicate_records,
        missing_records: final_report.missing_records,
        temporary_artifacts: final_report.temporary_artifacts,
        terminal_tasks: reports.iter().map(|report| report.terminal_tasks).sum(),
        terminal_charged_edges: reports
            .iter()
            .map(|report| report.terminal_charged_edges)
            .sum(),
        terminal_registries: reports.iter().fold((0, 0), |total, report| {
            (
                total.0 + report.terminal_registries.0,
                total.1 + report.terminal_registries.1,
            )
        }),
    };
    CheckpointSoakProcessEvidence {
        report,
        processes: reports,
        parent_timings,
        rss_samples,
    }
}

async fn run_checkpoint_soak_processes(
    run_root: &Path,
    mode: CheckpointSoakProcessMode,
) -> Result<CheckpointSoakProcessEvidence> {
    let run_root = run_root
        .canonicalize()
        .map_err(|source| CalcFlowError::Io {
            path: run_root.display().to_string(),
            source,
        })?;
    let executable = checkpoint_soak_test_executable()?;
    let executable_sha256 = checkpoint_soak_file_sha256(&executable)?;
    let commit = strict_command_output("git", &["rev-parse", "HEAD"])?;
    let mut plans = checkpoint_soak_process_plans(&run_root, &commit, &executable_sha256, mode);
    let mut reports = Vec::with_capacity(plans.len());
    let mut exit_codes = Vec::with_capacity(plans.len());
    let mut parent_timings = Vec::with_capacity(plans.len());
    let parent_started = std::time::Instant::now();
    for plan in &mut plans {
        bind_checkpoint_soak_parent_launch(plan, parent_started.elapsed())?;
        let launch_micros = plan.parent_launch_offset_micros;
        let (exit_code, report) = spawn_checkpoint_soak_process(&executable, plan).await?;
        parent_timings.push(CheckpointSoakParentTiming {
            generation: plan.generation,
            launch_micros,
            finish_micros: checkpoint_soak_parent_elapsed_micros(parent_started.elapsed())?,
        });
        exit_codes.push(exit_code);
        reports.push(report);
    }
    validate_checkpoint_soak_process_set(&plans, &reports, &exit_codes, &parent_timings)?;
    Ok(aggregate_checkpoint_soak_processes(
        reports,
        exit_codes,
        parent_timings,
    ))
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

async fn run_checkpoint_restart_soak_smoke() -> CheckpointRestartSoakReport {
    let directory = tempfile::tempdir().unwrap();
    run_checkpoint_soak_processes(directory.path(), CheckpointSoakProcessMode::Smoke)
        .await
        .unwrap()
        .report
}

#[cfg(target_os = "linux")]
fn checkpoint_soak_sample_max(
    evidence: &CheckpointSoakProcessEvidence,
    select: impl Fn(&CheckpointSoakProcessSample) -> usize,
) -> usize {
    evidence
        .processes
        .iter()
        .flat_map(|process| process.samples.iter().map(&select))
        .max()
        .unwrap_or(0)
}

#[cfg(target_os = "linux")]
fn checkpoint_soak_process_summary(
    commit: &str,
    evidence: &CheckpointSoakProcessEvidence,
    rss_gate: RssGate,
) -> serde_json::Value {
    let report = &evidence.report;
    json!({
        "schema": "calc-flow.m5-checkpoint-soak.v1",
        "type": "calc_flow_m5_checkpoint_soak_summary",
        "commit": commit,
        "target_duration_seconds": 1_200,
        "sample_count": evidence.rss_samples.len(),
        "restart_sample_indices": CHECKPOINT_SOAK_RESTART_SAMPLES,
        "restarts": report.restarts,
        "processes": evidence.processes.iter().zip(&evidence.parent_timings).map(|(process, timing)| json!({
            "generation": process.generation,
            "pid": process.pid,
            "exit_code": report.generation_exit_codes[process.generation],
            "sample_range": [process.sample_start, process.sample_end],
            "generation_started_seconds": Duration::from_micros(
                process.generation_started_micros,
            ).as_secs_f64(),
            "generation_finished_seconds": Duration::from_micros(
                process.generation_finished_micros,
            ).as_secs_f64(),
            "parent_launch_seconds": Duration::from_micros(timing.launch_micros).as_secs_f64(),
            "parent_finish_seconds": Duration::from_micros(timing.finish_micros).as_secs_f64(),
            "restored_epoch": process.restored_epoch,
            "terminal_epoch": process.terminal_epoch,
            "config_hash": process.config_hash,
        })).collect::<Vec<_>>(),
        "completed_epochs": report.completed_epochs,
        "compacted_window_epochs": report.compacted_window_epochs,
        "resource_bounds": {
            "maximum_task_count": checkpoint_soak_sample_max(evidence, |sample| sample.task_count),
            "maximum_queue_depth": checkpoint_soak_sample_max(
                evidence,
                |sample| sample.maximum_queue_depth,
            ),
            "maximum_charged_rows": checkpoint_soak_sample_max(
                evidence,
                |sample| sample.maximum_charged_rows,
            ),
            "maximum_charged_bytes": checkpoint_soak_sample_max(
                evidence,
                |sample| sample.maximum_charged_bytes,
            ),
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
        "source_open_events": evidence.processes.iter()
            .map(|process| process.source_open_events).sum::<usize>(),
        "source_close_events": evidence.processes.iter()
            .map(|process| process.source_close_events).sum::<usize>(),
        "sink_close_events": evidence.processes.iter()
            .map(|process| process.sink_close_events).sum::<usize>(),
    })
}

#[cfg(target_os = "linux")]
fn checkpoint_soak_evidence_directory() -> tempfile::TempDir {
    let target_root = std::env::var_os("CARGO_TARGET_DIR").map_or_else(
        || {
            PathBuf::from(strict_command_output("git", &["rev-parse", "--show-toplevel"]).unwrap())
                .join("target")
        },
        PathBuf::from,
    );
    std::fs::create_dir_all(&target_root).unwrap();
    tempfile::Builder::new()
        .prefix("m5-checkpoint-soak-")
        .tempdir_in(target_root)
        .unwrap()
}

#[cfg(target_os = "linux")]
async fn run_checkpoint_restart_linux_soak() {
    let directory = checkpoint_soak_evidence_directory();
    let commit = strict_command_output("git", &["rev-parse", "HEAD"]).unwrap();
    println!(
        "{}",
        checkpoint_restart_soak_metadata(
            &commit,
            &command_output("uname", &["-sr"]),
            &command_output("rustc", &["--version"]),
        )
    );
    let evidence =
        match run_checkpoint_soak_processes(directory.path(), CheckpointSoakProcessMode::Standard)
            .await
        {
            Ok(evidence) => evidence,
            Err(error) => {
                let retained = directory.keep();
                panic!(
                    "checkpoint soak evidence retained at {}: {error}",
                    retained.display()
                );
            }
        };
    let evidence_root = directory.keep();
    println!(
        "{}",
        json!({
            "schema": "calc-flow.m5-checkpoint-soak-evidence-root.v1",
            "type": "calc_flow_m5_checkpoint_soak_evidence_root",
            "commit": &commit,
            "path": &evidence_root,
        })
    );
    let report = &evidence.report;
    assert_eq!(report.restarts, 2);
    assert_eq!(report.generation_process_ids.len(), 3);
    assert!(report.generation_exit_codes.iter().all(|code| *code == 0));
    assert!(report.compacted_window_epochs > 0);
    assert!(report.maximum_manifest_count <= 3);
    assert_eq!(report.duplicate_records, 0);
    assert_eq!(report.missing_records, 0);
    assert_eq!(report.temporary_artifacts, 0);
    assert_eq!(report.terminal_tasks, 0);
    assert_eq!(report.terminal_charged_edges, 0);
    assert_eq!(report.terminal_registries, (0, 0));
    assert_eq!(observed_timeline_issue(&evidence.rss_samples), None);
    let rss_gate = evaluate_checkpoint_restart_rss_gate(&evidence.rss_samples)
        .expect("checkpoint soak RSS samples incomplete");
    assert!(
        rss_gate.passed,
        "checkpoint soak RSS guard failed: {rss_gate:?}"
    );
    println!(
        "{}",
        checkpoint_soak_process_summary(&commit, &evidence, rss_gate)
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

fn checkpoint_soak_process_report_fixture(
    plan: &CheckpointSoakProcessPlan,
    pid: u32,
) -> CheckpointSoakProcessReport {
    let generation_started_micros = plan.parent_launch_offset_micros;
    let restored_order = u64::try_from(plan.generation).unwrap() * 10;
    let terminal_order = restored_order + 10;
    let restored_cursor_orders = ["left", "right"]
        .into_iter()
        .map(|source_id| {
            (
                source_id.into(),
                (plan.generation > 0).then(|| hex::encode(restored_order.to_be_bytes())),
            )
        })
        .collect();
    let terminal_cursor_orders = ["left", "right"]
        .into_iter()
        .map(|source_id| (source_id.into(), hex::encode(terminal_order.to_be_bytes())))
        .collect();
    let samples: Vec<CheckpointSoakProcessSample> = (plan.sample_start..plan.sample_end)
        .enumerate()
        .map(|(local_index, index)| CheckpointSoakProcessSample {
            index,
            elapsed_micros: generation_started_micros
                + u64::try_from(local_index + 1).unwrap()
                    * u64::try_from(CHECKPOINT_SOAK_CADENCE.as_micros()).unwrap(),
            rss_kib: 100_000,
            task_count: 8,
            maximum_queue_depth: 1,
            maximum_charged_rows: 1,
            maximum_charged_bytes: 1,
            completed_checkpoints: 1,
            failed_checkpoints: 0,
            manifest_count: 2,
            state_bytes: 1,
        })
        .collect();
    CheckpointSoakProcessReport {
        schema: CHECKPOINT_SOAK_PROCESS_SCHEMA.into(),
        commit: plan.commit.clone(),
        executable_sha256: plan.executable_sha256.clone(),
        config_hash: plan.config_hash.clone(),
        run_root: plan.run_root.clone(),
        parent_pid: plan.parent_pid,
        pid,
        generation: plan.generation,
        sample_start: plan.sample_start,
        sample_end: plan.sample_end,
        generation_started_micros,
        generation_finished_micros: samples
            .last()
            .map_or(generation_started_micros + 250_000, |sample| {
                sample.elapsed_micros + 250_000
            }),
        restored_epoch: (plan.generation > 0).then_some(restored_order),
        restored_cursor_orders,
        terminal_epoch: terminal_order,
        terminal_cursor_orders,
        restored_window_state: plan.generation > 0,
        restored_watermarks: plan.generation > 0,
        restored_progress: plan.generation > 0,
        samples,
        compacted_epochs: vec![terminal_order],
        maximum_manifest_count: 2,
        maximum_state_bytes: 1,
        completed_checkpoints: 1,
        failed_checkpoints: 0,
        source_open_events: 2,
        source_close_events: 2,
        sink_close_events: 2,
        terminal_cause: if plan.final_generation {
            "natural_end".into()
        } else {
            "explicit_cancel".into()
        },
        terminal_tasks: 0,
        terminal_charged_edges: 0,
        terminal_registries: (0, 0),
        source_records: 20,
        output_records: 40,
        duplicate_records: 0,
        missing_records: 0,
        temporary_artifacts: 0,
    }
}

#[test]
#[allow(
    clippy::too_many_lines,
    reason = "one fail-closed evidence test mutates every independent process proof field"
)]
fn checkpoint_restart_process_evidence_fails_closed() {
    let directory = tempfile::tempdir().unwrap();
    let mut plans = checkpoint_soak_process_plans(
        directory.path(),
        &"a".repeat(40),
        &"b".repeat(64),
        CheckpointSoakProcessMode::Standard,
    );
    for plan in &mut plans {
        let generation = u64::try_from(plan.generation).unwrap();
        bind_checkpoint_soak_parent_launch(
            plan,
            Duration::from_micros(250_000 + generation * 400_500_000),
        )
        .unwrap();
    }
    let reports = plans
        .iter()
        .enumerate()
        .map(|(generation, plan)| {
            checkpoint_soak_process_report_fixture(
                plan,
                10_000 + u32::try_from(generation).unwrap(),
            )
        })
        .collect::<Vec<_>>();
    let exits = vec![0; CHECKPOINT_SOAK_GENERATIONS];
    let parent_timings = reports
        .iter()
        .map(|report| CheckpointSoakParentTiming {
            generation: report.generation,
            launch_micros: report.generation_started_micros,
            finish_micros: report.generation_finished_micros + 250_000,
        })
        .collect::<Vec<_>>();
    validate_checkpoint_soak_process_set(&plans, &reports, &exits, &parent_timings).unwrap();

    let mut bounded_scheduler_jitter = reports.clone();
    bounded_scheduler_jitter[0].samples[0].elapsed_micros += 1_100_000;
    validate_checkpoint_soak_process_set(
        &plans,
        &bounded_scheduler_jitter,
        &exits,
        &parent_timings,
    )
    .unwrap();

    let mut abandoned_nonterminal_staging = reports.clone();
    abandoned_nonterminal_staging[0].temporary_artifacts = 7;
    validate_checkpoint_soak_process_set(
        &plans,
        &abandoned_nonterminal_staging,
        &exits,
        &parent_timings,
    )
    .unwrap();

    let mut repeated_pid = reports.clone();
    repeated_pid[1].pid = repeated_pid[0].pid;
    assert!(matches!(
        validate_checkpoint_soak_process_set(&plans, &repeated_pid, &exits, &parent_timings),
        Err(CalcFlowError::InvalidArgument { message, .. })
            if message.contains("not distinct")
    ));
    let mut bad_exit = exits.clone();
    bad_exit[1] = 9;
    assert!(
        validate_checkpoint_soak_process_set(&plans, &reports, &bad_exit, &parent_timings).is_err()
    );

    let mut wrong_identity = reports.clone();
    wrong_identity[1].config_hash = "c".repeat(64);
    assert!(
        validate_checkpoint_soak_process_set(&plans, &wrong_identity, &exits, &parent_timings)
            .is_err()
    );
    let mut wrong_head = reports.clone();
    wrong_head[1].commit = "d".repeat(40);
    assert!(
        validate_checkpoint_soak_process_set(&plans, &wrong_head, &exits, &parent_timings).is_err()
    );
    let mut wrong_root = reports.clone();
    wrong_root[1].run_root.push("different-root");
    assert!(
        validate_checkpoint_soak_process_set(&plans, &wrong_root, &exits, &parent_timings).is_err()
    );
    let mut wrong_generation = reports.clone();
    wrong_generation[1].generation = 2;
    assert!(
        validate_checkpoint_soak_process_set(&plans, &wrong_generation, &exits, &parent_timings)
            .is_err()
    );
    let mut in_memory_only = reports.clone();
    in_memory_only[1].restored_epoch = None;
    assert!(
        validate_checkpoint_soak_process_set(&plans, &in_memory_only, &exits, &parent_timings)
            .is_err()
    );

    let mut sample_gap = reports.clone();
    sample_gap[1].samples.remove(0);
    assert!(
        validate_checkpoint_soak_process_set(&plans, &sample_gap, &exits, &parent_timings).is_err()
    );
    let mut sample_duplicate = reports.clone();
    let duplicate = sample_duplicate[0].samples[39].clone();
    sample_duplicate[1].samples.insert(0, duplicate);
    assert!(
        validate_checkpoint_soak_process_set(&plans, &sample_duplicate, &exits, &parent_timings)
            .is_err()
    );

    let mut mistimed_sample = reports.clone();
    mistimed_sample[0].samples[5].elapsed_micros += 3_000_000;
    assert!(matches!(
        validate_checkpoint_soak_process_set(
            &plans,
            &mistimed_sample,
            &exits,
            &parent_timings,
        ),
        Err(CalcFlowError::InvalidArgument { message, .. })
            if message.contains(
                "generation 0 sample 5 elapsed 63000000us outside target 60000000us +/-2000000us"
            )
    ));

    let mut unbounded_gap_reports = reports.clone();
    let mut unbounded_gap_timings = parent_timings.clone();
    let shifted_launch = parent_timings[0].finish_micros
        + u64::try_from(MAX_CHECKPOINT_SOAK_RESTART_GAP.as_micros()).unwrap()
        + 1_000_000;
    let shift = shifted_launch - unbounded_gap_reports[1].generation_started_micros;
    unbounded_gap_reports[1].generation_started_micros += shift;
    unbounded_gap_reports[1].generation_finished_micros += shift;
    for sample in &mut unbounded_gap_reports[1].samples {
        sample.elapsed_micros += shift;
    }
    unbounded_gap_timings[1].launch_micros += shift;
    unbounded_gap_timings[1].finish_micros += shift;
    assert!(matches!(
        validate_checkpoint_soak_timeline(
            &plans,
            &unbounded_gap_reports,
            &unbounded_gap_timings,
        ),
        Err(CalcFlowError::InvalidArgument { message, .. })
            if message.contains("restart gap")
    ));

    let mut forged_parent_timings = parent_timings.clone();
    forged_parent_timings[1].finish_micros = reports[1].generation_finished_micros - 10_000;
    assert!(matches!(
        validate_checkpoint_soak_timeline(&plans, &reports, &forged_parent_timings),
        Err(CalcFlowError::InvalidArgument { message, .. })
            if message.contains("does not bound")
    ));
}

#[test]
fn checkpoint_restart_process_document_rejects_missing_and_malformed_reports() {
    let directory = tempfile::tempdir().unwrap();
    let mut plan = checkpoint_soak_process_plans(
        directory.path(),
        &"a".repeat(40),
        &"b".repeat(64),
        CheckpointSoakProcessMode::Standard,
    )
    .remove(0);
    bind_checkpoint_soak_parent_launch(&mut plan, Duration::from_micros(1_234_567)).unwrap();
    let report = checkpoint_soak_process_report_fixture(&plan, 10_000);
    let round_trip = directory.path().join("round-trip.json");
    write_checkpoint_soak_document(&round_trip, &report).unwrap();
    assert_eq!(
        read_checkpoint_soak_document::<CheckpointSoakProcessReport>(&round_trip).unwrap(),
        report,
    );

    let missing = directory.path().join("missing.json");
    let missing_result: Result<CheckpointSoakProcessReport> =
        read_checkpoint_soak_document(&missing);
    assert!(missing_result.is_err());

    let malformed = directory.path().join("malformed.json");
    std::fs::write(&malformed, b"{not-json}").unwrap();
    let malformed_result: Result<CheckpointSoakProcessReport> =
        read_checkpoint_soak_document(&malformed);
    assert!(matches!(
        malformed_result,
        Err(CalcFlowError::InvalidArgument { message, .. })
            if message.contains("malformed")
    ));
}

#[tokio::test]
async fn checkpoint_restart_soak_generation_child_process() {
    let Some(plan_path) = std::env::var_os(CHECKPOINT_SOAK_CHILD_ENV).map(PathBuf::from) else {
        return;
    };
    let plan: CheckpointSoakProcessPlan = read_checkpoint_soak_document(&plan_path).unwrap();
    let report = run_checkpoint_soak_child(&plan).await.unwrap();
    write_checkpoint_soak_document(&checkpoint_soak_report_path(&plan), &report).unwrap();
}

#[tokio::test]
async fn checkpoint_restart_soak_smoke_exercises_retention_and_compaction() {
    let report = run_checkpoint_restart_soak_smoke().await;

    assert_eq!(report.restarts, 2);
    assert_eq!(report.generation_process_ids.len(), 3);
    assert_eq!(report.generation_exit_codes, [0, 0, 0]);
    assert_eq!(
        report
            .generation_process_ids
            .iter()
            .copied()
            .collect::<BTreeSet<_>>()
            .len(),
        3,
        "every restart generation must run in a distinct OS process"
    );
    assert!(
        report
            .generation_process_ids
            .iter()
            .all(|pid| *pid != std::process::id()),
        "the parent test process must not execute a soak generation"
    );
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
    assert_eq!(metadata["cadence_tolerance_seconds"], 2);
    assert_eq!(metadata["maximum_restart_gap_seconds"], 60);
    assert_eq!(
        metadata["timing_source"],
        "parent_std_instant_plus_child_local_instant"
    );
    assert_eq!(metadata["restart_sample_indices"], json!([39, 79]));
    assert_eq!(metadata["restart_kind"], "os_process");
    assert_eq!(metadata["process_generations"], 3);
    assert_eq!(
        metadata["process_sample_ranges"],
        json!([[0, 40], [40, 80], [80, 120]])
    );
    assert_eq!(
        metadata["child_report_schema"],
        CHECKPOINT_SOAK_PROCESS_SCHEMA
    );
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
async fn private_benchmark_smoke_runs_twelve_honest_cases() {
    let private_paths = run_private_checkpoint_path_benchmark_smoke().await;
    assert_eq!(
        private_paths,
        BTreeMap::from([
            ("m5/private_path/barrier_cut_single_source", 1),
            ("m5/private_path/barrier_cut_two_source_fan_out", 4),
            ("m5/private_path/pass_through_two_input_alignment", 1),
            ("m5/private_path/window_two_input_alignment", 3),
            ("m5/private_path/dirty_window_state_stage", 2),
            ("m5/private_path/non_empty_manifest_publication", 1),
            ("m5/private_path/retained_delta_compacted_base_restore", 3),
            ("m5/private_path/single_transactional_sink_commit", 1),
            ("m5/private_path/multi_transactional_sink_commit", 2),
        ])
    );

    for (mode, expected_manifests) in [
        (PrivateFullRunnerMode::NoCheckpoint, 0),
        (PrivateFullRunnerMode::CheckpointDisabled, 1),
        (PrivateFullRunnerMode::CheckpointEnabled, 2),
    ] {
        let report = prepare_private_full_runner_benchmark(mode)
            .await
            .measure()
            .await;
        assert_eq!(report.completed_jobs, 1);
        assert_eq!(report.failed_jobs, 0);
        assert_eq!(report.visible_records, 4);
        if expected_manifests == 2 {
            assert!(report.manifest_count >= expected_manifests);
        } else {
            assert_eq!(report.manifest_count, expected_manifests);
        }
    }
}

#[tokio::test(start_paused = true)]
#[allow(
    clippy::too_many_lines,
    reason = "the late periodic race proves one epoch across live completion and terminal restart"
)]
async fn late_periodic_cut_promotes_the_same_epoch_to_terminal() {
    let directory = tempfile::tempdir().unwrap();
    let sink_root = directory.path().join("sinks");
    let manifest_root = directory.path().join("manifests");
    let backend = Arc::new(
        LocalStateBackend::new(directory.path().join("state"))
            .await
            .unwrap(),
    );
    let source_opens = Arc::new(Mutex::new(Vec::new()));
    let source_closed = Arc::new(AtomicUsize::new(0));
    let sink_closed = Arc::new(AtomicUsize::new(0));
    let sink_writes = Arc::new(AtomicUsize::new(0));
    let gate = CheckpointStartedTestGate::default();
    let config = StreamRuntimeConfig {
        checkpoint_interval: Duration::from_millis(1),
        checkpoint_timeout: Duration::from_secs(10),
        retained_epochs: 2,
        ..StreamRuntimeConfig::default()
    };
    let spec = checkpoint_matrix_spec(
        30_001,
        &sink_root,
        &source_opens,
        &source_closed,
        &sink_closed,
        &sink_writes,
        Duration::from_secs(1),
        false,
        CancellationToken::new(),
        true,
    );
    let checkpoint = CheckpointRuntimeSpec::new(
        Arc::clone(&backend) as Arc<dyn StateBackend>,
        &manifest_root,
        config,
    )
    .unwrap()
    .with_started_gate(gate.clone());
    let mut runner = ContinuousRunner::new();
    let job = runner.start_checkpointed(spec, checkpoint).await.unwrap();

    tokio::task::yield_now().await;
    for _ in 0..5 {
        if gate.has_entered() {
            break;
        }
        tokio::time::advance(Duration::from_millis(1)).await;
        tokio::task::yield_now().await;
    }
    assert!(
        gate.has_entered(),
        "periodic Started boundary was not reached"
    );
    let started = job.status();
    let started_checkpoint = started.checkpoint.as_ref().unwrap();
    assert_eq!(started_checkpoint.current_epoch, Some(Epoch::INITIAL));
    assert!(!started_checkpoint.terminal);
    assert!(!started.sources.values().all(|source| source.ended));
    for _ in 0..3 {
        tokio::time::advance(Duration::from_secs(1)).await;
        tokio::task::yield_now().await;
    }
    assert!(job.status().sources.values().all(|source| source.ended));
    gate.release();

    let outcome = tokio::time::timeout(Duration::from_secs(5), job.wait())
        .await
        .expect("the promoted terminal checkpoint hung");
    assert_eq!(outcome.state, ContinuousJobState::Completed);
    let status = job.status();
    let checkpoint = status.checkpoint.as_ref().unwrap();
    assert_eq!(checkpoint.current_epoch, None);
    assert_eq!(checkpoint.phase, None);
    assert!(!checkpoint.terminal);
    assert_eq!(checkpoint.last_completed_epoch, Some(Epoch::INITIAL));
    assert_eq!(status.metrics.checkpoints.requested, 1);
    assert_eq!(status.metrics.checkpoints.terminal_requested, 1);
    assert_eq!(status.metrics.checkpoints.completed, 1);
    assert_eq!(status.metrics.checkpoints.terminal_completed, 1);
    assert_eq!(status.metrics.checkpoints.failed, 0);
    assert!(status.tasks.is_empty());
    assert!(status.edges.values().all(|edge| edge.queue_depth == 0));
    drop(job);
    runner.shutdown().await.unwrap();
    assert_eq!(runner.registry_counts(), (0, 0));

    let manifests = checkpoint_manifest_documents(&manifest_root).await;
    assert_eq!(manifests.len(), 1);
    let manifest = &manifests[0];
    assert_eq!(manifest.epoch(), Epoch::INITIAL);
    assert!(manifest.sources().values().all(|source| source.ended));
    assert!(manifest.operators().values().all(|operator| {
        operator
            .progress
            .values()
            .all(|ingress| matches!(ingress.state, crate::ManifestIngressState::Ended))
    }));
    assert_eq!(
        read_checkpoint_matrix_visible(&sink_root)
            .await
            .values()
            .map(Vec::len)
            .sum::<usize>(),
        4,
    );
    assert_eq!(source_opens.lock().len(), 2);
    assert_eq!(source_closed.load(Ordering::SeqCst), 2);
    assert_eq!(sink_closed.load(Ordering::SeqCst), 2);

    let restart_spec = checkpoint_matrix_spec(
        30_002,
        &sink_root,
        &source_opens,
        &source_closed,
        &sink_closed,
        &sink_writes,
        Duration::ZERO,
        false,
        CancellationToken::new(),
        true,
    );
    let restart_checkpoint =
        CheckpointRuntimeSpec::new(backend as Arc<dyn StateBackend>, &manifest_root, config)
            .unwrap();
    let mut restart_runner = ContinuousRunner::new();
    let restart_job = restart_runner
        .start_checkpointed(restart_spec, restart_checkpoint)
        .await
        .unwrap();
    let restart_outcome = tokio::time::timeout(Duration::from_secs(5), restart_job.wait())
        .await
        .expect("terminal restart hung");
    assert_eq!(restart_outcome.state, ContinuousJobState::Completed);
    assert_eq!(source_opens.lock().len(), 2);
    assert_eq!(source_closed.load(Ordering::SeqCst), 2);
    assert_eq!(sink_closed.load(Ordering::SeqCst), 4);
    assert_eq!(checkpoint_manifest_documents(&manifest_root).await.len(), 1);
    assert_eq!(
        read_checkpoint_matrix_visible(&sink_root)
            .await
            .values()
            .map(Vec::len)
            .sum::<usize>(),
        4,
    );
    drop(restart_job);
    restart_runner.shutdown().await.unwrap();
    assert_eq!(restart_runner.registry_counts(), (0, 0));
}

#[test]
fn private_benchmark_errors_name_the_candidate_only_gate() {
    let CalcFlowError::InvalidArgument { field, .. } = benchmark_evidence_error("invalid") else {
        panic!("benchmark evidence error must be an invalid argument");
    };

    assert_eq!(field, "CALC_FLOW_M5_CHECKPOINT_BENCHMARK");
}

#[test]
fn private_full_runner_self_overhead_modes_use_identical_sink_pacing() {
    assert_eq!(
        private_full_runner_sink_delay(PrivateFullRunnerMode::CheckpointDisabled),
        private_full_runner_sink_delay(PrivateFullRunnerMode::CheckpointEnabled),
    );
    assert_eq!(
        private_full_runner_sink_delay(PrivateFullRunnerMode::CheckpointEnabled),
        Duration::from_millis(5),
    );
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
fn private_benchmark_metadata_reports_honest_absolute_only_m5_cases() {
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
            "m5/private_path/barrier_cut_single_source",
            "m5/private_path/barrier_cut_two_source_fan_out",
            "m5/private_path/pass_through_two_input_alignment",
            "m5/private_path/window_two_input_alignment",
            "m5/private_path/dirty_window_state_stage",
            "m5/private_path/non_empty_manifest_publication",
            "m5/private_path/retained_delta_compacted_base_restore",
            "m5/private_path/single_transactional_sink_commit",
            "m5/private_path/multi_transactional_sink_commit",
            "m5/private_full_path/no_checkpoint",
            "m5/private_full_path/checkpoint_disabled",
            "m5/private_full_path/checkpoint_enabled",
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
            .contains("cargo test --release --locked")
    );
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
    assert_eq!(metadata["measurements"].as_array().unwrap().len(), 12);
    assert_eq!(metadata["overall_result"], "absolute_only");
}

#[test]
fn private_benchmark_contract_uses_honest_cases_and_batched_setup() {
    assert_eq!(
        M5_PRIVATE_BENCHMARK_CASES.as_slice(),
        [
            "m5/private_path/barrier_cut_single_source",
            "m5/private_path/barrier_cut_two_source_fan_out",
            "m5/private_path/pass_through_two_input_alignment",
            "m5/private_path/window_two_input_alignment",
            "m5/private_path/dirty_window_state_stage",
            "m5/private_path/non_empty_manifest_publication",
            "m5/private_path/retained_delta_compacted_base_restore",
            "m5/private_path/single_transactional_sink_commit",
            "m5/private_path/multi_transactional_sink_commit",
            "m5/private_full_path/no_checkpoint",
            "m5/private_full_path/checkpoint_disabled",
            "m5/private_full_path/checkpoint_enabled",
        ]
    );
    let source = include_str!("soak.rs");
    assert_eq!(
        source
            .lines()
            .filter(|line| line.trim_start().starts_with("bencher.iter_batched("))
            .count(),
        12
    );
    assert!(source.contains("BatchSize::SmallInput"));
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
    // The recorded artifact path uses platform separators.
    let sample_path = measurement["artifacts"]["sample"]["path"]
        .as_str()
        .unwrap()
        .replace('\\', "/");
    assert!(sample_path.ends_with("candidate-sha/sample.json"));
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
        bencher.iter_batched(
            || prepare_private_barrier_cut_benchmark(&["source"], 1),
            |fixture| black_box(runtime.block_on(fixture.measure())),
            BatchSize::SmallInput,
        );
    });
    criterion.bench_function(M5_PRIVATE_BENCHMARK_CASES[1], |bencher| {
        bencher.iter_batched(
            || prepare_private_barrier_cut_benchmark(&["left", "right"], 2),
            |fixture| black_box(runtime.block_on(fixture.measure())),
            BatchSize::SmallInput,
        );
    });
    criterion.bench_function(M5_PRIVATE_BENCHMARK_CASES[2], |bencher| {
        bencher.iter_batched(
            || {
                runtime.block_on(prepare_private_alignment_benchmark(
                    PrivateAlignmentKind::PassThrough,
                ))
            },
            |fixture| black_box(runtime.block_on(fixture.measure())),
            BatchSize::SmallInput,
        );
    });
    criterion.bench_function(M5_PRIVATE_BENCHMARK_CASES[3], |bencher| {
        bencher.iter_batched(
            || {
                runtime.block_on(prepare_private_alignment_benchmark(
                    PrivateAlignmentKind::Window,
                ))
            },
            |fixture| black_box(runtime.block_on(fixture.measure())),
            BatchSize::SmallInput,
        );
    });
    criterion.bench_function(M5_PRIVATE_BENCHMARK_CASES[4], |bencher| {
        bencher.iter_batched(
            || runtime.block_on(prepare_private_state_stage_benchmark()),
            |fixture| black_box(runtime.block_on(fixture.measure())),
            BatchSize::SmallInput,
        );
    });
    criterion.bench_function(M5_PRIVATE_BENCHMARK_CASES[5], |bencher| {
        bencher.iter_batched(
            || runtime.block_on(prepare_private_manifest_publication_benchmark()),
            |fixture| black_box(runtime.block_on(fixture.measure())),
            BatchSize::SmallInput,
        );
    });
    criterion.bench_function(M5_PRIVATE_BENCHMARK_CASES[6], |bencher| {
        bencher.iter_batched(
            || runtime.block_on(prepare_private_retained_restore_benchmark()),
            |fixture| black_box(fixture.measure()),
            BatchSize::SmallInput,
        );
    });
    criterion.bench_function(M5_PRIVATE_BENCHMARK_CASES[7], |bencher| {
        bencher.iter_batched(
            || runtime.block_on(prepare_private_sink_commit_benchmark(1)),
            |fixture| black_box(runtime.block_on(fixture.measure())),
            BatchSize::SmallInput,
        );
    });
    criterion.bench_function(M5_PRIVATE_BENCHMARK_CASES[8], |bencher| {
        bencher.iter_batched(
            || runtime.block_on(prepare_private_sink_commit_benchmark(2)),
            |fixture| black_box(runtime.block_on(fixture.measure())),
            BatchSize::SmallInput,
        );
    });
    criterion.bench_function(M5_PRIVATE_BENCHMARK_CASES[9], |bencher| {
        bencher.iter_batched(
            || {
                runtime.block_on(prepare_private_full_runner_benchmark(
                    PrivateFullRunnerMode::NoCheckpoint,
                ))
            },
            |fixture| black_box(runtime.block_on(fixture.measure())),
            BatchSize::SmallInput,
        );
    });
    criterion.bench_function(M5_PRIVATE_BENCHMARK_CASES[10], |bencher| {
        bencher.iter_batched(
            || {
                runtime.block_on(prepare_private_full_runner_benchmark(
                    PrivateFullRunnerMode::CheckpointDisabled,
                ))
            },
            |fixture| black_box(runtime.block_on(fixture.measure())),
            BatchSize::SmallInput,
        );
    });
    criterion.bench_function(M5_PRIVATE_BENCHMARK_CASES[11], |bencher| {
        bencher.iter_batched(
            || {
                runtime.block_on(prepare_private_full_runner_benchmark(
                    PrivateFullRunnerMode::CheckpointEnabled,
                ))
            },
            |fixture| black_box(runtime.block_on(fixture.measure())),
            BatchSize::SmallInput,
        );
    });
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
async fn partial_sink_commit_fault_preserves_the_second_sink_for_forward_recovery() {
    for mode in CheckpointFaultMode::ALL {
        let report =
            run_checkpoint_restart_fault_case(CheckpointFaultPoint::PartialSinkCommit, mode).await;

        assert_eq!(
            report.committed_sinks_before_restart,
            BTreeSet::from(["sink-a".into()]),
            "first sink must commit before the injected fault at {mode:?}"
        );
        assert_eq!(
            report.prepared_sinks_before_restart,
            BTreeSet::from(["sink-b".into()]),
            "second sink must remain prepared for forward recovery at {mode:?}"
        );
        assert_eq!(report.restored_epoch, Some(Epoch::INITIAL.next().unwrap()));
        assert_eq!(
            report.restart_source_opens,
            vec![
                ("left".into(), Some(41_u64.to_be_bytes().to_vec())),
                ("right".into(), Some(73_u64.to_be_bytes().to_vec())),
            ]
        );
        assert_eq!(
            report.final_manifest_cursors["left"].order,
            hex::encode(41_u64.to_be_bytes())
        );
        assert!(report.final_manifest_cursors["left"].payload.is_empty());
        assert_eq!(
            report.final_manifest_cursors["right"].order,
            hex::encode(73_u64.to_be_bytes())
        );
        assert!(report.final_manifest_cursors["right"].payload.is_empty());
        assert_eq!(report.visible_records, 4);
        assert_eq!(report.duplicate_records, 0);
        assert_eq!(report.missing_records, 0);
    }
}

#[tokio::test]
#[allow(
    clippy::too_many_lines,
    reason = "the fault matrix asserts every recovery invariant for each point and mode"
)]
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
                match point {
                    CheckpointFaultPoint::ManifestRename => 2,
                    CheckpointFaultPoint::PartialSinkCommit => 1,
                    _ => 0,
                },
                "prepared-artifact preservation mismatch at {point:?}/{mode:?}"
            );
            if point == CheckpointFaultPoint::PartialSinkCommit {
                assert_eq!(
                    report.committed_sinks_before_restart,
                    BTreeSet::from(["sink-a".into()]),
                    "first sink must commit before the injected fault at {mode:?}"
                );
                assert_eq!(
                    report.prepared_sinks_before_restart,
                    BTreeSet::from(["sink-b".into()]),
                    "second sink must remain prepared for forward recovery at {mode:?}"
                );
            }
            assert_eq!(
                report.restored_epoch,
                Some(if durable_before_restart {
                    Epoch::INITIAL.next().unwrap()
                } else {
                    Epoch::INITIAL
                }),
                "final epoch mismatch at {point:?}/{mode:?}"
            );
            assert_eq!(
                report.restart_source_opens,
                vec![
                    (
                        "left".into(),
                        durable_before_restart.then(|| 41_u64.to_be_bytes().to_vec()),
                    ),
                    (
                        "right".into(),
                        durable_before_restart.then(|| 73_u64.to_be_bytes().to_vec()),
                    ),
                ],
                "restart connectors must open once at their exact durable cursors"
            );
            assert_eq!(
                report.restored_cursor_orders,
                BTreeMap::from([
                    ("left".into(), 41_u64.to_be_bytes().to_vec()),
                    ("right".into(), 73_u64.to_be_bytes().to_vec()),
                ]),
                "restored source cursor mismatch at {point:?}/{mode:?}"
            );
            assert_eq!(
                report.final_manifest_cursors["left"].order,
                hex::encode(41_u64.to_be_bytes()),
                "final left cursor order mismatch at {point:?}/{mode:?}"
            );
            assert!(report.final_manifest_cursors["left"].payload.is_empty());
            assert_eq!(
                report.final_manifest_cursors["right"].order,
                hex::encode(73_u64.to_be_bytes()),
                "final right cursor order mismatch at {point:?}/{mode:?}"
            );
            assert!(report.final_manifest_cursors["right"].payload.is_empty());
            assert!(
                report.sources_ended,
                "sources not ended at {point:?}/{mode:?}"
            );
            assert!(
                report.watermarks_restored,
                "watermarks not restored at {point:?}/{mode:?}"
            );
            assert_eq!(
                report.window_state_restored, durable_before_restart,
                "window restore evidence mismatch at {point:?}/{mode:?}"
            );
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
        sink_root.join("sink-a/visible-00000000000000000001.json"),
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

#[tokio::test]
async fn checkpoint_matrix_sink_commits_bounded_epoch_files() {
    let directory = tempfile::tempdir().unwrap();
    let root = directory.path().join("sink-a");
    let mut sink = CheckpointMatrixSink {
        sink_id: "sink-a",
        root: root.clone(),
        epoch: None,
        pending: Vec::new(),
        written_records: Arc::new(AtomicUsize::new(0)),
        closed: Arc::new(AtomicUsize::new(0)),
        write_delay: Duration::ZERO,
    };
    sink.open().await.unwrap();
    let epochs = [Epoch::INITIAL, Epoch::INITIAL.next().unwrap()];
    for (epoch, record) in epochs.into_iter().zip(["first", "second"]) {
        sink.begin_epoch(epoch).await.unwrap();
        sink.pending.push(record.into());
        let state = sink.pre_commit(epoch).await.unwrap();
        sink.commit(epoch, &state).await.unwrap();
    }

    assert!(!root.join("visible.json").exists());
    assert!(root.join("visible-00000000000000000001.json").exists());
    assert!(root.join("visible-00000000000000000002.json").exists());
    assert_eq!(
        read_checkpoint_matrix_visible(directory.path()).await["sink-a"],
        ["first", "second"]
    );
}
