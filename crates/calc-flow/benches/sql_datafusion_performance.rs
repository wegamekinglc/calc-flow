//! Same-binary Calc Flow SQL versus raw `DataFusion` performance evidence.

#![allow(
    clippy::cast_possible_truncation,
    clippy::cast_precision_loss,
    clippy::cast_sign_loss,
    reason = "benchmark summaries convert bounded sample counts and nanosecond metrics to f64"
)]

use std::{
    collections::BTreeMap,
    error::Error,
    fs,
    path::{Path, PathBuf},
    process::Command,
    sync::Arc,
    thread,
    time::{Duration, Instant},
};

use calc_flow::{
    Batch, BatchMetadata, DATAFUSION_ACTIVE_ENTITIES_METADATA_KEY, DataFusionConfig,
    DataFusionParallelismMode, ExecutionOptions, PipelineBuilder, SqlOperator, UdfRegistry,
};
use datafusion::{
    arrow::{
        array::{
            Array, Float64Array, Float64Builder, StringArray, StringBuilder,
            TimestampMicrosecondArray, TimestampMicrosecondBuilder, UInt64Array, UInt64Builder,
        },
        compute::concat_batches,
        datatypes::{DataType, Field, Schema, TimeUnit},
        record_batch::RecordBatch,
    },
    datasource::MemTable,
    execution::context::{SessionConfig, SessionContext},
    physical_plan::{
        ExecutionPlan, ExecutionPlanProperties, displayable, execute_stream, metrics::MetricValue,
    },
    sql::parser::DFParser,
};
use futures::{StreamExt, TryStreamExt};
use serde::Serialize;
use serde_json::json;
use sha2::{Digest, Sha256};

const DATAFUSION_VERSION: &str = "54.0.0";
const ARROW_VERSION: &str = "58.3.0";
const DEFAULT_BATCH_SIZE: usize = 8_192;
const DEFAULT_ENTITIES: usize = 64;
const DEFAULT_SAMPLES: usize = 20;
const DEFAULT_WARMUPS: usize = 1;
const MIN_ROWS_PER_PARTITION: usize = 65_536;
const SMALL_ROWS_THRESHOLD: usize = 10_001;
const BOOTSTRAP_RESAMPLES: usize = 20_000;
const RTOL: f64 = 1e-10;
const ATOL: f64 = 1e-10;

type BenchResult<T> = Result<T, Box<dyn Error + Send + Sync>>;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum Profile {
    SerialControl,
    MatchedAdaptive,
    P32Saturation,
    Matrix,
    Attribution,
}

impl Profile {
    fn parse(value: &str) -> BenchResult<Self> {
        match value {
            "serial-control" => Ok(Self::SerialControl),
            "matched-adaptive" => Ok(Self::MatchedAdaptive),
            "p32-saturation" => Ok(Self::P32Saturation),
            "matrix" => Ok(Self::Matrix),
            "attribution" => Ok(Self::Attribution),
            _ => Err(format!("unsupported profile {value:?}").into()),
        }
    }

    const fn as_str(self) -> &'static str {
        match self {
            Self::SerialControl => "serial-control",
            Self::MatchedAdaptive => "matched-adaptive",
            Self::P32Saturation => "p32-saturation",
            Self::Matrix => "matrix",
            Self::Attribution => "attribution",
        }
    }

    const fn default_rows(self) -> usize {
        match self {
            Self::P32Saturation => 2_097_152,
            Self::Matrix => 100_000,
            Self::SerialControl | Self::MatchedAdaptive | Self::Attribution => 1_000_000,
        }
    }

    const fn default_partitions(self) -> usize {
        match self {
            Self::SerialControl => 1,
            Self::Matrix => 4,
            Self::MatchedAdaptive | Self::P32Saturation | Self::Attribution => 32,
        }
    }
}

#[derive(Debug)]
struct Args {
    profile: Profile,
    output: PathBuf,
    rows: usize,
    entities: usize,
    batch_size: usize,
    partitions: usize,
    samples: usize,
    warmups: usize,
}

impl Args {
    fn parse() -> BenchResult<Self> {
        // The closed flag parser rejects unknown options and validates every
        // numeric value before benchmark construction.
        // #lizard forgives
        let mut values = std::env::args().skip(1); // nosemgrep
        let mut profile = Profile::MatchedAdaptive;
        let mut output = None;
        let mut rows = None;
        let mut entities = None;
        let mut batch_size = None;
        let mut partitions = None;
        let mut samples = None;
        let mut warmups = None;
        while let Some(flag) = values.next() {
            if flag == "--bench" {
                continue;
            }
            let value = values
                .next()
                .ok_or_else(|| format!("{flag} requires one value"))?;
            match flag.as_str() {
                "--profile" => profile = Profile::parse(&value)?,
                "--output" => output = Some(PathBuf::from(value)),
                "--rows" => rows = Some(parse_positive(&value, "rows")?),
                "--entities" => entities = Some(parse_positive(&value, "entities")?),
                "--batch-size" => batch_size = Some(parse_positive(&value, "batch-size")?),
                "--partitions" => partitions = Some(parse_positive(&value, "partitions")?),
                "--samples" => samples = Some(parse_positive(&value, "samples")?),
                "--warmups" => warmups = Some(parse_positive(&value, "warmups")?),
                _ => return Err(format!("unknown argument {flag:?}").into()),
            }
        }
        let rows = rows.unwrap_or_else(|| profile.default_rows());
        let entities = entities.unwrap_or(DEFAULT_ENTITIES);
        if entities > rows {
            return Err("entities must not exceed rows".into());
        }
        Ok(Self {
            profile,
            output: resolve_output_path(&output.ok_or("--output is required")?)?,
            rows,
            entities,
            batch_size: batch_size.unwrap_or(DEFAULT_BATCH_SIZE),
            partitions: partitions.unwrap_or_else(|| profile.default_partitions()),
            samples: samples.unwrap_or(DEFAULT_SAMPLES),
            warmups: warmups.unwrap_or(DEFAULT_WARMUPS),
        })
    }
}

fn parse_positive(value: &str, name: &str) -> BenchResult<usize> {
    let parsed = value.parse::<usize>()?;
    if parsed == 0 {
        return Err(format!("{name} must be positive").into());
    }
    Ok(parsed)
}

#[derive(Serialize)]
struct Report {
    schema_version: u32,
    git_sha: String,
    profile: String,
    environment: Environment,
    cases: Vec<CaseEvidence>,
}

#[derive(Serialize)]
struct Environment {
    machine_fingerprint: String,
    dependency_fingerprint: String,
    workload_fingerprint: String,
    datafusion_version: String,
    arrow_version: String,
    build_profile: String,
    allocator: String,
    os: String,
    arch: String,
    cpu_model: String,
    available_parallelism: usize,
    rust_version: String,
    git_dirty: bool,
}

#[derive(Serialize)]
struct CaseEvidence {
    name: String,
    rows: usize,
    active_entities: usize,
    window: usize,
    /// The workload's true output cardinality: equal to `rows` only for
    /// row-preserving queries; `filter` shrinks it and `group_by` collapses it
    /// to one row per entity.
    output_rows: usize,
    warmups: usize,
    rolling_rewrite_enabled: bool,
    sample_order: Vec<String>,
    calc_flow: EngineEvidence,
    raw_datafusion: EngineEvidence,
    paired_ratios: Vec<f64>,
    paired_ratio_median: f64,
    paired_ratio_ci_low: f64,
    paired_ratio_ci_high: f64,
    correctness: Correctness,
    comparability: Comparability,
    speedup_conclusion: Option<String>,
}

#[derive(Serialize)]
struct EngineEvidence {
    parallelism_mode: String,
    configured_partitions: usize,
    requested_partitions: usize,
    effective_partitions: usize,
    available_parallelism: usize,
    max_partitions: usize,
    min_rows_per_partition: usize,
    small_rows_threshold: usize,
    parallelism_decision_reused: bool,
    decision_input_rows: usize,
    decision_active_entities: Option<usize>,
    decision_active_entities_source: String,
    partition_limit_reason: String,
    batch_size: usize,
    input_logical_partitions: usize,
    input_batch_rows: Vec<usize>,
    normalized_plan_hash: String,
    bounded_window_agg_count: usize,
    samples_ms: Vec<f64>,
    median_ms: f64,
    p25_ms: f64,
    p75_ms: f64,
    mad_ms: f64,
    cv: f64,
    cpu_time_ms: f64,
    peak_rss_bytes: usize,
    spill_bytes: usize,
    empty_partitions: usize,
    partition_rows: Vec<usize>,
    partition_skew: f64,
    window_compute_ms: f64,
    repartition_sort_compute_ms: f64,
    window_operator_count: usize,
    repartition_operator_count: usize,
    sort_operator_count: usize,
    coalesce_operator_count: usize,
    phase_medians_ms: PhaseMedians,
    phase_samples_ms: BTreeMap<String, Vec<f64>>,
}

#[derive(Clone, Default, Serialize)]
struct PhaseMedians {
    runtime_acquire: f64,
    session_state_create: f64,
    input_adapter: f64,
    table_register: f64,
    sql_parse: f64,
    logical_optimize: f64,
    physical_plan: f64,
    execution_to_first_batch: f64,
    execution_remaining: f64,
    collect_or_coalesce: f64,
    output_arrow_wrap: f64,
    audit: f64,
    metrics_traversal: f64,
    physical_plan_string: f64,
    batch_envelope: f64,
    run_result: f64,
    run_session_envelope: f64,
}

#[derive(Serialize)]
#[allow(
    clippy::struct_excessive_bools,
    reason = "the evidence schema records each independent correctness contract explicitly"
)]
struct Correctness {
    schema: bool,
    rows: bool,
    keys: bool,
    order: bool,
    null_nan_mask: bool,
    values: bool,
    rtol: f64,
    atol: f64,
}

#[derive(Serialize)]
struct Comparability {
    comparable: bool,
    mismatches: Vec<String>,
}

/// Selects the canonical output row identity used for correctness alignment.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum RowIdentity {
    /// The unique `(symbol, event_time, sequence)` input key is preserved.
    Row,
    /// One output row per grouping key, keyed by `symbol` alone.
    Group,
}

struct Workload {
    name: &'static str,
    window: usize,
    output_column: &'static str,
    identity: RowIdentity,
    /// The SQL joins the `dimension` side input beside `input`.
    dimension: bool,
    sql: String,
}

struct Sample {
    elapsed_ms: f64,
    phases: PhaseMedians,
    plan_text: String,
    configured_partitions: usize,
    requested_partitions: usize,
    effective_partitions: usize,
    parallelism_mode: String,
    available_parallelism: usize,
    max_partitions: usize,
    min_rows_per_partition: usize,
    small_rows_threshold: usize,
    parallelism_decision_reused: bool,
    decision_input_rows: usize,
    decision_active_entities: Option<usize>,
    decision_active_entities_source: String,
    partition_limit_reason: String,
    batch_size: usize,
    partition_rows: Vec<usize>,
    spill_bytes: usize,
    elapsed_compute_ns: usize,
    window_compute_ns: usize,
    repartition_sort_compute_ns: usize,
    window_operator_count: usize,
    repartition_operator_count: usize,
    sort_operator_count: usize,
    coalesce_operator_count: usize,
    peak_rss_bytes: usize,
}

#[derive(Default)]
struct PlanStatistics {
    partition_rows: Vec<usize>,
    window_partition_rows: Vec<usize>,
    spill_bytes: usize,
    elapsed_compute_ns: usize,
    window_compute_ns: usize,
    repartition_sort_compute_ns: usize,
    window_operator_count: usize,
    repartition_operator_count: usize,
    sort_operator_count: usize,
    coalesce_operator_count: usize,
}

// The sampler reads the kernel-maintained peak-RSS high-water mark instead of
// polling `VmRSS` from a helper thread: a polling thread competes with the
// measured execution and can miss transient peaks between polls, while the
// kernel counter is exact and costs nothing inside the timed window.
struct RssSampler {
    start_rss_bytes: usize,
}

impl RssSampler {
    fn start() -> Self {
        reset_peak_rss_window();
        Self {
            start_rss_bytes: current_rss_bytes(),
        }
    }

    fn finish(self) -> usize {
        peak_rss_bytes().max(self.start_rss_bytes).max(1)
    }
}

fn status_field_kilobytes(prefix: &str) -> Option<usize> {
    fs::read_to_string("/proc/self/status")
        .ok()?
        .lines()
        .find_map(|line| {
            line.strip_prefix(prefix)?
                .split_whitespace()
                .next()?
                .parse::<usize>()
                .ok()
        })
}

fn status_field_bytes(prefix: &str) -> usize {
    status_field_kilobytes(prefix)
        .and_then(|kilobytes| kilobytes.checked_mul(1_024))
        .unwrap_or(1)
}

fn current_rss_bytes() -> usize {
    status_field_bytes("VmRSS:")
}

fn peak_rss_bytes() -> usize {
    status_field_bytes("VmHWM:")
}

// Writing `5` to `clear_refs` re-anchors `VmHWM` at the current footprint so
// each sample reports its own peak. A failed reset (for example on a hardened
// host) only makes later windows cumulative; the reported maximum stays a
// valid process peak.
fn reset_peak_rss_window() {
    let _ = fs::write("/proc/self/clear_refs", b"5");
}

fn workloads() -> [Workload; 6] {
    let average = |window: usize| {
        format!(
            "avg(price) OVER (PARTITION BY symbol ORDER BY event_time, sequence ROWS BETWEEN {} PRECEDING AND CURRENT ROW)",
            window - 1
        )
    };
    [
        Workload {
            name: "sma_20",
            window: 20,
            output_column: "sma_20",
            identity: RowIdentity::Row,
            dimension: false,
            sql: format!(
                "SELECT event_time, sequence, symbol, price, {} AS sma_20 FROM input",
                average(20)
            ),
        },
        Workload {
            name: "dual_sma_spread",
            window: 20,
            output_column: "sma_spread",
            identity: RowIdentity::Row,
            dimension: false,
            sql: format!(
                "SELECT event_time, sequence, symbol, price, ({}) - ({}) AS sma_spread FROM input",
                average(5),
                average(20)
            ),
        },
        // The operator scenarios keep the paired engines on identical physical
        // plans: `filter` uses a float predicate because the Calc Flow-only
        // UInt64 modulo specialization would diverge the normalized plan hash.
        Workload {
            name: "projection",
            window: 0,
            output_column: "value",
            identity: RowIdentity::Row,
            dimension: false,
            sql: "SELECT event_time, sequence, symbol, price * 2 + 1 AS value FROM input"
                .to_owned(),
        },
        Workload {
            name: "filter",
            window: 0,
            output_column: "value",
            identity: RowIdentity::Row,
            dimension: false,
            sql: "SELECT event_time, sequence, symbol, price AS value FROM input \
                  WHERE price > 105.0"
                .to_owned(),
        },
        Workload {
            name: "group_by",
            window: 0,
            output_column: "value",
            identity: RowIdentity::Group,
            dimension: false,
            sql: "SELECT symbol, SUM(price) AS value FROM input GROUP BY symbol".to_owned(),
        },
        Workload {
            name: "join",
            window: 0,
            output_column: "value",
            identity: RowIdentity::Row,
            dimension: true,
            sql: "SELECT event_time, sequence, symbol, price * factor AS value \
                  FROM input JOIN dimension USING (symbol)"
                .to_owned(),
        },
    ]
}

fn input_batches(rows: usize, entities: usize, batch_size: usize) -> BenchResult<Vec<RecordBatch>> {
    let schema = Arc::new(Schema::new(vec![
        Field::new(
            "event_time",
            DataType::Timestamp(TimeUnit::Microsecond, None),
            false,
        ),
        Field::new("sequence", DataType::UInt64, false),
        Field::new("symbol", DataType::Utf8, false),
        Field::new("price", DataType::Float64, false),
    ]));
    let symbols = (0..entities)
        .map(|entity| format!("S{entity:03}"))
        .collect::<Vec<_>>();
    let mut batches = Vec::with_capacity(rows.div_ceil(batch_size));
    for offset in (0..rows).step_by(batch_size) {
        let length = batch_size.min(rows - offset);
        let mut event_time = TimestampMicrosecondBuilder::with_capacity(length);
        let mut sequence = UInt64Builder::with_capacity(length);
        let mut symbol = StringBuilder::with_capacity(length, length.saturating_mul(4));
        let mut price = Float64Builder::with_capacity(length);
        for row in offset..offset + length {
            let entity = row % entities;
            let position = row / entities;
            event_time
                .append_value(1_767_225_600_000_000_i64 + i64::try_from(position)? * 1_000_000);
            sequence.append_value(u64::try_from(row)?);
            symbol.append_value(&symbols[entity]);
            price.append_value(
                100.0
                    + f64::from(u32::try_from((row * 17) % 1_000)?) / 100.0
                    + f64::from(u32::try_from(entity)?) / 100.0,
            );
        }
        batches.push(RecordBatch::try_new(
            Arc::clone(&schema),
            vec![
                Arc::new(event_time.finish()),
                Arc::new(sequence.finish()),
                Arc::new(symbol.finish()),
                Arc::new(price.finish()),
            ],
        )?);
    }
    Ok(batches)
}

fn benchmark_batch(batches: Vec<RecordBatch>, entities: usize) -> BenchResult<Batch> {
    let metadata = BatchMetadata::new(
        "sql-datafusion-performance",
        0,
        BTreeMap::from([(
            DATAFUSION_ACTIVE_ENTITIES_METADATA_KEY.to_owned(),
            json!(entities),
        )]),
    )?;
    Ok(Batch::table(batches, metadata)?)
}

/// Builds the per-symbol `dimension` side input for the join workload with
/// the same `S###` symbols and `entity + 1` factors as the Python suite.
fn dimension_record_batch(entities: usize) -> BenchResult<RecordBatch> {
    let schema = Arc::new(Schema::new(vec![
        Field::new("symbol", DataType::Utf8, false),
        Field::new("factor", DataType::Float64, false),
    ]));
    let mut symbol = StringBuilder::with_capacity(entities, entities.saturating_mul(4));
    let mut factor = Float64Builder::with_capacity(entities);
    for entity in 0..entities {
        symbol.append_value(format!("S{entity:03}"));
        factor.append_value(f64::from(u32::try_from(entity)?) + 1.0);
    }
    Ok(RecordBatch::try_new(
        schema,
        vec![Arc::new(symbol.finish()), Arc::new(factor.finish())],
    )?)
}

fn benchmark_dimension_batch(entities: usize) -> BenchResult<Batch> {
    Ok(Batch::table(
        vec![dimension_record_batch(entities)?],
        BatchMetadata::default(),
    )?)
}

fn config(args: &Args) -> DataFusionConfig {
    DataFusionConfig {
        batch_size: args.batch_size,
        target_partitions: args.partitions,
        min_rows_per_partition: MIN_ROWS_PER_PARTITION,
        small_rows_threshold: SMALL_ROWS_THRESHOLD,
        enable_rolling_rewrite: false,
        collect_diagnostics: true,
        ..DataFusionConfig::default()
    }
}

fn effective_partitions(args: &Args, dimension_rows: usize) -> usize {
    // The engine derives its fixed-mode work cap from every input table's
    // rows combined, so the raw mirror must include the dimension side.
    args.partitions.min(
        (args.rows + dimension_rows)
            .div_ceil(MIN_ROWS_PER_PARTITION)
            .max(1),
    )
}

fn build_plan(
    config: DataFusionConfig,
    workload: &Workload,
) -> BenchResult<calc_flow::BatchExecutionPlan> {
    let mut aliases = vec!["input".to_owned()];
    if workload.dimension {
        aliases.push("dimension".to_owned());
    }
    let operator = SqlOperator::new(workload.name, &workload.sql, aliases, Vec::new())?;
    Ok(PipelineBuilder::new(workload.name)?
        .with_datafusion_config(config)
        .add_node("sql", Box::new(operator))?
        .compile_batch(&UdfRegistry::new().snapshot())?)
}

// `retain_output` selects whether the sample keeps its output batch for the
// warm-up correctness pair; timed samples drop theirs so a long run does not
// accumulate one output per sample. The timed envelope is identical either way.
async fn calc_flow_sample(
    plan: &calc_flow::BatchExecutionPlan,
    batch: &Batch,
    dimension: Option<&Batch>,
    retain_output: bool,
) -> BenchResult<(Sample, Option<Batch>)> {
    let rss = RssSampler::start();
    let started = Instant::now();
    let mut inputs = BTreeMap::from([("input".to_owned(), batch.clone())]);
    if let Some(dimension) = dimension {
        inputs.insert("dimension".to_owned(), dimension.clone());
    }
    let result = plan.execute(inputs, ExecutionOptions::default()).await?;
    let elapsed_ms = milliseconds(started.elapsed());
    let peak_rss_bytes = rss.finish();
    let metric = result
        .datafusion_metrics
        .first()
        .ok_or("Calc Flow sample produced no DataFusion metric")?;
    let output = result
        .outputs
        .get("output")
        .ok_or("Calc Flow sample produced no output")?;
    let output = if retain_output {
        Some(output.clone())
    } else {
        None
    };
    let mut phases = PhaseMedians {
        runtime_acquire: ns_ms(metric.runtime_acquire_ns),
        session_state_create: ns_ms(metric.session_state_create_ns),
        input_adapter: ns_ms(metric.input_adapter_ns),
        table_register: ns_ms(metric.table_register_ns),
        sql_parse: ns_ms(metric.sql_parse_ns),
        logical_optimize: ns_ms(metric.logical_planning_ns),
        physical_plan: ns_ms(metric.physical_planning_ns),
        execution_to_first_batch: ns_ms(metric.execution_to_first_batch_ns),
        execution_remaining: ns_ms(metric.execution_remaining_ns),
        collect_or_coalesce: ns_ms(metric.collect_ns),
        output_arrow_wrap: ns_ms(metric.output_arrow_wrap_ns),
        audit: ns_ms(metric.audit_ns),
        metrics_traversal: ns_ms(metric.metrics_traversal_ns),
        physical_plan_string: ns_ms(metric.physical_plan_string_ns),
        batch_envelope: ns_ms(metric.batch_envelope_ns),
        run_result: ns_ms(metric.run_result_ns),
        run_session_envelope: 0.0,
    };
    phases.run_session_envelope = (elapsed_ms - exclusive_phase_total(&phases)).max(0.0);
    Ok((
        Sample {
            elapsed_ms,
            phases,
            plan_text: metric.physical_plan.clone(),
            configured_partitions: metric.configured_target_partitions,
            requested_partitions: metric.requested_target_partitions,
            effective_partitions: metric.effective_target_partitions,
            parallelism_mode: match metric.parallelism_mode {
                DataFusionParallelismMode::Fixed => "fixed",
                DataFusionParallelismMode::Auto => "auto",
            }
            .to_owned(),
            available_parallelism: metric.available_parallelism,
            max_partitions: metric.max_partitions,
            min_rows_per_partition: metric.min_rows_per_partition,
            small_rows_threshold: metric.small_rows_threshold,
            parallelism_decision_reused: metric.parallelism_decision_reused,
            decision_input_rows: metric.decision_input_rows,
            decision_active_entities: metric.decision_active_entities,
            decision_active_entities_source: metric.decision_active_entities_source.clone(),
            partition_limit_reason: metric.partition_limit_reason.clone(),
            batch_size: metric.configured_batch_size,
            partition_rows: if metric.window_partition_rows.is_empty() {
                metric.output_partition_rows.clone()
            } else {
                metric.window_partition_rows.clone()
            },
            spill_bytes: metric.spill_bytes,
            elapsed_compute_ns: metric.elapsed_compute_ns,
            window_compute_ns: metric.window_compute_ns,
            repartition_sort_compute_ns: metric.repartition_sort_compute_ns,
            window_operator_count: metric.window_operator_count,
            repartition_operator_count: metric.repartition_operator_count,
            sort_operator_count: metric.sort_operator_count,
            coalesce_operator_count: metric.coalesce_operator_count,
            peak_rss_bytes,
        },
        output,
    ))
}

#[allow(
    clippy::too_many_lines,
    reason = "all raw DataFusion phase boundaries stay adjacent for attribution"
)]
async fn collect_first_batch(
    stream: &mut datafusion::physical_plan::SendableRecordBatchStream,
) -> BenchResult<(Option<RecordBatch>, f64)> {
    let execution_start = Instant::now();
    let first = stream.next().await.transpose()?;
    Ok((first, milliseconds(execution_start.elapsed())))
}

async fn collect_remaining_batches(
    stream: &mut datafusion::physical_plan::SendableRecordBatchStream,
) -> BenchResult<(Vec<RecordBatch>, f64)> {
    let remaining_start = Instant::now();
    let remaining = stream.try_collect::<Vec<_>>().await?;
    Ok((remaining, milliseconds(remaining_start.elapsed())))
}

struct RawPlan {
    dataframe: datafusion::dataframe::DataFrame,
    sql_parse_ms: f64,
    logical_optimize_ms: f64,
    physical_plan_ms: f64,
    physical_plan: Arc<dyn ExecutionPlan>,
}

async fn raw_plan_sql(context: &SessionContext, workload: &Workload) -> BenchResult<RawPlan> {
    let parse_start = Instant::now();
    let _statements = DFParser::parse_sql(&workload.sql)?;
    let sql_parse_ms = milliseconds(parse_start.elapsed());
    let logical_start = Instant::now();
    let dataframe = context.sql(&workload.sql).await?;
    let logical_optimize_ms = milliseconds(logical_start.elapsed());
    let (physical_plan, physical_plan_ms) = create_physical_plan(&dataframe).await?;
    Ok(RawPlan {
        dataframe,
        sql_parse_ms,
        logical_optimize_ms,
        physical_plan_ms,
        physical_plan,
    })
}

struct RawCollected {
    first: Option<RecordBatch>,
    remaining: Vec<RecordBatch>,
    first_batch_ms: f64,
    remaining_ms: f64,
    total_ms: f64,
}

async fn raw_execute(
    physical_plan: &Arc<dyn ExecutionPlan>,
    dataframe: &datafusion::dataframe::DataFrame,
) -> BenchResult<RawCollected> {
    let execution_start = Instant::now();
    let mut stream = execute_stream(Arc::clone(physical_plan), Arc::new(dataframe.task_ctx()))?;
    let (first, first_batch_ms) = collect_first_batch(&mut stream).await?;
    let (remaining, remaining_ms) = collect_remaining_batches(&mut stream).await?;
    Ok(RawCollected {
        first,
        remaining,
        first_batch_ms,
        remaining_ms,
        total_ms: milliseconds(execution_start.elapsed()),
    })
}

struct WrappedOutput {
    batch: Batch,
    arrow_wrap_ms: f64,
    envelope_ms: f64,
}

fn wrap_output_batch(
    first: Option<RecordBatch>,
    remaining: Vec<RecordBatch>,
) -> BenchResult<WrappedOutput> {
    let output_wrap_start = Instant::now();
    let mut batches = Vec::with_capacity(remaining.len() + usize::from(first.is_some()));
    batches.extend(first);
    batches.extend(remaining);
    if batches.is_empty() {
        batches.push(RecordBatch::new_empty(Arc::new(Schema::empty())));
    }
    let arrow_wrap_ms = milliseconds(output_wrap_start.elapsed());
    let envelope_start = Instant::now();
    let batch = Batch::table(batches, BatchMetadata::default())?;
    Ok(WrappedOutput {
        batch,
        arrow_wrap_ms,
        envelope_ms: milliseconds(envelope_start.elapsed()),
    })
}

async fn create_physical_plan(
    dataframe: &datafusion::dataframe::DataFrame,
) -> BenchResult<(Arc<dyn ExecutionPlan>, f64)> {
    let physical_start = Instant::now();
    let physical_plan = dataframe.create_physical_plan().await?;
    Ok((physical_plan, milliseconds(physical_start.elapsed())))
}

fn raw_session(
    batch: &Batch,
    dimension: Option<&Batch>,
    batch_size: usize,
    effective_partitions: usize,
) -> BenchResult<(SessionContext, f64, f64, f64, f64)> {
    let runtime_start = Instant::now();
    let session_config = SessionConfig::new()
        .with_batch_size(batch_size)
        .with_target_partitions(effective_partitions);
    let runtime_acquire = milliseconds(runtime_start.elapsed());
    let session_start = Instant::now();
    let context = SessionContext::new_with_config(session_config);
    let session_state_create = milliseconds(session_start.elapsed());
    let input_start = Instant::now();
    let provider = input_mem_table(batch)?;
    let dimension_provider = match dimension {
        Some(value) => Some(input_mem_table(value)?),
        None => None,
    };
    let input_adapter = milliseconds(input_start.elapsed());
    let register_start = Instant::now();
    context.register_table("input", Arc::new(provider))?;
    if let Some(provider) = dimension_provider {
        context.register_table("dimension", Arc::new(provider))?;
    }
    let table_register = milliseconds(register_start.elapsed());
    Ok((
        context,
        runtime_acquire,
        session_state_create,
        input_adapter,
        table_register,
    ))
}

async fn raw_datafusion_sample(
    batch: &Batch,
    dimension: Option<&Batch>,
    workload: &Workload,
    configured_partitions: usize,
    effective_partitions: usize,
    batch_size: usize,
    retain_output: bool,
) -> BenchResult<(Sample, Option<Batch>)> {
    // All raw-engine phase boundaries stay adjacent so the attribution report
    // cannot mix timings from different execution envelopes.
    // #lizard forgives
    let rss = RssSampler::start();
    let total_start = Instant::now();
    let (context, runtime_acquire, session_state_create, input_adapter, table_register) =
        raw_session(batch, dimension, batch_size, effective_partitions)?;
    let plan = raw_plan_sql(&context, workload).await?;
    let dataframe = plan.dataframe;
    let physical_plan = plan.physical_plan;
    let plan_string_start = Instant::now();
    let plan_text = displayable(physical_plan.as_ref()).indent(true).to_string();
    let physical_plan_string = milliseconds(plan_string_start.elapsed());
    let metrics_plan = Arc::clone(&physical_plan);
    let collected = raw_execute(&physical_plan, &dataframe).await?;
    let sql_parse = plan.sql_parse_ms;
    let logical_optimize = plan.logical_optimize_ms;
    let physical_plan_ms = plan.physical_plan_ms;
    let execution_to_first_batch = collected.first_batch_ms;
    let execution_remaining = collected.remaining_ms;
    let collect_or_coalesce = collected.total_ms;
    let output = wrap_output_batch(collected.first, collected.remaining)?;
    let output_arrow_wrap = output.arrow_wrap_ms;
    let batch_envelope = output.envelope_ms;
    let output = output.batch;
    let metrics_start = Instant::now();
    let plan_statistics = plan_statistics(metrics_plan.as_ref(), output.num_rows());
    let metrics_traversal = milliseconds(metrics_start.elapsed());
    let elapsed_ms = milliseconds(total_start.elapsed());
    let peak_rss_bytes = rss.finish();
    let available_parallelism = thread::available_parallelism()
        .map(std::num::NonZero::get)
        .unwrap_or(1);
    // The engine's parallelism decision reads every registered input at once:
    // with a dimension side it reports combined rows and no single-input
    // entity fact, and the raw mirror reports the identical decision fields.
    let decision_input_rows = raw_decision_input_rows(batch, dimension);
    let (decision_active_entities, decision_active_entities_source) =
        raw_decision_entities(batch, dimension);
    let mut phases = PhaseMedians {
        runtime_acquire,
        session_state_create,
        input_adapter,
        table_register,
        sql_parse,
        logical_optimize,
        physical_plan: physical_plan_ms,
        execution_to_first_batch,
        execution_remaining,
        collect_or_coalesce,
        output_arrow_wrap,
        audit: 0.0,
        metrics_traversal,
        physical_plan_string,
        batch_envelope,
        run_result: 0.0,
        run_session_envelope: 0.0,
    };
    phases.run_session_envelope = (elapsed_ms - exclusive_phase_total(&phases)).max(0.0);
    Ok((
        Sample {
            elapsed_ms,
            phases,
            plan_text,
            configured_partitions,
            requested_partitions: configured_partitions,
            effective_partitions,
            parallelism_mode: "fixed".to_owned(),
            available_parallelism,
            max_partitions: 32,
            min_rows_per_partition: MIN_ROWS_PER_PARTITION,
            small_rows_threshold: SMALL_ROWS_THRESHOLD,
            parallelism_decision_reused: false,
            decision_input_rows,
            decision_active_entities,
            decision_active_entities_source: decision_active_entities_source.to_owned(),
            partition_limit_reason: raw_limit_reason(effective_partitions, configured_partitions),
            batch_size,
            partition_rows: raw_partition_rows(&plan_statistics),
            spill_bytes: plan_statistics.spill_bytes,
            elapsed_compute_ns: plan_statistics.elapsed_compute_ns,
            window_compute_ns: plan_statistics.window_compute_ns,
            repartition_sort_compute_ns: plan_statistics.repartition_sort_compute_ns,
            window_operator_count: plan_statistics.window_operator_count,
            repartition_operator_count: plan_statistics.repartition_operator_count,
            sort_operator_count: plan_statistics.sort_operator_count,
            coalesce_operator_count: plan_statistics.coalesce_operator_count,
            peak_rss_bytes,
        },
        if retain_output { Some(output) } else { None },
    ))
}

fn plan_statistics(plan: &dyn ExecutionPlan, output_rows: usize) -> PlanStatistics {
    // The recursive accumulator gathers all operator and timing facts during
    // one traversal of the exact measured plan.
    // #lizard forgives
    let partition_count = plan.output_partitioning().partition_count().max(1);
    let mut statistics = PlanStatistics {
        partition_rows: vec![0; partition_count],
        ..PlanStatistics::default()
    };
    let name = plan.name();
    let is_window = name.contains("WindowAggExec") || name.contains("CalcFlowRollingExec");
    let is_repartition = name.contains("RepartitionExec");
    let is_sort = name.contains("SortExec");
    statistics.window_operator_count = usize::from(is_window);
    statistics.repartition_operator_count = usize::from(is_repartition);
    statistics.sort_operator_count = usize::from(is_sort);
    statistics.coalesce_operator_count = usize::from(name.contains("Coalesce"));
    if let Some(metrics) = plan.metrics() {
        for metric in metrics.iter() {
            if matches!(metric.value(), MetricValue::OutputRows(_)) {
                if let Some(partition) = metric.partition() {
                    if let Some(rows) = statistics.partition_rows.get_mut(partition) {
                        *rows = rows.saturating_add(metric.value().as_usize());
                    }
                } else if partition_count == 1 {
                    statistics.partition_rows[0] =
                        statistics.partition_rows[0].saturating_add(metric.value().as_usize());
                }
            }
        }
        statistics.spill_bytes = metrics.spilled_bytes().unwrap_or(0);
        statistics.elapsed_compute_ns = metrics.elapsed_compute().unwrap_or(0);
    }
    if partition_count == 1 && statistics.partition_rows[0] == 0 {
        statistics.partition_rows[0] = output_rows;
    }
    if is_window {
        statistics
            .window_partition_rows
            .clone_from(&statistics.partition_rows);
    }
    for child in plan.children() {
        let child = plan_statistics(child.as_ref(), 0);
        statistics.spill_bytes = statistics.spill_bytes.saturating_add(child.spill_bytes);
        statistics.elapsed_compute_ns = statistics
            .elapsed_compute_ns
            .saturating_add(child.elapsed_compute_ns);
        statistics.window_compute_ns = statistics
            .window_compute_ns
            .saturating_add(child.window_compute_ns);
        statistics.repartition_sort_compute_ns = statistics
            .repartition_sort_compute_ns
            .saturating_add(child.repartition_sort_compute_ns);
        statistics.window_operator_count = statistics
            .window_operator_count
            .saturating_add(child.window_operator_count);
        statistics.repartition_operator_count = statistics
            .repartition_operator_count
            .saturating_add(child.repartition_operator_count);
        statistics.sort_operator_count = statistics
            .sort_operator_count
            .saturating_add(child.sort_operator_count);
        statistics.coalesce_operator_count = statistics
            .coalesce_operator_count
            .saturating_add(child.coalesce_operator_count);
        if statistics.window_partition_rows.is_empty() {
            statistics.window_partition_rows = child.window_partition_rows;
        }
    }
    if let Some(metrics) = plan.metrics() {
        let elapsed_compute_ns = metrics.elapsed_compute().unwrap_or(0);
        if is_window {
            statistics.window_compute_ns = statistics
                .window_compute_ns
                .saturating_add(elapsed_compute_ns);
        }
        if is_repartition || is_sort {
            statistics.repartition_sort_compute_ns = statistics
                .repartition_sort_compute_ns
                .saturating_add(elapsed_compute_ns);
        }
    }
    statistics
}

#[allow(
    clippy::too_many_lines,
    reason = "the AB/BA ordering and paired evidence assembly remain adjacent for auditability"
)]
async fn warmup_outputs(
    args: &Args,
    plan: &calc_flow::BatchExecutionPlan,
    batch: &Batch,
    dimension: Option<&Batch>,
    workload: &Workload,
    effective_partitions: usize,
) -> BenchResult<(Option<Batch>, Option<Batch>)> {
    let mut calc_warm_output = None;
    let mut raw_warm_output = None;
    for _ in 0..args.warmups {
        let (_calc_warm, calc_output) = calc_flow_sample(plan, batch, dimension, true).await?;
        calc_warm_output = calc_output;
        let (_raw_warm, raw_output) = raw_datafusion_sample(
            batch,
            dimension,
            workload,
            args.partitions,
            effective_partitions,
            args.batch_size,
            true,
        )
        .await?;
        raw_warm_output = raw_output;
    }
    Ok((calc_warm_output, raw_warm_output))
}

async fn timed_samples(
    args: &Args,
    plan: &calc_flow::BatchExecutionPlan,
    batch: &Batch,
    dimension: Option<&Batch>,
    workload: &Workload,
    effective_partitions: usize,
) -> BenchResult<TimedPair> {
    let mut calc_samples = Vec::with_capacity(args.samples);
    let mut raw_samples = Vec::with_capacity(args.samples);
    let mut sample_order = Vec::with_capacity(args.samples);
    for sample in 0..args.samples {
        if sample % 2 == 0 {
            sample_order.push("ab".to_owned());
            calc_samples.push(calc_flow_sample(plan, batch, dimension, false).await?.0);
            raw_samples.push(
                raw_datafusion_sample(
                    batch,
                    dimension,
                    workload,
                    args.partitions,
                    effective_partitions,
                    args.batch_size,
                    false,
                )
                .await?
                .0,
            );
        } else {
            sample_order.push("ba".to_owned());
            raw_samples.push(
                raw_datafusion_sample(
                    batch,
                    dimension,
                    workload,
                    args.partitions,
                    effective_partitions,
                    args.batch_size,
                    false,
                )
                .await?
                .0,
            );
            calc_samples.push(calc_flow_sample(plan, batch, dimension, false).await?.0);
        }
    }
    Ok(TimedPair {
        calc: calc_samples,
        raw: raw_samples,
        order: sample_order,
    })
}

async fn benchmark_case(
    args: &Args,
    workload: &Workload,
    batch: &Batch,
    dimension: Option<&Batch>,
) -> BenchResult<CaseEvidence> {
    // Warm-up, alternating AB/BA sampling, correctness, and evidence assembly
    // intentionally form one auditable paired-measurement boundary.
    let config = config(args);
    let dimension_rows = dimension.map_or(0, Batch::num_rows);
    let effective_partitions = effective_partitions(args, dimension_rows);
    let plan = build_plan(config, workload)?;
    let (calc_warm_output, raw_warm_output) = warmup_outputs(
        args,
        &plan,
        batch,
        dimension,
        workload,
        effective_partitions,
    )
    .await?;
    let calc_warm = calc_warm_output
        .as_ref()
        .ok_or("missing Calc Flow warm-up")?;
    let raw_warm = raw_warm_output
        .as_ref()
        .ok_or("missing raw DataFusion warm-up")?;
    let correctness = compare_outputs(calc_warm, raw_warm, workload)?;
    let output_rows = calc_warm.num_rows();
    // The warm-up outputs have proved equivalence; dropping them before the
    // sampling loop keeps the run's retained memory flat instead of growing
    // by one output batch per sample.
    drop(calc_warm_output);
    drop(raw_warm_output);
    let samples = timed_samples(
        args,
        &plan,
        batch,
        dimension,
        workload,
        effective_partitions,
    )
    .await?;
    assemble_case_evidence(args, workload, correctness, output_rows, batch, samples)
}

#[allow(
    clippy::too_many_lines,
    reason = "strict benchmark evidence is assembled in one auditable boundary"
)]
/// The session and partitioning fields that must stay fixed across samples.
fn partitioning_of(
    sample: &Sample,
) -> (&str, &str, usize, usize, usize, usize, usize, usize, usize) {
    (
        &sample.parallelism_mode,
        &sample.partition_limit_reason,
        sample.configured_partitions,
        sample.requested_partitions,
        sample.effective_partitions,
        sample.available_parallelism,
        sample.max_partitions,
        sample.min_rows_per_partition,
        sample.small_rows_threshold,
    )
}

/// The decision and physical-plan fields that must stay fixed across samples.
fn decision_of(sample: &Sample) -> (bool, usize, Option<usize>, &str, &str) {
    (
        sample.parallelism_decision_reused,
        sample.decision_input_rows,
        sample.decision_active_entities,
        &sample.decision_active_entities_source,
        &sample.plan_text,
    )
}

/// Proves every sample used one identical configuration and physical plan.
fn samples_share_configuration(samples: &[Sample]) -> BenchResult<()> {
    let first = samples.first().ok_or("benchmark produced no samples")?;
    let shared = samples.iter().all(|sample| {
        partitioning_of(sample) == partitioning_of(first)
            && decision_of(sample) == decision_of(first)
            && sample.batch_size == first.batch_size
    });
    if shared {
        Ok(())
    } else {
        Err("engine configuration or physical plan changed between samples".into())
    }
}

fn engine_evidence(
    samples: &[Sample],
    input_batch_rows: &[usize],
    output_rows: usize,
) -> BenchResult<EngineEvidence> {
    // This function first proves every sample used identical configuration and
    // then emits the complete fail-closed engine evidence record.
    samples_share_configuration(samples)?;
    let first = samples.first().ok_or("benchmark produced no samples")?;
    let sample_ms = samples
        .iter()
        .map(|sample| sample.elapsed_ms)
        .collect::<Vec<_>>();
    let median_ms = median(&sample_ms);
    let partition_rows = samples
        .last()
        .ok_or("benchmark produced no final sample")?
        .partition_rows
        .clone();
    // Partition skew averages the engine's output rows, not the query input
    // rows: `filter` and `group_by` change the output cardinality.
    let average_partition_rows = output_rows as f64 / partition_rows.len() as f64;
    let partition_skew =
        partition_rows.iter().copied().max().unwrap_or(0) as f64 / average_partition_rows;
    Ok(EngineEvidence {
        parallelism_mode: first.parallelism_mode.clone(),
        configured_partitions: first.configured_partitions,
        requested_partitions: first.requested_partitions,
        effective_partitions: first.effective_partitions,
        available_parallelism: first.available_parallelism,
        max_partitions: first.max_partitions,
        min_rows_per_partition: first.min_rows_per_partition,
        small_rows_threshold: first.small_rows_threshold,
        parallelism_decision_reused: first.parallelism_decision_reused,
        decision_input_rows: first.decision_input_rows,
        decision_active_entities: first.decision_active_entities,
        decision_active_entities_source: first.decision_active_entities_source.clone(),
        partition_limit_reason: first.partition_limit_reason.clone(),
        batch_size: first.batch_size,
        input_logical_partitions: 1,
        input_batch_rows: input_batch_rows.to_vec(),
        normalized_plan_hash: sha256(first.plan_text.as_bytes()),
        bounded_window_agg_count: first.plan_text.matches("BoundedWindowAggExec").count(),
        samples_ms: sample_ms.clone(),
        median_ms,
        p25_ms: percentile(&sample_ms, 0.25),
        p75_ms: percentile(&sample_ms, 0.75),
        mad_ms: median(
            &sample_ms
                .iter()
                .map(|value| (value - median_ms).abs())
                .collect::<Vec<_>>(),
        ),
        cv: coefficient_of_variation(&sample_ms),
        cpu_time_ms: median(
            &samples
                .iter()
                .map(|sample| sample.elapsed_compute_ns as f64 / 1_000_000.0)
                .collect::<Vec<_>>(),
        ),
        peak_rss_bytes: samples
            .iter()
            .map(|sample| sample.peak_rss_bytes)
            .max()
            .unwrap_or(1),
        spill_bytes: samples
            .iter()
            .map(|sample| sample.spill_bytes)
            .max()
            .unwrap_or(0),
        empty_partitions: partition_rows.iter().filter(|rows| **rows == 0).count(),
        partition_rows,
        partition_skew,
        window_compute_ms: median(
            &samples
                .iter()
                .map(|sample| sample.window_compute_ns as f64 / 1_000_000.0)
                .collect::<Vec<_>>(),
        ),
        repartition_sort_compute_ms: median(
            &samples
                .iter()
                .map(|sample| sample.repartition_sort_compute_ns as f64 / 1_000_000.0)
                .collect::<Vec<_>>(),
        ),
        window_operator_count: first.window_operator_count,
        repartition_operator_count: first.repartition_operator_count,
        sort_operator_count: first.sort_operator_count,
        coalesce_operator_count: first.coalesce_operator_count,
        phase_medians_ms: median_phases(samples),
        phase_samples_ms: phase_samples(samples),
    })
}

fn phase_samples(samples: &[Sample]) -> BTreeMap<String, Vec<f64>> {
    let mut values = BTreeMap::new();
    for (name, select) in [
        (
            "runtime_acquire",
            (|v: &PhaseMedians| v.runtime_acquire) as fn(&PhaseMedians) -> f64,
        ),
        ("session_state_create", |v: &PhaseMedians| {
            v.session_state_create
        }),
        ("input_adapter", |v: &PhaseMedians| v.input_adapter),
        ("table_register", |v: &PhaseMedians| v.table_register),
        ("sql_parse", |v: &PhaseMedians| v.sql_parse),
        ("logical_optimize", |v: &PhaseMedians| v.logical_optimize),
        ("physical_plan", |v: &PhaseMedians| v.physical_plan),
        ("execution_to_first_batch", |v: &PhaseMedians| {
            v.execution_to_first_batch
        }),
        ("execution_remaining", |v: &PhaseMedians| {
            v.execution_remaining
        }),
        ("collect_or_coalesce", |v: &PhaseMedians| {
            v.collect_or_coalesce
        }),
        ("output_arrow_wrap", |v: &PhaseMedians| v.output_arrow_wrap),
        ("audit", |v: &PhaseMedians| v.audit),
        ("metrics_traversal", |v: &PhaseMedians| v.metrics_traversal),
        ("physical_plan_string", |v: &PhaseMedians| {
            v.physical_plan_string
        }),
        ("batch_envelope", |v: &PhaseMedians| v.batch_envelope),
        ("run_result", |v: &PhaseMedians| v.run_result),
        ("run_session_envelope", |v: &PhaseMedians| {
            v.run_session_envelope
        }),
    ] {
        values.insert(
            name.to_owned(),
            samples
                .iter()
                .map(|sample| select(&sample.phases))
                .collect(),
        );
    }
    values
}

fn median_phases(samples: &[Sample]) -> PhaseMedians {
    let phase = |select: fn(&PhaseMedians) -> f64| {
        median(
            &samples
                .iter()
                .map(|sample| select(&sample.phases))
                .collect::<Vec<_>>(),
        )
    };
    PhaseMedians {
        runtime_acquire: phase(|value| value.runtime_acquire),
        session_state_create: phase(|value| value.session_state_create),
        input_adapter: phase(|value| value.input_adapter),
        table_register: phase(|value| value.table_register),
        sql_parse: phase(|value| value.sql_parse),
        logical_optimize: phase(|value| value.logical_optimize),
        physical_plan: phase(|value| value.physical_plan),
        execution_to_first_batch: phase(|value| value.execution_to_first_batch),
        execution_remaining: phase(|value| value.execution_remaining),
        collect_or_coalesce: phase(|value| value.collect_or_coalesce),
        output_arrow_wrap: phase(|value| value.output_arrow_wrap),
        audit: phase(|value| value.audit),
        metrics_traversal: phase(|value| value.metrics_traversal),
        physical_plan_string: phase(|value| value.physical_plan_string),
        batch_envelope: phase(|value| value.batch_envelope),
        run_result: phase(|value| value.run_result),
        run_session_envelope: phase(|value| value.run_session_envelope),
    }
}

fn exclusive_phase_total(phases: &PhaseMedians) -> f64 {
    phases.runtime_acquire
        + phases.session_state_create
        + phases.input_adapter
        + phases.table_register
        + phases.sql_parse
        + phases.logical_optimize
        + phases.physical_plan
        + phases.execution_to_first_batch
        + phases.execution_remaining
        + phases.output_arrow_wrap
        + phases.metrics_traversal
        + phases.batch_envelope
        + phases.run_result
}

fn comparability_mismatches(calc: &EngineEvidence, raw: &EngineEvidence) -> Vec<String> {
    // Keep the explicit field-by-field list synchronized with the public
    // comparability contract and preserve stable mismatch ordering.
    // #lizard forgives
    let mut mismatches = Vec::new();
    if calc.parallelism_mode != raw.parallelism_mode {
        mismatches.push("parallelism_mode".to_owned());
    }
    if calc.configured_partitions != raw.configured_partitions {
        mismatches.push("configured_partitions".to_owned());
    }
    if calc.effective_partitions != raw.effective_partitions {
        mismatches.push("effective_partitions".to_owned());
    }
    if calc.available_parallelism != raw.available_parallelism {
        mismatches.push("available_parallelism".to_owned());
    }
    if calc.max_partitions != raw.max_partitions {
        mismatches.push("max_partitions".to_owned());
    }
    if calc.min_rows_per_partition != raw.min_rows_per_partition {
        mismatches.push("min_rows_per_partition".to_owned());
    }
    if calc.small_rows_threshold != raw.small_rows_threshold {
        mismatches.push("small_rows_threshold".to_owned());
    }
    if calc.decision_input_rows != raw.decision_input_rows {
        mismatches.push("decision_input_rows".to_owned());
    }
    if calc.decision_active_entities != raw.decision_active_entities {
        mismatches.push("decision_active_entities".to_owned());
    }
    if calc.decision_active_entities_source != raw.decision_active_entities_source {
        mismatches.push("decision_active_entities_source".to_owned());
    }
    if calc.parallelism_decision_reused != raw.parallelism_decision_reused {
        mismatches.push("parallelism_decision_reused".to_owned());
    }
    if calc.batch_size != raw.batch_size {
        mismatches.push("batch_size".to_owned());
    }
    if calc.input_logical_partitions != raw.input_logical_partitions {
        mismatches.push("input_logical_partitions".to_owned());
    }
    if calc.input_batch_rows != raw.input_batch_rows {
        mismatches.push("input_batch_rows".to_owned());
    }
    if calc.normalized_plan_hash != raw.normalized_plan_hash {
        mismatches.push("normalized_plan_hash".to_owned());
    }
    mismatches
}

fn raw_decision_input_rows(batch: &Batch, dimension: Option<&Batch>) -> usize {
    batch
        .num_rows()
        .saturating_add(dimension.map_or(0, Batch::num_rows))
}

fn raw_decision_entities(
    batch: &Batch,
    dimension: Option<&Batch>,
) -> (Option<usize>, &'static str) {
    if dimension.is_some() {
        return (None, "multiple_inputs");
    }
    let entities = batch
        .metadata()
        .attributes()
        .get(DATAFUSION_ACTIVE_ENTITIES_METADATA_KEY)
        .and_then(serde_json::Value::as_u64)
        .and_then(|value| usize::try_from(value).ok());
    (entities, "batch_metadata")
}

fn raw_limit_reason(effective_partitions: usize, configured_partitions: usize) -> String {
    if effective_partitions < configured_partitions {
        "minimum_rows_per_partition"
    } else {
        "configured_target_partitions"
    }
    .to_owned()
}

fn raw_partition_rows(statistics: &PlanStatistics) -> Vec<usize> {
    if statistics.window_partition_rows.is_empty() {
        statistics.partition_rows.clone()
    } else {
        statistics.window_partition_rows.clone()
    }
}

fn input_mem_table(batch: &Batch) -> BenchResult<MemTable> {
    let records = batch.table_payload()?;
    Ok(MemTable::try_new(
        Arc::clone(records.schema()),
        vec![records.batches().to_vec()],
    )?)
}

fn masks_match(
    calc: &Float64Array,
    raw: &Float64Array,
    calc_index: usize,
    raw_index: usize,
) -> bool {
    calc.is_null(calc_index) == raw.is_null(raw_index)
        && (calc.is_null(calc_index)
            || calc.value(calc_index).is_nan() == raw.value(raw_index).is_nan())
}

fn values_match(
    calc: &Float64Array,
    raw: &Float64Array,
    calc_index: usize,
    raw_index: usize,
) -> bool {
    if calc.is_null(calc_index) || raw.is_null(raw_index) {
        return calc.is_null(calc_index) == raw.is_null(raw_index);
    }
    let left = calc.value(calc_index);
    let right = raw.value(raw_index);
    (left.is_nan() && right.is_nan())
        || left.to_bits() == right.to_bits()
        || (left - right).abs() <= ATOL + RTOL * right.abs()
}

struct TimedPair {
    calc: Vec<Sample>,
    raw: Vec<Sample>,
    order: Vec<String>,
}

fn assemble_case_evidence(
    args: &Args,
    workload: &Workload,
    correctness: Correctness,
    output_rows: usize,
    batch: &Batch,
    samples: TimedPair,
) -> BenchResult<CaseEvidence> {
    let input_batch_rows = batch
        .table_payload()?
        .batches()
        .iter()
        .map(RecordBatch::num_rows)
        .collect::<Vec<_>>();
    let calc_evidence = engine_evidence(&samples.calc, &input_batch_rows, output_rows)?;
    let raw_evidence = engine_evidence(&samples.raw, &input_batch_rows, output_rows)?;
    let mismatches = comparability_mismatches(&calc_evidence, &raw_evidence);
    let comparable = mismatches.is_empty();
    let ratios = calc_evidence
        .samples_ms
        .iter()
        .zip(&raw_evidence.samples_ms)
        .map(|(calc, raw)| calc / raw)
        .collect::<Vec<_>>();
    let (paired_ratio_ci_low, paired_ratio_ci_high) = bootstrap_median_interval(&ratios);
    let paired_ratio_median = median(&ratios);
    Ok(CaseEvidence {
        name: workload.name.to_owned(),
        rows: args.rows,
        active_entities: args.entities,
        window: workload.window,
        output_rows,
        warmups: args.warmups,
        rolling_rewrite_enabled: false,
        sample_order: samples.order,
        calc_flow: calc_evidence,
        raw_datafusion: raw_evidence,
        paired_ratios: ratios,
        paired_ratio_median,
        paired_ratio_ci_low,
        paired_ratio_ci_high,
        correctness,
        comparability: Comparability {
            comparable,
            mismatches,
        },
        speedup_conclusion: comparable
            .then(|| format!("calc_flow_over_raw={paired_ratio_median:.6}x")),
    })
}

fn concat_pair(
    calc_table: &calc_flow::TableBatch,
    raw_table: &calc_flow::TableBatch,
) -> BenchResult<(RecordBatch, RecordBatch)> {
    Ok((
        concat_batches(calc_table.schema(), calc_table.batches())?,
        concat_batches(raw_table.schema(), raw_table.batches())?,
    ))
}

type CanonicalPair = (Vec<(RowKey, usize)>, Vec<(RowKey, usize)>);

fn canonical_pair(
    calc_batch: &RecordBatch,
    raw_batch: &RecordBatch,
    identity: RowIdentity,
) -> BenchResult<CanonicalPair> {
    Ok((
        canonical_rows(calc_batch, identity)?,
        canonical_rows(raw_batch, identity)?,
    ))
}

fn compare_outputs(calc: &Batch, raw: &Batch, workload: &Workload) -> BenchResult<Correctness> {
    // Schema, keys, null/NaN masks, and numeric values are one conjunctive
    // correctness result for each measured pair.
    // #lizard forgives
    let calc_table = calc.table_payload()?;
    let raw_table = raw.table_payload()?;
    let schema = calc_table.schema() == raw_table.schema();
    let rows = calc.num_rows() == raw.num_rows();
    let (calc_batch, raw_batch) = concat_pair(calc_table, raw_table)?;
    let (calc_rows, raw_rows) = canonical_pair(&calc_batch, &raw_batch, workload.identity)?;
    let keys = calc_rows
        .iter()
        .map(|(key, _)| key)
        .eq(raw_rows.iter().map(|(key, _)| key));
    // SQL without an outer ORDER BY is a relation. Compare it in canonical
    // unique-key order outside the timed envelope so partition scheduling
    // cannot masquerade as a correctness failure.
    let order = keys;
    let calc_values = float_column(&calc_batch, workload.output_column)?;
    let raw_values = float_column(&raw_batch, workload.output_column)?;
    let mut aligned_indices = calc_rows
        .iter()
        .map(|(_, index)| *index)
        .zip(raw_rows.iter().map(|(_, index)| *index));
    let null_nan_mask = aligned_indices
        .clone()
        .all(|(calc_index, raw_index)| masks_match(calc_values, raw_values, calc_index, raw_index));
    let values = aligned_indices.all(|(calc_index, raw_index)| {
        values_match(calc_values, raw_values, calc_index, raw_index)
    });
    Ok(Correctness {
        schema,
        rows,
        keys,
        order,
        null_nan_mask,
        values,
        rtol: RTOL,
        atol: ATOL,
    })
}

/// The canonical output row key for one workload identity kind.
#[derive(Clone, Debug, Eq, Ord, PartialEq, PartialOrd)]
enum RowKey {
    Row(String, i64, u64),
    Group(String),
}

fn canonical_rows(batch: &RecordBatch, identity: RowIdentity) -> BenchResult<Vec<(RowKey, usize)>> {
    let symbols = batch
        .column_by_name("symbol")
        .ok_or("symbol column is missing")?
        .as_any()
        .downcast_ref::<StringArray>()
        .ok_or("symbol column is not Utf8")?;
    let mut rows = match identity {
        RowIdentity::Group => (0..batch.num_rows())
            .map(|index| (RowKey::Group(symbols.value(index).to_owned()), index))
            .collect::<Vec<_>>(),
        RowIdentity::Row => row_identity_keys(batch, symbols)?,
    };
    rows.sort_by(|left, right| left.0.cmp(&right.0));
    if rows.windows(2).any(|pair| pair[0].0 == pair[1].0) {
        return Err("benchmark row identity is not unique".into());
    }
    Ok(rows)
}

fn row_identity_keys(
    batch: &RecordBatch,
    symbols: &StringArray,
) -> BenchResult<Vec<(RowKey, usize)>> {
    let event_times = batch
        .column_by_name("event_time")
        .ok_or("event_time column is missing")?
        .as_any()
        .downcast_ref::<TimestampMicrosecondArray>()
        .ok_or("event_time column is not Timestamp(Microsecond)")?;
    let sequences = batch
        .column_by_name("sequence")
        .ok_or("sequence column is missing")?
        .as_any()
        .downcast_ref::<UInt64Array>()
        .ok_or("sequence column is not UInt64")?;
    Ok((0..batch.num_rows())
        .map(|index| {
            (
                RowKey::Row(
                    symbols.value(index).to_owned(),
                    event_times.value(index),
                    sequences.value(index),
                ),
                index,
            )
        })
        .collect::<Vec<_>>())
}

fn float_column<'a>(batch: &'a RecordBatch, name: &str) -> BenchResult<&'a Float64Array> {
    batch
        .column_by_name(name)
        .ok_or_else(|| format!("output column {name:?} is missing"))?
        .as_any()
        .downcast_ref::<Float64Array>()
        .ok_or_else(|| format!("output column {name:?} is not Float64").into())
}

fn median(values: &[f64]) -> f64 {
    percentile(values, 0.5)
}

fn percentile(values: &[f64], fraction: f64) -> f64 {
    let mut ordered = values.to_vec();
    ordered.sort_by(f64::total_cmp);
    let position = (ordered.len() - 1) as f64 * fraction;
    let lower = position.floor() as usize;
    let upper = position.ceil() as usize;
    if lower == upper {
        ordered[lower]
    } else {
        let weight = position - lower as f64;
        ordered[lower] * (1.0 - weight) + ordered[upper] * weight
    }
}

fn coefficient_of_variation(values: &[f64]) -> f64 {
    let mean = values.iter().sum::<f64>() / values.len() as f64;
    let variance = values
        .iter()
        .map(|value| (value - mean).powi(2))
        .sum::<f64>()
        / values.len() as f64;
    variance.sqrt() / mean
}

fn bootstrap_median_interval(ratios: &[f64]) -> (f64, f64) {
    let mut seed = 0x9e37_79b9_7f4a_7c15_u64;
    let mut medians = Vec::with_capacity(BOOTSTRAP_RESAMPLES);
    for _ in 0..BOOTSTRAP_RESAMPLES {
        let mut sample = Vec::with_capacity(ratios.len());
        for _ in ratios {
            seed ^= seed << 13;
            seed ^= seed >> 7;
            seed ^= seed << 17;
            sample.push(ratios[seed as usize % ratios.len()]);
        }
        medians.push(median(&sample));
    }
    (percentile(&medians, 0.025), percentile(&medians, 0.975))
}

fn milliseconds(duration: Duration) -> f64 {
    duration.as_secs_f64() * 1_000.0
}

fn ns_ms(nanoseconds: u64) -> f64 {
    nanoseconds as f64 / 1_000_000.0
}

fn sha256(bytes: &[u8]) -> String {
    hex::encode(Sha256::digest(bytes))
}

fn command_output(program: &str, arguments: &[&str]) -> BenchResult<String> {
    let output = Command::new(program).args(arguments).output()?;
    if !output.status.success() {
        return Err(format!("{program} {arguments:?} failed with {}", output.status).into());
    }
    Ok(String::from_utf8(output.stdout)?.trim().to_owned())
}

fn cpu_model() -> String {
    fs::read_to_string("/proc/cpuinfo")
        .ok()
        .and_then(|cpuinfo| {
            cpuinfo.lines().find_map(|line| {
                line.strip_prefix("model name\t:")
                    .map(str::trim)
                    .map(str::to_owned)
            })
        })
        .unwrap_or_else(|| "unknown-cpu".to_owned())
}

fn environment(args: &Args, cases: &[CaseEvidence]) -> BenchResult<Environment> {
    let os = std::env::consts::OS.to_owned();
    let arch = std::env::consts::ARCH.to_owned();
    let cpu_model = cpu_model();
    let available_parallelism = thread::available_parallelism()
        .map(std::num::NonZero::get)
        .unwrap_or(1);
    let rust_version = command_output("rustc", &["--version"])?;
    let git_dirty = !command_output(
        "git",
        &["status", "--porcelain", "--untracked-files=normal"],
    )?
    .is_empty();
    let machine_fingerprint =
        sha256(format!("{os}|{arch}|{cpu_model}|{available_parallelism}").as_bytes());
    let workspace_root = workspace_root()?;
    let cargo_lock = fs::read(workspace_root.join("Cargo.lock"))?;
    let dependency_fingerprint = sha256(
        [
            cargo_lock.as_slice(),
            DATAFUSION_VERSION.as_bytes(),
            ARROW_VERSION.as_bytes(),
            rust_version.as_bytes(),
        ]
        .concat()
        .as_slice(),
    );
    let workload_fingerprint = sha256(
        serde_json::to_vec(&json!({
            "profile": args.profile.as_str(),
            "rows": args.rows,
            "entities": args.entities,
            "batch_size": args.batch_size,
            "partitions": args.partitions,
            "cases": cases.iter().map(|case| (&case.name, case.window)).collect::<Vec<_>>(),
        }))?
        .as_slice(),
    );
    Ok(Environment {
        machine_fingerprint,
        dependency_fingerprint,
        workload_fingerprint,
        datafusion_version: DATAFUSION_VERSION.to_owned(),
        arrow_version: ARROW_VERSION.to_owned(),
        build_profile: if cfg!(debug_assertions) {
            "debug"
        } else {
            "release"
        }
        .to_owned(),
        allocator: "system".to_owned(),
        os,
        arch,
        cpu_model,
        available_parallelism,
        rust_version,
        git_dirty,
    })
}

fn workspace_root() -> BenchResult<&'static Path> {
    Ok(Path::new(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .and_then(Path::parent)
        .ok_or("calc-flow manifest has no workspace root")?)
}

// `cargo bench` runs this harness with the package root as the working
// directory, so a relative `--output` must not be resolved against the current
// directory: anchor it at the workspace root to keep the report location
// independent of the caller's working directory.
fn resolve_output_path(output: &Path) -> BenchResult<PathBuf> {
    if output.is_absolute() {
        return Ok(output.to_path_buf());
    }
    let anchored = workspace_root()?.join(output);
    eprintln!(
        "sql_datafusion_performance: relative --output anchored at workspace root: {}",
        anchored.display()
    );
    Ok(anchored)
}

fn write_report(path: &Path, report: &Report) -> BenchResult<()> {
    if let Some(parent) = path.parent() {
        fs::create_dir_all(parent)?;
    }
    let temporary = path.with_extension("tmp");
    fs::write(&temporary, serde_json::to_vec_pretty(report)?)?;
    fs::rename(temporary, path)?;
    Ok(())
}

#[cfg(test)]
fn run_output_anchor_tests() -> BenchResult<()> {
    let absolute = workspace_root()?.join("evidence.json");
    assert_eq!(resolve_output_path(&absolute)?, absolute);

    let anchored = resolve_output_path(Path::new("benchmark-results/sql.json"))?;
    assert_eq!(
        anchored,
        workspace_root()?.join("benchmark-results/sql.json")
    );

    // The documented example uses a workspace-relative path; it must resolve
    // at the workspace root even though `cargo bench` runs from the package
    // root.
    let documented = resolve_output_path(Path::new("target/sql-datafusion/matched-first.json"))?;
    assert_eq!(
        documented,
        workspace_root()?.join("target/sql-datafusion/matched-first.json")
    );
    Ok(())
}

// The kernel peak counter and its `clear_refs` reset are Linux contracts;
// `scripts/run_rust_tests.py` also drives this self-test there. Other hosts
// (including the Windows CI leg) skip it instead of failing on absent
// `/proc` entries, while the sampler itself keeps its cross-platform floor.
#[cfg(all(test, target_os = "linux"))]
fn run_rss_window_tests() -> BenchResult<()> {
    // The kernel peak counter must never trail the current footprint.
    let rss = current_rss_bytes();
    let peak = peak_rss_bytes();
    if peak < rss {
        return Err(format!("kernel peak RSS {peak} is below current RSS {rss}").into());
    }

    // Resetting the window must re-anchor the peak to the current footprint,
    // not to the process-wide maximum (allocator churn gets bounded slack).
    reset_peak_rss_window();
    let reset_peak = peak_rss_bytes();
    let reset_slack = 16 * 1024 * 1024;
    if reset_peak > current_rss_bytes().saturating_add(reset_slack) {
        return Err(format!("reset peak RSS {reset_peak} stayed above the current window").into());
    }

    // A transient allocation inside one sample window must be captured
    // exactly, including after its buffers are dropped. Writing through each
    // page pins physical memory; a zero-only allocation would map the shared
    // zero page and allocate nothing.
    let sampler = RssSampler::start();
    let baseline = current_rss_bytes();
    let mut transient = vec![0_u8; 64 * 1024 * 1024];
    for page in transient.chunks_mut(4_096) {
        page[0] = 1;
    }
    std::hint::black_box(&transient);
    drop(transient);
    let observed = sampler.finish();
    let expected_growth = 32 * 1024 * 1024;
    if observed < baseline.saturating_add(expected_growth) {
        return Err(format!(
            "sample peak RSS {observed} missed a {expected_growth}-byte transient window"
        )
        .into());
    }
    Ok(())
}

#[cfg(test)]
fn retention_case_args(root: &Path) -> Args {
    Args {
        profile: Profile::SerialControl,
        output: root.join("target/sql-datafusion/retention-test.json"),
        rows: 4_096,
        entities: 4,
        batch_size: 1_024,
        partitions: 1,
        samples: 2,
        warmups: 1,
    }
}

#[cfg(test)]
fn correctness_is_complete(correctness: &Correctness) -> bool {
    [
        correctness.schema,
        correctness.rows,
        correctness.keys,
        correctness.order,
        correctness.null_nan_mask,
        correctness.values,
    ]
    .into_iter()
    .all(|flag| flag)
}

#[cfg(test)]
async fn timed_sample_drops_output(
    args: &Args,
    batch: &Batch,
    workload: &Workload,
) -> BenchResult<()> {
    let plan = build_plan(config(args), workload)?;
    let (sample, retained) = calc_flow_sample(&plan, batch, None, false).await?;
    if retained.is_some() {
        return Err("timed Calc Flow sample retained its output batch".into());
    }
    if sample.elapsed_ms <= 0.0 {
        return Err("timed Calc Flow sample recorded no elapsed time".into());
    }
    Ok(())
}

#[cfg(test)]
async fn run_sample_retention_tests() -> BenchResult<()> {
    // One tiny paired case must still prove output equivalence from its
    // warm-up pair while timed samples retain no output batches at all.
    let args = retention_case_args(workspace_root()?);
    let records = input_batches(args.rows, args.entities, args.batch_size)?;
    let batch = benchmark_batch(records, args.entities)?;
    let workload = &workloads()[0];
    timed_sample_drops_output(&args, &batch, workload).await?;
    let case = benchmark_case(&args, workload, &batch, None).await?;
    if !correctness_is_complete(&case.correctness) {
        return Err("retention restructure lost the warm-up correctness pair".into());
    }
    Ok(())
}

#[cfg(test)]
fn operator_output_rows_are_expected(name: &str, output_rows: usize, args: &Args) -> bool {
    match name {
        "projection" | "join" => output_rows == args.rows,
        "group_by" => output_rows == args.entities,
        "filter" => output_rows > 0 && output_rows < args.rows,
        _ => false,
    }
}

async fn operator_workload_case_is_valid(
    args: &Args,
    workload: &Workload,
    batch: &Batch,
    dimension: &Batch,
) -> BenchResult<()> {
    let case_dimension = workload.dimension.then_some(dimension);
    let case = benchmark_case(args, workload, batch, case_dimension).await?;
    if !correctness_is_complete(&case.correctness) {
        return Err(format!("operator workload {:?} lost correctness", workload.name).into());
    }
    if !case.comparability.comparable {
        return Err(format!(
            "operator workload {:?} is not comparable: {:?}",
            workload.name, case.comparability.mismatches
        )
        .into());
    }
    if !operator_output_rows_are_expected(workload.name, case.output_rows, args) {
        return Err(format!(
            "operator workload {:?} recorded unexpected output rows {}",
            workload.name, case.output_rows
        )
        .into());
    }
    Ok(())
}

fn require_operator_workloads() -> BenchResult<()> {
    let names = workloads().map(|workload| workload.name);
    let missing = ["projection", "filter", "group_by", "join"]
        .into_iter()
        .filter(|expected| !names.contains(expected))
        .collect::<Vec<_>>();
    match missing.as_slice() {
        [] => Ok(()),
        [first, ..] => Err(format!("operator workload {first:?} is missing").into()),
    }
}

async fn run_operator_workload_tests() -> BenchResult<()> {
    // The operator scenarios extend the rolling pair with row-local,
    // cardinality-changing, and two-input SQL shapes; each must stay correct,
    // comparable, and record its true output cardinality.
    require_operator_workloads()?;
    let args = retention_case_args(workspace_root()?);
    let records = input_batches(args.rows, args.entities, args.batch_size)?;
    let batch = benchmark_batch(records, args.entities)?;
    let dimension = benchmark_dimension_batch(args.entities)?;
    for workload in workloads().iter().filter(|workload| workload.window == 0) {
        operator_workload_case_is_valid(&args, workload, &batch, &dimension).await?;
    }
    Ok(())
}

#[cfg(test)]
async fn run_bench_self_tests() -> BenchResult<()> {
    if let Err(error) = run_output_anchor_tests() {
        eprintln!("sql_datafusion_performance output-anchor tests: {error}");
        std::process::exit(1);
    }
    #[cfg(target_os = "linux")]
    if let Err(error) = run_rss_window_tests() {
        eprintln!("sql_datafusion_performance rss-window tests: {error}");
        std::process::exit(1);
    }
    if let Err(error) = run_sample_retention_tests().await {
        eprintln!("sql_datafusion_performance sample-retention tests: {error}");
        std::process::exit(1);
    }
    if let Err(error) = run_operator_workload_tests().await {
        eprintln!("sql_datafusion_performance operator-workload tests: {error}");
        std::process::exit(1);
    }
    Ok(())
}

async fn run_all_workloads(args: &Args) -> BenchResult<Vec<CaseEvidence>> {
    let records = input_batches(args.rows, args.entities, args.batch_size)?;
    let batch = benchmark_batch(records, args.entities)?;
    let dimension = benchmark_dimension_batch(args.entities)?;
    let mut cases = Vec::new();
    for workload in workloads() {
        let case_dimension = workload.dimension.then_some(&dimension);
        cases.push(benchmark_case(args, &workload, &batch, case_dimension).await?);
    }
    Ok(cases)
}

#[tokio::main]
async fn main() -> BenchResult<()> {
    // Benchmark setup, both workload cases, provenance, and atomic publication
    // remain one top-level evidence lifecycle.
    #[cfg(test)]
    if std::env::args_os().len() == 1 {
        // nosemgrep
        return run_bench_self_tests().await;
    }

    let args = Args::parse()?;
    let cases = run_all_workloads(&args).await?;
    let report = Report {
        schema_version: 1,
        git_sha: command_output("git", &["rev-parse", "HEAD^{commit}"])?,
        profile: args.profile.as_str().to_owned(),
        environment: environment(&args, &cases)?,
        cases,
    };
    write_report(&args.output, &report)?;
    Ok(())
}
