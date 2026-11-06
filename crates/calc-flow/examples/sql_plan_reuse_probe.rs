//! Exploratory probe for repeated state-feedback batch SQL execution.
//!
//! Compiles one two-input `UNION ALL` + `GROUP BY` batch graph, then calls
//! `BatchExecutionPlan::execute` once per input batch while feeding the
//! previous full result back as `previous_positions`. Every run's
//! `DataFusionQueryMetric` phases are summed so the per-batch cost of session
//! creation, table registration, planning, and execution can be attributed.
//! Final positions are checked against an independent accumulator before any
//! timing is reported.
//!
//! Environment: `PLAN_REUSE_PROBE_ROWS` (business rows, default 1,000,000),
//! `PLAN_REUSE_PROBE_BATCHES` (comma-separated batch sizes, default
//! `1000,10000`), `PLAN_REUSE_PROBE_PARTITIONS` (comma-separated fixed
//! partition counts, default `1,4`).
//!
//! Run with:
//! ```bash
//! cargo run -p calc-flow --release --example sql_plan_reuse_probe
//! ```

use std::{
    collections::{BTreeMap, HashMap},
    sync::Arc,
    time::Instant,
};

use calc_flow::{
    Batch, BatchExecutionPlan, BatchMetadata, DataFusionConfig, DataFusionParallelismMode,
    DataFusionQueryMetric, ExecutionOptions, PipelineBuilder, SqlOperator, UdfRegistry,
};
use datafusion::arrow::{
    array::{Array, Int64Array, StringArray},
    datatypes::{DataType, Field, Schema, SchemaRef},
    record_batch::RecordBatch,
};

type ProbeResult<T> = Result<T, Box<dyn std::error::Error + Send + Sync>>;

const ACCOUNTS: usize = 100;
const STOCKS: usize = 100;

const QUERY: &str = "SELECT account, stock, SUM(position) AS position FROM (
    SELECT account, stock, position FROM previous_positions
    UNION ALL
    SELECT account, stock, CASE WHEN side = 'buy' THEN quantity ELSE -quantity END AS position
    FROM input WHERE flow_type = 'trade'
) GROUP BY account, stock";

fn position_schema() -> SchemaRef {
    Arc::new(Schema::new(vec![
        Field::new("account", DataType::Utf8, true),
        Field::new("stock", DataType::Utf8, true),
        Field::new("position", DataType::Int64, true),
    ]))
}

fn input_schema() -> SchemaRef {
    Arc::new(Schema::new(vec![
        Field::new("account", DataType::Utf8, false),
        Field::new("stock", DataType::Utf8, false),
        Field::new("side", DataType::Utf8, false),
        Field::new("quantity", DataType::Int64, false),
        Field::new("flow_type", DataType::Utf8, false),
    ]))
}

/// Deterministic business row: every tenth row is a quote, the rest trades.
fn row(index: usize) -> (String, String, &'static str, i64, &'static str) {
    let account = format!("A{:03}", index % ACCOUNTS);
    let stock = format!("S{:03}", (index / ACCOUNTS) % STOCKS);
    let side = if index.is_multiple_of(3) {
        "sell"
    } else {
        "buy"
    };
    let quantity = i64::try_from(index % 7 + 1).expect("small quantity");
    let flow_type = if index % 10 == 9 { "quote" } else { "trade" };
    (account, stock, side, quantity, flow_type)
}

fn input_batch(start: usize, rows: usize) -> ProbeResult<Batch> {
    let mut accounts = Vec::with_capacity(rows);
    let mut stocks = Vec::with_capacity(rows);
    let mut sides = Vec::with_capacity(rows);
    let mut quantities = Vec::with_capacity(rows);
    let mut flows = Vec::with_capacity(rows);
    for index in start..start + rows {
        let (account, stock, side, quantity, flow_type) = row(index);
        accounts.push(account);
        stocks.push(stock);
        sides.push(side);
        quantities.push(quantity);
        flows.push(flow_type);
    }
    let record = RecordBatch::try_new(
        input_schema(),
        vec![
            Arc::new(StringArray::from(accounts)) as Arc<dyn Array>,
            Arc::new(StringArray::from(stocks)),
            Arc::new(StringArray::from(sides)),
            Arc::new(Int64Array::from(quantities)),
            Arc::new(StringArray::from(flows)),
        ],
    )?;
    Ok(Batch::table(vec![record], BatchMetadata::default())?)
}

fn expected_positions(total_rows: usize) -> BTreeMap<(String, String), i64> {
    let mut positions = HashMap::new();
    for index in 0..total_rows {
        let (account, stock, side, quantity, flow_type) = row(index);
        if flow_type != "trade" {
            continue;
        }
        let signed = if side == "buy" { quantity } else { -quantity };
        *positions.entry((account, stock)).or_insert(0) += signed;
    }
    positions.into_iter().collect()
}

fn observed_positions(batch: &Batch) -> ProbeResult<BTreeMap<(String, String), i64>> {
    let table = batch.table_payload()?;
    let mut positions = BTreeMap::new();
    for record in table.batches() {
        let accounts = record
            .column(0)
            .as_any()
            .downcast_ref::<StringArray>()
            .ok_or("account column")?;
        let stocks = record
            .column(1)
            .as_any()
            .downcast_ref::<StringArray>()
            .ok_or("stock column")?;
        let values = record
            .column(2)
            .as_any()
            .downcast_ref::<Int64Array>()
            .ok_or("position column")?;
        for index in 0..record.num_rows() {
            positions.insert(
                (
                    accounts.value(index).to_owned(),
                    stocks.value(index).to_owned(),
                ),
                values.value(index),
            );
        }
    }
    Ok(positions)
}

fn plan(partitions: usize) -> ProbeResult<BatchExecutionPlan> {
    let config = DataFusionConfig {
        batch_size: 8_192,
        target_partitions: partitions,
        parallelism_mode: DataFusionParallelismMode::Fixed,
        min_rows_per_partition: 1,
        collect_diagnostics: false,
        ..DataFusionConfig::default()
    };
    let operator = SqlOperator::new(
        "positions",
        QUERY,
        vec!["input".into(), "previous_positions".into()],
        Vec::new(),
    )?;
    Ok(PipelineBuilder::new("sql-plan-reuse-probe")?
        .with_datafusion_config(config)
        .add_node("positions", Box::new(operator))?
        .compile_batch(&UdfRegistry::new().snapshot())?)
}

#[derive(Default)]
struct Phases {
    runs: u64,
    session_create_ns: u64,
    register_ns: u64,
    parse_ns: u64,
    logical_ns: u64,
    physical_ns: u64,
    physical_plans: u64,
    execution_ns: u64,
    wrap_ns: u64,
}

impl Phases {
    fn add(&mut self, metric: &DataFusionQueryMetric) {
        self.runs += 1;
        self.session_create_ns += metric.session_state_create_ns;
        self.register_ns += metric.input_adapter_ns + metric.table_register_ns;
        self.parse_ns += metric.sql_parse_ns;
        self.logical_ns += metric.logical_planning_ns;
        self.physical_ns += metric.physical_planning_ns;
        self.physical_plans += u64::from(metric.physical_planning_count);
        self.execution_ns += metric.execution_ns;
        self.wrap_ns += metric.output_arrow_wrap_ns
            + metric.batch_envelope_ns
            + metric.run_result_ns
            + metric.metrics_traversal_ns;
    }

    /// Mean microseconds per run; the probe reports whole microseconds.
    fn per_run_us(&self, total_ns: u64) -> u64 {
        total_ns / self.runs.max(1) / 1_000
    }
}

async fn measure(total_rows: usize, batch_rows: usize, partitions: usize) -> ProbeResult<()> {
    let plan = plan(partitions)?;
    let batches = total_rows.div_ceil(batch_rows);
    let inputs = (0..batches)
        .map(|index| {
            let start = index * batch_rows;
            input_batch(start, batch_rows.min(total_rows - start))
        })
        .collect::<ProbeResult<Vec<_>>>()?;
    let mut latest = Batch::table(
        vec![RecordBatch::new_empty(position_schema())],
        BatchMetadata::default(),
    )?;
    let mut phases = Phases::default();
    let started = Instant::now();
    for input in &inputs {
        let mut result = plan
            .execute(
                BTreeMap::from([
                    ("input".to_owned(), input.clone()),
                    ("previous_positions".to_owned(), latest.clone()),
                ]),
                ExecutionOptions::default(),
            )
            .await?;
        for metric in &result.datafusion_metrics {
            phases.add(metric);
        }
        latest = result.outputs.remove("output").ok_or("missing output")?;
    }
    let elapsed = started.elapsed();
    let wall_ms = elapsed.as_secs_f64() * 1_000.0;
    let per_batch_us = elapsed.as_micros() / batches.max(1) as u128;
    let observed = observed_positions(&latest)?;
    let expected = expected_positions(total_rows);
    if observed != expected {
        return Err(format!(
            "final positions differ for batch_rows={batch_rows} partitions={partitions}: {} observed vs {} expected keys",
            observed.len(),
            expected.len()
        )
        .into());
    }
    println!(
        "batch_rows={batch_rows} partitions={partitions} batches={batches} wall_ms={wall_ms:.1} \
         per_batch_us={per_batch_us} session_us={} register_us={} parse_us={} logical_us={} \
         physical_us={} execution_us={} wrap_us={} physical_plans={}",
        phases.per_run_us(phases.session_create_ns),
        phases.per_run_us(phases.register_ns),
        phases.per_run_us(phases.parse_ns),
        phases.per_run_us(phases.logical_ns),
        phases.per_run_us(phases.physical_ns),
        phases.per_run_us(phases.execution_ns),
        phases.per_run_us(phases.wrap_ns),
        phases.physical_plans,
    );
    Ok(())
}

fn env_list(name: &str, default: &[usize]) -> Vec<usize> {
    std::env::var(name)
        .ok()
        .map(|value| {
            value
                .split(',')
                .filter_map(|item| item.trim().parse().ok())
                .collect::<Vec<_>>()
        })
        .filter(|values| !values.is_empty())
        .unwrap_or_else(|| default.to_vec())
}

#[tokio::main]
async fn main() -> ProbeResult<()> {
    let total_rows: usize = std::env::var("PLAN_REUSE_PROBE_ROWS")
        .ok()
        .and_then(|value| value.parse().ok())
        .unwrap_or(1_000_000);
    let batch_sizes = env_list("PLAN_REUSE_PROBE_BATCHES", &[1_000, 10_000]);
    let partition_counts = env_list("PLAN_REUSE_PROBE_PARTITIONS", &[1, 4]);
    println!("business_rows={total_rows} accounts={ACCOUNTS} stocks={STOCKS}");
    for &batch_rows in &batch_sizes {
        for &partitions in &partition_counts {
            measure(total_rows, batch_rows, partitions).await?;
        }
    }
    Ok(())
}
