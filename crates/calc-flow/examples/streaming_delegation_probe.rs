//! Delegated delta-aggregation probe for the streaming stock-positions
//! workload.
//!
//! Runs the example-30 pipeline (filter/CASE expression node feeding a
//! grouped-SUM SQL node, checkpoints disabled) with a fixed `DataFusion`
//! partition configuration on the streaming graph. On main the partition
//! configuration is inert for the compact path, so the same binary is a
//! paired baseline; with delegated delta aggregation the per-batch GROUP BY
//! runs as a partitioned aggregation for batches past the delegation floor.
//!
//! Environment: `DELEGATION_PROBE_PARTITIONS` (default 4), rows and batch
//! size through the shared `--rows/--batch-size` options.
//!
//! Run with:
//! ```bash
//! cargo run -p calc-flow --release --example streaming_delegation_probe -- --rows 10000000 --batch-size 1000000
//! ```

use std::{collections::BTreeMap, sync::Arc, time::Instant};

use calc_flow::{
    DataFusionConfig, DataFusionParallelismMode, PipelineBuilder, SinkBinding, SourceBinding,
    SqlOperator, StreamingRunner, WatermarkPolicy,
};

#[path = "support/stock_account_data.rs"]
mod stock_account_data;
mod support;
use stock_account_data::{DemoSource, LastSink, Latest, Options, check_positions};
use support::ExampleResult;

const QUERY: &str =
    "SELECT account, stock, SUM(delta) AS position FROM input GROUP BY account, stock";

fn partitions() -> usize {
    std::env::var("DELEGATION_PROBE_PARTITIONS")
        .ok()
        .and_then(|value| value.parse().ok())
        .unwrap_or(4)
}

#[tokio::main]
async fn main() -> ExampleResult {
    let options = Options::parse(false)?;
    let partitions = partitions();
    let last: Latest = Arc::default();
    let source = SourceBinding::new(DemoSource::trades(options))
        .with_watermark_policy(WatermarkPolicy::Disabled { idle_timeout: None });
    let sink = SinkBinding::ordinary("collector", LastSink(Arc::clone(&last)))?;
    let config = DataFusionConfig {
        batch_size: 8_192,
        target_partitions: partitions,
        parallelism_mode: DataFusionParallelismMode::Fixed,
        min_rows_per_partition: 1,
        collect_diagnostics: false,
        ..DataFusionConfig::default()
    };
    let operator = SqlOperator::new("positions", QUERY, vec!["input".into()], vec![])?;
    let deltas = support::expression(
        "deltas",
        "",
        &[
            "account",
            "stock",
            "CASE WHEN side = 'buy' THEN quantity ELSE -quantity END AS delta",
        ],
        Some("flow_type = 'trade'"),
    )?;
    let builder = PipelineBuilder::new("streaming-delegation-probe")?
        .with_datafusion_config(config)
        .add_node("deltas", deltas)?
        .add_node("positions", Box::new(operator))?;
    let connected = support::connect_all(builder, &[("deltas", "output", "positions", "input")])?;
    let plan = support::best_effort_plan(connected)?;
    let runner = StreamingRunner::without_checkpoints(
        plan,
        BTreeMap::from([("input".into(), source)]),
        BTreeMap::from([("output".into(), vec![sink])]),
    )?
    .with_runtime_config(options.runtime_config()?)?;
    let started = Instant::now();
    let job = runner.start().await?;
    let outcome = job.wait().await;
    assert_eq!(outcome.state, calc_flow::JobState::Completed, "{outcome:?}");
    let elapsed = started.elapsed();
    let snapshot = last.lock().await.take();
    let result = snapshot
        .as_ref()
        .ok_or("stream produced no position snapshots")?;
    check_positions(result, options)?;
    println!(
        "partitions={partitions} rows={} batch_size={} elapsed_s={:.3} rows_out={}",
        options.rows,
        options.batch_size,
        elapsed.as_secs_f64(),
        result.num_rows()
    );
    Ok(())
}
