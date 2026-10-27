//! Python 30: compact incremental positions, consuming lazy trade batches.
#[path = "support/stock_account_data.rs"]
mod stock_account_data;
mod support;
use calc_flow::{
    PipelineBuilder, SinkBinding, SourceBinding, StreamRuntimeConfig, StreamingRunner,
    WatermarkPolicy,
};
use std::{collections::BTreeMap, sync::Arc};
use stock_account_data::{DemoSource, LastSink, Latest, Options, check_positions};
use support::{ExampleResult, best_effort_plan, edge, expression, sql};

#[tokio::main]
async fn main() -> ExampleResult {
    let options = Options::parse(false)?;
    let graph = PipelineBuilder::new("stock-positions")?
        .add_node(
            "deltas",
            expression(
                "deltas",
                "",
                &[
                    "account",
                    "stock",
                    "CASE WHEN side = 'buy' THEN quantity ELSE -quantity END AS delta",
                ],
                Some("flow_type = 'trade'"),
            )?,
        )?
        .add_node(
            "positions",
            sql(
                "positions",
                "SELECT account, stock, SUM(delta) AS position FROM input GROUP BY account, stock",
                &["input"],
            )?,
        )?
        .connect(edge("deltas", "output", "positions", "input")?)?;
    let plan = best_effort_plan(graph)?;
    let last: Latest = Arc::default();
    let source = SourceBinding::new(DemoSource::trades(options))
        .with_watermark_policy(WatermarkPolicy::Disabled { idle_timeout: None });
    let job = StreamingRunner::without_checkpoints(
        plan,
        BTreeMap::from([("input".into(), source)]),
        BTreeMap::from([(
            "output".into(),
            vec![SinkBinding::ordinary(
                "collector",
                LastSink(Arc::clone(&last)),
            )?],
        )]),
    )?
    .with_runtime_config(StreamRuntimeConfig {
        checkpointing: false,
        ..StreamRuntimeConfig::default()
    })?
    .start()
    .await?;
    let outcome = job.wait().await;
    assert_eq!(outcome.state, calc_flow::JobState::Completed, "{outcome:?}");
    let last = last.lock().await;
    let result = last
        .as_ref()
        .ok_or("stream produced no position snapshots")?;
    check_positions(result, options)?;
    support::print(std::slice::from_ref(result))
}
