//! Python 25: append filtered totals to the prepared `ClickHouse` table.
mod support;
use serde_json::json;
use support::ExampleResult;
#[tokio::main]
async fn main() -> ExampleResult {
    support::require_env("CALC_FLOW_CH_URL")?;
    let mut registry = support::registry()?;
    calc_flow_connectors::register_clickhouse_connectors(&mut registry)?;
    support::write_sink(
        "clickhouse",
        json!({"table":"calc_flow_example_totals","retry_deduplicated":false}),
        Some(support::secret("CALC_FLOW_CH_URL")),
        &registry,
    )
    .await
}
