//! Python 23: append two filtered totals to the prepared `PostgreSQL` table.
mod support;
use serde_json::json;
use support::ExampleResult;
#[tokio::main]
async fn main() -> ExampleResult {
    support::require_env("CALC_FLOW_PG_URL")?;
    let mut registry = support::registry()?;
    calc_flow_connectors::register_postgresql_connectors(&mut registry)?;
    support::write_sink(
        "postgresql",
        json!({"table":"calc_flow_example_totals","mode":"append"}),
        Some(support::secret("CALC_FLOW_PG_URL")),
        &registry,
    )
    .await
}
