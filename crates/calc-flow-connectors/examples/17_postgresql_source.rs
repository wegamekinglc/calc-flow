//! Python 17: bounded repeatable-read `PostgreSQL` snapshot to Parquet.
mod support;
use serde_json::json;
use support::ExampleResult;
#[tokio::main]
async fn main() -> ExampleResult {
    support::require_env("CALC_FLOW_PG_URL")?;
    let mut registry = support::registry()?;
    calc_flow_connectors::register_postgresql_connectors(&mut registry)?;
    let source = support::source(
        "postgresql",
        json!({"table":"calc_flow_example_orders","mode":"snapshot","max_batch_rows":1,"max_batch_bytes":8_388_608}),
        None,
        Some(support::secret("CALC_FLOW_PG_URL")),
    );
    support::read_source(source, &registry, false, "best_effort").await
}
