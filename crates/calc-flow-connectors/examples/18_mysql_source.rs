//! Python 18: bounded `InnoDB` snapshot with TLS enabled by default.
mod support;
use serde_json::json;
use support::ExampleResult;
#[tokio::main]
async fn main() -> ExampleResult {
    support::require_env("CALC_FLOW_MYSQL_URL")?;
    let mut registry = support::registry()?;
    calc_flow_connectors::register_mysql_connectors(&mut registry)?;
    let source = support::source(
        "mysql",
        json!({"table":"calc_flow_example_orders","mode":"snapshot","tls":std::env::var("CALC_FLOW_MYSQL_PLAINTEXT").as_deref()!=Ok("1"),"max_batch_rows":1,"max_batch_bytes":8_388_608}),
        None,
        Some(support::secret("CALC_FLOW_MYSQL_URL")),
    );
    support::read_source(source, &registry, false, "best_effort").await
}
