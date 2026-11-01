//! Python 24: append filtered totals to `MySQL` with TLS enabled by default.
mod support;
use serde_json::json;
use support::ExampleResult;
#[tokio::main]
async fn main() -> ExampleResult {
    support::require_env("CALC_FLOW_MYSQL_URL")?;
    let mut registry = support::registry()?;
    calc_flow_connectors::register_mysql_connectors(&mut registry)?;
    support::write_sink("mysql",json!({"table":"calc_flow_example_totals","mode":"append","tls":std::env::var("CALC_FLOW_MYSQL_PLAINTEXT").as_deref()!=Ok("1")}),Some(support::secret("CALC_FLOW_MYSQL_URL")),&registry).await
}
