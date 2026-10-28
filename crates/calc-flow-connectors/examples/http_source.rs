//! Python 20: poll the prepared JSON Lines HTTP endpoint.
mod support;
use serde_json::json;
use support::ExampleResult;
#[tokio::main]
async fn main() -> ExampleResult {
    support::require_env("CALC_FLOW_HTTP_URL")?;
    let mut registry = support::registry()?;
    calc_flow_connectors::register_http_connectors(&mut registry)?;
    let source = support::source(
        "http",
        json!({"conditional":true,"poll_interval_ms":1000,"timeout_seconds":10,"max_retries":2,"max_response_bytes":8_388_608,"max_batch_rows":8192}),
        Some("json"),
        Some(support::secret("CALC_FLOW_HTTP_URL")),
    );
    support::read_source(source, &registry, true, "best_effort").await
}
