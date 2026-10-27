//! Python 21: consume JSON frames from the prepared `WebSocket` service.
mod support;
use serde_json::json;
use support::ExampleResult;
#[tokio::main]
async fn main() -> ExampleResult {
    support::require_env("CALC_FLOW_WS_URL")?;
    let mut registry = support::registry()?;
    calc_flow_connectors::register_websocket_connectors(&mut registry)?;
    let source = support::source(
        "websocket",
        json!({"backpressure":"block","max_frame_bytes":1_048_576,"max_batch_rows":8192,"max_batch_bytes":8_388_608}),
        Some("json"),
        Some(support::secret("CALC_FLOW_WS_URL")),
    );
    support::read_source(source, &registry, true, "best_effort").await
}
