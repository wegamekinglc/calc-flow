//! Python 22: write two filtered totals with Kafka transactional delivery.
mod support;
use serde_json::json;
use support::ExampleResult;
#[tokio::main]
async fn main() -> ExampleResult {
    let mut registry = support::registry()?;
    calc_flow_connectors::register_kafka_connectors(&mut registry)?;
    support::write_sink("kafka",json!({"bootstrap_servers":std::env::var("CALC_FLOW_KAFKA_BOOTSTRAP").unwrap_or_else(|_|"127.0.0.1:9092".into()),"topic":"calc-flow-example-totals","ledger_topic":"calc-flow-example-totals-ledger","format":"json"}),None,&registry).await
}
