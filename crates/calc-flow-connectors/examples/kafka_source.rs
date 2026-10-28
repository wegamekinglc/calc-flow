//! Python 16: read a prepared JSON Kafka topic into managed Parquet output.
mod support;
use serde_json::json;
use support::ExampleResult;
#[tokio::main]
async fn main() -> ExampleResult {
    let mut registry = support::registry()?;
    calc_flow_connectors::register_kafka_connectors(&mut registry)?;
    let source = support::source(
        "kafka",
        json!({"schema":support::order_schema(true),"bootstrap_servers":std::env::var("CALC_FLOW_KAFKA_BOOTSTRAP").unwrap_or_else(|_|"127.0.0.1:9092".into()),"topic":"calc-flow-example-orders","partitions":[0],"auto_offset_reset":"earliest","format":"json","max_batch_rows":8192,"max_batch_bytes":8_388_608}),
        Some("json"),
        None,
    );
    support::read_source(source, &registry, true, "at_least_once").await
}
