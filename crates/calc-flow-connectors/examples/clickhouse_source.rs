//! Python 19: `ClickHouse` snapshot with explicit schema and unique cursor.
mod support;
use serde_json::json;
use support::ExampleResult;
#[tokio::main]
async fn main() -> ExampleResult {
    support::require_env("CALC_FLOW_CH_URL")?;
    let mut registry = support::registry()?;
    calc_flow_connectors::register_clickhouse_connectors(&mut registry)?;
    let mut columns = vec![json!({"name":"sequence","data_type":"int64","nullable":false})];
    columns.extend(support::order_schema(false).as_array().unwrap().clone());
    let source = support::source(
        "clickhouse",
        json!({"table":"calc_flow_example_orders","mode":"snapshot","cursor_column":"sequence","tie_breaker_column":"id","tie_breaker_unique":true,"columns":columns,"max_batch_rows":1,"max_batch_bytes":8_388_608}),
        None,
        Some(support::secret("CALC_FLOW_CH_URL")),
    );
    support::read_source(source, &registry, false, "at_least_once").await
}
