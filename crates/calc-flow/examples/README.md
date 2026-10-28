# Calc Flow Rust examples

Every runnable Python example in [`examples/`](../../../examples/README.md)
has a Rust counterpart below. Core examples live here; examples using native
connectors live in [`calc-flow-connectors/examples/`](../../calc-flow-connectors/examples/).
Run commands from the repository root. No Python interpreter or native Python
extension is needed.

```bash
cargo run -p calc-flow --example datafusion_pipeline
cargo run -p calc-flow --example registered_udf
cargo run -p calc-flow --example symbolic_streaming_recovery
cargo run -p calc-flow --example sql_stream_switch -- --engine both
cargo run -p calc-flow-connectors --example file_source --features file
```

Rust uses `PipelineBuilder`, native operators, immutable `Batch` values, and
owned `StreamingRunner` jobs directly. The `symbolic_*` filenames identify
the matching Python demonstrations; compose their native stages explicitly in
Rust. Named outputs use physical graph bindings, such as `totals.output`.

The array demonstrations register an illustrative `rust-array` provider using
owned Float64 arrays and Rust arithmetic in place of Python's NumPy/JAX hosts.
They preserve the calculations, immutable inputs, matrix shapes, provider gate,
and one-time static-weight placement. They do not call NumPy or JAX. Table
expressions and SQL continue to execute in DataFusion.

## Python-to-Rust inventory

| Python example                                                                                   | Rust example                                                                             | Connector features |
|--------------------------------------------------------------------------------------------------|------------------------------------------------------------------------------------------|--------------------|
| [01_datafusion_pipeline.py](../../../examples/01_datafusion_pipeline.py)                         | [datafusion_pipeline.rs](datafusion_pipeline.rs)                                         | —                  |
| [02_sql_join.py](../../../examples/02_sql_join.py)                                               | [sql_join.rs](sql_join.rs)                                                               | —                  |
| [03_registered_udf.py](../../../examples/03_registered_udf.py)                                   | [registered_udf.rs](registered_udf.rs)                                                   | —                  |
| [04_continuous_runtime.py](../../../examples/04_continuous_runtime.py)                           | [continuous_runtime.rs](continuous_runtime.rs)                                           | —                  |
| [05_async_execution.py](../../../examples/05_async_execution.py)                                 | [async_execution.rs](async_execution.rs)                                                 | —                  |
| [06_numpy_array.py](../../../examples/06_numpy_array.py)                                         | [numpy_array.rs](numpy_array.rs)                                                         | —                  |
| [07_array_and_dataframe.py](../../../examples/07_array_and_dataframe.py)                         | [array_and_dataframe.rs](array_and_dataframe.rs)                                         | —                  |
| [08_streaming_recovery.py](../../../examples/08_streaming_recovery.py)                           | [streaming_recovery.rs](streaming_recovery.rs)                                           | —                  |
| [09_symbolic_financial_features.py](../../../examples/09_symbolic_financial_features.py)         | [symbolic_financial_features.rs](symbolic_financial_features.rs)                         | —                  |
| [10_symbolic_streaming_recovery.py](../../../examples/10_symbolic_streaming_recovery.py)         | [symbolic_streaming_recovery.rs](symbolic_streaming_recovery.rs)                         | —                  |
| [11_symbolic_static_matrix.py](../../../examples/11_symbolic_static_matrix.py)                   | [symbolic_static_matrix.rs](symbolic_static_matrix.rs)                                   | —                  |
| [12_symbolic_stream_join.py](../../../examples/12_symbolic_stream_join.py)                       | [symbolic_stream_join.rs](symbolic_stream_join.rs)                                       | —                  |
| [13_symbolic_relational_dag.py](../../../examples/13_symbolic_relational_dag.py)                 | [symbolic_relational_dag.rs](symbolic_relational_dag.rs)                                 | —                  |
| [14_project_persistence.py](../../../examples/14_project_persistence.py)                         | [project_persistence.rs](project_persistence.rs)                                         | —                  |
| [15_file_source.py](../../../examples/15_file_source.py)                                         | [file_source.rs](../../calc-flow-connectors/examples/file_source.rs)                     | file               |
| [16_kafka_source.py](../../../examples/16_kafka_source.py)                                       | [kafka_source.rs](../../calc-flow-connectors/examples/kafka_source.rs)                   | file,kafka         |
| [17_postgresql_source.py](../../../examples/17_postgresql_source.py)                             | [postgresql_source.rs](../../calc-flow-connectors/examples/postgresql_source.rs)         | file,postgresql    |
| [18_mysql_source.py](../../../examples/18_mysql_source.py)                                       | [mysql_source.rs](../../calc-flow-connectors/examples/mysql_source.rs)                   | file,mysql         |
| [19_clickhouse_source.py](../../../examples/19_clickhouse_source.py)                             | [clickhouse_source.rs](../../calc-flow-connectors/examples/clickhouse_source.rs)         | file,clickhouse    |
| [19_sql_expression_pipeline.py](../../../examples/19_sql_expression_pipeline.py)                 | [sql_expression_pipeline.rs](sql_expression_pipeline.rs)                                 | —                  |
| [20_http_source.py](../../../examples/20_http_source.py)                                         | [http_source.rs](../../calc-flow-connectors/examples/http_source.rs)                     | file,http          |
| [20_streaming_pipeline.py](../../../examples/20_streaming_pipeline.py)                           | [streaming_pipeline.rs](streaming_pipeline.rs)                                           | —                  |
| [21_streaming_outputs.py](../../../examples/21_streaming_outputs.py)                             | [streaming_outputs.rs](streaming_outputs.rs)                                             | —                  |
| [21_websocket_source.py](../../../examples/21_websocket_source.py)                               | [websocket_source.rs](../../calc-flow-connectors/examples/websocket_source.rs)           | file,websocket     |
| [22_kafka_sink.py](../../../examples/22_kafka_sink.py)                                           | [kafka_sink.rs](../../calc-flow-connectors/examples/kafka_sink.rs)                       | file,kafka         |
| [22_stream_asof_join.py](../../../examples/22_stream_asof_join.py)                               | [stream_asof_join.rs](stream_asof_join.rs)                                               | —                  |
| [23_postgresql_sink.py](../../../examples/23_postgresql_sink.py)                                 | [postgresql_sink.rs](../../calc-flow-connectors/examples/postgresql_sink.rs)             | file,postgresql    |
| [24_mysql_sink.py](../../../examples/24_mysql_sink.py)                                           | [mysql_sink.rs](../../calc-flow-connectors/examples/mysql_sink.rs)                       | file,mysql         |
| [25_clickhouse_sink.py](../../../examples/25_clickhouse_sink.py)                                 | [clickhouse_sink.rs](../../calc-flow-connectors/examples/clickhouse_sink.rs)             | file,clickhouse    |
| [26_late_side_output.py](../../../examples/26_late_side_output.py)                               | [late_side_output.rs](late_side_output.rs)                                               | —                  |
| [27_kafka_protobuf_source.py](../../../examples/27_kafka_protobuf_source.py)                     | [kafka_protobuf_source.rs](../../calc-flow-connectors/examples/kafka_protobuf_source.rs) | file,kafka         |
| [28_kafka_custom_decoder.py](../../../examples/28_kafka_custom_decoder.py)                       | [kafka_custom_decoder.rs](../../calc-flow-connectors/examples/kafka_custom_decoder.rs)   | file,kafka         |
| [29_sql_stream_switch.py](../../../examples/29_sql_stream_switch.py)                             | [sql_stream_switch.rs](sql_stream_switch.rs)                                             | —                  |
| [30_streaming_stock_positions.py](../../../examples/30_streaming_stock_positions.py)             | [streaming_stock_positions.rs](streaming_stock_positions.rs)                             | —                  |
| [31_streaming_account_market_values.py](../../../examples/31_streaming_account_market_values.py) | [streaming_account_market_values.rs](streaming_account_market_values.rs)                 | —                  |
| [symbolic_event_window.py](../../../examples/symbolic_event_window.py)                           | [symbolic_event_window.rs](symbolic_event_window.rs)                                     | —                  |

`support/` contains shared Arrow fixtures, replay sources, collectors, matrix
factories, and lazy trade/quote data. In particular,
[`support/stock_account_data.rs`](support/stock_account_data.rs) corresponds to
[`stock_account_data.py`](../../../examples/stock_account_data.py). Source and
sink traits own their asynchronous lifecycle; examples check their results
with assertions that remain enabled in release builds.

## Connector services

Connector examples require their listed Cargo features, plus the same prepared
services, environment variables, and sample destinations as the Python examples.
Follow the [service setup](../../../examples/README.md#connector-readwrite-examples) and
linked connector guides. Sink examples leave two rows in the external
destination; local files and checkpoint directories are temporary.

```bash
cargo run -p calc-flow-connectors --example kafka_source --features file,kafka
cargo run -p calc-flow-connectors --example postgresql_source --features file,postgresql
cargo run -p calc-flow-connectors --example mysql_sink --features file,mysql
cargo run -p calc-flow-connectors --example http_source --features file,http
cargo run -p calc-flow-connectors --example websocket_orders --features websocket
cargo run -p calc-flow-connectors --example websocket_source --features file,websocket
```

Run `websocket_orders` in a separate terminal as the Rust equivalent of
[`services/websocket_orders.py`](../../../examples/services/websocket_orders.py).
It serves the same `orders.jsonl` fixture on `127.0.0.1:8765` until interrupted.
For HTTP, use the static HTTP server command in the HTTP connector guide.

The existing `kafka_protobuf_source` and `kafka_custom_decoder` examples retain
their offline decoder demos. Pass `--consume` after Cargo's `--`, or set
`CALC_FLOW_KAFKA_BOOTSTRAP`, to additionally run
the prepared topic through the managed calculation and Parquet sink. Protobuf
sample production is available with:

```bash
cargo run -p calc-flow-connectors --example kafka_protobuf_source --features file,kafka -- --produce
```

## Stock and account examples

Direct execution keeps Python's defaults: 10,000,000 business flows, 100 accounts,
100 stocks, and batches of 1,000 rows; the valuation demo adds 10,000 quotes.
Both sources generate batches lazily, and sinks retain only the latest snapshot.
Independent arithmetic checks verify final positions and valuations, including
older quotes delivered last. Use a small input for a quick run:

```bash
cargo run -p calc-flow --example streaming_stock_positions -- --rows 30 --accounts 2 --stocks 3 --batch-size 4
cargo run -p calc-flow --example streaming_account_market_values -- --rows 30 --accounts 2 --stocks 3 --batch-size 4 --quote-rows 9
```

## Run the inventory

```bash
python scripts/run_examples.py --surface rust
# Requires all services and prepared sink destinations:
python scripts/run_examples.py --surface rust --include-services
```

The runner skips external services by default, supplies the required connector
features, explicitly consumes both Kafka decoder topics when services are enabled,
and uses the small stock/account inputs shown above. It also retains
`expression_pipeline`, `windowed_streaming`, and the transactional
`late_output_recovery` demonstration. The WebSocket server runs separately.

`export_schema` and `gen_v3_schema` remain explicit tooling commands because
the generator changes a checked-in contract:

```bash
cargo run -p calc-flow --example export_schema
cargo run -p calc-flow --example gen_v3_schema
```
