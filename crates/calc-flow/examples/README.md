# Calc Flow Rust examples

Every runnable Python example in [`examples/`](../../../examples/README.md)
has a Rust counterpart with the same number and filename stem below (01–36).
Core examples live here; examples using native
connectors live in [`calc-flow-connectors/examples/`](../../calc-flow-connectors/examples/).
Run commands from the repository root. No Python interpreter or native Python
extension is needed.

```bash
cargo run -p calc-flow --example 01_datafusion_pipeline
cargo run -p calc-flow --example 03_registered_udf
cargo run -p calc-flow --example 10_symbolic_streaming_recovery
cargo run -p calc-flow --example 29_sql_stream_switch -- --engine both
cargo run -p calc-flow-connectors --example 15_file_source --features file
```

Rust uses `PipelineBuilder`, native operators, immutable `Batch` values, and
owned `StreamingRunner` jobs directly. The `symbolic_*` filenames identify
the matching Python demonstrations; compose their native stages explicitly in
Rust. Named outputs use physical graph bindings, such as `totals.output`.

The examples retain the Python calculations and output information: result
columns and values, named branches, source cursors, checkpoint epochs, delivery
status, and the total elapsed time in examples 30 and 31. Rust displays Arrow
tables in its native table layout and native graph stages in execution order.
Provider catalogs and diagnostics identify the actual Rust implementation.
Elapsed time is printed as `total elapsed: ... s` with three decimal places;
it covers setup and stream consumption, excludes result printing and Cargo's
build, and varies between runs.

The array demonstrations register an illustrative `rust-array` provider using
owned Float64 arrays and Rust arithmetic in place of Python's NumPy/JAX hosts.
They preserve the calculations, immutable inputs, matrix shapes, provider gate,
and one-time static-weight placement. They do not call NumPy or JAX. Table
expressions and SQL continue to execute in DataFusion.

## Python-to-Rust inventory

| Python example                                                                                   | Rust example                                                                                   | Connector features |
|--------------------------------------------------------------------------------------------------|------------------------------------------------------------------------------------------------|--------------------|
| [01_datafusion_pipeline.py](../../../examples/01_datafusion_pipeline.py)                         | [01_datafusion_pipeline.rs](01_datafusion_pipeline.rs)                                         | —                  |
| [02_sql_join.py](../../../examples/02_sql_join.py)                                               | [02_sql_join.rs](02_sql_join.rs)                                                               | —                  |
| [03_registered_udf.py](../../../examples/03_registered_udf.py)                                   | [03_registered_udf.rs](03_registered_udf.rs)                                                   | —                  |
| [04_continuous_runtime.py](../../../examples/04_continuous_runtime.py)                           | [04_continuous_runtime.rs](04_continuous_runtime.rs)                                           | —                  |
| [05_async_execution.py](../../../examples/05_async_execution.py)                                 | [05_async_execution.rs](05_async_execution.rs)                                                 | —                  |
| [06_numpy_array.py](../../../examples/06_numpy_array.py)                                         | [06_numpy_array.rs](06_numpy_array.rs)                                                         | —                  |
| [07_array_and_dataframe.py](../../../examples/07_array_and_dataframe.py)                         | [07_array_and_dataframe.rs](07_array_and_dataframe.rs)                                         | —                  |
| [08_streaming_recovery.py](../../../examples/08_streaming_recovery.py)                           | [08_streaming_recovery.rs](08_streaming_recovery.rs)                                           | —                  |
| [09_symbolic_financial_features.py](../../../examples/09_symbolic_financial_features.py)         | [09_symbolic_financial_features.rs](09_symbolic_financial_features.rs)                         | —                  |
| [10_symbolic_streaming_recovery.py](../../../examples/10_symbolic_streaming_recovery.py)         | [10_symbolic_streaming_recovery.rs](10_symbolic_streaming_recovery.rs)                         | —                  |
| [11_symbolic_static_matrix.py](../../../examples/11_symbolic_static_matrix.py)                   | [11_symbolic_static_matrix.rs](11_symbolic_static_matrix.rs)                                   | —                  |
| [12_symbolic_stream_join.py](../../../examples/12_symbolic_stream_join.py)                       | [12_symbolic_stream_join.rs](12_symbolic_stream_join.rs)                                       | —                  |
| [13_symbolic_relational_dag.py](../../../examples/13_symbolic_relational_dag.py)                 | [13_symbolic_relational_dag.rs](13_symbolic_relational_dag.rs)                                 | —                  |
| [14_project_persistence.py](../../../examples/14_project_persistence.py)                         | [14_project_persistence.rs](14_project_persistence.rs)                                         | —                  |
| [15_file_source.py](../../../examples/15_file_source.py)                                         | [15_file_source.rs](../../calc-flow-connectors/examples/15_file_source.rs)                     | file               |
| [16_kafka_source.py](../../../examples/16_kafka_source.py)                                       | [16_kafka_source.rs](../../calc-flow-connectors/examples/16_kafka_source.rs)                   | file,kafka         |
| [17_postgresql_source.py](../../../examples/17_postgresql_source.py)                             | [17_postgresql_source.rs](../../calc-flow-connectors/examples/17_postgresql_source.rs)         | file,postgresql    |
| [18_mysql_source.py](../../../examples/18_mysql_source.py)                                       | [18_mysql_source.rs](../../calc-flow-connectors/examples/18_mysql_source.rs)                   | file,mysql         |
| [19_clickhouse_source.py](../../../examples/19_clickhouse_source.py)                             | [19_clickhouse_source.rs](../../calc-flow-connectors/examples/19_clickhouse_source.rs)         | file,clickhouse    |
| [20_http_source.py](../../../examples/20_http_source.py)                                         | [20_http_source.rs](../../calc-flow-connectors/examples/20_http_source.rs)                     | file,http          |
| [21_websocket_source.py](../../../examples/21_websocket_source.py)                               | [21_websocket_source.rs](../../calc-flow-connectors/examples/21_websocket_source.rs)           | file,websocket     |
| [22_kafka_sink.py](../../../examples/22_kafka_sink.py)                                           | [22_kafka_sink.rs](../../calc-flow-connectors/examples/22_kafka_sink.rs)                       | file,kafka         |
| [23_postgresql_sink.py](../../../examples/23_postgresql_sink.py)                                 | [23_postgresql_sink.rs](../../calc-flow-connectors/examples/23_postgresql_sink.rs)             | file,postgresql    |
| [24_mysql_sink.py](../../../examples/24_mysql_sink.py)                                           | [24_mysql_sink.rs](../../calc-flow-connectors/examples/24_mysql_sink.rs)                       | file,mysql         |
| [25_clickhouse_sink.py](../../../examples/25_clickhouse_sink.py)                                 | [25_clickhouse_sink.rs](../../calc-flow-connectors/examples/25_clickhouse_sink.rs)             | file,clickhouse    |
| [26_late_side_output.py](../../../examples/26_late_side_output.py)                               | [26_late_side_output.rs](26_late_side_output.rs)                                               | —                  |
| [27_kafka_protobuf_source.py](../../../examples/27_kafka_protobuf_source.py)                     | [27_kafka_protobuf_source.rs](../../calc-flow-connectors/examples/27_kafka_protobuf_source.rs) | file,kafka         |
| [28_kafka_custom_decoder.py](../../../examples/28_kafka_custom_decoder.py)                       | [28_kafka_custom_decoder.rs](../../calc-flow-connectors/examples/28_kafka_custom_decoder.rs)   | file,kafka         |
| [29_sql_stream_switch.py](../../../examples/29_sql_stream_switch.py)                             | [29_sql_stream_switch.rs](29_sql_stream_switch.rs)                                             | —                  |
| [30_streaming_stock_positions.py](../../../examples/30_streaming_stock_positions.py)             | [30_streaming_stock_positions.rs](30_streaming_stock_positions.rs)                             | —                  |
| [31_streaming_account_market_values.py](../../../examples/31_streaming_account_market_values.py) | [31_streaming_account_market_values.rs](31_streaming_account_market_values.rs)                 | —                  |
| [32_sql_expression_pipeline.py](../../../examples/32_sql_expression_pipeline.py)                 | [32_sql_expression_pipeline.rs](32_sql_expression_pipeline.rs)                                 | —                  |
| [33_streaming_pipeline.py](../../../examples/33_streaming_pipeline.py)                           | [33_streaming_pipeline.rs](33_streaming_pipeline.rs)                                           | —                  |
| [34_streaming_outputs.py](../../../examples/34_streaming_outputs.py)                             | [34_streaming_outputs.rs](34_streaming_outputs.rs)                                             | —                  |
| [35_stream_asof_join.py](../../../examples/35_stream_asof_join.py)                               | [35_stream_asof_join.rs](35_stream_asof_join.rs)                                               | —                  |
| [36_symbolic_event_window.py](../../../examples/36_symbolic_event_window.py)                     | [36_symbolic_event_window.rs](36_symbolic_event_window.rs)                                     | —                  |

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
cargo run -p calc-flow-connectors --example 16_kafka_source --features file,kafka
cargo run -p calc-flow-connectors --example 17_postgresql_source --features file,postgresql
cargo run -p calc-flow-connectors --example 24_mysql_sink --features file,mysql
cargo run -p calc-flow-connectors --example 20_http_source --features file,http
cargo run -p calc-flow-connectors --example websocket_orders --features websocket
cargo run -p calc-flow-connectors --example 21_websocket_source --features file,websocket
```

Run `websocket_orders` in a separate terminal as the Rust equivalent of
[`services/websocket_orders.py`](../../../examples/services/websocket_orders.py).
It serves the same `orders.jsonl` fixture on `127.0.0.1:8765` until interrupted.
For HTTP, use the static HTTP server command in the HTTP connector guide.

The existing `27_kafka_protobuf_source` and `28_kafka_custom_decoder` examples retain
their offline decoder demos. Pass `--consume` after Cargo's `--`, or set
`CALC_FLOW_KAFKA_BOOTSTRAP`, to additionally run
the prepared topic through the managed calculation and Parquet sink. Protobuf
sample production is available with:

```bash
cargo run -p calc-flow-connectors --example 27_kafka_protobuf_source --features file,kafka -- --produce
```

## Stock and account examples

Direct execution keeps Python's defaults: 10,000,000 business flows, 100 accounts,
100 stocks, and batches of 1,000 rows; the valuation demo adds 10,000 quotes.
Both sources generate batches lazily, and sinks retain only the latest snapshot.
Independent arithmetic checks verify final positions and valuations, including
older quotes delivered last. Use a small input for a quick run:

```bash
cargo run -p calc-flow --example 30_streaming_stock_positions -- --rows 30 --accounts 2 --stocks 3 --batch-size 4
cargo run -p calc-flow --example 31_streaming_account_market_values -- --rows 30 --accounts 2 --stocks 3 --batch-size 4 --quote-rows 9
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
