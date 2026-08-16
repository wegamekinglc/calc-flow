# calc-flow-connectors

Connector implementations for the Calc-Flow 3.0 continuous runtime. The
crate owns the transport and codec glue; every connector registers through
the trusted `calc_flow::ConnectorRegistry` and implements the A6-public
`StreamSource` / `StreamSink` / `TransactionalStreamSink` lifecycles.

## Feature gates

Each transport compiles behind its own feature; lightweight pure-Rust
format codecs (CSV, newline JSON) are always compiled. The default
feature set is `file`.

| Feature | Surface                                             |
| ------- | --------------------------------------------------- |
| `file`  | Parquet codec, file/directory snapshot source, transactional Parquet sink |

Kafka, PostgreSQL, ClickHouse, HTTP, and WebSocket transports arrive in
later M6 tasks behind their own features.
