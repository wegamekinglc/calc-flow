# calc-flow-connectors

Connector implementations for the Calc-Flow 4.0 continuous runtime. The
crate owns the transport and codec glue; every connector registers through
the trusted `calc_flow::ConnectorRegistry` and implements the A6-public
`StreamSource` / `StreamSink` / `TransactionalStreamSink` lifecycles.

## Feature gates

Each transport compiles behind its own feature; lightweight pure-Rust
format codecs (CSV, newline JSON) are always compiled. The default
feature set is `file`.

| Feature      | Surface                                                                   |
|--------------|---------------------------------------------------------------------------|
| `file`       | Parquet codec, file/directory snapshot source, transactional Parquet sink |
| `kafka`      | Kafka source and transactional sink                                       |
| `postgresql` | Snapshot/polling and commit-ordered CDC sources, transactional sink       |
| `mysql`      | InnoDB snapshot/polling source, append/upsert and epoch-ledger sinks      |
| `clickhouse` | Bounded/polling source and deduplicating sink                             |
| `http`       | Polling source with conditional-request validators                        |
| `websocket`  | JSON-lines streaming source                                               |

The available identities and per-transport options are documented in
[docs/connectors.md](../../docs/connectors.md); the running process is
authoritative through its capability response.
