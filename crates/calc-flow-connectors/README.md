# calc-flow-connectors

Connector implementations for the Calc Flow continuous runtime. The
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
[the connector guides](../../docs/connectors/README.md); the running process is
authoritative through its capability response.

## TLS dependencies

WebSocket connections use the Mozilla trust anchors from `webpki-roots` 1.0.10
directly through a custom Rustls connector. Certificate and hostname validation
remain enabled by default; only the existing explicit `insecure` option bypasses
verification. Plain `ws://` connections remain supported.

PostgreSQL retains its existing TLS negotiation and verification modes. Until
`pgwire-replication` publishes a release using roots v1, a library-only copy of
its registry 0.4.0 release replaces only that dependency requirement. Its Rust
executable sources are unchanged; the original registry checksum, per-file
hashes and whitespace normalization are recorded in
[the backport provenance](../../vendor/pgwire-replication/provenance.json), with
maintenance instructions in [the backport note](../../vendor/pgwire-replication/README.calc-flow.md).
No Git/alternate registry source, audit ignore or deny-policy exception is added.
