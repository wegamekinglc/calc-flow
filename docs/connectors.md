# Connector and stream-project guide

Project v3 selects connectors by the exact data-only identity
`(provider, name, version)`. A binding contains non-secret options and named
`SecretReference` values; the trusted factory resolves those references only
when its owning runtime task opens. The default Python wheel contains the file
connector. Kafka, PostgreSQL, ClickHouse, HTTP, and WebSocket are opt-in native
features and appear in `connector_capabilities()` only when compiled in.

The examples below are project fragments. The complete contract is
[`schemas/project-v3.schema.json`](../schemas/project-v3.schema.json), and the
running process is authoritative for available identities and options through
its capability response.

This guide describes transport configuration. Read the
[continuous streaming guide](streaming-guide.md) for source/sink lifecycle,
watermarks, job control, checkpoint transactions, and operational practice.

## Delivery boundaries

| Connector and mode         | Replay                     | Sink completion                          | Maximum claim                          |
| -------------------------- | -------------------------- | ---------------------------------------- | -------------------------------------- |
| File snapshot / Parquet    | Exact file and row cursor  | Atomic epoch directory publication       | Exactly once on supported local FS     |
| Kafka                      | Exact partition offsets    | Transactional target plus compact ledger | Exactly once after ledger preflight    |
| PostgreSQL snapshot        | Unreplayable transaction   | N/A                                      | Best effort source                     |
| PostgreSQL incremental/CDC | Exact composite cursor/LSN | Same-transaction epoch ledger            | Exactly once with transactional sink   |
| ClickHouse polling         | Exact bounded cursor       | Stable insert deduplication token        | At least once; retry deduplicated only |
| HTTP polling               | Unreplayable               | N/A                                      | Best effort                            |
| WebSocket `block`          | Unreplayable               | N/A                                      | Best effort                            |
| WebSocket `drop_oldest`    | Lossy and observable       | N/A                                      | Best effort                            |

HTTP ETag and Last-Modified validators can suppress an unchanged response,
but cannot seek historical endpoint representations. They therefore never
upgrade HTTP to exact replay. ClickHouse deduplication has a server-defined
retention horizon and is not unconditional exactly-once.

Project delivery requests use `"best_effort"`, `"at_least_once"`, or
`"exactly_once"`. An omitted request defaults to at-least-once, but status
reports a best-effort effective guarantee for every route containing a lossy
or unreplayable source. Best-effort requests are never silently upgraded, and
incompatible exactly-once requests fail before connectors open.

## File source and Parquet sink

```json
{
  "sources": [{
    "binding": "input",
    "connector": {
      "provider": "calc-flow-connectors",
      "name": "file",
      "version": "2.0.0"
    },
    "format": {"name": "csv", "version": "1"},
    "options": {
      "path": "data/orders.csv",
      "format": "csv",
      "header": true,
      "max_batch_rows": 8192,
      "max_batch_bytes": 8388608
    },
    "watermark": {"policy": "disabled"}
  }],
  "sinks": [{
    "binding": "output",
    "connector": {
      "provider": "calc-flow-connectors",
      "name": "file",
      "version": "2.0.0"
    },
    "format": {"name": "parquet", "version": "1"},
    "options": {"path": "output", "output": "orders"},
    "delivery": "exactly_once"
  }]
}
```

Paths reject traversal and symlink escapes. Epoch staging stays below the
configured output and becomes visible only through the transactional sink
protocol.

## Kafka

```json
{
  "sources": [{
    "binding": "input",
    "connector": {
      "provider": "calc-flow-connectors",
      "name": "kafka",
      "version": "2.0.0"
    },
    "format": {"name": "json", "version": "1"},
    "options": {
      "bootstrap_servers": "127.0.0.1:9092",
      "topic": "orders",
      "partitions": [0, 1],
      "auto_offset_reset": "earliest",
      "format": "json"
    },
    "watermark": {
      "policy": "bounded_out_of_orderness",
      "column": "event_time",
      "delay_ms": 5000,
      "emit_interval_ms": 1000,
      "idle_timeout_ms": 30000
    }
  }],
  "sinks": [{
    "binding": "output",
    "connector": {
      "provider": "calc-flow-connectors",
      "name": "kafka",
      "version": "2.0.0"
    },
    "format": {"name": "json", "version": "1"},
    "options": {
      "bootstrap_servers": "127.0.0.1:9092",
      "topic": "totals",
      "ledger_topic": "calc-flow-totals-ledger",
      "pipeline": "orders",
      "output": "output",
      "format": "json"
    },
    "delivery": "exactly_once"
  }]
}
```

The ledger topic must be dedicated, have exactly one partition, and use only
`cleanup.policy=compact`. Calc Flow derives the transactional ID from pipeline
and output identity; a project cannot supply it. Recovery validates the exact
prepared record bytes and checks the committed epoch marker before replay.

## PostgreSQL

All PostgreSQL bindings require the secret slot `url`, for example:

```json
"secrets": {
  "url": {"resolver": "environment", "key": "CALC_FLOW_PG_URL"}
}
```

A repeatable-read snapshot uses `"mode": "snapshot"` and is intentionally
unreplayable because a restarted process cannot recreate the same database
transaction. Exact incremental polling uses a unique ordered composite cursor:

```json
"options": {
  "table": "orders",
  "mode": "incremental_query",
  "cursor_columns": ["updated_at", "id"],
  "poll_interval_ms": 500,
  "max_batch_rows": 8192,
  "max_batch_bytes": 67108864
}
```

Logical CDC requires a frozen Arrow schema, an existing publication, and an
explicit durable slot policy:

```json
"options": {
  "table": "orders",
  "mode": "logical_cdc",
  "slot": "calc_flow_orders",
  "publication": "calc_flow_publication",
  "slot_policy": "create_with_snapshot",
  "require_before": true,
  "columns": [
    {"name": "id", "data_type": "int64", "nullable": false},
    {"name": "event_time", "data_type": "timestamp[us]", "nullable": false},
    {"name": "amount", "data_type": "float64", "nullable": false}
  ],
  "max_transaction_rows": 1000000,
  "max_transaction_bytes": 268435456
}
```

The exported snapshot and `pgoutput` start LSN form one gap-free boundary.
Large transactions may produce several bounded batches, but checkpoint
admission remains closed until the transaction's final batch is accepted. A
transactional sink uses `"mode": "transactional"` plus stable `pipeline` and
`output` options; it writes the epoch ledger and target rows in the same
database transaction. `append` and `upsert` remain at-least-once modes.

## ClickHouse

Every source requires a frozen Arrow schema and an explicitly unique composite
cursor. Snapshot mode fixes an upper bound at startup; incremental mode polls
beyond the last accepted `(cursor, tie_breaker)` pair.

```json
"options": {
  "table": "orders",
  "mode": "incremental_query",
  "cursor_column": "event_time",
  "tie_breaker_column": "id",
  "tie_breaker_unique": true,
  "columns": [
    {"name": "event_time", "data_type": "timestamp[us]", "nullable": false},
    {"name": "id", "data_type": "uint64", "nullable": false},
    {"name": "amount", "data_type": "float64", "nullable": false}
  ]
}
```

Source and sink bindings require the `url` secret slot. A sink may set
`"retry_deduplicated": true` with stable `pipeline` and `output` options. It
then persists the exact insert block and token for retry, but the project must
still request `"delivery": "at_least_once"`.

## HTTP and WebSocket

HTTP requires the `url` secret slot and optionally accepts an `authorization`
secret. Bounded polling options include `timeout_seconds`,
`max_response_bytes`, `max_retries`, `retry_backoff_ms`, and `conditional`.
TLS verification is enabled unless `insecure` is explicitly true.

WebSocket requires the `url` secret slot and decodes bounded JSON frames. Use
`"backpressure": "block"` to pause live reads at the bounded queue, or choose
`"drop_oldest"` to evict the oldest frame and expose a cumulative dropped
counter in source metadata. Both modes are unreplayable; `drop_oldest` is also
explicitly lossy. `max_frame_bytes` must remain below `max_batch_bytes`.
Sink bindings on routes fed by HTTP or WebSocket should request
`"delivery": "best_effort"` so the project states the effective contract
directly.

## Union and event-time windows

Project v3 represents the built-in same-schema union directly:

```json
{
  "id": "merge",
  "operator": {"kind": "union"},
  "input_ports": [
    {"name": "left", "kind": "table", "required": true},
    {"name": "right", "kind": "table", "required": true}
  ]
}
```

Window nodes require one exact table input schema. Geometry is expressed in
exact microseconds; `slide_micros` distinguishes a hopping window from a
tumbling window.

```json
{
  "id": "minute_totals",
  "operator": {
    "kind": "window",
    "spec": {
      "event_time_column": "event_time",
      "group_by": ["account"],
      "geometry": {"kind": "tumbling", "size_micros": 60000000},
      "aggregates": [
        {"function": "sum", "column": "amount", "output": "total"}
      ]
    }
  },
  "input_ports": [{
    "name": "input",
    "kind": "table",
    "required": true,
    "schema": [
      {"name": "event_time", "data_type": "timestamp[us]", "nullable": false},
      {"name": "account", "data_type": "string", "nullable": false},
      {"name": "amount", "data_type": "float64", "nullable": false}
    ]
  }]
}
```

## Recovery ownership

Set stream runtime and managed state once at the project root:

```json
{
  "runtime": {
    "mode": "stream",
    "options": {
      "checkpoint_interval_ms": 30000,
      "max_batch_rows": 10000,
      "max_batch_bytes": 67108864
    }
  },
  "state": {"root": ".calc-flow-state/orders", "retention": 3}
}
```

`StreamingRunner` consumes the compiled plan and returns the sole owning job.
A durable manifest records source progress, operator segments, and sink
pre-commit evidence. On restart, the runtime selects the latest complete
manifest, restores every participant before opening the data gate, and
finishes any idempotent post-manifest sink commit. Studio projects use the
same native owner through `/api/v3/jobs`; Studio never exposes raw cursors,
secret values, connector state, or filesystem paths.
