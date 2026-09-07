# ClickHouse connector

[Documentation](../README.md) / [Connectors](README.md) / ClickHouse

Read bounded snapshots or poll using a frozen Arrow schema and composite
cursor. Sink retries can use stable insert-deduplication tokens.

Identity: `calc-flow-connectors/clickhouse/2.0.0`.

## Python example

Run [19_clickhouse_source.py](../../examples/19_clickhouse_source.py).
`build_project` freezes the four-column Arrow schema and selects
`(sequence, id)` as the unique cursor. `run` consumes the bounded snapshot,
verifies Parquet totals `[20.0, 60.0]`, and prints at-least-once delivery.

Build a Python wheel with `--features connector-clickhouse`, retaining the
default file connector for output. Follow the
[shared environment preparation](README.md#prepare-the-python-environment)
before the service commands below.

```bash
docker run --rm -d --name calc-flow-example-ch -e CLICKHOUSE_PASSWORD=calcflow-example -p 127.0.0.1:8123:8123 clickhouse/clickhouse-server:24.12
```

Wait for database readiness in `docker logs calc-flow-example-ch`, then
prepare the sample table and run the Python consumer:

```bash
docker exec -i calc-flow-example-ch clickhouse-client --password calcflow-example --multiquery <<'SQL'
CREATE TABLE calc_flow_example_orders (
    sequence Int64,
    id Int64,
    quantity Int64,
    price Float64
) ENGINE=MergeTree ORDER BY (sequence, id);
INSERT INTO calc_flow_example_orders VALUES (1, 1, 2, 10.0), (2, 2, 3, 20.0);
SQL
export CALC_FLOW_CH_URL='http://default:calcflow-example@127.0.0.1:8123'
uv run --no-sync python examples/19_clickhouse_source.py
```

The source freezes its upper cursor at startup and ends when that snapshot
is consumed. Its Arrow schema explicitly matches all four non-null columns.
The prepared rows make `(sequence, id)` unique; MergeTree ordering itself
does not enforce uniqueness. The example requests at-least-once delivery.
See [ClickHouse](#project-configuration) for polling and sink limits.

After the example completes, stop the disposable demo service:

```bash
docker stop calc-flow-example-ch
```

The `--rm` option removes its container data on stop; the setup above can
recreate it.

## Project configuration

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

See the [connector overview](README.md) for shared delivery, secret,
and recovery rules.
