# PostgreSQL connector

[Documentation](../README.md) / [Connectors](README.md) / PostgreSQL

Read repeatable-read snapshots, poll with composite cursors, or consume
logical CDC. Sinks support append, upsert, and transactional epoch writes.

Identity: `calc-flow-connectors/postgresql/2.0.0`.

## Python example

Run [17_postgresql_source.py](../../examples/17_postgresql_source.py).
`build_project` selects `snapshot` on `calc_flow_example_orders` and resolves
its URL through `CALC_FLOW_PG_URL`. `run` reads one-row pages until the source
ends, verifies Parquet totals `[20.0, 60.0]`, and prints best-effort delivery.

Build a Python wheel with `--features connector-postgresql`, retaining the
default file connector for output. Follow the
[shared environment preparation](README.md#prepare-the-python-environment)
before the service commands below.

```bash
docker run --rm -d --name calc-flow-example-pg -e POSTGRES_PASSWORD=calcflow-example -p 127.0.0.1:5432:5432 postgres:16
```

Wait for database readiness in `docker logs calc-flow-example-pg`, then
prepare the sample table and run the Python consumer:

```bash
docker exec -i calc-flow-example-pg psql -U postgres -v ON_ERROR_STOP=1 <<'SQL'
CREATE TABLE calc_flow_example_orders (
    id BIGINT PRIMARY KEY,
    quantity BIGINT NOT NULL,
    price DOUBLE PRECISION NOT NULL
);
INSERT INTO calc_flow_example_orders VALUES (1, 2, 10.0), (2, 3, 20.0);
SQL
export CALC_FLOW_PG_URL='postgresql://postgres:calcflow-example@127.0.0.1:5432/postgres?sslmode=disable'
uv run --no-sync python examples/17_postgresql_source.py
```

The source reads one repeatable-read transaction and ends naturally. It uses
best-effort delivery because a restarted process cannot recreate that
transaction. `sslmode=disable` here is specific to the local demo service.
For incremental polling and logical CDC configuration, including publication
and slot ownership, see [PostgreSQL](#project-configuration).

After the example completes, stop the disposable demo service:

```bash
docker stop calc-flow-example-pg
```

The `--rm` option removes its container data on stop; the setup above can
recreate it.

## Project configuration

These bindings also apply to graphs exported from Python expressions; follow
[expression-to-connector integration](README.md#connect-python-expressions-to-transports)
to supply physical graph bindings and explicit runtime/state settings.

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

See the [connector overview](README.md) for shared delivery, secret,
and recovery rules.
