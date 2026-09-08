# MySQL connector

[Documentation](../README.md) / [Connectors](README.md) / MySQL

Read InnoDB consistent snapshots or poll with a monotonic integer cursor.
Sinks support append, upsert, and recoverable epoch transactions.

Identity: `calc-flow-connectors/mysql/1.0.0`.

## Python example

Run [18_mysql_source.py](../../examples/18_mysql_source.py).
`build_project` selects an InnoDB `snapshot`, resolves `CALC_FLOW_MYSQL_URL`,
and keeps TLS enabled unless the local-demo plaintext opt-in is set. `run`
reads primary-key-ordered pages and verifies Parquet totals `[20.0, 60.0]`
with best-effort delivery.

Build a Python wheel with `--features connector-mysql`, retaining the
default file connector for output. Follow the
[shared environment preparation](README.md#prepare-the-python-environment)
before the service commands below.

```bash
docker run --rm -d --name calc-flow-example-mysql -e MYSQL_ROOT_PASSWORD=calcflow-example -e MYSQL_DATABASE=calcflow -p 127.0.0.1:3306:3306 mysql:8.4
```

Wait for database readiness in `docker logs calc-flow-example-mysql`, then
prepare the sample table and run the Python consumer:

```bash
docker exec -i -e MYSQL_PWD=calcflow-example calc-flow-example-mysql mysql -u root calcflow <<'SQL'
CREATE TABLE calc_flow_example_orders (
    id BIGINT PRIMARY KEY,
    quantity BIGINT NOT NULL,
    price DOUBLE NOT NULL
) ENGINE=InnoDB;
INSERT INTO calc_flow_example_orders VALUES (1, 2, 10.0), (2, 3, 20.0);
SQL
export CALC_FLOW_MYSQL_URL='mysql://root:calcflow-example@127.0.0.1:3306/calcflow'
export CALC_FLOW_MYSQL_PLAINTEXT=1
uv run --no-sync python examples/18_mysql_source.py
```

The source reads a consistent InnoDB snapshot, ordered by primary key, with
best-effort delivery. The plaintext opt-in is for this loopback demo; omit
`CALC_FLOW_MYSQL_PLAINTEXT` for the default verified TLS connection. The
connector identity is `mysql/1.0.0`. Incremental mode requires explicit
monotonic cursor assumptions and does not provide binlog CDC; see
[MySQL](#project-configuration).

After the example completes, stop the disposable demo service:

```bash
docker stop calc-flow-example-mysql
```

The `--rm` option removes its container data on stop; the setup above can
recreate it.

## Project configuration

These bindings also apply to graphs exported from Python expressions; follow
[expression-to-connector integration](README.md#connect-python-expressions-to-transports)
to supply physical graph bindings and explicit runtime/state settings.

Enable the Rust connector crate's `mysql` feature, or build the Python binding
with `--features connector-mysql`. Register Rust factories with
`register_mysql_connectors`; the Python binding registers them automatically
and exposes them through `Runtime().capabilities().connectors`, including Studio's
existing capability API. The default wheel still includes only `file`.

```json
{
  "sources": [{
    "binding": "input",
    "connector": {
      "provider": "calc-flow-connectors",
      "name": "mysql",
      "version": "1.0.0"
    },
    "secrets": {"url": {"resolver": "environment", "key": "CALC_FLOW_MYSQL_URL"}},
    "options": {
      "table": "orders",
      "mode": "incremental_query",
      "cursor_columns": ["id"],
      "assume_monotonic_cursor": true,
      "max_batch_rows": 8192,
      "max_batch_bytes": 8388608,
      "poll_interval_ms": 1000
    },
    "watermark": {"policy": "disabled"}
  }],
  "sinks": [{
    "binding": "output",
    "connector": {
      "provider": "calc-flow-connectors",
      "name": "mysql",
      "version": "1.0.0"
    },
    "secrets": {"url": {"resolver": "environment", "key": "CALC_FLOW_MYSQL_URL"}},
    "options": {
      "table": "orders_out",
      "mode": "transactional",
      "pipeline": "orders-copy",
      "output": "orders",
      "max_epoch_rows": 100000,
      "max_epoch_bytes": 67108864
    },
    "delivery": "exactly_once"
  }]
}
```

The `url` secret contains `mysql://user:password@host:3306/database`; options
reject inline credentials and unknown keys. Identifiers are ASCII names of
1 to 64 bytes, quoted with backticks; select the database in the URL rather
than qualifying `table`. TLS verifies the server certificate and hostname by
default. `tls: false` explicitly enables plaintext for a trusted local test
service. `timeout_seconds` defaults to 30 and bounds database operations.
Connections use UTC and strict SQL conversion; errors expose server codes and
SQLSTATE without SQL values, passwords, or connection URLs.

Both source modes require an existing InnoDB table. `snapshot` is the default
and reads a single repeatable-read consistent transaction in bounded pages,
ordered by the table's required primary key. A snapshot cannot resume its
original transaction after restart; use a best-effort delivery request.
The optional `columns` list contains exact `ArrowFieldSpec` projections,
including nullability. Unsupported types, schema drift, and oversized batches
fail before advancing the returned cursor.

`incremental_query` requires non-null integer cursor columns containing a
complete unique index. Composite cursors use lexicographic ordering and bound
parameters. Set `assume_monotonic_cursor: true` only when rows are immutable,
retained for replay, and become visible in strictly increasing cursor order.
`AUTO_INCREMENT` alone does not establish commit ordering across concurrent
transactions. Updates, deletes, late commits below the cursor, and binlog CDC
are outside this mode's contract. The connector validates the index and types;
the application owns the immutability, retention, and commit-order assumptions.
This mode advertises exact positioning only under those assumptions.

| MySQL type               | Arrow representation                        |
|--------------------------|---------------------------------------------|
| Signed/unsigned integers | Corresponding signed/unsigned integer width |
| `MEDIUMINT`              | `int32` / `uint32`                          |
| `YEAR`                   | `uint16`                                    |
| `FLOAT`, `DOUBLE`        | `float32`, `float64`                        |
| Text, enum, set, JSON    | UTF-8 string                                |
| `DECIMAL`                | Exact decimal string                        |
| Binary, blob, bit        | Binary bytes                                |
| `DATE`                   | `date32`                                    |
| `DATETIME`, `TIMESTAMP`  | `timestamp[us]`, UTC session                |
| `TIME`                   | Signed `HH:MM:SS.ffffff` string             |

`TINYINT(1)` remains an integer on reads. Sinks additionally accept Arrow
booleans and `decimal128`, using bound parameters without float conversion.
Invalid/zero dates and non-finite sink floats fail closed. Unsigned 64-bit
integers retain their full range in Arrow and checkpoint cursor payloads.

Sink modes are `append` (default), `upsert`, and `transactional`. Append and
upsert commit each batch atomically, using multi-row statements capped at
1000 rows and bounded by MySQL parameter limits and the configured byte budget.
All statement chunks share the batch transaction. Upsert uses MySQL's unique/primary key
conflict semantics and updates all supplied columns; it does not accept a
PostgreSQL-style conflict target. Transactional mode stages bounded immutable
Arrow segments, then inserts data and the epoch ledger in one InnoDB
transaction after manifest publication. Recovery validates the sink identity,
epoch, schema, rows, and segment checksum; a repeated epoch with different
content fails even when its row count matches. Retain the reserved
`calc_flow_mysql_epoch_ledger` table for the entire checkpoint lineage. Use
stable, distinct `pipeline`/`output` identities for independent sink lineages.
Target tables must exist; the connector creates only its validated ledger.
Exactly-once assumes transactional target-side effects, so triggers must not
write to nontransactional tables or perform external effects.

The MySQL 8.4 service tests require `CALC_FLOW_CONNECTOR_CONTAINERS=1` and
`CALC_FLOW_MYSQL_TEST_URL`. Run them with
`cargo test -p calc-flow-connectors --features mysql --test mysql_connector -- --ignored --test-threads=1`.
The combined Rust coverage runner requires this service alongside Kafka,
PostgreSQL CDC, and ClickHouse.

See the [connector overview](README.md) for shared delivery, secret,
and recovery rules.
