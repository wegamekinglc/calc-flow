# HTTP connector

[Documentation](../README.md) / [Connectors](README.md) / HTTP

Poll an HTTP endpoint for JSON Lines with response limits, retries, and
conditional-request validators. This connector is source-only.

Identity: `calc-flow-connectors/http/2.0.0`.

## Python example

Run [20_http_source.py](../../examples/20_http_source.py).
`build_project` resolves `CALC_FLOW_HTTP_URL`, enables conditional requests,
and bounds requests and decoded batches. `run` waits for two delivered rows,
drains the polling job, and verifies Parquet totals `[20.0, 60.0]` with
best-effort delivery.

Build a Python wheel with `--features connector-http`, retaining the
default file connector for output. Follow the
[shared environment preparation](README.md#prepare-the-python-environment)
before the service commands below.

Serve the included [two JSON Lines orders](../../examples/data/orders.jsonl):

```bash
uv run --no-sync python -m http.server 8000 --bind 127.0.0.1 --directory examples/data
```

In another terminal:

```bash
export CALC_FLOW_HTTP_URL=http://127.0.0.1:8000/orders.jsonl
uv run --no-sync python examples/20_http_source.py
```

The source bounds requests and response sizes, enables conditional requests,
and requests best-effort delivery. The static server's Last-Modified validator
prevents unchanged polls from duplicating the sample. A real endpoint must
return newline-delimited JSON objects, with stable ETag or Last-Modified
validators to suppress unchanged responses; a JSON array is a different
format. Validators do not make historical representations replayable.

The calculation uses `cf.table_input`, `pipe(order_totals)`, overloaded
arithmetic/comparison, and `with_columns` → `filter` → `select`, then exports
with `Program.to_project(mode="stream")`. The JSON decoder infers fields in
name order: `id`, `price`, `quantity`. Its declared schema follows that order;
there is no transport option to override the inferred JSON schema. Quantity
is `int64` and is explicitly cast to `float64` before multiplication.

The native connector has no sink direction; this example writes its result
through the file connector. Output files and temporary checkpoints are
removed on exit, and a new invocation starts a new lineage.

Stop the demo server with Ctrl-C after the consumer completes.

## Project configuration

These bindings also apply to graphs exported from Python expressions; follow
[expression-to-connector integration](README.md#connect-python-expressions-to-transports)
to supply physical graph bindings and explicit runtime/state settings.

HTTP requires the `url` secret slot and optionally accepts an `authorization`
secret. The Python example uses this source binding:

```json
{
  "binding": "input",
  "connector": {
    "provider": "calc-flow-connectors",
    "name": "http",
    "version": "2.0.0"
  },
  "format": {"name": "json", "version": "1"},
  "secrets": {
    "url": {"resolver": "environment", "key": "CALC_FLOW_HTTP_URL"}
  },
  "options": {
    "conditional": true,
    "poll_interval_ms": 1000,
    "timeout_seconds": 10,
    "max_retries": 2,
    "max_response_bytes": 8388608,
    "max_batch_rows": 8192
  },
  "watermark": {"policy": "disabled"}
}
```

Add a named secret reference in the `authorization` slot when the endpoint
needs an Authorization header. Bounded polling options also include
`retry_backoff_ms`. TLS verification is enabled unless `insecure` is
explicitly true.

ETag and Last-Modified validators can suppress an unchanged response, but
cannot seek historical endpoint representations. HTTP remains unreplayable;
its output route should request `"delivery": "best_effort"`. The source
does not provide an HTTP sink.

See the [connector overview](README.md) for shared delivery, secret,
and recovery rules.
