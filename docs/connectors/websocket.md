# WebSocket connector

[Documentation](../README.md) / [Connectors](README.md) / WebSocket

Read bounded JSON frames from a WebSocket endpoint with explicit
backpressure. This connector is source-only.

Identity: `calc-flow-connectors/websocket/2.0.0`.

## Python example

Run [21_websocket_source.py](../../examples/21_websocket_source.py).
`build_project` resolves `CALC_FLOW_WS_URL` and selects blocking backpressure.
`run` waits for two delivered rows, drains the live job, and verifies Parquet
totals `[20.0, 60.0]` with best-effort delivery.

Build a Python wheel with `--features connector-websocket`, retaining the
default file connector for output. Follow the
[shared environment preparation](README.md#prepare-the-python-environment)
before the service commands below.

The small loopback server uses `websockets` solely to prepare the demo; the
consumer is the native Rust connector. Start it with:

```bash
uv run --no-sync --with websockets python examples/services/websocket_orders.py
```

In another terminal:

```bash
export CALC_FLOW_WS_URL=ws://127.0.0.1:8765
uv run --no-sync python examples/21_websocket_source.py
```

The server sends one order per text frame and waits for the consumer to
disconnect. The source uses `backpressure: block`, explicit frame/batch
bounds, and best-effort delivery. Blocking backpressure does not add replay
after a disconnect. See [the WebSocket contract](#project-configuration).

Stop the demo server with Ctrl-C after the consumer completes.

## Project configuration

These bindings also apply to graphs exported from Python expressions; follow
[expression-to-connector integration](README.md#connect-python-expressions-to-transports)
to supply physical graph bindings and explicit runtime/state settings.

WebSocket requires the `url` secret slot and decodes bounded JSON frames.
The Python example uses this source binding:

```json
{
  "binding": "input",
  "connector": {
    "provider": "calc-flow-connectors",
    "name": "websocket",
    "version": "2.0.0"
  },
  "format": {"name": "json", "version": "1"},
  "secrets": {
    "url": {"resolver": "environment", "key": "CALC_FLOW_WS_URL"}
  },
  "options": {
    "backpressure": "block",
    "max_frame_bytes": 1048576,
    "max_batch_rows": 8192,
    "max_batch_bytes": 8388608
  },
  "watermark": {"policy": "disabled"}
}
```

Use `"backpressure": "block"` to pause live reads at the bounded queue, or
choose `"drop_oldest"` to evict the oldest frame and expose a cumulative
dropped counter in source metadata. Both modes are unreplayable;
`drop_oldest` is also explicitly lossy. `max_frame_bytes` must remain below
`max_batch_bytes`. TLS verification is enabled unless `insecure` is
explicitly true.

Output routes fed by WebSocket should request
`"delivery": "best_effort"`. Blocking backpressure does not make a source
replayable and cannot provide exactly-once delivery. The connector does not
provide a WebSocket sink.

See the [connector overview](README.md) for shared delivery, secret,
and recovery rules.
