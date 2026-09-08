# File connector

[Documentation](../README.md) / [Connectors](README.md) / File

Read finite files or directory snapshots as CSV, JSON Lines, or Parquet,
and write Parquet output through an atomic epoch sink.

Identity: `calc-flow-connectors/file/2.0.0`.

## Python example

Run [15_file_source.py](../../examples/15_file_source.py).
`write_input` creates the two orders; `build_project(directory, format_name)`
selects the file codec; `run` waits for natural source completion, reads the
Parquet output, and checks both totals and the exactly-once delivery status.

The default Python wheel includes this connector. After
[preparing the Python environment](README.md#prepare-the-python-environment),
run the standalone [example 15](../../examples/15_file_source.py):

```bash
uv run --no-sync python examples/15_file_source.py
```

The script creates and removes its own input, output, and checkpoint
directories. It repeats the same calculation for CSV, JSON Lines, and Parquet;
each format must produce totals equivalent to `[20.0, 60.0]` with
`exactly_once` delivery. No external service or environment variable is needed.

CSV and Parquet inputs must fit the configured file and batch bounds; JSON
Lines can advance in bounded row chunks.

## Project configuration

These bindings also apply to graphs exported from Python expressions; follow
[expression-to-connector integration](README.md#connect-python-expressions-to-transports)
to supply physical graph bindings and explicit runtime/state settings.

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

See the [connector overview](README.md) for shared delivery, secret,
and recovery rules.
