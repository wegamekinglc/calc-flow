# Example learning paths

[Documentation](README.md) / 1. First programs

The runnable inventory, commands, dependencies, and expected results live in
[examples/README.md](../examples/README.md). Choose a path below after
[installation](getting-started.md). Every numbered Python file runs on its own;
the learning order does not require importing or running an earlier example.

## Calculate a finite dataset

Read the [batch guide](batch-guide.md) while running 01 → 05 → 09:
compose and filter order expressions with `cf.compute`, collect several named
outputs, await a calculation in asyncio, then build reusable financial features.
Continue to [example 14](../examples/14_project_persistence.py) to export a
`Program` and reload its native graph through JSON/YAML and the file store.

For advanced integration, run 02 → 03 to join named inputs with explicit SQL
and register a vectorized scalar UDF. Runtime contributors can pair Rust's
`expression_pipeline` with `sql_join`. The Rust expression
program uses the small `[3, 7]` addition from the introduction; Python 01 uses
order totals. The SQL programs share the same order/fee dataset.

## Calculate with arrays

Read the [array guide](array-guide.md) while running 06 → 07 → 11: center a
NumPy array, multiply Arrow columns by NumPy/JAX weights, then reuse static
weights in a symbolic batch or continuous program.

## Operate a recoverable stream

Read the [streaming guide](streaming-guide.md) while running 04 → 08 → 10:
own a source/sink lifecycle, recover a completed stream, then restore a
multi-stage rolling calculation from a checkpoint taken during processing.
For runtime contributors, Rust's `continuous_runtime` and `windowed_streaming` demonstrate the native
traits and watermark-driven tumbling windows.

Those examples use local application-owned connectors and temporary state
directories. Continue with [example 15](../examples/15_file_source.py) for the
native file source in CSV, JSON Lines, and Parquet formats. Examples 16–21
cover Kafka, PostgreSQL, MySQL, ClickHouse, HTTP, and WebSocket sources, each
with a calculation and checked Parquet output. Follow the
[connector overview](connectors/README.md) to choose a transport; each
connector page includes its native wheel feature, Python example, service
preparation, and replay/delivery contract.

## Compose financial and relational calculations

Read [expression workflows](symbolic-workflows.md) while running 09 → 10 → 11
for financial features, recovery, and matrices. Run 12 → 13 for a bounded
two-stream match followed by an ordered nested join. Consult the
[expression API](symbolic-api.md) for declaration and ordering requirements.

## Use the local browser application

Read the [Studio guide](studio-guide.md) to edit and inspect the same project
format. Batch graphs and symbolic static-input declarations can be inspected;
the job API requires a connector-backed stream project and does not accept
live static values.

Next: [batch calculations](batch-guide.md).
