# Example learning paths

[Documentation](README.md) / 1. First programs

The runnable inventory, commands, dependencies, and expected results live in
[examples/README.md](../examples/README.md). Choose a path below after
[installation](getting-started.md). Every numbered Python file runs on its own;
the learning order does not require importing or running an earlier example.

## Calculate a finite dataset

Read the [batch guide](batch-guide.md) while running 01 → 02 → 03 → 05:
calculate and filter order totals, join orders to fees, register a scalar
function, then execute a plan in an asyncio application. Continue to
[example 14](../examples/14_project_persistence.py) to save and reload the graph.

Rust users can pair `expression_pipeline` with `sql_join`. The Rust expression
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
Rust's `continuous_runtime` and `windowed_streaming` demonstrate the native
traits and watermark-driven tumbling windows.

The examples use local application-owned connectors and temporary state
directories. To connect a real transport, use the data-only fragments in the
[connector guide](connectors.md) and supply that transport's service and
credentials separately.

## Compose financial and relational calculations

Read [symbolic workflows](symbolic-workflows.md) while running 09 → 10 → 11
for financial features, recovery, and matrices. Run 12 → 13 for a bounded
two-stream match followed by an ordered nested join. Consult the
[symbolic API](symbolic-api.md) for declaration and ordering requirements.

## Use the local browser application

Read the [Studio guide](studio-guide.md) to edit and inspect the same project
format. Batch graphs and symbolic static-input declarations can be inspected;
the job API requires a connector-backed stream project and does not accept
live static values.

Next: [batch calculations](batch-guide.md).
