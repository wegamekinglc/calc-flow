# Introduction to Calc Flow

[Documentation](README.md) / 1. Overview

Calc Flow 4.0 calculates over immutable Arrow batches and stateful streams.
Use it to build a graph of calculations, execute a finite dataset, or keep a
calculation running as data arrives. Python and Rust applications use the same
Rust engine. Calc Flow Studio provides a separate local browser interface.

## What you can do

- Calculate columns, select and filter rows, and join tables with read-only
  DataFusion SQL: [batch guide](batch-guide.md).
- Register typed scalar UDFs and explicitly selected NumPy/JAX array providers:
  [batch guide](batch-guide.md#registered-scalar-functions) and
  [array guide](array-guide.md).
- Compute rolling features, cross-section statistics, and bounded event-time
  joins: [symbolic workflows](symbolic-workflows.md).
- Consume async sources, write to sinks, checkpoint state, and resume jobs:
  [streaming guide](streaming-guide.md).
- Persist strict JSON/YAML projects and use registered file, Kafka, PostgreSQL,
  MySQL, ClickHouse, HTTP, or WebSocket connectors:
  [projects](projects-guide.md) and [connectors](connectors.md).
- Edit and inspect projects and operate local jobs in [Studio](studio-guide.md).

## The basic vocabulary

`Batch` is the immutable data envelope passed into and out of a calculation:

- table batches hold Arrow record batches;
- array batches hold a payload for an explicitly selected external provider;
- metadata contains a source identifier, non-negative sequence, and
  JSON-compatible attributes.

A `Port` names an operator input or output and declares its batch kind and
optional exact Arrow schema. An operator performs one calculation. A pipeline
connects operators into a directed acyclic graph. Compilation validates the
connections and required registrations, then returns an execution plan.

A project is the data-only description of a graph and its execution settings.
A streaming job owns a running plan, its sources and sinks, and its lifecycle.
A checkpoint records a consistent source position and operator state from
which a compatible job can recover.

## A first calculation

This small batch calculation adds two Arrow columns:

```python
import pyarrow as pa

from calc_flow import Batch, PipelineBuilder

batch = Batch.from_pyarrow(pa.table({"a": [1, 3], "b": [2, 4]}))
plan = (
    PipelineBuilder("totals").expression("calculate", "total = a + b").compile_batch()
)
result = plan.execute({"input": batch})
assert result.outputs["output"].to_pyarrow()["total"].to_pylist() == [3, 7]
```

Builder methods return new builders. Compilation exposes unconnected inputs
and outputs by name, so the `input` and `output` keys above are graph ports.
Execution returns named batches and timings without mutating the input.

Run [example 01](../examples/01_datafusion_pipeline.py) to extend this pattern
to order totals, projection, and filtering. Rust users can run the equivalent
addition in [expression_pipeline.rs](../crates/calc-flow/examples/expression_pipeline.rs).

## Choose how to declare and execute a calculation

Use `PipelineBuilder` to name operators and connect their ports directly.
Use Python's `calc_flow.symbolic` to compose typed expressions and features,
analyze them before supplying data, and compile them into a native graph.
Both forms produce native execution plans.

Use `compile_batch()` when all inputs are available and you want a `RunResult`.
In Python, `execute()` is the blocking entry point and `execute_async()` is
for an active event loop. See examples [01](../examples/01_datafusion_pipeline.py)
and [05](../examples/05_async_execution.py).

Use `compile_stream()` when inputs arrive over time or the graph needs
event-time progress and recoverable state. A `StreamingRunner` starts the
plan and returns an owning `StreamingJob`. See examples
[04](../examples/04_continuous_runtime.py) and
[08](../examples/08_streaming_recovery.py).

## Supported boundaries

DataFusion 54 executes table expressions and SQL. SQL nodes accept one
read-only `SELECT` or CTE. Array providers are registered explicitly and use
a bounded expression language. Graphs exchange `Batch` values rather than
raw tables or arrays, and configuration contains data and registration
references rather than executable objects.

Projects use format `3`; managed checkpoint manifests also use version `3`.
These protocol versions are separate from package version `4.0.0`.
Streaming delivery is checked per output against source, operator, and sink
capabilities. Ordinary sinks can receive duplicates after recovery; exactly-once
delivery requires a compatible route and transactional or epoch-idempotent
sink evidence. Studio serves `/api/v3` on loopback for local use.

Continue to [getting started](getting-started.md), then choose a program from
the [example learning paths](examples.md). For implementation ownership, read
the separate [architecture guide](design.md).
