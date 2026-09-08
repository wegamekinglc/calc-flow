# Introduction to Calc Flow

[Documentation](README.md) / 1. Overview

Calc Flow is a Python calculation library for Arrow tables and stateful streams.
Write calculations as immutable Python expressions, compute a finite dataset,
or keep the same calculation running as data arrives. The internal Rust runtime
owns table execution, state, and recovery. Calc Flow Studio provides a separate
local browser interface.

## What you can do

- Calculate columns with Python operators, select and filter rows, and collect
  Arrow results: [batch guide](batch-guide.md).
- Use explicit graph construction and read-only DataFusion SQL for advanced
  integrations: [SQL joins](batch-guide.md#named-inputs-and-sql-joins).
- Register typed scalar UDFs and explicitly selected NumPy/JAX array providers:
  [batch guide](batch-guide.md#registered-scalar-functions) and
  [array guide](array-guide.md).
- Compute rolling features, cross-section statistics, and bounded event-time
  joins: [expression workflows](symbolic-workflows.md).
- Consume async sources, write to sinks, checkpoint state, and resume jobs:
  [streaming guide](streaming-guide.md).
- Persist strict JSON/YAML projects and use registered file, Kafka, PostgreSQL,
  MySQL, ClickHouse, HTTP, or WebSocket connectors:
  [projects](projects-guide.md) and [connectors](connectors/README.md).
- Edit and inspect projects and operate local jobs in [Studio](studio-guide.md).

## The basic vocabulary

A `TableExpr` declares a table calculation; `t["price"]` selects a `ColumnExpr`.
Python arithmetic, comparisons, and `&`, `|`, `~` compose expressions.
`with_columns`, `select`, and `filter` return new declarations without changing
inputs. `compute(data, build)` supplies the input schema and returns an Arrow
table. A `Program` gives reusable calculations named inputs and outputs.

For runtime integration, `Batch` is the immutable data envelope:

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
import calc_flow as cf

data = pa.table({"a": [1, 3], "b": [2, 4]})
result = cf.compute(data, lambda t: t.select(total=t["a"] + t["b"]))
assert result.to_pydict() == {"total": [3, 7]}
```

`compute` calls the synchronous builder once, compiles its returned table
expression, and executes in the Rust runtime. The result contains only `total`;
the caller's Arrow table remains unchanged. Run
[example 01](../examples/01_datafusion_pipeline.py) for order totals, named
outputs, projection, and filtering.

## Choose how to declare and execute a calculation

Start with `cf.compute(data, build)` or `await cf.compute_async(data, build)`
for calculations that need no ordering declaration. No explicit schema, input
name, `Batch`, runtime, or plan is needed for supported Arrow data. Types remain
strict; convenience execution does not coerce columns.

Use `cf.table_input(name, schema=...)` and `cf.Program(name, outputs={...})`
for reusable declarations, named outputs, analysis, or project export.
Temporal calculations declare entity, event-time, and sequence keys on
`table_input`; see [temporal ordering](python-api.md#temporal-ordering).
`TableExpr.collect` returns one Arrow table; `Program.collect` returns tables by
logical output name. Async forms use the same calculation and cancellation
contract. See [batch calculations](batch-guide.md).

Use `program.compile_stream()` and `StreamingRunner` when inputs arrive over
time and need event-time progress or recoverable state. Schemas, ordering,
sources, sinks, and managed checkpoints are explicit. Follow
[expression workflows](symbolic-workflows.md) and the
[streaming guide](streaming-guide.md).

For integrations needing diagnostics or owned plan state, compile and execute a
plan directly. `Runtime`, `Batch`, `PipelineBuilder`, formula strings, and SQL
remain supported advanced APIs. Rust crate APIs are documented as
[runtime implementation and extension reference](rust-api.md).

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
