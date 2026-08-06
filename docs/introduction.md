# Calc Flow 2.0 architecture

Calc Flow is a Rust-native stateful calculation engine for micro-batch and
streaming data. The `calc-flow` crate owns the data contract, graph compiler,
DataFusion execution, project validation, checkpointing, and runner semantics.
The `calc-flow` Python distribution exposes that engine through PyO3 and adds
functional Python builders plus explicitly registered NumPy/JAX providers.

The browser Studio is not part of the core wheel. The separately packaged
`calc-flow-studio` FastAPI application hosts a local v2 API and built React
assets.

## A first example

The same calculation runs on both surfaces. A one-node expression pipeline named
`totals` computes `total = a + b` over `a = [1, 3]` and `b = [2, 4]` and emits
`[3, 7]`. The Python and Rust snippets below are true twins.

Python:

```python
import pyarrow as pa

from calc_flow import Batch, PipelineBuilder

batch = Batch.from_pyarrow(pa.table({"a": [1, 3], "b": [2, 4]}))
plan = PipelineBuilder("totals").expression("calculate", "total = a + b").compile()
result = plan.execute({"input": batch})

assert result.outputs["output"].to_pyarrow()["total"].to_pylist() == [3, 7]
```

Rust ([`crates/calc-flow/examples/expression_pipeline.rs`](../crates/calc-flow/examples/expression_pipeline.rs)):

```rust
use std::{collections::BTreeMap, sync::Arc};

use calc_flow::{
    Batch, BatchMetadata, ExecutionOptions, ExpressionOperator, PipelineBuilder, UdfRegistry,
};
use datafusion::arrow::{array::Int64Array, record_batch::RecordBatch};

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let plan = PipelineBuilder::new("totals")?
        .add_node(
            "calculate",
            Box::new(ExpressionOperator::new(
                "calculate",
                "total = a + b",
                Vec::new(),
                None,
                Vec::new(),
            )?),
        )?
        .compile_batch(&UdfRegistry::new().snapshot())?;
    let input = RecordBatch::try_from_iter(vec![
        (
            "a",
            Arc::new(Int64Array::from(vec![1, 3])) as Arc<dyn datafusion::arrow::array::Array>,
        ),
        ("b", Arc::new(Int64Array::from(vec![2, 4])) as _),
    ])?;
    let result = plan
        .execute(
            BTreeMap::from([(
                "input".into(),
                Batch::table(vec![input], BatchMetadata::default())?,
            )]),
            ExecutionOptions::default(),
        )
        .await?;
    let output = result.outputs["output"].table_payload()?;
    let totals = output.batches()[0]
        .column_by_name("total")
        .expect("expression output contains total")
        .as_any()
        .downcast_ref::<Int64Array>()
        .expect("total is an Int64 column");

    assert_eq!(totals.values(), &[3, 7]);
    println!("calculated totals: {totals:?}");
    Ok(())
}
```

The same example is reused across the [Python](python-api.md), [Rust](rust-api.md),
and [getting started](getting-started.md) guides so the two surfaces read as one
engine.

## Batch contract

Every public graph data item is an immutable `Batch` envelope:

- table batches contain Arrow record batches and are calculated exclusively by
  Apache DataFusion;
- external batches hold an explicitly registered provider payload, such as an
  owned read-only NumPy or JAX array;
- metadata contains a source identifier, non-negative sequence, and
  JSON-compatible attributes.

Rust constructs table batches with `Batch::table`. Python constructs them with
`Batch.from_pyarrow`, accepting an Arrow C Stream provider such as a
`pyarrow.Table`. Python arrays use `Batch.from_array(..., backend="numpy")`
or `backend="jax"` after the matching provider has been registered.

Raw tables and arrays never cross graph, plan, or runner boundaries. Batch
accessors reject the wrong payload kind. Metadata and Python-facing payloads
are defensively copied or made read-only so callers cannot mutate graph-owned
state.

Inside the Rust core, the streaming surface moves stream traffic on one typed
`StreamMessage` per edge: a public data variant wrapping an immutable `Batch`,
plus watermark, barrier, idle, and end-of-input control variants created only
through crate-private constructors. Typed `EventTime` and `Epoch` values carry
event-time progress and checkpoint identity. The message and job context live
in `crates/calc-flow/src/runtime/streaming/`; event time and epoch live in
`crates/calc-flow/src/time/`. The public data contract remains `Batch`; see
the [stream message envelope](runtime-envelope.md) for the full message, time,
and delivery contract.

## Graph compilation

The Rust `PipelineBuilder` owns a directed acyclic graph of named operators
and edges. Every operator declares named input and output `Port` values with a
batch kind, required flag, and optional exact Arrow schema. Compilation checks:

- valid and unique node/port identifiers;
- connected endpoint existence and kind/schema compatibility;
- one writer per input;
- acyclic topology;
- explicit UDF/provider availability;
- deterministic graph inputs, outputs, order, and fingerprint.

The Python `PipelineBuilder` is a functional projection of the same v2 project
model. Its `expression`, `sql`, `external`, `table_matmul`, and `connect`
methods return new builders. `compile()` serializes the strict project and
asks the Rust `Runtime` to validate and compile it.

`table_matmul` creates a provider-backed `table_matmul@1` external node with
the required mixed-kind inputs `table` (table) and `weights` (array), and the
array output `output`. It selects ordered numeric table columns and multiplies
the resulting table matrix by same-backend weights.

Unconnected required inputs become named graph inputs. Duplicate port names are
qualified with the node ID. Unconnected outputs become named graph outputs.

## Table execution

DataFusion is the only table query/calculation engine. `ExpressionOperator`
performs one calculated expression or projection, optionally with a filter.
`SqlOperator` performs one read-only `SELECT` or common table expression over
named table aliases.

DDL, DML, utility commands, multi-statement SQL, and table backend selectors are
rejected. Each table or mixed execution owns one run-scoped DataFusion session
and registers only the UDFs selected by the compiled plan. An external-only
plan stores no DataFusion configuration or UDF snapshot and creates no
DataFusion runtime when it executes.

A `BatchExecutionPlan` returns a `RunResult` containing:

- named terminal output batches;
- run and pipeline identity metadata;
- per-node duration plus input/output row counts;
- DataFusion logical/physical plans, planning time, execution time, and rows.

Python's blocking `execute()` rejects calls from a running event loop.
`execute_async()` releases Python execution while Rust runs and supports
cooperative cancellation. The event-loop rejection happens before blocking
execution validates inputs or options. Both forms accept a keyword-only
`options=` parameter containing a native, frozen `ExecutionOptions` value;
the value's own `ExecutionOptions(settings, deadline)` constructor remains
positional-or-keyword. Its `settings` are a deep-copied, strict JSON mapping
and its optional `deadline` is an absolute, timezone-aware `datetime`. Nested
mappings are copied in one pass, and `settings=None` means empty settings.
Accepted deadlines at any valid UTC offset are normalized to `datetime.UTC`
with microseconds preserved. A deadline that is already expired, or is
crossed at an execution boundary, raises
`calc_flow.CancelledError` after rollback; cancelling a still-pending
surrounding asyncio task instead raises `asyncio.CancelledError` after native
cleanup completes. If native execution is already terminal when cancellation
is handled, its result or exception wins. These checks do not preempt a
non-cooperative callback already in progress. One options value can be reused
safely because each execution receives a fresh native cancellation token.
The absolute deadline continues while a run waits behind another invocation
of the same plan. Cancelling a queued invocation remains isolated from the
active run, and an observed post-provider cancellation or deadline wins over
that provider's error.

An `ExecutionOptions.deadline` is an absolute cooperative engine deadline.
Studio preview `timeout_seconds` is a separate process-level preview limit and
does not inject execution settings or a deadline into the compiled plan.

Single-input and mapping provider callbacks keep their historical
`(batch_or_inputs, provider_options)` ABI when `accepts_context=False`, the
default. Registrations with `accepts_context=True` instead receive
`(batch_or_inputs, provider_options, context)`, where the frozen,
engine-created `ProviderContext` exposes fresh copies of the authoritative run
settings and the normalized deadline. Context delivery is explicit: Calc Flow
does not infer callback arity or retry after `TypeError`, and Python never
exposes the native cancellation token through its public API.

## Trusted extensions

Configurations never contain executable source, Python import paths, or
callable objects.

Rust applications register native DataFusion scalar UDFs in `UdfRegistry` and
external batch or stream operator factories in `ProviderRegistry`.
Compilation consumes an immutable registry snapshot.

Python applications register trusted scalar UDF callbacks on `Runtime` with:

- provider, lowercase function name, and stable version;
- exact Arrow input type names and one return type;
- DataFusion volatility;
- a vectorized callable returning a PyArrow array or scalar.

Nodes explicitly select a `(provider, name, version)` tuple. Calc Flow checks
input types before callback entry and validates output type and length before
downstream execution. Catalog output is sorted JSON-compatible metadata and
does not expose callback objects.

## Optional array providers

NumPy and JAX are Python-owned providers registered with `register_numpy` or
`register_jax`. Where explicitly registered, each provides `expression@1` for
bounded array expressions and mapped `table_matmul@1` for table-array matrix
multiplication. An external v2 node selects provider `numpy` or `jax`, an
operator, and version `1`.

The mapped matrix operator receives a `table` batch and same-backend `weights`
array batch and returns an `output` array batch. It borrows Arrow column
buffers zero-copy at the provider boundary while building one dense host
staging matrix from the selected ordered columns; this is not an end-to-end
zero-copy path. NumPy writes the multiplication into a Rust-owned result.
JAX performs one host-to-device transfer for the table matrix and keeps the
result on the weights device.

External operators receive an engine-neutral run context. They share graph
validation, cancellation, rollback, node timing, runners, and checkpoints with
table operators, but they do not receive or initialize the table engine. An
external-only result therefore has an empty DataFusion metrics list. Mixed
graphs create one DataFusion runtime for their table nodes while keeping the
external provider boundary independent.

The evaluator parses an allowlisted Python expression AST; it never calls
`eval`. It supports arithmetic, bounded literal powers, approved reductions,
transpose, and reshape. Imports, attributes, comprehensions, lambdas, arbitrary
names/calls, unbounded shapes, and backend changes are rejected.

## Projects and schema

The canonical project format is a strict data-only document with
`format_version: 2`. The Rust `ProjectSpec`, generated JSON Schema, Python
`ProjectDocument`, FastAPI request models, and generated TypeScript contract
all describe the same structure.

Projects contain graph nodes/edges/ports, DataFusion settings, UDF references,
sample data sources, and bounded run options. Unknown fields and duplicate JSON
keys are rejected. JSON is canonical storage; YAML is safe import/export only.
`FileProjectStore` writes hashed filenames atomically.

The checked schema is [schemas/project-v2.schema.json](../schemas/project-v2.schema.json).

## Streaming and recovery

Rust and Python expose two runner modes:

- `MicroBatchRunner` opens a replayable source at the committed JSON cursor
  and pulls formed `(Batch, cursor, sequence)` items;
- `StreamingRunner` accepts one formed batch per step.

Sinks are routed by graph output name. Sync and async Python sink callbacks are
supported. Runners validate all routes before leasing a plan or delivering any
output.

After execution, every configured sink must succeed before the checkpoint is
committed. On execution, sink, or checkpoint failure, owned plan state is
restored and the current source item remains retryable. A sink may therefore
see the same batch again: delivery is at least once.

`FileCheckpointStore` writes versioned JSON atomically. A checkpoint identifies
the pipeline name/fingerprint, source cursor/sequence, node-keyed state, and
creation time. One runner has an exclusive lease on a stateful plan.

The crate also contains the complete M2 continuous-runtime skeleton:
whole-job preflight, bounded source/operator/sink tasks, a private runner/job/
reaper lifecycle, deterministic status and metrics, panic containment, and
bounded launch cleanup. Those types remain crate-private. They do not expose a
continuous runner or public watermark/checkpoint interface, and they do not
change the existing public v2 `MicroBatchRunner` or `StreamingRunner`, which
still submit formed `Batch` values only. Public source-driven runner
integration is a separately reviewed post-M5 change.

Every private runtime edge uses the public two-field `EdgeBudget`. For
`EdgeBudget::new(R, B)`, at most `R` envelopes, `R` rows, and `B` bytes may be
reserved independently. The type and function signatures are unchanged.
Direct `edge_channel` callers must choose
`R >= max(required_row_limit, required_simultaneous_messages)`. The [stream
message envelope](runtime-envelope.md) documents the exact message, time,
channel, lifecycle, and current delivery boundaries.

## Python binding boundary

The Python package lives under `python/calc_flow/`. The native extension is
`calc_flow._native`; pure Python modules provide:

- functional builder and runtime wrappers;
- strict Pydantic project documents;
- async/blocking file-store adapters;
- micro-batch and streaming runner adapters;
- NumPy/JAX provider registration and bounded array expressions;
- stable Python exception classes that mirror native error categories.

There is no `src/calc_flow/` execution path in v2.

## Local Studio

`web-ui/backend/` provides the local-only `calc-flow-studio` package.
FastAPI routes are under `/api/v2`. The service stores projects and
checkpoints asynchronously and executes bounded previews in spawned workers
with timeout, CPU, memory, output, cancellation, and lifecycle controls.
The UDF-only `/catalog` compatibility route remains unchanged.
`/capabilities` separately reports the parent runtime-session compile snapshot
and the spawned worker's serialized, lazy-built-in, or unavailable
reconstruction projection.

`web-ui/` is a React/TypeScript/Vite application using React Flow. API types
are generated from `web-ui/openapi.json`. The Studio edits v2 project
documents, validates graphs, previews results and metrics, controls runner
checkpoints, and compares local benchmark reports.

The server binds only to loopback. It has no public-hosting or authentication
mode.

## Compatibility

Calc Flow 2.0 does not load Calc Flow 1.x project documents or checkpoints.
Recreate projects with the v2 schema and restart stateful processing from a
chosen source boundary. No automated converter is provided. See the
[v2 release guide](v2-release.md) for the upgrade checklist.

The historical [v1 API](v1-final-api.md) and
[v0.2 migration guide](migration-v0.2.md) are references only. The frozen v1
implementation is available at the `v1-python-final` tag, while
[`tests/fixtures/v1/`](../tests/fixtures/v1/) preserves the semantic corpus
used to prove v2 parity.
