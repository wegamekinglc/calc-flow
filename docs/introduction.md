# Calc Flow 4.0 architecture

Calc Flow is a Rust-native stateful calculation engine for micro-batch and
streaming data. The `calc-flow` crate owns the data contract, graph compiler,
DataFusion execution, project validation, checkpointing, and runner semantics.
The `calc-flow` Python distribution exposes that engine through PyO3 and adds
functional Python builders plus explicitly registered NumPy/JAX providers.

The browser Studio is not part of the core wheel. The separately packaged
`calc-flow-studio` FastAPI application hosts the local `/api/v3` continuous
job API and built React assets. The `calc-flow-connectors` crate owns
connector implementations behind per-transport feature gates.

For a component-by-component ownership map and lifecycle sequences, read the
[design and architecture guide](design.md). For a practical source-to-recovery
tutorial, read the [continuous streaming guide](streaming-guide.md). Every
runnable program is indexed in the [example guide](examples.md).

## A first example

The same calculation runs on both surfaces. A one-node expression pipeline named
`totals` computes `total = a + b` over `a = [1, 3]` and `b = [2, 4]` and emits
`[3, 7]`. The Python and Rust snippets below are true twins.

Python:

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

The Python `PipelineBuilder` is a functional projection of the same v3 project
model. Its `expression`, `sql`, `external`, `table_matmul`, and `connect`
methods return new builders. `compile_batch()` and `compile_stream()` serialize
the strict graph project and ask the Rust `Runtime` to validate and compile it.

Rust stream graphs can also contain `UnionOperator`,
`WindowAggregateOperator`, and `StreamJoinOperator`. A `WindowSpec` declares
fixed UTC tumbling or hopping geometry, event-time and grouping columns, and
ordered count, sum, minimum, maximum, or average aggregates. Window nodes are
rejected by `compile_batch()` and accepted by `compile_stream()` after
geometry, schema, aggregate, overlap, and output-name validation. A
`StreamJoinSpec` declares a two-input bounded event-time inner equi-Join with
explicit asymmetric time bounds, required state and match limits, and a
derived prefixed output schema; see the
[continuous streaming guide](streaming-guide.md).

A `RollingSpec` declares native row-window lag, delta, and aggregate outputs
over entity-partitioned, event-time-ordered rows: ordered partition and
sequence keys, a non-null UTC `timestamp[us]` event-time column, allowed
lateness, and a late-row policy; the aggregates — count, sum, mean, variance,
and standard deviation — read a per-entity row frame with a minimum-period
null gate and IEEE infinity semantics. `RollingOperator` compiles into both
batch and stream graphs, emits the input fields followed by the declared
rolling outputs, and checkpoints its stream state at the aligned epoch cut.

A `CrossSectionSpec` declares native complete-group rank, percentile,
z-score, and demean outputs over exact-time or fixed-bucket groups: a
non-null UTC `timestamp[us]` event-time column, ordered entity and sequence
keys, an optional partition key, the grouping, per-output ordering choices
and minimum samples, allowed lateness, and a late-row policy. One micro-batch
is never evidence of completeness — `CrossSectionOperator` accumulates groups
across envelopes, closes each group when the input watermark reaches its
finality coordinate (or at end of input), and emits it once in canonical
order before releasing its state. It compiles into both batch and stream
graphs and checkpoints its open groups at the aligned epoch cut.

`table_matmul` creates a provider-backed `table_matmul@1` external node with
the required mixed-kind inputs `table` (table) and `weights` (array), and the
array output `output`. It selects ordered numeric table columns and multiplies
the resulting table matrix by same-backend weights.

Unconnected required inputs become named graph inputs. Duplicate port names are
qualified with the node ID. Unconnected outputs become named graph outputs.

## Symbolic declarations

Python applications can also declare a calculation symbolically before any
graph is built. The `calc_flow.symbolic` package holds typed immutable column,
table, array, and parameter expressions with canonical v1 digests; an ordered
`FeatureSet` of uniquely named features that `TableExpr.with_columns` appends
as derived columns; and an immutable `Program` over declared inputs and
outputs. The package is declaration, analysis, and compilation
only: it exposes no `eval`, data, or runner path.
`Program.compile_batch(runtime)` and `Program.compile_stream(runtime)` lower
row-local expressions, the `ts` rolling declarations — lag, delta, and the
count/sum/mean/variance/stddev aggregates — and the `cs` cross-section
declarations — rank, percentile, demean, and z-score — into the same strict
project-v3 execution plans, and execution remains owned by the execution
plans and runners above.

Every `Program` carries a runtime-independent v1 fingerprint over its
declaration graph. `Program.analyze(runtime, mode=...)` verifies the program
statically — value types, domains and lineages, symbolic dimensions,
attachment compatibility, state requirements, and stream safety — from the
declaration graph alone, reporting immutable issues on stable output- or
input-rooted paths; `Program.explain(runtime, mode=...)` renders the same
facts deterministically. Type inference proves only what the runtime
capability snapshot proves, so cross-type arithmetic requires an explicit
`row.cast`. The full contract is in the [Python API guide](python-api.md).

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
multiplication. An external v3 node selects provider `numpy` or `jax`, an
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
`format_version: 3`. The Rust `ProjectSpec`, generated JSON Schema, Python
`ProjectDocument`, FastAPI request models, and generated TypeScript contract
all describe the same structure.

Projects contain an explicit batch or stream runtime, graph
nodes/edges/ports, DataFusion settings, UDF references, bounded run options,
and managed state settings. Batch projects may carry sample data sources.
Stream projects bind graph inputs and outputs to exact registered connector
and format identities, name secret references resolved only at connector open,
and declare watermark and per-output delivery requirements. Unknown fields and
duplicate JSON keys are rejected. JSON is canonical storage; YAML is safe
import/export only. `FileProjectStore` writes hashed filenames atomically.

The checked schema is [schemas/project-v3.schema.json](../schemas/project-v3.schema.json).

## Streaming and recovery

Rust and Python expose one source-driven continuous runtime. A
`StreamExecutionPlan` freezes source and sink binding IDs and delivery
requirements. `StreamingRunner` owns that plan, all `SourceBinding` and
`SinkBinding` values, a `ManagedCheckpointRuntime`, and an optional
`StreamRuntimeConfig`. Starting consumes the runner and returns the sole
`StreamingJob` lifecycle owner.

Sources and sinks are async lifecycle connectors. A source declares replay,
delivery, schema, watermark, and batch-bound capabilities before it opens.
Best-effort routes remain explicit in the requested/effective delivery proof.
Ordinary sinks can provide at-least-once delivery when the reachable source is
lossless and replayable; transactional or
epoch-idempotent bindings can prove exactly-once compatibility for a requested
output.

The runtime performs whole-job preflight, runs bounded source/operator/sink
tasks, and owns job-scoped event-time progress, window execution, aligned epoch
checkpoints, manifest-last publication, operator-state restore, transactional
sink completion, terminal checkpoints, deterministic status and metrics, panic
containment, bounded launch cleanup, and owner-settled cancellation. Its
progress driver owns watermark generation, idle/reactivation handling,
deterministic timer ordering, and multi-ingress progress.

Every compiled operator carries its graph node ID as its stable checkpoint
identity and is classified as stateless, versioned checkpointed-stateful, or
unproven. Checkpointed jobs reject unproven operators before task registration,
and recovery checks stored operator-state versions before restore. Each source
binding name remains its stable identity across recovery; its capability
contract is sampled once before open and included in durable identity
validation.

The Rust state surface is public and data-only. `StateBackend` opens an
exclusive `StateLineageBackend`; `LocalStateBackend` manages immutable,
checksum-addressed segments. `CheckpointManifest` is the strict bounded v3
manifest for source, operator, sink, watermark, and segment state. The runtime
uses it as its durable checkpoint and recovery truth. `StreamingJob` exposes
checkpoint, shutdown, cancel, wait, status, and outcome controls without
exposing cursor payloads, connector state, or filesystem paths.

The v2 micro-batch and push runners, project format, Studio REST API, and public
checkpoint-document store are removed. Every
runtime edge uses the public two-field `EdgeBudget`: for
`EdgeBudget::new(R, B)`, at most `R` envelopes, `R` rows, and `B` bytes are
reserved independently. Direct `edge_channel` callers must choose
`R >= max(required_row_limit, required_simultaneous_messages)`. The [stream
message envelope](runtime-envelope.md) documents the exact message, time,
channel, lifecycle, and delivery boundaries.

## Python binding boundary

The Python package lives under `python/calc_flow/`. The native extension is
`calc_flow._native`; pure Python modules provide:

- functional builder and runtime wrappers;
- immutable symbolic declarations, programs, and static analysis
  (`calc_flow.symbolic`);
- strict Pydantic project documents;
- async/blocking project-store adapters;
- the source-driven continuous runner/job adapter;
- NumPy/JAX provider registration and bounded array expressions;
- stable Python exception classes that mirror native error categories.

There is no `src/calc_flow/` execution path in v3.

## Local Studio

`web-ui/backend/` provides the local-only `calc-flow-studio` package.
FastAPI routes are under `/api/v3`. The service stores projects asynchronously
and executes bounded continuous jobs in spawned workers with concurrency,
resident-memory, checkpoint-disk, cancellation, and lifecycle controls.
The UDF-only `/catalog` route remains available under the v3 prefix.
`/capabilities` separately reports the parent runtime-session compile snapshot
and the spawned worker's serialized, lazy-built-in, or unavailable
reconstruction projection.

`web-ui/` is a React/TypeScript/Vite application using React Flow. API types
are generated from `web-ui/openapi.json`. The Studio edits v3 project
documents, validates graphs, observes results and metrics, and controls
continuous-job checkpoints, graceful shutdown, and cancellation.

The server binds only to loopback. It has no public-hosting or authentication
mode.

## Compatibility

Calc Flow 4.0 does not load project-v2 documents or expose Studio `/api/v2`.
Recreate projects with the v3 schema, register connector factories and secret
resolvers in the trusted runtime, and restart stateful processing from a chosen
source boundary. No automated converter is provided. See the
[v2-to-v3 migration guide](migration-v2-to-v3.md) for the upgrade checklist.

The historical [v1 API](v1-final-api.md) and
[v0.2 migration guide](migration-v0.2.md) are references only. The frozen v1
implementation is available at the `v1-python-final` tag, while
[`tests/fixtures/v1/`](../tests/fixtures/v1/) preserves the semantic corpus
as historical parity evidence.
