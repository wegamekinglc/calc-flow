# Calc Flow 2.0 architecture

Calc Flow is a Rust-native stateful calculation engine for micro-batch and
streaming data. The `calc-flow` crate owns the data contract, graph compiler,
DataFusion execution, project validation, checkpointing, and runner semantics.
The `calc-flow` Python distribution exposes that engine through PyO3 and adds
functional Python builders plus explicitly registered NumPy/JAX providers.

The browser Studio is not part of the core wheel. The separately packaged
`calc-flow-studio` FastAPI application hosts a local v2 API and built React
assets.

## Batch contract

Every graph item is an immutable `Batch` envelope:

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
model. Its `expression`, `sql`, `external`, and `connect` methods return
new builders. `compile()` serializes the strict project and asks the Rust
`Runtime` to validate and compile it.

Unconnected required inputs become named graph inputs. Duplicate port names are
qualified with the node ID. Unconnected outputs become named graph outputs.

## Table execution

DataFusion is the only table query/calculation engine. `ExpressionOperator`
performs one calculated expression or projection, optionally with a filter.
`SqlOperator` performs one read-only `SELECT` or common table expression over
named table aliases.

DDL, DML, utility commands, multi-statement SQL, and table backend selectors are
rejected. Each execution owns a run-scoped DataFusion session and registers
only the UDFs selected by the compiled plan.

An `ExecutionPlan` returns a `RunResult` containing:

- named terminal output batches;
- run and pipeline identity metadata;
- per-node duration plus input/output row counts;
- DataFusion logical/physical plans, planning time, execution time, and rows.

Python's blocking `execute()` rejects calls from a running event loop.
`execute_async()` releases Python execution while Rust runs and supports
cooperative cancellation.

## Trusted extensions

Configurations never contain executable source, Python import paths, or
callable objects.

Rust applications register native DataFusion scalar UDFs in `UdfRegistry` and
external operator factories in `ProviderRegistry`. Compilation consumes an
immutable registry snapshot.

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
`register_jax`. An external v2 node selects provider `numpy` or `jax`,
operator `expression`, version `1`, and a bounded expression.

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

`web-ui/` is a React/TypeScript/Vite application using React Flow. API types
are generated from `web-ui/openapi.json`. The Studio edits v2 project
documents, validates graphs, previews results and metrics, controls runner
checkpoints, and compares local benchmark reports.

The server binds only to loopback. It has no public-hosting or authentication
mode.

## Compatibility

Calc Flow 2.0 does not load Calc Flow 1.x project documents or checkpoints.
Recreate projects with the v2 schema and restart stateful processing from a
chosen source boundary. No automated converter is provided.

Historical v1 documents under `docs/v1-final-api.md` and
`docs/migration-v0.2.md` are references only. The frozen v1 implementation is
available at the `v1-python-final` tag, while `tests/fixtures/v1/` preserves
the semantic corpus used to prove v2 parity.
