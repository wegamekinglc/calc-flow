# Symbolic computation workflows

Calc Flow's symbolic API declares immutable calculations and lowers them into
the same strict project-v3 graph that the native engine and Studio use. No
Python callback or expression object captured by a symbolic declaration runs
while a lowered native operator executes. Explicitly registered runtime
providers and application-owned Python sources and sinks can still invoke
Python through their normal interfaces. Studio does not contain a second
symbolic compiler. This guide connects the public declarations to batch,
continuous, recovery, array-provider, inspection, and performance workflows
implemented in Calc Flow 4.0.

The complete declaration reference is in the [Python API guide](python-api.md).
Use this guide to choose an executable example and understand the boundary
between compile-time facts and runtime measurements.

## Compose and run financial features

[`09_symbolic_financial_features.py`](../examples/09_symbolic_financial_features.py)
builds one reusable `FeatureSet` containing one-period simple and log returns,
a three-row price mean, EMA, and standard deviation, a fast/slow MACD,
Bollinger bands, a composed three-row RSI, an exact-time cross-section volume
z-score, and a liquidity-adjusted momentum. The example:

1. declares the input schema and its entity, event-time, and sequence keys;
2. calls `Program.analyze(runtime, mode="batch")` before compilation;
3. prints `Program.explain(...)`, including physical sharing and bounded-state
   estimates; and
4. lowers and executes the program with `compile_batch`.

Declarations only capture names, types, shapes, and expression structure.
They never read the Arrow rows used later by `BatchExecutionPlan.execute`.
Structurally identical expressions can therefore be shared by the complete
program without changing the result or mutating the declaration graph.

The independently derived Finance-Python-inspired acceptance vectors live in
[`test_symbolic_finance_reference.py`](../python/tests/test_symbolic_finance_reference.py).
Their provenance is pinned to the upstream
[rolling and cross-section tests](https://github.com/alpha-miner/Finance-Python/tree/3e33d3e70c3458b4c6dcf76b88df6148229b402c/PyFin/tests/Analysis).
They intentionally apply Calc Flow's frozen percentile, tie, Arrow-null, and
NaN rules rather than importing Finance-Python or treating its mutable holder
semantics as an oracle. Rolling operands may be source columns, aliases, pure
row-local expressions, or earlier rolling results. The compiler schedules an
innermost-first DAG and inserts deterministic row-local stages before rolling
and cross-section state when needed. The reference suite includes RSI's delta,
positive/negative projection, rolling means, and final ratio, plus
independently derived EMA and MACD vectors. EWMA uses exact first-valid-sample seeding and the
unadjusted `alpha = 2 / (span + 1)` recurrence. Stream checkpoints persist its
constant accumulator in rolling state layout v2 rather than approximating it
from a retained row window.

Run it from a source checkout with:

```bash
uv run python examples/09_symbolic_financial_features.py
```

## Run continuously and recover

[`10_symbolic_streaming_recovery.py`](../examples/10_symbolic_streaming_recovery.py)
lowers a two-stage rolling program with `compile_stream`, binds an
application-owned replayable source and sink, and uses
`ManagedCheckpointRuntime`. It pauses after three rows, requests an aligned
checkpoint, cancels, and resumes from the stored cursor with both rolling
states restored. A final process-lifecycle run against the terminal checkpoint
also proves that recovery neither reopens the ended source nor duplicates sink
output.

```bash
uv run python examples/10_symbolic_streaming_recovery.py
```

Temporal rolling and cross-section declarations add event-time finality. Their
input must declare the required event-time, entity, and sequence keys. The
lowered rolling or cross-section node remains the only implementation of its
state and watermark rules; the streaming runner checkpoints that native state
using the ordinary project-v3 recovery contract.

## Use static matrices with NumPy or JAX

[`11_symbolic_static_matrix.py`](../examples/11_symbolic_static_matrix.py)
declares a NumPy `weights` parameter, turns selected table columns into a dense
matrix, multiplies them, and attaches the named result column to the table. It
executes the same program in batch and segmented stream modes and verifies that
the immutable weights are placed once for the whole stream job.

```bash
uv run python examples/11_symbolic_static_matrix.py
```

Provider selection is explicit. Register the provider on the exact `Runtime`
used for analysis, explanation, and compilation:

```python
runtime = Runtime()
register_numpy(runtime)  # declarations use backend="numpy"

# Or install calc-flow[jax], use backend="jax", and register JAX.
register_jax(runtime)
```

[`07_array_and_dataframe.py`](../examples/07_array_and_dataframe.py) runs the
equivalent public NumPy and JAX table-to-matrix boundary. A JAX float64
declaration additionally requires JAX x64 support; Calc Flow fails closed
rather than silently narrowing it. In a stream, pass declared values through
`StreamingRunner(..., static_inputs={"weights": weights})`. Static values are
latched and digested before any source opens, and a restart with different
bytes is rejected against the checkpoint lineage.

## Read capability failures

Call `Program.analyze` before compile when displaying several declaration
issues at once. Each `AnalysisIssue` has a stable path, code, and message;
common codes include `capability_mismatch`, `ordering_required`,
`schema_mismatch`, and `unbounded_state`. Compilation raises `CompileError` for
the first unsupported declaration or lowering rule.

The native project compiler performs a final provider gate. For example,
`11_symbolic_static_matrix.py` first compiles without NumPy registration and
shows the strict `missing_provider` `ConfigError`, then registers NumPy and
continues. Do not catch a generic exception and continue with a different
backend: provider identity is part of the compiled plan and its fingerprint.

## Interpret performance output

`Program.explain(runtime, mode=...)` reports deterministic compile-time facts:

- shared expression, rolling, cross-section, and array stages;
- bounded row or duration retention and watermark finality;
- selected provider identity and calls per micro-batch;
- table-to-dense, host-to-device, and result-attachment copy boundaries; and
- known static-weight bytes.

These are plan estimates and shape facts, not sampled runtime telemetry. Use
execution timings, stream status, provider metadata, and process metrics for
measured latency, resident memory, and copy volume. A plan estimate can explain
where work must occur; it cannot promise a device transfer time or peak RSS.

## Inspect a lowered project in Studio

Studio accepts and saves only a strict `ProjectDocument` v3. Selecting a node
shows a **Lowered project inspection** section derived from that document:

- serialized source expressions or lowered rolling/cross-section operations;
- node kind and exact external provider identity;
- bounded state and watermark requirements for native nodes, or `unknown` for
  external-provider lifecycle facts not encoded by `ProjectDocument`;
- static input declarations and known byte sizes; and
- table/array, host/device, static-placement, and result-attachment copy
  boundaries.

Copy-boundary facts are shown only for the recognized direct-matmul document
shape emitted for the built-in `numpy:symbolic_matrix@1` and
`jax:symbolic_matrix@1` providers. Arbitrary external providers and unrecognized
or extended option shapes can attach different semantics to similarly named
fields, so Studio does not infer lifecycle or copy facts for them.

The section is an inspector, not a compiler. It does not reconstruct the
original Python expression objects, execute Python callables, or infer facts
from live row values. Its state sizes are lower bounds or declared limits;
runtime status and metrics remain authoritative.

Studio can inspect a project that declares static inputs, but `POST /jobs`
rejects that project with `422` because the REST contract intentionally has no
field for live static values. Execute such a document through the Python
stream runner, where `static_inputs` is an explicit application-owned mapping.

## Artifact boundary

Published wheels, sdists, and crates contain public runtime code, generated
contracts, examples, and current documentation. Repository-only agent files,
dated implementation plans/specifications under `docs/superpowers`, and design
workspaces are not public package content. The release inspector enforces this
boundary for every artifact; see the [Python release guide](python-release.md)
for the complete build and isolated-install smoke procedure.
