# Expression computation workflows

[Documentation](README.md) / 2.4 Expression workflows

Calc Flow's Python expression API, imported from `calc_flow`, declares immutable
calculations and lowers them into
the same strict project-v3 graph that the native engine and Studio use. No
Python callback or expression object captured by a symbolic declaration runs
while a lowered native operator executes. Explicitly registered runtime
providers and application-owned Python sources and sinks can still invoke
Python through their normal interfaces. Studio does not contain a second
symbolic compiler. This guide connects the public declarations to batch,
continuous, recovery, array-provider, inspection, and performance workflows
implemented in Calc Flow 4.0.

The complete declaration reference is in the [expression API](symbolic-api.md).
Use this guide to choose an executable example and understand the boundary
between compile-time facts and runtime measurements.

On this page:

- [Compose and run financial features](#compose-and-run-financial-features)
- [Run continuously and recover](#run-continuously-and-recover)
- [Aggregate event-time windows](#aggregate-event-time-windows)
- [Join two symbolic streams](#join-two-symbolic-streams)
- [Use static matrices with NumPy or JAX](#use-static-matrices-with-numpy-or-jax)
- [Read capability failures](#read-capability-failures)
- [Interpret performance output](#interpret-performance-output)
- [Inspect a lowered project in Studio](#inspect-a-lowered-project-in-studio)

## Compose and run financial features

[`09_symbolic_financial_features.py`](../examples/09_symbolic_financial_features.py)
builds a reusable Python function returning a named expression mapping containing one-period simple and log returns,
a three-row price mean, EMA, and standard deviation, a fast/slow MACD,
Bollinger bands, a composed three-row RSI, an exact-time cross-section volume
z-score, and a liquidity-adjusted momentum. The example:

1. declares the input schema and its entity, event-time, and sequence keys;
2. passes the function's mapping to `with_columns` and declares a named output;
3. calls `program.analyze()` and prints `program.explain()`, using a default
   runtime to report physical sharing and bounded-state estimates; and
4. collects Arrow output with `program.collect({"quotes": input_table})["signals"]`.

Declarations only capture names, types, shapes, and expression structure.
They never read the Arrow rows supplied later to `collect` or explicit plan
execution. Use `cf.compute` for the shortest single-input calculation and a
`Program` for reusable logical names. See the [batch guide](batch-guide.md).
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
constant accumulator exactly; [native rolling state](symbolic-design.md#native-rolling-state)
describes the declaration and writer layouts.

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
input must declare the required event-time, entity, and sequence keys, with
non-null `timestamp[us, UTC]` event time. Arrow inference alone does not make a
timestamp field non-null. Build the exact schema before declaring the input.
`program.compile_stream(runtime)` returns a plan whose `source_binding_ids`,
`static_input_ids`, and `sink_binding_ids` are physical graph names, independent
of the logical names used by `Program.collect`. The
lowered rolling or cross-section node remains the only implementation of its
state and watermark rules; the streaming runner checkpoints that native state
using the ordinary project-v3 recovery contract.

## Aggregate event-time windows

[`symbolic_event_window.py`](../examples/symbolic_event_window.py) declares
two UTC one-minute windows grouped by symbol. It computes non-null trade
count, volume, low, high, and arithmetic average price, then derives
`price_range` from the window's high and low columns. The named `minute` and
`summary` outputs share one native window state owner.

```bash
uv run python examples/symbolic_event_window.py
```

The source sends two public `Watermark` events through
`SourceProvidedWatermarks`, each equal to a window end. Each closes its window
and appends final rows in native window/key order. Null-time rows are dropped,
and all-null aggregate
inputs retain the native count/null result rules. The source and sinks use
finite synthetic data with no external service.

The three result rows have volumes `[40, None, 20]`, average prices
`[101.0, None, 100.0]`, and price ranges `[2.0, None, 20.0]`. The example
checks exact Arrow schemas and row order as well as values.

Use `window.tumbling` or `window.hopping` with a non-empty `aggregates`
sequence. The helpers `window.count`, `window.sum`, `window.min`,
`window.max`, and `window.avg` reference named columns. Derive an ordinary
aggregate input before the window with `with_columns`; use `table.project`,
`table.filter`, or `with_columns` to transform final window rows afterward.
The timestamp used for assignment must pass through from the source unchanged
or by a pure rename. Event windows accept nullable time and do not require
rolling's entity and sequence declarations.

Window results have their own row origin. Mixing them with original input
columns or arrays by position is invalid. Each window path permits one window
and stateless table work on either side; joins, rolling/cross-section stages,
another window, and matrix attachment cannot enter that path. Independent
windows and other legal output branches may coexist. The
[symbolic window reference](symbolic-api.md#symbolic-event-time-window-aggregation)
defines types, geometry, stable diagnostics, and lateness-option rules.

For recovery, compile the same declaration again on the same `Runtime`,
create fresh bindings and a runner, and reuse the managed checkpoint root.
Each stream compile returns an independent owning native plan. Open window
state restores through the ordinary native checkpoint protocol; window
declarations add no Python state or checkpoint format.

## Join two symbolic streams

[`12_symbolic_stream_join.py`](../examples/12_symbolic_stream_join.py)
declares authorization and payment inputs, joins equal account keys inside
inclusive event-time bounds, and derives a row-local amount check from the
prefixed output fields. It analyzes and compiles in stream mode, then runs the
same declaration with two independently segmented sources and a fresh plan for
each job:

```bash
uv run python examples/12_symbolic_stream_join.py
```

`table.stream_join` requires the existing public `JoinTimeBounds` and
`JoinStateLimits`; no symbolic copy of those configuration types exists. Both
inputs declare exact schemas plus event-time, entity, and sequence ordering.
The compiler lowers one native `stream_join@1`, so its watermarks, state
eviction, match ordering, metrics, checkpoint v1 state, and recovery rules are
the same ones used by `PipelineBuilder.stream_join`.

A join without output ordering can be a terminal output or feed stateless work.
For nested joins or downstream rolling/cross-section state, declare all of
`output_entity_by`, `output_event_time`, and `output_sequence_by`. Analysis
requires the prefixed left join keys, either prefixed join event time, and the
concatenated prefixed left/right input sequence keys. Missing or projected-away
facts fail with `ordering_required`; metadata never sorts data or runs Python.

[`13_symbolic_relational_dag.py`](../examples/13_symbolic_relational_dag.py)
uses that proof to feed an authorization/payment match into a settlement join.
Independent joins and unrelated output branches may coexist, while each unique
join digest retains one native state owner and checkpoint entry. Matrix
attachment around a join remains unsupported. Event-window paths use the
separate stateless composition rules above and cannot include a join.

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

# Or install calc-flow-python[jax], use backend="jax", and register JAX.
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

- shared expression, rolling, cross-section, event-window, and array stages;
- bounded row or duration retention and watermark finality;
- selected provider identity and calls per micro-batch;
- table-to-dense, host-to-device, and result-attachment copy boundaries; and
- known static-weight bytes.

These are plan estimates and shape facts, not sampled runtime telemetry. Use
execution timings, stream status, provider metadata, and process metrics for
measured latency, resident memory, and copy volume. A plan estimate can explain
where work must occur; it cannot promise a device transfer time or peak RSS.

## Inspect a lowered project in Studio

Export expressions with `program.to_project(runtime, mode="stream")` or the
default batch mode. [Example 14](../examples/14_project_persistence.py) shows the
public export and JSON/YAML/store round trip. Export contains native graph and
input placeholders, without live data, builders, or Python logical aliases.
Stream launch still requires explicit operational bindings and state settings.

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

For implementation details, continue with [symbolic compiler design](symbolic-design.md).
For serialization, continue with [projects and persistence](projects-guide.md).
