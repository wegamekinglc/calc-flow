# Symbolic Computation Engine Design

## Status

The SCE-00 contract was approved on 2026-08-22 after one blocker-only
correction round. This document remains a point-in-time design record, not
current API guidance and not evidence that the described symbolic surface or
native operators have been implemented. The controlling semantic freeze is
[Symbolic Computation Contract Freeze](../../../.codex/artifacts/specs/symbolic-computation-contract.md),
the exact public and serialized shape is
[Symbolic Computation Engine API Contract](../../../.codex/artifacts/api-notes/symbolic-computation-engine.md),
and the final gate is
[Symbolic Computation Engine Contract Critique](../../../.codex/artifacts/critiques/symbolic-computation-engine.md).
The phased delivery plan is
[Symbolic Computation Engine Implementation Plan](../plans/2026-08-22-symbolic-computation-engine.md).

## Problem

Calc Flow exposes immutable table and array `Batch` values, functional graph
construction, DataFusion table execution, registered NumPy/JAX array
providers, and a checkpointed continuous runtime. Users must nevertheless
construct relatively low-level graph nodes and manually preserve the semantic
differences between row-local expressions, entity time series, event-time
windows, cross sections, and matrices.

The requested Python symbolic layer must make those calculations strongly
composable without becoming a second calculation engine. In particular, it
must not reintroduce a Python implementation of table, matrix, window, or
stream state. It must produce calc-flow plans whose batch and streaming
results do not depend on arbitrary micro-batch boundaries.

The design takes inspiration from Finance-Python's
[`PyFin.Analysis`](https://github.com/alpha-miner/Finance-Python/tree/master/PyFin/Analysis)
package:

- value holders compose through operator overloading;
- field dependencies and lookback windows propagate through compositions;
- time-series and cross-sectional transforms have separate concepts; and
- higher-level indicators are assembled from a smaller accumulator catalog.

It intentionally does not reproduce the reference implementation's mutable
`push`/`value` holders, per-security state objects, deep-copied execution
trees, Pandas transformation path, or one-class-per-indicator hierarchy.

## Goals

- Add a Python-native, immutable symbolic DSL under `calc_flow.symbolic`.
- Keep the symbolic package limited to graph construction, static analysis,
  optimization, lowering, and native plan compilation.
- Make ordinary Python functions the primary composition mechanism.
- Represent row-local, temporal, cross-sectional, windowed, and tensor
  semantics explicitly enough to reject unsafe stream plans before startup.
- Lower calculations to the existing DataFusion, Rust operator, provider,
  checkpoint, and runner boundaries.
- Preserve strict, data-only project documents with no callable, pickle,
  import path, closure, secret value, or executable object.
- Produce deterministic projects, fingerprints, errors, explain output, and
  observable result ordering.
- Keep runtime throughput near equivalent hand-built calc-flow plans through
  projection fusion, common-subexpression elimination, state sharing, and
  vectorized provider calls.
- Support batch and final-only append-only streaming for the approved initial
  primitive catalog.

## Non-Goals

- Symbolic algebra solving, differentiation, or SymPy compatibility.
- Compatibility classes or aliases for Finance-Python value holders.
- A Python evaluator, preview evaluator, fallback engine, or state store.
- Point-for-point reimplementation of every Finance-Python indicator.
- Arbitrary Python stateful primitives in serialized projects.
- Streaming joins, retraction/changelog outputs, or dynamic matrix-weight
  streams in the first release.
- Implicit conversion between Arrow tables and Array API values.
- Making project documents carry live `Batch`, Arrow, NumPy, or JAX values.
- Replacing `ProjectDocument` as the canonical executable project contract.

## Architectural Boundary

The symbolic layer has no data execution API. Its complete data path is:

```text
Python symbolic API
        |
        v
immutable symbolic DAG
        |
        +-- type, shape, domain, lineage, and state analysis
        +-- capability validation
        +-- CSE, fusion, and deterministic lowering
        |
        v
strict calc-flow project v3
        |
        v
Rust graph compiler
        |
        +-- BatchExecutionPlan
        |     +-- DataFusion table work
        |     +-- native rolling/cross-section kernels
        |     `-- registered array providers
        |
        `-- StreamExecutionPlan
              +-- watermarks and event-time finality
              +-- bounded operator state
              +-- epoch checkpoints and recovery
              `-- vectorized stateless providers
```

The symbolic package may expose `compile_batch()` and `compile_stream()`
because compilation is declaration processing. It must not expose `eval()`,
`push()`, `value()`, `transform()`, or a convenience method that accepts data
and silently executes it. Users continue to call `BatchExecutionPlan.execute`
or construct `StreamingRunner` explicitly.

## Public Python Surface

The initial public concepts are intentionally small:

```python
Expr[T]
ColumnExpr
TableExpr
ArrayExpr
Parameter
FeatureSet
Program
```

The primary namespaces are:

- `row`: row-local scalar and column transforms;
- `ts`: entity-partitioned temporal transforms;
- `cs`: complete event-time cross-section transforms;
- `window`: final event-time aggregations that change row cardinality;
- `table`: projection, filtering, explicit table/array attachment, and later
  relational composition; and
- `linalg`: explicit table-to-array projection and Array API expressions.

A representative declaration is:

```python
from calc_flow import Runtime
from calc_flow import symbolic as sf

quotes = sf.table_input(
    "quotes",
    schema=(
        sf.Field("ts", "timestamp[us, UTC]", nullable=False),
        sf.Field("sequence", "uint64", nullable=False),
        sf.Field("symbol", "string", nullable=False),
        sf.Field("industry", "string"),
        sf.Field("close", "float64"),
        sf.Field("volume", "float64"),
    ),
    entity_by=("symbol",),
    event_time="ts",
    sequence_by=("sequence",),
)

log_price = sf.row.log(quotes["close"])
return_1 = sf.ts.delta(log_price, periods=1)
momentum_20 = sf.ts.mean(
    return_1,
    window=sf.rows(20),
    min_periods=20,
)
alpha = sf.cs.zscore(
    momentum_20,
    group=sf.exact_time(
        quotes["ts"],
        partition_by=(quotes["industry"],),
    ),
)

features = quotes.with_columns(
    sf.FeatureSet(
        (
            ("return_1", return_1),
            ("momentum_20", momentum_20),
            ("alpha", alpha),
        )
    )
)

weights = sf.parameter(
    "weights",
    kind="array",
    backend="numpy",
    dtype="float64",
    shape=(3, 1),
    mutability="static",
)
matrix = sf.linalg.from_columns(
    features,
    columns=("return_1", "momentum_20", "alpha"),
    backend="numpy",
)
score = sf.linalg.matmul(matrix, weights)
signals = sf.table.attach_columns(features, score, names=("score",))

program = sf.Program(
    "alpha-signals",
    inputs=(quotes, weights),
    outputs=(("signals", signals),),
)
batch_plan = program.compile_batch(Runtime())
stream_plan = program.compile_stream(
    Runtime(),
    allowed_lateness_micros=2_000_000,
    late_policy="error",
)
```

These signatures are frozen by the SCE-00 API contract, but the surface remains
proposed until its dependent implementation issues land.

## Composition Model

Ordinary Python functions compose symbolic expressions immediately. The
callable is not retained or serialized:

```python
def winsorized_zscore(
    value: sf.ColumnExpr,
    *,
    group: sf.CrossSectionGroup,
    lower: float,
    upper: float,
) -> sf.ColumnExpr:
    winsorized = sf.cs.winsorize(
        value,
        group=group,
        lower=lower,
        upper=upper,
    )
    return sf.cs.zscore(
        winsorized,
        group=group,
    )
```

`FeatureSet` owns an ordered, duplicate-free set of named expressions. It
allows the compiler to produce one projection or one shared stateful node for
many outputs instead of materializing a node per Python expression.

Inputs are treated as read-only. Constructors defensively copy caller-owned
mappings and sequences into immutable tuples or frozen JSON values. Public
expression objects do not expose their internal collections for mutation.

Column and array comparison operators return symbolic expressions in their own
domains; table expressions reject scalar operators and require table namespace
operations. Public expressions use `identical(other)` for structural identity,
while expression-containing grouping, feature, and program values compare only
by object identity. Compiler-internal node keys retain ordinary value equality.

## Symbolic Intermediate Representation

The declaration node is deliberately smaller than its analysis result:

```python
@dataclass(frozen=True, slots=True)
class ExprNode:
    op: OpRef
    args: tuple[ExprNode, ...]
    attrs: FrozenJson
```

`OpRef` contains a stable primitive name and version. Built-in primitives are
resolved through a closed symbolic catalog. Custom native/provider primitives
are referenced by explicit provider, name, and version identities already
understood by `RuntimeCapabilities`; project documents never store a Python
lowering callback.

The analyzer returns immutable side tables keyed by canonical node digest. It
does not write inferred facts back into declaration nodes.

### Value Types

The value-type system covers:

- Arrow scalar and column data types with nullability;
- exact table schemas;
- array backend, dtype, rank, and symbolic dimensions;
- boolean expressions; and
- scalar literals and external parameters.

Table numeric promotion follows DataFusion/Arrow rules. Array promotion uses
the selected provider's safe dtype rules. Table-to-array and array-to-table
movement is always explicit. Arrow null and floating-point NaN remain distinct
semantic values.

### Domains

Every analyzed value has one domain:

- `ScalarDomain`: a literal or external scalar;
- `RowDomain`: an input-table row-aligned value;
- `TemporalDomain`: an entity-partitioned, event-time-ordered value;
- `CrossSectionDomain`: a value over a complete event-time/bucket group;
- `WindowDomain`: a cardinality-changing final window result; or
- `TensorDomain`: an array with shape and optional table row-axis lineage.

Domain transitions are explicit. A matrix derived from table columns retains a
row-axis lineage token. It can be attached only to a table carrying the same
lineage and only when its known output width matches the declared names.

### State Requirements

Each primitive declares one of:

- `stateless`;
- `rows(n)`;
- `duration(micros)`;
- `constant_state`, such as an exponentially weighted statistic;
- `cross_section_barrier`;
- `static_side_input`; or
- `unbounded`.

Unbounded state is rejected in stream mode by default. The analyzer estimates
state from window width, retained columns, active group assumptions, and
primitive-specific accumulator costs. Estimates are explanatory and admission
inputs; runtime resident-memory limits remain authoritative.

### Stream Contracts

Primitive metadata records:

- supported lifecycle modes;
- whether the result is micro-batch invariant;
- watermark and ordering requirements;
- statefulness and checkpoint support;
- determinism and replay safety; and
- append-only/final-only output behavior.

The first release supports only final, append-only stateful outputs. A
primitive that would require an earlier output to be retracted is rejected.

## Analysis and Lowering

Compilation follows one deterministic pipeline:

1. Copy and freeze input declarations.
2. Normalize literals, parameters, and primitive attributes.
3. Infer value types, domains, lineages, dependencies, and state requirements.
4. Validate the selected runtime mode and capability snapshot.
5. Validate ordering, watermark, lateness, bounded state, and replay contracts.
6. Eliminate common subexpressions.
7. Share compatible temporal or cross-section state.
8. Fuse consecutive row-local expressions.
9. Fuse consecutive array expressions.
10. Insert materialization only at domain, lifecycle, or backend boundaries.
11. Generate deterministic node IDs and one strict project-v3 document.
12. Invoke the existing Rust graph compiler for port, kind, schema, UDF,
    topology, input/output, and fingerprint validation.

The compile cache key includes the symbolic fingerprint, mode, exact input
schemas, runtime capability schema and revision, and selected provider/UDF
versions. It must never depend on Python object identity.

## Batch and Stream Semantics

### Row-Local Expressions

Arithmetic, comparison, boolean, `where`, cast, and supported scalar functions
lower to fused `ExpressionOperator` projections. These operations are safe in
stream mode only when they are row local. Arbitrary SQL window, aggregate, or
cross-batch behavior must not be hidden inside a stream expression node.

### Rolling Temporal Transforms

A new `RollingOperator` is required. Its batch and stream lifecycles share the
same Rust/Arrow calculation kernels and semantic rules; only scheduling,
reorder buffering, and durable state differ.

The initial functions are:

- lag and delta;
- count, sum, mean, min, and max;
- variance and standard deviation; and
- covariance and correlation.

Row windows use bounded shared history. Duration windows use event time and
evict values that have left the declared interval. Implementations use compact
per-entity rows, reversible accumulators, numerically stable variance state,
and monotonic queues rather than per-symbol Python objects.

Each input row has a deterministic total order from partition keys, event
time, and sequence fields. A stream buffers bounded out-of-order rows until a
watermark makes them final. Checkpoints persist both active windows and
unfinalized reorder buffers.

### Cross-Section Transforms

A new `CrossSectionOperator` is required for rank, percentile, z-score,
demean, and winsorization.

Its semantic group is the exact event time or declared time bucket plus
optional group keys. Batch execution receives complete groups. Stream
execution buffers a group until the watermark has passed its end plus allowed
lateness, then sorts/calculates/emits it once and releases state.

Tie method, sort direction, null placement, minimum sample count, and sample
versus population standard deviation are explicit configuration. The engine
must never assume that one incoming micro-batch is a complete cross section.

### Existing Event-Time Aggregates

The existing `WindowAggregateOperator` remains the cardinality-changing
tumbling/hopping final aggregate. It is not reused to pretend that a rolling
per-input-row feature has event-window semantics. The symbolic catalog exposes
the two concepts separately.

### Matrix Operations

Existing NumPy/JAX expressions and `table_matmul` remain provider-owned
calculation capabilities. The symbolic layer performs only dtype, shape,
lineage, and capability analysis before emitting provider nodes.

Streaming matrix support requires:

- an explicitly registered stateless stream-provider factory;
- a `microbatch_invariant` capability promise;
- one provider call per fused array segment and micro-batch; and
- immutable static side inputs for weights.

The frozen runner boundary is:

```python
StreamingRunner(
    plan,
    sources=continuous_sources,
    sinks=sinks,
    static_inputs={"weights": weight_batch},
)
```

Static inputs are validated and latched before operator tasks start. Static
declarations participate in the semantic plan fingerprint. Payload digests
participate only in prepared-job identity and manifest compatibility after the
existing pipeline name and semantic plan fingerprint select a lineage. A
payload digest never enters `StateLineageKey`, the lineage-directory hash, the
semantic plan fingerprint, or manifest discovery filters. Recovery with a
missing, extra, or changed digest therefore reaches the existing lineage and
fails before sources open. Dynamic weights are outside the first release.

Only row-axis-independent array work is stream safe initially: elementwise
operations, feature-axis reductions, and per-row multiplication by static
weights. An array reduction across arbitrary micro-batch rows is rejected
unless expressed through an explicit temporal or event window.

## Core and Project Additions

The intended native additions are:

- `RollingSpec` and `RollingOperator`;
- `CrossSectionSpec` and `CrossSectionOperator`;
- new project-v3 operator variants for those native stateful operations;
- lifecycle-aware operator/provider capabilities;
- an internal stateless stream-provider registration seam; and
- stream-plan and runner support for immutable static inputs.

Project schema, OpenAPI, and generated TypeScript artifacts move in the same
commit as any serialized operator/configuration change. Older project-v3
documents remain valid. Older runtimes may reject the new operator kinds,
which is reported through capability/project validation rather than a Python
compatibility shim.

## Runtime Capability Model

`RuntimeCapabilities` must describe enough information for lowering decisions.
The next schema includes, at minimum:

```python
OperatorCapability(
    kind="rolling",
    version="1",
    input_ports=(ProviderPort("input", "table", required=True),),
    output_ports=(ProviderPort("output", "table", required=True),),
    modes=("batch", "stream"),
    finality="per_row_final",
    requires_datafusion=False,
    stateful=True,
    microbatch_invariant=True,
    requires_watermark=True,
    checkpoint_support="checkpointed_stateful",
    state_version=1,
    deterministic=True,
    replay_safe=True,
)
```

Provider capabilities carry the same lifecycle and partition contract. A
batch-only NumPy/JAX registration cannot be selected by `compile_stream()`.
An existing provider with no finality proof reports `finality="unproven"`,
remains usable in batch mode, and is rejected by stream compilation before its
factory or callback runs.
Capability output stays immutable, defensively copied, session-scoped, and
deterministically ordered.

## Performance Contract

The implementation is accepted only if it preserves these properties:

- no symbolic Python call occurs while a batch or stream job executes;
- row-local outputs are fused into the minimum practical DataFusion
  projections;
- compatible rolling outputs share one history and partition index;
- compatible cross-section outputs share grouping and sorting;
- compatible array expressions share one provider call per micro-batch;
- no per-security Python state object exists;
- Arrow buffers remain immutable and shared unless a physical layout change
  requires materialization;
- table-to-dense and host-to-device copies are explicit in metrics/explain;
- state memory is bounded by active groups and declared windows; and
- checkpoint size and time scale linearly with active state.

Paired benchmarks compare generated plans with equivalent hand-built
calc-flow plans on the same process, inputs, order, and machine. The initial
row-local regression gate is five percent. Stateful and matrix gates are set
from the approved baseline before their implementation begins rather than
inventing absolute throughput targets.

## Errors, Security, and Serialization

Errors preserve a stable path from program output through the failing
primitive, for example:

```text
outputs.signals.score.matmul.right.shape[0]
outputs.alpha.zscore.event_time
```

Symbolic normalization accepts only strict JSON attributes. It rejects
callables, modules, classes, file handles, arbitrary objects, duplicate output
names, unknown primitive versions, implicit backend conversion, unresolved
types, unsafe volatility, and unsupported lifecycle selection.

Python scalar UDFs remain trusted runtime registrations selected by exact
`UdfReference`. Streaming plans that require deterministic replay reject
volatile UDF paths unless a separately approved runtime contract proves their
semantics.

No formula parser uses Python `eval`. If a later data-only formula document is
approved, it remains a declaration format lowered to `ProjectDocument`; it
does not become a second executable project contract.

## Testing Strategy

Every behavior change starts with a focused failing test. Tests are placed in
the owning surface:

- symbolic Python behavior under `python/tests/`;
- PyO3 behavior as inline Rust tests and Python public tests;
- core operator unit tests beside their Rust source;
- graph/runtime integration under `crates/calc-flow/tests/`; and
- Studio backend/frontend tests in their existing packages.

The central equivalence matrix is:

```text
same bounded data
  x batch or final-only stream
  x several micro-batch segmentations
  x uninterrupted or checkpoint/recovery execution
  x ordered or bounded out-of-order arrival
  x null and NaN cases
```

Property tests vary segmentation, watermark placement, entity interleaving,
duplicate event times, checkpoint epochs, and supported primitive
compositions. Floating-point comparisons use the semantic tolerances frozen
per primitive; row order, null placement, output names, schemas, and metrics
remain exact.

## Initial Release Scope

The first supported catalog is deliberately compositional:

- row: arithmetic, comparison, boolean, `where`, coalesce, log, exp, sqrt,
  abs, clip, and cast;
- temporal: lag, delta, count, sum, mean, min, max, variance, standard
  deviation, covariance, and correlation;
- cross section: rank, percentile, z-score, demean, and winsorization;
- matrix: explicit column projection, elementwise expressions, static-weight
  matrix multiplication, and explicit attachment to the originating table;
  and
- batch/stream compilation with watermarks, late-event policy, checkpoints,
  recovery, CSE, projection fusion, and state sharing.

RSI, MACD, Bollinger bands, momentum, volatility, and comparable factor
families are expressed as compositions. A new primitive is justified only by
an unavailable algorithm, materially better native state structure, or a
measured fusion/performance requirement.

## Deferred Work

Cross-section top/bottom selection and mean fill are not part of the frozen
initial primitive catalog. They require a separate approved API and semantic
design after the initial release; implementations must not infer signatures or
lower them through another primitive.

## Frozen Decision Gates

SCE-00 closed every decision that can affect a public API, serialized data,
durable state, finality, or recovery:

1. the API contract sections 2 and 3 freeze exact Python and Rust names,
   signatures, and exports;
2. semantic contract D5 freezes row frames as inclusive `[i-n+1, i]`, duration
   frames as `(t-d, t]`, positive minimum counts, and deterministic order;
3. D6 freezes exact-time/fixed-bucket completeness, ties, null/NaN treatment,
   statistics, and canonical row order;
4. D7 freezes append-only final output, `error`/`drop`, envelope atomicity,
   redacted row-index diagnostics, and checked `max(W - t)` late metrics;
5. API contract section 5 freezes strict additive `rolling` and
   `cross_section` project-v3 variants with explicit defaults;
6. D9 and API contract section 6 freeze capability schema v2, including
   conservative legacy-provider `finality="unproven"`;
7. D10–D11 and API contract sections 7–8 freeze static-input ownership,
   canonical digest bytes, existing-lineage comparison, and recovery; and
8. D13 freezes per-primitive floating-point tolerances while keeping schemas,
   ordering, missing-value positions, and non-floating values exact.

The adversarial critique approved these choices after closing four blocking
findings in the permitted correction round. This gate authorizes only the
downstream tasks and does not claim that any symbolic engine surface or native
operator is implemented today.
