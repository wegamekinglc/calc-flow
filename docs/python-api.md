# Python API

The `calc-flow==4.0.0` Python package is a PyO3 binding to the Rust engine plus
small functional adapters. Python 3.13 or newer is required.

## Install and develop

```bash
uv add calc-flow
uv add "calc-flow[numpy]"  # optional
uv add "calc-flow[jax]"    # optional
```

From a source checkout:

```bash
uv sync --extra dev
uv run maturin develop
JAX_PLATFORMS=cpu uv run pytest python/tests -q
```

## Table batches and builder

```python
import pyarrow as pa

from calc_flow import Batch, PipelineBuilder

batch = Batch.from_pyarrow(pa.table({"a": [1, 3], "b": [2, 4]}))
builder = PipelineBuilder("totals")
configured = builder.expression("calculate", "total = a + b")
plan = configured.compile_batch()
result = plan.execute({"input": batch})

assert builder.project["graph"]["nodes"] == []
assert result.outputs["output"].to_pyarrow()["total"].to_pylist() == [3, 7]
```

Builder methods return new values. `Batch.from_pyarrow` and every runner/plan
boundary treat caller inputs as read-only. Result mappings and metadata are
defensive copies.

An expression node accepts exactly one calculation expression or a non-empty
`select` projection; `filter` may accompany either mode. Connect nodes with
`connect(source_node, target_node, source_port="output", target_port="input")`.

## Multi-input SQL

```python
plan = (
    PipelineBuilder("orders-and-fees")
    .sql(
        "join",
        "SELECT orders.order_id, orders.amount - fees.fee AS net "
        "FROM orders JOIN fees ON orders.order_id = fees.order_id "
        "ORDER BY orders.order_id",
        aliases=("orders", "fees"),
    )
    .compile_batch()
)
result = plan.execute(
    {
        "orders": Batch.from_pyarrow(
            pa.table({"order_id": [1, 2, 3], "amount": [75, 120, 40]})
        ),
        "fees": Batch.from_pyarrow(
            pa.table({"order_id": [1, 2, 3], "fee": [5, 12, 4]})
        ),
    }
)
assert result.outputs["output"].to_pyarrow()["net"].to_pylist() == [70, 108, 36]
```

Only one read-only DataFusion `SELECT` or CTE is accepted. The full version is
[`examples/02_sql_join.py`](../examples/02_sql_join.py), which mirrors the Rust
`sql_join.rs` example.

## Trusted Python scalar UDFs

```python
import pyarrow as pa
import pyarrow.compute as pc

from calc_flow import PipelineBuilder, Runtime

runtime = Runtime()


def double_amount(amount: pa.Array) -> pa.Array:
    return pc.multiply(amount, 2)


runtime.register_scalar_udf(
    provider="python",
    name="double_amount",
    version="1",
    input_types=("int64",),
    return_type="int64",
    volatility="immutable",
    function=double_amount,
)
plan = (
    PipelineBuilder("registered-udf")
    .expression(
        "calculate",
        "total = double_amount(amount)",
        udfs=(("python", "double_amount", "1"),),
    )
    .compile_batch(runtime)
)
```

Callbacks are trusted application code, vectorized over PyArrow arrays, and
never serialized. Registration and execution enforce exact Arrow types,
result length/type, and explicit node references. The full version is
[`examples/03_registered_udf.py`](../examples/03_registered_udf.py).

## Runtime capabilities

`Runtime.capabilities()` returns a frozen `RuntimeCapabilities` value. It is a
data-only snapshot: callbacks and source/import paths are neither returned nor
inspected.

```python
from calc_flow import (
    ProviderOption,
    ProviderOptionsSchema,
    Runtime,
)


def normalize_callback(batch, options):
    return batch


runtime = Runtime()
runtime.register_provider(
    "acme",
    "normalize",
    "1",
    normalize_callback,
    options_schema=ProviderOptionsSchema(
        fields=(ProviderOption("scale", "number", required=True),)
    ),
)
snapshot = runtime.capabilities()

assert snapshot.schema_version == 2
assert snapshot.scope.kind == "runtime_session"
assert snapshot.scope.revision == 1
assert snapshot.providers[0].name == "normalize"
```

The public frozen values are `ProviderOption`, `ProviderOptionsSchema`,
`ProviderPort`, `ProviderCapability`, `UdfCapability`, `OperatorCapability`,
`ConnectorCapabilities`, `ConnectorCapability`, `CapabilityRule`,
`ProviderArrayRules`, `RuntimeSessionScope`, and
`RuntimeCapabilities`. Provider option schema
version 1 supports only named scalar string, integer, number, or boolean
fields. `options_schema=None` means no declarative editor is available; it
does not mean every option is valid. The provider callback remains
authoritative during compilation.

Capability schema version 2 makes every operator and provider entry
lifecycle-aware. `OperatorCapability` and `ProviderCapability` report
`modes`, `finality`, `stateful`, `microbatch_invariant`, `requires_watermark`,
`checkpoint_support`, `state_version`, `deterministic`, and `replay_safe`;
providers additionally report `supports_static_inputs`, `partition_contract`,
and optional `array_rules`. The vocabularies are closed: execution modes are
`batch` and `stream`; output finality is `per_row_final`,
`group_final_append_only`, or `unproven`; checkpoint support is `stateless`,
`checkpointed_stateful`, or `unproven`; and a provider partition contract is
`none` or `row_axis_independent`. `state_version` is a positive integer
exactly when `checkpoint_support` is `checkpointed_stateful` and `None`
otherwise, and a `stateless` capability must set `stateful=False`.
Construction validates strictly and fails closed: closed-vocabulary and
cross-field violations raise `ValueError`, while non-strict data (a `list`
where a tuple is declared, a non-`bool` boolean field) raises `TypeError`.

`CapabilityRule` identities are closed and versioned. The accepted identities
are `array_api_safe_dtype@1`, `elementwise_broadcast@1`,
`feature_axis_reduction@1`, and `table_matmul_static_rhs@1`; any other
name/version pair fails construction. `ProviderArrayRules` pairs the exact
`supported_dtypes` tuple with a `safe_dtype_rule` and `shape_rules`, and
stores both tuples sorted by identity.

The `operators` tuple contains exactly `cross_section@1`, `expression@1`,
`rolling@1`, `sql@1`, and `stream_join@1`, with truths anchored in the
engine implementation:

| Operator          | Modes         | Finality                | Checkpoint support    | State version |
| ----------------- | ------------- | ----------------------- | --------------------- | ------------- |
| `cross_section@1` | batch, stream | group_final_append_only | checkpointed_stateful | 1             |
| `expression@1`    | batch, stream | per_row_final           | stateless             | —             |
| `rolling@1`       | batch, stream | per_row_final           | checkpointed_stateful | 1             |
| `sql@1`           | batch, stream | unproven                | stateless             | —             |
| `stream_join@1`   | stream        | unproven                | checkpointed_stateful | 1             |

`cross_section@1`, `rolling@1`, and `stream_join@1` are the stateful
operators and the only ones that require a watermark; `cross_section@1`,
`expression@1`, and `rolling@1` are micro-batch invariant. All five report
`deterministic=True` and `replay_safe=True`. For `sql@1` those two claims
hold from the engine viewpoint: exactly-once stream
plans reject nodes that select volatile registered UDFs, and stream
compilation rejects read-only queries that call volatile built-in SQL
functions such as `random()` or wall-clock built-ins such as `now()`,
`current_date()`, and `current_time()` (aliases included).

Providers registered through `register_provider` keep their existing
signature and source compatibility. The registration API accepts no lifecycle
metadata, so a registered provider's entry is always batch-only with
conservative values: `modes=("batch",)`, `finality="unproven"`,
`stateful=False`, `microbatch_invariant=False`, `requires_watermark=False`,
`checkpoint_support="stateless"`, `state_version=None`,
`deterministic=False`, `replay_safe=False`, `supports_static_inputs=False`,
`partition_contract="none"`, and `array_rules=None`. A registration record
carrying forged lifecycle keys is ignored rather than upgraded, and omission
never opts a provider into stream execution. `finality="unproven"` means
registration evidence establishes no output-finality contract; it is the
truthful conservative value for a batch-only registration and does not narrow
existing batch selectability.

The trusted `register_numpy` and `register_jax` helpers additionally attach a
process-local stateless stream proof to their `expression@1` registrations.
Those provider entries report `modes=("batch", "stream")`,
`finality="per_row_final"`, `microbatch_invariant=True`,
`deterministic=True`, `replay_safe=True`, and the
`row_axis_independent`/`elementwise_broadcast@1` array contract. The public
`Runtime.register_provider` signature has no lifecycle arguments, so an
arbitrary Python callback remains batch-only rather than acquiring this proof
from caller-supplied metadata.

Compiled-in connector registrations surface on the snapshot's `connectors`
tuple as `ConnectorCapability` entries. Each entry pairs its
`(provider, name, version)` identity and `source`/`sink`/`both` kind with a
`ConnectorCapabilities` value (`delivery`, `replay`, `watermark`,
`transaction`, and the `snapshot`/`polling`/`cdc`/`lookup` flags), the
declared formats, and an options schema.

The session ID is stable for one runtime. A successful registry entry advances
the revision exactly once; rejected duplicates do not. NumPy/JAX helpers add
`expression@1` and mapped `table_matmul@1` as separate entries, so one helper
normally advances by two and can expose a real partial success if its second
entry already exists. Previously returned snapshots remain isolated from
later revisions. Snapshots are immutable and defensively copied: mutating a
caller-owned sequence after registration cannot change a returned snapshot.
Operators sort by `(kind, version)`; providers and UDFs sort by
`(provider, name, version)`.

## Execution options and provider context

Use the frozen native `ExecutionOptions` value to attach run-scoped settings
and an absolute UTC deadline:

```python
from datetime import UTC, datetime, timedelta

from calc_flow import ExecutionOptions

options = ExecutionOptions(
    settings={"request": {"tenant": "demo", "attempt": 1}},
    deadline=datetime.now(UTC) + timedelta(seconds=30),
)
result = plan.execute({"input": batch}, options=options)
```

Every object position in `settings` may use a
`collections.abc.Mapping`. Calc Flow calls each mapping's `items()` once,
consumes that iterator once, and copies it to a built-in `dict`; it does not
consult `len()`, `keys()`, or `__getitem__()`. Mapping subclasses are accepted,
while sequence containers must be exact built-in `list` values. Object keys
and leaves must be exact built-in `str`, `None`, `bool`, `int`, or finite
`float` values. Integers must be in `-2**63 .. 2**64 - 1`. Coercion-only
objects, unsupported subclasses, tuples, sets, bytes, duplicate keys,
surrogate code points, excessive depth, and cycles are rejected. Validation
errors expose only stable settings paths and fixed messages; exceptions from
caller mappings are not retained or chained.

Construction deep-copies the complete accepted graph; mutating the source or
any nested caller container cannot change the options. Every
`options.settings` read returns another deep `dict`/`list` copy, so mutating an
observation cannot change a later read or execution. Omitting `settings`
creates an empty mapping, and passing `None` explicitly has the same meaning.
`deadline` accepts `None` or any valid timezone-aware `datetime`. Calc Flow
normalizes every accepted offset to `datetime.UTC` and preserves
microseconds; it rejects naive, invalid, and out-of-range UTC conversions with
fixed redacted errors.

The `ExecutionOptions(settings, deadline)` constructor accepts positional or
keyword arguments. In contrast, `options` is keyword-only in both
`plan.execute(inputs, *, options=None)` and
`plan.execute_async(inputs, *, options=None)`. Omitting the plan option
preserves the existing default behavior.

Provider callbacks remain two-argument callables unless the registration
explicitly opts into run context:

```python
def contextual_provider(batch, provider_options, context):
    tenant = (context.settings or {}).get("request", {}).get("tenant")
    return batch


runtime.register_provider(
    "acme",
    "contextual",
    "1",
    contextual_provider,
    accepts_context=True,
)
```

A single provider callback receives one `Batch`: `(batch, provider_options)`
when `accepts_context=False`, the default, or
`(batch, provider_options, context)` when it is true. A mapping provider
registered with `_register_mapping_provider` receives its named input mapping
instead: `(inputs, provider_options)` when false or
`(inputs, provider_options, context)` when true. Existing two-argument
providers therefore remain source-compatible.

Each callback is invoked exactly once under the selected ABI. The frozen,
engine-created `ProviderContext` exposes the authoritative run
`context.settings` and `context.deadline`, not values merged into the separate
compile-time `provider_options` mapping. Every settings read returns a fresh
deep copy, and the deadline is the normalized aware UTC value or `None`. The
flag must be an exact `bool`; Calc Flow does not infer arity or retry a
callback after `TypeError`. Native cancellation tokens are intentionally not
part of the public Python API.

The feature is additive: existing `execute(inputs)`, `execute_async(inputs)`,
and two-argument providers retain their behavior. Run settings, deadlines,
and provider-context opt-in are not serialized into projects, checkpoints, or
Studio API payloads and do not change those formats.

`ExecutionOptions.deadline` is an absolute cooperative engine deadline.
Studio's `RunOptions.timeout_seconds` is instead a process-level preview limit; it
does not populate execution settings or a deadline in the worker.

## Async execution

```python
from datetime import UTC, datetime, timedelta

from calc_flow import ExecutionOptions


async def calculate() -> list[int]:
    plan = (
        PipelineBuilder("async-example")
        .expression("calc", "total = a + b")
        .compile_batch()
    )
    options = ExecutionOptions(
        settings={"request": {"source": "async-example"}},
        deadline=datetime.now(UTC) + timedelta(seconds=30),
    )
    result = await plan.execute_async(
        {"input": Batch.from_pyarrow(pa.table({"a": [1, 3], "b": [2, 4]}))},
        options=options,
    )
    return result.outputs["output"].to_pyarrow()["total"].to_pylist()
```

Blocking `execute`, store, and runner methods reject a running event loop. Use
their async forms in servers and asyncio applications. `plan.execute()` checks
for a running event loop before it validates inputs or options, so that usage
error has precedence. An already-expired or crossed execution deadline raises
`calc_flow.CancelledError` after transactional rollback. Cancelling a
still-pending surrounding asyncio task instead raises
`asyncio.CancelledError`. The handler makes one terminal-state decision: if
native execution is already terminal, its result or exception remains
observable; otherwise it sends exactly one native cancellation request and
waits through repeated Python task cancellation until cleanup finishes. Once
the cancellation request is sent, the caller's `asyncio.CancelledError` wins
over any native outcome observed during that drain: a native failure landing
mid-drain is retrieved and discarded, never re-raised to the caller.

Awaiting task cancellation waits until the current native operation and
run-owned cleanup finish; no work or input payload continues detached. The
plan recovers its pre-run state before its next public operation. Deadline and
task cancellation are cooperative at safe boundaries, so neither preempts a
Python callback, DataFusion query, or other non-cooperative operation already
in progress; cleanup resumes when that operation yields. The same
`ExecutionOptions` value can be reused concurrently because each run receives
independent native cancellation state. No cancellation token is part of the
public Python API.

For executions queued behind another invocation of the same plan, the
absolute deadline keeps elapsing. Cancelling a queued invocation neither
cancels the active run nor creates partial plan state. Once a deadline or
accepted task cancellation is observed at a post-provider boundary, it wins
over that provider's error. Recovery, input, snapshot, and transaction-marker
failures that occur before the first deadline check retain their existing
precedence.

The full version is [`examples/05_async_execution.py`](../examples/05_async_execution.py).

## NumPy and JAX

```python
import numpy as np

from calc_flow import Batch, PipelineBuilder, Runtime, register_numpy

runtime = Runtime()
register_numpy(runtime)
plan = (
    PipelineBuilder("numpy-array")
    .external(
        "center",
        "numpy",
        "expression",
        "1",
        {"expression": "x - mean(x)"},
    )
    .compile_batch(runtime)
)
batch = Batch.from_array(np.array([1.0, 2.0, 4.0, 6.0]), backend="numpy")
centered = plan.execute({"input": batch}).outputs["output"].array
```

Owned arrays are read-only. The bounded expression evaluator allows arithmetic,
reductions, transpose, and reshape; it rejects Python execution features and
backend changes. Operation results, including broadcast binary operations, are
capped at 10,000,000 elements so a single expression cannot allocate an
unbounded output. The input batch itself is exempt, so reductions over larger
inputs remain valid. `register_jax` provides the same explicit provider
boundary.
The full version is [`examples/06_numpy_array.py`](../examples/06_numpy_array.py).

Both helpers also make `expression@1` available to stream compilation, but
only for a conservative row-axis-independent subset. The parsed expression
must reference `x` and may contain no function call or matrix multiplication;
for example, `x * 2 + 1` is eligible, while `sum(x)`, `transpose(x)`,
`reshape(x, ...)`, `x @ x`, and a constant-only expression fail compilation.
The broader bounded expression language remains available to batch plans.

For this stateless stream path, the provider callback receives
`(batch, provider_options)` exactly once for each accepted data micro-batch.
It runs on a blocking worker and receives no public cancellation token or
`ProviderContext`. A callback exception surfaces as a provider error and emits
nothing. Cancellation is checked before dispatch and again after a successful
callback, before output validation and emission. The running
`spawn_blocking` callback itself cannot be preempted, so cancellation waits for
it to return; once the post-callback check observes cancellation, the result is
discarded and no output is emitted.

### Table-array matrix multiplication

`pyarrow.Table` is the table input for table-array matrix multiplication. The
immutable builder method is:

```python
def table_matmul(
    self,
    node_id: str,
    *,
    backend: Literal["numpy", "jax"],
    columns: Sequence[str],
) -> PipelineBuilder: ...
```

It selects `columns` in the supplied order, accepts a rank-two `weights` array
with shape `(len(columns), output_width)`, and returns a same-backend array
batch named `output`. `table_matmul@1` remains batch-only in this release;
static-input matrix multiplication is follow-up work rather than part of the
stateless stream-provider proof.

```python
import numpy as np
import pyarrow as pa

from calc_flow import Batch, PipelineBuilder, Runtime, register_numpy

runtime = Runtime()
register_numpy(runtime)
plan = (
    PipelineBuilder("table-matrix")
    .table_matmul("multiply", backend="numpy", columns=("quantity", "unit_price"))
    .compile_batch(runtime)
)
result = (
    plan.execute(
        {
            "table": Batch.from_pyarrow(
                pa.table({"quantity": [3.0, 1.0], "unit_price": [10.0, 12.0]})
            ),
            "weights": Batch.from_array(
                np.array([[2.0, 0.0], [0.0, 1.0]]), backend="numpy"
            ),
        }
    )
    .outputs["output"]
    .array
)

assert result.tolist() == [[6.0, 10.0], [2.0, 12.0]]
```

After input `Batch` construction, the operator makes no redundant execution
copies. NumPy allocates one dense table matrix and one result. JAX permits one
host staging buffer, one device table buffer, and one device result; it does
not promise a host-free transfer path. The construction of caller input
`Batch` values is outside these execution ceilings. JAX performs no result-to-host round trip during operator execution. The runnable NumPy and
optional JAX paths are in
[`examples/07_array_and_dataframe.py`](../examples/07_array_and_dataframe.py).

## Symbolic declarations and static analysis

`calc_flow.symbolic` is the pure declaration surface: typed immutable
expressions, feature sets, and programs with canonical identities plus static
analysis over the declaration graph. It has no data execution path — there is
no `eval`, `push`, `value`, `transform`, preview evaluator, or formula parser —
and execution stays owned by the existing execution plans and runners.

```python
from calc_flow import Runtime
from calc_flow.symbolic import FeatureSet, Field, Program, table_input

quotes = table_input(
    "quotes",
    schema=[
        Field("ts", "timestamp[us, UTC]", nullable=False),
        Field("x", "float64"),
        Field("y", "float64"),
    ],
)
signals = quotes.with_columns(FeatureSet([("score", quotes["x"] + quotes["y"])]))
program = Program("p", inputs=[quotes], outputs=[("signals", signals)])

result = program.analyze(Runtime(), mode="batch")
assert result.issues == ()
```

`table_input` declares one named table input with an exact ordered schema — a
sequence of `Field` values, never a mapping; `event_time`, `entity_by`, and
`sequence_by` declare the ordering facts of a temporal input. Selecting
`quotes["x"]` builds a `ColumnExpr`; arithmetic, comparison, and boolean
composition build immutable expression nodes whose canonical v1 digests are
stable across processes. Public comparisons build symbolic expressions:
converting an expression to `bool` fails, and `identical()` is the structural
identity check.

A `FeatureSet` is an ordered immutable set of uniquely named column
expressions; `with_feature` appends one. `TableExpr.with_columns(features)`
returns a new table with the declared features appended as derived columns.
The `row`, `ts`, `cs`, `table`, `linalg`, and `window` namespaces expose
row-local functions, rolling frames (`rows`/`duration`), cross-section groups
(`exact_time`/`event_time_bucket`), table bridges, and matrix work.

A `Program` declares uniquely named inputs (`table_input` or `parameter`
values) and outputs (tables or arrays) in declaration order. Its `fingerprint`
is the runtime-independent `calc_flow.symbolic.declaration.v1` program
fingerprint over every unique node reachable from a declared input or output;
it does not depend on construction history and is stable across conforming
implementations. Duplicate declared names fail at construction with stable
paths such as `inputs.quotes: duplicate_name`. An input referenced by an
output but missing from `inputs` is reported during analysis as an issue
rooted at `inputs.<name>`.

`Program.analyze(runtime, mode=...)` and `Program.explain(runtime, mode=...)`
require an explicit `Runtime` and a `batch` or `stream` mode; both consume one
immutable capability snapshot and record its session and revision. From the
declaration graph alone — no data object, source, sink, or runner is accepted —
the analysis proves:

- value types, proving only what the capability snapshot proves: identical
  operand types, the frozen rolling/cross-section output-type table, and exact
  field resolution; cross-type arithmetic needs an explicit `row.cast`;
- domains and row lineages, rejecting cross-input lineage mixing;
- symbolic dimensions through `linalg.from_columns` and `linalg.matmul`,
  retaining the row axis of a table-derived array;
- attachment compatibility for `table.attach_columns`;
- state requirements per output, rendered by `explain` as
  `state cross_section, duration(60000000)`-style facts; and
- stream safety: temporal and cross-section inputs need an event-time column
  with entity and sequence keys, and a stream-mode array output with row-axis
  lineage is reported as unbounded state.

`analyze` returns an immutable `AnalysisResult` carrying `mode`,
`program_fingerprint`, `capability_session_id`, `capability_revision`, and an
`issues` tuple; each finding is an immutable `AnalysisIssue` with a stable
`path`, `code`, and `message`. Paths start at a named program output or input —
for example `outputs.signals.score`, `outputs.scores.matmul.right.shape[0]`,
`inputs.quotes.sequence_by[0]`, and `static_inputs.weights`. Analysis is
deterministic: it never mutates a declaration node, and repeated runs return
equal results. The analysis issue codes are `capability_mismatch`,
`duplicate_name`, `ordering_required`, `schema_mismatch`, `unbounded_state`,
`unresolved_type`, and `unsupported_type`; construction errors raise
`ValueError` or `TypeError` with the same path grammar. `explain` renders the
same facts as a deterministic multi-line report.

### Row-local compilation

`Program.compile_batch(runtime)` and `Program.compile_stream(runtime, *,
allowed_lateness_micros=0, late_policy="error")` lower a program to the
existing execution plans. Compilation is declaration processing only: it
captures one immutable capability snapshot, lowers one strict project-v3
document, and invokes the Rust graph compiler for final port, kind, schema,
topology, and fingerprint validation. No data object, source, sink, or runner
is accepted, and no symbolic Python runs while a compiled plan executes.

Row-local declarations — literals, fields, arithmetic, comparison, boolean
composition, `where`, `coalesce`, `log`/`exp`/`sqrt`/`abs`/`clip`/`cast`, and
the `table.project`/`table.filter`/`with_columns` table operations — fuse into
one `expression` node per program output; a `where`/filter predicate becomes
the node's `WHERE` clause. Structurally identical non-trivial subexpressions
referenced at least twice are computed exactly once: they materialize as
`__cf_cse_N` columns in deterministic tier nodes (`<output>__cf_cse_<k>`)
ahead of the fused node, so a 20-output `FeatureSet` with no shared
subexpressions compiles to a single fused node. Node IDs and the plan
fingerprint are deterministic, and the lowered project carries strict JSON
only.

Programs with one input and one output bind the plan endpoints `input` and
`output`, matching the `PipelineBuilder` convention; multi-branch graphs name
endpoints `<node>.input` and `<node>.output` deterministically. Batches
supplied at execution must match the declared input schema exactly.

`ts.lag`, `ts.delta`, and the rolling aggregates `ts.count`, `ts.sum`,
`ts.mean`, `ts.min`, `ts.max`, `ts.variance`, `ts.stddev`,
`ts.covariance`, and `ts.correlation` lower to one native `rolling` node
per program output, placed ahead of the fused row-local stages. Rolling
requires the input table to declare its `entity_by`, `event_time`, and
`sequence_by` ordering keys; a program missing them fails with
`ordering_required`. In this release every rolling argument must be a plain
input column: a computed operand such as `ts.lag(quotes["x"] + 1.0)`, either
operand of a pair aggregate, or a rolling aggregate over a derived column,
fails with `unsupported_type` rooted at the output feature. `periods` is a
positive integer defaulting to `1`; an aggregate declares `rows(size)` or
`duration(micros)` as its window with `min_periods` (default `1`, capped at
the frame size for `rows` windows only) and — for `variance`, `stddev`,
`covariance`, and `correlation` — `ddof` of `0` or `1` (default `1`), and
construction rejects other values with `ValueError`. Every rolling
occurrence in one output shares that output's rolling node: a whole-feature
occurrence keeps its feature name as the output column, while an occurrence
nested inside a larger expression materializes as a deterministically named
`<output>__cf_roll_<index>` column that the fused stages then reference. A
lag, delta, `min`, or `max` column keeps the input column's type; the
remaining aggregate columns take the frozen output type — `uint64` for
`count`, `int64` or `uint64` for an integer `sum`, `float64` otherwise —
and the engine evaluates the frozen window semantics described in the
[Rust API guide](rust-api.md). A
filter declared below every rolling feature becomes a deterministic
`<output>__cf_prefilter` expression node feeding the rolling node; a filter
declared above them applies after it. The lowered node's frozen spec carries
`configuration_version` and `state_layout_version` 1, the declared ordering
keys, one entry per rolling output (`kind`, `primitive_version` 1, `input`,
`output`, and `periods`, or `frame`, `min_periods`, and — for the
statistical kinds — `ddof`; the pair kinds carry `left` and `right` in place
of `input`), the `allowed_lateness_micros` and `late_policy`
values validated by `compile_stream` — `error` lowers to an envelope-scoped
rejection and `drop` to a metrics-recorded drop — and the
`stateful_numeric_v1` value policy, which preserves a null or NaN current or
referenced value. Batch lowering writes the default lateness values, batch
evaluation classifies no late rows, and a program without rolling
declarations is unaffected by the lateness arguments.

`cs.rank`, `cs.percentile`, `cs.demean`, and `cs.zscore` lower to one native
`cross_section` node per program output, placed ahead of the fused row-local
stages, with the same `ordering_required` key requirement as rolling and —
in this release — the same plain-input-column rule for the measured value,
the grouping event time, and every group column. A group is declared with
`exact_time(event_time, *, partition_by=())` or
`event_time_bucket(event_time, *, width_micros, partition_by=())`, and every
cross-section occurrence in one output must share one grouping declaration:
the same partition columns and the same grouping shape — exact time, or one
bucket width — or compilation fails with `schema_mismatch` rooted at the
output. `rank` and `percentile` carry `direction` (default `ascending`),
`tie_method` (default `average`), and `null_placement` (default `exclude`);
every primitive carries `min_samples` (default `1`), and `zscore` adds `ddof`
of `0` or `1` (default `0`), with construction rejecting other values. As
with rolling, a whole-feature occurrence keeps its feature name as the
output column while an occurrence nested inside a larger expression
materializes as a deterministically named `<output>__cf_cs_<index>` column,
and a filter below every cross-section feature becomes the shared
`<output>__cf_prefilter` node. Every cross-section column is nullable
`float64`, and the engine evaluates the frozen complete-group semantics
described in the [Rust API guide](rust-api.md). The lowered node's frozen
spec carries `configuration_version` and `state_layout_version` 1, the
declared ordering and partition keys, the shared grouping, one entry per
output, the `allowed_lateness_micros` and `late_policy` values validated by
`compile_stream`, and the `nan_exclude_preserve_v1` value policy.

Analysis rejections surface as `CompileError` with the first issue's
`{path}: {code}: {message}`. Declarations the lowerer does not implement —
event `window` nodes, `linalg` array bridges, and `parameter` static inputs
— fail with the eighth issue code `unknown_primitive_version` rooted at the
output or `static_inputs.<name>`, in both batch and stream modes; a stream
aggregate or SQL window is never silently made batch-local. A
`cs.winsorize` declaration — the one cross-section primitive withheld in
this release — fails with `unsupported_type` rooted at the output feature.
Array outputs
fail with `unknown_primitive_version` in batch mode; stream mode rejects
them earlier, at the analysis phase, with `unbounded_state` rooted at
`outputs.<name>` — the stream-safety rule for an array output with row-axis
lineage described above. Casts to non-portable targets fail with
`unsupported_type` at `outputs.<name>.cast.data_type`.

The aggregate acceptance tests compare the engine against an independent
reference model whose finite-path mean is a naive sum. Its fixtures must
stay in magnitudes where a naive sum cannot overflow — the committed
fixtures are small finite magnitudes — because a 1e308-scale finite fixture
would overflow the oracle's sum while the engine's incremental mean stays
correct; extending the fixtures that far requires a West-based oracle
first.

Stream plans also reject read-only queries that call volatile built-in
functions (for example `random()`) or the wall-clock built-ins `now`,
`current_date`, and `current_time` (aliases such as `current_timestamp`
and `today` included): `compile_stream` resolves every function in an
expression or SQL node's query against the built-in default function
registry, matching the resolved canonical name, and fails volatile and
wall-clock calls before any source opens, so the deterministic,
replay-safe lifecycle claims of those operators remain truthful.

## Projects and persistence

`ProjectDocument` validates a strict `format_version: 3` mapping with the Rust
schema. `project_json_schema()` returns the generated schema;
`validate_project_json(document)` returns canonical JSON. Invalid documents
raise a pydantic `ValidationError` whose entries carry the engine's stable
issue codes as `type` and the failing project path as `loc`; malformed stream
Join input reports codes such as `invalid_time_bound` and
`unsupported_join_type` instead of a flattened message.

Project v3 selects `runtime.mode` explicitly. Stream projects carry exact
connector and format identities, non-secret options, named secret references,
watermark policy, managed state settings, and per-output best-effort,
at-least-once, or exactly-once delivery requests.
`compile_stream_project(project, runtime=...)` resolves those references
through the runtime's connector registry and secret resolver. Its returned plan
owns deferred connector bindings and project runtime/state settings, so launch
uses `StreamingRunner(plan)` without separate Python connector objects.
Stream projects may additionally declare immutable static side inputs; see
[static input declarations](connectors.md#static-input-declarations) for the
syntax and validation rules. `PipelineBuilder.compile_stream()` is the
graph-only path for application-owned `SourceBinding` and `SinkBinding`
values. Project JSON never embeds a connector object, credential value, or
live static payload.

`FileProjectStore` has async `create`, `put`, `get`, `list`, and `delete`
methods and explicit `*_blocking` variants. Safe JSON/YAML import/export helpers
live in `calc_flow.store`. Continuous checkpoint documents are internal to
`ManagedCheckpointRuntime`; the package has no public checkpoint-document
store.

## Streaming runner

`PipelineBuilder.compile_stream()` returns a distinct `StreamExecutionPlan`.
The plan records immutable source/sink binding IDs and optional per-output
`StreamRequirements`; it cannot execute as a batch plan. A
`StreamingRunner` owns that plan, all connector bindings, one
`ManagedCheckpointRuntime`, and optional `StreamRuntimeConfig`:

```python
plan = PipelineBuilder("orders").expression("total", "total = a + b").compile_stream()
runner = StreamingRunner(
    plan,
    {"input": SourceBinding(source, watermark_policy=DisabledWatermarks())},
    {"output": [SinkBinding.ordinary("archive", sink)]},
    ManagedCheckpointRuntime(".calc-flow-continuous"),
)
job = await runner.start_async()
print(job.status())  # synchronous and safe inside the event loop
outcome = await job.wait_async()
```

The constructor accepts one further keyword-only argument,
`static_inputs: Mapping[str, Batch] | None = None`, supplying the immutable
per-job side values a plan declares. `None` normalizes to an empty mapping;
non-`str` keys or non-`Batch` values raise `TypeError` before native
construction, and the mapping is defensively copied. Project-backed plans
reject externally supplied `sources`, `sinks`, `checkpoints`, and `config`
but exempt `static_inputs`, which is required when the plan declares static
inputs. `plan.static_input_ids` returns the declared names as a sorted tuple
and `plan.source_binding_ids` excludes them. The values are validated,
latched, and digested exactly once inside `start_async`/`start`, before any
source or connector lifecycle runs; restarts that supply different values are
rejected against the recorded digest before sources open. The full per-job
semantics, digest contract, and recovery behavior are in
[static inputs](streaming-guide.md#static-inputs).

A bounded event-time Join is declared on the builder with
`stream_join(name, *, left_schema, right_schema, left_keys, right_keys,
left_event_time, right_event_time, bounds, limits, left_prefix="left",
right_prefix="right")`. Input schemas are `ArrowFieldSpec` sequences and the
`JoinTimeBounds`/`JoinStateLimits` values are required; the
[continuous streaming guide](streaming-guide.md) carries the full
declaration.

Source `open`, `next`, and `close` and every sink lifecycle method must be
declared with `async def`; binding construction rejects invalid method shapes
without invoking the connector. `start_async()` consumes its runner exactly
once. Jobs expose async checkpoint, shutdown, cancel, and wait operations plus
guarded blocking forms for callers outside an event loop. Cancelling a
`wait_async()` observer leaves the job running; explicit cancellation uses
`cancel_async()`.

Blocking `start()` creates a dedicated event-loop thread and keeps it for the
job's async connectors. Later blocking job operations run on that owning loop;
terminal async operations called from another event loop are marshalled back
to it. Blocking or async `shutdown`, `cancel`, and `wait` release connector
roots and stop and join the thread after the native terminal outcome. Dropping
the last job owner schedules cancellation and settlement on the owning loop,
then reclaims the thread. Cancelling an async terminal observer before native
termination leaves cleanup running to convergence, so observer cancellation
does not strand the loop thread. If the native terminal outcome has already
linearized and cancellation arrives during thread cleanup, that outcome wins
and cleanup still completes.

`Cursor` payloads, capability/config mappings, pre-commit values, recovery
values, status, and outcomes cross the boundary as defensive copies. Status
and outcomes are typed: `job.status()` returns a `JobStatus` mapping that
includes `stream_joins`, a per-node mapping of `StreamJoinStatus` values with
`StreamJoinSideStatus` per side (empty when the graph has no Join node).
Managed checkpoint recovery reopens a live replayable source with a cursor
bound to the exact source-map key. A terminal manifest instead returns
completed without reopening ended sources or duplicating final output. See
[`04_continuous_runtime.py`](../examples/04_continuous_runtime.py),
[`08_streaming_recovery.py`](../examples/08_streaming_recovery.py), and the
[continuous streaming guide](streaming-guide.md).

The A6 cutover has no compatibility aliases: `MicroBatchRunner`,
`LegacyStreamingRunner`, the batch-plan `StreamingRunner` overload, and
`FileCheckpointStore` are removed. Compile a `StreamExecutionPlan` and migrate
connectors to the async `StreamSource`/`StreamSink` protocols.

## Exceptions

Catch the narrowest exported class: `ConfigError`, `CompileError`,
`ExecutionError`, `ProviderError`, `CheckpointError`, or `CancelledError`.
All derive from `CalcFlowError`; provider/cancellation errors are execution
errors. Continuous lifecycle failures use payload-safe
`StreamingRuntimeError`; an indeterminate manifest publication uses its
`CheckpointPublicationUnknownError` subclass. These exceptions expose only
structured category, job/epoch/phase/component identifiers, diagnostic ID,
and deterministic position fields—never connector values, cursor payloads,
paths, callback representations, or raw source chains.

## More examples

Every file under [`examples/`](../examples/README.md) is executable against the
installed 4.0 wheel. See the [cross-language inventory](examples.md) or run
all user examples with
`JAX_PLATFORMS=cpu uv run python scripts/run_examples.py`.
