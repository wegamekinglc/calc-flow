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
`connect(source, target, source_port="output", target_port="input")`.

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
`CapabilityRule`, `ProviderArrayRules`, `RuntimeSessionScope`, and
`RuntimeCapabilities`. Provider option schema
version 1 supports only named scalar string, integer, number, or boolean
fields. `options_schema=None` means no declarative editor is available; it
does not mean every option is valid. The provider callback remains
authoritative during compilation.

The session ID is stable for one runtime. A successful registry entry advances
the revision exactly once; rejected duplicates do not. NumPy/JAX helpers add
`expression@1` and mapped `table_matmul@1` as separate entries, so one helper
normally advances by two and can expose a real partial success if its second
entry already exists. Previously returned snapshots remain isolated from
later revisions.

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
batch named `output`.

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
`PipelineBuilder.compile_stream()` is the graph-only path for
application-owned `SourceBinding` and `SinkBinding` values. Project JSON never
embeds a connector object or credential value.

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
installed 3.0 wheel. See the [cross-language inventory](examples.md) or run
all user examples with
`JAX_PLATFORMS=cpu uv run python scripts/run_examples.py`.
