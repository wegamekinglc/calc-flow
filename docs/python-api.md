# Python API

[Documentation](README.md) / 3.2 Python API

For a guided first calculation, read [batch calculations](batch-guide.md).
This page describes methods and their contracts; the larger symbolic surface
has its own [reference](symbolic-api.md).

The `calc-flow==4.0.0` Python package is a PyO3 binding to the Rust engine plus
small functional adapters. Python 3.13 or newer is required.

On this page:

- [Install and develop](#install-and-develop)
- [Table batches and builder](#table-batches-and-builder)
- [Multi-input SQL](#multi-input-sql)
- [Trusted Python scalar UDFs](#trusted-python-scalar-udfs)
- [Runtime capabilities](#runtime-capabilities)
- [Execution options and provider context](#execution-options-and-provider-context)
- [Async execution](#async-execution)
- [NumPy and JAX](#numpy-and-jax)
- [Symbolic declarations and static analysis](#symbolic-declarations-and-static-analysis)
- [Projects and persistence](#projects-and-persistence)
- [Streaming runner](#streaming-runner)
- [Exceptions](#exceptions)
- [More examples](#more-examples)

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

Use the immutable builder method to select parallelism and diagnostic controls:

```python
builder = PipelineBuilder("parallel-sql").with_datafusion_config(
    parallelism_mode="auto",
    max_partitions=16,
    min_rows_per_partition=65_536,
    small_rows_threshold=10_001,
    enable_rolling_rewrite=True,
    collect_diagnostics=True,
)
```

Auto mode requires trusted `calc_flow.datafusion.active_entities` batch
metadata and otherwise uses p1 without scanning the table. Fixed p1 remains the
default. See [SQL and DataFusion performance controls](sql-datafusion-performance.md)
for all fields, telemetry, evidence gates, and rollback steps.

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

assert snapshot.schema_version == 3
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

Capability schema version 3 makes every operator and provider entry
lifecycle-aware. `OperatorCapability` and `ProviderCapability` report
`modes`, `finality`, `stateful`, `microbatch_invariant`, `requires_watermark`,
`checkpoint_support`, `state_version`, `deterministic`, and `replay_safe`;
operators additionally report `state_layouts`, the advertised checkpoint layout
inventory. Inspect the native operator contract when evaluating checkpoint
compatibility; the rolling catalog limitation is described below. Providers additionally report
`supports_static_inputs`, `partition_contract`,
and optional `array_rules`. The vocabularies are closed: execution modes are
`batch` and `stream`; output finality is `per_row_final`,
`group_final_append_only`, or `unproven`; checkpoint support is `stateless`,
`checkpointed_stateful`, or `unproven`; and a provider partition contract is
`none` or `row_axis_independent`. `state_version` is a positive integer
exactly when `checkpoint_support` is `checkpointed_stateful` and `None`
otherwise, a `stateless` capability must set `stateful=False`, and
`state_layouts` is a strictly ascending tuple of positive integers that must
be empty unless `checkpointed_stateful` and must contain `state_version`.
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

| Operator          | Modes         | Finality                | Checkpoint support    | State version | State layouts |
|-------------------|---------------|-------------------------|-----------------------|---------------|---------------|
| `cross_section@1` | batch, stream | group_final_append_only | checkpointed_stateful | 1             | 1             |
| `expression@1`    | batch, stream | per_row_final           | stateless             | —             | —             |
| `rolling@1`       | batch, stream | per_row_final           | checkpointed_stateful | 1             | 1, 2          |
| `sql@1`           | batch, stream | unproven                | stateless             | —             | —             |
| `stream_join@1`   | stream        | unproven                | checkpointed_stateful | 1             | 1             |

The Python capability catalog currently reports only layouts `1` and `2`
for `rolling@1`, while the native operator writes columnar checkpoint layout
`3`. The catalog therefore omits the current writer layout; do not use this
inventory alone to decide rolling checkpoint compatibility. See
[native rolling state](symbolic-design.md#native-rolling-state) for the
implemented encoding and restore rules.

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

The trusted `register_numpy` and `register_jax` helpers additionally attach
process-local stateless stream proofs to their `expression@1` and
`symbolic_matrix@1` registrations. Those provider entries report
`modes=("batch", "stream")`,
`finality="per_row_final"`, `microbatch_invariant=True`,
`deterministic=True`, `replay_safe=True`, and a `row_axis_independent` array
contract. `expression@1` carries `elementwise_broadcast@1` and does not support
static inputs; `symbolic_matrix@1` carries the matrix shape rules and does.
The public `Runtime.register_provider` signature has no lifecycle arguments,
so an arbitrary Python callback remains batch-only rather than acquiring this
proof from caller-supplied metadata.

Compiled-in connector registrations surface on the snapshot's `connectors`
tuple as `ConnectorCapability` entries. Each entry pairs its
`(provider, name, version)` identity and `source`/`sink`/`both` kind with a
`ConnectorCapabilities` value (`delivery`, `replay`, `watermark`,
`transaction`, and the `snapshot`/`polling`/`cdc`/`lookup` flags), the
declared formats, and an options schema.

The session ID is stable for one runtime. A successful registry entry advances
the revision exactly once; rejected duplicates do not. NumPy/JAX helpers add
`expression@1`, mapped `table_matmul@1`, and mapped `symbolic_matrix@1` as
separate entries, so one helper normally advances by three and can expose a
real partial success if a later entry already exists. Previously returned
snapshots remain isolated from later revisions. Snapshots are immutable and
defensively copied: mutating a caller-owned sequence after registration cannot
change a returned snapshot. Operators sort by `(kind, version)`; providers and
UDFs sort by `(provider, name, version)`.

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
batch named `output`. The direct `table_matmul@1` operator remains batch-only.
Streaming static-weight multiplication uses the separately registered
`symbolic_matrix@1` provider through the explicit symbolic compilation shape
described below; `table_matmul@1` itself does not acquire a stream lifecycle.

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

The [symbolic API reference](symbolic-api.md) covers typed declarations,
`FeatureSet`, `Program`, static analysis, ordering, and the supported
batch/stream compilation shapes. Use the [symbolic workflow guide](symbolic-workflows.md)
with examples 09–13 to learn these features. Compiler ownership and physical
sharing are described in [symbolic compiler design](symbolic-design.md).

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
The [symbolic workflow guide](symbolic-workflows.md) maps the symbolic examples
to analysis, lowering, checkpoint recovery, static inputs, and Studio.
