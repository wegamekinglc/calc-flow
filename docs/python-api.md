# Python API

The `calc-flow==2.0.0` Python package is a PyO3 binding to the Rust engine plus
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
plan = configured.compile()
result = plan.execute({"input": batch})

assert builder.project["pipeline"]["nodes"] == []
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
    .compile()
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
    .compile(runtime)
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

assert snapshot.schema_version == 1
assert snapshot.scope.kind == "runtime_session"
assert snapshot.scope.revision == 1
assert snapshot.providers[0].name == "normalize"
```

The public frozen values are `ProviderOption`, `ProviderOptionsSchema`,
`ProviderPort`, `ProviderCapability`, `UdfCapability`, `OperatorCapability`,
`RuntimeSessionScope`, and `RuntimeCapabilities`. Provider option schema
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
        PipelineBuilder("async-example").expression("calc", "total = a + b").compile()
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
waits through repeated Python task cancellation until cleanup finishes.

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
    .compile(runtime)
)
batch = Batch.from_array(np.array([1.0, 2.0, 4.0, 6.0]), backend="numpy")
centered = plan.execute({"input": batch}).outputs["output"].array
```

Owned arrays are read-only. The bounded expression evaluator allows arithmetic,
reductions, transpose, and reshape; it rejects Python execution features and
backend changes. `register_jax` provides the same explicit provider boundary.
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
    .compile(runtime)
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

## Projects and persistence

`ProjectDocument` validates a strict `format_version: 2` mapping with the Rust
schema. `project_json_schema()` returns the generated schema;
`validate_project_json(document)` returns canonical JSON.

`FileProjectStore` has async `create`, `put`, `get`, `list`, and `delete`
methods and explicit `*_blocking` variants. Safe JSON/YAML import/export helpers
live in `calc_flow.store`. `FileCheckpointStore` exposes async and blocking
`save`, `load`, and `delete` operations.

## Micro-batch runner

A Python source owns replay state:

```python
class Source:
    def open(self, cursor: object) -> None:
        self.offset = 0 if cursor is None else int(cursor["offset"])

    def next(self):
        if self.offset == len(self.values):
            return None
        value = self.values[self.offset]
        self.offset += 1
        return (
            Batch.from_pyarrow(pa.table({"value": [value]})),
            {"offset": self.offset},
            self.offset,
        )
```

Construct `MicroBatchRunner(plan, source, checkpoints, sinks=...,
checkpoint_every=...)`. Call `next_async()` until it returns `None`, or use
blocking `next()` outside an event loop. Recovery opens a new source with the
last committed cursor.

## Streaming runner

`StreamingRunner(plan, checkpoints)` processes a formed batch through
`step_async(batch, sinks=...)` or blocking `step`. Sink mappings are
`dict[str, Sequence[Callable[[Batch], object]]]`; callbacks may be sync or
async. All routes are validated before delivery.

Both runner modes checkpoint only after all sinks succeed and provide
at-least-once delivery. `reset[_async]` clears recovery state and
`plan_snapshot[_async]` returns a defensive state document.

## Exceptions

Catch the narrowest exported class: `ConfigError`, `CompileError`,
`ExecutionError`, `ProviderError`, `CheckpointError`, or `CancelledError`.
All derive from `CalcFlowError`; provider/cancellation errors are execution
errors.

## More examples

Every file under [`examples/`](../examples/README.md) is executable against the
installed v2 wheel. See the linked inventory for the commands.
