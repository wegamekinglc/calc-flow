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

## Async execution

```python
async def calculate() -> list[int]:
    plan = (
        PipelineBuilder("async-example").expression("calc", "total = a + b").compile()
    )
    result = await plan.execute_async(
        {"input": Batch.from_pyarrow(pa.table({"a": [1, 3], "b": [2, 4]}))}
    )
    return result.outputs["output"].to_pyarrow()["total"].to_pylist()
```

Blocking `execute`, store, and runner methods reject a running event loop. Use
their async forms in servers and asyncio applications. Cancelling an async
execution or runner call waits for native cleanup before the plan is reusable.
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
