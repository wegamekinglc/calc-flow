# Python API

The `calc-flow==2.0.0` Python package is a PyO3 binding to the Rust engine plus
small functional adapters. Python 3.13 or newer is required.

## Install and develop

```bash
uv add calc-flow
uv add "calc-flow[numpy]"  # optional
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
    PipelineBuilder("join")
    .sql(
        "query",
        "SELECT left.a + right.b AS total FROM left CROSS JOIN right",
        aliases=("left", "right"),
    )
    .compile()
)
result = plan.execute(
    {
        "left": Batch.from_pyarrow(pa.table({"a": [2]})),
        "right": Batch.from_pyarrow(pa.table({"b": [3]})),
    }
)
assert result.outputs["output"].to_pyarrow()["total"].to_pylist() == [5]
```

Only one read-only DataFusion `SELECT` or CTE is accepted.

## Trusted Python scalar UDFs

```python
import pyarrow.compute as pc

from calc_flow import PipelineBuilder, Runtime

runtime = Runtime()
runtime.register_scalar_udf(
    provider="python",
    name="double_value",
    version="1",
    input_types=("int64",),
    return_type="int64",
    volatility="immutable",
    function=lambda values: pc.multiply(values, 2),
)
plan = (
    PipelineBuilder("udf")
    .expression(
        "calculate",
        "result = double_value(value)",
        udfs=(("python", "double_value", "1"),),
    )
    .compile(runtime)
)
```

Callbacks are trusted application code, vectorized over PyArrow arrays, and
never serialized. Registration and execution enforce exact Arrow types,
result length/type, and explicit node references.

## Async execution

```python
async def calculate() -> list[int]:
    plan = PipelineBuilder("async").expression("calc", "b = a + 1").compile()
    result = await plan.execute_async(
        {"input": Batch.from_pyarrow(pa.table({"a": [1, 2]}))}
    )
    return result.outputs["output"].to_pyarrow()["b"].to_pylist()
```

Blocking `execute`, store, and runner methods reject a running event loop. Use
their async forms in servers and asyncio applications. Cancelling an async
execution or runner call waits for native cleanup before the plan is reusable.

## NumPy and JAX

```python
import numpy as np

from calc_flow import Batch, PipelineBuilder, Runtime, register_numpy

runtime = Runtime()
register_numpy(runtime)
plan = (
    PipelineBuilder("arrays")
    .external(
        "center",
        "numpy",
        "expression",
        "1",
        {"expression": "x - mean(x)"},
    )
    .compile(runtime)
)
batch = Batch.from_array(np.array([1.0, 2.0, 3.0]), backend="numpy")
centered = plan.execute({"input": batch}).outputs["output"].array
```

Owned arrays are read-only. The bounded expression evaluator allows arithmetic,
reductions, transpose, and reshape; it rejects Python execution features and
backend changes. `register_jax` provides the same explicit provider boundary.

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

Every file under `examples/` is executable against the installed v2 wheel. See
`examples/README.md` for the inventory and commands.
