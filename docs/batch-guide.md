# Batch calculations

[Documentation](README.md) / 2.1 Batch calculations

A batch plan receives named immutable `Batch` inputs and returns a `RunResult`
with named outputs, node timings, and DataFusion metrics. Start here when the
complete input is available. These programs need the core package and
PyArrow; see [installation](getting-started.md).

## Expressions, projection, and filtering

Run [01_datafusion_pipeline.py](../examples/01_datafusion_pipeline.py):

```bash
uv run --no-sync python examples/01_datafusion_pipeline.py
```

The input has three orders with `quantity` and `unit_price` columns. The first
node calculates `gross = quantity * unit_price`. The second selects
`order_id` and `gross`, keeping rows where `gross >= 20`:

```python
plan = (
    PipelineBuilder("datafusion-quickstart")
    .expression("calculate_gross", "gross = quantity * unit_price")
    .expression("large_orders", "", select=("order_id", "gross"), filter="gross >= 20")
    .connect("calculate_gross", "large_orders")
    .compile_batch()
)
```

The output is:

```text
[{'order_id': 'A-100', 'gross': 30}, {'order_id': 'A-102', 'gross': 40}]
```

The empty expression selects projection mode; a node needs either one
calculation or a non-empty projection. A filter can accompany either.
`connect` names the producer and consumer nodes; its default ports are
`output` and `input`. Unconnected ports become the plan's named inputs and
outputs. Builder operations leave the original builder unchanged.

## Named inputs and SQL joins

Run [02_sql_join.py](../examples/02_sql_join.py):

```bash
uv run --no-sync python examples/02_sql_join.py
```

The SQL node declares aliases `orders` and `fees`. Supply batches under those
same keys when calling `execute`. It joins on `order_id`, subtracts each fee
from the amount, and uses `ORDER BY` for a defined result order. The resulting
`net` values are `[70, 108, 36]` for order IDs `[1, 2, 3]`.

A SQL node accepts one read-only `SELECT` or CTE. DDL, DML, utility commands,
and multiple statements fail validation. Table calculation uses DataFusion;
there is no table backend selector.

## Registered scalar functions

Run [03_registered_udf.py](../examples/03_registered_udf.py):

```bash
uv run --no-sync python examples/03_registered_udf.py
```

The example registers `double_amount` on a `Runtime`, declaring provider
`python`, version `1`, exact `int64` input/output types, and `immutable`
volatility. The callback receives an Arrow array and uses `pyarrow.compute`
to return a vectorized result. The node explicitly selects
`("python", "double_amount", "1")`, then compiles with that runtime.
Amounts `[100, 250, 400]` produce totals `[200, 500, 800]`.

Registration installs trusted application code. The project stores only the
function identity; loading a project requires registering the same function
before compiling. See [projects and persistence](projects-guide.md).

## Async execution and deadlines

Run [05_async_execution.py](../examples/05_async_execution.py):

```bash
uv run --no-sync python examples/05_async_execution.py
```

The example starts an asyncio heartbeat alongside `execute_async()`, passes
copied request settings and a timezone-aware deadline, and prints totals
`[3, 7]`. Use the async entry point in an event loop; blocking `execute()`
rejects that context. `ExecutionOptions` normalizes its deadline to UTC and
does not retain mutable caller settings.

Inspect `result.outputs` for data and `result.node_timings` for per-node row
counts and duration. Table runs also populate `result.datafusion_metrics`.
Plan text and timing are diagnostics rather than a stable formatted output
to compare byte for byte. For tuning, see [SQL performance controls](sql-datafusion-performance.md).

For rolling and cross-section batch calculations, continue with
[symbolic workflows](symbolic-workflows.md). Exact methods and cancellation
semantics are in the [Python API](python-api.md).

Next: [arrays and matrices](array-guide.md).
