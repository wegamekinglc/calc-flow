# Batch calculations

[Documentation](README.md) / 2.1 Batch calculations

Use `cf.compute(data, build)` when a complete Arrow input is available.
The builder composes immutable table expressions and the result is a
`pyarrow.Table`. Supported Arrow schema, default names, and the internal
runtime are selected automatically. See [installation](getting-started.md).

## Expressions, projection, and filtering

Run [01_datafusion_pipeline.py](../examples/01_datafusion_pipeline.py):

```bash
uv run --no-sync python examples/01_datafusion_pipeline.py
```

The example's reusable builder calculates gross amounts and a floating fee,
filters derived values, and selects the final columns:

```python
import calc_flow as cf
import pyarrow as pa

data = pa.table({
    "order_id": ["A-100", "A-101", "A-102"],
    "quantity": [3, 1, 4],
    "unit_price": [10, 12, 10],
})

def large_orders(t: cf.TableExpr) -> cf.TableExpr:
    gross = t["quantity"] * t["unit_price"]
    enriched = t.with_columns(gross=gross, fee=cf.row.cast(gross, "float64") / 10.0)
    return enriched.filter(enriched["gross"] >= 20).select("order_id", "gross")

rows = cf.compute(data, large_orders)
assert rows.to_pylist() == [
    {"order_id": "A-100", "gross": 30},
    {"order_id": "A-102", "gross": 40},
]
```

The runnable example calls the same builder with its orders table. The result
contains:

```text
[{'order_id': 'A-100', 'gross': 30}, {'order_id': 'A-102', 'gross': 40}]
```

`t["quantity"]` selects a column expression. Arithmetic and comparisons build
calculations, not formula strings. Compose boolean conditions with parenthesized
comparisons and `&`, `|`, `~`; Python `and`, `or`, `not`, and expression truth
checks are invalid. Use `cf.lit` for a standalone literal and `identical()`
for structural identity.

`with_columns` accepts keyword expressions or an ordered mapping;
`select` takes literal existing column names followed by named expressions;
`filter` takes a boolean expression. Each returns a new declaration and leaves
its inputs unchanged. Read a derived column from the returned table, as the
filter above does. Expressions within one `with_columns` call share its input.

Column names are append-only. Neither `with_columns` nor named `select`
expressions replace existing columns, and duplicate output names fail. Use a
new name for a derived value. The schema/name and row-lineage rules are in the
[Python reference](python-api.md#table-expressions-and-names).

Types stay strict: integer arithmetic does not automatically become floating
arithmetic. The fee above uses `cf.row.cast(gross, "float64") / 10.0` explicitly.
Supported field types, field nullability, and order come from Arrow; unsupported
schemas fail with the input field path. Rename fields outside the portable
identifier rule `[A-Za-z_][A-Za-z0-9_]*` before passing data to the expression API.

## Reusable programs and named outputs

Declare a table once from an Arrow schema, then name several output tables.
Continue with the `data` orders table above:

```python
t = cf.table_input("orders", schema=data.schema)
gross = t["quantity"] * t["unit_price"]
program = cf.Program("orders", outputs={
    "totals": t.select("order_id", gross=gross),
    "quantities": t.select("order_id", "quantity"),
})
tables = program.collect({"orders": data})
assert list(tables) == ["totals", "quantities"]
assert tables["totals"]["gross"].to_pylist() == [30, 12, 40]
```

Omitted `Program.inputs` are discovered from the outputs. Input mappings use
those declared names; output mappings retain logical names and insertion order.
For a single output, `TableExpr.collect(data)` accepts a table directly when
there is exactly one table root. Multiple roots require a mapping.
`Program.collect` always requires a mapping. The same methods accept table
`Batch` and Arrow `RecordBatch` inputs.

Each convenience call compiles a fresh batch plan, even with a shared explicit
`Runtime`. Repeated calls do not retain rolling history and do not reset or
reuse an explicitly cached plan. For owned plan state, snapshot/restore/reset,
node timings, or DataFusion metrics, use `program.compile_batch(runtime)` and
the explicit execution APIs in the [Python reference](python-api.md).

## Financial expressions

Reusable Python functions can compose rolling formulas and named results. This
example declares its event-time field explicitly:

```python
quote_schema = pa.schema([
    pa.field("symbol", pa.string(), nullable=False),
    pa.field("ts", pa.timestamp("us", tz="UTC"), nullable=False),
    pa.field("price", pa.float64(), nullable=False),
])
quotes = pa.table({
    "symbol": ["AAA", "AAA", "AAA"],
    "ts": [1_000_000, 2_000_000, 3_000_000],
    "price": [100.0, 102.0, 101.0],
}, schema=quote_schema)

def signals(q: cf.TableExpr) -> cf.TableExpr:
    price = q["price"]
    momentum = price / cf.ts.lag(price) - 1.0
    mean = cf.ts.mean(price, window=cf.rows(3))
    return q.select("symbol", "ts", momentum=momentum, mean_price=mean)

result = cf.compute(
    quotes,
    signals,
    entity_by=("symbol",),
    event_time="ts",
    sequence_by=("ts",),
)
```

Rolling needs non-empty entity and sequence keys plus non-null
`timestamp[us, UTC]` event time. Arrow's ordinary inferred timestamp fields are
nullable even when all values are present; declare `nullable=False` in the schema.
The `price` field and its divisors here are floating-point. Null/warm-up and
ordering rules remain those of the native rolling operators.

[Example 09](../examples/09_symbolic_financial_features.py) extends this pattern
with reusable mappings for returns, Bollinger bands, RSI, EMA/MACD, and
cross-section statistics. See [expression workflows](symbolic-workflows.md).

## Async execution and deadlines

Run [05_async_execution.py](../examples/05_async_execution.py):

```bash
uv run --no-sync python examples/05_async_execution.py
```

The example awaits `cf.compute_async`, starts an asyncio heartbeat, passes
request settings and an aware deadline through `cf.ExecutionOptions`, and checks
totals `[3, 7]`. `TableExpr.collect_async` and `Program.collect_async` use the
same native execution bridge. Blocking `compute` and `collect` reject an active
event loop. Builders remain synchronous and execute once when the API is called.

Async calls copy input mappings and capture `Batch` references before execution.
Arrow buffers remain shared; keep the underlying storage read-only until the
awaited execution completes. No table-content deep copy is promised. Arrow
schema/field metadata are omitted from the internal execution schema only;
caller objects and metadata, including `Batch.metadata`, remain intact.

Options copy caller settings and normalize aware deadlines to UTC. Deadlines
are cooperative; cancellation waits for native work and cleanup to settle.
See the [full cancellation contract](python-api.md#async-execution).

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

Formula strings, explicit graph connections, and read-only SQL remain supported
integration APIs. They are useful for registered UDF calls and graph/provider
operations outside the expression catalog. They execute in the same Rust
runtime. For diagnostics and tuning, see
[SQL performance controls](sql-datafusion-performance.md).

Next: [arrays and matrices](array-guide.md).
