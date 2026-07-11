# Migrating to Calc Flow v0.2

Calc Flow v0.2 intentionally breaks the prototype API to establish explicit
batch ownership and DataFusion-only table execution.

## Wrap all pipeline data in Batch

Raw Arrow tables, record batches, and arrays are no longer accepted by engines,
operators, pipelines, or runners.

Before:

```python
result = engine.evaluate("c = a + b", table)
```

After:

```python
from calc_flow import Batch

result = engine.evaluate("c = a + b", Batch.table(table))
table = result.table_payload
```

Use `Batch.array(array)` for Array API data and
`Batch.from_tabular_protocol(value)` for generic Arrow-compatible producers.

## Use DataFusion for table calculations

`DataFrameEngine`, `PandasEngine`, and `PolarsEngine` have been removed. Replace
them with:

```python
from calc_flow.engine import DataFusionEngine
```

`DataFusionEngine.evaluate()` handles a calculated expression or assignment.
`DataFusionEngine.sql()` handles named table batches and a single `SELECT`
query.

Calc Flow no longer installs pandas or Polars. External objects may still work
through Arrow C Stream or DataFrame Interchange without a library-specific
adapter or compatibility promise.

## Update array calls

Array engine operations now accept a `Batch` as their primary operand and
return a `Batch`:

```python
batch = Batch.array(np.asarray([1, 2, 3]))
result = NumpyEngine().multiply(batch, 2)
assert result.array_payload.tolist() == [2, 4, 6]
```

Expression evaluation no longer executes arbitrary Python. Expressions using
imports, arbitrary attribute access, comprehensions, lambdas, or unapproved
function calls must be replaced with supported `xp` calls or explicit engine
operations.

## Consume micro-batch results

`Operator.apply(batch)` and `Pipeline.apply(batch)` have been replaced by named
port execution. Operators implement:

```python
def process(self, inputs, context):
    return {"output": transformed_batch}
```

Build and compile a graph before execution:

```python
plan = (
    Pipeline("example")
    .then(ExpressionOperator("calculate", "c = a + b"))
    .compile()
)
run = plan.execute({"input": batch})
batch = run.output
```

Use `add_node()` and `connect()` for branching or multi-input nodes. External
inputs and terminal outputs are named from their ports; repeated terminal names
are qualified as `node.port`.

`MicroBatchRunner.run()` is an iterator of `RunResult` objects:

```python
for run in runner.run(source, sink):
    consume(run.output)
```

The runner now accepts a `Source` with `read(cursor=None)`, rather than a raw
iterator. The unused runner `batch_size` argument and offset-based recovery have
been removed. Use `BatchingSource` or form batches and source cursors in a custom
source.

`CheckpointManager` has been replaced by the `CheckpointStore` protocol and
`FileCheckpointStore`. Existing checkpoint JSON is incompatible. New
checkpoints are versioned and fingerprinted; reset old checkpoint directories
when upgrading. A checkpoint is saved only after configured sinks succeed, and
delivery is at-least-once.

## Register custom calculations explicitly

Arbitrary table callbacks and inline code are not UDF configuration. Install a
trusted vectorized implementation in `UdfRegistry`, give it a stable name and
version, and list a `UdfReference` on every operator that uses it. Only the name
and version belong in serialized configuration.

Table UDFs must declare Arrow input and return fields plus DataFusion volatility.
Array UDFs declare their argument count and must return an Array API object in
the current backend. Aggregate, window, table-generating, row-object, and
stateful UDFs are not supported in v0.2.

## Replace ad-hoc graph dictionaries

Persisted projects now validate through strict Pydantic configuration models.
Set `format_version` to `"1"`, choose expression or SQL for every table node,
declare edges by node and port, and store UDFs only as name/version references.
Table nodes do not accept a dataframe backend field.

Canonical project storage is JSON. YAML files may be safely imported or
exported, but are rewritten as canonical JSON on save. Unknown fields are
errors, so remove obsolete pandas/Polars selectors and inline callback/import
configuration before importing an older project.

## Install optional array backends

The core installation contains PyArrow and DataFusion. Install only the array
backend needed by the application:

```bash
uv add "calc-flow[numpy]"
uv add "calc-flow[jax]"
```

Development installs include both backends:

```bash
uv sync --extra dev
```
