# Calc Flow

## Introduction

Calc Flow is a micro-batch and streaming stateful calculation engine based on
Arrow-backed batches. Data moves through pipelines as `pyarrow.Table` or
`pyarrow.RecordBatch` values, while computation can be delegated to dataframe
or array engines.

## Calculation Modes

Calc Flow supports two execution modes:

* micro-batch mode for iterators of batches ranging from dozens to tens of
  thousands of rows;
* streaming mode for one batch per `step()` call.

Both runners use checkpoints for state recovery. Calling `reset()` clears
operator state and the persisted checkpoint for that pipeline.

## Batches

```python
from calc_flow import Batch

batch = Batch.from_pylist([
    {"a": 1, "b": 10},
    {"a": 2, "b": 20},
])
```

`Batch` accepts Arrow tables and record batches directly, and can also be built
from pandas, polars, Python row dictionaries, or array-like columns.

## Engines

Dataframe engines:

* `PandasEngine` supports expression evaluation through `DataFrame.eval`;
* `PolarsEngine` supports expression evaluation and SQL through Polars SQL;
* `DataFusionEngine` supports expression evaluation and SQL through Apache
  DataFusion.

Array engines:

* `NumpyEngine` evaluates array expressions over batch columns;
* `JaxEngine` uses the same array expression path with JAX.

```python
from calc_flow.engine.dataframe import DataFusionEngine

engine = DataFusionEngine()
result = engine.evaluate("c = a + b", batch)
assert result.to_pylist() == [
    {"a": 1, "b": 10, "c": 11},
    {"a": 2, "b": 20, "c": 22},
]
```

## Pipelines

```python
from calc_flow import Pipeline, StatelessOperator
from calc_flow.engine.dataframe import DataFusionEngine

pipeline = Pipeline("example").add(
    StatelessOperator("add_c", lambda b: DataFusionEngine().evaluate("c = a + b", b))
)

result = pipeline.apply(batch)
```

Operator names must be unique within a pipeline because checkpoints are keyed by
operator name.
