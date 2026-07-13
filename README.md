# Calc Flow

Calc Flow is a micro-batch and streaming stateful calculation engine. Tabular
data is carried in immutable `Batch` envelopes backed by Apache Arrow and is
queried exclusively with Apache DataFusion. Array data remains backed by a
Python Array API implementation such as NumPy or JAX.

Python v1 is frozen at the `v1-python-final` tag. This document and the
committed semantic corpus are behavioral references for the Python v1
implementation; they are not a promise that Rust v2 will preserve Python API
compatibility.

## Installation

Install the core Arrow and DataFusion runtime:

```bash
uv add calc-flow
```

Install an optional array backend when needed:

```bash
uv add "calc-flow[numpy]"
uv add "calc-flow[jax]"
```

## Examples

Runnable examples live in [`examples/`](examples/README.md). They cover linear
DataFusion expressions, multi-input SQL with graph branching, registered UDFs,
micro-batch recovery, persisted project configuration, and optional NumPy array
processing. A clean quickstart notebook is included under `examples/notebooks/`.

For example:

```bash
uv run python examples/01_datafusion_pipeline.py
uv run python examples/04_micro_batch_recovery.py
```

## Batches

`Batch` is an immutable envelope that carries a table or array together with
ordering and provenance metadata.

```python
import pyarrow as pa

from calc_flow import Batch, BatchMetadata

batch = Batch.table(
    pa.table({"a": [1, 2], "b": [10, 20]}),
    metadata=BatchMetadata(sequence=0, source_id="example"),
)
```

`Batch.table()` accepts `pyarrow.Table` and `pyarrow.RecordBatch`. Generic
objects exposing the Arrow C Stream or DataFrame Interchange protocols can be
loaded with `Batch.from_tabular_protocol()` without adding a direct pandas or
Polars dependency.

Array batches require an object implementing `__array_namespace__`:

```python
import numpy as np

array_batch = Batch.array(np.asarray([1, 2, 3]))
```

## DataFusion calculations

DataFusion is the only table query and calculation engine.

```python
import pyarrow as pa

from calc_flow import Batch
from calc_flow.engine import DataFusionEngine

batch = Batch.table(pa.table({"a": [1, 2], "b": [10, 20]}))
result = DataFusionEngine().evaluate("c = a + b", batch)

assert result.table_payload.to_pylist() == [
    {"a": 1, "b": 10, "c": 11},
    {"a": 2, "b": 20, "c": 22},
]
```

Multi-table SQL accepts named table batches:

```python
result = DataFusionEngine().sql(
    "select l.id, l.a + r.b as total "
    "from left_table l join right_table r on l.id = r.id",
    {"left_table": left_batch, "right_table": right_batch},
)
```

Only a single `SELECT` statement or CTE is accepted. DDL, DML, and
multi-statement SQL are rejected.

## Registered UDFs

Custom table calculations are trusted, registered, vectorized DataFusion
scalar UDFs. Each version declares named Arrow input fields, an Arrow return
field, volatility, and description.

```python
import pyarrow as pa
import pyarrow.compute as pc

from calc_flow import ExpressionOperator, Pipeline, UdfReference, UdfRegistry

registry = UdfRegistry()

@registry.datafusion_scalar(
    name="double_value",
    version="1",
    input_fields=[pa.field("value", pa.int64())],
    return_field=pa.field("result", pa.int64()),
    volatility="immutable",
)
def double_value(values):
    return pc.multiply(values, 2)

plan = (
    Pipeline("udf-example", udf_registry=registry)
    .then(
        ExpressionOperator(
            "calculate",
            "doubled = double_value(value)",
            udfs=[UdfReference("double_value", "1")],
        )
    )
    .compile()
)
```

Only UDF versions explicitly referenced by operators are installed into a
run's DataFusion session. Multiple versions may exist in the application
registry, but one run cannot bind two versions to the same SQL function name.
Input nullability, output type/nullability, output length, and implementation
failures are checked before downstream nodes execute.

Array UDFs use the parallel `registry.array()` decorator and are callable only
from `ArrayExpressionOperator` or an array engine configured with an explicit
`UdfReference`. Their results must remain in the active NumPy or JAX backend.

`registry.catalog()` returns JSON-compatible metadata for discovery. Catalogs
and serialized pipeline references never contain function source code, Python
import paths, or callable objects.

## Array calculations

Array engines operate on array batches and preserve batch metadata:

```python
from calc_flow.engine import NumpyEngine

engine = NumpyEngine()
result = engine.evaluate("xp.maximum(x * 2, 3)", array_batch)
assert result.array_payload.tolist() == [3, 4, 6]
```

Array expressions are interpreted through an allowlisted AST. They cannot
import modules, access arbitrary attributes, call builtins, or execute Python
statements.

## Pipelines and runners

Pipelines are compiled DAGs with named, typed ports. `then()` and `add()` are
linear construction sugar; `add_node()` and `connect()` build branches and
multi-input graphs.

```python
from calc_flow import ExpressionOperator, Pipeline

plan = (
    Pipeline("example")
    .then(ExpressionOperator("add_c", "c = a + b"))
    .compile()
)

run = plan.execute({"input": batch})
result = run.output
```

`RunResult` contains named terminal outputs, per-node timings, DataFusion query
plans and timings, warnings, and run metadata. One run owns one DataFusion
session shared by its table operators.

`ExpressionOperator` supports calculated expressions, projection, and filters.
`SqlOperator` exposes named table input ports for joins, aggregation, windows,
sorting, and other single-query SQL:

```python
from calc_flow import SqlOperator

join = SqlOperator(
    "join",
    "select l.id, l.amount + r.amount as total "
    "from left_table l join right_table r on l.id = r.id",
    inputs=("left_table", "right_table"),
)
```

## Sources, sinks, and recovery

Sources implement `read(cursor=None)` and sinks implement `write(batch)`.
`BatchingSource` forms Arrow batches by row and byte limits and records the next
source cursor in batch metadata.

```python
from calc_flow import BatchingSource, FileCheckpointStore, MicroBatchRunner

source = BatchingSource(records, source_id="orders", max_rows=10_000)
runner = MicroBatchRunner(
    plan,
    checkpoint_store=FileCheckpointStore(".calc-flow-checkpoints"),
)

for run in runner.run(source, sink):
    print(run.node_timings)
```

Checkpoints are versioned and include the compiled pipeline fingerprint, source
cursor, sequence, state by node ID, and creation time. They are committed only
after every configured sink succeeds. Recovery rejects stale fingerprints and
provides at-least-once delivery.

## Project configuration and local studio

`ProjectConfig` is the versioned, data-only representation of a graph. It
contains nodes, edges, typed ports, DataFusion settings, UDF name/version
references, sample sources, and bounded run options. `compile_project()` turns
the same model used by the API into an `ExecutionPlan`.

Projects are stored as deterministic formatted JSON through `FileProjectStore`.
YAML is supported only for safe import and export. Unknown fields, executable
objects, inline source, import paths, invalid SQL modes, and table backend
selectors are rejected.

Start the API and Vite development server together from the repository root:

```bash
./web-ui/scripts/start_web_ui.sh
```

Open `http://127.0.0.1:5173`, then stop both managed process groups with:

```bash
./web-ui/scripts/stop_web_ui.sh
```

The start script installs locked frontend dependencies when `node_modules` is
missing, waits for both health endpoints, and records process IDs and logs under
`.calc-flow-web/`. Start and stop are idempotent, and stale process records are
cleaned up on the next start.

For a production-style local build, run `npm ci && npm run build:wheel` in
`web-ui/`, then run `uv run --package calc-flow-studio calc-flow-web` from the
repository root and open `http://127.0.0.1:8765`. The core `calc-flow` package
contains no FastAPI service or studio assets. The server refuses non-loopback
bind addresses and has no public-hosting or authentication mode in v0.2.

During frontend development, Vite proxies `/api` to the loopback API. The
studio supports visual fan-out/fan-in editing, DataFusion expression and SQL
forms, Arrow schemas, registered UDF selection, CSV/JSON/Arrow IPC samples,
validation, result tables, logical/physical plan and timing inspection,
runner-checkpoint inspection/reset, and local benchmark report comparison.
Preview runs remain stateless and do not create runner checkpoints.

Preview runs use managed worker processes. Defaults are capped at 10 MiB,
100,000 input rows, 30 seconds, and 512 MiB resident memory. The REST API is
under `/api/v1`; generated TypeScript types come from its OpenAPI document.

## Development

```bash
uv sync --extra dev
uv run pytest
uv run pytest -n auto
uv run pytest --cov=calc_flow --cov-report=term-missing
uv run ruff check .
uv run ruff format --check .
cd web-ui && npm ci && npm run build && npm test && npm run test:e2e
```

Run deterministic informational benchmarks with:

```bash
uv sync --extra benchmark
CALC_FLOW_BENCHMARK_SCALE=overhead \
  uv run pytest benchmarks --benchmark-only \
  --benchmark-json=benchmark-results/overhead.json
```

See [`benchmarks/README.md`](benchmarks/README.md) for scales, recorded metrics,
and comparison policy. See [`docs/api-reference.md`](docs/api-reference.md) for
the supported Python and local HTTP APIs.

See `docs/migration-v0.2.md` for the breaking API changes and
`design/v0.2-refactor-plan.md` for the remaining roadmap.
