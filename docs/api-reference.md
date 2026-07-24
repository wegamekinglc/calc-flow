# Calc Flow 2.0 API reference

Calc Flow has three supported surfaces:

| Surface            | Package or path                    | Purpose                                      |
| ------------------ | ---------------------------------- | -------------------------------------------- |
| Rust core          | `calc-flow = "2.0.0"`              | Native batches, graphs, execution, recovery  |
| Python binding     | `calc-flow==2.0.0`                 | PyO3 engine access and Python integrations   |
| Local Studio API   | `calc-flow-studio==2.0.0`          | Loopback FastAPI service and React assets    |

For examples and lifecycle detail, see [the Rust API](rust-api.md) and
[the Python API](python-api.md).

## Examples

Minimal end-to-end calculation (Python):

```python
import pyarrow as pa

from calc_flow import Batch, PipelineBuilder

batch = Batch.from_pyarrow(pa.table({"a": [1, 3], "b": [2, 4]}))
plan = (
    PipelineBuilder("totals")
    .expression("calculate", "total = a + b")
    .compile()
)
result = plan.execute({"input": batch})

assert result.outputs["output"].to_pyarrow()["total"].to_pylist() == [3, 7]
```

The runnable inventories span both surfaces and share datasets and expressions:

- Python: [`examples/README.md`](../examples/README.md) — expression pipeline,
  SQL join, registered UDF, micro-batch recovery, async execution, and NumPy
  arrays plus NumPy/JAX `pyarrow.Table` matrix multiplication.
- Rust: [`crates/calc-flow/examples/README.md`](../crates/calc-flow/examples/README.md)
  — `expression_pipeline.rs`, `sql_join.rs`, `micro_batch_recovery.rs`, and
  `export_schema.rs`.

## Rust modules and exports

The `calc_flow` crate re-exports its supported public types from
`crates/calc-flow/src/lib.rs`.

| Area                 | Primary APIs                                                        |
| -------------------- | ------------------------------------------------------------------- |
| Data                 | `Batch`, `BatchKind`, `BatchMetadata`, `TableBatch`                 |
| Graph                | `PipelineBuilder`, `Edge`, `PortEndpoint`, `ExecutionPlan`          |
| Operators            | `Port`, `Operator`, `ExpressionOperator`, `SqlOperator`             |
| Execution            | `ExecutionOptions`, `RunResult`, `RunMetadata`, `NodeTiming`        |
| UDF/providers        | `UdfRegistry`, `UdfReference`, `ProviderRegistry`                   |
| Sources and sinks    | `Source`, `SourceItem`, `Sink`, `BatchingSource`                    |
| Recovery             | `MicroBatchRunner`, `StreamingRunner`, `CheckpointStore`            |
| Projects             | `ProjectSpec`, `compile_project`, `validate_project`                |
| Persistence          | `FileProjectStore`, `FileCheckpointStore`                           |
| Errors               | `CalcFlowError`, `Result<T>`                                        |

Generate local rustdoc with:

```bash
RUSTDOCFLAGS="-D warnings" cargo doc --workspace --all-features --no-deps
```

## Python package

Import the main surface from `calc_flow`.

### Batch

| Member                                      | Contract                                                   |
| ------------------------------------------- | ---------------------------------------------------------- |
| `Batch.from_pyarrow(table, metadata=None)`  | Own an Arrow C Stream as an immutable table batch          |
| `Batch.from_array(array, backend=..., ...)` | Own a read-only explicitly named array-provider payload    |
| `to_pyarrow()`                              | Return the table payload or reject a non-table batch       |
| `array` / `backend`                         | Return array-provider data or reject a table batch         |
| `kind` / `num_rows` / `metadata`            | Return defensive batch observations                        |

### Graph and execution

`PipelineBuilder(name)` supports:

- `expression(name, expression, *, select=(), filter=None, udfs=())`;
- `sql(name, query, *, aliases=("input",), udfs=())`;
- `external(node_id, provider, name, version, options)`;
- `table_matmul(node_id, *, backend: Literal["numpy", "jax"],
  columns: Sequence[str])`;
- `connect(source_node, target_node, *, source_port="output",
  target_port="input")`;
- `compile(runtime=None) -> ExecutionPlan`.

`ExecutionPlan` exposes immutable `name` and `fingerprint`.
`execute(inputs)` is blocking and rejects a running event loop.
`execute_async(inputs)` is the asynchronous form. `snapshot[_async]`,
`restore[_async]`, and `reset[_async]` provide the plan-state lifecycle.

`RunResult` exposes defensive `outputs`, `metadata`, `node_timings`, and
`datafusion_metrics` values.

### Runtime and UDFs

`Runtime` compiles strict project JSON, reports validation results, registers
trusted providers/scalar UDFs, and returns a redacted metadata catalog.
`register_scalar_udf` requires provider/name/version, exact input type names,
return type, volatility, and a vectorized callable.

`register_numpy(runtime)` installs `numpy:expression@1` and
`numpy:table_matmul@1`; `register_jax(runtime)` installs `jax:expression@1`
and `jax:table_matmul@1`. The matrix providers accept an Arrow table input
named `table` and same-backend array weights named `weights`, then produce an
array output. After input `Batch` construction they make no redundant
execution copies: NumPy uses one dense table allocation and one result; JAX
permits one host staging buffer, one device table buffer, and one device
result. JAX performs no result-to-host round trip during operator execution.
These are execution ceilings, not end-to-end zero-copy claims.

### Projects and stores

`ProjectDocument` is the strict Pydantic root model for format v2.
`project_json_schema()` returns the Rust-generated schema and
`validate_project_json()` returns canonical validated JSON.

`FileProjectStore` provides async `create`, `put`, `get`, `list`, and
`delete` plus explicit `*_blocking` forms. Import/export helpers accept safe
JSON or YAML. `FileCheckpointStore` provides the equivalent async and blocking
checkpoint operations.

### Runners

`MicroBatchRunner(plan, source, checkpoints, *, sinks=None,
checkpoint_every=100)` accepts a source with `open(cursor)` and `next()`.
`next()` returns one `RunResult` or `None`; `next_async()` is the
non-blocking form.

`StreamingRunner(plan, checkpoints)` accepts a formed batch through
`step(..., sinks=None)` or `step_async(..., sinks=None)`.

Sink mappings use output names and sequences of sync or async callbacks. Both
runners expose reset and plan-snapshot operations.

## Local HTTP API

The separate Studio service exposes its supported API under `/api/v2`.

| Method                 | Route                       | Purpose                                  |
| ---------------------- | --------------------------- | ---------------------------------------- |
| `GET`                  | `/catalog`                  | Operators, UDFs, Arrow types, limits     |
| `GET`                  | `/schema/project`           | Rust-generated v2 project JSON Schema    |
| `GET`, `POST`          | `/projects`                 | List or create projects                  |
| `POST`                 | `/projects/import`          | Safely import JSON or YAML               |
| `GET`, `PUT`, `DELETE` | `/projects/{id}`            | Read, replace, or delete a project       |
| `GET`                  | `/projects/{id}/export`     | Export canonical JSON or safe YAML       |
| `POST`                 | `/projects/{id}/validate`   | Validate and compile a stored graph      |
| `GET`, `DELETE`        | `/projects/{id}/checkpoint` | Inspect or reset recovery state          |
| `POST`                 | `/projects/{id}/runs`       | Start a bounded preview worker           |
| `GET`                  | `/runs/{id}`                | Read preview status/results              |
| `GET`                  | `/runs/{id}/events`         | Stream run events                        |
| `DELETE`               | `/runs/{id}`                | Cancel a managed preview                 |

The checked contract is [web-ui/openapi.json](../web-ui/openapi.json).
`npm run sync:api` regenerates it and
`web-ui/src/api/schema.d.ts` from the FastAPI application.

## Error categories

Rust returns `CalcFlowError` variants. Python exposes the stable hierarchy:
`CalcFlowError`, `ConfigError`, `CompileError`, `ExecutionError`,
`ProviderError`, `CheckpointError`, and `CancelledError`.

Invalid user documents and graph definitions are configuration/compile errors;
execution, callbacks, source/sink failures, cancellation, and checkpoint
storage preserve their more specific categories.

## Version and compatibility

The Rust crate, Python binding, Studio package, and frontend are versioned
`2.0.0`. Project format version `2` and checkpoint format versioning are
separate protocol values.

Calc Flow 2.0 does not load v1 projects or checkpoints. See
[the release guide](v2-release.md) for the required migration boundary.
