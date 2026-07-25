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
plan = PipelineBuilder("totals").expression("calculate", "total = a + b").compile()
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
[`lib.rs`](../crates/calc-flow/src/lib.rs).

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
`execute(inputs, *, options=None)` is blocking and rejects a running event
loop. `execute_async(inputs, *, options=None)` is the asynchronous form.
`snapshot[_async]`,
`restore[_async]`, and `reset[_async]` provide the plan-state lifecycle.

`RunResult` exposes defensive `outputs`, `metadata`, `node_timings`, and
`datafusion_metrics` values.

`ExecutionOptions(settings={}, deadline=None)` is frozen and reusable; an
omitted settings argument creates an empty mapping.
`settings` must be a mapping whose complete object graph is strict JSON:
string keys; exact `None`, `bool`, `int`, finite `float`, `str`, `list`, and
`dict` values; integers in the inclusive range `-2**63 .. 2**64-1`; and no
cycles. The root settings mapping is depth 0, and every child value must be at
depth 32 or less. Calc Flow copies the caller's graph at construction and
returns fresh copies from the getter. `deadline` must be a timezone-aware UTC
`datetime` with a zero offset and is normalized without losing microseconds.

An expired or crossed deadline raises `calc_flow.CancelledError`. Native
deadline and cancellation checks are cooperative at safe execution
boundaries, with state rolled back before the plan is reusable. Cancelling the
Python task returned by `execute_async` raises `asyncio.CancelledError` only
after native cleanup. Python does not expose a native cancellation token.

### Runtime and UDFs

`Runtime` compiles strict project JSON, reports validation results, registers
trusted providers/scalar UDFs, returns the compatibility UDF catalog, and
exposes an immutable runtime-session capability snapshot through
`capabilities()`.
`register_scalar_udf` requires provider/name/version, exact input type names,
return type, volatility, and a vectorized callable.

`register_provider` and `_register_mapping_provider` accept keyword-only
`accepts_context=False`. A single provider registered with `register_provider`
receives one `Batch` as its first argument: `(batch, options)` by default or
`(batch, options, context)` when opted in. A mapping provider registered with
`_register_mapping_provider` receives a named input mapping:
`(inputs, options)` or `(inputs, options, context)`. The frozen,
engine-created `ProviderContext` exposes defensive `settings` and `deadline`
observations. Calc Flow does not inspect callback signatures or retry a call
with another arity.

Capability schema version 1 contains only frozen data:
`RuntimeSessionScope`, `OperatorCapability`, `UdfCapability`,
`ProviderCapability`, `ProviderPort`, `ProviderOptionsSchema`, and
`ProviderOption`. Its session ID is stable for one `Runtime`; its revision
advances once for each successful registry entry. Failed duplicates do not
advance it, while the two-entry NumPy and JAX helpers can expose a real partial
success. Snapshots already returned to callers do not change when later
registrations advance the revision.

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

| Method                 | Route                       | Purpose                                      |
| ---------------------- | --------------------------- | -------------------------------------------- |
| `GET`                  | `/catalog`                  | Compatibility UDF-only top-level array       |
| `GET`                  | `/capabilities`             | Runtime and preview-worker capabilities      |
| `GET`                  | `/schema/project`           | Rust-generated v2 project JSON Schema        |
| `GET`, `POST`          | `/projects`                 | List or create projects                      |
| `POST`                 | `/projects/import`          | Safely import JSON or YAML                   |
| `GET`, `PUT`, `DELETE` | `/projects/{id}`            | Read, replace, or delete a project           |
| `GET`                  | `/projects/{id}/export`     | Export canonical JSON or safe YAML           |
| `POST`                 | `/projects/{id}/validate`   | Validate and compile a stored graph          |
| `GET`, `DELETE`        | `/projects/{id}/checkpoint` | Inspect or reset recovery state              |
| `POST`                 | `/projects/{id}/runs`       | Start a bounded preview worker               |
| `GET`                  | `/runs/{id}`                | Read typed preview status/results            |
| `GET`                  | `/runs/{id}/events`         | Stream run events                            |
| `DELETE`               | `/runs/{id}`                | Cancel a managed preview                     |

`/capabilities` deliberately separates two scopes. `runtime` is the parent
session snapshot used for compilation. `preview.workerRegistrations` describes
whether a matching registration can be reconstructed as `serialized`,
`lazyBuiltin`, or `unavailable` in a spawned preview worker. Transportability
is not a promise that arbitrary project input is executable: normal compile,
Port, option, input-format, and resource-limit checks still apply.

When an older Studio server returns `404` for `/capabilities`, clients may use
`/catalog` only to populate a scalar-UDF picker. That fallback does not
discover providers, built-in Operators, portable Arrow types, preview limits,
worker reconstruction, or transportability; clients must not infer any of
those capabilities from it.

Only NumPy/JAX `expression@1` is eligible for lazy built-in reconstruction in
schema version 1. There is no lazy `table_matmul@1`; a parent registration may
therefore be compile-capable while preview reconstruction is unavailable.
Clients should disable only the unsupported preview action and keep project
editing and parent-runtime validation available.

Capability schema version 1 is closed. The browser decoder rejects an unknown
version or any extra field before React receives it. Validation and run
responses now use generated discriminated unions, which is an intentional
generated-client source break and requires backend/frontend deployment
coupling. The current compatibility gate covers a version-1 client against
newer or extended payloads and the new client against the old response shape.
An explicit version-2 decoder that can fall back to the version-1 fixture is
deferred until a real version-2 schema exists; no synthetic version-2
migration fixture is claimed in this release.

The checked contract is [web-ui/openapi.json](../web-ui/openapi.json).
`npm run sync:api` regenerates it and
[`schema.d.ts`](../web-ui/src/api/schema.d.ts) from the FastAPI application.

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
