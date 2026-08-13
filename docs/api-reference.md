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
  SQL join, registered UDF, continuous execution, async batch execution, and NumPy
  arrays plus NumPy/JAX `pyarrow.Table` matrix multiplication.
- Rust: [`crates/calc-flow/examples/README.md`](../crates/calc-flow/examples/README.md)
  — `expression_pipeline.rs`, `sql_join.rs`, `continuous_runtime.rs`, and
  `export_schema.rs`.

## Rust modules and exports

The `calc_flow` crate re-exports its supported public types from
[`lib.rs`](../crates/calc-flow/src/lib.rs).

| Area               | Primary APIs                                                                                                                        |
| ------------------ | ----------------------------------------------------------------------------------------------------------------------------------- |
| Data               | `Batch`, `BatchKind`, `BatchMetadata`, `TableBatch`                                                                                 |
| Batch graph        | `PipelineBuilder`, `Edge`, `PortEndpoint`, `BatchExecutionPlan`                                                                     |
| Stream plan        | `StreamExecutionPlan`, `StreamRequirements`, `DeliveryGuarantee`, `StreamRuntimeConfig`                                             |
| Operator traits    | `Port`, `OperatorMetadata`, `NodeOperator`, `BatchOperator`, `StreamOperator`, `OperatorStateSnapshot`                              |
| Built-in operators | `ExpressionOperator`, `SqlOperator`, `UnionOperator`, `WindowAggregateOperator`                                                     |
| Window model       | `WindowSpec`, `WindowGeometry`, `AggregateSpec`, `AggregateFunction`, `MAX_WINDOW_OVERLAP`                                          |
| Execution          | `ExecutionOptions`, `RunResult`, `RunMetadata`, `NodeTiming`                                                                        |
| Stream model       | `StreamMessage`, `StreamMessageKind`, `StreamJobContext`, `EventTime`, `Epoch`                                                      |
| Stream channel     | `EdgeBudget`, `EnvelopeCost`, `ChannelMetrics`, `EdgeSender`, `EdgeReceiver`, `edge_channel`                                        |
| State backend      | `StateBackend`, `StateLineageBackend`, `StateLineageKey`, `StateHandle`, `LocalStateBackend`                                        |
| State manifest     | `CheckpointManifest`, `CheckpointManifestFields`, `ManifestExpectation`, `OperatorManifestEntry`, `RecoveryStatus`                  |
| UDF/providers      | `UdfRegistry`, `UdfReference`, `ProviderRegistry`, `BatchOperatorFactory`, `StreamOperatorFactory`                                  |
| Sources and sinks  | `StreamSource`, `StreamSink`, `TransactionalStreamSink`, `SourceBinding`, `SinkBinding`                                              |
| Continuous runtime | `StreamingRunner`, `StreamingJob`, `ManagedCheckpointRuntime`, `Cursor`, `SourceEvent`, `JobStatus`, `JobOutcome`                    |
| Projects           | `ProjectSpec`, `compile_project`, `validate_project`                                                                                |
| Persistence        | `FileProjectStore`, `LocalStateBackend`, `CheckpointManifest`                                                                        |
| Errors             | `CalcFlowError`, `Result<T>`                                                                                                        |

`compile_project` produces a `BatchExecutionPlan`. `compile_batch` and
`compile_stream` are the Rust graph-compilation entry points. A
`StreamExecutionPlan` is consumed by the public source-driven
`StreamingRunner`. The runner owns source and sink bindings plus a
`ManagedCheckpointRuntime`; `start(self)` consumes it and returns the sole
`StreamingJob` lifecycle owner. The v2 source/sink traits, micro-batch runner,
push runner, and public checkpoint-document store are not exported.

`EdgeBudget::new(R, B)` keeps its two-field public shape and caps queued
envelopes and charged rows independently at `R`, plus charged bytes at `B`.
Direct `edge_channel` callers must choose
`R >= max(required_row_limit, required_simultaneous_messages)`.

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
loop before validating inputs or options.
`execute_async(inputs, *, options=None)` is the asynchronous form.
In both signatures, `options` is keyword-only and `None` preserves the
existing default execution behavior. `snapshot[_async]`, `restore[_async]`,
and `reset[_async]` provide the plan-state lifecycle.

`RunResult` exposes defensive `outputs`, `metadata`, `node_timings`, and
`datafusion_metrics` values.

`ExecutionOptions` and `ProviderContext` are frozen native classes exported
from the package root. The `ExecutionOptions(settings={}, deadline=None)`
constructor accepts its two arguments positionally or by keyword; this is
separate from the keyword-only `options=` plan parameter. An omitted settings
argument or explicit `settings=None` creates an empty mapping. Every object
position may use a `collections.abc.Mapping`; Calc Flow calls its `items()`
once, consumes that iterator once, and copies it to a built-in `dict`. Mapping
subclasses are accepted, while sequence containers must be exact built-in
`list` values. Keys and scalar leaves must be exact built-in `str`, `None`,
`bool`, `int`, or finite `float` values. Integers must be in the inclusive
range `-2**63 .. 2**64 - 1`. Duplicate or surrogate-containing keys/strings,
cycles, unsupported subclasses, and values deeper than 32 are rejected with
stable redacted paths.

Construction deep-copies the complete accepted graph, including shared nested
aliases, and neither retains nor mutates caller containers. Every
`options.settings` read returns another deep `dict`/`list` copy. Exceptions
from caller mappings are replaced without retaining their value, type,
message, or traceback. `deadline` accepts `None` or any valid timezone-aware
`datetime`. Accepted offsets are normalized to `datetime.UTC` without losing
microseconds; naive, invalid, and out-of-range UTC conversions are rejected
with fixed redacted errors.

An already expired or crossed deadline raises `calc_flow.CancelledError`
after transactional rollback. Cancelling a still-pending Python task returned
by `execute_async` instead raises `asyncio.CancelledError`; if native
execution is terminal when the cancellation handler starts, its result or
exception wins. Otherwise one native cancellation request wins, even if the
Python task is cancelled repeatedly while cleanup is held. After that
request, cancellation keeps precedence: a native failure observed during the
drain is retrieved and discarded, never re-raised over the caller's
`asyncio.CancelledError`. Awaited task
cancellation waits for the native operation and its cleanup, so no run-owned
work continues detached and the plan recovers before its next public
operation.

Both deadline and task cancellation are cooperative at safe execution
boundaries. They do not preempt a Python callback, DataFusion query, or other
non-cooperative operation already in progress; cleanup continues after that
operation yields. Every execution receives independent native cancellation
state, but the public Python API does not expose the native cancellation
token.

An absolute deadline continues to elapse while a run waits behind another
execution of the same plan. Cancelling such a queued run does not cancel the
active run or create partial plan state. Once a deadline or accepted task
cancellation is observed at a post-provider boundary, cancellation wins over
a provider error from that operation; recovery, input, snapshot, and
transaction-marker failures that occur before the first deadline check retain
their existing precedence.

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
engine-created `ProviderContext` exposes the authoritative run `settings` and
normalized `deadline`; each settings read is a fresh deep copy. This run
context is separate from the compile-time provider `options` mapping. Calc
Flow does not inspect callback signatures or retry a call with another arity.
The default false flag preserves all existing provider registrations, while
the true form opts into the additive three-argument ABI.

These execution changes are additive. Existing `execute(inputs)`,
`execute_async(inputs)`, and two-argument providers retain their behavior.
Execution settings and deadlines are per-run values; provider-context opt-in
belongs to the runtime registration. Neither changes project or checkpoint
formats, fingerprints, Studio REST/OpenAPI, or capability schemas.

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
JSON or YAML. Continuous checkpoints are owned by
`ManagedCheckpointRuntime`; applications do not read or write checkpoint
documents through a public store.

### Continuous runner

`StreamingRunner(stream_plan, sources, sinks, checkpoints, *, config=None)`
owns async `StreamSource` and sink connectors. `start_async()` consumes the
runner and returns a `StreamingJob`; use `trigger_checkpoint_async()`,
`shutdown_async()`, `cancel_async()`, or `wait_async()` to drive the job.
Guarded blocking forms are available outside an event loop. Connector methods
must be declared with `async def`.

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

`CalcFlowError::TaskPanicked { task_id, message }` reports a captured private
runtime task panic. Internally captured panic text is valid UTF-8 and at most
1,024 bytes including the ellipsis. Python maps an unexpected native panic
through its existing `ExecutionError` category; the private source-driven
continuous runtime adds no Python API.

## Version and compatibility

The Rust crate, Python binding, Studio package, and frontend are versioned
`2.0.0`. Project format version `2` and checkpoint format versioning are
separate protocol values.

Calc Flow 2.0 does not load v1 projects or checkpoints. See
[the release guide](v2-release.md) for the required migration boundary.
