# Calc Flow 4.0 API reference

Calc Flow has three supported surfaces:

| Surface          | Package or path           | Purpose                                     |
| ---------------- | ------------------------- | ------------------------------------------- |
| Rust core        | `calc-flow = "4.0.0"`     | Native batches, graphs, execution, recovery |
| Python binding   | `calc-flow==4.0.0`        | PyO3 engine access and Python integrations  |
| Local Studio API | `calc-flow-studio==4.0.0` | Loopback FastAPI service and React assets   |

For examples and lifecycle detail, see the [executable example guide](examples.md),
[Rust API](rust-api.md), [Python API](python-api.md), and
[continuous streaming guide](streaming-guide.md).

## Examples

Minimal end-to-end calculation (Python):

```python
import pyarrow as pa

from calc_flow import Batch, PipelineBuilder

batch = Batch.from_pyarrow(pa.table({"a": [1, 3], "b": [2, 4]}))
plan = (
    PipelineBuilder("totals").expression("calculate", "total = a + b").compile_batch()
)
result = plan.execute({"input": batch})

assert result.outputs["output"].to_pyarrow()["total"].to_pylist() == [3, 7]
```

The runnable inventories span both surfaces and share datasets and expressions:

- Python: [`examples/README.md`](../examples/README.md) — expression pipeline,
  SQL join, registered UDF, continuous execution and recovery, async batch
  execution, and NumPy arrays plus NumPy/JAX `pyarrow.Table` matrix multiplication.
- Rust: [`crates/calc-flow/examples/README.md`](../crates/calc-flow/examples/README.md)
  — expression, SQL, continuous lifecycle, event-time window, and schema tools.

Run all user examples with
`JAX_PLATFORMS=cpu uv run python scripts/run_examples.py`.

## Rust modules and exports

The `calc_flow` crate re-exports its supported public types from
[`lib.rs`](../crates/calc-flow/src/lib.rs).

| Area                | Primary APIs                                                                                                                                           |
| ------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------ |
| Data                | `Batch`, `BatchKind`, `BatchMetadata`, `TableBatch`                                                                                                    |
| Batch graph         | `PipelineBuilder`, `Edge`, `PortEndpoint`, `BatchExecutionPlan`                                                                                        |
| Stream plan         | `StreamExecutionPlan`, `StreamRequirements`, `DeliveryGuarantee`, `StreamRuntimeConfig`                                                                |
| Operator traits     | `Port`, `OperatorMetadata`, `NodeOperator`, `BatchOperator`, `StreamOperator`, `StreamOperatorLifecycle`, `OperatorStateSnapshot`                      |
| Built-in operators  | `ExpressionOperator`, `SqlOperator`, `RollingOperator`, `CrossSectionOperator`, `UnionOperator`, `WindowAggregateOperator`, `StreamJoinOperator`       |
| Window model        | `WindowSpec`, `WindowGeometry`, `AggregateSpec`, `AggregateFunction`, `MAX_WINDOW_OVERLAP`                                                             |
| Rolling model       | `RollingSpec`, `RollingOutputSpec`, `LatePolicySpec`, `LateErrorScope`, `RollingValuePolicy`                                                           |
| Cross-section model | `CrossSectionSpec`, `CrossSectionGroupingSpec`, `CrossSectionOutputSpec`, `CrossSectionValuePolicy`, `RankTieMethod`, `SortDirection`, `NullPlacement` |
| Stream join model   | `StreamJoinSpec`, `StreamJoinType`, `JoinTimeBounds`, `JoinStateLimits`, `StreamJoinStatus`                                                            |
| Execution           | `ExecutionOptions`, `RunResult`, `RunMetadata`, `NodeTiming`                                                                                           |
| Stream model        | `StreamMessage`, `StreamMessageKind`, `StreamJobContext`, `EventTime`, `Epoch`                                                                         |
| Stream channel      | `EdgeBudget`, `EnvelopeCost`, `ChannelMetrics`, `EdgeSender`, `EdgeReceiver`, `edge_channel`                                                           |
| State backend       | `StateBackend`, `StateLineageBackend`, `StateLineageKey`, `StateHandle`, `LocalStateBackend`                                                           |
| State manifest      | `CheckpointManifest`, `CheckpointManifestFields`, `ManifestExpectation`, `OperatorManifestEntry`, `RecoveryStatus`                                     |
| UDF/providers       | `UdfRegistry`, `UdfReference`, `ProviderRegistry`, `BatchOperatorFactory`, `StreamOperatorFactory`                                                     |
| Sources and sinks   | `StreamSource`, `StreamSink`, `TransactionalStreamSink`, `SourceBinding`, `SinkBinding`                                                                |
| Continuous runtime  | `StreamingRunner`, `StreamingJob`, `ManagedCheckpointRuntime`, `Cursor`, `SourceEvent`, `JobStatus`, `JobOutcome`                                      |
| Static inputs       | `StaticInputSpec`, `StaticInputDigest`, `StaticMutability`, `STATIC_INPUT_DIGEST_VERSION`, `StaticArraySnapshot`, `StaticArrayValues`                  |
| Projects            | `ProjectSpec`, `compile_project`, `validate_project`                                                                                                   |
| Persistence         | `FileProjectStore`, `LocalStateBackend`, `CheckpointManifest`                                                                                          |
| Errors              | `CalcFlowError`, `Result<T>`                                                                                                                           |

`compile_project` produces a `BatchExecutionPlan`. `compile_batch` and
`compile_stream` are the Rust graph-compilation entry points. A
`StreamExecutionPlan` is consumed by the public source-driven
`StreamingRunner`. The runner owns source and sink bindings plus a
`ManagedCheckpointRuntime`; `start(self)` consumes it and returns the sole
`StreamingJob` lifecycle owner. The complete SCE-13 crate-root export delta is
exactly `StaticArraySnapshot` and `StaticArrayValues`.
`Batch::static_array_snapshot()` is an explicit owned host-neutral copy for a
latched static array: its backend, dtype, shape, optional full null bitmap, and
compact scalar carrier are available through read-only accessors. The snapshot
and value enum are non-exhaustive and intentionally provide no `Clone`,
payload-bearing `Debug`, serde, or mutation surface.

Python static placement creates one transient `O(n)` snapshot clone per placed
static input inside a blocking worker. `static_placement_bytes` is the logical
provider-transfer count — dtype width multiplied by logical element count —
reported on first placement and zero for cached later micro-batches. It does
not measure peak memory, process RSS, or the internal snapshot clone; the
engine latch, snapshot carriers, Python host list, NumPy storage, and a
provider-owned JAX result may coexist during first placement. The v2
source/sink traits, micro-batch runner, push runner, and public
checkpoint-document store are not exported.

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

| Member                                      | Contract                                                |
| ------------------------------------------- | ------------------------------------------------------- |
| `Batch.from_pyarrow(table, metadata=None)`  | Own an Arrow C Stream as an immutable table batch       |
| `Batch.from_array(array, backend=..., ...)` | Own a read-only explicitly named array-provider payload |
| `to_pyarrow()`                              | Return the table payload or reject a non-table batch    |
| `array` / `backend`                         | Return array-provider data or reject a table batch      |
| `kind` / `num_rows` / `metadata`            | Return defensive batch observations                     |

### Graph and execution

`PipelineBuilder(name)` supports:

- `expression(name, expression, *, select=(), filter=None, udfs=())`;
- `sql(name, query, *, aliases=("input",), udfs=())`;
- `external(node_id, provider, name, version, options)`;
- `table_matmul(node_id, *, backend: Literal["numpy", "jax"],
  columns: Sequence[str])`;
- `stream_join(name, *, left_schema, right_schema, left_keys, right_keys,
  left_event_time, right_event_time, bounds, limits, left_prefix="left",
  right_prefix="right")`;
- `connect(source_node, target_node, *, source_port="output",
  target_port="input")`;
- `compile_batch(runtime=None) -> BatchExecutionPlan`;
- `compile_stream(*, requirements=None, runtime=None) -> StreamExecutionPlan`.

`BatchExecutionPlan` (also exported as `ExecutionPlan`) exposes immutable
`name` and `fingerprint`. `execute(inputs, *, options=None)` is blocking and
rejects a running event loop before validating inputs or options;
`execute_async(inputs, *, options=None)` is the asynchronous form. In both
signatures, `options` is keyword-only and `None` preserves the existing
default execution behavior. `snapshot[_async]`, `restore[_async]`, and
`reset[_async]` provide the batch-plan state lifecycle.

`StreamExecutionPlan` exposes immutable `name`, `fingerprint`, `requirements`,
`source_binding_ids`, `static_input_ids`, and `sink_binding_ids`; it is
consumed by `StreamingRunner` rather than executed directly.

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

Capability schema version 2 contains only frozen data:
`RuntimeSessionScope`, `OperatorCapability`, `UdfCapability`,
`ProviderCapability`, `ProviderPort`, `ProviderOptionsSchema`,
`ProviderOption`, `CapabilityRule`, and `ProviderArrayRules`. Operator and
provider entries carry the lifecycle vocabulary — modes, finality,
statefulness, micro-batch invariance, watermark requirement, checkpoint
support and state version, determinism, and replay safety — validated
fail-closed against closed vocabularies; see the
[Python API guide](python-api.md) for the full contract. Its session ID is
stable for one `Runtime`; its revision advances once for each successful
registry entry. Failed duplicates do not advance it, while the three-entry
NumPy and JAX helpers can expose a real partial success. Snapshots already
returned to callers do not change when later registrations advance the
revision.

`register_numpy(runtime)` installs `numpy:expression@1`,
`numpy:table_matmul@1`, and `numpy:symbolic_matrix@1`;
`register_jax(runtime)` installs the same three identities under `jax`.
`table_matmul@1` accepts an Arrow table named `table` and same-backend array
weights named `weights`, then produces an array output. After input `Batch`
construction it makes no redundant execution copies: NumPy uses one dense
table allocation and one result; JAX permits one host staging buffer, one
device table buffer, and one device result. JAX performs no result-to-host round trip during operator execution. These are execution ceilings, not
end-to-end zero-copy claims. `symbolic_matrix@1` is the fused
table-to-array-to-table provider described in the
[symbolic matrix compilation guide](python-api.md#symbolic-matrix-compilation).

### Projects and stores

`ProjectDocument` is the strict Pydantic root model for format v3.
`project_json_schema()` returns the Rust-generated schema and
`validate_project_json()` returns canonical validated JSON.

Project v3 carries an explicit batch or stream runtime. Stream documents bind
graph endpoints to exact connector and format identities, refer to named
secrets, configure watermarks and managed state, and request delivery per
output as best-effort, at-least-once, or exactly-once. Trusted connector
factories and the secret resolver are runtime
registrations and are never serialized into the project.

`FileProjectStore` provides async `create`, `put`, `get`, `list`, and
`delete` plus explicit `*_blocking` forms. Import/export helpers accept safe
JSON or YAML. Continuous checkpoints are owned by
`ManagedCheckpointRuntime`; applications do not read or write checkpoint
documents through a public store.

### Continuous runner

`StreamingRunner(stream_plan, sources, sinks, checkpoints, *, config=None,
static_inputs=None)` owns async `StreamSource` and sink connectors.
`start_async()` consumes the
runner and returns a `StreamingJob`; use `trigger_checkpoint_async()`,
`shutdown_async()`, `cancel_async()`, or `wait_async()` to drive the job.
Guarded blocking forms are available outside an event loop. Connector methods
must be declared with `async def`. `static_inputs` supplies the immutable
`Batch` side values a plan declares; see
[static inputs](streaming-guide.md#static-inputs).

Blocking `start()` owns a dedicated event-loop thread for the lifetime of its
job. Blocking terminal calls execute on that loop, and async terminal calls
from another loop are marshalled to it. Terminal `shutdown`, `cancel`, and
`wait`, whether async or blocking, release connector roots and stop and join
the owned thread; dropping the job schedules cancellation and settlement
before reclaiming it. Observer cancellation during terminal cleanup cannot
strand the thread. Once the native terminal outcome has linearized, it wins
over cancellation that arrives while the thread is being reclaimed.

### Symbolic declarations

`calc_flow.symbolic` is the pure declaration surface: immutable expressions,
programs, static analysis, and row-local compilation with no data execution
path. Every expression, feature, program, and analysis result is immutable;
constructors copy caller-owned sequences and mappings.

| Member                                                                        | Contract                                                                                                             |
| ----------------------------------------------------------------------------- | -------------------------------------------------------------------------------------------------------------------- |
| `Expr` / `ColumnExpr` / `ArrayExpr` / `TableExpr` / `Parameter`               | Immutable typed declaration values with v1 digests                                                                   |
| `table_input(name, *, schema, entity_by=(), event_time=None, sequence_by=())` | Declare one named table input                                                                                        |
| `parameter(name, *, kind=...)`                                                | Declare one named static table or array input                                                                        |
| `Field(name, data_type, nullable=True)`                                       | One exact table field declaration                                                                                    |
| `rows(size)` / `duration(micros)`                                             | Row-count and exact-microsecond rolling frames                                                                       |
| `exact_time(...)` / `event_time_bucket(...)`                                  | Cross-section group declarations                                                                                     |
| `row` / `ts` / `cs` / `table` / `linalg` / `window`                           | Namespace functions over expressions                                                                                 |
| `FeatureSet(features=())` / `.with_feature(name, value)`                      | Ordered uniquely named column expressions                                                                            |
| `TableExpr.with_columns(features)`                                            | Append a feature set as derived columns                                                                              |
| `Program(name, *, inputs=(), outputs=())`                                     | Declared inputs and outputs with the runtime-independent v1 fingerprint                                              |
| `Program.analyze(runtime, *, mode)` / `.explain(runtime, *, mode)`            | Static analysis and deterministic fact rendering                                                                     |
| `Program.compile_batch(runtime)` / `.compile_stream(runtime, *, ...)`         | Lower supported row-local, rolling, cross-section, and symbolic matrix declarations to a strict project-v3 plan      |
| `AnalysisIssue` / `AnalysisResult`                                            | Immutable findings with stable output/input-rooted paths                                                             |

Structural identity uses `identical()`; public comparison operators build
symbolic expressions, and converting one to `bool` fails. See the
[Python API guide](python-api.md) for the declaration, fingerprint,
analysis, and compilation contract.

## Local HTTP API

The separate Studio service exposes its supported API under `/api/v3`.

| Method                 | Route                     | Purpose                                           |
| ---------------------- | ------------------------- | ------------------------------------------------- |
| `GET`                  | `/catalog`                | UDF-only top-level array                          |
| `GET`                  | `/capabilities`           | Runtime, connector, and worker capabilities       |
| `GET`                  | `/schema/project`         | Rust-generated v3 project JSON Schema             |
| `GET`, `POST`          | `/projects`               | List or create projects                           |
| `POST`                 | `/projects/import`        | Safely import JSON or YAML                        |
| `GET`, `PUT`, `DELETE` | `/projects/{id}`          | Read, replace, or delete a project                |
| `GET`                  | `/projects/{id}/export`   | Export canonical JSON or safe YAML                |
| `POST`                 | `/projects/{id}/validate` | Validate and compile a stored graph               |
| `POST`, `GET`          | `/jobs`                   | Start a continuous job or list owned jobs         |
| `GET`                  | `/jobs/{id}`              | Read typed job lifecycle status                   |
| `GET`                  | `/jobs/{id}/events`       | Resume-safe job event SSE                         |
| `POST`                 | `/jobs/{id}/checkpoint`   | Trigger one durable checkpoint                    |
| `POST`                 | `/jobs/{id}/shutdown`     | Request graceful terminal checkpoint and shutdown |
| `POST`                 | `/jobs/{id}/cancel`       | Cancel and settle a job                           |
| `GET`                  | `/resource-limits`        | Read enforced continuous-job resource bounds      |

`/capabilities` deliberately separates two scopes. `runtime` is the parent
session snapshot used for compilation. `preview.workerRegistrations` describes
whether a matching registration can be reconstructed as `serialized`,
`lazyBuiltin`, or `unavailable` in a spawned worker. Transportability
is not a promise that arbitrary project input is executable: normal compile,
Port, option, input-format, and resource-limit checks still apply.

Only NumPy/JAX `expression@1` is eligible for lazy built-in reconstruction.
There is no lazy `table_matmul@1`; a parent registration may therefore be
compile-capable while worker reconstruction is unavailable. Clients should
disable only the unsupported job action and keep project editing and
parent-runtime validation available.

The response envelope carries one schema version: `schemaVersion` is `2`, and
the nested `runtime` object omits its own version field. The browser decoder
rejects a response whose `schemaVersion` is not `2` or that carries any extra
field before React receives it. Unknown capability-rule identities, unknown
lifecycle vocabulary, and inconsistent `stateVersion`/`stateful` combinations
are rejected by the backend response models as well as the decoder.
Validation and job responses use generated discriminated unions, so
backend and frontend must be deployed from the same generated contract. SSE
event payloads are not part of that generated contract: the events route
serializes the backend event model directly, and the frontend maintains the
corresponding event type by hand.

Project writes that fail validation answer `422` with a structured envelope on
`POST /projects`, `POST /projects/import`, and `PUT /projects/{id}`: the
`detail` field is either an invalid `ValidationReport`
(`kind: "invalid"`, `issues` of `path`/`code`/`message`, `fingerprint: null`)
or the standard request-validation error list. Malformed stream Join input
carries the stable issue codes `unsupported_join_type`, `invalid_time_bound`,
`invalid_join_limit`, `invalid_join_keys`, `incompatible_key_type`,
`invalid_event_time`, and `invalid_output_prefix`.

REST payloads cannot carry live values, so `POST /jobs` fails closed for a
stored project that declares static inputs: the response is `422`, its
`detail` names the first `static_inputs.{name}` as unresolvable over REST, and
no run, handle, or worker is created. Supply static values through the Python
runtime instead.

The checked contract is [web-ui/openapi.json](../web-ui/openapi.json).
`npm run sync:api` regenerates it and
[`schema.d.ts`](../web-ui/src/api/schema.d.ts) from the FastAPI application.

## Error categories

Rust returns `CalcFlowError` variants. Python exposes the stable hierarchy:
`CalcFlowError`, `ConfigError`, `CompileError`, `ExecutionError`,
`ProviderError`, `CheckpointError`, `CancelledError`,
`StreamingRuntimeError`, and `CheckpointPublicationUnknownError`.

Invalid user documents and graph definitions are configuration/compile errors;
execution, callbacks, source/sink failures, cancellation, and checkpoint
storage preserve their more specific categories.

`CalcFlowError::TaskPanicked { task_id, message }` reports a captured private
runtime task panic. Internally captured panic text is valid UTF-8 and at most
1,024 bytes including the ellipsis. Python maps an unexpected native panic
through its existing `ExecutionError` category. Public continuous lifecycle
failures use payload-safe `StreamingRuntimeError` projections; indeterminate
manifest publication uses `CheckpointPublicationUnknownError`.

## Version and compatibility

The Rust crate, Python binding, Studio package, and frontend are versioned
`4.0.0`. Project format version `3` and checkpoint-manifest version `3` are
separate protocol values from the package version.

Calc Flow 4.0 does not load project-v2 documents or expose Studio `/api/v2`.
See the [v2-to-v3 migration guide](migration-v2-to-v3.md) for the required
migration boundary.
