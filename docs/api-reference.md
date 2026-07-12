# Calc Flow v0.2 API reference

This reference summarizes Calc Flow's supported Python and local HTTP surfaces.
All table values cross public runtime boundaries inside immutable `Batch`
objects and all table calculation uses Apache DataFusion.

## Batch and metadata

Import these names from `calc_flow`.

### `Batch`

| Member                                               | Contract                                                                         |
| ---------------------------------------------------- | -------------------------------------------------------------------------------- |
| `Batch.table(data, metadata=None)`                   | Accept a `pyarrow.Table` or `pyarrow.RecordBatch` and create a table batch.       |
| `Batch.from_tabular_protocol(data, metadata=None)`   | Normalize an Arrow C Stream or DataFrame Interchange producer to a table batch.  |
| `Batch.array(data, metadata=None)`                   | Create an array batch from an object implementing `__array_namespace__`.         |
| `table_payload` / `array_payload`                    | Return the kind-specific payload or reject the wrong kind.                       |
| `schema`                                             | Return the Arrow schema for table batches.                                       |
| `num_rows`                                           | Return the table row count or leading array dimension.                           |
| `with_payload(payload)`                              | Return a new batch while preserving metadata.                                    |
| `with_metadata(**changes)`                           | Return a new batch with replaced immutable metadata.                             |

`BatchMetadata` carries `batch_id`, sequence, source ID, replay cursor, event
time, watermark, and immutable JSON-compatible attributes. `to_dict()` returns
a JSON-compatible document.

## DataFusion engines

Import engine APIs from `calc_flow.engine`.

### `DataFusionEngine`

`evaluate(expression, batch)` evaluates a SQL expression or assignment over
one table batch. `sql(query, tables)` executes one read-only `SELECT` or CTE over
named table batches. Both methods create and close a run-scoped session.

### `DataFusionRuntime`

`DataFusionRuntime(config=None, *, udfs=())` owns a reusable DataFusion
`SessionContext`. Use `evaluate()` and `sql()` as above, inspect `metrics`, and
call `close()` or use it as a context manager.

Every `DataFusionQueryMetrics` value contains:

- optional node ID;
- planning and execution nanoseconds;
- output rows;
- optimized logical plan;
- physical plan.

`DataFusionConfig` controls execution `batch_size`, `target_partitions`, and
repartitioning for aggregations, joins, sorts, and windows. These values do not
control source batch formation.

### Optional array engines

`NumpyEngine` and `JaxEngine` accept array batches. Their `evaluate()` methods
interpret an allowlisted expression AST. They also expose `add`, `subtract`,
`multiply`, `divide`, `matmul`, `sum`, `mean`, `max`, `min`, `transpose`, and
`reshape` operations.

## Operators and graph execution

| API                                               | Contract                                                                       |
| ------------------------------------------------- | ------------------------------------------------------------------------------ |
| `Port(name, kind, required=True, schema=None)`     | Declare a typed operator boundary.                                             |
| `Operator.process(inputs, context)`               | Map named batch inputs to named batch outputs.                                 |
| `StatelessOperator(name, fn, ...)`                | Wrap a pure process function.                                                  |
| `StatefulOperator`                                | Provide deep-copying `snapshot`, `restore`, and `reset` behavior.              |
| `ExpressionOperator`                              | Execute one DataFusion calculation, projection, or filter.                     |
| `SqlOperator`                                     | Execute multi-input DataFusion SQL.                                            |
| `ArrayExpressionOperator`                         | Execute an allowlisted NumPy or JAX expression.                                |

Build a graph with `Pipeline.add_node()` and `Pipeline.connect()`. `then()` and
`add()` are linear single-port conveniences. `compile()` validates the graph and
returns an immutable `ExecutionPlan`.

`ExecutionPlan.execute(inputs, *, cancellation=None, deadline=None,
settings=None)` returns a `RunResult`. Results expose named terminal `outputs`,
node timings, DataFusion metrics, warnings, and run metadata. `output` is a
convenience property only when the graph has exactly one terminal output.

`RunContext` gives operators the run/node IDs, shared DataFusion runtime,
selected UDF registry, cancellation/deadline controls, and read-only settings.

## Registered UDFs

`UdfRegistry.datafusion_scalar(...)` registers a trusted vectorized Arrow
implementation with a stable name/version, input and return fields, volatility,
and description. `UdfRegistry.array(...)` registers a vectorized Array API
implementation with an argument count.

Operators reference implementations only through `UdfReference(name,
version)`. Compile the pipeline after all required implementations are
registered. `catalog()` returns JSON-compatible discovery metadata without
source code, import paths, or callable objects.

## Sources, sinks, and recovery

`Source.read(cursor=None)` yields formed `Batch` objects. `BatchingSource`
groups an in-memory record sequence by `max_rows` and `max_bytes`. `Sink.write`
accepts one output batch.

`MicroBatchRunner.run(source, sink=None)` yields every `RunResult` and commits a
checkpoint after successful sink delivery at the configured interval.
`StreamingRunner.step(batch, sink=None)` provides the equivalent one-batch
contract. Both use at-least-once delivery.

`FileCheckpointStore(directory)` atomically persists versioned `Checkpoint`
documents. Recovery validates the pipeline fingerprint before restoring node
state and the source cursor.

## Project configuration

`ProjectConfig` is the canonical, strict, data-only project model. It contains
`PipelineConfig`, `NodeConfig`, `EdgeConfig`, `PortConfig`, `DataFusionConfig`,
`DataSourceConfig`, and `RunOptions` values.

- `compile_project(project, udf_registry=None)` returns an `ExecutionPlan`.
- `validate_project(project, udf_registry=None)` returns a `ValidationReport`.
- `FileProjectStore(directory)` provides atomic `list`, `get`, `create`, `put`,
  and `delete` operations.

Canonical persistence is sorted formatted JSON. YAML uses safe import/export
only. Unknown fields and executable configuration values are rejected.

## Local HTTP API

The separate `calc-flow-studio` workspace package owns the `calc-flow-web`
command. The unauthenticated service binds only to loopback and exposes these
routes under `/api/v1`:

| Method                 | Route                       | Purpose                                                    |
| ---------------------- | --------------------------- | ---------------------------------------------------------- |
| `GET`                  | `/catalog`                  | Operators, UDF metadata, Arrow types, and preview limits.  |
| `GET`                  | `/schema/project`           | JSON Schema for `ProjectConfig`.                           |
| `GET`, `POST`          | `/projects`                 | List projects or create one with a server-assigned ID.     |
| `POST`                 | `/projects/import`          | Safely import JSON or YAML.                                |
| `GET`, `PUT`, `DELETE` | `/projects/{id}`            | Read, replace, or delete a project.                        |
| `GET`                  | `/projects/{id}/export`     | Export canonical JSON or safe YAML.                        |
| `POST`                 | `/projects/{id}/validate`   | Compile and validate a stored graph.                       |
| `GET`, `DELETE`        | `/projects/{id}/checkpoint` | Inspect or reset runner recovery state.                    |
| `POST`                 | `/projects/{id}/runs`       | Start a bounded preview worker.                            |
| `GET`                  | `/runs/{id}`                | Read preview status and results.                           |
| `GET`                  | `/runs/{id}/events`         | Stream run events.                                         |
| `DELETE`               | `/runs/{id}`                | Cancel a managed preview.                                  |

The checked-in OpenAPI document is
[`web-ui/openapi.json`](../web-ui/openapi.json). It is the source for generated
TypeScript request and response types.
