# Calc Flow

## Introduction

Calc Flow is a stateful calculation library for micro-batch and streaming data.
It separates the in-memory data format from execution while keeping each data
path explicit:

* tabular data uses Apache Arrow and Apache DataFusion;
* array data uses Python Array API-compatible objects with an explicitly chosen
  backend.

## Batch data contract

Every pipeline item is an immutable `Batch` envelope. A batch contains:

* an Arrow table or Array API payload;
* a stable batch identifier;
* optional source, cursor, sequence, event-time, and watermark metadata;
* immutable JSON-compatible attributes.

`pyarrow.RecordBatch` inputs are wrapped as a single `pyarrow.Table` without
copying their Arrow buffers. Tabular objects that expose the Arrow C Stream or
DataFrame Interchange protocols can also be normalized to Arrow. Calc Flow does
not depend on or expose APIs for a particular dataframe library.

Array inputs must expose `__array_namespace__`. Calc Flow does not implicitly
move data between array backends.

## Calculation modes

Calc Flow supports:

* micro-batch processing over a replayable source of formed batches;
* streaming processing of one batch per `step()` call.

Both modes return `RunResult` objects containing named outputs and execution
observations. Synchronous source iteration and sink writes provide backpressure.

## DAG execution

An operator declares named input and output `Port` objects and implements
`process(inputs, context)`. Ports declare table or array batch kind and may
declare an exact Arrow schema. `Pipeline.compile()` validates endpoints,
single-writer input ports, kinds, schemas, and acyclic topology before producing
an immutable `ExecutionPlan`.

Unconnected input ports become named graph inputs. Unconnected output ports
become named graph outputs. Fan-out reuses an immutable batch; fan-in operators
such as `SqlOperator` receive multiple named batches. `then()` and `add()` are
conveniences for single-input, single-output chains.

Each execution creates a `RunContext` with a shared run-scoped DataFusion
session, cancellation and deadline checks, read-only settings, and identifiers.
The result records per-node row counts and timings plus DataFusion logical and
physical plans, planning time, and execution time.

## Table calculations

Apache DataFusion is the sole table query and calculation engine. It accepts
Arrow-backed batch inputs and returns Arrow-backed batch outputs.

Supported table operations include:

* calculated columns and projections;
* filters;
* joins across named input batches;
* aggregation, ordering, and SQL expressions;
* a single `SELECT` statement or common table expression.

Calc Flow does not include pandas or Polars calculation engines. Objects
produced by external libraries are accepted only through standard Arrow
interchange protocols.

`ExpressionOperator` handles projections, calculated expressions, filters, and
built-in functions. `SqlOperator` exposes named input aliases for joins,
aggregation, windows, sorting, and general `SELECT` queries. DDL, DML, utility
statements, and multi-statement SQL are rejected.

## Registered UDFs

Use DataFusion built-ins and Arrow compute functions when possible. Custom
table functions are trusted DataFusion scalar UDFs installed through an
application-owned `UdfRegistry`. A scalar UDF declares:

* a lowercase SQL name and stable version;
* named Arrow input fields, which form its parameter schema;
* an Arrow return field;
* immutable, stable, or volatile behavior;
* a description and vectorized implementation.

Operators explicitly list `UdfReference(name, version)` values. Compilation
rejects unknown versions and conflicting versions of the same DataFusion SQL
name. A run-scoped registry snapshot prevents later registry mutations from
changing a compiled plan. Only selected functions are registered in the run's
DataFusion session.

Scalar implementations receive Arrow arrays and execute over DataFusion record
batches. Calc Flow checks input type/nullability, return type/nullability, and
output length. Contract violations stop the current node and prevent downstream
execution.

Array UDFs are similarly trusted and registered by name, version, and argument
count. The restricted array AST permits direct calls only to UDFs explicitly
selected for that node. Returned objects must implement the Array API and must
remain in the active NumPy or JAX backend.

Registry catalogs are JSON-compatible metadata. Serializable references contain
only a name and version; pipeline or web configuration cannot contain inline
source, callable objects, or Python import paths. Aggregate, window, table,
stateful, and row-object UDFs are outside the v0.2 contract.

## Array calculations

NumPy and JAX are optional Array API backends. Array engines provide explicit
element-wise, matrix, reduction, and shape operations as well as a restricted
expression evaluator.

Expression evaluation supports approved arithmetic, comparisons, indexing,
slicing, and `xp` namespace calls. It rejects imports, arbitrary attribute
access, comprehensions, lambdas, and builtin calls.

## State and recovery

Stateful operators expose snapshot, restore, and reset operations. A `Source`
replays from a JSON-compatible cursor, while a `Sink` accepts one output batch.
`BatchingSource` groups an in-memory record sequence using `max_rows` and
`max_bytes` limits.

`FileCheckpointStore` writes versioned JSON atomically. Each checkpoint contains
the compiled pipeline fingerprint, source cursor and sequence, JSON-compatible
operator state keyed by node ID, and creation time. A stale fingerprint is
rejected until the checkpoint is reset or migrated.

Runners write configured sinks before committing a checkpoint. If a later sink
or checkpoint write fails, in-memory operator state rolls back. A sink that
accepted a result before another sink failed may receive it again during
recovery; delivery is therefore at-least-once.

## Versioned project configuration

The Python graph and browser use the same strict `ProjectConfig` model. Its
format version is independent of package version and currently equals `"1"`.
It contains a `PipelineConfig`, node and edge definitions, named typed ports,
DataFusion execution settings, UDF references, saved sample sources, and run
limits.

Node configuration is mode-specific. Table nodes are either expression or SQL
nodes and never expose a backend selector. Array nodes explicitly select NumPy
or JAX. Configurations contain UDF names and versions but never implementations.
Unknown properties are rejected rather than ignored.

`FileProjectStore` persists canonical, sorted, formatted JSON using an atomic
temporary-file replacement. Project IDs are hashed into filenames. YAML uses a
safe parser and is import/export only; it is never the canonical stored form.

## Local API and web UI

The optional `web` extra provides a FastAPI service under `/api/v1`. It exposes
the operator/UDF catalog, project CRUD and import/export, graph validation,
preview runs, event streams, cancellation, and runner-checkpoint inspection and
reset. The service binds only to loopback and is intentionally single-user and
unauthenticated.

Preview inputs accept inline records, CSV, JSON/NDJSON, and base64 Arrow IPC.
They are parsed directly into Arrow without a dataframe library. The API checks
combined byte and row limits before starting a run. Managed worker processes
enforce deadlines, CPU limits, resident-memory monitoring, bounded output rows,
and explicit termination on cancellation.

The React/TypeScript/Vite client uses React Flow for named-port DAG editing and
generated OpenAPI types for API contracts. It provides DataFusion expression
and SQL editors, schema forms, UDF discovery, input upload, validation, preview
tables, node timings, logical/physical execution plans, runner recovery
controls, and local comparison of `pytest-benchmark` JSON reports. Preview runs
remain stateless and do not create runner checkpoints.

## Design requirements

* Arrow is the canonical in-memory tabular format.
* DataFusion performs all table queries and calculations.
* Python Array API objects remain owned by their selected array backend.
* Batch metadata is immutable and JSON-compatible.
* Pipelines reject raw table and array inputs outside a `Batch` envelope.
* Optional array packages are not required for table-only installations.
* The Phase 5 benchmark harness and informational CI reports are implemented;
  stable-runner baseline sampling remains in
  `design/v0.2-refactor-plan.md`.
