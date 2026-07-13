# Python v1 final API

Python v1 is frozen at the `v1-python-final` tag. This document and the
committed semantic corpus are behavioral references for the Python v1
implementation; they are not a promise that Rust v2 will preserve Python API
compatibility.

The names below are every public symbol exported by `calc_flow.__all__` at the
freeze. The responsibility groups describe their v1 role; detailed member and
method contracts remain in [the API reference](api-reference.md).

The deterministic corpus under `tests/fixtures/v1/` records representative
Arrow inputs and expected outputs for expression assignment, multi-input SQL,
empty tables, metadata round trips, and state rollback. Regenerate it from the
frozen package with `uv run python scripts/export_v1_contract_fixtures.py`.

## Batch responsibilities

- `Batch` — immutable envelope for Arrow table or Python Array API payloads and
  their metadata.
- `BatchKind` — discriminates table batches from array batches.
- `BatchMetadata` — immutable ordering, provenance, cursor, event-time, and
  JSON-compatible attribute metadata.

## Graph responsibilities

- `Edge` — immutable connection between two graph port endpoints.
- `ExecutionPlan` — immutable validated topology that executes named batch
  inputs and owns checkpoint lifecycle operations.
- `NodeTiming` — timing record for one node execution.
- `Pipeline` — mutable graph builder that validates and compiles operators into
  an execution plan.
- `PortEndpoint` — identifies a node and one of its named ports.
- `RunMetadata` — immutable run identity, pipeline fingerprint, timestamps,
  settings, and source progress metadata.
- `RunResult` — named terminal batches plus timings, DataFusion metrics,
  warnings, and run metadata.

## Operator responsibilities

- `ArrayExpressionOperator` — evaluates an allowlisted NumPy or JAX expression
  and explicitly referenced array UDFs.
- `ExpressionOperator` — performs a DataFusion expression, projection, or
  filter over one table input.
- `Operator` — abstract typed-port processing and checkpoint lifecycle
  contract.
- `Port` — named operator boundary with a batch kind, required flag, and
  optional exact Arrow schema.
- `SqlOperator` — performs one read-only multi-input DataFusion query.
- `StatefulOperator` — operator base that owns mutable state and deep-copying
  snapshot, restore, and reset behavior.
- `StatelessOperator` — operator implementation for a pure processing callable.

## Runtime responsibilities

- `BatchingSource` — groups in-memory records into table batches constrained by
  row and byte limits.
- `CancellationToken` — thread-safe cooperative cancellation signal for a run.
- `DataFusionExecutionError` — reports a DataFusion planning or execution
  failure at the table runtime boundary.
- `MicroBatchRunner` — reads a source, executes batches, delivers to sinks, and
  periodically checkpoints after successful delivery.
- `RunCancelledError` — signals that cancellation or a deadline stopped a run.
- `RunContext` — per-run operator context containing cancellation controls,
  shared DataFusion runtime, selected UDFs, settings, and node identity.
- `Sink` — protocol for delivering one output batch.
- `Source` — protocol for reading batches from an optional recovery cursor.
- `StreamingRunner` — executes and delivers one formed batch at a time, then
  checkpoints after sink success.

## Configuration responsibilities

- `CONFIG_FORMAT_VERSION` — current serialized project configuration format
  version.
- `ArrowFieldConfig` — data-only Arrow field name, type, and nullability model.
- `DataFusionConfig` — strict DataFusion session execution settings.
- `DataSourceConfig` — bounded sample input configuration for a graph port.
- `EdgeConfig` — data-only connection between configured source and target
  ports.
- `InputFormat` — supported serialized sample input formats.
- `NodeConfig` — strict data-only operator node configuration.
- `NodeKind` — supported configured operator kinds.
- `PipelineConfig` — configured graph name, nodes, and edges.
- `PortConfig` — configured external port name, batch kind, and optional Arrow
  schema.
- `PositionConfig` — editor position metadata for a configured node.
- `ProjectConfig` — canonical versioned, data-only project model.
- `RunOptions` — bounded preview-run timeout, row, byte, memory, and output
  settings.
- `UdfReferenceConfig` — serializable UDF name and version reference.
- `ValidationIssue` — one structured configuration validation error or warning.
- `ValidationReport` — aggregate validation result and its issues.
- `compile_project` — validates a project and produces an execution plan.
- `validate_project` — returns configuration issues without executing the
  project.

## Storage responsibilities

- `Checkpoint` — versioned pipeline fingerprint, source progress, node state,
  and creation time value.
- `CheckpointError` — base error for checkpoint persistence and recovery.
- `CheckpointFormatError` — reports malformed or unsupported checkpoint data.
- `CheckpointMismatchError` — reports a checkpoint that belongs to a different
  pipeline or compiled fingerprint.
- `CheckpointStore` — protocol for loading, saving, and deleting checkpoints.
- `FileCheckpointStore` — atomic local JSON checkpoint implementation with
  hashed pipeline filenames.
- `FileProjectStore` — atomic local JSON project store with safe YAML import and
  export.
- `ProjectConflictError` — reports creation of a project ID that already exists.
- `ProjectFormatError` — reports invalid persisted or imported project data.
- `ProjectNotFoundError` — reports access to an unknown project ID.
- `ProjectStoreError` — base error for project persistence operations.

## UDF responsibilities

- `ArrayUdf` — trusted array UDF definition with stable identity and argument
  count.
- `DataFusionScalarUdf` — trusted vectorized DataFusion scalar UDF definition
  with Arrow fields and volatility.
- `DuplicateUdfError` — reports duplicate registration of a UDF name and
  version.
- `UdfError` — base error for UDF registration, selection, and execution.
- `UdfExecutionError` — reports a registered UDF implementation or result
  contract failure.
- `UdfKind` — discriminates DataFusion scalar UDFs from array UDFs.
- `UdfReference` — serializable UDF name and version selected by an operator.
- `UdfRegistry` — mutable owner of trusted UDF implementations and catalog
  metadata.
- `UdfRegistrySnapshot` — immutable compile-time selection and lookup view.
- `UdfVersionConflictError` — reports incompatible versions selected for one
  DataFusion function name.
- `UdfVolatility` — DataFusion scalar UDF volatility classification.
- `UnknownUdfError` — reports a reference to an unregistered UDF version.
