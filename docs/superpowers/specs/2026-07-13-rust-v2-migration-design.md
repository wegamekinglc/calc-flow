# Calc Flow Rust V2 Migration Design

**Date:** 2026-07-13

**Status:** Approved

## Objective

Replace the Python implementation of Calc Flow's core library with a Rust
implementation while preserving two supported public surfaces: an idiomatic
Rust API and a Python API backed by Rust. Keep Calc Flow Studio's backend in
Python using FastAPI. Retain NumPy and JAX as Python-hosted optional array
providers rather than reimplementing them in Rust.

This is an intentionally breaking v2. Existing v1 Python imports, project
documents, checkpoints, and serialized state are not compatibility targets and
receive no automated migration path.

## Confirmed Product Decisions

- V2 may redesign both the Rust and Python APIs.
- Python v1 development freezes while the v2 rewrite proceeds.
- The rewrite uses a monolithic Rust core crate, followed by Python bindings and
  Studio integration after core parity.
- NumPy and JAX remain Python-side adapters invoked by a Rust-owned scheduler.
- Graphs containing Python array nodes run only in a Python-hosted process.
  Rust-only hosts reject those graphs during compilation.
- V1 project documents and checkpoints are not migrated automatically.
- The first production release supports Python 3.13 or newer and prebuilt
  wheels for Linux x86_64/aarch64, macOS x86_64/arm64, and Windows x86_64.
- The Rust library is published as a Rust crate as well as through the Python
  distribution.
- The FastAPI Studio backend remains Python and continues to own HTTP, SSE,
  process-isolation, and worker-lifecycle concerns.

## Selected Migration Approach

The selected approach is a monolithic rewrite. The core is ported into one
primary Rust crate before Python bindings and FastAPI integration are completed.
This minimizes early workspace and cross-crate coordination, at the cost of
discovering some binding and packaging risks later than a vertical-slice
migration would.

"Monolithic" describes the public crate boundary, not the source layout. The
core crate retains focused internal modules with explicit responsibilities and
one-way dependencies. Python bindings live in a separate, thin extension crate
because PyO3 and `cdylib` packaging are integration concerns rather than core
engine responsibilities.

The rejected alternatives were:

- A contract-first multi-crate workspace with end-to-end vertical slices. It
  offers earlier integration feedback but was not selected.
- A separate Rust execution service accessed over IPC or Arrow Flight. It adds
  operational and protocol complexity and weakens the embeddable-library goal.

## Repository Shape

The target repository remains a monorepo:

```text
calc-flow/
|-- Cargo.toml
|-- crates/
|   |-- calc-flow/          # monolithic Rust core
|   `-- calc-flow-python/   # thin PyO3 extension
|-- python/
|   `-- calc_flow/          # Python facade, typing, NumPy/JAX adapters
`-- web-ui/
    `-- backend/            # retained Python/FastAPI service
```

The core crate contains focused modules corresponding to the existing product
responsibilities: batches, context and cancellation, expressions, DataFusion,
UDFs, operators, graph construction and compilation, execution, checkpoints,
project configuration and storage, sources and sinks, micro-batch execution,
and streaming execution.

## Ownership Boundaries

### Rust Core

Rust owns the canonical implementations of:

- Immutable batch envelopes and metadata.
- Arrow-backed table payloads.
- Graph models, ports, connections, validation, compilation, and deterministic
  fingerprints.
- Topological scheduling and run lifecycle.
- DataFusion expression and SQL execution.
- Built-in operators and native Rust UDF registration.
- Run cancellation, timings, query metrics, and structured tracing.
- Stateful snapshots, rollback, reset, and checkpoint values.
- Project and checkpoint formats and filesystem stores.
- Source and sink traits, batching, micro-batch runners, and streaming runners.
- The canonical v2 configuration schema and validation rules.

Rust table pipelines must build and run without Python installed.

### Python Binding and Facade

The PyO3 extension exposes Rust-owned types and operations without duplicating
core execution or validation. The Python facade provides Pythonic construction,
typing, exception classes, sync and async entry points, and conversions to and
from PyArrow.

Python also owns optional providers for NumPy, JAX, and trusted Python UDFs.
Those providers implement Rust-defined external-provider interfaces through the
binding layer. Persisted configuration contains only provider, name, version,
and data-only options. It never contains callables, source, or import paths.

### FastAPI Studio Backend

The backend stays under `web-ui/backend` as a separate Python package. It owns:

- FastAPI routes and `/api/v2` HTTP contracts.
- Pydantic route and response models.
- SSE event production.
- Spawned worker processes and their time, CPU, memory, output, and cancellation
  controls.
- Loopback-only serving.
- OpenAPI generation for the React application.

Workers call the Rust-backed Python API. The backend does not reimplement graph
validation, execution, or checkpoint semantics.

## Batch and Extension Model

The canonical Rust `Batch` is immutable and preserves metadata across
transformations. Table payloads consist of Arrow schemas and record batches.
Array payloads are represented by an opaque, owned extension payload behind a
Rust trait boundary. The core scheduler can route and validate extension
payloads without depending on Python.

The PyO3 crate supplies the concrete Python payload implementation. It stores a
safe owned reference to the Python array object and accesses it only while the
GIL is held. NumPy adapters make defensive owned inputs read-only before placing
them in an immutable batch. JAX values retain their functional backend
semantics. External payload implementations must provide batch-kind identity,
debug metadata, safe cloning semantics, and owned lifetime behavior.

PyArrow tables cross the boundary through Arrow's C Data Interface. Conversion
logic is centralized so ownership, lifetime, chunking, schema, and copy behavior
are consistent and testable.

## Graph Compilation

Rust provides a builder API and a strict data-only `ProjectSpec`. Both compile
through the same validation path. Compilation checks:

- External inputs and outputs.
- Required ports.
- Batch kinds and exact Arrow schemas where declared.
- Single-writer input rules.
- Cycles and deterministic topology.
- Operator configuration.
- Referenced UDF and provider names and exact versions.
- Conflicts between selected DataFusion UDF versions.
- Availability of Python-only providers in the current host.

Compilation produces an immutable `ExecutionPlan`, a snapshot of selected UDF
and provider metadata, and a deterministic v2 fingerprint. V2 fingerprints are
not required to match Python v1 fingerprints.

## Execution Data Flow

Both public APIs create the same Rust-owned project and execution plan:

```text
Rust builder / Python facade / project JSON
                    |
                    v
          Rust schema validation
                    |
                    v
       immutable compiled ExecutionPlan
                    |
                    v
             async Rust runner
          /                     \
         v                       v
 Rust/DataFusion operator    Python provider
          \                     /
           v                   v
            immutable Batch outputs
                    |
                    v
        sinks succeed -> atomic checkpoint
```

Every run creates a Rust `RunContext` containing cancellation, settings,
metadata, a per-run DataFusion session, and metrics collectors. Rust schedules
nodes topologically. Rust and DataFusion work executes without holding the
Python GIL.

Table operators register Arrow inputs into a run-scoped DataFusion
`SessionContext`, execute expressions or a single `SELECT`/CTE query, collect
results and metrics, and clean up temporary registrations. DDL, DML, and utility
SQL remain rejected.

For Python-hosted array or UDF nodes, Rust invokes the registered external
provider. The binding acquires the GIL only for the Python call. Pure Rust hosts
fail compilation when such a provider is required rather than spawning Python
implicitly.

## Async and Concurrency Model

Rust execution, DataFusion operations, sources, sinks, and runners are
async-first. The Rust API exposes asynchronous execution as the primary path.
The Python facade exposes both an awaitable API and a blocking convenience API.
The blocking API must detect an incompatible running event loop rather than
silently blocking it.

Python callbacks are isolated at explicit external-operator boundaries. The
binding does not hold the GIL during graph scheduling or DataFusion execution.
The FastAPI server never runs core work directly on its event loop; it delegates
work to the existing spawned-worker boundary.

## State, Delivery, and Checkpointing

Before a run begins, the execution plan snapshots every stateful operator. A
failed operator, Python exception, cancellation, output-limit failure, sink
failure, or checkpoint failure restores in-memory operator state to the
pre-run snapshot where delivery has not committed.

Micro-batch and streaming modes retain at-least-once delivery. Sinks complete
before the checkpoint advances. Checkpoints are versioned v2 values containing
the pipeline identity and fingerprint, source cursor and sequence, node-keyed
JSON-compatible state, and creation time. Filesystem checkpoint writes remain
atomic and pipeline names are converted to safe hashed filenames.

V1 checkpoint state is rejected with an unsupported-version error.

## Public APIs

The Rust surface includes immutable batches, the pipeline builder and execution
plan, run results, built-in operators, operator/source/sink/provider traits,
project validation, stores, runners, UDF registration, and structured errors.
The API is idiomatic Rust and is not constrained to reproduce Python v1 naming.

The Python facade presents equivalent concepts with Python naming and typing.
For example, Python users construct a batch from a PyArrow table, build or load
a project, compile it through Rust, and receive a Python wrapper around the
Rust-owned `RunResult`. Python wrappers do not contain a second graph compiler
or execution scheduler.

## Configuration and Schema

Rust `ProjectSpec` structures use Serde, reject unknown fields, and generate the
canonical v2 JSON Schema. The Python package provides a Pydantic-compatible
wrapper whose validation hook calls Rust and whose JSON-schema hook exposes the
generated Rust schema. This lets FastAPI remain Python-native while preventing
configuration rules from drifting between languages.

The new version identifiers are:

- Project schema version 2.
- Checkpoint schema version 2.
- Studio routes under `/api/v2`.
- Rust crate named `calc-flow`.
- Python distribution named `calc-flow` and import package named `calc_flow`.

V1 documents and imports receive explicit unsupported-version or missing-API
errors. No compatibility shim or automated converter is included.

## Error Handling and Observability

Rust exposes a typed error hierarchy with stable categories for configuration,
graph compilation, DataFusion execution, external providers, state,
checkpointing, stores, cancellation, resources, and I/O. Errors carry relevant
node, port, expression, path, and source context without exposing input data.
Expected user errors never rely on panics.

PyO3 maps Rust categories into a stable Python exception hierarchy. Python
exceptions raised by providers become external-provider errors that preserve
useful traceback context. Panics are contained at the FFI boundary and reported
as internal defects rather than unwinding into Python.

Structured tracing spans cover runs, nodes, DataFusion queries, sinks, and
checkpoints. `RunResult` exposes stable run metadata, node timings, and
DataFusion metrics. The FastAPI backend bridges relevant Rust tracing into
Python logging without recording sensitive input values.

## Migration Phases

### Phase 0: Freeze and Characterize V1

Tag the final Python v1 release, document its support status, inventory public
behavior, produce language-neutral Arrow/JSON fixtures, and record performance
baselines. V1 remains available as a read-only reference until v2 is released.

### Phase 1: Establish the Rust Core

Add the Cargo workspace and core crate, establish Rust CI and quality gates,
then implement errors, immutable metadata, batches, cancellation, and run
context.

### Phase 2: Port Calculation Capabilities

Port expression handling, DataFusion execution, SQL validation, query metrics,
and native Rust UDF registration.

### Phase 3: Port Graph Execution

Implement ports, operators, graph building, compilation, fingerprints,
topological scheduling, state snapshots, rollback, cancellation, and run
results.

### Phase 4: Port Persistence and Runners

Implement the v2 project schema, project and checkpoint stores, sources, sinks,
batching, micro-batch runners, and streaming runners. Complete and document the
Rust public API before binding work begins.

### Phase 5: Add Python Support

Add PyO3 and Maturin, Python wrappers and type hints, Arrow interchange,
exception mapping, sync and async execution, Python UDFs, and NumPy/JAX
providers.

### Phase 6: Reconnect FastAPI Studio

Move the backend to `/api/v2`, delegate project validation and execution to
Rust, retain Python process isolation, regenerate OpenAPI, and update the React
client only for changed contracts.

### Phase 7: Package and Release

Build and test Rust crates and Python wheels on every supported platform, run
all correctness, security, coverage, end-to-end, and performance gates, publish
release candidates, and remove the Python v1 core only after the final v2 gate.

## Verification Strategy

Rust unit tests cover every focused module. Public integration tests cover
cross-module behavior. Language-neutral fixtures hold Arrow IPC inputs and
canonical JSON expectations for use from both Rust and Python.

Core test coverage includes:

- Arrow schemas, nulls, chunking, metadata, empty batches, and ownership.
- DataFusion expressions, assignments, projections, filters, joins, CTEs,
  rejected SQL, UDF selection, metrics, and temporary-table cleanup.
- Graph endpoints, required ports, kinds, schemas, writers, cycles, topology,
  and fingerprints.
- Stateful isolation and rollback after execution, delivery, cancellation, and
  checkpoint failures.
- Project and checkpoint atomicity, corruption, safe filenames, and concurrent
  revisions.
- Property-generated DAGs, batch boundaries, metadata, and project documents.

The Rust core retains a 90% coverage floor. The FastAPI backend retains its 85%
coverage floor.

Python binding tests cover Arrow transfer behavior, exception mapping, GIL
release, NumPy ownership, JAX backend retention, callback failures, rollback,
async and blocking behavior, and repeated creation/destruction for leaks.

Studio tests cover worker resource limits, cancellation and termination, SSE
ordering, output bounds, validation, and loopback serving. React builds, Vitest,
Playwright, and dependency audits remain required.

Benchmarks compare v2 against the frozen v1 baseline for empty plans,
single-operator overhead, compilation, DataFusion expressions and SQL, state
and checkpoint costs, Python/Rust crossings, NumPy/JAX callbacks, worker
startup, and end-to-end Studio runs. Material regressions require explicit
documentation and approval.

## Risks and Controls

- DataFusion semantic drift is controlled with shared Arrow fixtures and
  explicit SQL and expression acceptance tests.
- Arrow ownership errors are controlled by centralized conversion code and
  repeated transfer/destruction tests.
- GIL deadlocks and event-loop blocking are controlled through narrow callback
  boundaries, GIL-release tests, explicit async APIs, and worker isolation.
- State corruption is controlled by pre-run snapshots and fault injection at
  operators, sinks, cancellation, and checkpoint writes.
- Monolithic-crate growth is controlled through focused modules and enforced
  dependency direction inside the crate.
- Native packaging risk is controlled by continuously building wheels for all
  production targets.
- Uncooperative native work is contained by the Python backend's operating
  system process boundary and forced-termination controls.
- The feature-freeze cost is controlled by recording deferred v1 requests and
  reconsidering them only after the corresponding v2 subsystem is complete.

## Explicit Non-Goals

- V1 import, project, checkpoint, or fingerprint compatibility.
- Automated v1 artifact conversion.
- A Rust replacement for FastAPI.
- Rust-hosted Python worker management.
- Distributed execution, IPC execution services, or Arrow Flight protocols.
- A React Studio redesign unrelated to v2 contracts.
- An alternative table engine.
- Rust reimplementations of NumPy or JAX.
- A stable C ABI or dynamically loaded native-plugin ABI in v2.0.

## Completion Criteria

The migration is complete when all of the following are true:

1. A Rust-only application can build, run, checkpoint, and recover table
   pipelines without Python installed.
2. Python users can build and execute equivalent Rust-backed table pipelines.
3. Python-hosted graphs can run NumPy and JAX operators with correct ownership,
   cancellation, error mapping, and rollback.
4. FastAPI Studio can create, validate, run, cancel, observe, and persist v2
   projects end to end.
5. Rust crates and Python wheels install and pass smoke tests on every selected
   production platform.
6. Rust, Python, backend, frontend, end-to-end, audit, coverage, and benchmark
   gates pass.
7. Documentation covers both APIs, the v2 configuration format, operational
   limits, packaging, and the lack of v1 artifact compatibility.
8. Removing the frozen Python v1 core does not affect any v2 execution path.
