# Lazy Run-Scoped DataFusion Runtime Design

**Status:** Proposed for review

**Date:** 2026-07-18

**Base:** `b333121b282861ea03e006db7a2a232f6a6566c2`

## Problem

Every `ExecutionPlan::execute` currently constructs a complete run-scoped
DataFusion `SessionContext` and registers selected native UDFs before executing
the first node. External-only plans still pay that cost even though their
operators never call DataFusion.

Phase 2 of the array benchmark investigation measured
`DataFusionRuntime::new` at 30.257 microseconds and runtime construction plus an
empty UDF selection at 30.913 microseconds. The compatible provider-to-plan
gaps were 48.593-80.152 microseconds. Runtime setup therefore exceeded the
approved core-design gate of both 10 microseconds and 20% of remaining plan
overhead. Even the deliberately conservative uncertainty calculation left the
worst runtime share at 29.153%.

NumPy and JAX plans use external providers. They need the run context,
transaction, rollback, cancellation, validation, routing, timings, and result
assembly, but they do not need a DataFusion session unless an operator actually
requests one.

## Goals

1. Avoid constructing a DataFusion `SessionContext` for runs whose operators
   never execute a DataFusion query.
2. Preserve one isolated DataFusion session per run whenever table execution
   needs it.
3. Preserve the public `DataFusionRuntime`, `OperatorContext`, `Operator`,
   `ExecutionPlan`, and `RunResult` APIs.
4. Preserve eager configuration and selected-UDF validation.
5. Preserve transactions, rollback, cancellation, node timing, metrics,
   checkpoint, runner, and error behavior.
6. Demonstrate the improvement with same-host Criterion evidence without
   making timing a CI correctness gate.

## Non-goals

- Reusing or pooling `SessionContext` values across runs.
- Storing a session in `ExecutionPlan` or a runner.
- Adding an unchecked or reduced-semantics execution path.
- Skipping state snapshots, rollback markers, validation, cancellation checks,
  node timing, metrics collection, or result metadata.
- Changing Python provider callbacks, PyO3 ownership, array expression
  evaluation, project documents, fingerprints, checkpoints, or Studio APIs.
- Adding a compile-time operator capability flag.
- Changing the DataFusion configuration or selected-UDF contract.
- Adding performance thresholds to unit tests or CI.

## Required invariants

### Runtime lifetime

- `DataFusionRuntime` remains created once per `ExecutionPlan::execute` call.
- A valid table query creates at most one `SessionContext` in that runtime.
- All queries in one run share that one session and its stable UDF catalog.
- No session, table registration, UDF catalog, or query metric survives the
  run that owns it.
- Closing an unused runtime must not construct a session.
- A closed runtime must not construct a session or accept later queries.

### Validation and errors

- Invalid `batch_size` and `target_partitions` fail in
  `DataFusionRuntime::new`, before operator execution.
- `register_udfs` continues to reject missing native implementations, invalid
  selections, and SQL-name collisions before operator execution.
- External UDF references remain metadata and never initialize DataFusion.
- DataFusion planning, registration, and execution errors retain the current
  node identity and sanitized messages.
- Query failure leaves the same runtime reusable for a later valid query.
- Plan failure retains the existing rollback and plan-reuse behavior.

### Concurrency and cancellation

- The existing asynchronous query lock continues to serialize temporary table
  aliases and first-session initialization.
- The first session is published only after it is completely configured with
  all selected native UDFs.
- Concurrent first queries cannot observe a partially initialized catalog or
  create multiple sessions.
- Cancellation checks and plan lifecycle locks remain in their current
  positions; lazy initialization does not introduce a new await point.

### Observability

- External-only runs return an empty `datafusion_metrics` list, as before.
- Table runs retain monotonic per-run query IDs starting at one.
- Logical plans, physical plans, row counts, planning time, and execution time
  retain their current meaning.
- `NodeTiming.duration_ns` continues to measure the complete
  `Operator::process` call. The first DataFusion operator therefore includes
  lazy session initialization in its duration, where eager setup previously
  occurred before node timing began. This is an intentional attribution
  change; the field, unit, ordering, and timing boundary do not change.
- Pipeline fingerprints and serialized project documents do not change.

## Alternatives considered

| Approach                                      | Advantages                                                   | Rejected cost or risk                                                                 |
| --------------------------------------------- | ------------------------------------------------------------ | ------------------------------------------------------------------------------------- |
| Lazy session inside `DataFusionRuntime`       | Preserves public APIs and run isolation; pays only on query  | Requires careful deferred UDF installation and first-query concurrency tests          |
| Compile-time `requires_datafusion` plan flag  | Makes the decision explicit before execution                 | Adds operator capability plumbing and still needs an optional or dummy runtime context |
| Cache a session on the plan or across runs    | Could also reduce setup for repeated table runs              | Violates the approved run-scoped boundary and complicates tables, UDFs, metrics, and cancellation |

The selected approach is lazy session construction inside
`DataFusionRuntime`. It is the narrowest change that removes the measured cost
for external plans without changing what operators receive.

## Architecture

### Runtime representation

`DataFusionRuntime` retains its public type and methods. Its private state
changes from an eager `SessionContext` to:

```rust
pub struct DataFusionRuntime {
    config: DataFusionConfig,
    context: OnceLock<SessionContext>,
    selected_udfs: Vec<Arc<ScalarUDF>>,
    query_lock: AsyncMutex<()>,
    metrics: Mutex<Vec<DataFusionQueryMetric>>,
    next_query: AtomicU64,
    closed: AtomicBool,
}
```

`DataFusionRuntime::new` validates and stores `DataFusionConfig`, creates the
empty synchronization and metrics state, and leaves `context` empty. It does
not create a DataFusion session.

A private `context()` helper uses `OnceLock::get_or_init` to create the
`SessionConfig`, construct the `SessionContext`, register every previously
validated native UDF, and then publish the completed session. Session creation
is infallible after configuration and UDF validation, so fallible work remains
outside the initializer.

### UDF preparation

`register_udfs` keeps its `&mut self` signature and current validation order:

1. reject a closed runtime;
2. validate selected references;
3. resolve native references from the compile-time snapshot;
4. validate SQL names for collisions;
5. store the resolved `Arc<ScalarUDF>` values when no session exists;
6. if a session already exists, register the validated UDFs immediately.

The production execution path calls `register_udfs` before sharing the runtime,
so the common case stores a stable prepared catalog without initializing the
session. Supporting an already-initialized runtime preserves the current public
method's behavior for direct Rust callers.

### Query path

`evaluate` continues to normalize an expression into a read-only query and
delegates to `sql`.

`sql` retains the current order for operations that can fail without a
session:

1. reject a closed runtime;
2. reject an empty table map;
3. validate the read-only query;
4. acquire the query lock;
5. recheck that the runtime is open;
6. obtain the lazily initialized session;
7. register temporary input aliases;
8. plan, execute, collect, and record metrics;
9. deregister aliases before releasing the query lock.

External operators receive the same `&DataFusionRuntime` in
`OperatorContext`. If they never call `evaluate` or `sql`, the `OnceLock`
remains empty and the run closes without constructing DataFusion.

### Close and metrics

`close` remains idempotent and only marks the runtime closed. It does not force
initialization and does not need to extract or mutate the session.

`metrics` remains independent of session creation. An unused runtime and an
external-only run both return an empty vector.

## Data flow

```text
ExecutionPlan::execute
    -> transaction, input validation, snapshot, rollback marker
    -> DataFusionRuntime::new(config)          [validate only]
    -> register_udfs(snapshot, selected)       [resolve and store only]
    -> execute nodes with unchanged OperatorContext
         -> external operator                 [no session]
         -> expression/SQL operator
              -> DataFusionRuntime::sql
              -> first call initializes one session and catalog
              -> later calls reuse that run-local session
    -> close runtime
    -> collect metrics and RunResult
    -> commit or rollback transaction
```

## Failure handling

Configuration and UDF preparation remain eager so errors cannot move into a
later arbitrary node. Session creation itself has no fallible step after those
checks. Query validation still precedes initialization, which means an invalid
direct query can fail without allocating a session; its error variant and
message remain unchanged.

If a table registration, planning, or execution error occurs after
initialization, temporary registrations unwind through the existing RAII guard.
The runtime remains usable for another query, and an enclosing plan still runs
the existing rollback path. Lazy state is entirely run-local and is discarded
with the runtime after success or failure.

## Testing strategy

### Focused correctness tests

- Prove a newly constructed runtime has no initialized session.
- Prove valid and invalid UDF preparation does not initialize a session.
- Prove closing a fresh runtime does not initialize a session and later queries
  retain the current closed-runtime error.
- Prove the first valid query initializes a session and a second query reuses
  it while query IDs remain one and two.
- Prove a selected native UDF is available on the first lazy query.
- Prove concurrent first queries share one initialized session and retain alias
  isolation.
- Prove the external passthrough plan remains correct and returns empty
  DataFusion metrics through the existing public benchmark path.
- Preserve the existing expression, SQL, UDF, rollback, cancellation, runner,
  and checkpoint test suites unchanged.

The `datafusion.rs` in-file tests inspect the private `OnceLock` directly.
Production code does not gain an initialization-state probe or any method used
only by tests.

### Performance evidence

Keep the existing Criterion cases so results remain comparable:

- `execute/datafusion_runtime_new` measures the now-lightweight validated
  runtime envelope;
- `execute/datafusion_runtime_new_register_udfs` measures runtime preparation
  with an empty selection;
- `execute/external_passthrough_1000_rows` measures the target external-only
  plan path;
- `execute/expression_1024_rows` controls the table path.

Run the baseline from merged `main` and the candidate on the same host with the
same Rust toolchain and benchmark profile. The change is useful only if the
external passthrough improves by at least 10 microseconds and 20% without a
material expression-path regression. Record point estimates and confidence
intervals in a handoff document. Do not encode these thresholds in tests or CI.

The Python contract-v2 provider and plan benchmarks remain informational. Run
them on the same host and reject incompatible or noisy comparisons according to
the existing benchmark contract.

## Verification

The implementation PR must run every command in `AGENTS.md`, including Rust
format, Clippy, tests, 90% coverage, rustdoc, Python tests and Ruff, Studio
backend coverage, frontend generation/build/tests/E2E/audit, dependency policy,
release helper tests, generated-artifact checks, and `git diff --check`.

The PR must also confirm:

- no generated `python/calc_flow/_native*.so` remains;
- project schema, OpenAPI, and generated TypeScript API have no diff;
- the exact pushed head owns the passing CI results;
- all actionable review threads are addressed and resolved.

## Rollout

This is one independently reviewable performance PR based on merged `main`.
The public API and serialized contracts do not change, so no version bump,
schema regeneration, migration, feature flag, or compatibility shim is needed.

If the measured gate is missed, keep the correctness refactor only if it has a
clear non-performance benefit; otherwise revert it and retain the Phase 2
evidence. Any later cross-run session reuse or operator capability protocol
requires another design.
