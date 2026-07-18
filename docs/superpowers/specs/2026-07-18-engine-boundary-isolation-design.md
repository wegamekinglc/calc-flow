# Engine Boundary Isolation Design

**Status:** Implemented with measured ownership correction

**Date:** 2026-07-18

**Branch:** `feature/lazy-datafusion-runtime`

## Problem

Calc Flow distinguishes table and array values, but it does not isolate their
execution engines. `BatchKind::Table` is evaluated by DataFusion while
`BatchKind::Array` is evaluated by explicitly registered external providers
such as NumPy and JAX. Despite that distinction, every compiled plan stores
DataFusion configuration and UDF state, every execution creates a
`DataFusionRuntime`, and every operator receives that runtime through
`OperatorContext`.

Lazy DataFusion session construction removes the largest measured allocation
cost from external-only plans, but it does not establish an engine boundary.
An array-only plan is still rejected by invalid DataFusion settings, changes
fingerprint when unused DataFusion settings change, exposes DataFusion metrics,
and must be built through crates that depend unconditionally on DataFusion.

The current Python array implementation is also not a first-class core engine.
It is a bounded Python evaluator hosted behind the generic external-provider
callback contract. That callback correctly performs no table calculation, but
it still runs inside a table-aware plan and result lifecycle.

## Evidence

The exact branch head before this design produced the following focused
diagnostics on one host:

| Case                       | Direct/provider time | Plan time    | Residual plan cost |
| -------------------------- | -------------------- | ------------ | ------------------ |
| Python identity array      | `0.070 us`           | `4.8-5.6 us` | about `5 us`       |
| NumPy mean, 1,000 elements | `14.64 us`           | `20.08 us`   | `5.44 us`          |
| JAX mean, 1,000 elements   | `64.66 us`           | `83.26 us`   | `18.61 us`         |
| Rust external passthrough  | not applicable       | `2.365 us`   | core plan control  |

These probes are diagnostic rather than acceptance measurements. They show
that residual array overhead exists after DataFusion session construction is
deferred, but they do not define a timing gate.

Two correctness probes demonstrate contract coupling:

- changing only DataFusion configuration changes an array-only plan
  fingerprint;
- setting `datafusion.batch_size` to zero rejects an array-only plan during
  compilation.

## Goals

1. Make graph orchestration engine-neutral.
2. Ensure an external/array operator cannot access a DataFusion runtime through
   its execution context.
3. Create and prepare table-engine resources only for plans containing table
   operators.
4. Preserve mixed graphs containing both table and external operators.
5. Ensure unused table configuration cannot reject or change the identity of
   an array-only plan.
6. Preserve run identity, cancellation, deadlines, transactions, rollback,
   snapshots, node timing, routing, runners, sinks, and checkpoints.
7. Preserve one run-scoped DataFusion session for all table nodes in one run.
8. Retain lazy session initialization inside that run-scoped table runtime.
9. Reduce avoidable Python callback work after the core engine boundary is
   established without sharing Python payload roots across GC containers.
10. Prove isolation with correctness tests before using benchmarks as
    supporting evidence.

## Non-goals

- Do not add a second Rust array calculation implementation. NumPy and JAX
  remain optional Python-owned providers.
- Do not allow raw arrays or tables to cross graph, runner, or sink boundaries.
- Do not skip graph validation, transaction capture, rollback, cancellation,
  node timing, or result metadata for array plans.
- Do not add an unchecked array fast path beside `ExecutionPlan`.
- Do not cache DataFusion sessions across runs or store one on a compiled plan.
- Do not change the bounded array expression language or use Python `eval`.
- Do not split the Cargo workspace into new published crates in this change.
  Physical dependency and packaging isolation is a separate follow-up after
  runtime and contract isolation are proven.
- Do not introduce a project format v3 in this change. Existing format-v2
  documents remain readable and writable.

## Alternatives

### Separate `TableExecutionPlan` and `ArrayExecutionPlan`

This provides strong isolation for single-engine graphs but cannot naturally
represent a valid graph containing both table and external nodes. A third
mixed-plan type would duplicate graph, transaction, runner, and checkpoint
logic. This approach is rejected.

### Keep one `OperatorContext` with `Option<&DataFusionRuntime>`

This avoids constructing a runtime for array plans, but every external
operator still compiles against a DataFusion-bearing context and can branch on
table state. It fixes allocation without fixing the type boundary. This
approach is rejected.

### Capability-aware compiled nodes

One engine-neutral graph plan owns common lifecycle behavior. Compiled nodes
are classified as table or external, and execution passes a context specific
to that class. Mixed graphs retain one topology and one transaction while
engine resources remain isolated. This is the selected approach.

## Architecture

### Engine-neutral operator context

The public context used by external and custom operators contains only the
run-scoped orchestration contract:

```rust
pub struct OperatorContext<'a> {
    pub run: &'a RunContext,
}
```

It contains no DataFusion type, configuration, UDF catalog, metrics collector,
or engine lookup method. `ExternalOperatorFactory` continues to produce
operators using this context, so Python NumPy/JAX providers cannot request a
table session.

### Table-specific execution seam

Built-in `ExpressionOperator` and `SqlOperator` execute through a private
table-specific seam:

```rust
struct TableOperatorContext<'a> {
    run: &'a RunContext,
    datafusion: &'a DataFusionRuntime,
}
```

The table context is not exposed to external provider factories. The compiled
node representation records whether a node uses the external operator seam or
one of the built-in table operators. Common metadata and lifecycle operations
remain available through a focused internal abstraction so snapshot, restore,
reset, ports, configuration, and UDF references are not duplicated.

The public builder continues to accept built-in expression/SQL operators and
custom external operators. Any necessary source-compatible conversion is
implemented at the builder boundary; execution must not recover engine kind by
downcasting trait objects at runtime.

This classification is deliberately source-breaking outside the common boxed
builder calls: built-in table operators no longer implement the external
`Operator` trait, `OperatorContext` no longer exposes a DataFusion field, and
`ExecutionPlan::datafusion_config()` returns `Option<DataFusionConfig>`. The
change must ship in a breaking-release window and is documented in
`docs/rust-api.md`.

### Compiled engine requirements

Compilation derives immutable requirements from the classified nodes:

```rust
struct TablePlanResources {
    config: DataFusionConfig,
    udfs: UdfRegistrySnapshot,
    selected_udfs: Vec<UdfReference>,
}

struct ExecutionPlan {
    // common graph and lifecycle fields
    table: Option<TablePlanResources>,
}
```

An array/external-only plan stores `None`. A table-only or mixed plan stores
`Some` and validates the DataFusion configuration and selected table UDFs.
External UDF/provider metadata remains attached to its external operator and
does not initialize table resources.

### Execution data flow

```text
ExecutionPlan::execute
    -> acquire transaction and capture rollback state
    -> construct RunContext
    -> if table resources exist:
           prepare one run-scoped lazy DataFusionRuntime
       else:
           allocate no table runtime
    -> execute nodes in deterministic topology
           external node -> OperatorContext { run }
           table node    -> TableOperatorContext { run, datafusion }
    -> collect common node timings and outputs
    -> collect table metrics only when table runtime exists
    -> commit or roll back through the unchanged plan transaction
```

The runner, source, sink, checkpoint, cancellation, and node timing boundaries
remain common because they are graph semantics rather than calculation-engine
semantics.

### Project format and fingerprint

Format-v2 retains the optional/defaulted `pipeline.datafusion` field for
backward compatibility. Semantic use becomes conditional:

- table-only and mixed plans validate it, store it, and include it in the
  fingerprint;
- external/array-only plans do not validate it as an active engine setting and
  omit it from the semantic fingerprint;
- two otherwise identical array-only plans have the same fingerprint when only
  the unused DataFusion field differs;
- a table-only or mixed plan with invalid DataFusion configuration still fails
  before execution.

The canonical stored project remains unchanged. The fingerprint represents
active execution semantics, not irrelevant document defaults. This changes
array-only fingerprints once and must be called out as checkpoint-impacting in
the PR.

### Result metrics

`RunResult::datafusion_metrics` remains for format-v2 and public API
compatibility. It is populated only when table resources exist and is empty for
external-only plans. The plan must not construct a `DataFusionRuntime` merely
to produce that empty vector.

A generic engine-metrics replacement is deferred to a future project format
and public API revision; renaming the field is not required to enforce the
runtime boundary in this change.

### Python binding boundary

The Python binding retains one Tokio runtime because blocking/async facades,
runners, stores, and cancellation are common orchestration services rather
than table-engine services.

After core isolation is green, the binding removes one avoidable cost without
changing provider semantics:

1. provider options are encoded once when the Python operator is created;
   every call still receives a fresh JSON-compatible value so a mutating
   callback cannot affect the compiled operator or later calls.

Directly wrapping the `RunResult` payload in the `outputs` getter was tested
and rejected: it caused all 100 GC-cycle probes to remain alive. `RunResult`
and each returned `Batch` require independently clearable Python payload roots,
so the getter retains its second rehome operation.

The direct provider callback, plan callback, blocking execute, and async
execute paths must retain Python GC roots and caller-owned input immutability.

## Error handling

- Array/external execution cannot produce a DataFusion configuration or query
  error because it has no table runtime.
- Table configuration and selected native UDF failures remain eager before the
  first node executes.
- External provider failures preserve provider, name, version, and callback
  message.
- A mixed plan cannot execute a table node without prepared table resources;
  such a state is an internal invariant error, not a user configuration error.
- Any table or external node failure follows the existing rollback path.
- Closing an unused table runtime remains allocation-free, while an
  external-only plan has no table runtime to close.

## Testing

Every behavior change starts with a focused failing test.

### Core isolation tests

- an external-only operator context exposes run state and no table runtime;
- compiling an array-only plan with invalid unused DataFusion configuration
  succeeds;
- unused DataFusion configuration does not change an array-only fingerprint;
- a table or mixed plan still rejects invalid DataFusion configuration;
- an external-only execution returns empty DataFusion metrics without creating
  table resources;
- a mixed plan creates one table runtime and shares it across its table nodes;
- table queries, selected UDFs, query IDs, metrics, cancellation, rollback, and
  reuse retain current behavior.

### Python binding tests

- NumPy/JAX callbacks still receive independent options mappings;
- a callback mutation cannot affect the next execution;
- array output ownership and GC roots remain valid after plan/runtime wrappers
  are cleared;
- blocking and async array executions return identical batches and metadata;
- table execution remains unchanged.

### Benchmark evidence

Criterion remains the primary stable attribution tool:

- `execute/external_passthrough_1000_rows` measures the engine-neutral Rust
  graph shell;
- DataFusion runtime cases prove table setup remains lazy and run-scoped;
- `execute/expression_1024_rows` remains the table control.

The Python contract-v2 report retains provider and plan measurements for NumPy
and JAX. Timing classifications remain subject to compatible fingerprints and
the existing 5% CoV rule. The proposed no-op Python array-provider diagnostic
was not added in this change, so the evidence does not attribute the remaining
gap among Python binding, provider, and backend components.

## Acceptance criteria

1. External/array operator code cannot access `DataFusionRuntime` through its
   execution context.
2. Array-only compilation and fingerprinting are independent of unused
   DataFusion configuration.
3. Array-only execution allocates no `DataFusionRuntime`.
4. Mixed plans retain a single run-scoped lazy DataFusion session.
5. Existing transaction, runner, cancellation, rollback, timing, UDF, and
   table query tests pass.
6. Focused Rust, Python, schema, formatting, lint, documentation, coverage, and
   benchmark commands are recorded in the handoff.
7. No generated `python/calc_flow/_native*.so` remains in source.

## Follow-up boundary

After this change is merged and measured, a separate design may split the
workspace into an engine-neutral graph crate, a DataFusion table adapter, and
the Python integration crate or feature sets. That packaging change must
address public crate names, Arrow dependencies, wheel contents, feature
matrices, documentation, release inspection, and MSRV independently. Runtime
isolation does not depend on completing that physical split.
