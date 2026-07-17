# Array Benchmark Regression Design

**Status:** Approved for implementation planning

**Date:** 2026-07-16

**Baseline:** `b42b687c8021291bcb537a57fded7a40f7f8477d`

## Problem

The Rust-native v2 array benchmarks show a large regression relative to the
removed Python v1 engine at the `overhead` scale. Same-host measurements with
Python 3.13.9, NumPy 2.5.1, JAX 0.10.2, and the CPU backend reproduced the
following changes:

| Backend | Scenario    | v1 direct | v2 plan   | Slowdown |
| ------- | ----------- | --------: | --------: | -------: |
| NumPy   | Elementwise |   7.58 us | 134.81 us |   17.80x |
| NumPy   | Reduction   |   4.65 us | 109.01 us |   23.42x |
| NumPy   | Matmul      |   4.03 us | 110.63 us |   27.48x |
| JAX     | Elementwise |  89.46 us | 270.86 us |    3.03x |
| JAX     | Reduction   |  54.11 us | 208.09 us |    3.85x |
| JAX     | Matmul      |  72.98 us | 169.47 us |    2.32x |

The slowdown is not primarily inside NumPy or JAX. The v1 benchmark timed a
direct array-engine call, while the v2 benchmark times a complete compiled
plan. The v2 path includes Python-to-Rust binding work, Tokio blocking,
transactional state capture, input/output validation, a run-scoped DataFusion
session, a Python provider callback, result metadata, and immutable output
ownership.

Three measured costs dominate small workloads:

1. A no-op external plan has a fixed graph and Python callback cost of roughly
   39-42 us before constructing a new output batch.
2. Array expressions are parsed and structurally validated on every provider
   call, adding roughly 11-22 us directly and 20-50 us in the complete path.
3. NumPy ownership first copies into a C-order array and then copies into an
   immutable `bytes` object. The cost grows from about 3.9 us for 1,000
   `float64` elements to about 1.93 ms for 1,000,000 elements.

The existing benchmark comparison can also produce misleading conclusions.
It accepts reports from different machines, dependency versions, dtypes, and
workload definitions as long as scenario/backend and scale appear compatible.

## Goals

1. Make benchmark comparisons reject incompatible environments and workloads.
2. Separate upstream backend cost, Calc Flow provider cost, full-plan cost, and
   ownership cost so future regressions identify the failing layer.
3. Reuse successful parsed expressions through a bounded process-local cache
   while preserving runtime intermediate-result validation.
4. Reduce NumPy immutable ownership to one full payload copy without weakening
   caller isolation or read-only guarantees.
5. Produce measured evidence before considering execution-core changes.
6. Preserve all public v2 project, provider, batch, execution, checkpoint, and
   cancellation contracts in the semantics-preserving phase.

## Non-goals

- Reintroducing any Python v1 engine or compatibility shim.
- Bypassing the bounded AST evaluator with `eval`, `exec`, generated Python, or
  executable project data.
- Making microbenchmarks hard CI gates before 20 compatible main-branch samples
  exist.
- Reusing caller-owned mutable NumPy storage.
- Adding an unchecked execution API.
- Skipping transaction, cancellation, rollback, node timing, or checkpoint
  behavior during the semantics-preserving phase.
- Changing the Studio HTTP API or adding benchmark persistence.
- Changing execution-core semantics without a second reviewed design after the
  Phase 1 evidence gate.

## Required invariants

### Array safety

- `Batch.from_array` never mutates caller-owned values.
- A NumPy batch owns a stable snapshot independent of subsequent caller
  mutation.
- Every reachable NumPy array in the returned base chain is non-writeable and
  cannot have writeability re-enabled.
- Supported dtype aliases retain their normalized native dtype.
- Unsupported, object, executable-element, and non-native-endian dtypes fail
  before magic-method dispatch.
- JAX batches and provider outputs remain `jax.Array` values.
- Metadata is defensively copied and preserved exactly.

### Expression safety

- Expression length, AST node count, AST depth, exponent, reshape rank,
  dimension, and element limits remain unchanged.
- Invalid expressions fail during compilation.
- Invalid expressions are not cached.
- Every intermediate operation result is validated during every execution.
- Cached AST objects are private, bounded, and read-only by convention; no
  caller receives a reference.
- No parsed AST or prepared callback is serialized into project documents,
  fingerprints, checkpoints, or catalogs.

### Execution semantics

- Provider failures retain provider, name, version, and category.
- A failed callback leaves the plan reusable after rollback.
- Existing dynamic provider callbacks continue to receive a fresh options
  mapping for each execution.
- Existing GC traversal, Python owner retention, async cancellation, and runner
  lifecycle behavior remain unchanged unless a later design explicitly proves
  and approves a safe change.

## Two-phase architecture

### Phase 1: semantics-preserving benchmark and provider improvements

Phase 1 contains four independently reviewable units:

1. A strict benchmark identity and compatibility contract.
2. Layered array benchmarks for backend, provider, plan, and ownership costs.
3. A bounded successful-expression parse cache.
4. A one-copy immutable NumPy snapshot.

Phase 1 concludes with a same-host A/B evidence gate. It does not change the
Rust execution lifecycle.

### Phase 2: measurement and architecture decision

Phase 2 adds targeted Rust and Python boundary measurements only if Phase 1
does not meet its acceptance target. The measurements isolate:

- `DataFusionRuntime::new` and selected-UDF registration;
- native external passthrough execution;
- transaction snapshot and rollback bookkeeping;
- Python provider bridging and result rehoming.

An execution-core optimization requires a new design if one component accounts
for at least 10 us and at least 20% of the remaining compatible full-plan
overhead. Without that evidence, the work stops after Phase 1.

## Benchmark contract

### Measurement scopes

Array measurements use four stable scope identifiers:

- `backend_kernel`: the raw NumPy or JAX operation, used as an environment
  control.
- `provider_boundary`: the Calc Flow bounded evaluator and batch ownership
  boundary without graph execution.
- `plan_end_to_end`: the public compiled-plan execution path.
- `batch_ownership`: `Batch.from_array` snapshot construction by payload size.

The existing array benchmark is split by responsibility:

- `benchmarks/test_array_kernel.py`
- `benchmarks/test_array_provider.py`
- `benchmarks/test_array_plan.py`
- `benchmarks/test_array_ownership.py`

The complete `pytest benchmarks` command continues to discover all scopes.

### Required metadata

Every new array benchmark record contains these JSON-compatible fields in
`extra_info`:

- `benchmark_contract_version`: integer `2`;
- `scenario`: stable logical operation name;
- `scope`: one of the four identifiers above;
- `workload_version`: integer starting at `1` for the new contract;
- `backend`: `numpy` or `jax`;
- `scale`, `table_rows`, `array_elements`, and `matrix_dimension`;
- `input_rows` and `output_rows`;
- `expression`: exact normalized expression or a stable operation label;
- `input_dtype` and `output_dtype` as observed at runtime;
- `jax_platform` and `jax_enable_x64` for JAX, otherwise absent;
- `python_version`, `numpy_version`, `jax_version`, and `jaxlib_version` when
  applicable;
- `machine_identity` and `machine_fingerprint`;
- `dependency_identity` and `dependency_fingerprint`;
- `workload_fingerprint`;
- `process_rss_bytes`.

Fingerprints are lower-case SHA-256 hex digests of canonical JSON documents
with sorted keys and compact separators.

### Stable machine identity

The machine identity includes only:

- operating system;
- machine architecture;
- normalized CPU brand;
- logical CPU count;
- Python implementation.

It excludes hostname, current CPU frequency, timestamps, process identifiers,
and memory usage. An Intel and AMD GitHub-hosted runner are incompatible even
when both use `ubuntu-latest`.

### Dependency identity

The dependency identity includes Python, NumPy, JAX, and JAXlib versions that
can change array behavior. Calc Flow source commit and package version remain
report labels rather than compatibility fields because the purpose of the
comparison is to measure Calc Flow changes.

### Workload identity

The workload fingerprint covers contract version, scenario, scope, workload
version, backend, actual input/output dtype, scale dimensions, exact expression
or operation label, and backend configuration. Any change to those fields
creates a new workload rather than a regression against the old workload.

### Compatibility result

The frontend comparison returns a structured result:

```typescript
type CompatibilityStatus = "compatible" | "incompatible" | "unverified";

interface BenchmarkCompatibilityIssue {
  code:
    | "missing_contract_metadata"
    | "contract_version_mismatch"
    | "machine_mismatch"
    | "dependency_mismatch"
    | "scale_mismatch"
    | "scope_mismatch"
    | "workload_mismatch"
    | "dtype_mismatch"
    | "backend_configuration_mismatch";
  field: string;
  baseline: unknown;
  current: unknown;
}

interface BenchmarkComparisonResult {
  status: CompatibilityStatus;
  rows: BenchmarkComparisonRow[];
  issues: BenchmarkCompatibilityIssue[];
}
```

Compatible reports receive the existing stable, noisy, regression, and
improvement classifications. Incompatible reports receive explicit issues and
no classified rows. Legacy reports missing contract-v2 metadata are
`unverified`; they remain viewable but receive no performance classification.
Malformed statistics, duplicate workload identities, or structurally invalid
metadata remain hard parse errors.

## Expression preparation

The public-facing parser retains object validation and delegates successful
strings to a bounded cache:

```text
untrusted object
    -> string and length validation
    -> cached parser keyed by exact string
        -> miss: ast.parse, bounds checks, syntax validation
        -> hit: reuse private AST
    -> evaluator with per-operation result validation
```

The cached function accepts only a validated `str`, uses `functools.lru_cache`
with `maxsize=256`, and returns only successful `ast.Expression` objects.
Python does not cache raised exceptions, so invalid input is revalidated on
each attempt. Tests may clear and inspect the private cache; production callers
cannot access it through the public package exports.

The evaluator never mutates AST nodes. Intermediate NumPy/JAX values continue
to pass through their current validator after every AST operation.

### Optional prepared-provider midpoint

After the cache change, a midpoint benchmark determines whether per-run option
conversion remains material. If controlled probes attribute less than 10 us or
less than 10% of compatible full-plan time to option conversion and dynamic
callback dispatch, the existing callback boundary is retained without adding a
new protocol.

If option conversion still accounts for at least 10 us and 10% of compatible
full-plan time, a later Phase 1 task may add this backward-compatible protocol:

```python
class PreparableProvider(Protocol):
    def prepare(self, options: Mapping[str, object]) -> Callable[[Batch], Batch]: ...
```

The factory calls `prepare` during plan compilation after validating the
options. A returned callable is rooted and traversed with the plan and receives
only the input batch. Providers without `prepare` retain the current
`callback(batch, fresh_options)` behavior. Preparation failures are reported as
compile-time `provider.options` errors. This protocol is not implemented unless
the midpoint evidence satisfies the threshold.

## NumPy immutable ownership

The current two-copy path becomes one immutable snapshot:

```python
array = np.asarray(value)
_validate_numpy_dtype(array.dtype)
immutable_bytes = array.tobytes(order="C")
return np.frombuffer(immutable_bytes, dtype=array.dtype).reshape(array.shape)
```

`tobytes(order="C")` creates the sole full payload copy and captures logical
C-order values from contiguous, non-contiguous, transposed, Fortran-order, and
negative-stride inputs. `np.frombuffer` over `bytes` produces a non-writeable
view whose ultimate owner cannot be made writeable.

The implementation does not attempt zero-copy reuse of an apparently
read-only NumPy array. Proving arbitrary base-chain ownership is more complex
than the measured benefit justifies and is outside this design.

The output preserves exact normalized dtype and shape, including scalar and
empty arrays. Metadata handling remains in the existing PyO3 boundary.

## Failure handling

### Benchmark failures

- Incompatibility is data and produces structured issues.
- Malformed JSON, non-finite or non-positive statistics, invalid rounds, and
  duplicate workload identities are parse errors.
- Missing contract-v2 metadata is unverified, not compatible.
- No incompatible or unverified row is called stable, improved, or regressed.

### Expression failures

- Existing error text and provider categories remain stable.
- Invalid expressions fail during compile validation and are not cached.
- Cache eviction changes performance only; it cannot change results or errors.
- Prepared-provider failures, if the optional protocol is activated, fail plan
  compilation rather than first execution.

### Ownership failures

- Allocation failures propagate through the existing provider error boundary.
- There is no mutable-view fallback.
- Unsupported dtypes fail before copying or arithmetic dispatch.

## Verification design

### Deterministic tests

Focused tests prove:

- one cache miss per unique successful expression;
- bounded cache eviction and repeated invalid-expression validation;
- unchanged unsafe syntax, bounds, intermediate-result, and error behavior;
- caller isolation and an irrevocably non-writeable NumPy base chain;
- contiguous, non-contiguous, transposed, Fortran-order, negative-stride,
  scalar, empty, subclass, view, and every supported dtype input;
- exact metadata, dtype, shape, and row count;
- complete deterministic benchmark identities and fingerprints;
- all compatibility statuses and issue codes;
- duplicate identity and malformed report rejection;
- legacy report display without classification;
- unchanged provider GC, rollback, cancellation, and reuse behavior.

Timing thresholds are never asserted in unit tests.

### Performance method

The exact source baseline is
`b42b687c8021291bcb537a57fded7a40f7f8477d`. Baseline and candidate are built
in separate worktrees with one resolved dependency version set. Runs alternate
baseline and candidate order on the same idle host with CPU-only JAX.

Every claim reports absolute time, ratio, coefficient of variation, machine
fingerprint, dependency fingerprint, and workload fingerprint. Reports with a
compatibility issue do not enter the summary. Noisy cases remain informational
and are rerun rather than silently averaged into a conclusion.

Phase 1 acceptance requires:

- no compatible scenario slower by more than 5%;
- at least 20% geometric-mean improvement across the six comparable NumPy/JAX
  `plan_end_to_end` overhead cases;
- at least 30% improvement in NumPy `batch_ownership` at 100,000 and 1,000,000
  elements;
- statistically unchanged `backend_kernel` controls;
- complete benchmark-contract metadata in every generated report.

CI benchmarks remain informational until at least 20 compatible main-branch
samples exist, as required by the benchmark documentation.

### Full repository gates

Before completion, run every command group from `AGENTS.md`:

- Rust format, Clippy, workspace tests, 90% line coverage, and rustdoc;
- Maturin development install, CPU-only Python tests, and Ruff;
- Studio backend coverage;
- frontend API sync, build, unit tests, Playwright, and production audit;
- Rust supply-chain and release-helper tests;
- generated schema/OpenAPI/TypeScript no-diff checks;
- `git diff --check`.

No generated `python/calc_flow/_native*.so` may remain in source.

## Delivery boundaries

The implementation should use narrow commits in this order:

1. benchmark contract and strict comparison;
2. layered array benchmark coverage;
3. bounded expression parse cache;
4. one-copy NumPy ownership;
5. Phase 1 performance evidence;
6. optional provider preparation only if its midpoint gate passes;
7. optional execution-core measurement only if Phase 1 misses its final gate.

Execution-core behavior is not changed by this design. A measured core change
requires a new design and explicit approval.
