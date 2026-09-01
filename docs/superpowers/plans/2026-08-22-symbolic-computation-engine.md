# Symbolic Computation Engine Implementation Plan

> **Historical status:** SCE-00 was approved on 2026-08-22. No downstream
> implementation task became complete merely because its contract was frozen
> here.
>
> **Historical baseline:** `main@f6b8a6f90b7a978de1976f5a163ea689b989caee`
> after PR #166.
>
> **Implementation snapshot:** The SCE-01 through SCE-16 milestone changes are
> merged through `main@0bac2b01cf4ea9793859976bfe4ecfe8074581af` on
> 2026-09-01. Follow-up hardening now lowers stateful expressions as a
> deterministic innermost-first DAG: row-local operands materialize before
> rolling or cross-section state, rolling output may feed a later rolling
> stage, and compatible multi-stage branches share physical state. Symbolic
> mid-checkpoint recovery and Finance-Python-inspired RSI composition cover
> the new boundary. SCE-16 subsequently freezes and implements unadjusted
> EMA/EWMA, durable rolling state layout v2, and MACD composition.
>
> **Design contract:**
> [Symbolic Computation Engine Design](../specs/2026-08-22-symbolic-computation-engine-design.md),
> [semantic freeze](../../../.codex/artifacts/specs/symbolic-computation-contract.md),
> [API contract](../../../.codex/artifacts/api-notes/symbolic-computation-engine.md),
> and [approved critique](../../../.codex/artifacts/critiques/symbolic-computation-engine.md).
>
> **Scope:** Deliver a Python symbolic declaration and compiler layer that
> delegates all data calculation to calc-flow's existing or approved native
> table, array, batch, stream, state, checkpoint, and runner capabilities.

## 1. Protected Baseline

Every task starts from these repository invariants and must not weaken them:

- `Batch` remains the immutable table/array graph and runner envelope.
- Apache DataFusion remains the table engine; the symbolic package is not a
  second table implementation.
- NumPy/JAX remain optional, explicitly registered external providers.
- `ProjectDocument` and generated project-v3 schema remain the canonical
  data-only executable contract.
- The Python package remains a PyO3 binding plus functional adapters, not a
  second engine.
- Public stream control construction, task/coordinator internals, and reaper
  ownership remain private.
- Managed stream recovery continues to use strict v3 manifests and
  `LocalStateBackend` segments.
- Caller-owned mappings, sequences, Arrow tables, arrays, and JSON values are
  never mutated.
- Workspace lints, Rust 2024, `unsafe_code = "forbid"`, the 90% Rust coverage
  floor, and the Studio 85% backend floor remain unchanged.
- Historical `tests/fixtures/v1/` and the removed Python v1 implementation are
  not compatibility targets.

## 2. Delivery Strategy

Each task has one issue, one `feature/<description>` branch, and one PR. A PR
must be independently useful, leave `main` green, contain no placeholder
module, and update generated artifacts in the same commit as their source
contract.

Each behavior-changing PR follows this sequence:

1. Add a focused failing test and record the expected failure.
2. Implement the narrowest complete behavior.
3. Add negative/error-path tests.
4. Add batch/stream equivalence tests when both modes are supported.
5. Add checkpoint/recovery tests for stateful behavior.
6. Run the complete command group for every affected surface.
7. Run paired benchmarks when the data path changes.

The initial development branch is `feature/symbolic-engine`. Stateful project
format work may use a short-lived integration branch only if several stacked
PRs must update the same generated schema/OpenAPI files. The integration branch
is deleted after the atomic cutover; it is not a permanent release branch.

## 3. Phase and Task Map

| Issue    | Phase | Delivery                                  | Depends on             | Single-track weeks |
| -------- | ----- | ----------------------------------------- | ---------------------- | ------------------ |
| [SCE-00] | P0    | freeze symbolic semantic decisions        | —                      | 1.0                |
| [SCE-01] | P0    | capture table, rolling, and matrix bases  | SCE-00                 | 0.5                |
| [SCE-02] | P1    | immutable expression IR and namespaces    | SCE-00                 | 1.0                |
| [SCE-03] | P1    | program, type/domain/state analysis       | SCE-02                 | 1.0                |
| [SCE-04] | P2    | lifecycle-aware runtime capabilities      | SCE-00                 | 1.0                |
| [SCE-05] | P2    | fused row-local batch/stream lowering     | SCE-03, SCE-04         | 1.5                |
| [SCE-06] | P3    | rolling row windows with lag/delta        | SCE-05                 | 2.0                |
| [SCE-07] | P3    | rolling numeric aggregates and sharing    | SCE-06                 | 2.0                |
| [SCE-08] | P3    | duration windows, covariance, correlation | SCE-07                 | 2.0                |
| [SCE-09] | P4    | cross-section rank, percentile, z-score   | SCE-05                 | 2.0                |
| [SCE-10] | P4    | cross-section winsorization               | SCE-09                 | 1.5                |
| [SCE-11] | P5    | immutable static stream inputs            | SCE-05                 | 2.0                |
| [SCE-12] | P5    | stateless stream provider lifecycle       | SCE-04, SCE-11         | 1.5                |
| [SCE-13] | P5    | symbolic table/array bridges and matmul   | SCE-03, SCE-12         | 1.5                |
| [SCE-14] | P6    | cross-domain CSE, fusion, explain, cache  | SCE-08, SCE-10, SCE-13 | 2.0                |
| [SCE-15] | P6    | Studio, docs, hardening, release checks   | SCE-14                 | 2.0                |
| [SCE-16] | P7    | durable EWMA and composed MACD            | SCE-08, SCE-14         | 1.0                |

### Current Delivery Snapshot

| Tasks                 | State       | Merged evidence                                                     |
| --------------------- | ----------- | ------------------------------------------------------------------- |
| SCE-00–SCE-04         | merged      | contract `b9deff7`, IR `72f8ab0`, analysis/runtime through `fab89de` |
| SCE-05–SCE-08         | merged      | row/rolling through `b53b8e9`; paired gates `b39208f`, `7d9a2b8`    |
| SCE-09–SCE-10         | merged      | cross-section `9affe0a`, grouped additions `7389c6f`                 |
| SCE-11–SCE-13         | merged      | static/provider/matrix through `bde42be`                             |
| SCE-14                | merged      | optimizer, explain, and cache `5fd256d`                              |
| SCE-15                | merged      | Studio, docs, and release integration `94b9f6e`                      |
| SCE-16                | merged      | EWMA contract, layout v2, and MACD `0bac2b0` (#221)                   |

This table records implementation history rather than changing the frozen
semantics below. Later correctness or composability follow-ups receive their
own RED tests and review evidence.

The single-track median is about 24.5 engineer-weeks. Two engineers split
between native streaming/state work and Python/compiler/provider work can
target 11 to 14 calendar weeks after SCE-00, subject to review and CI latency.
These are planning estimates, not delivery promises.

## 4. Phase P0: Semantic Freeze and Baselines

### [SCE-00] Freeze the Symbolic Contract

**Status:** Approved on 2026-08-22 after one blocker-only correction round.

**Branch:** `feature/symbolic-contract`

**PR title:** `docs: define symbolic computation contracts`

**Goal:** Approve or revise the paired design record before any public API,
project variant, runner parameter, or durable state layout is implemented.

**Required decisions:**

- public namespaces, expression types, and `Program` compile boundaries;
- canonical node identity and symbolic `==` behavior;
- Arrow/Array API type promotion and null/NaN handling;
- row-count and duration-window interval semantics;
- event-time/sequence ordering and duplicate-time behavior;
- cross-section group finality, tie rules, and output order;
- final-only late-event policies;
- `RollingSpec` and `CrossSectionSpec` project shapes;
- runtime capability schema revision;
- static input ownership, digest, and recovery behavior; and
- floating-point equivalence tolerances per primitive.

**Files:**

- this design and implementation-plan pair;
- `.codex/artifacts/specs/symbolic-computation-contract.md`;
- `.codex/artifacts/api-notes/symbolic-computation-engine.md`; and
- `.codex/artifacts/critiques/symbolic-computation-engine.md`.

**Exit gate:** No unresolved decision affects serialized data, public API,
durable state, output finality, or recovery. The document continues to state
that no implementation exists.

**Disposition:** Satisfied. The semantic and API contracts are frozen, the
critique is approved, and downstream issues remain responsible for every
runtime, operator, lowerer, runner, schema, and execution change.

### [SCE-01] Capture Baseline Performance

**Branch:** `feature/symbolic-baselines`

**PR title:** `perf: capture symbolic execution baselines`

**Goal:** Measure equivalent hand-built calc-flow plans before the compiler or
new stateful operators can influence results.

**Files:**

- `benchmarks/test_symbolic_baseline.py`;
- `crates/calc-flow/benches/rolling_baseline.rs` if native rolling prototypes
  are required for algorithm selection; and
- `benchmarks/README.md` for commands and result interpretation.

**Scenarios:**

1. One DataFusion projection producing 20 derived numeric columns.
2. Interleaved entities with 20-row and 60-row temporal features.
3. Cross-section rank/z-score over controlled complete groups.
4. NumPy/JAX `table_matmul` with representative shapes.
5. Existing stream checkpoint throughput with similarly sized operator state.

**Metrics:** rows/s, batches/s, peak RSS, provider calls, Arrow/dense copy
bytes, checkpoint bytes, checkpoint duration, and recovery duration.

**Exit gate:** Reproducible paired-benchmark commands and recorded noise. No
absolute performance claim is introduced.

## 5. Phase P1: Immutable IR and Composer

### [SCE-02] Add Immutable Symbolic Expressions

**Branch:** `feature/symbolic-expressions`

**PR title:** `feat: add immutable symbolic expressions`

**Goal:** Deliver a useful calculation-declaration surface without lowering
or executing data.

**Files:**

```text
python/calc_flow/symbolic/__init__.py
python/calc_flow/symbolic/expr.py
python/calc_flow/symbolic/nodes.py
python/calc_flow/symbolic/types.py
python/calc_flow/symbolic/domains.py
python/calc_flow/symbolic/windows.py
python/calc_flow/symbolic/ops.py
python/tests/test_symbolic_expr.py
python/tests/test_symbolic_types.py
python/tests/test_symbolic_domains.py
```

**RED tests:**

- constructors do not mutate caller mappings/sequences;
- callable/non-JSON attributes are rejected;
- canonical digest ignores mapping insertion order but retains semantic output
  order;
- table and array values cannot mix implicitly;
- symbolic `and`/`or` misuse produces an actionable error;
- `identical()` distinguishes structure without evaluating symbolic equality;
- no public `eval`, `push`, `value`, or `transform` member exists.

**Implementation:** Frozen/slotted wrappers, immutable node arguments, strict
frozen JSON attributes, versioned `OpRef`, row/temporal/cross-section/window/
tensor namespaces, explicit row/time windows, and deterministic formatting.

**Exit gate:** Expressions compose and explain their declarations without any
data object or runtime calculation path.

### [SCE-03] Add Programs and Static Analysis

**Branch:** `feature/symbolic-program-analysis`

**PR title:** `feat: analyze symbolic programs`

**Goal:** Infer everything required to make a safe lowering decision.

**Files:**

```text
python/calc_flow/symbolic/program.py
python/calc_flow/symbolic/analyzer.py
python/calc_flow/symbolic/errors.py
python/tests/test_symbolic_program.py
python/tests/test_symbolic_analysis.py
```

**RED tests:**

- duplicate input/output/feature names fail with stable paths;
- field references preserve the exact input schema dependency;
- array rank and symbolic dimensions propagate through matmul;
- table-derived arrays retain row-axis lineage;
- incompatible attachment is rejected;
- stream mode rejects unbounded state and missing event-time/sequence facts;
- `analyze()` and `explain()` require an explicit `Runtime` and capture one
  immutable capability session/revision snapshot;
- error paths begin at the named program output;
- analysis results are immutable and deterministic.

**Implementation:** `table_input`, `parameter`, `FeatureSet`, `Program`, value
type/domain/lineage inference, state requirement propagation, stream-safety
analysis, one explicit immutable `Runtime` capability snapshot per analysis,
and `explain()` facts. Do not add a project lowerer in this task.

**Exit gate:** A declaration fingerprint remains runtime independent. A
complete program can be analyzed and explained against the explicitly supplied
`Runtime` capability snapshot without accepting data or executing a job.

## 6. Phase P2: Lifecycle Capabilities and Row-Local MVP

### [SCE-04] Expose Lifecycle-Aware Capabilities

**Branch:** `feature/lifecycle-capabilities`

**PR title:** `feat: expose operator lifecycle capabilities`

**Goal:** Prevent symbolic lowering from selecting a primitive/provider in a
runtime mode it cannot execute safely.

**Files:**

- `python/calc_flow/capabilities.py`;
- `python/calc_flow/pipeline.py`;
- `crates/calc-flow-python/src/config.rs`;
- `python/calc_flow/_native.pyi`;
- focused Python/PyO3 tests; and
- Studio models/generated artifacts if the capability contract is transported
  through `/api/v3`.

**Capability additions:** modes, stateful, micro-batch invariant, watermark
requirement, checkpoint support, determinism, and replay safety. Provider
registrations initially remain batch-only unless a real stream factory exists.

**RED tests:** lifecycle omission, missing stream factory, immutable snapshot,
session/revision consistency, deterministic provider ordering, and defensive
options-schema copies.

**Exit gate:** Existing runtime registrations remain source compatible, and
the capability surface truthfully reports only implemented lifecycles.

### [SCE-05] Lower and Fuse Row-Local Expressions

**Branch:** `feature/symbolic-row-lowering`

**PR title:** `feat: compile symbolic row expressions`

**Goal:** Produce the first end-to-end batch/stream symbolic feature with no
native stateful-operator additions.

**Files:**

```text
python/calc_flow/symbolic/optimizer.py
python/calc_flow/symbolic/lower.py
python/tests/test_symbolic_lowering.py
python/tests/test_symbolic_batch.py
python/tests/test_symbolic_stream.py
```

**Supported primitives:** literal, field, arithmetic, comparison, boolean,
`where`, coalesce, log, exp, sqrt, abs, clip, cast, select, filter, and
`with_columns`.

**RED tests:**

- a 20-output `FeatureSet` lowers to one fused expression node when safe;
- structurally equal subexpressions are calculated once;
- generated node IDs and project fingerprints are deterministic;
- project configuration contains strict JSON only;
- batch output equals stream output for 1-, 7-, and 1,000-row segmentations;
- an attempted stream aggregate/SQL window is rejected rather than silently
  made batch-local; and
- the Rust compiler still owns final port/schema/topology validation.

**Performance gate:** Generated row-local throughput is no more than five
percent below the equivalent hand-built plan under the paired baseline, and
runtime symbolic Python calls are zero.

**Milestone gate A:** A releasable stateless symbolic MVP exists. Do not start
serialized stateful operators if this gate is not satisfied.

## 7. Phase P3: Temporal Rolling Operators

### [SCE-06] Add Row-Window Lag and Delta

**Branch:** `feature/rolling-lag-delta`

**PR title:** `feat: add rolling lag and delta`

**Goal:** Establish one complete native rolling vertical slice including
batch, stream, project, checkpoint, Python lowering, and generated contracts.

**Core files:**

- `crates/calc-flow/src/operator/rolling.rs`;
- `crates/calc-flow/src/operator/mod.rs`;
- `crates/calc-flow/src/config.rs`;
- `crates/calc-flow/src/lib.rs` as required for public types;
- project schema/OpenAPI/TypeScript generated artifacts;
- Python builder/stub/lowerer updates; and
- Rust/Python integration tests.

**Initial spec:** partition columns, event-time column, sequence columns,
ordered output declarations with primitive version, lag/delta function, input
field, output field, and positive `periods`.

**RED tests:** duplicate names, invalid periods, missing exact schema,
unsupported type, entity interleaving, duplicate timestamps, segmentation
invariance, watermark progress, checkpoint/restore/reset, output order, and
project canonicalization.

**Exit gate:** Batch and final-only stream produce the same ordered lag/delta
rows across segmentation and recovery.

### [SCE-07] Add Shared Numeric Rolling State

**Branch:** `feature/rolling-aggregates`

**PR title:** `feat: add rolling numeric aggregates`

**Goal:** Add count, sum, mean, min, max, variance, and standard deviation while
sharing history for compatible outputs.

**Implementation:** compact per-entity row history, reversible count/sum,
stable add/remove variance state, min/max monotonic queues, batched Arrow output
builders, exact null/NaN/minimum-period rules, and immutable snapshot segments.

**RED tests:** overflow, all-null windows, NaN policy, insufficient samples,
multiple columns, multiple compatible outputs, high-cardinality entities,
state checksums, restore compatibility, and cancellation during output.

**State gate:** Measured resident state and checkpoint size grow linearly with
active entities and retained rows. No Python object is allocated per entity.

### [SCE-08] Add Duration Windows and Correlation

**Branch:** `feature/rolling-time-correlation`

**PR title:** `feat: add rolling time correlations`

**Goal:** Complete the initial temporal catalog.

**Implementation:** duration-window support for the delivered rolling
primitives, bounded reorder buffers, open/closed interval rules,
covariance/correlation, allowed lateness, error/drop policies, and
deterministic late-row metrics.

**RED tests:** interval boundaries, watermark at exact boundary, bounded
out-of-order arrival, too-late rows, zero variance, numerical stability,
checkpoint with unfinalized reorder state, and randomized batch/stream parity.

**Milestone gate B:** Temporal outputs are segmentation invariant, recovery
safe, bounded, and within the performance envelope approved from SCE-01.

## 8. Phase P4: Cross-Section Operators

### [SCE-09] Add Rank, Percentile, and Z-Score

**Branch:** `feature/cross-section-core`

**PR title:** `feat: add cross-section transforms`

**Goal:** Establish complete-group batch/stream semantics with a shared native
kernel.

**Core files:**

- `crates/calc-flow/src/operator/cross_section.rs`;
- operator/config/lib exports;
- project schema/OpenAPI/TypeScript generated artifacts;
- Python lowering; and
- focused Rust/Python tests.

**Initial functions:** rank, percentile, z-score, and demean. Configuration
includes event time or bucket, ordered group columns, sequence columns, tie
method, direction, null placement, minimum samples, and population/sample
standard-deviation choice.

**RED tests:** one group split across batches, several groups in one batch,
interleaved groups, ties, nulls, incomplete group before watermark, final
emission after watermark, half-built group checkpoint, and deterministic row
order.

### [SCE-10] Add Cross-Section Winsorization

**Branch:** `feature/cross-section-winsorization`

**PR title:** `feat: add cross-section winsorization`

**Goal:** Add winsorize while sharing the existing grouping and sort pass.

**RED tests:** industry/group partitioning, exact versus bucketed event time,
multiple outputs, late-event error/drop, released state after watermark,
checkpoint/recovery, and output-schema collisions.

**Milestone gate C:** No result depends on micro-batch boundaries, closed
groups release state promptly, and batch/stream tie/null rules are identical.

## 9. Phase P5: Matrix Streaming and Static Parameters

### [SCE-11] Add Immutable Static Stream Inputs

**Branch:** `feature/static-stream-inputs`

**PR title:** `feat: add static stream inputs`

**Goal:** Supply immutable weights/configuration batches once per job without
pretending they are infinite sources.

**Frozen public contract:** Implement the exact additive, keyword-only
`StreamingRunner(..., static_inputs=...)` signature from the approved API note;
do not reopen its name, ownership, digest, or recovery semantics here.

**Implementation areas:** stream plan external-input descriptors, whole-job
preflight, runner/source coverage, operator startup, job/checkpoint lineage,
Python ownership, status/error projection, and Studio worker restoration.

**RED tests:** missing/extra input, wrong kind/schema/backend, caller mutation,
operator visibility before sources open, input released once, restore with a
different digest, cancellation/startup failure, and no secret/raw payload in
status or errors.

**Exit gate:** Static values are validated/latched once, never sent as repeated
stream data, and cannot change across durable recovery.

### [SCE-12] Add Stateless Stream Provider Factories

**Branch:** `feature/stateless-stream-providers`

**PR title:** `feat: add stateless stream providers`

**Goal:** Register explicitly safe vectorized providers for both lifecycles.

**Implementation:** a lifecycle-specific stream factory, exact port contracts,
`microbatch_invariant` declaration, no checkpoint state, bounded async/GIL
boundary, one callback per fused segment, and capability reporting. Built-in
NumPy/JAX registrations opt in only after their operation is proved row-axis
independent.

**RED tests:** batch-only provider selected by stream plan, false capability
claim, wrong mapping ports, callback error/cancellation, provider output
immutability, replay behavior, and exactly-once rejection for nondeterministic
providers.

### [SCE-13] Add Table/Array Bridges and Static Matmul

**Branch:** `feature/symbolic-matrix-streaming`

**PR title:** `feat: compile symbolic matrix expressions`

**Goal:** Complete explicit table-to-array-to-table compositions for batch and
stream.

**Functions:** `linalg.from_columns`, allowlisted/fused elementwise array
expressions, static-weight `matmul`, and `table.attach_columns`.

**RED tests:** selected-column order, row-axis lineage, rank/shape/output-width
errors, safe dtype promotion, JAX x64 behavior, row-count mismatch, cross-
backend rejection, provider call count, copy-byte metrics, weight placement
once, and segmentation-invariant results.

**Milestone gate D:** NumPy/JAX calls occur once per fused segment and
micro-batch, weights are transferred once per job, and matrix outputs retain
the approved batch/stream tolerance contract.

## 10. Phase P6: Optimization, Studio, and Release

### [SCE-14] Add Cross-Domain Optimization and Explain

**Branch:** `feature/symbolic-optimization`

**PR title:** `perf: fuse symbolic execution stages`

**Goal:** Optimize complete programs after all initial domains exist.

**Implementation:** program-wide CSE, projection/filter pushdown, compatible
rolling-state sharing, cross-section grouping/sort sharing, array fusion,
materialization-boundary selection, deterministic compile cache, and explain
facts for state/copy/provider costs.

**RED tests:** rewrite equivalence, no unsafe filter movement across temporal
or cross-section boundaries, provider/version-sensitive cache keys, runtime
revision invalidation, deterministic explain output, and cache value
immutability.

**Performance gate:** Repeat every SCE-01 scenario in paired order. A
statistically meaningful regression blocks release until explained and
approved; benchmark noise alone does not.

### [SCE-15] Integrate Studio, Documentation, and Release Gates

**Branch:** `feature/symbolic-release`

**PR title:** `docs: publish symbolic computation workflows`

**Goal:** Make the approved surface operable and verifiable without adding a
second Studio compiler.

**Studio scope:** Display source expressions, lowered calc-flow nodes,
stateful/watermark/static-input annotations, estimated state, provider
identity, and table/array copy boundaries. Studio submits/executes the strict
lowered ProjectDocument only.

**Documentation:** Python API, examples, batch workflow, continuous workflow,
checkpoint recovery, NumPy/JAX static weights, composed financial indicators,
capability failures, and performance interpretation.

**Release verification:** Complete repository commands, schema drift checks,
wheel/source/crate/Studio package inspection, clean-environment installs, and
smoke checks required by `AGENTS.md`.

**Milestone gate E:** Public documentation describes only implemented
behavior; no proposed member is advertised early; all generated contracts are
clean; package contents contain no repository-only plan/spec guidance.

### [SCE-16] Add Durable EWMA and Composed MACD

**Branch:** `feature/symbolic-exponential-indicators`

**PR title:** `feat: add symbolic exponential indicators`

**Goal:** Add the deferred exponential primitive without creating a Python
execution path or a separate MACD accumulator.

**Semantic contract:** `ewma@1` uses the unadjusted recurrence with
`alpha = 2 / (span + 1)`, exact first-valid-sample seeding, ignored null/NaN
inputs, and a valid-sample `min_periods` gate. `ts.ema` is an identity alias;
`ts.macd` expands to the difference between fast and slow EWMA nodes.

**Durable state:** Existing rolling declarations retain state layout v1.
Declarations containing EWMA use layout v2 and persist one valid count and
exact IEEE binary64 accumulator per shared `(entity, input, span)` group.
Restore validates layout, schema, configuration, entity ordering, group kind,
ordinal uniqueness, and state row completeness before installation.

**RED and reference tests:** Cover declaration rejection, type analysis,
project lowering, first-sample and missing-value vectors independently derived
from Finance-Python commit `3e33d3e`, entity isolation, shared state,
segmentation invariance, batch/stream parity, and mid-checkpoint recovery with
no retained-history reconstruction.

**Performance gate:** Compare the public symbolic EWMA/MACD plan with the same
native project-v3 rolling declaration in alternating paired order. Report the
distribution and provenance; do not treat an unpaired timing as acceptance.

## 11. Cross-Cutting Test Matrix

Every supported stateful primitive is exercised over:

```text
same bounded input
  x batch / final-only stream
  x single / small / large micro-batches
  x ordered / bounded out-of-order arrival
  x uninterrupted / checkpoint-recovered execution
  x no-null / null / NaN values
  x one / many / high-cardinality groups
```

Additional fixed properties are:

- ProjectDocument contains strict data only.
- Symbolic construction never mutates caller inputs.
- Runtime data never flows through the symbolic package.
- Control messages remain runner-owned.
- Late-row, state-byte, checkpoint-byte, provider-call, and copy-byte metrics
  are deterministic.
- Floating calculations follow frozen tolerances; schemas, names, row order,
  null placement, and delivery status are exact.
- Restore rejects incompatible operator versions, schemas, state layouts, and
  static-input digests before sources resume.

## 12. Verification by Surface

Documentation-only tasks run at least:

```bash
git diff --check
```

Python symbolic tasks run:

```bash
uv run maturin develop
JAX_PLATFORMS=cpu uv run pytest python/tests -q
uv run ruff check .
uv run ruff format --check .
git diff --check
```

Native operator/runtime tasks also run:

```bash
cargo fmt --all --check
cargo clippy --workspace --all-targets --all-features -- -D warnings
uv run python scripts/run_rust_tests.py
RUSTDOCFLAGS="-D warnings" cargo doc --workspace --all-features --no-deps
```

Rust coverage runs with the required Kafka, PostgreSQL, and ClickHouse service
environment from `AGENTS.md` and retains the 90% workspace line floor.

Project/API/Studio changes additionally run backend coverage, `npm run
sync:api`, frontend build/unit/e2e tests, audit, and exact generated-artifact
drift checks. Release tasks run every maintained command and package inspector
listed in `AGENTS.md`.

## 13. Pull Request Contract

Each PR title uses an approved prefix and remains under 70 characters. Its
body contains:

```markdown
## Summary

- semantic outcome
- affected public/serialized/durable boundaries
- explicit non-goals

## Test plan

- RED test and expected failure
- focused verification
- complete surface commands
- paired benchmark or reason no data-path benchmark applies
```

PRs preserve unrelated worktree changes, stage only intended files, and use
imperative commit summaries under 72 characters. Stateful PRs include durable
layout/version review and recovery evidence. Serialized-contract PRs include
their schema/OpenAPI/generated TypeScript changes in the same commit.

## 14. Deferred Work

These items require separate approved designs after the initial release:

- streaming relational or temporal joins;
- update/retraction/changelog outputs;
- epoch-aligned dynamic matrix weights;
- arbitrary user-defined stateful providers;
- sparse arrays and distributed device placement;
- cross-row array reductions without explicit windows;
- a portable serialized formula document; and
- compatibility aliases for external symbolic libraries.

Cross-section top/bottom selection and mean fill were originally deferred but
were subsequently approved and delivered by SCE-10; they are no longer
deferred work.

The absence of these features does not justify a Python execution fallback.
Unsupported compositions fail during symbolic analysis or native compilation.
