# Branch Review: Inline Grouped Integer SUM State

**Branch:** `feature/sum-inline-state` → `main`

**Base:** `222f4e2d88f56b5b4ce635abab24eceed732da6d`

**Reviewed source:** local checkpoint
`d0bbca1bf92b0ae4a2a8b44539f1aabecbe8b721`, tree
`5f45ac22c52623f6478feb1a5dfb613e47bf3fc1`.
The author committed fourteen Rust files; the worktree was clean at handoff.

**Reviewed tools:** `b3fbd09aee2136498e27334c8ead35e35f4d0cef`,
following `259ad4e3`. Specification and CI registration are in ancestor
`47f5bcbf`. These are local verification identities, not an assumed public
candidate or measured product.

**Upstream:** [Study specification](../specs/sum-inline-state-20261105.md)
and the previous experiment's
[next bounded hypothesis](../../../benchmarks/sum-state/sum-batch-parallel-20261104.md#next-bounded-hypothesis).

**Files changed against base:** 20, including this review.

No PR/check-run snapshot exists for this source handoff. The parent owns
publication, final CI verification, and merge.

## Summary

**Premeasurement SOURCE/tool PASS.** Exactly one eligible grouped integer
SUM stores one compact typed nullable state and derives its result from it.
A plan-bound cache removes repeated field-metadata allocation. All other
aggregate shapes retain generic state/result vectors.

The reviewed implementation preserves checkpoint encoding, output acceptance,
and transactional rollback. Focused allocation evidence establishes a
component improvement; no end-to-end speedup or delivery acceptance is
claimed. Public candidate identity, fresh candidate build, actual sampling
gate, performance acceptance, and final required remote checks remain pending.

## Source and Domain Review

`bind_inline_sum` runs once after existing physical plan binding and native
strategy selection. Admission requires nonempty grouping keys, exactly one
actual trusted built-in non-distinct unordered SUM, and Int64/UInt64 result
type. Global SUM, mixed/multiple aggregates, float/decimal, distinct, and
custom implementations do not enter this representation.

The Copy cache contains only a typed NULL template; it never stores a
committed sum. Candidates copy that template and use the existing native
grouped accumulator guarantee. Partial scatter extracts the sole Arrow state
scalar directly, without a transient state vector. Integer arithmetic and
history reduction remain in the existing DataFusion path.

`InlineSum` preserves signedness and nullable Option values. Typed NULL
remains distinct from zero. `StateView` owns only a fixed-width stack scalar
for inline groups and borrows generic state slices. Generic export/result
inspection uses borrowed Cow values, preserving variable-width clone
behavior. Final output converts the fixed scalar to owned Arrow data;
held earlier outputs remain immutable.

Only candidate state changes before output acceptance. Existing global slot
order, WHERE versus FILTER group existence, full snapshots, cancellation
checks, dirty tracking, capture ownership, and commit order are unchanged.
Dropping a transaction, refusing output, or cancellation cannot install the
new state. Global-record/coalescer paths remain explicitly Generic.

Group fields still release key/value/aggregate storage before their
reservation. Conservative aggregate fees are retained. Group, Candidate,
map slots, container capacity, and imports continue using the existing
dynamic size-based charges. No public API, project schema, dependency,
checkpoint-policy version, state-layout version, or new worker model is
introduced.

## Design Findings Closed Before Measurement

1. **Full ScalarValue storage enlarged unrelated containers.** The initial
   full-scalar enum was 64 bytes versus the old two-Vec payload's 48.
   Group/Candidate/map-tuple sizes would become 112/144/160 rather than
   96/128/136 bytes. The chosen compact typed integer payload restores the
   old sizes. That discarded design was not benchmarked.
2. **A held-allocation probe omitted container capacity.** Its retained
   candidate Vec was prepared outside the counter. Target-relative old-layout
   assertions and a second probe now include actual candidate-map, group
   buffer, index, and allocation peaks. Generic positive controls cover
   structural size as well as owned state allocation.
3. **Lower held counts hid more transient allocations.** The uncached draft
   reduced held allocations from 1,024 to 256 for 256 groups but increased
   total allocations from 1,792 to 2,560. DataFusion 55 `field()` clones and
   renames field metadata. Two extra calls per candidate caused the increase.
   An actual total-count RED preceded the immutable plan-cache fix.
4. **The nullable fixture changed its schema between phases.** Arrow
   `try_from_iter` inferred nullability from current null counts. Explicit
   stable nullable fields fix the fixture. Refusal now requires the injected
   `inline-reject` node rather than any error. The two schema-fixture
   failures are not production regressions or valid behavior RED evidence.
5. **Invalid profiles could be reported accepted.** Two invalid decoded
   records now remain in raw evidence while accepted count is zero,
   disposition is nonaccepted, and completed acquisition is revoked.
6. **Python helper drift lacked enforcement.** Frozen SHA-256 inventory is
   checked before setup, after native post-seals, and before/after
   publication. Drift demotes durable source/published/raw/manifest receipts,
   preserves prior failure, and records observed helper hashes.

No performance trial or expanded case set was used to select these fixes.

## Allocation, Footprint, and Funding Evidence

The synchronous probe updates 256 existing-key candidates through actual
candidate construction, partial scatter, and finalization. Fixture creation,
native partial arrays, and the retained candidate buffer are outside this
first counter. All counted candidates stay alive through its end.

| Metric, 256 groups | Legacy | Reviewed |
|--------------------|--------|----------|
| Total allocations  | 1,792  | 256      |
| Held allocations   | 1,024  | 256      |
| Held bytes         | 104448 | 4096     |

The final mixed SUM+COUNT control retains 3,840 total allocations, 1,536 held
allocations, and 145,408 held bytes, unchanged at this boundary. These counts
describe this component, not whole-job allocations, latency, or RSS.

Target-local assertions compare against the old-equivalent layout, rather
than assuming Linux constants across CI targets. On this build:

| Layout                         | Legacy bytes | Reviewed bytes |
|--------------------------------|--------------|----------------|
| Aggregate state/result payload | 48           | 48             |
| Group                          | 96           | 96             |
| Candidate                      | 128          | 128            |
| Candidate map tuple            | 136          | 136            |

The complete test plan occupies 992 bytes, within the existing 4,096-byte
structural floor; the fixed cache adds no heap allocation. This does not
lower compact-state fees or establish additional budget capacity.

The separate full staged probe counts container creation and growth inside
the counter. Actual candidate-map/group-buffer/index capacities are
448/256/448.

| Full staged case | Peak owned heap bytes | Prepaid bytes |
|------------------|-----------------------|---------------|
| Inline SUM       | 111648                | 407552        |
| Generic mixed    | 252960                | 452608        |

Both peaks are below their existing reservations. The one-byte-short
candidate-credit test refuses atomically, preserves exported records and
accounting, and admits exactly at the required credit. Dropping all tested
owners releases pool reservations. These are representative component
ownership checks plus unchanged-layout/source proofs, not an engine RSS cap.

## Checkpoint Compatibility

The eligible SUM's native-v3 descriptor, field order, schema, policy identity,
and wire-layout logic are unchanged. Export changes only how a scalar is borrowed or owned.
Import retains existing schema/state/result validation before packing inline
storage; its plan cache is rebuilt from trusted bindings and is not serialized.

The new test compares legacy Vec-equivalent and inline records, including
Arrow IPC byte equality, through the shared unchanged writer. It is
representation-byte equivalence, not a separately generated historical golden
checkpoint. Compatibility evidence combines this check with unchanged
descriptor/policy/wire logic, import validation, continuation, and the existing
codec/capture/recovery controls. No historical fixture is rewritten.

## Build and Test Results

From the development worktree, the author used:

```text
CARGO_TARGET_DIR=<root>/target CARGO_BUILD_JOBS=2 CARGO_INCREMENTAL=0 cargo test --locked -p calc-flow --lib --features benchmark-internals grouped_inline -- --nocapture
CARGO_TARGET_DIR=<root>/target CARGO_BUILD_JOBS=2 CARGO_INCREMENTAL=0 cargo clippy --locked -p calc-flow --lib --tests --example grouped_sum_state_benchmark --features benchmark-internals -- -D warnings
```

- **Rust: passed focused scope.** Final fresh compilation took 193 seconds;
  **8/8** new tests passed in 0.46 seconds. Strict scoped Clippy passed in
  88 seconds. The final result includes the borrowed inline replacement,
  numeric/ordered result helper, and value-changing discarded transaction.
- The same fresh `calc_flow-69fc90115393543c` binary ran **19/19** selected
  existing controls in 2.53 seconds: **27 unique Rust tests total**.
  The controls log retains the exact multi-filter command with
  `--exact --test-threads=2 --nocapture`. Binary SHA-256:
  `c832284c0e381f35e8639eb333fe315cdfa8cb841be735227d7d86728a54ed9e`.
  Its new semantic module appears in the compiler dependency evidence.
- **Python: passed focused scope.**
  `<root>/.venv/bin/python -m unittest scripts.test_sum_inline_state`:
  **14 passed**, 0.019 seconds, after genuine fixed-case/profile and three
  publication/helper-seal RED observations. The earlier eleven tests are
  included in fourteen, not additional unique tests.
- CI registration is reviewed in the existing step after development
  dependency installation and the complexity gate. Its configuration control
  includes the new module, which remains absent from the bare package job.
- Author/root report formatting, schema and unchanged-harness checks,
  whitespace, Ruff normal/format/complexity checks, and changed-function
  Lizard checks passed. Fifty new/helper Rust functions have maximum CCN 7;
  the existing seed/census paths remain at 8/6. New Python driver/tests have
  maximum CCN 6/3 and no new waivers.
- Reviewer read actual final GREEN/controls/Clippy logs under
  `<root>/target/sum-inline-state-{final-green,controls,strict-clippy-final}.log`
  and reviewed frozen source/tools. No reviewer native build, test,
  measurement, repeated Python check, or remote operation was run.
- **New unresolved failures:** none. Full workspace suites, combined 90%
  Rust coverage, connector services, remote Codacy, cross-platform CI, and
  final delivery checks were not run locally or represented as passed.

## Test Coverage

The eight new controls cover single-SUM signed/unsigned prefixes with 8,193
historical all-NULL groups, sparse valid updates, FILTER-only group creation,
WHERE rejection, dense NULL/history merge, empty input, native restore,
wrapping continuation, held outputs, and discarded value-changing updates.
They also cover public output refusal/capture equality/retry, legacy-shaped
IPC equality, global single integer/float SUM coalescer continuation,
layout/container funding, and exact-credit refusal.

The nineteen existing controls include seven codec/resource/corruption tests,
two plan/admission tests, two mixed/integer controls, float/string behavior,
two ordered-LAST controls, actual finalization cancellation, native capture,
and two UTF-8 output/recovery cases. The cancellation query uses exactly one
SUM and exercises the inline path, including native keys.

## Measurement Protocol Review

Tool checkpoint b3fbd09a is approved within its reviewed scope. It reuses the
maintained resource, process-cleanup, oracle, statistics, profile, and artifact
helpers. A fixed ten-file active helper/test inventory is verified; this is
not a claim to seal every transitive import or system dependency. Unused
catalog constants imported by the general reporter do not participate in
these SUM statistics; native builder helpers belong to separate build evidence.

The draft protocol SHA-256 is
`d82589c17a0714d7a3ff4ccc8d58d4b0edc942316e18d2cd45c90f314bce9115`.
Its candidate revision/receipt are still null, so it cannot authorize sampling.

- Three fixed cases: dense one CPU, dense four CPUs `[0,2,4,6]`, sparse one
  CPU. Each has two rounds of ten alternating pairs and one warmup per arm
  per round: 120 formal observations and twelve warmups.
- Required baseline/candidate dense-one-CPU captures run first, six jobs
  each excluding the first. Both need five nonempty processing windows,
  matching binary hashes, exact per-thread timestamp/count alignment within
  10 ns, and no lost/unwind/warning diagnostics. There is no SQL-worker gate
  for this serial representation change.
- One 600-second budget includes preflight, fixtures, startup, warmup,
  independent twenty-one-snapshot oracles, capture/decoding, statistics,
  publication, and cleanup. Builds are separate. No retries, extra cases,
  or favorable resampling are allowed.
- af400 is incompatible with current compiled source. Both products require
  fresh core/example compilation of current222 and reviewed public candidate,
  with common657 instrumentation only in frozen native trees.
- All cases keep one DataFusion partition and default input partitions one.
  Generic/UInt64/nullable/checkpoint latency and finite/hot-key/other thread
  counts remain unmeasured. Layout/allocation/correctness controls do not
  establish their latency equivalence.
- Acceptance needs all evidence complete, exact maintained `improved` on
  dense one or four CPUs, and no selected `regression`. Inconclusive remains
  uncertainty; no requirement that all other intervals exclude +5% is added.
- Late clock/helper failures revoke completed receipts and false acceptance,
  rehash published artifacts, and retain partial/raw evidence. Source/binary
  post-seals and owned cleanup remain mandatory.

The fresh baseline build is separately recorded at 597.861 seconds; no
comparison has run and no runtime improvement is established.

## Blocking Issues

None remain for the reviewed source/tool scope. Publishing the reviewed
source and building its exact frozen candidate may proceed. Actual public
source/build/protocol identity and acquisition authorization still require
their final gate. Performance acceptance and final required CI/Codacy,
coverage, and cross-platform checks remain merge blockers while unresolved.

## Style and Documentation Consistency

No new dependency, unsafe exception, public knob, blanket security/complexity
exclusion, or fee-capacity claim is introduced. The specification describes
compact typed nullable state, preserved generic layout, immutable outputs,
and unchanged checkpoint contracts. Final measured results and application
documentation must be reconciled after the frozen comparison; source-level
allocation evidence must not be presented as an end-to-end speedup.

## Verdict

**Approve — premeasurement SOURCE/tool PASS**, bound to local d0bbca1b and
tool b3fbd09a. The reviewed design findings are closed with actual allocation,
footprint, semantic, recovery, and failure evidence.

This permits public source publication and a fresh exact-candidate build.
It does not authorize a null-candidate protocol, claim performance acceptance,
or establish merge readiness. The final performance and required remote
results must be reviewed before delivery, using this same canonical artifact.
