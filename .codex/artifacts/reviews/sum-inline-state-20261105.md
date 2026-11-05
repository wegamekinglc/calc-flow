# Branch Review: Inline Grouped Integer SUM State

**Branch:** `feature/sum-inline-state` → `main`

**Base:** `222f4e2d88f56b5b4ce635abab24eceed732da6d`

**Measured source:** public
[74a1eaf2183adfd04fcb1aeecd655c6df73ed16f](https://github.com/wegamekinglc/calc-flow/commit/74a1eaf2183adfd04fcb1aeecd655c6df73ed16f),
tree `04b554c37f0afe8e40fa8e821b9c6b808418f558`. It exactly matches the
reviewed source/tool tree after the initial review.

**Final delivery verification checkpoint:** local
`d0c60561f4a5cd5355736722956c6bbf029b7075`, tree
`9310d1a17cd048fcc15cd1061795fa6de3c74caa`, before this review update.
Only evidence and documentation differ from the measured source.

**Original source verification checkpoint:** local
`d0bbca1bf92b0ae4a2a8b44539f1aabecbe8b721`, tree
`5f45ac22c52623f6478feb1a5dfb613e47bf3fc1`.
The author committed fourteen Rust files; the worktree was clean at handoff.

**Reviewed tools:** `b3fbd09aee2136498e27334c8ead35e35f4d0cef`,
following `259ad4e3`. Specification and CI registration are in ancestor
`47f5bcbf`. These are local verification identities; public 74a1eaf2 is the
actual measured candidate.

**Upstream:** [Study specification](../specs/sum-inline-state-20261105.md)
and the previous experiment's
[next bounded hypothesis](../../../benchmarks/sum-state/sum-batch-parallel-20261104.md#next-bounded-hypothesis).

**Files changed against base:** 33, including this review.

The final review precedes opening the delivery PR; no final remote check
result is claimed. The parent owns publication, required CI completion,
and merge.

## Summary

**Final source, performance, and documentation PASS, with required remote
checks still pending.** The frozen comparison meets its acceptance rule:
dense one/four-CPU and sparse one-CPU whole-run wall medians improve by
28.24%, 31.18%, and 13.93%, respectively. Every case has the exact maintained
`improved` verdict from its two rounds, with complete correctness and resource
evidence.

Exactly one eligible grouped integer SUM stores compact typed nullable state
and derives its result from it. A plan-bound cache removes repeated field
metadata allocation. Generic representation, checkpoint encoding, output
acceptance, and rollback remain intact. The benefit is scoped to this frozen
three-case comparison; generic/UInt64/nullable/checkpoint performance remains
unmeasured. Pending CI is not merge readiness.

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
  measurement, repeated Python test, or remote operation was run. Final
  artifact verification only reads the sealed observations and receipts.
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

## Measurement Protocol and Actual Acquisition Gate

Tool checkpoint b3fbd09a is approved within its reviewed scope. It reuses the
maintained resource, process-cleanup, oracle, statistics, profile, and artifact
helpers. A fixed ten-file active helper/test inventory is verified; this is
not a claim to seal every transitive import or system dependency. Unused
catalog constants imported by the general reporter do not participate in
these SUM statistics; native builder helpers belong to separate build evidence.

The actual [frozen protocol](../../../benchmarks/sum-state/sum-inline-state-20261105.protocol.json)
SHA-256 is
`1bb73519f7e11e686fbe562fa6490597e272ccb9bf6fad758faf1f43d4c9f6c4`.
Reviewer approved the filled public-candidate/receipt gate before acquisition.
The earlier null-candidate draft was not used to authorize sampling.

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

Both builds actually recompiled core and example (`fresh=false` in all four
Cargo records), from the expected permanent frozen source paths. Baseline
and candidate source inventories contain 1,004 and 1,008 entries. Live files,
source-before maps, actual compiler records, copied dependency files, and
public-source equivalence were verified. The candidate core depfile includes
the new production `group_aggregates.rs` module.

Both products use identical compiler/lock/features/flags/configuration and
the common657-only harness overlay. Actual identities are:

- Baseline 222f4e2d binary SHA-256
  `c99c5bb268f320d71f4efb957d2ac4a709ddc231751536836f1d64a9c0e2c83b`;
  build receipt SHA-256
  `3f2dba8cac80662e357305e10adeb4af089e084e28929a4a50775214ffb3b7d2`.
- Public candidate 74a1eaf2 binary SHA-256
  `8378ba7b8683780868604ce70568aa626f61e0abcd1742ec84f050b95aef1ed3`;
  build receipt SHA-256
  `49f7a0922ced5b9a599de5701a1ecb5b996f632ebb480d813b2ccf2702ff5feb`.

Build times are 597.861 and 548.642 seconds, with separately recorded build
verification. They are outside the acquisition budget. Mandatory live quiet
guards ran inside acquisition; no native build overlapped measurement.

## Completed Performance Evidence

The single acquisition completed in **102.02959463198204/600 seconds**, prior
debit zero, retries zero. It contains exactly 120 formal observations,
twelve warmups, and two six-job profile processes. All 144 workload jobs pass
the maintained independent twenty-one-snapshot/schema/resource/lifecycle
validators. All 135 owned workload invocation receipts confirm reaping.

Each dense run has 10,000 seed rows plus twenty 100,000-row source batches;
sparse uses the same seeds plus twenty 100-row batches. Each retains all
twenty-one full snapshots. The primary timer starts after IPC preload and
covers graph construction, planning, execution, retained output collection,
natural end, and cleanup. Oracles run afterward, within the shared budget.
These are whole-run elapsed times, not per-row or per-Batch latency.

| Case             | Baseline P50 ms | Candidate P50 ms | Change  | Verdict  |
|------------------|-----------------|------------------|---------|----------|
| Dense, one CPU   | 263.467         | 189.067          | -28.24% | improved |
| Dense, four CPUs | 250.388         | 172.311          | -31.18% | improved |
| Sparse, one CPU  | 27.236          | 23.442           | -13.93% | improved |

The maintained paired-median intervals are:

| Case             | Round one interval   | Round two interval   |
|------------------|----------------------|----------------------|
| Dense, one CPU   | [-35.354%, -26.351%] | [-31.594%, -25.657%] |
| Dense, four CPUs | [-34.011%, -29.002%] | [-34.139%, -27.098%] |
| Sparse, one CPU  | [-20.951%, -7.309%]  | [-19.128%, -9.263%]  |

Every upper endpoint is below -5%; therefore all three cases are `improved`
and the frozen acceptance predicate is true. Reviewer checked the emitted
endpoints/rule without rerunning statistics or collecting extra observations.
The 120 CSV rows match the complete raw observation inventory and wall values.

Processing-interval CPU P50 falls 263.453→186.243 ms, 297.202→219.231 ms,
and 27.236→23.443 ms for these cases. Whole-process peak RSS P50 falls
231.06→227.19 MiB, 243.88→239.95 MiB, and 70.52→66.50 MiB. The four-CPU
case's median average cores are 1.189→1.280. RSS includes input preload,
retained outputs, and validation; it is not isolated SQL working set or a
budget-capacity guarantee. No new SUM parallelism is claimed.

### Profile Interpretation

Fresh dense single-CPU profiles contain 1,288 baseline samples across windows
[258,281,248,250,251], and 878 candidate samples across
[177,175,175,171,180]. Baseline/candidate unresolved leaf counts are 374/184.
The 3,016/2,441 original timestamps match imported per-thread clocks, with
maximum error 5.960464477539063e-8 ms, below the 10 ns gate. Neither profile
has lost/unwind/warning diagnostics.

Group destruction's inclusive share changes 14.75%→4.78%, candidate
construction 14.44%→8.20%, and scalar destruction 2.87%→0.80%. The maintained
disjoint commit bucket changes 14.67%→4.90%. These observations support the
allocation/destruction mechanism alongside the component counter. Inclusive
shares overlap and separate processing cohorts do not establish paired
phase-wall deltas. Missing/inlined symbols do not imply zero cost.

The combined packing/cache candidate was measured; their separate wall-time
contributions were not isolated. Row conversion/filtering remain possible
future hypotheses. No additional variant, profile, case, or observation was
run using the unused budget, and no new speedup is promised for those paths.

### Artifact Integrity and Final Source Identity

The nine sealed acquisition files and protocol remain byte-identical. The
[manifest](../../../benchmarks/sum-state/sum-inline-state-20261105/evidence-manifest.json)
SHA-256 is
`bf4c824baa46fa21a4225c3e4865bc96117febfd1d585be29fe4b4ce1dfb604c`.
All eight recorded file hashes/byte counts match. The 297,034-byte complete
raw evidence SHA-256 is
`21500dcdb7a9db4361d9e643f0e4a96dba22cf3c99176cfb4c951578e60572d6`.

Raw evidence retains source/build receipts, all formal/warmup/profile jobs,
snapshot oracles, per-thread timestamps, alignment, stacks, resources, and
cleanup. Original large perf files remain at disclosed local paths with
hashes; imported full stacks are in the public artifact set.

Reviewer verified native live source/binary post-seals and equality of all
ten frozen helper hashes before/after acquisition. Published summary records
two accepted profiles, verified helper seals, complete analysis, completed
status, null failure, and true performance acceptance. No sealed file was
edited to change a verdict or narrative.

Final delivery checkpoint d0c60561 has no change from measured public P74 in
crates, Cargo files, configuration, scripts/helpers, schema, or Python API.
Only evidence and documentation were added. The final delivery does not
substitute a new unmeasured implementation.

## Blocking Issues

None remain for reviewed source, tools, performance evidence, or outcome
documentation. The frozen performance contract is met. Final required
CI/Codacy, coverage, and cross-platform checks remain merge blockers while
pending or unresolved; no final remote success is asserted here.

## Style and Documentation Consistency

No new dependency, unsafe exception, public knob, blanket security/complexity
exclusion, or fee-capacity claim is introduced. The specification describes
compact typed nullable state, preserved generic layout, immutable outputs,
and unchanged checkpoint contracts. The dated changelog and benchmark index
link the accepted study; older cohorts and the rejected parallel experiment
remain clearly labeled. The parent analysis describes whole-run timings,
paired intervals, component versus process resource scope, profile limits,
and explicitly unmeasured paths. No guide/API command change is needed for
this internal representation optimization.

## Verdict

**Approve — final specialist source, performance, and documentation PASS**,
bound to measured public 74a1eaf2 and local final delivery d0c60561.
The design findings are closed, all three measured cases are improved under
the frozen rule, and complete campaign and provenance evidence is
preserved.

This is not a green-CI or merge-ready statement. The exact final PR head must
pass all required CI, Codacy, coverage, and cross-platform checks before the
already-authorized merge. No new measurement is needed for evidence/document
additions that leave the reviewed measured source and helpers unchanged.
