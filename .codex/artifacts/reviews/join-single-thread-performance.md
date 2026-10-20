# Join Single-Thread Performance - Specialist Review

**Branch:** `feature/join-single-thread-performance` → `main`

**Baseline:** `78b75724f408f9d46a479488b4ecc76ea17aec68`

**Implementation:** `98012d8a91bb3d0847a50cb77d21025f13629864`

**Tree:** `1ae0a8e48818c7464fbf51425c9931347cafa928`

**Review state:** Final implementation and focused-performance approval;
required GitHub CI remains unresolved.

## Summary

The patch removes repeated source-width and retained-chunk resolution, groups
short-string hash accounting into bounded blocks, appends existing match runs,
and introduces compact ASOF right-side output positions. It preserves the public
API, 1M transport default, unlimited implicit memory policy, canonical keys, and
checkpoint formats. The specification and critic's narrowed scope correctly
distinguish existing main optimizations from newly removed work.

This is a branch review of the committed implementation above, reviewed in the
isolated collaboration worktree. Its release build uses the separate clean
`target/worktrees/join-single-thread-candidate-sealed` checkout at that exact
commit. No GitHub check runs or prior GitHub reviews for this new branch were
used for this review, and no GitHub review was submitted. The original user
worktree was not changed.

## Build and Test Results

- Rust: **487 distinct focused tests passed** across the directly affected
  behavior: 387 ASOF, 98 Join, and two runtime-helper tests. Of these, 446 have
  durable logs with hashes recorded in `verification.json`; the other 41 are
  earlier unchanged-module checks preserved in exact coordinator tool results.
  This reviewer verified every durable log's recorded hash and passing summary.
- Fresh scoped Clippy: **Passed**, with an actual `Checking calc-flow` entry
  naming this feature worktree and a successful 1m 31s completion. The earlier
  cross-worktree `Fresh` cache observation is excluded from acceptance evidence.
- Formatting, generated project schema, and whitespace: **Passed**, recorded
  as `cargo fmt --all --check`,
  `git diff --exit-code -- schemas/project-v3.schema.json`, and `git diff --check`.
- Python: **32 focused installed-candidate tests passed** for ASOF streaming,
  symbolic stream Join acceptance, and stream budget defaults. The log verifies
  that the installed native module came from the sealed candidate release.
  Public Python source is unchanged; the complete Python matrix was not run.
- Measurement helpers: **11 native-free checks passed**, including after the
  sealed-source path adjustment; their preparation record distinguishes those
  checks from actual acquisition.
- Performance: **Completed** all five predeclared cases, 200 formal observations
  and 20 excluded warmups, in **220.08 seconds** against the fixed 600-second
  acquisition budget. No case was retried, expanded, or left incomplete.
- New failures: **None** after the final fixes. Expected TDD RED results are
  retained separately and are not final regressions.
- CI and combined Rust/connector coverage: **Not run for this new local branch**;
  no green CI or 90% coverage claim is established.
- No native build, test execution, benchmark, or remote operation was started by
  this reviewer. The coordinator owns native execution and source freezes.

The coordinator compiled the selected library tests with
`cargo test -p calc-flow --lib --features benchmark-internals --no-run`, using
`CARGO_BUILD_JOBS=2`, the repository `target/`, and its managed `PYO3_PYTHON`.
The final ASOF filter includes the bounded owner policy, grouped admission,
identity/duplicate precedence, supported payloads, checkpoint/restore,
cancellation, and output ownership. Join checks cover materialization (17),
output projection (2), retained rows (6), local dictionary work (17), matches
(6), owner pair assembly (1), native gather/window behavior (9), lazy deltas
(11), persistent dictionary behavior (13), and expiration/native allocation
controls (16). Expected `catch_unwind` diagnostics in the projection tests are
not test failures. Detailed provenance is recorded in
`target/performance/join-single-thread-performance/verification.json`.

## Blocking Issues

No remaining implementation, focused-test, format, lint, or focused-performance
findings. Required CI, coverage, and cross-platform checks remain unresolved and
therefore block any merge. The prior PR's green checks do not establish these
gates for this implementation.

Two findings were resolved before sealing the implementation. Multi-source
materialization now performs one baseline interleave instead of prefix/tail
materialization followed by another full-output copy. The Tokio owner starts
with legacy tuples unless a bounded metadata proof establishes at most one
right source: both live owners and physical hash-table backing are checked
before scanning. Larger, multi-right-owner, and historically sparse pools fall
back immediately. Thus the owner never converts an accumulated compact prefix;
conditional conversion remains on native fragment workers with bounded stop
checks, funded old/new overlap, and a final check before publishing tuples.

## Style Issues

None found in the frozen Join stage or compact-position representation.

## Test Coverage

The Join tests exercise stable and alternating immutable source identities,
nullable slices, duplicate and empty projections, sparse chunk coordinates,
backwards revisits, short/mixed UTF-8 word-hash parity, byte/visit cancellation,
forced collisions, worker continuation, sliced run views, and bulk owner fill.
The borrowed cursors add constant storage and cannot outlive the immutable state;
they do not allocate over sparse historical chunk gaps.

The compact ASOF tests cover leading unmatched rows, matching physical row zero,
conditional funded source promotion, direct range slices, and synthetic indices
wider than `u32`. The new segmented regression builds compact and promoted plans
through the production builder and checks cross-fragment ranges and credit
release. A separate allocation regression rejects the earlier additional
full-output copy. Cancellation, prepaid overlap, giant UTF-8 parent release, and
source-null cases have explicit passing tests. The final owner-policy regression
observed 64 prefix conversions before the fix and zero afterward; topology and
historically sparse backing controls exercise conservative fallback selection.

The frozen partial-eviction implementation preserves expiration-plan order,
canonical lease fallback for restored rows, and the original native truncation,
all-binding removal, then per-key retirement phases. It computes aggregate
dirty/metric updates before mutation and moves captured raw key owners into
tombstones. Canonical binding resolution remains per expired row; it is not a
new per-key-only lookup optimization. Coordinator-observed RED regressions cover
raw clone counts, restored slot reuse/order, metric and dirty failures, with a
passing tombstone-funding refusal control. Its candidate lazy-delta checks pass.

## Performance Evidence

The exact comparison is main `78b75724` against implementation `98012d8a`, both
clean, default-feature release builds with identical compiler, dependency lock,
flags, input bytes, explicit budgets, and maintained harness. Build times of
612.73 and 599.60 seconds are separate from acquisition. Neither diagnostic
features nor historical PR #482 timings enter these samples.

Each case uses 1M workload rows, two rounds of ten adjacent alternating AB/BA
pairs, and opposite round leads. Within each comparison, CPU affinity and
explicit Tokio workers match between arms. The 32-CPU case uses 32 logical CPUs;
WSL reports 16 core groups, so it is not evidence from 32 physical cores. The
ready-source-enqueue through Arrow-output timing window is unchanged.

| Case (CPU / source batch) | Main P50 ms | Candidate P50 ms | Change  | Round 1 interval % | Round 2 interval % | Verdict                 |
|---------------------------|-------------|------------------|---------|--------------------|--------------------|-------------------------|
| Join (1 / 1M)             | 90.81       | 78.51            | -13.54% | [-18.30, -3.26]    | [-22.31, -8.21]    | no-confirmed-regression |
| ASOF (1 / 1M)             | 86.75       | 84.02            | -3.15%  | [-10.95, +3.37]    | [-7.57, +3.88]     | no-confirmed-regression |
| Interval (1 / 1M)         | 799.04      | 672.51           | -15.83% | [-20.80, -16.82]   | [-16.43, -12.48]   | improved                |
| Interval (32 / 1M)        | 578.57      | 487.17           | -15.80% | [-27.02, +1.46]    | [-25.37, -11.12]   | no-confirmed-regression |
| Join (1 / 64k)            | 63.20       | 59.17            | -6.37%  | [-14.84, -3.56]    | [-9.44, -2.04]     | no-confirmed-regression |

Intervals and verdicts come from maintained `report.comparison`, including its
paired-median interval method and 5% threshold. Only single-CPU interval meets
the rule for confirmed improvement in both rounds. The other four lower P50s
are descriptive; they are not confirmed improvements or proofs of equivalence.
All five meet the scoped no-confirmed-regression gate, and none is inconclusive
or a confirmed regression under that maintained rule. This is a per-case review,
not a complete-suite acceptance verdict.

This reviewer checked the frozen plan and build-manifest hashes, matching
per-round machine/dependency/workload identities, all 220 all-row oracle results,
all 20 completed worker records, and the absence of recorded interference or
cleanup failures. All 20 worker stderr logs are empty. Join/ASOF produce exactly
1M rows; each interval sample produces exactly 10,998,080 pairs. Interval status
records 999,680 evictions and 320 retained rows per side, with zero state/match
limit failures and valid progress. The 64k Join control retains zero left rows
at its accepted cut.

Process-lifetime peak RSS and whole-sample CPU are recorded, including fixture,
oracle, and lifecycle work outside the ready timer. They do not establish
native allocation peaks, memory regression equivalence, or timed worker
utilization. WSL host-wide quiet and host power mode remain unobservable.
Unmeasured paths include Join/ASOF multicore, other scales, skewed/wide/nested
payload timings, sparse long-lived state, checkpoint/recovery timings, and
external-engine comparisons. The 64k Join control uses a different maintained
adapter from the 1M bounded-feed Join, so cross-case ratios do not isolate batch
size. No new profile acquisition was performed.

The complete raw timings, maintained P95s, resource records, correctness proofs,
and provenance are in
`target/performance/join-single-thread-performance/reports/measurement.json`;
installed Python evidence is in `installed-candidate-tests.json` and its log.
This approval applies to measured implementation `98012d8a`; later evidence-only
commits must preserve its product and maintained-harness source.

## Documentation Consistency

- The ASOF guide and changelog now describe compact single-source `take`, the
  multiple-source `interleave` fallback, and the completed eviction work.
- Interval benchmark order hashes are descriptive only: its maintained
  two-input oracle checks exact pair multiplicity, identity, values, and schema
  independently of cross-input scheduling. Focused Rust tests establish the
  guaranteed incoming/probe/build-run order. The coordinator agreed to make
  this distinction explicit; do not introduce an unsupported global-order gate.

## Verdict

**Approve** the reviewed implementation `98012d8a` and its focused performance
acceptance. No implementation or observed performance regression finding remains.
Required GitHub CI, combined coverage, and cross-platform gates remain unresolved;
this branch review is not a merge-ready claim or authority to merge.
