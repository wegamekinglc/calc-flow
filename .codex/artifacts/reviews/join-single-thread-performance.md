# Join Single-Thread Performance - Specialist Review

**Branch:** `feature/join-single-thread-performance` → `main`  
**Baseline:** `78b75724f408f9d46a479488b4ecc76ea17aec68`  
**Review state:** Frozen implementation approved; final verification and bounded
performance evidence are pending.

## Summary

The patch removes repeated source-width and retained-chunk resolution, groups
short-string hash accounting into bounded blocks, appends existing match runs,
and introduces compact ASOF right-side output positions. It preserves the public
API, 1M transport default, unlimited implicit memory policy, canonical keys, and
checkpoint formats. The specification and critic's narrowed scope correctly
distinguish existing main optimizations from newly removed work.

This is a branch review of the collaboration worktree, including its explicitly
frozen uncommitted implementation. There is no new PR, check-run snapshot, or
prior GitHub review for this branch. The original user worktree was not changed.

## Build and Test Results

- Rust: coordinator-reported first Join stage **58 focused tests passed** after
  observed source-width, retained-lookup, short-string debit, and bulk-append RED
  regressions. The preceding ASOF candidate passed 384 ASOF tests; the final
  bounded owner policy still needs its GREEN run. Partial eviction passed 11
  lazy-delta tests, and another 13 persistent-dictionary plus 16 expiration,
  heap, and native-allocation checks passed. Final lint evidence is pending.
- Python product API: unchanged; product suites not run by this reviewer.
- Measurement helpers: coordinator-reported **11 native-free checks passed**.
- Performance: no acquisition completed at this intermediate review.
- No native build, test execution, benchmark, or remote operation was started by
  this reviewer. The coordinator owns native execution and source freezes.

The coordinator compiled the selected library tests with
`cargo test -p calc-flow --lib --features benchmark-internals --no-run`, using
`CARGO_BUILD_JOBS=2`, the repository `target/`, and its managed `PYO3_PYTHON`.
The recorded filters cover materialization (17), output projection (2), retained
rows (6), local dictionary work (17), matches (6), owner pair assembly (1), and
native gather/window behavior (9). Expected `catch_unwind` diagnostics in the
projection tests are not test failures.

## Blocking Issues

No remaining implementation findings. Final policy tests, fresh scoped lint and
format checks, and bounded current-main performance evidence remain acceptance
gates; this intermediate report does not claim they have passed.

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
source-null cases have explicit tests. Final execution evidence is pending.

The frozen partial-eviction implementation preserves expiration-plan order,
canonical lease fallback for restored rows, and the original native truncation,
all-binding removal, then per-key retirement phases. It computes aggregate
dirty/metric updates before mutation and moves captured raw key owners into
tombstones. Canonical binding resolution remains per expired row; it is not a
new per-key-only lookup optimization. Coordinator-observed RED regressions cover
raw clone counts, restored slot reuse/order, metric and dirty failures, with a
passing tombstone-funding refusal control. Its candidate lazy-delta checks pass.

## Documentation Consistency

- The ASOF guide and changelog now describe compact single-source `take`, the
  multiple-source `interleave` fallback, and the completed eviction work.
- Interval benchmark order hashes are descriptive only: its maintained
  two-input oracle checks exact pair multiplicity, identity, values, and schema
  independently of cross-input scheduling. Focused Rust tests establish the
  guaranteed incoming/probe/build-run order. The coordinator agreed to make
  this distinction explicit; do not introduce an unsupported global-order gate.

## Verdict

**Comment Only** at this intermediate stage: the frozen source has no remaining
review findings; final verification and performance acceptance are pending.
