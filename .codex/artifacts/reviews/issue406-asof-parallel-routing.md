# Issue 406: parallel ASOF capture routing review

**Branch:** `feature/issue406-asof-parallel-routing`
**Base:** `edb73649290ba6509dcd62c94e609d7948d6275a`
**Reviewed head:** `05c13fd983dd1ead625eef2278eb7127dac0c61c`
**Files:** seven, including the test-only `LeftState::output_iter` correction
and first-row capture simplification
**Evidence addendum:** reviewed analysis working diff over that source head

This branch review uses the implementation owner's isolated worktree and
preserves other edits. No PR/check-run snapshot or remote mutation was made.
The reviewer owns only this report and ran no product builds, tests or profiling.

## Summary

Parallel probe preparation now resolves eligible chunk dictionary keys into a
fixed 64-slot array and appends worker rows using integer IDs. The existing
four-selected-rows-per-slot threshold and generic route lookup remain in place
for short runs, large dictionaries and legacy rows.

The source and focused correctness evidence support approval. The subsequent
paired comparison improves OFF/32 relative to the exact parent product; its
single-CPU control remains inconclusive. The diagnostic profile and hash
regression alone establish no speedup, and the earlier merged-main comparison
retains its original classification.

## Build and test results

- Rust RED: `capture-red.log` records 1,037 encoded-key hashes for 1,024 rows
  and sixteen keys, failing the bound of 128 before implementation.
- First GREEN: `capture-green.log` records the affected registered regression
  passing in 0.03 seconds.
- Scoped finalization: `finalize-tests.log` records 27 passing tests and one
  new fixture failure. The one-key fixture has no right rows; constructing an
  empty list of RecordBatches violated the existing Batch contract. The helper
  now skips that empty submission, and `capture-boundary-tests.log` records the
  exact affected test, including all new boundary controls, passing in 0.13
  seconds. The unchanged 27 were not repeated.
- Clippy initially identified production-dead `LeftState::output_iter` after
  capture switched to output runs. It is now `#[cfg(test)]`, without a lint
  allowance. `clippy-fixed.log` records scoped all-feature core/library/test
  Clippy completing successfully in 59.41 seconds.
- Owner format, schema-drift and whitespace checks pass. Reviewer schema-drift
  and diff-whitespace checks also pass.
- Complexity: inspected Lizard 1.24.1 script and log recognize all 36 functions
  in the three complete probe/test modules, explicitly cover all changed
  helpers and report a maximum CCN of eight. The older 1.17.31 parser's
  truncated Rust result is preserved as incomplete evidence, not certification.
  The tool is installed under `target`, with no project dependency change.
- After this check found CCN ten in the first `capture_run` version, the owner
  removed two per-row offset branches by extracting the already-checked first
  row. That simplification is source-reviewed. Final
  `capture-simplified-tests.log` records the exact affected test passing in
  0.13 seconds, and `clippy-final.log` records scoped Clippy passing in 26.41
  seconds. The unchanged 27 passing tests were not repeated.
- Python package/runtime: not changed or rerun. Workspace coverage and current
  required CI are not certified by this review.
- New unresolved failures: none. New paired OFF/32 is improved relative to its
  exact parent; OFF/1 is inconclusive. This establishes no blanket absence of
  regression, and the prior merged-main OFF/32 result remains inconclusive.

The final focused commands run at the implementation worktree root, with the
recorded shared build directory and two Cargo build jobs:

```bash
export CARGO_TARGET_DIR=/home/wegamekinglc/dev/github/my-claude/workspace/calc-flow/target/issue406-join-hotpaths-build
export CARGO_BUILD_JOBS=2
cargo test -p calc-flow --lib operator::asof::finalize::probe::tests::key_shard_budget_rejection_refunds_and_preserves_input -- --exact
cargo clippy -p calc-flow --lib --tests --all-features -- -D warnings
```

## Blocking issues

None in the reviewed source and focused verification. The fixture and Clippy
failures above have concrete diagnoses, fixes and passing focused evidence.

## Style issues

None found. The helper test module follows the mirrored Rust test layout, uses
the existing registration and adds no dependency or lint exception. The private
`OutputRun` re-export changes no public API.

## Correctness, budgets and ownership

- `capture` retains canonical output-run order and one global selected-row
  position. Missing buckets append no worker row but still advance that
  position, so `restore_order` leaves the same unmatched slots as before.
- `capture_run` checks the first selected position before route preparation,
  then checks each remaining global position once. This preserves the existing
  1,024-row cancellation/yield boundaries across independent and interleaved
  runs. Route setup between checks is bounded by 64 slots.
- The first-row extraction's `expect` relies on the private caller invariant:
  exhausted prefixes skip `capture_run`, and each yielded output run is
  nonempty. The remaining iterator's offset maps to `position + offset + 1`.
  Zero-count capture still calls no run helper, and short/legacy runs retain
  their existing first-row behavior.
- The array is rebuilt per run, so local key IDs from independent dictionaries
  cannot alias each other. Retired dictionary holes remain `None`, as do absent
  right buckets; the existing keyed iterator preserves time and selected row
  positions through partial consumption.
- `capture_route` deduplicates actual right buckets by encoded bytes in the
  existing global map. It preserves `key_shard(key) % workers`, borrowed key
  lifetimes and the owned right-bucket Arc. Shard and bucket coordinates stay
  stable until all captures finish; only then does shard compaction occur.
- Resolving unused dictionary entries within an eligible selected run cannot
  exceed the existing route funding: dictionary slots are at most one quarter
  of that run's selected rows, generic additions are at most their row count,
  and actual positive routes cannot exceed the right dictionary size. Thus
  global capacity remains bounded by `min(count, right.len())`.
- The local array adds bounded scratch under the existing fixed allowance.
  Input/output reservations, iteration scratch and retained right-metadata/
  encoding-owner funding remain unchanged. Reservations and captured owners
  unwind on failure or cancellation; submitted work retains its input credit
  until owned worker exit.
- Worker selection, `ProbeWork::run`, monotonic candidate selection, tolerance,
  sequence dominance, output-plan construction and collector acceptance remain
  unchanged. No caller buffer, retained state or checkpoint format is mutated.

## Test coverage

The direct complexity check compares restored capture matches with binary
search for sixteen keys, including two missing keys, and checks status plus
reservation refund. New capture controls cover 1/63/64/65 dictionary slots,
selected counts 1/17/full, consumed prefixes 0/17, two independent chunk
dictionaries over 4,096 rows, a retired hole in exactly 64 slots, and legacy
rows. Each control checks restored matches, status and pool refunds and rejects
pre-cancelled capture without leaked credit.

The passing existing finalization controls include canonical order and exact
floating-point bits, cold checkpoint recovery, abandoned worker bucket funding,
collector rejection with identical checkpoint metadata/segments and exact
continuation, plus output-workspace and retirement boundaries. Unchanged
right-bucket tests provide tolerance/tie/identity-only cursor equivalence; this
increment does not alter that matching algorithm.

## Diagnostic evidence and performance limits

The frozen profile under `target/issue406-routing-parallel-profile-20261013/`
uses exact ordinary product `6739fa2b`, native SHA-256
`a131c569adaa5fdb629e868be6af42ca0e5a3e31359694a37a39116930340011`.
The reviewer verified its plan SHA-256
`97ca749aceebfd3b35c5a64623b6a1539bfa44a5b31d2da8d94e563f506c8fd9`,
all 1,650 frozen input hashes and 23 result seals, and all six raw output
oracles/lifecycles/cleanups with true OFF zero-file evidence and 32-CPU affinity.
The fixed five windows contain 659/697/669/669/660 samples, totaling 3,354;
3,075 include native frames. Inclusive elapsed time is 10.614190515 / 180
seconds, with all owned groups settled and no forced kills or survivors.

Stored matching shares are 33.81% inclusive / 18.78% self, and output-view
self share is 5.28%. Shares overlap. Sixteen lost lifetime events have unknown
window membership; host isolation on WSL2 is unverified. These are diagnostic
observations, not a causal timing or regression verdict.

The reviewer confirmed `6739fa2b` and parent `edb73649` have no difference under
`crates`, `python`, `Cargo.toml` or `Cargo.lock`. The declared comparison against
that exact ordinary parent product isolates this increment. Its two OFF cases
at 32 and one CPU, two rounds of ten adjacent AB/BA pairs, one warmup per arm
per round and 600-second inclusive budget are proportional. Candidate builds
are separate. Acquisition must preserve maintained oracles, release identities,
CPU-clock scopes, cleanup, paired statistics and strict no-retry behavior.

## Completed paired evidence addendum

The completed attempt under `target/issue406-parallel-routing-paired-20261013/`
compares parent product `6739fa2b`, native
`a131c569adaa5fdb629e868be6af42ca0e5a3e31359694a37a39116930340011`,
with exact candidate `05c13fd9`, native
`cbc1a85acd9d3b1d6baf0ec2e5fe0c888a5ec53353ee2176d1ae56281a6c600b`.
The plan SHA-256 is
`bbd2edcb3f8028ba8bd3d75b8e23f97f2a6822ca5875717b2adda4096bc40b92`.
The baseline is the exact parent product, rather than a new merged-main
comparison. The candidate ordinary release build takes 428.448805 seconds,
separately from acquisition; parent binaries are reused.

The reviewer independently verified all **3,109 frozen input hashes** and
**198 result-evidence hashes**, and validated both ordinary release audits with
equal complete inventories of **294 compiled artifacts** per arm, normalized
features/profiles, environment, interpreter, toolchain and lock identity.
Both cases contain the planned two rounds of ten adjacent AB/BA pairs and one
warmup per arm per round: **80 measured observations and eight warmups**.
All 88 raw results pass the full million-row oracle, completed lifecycle,
owned cleanup, true OFF zero-file checks and expected product/CPU-affinity
identity. Raw/normalized digests, sample indices, actual pair order and exact
processing-clock bounds also agree. No samples are retried, removed or added.

Recomputing the maintained comparison reproduces both stored statistics
objects exactly:

- OFF/32: parent p50 **178.5173225 ms**, candidate **154.689556 ms**,
  **-13.347593%**, verdict **improved**. Round paired medians are -12.173645%
  and -13.886924%, with intervals [-15.594299%, -10.678862%] and
  [-14.898605%, -10.264516%]. Both upper bounds are below -5%.
- OFF/1: parent p50 **122.682333 ms**, candidate **123.9289735 ms**,
  **+1.016153%**, verdict **inconclusive**. Round medians are -4.660668%
  and +1.982088%, with intervals [-11.549431%, +4.195972%] and
  [-3.276389%, +9.251586%]. This proves neither equivalence nor a confirmed
  regression; no blanket regression-free statement follows.

Each ten-pair order-statistic interval has **97.8515625%** coverage. The
immutable generated `results/performance.md` retains legacy `Merged main p50`
and `CI95` headings. The reviewed live analysis explicitly corrects their
meaning to the exact parent comparator and actual coverage. Sealed originals
remain unchanged.

The recomputed processing CPU medians match the independent audit. At 32 CPUs,
process CPU is 205.404 → 180.801 ms and the busiest observed thread is
84.447 → 66.361 ms; at one CPU, process CPU is 122.466 → 122.577 ms.
The reviewer also confirmed the three Appendix A helper source segments match
the sealed hook. CPU-counter read boundaries have uncertainty and cannot
reconstruct retired threads; these diagnostics neither replace paired wall
statistics nor establish exact thread accounting. Whole-job CPU is a separate
scope, and WSL2 host isolation remains unverified.

Inclusive acquisition and sealing take **39.018610928 / 600 seconds**. All owned
groups settle, with zero forced kills, survivors, pending creations or cleanup
errors. Candidate p50 values still miss the one/32-CPU OFF latency targets of
100 / 30 ms. The other seven acceptance rows remain unmeasured by this
increment. The earlier merged-main comparison and both stopped ON attempts
retain their original evidence and conclusions.

## Documentation consistency

The specification and analysis match the private implementation and preserve
all eight issue targets, the earlier OFF/32 inconclusive result and both stopped
ON acquisitions. They separate diagnostic evidence, source correctness and the
completed parent/candidate comparison. The final analysis addendum matches the
verified identities, sample counts, statistics, CPU scopes and limitations,
including the immutable generated-label caveat. No public command,
configuration, checkpoint or domain contract changes require a new application
guide.

## Verdict

**Approve** the source increment at `05c13fd9` and its reviewed evidence/report
addendum. No unresolved correctness, budget, ownership, style, focused-test or
reporting finding remains. OFF/32 improves against its exact parent; OFF/1 is
inconclusive. Issue-wide targets, ON qualification and required CI/coverage
remain unmet or uncertified. This is not a merge-readiness or eight-target
acceptance claim.
