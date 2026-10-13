# Issue 406: complete ASOF payload bulk-copy review

**Branch:** `feature/issue406-asof-bulk-copy`
**Parent:** `467a3a9b257164f9f85d69c2cc83ac32ab47f0c1`
**Ordinary parent product:** `8d3f6933c53f0dab1a562c79ad355cd9e08c665f`
**Reviewed state:** implementation working diff over `467a3a9b`, including
the exact former-production allocation reference and its final focused evidence

The reviewer owns only this report in the implementation owner's worktree.
No product/test files were edited and no build, test, profiling, CI query or
remote mutation was performed.

## Summary

Non-shareable complete records of at least 64 rows can copy supported flat,
non-null arrays through one contiguous Arrow range per column. Existing exact
sharing runs first; short, nullable and unsupported arrays retain whole-record
vectorized take. The former-component allocation reference now preserves the
original fallible collection and error mapping; its corrected measurements
resolve the evidence blocker. New paired timing remains pending.

## Build and test results

- `compact-red.log` records the existing direct cancellation/abandonment
  registration failing with **2,046 identity indices for 2,046 selected rows**.
- The first GREEN compilation fails because the test-owned operator lacks
  `mut`; `compact-green.log` preserves that error. It is corrected before test
  execution. `compact-green-fixed.log` then records the affected registration
  and new counter/parity controls passing in **0.03 seconds**.
- Explicit exact-buffer sharing and old/new component allocation controls pass
  with the original cancellation/abandonment assertions in **0.11 seconds**
  (`compact-sharing-allocation.log`).
- `direct-lifecycle.log` records **eight directly affected tests passed** in
  **12.12 seconds**, serially. They cover exact legacy checkpoint bytes and
  cold continuation, nullable projection/caller immutability, workspace/state
  error priority, credit limits, dictionary fallback and canonical descriptors.
- Initial scoped Clippy finds two test-only issues: a redundant `as_bytes`
  closure and potentially truncating cast. They are corrected without a lint
  allowance. The next combined command stops at formatting and runs no test
  or lint observation. Those failures remain documented.
- Earlier `compact-lint-fixed.log` records the exact changed test passing in
  **0.03 seconds** after a **2m 51s** compile. `clippy-fixed.log` records scoped
  core/library/test all-feature Clippy passing with `-D warnings` in **2m 17s**.
  The unchanged eight passing tests were not repeated.
- After the review finding below, `compact-exact-reference.log` records the
  same registration and all its helper controls passing in **0.03 seconds**
  after a **3m 40s** compile, with the exact former-production reference.
  `clippy-exact-reference.log` records final scoped all-feature core/library/
  test Clippy passing with `-D warnings` in **49.62 seconds**. Product copying
  code and the eight passing direct/lifecycle tests are unchanged.
- `complexity-exact-reference.log` recognizes all **nineteen changed/new
  functions** with Lizard 1.24.1, maximum CCN **eight**.
- Final owner format, schema-drift and whitespace checks pass. Reviewer schema
  and diff checks pass. Required CI and workspace coverage are not certified.

Owner commands run at the implementation worktree root with the recorded
shared build directory, two build jobs and serial test settings:

```bash
export CARGO_TARGET_DIR=/home/wegamekinglc/dev/github/my-claude/workspace/calc-flow/target/issue406-join-hotpaths-build
export CARGO_BUILD_JOBS=2
export RUST_TEST_THREADS=1
cargo test -p calc-flow --lib operator::asof::admission::direct::tests::test_direct_left_cancelled_and_abandoned_work_retains_credit_until_drain -- --exact --nocapture
cargo test -p calc-flow --lib operator::asof::admission::identity_tests::columnar::direct -- --test-threads=1 --nocapture
cargo clippy -p calc-flow --lib --tests --all-features -- -D warnings
cargo fmt --all --check
git diff --exit-code -- schemas/project-v3.schema.json
git diff --check
```

## Blocking issues

None remain.

**Resolved — `direct/tests/compact.rs`, `reference`:** the initial reference
collected a plain `Vec<ArrayRef>` after unwrapping `take`, whereas former
production collected `Result<Vec<ArrayRef>>`. In pinned Rust 1.88 the latter's
`GenericShunt` has a zero lower size hint, so the retained Vec capacity and
allocation measurements can differ. That reference was a valid value oracle
but was insufficient for an exact former-component allocation comparison.

The corrected reference returns `Result<RecordBatch>` and retains the original
`map_err`, `collect::<Result<Vec<_>>>()` and RecordBatch error mapping, with
unwraps outside the reference. The reviewer independently compared complete
function bodies after only function-name, error-module-path and whitespace
normalization and confirmed equality with ordinary parent `8d3f6933`.
`reference-source-match.json` records the owner's matching normalized hashes.
The final focused test and Clippy pass; updated analysis uses corrected bytes
while retaining and qualifying the original logs. No product change was needed.

## Style issues

None identified. The new private module keeps copying and its eligibility
boundary together, uses existing Arrow dependencies and introduces no unsafe
code, public API or lint exception. Mutation is confined to an owned Arrow
builder; caller arrays remain read-only.

## Correctness, funding and ownership

- `compact_complete_record` retains the original `can_share_batch` call before
  all copying decisions. Exact-share behavior remains a RecordBatch clone.
  Reconstructing copied records uses the original schema and metadata.
- Bulk eligibility requires at least 64 rows, no physical validity bitmap in
  any column and a supported flat type. A bitmap whose bits are all valid still
  declines; this preserves the original canonical take behavior.
- The pinned Arrow 58.3.0 implementation supports the selected numeric,
  temporal, Boolean, binary/string and fixed-binary types through
  `MutableArrayData`. View and other unsupported shapes retain take.
- `ArrayData::buffer<T>` starts at `data.offset`. Thus variable capacity uses
  the selected slice's first/last offsets, rather than its parent's span.
  Validated offsets refer to a live buffer, so their nonnegative address-domain
  conversion and subtraction share Arrow's existing validity invariant.
- `MutableArrayData::extend(0, 0, len)` copies the complete selected range,
  including bit-offset Boolean data and exact IEEE payload bits. Borrowed source
  data stays live until the owned output is frozen. No per-row index vector is
  constructed on this branch.
- Input reservations precede copying and retain their original sizes/lifetimes.
  The threshold bounds fixed Arrow setup against existing scratch headroom;
  focused peak-allocation controls check representative reservations. Copying
  increases setup allocation count, as the measured tradeoff below records.
- Source/identity selection, payload framing, worker thresholds, cancellation
  checks and funding/drain ownership are unchanged. Take and RecordBatch errors
  retain their existing Arrow error mapping.

## Test coverage

The independent reference retains the complete former fallible share-or-
identity-index-take implementation. Counter/parity controls cover
0/1/63/64/65/127/1,024-row cuts,
24 supported flat types, sliced variable spans, Boolean bit offsets, signed
zero/NaN/infinity bits, physical null spans and all-valid validity buffers.
They compare schema metadata, encoded IPC bytes and caller bytes before/after.
Buffer-overlap assertions check independent owned flat copies; view fallback
retains its original sharing semantics. The passing explicit share control
checks Arc identity for an exact non-sliced record with zero identity indices.

These scenarios reuse the existing registered direct lifecycle test. Its
original cancellation/abandonment assertions remain: running work retains
owners and credit until native drain. The eight passing direct/lifecycle
controls retain their existing workspace, error-priority and checkpoint
continuation assertions.

## Allocation tradeoff

The corrected former-production reference is measured beside the candidate
for the same three-column sliced records in `compact-exact-reference.log`:

- At **64 rows**, allocation count is **20 → 39**, peak bytes are
  **2,756 → 2,256**, and retained bytes are **2,188 → 2,232**, under the existing
  **12,858-byte** reservation.
- At **1,024 rows**, allocation count is **20 → 39**, peak bytes are
  **33,476 → 25,296**, and retained bytes are **25,228 → 25,272**, under the
  existing **166,458-byte** reservation.

Allocation count increases **20 → 39** and retained bytes increase **44** in
both controls, while component peak bytes decrease and fit unchanged
reservations. IPC parity plus exact legacy checkpoint controls pass.

Original simplified-reference logs remain preserved with old peaks of
2,740/33,460 and retained bytes of 2,172/25,212. Their apparent 60-byte retained
delta is superseded by the corrected comparison, and the analysis explicitly
explains why that old reference was not exact. These are component/funding
observations, not complete phase-wide allocation or RSS acceptance.

## Documentation and performance scope

The specification preserves all eight issue targets and prior inconclusive/
stopped evidence. Its 11.3956% compaction profile share retains exact product
`05c13fd9`; the reviewer confirmed `direct.rs` is unchanged between that product
and ordinary parent `8d3f6933`. This is diagnostic motivation, not a profile of
the new source or a timing claim.

The predeclared selection uses exact ordinary parent `8d3f6933`, with separate
candidate builds estimated at 6–14 minutes. Three proportional cases cover
OFF/64k at 32 and one CPU plus the necessary OFF/1,024/one-CPU setup control.
Counts are fixed at two rounds of ten adjacent AB/BA pairs and one warmup per
arm per round within 600 inclusive seconds after candidate build completion.
No timing acquisition has started.

The first selection names row 7 and is preserved in
`prebuild-selection-initial.json`. Before any candidate build or timing,
inspection establishes that the maintained runner selects `row7_releases`
and feature-enabled dispatch-trace wheels for that row. Ordinary wheels cannot
execute it. The revised selection keeps three cases but changes the 1,024-row
control to ordinary **row 6**, with **`processing_seconds` for every case**.
CPU hooks retain those original processing bounds. No observation is retried,
timer overridden or case added.

The existing four harness validator tests pass initially. After this amendment,
three ON guard tests pass while the dependency-feature fixture fails because it
still names row 7. Correcting that fixture makes the exact fourth test pass
alone; both logs remain. Source inspection and the prepared wrapper confirm
the revised fixed inventory and explicit metric label.

The ordinary 1,024-row control cannot certify row-7 settled timing or its
matched-projection harness-overhead ratio. Sampled RSS and focused component
allocations likewise cannot certify complete phase-wide allocation/RSS, ON or
all-eight acceptance. The analysis preserves earlier stopped/inconclusive
results and the shared-environment isolation requirement.

## Verdict

**Approve — source implementation and recorded focused evidence.** The former-
component allocation reference finding is resolved, and no correctness,
ownership, style or documentation blocker remains. Paired timing, complete
phase allocation/RSS, ON and all-eight acceptance remain unproved; required
CI/coverage are uncertified. This is a local specialist handoff, not a green-CI
or merge-ready claim.
