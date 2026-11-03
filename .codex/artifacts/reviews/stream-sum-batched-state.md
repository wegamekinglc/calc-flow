## PR #533 Review: perf: batch merge grouped integer SUM state

**Author:** wegamekinglc | **Branch:** feature/stream-sum-batched-state → main | **Files:** 33

Reviewed public head: `e2916cb4b3e503133a6c0463be749dd66a8f9718`.
Base: `199d70b4156591153eb6b15cc2c8ffc238b3348f`.
The review used a detached worktree at that public head. The final measured
product revision is `84186554b078e25e71003a74fedc91c63b2b7f70`; its two changed
production modules and common benchmark source have the same SHA-256 values
at the reviewed head. Subsequent changes contain evidence and CSV attributes.

### Summary

Grouped built-in Int64/UInt64 SQL `SUM` now merges committed history into native
batch partials after processing input, then extracts combined state directly.
The final implementation borrows historical integer state by stable group slot
instead of cloning it into every pending candidate. Other scalar accumulator
paths also avoid redundant candidate copies after their accumulators have
already imported the old state.

This completes the planned first state-processing milestone. Composite-key
rewrites, group sharding, and finite filter/projection changes remain explicit
conditional follow-ups; this PR does not claim to implement those changes or a
new parallel execution model.

### Build and Test Results

Evidence was inspected rather than repeating passing native builds or tests:

- Rust: **Passed, focused scope.** The initial implementation passed 303 related
  SQL controls. Its final enum form passed five grouped tests. The final borrowed
  history form passed six grouped tests, four transaction/checkpoint/budget/
  cancellation/output-refusal controls, and the final allocation-cost test after
  the last fixture-only lint correction.
- Meaningful RED evidence exists for both performance properties: the original
  path constructed scalar SUM accumulators, and the first optimized candidate
  still retained 256 bytes of integer historical scalar buffers where the final
  candidate requires zero. The latter test also positively checks that the
  sequential floating path retains the history it needs. Zero-test fingerprint
  runs and earlier fixture failures are not counted as RED or GREEN evidence.
- Strict scoped Clippy: **Passed**, including the core library, library tests,
  and feature-gated diagnostic example with warnings denied. Earlier lint
  failures were corrected; no lint relaxation or new dependency was introduced.
- Python measurement tooling: **Passed**, 22 focused collector/publication tests
  and three builder/provenance tests, plus focused Ruff checks. The Python
  application API was not changed or rebuilt.
- Formatting, generated-schema drift, and complete diff whitespace checks:
  **Passed**, as recorded by the root handoff. The narrow CSV `cr-at-eol`
  attribute preserves immutable evidence bytes and the ordinary whitespace
  checks.
- Full workspace, connector-container, Python application, cross-platform and
  coverage matrices: **Not run locally**. This review does not establish the
  workspace coverage gate or required CI results.
- New unresolved failures: **None in the inspected focused evidence**.

The reviewer inspected `git diff origin/main...HEAD`, the final changed source,
benchmark/collector/build/profile/publication helpers and tests, both result
reports, complete compressed evidence, manifests, and the native RED/GREEN/
Clippy logs. `gh pr view 533` confirmed that the reviewed public head is the
actual PR head. No CI status snapshot or remote review submission was made by
this reviewer; the root owns the single final CI snapshot and PR handoff.

The recorded final-source commands ran from the W2 worktree with
`CARGO_TARGET_DIR` set to the repository's shared `target/` and
`CARGO_BUILD_JOBS=2`:

```bash
cargo test --locked -p calc-flow --lib \
  operator::sql::incremental::grouped_ -- --test-threads=2
"$CARGO_TARGET_DIR/debug/deps/calc_flow-e0b2d4c76e5c5e87" \
  test_checkpoint_contains_native_group_states_instead_of_admitted_history \
  test_sql_incremental_cancel_stops_group_finalization_and_rolls_back \
  test_sql_rejected_output_preserves_clean_and_dirty_state \
  test_compact_delta_refused_new_group_refunds_and_keeps_capture \
  --test-threads=2
cargo test --locked -p calc-flow --lib \
  test_grouped_integer_sum_pending_history_does_not_retain_scalar_buffers \
  -- --nocapture
cargo clippy --locked -p calc-flow --lib --tests \
  --features benchmark-internals --example grouped_sum_state_benchmark \
  -- -D warnings
```

The corresponding inspected logs are `target/stream-sum-history-green.log`,
`target/stream-sum-history-transaction-controls.log`,
`target/stream-sum-history-final-cost-test.log`, and
`target/stream-sum-history-clippy-fixed.log` in the original repository.

### Blocking Issues

None remain in the reviewed code or scoped evidence.

A tooling blocker found during review was corrected before final acquisition:
late source verification or publication could previously leave authoritative
records marked complete after the deadline. `finalize_acquisition` now updates
and persists stopped status after final seals; late publication rewrites source
and published status, raw evidence, README, dependent hashes and the phase
receipt before returning an error. Focused mocked-clock tests inspect the
durable records and hash changes after the optimistic writes.

### Correctness and Resource Review

- Admission requires the concrete DataFusion `Sum` implementation, a supported
  Int64/UInt64 result, and no distinct/order modifier. Existing logical-plan
  admission still rejects custom aggregates and unsupported shapes.
- DataFusion 55 scalar and grouped integer SUM both use wrapping addition.
  Historical integer states are merged exactly once after this batch's partial
  updates. Floating/decimal arithmetic and their ordering constraints are not
  admitted by the new integer path.
- Sequential float slots are separate from integer-history merging. The
  ordered-LAST_VALUE shape currently admits only its single aggregate, so it
  cannot preseed an integer SUM that would be merged a second time.
- `committed` groups stay immutable and their slots stay stable until commit.
  New group slots seed empty history. All merged values remain transaction-owned
  until successful output and commit, preserving discard, cancellation and
  output-refusal rollback.
- History conversion is bounded by `CHUNK_ROWS`, with cancellation checks and
  awaited yields between chunks. Existing conservative partial-state charges
  cover native state, temporary fixed-width arrays/ranks and bitmaps before
  allocation; actual native size is checked after merging. The change makes no
  claim that a configured memory budget admits more groups.
- Integer checkpoint state remains one nullable scalar with the same native
  field, policy, identity and layout. Restore code is unchanged. Recovery and
  continue tests exercise that contract; no protocol/version change is present.

### Test Coverage

The focused integer prefix oracle covers Int64 and UInt64 wrapping boundaries,
native and composite/string keys, null keys and values, aggregate filters,
mixed SUM/COUNT, empty and sparse inputs, more than 8,192 historical group
ranks, and discarded transaction replay. The borrowed-history test adds a mixed
integer/floating positive control and verifies reservation refund on drop.
Existing related SQL controls cover mixed extrema/average aggregates, compact
budget refusal, output rejection, cancellation and checkpoint continuation.

### Performance Evidence

The committed artifacts distinguish the two actual candidates:

- `benchmarks/sum-state/20261104` retains all four initial cases, 160 formal
  observations, all warmup/profile rows and two accepted CPU profiles. The
  original streaming labels remain `no-confirmed-regression`; the finite
  control remains `inconclusive`. These results were not relabeled or replaced.
- `benchmarks/sum-state/20261104-history-borrow` retains three complete final
  stream cases, 120 formal observations and 12 excluded warmups. Every
  observation uses the common harness and checks all 21 complete snapshot
  values, schema/hash evidence, actual CPU/thread settings and clean lifecycle.
  Fresh alternating baseline/candidate pairs compare the original baseline
  directly with the final product source, not with selected earlier medians.

| Final-source case  | Baseline P50 ms | Candidate P50 ms | Change  | Maintained verdict |
|--------------------|-----------------|------------------|---------|--------------------|
| stock-stream-1cpu  | 364.328         | 290.357          | -20.30% | improved           |
| stock-stream-8cpu  | 378.398         | 293.215          | -22.51% | improved           |
| sparse-stream-1cpu | 37.855          | 35.931           | -5.08%  | inconclusive       |

Both dense cases satisfy the unchanged two-round paired rule and 5% threshold.
Their intervals are preserved in the report and raw statistics. The sparse
case does not establish improvement, equivalence or regression.

Build receipts retain compiler/release/feature identity, common harness hash,
binary seals and 1,002 baseline / 1,003 final-source file seals. The root
verified the complete source maps and artifact manifest hashes; this reviewer
also checked the final production-module and harness seals against the public
review worktree. Readable CSV and complete compressed observations preserve
the data needed to audit the stated results.

The cumulative measurement/analysis/publication receipt is 581.615330872 seconds
within the original 600-second budget; builds are separately timed. The final
candidate profile was skipped by its predeclared remaining-time admission.
Historical profiles are labeled as earlier-cohort context, not a measured
profile of the borrowed-history candidate. No retry or sample-count expansion
was used to obtain the final verdicts.

### Style Issues

None requiring correction in the reviewed head.

### Documentation Consistency and Limits

The changelog, benchmark index and reproduction/result guides match the final
state-processing implementation and keep the original unfavorable or
inconclusive classifications. Important limits are stated explicitly:

- Final-source finite execution was not measured; its earlier inconclusive
  control cannot prove final equivalence or lack of regression.
- Peak RSS includes preload, retained snapshots and validation; this is not an
  operator-only memory result. The common harness retains every snapshot, so
  absolute times are not interchangeable with the older final-only sink.
- Eight logical CPUs report four core IDs on this WSL2 host. Host load/power
  state, off-CPU waits, hardware counters and memory bandwidth are unmeasured.
- No final-profile claim, new multicore scaling claim, general project-wide
  speedup, checkpoint performance claim, or benefit for unimplemented key/
  sharding/filter work is established.

At review time PR #533 remains a draft and its body still describes the initial
candidate and pending follow-up. The root's final handoff must replace that body
with the final measured behavior, scope and validation, then record the one
non-blocking CI snapshot. This is a PR presentation handoff task, not a request
for more product changes or measurements.

### Verdict

**Approve — PASS for the reviewed implementation and explicitly scoped evidence.**

No code-review blocker remains. Required CI, coverage and cross-platform checks
have not been certified by this review; unresolved required checks still block
merge. This review grants no merge authority.
