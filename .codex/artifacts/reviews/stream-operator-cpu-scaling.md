# Branch Review: stream operator CPU scaling

**Branch:** `feature/stream-operator-cpu-scaling`

**Base:** `e35ff1190e8eba08ad3c3135299bbca6ac029698`

**Reviewed source:** frozen working diff over that base

**Files:** 34 implementation, test, specification and documentation files,
excluding this review

The review uses the implementation owner's isolated worktree. The reviewer
owns only this report and leaves that worktree in place for the ongoing task.
There is no PR, check-run snapshot or prior GitHub review for this working diff;
no remote action was taken.

## Summary

ASOF now sizes eligible output prefixes from the configured edge and available
workspace, and routes temporary admission/probe work independently of its
unchanged persistent state shards. Rolling uses CPU-sized whole-entity lanes
and compact row schedules; cross-section mean distributes complete time groups;
eligible row-local stream SQL reuses the existing immutable compiled path.

The implementation respects the specification's public-surface, arithmetic,
rollback and checkpoint constraints. The reviewed source and focused evidence
support approval; the separate performance comparison is not part of this
source verdict.

## Build and Test Results

- Rust: **Passed** focused verification. The first new-test execution passed
  24 of 28 tests. Three refund fixtures expected the job-owned 16,384-byte
  gather-home control reservation to disappear before dropping the job; the
  fourth used the upstream name for downstream metadata. After correcting
  those fixtures, all four and the additional eight-lane primary-error
  regression passed. This provides **29 distinct new passing tests** across
  the two executions, without repeating unchanged passing cases.
- Existing affected Rust regressions: **140 passed**, zero failed, in
  12.77 seconds. Coverage includes ASOF admission/finalization, cross-section
  state, rolling kernels, numeric-work ownership, managed runner lifecycle,
  checkpoint recovery and rejected SQL output.
- Scoped Clippy: **Passed**, completing in 1 minute 37 seconds. Scoped
  `rustfmt --check` over all 28 changed/new Rust files also passed.
- Reviewer `git diff --check e35ff119 -- .` and
  `git diff --exit-code e35ff119 -- schemas/project-v3.schema.json`: **Passed**
  in the implementation worktree. The changed-file whitespace/newline audit
  found no source or documentation issues.
- Python: **Not touched**; Python/native-wheel suites were not run.
- New unresolved failures: **None**. The four initial fixture failures have
  specific corrections and passing focused evidence.
- Regressions: **None observed** in the 140 selected existing tests. Full
  workspace, connector, cross-platform and 90% coverage gates were not run
  locally and are not certified by this review. CI is absent for this diff.

Evidence is recorded under the repository-root
`target/performance/stream-operator-cpu-scaling/`, including
`first-green.log`, `corrected-green.log`, `regressions.log`, their execution
JSON files, `corrected-green-build.jsonl`, `final-format.log` and
`final-clippy.log`. The reviewer inspected these records without launching
another Cargo build or test run. The build harness verifies that Cargo's
executable artifact belongs to this worktree's `calc-flow` manifest.

The focused build and Clippy commands were:

```bash
CARGO_TARGET_DIR=<repository-root>/target CARGO_BUILD_JOBS=2 \
  cargo test -p calc-flow --lib --no-run --message-format=json
CARGO_TARGET_DIR=<repository-root>/target CARGO_BUILD_JOBS=2 \
  cargo clippy -p calc-flow --lib --tests --features benchmark-internals -- -D warnings
```

The discovered test executable ran with `--test-threads=1`. Exact test filters,
source location and exit codes are retained in `first-green-execution.json`,
`corrected-green-execution.json` and `regressions-execution.json`.

## Blocking Issues

None remain in the reviewed source or focused verification.

The intermediate dynamic diagnostic buffer could retain its high-water
allocation after the active scratch ledger returned to zero. The final
`OwnerState::diagnostics` restores the existing inline two-entry storage.
Primary-error selection still examines every lane; the new lane-three and
lane-seven failure regression confirms this independently of the bounded
secondary diagnostics.

## Style Issues

None found. Changes preserve the functional declaration surface, caller-owned
Arrow buffers, explicit error flow, existing lint policy and local test
fixtures. No dependency, public configuration, schema or checkpoint wire
format change was introduced.

## Test Coverage

- ASOF controls cover larger configured prefixes, smaller explicit edges,
  workspace halving, oversized accounting, cancellation, and temporary routing
  across 16/32 units. Existing focused controls exercise exact output/order,
  retained owners, collector rejection, refunds and cold recovery. Persistent
  eight-shard state identity remains separate from temporary full-hash routes.
- Rolling controls cover one-CPU fallback, eight concurrent owned lanes,
  large batches/scratch, more than 32 eligible lanes, reused entity counts,
  single-pass row schedules, exact floating-point bits and touched state.
  Existing tests retain partial-launch, panic, cancellation, abandoned-work,
  deterministic error selection and checkpoint-drain coverage.
- Cross-section controls preserve complete time-group arithmetic, exact mean
  bits, metadata, shared payload buffers and output chunk geometry. They cover
  finite-pool fallback, emitter retry, cancellation/abandonment, restore/reset,
  and downstream worker reuse during an awaited upstream emit. The stopped
  worker test checks live input ownership and funding before dropping the job,
  then checks worker release before the final whole-job reservation refund.
- SQL controls cover immutable hot-path reuse, cancellation/retry, hot errors,
  and order/limit fallback. Inspection confirms selected-UDF and aggregate
  paths retain SQL execution, while unsupported compiled plans fall back
  through the existing template machinery.

The arithmetic and lifecycle evidence is scoped to these selected cases;
broader coverage remains the CI responsibility. No performance result or
speedup is inferred from worker selection or correctness tests.

## Documentation Consistency

The specification, changelog and changed ASOF, runtime-envelope, SQL-performance
and streaming guides agree with the implementation. They describe dynamic
worker sizing, the distinction between rolling scratch accounting and public
state budgets, cross-section complete-group/funding behavior, and compiled
SQL eligibility/fallback. Obsolete two-lane/private-ceiling descriptions were
updated. No missing public-command or architecture update was found.

Changed Markdown has no new table-alignment, trailing-whitespace or final-newline
issue. No additional documentation work is required for this source increment.

## Verdict

**Approve** the frozen source diff and focused correctness evidence. No
unresolved correctness, ownership, budget, style, documentation or focused-test
finding remains. This is not performance acceptance, full-coverage certification
or a merge-readiness claim; performance evidence and required CI remain separate
gates.
