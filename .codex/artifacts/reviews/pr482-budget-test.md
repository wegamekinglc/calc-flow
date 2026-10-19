# PR #482 Review: EdgeBudget contract test correction

**Author:** Cheng Li | **Branch:** `feature/join-cpu-pipeline` → `main`

**Reviewed source:** `db234bfe320eba2624961b178706b4232223ef27`

**Base:** `e580eb28a013e02de85583ee2b428bcda70629d0` | **Files:** 1

## Summary

The `pipeline_compile` integration test still expected byte budgets above
256 MiB to fail. That assertion contradicts the existing `EdgeBudget::new`
contract, its core unit tests, and the documented caller-selected byte limit.
This test-only change replaces the obsolete expectation with positive-byte
acceptance and preserves row-limit rejection, adding zero-limit coverage.

The reviewer read the complete changed test file and checked the relevant
implementation and guides. Production code is unchanged.

## Build and Test Results

The orchestrator ran native checks; the reviewer inspected their logs under
`target/pr482-ci-followup/` in the parent checkout.

- CI RED at `e580eb28`: Windows `pipeline_compile` reported **24 passed,
  one failed**, in 0.02 seconds (`windows-current.log`). The failure is the
  obsolete assertion in `edge_budget_rejects_unbounded_runtime_limits`.
  This observed failure supplies the regression evidence; it was not rebuilt
  locally merely to repeat the old assertion.
- Rust GREEN at `db234bfe`: the complete affected integration target reported
  **26 passed, zero failed, ignored or filtered**, in 0.01 seconds after a
  2m 04s build (`budget-test-green.log`).
- Target-scoped Clippy: **Passed**, exit code 0 in 0.88 seconds
  (`budget-test-clippy.log`). Formatting: **Passed**,
  `cargo fmt --all --check`, as recorded by the orchestrator.
- Python: **Not touched**. No package rebuild or timing measurement needed.
- New failures: the stale integration expectation is fixed; no unresolved
  local failure remains. No production regression is introduced by this
  test-only patch.

The source-bound commands were:

```bash
cargo test --locked -p calc-flow --test pipeline_compile \
  --features benchmark-internals -- --test-threads=1
cargo clippy --locked -p calc-flow --test pipeline_compile \
  --features benchmark-internals -- -D warnings
```

Both used the parent checkout's `target/`, two Cargo build jobs, and its
managed `.venv/bin/python` as `PYO3_PYTHON`. Reviewer-run whitespace and
production/schema drift checks for `e580eb28..db234bfe` returned exit code 0.
No native command, full local matrix, coverage run or remote check was run by
the reviewer. Replacement required CI remains pending at this handoff.

## Blocking Issues

No findings in the reviewed patch. The previous failed CI run does not permit
merge; the latest required CI, coverage and Codacy checks must pass first.

## Style Issues

None. The two focused cases follow the integration module's conventions.
No dependency, fixture, suppression or production API changes are included.

## Test Coverage

The rejection case checks zero rows, zero bytes and rows above `MAX_ROWS`.
The acceptance case checks byte limits of 1, 256 MiB plus 1, and `usize::MAX`,
verifying that the constructor preserves both caller values. It also pins
`MAX_BYTES` and the Rust default byte budget to `usize::MAX`.

These assertions preserve the one-million-row ceiling while testing the
absence of an arbitrary lower byte ceiling. Constructing these budgets does
not allocate their advertised capacity. The remaining 24 integration tests
are unchanged and pass in the same focused run.

## Documentation Consistency

The corrected test agrees with [runtime defaults](../../../docs/runtime-envelope.md)
and the [streaming guide](../../../docs/streaming-guide.md). No guide update
is needed. The distinct historical project-v3 defaults remain unchanged.

The [payload-cursor review](pr482-payload-cursor.md) remains applicable to the
unchanged runtime. Previous timing results keep their original measured
source; this test-only correction makes no new performance claim.

## Verdict

**Approve — reviewed test correction and completed focused verification.**

All 26 affected integration tests, target-scoped Clippy and formatting pass
at `db234bfe`. The authorized merge remains gated on green required CI,
coverage and Codacy for the final head.
