# PR #482 Review: CPU pipeline remediation

**Branch:** `feature/join-cpu-pipeline` → `main`

**Reviewed source:** `25e8b4c33114f1f85a8473136bf3c1f7991873e5`

The helper refactor is sealed at parent
`49a5d62aa5b752573cd2eb15d4e78d695e5f9567`. Follow-ups repair the test-clock
branch at `1c086de8` and the route helper's borrow signature at `25e8b4c3`,
as described below.

**Refactor base:** `3aa0d57ac8d561426403816dc4943f9d4a0ddd1b`, which merges
the published PR branch with `main` at `4c8d23ba`.

**Scope:** 17 remediation files, plus the `admitted.rs` merge resolution and
its payload/materialization integration. This supplements the original
[CPU pipeline review](join-cpu-pipeline.md); it does not repeat that review or
claim a complete independent review of every upstream `main` change.

The reviewer used the implementation owner's isolated worktree and changed
only this report. Native builds, tests, performance acquisition, GitHub
metadata and check-run observations belong to the orchestrator. The worktree
remains in use for that verification. No remote review or merge was submitted.

## Summary

The remediation extracts bounded helpers from the functions identified by
the PR's complexity findings. Source review found that these extractions
preserve allocation and traversal counts, reservation ownership, cancellation
cadence, deterministic ordering, and output/state commit boundaries.

The main-branch merge retains both payload access without row metadata and
the PR's segmented cursor and dense-key interfaces. Two explicit workload
migrations preserve historical benchmark attestations after the PR pins the
old output budget in those benchmark fixtures.

## Build and Test Results

Evidence is recorded under `target/performance/pr482-remediation/` in the
parent checkout. Results remain bound to their tested source:

- Rust: **Passed focused verification after repairs**. The
  feature-enabled suite at `49a5d62a` completed with **219 passed, one failed
  and one ignored** in 53.41 seconds (`remediation-core.log`). Its execution
  record, `remediation-core-execution.json`, records the exact filtered test
  binary command, source worktree and exit code 101. The failure and narrow
  repair are assessed below. The existing ignored `test_morsel_paired_resources`
  is an explicit D7 timing diagnostic. At `1c086de8`, the focused deadline and
  budget checks passed: **four passed, zero failed or ignored**, in 5.25 seconds
  (`remediation-deadline.log` and its execution record). The feature-gate PyO3
  check also passed at that source: **one passed**, in 0.02 seconds
  (`final-binding-feature.log`). These are separate source-bound results, not
  a combined count of unique tests for the latest source. No native commands
  were run by the reviewer, and the 219 unchanged passing checks are not rerun.
- Final-source Rust tests: **seven passed, zero failed or ignored** at
  `25e8b4c3`, in 1.10 seconds (`remediation-route.log` and its execution record).
  These cover fragment cancellation, competing budgets, prefix allocations,
  cooperative capture, optional funding refusal, operator-name funding and
  segmented output ownership.
- Scoped Clippy: **Passed after repair**.
  `final-clippy.log` records one `needless_pass_by_value` finding in
  `Work::resolve_route`. Commit `25e8b4c3` repairs the borrow signature without
  a suppression. The final command passed in 1m 09s (`final-clippy-green.log`):
  `cargo clippy -p calc-flow -p calc-flow-python --lib --tests --features benchmark-internals --locked -- -D warnings`.
- Python installed-package checks: **seven passed** in 0.21 seconds against
  the sealed default release at `25e8b4c3` (`installed-host-check.log` and
  `installed-candidate-proof.json`). The selected modules are
  `test_symbolic_stream_join_acceptance.py` and `test_stream_budget_defaults.py`.
  Diagnostic methods are absent from this release. An earlier sandbox
  invocation stalled; the same artifact passed in the host environment,
  without a code change. Both observations remain recorded.
- Python helper checks: the test owner reports **27 dependency/harness tests
  and nine workload-migration tests passed**, including the new negative
  cases. Changed-test Ruff, scoped Rust formatting and the local complexity
  ratchet passed. These are focused results, not full-suite results.
- Default release build: **Passed**, 654.316 seconds, accounted separately
  from acquisition. The sealed manifest records `maturin build --release
  --locked`, default features, the exact source/tree, and native SHA-256
  `bfab76292304d732306e3adb6ac0ffca6ddfd7d851a1c238e178be212a29010e`.
- New failures: the first remediation test run exposed
  `test_morsel_cancel_deadline_panic_and_reset_keep_worker_leases_until_cleanup`.
  Source diagnosis identifies an existing test-clock mismatch, rather than a
  changed helper algorithm. The failed result remains recorded alongside the
  passing focused repair. The separate Clippy finding is recorded above.
- Unresolved new failures: **None** in the completed local verification.
- Regressions: no behavior regression identified by source comparison. The
  fixed performance comparison found no confirmed regression under the
  maintained rule; four cases remain inconclusive, as detailed below.
- CI and Codacy: no reviewer snapshot was taken. Current-head results remain
  pending the orchestrator's handoff; the local complexity report is not a
  substitute for remote Codacy acceptance.

Reviewer-run static checks in the isolated worktree all returned exit code 0:

```bash
git diff --check 3aa0d57a HEAD
git diff --exit-code 3aa0d57a HEAD -- schemas/project-v3.schema.json
git diff --exit-code 3aa0d57a HEAD -- benchmarks/rust-dependency-migrations.json
```

Read-only SHA-256 comparisons confirmed that each new workload migration's
baseline bytes match the corresponding benchmark at `4c8d23ba`, and its
candidate bytes match the reviewed source. The historical dependency registry
is unchanged.

The reviewer also ran `git diff --check` and the corresponding schema-drift
check for `49a5d62a..1c086de8` and `1c086de8..25e8b4c3`; all returned exit
code 0.

No full workspace suite, full Python product suite, connector services,
coverage run or performance workload was executed by this reviewer. In
particular, this handoff does not establish the combined 90% Rust line gate.

## Blocking Issues

No source blockers found in the reviewed scope.

Codacy acceptance and green required CI checks remain necessary before merge.
The completed local checks and measurements do not replace those gates.
Failures or unresolved required checks continue to block merge.

## Style Issues

No new convention violations found in the remediation. The helpers have
specific responsibilities; they do not introduce suppressions or change the
public API, dependency manifests, workspace lints, or caller-owned inputs.

The recorded local complexity comparison contains 25 original findings
addressed by the refactor. The only remaining functions above the configured
threshold in the inspected report are the unchanged `merge_numeric_rows` and
`test_engine_matrix_covers_every_decade_and_requested_library`. This is local
evidence only; final remote results remain pending.

## Test Coverage

The source review checked the existing tests and ownership paths associated
with the changed helpers. The refactor preserves these contracts:

- ASOF fragment matching and prefix preparation retain the same source order,
  dictionary route selection, independent reservations and cancellation
  checks. Serial and segmented materialization keep the same error mapping
  and workspace lifetime.
- Interval retained selection still visits groups and mask blocks in order,
  preserves original per-row charges, and uses the same worker cleanup and
  serial fallback behavior. Key intern/remap helpers retain their bounded
  cooperative steps and cleanup ownership.
- Envelope preparation validates sequence capacity before output delivery;
  the owner still performs state installation at the existing acceptance
  boundary. Checked resource arithmetic is algebraically unchanged.
- Rolling scratch sizing and lane balancing retain their checked arithmetic,
  rejection conditions and existing owned-state boundary.
- The merge keeps `AdmittedRows::payload`, `cursor_row`, and `dense_key_range`.
  Output materialization and owned gather use payload-only access; retention
  and dictionary preparation still use the required row metadata.

The new workload-consistency fixtures accept the current exact source or an
explicit direct migration for the same target. Their six negative cases
reject a wrong target, reversed direction, unknown declared bytes, unknown
current bytes, a missing declaration, and a merely transitive chain. An
immutability assertion protects the supplied migration data. Production
attestation validation is unchanged.

### Test-clock repair

The failing lifecycle fixture deliberately freezes the job's watch clock,
waits four seconds beyond its three-second wall-clock deadline, then advances
the controlled clock only after a worker owns its lease. Before the repair,
`output_gather::owned::cancelled` still used real UTC time, while
`StreamJobContext::check_cancelled` used the controlled clock. The wait could
therefore return before the fixture expired the job and reach the retained
preparation's defensive `unreachable!`.

Commit `1c086de8` adds eight lines under `cfg(test)` to wait on the configured
watch clock when a deadline is present. Its `>=` predicate matches the job
check and `GatherStop::wait`. The owned receiver clone and short-lived result
borrow introduce no guard across a later await; cancellation selection and
the fallback clock path retain their order. Production configuration is
unchanged, and the existing failing fixture is preserved as the red test.
No source blocker was found in this repair. The four selected deadline and
budget tests passed on that source, including the previously failing fixture.

### Route borrow repair

Clippy rejected the extracted helper's `Option<&mut Route>` value parameter.
Commit `25e8b4c3` changes the caller to borrow the existing optional route and
the helper parameter to `&mut Option<Route>`. Matching still borrows the same
bucket, cursor and latest-source cache; the empty route, updates, fallible
operations and ordering are unchanged. No route is cloned or moved and no
allocation is added. No source blocker was found in this two-line repair;
its seven focused ASOF tests and repaired Clippy check passed.

No passing count from an earlier implementation is relabelled as a result
for the final source.

## Documentation Consistency

The refactor preserves the behavior documented by the
[CPU pipeline specification](../specs/join-cpu-pipeline.md), ASOF guide and
streaming guide; no public command or architecture update is required for
these helper extractions.

The new workload registry entries explain their exact source-byte changes,
historical output budget and direction. They preserve historical dependency
attestations and timing evidence. The original review and performance report
remain bound to their original measured source. Any remediation timing claims
must cite the new comparison and its limitations.

## Performance Verification

The comparison uses the original default release at `61aab0e5` and the sealed
default release at `25e8b4c3`, with the same maintained harness on both arms.
Exactly five configurations completed: two rounds of ten adjacent AB/BA pairs
per configuration, with reversed first-pair order in round two. The final
inventory is **200 formal observations, 20 excluded warmups and 20 completed
workers**. There were no cleanup errors or added diagnostic runs.

The first attempt failed its native identity check before hello, preparation,
warmup or sampling because a relocated manifest directory was incorrectly
treated as the package site. The reviewer approved one technical recovery
after checking all 18 archived files, the unchanged worker and build manifests,
and the strict package/native checks. The repaired controller derives the site
from the validated native path. The failure consumed 2.265889 seconds; the
corrected acquisition consumed 120.411500 seconds. Their **122.677389-second
combined cost** remains within the original 600-second budget. No statistical
retry, sample expansion or source change occurred.

Positive P50 changes below mean slower. The intervals are the two rounds'
paired-median percentage-change intervals, each with 97.8515625% coverage;
they are distinct from the descriptive ratio of aggregate P50 values.

| Configuration            | P50 change | Round 1 paired CI  | Round 2 paired CI | Maintained verdict      |
|--------------------------|------------|--------------------|-------------------|-------------------------|
| ASOF 1M, 1 CPU           | +1.13%     | [-5.68%, +13.80%]  | [-2.62%, +11.36%] | inconclusive            |
| ASOF 1M, 32 CPUs         | +1.64%     | [-12.02%, +15.18%] | [-3.15%, +23.42%] | inconclusive            |
| Interval 262144, 1 CPU   | -1.23%     | [-3.08%, +0.11%]   | [-5.70%, +1.01%]  | no-confirmed-regression |
| Interval 262144, 8 CPUs  | -3.67%     | [-6.99%, +11.01%]  | [-9.18%, +4.28%]  | inconclusive            |
| Interval 262144, 32 CPUs | +1.55%     | [-3.11%, +8.48%]   | [-1.04%, +11.15%] | inconclusive            |

The maintained rule requires both rounds' lower bounds above +5% to confirm
a regression. No case met that rule, and no case established an improvement.
The four inconclusive results do not establish equivalence or absence of a
regression. No retries were used to seek a different verdict.

The independent performance audit verified raw worker responses, source/build
identity, fingerprints, pair order, all-row correctness, resource and completion
evidence, and recomputed the maintained statistics and order-statistic
intervals. The reviewer checked the resulting inventory and verdicts against
the completed report. Evidence seals are:

- `reports/measurement.json`:
  `13b1e2afddc6fe1019edc0bd4fff183d3f1ba3357d71d8f26280079700c2d74e`.
- `formal-offline-audit.json`:
  `12e2515a6b8e801d87ae2f14eb224deb14e067767fdf2f55fa8e66a11fcebf0d`.
- Approved recovery plan:
  `3c0e89048f27dfc866094ba27d16f136081a768805494a734078684bf3273950`.

These results cover the fixed ASOF and interval configurations. Copied ASOF
row gathering, rolling performance, small/skewed/wide/nested inputs,
long-lived state and checkpoint/recovery timing remain unmeasured. WSL CPU
affinity does not establish the host's physical-core topology or power state.

## Verdict

**Approve — reviewed source and completed focused verification.**

No source blockers or unresolved local failures remain at `25e8b4c3`. The
bounded performance task is complete with no confirmed regression and four
inconclusive cases. This is source-review approval, not a claim of green CI,
performance equivalence, or merge readiness. Required CI and Codacy acceptance
remain pending, and merge still requires explicit authority.
