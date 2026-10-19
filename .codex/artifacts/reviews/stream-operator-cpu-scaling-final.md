# Branch Review: final stream operator CPU scaling scope

**Branch:** `feature/stream-operator-cpu-scaling-final`

**Feature base:** `e35ff1190e8eba08ad3c3135299bbca6ac029698`

**Reviewed withdrawal:** working diff over
`ca80229738f169ae6880f72392683477f8e03c7f`

**Files:** eight withdrawal/source-documentation files, plus the historical
review preface and this report

This review uses the implementation owner's isolated final worktree and leaves
it available for the pending release build. The reviewer changed only these
two review documents and ran no native build, test or measurement. No PR,
check-run snapshot or prior GitHub review exists for this working diff; no
remote action was taken.

## Summary

The final candidate retains CPU-sized ASOF and rolling work and immutable
compiled execution for eligible row-local SQL. It removes the cross-section
parallel experiment and restores that operator exactly to the accepted base.

The recorded first comparison explains the withdrawal: cross-section median
latency increased from 10.304 to 24.104 ms, or 133.93%. Both round confidence
intervals exceed the maintained +5% regression threshold. That original
candidate and its complete evidence remain separate from the final candidate.

## Build and Test Results

- Rust source verification: **Passed**. A byte comparison against Git blobs
  confirms that `cross_section.rs` is exactly the `e35ff119` module, all three
  experimental cross-section modules are absent, and all **24 retained changed
  Rust files** are exactly their previously reviewed `ca802297` contents.
  The entire product difference from `ca802297` consists of that restoration
  and those three removals. No stale parallel-mean reference remains in source
  or current public guides.
- Prior Rust verification belongs to `ca802297`: **29 new tests and 140 focused
  existing regressions passed**, together with scoped format and Clippy. Six
  new cross-section experiment tests were removed; the **23 retained new test
  cases** and their implementation sources are byte-identical. These results
  are inherited evidence, not a claim that the final binary was tested.
- Final release compilation and the planned **21 focused installed-wheel Python
  checks are pending** at this source freeze. They must pass against the exact
  final candidate before its verification conditions are closed. The earlier
  candidate's package results must not be relabeled as final results.
- Reviewer `git diff --check ca802297` and
  `git diff --exit-code e35ff119 -- schemas/project-v3.schema.json`: **Passed**.
  The final review documents also pass whitespace and final-newline checks.
- New unresolved test failures: **None observed**. No fresh final-candidate
  Rust test run or full workspace, connector, cross-platform or coverage run
  was performed. Required CI and the 90% coverage gate remain uncertified.
- Performance regression: the original cross-section experiment is confirmed
  regressed. The source responsible for that experiment has been withdrawn;
  final-candidate timing is still pending.

The final `cross_section.rs` SHA-256 is
`013a39f73c81767f7661fd5ffbb9fa79a659df83b499db120523452ef3f5d865`.
Identity checks read the complete product diff and compare complete file bytes,
including the retained tests, against their designated commits.

## Blocking Issues

No source finding remains. The pending exact-source release compilation and
21 focused wheel checks are verification conditions, not passing results.

A separate final Rust unit-test build is not required solely for this
withdrawal. It introduces no new production logic, restores the complete
cross-section module rather than combining old/new state layouts, removes
its test-module registration with the removed tests, and leaves every retained
changed Rust file identical to the reviewed/tested candidate. Final release
compilation checks the resulting module graph; focused wheel tests exercise
the installed final runtime. A build or test failure would require diagnosis
and could justify additional narrowly scoped Rust verification.

## Style Issues

None found. The restoration preserves the original private state layout,
public surface, logical state budgeting, immutable input ownership and existing
checkpoint representation. No dependency, lint exception or schema change was
introduced. The historical review is explicitly labeled with its original
candidate identity.

## Test Coverage

The retained ASOF tests cover configurable prefixes, workspace fallback,
cancellation and routing beyond the persistent shard count. Rolling retains
dynamic lane/one-CPU selection, stable schedules, reused entity counts, exact
floating-point output and state, all-lane primary-error selection and owned
abandonment/drain tests. SQL retains compiled hot-path reuse, error/cancellation
and unsupported-plan fallback checks. Their prior evidence and limits are
documented in the historical source review.

The restored cross-section implementation has no pending parallel work or
new boundary sidecar to settle during restore, reset or emission. Its original
columnar order proof, pending RecordBatch, serial mean construction and
checkpoint materialization are restored together. Removing only the six
experiment-specific tests preserves the existing cross-section tests. No
new inter-operator lifecycle dependency is introduced by this restoration.

## Documentation Consistency

The specification, changelog, runtime-envelope guide and streaming guide now
describe the retained three-path scope. They remove claims that cross-section
mean uses the new CPU service, retain it as a measurement regression control,
and preserve the original rejected comparison as historical evidence. The
ASOF and SQL guides remain identical to the reviewed original candidate.

The first acquisition's stored report records 48.114798914 seconds and the
cross-section regression quoted above. The proposed corrected-source run uses
the same five configurations, two rounds of ten adjacent pairs, 200 formal
observations and 20 warmups, within 551 seconds of remaining acquisition
budget. A changed exact candidate is required; no old samples or original
candidate release may stand in for the corrected source. Builds are recorded
separately. This review does not certify a final performance result or grant
measurement execution authority.

No stale current-behavior documentation, Markdown whitespace or link issue was
found in the reviewed withdrawal. The historical review's link resolves to
this report.

## Verdict

**Approve** the exact withdrawal and retained source, conditional on successful
compilation of the frozen final release and its 21 planned focused wheel tests.
No additional source fix or Rust test rebuild is required by this review.
Performance acceptance, required CI and coverage remain separate gates; the
original candidate's successful tests and rejected performance result retain
their original identities.
