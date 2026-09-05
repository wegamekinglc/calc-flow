# Small-append streaming performance follow-up

Status: implementation and measurement complete; the small-append latency
breakthrough is not established. Baseline:
`6515f4d19317f6b05573f41078177d866cc1eee1`.
Continue the existing PR #242 without merging it or changing the original
two-round report's measured identities.

## Scope and order

- [x] Add legal sparse-entity workloads with 1/4/16/64 appended rows, separate
  from the original complete-entity-tick matrix. Validate warm and appended
  identity, order, numerical values, finality, and advancing state.
- [x] Add opt-in callback boundary diagnostics, excluded from authoritative
  uninstrumented timings. Separate attachment/preparation, loop dispatch,
  callback completion, and Rust resumption; wall intervals are not CPU shares.
- [x] Reduce connector callback attachment/scheduling round trips while
  preserving cancellation, context isolation, custom task factories, failure
  cleanup, and owned GC roots.
- [x] Avoid rebuilding row-count maps for resident entities. Keep existing
  encoding and atomic state staging until their further optimization is
  justified by measurement; preserve numerical rebasing, output budgets,
  fallback behavior and checkpoint recovery.
- [x] Evaluate whether safe physical projection fusion is justified after the
  bridge/state changes. Keep only demonstrated improvements; any deferral must
  identify the measured cost and semantic tradeoff.
- [x] Compare clean release builds with paired samples, independent scenario
  workers and randomized case order. Preserve forced/normal GC as separate
  modes, record P50/P95 and confidence intervals, and rerun the original matrix.
- [x] Add correctness/allocation/lifecycle regressions and measured evidence
  to the same PR. Final-head CI/coverage/review status is maintained in the PR
  after each push; it is not inferred from earlier commits or this checklist.

## Constraints

Do not remove Python from the published end-to-end timing boundary, weaken
watermarks or cancellation, replace the StreamingRunner with a public push
runner, or trade output latency for larger hidden batches. Source-protocol
redesign and durable checkpoint-format changes are outside this follow-up.
The primary dirty checkout is not an implementation target.

Every behavior change starts with an observed failing regression. Benchmarks
run only after compilation/tests finish, with separate Cargo target directories
and immutable baseline/candidate provenance. No numerical speedup is promised
before paired measurements.

## Verification record

- Sparse workload RED: the old `ScenarioConfig` rejected `append_entities`.
  GREEN: 48 workload/oracle tests passed against the baseline native wheel,
  including independent full-history oracle comparisons across window bounds.
- Callback diagnostics RED: `_StreamingJob` lacked `_take_callback_profile`.
  The first instrumented candidate wheel passed all 961 Python/workload tests.
  Final cleaned-source wheel validation remains required.
- Resident-entity allocation RED: 64 already known entities allocated 7 times.
  An intermediate `Option<Vec<_>>` collect still allocated 5 times because its
  iterator did not guarantee its lower size bound. Use explicit result capacity;
  do not relax the one-allocation assertion.
- Controller RED: fresh/seeded/sparse case helpers were absent. GREEN: all 20
  controller tests passed, including worker cleanup on failed measurements.
- Full Rust compilation initially caused substantial swap pressure. It was
  stopped and resumed with two build jobs. No benchmark was timed concurrently.
- Core unit GREEN: 765 passed, four opt-in soaks ignored; the resident-entity
  regression now passes at one allocation. PyO3 unit GREEN: 112 passed before
  the final error-label-preservation cleanup; rerun on the frozen source.
- Final-source all-feature/all-target Clippy, formatting, Ruff and complexity
  gates passed. Release measurements and full integration/remote gates pending.

## Final evidence

See [the measured report](../../small-append-results-818d69c.md). Clean engine
builds compare `6515f4d` with `818d69c`. The original and sparse matrices,
same-wheel 32-to-1/4 worker experiments, and declared 100-pair follow-up retain
all 7,600 uninstrumented samples, including unfavorable and inconclusive
results. All pass strict correctness. Two separate instrumented runs add
60 correct appends without being used as speedup evidence.

The initial 4% negative signal did not establish a regression in the declared
follow-up. Most small-row intervals still include no improvement. Keep the
one-allocation resident-entity optimization, but do not claim a small-batch
latency breakthrough. Neither scheduler defaults nor source protocols change.
Projection computation averages about 7.2 microseconds in the representative
counter delta; task/channel fusion remains unimplemented because its separate
benefit and semantic obligations were not isolated by these counters.

Final local validation passes the full Rust harness, three serial 112-test
PyO3 runs, 962 Python/workload tests and 23 controller tests. Remote failures
found in the first push were traced to timezone-database-dependent benchmark
conversion, a test relying on an existing target directory, and analysis
complexity/random-generator warnings. Their fixes preserve timed engine code
and have focused regressions. Current remote acceptance is recorded at the
published head in PR #242; this work does not authorize a merge.
