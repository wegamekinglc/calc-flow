# Weekly implementation review

Review mainline changes committed from 2026-10-31 00:00 Asia/Shanghai through
the captured current main `1b91bcc00a0623d5ece37ac654bf6e2a4f045939`.
The preceding mainline commit is
`ce313488017d63fc626731f9731a4bec931fc343`; the interval contains 36 mainline
commits. Audit the resulting implementation, including adjacent callers and
tests, rather than assuming every historical experiment remains enabled.

The user requests simplification, optimization and bug fixes, a grounded plan
for remaining performance opportunities, and protection against benchmark
regressions. Preserve the original dirty stock-comparison worktree. All edits
use the isolated `fix/weekly-review-20261107` worktree; generated evidence uses
the repository `target/weekly-review-20261107` tree.

Review responsibilities:

- SUM state, streaming joins/ASOF and runtime: `weekly_sum_stream`.
- SQL planning, native joins, rolling extrema and finite parallelism:
  `weekly_sql`.
- Python configuration, array/stream inputs and Arrow validation:
  `weekly_python`.
- Benchmark tooling, integration, remediation selection and delivery: parent.

Confirm a concrete reproduction before changing semantics. Every behavior
fix starts with an observed focused failing test. Prefer small conservative
admission/fallback fixes over new algorithms during this review. Do not
change public schemas, historical benchmark evidence, statistical thresholds
or retired experiment source merely to simplify the working implementation.
Review actual final changes independently before delivery.

Before performance execution, append the exact baseline/candidate identities,
selected cases and rationale, fixed samples, maintained harness/statistics,
build estimate and acquisition budget. Select at most five representative
cases; acquisition totals at most 600 seconds including fixture preparation,
warmup, correctness, startup and cleanup. Build costs are separate. Stop if
the environment is unsuitable, preserve partial results and settle owned
processes. Do not expand samples or retry for a favorable result.

Apply the maintained two-round paired 5% regression rule when using the
unified suite. An inconclusive result is neither a confirmed regression nor
proof of equivalence. Do not claim an unmeasured future proposal has no
regression: require each proposal to include its expected benefit, semantic
risks, relevant controls and an explicit acceptance/withdrawal gate.

Local verification is focused. Full regression, coverage and platform checks
belong to CI. Record reviewed coverage, unresolved findings, tests and actual
performance limitations accurately. Existing authorization permits a reviewed
PR and merge only after required final checks succeed.

## Fixed performance controls

The implementation repairs admission and accounting rather than introducing
a new optimization algorithm. Compare exact sealed release wheels from the
captured main and the eventual frozen fix commit. Reuse `EngineCase` for the
first three workloads and the maintained Program fixture/declaration/oracle
for the fourth; no complete benchmark suite or optional extra case is selected.

| Case                     | Input                  | Connection to the change                         |
|--------------------------|------------------------|--------------------------------------------------|
| sql-asof-1000            | 1,000 rows             | Canonical native ASOF admission and metrics      |
| sql-interval-1000        | 1,000 rows             | Canonical native interval admission             |
| stream-asof-10000        | 10,000 rows; batch 1024 | Projected fixed-width compact output accounting  |
| sequence-projection-1000 | 1,000 rows; 640/360     | List capture and ordinary Arrow sequence replay |

Every case uses two rounds of ten alternating pairs through maintained
`release_pairs.collect_case` and `evaluate_case`. Each observation owns a fresh
process, one warmup and three timed samples; its median is the paired value.
The frozen campaign therefore has 160 worker processes, 160 warmups and 480
timed samples. Keep each raw sample, full oracle result, resource observation,
fixture/machine/dependency fingerprint and exact native/wheel/source identity.
All workers use CPU 0, one Tokio worker and one external helper/Arrow pool
thread. The supervisor and controller use CPU 31. The two-second guest quiet
guard requires selected-CPU and average load no greater than 10%; external WSL
host load and power remain unobservable.

The original 600-second wall clock includes controller startup, preflight,
wheel installation, fixtures, warmups, correctness, samples, reduction,
postseals and process settlement. Interrupt at 585 seconds, with all cleanup
capped by the original deadline. All builds and focused tests finish before
acquisition. Budget builds separately at approximately 15–20 minutes per
wheel; ordinary dependency cache reuse does not replace an exact source build.
There is no scientific probe, automatic retry or additional sample allocation.

Previously wrong CTE/alias queries are correctness regressions tested against
DataFusion, not equal-output latency comparisons: correct fallback can cost
more than an invalid shortcut. The ordinary benchmark controls remain admitted.
The nullable-buffer accounting case is covered by exact byte/budget tests;
its normal ASOF performance control uses nonnull prices. Configuration-error
serialization and the benchmark-only setter are outside ordinary hot paths.

Reject a confirmed maintained regression; retain inconclusive results and do
not label them equivalence. Future algorithm proposals need a new explicit
benefit hypothesis, relevant controls and the same correctness/resource and
regression gates before they may replace the current implementation.
