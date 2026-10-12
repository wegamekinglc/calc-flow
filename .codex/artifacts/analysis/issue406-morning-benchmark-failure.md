# October 12 morning benchmark investigation

## Confirmed failure

[Scheduled run 38178694560](https://github.com/wegamekinglc/calc-flow/actions/runs/38178694560)
started at 06:00:49 Asia/Shanghai and completed at 08:11:15. The complete report
contains 809 result rows and exactly one error:
`engines/10000000/calc-flow-stream/group_by: repeated regression exceeds +5%`.
The engine shard failed its regression check; the aggregate gate propagated that
failure. This was neither a missing-result failure nor a failed correctness oracle.

| CI observation             | Value                                      |
|----------------------------|--------------------------------------------|
| Baseline                   | `89e511416d5793e11cefd74b1a7c09aae96f81bd` |
| Candidate                  | `8ee5b71b7b862d4e44bd0be390d182cf950d17f7` |
| Baseline / candidate p50   | 853.563 / 953.112 ms                       |
| Combined p50 change        | +11.663%                                   |
| Round 1 paired median / CI | +11.740% [+11.251%, +12.360%]              |
| Round 2 paired median / CI | +11.583% [+11.330%, +11.790%]              |

Each round contains ten adjacent AB/BA pairs with successful oracles. Both lower
confidence bounds exceed the maintained +5% threshold, so the original run is a
valid regression verdict under that rule. The exact order-statistic interval is
nominally 95%, conservatively 97.85% at ten pairs under its iid assumption.
No threshold, sample filtering or verdict was changed.

The original native SHA256 values are:

- Baseline: `48e63a99052bece0cf59309b9057e3e950a8673383a0fbf3b6ca5e1cdc5454a6`.
- Candidate: `ef24e1d8fefb075909f602a054b03a621200abe4230823fd8259ad2b3a9defa5`.

Both release manifests record Rust 1.88.0, Python 3.13.16, equal Cargo lock hashes
and clean exact source revisions. The runner exposed four logical CPUs; both
arms used Tokio/Polars pools of 32, NumPy 2.5.2 and PyArrow 24.0.0. These thread
settings are existing catalog settings, not a newly imposed runtime restriction.

## Narrowing the cause

The source delta changes ASOF key admission and tests. The window/group aggregation
implementation, benchmark inputs and group_by timing adapter are unchanged.
The same morning run's 1M group_by pair was -0.698%, with both round intervals
below +5%. Other 10M paths did not trigger regression. The preceding manual full
run 38164813243 reported 10M group_by 770.869 / 771.040 ms (+0.022%); that run has
different artifacts and host history and cannot invalidate the morning result.

Downloaded the original wheels and verified both wheel/native seals using the
maintained release loader. A fixed local diagnostic reused the maintained
`measure_case` collector, oracle, alternating pairs and verdict, with A/A, B/B,
then A/B. Each comparison had two rounds of ten pairs and one warmup per side per
round. The recorded total budget was 600 seconds, including creation, startup,
oracles and cleanup; no builds were needed. Collection completed in 119.327 seconds
with no observed native-build overlap.

| Local diagnostic | First arm p50 | Second arm p50 | Change  | Verdict                 |
|------------------|---------------|----------------|---------|-------------------------|
| A/A              | 489.122 ms    | 489.322 ms     | +0.041% | no-confirmed-regression |
| B/B              | 494.282 ms    | 494.593 ms     | +0.063% | no-confirmed-regression |
| A/B              | 493.931 ms    | 498.305 ms     | +0.885% | no-confirmed-regression |

A/B paired intervals were +0.635% [-1.219%, +2.336%] and +0.509%
[-0.456%, +2.745%]. This local WSL host (Python 3.13.9, affinity 0–3 on a 32-CPU
host) is not the original GitHub environment. It did not reproduce +11.66%; it
does not establish equivalence or explain the original cause. Artifact seals
match the morning wheels; host/interpreter differences remain explicit.

## Follow-up and limits

Triggered exactly one existing `suite-recovery` run,
[38193911010](https://github.com/wegamekinglc/calc-flow/actions/runs/38193911010),
for the failed `engines-10000000` shard. It reuses the original run's sealed wheels
and exact source refs, preserves the complete shard and unchanged gate, and keeps
resource diagnostics. It does not rebuild or substitute the current ASOF feature.
Its completion and raw evidence must be evaluated before claiming the failure
resolved. Do not retry repeatedly for a passing result.

The workflow checkout for recovery is release main `8ce5380b`, advanced after the
morning run; its product inputs remain the explicit morning refs. The release
version update must not be mislabeled as the measured product.

Root cause attribution is still open. A host-dependent binary/code-layout effect
or runner behavior is a hypothesis, not an established finding. No production
change or benchmark-gate relaxation is justified by these results alone.

Raw logs, manifests, original reports, failed-case samples, local fixed plan,
per-comparison evidence and resource trace are preserved under the main checkout's
`target/issue406-morning-benchmark-failure/`.
