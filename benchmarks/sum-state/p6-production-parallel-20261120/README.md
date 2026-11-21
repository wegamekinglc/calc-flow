# P6 production parallel activation evidence

## Outcome

Real production parallel activation is confirmed, but it is slower on eight CPUs and does not approach the P6 four-core 3x gate.

| Case | Baseline P50 | Candidate P50 | Time change | Verdict |
|------|-------------:|--------------:|------------:|---------|
| stock-stream-1cpu | 201.765 ms | 200.433 ms | −0.66% | no confirmed regression |
| stock-stream-8cpu | 137.602 ms | 159.104 ms | +15.63% | inconclusive |
| sparse-stream-1cpu | 21.106 ms | 21.891 ms | +3.72% | inconclusive |
| stock-batch-8cpu | 59.151 ms | 57.182 ms | −3.33% | inconclusive |

The compatible serial baseline is `2ac1af5f`, which contains the same diagnostic harness interface as the candidate `6dcacab6`. The earlier `aefd17dc` harness was rejected by identity validation and was not substituted with an unchecked comparison. Each case completed two rounds of ten AB/BA pairs with full snapshot and lifecycle oracles. Acquisition took 119.586 seconds; the two release builds took 839.822 seconds (baseline) and 637.970 seconds (candidate) separately.

A separate 70,001-row test proves that at least two real workers activate, output matches the naive oracle and all memory is released. SQL 345 tests and all-feature Clippy passed. That activation correctness does not establish a speedup.

## Stage attribution

The candidate's scoped workers execute during the measured path. Example first-round route/apply times were approximately 60/53 ms for the serial baseline and 65/73 ms for the parallel candidate, so activation did not reduce apply time. Candidate process CPU was also higher. The next experiment must reduce per-old-group global KeyStore locking without adding serial partition scans.

## Evidence

- `plan.json`: cases, budgets, source seals and build receipts.
- `environment.json`: host, affinity and harness identity.
- `samples.csv`: all 160 formal observations, including `keyed_sum_stage_ns`.
- `summary.json`: paired statistics and resource summaries.

No retries or additional cases were used. The raw acquisition remains local under `target/p6-keyed-measurement-parallel-6dcacab6-2ac1af5f/`; compact evidence is committed here.
