# P6 dense UTF-8 keyed SUM experiment — rejected

Candidate `aa63c80c` introduced borrowed two-UTF-8-key lookup and dense Int64 accumulation. The independent P4 measurement is in `../p4-dense-10m-20261121/result.json` and is also a failure. **This implementation is not a performance upgrade and must not remain in the default path.**

The compatible recent baseline is `2ac1af5f` with the current harness. The earlier `aefd17dc` source was rejected before measurement because it lacked `keyed_sum_stage_ns`; no samples ran in that protocol attempt. Both arms were then built with the same harness. The driver completed 160 observations in 127.048 seconds within the 600-second budget, with full snapshot and lifecycle oracles passing in every case.

| Case | Baseline P50 | Candidate P50 | Time change | Verdict |
|------|-------------:|--------------:|------------:|---------|
| stock-stream-1cpu | 202.164 ms | 369.360 ms | +82.70% | regression |
| stock-stream-8cpu | 137.041 ms | 254.165 ms | +85.47% | regression |
| sparse-stream-1cpu | 20.633 ms | 33.882 ms | +64.21% | regression |
| stock-batch-8cpu | 56.184 ms | 55.019 ms | −2.07% | inconclusive |

The dense path bypassed the old route/apply instrumentation, so the regression's precise stage attribution is not established by this acquisition. The results do not justify a claim that borrowed lookup, dense updates or transaction commit individually caused it. A future attempt needs direct stage measurements before another performance comparison; this experiment is reverted rather than retimed until a favorable median appears.

`summary.json`, `samples.csv` and `environment.json` retain the maintained acquisition evidence; the oversized acquisition dump was removed during PR cleanup. The implementation candidate's unit tests passed, including rollback, but those correctness checks do not establish performance. The candidate and recent-baseline release builds took approximately 1,006.5 and 979.6 seconds separately.
