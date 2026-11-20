# P6 dense partition marks evidence

## Outcome

Dense per-partition marks removed the measured serial and multicore regressions from the first fixed-partition implementation. The candidate is back near the serial baseline, but the P6 four-core 3x speedup gate is still **not met**.

| Case | Baseline P50 | Candidate P50 | Time change | Verdict |
|------|-------------:|--------------:|------------:|---------|
| stock-stream-1cpu | 204.111 ms | 198.546 ms | −2.73% | no confirmed regression |
| stock-stream-8cpu | 134.559 ms | 136.283 ms | +1.28% | no confirmed regression |
| sparse-stream-1cpu | 20.062 ms | 20.402 ms | +1.70% | inconclusive |
| stock-batch-8cpu | 53.441 ms | 54.028 ms | +1.10% | inconclusive |

The baseline is `aefd17dc006270457c2e8ccc585111d6f4f1b5ea`; the candidate is `d80a334a137fb72fe4513fd923f5cbfb23943ec8`. Per-partition BTreeMap marks are replaced by dense `u64` vectors indexed by `group_id / N`, without changing SUM, undo or checkpoint layout identity.

Each case completed two rounds of ten AB/BA pairs with full snapshot and lifecycle oracles. Acquisition took 118.698 seconds within the 600-second budget; release build took 628.631 seconds separately. The candidate binary SHA-256 is `d9453dd8c07ae4b5cfe16eaa644f9f99bede1ca9658f488325548e478e25574e`.

The earlier `212ca55b` candidate took 249.977 ms (1 CPU) and 164.141 ms (8 CPU) in the first comparison, with confirmed regressions of +25.06% and +20.52%. Those regressions are no longer present. This is a correctness-of-performance repair, not evidence of the required parallel speedup; shared-lock and ownership costs remain the next measured target.

## Evidence

- `plan.json`: cases, budgets, source seals and build receipts.
- `environment.json`: host, affinity and harness identity.
- `samples.csv`: all 160 formal observations.
- `summary.json`: paired statistics and resource summaries.

The maintained `scripts.benchmark_suite.sum_state` driver was used with no profiler and no retries. The raw acquisition remains local under `target/p6-keyed-measurement-dense-marks-d80a334a/`; compact evidence is committed here.
