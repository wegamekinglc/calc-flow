# P6 worker-prefetch lease evidence

## Outcome

Worker-local old-group lease prefetch reduced some of the parallel overhead but still failed to establish a speedup. Default parallel execution has been disabled again by reverting both the prefetch and record-level activation changes.

| Case | Baseline P50 | Candidate P50 | Time change | Verdict |
|------|-------------:|--------------:|------------:|---------|
| stock-stream-1cpu | 197.306 ms | 198.204 ms | +0.45% | no confirmed regression |
| stock-stream-8cpu | 137.729 ms | 150.149 ms | +9.02% | inconclusive |
| sparse-stream-1cpu | 21.094 ms | 20.984 ms | −0.52% | no confirmed regression |
| stock-batch-8cpu | 56.605 ms | 55.906 ms | −1.23% | inconclusive |

The compatible serial baseline is `2ac1af5f`; the candidate is `84570246`. Each case completed two rounds of ten AB/BA pairs with complete snapshot and lifecycle oracles. Acquisition took 120.652 seconds; release build took 632.512 seconds separately.

The eight-CPU result is still slower than both the serial baseline and the unmodified `6dcacab6` activation path's 159.104 ms. No 3x scaling exists. The changes were reverted in `ebc64d9f` (prefetch) and `3c830fee` (activation), restoring the relevant source to `2ac1af5f`.

This closes the current thread-per-record and lock-prefetch approach. A later P6 attempt needs a different routing and state-ownership design rather than more scoped threads or lease batching. All raw results remain available even though the implementation was withdrawn.

## Evidence

- `plan.json`: cases, budgets, source seals and build receipts.
- `environment.json`: host, affinity and harness identity.
- `samples.csv`: all 160 formal observations, including `keyed_sum_stage_ns`.
- `summary.json`: paired statistics and resource summaries.

No retries or additional cases were used. The raw acquisition remains local under `target/p6-keyed-measurement-worker-prefetch-84570246/`; compact evidence is committed here.
