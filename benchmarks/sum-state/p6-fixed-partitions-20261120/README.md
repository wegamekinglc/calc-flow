# P6 fixed-partition keyed SUM evidence

## Outcome

The fixed-partition parallel implementation is **not a performance upgrade** in this comparison. The P6 four-core 3x speedup gate is not met. Both dense streaming cases are regressions under the maintained decision rule; neither control establishes equivalence.

| Case              | Baseline P50 | Candidate P50 | Time change | Verdict      |
|-------------------|-------------:|--------------:|------------:|--------------|
| stock-stream-1cpu | 199.879 ms   | 249.977 ms    | +25.06%     | regression   |
| stock-stream-8cpu | 136.195 ms   | 164.141 ms    | +20.52%     | regression   |
| sparse-stream-1cpu | 20.852 ms    | 21.931 ms     | +5.18%      | inconclusive |
| stock-batch-8cpu  | 53.820 ms    | 54.802 ms     | +1.82%      | inconclusive |

The baseline is `aefd17dc006270457c2e8ccc585111d6f4f1b5ea` (fixed-N ownership with serial apply). The candidate is `212ca55b8b8a7deabd37a33f105122c6458e31a8` (scoped parallel apply). Each case completed two rounds of ten AB/BA pairs with full snapshot and lifecycle oracles. Acquisition completed in 120.299 seconds within the pre-registered 600-second budget; no retries or additional cases were used.

Release builds are reported separately: baseline 671.666 seconds and candidate 970.241 seconds. The baseline binary SHA-256 is `c8cae2ab4407eba8ea2b2ccf630014cbb0cc3ddb27bc819d8e11669668c5929e`; the candidate binary SHA-256 is `68ec757f5da505a964e4189ae12d5551b068e9069a4ae738c74dc1571948374c`.

## Evidence

- `plan.json`: pre-registered cases, budgets, source seals and build receipts.
- `environment.json`: host, affinity and harness identity.
- `samples.csv`: all 160 formal observations.
- `summary.json`: per-round paired statistics and resource summaries.

The collection used the maintained `scripts.benchmark_suite.sum_state` driver with no profiler. Its four workloads are 10k seeds plus 20 batches, with 100k dense rows or 100 sparse updates per batch. The eight-logical-CPU cases report four physical core IDs on this host. Finite execution is a control, not the P6 parallel-state acceptance path.

## Reproduction

From the candidate repository root, with no other builds or benchmarks running:

```bash
uv run python -m scripts.benchmark_suite.sum_state \
  --baseline-build target/p6-keyed-build/baseline/build.json \
  --candidate-build target/p6-keyed-build/candidate/build.json \
  --output /absolute/path/to/new-p6-keyed-measurement
```

The local raw acquisition, including every per-process command and oracle receipt, remains under `target/p6-keyed-measurement-212ca55b/`; only the compact evidence above is committed.
