# P4 dense 10M grouped SUM — gate failed

Candidate `aa63c80c` was measured with the original `30_streaming_stock_positions` workload: 10,000,000 rows, 100x100 key groups, 1,000,000-row batches, one CPU and the example's built-in `check_positions` validation on every completed process. Two warmups and ten measured processes completed in 29.587 seconds within the 600-second budget. No retries or extra cases were used.

| Baseline P50 | Candidate P50 | Candidate P95 | Target | Verdict |
|-------------:|--------------:|--------------:|-------:|---------|
| 1,669 ms | **2,448 ms** | 2,502 ms | ≤500 ms | **Failed** |

The candidate is slower than the historical 1.669-second baseline and does not meet the absolute target. The independent P6 paired results in `../p6-dense-utf8-20261121/` also show regressions. This experiment is rejected and reverted rather than being presented as a valid keyed-state improvement.

`result.json` records every process's wall time, example-reported elapsed time, correctness result, binary identity and budget. The baseline comes from `../../stock-positions/engine-core-sweep-20261107.md` and uses the same example and built-in check.
