# P6 dense SUM vectors — improved, gate still open

Runtime candidate `439395a5` replaces per-partition `BTreeMap<GroupId, SumValue>` values with dense vectors indexed by `group_id / N`. Routing, marks, first-seen order, undo and checkpoint representation are unchanged. The sealed build receipt has source HEAD `e205b0bf` after documentation-only corrections, binary SHA-256 `d905f4f0e1f78de6c5422738f9c83d056e2c8a83967b2f500b4073cfa8925bae`, and took 1,035.377 seconds separately.

Seven focused keyed-SUM tests, the complete 344-test SQL module and all-feature Clippy passed before sealing. Two rounds of ten AB/BA pairs completed 160 observations in 120.656 seconds with full snapshot and lifecycle oracles passing.

| Case | Baseline P50 | Candidate P50 | Time change | Descriptive speedup | Verdict |
|------|-------------:|--------------:|------------:|--------------------:|---------|
| stock-stream-1cpu | 204.988 ms | 136.503 ms | −33.41% | 1.50× | improved |
| stock-stream-8cpu | 138.317 ms | 97.288 ms | −29.66% | 1.42× | improved |
| sparse-stream-1cpu | 21.427 ms | 15.022 ms | −29.89% | 1.43× | improved |
| stock-batch-8cpu | 56.234 ms | 56.271 ms | +0.07% | 1.00× | inconclusive |

The four-physical-core ≥3× gate is **not met**. The candidate remains because it is a statistically supported improvement, not because the absolute gate passed.

A diagnostic stock one-CPU observation reduced keyed apply from approximately 72 ms in the old implementation to **21.506 ms**, while route remained **58.818 ms**. Candidate operator processing was 86.820 ms, with expression processing another 43.080 ms. These single-observation stage values explain the next target but are not medians or independent acceptance evidence. Further work must address key routing and whole-batch execution costs rather than merely retrying the previously rejected parallel apply design.

Raw acquisition is in `summary.json`, `samples.csv`, `environment.json` and `driver-plan.json`. The separate 10M P4 acquisition uses the same runtime candidate and is registered in `../p4-vector-sums-10m-20261121/plan.json`; no P4 result is claimed here.
