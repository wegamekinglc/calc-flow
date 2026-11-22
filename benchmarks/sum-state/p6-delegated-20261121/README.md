# P6 DataFusion delegation — rejected candidates

Candidate `5d666afa` required a `sequence` column to order grouped output by each group's first row. The maintained stock fixture and example30 have no such column: their input is `account/stock/quantity/side/price/flow_type`. The delegation therefore did not activate on the principal workload.

The compatible baseline is `2ac1af5f`. Two rounds of ten AB/BA pairs completed 160 observations in 127.098 seconds, with full snapshot and lifecycle oracles passing. Results are near the baseline and do not meet the four-physical-core ≥3× gate: stock one CPU −0.48% (no confirmed regression), stock eight logical CPUs +1.61% (inconclusive), sparse +1.25% (inconclusive), finite control −0.05% (inconclusive). These measurements must not be presented as evidence that whole-batch DataFusion aggregation itself is fast or slow.

The next candidate replaces the nonexistent-column dependency with a single-partition `ROW_NUMBER()` ordinal used only to recover first-seen group order. The stock SUM still merges into existing keyed state. The retained acquisition evidence is in `summary.json`, `samples.csv` and `environment.json`; oversized raw observations were removed during PR cleanup.

## Ordinal candidate result

Runtime candidate `74ac0775`, sealed at source HEAD `1aab97c6` after a documentation-only plan commit, removed the missing-column gate but did not activate under the P6 harness. Delegation required `DataFusionConfig::default()`, while the stock cases explicitly select fixed parallelism and `min_rows_per_partition=1`. The candidate's approximately 59/72 ms route/apply stage timings match the baseline and confirm continued per-row keyed execution. Two rounds of ten AB/BA pairs completed 160 observations in 123.625 seconds with full snapshot and lifecycle oracles passing.

| Case | Baseline P50 | Candidate P50 | Time change | Verdict |
|------|-------------:|--------------:|------------:|---------|
| stock-stream-1cpu | 204.650 ms | 203.643 ms | −0.49% | no confirmed regression |
| stock-stream-8cpu | 138.798 ms | 137.206 ms | −1.15% | no confirmed regression |
| sparse-stream-1cpu | 21.430 ms | 21.852 ms | +1.97% | inconclusive |
| stock-batch-8cpu | 57.138 ms | 57.916 ms | +1.36% | inconclusive |

The four-physical-core ≥3× gate is not met, but these observations do not establish a performance verdict for DataFusion aggregation itself. Acquisition evidence is preserved in `ordinal-summary.json`, `ordinal-samples.csv` and `ordinal-environment.json`; the oversized acquisition dump was removed during PR cleanup. The delegation approach was withdrawn based on the activated P4 result, not retimed under a changed P6 configuration; no additional samples were collected.
