# P6 DataFusion delegation — rejected candidates

Candidate `5d666afa` required a `sequence` column to order grouped output by each group's first row. The maintained stock fixture and example30 have no such column: their input is `account/stock/quantity/side/price/flow_type`. The delegation therefore did not activate on the principal workload.

The compatible baseline is `2ac1af5f`. Two rounds of ten AB/BA pairs completed 160 observations in 127.098 seconds, with full snapshot and lifecycle oracles passing. Results are near the baseline and do not meet the four-physical-core ≥3× gate: stock one CPU −0.48% (no confirmed regression), stock eight logical CPUs +1.61% (inconclusive), sparse +1.25% (inconclusive), finite control −0.05% (inconclusive). These measurements must not be presented as evidence that whole-batch DataFusion aggregation itself is fast or slow.

The next candidate replaces the nonexistent-column dependency with a single-partition `ROW_NUMBER()` ordinal used only to recover first-seen group order. The stock SUM still merges into existing keyed state. Raw acquisition is in `summary.json`, `samples.csv`, `environment.json` and `driver-plan.json` (copied from the maintained driver output).

## Ordinal candidate result

Runtime candidate `74ac0775`, sealed at source HEAD `1aab97c6` after a documentation-only plan commit, activated on the real stock schema. It also failed the performance gate. Two rounds of ten AB/BA pairs completed 160 observations in 123.625 seconds with full snapshot and lifecycle oracles passing.

| Case | Baseline P50 | Candidate P50 | Time change | Verdict |
|------|-------------:|--------------:|------------:|---------|
| stock-stream-1cpu | 204.650 ms | 203.643 ms | −0.49% | no confirmed regression |
| stock-stream-8cpu | 138.798 ms | 137.206 ms | −1.15% | no confirmed regression |
| sparse-stream-1cpu | 21.430 ms | 21.852 ms | +1.97% | inconclusive |
| stock-batch-8cpu | 57.138 ms | 57.916 ms | +1.36% | inconclusive |

The four-physical-core ≥3× gate is not met. Raw acquisition is preserved in `ordinal-summary.json`, `ordinal-samples.csv`, `ordinal-environment.json` and `ordinal-driver-plan.json`. The candidate is rejected rather than retimed; no additional samples were collected.
