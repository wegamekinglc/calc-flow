# P6 DataFusion delegation — sequence-gate result

Candidate `5d666afa` required a `sequence` column to order grouped output by each group's first row. The maintained stock fixture and example30 have no such column: their input is `account/stock/quantity/side/price/flow_type`. The delegation therefore did not activate on the principal workload.

The compatible baseline is `2ac1af5f`. Two rounds of ten AB/BA pairs completed 160 observations in 127.098 seconds, with full snapshot and lifecycle oracles passing. Results are near the baseline and do not meet the four-physical-core ≥3× gate: stock one CPU −0.48% (no confirmed regression), stock eight logical CPUs +1.61% (inconclusive), sparse +1.25% (inconclusive), finite control −0.05% (inconclusive). These measurements must not be presented as evidence that whole-batch DataFusion aggregation itself is fast or slow.

The next candidate replaces the nonexistent-column dependency with a single-partition `ROW_NUMBER()` ordinal used only to recover first-seen group order. The stock SUM still merges into existing keyed state. Raw acquisition is in `summary.json`, `samples.csv`, `environment.json` and `driver-plan.json` (copied from the maintained driver output).
