# Public ASOF watermark phases

This diagnostic compares the original 11.14 release (7012b5b4) with main (72072c1d) on the managed four-thread public ASOF workload. Both sides received one million rows covering the same 64 symbols, matched with zero tolerance. The two diagnostic wheels were built independently after adding the task-local phase clocks to their respective source trees. The exact patches, reconstructed source trees, build and interpreter identities are stored in the archive.

## All finalization callbacks

Times and differences are milliseconds. Each round's last column pair reports the median of ten main-minus-release differences. P50 and P95 pool the twenty formal samples per arm.

| Elapsed            | 11.14 P50 | Main P50 | 11.14 P95 | Main P95 | Round 1 Δ | Round 2 Δ |
|--------------------|-----------|----------|-----------|----------|-----------|-----------|
| Finalization total | 17.484    | 65.649   | 21.957    | 73.010   | +46.981   | +47.668   |
| Matching           | 8.819     | 18.448   | 12.419    | 22.987   | +10.322   | +9.494    |
| Arrow output       | 0.029     | 29.481   | 0.035     | 32.528   | +29.449   | +29.443   |
| Left state commit  | 0.026     | 8.541    | 0.030     | 9.700    | +8.422    | +8.546    |
| Right eviction     | 9.043     | 9.168    | 10.231    | 10.386   | -0.224    | +0.549    |
| Emit and await     | 0.009     | 0.017    | 0.012     | 0.020    | +0.008    | +0.008    |
| Other finalization | 0.013     | 0.028    | 0.026     | 0.039    | +0.018    | +0.012    |

## Callbacks that emitted rows

Times and differences are milliseconds. Each round's last column pair reports the median of ten main-minus-release differences. P50 and P95 pool the twenty formal samples per arm.

| Elapsed            | 11.14 P50 | Main P50 | 11.14 P95 | Main P95 | Round 1 Δ | Round 2 Δ |
|--------------------|-----------|----------|-----------|----------|-----------|-----------|
| Finalization total | 17.471    | 63.225   | 21.945    | 71.377   | +43.714   | +45.182   |
| Matching           | 8.817     | 18.447   | 12.418    | 22.986   | +10.323   | +9.496    |
| Arrow output       | 0.029     | 29.481   | 0.035     | 32.528   | +29.449   | +29.443   |
| Left state commit  | 0.026     | 8.541    | 0.030     | 9.700    | +8.422    | +8.546    |
| Right eviction     | 9.040     | 5.704    | 10.228    | 9.204    | -3.528    | -1.766    |
| Emit and await     | 0.009     | 0.017    | 0.012     | 0.020    | +0.008    | +0.008    |
| Other finalization | 0.003     | 0.009    | 0.004     | 0.021    | +0.007    | +0.006    |

The instrumented public ready-to-Arrow P50 was 61.323 ms for release and 201.792 ms for main. All 44 executions validated the full one-million-row output with zero maximum absolute error. Each request's finalize trace matched and emitted exactly one million rows.

The exclusive native clocks follow the ASOF operator task across awaits. They measure matching, Arrow materialization, left-state commit, right-side eviction, emit/await and other finalization. The six times sum exactly to each callback's wall clock. Before/after state gauges, emitted/matched/evicted deltas and actual successful gather-ticket counts are preserved by callback and request. All callbacks include pre-watermark and EOF work, some of which can lie outside the ordinary public ready-to-Arrow interval. Input admission is outside these native stage clocks. Parallel release fragments can perform embedded planning work inside worker waits. Trace serialization occurs after the native callback clock. Its overhead can affect the enclosing instrumented run.

The acquisition took 21.78 seconds including interpreter startup and exit, within the 120-second budget. The protocol and original evidence preserve worker-local trace offsets, worker responses, measured intervals, managed interpreter and wheel identities, exact source patches, resource telemetry and stop information. [Summary and per-request data](result.json) · [Original evidence](raw-evidence.tar.gz) · [Receipt](receipt.json)
