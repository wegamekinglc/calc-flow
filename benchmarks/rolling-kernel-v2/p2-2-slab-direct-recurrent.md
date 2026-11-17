# Paired native rolling-dispatch evidence

Release test binaries alternate AB/BA across rounds; each round warms both recorder arms and collects 30 timed dispatches per case.
Status: completed; 10 / 10 rounds completed; 5.1 s elapsed of a 590 s budget.

| Case         | Arm     | Rounds | Baseline P50 / P95 µs | Candidate P50 / P95 µs | Paired-round speedup (95% CI) | Verdict      |
|--------------|---------|--------|-----------------------|------------------------|-------------------------------|--------------|
| 1x1-dual     | absent  | 10     | 21.1 / 27.3           | 18.1 / 20.8            | 1.18x (0.51–1.80)             | inconclusive |
| 1x1-dual     | present | 10     | 24.7 / 39.4           | 20.8 / 27.7            | 1.22x (0.55–1.81)             | inconclusive |
| 6400x64-dual | absent  | 10     | 1,091.9 / 1,420.3     | 1,031.5 / 1,165.5      | 1.09x (1.02–1.21)             | faster       |
| 6400x64-dual | present | 10     | 1,102.7 / 1,403.0     | 1,034.0 / 1,175.3      | 1.11x (1.03–1.22)             | faster       |

## Absolute gate values

Gate values use candidate P50 over all retained samples. Incomplete budget-stopped reports do not assert gate outcomes.

| Gate | Case         | Arm     | Candidate value | Limit   | Unit          | Met   |
|------|--------------|---------|-----------------|---------|---------------|-------|
| G1   | 1x1-dual     | absent  | 18.133          | 1.000   | µs            | False |
| G2   | 6400x64-dual | present | 54.080          | 4.000   | ns/window-row | False |
| G3   | 6400x64-dual | absent  | 1031.548        | 100.000 | µs            | False |

G2 uses recorder-present numeric_update ns divided by 6,400 × 2 window rows. P2.1 cross-boundary speedups are informational.
