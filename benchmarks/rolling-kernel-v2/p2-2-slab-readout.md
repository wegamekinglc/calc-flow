# Paired native rolling-dispatch evidence

Release test binaries alternate AB/BA across rounds; each round warms both recorder arms and collects 30 timed dispatches per case.
Status: completed; 10 / 10 rounds completed; 4.0 s elapsed of a 590 s budget.

| Case         | Arm     | Rounds | Baseline P50 / P95 µs | Candidate P50 / P95 µs | Paired-round speedup (95% CI) | Verdict |
|--------------|---------|--------|-----------------------|------------------------|-------------------------------|---------|
| 1x1-dual     | absent  | 10     | 6.0 / 13.0            | 5.2 / 7.1              | 1.15x (1.12–1.20)             | faster  |
| 1x1-dual     | present | 10     | 7.6 / 14.7            | 6.4 / 8.9              | 1.20x (1.17–1.24)             | faster  |
| 6400x64-dual | absent  | 10     | 446.0 / 716.4         | 1,467.9 / 1,602.6      | 0.31x (0.30–0.32)             | slower  |
| 6400x64-dual | present | 10     | 449.5 / 731.0         | 1,454.3 / 1,604.5      | 0.31x (0.30–0.32)             | slower  |

## Absolute gate values

Gate values use candidate P50 over all retained samples. Incomplete budget-stopped reports do not assert gate outcomes.

| Gate | Case         | Arm     | Candidate value | Limit   | Unit          | Met   |
|------|--------------|---------|-----------------|---------|---------------|-------|
| G1   | 1x1-dual     | absent  | 5.217           | 1.000   | µs            | False |
| G2   | 6400x64-dual | present | 102.990         | 4.000   | ns/window-row | False |
| G3   | 6400x64-dual | absent  | 1467.900        | 100.000 | µs            | False |

G2 uses recorder-present numeric_update ns divided by 6,400 × 2 window rows. P2.1 cross-boundary speedups are informational.
