# Paired native rolling-dispatch evidence

Release test binaries alternate AB/BA across rounds; each round warms both recorder arms and collects 30 timed dispatches per case.
Status: completed; 10 / 10 rounds completed; 2.6 s elapsed of a 590 s budget.

| Case         | Arm     | Rounds | Baseline P50 / P95 µs | Candidate P50 / P95 µs | Paired-round speedup (95% CI) | Verdict |
|--------------|---------|--------|-----------------------|------------------------|-------------------------------|---------|
| 1x1-dual     | absent  | 10     | 5.9 / 7.0             | 5.3 / 11.0             | 1.12x (1.07–1.15)             | faster  |
| 1x1-dual     | present | 10     | 7.5 / 9.5             | 6.5 / 12.5             | 1.15x (1.11–1.19)             | faster  |
| 6400x64-dual | absent  | 10     | 434.6 / 489.7         | 1,088.8 / 1,152.0      | 0.40x (0.39–0.40)             | slower  |
| 6400x64-dual | present | 10     | 440.5 / 515.1         | 1,089.0 / 1,138.7      | 0.41x (0.40–0.41)             | slower  |

## Absolute gate values

Gate values use candidate P50 over all retained samples. Incomplete budget-stopped reports do not assert gate outcomes.

| Gate | Case         | Arm     | Candidate value | Limit   | Unit          | Met   |
|------|--------------|---------|-----------------|---------|---------------|-------|
| G1   | 1x1-dual     | absent  | 5.300           | 1.000   | µs            | False |
| G2   | 6400x64-dual | present | 74.641          | 4.000   | ns/window-row | False |
| G3   | 6400x64-dual | absent  | 1088.822        | 100.000 | µs            | False |

G2 uses recorder-present numeric_update ns divided by 6,400 × 2 window rows. P2.1 cross-boundary speedups are informational.
