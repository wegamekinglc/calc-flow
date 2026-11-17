# Paired native rolling-dispatch evidence

Release test binaries alternate AB/BA across rounds; each round warms both recorder arms and collects 30 timed dispatches per case.
Status: completed; 10 / 10 rounds completed; 5.5 s elapsed of a 590 s budget.

| Case         | Arm     | Rounds | Baseline P50 / P95 µs | Candidate P50 / P95 µs | Paired-round speedup (95% CI) | Verdict |
|--------------|---------|--------|-----------------------|------------------------|-------------------------------|---------|
| 1x1-dual     | absent  | 10     | 21.4 / 29.6           | 17.9 / 25.2            | 1.18x (1.13–1.46)             | faster  |
| 1x1-dual     | present | 10     | 25.1 / 31.5           | 20.6 / 28.4            | 1.20x (1.15–1.45)             | faster  |
| 6400x64-dual | absent  | 10     | 1,176.7 / 1,421.1     | 1,402.2 / 1,728.6      | 0.84x (0.76–0.89)             | slower  |
| 6400x64-dual | present | 10     | 1,185.5 / 1,386.5     | 1,407.2 / 1,743.7      | 0.84x (0.75–0.90)             | slower  |

## Absolute gate values

Gate values use candidate P50 over all retained samples. Incomplete budget-stopped reports do not assert gate outcomes.

| Gate | Case         | Arm     | Candidate value | Limit   | Unit          | Met   |
|------|--------------|---------|-----------------|---------|---------------|-------|
| G1   | 1x1-dual     | absent  | 17.950          | 1.000   | µs            | False |
| G2   | 6400x64-dual | present | 83.310          | 4.000   | ns/window-row | False |
| G3   | 6400x64-dual | absent  | 1402.190        | 100.000 | µs            | False |

G2 uses recorder-present numeric_update ns divided by 6,400 × 2 window rows. P2.1 cross-boundary speedups are informational.
