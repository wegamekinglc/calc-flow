# Paired native rolling-dispatch evidence

Release test binaries alternate AB/BA across rounds; each round warms both recorder arms and collects 30 timed dispatches per case.
Status: completed; 10 / 10 rounds completed; 1.4 s elapsed of a 590 s budget.

| Case         | Arm     | Rounds | Baseline P50 / P95 µs | Candidate P50 / P95 µs | Paired-round speedup (95% CI) | Verdict |
|--------------|---------|--------|-----------------------|------------------------|-------------------------------|---------|
| 1x1-dual     | absent  | 10     | 5.9 / 8.0             | 5.2 / 6.7              | 1.14x (1.12–1.15)             | faster  |
| 1x1-dual     | present | 10     | 7.4 / 8.9             | 6.4 / 9.0              | 1.17x (1.15–1.18)             | faster  |
| 6400x64-dual | absent  | 10     | 431.0 / 467.2         | 371.8 / 404.5          | 1.15x (1.14–1.17)             | faster  |
| 6400x64-dual | present | 10     | 431.7 / 466.4         | 372.6 / 406.6          | 1.15x (1.13–1.18)             | faster  |

## Absolute gate values

Gate values use candidate P50 over all retained samples. Incomplete budget-stopped reports do not assert gate outcomes.

| Gate | Case         | Arm     | Candidate value | Limit   | Unit          | Met   |
|------|--------------|---------|-----------------|---------|---------------|-------|
| G1   | 1x1-dual     | absent  | 5.173           | 1.000   | µs            | False |
| G2   | 6400x64-dual | present | 19.291          | 4.000   | ns/window-row | False |
| G3   | 6400x64-dual | absent  | 371.773         | 100.000 | µs            | False |

G2 uses recorder-present numeric_update ns divided by 6,400 × 2 window rows. P2.1 cross-boundary speedups are informational.
