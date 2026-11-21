# Paired native rolling-dispatch evidence

Release test binaries alternate AB/BA across rounds; each round warms both recorder arms and collects 30 timed dispatches per case.
Status: completed; 10 / 10 rounds completed; 1.5 s elapsed of a 590 s budget.

| Case         | Arm     | Rounds | Baseline P50 / P95 µs | Candidate P50 / P95 µs | Paired-round speedup (95% CI) | Verdict |
|--------------|---------|--------|-----------------------|------------------------|-------------------------------|---------|
| 1x1-dual     | absent  | 10     | 6.1 / 7.5             | 5.4 / 6.5              | 1.13x (1.10–1.17)             | faster  |
| 1x1-dual     | present | 10     | 7.7 / 9.0             | 6.7 / 8.2              | 1.14x (1.11–1.18)             | faster  |
| 6400x64-dual | absent  | 10     | 447.7 / 508.5         | 399.0 / 457.0          | 1.12x (1.10–1.15)             | faster  |
| 6400x64-dual | present | 10     | 449.6 / 490.6         | 400.5 / 502.4          | 1.13x (1.09–1.16)             | faster  |

## Absolute gate values

Gate values use candidate P50 over all retained samples. Incomplete budget-stopped reports do not assert gate outcomes.

| Gate | Case         | Arm     | Candidate value | Limit   | Unit          | Met   |
|------|--------------|---------|-----------------|---------|---------------|-------|
| G1   | 1x1-dual     | absent  | 5.394           | 1.000   | µs            | False |
| G2   | 6400x64-dual | present | 20.882          | 4.000   | ns/window-row | False |
| G3   | 6400x64-dual | absent  | 399.043         | 100.000 | µs            | False |

G2 uses recorder-present numeric_update ns divided by 6,400 × 2 window rows. P2.1 cross-boundary speedups are informational.
