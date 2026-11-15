# Paired native rolling-dispatch evidence

Release test binaries alternate AB/BA across rounds; each round warms both recorder arms and collects 30 timed dispatches per case.
Status: completed; 10 / 10 rounds completed; 14.8 s elapsed of a 590 s budget.

| Case              | Arm     | Rounds | Baseline P50 / P95 µs | Candidate P50 / P95 µs | Paired-round speedup (95% CI) | Verdict      |
|-------------------|---------|--------|-----------------------|------------------------|-------------------------------|--------------|
| 1x1-dual          | absent  | 10     | 6.2 / 6.9             | 6.3 / 7.0              | 1.00x (0.98–1.02)             | inconclusive |
| 1x1-dual          | present | 10     | 7.8 / 8.7             | 7.8 / 8.6              | 1.00x (0.98–1.02)             | inconclusive |
| 99x1-dual         | absent  | 10     | 10.7 / 11.2           | 10.7 / 11.8            | 1.00x (0.98–1.03)             | inconclusive |
| 99x1-dual         | present | 10     | 12.2 / 12.9           | 12.2 / 13.2            | 1.00x (0.98–1.02)             | inconclusive |
| 6400x64-dual      | absent  | 10     | 452.8 / 485.0         | 452.6 / 495.7          | 1.00x (1.00–1.01)             | inconclusive |
| 6400x64-dual      | present | 10     | 452.4 / 489.8         | 452.9 / 489.1          | 1.00x (0.99–1.01)             | inconclusive |
| 64x64-dual-tick   | absent  | 10     | 100.8 / 115.2         | 100.5 / 113.0          | 0.99x (0.99–1.01)             | inconclusive |
| 64x64-dual-tick   | present | 10     | 101.7 / 112.4         | 102.1 / 112.8          | 1.00x (0.99–1.01)             | inconclusive |
| 6400x64-dual-lag1 | absent  | 10     | 9,031.8 / 10,198.4    | 9,056.5 / 9,952.9      | 1.00x (0.99–1.01)             | inconclusive |
| 6400x64-dual-lag1 | present | 10     | 9,500.8 / 11,363.8    | 9,473.3 / 10,356.3     | 1.00x (0.99–1.01)             | inconclusive |

## Absolute gate values

Gate values use candidate P50 over all retained samples. Incomplete budget-stopped reports do not assert gate outcomes.

| Gate | Case         | Arm     | Candidate value | Limit   | Unit          | Met   |
|------|--------------|---------|-----------------|---------|---------------|-------|
| G1   | 1x1-dual     | absent  | 6.264           | 1.000   | µs            | False |
| G2   | 6400x64-dual | present | 19.577          | 4.000   | ns/window-row | False |
| G3   | 6400x64-dual | absent  | 452.568         | 100.000 | µs            | False |

G2 uses recorder-present numeric_update ns divided by 6,400 × 2 window rows. P2.1 cross-boundary speedups are informational.
