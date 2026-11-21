# Paired native rolling-dispatch evidence

Release test binaries alternate AB/BA across rounds; each round warms both recorder arms and collects 30 timed dispatches per case.
Status: completed; 10 / 10 rounds completed; 1.7 s elapsed of a 590 s budget.

| Case            | Arm     | Rounds | Baseline P50 / P95 µs | Candidate P50 / P95 µs | Paired-round speedup (95% CI) | Verdict      |
|-----------------|---------|--------|-----------------------|------------------------|-------------------------------|--------------|
| 1x1-dual        | absent  | 10     | 5.4 / 6.7             | 5.5 / 7.2              | 0.97x (0.95–1.01)             | inconclusive |
| 1x1-dual        | present | 10     | 6.5 / 8.4             | 6.6 / 8.6              | 0.98x (0.95–1.00)             | slower       |
| 6400x64-dual    | absent  | 10     | 493.7 / 589.5         | 497.9 / 589.0          | 1.00x (0.97–1.02)             | inconclusive |
| 6400x64-dual    | present | 10     | 501.1 / 610.7         | 504.8 / 587.2          | 1.00x (0.96–1.01)             | inconclusive |
| 64x64-dual-tick | absent  | 10     | 29.4 / 37.6           | 29.0 / 32.9            | 1.01x (1.00–1.03)             | inconclusive |
| 64x64-dual-tick | present | 10     | 30.5 / 40.5           | 30.1 / 34.1            | 1.02x (1.00–1.03)             | inconclusive |

## Absolute gate values

Gate values use candidate P50 over all retained samples. Incomplete budget-stopped reports do not assert gate outcomes.

| Gate | Case         | Arm     | Candidate value | Limit   | Unit          | Met   |
|------|--------------|---------|-----------------|---------|---------------|-------|
| G1   | 1x1-dual     | absent  | 5.457           | 1.000   | µs            | False |
| G2   | 6400x64-dual | present | 28.428          | 4.000   | ns/window-row | False |
| G3   | 6400x64-dual | absent  | 497.927         | 100.000 | µs            | False |

G2 uses recorder-present numeric_update ns divided by 6,400 × 2 window rows. P2.1 cross-boundary speedups are informational.
