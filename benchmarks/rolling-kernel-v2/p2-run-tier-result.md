# Paired native rolling-dispatch evidence

Release test binaries alternate AB/BA across rounds; each round warms both recorder arms and collects 30 timed dispatches per case.
Status: completed; 10 / 10 rounds completed; 5.4 s elapsed of a 590 s budget.

| Case            | Arm     | Rounds | Baseline P50 / P95 µs | Candidate P50 / P95 µs | Paired-round speedup (95% CI) | Verdict      |
|-----------------|---------|--------|-----------------------|------------------------|-------------------------------|--------------|
| 1x1-dual        | absent  | 10     | 5.5 / 7.9             | 5.3 / 6.2              | 1.02x (0.97–1.06)             | inconclusive |
| 1x1-dual        | present | 10     | 6.7 / 9.6             | 6.5 / 8.3              | 1.01x (0.98–1.07)             | inconclusive |
| 6400x64-dual    | absent  | 10     | 487.1 / 574.2         | 488.1 / 614.6          | 0.99x (0.96–1.01)             | inconclusive |
| 6400x64-dual    | present | 10     | 499.6 / 579.8         | 482.0 / 625.7          | 1.05x (1.02–1.06)             | faster       |
| 64x64-dual-tick | absent  | 10     | 29.2 / 38.8           | 31.1 / 41.3            | 0.94x (0.91–0.96)             | slower       |
| 64x64-dual-tick | present | 10     | 30.2 / 39.0           | 32.4 / 42.0            | 0.94x (0.91–0.96)             | slower       |

## Absolute gate values

Gate values use candidate P50 over all retained samples. Incomplete budget-stopped reports do not assert gate outcomes.

| Gate | Case         | Arm     | Candidate value | Limit   | Unit          | Met   |
|------|--------------|---------|-----------------|---------|---------------|-------|
| G1   | 1x1-dual     | absent  | 5.342           | 1.000   | µs            | False |
| G2   | 6400x64-dual | present | 23.929          | 4.000   | ns/window-row | False |
| G3   | 6400x64-dual | absent  | 488.114         | 100.000 | µs            | False |

G2 uses recorder-present numeric_update ns divided by 6,400 × 2 window rows. P2.1 cross-boundary speedups are informational.
