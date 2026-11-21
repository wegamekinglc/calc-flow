# Paired native rolling-dispatch evidence

Release test binaries alternate AB/BA across rounds; each round warms both recorder arms and collects 30 timed dispatches per case.
Status: completed; 10 / 10 rounds completed; 3.6 s elapsed of a 590 s budget.

| Case            | Arm     | Rounds | Baseline P50 / P95 µs | Candidate P50 / P95 µs | Paired-round speedup (95% CI) | Verdict      |
|-----------------|---------|--------|-----------------------|------------------------|-------------------------------|--------------|
| 1x1-dual        | absent  | 10     | 5.4 / 7.3             | 5.6 / 7.2              | 0.98x (0.95–1.00)             | slower       |
| 1x1-dual        | present | 10     | 6.6 / 8.4             | 6.8 / 8.8              | 0.98x (0.95–1.01)             | inconclusive |
| 6400x64-dual    | absent  | 10     | 494.7 / 565.2         | 484.4 / 560.6          | 1.02x (0.99–1.04)             | inconclusive |
| 6400x64-dual    | present | 10     | 501.7 / 575.7         | 487.9 / 577.0          | 1.01x (0.98–1.03)             | inconclusive |
| 64x64-dual-tick | absent  | 10     | 29.1 / 32.0           | 29.0 / 50.2            | 1.00x (0.97–1.05)             | inconclusive |
| 64x64-dual-tick | present | 10     | 30.3 / 35.7           | 30.3 / 51.0            | 1.00x (0.96–1.04)             | inconclusive |

## Absolute gate values

Gate values use candidate P50 over all retained samples. Incomplete budget-stopped reports do not assert gate outcomes.

| Gate | Case         | Arm     | Candidate value | Limit   | Unit          | Met   |
|------|--------------|---------|-----------------|---------|---------------|-------|
| G1   | 1x1-dual     | absent  | 5.561           | 1.000   | µs            | False |
| G2   | 6400x64-dual | present | 23.708          | 4.000   | ns/window-row | False |
| G3   | 6400x64-dual | absent  | 484.409         | 100.000 | µs            | False |

G2 uses recorder-present numeric_update ns divided by 6,400 × 2 window rows. P2.1 cross-boundary speedups are informational.
