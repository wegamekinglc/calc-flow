# Paired native rolling-dispatch evidence

Release test binaries alternate AB/BA across rounds; each round warms both recorder arms and collects 30 timed dispatches per case.
Status: completed; 10 / 10 rounds completed; 2.9 s elapsed of a 590 s budget.

| Case            | Arm     | Rounds | Baseline P50 / P95 µs | Candidate P50 / P95 µs | Paired-round speedup (95% CI) | Verdict      |
|-----------------|---------|--------|-----------------------|------------------------|-------------------------------|--------------|
| 1x1-dual        | absent  | 10     | 5.6 / 12.0            | 5.6 / 11.5             | 0.98x (0.96–1.04)             | inconclusive |
| 1x1-dual        | present | 10     | 6.8 / 13.9            | 6.7 / 13.8             | 0.99x (0.97–1.03)             | inconclusive |
| 6400x64-dual    | absent  | 10     | 507.2 / 604.0         | 522.6 / 640.4          | 0.97x (0.96–0.99)             | slower       |
| 6400x64-dual    | present | 10     | 503.1 / 607.3         | 511.1 / 604.8          | 0.98x (0.96–1.00)             | slower       |
| 64x64-dual-tick | absent  | 10     | 30.3 / 41.2           | 30.1 / 43.5            | 1.02x (0.96–1.08)             | inconclusive |
| 64x64-dual-tick | present | 10     | 31.4 / 42.6           | 31.4 / 45.3            | 1.01x (0.98–1.05)             | inconclusive |

## Absolute gate values

Gate values use candidate P50 over all retained samples. Incomplete budget-stopped reports do not assert gate outcomes.

| Gate | Case         | Arm     | Candidate value | Limit   | Unit          | Met   |
|------|--------------|---------|-----------------|---------|---------------|-------|
| G1   | 1x1-dual     | absent  | 5.554           | 1.000   | µs            | False |
| G2   | 6400x64-dual | present | 28.781          | 4.000   | ns/window-row | False |
| G3   | 6400x64-dual | absent  | 522.565         | 100.000 | µs            | False |

G2 uses recorder-present numeric_update ns divided by 6,400 × 2 window rows. P2.1 cross-boundary speedups are informational.
