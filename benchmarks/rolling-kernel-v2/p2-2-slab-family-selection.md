# Paired native rolling-dispatch evidence

Release test binaries alternate AB/BA across rounds; each round warms both recorder arms and collects 30 timed dispatches per case.
Status: completed; 10 / 10 rounds completed; 6.6 s elapsed of a 590 s budget.

| Case         | Arm     | Rounds | Baseline P50 / P95 µs | Candidate P50 / P95 µs | Paired-round speedup (95% CI) | Verdict      |
|--------------|---------|--------|-----------------------|------------------------|-------------------------------|--------------|
| 1x1-dual     | absent  | 10     | 19.2 / 34.9           | 16.2 / 22.2            | 1.23x (0.86–1.88)             | inconclusive |
| 1x1-dual     | present | 10     | 22.4 / 37.9           | 18.5 / 30.2            | 1.22x (0.70–1.81)             | inconclusive |
| 6400x64-dual | absent  | 10     | 1,064.0 / 1,253.6     | 1,558.5 / 1,768.5      | 0.68x (0.65–0.71)             | slower       |
| 6400x64-dual | present | 10     | 1,072.8 / 1,285.2     | 1,548.3 / 1,775.4      | 0.68x (0.66–0.73)             | slower       |

## Absolute gate values

Gate values use candidate P50 over all retained samples. Incomplete budget-stopped reports do not assert gate outcomes.

| Gate | Case         | Arm     | Candidate value | Limit   | Unit          | Met   |
|------|--------------|---------|-----------------|---------|---------------|-------|
| G1   | 1x1-dual     | absent  | 16.216          | 1.000   | µs            | False |
| G2   | 6400x64-dual | present | 95.908          | 4.000   | ns/window-row | False |
| G3   | 6400x64-dual | absent  | 1558.463        | 100.000 | µs            | False |

G2 uses recorder-present numeric_update ns divided by 6,400 × 2 window rows. P2.1 cross-boundary speedups are informational.
