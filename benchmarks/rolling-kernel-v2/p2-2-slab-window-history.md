# Paired native rolling-dispatch evidence

Release test binaries alternate AB/BA across rounds; each round warms both recorder arms and collects 30 timed dispatches per case.
Status: completed; 10 / 10 rounds completed; 6.4 s elapsed of a 590 s budget.

| Case         | Arm     | Rounds | Baseline P50 / P95 µs | Candidate P50 / P95 µs | Paired-round speedup (95% CI) | Verdict |
|--------------|---------|--------|-----------------------|------------------------|-------------------------------|---------|
| 1x1-dual     | absent  | 10     | 18.0 / 26.3           | 14.4 / 16.7            | 1.23x (1.18–1.90)             | faster  |
| 1x1-dual     | present | 10     | 21.2 / 30.3           | 16.5 / 19.4            | 1.25x (1.20–1.86)             | faster  |
| 6400x64-dual | absent  | 10     | 957.6 / 1,196.4       | 1,557.1 / 1,744.3      | 0.62x (0.57–0.65)             | slower  |
| 6400x64-dual | present | 10     | 962.6 / 1,132.4       | 1,557.8 / 1,797.1      | 0.62x (0.59–0.66)             | slower  |

## Absolute gate values

Gate values use candidate P50 over all retained samples. Incomplete budget-stopped reports do not assert gate outcomes.

| Gate | Case         | Arm     | Candidate value | Limit   | Unit          | Met   |
|------|--------------|---------|-----------------|---------|---------------|-------|
| G1   | 1x1-dual     | absent  | 14.352          | 1.000   | µs            | False |
| G2   | 6400x64-dual | present | 96.415          | 4.000   | ns/window-row | False |
| G3   | 6400x64-dual | absent  | 1557.099        | 100.000 | µs            | False |

G2 uses recorder-present numeric_update ns divided by 6,400 × 2 window rows. P2.1 cross-boundary speedups are informational.
