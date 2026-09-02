# SCE-16 Exponential Indicators Performance Gate

This evidence closes the deferred performance acceptance criterion of the
SCE-16 milestone delivered by
[PR #221](https://github.com/wegamekinglc/calc-flow/pull/221) (issue #220).
Functional equivalence, entity isolation, shared state, segmentation
invariance, batch/stream parity, and mid-checkpoint recovery without
retained-history reconstruction are covered separately by the symbolic
finance reference and rolling acceptance tests.

## Frozen comparison

The benchmark commit is
`195157889a27241bc5c498a858cb6dace9b5fef9`. Both raw reports record that
exact clean commit and the same stable machine fingerprint. The standard-scale
workload contains 99,968 deterministic quote rows across 64 entities. It
produces one 99,968-row output per plan.

Each report builds equivalent plans from the same immutable input:

- a hand-built project-v3 rolling declaration containing EWMA span 12 and
  span 26 outputs over the same entity, event-time, and sequence ordering;
  and
- one public symbolic program whose `ts.ewma` and `ts.macd` declarations
  lower to the same native project-v3 rolling declaration.

Before timing, the benchmark requires equal plan fingerprints and exact Arrow
output equality, then measures 30 hand-built/symbolic pairs in the same
process, alternating hand-built-first and symbolic-first order. Two
independent reports provide 60 pairs total, evenly split between orders.

The verifier bootstraps paired log ratios with the fixed seed `20260829` and
20,000 resamples. It passes only when the 95% upper confidence bound is at or
below the five-percent regression limit. An interval crossing five percent is
`inconclusive`, not a pass.

## Exponential indicator result

| Scenario                       | Pairs | Geometric regression | 95% bootstrap interval | Gate    | Decision |
| ------------------------------ | ----: | -------------------: | ---------------------: | ------: | -------- |
| `sce16_exponential_indicators` |    60 |               +0.024% |      -0.733% to +0.712% | +5.000% | pass     |

The public symbolic EWMA/MACD plan is performance equivalent to the hand-built
native project-v3 rolling declaration for this workload: the paired interval
is centered on zero and stays far inside the gate. This is a
compilation-overhead claim for the frozen workload, not an absolute throughput
claim. The captured environment resolved Python 3.13.13, NumPy 2.5.2, and
PyArrow 24.0.0 under the frozen benchmark dependency set.

## Artifacts and reproduction

- `sce16-1951578-paired1.json` and `sce16-1951578-paired2.json` contain the 60
  same-process alternating hand-built/symbolic pairs.
- `sce16-1951578-summary.json` contains the validated bootstrap decision,
  report hashes, provenance, machine identity, and workload metadata.

Reproduce one report from the exact benchmark commit on a Linux benchmark
host:

```bash
CALC_FLOW_BENCHMARK_SCALE=standard \
  JAX_PLATFORMS=cpu \
  uv run --extra benchmark pytest \
  benchmarks/test_symbolic_baseline.py::test_sce16_exponential_indicator_pair \
  -q --benchmark-only --benchmark-json=<output>.json
```

Verify two or more reports:

```bash
uv run python scripts/verify_symbolic_milestone_perf.py \
  --report <first>.json \
  --report <second>.json \
  --scenario sce16_exponential_indicators
```
