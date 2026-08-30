# SCE-08 Temporal Rolling Performance Gate

This evidence closes the performance-only gap in
[GitHub #175](https://github.com/wegamekinglc/calc-flow/issues/175). Duration
windows, `rolling_min`, `rolling_max`, `rolling_cov`, `rolling_corr`, recovery,
bounded state, and batch-segmentation invariance were already delivered by
PR #206. This record does not replace those correctness tests.

## Frozen comparison

The benchmark commit is
`41471acf05d6c54308bd6bed28f2eaf3b66c3daf`. Both raw reports record that
exact clean commit and the same stable machine fingerprint. The standard-scale
workload contains 99,968 deterministic quote rows across 64 entities, a
60-second duration window, and four temporal outputs. The benchmark confirms
that retained state never exceeds 60 rows per entity.

Each report builds two equivalent plans from the same input:

- a static project-v3 plan containing the native duration rolling node; and
- the public `Program.compile_batch()` symbolic path.

Before timing, the test requires identical compiled fingerprints, exact
minimum/maximum output, tolerance-checked covariance/correlation output, and
exactly one DataFusion identity query. Compilation is outside the timed
boundary. Each report then measures 30 pairs in the same process, alternating
native-first and symbolic-first order. Two independent reports provide 60
pairs total, evenly split between the two orders.

The verifier bootstraps the paired log ratios with the fixed seed `20260829`
and 20,000 resamples. It passes only when the 95% upper confidence bound is at
or below the five-percent regression limit. An interval crossing five percent
is `inconclusive`, not a pass.

## Result

| Scenario                 | Pairs | Geometric regression | 95% bootstrap interval   | Gate    | Decision |
| ------------------------ | ----: | -------------------: | -----------------------: | ------: | -------- |
| `sce08_temporal_catalog` |    60 |              +1.016% |       +0.257% to +1.836% | +5.000% | pass     |

The upper bound is 3.164 percentage points below the gate. This is a relative
execution-overhead comparison only; it makes no absolute throughput claim.

## Artifacts and reproduction

- `sce08-41471ac-paired1.json` — 30 alternating pairs.
- `sce08-41471ac-paired2.json` — independent 30-pair repeat.
- `sce08-41471ac-summary.json` — validated comparison, report SHA-256 values,
  provenance, workload metadata, and decision.

Reproduce one raw report from the exact benchmark commit:

```bash
CALC_FLOW_BENCHMARK_SCALE=standard \
  JAX_PLATFORMS=cpu \
  uv run --extra benchmark pytest \
  benchmarks/test_symbolic_baseline.py::test_sce08_temporal_milestone_pair -q \
  --benchmark-only --benchmark-json=<output>.json
```

Verify two or more reports:

```bash
uv run python scripts/verify_symbolic_milestone_perf.py \
  --report <first>.json \
  --report <second>.json \
  --scenario sce08_temporal_catalog
```
