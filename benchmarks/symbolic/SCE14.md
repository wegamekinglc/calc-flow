# SCE-14 Cross-Domain Optimization Performance Gate

This evidence closes the performance acceptance criterion in
[GitHub #181](https://github.com/wegamekinglc/calc-flow/issues/181). Functional
equivalence, immutable compilation caching, deterministic explain output,
state sharing, materialization boundaries, and unsafe-finality rejection are
covered separately by the symbolic optimizer tests.

## Frozen comparison

The optimization benchmark commit is
`9680d60ba81e45f6cc5f6eafbd6eef586c26f210`. Both raw reports record that
exact clean commit and the same stable machine fingerprint. The standard-scale
workload contains 99,968 deterministic quote rows across 64 entities and eight
industries. It produces two 99,968-row outputs.

Each report builds equivalent plans from the same immutable input:

- a reference made from two independently compiled programs, containing four
  state stages in total; and
- one combined symbolic program whose optimizer emits two shared state stages.

Before timing, the benchmark requires exact Arrow equality for both outputs
and confirms the four-to-two state-stage reduction. It then measures 30 pairs
in the same process, alternating reference-first and optimized-first order.
Two independent reports provide 60 pairs total, evenly split between orders.

The verifier bootstraps paired log ratios with the fixed seed `20260829` and
20,000 resamples. It passes only when the 95% upper confidence bound is at or
below the five-percent regression limit. An interval crossing five percent is
`inconclusive`, not a pass.

## Optimization result

| Scenario                       | Pairs | Geometric regression | 95% bootstrap interval     | Gate    | Decision |
| ------------------------------ | ----: | -------------------: | -------------------------: | ------: | -------- |
| `sce14_cross_domain_sharing`   |    60 |             -45.317% |       -45.517% to -45.112% | +5.000% | pass     |

The optimized combined program is faster in this relative comparison. This is
an execution-overhead claim for the frozen workload, not an absolute
throughput claim.

## SCE-01 regression rerun

The original SCE-01 standard-scale scenarios were rerun against base commit
`bde42be38549e5a990ad1125adc2d108ff19f96e` in balanced report order:
base then feature, followed by feature then base. Each pair had to remain at or
below the same +5% regression threshold; the geometric value summarizes the
two order-balanced ratios.

| Scenario                                  | Base-first pair | Feature-first pair | Geometric regression | Gate    | Decision |
| ----------------------------------------- | --------------: | -----------------: | -------------------: | ------: | -------- |
| `symbolic_projection_20_columns`          |         -2.647% |            -2.541% |              -2.594% | +5.000% | pass     |
| `symbolic_rolling_20_60_row_features`     |         -0.490% |            -0.314% |              -0.402% | +5.000% | pass     |
| `symbolic_cross_section_rank_zscore`      |         +2.384% |            -0.279% |              +1.044% | +5.000% | pass     |
| `symbolic_table_matmul_numpy`             |         -5.889% |            +0.375% |              -2.807% | +5.000% | pass     |
| `symbolic_table_matmul_jax`               |         -2.487% |            -0.166% |              -1.333% | +5.000% | pass     |
| `symbolic_stream_window_checkpoint`       |         +2.372% |            -2.172% |              +0.074% | +5.000% | pass     |

Exploratory reports were discarded before freezing this evidence when the host
changed performance state and when a recapture used a PEP 517 release install
instead of the documented default-debug `maturin develop` environment. The
accepted reports above were captured only after both base and feature used the
same stable build mode; every accepted pair records a clean exact commit.

## Artifacts and reproduction

- `sce14-9680d60-paired1.json` and `sce14-9680d60-paired2.json` contain the 60
  same-process alternating optimization pairs.
- `sce14-9680d60-summary.json` contains the validated bootstrap decision,
  report hashes, provenance, machine identity, and workload metadata.
- The four `sce14-sce01-*.json` reports contain the two order-balanced SCE-01
  comparisons; `sce14-sce01-summary.json` records their hashes and decisions.

Reproduce one optimization report from the exact benchmark commit:

```bash
CALC_FLOW_BENCHMARK_SCALE=standard \
  JAX_PLATFORMS=cpu \
  uv run --extra benchmark pytest \
  benchmarks/test_symbolic_baseline.py::test_sce14_cross_domain_sharing_pair \
  -q --benchmark-only --benchmark-json=<output>.json
```

Verify two or more optimization reports:

```bash
uv run python scripts/verify_symbolic_milestone_perf.py \
  --report <first>.json \
  --report <second>.json \
  --scenario sce14_cross_domain_sharing
```
