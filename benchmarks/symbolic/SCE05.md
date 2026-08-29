# SCE-05 Row-Local Performance Gate

This evidence closes the performance-only gap in
[GitHub #172](https://github.com/wegamekinglc/calc-flow/issues/172). Functional
lowering, deterministic fusion/CSE, batch/stream segmentation equivalence,
Rust compiler ownership, and zero symbolic Python calls during execution were
already delivered by PR #194. This record does not replace those correctness
tests.

## Frozen comparison

The benchmark commit is
`b5b0af00d41c5b56e33b3ae25cdeb2c9a9f0a27c`. Both raw reports record that
exact clean commit and the same stable machine fingerprint. The standard-scale
workload contains 99,968 deterministic rows and 20 derived outputs.

Each report builds two equivalent plans from the same input:

- a hand-built single SQL projection; and
- the public `Program.compile_batch()` symbolic path.

The test requires byte-equivalent Arrow output and exactly one DataFusion query
for the symbolic plan before timing. It then measures 60 pairs in the same
process, alternating hand-built-first and symbolic-first order. Two independent
reports provide 120 pairs total, evenly split between the two orders.

The verifier bootstraps the paired log ratios with the fixed seed `20260829`
and 20,000 resamples. It passes only when the 95% upper confidence bound is at
or below the five-percent regression limit. An interval crossing five percent
is `inconclusive`, not a pass.

## Result

| Scenario                       | Pairs | Geometric regression | 95% bootstrap interval | Gate    | Decision |
| ------------------------------ | ----: | -------------------: | ---------------------: | ------: | -------- |
| `sce05_row_local_20_columns`   |   120 |              +0.776% |       -1.116% to +3.640% | +5.000% | pass     |

The upper bound is 1.360 percentage points below the gate. This is a relative
comparison only; it makes no absolute throughput claim.

## Artifacts and reproduction

- `sce05-b5b0af0-paired1.json` — 60 alternating pairs.
- `sce05-b5b0af0-paired2.json` — independent 60-pair repeat.
- `sce05-b5b0af0-summary.json` — validated comparison, report SHA-256 values,
  provenance, workload metadata, and decision.

Reproduce one raw report from the exact benchmark commit:

```bash
CALC_FLOW_BENCHMARK_SCALE=standard \
  JAX_PLATFORMS=cpu \
  uv run --extra benchmark pytest benchmarks/test_symbolic_baseline.py -q \
  -k sce05_row_local --benchmark-only --benchmark-json=<output>.json
```

Verify two or more reports:

```bash
uv run python scripts/verify_symbolic_milestone_perf.py \
  --report <first>.json \
  --report <second>.json \
  --scenario sce05_row_local_20_columns
```
