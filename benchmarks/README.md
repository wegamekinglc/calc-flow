# Benchmarks

Calc Flow benchmarks use deterministic Arrow and Array API inputs and export
complete `pytest-benchmark` JSON reports. They are informational until at least
20 comparable main-branch samples have been collected on stable runners.

Install and run the overhead suite:

```bash
uv sync --extra benchmark
CALC_FLOW_BENCHMARK_SCALE=overhead \
  uv run pytest benchmarks --benchmark-only \
  --benchmark-json=benchmark-results/overhead.json
```

Available scales:

| Scale      | Table rows | Array elements | Matrix dimension |
| ---------- | ---------: | -------------: | ---------------: |
| `overhead` |        100 |          1,000 |               16 |
| `small`    |     10,000 |         10,000 |               64 |
| `standard` |    100,000 |        100,000 |              256 |
| `nightly`  |  1,000,000 |      1,000,000 |              512 |

Each benchmark group header reports the active problem scale, for example
`datafusion-expression [overhead rows=100 array=1000 matmul=16]`, so every
results section names the data size its timings were measured at. The JSON
`extra_info` for every case records the scenario, scale, scale dimensions
(table rows, array elements, matrix dimension), input/output rows, process
RSS, and active array backend. DataFusion cases additionally record planning
time, execution time, and query count reported by the runtime. JAX benchmarks
warm the operation and synchronize device results inside the timed call.

Compare saved reports with `pytest-benchmark` after collecting compatible
runner samples. Do not compare results across different machines, dependency
versions, power modes, or benchmark scales.
