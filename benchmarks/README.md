# Benchmarks

Calc Flow benchmarks use deterministic Arrow and Array API inputs and export
complete `pytest-benchmark` JSON reports. They are informational until at least
20 comparable main-branch samples have been collected on stable runners.

Install and run the overhead suite:

```bash
uv sync --extra benchmark
CALC_FLOW_BENCHMARK_SCALE=overhead \
  JAX_PLATFORMS=cpu \
  uv run pytest benchmarks --benchmark-only \
  --benchmark-json=benchmark-results/overhead.json
```

Available scales:

| Scale      | Table rows | Array elements | Matrix dimension |
| ---------- | ---------: | -------------: | ---------------: |
| `overhead` |      1,000 |          1,000 |               16 |
| `small`    |     10,000 |         10,000 |               64 |
| `standard` |    100,000 |        100,000 |              256 |
| `nightly`  |  1,000,000 |      1,000,000 |              512 |

Each benchmark reports the active problem scale in two places: the group
header carries the full spec (for example
`datafusion-expression [overhead rows=1000 array=1000 matmul=16]`), and each
test ID / results-table row carries the scale name (for example
`test_group_by_aggregation[overhead]`), so a result names its data size
whether read in context or in isolation. The JSON `extra_info` for every case
records the scenario, scale, scale dimensions (table rows, array elements,
matrix dimension), input/output rows, process RSS, and active array backend.
DataFusion cases additionally record planning time, execution time, and query
count reported by the runtime. JAX benchmarks warm the operation and
synchronize device results inside the timed call.

The v2 suite covers DataFusion projections, filters, aggregates, joins,
windows, trusted Python scalar UDFs, explicit session configuration, and
repeated execution of a compiled plan. Array scenarios run the bounded v2
expression provider for both NumPy and JAX: elementwise arithmetic,
reductions, matrix multiplication, and transpose/reshape. Runtime scenarios
cover graph fan-out plus strict v2 checkpoint serialization, atomic writes,
and recovery reads.

Compare saved reports with `pytest-benchmark` after collecting compatible
runner samples. Do not compare results across different machines, dependency
versions, power modes, or benchmark scales.
