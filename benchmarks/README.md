# Benchmarks

Calc Flow benchmarks use deterministic Arrow and Array API inputs and export
complete `pytest-benchmark` JSON reports. They are informational until at least
20 comparable main-branch samples have been collected on stable runners.

Install and run the overhead suite:

```bash
uv sync --extra benchmark
mkdir -p target/benchmark-results
CALC_FLOW_BENCHMARK_SCALE=overhead \
  JAX_PLATFORMS=cpu \
  uv run pytest benchmarks --benchmark-only \
  --benchmark-json=target/benchmark-results/overhead.json
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
count reported by the runtime.

The v2 suite covers DataFusion projections, filters, aggregates, joins,
windows, trusted Python scalar UDFs, explicit session configuration, and
repeated execution of a compiled plan. Runtime scenarios cover graph fan-out
plus strict v2 checkpoint serialization, atomic writes, and recovery reads.

## Array measurement scopes

Array benchmarks use the same deterministic elementwise, reduction, matrix
multiplication, and transpose/reshape workloads for NumPy and JAX. Each scope
has an explicit timing boundary:

- `backend_kernel` times only the raw backend operation and JAX completion.
  NumPy input creation, JAX transfer, warm-up, metadata, and assertions remain
  outside timing.
- `provider_boundary` times bounded parsing/evaluation, intermediate
  validation, Python/native conversion, output ownership, and JAX completion.
  The namespace, native input batch, warm-up, metadata, and assertions remain
  outside timing.
- `plan_end_to_end` times exactly
  `plan.execute({"input": values}).outputs["output"]` plus JAX completion.
  Provider registration, graph compilation, input construction, warm-up, and
  metadata remain outside timing.
- `batch_ownership` times `Batch.from_array(source, backend=backend)` plus JAX
  completion. Source creation, warm-up, metadata, caller-isolation assertions,
  and NumPy base-chain assertions remain outside timing.

Every timed callable is warmed once. JAX values are synchronized during the
warm-up and every timed call so asynchronous dispatch is not reported as
completed work.

The `plan_end_to_end` acceptance set contains six cases: elementwise, mean,
and matrix multiplication for NumPy and JAX. Transpose/reshape remains a
diagnostic for each backend but is excluded from the acceptance geometric
mean. Run `batch_ownership` at both `standard` and `nightly` when evaluating
the 100,000- and 1,000,000-element NumPy ownership thresholds.

## Array compatibility contract

Array reports use contract version 2 and record the workload version,
scenario, scope, backend, expression, scale and dimensions, input/output rows,
observed input/output dtypes, backend configuration, process RSS, and complete
machine, dependency, and workload identities with lower-case SHA-256
fingerprints. JAX reports additionally record JAX/JAXlib versions, platform,
and x64 mode; NumPy reports omit those fields.

Classify performance only when contract-v2 reports have matching machine,
dependency, and workload fingerprints. Incompatible identities produce no
timing classification. Reports without contract-v2 metadata are legacy
artifacts: the Studio can display them as `unverified`, but it must not label
them stable, improved, or regressed.

Compare saved reports with `pytest-benchmark` after collecting compatible
runner samples. Do not compare results across different machines, dependency
versions, power modes, or benchmark scales. CI publishes these measurements as
informational artifacts; it does not fail builds on benchmark deltas until at
least 20 comparable main-branch samples exist on stable runners.
