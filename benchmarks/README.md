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

## Symbolic execution baselines

`test_symbolic_baseline.py` retains the hand-built SCE-01 baselines and adds
same-process hand-built/symbolic pairs as later milestones land. The baseline
method is documented in [symbolic/BASELINE.md](symbolic/BASELINE.md); the
accepted milestone gates and their raw evidence are documented in
[symbolic/SCE05.md](symbolic/SCE05.md),
[symbolic/SCE08.md](symbolic/SCE08.md),
[symbolic/SCE14.md](symbolic/SCE14.md), and
[symbolic/SCE16.md](symbolic/SCE16.md).

| Scenario                                  | Timed boundary                                             |
| ----------------------------------------- | ---------------------------------------------------------- |
| `symbolic_projection_20_columns`          | one DataFusion execute of a 20-column row-local SQL        |
| `symbolic_rolling_20_60_row_features`     | one DataFusion execute of rolling window SQL               |
| `symbolic_cross_section_rank_zscore`      | one DataFusion execute of complete-group rank/z-score      |
| `symbolic_table_matmul_numpy`/`_jax`      | SQL features plus one counting table_matmul call           |
| `symbolic_stream_window_checkpoint`       | full stream lifecycle (see below)                          |
| `sce05_row_local_20_columns`              | alternating hand-built/symbolic single projections         |
| `sce08_temporal_catalog`                  | alternating native/symbolic duration rolling runs          |
| `sce14_cross_domain_sharing`              | separate versus shared rolling and cross-section branches  |
| `sce16_exponential_indicators`            | alternating hand-built/symbolic EWMA and MACD runs          |
| `symbolic_multistage_rolling_sharing`     | separate versus shared two-stage rolling output branches   |
| `incremental_rolling_vs_sql_window`       | native incremental rolling versus SQL row-window functions |

Every scenario records rows and peak RSS (`VmHWM`) in `extra_info`, then adds
the applicable batch, provider or DataFusion query, and Arrow/dense-copy
measurements for its execution path. The stream scenario additionally records
checkpoint duration, checkpoint bytes, and recovery duration from a dedicated
measured lifecycle: run to half the batches, checkpoint, cancel, then restore
from the durable checkpoint in a second runner. Cancelling drops the unconsumed
half by design, so its output expectation counts only the windows accumulated
before the checkpoint.

The stream workload is capped at 50,000 rows regardless of scale so the
`nightly` matrix stays bounded; every other dimension (seed, entity count,
batch size, window size) is fixed in `symbolic_support.py` and identical
across scales, which keeps paired comparisons valid. The matmul scenarios
likewise cap rows at 400,000 so the dense 20-column feature matrix stays
under the runtime's owned-NumPy 10,000,000-element conversion limit.

Run the stream lifecycle in its own process. The PR smoke runs the same
node selection at `overhead` scale and the scheduled workflow uses
`standard`; either way symbolic compilation cases cannot retain allocator or
memory-pool state before the stream measurement:

```bash
CALC_FLOW_BENCHMARK_SCALE=standard \
  JAX_PLATFORMS=cpu \
  uv run pytest \
  benchmarks/test_symbolic_baseline.py::test_stream_window_checkpoint_and_recovery \
  --benchmark-only --benchmark-json=<stream-output>.json
uv run python scripts/verify_stream_lifecycle_evidence.py \
  <stream-output>.json --minimum-rounds 20
```

The evidence contract requires at least 20 lifecycle rounds and 20 diagnostic
samples. It records startup, steady processing, checkpoint, cancellation,
restore, and shutdown durations; checkpoint-byte and checkpoint/restore
p50/p95 values; RSS before and after; peak RSS; and matching machine,
dependency, and workload fingerprints. Release exact-ref comparison fails when
the candidate p50 exceeds the baseline p95 plus 5%, or when checkpoint-byte
p50 crosses the equivalent bound.

Scheduled Rust evidence runs both `core` and `stream_join_perf`. The core
harness includes two-, four-, and eight-channel fan-in, four-way fan-out, and
saturated backpressure cases. `benchmark-provenance.json` binds the artifact to
the exact commit, Rust toolchain, Cargo lock, runner identity, CPU identity, and
benchmark source hashes. Release gates compare both Rust harnesses at exact
baseline and candidate refs; scheduled timing remains informational.

Studio observability coverage runs separately with `standard` inputs. It
measures checkpoint-directory scans for 1x100, 10x100, and 100x10 job/file
shapes plus 10,000-row JSON and chunked Arrow IPC decode/`combine_chunks`
paths. The frontend benchmark exercises report matching at 100 and 1,000
cases and records the exact commit, Node/npm versions, package lock, and
benchmark source hashes in its artifact. These scheduled Studio and
frontend results remain informational: no automated threshold consumes them,
and their workflow-presence tests fail closed if a scenario is silently
removed. The only timing gates are the release exact-ref comparison above and
the isolated stream lifecycle evidence contract.

Reproduce a recorded run with:

```bash
CALC_FLOW_BENCHMARK_SCALE=standard \
  JAX_PLATFORMS=cpu \
  uv run pytest benchmarks/test_symbolic_baseline.py -q --benchmark-only \
  --benchmark-json=<output>.json
```

The focused rolling comparison validates native and SQL results before timing,
then alternates 30 samples of each implementation in the same process. See
[rolling/DAL184.md](rolling/DAL184.md) for its null and boundary semantics,
environment contract, reproduction command, and latest measured summary.
