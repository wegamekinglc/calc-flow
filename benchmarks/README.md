# Benchmarks

Calc Flow benchmarks use deterministic Arrow and Array API inputs and export
complete `pytest-benchmark` JSON reports. They are informational until at least
20 comparable main-branch samples have been collected on stable runners.

## Layout

| Path                                               | Contents                                                         |
| -------------------------------------------------- | ---------------------------------------------------------------- |
| `support.py`                                       | Scale selection, metric recording, and identity fingerprints     |
| `array_support.py`, `test_array_*.py`              | Array API kernel, provider, plan, and ownership scopes           |
| `symbolic_support.py`, `test_symbolic_baseline.py` | Symbolic baselines, milestone pairs, and stream lifecycle        |
| `test_datafusion.py`, `test_runtime.py`            | DataFusion operator scenarios and graph fan-out                  |
| `test_rolling_kernel.py`                           | Paired rolling-kernel gate against a DataFusion window reference |
| `rolling_indicator_comparison.py`                  | Standalone cross-library rolling comparison driver               |
| `finance_python_rolling_runner.py`                 | Isolated legacy Finance-Python worker subprocess                 |
| `test_rolling_indicator_comparison.py`             | Correctness tests for the comparison harness                     |
| `rolling/`, `symbolic/`                            | Frozen JSON and Markdown evidence referenced by this README      |

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
repeated execution of a compiled plan. The runtime scenario covers graph
fan-out.

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

## Rolling indicator implementation comparison

`rolling_indicator_comparison.py` is a standalone, correctness-gated example
that compares either one 20-row per-symbol moving average or the composed
dual-SMA spread `SMA(5) - SMA(20)` across:

- the current native incremental rolling operator declared with `ts.mean`;
- the earlier DataFusion `AVG(...) OVER (... ROWS ...)` SQL path;
- Finance-Python 0.9.10's public `MA(...).transform(...)` operator at the
  finance-reference commit `3e33d3e70c3458b4c6dcf76b88df6148229b402c`; and
- [TA-Lib Python 0.7.1](https://github.com/TA-Lib/ta-lib-python/tree/v0.7.1)
  `SMA`, backed by its bundled C library, at release commit
  `a9ff1b47b3ddbd57274116645d688c0ed677338b`.

The default matrix uses up to 64 symbols and 10, 100, 1,000, 10,000, 100,000,
and 1,000,000 rows. The 10-, 100-, and 1,000-row cases measure startup and
partially filled 20-row per-symbol windows; 10,000 rows and above include full
windows for every symbol. Plans, inputs, legacy-worker startup, and warm-up are
outside the timed boundary. TA-Lib accepts one series per call, so its timed
boundary includes per-symbol slicing and contiguous copies, one or two `SMA`
calls per symbol, composition, and reconstruction of timestamp-major output
order.

Before sampling, Calc Flow and Finance-Python outputs must match an independent
direct-window oracle within `rtol=1e-10` and `atol=1e-10`. TA-Lib must match
the same oracle after its documented full-window warm-up: `SMA(20)` returns
`NaN` for each symbol's first 19 observations. The report records the number of
valid TA-Lib outputs at each scale; cases with zero valid outputs measure
all-warm-up invocation cost and are not equivalent valid-output throughput
comparisons.

Twenty samples rotate the four-method order evenly, force garbage collection
outside each timing boundary, and report medians, throughput, raw samples, and
native-relative ratios. Calc Flow and Finance-Python samples repeat a complete
execution until they cover about 50,000 rows, capped at 50 iterations; TA-Lib
uses an independent cap of 200 iterations and targets about 5,000,000 rows to
stabilize its much shorter calls. Large cases reduce these repetition counts.

Finance-Python does not build in Calc Flow's Python 3.13 environment. Prepare a
separate Python 3.9 environment from the frozen upstream source:

```bash
git clone https://github.com/alpha-miner/Finance-Python.git \
  target/third-party/finance-python
git -C target/third-party/finance-python checkout \
  3e33d3e70c3458b4c6dcf76b88df6148229b402c
UV_CACHE_DIR=target/uv-cache uv venv \
  target/finance-python-venv --python 3.9
UV_CACHE_DIR=target/uv-cache uv pip install \
  --python target/finance-python-venv/bin/python \
  'setuptools<70' wheel 'Cython==0.29.37' 'numpy==1.26.4' \
  'pandas==1.5.3' 'scipy==1.13.1' 'simpleutils>=0.1.0' 'six>=1.10.0'
UV_CACHE_DIR=target/uv-cache uv pip install \
  --python target/finance-python-venv/bin/python \
  --no-build-isolation --no-deps target/third-party/finance-python
```

The benchmark extra installs the pinned TA-Lib wheel. Run the comparison and
retain the full provenance and samples:

```bash
UV_CACHE_DIR=target/uv-cache uv run --extra benchmark python \
  benchmarks/rolling_indicator_comparison.py \
  --finance-python-python target/finance-python-venv/bin/python \
  --finance-python-root target/third-party/finance-python \
  --json target/benchmark-results/rolling-indicator-comparison.json
```

Select the cross-library composite indicator explicitly and save it separately:

```bash
UV_CACHE_DIR=target/uv-cache uv run --extra benchmark python \
  benchmarks/rolling_indicator_comparison.py \
  --indicator dual_sma_spread --fast-window 5 --window 20 \
  --finance-python-python target/finance-python-venv/bin/python \
  --finance-python-root target/third-party/finance-python \
  --json target/benchmark-results/composite-indicator-comparison.json
```

This is a diagnostic cross-runtime and cross-abstraction comparison, not an
accepted regression gate: Finance-Python necessarily runs in a separate legacy
interpreter, while TA-Lib calls a much thinner C-backed API. Results from
virtualized or changing-power hosts must not be generalized to production
hardware.

The full single-SMA and composed dual-SMA tables, their interpretation, and the
TA-Lib-streaming-inspired engine roadmap are preserved in the
[rolling and DataFusion upgrade plan](../docs/superpowers/plans/2026-09-04-ta-lib-streaming-rolling-engine-upgrade.md).
The clean-commit reports with full provenance and raw samples are
[`SMA(20)`](rolling/rolling-mean-0052da7.json) and
[`SMA(5) - SMA(20)`](rolling/dual-sma-spread-0052da7.json).
The P1 Arrow-native fast-path diagnostic (three scales, four samples per
scale) is retained separately as
[`SMA(20)`](rolling/p1-rolling-mean-b6bfeed.json) and
[`SMA(5) - SMA(20)`](rolling/p1-dual-sma-spread-b6bfeed.json); it is exact-SHA
engineering evidence, not the final 60-pair regression gate.

The release-facing same-process gate is intentionally separate from that
cross-library diagnostic. It compares the ordered primitive kernel with an
equivalent DataFusion `SUM/COUNT` window reference for both `SMA(20)` and
`SMA(5) - SMA(20)`. The reference shape is outside the `AVG` rewrite allowlist,
so the validator can prove that it still exercised DataFusion's standard
window executor. Each case validates both outputs against an independent
direct-window oracle before collecting 60 alternating pairs:

```bash
CALC_FLOW_BENCHMARK_SCALE=overhead JAX_PLATFORMS=cpu \
  uv run --extra benchmark pytest benchmarks/test_rolling_kernel.py -q \
  --benchmark-only \
  --benchmark-json=target/benchmark-results/rolling-kernel.json

candidate_sha="$(git rev-parse HEAD)"
uv run python scripts/verify_symbolic_milestone_perf.py \
  --report target/benchmark-results/rolling-kernel.json \
  --scenario rolling_kernel_sma20 \
  --scenario rolling_kernel_dual_sma_5_20 \
  --expected-commit "${candidate_sha}" \
  --output target/benchmark-results/rolling-kernel-evidence.json
```

The validator requires a clean exact commit, stable machine/dependency/workload
fingerprints, non-vacuous finite oracle coverage, zero reference rewrites, and
a 95% bootstrap interval from 20,000 resamples. An interval wholly below `+5%`
passes, wholly above it fails, and an overlapping interval is inconclusive.
The cross-library table remains informational because Finance-Python runs in a
legacy interpreter and TA-Lib exposes a thinner C-backed call boundary.

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

Every scenario records rows, batches, peak RSS (`VmHWM`), provider or
DataFusion query counts, and Arrow/dense copy bytes in `extra_info`. The
stream scenario additionally records checkpoint duration, checkpoint bytes,
and recovery duration from a dedicated measured lifecycle: run to half the
batches, checkpoint, cancel, then restore from the durable checkpoint in a
second runner. Cancelling drops the unconsumed half by design, so its output
expectation counts only the windows accumulated before the checkpoint.

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
