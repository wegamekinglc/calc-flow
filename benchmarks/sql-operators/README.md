# SQL operator coverage: DAL-389

All 98 added SQL/DataFusion cells completed: two rounds of ten measured samples per cell, 1,960 measured correctness checks passed. Maximum absolute oracle error: 0. Collection elapsed 1184.6 seconds within the declared 2,700-second budget; builds and validation were separate.

The gaps were cumulative average, 64/256-row argmax, 64-row distinct count, exact-time cross-section mean, ten-second tumbling sum and zero-tolerance ASOF. SQL argmax now resolves DataFusion's `array_position` through an explicit native session registration. The other gaps needed SQL queries and catalog registration. No case uses a streaming fallback.

Baseline catalog: `abadd7fe6a02acfc6d5b4f55ba1ab6ce22b4fcb8` (the default branch at task start).
Measured candidate: `23edd3a39ad5b8ab9c493387b3e0ba9d22641442`, clean release build.

[Raw results](results.json) retain every measured sample, correctness result, round environment and native hash. [Build record](build.json) records the exact command, compiler, source/lock hashes and wheel/native seals. [Host record](host.json) records CPU and memory. The issue attachment also includes worker requests, lifecycle traces, partial-round records, build and validation logs.

## Parameters and interpretation

This is the 98-cell addition to the maintained matrix. Existing catalog cells were not remeasured. The original row scales (10 through 10M), exact-eighth prices, entity counts, 64,000-row Arrow chunks, 32 target partitions, 8,192-row execution batches and execute-to-Arrow timer are retained. Fixture and graph construction, warmup and correctness validation stay outside timing; SQL planning/execution and Arrow materialization remain inside timing. Each round uses a fresh release-bound worker and validates its loaded native hash.

Calc Flow SQL rows are `new-coverage` because the baseline catalog lacks their IDs; raw DataFusion rows are `external-reference`. These candidate-only measurements do not establish a speedup or a paired regression verdict. ASOF preserves the original zero tolerance and uses a left equality join with latest-reference-sequence tie breaking; this does not measure general nonzero-tolerance ASOF. Detailed operator contracts are in [the suite guide](../../docs/benchmark-suite.md#inputs-correctness-and-timing-boundaries).

## Environment

2026-10-20; Intel Core i9-13900HX, 16 exposed cores / 32 logical CPUs, 31.2 GiB RAM; Linux 5.15.167.4 WSL2. This shared host supplies informational timing results.
Python 3.13.9; NumPy 2.5.2; PyArrow 24.0.0; DataFusion 55.0.0; Rust/Cargo 1.94.0. The measured wheel is CP313/CP313 with release optimization. Tokio workers and target partitions are 32; OMP/OpenBLAS/MKL threads are 1. Full worker environments are retained in the raw results.

The native runtime reuses the already-resolved `datafusion-functions-nested` dependency and registers only `array_position`. An offline Cargo metadata comparison confirms identical versions and enabled features for all 322 registry packages against the baseline; the lockfile adds only the core crate's direct dependency edge. Existing compiled-dependency and workload gates are unchanged.

Native SHA-256: `13e801f24536b27553bc9bff8897f363f8f023da5ef97981f67f3c4c52b30122`.

## Results

All times are milliseconds. P50 and P95 use the maintained suite's statistics over both complete rounds. Every cell below passed its independent oracle on every warmup and measured execution.

| Input rows | Scenario   | Calc Flow SQL P50 | Calc Flow SQL P95 | DataFusion P50 | DataFusion P95 |
|------------|------------|-------------------|-------------------|----------------|----------------|
| 10         | average    | 0.516             | 0.754             | 0.652          | 0.784          |
| 10         | argmax64   | 0.917             | 1.317             | 0.941          | 1.069          |
| 10         | argmax256  | 0.869             | 1.037             | 0.969          | 1.213          |
| 10         | unique64   | 0.616             | 1.055             | 0.679          | 0.845          |
| 10         | cs_mean    | 0.488             | 0.662             | 0.560          | 0.964          |
| 10         | window_sum | 0.830             | 1.118             | 1.601          | 2.113          |
| 10         | asof_join  | 1.043             | 1.549             | 1.854          | 2.635          |
| 100        | average    | 0.533             | 0.716             | 0.660          | 0.909          |
| 100        | argmax64   | 0.941             | 1.114             | 1.035          | 1.233          |
| 100        | argmax256  | 0.962             | 1.264             | 1.007          | 1.185          |
| 100        | unique64   | 0.577             | 0.741             | 0.674          | 0.811          |
| 100        | cs_mean    | 0.521             | 0.654             | 0.553          | 0.663          |
| 100        | window_sum | 0.789             | 0.972             | 1.846          | 2.693          |
| 100        | asof_join  | 1.171             | 1.642             | 1.777          | 2.587          |
| 1,000      | average    | 0.776             | 0.999             | 0.836          | 1.042          |
| 1,000      | argmax64   | 2.284             | 2.679             | 2.239          | 2.564          |
| 1,000      | argmax256  | 2.266             | 2.484             | 2.208          | 2.514          |
| 1,000      | unique64   | 0.818             | 1.094             | 0.941          | 1.147          |
| 1,000      | cs_mean    | 0.511             | 0.711             | 0.616          | 0.757          |
| 1,000      | window_sum | 0.793             | 0.921             | 2.661          | 3.685          |
| 1,000      | asof_join  | 2.221             | 2.648             | 2.432          | 2.925          |
| 10,000     | average    | 2.821             | 2.993             | 2.502          | 4.093          |
| 10,000     | argmax64   | 21.401            | 22.664            | 5.715          | 8.235          |
| 10,000     | argmax256  | 26.228            | 27.705            | 6.218          | 6.959          |
| 10,000     | unique64   | 3.613             | 3.935             | 2.503          | 3.535          |
| 10,000     | cs_mean    | 0.630             | 0.840             | 1.760          | 3.658          |
| 10,000     | window_sum | 1.053             | 1.206             | 3.091          | 4.442          |
| 10,000     | asof_join  | 14.159            | 16.942            | 6.201          | 6.824          |
| 100,000    | average    | 10.671            | 13.048            | 6.897          | 8.343          |
| 100,000    | argmax64   | 123.206           | 133.953           | 30.078         | 34.047         |
| 100,000    | argmax256  | 311.799           | 352.463           | 70.645         | 75.636         |
| 100,000    | unique64   | 16.679            | 17.681            | 7.918          | 8.800          |
| 100,000    | cs_mean    | 3.489             | 4.562             | 3.437          | 3.910          |
| 100,000    | window_sum | 2.983             | 3.353             | 3.665          | 4.407          |
| 100,000    | asof_join  | 49.982            | 61.167            | 30.954         | 33.320         |
| 1,000,000  | average    | 35.191            | 39.381            | 46.835         | 53.612         |
| 1,000,000  | argmax64   | 398.216           | 524.066           | 310.890        | 347.379        |
| 1,000,000  | argmax256  | 1111.920          | 1609.588          | 777.840        | 806.678        |
| 1,000,000  | unique64   | 61.684            | 69.401            | 63.607         | 68.218         |
| 1,000,000  | cs_mean    | 10.398            | 10.918            | 19.500         | 22.904         |
| 1,000,000  | window_sum | 6.845             | 7.933             | 7.730          | 8.938          |
| 1,000,000  | asof_join  | 284.997           | 306.299           | 268.162        | 276.806        |
| 10,000,000 | average    | 325.997           | 361.720           | 334.146        | 352.444        |
| 10,000,000 | argmax64   | 3264.711          | 3361.253          | 3057.253       | 3143.271       |
| 10,000,000 | argmax256  | 8623.695          | 8778.641          | 8170.833       | 8302.964       |
| 10,000,000 | unique64   | 503.758           | 531.901           | 498.679        | 517.405        |
| 10,000,000 | cs_mean    | 79.475            | 99.917            | 79.361         | 86.848         |
| 10,000,000 | window_sum | 29.366            | 34.539            | 32.035         | 33.386         |
| 10,000,000 | asof_join  | 2719.667          | 2955.623          | 2538.415       | 2674.797       |

## Commands and validation

The actual collection command was:

```bash
PYTHONPATH=. target/benchmark-venv/bin/python benchmarks/sql-operators/collect.py \
  --release target/sql-release/release.json \
  --baseline-source target/sql-baseline \
  --output target/sql-operator-results --budget-seconds 2700
```

The baseline checkout used `git worktree add --detach target/sql-baseline abadd7fe6a02acfc6d5b4f55ba1ab6ce22b4fcb8`. Dependencies came from `uv pip sync --python target/benchmark-venv/bin/python --require-hashes benchmarks/requirements.lock`. The actual native build command is recorded in `build.json`; it used `maturin build --release --locked`. For a fresh run, the maintained `python -m scripts.benchmark_suite build --source . --output target/sql-release` entry point produces a new sealed manifest (with its own build/ABI identity), then the same collector can be used against the baseline catalog.

Validation used the same native SHA-256 as the sealed measurement wheel:

```bash
# 47 passed
JAX_PLATFORMS=cpu TOKIO_WORKER_THREADS=32 target/benchmark-venv/bin/python -m pytest \
  python/tests/test_native_sql_array_position.py benchmarks/test_sql_operators.py -q
# 60 passed; 141 unrelated cases deselected
JAX_PLATFORMS=cpu TOKIO_WORKER_THREADS=32 target/benchmark-venv/bin/python -m pytest \
  benchmarks/test_engine_comparison.py -q \
  -k 'interval_sql or (engine_outputs and (calc-flow-sql or datafusion))'
# 63 passed
target/benchmark-venv/bin/python -m unittest scripts.test_benchmark_suite \
  scripts.test_benchmark_measure scripts.test_benchmark_engine_lifecycle -q
# 21 passed
target/benchmark-venv/bin/python -m unittest scripts.test_benchmark_workflow -q
# Passed; build cache selected with CARGO_TARGET_DIR
cargo clippy --release --locked -p calc-flow --lib -- -D warnings
cargo fmt --all --check
```

Changed Python files also passed scoped Ruff checks and formatting. Native regressions cover schema planning and execution without the optional Python DataFusion package. Benchmark regressions cover independent oracles, actual 640-row native stream batches, input immutability, oldest argmax ties, distinct-value expiry, UTC/epoch window boundaries and unmatched/latest-tie ASOF rows. Full regression, cross-platform tests and coverage remain CI gates.
