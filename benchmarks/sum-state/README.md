# Grouped integer SUM state measurements

The Rust example measures an individual update on a running graph. Build
`grouped_sum_state_benchmark` with `--release --features benchmark-internals`
and pass `--sparse-update sparse.arrow snapshot` or
`--sparse-update sparse.arrow changelog`. Both modes preload the standard
fixture, deliver the 10,000-row seed, and then send one 100-row update at a
time. Each interval starts immediately before submitting a ready input and ends
when the caller receives its output from the sink. The worker retains all 21
outputs through natural end and checks the complete positions after every batch.
The JSON includes the 20 warm update durations in nanoseconds.

Use a new fixture directory to measure the two output modes on the same build:

```bash
cargo build --release -p calc-flow --features benchmark-internals \
  --example grouped_sum_state_benchmark
target/release/examples/grouped_sum_state_benchmark \
  --prepare target/grouped-sum-fixtures
target/release/examples/grouped_sum_state_benchmark \
  --sparse-update target/grouped-sum-fixtures/sparse.arrow snapshot
target/release/examples/grouped_sum_state_benchmark \
  --sparse-update target/grouped-sum-fixtures/sparse.arrow changelog
```

## Compare product revisions

This focused comparison measures the stock-position graph from Rust examples
29/30 on the baseline and candidate product revisions, using one common harness.
It follows the maintained two-round paired-median method in
`scripts/benchmark_suite/report.py`, with ten alternating pairs per round and
its existing 5% decision rule. It does not extend the scheduled case inventory.

| Case                 | Input                            | CPUs | Partitions |
|----------------------|----------------------------------|------|------------|
| `stock-stream-1cpu`  | 10k seeds + 20 × 100k stock rows | 1    | 1          |
| `stock-stream-8cpu`  | 10k seeds + 20 × 100k stock rows | 8    | 1          |
| `sparse-stream-1cpu` | 10k seeds + 20 × 100 updates     | 1    | 1          |
| `stock-batch-8cpu`   | 10k seeds + 20 × 100k stock rows | 8    | 8          |

The dense cases contain 100 accounts × 100 stocks. The sparse case initializes
all 10,000 groups and updates only 100 of them. Stream cases require compact
state with the same 10,000-group budget and checkpointing disabled. All cases
use fixed DataFusion partitions, `min_rows_per_partition=1`, default batch size
8192, and diagnostics enabled. CPU affinity is `0` or `0..7`, with matching
`TOKIO_WORKER_THREADS`. The controller uses CPU 31. Run on a host with those
CPUs available in its affinity mask.

The primary timer starts after IPC input preload and covers graph construction,
compilation, execution, output collection, natural end, and cleanup. Stream
outputs are retained until the timer ends so every snapshot can be checked
against an independent arithmetic reference afterward. Compare matching cases
across builds. The finite case returns one snapshot and the streaming cases
return 21.

Every observation verifies all output rows, names/types/nullability through the
schema digest, unique keys, every snapshot digest, input identity, and clean
stream lifecycle. Native process observations independently confirm affinity and
thread environment after the primary interval. CPU time covers the same
processing interval. Sampled peak
RSS covers the entire fresh worker, including preload and validation, and is
not an isolated operator working-set metric.

## Reproduction

Choose compatible baseline and candidate checkouts. The builder copies the
candidate's benchmark harness into the disposable baseline checkout. It
verifies source seals and forces the product and example to rebuild. Build
time is separate from the acquisition budget.

Run the commands from the current checkout. Replace `BASELINE_SHA` with the
baseline commit and use a clean candidate checkout at `.`. The repository's
`dev` extra supplies `psutil` for the Python controllers.

```bash
git worktree add --detach target/sum-state-baseline BASELINE_SHA
uv run --extra dev python -m scripts.benchmark_suite.sum_state_build \
  --source target/sum-state-baseline --harness-source . \
  --output target/sum-state-build/baseline --target target/sum-state-cargo/baseline
uv run --extra dev python -m scripts.benchmark_suite.sum_state_build \
  --source . --harness-source . \
  --output target/sum-state-build/candidate --target target/sum-state-cargo/candidate
```

Each `build.json` records the exact revision, command, source root and SHA-256
map, binary and common-harness hashes, Cargo lock, compiler, features and build
duration. The collector requires matching shared support modules and vendored
source as well as lock/compiler/features/harness. It verifies all source seals
before and after collection. Build logs and actual Cargo rebuild evidence remain
alongside the receipts.

From the candidate repository root, after all builds/tests have stopped:

```bash
uv run --extra dev python -m scripts.benchmark_suite.sum_state \
  --baseline-build 'target/sum-state-build/baseline/build.json' \
  --candidate-build 'target/sum-state-build/candidate/build.json' \
  --output target/sum-state-results
```

Optional Linux CPU sampling adds `--perf '/absolute/path/to/perf'` and
`--samply '/absolute/path/to/samply'` and, for an
unpacked local perf package, `--perf-library '/absolute/path/to/libraries'`.
Only the baseline and candidate principal one-CPU stream case are sampled, with
six jobs per process and the first excluded. Sampling uses user-space CPU-clock
at 999 Hz with DWARF stacks; exact monotonic processing windows are recorded.
Profiles are separate from unprofiled timings. The collector imports them with
Samply, resolves exact ELF symbol ranges with GNU nm, verifies every per-thread
imported timestamp against original perf timestamps, and attributes samples only
to the five accepted processing windows. `profiles/*/stacks.json.gz` retains all
accepted stacks and counts; `profile-summary.json` records inclusive/self tables,
thread distribution, unresolved leaf samples and disjoint phase counts.

The immutable plan is written before fixture generation. The 600-second total
budget covers fixture creation, preflight, warmups, process startup, full oracle
checks, collection, profile analysis and cleanup; build time is separate.
A competing build/benchmark, failed oracle, source/binary drift, or exhausted
budget stops acquisition without resampling. Offline profile analysis must also
stop at the original deadline. Partial output remains explicitly incomplete.

`acquisition.json` preserves every formal observation, config, resource receipt,
per-snapshot digest, and pair order through the per-invocation command files.
`summary.json` contains all per-case statistics and intervals. `plan.json`,
`environment.json` and fixture manifests preserve provenance. Accepted profile
summaries retain hashes and exact sampling windows.
