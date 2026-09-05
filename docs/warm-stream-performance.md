# Warm-state streaming performance

This benchmark measures the production Python `StreamingRunner` path with a
long-lived native rolling operator. It is not a standalone kernel benchmark,
and it is not directly interchangeable with a TA-Lib array call or a SQL query
over a complete historical table.

The [measured two-round report](warm-stream-results-39a0c7f.md) compares clean
release builds at `9044501` and `39a0c7f`, with all raw samples and explicit
forecast misses. Later report-only commits do not change the measured engine.

## What is measured

Each scenario starts one runner, feeds and validates historical data, and
advances a source-provided watermark. Each timed append starts with a
preconstructed immutable Data event and ends when the Python sink has converted
the emitted Batch with `to_pyarrow` and the benchmark has received its table.
Source polling, Rust/Python dispatch, task/channel work, rolling ingestion,
watermark finalization, projection, and sink handling remain inside timing.
Compilation, startup, input construction, historical warmup, correctness
validation, worker IPC, checkpoints, and shutdown remain outside timing.

Baseline and candidate execute the same graph and input values in separate
persistent Python processes. The controller alternates AB/BA order; both
workers advance through the same appended row identities. The current controller
creates a fresh process pair for each scenario and shuffles scenario order with
the recorded `--seed` (default `20260905`). The older two-round report used
persistent workers across scenarios; its results are not silently relabeled.
Nothing prefetches
extra source events to improve the timing. Scheduled checkpoints are set to a
24-hour interval so they do not overlap these bounded runs.

The default matrix has 64 entities and seven distinct (history, append) pairs:

| Dimension        | History rows                           | Appended rows           |
| ---------------- | -------------------------------------- | ----------------------- |
| Historical depth | 10,240 / 102,400 / 1,024,000 / 10,240,000 | 64                      |
| Incremental size | 1,024,000                              | 64 / 640 / 6,400 / 64,000 |

Each pair runs SMA(20) and SMA(5) minus SMA(20), with both forced collection
before each sample and normal Python GC. Normal GC is not disabled; the
controller still introduces untimed IPC gaps between appends. The two GC
modes must be compared separately. Removing the pre-sample `gc.collect()`
from the older harness is not an engine optimization.

## Reproduce

Use Python 3.13 or newer with the benchmark dependencies installed. Create a
clean checkout of the selected baseline under `target/warm-baseline`; keep the
candidate in the current checkout. Build each with a separate Cargo target
directory so worktree-local dependency caches cannot be confused:

```bash
uv run --extra benchmark python scripts/profile_warm_stream.py build \
  --source target/warm-baseline \
  --target target/warm-baseline-cargo \
  --output target/warm-baseline-wheels
uv run --extra benchmark python scripts/profile_warm_stream.py build \
  --source . \
  --target target/warm-candidate-cargo \
  --output target/warm-candidate-wheels
uv run --extra benchmark python scripts/profile_warm_stream.py compare \
  --baseline-build target/warm-baseline-wheels/build.json \
  --candidate-build target/warm-candidate-wheels/build.json \
--samples 30 --output target/warm-stream.json
```

The native warm-state workload and its regression tests do not require TA-Lib.
That optional package is imported only when the separate cross-library
comparison executes TA-Lib or checks its version.

### Small and sparse appends

Pass `--append-entities` to select a separate sparse workload. History still
contains all 64 entities; increments cycle over the selected active entities.
Each appended row gets its own increasing timestamp, so even a one-row append
can legally advance its watermark without closing a later row's timestamp.
Prices retain the same deterministic sequence-based generator. The oracle
uses only each entity's last window context, and is independently checked
against complete per-entity histories across the warm/incremental boundary.

```bash
uv run --extra benchmark python scripts/profile_warm_stream.py compare \
  --baseline-build target/warm-baseline-wheels/build.json \
  --candidate-build target/warm-candidate-wheels/build.json \
  --history-rows 1024000 --append-rows 1 4 16 64 \
  --append-entities 1 4 16 64 --samples 30 \
  --output target/small-appends.json
```

The sparse matrix is the Cartesian product of history, append and active-entity
counts. Omitting `--append-entities` preserves the original seven-point matrix
and complete-tick timestamps. These layouts must not be pooled together.

### Opt-in callback diagnostics

The private native job diagnostics are disabled by default. Enabling them
records up to 1,024 completed or cancelled callback requests, evicts the oldest
records at capacity and reports how many were dropped. Draining returns JSON
and resets the buffer/counter; terminal root cleanup does not erase records.
No Python objects are retained by this diagnostic buffer.

```bash
uv run --extra benchmark python scripts/profile_warm_stream.py diagnose \
  --build target/warm-candidate-wheels/build.json \
  --history-rows 1024000 --append-rows 64 --samples 30 --forced-gc \
  --output target/callback-diagnostics.json
```

Offsets are nanoseconds relative to the start of one bridge call:
`attached_ns` marks interpreter attachment; `queued_ns` follows callback
creation, awaitability checks and scheduler construction; `dispatched_ns`
marks entry into the captured Python loop; `completed_ns` precedes the
completion-channel send; `elapsed_ns` marks Rust's return or cancellation.
Missing boundaries stay null, with explicit ready/completed/failed/cancelled
outcomes. Queue time includes submission and dispatch; completed-to-elapsed
includes Rust resumption and cleanup. Source decoding happens afterwards.

These are instrumented wall intervals, not CPU samples or authoritative
speedup evidence. A prearmed source request may wait across input construction,
GC, worker IPC or shutdown. Therefore its callback lifetime must not be summed
as if it belonged entirely to one timed append. Run diagnostics separately
from uninstrumented paired comparisons.

The build command runs Maturin in release mode, verifies that source identity
did not change during the build, and records the source, lockfile, compiler,
wheel, and native-module fingerprints. The comparator rejects changed source
or wheel content, mismatched toolchains/dependencies, and a worker loading a
different native module. It records whether each source checkout was clean.
Dirty-checkout measurements are exploratory, not final-commit evidence.

The comparator installs wheels into temporary directories beneath the report
directory. It does not overwrite the caller's installed Calc Flow or place a
native shared library under `python/calc_flow`. Run measurements without
concurrent compiles, tests, or other benchmark workloads. Avoid rewriting a
running Rust test executable from another build: checkpoint smoke tests may
re-execute that exact binary.

## Interpreting the report

JSON retains every paired latency, correctness result, callback timestamp
interval, initial/final operator status, and terminal state. Markdown reports
P50/P95 and median paired speedup with a seeded 20,000-resample bootstrap
interval. A speedup greater than one favors the candidate. The interval is
not a guarantee for another machine, entity count, window size, or connector.
P95 from small sample counts is exploratory; do not publish a five-pair smoke
run as stable tail-latency evidence.

The report summarizes callback intervals separately for both builds. JSON
retains interval P50/P95 for every scenario; Markdown includes the interval
P50s for the fixed 1,024,000-row history with forced collection. Each sample's
four non-overlapping callback intervals must cover its total timed duration,
and conversion must fit within the sink-to-receive interval. Independent
interval medians need not add up to the total latency median.

Callback wall intervals are not CPU attribution: work in different tasks can
overlap. `to_pyarrow` is a subset of the sink-to-receive interval, so adding
both double-counts conversion. Operator `processing_duration_micros` includes
successful Data and watermark handlers in the candidate; the watermark subset
is also available as `watermark_processing_duration_micros`. Older baseline
metrics omit watermark handling and cannot be compared as if they measured
the same total. End-handler time is excluded from these operator counters.

The optimization keeps a validated ordered columnar path and an equivalent
general fallback. It reduces row materialization, per-row output-budget
allocations, whole-state cloning, unchanged retained-row copying, and redundant
SQL work. The Python bridge coalesces ready wakeups and removes extra event-loop
turns for immediately completed callbacks. Neither changes finality or the
durable checkpoint layout. See [runtime-envelope.md](runtime-envelope.md) for
the scheduling and metric contract.

Pre-implementation latency ranges are hypotheses. The PR's measured report
must explicitly identify targets reached, targets missed, uncertainty, and any
remaining bottleneck; completing an optimization does not prove its forecast.
