# Unified benchmark suite

[Documentation](README.md) / 5.2 Benchmark suite

The suite reports complete workloads and repeated base/head comparisons.
`.github/workflows/benchmark-suite.yml` is the shared entrypoint for ordinary
non-documentation Linux PR/main CI and daily/manual benchmarks. Windows keeps
its existing correctness gates. SQL adaptive tuning experiments remain
supplemental nightly/weekly jobs; they are not missing required suite shards.

On this page:

- [Complete inventory](#complete-inventory)
- [Inputs, correctness and timing boundaries](#inputs-correctness-and-timing-boundaries)
- [Revision comparisons and regression gate](#revision-comparisons-and-regression-gate)
- [Reports and failure behavior](#reports-and-failure-behavior)
- [Local reproduction](#local-reproduction)

## Complete inventory

The catalog is executable: `python -m scripts.benchmark_suite catalog` emits
the same 21 shards consumed by CI. The slow Python `nightly` scale is
excluded from this suite, including its daily/manual workflow calls; overhead,
small and standard remain. The separate engine and warm-state matrices still
run every decade through 10M rows. Dynamic pytest, Criterion and Vitest
inventories preserve benchmark cases without a second hand-written case list.

| Family          | Dimensions                                          | Cases per dimension                                   |
|-----------------|-----------------------------------------------------|-------------------------------------------------------|
| Python          | overhead 1k, small 10k, standard 100k               | All collected non-lifecycle pytest benchmarks         |
| Engines         | 10, 100, 1k, 10k, 100k, 1M, 10M rows                | 22 supported engine/scenario combinations             |
| Warm streaming  | 10, 100, 1k, 10k, 100k, 1M, 10M history; append 64  | SMA(20), SMA(5) minus SMA(20)                         |
| Warm append     | History 1M; append 1, 4, 16, 64, 640, 6,400, 64,000 | Both indicators; append 64 shared with history matrix |
| Rust            | Every `[[bench]]` target in the core crate          | Core, allocation, state/window, join, SQL/DataFusion  |
| Studio/frontend | Python HTTP benchmarks and Vitest benchmark files   | All collected benchmark cases                         |
| Lifecycle       | Isolated checkpoint/recovery benchmark              | Existing minimum-20-round evidence validation         |

There are 154 engine cases and 26 warm cases, in addition to dynamically
discovered cases. Warm cases use one entity to support one-row appends.
Compare measurements only when entity count, history depth, append size,
and timing boundaries match.

| Backend          | Projection  | Filter      | Group by    | Join        | SMA(20) | Dual SMA |
|------------------|-------------|-------------|-------------|-------------|---------|----------|
| Calc Flow SQL    | Yes         | Yes         | Yes         | Yes         | Yes     | Yes      |
| Raw DataFusion   | Yes         | Yes         | Yes         | Yes         | Yes     | Yes      |
| Polars           | Yes         | Yes         | Yes         | Yes         | Yes     | Yes      |
| Native streaming | Unsupported | Unsupported | Unsupported | Unsupported | Yes     | Yes      |
| TA-Lib           | Unsupported | Unsupported | Unsupported | Unsupported | Yes     | Yes      |

Unsupported operations are explicit cells, not silent dependency skips.
Missing DataFusion, Polars or TA-Lib fails its shard. DataFusion Python 54
matches the core's DataFusion major; the shared requirements file pins all
Python build/benchmark/Studio dependencies with hashes.

## Inputs, correctness and timing boundaries

Engine comparisons use identical Arrow input bytes, deterministic entity and
timestamp ordering, up to 64 entities, and 64,000-row input batches. Prices are
bounded exact eighths: `100 + sequence % 257 / 8 + sequence % entities / 8`.
This controls decimal accumulation drift in long rolling performance runs;
it is not a claim of numerical accuracy on arbitrary decimal sequences.
Decimal numerical regression fixtures are checked separately from the
performance workload.
Independent NumPy/direct-window oracles check every measured output, all
payload columns, row counts, warm-up NaNs and finalization. Floating results
use `rtol=1e-10`, `atol=1e-10`, `equal_nan=True`. Both engine-matrix SMA forms
require a full 20-row slow window. Ten-row cases therefore have **zero finite SMA
outputs**: they measure invocation/warm-up cost, not valid-output throughput.

Warm cases retain the partial-window oracle and runner configuration.
`H0` in the result table is the initial historical preload, not a restored
snapshot before every sample. One untimed append warms the worker, so the
first timed cursor is `H0 + append_rows`; subsequent appends advance it.
Each raw sample records its exact `start_row`, checked against that sequence
on both sides. Warm cases retain the existing decimal input fixture.

| Scope                  | Included in measurement                                                                    | Excluded                                                    |
|------------------------|--------------------------------------------------------------------------------------------|-------------------------------------------------------------|
| Calc Flow SQL          | Plan execution, run session, registration, SQL planning/execution, output `to_pyarrow`     | Input construction, graph compilation, warm-up, validation  |
| Raw DataFusion         | Python `SessionContext`, table registration, SQL planning/collection, Arrow table          | Input construction, query text, warm-up, validation         |
| Polars                 | Lazy-plan collection and Arrow output                                                      | Arrow input conversion, lazy expression construction        |
| TA-Lib                 | Per-entity contiguous copies, SMA calls, composition, Arrow output                         | Input construction and validation                           |
| Ready native streaming | Input enqueue, sources/tasks/channels, rolling, watermarks, sink and combined Arrow output | Plans, input events, runner startup/readiness, EOF/shutdown |
| Warm native streaming  | Preconstructed data enqueue, live source/task/channel, rolling/finalization, sink to Arrow | Compilation, runner start, historical preload, validation   |

SQL/raw-DataFusion target partitions, Tokio workers and Polars threads
are fixed to 32. BLAS helper pools use one thread. The Python benchmark fixtures retain
their own query configuration and timing boundaries. Cross-library native
streaming uses an already-started runner with **empty rolling state**. Each
invocation compiles a fresh single-use plan, awaits runner startup and the
source's first poll through the startup data gate, then starts the timer before
enqueueing preconstructed data and watermarks. It stops after every expected
row reaches the sink and the Arrow tables are combined. EOF, job completion
checks and cancellation/cleanup happen afterward, outside timing; unexpected
extra output during completion still fails the sample. No dummy data or history
is preloaded. Persistent warm append remains a separate workload.

Cross-library columns are application-boundary references, not interchangeable
kernel measurements. The native column is labeled `Native stream (ready)`.
Report contract v3 validates the ready-runner timing scope and complete
sample statistics. Both revisions must be measured with the same scope;
do not subtract a separately measured startup time from another report.
These settings describe target/pool sizes, not measured CPU utilization;
TA-Lib calls remain sequential per-series operations.

## Revision comparisons and regression gate

CI resolves immutable base/head commits before building clean release wheels.
PRs compare the event's base SHA with its head SHA. Pushes use `before`;
scheduled/manual runs default to the head's first parent. A manual full
baseline SHA can override that choice. There is no silent fallback to a
different successful run or debug wheel.

For every Calc Flow engine/warm case:

1. Install both sealed wheels into separate import directories on one runner.
2. Start a fresh worker pair, verify loaded native hashes, dependencies,
   machine/thread identities and workload dimensions; warm up outside timing.
3. Collect ten pairs in alternating AB/BA order. Warm workers advance through
   exactly the same input cursors. No forced GC is included in the interval.
4. Repeat with a fresh worker pair. Retain every original pair; estimate each
   round's median of `100 * (candidate_i / baseline_i - 1)` and its interval.
   Show combined P50/P95, throughput and round minimum ratios separately.

For each round, use a conservative exact 95% median confidence interval from
binomial order statistics, without interpolation or bootstrap randomness.
With ten pairs its bounds are the second and ninth sorted changes, giving
97.85% coverage under independent pairs with a common change distribution.
This construction follows [NIST TN 2119, section 5.3](https://nvlpubs.nist.gov/nistpubs/TechnicalNotes/NIST.TN.2119.pdf).
Alternating AB/BA mitigates drift but does not prove independence or remove
hosted-runner autocorrelation. Coverage is per round, not simultaneous over
the complete matrix; a median interval does not bound tail latency.

The gate fails only when **both** round confidence lower bounds exceed +5%.
If any upper bound exceeds +5% without both lower bounds exceeding it, the
result is `inconclusive`. Both upper bounds below -5% indicate `improved`;
otherwise the result is `no-confirmed-regression`, not proof of equivalence.
Minimum ratios remain diagnostic: comparing unrelated best samples can signal
a slowdown even with identical binaries and nearly unchanged P50 values.
The fixed +5% threshold, two-round sample budget and correctness checks remain
unchanged; CI does not retry measurements to select a passing timing result.
External libraries are measured references, never fake historical baselines.
The pytest/Criterion/Vitest suites run ABBA whole-suite
blocks. Their deltas remain informational because those blocks are not
per-call paired observations. Allocation counters have a separate unit-correct
table and are not mislabeled as milliseconds.

## Reports and failure behavior

The final always-run job publishes all result rows, with dimensions, timing
scope, base/head P50, P95, rows/s, percentage change, diagnostic round minima,
paired round medians with confidence bounds, and verdict.
A second table places all five engine implementations side by side. No top-N
filtering is applied. Build, measurement and summary artifacts retain raw
JSON/JSONL samples, original runner formats, stdout/stderr, release/native and
harness hashes, exact source refs and environment identities for 30 days.
Vitest's default JSON reporter discards samples; the adapter explicitly
collects its retained task samples and verifies their counts before export.

Wrong/dirty releases, incompatible workload fingerprints, missing or duplicate
cases/shards, changed confirmation environments, failed correctness and
nonfinite observations all fail closed. Missing known catalog cases appear as
error rows. If a suite runner fails before discovery, its unavailable inventory
is explicitly shown rather than invented. The complete Markdown/JSON remains
an artifact if it exceeds GitHub's step-summary size limit; overflow fails
instead of silently truncating rows.

## Local reproduction

Use clean candidate and baseline checkouts. Run the current candidate harness
for both releases; it supplies the same workload to both engine revisions.
Generated files stay under `target/`, not in `python/calc_flow/`.

```bash
UV_CACHE_DIR=target/uv-cache uv venv target/benchmark-venv
UV_CACHE_DIR=target/uv-cache uv pip sync \
  --python target/benchmark-venv/bin/python \
  --require-hashes benchmarks/requirements.lock
git worktree add --detach target/base HEAD^
target/benchmark-venv/bin/python -m scripts.benchmark_suite build \
  --source target/base --output target/releases/baseline
target/benchmark-venv/bin/python -m scripts.benchmark_suite build \
  --source . --output target/releases/candidate
target/benchmark-venv/bin/python -m scripts.benchmark_suite run \
  --shard engines-1000 \
  --baseline target/releases/baseline/release.json \
  --candidate target/releases/candidate/release.json \
  --baseline-source target/base --output target/results/engines-1000
```

Run every emitted catalog shard to reproduce the complete CI gate. A single
shard's own `summary.md` is useful locally; the complete summarizer deliberately
fails when shards are missing. To update dependencies, regenerate and commit
`benchmarks/requirements.lock` using the command in its header. CI checks lock
drift before its adapter tests.
