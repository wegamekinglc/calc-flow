# Unified benchmark suite

[Documentation](README.md) / 5.2 Benchmark suite

The suite reports complete workloads and repeated base/head comparisons.
`.github/workflows/benchmark-suite.yml` runs independently of regular CI at
06:00 and 18:00 Asia/Shanghai every day (22:00 and 10:00 UTC), and also supports
manual runs. Regular Linux and Windows CI retain unit tests and coverage gates.
Benchmark and performance support script tests, warm-stream scenario tests, and
performance controller tests run in this scheduled workflow. Supplemental SQL
adaptive tuning experiments run only by manual dispatch from `benchmarks.yml`;
they are not required suite shards.

Historical JSON results under `benchmarks/rolling/` are stored with Git LFS.
Install Git LFS and run `git lfs pull --include='benchmarks/rolling/*.json'`
when inspecting those results locally. Scheduled benchmark runs generate their
own artifacts and do not read these historical files.

On this page:

- [Feature measurement scope](#feature-measurement-scope)
- [Complete inventory](#complete-inventory)
- [Inputs, correctness and timing boundaries](#inputs-correctness-and-timing-boundaries)
- [Streaming operator examples](#streaming-operator-examples)
- [ASOF settlement measurements](#asof-settlement-measurements)
- [Join materialization measurements](#join-materialization-measurements)
- [Revision comparisons and regression gate](#revision-comparisons-and-regression-gate)
- [Standalone paired measurements](#standalone-paired-measurements)
- [Reports and failure behavior](#reports-and-failure-behavior)
- [Local reproduction](#local-reproduction)
- [Performance-plan diagnostics](#performance-plan-diagnostics)

## Feature measurement scope

The complete inventory below belongs to scheduled or explicitly requested broad
measurements. Feature development follows
[AGENTS.md verification](../AGENTS.md#verification): select at most five
representative cases tied to the changed code, necessary controls, or known
regressions. Choose the smallest relevant inputs rather than a matrix of every
size, type, mode, and checkpoint state.

Record the case rationale, comparison revisions, fixed sample counts, and total
wall-clock budget before execution. The default is ten minutes across the
selected cases, including setup, restore, preflight, warmup, oracle checks, and
process lifecycle costs; estimate builds separately. Reuse the existing harness
and retain its correctness, resource, statistics, and provenance requirements.
When the budget or environment prevents a valid conclusion, settle owned
processes and report the limits. Do not automatically enlarge the test set,
increase samples, or repeat attempts to obtain a favorable result.

Count each workload as a case; an atomic acceptance set is not one selection.
For example, `plan_end_to_end` defines a six-case acceptance set. A focused
subset reports per-case results; the set's aggregate and acceptance verdict
apply to complete-set measurements. Do not automatically complete the set to
satisfy its aggregate contract; complete-set measurements retain their scheduled
or explicitly requested broad scope.

An inconclusive comparison cannot establish a gain, equivalence, or a confirmed
regression. Use the maintained regression rule rather than requiring every
interval to exclude the threshold. This scope policy does not change the
scheduled inventory or its verdict calculation.

## Complete inventory

The catalog is executable: `python -m scripts.benchmark_suite catalog` emits
the same 22 shards consumed by the scheduled and manual suite. The Python
matrix includes overhead, small, standard and nightly scales.
The separate engine and warm-state matrices still run every decade
through 10M rows. Dynamic pytest, Criterion and Vitest inventories preserve
benchmark cases without a second hand-written case list.

| Family          | Dimensions                                          | Cases per dimension                                       |
|-----------------|-----------------------------------------------------|-----------------------------------------------------------|
| Python          | overhead 1k, small 10k, standard 100k, nightly 1M   | All collected non-lifecycle pytest benchmarks             |
| Engines         | 10, 100, 1k, 10k, 100k, 1M, 10M rows                | 57 through 10k, 65 at 100k/1M, 51 at 10M                  |
| Warm streaming  | 10, 100, 1k, 10k, 100k, 1M, 10M history; append 64  | SMA(20), SMA(5) minus SMA(20)                             |
| Warm append     | History 1M; append 1, 4, 16, 64, 640, 6,400, 64,000 | Both indicators; append 64 shared with history matrix     |
| Rust            | Every `[[bench]]` target in the core crate          | Core, allocation, state/window, Join/ASOF, SQL/DataFusion |
| Studio/frontend | Python HTTP benchmarks and Vitest benchmark files   | All collected benchmark cases                             |
| Lifecycle       | Isolated checkpoint/recovery benchmark              | Existing minimum-20-round evidence validation             |

The Python shard includes nine streaming operator examples at every Python
scale and four selected `Program.execute` cases. The Rust `stream_union` target
measures native Union forwarding. These cases extend the inventory without
adding a new shard or changing the scheduled 06:00 and 18:00 runs.

There are 409 engine cases and 26 warm cases, in addition to dynamically
discovered cases. Warm cases use one entity to support one-row appends.
Compare measurements only when entity count, history depth, append size,
and timing boundaries match.

The rust shard also runs the informational connector decode comparison
(`cargo test -p calc-flow-connectors --features kafka --lib perf:: --release
-- --ignored --nocapture`), which prints the protobuf and JSON-lines decode
throughput side by side in the step log and uploads
`decode-throughput/run.log` with the shard's measured results. It carries
no regression verdict.

| Backend          | Projection  | Filter      | Group by    | Join        | SMA(20) | Dual SMA | Interval Join |
|------------------|-------------|-------------|-------------|-------------|---------|----------|---------------|
| Calc Flow SQL    | Yes         | Yes         | Yes         | Yes         | Yes     | Yes      | Through 1M    |
| Raw DataFusion   | Yes         | Yes         | Yes         | Yes         | Yes     | Yes      | Through 1M    |
| Polars 1T / 32T  | Yes         | Yes         | Yes         | Yes         | Yes     | Yes      | Through 1M    |
| Native streaming | Yes         | Yes         | Yes         | Yes         | Yes     | Yes      | Through 1M    |
| TA-Lib           | Unsupported | Unsupported | Unsupported | Unsupported | Yes     | Yes      | Unsupported   |
| Finance-Python   | Unsupported | Unsupported | Unsupported | Unsupported | Yes     | Yes      | Unsupported   |

| Backend          | Average     | Argmax 64   | Argmax 256  | Unique 64   | CS mean     | Window sum  | ASOF join   |
|------------------|-------------|-------------|-------------|-------------|-------------|-------------|-------------|
| Native streaming | Yes         | Yes         | Yes         | Yes         | Yes         | Yes         | Yes         |
| Finance-Python   | Yes         | Yes         | Yes         | Yes         | Yes         | Unsupported | Unsupported |
| Polars 1T / 32T  | Unsupported | Unsupported | Unsupported | Unsupported | Unsupported | Unsupported | Yes         |
| Other libraries  | Unsupported | Unsupported | Unsupported | Unsupported | Unsupported | Unsupported | Unsupported |

Unsupported operations are explicit cells, not silent dependency skips.
Native streaming measures `join` through the bounded temporal join with the
dimension side seeded at the stream origin and its watermark sealing the quote
time range. Join, ASOF join and window sum all run through 10M rows. Static Join
readiness includes loading the dimension and observing its sealing watermark
before the timer starts or any quote is fed. Its limit permits the dimension
plus one input batch, and the sample rejects retained or evicted quote rows.
ASOF feeds one interleaved batch pair at a time and awaits delivery of every
left row in that pair before feeding the next. Window sum uses ten-second
tumbling windows. These cases retain bounded in-flight state
rather than both complete input streams.
Missing DataFusion, Polars, TA-Lib, or Finance-Python fails its shard. DataFusion Python 54 matches
the core's DataFusion major; the shared requirements file pins all Python
build/benchmark/Studio dependencies with hashes.

Finance-Python 0.9.10 runs from commit
`3e33d3e70c3458b4c6dcf76b88df6148229b402c` in a separate Python 3.9
environment. The scheduled engine shards install that commit and pin its build
and runtime dependencies. Its worker checks each timed result against the
untimed warm output with a SHA-256 digest; the parent checks that warm output
against the independent Arrow oracle. The additional native and Finance-Python
rolling and cross-section cases run at every tier through 10M rows. `window_sum`
and `asof_join` have no equivalent Finance-Python operator in this comparison.

## Inputs, correctness and timing boundaries

Engine comparisons use identical Arrow input bytes within Python 3.13; the
isolated Python 3.9 Finance-Python worker reconstructs the same deterministic
values in pandas. All cases use deterministic entity and
timestamp ordering, up to 64 entities, and 64,000-row input batches. Prices are
bounded exact eighths: `100 + sequence % 257 / 8 + sequence % entities / 8`.
This controls decimal accumulation drift in long rolling performance runs;
it is not a claim of numerical accuracy on arbitrary decimal sequences.
Decimal numerical regression fixtures are checked separately from the
performance workload.

`interval_join` uses exactly 64 keys and inclusive bounds of five seconds
before and after each left event. Both inputs contain the same Arrow rows and
project `sequence`, `right_sequence`, and the price product. The oracle checks
all pair identities and values, including boundary equality and duplicate-key
multiplicity. Native execution retains both sides and exercises watermark
eviction; an on-time out-of-order fixture covers arrival order separately.
SQL/DataFusion expand input across eleven literal integer-second offsets and
join the reference once, keeping one hash build within the runtime memory
budget. Polars uses eleven equivalent offset equality probes for this
one-tick-per-key grid. Their case identity records
`integer-second-offset-equality-v1`; these references do not measure arbitrary
non-grid interval SQL. Every interval backend has an explicit 1M-input-row
cap: the 10M fixture would emit roughly 110M rows. Its native timing scope is
`ready-enqueue-to-arrow/retained-interval-v1`.

Join, interval Join, ASOF, and projection also have native throughput cases
with 1,024-row input batches and exact-cursor event-log sources. Their scope is
`ready-enqueue-to-arrow/exact-cursor-batch-1024-v1`. The 64,000-row cases remain.
The interval cap applies to both batch sizes.

Checkpoint/recovery variants run at 100k and 1M input rows with both batch
sizes. They use a declared 100 ms checkpoint interval and a distinct
`checkpoint-duration` workload. After a nonterminal input prefix reaches the
sink, feeding stops, a declared 100 ms delay runs, and the adapter awaits a
durable epoch acknowledgement. It cancels before additional input, restarts
the same graph and bindings from the managed manifest, and combines the
accepted prefix with the resumed suffix without deduplication. Ordinary sink
delivery remains at-least-once. Each source restores the exact next data
position from a stable cursor and legally replays its equal watermark.

The lifecycle scope
`ready-enqueue-checkpoint-100ms-ack-recover-to-arrow-v1` includes the delay,
checkpoint acknowledgement, cancellation, manifest reading, plan recompilation
and runner restart. Initial input construction, compilation and readiness
precede timing; final EOF/terminal checkpoint/cleanup and validation follow it.
Reports show lifecycle variants in a separate table; their duration is not a
throughput-kernel measurement. Raw evidence must attest batch rows, interval,
source mode and binding IDs, replay mode, workload, scope, an acknowledged
nonterminal epoch, and successful recovery. Merely configuring a timer fails
validation. New variants remain `new-coverage` until a matching baseline
catalog declares the same dimensions and scope.
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
| Polars                 | Streaming-engine lazy-plan collection and Arrow output                                     | Arrow input conversion, lazy expression construction        |
| TA-Lib                 | Per-entity contiguous copies, SMA calls, composition, Arrow output                         | Input construction and validation                           |
| Finance-Python         | Public operator `transform` over a prepared pandas frame and NumPy extraction              | Frame construction, worker startup, warm-up, validation     |
| Ready native streaming | Input enqueue, sources/tasks/channels, rolling, watermarks, sink and combined Arrow output | Plans, input events, runner startup/readiness, EOF/shutdown |
| Warm native streaming  | Preconstructed data enqueue, live source/task/channel, rolling/finalization, sink to Arrow | Compilation, runner start, historical preload, validation   |

SQL/raw-DataFusion target partitions and Tokio workers are fixed to 32.
Polars references run in separate fresh processes with `POLARS_MAX_THREADS=32`
or `1`, reported as `Polars (32T)` and `Polars (1T)`. Measurement and artifact
validation check the actual Polars pool size against each case.
BLAS helper pools use one thread. The Python benchmark fixtures retain
their own query configuration and timing boundaries. Cross-library native
streaming uses an already-started runner with **empty rolling state**. Each
invocation compiles a fresh single-use plan, awaits runner startup and the
source's first poll through the startup data gate, then starts the timer before
enqueueing preconstructed data and watermarks. It stops after every expected
row reaches the sink and the Arrow tables are combined. EOF, job completion
checks and cancellation/cleanup happen afterward, outside timing; unexpected
extra output during completion still fails the sample. No dummy data or history
is preloaded for ordinary stream cases. Static Join preloads its dimension and
waits for the Join's accepted right watermark before timing; the timed interval
starts with the first quote enqueue. The left-state assertion happens after
timing. Persistent warm append remains a separate workload.

Cross-library columns are application-boundary references, not interchangeable
kernel measurements. The native column is labeled `Native stream (ready)`.
Report contract v3 validates the ready-runner timing scope and complete
sample statistics. Both revisions must be measured with the same scope;
do not subtract a separately measured startup time from another report.
The current native stream scope is
`ready-enqueue-to-arrow/bounded-feeds-v6`: ASOF lockstep waits use
sink delivery events, with no `job.status()` polling in the timed ASOF path.
Delivery of all accepted left rows proves finality for this workload, but
does not wait for the operator's subsequent status update. Static Join polls
right-side progress only during dimension setup, outside timing. Its post-timing
status assertion rejects any quote retention or eviction. The native static
Join boundary excludes dimension admission; reference-library measurements
retain their own documented preparation boundaries. Removing this setup cost
is a scope change, not evidence of a Join kernel speedup.
A baseline declaring a different stream scope makes native stream cases
`new-coverage`; SQL and warm-append comparisons retain their existing gates.
No performance improvement is inferred across the scope change.
These settings describe target/pool sizes, not measured CPU utilization;
TA-Lib calls remain sequential per-series operations.

## Streaming operator examples

`benchmarks/test_stream_operators.py` exercises Expression, SQL, Rolling,
CrossSection, WindowAggregate, StreamJoin, and StreamAsofJoin through fresh
owned `Program.stream` jobs. Rolling has one incremental-state case and two
scan cases with 64- and 256-row frames; each scan case requests argmax,
argmin, rank, quantile, distinct count, and linear decay. CrossSection
requests mean, residual, and top/bottom quantile masks. The native-only Union
operator has a two-input forwarding case in the Rust `stream_union` Criterion
target.

The Python timing scope includes compilation, runner startup, prepared Arrow
batch delivery, watermarks, output conversion, and job completion. Input
construction, batch splitting, and output validation occur outside the timed
call. Each case uses 16 entities and 640-row batches, with 10-tick aggregate
windows. The input cap is 20,000 rows, giving 960, 9,920, and 20,000 rows at
the overhead, small, and standard scales. Join and ASOF receive the same
prepared table on two distinct bindings. The Python Join and ASOF cases cap
each input at 3,200 and 1,920 rows; the engine matrix and dedicated Rust
targets measure larger join workloads. These measurements are
informational and have a different scope from the ready-runner engine matrix.

`benchmarks/test_program_engine.py` adds four separately identified cases:
SQL and streaming `Program.execute` over the same projection or cumulative
aggregate declaration. The timer starts after engine selection and input
construction and ends at Arrow output. Streaming includes its owned job and
returns one full aggregate snapshot per 640-row batch. Inputs cap at 20,000
rows. Existing compiled-plan and direct `Program.stream` case identities and
timing boundaries remain unchanged.

## ASOF settlement measurements

`stream_asof_perf` is an independent Rust target for bounded backward ASOF.
It measures `operator-watermark-settlement`: one large watermark advance
settles preloaded pending left rows against retained right history. Input
construction/admission, optional operator snapshot restoration, checkpoint
capture, and the full row oracle remain outside performance samples.
The restored workload uses an in-memory operator snapshot; it does not measure
managed checkpoint publication, source replay, or job restart.

The target and `scripts/benchmark_suite/asof.py` share this exact inventory.
`pending` counts left rows to finalize; `retained` counts right payload rows,
not total charged state. Total state also includes pending left and any
identity-only entries. Every workload retains its right rows after settlement.

| Case                 | Pending left | Retained right | Keys               | Restored |
|----------------------|--------------|----------------|--------------------|----------|
| `balanced_512`       | 512          | 512            | 32, balanced       | No       |
| `balanced_2048`      | 2,048        | 2,048          | 32, balanced       | No       |
| `balanced_8192`      | 8,192        | 8,192          | 32, balanced       | No       |
| `fixed128_right512`  | 128          | 512            | 32, balanced       | No       |
| `fixed128_right2048` | 128          | 2,048          | 32, balanced       | No       |
| `fixed128_right8192` | 128          | 8,192          | 32, balanced       | No       |
| `skew_8192`          | 8,192        | 8,192          | About 90% on key 0 | No       |
| `restored_skew_8192` | 8,192        | 8,192          | About 90% on key 0 | Yes      |

Output chunks adapt to the context's default 10,000-row, 64 MiB edge budget
and available workspace. The oracle requires positive, bounded chunk sizes
covering every pending row; historical 128-row chunks remain valid. Left time
starts at 1,000,000 microseconds, right time is 999,999, both watermarks advance
to 2,000,000, and tolerance is 10,000,000 microseconds. The state limits are
100,000 rows and 512 MiB. These choices exercise stable right history; they do
not cover eviction-heavy settlement or imply that arbitrary payloads fit the
same chunk size. See [ASOF state and workspace](asof-join-guide.md#bounded-state-and-workspace).

Run from the repository root:

```bash
cargo bench --locked -p calc-flow --bench stream_asof_perf -- --output target/asof.json
cargo bench --locked -p calc-flow --bench stream_asof_perf -- --check --output target/asof-check.json
```

Normal mode runs the strict-frontier/cancellation/restore check, then one full
row oracle and 20 samples for each of the eight cases. `--check` and `--test`
run the same correctness checks with empty `samples` arrays; their oracle
diagnostics are not timing evidence. `--output` writes JSON and creates parent
directories; without it the report is printed to stdout.
The untimed cancellation check seeds 300 more left rows than the default
output row budget, accepts one prefix, cancels the next emission, and restores
the remaining rows without gaps or duplicates.

The report schema is `calc-flow.asof-finalization.v1`. Raw observations retain
elapsed seconds, output rows, chunk row counts and cumulative emission times,
maximum logical chunk bytes, before/after status and checkpoint sizes, and
untimed admission/restore/capture durations. Allocation totals and peaks count
only the measured thread; they exclude allocations on spawned output workers.
Process RSS is sampled separately at 1 ms intervals
through Linux `/proc` and may miss shorter peaks; `rss_available=false`
means RSS is unavailable, not zero memory use.

The Rust adapter saves each block's `stream_asof_perf/asof.json`. Its loader
requires all eight unique cases, exact configurations, successful full-row
oracles, at least 20 samples per case, valid times/counts, bounded chunk
coverage, and the expected final status. It retains all diagnostic observations
in normalized metadata. Oracle-only reports intentionally fail this sampling
contract.

Standalone samples and the Rust shard's ABBA blocks do not constitute a
two-revision paired result. A target absent from the baseline is candidate-only
`new-coverage`; removing a baseline target fails. An ASOF optimization claim
requires separate comparable builds and per-case interleaved observations under
the [paired comparison contract](#revision-comparisons-and-regression-gate).
Keep product refs, benchmark source, compiled dependencies, binary hashes,
machine identity, and any separately sourced comparison harness with that
evidence. Do not transfer a measured verdict to a later source or build.

`stream_asof_e2e` measures `operator-admission-settlement` separately from the
settlement-only target. Each case admits 100,000 rows per side with 64 keys,
settles the output, and validates every matched row outside timing. The four
cases are `admit_settle_100k` (64,000-row batches), `eviction_ticks` (1,024-row
batches with a watermark after each), `out_of_order_within_watermark` (reversed
rows within each admitted batch), and `composite_key` (two key columns). It
reports elapsed seconds and measured-thread allocation totals, peaks, and
counts for each invocation. The allocation counters exclude output work on
`spawn_blocking` threads, so they are not whole-operator memory figures. The
timing boundary includes that output work but excludes source, sink, checkpoint
publication, and Python adapter time.

```bash
cargo bench --locked -p calc-flow --bench stream_asof_e2e -- --output ../../target/asof-e2e.json
cargo bench --locked -p calc-flow --bench stream_asof_e2e -- --check --output ../../target/asof-e2e-check.json
```

Normal mode records one correctness oracle and 20 samples per case. Check mode
records only the oracle. The Rust benchmark adapter requires all four cases and
retains each timing and allocation observation under the
`calc-flow.asof-e2e.v1` report contract.

## Join materialization measurements

`stream_join_materialization` is an independent Rust benchmark target.
Its `operator-bounded-edge` scope includes the incoming batch's
`process_data`, output materialization, a real bounded edge, and a draining
sink. The slow sink waits after each received chunk, including the final one.
Input construction, left-state preload, runtime startup and full row-oracle
validation are outside performance samples. This is an operator/edge boundary,
not end-to-end managed-job startup or checkpoint recovery.

All four maintained cases use 10 keys and 1,000 incoming right rows. The
preloaded left side has `10 * fan` rows; each incoming row matches `fan`
left rows. Payload width applies to each side's UTF-8 payload. The default
edge budget is 10,000 rows and 64 MiB of logical bytes.

| Case               | Payload per side | Fan-out | Sink delay per chunk | Output rows | Output chunks |
|--------------------|------------------|---------|----------------------|-------------|---------------|
| `narrow_f100_fast` | 128 B            | 100     | 0 ms                 | 100,000     | 10            |
| `wide_f10_fast`    | 1 KiB            | 10      | 0 ms                 | 10,000      | 1             |
| `wide_f100_fast`   | 1 KiB            | 100     | 0 ms                 | 100,000     | 10            |
| `wide_f100_slow`   | 1 KiB            | 100     | 10 ms                | 100,000     | 10            |

Run from the repository root:

```bash
cargo bench --locked -p calc-flow --bench stream_join_materialization -- --check
cargo bench --locked -p calc-flow --bench stream_join_materialization -- --output target/join-materialization.json
```

Normal standalone mode runs the pre-cancelled-input check, then one full row
oracle and 20 samples per case. `--check` and `--test` run correctness checks
with empty `samples` arrays: oracle diagnostics are not performance evidence.
`--output` writes JSON and creates parent directories; otherwise the report
goes to stdout. These default samples are independent collection, not an
interleaved comparison between versions.

The `calc-flow.join-materialization.v1` report retains sample times, input
and output rows, chunk counts, maximum logical chunk bytes, queue high-water
bytes, blocked sends and blocked duration. Allocation totals, counts and
active peaks cover the measured execution thread. Process RSS is separate:
Linux `/proc` sampling runs about every 1 ms and includes a first-emit
observation. It can miss shorter peaks; `rss_available=false` means the
measurement is unavailable. A 64 MiB logical edge budget does not cap total
RSS or sink-held output. The [Join memory boundary](streaming-guide.md#join-output-materialization-and-recovery)
also includes retained state, match descriptors, equality-probe scratch and
nested/dictionary preflight costs; these four flat-payload cases do not
establish performance for all payload types.

The Rust adapter saves each block's
`stream_join_materialization/materialization.json`.
`scripts/benchmark_suite/join_materialization.py` requires a nonempty inventory
with unique names, a successful full-row oracle with the configured output
count, at least 20 samples per case, positive finite sample times, and valid
allocation/RSS/queue/backpressure diagnostics. Configured `incoming` and `fan`
values must be positive integers; oracle and sample output-row counts must be
integers matching their product. Boolean and floating-point row counts are
rejected. It retains the configuration, oracle and all raw observations in
normalized metadata. Oracle-only reports do not satisfy this sampling contract.

The Rust shard measures a target absent from the baseline as candidate-only
`new-coverage`; removing a baseline target fails. Whole-suite ABBA deltas
remain informational. A separate version comparison must retain compatible
machine/dependency/workload identities, actual product refs, benchmark source,
sealed binaries and the comparison harness, using the existing
[two-round paired contract](#revision-comparisons-and-regression-gate).
Neither candidate-only samples nor oracle checks supply a paired verdict, and
a later head does not inherit measurements from an earlier build.

## Revision comparisons and regression gate

The benchmark workflow resolves immutable base/head commits before building
clean release wheels. Scheduled and manual runs default to the head's first
parent. A manual full baseline SHA can override that choice. There is no silent
fallback to a different successful run or debug wheel.

For every Calc Flow engine/warm case:

1. Install both sealed wheels into separate import directories on one runner.
2. Engine workers use the common candidate harness for both sealed wheels;
   warm workers load adapters from the source checkout matching their wheel.
   Verify loaded native hashes, dependencies, machine/thread identities and
   workload dimensions; warm up outside timing.
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
unchanged; the suite does not retry measurements to select a passing timing result.
External libraries are measured references, never fake historical baselines.
The pytest/Criterion/Vitest suites run ABBA whole-suite
blocks. Their deltas remain informational because those blocks are not
per-call paired observations. Allocation counters have a separate unit-correct
table and are not mislabeled as milliseconds.

Cases the baseline catalog never declared — a newly added engine/warm case or
suite-block benchmark — are measured on the candidate build alone and
reported with the `new-coverage` verdict instead of the paired gate; cases
the baseline declared keep the full interleaved gate. The baseline's case ids
are resolved from the baseline source's declarative catalog forms, and a
baseline file outside those forms fails closed to full gating.

Python suite blocks execute each checkout's own benchmark tests with the
current harness's collector and shared `benchmarks/support.py`, including
when the baseline checkout has an older recorder. Reports retain benchmark
source and shared-support hashes; the support also contributes to the harness
hash. Ordinary Python cases record raw machine, dependency, and workload
identities with SHA-256 fingerprints. Workload identity includes scenario,
timing scope, backend, scale, dimensions, row counts, and seed; array cases
retain their more specific contract-v2 identities.

The frontend runner records the actual Node process's hardware and runtime
identity, including CPU models/count, memory, architecture, Node/V8 versions,
and thread-related settings. Each workload binds the case/group and hashes
of the benchmark sources, Vite config, runner, and identity collector.
Dependency identity uses `frontend-npm-lock-v1`: only the top-level and root
package version fields are omitted from the comparison lock. The unmodified
lock hash and both version values remain in provenance. Missing, malformed,
corrupt, or incompatible comparison identities produce an error without a
timing classification, including when both sides lack the same fingerprint.
Comparable ABBA suite blocks remain informational.

The Rust suite retains the full `Cargo.lock` hash and aggregate compiled
dependency identity for provenance. Each case's comparison fingerprint covers
only its own benchmark target's compiled registry packages, their lockfile
checksums, enabled features, target kinds, profiles, and Rust/Cargo versions.
Adding a target does not invalidate comparisons for unchanged shared targets.
The inventory comes from Cargo's
[compiler-artifact messages](https://doc.rust-lang.org/cargo/reference/external-tools.html#artifact-messages),
including cache hits. Unused optional connector dependencies can change without
invalidating a core-only comparison. Changes to compiled dependencies still
fail closed, as do incomplete build logs or unsupported dependency sources.
The core package's source revision remains bound to the release identity.

The `stream_join_perf` compaction and control cases are
`checkpoint/prepare_then_left_500_{compact,steady}_60k`. Both time asynchronous
checkpoint preparation followed by the 500-row handler; fixture construction,
capture and restore stay outside timing. The adapter compiles the same declared
lifecycle harness against both exact product revisions in owned build clones,
leaving the supplied checkouts unchanged. Provenance retains the original bench
digest, the effective measured harness digest, both product and harness revisions,
and compiled dependencies. The former pure-handler case IDs are retired.

Rust workload fingerprints are scoped per bench target: each case's
`workload_fingerprint` covers only its own `crates/calc-flow/benches/<target>.rs`
bytes, so editing one bench source removes timing classification from that
target's cases alone. A bench source change that only affects the harness
pipeline has one explicit, auditable path to a green comparison: declare it in
`benchmarks/rust-workload-migrations.json`
with the target name, the exact baseline and candidate source SHA-256 values,
a reason, and a reviewing reference. A declaration applies only when both
sides' observed source bytes match it exactly; the accepted cases then carry a
`workload_migration` marker naming the reference, both revisions' provenance
documents keep their real differing workload identities, and the applied
migrations are listed in the shard's JSON artifact. Undeclared or mismatched
workload changes still fail closed, now scoped to the changed target.
The common Join harness above also requires an exact migration declaration
when replacing an older source. Its revised timing boundary uses new case IDs
and is compiled on both products, so historical pure-handler timings are never
compared against checkpoint-preparation timings.

## Standalone paired measurements

The optional `python -m scripts.release_performance` command measures ordinary
Python cases and the Rust `core` and `stream_join_perf` targets outside the
Python package release workflow. It builds and installs sealed
baseline/candidate wheels separately and records the loaded
Python native hash and each Rust benchmark binary hash. The formal baseline
and candidate commits must differ; `scripts/release_baseline.py` selects the
baseline for a manual comparison.

The standalone Python collector runs the current candidate's benchmark
declarations against both sealed native builds at `overhead` scale. Rust runs each
revision's compiled cases with the compiled-dependency and target-scoped
workload identities described above. Both sides must have matching, nonempty,
duplicate-free inventories; this collector has no `new-coverage` exemption.

The `core` Criterion target uses `cargo rustc --profile bench` with 64-byte
loop alignment applied only to the bench target. This keeps its sub-nanosecond
plan getter check from changing when a version-only binary layout shift places
the loop across an instruction-cache line. The product library retains the
ordinary bench profile, and the other Rust targets keep their existing build
command. The baseline and candidate use the same alignment setting.

Each case receives two rounds of ten adjacent baseline/candidate invocation
pairs, alternating AB/BA. Every invocation starts a fresh isolated process.
Its observation is the median of its saved pytest or Criterion raw samples,
using the fixture's existing timing boundary. Process startup, builds,
warm-up, and correctness checks stay outside that boundary; JAX completion
remains inside its timed calls. Separate summary means or whole-suite ABBA
blocks cannot substitute for these invocation pairs.

All observations must match the expected sealed native/binary hash and have
compatible machine, dependency, and workload identities. Raw identity objects
must reproduce their fingerprints. Missing pairs, reused worker identities,
incorrect execution order, invalid samples, or failed correctness checks are
evidence errors and invalidate the comparison.

The collector applies the same two-round paired-median interval and +5%
verdict rules as the engine/warm gate. A timing-only `inconclusive` result does
not itself fail this gate, but is not proof of equivalence or improvement.
Invalid or incomparable evidence fails regardless of timing. The separate
stream lifecycle quantile, rolling-kernel, and allocation gates still apply.
`--allow-dependency-drift` records acknowledgement only; it does not permit
classification across incompatible dependencies or waive identity checks.
`scripts/verify_perf_gates.py` rejects independent pytest/Criterion summaries
as paired evidence.

## Reports and failure behavior

The final always-run job publishes all result rows, with dimensions, timing
scope, base/head P50, P95, rows/s, percentage change, diagnostic round minima,
paired round medians with confidence bounds, and verdict.
A second table places every supported engine implementation side by side. No top-N
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

The standalone collector writes `results.json` and `summary.md` for each suite.
Its merge command checks sealed release manifests, Rust build provenance,
inventories, and raw pairs, then writes the combined verdict. Each
Rust build records its binary SHA-256 in `binary-sha256.json`; suite reports
retain the same digest so the merge can check every case seal against its
build. Each case retains `pairs.json`,
per-invocation `observation.json`,
raw pytest/Criterion data, and a `failure.json` when an invocation fails.
Command records beside logs include arguments, working directory, thread
settings, exit code, and errors. Build records, dependency provenance, and
harness hashes remain available with collected samples when a later step
fails. The Python publishing workflow does not run this collector.

## Local reproduction

Use clean candidate and baseline checkouts. Run the current candidate harness
for both releases; it supplies the same workload to both engine revisions.
Generated files stay under `target/`, not in `python/calc_flow/`.
Before an engine shard, prepare the pinned Finance-Python 3.9 environment using
the commands in [benchmarks/README.md](../benchmarks/README.md).

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
FINANCE_PYTHON_PYTHON=target/finance-python-venv/bin/python \
  target/benchmark-venv/bin/python -m scripts.benchmark_suite run \
  --shard engines-1000 \
  --baseline target/releases/baseline/release.json \
  --candidate target/releases/candidate/release.json \
  --baseline-source target/base --output target/results/engines-1000
```

Run every emitted catalog shard to reproduce the complete benchmark gate. A single
shard's own `summary.md` is useful locally; the complete summarizer deliberately
fails when shards are missing. To update dependencies, regenerate and commit
`benchmarks/requirements.lock` and `benchmarks/finance-python-requirements.lock`
using the commands in their headers. The suite checks both locks for drift
before its adapter tests.

## Performance-plan diagnostics

`scripts/measure_performance_plan.py` supplements the complete suite with an
explicit inventory for filter coercion, full-window SQL and Native SMA,
continuing history, sparse appends, batch/entity/window sensitivity, and
small-request tails. It uses the same workload adapters and thread settings
for both release wheels. The separately named `filter_uint64_modulo` query
does not replace the original `filter` query. Native warm cases retain their
partial-window semantics and advancing cursors; they do not substitute for
the empty-state full-window comparison.

```bash
python -m scripts.measure_performance_plan --group all --list
python -m scripts.measure_performance_plan \
  --baseline-build target/performance/baseline-build.json \
  --candidate-build target/performance/candidate-build.json \
  --group core --root target/performance/core-results
```

The explicit `entity-parallel` group compares the predeclared warm
H64k/A64k/E64/B64k dual-SMA target and same-shaped single-SMA control using
two rounds of ten pairs. `entity-parallel-tail` selects exactly those same
workloads with three fresh process pairs and 1,000 appends per pair. Both
groups are additional to `all`; the original small-append `tail` inventory
remains intact. Use the matched serial-route release as the baseline for
independent entity-parallel evidence, with owner initialization on both sides.

```bash
python -m scripts.measure_performance_plan --group entity-parallel-tail --list
python -m scripts.measure_performance_plan \
  --baseline-build target/performance/entity-control-build.json \
  --candidate-build target/performance/candidate-build.json \
  --group entity-parallel-tail --root target/performance/entity-tail-results
```

This tail group collects 3,000 samples per revision and workload, with the
same complete Arrow timing boundary and advancing state as the median group.
It reports descriptive tail quantiles and first-sink latency; its existence
does not establish that the parallel route ran, that tails improved, or that
the target median gate passed. Retain separate path-use, lifecycle, skew and
memory evidence for the final candidate.

The build records identify the clean source commit/tree, release profile,
features, compiler and lockfile, and the wheel and extracted native paths and
SHA-256 hashes. The controller verifies the wheel/native relationship and
the module actually imported by each worker. Keep these records with the
original release build logs; a manually asserted clean flag is not build
provenance. Core-only workers inspect NumPy and PyArrow without importing
unmeasured external engines.

Every warmup and measured output is also compared directly between revisions
through temporary Arrow IPC files outside timing. Schema metadata, validity,
identities, payloads, row counts and special-value classifications must agree;
finite values retain the workload's `1e-10` tolerances. SQL output is aligned
by its existing unique key outside timing; Native delivered order is preserved
and checked directly. Files are removed after each pair to bound retained
comparison data.

Warm completion also checks the sink's total delivered rows and rejects any
extra table after EOF. Terminal global or rolling-metric overflow, unfinished
callback observations and inconsistent exclusive-stage sums invalidate the
evidence. Different-binary candidate SQL diagnostics must demonstrate the
expected COUNT/AVG physical rewrite or UInt64 filter predicate; a successful
query returning the right values through fallback is not path-use evidence.

Core and sensitivity cases use two fresh process pairs with ten alternating
AB/BA pairs per round and the suite's exact median confidence interval.
Tail cases use three fresh process pairs with 1,000 appends each, retaining
all samples and reporting P50/P95/P99 separately from median confidence
intervals. Tail quantiles remain descriptive; a median interval does not
establish a tail-latency improvement. Identical-wheel runs are explicitly
labeled harness self-checks. Missing cases and failed correctness remain
errors, and inconclusive timing is not a passed improvement gate.
Every worker response and comparison is journaled outside timing, and each
completed round is saved before the next begins. A later failure retains
earlier raw samples and any failed Arrow comparison files for diagnosis.
