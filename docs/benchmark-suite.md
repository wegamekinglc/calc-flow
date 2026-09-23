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
- [ASOF settlement measurements](#asof-settlement-measurements)
- [Join materialization measurements](#join-materialization-measurements)
- [Revision comparisons and regression gate](#revision-comparisons-and-regression-gate)
- [Release acceptance measurements](#release-acceptance-measurements)
- [Reports and failure behavior](#reports-and-failure-behavior)
- [Local reproduction](#local-reproduction)
- [Performance-plan diagnostics](#performance-plan-diagnostics)

## Complete inventory

The catalog is executable: `python -m scripts.benchmark_suite catalog` emits
the same 21 shards consumed by CI. The slow Python `nightly` scale is
excluded from this suite, including its daily/manual workflow calls; overhead,
small and standard remain. The separate engine and warm-state matrices still
run every decade through 10M rows. Dynamic pytest, Criterion and Vitest
inventories preserve benchmark cases without a second hand-written case list.

| Family          | Dimensions                                          | Cases per dimension                                       |
|-----------------|-----------------------------------------------------|-----------------------------------------------------------|
| Python          | overhead 1k, small 10k, standard 100k               | All collected non-lifecycle pytest benchmarks             |
| Engines         | 10, 100, 1k, 10k, 100k, 1M, 10M rows                | 26 supported engine/scenario combinations                 |
| Warm streaming  | 10, 100, 1k, 10k, 100k, 1M, 10M history; append 64  | SMA(20), SMA(5) minus SMA(20)                             |
| Warm append     | History 1M; append 1, 4, 16, 64, 640, 6,400, 64,000 | Both indicators; append 64 shared with history matrix     |
| Rust            | Every `[[bench]]` target in the core crate          | Core, allocation, state/window, Join/ASOF, SQL/DataFusion |
| Studio/frontend | Python HTTP benchmarks and Vitest benchmark files   | All collected benchmark cases                             |
| Lifecycle       | Isolated checkpoint/recovery benchmark              | Existing minimum-20-round evidence validation             |

There are 180 engine cases and 26 warm cases, in addition to dynamically
discovered cases. Warm cases use one entity to support one-row appends.
Compare measurements only when entity count, history depth, append size,
and timing boundaries match.

The rust shard also runs the informational connector decode comparison
(`cargo test -p calc-flow-connectors --features kafka --lib perf:: --release
-- --ignored --nocapture`), which prints the protobuf and JSON-lines decode
throughput side by side in the step log and uploads
`decode-throughput/run.log` with the shard's measured results. It carries
no regression verdict.

| Backend          | Projection  | Filter      | Group by    | Join        | SMA(20) | Dual SMA |
|------------------|-------------|-------------|-------------|-------------|---------|----------|
| Calc Flow SQL    | Yes         | Yes         | Yes         | Yes         | Yes     | Yes      |
| Raw DataFusion   | Yes         | Yes         | Yes         | Yes         | Yes     | Yes      |
| Polars           | Yes         | Yes         | Yes         | Yes         | Yes     | Yes      |
| Native streaming | Yes         | Yes         | Yes         | Yes         | Yes     | Yes      |
| TA-Lib           | Unsupported | Unsupported | Unsupported | Unsupported | Yes     | Yes      |

Unsupported operations are explicit cells, not silent dependency skips.
Native streaming measures `join` through the bounded temporal join with the
dimension side complete at the stream origin; its evidence stops at the
100,000-row tier. The 1M and 10M tiers stay unsupported in the catalog for
performance (user-directed pacing constraint, 2026-09-20): the join retains
one state row per matched input row for the whole run, so a sample needs
roughly 5 seconds at 1M and 200 seconds at 10M on the dev machine, which
would slow the whole suite's cadence. Missing
DataFusion, Polars or TA-Lib fails its shard. DataFusion Python 54 matches
the core's DataFusion major; the shared requirements file pins all Python
build/benchmark/Studio dependencies with hashes.

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
| Polars                 | Streaming-engine lazy-plan collection and Arrow output                                     | Arrow input conversion, lazy expression construction        |
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

| Case                 | Pending left | Retained right | Keys               | Restored | Output chunks |
|----------------------|--------------|----------------|--------------------|----------|---------------|
| `balanced_512`       | 512          | 512            | 32, balanced       | No       | 4             |
| `balanced_2048`      | 2,048        | 2,048          | 32, balanced       | No       | 16            |
| `balanced_8192`      | 8,192        | 8,192          | 32, balanced       | No       | 64            |
| `fixed128_right512`  | 128          | 512            | 32, balanced       | No       | 1             |
| `fixed128_right2048` | 128          | 2,048          | 32, balanced       | No       | 1             |
| `fixed128_right8192` | 128          | 8,192          | 32, balanced       | No       | 1             |
| `skew_8192`          | 8,192        | 8,192          | About 90% on key 0 | No       | 64            |
| `restored_skew_8192` | 8,192        | 8,192          | About 90% on key 0 | Yes      | 64            |

These fixtures produce 128-row chunks of 8 KiB logical output bytes. Left time
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

The report schema is `calc-flow.asof-finalization.v1`. Raw observations retain
elapsed seconds, output rows, chunk row counts and cumulative emission times,
maximum logical chunk bytes, before/after status and checkpoint sizes, and
untimed admission/restore/capture durations. Allocation totals count cumulative
bytes allocated during settlement; allocation peaks count peak active bytes
in the measured thread. Process RSS is sampled separately at 1 ms intervals
through Linux `/proc` and may miss shorter peaks; `rss_available=false`
means RSS is unavailable, not zero memory use.

The Rust adapter saves each block's `stream_asof_perf/asof.json`. Its loader
requires all eight unique cases, exact configurations, successful full-row
oracles, at least 20 samples per case, valid times/counts, 128-row chunk
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

Rust workload fingerprints are scoped per bench target: each case's
`workload_fingerprint` covers only its own `crates/calc-flow/benches/<target>.rs`
bytes, so editing one bench source removes timing classification from that
target's cases alone. A bench source change that only affects the harness
pipeline — not the measured workload — has one explicit, auditable path to a
green comparison: declare it in `benchmarks/rust-workload-migrations.json`
with the target name, the exact baseline and candidate source SHA-256 values,
a reason, and a reviewing reference. A declaration applies only when both
sides' observed source bytes match it exactly; the accepted cases then carry a
`workload_migration` marker naming the reference, both revisions' provenance
documents keep their real differing workload identities, and the applied
migrations are listed in the shard's JSON artifact. Undeclared or mismatched
workload changes still fail closed, now scoped to the changed target.

## Release acceptance measurements

The release workflow uses `python -m scripts.release_performance` for ordinary
Python cases and the Rust `core` and `stream_join_perf` targets. It builds and
installs sealed baseline/candidate wheels separately and records the loaded
Python native hash and each Rust benchmark binary hash. The formal baseline
and candidate commits must differ; baseline selection follows the
[release baseline contract](python-release.md#first-release-performance-baseline).

Python release collection runs the current candidate's benchmark declarations
against both sealed native builds at `overhead` scale. Rust runs each
revision's compiled cases with the compiled-dependency and target-scoped
workload identities described above. Both sides must have matching, nonempty,
duplicate-free inventories; this release path has no `new-coverage` exemption.

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
evidence errors and block acceptance.

The collector applies the same two-round paired-median interval and +5%
verdict rules as the engine/warm gate. A timing-only `inconclusive` result does
not itself fail this gate, but is not proof of equivalence or improvement.
Invalid or incomparable evidence fails regardless of timing. The separate
stream lifecycle quantile, rolling-kernel, and allocation gates still apply.
`--allow-dependency-drift` records acknowledgement only; it does not permit
classification across incompatible dependencies or waive release acceptance.
`scripts/verify_perf_gates.py` rejects independent pytest/Criterion summaries
as release pairing evidence. See the
[release command and retained evidence](python-release.md#performance-acceptance-and-failure-evidence)
for execution and failure inspection.

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

Release CI collects the Python, Rust core, and stream join suites in parallel.
Each suite writes `results.json` and `summary.md`; the acceptance job downloads
all three artifacts, checks their sealed release manifests, Rust build
provenance, inventories, and raw pairs, and writes the merged verdict. Each
Rust build records its binary SHA-256 in `binary-sha256.json`; suite reports
retain the same digest so the merge can check every case seal against its
build. Each case retains `pairs.json`,
per-invocation `observation.json`,
raw pytest/Criterion data, and a `failure.json` when an invocation fails.
Command records beside logs include arguments, working directory, thread
settings, exit code, and errors. Build records, dependency provenance, and
harness hashes remain available with collected samples when a later step
fails. Release CI's always-run summary and 30-day artifact also record
performance/security/soak outcomes and why a downstream step was skipped;
see [release failure evidence](python-release.md#performance-acceptance-and-failure-evidence).

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
