# Materialized Arrow source: scoped validation

DAL-378 implements the materialized Table/RecordBatch portion of #440 on main
`658527f8354a92163362d51c9ec7de9f68552836`. It does not implement readers,
arbitrary C stream providers, persistent replay or a general connector bridge.

## Measurement plan (fixed before collection)

Compare the existing iterable adapter (baseline) with direct materialized input
(candidate) in the same sealed release wheel. This isolates the input path;
it is not a comparison between different Join implementations or historical
builds. Use the maintained `release_pairs.collect_case` and `evaluate_case`
contracts: two rounds of ten alternating AB/BA pairs per case, a fresh process
per invocation, complete output oracle and immutable input checks. Exactly two
cases, with no retries or expansion to a larger matrix:

- Projection: 512 int64 rows in 64 records of eight rows; verifies every doubled
  value and the complete output schema. This isolates small-batch bridge cost.
- Interval Join: 64 rows per side in eight records per side, one match per key,
  zero-width time bounds and disabled watermarks; verifies all 64 pair identities,
  values, multiplicities and the complete schema. This checks a stateful lifetime.

Each invocation includes fixture creation, graph compilation/preflight,
temporary managed checkpoint setup, all outputs, terminal job cleanup,
correctness checks and garbage collection. Controller wall time also includes
process startup, imports, release identity verification and exit. Record CPU
time and process peak RSS (both process-wide), inner lifecycle time, Arrow
allocator bytes before/after, native task count and Python source callback count.
Input allocation is intentionally inside the lifecycle. Logical edge budgets
are not asserted to bound RSS. Arrow allocated bytes must return to baseline.

Total collection budget: 600 seconds including all worker startup and cleanup;
stop on exhaustion or incompatible evidence. Build time is separate (estimated
3–10 minutes with compatible dependency cache, longer for a cold release build).
Use the existing paired-median +5% rule and report uncertainty. This focused
comparison does not establish a whole-suite performance verdict.

The worker is `benchmarks/arrow_source.py`. A controller calls it with `--case`,
`--side`, `--build`, and `--output`, then uses the controller's process wall time
as the observation's single `samples` value and assigns a unique `worker` ID.
The worker independently verifies the sealed wheel and native hash using
`benchmarks.native_release_identity.release_identity`.

## Functional evidence

Recorded separately from performance collection. The focused Python suite
covers public streams, sliced/multi-record/empty data, schema normalization and
input metadata immutability, row/byte splitting, default and explicit temporal
policies, bounded slow-consumer progress, early exit, metadata and binding reuse,
and foreign-buffer owner release on EOF, sink failure and cancellation.
An inline Rust test checks partial/unread record release, idempotent close,
metadata/schema preservation and rejection of a recovery cursor. Full regression
and coverage remain CI responsibilities.

## Recorded results

Measured code: `e0070e25960760c3aa9360b4c90af09178c8b429` (the later
validation-record commit changes only this document). The production Rust source
is unchanged from `fa24c9fbf0b05dae2086ea6c58db86680186c14d`; its native unit,
Clippy and rustdoc evidence is reused. The Python cleanup follow-up releases
captured input references and queued outputs even when a closed StreamResults
object is retained for job-status inspection. Two targeted owner reproductions
failed before that cleanup change; EOF, early exit and close-before-entry now
have regressions.

- Final Python source with its matching debug native: 128 passed in 3.12 s
  (26 Arrow-input tests plus 102 existing stream/runtime tests).
- Final sealed release wheel: the 26 Arrow-input tests passed in 0.85 s.
- Native owner/metadata/replay unit: 1 passed; 121 unrelated tests filtered out.
- Scoped Clippy, rustdoc with denied warnings, Cargo fmt, affected Python Ruff
  lint/format and `git diff --check`: passed.
- Cold release compilation took 16 m 14 s; final Python-only wheel repack reused
  the identical native artifact (Cargo reported 0.17 s). Collection took 68.10 s
  of the fixed 600 s budget, with all 80 fresh workers and no retries.

Both sides use the same sealed wheel. A = iterable, B = materialized. Process
wall time includes startup/imports and release verification; inner lifecycle
includes fixtures, compilation, temporary managed state, execution, oracle and
cleanup. The native input remains materialized throughout the run.

| Case       | Process p50 A → B (ms) | Peak RSS median A → B (MiB) | Python source calls A → B |
|------------|------------------------|-----------------------------|---------------------------|
| Projection | 845.31 → 842.77        | 149.26 → 148.40             | 65 → 0                    |
| Join       | 831.82 → 840.45        | 136.81 → 135.84             | 18 → 0                    |

The maintained paired-median rule reports **no-confirmed-regression** for both
cases. Projection round intervals are −2.55%…+1.20% and −2.05%…+1.89%; Join
intervals are −4.00%…+1.34% and −2.61%…+4.54% (coverage at least 95%). These
observations do not establish an end-to-end speedup or lower memory peak.
Maximum observed process RSS was 151.46 → 152.36 MiB for projection and
140.89 → 140.30 MiB for Join. Inner lifecycle medians were 42.29 → 31.62 ms
and 39.25 → 35.36 ms, respectively; these are descriptive measurements rather
than a separately evaluated speedup gate. CPU-time medians were 4.253 → 4.236 s
and 4.246 → 4.256 s (summed across process threads).

Every worker passed the full value/schema/multiplicity oracle and input
immutability check, reported terminal task_count=0, and returned Arrow allocator
bytes from 0 to 0. Measurements ran on a shared Linux host, Intel Core
i9-13900HX, 32 logical CPUs, CPython 3.13.9 and PyArrow 24.0.0. RSS is a
process-wide high-water observation, not source-owned memory accounting.

Native SHA-256:
`bb3e367618e3249ffcc5dc6907379a6b038191317d44f9d95c883676a8e5d686`.
Wheel SHA-256:
`b771476e83c00b3c1f6d96bb1f9e654c75ceecdd90ce2d8c099615e8c4ac2bfd`.
The DAL-378 result comment attaches raw pairs/provenance and the focused
collector, which uses the maintained pairing and evaluation functions.

## Executed checks

The Rust commands used the managed venv interpreter via `PYO3_PYTHON`; the
Rust test also set the interpreter's shared-library directory in LD_LIBRARY_PATH.
No full-workspace regression, full coverage or CI wait was run locally.

```bash
PYTHONPATH=target/owner-test-site JAX_PLATFORMS=cpu .venv/bin/python -m pytest \
  python/tests/test_arrow_stream_inputs.py python/tests/test_stream_results.py \
  python/tests/test_stream_entrypoints.py python/tests/test_static_stream_inputs.py \
  python/tests/test_continuous_runtime.py -q
PYTHONPATH=target/arrow-source-release-final/site JAX_PLATFORMS=cpu \
  .venv/bin/python -m pytest python/tests/test_arrow_stream_inputs.py -q
cargo test -p calc-flow-python --lib arrow_source::tests -- --test-threads=1
cargo clippy -p calc-flow-python --lib -- -D warnings
cargo fmt --all --check
RUSTDOCFLAGS="-D warnings" cargo doc -p calc-flow-python --no-deps
.venv/bin/ruff check python/calc_flow/runtime.py python/calc_flow/_stream_inputs.py \
  python/calc_flow/stream.py python/calc_flow/_native.pyi \
  python/tests/test_arrow_stream_inputs.py benchmarks/arrow_source.py
.venv/bin/ruff format --check python/calc_flow/runtime.py \
  python/calc_flow/_stream_inputs.py python/calc_flow/stream.py \
  python/calc_flow/_native.pyi python/tests/test_arrow_stream_inputs.py \
  benchmarks/arrow_source.py
git diff --check
.venv/bin/python scripts/profile_warm_stream.py build --source . --target target \
  --output target/arrow-source-release-final
PYTHONPATH=. .venv/bin/python target/arrow-source-collect.py
```

The source overlay for the first check copied the current Python package to
`target/owner-test-site` beside the native module built from the unchanged Rust
source. The release check used the verified final wheel without that overlay.

Remaining boundaries: materialized input storage and slices can pin backing
buffers beyond logical edge credits; record/budget boundaries affect per-batch
outputs such as cumulative SQL snapshots. A reusable explicit binding retains
its buffers until it is released. No persistent cursor replay, reader support,
generic C stream bridge, full-suite performance or whole-workspace coverage
claim is made here. Full regression and the existing coverage gates remain CI.

## CI compatibility follow-up

The follow-up to `c0042e8` preserves the native binding protocol: `_native_*`
implementations need no `source` attribute. Only an optional `_ArrowSource`
descriptor selects the materialized path; other property exceptions propagate.
The existing connector completion/checkpoint/status unit now passes unchanged,
and a new unit protects propagation of a failing `source` property.

The Join regression owns its fixture and full schema/value/multiplicity oracle
inside the test module. A newly built debug wheel passed 37 focused tests from
an isolated directory where neither `benchmarks` nor `scripts` is importable:
28 Arrow-input cases and nine directly affected connector/start/close/cancel
cases. Native connector, property-error and Arrow owner/replay units each passed.

The four flagged functions were split without changing their workloads,
validation errors, binary search or temporal state updates. Lizard CCN is now
3/7 for `workload`/`collect` and 6/6 for `bounded_batch`/`validate_time`; all
extracted helpers are at most 6. The existing complexity ratchet, scoped Ruff,
Cargo fmt and scoped Clippy with denied warnings pass without rule changes.
One correctness-only invocation per case/side still reports projection source
callbacks 65 → 0 and Join 18 → 0, task_count=0 and Arrow allocator bytes 0 → 0.
The release performance measurements above retain their original code and
artifact identities; no new timing or RSS comparison was collected.

The Windows runtime-drop unraisable warning predates this PR. The Windows
Python job on the review baseline `658527f` already reports the same panic in
the same UDF test (1772 passed, five skipped, one warning):
[baseline Windows log](https://github.com/wegamekinglc/calc-flow/actions/runs/39107510462/job/117616482758).
This follow-up leaves that existing runtime-lifetime issue unchanged. Local
checks ran on Linux; Windows parity, full regression and coverage remain CI.
