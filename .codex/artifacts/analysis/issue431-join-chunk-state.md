# Issue 431: Join chunk state and batch selection

Baseline: main `da004efe6c14ac90249dd8c33144ff6c6a30c1d7`, including #430.
The task uses `feature/dal341-join-chunk-state`; no shared identity or ASOF
ordering contract changes.

## Implementation

Online state stores columnar metadata in 1,024-row chunks, with shared payload
sources and stable chunk/row coordinates. Expiration stores sorted incoming
runs and one heap head per run. Removing a row releases its source reference;
empty chunks retire without relocating other rows or repairing dense indices.
Logical row/byte gauges remain incremental, including complete expiration.

Native probing retains count/limit/fill and stores opposite references plus an
incoming identity range or incoming runs. Fragment reservations cover both
selection arrays; fragment backing dies before its reservation. SQL fallback
builds retained key columns with batch take/interleave, without row-batch views.

Output uses take/interleave for single/multiple sources, including legacy and
nested payloads. Encoded children are compacted in whole columns and view data
is detached. Kernel errors propagate; there is no per-row slice/concat fallback.

Chunk metadata and expiration preparation precede output. Abandoned preparation
releases new reservations; state publication still follows all successful
emissions. The original unfunded legacy admission route remains available only
when optional acceleration refuses legacy input. Recovered funded expiration
state keeps mandatory admission. V1 fees, input immutability, physical row IDs,
ordering and checkpoint layouts 1/2/3 are unchanged. `StoredRow` remains a
transient admission/historical-reader adapter so existing validation/error priority and wire
readers are preserved; it is absent from online retained storage.

## Verification scope

Select Join unit controls, SQL/property and state recovery integration tests,
scoped all-feature Clippy, formatting, complexity and schema/whitespace checks.
Full workspace regression, cross-platform and total coverage remain CI gates.

Allocation/RSS acquisition is limited to the maintained materialization
harness's four correctness cases, with a ten-minute collection budget, builds
separate. Development-profile allocation and RSS observations are diagnostics;
this work makes no release timing or R16 performance acceptance claim.

## Diagnostic acquisition plan

Candidate: `f05e3558b788283f04497c06ca15d2fe26f25300`, integrating main
`718ab62ff5ad562994a694a20102b89991fc6b2b`. Product implementation is
`bcd9ae3d`; `7d53cdf5` corrects the ownership test fixture, and `dd9bd12c`
limits a test/benchmark row adapter to its callers. Main's grouped Join probe
stop checks are preserved alongside the new selection allocation. The bounded-pool
follow-up `f3af620d` admits mandatory state/expiration metadata before optional
native-index growth, reserves actual source slots plus one replacement slot,
and writes V2 bases from borrowed row-source references.
The baseline above is recorded for provenance and is not measured in this
candidate-only diagnostic. No timing comparison is planned.

Use the maintained `stream_join_materialization` worker mode, one fresh process
per case. Each process first validates every output row, then records exactly
one sample with the row oracle outside the allocation measurement. All cases
use 10 keys and 1,000 incoming rows. The four cases exercise opposite-reference
fan-out, variable-width batch gathering, and bounded sink backpressure:

| Case              | Fan-out | Payload bytes/side | Sink delay/chunk |
| ----------------- | ------- | ------------------ | ---------------- |
| narrow_f100_fast   | 100     | 128                | 0 ms             |
| wide_f10_fast      | 10      | 1,024              | 0 ms             |
| wide_f100_fast     | 100     | 1,024              | 0 ms             |
| wide_f100_slow     | 100     | 1,024              | 10 ms            |

Fixed budget: 600 seconds across process startup, fixtures, full-row oracle,
one diagnostic sample and cleanup for all four cases; compilation is separate.
Use the development profile with debug information disabled, two build jobs and
Rust 1.88.0. Allocation counters cover the maintained harness's measured thread
and include operator plus sink/edge work; preloading and oracle verification
are excluded from the reported sample. RSS is sampled every millisecond after
preload and can miss shorter peaks. The environment is not a sealed release
benchmark and has no CPU affinity reservation.

## Verified result

Product source tested and measured: `f05e3558b788283f04497c06ca15d2fe26f25300`.
The results-only documentation commit does not change Rust source, dependencies,
or build configuration.

- Join module, diagnostic feature: 386 cases passed in the full run at
  `f3af620d`, with two failures and one explicitly ignored 4M-row experiment.
  Both failures were fixed in `f05e3558` and passed targeted reruns, yielding
  passing evidence for 388 Join cases. The lookup fixture now uses complete
  data preparation while preserving its three-lookup assertion. The grouped
  deadline waiter rechecks UTC after waking before returning, eliminating the
  observed early-timer panic. Other passing cases were not rerun for this fix.
- `cargo test --locked -p calc-flow --features benchmark-checkpoint-diagnostic
  --lib operator::join::benchmark_checkpoint::tests::test_dirty_replacement_fits_bounded_pool_and_preserves_owned_history
  -- --exact --test-threads=1`: passed with the original 10 MiB pool, full
  state/history oracle, alias preservation and final zero pool balance.
  This case also passed in the subsequent full Join run.
- Final targeted reruns used the same Cargo feature/lib options, filtering
  `test_probe_owner_cancel_and_deadline_stop_group_pollers_and_refund` and
  `test_native_probe_resolves_each_distinct_key_once_before_both_window_passes`:
  one case passed each. Cancellation/deadline checks cover both count and fill.
- `cargo test --locked -p calc-flow --test stream_join_properties
  --test stream_join_state --test stream_join_spec -- --test-threads=1`:
  20 passed (4 property, 10 state, 6 specification), on final product source.
- `cargo clippy --locked -p calc-flow --lib --tests --benches --all-features
  -- -D warnings`: passed on final product source.
- `cargo fmt --all --check`, `git diff --check`, and generated project-v3 schema
  no-drift check against integrated main: passed.
- Lizard 1.23.0: 349 changed/new Rust functions,
  maximum CCN 8, no findings above eight. The receipt hashes
  every changed Rust file. This is local evidence, not a remote Codacy verdict.

The full Join executable was built through Cargo with
`--features benchmark-checkpoint-diagnostic --lib`, then run serially with
`timeout 180s target/debug/deps/calc_flow-abab16b2e36df81b operator::join::
--test-threads=1`. Build settings for these commands were
`CARGO_BUILD_JOBS=2 CARGO_PROFILE_DEV_DEBUG=0 CARGO_PROFILE_TEST_DEBUG=0`.

## Allocation and RSS observations

The final maintained harness build used
`cargo bench --locked -p calc-flow --bench stream_join_materialization
--profile dev --no-run` (9.72 seconds after the final library build; builds are
outside the collection budget). Each worker received its case JSON through
`CALC_FLOW_JOIN_CONFIG`, followed by `sample` and `quit` on stdin. All four
full-row oracles and four single diagnostic samples completed in
2.14 seconds of the fixed 600-second budget.

| Case             | Thread allocations | Total allocated MiB | Peak allocated MiB | Peak RSS MiB |
| ---------------- | ------------------ | ------------------- | ------------------ | ------------ |
| narrow_f100_fast | 1,102              | 33.15               | 7.33               | 47.41        |
| wide_f10_fast    | 276                | 22.28               | 21.74              | 61.87        |
| wide_f100_fast   | 1,102              | 204.91              | 42.37              | 83.72        |
| wide_f100_slow   | 1,102              | 204.91              | 62.43              | 104.58       |

Host: Intel Core i9-13900HX, Linux WSL2; Rust 1.88.0. Binary SHA-256:
`3edbda5ccb22265dcee301ec53a0ccafa608256a06f756da1cf96bb0e73f59f0`.
Allocation values cover the measured thread and exclude the row oracle;
allocations on other worker threads are not included. RSS includes runtime and
allocator caches from the preceding oracle and samples the whole worker process.
It is not an isolated instantaneous high-water measurement. Slow-sink queue
ownership raises the observed peak as expected; no baseline or release timing
comparison is asserted.

Raw acquisition/provenance and changed-function receipts accompany the Multica
handoff as attachments. Full workspace, cross-platform and 90% coverage gates
remain CI work. The ignored 4M-row experiment and formal R16 performance targets
were not run or claimed here.
