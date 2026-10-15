# Issue 436: ASOF columnar merge completion

Baseline: main `718ab62ff5ad562994a694a20102b89991fc6b2b`, including #435.
The task branch is `feature/dal342-asof-columnar-merge` (DAL-342).

## Implementation

General admission keeps the canonical `(time, key bytes, sequence)` permutation
and builds per-key physical positions from it. Duplicate proofs merge those
positions with resident identities. Right proofs include payload, ordered
identity-only, and general identity-only histories. Left proofs reserve their
chunk-iterator scratch and use the existing scalar proof if that optional
reservation is refused, preserving rejection priority.

Integer-sequence right installation borrows each reserved bucket once and
merges backwards into its existing column capacities. Each displaced resident
slot moves once. Parallel workers borrow the same immutable column/permutation
owner and batch references; they produce detached buckets without allocating
canonical row descriptor vectors. Canonical sequence ownership continues to
use its existing explicit compatibility boundary.

Ordered and normalized unordered left chunks share the monotonic finalization
kernel. Repeated candidates use a per-key latest Arrow payload route. Fixed
width values remain in their funded Arrow columns; variable values keep their
row references. General identity history advances through a borrowed tree
range, rather than seeking that tree for every left row. Dictionaries beyond
64 keys use funded route vectors when at least four selected rows amortize
each slot. Legacy and short runs keep their generic route; cursor funding or
eligibility refusal retains binary-search finalization.

Route and cursor size assertions accompany the workspace bounds. Existing
output source registration, logical charges, prefix ownership, output limits,
successful-emission commit boundaries, and checkpoint layouts remain in place.
No public schema, binding, or generated API type changes are required.

## Contract boundaries

- Arrow Dictionary payload types are rejected by the existing flat ASOF public
  schema validator. This change tests that rejection and does not add a new
  payload type contract. Variable Utf8 payloads use the existing SQL/legacy
  differential and checkpoint/recovery cases.
- Canonical string/composite sequence owner accounting still materializes its
  compatibility rows. Integer sequences use the columnar path for general
  keys and unordered input, including parallel installation.
- Identity-only history remains necessary for duplicate detection after a
  payload is retired. This change neither expires that history early nor
  changes frozen logical state fees.
- Full CI, cross-platform coverage, and the 90% Rust coverage gate belong to
  the existing CI. Formal performance acceptance remains #443.

## Resource observation protocol

Two diagnostic workloads will run through the existing
`caller_and_actual_worker_allocations_scale_with_distinct_keys` unit test:
1,024 and 64,000 rows, each with 64 keys. These cover inline and owned worker
construction after removing the ordered generic descriptor route. The
candidate's Rust 1.88.0 debug executable compares actual caller and chunk
allocations with its forced scalar reference. `/usr/bin/time -v` records the
entire reference-and-candidate test process's peak RSS, without a per-arm RSS
claim. One invocation, 60-second execution budget, compilation excluded.
The separate merge regression asserts zero allocations after column reservation.
No release timing or speed claim is planned; these observations cannot establish
the ASOF 40/30 ms targets.

## Resource observations

The selected invocation completed within the 60-second budget and passed its
allocation, thread, cleanup, and scalar-reference assertions. Counts and bytes
below are allocator `count_total` and `bytes_total`, not retained-state fees.
Inline chunk measurements overlap caller measurements; the large chunk is
measured on its actual owned worker.

| Rows   | Caller count ref/current | Caller bytes ref/current | Chunk count ref/current | Chunk bytes ref/current |
|--------|--------------------------|--------------------------|-------------------------|-------------------------|
| 1,024  | 104 / 159                | 231,276 / 165,552        | 27 / 26                 | 47,508 / 27,124         |
| 64,000 | 95 / 150                 | 11,094,538 / 7,723,724   | 27 / 26                 | 2,314,644 / 1,286,644   |

The new caller has 55 additional allocations in both cases, while cumulative
allocation bytes fall. Actual chunk allocations stay at 26 for both sizes.
The whole reference-and-candidate test process peaks at **92,392 KiB RSS**;
this is neither a per-arm RSS comparison nor production release evidence.
The separate reserved-merge regression passed its zero-allocation assertion
for all supported integer widths and canonical history.

The resource command was:

```bash
timeout 60s /usr/bin/time -v target/debug/deps/calc_flow-b7e69e6a365d09b0 \\
  --exact operator::asof::admission::identity_tests::columnar::caller_and_actual_worker_allocations_scale_with_distinct_keys \\
  --test-threads=1 --nocapture
```

## Verification

All Rust commands used pinned Rust 1.88.0, locked dependencies, two build jobs,
and serial tests. The final affected unit coverage is 69 tests: 67 passed in
the admission/matching selection, followed by two passing focused rechecks
of the descriptor-free boundary expectations. Product code did not change
between those runs; the final changes correct two test classifications of
the new general columnar route.

```bash
CARGO_BUILD_JOBS=2 cargo test --locked -p calc-flow --lib -- \\
  operator::asof::admission:: matching_aggregates_batch_key_and_shared_sequence_owners_by_run \\
  --test-threads=1 --quiet
# Final selection: 67 passed; the remaining two expectations were corrected below.
CARGO_BUILD_JOBS=2 cargo test --locked -p calc-flow --lib -- \\
  test_direct_left_dictionary_boundaries_and_fallback_match_legacy \\
  test_direct_left_uses_actual_restored_maximum --test-threads=1 --quiet
# 2 passed.
CARGO_BUILD_JOBS=2 cargo test --locked -p calc-flow \\
  --test stream_asof_join_properties --test stream_asof_join_boundaries \\
  -- --test-threads=1 --quiet
# 8 properties and 3 boundaries passed.
```

The initial broader 124-test selection supplied 55 passing, unchanged
right-run, output-plan, probe, and dominated-history checks. Its two failures
were in the admission/matching selection rechecked above. That run reported
122 passed and 2 failed; it is not represented as a fully passing invocation.
The retained passing checks include the reserved-merge allocation assertion,
identity-history cursor parity, cancellation, and retired-prefix preservation.

Formatting, whitespace, schema drift, and Lizard 1.23.0 checks passed. Changed
production functions in admission, merge, cursors, and output planning stay
within the existing complexity limit of 8. No full workspace test, coverage,
cross-platform, or paired release performance result is claimed.

The final `CARGO_BUILD_JOBS=2 cargo clippy --locked -p calc-flow --lib --tests -- -D warnings`
and `cargo fmt --all --check` both passed on the delivered product and test source.

## Windows checkpoint isolation follow-up

Product source remains at `3318a3d61d259d1c9064f105a0e43dcb56f1d475`.
The follow-up changes only the Windows test schedule, its Python harness,
the harness/workflow checks, and this evidence record.

The [failed Windows job](https://github.com/wegamekinglc/calc-flow/actions/runs/38786091579/job/116598420211)
ran the all-features library suite in parallel: 2,044 passed, four failed,
four ignored. Two ASOF operator fixtures expired their five-second barrier
ACK waits. The collector recovery fixture expired epoch 1 in `SourcesCut`;
the Join-to-Window recovery fixture expired epoch 2 and correctly reported
`RecoveryRequired` instead of claiming completion. These runtime tests,
the Join/Window implementations, and checkpoint coordinator were unchanged
by the product commit.

`SourcesCut` means all source ACKs have arrived and operator ACKs remain
outstanding (`checkpoint/coordinator.rs`, `next_advancement`). Operator ACKs
follow state staging (`operator_task.rs`, `capture_operator_checkpoint`),
including blocking-worker file `sync_all` in `state/local.rs`. Thus these
waits include durable local I/O, rather than only watermark processing.
The Windows workflow already isolates checkpoint soak tests for this reason.
The [exact base Windows job](https://github.com/wegamekinglc/calc-flow/actions/runs/38777032052/job/116578428023)
at `718ab62ff5ad562994a694a20102b89991fc6b2b` passed the same 2,052-test
selection (2,048 passed, four ignored), including all four affected tests.

This evidence points to Windows parallel checkpoint progress/I/O contention;
the old logs do not identify the stalled syscall or prove a specific host
cause. Linux did not reproduce the deadline failures. No semantic product
failure or confirmed flaky-test classification is claimed. The new Windows
CI run remains necessary to validate isolation.

Only the four exact test names move from the Windows parallel leg into one
serial all-features library invocation. The harness now accepts repeated
`--lib-skip` options without replacing an earlier filter. The original
five/ten-second deadlines, fixture code, and all assertions remain intact:
Idle suppression/reactivation, extreme watermarks and EOF, checkpoint cuts,
backpressure cancellation, released edge charges, and restored Join/Window
results are still exercised. Source/operator/sink tasks inside each job
retain their asynchronous concurrency. Other suites keep their schedules.

Validation on Linux, Rust 1.88.0, existing checkout and debug profile:

```bash
CARGO_BUILD_JOBS=2 cargo test --locked -p calc-flow --lib -- \
  test_asof_idle_never_forwards_with_or_without_pending_left \
  test_asof_minimum_watermark_has_no_underflow_and_maximum_requires_eof \
  test_asof_collector_backpressure_cancellation_restores_last_published_pending_cut \
  ac5_checkpoint_restore_preserves_the_join_window_result \
  --test-threads=1 --nocapture
# 4 passed; 2,111 filtered out.

CARGO_BUILD_JOBS=2 cargo test --locked -p calc-flow --lib --all-features -- \
  runtime::streaming::operator_task::tests::asof_tests::test_asof_idle_never_forwards_with_or_without_pending_left \
  runtime::streaming::operator_task::tests::asof_tests::test_asof_minimum_watermark_has_no_underflow_and_maximum_requires_eof \
  runtime::streaming::runner::tests::asof_tests::test_asof_collector_backpressure_cancellation_restores_last_published_pending_cut \
  runtime::streaming::runner::tests::ac5_checkpoint_restore_preserves_the_join_window_result \
  --exact --test-threads=1 --nocapture
# 4 passed; 2,123 filtered out. This verifies the full names and CI feature set.

python -m unittest \
  scripts.test_run_rust_tests.RustTestHarnessTests.test_lib_skip_appends_every_filter_to_the_calc_flow_leg_only \
  scripts.test_run_rust_tests.RustTestHarnessTests.test_lib_skip_is_ignored_by_compile_only_runs \
  scripts.test_run_rust_tests.RustTestHarnessTests.test_lib_skip_rejects_an_empty_filter \
  scripts.test_run_rust_tests.WindowsCheckpointSoakIsolationTests
# 8 passed. Checks preserve all skip filters, compile-only behavior,
# the exact four serial selections, and existing soak/exit-status handling.
```

Ruff lint/format, safe YAML parsing, whitespace, and schema-drift checks passed.
Lizard 1.23.0 reports complexity 8 for the changed `_run_cargo_tests` function.
Existing product correctness, Clippy, format, and resource evidence above
remain valid; no product source changed and none of those suites were rerun.
