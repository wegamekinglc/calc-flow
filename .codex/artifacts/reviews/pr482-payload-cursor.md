# PR #482 Review: payload-only Join cursors

**Author:** Cheng Li | **Branch:** `feature/join-cpu-pipeline` → `main`

**Reviewed source:** `90c06e09412ef225dc5941e6b18bf1ab1e668409`

**Base:** `c0cd587572e311faa21d9f3a0e2564e63f9ba1d5` | **Files:** 5

This focused review supplements the [remediation review](pr482-remediation.md).
The reviewer read the changed files and their affected callers in the
orchestrator's isolated worktree. Only this report was edited by the reviewer;
native verification and remote actions remain with the orchestrator.

## Summary

Join output gathering used the metadata-bearing row cursor even when an
admission adapter supported payload access without row metadata. The existing
payload-only regression test failed in Windows CI and reproduced locally.
The fix gives the cursor a payload operation and uses payload access in the
three related output-preflight paths.

The native row and payload cursors share the same segment-binding logic.
Metadata-bearing consumers retain their row operation, while output paths
preserve payload identity, ordering, projections and chunk-cost calculations.

## Build and Test Results

Evidence is under `target/pr482-ci-followup/` in the parent checkout.

- Historical Windows Rust result: **Failed**, with **2,311 passed, one failed
  and six ignored**, in 125.24 seconds (`windows-rust.log`). The failed test is
  `test_output_payload_access_keeps_order_without_reading_metadata_or_the_unused_side`.
- Existing regression reproduced locally before the fix: **zero passed,
  one failed**, in 1.27 seconds (`payload-red.log`). Its stack shows
  `gather_columns` → `AdmittedCursor::row` → the default `cursor_row` →
  `PayloadOnly::row`.
- New preflight regression before the implementation: **zero passed,
  one failed**, in 0.00 seconds after a 1m 06s build (`preflight-red.log`).
  It fails on the same prohibited metadata access.
- Final-source focused Rust tests: **Passed**, with **18 passed, zero failed
  and zero ignored**, in 0.78 seconds after a 2m 10s build
  (`payload-green.log`). These results are bound to `90c06e09` and include
  both regression tests, the affected materialization checks, cursor parity
  and the segment-lookup bound.
- Scoped Clippy: **Passed**, exit code 0 after 2m 18s (`clippy.log`), using
  `cargo clippy --locked -p calc-flow --lib --tests --features benchmark-internals -- -D warnings`.
- Formatting: **Passed**, `cargo fmt --all --check`, exit code 0.
- Python: **Not touched**; no Python suite or package rebuild requested.
- New failures: the reproduced payload-only contract violation above is fixed
  in the completed focused run. No unresolved new local test failures remain.
- Regressions: none identified by source review or the focused checks.

The local five-file complexity comparison reports no new or increased
violations above its threshold (`complexity.json`); this does not substitute
for remote Codacy acceptance.

`verification.json` records the source, environment, commands, results and log
hashes. `source-proof.json` binds the five source-file hashes to `90c06e09`.
The focused green command was:

```bash
cargo test --locked -p calc-flow --lib --features benchmark-internals -- \
  output_projection::tests \
  materialization::tests \
  test_single_and_multiple_records_preserve_admitted_coordinates \
  test_sparse_accessors_preserve_logical_positions_and_payload_offsets \
  batch_parallel_output_cursor_bounds_segment_work_for_selected_sliced_rows \
  scaling_keys_dense_projection_preserves_group_view_coordinates \
  --test-threads=1
```

The orchestrator used the parent checkout's `target/`, two Cargo build jobs
and its managed `.venv/bin/python` as `PYO3_PYTHON`. The initial existing-test
RED used the compatible executable built at `25e8b4c3`; the preflight RED used
`c0cd5875` plus the preserved test-only patch. Neither is described as a test
of the final fixed source.

Reviewer-run static checks returned exit code 0:

```bash
git diff --check c0cd5875 HEAD
git diff --exit-code c0cd5875 HEAD -- schemas/project-v3.schema.json
```

No full local workspace suite, connector services, coverage run, native build,
or timing workload was executed by this reviewer. This review does not
establish the combined 90% Rust line gate. The orchestrator owns the final CI
and Codacy observations; the prior Codacy success is not a new-head result.

## Blocking Issues

No source blockers or unresolved local failures remain. The failing Windows
run remains a merge blocker until the replacement required checks pass; green
required CI, coverage and Codacy are still necessary before the authorized
merge.

## Style Issues

No new convention violations found. The new cursor helper is private and
borrows the existing immutable admission data. Cursor mutation remains local
to its owned traversal state. The patch adds no dependencies, public API,
suppression, unsafe code or caller-owned mutation.

## Test Coverage

- The original failing payload-only materialization test is unchanged. It
  covers both input directions, output ordering, payload identity, invalid
  positions and access to one side without reading the unused side.
- The new preflight test covers both input directions with fixed-width,
  variable-width, nested, duplicate and empty projections. Disabling cached
  charges forces detailed preflight; exact chunk ranges and materialized
  output are compared with the ordinary adapter.
- The accessor helper now compares row and payload cursors on ascending,
  descending and repeated positions. Existing fixtures cover empty batches,
  multiple records and empty segments, sparse selections, projected views,
  and copied payloads with a nonzero base. Assertions check offsets, column
  and schema identity, row IDs, event times, retain flags and invalid bounds.
- The existing selected-output cursor test bounds segment lookup work to
  at most 12 searches across repeated, forward and backward positions.
  The existing projected dense-key cursor case checks the metadata-bearing
  consumer after extraction of the shared helper.

Source comparison confirms that `AdmittedBatch::cursor_view` contains the
previous cursor's bounds check, cached forward traversal and backward-search
fallback. `cursor_row` still calls `GroupView::row`; the new payload cursor
calls `GroupView::payload`. The trait default delegates to `payload`, so an
adapter's payload-only implementation is respected.

The three `JoinOutput` preflight changes substitute payload access without
changing row-cache keys, seen-row tracking, fixed/nested cost arithmetic,
iteration counts, budget errors or materialization boundaries. No allocation,
copy, state commit, checkpoint or asynchronous lifecycle boundary is added.

### Performance scope

The structural cursor-work test and source comparison are the appropriate
checks for this repair. No ASOF code or workload changed, and no fresh timing
measurement was requested for this narrow fix. The previous five-case
measurement remains bound to `25e8b4c3`; it must not be relabelled as a timing
result for `90c06e09`. This review makes no new speedup or equivalence claim.

## Documentation Consistency

The patch restores the existing internal payload-only contract. Public
behavior, configuration, commands, project schema and checkpoint formats do
not change, so no user-guide update is needed. The earlier review and
performance artifacts remain intact and bound to their recorded source.

## Verdict

**Approve — reviewed source and completed focused verification.**

The fix at `90c06e09` resolves the reproduced payload-only contract failure
and preserves the reviewed cursor-work bound. All 18 focused tests,
formatting and scoped Clippy pass. No new timing claim is made.

Replacement required CI results are still pending at this review handoff.
This source approval does not establish merge readiness; the authorized merge
remains gated on the orchestrator's verification of green required checks.
