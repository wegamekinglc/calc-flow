# Continuous Streaming M4 Implementation Evidence

## Scope and authority

This artifact records implementation evidence for the M4 state/window delta
approved in PR #88. The controlling order is:

1. `../specs/m4-state-window.md`;
2. `../api-notes/m4-state-window.md`;
3. `../critiques/m4-state-window.md`;
4. compatible M2/M3 behavior at
   `main@370ae1e7549ae59c8785fd397983e41741cdc358`.

M4 adds the public state model/backend contract and window specification,
crate-private immutable local-state publication, retained-state compaction,
incremental tumbling/hopping execution, operator snapshot/restore, benchmarks,
and window-state soak evidence. It does not add running-job barriers,
manifest selection on runner restart, durable restart coordination, a public
runner control surface, Python bindings, or REST/Studio contracts. Those
boundaries remain assigned to M5 or later milestones.

## Delivery plan and implementation

- **WP1 state and manifest:** `state/{backend,manifest,segment}.rs` define
  portable immutable handles, lineage-scoped backend sessions, strict bounded
  canonical v3 manifests, ordered inventories, deterministic last-operation
  folding, and immutable compaction replacement.
- **WP2 local backend:** `state/local.rs` owns canonical managed roots,
  cross-process lineage leases, same-filesystem stage/validate/publish,
  checksum-verified loads, fail-closed orphan collection, and the private
  ten-boundary crash harness. Filesystem and Arrow work run outside Tokio
  executor threads.
- **WP3 compile contract:** `operator/window.rs` and pipeline compilation add
  strict tumbling/hopping geometry, the frozen aggregate/type matrix, stable
  output schemas and group-key encoding, stream-only compilation, and complete
  semantic fingerprints.
- **WP4 execution and persistence:** the window operator incrementally updates
  transactional accumulators, classifies assignment-level lateness, closes in
  stable key order, chunks output against the task-owned budget, persists
  dirty/tombstone deltas as Arrow IPC, restores complete retained inventories,
  and compacts before inventory bounds.
- **WP5 evidence:** `benches/m4_state_window.rs` isolates state/window costs.
  The existing two-source slow-sink soak now runs a schema-checked
  union/window/two-sink topology and records window checkpoint, compaction,
  inventory, key, delivery, queue, task, and RSS evidence.

## AC-01 through AC-15: state and manifest

- **AC-01:**
  `state_handle_rejects_every_non_portable_identity_and_path` and
  `state_handle_rejects_noncanonical_checksums_and_wrong_ownership` cover the
  complete portable identity, path, checksum, operator, and epoch boundary.
- **AC-02:** `manifest_canonical_bytes_ignore_mapping_insertion_order`.
- **AC-03:**
  `manifest_strict_loader_rejects_unknown_duplicate_missing_and_bounded_json`.
- **AC-04:**
  `manifest_validates_expected_plan_and_handle_ownership_before_load` and
  `manifest_accepts_older_inventory_handles_and_rejects_future_handles`.
- **AC-05:** `manifest_checksum_mismatch_fails_closed` and
  `invalid_restore_has_no_side_effect_and_repeated_restore_is_idempotent`.
- **AC-06:**
  `local_segment_state_machine_checks_visibility_bytes_and_large_payloads`
  proves staged bytes are not loadable; the private commit harness publishes
  the manifest only after all referenced segments validate and publish.
- **AC-07:**
  `local_segment_state_machine_checks_visibility_bytes_and_large_payloads`
  corrupts a committed segment and observes byte-length/checksum rejection
  before decode. `snapshot_segments_round_trip_through_the_validating_local_backend`
  provides the positive Arrow path.
- **AC-08:**
  `local_segment_state_machine_checks_visibility_bytes_and_large_payloads`
  round-trips a payload larger than 10 MiB, while
  `manifest_keeps_large_state_bytes_out_of_the_bounded_document` proves the
  bounded handle-only manifest.
- **AC-09:**
  `every_commit_fault_selects_only_the_previous_or_complete_manifest` injects
  each of the ten D3 failures and reopens only the previous or complete epoch.
- **AC-10 and AC-11:**
  `orphan_collection_removes_old_unreachable_and_preserves_retained_and_newer`
  removes the old unreachable segment while loading the retained and
  post-latest segments unchanged.
- **AC-12:**
  `lineage_lease_and_unexpected_managed_file_type_fail_closed`,
  `managed_symlink_fails_closed_before_orphan_collection`, and
  `orphan_collection_stops_after_the_first_delete_failure` cover lease,
  unexpected-file, symlink, and bounded delete failure behavior.
- **AC-13:** `compaction_fold_matches_an_independent_ordered_model` and
  `last_operation_wins_and_duplicate_operations_fail_closed`.
- **AC-14:**
  `compaction_triggers_before_configured_bounds_and_replaces_immutably` keeps
  the historical inventory unchanged, while the retained/newer orphan test
  proves retained historical handles remain loadable.
- **AC-15:**
  `compaction_triggers_before_configured_bounds_and_replaces_immutably` and
  `delta_threshold_prepares_one_replacement_base_before_inventory_growth`.

## AC-16 through AC-25: window compile contract

- **AC-16 and AC-17:**
  `duration_and_hopping_geometry_validation_names_the_exact_field` covers
  zero, sub-microsecond, overflowed, non-integral, indivisible, and excessive
  overlap geometry.
- **AC-18 and AC-19:**
  `event_time_group_and_output_names_fail_before_execution` covers event-time
  type/timezone, missing columns, duplicate declarations, and reserved-name
  collisions.
- **AC-20:** `accepted_aggregate_matrix_builds_the_exact_output_schema`.
- **AC-21:** `accepted_and_rejected_type_matrix_is_closed` and
  `group_type_matrix_accepts_only_the_frozen_g1_types`.
- **AC-22:** the strict `WindowGeometry`, `WindowSpec`, and `AggregateSpec`
  models expose only tumbling/hopping final-only append behavior. Session,
  early, allowed-lateness, side-output, update, and retract have no
  serializable or constructor representation.
- **AC-23 through AC-25:**
  `configuration_fingerprint_and_stream_only_mode_are_deterministic` changes
  declaration/layout semantics in the fingerprint, holds graph insertion
  order stable, and rejects the window operator from batch compilation.

## AC-26 through AC-39: assignment, aggregation, and emission

- **AC-26:**
  `tumbling_windows_use_euclidean_boundaries_and_emit_once_in_key_order`.
- **AC-27:**
  `hopping_row_receives_exact_overlap_in_oldest_start_order`.
- **AC-28:**
  `assignment_overflow_aborts_every_earlier_update_in_the_batch`.
- **AC-29:**
  `snapshot_persists_separate_assignment_late_and_null_time_metrics`.
- **AC-30:**
  `all_null_aggregate_inputs_emit_count_zero_and_nullable_results`.
- **AC-31:**
  `aggregate_overflow_aborts_the_whole_input_batch_transaction` and
  `count_and_average_counts_reject_uint64_overflow`.
- **AC-32:** `float_group_identity_distinguishes_signed_zero_and_nan_payloads`
  and `float_aggregates_canonicalize_nan_and_preserve_selected_scalar_bits`.
- **AC-33:**
  `oversized_group_key_fails_without_installing_partial_state`.
- **AC-34:** `randomized_partitions_match_a_finite_group_by_oracle`.
- **AC-35:**
  `ordered_rows_produce_identical_float_bits_across_input_batch_partitions`
  plus the randomized finite-oracle property test.
- **AC-36:**
  `tumbling_windows_use_euclidean_boundaries_and_emit_once_in_key_order`,
  `output_chunking_preserves_rows_and_uses_consecutive_sequences`, and
  `one_oversized_output_row_fails_before_returning_any_chunk`.
- **AC-37:** the tumbling close test and
  `unary_runtime_control_is_fifo_and_handler_precedes_watermark_forwarding`
  prove close eligibility and handler-before-forward ordering.
- **AC-38:** the tumbling test proves idempotent all-ended flush for remaining
  non-empty windows; the operator-task FIFO test proves exactly one forwarded
  `EndOfInput`; `real_soak_topology_graceful_smoke_conserves_every_accepted_sequence`
  exercises the complete runtime topology.
- **AC-39:** `watermark_gaps_do_not_materialize_empty_windows`.

## AC-40 through AC-52: late data and state lifecycle

- **AC-40 and AC-41:**
  `lateness_is_classified_by_window_end_not_row_timestamp` covers both the
  accepted 10:15 row at 10:30 and dropped 10:15 row at 11:05.
- **AC-42:**
  `hopping_row_drops_only_closed_assignments_against_input_watermark`.
- **AC-43:**
  `snapshot_persists_separate_assignment_late_and_null_time_metrics` records
  one affected batch for late assignments in that batch.
- **AC-44:**
  `window_metrics_accumulate_across_contexts_and_mirror_to_runtime_metrics`.
- **AC-45:** `metric_overflow_is_transactional`.
- **AC-46:** `LateMetricDelta`, operator progress/status, snapshot metadata,
  and structured soak records contain only checked counts and maximum lateness;
  source inspection rejects payload-bearing fields. Late and null input tests
  use distinguishable payloads and observe only those aggregate metrics.
- **AC-47:**
  `checkpoint_is_incremental_arrow_ipc_and_restore_replaces_live_state` and
  `later_delta_restores_with_the_complete_retained_segment_inventory`.
- **AC-48:**
  `checkpointed_close_transition_and_output_sequence_survive_restore`.
- **AC-49 and AC-50:**
  `invalid_restore_has_no_side_effect_and_repeated_restore_is_idempotent`, the
  checkpointed close-transition test, and the randomized partition oracle
  prove boundary/replay convergence and stable output ordering.
- **AC-51:**
  `delta_threshold_prepares_one_replacement_base_before_inventory_growth`,
  `compaction_fold_matches_an_independent_ordered_model`, and the complete
  retained-inventory restore test.
- **AC-52:**
  `prepared_large_snapshot_keeps_the_next_control_handler_responsive` uses
  paused Tokio time and 4,096 groups; snapshot preparation performs Arrow
  encoding in a blocking worker before the synchronous checkpoint handoff.

## AC-53 through AC-58: regression and delivery gates

- **AC-53:** the complete Rust harness retains every M2 lifecycle,
  cancellation, ownership, backpressure, fan-out, and cleanup test. The three
  non-ignored soak topology tests also pass.
- **AC-54:** the complete Rust harness retains every M3 watermark,
  idle/reactivate, progress trace, receipt, snapshot, and deterministic replay
  test.
- **AC-55:** the diff from M4 base is empty for `python/`,
  `web-ui/openapi.json`, `web-ui/src/api/schema.d.ts`, and
  `schemas/project-v2.schema.json`. Existing public runner APIs are unchanged.
- **AC-56:** isolated Criterion measurements and the paired unrelated-path
  baseline are recorded below.
- **AC-57:** the named gate is
  `twenty_minute_two_source_slow_sink` with `CALC_FLOW_STREAM_SOAK=1`.
  Its `calc-flow.m4-soak-log.v1` result records exactly 120 ten-second samples,
  the final inventory/key/queue/task state, delivery conservation, and RSS
  gate. The exact final-head raw-log checksum and result are recorded in the
  implementation PR after the run; a later push invalidates that result.
- **AC-58:** local repository gates are recorded below. Copilot, Codacy,
  review-thread, exact-head CI, mergeability, and merge evidence are recorded
  in the implementation PR. This artifact does not claim those remote gates
  before they finish on the final pushed SHA.

## Local verification and benchmark evidence

The implementation source was validated with the repository's complete Rust,
Python, Studio, frontend, release-helper, and supply-chain command groups.
Notable results before PR publication were:

- Rust formatting, workspace Clippy with warnings denied, Rustdoc, the Rust
  harness including the isolated PyO3 target, and all focused M4 tests passed.
- Workspace line coverage passed at **91.00%** against the 90% floor with the
  managed Python library path supplied to the PyO3 test binary.
- Python passed 436 tests plus Ruff check and format check.
- Studio backend passed 150 tests with 4 skipped and 93.88% coverage.
- Frontend API sync/build, 182 Vitest tests, 4 Playwright tests, and production
  npm audit passed; the production audit reported zero vulnerabilities.
- RustSec audit, cargo-deny, and 48 release-helper tests passed, with one
  expected helper skip.
- Generated project schema, OpenAPI, and TypeScript contracts were unchanged;
  no generated `_native*.so` remained in source.

The isolated M4 Criterion runs used 100 samples and reported these 95%
confidence intervals:

- incremental state write, 64 KiB: **335.00–349.56 us**;
- full state restore, 4 MiB: **1.9801–2.0037 ms**;
- compaction read, 8 x 64 KiB: **534.68–550.47 us**;
- Arrow window restore, 1,024 keys: **393.22–407.61 us**;
- tumbling update, 1,024 rows: **860.90–876.01 us**;
- hopping update, 1,024 rows: **8.1297–8.2133 ms**.

The paired unrelated-path run compared base `370ae1e` with the M4 runtime
implementation using 15-second measurement windows and 100 samples. Every 95%
upper confidence bound stayed below the +5% gate: stream data **-1.3851%**,
stream control **-0.3415%**, unary **+0.5611%**, and fan-out **+0.0289%**.

The root environment could not perform a networked `uv sync` under the local
execution policy, so dependency synchronization used the locked offline cache.
All locked build/test commands then passed in that managed environment. CI is
the authoritative clean-environment synchronization check.
