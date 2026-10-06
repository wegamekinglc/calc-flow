# Stream Join safety and J1

## Scope

Issue #363 Phase 0.6 and J1.1–J1.6, following the shared
[specification](../specs/stream-join-asof-acceleration.md). Public matching,
ordering, logical V1 charges, state decisions, and V1 checkpoint encoding stay
fixed. Moving compaction to asynchronous preparation may change whether a
capture carries deltas or an equivalent base for a particular handler history;
the frozen fixture and canonical encoding remain compatibility evidence.
J2 representation changes and ASOF traversal changes are separate work.

## Design checkpoint

- An order oracle checks every concrete arrival schedule by incoming physical
  position, opposite event time, and opposite physical row ID. Nullable time,
  nullable key, and late rows consume IDs before being dropped. Random capture
  cuts must preserve continuation output, batch sequence, and counters.
- Capture V1 bytes before optimization, including repeated non-tail eviction,
  tombstone ordering, and a canonical compacted base. The fixture lives beside
  the Join source; historical `tests/fixtures/v1/` stays untouched.
- Maintain a stable-order dirty log with an upsert identity index. Coalescing
  removes only evicted uncaptured identities; live iteration preserves the old
  dirty operation sequence without traversing freed slots.
- Index retained expiration by `(event time, row ID)`. A dense row slot can be
  removed without moving every survivor; a separate stable ordinal preserves
  the old eviction/tombstone order. Restored ordinals follow the existing fold
  order and new rows receive the next monotonic ordinal. A `u128` ordinal covers
  restored `usize` rows plus all later `u64` physical rows without a new overflow
  decision. Output and base capture keep their existing canonical sorts.
- Update retained row/byte gauges incrementally. Recount remains a restore
  validation operation. Encode retained keys once and reuse their byte length
  for the frozen charge. Decode the timestamp array type once per record,
  preserving checked multiplication and negative nanosecond floor conversion.
- Expiration and upsert lookup add auxiliary descriptors proportional to
  retained rows and pending changes. Frozen logical charges stay unchanged;
  paired measurements must report RSS rather than infer a memory reduction
  from the reduced traversal count.
- Prepare base compaction through the bounded owned worker boundary, with
  immutable input ownership and cancellation cleanup. Data/progress do not
  initiate bulk compaction. A dropped attempt cannot install stale bytes after
  reset or later mutation. Direct capture carries dirty segments without an
  unexpected full-state encoding.

## Verification record

- `cargo test --locked -p calc-flow --test stream_join_properties
  ordered_schedule_and_random_checkpoint_cuts -- --nocapture`: passed, with
  32 deterministic generated cases. This closes a coverage gap; no behavioral
  failure is claimed for the old implementation.
- Focused inline compilation initially failed with eight missing
  `reset_join_work`/`join_work` symbols. Test-only instrumentation now counts
  existing retained/log traversals, key encodings, and timestamp dispatch.
- Instrumented pre-J1 Join module: 35 passed, six failures. Genuine complexity
  red results were 160 retained visits for a no-expiry update (expected zero),
  158 retained visits for two sparse expirations (expected two), six key
  encodings for three retained rows (expected three), and three timestamp type
  dispatches for one record (expected one). The compaction test also failed
  because a data handler cleared the scheduled compaction flag.
- The sixth failure was the initially empty historical fixture target. Five
  epochs were captured from this unchanged pre-J1 implementation and frozen in
  `join/fixtures/checkpoint-v1.json`; fixture creation is a coverage artifact,
  not a claimed runtime defect. The subsequent green run verified exact bytes
  for all five captures, plus restore and continuation of the final compacted
  capture.
- Partial J1 verification: 40 passed, four expected failures. All frozen V1
  capture bytes, final-capture restore/continuation, zero no-expiry visits, two
  sparse retained visits, one key encoding per retained row, and one timestamp
  dispatch per record passed.
  Actual coalescing red results were six visits for one eviction from three
  dirty rows and 160 visits for two evictions from 80 dirty rows.
- A new terminal ordering guard failed with `[0,4,2]` against `[0,2,4]` after
  sparse removal. End now shares the stable retention ordinal, preventing
  terminal tombstone wire drift.
- Indexed log and stable End verification: 43 passed, with only the two J1.6
  guards still failing. One dirty-row eviction now visits one pending identity,
  and two sparse evictions visit two identities. Exact terminal order and all
  five frozen captures pass. The direct capture without an intervening handler
  carries valid V1 deltas, but the old preparation still leaves four deltas
  instead of preparing two base segments. This records the expected J1.6 red.
- The coherent Phase 0.6 and J1.1–J1.5 stage excludes the future J1.6 guards:
  42 focused inline tests and all four Join property tests pass after the
  argument/fixture refactors. `cargo test --locked -p calc-flow --lib --test
  stream_join_properties --no-run` prepared the exact binaries; only the Join
  unit filter and property binary were executed. Core lib/tests Clippy passes
  with warnings denied. Owned Rust formatting, whitespace, and generated
  contract drift checks are clean.
- Rebase onto the accepted-ingress status stage `94383f5b` applies without
  conflicts. The integrated production/test tree at `5751377e` passes all
  43 focused Join inline tests and all four property tests; frozen checkpoint
  metadata and segment bytes remain unchanged. Integrated core lib/tests
  Clippy also passes with warnings denied. The final-capture continuation
  wording correction changes documentation only.
- Final specialist review approved the integrated Phase 0.6 and J1.1–J1.5
  source and artifact, including the corrected final-capture continuation
  description. Asynchronous compaction ownership is a later J1.6 slice, and
  paired performance measurements are still pending. No measured performance
  improvement is claimed. Full workspace regression, coverage, and
  cross-platform gates remain CI responsibilities.
- The property schedule's reference state now uses explicit left/right slots
  and a separate retention predicate to address the Codacy complexity finding.
  Physical IDs, both interval directions and the independent checkpoint oracle
  remain intact. All four properties and scoped target Clippy pass; final
  specialist review approved the reference refactor and review-status record.

Shared Cargo artifacts use relative dependency paths and timestamp freshness.
The first inline filter reused another worktree's newer test binary and ran
zero tests. Touching this worktree's modified Join source forced the correct
rebuild; that zero-test invocation is not counted as verification.
