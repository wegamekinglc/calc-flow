# ASOF columnar admission - Critic Critique

## Target

- [Acceleration spec](../specs/stream-join-asof-acceleration.md), FR3/FR4/FR16,
  and [Introduction](../../../docs/introduction.md), immutable Batch ownership.
- Main `eccb26973811bc476f0944b977ddedf8564b0237`, including A2 and paid
  retirement. This supersedes the design verdict in the read-only prior artifact
  `.worktrees/stream-asof-output-ranges/.codex/artifacts/critiques/stream-asof-columnar-admission.md`.
- Static design review only; no source changes, build, tests, measurements, or
  remote operations. A3 output-range planning is outside this gate.

## Verdict

**Approve** a narrowly eligible left-admission chunk-construction fast path
under the constraints below. There is no unresolved design blocker within
that scope. Output/state/resource parity remains required implementation
acceptance; unobserved performance gains are not an additional design gate.
Right-bucket bulk construction, removal of shared admission descriptors, and
new Arrow backing-sharing policies require a separate concrete review.

## Findings and approved boundary

### Blocking Issues

None for the bounded scope. The following are restrictions of this approval,
not optional implementation recommendations.

### Significant Concerns

- **Do not reuse left admission key indices blindly.**
  [admission.rs](../../../crates/calc-flow/src/operator/asof/admission.rs),
  `input_identity`, assigns `key_index = 0` for every inline left key.
  `AdmissionRef.key_index` is therefore not a valid chunk dictionary ID for
  multiple short keys. The fast path must reproduce `ChunkData`'s canonical
  first-occurrence key IDs/counts; 64 different short keys must not collapse
  into one. Reuse canonical key handles where valid, including resident key
  owners, without changing owner forms or encounter order.
- **A columnar allocation can still change the resource contract.**
  [left.rs](../../../crates/calc-flow/src/operator/asof/state/left.rs),
  `checkpoint_capacities`, records actual capacities, and `intern_chunk_key`
  starts keys/counts at capacity one with its existing growth policy. Exact
  distinct-count preallocation can change capacity for three/five keys.
  Preserve all six actual capacities: owned times, optional positions,
  keys/counts, key IDs, and sequence storage. Synthetic charge parity with
  different owner wire inventory is insufficient.
- **Keep both retirement waits and gather registration guards.** The latest
  ownership boundaries are detailed below; a fast-path early return must not
  bypass them.

### Minor / Style Notes

No public API, dictionary/nested payload support, budget, timestamp unit,
checkpoint version, or generated contract change is justified by this scope.

## Specific optimization scope

[PreparedLeftChunk::prepare_checked](../../../crates/calc-flow/src/operator/asof/state/left.rs)
collects a borrowed identity/position vector for each payload, then
`ChunkData::prepare` checks order and re-interns keys while rebuilding columns.
The fast path can remove that intermediate collection and repeated preparation
from already validated ordered admission ranges. Keep `Admission.rows`, whose
length and content remain used by status, capacity preflight, and journals;
removing it is not needed for this first delivery. This addresses admission,
not A3 output planning. No measured improvement is asserted here.

Eligibility requires all of the following:

- Left input with exact validated schema, non-null identities, every row
  accepted, contiguous physical ranges, and no late/drop/overlap/duplicate row.
  Empty inputs and any failed eligibility proof use the existing path.
- Microsecond event time, exactly one supported signed/unsigned integer
  sequence with its native 1/2/4/8-byte width, and one supported scalar integer
  or Utf8/LargeUtf8 key. Composite keys/sequences and string sequences remain
  fallback shapes for this delivery.
- Strict canonical `(time, key bytes, sequence)` order within and across all
  records, and a first identity strictly after the actual live committed left
  maximum, including restored or previously out-of-order state. Compare key
  bytes, never dictionary IDs; typed sequence order must equal canonical order.
- Existing canonical key/sequence encoding and first-occurrence key dictionary
  order. Preserve inline/shared/batch encoding forms, capacities, reference
  topology, and retained dictionary bytes. Keep the established string-key
  encoder, which already encodes distinct values once.

## Funding, registration, and retirement constraints

Times stay in an owned `Vec<i64>` with legacy capacity; integer sequences keep
the exact-width owned prefix copy in
[sequences.rs](../../../crates/calc-flow/src/operator/asof/state/sequences.rs),
`from_array_prefix`, including sliced offsets and little-endian wire storage.
Unknown Arrow backing cannot become safe merely because visible capacity
equals length. Keep the owned-copy safeguard demonstrated by
`detached_compaction_does_not_retain_unknown_external_time_owner`.

Retain `compact_accepted_rows`/`encode_payload`, full-slice sharing eligibility,
projection, payload batch keys/refcounts, IPC/body charges, and lazy encoding.
Known larger slices keep funded compaction and alias-capacity accounting;
unknown external backing retains the existing validated payload decision and
owned chunk copies. This gate introduces no new direct backing retention.

Keep identity/payload/key-copy/descriptor/staging/journal/index reservations,
their acquisition order, and the existing bounded CPU-work threshold.
Duplicate-versus-workspace precedence continues through the established
fallback duplicate validator. No new queue, worker allowance, or memory budget
is permitted. The 32 MiB process infrastructure ceiling remains distinct from
operator limits and must keep the same refusal behavior.

[mod.rs](../../../crates/calc-flow/src/operator/asof/mod.rs) waits for retirement
before `process_data`, progress, async checkpoint preparation, and End.
`install_admission` commits synchronously after all preflight, drops its local
leases, then awaits retirement. The optimized admission must use these same
entry/exit boundaries; cancellation during the final wait cannot roll back
already committed admission or imply retired ownership was refunded.

[retirement.rs](../../../crates/calc-flow/src/operator/asof/retirement.rs)
registers an operator ticket and job `RetirementGuard`. In
[copy.rs](../../../crates/calc-flow/src/operator/asof/copy.rs) and
[payload.rs](../../../crates/calc-flow/src/operator/asof/state/payload.rs), field
drop order releases originals/replacements and workspace before the ticket.
Reset/restore do not clear the shared pending retirement owner; job drain must
still await it after operator drop. Reuse these paths without early ticket or
credit release.

[gather_work.rs](../../../crates/calc-flow/src/runtime/streaming/gather_work.rs),
`ensure_pool`, awaits `wait_retirement` before growing/registering workers.
The fast path cannot submit native work through another executor or registry
to skip this generation guard. Detached work owns its input descriptors,
buffers, and funded lease until native completion and cleanup.

## Minimal test-first acceptance and measurement

1. **Record real red evidence for removed work.** Add non-timing instrumentation
   that demonstrates the legacy per-payload borrowed-vector/repeated chunk
   preparation, then require its absence on eligible admission. Existing parity
   tests that already pass are coverage evidence, not a fabricated red run.
2. **Eligible differential fixture.** Force legacy and eligible paths with
   immutable inputs: 0/1/1,024-row boundaries, 1/3/5/64 short keys, long and
   non-ASCII keys, all integer sequence widths/extrema, equal-time ties, and
   multi-record cuts. Compare status, inventories/index length, six actual
   capacities, payload owner/refcounts, and encoding-owner topology. Specifically
   include 64 inline keys to detect the fixed-zero admission-index trap.
3. **Order and fallback.** Compare against canonical encoding for negative and
   extreme integers and empty strings; test reversal, boundary equality,
   duplicates, mixed late input, composite/string sequences, restored state,
   and prior legal disorder. Preserve fallback reasons, counters, and inputs.
4. **Wire and prefixes.** Compare same-epoch metadata, journal edits/owner
   records, and every segment byte/hash after admission, dirty/repeated capture,
   restore/continuation, and partial finalization/eviction/compaction. Semantic
   version 3, layout/accounting 10, and `CFASRW10` framing remain exact.
5. **Tight budgets and backing.** Check legacy thresholds immediately below/at
   each affected reservation and state charge, a 32 MiB pool/limit, duplicate
   plus insufficient workspace, known aliased/oversized backing, unknown
   external backing, and projected/full payloads. Preserve rejection category,
   counters, unchanged state, and cleanup; a formerly rejected shape must not
   be relabeled a speedup.
6. **Live retirement.** Hold a worker/ticket through nonempty eligible
   admission, cancellation, dropped waits, reset/restore, and job drain. Before
   release, admission/registration waits remain pending and owners/credit live;
   after release, cleanup completes. Use the existing tests
   `dropped_retirement_waits_stay_funded_across_reset_restore_and_cancellation`
   and `managed_job_drain_owns_retirement_after_operator_drop` as analogues.

Measure the 64-key ordered integer-sequence fixture at the established small
and 64,000-row batch sizes. Count actual allocations across the caller and
detached worker paths, distinguish owners/batches from per-row allocations,
and report peak process RSS including live workers and funded replacements.
Do not infer worker allocations from a counter scoped only to the caller
thread. Timing claims require the established two rounds of ten sealed AB/BA
pairs with identical oracle, work, scope, and thread identity; report reached
or unreached targets. These are delivery evidence requirements, not a demand
to observe planning gains before starting approved implementation.

## Counter-Proposal and axis audit

Use the existing admission envelope, payload ownership, funded worker path,
and final install boundary; optimize only ordered left chunk preparation first.
This has a concrete savings mechanism without exposing a new public surface.
Correctness, immutability, compatibility, ergonomics, hidden ordering/backing
assumptions, edge cases, tests, performance evidence, and scope are covered
above. Bulk right admission remains a separate proof because its reserve/growth
capacities and journal order can differ from the current per-key insert path.

## Questions for the Author

None requiring the user. After A3's specialist review, route this bounded scope
and the actual differential evidence to the implementer/final reviewer. Do not
broaden eligibility merely to reach a target timing.
