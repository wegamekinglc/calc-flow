# Stream Join and ASOF acceleration - Specification

## Source and scope

- [Issue #363](https://github.com/wegamekinglc/calc-flow/issues/363), including
  its measurement appendices; user continuation request on 2026-10-06.
- [Introduction](../../../docs/introduction.md),
  [streaming guide](../../../docs/streaming-guide.md),
  [ASOF contract](../../../docs/asof-join-guide.md), and
  [frozen ASOF invariants](../../../docs/plans/2026-09-30-streaming-asof-acceleration.md#3-必须保持的不变量).
- Implementation base: `9b1535bc`; the evidence-foundation delivery covers
  Phase 0.1, 0.5, and 0.7. This specification covers remaining Phase 0.2–0.4,
  0.6, J1.1–J1.6, J2a/J2b, J3, and A1–A5.

The issue identifies per-row object creation, repeated retained-state scans,
and checkpoint preparation in native stream Join and ASOF hot paths. The
lookup benchmark alone cannot demonstrate improvements to retained interval
state or checkpoint-enabled operation. Delivery therefore needs both preserved
public behavior and comparable, oracle-checked performance evidence.

The user's instruction to continue the plan is interpreted as choosing the
issue's recommended D1–D6 direction: read Join v1 and write v2 at J2b; native
stream equality probing; bounded shared gather workers with deterministic
merges; scheduled interval/checkpoint/small-batch coverage; Join first, with
A1/A2 independently deliverable; additive Join progress status. These are
implementation assumptions inferred from the continuation instruction, not
separately recorded public-contract approvals. Critic review must resolve a
source contradiction before the affected high-risk implementation proceeds.

## Goals and exclusions

- Preserve Join and ASOF results, resource decisions, recovery, cancellation,
  structured failures, and deterministic ordering while removing avoidable work.
- Make retained-state, checkpoint-enabled, and small-batch behavior measurable
  in the scheduled suite, with sealed builds and explicit coverage identity.
- Measure every delivered optimization stage and report whether its planning
  target was reached; an implementation result is not evidence of a speedup.

Batch SQL joins, DataFusion SQL/expression semantics, caller-owned Arrow
mutation, public runner-control injection, relaxed resource limits, and changes
to the ASOF v3 checkpoint contract are outside this work. The allocator
experiment is optional follow-up after J2/A4, not required for these phases.

## Contracts preserved by every phase

**FR1 — Join matching and order.** Match identical supported key tuples within
the inclusive interval `[left_time - before, left_time + after]`. Null event
times and null keys retain their existing drop precedence and counters. A row
is late only below its own side's accepted watermark; equality stays on-time.
Every physical input row consumes its row ID, including dropped rows. For each
incoming batch, emit exactly the existing sequence ordered by
`(incoming position, opposite event time, opposite row ID)`. Compare this
sequence for each concrete arrival schedule; different arrival schedules need
not produce the same Join emission sequence.

**FR2 — Join charges and atomicity.** Frozen V1 row charges remain unchanged
through J3, including exact retained-byte gauges, prospective limit decisions,
and restore validation. Row, match, counter, and single-output-row preflight
failures happen before any output as they do today. Every output range is
preflighted against edge budgets; dictionary output excludes unreferenced
values, and nested/variable-width payload checks remain effective. Commit new
retained state and batch-level counters only after all output chunks have been
accepted. Already accepted chunks and their sequence advancement follow the
existing cancellation behavior; cancellation does not retract sink writes.
Evict only when `time + extension < opposite watermark` or that input ended.
Join V1 retained bytes are a logical charge, not a process-memory ceiling.
Preserve the existing documented nested/dictionary trial-allocation boundary;
an edge's logical byte budget does not become a hard allocation ceiling.

**FR3 — ASOF semantics and v3 accounting.** Preserve backward candidate
selection by maximum `(right_time, right_sequence)` inside inclusive tolerance,
strict dual-watermark finality, unmatched left rows, and canonical output order
`(left_time, key bytes, sequence)`. Preserve non-null identity validation,
late/duplicate precedence, identity-only retention, output prefixes, capacity
and owner charges, workspace ceiling, and structured reasons. Sink acceptance
remains the commit boundary for each finalized prefix. Cancellation preserves
accepted prefixes and remaining rows; reset does not release a live worker's
input ownership or funded workspace early. Preserve semantic state version 3,
current row-log layout/accounting version 10,
unchanged `CFASRW10` bytes, canonical capture, restore validation, and rejection
of ASOF v1/v2 and unsupported older row-log layouts.

**FR4 — Execution ownership.** CPU-heavy preparation uses the established
bounded blocking/gather boundary rather than blocking Tokio executor threads.
Scratch, replacement buffers, shared input owners, and worker lifetimes obey
their existing accounting/preflight contracts through failure, cancellation,
reset, and join cleanup; ASOF's explicit workspace ceiling remains fail-closed.
No phase enlarges queues or reduces output columns/work to obtain a speedup.
Python remains a declaration/binding surface; native equality probing applies
only to the built-in stream Join, with DataFusion remaining the SQL/expression
engine. Public project graphs, API declarations, and errors stay compatible.

## Phase requirements

**FR5 — Phase 0.2 observable Join progress.** Add per-side nullable integer
`watermark_micros`, boolean `idle`, and boolean `ended` through Rust, Python,
and Studio status. Report accepted ingress progress, including restore,
reactivation, and End; never infer it from retained rows. Preserve all existing
status fields. Studio uses canonical decimal strings for full-range timestamps
and JSON null for absent values, matching its ASOF progress convention. Update
OpenAPI and generated TypeScript together. The static benchmark waits for the
sealing right watermark before feeding the first quote; it must retain and
evict zero left rows throughout, with state rows bounded by dimension rows plus
one batch. No timed busy status polling is allowed.

**FR6 — Phase 0.3 interval coverage.** Add a two-stream interval Join with 64
keys and inclusive ±5-second bounds, using the same input rows and projected
result for native stream, calc-flow SQL, DataFusion, and Polars references.
It must exercise both-side retention and watermark eviction, with oracle
coverage for boundary equality, duplicate keys, and out-of-order on-time rows.
Reference result normalization must not erase the native order tests in FR1.

**FR7 — Phase 0.4 replay and dimensions.** Add genuinely replay-capable
benchmark sources with stable cursors, exact next-position recovery, and legal
watermark replay. Add scheduled checkpoint-enabled and 1,024-row variants of
Join, interval Join, ASOF, and projection, retaining the existing 64,000-row
cases. A checkpoint-enabled evidence run must publish at least one nonterminal
epoch while measured input is processed; merely configuring a 100 ms timer
without exercising it is insufficient. Record interval, batch rows, workload,
scope, accepted epochs, and recovery mode in case identity/evidence. Short
workloads may need a distinct documented duration/workload case to exercise
the timer; do not silently pad an existing throughput workload. Verify resume
against uninterrupted output, respecting each sink's declared delivery proof.
Runtime failures terminate the measurement promptly through owned cleanup.

**FR8 — Phase 0.6 safety evidence.** Add reproducible order-sensitive Join
property coverage across partitions and legal interleavings, random checkpoint
cuts, and durable v1 fixtures captured before J2 changes. Exercise rows dropped
before matching so row IDs cannot accidentally become accepted-row IDs. Add
non-timing complexity instrumentation for no-op/evicting Join progress and ASOF
left traversal. Record the failing or missing behavior before its implementing
phase; a new test that already passes is a coverage gap, not a claimed red run.

**FR9 — J1.1–J1.5.** Progress with no evictions performs no dirty-log or
retained-row traversal. Tombstone/coalescing work is proportional to evicted
identities, not the entire uncaptured log. Maintain exact row/byte gauges
incrementally; recount only during restore/debug verification. Expiration
checks skip unaffected state, and affected work visits expired entries rather
than all live rows. Retained keys are canonically encoded once and reused for
charges. Timestamp decoding and checked unit conversion occur by batch,
preserving every supported timestamp unit/time zone and
`JoinTimeConversionFailed`. Checkpoint v1 bytes remain identical for the same
state, epoch, and dirty history.

**FR10 — J1.6.** Bulk base compaction moves out of data/progress handlers into
asynchronous checkpoint preparation. Capture continues to scale with dirty
changes plus carried segment metadata; it never scans/re-encodes the whole live
state unexpectedly. Cancellation or failed preparation retains a usable prior
checkpoint and unconsumed changes; a worker cannot install stale preparation
after reset or intervening state changes. Direct synchronous capture remains
well-defined without weakening managed asynchronous ownership.

**FR11 — J2a.** Replace per-row retained payload objects and per-batch SQL
equality queries with columnar retained chunks and canonical native key
equality. Supported integer/string/date/timestamp and composite keys retain
exact equality; unsupported key shapes remain rejected. Preserve physical row
IDs, inclusive range probing, opposite `(time, row ID)` order for on-time
out-of-order inserts, flat V1 charges, dictionary garbage collection, and
nested-payload/output preflight. Reclaim fully unreferenced payload chunks and
bound sparse retained backing memory through funded compaction. Continue to
write/read v1 wire format until J2b; immutable input sharing must not weaken
existing owner or payload-compaction checks. Report oversized backing-slice
retention separately from frozen logical charges. Allocation/RSS evidence must
include workers.

**FR12 — J2b compatibility.** New captures write Join layout v2 using canonical
columnar base/delta state; dirty capture remains incremental and compaction
remains asynchronous. Read valid v1 and v2; after restoring v1, continuation
and a subsequent v2 checkpoint preserve outputs, counters, progress, row IDs,
and sequence. Reject unknown versions, malformed framing/indices, duplicate or
invalid references, tampered charges, segment corruption, and incompatible
spec/schema before installing any state. Include a managed manifest produced
by the prior v1 release in the migration test, not only an operator snapshot.

Representation-only changes must preserve the semantic graph identity. Keep
Join's checkpoint capability at semantic state version 1 and retain exact
pipeline fingerprint/version checks. Only the private Join wire layout moves
from 1 to 2, with its decoder accepting those two proven compatible layouts.
A managed old-release fixture must resume, continue, capture v2, and restart
again from that capture. Adding an operator v1 reader alone does not satisfy
D1. Continue rejecting altered graphs, source/sink binding identities, schemas,
UDF versions, and checkpoint lineage.
Update Studio checkpoint-family handling and normative format documentation.

**FR13 — J3.** Large Join batches may use the existing bounded gather pool for
probe/materialization under deterministic size thresholds. Concatenate output
in incoming-range order and preserve exact within-range FR1 order. State
mutation stays owned and sequential. Small batches retain a bounded fallback;
parallel scratch, active worker ownership, failures, and cancellation obey
FR2/FR4. The optional sealed-side specialization must pass the same tests.

**FR14 — A1/A2.** ASOF ordered chunk traversal and ownership accounting operate
on maximal ordered runs where possible; overlap retains exact canonical merge
order. For nonoverlapping chunks, iterator/heap steps scale with runs/chunks,
not individual rows; row-wise candidate work may remain linear. Owner changes
aggregate by batch/key/sequence owner without losing partially referenced
capacity. Prefix inventories, journal changes, and canonical v3 state match
the pre-optimization behavior on generated inputs.

**FR15 — A3.** Represent left output using source ranges and right output using
candidate references without rebuilding left columns row by row. Estimate
variable-width workspace from the selected ranges before allocating. Preserve
projected/full output, null candidates, backing-allocation charges, edge chunk
splitting, prefix commit, and cancellation inventories exactly.

**FR16 — A4.** A columnar admission fast path is eligible only after proving
strict canonical `(time, key bytes, sequence)` order and a valid boundary
against committed state. It must preserve native integer sequence widths,
canonical dictionary bytes, duplicate identity decisions, v3 inventory/journal
and checkpoint bytes. No implicit type casts or ordering assumptions from key
ID assignment are permitted. Preserve the original timestamp and encoding
owners, their funded capacities, and exact legacy charge-limit decisions.
Direct Arrow backing sharing requires proof that it retains no larger hidden
allocation; otherwise use the existing funded owned-copy path. Late/drop,
overlap, reversal, duplicate, string
sequence, or unproven typed ordering uses the existing validated fallback.
Test typed ordering against canonical encoding for negative/extreme integers,
empty/non-ASCII strings, composite keys, and equal-time ties. For the bounded
64-key ordered integer-sequence workload, allocation count grows with batches
and distinct owners, not one allocation per row; arbitrary unique-key input
is not claimed to have constant allocation count.

**FR17 — A5.** Parallel admission, candidate preparation, and materialization
preserve global canonical output and deterministic v3 snapshots independently
of shard/task completion order. Keep sequential and constrained-workspace
fallbacks. A cancelled/failed shard cannot commit a partial admission or
unaccepted prefix; all outstanding workers retain funded owners until cleanup.

## Measurement and delivery acceptance

The issue's historical measurements are investigation references, not paired
release baselines. Each performance claim requires two rounds of ten AB/BA
pairs on sealed release wheels, on one runner with the same candidate harness,
dependencies, input/output oracle, workload, scope, and actual thread identities.
Changed/new scope is `new-coverage`; do not label a v4→v5 difference a runtime
speedup. The evidence-foundation PR reports v5 timings and the removed polling
cost as diagnostics separately from a same-scope runtime comparison.

| Milestone       | Planning target / required observation                                           |
| --------------- | -------------------------------------------------------------------------------- |
| Phase 0.1       | 100k/1,024-row ASOF harness overhead ≤ 2× the matched projection floor           |
| J1              | Retained Join 1M→4M scaling ≤ 4.4×; no-op progress has zero retained visits      |
| J1.6            | Handler latency during compaction ≤ 2× steady; report p50 and p95/p99            |
| J2              | 1M/10M lookup Join p50 ≤ 60/600 ms; capture/restore ≤ 0.1/0.2 µs per row         |
| J2, A4          | Report allocation count and peak RSS including worker paths                      |
| J3              | 1M/10M lookup Join p50 ≤ 15/150 ms; 1,024-row operator batch ≤ 0.05 ms           |
| A4              | 1M ASOF p50 ≤ 100 ms, with all declared output work preserved                    |
| A5              | 1M ASOF p50 ≤ 30 ms and ≥ 8× versus the single-thread A4 kernel                  |

Record reached/unreached targets explicitly; no estimate becomes an observed
gain. Operator microbenchmarks resolve small-batch costs that Python harness
latency cannot. Report the ASOF operator and projection floor separately and
describe any cross-boundary overhead estimate as diagnostic. Paired uncertainty
must follow the suite's regression policy rather than cherry-picked minimums.

- [ ] Each phase has recorded focused red/coverage-gap evidence, passing named
  tests, and final specialist review of its actual delivered head.
- [ ] `cargo test --locked -p calc-flow --test stream_join_properties` and
  `--test stream_asof_join_properties` cover ordered results and random cuts;
  scoped inline Join/ASOF tests cover affected limits, migration, and ownership.
- [ ] Appropriate focused benchmark adapter/suite tests, Python binding tests,
  and Studio status/model/API tests pass for each affected phase. Run module
  format/lint checks only for its changed surfaces; record exact commands.
- [ ] v1 Join fixtures resume both directly and through managed manifests;
  tampering or changed semantic graph still fails closed. ASOF v3 fixtures and
  fingerprint tests continue to pass without format drift.
- [ ] Run scope/dimension evidence rejects missing or mislabeled checkpoint,
  batch, replay, thread, native hash, or wheel identity. New variants appear in
  scheduled measurement and reports; unsupported references stay explicit.
- [ ] A report records every claimed stage's two-round paired samples, oracle
  verdict, timing scope, uncertainty, target result, and relevant latency/RSS.
- [ ] Update streaming/ASOF/API/benchmark documentation and CHANGELOG for the
  actual delivered behavior. Regenerate changed OpenAPI/TypeScript together;
  unchanged project schema has no drift. `git diff --check` passes.
- [ ] Full Rust regression/90% coverage, Studio 85% coverage, and required
  cross-platform checks are assigned to CI. Pending CI is disclosed and does
  not establish merge readiness; no merge without explicit authority and green
  required checks.

Delivery follows issue PR dependencies: evidence/progress and safety coverage;
J1 and asynchronous compaction; A1/A2; retained/checkpoint benchmark coverage;
A3/A4 and J2a/J2b after critic review; then J3/A5. Independent steps can be
delivered without weakening prerequisite safety gates. Submit reviewable PRs
and measurement records under the user's authority; do not fold unmeasured
later stages into a claimed complete performance milestone.

## Open design gates

- Critic must approve the narrowly scoped Join v1 managed-recovery mechanism
  before J2b (FR12); current exact capability/fingerprint checks make this a
  real compatibility gate.
- Critic must confirm A4 fast-path capacity/owner choices reproduce FR16's v3
  inventory and bytes; if a shape cannot meet that contract, retain its fallback.
- Size/shard thresholds and scheduled duration for checkpoint-on cases are
  implementation choices supported by measured crossover and an exercised
  nonterminal epoch, not additional user questions.
