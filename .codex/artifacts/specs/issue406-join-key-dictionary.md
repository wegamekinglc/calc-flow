# Issue 406: Join persistent key dictionary - Specification

Status: critic Proceed with caveats at `1901e458`; bounded specification
clarifications incorporated. Implementation evidence and final review are pending.

## Source and integration basis

- Issue: [#406](https://github.com/wegamekinglc/calc-flow/issues/406), phase 1.2,
  and [design note 1.2](https://github.com/wegamekinglc/calc-flow/issues/406#issuecomment-6097457069).
- Domain: [Introduction](../../../docs/introduction.md#the-basic-vocabulary),
  [supported boundaries](../../../docs/introduction.md#supported-boundaries).
- Rules: [AGENTS.md](../../../AGENTS.md#verification),
  [code-style](../../../.agents/skills/code-style/SKILL.md), and
  [specialist delivery policy](../../agents/README.md#shared-delivery-policy).
- Inspected committed base: `bb142320062995aeb318ad406c901288537fd766`.
  The sibling `feature/issue406-join-batch-admission` worktree contains the
  provisional `AdmittedBatch` integration surface; its
  `.codex/artifacts/specs/issue406-join-batch-admission.md` specification in
  that sibling worktree is authoritative
  for incoming ownership, selection coordinates and workspace preflight.
  Start implementation only after that source has a tested, committed and
  reviewed integration head; record that head in the implementation evidence.
- An independently reviewed codec correction described below is a prerequisite.
  Its final head must also be recorded; this specification does not silently
  treat the correction as existing behavior.

## Problem statement

Join deduplicates framed keys within each incoming batch. It still rebuilds key
frames for repeated values in later batches and maintains independent key lookup
tables in the left and right `NativeIndex` instances. Existing dense IDs identify
batch-local keys or one side's index slots; they do not identify a key throughout
the operator. Sharing those IDs without explicit lifetimes would misroute probes
when slots are freed, restored, or observed by an owned worker.

## Goals and scope

- Give equal keys on both sides one persistent, operator-local dense identity
  while they remain retained or leased by an active operation.
- Bind Arrow key types at batch/column boundaries; probe known keys without
  rebuilding their V1 framed bytes. Encode canonical bytes once when a new
  retained key identity is prepared, and reuse them for charges and history.
- Cover every type admitted by `supported_key_type`, every accepted composite
  combination, multi-record input, sparse selection and projected payloads.
- Preserve Join results, order, V1 logical charges, limits, checkpoints and
  existing ownership/cancellation behavior under an explicit funding policy.
- Establish one crate-private key-identity foundation suitable for later ASOF
  adoption. ASOF's existing right-state dictionary is evidence, not an API to
  adopt unchanged: it also owns right buckets, expiration heaps and ASOF state.

The reusable boundary concerns identity, ownership and leases. Equality codecs,
canonical bytes, byte ordering, sortable ranks and checkpoint formats remain
operator-specific. Join's V1 little-endian frames are neither ASOF's Arrow-row
identity encoding nor its sortable key rank; opaque arrival IDs are not ranks.
No ASOF migration or unused generic codec adapter belongs to this increment.

This increment changes Rust internals. It adds no public API, `BatchKind`, port,
configuration field, provider, Python stub, checkpoint layout or project schema.
It does not migrate ASOF, replace `StoredRow`, redesign time runs, expiration
maps or dirty logs, add chunk journals, change match-pair materialization, widen
parallelism, or change payload ownership policy. Phase 1.1 charge sums from
offsets, phases 1.3-1.5 and the remaining phases remain separate obligations.
Do not mark the complete shared foundation or #406 complete from this delivery.

## Required codec prerequisite

Construction currently accepts Int8, Date32 and Date64, but the committed
`key_value_bytes` implementation rejects negative Int8 through `u8::try_from`
and downcasts date arrays using Int32Type/Int64Type. The test
`test_legacy_int8_and_date_codec_behavior_stays_on_fallback` deliberately
records those retained-row `Internal` errors. They are accidental errors to
correct in a separate focused change, rather than an excuse to exclude three
accepted types from this increment.

Keep each V1 block as tag, u32 little-endian timezone length, timezone bytes,
u32 little-endian value length, raw value bytes. Int8 retains tag 2 and length 1,
using signed two's-complement bytes (`-1` is `ff`, `-128` is `80`). Date32 retains
tag 12 and length 4, read from Date32Type as native signed i32 little-endian.
Date64 retains tag 13 and length 8, read from Date64Type as native signed i64
little-endian. Date key values stay in their physical day/millisecond units;
event-time normalization does not apply. Previously successful encodings,
fees, fingerprints and wire layouts remain byte-for-byte unchanged.

The correction must first record focused failures for the intended new successful
values, replace only the accidental-error assertions, and verify retained state,
matching and recovery. Null/late/non-retained handling and first-error ordering
remain as specified by phase 1.1. This prerequisite needs final reviewer approval;
no user algorithm decision is needed.

## Functional requirements

### Key identity and equality

- **FR1 - One key namespace.** One active Join operator uses one key namespace
  across both sides. Equal left and right values resolve to the same identity;
  a side-local index slot is never interpreted as that shared identity. Another
  operator instance, a reset, or a restored replacement state has its own
  namespace. Handles are private and never influence observable output order.
- **FR2 - Exact equality.** With the already required identical Arrow types
  on paired columns, dictionary equality must equal complete V1 canonical-byte
  equality, including key-column order, lengths, empty values, embedded NUL,
  non-ASCII text, timestamp unit and timezone. Hash equality alone is insufficient.
  Forced collisions must still distinguish different values and composite
  boundaries such as `(a, bc)` and `(ab, c)`. Null-key rows remain dropped before
  dictionary access; a null elsewhere in a selected input array does not force
  admitted non-null rows onto the legacy framing route.
- **FR3 - Complete accepted types.** Cover Boolean, signed Int8/16/32/64,
  unsigned UInt8/16/32/64, Utf8, LargeUtf8, Date32, Date64 and Timestamp in all
  four units, with accepted timezone metadata. These are the 17 V1 tags, not
  17 newly introduced Arrow types. Include mixed/composite keys, long strings,
  sliced arrays and multiple record segments. Float, binary, dictionary-encoded
  or nested key types remain rejected where currently rejected; nested payload
  columns remain valid and do not change key eligibility.

### Batch and column processing

- **FR4 - Bound once, visit columns.** Consume the reviewed `AdmittedBatch`
  selection directly. Preserve dense probe ordinal, original physical row-ID
  position and current payload position as distinct coordinates. Key-type
  dispatch/downcasts belong to each record-segment/column binding, rather than
  a key-column loop repeated inside every row's framed encoder. Fixed-width
  single keys and short strings must avoid a per-row framed buffer; wide strings
  and composite keys must hash/compare selected typed columns without first
  rebuilding the complete V1 frame for each row. The implementer chooses concrete
  table/encoder representations within these reviewed constraints; this
  requirement does not mandate a hasher.
- **FR5 - Read-only probe.** Probing returns a known handle or an explicit
  unknown result per admitted position. Unknown probe-only keys create no live
  entry, canonical frame, retained reference, free-list entry or persistent
  allocation. The result order is admission order. Empty opposite state and
  an ended opposite ingress are included: keys with no future retention need
  no canonical frame merely because they passed admission.
- **FR6 - Retention preparation.** Only the retained selection may prepare new
  identities and canonical bytes. Repeated new values in the same operation
  share one staged canonical owner; known values across subsequent batches and
  opposite sides reuse the live owner. Reuse canonical length for V1 charges;
  do not encode again for sizing, admission fees, index lookup, checkpoint
  logging, declined owned capture, serial retry or output routing. A new lifetime
  after full eviction may encode the value again; “once” is per identity
  lifetime, not an unbounded global cache of every value ever seen.

### Existing state and lookup integration

- **FR7 - Retain the compatibility envelope.** `StoredRow` may continue to own
  a canonical `Arc<FramedKey>` for history and exact-fee validation. Repeated
  rows must alias the dictionary's canonical owner while its identity is active.
  Private handle storage is allowed but all newly introduced capacity is funded;
  changing `size_of::<StoredRow>()` must not redefine the frozen V1 row fee.
  Pending Upsert and Tombstone operations own canonical bytes or an equally safe
  immutable lease, never a naked recyclable ID.
- **FR8 - Index by identity.** Left and right `NativeIndex` keep their existing
  per-key time-qualified runs and `(event_time, row_id)` ordering. Dictionary
  identity resolution replaces repeated canonical-byte interning/hashing inside
  those indexes. Side-local run slots may remain an implementation detail with
  an explicit mapping from the shared identity. Native count and fill, estimated
  visit gates, owned probe capture and serial continuation use the same admitted
  handle results; a declined capture must not resolve or encode them again.
  Do not maintain a second independent content dictionary in either side index.
- **FR9 - Derived acceleration and refusal.** The dictionary and native indexes
  are derived acceleration, not a new durable source of truth. Their checked
  grant refusal must preserve the existing batched SQL equality route and its
  one-row zero-headroom acceptance. No finite default pool is introduced.
  A partial dictionary must never turn an absent entry for a retained key into
  a false “no match”; use the established SQL route for an operation lacking a
  complete admitted native view, or rebuild a complete view under a later grant.
  Rebuild/re-enable work is explicit, funded, bounded by the scheduling contract
  and separately counted, not silently repeated on every ordinary batch.

The ordinary funded route must cover all FR3 types without a type/nullable/
multi-record cutoff. A grant-refused compatibility route must have an explicit
reason counter and focused test, retain the phase 1.1 batch admission and Arrow
column selection, and introduce no new unfunded heap dictionary or row adapter.
Existing legacy canonical-row encoding under genuine optional-grant refusal is
an explicitly recorded compatibility exception, not evidence of the new path's
structural acceptance. A claim that this exception is eliminated requires its
own funded vectorized implementation and tests. The implementation report must
name each retained exception and its existing maintained workload coverage;
unexplained fallback is blocking. Non-budget errors, cancellation and deadlines
must propagate rather than trigger a grant-refusal retry.

The legacy exception inventory starts with `join.rs::retained_rows`'s absent-
native-key branch: `encode_join_key_columns_v1` allocates a Vec, converted through
`columnar.rs::FramedKey::from` and wrapped in an Arc without independent workspace
funding. It exempts only that existing canonical-owner allocation under genuine
grant refusal, not new dictionaries, handle tables, lease owners or row adapters.
Record final helper names, allocation sites and framing counters after integration.
The existing `paid_key_pressure_case` control matches through SQL but does not
commit retained rows; it alone cannot establish production retention parity.
No maintained suite case exercising this refusal exception has been identified
by this specification. Identify one with an actual refusal witness, or supply
a bounded correctness/counter variant in the maintained harness before claiming
acceptance; a normal funded case or unit test alone does not close that gap.

### Transactions, lifetimes and recycling

- **FR10 - Prepare, then commit.** New keys, canonical owners, retained-count
  deltas, native index additions and required credits are prepared without
  publishing new dictionary identities or counts in committed operator state.
  Commit them with retained rows only after every output chunk has emitted
  successfully. Preflight fallible capacity/refcount/ID arithmetic before any
  output. Partial-emission failure can leave already emitted chunks as today,
  but must leave no committed incoming rows, live dictionary entry or retained
  count from the failed operation; do not promise rollback of escaped output.
- **FR11 - Exact references.** An active entry's retained count equals its
  committed rows across both sides. Each successful retention increments once;
  eviction, clear/reset and replacement-state installation retire the matching
  old references exactly once. Repointing a swapped row or replacing/projecting
  a payload does not change key references. Cancellation, deadline, a dropped
  preparation future, state/match refusal and collector failure discard staged
  references without affecting committed counts. Checked overflow/underflow
  must not wrap, alias a sentinel or leave a partially published entry.
- **FR12 - Recycle safely.** Zero retained count removes the key from new live
  lookup once no staged transaction requires that identity. Do not recycle its
  slot while a worker, captured result or other operation can still dereference
  the old identity. Such operations hold an explicit immutable namespace/entry
  lease whose funding survives through actual input/result release. Reuse is
  allowed after the last such lease settles. If an implementation chooses earlier
  slot reuse, it must additionally prove generation-checked identities and
  immutable old views; unversioned raw-u32 aliasing is forbidden. Canonical
  byte-only history owners may outlive the slot without keeping it live, because
  they never resolve a recyclable ID. Re-retention during an active lease must
  have unambiguous equality and must not produce two simultaneous live identities
  for the same key in one namespace.
- **FR13 - Bounded lifetime.** Repeated disjoint high-cardinality batches followed
  by full eviction must not accumulate canonical keys or ever-growing logical
  slot directories. Reuse freed slots; unused capacity may retain its declared
  funding until reuse or bounded compaction. Fully drained empty state releases
  dictionary backing, except explicitly bounded reusable control storage.
  Backing retained by real worker/history leases is reported separately from
  live entries and released at their existing retirement boundary. No whole-input
  Arrow owner, unrelated payload column, unselected row or original string buffer
  is kept solely to remember a canonical key. Canonical storage is independently
  owned; dropping an expired batch must not leave its pooled scratch reservation
  pinned by an unrelated surviving key.
- **FR14 - Checked funding.** Reserve before allocating every new table,
  directory/free-list/refcount/handle vector, typed-hash workspace, canonical
  byte owner, lease/control owner and replacement. Cover allocated capacity,
  allocator/control layouts and old-plus-new reallocation overlap. Transfer
  credit with captured inputs, results and surviving canonical owners; splitting
  scratch and persistent credit must not retire a reservation while the owned
  allocation remains live. Rollback restores the prior committed view and
  refunds owned temporary credit after actual worker retirement. Preserve the
  existing legacy state-parent contract; this increment does not claim that
  old Rowed payloads suddenly become funded state. FR9's specifically inventoried
  legacy canonical allocation is also exempt from a new mandatory grant.
  State which reservation covers each canonical allocation separately from
  table/handle backing and scratch. Canonical credit follows its byte owner
  through retained rows and pending history, even after its entry or namespace
  retires; slot recycling cannot refund live canonical bytes or pin unrelated
  scratch. Observe intermediate retirement, not only final pool zero.
- **FR15 - Bounded work and errors.** Cooperative checks/yields must cover key
  binding, selected-value hashing, collision chains, wide values, new canonical
  construction and rebuild/restore work. Charge actual visited rows/columns/
  bytes to the existing quantum; a long single string or collision chain must
  not evade the bound. No shared cancellation mutex per row. Exhaustion of the
  chosen dense-ID domain, a reserved unknown sentinel, checked length/capacity,
  reference count or generation must fail/decline through an explicit typed
  path before publication and output; never `as`-truncate, wrap or panic on
  caller-controlled scale. Use small test domains/hooks to reach boundaries.

## State, checkpoints and compatibility

Keep inclusive Join time bounds, ingress-specific null/late handling and drop
metrics, every-physical-row ID consumption, and emission order `(incoming
position, opposite event time, opposite row id)`. Preserve external schemas,
consumer projection, project fingerprint, logical names, full V1 fee sums and
state/match-limit outcomes and counters. Dictionary physical deduplication does
not reduce `retained_bytes`: each retained row still carries the complete V1
key-length term and its logical payload fee, including omitted projected columns.
Key workspace refusal remains distinct from state/match limit failure.

Runtime handles are not wire identities. Existing layout 1/2/3 readers, current
full/projected writer selection, codec versions, managed manifest version,
projection descriptors, `charge_mode` and `HistoryIdentity` remain unchanged.
Validate complete historical payload/key/time/fee facts, including overwritten
and tombstoned records, before deduplicating the candidate's live keys. In
particular, checkpoint_v2/row.rs currently re-encodes a historical payload's
canonical key to check `locator.key`; retaining that validation is required and
is outside the ordinary-admission “once per distinct key” counter boundary.

After validation/migration, a candidate may coalesce live canonical owners
across both sides and build fresh private handles under checked funding. A
refused optional acceleration build keeps a valid compatible restored state
on the explicit SQL route; it cannot publish an incomplete native view.
Install the complete candidate atomically. Cancelled/invalid restore leaves
the prior dictionary, state, writer and pending history intact. Rebuild handle
numbers may differ after restart; results, row IDs, logical fees, history bytes
and validation requirements may not depend on them. Existing pending/cached/
owned checkpoint work must remain interpretable after eviction, ID recycling,
checkpoint capture, release, retry and restoration.

## Inputs and outputs

| Name                | Type / surface                      | Constraints                                                       |
|---------------------|-------------------------------------|-------------------------------------------------------------------|
| Admitted keys       | Selected Arrow columns              | Non-null admitted positions; reviewed batch selection coordinates |
| Key schema          | Identical paired Arrow DataTypes    | Existing 17 V1 tags and arbitrary accepted composite order        |
| Retained selection  | Batch retention bits                | Only selected retained candidates may prepare new entries         |
| Probe identities    | Private dense handle or unknown     | One result per admitted ordinal; namespace and lease apply        |
| Canonical owner     | Independently owned V1 key bytes    | One owner per active identity; no borrowed input payload owner    |
| Retained references | Checked counts over both sides      | Exact committed row count; staged deltas remain private           |
| Dictionary capacity | Checked optional workspace credit   | Domain/capacity/overlap covered before allocation                 |
| Historical keys     | Existing checkpoint canonical bytes | Validated without serializing private dictionary handles          |

## Acceptance criteria and focused TDD

Before implementation, add focused failing controls for repeated keys across
two batches and both sides: the current per-batch interner must visibly encode
again. Record actual expected failures before changing behavior. Add the
following controls in existing focused Join modules or new directly adjacent
test modules; test-only reference row fixtures must not select a production row
adapter.

- [ ] **AC1 - Persistent aliases.** Retain repeated equal keys in separate
  multi-record batches, on both sides, with unequal event-time order. They have
  stable shared handles/canonical owners while live. After the first birth,
  ordinary admitted/probe processing adds zero canonical key encodings and
  zero legacy arena frames; result values and sorted match order agree with
  the existing oracle.
- [ ] **AC2 - Distinct work.** A funded batch containing N retained rows and D
  previously unknown distinct keys creates D canonical frames, not N. This
  holds for fixed-width, short string, long string and composite representatives.
  Probe-only new values create zero canonical frames/entries. Counters cover
  sizing, fee preparation, capture decline, serial continuation and commit;
  resetting a counter between those stages cannot hide duplicate work.
- [ ] **AC3 - Column dispatch.** Doubling selected rows with the same record
  segmentation/key schema does not double type-binding/downcast counts.
  Record counts for each bounded pass; column-value visits may grow with rows,
  while framed-byte construction only grows with new retained identities.
  Known-key index count/fill adds no canonical content hashing or frame rebuild.
- [ ] **AC4 - Equality and collision.** Compare identity equality against an
  independently constructed V1-byte oracle for every tag, signed boundaries,
  unsigned maxima, dates on both sides of zero, all timestamp units/timezones,
  empty/1/8/9-byte/non-ASCII/NUL/long strings, and composite boundary collisions.
  Force equal hashes; distinct canonical keys never share an identity or match.
  Include sliced arrays and nullable physical input with a non-null selection.
- [ ] **AC5 - Semantics and fees.** Sparse null/late/retain masks, dense selections,
  incoming left/right, empty input and projected/nested payloads preserve physical
  ID gaps, timestamps, metrics, full V1 per-row fees, output schema and order.
  State/match refusal occurs before output and commits no key changes; reference
  tests use actual full fees rather than dictionary allocation sizes.
- [ ] **AC6 - Refusal.** Refuse dictionary build/growth, native run/index growth,
  owned capture and later submission independently. Assert the explicit decline
  reason, batched SQL result parity, preserved known canonical owners, no partial
  live-key coverage exposed to native lookup, and existing one-row zero-headroom
  acceptance. Genuine cancellation/deadline/non-budget failure does not retry.
  Add an actual production preparation/emission/retention-commit control under
  dictionary refusal with exact fees and cleanup. Name/count every surviving
  legacy framing exception and its maintained case, closing the FR9 coverage gap.
- [ ] **AC7 - Publication boundary.** Cancel or drop the future during new-key
  preparation, while an actual owned probe is held at its input gate, and at its
  result-release gate. Fail a collector after an emitted chunk. Committed rows,
  key entries/counts and pending additions remain unchanged by the failed
  operation; already escaped chunks follow the existing job-abort contract.
  Retry on a fresh healthy operation produces the same identities/fees/results.
  Exercise cancellation during a long single key value and a forced collision
  chain, rather than demonstrating cooperative work only on many short rows.
- [ ] **AC8 - Eviction and leases.** Keep the same key on both sides, evict either
  side first, then the last row. Retained counts follow actual rows; a worker
  holding the old identity cannot observe a newly recycled key. Hold a byte-only
  tombstone/history owner across recycling and prove it still serializes its
  original bytes. Re-retain the same value while an old lease is active.
- [ ] **AC9 - Bounded ownership.** Repeated unique-key batches with complete
  eviction reuse slots and do not grow the logical directory with cumulative
  arrivals. Weak input owners and unselected payload/string buffers become
  unreachable at the expected boundary. Allocation/credit observations cover
  transient replacement peaks and surviving canonical owners; after dropping
  the operator and settling owned work, all owned pool credit returns to zero.
  The cross-batch shared-owner control also observes credit retirement after
  each batch expiry, byte-only history release and actual worker result release.
- [ ] **AC10 - Exhaustion.** A reduced test ID/refcount/generation domain and
  checked-size fixtures reach each overflow without huge allocation. No sentinel
  alias, false match, panic, wrapped count, output or partial publication occurs;
  prior committed state remains usable and funded.
- [ ] **AC11 - Recovery.** Authentic layout 1/2/3 recovery and full/projected
  migration retain strict historical validation. Duplicate live keys across
  restored sides coalesce safely; a subsequent sparse physical-ID-gap batch
  matches correctly. Corrupt a historical key later tombstoned and still reject
  it. Use different old and candidate rows to distinguish three assertions:
  cancelled/invalid restore preserves the old installed rows, dictionary,
  writer/history identity and pending operations; successful validated restore
  with refused optional acceleration installs the candidate's rows/fees and
  writer/history/pending state on the SQL route; a later refused re-enable
  preserves that installed restored state. No assertion may substitute the old
  rows for a successfully installed SQL-only candidate.
  Capture/rebase/release/retry with old canonical owners across slot reuse does
  not serialize private IDs or alter history identity/wire contracts.
- [ ] **AC12 - Shared boundary.** The final review identifies one reusable
  crate-private key-identity implementation and confirms that Join has no
  parallel left/right content interner. ASOF source, behavior and checkpoint
  formats remain unchanged in this increment; no unused ASOF adapter or future
  public scaffolding is added merely to reserve the boundary.
- [ ] **AC13 - Checks and documentation.** Named affected unit tests, core
  lib/test Clippy, necessary formatting, changed/new-function CCN at most 8,
  `git diff --check`, and generated project-schema no-drift pass. Update the
  relevant internal execution/working-set documentation where behavior changes.
  Full regression, cross-platform and combined Rust 90% coverage belong to CI.
  Final specialist source review is required; pending/absent CI is not green
  and does not authorize merge.

Counters establish execution structure, not speedup or whole-process memory
claims. Preserve the eight original issue targets: Join 1M ≤60/15 ms, Join 10M
≤600/150 ms, retention scaling ≤4.4x, capture/restore ≤0.1/0.2 microseconds per
row, handler latency during compaction ≤2x steady at p50/p95/p99, ASOF 1M
≤100/30 ms, ASOF 1,024-row overhead ≤2x projection floor, and Join 1,024-row
callback cost ≤0.05 ms. This specification certifies none of them. Any later
performance task must preregister the maintained workload subset, baseline and
candidate, fixed samples and total budget, use sealed builds and the maintained
paired evidence rules, and respect prior stopped acquisitions. Do not create a
benchmark framework or run timing measurements to validate this specification.

## Source inspection and handoff

Read-only source inspection covered these paths on the committed base:

- `crates/calc-flow/src/operator/join.rs`: StoredRow/PendingOp/RetainedRows,
  preparation/emission/commit, key schema validation, supported types,
  retained_rows, V1 fee computation, canonical framing and key-value codec.
- `crates/calc-flow/src/operator/join/{native_lookup,native_dictionary,key_arena,borrowed_key,columnar,sql_key_scratch,state_projection}.rs`:
  batch interning, per-side dense runs, exact equality and funding boundaries.
- `crates/calc-flow/src/operator/join/data_work/{inputs,process}.rs`:
  owned input lifetime, visit gate, declined-key reuse and result recovery.
- `crates/calc-flow/src/operator/join/checkpoint_v2/{candidate,row}.rs`,
  `checkpoint_v2/candidate/migration.rs`,
  `checkpoint_v2/writer/identity.rs` and `state_projection/admission.rs`:
  historical validation, candidate funding, layout migration and history identity.
- `crates/calc-flow/src/operator/join/tests/{j2a_dictionary_tests,native_index_allocation_tests,sql_key_scratch_tests,columnar_state_tests}.rs`:
  collision/owner/expiry, paid-scratch refusal and accidental codec-error controls.
- `crates/calc-flow/src/operator/asof/state/key_dictionary.rs`,
  `state/key_dictionary/entries.rs` and `state/string_keys.rs`: existing
  ASOF-specific dense storage, compaction/funding integration and string batching.
- `crates/calc-flow/src/operator/asof/{identity,identity_compare}.rs`:
  operator-specific Arrow-row identities and sign-adjusted big-endian values.

A bounded static search of `benches/stream_join_perf.rs`,
`benches/stream_join_materialization.rs` and their `scripts/benchmark_suite/`
adapters did not identify a witnessed optional-dictionary-refusal suite control;
this is a coverage gap to resolve, not evidence that every harness lacks one.

Provisional sibling inspection additionally covered `join/admitted.rs`,
`join/key_arena.rs`, `join/data_work/inputs.rs` and the phase 1.1 specification.
The reviewed implementation must rebind these requirements to the final batch
admission head; these reads do not approve its current work in progress.

## Open decisions and gates

There is no unresolved user-facing API choice. The bounded critic verdict is
Proceed with caveats. The implementer records the concrete shared internal
boundary, handle/lease recycling proof, per-owner reservation inventory and
grant-refused legacy framing coverage against FR9; final source review verifies
that evidence. These are implementation choices constrained above, not permission to
exclude accepted types or introduce an unreported row route. Implementation
handoff is gated on the reviewed phase 1.1 and codec prerequisite heads; final
phase 1.2 acceptance remains gated on the focused evidence and specialist review.
