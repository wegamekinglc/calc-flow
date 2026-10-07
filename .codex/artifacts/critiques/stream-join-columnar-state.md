# Stream Join columnar state - Critic Critique

## Target

- [Acceleration spec](../specs/stream-join-asof-acceleration.md), FR1–FR4 and
  FR11/FR12; [Introduction](../../../docs/introduction.md), immutable Batch
  ownership and compatible recovery.
- Static production baseline: main
  `eccb26973811bc476f0944b977ddedf8564b0237`; the key codec was also inspected
  at pre-365–370 `49d346df`.
- No source changes, build, tests, measurements, or remote operations were
  performed. J1.6's unsealed implementation is not an approved dependency.

## Verdict

**Approve with the bounded implementation conditions below.** Columnar state
and native equality are viable for the proven subset; all declared key types
must retain their current public support and success/failure behavior through
fallback. The managed read-v1/write-v2 mechanism is approved with semantic
capability 1 and unchanged exact identity gates. This does not approve a claim
that every declared type has native coverage or that performance targets have
already been reached.

## Findings

### Blocking Issues for an unrestricted replacement

- **The existing V1 key encoder is not total over declared supported types.**
  [join.rs](../../../crates/calc-flow/src/operator/join.rs),
  `supported_key_type` and `validate_key_pair_types`, admit Boolean, signed and
  unsigned 8/16/32/64-bit integers, Utf8/LargeUtf8, Date32/Date64, and timestamps
  in all units/timezones, including composite tuples. Paired Arrow data types
  must be exactly equal; no dictionary, float, decimal, binary, or nested key
  is admitted. Event-time columns separately require timestamps with UTC or
  absent timezone, and checked conversion to microseconds.

  However, `key_value_bytes` uses `u8::try_from` for Int8 and downcasts Date32/64
  as `PrimitiveArray<Int32Type/Int64Type>`. Negative Int8 therefore returns
  `Internal`; ordinary Date32/64 arrays fail the primitive downcast. These
  branches and `primitive<T>` are identical at `49d346df` and `eccb2697`:
  they are pre-existing defects, not regressions from PRs 365–370.

  `retained_rows` encodes only `retain = true` rows, after the SQL probe. Encoding
  all admitted probe rows would introduce failures on formerly successful
  non-retained input. `validate_restored_row_payload` recomputes that same
  charge/key codec on V1 restore: adding a generic reader does not make a
  fabricated nonempty negative-Int8/Date snapshot valid. The old release
  could not normally capture those retained rows.
  - **Required boundary:** Int8 and Date32/64, and any composite containing
    them, keep the legacy SQL plus row-state path for this delivery. Do not
    silently cast dates or fix negative Int8 within J2. A separate correctness
    change with its own tests is prerequisite to claiming complete native
    support for those declarations.

- **V1 byte parity can conflict with physical payload compaction.**
  [row_ipc.rs](../../../crates/calc-flow/src/operator/join/row_ipc.rs) writes
  each stored row as a standalone stream; dictionary/nested schemas retain
  the fresh `StreamWriter` path. Compacting/remapping a dictionary or changing
  slice/null-buffer representation can change IPC bytes despite equal values.
  [join.rs](../../../crates/calc-flow/src/operator/join.rs) includes those exact
  bytes in `CFJOIN1\0` bases and `CFJDLT1\0` dirty upserts.
  - **Required boundary:** J2a may use columnar retention/compaction only where
    same-input row IPC, metadata, charge, identity order, and dirty-op sequence
    are proven byte-identical. Dictionary/nested/unproved shapes retain the
    original row-state and writer path. Output dictionary garbage collection
    remains mandatory and separate from state/checkpoint representation.

### Significant Concerns

- A bound on logical state charges is not a bound on retained Arrow backing,
  worker input ownership, or RSS. Native state must not keep arbitrarily large
  source slices under one small live row and call that memory bounded.
- Per-key native range order must be `(event_time, physical row ID)`, including
  legal out-of-order insertion. Hash/dictionary order cannot define output.
- Checkpoint family, public operator identity, private layout, and symbolic
  declaration version are distinct; changing one must not impersonate another.

### Minor / Style Notes

Keep changes in the native built-in stream Join. Batch SQL/DataFusion behavior,
project schemas, public source-control construction, and UDF contracts remain
outside this optimization.

## Approved J2a scope

Native equality is eligible for Boolean, Int16/Int32/Int64, UInt8/16/32/64,
Utf8/LargeUtf8, Timestamp, and composites made only of those types. Use the
existing framed V1 key bytes, whose type/unit tag, timezone, and value-length
blocks make equal non-null tuples byte-equal within an exact paired schema.
Strings compare exact UTF-8 bytes, without normalization or collation;
timestamp keys compare raw values in their declared unit, not event-time
microseconds. Hash hits still compare complete bytes. Little-endian key bytes
are an equality encoding, not chronological or signed-value sort order.
Preserve exact schema/column/ingress validation and existing error categories.
Keep the public DataFusion requirement while supported shapes still use the
SQL fallback; removing one private query does not justify changing the graph's
engine/capability identity.

A retained chunk owns one immutable admitted payload and columnar locators for
live rows, physical row IDs, converted event times, canonical keys, and frozen
charges. A per-key index resolves those locators in `(time, row ID)` order;
native probing applies inclusive bounds in checked i128 arithmetic and emits
in incoming admitted-position order. This is a bounded counter-proposal, not
a serialization specification. Do not expose transient chunk IDs publicly.

Reserve a row ID for every physical input row before null-time, null-key, then
late classification. Dropped rows still advance IDs; duplicate keys remain
distinct rows. Own-side watermark equality stays on-time. Retain/evict decisions
continue using the opposite accepted progress and strict existing expiration
inequality. Do not derive progress from chunks; standalone restore has no
observed ingress progress until supplied, whereas managed restore restores
manifest progress before startup acknowledgement.

Keep exact V1 per-row charges and prospective decisions from
`state_row_charge_with_key`/`logical_cell_charge`, including null, variable,
dictionary-selected value, and nested charges. Native backing/index/scratch
accounting is separate evidence; it cannot alter the recorded logical charge
or relax/relabel `JoinStateLimitExceeded`. Filter by time before enforcing the
match cap. Preflight the complete output sequence and every output range before
emission; retain the generic nested/dictionary trial-allocation boundary and
`concat_output_column` garbage collection, including a singleton dictionary.
Commit input state/counters only after all chunks are accepted; already accepted
outputs retain existing cancellation/replay semantics.

J2a still reads/writes layout 1 with the same metadata, segment names,
`CFJOIN1\0`/`CFJDLT1\0`, key bytes, standalone row IPC, canonical base ordering,
and pending-op order/coalescing. An ephemeral row view at capture is acceptable;
serializing a whole new columnar chunk into V1 is not. Existing frozen fixture
bytes must remain exact. Dictionary/nested payloads may still use native probing
for eligible keys while their retention/output/checkpoint stays generic.

## Backing memory and owned work

Reclaim chunks once all runtime, pending-delta, checkpoint-snapshot, and worker
references finish; zero live rows alone is not proof of released ownership.
For proven known backing, record actual owner capacities/aliases and compact
sparse chunks with an explicit deterministic policy bounding retained backing
relative to live data plus the largest active chunk. The concrete threshold
and measured bound must be recorded; do not promise an RSS ceiling from V1
logical charges. Report oversized backing retained by legacy fallback separately.

Unknown external owners, dictionary/nested shapes, or copies without proven V1
byte parity remain legacy retention shapes. Visible `Buffer::capacity`/length
cannot prove an external owner's full allocation. Do not detach an unknown
large source owner on a worker and fund only its visible slice. Broader backing
policy or support changes require separate correctness review.

Sparse replacement workers must hold funded old inputs, full known backing,
descriptors, indices, temporary and replacement allocations until native work
and actual refunds finish. Use the established bounded owned-work path and
existing runtime resource boundary, not raw/unbounded `spawn_blocking`, a new
public budget, or a logical-charge substitute. A denied reservation keeps the
original valid state/funded ownership and the reviewed fallback; it cannot
install partially compacted references or report nonexistent reclamation.
Such a fallback must be identified separately from the proven bounded
columnar regime; a failed compaction cannot silently count toward its memory
bound. Do not bypass funding merely to avoid a fallback.

J1.6 must seal before rebasing this work. Its pending design uses immutable
retained snapshots for bounded asynchronous base preparation, then installs
only while the observer still owns the operator; synchronous capture handles
dirty/carried state. J2 must preserve that ownership/prepare→capture boundary,
stale-completion rejection after reset/restore, and the final true-refund wait.
Do not put `Arc::make_mut` whole-state cloning back into Tokio handlers or
replace a snapshot owner before the native attempt/escaped output credit has
actually released. No unsubmitted cleanup-observer API is assumed approved here.

## Approved J2b migration mechanism

Keep `PipelineBuilder` Join capability `state_version = 1`, reserved capability
metadata, public built-in `stream_join@1`, project configuration, and semantic
fingerprint inputs unchanged. Only Join's private writer changes to layout 2;
its private decoder accepts validated layouts 1 and 2. The exact decoder in
[pipeline/stream.rs](../../../crates/calc-flow/src/pipeline/stream.rs) and
fingerprint in [pipeline/compile.rs](../../../crates/calc-flow/src/pipeline/compile.rs)
stay strict. No fingerprint rewriting, migration aliases, ignored fields,
family-wide version fallback, or global decoder relaxation is permitted.

Version 2's normative framing/inventory must be frozen before implementation:
canonical columnar base plus incremental dirty references, stable deterministic
IDs/order, explicit schemas and validated row locators/charges, strict bounded
length/index parsing, and unknown-version rejection. Support valid V1 generic
payload state as well as the columnar subset; malformed refs cannot be hidden
behind fallback. After V1 restore, write only valid V2 state on the next capture.
Old binaries are not promised forward readability of V2 captures.

[manifest.rs](../../../crates/calc-flow/src/state/manifest.rs) validates exact
pipeline identity and participant sets; [transaction.rs](../../../crates/calc-flow/src/state/transaction.rs)
revalidates referenced segment bytes at recovery. [runner.rs](../../../crates/calc-flow/src/runtime/streaming/runner.rs)
checks capability, loads/decode snapshots, and restores ingress/output-frontier
state through the existing data gate. Preserve lineage, source identity/cursor,
sink delivery proof, static input, schema and UDF-version checks. Parse and
validate replacement state entirely before installing it. A higher corrupt
manifest still fails instead of silently falling back to an older cut.

## Minimum TDD and authentic compatibility fixture

1. Compare native and original SQL/row-state paths for every eligible scalar
   and composite, null-drop precedence, physical IDs after dropped rows, equal
   timestamps/duplicates, inclusive bounds, extreme times, ns keys distinct
   within one microsecond, legal disorder, both arrival orders and partitions.
   Include Int8/Date retainless probes and retained failures as fallback
   coverage. Preserve unsupported-key construction rejection.
2. Retain order-sensitive generated arrival schedules and random checkpoint
   cuts; compare exact output sequences/status and resumed continuation.
   Exercise match/state/counter/single-row and range preflight failures plus
   partial-send cancellation, dictionary garbage collection and nested output.
3. Compare all frozen V1 captures, not only decoded values, across dirty deltas,
   base compaction, metadata/null-buffer/dictionary schemas, slice offsets,
   restore and continuation. Record a real failing native/columnar complexity
   test before implementation; already-green parity additions are coverage.
4. Prove sparse/backing accounting with known aliases, large slices, unknown
   owners, pending/snapshot references, cancellation/reset and blocked native
   workers. Assert inputs remain funded until genuine attempt/output cleanup,
   no stale install, and no early ownership/refund claim.
5. Before any J2 codec change, capture an authentic nonterminal managed root
   using a sealed prior V1 release wheel/build with recorded revision/wheel
   hash. Use stable replay-capable sources and sink identities with both sides
   retained, physical-ID gaps, legal disorder, progress and output sequence.
   Trigger and await a durable checkpoint, cancel/drain without publishing a
   terminal cut, then preserve complete original manifest/state/source/sink
   files and hashes. Record project/bindings/UDF catalog, inputs, sink proof,
   cursor semantics and expected continuation; no rewritten fingerprint or
   checksum. Existing operator-only `checkpoint-v1.json` is insufficient.
6. A fresh J2b process must open that unmodified root, acknowledge restored
   progress, resume from exact source cursors, continue, publish a V2 checkpoint,
   and restart again in another process. Compare with uninterrupted output,
   respecting ordinary-sink replay or durable deduplication proof; do not claim
   exact-once for an ordinary sink. Archive the actual first and second roots
   and provenance rather than synthesizing V1 with the new writer.
7. Tamper graph/keys/bounds, paired schema, selected UDF version, source/sink
   bindings, lineage, capability metadata, segment checksum, framing, locator
   range/duplicates, row IDs and charges. Every case fails closed before
   replacement state/output installation or source data admission. Permit only
   the existing diagnostics-only runtime-config tuning differences.

## Studio, documentation, and measurement scope

The current public inventory is in
[capabilities.py](../../../python/calc_flow/capabilities.py): Join semantic
state version 1 and `state_layouts=(1,)`. At J2b advertise `(1, 2)` only after
both readers are proven, keeping semantic version 1. Studio's
[models.py](../../../web-ui/backend/src/calc_flow_studio/models.py) already
supports a sorted layout inventory distinct from semantic version.
[RunManager](../../../web-ui/backend/src/calc_flow_studio/run_manager.py)
selects the root by project ID and delegates native compatibility; preserve
that root and test same-project migration. Do not invent a layout-dependent
checkpoint directory or change symbolic declaration `stream_join@2` identity.
Update capability/symbolic recovery/Studio tests and normative format/recovery
docs. Regenerate OpenAPI/TypeScript only for actual schema changes; inventory
value changes alone do not justify generated-contract drift.

Measure lookup plus retained interval Join, small batches, checkpoint-enabled
capture/restore and compaction latency, actual allocations and peak RSS including
workers. Separate native and fallback coverage. Paired sealed measurements are
delivery evidence; estimates and this static approval prove no observed gain.

## Counter-Proposal and Questions for the Author

Stage native probing, proven columnar retention with exact V1 serialization,
then private V2 migration. Keep legacy shapes observable and correct instead
of combining codec bug fixes, backing policy changes, or public graph migration
with the optimization. No new user decision is required; route the actual V2
format and sealed J1.6 ownership interface to the specialists before coding
those dependent pieces.
