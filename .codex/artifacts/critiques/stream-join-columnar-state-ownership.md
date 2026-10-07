# Stream Join columnar ownership - Critic Critique

## Target and frozen evidence

- Requirements: [FR1–FR4 and FR11–FR12](../specs/stream-join-asof-acceleration.md).
- Proposal: [minimum owned-copy mechanism](../analysis/stream-join-columnar-state.md#proposed-minimum-owned-copy-mechanism-for-critic-decision).
- Earlier [J2 gate](stream-join-columnar-state.md) and locator review are historical, unchanged artifacts. Their native-only approval does not approve this locator source.
- Domain: [immutable Batch and checkpoint](../../../docs/introduction.md#the-basic-vocabulary), and [Join output materialization and recovery](../../../docs/streaming-guide.md#join-output-materialization-and-recovery).

The author froze `target/issue363-j2a-wip/locator-interim-review-v3/` before this
verdict. Independent stdlib hashing matched all thirteen receipt entries,
including all eight Rust files, analysis, commands and three recorded excerpts.
Receipt SHA-256:
`cb21e4efd37fe0e33f0cbab52f6c159a16b547128e9a0c2d0e9475de906e2bbf`.
Analysis: `728712f5936a9e39d42ab3a41ee710be4f0986492af2a79ae6258c0d867558ff`.
Commands: `75c800d29e61aff3ffd0eea6f9e5c72cdcbcfa1a4cba202c6fd6c3168c6ba9dc`.
The proof receipt under `target/issue363-j2a-ownership-critique/` records exact
source/dependency identities and checks; it contains no new native execution.

## Verdict

**Request Changes for frozen locator source. Approve the author's borrow-only,
prepaid core-copy direction with the frozen conditions below.** This is a
design gate for the next bounded implementation slice, not source approval,
completion of J2a, sparse-worker approval or a performance claim. The existing
approved native-key-only scope remains intact. Sparse replacement and V2 must
not proceed on the current caller-borrowing ownership premise.

## Blocking findings

### Positive capacity is not a complete-owner certificate

`join/columnar.rs:127` sums nonzero `Buffer::capacity()` and
`shared_payload` uses that sum to approve a caller-borrowing chunk. The locked
`arrow-buffer` **58.3.0** implementation contradicts that premise:

- `src/bytes.rs:106`: Standard returns allocation layout size; Custom returns
  only its supplied size, expressly permitting larger underlying capacity.
- `src/bytes.rs:234`: safe conversion from `bytes::Bytes` stores Custom(owner,
  visible length). `buffer/immutable.rs:528` exposes that conversion to Buffer.
- `buffer/immutable.rs:169`: custom allocation also records only supplied len.
  Its capacity rustdoc at line 201 is insufficient evidence against these
  implementations. Locked `bytes` **1.12.1** has safe `Bytes::from_owner`.

The author's actual session **10253**, exit **101**, demonstrates an owner
exposing 8 bytes while keeping a hidden 1 MiB allocation alive through Shared;
actual Shared credit is **1,816 bytes**. This is not repaired by canonical
schema rebinding, native concrete array types, pointer equality, reference
counts, full visible slices, or all-valid masks. `funded_owner()` repeating
the production formula is not an independent complete-owner oracle.

**Required correction:** a raw-input Shared constructor must refuse such
borrowing. Only constructor-controlled private provenance for core allocations
can authorize Shared. A caller-supplied boolean/token or a method taking
arbitrary ArrayRefs plus credit does not provide that provenance. Unknown
caller owners either remain in the existing Legacy boundary or are copied
through the approved narrow ingress below. They never enter detached owned
work under credit for visible bytes alone.

### Native controls are not completely prepaid

`payload_controls` at `columnar.rs:143` uses `1,024 + 256 * columns`.
The author's allocator session **1846**, exit **101**, observed mixed 128-column
requested allocation **peak = live = 113,380 > credit 111,484**. Canonical
schema/type ownership fixes address a different allocation family.

**Required correction:** replace this incomplete allowance with a checked
inventory of the constructors actually used: typed value/offset capacities,
buffer owner and Arc allocations, concrete array wrappers, RecordBatch column
vector, chunk owner, selection/locator descriptors, builder controls, and
temporary/replacement overlap. Include allocation rounding and any
reallocation overlap before allocation. Charge each independently allocated
owner; alias deduplication requires the private provenance. Locked private
Arrow controls require a defensible bound, not a larger unexplained constant.
The independent 3/16/128-column allocator check must compare real requested
live/peak allocations with actual guards and prove final allocation/pool zero.
The approved native index Entry128/Base1024 fees and FlatV1 fees do not change.

### Arrow value equality does not prove exact V1 row IPC

Locked `arrow-select/src/take.rs:494` drops raw string bytes belonging to null
slots. Builders can likewise replace primitive/null payloads. Locked IPC
`writer.rs:1807` writes physical validity and selected raw value/offset buffers;
the required V1 evidence is byte equality, not decoded equality.

There is a second counterexample even for **nonnull Boolean** payloads.
`arrow-buffer/src/buffer/immutable.rs:341` takes a byte slice for byte-aligned
`bit_slice`, preserving unused tail bits; an unaligned slice reconstructs bits.
IPC Boolean payloads and existing validity use this method. A source byte
`00000010` containing `[false, true]` emits byte `02` for its first one-row
Boolean slice. Packing only its first false value into a fresh bitmap emits
`00`. This is a static consequence of the locked code, not a newly executed
native probe. Reindexing rows changes alignment and can change those bytes.

**Required correction:** first copy payload scope excludes Boolean and every
column with `nulls().is_some()`, including all-valid/null-buffer identity
cases. A nullable schema with no actual null buffer remains eligible. Boolean
keys can still use approved native equality while their payload stays Legacy.
Broader masks/Boolean require a separate byte-preserving representation proof;
copying logical values or normalizing padding is insufficient. Do not secretly
change V1 framing to solve this problem.

## Conditions for the approved next slice

The author's seven-point proposal is implementable with these restrictions:

1. **Bounded borrow only.** Each copy/planning quantum touches at most **4 KiB
   of payload bytes and 64 cell/header visits**, then checks cancellation and
   yields cooperatively. The bounds are conjunctive; 64 rows times 64 columns
   is not 64 visits. Count offset/selection and wrapper-construction work too.
   Split long strings across byte quanta. Do not call whole-record `take`,
   concat, IPC, hash, zero-fill or realloc-copy outside those bounds and call
   it bounded. This bounds new work units, not allocator wall-clock latency.
   The original input is borrowed only inside `process_data`'s admission
   future; no worker, spawned task or escaped cleanup owns it. This preserves
   the existing input lifetime and does not claim to bound caller backing RSS.
2. **Prepaid core owners.** One privately constructed builder/buffer set per
   selected source record, with checked whole-result capacity and temporary
   overlap prepaid through existing runtime reservation. No one payload owner
   per row/quantum. No caller ArrayRef, Buffer, identity `take`, validity or null
   owner is shared back into this result. Reuse canonical operator-owned
   SchemaRef/types; keep caller metadata equal without retaining its owner.
   Admission borrowing can use that live operator owner. A worker that may
   outlive the operator must also carry a separately justified, fully funded
   metadata/type owner, or remove that lifetime extension before dispatch.
3. **First proven payload scope.** Existing native-key eligibility plus only
   Int16/32/64, UInt8/16/32/64, Utf8/LargeUtf8 and Timestamp payload columns,
   with every actual null buffer absent. Copy raw primitive bytes; copy each
   selected string's complete raw offset range and checked normalized offsets.
   Timestamp payload/key raw unit and canonical timezone remain exact; event
   time still uses the existing checked microsecond conversion. Boolean,
   Int8/Date, nullable masks, dictionary/nested and other shapes retain their
   existing Legacy support/errors. No public schema support is added/removed,
   and the pre-existing Int8/Date codec bugs are not silently fixed.
4. **Exact selected identity.** Classify using existing null-time, null-key,
   late precedence; consume all physical IDs, including drops. Pack only the
   selected admitted positions, keeping a paid mapping to original IDs,
   source positions, times and retain flags. Retainless rows needed by the
   current probe/output can be temporary; committed live/dirty membership is
   explicit. Preserve incoming output order and opposite `(time,row ID)`
   order, FlatV1 exact fees/caps, error precedence, and V1 per-row bytes. Do not
   encode formerly retainless failing Int8/Date rows on a new path. Denied copy
   funding or unproved shapes select existing Legacy without new validation
   failures. Never publish a partial copy or state before output acceptance.
5. **Persistent credit and cleanup.** Core chunk owners retain their real
   resident guard until every live/dirty/snapshot/worker alias dies. Buffers,
   metadata and descriptors drop before credit. Temporary construction credit
   releases only after its objects; cancellation drops uninstalled builders.
   Dirty/live aliases do not pay for one owner twice, but their newly allocated
   locator/container nodes need their own prepaid inventory. Admission has no
   detached attempt and cannot add a background automatic installer.

These conditions permit real columnar success on the 64-key integer fixtures
and nonnull string fixtures; routing all input to Legacy is only a containment
fix, not completion of the requested payload optimization.

## Existing hook mapping and later sparse proof obligations

- **Copy scratch, selection, constructors and resident capacities.** `native_lookup.rs:128::optional_credit` uses the existing incremental runtime pool; `datafusion.rs:419::incremental_reservation` supplies real guards. Prepay before allocation, preserve optional denial fallback, and attach resident guard to the new chunk. Logical FlatV1 charge is not physical funding.
- **Dirty and live aliases.** `join.rs:3795::commit_prepared` clones payload into `PendingOp::Upsert` and retained rows. The same chunk guard survives both; separately fund new sparse membership/control structures. `PendingLog::remove_upsert/clear` release only their own references.
- **Checkpoint snapshot owners.** `checkpoint_compaction.rs::InputOwners` owns the real `Arc<Vec<StoredRow>>`; its Drop releases input before its release signal. Existing `await_compaction_release` waits both owner release and real refund before mutable handlers. Do not put a bulk `Arc::make_mut` clone on Tokio or change exact V1 prepare→capture.
- **Sparse inputs and workspace.** Only certified core chunks may enter `OwnedCpuWork`; old chunks retain resident credit. A new Join-private calculation must prepay selected-index/mapping copies, constructors, replacement peak and cleanup controls through the existing pool. `reserve_compaction_workspace` is a hook precedent; its logical-row formula is not proof of sparse physical capacity. No new public budget or public gather rewrite.
- **Sparse replacement output.** Prepay replacement resident guard independently of scratch. `submit_observed_work` plus `ObservedTicket::finish` keeps attempt/output credit; `ObservedOutput::install` runs with that credit held. Installed chunks must retain their own resident guard when observed output credit refunds. Do not move bare arrays out then drop their sole funding.
- **Actual release and managed close.** `gather_work/cleanup.rs::AttemptCleanup::wait` requires absence from Active/Parked/Dropping and the real refund signal, even if the home expires. `retain_retirement` pins job retirement. Install cleanup tracking after accepted attempt and before its first cancellable pool await; retain it across dropped futures/reset/restore. Owner release notification alone is insufficient.
- **Cancellation and stale installation.** Existing checkpoint installation is performed by the successful exclusive operator future, with cancellation checked, not by a detached worker. Sparse work must use the same rule plus exact current chunk/history validation. All mutable handler/checkpoint paths must wait any sparse cleanup too; a new attempt must not overwrite a live checkpoint observer. Reset/restore may replace logical state but cannot discard pending cleanup or install an old result.
- **Public output.** `materialization::JoinOutput::ranges`, `emit_prepared` and independently materialized output chunks retain existing edge row/byte preflight and acceptance order. Dictionary GC/nested trial-allocation fallback remains unchanged. These existing logical edge checks are not a new output-allocation/RSS ceiling.

Sparse eligibility, thresholds and density bounds still need their separate
gate. Maintain incremental per-chunk live membership and mark only chunks
affected by actual eviction; no full-state scan on no-expiry progress. Count
dirty upserts, snapshot and worker references when judging reclamation, and
do not refund an old owner's allocation while any of them remain. Replacement
must update actual locators for the intended live/dirty set without mutating
immutable historical snapshots. The two existing sparse REDs remain unchanged:
large source slicing, and **196,896 > 4,096 bytes** after 8,191/8,192 evictions.

## Minimal independent TDD and handoff requirements

- Preserve raw constructor opaque-owner refusal RED until corrected; add the
  managed-copy success analogue: same hidden owner, actual private core chunk,
  Weak marker gone after input release, original row IPC/charges/order equal,
  actual credit and requested allocator peak/live bounded, final exact zero.
- Exercise identity selection and selective late/drop/retainless positions,
  physical-ID gaps, empty record/empty strings, nonzero source slices, all
  timestamp units/timezones, and mixed 3/16/128 flat columns. Compare every
  standalone V1 row IPC and the five frozen full inventories/continuations
  with unchanged Legacy. Keep Boolean/all-valid-mask/null-slot adversaries
  on actual Legacy and prove the public native-key behavior stays unchanged.
- Independently measure requested controls/value capacities at construction,
  input drop, dirty-only ownership, capture, eviction and final drop; test
  denied funding before and during construction, cancellation at a real yield,
  and no changed state/output/counters before commit. Reusing production fee
  calculation as an oracle or subtracting worker totals proves nothing.
- Before sparse source approval: retain both existing density REDs and real
  Pending/refund tests; test worker release while old/new dirty/snapshot owners
  remain, failure/cancel/drop/reset/restore, stale result rejection, failed
  admission, and worker output credit held through install. Include actual
  worker allocations in later paired allocation/RSS measurements.

The author reports schema-owner and direct-offset fanout RED→GREEN, and exact
V1/native behavior passed the reported subset. Four ownership/backing REDs
remain. Current locator Clippy/properties/full module/performance were not run
and are not inferred from prior native-only approval. This critic ran only
static reads and stdlib hashing, edited only this artifact and its proof
directory, and owns **zero running processes**. No source, spec, frozen
historical critique, remote state, build or native test was changed/run.
