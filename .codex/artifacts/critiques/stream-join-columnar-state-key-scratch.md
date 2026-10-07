# Stream Join key scratch and metadata ownership - Critic Critique

## Target and frozen evidence

- [Owned-ingress analysis](../analysis/stream-join-columnar-state.md#new-sql-scratch-ownership-boundary-actual-red-design-unresolved), [previous ownership gate](stream-join-columnar-state-ownership.md), and [FR1–FR4 / FR11](../specs/stream-join-asof-acceleration.md).
- [Introduction: immutable Batch and compatible recovery](../../../docs/introduction.md#the-basic-vocabulary), and [Join materialization/recovery](../../../docs/streaming-guide.md#join-output-materialization-and-recovery).

The author finalized `target/issue363-j2a-wip/owned-ingress-partial-v4/` before
this verdict. Independent stdlib hashing matched **27 frozen files**, all
**9 live Rust files**, and the live analysis. The manifest's actual filename is
`rust-source-manifest.txt`, as specified by its receipt; no `.json` file is
assumed. Receipt SHA-256:
`6b0be5f5eb814f44bc774c5dbfc113328426c7a3be6023dcaaad98686a35ec19`.
Rust manifest: `ad66a695b80e8ba83f9479333ba93d8b91d62e49eea86f2f2db25e8976de0ba3`.
Analysis: `7a08ddfeb82a976e3d5593ecd63309183c4ea218a5eb64b825abe762fc61d2d2`.

## Verdict

**Request Changes for partial-v4 source. DesignApprove the two narrow
correction directions below, with their explicit lifecycle and TDD gates.**
This approves bounded private work on key scratch and constructor-proven
metadata; it does not approve the present source, a local-reservation-only
patch, a TaskContext-only keepalive, Sparse, V2, or a general execution
framework. Whole J2a remains incomplete. Previous native-only/source approvals
and the historical ownership critique remain unchanged.

The recorded owned-ingress corrections and focused GREEN results are useful:
17 selected columnar cases, five exact V1 captures/continuations, six native,
14 checkpoint, four property tests, and scoped Clippy. None closes the actual
SQL scratch RED or the schema bucket proof gap. Post-eviction density remains
the earlier **196,896 > 4,096 bytes** RED and is outside this design slice.

## Boundary 1: paid key scratch must survive every actual consumer

### What fails

`join.rs:4526::probe_key_batch`, `state_key_batch`, and `key_probe_batch` form
RowViews and concat single-row key arrays. Locked singleton concat can share
the original backing. Author session **37153**, exit **101**, verifies the
correct key value 7 before finding **32,864 > 4,096 bytes** for one key row
from a certified 4,096-row chunk. This is genuine retained backing, not a
timing result.

`CachedRetainedKeys` at `join.rs:2886` holds naked `RecordBatch`/row IDs, and
`opposite_state_keys` returns a naked batch clone. `sql_key_pairs` then hands
raw tables to `DataFusionRuntime::sql_validated`. `TableRegistrations::register`
at `datafusion.rs:1098` clones schema/batches into MemTable; physical plans,
streams and spawned producers can retain arrays independently of the source
RowView, operator or function-local guard. Deregistration is alias cleanup,
not proof of producer completion or array release.

The existing RED reads keys after dropping the operator and currently expects
pool zero. A correct **paid** escape must instead retain its actual credit
until the escaped key owner drops. Preserve the real backing/value assertions;
strengthen the funding assertion to distinguish paid ownership from premature
refund, rather than preserving that unsafe zero-pool expectation.

### Approved minimum private correction

- Introduce a Join-private paid scratch owner containing the actual copied key
  arrays, exact scratch schema/type/timezone owner, position/physical-row-ID
  buffers, cache validity IDs, constructor/container controls, and real
  reservation. Only its private core-copy constructor certifies complete
  allocation provenance. Accept supported existing native/core shapes first;
  unsupported/Legacy shapes retain their original SQL behavior and errors.
- Read selected key columns directly using payload columns/offsets. Copy even
  identity/singleton selections into constructor-owned buffers; full RowViews
  and singleton concat are not a detachment proof. Reuse the bounded admission
  quantum and checked allocation inventory, including builder/transient
  overlap. No public Batch change, dictionary/nested expansion or codec fix.
- Cache and temporary state/probe tables carry the owner, not just a cloned
  RecordBatch. Validate exact physical IDs on cache reuse, preserve original
  invalidation, and retain the existing 32 MiB cache admission policy. Omitting
  the cache is an optimization fallback, not a validation error. Charge copied
  positions/IDs and cache vectors separately from key buffers and logical V1
  fees. Any retained arrays/schema after cache invalidation still own credit.
- A tiny **crate-private DataFusion owned-input entry** is required unless
  the implementer proves equivalent ownership on the actual escaped buffers
  and all metadata users. It must accept private owners alongside the unchanged
  internal equality query/tables, propagate them into registered providers,
  physical input scans/streams and spawned producers, and return an owned
  result/decode boundary. A bare `sql_validated(...).await` surrounded by a
  local reservation does not meet this condition.

The minimum route must retain credit in the actual object clone graph, or
retain an explicit real-completion/refund observer for every escaping consumer.
Safe private buffer-owner wrappers are a possible way to make arbitrary
ArrayRef clones retain a funding lease, but only with constructor-proven native
input owners, their full capacities and wrapper-control inventory. The lease
must not own its own arrays and form a reference cycle. Funding a wrapper for
visible caller bytes does not certify opaque caller allocation. Schema-only
owners and empty results still need explicit lifetime treatment.

### Exact hooks and the missing guarantee

- Existing `datafusion.rs:419::incremental_reservation` and Join
  `optional_credit` provide real prepayment. They do not attach that payment to
  cloned arrays. Resident scratch funding must remain independent of temporary
  copy/plan/decode funding; no credit may be dropped before its objects.
- `datafusion_compact.rs::PaidSqlPlan` and
  `DataFusionRuntime::retained_sql_plan_sync` demonstrate a **synchronous**
  logical-plan owner with reservation. They do not retain credits for Join's
  asynchronous physical query or spawned task cleanup. Reuse the ownership
  pattern, not an unsupported claim that this already solves `sql_validated`.
- `execute_query` owns registrations, planned physical plan, metrics-plan
  clone and collected stream. The private route must protect all of them on
  planning/execution/decode error and dropped futures. Keep query-lock/alias
  cleanup ordering and existing selected UDF/session behavior unchanged.
- Locked DF54 supports `SessionConfig::with_extension` and
  `TaskContext::with_session_config`, but **TaskContext alone is insufficient**.
  `RepartitionExec` at `repartition/mod.rs:488–549` spawns futures containing
  input stream/channels without TaskContext. `HashJoinStream` has no retained
  TaskContext field. An input stream lease must continue into any batches
  surviving stream consumption, build/channel buffers, and cleanup tasks.
  Merely wrapping the returned root stream is also insufficient after it drops.
- A completed public query future, alias deregistration, stream Drop or task
  abort request is not actual release. Existing
  `job.gather_owner().retain_retirement()` can retain managed job cleanup
  ownership. Its guard must survive producer cleanup and release **after**
  the last input/output owners and their real refunds. Existing
  `AttemptCleanup` observes gather attempts only; it cannot be asserted to
  track DataFusion tasks that were never submitted through gather.
- Keep returned SQL result owners until pair decoding finishes. Prepay any
  newly optimized pair/decode vector and replacement overlap; otherwise keep
  the original validated decoding boundary outside the new optimized funding
  claim. This does not introduce a new pair cap or change match-limit/error
  precedence. Never expose a naked result with its only funding already gone.

### Funding refusal must preserve prior SQL acceptance

Optional scratch/cache/copy/decode admission denial must abandon that
optimization and invoke the existing validated SQL/decoding behavior, without
a new error, wider pool, changed V1 fees or altered public support. First drop
failed candidate objects, then their credits. If the fallback tables borrow
certified chunks, retain those existing chunk owners/credits through the same
actual DF consumer lifetime; returning to naked arrays is not safe fallback.
Legacy caller-owned shapes remain within the existing Legacy accounting
boundary, excluded from the new proven scratch/backing claim.

Optional paid cache/copy credit must not itself turn a formerly accepted SQL
query into a new memory-budget failure. Under pressure, drop optional caches
and candidate funding before old SQL fallback; if an optimized execution must
be abandoned for that reason, complete actual cleanup before an internal
read-only equality retry. Identify the typed internal DF budget error before
`datafusion_error` flattens it into text; do not classify failures by message
matching. Do not retry unrelated query/type errors, install
partial state, emit output or increment failure counters twice. If the private
owner route cannot preserve this refusal behavior without untracked consumers,
that specific hook design remains **Request Changes**; it cannot be bypassed
with a new rejection or an unbounded/untracked spawn.

## Boundary 2: fresh metadata ownership, not arbitrary HashMap capacity

`columnar/owned_copy.rs:111::metadata_inventory` infers bucket count from
`2 * (capacity + 1)`. Its source is an arbitrary canonical Schema metadata map.
The author's Rust/hashbrown deletion-history analysis invalidates using a fresh
map sample as proof: erased/tombstoned slots can reduce remaining growth without
releasing raw bucket allocation. Even an empty map is not provenance evidence.
Fresh String/Arc and buffer tests do not establish this missing bucket bound.

**DesignApprove:** construct an optional Join-private canonical metadata owner
from exact logical schema values at the cold boundary, with its checked actual
constructor inventory prepaid before allocating or capturing its objects.
Build fresh Schema/Fields/
Field Arcs, field names, schema/field metadata keys and values, and exact
timestamp type/timezone controls. Use maps privately constructed from known
entry counts with checked locked constructor bucket/control allocation bounds;
no deletion, unproved growth, caller-map cloning or mutation after sealing.
Record actual capacities of the core String/vector allocations. This owner
can be shared through its own real guard; it must not rely on the operator
staying alive. A public-capacity predicate or forgeable caller flag is not its
certificate. Keep the original port/schema declaration unchanged.

Unproved or denied normalization selects original Legacy payload behavior.
This is shape/funding fallback, not permanent disablement of every input:
ordinary fixed-width/string schemas and fresh constructed metadata must still
exercise the paid columnar path. It must not silently drop metadata, change
field flags/order, timestamp timezone spelling or public schema equality.

**Exact V1 is a separate gate.** Locked Arrow58.3
`ipc/convert.rs:132::metadata_to_fb` sorts metadata keys, which supports fresh
map ownership without relying on insertion order. It is not a substitute for
an actual complete row-IPC comparison. Before implementation, add a focused
test demanding private normalized ownership, exact original schema/field
metadata and every V1 schema prefix/row IPC/full inventory, from maps with
different insertion/delete histories and high reserved String/map capacity.
Record its real RED, then require exact GREEN on the constructed path. If a
profile changes wire bytes, keep that profile Legacy until a concrete
byte-preserving owner representation is proven. Schema `==` alone cannot pass.

## Minimum implementation-first RED acceptance gates

1. Preserve the **32,864 > 4,096** singleton key scratch RED, then require the
   actual private paid path to detach key and ID buffers, preserve values/order
   and keep real funding after operator/cache/source drop until the scratch
   owner drops. Cover state and probe tables, cache-hit and miss, zero matches,
   strings/timestamps/composites in the approved scope, and final exact zero.
2. Hold a real DF input/producer after query future cancellation, cache
   invalidation and operator drop. Verify keys/metadata remain readable and
   their credit/retirement stays live; managed close and a subsequent mutation
   must not claim cleanup/refund complete until actual producer/owner release.
   Include planning error, stream error, result-decoding error and dropped
   result; a mock local Arc-only or query-future completion test is insufficient.
3. Force scratch, cache, copy and decode admission denial under the unchanged
   runtime budget. Compare output sequence, physical IDs, original SQL errors,
   counters and exact V1 snapshots with Legacy. The one-row bounded scratch
   assertion applies to optimized success; denial is explicitly fallback,
   never relabelled as proven bounded scratch. Keep owner protection on borrowed
   certified fallback chunks. Include SQL memory pressure caused by optional
   paid candidates and exact post-cleanup retry/refund boundaries.
4. Independently construct a metadata map with insert/delete/tombstone history
   and spare String capacity, and record its real ownership/allocation RED.
   Prove the fresh owner drops caller allocations, preserves exact V1 metadata
   bytes, prepays requested live/peak controls, outlives operator safely and
   finally refunds zero. Cover empty-after-delete, field and schema metadata,
   multiple entries/orders, timezone and optional normalization refusal.

No Sparse design/implementation, V2 migration, public Batch/gather/UDF/API
change, fee/budget increase or performance measurement is authorized by this
critique. Nullable/Boolean payloads retain Legacy to preserve raw V1 bits.
Only this new artifact and its static proof directory were written; no source,
foreign worktree, historical artifact or remote state was changed. Critic
owned processes: **0**; no build, native import/test/probe or sampling was run.
