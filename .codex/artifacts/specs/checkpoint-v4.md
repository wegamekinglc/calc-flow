# P3 Checkpoint v4 — specification

Date: 2026-11-16. Source checkout: `f09d4e45`. Delivery: implementation steps
are commits in the single integrated re-architecture PR #619, not separate PRs.
Plan:
[streaming core re-architecture](../../../docs/plans/2026-11-15-streaming-core-rearchitecture.md),
especially decisions 3–5, §4.2, §6 and §7.4. P2 dependency:
[Rolling kernel v2](rolling-kernel-v2.md), §5.1, §5.4, §7 and §9.

Status: ready for public-API confirmation and failure-model critique. The
protocol, compatibility and performance gates below are fixed requirements, not
alternative proposals. No builds or measurements were run for this document.

## 1. Fixed decisions and source anchors

- Old checkpoints are not restorable. Manifest v4 is the only readable managed
  format; v3 and every older operator layout presented as a durable checkpoint
  are rejected. Restore from v4 continues bit-identically.
- Capture and persistence are separate. A successful capture freezes immutable
  state and forwards the runner barrier without waiting for encoding or disk.
  Only a durable manifest authorizes sink commit and source acknowledgement.
- At most one epoch is in flight. A following barrier is backpressured until the
  preceding epoch has completed or the job has failed.
- `unsafe_code = "forbid"` remains unchanged. The convenience-stream and
  explicit-runner checkpoint defaults from decision 5 remain unchanged.
- `tests/fixtures/v1/` remains immutable historical evidence.

The following current-source facts were verified at `f09d4e45`. Rust paths are
relative to `crates/calc-flow/src/` unless they begin with `docs/` or
`.codex/`:

- `StateSegment` hashes at construction (`operator/stream.rs:94-153`), while
  local staging and validation hash again (`state/local.rs:569-649`,
  `:821-923`). Publication renames and syncs directories separately
  (`state/local/publication.rs:10-35`).
- The operator barrier waits for capture, encoding and staging before its ack
  and downstream barrier (`runtime/streaming/operator_task.rs:1723-1745`,
  `:1826-1856`). The old rule that snapshot/stage failure forwards no barrier
  is documented at `docs/runtime-envelope.md:574-582`; P3 replaces that rule.
- `ManifestTransaction.operation` is the job-wide serialization point
  (`state/transaction.rs:159-165`), taken for staging and publication at
  `:348-359` and `:484-519`. Manifest write/rename/root-sync and uncertain
  outcome classification are at `:1385-1429` and `:1492-1557`.
- Manifest v3, its participant entries and strict validation are at
  `state/manifest.rs:18-21`, `:194-255`, `:331-359` and `:515-530`.
- Source cut/commit/abort ownership is at
  `runtime/streaming/source_task.rs:943-1034`. Sink pre-commit and its wait for
  Abort/Preserve/ManifestDurable are at `runtime/streaming/sink_task.rs:647-727`
  and `:855-904`. The public transactional sink lifecycle is at
  `continuous.rs:721-750`.
- SQL's bitmap-plus-slot-list `DirtyGroups` is the existing dirty-set model
  (`operator/sql/dirty.rs:3-63`). P3 generalizes its first-dirty marking; it
  does not reuse SQL's `MemoryReservation` per operator entity.
- P1 sessions currently ignore checkpoint configuration
  (`runtime/streaming/session.rs:51-52`) and have Open/Finished/Failed lifecycle
  states at `:203-207`. P1 explicitly deferred session checkpoints (P1 spec
  `.codex/artifacts/specs/stream-session.md:51-57`).

P2.3's `state/table.rs`, layout 7 and undo log, and P2.7's synchronous Rolling
operator have not landed in this checkout. Their specification anchors are
`rolling-kernel-v2.md:439-464`, `:523-541`, `:756-784` and `:1004-1021`.

## 2. Problem, goals and non-goals

Checkpoint-on currently selects different operator hot paths, serializes dirty
work through a job-global staging lock, and holds barriers through encoding,
hashing, writing and fsync. Existing evidence includes enabled-but-untriggered
costs of about 60% for `window_sum` and 54% for interval join, with additional
stalls when checkpoints fire. The review also counts nine operator families,
about twenty layout/version constants and about twelve binary magics. Those
figures are historical diagnosis, not P3 acceptance measurements.

Goals:

1. One checkpoint-on/off execution path for migrated operators, with runtime-
   owned key-group state partitions, copy-on-write pages and dirty tracking.
2. Capture-freeze-forward execution with bounded asynchronous persistence and
   no job-global disk-I/O lock.
3. One content hash per newly persisted object on its write path and epoch-
   batched filesystem durability.
4. A strict manifest v4 with per-partition roots, deterministic recovery
   identity, complete restore validation and explicit v3 rejection.
5. Deterministic convergence for cancellation, I/O failure, panic, uncertain
   publication, process exit and stop, without acknowledging incomplete state.
6. Opaque-block adapters for operators not yet migrated, without making their
   old layouts readable as standalone v3 checkpoints.
7. The unchanged P3 gates: migrated Rolling enabled-but-untriggered overhead
   at most 3%; sustained checkpointed throughput at least 90% of off at a 1 s
   interval and 75% at a 100 ms interval.

Non-goals:

- Reading, translating or falling back to v3 checkpoints.
- P4's shared hash router, P5's Join/ASOF rebuild, P6 sharding/morsel pool or
  P7's remaining operator migrations. P3's key-group model must not pre-empt
  their routing decisions.
- Changes to rolling arithmetic, output declarations, project format 3,
  watermark semantics, delivery guarantees or the approved performance gates.
- A non-blocking session checkpoint API, exactly-once ordinary sinks, or a new
  general benchmark framework.
- Claiming power-loss durability on platforms where directory synchronization
  is unavailable; the existing Unix/non-Unix distinction remains.

## 3. State table and page contract

### 3.1 Ownership and key-group partitions

A `StateTable` belongs to one compiled operator and contains logical columns.
It is divided into `StatePartition`s addressed by `KeyGroupId(u32)`. The
key-group roster and assignment rule are fixed at plan preparation and enter
the recovery fingerprint. A future shard owns whole key groups; no page is
shared between key groups.

P3 does not implement P4 routing. The P2.3 Rolling adapter must expose a stable
entity-to-key-group mapping without changing kernel calls. Its initial mapping
is the single plan-fixed group `0`; synthetic multi-group tables exercise the
partition contract. P4 may admit additional groups only under a new partitioner
identity in the recovery fingerprint.

Every manifest names every prepared operator, including stateless operators.
Every table-backed operator names its complete prepared key-group set, including
empty groups. A missing, additional or duplicated group fails restore.

### 3.2 Page identity and geometry

A page is the unit of freezing, copy-on-write, dirty marking and persistence.
`PageId` contains `(key_group, slot: u32, generation: u64)`. A slot indexes the
partition page table and dirty bitmap. Replacing or deleting a page increments
its generation; a handle always carries the full ID, so slot reuse cannot alias
a frozen or durable page. `next_generation` and free-slot state are restored and
included in undo state.

Page geometry is internal and fixed for v4:

- A keyed fixed-width column page contains at most 1,024 dense rows of one key
  group, including validity and raw floating-point bits. A logical row never
  straddles pages.
- A ring page contains one whole entity ring in logical oldest-to-newest order,
  its head/length/capacity and validity. Ring growth creates a new generation.
  A ring larger than the normal page target is one oversize page; it is never
  split when the P2 kernel requires contiguous slices.
- A value-log page targets 64 KiB and contains whole values plus their index.
  An oversize value occupies one page. Existing u32 value handles remain stable
  through the table's handle index.
- A pending/append-log page owns one immutable retained Arrow batch slice and
  its typed index columns. Rollback truncates whole appended pages or restores
  the prior page generation.
- Partition scalar state and dictionary metadata live in metadata pages; the
  partition index persists slot/generation/free-slot metadata. Output sequence,
  last identity, rebuild counters and family shifts are state, not recomputed
  restore hints.

All arithmetic on offsets, capacities and byte counts is checked. Safe Rust
owns every buffer; kernels see the same P2 row and slice operations and do not
see page handles, encodings or checkpoint epochs.

### 3.3 Dirty sets, freeze and copy-on-write

Each partition has a dirty bitmap and a first-dirty slot list, following the
`DirtyGroups` invariant: the first mark sets the bit and appends the slot;
later marks are bit-only. Entries include page replacement and tombstones.
With checkpointing off, the dirty set is a zero-capacity no-op, pages are never
frozen, and mutation stays in place.

`freeze(E)` requires no active undo transaction or borrowed mutable slice. In
one caller-thread operation it:

1. verifies that no earlier epoch is active and reserves the §7 memory budget;
2. swaps each touched partition's current dirty bitmap/list for an empty one;
3. clones the `Arc` for each dirty live page, stamps the resulting frozen
   reference with `E`, and records deletions as tombstones;
4. clones the immutable last-completed lineage/partition-index reference and
   the runtime's committed ingress/output progress; and
5. returns an immutable `TableFreeze` to the persistence service.

It performs no Arrow encoding, content hashing, allocation proportional to
clean state, filesystem operation or wait. Capture work and copied metadata are
`O(dirty pages + dirty metadata)`, not `O(total history)`. Clean pages are
reached through the previous partition root.

While `E` persists, the first mutation of a page frozen at `E` allocates a new
current generation and copies only that page before mutation. Subsequent
mutations use the new current page. Mutation of an unfrozen page stays in
place and marks it dirty for the next epoch. Frozen pages are immutable and are
released only after manifest durability or settled failure. A following freeze
cannot overlap this one.

### 3.4 Undo relationship and logical accounting

P2's undo log remains the batch-atomicity mechanism. P3 extends its first-touch
records to cover the page descriptor, current `Arc`, generation/free-slot
metadata, dirty bit/list length and logical-byte counters. It stores references,
not a second copy of every touched entity.

- Commit occurs only after output emission succeeds. It retains the new dirty
  state and any COW replacement.
- Error, cancellation or budget rejection restores the exact prior descriptors,
  dirty set, ring heads, dictionary/log lengths and scalar fields. A page that
  was already dirty before the callback remains dirty; a newly dirtied page is
  restored to clean.
- Freeze occurs only between committed transactions. Restore starts with an
  empty undo log and empty dirty sets.
- Live logical bytes are computed from current column/page capacities. COW
  candidates count before commit, so checkpointing cannot bypass the operator's
  existing `StateBudget`. Checkpoint-retained bytes are separately accounted in
  §7; neither figure is presented as process RSS.

## 4. Capture-freeze-forward protocol

### 4.1 Epoch phases

The v4 coordinator owns the checked sequence:

`Requested → SourcesCut → ParticipantsCaptured → SinksPrecommitted →
Persisted → ManifestDurable → SinksCommitted → SourcesAcked → Completed`.

`Persisted` means every source-history, operator and sink object needed for the
manifest is committed and every partition root is available. Persistence results
may arrive before sink pre-commit and are recorded without skipping the join
condition. Persistence starts
as soon as each capture arrives and overlaps later capture, barrier forwarding
and post-barrier computation. `ReadyToPublish` requires both
`SinksPrecommitted` and all persistence results; it never means merely queued.

Only `ManifestDurable` is a durable commit point. Capture/ack, barrier delivery,
`Persisted`, and sink pre-commit do not advance the public last-completed epoch,
acknowledge a source cursor or authorize an external commit. A manual request
returns success only after `SourcesAcked`; status then reports `Completed`.

Participant IDs and epochs are exact. Byte-identical duplicate capture or
persistence results are idempotent. Foreign, regressed, future, conflicting or
out-of-phase evidence fails the epoch and job. A capture digest covers the
stable participant metadata and page/block identities; a persistence result
additionally carries the final handles. Conflicting content for one identity is
a protocol failure even if its metadata digest matches.

### 4.2 Runner sequence

1. **Source cut.** Keep the existing accepted-slot settlement, ready pre-cut
   drain and route FIFO. Every live replayable source freezes its cursor,
   sequence, end flag, watermark-policy state and history descriptor for `E`.
   Successful barrier fan-out resumes admission but does not promote the
   durable cursor or call the external acknowledger. Ended sources contribute
   their final cut out of band.
2. **Aligned capture.** When every required operator ingress reaches `E`, all
   prior callbacks and undo transactions are complete. A migrated table calls
   `freeze(E)`; an opaque adapter captures its §5.4 block; runtime ingress and
   output-frontier state is copied. The capture and its memory reservation must
   be accepted by the persistence service before the operator ack or barrier.
3. **Forward and continue.** On capture acceptance, the operator sends its
   non-durable capture ack, forwards `Barrier(E)` and reopens live ingresses.
   Later Data/Watermark callbacks run normally and COW protects the freeze.
   There is no encode, stage, validate, publish or fsync wait in this path.
4. **Sink pre-commit.** After the aligned barrier reaches each output, ordinary
   sinks record `Ordinary` with no transactional state. Transactional and
   unbounded epoch-idempotent sinks pre-commit bounded metadata and connector
   blocks. After ack, every sink task awaits the epoch decision; transactional
   sinks do not accept post-barrier writes into the pre-committed transaction.
5. **Background persistence.** The service encodes and persists frozen objects,
   builds replacement partition indexes and returns canonical participant
   entries. It never mutates live operator pages.
6. **Manifest last.** Once all participants and handles are present, construct
   and fully validate manifest v4, then write, sync, atomically rename and
   parent-sync it. Only successful completion reports `ManifestDurable`.
7. **Commit and ack.** Send sink commit commands in stable order and await all
   finalizations. Then promote the frozen source cursors, invoke their durable
   acknowledgers in stable source order, resume the ordinary durable cursor,
   report `SourcesAcked`, and complete the epoch. Frozen-page reservations may
   be released after manifest durability, but no next epoch starts before
   `Completed`.
8. **Retention.** Retention and orphan collection run only after settlement.
   They retain the selected manifest, configured prior epochs, all partition
   roots and recursively referenced pages, and any still-pinned objects. A
   retention failure after commit does not invalidate the completed epoch.

### 4.3 One in flight and backpressure

The scheduler does not allocate a source cut or inject a barrier for `E+1`
until `E` is `Completed`. Periodic, manual and terminal requests retain the
existing bounded FIFO and epoch allocator; queue capacity wait is backpressure,
not an additional frozen epoch. Receiving `Barrier(E+1)` while `E` is incomplete
is a protocol violation, not a reason to queue its state.

Post-barrier data remains bounded by existing source slots and edge budgets.
If those fill while a sink awaits the manifest, normal edge backpressure
applies. The in-flight byte reservation in §7 is independent of, and additional
to, those edge limits.

### 4.4 Session checkpoint between pushes

P3 session support depends on P2.7's all-sync Rolling drive and on the public
storage/restore signature confirmed by `cf-api-designer`. Its semantic contract
is fixed here:

- `checkpoint` is legal only on an open session between completed
  `push`/`advance` calls. `&mut self` and the Python busy gate establish the
  quiescent point; no barrier, source task, sink task or progress task is
  introduced.
- It freezes the synthetic input's next sequence/watermark/generator state and
  every operator at that point, then persists manifest v4 with job kind
  `session`, no connector cursor and an empty sink set. It returns only after
  the manifest is durable. The first API is therefore synchronously durable,
  not a non-blocking enqueue.
- A validation failure before freeze leaves the session usable and unchanged.
  A capture, persistence or publication failure fails the session; later calls
  require a newly restored session. Close/stop settles admitted persistence
  before releasing the lineage and does not implicitly create a checkpoint.
- Restore installs the persisted next sequence, watermark and operator pages.
  The caller supplies any replay suffix; the session does not claim connector
  replay or exactly-once delivery. Restoring a finished runner manifest into a
  session, or the reverse, fails the job-kind check.
- Blocking checkpoint construction/restore/checkpoint calls follow the blocking
  convenience rule and reject a running event loop. Ordinary pushes retain
  P1's caller-thread contract; because the first checkpoint call returns only
  after durability, it leaves no hidden I/O wait for the next push.

### 4.5 Terminal epoch

After all inputs end, `on_end` and final emissions complete before capture. The
terminal epoch uses the same freeze, persistence, manifest, sink and source
ordering without a post-end barrier. A terminal manifest requires every source
and operator ingress to be ended. Recovery restores ended state, completes sink
recovery/commit, and never repeats `on_end` or reopens ended sources.

## 5. Asynchronous persistence and manifest v4

### 5.1 Objects, hashing and fsync

V4 introduces one new outer binary envelope magic, `CFST4\0`, with a checked
object kind: `page`, `partition_index`, `source_history` or `sink`. Page payloads
use canonical Arrow IPC with exact schemas and raw value bits. Partition indexes
use a canonical, strict, duplicate-free encoding of the complete live page
inventory. Source-history and connector payloads are length-delimited bytes. An
opaque block instead stores the exact existing `StateSegment` bytes and reuses
the digest computed once at its capture; the manifest supplies its v4 kind and
fingerprint, with no additional wrapper, magic or hash. Nested legacy bytes have
no runtime meaning until their owning opaque decoder receives them.

Each immutable handle records owner, creating epoch, stable object ID, portable
committed path, byte length and lowercase SHA-256. Paths are lineage-scoped and
include epoch, full page/object generation and digest; existing different bytes
at one path are a conflict, never an overwrite.

For each newly created non-opaque object, one bounded blocking worker produces
final bytes and computes SHA-256 once. An opaque block carries its existing
capture digest. The same bytes and digest pass to staging and the
handle. There is no read-back hash between staging and rename, no second hash
at publication, and no per-epoch rehash of a carried object. A fresh recovery
still hashes every loaded object once; a manifest or index hash does not count
as rehashing page payload bytes.

Persistence is epoch-batched:

- at most four blocking write/encode workers operate on disjoint objects, with
  deterministic result ordering independent of completion order;
- each newly written file is written and synced once, each affected staging
  directory is synced once after its epoch batch, and directory creation is
  confirmed once per epoch;
- validated staging files are renamed without another payload read, then each
  affected committed directory is synced once for the epoch; and
- the manifest has one separate temp-file write/sync, atomic no-clobber rename
  and manifest-root sync.

No mutex spans the job's disk operations. Per-directory exclusion, where needed
for local namespace safety, cannot become a job-global stage/publish lock.
Cancellation settles every admitted worker before lineage release.

### 5.2 Manifest structure

Manifest v4 remains bounded canonical JSON with duplicate keys, unknown fields,
excess depth and documents over 10 MiB rejected. `format_version` is `4`;
`Epoch(0)` remains invalid. Its top-level fields are:

- `pipeline_name`, `pipeline_fingerprint`, `runtime_config_hash`, `epoch`,
  `created_at`, `recovery_status = final`, and `job_kind = runner|session`;
- `recovery_fingerprint`, defined in §5.3;
- exact `sources`, `operators`, `sinks` and `static_inputs` maps in canonical
  order; and
- `manifest_fingerprint`, SHA-256 over canonical JSON of every top-level field
  except the fingerprint itself, including the epoch, job kind and complete
  participant state.

Runner participant rules retain non-empty prepared source/operator/sink sets.
A session has exactly one synthetic source, its prepared operators and no
sinks. All maps must exactly match the prepared job.

Entries have these contracts:

- **Source:** the existing exact cursor/order/payload, prepared identity hash,
  next sequence, ended flag, watermark-policy state and optional history. V4
  history references v4 handles; no legacy omitted-history normalization is
  retained.
- **Table operator:** typed ingress progress, an explicit optional output
  frontier, operator state fingerprint, and a canonical `partitions` array.
  Each element is `(key_group, root handle)`. The root identifies a partition
  index containing every live `(PageId, column/kind, logical range, byte length,
  SHA-256, handle)` and the slot/generation/free-slot metadata needed for later
  allocation. Deleted pages are absent. Empty prepared groups have an
  empty-index root. The manifest does not inline every page.
- **Opaque operator:** typed ingress progress, explicit optional output
  frontier, state fingerprint, bounded inline metadata and canonical opaque
  block handles. It has no partition field.
- **Sink:** unchanged delivery evidence (`Ordinary`, `EpochIdempotent` with
  mechanism/retention, or `Transactional`), required nullable pre-commit
  metadata and v4 connector block handles. Ordinary sinks carry neither
  pre-commit nor blocks.

Partition indexes and handles are strictly ordered and unique. Handle epochs
may be older than the manifest for carries but never newer. Reachability
includes partition roots and every page referenced by their indexes.

### 5.3 Recovery fingerprint

`pipeline_fingerprint` remains the graph's semantic fingerprint. A separate
`recovery_fingerprint` is SHA-256 over one canonical object containing:

- manifest format `4`, protocol `capture-freeze-forward-v1`, object envelope
  `CFST4`, and job kind;
- pipeline name/fingerprint and exact source/operator/sink/static-input IDs;
- each source's prepared identity, replay/watermark contract and history
  contract;
- each operator's stable ID/kind, state schema and logical layout, exact input
  schemas, state-table column inventory, key-group assignment/partitioner
  version, numeric contract ID, external state version, and required
  DataFusion/Arrow state identity; and
- each sink's prepared identity, delivery mechanism and retention class, plus
  every static-input digest.

For Rolling, P2's logical layout 7 and `compensated-sliding-1` remain in this
identity. P3 deletes layout 7's temporary whole-table transport codec, not its
logical column invariants. Runtime interval, timeout, retention, worker count
and in-flight byte limit affect only `runtime_config_hash`, which remains a
non-blocking diagnostic.

Any recovery-fingerprint mismatch fails before state installation. A runtime
config difference alone never selects a different state decoder.

### 5.4 Un-migrated operators as opaque state

Until their assigned P4/P5/P7 migration, checkpoint-capable non-Rolling
operators use an opaque v4 adapter. Existing SQL loan, Join owned capture, ASOF
managed capture and ordinary synchronous capture may still produce their old
`OperatorStateSnapshot` at the aligned cut. The runtime wraps its inline
metadata, state version/layout/schema identity and immutable segments as v4
opaque blocks; only staging/publication moves to the background. This exception
does not add a checkpoint branch to migrated Rolling.

Restore validates the v4 manifest, block handles and operator fingerprint,
then invokes that operator's existing decoder/restore on the exact opaque
bytes. The runtime never interprets nested base/delta journals. Each later
migration replaces the adapter with state partitions and deletes that family's
nested layouts, encoders, decoders and magics in the same commit.

Every admitted opaque family must pass v4 save/reload/continue bit-identity
coverage. A family that cannot satisfy it is rejected from v4 preflight until
migrated; v4 does not silently weaken the restore guarantee.

## 6. Failure matrix

An epoch with any failed participant is wholly invalid. There is no partial or
mixed-page manifest and no recovery from an incomplete freeze. Once a barrier
has been forwarded, post-barrier in-memory progress cannot be advertised as
durable; failure terminates the job and restart uses the selected durable
manifest. The old "stage failure forwards no barrier" rule no longer applies.

| Event                                      | Required convergence                                                                                                                                                                                                                     |
|--------------------------------------------|------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| Capture/freeze rejects before forwarding   | Send no capture ack or downstream barrier from that operator, cancel the epoch/job, settle admitted I/O, and leave the last manifest as recovery truth. Earlier forwarded barriers do not make the epoch publishable.                    |
| Stage/write failure after barrier forward  | Mark `E` failed, publish no manifest, and send no source ack. If manifest absence is established, abort prepared sinks. Partial files are unreachable orphans. Post-barrier ordinary output may duplicate after replay.                |
| Disk full during state/index write         | Converge as a stage/write failure with the original I/O category. Do not skip the object, publish a reduced index, or retry under a new epoch. Clean orphans only after the active attempt has settled.                                 |
| Disk full during manifest publication      | If the destination is proved absent, abort prepared sinks and fail the job. If rename installed it, the bytes differ, or the outcome cannot be read, preserve sink transactions and return `RecoveryRequired`; never guess abort.       |
| Checksum mismatch before publication       | Reject the object and invalidate `E`; no manifest or source ack. A same-path/different-content object is a conflict. Do not replace it with carried state.                                                                               |
| Checksum/schema/invariant failure on read  | Fail closed before operator readiness, sink recovery commit, source ack or user success. Do not fall back to an older epoch after a corrupt higher candidate.                                                                          |
| Encode/hash/write worker panic             | Convert the join/panic to the epoch's primary failure, cancel siblings, and settle every admitted operation. Apply the manifest-absent versus installed/uncertain rule before Abort versus Preserve.                                    |
| Operator/source/sink task panic            | Preserve the existing stable task identity and primary error ordering. Cancel and settle the persistence epoch; a late worker cannot publish after a proved-absent decision.                                                             |
| Cancellation before manifest durability    | Stop new work but settle admitted filesystem calls under the lease. If absence is then proved, abort sinks and invalidate `E`. If installed or uncertain, preserve sinks and report `RecoveryRequired`. Never acknowledge the cut early. |
| Uncertain manifest rename/root sync/ack    | Treat an installed or unreadable destination as recovery intent. Preserve prepared sinks, omit source ack, and require restart to select and validate the manifest. A different destination payload is a conflict.                       |
| Sink commit failure after durable manifest | Never abort the durable epoch. Complete idempotent commit forward where possible; otherwise report `RecoveryRequired`. Restart resumes partial multi-sink commit from the manifest in stable order.                                    |
| Source ack failure after sink commit       | Keep the durable manifest and report `RecoveryRequired`. Restart seeks from the manifest and repeats only an idempotent cursor acknowledgement; it does not rewrite operator state or recommit a completed sink unnecessarily.           |
| Process exit before manifest durability    | On restart, ignore/clean settled orphans and recover from the previous manifest. Reconcile or abort unpublished sink work. Ordinary duplicates are allowed; no incomplete freeze is read.                                                |
| Process exit after manifest durability     | Select and fully validate `E`, restore exact state/cursors, complete transactional or epoch-idempotent sink recovery, then resume the source suffix. Lack of a final ack does not invalidate `E`.                                       |
| Then-stop after any event                  | Stop admission and new epochs, cancel compute, settle admitted I/O and connector calls, join every task/worker, and release frozen refs, reservations and the lease. Restart only through manifest selection; no late background commit.   |
| Retention/cleanup failure after completion | Keep `E` completed and durable. Report the maintenance/job failure without deleting reachable state or replaying a completed sink commit as though `E` were absent.                                                                      |

Manifest publication uses a single durable intent record. Abort is legal only
when absence is established. After a durable manifest, cancellation and stop
never change the recovery decision back to abort.

## 7. Memory and in-flight limits

Add `checkpoint_inflight_max_bytes` to Rust/Python stream runtime configuration,
default `1 GiB`, positive and at most the largest JSON-safe integer. It is a
logical checkpoint-retention limit, not RSS, and enters only the runtime-config
hash. Project format remains 3; an omitted field uses the default.

For one active epoch, reserve before capture acceptance:

- `F`: bytes of frozen pages that can become non-live after later COW;
- `O`: the opaque/source/sink capture upper bound, acquired before old-family
  encoding and including its existing capture credit;
- `S`: maximum simultaneous encoded/write buffers under four workers, with
  oversize objects serialized when necessary; and
- `M`: frozen descriptors, dirty bitmaps/lists, partition indexes and manifest
  storage, using checked capacity arithmetic.

The reservation is `F + O + S + M`. Frozen pages still current in live state
are not charged a second time as process memory; `F` is the worst additional
retention as COW replaces them. Actual accounting tracks frozen-nonlive pages,
opaque blocks, encode buffers, metadata and active COW candidates separately.
It must never exceed the reservation, and live plus checkpoint-candidate state
must still pass the operator's own state budget.

If the reservation exceeds the configured limit, capture fails before its ack
or barrier. With no earlier active epoch there is no wait that can create room,
so this is a checkpoint-budget failure, not an unbounded queue. At manifest
durability, release page/encode reservations no longer needed for recovery; at
settled failure or stop, release all of them. One active epoch and the bounded
request FIFO are hard limits independent of byte count.

Recovery preflights every document/index/segment length before allocation. A
partition index is capped at 64 MiB with canonical entry-count checks; page and
opaque lengths must fit `usize`, the restored state budget and available read
buffers. Recovery reads one bounded object per worker and never materializes a
second full lineage merely to validate it.

## 8. Restore, compatibility and bit identity

Startup performs these steps before releasing the data gate or reporting
restored readiness:

1. Open the exclusive lineage and enumerate bounded manifest candidates. Select
   the highest complete v4 epoch; filename and manifest epoch must match.
2. Parse strict canonical JSON and verify `manifest_fingerprint` before reading
   referenced state. Validate job kind, semantic/recovery fingerprints, exact
   participant/static-input sets, epoch and source/sink capabilities.
3. Load and checksum every selected partition index. Validate its key group,
   column inventory, page IDs/generations, logical ranges, canonical order,
   unique paths and complete prepared-group coverage before page installation.
4. Preflight lengths/budgets, then load and checksum every referenced page,
   opaque block, source-history object and sink block. Reuse verified loaded
   bytes for installation; selection and restore must not hash the same object
   twice in one recovery.
5. Validate table invariants without arithmetic reconstruction: ring length and
  capacity, dictionary/EntityId coverage, raw counters against ring membership,
  family shifts, rebuild cadence state, pending-log identities, progress/output
  frontiers and terminal agreement. Install exact stored buffers and an empty
   undo/dirty state.
6. Restore source cursors/watermark state and operator progress, recover sinks,
   and only then admit input. `next_epoch` is the checked successor of the
   selected epoch.

For a fixed declaration and canonical input suffix, outputs, order, watermarks,
source sequences and later v4 state bytes are bit-identical to an uninterrupted
control. Rolling accumulators, shifts, recurrence state and history are loaded,
not replayed/rebuilt into potentially different floating-point bits. Manifest
timestamps and filenames are excluded from the state-byte comparison.

A v3 manifest returns `UnsupportedVersion { expected: 4, found: 3 }` at the
format parser. Runner/session startup presents the safe recovery error with the
explicit instruction `checkpoint manifest v3 is not restorable; start a new
lineage`. A v3 document is never normalized, partially read or skipped in favor
of an older/newer lineage. Unknown v4 object kinds, nested layout mismatch and
a generation regression for the same slot in a replacement root fail closed.

## 9. Test requirements

Every behavior change begins with a focused red test in its implementation
commit. Keep full regression and the 90% Rust line-coverage gate in CI.

1. **State table:** first-dirty marking, checkpoint-off zero dirty allocation,
   in-place writes before freeze, COW after freeze, frozen-page immutability,
   tombstones/generation reuse, oversize rings/values, multiple key groups and
   freeze cost independent of clean pages.
2. **Undo:** injected emission, cancellation and budget failures restore exact
   page refs, dirty metadata, logs, dictionary, scalar fields and byte counters
   with checkpointing on and off. Freeze during an open transaction is rejected.
3. **Protocol:** gated persistence proves capture ack, barrier forwarding and a
   post-barrier Data callback precede persistence resolution; no manifest,
   durable status, sink commit or source ack can precede its prerequisite.
   Cover duplicate/conflicting evidence, next-barrier backpressure and terminal
   capture.
4. **Persistence:** instrument hash and fsync calls per object/directory,
   concurrent disjoint writes, canonical result ordering, carry reuse, no
   read-back publication hash, recursive retention and orphan cleanup.
5. **Failure matrix:** inject every §6 row, including after-forward stage
   failure, disk full, write/checksum failure, real worker panic, rename/root-
   sync uncertainty, process kill on both sides of publication, and then-stop.
   Assert task joining, lease release, zero remaining reservation and the exact
   Abort/Preserve/source-ack decision.
6. **Manifest/restore:** canonical v4 golden, every fingerprint component,
   malformed/duplicate/unknown fields, corrupt higher candidate, partition-index
   corruption, budget refusal and v3 rejection. Cover at least 50 seeded Rolling
   checkpoint positions/batch splits with bit-identical continuation and later
   state bytes.
7. **Opaque adapters:** one save/reload/continue test for every compiled
   checkpoint-capable family, including progress/frontier/terminal state and
   wrong-family rejection. No v3 fixture becomes a v4 compatibility fixture.
8. **Session:** checkpoint only between calls, no barrier/task creation,
   sequence/watermark restore, caller-supplied suffix, failed/busy/closed
   behavior, running-loop rejection for blocking checkpoint operations, and
   close during admitted persistence.
9. **Delivery:** ordinary replay may duplicate; transactional and unbounded
   epoch-idempotent sinks complete partial commits during recovery before any
   source suffix is released.

Focused local checks for the implementation commits are:

```bash
cargo fmt --all --check
cargo test -p calc-flow --lib state::
cargo test -p calc-flow --lib runtime::streaming::checkpoint
cargo test -p calc-flow --lib runtime::streaming::operator_task
cargo test -p calc-flow --lib runtime::streaming::session
cargo test -p calc-flow --test rolling_state
cargo clippy -p calc-flow -p calc-flow-python --lib --tests -- -D warnings
uv run maturin develop
JAX_PLATFORMS=cpu uv run pytest python/tests/test_stream_session.py python/tests/test_checkpointing.py python/tests/test_continuous_runtime.py -q
uv run ruff check python/calc_flow python/tests/test_stream_session.py python/tests/test_checkpointing.py python/tests/test_continuous_runtime.py
uv run ruff format --check python/calc_flow python/tests/test_stream_session.py python/tests/test_checkpointing.py python/tests/test_continuous_runtime.py
git diff --check
```

Full workspace tests, docs, coverage and cross-platform checks remain CI gates.

## 10. Pre-registered measurements

Register build identities, raw samples and reports under
`benchmarks/checkpoint-v4/`. The four cases below are the complete P3 set. The
combined wall-clock budget is **600 s**, including fixtures, startup, warm-up,
checkpoint coverage validation, Arrow/oracle checks, restore checks, process
startup and cleanup. Builds are reported separately. Exhaustion, an unsuitable
host or missing coverage stops measurement; no cases, samples or retries are
added.

All cases use the P2 graph `difference(mean(5), mean(20))` followed by an exact
column projection, 64-bit prices, deterministic timestamps and the P2 numeric
oracle. The candidate's checkpoint-off arm is the reference for each ratio, so
P3 cannot pass by weakening the off path. Existing evidence from the P3.1
parent is linked as an informational revision control; no additional control
arms are run. All timed runs use release binaries, four Tokio
workers, one helper thread per BLAS pool, and no overlapping owned build/test.

| Case | Workload and fixed protocol                                                                                                                                                                         | Gate                                                                 |
|------|------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|----------------------------------------------------------------------|
| M1   | Direct sync Rolling `1x1-dual`, warm 4,096-row history; checkpoint on with no epoch versus off. Two rounds of ten adjacent AB/BA pairs; each observation is the median of 30 dispatches.        | Median paired `on/off - 1` ≤ 3%.                                    |
| M2   | M1 at `6400x64-dual`, recorder absent; same fixed pairs and timing boundary.                                                                                                                        | Median paired `on/off - 1` ≤ 3%.                                    |
| M3   | Sustained replayable-source Rolling runner, 64 entities and 6,400-row batches; 1 s checkpoints versus off. Two rounds of ten AB/BA pairs, one warm-up per arm/round, 5 s measured per arm.      | Median paired `on_rows_s/off_rows_s` ≥ 0.90.                        |
| M4   | M3 with a 100 ms checkpoint interval and the same fixed counts/budget.                                                                                                                               | Median paired `on_rows_s/off_rows_s` ≥ 0.75.                        |

M3/M4 count only input rows whose expected output reached the ordinary sink by
the 5 s cutoff; stop admission at cutoff and settle/cancel outside the timer.
Each arm validates its complete emitted prefix against the oracle. An on sample
must contain at least four completed nonterminal epochs in M3 or forty in M4;
off must prove zero checkpoint objects. Record accepted/emitted rows, epoch
counts, freeze-to-forward latency, persistence latency, edge high-water marks,
checkpoint reservation/actual bytes and recovery validation time. Missing
coverage or failed cleanup invalidates the sample.

For every case, report each round's exact paired-median interval and the pooled
median gate metric. A result whose per-round interval crosses the gate is
`inconclusive` and does not establish that gate. Do not substitute independent
P50s, best samples or terminal-checkpoint-only timing for these ratios. If a
gate misses, stop with the stage/resource breakdown for a decision; do not
expand the measurement set.

## 11. Commit sequence and deletions

Each row is a red→green commit or small commit group in PR #619. It deletes the
implementation it replaces in the same step and runs only focused local checks;
CI owns full regression.

| Step | Prefix      | Content and dependency                                                                                                                                                  |
|------|-------------|--------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| P3.1 | `test:`     | Fake-table protocol/failure harness, v4 goldens and measurement registration. Needs P2.1 harness; final Rolling baselines wait for P2.7.                              |
| P3.2 | `refactor:` | COW pages, key-group partitions, dirty sets, undo integration and logical accounting behind the P2.3 `state/table.rs` seam. No Rolling kernel edits.                 |
| P3.3 | `feat:`     | V4 object writer/reader, four-worker persistence, single hash, epoch fsync batches, partition roots and manifest/fingerprint validation.                               |
| P3.4 | `refactor:` | Capture-freeze-forward coordinator, source/operator/sink wiring, one-in-flight backpressure, terminal path and opaque adapters. Rolling switch needs P2.7 and P3.2–3. |
| P3.5 | `feat:`     | Synchronously durable session checkpoint between pushes and v4 session restore, after P2.7 and the public API note.                                                   |
| P3.6 | `test:`     | Subprocess/fault matrix, bit-identity and memory/retention completion; each behavior was already red→green in its owning step.                                        |
| P3.7 | `perf:`     | Run only the four registered cases and commit their evidence; a missed gate stops for a decision.                                                                       |
| P3.8 | `docs:`     | Reconcile public docs, stubs/schema, changelog and plan execution record through `cf-doc-writer`.                                                                       |

By the end of P3:

- Delete the job-global stage/publish operation lock and inline operator/sink
  staging from the barrier path.
- Delete v4 write-path read-back hashing and duplicate segment hashes; retain
  fresh-recovery verification.
- Delete v3 manifest normalization/readers and the old durable transaction
  protocol. Do not leave a v3 read-only fallback.
- Replace P2 Rolling's whole-table owned-array `snapshot`/Arrow `StateSegment`
  transport with frozen pages and partition roots. Keep layout 7 only as the
  logical Rolling state/fingerprint identity described in §5.3.
- Remove Rolling's sync checkpoint/restore transport hooks once its table is
  runtime-captured; its P2 numeric kernels and undo behavior remain.
- Keep non-migrated families' nested codecs only inside their opaque adapter.
  P4/P5/P7 delete each family's codec when it migrates; P3 does not leave those
  families on the v3 manifest or synchronous staging protocol.
- Delete superseded tests that require staging before barrier forwarding,
  replacing them with the protocol/failure pins above. Do not alter historical
  v1 fixtures.

## 12. Acceptance criteria

- [ ] Given checkpointing on but no epoch, Rolling executes the same kernels as
      off, records no per-entity checkpoint reservation, allocates no dirty-set
      capacity, and contains no Rolling `.checkpointing()` branch.
- [ ] Given a frozen page and subsequent writes, every v4 reader observes the
      frozen bytes while current state advances through COW; undo restores the
      exact pre-callback page and dirty state after every injected failure.
- [ ] Given a persistence gate that has not resolved, capture ack, barrier
      forwarding and post-barrier processing complete, while manifest
      publication, durable status, sink commit and source ack remain blocked.
- [ ] Given any participant/write/checksum/panic failure, no incomplete manifest
      is published and no source cursor is acknowledged. Given uncertain or
      installed manifest publication, sinks are preserved and restart resolves
      the manifest rather than aborting it.
- [ ] At most one epoch owns frozen state; a following barrier, queued captures
      and checkpoint memory remain within the configured limits. Stop joins all
      workers and releases reservations and lineage ownership.
- [ ] A v4 manifest round-trips canonically, detects every identity/participant/
      partition/page mutation, and rejects v3 with the documented new-lineage
      error. No v3 fallback is present.
- [ ] Given a valid v4 checkpoint and replay suffix, restored output bits,
      order, watermarks, source sequences and subsequent state bytes equal an
      uninterrupted control for Rolling and every admitted opaque family.
- [ ] A session checkpoints only at a between-push quiescent point, restores its
      synthetic input and operator state, and never claims connector replay or
      sink exactly-once behavior.
- [ ] Hash/fsync instrumentation proves one write-path content hash per object,
      no read-back publication hash, and the specified per-epoch directory
      batching. No job-global lock spans staging or publication I/O.
- [ ] M1–M4 establish the unchanged ≤3%, ≥90% and ≥75% gates under §10, or the
      integrated PR stops with inconclusive/missed-gate evidence rather than
      claiming success.
- [ ] The focused checks in §9 pass locally for affected modules; full
      regression, docs and the Rust 90% line-coverage floor pass in CI.
- [ ] `docs/introduction.md`, `docs/runtime-envelope.md`,
      `docs/streaming-guide.md`, `docs/rust-api.md`, `docs/python-api.md`,
      `docs/design.md`, `AGENTS.md`, `_native.pyi`, the generated project schema
      if the runtime field is serialized, and `CHANGELOG.md` agree with v4.
      Historical v1 fixtures are unchanged.

## 13. API confirmations and routing

`cf-api-designer` confirms the public names and signatures without reopening the
protocol: `checkpoint_inflight_max_bytes`, session storage/restore construction
and checkpoint return type, v4 manifest/partition handle types, and the safe
checkpoint-budget/recovery error projection. It also confirms the exact Python
stub and project-schema diff.

After that note, `cf-critic` reviews the freeze/undo invariant, manifest commit
point, §6 failure matrix, uncertain publication and memory proof. No protocol
semantics above are left as implementation choices.
