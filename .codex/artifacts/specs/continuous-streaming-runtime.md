# Continuous Streaming Runtime - Specification

Artifact slug: `continuous-streaming-runtime` (shared by this specification, the
M0.2 API note, and the M0.2 critique; definitions here must stay stable and
quotable across all three).

## Source

- Plan task: M0.1 of
  [`docs/superpowers/plans/2026-08-02-continuous-streaming-v3.md`](../../../docs/superpowers/plans/2026-08-02-continuous-streaming-v3.md)
  (section 6, Task M0.1), including the plan-wide invariants of section 5 and
  the process requirements of section 16.
- Research basis:
  [`docs/research/2026-08-02-arroyo-risingwave-streaming-research.md`](../../../docs/research/2026-08-02-arroyo-risingwave-streaming-research.md)
- Related docs: [`docs/introduction.md`](../../../docs/introduction.md) (v2 data
  contract and runner semantics),
  [`docs/runtime-envelope.md`](../../../docs/runtime-envelope.md) (v2 internal
  envelope; superseded by the v3 message model in S1),
  [`AGENTS.md`](../../../AGENTS.md) (commands, coding style, verification),
  [`.claude/rules/code-style.md`](../../../.claude/rules/code-style.md).
- Status: semantic baseline for Calc-Flow 3.0. This document freezes semantics
  only. Rust/Python type names, trait signatures, error variant names, and JSON
  field layouts are owned by the M0.2 API note under the same slug. The key
  words MUST, MUST NOT, SHOULD, and MAY are normative. Revision 2 applies M0
  critique round 1 (`../../critiques/continuous-streaming-runtime.md`):
  D4.1/D4.3/D4.5, D5.1/D5.3/D5.4, D9.4/D9.6, section 1 cursor definition,
  S2.2, S7.3, S7.5, S9.2/S9.4, S10.3, I4/I5/I6, NFR-5, FR17/FR23, and
  AC-D4/AC-D5/AC-S10/AC-I. Revision 3 applies M0 critique round 2 (finding
  N1): the S9.1 matrix sink cell and the S9.2 sink clause now require an
  unbounded retention class for an epoch-idempotent sink to satisfy an
  exactly-once compile, matching the API note's rule.

## Problem Statement

Calc-Flow 2.0 is a bounded micro-batch engine: one call processes one formed
input set, nodes run once per call in topological order, and the crate-private
watermark/epoch markers are opaque occurrences with no event-time value, no
monotonicity, no idle or late-data meaning, no checkpoint identity, and no
runner contract. Calc-Flow 3.0 adds a continuously running, source-driven,
stateful streaming runtime with event time, windows, epoch checkpointing, and
transactional sinks. Watermark, barrier, cancellation, and recovery semantics
that are coded before they are specified become unchangeable public behavior
(research section 11, semantic risk), so every semantic rule the runtime will
depend on is frozen here, before any production code exists.

## Goals

- Freeze the message, ordering, time, window, checkpoint, recovery, lifecycle,
  and delivery-guarantee semantics of the 3.0 continuous runtime so that M1-M7
  implementation never has to invent a semantic rule on the spot.
- Freeze the five mandated decision points (D1-D5) plus the additional
  decisions the plan explicitly assigns to M0 (D6-D9).
- Give every semantic rule at least one executable acceptance-test statement
  (plan M0.1 acceptance gate).
- Record every plan non-goal as a normative non-goal of this specification.
- Keep the batch executor's existing v2 behavior contract untouched except
  where the plan's sanctioned breaking changes (section 1.3 of the plan)
  explicitly replace it.

## Non-Goals

The following are normative non-goals for 3.0. Any future change to this list
is a spec amendment under this slug, not an implementation choice.

- **NG1** - No distributed workers, shuffle, slots, rescale, or controller HA.
  The runtime is single-process and embedded.
- **NG2** - No PostgreSQL wire protocol and no RisingWave-style serving
  database or queryable materialized views.
- **NG3** - No Hummock equivalent, no general distributed LSM, no mandatory
  object storage. The first state backend is a local filesystem backend.
- **NG4** - No session windows, no early triggers, no processing-time
  triggers, no allowed lateness, no late-data side output. Configurations
  asking for any of these MUST fail at compile time.
- **NG5** - No engine-level changelog/retract/upsert stream. Windows are
  final-only (S6). PostgreSQL CDC output is an append-only change-event
  envelope; it does not materialize table state.
- **NG6** - No multi-input incremental SQL join and no event-time temporal
  join. Multi-alias SQL in stream mode fails at compile time.
- **NG7** - No automatic schema evolution, no ClickHouse CDC, and no native
  database CDC other than PostgreSQL. A CDC schema change stops the source and
  reports the old and new schema.
- **NG8** - No deep forks of DataFusion, Arrow, or sqlparser.
- **NG9** - No compatibility layer or automatic migration for project v2,
  checkpoint v2, the v2 runner APIs, Python v2 signatures, or Studio
  `/api/v2`. v2 documents are rejected with `UnsupportedVersion`.
- **NG10** - No unaligned checkpoints in 3.0. Barrier alignment (S7) is the
  only checkpoint protocol.
- **NG11** - No global ordering guarantee across fan-out sibling branches or
  across union ingresses beyond what S3/S4 state; no cross-input business
  order may be promised or tested.
- **NG12** - No per-row lateness criterion anywhere in the engine (D2). No
  window update, retraction, or re-emission of a closed window within one
  continuous fault-free run.
- **NG13** - `tests/fixtures/v1/` remains immutable historical parity
  evidence. It is not a v3 runtime or package path and no 3.0 runtime path may
  read it.

## 1. Definitions and notation

- **Edge** - one directed connection between a single producer (source task or
  operator task output) and a single consumer (operator task ingress or sink
  task). Every edge has exactly one producer and exactly one consumer.
- **Ingress** - one named input of an operator task, fed by exactly one edge.
- **`StreamMessage`** - the single ordered message type carried by an edge:
  `Data(Batch)`, `Watermark(EventTime)`, `Barrier(Epoch)`, `Idle`,
  `EndOfInput`. `Data` carries the existing immutable public `Batch` envelope;
  raw tables and arrays never cross a graph, plan, or runner boundary.
- **`EventTime`** - a strongly typed UTC instant; internal representation and
  conversions are frozen in D1.
- **Watermark** - an `EventTime` value asserting that the producer believes no
  future data on that edge will carry an event time below it.
- **`WM_in(op)`** - the current input watermark of an operator: the minimum of
  the last-received watermarks over its primed Active ingresses (S5). It is
  undefined while any Active ingress has not yet delivered a first watermark.
- **`Epoch`** - a strongly typed checkpoint identifier (D9). A **barrier** is
  the `Barrier(Epoch)` message that separates pre-epoch data from post-epoch
  data on an edge.
- **Cursor** - a source-defined, totally ordered replay position. Its
  representation carries a connector-encoded ordered byte key whose bytewise
  lexicographic order IS the source order, plus an opaque payload (the
  ordered-envelope contract fixed by the M0.2 API note); the runtime compares
  only the ordered key. **Sequence**
  - a Calc-Flow-owned per-source item counter (S2).
- **State segment** - an immutable file holding keyed operator state (D6).
  **Checkpoint manifest** - the bounded, atomically published metadata document
  that is the sole record of a completed epoch (D4).
- **Job / supervisor / reaper** - the ownership chain for all runtime tasks
  (D5).
- **Delivery guarantee** - the per-sink contract derived from the capability
  matrix in S9.

## 2. Frozen decisions

Each decision below is numbered for direct citation. D1-D5 are the five
mandated decision points. D6-D9 are additional choices the plan explicitly
assigns to M0 (plan sections 3, 5, 6, and task M4.1/M4.3/M5.2/M1.2 cross
references).

### D1 - EventTime representation, conversions, and truncation direction

**D1.1 Internal representation.** `EventTime` is a newtype over a signed
64-bit integer counting **microseconds since the Unix epoch
(1970-01-01T00:00:00Z), always UTC**. The ordering is total across pre- and
post-epoch values. Public APIs (Rust, Python, project documents, checkpoint
documents, manifests, metrics, logs) MUST NOT expose a bare `i64` timestamp;
they expose `EventTime` or its exact serialized microsecond value.

**D1.2 Serialized precision.** Checkpoint manifests, state metadata, and any
other durable or observed representation of an `EventTime` store the exact
internal microsecond value. Lossless round-trip through persistence is
required; recovery MUST NOT shift any watermark, window bound, or event time.

**D1.3 Import conversions (Arrow timestamp column to `EventTime`).** The
conversion is selected by the Arrow column's time unit and MUST be checked -
overflow or underflow returns a typed error carrying the source and column
path; silent wrap is forbidden:

- `timestamp(s)`: value x 1,000,000, checked. Overflow is an error.
- `timestamp(ms)`: value x 1,000, checked. Overflow is an error.
- `timestamp(us)`: exact identity.
- `timestamp(ns)`: value divided by 1,000 with **floor** (toward negative
  infinity), never truncation toward zero. This conversion cannot overflow.

**D1.4 Export conversions (`EventTime` to an Arrow representation).**
Producing a finer or equal unit is exact or checked: microseconds are exact;
nanoseconds multiply by 1,000, checked (overflow is an error, since the
internal microsecond range exceeds the nanosecond `i64` range). Producing any
coarser representation applies **floor** (toward negative infinity):
milliseconds divide by 1,000, seconds divide by 1,000,000.

**D1.5 Single uniform truncation direction.** Whenever any coarser
representation is produced from a finer value - import, export, window
assignment, watermark computation, or metrics - the direction is always floor
toward negative infinity, with no per-call-site choice. Worked examples:
`-1,500 ns` imports to `-2 us` (not `-1`); `-1 ns` imports to `-1 us`;
`1,999 ns` imports to `1 us`; exporting `-1 us` to seconds yields `-1 s` (not
`0 s`); exporting `999,999 us` to seconds yields `0 s`. Rationale (research
section 8.3, plan section 6): flooring event times never moves a row into a
later window, and flooring watermarks keeps the progress estimate
conservative; mixing directions produces both window misassignment and
premature window closes.

**D1.6 Timezone rule.** An Arrow timestamp column declared as an event-time
column MUST be timezone-naive (interpreted as UTC) or carry the explicit
timezone string `"UTC"`. Any other timezone is rejected during source-binding
or compile validation with the column path in the error.

### D2 - Late-data boundary is per window; the per-row criterion is forbidden

**D2.1 Definition.** Within a window operator, a row is **late for window W**
if and only if `window_end(W) <= WM_in`, where `WM_in` is the operator's
current input watermark at the moment the row is evaluated (S5 defines the
value and its monotonicity). Equivalently: the row is late for exactly those
of its assigned windows that have already closed. While `WM_in` is undefined
(S5.2), no row is late and no window closes.

**D2.2 Forbidden criterion.** The alternative rule `event_time <= watermark`
(per-row, whether against the input or output watermark) MUST NOT be used
anywhere in the engine. It is unsound whenever the watermark delay is smaller
than the window size: with a 1-hour tumbling window and a current watermark of
10:30, a row with event time 10:15 belongs to `[10:00, 11:00)`, whose
`window_end` (11:00) is greater than the watermark, so the window is still
open and the row MUST be accepted and accumulated; the forbidden criterion
would drop it. Using the forbidden rule therefore silently discards large
volumes of normal, merely out-of-order data.

**D2.3 Hopping windows.** Lateness is evaluated independently per assigned
window. A row that hits both closed and open windows contributes its closed
assignments to the late metrics and is accumulated normally into the open
windows; the open-window accumulation MUST NOT be counted as late.

**D2.4 Non-window operators (explicit separate decision).** No non-window
operator in 3.0 defines a lateness criterion. Stateless stream operators
(expression, single-alias SQL, union) process every data batch they receive
regardless of any event-time content, and sources never drop data as late.
Any future stateful, time-aware operator that needs a late criterion MUST
define it over its own state boundary (its analogue of `window_end`) through
an amendment to this specification that records the criterion and its
relationship to D2.1; the per-row criterion of D2.2 remains forbidden.

**D2.5 Late-data policy and metrics.** 3.0 implements exactly one late-data
policy: **Drop**. Configurations requesting `allow` or `side_output` MUST fail
at compile time (NG4). The runtime records, per operator (attributable to
window and source): a cumulative count of dropped row-window assignments
(`late_rows`), a count of batches that contained at least one dropped
assignment (`affected_batches`), and the maximum observed lateness, defined as
`max(WM_in - window_end)` over dropped assignments at drop time. Row payloads
MUST NOT appear in metrics or logs. After recovery, watermarks and late
metrics MUST NOT regress and MUST NOT double-count already committed values.

### D3 - Source barrier responsiveness: the runtime prefetch slot model

**D3.1 Chosen model.** The runtime uses **model (b), a runtime prefetch
slot**. Model (a) - requiring every `StreamSource::next()` future to be
cancellation-safe at arbitrary await points and then reused - is rejected.

**D3.2 Structure.** Per source binding, the runtime owns two cooperative
units: a **pump**, which is the only caller of the source's
next-item operation and has at most one such call in flight, and the **source
task**, which consumes items from a **one-item prefetch slot** fed by the
pump, enqueues them onto output edges, and handles control requests (barrier,
cancel, shutdown). The source task never awaits external I/O on its control
path; only the pump does.

**D3.3 Why not cancellation-safe `next()`.** Model (a) makes a property that
the type system cannot check - "dropping this future at any await point loses
no data, advances no cursor, and the source remains fully usable afterwards" -
a per-connector proof obligation, including across the PyO3/asyncio boundary,
where mid-stream cancellation-then-resume is notoriously fragile. A single
non-compliant connector would silently break at-least-once and exactly-once
guarantees. Model (b) concentrates the hard property in one runtime component
that is testable once with paused Tokio time, and gives connector implementors
a weak, mechanically checkable contract (D3.5).

**D3.4 Barrier latency bound.** Under D3.2, external poll latency NEVER
appears on the barrier path. From delivery of the coordinator's barrier
request to the source task, the barrier is placed on all output edges after at
most: (i) O(1) control-message delivery and bookkeeping, and (ii) completion
of at most one in-flight edge enqueue per output edge, which can block only on
downstream backpressure. Any stall on that path beyond the configured
checkpoint timeout is job-fatal (D7), so the enforceable upper bound on
barrier responsiveness is the checkpoint timeout, and a quiet-but-healthy
source - one with no external data for hours - responds to barrier and cancel
requests immediately. An item sitting in the prefetch slot or in the pump's
in-flight poll when the barrier request arrives is post-barrier data (S7.4);
the barrier's snapshot cursor covers only items already enqueued to output
edges.

**D3.5 Obligations on connector implementors.**

1. The next-item call MAY block waiting for external data for an unbounded
   time; it is NOT required to be cancellation-safe, because the runtime never
   cancels it mid-stream.
2. The runtime guarantees exactly one in-flight next-item call per source
   instance and never polls concurrently.
3. A pending next-item call is dropped only at source teardown (graceful
   shutdown, cancel, or failure convergence). After such a drop the runtime
   NEVER calls the next-item operation again on that instance.
4. The source MUST NOT treat an item as consumed until the call returning it
   has completed: the durable cursor advances only through the checkpoint
   protocol (S2.3, S7.4), so an item lost with a dropped poll is re-read on
   recovery (replayable sources) or lost (non-replayable sources, consistent
   with S9's best-effort tier).
5. `close()`/teardown MUST release external resources without requiring the
   dropped call to complete, and MUST tolerate being called with a poll
   dropped mid-flight.
6. The same contract applies to Python sources across the PyO3 boundary:
   teardown cancellation is delivered once, never mid-stream-and-resume. For
   synchronous Python sources the dropped poll is an abandoned blocking call
   and `close()` must be able to unblock it; the mechanics are fixed by the
   M0.2 API note.

### D4 - Checkpoint commit order: validated segments first, manifest last and sole truth

**D4.1 Commit order.** A checkpoint epoch commits in exactly this order:

1. State segments are staged under the managed state root's staging area.
2. Each segment is written durably (flushed and fsynced per the backend's
   durability contract, staging and committed areas on the same filesystem so
   that publication is an atomic rename).
3. Each segment is **validated**: its recorded byte length and SHA-256
   checksum are verified against the staged metadata, re-reading the file as
   the backend requires. A segment failing validation aborts the epoch; no
   manifest for that epoch may ever be published.
4. Each validated segment is **published** into the committed area by an
   atomic rename from its staged path, followed by an fsync of the committed
   parent directory, BEFORE the manifest that references it is published. The
   manifest's `relative_path` always denotes the committed path.
5. The checkpoint manifest - containing, among other fields, the segment
   handles (`operator_id`, `epoch`, `segment_id`, relative path, byte length,
   SHA-256) - is published **atomically and last** (write to a temporary name,
   fsync, atomic rename, fsync the parent directory).

A manifest MUST NOT reference an unvalidated or unpublished segment.

**D4.2 Sole source of truth.** The checkpoint manifest is the ONLY source of
truth for "the most recently completed epoch". Recovery reads manifests only:
it selects the highest-epoch valid manifest and reconstructs state exclusively
from the segments that manifest references. Segments not referenced by any
retained manifest are garbage: they are collected by retention/compaction,
they take no part in recovery decisions, and their presence MUST NOT fail
recovery.

**D4.3 Crash behavior.** Because of D4.1 and D4.2, every crash point has a
deterministic recovery outcome:

| Crash point                                  | Recovery-visible state                                                                           |
| -------------------------------------------- | ------------------------------------------------------------------------------------------------ |
| During segment staging or durable write      | Previous manifest stands; partial segment is orphaned and collected                              |
| After validation, before segment rename      | Previous manifest stands; validated segment orphaned and collected                               |
| After segment rename, before manifest rename | Previous manifest stands; the committed but unreferenced segment is garbage and collected (D4.2) |
| During manifest rename                       | Exactly one manifest version survives; the rename is atomic                                      |
| After manifest rename                        | New manifest stands; every segment it references is validated and committed                      |

**D4.4 Boundedness.** The manifest is a bounded document: it carries metadata
and handles only, never keyed row state. Keyed state of arbitrary size lives
in segments (S9, NFR-2).

**D4.5 Orphan collection never races an in-flight epoch.** Orphan collection
and retention treat as retained: (i) every segment referenced by a retained
manifest, (ii) every segment belonging to an in-flight checkpoint epoch, and
(iii) every segment created after the latest retained manifest. Equivalently:
collection is serialized against checkpoint commits by the coordinator.

### D5 - Task ownership: Drop cancels but never joins; supervisor registry plus runner-scoped reaper

**D5.1 Ownership chain.** Every task the runtime spawns - source pumps,
source tasks, operator tasks, sink tasks, coordinator-driving units, and any
compaction workers - is registered in the job **supervisor's** registry with a
stable task ID (registration order) BEFORE the task is first polled. The
supervisor is owned by the `StreamingJob` handle, and the runner core tracks
live jobs so that runner shutdown can cancel and join them (D5.4). There are
no detached Tokio tasks: every task has a cancellation token and a registered
owner - the supervisor, or the reaper after a job-handle Drop - for as long
as any handle is alive, and every `JoinHandle` held by the supervisor or the
reaper is joined on the paths D5.2 and D5.4 define.

**D5.2 Terminal paths always join.** On the success path (natural completion
of bounded sources), the failure path, the explicit-cancel path, and the
graceful-shutdown path, the owning component cancels as needed (cooperative
cancellation token; every task MUST observe the token at every await
boundary), closes ingress senders, and then **joins every registered task**
before reporting the terminal state (S8). After any of these paths completes,
the supervisor registry is observably empty.

**D5.3 Drop semantics.** `Drop` of the owning `StreamingJob` handle, when no
terminal method (`wait`, `shutdown`, `cancel`) ran first, has the contract
"cancel and release ownership": it triggers the cancellation token and
transfers the supervisor registry to the **runner-scoped reaper**, which
outlives the runner handle (D5.4). Drop MUST
NOT join (it cannot await) and MUST NOT block a Tokio worker thread. Dropping
the `wait()` FUTURE does not cancel the job; dropping the owning job HANDLE
does (plan task M2.4).

**D5.4 Reclamation (the plan's model).** The reaper joins transferred
registries lazily at two defined points: (i) when a new job is started on the
same runner, and (ii) during runner shutdown. The runner core is shared
between the runner handle and every live job handle (the exact ownership
shape is fixed by the M0.2 API note), so the reaper target outlives the
runner handle. Runner shutdown is idempotent and resumable: it cancels and
joins live jobs first and the reaper second, and only then reports closed.
The no-task-outlives guarantee binds on this completed-shutdown path: after
runner shutdown completes, the reaper registry and every job supervisor
registry are observably empty. Dropping the runner handle without shutdown
only fires the cancellation token - it is the sanctioned abandon-joins
escape hatch, MUST be observable (metric plus log), and tests assert it via
token state and task-completion flags, never sleeps. A caller-cancelled
shutdown leaves a live runner that converges on retry. Errors the reaper
observes while joining an abandoned registry (for example a task that
panicked while converging after a job-handle Drop) are logged and recorded
in metrics, and never fail the job that triggered the reaper pass.
Cancellation convergence is guaranteed because every
task observes the token cooperatively (D5.2); a task that never observes
cancellation is an implementation defect, not a semantic case.

**D5.5 Test assertions.** Tests assert reclamation exclusively through
observable state - supervisor registry length, reaper registry length,
per-task completion flags or `JoinHandle::is_finished()`, resource handles
(e.g., a mock source's closed flag), and completed checkpoint epochs - and
NEVER through sleeps, wall-clock waits, or other timing assertions. Paused
Tokio time MAY be used to drive timers deterministically.

### D6 - State segment format: Arrow IPC

The first-version state segment format is **Arrow IPC file format**. Rationale:
it preserves Arrow schemas and types (including nested types) losslessly,
supports incremental appends and streaming reads without an additional codec
dependency in the core crate, and keeps the segment envelope simple enough that
D4's length/checksum validation covers the whole file. Parquet remains a
connector-level format for sinks and is not a state segment format in 3.0.
Segments are immutable once validated; compaction reads committed segments and
writes new ones, never mutating in place. Compression inside the segment
envelope MAY be revisited after M4 with paired benchmark evidence from the
M0.3 baseline; any such change is a spec amendment.

### D7 - Checkpoint timeout is job-fatal

A checkpoint attempt is bounded end to end - from barrier injection through
alignment, snapshots, segment validation, and manifest publication - by the
configured checkpoint timeout. On expiry the job does NOT skip the epoch and
does NOT continue silently: the job converges to a terminal state with a
dedicated, deterministic checkpoint-timeout error (classified per S8.4). The
abandoned epoch leaves no manifest and therefore, per D4.2, never happened;
partially written segments are orphaned and collected. Transactional-sink
external commits happen strictly after manifest publication (S7.5) and are
retried idempotently under the job's normal error rules; they are not part of
the alignment timeout window.

### D8 - Window geometry: fixed origin, half-open intervals, integral overlap

- **D8.1** Windows are aligned to a fixed origin at the Unix epoch; 3.0 has no
  configurable window offset.
- **D8.2** Windows are half-open intervals `[window_start, window_end)`. A row
  at exactly `window_end` belongs to the next window.
- **D8.3** Window assignment uses floor division (toward negative infinity)
  over internal microsecond event times, so pre-1970 timestamps are assigned
  correctly and consistently with D1.5.
- **D8.4** 3.0 requires the hopping-window size to be an exact positive
  integer multiple of the slide: `size = m x slide` for integer `m >= 1`
  (this is the plan's M4.3 rule, approved here as the general rule; arbitrary
  non-integral size/slide ratios are rejected at compile time). Tumbling
  windows are the special case `slide = size`. Every row is then assigned to
  exactly `m` windows.
- **D8.5** A window closes, and its final result is emitted, when the
  operator's input watermark first satisfies `window_end <= WM_in` (the same
  boundary as D2.1). Emission and ordering semantics are frozen in S6.

### D9 - Epoch numbering and checkpoint request serialization

- **D9.1** `Epoch` is a newtype over `u64`. Value `0` is reserved as the
  "no checkpoint" sentinel and is never injected.
- **D9.2** A fresh job's first checkpoint is epoch `1`. Each successfully
  injected checkpoint barrier increments the epoch by exactly `1`. Epochs are
  strictly increasing within a job lineage.
- **D9.3** Recovery from a manifest at epoch `R` continues the lineage: the
  next checkpoint after recovery is epoch `R + 1`. Lineage continuity is keyed
  by pipeline fingerprint plus state root; starting from a fresh state root
  restarts numbering at `1`.
- **D9.4** An epoch that never got a manifest (timeout per D7, cancel per
  S7.6, crash per D4.3) never happened. Within the abandoning job the number
  is never reused, because that job is terminal (D7, S7.6). The recovering
  job re-issues the same number as its first checkpoint (`R + 1` per D9.3):
  the abandoned attempt left no manifest, and the recovering job replays the
  same deterministic prefix, so re-issued epochs are safe and keep
  `(fingerprint, sink, epoch)` idempotency keys stable (S7.5, S9.3).
- **D9.5** 3.0 allows exactly one in-flight checkpoint. Concurrent checkpoint
  requests (periodic and manual) are serialized FIFO: each queued request
  eventually injects its own strictly increasing epoch. Requests are NOT
  coalesced and NOT rejected in 3.0.
- **D9.6** A lineage (pipeline fingerprint plus state root) has exactly one
  active job. The managed state root and checkpoint store are exclusively
  locked by the running job (M4.2); a second `start()` against the same
  lineage fails with a conflict error (`CalcFlowError::Conflict`) before any
  source opens.

## 3. Semantic model

### S1 - Messages and per-edge ordering

- **S1.1** One edge, one producer, one consumer, one sequence: data and all
  control messages on the same edge share a single FIFO order. A barrier or
  watermark never overtakes data submitted before it on the same edge; data
  never overtakes control submitted before it.
- **S1.2** A source task enqueues all pre-barrier data onto its output edges
  before placing the barrier. The coordinator NEVER writes to a graph edge
  directly; it requests barriers from source tasks (S7.2).
- **S1.3** The runtime is the sole owner of control forwarding. Operators may
  emit data and closed-window results in response to watermarks through the
  collector; they MUST NOT forge, suppress, reorder, or emit barriers or
  epochs, and MUST NOT emit arbitrary watermarks. Control message construction
  is crate-private; sources create control events only through validated
  constructors, and validation (watermark monotonicity, epoch consistency)
  happens before enqueue, before any downstream side effect.
- **S1.4** `Idle` excludes an ingress from the watermark minimum only. It does
  NOT exempt the ingress from barrier alignment (S7.3), and an idle source
  still injects barriers.
- **S1.5** When new data arrives on an idle ingress, the ingress first returns
  to Active, and only then is the data evaluated for lateness (D2). The output
  watermark never retreats because of a reactivation.
- **S1.6** `EndOfInput` terminates one ingress permanently. An ended ingress
  is excluded from the watermark minimum and from the barrier alignment set.
  When ALL of an operator's ingresses have ended, the operator flushes exactly
  once (S6.3 for windows) and the runtime forwards exactly one `EndOfInput`
  per output edge. No synthetic terminal watermark is introduced. An ended
  source reports its final cursor and never reopens within the job.

### S2 - Source progress: sequence and cursor rules

- **S2.1 Sequence.** Each source binding owns a `u64` sequence per job
  lineage: it starts at `0` on the first-ever open, increases by exactly `1`
  per emitted item, is recorded in every checkpoint manifest, and continues
  from `last_sequence + 1` after recovery. It is never reused within a
  lineage. The binding ID is the authoritative source identity carried in
  batch metadata; metadata is written immutably, preserving existing
  attributes.
- **S2.2 Cursor monotonicity.** Within one run, the cursor reported with each
  item MUST strictly increase in the source's own order. The runtime enforces
  this by comparing each item's ordered cursor key bytewise against the
  ingress's last enqueued key; a repeated or regressed key fails closed before
  enqueue (the same pre-side-effect treatment as a regressed watermark, S5.4).
- **S2.3 Observed vs durable.** The runtime distinguishes the latest observed
  item/cursor (volatile, may be ahead) from the latest durable cursor
  (advanced only by a completed checkpoint manifest, D4). The barrier snapshot
  cursor is defined in S7.4.
- **S2.4 Duplicates.** Within one fault-free run the runtime emits no
  duplicates itself. After crash and recovery, items beyond the durable cursor
  are replayed, so sinks MAY observe duplicates: this is the at-least-once
  tier of S9. Exactly-once deduplication is a sink-side protocol property
  (S7.5), not a source property.
- **S2.5 Recovery.** A source opens at the resume cursor from the latest
  manifest (or at its configured beginning when no manifest exists) before any
  polling starts, and MUST honor that cursor exactly; a source that cannot
  honor it fails closed during open, before any item is emitted.

### S3 - Fan-out ordering

Each output edge of a node is an independent channel with its own FIFO and its
own accounting (S10). Fan-out clones share the immutable batch payload (Arrow
buffers are not copied), but each edge is charged independently - a logical
queue charge, not a claim about process RSS. There is NO ordering guarantee
between sibling branches: two downstream branches may observe the same item at
arbitrary relative times, bounded only by their own backpressure. Tests MUST
NOT assert cross-branch interleaving order.

### S4 - Multi-ingress selection (union and beyond)

Every ingress keeps its own FIFO. When several ingresses of one operator have
ready items, the selection order among them is unspecified: no business order,
no round-robin promise, no event-time merge. An implementation MAY be
deterministic for testability, but the contract forbids relying on any
particular selection order, and correctness properties (watermark minimum,
barrier alignment, final results) MUST hold independently of it.

### S5 - Watermark model

- **S5.1 Monotonicity.** On every ingress, watermarks are monotone
  non-decreasing. A source-provided watermark below the ingress's last
  watermark fails closed before enqueue. A generated watermark (S5.6) is
  computed so that later observations can never lower it.
- **S5.2 Output watermark.** An operator's input watermark `WM_in` is the
  minimum of the last-received watermarks over its primed Active ingresses.
  Idle ingresses (S1.4) and ended ingresses (S1.6) are excluded. An Active
  ingress that has delivered no watermark yet is unprimed: `WM_in` is
  undefined until every Active ingress is primed, no windows close, and no
  rows are late (D2.1). If all non-ended ingresses are idle, `WM_in` does not
  advance. The runtime forwards an output watermark only when the computed
  value strictly advances.
- **S5.3 Reactivation.** Data on an idle ingress returns it to Active before
  lateness evaluation (S1.5); the reactivated ingress may immediately produce
  late rows, but the output watermark never retreats.
- **S5.4 Fail-closed validation.** Watermark regression, cursor regression
  (S2.2), and epoch inconsistencies (S7.3) are detected and rejected before
  the offending message is enqueued - before any downstream observation or
  side effect.
- **S5.5 End and final flush.** When all ingresses have ended, the operator
  flushes once (for window operators: every still-open window emits exactly
  one final result, S6.3) and the runtime forwards one `EndOfInput` per
  output edge.
- **S5.6 Source watermark policies (3.0).** Exactly three policies exist:
  `SourceProvided`; `BoundedOutOfOrderness { event_time_column, delay,
  emit_interval, idle_timeout }`, which emits `max(observed event time) -
  delay` on a Tokio-time schedule and declares the source idle after
  `idle_timeout` without data; and `Disabled`. A missing or non-timestamp
  event-time column fails validation with the source and column path. Rows
  with null event times never produce watermarks, and an all-null batch
  produces no new watermark. Policy state and per-ingress progress are part of
  checkpoint state; after recovery, watermarks MUST NOT retreat. 3.0 supports
  table sources with a timestamp column and a fixed delay only; arbitrary SQL
  watermark expressions are postponed.

### S6 - Final-only tumbling/hopping windows

- **S6.1 Assignment.** Rows are assigned to windows per D8 (fixed origin,
  half-open intervals, floor division, `size = m x slide`). Rows whose event
  time is null cannot be assigned to any window; their handling (drop with a
  dedicated metric vs. rejection) is fixed by task M4.3, not by implementation
  convenience (see section 9).
- **S6.2 Trigger.** A window closes when the operator's input watermark first
  reaches or passes `window_end` (D8.5, same boundary as D2.1). On every
  input-watermark advance, the operator emits all newly closed windows BEFORE
  the runtime forwards the watermark downstream (handler-before-forward).
- **S6.3 Final-only emission.** Within one continuous fault-free run, each
  window is emitted at most once, at close (or at the all-ended flush, S5.5).
  Empty windows (no contributing rows for any group) emit nothing. After a
  failure and replay, an operator MAY re-emit an already closed window;
  therefore "the same window closes only once" is guaranteed exclusively at a
  transactional or idempotent sink boundary (S9), while the operator-layer
  guarantee is that replay converges to identical final values.
- **S6.4 Output order.** Windows emitted in one close event are ordered by
  `(window_start, window_end, stable group-key encoding)`; across close
  events, order follows watermark advancement. The group-key encoding is
  total, deterministic across runs and recoveries, and identical for identical
  key values; its exact byte encoding is an API-note decision (section 9).
  Output rows carry `window_start`, `window_end`, the group keys, and the
  named aggregates; window bounds are exported at full internal precision
  (timestamp microseconds; timezone annotation is an API-note decision).
- **S6.5 State lifecycle.** 3.0 aggregates are `count`, `sum`, `min`, `max`,
  `avg` over an explicitly enumerated type matrix (fixed by task M4.3).
  Incremental accumulators are Calc-Flow-owned (no DataFusion private planner
  nodes or forks, NG8). Only dirty keys are flushed per checkpoint. An emitted
  window's state is tombstoned only after the output is durable under the
  checkpoint protocol (S7); compaction MUST NOT change logical rows or their
  deterministic order.

### S7 - Epochs, barriers, alignment, timeout, cancellation, cursor boundary

- **S7.1 Coordination.** The coordinator owns epoch numbering (D9), sends a
  barrier request to every source task, collects source, operator, and sink
  acknowledgements, and publishes the manifest (D4).
- **S7.2 Source injection.** On a barrier request, a source task stops drawing
  from the prefetch slot (D3), records the snapshot cursor (S7.4), places
  `Barrier(epoch)` on ALL its output edges after all pre-barrier data (S1.2),
  acknowledges the cursor to the coordinator, and resumes. Idle sources inject
  barriers normally (S1.4); ended sources report their final cursor and do not
  reopen (S1.6). The coordinator never writes edges directly (S1.2).
- **S7.3 Multi-input alignment.** An operator task records barrier state per
  ingress. When `Barrier(E)` arrives first on ingress I, I is blocked - the
  task stops consuming I without draining or reordering it - while other
  ingresses keep flowing. When every non-ended required ingress has delivered
  `Barrier(E)`, the task performs its snapshot for epoch `E`, stages dirty
  state, and then acknowledges the coordinator and forwards the barrier in the
  order fixed by the M0.2 API note (section 9) - the verb sequence in this
  sentence is not an ordering claim; the only ordering constraint is that the
  snapshot precedes both the acknowledgement and the forwarding; snapshot
  failure MUST NOT forward and MUST fail the job. A barrier carrying a future or regressed epoch fails
  closed. Idle ingresses are NOT exempt from alignment; ended ingresses are
  excluded (S1.6).
- **S7.4 Cursor boundary.** The barrier's snapshot cursor covers exactly the
  items already enqueued to output edges before the barrier - no more, no
  less. An item in the prefetch slot or in the pump's in-flight poll (D3.4) is
  post-barrier. Pre-barrier data is inside the operator snapshot; post-barrier
  data is not. The acknowledged source cursors and the edge barriers together
  describe one replayable prefix of the job's inputs.
- **S7.5 Completion and sink protocol.** Sinks pre-commit only after receiving
  the epoch's barrier and all pre-barrier data. The manifest is published
  after all source/operator/sink acknowledgements and segment validation
  (D4.1). Transactional sinks perform their external commit strictly after the
  manifest is durable, retry it idempotently, and on recovery complete any
  durable pre-commit recorded in the manifest without rewriting data: the
  existence of the manifest marks the epoch's state complete, and per-sink
  completion is derived by the sink recovery protocol from the manifest's
  pre-commit metadata (no separate all-committed flag in 3.0; whether the
  manifest additionally carries an explicit recovery-status field is a layout
  decision for M5.1/M5.4). A failure before manifest publication aborts staged
  sink output; for a crash-abandoned epoch, the recovering job's
  sink-recovery pass aborts or removes staged artifacts not recorded in the
  latest manifest (the sink-side recovery contract is fixed by the M0.2 API
  note). A failure after publication MUST NOT abort output that recovery
  needs.
- **S7.6 Cancellation and timeout.** Cancellation always wins over an
  in-flight checkpoint: the epoch is abandoned, partial segments are orphaned
  (D4.3), no manifest exists, and the job converges to cancelled (S8).
  Checkpoint timeout is job-fatal per D7.

### S8 - Job lifecycle, terminal states, deterministic error selection

- **S8.1 States.** A job is in exactly one of: `running`, `draining`,
  `completed`, `cancelled`, `failed`, `recovery-required`. `running` and
  `draining` are non-terminal; the rest are terminal and immutable once
  reached.
- **S8.2 Transitions.** `running -> draining` on graceful shutdown request or
  when all sources have ended; `draining -> completed` after accepted data is
  drained to sinks, sinks are flushed, and a final epoch manifest (numbered
  per D9) is published. `cancelled` is reached from any non-terminal state via
  explicit cancellation, with no drain promise. `failed` or
  `recovery-required` is reached from any non-terminal state via failure
  convergence (D5.2).
- **S8.3 Terminal-state classification.** The mapping is deterministic:
  logic/deterministic error classes (operator errors, validation failures,
  invariant violations, panics) yield `failed`; infrastructure/transient
  classes (I/O, store, network, external system, checkpoint timeout per D7)
  yield `recovery-required`. Both are fully joined terminal states (D5.2).
  Recovery in 3.0 ALWAYS means starting a NEW job from the latest completed
  manifest; there is no in-place restart. The `failed`/`recovery-required`
  distinction is diagnostic: it declares whether plain restart-from-manifest
  is expected to make progress.
- **S8.4 Deterministic multi-error selection.** The supervisor serializes all
  terminal triggers through a single decision point. The first recorded
  trigger wins. Triggers observed in the same scheduling round are ordered by
  a fixed priority: (1) task failure or panic, (2) explicit cancel,
  (3) deadline expiry. When several task errors have already been observed
  simultaneously, they are returned ordered by stable task ID (D5.1), the
  first being the primary error and the rest attached in order. A panic
  becomes a typed internal error carrying the task identity. Deadline expiry
  and explicit cancel converge to one unique terminal state.
- **S8.5 Observability.** `wait()` is idempotent and returns the same terminal
  result to every caller. `status()` returns a deterministic snapshot -
  per-edge queue state, per-source progress, watermarks, current epoch, and
  job state - with stable ingress/task ID ordering (I2).

### S9 - Delivery guarantees and the capability matrix

- **S9.1 Matrix.** The attainable guarantee is a per-sink derived property of
  the whole pipeline, never a global slogan:

| Source capability                       | Operator class | Backpressure | Checkpointing         | Sink capability                                             | Attainable guarantee                            |
| --------------------------------------- | -------------- | ------------ | --------------------- | ----------------------------------------------------------- | ----------------------------------------------- |
| Non-replayable, or lossy (`DropOldest`) | any            | any          | any                   | any                                                         | Best-effort: loss and duplication both possible |
| Replayable, barrier-aligned cursor      | any            | `Block`      | Aligned epoch         | Ordinary                                                    | At-least-once                                   |
| Replayable, barrier-aligned cursor      | Deterministic  | `Block`      | Durable aligned epoch | Transactional, or epoch-idempotent with unbounded retention | Exactly-once                                    |

- **S9.2 Compile-time enforcement.** When a project requests
  `delivery = "exactly_once"`, compilation MUST fail before any external side
  effect (before any source open) if any required capability is missing:
  replayable source with barrier-aligned cursor, deterministic operators,
  `Block` backpressure on every relevant edge, durable aligned-epoch
  checkpointing, and a transactional sink, or an epoch-idempotent sink with
  an unbounded retention class and a declared basis (mechanism identifier
  plus retention class, recorded per S9.4); a bounded retention class fails
  this requirement with the configuration path named. A
  `DropOldest` policy anywhere on the path makes exactly-once compilation
  fail. The diagnostic names the precise configuration path that fails.
- **S9.3 Determinism.** Exactly-once requires operators that are pure
  functions of their inputs and checkpointed state; UDFs with volatile
  DataFusion volatility are incompatible with an exactly-once plan and fail
  compilation. Watermark, late-drop, and window semantics (D2, S5, S6) are
  deterministic functions of the replayed prefix.
- **S9.4 Reporting.** Guarantees are reported per sink (status snapshots,
  Studio, manifest delivery-capability records), and every report carries
  its evidence: a transactional binding reports its commit protocol, and an
  epoch-idempotent declaration reports its mechanism identifier and
  retention/expiry class (for example, a bounded dedup-token window reports
  as retry-deduplicated within that window, never as unconditional
  exactly-once). A sink's declared delivery capability is sampled once at
  binding validation and frozen in the compiled plan's per-sink delivery
  record; a capability that changes mid-job is a connector defect. An
  ordinary sink or a lossy source is NEVER reported as exactly-once.

### S10 - Bounded channels and backpressure

- **S10.1 Dual limits.** Every edge channel enforces TWO hard limits, checked
  atomically at enqueue time: logical rows and estimated bytes. Reservation
  happens before enqueue; release happens exactly once, when the receiver
  dequeues or the reservation is dropped; a cancelled send releases its
  reservation exactly once. Closing the receiver wakes all blocked senders
  with a closed signal.
- **S10.2 Accounting.** Table batches are charged the Arrow memory size of
  their visible slices; external payloads MUST provide an exact or
  conservative byte estimate with no opt-out. Sums use checked arithmetic;
  overflow is a typed error. Charges are logical reservations, not RSS
  measurements.
- **S10.3 Oversize.** A single message larger than the edge's byte limit fails
  at enqueue time; there is no "one oversize message" exception.
  Compilation on the project path, or runner construction on the builder
  path, MUST validate that every source's configured maximum batch bytes/rows
  fit within its downstream edges' capacities; the check always runs before
  any source opens. (Migration note for M1.3: the
  legacy `BatchingSource` historically emitted a first oversize item instead
  of failing; v3 unifies on fail-before-enqueue, and `CHANGELOG.md` records
  the behavior change.)
- **S10.4 Policies.** The default and only implicit policy is `Block`:
  senders await capacity, so backpressure propagates sink -> operator ->
  source task -> prefetch slot -> pump, ultimately pausing source polling.
  `DropOldest` MAY exist only where a user explicitly configures a lossy
  source; it MUST be observable through metrics and MUST make exactly-once
  compilation fail (S9.2).
- **S10.5 Implementation freedom.** The reservation mechanism (e.g., a single
  mutex-protected budget with notify) is an M1.4 implementation concern; the
  semantics are: atomic two-dimension reservation, exactly-once release, FIFO
  per S1, and no lost wakeups under the single-producer-per-edge invariant.

## 4. Program-wide invariants

- **I1 Safety.** `unsafe_code = "forbid"` everywhere, including connectors and
  the state backend.
- **I2 Deterministic ordering.** Manifest serialization, fingerprints, error
  aggregation, metrics snapshots, status snapshots, and generated documents
  use deterministic `BTreeMap` ordering wherever output is serialized or
  observed.
- **I3 No executable objects in configuration.** Configurations, project
  documents, manifests, and checkpoints carry data and handles only -
  `UdfReference(provider, name, version)` values, connector identities of the
  same shape, secret references - never source text, callables, import paths,
  or serialized objects.
- **I4 Secrets.** Secret values NEVER enter project documents, fingerprints,
  logs, metrics, state segments, checkpoint manifests, status snapshots, or
  error messages. Connectors resolve secrets through references at open time
  only. Connector errors raised after secret resolution MUST be wrapped so
  that client-library messages - which may echo resolved values, URLs, or
  auth headers - never propagate raw into the engine's error type; the
  wrapped error carries the connector identity and a sanitized message.
- **I5 Task ownership.** No detached Tokio tasks; the D5 ownership chain holds
  for every runtime-spawned task, including compaction workers. Connector
  implementations MUST NOT spawn Tokio tasks: blocking client work runs on
  the I6 mechanisms (blocking pools, dedicated threads, or true async I/O),
  and client-internal OS threads (such as a bundled client's background
  thread) are external resources owned by the connector instance and released
  by `close()`, not runtime tasks.
- **I6 Non-blocking executor.** File I/O, compression, bulk serialization and
  encoding, connector clients, and compaction MUST NOT block Tokio executor
  threads; they run on blocking pools, dedicated threads, or true async I/O.
  State capture on the barrier path is a cheap metadata handoff rather than
  bulk encoding (the capture-cost rule is fixed by the M0.2 API note).
- **I7 Fixtures.** `tests/fixtures/v1/` is immutable (NG13); harnesses that
  read it migrate to the v3 batch-compile path without modifying fixture data.
- **I8 Tests.** Correctness tests assert observable state, never timing
  (D5.5); every behavior change starts with a focused failing test that
  records the expected RED reason (plan section 5); container integration
  tests never mix into ordinary unit-test runs.
- **I9 Immutability.** `Batch` remains the immutable public data envelope;
  fan-out shares payloads without copying (S3); no caller-owned value is ever
  mutated.
- **I10 Single producer per edge.** Every edge has exactly one writer; this is
  a compile-checked invariant and a precondition of S10's reservation design.

## 5. Functional requirements

Each requirement cites its defining section; acceptance tests live in section 8.

- **FR1** - Data and control messages on one edge observe a single FIFO order
  (S1.1), and a source-placed barrier never overtakes pre-barrier data (S1.2,
  S7.2).
- **FR2** - Control construction and forwarding are runtime-owned and
  validated before enqueue; operators cannot emit barriers, epochs, or
  arbitrary watermarks (S1.3, S5.4).
- **FR3** - Per-source sequences start at 0, increase by exactly 1 per item,
  persist in manifests, and continue across recovery (S2.1).
- **FR4** - Cursors strictly increase within a run; regression or repetition
  fails closed before enqueue; durable cursors advance only via completed
  manifests (S2.2, S2.3).
- **FR5** - Recovery opens sources at the manifest resume cursor before
  polling; failure to honor it fails closed during open (S2.5).
- **FR6** - Fan-out edges deliver identical per-edge FIFO sequences with
  shared immutable payloads, independent per-edge charges, and no cross-branch
  order guarantee (S3).
- **FR7** - Union preserves per-ingress FIFO and promises no cross-ingress
  business order (S4).
- **FR8** - Watermarks are monotone non-decreasing per ingress; regressions
  fail before enqueue (S5.1, S5.4).
- **FR9** - The output watermark is the strict-advance-only minimum over
  primed Active ingresses; idle and ended ingresses are excluded; reactivation
  never retreats the watermark; all-ended triggers exactly one flush and one
  forwarded `EndOfInput` per output (S5.2, S5.3, S5.5, S1.5, S1.6).
- **FR10** - Source watermark policies are exactly the three variants of S5.6
  with the stated validation, null handling, idle, and checkpoint behavior.
- **FR11** - `EventTime` uses the internal representation, checked
  conversions, floor truncation, serialization precision, and timezone rule of
  D1.
- **FR12** - Lateness is decided per window by `window_end <= WM_in`; the
  per-row criterion of D2.2 is forbidden; hopping rows are evaluated per
  assigned window (D2.1-D2.3).
- **FR13** - The only late policy is Drop, with the D2.5 metrics and no
  payloads; `allow`/`side_output` configurations fail at compile time (D2.5,
  NG4).
- **FR14** - Source barrier responsiveness uses the D3 prefetch-slot model
  with its latency bound and connector obligations.
- **FR15** - Barrier injection, multi-input alignment, epoch validation, and
  the cursor boundary follow S7.2-S7.4; idle does not exempt alignment and
  ended ingresses are excluded (S7.3).
- **FR16** - Checkpoint timeout is job-fatal per D7; cancellation abandons the
  epoch with no manifest (S7.6, D4.3).
- **FR17** - Checkpoint commit order (stage, durable write, validate, publish
  segment, publish manifest), manifest sole-truth recovery, orphan garbage
  collection that never races an in-flight epoch, and manifest boundedness
  follow D4.
- **FR18** - State segments use the Arrow IPC format and are immutable once
  validated (D6).
- **FR19** - Window geometry (D8) and final-only trigger, emission, ordering,
  and state lifecycle (S6) hold; session/early/allowed-lateness/update
  configurations fail at compile time (NG4).
- **FR20** - Epochs follow D9: start at 1, increment by 1, recovery continues
  the lineage, abandoned epochs never happened, concurrent requests serialize
  FIFO without coalescing.
- **FR21** - Job states, transitions, terminal classification, deterministic
  multi-error selection, panic typing, and idempotent `wait()` follow S8.
- **FR22** - Delivery guarantees derive from the S9.1 matrix; exactly-once
  requests fail compilation before side effects when any capability is
  missing; guarantees are reported per sink (S9.2-S9.4).
- **FR23** - Edge channels enforce rows and bytes atomically at enqueue with
  exactly-once reservation release and sender wakeup on close; oversize single
  messages fail before enqueue and are pre-validated at compilation or runner
  construction, always before any source opens (S10.1-S10.3).
- **FR24** - `Block` is the default backpressure policy; `DropOldest` is
  explicit, observable, and exactly-once-incompatible (S10.4).
- **FR25** - Task registration, joining terminal paths, Drop-cancels-never-
  joins, and reaper reclamation follow D5.
- **FR26** - Invariants I1-I10 hold across all surfaces (section 4).

## 6. Non-functional requirements

- **NFR-1 Performance.** M0.3 records Criterion baselines (expression, SQL,
  external passthrough, DataFusion runtime creation, checkpoint persistence)
  plus the frozen allocation-regression harness state at the exact starting
  commit. The performance regression gate for later milestones is 5%, and it
  MUST be supported by same-machine paired data with confidence intervals, not
  single point estimates. Stream-runtime costs are attributed separately from
  existing DataFusion, Python, and Studio costs.
- **NFR-2 State scale.** Keyed state larger than 10 MiB checkpoints and
  restores correctly while the manifest itself stays bounded (D4.4). Checkpoint
  duration is analyzed against dirty-key volume, not total retained state.
- **NFR-3 Memory.** Steady-state queue charges never exceed configured edge
  budgets (S10), and a one-hour two-source slow-sink soak shows no sustained upward
  memory trend. External array payloads carry conservative byte accounting.
- **NFR-4 Recovery.** Cold-cache and warm-cache recovery are measured
  separately. Crash consistency is proven by the M5.5 fault matrix at every
  listed injection point, asserting recovered cursors, watermarks, window
  state, sink-visible rows, duplicate/missing counts, staged-artifact cleanup,
  latest completed epoch, and deterministic terminal errors.
- **NFR-5 Compatibility.** The breaking changes of plan section 1.3 are the
  only sanctioned ones: v2 project/checkpoint documents are rejected with
  `UnsupportedVersion` (NG9); `MicroBatchRunner` is replaced by bounded-source
  stream runs; `StreamingRunner` is redefined as the source-driven continuous
  runner; Python v2 signatures and Studio `/api/v2` are removed without
  aliases; `CHANGELOG.md` and the migration guide document each replacement
  path. The semantic fingerprint (execution mode, graph, operator
  configuration, UDF catalog, state-layout-affecting window/state semantics)
  determines checkpoint compatibility; runtime-tunable values (channel
  capacities, checkpoint interval) live in a separate runtime-config hash and
  MUST NOT invalidate checkpoints. Connector options stay out of the semantic
  fingerprint; instead, the manifest records a connector-declared source
  identity hash covering the data-semantics-affecting options (such as
  publication, query, topic, table, URL, or file paths), and recovery fails
  closed on mismatch. The per-connector split of options into
  data-semantics-affecting versus credential/transport is owned by M6.1's
  capability surface (section 9). Secret references feed neither the
  fingerprint nor the identity hash.
- **NFR-6 Observability.** Metrics cover input/output batches, rows, bytes,
  errors, blocked sends and durations, queue high-water marks, per-edge
  charges, source/sink progress, late data (D2.5), and job state, with no
  high-cardinality labels such as batch IDs (I2).

## 7. Inputs and outputs

Semantic quantities this specification fixes (Rust-facing representation;
exact type names belong to the M0.2 API note):

| Quantity                   | Representation                              | Units                              | Constraints                                                                |
| -------------------------- | ------------------------------------------- | ---------------------------------- | -------------------------------------------------------------------------- |
| `EventTime` (internal)     | newtype over `i64`                          | microseconds since Unix epoch, UTC | total order; full `i64` range; raw `i64` never public (D1.1)               |
| `EventTime` (durable)      | `i64` in manifests/metadata                 | microseconds since Unix epoch, UTC | exact internal value; lossless round-trip (D1.2)                           |
| Arrow timestamp import     | Arrow `timestamp(s/ms/us/ns)`               | column's declared unit             | checked conversion, floor for ns, overflow errors with column path (D1.3)  |
| Arrow timestamp export     | Arrow `timestamp(s/ms/us/ns)`               | target unit                        | exact or checked for finer/equal; floor for coarser (D1.4, D1.5)           |
| Window size / slide        | `u64` duration                              | microseconds                       | both > 0; `size = m x slide`, integer `m >= 1` (D8.4)                      |
| Watermark                  | `EventTime`                                 | microseconds                       | monotone non-decreasing per ingress (S5.1)                                 |
| `Epoch`                    | newtype over `u64`                          | count                              | starts at 1; +1 per injected checkpoint; 0 reserved (D9)                   |
| Sequence                   | `u64`                                       | items per source lineage           | starts at 0; strictly +1 per item; continues across recovery (S2.1)        |
| Cursor                     | ordered key plus opaque payload (section 1) | source-defined order               | strictly increasing within a run; durable only via manifest (S2.2, S2.3)   |
| Late metrics               | counters and a maximum gauge                | rows, batches, microseconds        | cumulative; no payloads; never regress after recovery (D2.5)               |
| Edge budget                | `(max rows, max estimated bytes)`           | rows and bytes                     | both > 0; enforced atomically at enqueue (S10.1, S10.2)                    |

## 8. Acceptance criteria

M0.1 artifact gate (this milestone):

- [ ] This artifact exists at
  `.codex/artifacts/specs/continuous-streaming-runtime.md` and Appendix A maps
  every plan M0.1 checkbox to the sections that freeze it; no checkbox is
  unmapped.
- [ ] D1-D5 are frozen with normative language and contain no TBD; every
  remaining choice is listed in section 9 with an explicit owner.
- [ ] The M0.2 API note and critique use the same
  `continuous-streaming-runtime` slug, and no definition in those artifacts
  contradicts this specification (plan M0.2 acceptance gate).
- [ ] The artifact lands as a documentation-only PR whose body cites this
  controlling artifact and follows plan section 16 (no production code, no
  test code).

Executable acceptance tests for M1-M7 (each semantic rule has at least one;
paused Tokio time is used wherever timers appear; no test uses timing
assertions, I8):

- [ ] **AC-D1** Conversion property test: random `i64` values across all four
  Arrow units either raise a checked error with the column path or convert to
  the mathematically floored microsecond value; `-1,500 ns -> -2 us`,
  `-1 ns -> -1 us`, `1,999 ns -> 1 us`; exporting `-1 us` yields `-1 s` and
  `999,999 us` yields `0 s`; `EventTime` orders pre- and post-epoch values;
  near-`i64::MAX` second/millisecond inputs error instead of wrapping;
  non-UTC timezones are rejected while naive and `"UTC"` are accepted.
- [ ] **AC-D2** Window late-boundary test: 1-hour tumbling window, watermark
  10:30, row at 10:15 is accepted and accumulated and absent from late
  metrics; watermark 11:05 makes a row for `[10:00, 11:00)` late with
  `late_rows + 1` and maximum lateness at least 5 minutes; a hopping row
  hitting one closed and one open window is dropped only for the closed
  assignment; a project configuring `allow` or `side_output` fails
  compilation.
- [ ] **AC-D3** Barrier-responsiveness test (paused time): with a source
  blocked in its next-item call, a barrier request completes and the barrier
  is observed on all output edges; the acknowledged cursor covers only
  pre-barrier items; when the blocked poll later resolves, its item is
  enqueued after the barrier. Teardown drops the pending call, never polls
  the instance again, and leaves the durable cursor unchanged (observed via a
  mock source's flags, not sleeps).
- [ ] **AC-D4** Commit-order fault test: injected crashes at each row of the
  D4.3 table - including the window between the segment rename and the
  manifest rename - recover to the stated manifest; orphaned and
  committed-but-unreferenced segments are collected and never fail recovery;
  a segment failing checksum/length validation aborts the epoch and no
  manifest references it. This injection-point set is handed to the M5.5
  fault matrix.
- [ ] **AC-D5** Ownership test: success, failure, cancel, and shutdown paths
  each end with an observably empty supervisor registry; dropping the job
  handle followed by runner shutdown ends with an empty reaper registry and
  set task-completion flags; runner shutdown with a live job cancels and
  joins the job first and the reaper second, ending with both registries
  empty; a caller-cancelled shutdown leaves a live runner that converges on
  retry; a bare runner-handle drop fires the token, is observable via metric
  and log, and is asserted via token state and task-completion flags. All
  assertions are on observable state.
- [ ] **AC-S1** Mixed data/watermark/barrier sequences on one edge are
  observed in submission order; a coordinator has no API path that writes to
  an edge directly.
- [ ] **AC-S2** Sequence test: sequences are strictly +1 per item per source;
  a regressed cursor is rejected before enqueue; recovery reopens at the
  durable cursor and continues the manifest sequence.
- [ ] **AC-S3/S4** Fan-out test: every branch receives the identical per-edge
  FIFO sequence while the test asserts nothing about cross-branch
  interleaving. Union test: per-ingress FIFO holds under interleaved
  readiness, all items are delivered, and watermark/barrier behavior is
  independent of selection order.
- [ ] **AC-S5** Watermark tests: minimum over two active inputs; a fast input
  cannot outrun a slow active input; idle inputs are excluded; reactivation
  can produce late rows but never retreats the output watermark; ended inputs
  are permanently excluded; all-ended produces exactly one flush and one
  `EndOfInput` per output; a regressed source-provided watermark fails before
  enqueue; an all-null timestamp batch produces no watermark.
- [ ] **AC-S6** Window tests: pre- and post-1970 assignment follows floor
  division; a window closes exactly when `window_end <= WM_in`; closed
  windows are emitted before the watermark is forwarded (observable
  downstream); within one run each window is emitted at most once; replay
  after failure converges to identical final values; output rows are ordered
  by `(window_start, window_end, group key)` deterministically across runs
  and recoveries; incremental results equal a finite-batch group-by oracle
  under randomized batch partitioning (property test).
- [ ] **AC-S7** Alignment tests: under every two-input barrier arrival
  permutation, the first-blocked ingress pauses while the other flows, and
  pre-barrier data is inside the snapshot while post-barrier data is not;
  future or regressed epochs fail closed; a timeout converges the job with
  the checkpoint-timeout classification and no manifest; cancellation
  mid-checkpoint converges to cancelled with orphaned segments collected;
  concurrent triggers produce strictly increasing epochs with one in flight.
- [ ] **AC-S8** Lifecycle tests: every state transition of S8.2 is reachable
  and terminal states are immutable; simultaneously observed errors are
  returned ordered by stable task ID; a panicking task yields a typed
  internal error with task identity; deadline and explicit cancel converge to
  one terminal state; `wait()` is idempotent; dropping the wait future does
  not cancel, dropping the job handle does.
- [ ] **AC-S9** Capability tests: requesting exactly-once with an ordinary
  sink, a lossy source, a volatile UDF, or a missing durable aligned
  checkpoint fails compilation before any source opens, naming the failing
  path; per-sink guarantees are reported exactly as the S9.1 matrix derives.
- [ ] **AC-S10** Channel tests: a full row budget blocks; a full byte budget
  blocks with few messages; an oversize single message fails before enqueue;
  a cancelled send releases its reservation exactly once; closing the
  receiver wakes blocked senders; compilation or runner construction rejects
  a source whose configured maximum batch exceeds its downstream edge
  capacity before any source opens; a slow sink observably pauses source
  polling.
- [ ] **AC-I** Invariant tests: serialized configurations, projects, and
  manifests contain no executable objects (structure property test); secret
  values are absent from every persisted and observed surface (redaction
  tests, including a forced failing source or sink open with a canary secret
  value, asserting the canary never appears in the propagated error);
  repeated serializations of manifests, fingerprints, metrics, and
  status snapshots are byte-identical (determinism); `tests/fixtures/v1/` is
  untouched (empty diff).

## 9. Open questions and explicit deferrals

Nothing below is a semantic TBD; each item is a named, owned decision that
does not block M1 from starting once M0.2 closes.

Owned by the M0.2 API note / critique (same slug):

- Rust and Python type names, ownership, async signatures, and error variant
  names for `StreamSource`, `StreamSink`, `StreamOperator`, `StreamCollector`,
  `StreamingRunner`, `StreamingJob`, and the checkpoint/state types; including
  `StreamCollector` object safety and the Python async/blocking split with
  running-event-loop rejection.
- **Barrier forwarding timing** after operator snapshot: immediate forwarding
  versus waiting for the coordinator acknowledgement (plan M0.2 checkbox).
  Constraint recorded here: snapshot success precedes forwarding either way
  (S7.3); choosing coordinator-ack costs checkpoint latency of roughly
  O(graph depth x round trip), whose bound the API note MUST state.
- The `calc-flow-python` to `calc-flow-connectors` dependency edge, connector
  cargo-feature gating, CI `--all-features` impact, and the connectors-crate
  coverage-gate treatment (plan section 3: an M0-level decision assigned here
  because it is a surface/packaging choice, not semantics).
- Project v3 top-level structure and secret-reference shape; Studio `/api/v3`
  routes and the SSE event model.
- The exact stable byte encoding of window group keys (the ordering CONTRACT
  is frozen in S6.4) and the timezone annotation of window bound columns.
- The cursor ordered-envelope representation referenced by section 1 and S2.2
  (ordered byte key plus opaque payload; the per-connector key encoding
  reuses the order-preserving encoding fixed for group keys).
- The sink-side contracts referenced by S7.5 and I6: the recovery contract
  (determining each recorded epoch's commit outcome from the manifest's
  pre-commit metadata plus the external system alone, and aborting or
  removing staged artifacts of crash-abandoned epochs), the teardown-drop
  tolerance contract for sink method futures, and the barrier-path
  capture-cost rule for state snapshots.
- Checkpoint manifest JSON field layout, including whether an explicit
  recovery-status field complements the S7.5 derivation rule.

Owned by later milestones, recorded so they cannot be invented ad hoc:

- Null event-time rows at a window input, and null/NaN/overflow/decimal/avg
  aggregate edge semantics plus the supported type matrix: fixed by task M4.3
  (S6.1, S6.5).
- Group-key bounds and equality: a bounded group-key size rule, and whether
  grouping equality matches the encoding order (including -0.0/+0.0 and NaN
  handling): fixed by task M4.3 (S6.4).
- The per-connector split of connector options into data-semantics-affecting
  (feeding the source identity hash, NFR-5) versus credential/transport
  (feeding neither hash): owned by M6.1's capability surface.
- The manual fresh-lineage procedure for operators of a poisoned lineage
  (stop the job, remove or repoint the state root and checkpoint directory):
  documented by M7.3 (D9.6).
- PostgreSQL client library selection (M6.4) and ClickHouse native/HTTP
  client selection (M6.5), after M0 review per the plan.
- Metrics label taxonomy beyond the no-high-cardinality rule (M2.5/M3.3).
- Segment-internal compression, only with paired benchmark evidence (D6).

## Appendix A. Plan M0.1 checkbox coverage map

| Plan M0.1 checkbox                                                                   | Spec sections             |
| ------------------------------------------------------------------------------------ | ------------------------- |
| FIFO of data, watermark, barrier, idle, end on an edge                               | S1.1, S1.2, AC-S1         |
| Source sequence/cursor: strictly increasing, duplicates, regression, recovery        | S2, FR3-FR5, AC-S2        |
| Fan-out order; no global order between sibling branches                              | S3, FR6, AC-S3/S4         |
| Union selection: per-ingress FIFO, no business order across ready ingresses          | S4, FR7, AC-S3/S4         |
| Watermark monotonicity, idle, reactivate, end, final flush                           | S5, S1.4-6, FR8-10, AC-S5 |
| `EventTime` precision, checked Arrow conversions, uniform floor truncation           | D1, FR11, AC-D1           |
| Late boundary per window (`window_end <= WM_in`); forbidden per-row criterion        | D2, FR12, FR13, AC-D2     |
| Source barrier responsiveness while waiting for external data                        | D3, FR14, AC-D3           |
| Segment-then-manifest publish order; manifest as sole truth                          | D4, FR17, AC-D4           |
| Final-only tumbling/hopping trigger and output order                                 | D8, S6, FR19, AC-S6       |
| Barrier alignment, timeout, cancel, source cursor boundary                           | S7, D7, FR15-16, AC-S7    |
| Job terminal states and deterministic multi-error selection                          | S8, FR21, AC-S8           |
| Full at-least-once/exactly-once capability matrix                                    | S9, FR22, AC-S9           |
| All plan non-goals written into the specification                                    | Non-Goals NG1-NG13        |

Additional frozen decisions the plan assigns to M0 beyond the checkbox list:
segment format (D6, from task M4.1), checkpoint timeout rule (D7, from task
M5.2's "fail per the M0 rule"), hopping size/slide divisibility (D8.4, from
task M4.3's "unless M0 approves a general rule"), and epoch start/increment
rules (D9, referenced by task M1.2's RED list).
