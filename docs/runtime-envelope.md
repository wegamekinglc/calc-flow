# Stream message envelope

[Documentation](README.md) / 4.2 Stream runtime contract

The v3 Rust core moves stream traffic on one typed message: `StreamMessage`.
Each stream edge carries a single ordered sequence of data and control
messages from one producer to one consumer. This document is the normative
contract for that envelope: the message type, its typed `EventTime` and
`Epoch` values, the job and operator contexts, the emission boundary stream
operators see, the compiled stream plan, and the delivery invariants the
runtime guarantees today. The source-driven runtime is public through the
crate-root `StreamingRunner` and owning `StreamingJob`; control construction,
checkpoint coordination, supervision, and reaper ownership remain internal
implementation details.

The public facade lives in `crates/calc-flow/src/continuous.rs`. Its runtime
implementation lives in `crates/calc-flow/src/runtime/streaming/` (the message,
bounded channel, job and task contexts, whole-job preflight, source, operator,
and sink tasks, the job-scoped progress driver, checkpoint coordinator,
supervisor, internal runner/reaper, metrics, and soak),
`crates/calc-flow/src/state/` (the v3 manifest, segments, state backends, and
production manifest transaction), `crates/calc-flow/src/time/` (event time and
epoch),
`crates/calc-flow/src/operator/stream.rs` (the operator traits and public
in-memory collector), `crates/calc-flow/src/operator/window.rs` (window
declarations and execution), and `crates/calc-flow/src/pipeline/stream.rs`
(the compiled stream plan). For a component overview, read [architecture](design.md). For application
usage, follow the [streaming guide](streaming-guide.md).

The public surface splits operators and plans by lifecycle. `BatchOperator`
and `BatchExecutionPlan` run finite one-shot graphs; `StreamOperator` and
`StreamExecutionPlan` compile continuously running graphs. On both sides,
`Batch` remains the public data envelope: raw tables and arrays never cross
a graph, plan, or runner boundary.

On this page:

- [`StreamMessage`: one message per edge](#streammessage-one-message-per-edge)
- [`EventTime`](#eventtime)
- [`Epoch`](#epoch)
- [Job and operator context](#job-and-operator-context)
- [The operator emission boundary](#the-operator-emission-boundary)
- [Stream plan compilation](#stream-plan-compilation)
- [Job preflight](#job-preflight)
- [Source tasks and accepted positions](#source-tasks-and-accepted-positions)
- [Operator tasks and barrier alignment](#operator-tasks-and-barrier-alignment)
- [Sink tasks](#sink-tasks)
- [Checkpoint coordination](#checkpoint-coordination)
- [Manifest publication and recovery](#manifest-publication-and-recovery)
- [Terminal recovery](#terminal-recovery)
- [Cancellation and ownership](#cancellation-and-ownership)
- [Status and metrics](#status-and-metrics)
- [Execution paths](#execution-paths)
- [Progress and replay](#progress-and-replay)
- [Window state and storage](#window-state-and-storage)
- [Public and private surfaces](#public-and-private-surfaces)

## `StreamMessage`: one message per edge

`StreamMessage` is the single ordered message carried by one stream edge. Its representation is private; the variants are:

| Variant      | Payload     | Contract meaning                                                     |
|--------------|-------------|----------------------------------------------------------------------|
| `Data`       | `Batch`     | One immutable data batch; the only variant public callers construct. |
| `Watermark`  | `EventTime` | An event-time progress estimate.                                     |
| `Barrier`    | `Epoch`     | A checkpoint barrier carrying its epoch.                             |
| `Idle`       | —           | Marks its ingress idle; excluded from watermark progress only.       |
| `EndOfInput` | —           | Terminates its ingress permanently.                                  |

The data variant wraps one immutable `Batch`. Fan-out clones share the
payload: cloning a message clones the handle, not the Arrow buffers.

Control messages are created only through crate-private constructors.
`StreamMessage::data` is the only public constructor, so operators, sources,
and other public consumers cannot forge, suppress, or reorder control; the type system, not a runtime check, enforces this. Validation of
control values — watermark monotonicity and epoch consistency — belongs to
the runtime paths that construct and enqueue them and runs before enqueue,
before any downstream side effect. The job-scoped progress driver
validates source-provided watermark monotonicity and owns generated
watermark/timer ordering. A checkpoint-enabled job validates the
single-flight epoch at its source cut, operator alignment, sink pre-commit,
manifest publication, and sink finalization boundaries. An internal job path
started without checkpoint wiring still fails closed if it receives a barrier.

Inspection goes through the message kind and typed accessors:

| Accessor            | Returns             | Meaning                                                |
|---------------------|---------------------|--------------------------------------------------------|
| `kind()`            | `StreamMessageKind` | The message kind, for inspection and routing.          |
| `as_data()`         | `Option<&Batch>`    | The data payload, when this is a data message.         |
| `as_watermark()`    | `Option<EventTime>` | The watermark value, when this is a watermark message. |
| `as_barrier()`      | `Option<Epoch>`     | The barrier epoch, when this is a barrier message.     |
| `is_idle()`         | `bool`              | Whether this message marks its ingress idle.           |
| `is_end_of_input()` | `bool`              | Whether this message terminates its ingress.           |

The `Debug` implementation shows kinds and typed business values only. Row
payloads, batch metadata, and attributes — which may carry secrets — never
appear in diagnostics.

## `EventTime`

`EventTime` is a newtype over a signed 64-bit integer counting microseconds
since the Unix epoch (1970-01-01T00:00:00Z), always UTC. Ordering is
total across pre- and post-epoch values. Public APIs never expose a bare
`i64` timestamp; they expose `EventTime` or its exact serialized
microsecond value. Serialization stores the exact microsecond count, so
durable representations round-trip losslessly.

Arrow timestamp columns import through `EventTime::import_timestamp`, which
selects the conversion by the column's time unit; `EventTime::export_timestamp`
produces one Arrow timestamp unit back:

| Arrow unit      | Import to `EventTime`                    | Export from `EventTime`    |
|-----------------|------------------------------------------|----------------------------|
| `timestamp(s)`  | multiply by 1,000,000, checked           | divide by 1,000,000, floor |
| `timestamp(ms)` | multiply by 1,000, checked               | divide by 1,000, floor     |
| `timestamp(us)` | exact identity                           | exact identity             |
| `timestamp(ns)` | divide by 1,000, floor (cannot overflow) | multiply by 1,000, checked |

Checked conversions fail with `CalcFlowError::InvalidArgument`; import
errors name the offending column, and silent wrap is forbidden.

Every conversion that produces a coarser representation from a finer value
floors toward negative infinity — one uniform direction with no per-call-site
choice. For example, `-1,500 ns` imports to `-2 µs`, `1,999 ns`
imports to `1 µs`, and exporting `-1 µs` to seconds yields `-1 s`, not
`0 s`. Flooring never moves a row into a later window and keeps watermark
progress estimates conservative.

An Arrow timestamp column declared as an event-time column must be
timezone-naive (interpreted as UTC) or carry the explicit timezone `"UTC"`;
any other timezone is rejected with the column path in the error.

## `Epoch`

`Epoch` is a newtype over `u64` identifying one checkpoint within a job
lineage. Value `0` is reserved as the "no checkpoint" sentinel and is
unconstructable: `Epoch::new` returns `None` for it, and it is never
injected. `Epoch::INITIAL` is `1`, the first checkpoint of a fresh
lineage, and `Epoch::next` increments by exactly one with a checked overflow. Epochs are strictly increasing within a job lineage. Serialization
stores the exact value. A fresh private checkpoint lineage begins at
`Epoch::INITIAL`; recovery selects the latest valid manifest and allocates its
epoch's checked successor. Exhausting `u64` fails rather than wrapping.

## Job and operator context

`StreamJobContext` is the immutable, job-scoped context shared by every task
of one streaming job. It carries the job ID, the plan fingerprint, run
settings as a `JsonMap`, an optional deadline, and the cancellation token.
The constructor requires a UTC deadline. `check_cancelled` fails with
`CalcFlowError::Cancelled` when cancellation was requested or the deadline
has passed.

The internal runtime derives immutable source, node, and sink task scopes
from this context. A scope carries its task kind and validated identity while
sharing the same job context; creating a blank, whitespace-only identity or
one containing NUL fails before task registration.

`StreamOperatorContext` is the execution context an operator task hands to
one stream operator. It borrows the job context and exposes:

- `job()` — the owning job's immutable context;
- `operator_id()` — the operator's node identity;
- `input_watermark()` — the current input watermark `WM_in`, or `None`
  while undefined;
- `check_cancelled()` — the job liveness check;
- `record_late_rows(dropped, max_lateness)` — cumulative late-data counters:
  dropped row-window assignments, the count of batches that contained at
  least one dropped assignment, and the maximum observed lateness. Row
  payloads are never accepted, and only window operators call this. Counter
  overflow fails transactionally instead of wrapping. Runtime-created
  contexts share one task-owned recorder, so late and null-event-time totals
  persist across handler calls and appear in crate-private status/metrics.

The crate-private task constructor also supplies the minimum effective output
edge budget. A window close uses that budget to split deterministic output
rows before enqueue; the public constructor
uses the default budget plus an isolated recorder.

## The operator emission boundary

`StreamOperator` is the continuously running operator trait: it receives one
named-ingress batch per call and emits only through a `StreamCollector`. The
port, schema, and UDF metadata accessors live on the shared `OperatorMetadata`
supertrait so the batch and stream compilers validate them once. The
handlers are:

- `process_data(ingress, batch, context, output)` — processes one batch
  from the named ingress. A failed handler never forwards a partial control
  event.
- `on_watermark(watermark, context, output)` — reacts to an input-watermark
  advance; a window operator emits newly closed windows before the runtime
  forwards the watermark.
- `on_end(context, output)` — flushes once after every ingress has ended.

Handlers never see barriers, and watermarks arrive as typed `EventTime`
values.

The state lifecycle is synchronous and executor-safe. `checkpoint(epoch)`
captures dirty state as an `OperatorStateSnapshot` in O(dirty-key
metadata) — never a bulk encode on the executor thread; durable staging is
runtime-owned. `restore(snapshot)` applies a captured snapshot, and
`reset()` returns the operator to its freshly constructed state. The
defaults are stateless: capture returns an empty snapshot, restore rejects a
non-empty one, and reset succeeds.

`OperatorStateSnapshot` carries small bounded JSON `inline_metadata` plus
named `segments` of allocation-shared `StateSegment` values (immutable bytes
with a SHA-256 computed once at construction, so carried segments move
between epochs without copying or re-hashing). Keyed row state never appears
inline, and no segment may carry secrets; the runtime assigns
segment paths, lengths, and checksums during staging, reusing the
already-committed handle for segment content that is unchanged since the
current session committed it.

The built-in window operator prepares immutable Arrow IPC deltas through a
blocking worker while processing data and control events. Its synchronous
checkpoint only assigns deterministic segment IDs, moves prepared buffers,
and captures bounded metadata. Restore validates every metadata field, Arrow
schema, group-key encoding, row order, and accumulator shape on a worker
before replacing live state. More than 32 retained deltas prepares one
replacement base before the next capture; old immutable segments remain live
until manifest reachability collection can remove them.

`StreamCollector` is the only way a stream operator emits. `emit(port,
batch)` validates the port name, the `BatchKind`, and the optional exact
Arrow schema against the compiled output ports, then enqueues the batch; a
validation failure returns `CalcFlowError::Compile` before the batch reaches
an edge, so an invalid batch never produces a downstream side effect. Control messages can never be emitted through this trait:
watermark, barrier, idle, and end-of-input forwarding is runtime-owned.

`EdgeCollector` is a public in-memory `StreamCollector` helper for tests,
benchmarks, and callers that invoke an operator directly. It validates against
the supplied output ports and stores data messages in per-port FIFO outboxes;
`drain(port)` empties one outbox, while an unknown port drains to empty. It is
not the collector used by the private continuous operator-task runtime.

The crate-private operator task instead constructs a
`ChannelStreamCollector` for each handler call. It performs the same
port/kind/schema validation, validates every destination, and then sends the
message directly through the output's bounded `EdgeSender` fan-out. Validation
therefore fails before any edge observes the batch, and every edge applies its
own capacity and accounting.

Runtime edges use `edge_channel`, a bounded FIFO with atomic envelope, row, and
estimated-byte accounting. For `EdgeBudget { max_rows: R, max_bytes: B }`, the
queue admits at most `R` envelopes, independently at most `R` charged rows,
and at most `B` charged bytes. Every data or control envelope consumes one
slot, including a zero-row/zero-byte data batch. Direct callers choose
`R >= max(required_row_limit, required_simultaneous_messages)`. Zero-row data and control traffic are bounded by the envelope limit.

Reservation is atomic with enqueue and is released exactly once on receive or
queue teardown. A blocked send owns no reservation. An oversized data message
fails before enqueue, a closed receiver wakes blocked senders, and each fan-out
edge is charged independently even though immutable payload buffers are
shared. The default `Block` path therefore propagates a slow consumer through
sink, operator, source task, prefetch slot, and source pump. FIFO order, close wakeup, and single-producer ownership apply to every edge.

## Stream plan compilation

`PipelineBuilder::compile_stream` validates and consumes a graph into an
immutable `StreamExecutionPlan`. Every stream-rule violation is reported at
compile time, before any source opens:

- multi-input (multi-alias) SQL nodes are rejected;
- nodes offering only a batch operator are rejected;
- delivery requirements naming an unknown graph output are rejected;
- volatile DataFusion scalar UDFs are rejected when any output requests
  exactly-once delivery;
- read-only expression and SQL queries that call a volatile built-in
  function or a wall-clock built-in function (`now`, `current_date`, or
  `current_time`) are rejected. Every call is resolved against the
  built-in default function registry and matched by its canonical name
  after parsing, so the `current_timestamp` and `today` aliases and the
  no-parenthesis keyword spellings are covered too, with the same error
  shape as the volatile rejection. Wall-clock reads would otherwise make
  a checkpoint replay of the same input produce different output.

Stream graphs compose unary expression nodes, single-input SQL nodes,
explicit stream providers, `RollingOperator`, `CrossSectionOperator`,
`UnionOperator`, `WindowAggregateOperator`, and the two-input bounded
event-time `StreamJoinOperator`. A union's two or more input ports share one
kind, one required flag, and one exact schema, or are all schema-less.

The compiled plan records the deterministic topology, stable edge IDs, the
source and sink binding slots (external input and output names in
deterministic order), the semantic fingerprint, and the per-output delivery
requirements. For stream plans, the semantic fingerprint also freezes every
operator's checkpoint capability and state-layout version. It never executes
directly. The public source-driven `StreamingRunner` consumes it into owned
runtime nodes, internal edges, and
synthesized bounded source and sink boundary edges. Pure array graphs need no
table engine:
`requires_datafusion` reports whether any node needs a DataFusion session.

`StreamRequirements` records the requested `DeliveryGuarantee` per graph
output — `BestEffort`, `AtLeastOnce`, or `ExactlyOnce`; outputs absent from the
map default to `AtLeastOnce`. The guarantee is scoped to one output, not a global property
of the plan. Compilation applies the deterministic-UDF rule above. Before a
checkpointed job starts, whole-job preflight proves every reachable
source, operator, bounded edge, and bound sink for each exactly-once output.
It reports the output and first incompatible stable component before connector
lifecycle work. An internal job path without checkpoint wiring rejects every
exactly-once request. The frozen requested/effective proof is kept for every
output. A best-effort request is never upgraded; an at-least-once request is
downgraded explicitly when a reachable source is lossy or unreplayable; and no
request is silently upgraded.

Two hashes describe the plan:

- the semantic fingerprint covers execution mode, graph structure, operator
  configurations, and the UDF catalog; it decides checkpoint compatibility;
- the runtime-config hash covers `StreamRuntimeConfig` — checkpoint interval
  and timeout, the per-edge envelope/row/byte budget, and retained epochs —
  and feeds observability and diagnostics only, so retuning it never invalidates
  checkpoints. Durations must be exact multiples of one microsecond, and
  both budget fields must be positive. The defaults are a 60-second checkpoint
  interval, a 600-second checkpoint timeout, 10,000 envelopes, 10,000 rows and
  64 MiB per edge, and two retained epochs.

## Job preflight

The crate-root `StreamingRunner` runs a bounded
source-to-operator-to-sink job with managed epoch checkpoints and returns an
owning `StreamingJob`. Its public connector and lifecycle types project the
internal task/coordinator machinery without exposing control-message
constructors or connector payloads.

One pure whole-job preflight consumes the plan and validates the context
fingerprint, runtime topology, every source and sink route, duplicate or
missing bindings, source capabilities and first-hop budgets, and sink delivery
mechanisms. Each external-input binding name is the source's stable runtime and
durable identity. Its declared Arrow schema, native-watermark behavior, replay
positioning, delivery losslessness, and batch bounds are sampled once and
frozen before lifecycle work; that descriptor participates in the durable
identity check on recovery. For every requested exactly-once output, preflight
walks the reachable subgraph and requires a lossless, exactly replayable source,
deterministic operator checkpoint/restore, bounded lossless edges, and only
transactional or unbounded epoch-idempotent sinks. A lossy source, ordinary
sink, or bounded idempotency horizon fails the named output before a connector
opens, an operator resets or restores, a state lineage is opened, or a task is
registered. Independently of an output request, a checkpoint-enabled job
admits only operators explicitly classified as stateless or versioned
checkpointed-stateful; an unproven third-party stream operator fails before
task registration. The compiled graph node ID is the stable operator ID and
must be unique and portable.

Status exposes both requested and effective delivery for each output. Lossy or
unreplayable routes project `BestEffort`, even when the project used the
default at-least-once request, so observability never overstates the runtime
contract.

Public diagnostic projection also treats source identity as untrusted. If a
validation path contains a non-portable source ID, projection fails closed to
the generic validation error `source ID is not a portable identifier`, with
`Source` as the component kind and no `component_id`. The original ID,
underlying message, and source chain never enter `Display`, `Debug`, or serde
output, so an invalid identifier cannot smuggle a secret into diagnostics.

## Source tasks and accepted positions

A source binding owns two supervised tasks:

- the pump is the only caller of `StreamSource::open` and `next`, keeps at
  most one poll in flight, and feeds a one-item prefetch slot;
- the source task drains that slot, remains responsive to cancellation, and
  submits each event through the job-scoped progress owner;
- teardown drops a blocked `open` or `next` future, never polls that source
  again, and calls `close` before the pump finishes.

The source boundary executes one ordered lifecycle: restore the
durable cursor/end state, open and read, participate in each checkpoint
barrier, observe end once, then close during teardown. A source restored as
ended skips open, read, and later barriers, so restart cannot repeat its end
effect.

For each data event, the binding ID replaces connector-supplied source
identity, the runtime assigns a checked per-binding sequence, and existing
JSON attributes are preserved immutably. Cursor order keys must strictly
advance, including past a recovery cursor, or the batch is rejected before
enqueue. A cursor tagged for another stable binding is likewise rejected
before its batch reaches an edge. Source-provided watermarks must not regress. `None` becomes one
ordered `EndOfInput`; source-provided `Idle` and watermark events share the
same per-edge FIFO as data. Progress keeps the latest observed cursor separate
from the durable recovery cursor.

For a periodic checkpoint, every live replayable source pauses new admission
and settles its accepted slot. The `LiveProgressCoordinator` then holds its
serial drive boundary, drains ready pre-cut inputs and timers, requires zero
pending receipts, and fans out `Barrier(E)` over the source routes it owns.
Only a successful fan-out promotes the observed cursor to durable and resumes
source polling. Idle sources participate. Ended and previously restored-ended
sources contribute their final cut out of band and never reopen or receive a
post-end barrier.

`TaskSupervisor` assigns stable IDs in registration order and start-gates
tasks until registration is complete. The first observed task failure starts
cancellation convergence, every sibling is joined, simultaneous primary
failures are reported in stable task-ID order, and panics become
`CalcFlowError::TaskPanicked` with the stable task identity. Source failures
signal cancellation before potentially blocking teardown work, so siblings
can converge promptly.

## Operator tasks and barrier alignment

One operator task owns each compiled stream operator. It selects ready
ingresses without weakening per-ingress FIFO, validates data emissions before
the first send, fans out over real bounded edges, and owns one lazy
operator-scoped DataFusion runtime when table work requires it. Aggregate
watermarks call `on_watermark` before runtime forwarding; aggregate idle and
all-ingress end are runtime-owned. Multi-ingress watermark and idle handling
uses the progress driver's minimum and idle-epoch semantics. `on_end` runs
exactly once after every ingress observes explicit end-of-input. A closed
channel without an earlier explicit end is a failure, not synthetic EOF.

In a checkpoint-enabled job, an operator removes only barrier-arrived
ingresses from receive selection; their post-barrier traffic remains in the
bounded edge while other live ingresses continue. Ended ingresses count as an
out-of-band cut. When every required ingress reaches epoch E, the task captures
and stages operator state, acknowledges the coordinator, immediately forwards
the barrier, and reopens its live ingresses. An identical repeated barrier is
idempotent. A foreign, conflicting, future, or regressed epoch fails closed,
and snapshot/stage failure forwards no barrier.
Stateful snapshots carry the operator's nonzero state-layout version in their
inline metadata. Recovery removes that envelope only after checking the stored
version against the compiled capability; missing or mismatched versions fail
closed before operator tasks are spawned. Stateless operators must produce and
restore empty state.

## Sink tasks

One sink task owns each graph output and writes each batch to its configured
sinks in stable order. The third sink does not see a batch when the second sink
fails. Natural completion and graceful shutdown drain the accepted prefix;
explicit cancellation makes no drain promise. A checkpointed at-least-once
output may use ordinary sinks: the manifest records `Ordinary` with no
pre-commit metadata, and restart does not call a transactional recovery hook.
This preserves the allowed replay-and-duplicate boundary. Transactional and
unbounded epoch-idempotent sinks instead begin the epoch, pre-commit bounded
canonical metadata after the barrier, and commit only after the manifest is
durable.

## Checkpoint coordination

The checkpoint coordinator owns one bounded FIFO and at most one active epoch.
Its checked phases are `Requested`, `SourcesCut`, `OperatorsSnapshotted`,
`SinksPrecommitted`, `ManifestDurable`, `SinksCommitted`, and `Completed`.
Participant sets come from preflight; byte-identical duplicate acknowledgements
are idempotent, while missing, foreign, out-of-epoch, or conflicting evidence
fails the job. Periodic, manual, and terminal requests share the FIFO and the
same checked epoch allocator; accepted manual requests are neither coalesced
nor removed when their waiter is dropped. Manual completion is reported only
after the manifest is durable and every sink commit is acknowledged. Timeout
never skips an epoch. Cancellation before manifest durability fails the active
and queued requests, while cancellation after durability finishes the sink
commit attempt or leaves the job in `RecoveryRequired`.

## Manifest publication and recovery

The production `ManifestTransaction` reuses the public v3
`CheckpointManifest` as the sole durable recovery truth. It serializes segment
stage/validation/publication, manifest-last atomic publication, strict bounded
candidate selection, referenced-segment reload, retention, and orphan
collection. Semantic identity and exact participant sets fail closed;
`runtime_config_hash` mismatch is a non-blocking private diagnostic. Recovery
selects the highest complete valid epoch and validates every segment length and
checksum. Corrupt higher candidates, links, unexpected entry types, and paths
outside the managed roots fail closed. Regular abandoned `.tmp*` manifest
files are removed during a serialized scan; links and directories are never
followed or removed as temporary files.

Every transactional sink pre-commits before manifest publication. A failure
before manifest installation aborts prepared transactions. After rename, an
installed manifest remains recovery intent even when parent-directory sync or
the publication acknowledgement is ambiguous: prepared sinks are preserved,
not aborted. A durable manifest authorizes idempotent external commit, and a
partial multi-sink commit completes forward during recovery. No second durable
completion record competes with the manifest. Retention failure after commit
fails the live job but does not invalidate the completed epoch.

Checkpointed startup remains gated: pure preflight completes before the runtime
opens the lineage, strictly selects and validates the manifest, reloads
operator state, creates fresh durable progress, resets/restores operators,
opens non-ended sources with their exact cursor, opens sinks, and completes
transactional recovery before the data gate is released. Ended sources are
removed before connector ownership.

The durable projection contains source cursor/sequence/end/watermark policy and
operator ingress progress/state handles; it never serializes execution trace history,
pending receipts, or process-local timer coordinates. Restored timers re-arm
from a fresh origin with their complete configured delay.

## Terminal recovery

After the data-plane end cut, the coordinator allocates a terminal epoch out
of band. Operators capture post-`on_end` state and sinks pre-commit final output
without any post-end barrier. A terminal manifest is recognized only when all
source and operator ingress entries are ended. Recovery from it opens only
sinks, completes any commit, closes resources, and returns natural completion;
it does not reopen sources or repeat `on_end`/final-window emission.

## Cancellation and ownership

The public `StreamingRunner` and `StreamingJob` own three-stage launch, status,
terminal arbitration, cancellation, graceful drain, and joining over an
internal runner-scoped reaper. Operator entry completes before connector open;
all connector opens complete before the data gate is released. A dropped start
observer cancels the provisional launch. A dropped job transfers convergence
ownership to the reaper, and a later start or runner shutdown joins it. Task
failure wins over explicit cancel, which wins over deadline expiry; concurrent
and repeated observers receive one immutable outcome.

Private source, operator, and sink tasks, the task supervisor, and runner
connector-open/close paths create operational
`CalcFlowError::TaskPanicked` values with a stable task ID. Together these
boundaries cover connector `open`, `next`, `write`, and `close` panics plus
operator and uncaught task panics. Captured panic text is valid UTF-8 and at
most 1,024 bytes; non-string payloads become `non-string panic payload`.

Source, operator, sink, progress, checkpoint, filesystem, and connector work
remain owner-settled under cancellation: the owner waits for in-flight state
and manifest operations before reporting terminal completion. A failed or
cancelled launch closes every resource whose open began in stable order. Each
close has its own private five-second bound, and a close error, panic, or
timeout becomes a typed bounded secondary diagnostic without replacing the
primary failure. Cleanup continues through later resources and converges task,
queue, state-lease, transaction, provisional-launch, and reaper ownership.

## Status and metrics

Public deterministic status and metrics projections cover task/terminal state,
source/operator/sink progress, per-edge accounting, and checkpoint request,
completion, failure, phase/alignment/total latency, state/manifest bytes,
restore latency, sink retries, orphan cleanup, and terminal outcome. The
checkpoint status payload contains only the current epoch and phase,
participant counts, elapsed time, last completed epoch, terminal flag, failure
category, and runtime-config mismatch flag. It has no cursor/pre-commit
payload, state bytes, filesystem path, row, attribute, secret, or arbitrary
metric label. Epoch values remain status fields rather than metric labels.

Operator `processing_duration` includes successful Data and watermark handlers,
including collector waits inside those handlers. The additional
`watermark_processing_duration` is the watermark-handler subset, not an
independent duration to add to the total. Python status exposes these as
`processing_duration_micros` and `watermark_processing_duration_micros`.
Failed handlers and end-of-input callbacks are not included. These elapsed
durations are not CPU-time measurements and are not additive across concurrent
operators. In particular, rolling finalization happens in the watermark
handler, not necessarily in the Data handler.

## Execution paths

For canonical ordered input and supported bounded typed row windows, rolling
buffers immutable Arrow batches until finality and computes directly from
their columns. It stages touched-entity kernel updates and newly retained
history rows, then commits them after successful output emission. Unchanged
retained rows are not cloned. Overlapping, unordered, or late envelopes and
unsupported window/type shapes use the general path. Checkpointing
materializes pending columnar buffers into the durable layout;
watermark, lateness, duplicate, null, output-budget, and recovery semantics
are enforced on both paths.

Column-only stream projections reuse Arrow arrays without SQL planning.
Arithmetic, filters, unsupported projections, and their errors retain the SQL
path. The Python connector bridge dispatches ready callbacks through one
job-scoped wakeup, yielding after at most 64 requests. Native coroutines may
start eagerly on the captured Python loop when no custom task factory is
installed; arbitrary awaitables and custom factories keep their scheduling
policy. Completed futures return without another completion-callback turn.
Per-request context copies, cancellation ownership, source polling order, and
bounded runtime backpressure remain in force; the bridge does not prefetch
source events.

## Progress and replay

The runtime prepares each source policy during whole-job preflight, then
routes raw source data/control through one job-scoped progress driver. That
driver alone owns the logical clock, binding/local/global ordering, finite
inbox fences, timer heap, idle epochs, aggregate progress, completion receipts,
and the lossless admission/drain/terminal/settlement execution trace.
Multi-ingress progress uses the minimum watermark of known active inputs; idle
and ended inputs are excluded. Data and legal watermarks reactivate before
processing, and all-ended emits one plain `EndOfInput` without a sentinel
watermark.

The crate-private transient snapshot captures the exact prepared/config,
upstream cursor/control, trace, gate/fence, allocator, aggregate, and timer
coordinate only at a receipt-quiescent boundary. Restore requires paused
upstreams at exact captured positions and field-for-field equality before any
state or runtime side effect is installed. This remains an in-process replay
surface, not the durable format. The checkpoint runtime separately projects
only bounded semantic source and operator progress into manifest entries and
reconstructs fresh process-local trace, receipt, and timer coordinates during
recovery.

The progress driver forwards old/equal/new event-time rows unchanged. It
neither classifies nor drops late rows and exposes no late-row runtime metric.

## Window state and storage

The public Rust surface includes a data-only `WindowSpec` for fixed UTC
tumbling and hopping windows and the stream-only `WindowAggregateOperator`.
The compiler validates geometry, overlap, exact event-time/group types,
aggregate combinations, output names, state layout, and deterministic
configuration fingerprints before source open. Supported aggregates are
`count`, `sum`, `min`, `max`, and `avg` over the supported type
matrix; decimal, session,
early-trigger, allowed-lateness, update, and retract forms remain unavailable.

Execution owns incremental accumulators in deterministic
`(window_start, window_end, G1 group key)` order. Null event times are dropped
and counted separately. A row-window assignment is late only when
`window_end <= WM_in`; a hopping row can therefore update open assignments
while dropping closed ones. Watermark and end handlers emit one final value
per non-empty window before runtime-owned control forwarding, chunk outputs to
the effective edge budget, and preserve checked operator-owned sequences
across snapshot/restore.

The public state surface consists of immutable `StateHandle` values,
lineage-exclusive `StateBackend` sessions, `LocalStateBackend`, and the strict
canonical `CheckpointManifest` v3 data model. The local backend stages,
syncs, re-reads, checksum-validates, and atomically publishes segments before
a manifest can reference them. Manifest selection is the sole recovery truth;
cleanup is reachability-based and fails closed on links or unexpected files.
The continuous runtime uses this backend and manifest as its production
checkpoint transaction and durable recovery truth.

## Public and private surfaces

Public: `StreamingRunner`, `StreamingJob`, `ManagedCheckpointRuntime`, source
and sink connector/binding types, status/outcome projections, the
`StreamMessage` handle and `StreamMessageKind`, the typed
`EventTime` and `Epoch` values, `StreamJobContext` and
`StreamOperatorContext`, the `StreamOperator` and `StreamCollector` traits,
`OperatorStateSnapshot`, `EdgeCollector`, and the compiled plan types
(`StreamExecutionPlan`, `StreamRequirements`, `DeliveryGuarantee`,
`StreamRuntimeConfig`, `EdgeBudget`). The bounded-edge surface is also public:
`edge_channel`, `EdgeSender`, `EdgeReceiver`, `EnvelopeCost`, and
`ChannelMetrics`. `CalcFlowError::TaskPanicked` is public because the error
enum is public and non-exhaustive. Operational instances come from the private
task and supervisor paths; public job controls are checkpoint, shutdown,
cancel, wait, and payload-safe status observation.

Crate-private: the message representation and the four control constructors,
the compiled-operator representation, the late-row recorder, internal source
cursor/binding representations, the channel-backed operator collector,
source/operator/sink tasks and progress snapshots, scoped task contexts,
whole-job preflight, the task supervisor, checkpoint coordinator/transaction
internals, raw status and metrics, and the runner-scoped reaper.

Runtime-owned control-message and transaction internals do not appear in
project-v3 documents or Studio responses. Python binds the public continuous
facade, while Studio owns persistent `/api/v3/jobs` lifecycle, checkpoint,
shutdown, cancellation, and SSE observation routes. The public v3 manifest
model and state backend remain the runtime's durable recovery truth; Studio
does not expose raw cursor, connector state, secret, or filesystem payloads.


See [verification](verification.md#streaming-stress-and-soak-checks) for the
stress and checkpoint-restart harnesses. These checks provide implementation
evidence; the contract above does not claim results for any particular run.

Next: [symbolic compiler design](symbolic-design.md).
