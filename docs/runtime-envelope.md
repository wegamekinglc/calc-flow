# Stream message envelope

The v3 Rust core moves stream traffic on one typed message: `StreamMessage`.
Each stream edge carries a single ordered sequence of data and control
messages from one producer to one consumer. This document is the normative
contract for that envelope: the message type, its typed `EventTime` and
`Epoch` values, the job and operator contexts, the emission boundary stream
operators see, the compiled stream plan, and the delivery invariants the
runtime guarantees today. The source-driven runtime remains an internal
vertical slice: it is runnable inside the crate, but is not yet a public
continuous runner.

The implementation lives in `crates/calc-flow/src/runtime/streaming/` (the
message, bounded channel, job and task contexts, whole-job preflight, source,
operator, and sink tasks, the job-scoped progress driver, transient progress
snapshot/status, supervisor, private runner/job/reaper, metrics, and soak),
`crates/calc-flow/src/time/` (event time and epoch),
`crates/calc-flow/src/operator/stream.rs` (the operator traits and public
in-memory collector), and `crates/calc-flow/src/pipeline/stream.rs` (the
compiled stream plan). The frozen semantics behind the contract are recorded
in the [continuous streaming runtime
specification](../.codex/artifacts/specs/continuous-streaming-runtime.md),
cited below as S and D items.

The public surface splits operators and plans by lifecycle. `BatchOperator`
and `BatchExecutionPlan` run finite one-shot graphs; `StreamOperator` and
`StreamExecutionPlan` compile continuously running graphs. On both sides,
`Batch` remains the public data envelope: raw tables and arrays never cross
a graph, plan, or runner boundary.

## `StreamMessage`: one message per edge

`StreamMessage` is the single ordered message carried by one stream edge
(S1.1). Its representation is private; the variants are:

| Variant       | Payload     | Contract meaning                                                      |
| ------------- | ----------- | --------------------------------------------------------------------- |
| `Data`        | `Batch`     | One immutable data batch; the only variant public callers construct.  |
| `Watermark`   | `EventTime` | An event-time progress estimate (S5).                                 |
| `Barrier`     | `Epoch`     | A checkpoint barrier carrying its epoch (S7).                         |
| `Idle`        | —           | Marks its ingress idle; excluded from watermark progress only (S1.4). |
| `EndOfInput`  | —           | Terminates its ingress permanently (S1.6).                            |

The data variant wraps one immutable `Batch`. Fan-out clones share the
payload: cloning a message clones the handle, not the Arrow buffers (S3).

Control messages are created only through crate-private constructors.
`StreamMessage::data` is the only public constructor, so operators, sources,
and other public consumers cannot forge, suppress, or reorder control
(S1.3); the type system, not a runtime check, enforces this. Validation of
control values — watermark monotonicity and epoch consistency — belongs to
the runtime paths that construct and enqueue them and runs before enqueue,
before any downstream side effect (S1.3, S5.4). The M3 job-scoped progress
driver validates source-provided watermark monotonicity and owns generated
watermark/timer ordering; barrier and epoch validation remain checkpoint-
protocol work listed under "Specified but not yet implemented".

Inspection goes through the message kind and typed accessors:

| Accessor            | Returns             | Meaning                                                  |
| ------------------- | ------------------- | -------------------------------------------------------- |
| `kind()`            | `StreamMessageKind` | The message kind, for inspection and routing.            |
| `as_data()`         | `Option<&Batch>`    | The data payload, when this is a data message.           |
| `as_watermark()`    | `Option<EventTime>` | The watermark value, when this is a watermark message.   |
| `as_barrier()`      | `Option<Epoch>`     | The barrier epoch, when this is a barrier message.       |
| `is_idle()`         | `bool`              | Whether this message marks its ingress idle.             |
| `is_end_of_input()` | `bool`              | Whether this message terminates its ingress.             |

The `Debug` implementation shows kinds and typed business values only. Row
payloads, batch metadata, and attributes — which may carry secrets — never
appear in diagnostics (invariant I4).

## `EventTime`

`EventTime` is a newtype over a signed 64-bit integer counting microseconds
since the Unix epoch (1970-01-01T00:00:00Z), always UTC (D1.1). Ordering is
total across pre- and post-epoch values. Public APIs never expose a bare
`i64` timestamp; they expose `EventTime` or its exact serialized
microsecond value. Serialization stores the exact microsecond count, so
durable representations round-trip losslessly (D1.2).

Arrow timestamp columns import through `EventTime::import_timestamp`, which
selects the conversion by the column's time unit; `EventTime::export_timestamp`
produces one Arrow timestamp unit back (D1.3, D1.4):

| Arrow unit      | Import to `EventTime`                    | Export from `EventTime`    |
| --------------- | ---------------------------------------- | -------------------------- |
| `timestamp(s)`  | multiply by 1,000,000, checked           | divide by 1,000,000, floor |
| `timestamp(ms)` | multiply by 1,000, checked               | divide by 1,000, floor     |
| `timestamp(us)` | exact identity                           | exact identity             |
| `timestamp(ns)` | divide by 1,000, floor (cannot overflow) | multiply by 1,000, checked |

Checked conversions fail with `CalcFlowError::InvalidArgument`; import
errors name the offending column, and silent wrap is forbidden.

Every conversion that produces a coarser representation from a finer value
floors toward negative infinity — one uniform direction with no per-call-site
choice (D1.5). For example, `-1,500 ns` imports to `-2 µs`, `1,999 ns`
imports to `1 µs`, and exporting `-1 µs` to seconds yields `-1 s`, not
`0 s`. Flooring never moves a row into a later window and keeps watermark
progress estimates conservative.

An Arrow timestamp column declared as an event-time column must be
timezone-naive (interpreted as UTC) or carry the explicit timezone `"UTC"`;
any other timezone is rejected with the column path in the error (D1.6).

## `Epoch`

`Epoch` is a newtype over `u64` identifying one checkpoint within a job
lineage (D9). Value `0` is reserved as the "no checkpoint" sentinel and is
unconstructable: `Epoch::new` returns `None` for it, and it is never
injected (D9.1). `Epoch::INITIAL` is `1`, the first checkpoint of a fresh
lineage, and `Epoch::next` increments by exactly one with a checked overflow
(D9.2). Epochs are strictly increasing within a job lineage. Serialization
stores the exact value.

The recovery and reuse rules that continue a lineage across jobs (D9.3–D9.6)
belong to the checkpoint protocol and arrive with it; see "Specified but not
yet implemented".

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
  while undefined (S5.2);
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
rows before enqueue; the public constructor remains source-compatible and
uses the default budget plus an isolated recorder.

## The operator emission boundary

`StreamOperator` is the continuously running operator trait: it receives one
named-ingress batch per call and emits only through a `StreamCollector`. The
port, schema, and UDF metadata accessors live on the shared `OperatorMetadata`
supertrait so the batch and stream compilers validate them once. The
handlers are:

- `process_data(ingress, batch, context, output)` — processes one batch
  from the named ingress. A failed handler never forwards a partial control
  event (S1.3).
- `on_watermark(watermark, context, output)` — reacts to an input-watermark
  advance; a window operator emits newly closed windows before the runtime
  forwards the watermark (S5.2).
- `on_end(context, output)` — flushes once after every ingress has ended
  (S1.6, S5.5).

Handlers never see barriers, and watermarks arrive as typed `EventTime`
values.

The state lifecycle is synchronous and executor-safe. `checkpoint(epoch)`
captures dirty state as an `OperatorStateSnapshot` in O(dirty-key
metadata) — never a bulk encode on the executor thread; durable staging is
runtime-owned (D4.1). `restore(snapshot)` applies a captured snapshot, and
`reset()` returns the operator to its freshly constructed state. The
defaults are stateless: capture returns an empty snapshot, restore rejects a
non-empty one, and reset succeeds.

`OperatorStateSnapshot` carries small bounded JSON `inline_metadata` plus
named byte `segments`. Keyed row state never appears inline (D4.4), and no
segment may carry secrets (I4); the runtime assigns segment paths, lengths,
and checksums during staging (D4.1).

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
an edge, so an invalid batch never produces a downstream side effect (S5.4,
S10.1). Control messages can never be emitted through this trait (S1.3):
watermark, barrier, idle, and end-of-input forwarding is runtime-owned.

`EdgeCollector` is a public in-memory `StreamCollector` helper for tests,
benchmarks, and callers that invoke an operator directly. It validates against
the supplied output ports and stores data messages in per-port FIFO outboxes;
`drain(port)` empties one outbox, while an unknown port drains to empty. It is
not the collector used by the production M2 task runtime.

The crate-private M3 operator task instead constructs a
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
`R >= max(required_row_limit, required_simultaneous_messages)`. The public
`EdgeBudget` shape and
`edge_channel` signature are unchanged, but zero-cost traffic can now block
earlier than it did under row/byte-only admission.

Reservation is atomic with enqueue and is released exactly once on receive or
queue teardown. A blocked send owns no reservation. An oversized data message
fails before enqueue, a closed receiver wakes blocked senders, and each fan-out
edge is charged independently even though immutable payload buffers are
shared. The default `Block` path therefore propagates a slow consumer through
sink, operator, source task, prefetch slot, and source pump. This is the narrow
M2 completion revision of S10.1 and S10.5: S10.2 row/byte accounting, S10.3
oversize and pre-open validation, S10.4 policy behavior, FIFO, close wakeup,
no-lost-wakeup, and one-producer ownership remain unchanged.

## Stream plan compilation

`PipelineBuilder::compile_stream` validates and consumes a graph into an
immutable `StreamExecutionPlan`. Every stream-rule violation is reported at
compile time, before any source opens:

- multi-input (multi-alias) SQL nodes are rejected;
- nodes offering only a batch operator are rejected;
- delivery requirements naming an unknown graph output are rejected;
- volatile DataFusion scalar UDFs are rejected when any output requests
  exactly-once delivery.

Stream graphs compose unary expression nodes, single-input SQL nodes,
explicit stream providers, and `UnionOperator`, whose two or more input
ports share one kind, one required flag, and one exact schema, or are all
schema-less.

The compiled plan records the deterministic topology, stable edge IDs, the
source and sink binding slots (external input and output names in
deterministic order), the semantic fingerprint, and the per-output delivery
requirements. It never executes directly. The crate-private M2 runner consumes
it into owned runtime nodes, internal edges, and synthesized bounded source and
sink boundary edges. Pure array graphs need no table engine:
`requires_datafusion` reports whether any node needs a DataFusion session.

`StreamRequirements` records the requested `DeliveryGuarantee` per graph
output — `AtLeastOnce` or `ExactlyOnce`; outputs absent from the map default
to `AtLeastOnce`. The guarantee is a per-sink contract, not a global
property of the plan. Exactly-once delivery itself is checkpoint-protocol
work; today the requirement is recorded in the plan and drives the
compile-time determinism check above.

Two hashes describe the plan (NFR-5):

- the semantic fingerprint covers execution mode, graph structure, operator
  configurations, and the UDF catalog; it decides checkpoint compatibility;
- the runtime-config hash covers `StreamRuntimeConfig` — checkpoint interval
  and timeout, the per-edge envelope/row/byte budget, and retained epochs —
  and feeds observability and diagnostics only, so retuning it never invalidates
  checkpoints. Durations must be exact multiples of one microsecond, and
  both budget fields must be positive. The defaults are a 60-second checkpoint
  interval, a 600-second checkpoint timeout, 10,000 envelopes, 10,000 rows and
  64 MiB per edge, and two retained epochs.

## Current internal M2 runtime slice

The crate-private M2 slice can run a bounded source-to-operator-to-sink job. It
is a complete internal runtime skeleton, not a public `StreamingRunner`
replacement.

One pure whole-job preflight consumes the plan and validates the context
fingerprint, runtime topology, every source and sink route, duplicate or
missing bindings, source capabilities and first-hop budgets, process-local
ordinary-sink capability, and the absence of pre-M5 exactly-once requests.
Capabilities are sampled once. No connector opens and no task is registered
until the complete job passes.

A source binding owns two supervised tasks (D3):

- the pump is the only caller of `StreamSource::open` and `next`, keeps at
  most one poll in flight, and feeds a one-item prefetch slot;
- the source task drains that slot, remains responsive to cancellation, and
  sends each event through the bounded first-hop edges;
- teardown drops a blocked `open` or `next` future, never polls that source
  again, and calls `close` before the pump finishes.

For each data event, the binding ID replaces connector-supplied source
identity, the runtime assigns a checked per-binding sequence, and existing
JSON attributes are preserved immutably. Cursor order keys must strictly
advance, including past a recovery cursor, or the batch is rejected before
enqueue. Source-provided watermarks must not regress. `None` becomes one
ordered `EndOfInput`; source-provided `Idle` and watermark events share the
same per-edge FIFO as data. Progress keeps the latest observed cursor separate
from the durable recovery cursor, which remains unchanged until checkpointing
is implemented.

`TaskSupervisor` assigns stable IDs in registration order and start-gates
tasks until registration is complete. The first observed task failure starts
cancellation convergence, every sibling is joined, simultaneous primary
failures are reported in stable task-ID order, and panics become
`CalcFlowError::TaskPanicked` with the stable task identity. Source failures
signal cancellation before potentially blocking teardown work, so siblings
can converge promptly.

One operator task owns each compiled stream operator. It selects ready
ingresses without weakening per-ingress FIFO, validates data emissions before
the first send, fans out over real bounded edges, and owns one lazy
operator-scoped DataFusion runtime when table work requires it. Aggregate
watermarks call `on_watermark` before runtime forwarding; aggregate idle and
all-ingress end are runtime-owned. Multi-ingress watermark/idle use the M3
minimum/idle-epoch semantics, and every barrier fails closed until M5.
`on_end` runs exactly once
after every ingress observes explicit end-of-input. A closed channel without
an earlier explicit end is a failure, not synthetic EOF.

One sink task owns each graph output and writes each batch to its configured
ordinary sinks in stable order. The third sink does not see a batch when the
second sink fails. Natural completion and graceful shutdown drain the accepted
prefix; explicit cancellation makes no drain promise. M2 ordinary sinks offer
process-local ordered delivery only, not a cross-process at-least-once claim.

The crate-private `ContinuousRunner` and `ContinuousJob` own three-stage
launch, status, terminal arbitration, cancellation, graceful drain, joining,
and a runner-scoped reaper. Operator entry completes before connector open;
all connector opens complete before the data gate is released. A dropped start
observer cancels the provisional launch. A dropped job transfers convergence
ownership to the reaper, and a later start or runner shutdown joins it. Task
failure wins over explicit cancel, which wins over deadline expiry; concurrent
and repeated observers receive one immutable outcome.

Private M2 source, operator, and sink tasks, the task supervisor, and runner
connector-open/close paths create operational
`CalcFlowError::TaskPanicked` values with a stable task ID. Together these
boundaries cover connector `open`, `next`, `write`, and `close` panics plus
operator and uncaught task panics. Captured panic text is valid UTF-8 and at
most 1,024 bytes; non-string payloads become `non-string panic payload`. On a
failed or cancelled launch, every resource whose open began is closed once in
stable resource order. Each close has its own private five-second bound, and a
close error, panic, or timeout becomes a typed bounded secondary diagnostic
without replacing the primary launch failure. Cleanup continues through later
resources and converges the provisional launch and reaper registries.

Private deterministic status and metrics cover task/terminal state,
source/operator/sink progress, per-edge input/output batches, rows, bytes,
queue depth and high-water marks, and blocked-send counts/durations. Their
registries use only stable preflight IDs and numeric values; batch IDs, cursor
payloads, watermarks, epochs, row values, attributes, secrets, and arbitrary
labels are absent.

The short stress suite covers 100 deterministic gate schedules plus sustained
zero-cost idle, watermark, and empty-data pressure. The universal calc-flow
soak is the ignored crate-private
`runtime::streaming::soak::twenty_minute_two_source_slow_sink` test:

```bash
CALC_FLOW_STREAM_SOAK=1 cargo test -p calc-flow --lib runtime::streaming::soak::twenty_minute_two_source_slow_sink -- --ignored --exact --nocapture
```

Every calc-flow soak uses exactly 1,200 measured seconds, a ten-second cadence,
exactly 120 Linux RSS samples, and a 30-sample/300-second warm-up. The M2 soak
also verifies slot pressure from zero-cost batches, accepted-to-both-sinks
conservation, graceful drain, connector closure, and final task, queue, live,
and reaper convergence. It remains opt-in and does not expand the public API.

## Delivery guarantees in the current runtime

The executing runners remain `MicroBatchRunner` and the push-based
`StreamingRunner` over `BatchExecutionPlan`. Both deliver sinks before
committing checkpoints, giving at-least-once delivery: after a failure a
sink may observe the same batch again. The streaming runner:

- requires a plan with exactly one external input and holds an exclusive
  lease on it;
- executes one pushed batch per `step`, writing all sinks before saving the
  checkpoint that commits the step;
- owns no replay cursor in push mode, so the caller retains or reconstructs
  the batch and resubmits it after a failed step;
- recovers once from the durable checkpoint and continues its sequence from
  there;
- rolls operator state back on failure, and refuses further work until
  `reset` if a rollback itself fails to complete.

For the public stream surface, the guarantees that hold today remain
compile-time and type-level: invalid stream graphs fail before any source
opens, emission validation fails closed before enqueue, and control
construction is impossible outside the crate. The internal M2 slice also
enforces the source ordering, backpressure, cancellation, close, and join
rules described above. It has no public runner, sink delivery protocol, or
checkpoint commit, so it does not add a user-visible delivery guarantee.

## Ownership, visibility, and non-goals

Public: the `StreamMessage` handle and `StreamMessageKind`, the typed
`EventTime` and `Epoch` values, `StreamJobContext` and
`StreamOperatorContext`, the `StreamOperator` and `StreamCollector` traits,
`OperatorStateSnapshot`, `EdgeCollector`, and the compiled plan types
(`StreamExecutionPlan`, `StreamRequirements`, `DeliveryGuarantee`,
`StreamRuntimeConfig`, `EdgeBudget`). The bounded-edge surface is also public:
`edge_channel`, `EdgeSender`, `EdgeReceiver`, `EnvelopeCost`, and
`ChannelMetrics`. `CalcFlowError::TaskPanicked` is public because the error
enum is public and non-exhaustive. Operational instances come from the private
M2 task, supervisor, and runner paths described above; M2 adds no public
runner/control API or panic-capture constructor.

Crate-private: the message representation and the four control constructors,
the compiled-operator representation, the late-row recorder, source cursor and
binding types, the channel-backed operator collector, source/operator/sink
tasks and progress snapshots, scoped task contexts, whole-job preflight, the
task supervisor, private status/metrics, `ContinuousRunner`, `ContinuousJob`,
and the runner-scoped reaper.

The envelope and its typed values do not appear in the current project-v2 or
checkpoint-v2 document formats, the Python binding, or Studio routes. M4 adds
the standalone v3 manifest model and state backend without replacing those v2
surfaces; runtime coordination and cross-language projections arrive with the
milestones below.

Non-goals:

- **No public runner control API.** Runners accept formed `Batch` values
  only; no public surface injects watermarks, barriers, idle, or
  end-of-input messages.
- **No executable-object serialization.** Operator configurations carry
  only `UdfReference` values — never source text, callables, or import
  paths — and checkpoint state carries JSON metadata and byte segments,
  never operator instances.
- **No partial public continuous API.** The internal M2 slice is not exported
  through aliases or a provisional runner. The existing v2 push runners and
  v2 project/checkpoint documents remain the current public surfaces until
  later milestones replace them atomically.
- **No payload leakage in diagnostics.** Debug output and metrics show
  kinds and typed business values only; row payloads, metadata, and
  attributes never appear (I4).

## M3 progress implementation boundary

M3 prepares each source policy during whole-job preflight, then routes raw
source data/control through one job-scoped progress driver. That driver alone
owns the logical clock, binding/local/global ordering, finite inbox fences,
timer heap, idle epochs, aggregate progress, completion receipts, and the
lossless admission/drain/terminal/settlement execution trace. Multi-ingress
progress uses the minimum watermark of known active inputs; idle and ended
inputs are excluded, data and legal watermarks reactivate before processing,
and all-ended emits one plain `EndOfInput` without a sentinel watermark.

The crate-private transient snapshot captures the exact prepared/config,
upstream cursor/control, trace, gate/fence, allocator, aggregate, and timer
coordinate only at a receipt-quiescent boundary. Restore requires paused
upstreams at exact captured positions and field-for-field equality before any
state or runtime side effect is installed. This provides snapshot-ready and
deterministically replayable progress state in one process. It is not
serialized, durable, a checkpoint, or crash recovery.

M3 forwards old/equal/new event-time rows unchanged. It neither classifies nor
drops late rows and exposes no late-row runtime metric.

## M4 state and window implementation boundary

M4 adds a public, data-only `WindowSpec` for fixed UTC tumbling and hopping
windows and the stream-only `WindowAggregateOperator`. The compiler validates
geometry, overlap, exact event-time/group types, aggregate combinations,
output names, state layout, and deterministic configuration fingerprints
before source open. Supported aggregates are `count`, `sum`, `min`, `max`,
and `avg` over the explicit first-version type matrix; decimal, session,
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
The M4 commit harness proves publication crash boundaries, but the running job
still rejects barriers and does not perform durable checkpoint coordination.

## M5 planned checkpoint boundary

The reviewed M5 plan is defined by the
[epoch checkpoint specification](../.codex/artifacts/specs/m5-epoch-checkpoint.md),
its [API note](../.codex/artifacts/api-notes/m5-epoch-checkpoint.md), and its
[adversarial critique](../.codex/artifacts/critiques/m5-epoch-checkpoint.md).
These documents describe future implementation; none of the behavior in this
section is present merely because the plan exists.

M5 reuses the M4 `CheckpointManifest` v3 as the single durable truth and
productionizes segment/manifest publication, strict latest-completed
selection, retention, and complete-job restore. Semantic identity includes
the pipeline fingerprint and exact participant sets. The runtime configuration
hash remains canonical diagnostic metadata but does not invalidate compatible
state. Recovery persists a bounded semantic projection of source/operator
progress, not the M3 execution trace, pending receipts, or process-local timer
coordinates.

One single-flight coordinator creates a global source cut through the
`LiveProgressCoordinator`, which remains the sole owner of source-output
edges. Operators block only ingresses that have received the current epoch,
snapshot after full alignment, and immediately forward the barrier after the
local snapshot succeeds. Transactional sinks pre-commit before manifest
publication and commit only after the manifest is durable; a post-manifest
failure completes forward during recovery.

Final-only output uses a coordinator-owned terminal epoch after the data-plane
end cut. No barrier follows `EndOfInput`. Recovery from a terminal manifest
finishes any sink commit and returns terminal success without reopening
sources or re-emitting final windows. Exactly-once is proved per output over
every reachable source, operator, edge policy, and sink before lifecycle work.
The milestone remains crate-private; public v2 checkpoint/runner APIs stay
unchanged until post-M5 A6.

## Specified but not yet implemented

The [specification](../.codex/artifacts/specs/continuous-streaming-runtime.md)
and the [v3 implementation plan](superpowers/plans/2026-08-02-continuous-streaming-v3.md)
assign the following behaviors to later milestones. They are not implemented
in the current tree, and this document does not describe them as present:

- **M5** — the checkpoint behavior summarized above: runtime-generated source
  cuts and barriers, alignment, epoch manifest publication/selection,
  complete-job and terminal restore, lineage recovery numbering, per-output
  capability proof, and exactly-once sink commit/recovery.
- **Post-M5 public A6 integration** — the source-driven runner/job, public
  source and sink bindings, status/control methods, and v2 runner replacement
  land atomically only after the M4 state and complete M5 durability contract
  have passed a separate review. M2 runtime internals complete does not expose
  or replace the current public v2 runners.
- **M6** — Python and Studio projections of the stream surface.
