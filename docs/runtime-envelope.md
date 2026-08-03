# Stream message envelope

The v3 Rust core moves stream traffic on one typed message: `StreamMessage`.
Each stream edge carries a single ordered sequence of data and control
messages from one producer to one consumer. This document is the normative
contract for that envelope: the message type, its typed `EventTime` and
`Epoch` values, the job and operator contexts, the emission boundary stream
operators see, the compiled stream plan, and the delivery invariants the
runtime guarantees today.

The implementation lives in `crates/calc-flow/src/runtime/streaming/` (the
message and the job context), `crates/calc-flow/src/time/` (event time and
epoch), `crates/calc-flow/src/operator/stream.rs` (the operator traits and
the validating collector), and `crates/calc-flow/src/pipeline/stream.rs`
(the compiled stream plan). The frozen semantics behind the contract are
recorded in the [continuous streaming runtime
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
before any downstream side effect (S1.3, S5.4). Those paths arrive with the
milestones listed under "Specified but not yet implemented".

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
  payloads are never accepted, and only window operators call this. The
  current version never fails; the `Result` keeps the frozen signature
  stable for the validation rules later milestones add. Reporting these
  counters as metrics arrives with the late-data policy work (D2.5).

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

`StreamCollector` is the only way a stream operator emits. `emit(port,
batch)` validates the port name, the `BatchKind`, and the optional exact
Arrow schema against the compiled output ports, then enqueues the batch; a
validation failure returns `CalcFlowError::Compile` before the batch reaches
an edge, so an invalid batch never produces a downstream side effect (S5.4,
S10.1). Control messages can never be emitted through this trait (S1.3):
watermark, barrier, idle, and end-of-input forwarding is runtime-owned.

`EdgeCollector` is the runtime-owned validating `StreamCollector`. One
collector is constructed per operator from the compiled output ports; each
port owns an outbox, and `drain(port)` returns that port's pending messages
in FIFO order, with an unknown port draining to empty. The in-memory outbox
is the current backing; bounded edge channels replace the storage without
changing this validation contract, as listed below.

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
ports share one kind and one exact schema, or are all schema-less.

The compiled plan records the deterministic topology, stable edge IDs, the
source and sink binding slots (external input and output names in
deterministic order), the semantic fingerprint, and the per-output delivery
requirements. It never executes directly; the continuous runner that
executes it arrives with the runtime milestones below. Pure array graphs
need no table engine: `requires_datafusion` reports whether any node needs
a DataFusion session.

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
  and timeout, the per-edge row/byte budget, and retained epochs — and feeds
  observability and diagnostics only, so retuning it never invalidates
  checkpoints. Durations must be exact multiples of one microsecond, and
  both budget limits must be positive. The defaults are a 60-second
  checkpoint interval, a 600-second checkpoint timeout, 10,000 rows and
  64 MiB per edge, and two retained epochs.

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

For the stream surface, the guarantees that hold today are compile-time and
type-level: invalid stream graphs fail before any source opens, emission
validation fails closed before enqueue, and control construction is
impossible outside the crate. The runtime ordering, forwarding, and
checkpoint semantics of the specification are enforced by machinery later
milestones add; they are listed below, not claimed here.

## Ownership, visibility, and non-goals

Public: the `StreamMessage` handle and `StreamMessageKind`, the typed
`EventTime` and `Epoch` values, `StreamJobContext` and
`StreamOperatorContext`, the `StreamOperator` and `StreamCollector` traits,
`OperatorStateSnapshot`, `EdgeCollector`, and the compiled plan types
(`StreamExecutionPlan`, `StreamRequirements`, `DeliveryGuarantee`,
`StreamRuntimeConfig`, `EdgeBudget`).

Crate-private: the message representation and the four control constructors,
the compiled-operator representation, and the late-row recorder.

The envelope and its typed values do not appear in the current project or
checkpoint document formats, the Python binding, or Studio routes. The
durable manifest and the cross-language projections arrive with the
milestones below.

Non-goals:

- **No public runner control API.** Runners accept formed `Batch` values
  only; no public surface injects watermarks, barriers, idle, or
  end-of-input messages.
- **No executable-object serialization.** Operator configurations carry
  only `UdfReference` values — never source text, callables, or import
  paths — and checkpoint state carries JSON metadata and byte segments,
  never operator instances.
- **No v2 compatibility surface.** The batch/stream split is an intentional
  breaking change; v2 runner APIs, documents, and shims are rejected rather
  than adapted (NG9).
- **No payload leakage in diagnostics.** Debug output and metrics show
  kinds and typed business values only; row payloads, metadata, and
  attributes never appear (I4).

## Specified but not yet implemented

The [specification](../.codex/artifacts/specs/continuous-streaming-runtime.md)
and the [v3 implementation plan](superpowers/plans/2026-08-02-continuous-streaming-v3.md)
assign the following behaviors to later milestones. They are not implemented
in the current tree, and this document does not describe them as present:

- **M1.3–M1.4** — bounded edge channels with row and byte hard limits and
  backpressure (S10), including uniform batch byte metering.
- **M2** — production construction and runtime-owned forwarding of control
  messages, per-edge FIFO channels, source and sink tasks, the task
  supervisor, and the continuous runner that executes `StreamExecutionPlan`
  (S1–S3).
- **M3** — source watermark policies, watermark monotonicity enforcement,
  idle declaration and reactivation, and the multi-ingress watermark minimum
  (S5); late-data drop behavior and its metrics (D2.5).
- **M4** — window assignment and final-only triggers (S6, D8); state
  backends and durable segment staging (D4).
- **M5** — barrier injection and alignment, epoch checkpoint manifests,
  lineage recovery numbering (D9.3–D9.6), and the exactly-once sink commit
  protocol (S7, S9).
- **M6** — Python and Studio projections of the stream surface.
