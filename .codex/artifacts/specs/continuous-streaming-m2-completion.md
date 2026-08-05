# Continuous Streaming M2 Internal Runtime Completion - Delta Specification

## Source and status

- Request: Cheng Li, 2026-08-06 — finish PR #83 as **M2 runtime internals
  complete**, using one universal 20-minute soak standard.
- Milestone plan:
  [`docs/superpowers/plans/2026-08-02-continuous-streaming-v3.md`](../../../docs/superpowers/plans/2026-08-02-continuous-streaming-v3.md),
  tasks M2.1-M2.5 and the M2 merge gate.
- Research basis:
  [`docs/research/2026-08-02-arroyo-risingwave-streaming-research.md`](../../../docs/research/2026-08-02-arroyo-risingwave-streaming-research.md).
- Controlling semantic specification:
  [`continuous-streaming-runtime.md`](continuous-streaming-runtime.md).
- Controlling public API note:
  [`../api-notes/continuous-streaming-runtime.md`](../api-notes/continuous-streaming-runtime.md).
- Approved critique:
  [`../critiques/continuous-streaming-runtime.md`](../critiques/continuous-streaming-runtime.md),
  round 3, `BLOCKS REMAINING: 0`.
- M2 completion API note and critique:
  [`../api-notes/continuous-streaming-m2-completion.md`](../api-notes/continuous-streaming-m2-completion.md)
  and
  [`../critiques/continuous-streaming-m2-completion.md`](../critiques/continuous-streaming-m2-completion.md),
  round 2 approved revision 3 with zero blocks; round 3 blocked revision 4 on
  B6 at artifact head `e77a8779f22f554c85a0ab7b9d78133e5292aa88`.
- Audit anchors, captured 2026-08-06: `origin/main` was
  `d45c2b26def2c4dfe179f2b7c7c1d2411fc069b6`; the PR head was
  `574c0fb7f678781370303100c9f5089f6ed59bca`; their merge base was
  `c4979061cc239b43617f642e2294794f2833d95d`; the PR was two commits ahead
  and one commit behind `origin/main`. The first PR commit
  `1d5546028e2ce9ebce59c976080c3d11c1225e16` is the historical skeleton
  baseline. It is not the current-PR baseline used by this revision.
- GitHub audit at `574c0fb`: 13 Actions checks passed; Codacy failed with 16
  new medium complexity findings; three Copilot threads were unresolved, one
  of them outdated but not resolved. These are current-gap evidence only.
  After any push, none of these old-head results is final-head evidence.
- Status: revision 5 is the proposed completion contract and current-PR
  ledger. It
  does not certify PR #83 or the continuous-streaming feature as complete.
  Revision 2 addressed B1, B2, B4, B5, S1, S2, S4, S6, S7, and M1 from the
  completion critique. Revision 3 accepted the crate-private real-runtime
  soak seam. Revision 4 reconciles those requirements with the current PR,
  closes the soak duration decision at 20 minutes, and adds the remaining
  review and delivery gates. Revision 5 closes B6 with a finite message-slot
  invariant and resolves the round-3 evidence, compatibility, verification,
  and performance advisories. It remains pending critic review.
- Scope precedence: D1-D9, S1-S10, I1-I10, NG1-NG13, and A1-A8 remain frozen
  except for soak duration. For every calc-flow soak, this later specification
  replaces an earlier one-hour duration, one-hour test name, sample-count
  default, or command with the universal 20-minute standard in M2C-NFR4 and
  M2C-FR32. That timing-only supersession includes the original plan's M2.5,
  M7.1, and verification-matrix wording, total-spec NFR-3, and the completion
  API note's soak seam. Those sources must be reconciled as documentation
  drift before delivery.

## Problem statement

PR #83 now implements most of the crate-private M2 runtime that revision 3
specified, including whole-job preflight, operator and sink tasks, the private
runner/job lifecycle, metrics, stress, a real-runtime soak harness, and
benchmarks. It is not deliverable yet: admission does not enforce the message
count already carried by `EnvelopeCost`, so sustained zero-row/zero-byte
traffic can grow an edge queue without bound. Two source diagnostics are not
binding qualified, panic payloads are unbounded, the harness still encodes one
hour, the required Linux 20-minute evidence does not exist, normative docs
describe implemented work as absent, Codacy reports 16 new medium complexity
findings, and three Copilot threads remain unresolved. Public A6 also requires
the checkpoint and state behavior assigned to M4/M5, so the public cut remains
a separate post-M5 integration gate.

## Completion definition

“M2 runtime internals complete” means all M2C-FR1-M2C-FR33, including every
letter-suffixed requirement, and all acceptance criteria in this document are
satisfied at one exact PR head. Every queued data or control envelope consumes
one finite message slot in addition to its row/byte charge. Completion also
requires the real 20-minute Linux soak and repository/review gates. It means
the internal source-driven graph is bounded, conserves the accepted prefix in
a fault-free run, and converges every owned task. It does **not** mean public
M2.4, public A6, M3-M7, checkpoint recovery, exactly-once delivery,
Python/Studio exposure, or the overall continuous-streaming feature is
complete.

## Current PR completion ledger

The ledger distinguishes the historical skeleton baseline from behavior
present at audited head `574c0fb`. “Implemented” is source-and-test evidence,
not final-head delivery evidence. “Remaining” is mandatory work for the
implementer/tester; deferred work must not be pulled into M2.

| Plan task | Current status                        | Current PR evidence                                    | Required closure                       |
| --------- | ------------------------------------- | ------------------------------------------------------ | -------------------------------------- |
| M2.1      | Implemented with G2 open              | Supervisor, lifecycle, launch, Drop, and reaper tests  | Bound panic text; pass final gates     |
| M2.2      | Implemented with G1 open              | Preflight, source, drain, fan-out, and resume tests    | Qualify both diagnostics; pass gates   |
| M2.3      | Implemented                           | Operator/FIFO/control/DataFusion/graph tests           | Preserve behavior through G5-G7        |
| M2.4      | Internal implementation present       | Sink order, private runner/job, drain/cancel/reaper    | Keep private; do not perform public A6 |
| M2.5      | Implemented with G3-G5/G8 open        | Metrics, stress, soak seam, and benchmarks             | Bound slots; run soak; clear gates     |
| M2 gate   | Behavior present; not yet deliverable | Two-source graph, backpressure, drain, and convergence | Satisfy G1-G8 at one exact pushed head |

The current public `StreamingRunner` remains the v2 push-based runner, and the
current `MicroBatchRunner`, v2 `Source`, v2 `Sink`, and v2 `CheckpointStore`
still exist. No M2 test or document may describe the crate-private runtime as
the public source-driven A6 runner.

Representative current-head evidence is:

- **M2.1:** `simultaneous_failures_are_returned_in_stable_task_order`,
  `terminal_observers_are_idempotent_and_dropped_wait_does_not_cancel`, and
  `dropped_job_transfers_driver_and_next_start_reaps_before_launch`.
- **M2.2:**
  `whole_job_preflight_freezes_capabilities_and_all_boundary_routes_before_open`,
  `data_metadata_cursor_fanout_and_end_are_ordered`,
  `two_sources_reopen_once_at_their_distinct_resume_cursors`, and
  `slow_sink_backpressures_both_sources_after_bounded_prefetch`.
- **M2.3:** `ready_selection_preserves_each_ingress_fifo_and_ends_once`,
  `validation_is_zero_output_but_late_branch_close_keeps_earlier_prefix`, and
  `expression_and_sql_reuse_one_lazy_task_owned_datafusion_runtime`.
- **M2.4:** `configured_sink_order_finishes_batch_before_next_batch`,
  `two_sources_union_expression_and_ordered_sinks_complete_end_to_end`, and
  `graceful_drains_the_accepted_prefix_while_cancel_makes_no_drain_promise`.
- **M2.5:**
  `seeded_paused_time_stress_runs_one_hundred_full_graph_schedules`, the
  current ignored soak plus its RSS/conservation smoke tests, and Criterion
  `stream/channel_data_roundtrip`, `stream/unary_operator_overhead`, and
  `stream/fanout_four`.

The evidence map is not an acceptance waiver. G1-G8 and the final-head matrix
still decide completion.

## Current acceptance gaps

- **G1 — Implementer/tester:** cursor regression/repeat uses
  `sources.<binding_id>.cursor`; watermark regression uses
  `sources.<binding_id>.watermark`; tests assert both exact fields and
  pre-enqueue failure.
- **G2 — Implementer/tester:** string panic payloads are bounded to 1,024
  UTF-8 bytes using M2C-FR1A; ASCII, multibyte-boundary, and non-string cases
  are tested.
- **G3 — Implementer/tester:** the ignored real-runtime soak is
  renamed/reconfigured for 20 minutes; the exact Linux command actually runs
  at the final head and emits boundedness/RSS, conservation, and convergence
  evidence.
- **G4 — Implementer/doc writer:** `docs/runtime-envelope.md` no longer calls
  whole-job preflight, M2.3, private M2.4, or M2.5 absent; all normative
  one-hour soak wording is reconciled to 20 minutes.
- **G5 — Implementer/reviewer:** all 16 current Codacy medium complexity
  findings are removed by behavior-preserving simplification; no waiver,
  exclusion, threshold, or analysis-configuration change is accepted.
- **G6 — Reviewer:** all three current Copilot threads are addressed and
  resolved; outdated status alone does not count as resolution.
- **G7 — Reviewer:** incorporate current `origin/main`, then prove the pushed
  exact head has every required Action and Codacy check green, zero unresolved
  threads, and mergeability clean.
- **G8 — Implementer/tester:** enforce the M2C-FR24A hard message-slot limit
  on every internal and synthesized boundary edge; prove bounded/resumable
  sustained `Idle`, non-regressing watermark, and empty zero-cost data traffic
  plus leak-free shutdown, cancellation, receiver-close, and task-error paths.

## Goals

- Complete a genuinely runnable source-to-operator-to-sink continuous graph
  for unary chains, fan-out, independent branches, and a two-source union.
- Complete structured lifecycle ownership for launch, natural completion,
  failure, explicit cancellation, graceful drain, deadline expiry, dropped
  job handles, and runner shutdown, without detached Tokio tasks.
- Validate every source and sink binding, capability bound, and route before
  any source or sink opens.
- Deliver ordinary sink behavior with deterministic configured ordering and
  honest, M2-scoped delivery reporting.
- Make boundedness, progress, task convergence, and backpressure observable
  through deterministic minimum metrics and status snapshots. Boundedness
  covers message slots, rows, and bytes for data and control traffic.
- Add deterministic short stress coverage, an explicitly opt-in soak, and
  compilable Criterion cases without making the 20-minute run part of ordinary
  CI.
- Keep all M3+ semantics out of this internal gate and keep existing public v2
  runners usable until their complete v3 replacement can land atomically.

## Non-goals

- No generated watermark policy, multi-input watermark minimum, idle
  reactivation semantics, late-data classification, or watermark recovery
  (M3).
- No keyed operator state, tumbling/hopping window execution, state segments,
  retention, or compaction (M4).
- No epoch coordinator, barrier injection/alignment, checkpoint manifest,
  source durable-cursor advancement, transactional sink, `trigger_checkpoint`
  implementation, or exactly-once claim (M5).
- No connector registry, project v3, Python, Studio, OpenAPI, packaging, or
  deployment work (M6+).
- No session windows, multi-input SQL joins, changelog/retract, distributed
  workers, or any other controlling-spec non-goal.
- No deletion of the v2 public runners in the unreviewed internal completion
  slice. The compatibility gate below controls that change.

## Functional requirements

### M2.1 — lifecycle and ownership completion

- **M2C-FR1 — Single terminal decision.** Every terminal trigger enters one
  serialized decision point. Already-observed task failure/panic wins over
  explicit cancel, which wins over deadline expiry, as required by S8.4.
  Later convergence errors are attached in stable `TaskId` order and never
  replace the primary trigger.
- **M2C-FR1A — Bounded panic payload.** A supervised string panic preserves
  at most 1,024 UTF-8 bytes. If truncation is required, the runtime retains the
  longest prefix that ends on a character boundary and leaves room for the
  three-byte UTF-8 ellipsis `…`; the stored message, including that ellipsis,
  is at most 1,024 bytes. A non-string payload remains exactly
  `non-string panic payload`. The existing stable task identity and public
  non-exhaustive `TaskPanicked` error shape do not change.
- **M2C-FR2 — Six states.** Runtime state contains exactly `running`,
  `draining`, `completed`, `cancelled`, `failed`, and `recovery-required`.
  Natural source exhaustion and graceful shutdown enter `draining` before
  `completed`; cancel does not promise drain. Terminal state is immutable.
- **M2C-FR3 — Classification.** Logic, validation, invariant, operator, and
  panic errors classify as `failed`. I/O, external-system, and other
  explicitly recoverable failures classify as `recovery-required`.
  Deadline handling must not be inferred from an empty error vector. The
  frozen M2 result is `state = Cancelled`, `cause = DeadlineExceeded`, with an
  empty error list unless teardown adds a secondary failure. It is not an M5
  checkpoint timeout and does not use `CheckpointTimeout`.
- **M2C-FR4 — Idempotent observation.** Terminal outcome is stored once and
  every `wait`, `cancel`, or `shutdown` caller observes that same outcome.
  Dropping a wait future does not cancel the job.
- **M2C-FR5 — Core-owned convergence driver.** The supervisor registry and
  partially progressed joins never become local state of a caller's terminal
  future. One `JobCore`-owned, registered convergence driver owns them and
  publishes the terminal outcome. `wait`, `cancel`, and `shutdown` submit or
  observe causes only; dropping any observer future cannot cancel, abort, lose,
  or strand the registry. Concurrent terminal calls observe the same driver.
- **M2C-FR5A — Drop/reaper ownership.** Dropping the owning job handle fires
  cancellation and transfers ownership of the convergence driver and its
  registry to a runner-scoped reaper; it never blocks or joins in `Drop`.
  Starting the next job drains the reaper first. Completed runner shutdown
  cancels and joins live jobs, then joins the reaper, leaving both registries
  empty. Terminal cause selection linearizes when the driver first commits a
  cause; transfer linearizes when `JobCore` changes its owner to the reaper;
  outcome publication linearizes when the immutable outcome cell is set.
- **M2C-FR6 — No abort-as-normal-join.** `abort_all` may exist only as a
  last-resort destructor safety net after cancellation has been fired. The
  observable success, failure, cancel, drain, wait, and runner-shutdown paths
  must cooperatively converge and join; they must not depend on destructor
  aborts to satisfy lifecycle tests.
- **M2C-FR7 — Closed ingress convergence.** Once a terminal trigger is
  selected, runtime-owned entry senders close before or as cancellation is
  broadcast so blocked tasks wake, and all registered source pumps, source
  tasks, operator tasks, and sink tasks are joined before a terminal method
  resolves.
- **M2C-FR7A — Premature channel close.** While state is `running` or
  `draining`, `EdgeReceiver::recv() == None` before that ingress observed an
  explicit `EndOfInput` is an unexpected `EdgeClosed` failure naming the edge
  and ingress. After cancel/failure convergence has marked an edge as
  runtime-closed, `None` is a normal wakeup and records no new error. If an
  unexpected close was observed before a competing terminal cause won, its
  `EdgeClosed` is retained as a stable-ID secondary; it never becomes natural
  EOF.
- **M2C-FR7B — Three-stage launch.** Launch is ordered as: (1) pure whole-job
  preflight with no tasks or lifecycle calls; (2) registration of every
  operator task, execution inside that task of fresh-entry `reset()` (or the
  controlling A3 `restore()` entry when a later validated recovery snapshot
  exists), and collection of every entry acknowledgement; (3)
  runner-core-owned concurrent source/sink open. M2 creates only fresh jobs,
  so its observable entry action is one `reset()` per operator; it must not
  invent recovery. Only
  after every open succeeds does the runtime release a separate data/control
  gate for pumps, ingress loops, and sink writes. An operator-entry failure
  cancels and joins the registered entry tasks and invokes no connector
  lifecycle method, including `open` or `close`.
- **M2C-FR7C — Cancellation-safe start observer.** After pure preflight and
  before the first operator entry or connector open, `RunnerCore` installs and
  owns one provisional launch driver. `start()` is only an observer with a
  cancel-on-drop guard. Dropping it before the job handle is atomically
  delivered requests `cancel -> close -> join -> reaper`; it never drops
  future-local connectors or tasks. A later `start()` first joins that
  abandoned launch/reaper entry before accepting its new owned spec. Runner
  shutdown cancels and joins a provisional launch before live jobs and the
  reaper. Concurrent start while a non-abandoned provisional launch remains
  active returns the frozen active-job conflict. The alternative “continue to
  an unobserved live job” is not used.

### M2.2 — source integration completion

- **M2C-FR8 — Graph-wide source preflight.** The complete source binding map
  is validated as one pure operation before any task is spawned or any
  `StreamSource::open` is called. Missing, unknown, and duplicate bindings,
  empty output routes, zero capability bounds, and a declared maximum batch
  exceeding any first-hop edge budget name the binding and route in the
  error.
- **M2C-FR9 — Open phase boundary.** Connector opens begin only after all
  registered operator-entry acknowledgements succeed. Source and sink opens
  may then run concurrently under the provisional launch driver. Any open
  failure cancels the phase, calls `close` exactly once for every resource
  whose open began (including the resource whose open returned `Err`), joins
  every open unit, and prevents data/control-gate release. The primary open
  error is returned with stable binding origin; cleanup failures are retained
  once in stable-ID runner/reaper diagnostics. Tests record lifecycle calls
  explicitly; they do not use sleeps.
- **M2C-FR10 — D3 preserved.** Each binding has exactly one pump, one source
  task, and one capacity-one prefetch slot. The pump is the only caller of
  `next`, at most one `next` is in flight, downstream capacity stops further
  polling after at most the one prefetched item, and teardown drops the poll
  once, never resumes it, then calls `close`.
- **M2C-FR10A — Drain linearization.** Each source has one serialized state:
  `Polling`, `Slotted`, `Draining`, or `Closed`. `Polling` means no event is
  accepted, even when connector I/O has become ready. `Slotted` begins only at
  the atomic commit of one observed `Ready` event into the capacity-one slot.
  The slot commit and drain cut are mutually exclusive transitions under the
  same synchronization point. A drain cut changes `Polling -> Draining` and
  drops the uncommitted poll once, or changes `Slotted -> Draining` while
  retaining that slot. `Draining` permits no new slot commit, drains the
  retained slot and every already-started edge send, emits EOF, closes, joins,
  then becomes `Closed`. Future readiness time is never an acceptance fact.
- **M2C-FR11 — Authoritative data envelope.** For every data item, the
  binding ID replaces connector metadata source, runtime sequence replaces
  connector sequence, attributes are preserved, cursor regression or repeat
  fails before any branch enqueue, and sequence overflow fails before the
  overflowing item is observed downstream. Cursor regression/repeat reports
  exact machine-readable field `sources.<binding_id>.cursor`; the binding
  must not appear only in free-form text.
- **M2C-FR12 — Fan-out.** In a fault-free run every fan-out branch observes
  the same per-edge message sequence, payloads remain immutable/shared, each
  branch is charged independently, and a slow branch backpressures that
  source. Tests assert no sibling-branch interleaving order. Fan-out is not
  transactional: validation and cost checks before the first send guarantee
  zero output only for a validation failure. If a later branch send closes, or
  a handler fails after an earlier successful `emit`, earlier branches may
  retain an observable prefix. Convergence performs no rollback. Per-edge
  enqueue metrics expose the delivered prefix, while a source/operator
  fully-fanned-out counter advances only after every branch succeeds.
- **M2C-FR13 — EOF and progress.** `None` produces exactly one ordered
  `EndOfInput` on every branch; no `next` follows; close completes before the
  pump reports success. Progress distinguishes latest observed cursor from
  durable cursor, and M2 never advances the durable cursor because manifests
  are M5 work. Graceful shutdown drains exactly the accepted prefix defined by
  M2C-FR10A: an event committed to the slot before the cut is included; a
  pending, ready-but-unpolled, or polled-but-not-yet-committed event is not.
  Explicit cancel may discard the slot and carries no drain promise.
- **M2C-FR14 — Watermark/idle shape only.** M2 retains
  `SourceEvent::Watermark(EventTime)` and `SourceEvent::Idle`, validates
  source-provided watermark regression before enqueue, reports it with exact
  machine-readable field `sources.<binding_id>.watermark`, and permits only
  the runtime to construct downstream control messages. The binding must not
  appear only in free-form text. M2 does not generate watermarks or claim M3
  minimum/reactivation semantics. A multi-ingress watermark or idle event
  fails before handler or downstream output. No implementation may substitute
  max, last-arrival, or arbitrary forwarding for M3's frozen minimum.

### M2.3 — operator task completion

- **M2C-FR15 — One task per node.** Each compiled stream node owns exactly one
  supervised task. It owns its operator value, ingress receivers, validating
  channel-backed collector, and operator-scoped resources for the task's full
  lifetime.
- **M2C-FR16 — Multi-ingress selection.** Each ingress remains FIFO. The task
  awaits all non-ended ingress receivers and processes whichever is ready,
  without promising round-robin or cross-ingress business order. The design
  must not rebuild or reorder an ingress queue in user space.
- **M2C-FR17 — Data dispatch.** Only `Data` invokes `process_data`, with the
  compiled ingress name and immutable batch. Unary expression and unary SQL
  produce their existing per-batch result; union forwards every batch and
  preserves each input's FIFO.
- **M2C-FR18 — Validating collector and fan-out.** `StreamCollector::emit`
  validates output port, kind, and exact schema before any successor observes
  the batch, then sends to every compiled outgoing edge through its bounded
  sender. Validation failure occurs before the first send and therefore emits
  no data. A send failure after one or more successful branches may leave the
  non-transactional prefix allowed by M2C-FR12. An operator error emits no
  runtime-owned control message for that input event, but an external operator
  that already completed one `emit` before returning an error may likewise
  have produced an earlier data prefix. Both cases converge the whole job and
  are visible in per-edge metrics.
- **M2C-FR19 — Runtime-owned control.** Operators cannot construct, suppress,
  or reorder control messages. For a supported unary watermark, the runtime
  calls `on_watermark` and forwards the watermark only after the handler
  succeeds. After all ingresses end, the runtime calls `on_end` once and then
  forwards one `EndOfInput` per output. Barrier handling is unreachable in M2
  production paths and must fail closed if injected by a test-only path.
- **M2C-FR19A — Receiver closure is not end.** Operator ingress selection
  tracks explicit EOF independently for every receiver. `recv() == None`
  before that marker follows M2C-FR7A and must never enter the all-ended path.
  The same rule applies to unary, union, and sink ingress receivers.
- **M2C-FR20 — DataFusion ownership.** Each table expression/SQL operator
  lazily creates at most one operator-scoped `DataFusionRuntime`, reuses it
  across all batches, and drops it on every terminal path. An array-only or
  union-only graph creates none. Runtime creation is per operator, never
  job-wide.
- **M2C-FR21 — Required vertical graph.** A two-source union feeding an
  expression and recording sink emits every input batch exactly once in a
  fault-free run, preserves per-source order, applies bounded backpressure to
  both sources, and naturally terminates after both sources end.

### M2.4-internal — ordinary sink, runner core, and job observer

- **M2C-FR22 — Graph-wide sink preflight.** The complete sink binding map is
  validated in the same pre-open phase as sources. Every external output has
  at least one sink unless a typed explicit discard policy exists; unknown,
  missing, and duplicate routes fail. Sink delivery capability is sampled at
  most once. No source or sink lifecycle method runs before all source and
  sink validation succeeds.
- **M2C-FR23 — Ordered ordinary delivery.** One external output with multiple
  ordinary sinks delivers each data batch to sinks in configured sequence.
  Batch `N + 1` is not delivered to any sink before batch `N` has completed
  that configured sequence. A failure stops further delivery, closes all
  opened sinks during convergence, and preserves the primary sink identity in
  the error.
- **M2C-FR24 — Backpressure.** Sink writes await completion inside the sink
  task. A blocked sink fills only bounded downstream edges, then parks its
  upstream operator and ultimately stops source polling; no unbounded staging
  queue is permitted between an edge and a sink.
- **M2C-FR24A — Three-dimensional edge admission.** Every enqueued
  `StreamMessage` consumes exactly one message slot, including `Data`,
  `Watermark`, `Idle`, `EndOfInput`, and a test-only injected `Barrier`.
  `Data` independently retains its measured row and byte charges; therefore
  an empty table/external batch with zero rows and a zero-byte estimate still
  consumes one slot. To close B6 without adding a public configuration or
  control surface, the hard per-edge slot limit is exactly the existing
  positive `EdgeBudget.max_rows` value. The same numeric value separately
  limits charged rows and charged messages; satisfying one dimension never
  exempts the other. Admission atomically requires:
  `queued_messages + 1 <= max_rows`,
  `charged_rows + message_rows <= max_rows`, and
  `charged_bytes + message_bytes <= max_bytes`. The rule applies to internal
  edges, synthesized source/sink boundary edges, and the existing bounded
  channel primitive. A full slot dimension parks even a zero-row/zero-byte
  sender until a dequeue releases one slot. No `max_messages` field, public
  control constructor, or public runner method is introduced in M2.
- **M2C-FR24B — Slot lifecycle.** Slot, row, and byte reservation is one
  atomic queue commit; a blocked send owns no reservation. Successful dequeue
  releases all three charges exactly once and then wakes the sender.
  Graceful shutdown drains every already-accepted queue item, committed source
  slot, and already-started edge send before ordered EOF and completion.
  Explicit cancellation may discard accepted contents but must close the
  receiver, wake every blocked send, release every queued charge exactly once,
  and join every task. Receiver close or upstream task error follows the
  M2C-FR7A primary/secondary error rule and leaves no queued slot, row, or byte
  charge after convergence.
- **M2C-FR25 — Natural completion and graceful drain.** All-ended sources
  propagate one terminal end through each node. Sink tasks consume every
  batch in the accepted prefix committed before each source's M2C-FR10A drain
  cut, then close. Natural completion and `shutdown` resolve only after that
  prefix is drained and all runtime tasks join. M2 must not claim a final
  epoch manifest or call an epoch-valued sink flush.
- **M2C-FR26 — Cancel.** `cancel` broadcasts cancellation, closes entry
  points, drops in-flight source/sink futures only under their teardown
  contracts, calls close, and joins all tasks. It does not promise that
  already accepted queue contents reach sinks.
- **M2C-FR27 — Status.** Status is a cheap synchronous deterministic snapshot
  containing job ID/state, stable task registry, per-edge channel metrics,
  per-source observed/durable/sequence/end progress, and per-sink delivered
  progress. Watermark and epoch fields may be structurally absent or
  explicitly unavailable; they must not fabricate values.
- **M2C-FR28 — Internal-only gate.** This artifact gates **M2 internal runtime
  completion** only. Every new source, sink, runner-core, launch-driver, job,
  status, metrics, and reaper type remains `pub(crate)` and is not re-exported.
  The total A6/M4/M5 durability contract supersedes the plan's original M2.4
  public cut; public source-driven A6 integration occurs only after complete
  M5 checkpoint behavior. Until then, the existing v2 `StreamingRunner` and
  `MicroBatchRunner` remain unchanged and tested. Status reports must say
  “M2 runtime internals complete”; they must not say “public M2 complete.”

### M2.5 — metrics, stress, soak, and benchmarks

- **M2C-FR29 — Minimum metrics.** A deterministic snapshot exposes, using
  stable edge/source/node/sink IDs only: input/output batches, rows and bytes;
  task and delivery errors; operator processing duration; sink write
  duration; blocked-send count/duration; current and high-water edge charges;
  source poll count and latest sequence/end state; sink delivered batch/row/
  byte counts; job terminal state and reaper joins. No batch ID, cursor
  payload, row payload, secret value, or unbounded label is accepted.
- **M2C-FR30 — Counter rules.** Counters are monotone, checked for overflow,
  and updated at one documented observation boundary. Per-edge input means the
  atomic successful enqueue/queue commit; per-edge output means successful
  dequeue together with reservation release. These boundaries apply equally
  to internal and synthesized source/sink boundary edges. Operator input is
  recorded after its dequeue; operator output only after every fan-out send
  for that `emit` succeeds; per-sink delivery only after that sink's write
  succeeds. Thus partial fan-out may show one edge enqueue while the operator
  fully-fanned-out output remains zero. Queue gauges are consistent snapshots
  of the channel's owned reservation.
- **M2C-FR30B — Slot observability.** The private M2 job status and soak
  stream report each stable edge's `message_slot_limit`, current
  `queue_depth`, `high_water_depth`, current/high-water rows and bytes, and
  blocked-send count/duration from one consistent snapshot. `queue_depth`
  equals charged messages. Both depth gauges are always at most
  `message_slot_limit`, including during sustained control or empty-data
  traffic. A zero-cost send blocked only by slots increments blocked-send
  accounting at the same boundaries as a row/byte-blocked send. Payloads,
  cursor values, and unbounded labels remain forbidden.
- **M2C-FR30A — Non-recursive error accounting.** The convergence driver
  atomically selects terminal cause and primary error before attempting error
  metrics; metrics can never replace that primary. It attempts each associated
  error-counter increment at most once in stable component order. If one
  overflows, the unchanged `u64::MAX` counter produces at most one
  metrics-overflow secondary for the entire outcome and sets a bounded,
  non-failing `metrics_overflowed = true` flag. That secondary is never fed
  back into error accounting. When an ordinary metrics record operation is
  itself the original failing operation, its overflow may be the primary.
  All other counters remain checked, never wrap or silently saturate, and an
  overflow flag update cannot fail or recurse.
- **M2C-FR31 — Short CI stress.** A paused-Tokio-time stress exercises two
  finite sources, union, unary operator, fan-out, slow sinks, cancellation,
  and EOF repeatedly. It asserts message slots, rows, and bytes never exceed
  their limits, per-source FIFO, no loss/duplication in a fault-free run, and
  empty supervisor/reaper registries. Seeds `0..100` feed a pure deterministic
  seed-to-gate generator (a small test-local fixed algorithm, not executor
  randomness) that produces a permutation of named source-ready, edge-release,
  sink-release, EOF, drain, and cancel gates. The harness releases exactly that
  sequence, advances paused time only for modeled timers, and prints the seed
  on failure. A separate paused-time unary phase sends at least ten times the
  slot limit for each of repeated `Idle`, repeated/non-regressing watermark,
  and verified zero-row/zero-byte `Data` while a receiver is gated. Each case
  must hit the exact slot limit, block without increasing depth, resume one
  send per dequeue, preserve FIFO, and converge under graceful shutdown,
  explicit cancel, receiver close, and injected upstream task error. It never
  assumes Tokio provides a seeded scheduler and never uses wall-clock
  performance assertions.
- **M2C-FR32 — Universal 20-minute soak.** The real-time two-source slow-sink
  soak lives in the crate-private `#[cfg(test)]` module
  `runtime::streaming::soak`; this architecture adds no public seam. The test
  is named `twenty_minute_two_source_slow_sink`, marked `#[ignore]`, guarded by
  `CALC_FLOW_STREAM_SOAK=1`, and never runs in ordinary `cargo test`. Its exact
  and superseding invocation is
  `CALC_FLOW_STREAM_SOAK=1 cargo test -p calc-flow --lib runtime::streaming::soak::twenty_minute_two_source_slow_sink -- --ignored --exact --nocapture`.
  A duration override, a `one_hour_*` target, and the former
  `--test stream_soak` invocation are not valid handoff evidence. When the
  environment variable is absent or differs from `1`, the first branch emits
  a structured disabled result before constructing a runner, opening a
  connector, spawning runtime work, or entering the timed loop.

  The enabled test exercises the real crate-private `ContinuousRunner`, source
  tasks, union/operator tasks, two ordered ordinary sinks, supervisor, and
  reaper for a 20-minute observation window. Its normal end is graceful job
  `shutdown` and must reach `Completed + GracefulShutdown` after draining;
  neither owning-job Drop nor `ExplicitCancel` may terminate the run. Job
  Drop/reaper behavior remains covered separately by AC-M2.1-C and the short
  lifecycle stress. At the drain cut the harness freezes the accepted
  per-source sequence map at the M2C-FR10A slot-commit boundary. Each sink must
  independently contain exactly that map, with identical per-source order and
  total: source-accepted = sink-one = sink-two, `missing = 0`, and
  `duplicate = 0`. Throughout the timed workload, at least every fourth data
  event is a harness-verified zero-row/zero-byte batch. The other events remain
  positive-row data. Both shapes share the same accepted-sequence oracle, so
  conservation covers empty batches rather than filtering them out.

  On Linux the soak samples `VmRSS` from `/proc/self/status` every ten seconds
  for exactly 120 samples during an exactly 1,200-second measured workload
  window. Sample one is taken at the end of the first ten-second interval and
  sample 120 at the workload deadline; there is no extra time-zero sample. It
  reports commit, kernel, `rustc`, allocator, RSS source, cadence, target
  duration, warm-up duration, and sample count in machine-readable output. The
  first five minutes, 30 samples, are warm-up.
  Each sample reports the M2C-FR30B slot fields as well as row/byte charges.
  Over the remaining 15 minutes it fails on any queue-budget breach,
  supervisor task-count growth after steady state, or a process-RSS
  least-squares slope above 1 MiB/hour together with a final five-minute
  median more than 8 MiB above the first post-warm-up five-minute median.
  Conservation is checked after graceful drain. Task-convergence evidence
  must record a positive steady task count during the run, final job task
  count zero, final queue depth/row/byte charge zero on every edge, and runner
  live/reaper registry counts `(0, 0)` after runner shutdown. At least one
  edge must report `high_water_depth == message_slot_limit` and a blocked send
  while processing the zero-cost batch shape; otherwise the soak did not
  exercise slot backpressure and fails.

  Non-Linux execution emits a structured `unsupported_platform` skip and is
  not passing evidence. Delivery requires a real captured result from the
  exact command at the synchronized final Linux head; an unrun command, a
  disabled result, or a skip fails AC-M2.5-C. The RSS thresholds are
  soak-only provisional guards, not a portable promise.
- **M2C-FR32A — Durable exact-head soak evidence.** `cf-tester` captures the
  exact command's complete, unfiltered combined stdout/stderr as one UTF-8
  command log; the engine metadata, sample, and result records embedded in
  that log are newline-delimited JSON. The tester records a manifest with
  schema `calc-flow.m2-soak-evidence.v1`, PR number, exact
  40-character head SHA, literal command, UTC start/end timestamps, elapsed
  seconds, process exit status, complete-log SHA-256, and part count. Without
  committing any log to the branch, the tester publishes the manifest and
  complete log to PR comments marked
  `calc-flow-m2-soak-evidence:v1`; if one comment would exceed 60,000
  characters, newline-aligned parts are numbered `1/N` through `N/N`, each
  with its own SHA-256, and the manifest lists their comment URLs in order.
  A summary, selected samples, local pathname, or external mutable link is not
  evidence. `cf-reviewer` retrieves the comments, reassembles and hashes the
  literal log, confirms exit status zero, exactly one metadata record, exactly
  120 ordered sample records over the 1,200-second workload window, exactly
  one final result record, all
  M2C-FR30B/M2C-FR32 gates, and equality of manifest SHA, log commit, and the
  then-current PR head. Any later push invalidates the bundle and requires a
  new run and new SHA-keyed comments; publishing comments never changes the
  tested Git head.
- **M2C-FR33 — Benchmarks.** Criterion adds channel data round trip,
  unary stream overhead, and fan-out cases. Ordinary verification only
  compiles them with `cargo test -p calc-flow --benches --no-run`. The opt-in
  baseline command is `cargo bench -p calc-flow --bench core -- stream` and
  produces the Criterion report. Because Criterion is an external target and
  this gate is crate-private, the unary case measures the already-public M1
  `StreamOperator` kernel plus public bounded-channel primitives; it does not
  expose an internal task or fake runner solely for measurement. M2 requires
  compilation, not an executed paired comparison. If a performance reviewer
  invokes the 5% gate, the exact base and head run on the same machine,
  toolchain, profile, allocator, and power/CPU policy. For each required case,
  the decision statistic is Criterion's bootstrap 95% confidence interval for
  relative change in mean execution time, head versus base. A case blocks only
  when the interval's lower bound is strictly greater than `+5.0%`; it passes
  when the upper bound is at most `+5.0%`. An interval touching or crossing
  `+5.0%` is inconclusive and requires one same-machine rerun with at least
  twice the original measurement sample size. If that rerun remains
  inconclusive, it is recorded as a non-blocking advisory, not reclassified
  from its point estimate. Any blocking case blocks the comparison; otherwise
  the comparison passes, with any residual advisory reported. When no paired
  comparison is invoked, all point estimates remain advisory and benchmark
  compilation is the only M2 performance gate.

## Non-functional requirements

- **M2C-NFR1 — Performance.** The channel, unary-stream, and fan-out
  benchmarks compile at the exact head. An invoked paired comparison follows
  M2C-FR33's single 95%-interval rule and one-rerun limit; no alternate point
  estimate, overlap heuristic, or aggregate average may override it.
- **M2C-NFR2 — State and checkpoints.** M2 persists no source progress,
  operator state, watermark, epoch, or sink transaction. Snapshot, restore,
  reset, checkpoint-v2, and durable-cursor behavior outside fresh operator
  entry remain unchanged; M4/M5 own new durability semantics.
- **M2C-NFR3 — Compatibility.** Relative to `origin/main`, preserve the
  PR-added semver-compatible public non-exhaustive
  `CalcFlowError::TaskPanicked { task_id, message }` as the sole public Rust
  API addition. Its fields and display `task <task_id> panicked: <message>`
  stay fixed; M2C-FR1A bounds only internal message capture. No other public
  variant, export, callable, runner, configuration field, or control
  constructor is added. Existing public Rust v2 runners, Python API/stub and
  wildcard error fallback, project/checkpoint JSON, Studio `/api/v2`, OpenAPI,
  and generated TypeScript contracts otherwise do not change.
- **M2C-NFR4 — Universal soak duration.** Twenty minutes is the only
  normative calc-flow soak duration for this run and future soak gates unless
  a later separately reviewed specification replaces it. A soak uses a
  20-minute measured workload window; setup and graceful teardown may make
  total command wall time longer. On Linux that window contains exactly 120
  ten-second RSS samples under M2C-FR32. One-hour names, defaults, commands,
  and acceptance wording are obsolete and must not remain normative.
- **M2C-NFR5 — Quality and final-head evidence.** Completion requires no new
  Codacy issue of any severity, no unresolved review thread, all required
  Actions green, clean mergeability, and the real Linux soak evidence at the
  same pushed head after incorporating current `origin/main`. The soak is
  evidence only through M2C-FR32A's reviewer-verified PR-comment bundle.
  Quality waivers, exclusions, threshold reductions, stale-head results, or a
  prose soak summary do not satisfy this requirement.

## Public compatibility and migration

- The baseline crate is still version 2.0 and exposes a functioning push
  `StreamingRunner` and `MicroBatchRunner`. Removing them now would be a net
  capability deletion because M2 has neither v3 checkpoint manifests nor a
  bounded-source replacement with durable cursor recovery.
- Therefore this delta requires the remaining M2 implementation to coexist
  crate-privately with the v2 public runners. No alias, deprecation shim, or
  second public `StreamingRunner` is introduced.
- Relative to `origin/main`, the already-PR-added non-exhaustive
  `CalcFlowError::TaskPanicked` variant is the sole public Rust API addition
  and must be preserved. M2 adds no other public variant, export, callable,
  configuration field, control constructor, or runner surface. Enforcing a
  slot for every message strengthens the existing bounded-channel semantics
  without changing the public `EdgeBudget` shape.
- This is a deliberate plan-ledger supersession: the original M2.4 public cut
  is not checked off by this gate. It is replaced by “M2 internal runtime
  completion” here and a distinct post-M5 public A6 integration gate.
- The eventual breaking public replacement remains the total API note's A6
  surface. Its deletion/replacement must land atomically with:
  source-driven start/status/shutdown/cancel/wait; the state/checkpoint
  behavior promised by A6; migration of Rust tests/examples/benches; removal
  of v2 `step`; bounded-source replacement for `MicroBatchRunner`; and explicit
  `CHANGELOG.md` plus migration-guide entries.
- Public `Batch`, `BatchExecutionPlan`, v2 runner/checkpoint behavior, Python,
  Studio, schema, and OpenAPI behavior must otherwise remain unchanged by the
  crate-private M2 internal-completion slice.
- No status or documentation may report public M2 completion, exactly-once,
  checkpoint recovery, event-time progress, state recovery, or cross-process
  at-least-once as available.

## Inputs and outputs

| Boundary                  | Input                                                                | Output / terminal behavior                                                                   |
| ------------------------- | -------------------------------------------------------------------- | -------------------------------------------------------------------------------------------- |
| Source preflight          | Compiled source slots, all bindings, all first-hop budgets           | Validated immutable wiring, or one deterministic error before any open                       |
| Provisional launch        | Validated owned job plus registered core-owned launch driver         | Entry acks, connector open, delivered job handle or cancel/close/join/reap                   |
| Edge channel              | Positive `EdgeBudget`, one data/control envelope, current charges    | FIFO commit within message-slot, row, and byte limits, or a parked/failed sender             |
| Source task               | `SourceEvent`, binding ID, resume cursor, next sequence              | FIFO `StreamMessage` fan-out plus volatile source progress                                   |
| Operator task             | Named ingress receivers and one owned `StreamOperator`               | Validated data fan-out and runtime-owned supported controls                                  |
| Ordinary sink task        | One external-output edge and ordered sink bindings                   | Sequential writes, close result, delivered progress                                          |
| Job supervisor            | Registered named futures and terminal triggers                       | One immutable terminal outcome and an empty registry                                         |
| Job status/metrics        | Runtime-owned counters, gauges, progress, and registry snapshots     | Deterministically ordered, payload-free point-in-time snapshot                               |
| M2 public compatibility   | Existing v2 public runner/checkpoint surfaces                        | No behavior or signature change until the A6 public gate is resolved                         |

## Acceptance criteria

### Focused behavior tests

- [ ] **AC-M2.1-A:** simultaneous task failures remain stably ordered over
  100 deterministic repetitions; a convergence error never replaces the
  primary failure; panic includes stable task identity.
- [ ] **AC-M2.1-A2:** an ASCII panic longer than 1,024 bytes and a multibyte
  panic whose 1,024-byte cut would split a scalar both produce a valid UTF-8
  message no longer than 1,024 bytes, ending in `…`; a short string is
  unchanged and a non-string payload is exactly `non-string panic payload`.
- [ ] **AC-M2.1-B:** success, failure, explicit cancel, deadline, natural
  drain, explicit shutdown, dropped wait future, dropped job handle, next
  start, and runner shutdown each reach the specified observable registry and
  terminal outcome without sleeps.
- [ ] **AC-M2.1-C:** dropping a job handle transfers a non-empty registry to
  the reaper; the next start or runner shutdown joins it and makes both
  registries empty. Dropping a runner without shutdown fires cancellation and
  records the sanctioned abandon-joins metric/log event.
- [ ] **AC-M2.1-D (B1):** all operator tasks are registered before entry; a
  fresh job calls `reset()` inside each task. A gated external operator whose
  `reset()` fails produces the primary error, joins every entry task, and
  leaves source-open, source-close, sink-open, and sink-close canaries all
  untouched. Successful entry acks still cannot process data until every
  connector open ack releases the separate data/control gate.
- [ ] **AC-M2.1-E (B3):** adversarial manual polling drops `wait`, `cancel`,
  and `shutdown` observers at every driver/join await. The core-owned driver
  retains the registry, a later observer completes it, concurrent terminal
  calls return the same immutable outcome, and owning-handle Drop plus runner
  shutdown cannot race ownership into an empty or lost registry.
- [ ] **AC-M2.1-F (B5):** drop the `start()` observer (a) after provisional
  registration, (b) during operator entry, (c) after one source opens while a
  second open is gated, (d) after all opens but before data-gate release, and
  (e) after the driver publishes a live job but before handle delivery. Every
  case requests cancellation, closes each begun connector once, joins/reaps
  the launch, and releases no unowned live job; next start waits for cleanup
  then succeeds, while runner shutdown performs the same cleanup.
- [ ] **AC-M2.2-A:** missing, unknown, duplicate, zero-bound, and oversize
  source bindings fail before a mock source's `open` flag changes. With two
  valid bindings, both reopen at their configured resume cursor.
- [ ] **AC-M2.2-B:** two sources with capacity-one prefetch and a full edge
  stop polling after the allowed one-item slot; cancellation drops a blocked
  `next`, calls `close`, never polls again, and joins pump and source task.
- [ ] **AC-M2.2-B2 (B2):** synchronization gates place the drain cut before
  poll, during pending poll, after external readiness but before runtime poll,
  after `Ready` observation but before slot commit, after slot commit, after
  slot dequeue with an edge send pending, after edge enqueue, and around a
  returned `None`. Only events whose slot commit linearized first are drained;
  every other poll is dropped once, never resumed, followed by one runtime
  EOF and `Closed`. Matching cancel cases may discard the slot.
- [ ] **AC-M2.2-C:** two fan-out branches receive identical fault-free FIFO
  sequences with authoritative metadata, strictly increasing cursors and
  sequences, preserved attributes, one EOF, and independent queue charges.
- [ ] **AC-M2.2-D (S2):** close branch two after branch one accepts a source
  event. The job fails without rollback; branch one retains exactly that
  prefix, branch two does not, edge-one enqueue metrics are one, and the
  source fully-fanned-out count remains zero. A pre-send validation failure
  instead leaves every branch and edge enqueue count at zero.
- [ ] **AC-M2.2-E:** for binding `orders`, repeated/regressed cursor emits
  exact field `sources.orders.cursor`, and regressed source watermark emits
  exact field `sources.orders.watermark`; both failures occur before the
  offending event reaches any fan-out branch.
- [ ] **AC-M2.3-A:** unary expression and SQL process multiple batches in
  ingress order; invalid output port/kind/schema fails before any successor
  observes it; array-only and union-only graphs create zero DataFusion
  runtimes, while each table node creates exactly one lazily.
- [ ] **AC-M2.3-B:** every readiness interleaving of a two-input union
  preserves each ingress FIFO and outputs all data, without asserting a
  cross-ingress order. Both ended inputs cause one `on_end` and one output
  EOF.
- [ ] **AC-M2.3-C:** unary watermark forwarding occurs only after successful
  `on_watermark`; a handler error forwards none; barrier injection is rejected
  in the M2 path. Multi-ingress watermark and idle both fail before handler or
  output; unary idle remains FIFO.
- [ ] **AC-M2.3-D (S4):** drop a producer without sending explicit EOF for a
  unary operator, one union ingress, and a sink ingress. While running or
  draining each yields `EdgeClosed` and never natural completion. Repeat after
  runtime convergence-close marking: `None` is a wakeup; an independently
  observed earlier close remains only a stable secondary.
- [ ] **AC-M2.3-E (S2):** an external operator completes one data `emit` and
  then returns an error from the same handler. The emitted per-edge prefix and
  enqueue metrics remain observable, no runtime-owned control for that event
  is forwarded, the handler error is primary, and convergence claims no
  rollback or cross-branch atomicity.
- [ ] **AC-M2.4-A:** all source and sink route errors are observed before any
  mock source/sink open flag changes. One output with three sinks observes
  each batch in sink order, and a failure at sink two prevents sink three from
  observing that batch.
- [ ] **AC-M2.4-B:** natural EOF and graceful shutdown drain exactly the
  M2C-FR10A slot-committed prefix and
  close every sink; explicit cancel makes no drain assertion but closes and
  joins every task; repeated terminal calls return the same outcome. For a
  slot-full edge, graceful shutdown drains the queued controls/empty batches,
  an already-started blocked send, and ordered EOF. Cancel, receiver close,
  and upstream task error each wake the blocked sender and leave current
  slot/row/byte charges zero after convergence.
- [ ] **AC-M2.4-C:** in-memory source -> union -> expression -> recording sink
  completes end to end with two sources, preserves per-source FIFO, applies
  slow-sink backpressure to both, and leaves all queues/tasks empty.
- [ ] **AC-M2.5-A:** metrics snapshot keys are deterministic, counters follow
  M2C-FR30/M2C-FR30B, no payload or secret canary appears, and current/high-
  water message depth, rows, and bytes never exceed configured capacity.
- [ ] **AC-M2.5-A2 (B4):** preset the task-error counter to `u64::MAX`, then
  submit an operator failure. The operator error remains primary, error
  accounting is attempted once, exactly one metrics-overflow secondary is
  attached, `metrics_overflowed` becomes true, and outcome publication
  terminates without recursion. A direct ordinary-counter overflow is primary
  when it was the original operation; all counters remain non-wrapping.
- [ ] **AC-M2.5-A3 (S6):** one successful enqueue increments edge input; its
  dequeue/release increments edge output. In a branch-two-close partial
  fan-out, edge one reports one input, edge two zero, and operator
  fully-fanned-out output zero. The same boundaries hold on synthesized source
  and sink boundary edges.
- [ ] **AC-M2.5-A4 (B6):** with `EdgeBudget::new(3, sufficient_bytes)`, a
  table-driven stalled-receiver test separately sends repeated `Idle`, equal
  or increasing watermarks, and a data batch first proven to charge one
  message, zero rows, and zero bytes. In each case three messages enqueue,
  `queue_depth == high_water_depth == message_slot_limit == 3`, row/byte
  charge remains zero, and polling the fourth send is pending with depth still
  three. One dequeue completes exactly one blocked send in FIFO order. Cycling
  at least 30 messages never exceeds three, records slot-only blocked sends,
  and ends with every current charge zero.
- [ ] **AC-M2.5-A5 (B6 lifecycle):** a real unary source/operator edge emits
  sustained `Idle`, non-regressing watermark, and empty zero-cost data traffic
  while its receiver is gated. Once the edge reaches its slot limit, source
  polling stops: the connector's recorded `next` count may increase only for
  the one poll already in flight, at most one item may occupy the D3 prefetch
  slot, and the settled count then remains fixed until dequeue or a terminal
  trigger. Separate gated cases prove graceful drain plus EOF, cancellation,
  receiver close, and upstream task error follow M2C-FR24B, preserve the
  required primary/secondary error, join every task, and leave queue/
  supervisor/reaper state empty.
- [ ] **AC-M2.5-B:** paused-time stress passes its boundedness, ordering,
  lifecycle, and leak assertions under at least 100 seeded schedules, and its
  zero-cost phase exercises at least ten slot-limit multiples of each required
  control/data shape without any slot, row, or byte breach.
- [ ] **AC-M2.5-C:** the crate-private soak is discoverable but ignored by
  ordinary test runs and exposes no public seam. Without
  `CALC_FLOW_STREAM_SOAK=1`, its first-branch guard performs no runtime launch
  or timed loop. At the synchronized final Linux head, the exact M2C-FR32
  command runs an exactly 1,200-second timed workload and takes exactly 120 RSS
  samples, ends through graceful drain with `Completed + GracefulShutdown`,
  proves
  equality of the accepted source map and both sink maps/totals with zero
  missing and duplicates, proves final task/queue/reaper convergence, and
  passes the RSS guard. At least every fourth accepted event is verified
  zero-row/zero-byte data; at least one edge reaches but never exceeds its
  message-slot limit and records a blocked send. Captured machine-readable
  output is mandatory; “not run,” disabled, skipped, or stale-head output
  fails this criterion.
  Criterion contains and compiles all three M2 cases.
- [ ] **AC-M2.5-C2 (S9):** the exact-head soak has one complete
  `calc-flow.m2-soak-evidence.v1` PR-comment bundle. Reassembly matches the
  manifest and per-part SHA-256 values; exit status is zero; start/end and
  sample timestamps prove the 1,200-second workload; exactly 120 ordered
  samples plus one metadata and one final result record are present; the
  manifest, log, and current PR head SHA are identical. Missing, truncated,
  edit-only summaries or evidence for an earlier head fail.
- [ ] **AC-PERF (S11):** benchmark compilation passes. If the paired gate is
  invoked, raw Criterion reports demonstrate M2C-FR33's 95%-relative-mean
  decision for every required case, including the one permitted doubled-
  sample rerun for each inconclusive interval. Any lower bound above `+5.0%`
  blocks; a still-crossing interval after the rerun is explicitly advisory and
  does not become blocking from its point estimate.
- [ ] **AC-COMPAT:** existing v2 streaming and micro-batch integration tests
  plus Python runner/exception-adapter tests remain unchanged and green unless
  a separately approved API-note revision authorizes the atomic public
  replacement. `TaskPanicked` remains the sole public Rust addition and maps
  through Python's existing wildcard fallback without adding a Python class.
- [ ] **AC-DOCS:** `docs/runtime-envelope.md` accurately describes whole-job
  preflight, private operator/sink/runner/reaper work, M2 metrics/stress/soak,
  and post-M5 public A6 deferral. Every normative calc-flow soak statement,
  test name, command, duration default, and sample count is consistent with
  M2C-NFR4; historical supersession notes may mention the former duration but
  cannot prescribe it.
- [ ] **AC-QUALITY:** Codacy reports zero new issues at the exact head,
  including closure of the audited 16 medium complexity findings, without a
  waiver, exclusion, threshold reduction, or analyzer-configuration change.
- [ ] **AC-REVIEW:** GitHub reports zero unresolved review threads, including
  the binding-qualified cursor, binding-qualified watermark, and bounded
  UTF-8 panic-payload threads.
- [ ] **AC-FINAL-HEAD:** the branch contains current `origin/main`; all
  required Actions and Codacy checks pass for the pushed exact head; the head
  is mergeable/clean. Results from `574c0fb` or any other earlier head are not
  completion evidence.
- [ ] **AC-VERIFY (S10):** every command group in the current root
  `AGENTS.md` passes at the exact final head, including Rust/PyO3, Python,
  Studio backend/frontend/browser, supply-chain, helper, coverage, rustdoc,
  generated-contract, and diff checks. The handoff maps each exact-head Action
  check to those groups; the focused commands below are not a waiver.

### Verification commands

The handoff must map every behavioral criterion above to a named test. Every
current command group in root `AGENTS.md` is mandatory for final completion.
The subset below is for focused RED/GREEN iteration and does not replace that
full matrix:

```bash
cargo fmt --all --check
cargo test -p calc-flow --lib runtime::streaming
cargo test -p calc-flow --test stream_operator --test stream_compile \
  --test streaming --test micro_batch
uv run python scripts/run_rust_tests.py
cargo llvm-cov --workspace --all-features --fail-under-lines 90
cargo test -p calc-flow --benches --no-run
cargo clippy --workspace --all-targets --all-features -- -D warnings
RUSTDOCFLAGS="-D warnings" cargo doc --workspace --all-features --no-deps
cargo audit --ignore RUSTSEC-2026-0176 --ignore RUSTSEC-2026-0177
cargo deny --locked check
git diff --exit-code -- \
  schemas/project-v2.schema.json \
  web-ui/openapi.json \
  web-ui/src/api/schema.d.ts
git diff --check
```

The Linux soak is a required external/manual gate outside ordinary CI. The
Criterion run is optional unless M2C-NFR1's paired comparison is invoked:

```bash
CALC_FLOW_STREAM_SOAK=1 cargo test -p calc-flow --lib runtime::streaming::soak::twenty_minute_two_source_slow_sink -- --ignored --exact --nocapture
cargo bench -p calc-flow --bench core -- stream
```

This specification revision is not evidence that the soak ran. Its complete
M2C-FR32A PR-comment bundle must record the Linux RSS/boundedness series,
graceful-drain conservation oracle, task/queue/reaper convergence, and machine
metadata. A local-only log or non-Linux structured skip is not a passing
artifact.
Criterion point estimates and confidence intervals are reported separately;
a performance comparison is a delivery gate only under M2C-FR33's paired-run
rule.

## Expected file scope

The current PR already owns the broad implementation files. Remaining edits
must be the narrowest changes that close G1-G8; production edits outside the
current PR's Rust/runtime/docs/benchmark surface require an explicit spec
delta.

- Modify `crates/calc-flow/src/runtime/streaming/channel.rs` so every data or
  control envelope consumes one finite slot under M2C-FR24A/M2C-FR24B, and add
  the focused zero-cost admission and lifecycle tests. Reuse
  `EdgeBudget.max_rows` as the independent slot ceiling; do not change the
  public `EdgeBudget` shape.
- Modify `crates/calc-flow/src/runtime/streaming/metrics.rs` and the private
  runner/status projection as needed to expose the bounded slot gauges and
  slot-only blocking observations required by M2C-FR30B.
- Modify `crates/calc-flow/src/runtime/streaming/source_task.rs` for exact
  binding-qualified cursor/watermark diagnostics and focused tests.
- Modify `crates/calc-flow/src/runtime/streaming/supervisor.rs` for the bounded
  UTF-8 panic payload and focused tests; reuse the same helper for open/task
  panic capture.
- Modify `crates/calc-flow/src/runtime/streaming/soak.rs` for the universal
  20-minute name, exact 1,200-second duration, exact 120-sample/30-sample
  warm-up defaults, zero-cost slot pressure, structured result, and
  task/queue/reaper convergence evidence.
- Simplify the current M2 runtime modules and tests as needed to clear all 16
  Codacy complexity findings without changing M2 behavior. No lint, Codacy,
  coverage, or analyzer waiver/configuration edit is in scope.
- Preserve the current PR's public non-exhaustive
  `CalcFlowError::TaskPanicked { task_id, message }` variant as the sole public
  Rust API addition. Do not add another public error variant, export,
  configuration field, control constructor, or runner surface.
- Keep focused tests beside the private modules or in existing Rust integration
  targets. Keep the ignored 20-minute harness in the crate-private `#[cfg(test)]`
  `runtime::streaming::soak` module so it exercises the private runtime without
  creating a public test seam.
- Reconcile `docs/runtime-envelope.md`, the original plan, the total runtime
  specification, and the completion API note/critique where they prescribe
  an obsolete one-hour soak or describe implemented M2 internals as absent.
  No document may claim public A6/checkpoint/event-time availability.
- Do not modify Python, Studio, schemas, project/checkpoint v2 fixtures, or
  `tests/fixtures/v1/`.

## Delivery sequence and gates

1. Add RED tests for G8's slot admission, zero-cost traffic, and terminal
   lifecycle criteria, and record the expected current-head failures.
2. Enforce three-dimensional slot/row/byte admission on every edge, add slot
   observability, and make the stress/soak exercise zero-cost slot pressure.
3. Add RED tests for G1-G3 and record the expected current-head failures.
4. Fix binding-qualified diagnostics and bounded panic capture, then run their
   focused tests.
5. Convert the soak seam and every governed normative reference to the
   universal 20-minute standard; run its short helper/smoke tests.
6. Remove the 16 Codacy complexity findings behavior-preservingly, keeping the
   existing named M2 tests green and adding no waiver.
7. Reconcile runtime documentation with the current implementation while
   preserving the post-M5 public A6 boundary.
8. Incorporate current `origin/main`; run the full root-`AGENTS.md` matrix and
   push once the intended diff is reviewed.
9. Run and capture the exact 20-minute/120-sample Linux soak at that pushed
   head, publish the M2C-FR32A SHA-keyed PR-comment bundle, and have the
   reviewer independently reassemble and verify it. Then require zero
   unresolved threads, green exact-head Actions and Codacy, and clean
   mergeability.

Every behavior change follows RED -> observed expected failure -> focused
GREEN -> affected matrix -> reviewer. A code change cannot be considered
complete without `cf-reviewer`; documentation disposition belongs to
`cf-doc-writer`. None of these steps performs the public A6 cut.

## Resolved milestone decisions

- The completion API note's Decision 1 is adopted: total A6 plus M4/M5
  supersedes the plan's original public M2.4 cut. This gate is internal only;
  public A6 integration is post-M5.
- Decision 2 is adopted: unary watermark/idle passes through the
  runtime-owned path; multi-ingress watermark/idle fails before handler or
  output until M3; barrier fails before output until M5.
- Decision 3 is adopted: ordinary sinks provide process-local ordered
  delivery only, not a cross-process at-least-once guarantee.
- Job deadline is frozen as `Cancelled + DeadlineExceeded`, not an open
  question and not `CheckpointTimeout`.
- Launch uses the three phases and cancel-on-dropped-start-observer semantics
  of M2C-FR7B/M2C-FR7C. Accepted drain prefix uses only the atomic slot commit
  of M2C-FR10A. Fan-out remains observably non-transactional on failure.
- All calc-flow soak gates now use the universal 20-minute measured workload
  window. Earlier one-hour defaults are obsolete; this is a closed decision,
  not a performance target or an optional shorter smoke.
- Source monotonicity diagnostics are binding-qualified with the exact fields
  in M2C-FR11/M2C-FR14. Panic text uses the 1,024-byte UTF-8-safe bound in
  M2C-FR1A without changing the public non-exhaustive error shape.
- Every data and control envelope consumes one slot. The hard per-edge slot
  ceiling is the existing positive `EdgeBudget.max_rows`, independently of
  the row ceiling; M2 adds no `max_messages` field or public configuration.
- Exact-head soak evidence is the complete SHA-verified command log and
  manifest published as the M2C-FR32A PR-comment bundle. Local summaries and
  earlier-head logs are not delivery evidence.
- Benchmark compilation is mandatory. A paired performance comparison is
  optional unless explicitly invoked; when invoked, M2C-FR33's Criterion
  bootstrap 95% relative-mean interval and one doubled-sample rerun are the
  only decision rule.
- The completion API note owns the exact core/driver/reaper and runtime-plan
  projection signatures. This spec owns their required observable outcomes.

## Open questions

- None for M2 behavior or public scope. G1-G8 are closure obligations, not
  permission to redesign the API or defer the 20-minute evidence.

## Handoff

Next role: `cf-critic` to review revision 5's B6 slot invariant, durable soak
evidence, paired-performance decision, exact diagnostic fields, panic bound,
and G1-G8 gates. After a zero-block verdict, `cf-implementer` closes
G1-G5/G8, `cf-tester` maps and runs the behavioral and exact
20-minute/120-sample evidence, and `cf-reviewer` enforces G6-G7 plus independent
evidence verification. No agent in this chain may expose or replace the public
v2 runner before the separately reviewed post-M5 A6 integration gate.
