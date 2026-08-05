# Continuous Streaming M2 Internal Runtime Completion - Delta Specification

## Source and status

- Request: Cheng Li, 2026-08-05 — determine whether M2 is complete and, if
  not, finish the remaining M2 work.
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
- Code baseline: commit `1d5546028e2ce9ebce59c976080c3d11c1225e16`
  on `feature/streaming-v3-m2-runtime-skeleton` (PR #83).
- Status: proposed **M2 internal runtime completion** delta. It is not the
  plan's original public M2.4 cut and must not be reported as public M2
  completion. Revision 2 addresses B1, B2, B4, B5, S1, S2, S4, S6, S7, and
  M1 from
  [`../critiques/continuous-streaming-m2-completion.md`](../critiques/continuous-streaming-m2-completion.md).
  Revision 3 accepts a crate-private real-runtime soak module and freezes its
  opt-in command, graceful-drain termination, and end-to-end conservation
  oracle; it does not claim that the one-hour gate has been run.
  This document does not repeat or
  amend the frozen D1-D9, S1-S10, I1-I10, NG1-NG13, or A1-A8 decisions. A
  requirement below narrows work to M2; the controlling artifacts win if a
  sentence can be read more broadly.

## Problem statement

PR #83 proves a crate-private source-to-bounded-edge-to-consumer vertical
slice, but it does not complete the plan's M2 milestone. Operator tasks, sink
tasks, graph-wide binding validation, a public continuous job lifecycle,
reaper ownership, M2 metrics, stress, soak, and benchmarks are absent. The
public A6 lifecycle also requires checkpoint and state capabilities assigned
to M4/M5. The total A6 contract therefore supersedes the plan's original
public M2.4 cut: this gate completes internal runtime infrastructure only, and
the public cut moves to the post-M5 A6 integration gate.

## Baseline inventory

The following ledger is normative for this delta: items marked implemented
must be preserved; items marked remaining are in scope here; deferred items
must not be pulled into M2.

| Task | Baseline implemented                  | Remaining                                               |
| ---- | ------------------------------------- | ------------------------------------------------------- |
| M2.1 | Context plus structured supervisor    | Full lifecycle, idempotency, Drop, and reaper           |
| M2.2 | D3 source pump/task plus source rules | Whole-job preflight and multi-source integration        |
| M2.3 | Operator traits and built-in kernels  | Operator task, collector, graph wiring, DataFusion life |
| M2.4 | Minimal consuming continuous job      | Sink task, drain/cancel, reusable job and runner        |
| M2.5 | Per-edge channel metrics              | Runtime metrics, stress, opt-in soak, and benchmarks    |

The current public `StreamingRunner` remains the v2 push-based runner, and the
current `MicroBatchRunner`, v2 `Source`, v2 `Sink`, and v2 `CheckpointStore`
still exist. No M2 test may describe the crate-private skeleton as the public
source-driven A6 runner.

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
  through deterministic minimum metrics and status snapshots.
- Add deterministic short stress coverage, an explicitly opt-in soak, and
  compilable Criterion cases without making a one-hour run part of ordinary
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
  overflowing item is observed downstream.
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
  source-provided watermark regression before enqueue, and permits only the
  runtime to construct downstream control messages. M2 does not generate
  watermarks or claim M3 minimum/reactivation semantics. A multi-ingress
  watermark or idle event fails before handler or downstream output. No
  implementation may substitute max, last-arrival, or arbitrary forwarding
  for M3's frozen minimum.

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
  and EOF repeatedly. It asserts queue charges never exceed configured rows
  or bytes, per-source FIFO, no loss/duplication in a fault-free run, and
  empty supervisor/reaper registries. Seeds `0..100` feed a pure deterministic
  seed-to-gate generator (a small test-local fixed algorithm, not executor
  randomness) that produces a permutation of named source-ready, edge-release,
  sink-release, EOF, drain, and cancel gates. The harness releases exactly that
  sequence, advances paused time only for modeled timers, and prints the seed
  on failure. It never assumes Tokio provides a seeded scheduler and never
  uses wall-clock performance assertions.
- **M2C-FR32 — One-hour soak.** The real-time two-source slow-sink soak lives
  in the crate-private `#[cfg(test)]` module `runtime::streaming::soak`; this
  test architecture is accepted and adds no public seam. The test is named
  `one_hour_two_source_slow_sink`, marked `#[ignore]`, guarded by
  `CALC_FLOW_STREAM_SOAK=1`, and never runs in ordinary `cargo test`. Its exact
  and superseding invocation is
  `CALC_FLOW_STREAM_SOAK=1 cargo test -p calc-flow --lib runtime::streaming::soak::one_hour_two_source_slow_sink -- --ignored --exact --nocapture`.
  The former `--test stream_soak` invocation is superseded and is not valid
  handoff evidence. When the environment variable is absent or differs from
  `1`, the test takes a first-branch disabled guard before constructing a
  runner, opening a connector, spawning runtime work, or entering the timed
  loop; it emits a structured disabled result and never performs the one-hour
  run.

  The enabled test exercises the real crate-private `ContinuousRunner`, source
  tasks, union/operator tasks, two ordered ordinary sinks, supervisor, and
  reaper. Its normal end is initiated through graceful job `shutdown` and must
  reach `Completed + GracefulShutdown` after draining; neither owning-job Drop
  nor `ExplicitCancel` may terminate the one-hour run. Job Drop/reaper behavior
  remains covered separately by AC-M2.1-C and the short lifecycle stress.
  At the drain cut the harness freezes the accepted per-source sequence map at
  the M2C-FR10A slot-commit boundary. Each of the two sinks must independently
  contain exactly that map, with the same per-source order and the same total:
  source-accepted = sink-one = sink-two, `missing = 0`, and `duplicate = 0`.

  On Linux the soak samples `VmRSS` from `/proc/self/status` every ten seconds,
  for at least 360 samples, and reports commit, kernel, `rustc`, allocator, RSS
  source, cadence, and sample count in machine-readable output. After a
  ten-minute warm-up, it fails on any queue-budget breach, task-count growth
  after steady state, conservation failure above, or a process-RSS
  least-squares slope above 1 MiB/hour together with a final five-minute median
  more than 8 MiB above the first post-warm-up five-minute median. On non-Linux
  it emits a structured `unsupported_platform` skip and cannot be cited as soak
  evidence. This revision does not claim that the one-hour run has been
  executed: the exact Linux command remains an external/manual delivery gate,
  and a handoff must report it as not run unless it has a real captured result.
  The RSS thresholds are soak-only provisional guards, not a portable promise.
- **M2C-FR33 — Benchmarks.** Criterion adds channel data round trip,
  unary stream overhead, and fan-out cases. Ordinary verification only
  compiles them with `cargo test -p calc-flow --benches --no-run`. The opt-in
  baseline command is `cargo bench -p calc-flow --bench core -- stream` and
  produces the Criterion report. Because Criterion is an external target and
  this gate is crate-private, the unary case measures the already-public M1
  `StreamOperator` kernel plus public bounded-channel primitives; it does not
  expose an internal task or fake runner solely for measurement. A 5%
  regression blocks delivery only when compared in a same-machine paired run
  with confidence intervals as required by NFR-1; absent that evidence,
  benchmark compilation is the M2 gate and point estimates are reported, not
  treated as correctness failures.

## Public compatibility and migration

- The baseline crate is still version 2.0 and exposes a functioning push
  `StreamingRunner` and `MicroBatchRunner`. Removing them now would be a net
  capability deletion because M2 has neither v3 checkpoint manifests nor a
  bounded-source replacement with durable cursor recovery.
- Therefore this delta requires the remaining M2 implementation to coexist
  crate-privately with the v2 public runners. No alias, deprecation shim, or
  second public `StreamingRunner` is introduced.
- This is a deliberate plan-ledger supersession: the original M2.4 public cut
  is not checked off by this gate. It is replaced by “M2 internal runtime
  completion” here and a distinct post-M5 public A6 integration gate.
- The eventual breaking public replacement remains the total API note's A6
  surface. Its deletion/replacement must land atomically with:
  source-driven start/status/shutdown/cancel/wait; the state/checkpoint
  behavior promised by A6; migration of Rust tests/examples/benches; removal
  of v2 `step`; bounded-source replacement for `MicroBatchRunner`; and explicit
  `CHANGELOG.md` plus migration-guide entries.
- Public `Batch`, `BatchExecutionPlan`, v2 checkpoint documents, Python,
  Studio, schema, and OpenAPI behavior must remain unchanged by the
  crate-private M2 internal-completion slice.
- No status or documentation may report public M2 completion, exactly-once,
  checkpoint recovery, event-time progress, state recovery, or cross-process
  at-least-once as available.

## Inputs and outputs

| Boundary                  | Input                                                                | Output / terminal behavior                                                                  |
| ------------------------- | -------------------------------------------------------------------- | ------------------------------------------------------------------------------------------- |
| Source preflight          | Compiled source slots, all bindings, all first-hop budgets           | Validated immutable wiring, or one deterministic error before any open                      |
| Provisional launch        | Validated owned job plus registered core-owned launch driver         | Entry acks, connector open, delivered job handle or cancel/close/join/reap                   |
| Source task               | `SourceEvent`, binding ID, resume cursor, next sequence              | FIFO `StreamMessage` fan-out plus volatile source progress                                  |
| Operator task             | Named ingress receivers and one owned `StreamOperator`               | Validated data fan-out and runtime-owned supported controls                                 |
| Ordinary sink task        | One external-output edge and ordered sink bindings                   | Sequential writes, close result, delivered progress                                         |
| Job supervisor            | Registered named futures and terminal triggers                       | One immutable terminal outcome and an empty registry                                        |
| Job status/metrics        | Runtime-owned counters, gauges, progress, and registry snapshots     | Deterministically ordered, payload-free point-in-time snapshot                              |
| M2 public compatibility   | Existing v2 public runner/checkpoint surfaces                        | No behavior or signature change until the A6 public gate is resolved                        |

## Acceptance criteria

### Focused behavior tests

- [ ] **AC-M2.1-A:** simultaneous task failures remain stably ordered over
  100 deterministic repetitions; a convergence error never replaces the
  primary failure; panic includes stable task identity.
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
  joins every task; repeated terminal calls return the same outcome.
- [ ] **AC-M2.4-C:** in-memory source -> union -> expression -> recording sink
  completes end to end with two sources, preserves per-source FIFO, applies
  slow-sink backpressure to both, and leaves all queues/tasks empty.
- [ ] **AC-M2.5-A:** metrics snapshot keys are deterministic, counters follow
  M2C-FR30, no payload or secret canary appears, and current/high-water charges
  never exceed configured capacity.
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
- [ ] **AC-M2.5-B:** paused-time stress passes its boundedness, ordering,
  lifecycle, and leak assertions under at least 100 seeded schedules.
- [ ] **AC-M2.5-C:** the crate-private soak is discoverable but ignored by
  ordinary test runs and exposes no public seam. Without
  `CALC_FLOW_STREAM_SOAK=1`, its first-branch guard performs no runtime launch
  or timed loop. The exact M2C-FR32 opt-in command ends through graceful drain,
  returns `Completed + GracefulShutdown`, proves equality of the accepted
  source map and both sink maps/totals with zero missing and duplicates, and
  produces the required Linux samples. It remains an external/manual gate and
  must be reported as not run until actually executed. Criterion contains and
  compiles all three M2 cases.
- [ ] **AC-COMPAT:** existing v2 streaming and micro-batch integration tests
  remain unchanged and green unless a separately approved API-note revision
  authorizes the atomic public replacement.

### Verification commands

The exact test target names may be inline unit modules or integration targets,
but the handoff must map every acceptance criterion above to one named test.
The minimum command matrix for the final M2 commit is:

```bash
cargo fmt --all --check
cargo test -p calc-flow --lib runtime::streaming
cargo test -p calc-flow --test stream_operator --test stream_compile \
  --test streaming --test micro_batch
cargo test -p calc-flow
cargo test -p calc-flow --benches --no-run
cargo clippy --workspace --all-targets --all-features -- -D warnings
RUSTDOCFLAGS="-D warnings" cargo doc --workspace --all-features --no-deps
git diff --check
```

The optional evidence commands, outside the ordinary matrix, are:

```bash
CALC_FLOW_STREAM_SOAK=1 cargo test -p calc-flow --lib runtime::streaming::soak::one_hour_two_source_slow_sink -- --ignored --exact --nocapture
cargo bench -p calc-flow --bench core -- stream
```

The soak command above is the external/manual gate; this specification
revision is not evidence that it was run. The soak report records the Linux
RSS, graceful-drain conservation oracle, and machine metadata required by
M2C-FR32. A non-Linux structured skip is not a passing soak artifact.
Criterion point estimates and confidence intervals are reported separately;
a performance comparison is a delivery gate only under M2C-FR33's paired-run
rule.

## Expected file scope

The implementer may narrow this list, but production edits outside it require
an explicit spec delta before implementation.

- Add `crates/calc-flow/src/runtime/streaming/operator_task.rs`.
- Add `crates/calc-flow/src/runtime/streaming/sink_task.rs`.
- Add `crates/calc-flow/src/runtime/streaming/metrics.rs`.
- Modify `crates/calc-flow/src/runtime/streaming/{mod,job,source_task,supervisor,channel,context}.rs`.
- Modify `crates/calc-flow/src/operator/{stream,expression,sql,union}.rs` only
  as needed for task-owned resources and instrumentation.
- Modify `crates/calc-flow/src/pipeline/stream.rs` only to expose crate-private
  compiled wiring/owned operators to the runtime; no public compile semantics
  change is authorized here.
- Do not add a public error variant in `crates/calc-flow/src/error.rs` for this
  gate; use the completion API note's private cause/origin records plus
  existing non-exhaustive errors.
- Add focused integration tests under
  `crates/calc-flow/tests/{stream_supervisor,stream_source,stream_operator_task,stream_job}.rs`
  or keep equivalent focused unit tests beside the private modules. Keep the
  ignored one-hour harness in the crate-private `#[cfg(test)]`
  `runtime::streaming::soak` module so it exercises the private runtime without
  creating a public test seam.
- Modify `crates/calc-flow/tests/support/mod.rs` for observable mock lifecycle
  probes only.
- Modify `crates/calc-flow/benches/core.rs` for the three M2 Criterion cases.
- Modify `docs/runtime-envelope.md` and `docs/introduction.md` only after the
  doc writer establishes whether an internal-completion statement is useful;
  neither may claim public A6/checkpoint/event-time availability.
- Do not modify Python, Studio, schemas, project/checkpoint v2 fixtures, or
  `tests/fixtures/v1/`.

## Delivery sequence and gates

1. Implement pure all-binding/topology preflight and the runner-core-owned
   provisional launch driver.
2. Register operator tasks, run fresh-entry `reset`, collect all entry acks,
   then implement concurrent connector open and the separate data/control
   gate. Prove start-observer cancellation at every phase before data work.
3. Complete source drain linearization, lifecycle ownership/reaper, and
   non-recursive terminal metrics while preserving the PR #83 tests.
4. Add M2.3 operator tasks and prove the two-source union-to-expression data/
   EOF path, unary control, fail-closed multi-input control, premature-close
   handling, and partial fan-out evidence.
5. Add crate-private ordinary sink tasks and process-local ordered drain/cancel
   behavior.
6. Add M2.5 metrics, seed-to-gate paused-time stress, ignored Linux soak, and
   public-M1-seam benchmarks.
7. Run tester, reviewer, and doc-writer stages. The deliverable may be called
   “M2 runtime internals complete” only after critic approval and exact-final
   verification; it never performs the public A6 cut.

Every behavior change follows RED -> observed expected failure -> focused
GREEN -> affected matrix -> reviewer. A code change cannot be considered
complete without `cf-reviewer`; documentation disposition belongs to
`cf-doc-writer`.

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
- The completion API note owns the exact core/driver/reaper and runtime-plan
  projection signatures. This spec owns their required observable outcomes.

## Handoff

Next role: `cf-critic`, after the matching completion API-note revision is on
disk. The critic must verify B1-B5 plus partial fan-out, premature close,
launch cancellation, and metrics-overflow tests are implementable without
M3-M5 work. Implementation must not start until that verdict has zero
remaining blocks.
