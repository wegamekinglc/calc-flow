# Continuous Streaming M2 Completion - Critic Critique

## Target

- Spec: `.codex/artifacts/specs/continuous-streaming-m2-completion.md`
- API note: `.codex/artifacts/api-notes/continuous-streaming-m2-completion.md`
- Controlling spec: `.codex/artifacts/specs/continuous-streaming-runtime.md`
- Controlling API note: `.codex/artifacts/api-notes/continuous-streaming-runtime.md`
- Approved total-artifact critique:
  `.codex/artifacts/critiques/continuous-streaming-runtime.md`, round 3
- Plan: `docs/superpowers/plans/2026-08-02-continuous-streaming-v3.md`,
  tasks M2.1-M2.5 and the M2 merge gate
- Code baseline: `1d5546028e2ce9ebce59c976080c3d11c1225e16`

## Verdict

**Block.** The decision to keep A6 private until M4/M5 is the honest public-API
choice, and the current compiled plan contains enough owned topology to make an
internal M2 runtime possible. Four lifecycle/correctness contracts are not yet
implementable as written, however: operator initialization is gated in the
wrong phase, the graceful-drain cut is defined by unobservable future readiness,
the take-once supervisor has no cancellation-safe ownership protocol, and
checked metrics overflow can recurse while recording the terminal error. A
fifth blocking edge case is a caller-dropped `start()` future during connector
open: the note promises close/join on open failure but gives this equally normal
Rust cancellation path no owner.

**BLOCKS REMAINING: 5**

## Findings

### Blocking Issues

- **B1 - The proposed open gate contradicts the controlling operator-entry
  order.** The completion API note says that, after pure preflight, sources and
  sinks open and "pumps, operator tasks, and sink writes remain start-gated
  until all opens succeed" (completion API note, source preflight/open phase,
  lines 189-210). The controlling A3 contract instead requires every operator
  `reset`/fresh-entry/`restore` to execute inside its registered operator task
  and finish before any source opens (total API note A3, lines 502-525). These
  cannot both hold with one start gate.
  - **Counter-example:** an external `StreamOperator::reset()` returns
    `CalcFlowError::Operator`. Following the completion note opens a Kafka
    source and an ordinary HTTP sink first, then releases the operator task and
    discovers the deterministic failure after external side effects. Following
    A3 requires polling the operator task before the connector open phase,
    violating the completion note's single gate.
  - **Suggested fix:** specify two gates. After pure whole-job validation,
    register every operator task and run only its entry phase; wait for all
    operator entry acknowledgements. Only then run the concurrent connector
    open phase. When all opens succeed, release a separate data/control gate for
    pumps, operator ingress loops, and sink writes. An operator-entry failure
    must join the registered init tasks without invoking any connector
    lifecycle method. Add a test with a failing external `reset()` and source/
    sink open canaries.

- **B2 - Graceful drain is defined by a fact the runtime cannot observe.** The
  completion API note defines the accepted prefix to include an event whose
  `next` "completed before the drain decision", including the capacity-one
  slot, while a still-pending `next` is dropped (lines 241-251). A Rust future
  has no observable completion timestamp before the executor polls it to
  `Ready`; a connector or worker may have completed the underlying I/O while
  the pump has not yet observed that readiness. No implementation can decide
  which side of this cut that event belongs to from the stated contract.
  - **Counter-example:** the slot is empty and the pump holds its pre-reserved
    permit while `next()` becomes ready on an external wake. Before Tokio polls
    the pump again, `shutdown()` serializes the drain decision. The note says
    the item is accepted if "completed" means connector completion, but the
    runtime cannot distinguish it from a genuinely pending poll and will drop
    it. If it polls once after the decision to discover readiness, it can accept
    an item that became ready after the decision. Either interpretation loses
    the promised cut.
  - **Suggested fix:** define acceptance exclusively by runtime observation:
    an event is accepted iff the pump has atomically committed it to the
    one-item slot before a serialized per-source drain cut. Reserve-before-poll
    remains valid, but readiness alone is not acceptance. Freeze a shared
    source state machine such as `Polling -> Slotted -> Draining -> Closed` and
    make the slot commit and drain transition mutually exclusive. The source
    task drains the committed slot and any already-enqueued edge send, emits
    EOF, then joins the pump; a poll not committed at the cut is dropped once
    and never resumed. Test all cut points with gates, not sleeps.

- **B3 - `JobCore`'s take-once supervisor is not safe under the very future-
  drop and handle-Drop behavior the note promises.** The completion API note
  gives `wait`, `shutdown`, and `cancel` concurrent `&self` methods, says a
  dropped wait future does not cancel the job, and says `JobCore` owns a
  take-once supervisor registry while owning-handle Drop atomically transfers
  it to the reaper (runner/reaper section, lines 445-508). It does not define
  where the registry lives while one terminal future is actively joining, or
  how Drop, another terminal caller, and runner shutdown arbitrate that
  ownership. A boolean `owns_registry` on the handle does not serialize an
  async take from the core. The baseline demonstrates the hazard: the current
  `wait(self)` holds `TaskSupervisor` in its future, and supervisor Drop calls
  `abort_all` (`job.rs` lines 77-109; `supervisor.rs` lines 239-243).
  - **Counter-example:** caller A polls `wait()` far enough to take the
    supervisor, then drops only the wait future. If the supervisor is
    future-local, Drop cancels/aborts the job, contradicting idempotent
    observation. If the registry is left marked "taken", caller B's `cancel()`
    or runner shutdown cannot join it. Concurrent owning-handle Drop can then
    transfer either an empty registry or race the active join, violating D5.
  - **Suggested fix:** freeze one atomic ownership state machine in `JobCore`,
    not a handle-local boolean. It must distinguish at least `CoreOwned`,
    `Driving`, `ReaperOwned`, and `Terminal`, and store the partially progressed
    supervisor back into core-owned state whenever a driving future is dropped.
    Alternatively, make one core-owned registered convergence-driver task and
    have every terminal method observe it; that driver itself must remain in a
    runner/job registry and be transferred to the reaper on handle Drop. State
    exact linearization points for terminal cause selection, registry transfer,
    and outcome publication. Add adversarial `poll`/drop tests at every await in
    wait, cancel, shutdown, and join, plus concurrent terminal-call and
    job-Drop/runner-shutdown races.

- **B4 - Checked error metrics can recursively create terminal errors.**
  M2C-FR30 says all counters are checked and that errors are counted when the
  terminal decision records them (completion spec lines 251-255). The API note
  says overflow returns `InvalidArgument` and converges the job (metrics
  section, lines 576-590), and the diagnostics table names that terminal error
  (lines 618-639). It never exempts the error counter that is being updated by
  the terminal decision.
  - **Counter-example:** inject `job.task_errors = u64::MAX` and submit an
    operator failure. Recording that primary error overflows and produces the
    metrics-overflow error. Recording the new error increments the same full
    counter, producing the same error recursively; the single terminal
    decision never reaches outcome publication. The focused checked-overflow
    acceptance test can expose this immediately.
  - **Suggested fix:** make terminal error accounting non-recursive. Record the
    selected primary outcome first. Attempt each error-counter increment at
    most once; if it overflows, attach one metrics-overflow secondary (or make
    it primary only when the original operation was itself a metrics record),
    set a non-failing bounded `metrics_overflowed` flag, and never feed that
    secondary back through error accounting. Specify whether the counter stays
    at `u64::MAX`; saturation is acceptable for the diagnostic flag even though
    ordinary counters remain checked.

- **B5 - Dropping `ContinuousRunner::start()` during open has no cleanup
  contract.** `start(&self)` is async and the completion note requires
  concurrent source/sink opens. It specifies close-all and join on an open
  error (lines 205-210), but says nothing about the caller dropping the start
  future after one or more opens begin. This is ordinary Rust cancellation,
  not a destructor-only abuse. Async `close()` cannot run from Future Drop,
  and moving connectors into future-local open units leaves no reaper target
  when that future disappears.
  - **Counter-example:** source A opens successfully, source B is blocked in
    `open`, and the caller's `tokio::select!` drops `start()`. A's external
    resource remains open and B's open future is dropped without the promised
    close-all convergence. No `ContinuousJob` exists for the caller to cancel,
    while a second `start()` can observe ambiguous active state.
  - **Suggested fix:** make provisional launch state runner-core-owned before
    the first open begins. A dropped start observer must either leave a
    registered launch driver that converges to a live job held by the runner,
    or request cancellation and transfer all open-phase units/resources to the
    runner reaper. Freeze what the next `start()` and runner `shutdown()` do
    with that provisional launch, and add a poll/drop test at every open-phase
    await. Do not rely on connector object Drop as a substitute for async
    `close()`.

### Significant Concerns

- **S1 - Deferring public A6 is correct, but calling this unchanged-plan M2
  "complete" is not yet honest.** The plan's merge strategy says M2.4
  publishes the unified runner (`plans/...continuous-streaming-v3.md`, line
  400); task M2.4 explicitly implements `StreamingRunner::start`, deletes the
  push runner and `MicroBatchRunner`, and gates on a usable source-driven
  runtime (lines 795-825). The delta instead says M2 completes as crate-private
  infrastructure and moves public A6 to a post-M4/M5 integration milestone
  (completion spec lines 421-435; completion API decision 1, lines 82-101).
  That is the safer technical decision, but it is a milestone deviation, not
  completion of the plan as currently written.
  - **Suggested fix:** the spec writer should explicitly rename this gate
    "M2 internal runtime completion", amend the plan/handoff ledger so public
    M2.4 is superseded rather than checked off, and name the later A6
    integration milestone. Reports must say "M2 runtime internals complete"
    until that plan revision lands. The API designer should keep A6 deferred;
    do not weaken its checkpoint promise merely to preserve the old schedule.

- **S2 - Fan-out failure is observably non-atomic and the artifacts should say
  so.** `ChannelStreamCollector::emit` prevalidates all branches, then awaits
  independent senders sequentially (completion API lines 274-291). If branch
  one accepts and branch two closes, branch one can process or write externally
  before `emit` returns the error. Output metrics remain zero because the
  all-branches boundary was not reached. FR12 only promises identical sequences
  in a fault-free run, so this is not a contradiction, but neither spec nor
  test plan acknowledges the branch divergence. The same issue exists for
  source fan-out and for an external operator that successfully emits once,
  then fails on a later `emit` in the same handler.
  - **Suggested fix:** state explicitly that M2 fan-out is not transactional:
    a failure may leave a delivered prefix on earlier branches; convergence
    cancels the job and no rollback or cross-branch atomicity is claimed.
    Define the output counter as "fully fanned-out emits", retain per-edge send
    counters so the partial delivery is observable, and add a branch-two-close
    test. Do not assert zero downstream data on an arbitrary send/handler
    failure; only validation-before-first-send has that property.

- **S3 - Runtime boundary edges and plan extraction need a named internal
  contract before wiring.** The current `StreamExecutionPlan` does own compiled
  nodes/operators, internal edges, and external input/output endpoints
  (`pipeline/stream.rs` lines 135-203), so owned non-clone operators are
  constructible by consuming the plan; this is **not a blocker**. However,
  source-to-first-node and last-node-to-sink channels are not entries in the
  compiled `edges` map, and there is no crate-private `into_runtime_parts()`.
  The proposed status/errors require stable IDs for every edge, including
  first-hop budget diagnostics, without defining those boundary IDs. Compiled
  operators are also wrapped in `Arc<Mutex<_>>`; an M2 task needs one owned
  value.
  - **Suggested fix:** add a crate-private consuming projection such as
    `StreamRuntimePlanParts` with owned nodes/operators, explicit source routes,
    explicit sink routes, and every internal/boundary edge's stable ID and
    budget. Either refactor stream compiled nodes to direct ownership or prove
    `Arc::try_unwrap` is infallible because the non-Clone plan exposes no Arc.
    Freeze deterministic boundary-edge ID formulas and add a full topology
    reconstruction test for unary, fan-out, independent branches, and union.

- **S4 - Channel closure must not masquerade as EOF.** `EdgeReceiver::recv()`
  returns `None` after sender teardown, while stream EOF is an explicit
  `EndOfInput`. The operator/sink task note only specifies handling the message
  enum. Treating `None` as ended would silently turn a producer that exited
  without EOF into natural completion and data loss.
  - **Suggested fix:** while the job is running or draining, `recv() == None`
    before explicit EOF is `EdgeClosed` with the ingress/edge origin. During an
    already-selected cancel/failure convergence it is normal wakeup or a
    secondary error according to one stated rule. Add missing-EOF tests for
    unary, union, and sink ingress.

- **S5 - Open-phase and sink teardown errors do not fit the proposed result
  surfaces.** Ordered sink writes preserve the failing sink through private
  `FailureOrigin`, and close failures are secondary in stable sink-ID order
  (completion API lines 366-379). But connector open happens before
  `start() -> Result<ContinuousJob>` returns, so there is no
  `ContinuousJobOutcome` in which to attach close-all failures. It is unclear
  whether a failed sink's own `close()` is always attempted and where multiple
  close failures are retained.
  - **Suggested fix:** define start-failure aggregation separately: return the
    deterministic primary open error with its binding origin, retain close
    failures in a bounded runner diagnostic/reaper record in stable ID order,
    and count them once. State that every resource whose `open` began,
    including the resource whose open returned `Err`, receives one `close`.
    For running sink tasks, keep write/open as primary and close errors as
    secondary; test sink-two write failure plus all close combinations.

- **S6 - Per-edge metrics observation boundaries are incomplete.** FR29 asks
  for input/output batches, rows, and bytes per edge, but FR30 names only
  operator-dequeue, fully-fanned-out operator output, and successful sink
  write. It does not say whether edge input means successful enqueue, whether
  edge output means dequeue, or how source/sink boundary edges count. The
  recorder sketch has no edge record method.
  - **Suggested fix:** define edge input at successful queue commit and edge
    output at successful dequeue, independently of operator-level fully-fanned
    output. Instrument the channel shared state or add stable edge-only recorder
    methods. Add a partial-fan-out metric test so edge one can show one enqueue
    while operator fully-fanned output remains zero.

- **S7 - The stress/soak/benchmark gates need executable harness details.** A
  paused Tokio runtime does not itself offer "seeded schedules"; the test must
  synthesize readiness permutations with seeded gates. The soak gives an RSS
  formula but no sample cadence, RSS provider/platform rule, allocator, or
  minimum sample count. The ordinary matrix compiles benches with
  `--no-run`, while FR33 also says they produce a baseline report. Finally,
  Criterion targets are external crates and cannot call the crate-private job,
  so "unary stream overhead" is ambiguous under the no-public-change gate.
  - **Suggested fix:** define a deterministic seed-to-gate schedule generator
    for 100 cases; specify the ignored soak command, Linux RSS source (or a
    reviewed portable provider), cadence, sample count, allocator/report
    metadata, and unsupported-platform behavior; keep the one-hour run outside
    ordinary CI. Distinguish bench compilation from an opt-in `cargo bench`
    baseline run. Define whether the unary case measures public
    `StreamOperator` + channels or add a non-public in-crate measurement seam;
    do not expose a fake public runtime solely for Criterion.

### Minor / Style Notes

- **M1 - M2C-FR3 still calls the deadline outcome an open API-designer
  decision after the API note closes it.** Completion spec lines 104-109 say
  the exact outcome is a decision item; the API note fixes
  `Cancelled/DeadlineExceeded` with no error at lines 420-437. Replace the stale
  sentence with the decided result so the spec is self-consistent.
- **M2 - Current documentation comments still promise public replacement in
  M2.4.** `runtime/streaming/mod.rs` lines 3-6 and several `dead_code` reasons,
  plus `pipeline/stream.rs` line 5, say M2.4 publishes the public runner. If A6
  is deferred, the eventual implementation/doc pass must update those comments
  without claiming public continuous runtime availability.

## Axis Sweep

- **Correctness:** blocked by B1-B4; immutable `Batch` sharing and per-ingress
  FIFO otherwise match `docs/introduction.md` and S1-S4.
- **Hidden assumptions:** B2 assumes readiness is observable; B3 assumes an
  async-taken registry still has an owner; S3 assumes boundary IDs exist.
- **Missing edge cases:** B5, partial fan-out, premature channel close, and
  multi-error sink teardown are missing from the acceptance map.
- **Backwards compatibility:** the private completion approach preserves the
  current v2 `StreamingRunner`, `MicroBatchRunner`, public tests, project/
  checkpoint v2 documents, and `tests/fixtures/v1/`. **OK**, provided no new
  re-export or fixture rewrite lands and AC-COMPAT remains green.
- **Performance:** bounded channels, one-item source slots, task-local
  DataFusion runtimes, and sequential configured sink delivery are coherent.
  The sequential sink order is intentional, not an accidental serialization.
  Benchmark execution details need S7.
- **Surface and ergonomics:** deferring A6 avoids a dishonest checkpoint facade.
  Private names carry no compatibility promise. Milestone reporting needs S1.
- **Test plan:** broad and capable of rejecting stubs, but it needs the
  adversarial lifecycle cases in B1-B5 and executable harness definitions in
  S7.
- **Risk and scope:** no M3-M5 implementation is required. Unary watermark
  pass-through plus multi-ingress fail-closed is a valid temporary M2 boundary;
  no generated watermark, barrier, state, checkpoint, or durability claim
  should enter this diff.

## Counter-Proposals

- **Three-phase launch:** pure owned preflight; registered operator-entry phase;
  concurrent connector-open phase; then release one data/control gate. This is
  the smallest launch shape satisfying both A3 and no-data-before-open.
- **Core-owned lifecycle driver:** keep the supervisor and partially joined
  state in `JobCore`; terminal callers submit causes and observe one driver.
  A dropped observer cannot own the registry, and handle Drop only changes the
  driver's owner from job core to reaper.
- **Runtime-observed drain cut:** accepted means committed to the capacity-one
  slot before the per-source drain transition, never "the connector had made
  the future ready". This yields a testable, exact prefix without probing a
  future after the cut.

## Questions for the Author

1. Will the milestone plan itself be revised so "M2 complete" means internal
   runtime completion, or should status reports retain a distinct
   "M2-internal complete / public M2.4 deferred" state?
2. Is `StreamOperator::reset()` required for every fresh internal M2 job, as
   controlling A3 says, and if so which registered phase runs it before open?
3. Which component owns a provisional launch when its `start()` observer is
   dropped?
4. Is partial fan-out delivery an explicitly accepted process-local failure
   outcome, and which per-edge metric exposes it?
5. What exact stable IDs name source/sink boundary channels synthesized from
   `StreamExecutionPlan::external_inputs/outputs`?

## Handoff

The `cf-spec-writer` should revise B1, B2, B4, B5, S1, S2, S4, S6, and S7,
because they are milestone/lifecycle/observation semantics. The
`cf-api-designer` should revise B3 and S3/S5: registry ownership and start-error
surfaces are API/ownership contracts, and the internal plan projection needs a
precise signature. After both revisions, route the artifacts back to
`cf-critic`; implementation should not cross the launch, drain, terminal, or
metrics-overflow boundaries until the five Blocks are closed.

---

## Round 2 - Focused Closure Review

### Verdict

**Approve.** Revision 2 closes every blocking issue and significant concern
from round 1 without weakening the controlling total-runtime contract or
pulling M3-M5 behavior into M2. The launch, drain, convergence, and metrics
contracts now have implementable linearization points, and the acceptance map
contains the adversarial cases needed to reject future-local ownership or
cleanup shortcuts.

**BLOCKS REMAINING: 0**

**Cleared for `cf-implementer`.** The labels below approve the revised
artifacts, not an implementation that has not yet been written or verified.

### Blocking-Issue Disposition

- **B1 - RESOLVED.** M2C-FR7B and the launch section now require pure
  preflight, registered in-task operator entry and acknowledgements, then
  concurrent connector open. `OperatorEntryGate` and `DataControlGate` are
  independent. A reset failure joins the registered entry tasks without any
  connector lifecycle call, and AC-M2.1-D pins both the call canaries and the
  post-entry data gate. Keeping the data gate closed until the start observer
  claims the handle is a safe strengthening: successful opens are parked, and
  it gives start cancellation a no-data-before-delivery boundary.

- **B2 - RESOLVED.** M2C-FR10A and `SourceAcceptState` define acceptance only
  at the atomic `Polling -> Slotted(event)` commit. Commit and drain cut share
  one synchronization point; connector readiness and `Poll::Ready`
  observation are explicitly non-authoritative. A committed event remains in
  the accepted prefix when transferred from the slot into an already-started
  edge send, which the source task drains before EOF. AC-M2.2-B2 covers all
  relevant cut points, including ready-but-unpolled, Ready-before-commit,
  slot-committed, and edge-send-pending cases. The cut is observable and
  implementable without polling a future after the decision.

- **B3 - RESOLVED.** The revised design removes the take-once registry from
  terminal observers entirely. One runner-registered `JobDriver` owns the
  `TaskSupervisor`, launch resources, and partially progressed joins for its
  whole lifetime; `RunnerLifecycleDriver` alone polls and joins job drivers.
  `wait`, `cancel`, and `shutdown` only observe or synchronously submit a
  cause. Their futures never own a `JoinHandle` or supervisor, so observer
  drop cannot abort or strand convergence.
  - The ownership state is self-consistent under the required races.
    `CoreOwned -> Driving` linearizes at job-driver gate release. Job or start
    observer Drop changes result ownership to `ReaperOwned` under
    `JobCoreState` while leaving the physical registration in the runner
    driver. Outcome publication stores the immutable outcome and `Terminal`
    atomically under that same lock. Whichever of Drop and publication wins
    first, publication ends in `Terminal` with the one joined result.
  - Runner shutdown closes admission, converges the provisional launch,
    submits cancellation to live jobs, waits for the same registered drivers,
    and then drains reaper diagnostics. Concurrent job Drop may change only
    the logical owner, not move the registry. Dropping the shutdown observer
    likewise leaves the lifecycle handle in `RunnerCore`. AC-M2.1-E and the
    manual-poll sketches cover observer drop at every await and the combined
    job-Drop/runner-shutdown race.

- **B4 - RESOLVED.** M2C-FR30A and
  `account_terminal_errors_once` freeze cause and primary failures first,
  attempt each associated error counter once, retain a full counter at
  `u64::MAX`, set an infallible bounded overflow flag, and attach at most one
  overflow secondary for the entire outcome. The secondary is explicitly not
  re-accounted. AC-M2.5-A2 covers both a full terminal-error counter and an
  ordinary record operation whose own overflow is primary. No recursive path
  remains.

- **B5 - RESOLVED.** M2C-FR7C installs a runner-core-owned provisional launch
  driver before lifecycle work, while `StartObserver` is only a cancel-on-Drop
  observer. Operator-entry and connector-open futures/resources remain owned
  by `JobDriver`. Drop before delivery changes
  `Provisional|ReadyUnclaimed -> CancelRequested`, submits cancellation, and
  makes the result reaper-owned; it never attempts async cleanup from Drop.
  The claim/drop race is serialized on `JobCoreState`, and the data gate stays
  closed until `ReadyUnclaimed -> Claimed`, so cancellation cannot conflict
  with the three-stage launch or release an unobserved live job. Next start and
  runner shutdown join the provisional/reaper registration first. AC-M2.1-F
  covers every entry/open/delivery boundary and exactly-once close for every
  began-open connector.

### Significant-Concern Disposition

- **S1 - RESOLVED.** The delta is consistently named **M2 internal runtime
  completion**, explicitly supersedes rather than checks off the original
  public M2.4 cut, places public A6 after complete M4/M5 integration, and
  freezes the reporting phrase “M2 runtime internals complete.” Existing v2
  runners remain available.

- **S2 - RESOLVED.** M2C-FR12/M2C-FR18 explicitly make fan-out
  non-transactional after the first successful send, preserve an earlier
  branch prefix, prohibit rollback claims, distinguish per-edge enqueue from
  fully-fanned-out output, and cover both branch-close and
  emit-then-handler-error cases in AC-M2.2-D/AC-M2.3-E.

- **S3 - RESOLVED.** `into_runtime_parts(self, default_budget)` consumes
  directly owned, non-Clone operators into `RuntimeStreamNode`; cloning,
  locks, and `Arc::try_unwrap` are forbidden. `StreamRuntimePlanParts` names
  nodes, internal edges, source routes, sink routes, endpoints, kinds, and
  budgets. Hex-encoded source/sink boundary-ID formulae are deterministic and
  collision-checked across the full edge set. Boundary edges are real charged
  metric/backpressure edges, and topology reconstruction tests cover unary,
  fan-out, independent branches, and union.

- **S4 - RESOLVED.** M2C-FR7A/M2C-FR19A track explicit EOF separately from
  receiver closure. Premature `None` is `EdgeClosed` with stable ingress
  origin while running/draining, while runtime-marked convergence close is a
  wakeup and an already-observed close remains a secondary. AC-M2.3-D covers
  unary, union, and sink ingress.

- **S5 - RESOLVED.** `StartFailure` returns the stable primary origin and a
  bounded runner diagnostic ID for cleanup failures. Every began-open
  resource, including the one returning `Err`, is closed once; diagnostics
  retain cleanup failures in stable order with explicit bounds and truncation
  state. Running sink write/open failure remains primary and close failures
  become ordered outcome secondaries.

- **S6 - RESOLVED.** M2C-FR30 and the recorder API define edge input at the
  successful atomic queue commit and edge output at dequeue plus reservation
  release, under the channel-state lock. The same rules cover internal and
  synthesized boundary edges. Operator and sink counters have separate
  documented boundaries, and AC-M2.5-A3 pins partial-fan-out values.

- **S7 - RESOLVED.** The short stress harness uses a fixed test-local
  seed-to-gate permutation rather than claiming executor seeding. The ignored
  soak now fixes invocation, Linux `VmRSS` provider, ten-second cadence,
  minimum samples, warm-up, slope/median thresholds, metadata, and non-Linux
  unsupported behavior. Bench compilation is separated from the opt-in
  Criterion baseline, and the external benchmark uses only the public M1
  operator/channel seam without exposing a fake runtime.

### Minor-Note Disposition

- **M1 - RESOLVED.** M2C-FR3 now freezes deadline as
  `Cancelled + DeadlineExceeded`, with no error unless teardown contributes a
  secondary, and explicitly excludes `CheckpointTimeout`.

- **M2 - RESOLVED at artifact level.** The revised status, FR28, compatibility
  section, and documentation-audience rules prohibit describing the private
  slice as the public M2.4 replacement. The implementation/reviewer/doc-writer
  pass must apply that rule to the stale baseline module and `dead_code`
  comments; this is a delivery check, not a remaining artifact ambiguity.

### Focused Consistency Result

- **Driver ownership:** self-consistent. Observer futures own no registry;
  physical driver registration never moves; cause commit, logical reaper
  transfer, and outcome publication have distinct serialized points.
- **Launch versus cleanup:** self-consistent. Entry precedes open, opened
  connectors stay parked, handle claim precedes data-gate release, and any
  earlier observer drop converges through the registered driver.
- **Drain cut:** implementable. Only slot commit accepts; drain and commit are
  mutually exclusive; accepted local/slot/send work drains before EOF.
- **Metrics overflow:** non-recursive by construction and bounded to one
  secondary.
- **Runtime-plan projection:** sufficient for direct operator ownership,
  complete boundary-channel construction, stable diagnostics, budgets,
  backpressure, and per-edge metrics.

No round-2 regression or new finding was introduced. Proceed to
`cf-implementer`, then require the planned `cf-tester`, `cf-reviewer`, and
`cf-doc-writer` gates before reporting the internal milestone complete.
