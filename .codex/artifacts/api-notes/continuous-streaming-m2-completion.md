# Continuous Streaming M2 Internal Runtime Completion - API Note

## Sources and status

- Delta specification:
  [`../specs/continuous-streaming-m2-completion.md`](../specs/continuous-streaming-m2-completion.md).
- Total semantic specification, controlling except for the explicit delta
  clauses below:
  [`../specs/continuous-streaming-runtime.md`](../specs/continuous-streaming-runtime.md).
- Frozen total API target, controlling except for matching explicit delta
  projections:
  [`continuous-streaming-runtime.md`](continuous-streaming-runtime.md), especially
  A2, A4, A5, A6, and A8.
- Approved total-artifact critique:
  [`../critiques/continuous-streaming-runtime.md`](../critiques/continuous-streaming-runtime.md),
  round 3, `BLOCKS REMAINING: 0`.
- Milestone plan:
  [`docs/superpowers/plans/2026-08-02-continuous-streaming-v3.md`](../../../docs/superpowers/plans/2026-08-02-continuous-streaming-v3.md),
  tasks M2.1-M2.5.
- M2 completion critique:
  [`../critiques/continuous-streaming-m2-completion.md`](../critiques/continuous-streaming-m2-completion.md),
  round 5 at `6d5740d3df733a02450d45bdde168a3dfb4b03e9` approves revision 6
  with `BLOCKS REMAINING: 0`.
- Audit anchors for this revision use four deliberately distinct terms. The
  **comparison baseline** is `origin/main` at
  `d45c2b26def2c4dfe179f2b7c7c1d2411fc069b6`; the **historical M2 skeleton**
  is `1d5546028e2ce9ebce59c976080c3d11c1225e16`; the **audited implementation
  head** is
  `574c0fb7f678781370303100c9f5089f6ed59bca`; and the **artifact starting
  head for this API-note revision** is
  `5fddf4fa3a0f71e4e0634789a95d795eb9b82f74`. Commits after `574c0fb`
  through `5fddf4f` change completion artifacts, not implementation code. The
  comparison baseline and branch merge at
  `c4979061cc239b43617f642e2294794f2833d95d`; at `5fddf4f`, the branch is
  nine commits ahead of and one commit behind that baseline. The historical
  skeleton is not the baseline for relative-to-`main` compatibility claims. A
  later **code-approved head** is
  `7deda2a0d801dac7ed5e8e75ca42272ecd72bb48`; it is the source and evidence
  anchor for this delivery reconciliation, not a replacement for the earlier
  historical audit labels.
- Status: revision 7 reconciles the approved M2-internal API/UX delta with the
  delivered implementation at code-approved head `7deda2a0`. Revision 2
  addressed B1-B5 and S1-S7 from
  [`../critiques/continuous-streaming-m2-completion.md`](../critiques/continuous-streaming-m2-completion.md).
  Revision 3 synchronized the crate-private real-runtime soak seam and the
  internal `LaunchId`/context job-ID split. Revision 4 reconciles that contract
  with audited implementation head `574c0fb`, the universal 20-minute soak
  standard, binding-qualified source diagnostics, bounded panic text, and the
  PR-added public `CalcFlowError::TaskPanicked` variant. Revision 5 closes B6
  at the design level with finite envelope-slot admission and resolves round-3
  compatibility, soak-evidence, verification, and performance precision.
  Revision 6 conforms that design to the delta specification's narrow B7 S10
  supersession and public-channel migration contract and received the round-5
  zero-block critique. Revision 7 records delivery without changing the
  approved requirements. The Rust snippets below remain precise
  implementation contracts; newly named M2 runtime types remain `pub(crate)`.
  This note does not authorize or revise the future public A6 runner surface.

### Delivered G1-G8 status at `7deda2a0`

The delivered test map closes AC-M2.5-A5's real-edge lifecycle case, every
tracked M2 B-family drain/backpressure/FIFO/stress case, and AC-M2.1-F's
dropped-start ownership and reaper convergence case.

| Gap | Delivered status                                                                                                      |
| --- | --------------------------------------------------------------------------------------------------------------------- |
| G1  | Closed: cursor and watermark regression fields are binding-qualified and fail before enqueue.                         |
| G2  | Closed: panic text is UTF-8-safe and at most 1,024 bytes; non-string payload text is fixed.                           |
| G3  | Closed: the exact 20-minute private soak command ran on Linux with 120/120 samples.                                   |
| G4  | Closed in source/rustdoc and reconciled here across the governing current-state documentation.                        |
| G5  | Closed: Codacy reported zero annotations at the code-approved head.                                                   |
| G6  | Closed: four of four threads were resolved; Copilot review `4871842015` added zero comments.                          |
| G7  | Closed for the code-approved head: 15/15 checks passed and GitHub reported `MERGEABLE`/`CLEAN`.                       |
| G8  | Closed: every edge enforces independent slot/row/byte bounds and terminal paths converge without leaked reservations. |

The durable exact-head evidence is the [PR #83 soak and performance
summary](https://github.com/wegamekinglc/calc-flow/pull/83#issuecomment-5201266650).
Its raw log SHA-256 is
`bc97c8f736ad41a4f228e07300f1ecd23c9af9fb09dc1be1718823430bd05f35`.
The run recorded 120/120 RSS samples, 96,124 accepted batches at each sink,
24,032 accepted zero-cost batches, zero missing/duplicate batches, zero final
task/queue/reaper leaks, and an RSS slope of -2.609 MiB/hour. The paired
base/head timing result is `INCONCLUSIVE` and non-blocking because `main` does
not contain the stream cases; benchmark compilation passed on both refs.

## Governing artifact precedence

Resolve any conflict in this order:

1. Revision 6 of the M2 completion delta specification controls this internal
   completion gate and its two explicit supersessions.
2. This API note controls exact Rust signatures and crate-private projections
   only where that delta delegates them; this note cannot contradict the
   delta.
3. The total continuous-streaming specification and total API note remain
   controlling for every clause not expressly superseded here.
4. The milestone plan and research document supply schedule and rationale but
   do not override either normative artifact.
5. The latest critique gates progression but does not silently rewrite
   behavior; implementation still requires a zero-block verdict.

The first explicit supersession remains timing-only: M2C-NFR4/M2C-FR32 replace
every earlier one-hour soak name, duration, sample default, or command with an
exactly 1,200-second workload and exactly 120 ten-second Linux samples. The
plan's public M2.4 cut also remains superseded as a schedule/reporting matter:
the frozen A6 behavior does not change, and its public integration stays a
separately reviewed post-M5 gate.

The second explicit supersession is limited to the third edge-admission
predicate:

- Total-spec S10.1 changes only from two exhaustive admission dimensions to
  three atomic predicates: envelopes against `max_rows`, rows independently
  against `max_rows`, and bytes against `max_bytes`. The public configuration
  still has exactly the two positive fields `max_rows` and `max_bytes`.
- Total-spec S10.5 changes only from two-dimensional to three-dimensional
  atomic reservation/release. Its FIFO, receiver-close wakeup, no-lost-wakeup,
  and single-producer-per-edge rules remain frozen.
- Total-spec FR23, the edge-budget inputs row, AC-S10, and their matching total
  API/documentation projections change only where row/byte admission is called
  exhaustive or zero-row/zero-byte traffic is said to bypass capacity. Those
  projections add the slot predicate and exact-capacity block/resume test.
- Total-spec S10.2 row/byte accounting, S10.3 oversize and pre-open source
  validation, S10.4 `Block`/explicit-loss policy, and every other S10 behavior
  remain frozen. D1-D9, S1-S9, I1-I10, NG1-NG13, and A1-A8 are likewise
  unchanged.

No unlisted rule or surface is implicitly superseded. The delta specification
is immediately controlling for this narrow conflict, but all governed specs,
API notes, rustdoc, tests, runtime documentation, and changelog text must be
reconciled before delivery.

## Audiences

- **Rust users:** no new callable runner or continuous-status surface in M2
  completion. Existing public v2 `StreamingRunner`, `MicroBatchRunner`,
  `Source`, `Sink`, checkpoint documents, and their delivery wording remain
  usable and unchanged. Relative to `main`, PR #83 adds one semver-compatible
  item to the public non-exhaustive error enum:
  `CalcFlowError::TaskPanicked`. The existing `EdgeBudget`/`edge_channel`
  signatures remain source-compatible, but `(R, B)` now caps queued envelopes
  at `R`, charged rows independently at `R`, and charged bytes at `B`. The
  future source-driven public A6 replacement remains a separately reviewed
  post-M5 change.
- **Connector and runtime implementers:** crate-private source, operator,
  ordinary-sink, lifecycle, status, metrics, and reaper contracts must be
  implementable and testable without inventing M3-M5 semantics.
- **Python users:** n/a for M2. No adapter, PyO3 type, exception, type-stub, or
  blocking/async method change is authorized.
- **Studio clients:** n/a for M2. No route, OpenAPI, SSE, schema, or generated
  TypeScript change is authorized.
- **Documentation readers:** the reviewed gate may be described only as “M2
  runtime internals complete.” It must not be described as public M2
  completion, a public runner, checkpoint recovery, end-to-end at-least-once,
  or exactly-once delivery.

## Comparison baseline and audited implementation surface

The comparison baseline exports two functioning v2 runners:

```rust
pub struct StreamingRunner { /* push-based BatchExecutionPlan runner */ }

impl StreamingRunner {
    pub fn new(
        plan: Arc<BatchExecutionPlan>,
        checkpoints: Arc<dyn CheckpointStore>,
    ) -> Result<Self>;

    pub async fn step(
        &mut self,
        batch: Batch,
        sinks: &mut SinkRouter,
    ) -> Result<RunResult>;

    pub async fn reset(&mut self) -> Result<()>;
}

pub struct MicroBatchRunner { /* v2 replayable-source runner */ }
```

The public M1 building blocks also remain exported: `StreamExecutionPlan`,
`StreamOperator`, `StreamCollector`, `StreamJobContext`, `StreamMessage`,
`EdgeBudget`, `EdgeSender`, `EdgeReceiver`, and `ChannelMetrics`. Control
message constructors are crate-private. At the comparison baseline,
`CalcFlowError` is
`#[non_exhaustive]` and already contains `EdgeClosed`.

At code-approved head `7deda2a0`, every callable runner signature and
re-export above remains unchanged from the comparison baseline. The PR adds
this public Rust error variant relative to that baseline:

```rust
#[error("task {task_id} panicked: {message}")]
TaskPanicked { task_id: u64, message: String },
```

Adding a variant to a non-exhaustive enum is semver-compatible, but it is still
a public Rust API change and must be reported as such. Only the crate-private
M2 supervisor and launch/runtime paths currently construct it; no public
continuous-runner callable is added. The existing Python wildcard conversion
would classify a future crossing instance as `ExecutionError`, so this slice
adds no Python exception class or type-stub member.

Code-approved head `7deda2a0` also contains the complete crate-private M2
runtime: whole-job preflight, source/operator/sink tasks, private runner and
job lifecycle, reaper ownership, status and metrics, stress, the exact
20-minute real-runtime soak seam, and Criterion cases. It enforces finite
envelope-slot admission, binding-qualified source fields, bounded panic text,
and bounded failed-launch cleanup. G1-G8 are closed at that code/evidence
anchor. These internals are implementation evidence, not a public runner; a
later documentation push changes the PR SHA and therefore requires a final-head
check/soak/review refresh before merge readiness can be assessed.

## Decisions

### Decision 1 - defer the public source-driven runner

**Approved choice for OQ1:** keep the complete M2 runtime crate-private. Do
not stage a public source-driven `StreamingRunner` or `StreamingJob` in M2.
Keep the current public v2 `StreamingRunner` and `MicroBatchRunner` unchanged.

Frozen A6 is one coherent public promise: its constructor accepts a
`StateBackend` and v3 `CheckpointStore`; `start` recovers the latest manifest;
`trigger_checkpoint` publishes an epoch; `shutdown` publishes a final epoch;
and `JobOutcome.completed_epoch` identifies durable completion. M2 has none
of those M4/M5 dependencies. Removing v2 now would delete working replay and
checkpoint behavior, while exporting an A6-shaped type with unsupported
checkpoint methods or fake/in-memory epochs would misrepresent durability.

The public A6 cut is therefore a separately reviewed post-M5 integration
change, after the M4 state backend and complete M5 manifest/coordinator/sink
protocol exist. At that point the replacement must be atomic: export A4-A8,
replace the v2 runner names, migrate Rust/Python/Studio/examples/tests, and
ship the required changelog and migration guide. Private M2 names carry no
source- or binary-compatibility promise and may be adapted when A6 lands.

This deliberately supersedes the original plan's public M2.4 cut. The
plan's `StreamingRunner::start`, v2 runner deletion, and public source-driven
acceptance gate are not checked off here. The only allowed completion label for
this slice is “M2 runtime internals complete”; public A6 remains outstanding.

### Decision 2 - unary control only; multi-ingress fails closed

**Approved choice for OQ2:** M2 implements runtime-owned unary control
forwarding and rejects `Watermark` or `Idle` at every multi-ingress operator
before downstream observation.

- Unary `Watermark(t)`: validate monotonicity at the source, call
  `on_watermark(t)`, and forward `t` only after the handler succeeds.
- Unary `Idle`: forward the runtime-created idle marker in FIFO order. A
  following data message is sufficient to mark the unary ingress active;
  M2 does not classify late data.
- Multi-ingress `Watermark` or `Idle`: return a typed existing
  `InvalidArgument` failure before calling an operator handler or sending any
  control message on an output edge. Data and all-ended EOF behavior remain
  supported for union and other multi-ingress tasks.
- Any `Barrier` in an M2 production or test-only operator path fails closed
  before handler or output. Barrier coordination belongs to M5.

This deliberately sacrifices M2 watermark availability for a two-source
union. It prevents the unsafe alternatives: arrival-order forwarding,
last-arrival, maximum, or arbitrary forwarding. The M3 implementation later
replaces the private rejection branch with the frozen per-ingress minimum,
idle exclusion, and reactivation rules; no public signature changes.

### Decision 3 - ordinary sink means process-local ordered delivery

**Approved choice for OQ3:** M2 ordinary sinks provide bounded,
process-local, ordered delivery only. The runtime may report completed writes,
drain, closure, and whether a source is replayable. It must not derive or
report an end-to-end at-least-once guarantee across process failure.

In a fault-free process, one output task delivers batch `N` to its configured
sinks in order before beginning batch `N + 1`; successful natural completion
or graceful shutdown drains accepted data and closes every sink. Explicit
cancel has no drain promise. A connector may tolerate duplicate writes, but
without a barrier-aligned durable source cursor and aligned manifest that
property is not an engine delivery guarantee. The first honest at-least-once
claim for the continuous runner belongs to M5's S9 capability derivation.

### Decision 4 - narrowly supersede S10 for finite envelope admission

**Authorized B6/B7 choice, pending the critic gate:** every queued
`StreamMessage` consumes exactly one finite slot, whether it is `Data`,
`Watermark`, `Idle`, `EndOfInput`, or a test-only injected `Barrier`. `Data`
also retains its independent measured row and byte charges. An empty table or
external batch whose measured cost is zero rows and zero bytes therefore still
consumes one slot.

M2 reuses the existing positive `EdgeBudget.max_rows` value as both the slot
ceiling and the row ceiling, applied independently. It does not add a
`max_messages` field, control-specific constructor, runner option, or other
public configuration/control surface. Admission is one atomic predicate and
queue commit:

```text
queued_envelopes + 1           <= R
charged_rows + message_rows    <= R
charged_bytes + message_bytes  <= B
```

For a public `EdgeBudget { max_rows: R, max_bytes: B }`, all three checked sums
must fit. A direct `edge_channel` caller chooses
`R >= max(required_row_limit, required_simultaneous_messages)` and
retains the required byte ceiling.
Existing struct literals, constructors, and call sites remain source-compatible;
data-only traffic in which every envelope charges at least one row retains its
prior effective limits, while zero-cost sends may now park earlier.

A blocked send owns no reservation. Dequeue releases its slot, row, and byte
charges exactly once before waking the sender; a full slot dimension therefore
parks a zero-row/zero-byte message until one queued envelope is removed.
Receiver close, explicit cancellation, and task error release every queued
charge and wake blocked sends during convergence; graceful shutdown drains
every accepted queue item, committed source slot, and already-started edge send
before ordered EOF. The same invariant applies to internal edges, synthesized
source/sink boundary edges, and the existing public `edge_channel` primitive.

This decision supersedes only total-spec S10.1/S10.5, FR23, the edge-budget
inputs row, AC-S10, and their matching projections to the extent necessary for
that third predicate. S10.2's row/byte accounting and checked arithmetic,
S10.3's oversize/pre-open validation, S10.4's backpressure/loss-policy rules,
and the FIFO, wakeup, and single-producer remainder of S10.5 do not change.

## Approved M2 visibility and signatures

All items in this section are `pub(crate)`. None is re-exported by
`runtime::streaming`, `runtime`, or `lib.rs`. The exact module split may be
narrowed by the implementer, but the ownership, arguments, outcomes, and
visibility below are normative for M2.

### Consuming runtime-plan projection

`StreamExecutionPlan` gains one consuming crate-private projection. No public
method or return value changes:

```rust
impl StreamExecutionPlan {
    pub(crate) fn into_runtime_parts(
        self,
        default_budget: EdgeBudget,
    ) -> Result<StreamRuntimePlanParts>;
}

pub(crate) struct StreamRuntimePlanParts {
    pub(crate) name: String,
    pub(crate) fingerprint: String,
    pub(crate) requirements: StreamRequirements,
    pub(crate) nodes: Vec<RuntimeStreamNode>,
    pub(crate) edges: BTreeMap<String, RuntimeEdge>,
    pub(crate) source_routes: BTreeMap<String, RuntimeSourceRoute>,
    pub(crate) sink_routes: BTreeMap<String, RuntimeSinkRoute>,
}

pub(crate) struct RuntimeStreamNode {
    pub(crate) node_id: String,
    pub(crate) operator: CompiledStreamOperator,
    pub(crate) input_ports: BTreeMap<String, Port>,
    pub(crate) output_ports: BTreeMap<String, Port>,
    pub(crate) ingress_edges: BTreeMap<String, String>,
    pub(crate) output_edges: BTreeMap<String, Vec<String>>,
}

pub(crate) struct RuntimeSourceRoute {
    pub(crate) binding_id: String,
    pub(crate) target: PortEndpoint,
    pub(crate) edge_id: String,
}

pub(crate) struct RuntimeSinkRoute {
    pub(crate) output_id: String,
    pub(crate) source: PortEndpoint,
    pub(crate) edge_id: String,
}

pub(crate) struct RuntimeEdge {
    pub(crate) stable_id: String,
    pub(crate) kind: RuntimeEdgeKind,
    pub(crate) producer: RuntimeProducer,
    pub(crate) consumer: RuntimeConsumer,
    pub(crate) budget: EdgeBudget,
}
```

The stream compiler stores private stream nodes as directly owned
`RuntimeStreamNode`-compatible values. It does not reuse the batch plan's
`Arc<Mutex<CompiledNode<_>>>` representation. Consuming the plan therefore
moves each non-`Clone` `CompiledStreamOperator` into exactly one operator task;
`Arc::try_unwrap`, lock-dependent extraction, and operator cloning are
forbidden. The existing public accessors continue to borrow the same logical
topology and retain their current results.

Internal compiled-edge IDs remain the existing
`source.node.port->target.node.port` value. Synthesized boundary IDs use
lowercase hexadecimal UTF-8 components:

```text
source/{hex(binding_id)}/{hex(target.node_id)}/{hex(target.port)}
sink/{hex(source.node_id)}/{hex(source.port)}/{hex(output_id)}
```

Hex components are self-delimiting by `/`; identifiers contain no `/` after
hex encoding. Projection validates uniqueness across the full internal plus
boundary ID set. `RuntimeEdgeKind::{SourceBoundary, Internal, SinkBoundary}`
and the structured producer/consumer endpoints retain readable identities for
diagnostics, so code never parses an ID back into topology. Every source and
sink boundary is a real bounded edge. Its unchanged two-field `EdgeBudget`
drives atomic slot/row/byte admission, and the edge participates in status,
partial fan-out evidence, backpressure, and enqueue/dequeue metrics.

### Bindings, preflight, and three-stage launch

The current private A4-shaped source types remain the basis. M2 accepts a
vector boundary that can detect duplicates before collection into a map:

```rust
pub(crate) struct NamedSourceBinding {
    pub(crate) binding_id: String,
    pub(crate) binding: SourceBinding,
}

pub(crate) struct NamedSinkBinding {
    pub(crate) output_id: String,
    pub(crate) sink_id: String,
    pub(crate) binding: OrdinarySinkBinding,
}

pub(crate) struct ContinuousJobSpec {
    pub(crate) context: StreamJobContext,
    pub(crate) plan: StreamExecutionPlan,
    pub(crate) sources: Vec<NamedSourceBinding>,
    /// Vector order is the configured delivery order per output.
    pub(crate) sinks: Vec<NamedSinkBinding>,
    pub(crate) edge_budget: EdgeBudget,
    /// Required explicit acknowledgement that this private run has no
    /// cross-process delivery guarantee.
    pub(crate) delivery_mode: M2DeliveryMode,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) enum M2DeliveryMode {
    ProcessLocalOrdered,
}

pub(crate) fn preflight_job(
    spec: ContinuousJobSpec,
) -> Result<ValidatedContinuousJob>;
```

`preflight_job` consumes `plan.into_runtime_parts(edge_budget)` and is a pure
ownership transform. It checks the complete source and sink route sets,
identifier validity, every capability, first-hop source budget, every
compiled/boundary edge budget, and unsupported delivery requests before it
spawns a task or calls a lifecycle method. It samples each source and sink
capability exactly once and freezes the result into `ValidatedContinuousJob`.
The returned value uses `BTreeMap` and compiled stable IDs; it never retains
caller-owned mutable containers.

The required `M2DeliveryMode::ProcessLocalOrdered` prevents internal call
sites from accidentally treating the run as fulfillment of the public
`StreamRequirements` delivery request. The M2 runner rejects every explicit
`ExactlyOnce` entry. It may execute a topology whose absent delivery entry
has the current public `AtLeastOnce` default only because this separate,
crate-private mode explicitly downgrades the run and no guarantee is exposed;
the future public A6 runner must enforce the full S9 matrix instead.

Launch then has exactly three phases and two independent gates:

1. Pure preflight above: no task and no source/sink/operator lifecycle call.
2. The runner-core launch driver registers every operator task before any is
   first polled. Releasing `OperatorEntryGate` lets each task call exactly one
   fresh-job `reset()` inside the task and return an `OperatorEntryAck`. A
   successful task then parks on `DataControlGate`. The driver waits for every
   entry ack. Any entry error is primary, cancels and joins all registered
   operator tasks, and leaves every source/sink open and close canary untouched.
3. Only after all entry acks succeed does the driver concurrently open every
   source at its configured cursor and every ordinary sink. It records one
   stable `ConnectorOpenAck` per began-open resource. Every open success is
   parked, not allowed to poll/write. After every open succeeds, the start
   observer atomically claims the job handle and the driver releases
   `DataControlGate` for source pumps/tasks, operator ingress loops, and sink
   writes.

```rust
pub(crate) struct OperatorEntryGate { /* one-shot broadcast */ }
pub(crate) struct DataControlGate { /* one-shot broadcast */ }

pub(crate) struct OperatorEntryAck {
    pub(crate) task_id: TaskId,
    pub(crate) node_id: String,
    pub(crate) result: Result<()>,
}

pub(crate) struct ConnectorOpenAck {
    pub(crate) origin: FailureOrigin,
    pub(crate) result: Result<()>,
}
```

An open error or dropped start observer requests driver convergence. Every
resource whose `open` began, including the resource whose `open` returned
`Err`, receives exactly one `close`. Close failures are aggregated as bounded
runner/reaper diagnostics in stable origin order; the data/control gate is
never released.

The private source shape remains:

```rust
pub(crate) struct Cursor {
    order: Vec<u8>,
    payload: JsonMap,
}

pub(crate) enum SourceEvent {
    Data { batch: Batch, cursor: Cursor },
    Watermark(EventTime),
    Idle,
}

pub(crate) struct SourceCapabilities {
    pub(crate) replayable: bool,
    pub(crate) max_batch_rows: usize,
    pub(crate) max_batch_bytes: usize,
}

#[async_trait]
pub(crate) trait StreamSource: Send {
    async fn open(&mut self, cursor: Option<Cursor>) -> Result<()>;
    async fn next(&mut self) -> Result<Option<SourceEvent>>;
    async fn close(&mut self) -> Result<()>;
    fn capabilities(&self) -> SourceCapabilities;
}
```

The D3 rustdoc contract on `next`/`close` remains mandatory. One validated
binding still owns exactly one pump, one source task, and one capacity-one
slot. M2 does not add a public `SourceEvent::End`; the first `None` remains
the sole source of one ordered EOF per output.

Graceful drain uses a source-task request separate from cancellation and this
serialized runtime-owned state:

```rust
pub(crate) enum SourceAcceptState {
    Polling,
    Slotted(PumpEvent),
    Draining(Option<PumpEvent>),
    Closed,
}
```

Only the atomic `Polling -> Slotted(event)` slot commit accepts an event;
connector readiness and `Poll::Ready` observation alone do not. Slot commit
and the drain cut use the same synchronization point. The cut performs
`Polling -> Draining(None)` or `Slotted(event) -> Draining(Some(event))`.
`Draining` forbids any later slot commit, drains the retained slot and any
already-started edge send, emits one EOF, closes and joins the source units,
then becomes `Closed`. A non-committed poll is dropped once and never resumed.
Explicit cancel may discard a slot and carries no drain promise.

### Binding-qualified cursor and watermark diagnostics

Cursor order is strict per source binding: each accepted data cursor must be
greater than the preceding cursor. A repeat or regression fails before the
offending batch is enqueued to any branch. The error is
`CalcFlowError::InvalidArgument` with machine-readable field
`sources.<binding_id>.cursor`; the binding identity may also appear in the
message, but it must not exist only there.

Source-provided watermarks are non-decreasing per binding. Equality is allowed;
a value less than the preceding watermark fails before any branch enqueue. The
error is `CalcFlowError::InvalidArgument` with machine-readable field
`sources.<binding_id>.watermark`. This validation does not generate a
watermark, expose progress publicly, or add M3 minimum/idle/reactivation
semantics.

At code-approved head `7deda2a0`, `source_task.rs` reports the exact fields
`sources.<binding_id>.cursor` and `sources.<binding_id>.watermark`, and focused
tests pin their pre-enqueue behavior. G1 is closed without a public signature
or control API.

### Operator task and validating collector

```rust
pub(crate) struct OperatorTaskInputs {
    pub(crate) node_id: String,
    pub(crate) operator: CompiledStreamOperator,
    /// Compiled ingress name -> its single owned receiver.
    pub(crate) ingresses: BTreeMap<String, EdgeReceiver>,
    /// Output port -> independently charged fan-out senders.
    pub(crate) outputs: BTreeMap<String, Vec<EdgeSender>>,
    pub(crate) output_ports: BTreeMap<String, Port>,
    pub(crate) context: StreamTaskContext,
    pub(crate) progress: OperatorProgress,
    pub(crate) metrics: MetricsRecorder,
    pub(crate) entry_gate: OperatorEntryGate,
    pub(crate) entry_ack: OperatorEntryAckSender,
    pub(crate) data_gate: DataControlGate,
}

pub(crate) fn spawn_operator_task(
    supervisor: &mut TaskSupervisor,
    inputs: OperatorTaskInputs,
) -> Result<TaskId>;

pub(crate) struct ChannelStreamCollector<'a> {
    node_id: &'a str,
    output_ports: &'a BTreeMap<String, Port>,
    outputs: &'a mut BTreeMap<String, Vec<EdgeSender>>,
    metrics: &'a MetricsRecorder,
}

#[async_trait]
impl StreamCollector for ChannelStreamCollector<'_> {
    async fn emit(&mut self, port: &str, batch: Batch) -> Result<()>;
}
```

`ChannelStreamCollector::emit` validates the port, batch kind, exact schema,
and the cost against every fan-out budget before the first send. It then
awaits each independently charged sender; immutable `Batch` clones share the
payload. Each successful queue commit advances that edge's enqueue counters;
the operator fully-fanned-out counter advances only after all sends for that
`emit` return successfully. There is no user-space output queue.

Fan-out is not transactional. If branch one commits and branch two closes,
branch one retains its observable prefix, edge one reports the enqueue, the
operator fully-fanned-out counter remains unchanged, and convergence performs
no rollback. The same rule applies when an external operator completes one
`emit` and then returns an error from the handler. Only validation/cost failure
before the first send guarantees zero downstream data.

The operator task owns its operator and lazily owns at most one
operator-scoped `DataFusionRuntime` for expression/SQL. Union-only and
array-only tasks construct none. It awaits only non-ended ingress receivers,
preserves each receiver's FIFO, and makes no cross-ingress order promise.

Its control dispatch is fixed for M2:

```rust
match message.kind() {
    StreamMessageKind::Data => process_data_then_validate_and_send(...).await,
    StreamMessageKind::Watermark if ingresses.len() == 1 => {
        operator.on_watermark(...).await?;
        forward_watermark(...).await
    }
    StreamMessageKind::Idle if ingresses.len() == 1 => {
        forward_idle(...).await
    }
    StreamMessageKind::Watermark | StreamMessageKind::Idle => {
        fail_multi_ingress_control_before_output(...)
    }
    StreamMessageKind::Barrier => fail_m2_barrier_before_output(...),
    StreamMessageKind::EndOfInput => mark_ended_and_maybe_finish(...).await,
}
```

After all ingresses end, `on_end` runs exactly once and must succeed before
the runtime forwards one EOF on every output. An operator error never causes
the runtime to forward the current control event.

Every ingress tracks `saw_explicit_eof` separately from channel closure.
While the job is `Running` or `Draining`, `recv() == None` before explicit EOF
is `EdgeClosed` with `FailureOrigin::Ingress { node_id, ingress, edge_id }`;
it never enters the all-ended path. Once the convergence driver has marked
the edge runtime-closed, `None` is only a wakeup and records no new error. An
unexpected close already observed before a competing cause was committed is
retained as a stable secondary. Sink ingress follows the same rule.

### Ordinary sink binding and task

M2 uses a smaller private trait rather than prematurely exporting A5's
checkpoint-valued `flush`, pre-commit, or transactional types:

```rust
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) enum M2SinkDelivery {
    /// Classification only; not an end-to-end delivery guarantee.
    ProcessLocalOrdered,
}

#[async_trait]
pub(crate) trait OrdinaryStreamSink: Send {
    /// Sampled exactly once during whole-job preflight.
    fn delivery_capability(&self) -> M2SinkDelivery {
        M2SinkDelivery::ProcessLocalOrdered
    }

    async fn open(&mut self) -> Result<()>;
    async fn write(&mut self, batch: &Batch) -> Result<()>;
    async fn close(&mut self) -> Result<()>;
}

pub(crate) struct OrdinarySinkBinding {
    sink: Box<dyn OrdinaryStreamSink>,
}

pub(crate) struct SinkTaskInputs {
    pub(crate) output_id: String,
    /// Stable sink IDs in configured delivery order.
    pub(crate) sinks: Vec<ValidatedOrdinarySink>,
    pub(crate) input: EdgeReceiver,
    pub(crate) context: StreamTaskContext,
    pub(crate) progress: SinkProgress,
    pub(crate) metrics: MetricsRecorder,
}

pub(crate) fn spawn_sink_task(
    supervisor: &mut TaskSupervisor,
    inputs: SinkTaskInputs,
) -> Result<TaskId>;
```

There is one sink task per external output. For each data message it awaits
`write` in configured sink order. Only after sink `k` succeeds does it call
sink `k + 1`; only after all sinks succeed do delivered counters advance and
the task receive the next message. EOF closes every opened sink and completes
the task. Explicit cancel or failure drops the one in-flight method only at
teardown, never resumes that sink, closes every opened sink, and preserves
close failures as secondary failures in stable sink-ID order.

`OrdinaryStreamSink::close` must tolerate a dropped `open` or `write` future
and release external resources without requiring that future to finish. A
sink implementation may not spawn unregistered Tokio tasks. Connector errors
must already be sanitized; the runtime retains the original `CalcFlowError`
with a private `FailureOrigin::Sink { output_id, sink_id }` rather than
rewriting it into an inaccurate public error variant.

M2 never calls `flush(Epoch)`, creates `SinkPreCommit`, or accepts a
`TransactionalSink`. Those remain exactly the future public A5/M5 surface.

### Public panic variant and bounded capture

The PR-added public error shape and display remain:

```rust
CalcFlowError::TaskPanicked { task_id: u64, message: String }
// "task <task_id> panicked: <message>"
```

Every M2 panic-capture path, including supervised tasks and connector open,
uses one crate-private conversion helper. A string payload of at most 1,024
UTF-8 bytes is preserved unchanged. For a longer string, the helper reserves
three bytes for `…`, keeps the longest prefix ending on a UTF-8 character
boundary within the remaining 1,021 bytes, and appends the ellipsis. The stored
message, including the ellipsis, is therefore valid UTF-8 and at most 1,024
bytes. A non-string payload remains exactly `non-string panic payload`.

At code-approved head `7deda2a0`, the shared `panic_message` helper is used by
supervision and connector lifecycle capture and implements the common bound.
ASCII, multibyte-boundary, short, and non-string tests close G2 while
preserving the public variant's fields and display.

### Private failure origins and start-failure aggregation

No additional public error variant is added merely to preserve internal
component identity. M2 wraps the typed error in a private, immutable record:

```rust
#[derive(Clone, Debug, Eq, Ord, PartialEq, PartialOrd)]
pub(crate) enum FailureOrigin {
    Preflight,
    OperatorEntry { node_id: String },
    SourceOpen { binding_id: String },
    SinkOpen { output_id: String, sink_id: String },
    SourceClose { binding_id: String },
    SinkClose { output_id: String, sink_id: String },
    Ingress { node_id: String, ingress: String, edge_id: String },
    Task { task_id: TaskId, task_name: String },
    Metrics { component_id: String, counter: &'static str },
}

pub(crate) struct RuntimeFailure {
    pub(crate) origin: FailureOrigin,
    pub(crate) error: CalcFlowError,
}

pub(crate) struct StartFailure {
    pub(crate) primary: Arc<RuntimeFailure>,
    /// Identifies the bounded runner diagnostic containing cleanup failures.
    pub(crate) diagnostic_id: Option<u64>,
}

pub(crate) type StartResult<T> = std::result::Result<T, StartFailure>;
```

Operator-entry and connector-open failures select their primary at the
launch driver's terminal decision point. The first observed scheduling round
is ordered by phase and then stable `FailureOrigin`; later errors cannot
replace it. An entry failure joins the entry tasks and has no connector
cleanup. An open failure marks every resource whose open began, including the
resource that returned `Err`, for exactly one close. After all close units and
the launch driver join, `StartObserver` returns the primary; close failures
are retained once in stable origin order in the referenced runner diagnostic.

```rust
pub(crate) struct RunnerDiagnosticsSnapshot {
    pub(crate) records: Vec<Arc<RunnerDiagnosticRecord>>,
    pub(crate) truncated_records: u64,
    pub(crate) diagnostics_overflowed: bool,
}

pub(crate) struct RunnerDiagnosticRecord {
    pub(crate) id: u64,
    pub(crate) launch_id: LaunchId,
    pub(crate) cleanup_failures: Vec<Arc<RuntimeFailure>>,
    pub(crate) failures_truncated: bool,
}
```

The runner retains at most 64 launch/reaper records and 64 cleanup failures
per record. Eviction is oldest-record-first and increments a bounded,
non-failing truncation indicator. A dropped start observer has no returned
`StartFailure`, so the same diagnostic record is its sole error report. For a
running sink task, the write/open failure remains the job primary and every
close failure is an outcome secondary in stable sink-ID order.

`LaunchId` is a private runner-allocated correlation key, distinct from the
job ID supplied by `ContinuousJobSpec::context`. It is used only for lifecycle
registry membership and runner/reaper diagnostics:

```rust
#[derive(Clone, Copy, Debug, Eq, Ord, PartialEq, PartialOrd)]
pub(crate) struct LaunchId(u64);
```

`ContinuousJob::id()`, `ContinuousJobStatus::job_id`, and every derived
`StreamTaskContext` use `ContinuousJobSpec::context.job_id`. They never expose
or substitute the internal `LaunchId`. Conversely, provisional/live/reaper
registry correlation and `RunnerDiagnosticRecord::launch_id` use `LaunchId`
so two launches cannot alias merely because callers reuse a context job ID.

### Terminal decision and deadline outcome

The current consuming `ContinuousJob::{wait,cancel}(self)` is replaced by an
idempotent observer over one runner-core-registered driver. M2 uses an
explicit cause so deadline expiry is never inferred from an empty task-error
vector and never borrows M5's `CheckpointTimeout`:

```rust
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) enum ContinuousJobState {
    Running,
    Draining,
    Completed,
    Cancelled,
    Failed,
    RecoveryRequired,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub(crate) enum TerminalCause {
    NaturalEnd,
    GracefulShutdown,
    ExplicitCancel,
    DeadlineExceeded,
    TaskFailure { primary_task_id: TaskId },
}

pub(crate) struct ContinuousJobOutcome {
    pub(crate) state: ContinuousJobState,
    pub(crate) cause: TerminalCause,
    /// Primary first, then same-round and convergence failures in stable ID
    /// order. FailureOrigin preserves source/node/output/sink identity.
    pub(crate) errors: Vec<Arc<RuntimeFailure>>,
}
```

The approved deadline terminal outcome is:

```text
state = Cancelled
cause = DeadlineExceeded
errors = [] unless teardown adds secondary failures
```

A job-level deadline is caller policy, not an M5 checkpoint timeout and not
an infrastructure-recovery claim. `CalcFlowError::CheckpointTimeout` is not
added or reused. The serialized decision point applies S8.4 priority within
one scheduling round: already-observed task failure/panic, then explicit
cancel, then deadline. Therefore simultaneous explicit cancel plus deadline
stores `ExplicitCancel`; simultaneous task failure plus either stores
`TaskFailure`. A task returning `CalcFlowError::Cancelled` during convergence
is not promoted to a primary task failure. If it observes the absolute
deadline before the deadline watcher, it submits `DeadlineExceeded` to the
same decision point.

Natural end or shutdown request first transitions `Running -> Draining`.
Completion is stored only after all accepted messages are delivered, all
sinks close, and every task joins. In this private M2 surface, `Completed`
means process-local drain only; it contains no epoch and is not the public
A6/S8 durable-completion promise.

### Core-owned launch/convergence driver and registry ownership

One long-lived `RunnerLifecycleDriver` is registered when the private runner
is created. It owns the `JoinSet` of provisional/job `JobDriver` tasks and is
the only unit that polls and joins them. Each `JobDriver` owns its
`TaskSupervisor`, launch resources, partially progressed worker joins, and
terminal arbiter for its whole lifetime. No caller future can take either
registry.

```rust
pub(crate) struct ContinuousRunner {
    core: Arc<RunnerCore>,
    closed: bool,
}

pub(crate) struct ContinuousJob {
    core: Arc<JobCore>,
    /// Non-Clone ownership token; Drop performs the reaper transition.
    ownership: JobOwnershipToken,
}

pub(crate) struct RunnerCore {
    commands: RunnerCommandSender,
    registry: Mutex<RunnerRegistryState>,
    driver: Mutex<RunnerDriverSlot>,
    diagnostics: RunnerDiagnostics,
}

pub(crate) struct JobCore {
    /// Copied from ContinuousJobSpec::context.job_id and exposed by the job,
    /// status, and derived task contexts.
    job_id: u64,
    commands: JobCommandSender,
    state: Mutex<JobCoreState>,
    changed: Notify,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) enum DriverOwnership {
    /// Registered in RunnerCore; its start gate has not yet been released.
    CoreOwned,
    /// RunnerLifecycleDriver is polling the registered JobDriver.
    Driving,
    /// Owning job/start observer disappeared; reaper diagnostics own result.
    ReaperOwned,
    /// RunnerLifecycleDriver joined JobDriver and published one outcome.
    Terminal,
}

pub(crate) struct JobCoreState {
    owner: DriverOwnership,
    launch_delivery: LaunchDeliveryState,
    selected_cause: Option<TerminalCause>,
    outcome: Option<Arc<ContinuousJobOutcome>>,
}

pub(crate) enum LaunchDeliveryState {
    Provisional,
    ReadyUnclaimed,
    Claimed,
    CancelRequested,
    Failed,
}

pub(crate) struct RunnerRegistryState {
    provisional: Option<LaunchId>,
    live_jobs: BTreeMap<LaunchId, Arc<JobCore>>,
    reaper_jobs: BTreeSet<LaunchId>,
    pending_start: Option<LaunchId>,
    shutting_down: bool,
}

impl ContinuousRunner {
    pub(crate) fn new() -> Self;
    pub(crate) fn diagnostics(&self) -> RunnerDiagnosticsSnapshot;

    /// A cancellation-safe observer. Pure preflight may happen before driver
    /// installation; no lifecycle begins until RunnerCore owns the registered
    /// JobDriver.
    pub(crate) fn start(
        &self,
        spec: ContinuousJobSpec,
    ) -> StartObserver;

    /// Submits shutdown synchronously; polling only observes core progress.
    pub(crate) fn shutdown(&mut self) -> RunnerShutdownObserver<'_>;
}

impl ContinuousJob {
    pub(crate) fn id(&self) -> u64;
    pub(crate) fn status(&self) -> ContinuousJobStatus;

    /// Observes only; dropping this future changes no cause or ownership.
    pub(crate) fn wait(&self) -> OutcomeObserver;
    /// Submits GracefulShutdown before returning the observer.
    pub(crate) fn shutdown(&self) -> OutcomeObserver;
    /// Submits ExplicitCancel before returning the observer.
    pub(crate) fn cancel(&self) -> OutcomeObserver;
}

pub(crate) struct StartObserver { /* custom Future + cancel-on-Drop guard */ }
pub(crate) struct OutcomeObserver { /* custom Future over JobCore::changed */ }
pub(crate) struct RunnerShutdownObserver<'a> { /* borrows runner mutably */ }

impl Future for StartObserver {
    type Output = StartResult<ContinuousJob>;
}

impl Future for OutcomeObserver {
    type Output = Arc<ContinuousJobOutcome>;
}

impl Future for RunnerShutdownObserver<'_> {
    type Output = Result<()>;
}
```

`ContinuousRunner::start` accepts a fresh owned spec, so one core can run
several jobs serially without connector/operator cloning. Before installing a
new launch, `RunnerLifecycleDriver` joins any abandoned provisional driver and
all reaper entries. A concurrent non-abandoned provisional/live job returns
the existing active-job `Conflict`.

`StartObserver` may temporarily own only an unopened preflight input. Before
operator entry, `RunnerCore` creates `JobCore`, inserts `CoreOwned`, and
registers `JobDriver` in the runner driver's `JoinSet` before releasing its
start gate. The transition `CoreOwned -> Driving` linearizes at that gate
release. From this point, dropping `StartObserver` locks `JobCoreState` and
performs `Provisional|ReadyUnclaimed -> CancelRequested`, submits explicit
cancel, and marks the driver `ReaperOwned`; connector objects and registries
remain in `JobDriver`. It never tries to run async cleanup from `Drop`.

After all connector-open acks succeed, the driver changes launch delivery to
`ReadyUnclaimed` but keeps `DataControlGate` closed. A successful observer
poll performs `ReadyUnclaimed -> Claimed`, creates the non-Clone ownership
token, and requests data-gate release under the same lock before returning
`Poll::Ready(Ok(job))`. This is the handle-delivery linearization point. If
observer Drop wins the lock, cancellation wins and no unobserved live job is
released. A drop after `Claimed` is the returned job handle's responsibility.

`wait`, `cancel`, and `shutdown` observers never poll a `JoinHandle` and never
own the supervisor. Cause submission happens synchronously before `cancel()`
or `shutdown()` returns its observer; dropping that observer does not retract
the cause. `wait()` submits nothing. The runner driver remains responsible for
polling `JobDriver` regardless of observer existence.

The job driver's terminal-cause commit is the first linearization point: it
stores one cause after applying task-failure > explicit-cancel > deadline for
the observed scheduling round. It then closes entry senders, converges and
joins every worker, performs non-recursive error accounting, and returns a
`DriverReport`. `RunnerLifecycleDriver` joins that registered `JobDriver` and,
under one `JobCoreState` lock, stores `Arc<ContinuousJobOutcome>`, changes the
owner to `Terminal`, and notifies observers. That atomic store/state change is
the outcome-publication linearization point; observers can never see a
terminal state without the immutable outcome or an outcome before driver join.

Dropping the claimed `ContinuousJob` before `Terminal` commits explicit
cancel and performs `CoreOwned|Driving -> ReaperOwned` under the same state
lock. This is the registry-transfer linearization point. The physical
`JobDriver` registration never leaves `RunnerLifecycleDriver`; `ReaperOwned`
changes who observes/retains the joined result, eliminating an async handle
move from `Drop`. At `Terminal`, job Drop is a no-op. In a race, whichever of
Drop's `ReaperOwned` transition or outcome publication acquires the lock first
defines the ordering; the latter always ends in `Terminal` with one outcome.

Runner shutdown first closes start admission, cancels and joins a provisional
launch, then submits cancel to live jobs, waits for their driver joins, and
finally drains reaper diagnostics. Concurrent job Drop may change a live
driver to `ReaperOwned`, but cannot move or lose its registration. The runner
lifecycle `JoinHandle` itself stays inside `RunnerCore::driver`; the custom
shutdown observer polls it in place under the driver-slot lock. Dropping that
observer leaves the handle in `RunnerCore`, so a later `shutdown()` resumes
and joins it. Only then is the runner marked closed.

Dropping `ContinuousRunner` without completed shutdown fires root
cancellation, increments `abandoned_runner_drops`, and logs one payload-free
diagnostic. This is the sanctioned abandon-joins path; only completed runner
shutdown carries the no-task-outlives guarantee. `abort_all` remains a
last-resort destructor safety net after cancellation, never a normal path.

These states are observable in adversarial tests but remain crate-private.
No handle-local `owns_registry` boolean exists.

### Status and metrics

Status and metrics are private deterministic values. They contain stable IDs
and numeric progress only; opaque cursor payloads, batch metadata attributes,
rows, secrets, and batch IDs never enter either surface.

```rust
#[derive(Clone, Debug, Eq, PartialEq)]
pub(crate) struct ContinuousJobStatus {
    pub(crate) job_id: u64,
    pub(crate) state: ContinuousJobState,
    pub(crate) terminal_cause: Option<TerminalCause>,
    pub(crate) tasks: BTreeMap<TaskId, TaskStatus>,
    pub(crate) edges: BTreeMap<String, ChannelMetrics>,
    pub(crate) sources: BTreeMap<String, SourceStatus>,
    pub(crate) nodes: BTreeMap<String, OperatorStatus>,
    pub(crate) sinks: BTreeMap<String, SinkStatus>,
    pub(crate) metrics: M2MetricsSnapshot,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub(crate) struct SourceStatus {
    pub(crate) replayable: bool,
    /// Ordered key only. The opaque Cursor.payload is never exposed.
    pub(crate) latest_observed_order: Option<Vec<u8>>,
    pub(crate) durable_order: Option<Vec<u8>>,
    pub(crate) next_sequence: Option<u64>,
    pub(crate) ended: bool,
}

#[derive(Clone, Debug, Default, Eq, PartialEq)]
pub(crate) struct M2MetricsSnapshot {
    pub(crate) job: JobMetrics,
    pub(crate) edges: BTreeMap<String, EdgeRuntimeMetrics>,
    pub(crate) sources: BTreeMap<String, SourceMetrics>,
    pub(crate) nodes: BTreeMap<String, OperatorMetrics>,
    pub(crate) sinks: BTreeMap<String, SinkMetrics>,
}

#[derive(Clone, Debug, Default, Eq, PartialEq)]
pub(crate) struct EdgeRuntimeMetrics {
    /// Fixed from RuntimeEdge::budget.max_rows during registry construction.
    pub(crate) message_slot_limit: usize,
    pub(crate) channel: ChannelMetrics,
    pub(crate) input_batches: u64,
    pub(crate) input_rows: u64,
    pub(crate) input_bytes: u64,
    pub(crate) output_batches: u64,
    pub(crate) output_rows: u64,
    pub(crate) output_bytes: u64,
    pub(crate) drop_invariant_violated: bool,
}

#[derive(Clone)]
pub(crate) struct MetricsRecorder { /* bounded stable-ID registry */ }

impl MetricsRecorder {
    pub(crate) fn new(
        edge_budgets: impl IntoIterator<Item = (String, EdgeBudget)>,
        sources: impl IntoIterator<Item = String>,
        nodes: impl IntoIterator<Item = String>,
        sinks: impl IntoIterator<Item = String>,
    ) -> Self;

    pub(crate) fn snapshot(&self) -> M2MetricsSnapshot;
    // Every checked record operation accepts IDs and numeric costs/durations,
    // never a cursor payload, row payload, attribute map, or arbitrary label.
    pub(crate) fn record_source_poll(&self, source_id: &str) -> Result<()>;
    pub(crate) fn record_operator_input(
        &self,
        node_id: &str,
        cost: EnvelopeCost,
    ) -> Result<()>;
    pub(crate) fn record_operator_output(
        &self,
        node_id: &str,
        cost: EnvelopeCost,
    ) -> Result<()>;
    pub(crate) fn record_sink_delivery(
        &self,
        sink_id: &str,
        cost: EnvelopeCost,
        elapsed: Duration,
    ) -> Result<()>;

    /// Called synchronously inside the edge channel's queue-state lock. A
    /// successful return plus the immediately following infallible push is
    /// the enqueue linearization point; overflow occurs before queue commit.
    pub(super) fn record_edge_enqueue(
        &self,
        edge_id: &str,
        traffic: EdgeTraffic,
    ) -> Result<()>;

    /// Called inside the same lock immediately before the infallible pop and
    /// reservation release. Overflow leaves the message queued.
    pub(super) fn record_edge_dequeue(
        &self,
        edge_id: &str,
        traffic: EdgeTraffic,
    ) -> Result<()>;

    /// Non-recursive terminal path. Attempts each component error counter
    /// once and returns at most one overflow secondary for the whole outcome.
    /// This method never returns Result and its flag update cannot fail.
    pub(crate) fn account_terminal_errors_once(
        &self,
        failures: &[Arc<RuntimeFailure>],
    ) -> Option<Arc<RuntimeFailure>>;
}
```

`JobMetrics` includes terminal state/cause, task-error count, reaper joins,
reaper join errors, and abandoned-runner drops. `EdgeRuntimeMetrics` includes
the fixed private `message_slot_limit`, the existing consistent
`ChannelMetrics`, and input/output batch, row, and byte counts. The strict
registry is constructed from stable edge IDs paired with their validated
budgets; it copies `budget.max_rows` into `message_slot_limit`. Consequently
the full private status projection is
`status.metrics.edges[edge_id].message_slot_limit` beside
`status.metrics.edges[edge_id].channel`. The existing
`status.edges[edge_id]` convenience projection remains the same
`ChannelMetrics` from that snapshot, and the public `ChannelMetrics` shape does
not gain a field. `SourceMetrics` includes poll count, data batch/row/byte counts,
latest sequence, end state, and errors. `OperatorMetrics` includes input and
fully-fanned-out output counts, processing duration, and errors.
`SinkMetrics` includes successfully delivered batch/row/byte counts, write
duration, and errors. `EdgeTraffic` is derived transiently from the message:
one batch for `Data`, its rows/estimated bytes, and zero batches/rows/bytes
for control. It stores no payload.

Every internal or boundary `EdgeSender` atomically checks the prospective slot,
row, and byte sums under the channel-state lock. If any dimension would exceed
its limit, it records one blocked send and parks without reserving anything.
Otherwise it calls `record_edge_enqueue` and commits the one-slot plus row/byte
reservation with the infallible queue push under that same lock. Every
`EdgeReceiver` calls `record_edge_dequeue` under that lock immediately before
the pop and one-time release. Thus an earlier branch may report one enqueue
while a later branch reports zero and the source/operator fully-fanned-out
counter remains zero. Operator input is recorded after edge dequeue; operator
output after all branch sends; per-sink delivery after that sink write
succeeds. `queue_depth` equals charged messages, and both it and
`high_water_depth` are always at most `message_slot_limit`; current/high-water
row and byte charges remain independently bounded. All gauges come from the
same consistent snapshot.

All ordinary counters are monotone checked `u64` and never wrap. An ordinary
record overflow is the operation's primary failure. Terminal convergence is
different: the driver first freezes cause and primary errors, then calls
`account_terminal_errors_once` in stable component order. A full counter stays
at `u64::MAX`; the method attaches at most one metrics-overflow secondary for
the entire outcome, atomically sets non-failing `metrics_overflowed = true`,
and never accounts that secondary again. Metrics therefore cannot replace the
original primary or recurse. Status contains no current/completed epoch,
watermark progress, or delivery-guarantee field in M2.

### Stress, soak, and benchmark seams

The short stress harness is test-only. Seeds `0..100` feed one fixed pure
seed-to-gate permutation function over named source-ready, edge-release,
sink-release, EOF, drain, and cancel gates. The harness releases exactly that
sequence, prints the seed on failure, and advances paused Tokio time only for
modeled timers; it does not claim Tokio itself has a seeded scheduler. A
separate paused-time unary phase sends at least ten times the configured slot
limit for each of sustained `Idle`, repeated non-regressing watermarks, and a
batch first verified to charge one message, zero rows, and zero bytes while the
receiver is gated. Each shape reaches the exact slot limit, leaves the next send
pending without queue growth, resumes exactly one send per dequeue, preserves
FIFO, and converges with zero charges under graceful shutdown, explicit cancel,
receiver close, and injected upstream task error.

The ignored soak lives in the crate-private `#[cfg(test)]`
`runtime::streaming::soak` module and exposes no public or integration-test
seam. It is exactly `twenty_minute_two_source_slow_sink`, guarded by
`CALC_FLOW_STREAM_SOAK=1`. Its exact and superseding invocation is:

```bash
CALC_FLOW_STREAM_SOAK=1 cargo test -p calc-flow --lib runtime::streaming::soak::twenty_minute_two_source_slow_sink -- --ignored --exact --nocapture
```

Without that environment variable set exactly to `1`, the first test branch
emits a structured disabled result and returns before constructing a runner,
opening a connector, spawning runtime work, or entering the timed loop. The
former `one_hour_*`, duration-override, and
`cargo test -p calc-flow --test stream_soak ...` commands are superseded and
are not valid soak evidence.

The enabled test exercises the real crate-private `ContinuousRunner`, source
tasks, union/operator tasks, two ordered ordinary sinks, supervisor, and
reaper. Normal termination is initiated by graceful job `shutdown()` and must
produce `Completed + GracefulShutdown` after drain; neither owning-job Drop
nor `ExplicitCancel` terminates the 20-minute case. At the drain cut the harness
freezes the accepted per-source sequence maps and total. Each sink's
per-source sequence maps and total must exactly equal the source-accepted
values, with `missing = 0` and `duplicate = 0`. Drop/reaper behavior is covered
by a separate short smoke/stress test rather than being used as the soak's
normal termination path. This test-only seam does not expand public v2,
Python, Studio, or any M3+ surface.

Throughout the enabled workload, at least every fourth data event is a harness-
verified zero-row/zero-byte batch; positive-row and empty batches share the
same accepted-sequence conservation oracle. At least one edge must reach
`high_water_depth == message_slot_limit` and record a blocked send while
processing zero-cost data, or the soak fails for not exercising slot
backpressure.

On Linux it reads `VmRSS` from `/proc/self/status` every ten seconds for an
exactly 1,200-second measured workload and exactly 120 samples. Sample one is
taken after the first interval and sample 120 at the deadline; there is no
time-zero sample. The first five minutes (30 samples) are warm-up; the
remaining 15 minutes feed the RSS slope and median gates. Every sample includes
each edge's `message_slot_limit`, current/high-water depth, current/high-water
rows and bytes, and blocked-send count/duration. The harness emits
commit/kernel/rustc/allocator/RSS-source/cadence/target-duration/warm-up/sample-
count metadata plus conservation and final task/queue/reaper convergence
evidence. A non-Linux run reports `unsupported_platform` and is not passing
soak evidence. The slope, median, task, budget, slot, and loss/duplication
thresholds are exactly M2C-FR32.

The tester retains the complete unfiltered combined stdout/stderr as one UTF-8
log containing newline-delimited JSON metadata, exactly 120 ordered samples,
and one final result. Without committing the log, the tester publishes it and
the `calc-flow.m2-soak-evidence.v1` manifest in SHA-256-verified, newline-
aligned PR comment parts marked `calc-flow-m2-soak-evidence:v1`. The manifest
names PR number, exact 40-character head SHA, literal command, UTC timestamps,
elapsed seconds, exit status, full-log hash, part count, and ordered comment
URLs. `cf-reviewer` must reassemble the literal log, verify all hashes and
record counts, and confirm that manifest SHA, log commit, and then-current PR
head are equal. A summary, selected samples, local path, mutable external link,
disabled result, non-Linux skip, or earlier-head bundle is not evidence; any
later push requires a new run and bundle.

The command ran successfully on Linux at code-approved head `7deda2a0`. The
[durable evidence bundle](https://github.com/wegamekinglc/calc-flow/pull/83#issuecomment-5201266650)
records 120/120 samples, 96,124 accepted batches at each sink, 24,032 accepted
zero-cost batches, zero loss or duplication, zero final task/queue/reaper
leaks, and an RSS slope of -2.609 MiB/hour. The raw log SHA-256 is
`bc97c8f736ad41a4f228e07300f1ecd23c9af9fb09dc1be1718823430bd05f35`.
This closes G3/G8 at that exact code head. The acceptance duration is an
observable test/delivery contract only; it does not justify a public duration
setting, runner control method, or integration-test seam. Any later push
invalidates exact-head evidence and requires a refresh.

Ordinary verification only compiles Criterion. The opt-in baseline is:

```bash
cargo bench -p calc-flow --bench core -- stream
```

Because Criterion is an external target and M2 is private, unary overhead
measures the already-public M1 `StreamOperator` kernel plus public bounded
channel primitives. No internal task/runner is exported for benchmarking.
Channel round-trip and fan-out use the same public primitives. M2 requires
benchmark compilation, not an executed paired comparison. If a reviewer
invokes the 5% comparison, base and head run on the same machine and
environment, and Criterion's bootstrap 95% confidence interval for relative
change in mean execution time decides each case: a lower bound strictly above
`+5.0%` blocks; an upper bound at most `+5.0%` passes; an interval touching or
crossing `+5.0%` is inconclusive. An inconclusive case gets one same-machine
rerun with at least twice the original measurement sample size. If it remains
inconclusive, it is a non-blocking advisory and its point estimate cannot
override the interval. When the comparison is not invoked, point estimates are
advisory and compilation is the only M2 performance gate.

These focused seams are not a repository-verification waiver. The synchronized
exact head must pass every current root-`AGENTS.md` command group, including
Rust/PyO3, Python runner and wildcard-exception compatibility, Studio backend,
frontend/browser/generated contracts, supply chain, helpers, coverage,
rustdoc, and diff checks. The handoff maps each exact-head Action check to
those groups.

## Public compatibility and migration

There is **no new public callable runner API** in the M2 completion diff.
There is one semver-compatible public Rust API **item addition** relative to
`main`: `CalcFlowError::TaskPanicked { task_id: u64, message: String }` with
display `task <task_id> panicked: <message>`. The enum was already
non-exhaustive, so downstream matching already requires a wildcard. Completion
work preserves this variant while bounding only the internally captured
`message`; it does not add or change the signature or fields of a public
constructor, method, trait, runner, status type, or control surface. M2C-FR1A
calls this shape “existing” relative to the historical skeleton and audited
implementation head; that wording does not erase its addition relative to the
comparison baseline.

The existing public bounded-channel signatures also remain structurally
unchanged:

```rust
pub struct EdgeBudget {
    pub max_rows: usize,
    pub max_bytes: usize,
}

pub fn edge_channel(
    edge: impl Into<String>,
    budget: EdgeBudget,
) -> Result<(EdgeSender, EdgeReceiver)>;
```

These signatures and all existing struct literals remain source-compatible,
but their behavior is deliberately observable: for `(R, B)`, at most `R`
envelopes may be queued, charged rows are independently at most `R`, and
charged bytes are at most `B`. A zero-row/zero-byte send can therefore park
earlier than under the old row/byte-only implementation. Direct callers choose
`R >= max(required_row_limit, required_simultaneous_messages)`; no automatic
rewrite or independently tunable message limit is provided.

This source-compatible admission correction is the one other public observable
delta, not another public API item addition. It is authorized only by the
narrow S10.1/S10.5/FR23/inputs/AC-S10 supersession above. `ChannelMetrics`
keeps its current public fields, and `queue_depth`/`high_water_depth` now remain
at most `R`; callers already know the budget they supplied. Only the
crate-private M2 status carries `message_slot_limit` beside the snapshot. No
public continuous-job status API, control constructor, runner, or
project/checkpoint/Python/Studio field is introduced.

- Do not add exports to `crates/calc-flow/src/lib.rs` or public re-exports in
  `runtime::streaming`/`runtime` for any type in this note.
- Do not add a second `StreamingRunner`, alias the v2 type, deprecate v2, or
  change existing v2 signatures, fixtures, or delivery behavior.
- Do not add `EdgeBudget::max_messages`, a control-specific budget, or a
  message-limit method. Document and enforce the independent `max_rows` slot
  ceiling on every `edge_channel` instead.
- Preserve S10.2-S10.4 and the unsuperseded FIFO, close-wakeup,
  no-lost-wakeup, and single-producer portions of S10.5. The new predicate is
  not permission to alter byte estimation, oversize errors, pre-open source
  validation, blocking/loss policies, or message order.
- Do not add `CheckpointTimeout` for M2. The PR-added public `TaskPanicked`
  shape and the comparison-baseline public `EdgeClosed` shape remain
  unchanged.
- Do not add another public error variant for sink identity, multi-ingress
  control rejection, deadline expiry, metrics overflow, reaper joins, or
  internal terminal causes. Use private identity/cause records plus existing
  non-exhaustive `CalcFlowError` variants.
- Do not change Python, Studio, project/checkpoint v2 documents, schemas,
  OpenAPI, generated TypeScript, examples, or `tests/fixtures/v1/`.
- The later A6 replacement remains a breaking 3.0 migration and must ship
  atomically in a separately reviewed post-M5 change with its state/checkpoint
  semantics and migration documentation.

## Error cases and exact diagnostics

Private preflight and control paths use existing `CalcFlowError` variants.
Errors retain a private structured origin for status/testing instead of
encoding every identity only in free-form text.

| Condition                        | Existing variant / exact field and message                                                                                                                                                 |
| -------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| Missing source binding           | `InvalidArgument { field: "sources.<binding_id>", message: "missing binding for external input <binding_id>" }`                                                                            |
| Unknown source binding           | `InvalidArgument { field: "sources.<binding_id>", message: "binding does not match a compiled external input" }`                                                                           |
| Duplicate source binding         | `InvalidArgument { field: "sources.<binding_id>", message: "binding is configured more than once" }`                                                                                       |
| Source has no first-hop edge     | `InvalidArgument { field: "sources.<binding_id>.outputs", message: "must contain at least one compiled edge" }`                                                                            |
| Zero source row bound            | `InvalidArgument { field: "sources.<binding_id>.capabilities.max_batch_rows", message: "must be greater than zero" }`                                                                      |
| Zero source byte bound           | `InvalidArgument { field: "sources.<binding_id>.capabilities.max_batch_bytes", message: "must be greater than zero" }`                                                                     |
| Source exceeds first-hop edge    | `InvalidArgument { field: "sources.<binding_id>.capabilities", message: "maximum batch (<rows> rows, <bytes> bytes) exceeds edge <edge_id> budget (<max_rows> rows, <max_bytes> bytes)" }` |
| Source cursor repeat/regression  | `InvalidArgument { field: "sources.<binding_id>.cursor", message: "source binding <binding_id> emitted a repeated or regressed cursor" }`; fails before every branch enqueue               |
| Source watermark regression      | `InvalidArgument { field: "sources.<binding_id>.watermark", message: "source binding <binding_id> regressed its watermark" }`; fails before every branch enqueue                           |
| Missing sink route               | `InvalidArgument { field: "sinks.<output_id>", message: "external output requires at least one ordinary sink" }`                                                                           |
| Unknown sink route               | `InvalidArgument { field: "sinks.<output_id>", message: "route does not match a compiled external output" }`                                                                               |
| Duplicate sink identity          | `InvalidArgument { field: "sinks.<output_id>.<sink_id>", message: "sink is configured more than once" }`                                                                                   |
| Unsupported exactly-once request | `InvalidArgument { field: "requirements.delivery.<output_id>", message: "exactly-once delivery requires aligned checkpoints and is unavailable before M5" }`                               |
| Multi-ingress watermark          | `InvalidArgument { field: "runtime.nodes.<node_id>.ingress.<ingress>.watermark", message: "multi-ingress watermark control is unavailable before M3; no downstream control was emitted" }` |
| Multi-ingress idle               | `InvalidArgument { field: "runtime.nodes.<node_id>.ingress.<ingress>.idle", message: "multi-ingress idle control is unavailable before M3; no downstream control was emitted" }`           |
| M2 barrier injection             | `InvalidArgument { field: "runtime.nodes.<node_id>.ingress.<ingress>.barrier", message: "barrier control is unavailable before M5; no downstream control was emitted" }`                   |
| Operator output port/kind/schema | Existing `Compile` collector validation, naming `node.<node_id>.outputs.<port>` before any send                                                                                            |
| Active job on runner             | `Conflict { resource: "streaming job", key: "active" }`                                                                                                                                    |
| Supervised panic                 | PR-added public `TaskPanicked { task_id, message }`, exact display `task <task_id> panicked: <message>`; captured text is valid UTF-8 and at most 1,024 bytes                              |
| Closed edge during convergence   | Existing `EdgeClosed { edge }`; recorded as secondary when cancellation is already selected                                                                                                |
| Deadline expiry                  | No `CalcFlowError`; stored as `Cancelled` plus `TerminalCause::DeadlineExceeded`                                                                                                           |

Angle-bracket tokens above are substituted with quoted stable IDs in the
actual diagnostic. Source/sink connector failures preserve their typed source
error and private `FailureOrigin`; error sanitization remains the connector
boundary's obligation. Neither metrics, logs, nor errors may include cursor
payloads, batch attributes, row values, or secret values.

## Happy-path internal test sketch

The primary vertical test intentionally uses only data and EOF because M3
owns union watermark/idle semantics:

```rust
#[tokio::test]
async fn two_sources_union_expression_and_ordered_sinks_drain() {
    let mut runner = ContinuousRunner::new();
    let probes = ProbeSet::new();
    let spec = two_source_union_expression_spec(
        finite_source("left", [left_0, left_1], probes.clone()),
        finite_source("right", [right_0, right_1], probes.clone()),
        slow_recording_sinks(["audit", "primary"], probes.clone()),
        EdgeBudget::new(1, ONE_BATCH_BYTES).unwrap(),
    );

    let job = runner.start(spec).await.unwrap();
    let outcome = job.wait().await;

    assert_eq!(outcome.state, ContinuousJobState::Completed);
    assert_eq!(outcome.cause, TerminalCause::NaturalEnd);
    assert_per_source_fifo(probes.deliveries());
    assert_each_batch_uses_sink_order(probes.deliveries(), ["audit", "primary"]);
    assert_eq!(job.status().tasks.len(), 0);
    assert_all_queues_empty_and_within_budget(job.status());
    assert_all_sources_and_sinks_closed(&probes);

    drop(job);
    runner.shutdown().await.unwrap();
}
```

Separate focused tests must pin:

- runtime-plan extraction moves each operator exactly once and reconstructs
  unary, fan-out, independent-branch, and union topology with the exact
  internal/source/sink edge IDs above;
- failing operator `reset` completes no source/sink lifecycle call, while all
  entry successes remain parked until connector opens and handle claim release
  the separate data/control gate;
- cursor repeat/regression reports exact field
  `sources.<binding_id>.cursor`, watermark regression reports exact field
  `sources.<binding_id>.watermark`, and each fails before the offending event
  is enqueued on any branch;
- panic capture preserves short strings, bounds long ASCII and multibyte
  strings to valid UTF-8 of at most 1,024 bytes ending in `…`, and maps a
  non-string payload to exactly `non-string panic payload`;
- unary watermark handler-before-forward and unary idle forwarding;
- multi-ingress watermark/idle and all barrier injection failing before
  downstream observation;
- source/sink route and capability failures before any open flag changes;
- `EdgeBudget::new(3, sufficient_bytes)` separately admitting exactly three
  repeated `Idle`, equal-or-increasing watermarks, and verified one-slot/
  zero-row/zero-byte data messages, leaving the fourth send pending at depth
  three, then completing exactly one blocked send per dequeue in FIFO order
  over at least 30 messages and finishing with every charge zero;
- an external Rust compile test constructing the unchanged
  `EdgeBudget { max_rows, max_bytes }` and calling the unchanged
  `edge_channel` signature; with `max_rows = 2`, two zero-row/zero-byte
  envelopes enqueue and the third parks until one dequeue, while the same
  suite preserves full-row/full-byte blocking, oversize failure,
  cancelled-send release, receiver-close wakeup, FIFO, and no lost wakeups;
- public-surface tests proving no message-limit field/method, public control
  constructor, source-driven continuous runner, or continuous-job status API
  is exported, while the unchanged `ChannelMetrics` fields report
  `queue_depth <= 2` and `high_water_depth <= 2`; the old control-capacity test
  is replaced or renamed and must not retain a “controls are never throttled”
  assertion;
- a real unary source/operator edge stopping source polling once sustained
  idle, watermark, or empty-data traffic reaches its slot limit, allowing only
  the one already-started poll and one D3 prefetch slot, then converging cleanly
  under separate graceful-drain, cancel, receiver-close, and upstream-error
  cases;
- three-sink ordering and sink-two failure preventing sink three;
- natural end/graceful shutdown drain versus explicit cancel no-drain;
- task failure > explicit cancel > deadline, including a task that first
  notices deadline through `StreamJobContext::check_cancelled`;
- repeated `wait`/`cancel`/`shutdown` returning the same `Arc` outcome;
- dropped wait future preserving the job and dropped job handle transferring
  to a non-empty reaper;
- next start and runner shutdown draining reaper entries;
- premature `recv() == None` failing rather than synthesizing EOF for unary,
  union, and sink ingress;
- branch-two close and emit-then-handler-error retaining non-transactional
  prefixes with per-edge enqueue evidence and zero fully-fanned-out count;
- a full terminal-error counter attaching one non-recursive metrics-overflow
  secondary while preserving the operator primary;
- payload-free deterministic status/metrics and every M2C-FR30 enqueue,
  dequeue, operator, and sink observation boundary.

## Adversarial lifecycle poll/drop test sketches

Terminal observer cancellation is tested by manually polling, not by sleeps:

```rust
for point in DriverAwaitPoint::ALL {
    let job = gated_job_at(point);
    let mut observer = Box::pin(job.wait()); // repeat with cancel/shutdown
    poll_until_gate(&mut observer, point);
    drop(observer);

    assert!(matches!(job.driver_owner(), DriverOwnership::Driving));
    let later = job.cancel().await;
    assert!(matches!(job.driver_owner(), DriverOwnership::Terminal));
    assert!(Arc::ptr_eq(&later, &job.wait().await));
    assert_registry_empty(&job);
}
```

For `cancel()` and `shutdown()`, the cause is asserted committed even when the
observer is dropped before its first poll. Manual gates cover cause receipt,
entry-sender close, each worker join, sink/source close, terminal metrics, and
outcome publication. A concurrent pair of terminal observers plus a job-handle
Drop/runner-shutdown race must end with one `Arc` outcome and no ownership
state other than `Terminal`; no path observes an empty or lost registry.

Start cancellation covers every provisional phase:

```rust
for point in [
    ProvisionalRegistered,
    OperatorEntryPending,
    OneConnectorOpened,
    AllOpenedBeforeDataGate,
    ReadyUnclaimed,
] {
    let mut start = Box::pin(runner.start(gated_spec(point)));
    poll_until_gate(&mut start, point);
    drop(start);

    assert_eq!(runner.provisional_owner(), DriverOwnership::ReaperOwned);
    release_all_teardown_gates(point);
    assert_each_began_open_resource_closed_once();
    assert_data_gate_never_released();

    let next = runner.start(finite_spec()).await.unwrap();
    assert_previous_launch_joined_before(&next);
    drop(next);
}
```

The handle-delivery race is synchronized on the `ReadyUnclaimed -> Claimed`
lock: observer poll winning returns an owning job and Drop winning requests
cancel. Runner shutdown repeats the same cases and must join provisional,
live, then reaper registrations. Open-error tests select the deterministic
primary origin, close the resource that returned `Err` as well as every other
began-open resource exactly once, and verify stable bounded cleanup diagnostics.

## Why this shape

- Deferring public A6 keeps one discoverable public continuous-runner story
  instead of exposing a checkpoint-shaped facade that cannot checkpoint.
- `Vec<Named*Binding>` exists only at the private preflight boundary so
  duplicate IDs remain observable. The validated form becomes ordered maps.
- One output-owned sink task is the smallest structure that guarantees both
  per-batch configured sink order and downstream backpressure without a
  staging queue.
- An `Arc<ContinuousJobOutcome>` makes repeated terminal observation
  implementable even though `CalcFlowError` contains non-`Clone` sources.
- A private `TerminalCause` keeps explicit cancel and deadline independently
  testable while correctly mapping both to `Cancelled` and avoiding M5's
  `CheckpointTimeout`.
- Stable-ID-only metrics recording prevents payloads and arbitrary labels
  from entering observability by construction.
- Reusing `max_rows` as an independent slot ceiling closes zero-cost queue
  growth without adding a third public budget knob; the private
  `message_slot_limit` projection makes the otherwise surprising reuse explicit
  in runtime and soak diagnostics.
- The migration rule `R >= max(required_row_limit,
  required_simultaneous_messages)` makes the source-compatible behavior change
  actionable for direct channel users instead of implying that prior
  zero-cost capacity remains valid.
- Enumerating the only superseded S10/FR23/AC-S10 clauses prevents the third
  predicate from becoming permission to redesign accounting, oversize errors,
  ordering, policies, or the deferred A6 surface.

## Explicitly deferred public A6 work

The following remains frozen by the total API note and is not weakened or
implemented by M2:

- public `Cursor`, `SourceEvent`, `StreamSource`, source policies, A5 sink
  traits/bindings, `StateBackend`, and v3 `CheckpointStore`;
- public source-driven `StreamingRunner::new` and `start`;
- public `StreamingJob::{trigger_checkpoint,status,shutdown,cancel,wait}` and
  public `JobStatus`/`JobOutcome` projections;
- final-manifest drain, `completed_epoch`, recovery from latest manifest,
  checkpoint timeout, barrier injection/alignment, transactional sink commit,
  and S9 capability reporting;
- Python/PyO3, project v3, Studio `/api/v3`, SSE, and migration surfaces.

## Public API delivery status

- G1-G8 are implemented and evidenced at code-approved head `7deda2a0`. All
  new runner, status, metrics, and control shapes remain crate-private.
- The PR-added `TaskPanicked` variant is the sole public Rust API item addition
  in this slice. Its fields and display are frozen; the private 1,024-byte
  capture bound is implemented. The source-compatible but
  observable `edge_channel` admission correction is a separate behavior delta,
  not another item addition.
- Public source-driven A6 remains blocked until the separately reviewed post-M5
  integration change. Exporting `ContinuousRunner`, `ContinuousJob`, source or
  sink bindings, status/metrics, soak controls, or checkpoint controls would
  violate this note.
- Any implementation need for another public error variant or a change to v2
  runner signatures is a new API-design issue and must stop for a reviewed
  note revision; it must not be inferred from an internal acceptance gap.

## Open questions

- **No open proposed M2 behavior or public-scope question.** OQ1-OQ3, B6's
  slot invariant, B7's governing precedence and migration rule, the deadline
  outcome, diagnostic fields, panic bound, soak duration/evidence, and optional
  performance decision are specified above. Revision 6 has a zero-block
  critique, and G1-G8 are closed at code-approved head `7deda2a0`.
- The public A6 integration is a separately reviewed post-M5 change containing
  both M4 state and complete M5 manifest/checkpoint behavior; release planning
  may name that cut, but must not split its durability promise.
- M3 may replace only the private multi-ingress control rejection branch. It
  must implement the frozen minimum/idle/reactivation semantics and retain the
  same runtime-owned control boundary.
- M5 owns whether private M2 ordinary sink adapters are replaced outright or
  wrapped by public A5 bindings. M2 private types carry no migration promise.

## Handoff

The four required decisions are delivered: public A6 is deferred, M2
multi-ingress watermark/idle fails closed while unary control passes through,
ordinary sinks report process-local ordered delivery only, and every data or
control envelope consumes one finite slot independently of its row/byte cost.
PR #83 adds no public callable runner API, but it does add the semver-compatible
public `TaskPanicked` item; the existing public channel also gains the
source-compatible, observable `(R, B) -> (R envelopes, R rows, B bytes)`
admission correction and no new field or callable. The completion work
preserves the error shape and bounds only internally captured text. The exact
20-minute, 120-sample real-runtime soak passed at `7deda2a0` with the durable
bundle linked above. At that code-approved head, the full supplied verification
and review evidence supports the precise label “M2 runtime internals complete.”
It does not certify a public continuous runner or the overall streaming
feature. A documentation-only successor head still needs refreshed exact-head
soak, checks, and Copilot review before merge readiness can be claimed.
