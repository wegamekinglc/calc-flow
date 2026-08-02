# Internal ordered runtime envelope

The Rust core uses a crate-private `RuntimeEnvelope` in two separate internal
paths: compiled endpoint storage wraps data, while the private control
scheduler owns pending control work. This is an internal execution contract,
not a public Rust, Python, or Studio API. Public graph inputs, operator inputs
and outputs, runner steps, and terminal results continue to use immutable
`Batch` values.

The implementation lives in `crates/calc-flow/src/runtime/envelope.rs`,
`crates/calc-flow/src/operator.rs`, `crates/calc-flow/src/pipeline.rs`, and
`crates/calc-flow/src/pipeline/control.rs`.

## Carrier and visibility

`RuntimeEnvelope` has two crate-private variants:

- `Data(Batch)` wraps the existing public data envelope. Endpoint storage uses
  this variant without changing the table or external-payload sharing rules.
- `Control(SharedControlMarker)` carries either a watermark marker or an epoch
  marker. The shared handle is an immutable `Arc` so fan-out clones the handle,
  not a data payload.

Each `ControlMarker` owns a kind and an opaque, UUID-backed
`ControlOccurrence`. The crate-private `watermark()` and `epoch()` mint
operations create a fresh occurrence. The marker itself is move-only; after a
dispatch consumes it, the runtime creates one `SharedControlMarker` and clones
only that shared handle for successor steps.

An occurrence distinguishes submissions for tests and diagnostics. It is not:

- an event-time value or watermark timestamp;
- a monotonic sequence;
- a checkpoint epoch ID;
- a persistent or replayable identity;
- a cross-input ordering key.

The marker, occurrence, shared handle, route, and envelope are not re-exported
from `calc_flow`. They are absent from project documents, checkpoints, graph
fingerprints, Python bindings, Studio routes, OpenAPI, and generated clients.

## Data dispatch

`ExecutionPlan::execute` still accepts `BTreeMap<String, Batch>` and returns a
`RunResult`. Internally, the plan stores external and produced endpoint values
as `RuntimeEnvelope::Data`. Input gathering accepts only the data variant and
passes the same `BTreeMap<String, Batch>` contract to operators. Terminal
output collection converts only data envelopes back to `Batch`.

A control envelope reaching the data gather or terminal-output path is an
internal routing error. It is never silently dropped or exposed as a new
`BatchKind` or port kind.

## Control route and forwarding ownership

Compilation derives a control-route status for every external graph input
after the normal topology has been validated. This derivation does not reject
a data graph and does not participate in its fingerprint. A supported route
contains stable target steps with:

- the target node and input port;
- the unique external-input or graph-edge ingress;
- successor step indices in compiled order.

The runtime is the only forwarding owner. Existing built-in and external
operators are `Transparent`: the runtime admits the target and forwards the
marker without calling a handler. A crate-private `SignalAwareOperator` can
instead receive `handle_control(&ControlMarker, &RunContext) -> Result<()>`.
The handler can observe the marker and mutate state covered by its normal
snapshot/restore lifecycle, but it cannot return, replace, suppress, or route a
control message.

The `SignalAwareOperator` capability and its construction path remain private.
The current signal-aware implementations are library-test probes; public
`PipelineBuilder` and project compilation create only existing data operators.

The crate-private direct control entry follows this sequence:

1. Look up the selected external input and preflight its complete reachable
   route.
2. Acquire the existing unleased plan transaction, recovering any abandoned
   direct operation first.
3. Convert the owned marker into one shared handle and create the origin
   pending step.
4. Snapshot operator lifecycle state and create the existing in-flight
   rollback marker.
5. Admit route steps in stable compiled order. For each target, check
   cancellation, cross the target-consumption observation boundary, then run
   transparent or aware handling.
6. After the handler and post-handler cancellation check both succeed, create
   successor pending steps by cloning the shared marker handle.
7. Commit the in-flight operation after the whole route succeeds, or use the
   existing rollback path after an error.

There is no public control-dispatch method, Python or Studio projection, or
runner-owned control entry. `MicroBatchRunner` and `StreamingRunner` continue
to submit formed `Batch` values only.

## Ordering and target consumption

For messages submitted serially to one supported external input, every
admitted target ingress observes the same submission order. This is the
implemented per-edge FIFO guarantee. It covers interleaved data, watermark,
and epoch submissions as well as consecutive control markers.

The control scheduler uses pending target steps; it does not create persistent
per-edge queues. Creating a pending step is not an observation. A target
consumes its ingress only after its pre-handler cancellation check succeeds and
before transparent handling or the aware handler is invoked.

At fan-out, each admitted target consumes the same occurrence once. Stable
compiled step order makes fault behavior deterministic, but it is not a
business guarantee of global order between branches. If an earlier branch
fails, its ingress has already been observed, its successors are not created,
and all unvisited sibling steps are discarded without observation.

These guarantees do not define an order between different external inputs and
do not provide a global order at a merge.

## Multi-input fail-closed boundary

Every declared input port on a control-reachable node counts as a potential
control ingress. Connected inputs, external inputs, and unconnected optional
ports all count; two target ports connected to the same source still count as
two ingresses.

A route is supported only when every reachable node has exactly one potential
ingress. Otherwise control preflight returns
`CalcFlowError::InvalidArgument` with `field = "control_input"` and identifies
the first conflicting node and its ingresses. The failure happens before the
plan transaction, snapshot, handler calls, target observations, or partial
forwarding.

This control-only restriction does not change normal graph validation. The
same multi-input graph can still compile and execute data through the public
API.

## Error and rollback boundary

Control dispatch uses the same plan run lock, lease exclusion, lifecycle
snapshot, in-flight operation marker, and `rollback_error` machinery as direct
data execution. A runner lease prevents the private direct entry from bypassing
runner ownership.

An aware handler's non-cancellation error becomes a node-scoped
`CalcFlowError::Operator` diagnostic containing the marker kind, occurrence,
and ingress. A cancellation returned by the handler keeps its cancellation
classification. Cancellation observed after the handler wins over a different
handler error.

When rollback succeeds, operator state returns to the pre-dispatch snapshot and
the in-flight marker is cleared, so a later data or freshly minted control
submission can run cleanly. Rollback does not erase diagnostic observations,
undo arbitrary external side effects, compensate an already visited sibling,
or provide cross-branch atomicity. The boundary is no stronger than the
existing data-handler failure boundary.

The private direct control entry does not read or write a checkpoint store.
Control markers are not checkpoint data, and retry mints a new occurrence
rather than replaying the failed occurrence.

## Compatibility and non-goals

The internal carrier does not change:

- the public Rust crate exports or `Operator` trait;
- Rust or Python execution and runner signatures;
- `Batch`, `BatchKind`, `Port`, `RunResult`, or `NodeTiming`;
- project or checkpoint formats;
- graph fingerprint inputs or values;
- Python exceptions, Studio REST, OpenAPI, or generated TypeScript.

No migration is required. This capability does not claim complete continuous
execution, watermark semantics, epoch/checkpoint semantics, cross-input global
ordering, barrier alignment, terminal control delivery, or exactly-once
processing.

## Downstream handoff

### D2: continuous execution

Reusable internal mechanisms:

- `RuntimeEnvelope` as a data/control carrier;
- compiled route lookup, deterministic dispatch, fan-out, and per-edge FIFO;
- runtime-only forwarding and the transparent/aware distinction;
- the plan run lock, lease exclusion, lifecycle transaction, and rollback
  boundary.

D2 still has to define source polling, bounded queues and backpressure,
multi-source scheduling, job-level cancellation and shutdown, terminal sink
control delivery, and the continuous lifecycle. The current
`StreamingRunner` is a push-based sequence of already formed batch steps, not
a continuous source-driving loop.

### D3: watermark semantics

Reusable internal mechanisms:

- the private watermark marker kind and shared occurrence carrier;
- handler-before-forward ordering;
- target-consumption observation and deterministic fan-out failure behavior;
- conservative multi-input fail-closed handling.

D3 still has to define the event-time value and type, monotonicity, idle-input
handling, late data, multi-input minimum/merge rules, window interaction, and
any public Rust, Python, or Studio surface. `ControlOccurrence` cannot be used
as event time.

### D6: epoch and checkpoint semantics

Reusable internal mechanisms:

- the private epoch marker kind and shared occurrence carrier;
- signal-aware state under the existing snapshot/restore lifecycle owner;
- handler-before-forward ordering;
- transaction, lease exclusion, in-flight operation, and rollback machinery.

D6 still has to define epoch identity, barrier alignment, the snapshot
boundary, marker persistence and restore, sink commit coordination, and
exactly-once semantics. `ControlOccurrence` cannot be persisted or interpreted
as a checkpoint epoch ID.
