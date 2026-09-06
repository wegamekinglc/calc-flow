# Continuous streaming guide

Calc Flow continuous jobs consume async sources, execute a compiled stream
graph, publish to async sinks, and recover from managed checkpoints. Use this
guide for application-owned connectors. For Kafka, PostgreSQL, MySQL, ClickHouse,
HTTP, WebSocket, files, and Parquet, combine it with the
[connector and stream-project guide](connectors.md).

## Choose batch or stream

Use `compile_batch()` when all named inputs are already available and one
execution should return one `RunResult`. Use `compile_stream()` when sources
arrive over time, state must survive restarts, event-time progress matters, or
the application needs a long-lived owning job.

Batch and stream plans are intentionally different types. A batch plan cannot
be passed to `StreamingRunner`, and a stream plan cannot be executed with
`execute()`.

## First Python continuous job

The complete runnable version is
[`04_continuous_runtime.py`](../examples/04_continuous_runtime.py). Its core is:

```python
plan = (
    PipelineBuilder("orders")
    .expression("calculate", "result = value + 1")
    .compile_stream()
)
runner = StreamingRunner(
    plan,
    {"input": SourceBinding(source, watermark_policy=DisabledWatermarks())},
    {"output": [SinkBinding.ordinary("archive", sink)]},
    ManagedCheckpointRuntime(".calc-flow-state/orders"),
)
job = await runner.start_async()
print(job.status())
outcome = await job.wait_async()
```

`PipelineBuilder.compile_stream()` is the graph-only path for connectors owned
by the application. A connector-backed project uses
`compile_stream_project(project)` and then `StreamingRunner(plan)`; the
compiled project already owns its registered source/sink factories, state root,
and runtime settings.

## Source contract

A Python source implements:

```python
class Source:
    def capabilities(self) -> SourceCapabilities: ...
    async def open(self, cursor: Cursor | None) -> None: ...
    async def next(self) -> Data | Watermark | Idle | None: ...
    async def close(self) -> None: ...
```

The runtime samples `capabilities()` once during preflight, before `open()`.
Keep it deterministic and free of I/O. It declares:

- whether an exact accepted position can be paused, reported, and sought;
- whether accepted data can be lost before runtime observation;
- strict maximum rows and bytes for one batch;
- exact or unknown Arrow schema evidence;
- whether the connector emits native watermarks.

A `Data` event pairs an immutable `Batch` with a `Cursor`. The cursor must name
the next unread position after that batch, not its starting position. Cursor
order bytes must advance monotonically, while the bounded JSON payload contains
only the connector position needed for replay. The runtime assigns the stable
source ID to an unbound cursor during admission.

Return `None` only for permanent end-of-input. Use `Idle()` when no data is
currently available but the source remains live. `close()` must be safe after
normal completion, launch failure, cancellation, or another connector's error.

Rust uses the same lifecycle through the `StreamSource` trait. See
[`continuous_runtime.rs`](../crates/calc-flow/examples/continuous_runtime.rs).

## Watermark policies

Each `SourceBinding` freezes exactly one policy:

| Policy                     | Use when                                                   |
| -------------------------- | ---------------------------------------------------------- |
| `SourceProvidedWatermarks` | The transport emits trustworthy timezone-aware watermarks  |
| `BoundedOutOfOrderness`    | Calc Flow should derive progress from an event-time column |
| `DisabledWatermarks`       | The graph is stateless or windows should close only at end |

Bounded out-of-orderness names the event-time column, maximum delay, emission
interval, and optional idle timeout. The progress driver computes the job
watermark from active ingresses. Idle and ended sources stop holding back the
minimum; data or a legal watermark reactivates an idle source.

Watermarks are monotone progress declarations, not filters. The progress
driver forwards data unchanged. A window operator applies its own late rule:
an assignment is late when its window end is less than or equal to the current
input watermark.

## Event-time windows

Rust applications create a `WindowSpec` and add a
`WindowAggregateOperator` to the graph:

```rust
let spec = WindowSpec::tumbling("event_time", Duration::from_secs(60))?
    .group_by(["account"])?
    .aggregate(AggregateFunction::Sum, "amount", "total")?;
let window = WindowAggregateOperator::new("minute_totals", input_schema, spec)?;
let plan = PipelineBuilder::new("orders")?
    .add_node("minute_totals", window)?
    .compile_stream(&udfs, &StreamRequirements::default())?;
```

Run the complete source-watermark-window-sink example with:

```bash
cargo run -p calc-flow --example windowed_streaming
```

Project v3 represents the same operator as a data-only `window` node. Use that
form from Python or Studio; the current functional Python builder does not add
a separate window convenience method. The [connector guide](connectors.md)
contains the exact project fragment.

Tumbling and hopping windows use fixed UTC microsecond geometry. Supported
aggregates are `count`, `sum`, `min`, `max`, and `avg` over the validated type
matrix. Output is deterministic by window bounds and group key. Empty windows
are not materialized.

## Sink contract

An ordinary sink implements async `open`, `write`, and `close`. It can provide
at-least-once delivery on a lossless replayable route, but recovery may call
`write` again for data beyond the last durable checkpoint.

Transactional and epoch-idempotent sinks additionally implement:

```text
begin_epoch -> write* -> pre_commit -> commit
                             └──────> abort (before durable manifest)
recover (after restart) -> finish the manifest-recorded decision
```

`pre_commit` returns bounded JSON evidence; it must not contain secret or row
payloads. `commit` must complete the exact prepared epoch. `recover` must make
the recorded decision idempotent. Choose `SinkBinding.transactional` or
`SinkBinding.epoch_idempotent` only when the connector actually implements
that protocol.

## Delivery requirements

Declare requirements during stream compilation:

```python
requirements = StreamRequirements({"output": DeliveryGuarantee.EXACTLY_ONCE})
plan = builder.compile_stream(requirements=requirements)
```

Omitted outputs request at-least-once. Exactly-once requires lossless exact
replay from every reachable source, deterministic restore for every stateful
operator, and transactional or qualifying epoch-idempotent evidence from every
bound sink. The entire route is checked before connector `open()`.

`job.status()["delivery"]` reports requested and effective guarantees per
output. Treat that status as the runtime proof for the compiled route, not as a
substitute for configuring the external system correctly.

## External provider lifecycles

Rust external stream operators must report a public `StreamOperatorLifecycle`
proof. The default is `Unproven`. When a trusted `StreamOperatorFactory`
creates a stateless operator, `compile_stream` requires
`microbatch_invariant: true`; if any output requests exactly-once delivery,
compilation also requires `deterministic: true` and `replay_safe: true`.
Stateful operators follow a separate path: a positive versioned
`CheckpointedStateful` capability is sufficient, without a stateless
lifecycle claim.

An `Unproven` operator is not rejected merely because ordinary stream
compilation ran. Instead, the existing second admission stage fails closed
when a runner is configured with a checkpoint runtime, before any connector
opens. Exactly-once delivery requires that checkpoint runtime, so it cannot
admit an unproven operator. This preserves the existing separation between
plan construction and checkpointed job admission without turning missing
evidence into an exactly-once claim.

The current Python stream-provider path is the trusted NumPy/JAX
`expression@1` registration installed by `register_numpy` or `register_jax`.
Each accepted data micro-batch calls the provider once with
`(batch, provider_options)`. Callback failures are provider errors and emit no
output. Cancellation is checked before dispatch and after a successful
callback, before emission. The callback runs through `spawn_blocking` and
cannot be preempted once running; if cancellation is observed when it returns,
its successful result is not emitted.

NumPy/JAX stream expressions must depend on `x` and contain neither function
calls nor matrix multiplication, keeping the accepted subset conservatively
row-axis-independent. Reductions, transpose, reshape, constant-only
expressions, and `@` therefore remain batch-only. `table_matmul@1` is also
batch-only. Symbolic programs that explicitly compose `linalg.from_columns`,
allowlisted elementwise operations, exactly one `linalg.matmul`, and
`table.attach_columns` instead lower to the stateless `symbolic_matrix@1`
stream provider. The static `weights` parameter occurs exactly once as that
matmul's direct right operand. The provider receives the table once per
micro-batch and reuses the job-latched weights across calls.

## Static inputs

A static input is an immutable per-job side value — model weights, a reference
matrix, a small lookup table — declared by the plan and supplied by the caller
at runner construction. It is latched once, never re-sent as stream data, and
never treated as a source.

Declarations are data-only: a project-v3 `static_inputs` array whose entries
name an unconnected external input port of a graph node. A table entry pins
the exact Arrow schema; an array entry pins the backend, dtype, and shape. The
[connector guide](connectors.md) carries the exact syntax and validation
rules.

Python callers supply values through the keyword-only runner argument:

```python
runner = StreamingRunner(
    plan,
    {"input": SourceBinding(source, watermark_policy=DisabledWatermarks())},
    {"output": [SinkBinding.ordinary("archive", sink)]},
    ManagedCheckpointRuntime(".calc-flow-state/weights"),
    static_inputs={"weights": weights_batch},
)
```

`None` normalizes to an empty mapping. Keys must be `str` and values must be
`Batch`; anything else raises `TypeError` before any native construction. The
mapping is defensively copied immediately. Project-backed plans keep rejecting
externally supplied `sources`, `sinks`, `checkpoints`, and `config`, but
`static_inputs` is exempt from that rejection and is required when the plan
declares static inputs. `plan.static_input_ids` returns the declared names,
and `plan.source_binding_ids` excludes them.

Rust uses an additive builder on the unchanged `StreamingRunner::new`:

```rust
let runner = StreamingRunner::new(plan, sources, sinks, checkpoints)?
    .with_static_inputs(BTreeMap::from([("weights".to_owned(), weights)]))?;
```

Validation, latching, and digest computation happen exactly once per job,
inside `start`, and complete before any source, operator, sink, or provider
lifecycle method runs. A missing or unexpected input, a wrong batch kind, a
table schema mismatch, an array backend/dtype/shape mismatch, and an
unsupported digest dtype all fail on a `static_inputs.{name}` error path
before sources open. After the latch the job-visible value is frozen: mutating
the caller's mapping, the original Python `Batch`, or externally mutable
NumPy backing memory cannot change what the job observes or what a later
restart compares. Handles are released exactly once on every exit path —
success, cancellation, startup failure, and recovery failure.

Each latched value is reduced to a lowercase-hexadecimal SHA-256 digest with
version string `calc_flow.static_input.digest.v1`. The byte-level encoding is
the canonical tagged-bytes grammar of
[API note §7](../.codex/artifacts/api-notes/symbolic-computation-engine.md):
independent of record-batch chunking, dictionary layout, strides, and batch
metadata, with NaNs canonicalized per dtype. Declarations join the semantic
plan fingerprint; payload digests never join the lineage key. A restart that
supplies a different value therefore reaches the existing lineage and is
rejected before sources open with the exact message:

```text
static_inputs.{name}.digest: checkpoint digest {stored} does not match prepared digest {prepared} for calc_flow.static_input.digest.v1
```

On the Python surface this recovery rejection arrives as a structured
`StreamingRuntimeError` with category `checkpoint_mismatch`; engine-level it
is the checkpoint-mismatch error class. Checkpoint manifests record one digest
entry per static input under the root field `static_inputs`; the field is
omitted when the set is empty, so existing manifests keep their bytes. Status,
metrics, and errors expose at most the input name, digest version, and digest
— never payloads or backing memory. The built-in NumPy/JAX
`symbolic_matrix@1` provider is the static-array consumer: the runner places
each declared weight array once per job on a blocking worker, caches only the
successfully placed immutable provider value after a cancellation check, and
reuses it for every data micro-batch. Its `static_placement_bytes` metadata is
logical provider transfer, not peak memory or process RSS. The
[API reference](api-reference.md) lists the complete exported static-input
surface.

## Bounded event-time Join

Calc Flow 4.0 adds a two-input inner equi-Join for stream plans. Each retained
row must fall inside the inclusive interval
`[left_time - before, left_time + after]`. The two source lineages must provide
watermark progress so the operator can evict rows that cannot match again.

Python declares exact input schemas and explicit limits; no unbounded defaults
exist:

```python
from datetime import timedelta

from calc_flow import ArrowFieldSpec, JoinStateLimits, JoinTimeBounds, PipelineBuilder

fields = (
    ArrowFieldSpec("account_id", "int64", False),
    ArrowFieldSpec("event_time", "timestamp[us]", False),
)
builder = PipelineBuilder("payments").stream_join(
    "payment_join",
    left_schema=fields,
    right_schema=fields,
    left_keys=("account_id",),
    right_keys=("account_id",),
    left_event_time="event_time",
    right_event_time="event_time",
    bounds=JoinTimeBounds(
        before=timedelta(minutes=5),
        after=timedelta(minutes=1),
    ),
    limits=JoinStateLimits(
        max_state_rows_per_side=100_000,
        max_state_bytes_per_side=128 * 1024 * 1024,
        max_matches_per_input_batch=1_000_000,
    ),
    left_prefix="authorization",
    right_prefix="payment",
)
```

Null event times, null keys, and rows strictly older than their own ingress
watermark do not match and are not retained. Equality with the current
watermark is accepted. Output order is deterministic, and checkpoint restore
preserves retained rows, row IDs, counters, and the independent output
frontier. State and match admission failures surface one of the four stable
`reason_code` values on terminal errors; callers should still retain a fallback
for future reason strings.

The Join has one required table output port named `output`, and its schema is
derived rather than declared. Every left column is emitted as
`left_prefix__name`, followed by every right column as `right_prefix__name`,
each keeping its input nullability. With the prefixes above the output columns
are `authorization__account_id`, `authorization__event_time`,
`payment__account_id`, and `payment__event_time`; downstream operators and
windows reference these prefixed names.

Each Join node also reports a payload-free status. The
`job.status()["stream_joins"]` mapping keys are node IDs; each value carries
per-side retained rows and bytes, evicted, late, and null drop counters,
`late_affected_batches`, `max_lateness_micros`, plus the node's
`emitted_match_rows`, `state_limit_failures`, and `match_limit_failures`. Jobs
without a Join node report an empty mapping. Studio progress events carry the
same per-node rows as a `stream_joins` list on the run event.

## Checkpoints and recovery

`ManagedCheckpointRuntime(root)` owns local manifest and state storage. Keep
one stable root per pipeline lineage. Do not edit, copy partially, or expose
its internal files through an API.

Request a checkpoint while the job is running:

```python
epoch = await job.trigger_checkpoint_async()
assert job.status()["checkpoint"]["last_completed_epoch"] == epoch
```

The returned epoch is durable. Operator segments have been published, the
manifest is durable, and the runtime has completed the required post-manifest
protocol before returning success.

Starting a compatible plan on the same root selects the latest complete
manifest and validates pipeline fingerprint, source/operator/sink identities,
state versions, and delivery evidence before opening the data gate. A
runtime-tuning change is visible through the runtime-config hash but does not
invalidate semantically compatible state.

Finite jobs publish a terminal checkpoint after final operator output. A
restart from that manifest returns the same terminal epoch without reopening
ended sources or writing final output twice. Run the proof:

```bash
uv run python examples/08_streaming_recovery.py
```

## Job lifecycle

| Operation                    | Meaning                                                         |
| ---------------------------- | --------------------------------------------------------------- |
| `status()`                   | Fresh synchronous, payload-safe observation                     |
| `trigger_checkpoint_async()` | Publish and await one durable epoch                             |
| `shutdown_async()`           | Stop admission, drain accepted work, publish terminal progress  |
| `cancel_async()`             | Cancel work and await bounded cleanup                           |
| `wait_async()`               | Observe natural terminal completion without changing state      |

Cancelling a task that is only awaiting `wait_async()` does not cancel the
job. Call `cancel_async()` explicitly. A runner can start once, and a job is the
sole lifecycle owner; create a fresh plan, bindings, and runner for a restart.

Blocking `start`, `trigger_checkpoint`, `shutdown`, `cancel`, and `wait`
variants exist for Python callers outside an event loop. They reject an
active event loop; async applications should always use the async forms.

## Runtime tuning and backpressure

`StreamRuntimeConfig` controls checkpoint interval and timeout, per-edge row
and byte budgets, and retained epochs. These values affect runtime behavior and
the diagnostic config hash, not the semantic plan fingerprint.

Choose an `EdgeBudget` large enough for the largest admitted source batch and
the number of simultaneous control envelopes. Rows and envelopes each have an
independent `max_rows` bound; bytes have `max_bytes`. A source declaration that
can exceed the effective edge budget fails before open.

Backpressure is expected. A slow sink eventually awaits upstream sends. Do not
hide that signal behind an unbounded queue inside a connector. If the external
transport cannot pause, declare the loss explicitly and use a best-effort
route.

## Status and diagnostics

Status includes job state, terminal cause, requested/effective delivery, task
counts, watermark, bounded edge/source/operator/sink metrics, per-node Join
state, and checkpoint summary. It intentionally excludes row data, cursor
payloads, pre-commit payloads, connector internals, secret values, and
filesystem paths.

Terminal outcomes are `completed`, `cancelled`, `failed`, or
`recovery_required` and include payload-safe structured errors. Preserve these
categories in application logs; do not replace them with raw connector
exceptions that might contain credentials or data.

## Production checklist

- Freeze exact schemas whenever the transport can provide them.
- Make each `Data` cursor represent the accepted cut after the batch.
- Bound source batches and connector-internal buffers.
- Pick a watermark policy that matches transport behavior.
- Request the weakest delivery contract that is correct, then confirm the
  effective proof in status.
- Use stable pipeline, binding, sink, state-root, and connector identities
  across restart.
- Put credentials in secret resolvers, never project options.
- Exercise checkpoint, graceful shutdown, explicit cancellation, source
  failure, sink failure, and restart before deployment.
- Monitor watermark stalls, edge saturation, checkpoint failures, task errors,
  and retained state size.

For the underlying message, progress, and checkpoint invariants, continue with
the [stream message envelope](runtime-envelope.md). For component ownership,
read the [design and architecture guide](design.md).
