# Continuous streaming guide

Calc Flow continuous jobs consume async sources, execute a compiled stream
graph, publish to async sinks, and recover from managed checkpoints. Use this
guide for application-owned connectors. For Kafka, PostgreSQL, ClickHouse,
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

Blocking `start`, `checkpoint`, `shutdown`, `cancel`, and `wait` variants exist
for Python callers outside an event loop. They reject an active event loop;
async applications should always use the async forms.

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
counts, watermark, bounded edge/source/operator/sink metrics, and checkpoint
summary. It intentionally excludes row data, cursor payloads, pre-commit
payloads, connector internals, secret values, and filesystem paths.

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
