# Continuous streaming guide

[Documentation](README.md) / 2.3 Continuous streaming

Use `TableExpr.stream` to consume an async source as Arrow results, or
`Program.stream` for named output events. One native job executes the expression
and SQL graph and retains state across batches. Enter the result owner with
`async with`, and consume it with `async for`.

For durable restart and delivery controls, compile the declaration and bind
explicit sources, sinks, and managed checkpoints. This guide covers both
convenience streams and application-owned connectors. For Kafka, PostgreSQL, MySQL, ClickHouse,
HTTP, WebSocket, files, and Parquet, combine it with the
[connector and stream-project guide](connectors/README.md).

On this page:

- [Choose batch or stream](#choose-batch-or-stream)
- [First Python continuous job](#first-python-continuous-job)
- [Named streaming outputs](#named-streaming-outputs)
- [Stream ownership and SQL boundaries](#stream-ownership-and-sql-boundaries)
- [Explicit connectors and recovery](#explicit-connectors-and-recovery)
- [Source contract](#source-contract)
- [Watermark policies](#watermark-policies)
- [Event-time windows](#event-time-windows)
- [Sink contract](#sink-contract)
- [Delivery requirements](#delivery-requirements)
- [External provider lifecycles](#external-provider-lifecycles)
- [Static inputs](#static-inputs)
- [Bounded event-time Join](#bounded-event-time-join)
- [Checkpoints and recovery](#checkpoints-and-recovery)
- [Job lifecycle](#job-lifecycle)
- [Runtime tuning and backpressure](#runtime-tuning-and-backpressure)
- [Status and diagnostics](#status-and-diagnostics)
- [Production checklist](#production-checklist)

## Choose batch or stream

Use `cf.compute` or `Program.collect` when all inputs are available and you want
Arrow tables. Compile an explicit batch plan when you need `RunResult` diagnostics
or owned plan state. Use `TableExpr.stream` or `Program.stream` when sources
arrive over time and the calculation must retain state between batches.
Use `Program.compile_stream()` with explicit bindings and a stable
`ManagedCheckpointRuntime` when state must survive process restarts or the
application owns transactional delivery.

Batch and stream plans are intentionally different types. A batch plan cannot
be passed to `StreamingRunner`, and a stream plan cannot be executed with
`execute()`.

## First Python continuous job

[20_streaming_pipeline.py](../examples/20_streaming_pipeline.py) composes a
price delta, a rolling mean of that delta, and SQL projection in one pipeline:

```python
from __future__ import annotations

import asyncio

import pyarrow as pa

import calc_flow as cf

SCHEMA = pa.schema(
    [
        pa.field("ts", pa.timestamp("us", tz="UTC"), nullable=False),
        pa.field("symbol", pa.string(), nullable=False),
        pa.field("price", pa.float64(), nullable=False),
    ]
)


async def batches():
    for ts, prices in [([1, 2], [10.0, 12.0]), ([3, 4], [15.0, 14.0])]:
        yield pa.table({"ts": ts, "symbol": ["a", "a"], "price": prices}, schema=SCHEMA)
    await asyncio.Event().wait()  # Keep the input open: output must not wait for EOF.


def features(t: cf.TableExpr) -> cf.TableExpr:
    delta = cf.ts.delta(t["price"])
    return t.select(delta=delta, mean_delta=cf.ts.mean(delta, window=cf.rows(2)))


async def main() -> None:
    source = cf.table_input(
        "quotes",
        schema=SCHEMA,
        entity_by=("symbol",),
        event_time="ts",
        sequence_by=("ts",),
    )
    output = source.pipe(features).sql("SELECT delta, mean_delta FROM input")
    tables = []
    async with asyncio.timeout(5), output.stream(batches()) as results:
        async for table in results:
            tables.append(table)
            if sum(batch.num_rows for batch in tables) >= 3:
                break
    actual = pa.concat_tables(tables).to_pydict()
    if actual != {"delta": [None, 2.0, 3.0], "mean_delta": [None, 2.0, 2.5]}:
        raise RuntimeError(actual)


if __name__ == "__main__":
    asyncio.run(main())
```

The first price in the second source batch uses the previous batch's price,
producing delta `3.0`. The nested rolling mean also retains history. Both state
stages live in one native stream job. The source stays open after two batches;
the consumer receives the first three rows before EOF and then exits. The
checked deltas are `[None, 2.0, 3.0]`, with means `[None, 2.0, 2.5]`. Context
exit cancels the waiting source and cleans up the job.

The input schema and ordering are declared once. Nondecreasing event times let
the default watermark close timestamps below the latest observed time. Here
time 4 remains buffered behind watermark 3. Equal timestamps may span batches;
they remain open until progress proves them complete. Select an explicit
[watermark policy](#watermark-policies) for unordered or source-provided progress.
Native finality rules still determine emission; one input batch need not produce
one output table.

## Named streaming outputs

[21_streaming_outputs.py](../examples/21_streaming_outputs.py) branches one
logical source into independent calculations:

```python
from __future__ import annotations

import asyncio

import pyarrow as pa

import calc_flow as cf


async def batches():
    yield pa.table({"value": [1, 2]})
    yield pa.table({"value": [3]})


async def main() -> None:
    source = cf.table_input("events", schema=pa.schema([("value", pa.int64())]))
    program = cf.Program(
        "branches",
        outputs={
            "double": source.select(value2=source["value"] * 2),
            "large": source.filter(source["value"] >= 2).select("value"),
        },
    )
    values = {"double": [], "large": []}
    async with program.stream({"events": batches()}) as results:
        async for output in results:
            values[output.name].extend(output.table.column(0).to_pylist())
    if values != {"double": [2, 4, 6], "large": [2, 3]}:
        raise RuntimeError(values)
    print(values)


if __name__ == "__main__":
    asyncio.run(main())
```

`Program.stream` takes a mapping keyed by declared input names and yields
immutable `StreamOutput` events. Each event contains `name` and `table`;
the example routes them by name and checks `double=[2, 4, 6]` and `large=[2, 3]`.
The stream preserves each output's order without promising a total order
between independent outputs or synchronized dictionaries across branches.
The example accumulates values for verification; that is not required by the
stream interface.

## Stream ownership and SQL boundaries

`stream(inputs, /, *, runtime=None, config=None, watermarks=None)` constructs a
one-shot `StreamResults` owner. A single-table declaration with no static parameters
accepts one async iterable or `SourceBinding` directly; multiple or static
inputs require a logical-name mapping. `Program.stream` always takes a mapping.
Input and watermark-policy mappings are copied immediately, but source references
and read-only Arrow buffers are shared. Opening sources, compiling the fresh plan,
and starting the native job happen on context entry.

Iteration requires an entered context and one consumer. Leaving `async with`,
breaking iteration inside it, explicit `await results.aclose()`, or task
cancellation settles the owned job and source cleanup. `aclose` is idempotent.
Native failure raises `StreamingRuntimeError`; it is not reported as successful
empty output. After entry, `results.job` provides status and job controls while
the context retains ownership.

Native row/byte edge limits and a bounded result queue apply backpressure to
slow consumers. `StreamRuntimeConfig.edge_budget` also bounds each iterable
input batch; schema mismatches and oversized batches fail with the input name.
This does not bound tables that application code accumulates after reading them.

A streaming SQL stage supports exactly one alias and evaluates its query
separately for each native input batch. SQL aggregates, `ORDER BY`, `LIMIT`, and
SQL window functions do not retain cross-batch SQL state. Multi-alias SQL is a
batch operation. Use native `ts` operations for rolling state and the documented
window/join declarations for event-time aggregation and matching.

SQL results have a new row lineage and no inherited temporal ordering. The
supported pipeline above calculates rolling values before SQL, then may apply
row-local expressions after it. A SQL result cannot currently feed a symbolic
event window or execute as a standalone array Program output; see
[SQL composition](symbolic-api.md#sql-composition).

Convenience streams use a temporary managed checkpoint root and ordinary
output sinks. Async iterable inputs provide best-effort delivery and no replay;
a generated counter is not a recoverable source cursor. Even a supplied
replayable `SourceBinding` does not make the temporary output iterator durable
or provide exactly-once application delivery. Use explicit sinks and a stable
checkpoint root for those guarantees. Temporary state is removed only after
native cleanup finishes. Arrow results omit the `Batch` envelope; use an
explicit sink when application processing needs its metadata or delivery
acknowledgement.

## Explicit connectors and recovery

Build expressions with `cf.table_input`, Python operators, and `cf.Program`.
Declare the input schema before sources start. For rolling and cross-section
work, declare entity/event-time/sequence keys and a non-null
`timestamp[us, UTC]` event-time field. Call `program.compile_stream(runtime)`
(or omit runtime when no custom registration is needed), then pass that plan,
sources, sinks, and `ManagedCheckpointRuntime` to `StreamingRunner`.

[Example 10](../examples/10_symbolic_streaming_recovery.py) is a complete
expression-based rolling job: it builds a program, binds `"input"` and
`"output"`, checkpoints during processing, and resumes with both state stages
restored. See [expression workflows](symbolic-workflows.md#run-continuously-and-recover).

Bind the plan's `source_binding_ids`, `static_input_ids`, and `sink_binding_ids`.
These are physical graph names. Direct `StreamingRunner` bindings use them;
`TableExpr.stream` and `Program.stream` translate logical input/output names. Each stream compile creates a fresh owning native
plan, and each runner starts once. Convenience batch collection never starts a
job or chooses checkpoint storage.

The explicit graph alternative is demonstrated by
[`04_continuous_runtime.py`](../examples/04_continuous_runtime.py):

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

Both expression and builder stream compilation produce graph-only plans for
connectors owned by the application. A connector-backed project uses
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

`stream(..., watermarks=None)` chooses a policy for each ordinary iterable input.
If the declaration has `event_time`, arriving timestamps must be non-null and
nondecreasing across every row and batch of that logical source. This ordering
applies across all entities, not separately per symbol. A decrease rejects the
batch with its input name before admission; the adapter never sorts or drops rows.

The default uses native `BoundedOutOfOrderness` with a one-microsecond delay,
a 100 ms emission interval, and no idle timeout. Its watermark is
`max_seen - 1 microsecond`. Rows at the latest timestamp remain open, allowing
equal timestamps to span batches and entities. A larger timestamp closes earlier
ones. The native timer keeps running while the next iterable read is suspended;
backpressure and scheduling can delay delivery. Empty batches do not advance
progress. Waiting at one timestamp alone cannot prove it complete.

Without declared event time, ordinary inputs use `DisabledWatermarks`.
Stateless expression and SQL stages still produce results as batches arrive.
Explicitly disabling watermarks on a temporal calculation leaves finalization
dependent on EOF or other progress provided by its graph.

Pass one existing `WatermarkPolicy` to `watermarks` for a single dynamic input,
or a mapping keyed by logical dynamic input names for several sources. Static
parameters do not count as dynamic inputs. Omitted mapping entries select the
default. Unknown or static names and invalid policy values fail before source
opening. A supplied `SourceBinding` already owns its policy and cannot be
overridden through this keyword.

The available policies are:

| Policy                     | Use when                                                   |
|----------------------------|------------------------------------------------------------|
| `SourceProvidedWatermarks` | The transport emits trustworthy timezone-aware watermarks  |
| `BoundedOutOfOrderness`    | Calc Flow should derive progress from an event-time column |
| `DisabledWatermarks`       | The graph is stateless or windows should close only at end |

Bounded out-of-orderness names the event-time column, maximum delay, emission
interval, and optional idle timeout. Durations must be positive. An explicit
`BoundedOutOfOrderness` permits disorder without the default monotonic-arrival
validation and publishes the native inclusive cutoff `max_seen - delay`.
Convenience rolling/cross-section compilation keeps zero allowed lateness and
the error policy: rows whose native closing coordinate is at or before the
published watermark fail rather than being silently dropped. Window and join operators retain
their separately documented late-data rules.

With `watermarks=SourceProvidedWatermarks()`, an iterable may yield existing
timezone-aware `Watermark` objects between Arrow batches. These events use the
source-provided native capability and do not advance data cursors. The source
must justify completeness; native validation rejects regressing watermarks.
Generated and disabled policies reject manually yielded watermarks. An explicit
policy takes responsibility for progress instead of the default order check.

The progress driver computes the job
watermark from active ingresses. Idle and ended sources stop holding back the
minimum; data or a legal watermark reactivates an idle source. A quiet source
is not automatically made idle by the default. For a node combining inputs,
finalization follows that node's aggregate ingress progress; independent output
branches are not synchronized. Selecting an idle timeout can permit progress
past a quiet source, whose later rows remain subject to native late-data rules.

Watermarks are monotone progress declarations, not filters. The progress
driver forwards data unchanged. A window operator applies its own late rule:
an assignment is late when its window end is less than or equal to the current
input watermark.

## Event-time windows

Python expressions imported from `calc_flow` declare the same native operator with
`window.tumbling` or `window.hopping` and an ordered sequence of
`window.count`, `window.sum`, `window.min`, `window.max`, or `window.avg`
aggregates. Run
[`symbolic_event_window.py`](../examples/symbolic_event_window.py) for a
grouped minute summary with explicit source watermarks. The
[symbolic window reference](symbolic-api.md#symbolic-event-time-window-aggregation)
defines exact types, row origins, and the supported stateless transformations
before and after the window.

The [Rust runtime reference](rust-api.md) covers native window extension work.

Runtime extension authors can create a `WindowSpec` and add a
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

Project v3 represents the operator as a data-only `window` node. Python and
Studio can use that form directly; the functional Python builder has no
separate window convenience method. The
[project guide](projects-guide.md#union-and-event-time-windows) contains the
exact project fragment.

Tumbling and hopping windows use fixed UTC microsecond geometry. Supported
aggregates are `count`, `sum`, `min`, `max`, and `avg` over the validated type
matrix. Output is deterministic by window bounds and group key. Empty windows
are not materialized. Null-time rows are dropped. Watermark equality with the
window end closes the window, and end-of-input flushes remaining state.
Hopping drops only already-closed assignments from a late row. No early,
update, or retraction output is emitted.

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

These controls apply to explicit stream plans and sinks. The convenience
iterator uses temporary state and does not provide durable application delivery.
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

An `Unproven` operator can compile for an ordinary stream plan, but
checkpointed job admission rejects it before any connector opens.
Exactly-once delivery requires checkpoints and therefore also rejects it.

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
[project guide](projects-guide.md#static-input-declarations) carries the exact
syntax and validation rules.

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

Rust supplies static values through the runner builder:

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

A restart with changed static values fails before sources open with a
structured `StreamingRuntimeError` in category `checkpoint_mismatch`.
Keep weights stable for a checkpoint lineage; use a separate lineage when the
calculation needs different static values. Status and errors expose only the
input name and digest information, never the payload.

Run [11_symbolic_static_matrix.py](../examples/11_symbolic_static_matrix.py)
to check batch/stream parity and one-time weight placement. For digest
encoding, manifest identity, copy boundaries, and provider placement, read
[symbolic compiler design](symbolic-design.md#static-values-and-matrix-placement).
The [API reference](api-reference.md) lists the exported static-input types.

## Bounded event-time Join

Stream plans support a two-input inner equi-Join. Each retained
row must fall inside the inclusive interval
`[left_time - before, left_time + after]`. The two source lineages must provide
watermark progress so the operator can evict rows that cannot match again.
Run [12_symbolic_stream_join.py](../examples/12_symbolic_stream_join.py) for
a complete match, then [13_symbolic_relational_dag.py](../examples/13_symbolic_relational_dag.py)
for nested joins.

Prefer `cf.table.stream_join` for expression composition. The examples above
use root `calc_flow` imports with exact input schemas, ordering declarations,
`JoinTimeBounds`, and `JoinStateLimits`. The
[expression join reference](symbolic-api.md#symbolic-bounded-stream-joins)
defines post-join ordering and nested composition.

The advanced `PipelineBuilder.stream_join` form also requires exact input
schemas and explicit limits; no unbounded defaults exist:

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

The operations below control an explicit `StreamingJob`. A `StreamResults`
context also owns its job: cancelling iteration or exiting the context cancels
and settles its live work.

| Operation                    | Meaning                                                        |
|------------------------------|----------------------------------------------------------------|
| `status()`                   | Fresh synchronous, payload-safe observation                    |
| `trigger_checkpoint_async()` | Publish and await one durable epoch                            |
| `shutdown_async()`           | Stop admission, drain accepted work, publish terminal progress |
| `cancel_async()`             | Cancel work and await bounded cleanup                          |
| `wait_async()`               | Observe natural terminal completion without changing state     |

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
