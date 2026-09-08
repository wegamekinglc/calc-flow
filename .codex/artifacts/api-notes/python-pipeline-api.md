# Python SQL and streaming pipelines

Implementation-ready API decision for PR 259, 2026-09-08, against head
`6c669ed2a037376a849061d44a6af2c83e16d566`. This note supersedes the earlier
`python-expression-api.md` compatibility and SQL/streaming positioning decisions
where they conflict. The user explicitly permits breaking Python API changes.
Python is the application API; Rust/DataFusion remains the calculation runtime.

## Review findings and scope

The current PR makes finite expression calculation short, but leaves SQL and
streaming behind different, substantially longer interfaces:

- `compute.py` offers four-argument `compute`/`compute_async` and Arrow results.
  `symbolic/expr.py:315` offers immutable expressions and collection, but no SQL
  table operation or reusable-function `pipe` method.
- `pipeline.py:884` exposes SQL through `PipelineBuilder.sql(name, query,
  aliases=...)`. `examples/02_sql_join.py` must still name a node, compile a plan,
  wrap inputs in `Batch`, and unwrap physical outputs.
- `symbolic/program.py:394` compiles streams, but users must write source/sink
  adapter classes and bind physical endpoints. The recovery example is over
  200 lines. This is a valid advanced contract, not a concise first stream.
- `runtime.py:902` and `StreamingJob` already own real native stream execution,
  cancellation and cleanup. Repeated `compute` calls cannot replace this path:
  every convenience batch call intentionally owns fresh state.
- `crates/calc-flow/src/operator/sql.rs` explicitly supports multiple aliases in
  batch mode and exactly one alias in stream mode. Streaming SQL is stateless
  per input batch; SQL `SUM`, `ORDER BY`, `LIMIT` and SQL window functions do not
  acquire across-batch semantics by entering a stream. Native rolling, event
  windows and stream joins are the existing stateful operators.

Implement a concise declaration/collection path covering SQL and streams, with
real composition and working examples. Do not change REST, project-v3 or managed
checkpoint-v3 formats, native delivery rules, provider registration, or the engine
architecture. A narrow internal SQL schema-planning binding is in scope because
SQL results must participate in typed expressions without handwritten schemas.

## Locked public surface

Keep `compute`/`compute_async`, `table_input`, immutable expressions, `Program`,
`collect`/`collect_async`, and ordinary Python functions as the main model. Add no
separate pipeline class, stage IR, global SQL registry, frame backend, decorator,
or serialized callback. A pipeline is a composed expression declaration.

```python
# Existing signatures remain the short finite execution boundary.
def compute(data, build, /, *, runtime=None, options=None) -> pa.Table: ...
def compute_async(data, build, /, *, runtime=None, options=None) -> Awaitable[pa.Table]: ...

def sql(query: str, /, **tables: TableExpr) -> TableExpr: ...

class TableExpr:
    def sql(self, query: str, /) -> TableExpr: ...

    def stream(
        self, inputs: StreamInput | Mapping[str, StreamInput | TableData], /,
        *, runtime: Runtime | None = None,
        config: StreamRuntimeConfig | None = None,
        watermarks: WatermarkPolicy | Mapping[str, WatermarkPolicy] | None = None,
    ) -> StreamResults[pa.Table]: ...

class Program:
    def stream(
        self, inputs: Mapping[str, StreamInput | TableData], /,
        *, runtime: Runtime | None = None,
        config: StreamRuntimeConfig | None = None,
        watermarks: WatermarkPolicy | Mapping[str, WatermarkPolicy] | None = None,
    ) -> StreamResults[StreamOutput]: ...

# Add once on the expression base, retaining the concrete self type in typing.
class Expr[T]:
    def pipe[SelfExpr: Expr, **P, R](
        self: SelfExpr, function: Callable[Concatenate[SelfExpr, P], R], /,
        *args: P.args, **kwargs: P.kwargs,
    ) -> R: ...

type TableData = pa.Table | pa.RecordBatch | Batch
type StreamInput = AsyncIterable[TableData | Watermark] | SourceBinding

@dataclass(frozen=True, slots=True)
class StreamOutput:
    name: str
    table: pa.Table

class StreamResults[T]:
    async def __aenter__(self) -> StreamResults[T]: ...
    async def __aexit__(self, exc_type, exc, traceback) -> None: ...
    def __aiter__(self) -> AsyncIterator[T]: ...
    async def __anext__(self) -> T: ...
    async def aclose(self) -> None: ...

    @property
    def job(self) -> StreamingJob: ...
```

The signatures above describe API contracts; implementation supplies complete
existing annotations to unchanged functions and accurate exception/context-manager
annotations. New functions/methods must satisfy the existing argument-count and
cyclomatic-complexity gates without suppressions or baseline increases.

`TableExpr.sql(query)` means `sql(query, input=self)`: the one documented local
table name is `input`. Top-level `sql` expresses multiple named table inputs.
Tables are explicit declarations; never inspect the caller's globals or locals.
The root function is lazy and always returns `TableExpr`, never an eager/lazy union.
For the common eager SQL case use
`cf.compute(data, lambda t: t.sql("SELECT ... FROM input"))`.

`pipe` applies `function(self, *args, **kwargs)` once during declaration. It works
for table and column expressions, so ordinary financial helper functions compose
without a feature framework. It can return a `Program` for branching or another
declared value; it does not require every function to return its input type.
Reject awaitable results with a pathful TypeError and close a returned coroutine.
Do not turn pipe functions into execution-time row UDFs. The original builder
exception and traceback survive.

This follows the explicit frame-first composition pattern in the official
[Polars pipe reference](https://docs.pola.rs/api/python/stable/reference/dataframe/api/polars.DataFrame.pipe.html)
and lazy SQL/native-expression mixing in the official
[Polars LazyFrame.sql reference](https://docs.pola.rs/api/python/stable/reference/lazyframe/api/polars.LazyFrame.sql.html).
Calc Flow keeps its own SQL dialect, alias name, stream lifecycle and capabilities.

## SQL declaration, schema and lowering

Add one immutable SQL table primitive to the existing symbolic IR, containing
query text and deterministic explicit alias/child pairs. Reuse native `sql` project
nodes. Alias names use existing portable identifier/port validation; reject no
tables, non-TableExpr values and ambiguous/duplicate output column names. Do not
guess schemas, execute Python query parsers, rewrite user SQL via string replacement,
or interpolate payloads into queries. SELECT/CTE-only validation remains native.

Infer each SQL output schema by native planning of its input Arrow schemas,
without reading data or executing a zero-row query. The reference implementation
path is `Runtime._infer_symbolic_expression_schema` in `pipeline.py`,
`_infer_expression_schema` in `crates/calc-flow-python/src/config.rs:668`, and
`DataFusionRuntime::infer_input_query_schema` in `datafusion.rs:299`. Generalize
the internal planning helper to register named empty-schema inputs, retain the
same physical-plan schema accuracy, and expose a private Python planning method.
The adapter cache must include query/alias/input schemas and the runtime
registration revision. Avoid stale schemas and changing the caller's runtime.
This is a schema bridge; it does not introduce a second execution path. Existing
matrix-provider boundaries retain their validated analyzer schema and generic
registered port contract; expression adapters carry those types into SQL.

Analyzed SQL output has its own row lineage, inferred field types/nullability,
and the union of upstream state dependencies. Do not carry temporal ordering or
entity declarations across arbitrary SQL: projections can rename/drop keys,
joins/aggregates change rows and ORDER BY is local to SQL execution. Row-local
expressions after SQL are supported. Rolling before SQL is supported. A temporal
expression directly after SQL fails the existing missing-ordering validation;
do not add an unsafe implicit ordering assertion or a new sorting API in this
change. Document this exact boundary, not a blanket SQL/streaming incompatibility.

Lower complete expression -> SQL -> expression DAGs to a single native graph.
There must be no intermediate Arrow collection in a declaration and no
per-batch reconstruction of declarations. SQL fragments retain exact input and
output schemas and source lineage. Shared inputs/outputs and repeated SQL alias
references must not duplicate stateful upstream work.

Batch SQL may join multiple inputs. Stream compilation rejects SQL nodes with
more than one alias before opening any source, even when aliases share a root.
Single-alias SQL can run before/after supported row-local stages and after native
rolling calculations. This API exposes native built-in SQL functions. Existing
explicit UDF-reference selection stays available in advanced graph construction;
do not auto-select all runtime registrations or add untyped UDF callables.

## Streaming ownership and execution

`stream(...)` constructs a one-shot `StreamResults` resource owner. It immediately
copies the input mapping and captures declarations/source/Batch references; Arrow
buffers stay shared and read-only. It does not consume an iterable, open connectors,
create state storage, compile, or start work until `async with` enters. Iteration
before entering and entering twice fail clearly. One consumer may iterate; reject
concurrent `__anext__` calls. There is no blocking stream facade.

On entry, validate declared input names and kinds and compile exactly one fresh
native stream plan. Bind names through lowering metadata, then launch exactly one
`StreamingRunner` and retain its `StreamingJob`. Rolling state lives in this native
job across every batch. Reusing the declaration for another stream creates fresh
state and a fresh owner. A stream never reuses a caller's cached batch plan.

Stream input handling:

- A single `StreamInput` is accepted only when a TableExpr has exactly one dynamic
  table root and no static parameters; otherwise require a logical-name mapping.
  Program always takes a mapping. Static parameter entries take the existing
  `TableData`/array-Batch forms and are captured once through native static bindings.
- An async iterable is adapted internally to the existing `StreamSource` protocol.
  Its iterator is acquired once on entry. Normalize Arrow batches with the current
  metadata-safe wrapper, validate exact declared schema, and assign monotonically
  increasing counter cursors. Declare `ReplayPositioning.UNSUPPORTED`,
  `SourceDeliveryCapability.LOSSY`, and the native watermark capability required
  by the selected policy below. A generated counter is not a replay position.
- Derive finite maximum source batch rows/bytes from `config.edge_budget`, using
  current defaults, and reject oversized inputs by source name before admission.
  No unbounded eager collection or background producer buffer is allowed.
- A supplied `SourceBinding` keeps its real capabilities and watermark policy.
  This supports event windows and existing custom connectors through the same
  concise output iterator. Explicit source-provided iterator watermarks also use
  the existing `Watermark` envelope and policy, as specified below.
- Source closure follows the native owner. For an iterable adapter, await `aclose`
  on the acquired iterator if it provides one, on normal exhaustion, failure or
  cancellation; perform it at most once. Do not mutate or clear the caller's
  container or close an unstarted iterable during Python argument validation.

### Progress correction: online temporal results

This bounded correction, 2026-09-09, supersedes the original blanket disabled-
watermark default. Reviewing implementation `945ab5c` exposed a real acceptance
gap: `RollingOperator.process_data` buffers rows and `on_watermark`/`on_end`
finalizes them. Cross-section and event-window operators likewise need progress.
An infinite plain iterable with disabled watermarks therefore never yielded these
results. Finite-feed EOF tests did not prove online streaming.

Add the single `watermarks` keyword shown above; each method has exactly five
parameters including `self`. Reuse existing `WatermarkPolicy` variants, without a
new public config wrapper or any change to `StreamRuntimeConfig`/native schemas.
Copy a supplied policy mapping at `stream()` call time alongside input bindings.
The policy applies to dynamic logical input names, not physical ports or outputs.

Policy selection is deterministic:

- `None`, or an omitted entry in a policy mapping, selects the default per source.
  A scalar policy is accepted when exactly one dynamic table input exists, even
  with additional static parameters. Multiple dynamic sources require a mapping.
  Reject unknown names, static parameter names, non-policy values and attempts to
  override a supplied `SourceBinding`; validate before opening sources. An existing
  binding already owns its policy and needs no extra keyword.
- For a plain iterable whose declaration has `event_time`, default to native
  `BoundedOutOfOrderness(event_time, timedelta(microseconds=1),
  timedelta(milliseconds=100))`, with no idle timeout. The adapter enforces that
  every arriving row has a non-null timestamp and event time is nondecreasing
  in arrival order, within and across batches, across the entire logical source.
  It validates the complete batch before admitting any row. A decrease fails with
  `stream.inputs.<name>.event_time: expected nondecreasing event time; select an
  explicit watermark policy for out-of-order input`. It neither sorts nor drops
  caller data. Do not infer this guarantee from `sequence_by` or only per entity.
- The generated watermark is `max_seen - 1 microsecond`, computed by the existing
  native progress driver. Native rolling closes rows at or before the watermark.
  Therefore timestamps equal to the current maximum remain open: equal timestamps
  may span batches or different entities without prematurely closing a group.
  A larger observed timestamp closes the previous timestamp. This one-microsecond
  offset is a safe completion boundary under the enforced monotonic contract,
  not a promise to buffer one microsecond of arbitrary disorder.
- Native timers publish progress even while the next iterable read is suspended.
  The 100 ms interval is a scheduling target, not an output latency guarantee under
  load or backpressure. No Python timer task is needed. Timestamp conversion,
  known-schema validation and representable-range errors remain native; never
  wrap/saturate the `i64` minimum-minus-delay edge to a false forward watermark.
- A plain input without declared event time defaults to `DisabledWatermarks`.
  Stateless expressions and SQL keep producing on each batch without any timestamp
  requirement. `DisabledWatermarks()` can also be selected explicitly to retain
  finite/EOF-driven temporal finalization; docs must identify that behavior.

For disorder, users pass an existing `BoundedOutOfOrderness("ts", delay, interval)`
directly to `watermarks`, or by logical source name. Its durations retain native
positive-duration validation. This explicit mode does not impose the default
monotonic-arrival validation. Native generation uses the existing inclusive cutoff
`max_seen - delay`; future temporal rows must be strictly after a published cutoff.
The convenience compiler keeps `allowed_lateness_micros=0` and `late_policy="error"`:
late envelopes fail, never silently disappear. It does not redefine the supplied
policy by adding a hidden offset, set late-drop, or pretend a watermark corrects
bad source ordering. Stateless operators still pass data through unchanged.

For explicit progress, `watermarks=SourceProvidedWatermarks()` permits the iterable
to yield existing timezone-aware `cf.Watermark(...)` objects between Arrow data.
The adapter reports `EMITS_NATIVE` and forwards these through `SourceBinding`;
watermark events do not increment data cursors. Native validation enforces monotone
watermarks and downstream operators enforce lateness. Generated/default/disabled
modes report `NEVER_EMITS` and reject manually yielded Watermark objects instead of
mixing two sources of progress. No new public envelope, replay cursor or source
class is required in application code. A watermark must be justified by the source:
the library cannot infer future completeness for arbitrary unordered inputs.

The default's monotonic order is global per source, not independently per symbol.
For example, entity A at time 20 followed by entity B at time 10 is rejected even
if each entity is individually ordered. Equal times across symbols are accepted;
existing entity/sequence row-identity validation still applies. Multiple sources
keep separate frontiers; the existing native aggregate uses the minimum of active
sources. Do not automatically mark a quiet source idle: stalled progress may be
needed for correctness. Explicit policies may opt into their existing idle-timeout
semantics, including native late-row errors if an old source later resumes.

Empty batches are valid, do not advance the adapter's last timestamp or native
maximum, and are not end-of-input. A source that pauses at one timestamp cannot
prove that timestamp complete by waiting; it needs a later timestamp, an explicit
source watermark, or EOF. Event windows still wait for their window-end frontier;
stream joins retain their existing matching and watermark-driven eviction rules.

Support was checked in `runtime.py` (`SourceBinding._native_policy` and Watermark
projection), `progress/generated.rs` (`observe_batch`, `on_timer`),
`progress/prepare.rs` (positive durations and schema), and `operator/rolling.rs`
(`process_data`, `on_watermark`, `is_late`, `closing_keys`). The correction is an
adapter/policy-selection change, not a new engine or a checkpoint-format change.

All stream convenience calls use a run-owned temporary managed checkpoint root
and ordinary output sinks. This provides true in-process stateful streaming and
native lifecycle cleanup, not restart recovery or exactly-once delivery. Do not
expose a `checkpoint_path` on this convenience iterator: arbitrary iterables cannot
seek and dequeuing output does not prove application delivery. Delete the temporary
root only after native terminal cleanup releases its files/locks. Perform filesystem
creation/removal through asynchronous offload, not blocking event-loop operations.

For durable recovery and transactional delivery continue to use explicit
`Program.compile_stream`, `StreamingRunner`, `SourceBinding`, `SinkBinding`, and
`ManagedCheckpointRuntime`; these are necessary operational controls, not old-API
compatibility aliases. Keep the existing recovery example and tests. Do not rewrite
their checkpoint, port, or delivery contracts merely to shorten the quickstart.

Output handling and termination:

- TableExpr yields Arrow tables. Program yields immutable `StreamOutput(name,
  table)` events, using logical output names. This is an event stream, not a
  fabricated synchronized dictionary across outputs. Preserve each output's order;
  do not promise a total ordering between independent outputs or one result per
  source batch. SQL/windows can change result cardinality and batch boundaries.
- Use bounded queue sink adapters with native edge budgets/backpressure and no
  task per batch. A bounded one-slot queue is sufficient with native batch limits.
  Do not enqueue terminal sentinels through a potentially full queue from `close`:
  use separate terminal observation, so cancellation can always finish.
- Own any job-wait/queue-wait tasks and await their cleanup. Native failure must
  wake an idle consumer even if no table is queued. Completed streams drain queued
  output, await native completion, release resources, then raise StopAsyncIteration.
  Failed jobs raise the existing `StreamingRuntimeError` with safe structured
  native reason details; do not translate failure into empty successful output.
- Breaking iteration inside the context, an exception in the context, explicit
  `aclose`, or task cancellation cancels a live job and awaits the existing native
  bounded source/sink cleanup. Preserve the original exception/cancellation after
  cleanup. `aclose` is idempotent. Cleanup cannot rely on garbage collection,
  detached tasks, or an event loop that will later disappear.
- `job` is available after successful entry for status/diagnostics and existing
  controls. Calling a job control does not transfer ownership away from the
  context. Checkpoint triggering within an ephemeral run does not enable restart.
  Arrow convenience results omit the native Batch envelope; applications needing
  Batch metadata/delivery acknowledgement use explicit sinks.

Reuse/generalize `_BatchBindings` into private declaration-to-native endpoint
metadata shared by collection and streaming. Cover SQL, shared/CSE roots, rolling,
event-window, stream-join and static/matrix lowering paths. Do not guess binding
identity from schemas, node-name strings or declaration order. A logical stream
input must be consumed/opened once: if lowering exposes several physical ingress
ports for the same root, create one native ingress/fan-out route rather than bind
the same source iterator to multiple owners. No aliases enter project-v3 serialization.

## Concrete example targets

The implementer ships these as real executable 20-50-line examples, with checks
active under optimized Python. The code below fixes the important results and
composition; add normal main guards without expanding each example beyond 50 lines.

### SQL and reusable pipeline: `examples/19_sql_expression_pipeline.py`

```python
"""Reuse table and column pipelines around a native SQL stage."""

from __future__ import annotations

import pyarrow as pa

import calc_flow as cf


def add_gross(t: cf.TableExpr) -> cf.TableExpr:
    return t.with_columns(gross=cf.row.cast(t["quantity"], "float64") * t["price"])


def discounted(t: cf.TableExpr, rate: float) -> cf.TableExpr:
    return t.select("order_id", net=t["gross"] * (1.0 - rate))


def pipeline(t: cf.TableExpr) -> cf.TableExpr:
    return (
        t.pipe(add_gross)
        .sql("SELECT order_id, gross FROM input WHERE gross >= 20 ORDER BY order_id")
        .pipe(discounted, rate=0.1)
    )


def main() -> None:
    orders = pa.table(
        {
            "order_id": [1, 2, 3],
            "quantity": [2, 1, 3],
            "price": [10.0, 5.0, 10.0],
        }
    )
    result = cf.compute(orders, pipeline)
    expected = {"order_id": [1, 3], "net": [18.0, 27.0]}
    if result.to_pydict() != expected:
        raise RuntimeError(result.to_pydict())
    print(result.to_pydict())


if __name__ == "__main__":
    main()
```

### SQL join and expression: modernize `examples/02_sql_join.py`

```python
"""Compose named SQL inputs and ordinary Python expression transforms."""

from __future__ import annotations

import pyarrow as pa

import calc_flow as cf


def main() -> None:
    orders = pa.table({"order_id": [1, 2, 3], "amount": [75, 120, 40]})
    fees = pa.table({"order_id": [1, 2, 3], "fee": [5, 12, 4]})
    joined = cf.sql(
        "SELECT o.order_id, o.amount - f.fee AS net "
        "FROM o JOIN f ON o.order_id = f.order_id ORDER BY o.order_id",
        o=cf.table_input("orders", schema=orders.schema),
        f=cf.table_input("fees", schema=fees.schema),
    )
    output = joined.pipe(lambda t: t.select("order_id", doubled=t["net"] * 2))
    result = output.collect({"orders": orders, "fees": fees})
    expected = {"order_id": [1, 2, 3], "doubled": [140, 216, 72]}
    if result.to_pydict() != expected:
        raise RuntimeError(result.to_pydict())
    print(result.to_pydict())


if __name__ == "__main__":
    main()
```

### Stateful streaming and composite features: `examples/20_streaming_pipeline.py`

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
    await asyncio.Event().wait()  # The source stays open; no EOF is delivered.


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
            if sum(part.num_rows for part in tables) >= 3:
                break
    actual = {  # Allow floating-point round-off in the example's verification.
        name: [None if value is None else round(value, 12) for value in values]
        for name, values in pa.concat_tables(tables).to_pydict().items()
    }
    expected = {"delta": [None, 2.0, 3.0], "mean_delta": [None, 2.0, 2.5]}
    if actual != expected:
        raise RuntimeError(actual)
    print(actual)


if __name__ == "__main__":
    asyncio.run(main())
```

The first value of the second batch uses the first batch's previous price; the
nested rolling mean also retains state. The paused source never emits EOF, yet
the first three rows become observable. The latest timestamp, 4, remains open
behind watermark 3. Breaking the result loop cancels and closes the source. Keep
the corresponding finite-feed test, which still returns all four rows after EOF.
The native null/min-periods contract remains authoritative; float verification
rounds to 12 places. The shipped example should stay focused at about 20-50 lines
of executable Python rather than adding a source/sink class. Example 21 needs no
call-site change: it deliberately demonstrates timestamp-free stateless results.

### Named branches: `examples/21_streaming_outputs.py`

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

Do not promise cross-output arrival order. A focused test additionally counts
iterator acquisition/closure to prove this branch pipeline consumes the source once.

## Errors, migration and documentation

Use existing construction TypeError/ValueError, analysis CompileError and execution/
streaming exception categories. New messages identify paths, for example
`sql.tables.orders: expected TableExpr`, `stream.inputs.quotes: missing input`,
`stream.inputs.quotes.schema: ...`, `stream: enter with async with before iteration`,
and `outputs.join: multi-input SQL has no incremental stream semantics`.
Never include row payloads, credentials or callback repr in diagnostics.

The user does not require compatibility. Do not add deprecation wrappers, aliases,
alternate argument layouts, a duplicate pipeline facade, or compatibility-only tests.
Existing advanced interfaces may stay when they provide necessary features. Their
presence is not an obligation to teach every spelling or preserve old shapes. Do
not spend this task deleting valid advanced functionality simply to create a break.

The concrete migration is:

- Basic `.sql(name, query).compile_batch().execute(...).outputs[...]` becomes
  `compute(data, lambda t: t.sql(query))` or `cf.sql(query, **tables).collect(inputs)`.
- Hand-connected expression/SQL nodes become `.pipe(...).sql(...).pipe(...)`.
- Quickstart source/sink classes, physical names and temporary checkpoint setup
  become `async with output.stream(feed()) as results: async for table in results`.
- Programs with outputs consume named `StreamOutput` events. Durable/custom delivery
  continues through the explicit runner and real replayable sources/sinks.
- Replace old documentation claims that SQL is only an advanced escape hatch, all
  streaming requires hand-managed ports, or compatibility is an acceptance target.

After code review, reconcile README, getting-started, introduction, Python/API
references, batch/stream guides, expression workflows, example indexes and
CHANGELOG once. Retain accurate recovery/late-data/provider/REST reference detail;
clearly separate per-batch SQL from native stateful operators. Update example-runner
registration if needed. Add the four actual examples above, not prose-only snippets.

## Focused acceptance and handoff

Implementation starts with the smallest failing behavior tests, then targeted green:

1. SQL-only compute, expression -> SQL -> expression, two-input SQL join, nested
   SQL/CTE and shared SQL output compute actual Arrow values with one native graph.
   Unknown aliases/columns, duplicate output fields, unsupported Arrow fields,
   non-SELECT/multi-statements fail clearly. Schema nullability/type comes from
   native planning, including nullable aggregates. A single narrow Rust/binding
   schema test proves planning does not execute data or selected Python callbacks.
2. Pipe table/column composition and arguments work; callable runs once during
   construction, original exceptions survive, async builders are rejected/closed,
   and no callback is serialized or invoked by execution. Ordinary overloaded
   arithmetic/boolean expression tests supply the existing semantic baseline.
3. Actual native stream tests split a rolling/composite calculation across two
   or more batches and assert continuity, including the second-batch boundary.
   A second run has fresh state. SQL stream stages produce correct per-batch
   results; multi-input SQL fails before source open. Named outputs/fan-out open
   and close each logical source once, including distinct same-schema inputs.
   The online correction additionally requires a source paused on its third
   `next` after admitting times `[1, 2]` and `[3, 4]`, without EOF: observe all
   three finalized rows `[None, 2, 3]` / `[None, 2, 2.5]` within a bounded wait
   while the job is still running and the source remains open. Count rows across
   arbitrary output-batch boundaries; time 4 remains buffered. The existing RED
   for this exact case failed with TimeoutError on `945ab5c`; source/job/tasks/temp
   cleanup passed. This is the required GREEN, not another finite-feed substitute.
   Add focused cases for within-batch and cross-batch descending timestamps,
   globally descending times across different entities, equal-time rows split
   across batches with valid distinct row identities, and empty batches that do
   not advance progress. Explicit bounded-disorder input must succeed inside its
   declared frontier and fail for a temporal row at/before an already observed
   watermark, with late_policy=error. Exercise an iterable-provided Watermark
   causing output while the source stays open, and reject that envelope under
   generated/disabled modes. Cover per-source policy mapping capture, bad names,
   SourceBinding override rejection, default no-time row-local output, and one
   multi-source case proving a quiet active input prevents premature finalization.
   Reuse existing cancellation/backpressure cleanup assertions with the new native
   progress timers. Use events/output observation to establish order, not sleeps
   alone. The focused RED and these directly affected cases are sufficient locally;
   no repeat whole-stream suite or native engine rewrite is required by this note.
4. One focused lifecycle test each for natural EOF, early break, producer error,
   schema mismatch/oversize input, idle source cancellation, output queue full on
   cancellation and start failure. Verify source closure, terminal job outcome,
   no owned tasks/threads left, temporary-state cleanup, and no swallowed failure.
   Cover mapping capture, pre-entry misuse, repeated close/entry and caller data
   immutability. Do not rely on sleeps alone to establish a lifecycle boundary.
5. Reuse directly affected stream window/join/static-binding tests to verify that
   generalized logical binding metadata does not swap/drop inputs. Run one existing
   managed rolling-recovery case to show advanced checkpoint semantics survive;
   the convenience iterator itself makes no replay guarantee.
6. Run the four touched/new examples and scoped Ruff/format/type checks. Run the
   actual existing complexity ratchet once because new convenience APIs recently
   failed it on this PR. Native checks are limited to the new schema helper/binding
   and compilation needed for affected Python tests. Do not run a local full
   workspace/performance suite. Full cross-platform/coverage/release/performance
   gates stay in CI; preserve all existing floors/baselines and protocol schemas.

No product ambiguity requires a separate spec/critic stage. The concrete risks are
SQL physical schema inference, correct SQL/stream lineage binding, and resource
cleanup under backpressure; the requirements above bound them. If an implementation
discovers a native limitation beyond the SQL schema bridge, report that specific
blocker before changing state/delivery/format semantics. Next owner is
`cf-implementer`, followed by mandatory `cf-reviewer` and one doc-writer alignment.

Design verification: read the relevant Python exports, expression/program/lowering,
runtime source, tests, native SQL operator/schema planner/binding, crate exports,
existing SQL/recovery examples and normative introduction/stream guide. This note
is the only file written by this design step. No build, tests, git operation,
commit, push or remote review mutation was performed.
