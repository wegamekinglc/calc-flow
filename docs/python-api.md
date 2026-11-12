# Python API

[Documentation](README.md) / 3.2 Python API

For a guided first calculation, read [batch calculations](batch-guide.md).
This page describes the Python application API and its contracts; the full
expression catalog has its own [reference](symbolic-api.md).

The `calc-flow-python` package exposes Python expressions and Arrow
execution over the internal Rust runtime through PyO3. CPython 3.9 or newer
is required. This page describes version `2026.11.2` in the current checkout;
use a source build when the published package has a different version.

On this page:

- [Install and develop](#install-and-develop)
- [Compute Arrow data](#compute-arrow-data)
- [Table expressions and names](#table-expressions-and-names)
- [Reusable programs and collection](#reusable-programs-and-collection)
- [SQL and pipeline composition](#sql-and-pipeline-composition)
- [Streaming results](#streaming-results)
- [LateOutputs and local late policy](#lateoutputs-and-local-late-policy)
- [Bounded backward ASOF Join](#bounded-backward-asof-join)
- [Choosing an integration API](#choosing-an-integration-api)
- [Table batches and builder](#table-batches-and-builder)
- [Multi-input SQL](#multi-input-sql)
- [Trusted Python scalar UDFs](#trusted-python-scalar-udfs)
- [Runtime capabilities](#runtime-capabilities)
- [Execution options and provider context](#execution-options-and-provider-context)
- [Async execution](#async-execution)
- [Native arrays](#native-arrays)
- [NumPy and JAX](#numpy-and-jax)
- [Symbolic declarations and static analysis](#symbolic-declarations-and-static-analysis)
- [Projects and persistence](#projects-and-persistence)
- [Streaming runner](#streaming-runner)
- [Exceptions](#exceptions)
- [More examples](#more-examples)

## Install and develop

```bash
uv add calc-flow-python
uv add "calc-flow-python[numpy]"  # optional
uv add "calc-flow-python[jax]"    # optional
```

The default Python dependencies are `pyarrow` and `typing-extensions`. The
`array` extra installs both NumPy and JAX; `benchmark` and `dev` are not runtime
requirements. Project-document validation uses the native Rust runtime and does
not require Pydantic. Maturin is required for source builds, not for installing
a prebuilt wheel.

From a source checkout:

```bash
uv sync --extra dev
uv run maturin develop
```

## Compute Arrow data

```python
import pyarrow as pa
import calc_flow as cf

data = pa.table({"a": [1, 3], "b": [2, 4]})
result = cf.compute(data, lambda t: t.select(total=t["a"] + t["b"]))
assert result.to_pydict() == {"total": [3, 7]}
```

`cf.compute(data, build, /, *, runtime=None, options=None) -> pyarrow.Table`
accepts an Arrow `Table`, `RecordBatch`, or table `Batch`. It derives the
supported field types, nullability, and column order from that input, calls the
synchronous builder exactly once with a `TableExpr`, and executes its returned
table declaration.
The builder must return `TableExpr`; asynchronous builders and other return
values fail before native execution. Builder exceptions keep their traceback.

`cf.compute_async(data, build, /, *, runtime=None, options=None)` returns an
awaitable Arrow table. Its builder and declaration preparation run when called;
awaiting it drives cancellation-aware native execution. `compute` rejects a running event loop
before invoking the builder. Both forms accept keyword-only `runtime` and
`options`.

Arrow schema and field metadata are omitted only from the internal execution
schema. The caller's objects and their metadata remain unchanged, and a table
`Batch` retains its `Batch.metadata`. Column buffers remain shared. Keep their
underlying storage read-only until execution completes; this is not a deep
copy of table contents. Async collection copies input mappings and captures
`Batch` references at call time, before awaiting execution.

Zero-column Arrow inputs can still contain rows. Metadata normalization
preserves those row counts for `compute`/`compute_async`, `TableExpr` and
`Program` collection (including async forms), and async iterable stream
inputs. Projecting a literal over a three-row, zero-column input therefore
produces three result rows.

### Temporal ordering

`compute` and `compute_async` infer an input with no ordering declaration.
For temporal calculations, set `entity_by`, `event_time`, and `sequence_by` on
`cf.table_input`, apply the same synchronous builder to that declaration, and
collect the returned table expression.

The [financial example](batch-guide.md#financial-expressions) declares `source`
for its `quotes` data and `signals` builder. Use
`signals(source).collect(quotes, runtime=runtime, options=options)` or
`await signals(source).collect_async(quotes, runtime=runtime, options=options)`;
both execution arguments remain optional. Named-output programs use
`Program.collect` or `Program.collect_async` with the same declared inputs.

Rolling and cross-section calculations require explicit `entity_by`,
`event_time`, and `sequence_by` declarations. Rolling event time must be a
non-null `timestamp[us, UTC]` field. Ordinary Arrow inference makes timestamp
fields nullable, even when no value is null; supply an explicit non-null schema
as in [example 09](../examples/09_symbolic_financial_features.py). Event windows
have their own [timestamp contract](symbolic-api.md#schema-and-geometry).

## Table expressions and names

Use `import calc_flow as cf`. `cf.table_input(name, /, *, schema, entity_by=(),
event_time=None, sequence_by=())` accepts a `pyarrow.Schema` or ordered
`Sequence[cf.Field]`. Declarations contain schema and expression structure;
they do not retain live data. `cf.lit(value)` constructs a scalar expression from
`None`, `bool`, `int`, finite `float`, or `str`. Cast an untyped null explicitly when its type cannot
be inferred.

| Operation                                      | Meaning                                                             |
| ---------------------------------------------- | ------------------------------------------------------------------- |
| `t["price"]`                                   | Select one named column expression                                  |
| `+`, `-`, `*`, `/`, unary `-`                  | Compose arithmetic, including reflected scalar arithmetic           |
| `==`, `!=`, `<`, `<=`, `>`, `>=`               | Build comparison expressions                                        |
| `&`, `\|`, `~`                                 | Compose boolean expressions; parenthesize comparisons               |
| `t.with_columns(mapping, **named)`             | Append named expressions; `mapping` is optional                     |
| `t.select(*columns, **named)`                  | Keep literal column names, then append named expressions in order   |
| `t.filter(predicate)`                          | Keep rows matching a boolean `ColumnExpr` over that table           |
| `expression.pipe(function, *args, **kwargs)`   | Apply a synchronous declaration function once                       |
| `t.sql(query)`                                 | Add SQL using this table as the local alias `input`                 |
| `expression.identical(other)`                  | Compare structural identity as a Python boolean                     |

Use `&`, `|`, and `~` instead of `and`, `or`, and `not`; converting an expression
to `bool` fails. Chained comparisons, `**`, `//`, and `%` are unsupported.
`select` requires at least one column and takes literal column-name strings,
not formula source. Use named arguments for calculations. `with_columns` also accepts a `FeatureSet`
and copies mappings in insertion order before appending keyword entries.

Named columns are append-only: they cannot replace an existing input column.
`select` follows this rule for its named expressions even if the original column
is not selected. Duplicate names across mappings/keywords or projected/derived
columns fail; use a new output name. Expressions in one `with_columns` call refer
to its input table, so build another table stage before referencing a derived
column. Cross-input or incompatible row-lineage arithmetic fails analysis.

Types are strict. Ordinary integer Arrow columns and integer literals do not
silently promote for arithmetic. For floating division of integer `gross`, use
`cf.row.cast(gross, "float64") / 10.0`, as in
[example 01](../examples/01_datafusion_pipeline.py). Unsupported Arrow types or
field names fail with an input/field path. Fields and derived column names must
satisfy the portable identifier rule `[A-Za-z_][A-Za-z0-9_]*`; rename columns in
Arrow before declaring them if necessary. String literals are values, never
parsed as formulas.

## Reusable programs and collection

`cf.Program(name, /, *, mode=..., inputs=None, outputs=())` accepts outputs as a mapping
in insertion order or a sequence of `(name, TableExpr | ArrayExpr)` pairs. Omitting
`inputs` discovers reachable table inputs, native array inputs and parameters in
deterministic order,
including when outputs are added later with `program.output(...)`.
Explicit input sequences are respected, including `inputs=()`; missing referenced
inputs become analysis errors. Conflicting roots with the same name fail.
Declarations and output mappings are copied and remain immutable.
Pass exactly one of `mode="batch"|"stream"` or legacy
`engine="sql"|"streaming"` when constructing the program, before adding declarations.
Neither is a default; explicit `None` and supplying both are errors. `program.execute(inputs)` uses that fixed engine.
Immutable `with_input` and `output` calls preserve it, and incompatible graph
shapes are rejected when outputs are declared. The same declarations retain
their fingerprint in either engine; use `program.for_mode("batch"|"stream")` to copy the declarations
into an independent wrapper for the other lifecycle. Calls for the other engine raise `RuntimeError`.
`analyze` and `explain` accept an explicit opposite `mode` only for hypothetical
diagnostics; it does not change the program engine or permit opposite-mode
compilation, export, or execution.

| Method                                                                                                | Result and input contract                                                        |
|-------------------------------------------------------------------------------------------------------|----------------------------------------------------------------------------------|
| `table_expr.collect(inputs, /, *, runtime=None, options=None)`                                        | One Arrow table; one table root accepts data directly, otherwise use a mapping   |
| `array_expr.collect_array(inputs, /, *, runtime, options=None, array_options=None)`                   | One NativeArray from native array/table roots; explicit registered Runtime       |
| `program.collect(inputs, /, *, runtime=None, options=None)`                                           | `dict[str, pyarrow.Table]` in logical output order; always supply a mapping      |
| `cf.Program(name, mode=...)`                                                                          | Require `"batch"` or `"stream"`; legacy engine is an exclusive alias             |
| `program.execute(inputs, /, *, runtime=None, options=None, config=None, watermarks=None)`             | SQL: named Arrow tables; streaming: owned `StreamOutput` events                  |
| `table_expr.collect_async(...)` / `program.collect_async(...)`                                        | Awaitable forms of the same contracts                                            |
| `table_expr.stream(inputs, /, *, runtime=None, config=None, watermarks=None)`                         | Owned async iterator of Arrow tables                                             |
| `program.stream(inputs, /, *, runtime=None, config=None, watermarks=None)`                            | Owned async iterator of named `StreamOutput` events; input mapping required      |
| `program.analyze(runtime=None, /, *, mode=None)`                                                      | Immutable analysis result; defaults to the selected engine                       |
| `program.explain(runtime=None, /, *, mode=None)`                                                      | Deterministic explanation text                                                   |
| `program.compile_batch(runtime=None, /)`                                                              | Explicit batch execution plan                                                    |
| `program.compile_stream(runtime=None, /, *, allowed_lateness_micros=0, late_policy="error")`          | Explicit stream plan for a runner                                                |
| `program.to_project(runtime=None, /, *, mode=None, allowed_lateness_micros=0, late_policy="error")`   | Validated, data-only project-v3 document for the selected engine                 |

`BatchExecutionPlan.external_inputs` and `.external_outputs` are deterministic
tuples of physical binding names. Use them when executing a reloaded project;
they do not restore the symbolic Program's logical alias mapping.

`program.collect_schema(runtime=None, /, *, mode=None)` returns exact named
Arrow schemas. `table_expr.collect_schema(*, runtime=None, mode="batch")` returns
one schema. Both plan without data and reject an active event loop; use a worker
for schema planning from async code. See [relation frontends](relational-frontends.md)
for the lifecycle, compatibility and export contracts.

`execute` uses finite table inputs with the SQL engine and stream inputs with
the streaming engine. `options` is SQL-only; `config` and `watermarks` are
streaming-only. Consume streaming results with `async with` and `async for`.
See [example 29](../examples/29_sql_stream_switch.py) for both modes using the
same SQL declaration.

Collection mappings use declared input names, and the returned mapping uses
logical output names. Missing, extra, or wrong-kind inputs fail with named paths.
Static array parameters require explicit provider registration on the selected
runtime. Native parameters accept a NativeArray or a native array Batch;
NumPy/JAX parameters use `Batch.from_array`. Table collection returns tables;
collect one native ArrayExpr with `collect_array`. Mixed table/array Program
results remain unsupported. See the
[capability matrix](symbolic-api.md#symbolic-declarations-and-static-analysis).

Omitting `runtime` for table collection selects a fresh default `Runtime`; pass
one explicitly for registered providers/UDFs. Native `collect_array` requires
an explicit runtime. Every convenience `compute` or `collect` call creates
a fresh batch plan, including repeated/concurrent calls using the same runtime.
Rolling state does not carry between those calls, and explicitly compiled cached
plans are neither reused nor reset. Use explicit plans for snapshot/restore/reset,
state reuse, timings, and DataFusion metrics.

Explicit batch plans also retain up to eight input-free DataFusion query templates
for repeated immutable queries whose physical plans contain only
data-independent operators (projections, filters, aggregates, unions, sorts
and repartitions) over in-memory scans of the registered tables. Joins keep
planning every run because their build side is chosen from statistics, and
limited queries keep planning because top-k operators carry execution-updated
pruning filters.
Reuse requires the same query, aliases, exact Arrow schemas including metadata,
and effective partition count. The compiled graph owns its configuration and UDF
snapshot; templates are private to that graph. Each execution binds new input
rows into a fresh physical tree and retains its own session, memory accounting,
metrics and cancellation. Ineligible queries retain normal planning. The first
execution prepares the template; convenience collection calls keep creating
fresh plans as described above.

Project export saves the native graph and data-only input placeholders. It does
not save data, builder functions, Python expression objects, logical aliases, or
a running job. Reloaded projects and explicit runners use the physical binding
names in their plans/documents. Collection and convenience stream methods
translate those names for Python callers. See [project export and reload](projects-guide.md).

## SQL and pipeline composition

`cf.sql(query: str, /, **tables: TableExpr) -> TableExpr` declares lazy,
read-only DataFusion SQL. Aliases are explicit portable identifiers bound to
table declarations; at least one is required. Raw Arrow data are supplied only
at execution. `t.sql(query, /)` is the single-table form with local alias
`input`. Query results have a natively planned schema, so `select`,
`with_columns`, `filter`, and column arithmetic can follow SQL without a manual
result schema or an intermediate collection. Duplicate SQL output names fail.

`expression.pipe(function, /, *args, **kwargs) -> R` invokes
`function(expression, *args, **kwargs)` once during declaration. It works for
table and column expressions, retains the function's return type, and may
return a Program for multiple outputs. It rejects awaitable results; a builder
exception retains its original traceback. These functions are never runtime
UDFs or serialized callbacks.

See the complete [SQL and Python pipeline](batch-guide.md#compose-sql-and-python-pipelines)
and [multi-input SQL example](#multi-input-sql). SQL output has a distinct row
lineage and does not inherit entity/event-time/sequence ordering. Row-local
work after SQL and rolling before SQL are supported. SQL-to-event-window paths
and standalone array Program outputs are unsupported; the
[composition reference](symbolic-api.md#sql-composition) gives the full boundary.

Batch SQL accepts multiple aliases. Append stream SQL accepts exactly one.
Queries with ordinary aggregates or `GROUP BY` accumulate state across batches and emit the full
current result after each batch: `SELECT SUM(value) AS total` emits `1`, then
`6` for input batches `[1]` and `[2, 3]`. A grouped result includes unchanged
groups in each snapshot. By default, retained SQL input has no fixed row or
byte cap. Set `StreamRuntimeConfig.sql_state_budget` to
`StateBudget(20_000_000, 8 << 30)` to limit cumulative accepted input, including
compact execution. Eligible aggregates keep compact state; unsupported shapes
recompute over retained input, so their memory, checkpoint size, and query cost
grow with stream length. Set `sql_compact_state_budget=StateBudget(100_000, 64 << 20)`
to require compact execution and limit each aggregate node's live groups and
conservative owned bytes. For this field, `StateBudget.max_rows` counts live
groups, including null-only groups. Owned bytes include keys, winners, complete
ordering tuples, capacities, cached results, dirty tracking, checkpoint caches
and the simultaneously held installed/candidate state. Held captures remain
charged until released. This separate budget defaults to `None`; input/output,
validation and serialization scratch use separate engine reservations. It is
not a serialization-peak or RSS limit. Both budgets apply when configured
together; neither changes the semantic plan fingerprint.

Grouped ordered `LAST_VALUE` can compact when it has one built-in aggregate,
direct group/value/ordering columns, supported types and the certified default
DataFusion execution model. The value supports integer/Float32/Float64;
ordering supports integer/Timestamp columns. Unsupported shapes use retained
execution unless an explicit compact budget requires rejection. Empty first
input defers the ordered physical proof; compact admission fixes the mode.
Explicit budgets reject unsupported aggregates/UDFs at schema preflight when
possible, otherwise before the first nonempty input commits or emits. Native
job failures propagate through convenience streams as `StreamingRuntimeError`.

Ordering uses the parsed per-term direction/null placement: defaults are
`ASC NULLS LAST` and `DESC NULLS FIRST`. Equal complete tuples keep the first
encountered winner. Default/`RESPECT NULLS` can select null; `IGNORE NULLS`
skips null values but preserves null-only groups. Declare a business sequence
in the SQL ordering tuple if needed; no tie-breaker is implicit. Full snapshots
still cost O(keys) output work, without delta output or fixed RSS guarantees.
See [ordered LAST_VALUE](streaming-guide.md#compact-ordered-last_value) for
the exact MVP shape and existing runtime evidence.

Compatible layout-3/4 checkpoints remain supported. Nonempty retained state
does not automatically migrate and conflicts with an explicit compact budget;
use a new lineage/state root and replay to switch. Ordered layout-5 full/delta
recovery checks identity, typed state and actual fees under current budgets,
rejecting unknown/corrupt state atomically. See
[SQL recovery compatibility](streaming-guide.md#sql-aggregate-recovery-compatibility)
and [SQL stream boundaries](streaming-guide.md#stream-ownership-and-sql-boundaries).
SQL without aggregation remains batch-local,
as do SQL window functions when used without a grouped or ordinary aggregate.
`ORDER BY` and `LIMIT` apply to each emitted snapshot.

`cf.snapshot_sql(query: str, /, **tables: TableExpr) -> TableExpr` provides
stream-only SQL over complete replacement relations, including multiple
aliases. Inputs need fixed Arrow schemas and valid SQL aliases without
case-insensitive duplicates.
Each data Batch (all RecordBatches together) replaces one relation. Empty data
initializes or clears it; missing data does not. After all aliases initialize,
each update emits one complete latest-available result. EOF retains the last
relation, idle does not recompute, and no cross-source event-time atomicity or
delta JOIN is promised. The output has new lineage, no inherited ordering and
must connect directly to sinks in this MVP.

Set `StreamRuntimeConfig.sql_snapshot_state_budget=StateBudget(rows, bytes)`
to bound each node's total latest rows and conservative owned state, including
installed/candidate relations and held captures. It defaults to `None` and is
independent of the two append-SQL budgets. Every full input/result must fit its
edge budget; oversize fails without splitting. Convenience materialized inputs
at snapshot ingress are delivered whole; explicit `SourceBinding.from_arrow`
is rejected there, including through row-local expressions, because it slices
tables. It can still feed an upstream append aggregate. Custom sources own the
complete-message boundary.

Durable recovery requires an explicit runner, replayable source bindings and a
stable checkpoint root. Ordinary iterables have no replay, even with temporary
checkpoints enabled. Snapshot initialization and relations restore together
with upstream state, with strict identity/version/digest/current-budget checks.
MVP ExactlyOnce requests involving this node are rejected before sources open.
See [snapshot semantics, budgets and recovery](streaming-guide.md#complete-replacement-snapshot-sql)
and [example 31](../examples/31_streaming_account_market_values.py).

## Streaming results

`TableExpr.stream(inputs, /, *, runtime=None, config=None, watermarks=None)` returns
`StreamResults[pyarrow.Table]`. `Program.stream(inputs, /, *, runtime=None,
config=None, watermarks=None)` returns `StreamResults[StreamOutput]`. Both require
`async with` and `async for`; there is no blocking convenience stream. Run the
[stateful pipeline](../examples/33_streaming_pipeline.py) and
[named-output example](../examples/34_streaming_outputs.py), or follow the
[streaming guide](streaming-guide.md#first-python-continuous-job).

A dynamic input accepts a materialized Arrow `Table`, `RecordBatch`, table
`Batch`, a non-empty sequence of those values, an `AsyncIterable` of those
values, or a `SourceBinding`. Sequences are schema-checked and concatenated into
one materialized input, so they keep the native source and record boundaries
without a Python source call per batch. Each supplied RecordBatch, including an
empty one, keeps its boundary; Tables use their Arrow-exported boundaries.
Zero-column inputs keep their row counts. The stream call captures sequence
item references immediately, independently of later changes to the caller's list.
Materialized inputs are imported once and sliced by a native source within
`config.edge_budget`, except complete relations feeding snapshot ingress,
which must fit the budget whole. They do not call a Python source per batch. If
all dynamic inputs are materialized Arrow, a native bounded result queue also
avoids Python sink calls per batch; iterable and mixed inputs retain that bridge.
With `SourceProvidedWatermarks`, the
iterable may also yield `Watermark` values. A single-table root without static
parameters accepts that source directly. Otherwise pass a mapping by logical
declaration name; Program always requires one. Static parameter entries hold
the declared table data or array `Batch`, and are latched once. Arrays require
an explicitly registered provider on the selected runtime. All outputs must
be table expressions.

Construction copies input and watermark mappings and captures source/Batch
references without consuming an iterable or starting work. Arrow buffers stay
shared and read-only.
Context entry validates inputs, compiles one fresh native stream plan, and
starts one native job. Native state persists across input batches; a new
`stream` call creates a new owner and fresh state. Ordinary iterables enforce
the declared Arrow schema and finite row/byte limits from
`config.edge_budget`. They provide no replay. A supplied `SourceBinding` keeps
its actual capabilities and watermark policy.

With `watermarks=None`, an iterable or materialized input declaring `event_time` must have
non-null, nondecreasing timestamps in arrival order across rows, batches, and
all entities in that source. The native runtime generates a watermark at
`max_seen - 1 microsecond`, with a 100 ms emission interval. Equal timestamps
may span batches; the latest timestamp stays open until a larger one arrives,
explicit progress is supplied, or the source ends. This allows finalized rolling
results to arrive before EOF. A decrease fails; the adapter does not sort or
drop rows. Inputs without event time use disabled watermarks, and stateless
expressions and SQL produce results as batches arrive.

`watermarks` accepts an existing `WatermarkPolicy` for one dynamic input or a
mapping by logical dynamic input name. Omitted mapping entries use the default.
Select `BoundedOutOfOrderness` for disorder, `SourceProvidedWatermarks` for
iterable-provided progress, or `DisabledWatermarks` for intentional EOF-driven
finalization on supported graphs. ASOF rejects disabled watermarks on every
reachable source before startup. Explicit policies replace the default order
validation. Unknown or static names, invalid policies, and overrides of a
`SourceBinding` fail.
See [watermark policies](streaming-guide.md#watermark-policies) for inclusive
cutoffs, late-data errors, and multi-source progress.

`SourceBinding.from_arrow(data, *, budget=None, watermark_policy=None)` supplies
the same native source to an explicit `StreamingRunner`. `budget` defaults to
`EdgeBudget()` and must fit the runner's edge budget. Its default watermark
policy is `DisabledWatermarks`; select `BoundedOutOfOrderness` explicitly for
event-time progress. Materialized input cannot provide native watermarks.
The explicit factory preserves the imported Arrow schema and table `Batch`
metadata; expression streams normalize schema metadata as ordinary iterables do.
Each runner has an independent cursor; the reusable binding retains its imported
buffers until released. Jobs release their own references on close, including
failure and cancellation. Slices can retain larger backing allocations: logical
edge byte limits do not bound input storage or process RSS. Neither this cursor
nor temporary checkpoints provide persistent replay or exactly-once delivery.
Readers and arbitrary Arrow stream providers are not accepted by this factory.

`StreamResults` supports async context management, async iteration, idempotent
`aclose()`, and a `job` property available after successful entry. A context
can be entered once, and only one consumer can call `__anext__` at a time.
The native job and bounded output queue apply backpressure. Completion drains
queued output and settles cleanup. Early exit, exceptions, and cancellation
cancel live work and await source/job/task cleanup. Native failure raises
`StreamingRuntimeError` with safe structured details.

`StreamOutput` is an immutable `name: str` / `table: pyarrow.Table` event.
Outputs are ordered within each name; independent outputs have no promised
combined order or synchronized dictionary. Do not assume one result table per
source batch. Results omit native `Batch` metadata and do not acknowledge
application delivery.

Every convenience stream owns temporary managed checkpoint storage removed
after native cleanup. Ordinary iterables have best-effort delivery, and neither
the iterator nor its temporary checkpoints provide durable restart or
exactly-once processing. For those contracts, use explicit source/sink bindings,
`Program.compile_stream`, `StreamingRunner`, and `ManagedCheckpointRuntime`.
See [explicit connectors and recovery](streaming-guide.md#explicit-connectors-and-recovery).

## LateOutputs and local late policy

`cf.with_late_output(value: TableExpr, /, *, allowed_lateness_micros: int = 0)
-> LateOutputs` declares paired normal and diagnostic outputs.
`LateOutputs` is frozen; its `output: TableExpr` and `late: TableExpr`
share one native state owner. Allowed lateness must be an exact Python
`int` in `0 <= L < 2**64`; booleans are rejected. Native closing-time
arithmetic must also fit the event-time range.

Consume both references explicitly in one Program, directly or through
supported transforms. Omitting either yields `unconsumed_output`.
One `Program.stream` iterator drains both names; opening independent streams
for the two references is not a paired execution. Use the complete
[late-row example](streaming-guide.md#route-late-rows) for the checked
`normal=[20.0]`, `late=[10.0]` result with explicit
`SourceProvidedWatermarks`.

The declaration accepts one current rolling or cross-section `with_columns`
or `filter` stage. Stateful operands must be existing columns of that input;
name row-local precomputation before declaring the stage. Nested stateful
operands, mixed state families, incompatible cross-section groups, and an
already-stateful input fail with `ambiguous_late_stage`. Batch compilation
and collection fail with `unsupported_mode`. Missing outputs and invalid
graphs fail before Source opening.

The marker's SideOutput policy and `allowed_lateness_micros` apply locally.
The `late_policy="error"`/`"drop"` and lateness arguments on
`Program.compile_stream` and `to_project` configure unmarked stages;
there is no global `"side_output"` value. The marker and its settings remain
part of declaration identity and exported native configuration.

The late schema is the full stage input plus nine non-null `_cf_late_*`
diagnostics. It has a distinct row lineage and no temporal ordering; it can
feed only built-in single-input expressions or SQL before a Sink.
Read the [late-row guide](streaming-guide.md#route-late-rows) for exact fields,
closing-coordinate equality, control messages, resource/failure boundaries,
and persistent recovery. The example's logical `normal`/`late` names map
to physical `normal.output`/`late.output` Sink bindings; inspect
`plan.sink_binding_ids` when using explicit runners or exported projects.

## Bounded backward ASOF Join

`cf.table.stream_asof_join` and `TableExpr.stream_asof_join` declare a
stream-only backward match. They require `tolerance=timedelta(...)` and
`limits=AsofStateLimits(max_state_rows, max_state_bytes)`, with optional
`keys`, `late_policy`, and `prefixes`. Time/sequence come from input temporal
metadata, and omitted keys use each input's entity keys.

The operator chooses at most one historical right row per accepted left row
and preserves unmatched left rows with nullable right fields. Equal-time ties
use a typed sequence order. Both input watermarks must strictly pass the left
time, or the corresponding input must end, before any final result is emitted.
Run [example 35](../examples/35_stream_asof_join.py) and read the
[ASOF guide](asof-join-guide.md) for exact types, inclusive tolerance, late and
duplicate rules, composition, bounded resources, status, and delivery.

The immutable root exports `AsofJoinSide(keys, event_time, sequence_by, prefix)`
and `AsofJoinSpec(left, right, tolerance, limits, late_policy="error")` support
explicit graph declarations. The advanced method is
`PipelineBuilder.stream_asof_join(name, *, left_schema, right_schema, spec)`;
its schema arguments are `ArrowFieldSpec` sequences. Both declaration paths
lower to the same native operator. Batch execution and collection reject ASOF.

## Choosing an integration API

Use root `calc_flow` expressions, SQL, `pipe`, collection, and streams for
application calculations. Use the explicit controls below for operational needs:

| Need                                     | Interface                                                      |
|------------------------------------------|----------------------------------------------------------------|
| Plan state and execution diagnostics     | `Program.compile_batch`, `BatchExecutionPlan`, and `RunResult` |
| Typed runtime UDF or provider selection  | `Runtime` registrations and explicitly selected builder nodes  |
| Durable recovery or transactional sinks  | `Program.compile_stream`, explicit bindings, and managed state |
| Native graph construction and inspection | `PipelineBuilder`, ports, and validated `ProjectDocument`      |

These expose the same native graph and runtime. They do not introduce a table
backend selector or serialize arbitrary Python execution.

## Table batches and builder

This advanced interface exposes graph nodes, ports, and execution diagnostics.
Use the expression and collection methods above for ordinary calculations.

```python
import pyarrow as pa

from calc_flow import Batch, PipelineBuilder

batch = Batch.from_pyarrow(pa.table({"a": [1, 3], "b": [2, 4]}))
builder = PipelineBuilder("totals")
configured = builder.expression("calculate", "total = a + b")
plan = configured.compile_batch()
result = plan.execute({"input": batch})

assert builder.project["graph"]["nodes"] == []
assert result.outputs["output"].to_pyarrow()["total"].to_pylist() == [3, 7]
```

Builder methods return new values. `Batch.from_pyarrow` and every runner/plan
boundary treat caller inputs as read-only. Result mappings and metadata are
defensive copies.

Arrow import resolves the current C stream exporter once per call. Arrow export
transfers owned record-batch references through the existing stream protocol,
sharing read-only buffers and preserving schema metadata and native record-batch
boundaries.
Repeated exports remain independent; each resolves the current PyArrow table
constructor. Reading `RunResult.outputs` copies the output mapping without
copying unrelated query metrics and run metadata.

An expression node accepts exactly one calculation expression or a non-empty
`select` projection; `filter` may accompany either mode. Connect nodes with
`connect(source_node, target_node, source_port="output", target_port="input")`.

## Multi-input SQL

This complete [02_sql_join.py](../examples/02_sql_join.py) example composes a
named SQL join and a Python column transform:

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
        raise RuntimeError(f"unexpected SQL join result: {result.to_pydict()}")
    print(result.to_pydict())


if __name__ == "__main__":
    main()
```

`Batch.from_pyarrow(table, active_entities=n)` (and `Batch.from_array`) records
a trusted positive active-entity count under
`calc_flow.datafusion.active_entities` for `auto` partition selection. Zero or a
conflicting existing value is rejected. This is the supported way to make
single-input batch graphs eligible for entity-aware automatic parallelism.
See [SQL and DataFusion performance](sql-datafusion-performance.md).

The SQL aliases `o` and `f` name the declarations inside the query; collection
binds data by logical input names `orders` and `fees`. The final doubled values
are `[140, 216, 72]`. Only one read-only DataFusion `SELECT` or CTE is accepted;
DDL, DML, utility commands, and multiple statements fail validation.

Use the immutable builder method to select parallelism and diagnostic controls:

```python
builder = PipelineBuilder("parallel-sql").with_datafusion_config(
    parallelism_mode="auto",
    max_partitions=16,
    min_rows_per_partition=65_536,
    small_rows_threshold=10_001,
    enable_rolling_rewrite=True,
    collect_diagnostics=True,
)
```

For batch graphs containing multi-input SQL, auto mode chooses one run-scoped
partition budget from total external table rows and host capacity before the
first node executes. Other graphs and runtimes require trusted
`calc_flow.datafusion.active_entities` batch metadata and otherwise use p1.
Both policies avoid scanning rows. `parallelism_mode="auto_rows"` additionally
uses input rows when no trusted active-entity metadata is supplied; entity
statistics, when present, only lower the row-based request. `auto_plan` instead
pins a batch graph to p1 only when every node is a proven simple row-local SQL
projection; other graphs use the row-based policy. Fixed p1 remains the default.
See [SQL and DataFusion performance controls](sql-datafusion-performance.md)
for all fields, telemetry, evidence gates, and rollback steps.

## Trusted Python scalar UDFs

```python
import pyarrow as pa
import pyarrow.compute as pc

from calc_flow import PipelineBuilder, Runtime

runtime = Runtime()


def double_amount(amount: pa.Array) -> pa.Array:
    return pc.multiply(amount, 2)


runtime.register_scalar_udf(
    provider="python",
    name="double_amount",
    version="1",
    input_types=("int64",),
    return_type="int64",
    volatility="immutable",
    function=double_amount,
)
plan = (
    PipelineBuilder("registered-udf")
    .expression(
        "calculate",
        "total = double_amount(amount)",
        udfs=(("python", "double_amount", "1"),),
    )
    .compile_batch(runtime)
)
```

Callbacks are trusted application code, vectorized over PyArrow arrays, and
never serialized. Registration and execution enforce exact Arrow types,
result length/type, and explicit node references. The full version is
[`examples/03_registered_udf.py`](../examples/03_registered_udf.py).

## Runtime capabilities

`Runtime.capabilities()` returns a frozen `RuntimeCapabilities` value. It is a
data-only snapshot: callbacks and source/import paths are neither returned nor
inspected.

```python
from calc_flow import (
    ProviderOption,
    ProviderOptionsSchema,
    Runtime,
)


def normalize_callback(batch, options):
    return batch


runtime = Runtime()
runtime.register_provider(
    "acme",
    "normalize",
    "1",
    normalize_callback,
    options_schema=ProviderOptionsSchema(
        fields=(ProviderOption("scale", "number", required=True),)
    ),
)
snapshot = runtime.capabilities()

assert snapshot.schema_version == 5
assert snapshot.scope.kind == "runtime_session"
assert snapshot.scope.revision == 1
assert snapshot.providers[0].name == "normalize"
```

The public frozen values are `ProviderOption`, `ProviderOptionsSchema`,
`ProviderPort`, `ProviderCapability`, `UdfCapability`, `OperatorCapability`,
`ConnectorCapabilities`, `ConnectorCapability`, `CapabilityRule`,
`ProviderArrayRules`, `RuntimeSessionScope`, and
`RuntimeCapabilities`. Provider option schema
version 1 supports only named scalar string, integer, number, or boolean
fields. `options_schema=None` means no declarative editor is available; it
does not mean every option is valid. The provider callback remains
authoritative during compilation.

Capability schema version 5 adds `ProviderCapability.state_layouts` to the
closed provider record. Version 4 introduced option-defined provider ports;
consumers must adopt the version-5 record explicitly. Project and manifest
formats remain 3. `OperatorCapability` and
`ProviderCapability` report
`modes`, `finality`, `stateful`, `microbatch_invariant`, `requires_watermark`,
`checkpoint_support`, `state_version`, `deterministic`, and `replay_safe`;
both report `state_layouts`, the advertised checkpoint layout inventory.
Inspect the native operator contract when evaluating checkpoint compatibility.
Providers additionally report
`supports_static_inputs`, `partition_contract`,
and optional `array_rules`. The vocabularies are closed: execution modes are
`batch` and `stream`; output finality is `per_row_final`,
`group_final_append_only`, or `unproven`; checkpoint support is `stateless`,
`checkpointed_stateful`, or `unproven`; and a provider partition contract is
`none`, `row_axis_independent` or `single_ordered_sequence`. The last means one
state owner consumes vector positions in single-ingress arrival order, without
independent partitions or implicit input alignment. `state_version` is a
positive integer
exactly when `checkpoint_support` is `checkpointed_stateful` and `None`
otherwise, a `stateless` capability must set `stateful=False`, and
`state_layouts` is a strictly ascending tuple of positive integers. It must
be nonempty for `checkpointed_stateful` and empty otherwise. The semantic
`state_version` and physical layout numbers are independent; the inventory
does not have to contain `state_version`.
Construction validates strictly and fails closed: closed-vocabulary and
cross-field violations raise `ValueError`, while non-strict data (a `list`
where a tuple is declared, a non-`bool` boolean field) raises `TypeError`.

`ProviderCapability.port_binding` defaults to `"fixed"`, preserving existing
port tuples and `options_schema`; `options_json_schema` defaults to `None`.
With `port_binding="options"`, both fixed port tuples are empty because the
required names, kinds and flags come from `options.inputs` and
`options.outputs`. This mode requires immutable strict JSON Schema text in
`options_json_schema`, and `options_schema` must be `None`. The two schema
fields cannot be combined. Empty tuples in this mode do not mean zero ports.
Fixed-port capabilities may also carry strict `options_json_schema` text when
`options_schema is None`; their explicit port tuples remain authoritative.
Old fixed-port records without JSON Schema retain their defaults.

`CapabilityRule` identities are closed and versioned. The accepted identities
are `array_api_safe_dtype@1`, `elementwise_broadcast@1`,
`feature_axis_reduction@1`, `table_matmul_static_rhs@1`,
`native_float_bool_dtype@1`, `native_scalar_vector_broadcast@1`,
`native_packed_rank2@1`, `native_matmul@1`, `native_axis_reduction@1`,
`native_table_static_rhs@1`, `native_float_dtype@1`,
`native_sequence_vector@1`, `native_sequence_indicators@2` and
`native_sequence_indicators@3`; any other
name/version pair fails construction.
The three matrix rules describe IR2 packed shapes, multiplication and axis
reductions; IR1 retains its scalar/vector shape rules. The sequence vector rule
retains IR1 vector shapes. The indicator rule names IR2's per-kernel alternatives:
RSI uses a vector and ATR uses packed high/low/close rows. The version-3 rule
adds MACD vector input with packed `[n, 3]` line/signal/histogram output. Dispatch by IR version
and kernel; listed alternatives do not allow every kernel to use every shape.
These rules establish neither table lineage nor a general array stream API.
`ProviderArrayRules` pairs the exact
`supported_dtypes` tuple with a `safe_dtype_rule` and `shape_rules`, and
stores both tuples sorted by identity.

The `operators` tuple contains exactly `cross_section@1`, `expression@1`,
`rolling@1`, `sql@1`, `stream_asof_join@1`, `stream_join@1`, and `window@1`,
with truths anchored in
the engine implementation:

| Operator             | Modes         | Finality                | Checkpoint support    | State version | State layouts    |
|----------------------|---------------|-------------------------|-----------------------|---------------|------------------|
| `cross_section@1`    | batch, stream | group_final_append_only | checkpointed_stateful | 1             | 1                |
| `expression@1`       | batch, stream | per_row_final           | stateless             | —             | —                |
| `rolling@1`          | batch, stream | per_row_final           | checkpointed_stateful | 1             | 1, 2, 3, 4, 5, 6 |
| `sql@1`              | batch, stream | unproven                | checkpointed_stateful | 2             | 3, 4             |
| `stream_asof_join@1` | stream        | group_final_append_only | checkpointed_stateful | 3             | 10               |
| `stream_join@1`      | stream        | unproven                | checkpointed_stateful | 1             | 1, 2, 3          |
| `window@1`           | batch, stream | group_final_append_only | checkpointed_stateful | 1             | 1, 2             |

The rolling inventory includes retained-history and accumulator layouts as well
as columnar layout `3` and cumulative layout `4`. See
[native rolling state](symbolic-design.md#native-rolling-state) for the
implemented encoding and restore rules.

`cross_section@1`, `rolling@1`, `stream_asof_join@1`, `stream_join@1`, and
`window@1` describe stream state and watermark requirements. Finite windows
finalize the complete input in fresh temporary state without watermarks or
checkpoint recovery. Aggregate-bearing
`sql@1` is stateful without requiring one. Row-local SQL remains stateless.
`cross_section@1`, `expression@1`, `rolling@1`, `stream_asof_join@1`, and
`window@1` are micro-batch invariant. All seven
report `deterministic=True` and `replay_safe=True`. For `sql@1` those two claims
hold from the engine viewpoint: exactly-once stream
plans reject nodes that select volatile registered UDFs, and stream
compilation rejects read-only queries that call volatile built-in SQL
functions such as `random()` or wall-clock built-ins such as `now()`,
`current_date()`, and `current_time()` (aliases included).

Neither `register_provider` nor `register_mapping_provider` accepts lifecycle
metadata, so a registered provider's entry is always batch-only with
conservative values: `modes=("batch",)`, `finality="unproven"`,
`stateful=False`, `microbatch_invariant=False`, `requires_watermark=False`,
`checkpoint_support="stateless"`, `state_version=None`,
`deterministic=False`, `replay_safe=False`, `supports_static_inputs=False`,
`partition_contract="none"`, and `array_rules=None`. Lifecycle keys forged
onto a [trusted registration record](#trusted-registration-records) are
unlisted keys: they are ignored rather than upgrading the entry, and omission
never opts a provider into stream execution. `finality="unproven"` means
registration evidence establishes no output-finality contract; it is the
truthful conservative value for a batch-only registration and does not narrow
existing batch selectability.

The trusted `register_numpy` and `register_jax` helpers additionally attach
process-local stateless stream proofs to their `expression@1` and
`symbolic_matrix@1` registrations. Those provider entries report
`modes=("batch", "stream")`,
`finality="per_row_final"`, `microbatch_invariant=True`,
`deterministic=True`, `replay_safe=True`, and a `row_axis_independent` array
contract. `expression@1` carries `elementwise_broadcast@1` and does not support
static inputs; `symbolic_matrix@1` carries the matrix shape rules and does.
The public `Runtime.register_provider` and
`Runtime.register_mapping_provider` signatures have no lifecycle arguments,
so an arbitrary Python callback remains batch-only rather than acquiring this
proof from caller-supplied metadata.

`calc_flow.array.register_native(runtime)` atomically registers the Rust
`native:array_expr@1` batch factory, `native:table_array@1` batch/stream
bridge and `native:array_sequence@1` stateful stream factory. These are four
lifecycle entries under three identities. Their frozen capabilities and complete
options JSON Schemas come from Rust. The array/table factories use option-defined
ports; the sequence factory has explicit fixed ports. The array factory supports native Boolean/float32/float64
scalars, vectors and packed matrices. Only true scalars broadcast; `[1]` remains
a vector, and non-scalar operands require matching shapes. Numeric promotion
preserves float32 unless an operand is float64. IR version 1 retains its original
scalar/vector contract; IR version 2 adds rank-two storage, matmul, reshape,
transpose and axis reductions.

The array factory reports `modes=("batch",)`, `stateful=False`,
`microbatch_invariant=False`, `requires_watermark=False`,
`checkpoint_support="unproven"`, `state_version=None`,
`supports_static_inputs=True`, `partition_contract="none"` and
`finality="unproven"`. Static inputs here cover finite batch parameters.
Its `state_layouts=()`; rolling state starts fresh on every batch invocation.
The table bridge reports
`modes=("batch", "stream")`, `stateful=False`, `microbatch_invariant=True`,
`checkpoint_support="stateless"`, `supports_static_inputs=True`,
`partition_contract="row_axis_independent"`, `state_layouts=()` and
`finality="per_row_final"`. It admits one table input with row-preserving native
work and an optional
latched static matrix RHS. Its stream mode does not add high-level ArrayExpr
streaming. All three native factories report `deterministic=True` and
`replay_safe=True` for identical inputs on the same implementation and numerical
profile; these claims do not promise cross-platform bitwise equality or delivery
guarantees. The sequence factory instead reports `modes=("stream",)`,
`stateful=True`, `microbatch_invariant=True`, `requires_watermark=False`,
`checkpoint_support="checkpointed_stateful"`, `state_version=1`,
`state_layouts=(1, 2, 3)` for the IR1/IR2/IR3 implementation, `supports_static_inputs=False`,
`partition_contract="single_ordered_sequence"` and `finality="per_row_final"`.
It uses explicit required array `input`/`output` ports and its own closed options
schema. Its checkpoint and managed restart contract belongs to that native
factory, not to a Python stateless-provider marker.
Its earlier IR1-only profile remains valid frozen evidence with layouts `(1,)`
and `native_sequence_vector@1`. The IR1/IR2 profile requires both supported
layouts, the vector rule and `native_sequence_indicators@2`, and the complete
closed schema alternatives. The IR3 profile adds layout 3,
`native_sequence_indicators@3` and a third schema branch for MACD while preserving
the two earlier branches. Incomplete or crossed profile evidence fails
validation. Registration always records its actual Rust producer's descriptor.

Use the array/table factories through native ArrayExpr declarations, advanced
`PipelineBuilder.external` or project surfaces. The sequence factory uses
advanced stream bindings. Registration retains no Python callback or provider
root. It prepares all three frozen capabilities before atomic
native insertion, publishes three registration records and invalidates symbolic
caches once. A collision in any native lifecycle, invalid descriptor, or cleared
runtime leaves records, capability revision and caches unchanged. Querying the
Rust descriptors alone does not register a provider.

Compiled-in connector registrations surface on the snapshot's `connectors`
tuple as `ConnectorCapability` entries. Each entry pairs its
`(provider, name, version)` identity and `source`/`sink`/`both` kind with a
`ConnectorCapabilities` value (`delivery`, `replay`, `watermark`,
`transaction`, and the `snapshot`/`polling`/`cdc`/`lookup` flags), the
declared formats, and an options schema.

The session ID is stable for one runtime. Revision equals the number of
successful registration records. The native bundle advances it by three across
four lifecycle entries; a rejected bundle leaves it unchanged. NumPy/JAX helpers add
`expression@1`, mapped `table_matmul@1`, and mapped `symbolic_matrix@1` as
separate entries, so one helper normally advances by three and can expose a
real partial success if a later entry already exists. Previously returned
snapshots remain isolated from later revisions. Snapshots are immutable and
defensively copied: mutating a caller-owned sequence after registration cannot
change a returned snapshot. Operators sort by `(kind, version)`; providers and
UDFs sort by `(provider, name, version)`.

### Trusted registration records

`capabilities()` is the data-only view of a runtime. Two further `Runtime`
methods return the registrations themselves, for an embedder that lists them
or rebuilds them on another runtime:

```python
def registration_snapshot(self) -> tuple[dict[str, Any], ...]: ...


def capability_registration_snapshot(
    self,
) -> tuple[RuntimeCapabilities, tuple[dict[str, Any], ...]]: ...
```

`registration_snapshot()` returns this runtime's successful trusted
registrations, oldest first. `capability_registration_snapshot()` returns
`capabilities()` and `registration_snapshot()` as one consistent pair. Each
record is a fresh plain `dict`. Python registrations retain their live
callable; a `native_provider` record holds identity and a frozen Rust-derived
`ProviderCapability`, without a callable. Records are process-local trusted
values, not a project serialization format.

Records are returned in registration order. The table describes the keys Calc
Flow produces today: a record carries every listed key with the listed type
and presence rule. A record may also carry keys the table does not list (see
unlisted keys below). Key order inside a record is not guaranteed.

| Kind              | Key                             | Value type                          | Present                            |
|-------------------|---------------------------------|-------------------------------------|------------------------------------|
| `provider`        | `kind`                          | `str`, equal to `"provider"`        | always                             |
| `provider`        | `provider`, `name`, `version`   | `str`                               | always                             |
| `provider`        | `callback`                      | callable                            | always                             |
| `provider`        | `options_schema`                | `ProviderOptionsSchema` or `None`   | always                             |
| `provider`        | `accepts_context`               | `bool`, equal to `True`             | only when registered with `True`   |
| `provider`        | `provider_mode`                 | `str`, equal to `"mapping"`         | only for mapping providers         |
| `provider`        | `input_ports`, `output_ports`   | `tuple[tuple[str, str], ...]`       | only for mapping providers         |
| `native_provider` | `kind`                          | `str`, equal to `"native_provider"` | always                             |
| `native_provider` | `provider`, `name`, `version`   | `str`                               | always                             |
| `native_provider` | `capability`                    | frozen `ProviderCapability`         | always                             |
| `scalar_udf`      | `kind`                          | `str`, equal to `"scalar_udf"`      | always                             |
| `scalar_udf`      | `provider`, `name`, `version`   | `str`                               | always                             |
| `scalar_udf`      | `input_types`                   | `tuple[str, ...]`                   | always                             |
| `scalar_udf`      | `return_type`, `volatility`     | `str`                               | always                             |
| `scalar_udf`      | `function`                      | callable                            | always                             |
| `kafka_decoder`   | `kind`                          | `str`, equal to `"kafka_decoder"`   | always                             |
| `kafka_decoder`   | `name`, `version`               | non-empty `str`                     | always                             |
| `kafka_decoder`   | `function`                      | callable                            | always                             |

- A `provider` record without `provider_mode` came from `register_provider`.
  A missing `accepts_context` means `False`.
- Port pairs are `(port_name, batch_kind)` in declaration order;
  `batch_kind` is `"table"` or `"array"`.
- A `kafka_decoder` record has no `provider` key and exists only on wheels
  built with `connector-kafka`.
- A `native_provider` record comes from `calc_flow.array.register_native`;
  its three records describe one array/sequence/table factory bundle. Replay
  that bundle
  by calling the helper once on the target runtime, rather than once per record.
  Missing identities or a contradictory sequence lifecycle fail validation;
  native snapshots never silently downgrade to a two-identity bundle.
- The listed keys of a record are the arguments of the one successful call
  that produced it: `register_provider`, `register_mapping_provider`,
  `register_scalar_udf`, `register_kafka_decoder`, or
  `calc_flow.array.register_native`.

Read records as a tolerant reader. Adding a record kind or adding a key is
not a breaking change. Removing or retyping a listed key, changing a presence
rule, or changing record order is breaking and is announced in
[CHANGELOG.md](../CHANGELOG.md) with its migration. Consumers read `kind`
first and use only the listed keys. A consumer that lists registrations may
skip kinds it does not handle. A consumer that replays them must treat a
skipped kind as a registration missing from the target runtime, and fail or
report it when the workload references it.

Replaying a record through the public registration methods reproduces its
identity, callable, ports, options schema, and context opt-in. For scalar UDFs
and batch-only providers that is the whole registration. That is not the whole
registration for the `expression@1` and `symbolic_matrix@1` providers
installed by `register_numpy` and `register_jax`: their stream capability
comes from a process-local proof that no public registration method accepts,
so a replayed copy is batch-only.
To get them with stream capability on another runtime, call the same helper
there instead of replaying their records. Do not present a replayed runtime
as equivalent to its source unless the `ProviderCapability` entries of the
replayed providers compare equal on both.

Records can carry unlisted keys holding private process-local values; today
`stream_lifecycle` on the records of those helper-installed providers is the
only one. Unlisted keys carry no compatibility promise, and no public
registration method accepts one, so a lifecycle key added to a record never
upgrades a provider. Build each registration call, and anything you
transport, from the listed keys only.

A snapshot copies the containers and shares the values:

- Fresh on every call: the outer `tuple` and each record `dict`. Adding,
  replacing, or deleting keys in a returned record never affects the runtime,
  a later snapshot, or another caller's snapshot.
- Shared by reference: every value. `callback` and `function` are the
  registered objects themselves, with their closures and state.
  `options_schema` is the frozen instance passed at registration. Port and
  input-type values are immutable tuples built from the caller's sequences at
  registration. A native record's `capability` and its nested array rules are
  frozen values; its JSON Schema is immutable text.
- All shared values other than the callables are immutable, so sharing is
  observable only through a callable.
- Neither snapshot method calls, inspects, or serializes a registered
  callable.
- Records are not data-only. They are not JSON-serializable and must not
  enter a project, checkpoint, catalog, REST payload, or log. Projects still
  name a registration only by its identity, and `capabilities()` remains the
  data-only view.
- Calc Flow makes no transport promise. Moving a record to another process
  means serializing its callable under the caller's own trust decision;
  deserializing it there executes code.

`capability_registration_snapshot()` derives both values from one copy of the
runtime's records, which is what makes the pair consistent. A snapshot never
observes a single registration call half-applied; a helper such as
`register_numpy` is several such calls and is not atomic. For one returned
pair `capabilities, records`:

- `records` are the registrations that `capabilities` describes at
  `capabilities.scope.revision`.
- `capabilities.providers` has exactly one entry per `provider` or
  `native_provider` record and
  `capabilities.udfs` one per `scalar_udf` record, with the same identities.
  Capability entries are sorted by identity while records keep registration
  order, so match them by `(provider, name, version)` rather than by
  position. `kafka_decoder` records have no capability entry.
- With no registration call, including a `register_numpy` or `register_jax`
  call, running or completing in between,
  `capabilities == runtime.capabilities()` and
  `records == runtime.registration_snapshot()`.

Across calls on one `Runtime`, `scope.session_id` is constant and
`scope.revision` never decreases; a successful registration advances it as
described above. Compare revisions for equality only: do not derive a record
count or a number of registrations from one.

On one `Runtime`, equal `(scope.session_id, scope.revision)` means the same
registrations in the same order with the same callables. It does not mean
equal provider lifecycle fields: the stream-proof step of `register_numpy` and
`register_jax` updates an existing record without advancing the revision.
Through the public API the difference is observable only when another thread
takes a snapshot while one of those helpers is running.

The example registers a mapping provider, takes one pair, and replays its
records onto a second runtime through the public registration methods:

```python
from calc_flow import Runtime
from calc_flow.array import register_native
from calc_flow.capabilities import runtime_capabilities


def scale(inputs, options):
    return {"output": inputs["weights"]}


parent = Runtime()
parent.register_mapping_provider(
    "acme",
    "scale",
    "1",
    scale,
    input_ports=(("table", "table"), ("weights", "array")),
    output_ports=(("output", "array"),),
)
register_native(parent)

capabilities, records = parent.capability_registration_snapshot()
assert capabilities.providers[0].name == records[0]["name"] == "scale"
runtime_capabilities(
    session_id=capabilities.scope.session_id,
    revision=capabilities.scope.revision,
    package_version=capabilities.package_version,
    registrations=records,
)  # Validate complete native bundle evidence before replaying any records.

worker = Runtime()
native_bundle_replayed = False
for record in records:
    if record["kind"] == "native_provider":
        if not native_bundle_replayed:
            register_native(worker)
            native_bundle_replayed = True
        continue
    if record["kind"] != "provider":  # read kind before any other key
        raise ValueError(f"cannot replay a {record['kind']} registration")
    identity = (record["provider"], record["name"], record["version"])
    options = {
        "options_schema": record["options_schema"],
        "accepts_context": record.get("accepts_context", False),
    }
    if record.get("provider_mode") == "mapping":
        worker.register_mapping_provider(
            *identity,
            record["callback"],
            input_ports=record["input_ports"],
            output_ports=record["output_ports"],
            **options,
        )
    else:
        worker.register_provider(*identity, record["callback"], **options)

# A batch-only provider replays completely; a helper-installed one does not.
assert worker.capabilities().providers == capabilities.providers
```

This example replays within the same native implementation/profile. Frozen
IR1 evidence is not rewritten into IR2 evidence. A target registration queries
its own Rust descriptor; equality or portability across implementation profiles
and package upgrades is not promised by these process-local records.

## Execution options and provider context

Use the frozen native `ExecutionOptions` value to attach run-scoped settings
and an absolute UTC deadline:

```python
from datetime import datetime, timedelta, timezone

from calc_flow import ExecutionOptions

options = ExecutionOptions(
    settings={"request": {"tenant": "demo", "attempt": 1}},
    deadline=datetime.now(timezone.utc) + timedelta(seconds=30),
)
result = plan.execute({"input": batch}, options=options)
```

Every object position in `settings` may use a
`collections.abc.Mapping`. Calc Flow calls each mapping's `items()` once,
consumes that iterator once, and copies it to a built-in `dict`; it does not
consult `len()`, `keys()`, or `__getitem__()`. Mapping subclasses are accepted,
while sequence containers must be exact built-in `list` values. Object keys
and leaves must be exact built-in `str`, `None`, `bool`, `int`, or finite
`float` values. Integers must be in `-2**63 .. 2**64 - 1`. Coercion-only
objects, unsupported subclasses, tuples, sets, bytes, duplicate keys,
surrogate code points, excessive depth, and cycles are rejected. Validation
errors expose only stable settings paths and fixed messages; exceptions from
caller mappings are not retained or chained.

Construction deep-copies the complete accepted graph; mutating the source or
any nested caller container cannot change the options. Every
`options.settings` read returns another deep `dict`/`list` copy, so mutating an
observation cannot change a later read or execution. Omitting `settings`
creates an empty mapping, and passing `None` explicitly has the same meaning.
`deadline` accepts `None` or any valid timezone-aware `datetime`. Calc Flow
normalizes every accepted offset to `datetime.timezone.utc` and preserves
microseconds; it rejects naive, invalid, and out-of-range UTC conversions with
fixed redacted errors.

The `ExecutionOptions(settings, deadline)` constructor accepts positional or
keyword arguments. In contrast, `options` is keyword-only in both
`plan.execute(inputs, *, options=None)` and
`plan.execute_async(inputs, *, options=None)`. Omitting the plan option
preserves the existing default behavior.

Provider callbacks remain two-argument callables unless the registration
explicitly opts into run context:

```python
def contextual_provider(batch, provider_options, context):
    tenant = (context.settings or {}).get("request", {}).get("tenant")
    return batch


runtime.register_provider(
    "acme",
    "contextual",
    "1",
    contextual_provider,
    accepts_context=True,
)
```

A single provider callback receives one `Batch`: `(batch, provider_options)`
when `accepts_context=False`, the default, or
`(batch, provider_options, context)` when it is true. A mapping provider
registered with `register_mapping_provider` receives its named input mapping
instead: `(inputs, provider_options)` when false or
`(inputs, provider_options, context)` when true.

Each callback is invoked exactly once under the selected ABI. The frozen,
engine-created `ProviderContext` exposes the authoritative run
`context.settings` and `context.deadline`, not values merged into the separate
compile-time `provider_options` mapping. Every settings read returns a fresh
deep copy, and the deadline is the normalized aware UTC value or `None`. The
flag must be an exact `bool`; Calc Flow does not infer arity or retry a
callback after `TypeError`. Native cancellation tokens are intentionally not
part of the public Python API.

Run settings, deadlines, and provider-context opt-in are not serialized into
projects or checkpoints.

`ExecutionOptions.deadline` is an absolute cooperative engine deadline.

## Async execution

```python
import asyncio
from datetime import datetime, timedelta, timezone

import pyarrow as pa

import calc_flow as cf


async def run() -> None:
    options = cf.ExecutionOptions(
        settings={"request": {"source": "async-example"}},
        deadline=datetime.now(timezone.utc) + timedelta(seconds=30),
    )
    heartbeat = asyncio.create_task(asyncio.sleep(0, result="event loop remained live"))
    execution = asyncio.create_task(
        cf.compute_async(
            pa.table({"a": [1, 3], "b": [2, 4]}),
            lambda t: t.select(total=t["a"] + t["b"]),
            options=options,
        )
    )
    print(await heartbeat)
    output = await execution
    if output["total"].to_pylist() != [3, 7]:
        raise RuntimeError(f"unexpected async totals: {output.to_pylist()}")
    print(output.to_pylist())
```

Blocking `compute`, `collect`, `execute`, store, and runner methods reject a
running event loop. Use
their async forms in servers and asyncio applications. `plan.execute()` checks
for a running event loop before it validates inputs or options, so that usage
error has precedence. An already-expired or crossed execution deadline raises
`calc_flow.CancelledError` after transactional rollback. Cancelling a
still-pending surrounding asyncio task instead raises
`asyncio.CancelledError`. The handler makes one terminal-state decision: if
native execution is already terminal, its result or exception remains
observable; otherwise it sends exactly one native cancellation request and
waits through repeated Python task cancellation until cleanup finishes. Once
the cancellation request is sent, the caller's `asyncio.CancelledError` wins
over any native outcome observed during that drain: a native failure landing
mid-drain is retrieved and discarded, never re-raised to the caller.

Awaiting task cancellation waits until the current native operation and
run-owned cleanup finish; no work or input payload continues detached. The
plan recovers its pre-run state before its next public operation. Deadline and
task cancellation are cooperative at safe boundaries, so neither preempts a
Python callback, DataFusion query, or other non-cooperative operation already
in progress; cleanup resumes when that operation yields. The same
`ExecutionOptions` value can be reused concurrently because each run receives
independent native cancellation state. No cancellation token is part of the
public Python API.

For executions queued behind another invocation of the same plan, the
absolute deadline keeps elapsing. Cancelling a queued invocation neither
cancels the active run nor creates partial plan state. Once a deadline or
accepted task cancellation is observed at a post-provider boundary, it wins
over that provider's error. Recovery, input, snapshot, and transaction-marker
failures that occur before the first deadline check retain their existing
precedence.

The full version is [`examples/05_async_execution.py`](../examples/05_async_execution.py).

## Native arrays

`calc_flow.array` exposes owned CPU arrays and synchronous Rust calculations.
The native owner accepts float32/float64 and Boolean scalars, vectors and
packed row-major matrices. Arrow imports
share read-only buffers and preserve offsets and null masks. NumPy imports
copy the caller's values before computation; NumPy is optional and imported
only for NumPy conversion.

```python
import pyarrow as pa
import calc_flow.array as cfa
from calc_flow import Batch

values = cfa.from_arrow(pa.array([2.0, None, 4.0]))
assert values.dtype == "float64"
assert values.shape == (3,)
assert cfa.sum(values).to_arrow().to_pylist() == [None]
assert cfa.mean(values, skip_nulls=True).to_arrow().to_pylist() == [3.0]
assert cfa.ewma(values, span=3).to_arrow().to_pylist() == [2.0, 2.0, 3.0]
batch = Batch.from_array(values, backend="native", metadata={"source": "prices"})
assert batch.array.to_arrow().to_pylist() == [2.0, None, 4.0]
```

`NativeArray` has read-only `dtype`, `shape`, `element_count`, `estimated_bytes` and
`retained_buffer_bytes` properties. `len(array)` counts logical rows, including
one row for a scalar. Matrices count their first dimension as rows;
`element_count` counts flat positions, so shape `(3, 0)` has three rows and no
positions. Arrow export is flat; NumPy export restores the stored shape. Byte estimates charge visible values, validity and shape;
retained bytes count backing-buffer capacities, rather than process RSS.
`from_arrow(array, shape=())` declares a scalar and requires one element;
the default shape is `(len(array),)`. `from_numpy(ndarray)` accepts exact native-endian
bool/float32/float64 ndarrays with zero, one or two dimensions, including
empty vectors and zero-sized matrix axes.

`to_arrow()` preserves validity. `to_numpy()` rejects arrays containing nulls
unless a numeric `fill_value` is supplied, for example
`values.to_numpy(fill_value=float("nan"))`. Boolean arrays also accept Boolean
fill values. Only null positions are filled;
valid NaN values remain valid NaNs. NumPy exports use immutable bytes storage:
neither the returned array nor its base arrays can regain write access.
Conversions never mutate the caller's array.

`sum`, `mean`, `min` and `max` return scalar native arrays by default. Nulls propagate
unless `skip_nulls=True`; empty or all-null inputs produce null scalars.
Sum and mean return float64; min and max retain the input dtype. Valid NaNs
follow the native numerical rules and are distinct from nulls.
`ewma(values, span=..., min_periods=1)` returns an aligned float64 vector,
seeds from the first valid value, and shares the table EWMA transition.
Null and NaN observations are skipped; their readout carries prior state once
the valid-observation warmup is met. `min_periods` may exceed `span`.

`rolling_mean(values, window=..., min_periods=1, numerical_profile="stable_v1")`
uses a row window: nulls and NaNs occupy positions without counting as valid
samples. `min_periods` must be between one and the window size. The default
retains the table `stable_v1` transition; `stable_v2` selects the existing
preview profile explicitly. Both return aligned float64 vectors.

`rsi(values, period=n)` and `atr(high, low, close, period=n)` use Wilder
transitions and return nullable float64 vectors aligned to the inputs. Period
is required and must be an exact positive Python integer no larger than
`2**64 - 1`; Boolean values are rejected. Inputs must be immutable NativeArray
float32/float64 vectors. ATR permits mixed floating dtypes, requires equal
lengths and uses no broadcasting. The eager functions need no provider
registration or NumPy/JAX import.

```python
import pyarrow as pa
import calc_flow.array as cfa

close = cfa.from_arrow(pa.array([10.0, 11.0, 12.0, 11.0, 11.0]))
high = cfa.from_arrow(pa.array([11.0, 12.0, 13.0, 12.0, 12.0]))
low = cfa.from_arrow(pa.array([9.0, 10.0, 11.0, 10.0, 10.0]))
assert cfa.rsi(close, period=2).to_arrow().to_pylist() == [
    None,
    None,
    100.0,
    50.0,
    50.0,
]
assert cfa.atr(high, low, close, period=2).to_arrow().to_pylist() == [
    None,
    None,
    2.0,
    2.0,
    2.0,
]
```

The first complete row establishes the previous price/close. The first valid
output follows `period` differences or true ranges, at index `period` in a
complete segment. A null or NaN resets warm-up; any missing ATR operand resets
all three inputs' segment. Valid infinities and unrepresentable finite
differences fail the call before publishing output. ATR checks every valid
operand for infinity even when another operand is missing, and validates
finite `high >= low` and `high - low` whenever that pair is present.

Flat RSI is 50; a positive-gain/zero-loss mean gives 100 and the reverse gives
0. Seeds and later means use the specified float64 incremental operation
order. Underflow can change the normalized ratio substantially: period 2 over
`[0.0, 2**-1074, 0.0]` yields `[None, None, 100.0]`, although its exact rational
ratio would give 50. All valid results follow the float64 transition, including
that case. Period one first returns at the second complete row. Large periods
use constant numerical state; `period=2**64 - 1` cannot complete warm-up before
the checked processed-position counter is exhausted.

`macd_components(values, fast_span=12, slow_span=26, signal_span=9,
min_periods=1, signal_min_periods=1)` returns a frozen, slotted
`calc_flow.MacdOutputs[NativeArray]`, also exported by `calc_flow.array`.
Its `line`, `signal` and `histogram` fields hold three immutable, aligned
nullable float64 arrays. Input must be one native float32/float64 vector;
`from_arrow` and `from_numpy` remain the explicit import boundaries.

All five counts are exact positive u64 values, with `fast_span < slow_span`.
Both minima may exceed their respective spans. The line is fast EWMA minus
slow EWMA, signal is the EWMA of that line, and histogram is line minus signal.
A null or NaN price skips the price averages but can yield a carried line;
that carried line still advances the signal average. Infinity and generated
NaN follow existing EWMA/subtraction arithmetic. The three recurrences advance
together even when the caller consumes only one field. `execution.max_elements`
counts all three output lengths, and the allocation limit includes all three
arrays. There are no Python indicator state handles.

The native eager elementwise functions accept explicit `NativeArray` operands:

- Floating arithmetic: `neg`, `abs`, `sqrt`, `log`, `exp`, `add`, `subtract`,
  `multiply`, `divide` and `pow`.
- Comparisons: `equal`, `not_equal`, `less`, `less_equal`, `greater` and
  `greater_equal`. Results use packed Boolean storage with independent validity.
- Boolean operations: `logical_not`, `logical_and`, `logical_or` and
  `logical_xor`. AND/OR use Kleene rules; XOR propagates nulls.
- Conditional selection: `where_(condition, if_true, if_false)` uses a Boolean
  condition; a null condition yields null.

Use a rank-0 native array for broadcasting, for example
`cfa.add(values, cfa.from_arrow(pa.array([2.0]), shape=()))`. A one-element
vector remains a vector and cannot broadcast to another vector length.
Mixed float32/float64 arithmetic promotes to float64; Boolean operands do not
implicitly convert to floating values. `pow` requires a finite, non-null
scalar exponent with magnitude at most 64. IEEE NaN/Inf arithmetic and
null propagation follow the Rust kernels, without Python operator evaluation.

`matmul(left, right)` requires non-null floating matrices with shapes `(m,k)`
and `(k,n)`, where `n > 0`. Zero rows and zero inner width are supported;
zero inner width produces positive zeros. Mixed floating dtypes promote to
float64; two float32 operands retain float32 multiplication and accumulation.
Each dot uses increasing inner indices, a separate multiply and addition, and
positive-zero initialization. This deterministic baseline does not promise
NumPy/BLAS parity or a particular speedup.

`reshape(array, shape)` owns a new descriptor and shares values/masks; rank must
be at most two and the checked product must preserve `element_count`. It has no
inferred `-1` dimension or order argument. `transpose(array)` swaps matrix axes
into packed row-major order. Scalars/vectors and empty/size-one matrix axes
remain aliases; other transposes copy values and masks. Budgets charge new
buffers and descriptors, while aliases do not charge existing input buffers.

`sum`, `mean`, `min` and `max` also accept `axis=None` and `keepdims=False`.
No axis means all positions; otherwise choose a nonnegative axis below rank.
Keepdims retains reduced dimensions at extent one; scalar full reductions stay
scalar. Each lane uses the original null/NaN/signed-zero and exact sum/mean
contracts. Empty/all-null accepted lanes return null, including sum. Matrices
must be explicitly reshaped to vectors before EWMA or rolling sequence work.
Elementwise operands broadcast only scalars or identical shapes: `(n,)` and
`(n,1)` remain distinct.

Run [`examples/native_matrices.py`](../examples/native_matrices.py) for an Arrow-only
matrix product, axis mean and shape-transform example.

Each calculation accepts `execution=cfa.ArrayExecutionOptions(...)` with
`max_elements=10_000_000` and `max_allocation_bytes=256 * 1024 * 1024` defaults.
These frozen per-call limits cover result elements and native output/scratch
buffers, excluding borrowed inputs and conversion snapshots. Native calculations
release the GIL. This eager API does not require provider registration;
graph execution remains a separate registered capability.

Run [`examples/native_arrays.py`](../examples/native_arrays.py) for an Arrow-only
moving-mean, conditional-selection and reduction example.


### Native sequence provider

The advanced stream identity `native:array_sequence@1` consumes one real native
float32 or float64 ingress and emits one aligned float64 vector Batch per input,
including empty chunks. Its descriptor pins the input dtype. EWMA, rolling mean
and RSI take vectors; ATR takes packed row-major `[n, 3]` high, low, close rows.
Packed ATR has one dtype, no implicit zip or broadcasting, and accepts `[0, 3]`.
Scalars, Boolean values, other shapes, static inputs and second dynamic ingresses
are rejected. Ordered positions advance state in arrival order. Batch metadata sequence numbers
do not reset, reorder or deduplicate the sequence. Table entities and event-time
reordering continue to use table operators.

The strict options object requires `ir_version`, `input_dtype` and `kernel`.
IR1 accepts `{"function":"rolling_mean","window":20}` with optional
`min_periods` and `numerical_profile`, or `{"function":"ewma","span":20}` with
optional `min_periods`. IR2 retains those kernels and adds
`{"function":"rsi","period":14}` and `{"function":"atr","period":14}` with
mandatory positive-u64 period. Missing/NaN resets and finite-domain errors use
the indicator contract above. `execution` holds output/allocation limits and `state`
holds `max_state_bytes`; both are optional and reject unknown keys. These options
use the sequence factory's schema, independently of array_expr IR1/IR2.
Compile it through `PipelineBuilder.external` with an explicitly registered
Runtime and run it through explicit native source/sink bindings. The old kernels
retain checkpoint layout 1 in every IR; RSI and ATR use layout 2. IR3 additionally
accepts `{"function":"macd_components"}` with the five optional MACD parameters
and their 12/26/9/1/1 defaults. Input is one float32/float64 vector; its single
float64 output has shape `[n, 3]` in line/signal/histogram order. The element
limit counts all `3*n` positions. MACD uses layout 3 and retains its full 56-byte
numerical group even when a downstream consumer selects one component. Semantic state
version 1 identifies their codec/recovery contracts independently of layout; these are
separate from capability schema 5 and project/manifest version 3.

The Python API adds no owned state class, `ArrayExpr.stream`, implicit input zip,
or mixed array/table stream result convenience method. Independent sequence
nodes own independent state; one sequence cannot be split into parallel
partitions. Existing NumPy/JAX lifecycle and numerical contracts remain unchanged.

### Native ArrayExpr declarations and collection

`array_input(name, dtype=..., shape=..., backend="native")` declares a dynamic
native array. Shapes have zero, one or two dimensions; `None` denotes an unknown
extent. `parameter(kind="array", backend="native", dtype=..., shape=...)`
declares a static array with concrete dimensions. Neither declaration captures
values. `Program.with_input` accepts an ArrayExpr only when it is an
`array_input` declaration.

Native ArrayExpr supports arithmetic, comparisons and Boolean operators, plus
`reshape(shape)`, `transpose()` and `sum/mean/min/max(axis=None, keepdims=False,
skip_nulls=False)`. Rust infers dtype and shape from descriptors; analysis never
allocates sample arrays or imports NumPy/JAX. Python integer scalar operands must
have magnitude at most `2**53` before encoding; finite Python float literals
retain the full float64 range. Boolean literals remain Boolean.

```python
import pyarrow as pa
import calc_flow as cf
import calc_flow.array as cfa

runtime = cf.Runtime()
cfa.register_native(runtime)
values = cf.array_input("values", dtype="float32", shape=(None, 2))
data = cfa.from_arrow(pa.array([1.0, 3.0, 2.0, 4.0], type=pa.float32()), shape=(2, 2))
result = values.mean(axis=1).collect_array({"values": data}, runtime=runtime)
assert result.dtype == "float64"
assert result.shape == (2,)
assert result.to_arrow().to_pylist() == [2.0, 3.0]
```

`collect_array` returns one NativeArray and requires an explicitly registered
Runtime. Its mapping uses logical declaration names: table inputs accept the
usual Arrow tables/batches; native array inputs and static parameters accept a
NativeArray or a Batch containing an actual native payload. Input mappings are
copied and payloads remain read-only. Every call compiles a fresh plan from a
cached declaration template; payloads are never cached. `array_options` limits
each lowered native IR execution, rather than total memory for an arbitrary
graph, and its normalized limits participate in the template cache key.
The blocking method rejects a running event loop. Async array collection,
mixed table/array Program outputs and `ArrayExpr.stream` are not available.
Advanced stateful vector streams use the separate
[native sequence provider](#native-sequence-provider). `compute`, TableExpr collection and Program collection return tables.

`linalg.from_columns(table, columns=..., backend="native")` establishes a
semantic row origin. Use the same TableExpr node in `from_columns` and
`attach_columns`; reuse a filtered or projected table for both operations.
`linalg.matmul` accepts arbitrary static parameter names;
row-preserving arithmetic and rank-two `mean(axis=1)` can precede
`table.attach_columns`. A native vector attaches as one column. A bridge segment
admits at most one matmul with a direct static RHS. A direct attachment of a
two-matmul chain is unsupported. To use successive table matmuls, explicitly
attach the first result, call `from_columns` on that new TableExpr, multiply and
attach again, then project the desired final columns. The runtime does not
create hidden intermediate columns. Independent array collection may split
multiple matmuls into separate batch segments. SQL may precede or follow the
bridge, and attachment may be followed directly by projection or filtering.
Independent table outputs may share one static declaration; bindings reuse the
same captured Batch across physical ports and the runtime owns static placement.

A table-rooted ArrayExpr can also use `collect_array`. The existing table/SQL
segments feed a native array output, followed by standalone batch segments for
full or axis-zero reductions, transpose and other operations that lose row
correspondence. Rank-two transpose, full/axis-zero reduction and an unproved
reshape erase the row origin permanently; matching row counts or restoring the
old shape cannot make them attachable. An independent `array_input` has no table
origin. NativeArray values in TableExpr/Program stream input mappings bind
static array parameters; they do not enable dynamic array stream inputs.
NumPy/JAX declarations retain their existing admission contracts.

## NumPy and JAX

```python
import numpy as np

from calc_flow import Batch, PipelineBuilder, Runtime, register_numpy

runtime = Runtime()
register_numpy(runtime)
plan = (
    PipelineBuilder("numpy-array")
    .external(
        "center",
        "numpy",
        "expression",
        "1",
        {"expression": "x - mean(x)"},
    )
    .compile_batch(runtime)
)
batch = Batch.from_array(np.array([1.0, 2.0, 4.0, 6.0]), backend="numpy")
centered = plan.execute({"input": batch}).outputs["output"].array
```

Owned arrays are read-only. The bounded expression evaluator allows arithmetic,
reductions, transpose, and reshape; it rejects Python execution features and
backend changes. Operation results, including broadcast binary operations, are
capped at 10,000,000 elements so a single expression cannot allocate an
unbounded output. The input batch itself is exempt, so reductions over larger
inputs remain valid. `register_jax` provides the same explicit provider
boundary.
The full version is [`examples/06_numpy_array.py`](../examples/06_numpy_array.py).

Both helpers also make `expression@1` available to stream compilation, but
only for a conservative row-axis-independent subset. The parsed expression
must reference `x` and may contain no function call or matrix multiplication;
for example, `x * 2 + 1` is eligible, while `sum(x)`, `transpose(x)`,
`reshape(x, ...)`, `x @ x`, and a constant-only expression fail compilation.
The broader bounded expression language remains available to batch plans.

For this stateless stream path, the provider callback receives
`(batch, provider_options)` exactly once for each accepted data micro-batch.
It runs on a blocking worker and receives no public cancellation token or
`ProviderContext`. A callback exception surfaces as a provider error and emits
nothing. Cancellation is checked before dispatch and again after a successful
callback, before output validation and emission. The running
`spawn_blocking` callback itself cannot be preempted, so cancellation waits for
it to return; once the post-callback check observes cancellation, the result is
discarded and no output is emitted.

### Table-array matrix multiplication

`pyarrow.Table` is the table input for table-array matrix multiplication. The
immutable builder method is:

```python
def table_matmul(
    self,
    node_id: str,
    *,
    backend: Literal["numpy", "jax"],
    columns: Sequence[str],
) -> PipelineBuilder: ...
```

It selects `columns` in the supplied order, accepts a rank-two `weights` array
with shape `(len(columns), output_width)`, and returns a same-backend array
batch named `output`. The direct `table_matmul@1` operator remains batch-only.
Streaming static-weight multiplication uses the separately registered
`symbolic_matrix@1` provider through the explicit symbolic compilation shape
described below; `table_matmul@1` itself does not acquire a stream lifecycle.

```python
import numpy as np
import pyarrow as pa

from calc_flow import Batch, PipelineBuilder, Runtime, register_numpy

runtime = Runtime()
register_numpy(runtime)
plan = (
    PipelineBuilder("table-matrix")
    .table_matmul("multiply", backend="numpy", columns=("quantity", "unit_price"))
    .compile_batch(runtime)
)
result = (
    plan.execute(
        {
            "table": Batch.from_pyarrow(
                pa.table({"quantity": [3.0, 1.0], "unit_price": [10.0, 12.0]})
            ),
            "weights": Batch.from_array(
                np.array([[2.0, 0.0], [0.0, 1.0]]), backend="numpy"
            ),
        }
    )
    .outputs["output"]
    .array
)

assert result.tolist() == [[6.0, 10.0], [2.0, 12.0]]
```

After input `Batch` construction, the operator makes no redundant execution
copies. NumPy allocates one dense table matrix and one result. JAX permits one
host staging buffer, one device table buffer, and one device result; it does
not promise a host-free transfer path. The construction of caller input
`Batch` values is outside these execution ceilings. JAX performs no result-to-host round trip during operator execution. The runnable NumPy and
optional JAX paths are in
[`examples/07_array_and_dataframe.py`](../examples/07_array_and_dataframe.py).

## Symbolic declarations and static analysis

`Program.analyze(options=AnalysisOptions(portability=request))` adds immutable
query history, update, finality, ordering and consumption-edge facts. Explicit
`portability=request` on Program compilation and execution checks those facts
before reading inputs or starting a job, including preparation-cache hits.
See [query history and portability](query-history-portability.md) for requests,
stable codes, comparison premises and proof limits.

The [expression API reference](symbolic-api.md) covers typed declarations,
`FeatureSet`, `Program`, static analysis, ordering, and the supported
batch/stream compilation shapes. Use the [expression workflow guide](symbolic-workflows.md)
with examples 09–13 and the
[event-window example](../examples/36_symbolic_event_window.py) to learn these
features. Fixed UTC tumbling/hopping aggregation uses immutable
`WindowAggregate` declarations and the existing native window operator;
see [event-window types and composition](symbolic-api.md#symbolic-event-time-window-aggregation).
Compiler ownership and physical sharing are described in
[symbolic compiler design](symbolic-design.md).

## Projects and persistence

`program.to_project()` exports a validated native graph as `ProjectDocument`.
Use `mode="stream"` for stream export; operational connector, state, and delivery
settings are still explicit. Python logical aliases and live data are not saved.
[Example 14](../examples/14_project_persistence.py) shows export and reloading
through the physical `input`/`output` bindings.

`ProjectDocument` validates a strict `format_version: 3` mapping with the Rust
schema. `project_json_schema()` returns the generated schema;
`validate_project_json(document)` returns canonical JSON. Invalid documents
raise `ProjectValidationError`, a `ValueError` whose `errors()` entries carry
the engine's stable issue codes as `type` and the failing project path as `loc`;
malformed stream Join input reports codes such as `invalid_time_bound` and
`unsupported_join_type` instead of a flattened message. `ProjectDocument` is not
a Pydantic model: catch `ProjectValidationError` rather than Pydantic's
`ValidationError`; Pydantic-specific `include`/`exclude` dump filters are not
supported.

Project v3 selects `runtime.mode` explicitly. Stream projects carry exact
connector and format identities, non-secret options, named secret references,
watermark policy, managed state settings, and per-output best-effort,
at-least-once, or exactly-once delivery requests.
`compile_stream_project(project, runtime=...)` resolves those references
through the runtime's connector registry and secret resolver. Its returned plan
owns deferred connector bindings and project runtime/state settings, so launch
uses `StreamingRunner(plan)` without separate Python connector objects.
Stream projects may additionally declare immutable static side inputs; see
[static input declarations](projects-guide.md#static-input-declarations) for the
syntax and validation rules. `PipelineBuilder.compile_stream()` is the
graph-only path for application-owned `SourceBinding` and `SinkBinding`
values. Project JSON never embeds a connector object, credential value, or
live static payload.

`FileProjectStore` has async `create`, `put`, `get`, `list`, and `delete`
methods and explicit `*_blocking` variants. Safe JSON/YAML import/export helpers
live in `calc_flow.store`. Continuous checkpoint documents are internal to
`ManagedCheckpointRuntime`; the package has no public checkpoint-document
store.

## Streaming runner

Use this explicit interface for durable recovery, transactional sinks, native
Batch metadata, and operational controls. For ordinary async iteration, use
[streaming results](#streaming-results).

`Program.compile_stream()` returns a distinct `StreamExecutionPlan` from
expression declarations. `PipelineBuilder.compile_stream()` is the explicit
graph alternative. Bind the physical `source_binding_ids`, `static_input_ids`,
and `sink_binding_ids`; these explicit runner bindings are physical names.
The convenience `stream` methods translate logical declaration names.
The plan records immutable source/sink binding IDs and optional per-output
`StreamRequirements`; it cannot execute as a batch plan. A
`StreamingRunner` owns that plan, all connector bindings, one
`ManagedCheckpointRuntime`, and optional `StreamRuntimeConfig`:

The explicit graph form is:

```python
plan = PipelineBuilder("orders").expression("total", "total = a + b").compile_stream()
runner = StreamingRunner(
    plan,
    {"input": SourceBinding(source, watermark_policy=DisabledWatermarks())},
    {"output": [SinkBinding.ordinary("archive", sink)]},
    ManagedCheckpointRuntime(".calc-flow-continuous"),
)
job = await runner.start_async()
print(job.status())  # synchronous and safe inside the event loop
outcome = await job.wait_async()
```

The constructor accepts one further keyword-only argument,
`static_inputs: Mapping[str, Batch] | None = None`, supplying the immutable
per-job side values a plan declares. `None` normalizes to an empty mapping;
non-`str` keys or non-`Batch` values raise `TypeError` before native
construction, and the mapping is defensively copied. Project-backed plans
reject externally supplied `sources`, `sinks`, `checkpoints`, and `config`
but exempt `static_inputs`, which is required when the plan declares static
inputs. `plan.static_input_ids` returns the declared names as a sorted tuple
and `plan.source_binding_ids` excludes them. The values are validated,
latched, and digested exactly once inside `start_async`/`start`, before any
source or connector lifecycle runs; restarts that supply different values are
rejected against the recorded digest before sources open. The full per-job
semantics, digest contract, and recovery behavior are in
[static inputs](streaming-guide.md#static-inputs).

A bounded event-time Join is declared on the builder with
`stream_join(name, *, left_schema, right_schema, left_keys, right_keys,
left_event_time, right_event_time, bounds, limits, left_prefix="left",
right_prefix="right")`. Input schemas are `ArrowFieldSpec` sequences and the
`JoinTimeBounds`/`JoinStateLimits` values are required; the
[continuous streaming guide](streaming-guide.md) carries the full
declaration.

Source `open`, `next`, and `close` and every sink lifecycle method must be
declared with `async def`; binding construction rejects invalid method shapes
without invoking the connector. `start_async()` consumes its runner exactly
once. Jobs expose async checkpoint, shutdown, cancel, and wait operations plus
guarded blocking forms for callers outside an event loop. Cancelling a
`wait_async()` observer leaves the job running; explicit cancellation uses
`cancel_async()`.

Blocking `start()` creates a dedicated event-loop thread and keeps it for the
job's async connectors. Later blocking job operations run on that owning loop;
terminal async operations called from another event loop are marshalled back
to it. Blocking or async `shutdown`, `cancel`, and `wait` release connector
roots and stop and join the thread after the native terminal outcome. Dropping
the last job owner schedules cancellation and settlement on the owning loop,
then reclaims the thread. Cancelling an async terminal observer before native
termination leaves cleanup running to convergence, so observer cancellation
does not strand the loop thread. If the native terminal outcome has already
linearized and cancellation arrives during thread cleanup, that outcome wins
and cleanup still completes.

`Cursor` payloads, capability/config mappings, pre-commit values, recovery
values, status, and outcomes cross the boundary as defensive copies.
`Cursor.payload` and `SinkRecovery.pre_commit` are read-only mappings. Reading
nested dicts or lists returns independent, mutable copies, including through
mapping views, `copy()`, and `dict(mapping)`; changing a copy leaves the owned
position or evidence unchanged. Use `json.dumps(dict(mapping))` to serialize it.
Status and outcomes are typed: `job.status()` returns a `JobStatus` mapping that
includes `stream_joins`, a per-node mapping of `StreamJoinStatus` values with
`StreamJoinSideStatus` per side (empty when the graph has no inner Join node).
Each side includes its last accepted `watermark_micros: int | None`, `idle`,
and `ended`. A non-terminal managed restart publishes restored progress at
running startup; terminal-manifest recovery does not restore these observations.
Idle preserves the watermark and is distinct from permanent end-of-input.
The separate `stream_asof_joins` mapping contains `StreamAsofJoinStatus` with
`StreamAsofJoinSideStatus` per side, retaining exact Python integer counters and
watermark microseconds. See [ASOF status](asof-join-guide.md#recovery-status-and-delivery).
Managed checkpoint recovery reopens a live replayable source with a cursor
bound to the exact source-map key. A terminal manifest instead returns
completed without reopening ended sources or duplicating final output. See
[`04_continuous_runtime.py`](../examples/04_continuous_runtime.py),
[`08_streaming_recovery.py`](../examples/08_streaming_recovery.py), and the
[continuous streaming guide](streaming-guide.md).

## Exceptions

Catch the narrowest exported class: `ConfigError`, `CompileError`,
`ExecutionError`, `ProviderError`, `CheckpointError`, or `CancelledError`.
All derive from `CalcFlowError`; provider/cancellation errors are execution
errors. Continuous lifecycle failures use payload-safe
`StreamingRuntimeError`; an indeterminate manifest publication uses its
`CheckpointPublicationUnknownError` subclass. These exceptions expose only
structured category, job/epoch/phase/component identifiers, diagnostic ID,
and deterministic position fields—never connector values, cursor payloads,
paths, callback representations, or raw source chains. Project-document
validation instead raises `ProjectValidationError`, a separate `ValueError`
with structured `errors()` entries rather than a `CalcFlowError` subclass.

## More examples

See [`examples/`](../examples/README.md) and the
[cross-language inventory](examples.md) for executable programs. Run the
examples that need no external service against a prepared installation with
`JAX_PLATFORMS=cpu uv run --no-sync python scripts/run_examples.py`.
The service-backed connector read/write examples require optional native
connector features, prepared source data, and empty sink destinations;
follow the [connector setup](connectors/README.md) before adding
`--include-services` to the runner command.
The [expression workflow guide](symbolic-workflows.md) maps the symbolic examples
to analysis, lowering, checkpoint recovery, and static inputs.
