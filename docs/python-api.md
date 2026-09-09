# Python API

[Documentation](README.md) / 3.2 Python API

For a guided first calculation, read [batch calculations](batch-guide.md).
This page describes the Python application API and its contracts; the full
expression catalog has its own [reference](symbolic-api.md).

The `calc-flow-python==4.0.0` package exposes Python expressions and Arrow
execution over the internal Rust runtime through PyO3. Python 3.13 or newer
is required.

On this page:

- [Install and develop](#install-and-develop)
- [Compute Arrow data](#compute-arrow-data)
- [Table expressions and names](#table-expressions-and-names)
- [Reusable programs and collection](#reusable-programs-and-collection)
- [SQL and pipeline composition](#sql-and-pipeline-composition)
- [Streaming results](#streaming-results)
- [Bounded backward ASOF Join](#bounded-backward-asof-join)
- [Choosing an integration API](#choosing-an-integration-api)
- [Table batches and builder](#table-batches-and-builder)
- [Multi-input SQL](#multi-input-sql)
- [Trusted Python scalar UDFs](#trusted-python-scalar-udfs)
- [Runtime capabilities](#runtime-capabilities)
- [Execution options and provider context](#execution-options-and-provider-context)
- [Async execution](#async-execution)
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

| Operation                                    | Meaning                                                           |
|----------------------------------------------|-------------------------------------------------------------------|
| `t["price"]`                                 | Select one named column expression                                |
| `+`, `-`, `*`, `/`, unary `-`                | Compose arithmetic, including reflected scalar arithmetic         |
| `==`, `!=`, `<`, `<=`, `>`, `>=`             | Build comparison expressions                                      |
| `&`, `\|`, `~`                               | Compose boolean expressions; parenthesize comparisons             |
| `t.with_columns(mapping, **named)`           | Append named expressions; `mapping` is optional                   |
| `t.select(*columns, **named)`                | Keep literal column names, then append named expressions in order |
| `t.filter(predicate)`                        | Keep rows matching a boolean `ColumnExpr` over that table         |
| `expression.pipe(function, *args, **kwargs)` | Apply a synchronous declaration function once                     |
| `t.sql(query)`                               | Add SQL using this table as the local alias `input`               |
| `expression.identical(other)`                | Compare structural identity as a Python boolean                   |

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

`cf.Program(name, /, *, inputs=None, outputs=())` accepts outputs as a mapping
in insertion order or a sequence of `(name, TableExpr | ArrayExpr)` pairs. Omitting
`inputs` discovers reachable table inputs and parameters in deterministic order.
Explicit input sequences are respected, including `inputs=()`; missing referenced
inputs become analysis errors. Conflicting roots with the same name fail.
Declarations and output mappings are copied and remain immutable.

| Method                                                                                                   | Result and input contract                                                        |
|----------------------------------------------------------------------------------------------------------|----------------------------------------------------------------------------------|
| `table_expr.collect(inputs, /, *, runtime=None, options=None)`                                           | One Arrow table; one table root accepts data directly, otherwise use a mapping   |
| `program.collect(inputs, /, *, runtime=None, options=None)`                                              | `dict[str, pyarrow.Table]` in logical output order; always supply a mapping      |
| `table_expr.collect_async(...)` / `program.collect_async(...)`                                           | Awaitable forms of the same contracts                                            |
| `table_expr.stream(inputs, /, *, runtime=None, config=None, watermarks=None)`                            | Owned async iterator of Arrow tables                                             |
| `program.stream(inputs, /, *, runtime=None, config=None, watermarks=None)`                               | Owned async iterator of named `StreamOutput` events; input mapping required      |
| `program.analyze(runtime=None, /, *, mode="batch")`                                                      | Immutable analysis result                                                        |
| `program.explain(runtime=None, /, *, mode="batch")`                                                      | Deterministic explanation text                                                   |
| `program.compile_batch(runtime=None, /)`                                                                 | Explicit batch execution plan                                                    |
| `program.compile_stream(runtime=None, /, *, allowed_lateness_micros=0, late_policy="error")`             | Explicit stream plan for a runner                                                |
| `program.to_project(runtime=None, /, *, mode="batch", allowed_lateness_micros=0, late_policy="error")`   | Validated, data-only project-v3 document                                         |

Collection mappings use declared input names, and the returned mapping uses
logical output names. Missing, extra, or wrong-kind inputs fail with named paths.
Supported static array parameters require `Batch.from_array` and explicit provider
registration on the selected runtime. Collection returns table outputs;
standalone array-output declarations are not executable. See the
[capability matrix](symbolic-api.md#symbolic-declarations-and-static-analysis).

Omitting `runtime` selects a fresh default `Runtime`; pass one explicitly for
registered providers/UDFs. Every convenience `compute` or `collect` call creates
a fresh batch plan, including repeated/concurrent calls using the same runtime.
Rolling state does not carry between those calls, and explicitly compiled cached
plans are neither reused nor reset. Use explicit plans for snapshot/restore/reset,
state reuse, timings, and DataFusion metrics.

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

Batch SQL accepts multiple aliases. Stream SQL accepts exactly one and applies
SQL independently to each native input batch. SQL aggregation, sorting, limits,
and window functions have batch-local semantics, including inside a stream.
Use native stateful declarations for cross-batch calculations.

## Streaming results

`TableExpr.stream(inputs, /, *, runtime=None, config=None, watermarks=None)` returns
`StreamResults[pyarrow.Table]`. `Program.stream(inputs, /, *, runtime=None,
config=None, watermarks=None)` returns `StreamResults[StreamOutput]`. Both require
`async with` and `async for`; there is no blocking convenience stream. Run the
[stateful pipeline](../examples/20_streaming_pipeline.py) and
[named-output example](../examples/21_streaming_outputs.py), or follow the
[streaming guide](streaming-guide.md#first-python-continuous-job).

A dynamic input accepts an `AsyncIterable` of Arrow `Table`, `RecordBatch`, or
table `Batch` values, or a `SourceBinding`. With `SourceProvidedWatermarks`, the
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

With `watermarks=None`, an iterable whose input declares `event_time` must have
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
Run [example 22](../examples/22_stream_asof_join.py) and read the
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

Auto mode requires trusted `calc_flow.datafusion.active_entities` batch
metadata and otherwise uses p1 without scanning the table. Fixed p1 remains the
default. See [SQL and DataFusion performance controls](sql-datafusion-performance.md)
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

assert snapshot.schema_version == 3
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

Capability schema version 3 makes every operator and provider entry
lifecycle-aware. `OperatorCapability` and `ProviderCapability` report
`modes`, `finality`, `stateful`, `microbatch_invariant`, `requires_watermark`,
`checkpoint_support`, `state_version`, `deterministic`, and `replay_safe`;
operators additionally report `state_layouts`, the advertised checkpoint layout
inventory. Inspect the native operator contract when evaluating checkpoint
compatibility; the rolling catalog limitation is described below. Providers additionally report
`supports_static_inputs`, `partition_contract`,
and optional `array_rules`. The vocabularies are closed: execution modes are
`batch` and `stream`; output finality is `per_row_final`,
`group_final_append_only`, or `unproven`; checkpoint support is `stateless`,
`checkpointed_stateful`, or `unproven`; and a provider partition contract is
`none` or `row_axis_independent`. `state_version` is a positive integer
exactly when `checkpoint_support` is `checkpointed_stateful` and `None`
otherwise, a `stateless` capability must set `stateful=False`, and
`state_layouts` is a strictly ascending tuple of positive integers that must
be empty unless `checkpointed_stateful` and must contain `state_version`.
Construction validates strictly and fails closed: closed-vocabulary and
cross-field violations raise `ValueError`, while non-strict data (a `list`
where a tuple is declared, a non-`bool` boolean field) raises `TypeError`.

`CapabilityRule` identities are closed and versioned. The accepted identities
are `array_api_safe_dtype@1`, `elementwise_broadcast@1`,
`feature_axis_reduction@1`, and `table_matmul_static_rhs@1`; any other
name/version pair fails construction. `ProviderArrayRules` pairs the exact
`supported_dtypes` tuple with a `safe_dtype_rule` and `shape_rules`, and
stores both tuples sorted by identity.

The `operators` tuple contains exactly `cross_section@1`, `expression@1`,
`rolling@1`, `sql@1`, `stream_asof_join@1`, `stream_join@1`, and `window@1`,
with truths anchored in
the engine implementation:

| Operator             | Modes           | Finality                  | Checkpoint support      | State version   | State layouts   |
|----------------------|-----------------|---------------------------|-------------------------|-----------------|-----------------|
| `cross_section@1`    | batch, stream   | group_final_append_only   | checkpointed_stateful   | 1               | 1               |
| `expression@1`       | batch, stream   | per_row_final             | stateless               | —               | —               |
| `rolling@1`          | batch, stream   | per_row_final             | checkpointed_stateful   | 1               | 1, 2            |
| `sql@1`              | batch, stream   | unproven                  | stateless               | —               | —               |
| `stream_asof_join@1` | stream          | group_final_append_only   | checkpointed_stateful   | 1               | 1               |
| `stream_join@1`      | stream          | unproven                  | checkpointed_stateful   | 1               | 1               |
| `window@1`           | stream          | group_final_append_only   | checkpointed_stateful   | 1               | 1               |

The Python capability catalog currently reports only layouts `1` and `2`
for `rolling@1`, while the native operator writes columnar checkpoint layout
`3`. The catalog therefore omits the current writer layout; do not use this
inventory alone to decide rolling checkpoint compatibility. See
[native rolling state](symbolic-design.md#native-rolling-state) for the
implemented encoding and restore rules.

`cross_section@1`, `rolling@1`, `stream_asof_join@1`, `stream_join@1`, and
`window@1` are the stateful operators and the only ones that require a watermark;
`cross_section@1`, `expression@1`, `rolling@1`, `stream_asof_join@1`, and
`window@1` are micro-batch invariant. All seven
report `deterministic=True` and `replay_safe=True`. For `sql@1` those two claims
hold from the engine viewpoint: exactly-once stream
plans reject nodes that select volatile registered UDFs, and stream
compilation rejects read-only queries that call volatile built-in SQL
functions such as `random()` or wall-clock built-ins such as `now()`,
`current_date()`, and `current_time()` (aliases included).

The `register_provider` API accepts no lifecycle
metadata, so a registered provider's entry is always batch-only with
conservative values: `modes=("batch",)`, `finality="unproven"`,
`stateful=False`, `microbatch_invariant=False`, `requires_watermark=False`,
`checkpoint_support="stateless"`, `state_version=None`,
`deterministic=False`, `replay_safe=False`, `supports_static_inputs=False`,
`partition_contract="none"`, and `array_rules=None`. A registration record
carrying forged lifecycle keys is ignored rather than upgraded, and omission
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
The public `Runtime.register_provider` signature has no lifecycle arguments,
so an arbitrary Python callback remains batch-only rather than acquiring this
proof from caller-supplied metadata.

Compiled-in connector registrations surface on the snapshot's `connectors`
tuple as `ConnectorCapability` entries. Each entry pairs its
`(provider, name, version)` identity and `source`/`sink`/`both` kind with a
`ConnectorCapabilities` value (`delivery`, `replay`, `watermark`,
`transaction`, and the `snapshot`/`polling`/`cdc`/`lookup` flags), the
declared formats, and an options schema.

The session ID is stable for one runtime. A successful registry entry advances
the revision exactly once; rejected duplicates do not. NumPy/JAX helpers add
`expression@1`, mapped `table_matmul@1`, and mapped `symbolic_matrix@1` as
separate entries, so one helper normally advances by three and can expose a
real partial success if a later entry already exists. Previously returned
snapshots remain isolated from later revisions. Snapshots are immutable and
defensively copied: mutating a caller-owned sequence after registration cannot
change a returned snapshot. Operators sort by `(kind, version)`; providers and
UDFs sort by `(provider, name, version)`.

## Execution options and provider context

Use the frozen native `ExecutionOptions` value to attach run-scoped settings
and an absolute UTC deadline:

```python
from datetime import UTC, datetime, timedelta

from calc_flow import ExecutionOptions

options = ExecutionOptions(
    settings={"request": {"tenant": "demo", "attempt": 1}},
    deadline=datetime.now(UTC) + timedelta(seconds=30),
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
normalizes every accepted offset to `datetime.UTC` and preserves
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
registered with `_register_mapping_provider` receives its named input mapping
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
projects, checkpoints, or Studio API payloads.

`ExecutionOptions.deadline` is an absolute cooperative engine deadline.
Studio's `RunOptions.timeout_seconds` is instead a process-level preview limit; it
does not populate execution settings or a deadline in the worker.

## Async execution

```python
import asyncio
from datetime import UTC, datetime, timedelta

import pyarrow as pa

import calc_flow as cf


async def run() -> None:
    options = cf.ExecutionOptions(
        settings={"request": {"source": "async-example"}},
        deadline=datetime.now(UTC) + timedelta(seconds=30),
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

The [expression API reference](symbolic-api.md) covers typed declarations,
`FeatureSet`, `Program`, static analysis, ordering, and the supported
batch/stream compilation shapes. Use the [expression workflow guide](symbolic-workflows.md)
with examples 09–13 and the
[event-window example](../examples/symbolic_event_window.py) to learn these
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
raise a pydantic `ValidationError` whose entries carry the engine's stable
issue codes as `type` and the failing project path as `loc`; malformed stream
Join input reports codes such as `invalid_time_bound` and
`unsupported_join_type` instead of a flattened message.

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
values, status, and outcomes cross the boundary as defensive copies. Status
and outcomes are typed: `job.status()` returns a `JobStatus` mapping that
includes `stream_joins`, a per-node mapping of `StreamJoinStatus` values with
`StreamJoinSideStatus` per side (empty when the graph has no inner Join node).
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
paths, callback representations, or raw source chains.

## More examples

See [`examples/`](../examples/README.md) and the
[cross-language inventory](examples.md) for executable programs. Run the
examples that need no external service against a prepared installation with
`JAX_PLATFORMS=cpu uv run --no-sync python scripts/run_examples.py`.
The connector `*_source.py` examples numbered 16–21 require optional native
connector features and prepared services;
follow the [connector setup](connectors/README.md) before adding
`--include-services` to the runner command.
The [expression workflow guide](symbolic-workflows.md) maps the symbolic examples
to analysis, lowering, checkpoint recovery, static inputs, and Studio.
