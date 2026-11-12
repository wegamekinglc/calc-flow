# DataFrame and SQL relations

DataFrame and SQL build the same lazy relation graph in Rust and Python.
Choose batch or stream separately on a named `Program`. Row-local calculations
can share a declaration across both lifecycles. SQL aggregates retain their
existing cumulative snapshot semantics in a stream.

## Python

```python
import pyarrow as pa
import calc_flow as cf

source = cf.table_input("events", schema=pa.schema([("x", pa.int64())]))
scaled = source.with_columns(y=source["x"] + cf.lit(1))
first = cf.sql("SELECT x, y FROM s WHERE y > 1", s=scaled)
selected = first.select(z=first["y"] * cf.lit(2))
result = selected.sql("SELECT z FROM input")
batch = cf.Program("mixed", mode="batch", outputs={"scaled": scaled, "result": result})
stream = batch.for_mode("stream")
assert batch.fingerprint == stream.fingerprint
assert batch.collect_schema()["result"] == pa.schema([("z", pa.int64())])
assert batch.collect({"events": pa.table({"x": [1, None, 3]})})["result"].to_pydict() == {"z": [4, 8]}
```

`collect_async` captures the input mapping when called and returns an awaitable.
A stream captures bindings when called and starts only on entering its owned
context:

```python
async def run(data):
    tables = await batch.collect_async({"events": data})
    async with stream.stream({"events": [data]}) as events:
        async for event in events:
            print(event.name, event.table)
    return tables
```

Exit, cancellation, exceptions and EOF await owned cleanup. Each fresh stream
context owns a fresh job; iterators and explicit source bindings have one active
owner. Convenience streams use temporary state and do not promise durable replay.
Use the explicit runner and managed checkpoint contracts for recovery.

Provide exactly one of `mode="batch"|"stream"` or legacy
`engine="sql"|"streaming"`. There is no default. Both parameters together are an
error even when consistent; explicit `None` is invalid. `.mode` and `.engine`
are read-only. Legacy spellings retain return types and native identities, with
no deprecation warning. Runtime signature reflection now shows omitted sentinels
for both choices; typing overloads express their mutual exclusion.

`for_mode` always creates a new wrapper containing declarations only. It preserves
input inference, explicit inputs and the declaration fingerprint, then checks the
target shape. Prepared caches still include mode and registration identity;
plans are distinct across modes and stream contexts always own a fresh job.
Within batch mode, the existing Runtime cache can reuse a compiled plan; each
execute call still has independent run state. The exact-schema UNION ALL
subset below also supports multiple stream aliases. Other multi-alias SQL
keeps its existing batch-only behavior.

`collect_schema` plans all table outputs without data, including zero-row inputs.
It returns exact column order, Arrow types and nullability. Array outputs reject
this table schema API. The synchronous method rejects an active event loop; use
`await asyncio.to_thread(program.collect_schema)` for planning in async code.
An explicit `mode` on schema/analyze/explain selects a diagnostic view without
changing the bound wrapper.

## Rust

The crate exports `dataframe::{table_input, lit, TableExpr, ColumnExpr, NamedExpr}`,
`sql::sql`, `program::{Program, CompileMode, CompileOptions}`, and matching Arrow
types through `calc_flow::arrow`. Declarations are immutable, cloneable, and
contain no rows or execution state. The independently runnable
[example 32](../crates/calc-flow/examples/32_sql_expression_pipeline.rs) composes
both frontends using public APIs.

Schema, analysis, export and compilation are async. Pass an immutable
`UdfRegistrySnapshot`; compilation creates the existing `BatchExecutionPlan` or
`StreamExecutionPlan`. Batch `execute` and stream runner source/sink maps use the
Program's logical input/output names. A shared input is consumed once, including
when several named outputs reference the same relation. `StreamRequirements`
also uses those logical output names. Low-level `PipelineBuilder` naming remains
unchanged.

Use explicit scalar literals and casts. Arithmetic operands must have the same
numeric type; true division requires floating operands. `&`, `|` and `!` implement
three-valued Boolean expressions. Filters keep only true, preserving duplicate
rows. Projection order is declared order; `with_columns` appends and never
replaces existing fields. Casts use the native table engine. NULL is distinct
from NaN; non-finite Rust floating literals reject at construction.

## Analysis, export and limits

`analyze` returns the existing history and portability facts. Add a
`PortabilityRequest` to analysis or compilation for an explicit comparison;
strict execution requires a supported proof before sources open. Admission and
portability are separate. Opaque SQL with insufficient proof remains unknown.
See [query history and portability](query-history-portability.md).

Schema and diagnostic errors identify logical input/output roles and stable
codes. SQL schema errors identify the SQL node or alias; unavailable source spans
are not invented. SQL creates new row lineage and clears temporal identity;
`ORDER BY` alone does not establish identity for temporal operators.

`to_project` returns the existing strict project-v3 data-only document. It carries
schema placeholders and declarations, with no actual input rows, callbacks,
prepared plans or state. Reloaded projects use the existing physical binding
names. Read these from the reloaded plan's `external_inputs`/`external_outputs`
instead of supplying the original logical aliases. Python batch plans expose these
physical names as immutable `external_inputs` / `external_outputs` tuples; stream
plans retain `source_binding_ids` / `sink_binding_ids`. Export is not a checkpoint;
batch query continuation is outside this contract.

The relational frontends cover row-local expressions, named DataFrame aggregates,
and the existing SQL relation surface. Eligible single-alias aggregates consume
complete replacements with an explicit state budget, as described below. General
joins, wider portable Arrow types, and comprehensive Polars compatibility remain
outside this scope.

## Explicit relation updates

Both DataFrame and SQL declarations use the same immutable Program choice:

```python
total = source.sql("SELECT SUM(x) AS total FROM input")
visible = total.filter(total["total"] >= cf.lit(0))
latest = cf.Program(
    "latest", mode="stream", outputs={"result": visible}
).with_update_semantics("latest_relation")
assert latest.update_semantics == "latest_relation"
```

Rust exports `UpdateSemantics` at the crate root and through `program`:

```rust
use calc_flow::{Program, UpdateSemantics};

fn choose(program: &Program) -> Program {
    program.with_update_semantics(UpdateSemantics::LatestRelation)
}
```

The default is `legacy_append`. With input batches [1] and [2, 3], the first
stream SUM emits snapshots 1 and 6. Under legacy semantics, a second ordinary SUM
appends those events and emits 1 and 7. Explicit latest admits an eligible second
aggregate with `sql_replacement_state_budget` and emits 1 and 6. Queries outside
the replacement allowlist reject before opening a Source, without requiring a
portability request; a missing replacement budget also rejects before startup.

Latest admits complete snapshots through proved deterministic single-input
projections, filters and row-local SQL/CTEs. These edges report
`complete_replacement`; their output remains `complete_snapshot` with
`latest_relation` history. A filter dropping every row still emits an exact-schema
empty relation event. No input event produces no update. All RecordBatches within
one Batch form one event; an oversized complete event fails its edge budget whole.

Known append-only consumers reject with `update_interpretation_mismatch`.
Unproved consumers, opaque providers and missing event-boundary evidence reject
with `update_semantics_unproved` or `update_event_boundary_unproved`.
`CompileError.query_analysis` in Python and Rust's `QueryUpdateSemantics`
contain the immutable report and logical consumer path. Analysis and export
remain available for unsupported declarations. Additional comparison requests
remain separate obligations.

`for_mode`, `with_input` and `output` preserve the choice. Latest changes
declaration, plan, cache and checkpoint identities; switching back to legacy
recomputes the original identity. Batch still consumes one independent finite
input per execution and has no cross-execution replacement state. Batch SUM
chains therefore need no replacement state budget. Queries outside the stream
replacement allowlist may still compile for finite batch execution.

Project-v3 persists `graph.update_semantics="latest_relation"`. Missing or explicit
legacy normalizes to the old omitted-field bytes and identities. The Rust public
`PipelineSpec` has a new field: full struct literals must add
`update_semantics: UpdateSemantics::LegacyAppend` (or `Default::default()`).
Low-level synchronous builders use the same selector and AST rules and require
exact declared table ports to prove event boundaries. They never block on schema
planning. Legacy checkpoints retain their existing layouts; an actual restore
across update choices rejects before Source acquisition. Existing snapshot SQL
initialization, slicing, output-route and delivery restrictions still apply.

## Same-schema union

Rust `a.union_all(&b)?` and Python `a.union_all(b)` append rows through the same
native UnionOperator in batch and stream. SQL
`SELECT x FROM a UNION ALL SELECT x FROM b` lowers each arm through the SQL
planner into that same union. Both frontends support downstream DataFrame/SQL
and shared named outputs without collecting intermediate tables.

| Interface        | Declaration                            | Batch and stream |
|------------------|----------------------------------------|------------------|
| Rust DataFrame   | `a.union_all(&b)?`                     | supported        |
| Rust SQL         | `calc_flow::sql::sql(query, aliases)?` | approved subset  |
| Python DataFrame | `a.union_all(b)`                       | supported        |
| Python SQL       | `cf.sql(query, a=a, b=b)`              | approved subset  |

Inputs must match exactly in column names, order, Arrow dtype and nullability.
Union preserves duplicates, NULL and empty inputs, creates a new row lineage
and clears temporal ordering. Each ingress keeps its FIFO; cross-source order
and stable ties are unspecified. Repeating an input (`a.union_all(a)`, or two
SQL aliases bound to `a`) computes its upstream once and consumes its rows twice.

The shared SQL subset is a pure UNION ALL tree with at least two distinct,
explicit aliases, each referenced exactly once, and no unused alias. Each arm
is a deterministic single-alias row-local SELECT with optional projection,
aliasing, wildcard and filter. Parentheses are supported. CTEs, joins,
subqueries, aggregates, DISTINCT, OVER, ORDER BY, limits, other set operators,
BY NAME, UDFs and volatile/clock functions are outside this subset. Each arm is
planned before set-operation coercion. Wider SQL, including coercion and a
single alias referenced repeatedly, retains its legacy batch behavior and
does not acquire a shared-union proof.

One batch execute consumes each input's complete Batch, including every
RecordBatch. A missing required input is an error; typed empty input is valid.
The same compiled finite plan can execute with new inputs and then empty inputs
without retaining old rows. Stream completion waits for all sources' EOF and
pending FIFO data. Its final relation is the concatenation of **all append
output rows**, including duplicates, rather than the last output Batch. Compare
using `final_relation / accepted_history / all_sources_eof` with unordered
ordering. Declared global ordering remains unproved.

Union consumes append events. Under legacy semantics it also appends complete
snapshot events. Explicit latest rejects any complete snapshot entering union
with `update_interpretation_mismatch` before sources, bindings, jobs or state
are acquired; strict final-relation comparison also rejects that legacy edge.
A finite aggregate result can still feed a batch union. This adds no
replacement-union state, budget parameter, checkpoint layout or batch recovery.
Low-level finite UnionOperator requires table ports with an explicit exact
schema and `required=true`; the existing wider stream forwarding stays available.

Query reports now use schema version 2, analysis revision 5. Old SQL declaration
digests stay unchanged, but admitted multi-alias SQL exports a new native union
topology and fingerprint. Saved legacy SQL projects retain their SQL nodes;
checkpoints with different fingerprints cannot be restored across the change.
See the runnable [Python example](../examples/finite_union.py) and
[Rust core example](../crates/calc-flow/examples/finite_union.rs).

## Fixed UTC event windows

Python `window.tumbling` / `window.hopping` with a nonempty `aggregates` sequence
and Rust `TableExpr::window(WindowSpec)` support both batch and stream modes.
The five reductions are non-NULL `count(column)`, `sum`, `min`, `max`, and `avg`.
SQL has an explicit narrow dialect lowered to the same native window operator:

```sql
SELECT window_start, window_end, symbol, COUNT(quantity) AS n, SUM(quantity) AS volume
FROM CF_TUMBLE('trades', 'ts', 10)
GROUP BY window_start, window_end, symbol
```

`CF_HOP('trades', 'ts', 10, 5)` adds the slide in integer microseconds.
Supply exactly the referenced table alias. SELECT and GROUP BY begin with both
boundaries and the same ordered group columns; every reduction needs `AS output`.
One top-level SELECT is supported. Filtering, derived operands and result
transformations belong in separate row-local DF/SQL stages. COUNT(*), aggregate
expressions/modifiers, CTEs, joins, OVER, DISTINCT, ordering, limits and offset
parameters are rejected by this dialect. Ordinary SQL retains its existing behavior.

Geometry is epoch aligned with half-open bounds and mathematical floor for
negative timestamps. Time accepts naive timestamp[ms] or naive/UTC timestamp[us].
The time column may pass through or be renamed by deterministic single-input
row-local SQL; arithmetic, casts and truncation are rejected. Window paths allow
one window and row-local transformations. Shared window outputs and independent
aggregate/union branches retain their own native owners.

Boundaries are nonnullable timestamp[us, UTC]; group fields keep their types and
nullability. Count is nonnullable uint64. Signed, unsigned and floating sums
promote to int64, uint64 and float64 respectively. Average is nullable float64;
min/max preserve the supported ordered input type. NULL time contributes no
assignment. A valid group whose operands are all NULL emits count 0 and NULL
reductions. Typed empty input emits zero windows.

Finite execution consumes every RecordBatch of the complete input, finalizes all
nonempty windows, and retains no state between executions of the same plan.
Stream data accumulates until a real watermark closes `end <= watermark`, or
real EOF drains the remainder. Empty data does not end or reset the stream.
No synthetic maximum watermark is used. Compare whole results with
`final_relation / accepted_history / all_sources_eof` only when all assignments
of the same complete input were accepted. A closed-window prefix remains
`finality_unproven` without a concrete selected cut. Explicit latest accepts
append inputs and rejects complete snapshots entering windows before resources
are acquired. Window state version remains 1; supported checkpoint layouts are
1 and 2, with no checkpoint format change.

Run [the Python example](../examples/finite_event_window.py) or
[the standalone Rust core example](../crates/calc-flow/examples/finite_event_window.rs).

## Named aggregates

`TableExpr.aggregate` builds a new relation through the same native SQL planner.
Python accepts an ordered mapping and optional `group_by` sequence. Rust accepts
`Vec<NamedAggregate>` and `Vec<String>`. Column `sum`, `count`, `min`, `max`, and
`mean` create aggregate declarations; `count_rows()` counts every row, including
NULL rows. Aggregate declarations are separate from row-local columns.

```python
summary = source.aggregate(
    {"total": source["x"].sum(), "non_null": source["x"].count(),
     "rows": cf.count_rows()},
    group_by=(),
)
```

```rust
# use calc_flow::{TableExpr, Result, count_rows};
fn summary(source: &TableExpr) -> Result<TableExpr> {
    source.aggregate(
        vec![source.column("x")?.sum().alias("total")?,
             count_rows().alias("rows")?], vec![],
    )
}
```

Operands must be direct columns of the input relation. Use `with_columns` first
for computed operands. Names must be unique across keys and results. Keys precede
reducers in output column order; grouped row order is unspecified. Aggregation
starts a new row lineage and clears temporal ordering. Planning derives exact
schema without reading rows.

SUM widens signed integers to Int64, unsigned integers to UInt64, and floats to
Float64. COUNT is non-null Int64. MIN/MAX preserve numeric input dtype; AVG is
Float64. Numeric reducers are nullable and ignore NULL. Empty global input emits
one row with typed NULL reducers and zero counts. Empty grouped input emits zero
rows. All-NULL keys form one group. Reducers support the ten ordinary integer and
float types; keys support Boolean, integers, Utf8 and LargeUtf8. COUNT(column)
also supports the existing portable scalar types.

Streaming append inputs produce cumulative snapshots. With explicit
`latest_relation`, a supported single-alias aggregate can consume another
complete snapshot, including through projection/filter nodes. It requires
`StreamRuntimeConfig(sql_replacement_state_budget=StateBudget(rows, bytes))`.
For source events `[1]`, `[2,3]`, SUM→SUM emits `1,6`; the default
`legacy_append` emits `1,7`. Batch executions each process only their own input.
See the streaming guide for replacement admission, budgeting and recovery.
