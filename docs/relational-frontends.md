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
execute call still has independent run state. Multi-alias
SQL cannot be converted to ordinary append streaming; stream-only relations
cannot be converted to batch.

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

This slice covers row-local DataFrame expressions and the existing SQL relation
surface. It does not add DataFrame aggregation, replacement consumption, general
joins, wider portable Arrow types, or comprehensive Polars compatibility.
