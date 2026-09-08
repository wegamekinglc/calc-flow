# Python expression API

Status: implementation-ready design, 2026-09-08. This note records the user-requested
Python-first refactor against `main` at
`c9f53abb906d2be7b55427e75c7a2710dc0e157b`. It supersedes older guidance that presents
Rust and Python as equivalent end-user entry points, and the earlier rule that
`calc_flow.symbolic` must be the only public declaration import path. It does not
supersede runtime, project-v3, checkpoint, or expression-identity semantics.

## Objective and reviewed surfaces

Calc Flow's product API is Python. Rust is the internal calculation runtime and
extension implementation. A Python user should compose calculations with ordinary
Python operators, receive Arrow results without handling graph ports, and progress
to reusable declarations and continuous execution through the same expression model.

The review covered package exports, `_native.pyi`, `pipeline.py`, symbolic expression,
program, analysis and lowering surfaces, Rust crate exports and native external-port
naming, the project model and schema, Studio OpenAPI and project inspection boundary,
package metadata, documentation navigation, and batch, financial, relational, array,
streaming and persistence examples. Existing symbolic tests were read for the
identity and compilation contracts; no engine execution or benchmark was run.

Findings:

1. `symbolic/expr.py` already has immutable table/column/array expressions and
   arithmetic, reflected arithmetic, comparisons and boolean operator overloads.
   `symbolic/lower/` already compiles these into project-v3 and the Rust runtime.
   The task is to make that implementation the usable default, not add another IR.
2. README, introduction, getting-started and Python batch guides instead lead with
   formula strings, explicit node names, `connect`, `Batch`, plan compilation,
   physical input/output names and unwrapping. The Rust quickstart has equal billing.
3. The existing object path requires a separate import namespace, duplicate Arrow
   schema declarations using `Field`, tuple-pair `FeatureSet` and `Program` arguments,
   an explicit `Runtime`, and manual execution-envelope/port handling.
4. Table composition is split between `table.project/filter` and
   `TableExpr.with_columns`. A normal pipeline should read left to right on the table.
5. Public program names do not generally equal native port names: a program with
   input `quotes` and output `signals` can execute using `input` and `output`;
   fan-out can expose `quotes.input`, `signals.output`, etc. Convenience execution
   must preserve declaration names independently of these physical names.
6. Batch symbolic compilation caches plan instances in `Runtime`. Existing plans
   expose snapshot/restore/reset and can own state. Independent convenience calls
   must not accidentally share state through that cache.
7. The symbolic declaration catalog is wider than executable lowering. Portable
   Arrow types, explicit ordering, unsupported standalone array outputs, stream-only
   joins/windows and registered matrix providers remain real boundaries.
8. Some module docstrings still say Program compilation is absent, despite its
   implemented compile methods. Product and API documentation need one coordinated
   update, including these stale inline contracts.

## Reference models

Polars demonstrates deferred composable expressions, overloaded arithmetic and
named expressions in a selection context. Calc Flow adopts that expression/context
separation and fluent table operations, while retaining its existing explicit
lineage and event-ordering model. This is an API-design inference from the
[Polars expression/context guide](https://docs.pola.rs/user-guide/concepts/expressions-and-contexts/)
and [expression reference](https://docs.pola.rs/api/python/stable/reference/expressions/index.html),
checked on 2026-09-08. It is not a promise of Polars feature or type-coercion parity.

The most relevant `finance-python` match is
[alpha-miner/Finance-Python](https://github.com/alpha-miner/Finance-Python), the project
to which [ChinaQuants/Finance-Python](https://github.com/ChinaQuants/Finance-Python)
redirects. This identification is an assumption based on its name and composable
indicator API, rather than the unrelated FinancePy pricing package. Its
[SecurityValueHolders source](https://github.com/alpha-miner/Finance-Python/blob/master/PyFin/Analysis/SecurityValueHolders.pyx)
defines arithmetic/reflected operators and a data transformation entry point.
Calc Flow adopts reusable calculation composition and short execution, retaining
its own immutable declarations and Rust-owned execution/state rather than importing
that library's mutable holder lifecycle. No dependency on either reference is added.

## One declaration model and the preferred imports

Use `import calc_flow as cf` throughout user documentation. Re-export the existing
objects themselves; do not wrap them in new subclasses or fork their implementations.

The exact additions to the root exports are:

```python
compute, compute_async, lit
Expr, ColumnExpr, TableExpr, ArrayExpr, Parameter, Field, FeatureSet, Program
table_input, parameter, row, ts, cs, table, linalg, window
rows, duration, exact_time, event_time_bucket
RowFrame, DurationFrame, CrossSectionGroup, EventTimeBucket, WindowAggregate
AnalysisIssue, AnalysisResult
```

`lit` is the only new primitive constructor: it exposes the already-existing scalar
literal IR node. `table_input`, `Program`, `row`, `ts`, and the other objects retain
identity across `calc_flow` and `calc_flow.symbolic` imports. Implementation may stay
under `symbolic/`; that directory name is an implementation detail.

Preserve all currently exported root names and `__all__` membership in this
compatibility release, then add the names above. Existing explicit and star imports
must continue working. Do not add runtime import warnings: `Runtime`, streaming
types, project types and `PipelineBuilder` remain legitimate advanced APIs.
Reducing the common concepts and required calls is the objective, not silently
breaking existing imports to reduce the export count.

The documented tiers are:

- Daily calculations: `compute`, table indexing/operators, `select`, `with_columns`,
  `filter`, and `row`/`ts`/`cs` functions as needed.
- Reusable declarations: `table_input`, `Program`, `collect`, analysis, project
  export and compilation, with typed schemas and declared ordering.
- Application integration: `Runtime`, `Batch`, execution plans/options, providers,
  UDF registration, streaming runners, project stores and connectors.
- Explicit graph/SQL integration: existing `PipelineBuilder` methods and raw
  project documents. Formula strings remain supported here and in serialized
  nodes; they cease to be the standard Python expression API.

`calc_flow.symbolic` stays a supported compatibility import spelling. Its separate
first-class tutorial narrative is retired. No `DataFrame` class, `from_arrow` object
that captures data inside expressions, global unbound `col` IR, formula parser,
Python evaluator, or restored v1 API is introduced.

## Preferred one-shot calculation

```python
import pyarrow as pa
import calc_flow as cf

data = pa.table({"a": [1, 3], "b": [2, 4]})
result = cf.compute(data, lambda t: t.select(total=t["a"] + t["b"]))
assert result.to_pydict() == {"total": [3, 7]}
```

Proposed signatures, with `TableData = pa.Table | pa.RecordBatch | Batch`:

```python
def compute(
    data: TableData,
    build: Callable[[TableExpr], TableExpr],
    /,
    *,
    entity_by: Sequence[str] = (),
    event_time: str | None = None,
    sequence_by: Sequence[str] = (),
    runtime: Runtime | None = None,
    options: ExecutionOptions | None = None,
) -> pa.Table: ...

def compute_async(
    data: TableData,
    build: Callable[[TableExpr], TableExpr],
    /,
    *,
    entity_by: Sequence[str] = (),
    event_time: str | None = None,
    sequence_by: Sequence[str] = (),
    runtime: Runtime | None = None,
    options: ExecutionOptions | None = None,
) -> Awaitable[pa.Table]: ...
```

The adapter normalizes one table, infers its exact supported schema, declares the
input as `input`, invokes `build` exactly once with that `TableExpr`, then evaluates
the returned declaration through the ordinary Program/lowering/native path. The
internal one-output Program is named `compute` with output `output`; those names
need not be typed by the caller. The function receives a declaration, never rows or
a mutable data wrapper. It can be a lambda or an ordinary reusable Python function.
It is invoked in-process as trusted application code and is never serialized or
called per row/batch by the runtime. A coroutine builder or any non-TableExpr result
is rejected with a clear builder-result error.

Data does not become part of an expression node, digest, compile cache key or
project. A supplied `Batch` must be table-kind and keeps its metadata; Arrow input
uses the existing immutable `Batch.from_pyarrow` boundary. Accept only these
explicit input types; implicit pandas, Polars, dict and array conversions would add
unclear data-copy and backend behavior. A caller can explicitly convert those to
Arrow. Unsupported Arrow field types produce an input-field error, not a lossy cast.

Ordering arguments are forwarded to the inferred `table_input`. No entity, timestamp,
sorting or sequence semantics are invented from row position. Stateful expressions
without required ordering continue to fail analysis. Rolling `event_time` must be
declared as a non-null UTC microsecond timestamp. Schema inference preserves the
input Arrow field's nullability; an observed absence of null values does not make
a nullable field non-nullable. Callers provide an appropriately declared Arrow
schema when constructing temporal input data.

## Table operations and schema construction

```python
def table_input(
    name: str,
    /,
    *,
    schema: pa.Schema | Sequence[Field],
    entity_by: Sequence[str] = (),
    event_time: str | None = None,
    sequence_by: Sequence[str] = (),
) -> TableExpr: ...

def lit(value: None | bool | int | float | str, /) -> ColumnExpr: ...

class TableExpr:
    def __getitem__(self, field: str, /) -> ColumnExpr: ...
    def with_columns(
        self,
        features: FeatureSet | Mapping[str, ColumnExpr] | None = None,
        /,
        **named: ColumnExpr,
    ) -> TableExpr: ...
    def select(self, *columns: str, **named: ColumnExpr) -> TableExpr: ...
    def filter(self, predicate: ColumnExpr, /) -> TableExpr: ...
```

Normalize Arrow schema fields by semantic dtype, order and nullability into the same
`Field` nodes used today. For example Arrow `double` maps to `float64`, `float` to
`float32`, and UTC microsecond timestamps to `timestamp[us, UTC]`; use explicit Arrow
type predicates/constructors, not arbitrary string replacement. Preserve the
existing supported type set and exact ordering constraints. Metadata is not
invented as a new symbolic schema feature; the underlying input remains unchanged.
Keep `Sequence[Field]` working, and do not add a third schema-map notation.

`with_columns` normalizes optional FeatureSet/mapping entries followed by keyword
entries in insertion order and creates the same existing `with_columns` node.
Reject duplicate derived names, including duplicates between mapping and keywords.
The existing append-only collision rule is preserved: derived names cannot replace
existing schema fields. Document this difference from Polars clearly; this task
does not silently redefine an existing IR primitive. Within a call, expressions
refer to the incoming table; dependent calculations use a second call or reuse
the earlier Python expression variable.

`filter` delegates to the existing `table.filter`. `select` combines the existing
`with_columns` and `table.project`: optional derived columns are appended, then the
result is projected in positional-column order followed by keyword order. Strings
are literal field names, never formula source. Field names must satisfy the existing
native project-v3 portable SQL identifier contract; names containing spaces or
quotes are rejected clearly, without introducing a new identifier or schema
contract. Require at least one selected/derived column, reject repeated output names,
and keep the same append-only collision rule for named derived columns. A simple
rename is `t.select(new_name=t["old_name"])`; selecting the existing field unchanged
is `t.select("old_name")`. No separate alias wrapper or naming IR is needed.

`lit` delegates to current strict scalar validation and the existing `literal`
node. Native scalar operands remain implicit literals in arithmetic. Typed nulls
continue to require an explicit cast where inference cannot determine a type.
The supported operators remain `+`, `-`, `*`, `/`, unary `-`, comparisons, `&`, `|`
and `~`, including existing reflected operators. `**`, `//`, `%`, Python `and/or`,
chained comparisons and implicit truth testing are not newly promised. Keep
`identical()` for structural comparison. Row functions and rolling indicators use
the existing `cf.row.*`, `cf.ts.*`, `cf.cs.*` vocabulary, avoiding a second set of
method aliases for every operation.

## Reusable programs and collection

```python
class Program:
    def __init__(
        self,
        name: str,
        /,
        *,
        inputs: Sequence[TableExpr | Parameter[object]] | None = None,
        outputs: Mapping[str, TableExpr | ArrayExpr]
        | Sequence[tuple[str, TableExpr | ArrayExpr]] = (),
    ) -> None: ...

    def collect(
        self,
        inputs: Mapping[str, TableData],
        /,
        *,
        runtime: Runtime | None = None,
        options: ExecutionOptions | None = None,
    ) -> dict[str, pa.Table]: ...

    def collect_async(
        self,
        inputs: Mapping[str, TableData],
        /,
        *,
        runtime: Runtime | None = None,
        options: ExecutionOptions | None = None,
    ) -> Awaitable[dict[str, pa.Table]]: ...

class TableExpr:
    def collect(
        self,
        inputs: TableData | Mapping[str, TableData],
        /,
        *,
        runtime: Runtime | None = None,
        options: ExecutionOptions | None = None,
    ) -> pa.Table: ...
    # collect_async has the same arguments and returns Awaitable[pa.Table].
```

Program output mappings normalize to the existing ordered tuple representation.
Only omitted/`None` inputs trigger root discovery from output declarations; explicit
`inputs=()` retains today's missing-input analysis behavior. Discover reachable
`table_input` and `parameter` nodes in deterministic output/argument traversal
order. Deduplicate equal declarations by full structural identity; two different
declarations with the same name fail, never silently choose one. Explicit input
ordering and existing golden fingerprints remain unchanged.

Program collection keys are the declared input/parameter names, and returned dict
keys are the declared output names in declaration order. All collect outputs must
be tables. Existing unsupported standalone-array output errors remain in effect;
supported table-attached matrices remain supported. Array parameters must be passed
as the existing provider-backed `Batch`, and providers must be explicitly registered
on the supplied Runtime. Do not infer a NumPy/JAX provider from arbitrary values.

Table collection constructs `Program("collect", outputs={"output": self})` and
delegates to the same collection implementation, returning its one Arrow table.
A bare Arrow/Batch input is accepted only when exactly one dynamic table root and
no static parameters are required; otherwise require a mapping by declared names.
Do not guess a root by schema. It is the useful shorthand when declaration and data
arrival are separate; `compute` remains the tutorial entry when data already exists.

Make existing Program `analyze`, `explain`, `compile_batch`, and `compile_stream`
accept `runtime: Runtime | None = None`; choose a fresh Runtime when omitted and
preserve positional Runtime calls. `analyze`/`explain` default `mode="batch"`.
`compile_stream` continues to require explicit stream compilation and retains its
existing lateness arguments. No global runtime or implicit provider registration.

### Execution ownership and binding rules

All convenience calls lower through the existing compiler and execute through the
existing `BatchExecutionPlan.execute/execute_async` boundary. Use one implementation
for validation, Arrow/Batch normalization, logical-name binding and output conversion.
Keep expressions pure and place orchestration helpers in a small Python module,
for example `python/calc_flow/compute.py`, with local imports as needed for cycles.

Each `compute`/`collect` call owns a fresh native batch plan. A supplied Runtime
selects registrations/capabilities and may reuse immutable analysis metadata; it
must not make separate collect calls reuse the cached stateful plan instance from
`Program.compile_batch`. Lower the document once and compile a fresh execution plan
with that runtime. Never reset, restore or mutate an existing cached/caller plan to
obtain this behavior. Existing explicit compile-and-execute state ownership remains
unchanged for users who need plans, metrics, snapshot/restore or batch sequences.

Keep a private association between logical input/output declarations and lowered
external endpoints during lowering, then apply the existing native naming rule:
a unique port name is bare; duplicate port names are `node_id.port`. Resolve binding
against this association and actual lowered topology. Do not zip dicts, split
arbitrary user names on punctuation, infer lineage from equal schemas, or assume
that node-id suffixes survive CSE. The serialized document and public native plan
port names stay unchanged. Existing `lower_program_document` retains its dict result;
binding metadata can remain an internal companion result/helper.

Validate missing/extra logical names and table-vs-array kind before execution.
Reject unconsumed explicitly declared inputs using the existing analysis/lowering
contract. Preserve supplied `Batch` metadata and never mutate caller mappings,
Arrow data or arrays. Output conversion returns new Python dicts.

Blocking `compute`/`collect` reject an active event loop before invoking a builder
or starting work and name the async alternative. Async forms use the existing
cancellation-aware native bridge; they must not call blocking execute, `asyncio.run`
or swallow cancellation. Copy caller-owned mappings and snapshot inputs at the
adapter boundary before asynchronous execution. Forward the exact provided
`ExecutionOptions` settings/deadline through the existing validated boundary.

## Before and after

Today, the default tutorial asks for graph details and formula source:

```python
plan = (
    PipelineBuilder("orders")
    .expression("gross", "gross = quantity * unit_price")
    .expression("large", "", select=("order_id", "gross"), filter="gross >= 20")
    .connect("gross", "large")
    .compile_batch()
)
rows = plan.execute({"input": Batch.from_pyarrow(data)}).outputs["output"].to_pyarrow()
```

The default replacement composes expressions without a manual schema, input name,
node name, port or runtime:

```python
import calc_flow as cf
import pyarrow as pa

data = pa.table({
    "order_id": ["A-100", "A-101", "A-102"],
    "quantity": [3, 1, 4],
    "unit_price": [10, 12, 10],
})

def large_orders(t: cf.TableExpr) -> cf.TableExpr:
    gross = t["quantity"] * t["unit_price"]
    enriched = t.with_columns(gross=gross, fee=cf.row.cast(gross, "float64") / 10.0)
    return enriched.filter(enriched["gross"] >= 20).select("order_id", "gross")

rows = cf.compute(data, large_orders)
assert rows.to_pylist() == [
    {"order_id": "A-100", "gross": 30},
    {"order_id": "A-102", "gross": 40},
]
```

`gross` is an integer expression here. Division uses an explicit floating cast and
floating divisor to satisfy the existing strict expression-type contract.

A reusable program exposes meaningful named outputs without physical port names:

```python
t = cf.table_input("orders", schema=data.schema)
gross = t["quantity"] * t["unit_price"]
program = cf.Program("orders", outputs={
    "totals": t.select("order_id", gross=gross),
    "quantities": t.select("order_id", "quantity"),
})
tables = program.collect({"orders": data})
assert list(tables) == ["totals", "quantities"]
assert tables["totals"]["gross"].to_pylist() == [30, 12, 40]
```

Financial formulas remain ordinary reusable expression functions:

```python
quote_schema = pa.schema([
    pa.field("symbol", pa.string(), nullable=False),
    pa.field("ts", pa.timestamp("us", tz="UTC"), nullable=False),
    pa.field("price", pa.float64(), nullable=False),
])
quotes = pa.table({
    "symbol": ["AAA", "AAA", "AAA"],
    "ts": [1_000_000, 2_000_000, 3_000_000],
    "price": [100.0, 102.0, 101.0],
}, schema=quote_schema)

def signals(q: cf.TableExpr) -> cf.TableExpr:
    price = q["price"]
    momentum = price / cf.ts.lag(price) - 1.0
    mean = cf.ts.mean(price, window=cf.rows(3))
    return q.select("symbol", "ts", momentum=momentum, mean_price=mean)

result = cf.compute(
    quotes,
    signals,
    entity_by=("symbol",),
    event_time="ts",
    sequence_by=("ts",),
)
```

The financial example explicitly declares floating `price` and non-null
`timestamp[us, UTC]` event time. A timestamp field inferred by ordinary `pa.table`
construction is nullable by default and would not satisfy the rolling contract,
even if every timestamp value is present. `compute` preserves the supplied schema;
existing type, null/warm-up and ordering semantics apply unchanged.

## Projects, streaming, providers and Rust

Add a public data-only export instead of asking users to import a lowering module:

```python
class Program:
    def to_project(
        self,
        runtime: Runtime | None = None,
        /,
        *,
        mode: Literal["batch", "stream"] = "batch",
        allowed_lateness_micros: int = 0,
        late_policy: Literal["error", "drop"] = "error",
    ) -> ProjectDocument: ...
```

This delegates to the same lowering and strict project validation used today.
`program.to_project().model_dump()` can be passed to the existing store/serialization
functions. It exports the native graph and data-only input placeholders, not data,
the builder function, expression objects or a resumable job. It does not promise
that reloading native project-v3 reconstructs Python expressions or logical program
aliases. A loaded project is compiled through the current Runtime/project APIs and
uses the physical graph binding names in the document. No schema field is added
for aliases or callables.

Existing project-v3, JSON/YAML, UDF reference and provider reference encodings remain
byte-compatible for equivalent explicit declarations. Internal rendered expression
strings are a valid wire/runtime representation; the user request replaces how
Python calculations are authored, not the serialized execution language. Existing
identifier and literal quoting must remain injection-safe; never use Python `eval`.

For continuous use, build the same typed declarations with explicit schema and
ordering, construct Program, and call `compile_stream(runtime)`. Continue to use
existing `StreamingRunner`, `SourceBinding`, `SinkBinding`, checkpoint configuration,
static-input bindings and job controls. The returned stream plan retains its current
physical `source_binding_ids`, `static_input_ids`, `sink_binding_ids` contract.
Convenience batch execution never starts a stream or invents a checkpoint path.

Preserve all registered connectors and project-backed `compile_stream_project`
workflows. Stream project export still requires the existing explicit operational
bindings/state settings to launch a job; no new defaults for delivery or recovery.
Studio `/api/v3`, OpenAPI/generated TypeScript types and project schema require no
changes. Studio continues to inspect lowered nodes rather than execute Python
builders or accept live static parameter data in REST job submission.

NumPy/JAX registration, exact matrix-shape boundaries, trusted vectorized scalar
UDF registration, arbitrary provider plugins and SQL stay available through the
existing advanced APIs. `compute` cannot silently add symbolic UDF-call primitives,
unsupported array outputs or DataFrame backend selection. Adapt existing matrix
examples to the root expression imports and mapping/keyword conveniences, retaining
explicit Runtime registration and `Batch.from_array` for static values.

Do not remove Rust exports, change crate package names, publishing, version alignment
or trait contracts just to mark Rust internal. Rust source, rustdoc and examples
remain useful to maintainers and provider authors. Change their product positioning
and documentation navigation to runtime implementation/extension reference. The
Python package is the user-facing product and Rust owns execution, tables, plans,
stream state, checkpoints and connectors. No native source change is expected unless
a concrete binding gap is discovered and reported before expanding scope.

## Errors and typing

Use current `TypeError`/`ValueError` for Python construction/adapter misuse and
existing `CompileError`, `ExecutionError`, `ProviderError` and streaming exceptions
for their respective phases. Preserve existing symbolic issue codes and paths.
New adapter errors must name the input/parameter/output and remediation, for example:

- `compute.build: expected TableExpr, got int`.
- `inputs.orders: missing table input` / `inputs.typo: unexpected input name`.
- `inputs.quotes.schema.price: unsupported_type: ...`.
- `collect: multiple inputs require a mapping by declared input name`.
- `compute() cannot run inside an event loop; use compute_async()`.
- Existing expression truth-value errors continue to suggest `&`, `|`, `~` and
  `identical()` as appropriate.

Builder exceptions should preserve the original exception and traceback; don't wrap
them as native execution failures or print callback representations. Diagnostics
must not include table contents or credentials.

Provide real Python 3.13 annotations for public functions and methods; the package
already ships `py.typed`. New pure-Python APIs belong in their source annotations,
not fictional `_native.pyi` declarations. Only change native stubs if actual binding
signatures change. Type-check/document `compute_async` as awaitable and collect return
types as Arrow tables/mappings. Avoid broad `Any` in new public arguments.

## Documentation and migration map

The implementation owner supplies focused runnable examples; the doc writer performs
one final coordinated alignment after review. Preserve historical artifacts and
`tests/fixtures/v1/`; do not rewrite old historical API notes as current guidance.

- `README.md`, `docs/introduction.md`, `docs/getting-started.md`: Python product
  positioning and the `compute` example first; move Rust installation/quickstart
  detail to runtime reference. Explain expressions before Batch/Port/graph concepts.
- `docs/README.md`, `docs/api-reference.md`, `docs/python-api.md`: one Python API
  learning path and preferred imports, then advanced integration; keep a complete
  reference of supported compatibility paths and explicit graph/SQL functionality.
- `docs/batch-guide.md`: operators, named expressions, reusable builders,
  select/filter, Arrow returns, multi-output Program collection, async/options and
  explicit SQL/UDF escape hatches. State schema and append-only name-collision rules.
- `docs/symbolic-api.md`, `docs/symbolic-workflows.md`: retitle in place as expression
  reference/workflows to preserve URLs; root imports and mapping/keyword examples;
  remove the obsolete claim that the user must always handle execution plans.
  Preserve the declaration-vs-runtime explanation and executable capability matrix.
- `docs/array-guide.md`, `docs/streaming-guide.md`, `docs/projects-guide.md`,
  `docs/studio-guide.md`, `docs/connectors/*.md`: align imports and terminology;
  explain how the same expressions reach each existing integration. Add public
  `to_project` export and retain operational, persistence and REST limitations.
- `docs/design.md`, `docs/symbolic-design.md`, `docs/runtime-envelope.md`,
  `docs/rust-api.md`: explicit Python API / Rust internal-runtime ownership;
  compiler, advanced runtime contracts, native extension references and unchanged
  checkpoint semantics. Preserve implementation diagrams and valid source links.
- `docs/examples.md`, `examples/README.md`, `examples/01_datafusion_pipeline.py`,
  `examples/05_async_execution.py`, and expression examples 09–13: make the default
  paths executable examples of the new API. Keep a deliberate advanced builder/SQL
  example and meaningful native examples rather than deleting coverage.
- `examples/14_project_persistence.py`: demonstrate expression Program export and
  round-trip via existing project/store APIs; explain native binding names on reload.
- `CHANGELOG.md`: record Python-first API, new convenience execution, schema and
  mapping acceptance, migration/compatibility, and unchanged protocols.
- `pyproject.toml` description, relevant Python module docstrings, and product
  overview paragraphs in `AGENTS.md`/`CLAUDE.md`: align positioning and new contracts.
  Do not rewrite specialist definitions, release machinery or verification policy.
- Performance, verification, release and backend READMEs need only terminology/link
  reconciliation where affected; measured historical results and valid commands stay
  intact. No whole-repository prose rewrite is required for unaffected content.

Migration is additive at the callable/import level: existing formula builder,
`FeatureSet` tuple pairs, explicit Program inputs/output pairs and positional Runtime
calls continue working. Examples teach new authors the expression route. Explicit
string SQL and project expressions remain advanced supported surfaces, not deprecated
runtime features. There is no version bump or v1 compatibility restoration in scope.

## Testable acceptance and bounded verification

1. The six-line `compute` quickstart executes from root imports with no manual schema,
   name, Batch, Runtime, plan or port and returns an Arrow table containing `[3, 7]`.
2. The order example computes multiple named expressions, filters derived values,
   projects named columns in the requested order, and leaves input data unchanged.
3. Root and compatibility imports are the same expression/program objects. Existing
   explicit constructors retain frozen digest/fingerprint golden vectors. New schema,
   mapping and keyword forms normalize identically to their old explicit equivalents.
4. Supported Arrow schema inference preserves field types/nullability/order; unsupported
   fields fail with named paths. Cover at least integer, floating, string, nullable
   fields and supported timestamps. Verify clear rejection of field names outside
   the existing portable identifier contract, and correct quoting of arbitrary
   string literal values so convenience syntax does not introduce formula parsing
   or injection.
5. Arithmetic/reflected arithmetic, boolean filters, composed financial functions,
   literal construction and structural identity retain existing semantics. Include
   a real computed-column regression, truth-value rejection, cross-lineage rejection,
   duplicate-output/collision errors and explicit temporal-ordering failure.
6. Automatic Program roots are deterministic; explicit `inputs=()` still reports
   missing declarations. Conflicting same-name roots fail. Mapping/kwargs inputs are
   copied and later caller mutation cannot change the declaration.
7. Program collection uses logical input/output names for one input, independent
   same-schema inputs, multiple outputs sharing an input, shared/CSE computation and
   supported table-attached matrix/static-input execution. Assert actual distinct
   values so a swapped mapping cannot pass; reject missing/extra/wrong-kind inputs.
8. Repeated and concurrent independent collect calls with the same explicit Runtime
   do not retain rolling state or reset an already compiled caller plan. Existing
   explicit plan snapshot/restore/reuse contracts remain unchanged.
9. Async convenience produces equal results, forwards options/deadlines and preserves
   cancellation cleanup through the existing native bridge. Blocking forms reject an
   event loop before invoking a builder. The builder is called once, with a TableExpr,
   and invalid/asynchronous builder results fail before native execution.
10. `to_project` yields a strict v3 document; existing JSON/YAML/store round-trip and
    native recompilation work. It serializes no data, function, expression object or
    alias-only schema extension. Existing explicit lowered documents, graph port
    names, schema/OpenAPI/generated API files and stream settings do not drift.
11. A focused existing streaming compile/recovery case and a registered-provider
    example continue to work with the new preferred imports/declarations. Do not
    broaden runtime feature claims or modify checkpoint hashes for syntax sugar.
12. Main documentation and runnable examples use one expression-first Python path;
    Rust is positioned as runtime/extension reference, advanced APIs remain
    discoverable, and limitations/compatibility are accurately documented.

Implementation starts with the smallest focused failing tests for the new behavior.
Run only the new public-API test module plus directly affected expression/program,
lowering/binding, async or provider cases and touched examples. Native builds are
needed only to supply the extension for those tests or if native code changes.
Run targeted Ruff/type checks and `git diff --check`; validate changed documentation
links/structure and generated-contract non-drift. The reviewer selects any additional
focused independent evidence based on the actual diff.

Full cross-platform regression, combined Rust 90% line coverage, Python coverage,
Studio backend 85%, release artifacts and routine performance gates remain CI's
responsibility. This design stage ran no builds/tests/performance and triggers no CI.
After a later authorized push take one non-blocking checks snapshot; pending is
reportable but not merge-ready. No push, PR creation or merge is authorized here.

## Handoff

There are no unresolved product decisions that require a critic or another spec.
Proceed to `cf-implementer`, then mandatory `cf-reviewer`, then one `cf-doc-writer`
alignment. The concrete implementation risks are logical-to-physical binding,
independent plan state ownership, schema normalization and async cancellation; the
acceptance items above bound them. If an implementation reveals that one of these
requires changing native state, checkpoint, IR identity or REST semantics, report
that specific blocker rather than broadening the design silently.
