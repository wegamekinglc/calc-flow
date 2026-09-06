# Symbolic API

[Documentation](README.md) / 3.3 Symbolic API

On this page:

- [Declarations and analysis](#symbolic-declarations-and-static-analysis)
- [Compilation](#symbolic-compilation)
- [Bounded stream joins](#symbolic-bounded-stream-joins)
- [Matrix compilation](#symbolic-matrix-compilation)

## Symbolic declarations and static analysis

For end-to-end batch, continuous, recovery, static NumPy/JAX matrix, Studio
inspection, and performance workflows, see the
[symbolic workflow guide](symbolic-workflows.md).

`calc_flow.symbolic` is the pure declaration surface: typed immutable
expressions, feature sets, and programs with canonical identities plus static
analysis over the declaration graph. It has no data execution path — there is
no `eval`, `push`, `value`, `transform`, preview evaluator, or formula parser —
and execution stays owned by the existing execution plans and runners.

The declaration catalog is intentionally wider than the implemented project
lowerers. Use this availability matrix when constructing user-facing formula
editors or validating stored declarations:

| Domain                   | Construct/analyze | Batch/stream compile  | Current lowering boundary                                             |
|--------------------------|-------------------|-----------------------|-----------------------------------------------------------------------|
| row-local columns        | yes               | yes                   | portable scalar types and the documented SQL allowlist                |
| rolling `ts`             | yes               | yes                   | source/alias/row-local operands and earlier rolling results           |
| cross-section `cs`       | yes               | yes                   | staged values; event time and partitions resolve to inputs or aliases |
| relational stream joins  | yes               | stream only           | independent/nested native joins with proved post-join ordering        |
| symbolic matrix          | yes               | exact supported shape | one static `weights` parameter and one allowlisted matmul             |
| event `window`           | yes               | no                    | declaration-only; compilation fails closed                            |
| standalone array outputs | yes               | no                    | arrays compile only through the supported table attachment            |

`Program.analyze` reports `unsupported_type` for stateful operands outside
the current materialization boundary, so a clean analysis does not advertise
an expression that the lowerer will reject for that reason.

```python
from calc_flow import Runtime
from calc_flow.symbolic import FeatureSet, Field, Program, table_input

quotes = table_input(
    "quotes",
    schema=[
        Field("ts", "timestamp[us, UTC]", nullable=False),
        Field("x", "float64"),
        Field("y", "float64"),
    ],
)
signals = quotes.with_columns(FeatureSet([("score", quotes["x"] + quotes["y"])]))
program = Program("p", inputs=[quotes], outputs=[("signals", signals)])

result = program.analyze(Runtime(), mode="batch")
assert result.issues == ()
```

`table_input` declares one named table input with an exact ordered schema — a
sequence of `Field` values, never a mapping; `event_time`, `entity_by`, and
`sequence_by` declare the ordering facts of a temporal input. Selecting
`quotes["x"]` builds a `ColumnExpr`; arithmetic, comparison, and boolean
composition build immutable expression nodes whose canonical v1 digests are
stable across processes. Public comparisons build symbolic expressions:
converting an expression to `bool` fails, and `identical()` is the structural
identity check.

A `FeatureSet` is an ordered immutable set of uniquely named column
expressions; `with_feature` appends one. `TableExpr.with_columns(features)`
returns a new table with the declared features appended as derived columns.
The `row`, `ts`, `cs`, `table`, `linalg`, and `window` namespaces expose
row-local functions, rolling frames (`rows`/`duration`), cross-section groups
(`exact_time`/`event_time_bucket`), table bridges, and matrix work.

A `Program` declares uniquely named inputs (`table_input` or `parameter`
values) and outputs (tables or arrays) in declaration order. Its `fingerprint`
is the runtime-independent `calc_flow.symbolic.declaration.v1` program
fingerprint over every unique node reachable from a declared input or output;
it does not depend on construction history and is stable across conforming
implementations. Duplicate declared names fail at construction with stable
paths such as `inputs.quotes: duplicate_name`. An input referenced by an
output but missing from `inputs` is reported during analysis as an issue
rooted at `inputs.<name>`.

`Program.analyze(runtime, mode=...)` and `Program.explain(runtime, mode=...)`
require an explicit `Runtime` and a `batch` or `stream` mode; both consume one
immutable capability snapshot and record its session and revision. From the
declaration graph alone — no data object, source, sink, or runner is accepted —
the analysis proves:

- value types, proving only what the capability snapshot proves: identical
  operand types, the safe float32/float64 Array API promotion, the frozen
  rolling/cross-section output-type table, and exact field resolution;
  unsupported row-local cross-type arithmetic needs an explicit `row.cast`;
- domains and row lineages, rejecting cross-input lineage mixing;
- symbolic dimensions through `linalg.from_columns` and `linalg.matmul`,
  retaining the row axis of a table-derived array;
- attachment compatibility for `table.attach_columns`;
- state requirements per output, rendered by `explain` as
  `state cross_section, duration(60000000)`-style facts; and
- stream safety: temporal and cross-section inputs need an event-time column
  with entity and sequence keys, and a stream-mode array output with row-axis
  lineage is reported as unbounded state.

`analyze` returns an immutable `AnalysisResult` carrying `mode`,
`program_fingerprint`, `capability_session_id`, `capability_revision`, and an
`issues` tuple; each finding is an immutable `AnalysisIssue` with a stable
`path`, `code`, and `message`. Paths start at a named program output or input —
for example `outputs.signals.score`, `outputs.scores.matmul.right.shape[0]`,
`inputs.quotes.sequence_by[0]`, and `static_inputs.weights`. Analysis is
deterministic: it never mutates a declaration node, and repeated runs return
equal results. For programs supported by lowering, `explain` also reports the
physical CSE, rolling, cross-section, and array-fusion stage counts. Its cost
section states bounded rolling rows or durations, cross-section group bounds,
bounded stream-join row/byte/match limits, retained fixed/variable-width
columns, explicit table-to-dense and host-to-device copy boundaries,
static-weight bytes when known, and provider calls per micro-batch. These are
compile-time estimates and shape facts;
runtime resident-memory and measured copy metrics remain authoritative. The
report never contains row payloads, static values, secrets, callable
representations, or object addresses. The frozen analysis vocabulary is
`capability_mismatch`, `duplicate_name`, `invalid_literal`,
`ordering_required`, `schema_mismatch`, `unbounded_state`,
`unknown_primitive_version`, `unresolved_type`, `unsupported_mode`, and
`unsupported_type`; construction errors raise `ValueError` or `TypeError`
with the same path grammar. `explain` renders the same facts as a deterministic
multi-line report.

## Symbolic compilation

`Program.compile_batch(runtime)` and `Program.compile_stream(runtime, *,
allowed_lateness_micros=0, late_policy="error")` lower a program to the
existing execution plans. Compilation is declaration processing only: it
captures one immutable capability snapshot, lowers one strict project-v3
document, and invokes the Rust graph compiler for final port, kind, schema,
topology, and fingerprint validation. No data object, source, sink, or runner
is accepted, and no symbolic Python runs while a compiled plan executes.

Row-local declarations — literals, fields, arithmetic, comparison, boolean
composition, `where`, `coalesce`, `log`/`exp`/`sqrt`/`abs`/`clip`/`cast`, and
the `table.project`/`table.filter`/`with_columns` table operations — fuse into
one `expression` node per program output; a `where`/filter predicate becomes
the node's `WHERE` clause. Structurally identical non-trivial subexpressions
referenced at least twice are computed exactly once: they materialize as
`__cf_cse_N` columns in deterministic tier nodes (`<output>__cf_cse_<k>`)
ahead of the fused node, so a 20-output `FeatureSet` with no shared
subexpressions compiles to a single fused node. Node IDs and the plan
fingerprint are deterministic, and the lowered project carries strict JSON
only.

Optimization runs over the complete program after analysis. Identical,
connected pure expression materializations are emitted once and fan out to
each consumer. Output branches over the same input and prefilter share one
rolling operator, including its history and partition index; branches over
the same upstream, partition keys, and exact-time or fixed-bucket finality
share one cross-section grouping/sort stage. A different prefilter, grouping,
bucket width, or upstream state stage is a hard materialization boundary, so
the optimizer does not move a filter across temporal or cross-section
finality. Table/array and backend transitions remain explicit boundaries, and
the allowlisted array expression is fused into one provider call per accepted
micro-batch.

Each `Runtime` keeps a runtime-scoped compile cache of immutable plan values.
The deterministic key contains the program fingerprint, batch/stream mode,
stream lateness policy, exact input declaration bytes (including schemas),
capability schema/session/revision, and selected operator, provider, and UDF
versions. Python object identity is never a key input. Repeating a compile on
the same runtime and key returns the cached plan; any successful provider,
stream-lifecycle, or UDF registration invalidates that runtime's entries.

Programs with one input and one output bind the plan endpoints `input` and
`output`, matching the `PipelineBuilder` convention; multi-branch graphs name
endpoints `<node>.input` and `<node>.output` deterministically. Batches
supplied at execution must match the declared input schema exactly.

## Symbolic bounded stream joins

`table.stream_join(left, right, /, *, left_keys, right_keys,
left_event_time, right_event_time, bounds, limits, left_prefix="left",
right_prefix="right", output_entity_by=(), output_event_time=None,
output_sequence_by=())` declares the existing native bounded inner join.
`bounds` is a public `JoinTimeBounds`; `limits` is a public `JoinStateLimits`.
Both key sequences are non-empty and equal in length, and the corresponding
resolved fields must have identical supported Arrow types. Each event-time
name resolves to a non-null `timestamp[us, UTC]` field.

Both inputs must declare `event_time`, non-empty `entity_by`, and non-empty
`sequence_by` ordering facts for stream analysis. Batch analysis and
compilation fail with `unsupported_mode`. The output schema is deterministic:
all left fields named `{left_prefix}__{name}` in source order, followed by all
right fields named `{right_prefix}__{name}` in source order, with exact type
and nullability preserved.

Omitting all three output-ordering arguments constructs the
symbolic `stream_join@1` declaration. Supplying any one requires all
three and builds symbolic `stream_join@2`: `output_entity_by` must be the
prefixed left join keys, `output_event_time` must select one prefixed join
event-time field, and `output_sequence_by` must concatenate every prefixed
left and right input sequence key. The metadata is immutable and declarative;
it does not sort rows or create a Python data path.

Programs may contain independent joins, ordered nested joins, outputs
unrelated to a join, and rolling or cross-section state after an ordered join.
Each unique declaration digest shares one physical join node. A joined value
feeding another join or stateful stage without complete valid ordering fails
with `ordering_required`. Projection removes ordering facts when it removes a
named ordering field. Matrix attachment around a join remains unsupported,
and symbolic event-window declarations still have no executable lowerer.

Lowering copies every declaration into the existing project-v3 join spec with
no symbolic-only serialized fields. Native watermarks, inclusive time bounds,
state limits, match order, metrics, checkpoint state v1, and recovery remain
authoritative. A direct join root exposes source binding ids `left` and
`right`; a relational DAG exposes `<declared-input>.input` source bindings.
Single-output plans retain sink binding `output`, while multi-output graphs use
ordinary `<node>.output` names. See
[`12_symbolic_stream_join.py`](../examples/12_symbolic_stream_join.py) for a
segmented two-source execution and the
[`13_symbolic_relational_dag.py`](../examples/13_symbolic_relational_dag.py)
for an ordered nested join.

## Symbolic matrix compilation

A single table output shaped as `table.attach_columns(table_value, array,
names=...)` lowers the explicit table/array bridge to the selected
`numpy:symbolic_matrix@1` or `jax:symbolic_matrix@1` provider. The array may
contain `linalg.from_columns`, allowlisted elementwise arithmetic and boolean
operations, finite literals, and exactly one `linalg.matmul`. The static array
parameter named `weights` must occur exactly once in the entire array
expression, as that matmul's direct right operand. The selected column order is
semantic; every `linalg.from_columns` source must be the table being attached,
and the derived array may attach only to that same table node and row lineage.
Rank two, matching matmul inner dimensions, a known positive output width, and
one attached name per result column are proved before lowering. Cross-backend
composition fails closed.

Batch plans bind the table and weights as `input` and `weights`. Stream plans
instead declare `weights` as a project-v3 static array input, so only `input`
is a source binding. The runner latches the caller value before opening a
source, places it into NumPy or JAX once per job, retains that immutable
provider array for every micro-batch, and invokes the fused provider once per
accepted data micro-batch. Output metadata exposes `provider_calls: 1` and
`copy_bytes` entries `table_to_array`, `array_to_table`, and `weights`. The
`weights` entry mirrors `static_placement_bytes`: dtype width multiplied by
logical element count on first placement and zero on cached later
micro-batches. It is logical provider transfer, not peak memory, process RSS,
or the transient internal snapshot clone. Results are row-axis independent and
therefore invariant to source segmentation within the normal NumPy/JAX
floating tolerance.

Selected Arrow columns must be unique, non-null primitive numerics with one
dtype. The provider chooses a lossless common backend dtype with the weights
before staging; float32/float64 promotes to float64. JAX rejects a required
float64 path when x64 is disabled and accepts it when `JAX_ENABLE_X64=true`.
Runtime-dependent schema, null, weight backend/shape, and output row-count
checks happen before attachment and report the failing provider field.

`ts.lag`, `ts.delta`, `ts.ewma` (with identity alias `ts.ema`), and the
rolling aggregates `ts.count`, `ts.sum`, `ts.mean`, `ts.min`, `ts.max`,
`ts.variance`, `ts.stddev`, `ts.covariance`, and `ts.correlation` lower to one
or more native `rolling` stages per program output, placed ahead of the fused
row-local stages. Rolling requires the input table to declare its `entity_by`,
`event_time`, and `sequence_by` ordering keys; a program missing them fails with
`ordering_required`. A rolling argument may resolve to a plain input column,
a direct or derived row-local alias, a pure row-local expression such as
`ts.lag(row.log(quotes["x"]))`, or an earlier rolling result. The lowerer
schedules nested state as an innermost-first DAG and materializes each distinct
row-local bridge once before its consuming state stage. Single-stage programs
retain the deterministic `<output>__cf_rolling_input` and
`<output>__cf_rolling` identifiers; multi-stage programs number both stage
kinds from one. Thus `ts.mean(ts.delta(quotes["x"]), window=rows(3))` lowers
to two rolling stages, while an RSI composition inserts its positive/negative
row-local projection between the delta and mean stages. `periods` is a
positive integer defaulting to `1`; an aggregate declares `rows(size)` or
`duration(micros)` as its window with `min_periods` (default `1`, capped at
the frame size for `rows` windows only) and — for `variance`, `stddev`,
`covariance`, and `correlation` — `ddof` of `0` or `1` (default `1`), and
construction rejects other values with `ValueError`. Every rolling
occurrence in one output shares its compatible rolling stage: a whole-feature
occurrence keeps its feature name as the output column, while an occurrence
nested inside a larger expression materializes as a deterministically named
internal column that the next state or fused row-local stage references.
Identical multi-stage pipelines across output branches share the same physical
state nodes. A lag, delta, `min`, or `max` column keeps the input column's type; the
remaining aggregate columns take the frozen output type — `uint64` for
`count`, `int64` or `uint64` for an integer `sum`, `float64` otherwise —
and the engine evaluates the frozen window semantics described in the
[Rust API guide](rust-api.md).
The direct difference of two `mean`, `variance`, `stddev`, or EWMA expressions
at the same finality boundary lowers as one native `difference` output. Its two
leaf states still participate in ordinary group sharing, but neither leaf is
materialized as a hidden Arrow column; the rolling builder writes only the
nullable `float64` result. This is the path used by dual-SMA spreads and the
fast-minus-slow portion of `ts.macd`.
`ts.ewma(value, span=n, min_periods=m)` instead declares constant exponential
state: `n` and `m` are positive, the first valid value seeds the unadjusted
average exactly, and later valid values apply
`average += 2 / (n + 1) * (value - average)`. Null and NaN inputs leave both
the valid count and average unchanged; infinity participates through ordinary
IEEE arithmetic. The result is nullable `float64`. `ts.macd` defaults to
`fast_span=12` and `slow_span=26`, requires the fast span to be smaller, and
is only the row-local difference of those two EWMA declarations, so ordinary
CSE and native state sharing apply.

A filter declared below every rolling feature becomes a deterministic
`<output>__cf_prefilter` expression node feeding the rolling node; a filter
declared above them applies after it. The lowered node's frozen spec carries
`configuration_version` 1 and the declared ordering
keys, one entry per rolling output (`kind`, `primitive_version` 1, `input`,
`output`, and `periods`, or `frame`, `min_periods`, and — for the statistical
kinds — `ddof`; the pair kinds carry `left` and `right` in place of `input`;
EWMA carries `span` and `min_periods`), the `allowed_lateness_micros` and `late_policy`
values validated by `compile_stream` — `error` lowers to an envelope-scoped
rejection and `drop` to a metrics-recorded drop — and the
`stateful_numeric_v1` value policy, which preserves a null or NaN current or
referenced value. Batch lowering writes the default lateness values, batch
evaluation classifies no late rows, and a program without rolling
declarations is unaffected by the lateness arguments. Non-EWMA declarations
use `state_layout_version` 1; a declaration containing EWMA uses version 2.
The native checkpoint writer uses columnar layout 3 and persists exponential
accumulators exactly; see [native rolling state](symbolic-design.md#native-rolling-state).

`cs.rank`, `cs.percentile`, `cs.demean`, `cs.zscore`, `cs.winsorize`,
`cs.top`, `cs.bottom`, and `cs.mean_fill` lower to one shared native
`cross_section` node per compatible grouping, placed ahead of the fused
row-local stages. The measured value may be a source column, alias, row-local
expression, or rolling result; row-local values materialize once immediately
before grouping. Event time and every group column must still resolve to a
source input column or direct alias. Compatible output branches share the
row-local materialization plus native grouping/sort state. They use the same
`ordering_required` key requirement as rolling. A group is declared with
`exact_time(event_time, *, partition_by=())` or
`event_time_bucket(event_time, *, width_micros, partition_by=())`, and every
cross-section occurrence in one output must share one grouping declaration:
the same partition columns and the same grouping shape — exact time, or one
bucket width — or compilation fails with `schema_mismatch` rooted at the
output. `rank` and `percentile` carry `direction` (default `ascending`),
`tie_method` (default `average`), and `null_placement` (default `exclude`);
every primitive carries `min_samples` (default `1`), and `zscore` adds `ddof`
of `0` or `1` (default `0`). `winsorize` requires finite `lower`/`upper`
probabilities with `0 <= lower <= upper <= 1`. `top` and `bottom` require a
positive `count`, default `include_ties` to true, and return nullable boolean
masks; disabling tie inclusion uses canonical row identity at the boundary.
`mean_fill` fills null float32/float64 values from the complete valid sample
while preserving NaN and the input floating type. Construction rejects every
invalid option before lowering. As with rolling, a whole-feature occurrence
keeps its feature name as the
output column while an occurrence nested inside a larger expression
materializes as a deterministically named `<output>__cf_cs_<index>` column,
and a filter below every cross-section feature becomes the shared
`<output>__cf_prefilter` node. Rank, percentile, demean, and z-score are
nullable float64; winsorize and mean-fill preserve float32/float64; top/bottom
are nullable boolean. The engine evaluates the frozen complete-group semantics
described in the [Rust API guide](rust-api.md). The lowered node's frozen
spec carries `configuration_version` and `state_layout_version` 1, the
declared ordering and partition keys, the shared grouping, one entry per
output, the `allowed_lateness_micros` and `late_policy` values validated by
`compile_stream`, and the `nan_exclude_preserve_v1` value policy.

Analysis rejections surface as `CompileError` with the first issue's
`{path}: {code}: {message}`. Declarations outside the implemented lowerers —
including event `window` nodes and `linalg`/`parameter` uses that do not form
the exact symbolic matrix compilation shape above — fail with the eighth issue
code `unknown_primitive_version` rooted at the output or
`static_inputs.<name>`, in both batch and stream modes; a stream aggregate or
SQL window is never silently made batch-local. Standalone array outputs fail
with `unknown_primitive_version` in batch mode; stream mode rejects them
earlier, at the analysis phase, with `unbounded_state` rooted at
`outputs.<name>` — the stream-safety rule for an array output with row-axis
lineage described above. Casts to non-portable targets fail with
`unsupported_type` at `outputs.<name>.cast.data_type`.

Stream plans also reject read-only queries that call volatile built-in
functions (for example `random()`) or the wall-clock built-ins `now`,
`current_date`, and `current_time` (aliases such as `current_timestamp`
and `today` included): `compile_stream` resolves every function in an
expression or SQL node's query against the built-in default function
registry, matching the resolved canonical name, and fails volatile and
wall-clock calls before any source opens, so the deterministic,
replay-safe lifecycle claims of those operators remain truthful.

Next: [Rust API](rust-api.md).
