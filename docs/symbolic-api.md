# Symbolic API

[Documentation](README.md) / 3.3 Symbolic API

On this page:

- [Declarations and analysis](#symbolic-declarations-and-static-analysis)
- [Compilation](#symbolic-compilation)
- [Event-time window aggregation](#symbolic-event-time-window-aggregation)
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

| Domain                   | Construct/analyze | Batch/stream compile   | Current lowering boundary                                             |
|--------------------------|-------------------|------------------------|-----------------------------------------------------------------------|
| row-local columns        | yes               | yes                    | portable scalar types and the documented SQL allowlist                |
| rolling `ts`             | yes               | yes                    | source/alias/row-local operands and earlier rolling results           |
| cross-section `cs`       | yes               | yes                    | staged values; event time and partitions resolve to inputs or aliases |
| relational stream joins  | yes               | stream only            | independent/nested native joins with proved post-join ordering        |
| symbolic matrix          | yes               | exact supported shape  | one static `weights` parameter and one allowlisted matmul             |
| event `window`           | yes               | stream with aggregates | fixed UTC tumbling/hopping; stateless table work on either side       |
| standalone array outputs | yes               | no                     | arrays compile only through the supported table attachment            |

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
  with entity and sequence keys; event windows require an exact supported
  timestamp field without that ordering metadata. A stream-mode array output
  with row-axis lineage is reported as unbounded state.

`analyze` returns an immutable `AnalysisResult` carrying `mode`,
`program_fingerprint`, `capability_session_id`, `capability_revision`, and an
`issues` tuple; each finding is an immutable `AnalysisIssue` with a stable
`path`, `code`, and `message`. Paths start at a named program output or input —
for example `outputs.signals.score`, `outputs.scores.matmul.right.shape[0]`,
`inputs.quotes.sequence_by[0]`, and `static_inputs.weights`. Analysis is
deterministic: it never mutates a declaration node, and repeated runs return
equal results. For programs supported by lowering, `explain` also reports the
physical CSE, rolling, cross-section, event-window, and array-fusion stage
counts. Its cost section states bounded rolling rows or durations,
cross-section group bounds, bounded stream-join row/byte/match limits, retained fixed/variable-width
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

`log`, `exp`, and `sqrt` compute floating inputs as `float64`; `clip`
preserves its floating input type. Lowering inserts explicit conversions for
`float32` expressions so native results match the analyzed schema, including
at an event-window boundary. Unsigned column negation is rejected during
analysis; use `row.cast` to select a supported signed or floating type first.

Optimization runs over the complete program after analysis. Identical,
connected pure expression materializations are emitted once and fan out to
each consumer. Output branches over the same input and prefilter share one
rolling operator, including its history and partition index; branches over
the same upstream, partition keys, and exact-time or fixed-bucket finality
share one cross-section grouping/sort stage. A different prefilter, grouping,
bucket width, or upstream state stage is a hard materialization boundary, so
the optimizer does not move a filter across temporal, cross-section, or
event-window finality. Table/array and backend transitions remain explicit
boundaries, and the allowlisted array expression is fused into one provider
call per accepted micro-batch.

Each `Runtime` keeps a bounded, runtime-scoped symbolic compile cache.
The deterministic key contains the program fingerprint, batch/stream mode,
stream lateness policy, exact input declaration bytes (including schemas),
capability schema/session/revision, and selected operator, provider, and UDF
versions. Python object identity is never a key input. Batch compilation
returns the cached immutable plan for the same key. Stream compilation caches
the immutable project JSON after successful native compilation and returns a
fresh owning native plan on every call: a `StreamingRunner` consumes its plan.
Compile again and create fresh bindings and a runner for each job or restart.
Any successful provider, stream-lifecycle, or UDF registration invalidates
that runtime's entries. Cache facts do not promise a measured compile speedup.

Programs with one input and one output bind the plan endpoints `input` and
`output`, matching the `PipelineBuilder` convention; multi-branch graphs name
endpoints `<node>.input` and `<node>.output` deterministically. Batches
supplied at execution must match the declared input schema exactly.

## Symbolic event-time window aggregation

`window.tumbling(value, /, *, event_time, size_micros, group_by=(),
aggregates=None)` and `window.hopping(value, /, *, event_time, size_micros,
slide_micros, group_by=(), aggregates=None)` declare fixed UTC event windows.
Pass a non-empty sequence of immutable `WindowAggregate(function, column,
output)` values to execute them with `Program.compile_stream(runtime)`.
The helpers `window.count`, `window.sum`, `window.min`, `window.max`, and
`window.avg` each accept a positional column name and a required keyword-only
`output` name. All three declaration fields are non-empty strings; `function`
is one of those five exact names. Declarations contain no expressions,
callables, SQL, or live data. Compute a row-local value with `with_columns`
before the window, then aggregate its named column.

`count(column)` counts non-null values and always requires a column; there is
no `count(*)` helper. Choose a non-null trade ID to count every trade.
`avg` is the ordinary arithmetic mean. It does not weight prices by volume.
Run the [minute aggregation example](../examples/symbolic_event_window.py)
for grouped count, volume, low, high, and average price with explicit source
watermarks.

The aggregate and grouping sequences are copied immediately and retain their
declaration order. Aggregate output names must be unique and must not collide
with a grouping key, `window_start`, or `window_end`. Grouping keys must also
be unique and cannot use those two reserved names. Multiple aggregates over
the same input column may use distinct output names; an output may reuse an
ordinary input column name that is absent from the result.

A non-empty `aggregates` sequence constructs `window_tumbling@2` or
`window_hopping@2`. Omitting it or passing `None` constructs the
declaration-only `@1` form with its stable canonical bytes and digest;
compilation rejects that form with `unknown_primitive_version`. An explicit
empty sequence is rejected with `invalid_literal`. Mappings, sets,
generators, strings, and sequences containing other value types are rejected.
The following execution rules apply to the aggregate-bearing `@2` forms.

### Schema and geometry

The input is a table with an exact ordered schema. `event_time` names a
`timestamp[ms]`, `timestamp[us]`, or `timestamp[us, UTC]` field. Naive
timestamps use the native UTC coordinate, and null timestamps are allowed.
Event windows do not require `entity_by`, `sequence_by`, or a non-null time
declaration. The time field must pass through from an input unchanged or via a
pure rename; time arithmetic, truncation, or casts that change its coordinate
are rejected because they do not preserve the source watermark coordinate.

Grouping keys and `min`/`max` inputs accept `bool`, signed and unsigned
8/16/32/64-bit integers, `float32`, `float64`, `string`, `large_string`,
`date32`, `date64`, `timestamp[us]`, and `timestamp[us, UTC]`.
Numeric aggregates accept those integer and floating types. `count` accepts
any currently supported portable symbolic field type representable in
project-v3, including `timestamp[ms]` and time fields. Those additional types
do not become valid grouping or `min`/`max` inputs.

| Output                       | Arrow type                 | Nullable       |
|------------------------------|----------------------------|----------------|
| `window_start`, `window_end` | `timestamp[us, UTC]`       | false          |
| Grouping key                 | Preserved input field type | Input nullable |
| `count`                      | `uint64`                   | false          |
| Signed integer `sum`         | `int64`                    | true           |
| Unsigned integer `sum`       | `uint64`                   | true           |
| Floating `sum`, every `avg`  | `float64`                  | true           |
| `min`, `max`                 | Preserved input field type | true           |

Output columns are exactly `window_start`, `window_end`, grouping keys in
declaration order, then aggregate outputs in declaration order. Raw input
columns, sequence keys, and internal helper columns are not appended.

For stateless table transformations before or after a window, analysis first
checks declared field names, types, and expression paths. It then confirms
the actual lowered expression stages, including shared-expression stages,
with the native stream schema planner. Names and types must match the frozen
declarations; nullability comes from the native result. This includes the
effects of DataFusion's CASE and boolean simplifications. Grouping keys
preserve the exact window-input field nullability; the window's aggregate
outputs retain the rules in the table above.

Schema confirmation uses the native column-projection path when applicable
and otherwise plans against an empty table with the declared Arrow schema.
It opens no source, processes no user rows, and executes no registered UDF.
Planning failures or incompatible names/types reject analysis before a job
starts. Each `Runtime` separately caches up to 128 successfully planned
immutable Arrow schemas, clearing them with its compile cache on successful
registration changes.

`size_micros` and `slide_micros` are exact positive Python integers no greater
than `2**64 - 1`; booleans, floats, strings, and durations are not accepted.
Hopping requires `size_micros % slide_micros == 0` and an overlap
`size_micros // slide_micros` from 1 through 1024. Geometry is anchored at the
Unix epoch with half-open intervals `[start, end)`. Event-time overflow that
depends on row values is checked by the native runtime.

### Final output, composition, and recovery

Each unique complete declaration lowers to one native
`WindowAggregateOperator` through the existing project-v3 `window` spec.
Identity includes the input graph, time field, geometry, ordered grouping
keys, and ordered aggregates with their output names. Identical declarations
share one physical window state across branches. Different aggregate lists,
ordering, or upstream filters create different state owners.

Data arrivals accumulate native state without early output. A watermark
closes windows whose end is less than or equal to it; end-of-input flushes the
remaining windows. Null-time rows are dropped and counted in native metrics.
An assignment is late when its end is less than or equal to the current input
watermark. Hopping drops only the closed assignments, so one row can still
contribute to other open windows. Empty windows produce no rows. Within an
existing group, all-null aggregate inputs produce zero for `count` and null
for the other functions. Each close sorts rows by window start, window end,
and the native stable group-key encoding, retaining native null, NaN,
signed-zero, overflow, metrics, and chunking semantics.

Each dependency path through a window supports stateless `table.project`,
`table.filter`, `with_columns`, and row-local expressions before and after
one window. Window results have a distinct row origin; they cannot mix with
raw input columns or arrays by position. The compiler rejects rolling,
cross-section, joins, another event window, array/matrix attachment, external
stateful providers, and cross-row reductions on that path. Independent
windows and unrelated legal outputs may coexist in the same `Program`.
Filters retain their declared side of the window finality boundary.

`allowed_lateness_micros` and `late_policy` on `compile_stream` belong to
rolling/cross-section operators. Event windows always apply native late
assignment dropping. A program containing windows but no independent
rolling/cross-section consumer rejects non-default values with
`capability_mismatch` at `<program>.compile_stream.<option>`; the default
`late_policy="error"` does not change window dropping into an error.
`analyze` and `explain` take their existing runtime and mode arguments, and
`explain` describes default compile options. Its window section reports
geometry, aggregates, physical state sharing, layout, finality, and ordering;
active window/group counts depend on runtime data.

Batch mode fails with `unsupported_mode`. Missing or unproven native
`window@1` capability fails with `capability_mismatch`. Invalid declarations
fail before any source opens, with field paths such as
`calc_flow.symbolic.window.hopping.slide_micros` or
`outputs.minute.window_tumbling.aggregates[0].column`; unsupported field
types append `.dtype` to the field path.

The native window owns accumulators, output sequence, and checkpoint layout
`1`. Bind sources and sinks through `StreamingRunner` and use
`ManagedCheckpointRuntime` for the existing aligned checkpoint and recovery
protocol. Recovering an open window restores that native state. There is no
Python row executor or separate Python window buffer. Session windows, local
calendar windows, offsets, early triggers, update/retraction outputs, custom
aggregates, and extra allowed-lateness semantics are unsupported.

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
named ordering field. Matrix attachment around a join remains unsupported.
Event windows have the separate stateless-path boundary described above and
cannot consume or feed a symbolic join.

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
evaluation classifies no late rows. Cross-section stages consume the same
options; event-window programs apply the option checks described above.
Non-EWMA declarations
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
including declaration-only `window_tumbling@1`/`window_hopping@1` nodes and
`linalg`/`parameter` uses that do not form the exact symbolic matrix compilation
shape above — fail with `unknown_primitive_version` rooted at the output or
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
