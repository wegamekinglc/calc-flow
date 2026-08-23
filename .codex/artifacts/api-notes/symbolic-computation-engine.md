# Symbolic Computation Engine API Contract

| Field             | Value                                                                   |
| ----------------- | ----------------------------------------------------------------------- |
| Status            | APPROVED — exact API/serialization freeze; implementation not started   |
| Issue             | GitHub #167 / SCE-00                                                    |
| Artifact slug     | `symbolic-computation-engine`                                           |
| Controlling spec  | `.codex/artifacts/specs/symbolic-computation-contract.md`, D1–D13       |
| Project format    | `3` (unchanged)                                                         |
| Capability schema | `2`                                                                     |
| Digest version    | `calc_flow.static_input.digest.v1`                                      |

## 1. Authority and scope

The controlling specification owns all semantics, defaults, finality, and
recovery behavior. This note makes only the exact spelling, type, nesting, and
encoding choices delegated to the API gate. It does not reopen D1–D13.

This note freezes:

- the initial Python declaration signatures and exports;
- the additive Rust specification/operator/runner signatures and crate-root
  exports;
- Python exception mapping and stable diagnostic paths;
- the strict project-v3 `rolling`, `cross_section`, and static-input shapes;
- capability schema v2 Python data classes and serialized camel-case shape;
- the canonical tagged bytes used for static-input SHA-256; and
- the checkpoint evidence and lookup order that reject changed static values
  in the existing lineage before sources open.

It does not authorize an implementation. Until the dependent SCE issues land,
these names remain proposed and must not be documented as available.

## 2. Python public surface

### 2.1 Module and exports

`calc_flow.symbolic` is the only public declaration module. The package root
adds only this module binding:

```python
from calc_flow import symbolic
```

`calc_flow.symbolic.__all__` is exactly:

```python
[
    "AnalysisIssue",
    "AnalysisResult",
    "ArrayExpr",
    "ColumnExpr",
    "CrossSectionGroup",
    "DurationFrame",
    "EventTimeBucket",
    "Expr",
    "FeatureSet",
    "Field",
    "Parameter",
    "Program",
    "RowFrame",
    "TableExpr",
    "cs",
    "duration",
    "event_time_bucket",
    "exact_time",
    "linalg",
    "parameter",
    "row",
    "rows",
    "table",
    "table_input",
    "ts",
    "window",
]
```

There are no compatibility aliases at `calc_flow` package root for expression
classes or namespace functions. In particular there is no public `eval`,
`push`, `value`, `transform`, preview evaluator, or formula parser.

### 2.2 Shared declaration values

All classes below are frozen and slotted. Constructors copy every supplied
mapping/sequence and never retain a caller-owned mutable collection.

```python
from dataclasses import dataclass
from typing import Generic, Literal, Never, TypeVar, overload
from collections.abc import Mapping, Sequence

type BatchKind = Literal["table", "array"]
type CompileMode = Literal["batch", "stream"]
type LatePolicy = Literal["error", "drop"]
type ScalarLiteral = None | bool | int | float | str
type ColumnOperand = ColumnExpr | ScalarLiteral
type ArrayOperand = ArrayExpr | Parameter[ArrayExpr] | ScalarLiteral

T = TypeVar("T")


@dataclass(frozen=True, slots=True)
class Field:
    name: str
    data_type: str
    nullable: bool = True


@dataclass(frozen=True, slots=True)
class RowFrame:
    size: int


@dataclass(frozen=True, slots=True)
class DurationFrame:
    micros: int


@dataclass(frozen=True, slots=True, eq=False)
class EventTimeBucket:
    event_time: ColumnExpr
    width_micros: int
    partition_by: tuple[ColumnExpr, ...] = ()


@dataclass(frozen=True, slots=True, eq=False)
class CrossSectionGroup:
    event_time: ColumnExpr
    bucket: EventTimeBucket | None
    partition_by: tuple[ColumnExpr, ...]


def rows(size: int, /) -> RowFrame: ...
def duration(micros: int, /) -> DurationFrame: ...

def exact_time(
    event_time: ColumnExpr,
    /,
    *,
    partition_by: Sequence[ColumnExpr] = (),
) -> CrossSectionGroup: ...

def event_time_bucket(
    event_time: ColumnExpr,
    /,
    *,
    width_micros: int,
    partition_by: Sequence[ColumnExpr] = (),
) -> CrossSectionGroup: ...
```

`RowFrame.size` and `DurationFrame.micros` are positive exact integers.
`EventTimeBucket.width_micros` is positive. A Python `bool` is rejected
wherever an integer is required. There are deliberately no unit-guessing
constructors and no `timedelta` serialization path; callers convert to exact
microseconds explicitly.

The initial portable field spelling adds `timestamp[us, UTC]` to the current
portable Arrow names. Temporal/cross-section event-time fields require that
exact spelling. Field order is the sequence order; a mapping is not accepted
as a schema.

### 2.3 Expressions, tables, features, and programs

```python
@dataclass(frozen=True, slots=True, eq=False)
class Expr(Generic[T]):
    @property
    def digest(self) -> str: ...
    def identical(self, other: object, /) -> bool: ...
    def explain(self) -> str: ...
    def __bool__(self) -> bool: ...
    __hash__ = None


@dataclass(frozen=True, slots=True, eq=False)
class ColumnExpr(Expr[object]):
    def __eq__(self, other: ColumnOperand, /) -> ColumnExpr: ...
    def __ne__(self, other: ColumnOperand, /) -> ColumnExpr: ...
    def __lt__(self, other: ColumnOperand, /) -> ColumnExpr: ...
    def __le__(self, other: ColumnOperand, /) -> ColumnExpr: ...
    def __gt__(self, other: ColumnOperand, /) -> ColumnExpr: ...
    def __ge__(self, other: ColumnOperand, /) -> ColumnExpr: ...
    def __add__(self, other: ColumnOperand, /) -> ColumnExpr: ...
    def __radd__(self, other: ColumnOperand, /) -> ColumnExpr: ...
    def __sub__(self, other: ColumnOperand, /) -> ColumnExpr: ...
    def __rsub__(self, other: ColumnOperand, /) -> ColumnExpr: ...
    def __mul__(self, other: ColumnOperand, /) -> ColumnExpr: ...
    def __rmul__(self, other: ColumnOperand, /) -> ColumnExpr: ...
    def __truediv__(self, other: ColumnOperand, /) -> ColumnExpr: ...
    def __rtruediv__(self, other: ColumnOperand, /) -> ColumnExpr: ...
    def __neg__(self) -> ColumnExpr: ...
    def __and__(self, other: ColumnOperand, /) -> ColumnExpr: ...
    def __rand__(self, other: ColumnOperand, /) -> ColumnExpr: ...
    def __or__(self, other: ColumnOperand, /) -> ColumnExpr: ...
    def __ror__(self, other: ColumnOperand, /) -> ColumnExpr: ...
    def __invert__(self) -> ColumnExpr: ...


@dataclass(frozen=True, slots=True, eq=False)
class ArrayExpr(Expr[object]):
    def __eq__(self, other: ArrayOperand, /) -> ArrayExpr: ...
    def __ne__(self, other: ArrayOperand, /) -> ArrayExpr: ...
    def __lt__(self, other: ArrayOperand, /) -> ArrayExpr: ...
    def __le__(self, other: ArrayOperand, /) -> ArrayExpr: ...
    def __gt__(self, other: ArrayOperand, /) -> ArrayExpr: ...
    def __ge__(self, other: ArrayOperand, /) -> ArrayExpr: ...
    def __add__(self, other: ArrayOperand, /) -> ArrayExpr: ...
    def __radd__(self, other: ArrayOperand, /) -> ArrayExpr: ...
    def __sub__(self, other: ArrayOperand, /) -> ArrayExpr: ...
    def __rsub__(self, other: ArrayOperand, /) -> ArrayExpr: ...
    def __mul__(self, other: ArrayOperand, /) -> ArrayExpr: ...
    def __rmul__(self, other: ArrayOperand, /) -> ArrayExpr: ...
    def __truediv__(self, other: ArrayOperand, /) -> ArrayExpr: ...
    def __rtruediv__(self, other: ArrayOperand, /) -> ArrayExpr: ...
    def __neg__(self) -> ArrayExpr: ...
    def __and__(self, other: ArrayOperand, /) -> ArrayExpr: ...
    def __rand__(self, other: ArrayOperand, /) -> ArrayExpr: ...
    def __or__(self, other: ArrayOperand, /) -> ArrayExpr: ...
    def __ror__(self, other: ArrayOperand, /) -> ArrayExpr: ...
    def __invert__(self) -> ArrayExpr: ...


@dataclass(frozen=True, slots=True, eq=False)
class TableExpr(Expr[object]):
    def __eq__(self, other: object, /) -> Never: ...
    def __ne__(self, other: object, /) -> Never: ...
    def __lt__(self, other: object, /) -> Never: ...
    def __le__(self, other: object, /) -> Never: ...
    def __gt__(self, other: object, /) -> Never: ...
    def __ge__(self, other: object, /) -> Never: ...
    def __add__(self, other: object, /) -> Never: ...
    def __radd__(self, other: object, /) -> Never: ...
    def __sub__(self, other: object, /) -> Never: ...
    def __rsub__(self, other: object, /) -> Never: ...
    def __mul__(self, other: object, /) -> Never: ...
    def __rmul__(self, other: object, /) -> Never: ...
    def __truediv__(self, other: object, /) -> Never: ...
    def __rtruediv__(self, other: object, /) -> Never: ...
    def __neg__(self) -> Never: ...
    def __and__(self, other: object, /) -> Never: ...
    def __rand__(self, other: object, /) -> Never: ...
    def __or__(self, other: object, /) -> Never: ...
    def __ror__(self, other: object, /) -> Never: ...
    def __invert__(self) -> Never: ...
    def __getitem__(self, field: str, /) -> ColumnExpr: ...
    def with_columns(self, features: FeatureSet, /) -> TableExpr: ...


@dataclass(frozen=True, slots=True, eq=False)
class Parameter(Expr[T], Generic[T]):
    @property
    def name(self) -> str: ...
    @property
    def kind(self) -> BatchKind: ...


@dataclass(frozen=True, slots=True, eq=False, init=False)
class FeatureSet:
    def __init__(
        self,
        features: Sequence[tuple[str, ColumnExpr]] = (),
        /,
    ) -> None: ...
    @property
    def features(self) -> tuple[tuple[str, ColumnExpr], ...]: ...
    def with_feature(self, name: str, value: ColumnExpr, /) -> FeatureSet: ...


@dataclass(frozen=True, slots=True)
class AnalysisIssue:
    path: str
    code: str
    message: str


@dataclass(frozen=True, slots=True)
class AnalysisResult:
    mode: CompileMode
    program_fingerprint: str
    capability_session_id: str
    capability_revision: int
    issues: tuple[AnalysisIssue, ...]


@dataclass(frozen=True, slots=True, eq=False, init=False)
class Program:
    def __init__(
        self,
        name: str,
        /,
        *,
        inputs: Sequence[TableExpr | Parameter[object]] = (),
        outputs: Sequence[tuple[str, TableExpr | ArrayExpr]] = (),
    ) -> None: ...
    @property
    def name(self) -> str: ...
    @property
    def inputs(self) -> tuple[TableExpr | Parameter[object], ...]: ...
    @property
    def outputs(self) -> tuple[tuple[str, TableExpr | ArrayExpr], ...]: ...
    @property
    def fingerprint(self) -> str: ...
    def with_input(self, value: TableExpr | Parameter[object], /) -> Program: ...
    def output(self, name: str, value: TableExpr | ArrayExpr, /) -> Program: ...
    def analyze(self, runtime: Runtime, /, *, mode: CompileMode) -> AnalysisResult: ...
    def explain(self, runtime: Runtime, /, *, mode: CompileMode) -> str: ...
    def compile_batch(self, runtime: Runtime, /) -> BatchExecutionPlan: ...
    def compile_stream(
        self,
        runtime: Runtime,
        /,
        *,
        allowed_lateness_micros: int = 0,
        late_policy: LatePolicy = "error",
    ) -> StreamExecutionPlan: ...
```

`__bool__` always raises `TypeError` with the exact guidance:

```text
symbolic expressions have no truth value; use &, |, and ~ for symbolic boolean composition, or identical() for structural identity
```

`identical()` is the only public boolean structural comparison. `ColumnExpr`
operators accept only `ColumnExpr` or scalar literals and always return
`ColumnExpr`. `ArrayExpr` operators accept only `ArrayExpr`, an array-kind
`Parameter[ArrayExpr]`, or scalar literals and always return `ArrayExpr`.
Reflected arithmetic and boolean composition use the same domain as their
forward forms. `Parameter` itself does not define scalar operator dunders; an
array parameter is accepted only when an `ArrayExpr` dunder dispatches or at a
signature that names `Parameter[ArrayExpr]`, including `linalg.matmul`.
Scalars must be strict finite JSON scalars; Python `bool` and `int` retain
distinct identities, as do integer and float values and positive/negative
floating zero.

Cross-domain operands fail at declaration construction with `TypeError`. The
message templates are exact; `{operator}` is the Python token and `{type}` is
the operand's module-qualified type name without a representation or address:

```text
symbolic column operator {operator} requires ColumnExpr or a strict scalar literal; got {type}
symbolic array operator {operator} requires ArrayExpr, Parameter[ArrayExpr], or a strict scalar literal; got {type}
```

`TableExpr` exposes only table namespace/member operations. Every scalar
arithmetic, comparison, or boolean-composition dunder shown in its signature
raises `TypeError` with:

```text
symbolic table expressions do not support scalar operator {operator}; select a column or use table/window operations
```

For these messages `{operator}` is one of `==`, `!=`, `<`, `<=`, `>`, `>=`,
`+`, `-`, `*`, `/`, unary `-`, `&`, `|`, or `~`. A table's structural
identity remains available through `identical()`.

A namespace call that receives an expression from the wrong domain raises
`TypeError` before node construction with this exact template:

```text
calc_flow.symbolic.{function}.{parameter}: expected {expected}; got {type}
```

`{function}` is the public dotted namespace function, `{parameter}` is its
signature parameter, `{expected}` is the exact annotation spelling from the
signature, and `{type}` uses the same module-qualified non-representational
spelling as the operator errors.

Every frozen/slotted dataclass whose stored fields directly or transitively
contain an `Expr` has generated equality disabled. The exact non-`Expr`
classes are `EventTimeBucket`, `CrossSectionGroup`, `FeatureSet`, and
`Program`; their `==`, `!=`, and hash use `object` identity and never compare
their fields. Thus equal-looking but separately constructed values compare
unequal, while `value == value` is true, and no comparison can invoke
`Expr.__bool__`. Expression nodes themselves retain the domain-specific rules
above, remain unhashable, and use `identical()` for structural identity.

`Program.inputs` and `Program.outputs` preserve declaration order and reject
duplicates. Every referenced table input/parameter must occur exactly once in
`inputs`; there is no implicit input discovery. `compile_*` requires an
explicit `Runtime` so one session/revision snapshot is unambiguous. Compilation
never accepts data and never starts a runner.

The constructors are:

```python
def table_input(
    name: str,
    /,
    *,
    schema: Sequence[Field],
    entity_by: Sequence[str] = (),
    event_time: str | None = None,
    sequence_by: Sequence[str] = (),
) -> TableExpr: ...

@overload
def parameter(
    name: str,
    /,
    *,
    kind: Literal["table"],
    schema: Sequence[Field],
    backend: None = None,
    dtype: None = None,
    shape: Sequence[int] = (),
    mutability: Literal["static"] = "static",
) -> Parameter[TableExpr]: ...

@overload
def parameter(
    name: str,
    /,
    *,
    kind: Literal["array"],
    schema: Sequence[Field] = (),
    backend: str,
    dtype: str,
    shape: Sequence[int],
    mutability: Literal["static"] = "static",
) -> Parameter[ArrayExpr]: ...
```

The two overloads are exclusive: a table parameter rejects backend/dtype/shape
and an array parameter rejects schema. The only accepted mutability spelling is
`"static"`. Dynamic parameters are not silently downgraded to static.

### 2.4 Namespace call signatures

`row` exposes only row-local expressions:

```text
row.where(condition: ColumnExpr, when_true: ColumnOperand, when_false: ColumnOperand, /) -> ColumnExpr
row.coalesce(*values: ColumnOperand) -> ColumnExpr
row.log(value: ColumnOperand, /) -> ColumnExpr
row.exp(value: ColumnOperand, /) -> ColumnExpr
row.sqrt(value: ColumnOperand, /) -> ColumnExpr
row.abs(value: ColumnOperand, /) -> ColumnExpr
row.clip(value: ColumnOperand, /, *, lower: ScalarLiteral, upper: ScalarLiteral) -> ColumnExpr
row.cast(value: ColumnOperand, data_type: str, /) -> ColumnExpr
```

`ts` exposes row-preserving temporal primitives. `window` is keyword-only and
means `RowFrame | DurationFrame`; it never means an event-time aggregate.

```text
ts.lag(value: ColumnExpr, /, *, periods: int = 1) -> ColumnExpr
ts.delta(value: ColumnExpr, /, *, periods: int = 1) -> ColumnExpr
ts.count(value: ColumnExpr, /, *, window: RowFrame | DurationFrame, min_periods: int = 1) -> ColumnExpr
ts.sum(value: ColumnExpr, /, *, window: RowFrame | DurationFrame, min_periods: int = 1) -> ColumnExpr
ts.mean(value: ColumnExpr, /, *, window: RowFrame | DurationFrame, min_periods: int = 1) -> ColumnExpr
ts.min(value: ColumnExpr, /, *, window: RowFrame | DurationFrame, min_periods: int = 1) -> ColumnExpr
ts.max(value: ColumnExpr, /, *, window: RowFrame | DurationFrame, min_periods: int = 1) -> ColumnExpr
ts.variance(value: ColumnExpr, /, *, window: RowFrame | DurationFrame, min_periods: int = 1, ddof: Literal[0, 1] = 1) -> ColumnExpr
ts.stddev(value: ColumnExpr, /, *, window: RowFrame | DurationFrame, min_periods: int = 1, ddof: Literal[0, 1] = 1) -> ColumnExpr
ts.covariance(left: ColumnExpr, right: ColumnExpr, /, *, window: RowFrame | DurationFrame, min_periods: int = 1, ddof: Literal[0, 1] = 1) -> ColumnExpr
ts.correlation(left: ColumnExpr, right: ColumnExpr, /, *, window: RowFrame | DurationFrame, min_periods: int = 1, ddof: Literal[0, 1] = 1) -> ColumnExpr
```

`cs` requires an explicit complete-group declaration:

```text
cs.rank(value: ColumnExpr, /, *, group: CrossSectionGroup, direction: Literal["ascending", "descending"] = "ascending", tie_method: Literal["average", "min", "max"] = "average", null_placement: Literal["exclude", "first", "last"] = "exclude", min_samples: int = 1) -> ColumnExpr
cs.percentile(value: ColumnExpr, /, *, group: CrossSectionGroup, direction: Literal["ascending", "descending"] = "ascending", tie_method: Literal["average", "min", "max"] = "average", null_placement: Literal["exclude", "first", "last"] = "exclude", min_samples: int = 1) -> ColumnExpr
cs.demean(value: ColumnExpr, /, *, group: CrossSectionGroup, min_samples: int = 1) -> ColumnExpr
cs.zscore(value: ColumnExpr, /, *, group: CrossSectionGroup, min_samples: int = 1, ddof: Literal[0, 1] = 0) -> ColumnExpr
cs.winsorize(value: ColumnExpr, /, *, group: CrossSectionGroup, lower: float, upper: float, min_samples: int = 1) -> ColumnExpr
```

`table`, `linalg`, and `window` expose the initial bridge/cardinality-changing
boundary without sharing names with rolling operations:

```text
table.project(value: TableExpr, columns: Sequence[str], /) -> TableExpr
table.filter(value: TableExpr, predicate: ColumnExpr, /) -> TableExpr
table.attach_columns(value: TableExpr, array: ArrayExpr, /, *, names: Sequence[str]) -> TableExpr
linalg.from_columns(value: TableExpr, /, *, columns: Sequence[str], backend: str) -> ArrayExpr
linalg.matmul(left: ArrayExpr, right: ArrayExpr | Parameter[ArrayExpr], /) -> ArrayExpr
window.tumbling(value: TableExpr, /, *, event_time: str, size_micros: int, group_by: Sequence[str] = ()) -> TableExpr
window.hopping(value: TableExpr, /, *, event_time: str, size_micros: int, slide_micros: int, group_by: Sequence[str] = ()) -> TableExpr
```

All default values above are semantic defaults from D1–D13. Lowering writes
them explicitly into stateful project variants; omission is never used to
select a different meaning.

### 2.5 Runner static inputs

The Python constructor is additive and keyword-only:

```python
class StreamingRunner:
    def __init__(
        self,
        plan: StreamExecutionPlan,
        sources: Mapping[str, SourceBinding] | None = None,
        sinks: Mapping[str, Sequence[SinkBinding]] | None = None,
        checkpoints: ManagedCheckpointRuntime | None = None,
        *,
        config: StreamRuntimeConfig | None = None,
        static_inputs: Mapping[str, Batch] | None = None,
    ) -> None: ...
```

`None` is normalized to an empty mapping. The mapping is copied immediately;
keys must be `str` and values must be `Batch`. Connector-backed project plans
continue to own sources, sinks, checkpoints, and config, but they do not own
static payload values, so `static_inputs` remains allowed and required when the
plan declares them. `StreamExecutionPlan.static_input_ids` returns the sorted
exact names as `tuple[str, ...]`; `source_binding_ids` excludes those names.

## 3. Rust public surface

### 3.1 Crate-root exports

The eventual implementation re-exports these additive names from
`calc_flow`:

```rust
CrossSectionGroupingSpec, CrossSectionOperator, CrossSectionOutputSpec,
CrossSectionSpec, CrossSectionValuePolicy, LateErrorScope, LatePolicySpec,
NullPlacement, RankTieMethod, RollingFrameSpec, RollingOperator,
RollingOutputSpec, RollingSpec, RollingValuePolicy, SortDirection,
StaticInputDigest, StaticInputSpec, StaticMutability,
STATIC_INPUT_DIGEST_VERSION
```

Existing exports and project format version remain unchanged.

### 3.2 Exact specification types

All serialized types derive `Clone`, `Debug`, `Eq`/`PartialEq`, `Serialize`,
`Deserialize`, and `JsonSchema` where the contained numeric types permit it.
Every struct/variant uses `deny_unknown_fields`. Enum values use snake case.

```rust
pub const ROLLING_CONFIGURATION_VERSION: u32 = 1;
pub const ROLLING_STATE_LAYOUT_VERSION: u32 = 1;
pub const CROSS_SECTION_CONFIGURATION_VERSION: u32 = 1;
pub const CROSS_SECTION_STATE_LAYOUT_VERSION: u32 = 1;
pub const STATIC_INPUT_DIGEST_VERSION: &str = "calc_flow.static_input.digest.v1";

#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize, Deserialize, JsonSchema)]
#[serde(rename_all = "snake_case")]
pub enum LateErrorScope { Envelope }

#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize, Deserialize, JsonSchema)]
#[serde(tag = "kind", rename_all = "snake_case", deny_unknown_fields)]
pub enum LatePolicySpec {
    Error { scope: LateErrorScope },
    Drop { metrics_version: u32 },
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize, Deserialize, JsonSchema)]
#[serde(tag = "kind", rename_all = "snake_case", deny_unknown_fields)]
pub enum RollingFrameSpec {
    Rows { size: u64 },
    Duration { micros: u64 },
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize, Deserialize, JsonSchema)]
#[serde(rename_all = "snake_case")]
pub enum RollingValuePolicy { StatefulNumericV1 }

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize, JsonSchema)]
#[serde(tag = "kind", rename_all = "snake_case", deny_unknown_fields)]
pub enum RollingOutputSpec {
    Lag { primitive_version: u32, input: String, output: String, periods: u64 },
    Delta { primitive_version: u32, input: String, output: String, periods: u64 },
    Count { primitive_version: u32, input: String, output: String, frame: RollingFrameSpec, min_periods: u64 },
    Sum { primitive_version: u32, input: String, output: String, frame: RollingFrameSpec, min_periods: u64 },
    Mean { primitive_version: u32, input: String, output: String, frame: RollingFrameSpec, min_periods: u64 },
    Min { primitive_version: u32, input: String, output: String, frame: RollingFrameSpec, min_periods: u64 },
    Max { primitive_version: u32, input: String, output: String, frame: RollingFrameSpec, min_periods: u64 },
    Variance { primitive_version: u32, input: String, output: String, frame: RollingFrameSpec, min_periods: u64, ddof: u8 },
    Stddev { primitive_version: u32, input: String, output: String, frame: RollingFrameSpec, min_periods: u64, ddof: u8 },
    Covariance { primitive_version: u32, left: String, right: String, output: String, frame: RollingFrameSpec, min_periods: u64, ddof: u8 },
    Correlation { primitive_version: u32, left: String, right: String, output: String, frame: RollingFrameSpec, min_periods: u64, ddof: u8 },
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct RollingSpec {
    pub configuration_version: u32,
    pub state_layout_version: u32,
    pub partition_by: Vec<String>,
    pub event_time: String,
    pub sequence_by: Vec<String>,
    pub outputs: Vec<RollingOutputSpec>,
    pub allowed_lateness_micros: u64,
    pub late_policy: LatePolicySpec,
    pub value_policy: RollingValuePolicy,
}
```

`LatePolicySpec::Drop.metrics_version` must equal `1`. That version means the
three D7 metrics `dropped_rows`, `affected_envelopes`, and
`max_lateness_micros`. It is explicit so an implementation cannot omit one
metric while still claiming `drop` support. `LatePolicySpec::Error` requires
`scope: "envelope"`; there is no row-scoped error spelling.

Metrics version `1` has the following exact transaction. The operator samples
its aggregate input watermark `W` once immediately before classifying an input
envelope, and every row in that envelope uses the same value. An undefined `W`
means no row is late and the envelope does not change late metrics. For each
dropped row with normalized event time `t`:

```text
lateness_micros = checked_u64(i128(W) - i128(t))
```

Allowed lateness `L` participates only in the close test (`t + L <= W`, or the
corresponding cross-section group test); it is not subtracted from this metric.
A bucketed cross section tests `bucket_end + L <= W` but still measures `W - t`
from the dropped row's own event time. The widened subtraction and conversion
are checked.

Each operator starts with `dropped_rows = 0`, `affected_envelopes = 0`, and
`max_lateness_micros = None`. `None` means no dropped row has been observed and
is distinct from `Some(0)`. For an envelope with `n > 0` dropped rows, one
atomic metric delta performs checked addition of `n` to `dropped_rows`, checked
addition of exactly one to `affected_envelopes`, and takes the maximum of the
prior value and every envelope `W - t`. Any arithmetic/conversion failure
rejects the envelope without changing metrics, calculation state, or output.

Job `dropped_rows` and `affected_envelopes` are checked sums of the
per-operator values; job `max_lateness_micros` is the maximum of all present
operator values and remains `None` only when every operator value is `None`.
These are operator-input decisions, so a branched row may contribute once per
operator that independently drops it.

The three per-operator values are checkpointed semantic state. Restore installs
them before replay and re-derives job totals. Metrics produced after the
selected epoch by a failed attempt are discarded; replay applies each
post-epoch envelope once. Terminal checkpoints retain the final values and
terminal recovery does not change them.

The required nonzero-lateness vector is exact: with `L = 5`, `W = 120`, and
one envelope at event times `(115, 114, 117)`, the first two rows are dropped
with lateness `(5, 6)`, producing `(dropped_rows, affected_envelopes,
max_lateness_micros) = (2, 1, Some(6))`. After checkpoint/restore, a second
envelope at `W = 123` with times `(118, 120)` produces `(3, 2, Some(6))`.
Crashing before that second result is checkpointed and replaying the envelope
must produce the same tuple, not double count it.

#### Public late-metric projection

Metrics version `1` reuses the existing public continuous-runtime status
fields; it does not add or rename an `OperatorStatus` field. The semantic-to-
public mapping is exact:

| D7 semantic name      | Rust `OperatorStatus` field | Python operator-status key | Public value                       |
| --------------------- | --------------------------- | -------------------------- | ---------------------------------- |
| `dropped_rows`        | `late_rows`                 | `late_rows`                | non-negative `u64` / Python `int`  |
| `affected_envelopes`  | `late_affected_batches`     | `late_affected_batches`    | non-negative `u64` / Python `int`  |
| `max_lateness_micros` | `max_lateness`              | `max_lateness_micros`      | `Option<Duration>` / `int \| None` |

The exact existing Rust subset remains:

```rust
pub struct OperatorStatus {
    // Existing fields before these values are unchanged.
    pub late_rows: u64,
    pub late_affected_batches: u64,
    pub max_lateness: Option<Duration>,
    // Existing fields after these values are unchanged.
}
```

The exact Python projection under
`JobStatus["operators"][operator_id]` remains:

```python
class OperatorStatus(TypedDict):
    # Existing keys before these values are unchanged.
    late_rows: int
    late_affected_batches: int
    max_lateness_micros: int | None
    # Existing keys after these values are unchanged.
```

For rolling and cross-section operators, one runtime input data batch is the
atomic input envelope from D7, so `late_affected_batches` is exactly
`affected_envelopes`; it is not an output-batch count. `late_rows` counts
operator-input row drop decisions. The existing window operator retains its
existing row-window-assignment meaning, and no legacy metric is reinterpreted
for that operator kind.

D7 job totals are checked, re-derived aggregates of the values under the
public `operators` map. This gate adds no Rust `JobStatus::late_rows`,
`JobStatus::late_affected_batches`, or `JobStatus::max_lateness` fields and no
Python top-level `late_rows`, `late_affected_batches`, or
`max_lateness_micros` keys. Consumers that need a job-wide view use checked
sums of the first two per-operator keys and the maximum of present third-key
values, exactly as D7 specifies. The totals are not checkpointed or serialized
separately; restore reinstalls each operator's values, and status projection
exposes those values after the internal totals have been re-derived. This
choice preserves the existing exact Rust `JobStatus` shape and Python status
key set.

Cross-section types are:

```rust
#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize, Deserialize, JsonSchema)]
#[serde(rename_all = "snake_case")]
pub enum SortDirection { Ascending, Descending }

#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize, Deserialize, JsonSchema)]
#[serde(rename_all = "snake_case")]
pub enum RankTieMethod { Average, Min, Max }

#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize, Deserialize, JsonSchema)]
#[serde(rename_all = "snake_case")]
pub enum NullPlacement { Exclude, First, Last }

#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize, Deserialize, JsonSchema)]
#[serde(rename_all = "snake_case")]
pub enum CrossSectionValuePolicy { NanExcludePreserveV1 }

#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize, Deserialize, JsonSchema)]
#[serde(tag = "kind", rename_all = "snake_case", deny_unknown_fields)]
pub enum CrossSectionGroupingSpec {
    ExactTime,
    FixedBucket { width_micros: u64 },
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize, JsonSchema)]
#[serde(tag = "kind", rename_all = "snake_case", deny_unknown_fields)]
pub enum CrossSectionOutputSpec {
    Rank { primitive_version: u32, input: String, output: String, direction: SortDirection, tie_method: RankTieMethod, null_placement: NullPlacement, min_samples: u64 },
    Percentile { primitive_version: u32, input: String, output: String, direction: SortDirection, tie_method: RankTieMethod, null_placement: NullPlacement, min_samples: u64 },
    Demean { primitive_version: u32, input: String, output: String, min_samples: u64 },
    Zscore { primitive_version: u32, input: String, output: String, min_samples: u64, ddof: u8 },
    Winsorize { primitive_version: u32, input: String, output: String, min_samples: u64, lower: f64, upper: f64 },
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct CrossSectionSpec {
    pub configuration_version: u32,
    pub state_layout_version: u32,
    pub event_time: String,
    pub entity_by: Vec<String>,
    pub partition_by: Vec<String>,
    pub sequence_by: Vec<String>,
    pub grouping: CrossSectionGroupingSpec,
    pub outputs: Vec<CrossSectionOutputSpec>,
    pub allowed_lateness_micros: u64,
    pub late_policy: LatePolicySpec,
    pub value_policy: CrossSectionValuePolicy,
}
```

`CrossSectionOutputSpec` is `PartialEq`, not `Eq`, only because winsor bounds
are floats. Deserialization and constructors reject non-finite bounds before a
public value is returned.

The operator and plan signatures are:

```rust
impl RollingSpec {
    pub fn validate(&self, input_schema: &Schema) -> Result<SchemaRef>;
}

impl RollingOperator {
    pub fn new(name: &str, input_schema: SchemaRef, spec: RollingSpec) -> Result<Self>;
    pub fn spec(&self) -> &RollingSpec;
}

impl CrossSectionSpec {
    pub fn validate(&self, input_schema: &Schema) -> Result<SchemaRef>;
}

impl CrossSectionOperator {
    pub fn new(name: &str, input_schema: SchemaRef, spec: CrossSectionSpec) -> Result<Self>;
    pub fn spec(&self) -> &CrossSectionSpec;
}

impl StreamExecutionPlan {
    pub fn static_input_ids(&self) -> Vec<&str>;
    pub fn static_inputs(&self) -> &BTreeMap<String, StaticInputSpec>;
}

impl StreamingRunner {
    #[must_use = "the runner owns the supplied static input handles"]
    pub fn with_static_inputs(
        self,
        inputs: BTreeMap<String, Batch>,
    ) -> Result<Self>;
}
```

`StreamingRunner::new` remains source compatible. The binding constructs it,
then calls `with_static_inputs`. `Batch` is clone-by-owned-handle, so the runner
gets an immutable engine-owned handle without mutating caller data.

### 3.3 Static declarations

Project-v3 and compiled plans use:

```rust
#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize, Deserialize, JsonSchema)]
#[serde(rename_all = "snake_case")]
pub enum StaticMutability { Static }

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize, JsonSchema)]
#[serde(tag = "kind", rename_all = "snake_case", deny_unknown_fields)]
pub enum StaticInputSpec {
    Table {
        name: String,
        mutability: StaticMutability,
        schema: Vec<ArrowFieldSpec>,
    },
    Array {
        name: String,
        mutability: StaticMutability,
        backend: String,
        dtype: String,
        shape: Vec<u64>,
    },
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct StaticInputDigest {
    pub digest_version: String,
    pub sha256: String,
}
```

`StaticInputSpec` deliberately does not derive `Eq`: its table variant contains
the baseline `ArrowFieldSpec`, which derives `PartialEq` but not `Eq`.
`StaticInputDigest` remains `Eq`.

`ProjectSpec` gains
`#[serde(default, skip_serializing_if = "Vec::is_empty")] pub static_inputs:
Vec<StaticInputSpec>`. Names are unique and must name graph external input
bindings. A binding named by `static_inputs` is not a source binding. Empty
lists are omitted so old project-v3 canonical JSON is byte-for-byte unchanged.

## 4. Python and Rust error surface

No new top-level exception hierarchy is added. Failures map to the existing
surface:

- wrong Python host types raise `TypeError` before declaration construction;
- invalid direct declaration values raise `ValueError` with the declaration
  path first;
- analysis, capability, or lowering rejection raises `CompileError`;
- strict project decode/validation raises `ConfigError`;
- duplicate row identity, checked overflow, or malformed runtime data becomes
  `ExecutionError` in batch and `StreamingRuntimeError` in stream;
- provider digest/canonicalization failure becomes `ProviderError`; and
- schema/config/layout/static-digest recovery mismatch becomes
  `CheckpointError` before a source opens.

Every engine-originated message begins with the stable path followed by a
colon. `CompileError` uses:

```text
{path}: {code}: {message}
```

The initial symbolic codes are `duplicate_name`, `invalid_literal`,
`unresolved_type`, `unsupported_type`, `unsupported_mode`,
`capability_mismatch`, `unbounded_state`, `ordering_required`,
`schema_mismatch`, and `unknown_primitive_version`.

The late-error runtime message is exactly shaped as:

```text
{output_path}: late_row: envelope rejected at row_index={row_index}; event_time_micros={t}, closed_at_watermark_micros={watermark}
```

`row_index` is the zero-based logical row position in the rejected input
envelope before any internal reorder. Entity-key and sequence-key values are
never formatted. Duplicate-identity and malformed-row diagnostics use the same
non-payload row index and may include the permitted event-time coordinate, but
must not include entity/sequence values. Before this error is returned, the
owning rolling/cross-section operator must have emitted no batch, recorded no
metric, and installed no state from any row in that input envelope. The
implementation therefore validates/classifies the complete envelope into an
operator-owned staged delta before the first state mutation or
`StreamCollector.emit` call. The serialized `late_policy` shape below makes
this envelope scope an explicit part of the configuration fingerprint.

Static recovery mismatch uses `CalcFlowError::CheckpointMismatch` and Python
`CheckpointError` with:

```text
static_inputs.{name}.digest: checkpoint digest {stored} does not match prepared digest {prepared} for calc_flow.static_input.digest.v1
```

Missing and extra names use `static_inputs.{name}`. Only names, versions, and
digests are exposed.

## 5. Strict project-v3 shapes

### 5.1 General rules

`OperatorSpec` gains exactly:

```rust
Rolling { spec: RollingSpec },
CrossSection { spec: CrossSectionSpec },
```

The serde discriminator remains the existing `operator.kind`. The spellings
are `"rolling"` and `"cross_section"`. They do not overload `expression`,
`sql`, `window`, or `external`.

For both operators:

- `input_ports` is exactly one required table port named `input` with a
  non-empty exact schema;
- `output_ports` is exactly one required table port named `output` with the
  exact derived schema;
- output fields are the input fields followed by declared outputs in order;
- all semantic fields shown below are required, even when equal to a default;
- unknown fields, empty ordered keys, duplicate output names, collisions,
  unknown versions, and non-applicable fields are rejected; and
- `position` remains the only non-semantic node field and may be omitted.

The symbolic lowerer always writes
`configuration_version: 1`, `state_layout_version: 1`, and
`primitive_version: 1`. There are no string/numeric version aliases.

### 5.2 Rolling

A canonical rolling node is:

```json
{
  "id": "rolling_features",
  "operator": {
    "kind": "rolling",
    "spec": {
      "configuration_version": 1,
      "state_layout_version": 1,
      "partition_by": ["symbol"],
      "event_time": "ts",
      "sequence_by": ["sequence"],
      "outputs": [
        {
          "kind": "mean",
          "primitive_version": 1,
          "input": "return_1",
          "output": "momentum_20",
          "frame": {"kind": "rows", "size": 20},
          "min_periods": 20
        },
        {
          "kind": "correlation",
          "primitive_version": 1,
          "left": "return_1",
          "right": "volume",
          "output": "return_volume_corr",
          "frame": {"kind": "duration", "micros": 60000000},
          "min_periods": 2,
          "ddof": 1
        }
      ],
      "allowed_lateness_micros": 0,
      "late_policy": {"kind": "error", "scope": "envelope"},
      "value_policy": "stateful_numeric_v1"
    }
  },
  "input_ports": [
    {
      "name": "input",
      "kind": "table",
      "required": true,
      "schema": [
        {"name": "ts", "data_type": "timestamp[us, UTC]", "nullable": false},
        {"name": "symbol", "data_type": "string", "nullable": false},
        {"name": "sequence", "data_type": "uint64", "nullable": false},
        {"name": "return_1", "data_type": "float64", "nullable": true},
        {"name": "volume", "data_type": "float64", "nullable": true}
      ]
    }
  ],
  "output_ports": [
    {
      "name": "output",
      "kind": "table",
      "required": true,
      "schema": [
        {"name": "ts", "data_type": "timestamp[us, UTC]", "nullable": false},
        {"name": "symbol", "data_type": "string", "nullable": false},
        {"name": "sequence", "data_type": "uint64", "nullable": false},
        {"name": "return_1", "data_type": "float64", "nullable": true},
        {"name": "volume", "data_type": "float64", "nullable": true},
        {"name": "momentum_20", "data_type": "float64", "nullable": true},
        {"name": "return_volume_corr", "data_type": "float64", "nullable": true}
      ]
    }
  ]
}
```

The exact per-kind required fields are those in `RollingOutputSpec`; a field
from another variant is unknown and rejected. In particular lag/delta cannot
carry `frame`, `min_periods`, or `ddof`, and non-statistical aggregates cannot
carry `ddof`. `partition_by`, `sequence_by`, and `outputs` are semantic arrays
and are never sorted by the serializer.

`value_policy: "stateful_numeric_v1"` means exactly D3: aggregate samples
exclude null and NaN, lag/delta preserve either missing value at the current or
referenced position, infinities remain numeric, and minimum counts count valid
samples rather than rows. No other rolling value-policy spelling is accepted.

### 5.3 Cross section

A canonical exact-time node is:

```json
{
  "id": "cross_section_features",
  "operator": {
    "kind": "cross_section",
    "spec": {
      "configuration_version": 1,
      "state_layout_version": 1,
      "event_time": "ts",
      "entity_by": ["symbol"],
      "partition_by": ["industry"],
      "sequence_by": ["sequence"],
      "grouping": {"kind": "exact_time"},
      "outputs": [
        {
          "kind": "rank",
          "primitive_version": 1,
          "input": "momentum_20",
          "output": "momentum_rank",
          "direction": "ascending",
          "tie_method": "average",
          "null_placement": "exclude",
          "min_samples": 1
        },
        {
          "kind": "zscore",
          "primitive_version": 1,
          "input": "momentum_20",
          "output": "alpha",
          "min_samples": 1,
          "ddof": 0
        }
      ],
      "allowed_lateness_micros": 0,
      "late_policy": {"kind": "error", "scope": "envelope"},
      "value_policy": "nan_exclude_preserve_v1"
    }
  },
  "input_ports": [
    {
      "name": "input",
      "kind": "table",
      "required": true,
      "schema": [
        {"name": "ts", "data_type": "timestamp[us, UTC]", "nullable": false},
        {"name": "symbol", "data_type": "string", "nullable": false},
        {"name": "industry", "data_type": "string", "nullable": true},
        {"name": "sequence", "data_type": "uint64", "nullable": false},
        {"name": "momentum_20", "data_type": "float64", "nullable": true}
      ]
    }
  ],
  "output_ports": [
    {
      "name": "output",
      "kind": "table",
      "required": true,
      "schema": [
        {"name": "ts", "data_type": "timestamp[us, UTC]", "nullable": false},
        {"name": "symbol", "data_type": "string", "nullable": false},
        {"name": "industry", "data_type": "string", "nullable": true},
        {"name": "sequence", "data_type": "uint64", "nullable": false},
        {"name": "momentum_20", "data_type": "float64", "nullable": true},
        {"name": "momentum_rank", "data_type": "float64", "nullable": true},
        {"name": "alpha", "data_type": "float64", "nullable": true}
      ]
    }
  ]
}
```

Bucket grouping changes only the grouping value:

```json
{"kind": "fixed_bucket", "width_micros": 60000000}
```

Rank/percentile carry direction, tie method, null placement, and minimum
samples explicitly. Demean, z-score, and winsorize reject those ordering
fields. Z-score alone carries `ddof`; winsorize alone carries `lower` and
`upper`. The lower/upper JSON values must be finite and satisfy D6.

`value_policy: "nan_exclude_preserve_v1"` means exactly D3/D6: NaN is
excluded from every order/statistic sample and remains NaN at its own row,
infinity is numeric, and null handling comes only from the per-output
`null_placement` field or the fixed non-ordering null-preservation rule. No
other cross-section value-policy spelling is accepted.

### 5.4 Static declaration JSON

Static inputs are a top-level project array because they classify graph input
bindings rather than operator options:

```json
{
  "static_inputs": [
    {
      "kind": "array",
      "name": "weights",
      "mutability": "static",
      "backend": "numpy",
      "dtype": "float64",
      "shape": [3, 1]
    }
  ]
}
```

A table declaration instead has exactly `kind`, `name`, `mutability`, and
`schema`. An array declaration has exactly the six fields shown. Payloads and
digests never appear in a project document. Static declaration order is
semantic in a `Program` fingerprint; compiled plan lookup uses a name-sorted
map after duplicate rejection.

### 5.5 Omission and default rules

The strict parser has no defaults inside `rolling` or `cross_section` specs.
Python constructor defaults are normalized and serialized as values:

- `allowed_lateness_micros` is `0`;
- `late_policy` is `{"kind":"error","scope":"envelope"}`;
- `min_periods` and `min_samples` are `1`;
- rolling statistical `ddof` is `1`;
- cross-section z-score `ddof` is `0`;
- direction is `ascending`, tie method is `average`, and null placement is
  `exclude`;
- all primitive/configuration/layout versions are `1`; and
- drop metrics version is `1`.

The non-default drop shape is exactly
`{"kind":"drop","metrics_version":1}`. It has no `scope` field. The error
shape has no `metrics_version` field.

Omitting any such field from a new variant is a project validation error. Old
operator variants retain their current omission/default rules unchanged.

## 6. Capability schema v2

### 6.1 Python data classes

Schema v2 keeps existing class names where their meaning remains sound and
adds the following exact vocabulary and nesting:

```python
type ExecutionMode = Literal["batch", "stream"]
type OutputFinality = Literal[
    "per_row_final", "group_final_append_only", "unproven"
]
type CheckpointSupport = Literal[
    "stateless", "checkpointed_stateful", "unproven"
]


@dataclass(frozen=True, slots=True)
class CapabilityRule:
    name: str
    version: str


@dataclass(frozen=True, slots=True)
class ProviderArrayRules:
    supported_dtypes: tuple[str, ...]
    safe_dtype_rule: CapabilityRule
    shape_rules: tuple[CapabilityRule, ...]


@dataclass(frozen=True, slots=True)
class OperatorCapability:
    kind: str
    version: str
    input_ports: tuple[ProviderPort, ...]
    output_ports: tuple[ProviderPort, ...]
    modes: tuple[ExecutionMode, ...]
    finality: OutputFinality
    requires_datafusion: bool
    stateful: bool
    microbatch_invariant: bool
    requires_watermark: bool
    checkpoint_support: CheckpointSupport
    state_version: int | None
    deterministic: bool
    replay_safe: bool


@dataclass(frozen=True, slots=True)
class ProviderCapability:
    provider: str
    name: str
    version: str
    input_ports: tuple[ProviderPort, ...]
    output_ports: tuple[ProviderPort, ...]
    options_schema: ProviderOptionsSchema | None
    modes: tuple[ExecutionMode, ...]
    finality: OutputFinality
    stateful: bool
    microbatch_invariant: bool
    requires_watermark: bool
    checkpoint_support: CheckpointSupport
    state_version: int | None
    deterministic: bool
    replay_safe: bool
    supports_static_inputs: bool
    partition_contract: Literal["none", "row_axis_independent"]
    array_rules: ProviderArrayRules | None


@dataclass(frozen=True, slots=True)
class RuntimeCapabilities:
    schema_version: Literal[2]
    scope: RuntimeSessionScope
    package_version: str
    project_format_versions: tuple[int, ...]
    batch_kinds: tuple[BatchKind, ...]
    portable_arrow_types: tuple[str, ...]
    operators: tuple[OperatorCapability, ...]
    udfs: tuple[UdfCapability, ...]
    providers: tuple[ProviderCapability, ...]
    connectors: tuple[ConnectorCapability, ...]
```

`ProviderPort`, `ProviderOption`, `ProviderOptionsSchema`,
`RuntimeSessionScope`, `UdfCapability`, and connector classes keep their schema
v1 Python fields. `RuntimeSessionScope.kind` remains `"runtime_session"` in
Python.

For checkpoint support, `state_version` is a positive integer exactly when the
value is `checkpointed_stateful`; otherwise it must be `None`. A stateless
capability must set `stateful=False`. Checkpoint support `unproven` cannot be
selected for a checkpointed stream plan. Finality is required even for
batch-only entries so lowering does not infer it from the mode or kind.
Finality `unproven` is the only truthful conservative value for a batch-only
registration that supplied no finality contract. It is never stream safe:
`compile_stream()` rejects every selected operator/provider whose finality is
`unproven`, regardless of its other lifecycle fields.

`partition_contract="row_axis_independent"` is required for a provider path
selected by `compile_stream()`; `"none"` is the conservative batch-only
value. `ProviderArrayRules` is present only for array-capable providers. Rule names
are closed, versioned identifiers understood by the symbolic analyzer. Initial
rule names are `array_api_safe_dtype`, `elementwise_broadcast`,
`feature_axis_reduction`, and `table_matmul_static_rhs`, all at version `1`.
The exact supported dtype tuple plus the selected rule identities proves dtype
and shape support; unknown rule identities fail capability validation.

All tuples retain semantically declared port/mode order. Top-level operators
sort by `(kind, version)`, UDFs/providers by `(provider, name, version)`, and
connectors by their existing key. `supported_dtypes` and `shape_rules` are
sorted lexicographically by identity. Capability objects are defensively
copied and session scoped.

### 6.2 Canonical serialized shape

The canonical standalone JSON projection uses the existing camel-case
conversion and includes the data class's schema version:

```json
{
  "schemaVersion": 2,
    "scope": {
      "kind": "runtimeSession",
      "sessionId": "0d27e589-8d73-4ce5-bc89-0d48a39bd87d",
      "revision": 4
    },
    "packageVersion": "3.0.0",
    "projectFormatVersions": [3],
    "batchKinds": ["array", "table"],
    "portableArrowTypes": ["bool", "float32", "float64", "timestamp[us, UTC]"],
    "operators": [
      {
        "kind": "rolling",
        "version": "1",
        "inputPorts": [{"name": "input", "kind": "table", "required": true}],
        "outputPorts": [{"name": "output", "kind": "table", "required": true}],
        "modes": ["batch", "stream"],
        "finality": "per_row_final",
        "requiresDatafusion": false,
        "stateful": true,
        "microbatchInvariant": true,
        "requiresWatermark": true,
        "checkpointSupport": "checkpointed_stateful",
        "stateVersion": 1,
        "deterministic": true,
        "replaySafe": true
      }
    ],
    "udfs": [],
    "providers": [
      {
        "provider": "numpy",
        "name": "table_matmul",
        "version": "1",
        "inputPorts": [
          {"name": "table", "kind": "table", "required": true},
          {"name": "weights", "kind": "array", "required": true}
        ],
        "outputPorts": [{"name": "output", "kind": "array", "required": true}],
        "optionsSchema": null,
        "modes": ["batch", "stream"],
        "finality": "per_row_final",
        "stateful": false,
        "microbatchInvariant": true,
        "requiresWatermark": false,
        "checkpointSupport": "stateless",
        "stateVersion": null,
        "deterministic": true,
        "replaySafe": true,
        "supportsStaticInputs": true,
        "partitionContract": "row_axis_independent",
        "arrayRules": {
          "supportedDtypes": ["float32", "float64"],
          "safeDtypeRule": {"name": "array_api_safe_dtype", "version": "1"},
          "shapeRules": [{"name": "table_matmul_static_rhs", "version": "1"}]
        }
      }
    ],
    "connectors": []
}
```

JSON `stateVersion` is required and explicitly `null` when inapplicable.
`optionsSchema` and `arrayRules` are likewise required nullable fields. Unknown
fields are rejected. The Studio `CapabilitiesResponse` keeps its exact
existing three fields: outer `schemaVersion` is `2`, `runtime` is the object
above with only its nested `schemaVersion` removed, and `preview` keeps the
existing `PreviewCapabilitiesResponse` shape unchanged. Thus the REST envelope
contains one schema-version field and no separate REST-only lifecycle
vocabulary.

Existing provider registration calls are source compatible. Unless the
registration explicitly supplies proven stream lifecycle metadata, their v2
entry is batch-only with:

```text
modes=("batch",), finality="unproven", stateful=False, microbatch_invariant=False,
requires_watermark=False, checkpoint_support="stateless",
state_version=None, deterministic=False, replay_safe=False,
supports_static_inputs=False, partition_contract="none", array_rules=None
```

The conservative `False` values are intentional; omission never opts a
provider into stream execution. Built-in operators report only modes that are
actually implemented when the snapshot is taken.

## 7. Static-input canonical tagged bytes

### 7.1 Digest envelope

The digest version string is exactly
`calc_flow.static_input.digest.v1`. The digest is lowercase hexadecimal
SHA-256 over one byte string constructed below. It is not SHA-256 over JSON,
Arrow IPC, NumPy storage, device memory, or `BatchMetadata`.

All integers in the encoding are unsigned big-endian unless a scalar rule says
otherwise. `U64(x)` is exactly eight bytes. `BYTES(x)` is `U64(len(x)) || x`.
`TEXT(s)` is `BYTES(UTF-8(s))` with no Unicode normalization. Counts and
lengths are checked to fit `u64`.

```text
MAGIC = ASCII("calc_flow.static_input.digest.v1") || 0x00
DIGEST_INPUT = MAGIC || 0x01 || TEXT(input_name) || PAYLOAD
TABLE_PAYLOAD = 0x10 || U64(field_count) || FIELD* || U64(row_count) || CELL*
ARRAY_PAYLOAD = 0x11 || TEXT(backend) || TYPE || U64(rank) || U64(dim)*
                || U64(element_count) || CELL*
FIELD = 0x20 || TEXT(field_name) || TYPE || NULLABLE
NULLABLE = 0x00 | 0x01
CELL = 0x30                         # null
     | 0x31 || SCALAR               # non-null
```

Table cells are encoded in logical row-major order: row zero across fields in
schema order, then row one, and so on. Record-batch/chunk boundaries have no
encoding. Array cells are in logical C order independent of physical strides,
layout, host/device location, or chunking. The array provider must prove that
`element_count` equals the checked product of dimensions. Table `row_count`
and array shape are encoded even for empty values.

`BatchMetadata` is excluded. Name is included once as shown. The batch kind is
the `0x10`/`0x11` tag. Table schema or array backend/dtype/shape is therefore
part of the digest without duplicating it in an outer JSON wrapper. Arrow
schema/field metadata other than field name, logical type, and nullability is
excluded.

### 7.2 Type descriptors and scalar bytes

The initial type tags are:

```text
0x40 bool                 scalar: 0x00 or 0x01
0x41 int8                 scalar: 1-byte two's complement
0x42 int16                scalar: 2-byte two's-complement big-endian
0x43 int32                scalar: 4-byte two's-complement big-endian
0x44 int64                scalar: 8-byte two's-complement big-endian
0x45 uint8                scalar: 1 byte
0x46 uint16               scalar: 2-byte big-endian
0x47 uint32               scalar: 4-byte big-endian
0x48 uint64               scalar: 8-byte big-endian
0x49 float32              scalar: canonical IEEE-754 bits, big-endian
0x4a float64              scalar: canonical IEEE-754 bits, big-endian
0x4b string               scalar: TEXT(UTF-8 value)
0x4c large_string         scalar: TEXT(UTF-8 value)
0x4d date32               scalar: int32 rule
0x4e date64               scalar: int64 rule
0x4f time32[s]            scalar: int32 rule
0x50 time64[us]           scalar: int64 rule
0x51 timestamp[ms]        scalar: int64 rule
0x52 timestamp[us]        scalar: int64 rule
0x53 timestamp[us, UTC]   scalar: int64 UTC epoch microseconds
0x54 dictionary           descriptor: TYPE(index) || TYPE(value) || ordered-byte
```

Dictionary index types are restricted to integer tags and value types to
non-dictionary types in digest v1. A dictionary cell resolves its logical
value and uses the value type's scalar rule; dictionary IDs, dictionary order,
and physical indices are absent. The descriptor retains index type, value
type, and the Arrow ordered flag (`0x00` for false, `0x01` for true), so exact
schema remains part of identity
while equivalent dictionary layouts hash equally.

For `float32`, every NaN is encoded as bits `0x7fc00000`; for `float64`, every
NaN is encoded as `0x7ff8000000000000`. All other floating values use their
exact IEEE bits. This preserves positive/negative zero and infinity sign while
discarding NaN sign, payload, and signaling/quiet differences.

Table digest v1 accepts the listed Arrow types and dictionary wrappers above.
Array digest v1 accepts `bool`, the integer types, `float32`, and `float64`.
An unsupported type/dtype fails at `static_inputs.{name}.dtype` or the precise
table schema path; it is never hashed through a provider-specific fallback.

### 7.3 Provider boundary

An array provider claiming `supports_static_inputs=True` must expose a trusted
engine registration hook that returns the declared backend, dtype, shape,
optional logical null mask, and an iterator/stream of logical C-order scalar
values. That hook is not callable from project data and is not included in
capability serialization. The engine checks the returned descriptor against
`StaticInputSpec` before hashing. A provider lacking this hook reports
`supports_static_inputs=False` and cannot satisfy a static declaration.

Canonicalization completes before any source, operator, sink, or provider
lifecycle factory is opened. A digest callback may read the already supplied
immutable payload, but no execution callback is invoked.

## 8. Checkpoint evidence and lineage selection

`CheckpointManifestFields` and `CheckpointManifest` gain a name-sorted map:

```rust
pub static_inputs: BTreeMap<String, StaticInputDigest>
```

The serialized root field is `static_inputs`. It uses
`#[serde(default, skip_serializing_if = "BTreeMap::is_empty")]` so existing v3
manifests with no static inputs remain readable and retain their canonical
bytes/checksum. When non-empty, the field is included in the manifest state
checksum as the exact canonical JSON object
`{"operators":...,"sinks":...,"sources":...,"static_inputs":...}`. When the
map is empty, the checksum input remains the existing exact three-field object
`{"operators":...,"sinks":...,"sources":...}`. All object keys use canonical
key order.
Raw values are never placed in manifest inline metadata or segments.

`ManifestExpectation` gains:

```rust
pub static_inputs: &'a BTreeMap<String, StaticInputDigest>
```

The recovery/preflight order is frozen:

1. copy and validate the exact static input mapping;
2. acquire engine-owned immutable `Batch` handles;
3. compute all canonical digests;
4. open the `StateLineageKey` selected by only the existing pipeline name and
   semantic plan fingerprint;
5. if a manifest exists, compare the exact static name/version/digest map;
6. validate all remaining manifest/segment/state invariants;
7. install state and static handles; and only then
8. open sources and operator/provider/sink lifecycles.

Static digests MUST NOT enter `StateLineageKey`, the lineage-directory hash,
the semantic plan fingerprint, or manifest discovery filters. Only static
declarations enter the plan fingerprint. This is essential: placing payload
digests in the lineage key would cause changed values to select an empty new
lineage and silently bypass recovery validation. Under this contract the same
declaration always reaches the existing lineage, where a changed, missing, or
extra digest raises `CheckpointMismatch` before source open.

Prepared-job identity and the checkpoint expectation include sorted
`(name, digest_version, sha256)` triples. A first launch with no manifest may
start with the validated set and records it in the first checkpoint. A launch
against a terminal manifest performs the same comparison before returning the
terminal outcome. Static handles are released exactly once on every exit path
and are never duplicated into a checkpoint.

## 9. Compatibility and migration

- `PROJECT_FORMAT_VERSION` remains `3`.
- Previously valid project-v3 documents remain valid and canonicalize
  unchanged because an empty `static_inputs` list is omitted and existing
  variants are untouched.
- Old runtimes reject `rolling`, `cross_section`, or `static_inputs` as an
  unsupported strict field/variant. There is no downgrade, alias, Python
  fallback, or executable shim.
- Project schema, Python `ProjectDocument`, Studio OpenAPI, and generated
  TypeScript move atomically with the Rust serialized types.
- Capability consumers must branch on `schemaVersion`; schema v1 clients must
  not interpret v2 positionally. `Runtime.capabilities()` returns only v2 after
  cutover.
- Existing provider registration calls remain source compatible but advertise
  conservative batch-only lifecycle facts until explicitly upgraded.
- A legacy provider registered without a finality contract serializes
  `finality: "unproven"`, remains selectable by batch compilation, and is
  rejected by stream compilation; omission or other lifecycle booleans cannot
  upgrade that value.
- `StreamingRunner::new` remains source compatible; Rust static inputs use the
  additive builder and Python uses the new keyword-only argument.
- Manifest format remains v3. Empty static evidence is backward compatible;
  non-empty evidence is understood only by runtimes implementing this gate.
- The first rolling/cross-section state layout has no migration. Any config,
  schema, primitive, operator, or layout mismatch fails before source open.
- Static payload changes are never a request for a fresh lineage. Starting a
  genuinely fresh lineage requires an explicit operational action outside this
  API contract (new pipeline identity/state root or deliberate state removal).

## 10. Reviewer-blocker acceptance

The public API implementation must carry focused tests for all three corrected
surfaces:

- every arithmetic, comparison, and boolean-composition dunder on two
  `ColumnExpr` values or a `ColumnExpr` plus a valid scalar returns
  `ColumnExpr`; the corresponding `ArrayExpr` cases, including an array-kind
  parameter on the right, return `ArrayExpr`;
- every Column/Array cross-domain operand, every array/column namespace
  mismatch, and every listed scalar dunder on `TableExpr` raises `TypeError`
  with the exact template above rather than constructing a wrongly typed node;
- `EventTimeBucket`, `CrossSectionGroup`, `FeatureSet`, and `Program` compare
  only by object identity, and comparisons of distinct equal-looking values do
  not invoke `Expr.__eq__` or `Expr.__bool__`;
- rolling/cross-section drop vectors appear in the existing Rust
  `OperatorStatus` and Python nested operator keys using the exact mapping
  above, including `None` versus zero maximum lateness and checkpoint/replay;
  and
- the Rust and Python top-level job status shapes gain no late-total fields or
  keys, while checked aggregation of their operator maps yields the D7 job
  totals.

## 11. Gate disposition

This note freezes the exact choices delegated by SCE-00:

1. Python/Rust names, signatures, exports, and error mapping are explicit.
2. Rolling/cross-section tags, nested variants, fields, defaults, and omission
   rules are strict and alias-free.
3. Capability schema v2 has one immutable Python shape and one camel-case JSON
   projection with conservative registration defaults.
4. Static input digest version, tagged bytes, scalar canonicalization, manifest
   evidence, and existing-lineage comparison order are byte/field exact.
5. `late_policy.error.scope = "envelope"` plus the staged processing rule makes
   the D7 no-output/no-state transaction observable and fingerprinted.
6. Column and array scalar domains, identity-only equality for expression-
   containing value objects, and the compatibility-preserving late-status
   projection are exact and have focused acceptance vectors.

No API-note blocker remains. The adversarial critique approved B1–B4 after its
one blocker-only review round, and the formal reviewer blockers on expression
domains, expression-containing dataclass equality, and late-status projection
are closed by this targeted correction. SCE-00 is approved for downstream
implementation; none of the frozen public surfaces is implemented by this
note.
