"""Immutable expressions with native execution through explicit collection.

Operators build declarations with canonical digests; collection lowers a fresh
native plan without capturing data in the expression. Use ``identical()`` for
structural comparison instead of symbolic equality or truth testing.
"""

from __future__ import annotations

import inspect
from collections.abc import Awaitable, Callable, Mapping, Sequence
from typing import TYPE_CHECKING, Generic, Literal, TypeVar, Union, overload

import pyarrow as pa
from typing_extensions import Concatenate, ParamSpec, Unpack

from calc_flow._compat import TypeAliasType, dataclass
from calc_flow.symbolic.domains import (
    array_operator_error,
    bool_error,
    column_operator_error,
    is_strict_scalar_type,
    namespace_error,
    table_operator_error,
    type_name,
)
from calc_flow.symbolic.nodes import (
    CBool,
    CDType,
    CEnum,
    CInt,
    CMap,
    CNull,
    CSeq,
    CStr,
    Node,
    build,
    explain_node,
    literal_value,
)
from calc_flow.symbolic.types import (
    TABLE_FIELD_TYPES,
    BatchKind,
    CompileMode,
    Field,
    ScalarLiteral,
    check_array_dtype,
    check_table_field_type,
    require_non_empty_str,
    require_non_negative_int,
)

if TYPE_CHECKING:
    from datetime import timedelta

    from calc_flow._native import Batch, ExecutionOptions, NativeArray
    from calc_flow._native_array import ArrayExecutionOptions
    from calc_flow.asof_join_spec import AsofStateLimits
    from calc_flow.compute import TableData
    from calc_flow.pipeline import Runtime
    from calc_flow.runtime import StreamRuntimeConfig, WatermarkPolicy
    from calc_flow.stream import StreamInput, StreamResults
    from calc_flow.symbolic.asof import _AsofJoinOptions
    from calc_flow.symbolic.distinct import DistinctSpec
    from calc_flow.symbolic.program import FeatureSet

ColumnOperand = TypeAliasType("ColumnOperand", Union["ColumnExpr", ScalarLiteral])
ArrayOperand = TypeAliasType(
    "ArrayOperand", Union["ArrayExpr", "Parameter[ArrayExpr]", ScalarLiteral]
)
T = TypeVar("T")
SelfExpr = TypeVar("SelfExpr", bound="Expr")
P = ParamSpec("P")
R = TypeVar("R")


@dataclass(frozen=True, slots=True, eq=False, repr=False)
class Expr(Generic[T]):
    """Base class of every immutable symbolic expression."""

    _node: Node

    @property
    def digest(self) -> str:
        """The lowercase hexadecimal v1 node digest of this declaration."""

        return self._node.digest

    def identical(self, other: object, /) -> bool:
        """Compare the complete normalized structure of two expressions."""

        if not isinstance(other, Expr):
            return False
        return (
            self._node.digest == other._node.digest
            and self._node.node_bytes == other._node.node_bytes
        )

    def explain(self) -> str:
        """Render the declaration tree deterministically."""

        return explain_node(self._node)

    def pipe(
        self: SelfExpr,
        function: Callable[Concatenate[SelfExpr, P], R],
        /,
        *args: P.args,
        **kwargs: P.kwargs,
    ) -> R:
        """Apply a synchronous declaration builder once, preserving its return type."""
        if not callable(function):
            raise TypeError("pipe.function: expected a synchronous callable")
        result = function(self, *args, **kwargs)
        if inspect.isawaitable(result):
            if inspect.iscoroutine(result):
                result.close()
            raise TypeError("pipe.result: use a synchronous declaration builder")
        return result

    def __bool__(self) -> bool:
        raise bool_error()

    def __repr__(self) -> str:
        return f"{type(self).__qualname__}(digest={self._node.digest})"

    __hash__ = None


def _literal_node(value: ScalarLiteral, /) -> Node:
    return build("literal", (), {"value": literal_value(value)})


def lit(value: ScalarLiteral, /) -> ColumnExpr:
    """Declare a strict scalar literal for a column expression."""
    return ColumnExpr(_literal_node(value))


def _column_operand(operand: object, operator: str, /) -> Node:
    if isinstance(operand, ColumnExpr):
        return operand._node
    if is_strict_scalar_type(operand):
        return _literal_node(operand)  # type: ignore[arg-type]
    raise column_operator_error(operator, operand)


def _array_operand(operand: object, operator: str, /) -> Node:
    if isinstance(operand, ArrayExpr):
        return operand._node
    if isinstance(operand, Parameter) and operand.kind == "array":
        return operand._node
    if is_strict_scalar_type(operand):
        return _literal_node(operand)  # type: ignore[arg-type]
    raise array_operator_error(operator, operand)


def _column_binary(
    left: ColumnExpr, right: object, primitive: str, operator: str, /
) -> ColumnExpr:
    return ColumnExpr(
        build(
            primitive,
            (
                _column_operand(left, operator),
                _column_operand(right, operator),
            ),
            {},
        )
    )


def _require_native_integer_literal(left: ArrayExpr, operand: object) -> None:
    if type(operand) is not int or abs(operand) <= 2**53:
        return
    from calc_flow.symbolic.native_arrays import is_native_array

    if is_native_array(left._node):
        raise ValueError("native array literal must satisfy abs(value) <= 2^53")


def _array_binary(
    left: ArrayExpr, right: object, primitive: str, operator: str, /
) -> ArrayExpr:
    _require_native_integer_literal(left, right)
    return ArrayExpr(
        build(
            primitive,
            (_array_operand(left, operator), _array_operand(right, operator)),
            {},
        )
    )


@dataclass(frozen=True, slots=True, eq=False, repr=False)
class ColumnExpr(Expr[object]):
    """A row-local column expression over one table lineage."""

    def sum(self) -> AggregateExpr:
        """Sum a direct numeric column, ignoring NULL values."""
        return AggregateExpr._column("SUM", self)

    def count(self) -> AggregateExpr:
        """Count non-NULL values of a direct column."""
        return AggregateExpr._column("COUNT", self)

    def min(self) -> AggregateExpr:
        """Minimum of a direct numeric column."""
        return AggregateExpr._column("MIN", self)

    def max(self) -> AggregateExpr:
        """Maximum of a direct numeric column."""
        return AggregateExpr._column("MAX", self)

    def mean(self) -> AggregateExpr:
        """Mean of a direct numeric column as Float64."""
        return AggregateExpr._column("AVG", self)

    def __eq__(self, other: ColumnOperand, /) -> ColumnExpr:
        return _column_binary(self, other, "eq", "==")

    def __ne__(self, other: ColumnOperand, /) -> ColumnExpr:
        return _column_binary(self, other, "ne", "!=")

    def __lt__(self, other: ColumnOperand, /) -> ColumnExpr:
        return _column_binary(self, other, "lt", "<")

    def __le__(self, other: ColumnOperand, /) -> ColumnExpr:
        return _column_binary(self, other, "le", "<=")

    def __gt__(self, other: ColumnOperand, /) -> ColumnExpr:
        return _column_binary(self, other, "gt", ">")

    def __ge__(self, other: ColumnOperand, /) -> ColumnExpr:
        return _column_binary(self, other, "ge", ">=")

    def __add__(self, other: ColumnOperand, /) -> ColumnExpr:
        return _column_binary(self, other, "add", "+")

    def __radd__(self, other: ColumnOperand, /) -> ColumnExpr:
        return _column_binary(self, other, "add", "+")

    def __sub__(self, other: ColumnOperand, /) -> ColumnExpr:
        return _column_binary(self, other, "sub", "-")

    def __rsub__(self, other: ColumnOperand, /) -> ColumnExpr:
        return _reflected_column(self, other, "sub", "-")

    def __mul__(self, other: ColumnOperand, /) -> ColumnExpr:
        return _column_binary(self, other, "mul", "*")

    def __rmul__(self, other: ColumnOperand, /) -> ColumnExpr:
        return _column_binary(self, other, "mul", "*")

    def __truediv__(self, other: ColumnOperand, /) -> ColumnExpr:
        return _column_binary(self, other, "truediv", "/")

    def __rtruediv__(self, other: ColumnOperand, /) -> ColumnExpr:
        return _reflected_column(self, other, "truediv", "/")

    def __neg__(self, /) -> ColumnExpr:
        return ColumnExpr(build("neg", (self._node,), {}))

    def __and__(self, other: ColumnOperand, /) -> ColumnExpr:
        return _column_binary(self, other, "and", "&")

    def __rand__(self, other: ColumnOperand, /) -> ColumnExpr:
        return _column_binary(self, other, "and", "&")

    def __or__(self, other: ColumnOperand, /) -> ColumnExpr:
        return _column_binary(self, other, "or", "|")

    def __ror__(self, other: ColumnOperand, /) -> ColumnExpr:
        return _column_binary(self, other, "or", "|")

    def __invert__(self, /) -> ColumnExpr:
        return ColumnExpr(build("not", (self._node,), {}))


def _reflected_column(
    left: ColumnExpr, right: object, primitive: str, operator: str, /
) -> ColumnExpr:
    """Build the reflected non-commutative operators ``right <op> left``."""

    return ColumnExpr(
        build(
            primitive,
            (
                _column_operand(right, operator),
                _column_operand(left, operator),
            ),
            {},
        )
    )


@dataclass(frozen=True, slots=True, eq=False, init=False)
class AggregateExpr:
    """Immutable aggregate declaration; obtain it from a column or count_rows."""

    _function: str
    _value: ColumnExpr | None

    def __init__(self) -> None:
        raise TypeError("AggregateExpr: use column reducers or count_rows()")

    @classmethod
    def _create(cls, function: str, value: ColumnExpr | None) -> AggregateExpr:
        aggregate = object.__new__(cls)
        object.__setattr__(aggregate, "_function", function)
        object.__setattr__(aggregate, "_value", value)
        return aggregate

    @classmethod
    def _column(cls, function: str, value: ColumnExpr) -> AggregateExpr:
        if value._node.op.name != "column_ref":
            raise ValueError(
                "aggregate.value: requires a direct column; use with_columns first"
            )
        return cls._create(function, value)

    def __bool__(self) -> bool:
        raise bool_error()


def count_rows() -> AggregateExpr:
    """Count every row, including rows containing NULL."""
    return AggregateExpr._create("COUNT", None)


@dataclass(frozen=True, slots=True, eq=False, repr=False)
class ArrayExpr(Expr[object]):
    """An immutable array declaration with explicit backend, dtype and lineage."""

    def collect_array(
        self,
        inputs: Mapping[str, NativeArray | Batch | TableData],
        /,
        *,
        runtime: Runtime,
        options: ExecutionOptions | None = None,
        array_options: ArrayExecutionOptions | None = None,
    ) -> NativeArray:
        """Collect one native batch result from declared array and table inputs."""
        from calc_flow.compute import _collect_array

        return _collect_array(self, inputs, runtime, options, array_options)

    def reshape(self, shape: Sequence[int], /) -> ArrayExpr:
        """Declare a packed scalar/vector/matrix reshape with concrete extents."""
        if isinstance(shape, (str, bytes)) or not isinstance(shape, Sequence):
            raise TypeError("ArrayExpr.reshape.shape: expected a sequence")
        if len(shape) > 2:
            raise ValueError("ArrayExpr.reshape.shape: expected rank at most two")
        dimensions = tuple(
            CInt(require_non_negative_int(value, f"ArrayExpr.reshape.shape[{index}]"))
            for index, value in enumerate(shape)
        )
        return ArrayExpr(
            build("array_reshape", (self._node,), {"shape": CSeq(dimensions)})
        )

    def transpose(self) -> ArrayExpr:
        """Declare a native transpose; rank zero and one remain unchanged."""
        return ArrayExpr(build("array_transpose", (self._node,), {}))

    def sum(
        self,
        *,
        axis: int | None = None,
        keepdims: bool = False,
        skip_nulls: bool = False,
    ) -> ArrayExpr:
        """Declare an exact native sum over all positions or one axis."""
        return _array_reduce(self, "sum", axis, keepdims, skip_nulls)

    def mean(
        self,
        *,
        axis: int | None = None,
        keepdims: bool = False,
        skip_nulls: bool = False,
    ) -> ArrayExpr:
        """Declare an exact native mean over all positions or one axis."""
        return _array_reduce(self, "mean", axis, keepdims, skip_nulls)

    def min(
        self,
        *,
        axis: int | None = None,
        keepdims: bool = False,
        skip_nulls: bool = False,
    ) -> ArrayExpr:
        """Declare a native minimum over all positions or one axis."""
        return _array_reduce(self, "min", axis, keepdims, skip_nulls)

    def max(
        self,
        *,
        axis: int | None = None,
        keepdims: bool = False,
        skip_nulls: bool = False,
    ) -> ArrayExpr:
        """Declare a native maximum over all positions or one axis."""
        return _array_reduce(self, "max", axis, keepdims, skip_nulls)

    def __eq__(self, other: ArrayOperand, /) -> ArrayExpr:
        return _array_binary(self, other, "eq", "==")

    def __ne__(self, other: ArrayOperand, /) -> ArrayExpr:
        return _array_binary(self, other, "ne", "!=")

    def __lt__(self, other: ArrayOperand, /) -> ArrayExpr:
        return _array_binary(self, other, "lt", "<")

    def __le__(self, other: ArrayOperand, /) -> ArrayExpr:
        return _array_binary(self, other, "le", "<=")

    def __gt__(self, other: ArrayOperand, /) -> ArrayExpr:
        return _array_binary(self, other, "gt", ">")

    def __ge__(self, other: ArrayOperand, /) -> ArrayExpr:
        return _array_binary(self, other, "ge", ">=")

    def __add__(self, other: ArrayOperand, /) -> ArrayExpr:
        return _array_binary(self, other, "add", "+")

    def __radd__(self, other: ArrayOperand, /) -> ArrayExpr:
        return _array_binary(self, other, "add", "+")

    def __sub__(self, other: ArrayOperand, /) -> ArrayExpr:
        return _array_binary(self, other, "sub", "-")

    def __rsub__(self, other: ArrayOperand, /) -> ArrayExpr:
        return _reflected_array(self, other, "sub", "-")

    def __mul__(self, other: ArrayOperand, /) -> ArrayExpr:
        return _array_binary(self, other, "mul", "*")

    def __rmul__(self, other: ArrayOperand, /) -> ArrayExpr:
        return _array_binary(self, other, "mul", "*")

    def __truediv__(self, other: ArrayOperand, /) -> ArrayExpr:
        return _array_binary(self, other, "truediv", "/")

    def __rtruediv__(self, other: ArrayOperand, /) -> ArrayExpr:
        return _reflected_array(self, other, "truediv", "/")

    def __neg__(self, /) -> ArrayExpr:
        return ArrayExpr(build("neg", (self._node,), {}))

    def __and__(self, other: ArrayOperand, /) -> ArrayExpr:
        return _array_binary(self, other, "and", "&")

    def __rand__(self, other: ArrayOperand, /) -> ArrayExpr:
        return _array_binary(self, other, "and", "&")

    def __or__(self, other: ArrayOperand, /) -> ArrayExpr:
        return _array_binary(self, other, "or", "|")

    def __ror__(self, other: ArrayOperand, /) -> ArrayExpr:
        return _array_binary(self, other, "or", "|")

    def __invert__(self, /) -> ArrayExpr:
        return ArrayExpr(build("not", (self._node,), {}))


def _array_reduce(
    value: ArrayExpr,
    function: str,
    axis: int | None,
    keepdims: bool,
    skip_nulls: bool,
) -> ArrayExpr:
    if axis is not None:
        require_non_negative_int(axis, f"ArrayExpr.{function}.axis")
    for name, flag in (("keepdims", keepdims), ("skip_nulls", skip_nulls)):
        if type(flag) is not bool:
            raise TypeError(f"ArrayExpr.{function}.{name}: expected a bool")
    return ArrayExpr(
        build(
            "array_reduce",
            (value._node,),
            {
                "function": CStr(function),
                "axis": CNull() if axis is None else CInt(axis),
                "keepdims": CBool(keepdims),
                "skip_nulls": CBool(skip_nulls),
            },
        )
    )


def _reflected_array(
    left: ArrayExpr, right: object, primitive: str, operator: str, /
) -> ArrayExpr:
    """Build the reflected non-commutative operators ``right <op> left``."""

    _require_native_integer_literal(left, right)
    return ArrayExpr(
        build(
            primitive,
            (
                _array_operand(right, operator),
                _array_operand(left, operator),
            ),
            {},
        )
    )


@dataclass(frozen=True, slots=True, eq=False, repr=False)
class TableExpr(Expr[object]):
    """An Arrow-backed table expression."""

    def distinct(self, columns: Sequence[str], /, *, spec: DistinctSpec) -> TableExpr:
        """Final DISTINCT with copy_visible_v2 owned peak accounting.

        Borrowed external ingress backing is excluded while synchronously copied;
        retained history, query workspace, output and captures are charged.
        Stream finalization requires proved boundedness or window closure.
        """
        from calc_flow.symbolic.distinct import declaration

        return TableExpr(declaration(self._node, columns, spec))

    def union_all(self, other: TableExpr, /) -> TableExpr:
        """Append rows with an exact matching schema, preserving duplicates.

        Each input keeps its FIFO order; cross-input order is unspecified.
        Complete snapshots cannot feed this append-only union under latest_relation.
        """
        if not isinstance(other, TableExpr):
            raise TypeError(
                f"union_all.other requires TableExpr, got {type_name(other)}"
            )
        return TableExpr(build("union_all", (self._node, other._node), {}))

    def aggregate(
        self,
        aggregates: Mapping[str, AggregateExpr],
        /,
        *,
        group_by: Sequence[str] = (),
    ) -> TableExpr:
        """Declare named reductions; keys precede reductions in output order.

        Empty global input has one row: COUNT is zero and numeric reducers are
        typed NULL. Grouped empty input has zero rows. Streaming append inputs
        accumulate; latest complete snapshots require an explicit runtime budget.
        """
        from calc_flow.symbolic.aggregate import aggregate

        return aggregate(self, aggregates, group_by)

    def collect_schema(
        self, *, runtime: Runtime | None = None, mode: CompileMode = "batch"
    ) -> pa.Schema:
        """Plan the exact Arrow schema without executing input data."""
        from calc_flow.symbolic.program import Program

        return Program(
            "table-schema", mode=mode, outputs={"result": self}
        ).collect_schema(runtime)["result"]

    def __eq__(self, other: object, /) -> bool:
        raise table_operator_error("==")

    def __ne__(self, other: object, /) -> bool:
        raise table_operator_error("!=")

    def __lt__(self, other: object, /) -> bool:
        raise table_operator_error("<")

    def __le__(self, other: object, /) -> bool:
        raise table_operator_error("<=")

    def __gt__(self, other: object, /) -> bool:
        raise table_operator_error(">")

    def __ge__(self, other: object, /) -> bool:
        raise table_operator_error(">=")

    def __add__(self, other: object, /) -> bool:
        raise table_operator_error("+")

    def __radd__(self, other: object, /) -> bool:
        raise table_operator_error("+")

    def __sub__(self, other: object, /) -> bool:
        raise table_operator_error("-")

    def __rsub__(self, other: object, /) -> bool:
        raise table_operator_error("-")

    def __mul__(self, other: object, /) -> bool:
        raise table_operator_error("*")

    def __rmul__(self, other: object, /) -> bool:
        raise table_operator_error("*")

    def __truediv__(self, other: object, /) -> bool:
        raise table_operator_error("/")

    def __rtruediv__(self, other: object, /) -> bool:
        raise table_operator_error("/")

    def __neg__(self, /) -> bool:
        raise table_operator_error("unary -")

    def __and__(self, other: object, /) -> bool:
        raise table_operator_error("&")

    def __rand__(self, other: object, /) -> bool:
        raise table_operator_error("&")

    def __or__(self, other: object, /) -> bool:
        raise table_operator_error("|")

    def __ror__(self, other: object, /) -> bool:
        raise table_operator_error("|")

    def __invert__(self, /) -> bool:
        raise table_operator_error("~")

    def __getitem__(self, field: str, /) -> ColumnExpr:
        if type(field) is not str:
            raise TypeError(
                "TableExpr field selection requires a string field name;"
                f" got {type_name(field)}"
            )
        return ColumnExpr(build("column_ref", (self._node,), {"name": CStr(field)}))

    def with_columns(
        self,
        features: FeatureSet | Mapping[str, ColumnExpr] | None = None,
        /,
        **named: ColumnExpr,
    ) -> TableExpr:
        """Append named expressions; existing columns cannot be replaced."""
        from calc_flow.symbolic.program import FeatureSet

        if features is None:
            entries = ()
        elif isinstance(features, FeatureSet):
            entries = features.features
        elif isinstance(features, Mapping):
            entries = tuple(features.items())
        else:
            raise namespace_error(
                "TableExpr.with_columns",
                "features",
                "FeatureSet | Mapping | None",
                features,
            )
        combined = FeatureSet((*entries, *named.items()))
        names = CSeq(tuple(CStr(name) for name, _ in combined.features))
        return TableExpr(
            build(
                "with_columns",
                (self._node, *(value._node for _, value in combined.features)),
                {"names": names},
            )
        )

    def select(self, *columns: str, **named: ColumnExpr) -> TableExpr:
        """Project literal column names followed by named derived expressions."""
        from calc_flow.symbolic.ops import table

        names = _validate_name_sequence(columns, "TableExpr.select.columns") + tuple(
            named
        )
        if not names:
            raise ValueError("TableExpr.select: select at least one column")
        if len(set(names)) != len(names):
            raise ValueError("TableExpr.select: duplicate output column name")
        source = self.with_columns(named) if named else self
        return table.project(source, names)

    def filter(self, predicate: ColumnExpr, /) -> TableExpr:
        """Keep rows matching a boolean expression over this table."""
        from calc_flow.symbolic.ops import table

        return table.filter(self, predicate)

    def sql(self, query: str, /) -> TableExpr:
        """Declare a read-only SQL stage using this table as the ``input`` alias."""
        from calc_flow.symbolic.sql import sql

        return sql(query, input=self)

    def stream_asof_join(
        self,
        right: TableExpr,
        /,
        *,
        tolerance: timedelta,
        limits: AsofStateLimits,
        **options: Unpack[_AsofJoinOptions],
    ) -> TableExpr:
        """Attach the latest historical right row after both inputs finalize.

        Accepts the same ``keys``, ``late_policy`` and ``prefixes`` options as
        ``cf.table.stream_asof_join``. ``collect`` consumes complete finite inputs;
        ``stream`` retains candidates until both input frontiers close.
        """
        from calc_flow.symbolic.asof import _AsofOptions, declare_asof

        return declare_asof(self, right, tolerance, limits, _AsofOptions(**options))

    def stream(
        self,
        inputs: StreamInput | Mapping[str, StreamInput | TableData | NativeArray],
        /,
        *,
        runtime: Runtime | None = None,
        config: StreamRuntimeConfig | None = None,
        watermarks: WatermarkPolicy | Mapping[str, WatermarkPolicy] | None = None,
    ) -> StreamResults[pa.Table]:
        """Own a native stream with ``async with`` and ``async for``.

        Declared event-time inputs advance safe watermarks by default and require
        nondecreasing arrival times. Select an explicit policy for unordered data
        or iterable-provided watermarks. Supplied SourceBinding policies stay owned
        by their bindings.
        """
        from calc_flow.stream import _stream_table

        return _stream_table(self, inputs, runtime, config, watermarks)

    def collect(
        self,
        inputs: TableData | Mapping[str, TableData | NativeArray],
        /,
        *,
        runtime: Runtime | None = None,
        options: ExecutionOptions | None = None,
    ) -> pa.Table:
        """Execute with named inputs, or a single table when there is one table root."""
        from calc_flow.compute import _require_blocking, _table_inputs
        from calc_flow.symbolic.program import Program

        _require_blocking("collect")
        program = Program("collect", engine="sql", outputs={"output": self})
        return program.collect(
            _table_inputs(program, inputs), runtime=runtime, options=options
        )["output"]

    def collect_async(
        self,
        inputs: TableData | Mapping[str, TableData | NativeArray],
        /,
        *,
        runtime: Runtime | None = None,
        options: ExecutionOptions | None = None,
    ) -> Awaitable[pa.Table]:
        """Copy input mappings, capture Batch references and await execution.

        Arrow buffers are shared; keep their underlying storage read-only until
        execution completes.
        """
        from calc_flow.compute import _collect_table_async, _table_inputs
        from calc_flow.symbolic.program import Program

        program = Program("collect", engine="sql", outputs={"output": self})
        return _collect_table_async(
            program, _table_inputs(program, inputs), runtime, options
        )


@dataclass(frozen=True, slots=True, eq=False, repr=False)
class Parameter(Expr[T]):
    """A named external static input of table or array kind.

    Parameters define no scalar operator dunders; an array-kind parameter
    participates in expressions only where a signature names it.
    """

    @property
    def name(self) -> str:
        value = self._node.attr("name")
        if isinstance(value, CStr):
            return value.value
        raise TypeError("parameter node is missing its name attribute")

    @property
    def kind(self) -> BatchKind:
        value = self._node.attr("kind")
        if isinstance(value, CEnum):
            return value.variant  # type: ignore[return-value]
        raise TypeError("parameter node is missing its kind attribute")


def _arrow_fields(schema: pa.Schema, path: str, /) -> tuple[Field, ...]:
    special = {
        "time32[s]": pa.time32("s"),
        "time64[us]": pa.time64("us"),
        "timestamp[ms]": pa.timestamp("ms"),
        "timestamp[us]": pa.timestamp("us"),
        "timestamp[us, UTC]": pa.timestamp("us", tz="UTC"),
    }
    types = {
        special[name] if name in special else pa.type_for_alias(name): name
        for name in TABLE_FIELD_TYPES
    }
    fields = []
    for field in schema:
        data_type = types.get(field.type)
        if data_type is None:
            raise ValueError(
                f"{path}.{field.name}: unsupported_type: "
                f"unsupported Arrow type {field.type}"
            )
        fields.append(Field(field.name, data_type, nullable=field.nullable))
    return tuple(fields)


def _validate_schema(
    schema: pa.Schema | Sequence[Field], path: str, /
) -> tuple[Field, ...]:
    if isinstance(schema, pa.Schema):
        schema = _arrow_fields(schema, path)
    fields: list[Field] = []
    seen: set[str] = set()
    for index, field in enumerate(schema):
        if not isinstance(field, Field):
            raise TypeError(
                f"{path}[{index}]: schema entries must be Field values; got"
                f" {type_name(field)}"
            )
        check_table_field_type(field.data_type, f"{path}[{index}].data_type")
        if field.name in seen:
            raise ValueError(
                f"{path}[{index}].name: duplicate_name: duplicate field"
                f" name {field.name!r}"
            )
        seen.add(field.name)
        fields.append(field)
    return tuple(fields)


def _validate_name_sequence(values: object, path: str, /) -> tuple[str, ...]:
    if isinstance(values, (str, bytes)) or not isinstance(values, Sequence):
        raise TypeError(
            f"{path} must be a sequence of strings; got {type_name(values)}"
        )
    names: list[str] = []
    for index, value in enumerate(values):
        names.append(require_non_empty_str(value, f"{path}[{index}]"))
    return tuple(names)


def _cv_schema(fields: Sequence[Field], /) -> CSeq:
    return CSeq(
        tuple(
            CMap.from_mapping(
                {
                    "name": CStr(field.name),
                    "data_type": CDType(field.data_type),
                    "nullable": CBool(field.nullable),
                }
            )
            for field in fields
        )
    )


def _require_native_descriptor(dtype: str, shape: Sequence[object], path: str) -> None:
    if dtype not in {"bool", "float32", "float64"}:
        raise ValueError(f"{path}.dtype: expected bool, float32 or float64")
    if len(shape) > 2:
        raise ValueError(f"{path}.shape: expected rank at most two")


def array_input(
    name: str,
    /,
    *,
    dtype: str,
    shape: Sequence[int | None],
    backend: Literal["native"] = "native",
) -> ArrayExpr:
    """Declare one dynamic native array without capturing its input values."""
    path = "array_input"
    name = require_non_empty_str(name, f"{path}.name")
    backend = require_non_empty_str(backend, f"{path}.backend")
    dtype = require_non_empty_str(dtype, f"{path}.dtype")
    if backend != "native":
        raise ValueError(f"{path}.backend: expected native")
    if isinstance(shape, (str, bytes)) or not isinstance(shape, Sequence):
        raise TypeError(f"{path}.shape: expected a sequence of dimensions")
    _require_native_descriptor(dtype, shape, path)
    dimensions = tuple(
        CNull()
        if value is None
        else CInt(require_non_negative_int(value, f"{path}.shape[{index}]"))
        for index, value in enumerate(shape)
    )
    return ArrayExpr(
        build(
            "array_input",
            (),
            {
                "name": CStr(name),
                "backend": CStr(backend),
                "dtype": CDType(dtype),
                "shape": CSeq(dimensions),
            },
        )
    )


def table_input(
    name: str,
    /,
    *,
    schema: pa.Schema | Sequence[Field],
    entity_by: Sequence[str] = (),
    event_time: str | None = None,
    sequence_by: Sequence[str] = (),
) -> TableExpr:
    """Declare one named table input with its exact schema and ordering."""

    if type(name) is not str:
        raise TypeError(
            f"inputs.table_input.name must be a non-empty string; got {type_name(name)}"
        )
    if not name:
        raise ValueError(
            "inputs.table_input.name: invalid_literal: must be a non-empty string"
        )
    base = f"inputs.{name}"
    fields = _validate_schema(schema, f"{base}.schema")
    entity = _validate_name_sequence(entity_by, f"{base}.entity_by")
    sequence = _validate_name_sequence(sequence_by, f"{base}.sequence_by")
    if event_time is not None:
        require_non_empty_str(event_time, f"{base}.event_time")
    node = build(
        "table_input",
        (),
        {
            "name": CStr(name),
            "schema": _cv_schema(fields),
            "entity_by": CSeq(tuple(CStr(value) for value in entity)),
            "event_time": CNull() if event_time is None else CStr(event_time),
            "sequence_by": CSeq(tuple(CStr(value) for value in sequence)),
        },
    )
    return TableExpr(node)


def _is_absent_or_empty(value: object, /) -> bool:
    """Type-safe absence check that never triggers element-wise equality.

    Only ``None`` and a genuinely empty ``list``/``tuple`` count as absent;
    array-like values (for example NumPy arrays) fail without invoking
    their overloaded ``==`` so the stable rejection path is reached.
    """

    return value is None or (isinstance(value, (list, tuple)) and len(value) == 0)


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


def _require_parameter_name(name: object, /) -> str:
    if type(name) is not str:
        raise TypeError(
            "static_inputs.parameter.name must be a non-empty string;"
            f" got {type_name(name)}"
        )
    if not name:
        raise ValueError(
            "static_inputs.parameter.name: invalid_literal: must be a non-empty string"
        )
    return name


def _require_parameter_mutability(mutability: object, base: str, /) -> None:
    if type(mutability) is not str:
        raise TypeError(
            f"{base}.mutability must be a string; got {type_name(mutability)}"
        )
    if mutability != "static":
        raise ValueError(
            f"{base}.mutability: invalid_literal: the only accepted"
            " mutability is 'static'"
        )


def _table_parameter(
    name: str,
    base: str,
    schema: object,
    backend: object,
    dtype: object,
    shape: object,
    /,
) -> Parameter[object]:
    for field_name, value in (("backend", backend), ("dtype", dtype)):
        if value is not None:
            raise ValueError(
                f"{base}.{field_name}: invalid_literal: table"
                f" parameters do not accept {field_name}"
            )
    if not _is_absent_or_empty(shape):
        raise ValueError(
            f"{base}.shape: invalid_literal: table parameters do not accept shape"
        )
    fields = _validate_schema(schema, f"{base}.schema")  # type: ignore[arg-type]
    node = build(
        "parameter",
        (),
        {
            "name": CStr(name),
            "kind": CEnum("batch_kind", "table"),
            "mutability": CEnum("mutability", "static"),
            "schema": _cv_schema(fields),
        },
    )
    return Parameter(node)


def _array_parameter(
    name: str,
    base: str,
    schema: object,
    backend: object,
    dtype: object,
    shape: object,
    /,
) -> Parameter[object]:
    if not _is_absent_or_empty(schema):
        raise ValueError(
            f"{base}.schema: invalid_literal: array parameters do not accept schema"
        )
    backend_value = require_non_empty_str(backend, f"{base}.backend")
    dtype_value = require_non_empty_str(dtype, f"{base}.dtype")
    check_array_dtype(dtype_value, f"{base}.dtype")
    if isinstance(shape, (str, bytes)) or not isinstance(shape, Sequence):
        raise TypeError(
            f"{base}.shape must be a sequence of integers; got {type_name(shape)}"
        )
    dims = tuple(
        require_non_negative_int(value, f"{base}.shape[{index}]")
        for index, value in enumerate(shape)
    )
    if backend_value == "native":
        _require_native_descriptor(dtype_value, dims, base)
    node = build(
        "parameter",
        (),
        {
            "name": CStr(name),
            "kind": CEnum("batch_kind", "array"),
            "mutability": CEnum("mutability", "static"),
            "backend": CStr(backend_value),
            "dtype": CDType(dtype_value),
            "shape": CSeq(tuple(CInt(value) for value in dims)),
        },
    )
    return Parameter(node)


def parameter(
    name: str,
    /,
    *,
    kind: object,
    schema: object = (),
    backend: object = None,
    dtype: object = None,
    shape: object = (),
    mutability: object = "static",
) -> Parameter[object]:
    """Declare one named static external parameter of table or array kind."""

    validated_name = _require_parameter_name(name)
    base = f"static_inputs.{validated_name}"
    if type(kind) is not str or kind not in ("table", "array"):
        if type(kind) is not str:
            raise TypeError(f"{base}.kind must be a string; got {type_name(kind)}")
        raise ValueError(
            f"{base}.kind: invalid_literal: kind must be 'table' or 'array'"
        )
    _require_parameter_mutability(mutability, base)
    if kind == "table":
        return _table_parameter(validated_name, base, schema, backend, dtype, shape)
    return _array_parameter(validated_name, base, schema, backend, dtype, shape)
