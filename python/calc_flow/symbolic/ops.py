"""The row, ts, cs, window, table, and linalg declaration namespaces.

Every namespace function only constructs declaration nodes. Domain checks
run before node construction so a wrongly typed operand never produces a
wrongly typed expression.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from typing import Literal

from calc_flow.symbolic.domains import (
    is_strict_scalar_type,
    namespace_error,
    type_name,
)
from calc_flow.symbolic.expr import (
    ArrayExpr,
    ColumnExpr,
    Parameter,
    TableExpr,
)
from calc_flow.symbolic.nodes import (
    CDType,
    CEnum,
    CFloat,
    CInt,
    CMap,
    CSeq,
    CStr,
    CValue,
    Node,
    build,
    literal_value,
)
from calc_flow.symbolic.types import (
    check_table_field_type,
    require_int,
    require_non_empty_str,
    require_positive_int,
)
from calc_flow.symbolic.windows import (
    CrossSectionGroup,
    DurationFrame,
    RowFrame,
)


def _column(value: object, function: str, parameter: str, /) -> ColumnExpr:
    if not isinstance(value, ColumnExpr):
        raise namespace_error(function, parameter, "ColumnExpr", value)
    return value


def _table(value: object, function: str, parameter: str, /) -> TableExpr:
    if not isinstance(value, TableExpr):
        raise namespace_error(function, parameter, "TableExpr", value)
    return value


def _array(value: object, function: str, parameter: str, /) -> ArrayExpr:
    if not isinstance(value, ArrayExpr):
        raise namespace_error(function, parameter, "ArrayExpr", value)
    return value


def _column_operand(value: object, function: str, parameter: str, /) -> Node:
    if isinstance(value, ColumnExpr):
        return value._node
    if is_strict_scalar_type(value):
        return build("literal", (), {"value": literal_value(value)})
    raise namespace_error(function, parameter, "ColumnOperand", value)


def _str_sequence(values: object, function: str, parameter: str, /) -> CSeq:
    if isinstance(values, (str, bytes)) or not isinstance(values, Sequence):
        raise namespace_error(function, parameter, "Sequence[str]", values)
    return CSeq(
        tuple(
            CStr(
                require_non_empty_str(
                    value, f"calc_flow.symbolic.{function}.{parameter}[{index}]"
                )
            )
            for index, value in enumerate(values)
        )
    )


def _frame(value: object, function: str, /) -> CMap:
    if isinstance(value, RowFrame):
        return CMap.from_mapping(
            {"frame": CEnum("frame", "rows"), "size": CInt(value.size)}
        )
    if isinstance(value, DurationFrame):
        return CMap.from_mapping(
            {"frame": CEnum("frame", "duration"), "micros": CInt(value.micros)}
        )
    raise namespace_error(function, "window", "RowFrame | DurationFrame", value)


def _min_periods(value: object, frame: object, function: str, /) -> int:
    path = f"calc_flow.symbolic.{function}.min_periods"
    periods = require_positive_int(value, path)
    if isinstance(frame, RowFrame) and periods > frame.size:
        raise ValueError(
            f"{path}: invalid_literal: min_periods must not exceed the row-frame size"
        )
    return periods


def _min_samples(value: object, function: str, /) -> int:
    return require_positive_int(value, f"calc_flow.symbolic.{function}.min_samples")


def _ddof(value: object, function: str, /) -> int:
    path = f"calc_flow.symbolic.{function}.ddof"
    ddof = require_int(value, path)
    if ddof not in (0, 1):
        raise ValueError(f"{path}: invalid_literal: ddof must be 0 or 1")
    return ddof


def _enum_value(
    value: object,
    family: str,
    variants: tuple[str, ...],
    function: str,
    parameter: str,
    /,
) -> CEnum:
    path = f"calc_flow.symbolic.{function}.{parameter}"
    if type(value) is not str:
        raise TypeError(f"{path} must be a string; got {type_name(value)}")
    if value not in variants:
        raise ValueError(
            f"{path}: invalid_literal: {parameter} must be one of"
            f" {', '.join(repr(variant) for variant in variants)}"
        )
    return CEnum(family, value)


class RowNamespace:
    """Row-local expressions over column operands."""

    __slots__ = ()

    def where(
        self,
        condition: ColumnExpr,
        when_true: object,
        when_false: object,
        /,
    ) -> ColumnExpr:
        return ColumnExpr(
            build(
                "where",
                (
                    _column(condition, "row.where", "condition")._node,
                    _column_operand(when_true, "row.where", "when_true"),
                    _column_operand(when_false, "row.where", "when_false"),
                ),
                {},
            )
        )

    def coalesce(self, *values: object) -> ColumnExpr:
        return ColumnExpr(
            build(
                "coalesce",
                tuple(
                    _column_operand(value, "row.coalesce", f"values[{index}]")
                    for index, value in enumerate(values)
                ),
                {},
            )
        )

    def log(self, value: object, /) -> ColumnExpr:
        return ColumnExpr(
            build(
                "log",
                (_column_operand(value, "row.log", "value"),),
                {},
            )
        )

    def exp(self, value: object, /) -> ColumnExpr:
        return ColumnExpr(
            build(
                "exp",
                (_column_operand(value, "row.exp", "value"),),
                {},
            )
        )

    def sqrt(self, value: object, /) -> ColumnExpr:
        return ColumnExpr(
            build(
                "sqrt",
                (_column_operand(value, "row.sqrt", "value"),),
                {},
            )
        )

    def abs(self, value: object, /) -> ColumnExpr:
        return ColumnExpr(
            build(
                "abs",
                (_column_operand(value, "row.abs", "value"),),
                {},
            )
        )

    def clip(
        self,
        value: object,
        /,
        *,
        lower: object,
        upper: object,
    ) -> ColumnExpr:
        function = "row.clip"
        lower_value = literal_value(lower)
        upper_value = literal_value(upper)
        numeric = (CInt, CFloat)
        if (
            isinstance(lower_value, numeric)
            and isinstance(upper_value, numeric)
            and lower_value.value > upper_value.value
        ):
            raise ValueError(
                f"calc_flow.symbolic.{function}.lower: invalid_literal:"
                " clip lower bound must not exceed the upper bound"
            )
        return ColumnExpr(
            build(
                "clip",
                (_column_operand(value, function, "value"),),
                {"lower": lower_value, "upper": upper_value},
            )
        )

    def cast(self, value: object, data_type: str, /) -> ColumnExpr:
        path = "calc_flow.symbolic.row.cast.data_type"
        dtype = require_non_empty_str(data_type, path)
        check_table_field_type(dtype, path)
        return ColumnExpr(
            build(
                "cast",
                (_column_operand(value, "row.cast", "value"),),
                {"data_type": CDType(dtype)},
            )
        )


class TsNamespace:
    """Row-preserving temporal primitives."""

    __slots__ = ()

    def lag(self, value: ColumnExpr, /, *, periods: int = 1) -> ColumnExpr:
        function = "ts.lag"
        return ColumnExpr(
            build(
                "lag",
                (_column(value, function, "value")._node,),
                {
                    "periods": CInt(
                        require_positive_int(
                            periods, f"calc_flow.symbolic.{function}.periods"
                        )
                    )
                },
            )
        )

    def delta(self, value: ColumnExpr, /, *, periods: int = 1) -> ColumnExpr:
        function = "ts.delta"
        return ColumnExpr(
            build(
                "delta",
                (_column(value, function, "value")._node,),
                {
                    "periods": CInt(
                        require_positive_int(
                            periods, f"calc_flow.symbolic.{function}.periods"
                        )
                    )
                },
            )
        )

    def _rolling(
        self,
        primitive: str,
        value: ColumnExpr,
        frame_value: object,
        min_periods: object,
        /,
    ) -> ColumnExpr:
        function = f"ts.{primitive}"
        column = _column(value, function, "value")
        frame = _frame(frame_value, function)
        periods = _min_periods(min_periods, frame_value, function)
        return ColumnExpr(
            build(
                primitive,
                (column._node,),
                {"frame": frame, "min_periods": CInt(periods)},
            )
        )

    def _rolling_ddof(
        self,
        primitive: str,
        value: ColumnExpr,
        frame_value: object,
        min_periods: object,
        ddof: object,
        /,
    ) -> ColumnExpr:
        function = f"ts.{primitive}"
        column = _column(value, function, "value")
        frame = _frame(frame_value, function)
        periods = _min_periods(min_periods, frame_value, function)
        return ColumnExpr(
            build(
                primitive,
                (column._node,),
                {
                    "frame": frame,
                    "min_periods": CInt(periods),
                    "ddof": CInt(_ddof(ddof, function)),
                },
            )
        )

    def count(
        self,
        value: ColumnExpr,
        /,
        *,
        window: RowFrame | DurationFrame,
        min_periods: int = 1,
    ) -> ColumnExpr:
        return self._rolling("count", value, window, min_periods)

    def sum(
        self,
        value: ColumnExpr,
        /,
        *,
        window: RowFrame | DurationFrame,
        min_periods: int = 1,
    ) -> ColumnExpr:
        return self._rolling("sum", value, window, min_periods)

    def mean(
        self,
        value: ColumnExpr,
        /,
        *,
        window: RowFrame | DurationFrame,
        min_periods: int = 1,
    ) -> ColumnExpr:
        return self._rolling("mean", value, window, min_periods)

    def min(
        self,
        value: ColumnExpr,
        /,
        *,
        window: RowFrame | DurationFrame,
        min_periods: int = 1,
    ) -> ColumnExpr:
        return self._rolling("min", value, window, min_periods)

    def max(
        self,
        value: ColumnExpr,
        /,
        *,
        window: RowFrame | DurationFrame,
        min_periods: int = 1,
    ) -> ColumnExpr:
        return self._rolling("max", value, window, min_periods)

    def variance(
        self,
        value: ColumnExpr,
        /,
        *,
        window: RowFrame | DurationFrame,
        min_periods: int = 1,
        ddof: Literal[0, 1] = 1,
    ) -> ColumnExpr:
        return self._rolling_ddof("variance", value, window, min_periods, ddof)

    def stddev(
        self,
        value: ColumnExpr,
        /,
        *,
        window: RowFrame | DurationFrame,
        min_periods: int = 1,
        ddof: Literal[0, 1] = 1,
    ) -> ColumnExpr:
        return self._rolling_ddof("stddev", value, window, min_periods, ddof)

    def covariance(
        self,
        left: ColumnExpr,
        right: ColumnExpr,
        /,
        *,
        window: RowFrame | DurationFrame,
        min_periods: int = 1,
        ddof: Literal[0, 1] = 1,
    ) -> ColumnExpr:
        function = "ts.covariance"
        return ColumnExpr(
            build(
                "covariance",
                (
                    _column(left, function, "left")._node,
                    _column(right, function, "right")._node,
                ),
                {
                    "frame": _frame(window, function),
                    "min_periods": CInt(_min_periods(min_periods, window, function)),
                    "ddof": CInt(_ddof(ddof, function)),
                },
            )
        )

    def correlation(
        self,
        left: ColumnExpr,
        right: ColumnExpr,
        /,
        *,
        window: RowFrame | DurationFrame,
        min_periods: int = 1,
        ddof: Literal[0, 1] = 1,
    ) -> ColumnExpr:
        function = "ts.correlation"
        return ColumnExpr(
            build(
                "correlation",
                (
                    _column(left, function, "left")._node,
                    _column(right, function, "right")._node,
                ),
                {
                    "frame": _frame(window, function),
                    "min_periods": CInt(_min_periods(min_periods, window, function)),
                    "ddof": CInt(_ddof(ddof, function)),
                },
            )
        )


class CsNamespace:
    """Cross-section statistics over explicit complete groups."""

    __slots__ = ()

    @staticmethod
    def _group(value: object, function: str, /) -> CrossSectionGroup:
        if not isinstance(value, CrossSectionGroup):
            raise namespace_error(function, "group", "CrossSectionGroup", value)
        return value

    @staticmethod
    def _grouping(group: CrossSectionGroup, /) -> CValue:
        if group.bucket is None:
            return CEnum("grouping", "exact_time")
        return CMap.from_mapping(
            {
                "grouping": CEnum("grouping", "fixed_bucket"),
                "width_micros": CInt(group.bucket.width_micros),
            }
        )

    @staticmethod
    def _group_args(group: CrossSectionGroup, /) -> tuple[Node, ...]:
        return (
            group.event_time._node,
            *(column._node for column in group.partition_by),
        )

    def rank(
        self,
        value: ColumnExpr,
        /,
        *,
        group: CrossSectionGroup,
        direction: Literal["ascending", "descending"] = "ascending",
        tie_method: Literal["average", "min", "max"] = "average",
        null_placement: Literal["exclude", "first", "last"] = "exclude",
        min_samples: int = 1,
    ) -> ColumnExpr:
        function = "cs.rank"
        validated = self._group(group, function)
        return ColumnExpr(
            build(
                "rank",
                (
                    _column(value, function, "value")._node,
                    *self._group_args(validated),
                ),
                {
                    "grouping": self._grouping(validated),
                    "direction": _enum_value(
                        direction,
                        "direction",
                        ("ascending", "descending"),
                        function,
                        "direction",
                    ),
                    "tie_method": _enum_value(
                        tie_method,
                        "rank_tie_method",
                        ("average", "min", "max"),
                        function,
                        "tie_method",
                    ),
                    "null_placement": _enum_value(
                        null_placement,
                        "null_placement",
                        ("exclude", "first", "last"),
                        function,
                        "null_placement",
                    ),
                    "min_samples": CInt(_min_samples(min_samples, function)),
                },
            )
        )

    def percentile(
        self,
        value: ColumnExpr,
        /,
        *,
        group: CrossSectionGroup,
        direction: Literal["ascending", "descending"] = "ascending",
        tie_method: Literal["average", "min", "max"] = "average",
        null_placement: Literal["exclude", "first", "last"] = "exclude",
        min_samples: int = 1,
    ) -> ColumnExpr:
        function = "cs.percentile"
        validated = self._group(group, function)
        return ColumnExpr(
            build(
                "percentile",
                (
                    _column(value, function, "value")._node,
                    *self._group_args(validated),
                ),
                {
                    "grouping": self._grouping(validated),
                    "direction": _enum_value(
                        direction,
                        "direction",
                        ("ascending", "descending"),
                        function,
                        "direction",
                    ),
                    "tie_method": _enum_value(
                        tie_method,
                        "rank_tie_method",
                        ("average", "min", "max"),
                        function,
                        "tie_method",
                    ),
                    "null_placement": _enum_value(
                        null_placement,
                        "null_placement",
                        ("exclude", "first", "last"),
                        function,
                        "null_placement",
                    ),
                    "min_samples": CInt(_min_samples(min_samples, function)),
                },
            )
        )

    def demean(
        self,
        value: ColumnExpr,
        /,
        *,
        group: CrossSectionGroup,
        min_samples: int = 1,
    ) -> ColumnExpr:
        function = "cs.demean"
        validated = self._group(group, function)
        return ColumnExpr(
            build(
                "demean",
                (
                    _column(value, function, "value")._node,
                    *self._group_args(validated),
                ),
                {
                    "grouping": self._grouping(validated),
                    "min_samples": CInt(_min_samples(min_samples, function)),
                },
            )
        )

    def zscore(
        self,
        value: ColumnExpr,
        /,
        *,
        group: CrossSectionGroup,
        min_samples: int = 1,
        ddof: Literal[0, 1] = 0,
    ) -> ColumnExpr:
        function = "cs.zscore"
        validated = self._group(group, function)
        return ColumnExpr(
            build(
                "zscore",
                (
                    _column(value, function, "value")._node,
                    *self._group_args(validated),
                ),
                {
                    "grouping": self._grouping(validated),
                    "min_samples": CInt(_min_samples(min_samples, function)),
                    "ddof": CInt(_ddof(ddof, function)),
                },
            )
        )

    def winsorize(
        self,
        value: ColumnExpr,
        /,
        *,
        group: CrossSectionGroup,
        lower: float,
        upper: float,
        min_samples: int = 1,
    ) -> ColumnExpr:
        function = "cs.winsorize"
        validated = self._group(group, function)
        lower_value = _finite_bound(lower, function, "lower")
        upper_value = _finite_bound(upper, function, "upper")
        if not 0 <= lower_value <= upper_value <= 1:
            path = (
                f"calc_flow.symbolic.{function}.lower"
                if lower_value < 0 or lower_value > upper_value
                else f"calc_flow.symbolic.{function}.upper"
            )
            raise ValueError(
                f"{path}: invalid_literal: bounds must satisfy 0 <= lower <= upper <= 1"
            )
        return ColumnExpr(
            build(
                "winsorize",
                (
                    _column(value, function, "value")._node,
                    *self._group_args(validated),
                ),
                {
                    "grouping": self._grouping(validated),
                    "min_samples": CInt(_min_samples(min_samples, function)),
                    "lower": _number_value(lower_value),
                    "upper": _number_value(upper_value),
                },
            )
        )


def _finite_bound(value: object, function: str, parameter: str, /) -> int | float:
    path = f"calc_flow.symbolic.{function}.{parameter}"
    if type(value) not in (int, float):
        raise TypeError(f"{path} must be a finite number; got {type_name(value)}")
    if isinstance(value, float) and not math.isfinite(value):
        raise ValueError(f"{path}: invalid_literal: must be finite")
    return value


def _number_value(value: int | float, /) -> CValue:
    return CInt(value) if type(value) is int else CFloat(value)


class TableNamespace:
    """Table-scope bridging and structural operations."""

    __slots__ = ()

    def project(self, value: TableExpr, columns: Sequence[str], /) -> TableExpr:
        function = "table.project"
        return TableExpr(
            build(
                "project",
                (_table(value, function, "value")._node,),
                {"columns": _str_sequence(columns, function, "columns")},
            )
        )

    def filter(self, value: TableExpr, predicate: ColumnExpr, /) -> TableExpr:
        function = "table.filter"
        return TableExpr(
            build(
                "filter",
                (
                    _table(value, function, "value")._node,
                    _column(predicate, function, "predicate")._node,
                ),
                {},
            )
        )

    def attach_columns(
        self,
        value: TableExpr,
        array: ArrayExpr,
        /,
        *,
        names: Sequence[str],
    ) -> TableExpr:
        function = "table.attach_columns"
        return TableExpr(
            build(
                "attach_columns",
                (
                    _table(value, function, "value")._node,
                    _array(array, function, "array")._node,
                ),
                {"names": _str_sequence(names, function, "names")},
            )
        )


class LinalgNamespace:
    """Explicit table/array boundary operations."""

    __slots__ = ()

    def from_columns(
        self,
        value: TableExpr,
        /,
        *,
        columns: Sequence[str],
        backend: str,
    ) -> ArrayExpr:
        function = "linalg.from_columns"
        backend_value = require_non_empty_str(
            backend, f"calc_flow.symbolic.{function}.backend"
        )
        return ArrayExpr(
            build(
                "from_columns",
                (_table(value, function, "value")._node,),
                {
                    "columns": _str_sequence(columns, function, "columns"),
                    "backend": CStr(backend_value),
                },
            )
        )

    def matmul(
        self,
        left: ArrayExpr,
        right: ArrayExpr | Parameter[ArrayExpr],
        /,
    ) -> ArrayExpr:
        function = "linalg.matmul"
        left_node = _array(left, function, "left")._node
        if isinstance(right, ArrayExpr) or (
            isinstance(right, Parameter) and right.kind == "array"
        ):
            right_node = right._node
        else:
            raise namespace_error(
                function,
                "right",
                "ArrayExpr | Parameter[ArrayExpr]",
                right,
            )
        return ArrayExpr(build("matmul", (left_node, right_node), {}))


class WindowNamespace:
    """Cardinality-changing event-time windows."""

    __slots__ = ()

    def tumbling(
        self,
        value: TableExpr,
        /,
        *,
        event_time: str,
        size_micros: int,
        group_by: Sequence[str] = (),
    ) -> TableExpr:
        function = "window.tumbling"
        return TableExpr(
            build(
                "window_tumbling",
                (_table(value, function, "value")._node,),
                {
                    "event_time": CStr(
                        require_non_empty_str(
                            event_time,
                            f"calc_flow.symbolic.{function}.event_time",
                        )
                    ),
                    "size_micros": CInt(
                        require_positive_int(
                            size_micros,
                            f"calc_flow.symbolic.{function}.size_micros",
                        )
                    ),
                    "group_by": _str_sequence(group_by, function, "group_by"),
                },
            )
        )

    def hopping(
        self,
        value: TableExpr,
        /,
        *,
        event_time: str,
        size_micros: int,
        slide_micros: int,
        group_by: Sequence[str] = (),
    ) -> TableExpr:
        function = "window.hopping"
        return TableExpr(
            build(
                "window_hopping",
                (_table(value, function, "value")._node,),
                {
                    "event_time": CStr(
                        require_non_empty_str(
                            event_time,
                            f"calc_flow.symbolic.{function}.event_time",
                        )
                    ),
                    "size_micros": CInt(
                        require_positive_int(
                            size_micros,
                            f"calc_flow.symbolic.{function}.size_micros",
                        )
                    ),
                    "slide_micros": CInt(
                        require_positive_int(
                            slide_micros,
                            f"calc_flow.symbolic.{function}.slide_micros",
                        )
                    ),
                    "group_by": _str_sequence(group_by, function, "group_by"),
                },
            )
        )


row = RowNamespace()
ts = TsNamespace()
cs = CsNamespace()
table = TableNamespace()
linalg = LinalgNamespace()
window = WindowNamespace()
