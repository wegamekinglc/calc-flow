from __future__ import annotations

import re

import pytest

from calc_flow.symbolic import (
    ArrayExpr,
    ColumnExpr,
    Field,
    TableExpr,
    cs,
    event_time_bucket,
    exact_time,
    linalg,
    parameter,
    row,
    rows,
    table,
    table_input,
    ts,
    window,
)

COLUMN_ERROR_TEMPLATE = (
    "symbolic column operator {operator} requires ColumnExpr or a strict"
    " scalar literal; got {type}"
)
ARRAY_ERROR_TEMPLATE = (
    "symbolic array operator {operator} requires ArrayExpr,"
    " Parameter[ArrayExpr], or a strict scalar literal; got {type}"
)
TABLE_ERROR_TEMPLATE = (
    "symbolic table expressions do not support scalar operator {operator};"
    " select a column or use table/window operations"
)


def _table() -> TableExpr:
    return table_input(
        "quotes",
        schema=[
            Field("ts", "timestamp[us, UTC]", nullable=False),
            Field("x", "float64"),
        ],
    )


def _column() -> ColumnExpr:
    return _table()["x"]


def _array() -> ArrayExpr:
    return linalg.from_columns(_table(), columns=["x"], backend="numpy")


def _array_parameter() -> parameter:  # type: ignore[valid-type]
    return parameter(
        "weights", kind="array", backend="numpy", dtype="float64", shape=(1,)
    )


@pytest.mark.parametrize(
    ("invoke", "token"),
    [
        (lambda a, b: a == b, "=="),
        (lambda a, b: a != b, "!="),
        (lambda a, b: a < b, "<"),
        (lambda a, b: a <= b, "<="),
        (lambda a, b: a > b, ">"),
        (lambda a, b: a >= b, ">="),
        (lambda a, b: a + b, "+"),
        (lambda a, b: a - b, "-"),
        (lambda a, b: a * b, "*"),
        (lambda a, b: a / b, "/"),
        (lambda a, b: a & b, "&"),
        (lambda a, b: a | b, "|"),
    ],
)
def test_column_operators_reject_array_domain(invoke, token) -> None:
    column = _column()
    with pytest.raises(
        TypeError,
        match=re.escape(
            COLUMN_ERROR_TEMPLATE.format(
                operator=token, type="calc_flow.symbolic.expr.ArrayExpr"
            )
        ),
    ):
        invoke(column, _array())


@pytest.mark.parametrize(
    ("invoke", "token"),
    [
        (lambda a, b: a == b, "=="),
        (lambda a, b: a + b, "+"),
        (lambda a, b: a & b, "&"),
    ],
)
def test_column_operators_reject_table_and_parameter_domains(invoke, token) -> None:
    column = _column()
    for foreign, type_name in (
        (_table(), "calc_flow.symbolic.expr.TableExpr"),
        (_array_parameter(), "calc_flow.symbolic.expr.Parameter"),
    ):
        with pytest.raises(
            TypeError,
            match=re.escape(
                COLUMN_ERROR_TEMPLATE.format(operator=token, type=type_name)
            ),
        ):
            invoke(column, foreign)


@pytest.mark.parametrize(
    ("invoke", "token"),
    [
        (lambda a, b: a == b, "=="),
        (lambda a, b: a != b, "!="),
        (lambda a, b: a < b, "<"),
        (lambda a, b: a <= b, "<="),
        (lambda a, b: a > b, ">"),
        (lambda a, b: a >= b, ">="),
        (lambda a, b: a + b, "+"),
        (lambda a, b: a - b, "-"),
        (lambda a, b: a * b, "*"),
        (lambda a, b: a / b, "/"),
        (lambda a, b: a & b, "&"),
        (lambda a, b: a | b, "|"),
    ],
)
def test_array_operators_reject_column_domain(invoke, token) -> None:
    array = _array()
    with pytest.raises(
        TypeError,
        match=re.escape(
            ARRAY_ERROR_TEMPLATE.format(
                operator=token, type="calc_flow.symbolic.expr.ColumnExpr"
            )
        ),
    ):
        invoke(array, _column())


def test_array_operators_reject_table_and_table_parameter_domains() -> None:
    array = _array()
    table_parameter = parameter(
        "reference", kind="table", schema=[Field("k", "string")]
    )
    for foreign, type_name in (
        (_table(), "calc_flow.symbolic.expr.TableExpr"),
        (table_parameter, "calc_flow.symbolic.expr.Parameter"),
    ):
        with pytest.raises(
            TypeError,
            match=re.escape(ARRAY_ERROR_TEMPLATE.format(operator="+", type=type_name)),
        ):
            array + foreign  # type: ignore[operator]


@pytest.mark.parametrize(
    ("invoke", "token"),
    [
        (lambda t: t == t, "=="),
        (lambda t: t != t, "!="),
        (lambda t: t < t, "<"),
        (lambda t: t <= t, "<="),
        (lambda t: t > t, ">"),
        (lambda t: t >= t, ">="),
        (lambda t: t + t, "+"),
        (lambda t: t + 1, "+"),
        (lambda t: 1 + t, "+"),
        (lambda t: t - t, "-"),
        (lambda t: 1 - t, "-"),
        (lambda t: t * t, "*"),
        (lambda t: 2 * t, "*"),
        (lambda t: t / t, "/"),
        (lambda t: t / 2, "/"),
        (lambda t: 2 / t, "/"),
        (lambda t: -t, "unary -"),
        (lambda t: t & t, "&"),
        (lambda t: True & t, "&"),
        (lambda t: t | t, "|"),
        (lambda t: True | t, "|"),
        (lambda t: ~t, "~"),
    ],
)
def test_table_scalar_dunders_are_rejected(invoke, token) -> None:
    with pytest.raises(
        TypeError,
        match=re.escape(TABLE_ERROR_TEMPLATE.format(operator=token)),
    ):
        invoke(_table())


def test_expressions_have_no_truth_value() -> None:
    message = (
        "symbolic expressions have no truth value; use &, |, and ~ for"
        " symbolic boolean composition, or identical() for structural"
        " identity"
    )
    column = _column()
    with pytest.raises(TypeError, match=re.escape(message)):
        bool(column)
    with pytest.raises(TypeError, match=re.escape(message)):
        _ = column and column
    with pytest.raises(TypeError, match=re.escape(message)):
        not _array()
    with pytest.raises(TypeError, match=re.escape(message)):
        not _table()


def test_expressions_are_unhashable() -> None:
    for expression in (_column(), _array(), _table(), _array_parameter()):
        with pytest.raises(TypeError):
            hash(expression)


def test_window_values_compare_only_by_identity() -> None:
    ts_column = _table()["ts"]
    first = exact_time(ts_column)
    second = exact_time(ts_column)
    assert first == first
    assert first != second
    assert hash(first) == hash(first)
    bucket_first = event_time_bucket(ts_column, width_micros=60)
    bucket_second = event_time_bucket(ts_column, width_micros=60)
    assert bucket_first == bucket_first
    assert bucket_first != bucket_second
    assert first != bucket_first


def test_namespace_functions_reject_wrong_domains() -> None:
    column = _column()
    array = _array()
    quotes = _table()
    with pytest.raises(
        TypeError,
        match=re.escape(
            "calc_flow.symbolic.ts.lag.value: expected ColumnExpr; got"
            " calc_flow.symbolic.expr.ArrayExpr"
        ),
    ):
        ts.lag(array)
    with pytest.raises(
        TypeError,
        match=re.escape(
            "calc_flow.symbolic.ts.covariance.right: expected ColumnExpr;"
            " got calc_flow.symbolic.expr.ArrayExpr"
        ),
    ):
        ts.covariance(column, array, window=rows(5))
    with pytest.raises(
        TypeError,
        match=re.escape(
            "calc_flow.symbolic.row.where.condition: expected ColumnExpr;"
            " got builtins.int"
        ),
    ):
        row.where(1, 1, 0)
    with pytest.raises(
        TypeError,
        match=re.escape(
            "calc_flow.symbolic.row.log.value: expected ColumnOperand; got"
            " calc_flow.symbolic.expr.TableExpr"
        ),
    ):
        row.log(quotes)
    with pytest.raises(
        TypeError,
        match=re.escape(
            "calc_flow.symbolic.cs.rank.value: expected ColumnExpr; got"
            " calc_flow.symbolic.expr.TableExpr"
        ),
    ):
        cs.rank(quotes, group=exact_time(column))
    with pytest.raises(
        TypeError,
        match=re.escape(
            "calc_flow.symbolic.cs.rank.group: expected CrossSectionGroup;"
            " got calc_flow.symbolic.expr.ColumnExpr"
        ),
    ):
        cs.rank(column, group=column)
    with pytest.raises(
        TypeError,
        match=re.escape(
            "calc_flow.symbolic.table.project.value: expected TableExpr;"
            " got calc_flow.symbolic.expr.ColumnExpr"
        ),
    ):
        table.project(column, ["x"])
    with pytest.raises(
        TypeError,
        match=re.escape(
            "calc_flow.symbolic.table.filter.predicate: expected"
            " ColumnExpr; got calc_flow.symbolic.expr.TableExpr"
        ),
    ):
        table.filter(quotes, quotes)
    with pytest.raises(
        TypeError,
        match=re.escape(
            "calc_flow.symbolic.table.attach_columns.array: expected"
            " ArrayExpr; got calc_flow.symbolic.expr.ColumnExpr"
        ),
    ):
        table.attach_columns(quotes, column, names=["score"])
    with pytest.raises(
        TypeError,
        match=re.escape(
            "calc_flow.symbolic.linalg.from_columns.value: expected"
            " TableExpr; got calc_flow.symbolic.expr.ArrayExpr"
        ),
    ):
        linalg.from_columns(array, columns=["x"], backend="numpy")
    with pytest.raises(
        TypeError,
        match=re.escape(
            "calc_flow.symbolic.linalg.matmul.left: expected ArrayExpr;"
            " got calc_flow.symbolic.expr.ColumnExpr"
        ),
    ):
        linalg.matmul(column, array)
    with pytest.raises(
        TypeError,
        match=re.escape(
            "calc_flow.symbolic.linalg.matmul.right: expected ArrayExpr |"
            " Parameter[ArrayExpr]; got calc_flow.symbolic.expr.ColumnExpr"
        ),
    ):
        linalg.matmul(array, column)
    with pytest.raises(
        TypeError,
        match=re.escape(
            "calc_flow.symbolic.window.tumbling.value: expected TableExpr;"
            " got calc_flow.symbolic.expr.ArrayExpr"
        ),
    ):
        window.tumbling(array, event_time="ts", size_micros=60)
    with pytest.raises(
        TypeError,
        match=re.escape(
            "calc_flow.symbolic.exact_time.event_time: expected"
            " ColumnExpr; got calc_flow.symbolic.expr.ArrayExpr"
        ),
    ):
        exact_time(array)
    with pytest.raises(
        TypeError,
        match=re.escape(
            "calc_flow.symbolic.exact_time.partition_by[0]: expected"
            " ColumnExpr; got builtins.int"
        ),
    ):
        exact_time(column, partition_by=[1])  # type: ignore[list-item]


def test_parameters_do_not_define_scalar_dunders() -> None:
    weights = _array_parameter()
    with pytest.raises(TypeError, match="unsupported operand type"):
        weights + 1  # type: ignore[operator]
    with pytest.raises(TypeError, match="unsupported operand type"):
        weights + weights  # type: ignore[operator]
    assert not hasattr(weights, "__add__")
