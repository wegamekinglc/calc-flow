"""Data-only reduction construction and schema-domain validation."""

from __future__ import annotations

from collections.abc import Mapping, Sequence

from calc_flow.symbolic.lower.segments import _quote_identifier
from calc_flow.symbolic.nodes import CSeq, CStr, build
from calc_flow.symbolic.types import require_non_empty_str

_FUNCTIONS = {name: name for name in ("SUM", "COUNT", "MIN", "MAX", "AVG")}
_NUMERIC = frozenset(
    {
        "int8",
        "int16",
        "int32",
        "int64",
        "uint8",
        "uint16",
        "uint32",
        "uint64",
        "float32",
        "float64",
    }
)
_GROUP_KEYS = _NUMERIC - {"float32", "float64"} | {
    "bool",
    "boolean",
    "string",
    "large_string",
}


def aggregate(source, aggregates, group_by):
    from calc_flow.symbolic.expr import TableExpr

    entries, groups = _entries(aggregates, group_by)
    reductions = tuple(_reducer(source, name, value) for name, value in entries)
    query = _query(groups, tuple(sql for _, sql in reductions))
    return TableExpr(
        build(
            "aggregate",
            (source._node,),
            {
                "query": CStr(query),
                "aliases": CSeq((CStr("input"),)),
                "group_by": CSeq(tuple(CStr(key) for key in groups)),
                "reducers": CSeq(tuple(spec for spec, _ in reductions)),
            },
        )
    )


def _entries(aggregates, group_by):
    if not isinstance(aggregates, Mapping):
        raise TypeError("aggregate.aggregates: expected a mapping")
    if not aggregates:
        raise ValueError("aggregate.aggregates: provide at least one aggregate")
    groups = _groups(group_by)
    entries = tuple(aggregates.items())
    names = (
        *groups,
        *(require_non_empty_str(name, "aggregate.aggregates") for name, _ in entries),
    )
    if len(set(names)) != len(names):
        raise ValueError(
            "aggregate: duplicate_name: group and result names must be unique"
        )
    return entries, groups


def _reducer(source, name, value):
    from calc_flow.symbolic.expr import AggregateExpr

    if not isinstance(value, AggregateExpr):
        raise TypeError(f"aggregate.aggregates.{name}: expected AggregateExpr")
    function = _FUNCTIONS.get(value._function)
    if function is None:
        raise ValueError(f"aggregate.aggregates.{name}: unsupported reducer")
    column = _operand(source, name, value)
    operand = _quote_identifier(column) if column else "*"
    sql = f"{function}({operand}) AS {_quote_identifier(name)}"
    return CSeq((CStr(name), CStr(function), CStr(column))), sql


def _operand(source, name, value):
    if value._value is None:
        return ""
    node = value._value._node
    if node.op.name != "column_ref" or node.args[0].digest != source.digest:
        raise ValueError(
            f"aggregate.aggregates.{name}.value: "
            "columns require the input relation lineage"
        )
    return node.attr("name").value


def _query(groups, reductions):
    """Join fixed SQL clauses containing only quoted names and admitted reducers."""
    projection = ", ".join((*(_quote_identifier(key) for key in groups), *reductions))
    clauses = ["SELECT", projection, "FROM input"]
    if groups:
        clauses.extend(
            ("GROUP BY", ", ".join(_quote_identifier(key) for key in groups))
        )
    return " ".join(clauses)


def validate(analyzer, node, path):
    facts = analyzer.table(node.args[0], f"{path}.aggregate.value")
    fields = {field.name: field.data_type for field in facts.schema}
    _validate_groups(analyzer, node, fields, path)
    _validate_reducers(analyzer, node, fields, path)


def _validate_groups(analyzer, node, fields, path):
    from calc_flow.symbolic.analyzer import _cstr_seq

    for index, key in enumerate(_cstr_seq(node.attr("group_by"))):
        if fields.get(key) not in _GROUP_KEYS:
            analyzer.issue(
                f"{path}.aggregate.group_by[{index}]",
                "unsupported_type",
                "group key requires boolean, integer or string",
            )


def _validate_reducers(analyzer, node, fields, path):
    from calc_flow.symbolic.analyzer import _cstr_seq
    from calc_flow.symbolic.query_analysis import _segment

    for reducer in node.attr("reducers").items:
        name, function, column = _cstr_seq(reducer)
        if _invalid_operand(column, function, fields):
            analyzer.issue(
                f"{_segment(f'{path}.aggregate.aggregates', name)}.value",
                "unsupported_type",
                f"{function} requires a direct numeric column",
            )


def _invalid_operand(column, function, fields):
    return column and (
        column not in fields or (function != "COUNT" and fields[column] not in _NUMERIC)
    )


def _groups(group_by):
    if isinstance(group_by, (str, bytes)) or not isinstance(group_by, Sequence):
        raise TypeError("aggregate.group_by: expected a sequence of strings")
    return tuple(require_non_empty_str(key, "aggregate.group_by") for key in group_by)
