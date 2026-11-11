"""Data-only reduction construction and schema-domain validation."""

from __future__ import annotations

from collections.abc import Mapping, Sequence

from calc_flow.symbolic.nodes import CSeq, CStr, build
from calc_flow.symbolic.types import require_non_empty_str


def aggregate(source, aggregates, group_by):
    from calc_flow.symbolic.expr import AggregateExpr, TableExpr

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

    def quote(value):
        return '"' + value.replace('"', '""') + '"'

    select = [quote(key) for key in groups]
    reducers = []
    for name, value in entries:
        if not isinstance(value, AggregateExpr):
            raise TypeError(f"aggregate.aggregates.{name}: expected AggregateExpr")
        column = ""
        if value._value is not None:
            node = value._value._node
            if node.op.name != "column_ref" or node.args[0].digest != source.digest:
                raise ValueError(
                    f"aggregate.aggregates.{name}.value: "
                    "columns require the input relation lineage"
                )
            column = node.attr("name").value
        reducers.append(CSeq((CStr(name), CStr(value._function), CStr(column))))
        select.append(
            f"{value._function}({quote(column) if column else '*'}) AS {quote(name)}"
        )
    query = f"SELECT {', '.join(select)} FROM input"
    if groups:
        query += " GROUP BY " + ", ".join(quote(key) for key in groups)
    return TableExpr(
        build(
            "aggregate",
            (source._node,),
            {
                "query": CStr(query),
                "aliases": CSeq((CStr("input"),)),
                "group_by": CSeq(tuple(CStr(key) for key in groups)),
                "reducers": CSeq(tuple(reducers)),
            },
        )
    )


def validate(analyzer, node, path):
    from calc_flow.symbolic.analyzer import _cstr_seq
    from calc_flow.symbolic.query_analysis import _segment

    facts = analyzer.table(node.args[0], f"{path}.aggregate.value")
    fields = {f.name: f.data_type for f in facts.schema}
    numeric = {
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
    keys = numeric - {"float32", "float64"} | {
        "bool",
        "boolean",
        "string",
        "large_string",
    }
    for index, key in enumerate(_cstr_seq(node.attr("group_by"))):
        if fields.get(key) not in keys:
            analyzer.issue(
                f"{path}.aggregate.group_by[{index}]",
                "unsupported_type",
                "group key requires boolean, integer or string",
            )
    for reducer in node.attr("reducers").items:
        name, function, column = _cstr_seq(reducer)
        if column and (
            column not in fields
            or (function != "COUNT" and fields[column] not in numeric)
        ):
            analyzer.issue(
                f"{_segment(f'{path}.aggregate.aggregates', name)}.value",
                "unsupported_type",
                f"{function} requires a direct numeric column",
            )


def _groups(group_by):
    if isinstance(group_by, (str, bytes)) or not isinstance(group_by, Sequence):
        raise TypeError("aggregate.group_by: expected a sequence of strings")
    return tuple(require_non_empty_str(key, "aggregate.group_by") for key in group_by)
