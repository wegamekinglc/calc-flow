"""Explicit lazy SQL declarations sharing the expression graph and native runtime."""

from __future__ import annotations

import re

from calc_flow.symbolic.expr import TableExpr
from calc_flow.symbolic.nodes import CSeq, CStr, build
from calc_flow.symbolic.types import require_non_empty_str


def sql(query: str, /, **tables: TableExpr) -> TableExpr:
    """Declare SELECT/CTE SQL over explicit table aliases without executing rows.

    SQL has its own row lineage and does not inherit temporal ordering. A stream
    accepts one alias. Aggregates emit cumulative snapshots after each batch;
    other SQL evaluates each native batch separately.
    """
    return _sql_declaration("sql", query, tables)


def snapshot_sql(query: str, /, **tables: TableExpr) -> TableExpr:
    """Declare stream-only SQL over complete replacement relations.

    Every input event replaces its alias, including an empty table. Once all
    aliases are initialized, each update emits the complete query result using
    the latest available relations. A Batch's RecordBatches form one snapshot.
    Inputs and results must fit their edge budgets without splitting.

    Use Program(engine="streaming") or TableExpr.stream. The result has a new
    row lineage, no temporal ordering, and must go directly to a sink. Exactly
    once delivery is unsupported. Custom SourceBindings must provide a complete
    relation per Data event; the operator cannot detect application slicing.
    SourceBinding.from_arrow cannot directly supply a snapshot input; pass the
    materialized table or an async iterable yielding whole tables instead.
    """
    return _sql_declaration("snapshot_sql", query, tables)


def _sql_declaration(kind: str, query: str, tables: dict[str, TableExpr]) -> TableExpr:
    require_non_empty_str(query, f"{kind}.query")
    if not tables:
        raise ValueError(f"{kind}.tables: provide at least one table alias")
    names = sorted(tables)
    for name in names:
        if re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", name) is None:
            raise ValueError(
                f"{kind}.tables.{name}: expected a portable SQL identifier"
            )
        if not isinstance(tables[name], TableExpr):
            raise TypeError(f"{kind}.tables.{name}: expected TableExpr")
    if kind == "snapshot_sql" and len({name.lower() for name in names}) != len(names):
        raise ValueError(
            "snapshot_sql.tables: aliases conflict after SQL normalization"
        )
    return TableExpr(
        build(
            kind,
            tuple(tables[name]._node for name in names),
            {
                "query": CStr(query),
                "aliases": CSeq(tuple(CStr(name) for name in names)),
            },
        )
    )
