"""Explicit lazy SQL declarations sharing the expression graph and native runtime."""

from __future__ import annotations

import re

from calc_flow.symbolic.expr import TableExpr
from calc_flow.symbolic.nodes import CSeq, CStr, build
from calc_flow.symbolic.types import require_non_empty_str


def sql(query: str, /, **tables: TableExpr) -> TableExpr:
    """Declare SELECT/CTE SQL over explicit table aliases without executing rows.

    SQL has its own row lineage and does not inherit temporal ordering. A stream
    accepts one alias and evaluates the SQL separately for each native batch.
    """
    require_non_empty_str(query, "sql.query")
    if not tables:
        raise ValueError("sql.tables: provide at least one table alias")
    names = sorted(tables)
    for name in names:
        if re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", name) is None:
            raise ValueError(f"sql.tables.{name}: expected a portable SQL identifier")
        if not isinstance(tables[name], TableExpr):
            raise TypeError(f"sql.tables.{name}: expected TableExpr")
    return TableExpr(
        build(
            "sql",
            tuple(tables[name]._node for name in names),
            {
                "query": CStr(query),
                "aliases": CSeq(tuple(CStr(name) for name in names)),
            },
        )
    )
