"""Collect consumed state expressions before MACD-only SSA forwarding."""

from __future__ import annotations

from typing import TYPE_CHECKING

from calc_flow.symbolic.expr import TableExpr
from calc_flow.symbolic.lower.macd_meaning import _key
from calc_flow.symbolic.nodes import CSeq, Node

if TYPE_CHECKING:
    from calc_flow.symbolic.analyzer import _Analyzer
    from calc_flow.symbolic.program import Program


def _names(value: CSeq) -> tuple[str, ...]:
    return tuple(item.value for item in value.items)


class _MacdDemands:
    """Own the backwards field-demand union for one immutable declaration."""

    def __init__(self, analyzer: _Analyzer) -> None:
        self._analyzer = analyzer
        self._requested: dict[tuple[str, bytes], set[str]] = {}
        self._expressions: dict[
            tuple[tuple[str, bytes], tuple[str, bytes]], tuple[Node, Node]
        ] = {}

    def _schema(self, table: Node) -> set[str]:
        return {
            field.name for field in self._analyzer.table(table, "macd.demand").schema
        }

    def _expression(self, node: Node, target: Node) -> set[str]:
        self._expressions[_key(node), _key(target)] = node, target
        if node.op.name == "column_ref":
            return {node.attr("name").value}
        return {name for child in node.args for name in self._expression(child, target)}

    def table(self, table: Node, requested: set[str]) -> None:
        key = _key(table)
        seen = self._requested.setdefault(key, set())
        added = requested - seen
        if not added:
            return
        seen.update(added)
        self._table(table, added)

    def _table(self, table: Node, requested: set[str]) -> None:
        kind = table.op.name
        if kind in {"late_output", "late_rows"}:
            return
        if kind == "with_columns":
            return self._columns(table, requested)
        if kind in {"project", "filter"}:
            extra = (
                self._expression(table.args[1], table.args[0])
                if kind == "filter"
                else set()
            )
            return self.table(table.args[0], requested | extra)
        if kind == "attach_columns":
            # Native attachment preserves the entire physical input table.
            self.array(table.args[1])
            return self.table(table.args[0], self._schema(table.args[0]))
        for child in table.args:
            if child.digest in self._analyzer._table_cache:
                self.table(child, self._schema(child))

    def _columns(self, table: Node, requested: set[str]) -> None:
        expressions = dict(zip(_names(table.attr("names")), table.args[1:]))
        inputs = requested - expressions.keys()
        for name in sorted(requested & expressions.keys()):
            inputs |= self._expression(expressions[name], table.args[0])
        self.table(table.args[0], inputs)

    def array(self, node: Node) -> None:
        if node.op.name == "from_columns":
            self.table(node.args[0], set(_names(node.attr("columns"))))
        else:
            for child in node.args:
                self.array(child)

    def collect(self, program: Program) -> tuple[tuple[Node, Node], ...]:
        for _, output in program.outputs:
            if isinstance(output, TableExpr):
                self.table(output._node, self._schema(output._node))
            else:
                self.array(output._node)
        return tuple(self._expressions.values())
