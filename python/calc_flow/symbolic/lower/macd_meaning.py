"""Canonical, complete-byte meanings for MACD-only state forwarding."""

from __future__ import annotations

from calc_flow.symbolic.analyzer import (
    _CROSS_SECTION,
    _ROLLING_PRIMITIVES,
    _ROW_LOCAL_PRIMITIVES,
    _Analyzer,
    _indicator_column_definition,
    _indicator_expression_matches,
    _indicator_reference_matches,
    _indicator_references,
    _indicator_row_set,
)
from calc_flow.symbolic.nodes import CStr, Node, build

_ROW_PRESERVING = frozenset({"project", "with_columns", "attach_columns"})
_EXPRESSION_OPS = _ROW_LOCAL_PRIMITIVES | _ROLLING_PRIMITIVES | _CROSS_SECTION
_NodeKey = tuple[str, bytes]
_StateKey = tuple[object, ...]


def _key(node: Node) -> _NodeKey:
    return node.digest, node.node_bytes


def _rebuild(node: Node, arguments: tuple[Node, ...]) -> Node:
    return build(
        node.op.name, arguments, dict(node.attrs.entries), version=node.op.version
    )


def _reference(table: Node, name: str) -> Node:
    return build("column_ref", (table,), {"name": CStr(name)})


def _ancestors(table: Node) -> tuple[Node, ...]:
    ancestors = [table]
    while table.op.name in _ROW_PRESERVING:
        table = table.args[0]
        ancestors.append(table)
    return tuple(reversed(ancestors))


class _MacdMeanings:
    """Own memoized proofs over one already-validated declaration graph."""

    def __init__(self, analyzer: _Analyzer) -> None:
        self._analyzer = analyzer
        self._memo: dict[tuple[_NodeKey, _NodeKey], Node] = {}

    def _field_matches(self, reference: Node, target: Node) -> bool:
        name = reference.attr("name")
        if not isinstance(name, CStr):
            return False
        source = self._analyzer.table(reference.args[0], "macd.source").schema
        actual = self._analyzer.table(target, "macd.target").schema
        return next(
            (field for field in source if field.name == name.value), None
        ) == next((field for field in actual if field.name == name.value), None)

    def available(self, expression: Node, table: Node) -> bool:
        return _indicator_expression_matches(expression, table, {}) and all(
            self._field_matches(reference, table)
            for reference in _indicator_references(expression)
        )

    def origin(self, expression: Node, target: Node) -> Node:
        return next(
            table for table in _ancestors(target) if self.available(expression, table)
        )

    def normalize(self, node: Node, target: Node) -> Node:
        key = _key(node), _key(target)
        if key not in self._memo:
            self._memo[key] = node
            self._memo[key] = self._normalize(node, target)
        return self._memo[key]

    def _normalize(self, node: Node, target: Node) -> Node:
        if node.op.name == "column_ref":
            return self._column(node, target)
        if node.op.name not in _EXPRESSION_OPS:
            return node
        return _rebuild(
            node, tuple(self.normalize(child, target) for child in node.args)
        )

    def _column(self, node: Node, target: Node) -> Node:
        if not _indicator_reference_matches(node, target, {}):
            return node
        name = node.attr("name").value
        definition = _indicator_column_definition(node.args[0], name)
        if definition is not None and _indicator_row_set(
            definition[1]
        ) == _indicator_row_set(target):
            return self.normalize(*definition)
        origin = self.origin(node, target)
        return _reference(origin, name)

    def state_key(self, node: Node, target: Node) -> _StateKey:
        normalized = self.normalize(node, target)
        facts = self._analyzer.table(target, "macd.context")
        return (
            _indicator_row_set(target),
            facts.entity_by,
            facts.event_time,
            facts.sequence_by,
            normalized.digest,
            normalized.node_bytes,
        )
