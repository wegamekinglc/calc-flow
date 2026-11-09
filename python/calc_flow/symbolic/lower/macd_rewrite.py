"""Rebind proven MACD references across typed row-preserving materialization."""

from __future__ import annotations

from collections.abc import Mapping

from calc_flow.symbolic.analyzer import (
    _CROSS_SECTION,
    _ROLLING_PRIMITIVES,
    _ROW_LOCAL_PRIMITIVES,
    _Analyzer,
    _indicator_reference_matches,
    _schema_fields,
)
from calc_flow.symbolic.lower.schema import infer_table_schema
from calc_flow.symbolic.nodes import CStr, Node, build
from calc_flow.symbolic.types import Field

_ROW_PRESERVING = frozenset({"project", "with_columns", "attach_columns"})


class _MacdMaterializedRewriter:
    """Own a substitution snapshot and its exact native schema witnesses."""

    def __init__(self, analyzer: _Analyzer, replacements: Mapping[str, Node]) -> None:
        self._analyzer = analyzer
        self._replacements = dict(replacements)
        self._nodes: dict[tuple[str, bytes], Node] = {}
        self._schemas: dict[tuple[str, bytes], tuple[Field, ...]] = {}

    def rewrite(self, node: Node) -> Node:
        key = node.digest, node.node_bytes
        if key not in self._nodes:
            self._nodes[key] = self._visit(node)
        return self._nodes[key]

    def _visit(self, node: Node) -> Node:
        replacement = self._replacements.get(node.digest)
        if replacement is not None:
            return replacement
        if node.op.name in {"with_columns", "filter"}:
            target = self.rewrite(node.args[0])
            arguments = (
                target,
                *(
                    self._expression(value, node.args[0], target)
                    for value in node.args[1:]
                ),
            )
        else:
            arguments = tuple(self.rewrite(value) for value in node.args)
        return build(
            node.op.name, arguments, dict(node.attrs.entries), version=node.op.version
        )

    def _expression(self, node: Node, original: Node, target: Node) -> Node:
        if node.op.name == "column_ref":
            return self._reference(node, original, target)
        if (
            node.op.name
            not in _ROW_LOCAL_PRIMITIVES | _ROLLING_PRIMITIVES | _CROSS_SECTION
        ):
            return self.rewrite(node)
        return build(
            node.op.name,
            tuple(self._expression(child, original, target) for child in node.args),
            dict(node.attrs.entries),
            version=node.op.version,
        )

    def _exact_boundary(self, table: Node) -> bool:
        while table.op.name in _ROW_PRESERVING:
            replacement = self._replacements.get(table.digest)
            if replacement is not None:
                return self._analyzer.table(
                    table, "macd.source"
                ).schema == _schema_fields(replacement.attr("schema"))
            table = table.args[0]
        return False

    def _schema(self, table: Node) -> tuple[Field, ...]:
        key = table.digest, table.node_bytes
        if key not in self._schemas:
            if table.op.name == "table_input":
                self._schemas[key] = _schema_fields(table.attr("schema"))
            else:
                runtime = self._analyzer._runtime
                if runtime is None:
                    return ()
                self._schemas[key] = infer_table_schema(
                    table, runtime, {}, mode=self._analyzer._mode
                )
        return self._schemas[key]

    def _matching_field(self, reference: Node, original: Node, target: Node) -> bool:
        name = reference.attr("name")
        if not isinstance(name, CStr):
            return False
        source = {
            field.name: field
            for field in self._analyzer.table(
                reference.args[0], "macd.reference"
            ).schema
        }
        before = {
            field.name: field
            for field in self._analyzer.table(original, "macd.target").schema
        }
        after = {field.name: field for field in self._schema(target)}
        return name.value in source and source[name.value] == before.get(
            name.value
        ) == after.get(name.value)

    def _reference(self, node: Node, original: Node, target: Node) -> Node:
        rewritten = self.rewrite(node)
        if not node.args:
            return rewritten
        if rewritten.args and (
            rewritten.args[0].digest,
            rewritten.args[0].node_bytes,
        ) == (target.digest, target.node_bytes):
            return rewritten
        if not self._exact_boundary(original) and not self._exact_boundary(
            node.args[0]
        ):
            return rewritten
        if not _indicator_reference_matches(
            node, original, {}
        ) or not self._matching_field(node, original, target):
            return rewritten
        return build(
            "column_ref", (target,), dict(node.attrs.entries), version=node.op.version
        )
