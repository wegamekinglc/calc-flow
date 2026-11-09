"""Forward one MACD owner across proven row-preserving native boundaries."""

from __future__ import annotations

import hashlib
from collections import defaultdict
from typing import TYPE_CHECKING

from calc_flow._compat import dataclass
from calc_flow.symbolic.analyzer import _ROLLING_PRIMITIVES, _indicator_expression_nodes
from calc_flow.symbolic.expr import ArrayExpr, TableExpr
from calc_flow.symbolic.lower.macd_boundaries import _program_has_macd, _walk
from calc_flow.symbolic.lower.macd_demand import _MacdDemands
from calc_flow.symbolic.lower.macd_meaning import (
    _EXPRESSION_OPS,
    _key,
    _MacdMeanings,
    _rebuild,
    _reference,
)
from calc_flow.symbolic.nodes import CEnum, CSeq, CStr, Node, build
from calc_flow.symbolic.program import Program

if TYPE_CHECKING:
    from calc_flow.symbolic.analyzer import _Analyzer


@dataclass(frozen=True, slots=True)
class _Demand:
    expression: Node
    origin: Node
    name: str


def _owner_key(node: Node, target: Node, meanings: _MacdMeanings) -> tuple[object, ...]:
    identity = build(
        "macd_component",
        node.args,
        {**dict(node.attrs.entries), "component": CEnum("macd_component", "line")},
    )
    return meanings.state_key(identity, target)


def _shared_demands(
    evaluations: tuple[tuple[Node, Node], ...], meanings: _MacdMeanings
) -> tuple[tuple[Node, Node], ...]:
    contexts: dict[
        tuple[object, ...], dict[tuple[str, bytes], list[tuple[Node, Node]]]
    ] = defaultdict(dict)
    for expression, target in evaluations:
        for node in _indicator_expression_nodes(expression):
            if node.op.name == "macd_component":
                contexts[_owner_key(node, target, meanings)].setdefault(
                    _key(target), []
                ).append((node, target))
    return tuple(
        value
        for frames in contexts.values()
        if len(frames) > 1
        for values in frames.values()
        for value in values
    )


class _MacdForwarding:
    """Own a data-only SSA namespace and immutable table rewrite memo."""

    def __init__(self, program: Program, analyzer: _Analyzer) -> None:
        self._program = program
        self._analyzer = analyzer
        self._meanings = _MacdMeanings(analyzer)
        self._demands: dict[tuple[object, ...], _Demand] = {}
        self._by_origin: dict[tuple[str, bytes], list[_Demand]] = defaultdict(list)
        self._tables: dict[
            tuple[str, bytes], tuple[Node, dict[tuple[object, ...], str]]
        ] = {}
        self._reserved = {
            field.name
            for _, value in program.outputs
            for node in _walk(value._node)
            if node.digest in analyzer._table_cache
            for field in analyzer.table(node, "macd.reserve").schema
        }

    def _name(self, identity: tuple[object, ...]) -> str:
        stem = "cf_macd_value_" + hashlib.sha256(repr(identity).encode()).hexdigest()
        name, ordinal = stem, 0
        while name in self._reserved:
            ordinal += 1
            name = f"{stem}_{ordinal}"
        self._reserved.add(name)
        return name

    def demand(self, node: Node, target: Node) -> None:
        normalized = self._meanings.normalize(node, target)
        for child in normalized.args:
            self._dependencies(child, target)
        key = self._meanings.state_key(normalized, target)
        if key not in self._demands:
            origin = self._meanings.origin(normalized, target)
            demand = _Demand(normalized, origin, self._name(key))
            self._demands[key] = demand
            self._by_origin[_key(origin)].append(demand)

    def _dependencies(self, node: Node, target: Node) -> None:
        if node.op.name in _ROLLING_PRIMITIVES:
            self.demand(node, target)
        elif node.op.name in _EXPRESSION_OPS:
            for child in node.args:
                self._dependencies(child, target)

    def _schema_names(self, table: Node) -> tuple[str, ...]:
        return tuple(
            field.name for field in self._analyzer.table(table, "macd.schema").schema
        )

    def _trim(self, original: Node, target: Node) -> Node:
        return build(
            "project",
            (target,),
            {
                "columns": CSeq(
                    tuple(CStr(name) for name in self._schema_names(original))
                )
            },
        )

    def _expression(
        self,
        node: Node,
        context: tuple[Node, Node],
        available: dict[tuple[object, ...], str],
    ) -> Node:
        original, target = context
        if node.op.name in _ROLLING_PRIMITIVES:
            name = available.get(self._meanings.state_key(node, original))
            if name is not None:
                return _reference(target, name)
        if node.op.name == "column_ref" and _key(node.args[0]) == _key(original):
            return _reference(target, node.attr("name").value)
        if node.op.name in _EXPRESSION_OPS:
            return _rebuild(
                node,
                tuple(
                    self._expression(child, context, available) for child in node.args
                ),
            )
        return self.node(node)

    def node(self, node: Node) -> Node:
        if node.digest in self._analyzer._table_cache:
            return self.table(node)[0]
        return _rebuild(node, tuple(self.node(child) for child in node.args))

    def table(self, original: Node) -> tuple[Node, dict[tuple[object, ...], str]]:
        if original.op.name in {"late_output", "late_rows"}:
            return original, {}
        key = _key(original)
        if key not in self._tables:
            target, available = self._table(original)
            self._tables[key] = self._inject(original, target, available)
        return self._tables[key]

    def _table(self, original: Node) -> tuple[Node, dict[tuple[object, ...], str]]:
        kind = original.op.name
        if kind in {"with_columns", "filter", "project", "attach_columns"}:
            target, inherited = self.table(original.args[0])
            available = dict(inherited)
            result = self._row_table(original, target, available)
            return (
                (self._trim(original, result), {})
                if kind == "filter"
                else (result, available)
            )
        arguments = tuple(self._clean_argument(child) for child in original.args)
        return _rebuild(original, arguments), {}

    def _clean_argument(self, node: Node) -> Node:
        if node.digest in self._analyzer._table_cache:
            target, available = self.table(node)
            return self._trim(node, target) if available else target
        return self.node(node)

    def _row_table(
        self, original: Node, target: Node, available: dict[tuple[object, ...], str]
    ) -> Node:
        attrs = dict(original.attrs.entries)
        if original.op.name == "project":
            selected = tuple(item.value for item in original.attr("columns").items)
            attrs["columns"] = CSeq(
                tuple(
                    CStr(name)
                    for name in (
                        *selected,
                        *sorted(set(available.values()) - set(selected)),
                    )
                )
            )
            return build("project", (target,), attrs)
        if original.op.name == "attach_columns":
            return _rebuild(
                original, (target, *(self.node(child) for child in original.args[1:]))
            )
        args = tuple(
            self._expression(child, (original.args[0], target), available)
            for child in original.args[1:]
        )
        return _rebuild(original, (target, *args))

    def _inject(
        self, original: Node, target: Node, inherited: dict[tuple[object, ...], str]
    ) -> tuple[Node, dict[tuple[object, ...], str]]:
        available = dict(inherited)
        for demand in self._by_origin.get(_key(original), ()):
            key = self._meanings.state_key(demand.expression, original)
            if key in available:
                continue
            args = tuple(
                self._expression(child, (original, target), available)
                for child in demand.expression.args
            )
            expression = _rebuild(demand.expression, args)
            target = build(
                "with_columns",
                (target, expression),
                {"names": CSeq((CStr(demand.name),))},
            )
            available[key] = demand.name
        return target, available

    def program(self) -> Program:
        outputs = []
        for name, value in self._program.outputs:
            target = self.node(value._node)
            if isinstance(value, TableExpr):
                outputs.append((name, TableExpr(self._trim(value._node, target))))
            else:
                outputs.append((name, ArrayExpr(target)))
        return Program(
            self._program.name,
            engine=self._program.engine,
            inputs=self._program.inputs,
            outputs=outputs,
        )


def _forward_native_macd(program: Program, analyzer: _Analyzer) -> Program:
    from calc_flow.symbolic.lower.native_arrays import _has_native

    if not _program_has_macd(program) or not _has_native(program):
        return program
    forward = _MacdForwarding(program, analyzer)
    demands = _shared_demands(
        _MacdDemands(analyzer).collect(program), forward._meanings
    )
    if not demands:
        return program
    for node, target in demands:
        forward.demand(node, target)
    return forward.program()
