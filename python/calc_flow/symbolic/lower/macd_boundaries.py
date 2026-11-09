"""Keep an upstream stateful filter before a consuming MACD declaration."""

from __future__ import annotations

from collections.abc import Mapping
from typing import TYPE_CHECKING

from calc_flow.symbolic.analyzer import (
    _contains_stateful_primitive,
    _indicator_expression_nodes,
)
from calc_flow.symbolic.expr import TableExpr
from calc_flow.symbolic.nodes import _EVENT_WINDOW_OPS, Node

if TYPE_CHECKING:
    from calc_flow.symbolic.program import Program

_TABLE_CHAIN = frozenset({"project", "filter", "with_columns", "attach_columns"})
_ROW_FRONTIERS = frozenset({"table_input", "project", "filter", "with_columns"})
_HARD_BOUNDARIES = frozenset(
    {
        "sql",
        "snapshot_sql",
        "attach_columns",
        "stream_join",
        "stream_asof_join",
        "late_output",
        "late_rows",
        *_EVENT_WINDOW_OPS,
    }
)


def _walk(root: Node) -> tuple[Node, ...]:
    found: dict[tuple[str, bytes], Node] = {}
    pending = [root]
    while pending:
        current = pending.pop()
        key = current.digest, current.node_bytes
        if key not in found:
            found[key] = current
            pending.extend(current.args)
    return tuple(found.values())


def _uses_macd(expression: Node) -> bool:
    return any(
        node.op.name == "macd_component"
        for node in _indicator_expression_nodes(expression)
    )


def _program_has_macd(program: Program) -> bool:
    return any(
        node.op.name == "macd_component"
        for _, value in program.outputs
        for node in _walk(value._node)
    )


def _dependency_walk(root: Node) -> tuple[Node, ...]:
    found: dict[tuple[str, bytes], Node] = {}
    pending = [root]
    while pending:
        node = pending.pop()
        key = node.digest, node.node_bytes
        if key in found:
            continue
        found[key] = node
        if node.op.name == "late_output":
            pending.append(node.args[0].args[0])
        else:
            pending.extend(node.args)
    return tuple(found.values())


def _macd_evaluation_table(node: Node) -> Node | None:
    if node.op.name not in {"with_columns", "filter"}:
        return None
    return node.args[0] if any(_uses_macd(value) for value in node.args[1:]) else None


def _stateful_filters(table: Node) -> tuple[Node, ...]:
    filters: list[Node] = []
    while table.op.name in _TABLE_CHAIN:
        if table.op.name == "filter" and _contains_stateful_primitive(table):
            filters.append(table)
        table = table.args[0]
    return tuple(filters)


def _macd_input_boundaries(program: Program) -> tuple[Node, ...]:
    boundaries: dict[tuple[str, bytes], Node] = {}
    for _, output in program.outputs:
        for node in _walk(output._node):
            table = _macd_evaluation_table(node)
            if table is not None:
                for boundary in _stateful_filters(table):
                    boundaries[boundary.digest, boundary.node_bytes] = boundary
    return tuple(boundaries[key] for key in sorted(boundaries))


def _macd_frontiers(program: Program, boundaries: tuple[Node, ...]) -> tuple[Node, ...]:
    candidates = {(node.digest, node.node_bytes): node for node in boundaries}
    candidates.update(
        ((value.digest, value._node.node_bytes), value._node)
        for _, value in program.outputs
        if isinstance(value, TableExpr)
    )
    for _, value in program.outputs:
        for node in _dependency_walk(value._node):
            for frontier in _node_frontiers(node):
                candidates[frontier.digest, frontier.node_bytes] = frontier
    return tuple(candidates[key] for key in sorted(candidates))


def _node_frontiers(node: Node) -> tuple[Node, ...]:
    own = (node,) if node.op.name in _HARD_BOUNDARIES else ()
    return (*own, *_frontier_inputs(node))


def _frontier_inputs(node: Node) -> tuple[Node, ...]:
    if node.op.name == "late_output":
        return (node.args[0].args[0],)
    if node.op.name == "late_rows":
        return ()
    if node.op.name in _HARD_BOUNDARIES | {"from_columns"}:
        return tuple(
            child
            for child in node.args
            if child.op.name in _ROW_FRONTIERS | _HARD_BOUNDARIES
        )
    return ()


def _ready_macd_frontiers(
    candidates: tuple[Node, ...],
    boundaries: tuple[Node, ...],
    materialized: Mapping[str, str],
) -> tuple[Node, ...]:
    boundary_keys = {(node.digest, node.node_bytes) for node in boundaries}
    pending = {
        node.digest
        for node in (*boundaries, *candidates)
        if node.digest not in materialized
        and (
            (node.digest, node.node_bytes) in boundary_keys
            or node.op.name in _HARD_BOUNDARIES
        )
    }
    return tuple(
        node for node in candidates if _frontier_ready(node, pending, materialized)
    )


def _frontier_ready(
    node: Node, pending: set[str], materialized: Mapping[str, str]
) -> bool:
    return node.digest not in materialized and not any(
        child.digest in pending and child.digest != node.digest
        for child in _dependency_walk(node)
    )
