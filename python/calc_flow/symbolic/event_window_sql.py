"""Native event-window SQL recognition and time-coordinate proofs."""

from __future__ import annotations

import json


def native_plan(node):
    from calc_flow import _native
    from calc_flow.symbolic.analyzer import _cstr_seq

    plan = json.loads(
        _native._event_window_query(
            node.attr("query").value, list(_cstr_seq(node.attr("aliases")))
        )
    )
    return plan


def window_node(node, plan):
    spec = plan["spec"]
    geometry = dict(spec["geometry"])
    kind = geometry.pop("kind")
    from calc_flow.symbolic.expr import TableExpr
    from calc_flow.symbolic.ops import window
    from calc_flow.symbolic.windows import WindowAggregate

    return getattr(window, kind)(
        TableExpr(node.args[0]),
        event_time=spec["event_time_column"],
        group_by=spec["group_by"],
        aggregates=[WindowAggregate(**item) for item in spec["aggregates"]],
        **geometry,
    )._node


def passthrough(node):
    from calc_flow import _native
    from calc_flow.symbolic.analyzer import _cstr_seq

    aliases = _cstr_seq(node.attr("aliases"))
    if len(aliases) != 1:
        return None
    return json.loads(
        _native._sql_passthrough_columns(node.attr("query").value, aliases[0])
    )


def time_is_input(node, name):
    operation = node.op.name
    if operation == "table_input":
        return True
    if operation == "sql":
        return _sql_time_is_input(node, name)
    if operation in {"project", "filter"}:
        return time_is_input(node.args[0], name)
    if operation == "with_columns":
        return _derived_time_is_input(node, name)
    return False


def _sql_time_is_input(node, name):
    columns = passthrough(node)
    if columns is None:
        return False
    source = columns.get(name, name if "*" in columns else None)
    return source is not None and time_is_input(node.args[0], source)


def _derived_time_is_input(node, name):
    from calc_flow.symbolic.analyzer import _cstr_seq

    names = _cstr_seq(node.attr("names"))
    if name not in names:
        return time_is_input(node.args[0], name)
    value = node.args[names.index(name) + 1]
    return value.op.name == "column_ref" and time_is_input(
        value.args[0], value.attr("name").value
    )


def cache_policy(program):
    from calc_flow.symbolic.lower.sql import _walk_nodes

    return (
        ("finite_event_window_v1",)
        if any(
            node.op.name == "sql" and _window_candidate(node)
            for _, value in program.outputs
            for node in _walk_nodes(value._node)
        )
        else ()
    )


def _window_candidate(node):
    from calc_flow.errors import CalcFlowError

    try:
        return native_plan(node) is not None
    except (CalcFlowError, ValueError):
        return True
