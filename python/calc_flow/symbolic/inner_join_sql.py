"""Native bounded-inner SQL declarations and local temporal passthrough."""

from __future__ import annotations

import json
from datetime import timedelta


def native_plan(node):
    from calc_flow import _native
    from calc_flow.symbolic.analyzer import _cstr_seq

    return json.loads(
        _native._inner_join_query(
            node.attr("query").value, list(_cstr_seq(node.attr("aliases")))
        )
    )


def join_node(node, plan):
    from calc_flow.join_spec import JoinStateLimits, JoinTimeBounds
    from calc_flow.symbolic.analyzer import _cstr_seq
    from calc_flow.symbolic.expr import TableExpr
    from calc_flow.symbolic.ops import table

    tables = dict(zip(_cstr_seq(node.attr("aliases")), node.args))
    spec = dict(plan["spec"])
    spec.pop("join_type")
    bounds = spec.pop("bounds")
    limits = spec.pop("limits")
    ordering = plan.get("output_ordering")
    options = (
        {}
        if ordering is None
        else {
            "output_entity_by": ordering["entity_by"],
            "output_event_time": ordering["event_time"],
            "output_sequence_by": ordering["sequence_by"],
        }
    )
    return table.stream_join(
        TableExpr(tables[plan["left_alias"]]),
        TableExpr(tables[plan["right_alias"]]),
        **spec,
        bounds=JoinTimeBounds(
            timedelta(microseconds=bounds["before_micros"]),
            timedelta(microseconds=bounds["after_micros"]),
        ),
        limits=JoinStateLimits(**limits),
        **options,
    )._node


def passthrough_facts(node, inputs, empty):
    from dataclasses import replace

    from calc_flow.symbolic.event_window_sql import passthrough

    columns = passthrough(node)
    if columns is None or len(inputs) != 1:
        return empty
    identity = _passthrough_identity(next(iter(inputs.values())), columns)
    if identity is None:
        return empty
    event, entity, sequence = identity
    return replace(empty, event_time=event, entity_by=entity, sequence_by=sequence)


def _passthrough_identity(child, columns):
    reverse = {source: output for output, source in columns.items()}

    def mapped(name):
        return reverse.get(name, name if "*" in columns else None)

    entity = tuple(mapped(name) for name in child.entity_by)
    sequence = tuple(mapped(name) for name in child.sequence_by)
    event = mapped(child.event_time) if child.event_time else None
    if not event or not all(entity) or not all(sequence):
        return None
    return event, entity, sequence


def temporal_sql_paths(program):
    from calc_flow.symbolic.lower.sql import _walk_nodes

    paths = set()
    for _, value in program.outputs:
        for node in _walk_nodes(value._node):
            if _is_inner_join(node):
                paths.update(_ancestor_sql_paths(node))
    return paths


def _is_inner_join(node):
    return node.op.name == "stream_join" or (
        node.op.name == "sql" and native_plan(node) is not None
    )


def _ancestor_sql_paths(node):
    from calc_flow.symbolic.lower.sql import _walk_nodes

    return {
        child.digest
        for arg in node.args
        for child in _walk_nodes(arg)
        if child.op.name == "sql"
    }


def cache_policy(program):
    from calc_flow.symbolic.lower.sql import _walk_nodes

    return (
        ("finite_inner_join_v1",)
        if any(
            node.op.name == "sql" and native_plan(node) is not None
            for _, value in program.outputs
            for node in _walk_nodes(value._node)
        )
        else ()
    )


def time_is_original(node, name, analyzer):
    from calc_flow.symbolic.event_window_sql import time_is_input

    if node.digest in analyzer._sql_join_nodes:
        node = analyzer._sql_join_nodes[node.digest]
    if node.op.name == "stream_join":
        return (
            node.attr("output_event_time").value == name
            if node.op.version >= 2
            else False
        )
    if node.op.name == "sql":
        return _sql_time_is_original(node, name, analyzer)
    if node.op.name in {"project", "filter"}:
        return time_is_original(node.args[0], name, analyzer)
    if node.op.name == "with_columns":
        return _derived_time_is_original(node, name, analyzer)
    return time_is_input(node, name)


def _sql_time_is_original(node, name, analyzer):
    from calc_flow.symbolic.event_window_sql import passthrough

    columns = passthrough(node)
    if columns is None:
        return False
    original = columns.get(name, name if "*" in columns else None)
    return original is not None and time_is_original(node.args[0], original, analyzer)


def _derived_time_is_original(node, name, analyzer):
    from calc_flow.symbolic.analyzer import _cstr_seq

    names = _cstr_seq(node.attr("names"))
    if name not in names:
        return time_is_original(node.args[0], name, analyzer)
    value = node.args[names.index(name) + 1]
    return value.op.name == "column_ref" and time_is_original(
        value.args[0], value.attr("name").value, analyzer
    )
