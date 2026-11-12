"""Final DISTINCT declarations; execution and schema authority live in Rust."""

from __future__ import annotations

import json
from collections.abc import Sequence

from calc_flow._compat import dataclass
from calc_flow.runtime import StateBudget
from calc_flow.symbolic.nodes import CSeq, CStr, build


@dataclass(frozen=True, slots=True, init=False)
class DistinctSpec:
    """A bounded history with ``copy_visible_v2`` input ownership.

    ``max_rows`` counts accepted history, including duplicates. ``max_bytes``
    bounds this node's owned participating peak, including query/capture scratch.
    Temporarily borrowed external ingress, other graph owners and RSS are excluded.
    """

    _wire: str

    def __init__(self):
        raise TypeError("use DistinctSpec.bounded_source or DistinctSpec.closed_window")

    @classmethod
    def bounded_source(cls, state_budget: StateBudget, /) -> DistinctSpec:
        """Finalize at real EOF after preflight proves a fixed finite source cut."""
        return cls._create({"kind": "bounded_source"}, state_budget)

    @classmethod
    def closed_window(
        cls,
        event_time: str,
        /,
        *,
        start_micros: int,
        end_micros: int,
        state_budget: StateBudget,
    ) -> DistinctSpec:
        """Accept [start,end), then finalize at a proved watermark or real EOF."""
        if not isinstance(event_time, str) or not event_time:
            raise ValueError("event_time_column must be a non-empty string")
        for value in (start_micros, end_micros):
            if type(value) is not int or not -(1 << 63) <= value < (1 << 63):
                raise ValueError(
                    "window bounds must be signed 64-bit integer microseconds"
                )
        if start_micros >= end_micros:
            raise ValueError("start_micros must be less than end_micros")
        return cls._create(
            dict(
                kind="closed_window",
                event_time_column=event_time,
                start_micros=start_micros,
                end_micros=end_micros,
            ),
            state_budget,
        )

    @classmethod
    def _create(cls, history, budget):
        if not isinstance(budget, StateBudget):
            raise TypeError("state_budget must be a StateBudget")
        instance = object.__new__(cls)
        object.__setattr__(
            instance,
            "_wire",
            json.dumps(
                dict(
                    ingress_policy="copy_visible_v2",
                    history=history,
                    state_budget=dict(
                        max_rows=budget.max_rows, max_bytes=budget.max_bytes
                    ),
                ),
                sort_keys=True,
                separators=(",", ":"),
            ),
        )
        return instance


def declaration(value, columns, spec):
    if not isinstance(spec, DistinctSpec):
        raise TypeError("spec must be a DistinctSpec")
    if not isinstance(columns, Sequence) or isinstance(columns, (str, bytes)):
        raise TypeError("columns must be a non-empty sequence of strings")
    columns = tuple(columns)
    if not columns or any(not isinstance(c, str) or not c for c in columns):
        raise ValueError("columns must be non-empty strings")
    if len(set(columns)) != len(columns):
        raise ValueError("columns must be unique")
    return build(
        "distinct",
        (value,),
        {"columns": CSeq(tuple(CStr(c) for c in columns)), "spec": CStr(spec._wire)},
    )


def native_plan(node):
    from calc_flow import _native
    from calc_flow.symbolic.analyzer import _cstr_seq

    return json.loads(
        _native._distinct_query(
            node.attr("query").value, list(_cstr_seq(node.attr("aliases")))
        )
    )


def relation_node(node, plan, schema):
    columns = plan["columns"]
    if columns is None:
        columns = [field.name for field in schema]
    return build(
        "distinct",
        (node.args[0],),
        {
            "columns": CSeq(tuple(CStr(c) for c in columns)),
            "spec": CStr(json.dumps(plan["spec"], sort_keys=True)),
        },
    )


def analyze(analyzer, node, path, *, role=None):
    from calc_flow import _native
    from calc_flow.symbolic.analyzer import TableFacts, _cstr_seq, _SQLRowOrigin
    from calc_flow.symbolic.event_window_sql import time_is_input
    from calc_flow.symbolic.lower.schema import _arrow_schema, _fields

    role = role or f"{path}.distinct"
    child = analyzer.table(node.args[0], f"{role}.value")
    analyzer._window_input_path(node.args[0], f"{role}.value", set())
    spec = json.loads(node.attr("spec").value)
    history = spec["history"]
    if history["kind"] == "closed_window" and not time_is_input(
        node.args[0], history["event_time_column"]
    ):
        analyzer.issue(
            f"{role}.spec.history",
            "capability_mismatch",
            "window time must be an unchanged input column or pure rename",
        )
    exact = child.schema
    if not analyzer._issues:
        exact = analyzer._native_row_schema(node.args[0], exact, f"{role}.value")
    analyzer._distinct_input_schemas[node.digest] = exact
    selected = tuple(
        f for name in _cstr_seq(node.attr("columns")) for f in exact if f.name == name
    )
    if not analyzer._issues and analyzer._runtime is not None:
        try:
            selected = _fields(
                _native._distinct_schema(
                    node.attr("spec").value,
                    list(_cstr_seq(node.attr("columns"))),
                    _arrow_schema(exact),
                )
            )
        except (ValueError, _native.CalcFlowError) as error:
            suffix = getattr(error, "_calc_flow_distinct_path", "distinct")
            suffix = suffix.removeprefix("operators.schema.").removeprefix("distinct")
            code = getattr(
                error, "_calc_flow_distinct_code", "input_normalization_unsupported"
            )
            analyzer.issue(f"{role}{suffix}", code, str(error))
    capability = (
        next(
            (c for c in analyzer._capabilities.operators if c.kind == "distinct"), None
        )
        if analyzer._capabilities
        else None
    )
    if (
        capability is None
        or capability.state_layouts != (2,)
        or analyzer._mode not in capability.modes
    ):
        analyzer.issue(
            role,
            "capability_mismatch",
            "requires native distinct@1 layout2 copy_visible_v2",
        )
    return TableFacts(
        selected, _SQLRowOrigin(node.digest), frozenset({"distinct"}), None, (), ()
    )


def cache_policy(program):
    from calc_flow.symbolic.lower.sql import _walk_nodes

    return (
        ("distinct_v2", "copy_visible_v2", "accounting2", "analysis9")
        if any(
            n.op.name == "distinct" or n.op.name == "sql" and native_plan(n) is not None
            for _, value in program.outputs
            for n in _walk_nodes(value._node)
        )
        else ()
    )


def has_distinct(node):
    if node.op.name == "distinct":
        return True
    if node.op.name == "sql":
        plan = native_plan(node)
        if plan is not None and "error" not in plan:
            return True
    return any(has_distinct(child) for child in node.args)


def check_lateness(program, mode, late_policy):
    """Reject a requested selection that the frozen v2 wire cannot preserve."""
    if mode != "stream" or late_policy == "error":
        return
    from calc_flow.symbolic import errors
    from calc_flow.symbolic.lower.sql import _walk_nodes

    for _, value in program.outputs:
        for node in _walk_nodes(value._node):
            spec = (
                json.loads(node.attr("spec").value)
                if node.op.name == "distinct"
                else None
            )
            if node.op.name == "sql":
                plan = native_plan(node)
                spec = plan.get("spec") if plan is not None else None
            if spec is not None and spec["history"]["kind"] == "closed_window":
                errors.raise_compile(
                    f"{program.name}.compile_stream.late_policy",
                    errors.CAPABILITY_MISMATCH,
                    "selected DISTINCT windows currently require late_policy='error'; "
                    "the frozen v2 declaration cannot preserve a drop selection",
                )
