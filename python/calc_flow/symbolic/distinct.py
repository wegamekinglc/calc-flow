"""Final DISTINCT declarations; execution and schema authority live in Rust."""

from __future__ import annotations

import json
from collections.abc import Sequence
from typing import Literal

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
        late_policy: Literal["error", "drop"] = "error",
    ) -> DistinctSpec:
        """Accept [start,end), then finalize at a proved watermark or real EOF.

        The local policy defaults to error. Drop records checkpointed metrics
        for post-close members without extending history or emitting again.
        Global compile options never override this selection.
        """
        validate_window(event_time, start_micros, end_micros, late_policy)
        return cls._create(
            dict(
                kind="closed_window",
                event_time_column=event_time,
                start_micros=start_micros,
                end_micros=end_micros,
            ),
            state_budget,
            late_policy,
        )

    @classmethod
    def _create(cls, history, budget, late_policy="error"):
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
                    **(
                        {"late_policy": {"kind": "drop", "metrics_version": 1}}
                        if late_policy == "drop"
                        else {}
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
    columns = key_columns(columns)
    return build(
        "distinct",
        (value,),
        {"columns": CSeq(tuple(CStr(c) for c in columns)), "spec": CStr(spec._wire)},
    )


def key_columns(columns):
    if not isinstance(columns, Sequence) or isinstance(columns, (str, bytes)):
        raise TypeError("columns must be a non-empty sequence of strings")
    columns = tuple(columns)
    if not columns or any(not isinstance(c, str) or not c for c in columns):
        raise ValueError("columns must be non-empty strings")
    if len(set(columns)) != len(columns):
        raise ValueError("columns must be unique")
    return columns


def validate_window(event_time, start, end, policy):
    if not isinstance(event_time, str) or not event_time:
        raise ValueError("event_time_column must be a non-empty string")
    validate_bounds(start, end)
    if type(policy) is not str:
        raise TypeError("late_policy must be a string")
    if policy not in ("error", "drop"):
        raise ValueError("late_policy must be 'error' or 'drop'")


def validate_bounds(start, end):
    for value in (start, end):
        if type(value) is not int or not -(1 << 63) <= value < (1 << 63):
            raise ValueError("window bounds must be signed 64-bit integer microseconds")
    if start >= end:
        raise ValueError("start_micros must be less than end_micros")


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
    from calc_flow.symbolic.analyzer import TableFacts, _SQLRowOrigin

    role = role or f"{path}.distinct"
    spec = json.loads(node.attr("spec").value)
    exact = input_schema(analyzer, node, role, spec)
    analyzer._distinct_input_schemas[node.digest] = exact
    selected = selected_schema(analyzer, node, role, exact)
    check_capability(analyzer, role, required_layout(spec))
    return TableFacts(
        selected, _SQLRowOrigin(node.digest), frozenset({"distinct"}), None, (), ()
    )


def input_schema(analyzer, node, role, spec):
    from calc_flow.symbolic.event_window_sql import time_is_input

    child = analyzer.table(node.args[0], f"{role}.value")
    analyzer._window_input_path(node.args[0], f"{role}.value", set())
    history = spec["history"]
    if history["kind"] == "closed_window" and not time_is_input(
        node.args[0], history["event_time_column"]
    ):
        analyzer.issue(
            f"{role}.spec.history",
            "capability_mismatch",
            "window time must be an unchanged input column or pure rename",
        )
    if analyzer._issues:
        return child.schema
    return analyzer._native_row_schema(node.args[0], child.schema, f"{role}.value")


def selected_schema(analyzer, node, role, exact):
    from calc_flow.symbolic.analyzer import _cstr_seq

    selected = tuple(
        f for name in _cstr_seq(node.attr("columns")) for f in exact if f.name == name
    )
    if analyzer._issues or analyzer._runtime is None:
        return selected
    return native_schema(analyzer, node, role, exact, selected)


def native_schema(analyzer, node, role, exact, selected):
    from calc_flow import _native
    from calc_flow.symbolic.analyzer import _cstr_seq
    from calc_flow.symbolic.lower.schema import _arrow_schema, _fields

    try:
        return _fields(
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
    return selected


def required_layout(spec):
    return 3 if spec.get("late_policy", {}).get("kind") == "drop" else 2


def check_capability(analyzer, role, layout):
    capability = (
        next(
            (c for c in analyzer._capabilities.operators if c.kind == "distinct"), None
        )
        if analyzer._capabilities
        else None
    )
    if (
        capability is None
        or not checkpoint_capability(capability, layout)
        or analyzer._mode not in capability.modes
    ):
        analyzer.issue(
            role,
            "capability_mismatch",
            "requires native distinct@1 with the local policy's layout "
            "and copy_visible_v2",
        )


def checkpoint_capability(capability, layout):
    return (
        capability.version == "1"
        and capability.state_version == 1
        and capability.checkpoint_support == "checkpointed_stateful"
        and layout in capability.state_layouts
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


def policy_explanations(node):
    from calc_flow.symbolic.lower.sql import _walk_nodes

    specs = (node_spec(current) for current in _walk_nodes(node))
    return sorted({policy_text(spec) for spec in specs if spec is not None})


def policy_text(spec):
    layout = required_layout(spec)
    policy = "drop" if layout == 3 else "error"
    return f"local DISTINCT late_policy={policy} layout={layout}"


def check_lateness(program, mode, late_policy, allowed_lateness):
    """Global options configure only independent unmarked temporal consumers."""
    if mode != "stream" or unmarked_consumers(program):
        return
    from calc_flow.symbolic.lower.sql import _walk_nodes

    for _, value in program.outputs:
        for node in _walk_nodes(value._node):
            spec = node_spec(node)
            if spec is not None and spec["history"]["kind"] == "closed_window":
                reject_unused_options(program, late_policy, allowed_lateness)


def unmarked_consumers(program):
    from calc_flow.symbolic.lower.segments import (
        _CROSS_SECTION_PRIMITIVES,
        _ROLLING_PRIMITIVES,
    )

    pending = [value._node for _, value in program.outputs]
    seen = set()
    primitives = _ROLLING_PRIMITIVES | _CROSS_SECTION_PRIMITIVES
    while pending:
        node = pending.pop()
        if node.digest in seen:
            continue
        seen.add(node.digest)
        if node.op.name in {"late_output", "late_rows"}:
            continue
        if node.op.name in primitives:
            return True
        pending.extend(node.args)
    return False


def node_spec(node):
    if node.op.name == "distinct":
        return json.loads(node.attr("spec").value)
    if node.op.name == "sql":
        plan = native_plan(node)
        return plan.get("spec") if plan is not None else None
    return None


def reject_unused_options(program, late_policy, allowed_lateness):
    from calc_flow.symbolic import errors

    for name, value, default in (
        ("late_policy", late_policy, "error"),
        ("allowed_lateness_micros", allowed_lateness, 0),
    ):
        if value != default:
            errors.raise_compile(
                f"{program.name}.compile_stream.{name}",
                errors.CAPABILITY_MISMATCH,
                "no applicable rolling/cross-section consumer; select the local policy "
                "with DistinctSpec.closed_window(late_policy='drop')",
            )
