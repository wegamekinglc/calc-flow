"""Explicit complete-history ordering; Rust owns parsing, schema and execution."""

from __future__ import annotations

import json
from collections.abc import Sequence
from typing import Literal

from calc_flow._compat import dataclass
from calc_flow.runtime import StateBudget
from calc_flow.symbolic.distinct import key_columns, validate_window
from calc_flow.symbolic.nodes import CStr, build


@dataclass(frozen=True, slots=True, init=False)
class SortWindow:
    """A fixed UTC [start,end) interval; closure still requires native progress."""

    event_time: str
    start_micros: int
    end_micros: int

    def __init__(
        self, event_time: str, /, *, start_micros: int, end_micros: int
    ) -> None:
        if not isinstance(event_time, str):
            raise TypeError("SortWindow.event_time must be a string")
        if not event_time:
            raise ValueError("SortWindow.event_time must be a nonempty string")
        start = _window_bound(start_micros, "SortWindow.start_micros")
        end = _window_bound(end_micros, "SortWindow.end_micros")
        if start >= end:
            raise ValueError("SortWindow.end_micros must be greater than start_micros")
        object.__setattr__(self, "event_time", event_time)
        object.__setattr__(self, "start_micros", start)
        object.__setattr__(self, "end_micros", end)


def _window_bound(value: int, path: str) -> int:
    if type(value) is not int:
        raise TypeError(f"{path} must be an int of signed 64-bit microseconds")
    if not -(1 << 63) <= value < (1 << 63):
        raise ValueError(f"{path} must fit signed 64-bit microseconds")
    return value


@dataclass(frozen=True, slots=True, init=False)
class SortKey:
    """A direct column comparator; NULLs default to ASC last / DESC first."""

    column: str
    descending: bool
    nulls: str

    def __init__(self, column: str, /, *, descending: bool = False, nulls=None):
        if not isinstance(column, str) or not column:
            raise ValueError("sort.by.column must be a nonempty string")
        if type(descending) is not bool:
            raise TypeError("sort.by.descending must be a bool")
        nulls = ("first" if descending else "last") if nulls is None else nulls
        if nulls not in ("first", "last"):
            raise ValueError("sort.by.nulls must be 'first' or 'last'")
        object.__setattr__(self, "column", column)
        object.__setattr__(self, "descending", descending)
        object.__setattr__(self, "nulls", nulls)

    def _descriptor(self):
        return dict(
            column=self.column,
            direction="descending" if self.descending else "ascending",
            nulls=self.nulls,
        )


@dataclass(frozen=True, slots=True, init=False)
class SortSpec:
    """A final-only domain and node-owned peak budget, retaining the complete H.

    k bounds output only. Borrowed external ingress, other owners and process
    RSS are outside the quota. Explicit reproducible sequence fields are checked
    unique over H; arrival order never supplies stable ties.
    """

    _wire: str

    def __init__(self):
        raise TypeError("use SortSpec.bounded_source or SortSpec.closed_window")

    @classmethod
    def bounded_source(
        cls,
        state_budget: StateBudget,
        /,
        *,
        offset: int = 0,
        sequence_by: Sequence[str] = (),
    ) -> SortSpec:
        """Finalize once at actual EOF after bounded-source preflight."""
        return cls._create(
            {"kind": "bounded_source"}, state_budget, offset, sequence_by, "error"
        )

    @classmethod
    def closed_window(
        cls,
        window: SortWindow,
        /,
        *,
        state_budget: StateBudget,
        offset: int = 0,
        sequence_by: Sequence[str] = (),
        late_policy: Literal["error", "drop"] = "error",
    ) -> SortSpec:
        """Select [start,end); a proved watermark or normal EOF closes H."""
        if not isinstance(window, SortWindow):
            raise TypeError("SortSpec.closed_window.window must be a SortWindow")
        validate_window(
            window.event_time, window.start_micros, window.end_micros, late_policy
        )
        return cls._create(
            dict(
                kind="closed_window",
                event_time_column=window.event_time,
                start_micros=window.start_micros,
                end_micros=window.end_micros,
            ),
            state_budget,
            offset,
            sequence_by,
            late_policy,
        )

    @classmethod
    def _create(cls, history, budget, offset, sequence_by, late_policy):
        if not isinstance(budget, StateBudget):
            raise TypeError("sort.spec.state_budget must be a StateBudget")
        _limit(budget.max_rows, "sort.spec.state_budget.max_rows")
        _limit(offset, "sort.spec.offset")
        if not isinstance(sequence_by, Sequence) or isinstance(
            sequence_by, (str, bytes)
        ):
            raise TypeError("sort.spec.ties requires a sequence of column names")
        columns = key_columns(sequence_by) if sequence_by else ()
        ties = (
            dict(kind="sequence", columns=list(columns))
            if columns
            else dict(kind="unspecified")
        )
        policy = (
            dict(kind="drop", metrics_version=1)
            if late_policy == "drop"
            else dict(kind="error", scope="envelope")
        )
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
                    offset=offset,
                    ties=ties,
                    late_policy=policy,
                ),
                sort_keys=True,
                separators=(",", ":"),
            ),
        )
        return instance


def _limit(value, path):
    if type(value) is not int:
        raise TypeError(f"{path} must be an int in 0..=4294967295")
    if not 0 <= value <= (1 << 32) - 1:
        raise ValueError(f"{path} must be in 0..=4294967295")


def declaration(value, by, spec, fetch=None):
    if not isinstance(spec, SortSpec):
        raise TypeError("spec must be a SortSpec")
    by = _sort_keys(by)
    if fetch is not None:
        _limit(fetch, "top_k.k")
        _limit(fetch + json.loads(spec._wire)["offset"], "top_k.k + offset")
    return build(
        "bounded_order",
        (value,),
        {
            "by": CStr(json.dumps([key._descriptor() for key in by], sort_keys=True)),
            "fetch": CStr(json.dumps(fetch)),
            "spec": CStr(spec._wire),
        },
    )


def _sort_keys(by):
    if not isinstance(by, Sequence) or isinstance(by, (str, bytes)):
        raise TypeError("sort.by must be a nonempty sequence of SortKey")
    by = tuple(by)
    if not by or any(not isinstance(key, SortKey) for key in by):
        raise ValueError("sort.by must contain SortKey values")
    if len({key.column for key in by}) != len(by):
        raise ValueError("sort.by columns must be unique")
    return by


def native_plan(node):
    from calc_flow import _native
    from calc_flow.symbolic.analyzer import _cstr_seq

    return json.loads(
        _native._order_query(
            node.attr("query").value, list(_cstr_seq(node.attr("aliases")))
        )
    )


def cache_policy(program):
    from calc_flow.symbolic.lower.sql import _walk_nodes

    present = any(
        node.op.name == "bounded_order"
        or node.op.name == "sql"
        and native_plan(node) is not None
        for _, value in program.outputs
        for node in _walk_nodes(value._node)
    )

    return (
        ("bounded_order_v1", "copy_visible_v2", "order_accounting1", "analysis10")
        if present
        else ()
    )


def check_lateness(program, mode, late_policy, allowed_lateness):
    from calc_flow.symbolic import errors
    from calc_flow.symbolic.distinct import unmarked_consumers

    if mode != "stream" or not cache_policy(program) or unmarked_consumers(program):
        return
    for name, value, default in (
        ("late_policy", late_policy, "error"),
        ("allowed_lateness_micros", allowed_lateness, 0),
    ):
        if value != default:
            errors.raise_compile(
                f"{program.name}.compile_stream.{name}",
                errors.CAPABILITY_MISMATCH,
                "no applicable rolling/cross-section consumer; select the local "
                "policy in SortSpec.closed_window",
            )


def relation_node(node, plan, schema):
    return build(
        "bounded_order",
        (node.args[0],),
        {
            key: CStr(json.dumps(plan[key], sort_keys=True))
            for key in ("by", "fetch", "spec")
        },
    )


def analyze(analyzer, node, path, *, role=None):
    from calc_flow.symbolic.analyzer import TableFacts, _SQLRowOrigin

    role = role or f"{path}.sort"
    spec = json.loads(node.attr("spec").value)
    exact, arrow_schema = _ordering_input_schema(analyzer, node, role, spec)
    analyzer._order_input_schemas[node.digest] = exact
    selected = _selected_schema(analyzer, node, role, exact, arrow_schema)
    _require_capability(analyzer, role, spec)
    return TableFacts(
        selected, _SQLRowOrigin(node.digest), frozenset({"bounded_order"}), None, (), ()
    )


def _ordering_input_schema(analyzer, node, role, spec):
    from calc_flow.symbolic.distinct import input_schema
    from calc_flow.symbolic.lower.schema import _arrow_schema

    exact = input_schema(analyzer, node, role, spec)
    try:
        arrow_schema = _arrow_schema(exact)
    except ValueError:
        analyzer.issue(
            f"{role}.value",
            "unsupported_type",
            "ordering input has a type outside the native schema wire domain",
        )
        arrow_schema = None
    from calc_flow.symbolic.event_window_sql import time_is_input

    for index, name in enumerate(spec["ties"].get("columns", ())):
        if not time_is_input(node.args[0], name):
            analyzer.issue(
                f"{role}.spec.ties.columns[{index}]",
                "capability_mismatch",
                "sequence must be an original input field or pure rename",
            )
    return exact, arrow_schema


def _selected_schema(analyzer, node, role, exact, arrow_schema):
    from calc_flow import _native
    from calc_flow.symbolic.lower.schema import _fields

    if not analyzer._issues and analyzer._runtime is not None:
        try:
            return _fields(
                _native._order_schema(
                    node.attr("spec").value,
                    node.attr("by").value,
                    json.loads(node.attr("fetch").value),
                    arrow_schema,
                )
            )
        except (ValueError, _native.CalcFlowError) as error:
            suffix = getattr(error, "_calc_flow_order_path", "sort")
            suffix = suffix.removeprefix("operators.schema.").removeprefix("sort")
            analyzer.issue(
                f"{role}{suffix}",
                getattr(error, "_calc_flow_order_code", "unsupported_type"),
                str(error),
            )
    return exact


def _order_capability(analyzer):
    if not analyzer._capabilities:
        return None
    return next(
        (c for c in analyzer._capabilities.operators if c.kind == "bounded_order"),
        None,
    )


def _require_capability(analyzer, role, spec):
    capability = _order_capability(analyzer)
    layout = 2 if spec["late_policy"]["kind"] == "drop" else 1
    if (
        capability is None
        or capability.version != "1"
        or capability.state_version != 1
        or layout not in capability.state_layouts
        or analyzer._mode not in capability.modes
    ):
        analyzer.issue(
            role,
            "capability_mismatch",
            "requires bounded_order@1 and the selected policy layout",
        )


def projection_query(node):
    from calc_flow.symbolic.lower.segments import _cstr_seq, _quote_identifier
    from calc_flow.symbolic.lower.sql import _walk_nodes

    if node.op.name != "project" or not any(
        child.op.name == "bounded_order"
        or child.op.name == "sql"
        and native_plan(child) is not None
        for child in _walk_nodes(node.args[0])
    ):
        return None
    columns = ", ".join(
        _quote_identifier(name) for name in _cstr_seq(node.attr("columns"))
    )
    return " ".join(("SELECT", columns, "FROM input"))
