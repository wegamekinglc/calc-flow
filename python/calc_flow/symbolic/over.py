"""Bounded analytic declarations; Rust owns the shared evaluator."""

from __future__ import annotations

import json
from collections.abc import Sequence
from typing import Literal

from calc_flow._compat import dataclass
from calc_flow.runtime import StateBudget
from calc_flow.symbolic.distinct import validate_window
from calc_flow.symbolic.order import SortKey, _limit, _sort_keys, _window_bound


def _names(values, path, *, empty=False):
    if not isinstance(values, Sequence) or isinstance(values, (str, bytes)):
        raise TypeError(f"{path} requires a sequence of column names")
    names = tuple(values)
    if not empty and not names:
        raise ValueError(f"{path} requires nonempty column names")
    for index, name in enumerate(names):
        _name(name, f"{path}[{index}]")
    if len(set(names)) != len(names):
        raise ValueError(f"{path} requires unique names")
    return names


def _name(value, path):
    if not isinstance(value, str):
        raise TypeError(f"{path} must be a string")
    if not value:
        raise ValueError(f"{path} must be nonempty")
    return value


def _keys(values):
    path = "over.spec.order.order_by"
    if not isinstance(values, Sequence) or isinstance(values, (str, bytes)):
        raise TypeError(f"{path} requires a sequence of SortKey")
    keys = tuple(values)
    for index, key in enumerate(keys):
        if not isinstance(key, SortKey):
            raise TypeError(f"{path}[{index}] requires a SortKey")
    if not keys:
        raise ValueError(f"{path} requires nonempty SortKey values")
    if len({key.column for key in keys}) != len(keys):
        raise ValueError(f"{path} requires unique columns")
    return _sort_keys(keys)


@dataclass(frozen=True, slots=True, init=False)
class OverOrder:
    """One shared partition and total order, with reproducible sequence fields."""

    partition_by: tuple[str, ...]
    order_by: tuple[SortKey, ...]
    sequence_by: tuple[str, ...]

    def __init__(
        self,
        partition_by: Sequence[str],
        order_by: Sequence[SortKey],
        /,
        *,
        sequence_by: Sequence[str],
    ) -> None:
        parts = _names(partition_by, "over.spec.order.partition_by", empty=True)
        keys = _keys(order_by)
        seq = _names(sequence_by, "over.spec.order.sequence_by")
        all_names = parts + tuple(key.column for key in keys) + seq
        if len(set(all_names)) != len(all_names):
            raise ValueError(
                "over.spec.order requires disjoint partition/order/sequence"
            )
        object.__setattr__(self, "partition_by", parts)
        object.__setattr__(self, "order_by", keys)
        object.__setattr__(self, "sequence_by", seq)

    def _descriptor(self):
        return dict(
            partition_by=list(self.partition_by),
            order_by=[key._descriptor() for key in self.order_by],
            sequence_by=list(self.sequence_by),
        )


@dataclass(frozen=True, slots=True, init=False)
class OverWindow:
    """A fixed UTC [start,end) domain, closed by proved progress or EOF."""

    event_time: str
    start_micros: int
    end_micros: int

    def __init__(self, event_time: str, /, *, start_micros: int, end_micros: int):
        event_time = _name(event_time, "over.spec.history.event_time_column")
        start = _window_bound(start_micros, "over.spec.history.start_micros")
        end = _window_bound(end_micros, "over.spec.history.end_micros")
        validate_window(event_time, start, end, "error")
        object.__setattr__(self, "event_time", event_time)
        object.__setattr__(self, "start_micros", start)
        object.__setattr__(self, "end_micros", end)


@dataclass(frozen=True, slots=True, init=False)
class OverSpec:
    """Final-only H with a mandatory budget for the entire owned peak.

    ROWS frames bound calculation, not retained history. Borrowed ingress,
    other graph owners and process RSS are outside the node quota.
    """

    order: OverOrder
    state_budget: StateBudget
    window: OverWindow | None
    late_policy: str
    _wire: str

    def __init__(self):
        raise TypeError("use OverSpec.bounded_source or OverSpec.closed_window")

    @classmethod
    def bounded_source(
        cls, order: OverOrder, /, *, state_budget: StateBudget
    ) -> OverSpec:
        """Finalize at actual EOF of a fixed bounded source cut."""
        return cls._create(order, state_budget, None, "error")

    @classmethod
    def closed_window(
        cls,
        window: OverWindow,
        order: OverOrder,
        /,
        *,
        state_budget: StateBudget,
        late_policy: Literal["error", "drop"] = "error",
    ) -> OverSpec:
        """Select one window; local Drop records post-close members only."""
        if not isinstance(window, OverWindow):
            raise TypeError("over.spec.history requires an OverWindow")
        validate_window(
            window.event_time, window.start_micros, window.end_micros, late_policy
        )
        return cls._create(order, state_budget, window, late_policy)

    @classmethod
    def _create(cls, order, budget, window, late_policy):
        if not isinstance(order, OverOrder):
            raise TypeError("over.spec.order requires an OverOrder")
        if not isinstance(budget, StateBudget):
            raise TypeError("over.spec.state_budget requires a StateBudget")
        _limit(budget.max_rows, "over.spec.state_budget.max_rows")
        history = (
            dict(kind="bounded_source")
            if window is None
            else dict(
                kind="closed_window",
                event_time_column=window.event_time,
                start_micros=window.start_micros,
                end_micros=window.end_micros,
            )
        )
        policy = (
            dict(kind="drop", metrics_version=1)
            if late_policy == "drop"
            else dict(kind="error", scope="envelope")
        )
        instance = object.__new__(cls)
        for name, value in dict(
            order=order, state_budget=budget, window=window, late_policy=late_policy
        ).items():
            object.__setattr__(instance, name, value)
        object.__setattr__(
            instance,
            "_wire",
            json.dumps(
                dict(
                    ingress_policy="copy_visible_v2",
                    history=history,
                    order=order._descriptor(),
                    state_budget=dict(
                        max_rows=budget.max_rows, max_bytes=budget.max_bytes
                    ),
                    late_policy=policy,
                ),
                sort_keys=True,
                separators=(",", ":"),
            ),
        )
        return instance


@dataclass(frozen=True, slots=True, init=False)
class RowsFrame:
    """Inclusive ROWS through CURRENT ROW, within the accepted H partition."""

    _wire: str

    def __init__(self):
        raise TypeError("use RowsFrame.cumulative or RowsFrame.preceding")

    @classmethod
    def cumulative(cls) -> RowsFrame:
        """Include every preceding partition row and the current row."""
        return cls._create(dict(kind="cumulative"))

    @classmethod
    def preceding(cls, rows: int, /) -> RowsFrame:
        """Include at most rows preceding rows plus the current row."""
        _limit(rows, "over.frame.preceding")
        return cls._create(dict(kind="preceding", rows=rows))

    @classmethod
    def _create(cls, descriptor):
        instance = object.__new__(cls)
        object.__setattr__(instance, "_wire", json.dumps(descriptor, sort_keys=True))
        return instance


@dataclass(frozen=True, slots=True, init=False)
class AnalyticExpr:
    """An input-only ROW_NUMBER, LAG(1), or checked integer SUM declaration."""

    _wire: str

    def __init__(self):
        raise TypeError("use AnalyticExpr.row_number, lag or sum")

    @classmethod
    def row_number(cls, name: str, /) -> AnalyticExpr:
        """Number each partition from one, producing nonnullable UInt64."""
        return cls._create("row_number", name)

    @classmethod
    def lag(cls, column: str, /, *, name: str) -> AnalyticExpr:
        """Read the preceding row, respecting NULLs; partition start is NULL."""
        return cls._create("lag", name, _name(column, "over.expressions.column"))

    @classmethod
    def sum(cls, column: str, /, *, frame: RowsFrame, name: str) -> AnalyticExpr:
        """Sum non-NULL integers, rejecting any output outside Int64/UInt64."""
        if not isinstance(frame, RowsFrame):
            raise TypeError("over.expressions.frame requires a RowsFrame")
        return cls._create("sum", name, _name(column, "over.expressions.column"), frame)

    @classmethod
    def _create(cls, kind, name, column=None, frame=None):
        value = dict(kind=kind, name=_name(name, "over.expressions.name"))
        if column is not None:
            value["column"] = _name(column, "over.expressions.column")
        if frame is not None:
            value["frame"] = json.loads(frame._wire)
        instance = object.__new__(cls)
        object.__setattr__(instance, "_wire", json.dumps(value, sort_keys=True))
        return instance


def declaration(value, expressions, spec):
    from calc_flow.symbolic.nodes import CStr, build

    if not isinstance(spec, OverSpec):
        raise TypeError("over.spec requires an OverSpec")
    if not isinstance(expressions, Sequence) or isinstance(expressions, (str, bytes)):
        raise TypeError("over.expressions requires a sequence of AnalyticExpr")
    expressions = tuple(expressions)
    if not expressions:
        raise ValueError("over.expressions requires nonempty AnalyticExpr values")
    for index, expression in enumerate(expressions):
        if not isinstance(expression, AnalyticExpr):
            raise TypeError(f"over.expressions[{index}] requires an AnalyticExpr")
    descriptors = [json.loads(e._wire) for e in expressions]
    names = [e["name"] for e in descriptors]
    if len(set(names)) != len(names):
        raise ValueError("over.expressions requires unique output names")
    return build(
        "bounded_over",
        (value,),
        {
            "spec": CStr(spec._wire),
            "expressions": CStr(json.dumps(descriptors, sort_keys=True)),
        },
    )


def native_plan(node):
    from calc_flow import _native
    from calc_flow.symbolic.analyzer import _cstr_seq

    return json.loads(
        _native._over_query(
            node.attr("query").value, list(_cstr_seq(node.attr("aliases")))
        )
    )


def relation_node(node, plan, schema):
    from calc_flow.symbolic.nodes import CStr, build

    return build(
        "bounded_over",
        (node.args[0],),
        {
            key: CStr(json.dumps(plan[key], sort_keys=True))
            for key in ("spec", "expressions")
        },
    )


def analyze(analyzer, node, path, *, role=None):
    from calc_flow import _native
    from calc_flow.symbolic.analyzer import TableFacts, _SQLRowOrigin
    from calc_flow.symbolic.distinct import input_schema
    from calc_flow.symbolic.event_window_sql import time_is_input
    from calc_flow.symbolic.lower.schema import _arrow_schema, _fields

    role = role or f"{path}.over"
    spec = json.loads(node.attr("spec").value)
    exact = input_schema(analyzer, node, role, spec)
    analyzer._over_input_schemas[node.digest] = exact
    for index, name in enumerate(spec["order"]["sequence_by"]):
        if not time_is_input(node.args[0], name):
            analyzer.issue(
                f"{role}.spec.order.sequence_by[{index}]",
                "capability_mismatch",
                "sequence must be an original input field or pure rename",
            )
    selected = exact
    if not analyzer._issues and analyzer._runtime is not None:
        try:
            selected = _fields(
                _native._over_schema(
                    node.attr("spec").value,
                    node.attr("expressions").value,
                    _arrow_schema(exact),
                )
            )
        except (ValueError, _native.CalcFlowError) as error:
            suffix = (
                getattr(error, "_calc_flow_over_path", "over")
                .removeprefix("operators.schema.")
                .removeprefix("over")
            )
            analyzer.issue(
                f"{role}{suffix}",
                getattr(error, "_calc_flow_over_code", "unsupported_type"),
                str(error),
            )
    _require_capability(analyzer, role, spec)
    return TableFacts(
        selected, _SQLRowOrigin(node.digest), frozenset({"bounded_over"}), None, (), ()
    )


def _require_capability(analyzer, role, spec):
    capability = (
        next(
            (c for c in analyzer._capabilities.operators if c.kind == "bounded_over"),
            None,
        )
        if analyzer._capabilities
        else None
    )
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
            "requires bounded_over@1 and the selected policy layout",
        )


def order_descriptor(node):
    spec = json.loads(node.attr("spec").value)
    order = spec["order"]
    return dict(
        by=[
            dict(column=name, direction="ascending", nulls="last")
            for name in order["partition_by"]
        ]
        + order["order_by"],
        fetch=None,
        spec=dict(
            ingress_policy=spec["ingress_policy"],
            history=spec["history"],
            state_budget=spec["state_budget"],
            offset=0,
            ties=dict(kind="sequence", columns=order["sequence_by"]),
            late_policy=spec["late_policy"],
        ),
        analytic=dict(
            spec=spec, expressions=json.loads(node.attr("expressions").value)
        ),
    )


def cache_policy(program):
    from calc_flow.symbolic.lower.sql import _walk_nodes

    return (
        ("bounded_over_v1", "copy_visible_v2", "over_accounting1", "analysis11")
        if any(
            n.op.name == "bounded_over"
            or n.op.name == "sql"
            and native_plan(n) is not None
            for _, value in program.outputs
            for n in _walk_nodes(value._node)
        )
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
                "policy in OverSpec.closed_window",
            )


def policy_explanations(node):
    from calc_flow.symbolic.lower.sql import _walk_nodes

    descriptions = set()
    for current in _walk_nodes(node):
        if current.op.name == "bounded_over":
            spec = json.loads(current.attr("spec").value)
        elif current.op.name == "sql":
            plan = native_plan(current)
            if plan is None or "error" in plan:
                continue
            spec = plan["spec"]
        else:
            continue
        policy = spec["late_policy"]["kind"]
        descriptions.add(
            "bounded_over_v1: complete accepted H; actual bounded EOF or proved "
            "window watermark closes once; sequence is non-NULL and unique over H; "
            "ROW_NUMBER/LAG(1)/checked integer SUM with explicit ROWS frames; "
            "max_bytes bounds owned participating peak including new columns, "
            "frame/sort/output and live captures; borrowed ingress and other "
            f"graph owners are excluded; local OVER late_policy={policy} "
            f"layout={2 if policy == 'drop' else 1}"
        )
    return sorted(descriptions)
