"""ASOF declarations and data-only specification resolution."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from datetime import timedelta
from typing import TYPE_CHECKING, Literal, TypedDict

from calc_flow.asof_join_spec import (
    AsofJoinSide,
    AsofJoinSpec,
    AsofStateLimits,
    _asof_wire_spec,
    _identity_columns,
)
from calc_flow.symbolic.nodes import CInt, CMap, CSeq, CStr, CValue, Node, build

if TYPE_CHECKING:
    from calc_flow.symbolic.analyzer import TableFacts, _Analyzer
    from calc_flow.symbolic.expr import TableExpr
    from calc_flow.symbolic.types import Field


class _AsofJoinOptions(TypedDict, total=False):
    keys: tuple[Sequence[str], Sequence[str]] | None
    late_policy: Literal["error", "drop"]
    prefixes: tuple[str, str]


@dataclass(frozen=True, slots=True)
class _AsofOptions:
    keys: tuple[Sequence[str], Sequence[str]] | None = None
    late_policy: Literal["error", "drop"] = "error"
    prefixes: tuple[str, str] = ("left", "right")

    def __post_init__(self) -> None:
        for name, value in (("keys", self.keys), ("prefixes", self.prefixes)):
            if value is not None and (type(value) is not tuple or len(value) != 2):
                raise TypeError(f"{name} must be a tuple of exactly two sides")
        if self.keys is not None:
            object.__setattr__(
                self,
                "keys",
                tuple(
                    _identity_columns(values, f"keys[{index}]")
                    for index, values in enumerate(self.keys)
                ),
            )


def _canonical(value: object) -> CValue:
    if isinstance(value, dict):
        return CMap.from_mapping(
            {name: _canonical(item) for name, item in value.items()}
        )
    if isinstance(value, list):
        return CSeq(tuple(_canonical(item) for item in value))
    return CInt(value) if type(value) is int else CStr(value)


def _wire(value: CValue) -> object:
    if isinstance(value, CMap):
        return {name: _wire(item) for name, item in value.entries}
    if isinstance(value, CSeq):
        return [_wire(item) for item in value.items]
    if isinstance(value, (CInt, CStr)):
        return value.value
    raise ValueError("ASOF specification must contain strict data")


def _node_spec(node: Node) -> AsofJoinSpec:
    wire = _wire(node.attr("spec"))
    required = {"left", "right", "tolerance_micros", "limits", "late_policy"}
    if not isinstance(wire, dict) or set(wire) != required:
        raise ValueError(
            "ASOF spec requires exactly left, right, tolerance_micros, "
            "limits and late_policy"
        )
    if type(wire["tolerance_micros"]) is not int:
        raise TypeError("ASOF tolerance_micros must be an exact int")
    return AsofJoinSpec(
        AsofJoinSide(**wire["left"]),
        AsofJoinSide(**wire["right"]),
        timedelta(microseconds=wire["tolerance_micros"]),
        AsofStateLimits(**wire["limits"]),
        wire["late_policy"],
    )


def _operand_facts(value: TableExpr, side: str) -> TableFacts:
    from calc_flow.symbolic.analyzer import _Analyzer
    from calc_flow.symbolic.expr import TableExpr
    from calc_flow.symbolic.types import TABLE_FIELD_TYPES

    if not isinstance(value, TableExpr):
        raise TypeError(f"{side} must be a calc_flow.TableExpr")
    analyzer = _Analyzer("stream", frozenset(), TABLE_FIELD_TYPES, False)
    facts = analyzer.table(value._node, side)
    for name, declared in (
        ("event_time", facts.event_time),
        ("sequence_by", facts.sequence_by),
    ):
        if not declared:
            raise ValueError(
                f"{side}.{name}: ordering_required: {side} input requires declared "
                f"{name} metadata for stream_asof_join"
            )
    return facts


def declare_asof(
    left: TableExpr,
    right: TableExpr,
    tolerance: timedelta,
    limits: AsofStateLimits,
    options: _AsofOptions,
) -> TableExpr:
    """Resolve temporal facts once and declare the native ASOF primitive."""
    from calc_flow.symbolic.expr import TableExpr

    left_facts = _operand_facts(left, "left")
    right_facts = _operand_facts(right, "right")
    effective = (
        (left_facts.entity_by, right_facts.entity_by)
        if options.keys is None
        else options.keys
    )
    spec = AsofJoinSpec(
        AsofJoinSide(
            effective[0],
            left_facts.event_time,
            left_facts.sequence_by,
            options.prefixes[0],
        ),
        AsofJoinSide(
            effective[1],
            right_facts.event_time,
            right_facts.sequence_by,
            options.prefixes[1],
        ),
        tolerance,
        limits,
        options.late_policy,
    )
    return TableExpr(
        build(
            "stream_asof_join",
            (left._node, right._node),
            {"spec": _canonical(_asof_wire_spec(spec))},
        )
    )


def _output_schema(
    operands: tuple[TableFacts, TableFacts], spec: AsofJoinSpec
) -> tuple[Field, ...]:
    from calc_flow.symbolic.types import Field

    return tuple(
        Field(
            f"{side.prefix}__{field.name}",
            field.data_type,
            field.nullable if index == 0 else True,
        )
        for index, (facts, side) in enumerate(
            zip(operands, (spec.left, spec.right), strict=True)
        )
        for field in facts.schema
    )


def analyze_asof(analyzer: _Analyzer, node: Node, path: str) -> TableFacts:
    from calc_flow.symbolic.analyzer import TableFacts

    if len(node.args) != 2:
        analyzer.issue(
            f"{path}.stream_asof_join",
            "invalid_literal",
            "ASOF requires exactly two table operands",
        )
        return TableFacts((), node.digest, frozenset(), None, (), ())
    left = analyzer.table(node.args[0], f"{path}.stream_asof_join.left")
    right = analyzer.table(node.args[1], f"{path}.stream_asof_join.right")
    try:
        spec = _node_spec(node)
    except (TypeError, ValueError, OverflowError) as error:
        analyzer.issue(f"{path}.stream_asof_join.spec", "invalid_literal", str(error))
        return TableFacts((), node.digest, left.state | right.state, None, (), ())
    from calc_flow.symbolic.asof_analysis import validate_asof

    validate_asof(analyzer, node, (left, right), spec, path)
    schema = _output_schema((left, right), spec)
    names = tuple(field.name for field in schema)
    if len(set(names)) != len(names):
        analyzer.issue(
            f"{path}.stream_asof_join.schema",
            "duplicate_name",
            "ASOF prefixes produce an output field collision",
        )
    return TableFacts(
        schema,
        node.digest,
        left.state | right.state | {"stream_asof_join"},
        f"{spec.left.prefix}__{spec.left.event_time}",
        tuple(f"{spec.left.prefix}__{key}" for key in spec.left.keys),
        tuple(f"{spec.left.prefix}__{key}" for key in spec.left.sequence_by),
    )
