"""Static schema, temporal identity and capability checks for ASOF."""

from __future__ import annotations

from typing import TYPE_CHECKING

from calc_flow.capabilities import ProviderPort
from calc_flow.symbolic.types import Field

if TYPE_CHECKING:
    from calc_flow.asof_join_spec import AsofJoinSide, AsofJoinSpec
    from calc_flow.symbolic.analyzer import TableFacts, _Analyzer
    from calc_flow.symbolic.nodes import Node

_SEQUENCE_TYPES = frozenset(
    {
        "int8",
        "int16",
        "int32",
        "int64",
        "uint8",
        "uint16",
        "uint32",
        "uint64",
        "string",
        "large_string",
    }
)
_KEY_TYPES = _SEQUENCE_TYPES | {"bool", "date32", "date64"}
_TIME_TYPE = "timestamp[us, UTC]"


def _capability(analyzer: _Analyzer, path: str) -> None:
    if analyzer._mode != "stream":
        analyzer.issue(
            path,
            "unsupported_mode",
            "stream_asof_join is available only in stream mode",
        )
        return
    capabilities = analyzer._capabilities
    if capabilities is None:
        return
    offered = tuple(
        operator
        for operator in capabilities.operators
        if operator.kind == "stream_asof_join"
    )
    expected = {
        "version": "1",
        "modes": ("stream",),
        "finality": "group_final_append_only",
        "requires_datafusion": True,
        "stateful": True,
        "microbatch_invariant": True,
        "requires_watermark": True,
        "checkpoint_support": "checkpointed_stateful",
        "state_version": 1,
        "state_layouts": (1,),
        "deterministic": True,
        "replay_safe": True,
        "input_ports": (
            ProviderPort("left", "table", True),
            ProviderPort("right", "table", True),
        ),
        "output_ports": (ProviderPort("output", "table", True),),
    }
    if len(offered) == 1 and all(
        getattr(offered[0], name) == value for name, value in expected.items()
    ):
        return
    analyzer.issue(
        path,
        "capability_mismatch",
        "stream_asof_join requires native stream_asof_join@1 with the exact "
        "finality, ports, state layout and replay contract",
    )


def _identity_field(
    analyzer: _Analyzer, facts: TableFacts, name: str, path: str
) -> Field | None:
    field = next((field for field in facts.schema if field.name == name), None)
    if field is None:
        analyzer.issue(
            path, "unresolved_type", f"unknown ASOF identity column {name!r}"
        )
    elif field.nullable:
        analyzer.issue(
            path, "type_mismatch", f"ASOF identity column {name!r} must be non-null"
        )
    return field


def _declared_side_metadata(
    analyzer: _Analyzer, facts: TableFacts, side: AsofJoinSide, path: str
) -> None:
    for name, requested, declared in (
        ("event_time", side.event_time, facts.event_time),
        ("sequence_by", tuple(side.sequence_by), facts.sequence_by),
    ):
        if not declared or requested != declared:
            analyzer.issue(
                f"{path}.{name}",
                "ordering_required",
                f"ASOF {name} must match current declared input metadata",
            )


def _side_metadata(
    analyzer: _Analyzer, facts: TableFacts, side: AsofJoinSide, path: str
) -> None:
    _declared_side_metadata(analyzer, facts, side, path)
    time = _identity_field(analyzer, facts, side.event_time, f"{path}.event_time")
    if time is not None and time.data_type != _TIME_TYPE:
        analyzer.issue(
            f"{path}.event_time",
            "ordering_required",
            "ASOF event time must have exact type timestamp[us, UTC]",
        )
    for index, name in enumerate(side.sequence_by):
        location = f"{path}.sequence_by[{index}]"
        field = _identity_field(analyzer, facts, name, location)
        if field is not None and field.data_type not in _SEQUENCE_TYPES:
            analyzer.issue(
                location,
                "type_mismatch",
                "ASOF sequence requires an integer or UTF-8 total-order type",
            )


def _key_types(
    analyzer: _Analyzer, fields: tuple[Field, Field], path: str, index: int
) -> None:
    left, right = fields
    if left.data_type != right.data_type:
        analyzer.issue(
            f"{path}.left.keys[{index}]",
            "type_mismatch",
            "ASOF key pairs require identical Arrow types",
        )
    for side, field in zip(("left", "right"), fields, strict=True):
        if field.data_type not in _KEY_TYPES and not field.data_type.startswith(
            "timestamp["
        ):
            analyzer.issue(
                f"{path}.{side}.keys[{index}]",
                "type_mismatch",
                "ASOF key requires a supported exact Arrow type",
            )


def _keys(
    analyzer: _Analyzer,
    facts: tuple[TableFacts, TableFacts],
    spec: AsofJoinSpec,
    path: str,
) -> None:
    for index, names in enumerate(zip(spec.left.keys, spec.right.keys, strict=True)):
        fields = tuple(
            _identity_field(analyzer, table, name, f"{path}.{side}.keys[{index}]")
            for side, table, name in zip(("left", "right"), facts, names, strict=True)
        )
        left, right = fields
        if left is not None and right is not None:
            _key_types(analyzer, (left, right), path, index)


def validate_asof(
    analyzer: _Analyzer,
    node: Node,
    facts: tuple[TableFacts, TableFacts],
    spec: AsofJoinSpec,
    path: str,
) -> None:
    role = f"{path}.stream_asof_join"
    _capability(analyzer, role)
    for index, (side, table, declaration) in enumerate(
        zip(("left", "right"), facts, (spec.left, spec.right), strict=True)
    ):
        _input_path(analyzer, node.args[index], f"{role}.{side}")
        _side_metadata(analyzer, table, declaration, f"{role}.{side}")
    _keys(analyzer, facts, spec, role)


def _input_path(analyzer: _Analyzer, node: Node, path: str) -> None:
    from calc_flow.symbolic.analyzer import _ROW_LOCAL_PRIMITIVES

    seen: set[str] = set()
    pending = [node]
    while pending:
        current = pending.pop()
        if current.digest in seen:
            continue
        seen.add(current.digest)
        if current.op.name in {"table_input", "stream_join", "stream_asof_join"}:
            continue
        if current.op.name not in _ROW_LOCAL_PRIMITIVES | {
            "project",
            "filter",
            "with_columns",
        }:
            analyzer.issue(
                path,
                "capability_mismatch",
                "ASOF operands support row-local transformations and declared "
                "join boundaries; "
                f"found {current.op.name}@{current.op.version}",
            )
            continue
        pending.extend(current.args)
