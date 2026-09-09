"""Native ASOF nodes for the shared relational graph lowerer."""

from __future__ import annotations

from typing import TYPE_CHECKING

from calc_flow.asof_join_spec import _asof_wire_spec
from calc_flow.symbolic.asof import _node_spec
from calc_flow.symbolic.lower.segments import _field_json
from calc_flow.symbolic.nodes import Node
from calc_flow.symbolic.types import Field

if TYPE_CHECKING:
    from calc_flow.symbolic.program import Program


def asof_nodes(program: Program) -> tuple[Node, ...]:
    found: dict[str, Node] = {}
    pending = [value._node for _, value in program.outputs]
    while pending:
        node = pending.pop()
        if node.digest in found:
            continue
        found[node.digest] = node
        pending.extend(node.args)
    return tuple(
        found[digest]
        for digest in sorted(found)
        if found[digest].op.name == "stream_asof_join"
    )


def native_asof_node(
    node: Node, node_id: str, left: tuple[Field, ...], right: tuple[Field, ...]
) -> dict[str, object]:
    return {
        "id": node_id,
        "input_ports": [
            {
                "kind": "table",
                "name": name,
                "required": True,
                "schema": [_field_json(field) for field in fields],
            }
            for name, fields in (("left", left), ("right", right))
        ],
        "operator": {
            "kind": "stream_asof_join",
            "spec": _asof_wire_spec(_node_spec(node)),
        },
        "output_ports": [],
    }


def explain_asof(nodes: list[dict[str, object]]) -> tuple[str, ...]:
    asofs = [
        node
        for node in nodes
        if node.get("operator", {}).get("kind") == "stream_asof_join"
    ]
    if not asofs:
        return ()
    lines = [f"    stream_asof_join state_stages {len(asofs)}"]
    for node in asofs:
        spec = node["operator"]["spec"]
        limits = spec["limits"]
        sides = " ".join(
            f"{name}.keys={','.join(spec[name]['keys'])} "
            f"{name}.event_time={spec[name]['event_time']} "
            f"{name}.sequence_by={','.join(spec[name]['sequence_by'])}"
            for name in ("left", "right")
        )
        lines.extend(
            (
                f"    stream_asof_join {node['id']} native_version=1 "
                "backward inclusive=true "
                f"tolerance_micros={spec['tolerance_micros']} "
                f"late_policy={spec['late_policy']} {sides}",
                "    left_preserving=true right_nullable=true "
                "finality=group_final_append_only "
                "closes_at=both_watermarks>left_time_or_side_eof "
                "idle_does_not_close=true "
                "output_time=left.time frontier_lag_micros=1",
                f"    state {node['id']} state_version=1 state_layout=1 "
                f"max_state_rows={limits['max_state_rows']} "
                f"max_state_bytes={limits['max_state_bytes']} "
                f"workspace_bytes={limits['max_state_bytes']} "
                "delivery=source_and_sink_dependent",
            )
        )
    return tuple(lines)
