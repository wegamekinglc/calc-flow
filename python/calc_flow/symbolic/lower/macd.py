"""Plan a shared MACD dependency prefix before named output branches."""

from __future__ import annotations

import hashlib
from dataclasses import replace

from calc_flow._compat import dataclass
from calc_flow._macd import _MACD_PARAMETER_NAMES
from calc_flow.pipeline import _canonical
from calc_flow.symbolic.analyzer import _schema_fields
from calc_flow.symbolic.lower.planners import (
    _plan_rolling,
    _plan_rolling_with_fields,
    _reserved_rolling_pipeline_ids,
    _rewrite_pipeline_environment,
    _rolling_depth,
    _share_rolling_plans,
    _shared_pipeline_stages,
    _unique_shared_node_id,
)
from calc_flow.symbolic.lower.segments import (
    _ROLLING_PRIMITIVES,
    _cint,
    _cstr,
    _RollingPipeline,
    _RollingPlan,
    _Segment,
)
from calc_flow.symbolic.nodes import Node

_NodeKey = tuple[str, bytes]
_OwnerKey = tuple[object, ...]


@dataclass(frozen=True, slots=True)
class _MacdGroup:
    context: str
    members: tuple[str, ...]
    components: tuple[Node, ...]


def _node_key(node: Node) -> _NodeKey:
    return node.digest, node.node_bytes


def _context(segment: _Segment) -> str:
    predicate = segment.predicate
    return _canonical(
        {
            "source": [segment.input_node.digest, segment.input_node.node_bytes.hex()],
            "predicate": None
            if predicate is None
            else [predicate.digest, predicate.node_bytes.hex()],
        }
    )


def _forest(roots: tuple[Node, ...]) -> dict[_NodeKey, Node]:
    found: dict[_NodeKey, Node] = {}
    pending = list(reversed(roots))
    while pending:
        node = pending.pop()
        key = _node_key(node)
        if key in found:
            continue
        found[key] = node
        if node.op.name != "column_ref":
            pending.extend(reversed(node.args))
    return found


def _segment_forest(segment: _Segment) -> dict[_NodeKey, Node]:
    roots = tuple(tree for _, tree in segment.env)
    if segment.post_predicate is not None:
        roots += (segment.post_predicate,)
    return _forest(roots)


def _owner(node: Node) -> _OwnerKey:
    argument = node.args[0]
    return (
        node.op.name,
        node.op.version,
        1,
        argument.digest,
        argument.node_bytes,
        *(_cint(node.attr(name)) for name in _MACD_PARAMETER_NAMES),
    )


def _collect_owners(
    segments: list[tuple[str, _Segment]],
) -> dict[str, dict[_OwnerKey, dict[str, dict[_NodeKey, Node]]]]:
    contexts: dict[str, dict[_OwnerKey, dict[str, dict[_NodeKey, Node]]]] = {}
    for name, segment in segments:
        owners = contexts.setdefault(_context(segment), {})
        for key, node in _segment_forest(segment).items():
            if node.op.name == "macd_component":
                owners.setdefault(_owner(node), {}).setdefault(name, {})[key] = node
    return contexts


def _shared_components(
    owners: dict[_OwnerKey, dict[str, dict[_NodeKey, Node]]],
) -> tuple[Node, ...]:
    components: dict[_NodeKey, Node] = {}
    for outputs in owners.values():
        if len(outputs) < 2:
            continue
        for values in outputs.values():
            components.update(values)
    return tuple(components[key] for key in sorted(components))


def _group_members(
    context: str, components: tuple[Node, ...], segments: list[tuple[str, _Segment]]
) -> tuple[str, ...]:
    dependencies = {
        key
        for key, node in _forest(components).items()
        if node.op.name in _ROLLING_PRIMITIVES
    }
    return tuple(
        name
        for name, segment in segments
        if _context(segment) == context
        and not dependencies.isdisjoint(_segment_forest(segment))
    )


def _groups(segments: list[tuple[str, _Segment]]) -> tuple[_MacdGroup, ...]:
    groups: list[_MacdGroup] = []
    for context, owners in _collect_owners(segments).items():
        components = _shared_components(owners)
        if components:
            groups.append(
                _MacdGroup(
                    context, _group_members(context, components, segments), components
                )
            )
    return tuple(groups)


def _prefix_name(group: _MacdGroup, reserved: set[str]) -> str:
    identity = _canonical(
        {
            "context": group.context,
            "components": [
                (node.digest, node.node_bytes.hex()) for node in group.components
            ],
        }
    )
    stem = "cf_macd_share_" + hashlib.sha256(identity.encode()).hexdigest()
    candidate, ordinal = stem, 0
    while any(
        name == candidate or name.startswith(candidate + "__") for name in reserved
    ):
        ordinal += 1
        candidate = f"{stem}_{ordinal}"
    reserved.add(candidate)
    return candidate


def _prefix(
    group: _MacdGroup, segment: _Segment, settings: tuple[int, str], reserved: set[str]
) -> _RollingPipeline:
    name = _prefix_name(group, reserved)
    env = tuple(
        (f"{name}__component_{index}", node)
        for index, node in enumerate(group.components)
    )
    source = replace(
        segment, fields=tuple(field for field, _ in env), env=env, post_predicate=None
    )
    result = _plan_rolling(name, source, "shared.macd", *settings)
    if result is None:
        raise RuntimeError("shared MACD dependency slice has no state")
    return replace(result, stages=_shared_pipeline_stages(name, result, reserved))


def _needed_stages(segment: _Segment, group: _MacdGroup) -> int:
    dependencies = _forest(group.components)
    return max(
        _rolling_depth(node)
        for key, node in _segment_forest(segment).items()
        if key in dependencies and node.op.name in _ROLLING_PRIMITIVES
    )


def _branch(
    name: str,
    segment: _Segment,
    group: _MacdGroup,
    prefix: _RollingPipeline,
    settings: tuple[int, str],
) -> _RollingPipeline:
    stages = prefix.stages[: _needed_stages(segment, group)]
    env, predicate = _rewrite_pipeline_environment(segment, stages)
    remaining = replace(segment, env=env, predicate=None, post_predicate=predicate)
    tail = _plan_rolling_with_fields(
        (name, f"outputs.{name}", *settings), remaining, stages[-1].output_fields
    )
    if tail is not None:
        return _RollingPipeline(
            (*stages, *tail.stages),
            tail.env,
            tail.post_predicate,
            tail.input_field_names,
            tail.output_fields,
            prefix.stages[0].node_id,
        )
    final = stages[-1]
    return _RollingPipeline(
        stages,
        env,
        predicate,
        final.input_field_names,
        final.output_fields,
        prefix.stages[0].node_id,
    )


def _reserved_names(
    segments: list[tuple[str, _Segment]], plans: dict[str, _RollingPipeline | None]
) -> set[str]:
    names = _reserved_rolling_pipeline_ids(segments, plans)
    for _, segment in segments:
        names.update(
            field.name for field in _schema_fields(segment.input_node.attr("schema"))
        )
        names.update(segment.fields)
        names.update(name for name, _ in segment.env)
    return names


def _ordinary_plans(
    segments: list[tuple[str, _Segment]], settings: tuple[int, str]
) -> dict[str, _RollingPipeline | None]:
    return {
        name: _plan_rolling(name, segment, f"outputs.{name}", *settings)
        for name, segment in segments
    }


def _share_remaining_plans(
    segments: list[tuple[str, _Segment]],
    plans: dict[str, _RollingPipeline | None],
    groups: tuple[_MacdGroup, ...],
) -> dict[str, _RollingPipeline | None]:
    selected = {name for group in groups for name in group.members}
    return _share_rolling_plans(
        segments,
        {name: None if name in selected else plan for name, plan in plans.items()},
    )


def _fixed_stage_ids(stage: _RollingPlan) -> tuple[str, ...]:
    names = (stage.node_id,)
    if stage.macd_input_identity is None and stage.materialization_node_id is not None:
        names += (stage.materialization_node_id,)
    return names


def _materialization_reserved(
    segments: list[tuple[str, _Segment]], plans: dict[str, _RollingPipeline | None]
) -> set[str]:
    names = {name for name, _ in segments}
    names.update(_cstr(segment.input_node.attr("name")) for _, segment in segments)
    for plan in plans.values():
        if plan is not None:
            for stage in plan.stages:
                names.update(_fixed_stage_ids(stage))
    return names


class _MacdMaterializationIds:
    """Own collision-safe physical names for canonical input materializations."""

    def __init__(self, reserved: set[str]) -> None:
        self._reserved = set(reserved)
        self._aliases: dict[tuple[str, str, str], str] = {}

    def stage(self, stage: _RollingPlan) -> _RollingPlan:
        node = stage.materialization_node
        if stage.macd_input_identity is None or node is None:
            return stage
        preferred = str(node["id"])
        identity = preferred, stage.macd_input_identity, _canonical(node)
        if identity not in self._aliases:
            self._aliases[identity] = _unique_shared_node_id(preferred, self._reserved)
        allocated = self._aliases[identity]
        return replace(
            stage,
            materialization_node_id=allocated,
            materialization_node={**node, "id": allocated},
        )


def _allocate_macd_materializations(
    segments: list[tuple[str, _Segment]], plans: dict[str, _RollingPipeline | None]
) -> dict[str, _RollingPipeline | None]:
    if not any(
        stage.macd_input_identity is not None
        for plan in plans.values()
        if plan is not None
        for stage in plan.stages
    ):
        return plans
    names = _MacdMaterializationIds(_materialization_reserved(segments, plans))
    return {
        name: None
        if plan is None
        else replace(plan, stages=tuple(names.stage(stage) for stage in plan.stages))
        for name, plan in plans.items()
    }


def _plan_rolling_program(
    segments: list[tuple[str, _Segment]], settings: tuple[int, str]
) -> dict[str, _RollingPipeline | None]:
    plans = _ordinary_plans(segments, settings)
    groups = _groups(segments)
    if not groups:
        return _allocate_macd_materializations(
            segments, _share_rolling_plans(segments, plans)
        )
    shared = _share_remaining_plans(segments, plans, groups)
    by_name = dict(segments)
    reserved = _reserved_names(segments, plans)
    for group in groups:
        prefix = _prefix(group, by_name[group.members[0]], settings, reserved)
        for name in group.members:
            shared[name] = _branch(name, by_name[name], group, prefix, settings)
    return _allocate_macd_materializations(segments, shared)
