"""Rolling and cross-section stage planning for the symbolic lowerer.

Moved verbatim from ``symbolic/lower.py``."""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import TYPE_CHECKING

from calc_flow.pipeline import (
    _canonical,
)
from calc_flow.symbolic import errors
from calc_flow.symbolic.analyzer import (
    _rolling_output_type,
    _schema_fields,
)
from calc_flow.symbolic.lower.segments import (
    _CROSS_SECTION_DDOF,
    _CROSS_SECTION_ORDERING,
    _CROSS_SECTION_PRIMITIVES,
    _ROLLING_DDOF_PRIMITIVES,
    _ROLLING_PAIR_PRIMITIVES,
    _ROLLING_PRIMITIVES,
    _cbool,
    _cint,
    _cnumber,
    _cstr,
    _cstr_seq,
    _expression_node,
    _field_json,
    _find_cross_section,
    _find_ready_rolling,
    _find_rolling,
    _fused_difference_outputs,
    _fused_float_leaf,
    _plan_stateful_inputs,
    _quote_identifier,
    _replace_materialized,
    _rolling_declaration_requires_ewma,
    _rolling_frame,
    _RollingPipeline,
    _RollingPlan,
    _Segment,
    _StatefulInputRequest,
)
from calc_flow.symbolic.nodes import (
    CEnum,
    CMap,
    Node,
)
from calc_flow.symbolic.types import Field

if TYPE_CHECKING:
    from calc_flow.symbolic.program import Program


# Rolling planning validates every lag/delta/aggregate occurrence with
# stable, declaration-ordered error paths before emitting the frozen node
# shape.
def _plan_rolling_stage(
    output_name: str,
    segment: _Segment,
    path: str,
    allowed_lateness_micros: int,
    late_policy: str,
    occurrences: tuple[Node, ...],
    input_fields: tuple[Field, ...],
    stage_number: int,
    stage_count: int,
    /,
) -> _RollingPlan:
    # #lizard forgives
    input_types = {field.name: field for field in input_fields}
    entity_by = _cstr_seq(segment.input_node.attr("entity_by"))
    sequence_by = _cstr_seq(segment.input_node.attr("sequence_by"))
    event_time = _cstr(segment.input_node.attr("event_time"))
    if not entity_by or not sequence_by or not event_time:
        errors.raise_compile(
            path,
            errors.ORDERING_REQUIRED,
            "rolling temporal primitives require declared entity_by,"
            " event_time, and sequence_by ordering keys on the input table",
        )

    whole_feature = {
        tree.digest: name
        for name, tree in segment.env
        if tree.op.name in _ROLLING_PRIMITIVES
    }
    fused_differences = _fused_difference_outputs(segment, occurrences)
    fused_leaf_digests = {
        argument.digest for _, tree in fused_differences for argument in tree.args
    }
    fused_roots = {tree.digest: name for name, tree in fused_differences}
    unfused_leaf_digests = {
        subtree.digest
        for _, tree in segment.env
        for subtree in _find_rolling(_replace_materialized(tree, fused_roots))
    }
    hidden_fused_leaf_digests = fused_leaf_digests - unfused_leaf_digests
    used_names = set(input_types) | {name for name, _ in segment.env}
    stage_fragment = "" if stage_count == 1 else f"{stage_number}_"
    stage_suffix = "" if stage_count == 1 else f"_{stage_number}"
    materialization = _plan_stateful_inputs(
        _StatefulInputRequest(
            output_name,
            path,
            f"roll_input_{stage_fragment}".removesuffix("_"),
            f"{output_name}__cf_rolling_input{stage_suffix}",
            "rolling",
        ),
        input_fields,
        used_names,
        tuple(
            (subtree.op.name, argument)
            for subtree in occurrences
            for argument in subtree.args
        ),
    )
    materializations = dict(materialization.names)
    state_input_fields = materialization.input_fields
    input_types = {field.name: field for field in state_input_fields}
    used_names = set(materialization.used_names)
    replacements: dict[str, str] = {}
    declarations: list[dict[str, object]] = []
    derived_fields: list[Field] = []
    for index, subtree in enumerate(occurrences):
        if subtree.digest in hidden_fused_leaf_digests:
            continue
        kind = subtree.op.name
        name = whole_feature.get(subtree.digest)
        if name is None:
            name = f"{output_name}__cf_roll_{stage_fragment}{index}"
            if name in used_names:
                errors.raise_compile(
                    f"{path}.{name}",
                    errors.DUPLICATE_NAME,
                    f"materialized rolling column {name!r} collides with a"
                    " declared field",
                )
            used_names.add(name)
        replacements[subtree.digest] = name
        operands: list[tuple[str, str, Field]] = []
        for role, argument in zip(
            ("input", "left", "right"), subtree.args, strict=False
        ):
            input_name = (
                _cstr(argument.attr("name"))
                if argument.op.name == "column_ref"
                else materializations[argument.digest]
            )
            field = input_types.get(input_name)
            if field is None:
                errors.raise_compile(
                    f"{path}.{name}",
                    errors.SCHEMA_MISMATCH,
                    f"rolling {kind} argument column {input_name!r} is not in the"
                    " input schema",
                )
            operands.append((role, input_name, field))
        periods = _cint(subtree.attr("periods"))
        if periods is not None:
            declarations.append(
                {
                    "kind": kind,
                    "primitive_version": 1,
                    "input": operands[0][1],
                    "output": name,
                    "periods": periods,
                }
            )
            derived_fields.append(Field(name, operands[0][2].data_type, nullable=True))
            continue
        if kind == "ewma":
            declarations.append(
                {
                    "kind": kind,
                    "primitive_version": 1,
                    "input": operands[0][1],
                    "output": name,
                    "span": _cint(subtree.attr("span")),
                    "min_periods": _cint(subtree.attr("min_periods")) or 1,
                }
            )
            derived_fields.append(Field(name, "float64", nullable=True))
            continue
        frame = _rolling_frame(subtree, f"{path}.{name}", kind)
        declaration: dict[str, object] = {
            "kind": kind,
            "primitive_version": 1,
            "output": name,
            "frame": frame,
            "min_periods": _cint(subtree.attr("min_periods")) or 1,
        }
        if kind in _ROLLING_PAIR_PRIMITIVES:
            declaration["left"] = operands[0][1]
            declaration["right"] = operands[1][1]
        else:
            declaration["input"] = operands[0][1]
        if kind in _ROLLING_DDOF_PRIMITIVES:
            ddof = _cint(subtree.attr("ddof"))
            declaration["ddof"] = 1 if ddof is None else ddof
        declarations.append(declaration)
        derived_fields.append(
            Field(
                name,
                _rolling_output_type(kind, operands[0][2].data_type) or "float64",
                nullable=True,
            )
        )

    for name, tree in fused_differences:
        replacements[tree.digest] = name
        declarations.append(
            {
                "kind": "difference",
                "primitive_version": 1,
                "left": _fused_float_leaf(
                    tree.args[0],
                    materializations,
                    input_types,
                    f"{path}.{name}.left",
                ),
                "right": _fused_float_leaf(
                    tree.args[1],
                    materializations,
                    input_types,
                    f"{path}.{name}.right",
                ),
                "output": name,
            }
        )
        derived_fields.append(Field(name, "float64", nullable=True))

    node_id = f"{output_name}__cf_rolling{stage_suffix}"
    node: dict[str, object] = {
        "id": node_id,
        "operator": {
            "kind": "rolling",
            "spec": {
                "configuration_version": 1,
                "state_layout_version": (
                    2
                    if any(
                        _rolling_declaration_requires_ewma(item)
                        for item in declarations
                    )
                    else 1
                ),
                "partition_by": list(entity_by),
                "event_time": event_time,
                "sequence_by": list(sequence_by),
                "outputs": declarations,
                "allowed_lateness_micros": allowed_lateness_micros,
                "late_policy": (
                    {"kind": "error", "scope": "envelope"}
                    if late_policy == "error"
                    else {"kind": "drop", "metrics_version": 1}
                ),
                "value_policy": "stateful_numeric_v1",
            },
        },
        "input_ports": [
            {
                "name": "input",
                "kind": "table",
                "required": True,
                "schema": [_field_json(field) for field in state_input_fields],
            }
        ],
        "output_ports": [
            {
                "name": "output",
                "kind": "table",
                "required": True,
                "schema": [
                    *(_field_json(field) for field in state_input_fields),
                    *(_field_json(field) for field in derived_fields),
                ],
            }
        ],
    }
    env = tuple(
        (name, _replace_materialized(tree, replacements)) for name, tree in segment.env
    )
    post_predicate = (
        None
        if segment.post_predicate is None
        else _replace_materialized(segment.post_predicate, replacements)
    )
    return _RollingPlan(
        node_id=node_id,
        node=node,
        materialization_node_id=materialization.node_id,
        materialization_node=materialization.node,
        env=env,
        post_predicate=post_predicate,
        input_field_names=(*input_types, *(field.name for field in derived_fields)),
        output_fields=(*state_input_fields, *derived_fields),
        replacements=tuple(replacements.items()),
    )


def _ready_rolling_occurrences(segment: _Segment, /) -> tuple[Node, ...]:
    ordered_trees = [tree for _, tree in segment.env]
    ordered_trees += [
        tree for tree in (segment.predicate, segment.post_predicate) if tree is not None
    ]
    occurrences: list[Node] = []
    seen: set[str] = set()
    for tree in ordered_trees:
        for subtree in _find_ready_rolling(tree):
            if subtree.digest not in seen:
                seen.add(subtree.digest)
                occurrences.append(subtree)
    return tuple(occurrences)


def _rolling_depth(node: Node, /) -> int:
    child_depth = max((_rolling_depth(argument) for argument in node.args), default=0)
    return child_depth + 1 if node.op.name in _ROLLING_PRIMITIVES else child_depth


def _rolling_stage_count(segment: _Segment, /) -> int:
    """Count the deterministic innermost-first rolling layers."""

    trees = [tree for _, tree in segment.env]
    trees += [
        tree for tree in (segment.predicate, segment.post_predicate) if tree is not None
    ]
    return max((_rolling_depth(tree) for tree in trees), default=0)


def _plan_rolling(
    output_name: str,
    segment: _Segment,
    path: str,
    allowed_lateness_micros: int,
    late_policy: str,
    /,
) -> _RollingPipeline | None:
    """Plan every innermost-first rolling layer for one output branch."""

    stage_count = _rolling_stage_count(segment)
    if stage_count == 0:
        return None
    stages: list[_RollingPlan] = []
    current = segment
    input_fields = _schema_fields(segment.input_node.attr("schema"))
    for stage_number in range(1, stage_count + 1):
        occurrences = _ready_rolling_occurrences(current)
        if not occurrences:
            raise RuntimeError("rolling stage count diverged during lowering")
        stage = _plan_rolling_stage(
            output_name,
            current,
            path,
            allowed_lateness_micros,
            late_policy,
            occurrences,
            input_fields,
            stage_number,
            stage_count,
        )
        stages.append(stage)
        current = replace(
            current,
            env=stage.env,
            post_predicate=stage.post_predicate,
        )
        input_fields = stage.output_fields
    final = stages[-1]
    return _RollingPipeline(
        tuple(stages),
        final.env,
        final.post_predicate,
        final.input_field_names,
        final.output_fields,
    )


@dataclass(frozen=True, slots=True)
class _CrossSectionPlan:
    """One lowered cross-section stage: the project node plus the rewritten
    row-local environment that references its output columns."""

    node_id: str
    node: dict[str, object]
    materialization_node_id: str | None
    materialization_node: dict[str, object] | None
    env: tuple[tuple[str, Node], ...]
    post_predicate: Node | None
    input_field_names: tuple[str, ...]
    output_fields: tuple[Field, ...]
    materializations: tuple[tuple[str, str], ...]
    replacements: tuple[tuple[str, str], ...]


def _cross_section_grouping(subtree: Node, path: str, /) -> dict[str, object]:
    """Render the frozen exact-time or fixed-bucket grouping JSON."""

    grouping = subtree.attr("grouping")
    if isinstance(grouping, CEnum) and grouping.variant == "exact_time":
        return {"kind": "exact_time"}
    if isinstance(grouping, CMap):
        tag = grouping.get("grouping")
        width = _cint(grouping.get("width_micros"))
        if isinstance(tag, CEnum) and tag.variant == "fixed_bucket" and width:
            return {"kind": "fixed_bucket", "width_micros": width}
    errors.raise_compile(
        path,
        errors.UNSUPPORTED_TYPE,
        "cross-section grouping is neither exact_time nor a fixed bucket",
    )
    raise AssertionError("unreachable")


def _enum_attr(subtree: Node, name: str, /) -> str:
    value = subtree.attr(name)
    return value.variant if isinstance(value, CEnum) else ""


def _grouping_shape(subtree: Node, /) -> tuple[str, int] | None:
    """Comparable grouping identity: the kind and, for buckets, the width."""

    grouping = subtree.attr("grouping")
    if isinstance(grouping, CEnum) and grouping.variant == "exact_time":
        return ("exact_time", 0)
    if isinstance(grouping, CMap):
        tag = grouping.get("grouping")
        width = _cint(grouping.get("width_micros"))
        if isinstance(tag, CEnum) and tag.variant == "fixed_bucket" and width:
            return ("fixed_bucket", width)
    return None


# Cross-section planning validates every occurrence with stable,
# declaration-ordered error paths before emitting the frozen node shape.
# #lizard forgives
def _plan_cross_section(
    output_name: str,
    segment: _Segment,
    path: str,
    allowed_lateness_micros: int,
    late_policy: str,
    input_fields_override: tuple[Field, ...] | None,
    /,
) -> _CrossSectionPlan | None:
    # #lizard forgives
    occurrences: list[Node] = []
    seen: set[str] = set()
    ordered_trees = [tree for _, tree in segment.env]
    ordered_trees += [
        tree for tree in (segment.predicate, segment.post_predicate) if tree is not None
    ]
    for tree in ordered_trees:
        for subtree in _find_cross_section(tree):
            if subtree.digest not in seen:
                seen.add(subtree.digest)
                occurrences.append(subtree)
    if not occurrences:
        return None

    input_fields = (
        _schema_fields(segment.input_node.attr("schema"))
        if input_fields_override is None
        else input_fields_override
    )
    input_types = {field.name: field for field in input_fields}
    entity_by = _cstr_seq(segment.input_node.attr("entity_by"))
    sequence_by = _cstr_seq(segment.input_node.attr("sequence_by"))
    event_time = _cstr(segment.input_node.attr("event_time"))
    if not entity_by or not sequence_by or not event_time:
        errors.raise_compile(
            path,
            errors.ORDERING_REQUIRED,
            "cross-section primitives require declared entity_by,"
            " event_time, and sequence_by ordering keys on the input table",
        )

    whole_feature = {
        tree.digest: name
        for name, tree in segment.env
        if tree.op.name in _CROSS_SECTION_PRIMITIVES
    }
    used_names = set(input_types) | {name for name, _ in segment.env}
    materialization = _plan_stateful_inputs(
        _StatefulInputRequest(
            output_name,
            path,
            "cs_input",
            f"{output_name}__cf_cross_section_input",
            "cross-section",
        ),
        input_fields,
        used_names,
        tuple((subtree.op.name, subtree.args[0]) for subtree in occurrences),
    )
    materializations = dict(materialization.names)
    state_input_fields = materialization.input_fields
    input_types = {field.name: field for field in state_input_fields}
    used_names = set(materialization.used_names)
    replacements: dict[str, str] = {}
    declarations: list[dict[str, object]] = []
    partition_columns: list[str] = []
    derived_fields: list[Field] = []
    for index, subtree in enumerate(occurrences):
        kind = subtree.op.name
        name = whole_feature.get(subtree.digest)
        if name is None:
            name = f"{output_name}__cf_cs_{index}"
            if name in used_names:
                errors.raise_compile(
                    f"{path}.{name}",
                    errors.DUPLICATE_NAME,
                    f"materialized cross-section column {name!r} collides with a"
                    " declared field",
                )
            used_names.add(name)
        replacements[subtree.digest] = name
        argument = subtree.args[0]
        input_name = (
            _cstr(argument.attr("name"))
            if argument.op.name == "column_ref"
            else materializations[argument.digest]
        )
        if input_name not in input_types:
            errors.raise_compile(
                f"{path}.{name}",
                errors.SCHEMA_MISMATCH,
                f"cross-section {kind} argument column {input_name!r} is not in"
                " the input schema",
            )
        event_time_argument = subtree.args[1]
        if event_time_argument.op.name != "column_ref":
            errors.raise_compile(
                f"{path}.{name}",
                errors.UNSUPPORTED_TYPE,
                "cross-section grouping event time must be an input column in"
                " this release",
            )
        event_time_name = _cstr(event_time_argument.attr("name"))
        if event_time_name != event_time:
            errors.raise_compile(
                f"{path}.{name}",
                errors.SCHEMA_MISMATCH,
                f"cross-section grouping event time {event_time_name!r} does not"
                " match the declared input event time",
            )
        group_partitions: list[str] = []
        for group_argument in subtree.args[2:]:
            if group_argument.op.name != "column_ref":
                errors.raise_compile(
                    f"{path}.{name}",
                    errors.UNSUPPORTED_TYPE,
                    "cross-section group columns must be input columns in this release",
                )
            group_name = _cstr(group_argument.attr("name"))
            if group_name not in input_types:
                errors.raise_compile(
                    f"{path}.{name}",
                    errors.SCHEMA_MISMATCH,
                    f"cross-section group column {group_name!r} is not in the"
                    " input schema",
                )
            group_partitions.append(group_name)
        grouping_shape = _grouping_shape(subtree)
        if index == 0:
            partition_columns = group_partitions
            declared_grouping = grouping_shape
        elif (
            partition_columns != group_partitions or declared_grouping != grouping_shape
        ):
            errors.raise_compile(
                path,
                errors.SCHEMA_MISMATCH,
                "cross-section primitives in one output must share one"
                " grouping declaration",
            )
        declaration: dict[str, object] = {
            "kind": kind,
            "primitive_version": 1,
            "input": input_name,
            "output": name,
        }
        if kind in _CROSS_SECTION_ORDERING:
            declaration["direction"] = _enum_attr(subtree, "direction") or "ascending"
            declaration["tie_method"] = _enum_attr(subtree, "tie_method") or "average"
            declaration["null_placement"] = (
                _enum_attr(subtree, "null_placement") or "exclude"
            )
        declaration["min_samples"] = _cint(subtree.attr("min_samples")) or 1
        if kind == _CROSS_SECTION_DDOF:
            declaration["ddof"] = _cint(subtree.attr("ddof")) or 0
        if kind == "winsorize":
            declaration["lower"] = _cnumber(subtree.attr("lower"))
            declaration["upper"] = _cnumber(subtree.attr("upper"))
        if kind in ("top", "bottom"):
            declaration["count"] = _cint(subtree.attr("count"))
            declaration["include_ties"] = _cbool(subtree.attr("include_ties"))
        declarations.append(declaration)
        output_type = (
            "bool"
            if kind in ("top", "bottom")
            else input_types[input_name].data_type
            if kind in ("winsorize", "mean_fill")
            else "float64"
        )
        derived_fields.append(Field(name, output_type, nullable=True))

    node_id = f"{output_name}__cf_cross_section"
    node: dict[str, object] = {
        "id": node_id,
        "operator": {
            "kind": "cross_section",
            "spec": {
                "configuration_version": 1,
                "state_layout_version": 1,
                "event_time": event_time,
                "entity_by": list(entity_by),
                "partition_by": list(partition_columns),
                "sequence_by": list(sequence_by),
                "grouping": _cross_section_grouping(
                    occurrences[0], f"{path}.{output_name}"
                ),
                "outputs": declarations,
                "allowed_lateness_micros": allowed_lateness_micros,
                "late_policy": (
                    {"kind": "error", "scope": "envelope"}
                    if late_policy == "error"
                    else {"kind": "drop", "metrics_version": 1}
                ),
                "value_policy": "nan_exclude_preserve_v1",
            },
        },
        "input_ports": [
            {
                "name": "input",
                "kind": "table",
                "required": True,
                "schema": [_field_json(field) for field in state_input_fields],
            }
        ],
        "output_ports": [
            {
                "name": "output",
                "kind": "table",
                "required": True,
                "schema": [
                    *(_field_json(field) for field in state_input_fields),
                    *(_field_json(field) for field in derived_fields),
                ],
            }
        ],
    }
    env = tuple(
        (name, _replace_materialized(tree, replacements)) for name, tree in segment.env
    )
    post_predicate = (
        None
        if segment.post_predicate is None
        else _replace_materialized(segment.post_predicate, replacements)
    )
    return _CrossSectionPlan(
        node_id,
        node,
        materialization.node_id,
        materialization.node,
        env,
        post_predicate,
        (*input_types, *(field.name for field in derived_fields)),
        (*state_input_fields, *derived_fields),
        materialization.names,
        tuple(replacements.items()),
    )


def _shared_materialized_name(
    prefix: str, counter: int, reserved: set[str], /
) -> tuple[str, int]:
    while True:
        name = f"__cf_shared_{prefix}_{counter}"
        counter += 1
        if name not in reserved:
            reserved.add(name)
            return name, counter


def _plan_outputs(plan: _RollingPlan | _CrossSectionPlan, /) -> list[dict[str, object]]:
    return plan.node["operator"]["spec"]["outputs"]  # type: ignore[index,return-value]


def _required_state_plan[StatePlanT: (_RollingPlan, _CrossSectionPlan)](
    plans: dict[str, StatePlanT | None], output_name: str, /
) -> StatePlanT:
    plan = plans[output_name]
    if plan is None:
        raise RuntimeError(f"missing shared-state plan for output {output_name!r}")
    return plan


def _merge_state_outputs[StatePlanT: (_RollingPlan, _CrossSectionPlan)](
    members: list[str],
    plans: dict[str, StatePlanT | None],
    prefix: str,
    /,
) -> tuple[StatePlanT, dict[str, str], list[dict[str, object]], tuple[Field, ...]]:
    first = _required_state_plan(plans, members[0])
    base_count = len(first.output_fields) - len(first.replacements)
    base_fields = first.output_fields[:base_count]
    reserved = {
        field.name
        for output_name in members
        for field in _required_state_plan(plans, output_name).output_fields
    }
    replacements: dict[str, str] = {}
    declarations: list[dict[str, object]] = []
    derived_fields: list[Field] = []
    counter = 0
    for output_name in members:
        plan = _required_state_plan(plans, output_name)
        declarations_by_name = {item["output"]: item for item in _plan_outputs(plan)}
        fields_by_name = {field.name: field for field in plan.output_fields}
        for digest, old_name in plan.replacements:
            if digest in replacements:
                continue
            new_name, counter = _shared_materialized_name(prefix, counter, reserved)
            replacements[digest] = new_name
            declarations.append({**declarations_by_name[old_name], "output": new_name})
            field = fields_by_name[old_name]
            derived_fields.append(
                Field(new_name, field.data_type, nullable=field.nullable)
            )
    return first, replacements, declarations, (*base_fields, *derived_fields)


def _shared_state_node(
    plan: _RollingPlan | _CrossSectionPlan,
    node_id: str,
    declarations: list[dict[str, object]],
    output_fields: tuple[Field, ...],
    /,
) -> dict[str, object]:
    operator = plan.node["operator"]
    spec = operator["spec"]  # type: ignore[index]
    return {
        **plan.node,
        "id": node_id,
        "operator": {
            **operator,  # type: ignore[arg-type]
            "spec": {**spec, "outputs": declarations},  # type: ignore[arg-type]
        },
        "output_ports": [
            {
                "name": "output",
                "kind": "table",
                "required": True,
                "schema": [_field_json(field) for field in output_fields],
            }
        ],
    }


def _unique_shared_node_id(stem: str, reserved: set[str], /) -> str:
    candidate = stem
    counter = 1
    while candidate in reserved:
        candidate = f"{stem}_{counter}"
        counter += 1
    reserved.add(candidate)
    return candidate


def _reserved_state_node_ids[StatePlanT: (_RollingPlan, _CrossSectionPlan)](
    segments: list[tuple[str, _Segment]],
    plans: dict[str, StatePlanT | None],
    /,
) -> set[str]:
    reserved = {output_name for output_name, _ in segments}
    reserved.update(_cstr(segment.input_node.attr("name")) for _, segment in segments)
    reserved.update(plan.node_id for plan in plans.values() if plan is not None)
    return reserved


def _shared_group_plans[StatePlanT: (_RollingPlan, _CrossSectionPlan)](
    members: list[str],
    by_name: dict[str, _Segment],
    first: StatePlanT,
    replacements: dict[str, str],
    node_id: str,
    node: dict[str, object],
    output_fields: tuple[Field, ...],
    /,
) -> dict[str, StatePlanT]:
    shared: dict[str, StatePlanT] = {}
    for output_name in members:
        segment = by_name[output_name]
        post_predicate = segment.post_predicate
        if post_predicate is not None:
            post_predicate = _replace_materialized(post_predicate, replacements)
        shared[output_name] = replace(
            first,
            node_id=node_id,
            node=node,
            env=tuple(
                (name, _replace_materialized(tree, replacements))
                for name, tree in segment.env
            ),
            post_predicate=post_predicate,
            input_field_names=tuple(field.name for field in output_fields),
            output_fields=output_fields,
            replacements=tuple(replacements.items()),
        )
    return shared


def _share_state_groups[StatePlanT: (_RollingPlan, _CrossSectionPlan)](
    groups: list[list[str]],
    segments: list[tuple[str, _Segment]],
    plans: dict[str, StatePlanT | None],
    prefix: str,
    suffix: str,
    /,
) -> dict[str, StatePlanT | None]:
    shared = dict(plans)
    by_name = dict(segments)
    reserved_ids = _reserved_state_node_ids(segments, plans)
    for members in groups:
        if len(members) < 2:
            continue
        first, replacements, declarations, output_fields = _merge_state_outputs(
            members, plans, prefix
        )
        node_id = _unique_shared_node_id(
            f"{members[0]}__cf_shared_{suffix}", reserved_ids
        )
        node = _shared_state_node(first, node_id, declarations, output_fields)
        shared.update(
            _shared_group_plans(
                members,
                by_name,
                first,
                replacements,
                node_id,
                node,
                output_fields,
            )
        )
    return shared


def _share_single_rolling_stages(
    segments: list[tuple[str, _Segment]],
    plans: dict[str, _RollingPlan | None],
    /,
) -> dict[str, _RollingPlan | None]:
    groups: dict[tuple[str, str | None, str | None], list[str]] = {}
    for output_name, segment in segments:
        plan = plans[output_name]
        if plan is not None:
            predicate = None if segment.predicate is None else segment.predicate.digest
            materialization = (
                None
                if plan.materialization_node is None
                else _canonical(plan.materialization_node)
            )
            groups.setdefault(
                (segment.input_node.digest, predicate, materialization), []
            ).append(output_name)
    return _share_state_groups(
        list(groups.values()), segments, plans, "roll", "rolling"
    )


def _share_rolling_plans(
    segments: list[tuple[str, _Segment]],
    plans: dict[str, _RollingPipeline | None],
    /,
) -> dict[str, _RollingPipeline | None]:
    """Preserve existing cross-output sharing for one-stage pipelines."""

    stages = {
        output_name: (
            None
            if pipeline is None or len(pipeline.stages) != 1
            else pipeline.stages[0]
        )
        for output_name, pipeline in plans.items()
    }
    shared_stages = _share_single_rolling_stages(segments, stages)
    shared = dict(plans)
    for output_name, stage in shared_stages.items():
        pipeline = plans[output_name]
        if pipeline is None or stage is None:
            continue
        shared[output_name] = _RollingPipeline(
            (stage,),
            stage.env,
            stage.post_predicate,
            stage.input_field_names,
            stage.output_fields,
        )
    return _share_identical_multi_stage_pipelines(segments, shared)


def _rolling_pipeline_identity(
    segment: _Segment, /
) -> tuple[str, str | None, tuple[str, ...]]:
    predicate = None if segment.predicate is None else segment.predicate.digest
    seen: set[str] = set()
    digests: list[str] = []
    trees = [tree for _, tree in segment.env]
    trees += [tree for tree in (segment.post_predicate,) if tree is not None]
    for tree in trees:
        for subtree in _find_rolling(tree):
            if subtree.digest not in seen:
                seen.add(subtree.digest)
                digests.append(subtree.digest)
    return segment.input_node.digest, predicate, tuple(digests)


def _shared_pipeline_stages(
    first_name: str,
    pipeline: _RollingPipeline,
    reserved_ids: set[str],
    /,
) -> tuple[_RollingPlan, ...]:
    stages: list[_RollingPlan] = []
    for index, stage in enumerate(pipeline.stages, start=1):
        node_id = _unique_shared_node_id(
            f"{first_name}__cf_shared_rolling_{index}", reserved_ids
        )
        materialization_id = None
        materialization = None
        if stage.materialization_node is not None:
            materialization_id = _unique_shared_node_id(
                f"{first_name}__cf_shared_rolling_input_{index}", reserved_ids
            )
            materialization = {
                **stage.materialization_node,
                "id": materialization_id,
            }
        stages.append(
            replace(
                stage,
                node_id=node_id,
                node={**stage.node, "id": node_id},
                materialization_node_id=materialization_id,
                materialization_node=materialization,
            )
        )
    return tuple(stages)


def _rewrite_pipeline_environment(
    segment: _Segment,
    stages: tuple[_RollingPlan, ...],
    /,
) -> tuple[tuple[tuple[str, Node], ...], Node | None]:
    env = segment.env
    post_predicate = segment.post_predicate
    for stage in stages:
        replacements = dict(stage.replacements)
        env = tuple(
            (name, _replace_materialized(tree, replacements)) for name, tree in env
        )
        if post_predicate is not None:
            post_predicate = _replace_materialized(post_predicate, replacements)
    return env, post_predicate


def _multi_stage_pipeline_groups(
    segments: list[tuple[str, _Segment]],
    plans: dict[str, _RollingPipeline | None],
    /,
) -> tuple[tuple[str, ...], ...]:
    groups: dict[tuple[str, str | None, tuple[str, ...]], list[str]] = {}
    for output_name, segment in segments:
        pipeline = plans[output_name]
        if pipeline is not None and len(pipeline.stages) > 1:
            groups.setdefault(_rolling_pipeline_identity(segment), []).append(
                output_name
            )
    return tuple(tuple(members) for members in groups.values() if len(members) > 1)


def _reserved_rolling_pipeline_ids(
    segments: list[tuple[str, _Segment]],
    plans: dict[str, _RollingPipeline | None],
    /,
) -> set[str]:
    reserved_ids = {output_name for output_name, _ in segments}
    for pipeline in plans.values():
        if pipeline is None:
            continue
        for stage in pipeline.stages:
            reserved_ids.add(stage.node_id)
            if stage.materialization_node_id is not None:
                reserved_ids.add(stage.materialization_node_id)
    return reserved_ids


def _shared_multi_stage_group(
    members: tuple[str, ...],
    segments: dict[str, _Segment],
    plans: dict[str, _RollingPipeline | None],
    reserved_ids: set[str],
    /,
) -> dict[str, _RollingPipeline]:
    first = plans[members[0]]
    if first is None:
        raise RuntimeError("missing multi-stage rolling pipeline")
    stages = _shared_pipeline_stages(members[0], first, reserved_ids)
    shared: dict[str, _RollingPipeline] = {}
    for output_name in members:
        env, post_predicate = _rewrite_pipeline_environment(
            segments[output_name], stages
        )
        shared[output_name] = _RollingPipeline(
            stages,
            env,
            post_predicate,
            first.input_field_names,
            first.output_fields,
        )
    return shared


def _share_identical_multi_stage_pipelines(
    segments: list[tuple[str, _Segment]],
    plans: dict[str, _RollingPipeline | None],
    /,
) -> dict[str, _RollingPipeline | None]:
    groups = _multi_stage_pipeline_groups(segments, plans)
    reserved_ids = _reserved_rolling_pipeline_ids(segments, plans)
    segments_by_name = dict(segments)
    shared = dict(plans)
    for members in groups:
        shared.update(
            _shared_multi_stage_group(
                members,
                segments_by_name,
                plans,
                reserved_ids,
            )
        )
    return shared


def _cross_section_group_identity(plan: _CrossSectionPlan, /) -> str:
    spec = plan.node["operator"]["spec"]  # type: ignore[index]
    return _canonical({key: value for key, value in spec.items() if key != "outputs"})


def _materialized_select_expression(plan: _CrossSectionPlan, name: str, /) -> str:
    node = plan.materialization_node
    if node is None:
        raise RuntimeError(f"missing cross-section materialization for {name!r}")
    suffix = f" AS {_quote_identifier(name)}"
    selects = node["operator"]["select"]  # type: ignore[index]
    for item in selects:
        if item.endswith(suffix):
            return item[: -len(suffix)]
    raise RuntimeError(f"missing materialized select for {name!r}")


@dataclass(frozen=True, slots=True)
class _SharedCrossInputs:
    base_fields: tuple[Field, ...]
    fields: tuple[Field, ...]
    expressions: tuple[str, ...]
    names: tuple[tuple[str, str], ...]

    @property
    def input_fields(self) -> tuple[Field, ...]:
        return (*self.base_fields, *self.fields)


def _collect_shared_cross_inputs(
    members: list[str],
    plans: dict[str, _CrossSectionPlan | None],
    /,
) -> _SharedCrossInputs:
    first = _required_state_plan(plans, members[0])
    base_count = (
        len(first.output_fields) - len(first.replacements) - len(first.materializations)
    )
    reserved = {
        field.name
        for output_name in members
        for field in _required_state_plan(plans, output_name).output_fields
    }
    names: dict[str, str] = {}
    fields: list[Field] = []
    expressions: list[str] = []
    counter = 0
    for output_name in members:
        plan = _required_state_plan(plans, output_name)
        fields_by_name = {field.name: field for field in plan.output_fields}
        for digest, old_name in plan.materializations:
            if digest in names:
                continue
            name, counter = _shared_materialized_name("cs_input", counter, reserved)
            names[digest] = name
            field = fields_by_name[old_name]
            fields.append(Field(name, field.data_type, nullable=field.nullable))
            expressions.append(_materialized_select_expression(plan, old_name))
    return _SharedCrossInputs(
        first.output_fields[:base_count],
        tuple(fields),
        tuple(expressions),
        tuple(names.items()),
    )


def _shared_cross_materialization(
    shared: _SharedCrossInputs, node_id: str, /
) -> dict[str, object] | None:
    if not shared.fields:
        return None
    return _expression_node(
        node_id,
        [
            *(_quote_identifier(field.name) for field in shared.base_fields),
            *(
                f"{expression} AS {_quote_identifier(field.name)}"
                for expression, field in zip(
                    shared.expressions, shared.fields, strict=True
                )
            ),
        ],
        None,
        shared.base_fields,
        shared.input_fields,
    )


def _aligned_cross_section_plan(
    plan: _CrossSectionPlan,
    shared: _SharedCrossInputs,
    node_id: str,
    materialization: dict[str, object] | None,
    /,
) -> _CrossSectionPlan:
    names = dict(shared.names)
    old_to_new = {old_name: names[digest] for digest, old_name in plan.materializations}
    outputs = [
        {**item, "input": old_to_new.get(item["input"], item["input"])}
        for item in _plan_outputs(plan)
    ]
    derived_fields = plan.output_fields[-len(plan.replacements) :]
    output_fields = (*shared.input_fields, *derived_fields)
    return replace(
        plan,
        node={
            **plan.node,
            "operator": {
                **plan.node["operator"],  # type: ignore[dict-item]
                "spec": {
                    **plan.node["operator"]["spec"],  # type: ignore[index]
                    "outputs": outputs,
                },
            },
            "input_ports": [
                {
                    "name": "input",
                    "kind": "table",
                    "required": True,
                    "schema": [_field_json(field) for field in shared.input_fields],
                }
            ],
            "output_ports": [
                {
                    "name": "output",
                    "kind": "table",
                    "required": True,
                    "schema": [_field_json(field) for field in output_fields],
                }
            ],
        },
        materialization_node_id=node_id if shared.fields else None,
        materialization_node=materialization,
        input_field_names=tuple(field.name for field in output_fields),
        output_fields=output_fields,
        materializations=shared.names,
    )


def _combined_cross_section_inputs(
    members: list[str],
    plans: dict[str, _CrossSectionPlan | None],
    node_id: str,
    /,
) -> dict[str, _CrossSectionPlan]:
    shared = _collect_shared_cross_inputs(members, plans)
    materialization = _shared_cross_materialization(shared, node_id)
    return {
        output_name: _aligned_cross_section_plan(
            _required_state_plan(plans, output_name),
            shared,
            node_id,
            materialization,
        )
        for output_name in members
    }


def _share_cross_section_plans(
    segments: list[tuple[str, _Segment]],
    upstream_ids: dict[str, str | None],
    plans: dict[str, _CrossSectionPlan | None],
    /,
) -> dict[str, _CrossSectionPlan | None]:
    groups: dict[tuple[str, str | None, str], list[str]] = {}
    for output_name, segment in segments:
        plan = plans[output_name]
        if plan is not None:
            upstream = upstream_ids[output_name] or segment.input_node.digest
            predicate = None if segment.predicate is None else segment.predicate.digest
            groups.setdefault(
                (upstream, predicate, _cross_section_group_identity(plan)), []
            ).append(output_name)
    aligned = dict(plans)
    reserved_ids = _reserved_state_node_ids(segments, plans)
    for group in groups.values():
        if len(group) < 2:
            continue
        materialization_id = _unique_shared_node_id(
            f"{group[0]}__cf_shared_cross_section_input", reserved_ids
        )
        aligned.update(_combined_cross_section_inputs(group, plans, materialization_id))
    return _share_state_groups(
        list(groups.values()), segments, aligned, "cs", "cross_section"
    )


def _check_declared_inputs(program: Program, /) -> None:
    for value in program.inputs:
        node = value._node
        if node.op.name == "parameter":
            errors.raise_compile(
                f"static_inputs.{_cstr(node.attr('name'))}",
                errors.UNKNOWN_PRIMITIVE_VERSION,
                "static parameters are not supported by the row-local lowerer",
            )


@dataclass(frozen=True, slots=True)
class _MatrixExpression:
    backend: str
    columns: tuple[str, ...]
    source_digests: frozenset[str]
    parameter: Node | None
    weights_count: int
    matmul_count: int
    matmul_rhs_is_weights: bool
    tree: dict[str, object]


@dataclass(frozen=True, slots=True)
class _LoweringValue:
    _node: Node


@dataclass(frozen=True, slots=True)
class _LoweringProgram:
    name: str
    inputs: tuple[_LoweringValue, ...]
    outputs: tuple[tuple[str, _LoweringValue], ...]
