"""Deterministic lowering of symbolic programs to strict project-v3.

The lowerer resolves each declared table output into one fused row-local
segment, renders the segment as DataFusion SQL inside strict project-v3
``expression`` nodes, and hands the document to the existing Rust graph
compiler for final port, schema, topology, and fingerprint validation. No data
object, source, sink, or runner is accepted here, and no symbolic Python runs
while a compiled plan executes.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Final

from calc_flow.pipeline import (
    Runtime,
    StreamRequirements,
    _canonical,
    _data_sources,
)
from calc_flow.symbolic import errors
from calc_flow.symbolic.analyzer import _require_mode, _run, _schema_fields
from calc_flow.symbolic.domains import type_name
from calc_flow.symbolic.nodes import (
    CBool,
    CDType,
    CFloat,
    CInt,
    CNull,
    CStr,
    CValue,
    Node,
    build,
)
from calc_flow.symbolic.optimizer import expression_refs, extract_common
from calc_flow.symbolic.types import Field

if TYPE_CHECKING:
    from calc_flow.pipeline import BatchExecutionPlan, StreamExecutionPlan
    from calc_flow.symbolic.program import Program

_COLUMN_PRIMITIVES: Final = frozenset(
    {
        "column_ref",
        "literal",
        "add",
        "sub",
        "mul",
        "truediv",
        "neg",
        "eq",
        "ne",
        "lt",
        "le",
        "gt",
        "ge",
        "and",
        "or",
        "not",
        "where",
        "coalesce",
        "log",
        "exp",
        "sqrt",
        "abs",
        "clip",
        "cast",
    }
)

_TABLE_OUTPUT_PRIMITIVES: Final = frozenset(
    {"table_input", "project", "filter", "with_columns"}
)

_ROLLING_PRIMITIVES: Final = frozenset({"lag", "delta"})

_U64_MAX: Final = (1 << 64) - 1

_BINARY_SQL: Final = {
    "add": "+",
    "sub": "-",
    "mul": "*",
    "truediv": "/",
    "eq": "=",
    "ne": "!=",
    "lt": "<",
    "le": "<=",
    "gt": ">",
    "ge": ">=",
    "and": "AND",
    "or": "OR",
}

_FUNCTION_SQL: Final = {"log": "ln", "exp": "exp", "sqrt": "sqrt", "abs": "abs"}

_CAST_TYPES: Final = {
    "bool": "BOOLEAN",
    "int8": "TINYINT",
    "int16": "SMALLINT",
    "int32": "INT",
    "int64": "BIGINT",
    "uint8": "TINYINT UNSIGNED",
    "uint16": "SMALLINT UNSIGNED",
    "uint32": "INT UNSIGNED",
    "uint64": "BIGINT UNSIGNED",
    "float32": "REAL",
    "float64": "DOUBLE",
}


@dataclass(frozen=True, slots=True)
class _Segment:
    """One fused row-local table resolution over one table input lineage.

    ``predicate`` filters declared *below* every rolling feature (they feed
    the rolling stage); ``post_predicate`` filters declared *above* them
    (they apply after). Without rolling primitives both fuse at the final
    stage, preserving the historical behavior.
    """

    input_node: Node
    fields: tuple[str, ...]
    env: tuple[tuple[str, Node], ...]
    predicate: Node | None
    post_predicate: Node | None = None


def _cstr(value: CValue | None, /) -> str:
    return value.value if isinstance(value, CStr) else ""


def _cstr_seq(value: CValue | None, /) -> tuple[str, ...]:
    from calc_flow.symbolic.nodes import CSeq

    if isinstance(value, CSeq):
        return tuple(item.value for item in value.items if isinstance(item, CStr))
    return ()


def _base_ref(name: str, /) -> Node:
    return build("column_ref", (), {"name": CStr(name)})


def _reject_primitive(path: str, node: Node, /) -> None:
    errors.raise_compile(
        path,
        errors.UNKNOWN_PRIMITIVE_VERSION,
        f"primitive {node.op.name!r} is not supported by the row-local lowerer",
    )


def _resolve_table(node: Node, path: str, /) -> _Segment:
    name = node.op.name
    if name == "table_input":
        fields = _schema_fields(node.attr("schema"))
        return _Segment(
            node,
            tuple(field.name for field in fields),
            tuple((field.name, _base_ref(field.name)) for field in fields),
            None,
        )
    if name == "project":
        child = _resolve_table(node.args[0], f"{path}.project.value")
        env = dict(child.env)
        columns = _cstr_seq(node.attr("columns"))
        return _Segment(
            child.input_node,
            columns,
            tuple((field, env[field]) for field in columns),
            child.predicate,
            child.post_predicate,
        )
    if name == "filter":
        child = _resolve_table(node.args[0], f"{path}.filter.value")
        predicate = _inline(node.args[1], dict(child.env), f"{path}.filter.predicate")
        if _segment_has_rolling(child) or any(True for _ in _find_rolling(predicate)):
            combined = (
                predicate
                if child.post_predicate is None
                else build("and", (child.post_predicate, predicate), {})
            )
            return _Segment(
                child.input_node, child.fields, child.env, child.predicate, combined
            )
        combined = (
            predicate
            if child.predicate is None
            else build("and", (child.predicate, predicate), {})
        )
        return _Segment(
            child.input_node, child.fields, child.env, combined, child.post_predicate
        )
    if name == "with_columns":
        child = _resolve_table(node.args[0], f"{path}.with_columns.value")
        env = dict(child.env)
        names = _cstr_seq(node.attr("names"))
        for index, feature in enumerate(names):
            env[feature] = _inline(node.args[index + 1], env, f"{path}.{feature}")
        return _Segment(
            child.input_node,
            (*child.fields, *names),
            tuple(env.items()),
            child.predicate,
            child.post_predicate,
        )
    _reject_primitive(path, node)


def _segment_has_rolling(segment: _Segment, /) -> bool:
    return any(True for _, tree in segment.env for _ in _find_rolling(tree)) or any(
        True
        for tree in (segment.predicate, segment.post_predicate)
        if tree is not None
        for _ in _find_rolling(tree)
    )


def _inline(node: Node, env: dict[str, Node], path: str, /) -> Node:
    name = node.op.name
    if name == "column_ref":
        return env[_cstr(node.attr("name"))]
    if name == "literal":
        return node
    if name not in _COLUMN_PRIMITIVES and name not in _ROLLING_PRIMITIVES:
        _reject_primitive(path, node)
    if name == "cast":
        _cast_target(node, path)
    return build(
        name,
        tuple(_inline(argument, env, path) for argument in node.args),
        dict(node.attrs.entries),
        version=node.op.version,
    )


def _find_rolling(node: Node, /):
    """Yield every lag/delta subtree in first-appearance order."""
    if node.op.name in _ROLLING_PRIMITIVES:
        yield node
    for argument in node.args:
        yield from _find_rolling(argument)


def _cast_target(node: Node, path: str, /) -> str:
    raw = node.attr("data_type")
    declared = _cstr(raw) or (raw.name if isinstance(raw, CDType) else "")
    target = _CAST_TYPES.get(declared)
    if target is None:
        errors.raise_compile(
            f"{path}.cast.data_type",
            errors.UNSUPPORTED_TYPE,
            f"cast target {declared!r} is not portable in the row-local lowerer",
        )
    return target


def _sql(node: Node, /) -> str:
    name = node.op.name
    if name == "column_ref":
        return _quote_identifier(_cstr(node.attr("name")))
    if name == "literal":
        return _sql_literal(node.attr("value"))
    if name in _BINARY_SQL:
        return f"({_sql(node.args[0])} {_BINARY_SQL[name]} {_sql(node.args[1])})"
    if name == "neg":
        return f"(-{_sql(node.args[0])})"
    if name == "not":
        return f"(NOT {_sql(node.args[0])})"
    if name == "where":
        return (
            f"(CASE WHEN {_sql(node.args[0])} THEN {_sql(node.args[1])}"
            f" ELSE {_sql(node.args[2])} END)"
        )
    if name == "coalesce":
        return "COALESCE(" + ", ".join(_sql(argument) for argument in node.args) + ")"
    if name in _FUNCTION_SQL:
        return f"{_FUNCTION_SQL[name]}({_sql(node.args[0])})"
    if name == "clip":
        value = _sql(node.args[0])
        lower = _sql_literal(node.attr("lower"))
        upper = _sql_literal(node.attr("upper"))
        return (
            f"(CASE WHEN {value} < {lower} THEN {lower}"
            f" WHEN {value} > {upper} THEN {upper} ELSE {value} END)"
        )
    if name == "cast":
        return f"CAST({_sql(node.args[0])} AS {_CAST_TYPES[_cast_type_name(node)]})"
    raise AssertionError(f"unlowerable primitive reached SQL rendering: {name}")


def _cast_type_name(node: Node, /) -> str:
    raw = node.attr("data_type")
    return _cstr(raw) or (raw.name if isinstance(raw, CDType) else "")


def _sql_literal(value: CValue | None, /) -> str:
    if isinstance(value, CNull) or value is None:
        return "NULL"
    if isinstance(value, CBool):
        return "TRUE" if value.value else "FALSE"
    if isinstance(value, CInt):
        return str(value.value)
    if isinstance(value, CFloat):
        return repr(value.value)
    if isinstance(value, CStr):
        return "'" + value.value.replace("'", "''") + "'"
    raise AssertionError(f"unsupported literal reached SQL rendering: {value!r}")


def _quote_identifier(name: str, /) -> str:
    return '"' + name.replace('"', '""') + '"'


def _select_item(name: str, tree: Node, /) -> str:
    if tree.op.name == "column_ref" and tree.attr("name") == CStr(name):
        return _quote_identifier(name)
    return f"{_sql(tree)} AS {_quote_identifier(name)}"


def _expression_node(
    node_id: str,
    select: list[str],
    filter_sql: str | None,
    input_schema: tuple[Field, ...] | None,
    output_schema: tuple[Field, ...] | None = None,
    /,
) -> dict[str, object]:
    node: dict[str, object] = {
        "id": node_id,
        "operator": {
            "kind": "expression",
            "expression": "",
            "select": select,
            "filter": filter_sql,
            "udfs": [],
        },
    }
    if input_schema is not None:
        node["input_ports"] = [
            {
                "name": "input",
                "kind": "table",
                "required": True,
                "schema": [_field_json(field) for field in input_schema],
            }
        ]
    if output_schema is not None:
        node["output_ports"] = [
            {
                "name": "output",
                "kind": "table",
                "required": True,
                "schema": [_field_json(field) for field in output_schema],
            }
        ]
    return node


def _field_json(field: Field, /) -> dict[str, object]:
    return {
        "name": field.name,
        "data_type": field.data_type,
        "nullable": field.nullable,
    }


def _cint(value: CValue | None, /) -> int | None:
    return value.value if isinstance(value, CInt) else None


def _replace_rolling(node: Node, replacements: dict[str, str], /) -> Node:
    replacement = replacements.get(node.digest)
    if replacement is not None:
        return _base_ref(replacement)
    return build(
        node.op.name,
        tuple(_replace_rolling(argument, replacements) for argument in node.args),
        dict(node.attrs.entries),
        version=node.op.version,
    )


@dataclass(frozen=True, slots=True)
class _RollingPlan:
    """One lowered rolling stage: the project node plus the rewritten
    row-local environment that references its output columns."""

    node_id: str
    node: dict[str, object]
    env: tuple[tuple[str, Node], ...]
    post_predicate: Node | None
    input_field_names: tuple[str, ...]
    output_fields: tuple[Field, ...]


# Rolling planning validates every lag/delta occurrence with stable,
# declaration-ordered error paths before emitting the frozen node shape.
def _plan_rolling(
    output_name: str,
    segment: _Segment,
    path: str,
    allowed_lateness_micros: int,
    late_policy: str,
    /,
) -> _RollingPlan | None:
    # #lizard forgives
    occurrences: list[Node] = []
    seen: set[str] = set()
    ordered_trees = [tree for _, tree in segment.env]
    ordered_trees += [
        tree for tree in (segment.predicate, segment.post_predicate) if tree is not None
    ]
    for tree in ordered_trees:
        for subtree in _find_rolling(tree):
            if subtree.digest not in seen:
                seen.add(subtree.digest)
                occurrences.append(subtree)
    if not occurrences:
        return None

    input_fields = _schema_fields(segment.input_node.attr("schema"))
    input_types = {field.name: field for field in input_fields}
    entity_by = _cstr_seq(segment.input_node.attr("entity_by"))
    sequence_by = _cstr_seq(segment.input_node.attr("sequence_by"))
    event_time = _cstr(segment.input_node.attr("event_time"))
    if not entity_by or not sequence_by or not event_time:
        errors.raise_compile(
            path,
            errors.ORDERING_REQUIRED,
            "rolling lag/delta requires declared entity_by, event_time, and"
            " sequence_by ordering keys on the input table",
        )

    whole_feature = {
        tree.digest: name
        for name, tree in segment.env
        if tree.op.name in _ROLLING_PRIMITIVES
    }
    used_names = set(input_types) | {name for name, _ in segment.env}
    replacements: dict[str, str] = {}
    declarations: list[dict[str, object]] = []
    derived_fields: list[Field] = []
    for index, subtree in enumerate(occurrences):
        kind = subtree.op.name
        name = whole_feature.get(subtree.digest)
        if name is None:
            name = f"{output_name}__cf_roll_{index}"
            if name in used_names:
                errors.raise_compile(
                    f"{path}.{name}",
                    errors.DUPLICATE_NAME,
                    f"materialized rolling column {name!r} collides with a"
                    " declared field",
                )
            used_names.add(name)
        replacements[subtree.digest] = name
        argument = subtree.args[0]
        if argument.op.name != "column_ref":
            errors.raise_compile(
                f"{path}.{name}",
                errors.UNSUPPORTED_TYPE,
                f"rolling {kind} argument must be an input column in this release",
            )
        input_name = _cstr(argument.attr("name"))
        field = input_types.get(input_name)
        if field is None:
            errors.raise_compile(
                f"{path}.{name}",
                errors.SCHEMA_MISMATCH,
                f"rolling {kind} argument column {input_name!r} is not in the"
                " input schema",
            )
        periods = _cint(subtree.attr("periods")) or 1
        declarations.append(
            {
                "kind": kind,
                "primitive_version": 1,
                "input": input_name,
                "output": name,
                "periods": periods,
            }
        )
        derived_fields.append(Field(name, field.data_type, nullable=True))

    node_id = f"{output_name}__cf_rolling"
    node: dict[str, object] = {
        "id": node_id,
        "operator": {
            "kind": "rolling",
            "spec": {
                "configuration_version": 1,
                "state_layout_version": 1,
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
                "schema": [_field_json(field) for field in input_fields],
            }
        ],
        "output_ports": [
            {
                "name": "output",
                "kind": "table",
                "required": True,
                "schema": [
                    *(_field_json(field) for field in input_fields),
                    *(_field_json(field) for field in derived_fields),
                ],
            }
        ],
    }
    env = tuple(
        (name, _replace_rolling(tree, replacements)) for name, tree in segment.env
    )
    post_predicate = (
        None
        if segment.post_predicate is None
        else _replace_rolling(segment.post_predicate, replacements)
    )
    return _RollingPlan(
        node_id,
        node,
        env,
        post_predicate,
        (*input_types, *(field.name for field in derived_fields)),
        (*input_fields, *derived_fields),
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


# The lowerer keeps per-output segment staging in one deterministic pass:
# stage order, edge wiring, and id assignment are semantic, so the rolling,
# prefilter, and CSE stages stay in one place.
def _lower_program(
    program: Program,
    mode: str,
    allowed_lateness_micros: int,
    late_policy: str,
    /,
) -> dict[str, object]:
    # #lizard forgives
    _check_declared_inputs(program)
    segments = []
    for output_name, value in program.outputs:
        node = value._node
        path = f"outputs.{output_name}"
        if node.op.name not in _TABLE_OUTPUT_PRIMITIVES:
            _reject_primitive(path, node)
        segments.append((output_name, _resolve_table(node, path)))
    consumed = {segment.input_node.digest for _, segment in segments}
    multi_output_lineages = {
        digest
        for digest in consumed
        if sum(1 for _, segment in segments if segment.input_node.digest == digest) > 1
    }
    fanout = len(program.inputs) > 1 or bool(multi_output_lineages)
    plans = {
        output_name: _plan_rolling(
            output_name,
            segment,
            f"outputs.{output_name}",
            allowed_lateness_micros,
            late_policy,
        )
        for output_name, segment in segments
    }
    rolling_digests = {
        segment.input_node.digest
        for (output_name, segment), plan in zip(segments, plans.values(), strict=True)
        if plan is not None
    }
    nodes: list[dict[str, object]] = []
    edges: list[dict[str, object]] = []
    fanout_ids: dict[str, str] = {}
    if fanout:
        for value in program.inputs:
            input_node = value._node
            if input_node.digest not in consumed:
                continue
            input_name = _cstr(input_node.attr("name"))
            schema = _schema_fields(input_node.attr("schema"))
            nodes.append(
                _expression_node(
                    input_name,
                    [_quote_identifier(field.name) for field in schema],
                    None,
                    schema,
                    schema if input_node.digest in rolling_digests else None,
                )
            )
            fanout_ids[input_node.digest] = input_name
    for output_name, segment in segments:
        rolling = plans[output_name]
        env = dict(segment.env)
        input_field_names = [
            field.name for field in _schema_fields(segment.input_node.attr("schema"))
        ]
        upstream_id: str | None = None
        final_predicate = segment.predicate
        if rolling is not None:
            if segment.predicate is not None:
                prefilter_id = f"{output_name}__cf_prefilter"
                input_fields = _schema_fields(segment.input_node.attr("schema"))
                nodes.append(
                    _expression_node(
                        prefilter_id,
                        [_quote_identifier(name) for name in input_field_names],
                        _sql(segment.predicate),
                        input_fields,
                        input_fields,
                    )
                )
                if fanout:
                    edges.append(
                        {
                            "source_node": fanout_ids[segment.input_node.digest],
                            "source_port": "output",
                            "target_node": prefilter_id,
                            "target_port": "input",
                        }
                    )
                upstream_id = prefilter_id
            nodes.append(rolling.node)
            if upstream_id is not None:
                edges.append(
                    {
                        "source_node": upstream_id,
                        "source_port": "output",
                        "target_node": rolling.node_id,
                        "target_port": "input",
                    }
                )
            elif fanout:
                edges.append(
                    {
                        "source_node": fanout_ids[segment.input_node.digest],
                        "source_port": "output",
                        "target_node": rolling.node_id,
                        "target_port": "input",
                    }
                )
            upstream_id = rolling.node_id
            env = dict(rolling.env)
            input_field_names = list(rolling.input_field_names)
            final_predicate = rolling.post_predicate
        elif segment.post_predicate is not None:
            final_predicate = (
                segment.post_predicate
                if segment.predicate is None
                else build("and", (segment.predicate, segment.post_predicate), {})
            )
        reserved = frozenset(env)
        fused = extract_common(
            tuple((field, env[field]) for field in segment.fields),
            final_predicate,
            reserved,
        )
        cse_order = [name for tier in fused.tiers for name, _ in tier]
        needed: set[str] = set()
        for _, tree in fused.selects:
            needed |= expression_refs(tree)
        if fused.predicate is not None:
            needed |= expression_refs(fused.predicate)
        tier_items: list[list[str]] = []
        for index in range(len(fused.tiers) - 1, -1, -1):
            tier = fused.tiers[index]
            defined = {name for name, _ in tier}
            passthrough = needed - defined
            items = [
                _quote_identifier(field)
                for field in input_field_names
                if field in passthrough
            ]
            items += [
                _quote_identifier(name) for name in cse_order if name in passthrough
            ]
            items += [
                f"{_sql(tree)} AS {_quote_identifier(name)}" for name, tree in tier
            ]
            tier_items.append(items)
            needed = set(passthrough)
            for _, tree in tier:
                needed |= expression_refs(tree)
        tier_items.reverse()
        stage_ids = [
            f"{output_name}__cf_cse_{index}" for index in range(1, len(fused.tiers) + 1)
        ] + [output_name]
        stage_selects = [
            *tier_items,
            [_select_item(field, tree) for field, tree in fused.selects],
        ]
        for position, (node_id, select) in enumerate(
            zip(stage_ids, stage_selects, strict=True)
        ):
            filter_sql = None
            if position == len(stage_ids) - 1 and fused.predicate is not None:
                filter_sql = _sql(fused.predicate)
            if position == 0:
                if upstream_id is not None:
                    edges.append(
                        {
                            "source_node": upstream_id,
                            "source_port": "output",
                            "target_node": node_id,
                            "target_port": "input",
                        }
                    )
                    input_schema = (
                        rolling.output_fields if rolling is not None else None
                    )
                elif fanout:
                    edges.append(
                        {
                            "source_node": fanout_ids[segment.input_node.digest],
                            "source_port": "output",
                            "target_node": node_id,
                            "target_port": "input",
                        }
                    )
                    input_schema = None
                else:
                    input_schema = _schema_fields(segment.input_node.attr("schema"))
            else:
                edges.append(
                    {
                        "source_node": stage_ids[position - 1],
                        "source_port": "output",
                        "target_node": node_id,
                        "target_port": "input",
                    }
                )
                input_schema = None
            nodes.append(_expression_node(node_id, select, filter_sql, input_schema))
    project: dict[str, object] = {
        "data_sources": [],
        "format_version": 3,
        "id": program.name,
        "name": program.name,
        "runtime": {"mode": mode, "options": {}},
        "graph": {"edges": edges, "name": program.name, "nodes": nodes},
    }
    project["data_sources"] = [] if mode == "stream" else _data_sources(project)
    return project


def _require_runtime(runtime: object, entry: str, /) -> Runtime:
    if not isinstance(runtime, Runtime):
        raise TypeError(
            f"{entry} requires an explicit calc_flow Runtime; got {type_name(runtime)}"
        )
    return runtime


def _check_expression_capability(
    program: Program,
    runtime: Runtime,
    mode: str,
    /,
) -> None:
    analyzer, capabilities = _run(program, runtime, mode)
    issues = analyzer.issues
    if issues:
        first = issues[0]
        errors.raise_compile(first.path, first.code, first.message)
    for operator in capabilities.operators:
        if operator.kind != "expression":
            continue
        if mode not in operator.modes:
            errors.raise_compile(
                program.name,
                errors.CAPABILITY_MISMATCH,
                f"the expression operator does not support {mode} mode in the"
                " selected capability snapshot",
            )
        if mode == "stream" and (
            operator.finality == "unproven"
            or not operator.microbatch_invariant
            or not operator.deterministic
            or not operator.replay_safe
        ):
            errors.raise_compile(
                program.name,
                errors.CAPABILITY_MISMATCH,
                "the expression operator does not prove stream lifecycle facts"
                " in the selected capability snapshot",
            )
        return
    errors.raise_compile(
        program.name,
        errors.CAPABILITY_MISMATCH,
        "the capability snapshot does not offer the expression operator",
    )


def _program_needs_rolling(program: Program, /) -> bool:
    return any(True for _, value in program.outputs for _ in _find_rolling(value._node))


# The capability gate conjoins the frozen stream lifecycle facts; every
# fact fails with the same stable capability_mismatch code.
def _check_rolling_capability(
    program: Program,
    capabilities: object,
    mode: str,
    /,
) -> None:
    # #lizard forgives
    for operator in capabilities.operators:
        if operator.kind != "rolling":
            continue
        if mode not in operator.modes:
            errors.raise_compile(
                program.name,
                errors.CAPABILITY_MISMATCH,
                f"the rolling operator does not support {mode} mode in the"
                " selected capability snapshot",
            )
        if mode == "stream" and (
            operator.finality == "unproven"
            or not operator.stateful
            or not operator.microbatch_invariant
            or operator.checkpoint_support != "checkpointed_stateful"
            or not isinstance(operator.state_version, int)
            or operator.state_version <= 0
            or not operator.deterministic
            or not operator.replay_safe
        ):
            errors.raise_compile(
                program.name,
                errors.CAPABILITY_MISMATCH,
                "the rolling operator does not prove stream lifecycle facts"
                " in the selected capability snapshot",
            )
        return
    errors.raise_compile(
        program.name,
        errors.CAPABILITY_MISMATCH,
        "the capability snapshot does not offer the rolling operator",
    )


def lower_program_document(
    program: Program,
    runtime: Runtime,
    mode: str,
    /,
    *,
    allowed_lateness_micros: int = 0,
    late_policy: str = "error",
) -> dict[str, object]:
    """Analyze and lower one program to its strict project-v3 document.

    The lateness arguments are validated whenever the program contains
    rolling primitives; row-local programs do not consume them.
    """

    selected = _require_runtime(runtime, "lower_program_document")
    mode_value = _require_mode(mode)
    _check_expression_capability(program, selected, mode_value)
    if _program_needs_rolling(program):
        _validate_lateness(allowed_lateness_micros, late_policy)
        _, capabilities = _run(program, selected, mode_value)
        _check_rolling_capability(program, capabilities, mode_value)
    return _lower_program(program, mode_value, allowed_lateness_micros, late_policy)


def compile_program_batch(program: Program, runtime: object, /) -> BatchExecutionPlan:
    """Lower one program to a strict project-v3 batch plan."""

    selected = _require_runtime(runtime, "compile_batch")
    document = lower_program_document(program, selected, "batch")
    return selected.compile_batch_project(_canonical(document))


def compile_program_stream(
    program: Program,
    runtime: object,
    allowed_lateness_micros: object,
    late_policy: object,
    /,
) -> StreamExecutionPlan:
    """Lower one program to a strict project-v3 continuous plan.

    The validated lateness arguments are written into every lowered rolling
    node; row-local programs are unaffected by them.
    """

    selected = _require_runtime(runtime, "compile_stream")
    _validate_lateness(allowed_lateness_micros, late_policy)
    document = lower_program_document(
        program,
        selected,
        "stream",
        allowed_lateness_micros=allowed_lateness_micros,
        late_policy=late_policy,
    )
    return selected._compile_stream_graph_project(
        _canonical(document), requirements=StreamRequirements()
    )


def _validate_lateness(allowed_lateness_micros: object, late_policy: object, /) -> None:
    if type(allowed_lateness_micros) is not int:
        raise TypeError(
            "allowed_lateness_micros must be an exact int; got"
            f" {type_name(allowed_lateness_micros)}"
        )
    if allowed_lateness_micros < 0:
        raise ValueError(
            "allowed_lateness_micros: invalid_literal: must be non-negative"
        )
    if allowed_lateness_micros > _U64_MAX:
        raise ValueError(
            "allowed_lateness_micros: invalid_literal: must fit the unsigned"
            " 64-bit microsecond range"
        )
    if type(late_policy) is not str:
        raise TypeError(f"late_policy must be a string; got {type_name(late_policy)}")
    if late_policy not in ("error", "drop"):
        raise ValueError(
            "late_policy: invalid_literal: must be 'error' or"
            f" 'drop'; got {late_policy!r}"
        )
