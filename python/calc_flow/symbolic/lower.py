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
    """One fused row-local table resolution over one table input lineage."""

    input_node: Node
    fields: tuple[str, ...]
    env: tuple[tuple[str, Node], ...]
    predicate: Node | None


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
        )
    if name == "filter":
        child = _resolve_table(node.args[0], f"{path}.filter.value")
        predicate = _inline(node.args[1], dict(child.env), f"{path}.filter.predicate")
        combined = (
            predicate
            if child.predicate is None
            else build("and", (child.predicate, predicate), {})
        )
        return _Segment(child.input_node, child.fields, child.env, combined)
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
        )
    _reject_primitive(path, node)


def _inline(node: Node, env: dict[str, Node], path: str, /) -> Node:
    name = node.op.name
    if name == "column_ref":
        return env[_cstr(node.attr("name"))]
    if name == "literal":
        return node
    if name not in _COLUMN_PRIMITIVES:
        _reject_primitive(path, node)
    if name == "cast":
        _cast_target(node, path)
    return build(
        name,
        tuple(_inline(argument, env, path) for argument in node.args),
        dict(node.attrs.entries),
        version=node.op.version,
    )


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
    return node


def _field_json(field: Field, /) -> dict[str, object]:
    return {
        "name": field.name,
        "data_type": field.data_type,
        "nullable": field.nullable,
    }


def _check_declared_inputs(program: Program, /) -> None:
    for value in program.inputs:
        node = value._node
        if node.op.name == "parameter":
            errors.raise_compile(
                f"static_inputs.{_cstr(node.attr('name'))}",
                errors.UNKNOWN_PRIMITIVE_VERSION,
                "static parameters are not supported by the row-local lowerer",
            )


def _lower_program(program: Program, mode: str, /) -> dict[str, object]:
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
                )
            )
            fanout_ids[input_node.digest] = input_name
    for output_name, segment in segments:
        env = dict(segment.env)
        reserved = frozenset(env)
        fused = extract_common(
            tuple((field, env[field]) for field in segment.fields),
            segment.predicate,
            reserved,
        )
        input_field_names = [
            field.name for field in _schema_fields(segment.input_node.attr("schema"))
        ]
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
                if fanout:
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


def lower_program_document(
    program: Program,
    runtime: Runtime,
    mode: str,
    /,
) -> dict[str, object]:
    """Analyze and lower one program to its strict project-v3 document."""

    selected = _require_runtime(runtime, "lower_program_document")
    mode_value = _require_mode(mode)
    _check_expression_capability(program, selected, mode_value)
    return _lower_program(program, mode_value)


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

    Row-local lowering has no stateful late-row surface; the lateness
    arguments are validated and accepted for forward compatibility with the
    stateful stages that consume them.
    """

    selected = _require_runtime(runtime, "compile_stream")
    _validate_lateness(allowed_lateness_micros, late_policy)
    document = lower_program_document(program, selected, "stream")
    return selected._compile_stream_graph_project(
        _canonical(document), requirements=StreamRequirements()
    )


def _validate_lateness(allowed_lateness_micros: object, late_policy: object, /) -> None:
    if type(allowed_lateness_micros) is not int:
        raise TypeError(
            "compile_stream allowed_lateness_micros must be an exact int; got"
            f" {type_name(allowed_lateness_micros)}"
        )
    if allowed_lateness_micros < 0:
        raise ValueError(
            "compile_stream allowed_lateness_micros: invalid_literal: must be"
            " non-negative"
        )
    if type(late_policy) is not str:
        raise TypeError(
            f"compile_stream late_policy must be a string; got {type_name(late_policy)}"
        )
    if late_policy not in ("error", "drop"):
        raise ValueError(
            "compile_stream late_policy: invalid_literal: must be 'error' or"
            f" 'drop'; got {late_policy!r}"
        )
