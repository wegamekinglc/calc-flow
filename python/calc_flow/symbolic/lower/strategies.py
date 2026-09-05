"""Matrix, stream-join, relational-DAG, and dedup lowering strategies.

Moved verbatim from ``symbolic/lower.py``."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Never

from calc_flow.join_spec import (
    JoinSideWire,
    bounds_wire,
    join_wire_spec,
    limits_wire,
)
from calc_flow.pipeline import (
    _canonical,
    _data_sources,
)
from calc_flow.symbolic import errors
from calc_flow.symbolic.analyzer import (
    TableFacts,
    _Analyzer,
    _schema_fields,
)
from calc_flow.symbolic.lower.planners import (
    _CrossSectionPlan,
    _LoweringProgram,
    _LoweringValue,
    _MatrixExpression,
)
from calc_flow.symbolic.lower.segments import (
    _MATRIX_PRIMITIVES,
    _cint,
    _cstr,
    _cstr_seq,
    _expression_node,
    _field_json,
    _quote_identifier,
    _reject_primitive,
    _RollingPipeline,
)
from calc_flow.symbolic.nodes import (
    CBool,
    CDType,
    CFloat,
    CInt,
    CSeq,
    Node,
    build,
)
from calc_flow.symbolic.types import Field

if TYPE_CHECKING:
    from calc_flow.symbolic.program import Program


def _matrix_literal(node: Node, path: str, /) -> bool | int | float:
    value = node.attr("value")
    if isinstance(value, (CBool, CInt, CFloat)):
        return value.value
    errors.raise_compile(
        path,
        errors.UNSUPPORTED_TYPE,
        "symbolic matrix literals must be finite bool, int, or float values",
    )


def _matrix_backend(
    left: _MatrixExpression,
    right: _MatrixExpression,
    path: str,
    /,
) -> str:
    if left.backend and right.backend and left.backend != right.backend:
        errors.raise_compile(
            path,
            errors.CAPABILITY_MISMATCH,
            "symbolic matrix operands must use one provider backend",
        )
    return left.backend or right.backend


def _matrix_columns(
    left: _MatrixExpression,
    right: _MatrixExpression,
    path: str,
    /,
) -> tuple[str, ...]:
    if left.columns and right.columns and left.columns != right.columns:
        errors.raise_compile(
            path,
            errors.SCHEMA_MISMATCH,
            "symbolic matrix operands must use one ordered column selection",
        )
    return left.columns or right.columns


def _matrix_parameter(
    left: _MatrixExpression,
    right: _MatrixExpression,
    path: str,
    /,
) -> Node | None:
    if (
        left.parameter is not None
        and right.parameter is not None
        and left.parameter.digest != right.parameter.digest
    ):
        errors.raise_compile(
            path,
            errors.CAPABILITY_MISMATCH,
            "one symbolic matrix output supports exactly one static parameter",
        )
    return left.parameter or right.parameter


def _merge_matrix_expression(
    left: _MatrixExpression,
    right: _MatrixExpression,
    path: str,
    /,
) -> tuple[str, tuple[str, ...], Node | None]:
    return (
        _matrix_backend(left, right, path),
        _matrix_columns(left, right, path),
        _matrix_parameter(left, right, path),
    )


def _matrix_leaf_expression(
    node: Node,
    path: str,
    operation: str,
    /,
) -> _MatrixExpression | None:
    if operation == "from_columns":
        return _MatrixExpression(
            _cstr(node.attr("backend")),
            _cstr_seq(node.attr("columns")),
            frozenset({node.args[0].digest}),
            None,
            0,
            0,
            True,
            {"op": "input"},
        )
    if operation == "parameter":
        name = _cstr(node.attr("name"))
        if name != "weights":
            errors.raise_compile(
                f"static_inputs.{name}",
                errors.CAPABILITY_MISMATCH,
                "symbolic matrix lowering currently requires the static array"
                " parameter name 'weights'",
            )
        return _MatrixExpression(
            _cstr(node.attr("backend")),
            (),
            frozenset(),
            node,
            1,
            0,
            True,
            {"op": "weights"},
        )
    if operation == "literal":
        return _MatrixExpression(
            "",
            (),
            frozenset(),
            None,
            0,
            0,
            True,
            {"op": "literal", "value": _matrix_literal(node, path)},
        )
    return None


def _matrix_unary_expression(
    node: Node,
    path: str,
    operation: str,
    /,
) -> _MatrixExpression:
    value = _matrix_expression(node.args[0], f"{path}.{operation}.value")
    return _MatrixExpression(
        value.backend,
        value.columns,
        value.source_digests,
        value.parameter,
        value.weights_count,
        value.matmul_count,
        value.matmul_rhs_is_weights,
        {"op": operation, "value": value.tree},
    )


def _matrix_rhs_is_weights(node: Node, operation: str, /) -> bool:
    return operation != "matmul" or (
        node.args[1].op.name == "parameter"
        and _cstr(node.args[1].attr("name")) == "weights"
    )


def _matrix_binary_expression(
    node: Node,
    path: str,
    operation: str,
    /,
) -> _MatrixExpression:
    left = _matrix_expression(node.args[0], f"{path}.{operation}.left")
    right = _matrix_expression(node.args[1], f"{path}.{operation}.right")
    backend, columns, parameter = _merge_matrix_expression(left, right, path)
    return _MatrixExpression(
        backend,
        columns,
        left.source_digests | right.source_digests,
        parameter,
        left.weights_count + right.weights_count,
        left.matmul_count + right.matmul_count + (operation == "matmul"),
        left.matmul_rhs_is_weights
        and right.matmul_rhs_is_weights
        and _matrix_rhs_is_weights(node, operation),
        {"left": left.tree, "op": operation, "right": right.tree},
    )


def _matrix_expression(node: Node, path: str, /) -> _MatrixExpression:
    operation = node.op.name
    leaf = _matrix_leaf_expression(node, path, operation)
    if leaf is not None:
        return leaf
    if operation not in _MATRIX_PRIMITIVES:
        _reject_primitive(path, node)
    if operation in ("neg", "not"):
        return _matrix_unary_expression(node, path, operation)
    return _matrix_binary_expression(node, path, operation)


def _static_array_declaration(node: Node, /) -> dict[str, object]:
    dtype = node.attr("dtype")
    shape = node.attr("shape")
    return {
        "backend": _cstr(node.attr("backend")),
        "dtype": dtype.name if isinstance(dtype, CDType) else "",
        "kind": "array",
        "mutability": "static",
        "name": _cstr(node.attr("name")),
        "shape": [
            dimension.value
            for dimension in (shape.items if isinstance(shape, CSeq) else ())
            if isinstance(dimension, CInt)
        ],
    }


def _matrix_program_output(
    program: Program | _LoweringProgram,
    /,
) -> tuple[str, Node] | None:
    if len(program.outputs) != 1:
        return None
    output_name, value = program.outputs[0]
    if value._node.op.name != "attach_columns":
        return None
    return output_name, value._node


def _required_matrix_expression(node: Node, output_name: str, /) -> _MatrixExpression:
    path = f"outputs.{output_name}.array"
    matrix = _matrix_expression(node.args[1], path)
    if not matrix.backend or not matrix.columns:
        errors.raise_compile(
            path,
            errors.UNRESOLVED_TYPE,
            "symbolic matrix output requires linalg.from_columns",
        )
    return matrix


def _required_matrix_parameter(
    matrix: _MatrixExpression,
    output_name: str,
    /,
) -> Node:
    parameter = matrix.parameter
    if parameter is None:
        errors.raise_compile(
            f"outputs.{output_name}.array",
            errors.UNRESOLVED_TYPE,
            "symbolic matrix output requires one static array parameter",
        )
    if _cstr(parameter.attr("name")) != "weights":
        errors.raise_compile(
            f"outputs.{output_name}.array",
            errors.UNRESOLVED_TYPE,
            "symbolic matrix output requires one static array parameter",
        )
    return parameter


def _require_frozen_matrix_shape(
    matrix: _MatrixExpression,
    output_name: str,
    /,
) -> None:
    if (
        matrix.weights_count != 1
        or matrix.matmul_count != 1
        or not matrix.matmul_rhs_is_weights
    ):
        errors.raise_compile(
            f"outputs.{output_name}.array",
            errors.CAPABILITY_MISMATCH,
            "symbolic matrix output requires exactly one matmul whose right"
            " operand is the static 'weights' parameter",
        )


def _attached_matrix_input(
    node: Node,
    matrix: _MatrixExpression,
    output_name: str,
    /,
) -> Node:
    attached_input = node.args[0]
    if matrix.source_digests != frozenset({attached_input.digest}):
        errors.raise_compile(
            f"outputs.{output_name}.value",
            errors.SCHEMA_MISMATCH,
            "attached matrix columns must come from the attached table",
        )
    return attached_input


def _matrix_external_node(
    output_name: str,
    node: Node,
    matrix: _MatrixExpression,
    /,
) -> dict[str, object]:
    return {
        "id": output_name,
        "input_ports": [
            {"kind": "table", "name": "input", "required": True},
            {"kind": "array", "name": "weights", "required": True},
        ],
        "operator": {
            "kind": "external",
            "name": "symbolic_matrix",
            "options": {
                "columns": list(matrix.columns),
                "expression": matrix.tree,
                "names": list(_cstr_seq(node.attr("names"))),
            },
            "provider": matrix.backend,
            "version": "1",
        },
        "output_ports": [{"kind": "table", "name": "output", "required": True}],
    }


def _matrix_upstream_project(
    program: Program | _LoweringProgram,
    attached_input: Node,
    upstream_id: str,
    mode: str,
    allowed_lateness_micros: int,
    late_policy: str,
    /,
) -> dict[str, object]:
    upstream = _LoweringProgram(
        program.name,
        tuple(
            _LoweringValue(item._node)
            for item in program.inputs
            if item._node.op.name == "table_input"
        ),
        ((upstream_id, _LoweringValue(attached_input)),),
    )
    from calc_flow.symbolic.lower.program import _lower_program as _row_local_lower

    return _row_local_lower(
        upstream,
        mode,
        allowed_lateness_micros,
        late_policy,
    )


def _raise_lowering_invariant(message: str, /) -> Never:
    raise RuntimeError(f"symbolic lowering invariant violated: {message}")


def _project_graph_lists(
    project: dict[str, object],
    /,
) -> tuple[list[object], list[object]]:
    graph = project.get("graph")
    if not isinstance(graph, dict):
        _raise_lowering_invariant("project.graph must be a mapping")
    nodes = graph.get("nodes")
    if not isinstance(nodes, list):
        _raise_lowering_invariant("project.graph.nodes must be a list")
    edges = graph.get("edges")
    if not isinstance(edges, list):
        _raise_lowering_invariant("project.graph.edges must be a list")
    return nodes, edges


def _project_document(
    name: str,
    mode: str,
    nodes: list[object],
    edges: list[object],
    /,
) -> dict[str, object]:
    project: dict[str, object] = {
        "data_sources": [],
        "format_version": 3,
        "id": name,
        "name": name,
        "runtime": {"mode": mode, "options": {}},
        "graph": {"edges": edges, "name": name, "nodes": nodes},
    }
    project["data_sources"] = [] if mode == "stream" else _data_sources(project)
    return project


def _wire_matrix_node(
    project: dict[str, object],
    external: dict[str, object],
    upstream_id: str,
    output_name: str,
    /,
) -> None:
    nodes, edges = _project_graph_lists(project)
    nodes.append(external)
    edges.append(
        {
            "source_node": upstream_id,
            "source_port": "output",
            "target_node": output_name,
            "target_port": "input",
        }
    )


def _lower_matrix_program(
    program: Program | _LoweringProgram,
    mode: str,
    allowed_lateness_micros: int,
    late_policy: str,
    /,
) -> dict[str, object] | None:
    output = _matrix_program_output(program)
    if output is None:
        return None
    output_name, node = output
    matrix = _required_matrix_expression(node, output_name)
    parameter = _required_matrix_parameter(matrix, output_name)
    _require_frozen_matrix_shape(matrix, output_name)
    attached_input = _attached_matrix_input(node, matrix, output_name)
    external = _matrix_external_node(output_name, node, matrix)
    upstream_id = f"{output_name}__cf_matrix_input"
    project = _matrix_upstream_project(
        program,
        attached_input,
        upstream_id,
        mode,
        allowed_lateness_micros,
        late_policy,
    )
    _wire_matrix_node(project, external, upstream_id, output_name)
    if mode == "stream":
        project["static_inputs"] = [_static_array_declaration(parameter)]
    else:
        project["data_sources"] = _data_sources(project)
    return project


def _stream_join_nodes(program: Program, /) -> tuple[Node, ...]:
    joins: dict[str, Node] = {}
    for _, value in program.outputs:
        for node in _walk_nodes(value._node):
            if node.op.name == "stream_join":
                joins[node.digest] = node
    return tuple(joins[digest] for digest in sorted(joins))


def _contains_node(root: Node, digest: str, /) -> bool:
    return any(node.digest == digest for node in _walk_nodes(root))


def _contains_primitive(root: Node, primitive: str, /) -> bool:
    return any(node.op.name == primitive for node in _walk_nodes(root))


def _replace_node(root: Node, digest: str, replacement: Node, /) -> Node:
    if root.digest == digest:
        return replacement
    return build(
        root.op.name,
        tuple(_replace_node(child, digest, replacement) for child in root.args),
        dict(root.attrs.entries),
        version=root.op.version,
    )


def _stream_join_node(
    node: Node,
    node_id: str,
    left_schema: tuple[Field, ...],
    right_schema: tuple[Field, ...],
    /,
) -> dict[str, object]:
    return {
        "id": node_id,
        "input_ports": [
            {
                "kind": "table",
                "name": "left",
                "required": True,
                "schema": [_field_json(field) for field in left_schema],
            },
            {
                "kind": "table",
                "name": "right",
                "required": True,
                "schema": [_field_json(field) for field in right_schema],
            },
        ],
        "operator": {
            "kind": "stream_join",
            "spec": join_wire_spec(
                JoinSideWire(
                    keys=_cstr_seq(node.attr("left_keys")),
                    event_time=_cstr(node.attr("left_event_time")),
                    prefix=_cstr(node.attr("left_prefix")),
                ),
                JoinSideWire(
                    keys=_cstr_seq(node.attr("right_keys")),
                    event_time=_cstr(node.attr("right_event_time")),
                    prefix=_cstr(node.attr("right_prefix")),
                ),
                bounds_wire(
                    _cint(node.attr("before_micros")),
                    _cint(node.attr("after_micros")),
                ),
                limits_wire(
                    _cint(node.attr("max_state_rows_per_side")),
                    _cint(node.attr("max_state_bytes_per_side")),
                    _cint(node.attr("max_matches_per_input_batch")),
                ),
            ),
        },
        "output_ports": [],
    }


@dataclass(frozen=True, slots=True)
class _StreamJoinSide:
    port: str
    node_id: str
    node: Node
    schema: tuple[Field, ...]


@dataclass(frozen=True, slots=True)
class _StreamJoinPlan:
    node: Node
    node_id: str
    sides: tuple[_StreamJoinSide, _StreamJoinSide]
    output_schema: tuple[Field, ...]


@dataclass(frozen=True, slots=True)
class _RelationalFragment:
    boundary: Node
    project: dict[str, object] | None


def _connected_input_endpoints(edges: list[object], /) -> set[tuple[str, str]]:
    connected: set[tuple[str, str]] = set()
    for edge in edges:
        if not isinstance(edge, dict):
            continue
        connected.add((str(edge["target_node"]), str(edge.get("target_port", "input"))))
    return connected


def _required_input_port_names(node: dict[str, object], /) -> tuple[str, ...]:
    ports = node.get("input_ports")
    if not isinstance(ports, list):
        return ("input",)
    if not ports:
        return ("input",)
    names: list[str] = []
    for port in ports:
        if not isinstance(port, dict):
            continue
        if port.get("required") is True:
            names.append(str(port["name"]))
    return tuple(names)


def _graph_node(node: object, /) -> dict[str, object]:
    if not isinstance(node, dict):
        _raise_lowering_invariant("graph node must be a mapping")
    return node


def _downstream_input_endpoints(
    nodes: list[object], edges: list[object], /
) -> tuple[tuple[str, str], ...]:
    connected = _connected_input_endpoints(edges)
    endpoints: list[tuple[str, str]] = []
    for raw_node in nodes:
        node = _graph_node(raw_node)
        node_id = str(node["id"])
        for name in _required_input_port_names(node):
            endpoint = (node_id, name)
            if endpoint not in connected:
                endpoints.append(endpoint)
    return tuple(sorted(endpoints))


def _pin_table_output(
    nodes: list[object], node_id: str, schema: tuple[Field, ...], /
) -> None:
    for node in nodes:
        if isinstance(node, dict) and node.get("id") == node_id:
            node["output_ports"] = [
                {
                    "kind": "table",
                    "name": "output",
                    "required": True,
                    "schema": [_field_json(field) for field in schema],
                }
            ]
            return
    _raise_lowering_invariant(f"missing stream join input stage {node_id!r}")


def _required_stream_join(program: Program, joins: tuple[Node, ...], /) -> Node:
    if len(joins) != 1:
        errors.raise_compile(
            program.name,
            errors.CAPABILITY_MISMATCH,
            "SCE-17 supports exactly one unique symbolic stream join per program",
        )
    return joins[0]


def _check_stream_join_outputs(program: Program, join: Node, /) -> None:
    for output_name, value in program.outputs:
        if _contains_node(value._node, join.digest):
            continue
        errors.raise_compile(
            f"outputs.{output_name}",
            errors.CAPABILITY_MISMATCH,
            "every output in a symbolic stream-join program must descend"
            " from its one shared join",
        )


def _check_stream_join_inputs(program: Program, join: Node, /) -> None:
    for side_name, side in zip(("left", "right"), join.args, strict=True):
        if not _contains_primitive(side, "attach_columns"):
            continue
        errors.raise_compile(
            f"{program.name}.stream_join.{side_name}",
            errors.CAPABILITY_MISMATCH,
            "matrix attachment around a symbolic stream join is not supported",
        )


def _stream_join_plan(
    program: Program, analyzer: _Analyzer, join: Node, /
) -> _StreamJoinPlan:
    left_facts = analyzer.table(join.args[0], f"{program.name}.stream_join.left")
    right_facts = analyzer.table(join.args[1], f"{program.name}.stream_join.right")
    join_facts = analyzer.table(join, f"{program.name}.stream_join")
    join_id = f"cf_stream_join_{join.digest[:16]}"
    return _StreamJoinPlan(
        join,
        join_id,
        (
            _StreamJoinSide(
                "left",
                f"{join_id}__left",
                join.args[0],
                left_facts.schema,
            ),
            _StreamJoinSide(
                "right",
                f"{join_id}__right",
                join.args[1],
                right_facts.schema,
            ),
        ),
        join_facts.schema,
    )


def _stream_join_upstream_outputs(
    plan: _StreamJoinPlan, /
) -> tuple[tuple[str, _LoweringValue], ...]:
    return tuple(
        (side.node_id, _LoweringValue(side.node))
        for side in plan.sides
        if side.node.op.name != "table_input"
    )


def _input_reaches_join_side(
    input_node: Node, upstream_sides: tuple[Node, ...], /
) -> bool:
    return any(_contains_node(side, input_node.digest) for side in upstream_sides)


def _stream_join_upstream_inputs(
    program: Program,
    upstream_outputs: tuple[tuple[str, _LoweringValue], ...],
    /,
) -> tuple[_LoweringValue, ...]:
    upstream_sides = tuple(value._node for _, value in upstream_outputs)
    inputs: list[_LoweringValue] = []
    for value in program.inputs:
        if value._node.op.name != "table_input":
            continue
        if _input_reaches_join_side(value._node, upstream_sides):
            inputs.append(_LoweringValue(value._node))
    return tuple(inputs)


def _stream_join_upstream_project(
    program: Program,
    plan: _StreamJoinPlan,
    mode: str,
    allowed_lateness_micros: int,
    late_policy: str,
    /,
) -> dict[str, object]:
    outputs = _stream_join_upstream_outputs(plan)
    if not outputs:
        return _project_document(program.name, mode, [], [])
    inputs = _stream_join_upstream_inputs(program, outputs)
    from calc_flow.symbolic.lower.program import _lower_program as _row_local_lower

    return _row_local_lower(
        _LoweringProgram(program.name, inputs, outputs),
        mode,
        allowed_lateness_micros,
        late_policy,
    )


def _wire_stream_join_inputs(
    nodes: list[object], edges: list[object], plan: _StreamJoinPlan, /
) -> None:
    for side in plan.sides:
        if side.node.op.name == "table_input":
            continue
        _pin_table_output(nodes, side.node_id, side.schema)
        edges.append(
            {
                "source_node": side.node_id,
                "source_port": "output",
                "target_node": plan.node_id,
                "target_port": side.port,
            }
        )


def _direct_stream_join_output(program: Program, plan: _StreamJoinPlan, /) -> bool:
    if len(program.outputs) != 1:
        return False
    return program.outputs[0][1]._node.digest == plan.node.digest


def _wire_stream_join_downstream(
    program: Program,
    plan: _StreamJoinPlan,
    mode: str,
    allowed_lateness_micros: int,
    late_policy: str,
    upstream_nodes: list[object],
    upstream_edges: list[object],
    /,
) -> None:
    from calc_flow.symbolic.expr import table_input

    virtual_id = f"{plan.node_id}__output"
    virtual = table_input(virtual_id, schema=plan.output_schema)
    downstream = _LoweringProgram(
        program.name,
        (_LoweringValue(virtual._node),),
        tuple(
            (
                output_name,
                _LoweringValue(
                    _replace_node(value._node, plan.node.digest, virtual._node)
                ),
            )
            for output_name, value in program.outputs
        ),
    )
    from calc_flow.symbolic.lower.program import _lower_program as _row_local_lower

    downstream_project = _row_local_lower(
        downstream,
        mode,
        allowed_lateness_micros,
        late_policy,
    )
    downstream_nodes, downstream_edges = _project_graph_lists(downstream_project)
    endpoints = _downstream_input_endpoints(downstream_nodes, downstream_edges)
    if not endpoints:
        _raise_lowering_invariant("stream join downstream has no input endpoint")
    upstream_nodes.extend(downstream_nodes)
    upstream_edges.extend(downstream_edges)
    upstream_edges.extend(
        {
            "source_node": plan.node_id,
            "source_port": "output",
            "target_node": target_node,
            "target_port": target_port,
        }
        for target_node, target_port in endpoints
    )


def _lower_stream_join_program(
    program: Program,
    analyzer: _Analyzer,
    mode: str,
    allowed_lateness_micros: int,
    late_policy: str,
    /,
) -> dict[str, object] | None:
    joins = _stream_join_nodes(program)
    if not joins:
        return None
    if _requires_relational_dag_lowering(program, joins):
        return _lower_relational_dag_program(
            program,
            analyzer,
            mode,
            allowed_lateness_micros,
            late_policy,
            joins,
        )
    join = _required_stream_join(program, joins)
    _check_stream_join_outputs(program, join)
    _check_stream_join_inputs(program, join)
    plan = _stream_join_plan(program, analyzer, join)
    upstream_project = _stream_join_upstream_project(
        program,
        plan,
        mode,
        allowed_lateness_micros,
        late_policy,
    )
    upstream_nodes, upstream_edges = _project_graph_lists(upstream_project)
    _wire_stream_join_inputs(upstream_nodes, upstream_edges, plan)
    upstream_nodes.append(
        _stream_join_node(
            plan.node,
            plan.node_id,
            plan.sides[0].schema,
            plan.sides[1].schema,
        )
    )
    if _direct_stream_join_output(program, plan):
        return upstream_project
    _wire_stream_join_downstream(
        program,
        plan,
        mode,
        allowed_lateness_micros,
        late_policy,
        upstream_nodes,
        upstream_edges,
    )
    return upstream_project


def _requires_relational_dag_lowering(
    program: Program, joins: tuple[Node, ...], /
) -> bool:
    if len(joins) != 1 or joins[0].op.version >= 2:
        return True
    join = joins[0]
    return any(
        not _contains_node(value._node, join.digest) for _, value in program.outputs
    )


def _relational_boundary(node: Node, path: str, /) -> Node:
    current = node
    while current.op.name in ("project", "filter", "with_columns"):
        current = current.args[0]
    if current.op.name in ("table_input", "stream_join"):
        return current
    if _contains_primitive(current, "attach_columns"):
        errors.raise_compile(
            path,
            errors.CAPABILITY_MISMATCH,
            "matrix attachment around a symbolic stream join is not supported",
        )
    _reject_primitive(path, current)


def _virtual_relational_input(node_id: str, facts: TableFacts, /) -> Node:
    from calc_flow.symbolic.expr import table_input

    return table_input(
        node_id,
        schema=facts.schema,
        entity_by=facts.entity_by,
        event_time=facts.event_time,
        sequence_by=facts.sequence_by,
    )._node


def _reachable_relational_sources(program: Program, /) -> frozenset[str]:
    return frozenset(
        node.digest
        for _, value in program.outputs
        for node in _walk_nodes(value._node)
        if node.op.name == "table_input"
    )


def _relational_source_name(node: Node, reserved_ids: frozenset[str], /) -> str:
    declared_name = _cstr(node.attr("name"))
    if declared_name is None:
        _raise_lowering_invariant("relational source is missing its declared name")
    if declared_name in reserved_ids:
        return f"cf_source_{node.digest[:16]}"
    return declared_name


def _relational_source_nodes(
    program: Program, reserved_ids: frozenset[str], /
) -> tuple[dict[str, str], list[dict[str, object]]]:
    reachable = _reachable_relational_sources(program)
    by_digest: dict[str, str] = {}
    nodes: list[dict[str, object]] = []
    for value in program.inputs:
        node = value._node
        if node.op.name != "table_input" or node.digest not in reachable:
            continue
        name = _relational_source_name(node, reserved_ids)
        schema = _schema_fields(node.attr("schema"))
        by_digest[node.digest] = name
        nodes.append(
            _expression_node(
                name,
                [_quote_identifier(field.name) for field in schema],
                None,
                schema,
                schema,
            )
        )
    return by_digest, nodes


def _relational_upstream_id(
    boundary: Node,
    sources: dict[str, str],
    joins: dict[str, _StreamJoinPlan],
    path: str,
    /,
) -> str:
    if boundary.op.name == "table_input":
        source_id = sources.get(boundary.digest)
        if source_id is None:
            _raise_lowering_invariant(
                f"missing declared source for relational boundary at {path}"
            )
        return source_id
    plan = joins.get(boundary.digest)
    if plan is None:
        _raise_lowering_invariant(
            f"missing physical join for relational boundary at {path}"
        )
    return plan.node_id


def _relational_fragment(
    program: Program,
    analyzer: _Analyzer,
    expression: Node,
    boundary: Node,
    output_id: str,
    path: str,
    mode: str,
    allowed_lateness_micros: int,
    late_policy: str,
    /,
) -> dict[str, object]:
    if boundary.op.name == "stream_join":
        facts = analyzer.table(boundary, f"{path}.boundary")
        declared = _virtual_relational_input(
            f"cf_join_output_{boundary.digest[:16]}", facts
        )
        lowered = _replace_node(expression, boundary.digest, declared)
    else:
        declared = boundary
        lowered = expression
    from calc_flow.symbolic.lower.program import _lower_program as _row_local_lower

    return _row_local_lower(
        _LoweringProgram(
            program.name,
            (_LoweringValue(declared),),
            ((output_id, _LoweringValue(lowered)),),
        ),
        mode,
        allowed_lateness_micros,
        late_policy,
    )


def _relational_output_fragments(
    program: Program,
    analyzer: _Analyzer,
    mode: str,
    allowed_lateness_micros: int,
    late_policy: str,
    /,
) -> dict[str, _RelationalFragment]:
    fragments: dict[str, _RelationalFragment] = {}
    for output_name, value in program.outputs:
        path = f"outputs.{output_name}"
        boundary = _relational_boundary(value._node, path)
        fragments[output_name] = _RelationalFragment(
            boundary,
            _relational_fragment(
                program,
                analyzer,
                value._node,
                boundary,
                output_name,
                path,
                mode,
                allowed_lateness_micros,
                late_policy,
            ),
        )
    return fragments


def _relational_join_fragments(
    program: Program,
    analyzer: _Analyzer,
    plans: dict[str, _StreamJoinPlan],
    mode: str,
    allowed_lateness_micros: int,
    late_policy: str,
    /,
) -> dict[tuple[str, str], _RelationalFragment]:
    fragments: dict[tuple[str, str], _RelationalFragment] = {}
    for plan in plans.values():
        for side in plan.sides:
            path = f"{program.name}.{plan.node_id}.{side.port}"
            boundary = _relational_boundary(side.node, path)
            project = None
            if side.node.digest != boundary.digest:
                project = _relational_fragment(
                    program,
                    analyzer,
                    side.node,
                    boundary,
                    side.node_id,
                    path,
                    mode,
                    allowed_lateness_micros,
                    late_policy,
                )
            fragments[(plan.node.digest, side.port)] = _RelationalFragment(
                boundary,
                project,
            )
    return fragments


def _wire_relational_fragment(
    nodes: list[object],
    edges: list[object],
    fragment: dict[str, object],
    upstream_id: str,
    output_id: str,
    output_schema: tuple[Field, ...] | None,
    /,
) -> None:
    fragment_nodes, fragment_edges = _project_graph_lists(fragment)
    endpoints = _downstream_input_endpoints(fragment_nodes, fragment_edges)
    if len(endpoints) != 1:
        _raise_lowering_invariant(
            f"relational fragment {output_id!r} has {len(endpoints)} input endpoints"
        )
    if output_schema is not None:
        _pin_table_output(fragment_nodes, output_id, output_schema)
    nodes.extend(fragment_nodes)
    edges.extend(fragment_edges)
    target_node, target_port = endpoints[0]
    edges.append(
        {
            "source_node": upstream_id,
            "source_port": "output",
            "target_node": target_node,
            "target_port": target_port,
        }
    )


def _wire_relational_join_side(
    program: Program,
    plan: _StreamJoinPlan,
    side: _StreamJoinSide,
    fragment: _RelationalFragment,
    sources: dict[str, str],
    joins: dict[str, _StreamJoinPlan],
    nodes: list[object],
    edges: list[object],
    /,
) -> None:
    path = f"{program.name}.{plan.node_id}.{side.port}"
    upstream_id = _relational_upstream_id(fragment.boundary, sources, joins, path)
    if fragment.project is None:
        edges.append(
            {
                "source_node": upstream_id,
                "source_port": "output",
                "target_node": plan.node_id,
                "target_port": side.port,
            }
        )
        return
    _wire_relational_fragment(
        nodes,
        edges,
        fragment.project,
        upstream_id,
        side.node_id,
        side.schema,
    )
    edges.append(
        {
            "source_node": side.node_id,
            "source_port": "output",
            "target_node": plan.node_id,
            "target_port": side.port,
        }
    )


def _wire_relational_output(
    output_name: str,
    fragment: _RelationalFragment,
    sources: dict[str, str],
    joins: dict[str, _StreamJoinPlan],
    nodes: list[object],
    edges: list[object],
    /,
) -> None:
    path = f"outputs.{output_name}"
    upstream_id = _relational_upstream_id(fragment.boundary, sources, joins, path)
    if fragment.project is None:
        _raise_lowering_invariant(f"relational output {output_name!r} has no fragment")
    _wire_relational_fragment(
        nodes,
        edges,
        fragment.project,
        upstream_id,
        output_name,
        None,
    )


def _relational_join_plans(
    program: Program,
    analyzer: _Analyzer,
    join_nodes: tuple[Node, ...],
    /,
) -> dict[str, _StreamJoinPlan]:
    for join in join_nodes:
        _check_stream_join_inputs(program, join)
    return {
        join.digest: _stream_join_plan(program, analyzer, join) for join in join_nodes
    }


def _relational_reserved_ids(
    program: Program,
    plans: dict[str, _StreamJoinPlan],
    join_fragments: dict[tuple[str, str], _RelationalFragment],
    output_fragments: dict[str, _RelationalFragment],
    /,
) -> frozenset[str]:
    reserved = {
        *(output_name for output_name, _ in program.outputs),
        *(plan.node_id for plan in plans.values()),
        *(side.node_id for plan in plans.values() for side in plan.sides),
    }
    for fragment in (*join_fragments.values(), *output_fragments.values()):
        if fragment.project is None:
            continue
        nodes, _ = _project_graph_lists(fragment.project)
        reserved.update(str(_graph_node(node)["id"]) for node in nodes)
    return frozenset(reserved)


def _append_relational_joins(
    program: Program,
    join_nodes: tuple[Node, ...],
    sources: dict[str, str],
    plans: dict[str, _StreamJoinPlan],
    fragments: dict[tuple[str, str], _RelationalFragment],
    nodes: list[object],
    edges: list[object],
    /,
) -> None:
    for join in join_nodes:
        plan = plans[join.digest]
        nodes.append(
            _stream_join_node(
                plan.node,
                plan.node_id,
                plan.sides[0].schema,
                plan.sides[1].schema,
            )
        )
        for side in plan.sides:
            _wire_relational_join_side(
                program,
                plan,
                side,
                fragments[(plan.node.digest, side.port)],
                sources,
                plans,
                nodes,
                edges,
            )


def _append_relational_outputs(
    program: Program,
    sources: dict[str, str],
    plans: dict[str, _StreamJoinPlan],
    fragments: dict[str, _RelationalFragment],
    nodes: list[object],
    edges: list[object],
    /,
) -> None:
    for output_name, _ in program.outputs:
        _wire_relational_output(
            output_name,
            fragments[output_name],
            sources,
            plans,
            nodes,
            edges,
        )


def _lower_relational_dag_program(
    program: Program,
    analyzer: _Analyzer,
    mode: str,
    allowed_lateness_micros: int,
    late_policy: str,
    join_nodes: tuple[Node, ...],
    /,
) -> dict[str, object]:
    plans = _relational_join_plans(program, analyzer, join_nodes)
    join_fragments = _relational_join_fragments(
        program,
        analyzer,
        plans,
        mode,
        allowed_lateness_micros,
        late_policy,
    )
    output_fragments = _relational_output_fragments(
        program,
        analyzer,
        mode,
        allowed_lateness_micros,
        late_policy,
    )
    reserved_ids = _relational_reserved_ids(
        program,
        plans,
        join_fragments,
        output_fragments,
    )
    sources, source_nodes = _relational_source_nodes(program, reserved_ids)
    nodes: list[object] = list(source_nodes)
    edges: list[object] = []
    _append_relational_joins(
        program,
        join_nodes,
        sources,
        plans,
        join_fragments,
        nodes,
        edges,
    )
    _append_relational_outputs(
        program,
        sources,
        plans,
        output_fragments,
        nodes,
        edges,
    )
    typed_nodes = [_graph_node(node) for node in nodes]
    typed_edges = [_graph_node(edge) for edge in edges]
    typed_nodes, typed_edges = _deduplicate_node_ids(typed_nodes, typed_edges)
    return _project_document(program.name, mode, typed_nodes, typed_edges)


def _required_segment_state_plan(
    rolling: _RollingPipeline | None, cross: _CrossSectionPlan | None, /
) -> _RollingPipeline | _CrossSectionPlan:
    plan = rolling if rolling is not None else cross
    if plan is None:
        raise RuntimeError("missing state plan for a stateful symbolic segment")
    return plan


def _walk_nodes(root: Node, /) -> tuple[Node, ...]:
    nodes: list[Node] = []

    def visit(node: Node) -> None:
        nodes.append(node)
        for child in node.args:
            visit(child)

    visit(root)
    return tuple(nodes)


def _deduplicated_expression_signature(
    node: dict[str, object],
    incoming: dict[str, list[dict[str, object]]],
    aliases: dict[str, str],
    output_ids: frozenset[str],
    /,
) -> str | None:
    node_id = str(node["id"])
    operator = node["operator"]
    if not isinstance(operator, dict):
        return None
    if operator.get("kind") != "expression" or node_id in output_ids:
        return None
    node_incoming = incoming.get(node_id, [])
    if len(node_incoming) != 1:
        return None
    edge = node_incoming[0]
    source_node = str(edge["source_node"])
    source = aliases.get(source_node, source_node)
    return _canonical(
        {
            "input_ports": node.get("input_ports", []),
            "operator": operator,
            "output_ports": node.get("output_ports", []),
            "source_node": source,
            "source_port": edge.get("source_port", "output"),
            "target_port": edge.get("target_port", "input"),
        }
    )


def _resolve_node_alias(node_id: str, aliases: dict[str, str], /) -> str:
    while node_id in aliases:
        node_id = aliases[node_id]
    return node_id


def _rewrite_deduplicated_edges(
    edges: list[dict[str, object]], aliases: dict[str, str], /
) -> list[dict[str, object]]:
    rewritten: list[dict[str, object]] = []
    seen: set[tuple[str, str, str, str]] = set()
    for edge in edges:
        target = str(edge["target_node"])
        if target in aliases:
            continue
        source = _resolve_node_alias(str(edge["source_node"]), aliases)
        normalized = {**edge, "source_node": source, "target_node": target}
        identity = (
            source,
            str(normalized.get("source_port", "output")),
            target,
            str(normalized.get("target_port", "input")),
        )
        if identity in seen:
            continue
        seen.add(identity)
        rewritten.append(normalized)
    return rewritten


def _deduplicate_pure_expression_nodes(
    nodes: list[dict[str, object]],
    edges: list[dict[str, object]],
    output_ids: frozenset[str],
    /,
) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    """Share identical connected expression stages across output branches."""

    incoming: dict[str, list[dict[str, object]]] = {}
    for edge in edges:
        incoming.setdefault(str(edge["target_node"]), []).append(edge)
    aliases: dict[str, str] = {}
    signatures: dict[str, str] = {}
    kept: list[dict[str, object]] = []
    for node in nodes:
        node_id = str(node["id"])
        signature = _deduplicated_expression_signature(
            node, incoming, aliases, output_ids
        )
        if signature is None:
            kept.append(node)
            continue
        canonical = signatures.get(signature)
        if canonical is None:
            signatures[signature] = node_id
            kept.append(node)
        else:
            aliases[node_id] = canonical
    return kept, _rewrite_deduplicated_edges(edges, aliases)


def _deduplicate_node_ids(
    nodes: list[dict[str, object]], edges: list[dict[str, object]], /
) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    """Collapse repeated references to one already-shared physical stage."""

    by_id: dict[str, str] = {}
    unique_nodes: list[dict[str, object]] = []
    for node in nodes:
        node_id = str(node["id"])
        canonical = _canonical(node)
        existing = by_id.get(node_id)
        if existing is None:
            by_id[node_id] = canonical
            unique_nodes.append(node)
        elif existing != canonical:
            raise RuntimeError(
                f"symbolic optimizer emitted conflicting node id {node_id!r}"
            )
    unique_edges: list[dict[str, object]] = []
    seen: set[tuple[str, str, str, str]] = set()
    for edge in edges:
        identity = (
            str(edge["source_node"]),
            str(edge.get("source_port", "output")),
            str(edge["target_node"]),
            str(edge.get("target_port", "input")),
        )
        if identity not in seen:
            seen.add(identity)
            unique_edges.append(edge)
    return unique_nodes, unique_edges
