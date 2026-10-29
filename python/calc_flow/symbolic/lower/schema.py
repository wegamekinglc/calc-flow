"""Plan exact table schemas through native expression and state boundaries."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from graphlib import TopologicalSorter
from typing import TYPE_CHECKING, Any

from calc_flow import _native
from calc_flow.symbolic.expr import TableExpr, table_input
from calc_flow.symbolic.nodes import CStr, Node, build
from calc_flow.symbolic.types import Field

if TYPE_CHECKING:
    from calc_flow.pipeline import Runtime


def _arrow_schema(fields: tuple[Field, ...], /) -> _native._ArrowSchema:
    return _native._ArrowSchema([(f.name, f.data_type, f.nullable) for f in fields])


def _fields(schema: object, /) -> tuple[Field, ...]:
    return tuple(Field(*field) for field in _native._schema_fields(schema))


def _port_schema(
    ports: Sequence[Mapping[str, Any]], name: str
) -> _native._ArrowSchema | None:
    for port in ports:
        if port["name"] == name and port.get("schema") is not None:
            return _arrow_schema(tuple(Field(**field) for field in port["schema"]))
    return None


def infer_document_schemas(
    document: Mapping[str, Any],
    runtime: Runtime,
    declared_outputs: Mapping[str, _native._ArrowSchema] | None = None,
) -> dict[str, _native._ArrowSchema]:
    """Plan expression schemas through declared native state boundaries."""
    graph = document["graph"]
    nodes = {node["id"]: node for node in graph["nodes"]}
    parents: dict[str, dict[str, str]] = {name: {} for name in nodes}
    for edge in graph["edges"]:
        parents[edge["target_node"]][edge["target_port"]] = edge["source_node"]
    schemas: dict[str, _native._ArrowSchema] = {}
    order = TopologicalSorter(
        {name: set(inputs.values()) for name, inputs in parents.items()}
    )
    for name in order.static_order():
        inputs = {port: schemas[parent] for port, parent in parents[name].items()}
        declared = None if declared_outputs is None else declared_outputs.get(name)
        schemas[name] = _node_schema(nodes[name], runtime, inputs, declared)
    return schemas


def _node_schema(
    node: Mapping[str, Any],
    runtime: Runtime,
    inputs: Mapping[str, _native._ArrowSchema],
    fallback: _native._ArrowSchema | None,
) -> _native._ArrowSchema:
    operator = node["operator"]
    if operator["kind"] == "expression":
        source = inputs.get("input")
        if source is None:
            source = _port_schema(node.get("input_ports", []), "input")
        return _expression_schema(
            operator, runtime, fallback if source is None else source
        )
    if operator["kind"] in {"sql", "snapshot_sql"}:
        return runtime._infer_symbolic_sql_schema(operator["query"], inputs)
    declared = _port_schema(node.get("output_ports", []), "output")
    if declared is None:
        declared = fallback
    if declared is None:
        raise RuntimeError(
            f"native {operator['kind']} schema boundary has no table schema"
        )
    return declared


def _expression_schema(
    operator: Mapping[str, Any], runtime: Runtime, schema: _native._ArrowSchema, /
) -> _native._ArrowSchema:
    if operator["kind"] != "expression" or operator["udfs"]:
        raise RuntimeError("row schema planning requires native expressions")
    return runtime._infer_symbolic_expression_schema(
        operator["select"], operator["filter"], schema
    )


def infer_table_schema(
    node: Node,
    runtime: Runtime,
    boundaries: Mapping[str, tuple[Field, ...]],
    /,
    *,
    mode: str = "stream",
) -> tuple[Field, ...]:
    """Plan the actual CSE stages without opening sources or executing rows."""
    from calc_flow.symbolic.lower.event_windows import _rewrite_nodes
    from calc_flow.symbolic.lower.program import lower_program_document
    from calc_flow.symbolic.program import Program

    sources = _schema_inputs(node, boundaries)
    rewritten = _rewrite_nodes(node, sources)
    fragment = Program(
        "symbolic-schema",
        engine="streaming" if mode == "stream" else "sql",
        outputs=[("cf_schema_result", TableExpr(rewritten))],
    )
    document = lower_program_document(fragment, runtime, mode)
    return _fields(infer_document_schemas(document, runtime)["cf_schema_result"])


def _schema_inputs(
    node: Node, boundaries: Mapping[str, tuple[Field, ...]]
) -> dict[str, Node]:
    sources: dict[str, Node] = {}
    pending = [node]
    visited: set[str] = set()
    while pending:
        current = pending.pop()
        if current.digest in visited:
            continue
        visited.add(current.digest)
        if current.digest not in boundaries:
            pending.extend(current.args)
            continue
        name = f"cf_schema_{len(sources)}"
        if current.op.name == "table_input":
            sources[current.digest] = build(
                "table_input", (), {**dict(current.attrs.entries), "name": CStr(name)}
            )
        else:
            sources[current.digest] = table_input(
                name, schema=boundaries[current.digest]
            )._node
    return sources
