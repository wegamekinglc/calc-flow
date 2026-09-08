"""Plan exact table schemas through native expression and state boundaries."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from graphlib import TopologicalSorter
from typing import TYPE_CHECKING, Any

import pyarrow as pa

from calc_flow.symbolic.expr import TableExpr, table_input
from calc_flow.symbolic.nodes import CStr, Node, build
from calc_flow.symbolic.types import Field

if TYPE_CHECKING:
    from calc_flow.pipeline import Runtime


def _arrow_schema(fields: tuple[Field, ...], /) -> pa.Schema:
    special = {
        "time32[s]": pa.time32("s"),
        "time64[us]": pa.time64("us"),
        "timestamp[ms]": pa.timestamp("ms"),
        "timestamp[us]": pa.timestamp("us"),
        "timestamp[us, UTC]": pa.timestamp("us", tz="UTC"),
    }
    return pa.schema(
        [
            pa.field(
                field.name,
                special[field.data_type]
                if field.data_type in special
                else pa.type_for_alias(field.data_type),
                nullable=field.nullable,
            )
            for field in fields
        ]
    )


def _fields(schema: pa.Schema, /) -> tuple[Field, ...]:
    aliases = {
        "date32[day]": "date32",
        "date64[ms]": "date64",
        "float": "float32",
        "double": "float64",
        "timestamp[us, tz=UTC]": "timestamp[us, UTC]",
    }
    return tuple(
        Field(
            field.name,
            aliases.get(str(field.type), str(field.type)),
            nullable=field.nullable,
        )
        for field in schema
    )


def _port_schema(ports: Sequence[Mapping[str, Any]], name: str) -> pa.Schema | None:
    for port in ports:
        if port["name"] == name and port.get("schema") is not None:
            return _arrow_schema(tuple(Field(**field) for field in port["schema"]))
    return None


def infer_document_schemas(
    document: Mapping[str, Any],
    runtime: Runtime,
    declared_outputs: Mapping[str, pa.Schema] | None = None,
) -> dict[str, pa.Schema]:
    """Plan expression schemas through declared native state boundaries."""
    graph = document["graph"]
    nodes = {node["id"]: node for node in graph["nodes"]}
    parents: dict[str, dict[str, str]] = {name: {} for name in nodes}
    for edge in graph["edges"]:
        parents[edge["target_node"]][edge["target_port"]] = edge["source_node"]
    schemas: dict[str, pa.Schema] = {}
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
    inputs: Mapping[str, pa.Schema],
    fallback: pa.Schema | None,
) -> pa.Schema:
    operator = node["operator"]
    if operator["kind"] == "expression":
        source = inputs.get("input")
        if source is None:
            source = _port_schema(node.get("input_ports", []), "input")
        return _expression_schema(
            operator, runtime, fallback if source is None else source
        )
    if operator["kind"] == "sql":
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
    operator: Mapping[str, Any], runtime: Runtime, schema: pa.Schema, /
) -> pa.Schema:
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
