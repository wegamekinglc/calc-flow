"""Infer exact window-boundary schemas with the native expression planner."""

from __future__ import annotations

from collections.abc import Mapping
from graphlib import TopologicalSorter
from typing import TYPE_CHECKING

import pyarrow as pa

from calc_flow.symbolic.expr import TableExpr, table_input
from calc_flow.symbolic.nodes import Node
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


def _fragment_schema(
    document: Mapping[str, object], runtime: Runtime, source_schema: pa.Schema, /
) -> pa.Schema:
    graph = document["graph"]
    nodes = {node["id"]: node for node in graph["nodes"]}
    parents: dict[str, list[str]] = {name: [] for name in nodes}
    for edge in graph["edges"]:
        parents[edge["target_node"]].append(edge["source_node"])
    schemas: dict[str, pa.Schema] = {}
    for name in TopologicalSorter(parents).static_order():
        dependencies = parents[name]
        if len(dependencies) > 1:
            raise RuntimeError("row schema planning requires one input per stage")
        operator = nodes[name]["operator"]
        if operator["kind"] != "expression" or operator["udfs"]:
            raise RuntimeError("row schema planning requires native expressions")
        schema = schemas[dependencies[0]] if dependencies else source_schema
        schemas[name] = runtime._infer_symbolic_expression_schema(
            operator["select"], operator["filter"], schema
        )
    return schemas["cf_schema_result"]


def infer_table_schema(
    node: Node,
    runtime: Runtime,
    boundaries: Mapping[str, tuple[Field, ...]],
    /,
) -> tuple[Field, ...]:
    """Plan the actual CSE stages without opening sources or executing rows."""
    from calc_flow.symbolic.lower.event_windows import _rewrite_nodes
    from calc_flow.symbolic.lower.program import lower_program_document
    from calc_flow.symbolic.program import Program

    if len(boundaries) != 1:
        raise RuntimeError("a window row path must have exactly one table boundary")
    digest, fields = next(iter(boundaries.items()))
    source = table_input("cf_schema_input", schema=fields)
    rewritten = _rewrite_nodes(node, {digest: source._node})
    fragment = Program(
        "symbolic-schema",
        inputs=[source],
        outputs=[("cf_schema_result", TableExpr(rewritten))],
    )
    document = lower_program_document(fragment, runtime, "stream")
    return _fields(_fragment_schema(document, runtime, _arrow_schema(fields)))
