"""Preserve a paired declaration as one native state owner with two ports."""

from __future__ import annotations

from typing import TYPE_CHECKING

from calc_flow.symbolic.errors import AMBIGUOUS_LATE_STAGE, raise_compile
from calc_flow.symbolic.expr import TableExpr, table_input
from calc_flow.symbolic.late_output import late_schema
from calc_flow.symbolic.lower.bindings import _BatchBindings
from calc_flow.symbolic.lower.event_windows import _rewrite_nodes, _table_port
from calc_flow.symbolic.lower.schema import _arrow_schema, infer_document_schemas
from calc_flow.symbolic.lower.segments import _cint, _expression_node, _quote_identifier
from calc_flow.symbolic.nodes import Node, build
from calc_flow.symbolic.program import Program

if TYPE_CHECKING:
    from calc_flow.symbolic.lower.sql import _SQLGraph


def append_late_outputs(graph: _SQLGraph, node: Node) -> str:
    """Splice one validated paired declaration into the native document.

    The fragment is lowered with ``late_policy="drop"`` so the shared state
    stage plans as usual; its single state node is then flipped to
    ``side_output`` and wired to the appended ``late`` port and exit node.
    """
    from calc_flow.symbolic.lower.program import lower_program_document
    from calc_flow.symbolic.lower.sql import _pin_fragment_schemas

    owner = node if node.op.name == "late_output" else node.args[0]
    value = owner.args[0]
    source = value.args[0]
    upstream = graph.table(source)
    facts = graph.analyzer.table(source, "late_output.input")
    virtual = table_input(
        graph.graph.allocate(f"cf_late_input_{owner.digest[:24]}"),
        schema=facts.schema,
        entity_by=facts.entity_by,
        event_time=facts.event_time,
        sequence_by=facts.sequence_by,
    )
    normal_id = graph.graph.allocate(f"cf_late_output_{owner.digest[:24]}")
    late_id = graph.graph.allocate(f"cf_late_rows_{owner.digest[:24]}")
    fragment = Program(
        graph.graph.program.name,
        outputs={
            normal_id: TableExpr(_rewrite_nodes(value, {source.digest: virtual._node}))
        },
    )
    bindings = _BatchBindings()
    document = lower_program_document(
        fragment,
        graph.analyzer._runtime,
        "stream",
        allowed_lateness_micros=_cint(owner.attr("allowed_lateness_micros")),
        late_policy="drop",
        _bindings=bindings,
    )
    normal_fields = graph.analyzer.table(owner, "late_output.output").schema
    schemas = infer_document_schemas(
        document, graph.analyzer._runtime, {normal_id: _arrow_schema(normal_fields)}
    )
    _pin_fragment_schemas(document, schemas)
    states = [
        item
        for item in document["graph"]["nodes"]
        if item["operator"]["kind"] in {"rolling", "cross_section"}
    ]
    if len(states) != 1:
        raise_compile(
            "late_output",
            AMBIGUOUS_LATE_STAGE,
            "late lowering requires exactly one rolling or cross-section stage;"
            f" found {len(states)}",
        )
    (state,) = states
    state["operator"]["spec"]["late_policy"] = {
        "kind": "side_output",
        "metrics_version": 1,
        "schema_version": 1,
    }
    fields = late_schema(facts.schema)
    state["output_ports"].append(_table_port(fields, "late"))
    document["graph"]["nodes"].append(
        _expression_node(
            late_id,
            [_quote_identifier(field.name) for field in fields],
            None,
            fields,
            fields,
        )
    )
    document["graph"]["edges"].append(
        {
            "source_node": state["id"],
            "source_port": "late",
            "target_node": late_id,
            "target_port": "input",
        }
    )
    graph._append_fragment(
        document,
        bindings,
        {virtual._node.attr("name").value: upstream},
        {normal_id: normal_id, late_id: late_id},
    )
    late = build("late_rows", (owner,), {})
    graph.schemas[owner.digest] = normal_fields
    graph.schemas[late.digest] = fields
    graph._remember(owner, normal_id)
    graph._remember(late, late_id)
    return graph.materialized[node.digest]
