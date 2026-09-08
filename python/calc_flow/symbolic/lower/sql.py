"""Compose lazy SQL boundaries and expression fragments into one native graph."""

from __future__ import annotations

from collections.abc import Mapping
from typing import TYPE_CHECKING, Any

from calc_flow.symbolic.analyzer import _Analyzer
from calc_flow.symbolic.expr import TableExpr, table_input
from calc_flow.symbolic.lower.bindings import _BatchBindings
from calc_flow.symbolic.lower.event_windows import (
    _rewrite_nodes,
    _table_port,
    _WindowGraph,
)
from calc_flow.symbolic.lower.schema import (
    _arrow_schema,
    _fields,
    infer_document_schemas,
)
from calc_flow.symbolic.lower.segments import (
    _cstr,
    _cstr_seq,
    _expression_node,
    _quote_identifier,
)
from calc_flow.symbolic.lower.strategies import _project_document
from calc_flow.symbolic.nodes import Node
from calc_flow.symbolic.program import Program
from calc_flow.symbolic.types import Field

if TYPE_CHECKING:
    import pyarrow as pa


def _walk_nodes(node: Node) -> tuple[Node, ...]:
    found: dict[str, Node] = {}

    def visit(current: Node) -> None:
        if current.digest in found:
            return
        found[current.digest] = current
        for child in current.args:
            visit(child)

    visit(node)
    return tuple(found.values())


def _shared_tables(program: Program, analyzer: _Analyzer) -> frozenset[str]:
    uses: dict[str, set[str]] = {}
    for name, output in program.outputs:
        uses.setdefault(output.digest, set()).add(f"output:{name}")
        for node in _walk_nodes(output._node):
            if node.digest in analyzer._table_cache:
                for child in node.args:
                    uses.setdefault(child.digest, set()).add(node.digest)
    return frozenset(
        digest
        for digest, parents in uses.items()
        if len(parents) > 1 and digest in analyzer._table_cache
    )


def _pin_fragment_schemas(
    document: dict[str, Any], schemas: Mapping[str, pa.Schema]
) -> None:
    raw = document["graph"]
    external = {
        node["id"] for node in raw["nodes"] if node["operator"]["kind"] == "external"
    }
    untyped = {
        edge["source_node"] for edge in raw["edges"] if edge["target_node"] in external
    }
    for node in raw["nodes"]:
        if node["operator"]["kind"] == "expression" and node["id"] not in untyped:
            node["output_ports"] = [_table_port(_fields(schemas[node["id"]]), "output")]


class _SQLGraph:
    def __init__(
        self,
        program: Program,
        analyzer: _Analyzer,
        allowed_lateness_micros: int,
        late_policy: str,
    ) -> None:
        self.graph = _WindowGraph(program)
        self.analyzer = analyzer
        self.allowed_lateness_micros = allowed_lateness_micros
        self.late_policy = late_policy
        self.shared = _shared_tables(program, analyzer)
        self.schemas: dict[str, tuple[Field, ...]] = {}
        self.replacements: dict[str, Node] = {}
        self.materialized: dict[str, str] = {}
        self.virtual_sources: dict[str, str] = {}

    def table(self, node: Node) -> str:
        existing = self.materialized.get(node.digest)
        if existing is not None:
            return existing
        if node.op.name == "table_input":
            self.schemas[node.digest] = self.analyzer.table(node, "sql.input").schema
            return self.graph.source(node)
        output_id = self.graph.allocate(f"cf_sql_{node.digest[:24]}")
        if node.op.name == "sql":
            self._append_sql(node, output_id)
        else:
            output_id = self._append_expression(node, output_id)
        facts = self.analyzer.table(node, f"{self.graph.program.name}.sql")
        virtual = table_input(
            self.graph.allocate(f"cf_sql_input_{node.digest[:24]}"),
            schema=self.schemas[node.digest],
            entity_by=facts.entity_by,
            event_time=facts.event_time,
            sequence_by=facts.sequence_by,
        )
        self.replacements[node.digest] = virtual._node
        self.virtual_sources[virtual.digest] = output_id
        self.materialized[node.digest] = output_id
        return output_id

    def _append_sql(self, node: Node, output_id: str) -> None:
        aliases = _cstr_seq(node.attr("aliases"))
        parents = [self.table(child) for child in node.args]
        schemas = [self.schemas[child.digest] for child in node.args]
        schema = self.analyzer.table(node, "sql.output").schema
        self.schemas[node.digest] = schema
        self.graph.nodes.append(
            {
                "id": output_id,
                "input_ports": [
                    _table_port(fields, alias)
                    for alias, fields in zip(aliases, schemas, strict=True)
                ],
                "operator": {
                    "kind": "sql",
                    "query": _cstr(node.attr("query")),
                    "aliases": list(aliases),
                    "udfs": [],
                },
                "output_ports": [_table_port(schema, "output")],
            }
        )
        self.graph.edges.extend(
            {
                "source_node": parent,
                "source_port": "output",
                "target_node": output_id,
                "target_port": alias,
            }
            for alias, parent in zip(aliases, parents, strict=True)
        )

    def _append_expression(self, node: Node, output_id: str) -> str:
        from calc_flow.symbolic.lower.program import lower_program_document

        for boundary in reversed(_walk_nodes(node)):
            if boundary.digest == node.digest:
                continue
            if (
                boundary.op.name in {"sql", "attach_columns", "stream_join"}
                or boundary.digest in self.shared
            ):
                self.table(boundary)
        rewritten = _rewrite_nodes(node, self.replacements)
        fragment = Program(
            self.graph.program.name, outputs={output_id: TableExpr(rewritten)}
        )
        bindings = _BatchBindings()
        document = lower_program_document(
            fragment,
            self.analyzer._runtime,
            self.analyzer._mode,
            allowed_lateness_micros=self.allowed_lateness_micros,
            late_policy=self.late_policy,
            _bindings=bindings,
        )
        declared = {
            output_id: _arrow_schema(self.analyzer.table(node, "sql.input").schema)
        }
        schemas = infer_document_schemas(document, self.analyzer._runtime, declared)
        self.schemas[node.digest] = _fields(schemas[output_id])
        _pin_fragment_schemas(document, schemas)
        upstreams = {
            _cstr(value._node.attr("name")): self._source(value._node)
            for value in fragment.inputs
            if isinstance(value, TableExpr)
        }
        self._append_fragment(document, bindings, upstreams, output_id)
        if node.op.name == "attach_columns":
            return self._provider_schema_boundary(output_id, self.schemas[node.digest])
        return output_id

    def _provider_schema_boundary(self, parent: str, fields: tuple[Field, ...]) -> str:
        output = self.graph.allocate(f"{parent}__typed")
        self.graph.nodes.append(
            _expression_node(
                output,
                [_quote_identifier(field.name) for field in fields],
                None,
                None,
                fields,
            )
        )
        self.graph.edges.append(
            {
                "source_node": parent,
                "source_port": "output",
                "target_node": output,
                "target_port": "input",
            }
        )
        return output

    def _source(self, node: Node) -> str:
        existing = self.virtual_sources.get(node.digest)
        return self.graph.source(node) if existing is None else existing

    def _append_fragment(
        self,
        document: dict[str, Any],
        bindings: _BatchBindings,
        upstreams: Mapping[str, str],
        output_id: str,
    ) -> None:
        raw = document["graph"]
        aliases = {
            node["id"]: output_id
            if node["id"] == output_id
            else self.graph.allocate(f"{output_id}__{node['id']}")
            for node in raw["nodes"]
        }
        self.graph.nodes.extend(
            {**node, "id": aliases[node["id"]]} for node in raw["nodes"]
        )
        self.graph.edges.extend(
            {
                **edge,
                "source_node": aliases[edge["source_node"]],
                "target_node": aliases[edge["target_node"]],
            }
            for edge in raw["edges"]
        )
        self._wire_inputs(bindings, upstreams, aliases)
        for declaration in document.get("static_inputs", []):
            if declaration not in self.graph.static_inputs:
                self.graph.static_inputs.append(declaration)

    def _wire_inputs(
        self,
        bindings: _BatchBindings,
        upstreams: Mapping[str, str],
        aliases: Mapping[str, str],
    ) -> None:
        for logical, endpoints in bindings.inputs.items():
            if logical not in upstreams:
                self.graph.bindings.inputs.setdefault(logical, set()).update(
                    (aliases[node], port) for node, port in endpoints
                )
                continue
            self.graph.edges.extend(
                {
                    "source_node": upstreams[logical],
                    "source_port": "output",
                    "target_node": aliases[node],
                    "target_port": port,
                }
                for node, port in endpoints
            )

    def outputs(self) -> None:
        for name, value in self.graph.program.outputs:
            upstream = self.table(value._node)
            fields = self.schemas[value.digest]
            self.graph.nodes.append(
                _expression_node(
                    name,
                    [_quote_identifier(field.name) for field in fields],
                    None,
                    fields,
                )
            )
            self.graph.edges.append(
                {
                    "source_node": upstream,
                    "source_port": "output",
                    "target_node": name,
                    "target_port": "input",
                }
            )
            self.graph.bindings.outputs[name] = (name, "output")


def lower_sql_program(
    program: Program,
    analyzer: _Analyzer,
    allowed_lateness_micros: int,
    late_policy: str,
) -> dict[str, object] | None:
    if not any(
        node.op.name == "sql"
        for _, value in program.outputs
        for node in _walk_nodes(value._node)
    ):
        return None
    graph = _SQLGraph(program, analyzer, allowed_lateness_micros, late_policy)
    graph.outputs()
    bindings = getattr(analyzer, "_bindings", None)
    if bindings is not None:
        bindings.inputs.update(graph.graph.bindings.inputs)
        bindings.outputs.update(graph.graph.bindings.outputs)
    document = _project_document(
        program.name, analyzer._mode, graph.graph.nodes, graph.graph.edges
    )
    if graph.graph.static_inputs:
        document["static_inputs"] = graph.graph.static_inputs
    return document
