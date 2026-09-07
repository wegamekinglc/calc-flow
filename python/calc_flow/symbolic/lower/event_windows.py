"""Lower event windows as shared native state boundaries in a table DAG.

Only declarations are rewritten here. The existing row/relational lowerers
compile fragments on either side of a native window; no expression optimizer
can see through that boundary and no Python object owns runtime window state.
"""

from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
from typing import TYPE_CHECKING

from calc_flow.symbolic import errors
from calc_flow.symbolic.analyzer import _Analyzer, _schema_fields
from calc_flow.symbolic.expr import TableExpr, table_input
from calc_flow.symbolic.lower.segments import (
    _cint,
    _cstr,
    _cstr_seq,
    _expression_node,
    _field_json,
    _quote_identifier,
)
from calc_flow.symbolic.lower.strategies import (
    _downstream_input_endpoints,
    _project_document,
    _project_graph_lists,
)
from calc_flow.symbolic.nodes import CMap, CSeq, CStr, Node, build
from calc_flow.symbolic.types import Field

if TYPE_CHECKING:
    from calc_flow.pipeline import Runtime
    from calc_flow.symbolic.program import Program


_WINDOWS = frozenset({"window_tumbling", "window_hopping"})


def _event_window_nodes(program: Program, /) -> tuple[Node, ...]:
    visited: dict[str, Node] = {}

    def visit(node: Node) -> None:
        if node.digest in visited:
            return
        visited[node.digest] = node
        for child in node.args:
            visit(child)

    for _, value in program.outputs:
        visit(value._node)
    return tuple(
        node
        for _, node in sorted(visited.items())
        if node.op.name in _WINDOWS and node.op.version == 2
    )


def _rewrite_nodes(root: Node, replacements: dict[str, Node], /) -> Node:
    rewritten = dict(replacements)

    def visit(node: Node) -> Node:
        if node.digest not in rewritten:
            rewritten[node.digest] = build(
                node.op.name,
                tuple(visit(child) for child in node.args),
                dict(node.attrs.entries),
                version=node.op.version,
            )
        return rewritten[node.digest]

    return visit(root)


def _reachable_inputs(root: Node, /) -> frozenset[str]:
    seen: set[str] = set()
    inputs: set[str] = set()

    def visit(node: Node) -> None:
        if node.digest in seen:
            return
        seen.add(node.digest)
        if node.op.name in ("table_input", "parameter"):
            inputs.add(node.digest)
        for child in node.args:
            visit(child)

    visit(root)
    return frozenset(inputs)


@dataclass(frozen=True, slots=True)
class _WindowPlan:
    node: Node
    node_id: str
    input_schema: tuple[Field, ...]
    output_schema: tuple[Field, ...]


def _window_node(plan: _WindowPlan, /) -> dict[str, object]:
    node = plan.node
    geometry: dict[str, object] = {
        "kind": "tumbling" if node.op.name == "window_tumbling" else "hopping",
        "size_micros": _cint(node.attr("size_micros")),
    }
    if node.op.name == "window_hopping":
        geometry["slide_micros"] = _cint(node.attr("slide_micros"))
    declared = node.attr("aggregates")
    if not isinstance(declared, CSeq) or any(
        not isinstance(item, CMap) for item in declared.items
    ):
        raise RuntimeError("analyzed event window has invalid aggregates")
    aggregates = [
        {key: _cstr(item.get(key)) for key in ("function", "column", "output")}
        for item in declared.items
    ]
    return {
        "id": plan.node_id,
        "input_ports": [_table_port(plan.input_schema, "input")],
        "operator": {
            "kind": "window",
            "spec": {
                "event_time_column": _cstr(node.attr("event_time")),
                "group_by": list(_cstr_seq(node.attr("group_by"))),
                "geometry": geometry,
                "aggregates": aggregates,
            },
        },
        "output_ports": [_table_port(plan.output_schema, "output")],
    }


def _table_port(schema: tuple[Field, ...], name: str, /) -> dict[str, object]:
    return {
        "name": name,
        "kind": "table",
        "required": True,
        "schema": [_field_json(field) for field in schema],
    }


class _WindowGraph:
    """Own graph assembly and its collision-checked physical node identities."""

    def __init__(self, program: Program) -> None:
        self.program = program
        self.nodes: list[dict[str, object]] = []
        self.edges: list[dict[str, object]] = []
        self.static_inputs: list[object] = []
        self._reserved = {name for name, _ in program.outputs}
        self._sources: dict[str, str] = {}

    def allocate(self, preferred: str) -> str:
        if (
            len(preferred) > 48
            or not preferred[0].isascii()
            or not preferred[0].isalpha()
            or any(
                not char.isascii() or not (char.isalnum() or char in "_-")
                for char in preferred
            )
        ):
            preferred = f"cf_{sha256(preferred.encode()).hexdigest()[:40]}"
        name = preferred
        ordinal = 0
        while name in self._reserved:
            ordinal += 1
            name = f"{preferred}_{ordinal}"
        self._reserved.add(name)
        return name

    def source(self, node: Node) -> str:
        if node.digest not in self._sources:
            name = self.allocate(_cstr(node.attr("name")))
            schema = _schema_fields(node.attr("schema"))
            self.nodes.append(
                _expression_node(
                    name,
                    [_quote_identifier(field.name) for field in schema],
                    None,
                    schema,
                    schema,
                )
            )
            self._sources[node.digest] = name
        return self._sources[node.digest]

    def append_fragment(
        self,
        document: dict[str, object],
        bindings: dict[str, str],
        output_ids: dict[str, str],
        scope: str,
    ) -> None:
        """Copy a fragment and wire each of its exposed table inputs once."""
        raw_nodes, raw_edges = _project_graph_lists(document)
        by_id = {node["id"]: node for node in raw_nodes}
        aliases = {
            name: output_ids[name]
            if name in output_ids
            else self.allocate(f"{scope}__{name}")
            for name in by_id
        }
        for node in raw_nodes:
            self.nodes.append({**node, "id": aliases[node["id"]]})
        self.edges.extend(
            {
                **edge,
                "source_node": aliases[edge["source_node"]],
                "target_node": aliases[edge["target_node"]],
            }
            for edge in raw_edges
        )
        for node_id, port in _downstream_input_endpoints(raw_nodes, raw_edges):
            ports = by_id[node_id].get("input_ports", [])
            if any(item["name"] == port and item["kind"] == "array" for item in ports):
                continue
            if len(bindings) == 1:
                upstream = next(iter(bindings.values()))
            elif node_id in bindings:
                upstream = bindings[node_id]
            else:
                raise RuntimeError(
                    f"unresolved event-window fragment source {node_id!r}"
                )
            self.edges.append(
                {
                    "source_node": upstream,
                    "source_port": "output",
                    "target_node": aliases[node_id],
                    "target_port": port,
                }
            )
        for declaration in document.get("static_inputs", []):
            if declaration not in self.static_inputs:
                self.static_inputs.append(declaration)


def _temporary_input(graph: _WindowGraph, value: TableExpr, /) -> TableExpr:
    # Generated names cannot collide with source/state nodes or a user's output
    # and its derived expression-stage names inside the delegated lowerer.
    preferred = f"cf_window_source_{value.digest}"
    name = graph.allocate(preferred)
    while any(name.startswith(f"{output}__cf_") for output, _ in graph.program.outputs):
        name = graph.allocate(f"_{name}")
    return TableExpr(
        build(
            "table_input", (), {**dict(value._node.attrs.entries), "name": CStr(name)}
        )
    )


def _append_window_inputs(
    graph: _WindowGraph,
    plans: tuple[_WindowPlan, ...],
    runtime: Runtime,
    /,
) -> None:
    from calc_flow.symbolic.lower.program import lower_program_document
    from calc_flow.symbolic.program import Program

    for plan in plans:
        source_digests = _reachable_inputs(plan.node.args[0])
        inputs = tuple(
            value for value in graph.program.inputs if value.digest in source_digests
        )
        if len(inputs) != 1 or inputs[0]._node.op.name != "table_input":
            raise RuntimeError("analyzed event window must have one table source")
        source_id = graph.source(inputs[0]._node)
        if plan.node.args[0].op.name == "table_input":
            upstream = source_id
        else:
            upstream = graph.allocate(f"{plan.node_id}__input")
            fragment = Program(
                graph.program.name,
                inputs=inputs,
                outputs=[(upstream, TableExpr(plan.node.args[0]))],
            )
            document = lower_program_document(fragment, runtime, "stream")
            # Pin the complete pre-window schema, including derived columns.
            for node in document["graph"]["nodes"]:
                if node["id"] == upstream:
                    node["output_ports"] = [_table_port(plan.input_schema, "output")]
            graph.append_fragment(
                document, {"input": source_id}, {upstream: upstream}, upstream
            )
        graph.nodes.append(_window_node(plan))
        graph.edges.append(
            {
                "source_node": upstream,
                "source_port": "output",
                "target_node": plan.node_id,
                "target_port": "input",
            }
        )


def _append_window_outputs(
    graph: _WindowGraph,
    plans: tuple[_WindowPlan, ...],
    runtime: Runtime,
    allowed_lateness_micros: int,
    late_policy: str,
    /,
) -> None:
    from calc_flow.symbolic.lower.program import lower_program_document
    from calc_flow.symbolic.program import Program

    replacements: dict[str, Node] = {}
    inputs: list[object] = []
    bindings: dict[str, str] = {}
    for plan in plans:
        virtual = table_input(
            graph.allocate(f"cf_window_output_{plan.node.digest}"),
            schema=plan.output_schema,
        )
        # The same safe naming policy applies to virtual and real inputs.
        virtual = _temporary_input(graph, virtual)
        replacements[plan.node.digest] = virtual._node
        inputs.append(virtual)
        bindings[_cstr(virtual._node.attr("name"))] = plan.node_id
    outside_windows = frozenset().union(
        *(
            _reachable_inputs(_rewrite_nodes(value._node, replacements))
            for _, value in graph.program.outputs
        )
    )
    for value in graph.program.inputs:
        if value.digest not in outside_windows:
            continue
        if isinstance(value, TableExpr):
            replacement = _temporary_input(graph, value)
            replacements[value.digest] = replacement._node
            inputs.append(replacement)
            bindings[_cstr(replacement._node.attr("name"))] = graph.source(value._node)
        else:
            inputs.append(value)
    outputs = tuple(
        (name, type(value)(_rewrite_nodes(value._node, replacements)))
        for name, value in graph.program.outputs
    )
    # Existing matrix lowering accepts one attached matrix result at a time;
    # preserve its static bindings while sharing the original table source.
    regular = tuple(
        item for item in outputs if item[1]._node.op.name != "attach_columns"
    )
    groups = ([regular] if regular else []) + [
        (item,) for item in outputs if item[1]._node.op.name == "attach_columns"
    ]
    for ordinal, group in enumerate(groups):
        reachable = frozenset().union(
            *(_reachable_inputs(value._node) for _, value in group)
        )
        selected_inputs = tuple(value for value in inputs if value.digest in reachable)
        selected_bindings = {
            _cstr(value._node.attr("name")): bindings[_cstr(value._node.attr("name"))]
            for value in selected_inputs
            if isinstance(value, TableExpr)
        }
        fragment = Program(graph.program.name, inputs=selected_inputs, outputs=group)
        document = lower_program_document(
            fragment,
            runtime,
            "stream",
            allowed_lateness_micros=allowed_lateness_micros,
            late_policy=late_policy,
        )
        graph.append_fragment(
            document,
            selected_bindings,
            {name: name for name, _ in group},
            f"cf_window_branch_{ordinal}",
        )


def _lower_event_window_program(
    program: Program,
    analyzer: _Analyzer,
    runtime: Runtime,
    allowed_lateness_micros: int,
    late_policy: str,
    /,
) -> dict[str, object] | None:
    windows = _event_window_nodes(program)
    if not windows:
        return None
    graph = _WindowGraph(program)
    plans = tuple(
        _WindowPlan(
            node,
            graph.allocate(f"cf_window_{node.digest[:24]}"),
            analyzer._window_input_schemas[node.digest],
            analyzer.table(node, f"{program.name}.window").schema,
        )
        for node in windows
    )
    _append_window_inputs(graph, plans, runtime)
    _append_window_outputs(graph, plans, runtime, allowed_lateness_micros, late_policy)
    document = _project_document(program.name, "stream", graph.nodes, graph.edges)
    if graph.static_inputs:
        document["static_inputs"] = graph.static_inputs
    return document


def _check_window_lateness_options(
    program: Program,
    has_temporal_consumer: bool,
    allowed_lateness_micros: int,
    late_policy: str,
    /,
) -> None:
    if has_temporal_consumer or not _event_window_nodes(program):
        return
    for name, value, default in (
        ("allowed_lateness_micros", allowed_lateness_micros, 0),
        ("late_policy", late_policy, "error"),
    ):
        if value != default:
            errors.raise_compile(
                f"{program.name}.compile_stream.{name}",
                errors.CAPABILITY_MISMATCH,
                "event windows use native late-assignment dropping; this option"
                " requires an independent rolling or cross-section consumer",
            )
