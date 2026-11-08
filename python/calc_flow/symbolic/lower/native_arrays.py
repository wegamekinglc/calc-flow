"""Native array segments joined to the existing SQL/table graph lowerer."""

from __future__ import annotations

from typing import TYPE_CHECKING, cast

from calc_flow.array import ArrayExecutionOptions
from calc_flow.symbolic import errors
from calc_flow.symbolic.expr import ArrayExpr
from calc_flow.symbolic.lower.segments import (
    _expression_node,
    _quote_identifier,
    _table_port,
)
from calc_flow.symbolic.lower.sql import _SQLGraph, _walk_nodes
from calc_flow.symbolic.lower.strategies import (
    _project_document,
    _static_array_declaration,
)
from calc_flow.symbolic.native_arrays import (
    NATIVE_ATTACHMENT_SOURCE_MESSAGE,
    build_native_program,
    input_name,
    is_native_array,
    native_attachment_source_matches,
    node_id,
    value,
)
from calc_flow.symbolic.nodes import CDType, CInt, CNull, CSeq, CStr, Node, build

if TYPE_CHECKING:
    from calc_flow.symbolic.analyzer import ArrayFacts, _Analyzer
    from calc_flow.symbolic.program import Program


def _array_port(name: str) -> dict[str, object]:
    return {"name": name, "kind": "array", "required": True, "schema": []}


def _edge(source: str, target: str, port: str) -> dict[str, object]:
    return {
        "source_node": source,
        "source_port": "output",
        "target_node": target,
        "target_port": port,
    }


def _has_native(program: Program) -> bool:
    return any(
        node.op.name in {"array_input", "parameter", "from_columns"}
        and value(node.attr("backend")) == "native"
        for _, output in program.outputs
        for node in _walk_nodes(output._node)
    )


def _virtual_array(node: Node, facts: ArrayFacts) -> Node:
    return build(
        "array_input",
        (),
        {
            "name": CStr(input_name(node)),
            "backend": CStr("native"),
            "dtype": CDType(cast(str, facts.dtype)),
            "shape": CSeq(
                tuple(
                    CNull() if dim is None else CInt(cast(int, dim))
                    for dim in facts.shape
                )
            ),
        },
    )


def _array_nodes(node: Node) -> tuple[Node, ...]:
    seen: set[str] = set()
    nodes: list[Node] = []

    def visit(current: Node) -> None:
        if current.digest in seen:
            return
        seen.add(current.digest)
        nodes.append(current)
        if current.op.name not in {
            "from_columns",
            "array_input",
            "parameter",
            "literal",
        }:
            for child in current.args:
                visit(child)

    visit(node)
    return tuple(nodes)


def _bridge_leaves(node: Node) -> tuple[Node, ...]:
    return tuple(
        item
        for item in _array_nodes(node)
        if item.op.name in {"from_columns", "array_input", "parameter"}
    )


def _bridge_admitted(node: Node, analyzer: _Analyzer) -> bool:
    leaves = _bridge_leaves(node)
    if not _bridge_inputs_admitted(leaves):
        return False
    operations = _array_nodes(node)
    return (
        analyzer.array(node, "native.bridge").lineage is not None
        and sum(item.op.name == "matmul" for item in operations) <= 1
        and _static_rhs_only(operations)
    )


def _bridge_inputs_admitted(leaves: tuple[Node, ...]) -> bool:
    return sum(item.op.name == "from_columns" for item in leaves) == 1 and not any(
        item.op.name == "array_input" for item in leaves
    )


def _static_rhs_only(operations: tuple[Node, ...]) -> bool:
    return all(
        child.op.name != "parameter" or (parent.op.name == "matmul" and index == 1)
        for parent in operations
        for index, child in enumerate(parent.args)
    )


def _attachment_rejection(node: Node) -> str:
    count = sum(item.op.name == "matmul" for item in _array_nodes(node))
    if count > 1:
        return (
            "native attachment supports at most one matmul between table stages; "
            f"found {count}"
        )
    return "native attachment requires one row-preserving table/static-RHS segment"


def _require_attachment_source(node: Node, attached: Node | None) -> None:
    if attached is not None and not native_attachment_source_matches(
        node, attached.args[0]
    ):
        errors.raise_compile(
            "native.attach_columns.array.source",
            errors.SCHEMA_MISMATCH,
            NATIVE_ATTACHMENT_SOURCE_MESSAGE,
        )


class _NativeGraph(_SQLGraph):
    def __init__(
        self,
        program: Program,
        analyzer: _Analyzer,
        lateness: int,
        policy: str,
        options: ArrayExecutionOptions,
    ) -> None:
        super().__init__(program, analyzer, lateness, policy)
        self.array_options = options
        self.array_nodes: dict[str, str] = {}
        self.static_nodes: dict[str, Node] = {}

    def _execution(self) -> dict[str, int]:
        return {
            "max_elements": self.array_options.max_elements,
            "max_allocation_bytes": self.array_options.max_allocation_bytes,
        }

    def table(self, node: Node) -> str:
        existing = self.materialized.get(node.digest)
        if existing is not None:
            return existing
        if node.op.name != "attach_columns" or not is_native_array(node.args[1]):
            return super().table(node)
        if not _bridge_admitted(node.args[1], self.analyzer):
            errors.raise_compile(
                "native.attach_columns",
                errors.UNSUPPORTED_TYPE,
                _attachment_rejection(node.args[1]),
            )
        output = self._bridge(node.args[1], node)
        self.schemas[node.digest] = self.analyzer.table(
            node, "native.attach_columns"
        ).schema
        self._remember(node, output)
        return output

    def array(self, node: Node) -> str:
        existing = self.array_nodes.get(node.digest)
        if existing is not None:
            return existing
        if _bridge_admitted(node, self.analyzer):
            output = self._bridge(node, None)
        else:
            output = self._standalone(node)
        self.array_nodes[node.digest] = output
        return output

    def _native_node(
        self,
        output: str,
        name: str,
        options: dict[str, object],
        inputs: list[dict[str, object]],
        output_port: dict[str, object],
    ) -> None:
        self.graph.nodes.append(
            {
                "id": output,
                "input_ports": inputs,
                "operator": {
                    "kind": "external",
                    "provider": "native",
                    "name": name,
                    "version": "1",
                    "options": options,
                },
                "output_ports": [output_port],
            }
        )

    def _bind(self, output: str, port: str, leaf: Node) -> None:
        logical = str(value(leaf.attr("name")))
        self.graph.bindings.inputs.setdefault(logical, set()).add((output, port))
        if leaf.op.name == "parameter":
            self.static_nodes[logical] = leaf

    def _bridge(self, node: Node, attached: Node | None) -> str:
        _require_attachment_source(node, attached)
        query, leaves = build_native_program(node, analyzer=self.analyzer)
        output = self.graph.allocate(f"cf_native_{(attached or node).digest[:24]}")
        declarations: list[dict[str, object]] = []
        ports: list[dict[str, object]] = []
        table_port = ""
        for port, leaf in leaves.items():
            if leaf.op.name == "from_columns":
                source = self.table(leaf.args[0])
                schema = self.analyzer.table(leaf.args[0], "native.input").schema
                ports.append(_table_port(schema, port))
                declarations.append(
                    {
                        "kind": "table",
                        "name": port,
                        "columns": value(leaf.attr("columns")),
                    }
                )
                self.graph.edges.append(_edge(source, output, port))
                table_port = port
            else:
                descriptor = next(
                    item for item in query["inputs"] if item["name"] == port
                )
                declarations.append(
                    {"kind": "array", "binding": "static", **descriptor}
                )
                ports.append(_array_port(port))
                self._bind(output, port, leaf)
        assertion, output_port = self._bridge_output(node, attached, table_port)
        options = {
            **query,
            "inputs": declarations,
            "outputs": [assertion],
            "execution": self._execution(),
        }
        self._native_node(output, "table_array", options, ports, output_port)
        return output

    def _bridge_output(
        self, node: Node, attached: Node | None, table_port: str
    ) -> tuple[dict[str, object], dict[str, object]]:
        base = {"name": "output", "node": node_id(node)}
        if attached is not None:
            schema = self.analyzer.table(attached, "native.output").schema
            return {
                **base,
                "kind": "table",
                "attach_to": table_port,
                "names": value(attached.attr("names")),
            }, _table_port(schema, "output")
        facts = self.analyzer.array(node, "native.output")
        return {
            **base,
            "kind": "array",
            "dtype": facts.dtype,
            "shape": list(facts.shape),
        }, _array_port("output")

    def _standalone(self, node: Node) -> str:
        upstreams: dict[str, str] = {}
        if node.op.name in {"array_input", "parameter"}:
            rewritten = node
        else:
            rewritten = self._standalone_root(node, upstreams)
        query, leaves = build_native_program(rewritten, analyzer=self.analyzer)
        facts = self.analyzer.array(node, "native.output")
        query["outputs"] = [
            {
                "name": "output",
                "node": node_id(rewritten),
                "dtype": facts.dtype,
                "shape": list(facts.shape),
            }
        ]
        query["execution"] = self._execution()
        output = self.graph.allocate(f"cf_native_{node.digest[:24]}")
        self._native_node(
            output,
            "array_expr",
            query,
            [_array_port(port) for port in leaves],
            _array_port("output"),
        )
        for port, leaf in leaves.items():
            upstream = upstreams.get(str(value(leaf.attr("name"))))
            if upstream is None:
                self._bind(output, port, leaf)
            else:
                self.graph.edges.append(_edge(upstream, output, port))
        return output

    def _standalone_root(self, node: Node, upstreams: dict[str, str]) -> Node:
        children = []
        for child in node.args:
            if child.op.name in {"literal", "array_input", "parameter"}:
                children.append(child)
                continue
            upstreams[input_name(child)] = self.array(child)
            children.append(
                _virtual_array(child, self.analyzer.array(child, "native.input"))
            )
        return build(
            node.op.name,
            tuple(children),
            dict(node.attrs.entries),
            version=node.op.version,
        )

    def outputs(self) -> None:
        for logical, expression in self.graph.program.outputs:
            if isinstance(expression, ArrayExpr):
                self._array_output(logical, expression._node)
            else:
                upstream = self.table(expression._node)
                schema = self.schemas[expression.digest]
                self.graph.nodes.append(
                    _expression_node(
                        logical,
                        [_quote_identifier(field.name) for field in schema],
                        None,
                        schema,
                    )
                )
                self.graph.edges.append(_edge(upstream, logical, "input"))
                self.graph.bindings.outputs[logical] = (logical, "output")

    def _array_output(self, logical: str, node: Node) -> None:
        upstream = self.array(node)
        facts = self.analyzer.array(node, "native.output")
        output = self.graph.allocate(f"cf_array_output_{logical}")
        descriptor = {"name": "input", "dtype": facts.dtype, "shape": list(facts.shape)}
        options = {
            "ir_version": 2,
            "inputs": [descriptor],
            "nodes": [{"id": "x", "op": "input", "name": "input"}],
            "outputs": [
                {
                    "name": "output",
                    "node": "x",
                    "dtype": facts.dtype,
                    "shape": list(facts.shape),
                }
            ],
            "execution": self._execution(),
        }
        self._native_node(
            output, "array_expr", options, [_array_port("input")], _array_port("output")
        )
        self.graph.edges.append(_edge(upstream, output, "input"))
        self.graph.bindings.outputs[logical] = (output, "output")

    def document(self, mode: str) -> dict[str, object]:
        self._check_explicit_inputs()
        project = _project_document(
            self.graph.program.name, mode, self.graph.nodes, self.graph.edges
        )
        specifications = self._static_specifications()
        if specifications:
            project["static_inputs"] = specifications
            static_names = {
                specification["name"]
                for specification in specifications
                if mode != "batch"
                or (
                    specification.get("kind") == "array"
                    and specification.get("backend") == "native"
                )
            }
            project["data_sources"] = [
                source
                for source in project["data_sources"]
                if source["input"] not in static_names
            ]
        return project

    def _check_explicit_inputs(self) -> None:
        from calc_flow.symbolic.program import _node_name

        for declaration in self.graph.program.inputs:
            name = _node_name(declaration._node)
            if name not in self.graph.bindings.inputs:
                root = (
                    "static_inputs"
                    if declaration._node.op.name == "parameter"
                    else "inputs"
                )
                errors.raise_compile(
                    f"{root}.{name}",
                    errors.INVALID_LITERAL,
                    "unconsumed explicit input; remove it from Program.inputs",
                )

    def _static_specifications(self) -> list[dict[str, object]]:
        inputs, _ = self.graph.bindings.names()
        native = [
            {**_static_array_declaration(node), "name": physical}
            for logical, node in self.static_nodes.items()
            for physical in inputs[logical]
        ]
        return [dict(declaration) for declaration in self.graph.static_inputs] + native


def lower_native_program(
    program: Program,
    analyzer: _Analyzer,
    mode: str,
    lateness_policy: tuple[int, str],
    array_options: ArrayExecutionOptions | None,
) -> dict[str, object] | None:
    if not _has_native(program):
        return None
    arrays = tuple(isinstance(output, ArrayExpr) for _, output in program.outputs)
    if any(arrays) and not all(arrays):
        errors.raise_compile(
            program.name,
            errors.UNSUPPORTED_TYPE,
            "mixed table/array Program outputs are unsupported; collect one ArrayExpr",
        )
    graph = _NativeGraph(
        program,
        analyzer,
        *lateness_policy,
        ArrayExecutionOptions() if array_options is None else array_options,
    )
    graph.outputs()
    document = graph.document(mode)
    bindings = analyzer._bindings
    if bindings is not None:
        bindings.inputs.update(graph.graph.bindings.inputs)
        bindings.outputs.update(graph.graph.bindings.outputs)
    return document
