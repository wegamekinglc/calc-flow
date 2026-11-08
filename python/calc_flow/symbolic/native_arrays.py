"""Data-only translation of native ArrayExpr declarations to Rust query IR."""

from __future__ import annotations

import json
from typing import TYPE_CHECKING, cast

from calc_flow.errors import CalcFlowError
from calc_flow.pipeline import JSONValue, _canonical
from calc_flow.symbolic.nodes import (
    CBool,
    CDType,
    CFloat,
    CInt,
    CNull,
    CSeq,
    CStr,
    CValue,
    Node,
    build,
)

if TYPE_CHECKING:
    from calc_flow.symbolic.analyzer import ArrayFacts, _Analyzer, _RowOrigin

_BINARY = {
    "add": "add",
    "sub": "subtract",
    "mul": "multiply",
    "truediv": "divide",
    "eq": "equal",
    "ne": "not_equal",
    "lt": "less",
    "le": "less_equal",
    "gt": "greater",
    "ge": "greater_equal",
    "and": "logical_and",
    "or": "logical_or",
    "power": "pow",
}
_UNARY = {
    "neg": "neg",
    "not": "logical_not",
    "abs": "abs",
    "sqrt": "sqrt",
    "exp": "exp",
    "log": "log",
}


def value(raw: CValue | None) -> JSONValue:
    if raw is None or isinstance(raw, CNull):
        return None
    if isinstance(raw, CDType):
        return raw.name
    if isinstance(raw, CSeq):
        return [value(item) for item in raw.items]
    if isinstance(raw, (CBool, CFloat, CInt, CStr)):
        return raw.value
    raise TypeError("native array declarations require JSON-compatible attributes")


def is_native_array(node: Node) -> bool:
    if node.op.name in {"array_input", "parameter", "from_columns"}:
        return value(node.attr("backend")) == "native"
    return any(is_native_array(child) for child in node.args)


NATIVE_ATTACHMENT_SOURCE_MESSAGE = (
    "native attachment requires from_columns and attachment to use "
    "the same TableExpr node"
)


def native_attachment_source_matches(array: Node, target: Node) -> bool:
    """Prove exact table-node identity without comparing schemas or row counts."""
    visited: set[str] = set()

    def visit(node: Node) -> bool:
        if node.digest in visited:
            return True
        visited.add(node.digest)
        if node.op.name == "from_columns":
            source = node.args[0]
            return (
                source.digest == target.digest
                and source.node_bytes == target.node_bytes
            )
        return all(visit(child) for child in node.args)

    return visit(array)


def node_id(node: Node) -> str:
    return f"n_{node.digest[:24]}"


def input_name(node: Node) -> str:
    return f"a_{node.digest[:24]}"


def _input(node: Node, analyzer: _Analyzer | None) -> dict[str, object]:
    if value(node.attr("backend")) != "native":
        raise TypeError("native ArrayExpr rejects implicit cross-backend conversion")
    if node.op.name == "from_columns":
        if analyzer is None:
            raise TypeError("native table input requires its declaration analyzer")
        facts = analyzer._from_columns(node, "native.from_columns")
        return {
            "name": input_name(node),
            "dtype": facts.dtype,
            "shape": [None, len(cast(list[str], value(node.attr("columns"))))],
        }
    return {
        "name": input_name(node),
        "dtype": value(node.attr("dtype")),
        "shape": value(node.attr("shape")),
    }


def _operation(node: Node) -> dict[str, object]:
    operation = node.op.name
    arguments = [node_id(child) for child in node.args]
    if operation in _BINARY:
        return {
            "op": "binary",
            "function": _BINARY[operation],
            "left": arguments[0],
            "right": arguments[1],
        }
    if operation in _UNARY:
        return {"op": "unary", "function": _UNARY[operation], "input": arguments[0]}
    if operation == "matmul":
        return {"op": "matmul", "left": arguments[0], "right": arguments[1]}
    if operation == "array_reduce":
        return _reduction_operation(node, arguments[0])
    if operation == "array_reshape":
        return {
            "op": "reshape",
            "input": arguments[0],
            "shape": value(node.attr("shape")),
        }
    if operation == "array_transpose":
        return {"op": "transpose", "input": arguments[0]}
    raise TypeError(f"unsupported native array primitive {operation!r}")


def _reduction_operation(node: Node, input_id: str) -> dict[str, object]:
    return {
        "op": "reduce",
        "input": input_id,
        **{
            name: value(node.attr(name))
            for name in ("function", "axis", "keepdims", "skip_nulls")
        },
    }


def build_native_program(
    node: Node,
    *,
    analyzer: _Analyzer | None = None,
) -> tuple[dict[str, object], dict[str, Node]]:
    """Preserve descriptors and operations; Rust supplies every inferred result."""
    inputs: list[dict[str, object]] = []
    bindings: dict[str, Node] = {}
    nodes: list[dict[str, object]] = []
    visited: set[str] = set()

    def visit(current: Node) -> None:
        if current.digest in visited:
            return
        visited.add(current.digest)
        if current.op.name in {"array_input", "parameter", "from_columns"}:
            descriptor = _input(current, analyzer)
            inputs.append(descriptor)
            bindings[input_name(current)] = current
            operation = {"op": "input", "name": descriptor["name"]}
        elif current.op.name == "literal":
            literal = value(current.attr("value"))
            operation = {
                "op": "literal",
                "dtype": "bool" if type(literal) is bool else "float64",
                "value": literal,
            }
        else:
            for child in current.args:
                visit(child)
            operation = _operation(current)
        nodes.append({"id": node_id(current), **operation})

    visit(node)
    return {
        "ir_version": 2,
        "inputs": inputs,
        "nodes": nodes,
        "outputs": [{"name": "output", "node": node_id(node)}],
    }, bindings


def _inference_node(node: Node, analyzer: _Analyzer, path: str) -> Node:
    if node.op.name in {"array_input", "parameter", "from_columns"}:
        return node
    children = tuple(_inference_operand(child, analyzer, path) for child in node.args)
    return build(
        node.op.name, children, dict(node.attrs.entries), version=node.op.version
    )


def _inference_operand(node: Node, analyzer: _Analyzer, path: str) -> Node:
    if node.op.name == "literal":
        return node
    facts = _native_operand_facts(node, analyzer, path)
    dimensions = tuple(
        CNull() if dim is None else CInt(cast(int, dim)) for dim in facts.shape
    )
    return build(
        "array_input",
        (),
        {
            "name": CStr(input_name(node)),
            "backend": CStr("native"),
            "dtype": CDType(cast(str, facts.dtype)),
            "shape": CSeq(dimensions),
        },
    )


def _native_operand_facts(node: Node, analyzer: _Analyzer, path: str) -> ArrayFacts:
    if not is_native_array(node):
        raise TypeError("native ArrayExpr requires resolved native operands")
    facts = analyzer.array(node, path)
    if facts.backend != "native" or facts.dtype is None:
        raise TypeError("native ArrayExpr requires resolved native operands")
    return facts


def _compatible_origin(
    node: Node, operands: tuple[ArrayFacts | None, ...]
) -> _RowOrigin | None:
    del node
    present = tuple(operand for operand in operands if operand is not None)
    if not present:
        return None
    origin = present[0].lineage
    return origin if all(operand.lineage == origin for operand in present) else None


def _matmul_origin(
    node: Node, operands: tuple[ArrayFacts | None, ...]
) -> _RowOrigin | None:
    left = operands[0]
    if left is None or node.args[1].op.name != "parameter":
        return None
    return left.lineage


def _reduction_origin(
    node: Node, operands: tuple[ArrayFacts | None, ...]
) -> _RowOrigin | None:
    source = operands[0]
    if source is None:
        return None
    return (
        source.lineage
        if value(node.attr("axis")) == 1 and len(source.shape) == 2
        else None
    )


def _transpose_origin(
    node: Node, operands: tuple[ArrayFacts | None, ...]
) -> _RowOrigin | None:
    del node
    source = operands[0]
    return source.lineage if source is not None and len(source.shape) < 2 else None


def _reshape_origin(
    node: Node, operands: tuple[ArrayFacts | None, ...]
) -> _RowOrigin | None:
    source = operands[0]
    if source is None:
        return None
    return (
        source.lineage
        if source.shape == tuple(cast(list[int], value(node.attr("shape"))))
        else None
    )


_ORIGINS = {
    "matmul": _matmul_origin,
    "array_reduce": _reduction_origin,
    "array_transpose": _transpose_origin,
    "array_reshape": _reshape_origin,
}


def _native_context(
    node: Node, analyzer: _Analyzer, path: str
) -> tuple[_RowOrigin | None, frozenset[str]]:
    if node.op.name == "from_columns":
        table = analyzer.table(node.args[0], path)
        return table.lineage, table.state
    if node.op.name == "parameter":
        parameter = analyzer._array_parameter(node, path)
        return None, parameter.state
    if node.op.name == "array_input":
        if node.digest not in analyzer._declared:
            analyzer.issue(
                f"inputs.{value(node.attr('name'))}",
                "unresolved_type",
                "array_input is not declared in Program inputs",
            )
        return None, frozenset()
    return _operand_context(node, analyzer, path)


def _operand_context(
    node: Node, analyzer: _Analyzer, path: str
) -> tuple[_RowOrigin | None, frozenset[str]]:
    operands = tuple(
        None
        if child.op.name == "literal"
        else _native_operand_facts(child, analyzer, path)
        for child in node.args
    )
    state = frozenset().union(
        *(operand.state for operand in operands if operand is not None)
    )
    return _ORIGINS.get(node.op.name, _compatible_origin)(node, operands), state


def analyze_native_array(node: Node, analyzer: _Analyzer, path: str) -> ArrayFacts:
    from calc_flow.symbolic.analyzer import ArrayFacts

    required = "table_array" if node.op.name == "from_columns" else "array_expr"
    capabilities = (
        () if analyzer._capabilities is None else analyzer._capabilities.providers
    )
    if not any(
        (item.provider, item.name, item.version) == ("native", required, "1")
        for item in capabilities
    ):
        analyzer.issue(
            path,
            "capability_mismatch",
            "register_native must install the native array factory before analysis",
        )
        return ArrayFacts("native", None, (), None, frozenset())
    try:
        origin, state = _native_context(node, analyzer, path)
        query, _ = build_native_program(
            _inference_node(node, analyzer, path), analyzer=analyzer
        )
        descriptor = json.loads(
            analyzer._runtime._infer_native_array_program(_canonical(query))
        )["outputs"][0]
    except (CalcFlowError, TypeError, ValueError) as error:
        analyzer.issue(path, "unsupported_type", str(error))
        return ArrayFacts("native", None, (), None, frozenset())
    return ArrayFacts(
        "native", descriptor["dtype"], tuple(descriptor["shape"]), origin, state
    )
