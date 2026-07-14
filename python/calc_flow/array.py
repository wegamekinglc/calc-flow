from __future__ import annotations

import ast
import math
import operator
from collections.abc import Mapping
from dataclasses import dataclass
from typing import TYPE_CHECKING

from calc_flow import _native

if TYPE_CHECKING:
    from calc_flow.pipeline import Runtime

_MAX_AST_NODES = 128
_MAX_AST_DEPTH = 24
_MAX_RESHAPE_RANK = 16
_MAX_RESHAPE_DIMENSION = 1_000_000
_MAX_RESHAPE_ELEMENTS = 10_000_000

_ALLOWED_BINARY = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.MatMult: operator.matmul,
    ast.Pow: operator.pow,
}
_ALLOWED_UNARY = {ast.UAdd: operator.pos, ast.USub: operator.neg}
_UNARY_FUNCTIONS = {"sum", "mean", "max", "min", "transpose"}
_ALLOWED_FUNCTIONS = _UNARY_FUNCTIONS | {"reshape"}


def _array_error(message: str) -> ValueError:
    return ValueError(f"invalid array expression: {message}")


def _parse_expression(expression: object) -> ast.Expression:
    if not isinstance(expression, str) or not expression.strip():
        raise _array_error("expression must be a non-empty string")
    try:
        parsed = ast.parse(expression, mode="eval")
    except (SyntaxError, ValueError) as error:
        raise _array_error("syntax is invalid") from error
    if not isinstance(parsed, ast.Expression):
        raise _array_error("syntax is invalid")
    nodes = list(ast.walk(parsed))
    if len(nodes) > _MAX_AST_NODES:
        raise _array_error(f"node limit is {_MAX_AST_NODES}")
    if _ast_depth(parsed) > _MAX_AST_DEPTH:
        raise _array_error(f"depth limit is {_MAX_AST_DEPTH}")
    _validate_node(parsed.body)
    return parsed


def _ast_depth(node: ast.AST) -> int:
    children = list(ast.iter_child_nodes(node))
    if not children:
        return 1
    return 1 + max(_ast_depth(child) for child in children)


def _validate_node(node: ast.AST) -> None:
    if isinstance(node, ast.Name):
        if node.id != "x":
            raise _array_error(f"unknown name {node.id!r}")
        return
    if isinstance(node, ast.Constant):
        if type(node.value) not in (int, float):
            raise _array_error("constants must be finite numbers")
        if isinstance(node.value, float) and not math.isfinite(node.value):
            raise _array_error("constants must be finite numbers")
        return
    if isinstance(node, ast.BinOp) and type(node.op) in _ALLOWED_BINARY:
        _validate_node(node.left)
        _validate_node(node.right)
        return
    if isinstance(node, ast.UnaryOp) and type(node.op) in _ALLOWED_UNARY:
        _validate_node(node.operand)
        return
    if isinstance(node, ast.Call):
        _validate_call(node)
        return
    raise _array_error(f"unsupported syntax {type(node).__name__}")


def _validate_call(node: ast.Call) -> None:
    if not isinstance(node.func, ast.Name):
        raise _array_error("functions must be called directly")
    name = node.func.id
    if name not in _ALLOWED_FUNCTIONS:
        raise _array_error(f"unknown function {name!r}")
    if node.keywords:
        raise _array_error("function keyword arguments are unsupported")
    expected = 1 if name in _UNARY_FUNCTIONS else 2
    if len(node.args) != expected:
        suffix = "" if expected == 1 else "s"
        raise _array_error(f"{name} expects {expected} argument{suffix}")
    _validate_node(node.args[0])
    if name == "reshape":
        _reshape_shape(node.args[1])


def _reshape_shape(node: ast.AST) -> tuple[int, ...]:
    if not isinstance(node, (ast.Tuple, ast.List)):
        raise _array_error("reshape shape must be a tuple or list")
    if len(node.elts) > _MAX_RESHAPE_RANK:
        raise _array_error(f"reshape rank limit is {_MAX_RESHAPE_RANK}")
    dimensions: list[int] = []
    for item in node.elts:
        value: object
        if isinstance(item, ast.Constant):
            value = item.value
        elif (
            isinstance(item, ast.UnaryOp)
            and isinstance(item.op, ast.USub)
            and isinstance(item.operand, ast.Constant)
            and type(item.operand.value) is int
        ):
            value = -item.operand.value
        else:
            raise _array_error("reshape dimensions must be integers")
        if type(value) is not int:
            raise _array_error("reshape dimensions must be integers")
        dimensions.append(value)
    if dimensions.count(-1) > 1:
        raise _array_error("reshape shape allows at most one -1")
    if any(dimension < -1 for dimension in dimensions):
        raise _array_error("reshape dimensions must be non-negative or -1")
    if any(dimension > _MAX_RESHAPE_DIMENSION for dimension in dimensions):
        raise _array_error(f"reshape dimension limit is {_MAX_RESHAPE_DIMENSION}")
    known_elements = math.prod(dimension for dimension in dimensions if dimension != -1)
    if known_elements > _MAX_RESHAPE_ELEMENTS:
        raise _array_error(f"reshape output limit is {_MAX_RESHAPE_ELEMENTS} elements")
    return tuple(dimensions)


def _evaluate(node: ast.AST, value: object, namespace: object) -> object:
    if isinstance(node, ast.Name):
        return value
    if isinstance(node, ast.Constant):
        return node.value
    if isinstance(node, ast.BinOp):
        function = _ALLOWED_BINARY[type(node.op)]
        return function(
            _evaluate(node.left, value, namespace),
            _evaluate(node.right, value, namespace),
        )
    if isinstance(node, ast.UnaryOp):
        function = _ALLOWED_UNARY[type(node.op)]
        return function(_evaluate(node.operand, value, namespace))
    if isinstance(node, ast.Call):
        name = node.func.id  # type: ignore[union-attr]
        function = getattr(namespace, name)
        arguments = [_evaluate(node.args[0], value, namespace)]
        if name == "reshape":
            arguments.append(_reshape_shape(node.args[1]))
        return function(*arguments)
    raise AssertionError("validated array expression contained an unsupported node")


def _validate_options(options: Mapping[str, object]) -> ast.Expression:
    unknown = set(options) - {"expression", "udfs"}
    if unknown:
        raise _array_error(f"unsupported options: {', '.join(sorted(unknown))}")
    udfs = options.get("udfs", [])
    if udfs:
        raise _array_error("custom array UDFs are unavailable")
    return _parse_expression(options.get("expression"))


def _validate_provider_options(
    provider: str, name: str, version: str, options: Mapping[str, object]
) -> None:
    if provider in {"numpy", "jax"} and name == "expression" and version == "1":
        _validate_options(options)


def _owned_numpy(value: object) -> object:
    import numpy as np

    owner = np.array(value, copy=True)
    owner.setflags(write=False)
    view = owner.view()
    view.setflags(write=False)
    return view


def _owned_jax(value: object) -> object:
    import jax
    import jax.numpy as jnp

    array = jnp.asarray(value)
    if not isinstance(array, jax.Array):
        raise TypeError("JAX batches require a jax.Array payload")
    return array


def _prepare_array(value: object, backend: str) -> tuple[object, int]:
    if backend == "numpy":
        owned = _owned_numpy(value)
    elif backend == "jax":
        owned = _owned_jax(value)
    else:
        owned = value
    shape = getattr(owned, "shape", None)
    length = int(shape[0]) if shape else 1
    if length < 0:
        raise ValueError("array length must not be negative")
    return owned, length


@dataclass(frozen=True, slots=True)
class _ArrayProvider:
    backend: str
    namespace: object

    def validate(self, options: Mapping[str, object]) -> None:
        _validate_options(options)

    def __call__(
        self, batch: _native.Batch, options: Mapping[str, object]
    ) -> _native.Batch:
        if batch.backend != self.backend:
            raise TypeError(
                f"provider requires backend {self.backend}, received {batch.backend}"
            )
        parsed = _validate_options(options)
        result = _evaluate(parsed.body, batch.array, self.namespace)
        if self.backend == "jax":
            import jax

            result = _owned_jax(result)
            if not isinstance(result, jax.Array):
                raise TypeError("JAX provider output must remain a jax.Array")
        return _native.Batch.from_array(
            result, backend=self.backend, metadata=batch.metadata
        )


def register_numpy(runtime: Runtime) -> None:
    import numpy as np

    runtime.register_provider("numpy", "expression", "1", _ArrayProvider("numpy", np))


def register_jax(runtime: Runtime) -> None:
    import jax.numpy as jnp

    runtime.register_provider("jax", "expression", "1", _ArrayProvider("jax", jnp))
