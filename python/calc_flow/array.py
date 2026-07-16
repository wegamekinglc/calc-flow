from __future__ import annotations

import ast
import math
import operator
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from functools import lru_cache
from typing import TYPE_CHECKING

from calc_flow import _native

if TYPE_CHECKING:
    from calc_flow.pipeline import Runtime

_MAX_AST_NODES = 128
_MAX_AST_DEPTH = 24
_MAX_EXPRESSION_LENGTH = 4096
_MAX_INTEGER_MAGNITUDE = 2**63 - 1
_MAX_POWER_EXPONENT_MAGNITUDE = 64
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
    if len(expression) > _MAX_EXPRESSION_LENGTH:
        raise _array_error(
            f"expression length limit is {_MAX_EXPRESSION_LENGTH} characters"
        )
    return _parse_valid_expression(expression)


@lru_cache(maxsize=256)
def _parse_valid_expression(expression: str) -> ast.Expression:
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
        if type(node.value) is int and abs(node.value) > _MAX_INTEGER_MAGNITUDE:
            raise _array_error(
                f"integer constant magnitude limit is {_MAX_INTEGER_MAGNITUDE}"
            )
        if isinstance(node.value, float) and not math.isfinite(node.value):
            raise _array_error("constants must be finite numbers")
        return
    if isinstance(node, ast.BinOp) and type(node.op) in _ALLOWED_BINARY:
        _validate_node(node.left)
        if isinstance(node.op, ast.Pow):
            exponent = _numeric_literal(node.right)
            if exponent is None:
                raise _array_error("power exponent must be a finite numeric literal")
            if abs(exponent) > _MAX_POWER_EXPONENT_MAGNITUDE:
                raise _array_error(
                    f"power exponent magnitude limit is {_MAX_POWER_EXPONENT_MAGNITUDE}"
                )
        _validate_node(node.right)
        return
    if isinstance(node, ast.UnaryOp) and type(node.op) in _ALLOWED_UNARY:
        _validate_node(node.operand)
        return
    if isinstance(node, ast.Call):
        _validate_call(node)
        return
    raise _array_error(f"unsupported syntax {type(node).__name__}")


def _numeric_literal(node: ast.AST) -> int | float | None:
    sign = 1
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, (ast.UAdd, ast.USub)):
        sign = -1 if isinstance(node.op, ast.USub) else 1
        node = node.operand
    if not isinstance(node, ast.Constant) or type(node.value) not in (int, float):
        return None
    if isinstance(node.value, float) and not math.isfinite(node.value):
        return None
    return sign * node.value


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


def _evaluate(
    node: ast.AST,
    value: object,
    namespace: object,
    validate_result: Callable[[object], None] | None = None,
) -> object:
    if isinstance(node, ast.Name):
        result = value
    elif isinstance(node, ast.Constant):
        result = node.value
    elif isinstance(node, ast.BinOp):
        function = _ALLOWED_BINARY[type(node.op)]
        result = function(
            _evaluate(node.left, value, namespace, validate_result),
            _evaluate(node.right, value, namespace, validate_result),
        )
    elif isinstance(node, ast.UnaryOp):
        function = _ALLOWED_UNARY[type(node.op)]
        result = function(_evaluate(node.operand, value, namespace, validate_result))
    elif isinstance(node, ast.Call):
        name = node.func.id  # type: ignore[union-attr]
        function = getattr(namespace, name)
        arguments = [_evaluate(node.args[0], value, namespace, validate_result)]
        if name == "reshape":
            arguments.append(_reshape_shape(node.args[1]))
        result = function(*arguments)
    else:
        raise AssertionError("validated array expression contained an unsupported node")
    if validate_result is not None:
        validate_result(result)
    return result


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

    array = np.asarray(value)
    _validate_numpy_dtype(array.dtype)
    owner = np.array(array, copy=True, order="C", subok=False)
    immutable_bytes = owner.tobytes(order="C")
    return np.frombuffer(immutable_bytes, dtype=owner.dtype).reshape(owner.shape)


def _validate_numpy_dtype(dtype: object) -> None:
    import numpy as np

    normalized = np.dtype(dtype)
    allowed_dtypes = frozenset(
        np.dtype(scalar_type)
        for scalar_type in (
            np.bool_,
            np.int8,
            np.int16,
            np.int32,
            np.int64,
            np.uint8,
            np.uint16,
            np.uint32,
            np.uint64,
            np.float32,
            np.float64,
            np.complex64,
            np.complex128,
        )
    )
    if normalized not in allowed_dtypes or not normalized.isnative:
        raise ValueError(
            f"NumPy arrays require a NumPy Array API dtype; received {normalized}"
        )


def _validate_python_operation_scalar(value: object) -> bool:
    if type(value) is int:
        if abs(value) > _MAX_INTEGER_MAGNITUDE:
            raise _array_error(
                f"integer constant magnitude limit is {_MAX_INTEGER_MAGNITUDE}"
            )
        return True
    return type(value) in (float, complex)


def _validate_numpy_operation_result(value: object) -> None:
    import numpy as np

    if _validate_python_operation_scalar(value):
        return
    if type(value) is np.ndarray or isinstance(value, np.generic):
        _validate_numpy_dtype(value.dtype)
        return
    raise TypeError("NumPy provider operations must produce arrays or numeric scalars")


def _validate_jax_operation_result(value: object) -> None:
    import jax

    if _validate_python_operation_scalar(value) or isinstance(value, jax.Array):
        return
    raise TypeError("JAX provider operations must produce arrays or numeric scalars")


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
        validate_result = {
            "jax": _validate_jax_operation_result,
            "numpy": _validate_numpy_operation_result,
        }[self.backend]
        result = _evaluate(parsed.body, batch.array, self.namespace, validate_result)
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
