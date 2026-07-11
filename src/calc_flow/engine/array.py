from __future__ import annotations

import ast
import operator
from collections.abc import Iterable, Mapping
from functools import lru_cache, reduce
from typing import Any

from calc_flow.batch import Batch, BatchKind
from calc_flow.engine.base import Engine
from calc_flow.expression import split_assignment
from calc_flow.udf import ArrayUdf, UdfReference, UdfRegistry, UdfRegistrySnapshot

_BINARY_OPERATORS = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.FloorDiv: operator.floordiv,
    ast.Mod: operator.mod,
    ast.Pow: operator.pow,
    ast.MatMult: operator.matmul,
    ast.BitAnd: operator.and_,
    ast.BitOr: operator.or_,
    ast.BitXor: operator.xor,
}
_UNARY_OPERATORS = {
    ast.UAdd: operator.pos,
    ast.USub: operator.neg,
    ast.Invert: operator.invert,
}
_COMPARISON_OPERATORS = {
    ast.Eq: operator.eq,
    ast.NotEq: operator.ne,
    ast.Lt: operator.lt,
    ast.LtE: operator.le,
    ast.Gt: operator.gt,
    ast.GtE: operator.ge,
}
_ALLOWED_NAMESPACE_FUNCTIONS = frozenset(
    {
        "abs",
        "add",
        "concatenate",
        "cos",
        "divide",
        "exp",
        "log",
        "matmul",
        "max",
        "maximum",
        "mean",
        "min",
        "minimum",
        "multiply",
        "permute_dims",
        "reshape",
        "sin",
        "sqrt",
        "stack",
        "subtract",
        "sum",
        "tanh",
        "where",
    }
)


@lru_cache(maxsize=256)
def _parse_expression(expression: str) -> ast.expr:
    try:
        return ast.parse(expression, mode="eval").body
    except SyntaxError as exc:
        msg = f"invalid array expression: {exc.msg}"
        raise ValueError(msg) from exc


class _ExpressionEvaluator:
    def __init__(
        self,
        data: Any,
        namespace: Any,
        udfs: Mapping[str, ArrayUdf] | None = None,
    ) -> None:
        self._data = data
        self._namespace = namespace
        self._udfs = udfs or {}

    def evaluate(self, node: ast.AST) -> Any:
        method = getattr(self, f"_evaluate_{type(node).__name__}", None)
        if method is None:
            msg = f"array expressions do not allow {type(node).__name__}"
            raise ValueError(msg)
        return method(node)

    def _evaluate_Constant(self, node: ast.Constant) -> Any:
        if node.value is None or isinstance(node.value, bool | int | float | str):
            return node.value
        msg = f"array expressions do not allow {type(node.value).__name__} literals"
        raise ValueError(msg)

    def _evaluate_Name(self, node: ast.Name) -> Any:
        if node.id == "x":
            return self._data
        msg = f"unknown array expression name {node.id!r}"
        raise ValueError(msg)

    def _evaluate_BinOp(self, node: ast.BinOp) -> Any:
        function = _BINARY_OPERATORS.get(type(node.op))
        if function is None:
            msg = f"array expressions do not allow {type(node.op).__name__}"
            raise ValueError(msg)
        return function(self.evaluate(node.left), self.evaluate(node.right))

    def _evaluate_UnaryOp(self, node: ast.UnaryOp) -> Any:
        function = _UNARY_OPERATORS.get(type(node.op))
        if function is None:
            msg = f"array expressions do not allow {type(node.op).__name__}"
            raise ValueError(msg)
        return function(self.evaluate(node.operand))

    def _evaluate_BoolOp(self, node: ast.BoolOp) -> Any:
        if isinstance(node.op, ast.And):
            function = self._namespace.logical_and
        elif isinstance(node.op, ast.Or):
            function = self._namespace.logical_or
        else:
            msg = f"array expressions do not allow {type(node.op).__name__}"
            raise ValueError(msg)
        return reduce(function, (self.evaluate(value) for value in node.values))

    def _evaluate_Compare(self, node: ast.Compare) -> Any:
        left = self.evaluate(node.left)
        comparisons = []
        for op, comparator in zip(node.ops, node.comparators, strict=True):
            function = _COMPARISON_OPERATORS.get(type(op))
            if function is None:
                msg = f"array expressions do not allow {type(op).__name__}"
                raise ValueError(msg)
            right = self.evaluate(comparator)
            comparisons.append(function(left, right))
            left = right
        return reduce(self._namespace.logical_and, comparisons)

    def _evaluate_Subscript(self, node: ast.Subscript) -> Any:
        return self.evaluate(node.value)[self.evaluate(node.slice)]

    def _evaluate_Slice(self, node: ast.Slice) -> slice:
        return slice(
            self.evaluate(node.lower) if node.lower is not None else None,
            self.evaluate(node.upper) if node.upper is not None else None,
            self.evaluate(node.step) if node.step is not None else None,
        )

    def _evaluate_Tuple(self, node: ast.Tuple) -> tuple[Any, ...]:
        return tuple(self.evaluate(item) for item in node.elts)

    def _evaluate_List(self, node: ast.List) -> list[Any]:
        return [self.evaluate(item) for item in node.elts]

    def _evaluate_Call(self, node: ast.Call) -> Any:
        if isinstance(node.func, ast.Name) and node.func.id in self._udfs:
            if node.keywords:
                msg = "array UDF calls do not allow keyword arguments"
                raise ValueError(msg)
            arguments = [self.evaluate(argument) for argument in node.args]
            return self._udfs[node.func.id].invoke(
                *arguments, namespace=self._namespace
            )

        if not (
            isinstance(node.func, ast.Attribute)
            and isinstance(node.func.value, ast.Name)
            and node.func.value.id == "xp"
            and node.func.attr in _ALLOWED_NAMESPACE_FUNCTIONS
        ):
            msg = "array expressions may call only approved xp functions"
            raise ValueError(msg)

        function = getattr(self._namespace, node.func.attr, None)
        if function is None:
            msg = f"array namespace does not provide {node.func.attr!r}"
            raise ValueError(msg)
        args = [self.evaluate(arg) for arg in node.args]
        kwargs = {}
        for keyword in node.keywords:
            if keyword.arg is None:
                msg = "array expressions do not allow keyword expansion"
                raise ValueError(msg)
            kwargs[keyword.arg] = self.evaluate(keyword.value)
        return function(*args, **kwargs)


def _evaluate_expression(
    expression: str,
    data: Any,
    namespace: Any,
    udfs: Mapping[str, ArrayUdf] | None = None,
) -> Any:
    assignment = split_assignment(expression)
    value_expression = assignment[1] if assignment is not None else expression
    arr = namespace.asarray(data)
    tree = _parse_expression(value_expression)
    return _ExpressionEvaluator(arr, namespace, udfs).evaluate(tree)


class ArrayEngine(Engine):
    """Base class for Array API-backed engines."""

    input_kind = BatchKind.ARRAY

    def __init__(
        self,
        *,
        udf_registry: UdfRegistry | UdfRegistrySnapshot | None = None,
        udfs: Iterable[UdfReference] = (),
    ) -> None:
        if isinstance(udf_registry, UdfRegistry):
            registry = udf_registry.snapshot()
        else:
            registry = udf_registry or UdfRegistrySnapshot()
        self._configured_udfs = registry.array_functions(tuple(udfs))

    @property
    def xp(self) -> Any:
        raise NotImplementedError

    def _array(self, data: Batch) -> Any:
        self._require_kind(data)
        return self.xp.asarray(data.array_payload)

    def _operand(self, value: Batch | Any) -> Any:
        if isinstance(value, Batch):
            return self._array(value)
        return self.xp.asarray(value)

    def add(self, left: Batch, right: Batch | Any) -> Batch:
        return left.with_payload(self.xp.add(self._array(left), self._operand(right)))

    def subtract(self, left: Batch, right: Batch | Any) -> Batch:
        return left.with_payload(
            self.xp.subtract(self._array(left), self._operand(right))
        )

    def multiply(self, left: Batch, right: Batch | Any) -> Batch:
        return left.with_payload(
            self.xp.multiply(self._array(left), self._operand(right))
        )

    def divide(self, left: Batch, right: Batch | Any) -> Batch:
        return left.with_payload(
            self.xp.divide(self._array(left), self._operand(right))
        )

    def matmul(self, left: Batch, right: Batch | Any) -> Batch:
        return left.with_payload(
            self.xp.matmul(self._array(left), self._operand(right))
        )

    def sum(self, data: Batch, *, axis: int | None = None) -> Batch:
        result = self.xp.asarray(self.xp.sum(self._array(data), axis=axis))
        return data.with_payload(result)

    def mean(self, data: Batch, *, axis: int | None = None) -> Batch:
        result = self.xp.asarray(self.xp.mean(self._array(data), axis=axis))
        return data.with_payload(result)

    def max(self, data: Batch, *, axis: int | None = None) -> Batch:
        result = self.xp.asarray(self.xp.max(self._array(data), axis=axis))
        return data.with_payload(result)

    def min(self, data: Batch, *, axis: int | None = None) -> Batch:
        result = self.xp.asarray(self.xp.min(self._array(data), axis=axis))
        return data.with_payload(result)

    def transpose(self, data: Batch, *, axes: tuple[int, ...] | None = None) -> Batch:
        arr = self._array(data)
        if axes is None:
            axes = tuple(range(arr.ndim - 1, -1, -1))
        return data.with_payload(self.xp.permute_dims(arr, axes=axes))

    def reshape(self, data: Batch, shape: int | tuple[int, ...]) -> Batch:
        return data.with_payload(self.xp.reshape(self._array(data), shape))


class NumpyEngine(ArrayEngine):
    """NumPy-backed computation engine via the Array API standard."""

    @property
    def xp(self) -> Any:
        import numpy as np

        return np

    def evaluate(
        self,
        expression: str,
        data: Batch,
        *,
        udfs: Mapping[str, ArrayUdf] | None = None,
    ) -> Batch:
        self._require_kind(data)
        active_udfs = self._configured_udfs if udfs is None else udfs
        result = _evaluate_expression(
            expression, data.array_payload, self.xp, active_udfs
        )
        return data.with_payload(self.xp.asarray(result))


class JaxEngine(ArrayEngine):
    """JAX-backed computation engine via the Array API standard."""

    @property
    def xp(self) -> Any:
        try:
            import jax.numpy as jnp
        except ImportError as exc:
            msg = "JaxEngine requires the 'jax' package"
            raise ImportError(msg) from exc

        return jnp

    def evaluate(
        self,
        expression: str,
        data: Batch,
        *,
        udfs: Mapping[str, ArrayUdf] | None = None,
    ) -> Batch:
        self._require_kind(data)
        active_udfs = self._configured_udfs if udfs is None else udfs
        result = _evaluate_expression(
            expression, data.array_payload, self.xp, active_udfs
        )
        return data.with_payload(self.xp.asarray(result))
