from __future__ import annotations

from typing import Any

from calc_flow.engine.base import Engine
from calc_flow.expression import split_assignment


class ArrayEngine(Engine):
    """Base class for array-backed engines using the Python array API standard."""

    @property
    def xp(self) -> Any:
        raise NotImplementedError

    def add(self, left: Any, right: Any) -> Any:
        return self.xp.add(self.xp.asarray(left), self.xp.asarray(right))

    def subtract(self, left: Any, right: Any) -> Any:
        return self.xp.subtract(self.xp.asarray(left), self.xp.asarray(right))

    def multiply(self, left: Any, right: Any) -> Any:
        return self.xp.multiply(self.xp.asarray(left), self.xp.asarray(right))

    def divide(self, left: Any, right: Any) -> Any:
        return self.xp.divide(self.xp.asarray(left), self.xp.asarray(right))

    def matmul(self, left: Any, right: Any) -> Any:
        return self.xp.matmul(self.xp.asarray(left), self.xp.asarray(right))

    def sum(self, data: Any, *, axis: int | None = None) -> Any:
        return self.xp.asarray(self.xp.sum(self.xp.asarray(data), axis=axis))

    def mean(self, data: Any, *, axis: int | None = None) -> Any:
        return self.xp.asarray(self.xp.mean(self.xp.asarray(data), axis=axis))

    def max(self, data: Any, *, axis: int | None = None) -> Any:
        return self.xp.asarray(self.xp.max(self.xp.asarray(data), axis=axis))

    def min(self, data: Any, *, axis: int | None = None) -> Any:
        return self.xp.asarray(self.xp.min(self.xp.asarray(data), axis=axis))

    def transpose(self, data: Any, *, axes: tuple[int, ...] | None = None) -> Any:
        arr = self.xp.asarray(data)
        if axes is None:
            axes = tuple(range(arr.ndim - 1, -1, -1))
        return self.xp.permute_dims(arr, axes=axes)

    def reshape(self, data: Any, shape: int | tuple[int, ...]) -> Any:
        return self.xp.reshape(self.xp.asarray(data), shape)


def _evaluate_expression(expression: str, data: Any, namespace: Any) -> Any:
    assignment = split_assignment(expression)
    value_expression = assignment[1] if assignment is not None else expression

    arr = namespace.asarray(data)
    scope = {"x": arr, "xp": namespace}
    return eval(value_expression, {"__builtins__": {}}, scope)


class NumpyEngine(ArrayEngine):
    """NumPy-backed computation engine via array API standard."""

    @property
    def xp(self) -> Any:
        import numpy as np

        return np

    def evaluate(self, expression: str, data: Any, **kwargs: Any) -> Any:
        return _evaluate_expression(expression, data, self.xp)


class JaxEngine(ArrayEngine):
    """JAX-backed computation engine via array API standard."""

    @property
    def xp(self) -> Any:
        try:
            import jax.numpy as jnp
        except ImportError as exc:
            msg = "JaxEngine requires the 'jax' package"
            raise ImportError(msg) from exc

        return jnp

    def evaluate(self, expression: str, data: Any, **kwargs: Any) -> Any:
        return _evaluate_expression(expression, data, self.xp)
