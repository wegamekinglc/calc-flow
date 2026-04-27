from __future__ import annotations

from typing import Any

import pyarrow as pa

from calc_flow.batch import Batch
from calc_flow.engine.base import Engine
from calc_flow.expression import split_assignment


class ArrayEngine(Engine):
    """Base class for array-backed engines using the Python array API standard."""


def _column_arrays(batch: Batch, namespace: Any) -> dict[str, Any]:
    return {
        name: namespace.asarray(
            batch.table[name].combine_chunks().to_numpy(zero_copy_only=False)
        )
        for name in batch.schema.names
    }


def _to_arrow_array(value: Any, rows: int) -> pa.Array:
    if hasattr(value, "tolist"):
        value = value.tolist()
    if not isinstance(value, list):
        value = [value] * rows
    return pa.array(value)


def _evaluate_expression(expression: str, batch: Batch, namespace: Any) -> Batch:
    assignment = split_assignment(expression)
    value_expression = assignment[1] if assignment is not None else expression

    if batch.is_array:
        array = namespace.asarray(batch.array)
        scope = {"array": array, "x": array, "xp": namespace}
        result = eval(value_expression, {"__builtins__": {}}, scope)
        if assignment is None:
            return Batch(result)
        column, _ = assignment
        return Batch.from_arrays({column: result})

    scope = _column_arrays(batch, namespace)
    scope["xp"] = namespace
    result = eval(value_expression, {"__builtins__": {}}, scope)
    result_array = _to_arrow_array(result, batch.num_rows)

    if assignment is None:
        return Batch(pa.table({"result": result_array}))

    column, _ = assignment
    table = batch.table
    index = table.schema.get_field_index(column)
    if index == -1:
        return Batch(table.append_column(column, result_array))
    return Batch(table.set_column(index, column, result_array))


class NumpyEngine(ArrayEngine):
    """NumPy-backed computation engine via array API standard."""

    def evaluate(self, expression: str, batch: Batch, **kwargs: Any) -> Batch:
        import numpy as np

        return _evaluate_expression(expression, batch, np)


class JaxEngine(ArrayEngine):
    """JAX-backed computation engine via array API standard."""

    def evaluate(self, expression: str, batch: Batch, **kwargs: Any) -> Batch:
        try:
            import jax.numpy as jnp
        except ImportError as exc:
            msg = "JaxEngine requires the 'jax' package"
            raise ImportError(msg) from exc

        return _evaluate_expression(expression, batch, jnp)
