from __future__ import annotations

from collections.abc import Iterator

import numpy as np
import pyarrow as pa
import pytest

from calc_flow.batch import Batch, BatchKind, BatchMetadata
from calc_flow.engine.array import ArrayEngine, JaxEngine, NumpyEngine
from calc_flow.engine.base import Engine


@pytest.fixture(params=["numpy", "jax"])
def engine(request: pytest.FixtureRequest) -> Iterator[ArrayEngine]:
    if request.param == "jax":
        pytest.importorskip("jax")
        yield JaxEngine()
    else:
        yield NumpyEngine()


def _batch(engine: ArrayEngine, value: object) -> Batch:
    return Batch.array(
        engine.xp.asarray(value), metadata=BatchMetadata(source_id="array-source")
    )


def _assert_array(result: Batch, expected: object) -> None:
    assert result.kind is BatchKind.ARRAY
    assert result.array_payload.tolist() == expected


def test_array_engines_are_engines(engine: ArrayEngine) -> None:
    assert isinstance(engine, Engine)
    assert engine.input_kind is BatchKind.ARRAY


@pytest.mark.parametrize(
    ("expression", "expected"),
    [
        ("x * 2", [2, 4, 6]),
        ("c = x + 3", [4, 5, 6]),
        ("xp.maximum(x, 2)", [2, 2, 3]),
        ("xp.where(x > 1, x, 0)", [0, 2, 3]),
        ("x[1:]", [2, 3]),
    ],
)
def test_array_engine_evaluate(
    engine: ArrayEngine, expression: str, expected: object
) -> None:
    result = engine.evaluate(expression, _batch(engine, [1, 2, 3]))
    _assert_array(result, expected)
    assert result.metadata.source_id == "array-source"


@pytest.mark.parametrize(
    "expression",
    [
        "__import__('os')",
        "x.__class__",
        "[item for item in x]",
        "lambda value: value",
        "open('/tmp/file')",
        "xp.__dict__",
    ],
)
def test_array_engine_rejects_unsafe_expressions(
    engine: ArrayEngine, expression: str
) -> None:
    with pytest.raises(ValueError):
        engine.evaluate(expression, _batch(engine, [1, 2, 3]))


@pytest.mark.parametrize(
    ("method", "left", "right", "expected"),
    [
        ("add", [1, 2], [10, 20], [11, 22]),
        ("subtract", [10, 20], [1, 2], [9, 18]),
        ("multiply", [1, 2], 10, [10, 20]),
        ("divide", [10.0, 20.0], 2.0, [5.0, 10.0]),
        ("matmul", [[1, 2], [3, 4]], [[5, 6], [7, 8]], [[19, 22], [43, 50]]),
    ],
)
def test_array_binary_operations(
    engine: ArrayEngine,
    method: str,
    left: object,
    right: object,
    expected: object,
) -> None:
    left_batch = _batch(engine, left)
    right_value = _batch(engine, right) if isinstance(right, list) else right

    result = getattr(engine, method)(left_batch, right_value)

    _assert_array(result, expected)
    assert result.metadata is left_batch.metadata


@pytest.mark.parametrize(
    ("method", "axis", "expected"),
    [
        ("sum", None, 10.0),
        ("sum", 0, [4.0, 6.0]),
        ("mean", None, 2.5),
        ("max", 0, [3.0, 4.0]),
        ("min", 0, [1.0, 2.0]),
    ],
)
def test_array_reductions(
    engine: ArrayEngine, method: str, axis: int | None, expected: object
) -> None:
    batch = _batch(engine, [[1.0, 2.0], [3.0, 4.0]])

    result = getattr(engine, method)(batch, axis=axis)

    _assert_array(result, expected)


def test_array_shape_operations(engine: ArrayEngine) -> None:
    batch = _batch(engine, [1, 2, 3, 4])

    reshaped = engine.reshape(batch, (2, 2))
    transposed = engine.transpose(reshaped)

    _assert_array(reshaped, [[1, 2], [3, 4]])
    _assert_array(transposed, [[1, 3], [2, 4]])


def test_array_engine_rejects_raw_or_table_input(engine: ArrayEngine) -> None:
    with pytest.raises(TypeError, match="requires a Batch"):
        engine.evaluate("x", np.asarray([1]))  # type: ignore[arg-type]

    with pytest.raises(TypeError, match="requires array batches"):
        engine.evaluate("x", Batch.table(pa.table({"x": [1]})))
