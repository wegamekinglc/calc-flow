from __future__ import annotations

import pytest

from calc_flow.batch import Batch
from calc_flow.engine.array import ArrayEngine, JaxEngine, NumpyEngine
from calc_flow.engine.base import Engine


def test_all_array_engines_are_engines() -> None:
    for cls in [NumpyEngine, JaxEngine]:
        engine = cls()
        assert isinstance(engine, Engine)
        assert isinstance(engine, ArrayEngine)


def test_numpy_engine_evaluate_assignment() -> None:
    engine = NumpyEngine()
    batch = Batch.from_pylist([{"a": 1, "b": 10}, {"a": 2, "b": 20}])

    result = engine.evaluate("c = a + b", batch)

    assert result.to_pylist() == [
        {"a": 1, "b": 10, "c": 11},
        {"a": 2, "b": 20, "c": 22},
    ]


def test_numpy_engine_evaluate_expression_result() -> None:
    engine = NumpyEngine()
    batch = Batch.from_arrays({"a": [1, 2], "b": [10, 20]})

    result = engine.evaluate("xp.maximum(a, b)", batch)

    assert result.schema.names == ["result"]
    assert result.to_pylist() == [{"result": 10}, {"result": 20}]


def test_numpy_engine_evaluate_array_batch() -> None:
    import numpy as np

    engine = NumpyEngine()
    batch = Batch.from_array(np.asarray([1, 2, 3]))

    result = engine.evaluate("array * 2", batch)

    assert result.is_array
    assert result.to_pylist() == [2, 4, 6]


def test_jax_engine_evaluate_assignment() -> None:
    pytest.importorskip("jax")

    engine = JaxEngine()
    batch = Batch.from_pylist([{"a": 1, "b": 10}, {"a": 2, "b": 20}])

    result = engine.evaluate("c = a + b", batch)

    assert result.to_pylist() == [
        {"a": 1, "b": 10, "c": 11},
        {"a": 2, "b": 20, "c": 22},
    ]


def test_jax_engine_evaluate_array_batch() -> None:
    jnp = pytest.importorskip("jax.numpy")

    engine = JaxEngine()
    batch = Batch.from_array(jnp.asarray([1, 2, 3]))

    result = engine.evaluate("array * 2", batch)

    assert result.is_array
    assert result.to_pylist() == [2, 4, 6]
