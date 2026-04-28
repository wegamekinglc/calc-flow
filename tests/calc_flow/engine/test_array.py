from __future__ import annotations

import numpy as np
import pytest

from calc_flow.engine.array import ArrayEngine, JaxEngine, NumpyEngine
from calc_flow.engine.base import Engine

# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------


def _assert_array(result: object, expected: object) -> None:
    assert hasattr(result, "__array_namespace__")
    assert result.tolist() == expected  # type: ignore[union-attr]


# ---------------------------------------------------------------------------
# existing evaluate tests (adapted for raw arrays)
# ---------------------------------------------------------------------------


def test_all_array_engines_are_engines() -> None:
    for cls in [NumpyEngine, JaxEngine]:
        engine = cls()
        assert isinstance(engine, Engine)
        assert isinstance(engine, ArrayEngine)


def test_numpy_engine_evaluate_expression() -> None:
    engine = NumpyEngine()
    arr = np.asarray([1, 2, 3])

    result = engine.evaluate("x * 2", arr)

    _assert_array(result, [2, 4, 6])


def test_numpy_engine_evaluate_scalar_expression() -> None:
    engine = NumpyEngine()
    arr = np.asarray([[1, 2], [3, 4]])

    result = engine.evaluate("xp.maximum(x[0], x[1])", arr)

    _assert_array(result, [3, 4])


def test_numpy_engine_evaluate_with_assignment() -> None:
    engine = NumpyEngine()
    arr = np.asarray([1, 2, 3])

    result = engine.evaluate("c = x * 2", arr)

    _assert_array(result, [2, 4, 6])


def test_jax_engine_evaluate() -> None:
    jnp = pytest.importorskip("jax.numpy")

    engine = JaxEngine()
    arr = jnp.asarray([1, 2, 3])

    result = engine.evaluate("x * 2", arr)

    _assert_array(result, [2, 4, 6])


def test_jax_engine_evaluate_array_batch() -> None:
    jnp = pytest.importorskip("jax.numpy")

    engine = JaxEngine()
    arr = jnp.asarray([1, 2, 3])

    result = engine.evaluate("x * 2", arr)

    assert hasattr(result, "__array_namespace__")
    assert result.tolist() == [2, 4, 6]


# ---------------------------------------------------------------------------
# element-wise binary operations — NumpyEngine
# ---------------------------------------------------------------------------


def test_numpy_add_two_arrays() -> None:
    engine = NumpyEngine()
    a = engine.xp.asarray([1, 2, 3])
    b = engine.xp.asarray([10, 20, 30])

    result = engine.add(a, b)

    _assert_array(result, [11, 22, 33])


def test_numpy_add_array_and_scalar() -> None:
    engine = NumpyEngine()
    a = engine.xp.asarray([1, 2, 3])

    result = engine.add(a, 10)

    _assert_array(result, [11, 12, 13])


def test_numpy_add_array_and_raw_array() -> None:
    engine = NumpyEngine()
    a = engine.xp.asarray([1, 2, 3])

    result = engine.add(a, np.asarray([10, 20, 30]))

    _assert_array(result, [11, 22, 33])


def test_numpy_subtract() -> None:
    engine = NumpyEngine()
    a = engine.xp.asarray([10, 20, 30])
    b = engine.xp.asarray([1, 2, 3])

    result = engine.subtract(a, b)

    _assert_array(result, [9, 18, 27])


def test_numpy_multiply() -> None:
    engine = NumpyEngine()
    a = engine.xp.asarray([1, 2, 3])

    result = engine.multiply(a, 10)

    _assert_array(result, [10, 20, 30])


def test_numpy_divide() -> None:
    engine = NumpyEngine()
    a = engine.xp.asarray([10.0, 20.0, 30.0])

    result = engine.divide(a, 2.0)

    _assert_array(result, [5.0, 10.0, 15.0])


# ---------------------------------------------------------------------------
# matmul — NumpyEngine
# ---------------------------------------------------------------------------


def test_numpy_matmul_two_arrays() -> None:
    engine = NumpyEngine()
    a = engine.xp.asarray([[1, 2], [3, 4]])
    b = engine.xp.asarray([[5, 6], [7, 8]])

    result = engine.matmul(a, b)

    _assert_array(result, [[19, 22], [43, 50]])


def test_numpy_matmul_array_and_raw_array() -> None:
    engine = NumpyEngine()
    a = engine.xp.asarray([[1, 2], [3, 4]])

    result = engine.matmul(a, np.asarray([[5, 6], [7, 8]]))

    _assert_array(result, [[19, 22], [43, 50]])


# ---------------------------------------------------------------------------
# reductions — NumpyEngine
# ---------------------------------------------------------------------------


def test_numpy_sum_axis_none() -> None:
    engine = NumpyEngine()
    a = engine.xp.asarray([[1, 2], [3, 4]])

    result = engine.sum(a)

    _assert_array(result, 10)


def test_numpy_sum_axis_zero() -> None:
    engine = NumpyEngine()
    a = engine.xp.asarray([[1, 2], [3, 4]])

    result = engine.sum(a, axis=0)

    _assert_array(result, [4, 6])


def test_numpy_mean_axis_none() -> None:
    engine = NumpyEngine()
    a = engine.xp.asarray([[1.0, 2.0], [3.0, 4.0]])

    result = engine.mean(a)

    _assert_array(result, 2.5)


def test_numpy_mean_axis_zero() -> None:
    engine = NumpyEngine()
    a = engine.xp.asarray([[1.0, 2.0], [3.0, 4.0]])

    result = engine.mean(a, axis=0)

    _assert_array(result, [2.0, 3.0])


def test_numpy_max_axis_none() -> None:
    engine = NumpyEngine()
    a = engine.xp.asarray([3, 1, 4, 1, 5])

    result = engine.max(a)

    _assert_array(result, 5)


def test_numpy_max_axis_zero() -> None:
    engine = NumpyEngine()
    a = engine.xp.asarray([[3, 1], [4, 1]])

    result = engine.max(a, axis=0)

    _assert_array(result, [4, 1])


def test_numpy_min_axis_none() -> None:
    engine = NumpyEngine()
    a = engine.xp.asarray([3, 1, 4, 1, 5])

    result = engine.min(a)

    _assert_array(result, 1)


def test_numpy_min_axis_zero() -> None:
    engine = NumpyEngine()
    a = engine.xp.asarray([[3, 1], [4, 1]])

    result = engine.min(a, axis=0)

    _assert_array(result, [3, 1])


# ---------------------------------------------------------------------------
# shape / transform — NumpyEngine
# ---------------------------------------------------------------------------


def test_numpy_transpose_default() -> None:
    engine = NumpyEngine()
    a = engine.xp.asarray([[1, 2], [3, 4]])

    result = engine.transpose(a)

    _assert_array(result, [[1, 3], [2, 4]])


def test_numpy_transpose_explicit_axes() -> None:
    engine = NumpyEngine()
    a = engine.xp.asarray([[1, 2], [3, 4]])

    result = engine.transpose(a, axes=(0, 1))

    _assert_array(result, [[1, 2], [3, 4]])


def test_numpy_reshape() -> None:
    engine = NumpyEngine()
    a = engine.xp.asarray([1, 2, 3, 4])

    result = engine.reshape(a, (2, 2))

    _assert_array(result, [[1, 2], [3, 4]])


# ---------------------------------------------------------------------------
# backend coercion
# ---------------------------------------------------------------------------


def test_numpy_add_coerces_scalar_input() -> None:
    engine = NumpyEngine()
    a = engine.xp.asarray([1, 2, 3])

    result = engine.add(a, 10)

    _assert_array(result, [11, 12, 13])


def test_jax_add_coerces_numpy_input() -> None:
    pytest.importorskip("jax")

    engine = JaxEngine()
    a = engine.xp.asarray([1, 2, 3])

    result = engine.add(a, np.asarray([10, 20, 30]))

    _assert_array(result, [11, 22, 33])


# ---------------------------------------------------------------------------
# JaxEngine operation tests
# ---------------------------------------------------------------------------


def test_jax_add() -> None:
    pytest.importorskip("jax")

    engine = JaxEngine()
    a = engine.xp.asarray([1, 2, 3])
    b = engine.xp.asarray([10, 20, 30])

    result = engine.add(a, b)

    _assert_array(result, [11, 22, 33])


def test_jax_subtract() -> None:
    pytest.importorskip("jax")

    engine = JaxEngine()
    a = engine.xp.asarray([10, 20, 30])
    b = engine.xp.asarray([1, 2, 3])

    result = engine.subtract(a, b)

    _assert_array(result, [9, 18, 27])


def test_jax_multiply() -> None:
    pytest.importorskip("jax")

    engine = JaxEngine()
    a = engine.xp.asarray([1, 2, 3])

    result = engine.multiply(a, 10)

    _assert_array(result, [10, 20, 30])


def test_jax_divide() -> None:
    pytest.importorskip("jax")

    engine = JaxEngine()
    a = engine.xp.asarray([10.0, 20.0, 30.0])

    result = engine.divide(a, 2.0)

    _assert_array(result, [5.0, 10.0, 15.0])


def test_jax_matmul() -> None:
    pytest.importorskip("jax")

    engine = JaxEngine()
    a = engine.xp.asarray([[1, 2], [3, 4]])
    b = engine.xp.asarray([[5, 6], [7, 8]])

    result = engine.matmul(a, b)

    _assert_array(result, [[19, 22], [43, 50]])


def test_jax_sum_axis_none() -> None:
    pytest.importorskip("jax")

    engine = JaxEngine()
    a = engine.xp.asarray([[1, 2], [3, 4]])

    result = engine.sum(a)

    _assert_array(result, 10)


def test_jax_sum_axis_zero() -> None:
    pytest.importorskip("jax")

    engine = JaxEngine()
    a = engine.xp.asarray([[1, 2], [3, 4]])

    result = engine.sum(a, axis=0)

    _assert_array(result, [4, 6])


def test_jax_mean_axis_none() -> None:
    pytest.importorskip("jax")

    engine = JaxEngine()
    a = engine.xp.asarray([[1.0, 2.0], [3.0, 4.0]])

    result = engine.mean(a)

    _assert_array(result, 2.5)


def test_jax_max_axis_none() -> None:
    pytest.importorskip("jax")

    engine = JaxEngine()
    a = engine.xp.asarray([3, 1, 4, 1, 5])

    result = engine.max(a)

    _assert_array(result, 5)


def test_jax_min_axis_none() -> None:
    pytest.importorskip("jax")

    engine = JaxEngine()
    a = engine.xp.asarray([3, 1, 4, 1, 5])

    result = engine.min(a)

    _assert_array(result, 1)


def test_jax_transpose() -> None:
    pytest.importorskip("jax")

    engine = JaxEngine()
    a = engine.xp.asarray([[1, 2], [3, 4]])

    result = engine.transpose(a)

    _assert_array(result, [[1, 3], [2, 4]])


def test_jax_reshape() -> None:
    pytest.importorskip("jax")

    engine = JaxEngine()
    a = engine.xp.asarray([1, 2, 3, 4])

    result = engine.reshape(a, (2, 2))

    _assert_array(result, [[1, 2], [3, 4]])


def test_jax_evaluate_expression() -> None:
    pytest.importorskip("jax")

    engine = JaxEngine()
    arr = engine.xp.asarray([1, 2, 3])

    result = engine.evaluate("x * 2", arr)

    _assert_array(result, [2, 4, 6])
