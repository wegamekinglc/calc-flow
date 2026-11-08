from __future__ import annotations

import numpy as np
import pyarrow as pa
import pytest

import calc_flow.array as cfa
from calc_flow import Batch, ConfigError


@pytest.mark.parametrize("dtype", ["bool", "float32", "float64"])
def test_native_matrix_numpy_snapshot_is_shaped_and_deeply_readonly(dtype: str) -> None:
    source = np.arange(12).reshape(3, 4).astype(dtype)[::-1, ::2]
    expected = source.copy()
    matrix = cfa.from_numpy(source)
    source[:] = 0
    assert matrix.shape == (3, 2)
    assert len(matrix) == 3
    assert matrix.element_count == 6
    assert len(matrix.to_arrow()) == 6
    output = matrix.to_numpy()
    np.testing.assert_array_equal(output, expected)
    while isinstance(output, np.ndarray):
        with pytest.raises(ValueError):
            output.setflags(write=True)
        output = output.base
    assert isinstance(output, bytes)
    assert Batch.from_array(matrix, backend="native").num_rows == 3


def test_native_matrix_arrow_mask_roundtrip_and_zero_columns() -> None:
    matrix = cfa.from_arrow(pa.array([1.0, None, 3.0, 4.0, 5.0, 6.0]), shape=(2, 3))
    with pytest.raises(ValueError, match="fill_value"):
        matrix.to_numpy()
    np.testing.assert_array_equal(
        matrix.to_numpy(fill_value=-1.0), [[1, -1, 3], [4, 5, 6]]
    )
    empty = cfa.from_numpy(np.empty((3, 0), dtype="float32"))
    assert len(empty) == 3
    assert empty.element_count == 0
    assert empty.to_numpy().shape == (3, 0)


def test_native_matmul_and_owned_shape_transforms() -> None:
    matrix = cfa.from_numpy(np.array([[1, 2, 3], [4, 5, 6]], dtype="float32"))
    weights = cfa.from_numpy(np.array([[1], [2], [3]], dtype="float64"))
    result = cfa.matmul(matrix, weights)
    assert result.dtype == "float64"
    np.testing.assert_array_equal(result.to_numpy(), [[14], [32]])
    flat = cfa.reshape(
        matrix,
        (6,),
        execution=cfa.ArrayExecutionOptions(
            max_allocation_bytes=np.dtype(np.uintp).itemsize
        ),
    )
    assert flat.shape == (6,)
    assert (
        flat.to_arrow().buffers()[1].address == matrix.to_arrow().buffers()[1].address
    )
    transposed = cfa.transpose(matrix)
    assert transposed.shape == (3, 2)
    np.testing.assert_array_equal(transposed.to_numpy(), [[1, 4], [2, 5], [3, 6]])
    assert (
        transposed.to_arrow().buffers()[1].address
        != matrix.to_arrow().buffers()[1].address
    )
    with pytest.raises(ConfigError, match="input.shape"):
        cfa.ewma(matrix, span=2)
    with pytest.raises(ConfigError, match="inputs.right.shape"):
        cfa.matmul(matrix, matrix)


@pytest.mark.parametrize("operation", ["sum", "mean", "min", "max"])
def test_native_matrix_reductions_axes_and_keepdims(operation: str) -> None:
    matrix = cfa.from_numpy(np.array([[1, 2, 3], [4, 5, 6]], dtype="float32"))
    function = getattr(cfa, operation)
    reference = getattr(np, operation)
    for axis, keepdims in [
        (None, False),
        (None, True),
        (0, False),
        (0, True),
        (1, False),
        (1, True),
    ]:
        result = function(matrix, axis=axis, keepdims=keepdims)
        expected = reference(
            np.array([[1, 2, 3], [4, 5, 6]]), axis=axis, keepdims=keepdims
        )
        assert result.shape == expected.shape
        np.testing.assert_array_equal(result.to_numpy(), expected)
        assert result.dtype == ("float32" if operation in ("min", "max") else "float64")
    with pytest.raises(ConfigError, match="options.axis"):
        function(matrix, axis=-1)
    with pytest.raises(ConfigError, match="options.axis"):
        function(matrix, axis=2)
    with pytest.raises(TypeError, match="keepdims"):
        function(matrix, keepdims=1)


def test_native_matrix_null_and_empty_reduction_lanes() -> None:
    matrix = cfa.from_arrow(pa.array([1.0, None, 3.0, 5.0]), shape=(2, 2))
    assert cfa.sum(matrix, axis=1).to_arrow().to_pylist() == [None, 8.0]
    assert cfa.sum(matrix, axis=1, skip_nulls=True).to_arrow().to_pylist() == [1.0, 8.0]
    empty = cfa.from_numpy(np.empty((0, 3)))
    assert cfa.sum(empty, axis=0).to_arrow().to_pylist() == [None, None, None]
    assert cfa.sum(empty, axis=1).shape == (0,)
    with pytest.raises(ConfigError, match="input.shape"):
        cfa.reshape(matrix, (3, 2))
    with pytest.raises(ConfigError, match="inputs.right.validity"):
        cfa.matmul(cfa.from_numpy(np.ones((2, 2))), matrix)


def test_native_matrix_example_runs(capsys: pytest.CaptureFixture[str]) -> None:
    import runpy
    from pathlib import Path

    runpy.run_path(
        str(Path(__file__).parents[2] / "examples" / "native_matrices.py"),
        run_name="__main__",
    )
    assert capsys.readouterr().out.splitlines() == [
        "Matrix product: [4.0, 5.0, 10.0, 11.0]",
        "Row means: [4.5, 10.5]",
    ]
