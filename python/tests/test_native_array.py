from __future__ import annotations

import runpy
import subprocess
import sys
from dataclasses import FrozenInstanceError
from pathlib import Path

import numpy as np
import pyarrow as pa
import pytest

import calc_flow.array as cfa
from calc_flow import Batch, ConfigError


def test_native_array_example_runs_without_optional_backends(
    capsys: pytest.CaptureFixture[str],
) -> None:
    runpy.run_path(
        str(Path(__file__).parents[2] / "examples" / "native_arrays.py"),
        run_name="__main__",
    )
    assert capsys.readouterr().out.splitlines() == [
        "Native values: [0.0, 0.0, 4.0, 6.0]",
        "Native sum: 10.0",
    ]


@pytest.mark.parametrize("dtype", ["float32", "float64"])
def test_native_numpy_snapshot_is_owned_and_deeply_readonly(dtype: str) -> None:
    source = np.array([1.0, -0.0, np.nan, 4.0], dtype=dtype)[::-1]
    expected = source.copy()
    array = cfa.from_numpy(source)
    source[:] = 100.0
    assert isinstance(array, cfa.NativeArray)
    assert array.dtype == dtype
    assert array.shape == (4,)
    assert len(array) == 4
    assert array.estimated_bytes == 4 * expected.itemsize + np.dtype(np.uintp).itemsize
    assert array.retained_buffer_bytes >= array.estimated_bytes
    assert "100.0" not in repr(array)
    result = array.to_numpy()
    np.testing.assert_array_equal(
        result.view("u" + str(expected.itemsize)),
        expected.view("u" + str(expected.itemsize)),
    )
    while isinstance(result, np.ndarray):
        with pytest.raises(ValueError):
            result.setflags(write=True)
        result = result.base
    assert isinstance(result, bytes)
    with pytest.raises(AttributeError):
        array.shape = (2, 2)


def test_native_arrow_slice_preserves_nulls_and_shares_buffers() -> None:
    source = pa.array([9.0, None, float("nan"), -0.0, 5.0]).slice(1, 3)
    array = cfa.from_arrow(source)
    exported = array.to_arrow()
    assert exported.null_count == 1
    assert exported[0].as_py() is None
    assert np.isnan(exported[1].as_py())
    assert np.signbit(exported[2].as_py())
    assert (
        exported.buffers()[1].address + exported.offset * 8
        == source.buffers()[1].address + source.offset * 8
    )
    with pytest.raises(ValueError, match="null|fill_value"):
        array.to_numpy()
    result = array.to_numpy(fill_value=-7.0)
    assert result[0] == -7.0
    assert np.isnan(result[1])
    assert np.signbit(result[2])
    with pytest.raises(ValueError):
        result.setflags(write=True)
    assert source[0].as_py() is None


def test_native_scalar_and_empty_shapes() -> None:
    scalar = cfa.from_numpy(np.array(3.0, dtype="float32"))
    assert scalar.shape == ()
    assert len(scalar) == 1
    assert scalar.to_numpy().shape == ()
    assert cfa.from_arrow(pa.array([3.0]), shape=()).shape == ()
    empty = cfa.from_arrow(pa.array([], type=pa.float64()))
    assert empty.shape == (0,)
    assert empty.to_numpy().shape == (0,)


def test_native_boolean_arrow_owner_preserves_mask_without_float_coercion() -> None:
    array = cfa.from_arrow(pa.array([True, None, False]))
    assert array.dtype == "bool"
    assert array.to_arrow().to_pylist() == [True, None, False]
    result = array.to_numpy(fill_value=False)
    np.testing.assert_array_equal(result, [True, False, False])
    assert result.dtype == np.dtype("bool")
    with pytest.raises(ValueError):
        result.setflags(write=True)
    with pytest.raises(ConfigError, match="input.dtype"):
        cfa.sum(array)


def test_native_conversions_reject_wrong_dtype_rank_and_shape() -> None:
    with pytest.raises(TypeError, match="ndarray"):
        cfa.from_numpy([1.0])
    with pytest.raises(ConfigError, match="input.dtype"):
        cfa.from_numpy(np.array([1], dtype="int64"))
    with pytest.raises(ConfigError, match="input.shape"):
        cfa.from_numpy(np.ones((1, 1, 1)))
    with pytest.raises(ConfigError, match="input.dtype"):
        cfa.from_arrow(pa.array([1]))
    with pytest.raises(ConfigError, match="input.shape"):
        cfa.from_arrow(pa.array([1.0]), shape=(2,))
    with pytest.raises(TypeError, match="shape"):
        cfa.from_arrow(pa.array([1.0]), shape=(True,))


def test_native_batch_uses_owned_payload_and_copies_metadata() -> None:
    array = cfa.from_arrow(pa.array([1.0, None]))
    metadata = {"label": ["original"]}
    batch = Batch.from_array(array, backend="native", metadata=metadata)
    metadata["label"].append("changed")
    assert batch.kind == "array"
    assert batch.backend == "native"
    assert batch.num_rows == 2
    assert batch.metadata == {"label": ["original"]}
    assert isinstance(batch.array, cfa.NativeArray)
    assert batch.array is not array
    assert batch.array.to_arrow().equals(array.to_arrow())
    assert (
        batch.array.to_arrow().buffers()[1].address
        == array.to_arrow().buffers()[1].address
    )
    with pytest.raises(TypeError, match="NativeArray"):
        Batch.from_array(np.array([1.0]), backend="native")


@pytest.mark.parametrize(
    "operation,expected", [("sum", 6.0), ("mean", 3.0), ("min", 2.0), ("max", 4.0)]
)
def test_native_reductions_preserve_null_policy(
    operation: str, expected: float
) -> None:
    array = cfa.from_arrow(pa.array([2.0, None, 4.0], type=pa.float32()))
    function = getattr(cfa, operation)
    assert function(array).to_arrow().to_pylist() == [None]
    result = function(array, skip_nulls=True)
    assert result.shape == ()
    assert result.to_arrow().to_pylist() == [expected]
    assert result.dtype == ("float32" if operation in ("min", "max") else "float64")
    assert array.to_arrow().to_pylist() == [2.0, None, 4.0]


def test_native_execution_options_are_frozen_and_enforce_native_budgets() -> None:
    options = cfa.ArrayExecutionOptions(max_elements=0)
    assert options.max_allocation_bytes == 256 * 1024 * 1024
    with pytest.raises((FrozenInstanceError, AttributeError)):
        options.max_elements = 1
    array = cfa.from_arrow(pa.array([1.0]))
    with pytest.raises(ConfigError, match="outputs.output.elements"):
        cfa.sum(array, execution=options)
    with pytest.raises(ConfigError, match="outputs.output.bytes|scratch.bytes"):
        cfa.mean(array, execution=cfa.ArrayExecutionOptions(max_allocation_bytes=0))
    with pytest.raises(TypeError, match="max_elements"):
        cfa.ArrayExecutionOptions(max_elements=True)
    with pytest.raises(TypeError, match="skip_nulls"):
        cfa.sum(array, skip_nulls=1)


def test_native_ewma_uses_table_warmup_and_missing_value_contract() -> None:
    array = cfa.from_arrow(pa.array([None, 2.0, float("nan"), 4.0, 8.0]))
    result = cfa.ewma(array, span=3, min_periods=2)
    assert result.to_arrow().to_pylist() == [None, None, None, 3.0, 5.5]
    assert result.dtype == "float64"
    with pytest.raises(ConfigError, match="span"):
        cfa.ewma(array, span=0)
    with pytest.raises(ConfigError, match="min_periods"):
        cfa.ewma(array, span=3, min_periods=0)


@pytest.mark.parametrize(
    "operation,expected",
    [
        ("neg", [-1.0, -4.0]),
        ("abs", [1.0, 4.0]),
        ("sqrt", [1.0, 2.0]),
        ("log", [0.0, np.log(4.0)]),
        ("exp", [np.e, np.exp(4.0)]),
    ],
)
def test_native_unary_functions_preserve_dtype_and_nulls(
    operation: str, expected: list[float]
) -> None:
    array = cfa.from_arrow(pa.array([1.0, None, 4.0], type=pa.float32()))
    result = getattr(cfa, operation)(array)
    assert result.dtype == "float32"
    assert result.to_arrow()[1].as_py() is None
    np.testing.assert_allclose(
        result.to_numpy(fill_value=0.0)[[0, 2]], expected, rtol=1e-6
    )


@pytest.mark.parametrize(
    "operation,expected",
    [
        ("add", [4.0, None, 6.0]),
        ("subtract", [0.0, None, 2.0]),
        ("multiply", [4.0, None, 8.0]),
        ("divide", [1.0, None, 2.0]),
        ("pow", [4.0, None, 16.0]),
        ("equal", [True, None, False]),
        ("not_equal", [False, None, True]),
        ("less", [False, None, False]),
        ("less_equal", [True, None, False]),
        ("greater", [False, None, True]),
        ("greater_equal", [True, None, True]),
    ],
)
def test_native_binary_functions_use_explicit_scalar_broadcast_and_promotion(
    operation: str, expected: list[float | bool | None]
) -> None:
    left = cfa.from_arrow(pa.array([2.0, None, 4.0], type=pa.float32()))
    right = cfa.from_arrow(pa.array([2.0]), shape=())
    result = getattr(cfa, operation)(left, right)
    assert result.to_arrow().to_pylist() == expected
    assert result.shape == (3,)
    assert result.dtype == (
        "float64"
        if operation in ("add", "subtract", "multiply", "divide", "pow")
        else "bool"
    )


@pytest.mark.parametrize(
    "operation,scalar,expected",
    [
        ("logical_and", False, [False, False, False]),
        ("logical_or", True, [True, True, True]),
        ("logical_xor", True, [False, None, True]),
    ],
)
def test_native_boolean_functions_use_kleene_null_rules(
    operation: str, scalar: bool, expected: list[bool | None]
) -> None:
    array = cfa.from_arrow(pa.array([True, None, False]))
    other = cfa.from_arrow(pa.array([scalar]), shape=())
    assert getattr(cfa, operation)(array, other).to_arrow().to_pylist() == expected
    assert cfa.logical_not(array).to_arrow().to_pylist() == [False, None, True]


def test_native_where_and_shape_errors_use_core_rules() -> None:
    condition = cfa.from_arrow(pa.array([True, None, False]))
    left = cfa.from_arrow(pa.array([2.0, None, 4.0], type=pa.float32()))
    scalar = cfa.from_arrow(pa.array([9.0]), shape=())
    assert cfa.where_(condition, left, scalar).to_arrow().to_pylist() == [
        2.0,
        None,
        9.0,
    ]
    vector = cfa.from_arrow(pa.array([2.0]))
    with pytest.raises(ConfigError, match="inputs.right.shape"):
        cfa.add(left, vector)
    with pytest.raises(ConfigError, match="inputs.right.shape"):
        cfa.pow(left, vector)
    with pytest.raises(ConfigError, match="outputs.output.elements"):
        cfa.add(left, scalar, execution=cfa.ArrayExecutionOptions(max_elements=0))


@pytest.mark.parametrize("profile", ["stable_v1", "stable_v2"])
def test_native_rolling_mean_preserves_row_window_and_profile(profile: str) -> None:
    array = cfa.from_arrow(pa.array([1.0, None, 3.0, float("nan"), 5.0]))
    result = cfa.rolling_mean(array, window=3, min_periods=2, numerical_profile=profile)
    assert result.to_arrow().to_pylist() == [None, None, 2.0, None, 4.0]
    assert result.dtype == "float64"
    with pytest.raises(ConfigError, match="options.min_periods"):
        cfa.rolling_mean(array, window=2, min_periods=3)
    with pytest.raises(ConfigError, match="options.numerical_profile"):
        cfa.rolling_mean(array, window=2, numerical_profile="unknown")


def test_native_arrow_path_does_not_import_optional_array_backends() -> None:
    program = """
import sys
class WithoutBackends:
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split('.')[0] in ('numpy', 'jax'):
            raise ImportError('optional backend deliberately unavailable')
sys.meta_path.insert(0, WithoutBackends())
import pyarrow as pa
import calc_flow.array as cfa
array = cfa.from_arrow(pa.array([1., None], type=pa.float64()))
assert array.to_arrow().to_pylist() == [1., None]
assert cfa.sum(array, skip_nulls=True).to_arrow().to_pylist() == [1.]
assert not any(name == 'jax' or name.startswith('jax.') for name in sys.modules)
assert not any(
    name == 'numpy' or name.startswith('numpy.')
    for name in sys.modules
)
"""
    # The interpreter and source are test-owned; no shell or caller input is used.
    subprocess.run(  # noqa: E501  # nosemgrep: python.lang.security.audit.dangerous-subprocess-use-audit.dangerous-subprocess-use-audit
        [sys.executable, "-c", program], check=True
    )
