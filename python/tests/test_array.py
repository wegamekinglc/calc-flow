from __future__ import annotations

import ast
import copy
import gc
import os
import re
import runpy
import subprocess
import sys
import warnings
import weakref
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pyarrow as pa
import pytest

import calc_flow.array as array_module
from calc_flow import (
    Batch,
    PipelineBuilder,
    ProviderError,
    Runtime,
    register_jax,
    register_numpy,
)


def test_array_and_dataframe_example_uses_table_matmul(
    capsys: pytest.CaptureFixture[str],
) -> None:
    example = Path(__file__).parents[2] / "examples" / "07_array_and_dataframe.py"

    runpy.run_path(example, run_name="__main__")

    assert capsys.readouterr().out.splitlines() == [
        "NumPy result: [[6.0, 10.0], [2.0, 12.0], [8.0, 10.0]]",
        "JAX result: [[6.0, 10.0], [2.0, 12.0], [8.0, 10.0]]",
    ]


def test_table_matmul_docs_describe_jax_result_residency() -> None:
    root = Path(__file__).parents[2]
    claim = "no result-to-host round trip during operator execution"
    documents = (
        root / "examples" / "README.md",
        root / "docs" / "python-api.md",
        root / "docs" / "api-reference.md",
    )

    for document in documents:
        assert claim in document.read_text(encoding="utf-8")


def test_array_and_dataframe_example_defers_and_survives_missing_jax() -> None:
    example = Path(__file__).parents[2] / "examples" / "07_array_and_dataframe.py"
    program = f"""
import builtins
import runpy
import sys

import calc_flow

assert not any(name == "jax" or name.startswith("jax.") for name in sys.modules)
namespace = runpy.run_path({str(example)!r})
assert not any(name == "jax" or name.startswith("jax.") for name in sys.modules)
original_import = builtins.__import__

def reject_jax(name, *args, **kwargs):
    if name == "jax" or name.startswith("jax."):
        raise ImportError("JAX intentionally unavailable")
    return original_import(name, *args, **kwargs)

builtins.__import__ = reject_jax
namespace["main"]()
"""

    completed = subprocess.run(
        [sys.executable, "-c", program],
        capture_output=True,
        check=False,
        cwd=example.parents[1],
        env={**os.environ, "JAX_PLATFORMS": "cpu"},
        text=True,
    )

    assert completed.returncode == 0, completed.stderr
    assert completed.stdout.splitlines() == [
        "NumPy result: [[6.0, 10.0], [2.0, 12.0], [8.0, 10.0]]",
        "JAX result: skipped; install calc-flow[jax]",
    ]


def _external(
    name: str,
    provider: str,
    expression: str,
    *,
    options: dict[str, object] | None = None,
) -> PipelineBuilder:
    return PipelineBuilder(name).external(
        "calc",
        provider,
        "expression",
        "1",
        {"expression": expression, **(options or {})},
    )


class _ArraySubclass(np.ndarray):
    pass


@dataclass(frozen=True, slots=True)
class _NumpyOwnershipCase:
    name: str
    factory: Callable[[], np.ndarray[Any, Any]]


def _c_contiguous_input() -> np.ndarray[Any, Any]:
    return np.arange(12, dtype=np.int64).reshape(3, 4)


def _non_contiguous_slice_input() -> np.ndarray[Any, Any]:
    return np.arange(24, dtype=np.int64).reshape(4, 6)[:, ::2]


def _transposed_input() -> np.ndarray[Any, Any]:
    return np.arange(12, dtype=np.int64).reshape(3, 4).T


def _fortran_order_input() -> np.ndarray[Any, Any]:
    return np.asfortranarray(np.arange(12, dtype=np.int64).reshape(3, 4))


def _negative_stride_input() -> np.ndarray[Any, Any]:
    return np.arange(12, dtype=np.int64)[::-2]


def _scalar_input() -> np.ndarray[Any, Any]:
    return np.array(7, dtype=np.int64)


def _shaped_empty_input() -> np.ndarray[Any, Any]:
    return np.empty((2, 0, 3), dtype=np.int64)


def _subclass_input() -> np.ndarray[Any, Any]:
    return np.arange(8, dtype=np.int64).view(_ArraySubclass)


def _nested_view_input() -> np.ndarray[Any, Any]:
    owner = np.arange(60, dtype=np.int64).reshape(5, 12)
    return owner[1:5, 1:11][:, ::3].T


def _dtype_input(dtype: type[np.generic]) -> np.ndarray[Any, Any]:
    return np.array([0, 1, 2], dtype=dtype)


_SUPPORTED_OWNERSHIP_DTYPES = (
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

_NUMPY_OWNERSHIP_CASES = (
    _NumpyOwnershipCase("c_contiguous", _c_contiguous_input),
    _NumpyOwnershipCase("non_contiguous_slice", _non_contiguous_slice_input),
    _NumpyOwnershipCase("transpose", _transposed_input),
    _NumpyOwnershipCase("fortran_order", _fortran_order_input),
    _NumpyOwnershipCase("negative_stride", _negative_stride_input),
    _NumpyOwnershipCase("zero_dimensional", _scalar_input),
    _NumpyOwnershipCase("shaped_empty", _shaped_empty_input),
    _NumpyOwnershipCase("ndarray_subclass", _subclass_input),
    _NumpyOwnershipCase("nested_view_base", _nested_view_input),
    *(
        _NumpyOwnershipCase(
            f"dtype_{np.dtype(dtype).name}",
            lambda dtype=dtype: _dtype_input(dtype),
        )
        for dtype in _SUPPORTED_OWNERSHIP_DTYPES
    ),
)


def _mutate_source(source: np.ndarray[Any, Any]) -> None:
    if source.size == 0:
        return
    if source.dtype == np.dtype(np.bool_):
        source.flat[0] = not bool(source.flat[0])
    else:
        source.flat[0] = source.flat[0] + 1


@pytest.mark.parametrize(
    "case",
    _NUMPY_OWNERSHIP_CASES,
    ids=lambda case: case.name,
)
def test_numpy_batch_ownership_preserves_layout_dtype_and_snapshot(
    case: _NumpyOwnershipCase,
) -> None:
    source = case.factory()
    expected = np.asarray(source).copy(order="C")
    metadata = {"case": case.name, "nested": {"value": 1}}
    batch = Batch.from_array(source, backend="numpy", metadata=metadata)

    _mutate_source(source)
    metadata["nested"]["value"] = 2
    output = batch.array

    assert type(output) is np.ndarray
    assert output.dtype == expected.dtype
    assert output.shape == expected.shape
    assert output.flags.c_contiguous
    np.testing.assert_array_equal(output, expected)
    assert batch.num_rows == (expected.shape[0] if expected.shape else 1)
    assert batch.metadata == {"case": case.name, "nested": {"value": 1}}

    current: object = output
    while isinstance(current, np.ndarray):
        assert not current.flags.writeable
        with pytest.raises(ValueError):
            current.setflags(write=True)
        current = current.base


def test_numpy_ownership_avoids_intermediate_array_copy(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = np.arange(12, dtype=np.int64).reshape(3, 4)[:, ::2]
    expected = source.copy(order="C")

    def reject_intermediate_copy(*_args: object, **_kwargs: object) -> None:
        pytest.fail("_owned_numpy must not call np.array")

    monkeypatch.setattr(np, "array", reject_intermediate_copy)

    output = array_module._owned_numpy(source)

    np.testing.assert_array_equal(output, expected)


def test_owned_numpy_result_is_adopted_without_copy_and_cannot_be_reopened() -> None:
    owned, token = Batch._new_owned_numpy((2, 2), "float64")
    assert type(owned) is np.ndarray
    assert owned.flags.writeable
    assert not owned.flags.owndata
    owned[:] = [[1.0, 2.0], [3.0, 4.0]]
    pointer = owned.__array_interface__["data"][0]

    batch = Batch._from_owned_array(
        owned,
        backend="numpy",
        token=token,
        metadata={"operation": "table_matmul"},
    )
    output = batch.array

    assert output is owned
    assert output.__array_interface__["data"][0] == pointer
    assert output.tolist() == [[1.0, 2.0], [3.0, 4.0]]
    assert output.flags.writeable is False
    assert not isinstance(output.base, np.ndarray)
    assert not hasattr(output.base, "ptr")
    with pytest.raises(ValueError):
        output.setflags(write=True)
    with pytest.raises(ValueError, match="already consumed"):
        Batch._from_owned_array(
            owned,
            backend="numpy",
            token=token,
            metadata={},
        )


@pytest.mark.parametrize(
    "dtype",
    [
        "int8",
        "int16",
        "int32",
        "int64",
        "uint8",
        "uint16",
        "uint32",
        "uint64",
        "float32",
        "float64",
        "complex64",
        "complex128",
    ],
)
def test_owned_numpy_supports_exact_native_numeric_dtypes(dtype: str) -> None:
    owned, _ = Batch._new_owned_numpy((2, 1), dtype)

    assert type(owned) is np.ndarray
    assert owned.dtype == np.dtype(dtype)
    assert owned.shape == (2, 1)
    assert owned.tolist() == [[0], [0]]
    assert owned.flags.writeable
    assert not owned.flags.owndata


@pytest.mark.parametrize("dtype", ["bool", "float16", "object", ">f8"])
def test_owned_numpy_rejects_unsupported_dtypes(dtype: str) -> None:
    with pytest.raises(
        ValueError,
        match=(
            "owned NumPy arrays require a supported native numeric dtype; "
            f"received {dtype}"
        ),
    ):
        Batch._new_owned_numpy((1, 1), dtype)


@pytest.mark.parametrize(
    ("shape", "message"),
    [
        ((), "must have at least one dimension"),
        ((1,) * 17, "must have at most 16 dimensions"),
        ((0,), "dimensions must be positive"),
        ((1_000_001,), "dimension exceeds 1000000"),
        ((1_000_000, 11), "element count exceeds 10000000"),
    ],
)
def test_owned_numpy_validates_shape_before_allocation(
    shape: tuple[int, ...],
    message: str,
) -> None:
    with pytest.raises(ValueError, match=message):
        Batch._new_owned_numpy(shape, "float64")


def test_owned_numpy_token_rejects_a_different_array() -> None:
    first, token = Batch._new_owned_numpy((1, 1), "float32")
    second, _ = Batch._new_owned_numpy((1, 1), "float32")
    with pytest.raises(ValueError, match="does not match"):
        Batch._from_owned_array(
            second,
            backend="numpy",
            token=token,
            metadata={},
        )
    assert first.flags.writeable
    batch = Batch._from_owned_array(
        first,
        backend="numpy",
        token=token,
        metadata={},
    )
    assert batch.array is first


def test_owned_array_token_is_private_frozen_and_non_cloneable() -> None:
    _, token = Batch._new_owned_numpy((1,), "float32")

    assert type(token).__name__ == "_OwnedArrayToken"
    assert not hasattr(token, "object_identity")
    assert not hasattr(token, "consumed")
    with pytest.raises(AttributeError):
        token.extra = True
    with pytest.raises(TypeError):
        type(token)()
    with pytest.raises(TypeError):
        copy.copy(token)


def test_owned_array_token_anchors_exact_identity_until_adoption() -> None:
    owned, token = Batch._new_owned_numpy((1,), "float32")
    owned_ref = weakref.ref(owned)

    del owned
    gc.collect()
    assert owned_ref() is not None

    del token
    gc.collect()
    assert owned_ref() is None


def test_owned_numpy_storage_lives_until_the_last_exported_object_is_gone() -> None:
    owned, token = Batch._new_owned_numpy((2,), "float64")
    owned_ref = weakref.ref(owned)
    batch = Batch._from_owned_array(
        owned,
        backend="numpy",
        token=token,
        metadata={},
    )

    del owned
    del token
    gc.collect()
    assert owned_ref() is batch.array

    exported = batch.array
    del batch
    gc.collect()
    assert owned_ref() is exported

    del exported
    gc.collect()
    assert owned_ref() is None


def test_owned_array_adoption_rejects_backend_and_token_mismatches() -> None:
    owned, token = Batch._new_owned_numpy((1,), "float64")
    with pytest.raises(TypeError, match="require an ownership token"):
        Batch._from_owned_array(
            owned,
            backend="numpy",
            token=None,
            metadata={},
        )
    with pytest.raises(TypeError, match="do not accept"):
        Batch._from_owned_array(
            owned,
            backend="jax",
            token=token,
            metadata={},
        )
    with pytest.raises(ValueError, match="must be 'numpy' or 'jax'"):
        Batch._from_owned_array(
            owned,
            backend="other",
            token=token,
            metadata={},
        )
    with pytest.raises(TypeError, match="require a jax.Array"):
        Batch._from_owned_array(
            owned,
            backend="jax",
            token=None,
            metadata={},
        )

    batch = Batch._from_owned_array(
        owned,
        backend="numpy",
        token=token,
        metadata={},
    )
    assert batch.array is owned


def test_owned_jax_result_retains_identity_and_device_without_numpy_conversion(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    jax = pytest.importorskip("jax")
    jnp = pytest.importorskip("jax.numpy")
    result = jnp.asarray([[1.0, 2.0]])
    device = result.device

    def reject_numpy_conversion(*_args: object, **_kwargs: object) -> None:
        pytest.fail("owned JAX adoption must not convert through NumPy")

    monkeypatch.setattr(np, "asarray", reject_numpy_conversion)
    batch = Batch._from_owned_array(
        result,
        backend="jax",
        token=None,
        metadata={},
    )

    assert isinstance(batch.array, jax.Array)
    assert batch.array is result
    assert batch.array.device == device


def test_array_expression_cache_reuses_successful_exact_strings() -> None:
    cache = array_module._parse_valid_expression
    cache.cache_clear()
    try:
        first = array_module._parse_expression("x + 1")
        second = array_module._parse_expression("x + 1")

        assert first is second
        assert cache.cache_info().hits == 1
        assert cache.cache_info().misses == 1
    finally:
        cache.cache_clear()


def test_array_expression_cache_is_bounded() -> None:
    cache = array_module._parse_valid_expression
    cache.cache_clear()
    try:
        for value in range(257):
            array_module._parse_expression(f"x + {value}")
        before = cache.cache_info()

        array_module._parse_expression("x + 0")
        after = cache.cache_info()

        assert before.maxsize == 256
        assert before.currsize == 256
        assert after.misses == before.misses + 1
    finally:
        cache.cache_clear()


def test_array_expression_cache_does_not_cache_invalid_expressions(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    cache = array_module._parse_valid_expression
    cache.cache_clear()
    parse_calls = 0
    original_parse = array_module.ast.parse

    def counting_parse(*args: object, **kwargs: object) -> ast.AST:
        nonlocal parse_calls
        parse_calls += 1
        return original_parse(*args, **kwargs)

    monkeypatch.setattr(array_module.ast, "parse", counting_parse)
    try:
        errors: list[str] = []
        for _ in range(2):
            with pytest.raises(ValueError) as caught:
                array_module._parse_expression("x +")
            errors.append(str(caught.value))

        assert parse_calls == 2
        assert errors == ["invalid array expression: syntax is invalid"] * 2
        assert cache.cache_info().currsize == 0
    finally:
        cache.cache_clear()


def test_array_expression_cache_preserves_runtime_intermediate_validation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    cache = array_module._parse_valid_expression
    cache.cache_clear()
    results = iter(
        (
            np.array([1], dtype=np.int64),
            np.array([object()], dtype=object),
        )
    )
    monkeypatch.setattr(np, "sum", lambda _value: next(results))
    try:
        runtime = Runtime()
        register_numpy(runtime)
        plan = _external("cached_validation", "numpy", "sum(x) + 1").compile(runtime)
        batch = Batch.from_array(np.array([1]), backend="numpy")
        before_execution = cache.cache_info()

        assert plan.execute({"input": batch}).outputs["output"].array.tolist() == [2]
        after_first_execution = cache.cache_info()
        with pytest.raises(ProviderError, match="NumPy Array API dtype"):
            plan.execute({"input": batch})
        after_second_execution = cache.cache_info()

        assert after_first_execution.hits == before_execution.hits + 1
        assert after_second_execution.hits == after_first_execution.hits + 1
        assert after_second_execution.misses == 1
    finally:
        cache.cache_clear()


def test_numpy_provider_owns_read_only_arrays() -> None:
    runtime = Runtime()
    register_numpy(runtime)
    source = np.array([1.0, 2.0])
    batch = Batch.from_array(source, backend="numpy")
    source[0] = 99
    plan = _external("arrays", "numpy", "x * 2").compile(runtime)

    output = plan.execute({"input": batch}).outputs["output"].array

    assert output.tolist() == [2.0, 4.0]
    assert not output.flags.writeable
    with pytest.raises(ValueError):
        output[0] = 10
    with pytest.raises(ValueError):
        output.setflags(write=True)


def test_numpy_batch_has_no_reachable_writable_array_base() -> None:
    batch = Batch.from_array(np.array([1, 2]), backend="numpy")
    output = batch.array
    reachable_arrays: list[np.ndarray[Any, Any]] = []
    current: object = output
    while isinstance(current, np.ndarray):
        reachable_arrays.append(current)
        current = current.base

    assert len(reachable_arrays) >= 1
    for reachable in reachable_arrays:
        assert not reachable.flags.writeable
        with pytest.raises(ValueError):
            reachable.setflags(write=True)
    assert batch.array.tolist() == [1, 2]


def test_array_batch_copies_metadata_and_reports_shape_length() -> None:
    metadata = {"nested": {"value": 1}}
    batch = Batch.from_array(np.zeros((3, 2)), backend="numpy", metadata=metadata)
    metadata["nested"]["value"] = 2

    assert batch.kind == "array"
    assert batch.num_rows == 3
    assert batch.metadata == {"nested": {"value": 1}}
    assert not batch.array.flags.writeable
    with pytest.raises(TypeError, match="table batches do not contain an array"):
        _ = Batch.from_pyarrow(pa.table({"value": [1]})).array


def test_numpy_table_matmul_multiplies_selected_arrow_columns() -> None:
    runtime = Runtime()
    register_numpy(runtime)
    table = pa.table(
        {
            "quantity": [3.0, 1.0, 4.0],
            "unit_price": [10.0, 12.0, 10.0],
            "ignored": [99.0, 99.0, 99.0],
        }
    )
    weights = Batch.from_array(
        np.array([[2.0, 0.0], [0.0, 1.0]], dtype=np.float64),
        backend="numpy",
    )
    plan = (
        PipelineBuilder("numpy-table-matmul")
        .table_matmul(
            "multiply",
            backend="numpy",
            columns=("quantity", "unit_price"),
        )
        .compile(runtime)
    )

    run = plan.execute(
        {
            "table": Batch.from_pyarrow(table, {"source": "orders"}),
            "weights": weights,
        }
    )
    output = run.outputs["output"]

    assert output.kind == "array"
    assert output.backend == "numpy"
    assert output.array.tolist() == [[6.0, 10.0], [2.0, 12.0], [8.0, 10.0]]
    assert output.metadata == {
        "backend": "numpy",
        "columns": ["quantity", "unit_price"],
        "operation": "table_matmul",
        "source": "orders",
    }
    assert run.datafusion_metrics == []


@pytest.mark.parametrize(
    ("arrow_type", "weight_dtype", "result_dtype"),
    [
        (pa.int8(), np.int8, np.int8),
        (pa.int16(), np.int16, np.int16),
        (pa.int32(), np.int32, np.int32),
        (pa.int64(), np.int64, np.int64),
        (pa.uint8(), np.uint8, np.uint8),
        (pa.uint16(), np.uint16, np.uint16),
        (pa.uint32(), np.uint32, np.uint32),
        (pa.uint64(), np.uint64, np.uint64),
        (pa.float16(), np.float32, np.float32),
        (pa.float32(), np.float32, np.float32),
        (pa.float64(), np.float64, np.float64),
    ],
)
def test_numpy_table_matmul_accepts_primitive_arrow_numeric_dtypes(
    arrow_type: pa.DataType,
    weight_dtype: type[np.generic],
    result_dtype: type[np.generic],
) -> None:
    runtime = Runtime()
    register_numpy(runtime)
    plan = (
        PipelineBuilder(f"numpy-table-{arrow_type}")
        .table_matmul("multiply", backend="numpy", columns=("value",))
        .compile(runtime)
    )

    output = plan.execute(
        {
            "table": Batch.from_pyarrow(
                pa.table({"value": pa.array([2, 3], type=arrow_type)})
            ),
            "weights": Batch.from_array(
                np.array([[2]], dtype=weight_dtype),
                backend="numpy",
            ),
        }
    ).outputs["output"]

    assert output.array.dtype == np.dtype(result_dtype)
    assert output.array.tolist() == [[4], [6]]


@pytest.mark.parametrize(
    ("options", "message"),
    [
        ({}, "columns must be a JSON array"),
        ({"columns": "value"}, "columns must be a JSON array"),
        ({"columns": []}, "columns must contain at least one name"),
        ({"columns": [""]}, "columns must contain non-empty strings"),
        ({"columns": ["value", "value"]}, "columns must be unique"),
        (
            {"columns": ["value"], "unexpected": True},
            "unsupported options: unexpected",
        ),
    ],
)
@pytest.mark.parametrize("backend", ["numpy", "jax"])
def test_table_matmul_rejects_invalid_configuration(
    backend: str,
    options: dict[str, object],
    message: str,
) -> None:
    namespace = np if backend == "numpy" else pytest.importorskip("jax.numpy")
    provider = array_module._TableMatmulProvider(backend, namespace)

    with pytest.raises(
        ValueError,
        match=f"^invalid table_matmul options: {message}",
    ):
        provider.validate(options)


@pytest.mark.parametrize("backend", ["numpy", "jax"])
def test_table_matmul_validates_every_input_before_allocation(
    backend: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    namespace: Any = np if backend == "numpy" else pytest.importorskip("jax.numpy")

    def weights_batch(value: object, *, dtype: object = np.float32) -> Batch:
        return Batch.from_array(
            namespace.asarray(value, dtype=dtype),
            backend=backend,
        )

    valid_table = Batch.from_pyarrow(
        pa.table({"value": pa.array([1.0, 2.0], type=pa.float32())})
    )
    valid_weights = weights_batch([[1.0]])

    def table_batch(values: object) -> Batch:
        return Batch.from_pyarrow(pa.table({"value": values}))

    unsupported_tables = [
        table_batch(pa.array([True, False], type=pa.bool_())),
        table_batch(pa.array(["1", "2"], type=pa.string())),
        table_batch(pa.array([b"1", b"2"], type=pa.binary())),
        table_batch(pa.array([1, 2], type=pa.date32())),
        table_batch(pa.array([1, 2], type=pa.decimal128(4, 0))),
        table_batch(pa.array([[1], [2]], type=pa.list_(pa.int64()))),
        table_batch(
            pa.DictionaryArray.from_arrays(
                pa.array([0, 1], type=pa.int8()),
                pa.array([1, 2], type=pa.int64()),
            )
        ),
    ]
    cases: list[tuple[str, dict[str, Batch], dict[str, object], str]] = [
        ("empty inputs", {}, {"columns": ["value"]}, "inputs"),
        (
            "missing weights",
            {"table": valid_table},
            {"columns": ["value"]},
            "inputs",
        ),
        (
            "unexpected input",
            {
                "table": valid_table,
                "weights": valid_weights,
                "extra": valid_weights,
            },
            {"columns": ["value"]},
            "inputs",
        ),
        (
            "table kind",
            {"table": valid_weights, "weights": valid_weights},
            {"columns": ["value"]},
            "table",
        ),
        (
            "weights kind",
            {"table": valid_table, "weights": valid_table},
            {"columns": ["value"]},
            "weights",
        ),
        (
            "weights backend",
            {
                "table": valid_table,
                "weights": Batch._from_external(
                    np.array([[1.0]]),
                    "other",
                    1,
                    {},
                ),
            },
            {"columns": ["value"]},
            "weights.backend",
        ),
        (
            "empty table",
            {
                "table": Batch.from_pyarrow(
                    pa.table({"value": pa.array([], type=pa.float64())})
                ),
                "weights": valid_weights,
            },
            {"columns": ["value"]},
            "table.rows",
        ),
        (
            "missing column",
            {"table": valid_table, "weights": valid_weights},
            {"columns": ["missing"]},
            "columns",
        ),
        (
            "ambiguous column",
            {
                "table": Batch.from_pyarrow(
                    pa.Table.from_arrays(
                        [pa.array([1.0]), pa.array([2.0])],
                        names=["value", "value"],
                    )
                ),
                "weights": valid_weights,
            },
            {"columns": ["value"]},
            "columns",
        ),
        (
            "null column",
            {
                "table": table_batch(pa.array([1.0, None], type=pa.float64())),
                "weights": valid_weights,
            },
            {"columns": ["value"]},
            "columns",
        ),
        *(
            (
                f"unsupported column {index}",
                {"table": table, "weights": valid_weights},
                {"columns": ["value"]},
                "columns",
            )
            for index, table in enumerate(unsupported_tables)
        ),
        (
            "weights rank zero",
            {
                "table": valid_table,
                "weights": weights_batch(1.0),
            },
            {"columns": ["value"]},
            "weights.rank",
        ),
        (
            "weights rank one",
            {
                "table": valid_table,
                "weights": weights_batch([1.0]),
            },
            {"columns": ["value"]},
            "weights.rank",
        ),
        (
            "weights rank three",
            {
                "table": valid_table,
                "weights": weights_batch([[[1.0]]]),
            },
            {"columns": ["value"]},
            "weights.rank",
        ),
        (
            "weights input width",
            {
                "table": valid_table,
                "weights": weights_batch(namespace.ones((2, 1))),
            },
            {"columns": ["value"]},
            "weights.shape[0]",
        ),
        (
            "weights output width",
            {
                "table": valid_table,
                "weights": weights_batch(namespace.empty((1, 0))),
            },
            {"columns": ["value"]},
            "weights.shape[1]",
        ),
        (
            "unsupported dtype",
            {
                "table": table_batch(
                    pa.array(
                        [1.0, 2.0],
                        type=pa.float16() if backend == "numpy" else pa.float64(),
                    )
                ),
                "weights": weights_batch(
                    namespace.ones((1, 1)),
                    dtype=np.int8 if backend == "numpy" else np.float32,
                ),
            },
            {"columns": ["value"]},
            "dtype",
        ),
    ]

    current_case = ""

    def reject_allocation(*_args: object, **_kwargs: object) -> None:
        pytest.fail(
            f"{current_case}: invalid inputs must fail "
            "before the first dense allocation"
        )

    monkeypatch.setattr(Batch, "_new_owned_numpy", reject_allocation)
    provider = array_module._TableMatmulProvider(backend, namespace)
    for name, inputs, options, field in cases:
        current_case = name
        with pytest.raises(
            (TypeError, ValueError),
            match=rf"^invalid table_matmul {re.escape(field)}:",
        ) as caught:
            provider(inputs, options)
        assert field in str(caught.value), name


def test_numpy_table_matmul_rejects_lossy_backend_promotion() -> None:
    class LossyNamespace:
        @staticmethod
        def result_type(*_dtypes: object) -> np.dtype[Any]:
            return np.dtype(np.float32)

        can_cast = staticmethod(np.can_cast)

    weights = np.ones((1, 1), dtype=np.int64)

    with pytest.raises(
        TypeError,
        match=r"^invalid table_matmul dtype: common dtype float32 is lossy",
    ):
        array_module._common_matrix_dtype(
            "numpy",
            LossyNamespace(),
            (np.dtype(np.int64),),
            weights,
        )


def test_jax_table_matmul_rejects_x64_narrowing_before_allocation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    jax = pytest.importorskip("jax")
    jnp = pytest.importorskip("jax.numpy")
    if jax.config.x64_enabled:
        pytest.skip("requires JAX x64 to be disabled")
    provider = array_module._TableMatmulProvider("jax", jnp)
    inputs = {
        "table": Batch.from_pyarrow(
            pa.table({"value": pa.array([1.0], type=pa.float64())})
        ),
        "weights": Batch.from_array(
            jnp.asarray([[1.0]], dtype=jnp.float32),
            backend="jax",
        ),
    }

    def reject_allocation(*_args: object, **_kwargs: object) -> None:
        pytest.fail("JAX dtype narrowing must fail before host staging allocation")

    monkeypatch.setattr(Batch, "_new_owned_numpy", reject_allocation)
    with (
        warnings.catch_warnings(record=True) as caught_warnings,
        pytest.raises(
            TypeError,
            match=(
                r"^invalid table_matmul dtype: JAX x64 is disabled "
                r"for \[float64, float32\]"
            ),
        ),
    ):
        provider(inputs, {"columns": ["value"]})

    assert not caught_warnings


@pytest.mark.parametrize(
    ("arrow_type", "weight_dtype", "common_dtype", "involved_dtypes"),
    [
        (pa.float16(), "float16", "float16", "float16, float16"),
        (pa.int8(), "bfloat16", "bfloat16", "int8, bfloat16"),
    ],
)
def test_jax_table_matmul_rejects_unsupported_staging_dtype_before_allocation(
    monkeypatch: pytest.MonkeyPatch,
    arrow_type: pa.DataType,
    weight_dtype: str,
    common_dtype: str,
    involved_dtypes: str,
) -> None:
    jnp = pytest.importorskip("jax.numpy")
    provider = array_module._TableMatmulProvider("jax", jnp)
    inputs = {
        "table": Batch.from_pyarrow(
            pa.table({"value": pa.array([1], type=arrow_type)})
        ),
        "weights": Batch.from_array(
            jnp.asarray([[1]], dtype=jnp.dtype(weight_dtype)),
            backend="jax",
        ),
    }

    def reject_allocation(*_args: object, **_kwargs: object) -> None:
        pytest.fail("unsupported staging dtype must fail before host allocation")

    monkeypatch.setattr(Batch, "_new_owned_numpy", reject_allocation)
    with pytest.raises(
        TypeError,
        match=(
            rf"^invalid table_matmul dtype: common dtype {common_dtype} "
            rf"is unsupported for \[{involved_dtypes}\]"
        ),
    ):
        provider(inputs, {"columns": ["value"]})


def test_jax_table_matmul_rejects_result_dtype_narrowing_before_adoption(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    pytest.importorskip("jax")
    jnp = pytest.importorskip("jax.numpy")
    runtime = Runtime()
    register_jax(runtime)
    plan = (
        PipelineBuilder("jax-result-dtype")
        .table_matmul("multiply", backend="jax", columns=("value",))
        .compile(runtime)
    )
    narrowed_result = jnp.asarray([[2.0]], dtype=jnp.float16)
    original_from_owned = Batch._from_owned_array
    adoptions: list[object] = []

    def tracked_adoption(
        array: object,
        *,
        backend: str,
        token: object,
        metadata: dict[str, object],
    ) -> Batch:
        adoptions.append(array)
        return original_from_owned(
            array,
            backend=backend,
            token=token,
            metadata=metadata,
        )

    monkeypatch.setattr(jnp, "matmul", lambda _left, _right: narrowed_result)
    monkeypatch.setattr(Batch, "_from_owned_array", staticmethod(tracked_adoption))

    with pytest.raises(
        ProviderError,
        match=r"invalid table_matmul dtype: JAX changed float32 to float16",
    ):
        plan.execute(
            {
                "table": Batch.from_pyarrow(
                    pa.table({"value": pa.array([1.0], type=pa.float32())})
                ),
                "weights": Batch.from_array(
                    jnp.asarray([[2.0]], dtype=jnp.float32),
                    backend="jax",
                ),
            }
        )

    assert not adoptions


def test_numpy_table_matmul_honors_copy_and_ownership_ceiling(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    table = pa.table(
        {
            "quantity": pa.chunked_array(
                [
                    pa.array([3.0, 1.0], type=pa.float64()),
                    pa.array([4.0], type=pa.float64()),
                ]
            ),
            "unit_price": pa.chunked_array(
                [
                    pa.array([10.0], type=pa.float64()),
                    pa.array([12.0, 10.0], type=pa.float64()),
                ]
            ),
        }
    )
    expected_table = table.to_pydict()
    caller_weights = np.array(
        [[2.0, 0.0], [0.0, 1.0]],
        dtype=np.float64,
    )
    expected_caller_weights = caller_weights.copy()
    weights_batch = Batch.from_array(caller_weights, backend="numpy")
    weights_payload = weights_batch.array
    expected_weights_payload = weights_payload.copy()
    runtime = Runtime()
    register_numpy(runtime)
    plan = (
        PipelineBuilder("numpy-table-copy-ceiling")
        .table_matmul(
            "multiply",
            backend="numpy",
            columns=("quantity", "unit_price"),
        )
        .compile(runtime)
    )

    def reject_owned_numpy(*_args: object, **_kwargs: object) -> None:
        pytest.fail("table_matmul execution must not use the defensive copy path")

    allocations: list[tuple[tuple[int, ...], str, int]] = []
    original_new_owned = Batch._new_owned_numpy

    def counted(shape: tuple[int, ...], dtype: str) -> tuple[object, object]:
        array, token = original_new_owned(shape, dtype)
        allocations.append((tuple(shape), dtype, array.__array_interface__["data"][0]))
        return array, token

    seen_weights: list[object] = []
    original_matmul = np.matmul

    def tracked_matmul(
        left: object,
        right: object,
        *,
        out: object,
    ) -> object:
        seen_weights.append(right)
        return original_matmul(left, right, out=out)

    monkeypatch.setattr(array_module, "_owned_numpy", reject_owned_numpy)
    monkeypatch.setattr(Batch, "_new_owned_numpy", counted)
    monkeypatch.setattr(np, "matmul", tracked_matmul)

    output_batch = plan.execute(
        {
            "table": Batch.from_pyarrow(table),
            "weights": weights_batch,
        }
    ).outputs["output"]
    output = output_batch.array

    assert [(shape, dtype) for shape, dtype, _pointer in allocations] == [
        ((3, 2), "float64"),
        ((3, 2), "float64"),
    ]
    assert output.__array_interface__["data"][0] == allocations[1][2]
    assert seen_weights == [weights_payload]
    assert seen_weights[0] is weights_payload
    assert output.tolist() == [[6.0, 10.0], [2.0, 12.0], [8.0, 10.0]]

    current: object = output
    while current is not None:
        if isinstance(current, np.ndarray):
            assert not current.flags.writeable
        current = getattr(current, "base", None)
    with pytest.raises(ValueError):
        output.setflags(write=True)

    assert table.to_pydict() == expected_table
    np.testing.assert_array_equal(caller_weights, expected_caller_weights)
    np.testing.assert_array_equal(weights_payload, expected_weights_payload)


def test_numpy_table_matrix_does_not_combine_chunks(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class ChunkedColumn:
        def __init__(self, chunks: list[pa.Array]) -> None:
            self.chunks = chunks

        def combine_chunks(self) -> None:
            pytest.fail("table_matmul must not combine Arrow chunks")

    class ChunkedTable:
        num_rows = 3

        def __init__(self) -> None:
            self.columns = {
                "left": ChunkedColumn([pa.array([1.0, 2.0]), pa.array([3.0])]),
                "right": ChunkedColumn([pa.array([4.0]), pa.array([5.0, 6.0])]),
            }

        def __getitem__(self, name: str) -> ChunkedColumn:
            return self.columns[name]

    def reject_combine_chunks(*_args: object, **_kwargs: object) -> None:
        pytest.fail("table_matmul must not combine Arrow chunks")

    monkeypatch.setattr(
        ChunkedColumn,
        "combine_chunks",
        reject_combine_chunks,
    )

    matrix = array_module._numpy_table_matrix(
        ChunkedTable(),
        ("left", "right"),
        np.dtype(np.float64),
    )

    assert matrix.tolist() == [[1.0, 4.0], [2.0, 5.0], [3.0, 6.0]]


@pytest.mark.parametrize(
    ("backend", "register"),
    [("numpy", register_numpy), ("jax", register_jax)],
)
def test_table_matmul_registration_and_fingerprint_are_deterministic(
    backend: str,
    register: Callable[[Runtime], None],
) -> None:
    runtimes = [Runtime(), Runtime()]
    for runtime in runtimes:
        register(runtime)

    registrations = runtimes[0]._registration_snapshot()
    assert [
        (entry["provider"], entry["name"], entry["version"]) for entry in registrations
    ] == [
        (backend, "expression", "1"),
        (backend, "table_matmul", "1"),
    ]

    plans = [
        PipelineBuilder(f"deterministic-{backend}-table-matmul")
        .table_matmul(
            "multiply",
            backend=backend,
            columns=("quantity", "unit_price"),
        )
        .compile(runtime)
        for runtime in runtimes
    ]
    assert plans[0].fingerprint == plans[1].fingerprint


def test_jax_table_matmul_stays_on_jax() -> None:
    jax = pytest.importorskip("jax")
    jnp = pytest.importorskip("jax.numpy")
    runtime = Runtime()
    register_jax(runtime)
    weights = Batch.from_array(
        jnp.asarray([[2.0, 0.0], [0.0, 1.0]], dtype=jnp.float32),
        backend="jax",
    )
    plan = (
        PipelineBuilder("jax-table-matmul")
        .table_matmul(
            "multiply",
            backend="jax",
            columns=("quantity", "unit_price"),
        )
        .compile(runtime)
    )
    table = Batch.from_pyarrow(
        pa.table(
            {
                "quantity": pa.array([3.0, 1.0, 4.0], type=pa.float32()),
                "unit_price": pa.array([10.0, 12.0, 10.0], type=pa.float32()),
            }
        )
    )

    output = plan.execute({"table": table, "weights": weights}).outputs["output"]

    assert isinstance(output.array, jax.Array)
    assert output.backend == "jax"
    assert output.array.tolist() == [[6.0, 10.0], [2.0, 12.0], [8.0, 10.0]]
    assert output.array.device == weights.array.device


def test_jax_table_matmul_preserves_device_identity_and_copy_ceiling(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    jax = pytest.importorskip("jax")
    jnp = pytest.importorskip("jax.numpy")
    table = pa.table(
        {
            "a": pa.array([1.0, 2.0, 3.0], type=pa.float32()),
            "b": pa.array([4.0, 5.0, 6.0], type=pa.float32()),
        }
    )
    expected_table = table.to_pydict()
    weights = Batch.from_array(
        jnp.asarray([[1.0], [2.0]], dtype=jnp.float32),
        backend="jax",
    )
    weights_payload = weights.array
    expected_weights = weights_payload.tolist()
    runtime = Runtime()
    register_jax(runtime)
    plan = (
        PipelineBuilder("jax-copy-ceiling")
        .table_matmul("multiply", backend="jax", columns=("a", "b"))
        .compile(runtime)
    )
    original_asarray = np.asarray
    original_new_owned = Batch._new_owned_numpy
    original_device_put = jax.device_put
    original_matmul = jnp.matmul
    original_from_owned = Batch._from_owned_array
    allocations: list[tuple[tuple[int, ...], str, object]] = []
    transfers: list[tuple[object, object | None, object]] = []
    multiplications: list[tuple[object, object, object]] = []
    adoptions: list[object] = []

    def guarded_asarray(
        value: object, *args: object, **kwargs: object
    ) -> np.ndarray[Any, Any]:
        assert not isinstance(value, jax.Array)
        return original_asarray(value, *args, **kwargs)

    def counted_allocation(shape: tuple[int, ...], dtype: str) -> tuple[object, object]:
        array, token = original_new_owned(shape, dtype)
        allocations.append((tuple(shape), dtype, array))
        return array, token

    def tracked_device_put(value: object, device: object | None = None) -> object:
        dense = original_device_put(value, device=device)
        transfers.append((value, device, dense))
        return dense

    def tracked_matmul(left: object, right: object) -> object:
        result = original_matmul(left, right)
        multiplications.append((left, right, result))
        return result

    def tracked_adoption(
        array: object,
        *,
        backend: str,
        token: object,
        metadata: dict[str, object],
    ) -> Batch:
        adoptions.append(array)
        return original_from_owned(
            array,
            backend=backend,
            token=token,
            metadata=metadata,
        )

    monkeypatch.setattr(np, "asarray", guarded_asarray)
    monkeypatch.setattr(Batch, "_new_owned_numpy", counted_allocation)
    monkeypatch.setattr(jax, "device_put", tracked_device_put)
    monkeypatch.setattr(jnp, "matmul", tracked_matmul)
    monkeypatch.setattr(Batch, "_from_owned_array", staticmethod(tracked_adoption))

    output = plan.execute(
        {
            "table": Batch.from_pyarrow(table),
            "weights": weights,
        }
    ).outputs["output"]

    assert [(shape, dtype) for shape, dtype, _array in allocations] == [
        ((3, 2), "float32")
    ]
    assert allocations[0][2].flags.c_contiguous
    assert len(transfers) == 1
    assert transfers[0][0] is allocations[0][2]
    assert transfers[0][1] is weights_payload.device
    assert len(multiplications) == 1
    assert multiplications[0][0] is transfers[0][2]
    assert multiplications[0][1] is weights_payload
    assert adoptions == [multiplications[0][2]]
    assert output.array is multiplications[0][2]
    assert output.array.device == weights_payload.device
    assert output.array.tolist() == [[9.0], [12.0], [15.0]]
    assert table.to_pydict() == expected_table
    assert weights.array is weights_payload
    assert weights_payload.tolist() == expected_weights


def test_jax_table_matmul_accepts_float64_when_x64_is_enabled() -> None:
    script = """
import jax
import jax.numpy as jnp
import pyarrow as pa

from calc_flow import Batch, PipelineBuilder, Runtime, register_jax

assert jax.config.x64_enabled
runtime = Runtime()
register_jax(runtime)
plan = (
    PipelineBuilder("jax-table-matmul-x64")
    .table_matmul("multiply", backend="jax", columns=("value",))
    .compile(runtime)
)
weights = Batch.from_array(
    jnp.asarray([[2.0]], dtype=jnp.float64),
    backend="jax",
)
output = plan.execute(
    {
        "table": Batch.from_pyarrow(
            pa.table({"value": pa.array([1.5, 2.5], type=pa.float64())})
        ),
        "weights": weights,
    }
).outputs["output"]
assert output.array.dtype == jnp.dtype(jnp.float64)
assert output.array.tolist() == [[3.0], [5.0]]
assert output.array.device == weights.array.device
"""
    completed = subprocess.run(
        [sys.executable, "-c", script],
        check=False,
        capture_output=True,
        env={
            **os.environ,
            "JAX_ENABLE_X64": "true",
            "JAX_PLATFORMS": "cpu",
        },
        text=True,
    )

    assert completed.returncode == 0, completed.stderr


def test_missing_python_provider_fails_during_compile() -> None:
    with pytest.raises(Exception, match="provider numpy:expression@1 is unavailable"):
        _external("arrays", "numpy", "x + 1").compile(Runtime())


@pytest.mark.parametrize(
    "expression",
    [
        "__import__('os').system('whoami')",
        "x.real",
        "(lambda value: value)(x)",
        "[value for value in x]",
        "unknown",
        "unknown(x)",
        "sum(x, axis=0)",
        "sum(x, **{})",
        "True",
        "1e309",
    ],
)
def test_array_expression_rejects_unsafe_syntax(expression: str) -> None:
    with pytest.raises(ValueError, match="array expression"):
        _external("unsafe", "numpy", expression).compile()


def test_array_expression_enforces_node_and_depth_limits() -> None:
    too_many_nodes = "+".join("x" for _ in range(200))
    too_deep = "-" * 40 + "x"

    with pytest.raises(ValueError, match="node limit"):
        _external("large", "numpy", too_many_nodes).compile()
    with pytest.raises(ValueError, match="depth limit"):
        _external("deep", "numpy", too_deep).compile()


@pytest.mark.parametrize(
    ("expression", "message"),
    [
        ("x" * 4097, "expression length limit is 4096 characters"),
        (str(2**64), "integer constant magnitude limit is 9223372036854775807"),
        ("x ** x", "power exponent must be a finite numeric literal"),
        ("x ** 65", "power exponent magnitude limit is 64"),
    ],
)
def test_array_expression_enforces_literal_work_limits(
    expression: str, message: str
) -> None:
    with pytest.raises(ValueError, match=message):
        _external("bounded", "numpy", expression).compile()


@pytest.mark.parametrize(
    ("expression", "expected"),
    [
        ("x ** 2", [4.0, 16.0]),
        ("x ** -2", [0.25, 0.0625]),
        ("x ** 0.5", [2**0.5, 2.0]),
    ],
)
def test_array_expression_supports_bounded_literal_exponents(
    expression: str, expected: list[float]
) -> None:
    runtime = Runtime()
    register_numpy(runtime)
    plan = _external("bounded", "numpy", expression).compile(runtime)

    output = plan.execute(
        {"input": Batch.from_array(np.array([2.0, 4.0]), backend="numpy")}
    ).outputs["output"]

    assert output.array.tolist() == pytest.approx(expected)


@pytest.mark.parametrize(
    ("expression", "message"),
    [
        ("sum()", "sum expects 1 argument"),
        ("transpose(x, x)", "transpose expects 1 argument"),
        ("reshape(x)", "reshape expects 2 arguments"),
        ("reshape(x, (2, True))", "reshape dimensions must be integers"),
        ("reshape(x, (2, -1, -1))", "at most one -1"),
        ("reshape(x, (2, -2))", "reshape dimensions must be non-negative or -1"),
        ("reshape(x, (1000001,))", "reshape dimension limit"),
    ],
)
def test_array_expression_validates_function_arguments(
    expression: str, message: str
) -> None:
    with pytest.raises(ValueError, match=message):
        _external("invalid", "numpy", expression).compile()


@pytest.mark.parametrize(
    ("source_shape", "expression", "message"),
    [
        ((1001, 1000), "reshape(x, (-1,))", "reshape dimension limit is 1000000"),
        (
            (1001, 10000),
            "reshape(x, (-1, 11))",
            "reshape output limit is 10000000 elements",
        ),
    ],
)
def test_numpy_provider_rejects_inferred_reshape_limits(
    source_shape: tuple[int, ...], expression: str, message: str
) -> None:
    runtime = Runtime()
    register_numpy(runtime)
    plan = _external("reshape_limit", "numpy", expression).compile(runtime)
    source = np.zeros(source_shape, dtype=np.uint8)

    with pytest.raises(ProviderError, match=message):
        plan.execute({"input": Batch.from_array(source, backend="numpy")})


@pytest.mark.parametrize(
    ("source_shape", "expression", "message"),
    [
        ((1001, 1000), "reshape(x, (-1,))", "reshape dimension limit is 1000000"),
        (
            (1001, 10000),
            "reshape(x, (-1, 11))",
            "reshape output limit is 10000000 elements",
        ),
    ],
)
def test_jax_provider_rejects_inferred_reshape_limits(
    source_shape: tuple[int, ...], expression: str, message: str
) -> None:
    jnp = pytest.importorskip("jax.numpy")
    runtime = Runtime()
    register_jax(runtime)
    plan = _external("reshape_limit", "jax", expression).compile(runtime)
    source = jnp.zeros(source_shape, dtype=jnp.uint8)

    with pytest.raises(ProviderError, match=message):
        plan.execute({"input": Batch.from_array(source, backend="jax")})


@pytest.mark.parametrize("backend", ["numpy", "jax"])
@pytest.mark.parametrize(
    ("source_shape", "expression", "expected_shape"),
    [
        ((1000, 1000), "reshape(x, (-1,))", (1_000_000,)),
        ((10_000_000,), "reshape(x, (-1, 10))", (1_000_000, 10)),
    ],
)
def test_inferred_reshape_accepts_exact_dimension_and_element_limits(
    backend: str,
    source_shape: tuple[int, ...],
    expression: str,
    expected_shape: tuple[int, ...],
) -> None:
    namespace: Any = np if backend == "numpy" else pytest.importorskip("jax.numpy")
    runtime = Runtime()
    {"numpy": register_numpy, "jax": register_jax}[backend](runtime)
    plan = _external("reshape_boundary", backend, expression).compile(runtime)
    source = namespace.zeros(source_shape, dtype=namespace.uint8)

    output = plan.execute({"input": Batch.from_array(source, backend=backend)}).outputs[
        "output"
    ]

    assert output.array.shape == expected_shape


@pytest.mark.parametrize("backend", ["numpy", "jax"])
def test_reshape_to_empty_shape_preserves_single_element(backend: str) -> None:
    namespace: Any = np if backend == "numpy" else pytest.importorskip("jax.numpy")
    runtime = Runtime()
    {"numpy": register_numpy, "jax": register_jax}[backend](runtime)
    plan = _external("reshape_scalar", backend, "reshape(x, ())").compile(runtime)
    source = namespace.asarray([5.0])

    output = plan.execute({"input": Batch.from_array(source, backend=backend)}).outputs[
        "output"
    ]

    assert output.array.shape == ()
    assert float(output.array) == 5.0


@pytest.mark.parametrize(
    ("backend", "message"),
    [
        ("numpy", "cannot reshape array of size 0"),
        ("jax", "integer modulo by zero"),
    ],
)
def test_reshape_with_zero_dimension_and_inferred_axis_is_rejected(
    backend: str, message: str
) -> None:
    namespace: Any = np if backend == "numpy" else pytest.importorskip("jax.numpy")
    runtime = Runtime()
    {"numpy": register_numpy, "jax": register_jax}[backend](runtime)
    plan = _external("reshape_zero", backend, "reshape(x, (0, -1))").compile(runtime)
    source = namespace.zeros(0)

    with pytest.raises(ProviderError, match=message):
        plan.execute({"input": Batch.from_array(source, backend=backend)})


def test_numpy_provider_supports_bounded_array_operations() -> None:
    runtime = Runtime()
    register_numpy(runtime)
    source = np.arange(6).reshape(2, 3)

    reshaped = _external("reshape", "numpy", "reshape(transpose(x), (6,))").compile(
        runtime
    )
    output = reshaped.execute(
        {"input": Batch.from_array(source, backend="numpy")}
    ).outputs["output"]

    assert output.array.tolist() == [0, 3, 1, 4, 2, 5]
    assert output.num_rows == 6
    assert not output.array.flags.writeable


@pytest.mark.parametrize(
    "dtype",
    [
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
    ],
)
def test_numpy_batches_accept_array_api_numeric_and_bool_dtypes(
    dtype: np.dtype[Any],
) -> None:
    batch = Batch.from_array(np.array([0, 1], dtype=dtype), backend="numpy")
    runtime = Runtime()
    register_numpy(runtime)
    output = (
        _external("dtype", "numpy", "x")
        .compile(runtime)
        .execute({"input": batch})
        .outputs["output"]
    )

    assert output.array.dtype == np.dtype(dtype)


@pytest.mark.parametrize(
    ("alias", "supported"),
    [
        pytest.param(
            np.longlong,
            np.int64,
            marks=pytest.mark.skipif(
                np.dtype(np.longlong) != np.dtype(np.int64),
                reason="NumPy longlong does not describe int64 on this platform",
            ),
        ),
        pytest.param(
            np.ulonglong,
            np.uint64,
            marks=pytest.mark.skipif(
                np.dtype(np.ulonglong) != np.dtype(np.uint64),
                reason="NumPy ulonglong does not describe uint64 on this platform",
            ),
        ),
    ],
)
def test_numpy_batches_accept_semantically_equivalent_dtype_aliases(
    alias: type[np.generic], supported: type[np.generic]
) -> None:
    batch = Batch.from_array(np.array([0, 1], dtype=alias), backend="numpy")
    runtime = Runtime()
    register_numpy(runtime)
    output = (
        _external("dtype_alias", "numpy", "x")
        .compile(runtime)
        .execute({"input": batch})
        .outputs["output"]
    )

    assert output.array.dtype == np.dtype(supported)


def test_numpy_provider_accepts_bool_scalar_results() -> None:
    runtime = Runtime()
    register_numpy(runtime)
    plan = _external("bool_scalar", "numpy", "max(x)").compile(runtime)

    output = plan.execute(
        {"input": Batch.from_array(np.array([False, True]), backend="numpy")}
    ).outputs["output"]

    assert output.array.dtype == np.dtype(np.bool_)
    assert output.array.item() is True


@pytest.mark.parametrize(
    "value",
    [
        np.array([1], dtype=object),
        np.array([1], dtype=np.float16),
        pytest.param(
            np.array([1], dtype=np.longdouble),
            marks=pytest.mark.skipif(
                np.dtype(np.longdouble) == np.dtype(np.float64),
                reason="NumPy longdouble aliases float64 on this platform",
            ),
        ),
        pytest.param(
            np.array([1], dtype=np.clongdouble),
            marks=pytest.mark.skipif(
                np.dtype(np.clongdouble) == np.dtype(np.complex128),
                reason="NumPy clongdouble aliases complex128 on this platform",
            ),
        ),
        np.array(["1"], dtype="U1"),
        np.array([b"1"], dtype="S1"),
        np.array([1], dtype="datetime64[D]"),
        np.array([1], dtype="timedelta64[D]"),
        np.array([(1,)], dtype=[("value", np.int64)]),
        np.array([1], dtype=np.dtype(np.int64).newbyteorder("S")),
    ],
)
def test_numpy_batches_reject_non_array_api_dtypes(value: np.ndarray[Any, Any]) -> None:
    with pytest.raises(ValueError, match="NumPy Array API dtype"):
        Batch.from_array(value, backend="numpy")


def test_numpy_object_input_is_rejected_before_dispatching_magic_methods() -> None:
    calls = 0

    class Magic:
        def __add__(self, _other: object) -> int:
            nonlocal calls
            calls += 1
            return 1

    source = np.array([Magic()], dtype=object)

    with pytest.raises(ValueError, match="NumPy Array API dtype"):
        Batch.from_array(source, backend="numpy")
    assert calls == 0


def test_numpy_intermediate_is_rejected_before_dispatching_magic_methods(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls = 0

    class Magic:
        def __add__(self, _other: object) -> int:
            nonlocal calls
            calls += 1
            return 1

    monkeypatch.setattr(np, "sum", lambda _value: np.array([Magic()], dtype=object))
    runtime = Runtime()
    register_numpy(runtime)
    plan = _external("object_output", "numpy", "sum(x) + 1").compile(runtime)

    with pytest.raises(ProviderError, match="NumPy Array API dtype"):
        plan.execute({"input": Batch.from_array(np.array([1]), backend="numpy")})
    assert calls == 0


def test_numpy_intermediate_rejects_unbounded_python_integer(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(np, "sum", lambda _value: 2**64)
    runtime = Runtime()
    register_numpy(runtime)
    plan = _external("integer_output", "numpy", "sum(x) + 1").compile(runtime)

    with pytest.raises(ProviderError, match="integer constant magnitude limit"):
        plan.execute({"input": Batch.from_array(np.array([1]), backend="numpy")})


def test_numpy_provider_preserves_metadata() -> None:
    runtime = Runtime()
    register_numpy(runtime)
    plan = _external("metadata", "numpy", "mean(x)").compile(runtime)

    output = plan.execute(
        {
            "input": Batch.from_array(
                np.array([1.0, 3.0]),
                backend="numpy",
                metadata={"request": "demo"},
            )
        }
    ).outputs["output"]

    assert output.metadata == {"request": "demo"}
    assert output.num_rows == 1
    assert output.array.item() == 2.0


def test_provider_rejects_backend_mismatches_and_table_batches() -> None:
    runtime = Runtime()
    register_numpy(runtime)
    plan = _external("backend", "numpy", "x + 1").compile(runtime)

    with pytest.raises(ProviderError, match="requires backend numpy"):
        plan.execute({"input": Batch._from_external(object(), "other", 1, {})})
    with pytest.raises(Exception, match="expects a Array batch"):
        plan.execute({"input": Batch.from_pyarrow(pa.table({"value": [1]}))})


def test_custom_array_udf_references_are_rejected_explicitly() -> None:
    with pytest.raises(ValueError, match="custom array UDFs are unavailable"):
        _external(
            "udf",
            "numpy",
            "custom(x)",
            options={"udfs": [{"name": "custom", "version": "1"}]},
        ).compile()


def test_callback_failure_leaves_plan_reusable() -> None:
    class FailOnce:
        calls = 0

        def validate(self, options: dict[str, object]) -> None:
            if options != {"increment": 1}:
                raise ValueError("unexpected options")

        def __call__(self, batch: Batch, options: dict[str, object]) -> Batch:
            self.calls += 1
            if self.calls == 1:
                raise RuntimeError("intentional callback failure")
            return Batch.from_array(
                batch.array + options["increment"],
                backend="numpy",
                metadata=batch.metadata,
            )

    runtime = Runtime()
    callback = FailOnce()
    runtime.register_provider("test", "increment", "1", callback)
    plan = (
        PipelineBuilder("reuse")
        .external("calc", "test", "increment", "1", {"increment": 1})
        .compile(runtime)
    )
    batch = Batch.from_array(np.array([1]), backend="numpy")

    with pytest.raises(ProviderError, match="intentional callback failure"):
        plan.execute({"input": batch})

    assert plan.execute({"input": batch}).outputs["output"].array.tolist() == [2]
    assert callback.calls == 2


def test_provider_registration_cycles_are_collected() -> None:
    class Callback:
        runtime: Runtime | None = None

        def validate(self, _options: dict[str, object]) -> None:
            pass

        def __call__(self, batch: Batch, _options: dict[str, object]) -> Batch:
            return batch

    runtime = Runtime()
    callback = Callback()
    callback.runtime = runtime
    callback_ref = weakref.ref(callback)
    runtime.register_provider("cycle", "identity", "1", callback)

    del callback
    del runtime
    gc.collect()

    assert callback_ref() is None


def test_jax_provider_retains_jax_arrays() -> None:
    jax = pytest.importorskip("jax")
    jnp = pytest.importorskip("jax.numpy")
    runtime = Runtime()
    register_jax(runtime)
    source = jnp.array([1.0, 2.0])
    plan = _external("jax", "jax", "x ** 2").compile(runtime)

    output = (
        plan.execute({"input": Batch.from_array(source, backend="jax")})
        .outputs["output"]
        .array
    )

    assert isinstance(output, jax.Array)
    assert output.tolist() == [1.0, 4.0]


def test_jax_provider_keeps_constant_results_on_jax() -> None:
    jax = pytest.importorskip("jax")
    jnp = pytest.importorskip("jax.numpy")
    runtime = Runtime()
    register_jax(runtime)
    plan = _external("jax_constant", "jax", "2").compile(runtime)

    output = (
        plan.execute({"input": Batch.from_array(jnp.array([1.0]), backend="jax")})
        .outputs["output"]
        .array
    )

    assert isinstance(output, jax.Array)
    assert output.item() == 2


def test_jax_provider_bounds_nested_integer_power_intermediates() -> None:
    pytest.importorskip("jax")
    jnp = pytest.importorskip("jax.numpy")
    runtime = Runtime()
    register_jax(runtime)
    plan = _external("jax_bounded", "jax", "(2 ** 64) ** 2").compile(runtime)

    with pytest.raises(ProviderError, match="integer constant magnitude limit"):
        plan.execute({"input": Batch.from_array(jnp.array([1.0]), backend="jax")})


def test_provider_registration_validates_public_arguments() -> None:
    runtime = Runtime()

    with pytest.raises(TypeError, match="callable"):
        runtime.register_provider("test", "identity", "1", object())
    with pytest.raises(Exception, match="duplicate provider"):
        runtime.register_provider("test", "identity", "1", lambda *_: None)
        runtime.register_provider("test", "identity", "1", lambda *_: None)


def test_registration_does_not_mutate_options() -> None:
    class Callback:
        def validate(self, options: dict[str, Any]) -> None:
            options["mutated"] = True

        def __call__(self, batch: Batch, _options: dict[str, Any]) -> Batch:
            return batch

    runtime = Runtime()
    runtime.register_provider("copy", "identity", "1", Callback())
    options: dict[str, object] = {"value": 1}

    PipelineBuilder("copy").external("calc", "copy", "identity", "1", options).compile(
        runtime
    )

    assert options == {"value": 1}


def test_external_provider_identity_and_nested_options_change_fingerprints() -> None:
    def identity(batch: Batch, _options: dict[str, object]) -> Batch:
        return batch

    runtime = Runtime()
    identities = [
        ("provider-a", "operation", "1"),
        ("provider-b", "operation", "1"),
        ("provider-a", "other", "1"),
        ("provider-a", "operation", "2"),
    ]
    for provider, name, version in identities:
        runtime.register_provider(provider, name, version, identity)

    configurations = [
        (*identities[0], {"nested": {"items": [1, {"value": "a"}]}}),
        (*identities[1], {"nested": {"items": [1, {"value": "a"}]}}),
        (*identities[2], {"nested": {"items": [1, {"value": "a"}]}}),
        (*identities[3], {"nested": {"items": [1, {"value": "a"}]}}),
        (*identities[0], {"nested": {"items": [1, {"value": "b"}]}}),
    ]
    batch = Batch.from_array(np.array([1]), backend="numpy")
    fingerprints = {
        PipelineBuilder("fingerprints")
        .external("calc", provider, name, version, options)
        .compile(runtime)
        .execute({"input": batch})
        .metadata["pipeline_fingerprint"]
        for provider, name, version, options in configurations
    }

    assert len(fingerprints) == len(configurations)
