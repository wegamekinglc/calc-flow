from __future__ import annotations

import ast
import copy
import gc
import weakref
from collections.abc import Callable
from dataclasses import dataclass
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
