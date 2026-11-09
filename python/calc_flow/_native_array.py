from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING

from calc_flow import _native
from calc_flow._compat import dataclass

if TYPE_CHECKING:
    import numpy as np
    import pyarrow as pa

NativeArray = _native.NativeArray


def _integer(value: int, field: str) -> int:
    if type(value) is not int:
        raise TypeError(f"{field} must be an integer")
    if value < 0:
        raise _native.ConfigError(f"{field}: must be nonnegative")
    return value


@dataclass(frozen=True, slots=True)
class ArrayExecutionOptions:
    """Per-call output element and allocation limits, excluding borrowed input."""

    max_elements: int = 10_000_000
    max_allocation_bytes: int = 256 * 1024 * 1024

    def __post_init__(self) -> None:
        _integer(self.max_elements, "max_elements")
        _integer(self.max_allocation_bytes, "max_allocation_bytes")


def from_numpy(values: np.ndarray) -> NativeArray:
    """Copy a rank-0/1/2 bool/float32/float64 ndarray into immutable storage."""
    import numpy as np

    if type(values) is not np.ndarray:
        raise TypeError("values must be an exact NumPy ndarray")
    if values.dtype not in (np.dtype("bool"), np.dtype("float32"), np.dtype("float64")):
        raise _native.ConfigError(
            "input.dtype: expected native-endian bool, float32 or float64"
        )
    if values.ndim > 2:
        raise _native.ConfigError("input.shape: expected rank at most two")
    return _native._native_array_from_bytes(
        values.tobytes(order="C"), str(values.dtype), list(values.shape)
    )


def from_arrow(
    values: pa.Array, *, shape: tuple[int, ...] | None = None
) -> NativeArray:
    """Share a read-only Arrow float or Boolean array, preserving masks and offsets."""
    import pyarrow as pa

    if not isinstance(values, pa.Array):
        raise TypeError("values must be a PyArrow Array")
    dimensions = (len(values),) if shape is None else tuple(shape)
    for dimension in dimensions:
        _integer(dimension, "input.shape")
    return _native._native_array_from_arrow(values, list(dimensions))


def _to_numpy(array: NativeArray, fill_value: float | bool | None) -> np.ndarray:
    import numpy as np

    arrow = array.to_arrow()
    if arrow.null_count and fill_value is None:
        raise ValueError("null values require an explicit numeric fill_value")
    if fill_value is not None and (
        (isinstance(fill_value, bool) and array.dtype != "bool")
        or not isinstance(fill_value, (int, float))
    ):
        raise TypeError("fill_value must be numeric")
    values = arrow.to_numpy(zero_copy_only=False)
    if arrow.null_count:
        values = values.copy()
        values[arrow.is_null().to_numpy(zero_copy_only=False)] = fill_value
    values = np.asarray(values, dtype=array.dtype)
    return np.frombuffer(values.tobytes(order="C"), dtype=array.dtype).reshape(
        array.shape
    )


def _execution(execution: ArrayExecutionOptions | None) -> ArrayExecutionOptions:
    if execution is None:
        return ArrayExecutionOptions()
    if not isinstance(execution, ArrayExecutionOptions):
        raise TypeError("execution must be ArrayExecutionOptions")
    return execution


def _reduce(
    array: NativeArray,
    operation: str,
    options: tuple[bool, int | None, bool],
    execution: ArrayExecutionOptions | None,
) -> NativeArray:
    skip_nulls, axis, keepdims = options
    if type(keepdims) is not bool:
        raise TypeError("keepdims must be a bool")
    if axis is not None:
        _integer(axis, "options.axis")
    if type(skip_nulls) is not bool:
        raise TypeError("skip_nulls must be a bool")
    limits = _execution(execution)
    return _native._native_array_reduce(
        array,
        operation,
        (skip_nulls, axis, keepdims),
        limits.max_elements,
        limits.max_allocation_bytes,
    )


def sum(
    array: NativeArray,
    *,
    skip_nulls: bool = False,
    axis: int | None = None,
    keepdims: bool = False,
    execution: ArrayExecutionOptions | None = None,
) -> NativeArray:
    """Return a float64 sum along the selected axis; nulls propagate."""
    return _reduce(array, "sum", (skip_nulls, axis, keepdims), execution)


def mean(
    array: NativeArray,
    *,
    skip_nulls: bool = False,
    axis: int | None = None,
    keepdims: bool = False,
    execution: ArrayExecutionOptions | None = None,
) -> NativeArray:
    """Return a float64 mean along the selected axis; nulls propagate."""
    return _reduce(array, "mean", (skip_nulls, axis, keepdims), execution)


def min(
    array: NativeArray,
    *,
    skip_nulls: bool = False,
    axis: int | None = None,
    keepdims: bool = False,
    execution: ArrayExecutionOptions | None = None,
) -> NativeArray:
    """Return the native minimum along the selected axis, preserving the input dtype."""
    return _reduce(array, "min", (skip_nulls, axis, keepdims), execution)


def max(
    array: NativeArray,
    *,
    skip_nulls: bool = False,
    axis: int | None = None,
    keepdims: bool = False,
    execution: ArrayExecutionOptions | None = None,
) -> NativeArray:
    """Return the native maximum along the selected axis, preserving the input dtype."""
    return _reduce(array, "max", (skip_nulls, axis, keepdims), execution)


def ewma(
    array: NativeArray,
    *,
    span: int,
    min_periods: int = 1,
    execution: ArrayExecutionOptions | None = None,
) -> NativeArray:
    """Return float64 EWMA with the native table missing-value/warmup contract."""
    limits = _execution(execution)
    return _native._native_array_ewma(
        array,
        _integer(span, "options.span"),
        _integer(min_periods, "options.min_periods"),
        limits.max_elements,
        limits.max_allocation_bytes,
    )


def rolling_mean(
    array: NativeArray,
    *,
    window: int,
    min_periods: int = 1,
    numerical_profile: str = "stable_v1",
    execution: ArrayExecutionOptions | None = None,
) -> NativeArray:
    """Return aligned float64 row-window means with the native table profile."""
    limits = _execution(execution)
    return _native._native_array_rolling_mean(
        array,
        _integer(window, "options.window"),
        _integer(min_periods, "options.min_periods"),
        numerical_profile,
        limits.max_elements,
        limits.max_allocation_bytes,
    )


def _indicator_period(period: int) -> int:
    if type(period) is not int:
        raise TypeError("options.period must be an integer")
    if not 1 <= period <= 2**64 - 1:
        raise _native.ConfigError("options.period: must be a positive u64")
    return period


def _indicator_owner(array: NativeArray, name: str) -> None:
    if not isinstance(array, NativeArray):
        raise TypeError(f"{name} must be a NativeArray")


def rsi(
    array: NativeArray,
    *,
    period: int,
    execution: ArrayExecutionOptions | None = None,
) -> NativeArray:
    """Return aligned Wilder RSI; missing prices reset its warm-up segment."""
    count = _indicator_period(period)
    _indicator_owner(array, "array")
    limits = _execution(execution)
    return _native._native_array_rsi(
        array, count, limits.max_elements, limits.max_allocation_bytes
    )


def atr(
    high: NativeArray,
    low: NativeArray,
    close: NativeArray,
    *,
    period: int,
    execution: ArrayExecutionOptions | None = None,
) -> NativeArray:
    """Return aligned Wilder ATR for equal-length floating high/low/close vectors."""
    count = _indicator_period(period)
    _indicator_owner(high, "high")
    _indicator_owner(low, "low")
    _indicator_owner(close, "close")
    limits = _execution(execution)
    return _native._native_array_atr(
        high, low, close, count, limits.max_elements, limits.max_allocation_bytes
    )


def _unary(
    array: NativeArray, operation: str, execution: ArrayExecutionOptions | None
) -> NativeArray:
    limits = _execution(execution)
    return _native._native_array_unary(
        array, operation, limits.max_elements, limits.max_allocation_bytes
    )


def _binary(
    left: NativeArray,
    right: NativeArray,
    operation: str,
    execution: ArrayExecutionOptions | None,
) -> NativeArray:
    limits = _execution(execution)
    return _native._native_array_binary(
        left, right, operation, limits.max_elements, limits.max_allocation_bytes
    )


def neg(
    array: NativeArray, *, execution: ArrayExecutionOptions | None = None
) -> NativeArray:
    """Negate floating values."""
    return _unary(array, "neg", execution)


def abs(
    array: NativeArray, *, execution: ArrayExecutionOptions | None = None
) -> NativeArray:
    """Return floating absolute values."""
    return _unary(array, "abs", execution)


def sqrt(
    array: NativeArray, *, execution: ArrayExecutionOptions | None = None
) -> NativeArray:
    """Return IEEE square roots."""
    return _unary(array, "sqrt", execution)


def log(
    array: NativeArray, *, execution: ArrayExecutionOptions | None = None
) -> NativeArray:
    """Return IEEE natural logarithms."""
    return _unary(array, "log", execution)


def exp(
    array: NativeArray, *, execution: ArrayExecutionOptions | None = None
) -> NativeArray:
    """Return IEEE exponentials."""
    return _unary(array, "exp", execution)


def logical_not(
    array: NativeArray, *, execution: ArrayExecutionOptions | None = None
) -> NativeArray:
    """Negate Boolean values, preserving nulls."""
    return _unary(array, "logical_not", execution)


def add(
    left: NativeArray,
    right: NativeArray,
    *,
    execution: ArrayExecutionOptions | None = None,
) -> NativeArray:
    """Add floating operands."""
    return _binary(left, right, "add", execution)


def subtract(
    left: NativeArray,
    right: NativeArray,
    *,
    execution: ArrayExecutionOptions | None = None,
) -> NativeArray:
    """Subtract floating operands."""
    return _binary(left, right, "subtract", execution)


def multiply(
    left: NativeArray,
    right: NativeArray,
    *,
    execution: ArrayExecutionOptions | None = None,
) -> NativeArray:
    """Multiply floating operands."""
    return _binary(left, right, "multiply", execution)


def divide(
    left: NativeArray,
    right: NativeArray,
    *,
    execution: ArrayExecutionOptions | None = None,
) -> NativeArray:
    """Divide floating operands using IEEE rules."""
    return _binary(left, right, "divide", execution)


def pow(
    left: NativeArray,
    right: NativeArray,
    *,
    execution: ArrayExecutionOptions | None = None,
) -> NativeArray:
    """Raise values to a finite scalar exponent with magnitude at most 64."""
    return _binary(left, right, "pow", execution)


def equal(
    left: NativeArray,
    right: NativeArray,
    *,
    execution: ArrayExecutionOptions | None = None,
) -> NativeArray:
    """Compare equal floating or Boolean operands."""
    return _binary(left, right, "equal", execution)


def not_equal(
    left: NativeArray,
    right: NativeArray,
    *,
    execution: ArrayExecutionOptions | None = None,
) -> NativeArray:
    """Compare unequal floating or Boolean operands."""
    return _binary(left, right, "not_equal", execution)


def less(
    left: NativeArray,
    right: NativeArray,
    *,
    execution: ArrayExecutionOptions | None = None,
) -> NativeArray:
    """Compare floating operands with less-than."""
    return _binary(left, right, "less", execution)


def less_equal(
    left: NativeArray,
    right: NativeArray,
    *,
    execution: ArrayExecutionOptions | None = None,
) -> NativeArray:
    """Compare floating operands with less-than-or-equal."""
    return _binary(left, right, "less_equal", execution)


def greater(
    left: NativeArray,
    right: NativeArray,
    *,
    execution: ArrayExecutionOptions | None = None,
) -> NativeArray:
    """Compare floating operands with greater-than."""
    return _binary(left, right, "greater", execution)


def greater_equal(
    left: NativeArray,
    right: NativeArray,
    *,
    execution: ArrayExecutionOptions | None = None,
) -> NativeArray:
    """Compare floating operands with greater-than-or-equal."""
    return _binary(left, right, "greater_equal", execution)


def logical_and(
    left: NativeArray,
    right: NativeArray,
    *,
    execution: ArrayExecutionOptions | None = None,
) -> NativeArray:
    """Return Kleene Boolean AND."""
    return _binary(left, right, "logical_and", execution)


def logical_or(
    left: NativeArray,
    right: NativeArray,
    *,
    execution: ArrayExecutionOptions | None = None,
) -> NativeArray:
    """Return Kleene Boolean OR."""
    return _binary(left, right, "logical_or", execution)


def logical_xor(
    left: NativeArray,
    right: NativeArray,
    *,
    execution: ArrayExecutionOptions | None = None,
) -> NativeArray:
    """Return null-propagating Boolean XOR."""
    return _binary(left, right, "logical_xor", execution)


def where_(
    condition: NativeArray,
    if_true: NativeArray,
    if_false: NativeArray,
    *,
    execution: ArrayExecutionOptions | None = None,
) -> NativeArray:
    """Select a branch by Boolean condition; null conditions produce nulls."""
    limits = _execution(execution)
    return _native._native_array_where(
        condition, if_true, if_false, limits.max_elements, limits.max_allocation_bytes
    )


def matmul(
    left: NativeArray,
    right: NativeArray,
    *,
    execution: ArrayExecutionOptions | None = None,
) -> NativeArray:
    """Multiply rank-2 non-null floating arrays in deterministic inner order."""
    limits = _execution(execution)
    return _native._native_array_matmul(
        left, right, limits.max_elements, limits.max_allocation_bytes
    )


def reshape(
    array: NativeArray,
    shape: Sequence[int],
    *,
    execution: ArrayExecutionOptions | None = None,
) -> NativeArray:
    """Return an owned packed shape alias, charging only the new descriptor."""
    dimensions = tuple(shape)
    for dimension in dimensions:
        _integer(dimension, "input.shape")
    limits = _execution(execution)
    return _native._native_array_reshape(
        array, list(dimensions), limits.max_elements, limits.max_allocation_bytes
    )


def transpose(
    array: NativeArray, *, execution: ArrayExecutionOptions | None = None
) -> NativeArray:
    """Swap matrix axes into packed row order; scalars and vectors are aliases."""
    limits = _execution(execution)
    return _native._native_array_transpose(
        array, limits.max_elements, limits.max_allocation_bytes
    )
