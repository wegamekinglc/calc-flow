from __future__ import annotations

import builtins

import pyarrow as pa
import pytest

import calc_flow as cf
import calc_flow.array as cfa


def _arguments(kind: str) -> tuple[cfa.NativeArray, ...]:
    values = cfa.from_arrow(pa.array([1.0, 2.0]))
    return (values,) if kind == "rsi" else (values, values, values)


@pytest.mark.parametrize("kind", ["rsi", "atr"])
def test_native_indicators_require_an_explicit_period(kind: str) -> None:
    with pytest.raises(TypeError, match="period"):
        getattr(cfa, kind)(*_arguments(kind))


@pytest.mark.parametrize("kind", ["rsi", "atr"])
@pytest.mark.parametrize("period", [True, 1.5, "2"])
def test_native_indicators_reject_non_integer_periods_before_binding(
    kind: str, period: object
) -> None:
    with pytest.raises(TypeError, match="period.*integer"):
        getattr(cfa, kind)(*_arguments(kind), period=period)


@pytest.mark.parametrize("kind", ["rsi", "atr"])
@pytest.mark.parametrize("period", [0, -1, 2**64])
def test_native_indicators_reject_periods_outside_positive_u64_before_binding(
    kind: str, period: int
) -> None:
    with pytest.raises(cf.ConfigError, match="options.period.*positive u64"):
        getattr(cfa, kind)(*_arguments(kind), period=period)


@pytest.mark.parametrize(
    "kind,operand", [("rsi", 0), ("atr", 0), ("atr", 1), ("atr", 2)]
)
def test_native_indicators_require_immutable_native_owners(
    kind: str, operand: int
) -> None:
    arguments = list(_arguments(kind))
    arguments[operand] = [1.0, 2.0]
    with pytest.raises(TypeError, match="NativeArray"):
        getattr(cfa, kind)(*arguments, period=2)


@pytest.mark.parametrize("kind", ["rsi", "atr"])
def test_native_indicators_validate_execution_limits_before_binding(kind: str) -> None:
    with pytest.raises(TypeError, match="execution.*ArrayExecutionOptions"):
        getattr(cfa, kind)(*_arguments(kind), period=2, execution=object())


def test_real_native_indicator_warmup_and_values() -> None:
    close = cfa.from_arrow(pa.array([10.0, 11.0, 12.0, 11.0, 11.0]))
    high = cfa.from_arrow(pa.array([11.0, 12.0, 13.0, 12.0, 12.0]))
    low = cfa.from_arrow(pa.array([9.0, 10.0, 11.0, 10.0, 10.0]))
    rsi = cfa.rsi(close, period=2)
    atr = cfa.atr(high, low, close, period=2)
    assert rsi.dtype == atr.dtype == "float64"
    assert rsi.shape == atr.shape == (5,)
    assert rsi.to_arrow().to_pylist() == [None, None, 100.0, 50.0, 50.0]
    assert atr.to_arrow().to_pylist() == [None, None, 2.0, 2.0, 2.0]
    assert close.to_arrow().to_pylist() == [10.0, 11.0, 12.0, 11.0, 11.0]
    subnormal = cfa.from_arrow(
        pa.array([0.0, float.fromhex("0x0.0000000000001p-1022"), 0.0])
    )
    assert cfa.rsi(subnormal, period=2).to_arrow().to_pylist() == [None, None, 100.0]


def test_real_native_indicators_reset_after_missing_values_with_mixed_dtypes() -> None:
    close = cfa.from_arrow(pa.array([1.0, 2.0, None, 4.0, 5.0, float("nan"), 7.0, 8.0]))
    high = cfa.from_arrow(
        pa.array([2.0, 3.0, None, 5.0, 6.0, 7.0, 8.0, 9.0], type=pa.float32())
    )
    low = cfa.from_arrow(pa.array([0.0, 1.0, 0.0, 3.0, 4.0, 5.0, 6.0, 7.0]))
    assert high.dtype == "float32"
    assert low.dtype == close.dtype == "float64"
    rsi = cfa.rsi(close, period=1)
    atr = cfa.atr(high, low, close, period=1)
    assert rsi.dtype == atr.dtype == "float64"
    assert rsi.to_arrow().to_pylist() == [
        None,
        100.0,
        None,
        None,
        100.0,
        None,
        None,
        100.0,
    ]
    assert atr.to_arrow().to_pylist() == [None, 2.0, None, None, 2.0, None, None, 2.0]


def test_real_native_indicators_reject_invalid_vector_descriptors() -> None:
    values = cfa.from_arrow(pa.array([1.0, 2.0]))
    boolean = cfa.from_arrow(pa.array([True, False]))
    matrix = cfa.from_arrow(pa.array([1.0, 2.0]), shape=(1, 2))
    short = cfa.from_arrow(pa.array([1.0]))
    with pytest.raises(cf.ConfigError, match=r"input\.shape"):
        cfa.rsi(matrix, period=2)
    with pytest.raises(cf.ConfigError, match=r"input\.dtype"):
        cfa.rsi(boolean, period=2)
    with pytest.raises(cf.ConfigError, match=r"inputs\.low\.shape"):
        cfa.atr(values, short, values, period=2)
    with pytest.raises(cf.ConfigError, match=r"inputs\.high\.shape"):
        cfa.atr(matrix, values, values, period=2)


def test_real_native_indicators_reject_infinity_and_invalid_ranges_before_missing() -> (
    None
):
    infinite = cfa.from_arrow(pa.array([float("inf")]))
    missing = cfa.from_arrow(pa.array([None], type=pa.float64()))
    low = cfa.from_arrow(pa.array([2.0]))
    high = cfa.from_arrow(pa.array([1.0]))
    with pytest.raises(cf.ConfigError, match=r"input\[0\].*finite non-missing"):
        cfa.rsi(infinite, period=2)
    with pytest.raises(cf.ConfigError, match=r"inputs\.low\[0\].*finite non-missing"):
        cfa.atr(missing, infinite, missing, period=2)
    with pytest.raises(
        cf.ConfigError, match=r"inputs\.high\[0\].*greater than or equal"
    ):
        cfa.atr(high, low, missing, period=2)
    assert high.to_arrow().to_pylist() == [1.0]
    assert low.to_arrow().to_pylist() == [2.0]


def test_real_native_indicators_accept_empty_vectors_and_large_periods() -> None:
    empty = cfa.from_arrow(pa.array([], type=pa.float32()))
    rsi = cfa.rsi(empty, period=2**64 - 1)
    atr = cfa.atr(empty, empty, empty, period=2**64 - 1)
    assert rsi.shape == atr.shape == (0,)
    assert rsi.dtype == atr.dtype == "float64"
    assert rsi.to_arrow().to_pylist() == atr.to_arrow().to_pylist() == []
    values = cfa.from_arrow(pa.array([1.0, 2.0]))
    assert cfa.rsi(values, period=2**64 - 1).to_arrow().to_pylist() == [None, None]
    assert cfa.atr(values, values, values, period=2**64 - 1).to_arrow().to_pylist() == [
        None,
        None,
    ]


def test_real_native_indicators_enforce_output_and_allocation_budgets() -> None:
    values = cfa.from_arrow(pa.array([1.0, 2.0]))
    elements = cfa.ArrayExecutionOptions(max_elements=1)
    allocation = cfa.ArrayExecutionOptions(max_allocation_bytes=0)
    with pytest.raises(cf.ConfigError, match=r"outputs\.output\.elements"):
        cfa.rsi(values, period=1, execution=elements)
    with pytest.raises(cf.ConfigError, match=r"outputs\.output\.elements"):
        cfa.atr(values, values, values, period=1, execution=elements)
    with pytest.raises(
        cf.ConfigError, match=r"outputs\.output\.bytes|execution\.max_allocation_bytes"
    ):
        cfa.rsi(values, period=1, execution=allocation)
    with pytest.raises(
        cf.ConfigError, match=r"outputs\.output\.bytes|execution\.max_allocation_bytes"
    ):
        cfa.atr(values, values, values, period=1, execution=allocation)


def test_real_native_indicators_do_not_import_numpy_or_jax(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    high = cfa.from_arrow(pa.array([2.0, 3.0, 4.0]))
    low = cfa.from_arrow(pa.array([0.0, 1.0, 2.0]))
    close = cfa.from_arrow(pa.array([1.0, 2.0, 3.0]))
    original_import = builtins.__import__

    def checked_import(name: str, *args: object, **kwargs: object) -> object:
        if name.split(".", 1)[0] in {"numpy", "jax"}:
            raise AssertionError("native indicators must not import NumPy or JAX")
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", checked_import)
    rsi = cfa.rsi(close, period=2)
    atr = cfa.atr(high, low, close, period=2)
    assert rsi.to_arrow().to_pylist() == [None, None, 100.0]
    assert atr.to_arrow().to_pylist() == [None, None, 2.0]
