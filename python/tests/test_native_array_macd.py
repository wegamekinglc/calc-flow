from __future__ import annotations

import importlib.util
import inspect
import sys
from dataclasses import FrozenInstanceError, replace

import pyarrow as pa
import pytest

import calc_flow as cf
import calc_flow.array as cfa
from calc_flow import _native


def test_macd_outputs_is_one_frozen_slotted_public_container() -> None:
    assert cf.MacdOutputs is cfa.MacdOutputs
    assert "MacdOutputs" in cf.__all__
    outputs = cf.MacdOutputs("line", "signal", "histogram")
    assert (outputs.line, outputs.signal, outputs.histogram) == (
        "line",
        "signal",
        "histogram",
    )
    assert not hasattr(outputs, "__dict__")
    with pytest.raises(FrozenInstanceError):
        outputs.line = "changed"  # type: ignore[misc]
    assert replace(outputs, signal="new").signal == "new"
    assert outputs.signal == "signal"


def test_macd_public_signatures_preserve_named_keyword_defaults() -> None:
    eager = inspect.signature(cfa.macd_components).parameters
    declaration = inspect.signature(cf.ts.macd_components).parameters
    defaults = {
        "fast_span": 12,
        "slow_span": 26,
        "signal_span": 9,
        "min_periods": 1,
        "signal_min_periods": 1,
    }
    assert list(eager) == ["array", *defaults, "execution"]
    assert list(declaration) == ["value", *defaults]
    assert eager["array"].kind is inspect.Parameter.POSITIONAL_OR_KEYWORD
    assert declaration["value"].kind is inspect.Parameter.POSITIONAL_ONLY
    assert {name: eager[name].default for name in defaults} == defaults
    assert {name: declaration[name].default for name in defaults} == defaults
    assert all(
        eager[name].kind is inspect.Parameter.KEYWORD_ONLY
        for name in (*defaults, "execution")
    )
    assert all(
        declaration[name].kind is inspect.Parameter.KEYWORD_ONLY for name in defaults
    )
    assert eager["execution"].default is None


def test_macd_generic_container_uses_the_python39_frozen_slot_helper(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from calc_flow import _compat, _macd

    spec = importlib.util.spec_from_file_location(
        "calc_flow._compat_macd_python39", _compat.__file__
    )
    assert spec is not None and spec.loader is not None
    compatibility = importlib.util.module_from_spec(spec)
    with monkeypatch.context() as patch:
        patch.setattr(sys, "version_info", (3, 9, 0))
        spec.loader.exec_module(compatibility)
    container_spec = importlib.util.spec_from_file_location(
        "calc_flow._macd_python39", _macd.__file__
    )
    assert container_spec is not None and container_spec.loader is not None
    module = importlib.util.module_from_spec(container_spec)
    monkeypatch.setitem(sys.modules, container_spec.name, module)
    monkeypatch.setattr(_compat, "dataclass", compatibility.dataclass)
    container_spec.loader.exec_module(module)
    output = module.MacdOutputs(1, 2, 3)
    assert (output.line, output.signal, output.histogram) == (1, 2, 3)
    assert not hasattr(output, "__dict__")
    with pytest.raises(FrozenInstanceError):
        output.signal = 4


def test_native_macd_adapter_forwards_one_normalized_tuple_and_owned_results(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    values = cfa.from_arrow(pa.array([1.0, 2.0]))
    calls: list[tuple[object, ...]] = []

    def binding(*arguments: object) -> tuple[cfa.NativeArray, ...]:
        calls.append(arguments)
        return values, values, values

    monkeypatch.setattr(
        _native, "_native_array_macd_components", binding, raising=False
    )
    output = cfa.macd_components(values)
    assert calls == [(values, (12, 26, 9, 1, 1), 10_000_000, 256 * 1024 * 1024)]
    assert isinstance(output, cf.MacdOutputs)
    assert output.line is output.signal is output.histogram is values


@pytest.mark.parametrize(
    "name",
    ["fast_span", "slow_span", "signal_span", "min_periods", "signal_min_periods"],
)
@pytest.mark.parametrize("invalid", [True, 0, 2**64])
def test_native_macd_validates_every_count_before_binding(
    name: str, invalid: object
) -> None:
    values = cfa.from_arrow(pa.array([1.0]))
    error = TypeError if invalid is True else cf.ConfigError
    with pytest.raises(error, match=name):
        cfa.macd_components(values, **{name: invalid})


@pytest.mark.parametrize("fast,slow", [(2, 2), (3, 2)])
def test_native_macd_rejects_reversed_spans(fast: int, slow: int) -> None:
    values = cfa.from_arrow(pa.array([1.0]))
    with pytest.raises(cf.ConfigError, match="fast_span.*less than slow_span"):
        cfa.macd_components(values, fast_span=fast, slow_span=slow)


def test_native_macd_allows_minima_above_spans_and_uses_requested_limits(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    values = cfa.from_arrow(pa.array([1.0]))
    calls: list[tuple[object, ...]] = []

    def binding(*arguments: object) -> tuple[cfa.NativeArray, ...]:
        calls.append(arguments)
        return values, values, values

    monkeypatch.setattr(
        _native, "_native_array_macd_components", binding, raising=False
    )
    cfa.macd_components(
        values,
        fast_span=1,
        slow_span=2,
        signal_span=1,
        min_periods=2**64 - 1,
        signal_min_periods=11,
        execution=cfa.ArrayExecutionOptions(max_elements=3, max_allocation_bytes=1024),
    )
    assert calls == [(values, (1, 2, 1, 2**64 - 1, 11), 3, 1024)]


def test_native_macd_rejects_non_owners_and_invalid_execution_limits() -> None:
    with pytest.raises(TypeError, match="NativeArray"):
        cfa.macd_components([1.0, 2.0])
    values = cfa.from_arrow(pa.array([1.0]))
    with pytest.raises(TypeError, match="execution.*ArrayExecutionOptions"):
        cfa.macd_components(values, execution=object())


def test_real_native_macd_matches_composed_ewmas_including_a_carried_line() -> None:
    values = cfa.from_arrow(pa.array([1.0, 2.0, None, 3.0]))
    parts = cfa.macd_components(
        values, fast_span=1, slow_span=2, signal_span=2, signal_min_periods=2
    )
    line = cfa.subtract(cfa.ewma(values, span=1), cfa.ewma(values, span=2))
    signal = cfa.ewma(line, span=2, min_periods=2)
    histogram = cfa.subtract(line, signal)
    assert parts.line.to_arrow().to_pylist() == line.to_arrow().to_pylist()
    assert parts.signal.to_arrow().to_pylist() == signal.to_arrow().to_pylist()
    assert parts.histogram.to_arrow().to_pylist() == histogram.to_arrow().to_pylist()
    assert parts.line.shape == parts.signal.shape == parts.histogram.shape == (4,)
    assert values.to_arrow().to_pylist() == [1.0, 2.0, None, 3.0]
