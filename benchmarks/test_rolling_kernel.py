"""Paired native rolling-kernel evidence against DataFusion window plans."""

from __future__ import annotations

from collections import defaultdict, deque
from typing import Any

import numpy as np
import pyarrow as pa
import pytest

from benchmarks.support import BenchmarkFixture, benchmark_group, selected_scale
from benchmarks.symbolic_support import (
    alternating_plan_samples,
    execute_compiled_plan,
    quote_workload,
    record_symbolic_benchmark,
    utc_event_time_batch,
)
from calc_flow import PipelineBuilder, Runtime
from calc_flow.symbolic import FeatureSet, Field, Program, rows, table_input, ts

_SMA_SCENARIO = "rolling_kernel_sma20"
_DUAL_SMA_SCENARIO = "rolling_kernel_dual_sma_5_20"
_WORKLOAD_CONTRACT = "rolling-kernel-paired-v1"
_COMPARISON_CONTRACT = "same-process-alternating-v1"
_PAIRED_SAMPLES = 60
_FAST_WINDOW = 5
_SLOW_WINDOW = 20
_RTOL = 1e-10
_ATOL = 1e-10


def _symbolic_plan(*, dual: bool) -> tuple[object, str]:
    quotes = table_input(
        "quotes",
        schema=[
            Field("event_time", "timestamp[us, UTC]", nullable=False),
            Field("sequence", "uint64", nullable=False),
            Field("symbol", "string", nullable=False),
            Field("industry", "string", nullable=False),
            Field("close", "float64", nullable=True),
            Field("volume", "float64", nullable=True),
        ],
        entity_by=["symbol"],
        event_time="event_time",
        sequence_by=["sequence"],
    )
    slow = ts.mean(quotes["close"], window=rows(_SLOW_WINDOW))
    indicator = (
        ts.mean(quotes["close"], window=rows(_FAST_WINDOW)) - slow if dual else slow
    )
    output = quotes.with_columns(FeatureSet((("indicator", indicator),)))
    program = Program(
        "rolling-kernel-paired",
        inputs=(quotes,),
        outputs=(("output", output),),
    )
    runtime = Runtime()
    return program.compile_batch(runtime), program.explain(runtime, mode="batch")


def _reference_query(*, dual: bool) -> str:
    def mean(window: int) -> str:
        frame = (
            "PARTITION BY symbol ORDER BY event_time, sequence "
            f"ROWS BETWEEN {window - 1} PRECEDING AND CURRENT ROW"
        )
        return f"SUM(close) OVER ({frame}) / COUNT(close) OVER ({frame})"

    indicator = (
        f"({mean(_FAST_WINDOW)}) - ({mean(_SLOW_WINDOW)})"
        if dual
        else mean(_SLOW_WINDOW)
    )
    return f"SELECT *, {indicator} AS indicator FROM input"


def _reference_plan(*, dual: bool) -> object:
    return (
        PipelineBuilder("rolling-kernel-reference")
        .sql("reference", _reference_query(dual=dual))
        .compile_batch()
    )


def _direct_window_oracle(table: pa.Table, *, dual: bool) -> np.ndarray:
    histories: dict[str, deque[float]] = defaultdict(lambda: deque(maxlen=_SLOW_WINDOW))
    expected = np.empty(table.num_rows, dtype=np.float64)
    symbols = table["symbol"].to_pylist()
    prices = table["close"].to_numpy(zero_copy_only=False)
    for index, (symbol, price) in enumerate(zip(symbols, prices, strict=True)):
        history = histories[symbol]
        history.append(float(price))
        slow = float(np.sum(history, dtype=np.float64)) / len(history)
        if dual:
            fast_values = tuple(history)[-_FAST_WINDOW:]
            fast = float(np.sum(fast_values, dtype=np.float64)) / len(fast_values)
            expected[index] = fast - slow
        else:
            expected[index] = slow
    return expected


def _ordered_indicator(result: object, output: str) -> np.ndarray:
    table = result.outputs[output].to_pyarrow()
    sequence = table["sequence"].to_numpy(zero_copy_only=False)
    values = table["indicator"].to_numpy(zero_copy_only=False)
    return values[np.argsort(sequence, kind="stable")]


def _run_paired_case(benchmark: BenchmarkFixture, *, dual: bool, scenario: str) -> None:
    workload = quote_workload()
    inputs = {"input": utc_event_time_batch(workload.batch)}
    reference = _reference_plan(dual=dual)
    optimized, optimized_explanation = _symbolic_plan(dual=dual)
    expected_state_groups = 2 if dual else 1
    assert (
        "selected=ordered_primitive profile=stable_v1 "
        "complexity=amortized_constant" in optimized_explanation
    )
    assert f"shared_state_groups={expected_state_groups}" in optimized_explanation
    assert "fallback=none" in optimized_explanation
    reference_result = execute_compiled_plan(reference, inputs)
    optimized_result = execute_compiled_plan(optimized, inputs)
    expected = _direct_window_oracle(inputs["input"].to_pyarrow(), dual=dual)
    reference_values = _ordered_indicator(reference_result, "output")
    optimized_values = _ordered_indicator(optimized_result, "output")
    np.testing.assert_allclose(reference_values, expected, rtol=_RTOL, atol=_ATOL)
    np.testing.assert_allclose(optimized_values, expected, rtol=_RTOL, atol=_ATOL)
    reference_metric = reference_result.datafusion_metrics[0]
    assert reference_metric["rolling_rewritten_windows"] == 0
    paired_samples = alternating_plan_samples(
        reference,
        optimized,
        inputs,
        sample_count=_PAIRED_SAMPLES,
    )

    record_symbolic_benchmark(
        benchmark,
        scenario=scenario,
        input_rows=workload.rows,
        output_rows=workload.rows,
        extra={
            "comparison_contract": _COMPARISON_CONTRACT,
            "workload_contract": _WORKLOAD_CONTRACT,
            "oracle": "independent_direct_window_v1",
            "oracle_checked_rows": workload.rows,
            "oracle_finite_rows": int(np.isfinite(expected).sum()),
            "oracle_rtol": _RTOL,
            "oracle_atol": _ATOL,
            "optimized_kernel": "ordered_primitive",
            "optimized_shared_state_groups": expected_state_groups,
            "reference_rolling_rewrites": 0,
            "fast_window": _FAST_WINDOW if dual else None,
            "slow_window": _SLOW_WINDOW,
            "paired_samples": paired_samples,
        },
    )

    def execute_pair() -> tuple[Any, Any]:
        return (
            execute_compiled_plan(reference, inputs),
            execute_compiled_plan(optimized, inputs),
        )

    reference_result, optimized_result = benchmark(execute_pair)
    assert reference_result.outputs["output"].num_rows == workload.rows
    assert optimized_result.outputs["output"].num_rows == workload.rows


@pytest.mark.benchmark(
    group=benchmark_group("rolling-kernel-sma20"), min_rounds=3, max_time=0.5
)
@pytest.mark.parametrize("_scale", [selected_scale().name])
def test_rolling_kernel_sma20(benchmark: BenchmarkFixture, _scale: str) -> None:
    _run_paired_case(benchmark, dual=False, scenario=_SMA_SCENARIO)


@pytest.mark.benchmark(
    group=benchmark_group("rolling-kernel-dual-sma"), min_rounds=3, max_time=0.5
)
@pytest.mark.parametrize("_scale", [selected_scale().name])
def test_rolling_kernel_dual_sma(benchmark: BenchmarkFixture, _scale: str) -> None:
    _run_paired_case(benchmark, dual=True, scenario=_DUAL_SMA_SCENARIO)
