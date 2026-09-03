"""Paired native rolling-kernel evidence against DataFusion window plans."""

from __future__ import annotations

import time
from collections import defaultdict, deque
from typing import Any

import numpy as np
import pyarrow as pa
import pytest

from benchmarks.support import BenchmarkFixture, benchmark_group, selected_scale
from benchmarks.symbolic_support import quote_workload, record_symbolic_benchmark
from calc_flow import Batch, PipelineBuilder, Runtime
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


def _utc_batch(batch: Batch) -> Batch:
    table = batch.to_pyarrow()
    event_time = table.schema.get_field_index("event_time")
    return Batch.from_pyarrow(
        table.set_column(
            event_time,
            pa.field("event_time", pa.timestamp("us", tz="UTC"), nullable=False),
            table["event_time"].cast(pa.timestamp("us", tz="UTC")),
        )
    )


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


def _timed_execute(plan: object, batch: Batch) -> tuple[Any, float]:
    started = time.perf_counter_ns()
    result = plan.execute({"input": batch})
    elapsed = (time.perf_counter_ns() - started) / 1_000_000_000
    return result, elapsed


def _alternating_samples(
    reference: object, optimized: object, batch: Batch
) -> list[dict[str, object]]:
    samples: list[dict[str, object]] = []
    for index in range(_PAIRED_SAMPLES):
        if index % 2 == 0:
            _reference_result, reference_seconds = _timed_execute(reference, batch)
            _optimized_result, optimized_seconds = _timed_execute(optimized, batch)
            order = "hand-built-first"
        else:
            _optimized_result, optimized_seconds = _timed_execute(optimized, batch)
            _reference_result, reference_seconds = _timed_execute(reference, batch)
            order = "symbolic-first"
        samples.append(
            {
                "order": order,
                "hand_built_seconds": reference_seconds,
                "symbolic_seconds": optimized_seconds,
            }
        )
    return samples


def _run_paired_case(benchmark: BenchmarkFixture, *, dual: bool, scenario: str) -> None:
    workload = quote_workload()
    batch = _utc_batch(workload.batch)
    reference = _reference_plan(dual=dual)
    optimized, optimized_explanation = _symbolic_plan(dual=dual)
    expected_state_groups = 2 if dual else 1
    assert (
        "selected=ordered_primitive profile=stable_v1 "
        "complexity=amortized_constant" in optimized_explanation
    )
    assert f"shared_state_groups={expected_state_groups}" in optimized_explanation
    assert "fallback=none" in optimized_explanation
    reference_result = reference.execute({"input": batch})
    optimized_result = optimized.execute({"input": batch})
    expected = _direct_window_oracle(batch.to_pyarrow(), dual=dual)
    reference_values = _ordered_indicator(reference_result, "output")
    optimized_values = _ordered_indicator(optimized_result, "output")
    np.testing.assert_allclose(reference_values, expected, rtol=_RTOL, atol=_ATOL)
    np.testing.assert_allclose(optimized_values, expected, rtol=_RTOL, atol=_ATOL)
    reference_metric = reference_result.datafusion_metrics[0]
    assert reference_metric["rolling_rewritten_windows"] == 0
    paired_samples = _alternating_samples(reference, optimized, batch)

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
            reference.execute({"input": batch}),
            optimized.execute({"input": batch}),
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
