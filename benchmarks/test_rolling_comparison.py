"""Compare native incremental rolling with DataFusion SQL window functions."""

from __future__ import annotations

import os
import platform
import statistics
import time
from importlib.metadata import version
from typing import Any

import numpy as np
import pyarrow as pa
import pytest

from benchmarks.support import BenchmarkFixture, benchmark_group, selected_scale
from benchmarks.symbolic_support import (
    SYMBOLIC_SEED,
    quote_workload,
    record_symbolic_benchmark,
)
from calc_flow import Batch, PipelineBuilder, Runtime
from calc_flow.symbolic import FeatureSet, Field, Program, rows, table_input, ts

SHORT_WINDOW = 20
LONG_WINDOW = 60
STEP_ROWS = 1
WARM_UP_RUNS = 1
PAIRED_SAMPLE_COUNT = 30
NULL_PERIODS = 29
RELATIVE_TOLERANCE = 1e-10
ABSOLUTE_TOLERANCE = 1e-12
SCENARIO = "incremental_rolling_vs_sql_window"
WORKLOAD_CONTRACT = "dal-184-rolling-comparison-v1"
_STATISTICS = ("mean", "stddev", "sum", "min", "max")
_FLOAT_FEATURES = tuple(
    f"{statistic}_close_{window}"
    for window in (SHORT_WINDOW, LONG_WINDOW)
    for statistic in _STATISTICS
)


def _sql_query() -> str:
    frame = (
        "PARTITION BY symbol ORDER BY event_time, sequence "
        "ROWS BETWEEN {preceding} PRECEDING AND CURRENT ROW"
    )
    expressions = []
    for window in (SHORT_WINDOW, LONG_WINDOW):
        rolling_frame = frame.format(preceding=window - 1)
        expressions.extend(
            (
                f"avg(close) OVER ({rolling_frame}) AS mean_close_{window}",
                f"stddev_samp(close) OVER ({rolling_frame}) AS stddev_close_{window}",
                f"sum(close) OVER ({rolling_frame}) AS sum_close_{window}",
                f"min(close) OVER ({rolling_frame}) AS min_close_{window}",
                f"max(close) OVER ({rolling_frame}) AS max_close_{window}",
            )
        )
    projection = ",\n  ".join(expressions)
    return f"""
SELECT
  event_time,
  sequence,
  symbol,
  industry,
  close,
  volume,
  {projection}
FROM input
ORDER BY event_time, symbol, sequence
"""


def _incremental_program() -> Program:
    quotes = table_input(
        "quotes",
        schema=(
            Field("event_time", "timestamp[us, UTC]", nullable=False),
            Field("sequence", "uint64", nullable=False),
            Field("symbol", "string", nullable=False),
            Field("industry", "string", nullable=False),
            Field("close", "float64"),
            Field("volume", "float64"),
        ),
        entity_by=("symbol",),
        event_time="event_time",
        sequence_by=("sequence",),
    )
    features: list[tuple[str, object]] = []
    for window_size in (SHORT_WINDOW, LONG_WINDOW):
        frame = rows(window_size)
        features.extend(
            (
                (f"mean_close_{window_size}", ts.mean(quotes["close"], window=frame)),
                (
                    f"stddev_close_{window_size}",
                    ts.stddev(quotes["close"], window=frame, ddof=1),
                ),
                (f"sum_close_{window_size}", ts.sum(quotes["close"], window=frame)),
                (f"min_close_{window_size}", ts.min(quotes["close"], window=frame)),
                (f"max_close_{window_size}", ts.max(quotes["close"], window=frame)),
            )
        )
    signals = quotes.with_columns(FeatureSet(features))
    return Program(
        "dal-184-incremental-rolling",
        inputs=(quotes,),
        outputs=(("signals", signals),),
    )


def _input_batch() -> tuple[Batch, int, int]:
    workload = quote_workload()
    table = workload.batch.to_pyarrow()
    event_time_index = table.schema.get_field_index("event_time")
    close_index = table.schema.get_field_index("close")
    event_time = table["event_time"].cast(pa.timestamp("us", tz="UTC"))
    close = table["close"].to_numpy(zero_copy_only=False)
    position = np.arange(workload.rows) // workload.entities
    null_mask = position % NULL_PERIODS == 0
    table = table.set_column(
        event_time_index,
        pa.field("event_time", pa.timestamp("us", tz="UTC"), nullable=False),
        event_time,
    ).set_column(
        close_index,
        pa.field("close", pa.float64()),
        pa.array(close, mask=null_mask),
    )
    return Batch.from_pyarrow(table), workload.entities, int(null_mask.sum())


def _assert_equivalent(incremental: pa.Table, sql: pa.Table) -> None:
    if incremental.column_names != sql.column_names:
        raise AssertionError(
            "rolling output columns differ: "
            f"{incremental.column_names} != {sql.column_names}"
        )
    exact_columns = ("event_time", "sequence", "symbol", "industry", "close", "volume")
    if not incremental.select(exact_columns).equals(sql.select(exact_columns)):
        raise AssertionError("rolling identity or input columns differ")
    for name in _FLOAT_FEATURES:
        incremental_values = incremental[name]
        sql_values = sql[name]
        if incremental_values.is_null().to_pylist() != sql_values.is_null().to_pylist():
            raise AssertionError(f"rolling null masks differ for {name}")
        np.testing.assert_allclose(
            incremental_values.to_numpy(zero_copy_only=False),
            sql_values.to_numpy(zero_copy_only=False),
            rtol=RELATIVE_TOLERANCE,
            atol=ABSOLUTE_TOLERANCE,
            equal_nan=True,
            err_msg=f"rolling values differ for {name}",
        )


def _timed_execute(plan: object, inputs: dict[str, Batch]) -> tuple[Any, float]:
    start = time.perf_counter_ns()
    result = plan.execute(inputs)
    seconds = (time.perf_counter_ns() - start) / 1_000_000_000
    return result, seconds


def _paired_samples(
    incremental_plan: object,
    sql_plan: object,
    inputs: dict[str, Batch],
) -> list[dict[str, object]]:
    samples: list[dict[str, object]] = []
    for index in range(PAIRED_SAMPLE_COUNT):
        if index % 2 == 0:
            _incremental_result, incremental_seconds = _timed_execute(
                incremental_plan, inputs
            )
            _sql_result, sql_seconds = _timed_execute(sql_plan, inputs)
            order = "incremental-first"
        else:
            _sql_result, sql_seconds = _timed_execute(sql_plan, inputs)
            _incremental_result, incremental_seconds = _timed_execute(
                incremental_plan, inputs
            )
            order = "sql-first"
        samples.append(
            {
                "order": order,
                "incremental_seconds": incremental_seconds,
                "sql_seconds": sql_seconds,
            }
        )
    return samples


def _timing_summary(
    samples: list[dict[str, object]],
    field: str,
    rows_count: int,
) -> dict[str, float]:
    durations = [float(sample[field]) for sample in samples]
    median_seconds = statistics.median(durations)
    return {
        "total_seconds": sum(durations),
        "mean_seconds": statistics.fmean(durations),
        "median_seconds": median_seconds,
        "throughput_rows_per_second": rows_count / median_seconds,
    }


@pytest.mark.benchmark(
    group=benchmark_group("rolling-comparison"), min_rounds=3, max_time=0.5
)
@pytest.mark.parametrize("_scale", [selected_scale().name])
def test_incremental_rolling_vs_sql_window(
    benchmark: BenchmarkFixture, _scale: str
) -> None:
    input_batch, entities, null_rows = _input_batch()
    inputs = {"input": input_batch}
    incremental_plan = _incremental_program().compile_batch(Runtime())
    sql_plan = (
        PipelineBuilder("dal-184-sql-window")
        .sql("features", _sql_query())
        .compile_batch()
    )

    incremental_warm = incremental_plan.execute(inputs)
    sql_warm = sql_plan.execute(inputs)
    incremental_output = incremental_warm.outputs["output"].to_pyarrow()
    sql_output = sql_warm.outputs["output"].to_pyarrow()
    _assert_equivalent(incremental_output, sql_output)

    samples = _paired_samples(incremental_plan, sql_plan, inputs)
    incremental_summary = _timing_summary(
        samples, "incremental_seconds", input_batch.num_rows
    )
    sql_summary = _timing_summary(samples, "sql_seconds", input_batch.num_rows)
    relative_change_percent = (
        (sql_summary["median_seconds"] - incremental_summary["median_seconds"])
        / sql_summary["median_seconds"]
        * 100.0
    )
    record_symbolic_benchmark(
        benchmark,
        scenario=SCENARIO,
        input_rows=input_batch.num_rows,
        output_rows=incremental_output.num_rows,
        metrics=incremental_warm.datafusion_metrics,
        extra={
            "comparison_contract": "same-process-alternating-v1",
            "workload_contract": WORKLOAD_CONTRACT,
            "build_profile": os.environ.get(
                "CALC_FLOW_BENCHMARK_PROFILE", "unspecified"
            ),
            "calc_flow_version": version("calc-flow"),
            "operating_system_release": platform.release(),
            "entities": entities,
            "input_seed": SYMBOLIC_SEED,
            "input_pattern": "deterministic interleaved per-symbol quotes",
            "null_rows": null_rows,
            "null_period_rows": NULL_PERIODS,
            "null_policy": "exclude null samples; null when no valid sample exists",
            "boundary_policy": "current row plus window_size - 1 preceding entity rows",
            "window_rows": [SHORT_WINDOW, LONG_WINDOW],
            "step_rows": STEP_ROWS,
            "features_per_window": list(_STATISTICS),
            "warm_up_runs_per_implementation": WARM_UP_RUNS,
            "paired_sample_count": PAIRED_SAMPLE_COUNT,
            "summary_statistic": "median of alternating same-process samples",
            "relative_tolerance": RELATIVE_TOLERANCE,
            "absolute_tolerance": ABSOLUTE_TOLERANCE,
            "incremental": incremental_summary,
            "sql_window": sql_summary,
            "incremental_change_percent": relative_change_percent,
            "regression_observed": relative_change_percent < 0.0,
            "paired_samples": samples,
        },
    )
    print(
        "DAL-184 rolling comparison: "
        f"rows={input_batch.num_rows}, "
        f"incremental={incremental_summary['median_seconds']:.6f}s "
        f"({incremental_summary['throughput_rows_per_second']:.0f} rows/s), "
        f"sql={sql_summary['median_seconds']:.6f}s "
        f"({sql_summary['throughput_rows_per_second']:.0f} rows/s), "
        f"incremental_change={relative_change_percent:+.2f}%"
    )

    def execute_pair() -> tuple[Any, Any]:
        return incremental_plan.execute(inputs), sql_plan.execute(inputs)

    incremental_result, sql_result = benchmark(execute_pair)
    _assert_equivalent(
        incremental_result.outputs["output"].to_pyarrow(),
        sql_result.outputs["output"].to_pyarrow(),
    )
