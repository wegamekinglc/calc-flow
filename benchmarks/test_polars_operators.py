from __future__ import annotations

import numpy as np
import pyarrow as pa
import pytest

from benchmarks.engine_comparison import EngineCase, Workload, _polars, workload
from scripts.benchmark_suite.catalog import (
    POLARS_CASES,
    ROW_SCALES,
    STREAM_CASES,
    engine_cases,
)

POLARS_GAPS = (
    "average",
    "argmax64",
    "argmax256",
    "unique64",
    "cs_mean",
    "window_sum",
)


def test_polars_catalog_covers_every_stream_scenario_at_original_scales():
    assert POLARS_CASES == STREAM_CASES
    for rows in ROW_SCALES:
        for backend in ("polars", "polars-1t"):
            registered = {
                case["scenario"]
                for case in engine_cases(rows)
                if case["backend"] == backend
            }
            assert set(POLARS_GAPS) <= registered


@pytest.mark.parametrize("scenario", POLARS_GAPS)
@pytest.mark.parametrize("rows", (10, 17_001, 64_001))
def test_added_polars_scenarios_match_independent_oracles(tmp_path, scenario, rows):
    runner = EngineCase(
        {"backend": "polars", "scenario": scenario, "rows": rows}, tmp_path
    )
    original = runner.data.table.combine_chunks()
    try:
        assert runner.sample()["correctness"]["passed"]
        assert runner.data.table.combine_chunks().equals(original)
    finally:
        runner.close()


@pytest.mark.parametrize("window", (64, 256))
def test_polars_argmax_uses_oldest_ties_and_entity_row_lags(window):
    original = workload(600)
    ticks = np.arange(600) // 2
    prices = np.where(ticks % 31 == 0, 10.0, (ticks % 7).astype(float))
    table = original.table.set_column(
        2, "symbol", pa.array((np.arange(600) % 2).astype(str))
    ).set_column(3, "price", pa.array(prices))
    # Arrival order differs from event-time/sequence order.
    data = Workload(table.take(np.arange(599, -1, -1)), original.dimension, 2)
    result = _polars(data, f"argmax{window}")().sort_by("sequence")
    expected = np.empty(600)
    for entity in range(2):
        series = prices[entity::2]
        for index in range(len(series)):
            frame = series[max(0, index - window + 1) : index + 1]
            expected[entity + index * 2] = len(frame) - 1 - np.argmax(frame)
    np.testing.assert_array_equal(result["value"].to_numpy(), expected)


def test_polars_unique_counts_repeated_values_as_the_window_expires():
    original = workload(150)
    prices = np.asarray([1.0] * 65 + [2.0] * 65 + [3.0] * 20)
    table = original.table.set_column(2, "symbol", pa.array(["S000"] * 150))
    table = table.set_column(3, "price", pa.array(prices))
    data = Workload(table, original.dimension, 1)
    result = _polars(data, "unique64")().sort_by("sequence")
    expected = [len(set(prices[max(0, i - 63) : i + 1])) for i in range(150)]
    np.testing.assert_array_equal(result["value"].to_numpy(), expected)


def test_polars_average_orders_by_event_time_then_sequence_per_entity():
    original = workload(6)
    table = pa.table(
        {
            "event_time": [2, 1, 1, 1, 2, 1],
            "sequence": [4, 2, 0, 3, 5, 1],
            "symbol": ["a", "a", "a", "b", "b", "b"],
            "price": [9.0, 3.0, 1.0, 6.0, 12.0, 2.0],
        },
        schema=original.table.schema,
    )
    result = _polars(Workload(table, original.dimension, 2), "average")()
    np.testing.assert_allclose(
        result.sort_by("sequence")["value"].to_numpy(),
        [1.0, 2.0, 2.0, 4.0, 13.0 / 3, 20.0 / 3],
    )


def test_polars_cross_section_mean_groups_exact_timestamps():
    original = workload(5)
    table = pa.table(
        {
            "event_time": [1, 2, 1, 3, 2],
            "sequence": range(5),
            "symbol": ["a", "a", "b", "b", "b"],
            "price": [1.0, 4.0, 3.0, 7.0, 8.0],
        },
        schema=original.table.schema,
    )
    result = _polars(Workload(table, original.dimension, 2), "cs_mean")()
    assert result.sort_by("sequence")["value"].to_pylist() == [2, 6, 2, 7, 6]


def test_polars_tumbling_sum_uses_epoch_anchored_half_open_windows():
    original = workload(7)
    table = pa.table(
        {
            "event_time": [-1, 0, 9_999_999, 10_000_000, 19_999_999, 20_000_000, 0],
            "sequence": range(7),
            "symbol": ["a"] * 6 + ["b"],
            "price": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 10.0],
        },
        schema=original.table.schema,
    )
    result = _polars(Workload(table, original.dimension, 2), "window_sum")()
    result = result.sort_by([("symbol", "ascending"), ("window_start", "ascending")])
    assert result["window_start"].type == pa.timestamp("us", tz="UTC")
    assert result["window_start"].cast(pa.int64()).to_pylist() == [
        -10_000_000,
        0,
        10_000_000,
        20_000_000,
        0,
    ]
    assert result["value"].to_pylist() == [1.0, 5.0, 9.0, 6.0, 10.0]
