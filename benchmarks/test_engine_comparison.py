from __future__ import annotations

from pathlib import Path

import numpy as np
import pyarrow as pa
import pytest

from benchmarks import engine_comparison
from benchmarks.engine_comparison import (
    EngineCase,
    expected_output,
    sql_query,
    workload,
)
from scripts.benchmark_suite.catalog import STREAM_CASES, engine_cases


def test_sql_queries_reject_unknown_scenario_names():
    with pytest.raises(ValueError, match="unsupported SQL benchmark scenario"):
        sql_query("sma20; DROP TABLE input")


def test_polars_samples_collect_through_the_streaming_engine(monkeypatch):
    recorded: dict[str, object] = {}

    class RecordingResult:
        def to_arrow(self):
            return pa.table({"value": [1.0]})

    class RecordingPlan:
        def collect(self, **kwargs):
            recorded.update(kwargs)
            return RecordingResult()

    monkeypatch.setattr(
        engine_comparison, "_polars_plan", lambda data, scenario: RecordingPlan()
    )
    result = engine_comparison._polars(workload(101), "projection")()
    assert recorded == {"engine": "streaming"}
    assert result == pa.table({"value": [1.0]})


@pytest.mark.parametrize("scenario", STREAM_CASES)
def test_ready_stream_repeated_samples_use_fresh_execution_plans(scenario, tmp_path):
    case = next(
        case
        for case in engine_cases(10)
        if case["backend"] == "calc-flow-stream" and case["scenario"] == scenario
    )
    runner = EngineCase(case, tmp_path)
    try:
        for _ in range(3):
            assert runner.sample()["correctness"]["passed"]
    finally:
        runner.close()


@pytest.mark.parametrize("count", [64_001, 128_000])
@pytest.mark.parametrize(
    "scenario",
    tuple(
        scenario
        for scenario in STREAM_CASES
        if scenario not in ("window_sum", "asof_join")
    ),
)
def test_ready_stream_finalizes_every_chunk_before_eof(count, scenario, tmp_path):
    # Boundary sizes are correctness fixtures, not suite tiers: the catalog's
    # stream-join evidence cap must not drop the chunk-finalization coverage,
    # so the case is built here instead of looked up from engine_cases.
    case = {
        "id": f"engines/{count}/calc-flow-stream/{scenario}",
        "family": "engines",
        "backend": "calc-flow-stream",
        "scenario": scenario,
        "rows": count,
        "scope": "ready-enqueue-to-arrow",
    }
    runner = EngineCase(case, tmp_path)
    try:
        for _ in range(2):
            sample = runner.sample()
            assert sample["seconds"] > 0
            # The stream output cardinality matches the scenario oracle: one
            # row per input for row-local scenarios, one row per group for
            # group_by, and the filtered count for filter.
            assert sample["correctness"]["rows"] == runner.expected.num_rows
            assert sample["correctness"]["finite_rows"] > 0
    finally:
        runner.close()


def test_performance_prices_are_exact_eighths_with_bounded_magnitude():
    prices = workload(1_001).table["price"].to_numpy()
    assert np.all(prices * 8 == np.floor(prices * 8))
    assert np.all((prices >= 64) & (prices < 256))


@pytest.mark.parametrize("window", [64, 256])
def test_argmax_oracle_covers_full_periodic_windows(window):
    data = workload(20_000)
    actual = expected_output(data, f"argmax{window}")["value"].to_numpy()
    prices = data.table["price"].to_numpy()
    for entity in range(data.entities):
        series = prices[entity :: data.entities]
        for tick in (window - 1, window, 256, 257, 300):
            index = entity + tick * data.entities
            if index >= len(prices):
                continue
            trailing = series[max(0, tick - window + 1) : tick + 1]
            expected_age = len(trailing) - 1 - np.argmax(trailing)
            assert actual[index] == expected_age


@pytest.mark.parametrize(
    "scenario",
    (
        "average",
        "argmax64",
        "argmax256",
        "unique64",
        "cs_mean",
        "window_sum",
        "asof_join",
    ),
)
def test_new_stream_operators_match_independent_oracle(scenario, tmp_path):
    case = next(
        case
        for case in engine_cases(101)
        if case["backend"] == "calc-flow-stream" and case["scenario"] == scenario
    )
    runner = EngineCase(case, tmp_path)
    try:
        assert runner.sample()["correctness"]["passed"]
    finally:
        runner.close()


@pytest.mark.parametrize(
    "case", [*engine_cases(10), *engine_cases(101)], ids=lambda case: case["id"]
)
def test_engine_outputs_match_independent_oracle(case: dict, tmp_path: Path):
    runner = EngineCase(case, tmp_path)
    try:
        result = runner.sample()
        assert result["seconds"] > 0
        assert result["correctness"]["passed"]
        if case["rows"] >= 20 or case["scenario"] not in ("sma20", "dual_sma"):
            assert result["correctness"]["finite_rows"] > 0
        else:
            assert result["correctness"]["finite_rows"] == 0
    finally:
        runner.close()


@pytest.mark.parametrize("count", [10, 19, 20, 21, 41, 1_001])
def test_full_window_oracle_matches_direct_slices(count: int):
    data = workload(count)
    for scenario in ("sma20", "dual_sma"):
        expected = expected_output(data, scenario)
        prices = data.table["price"].to_numpy()
        direct = np.full(count, np.nan)
        for index in range(count):
            entity = index % data.entities
            history = prices[entity : index + 1 : data.entities]
            if len(history) >= 20:
                direct[index] = np.mean(history[-20:])
                if scenario == "dual_sma":
                    direct[index] = np.mean(history[-5:]) - direct[index]
        np.testing.assert_allclose(expected["value"].to_numpy(), direct, equal_nan=True)


def test_corrupted_output_is_rejected(tmp_path: Path):
    case = next(c for c in engine_cases(101) if c["backend"] == "ta-lib")
    runner = EngineCase(case, tmp_path)
    try:
        index = runner.expected.column_names.index("value")
        bad = runner.expected.set_column(
            index, "value", pa.array(np.zeros(runner.expected.num_rows))
        )
        with pytest.raises(AssertionError):
            runner.validate(bad)
    finally:
        runner.close()
