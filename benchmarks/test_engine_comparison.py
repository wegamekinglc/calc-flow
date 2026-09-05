from __future__ import annotations

from pathlib import Path

import numpy as np
import pyarrow as pa
import pytest

from benchmarks.engine_comparison import (
    EngineCase,
    expected_output,
    sql_query,
    workload,
)
from scripts.benchmark_suite.catalog import engine_cases


def test_sql_queries_reject_unknown_scenario_names():
    with pytest.raises(ValueError, match="unsupported SQL benchmark scenario"):
        sql_query("sma20; DROP TABLE input")


@pytest.mark.parametrize("scenario", ["sma20", "dual_sma"])
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
@pytest.mark.parametrize("scenario", ["sma20", "dual_sma"])
def test_ready_stream_finalizes_every_chunk_before_eof(count, scenario, tmp_path):
    case = next(
        case
        for case in engine_cases(count)
        if case["backend"] == "calc-flow-stream" and case["scenario"] == scenario
    )
    runner = EngineCase(case, tmp_path)
    try:
        for _ in range(2):
            sample = runner.sample()
            assert sample["seconds"] > 0
            assert sample["correctness"]["rows"] == count
            assert sample["correctness"]["finite_rows"] > 0
    finally:
        runner.close()


def test_performance_prices_are_exact_eighths_with_bounded_magnitude():
    prices = workload(1_001).table["price"].to_numpy()
    assert np.all(prices * 8 == np.floor(prices * 8))
    assert np.all((prices >= 64) & (prices < 256))


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
