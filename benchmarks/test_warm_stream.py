"""Correctness and accounting guards for warm StreamingRunner measurements."""

from __future__ import annotations

import asyncio
import json
import runpy
import sys
import zoneinfo
from datetime import timedelta
from pathlib import Path

import numpy as np
import pyarrow as pa
import pytest

from benchmarks.warm_stream import (
    BASE,
    ScenarioConfig,
    WarmScenario,
    _expected,
    _input_table,
    _prepared_events,
    _segment,
    _summary,
    _validate_output,
)


def test_warm_shared_helpers_do_not_require_optional_talib(monkeypatch) -> None:
    monkeypatch.setitem(sys.modules, "talib", None)
    helpers = runpy.run_path(
        str(Path(__file__).with_name("rolling_indicator_comparison.py"))
    )
    values = helpers["expected_rolling_mean"](
        np.arange(64, dtype=np.float64), entities=64, window=20
    )
    np.testing.assert_array_equal(values, np.arange(64, dtype=np.float64))


def test_warm_oracle_uses_only_window_context() -> None:
    for indicator in ("rolling_mean", "dual_sma_spread"):
        values = _expected(
            ScenarioConfig(indicator=indicator),
            start=1_024_000,
            rows=64,
        )
        assert values.shape == (64,)
        assert np.isfinite(values).all()


def test_warm_summary_reports_tail_latency_and_original_samples() -> None:
    samples = [0.001, 0.002, 0.003, 0.004]
    summary = _summary(samples, 64)
    assert summary["median_seconds"] == 0.0025
    assert summary["p95_seconds"] == pytest.approx(0.00385)
    assert summary["samples_seconds"] == samples


def test_warm_validation_rejects_changed_identity_and_values() -> None:
    config = ScenarioConfig()
    expected = _expected(
        config,
        start=1_024,
        rows=64,
    )
    table = _segment(1_024, 64, 64).append_column("moving_average", pa.array(expected))
    kwargs = dict(
        config=config,
        start=1_024,
        rows=64,
    )
    assert _validate_output(table, **kwargs)["passed"]
    broken = table.set_column(1, "sequence", pa.array(np.zeros(64, dtype=np.uint64)))
    with pytest.raises(RuntimeError, match="sequence"):
        _validate_output(broken, **kwargs)
    broken = table.set_column(4, "moving_average", pa.array(expected + 1))
    with pytest.raises(RuntimeError, match="correctness"):
        _validate_output(broken, **kwargs)


@pytest.mark.parametrize("indicator", ("rolling_mean", "dual_sma_spread"))
def test_warm_scenario_advances_one_runner_and_validates_every_append(
    tmp_path, indicator
) -> None:
    async def exercise() -> None:
        scenario = await WarmScenario.start(
            ScenarioConfig(history_rows=128, append_rows=64, indicator=indicator),
            tmp_path,
        )
        try:
            first = await scenario.sample(collect_gc=False)
            second = await scenario.sample(collect_gc=True)
            assert first["start_row"] == 128
            assert second["start_row"] == 192
            assert first["correctness"]["passed"]
            assert second["correctness"]["passed"]
            assert first["seconds"] > 0
            assert first["phases_seconds"]["to_pyarrow"] >= 0
        finally:
            assert (await scenario.finish())["state"] == "completed"

    asyncio.run(exercise())


def test_warm_configuration_rejects_incomplete_ticks_and_invalid_sizes() -> None:
    for kwargs in ({"append_rows": 65}, {"history_rows": 0}, {"window": 0}):
        with pytest.raises(ValueError):
            ScenarioConfig(**kwargs)


@pytest.mark.parametrize("append_rows", (1, 4, 16, 64))
@pytest.mark.parametrize("append_entities", (1, 4, 16, 64))
@pytest.mark.parametrize("indicator", ("rolling_mean", "dual_sma_spread"))
def test_sparse_appends_keep_all_history_entities_and_finalize_each_batch(
    tmp_path, append_rows, append_entities, indicator
) -> None:
    async def exercise() -> None:
        config = ScenarioConfig(
            history_rows=128,
            append_rows=append_rows,
            append_entities=append_entities,
            indicator=indicator,
        )
        scenario = await WarmScenario.start(config, tmp_path)
        try:
            for index in range(3):
                sample = await scenario.sample(collect_gc=False)
                assert sample["start_row"] == 128 + index * append_rows
                assert sample["correctness"]["passed"]
            status = scenario.job.status()
            assert all(node["late_rows"] == 0 for node in status["operators"].values())
        finally:
            assert (await scenario.finish())["state"] == "completed"

    asyncio.run(exercise())


def test_sparse_configuration_rejects_invalid_active_entity_counts() -> None:
    for count in (0, -1, 65, 1.5, True):
        with pytest.raises(ValueError, match="append_entities"):
            ScenarioConfig(append_rows=1, append_entities=count)


def test_prepared_watermarks_do_not_need_a_timezone_database(monkeypatch) -> None:
    def missing_timezone(_name):
        raise zoneinfo.ZoneInfoNotFoundError("no timezone database")

    monkeypatch.setattr(zoneinfo, "ZoneInfo", missing_timezone)
    for active, rows in ((None, 64), (1, 1), (64, 4)):
        config = ScenarioConfig(
            history_rows=128, append_entities=active, append_rows=rows
        )
        data, watermark = _prepared_events(config, 128, rows)
        assert data.batch.num_rows == rows
        seconds = (128 + rows - 1) // 64 if active is None else 128 + rows - 1
        assert watermark.at == BASE + timedelta(seconds=seconds)


@pytest.mark.parametrize("indicator", ("rolling_mean", "dual_sma_spread"))
@pytest.mark.parametrize("active", (1, 4, 16, 64))
def test_sparse_oracle_matches_full_per_entity_history(indicator, active) -> None:
    config = ScenarioConfig(
        history_rows=128, append_rows=1, append_entities=active, indicator=indicator
    )
    total = config.history_rows + 32 * active + 7
    table = _input_table(config, 0, total).select(["symbol", "price"]).to_pydict()
    histories: dict[str, list[float]] = {}
    expected = []
    for symbol, price in zip(table["symbol"], table["price"], strict=True):
        values = histories.setdefault(symbol, [])
        values.append(price)
        slow = sum(values[-config.window :]) / len(values[-config.window :])
        fast = sum(values[-config.fast_window :]) / len(values[-config.fast_window :])
        expected.append(fast - slow if indicator == "dual_sma_spread" else slow)
    for start in (128, 129, 128 + 19 * active, 128 + 31 * active):
        rows = min(17, total - start)
        np.testing.assert_allclose(
            _expected(config, start=start, rows=rows),
            expected[start : start + rows],
            rtol=1e-10,
            atol=1e-10,
        )


def test_callback_diagnostics_are_opt_in_and_survive_terminal_cleanup(tmp_path) -> None:
    async def exercise() -> None:
        scenario = await WarmScenario.start(ScenarioConfig(history_rows=128), tmp_path)
        native = scenario.job._inner
        try:
            assert json.loads(native._take_callback_profile()) == {
                "records": [],
                "dropped": 0,
            }
            native._enable_callback_profiling()
            native._enable_callback_profiling()
            await scenario.sample(collect_gc=False)
        finally:
            assert (await scenario.finish())["state"] == "completed"
        profile = json.loads(native._take_callback_profile())
        assert profile["dropped"] == 0
        records = profile["records"]
        assert {record["callback"] for record in records} >= {
            "source.next",
            "_native_write",
        }
        for record in records:
            if record["outcome"] == "completed":
                assert 0 < record["attached_ns"] <= record["queued_ns"]
                assert record["queued_ns"] <= record["dispatched_ns"]
                assert (
                    record["dispatched_ns"]
                    <= record["completed_ns"]
                    <= record["elapsed_ns"]
                )
        assert json.loads(native._take_callback_profile())["records"] == []

    asyncio.run(exercise())
