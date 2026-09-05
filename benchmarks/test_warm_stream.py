"""Correctness and accounting guards for warm StreamingRunner measurements."""

from __future__ import annotations

import asyncio

import numpy as np
import pyarrow as pa
import pytest

from benchmarks.warm_stream import (
    ScenarioConfig,
    WarmScenario,
    _expected,
    _segment,
    _summary,
    _validate_output,
)


def test_warm_oracle_uses_only_window_context() -> None:
    for indicator in ("rolling_mean", "dual_sma_spread"):
        values = _expected(
            start=1_024_000,
            rows=64,
            entities=64,
            indicator=indicator,
            fast_window=5,
            window=20,
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
    expected = _expected(
        start=1_024,
        rows=64,
        entities=64,
        indicator="rolling_mean",
        fast_window=5,
        window=20,
    )
    table = _segment(1_024, 64, 64).append_column("moving_average", pa.array(expected))
    kwargs = dict(
        start=1_024,
        rows=64,
        entities=64,
        indicator="rolling_mean",
        fast_window=5,
        window=20,
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
