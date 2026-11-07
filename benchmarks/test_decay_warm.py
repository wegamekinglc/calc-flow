"""Finite weighted-decay oracle and native-witness guards for P2 profiling."""

from __future__ import annotations

import asyncio
from copy import deepcopy

import numpy as np
import pytest

from benchmarks import warm_stream
from scripts.benchmark_suite import decay_warm


def test_decay_oracle_keeps_right_aligned_warmup_weights():
    values = np.array([1.0, 2.0, 3.0, 4.0])
    actual = warm_stream.expected_decay(values, entities=1, window=3)
    np.testing.assert_allclose(actual, [1.0, 8 / 5, 14 / 6, 20 / 6], rtol=1e-12)


def test_decay_case_selection_fills_the_large_window_and_keeps_four_cases():
    assert len(decay_warm.CASES) == 4
    for case in decay_warm.CASES:
        assert case["entities"] == 1
        assert case["history_rows"] == case["history_segment_rows"] == 1000
        assert case["history_rows"] // case["entities"] >= case["window"]
    assert [(case["window"], case["append_rows"]) for case in decay_warm.CASES] == [
        (20, 1024),
        (1000, 1024),
        (1000, 1024),
        (1000, 1),
    ]


def test_native_decay_witness_separates_arithmetic_and_copy_work():
    counters = dict.fromkeys(decay_warm.WORK_FIELDS, 0)
    counters.update(transitions=10, certified_rows=10, cache_copied_samples=1000)
    before = {"rolling_metrics": {"roll": {"data": {"_decay_work": counters}}}}
    after = deepcopy(before)
    final = after["rolling_metrics"]["roll"]["data"]["_decay_work"]
    final.update(
        transitions=12,
        certified_rows=11,
        scheduled_reseed_rows=1,
        scheduled_reseed_samples=1000,
        cache_copied_samples=3000,
    )
    work = decay_warm.work_witness({"before_status": before, "after_status": after})
    assert work["available"]
    assert work["delta"]["transitions"] == 2
    assert work["delta"]["certified_rows"] == 1
    assert work["delta"]["scheduled_reseed_samples"] == 1000
    assert work["delta"]["cache_copied_samples"] == 2000
    assert decay_warm.work_witness({"before_status": {}, "after_status": {}}) == {
        "available": False,
        "before": None,
        "after": None,
        "delta": None,
    }


def test_decay_and_mixed_mean_advance_the_old_reference_runtime(tmp_path):
    async def exercise(indicator, directory):
        config = warm_stream.ScenarioConfig(
            history_rows=8,
            history_segment_rows=8,
            append_rows=4,
            entities=1,
            window=3,
            fast_window=1,
            indicator=indicator,
        )
        scenario = await warm_stream.WarmScenario.start(config, directory)
        try:
            assert (await scenario.sample(collect_gc=False))["correctness"]["passed"]
        finally:
            assert (await scenario.finish())["state"] == "completed"

    for indicator in ("rolling_decay", "decay_mean"):
        asyncio.run(exercise(indicator, tmp_path / indicator))


def test_decay_acceptance_requires_a_target_win_and_no_case_regression():
    cases = [
        {"paired_speedup_median": 1.0, "paired_speedup_ci95": [0.99, 1.01]}
        for _ in range(4)
    ]
    assert not decay_warm.acceptance(cases)["accepted"]
    cases[1] = {"paired_speedup_median": 1.10, "paired_speedup_ci95": [1.02, 1.20]}
    assert decay_warm.acceptance(cases)["accepted"]
    cases[3] = {"paired_speedup_median": 0.90, "paired_speedup_ci95": [0.88, 0.93]}
    assert not decay_warm.acceptance(cases)["accepted"]
    assert decay_warm.acceptance(cases)["confirmed_regressions"] == [3]


def test_decay_acceptance_rejects_a_missing_copy_risk_control():
    cases = [
        {"paired_speedup_median": 1.10, "paired_speedup_ci95": [1.02, 1.20]}
        for _ in range(3)
    ]
    with pytest.raises(ValueError, match="four"):
        decay_warm.acceptance(cases)
