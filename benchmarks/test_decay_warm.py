"""Finite weighted-decay oracle and native-witness guards for P2 profiling."""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
from copy import deepcopy
from types import SimpleNamespace
from unittest.mock import AsyncMock

import numpy as np
import pytest

from benchmarks import warm_stream
from scripts.benchmark_suite import decay_warm


@pytest.mark.parametrize("value", ["nan", "inf", "-inf", "70", "0", "-1", "601"])
def test_decay_phase_budget_rejects_invalid_resource_limits(value):
    with pytest.raises(argparse.ArgumentTypeError, match="budget"):
        decay_warm.phase_budget(value)


def test_decay_cli_keeps_the_original_default_budget(tmp_path, monkeypatch):
    collector = AsyncMock()
    monkeypatch.setattr(decay_warm, "collect", collector)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "decay_warm",
            "--baseline-build",
            "baseline.json",
            "--candidate-build",
            "candidate.json",
            "--output",
            str(tmp_path),
        ],
    )
    decay_warm.main()
    assert collector.call_args.args[0].budget_seconds == 600


@pytest.mark.parametrize("budget,acquire", [(600, 530), (574.654709, 504.654709)])
def test_decay_phase_budget_preserves_seventy_seconds_for_cleanup(
    budget, acquire, tmp_path, monkeypatch
):
    async def cases(_args, _manifests, report):
        report["cases"] = [
            {"paired_speedup_median": 1.0, "paired_speedup_ci95": [0.99, 1.01]}
            for _ in range(4)
        ]

    monkeypatch.setattr(
        decay_warm.profile, "_compatible_builds", AsyncMock(return_value=[])
    )
    monkeypatch.setattr(decay_warm, "harness_identity", lambda: {})
    monkeypatch.setattr(decay_warm.psutil, "process_iter", lambda *_args: [])
    monkeypatch.setattr(decay_warm, "collect_cases", cases)
    output = tmp_path / "collection"
    asyncio.run(
        decay_warm.collect(
            SimpleNamespace(
                output=output,
                baseline_build=tmp_path / "baseline.json",
                candidate_build=tmp_path / "candidate.json",
                budget_seconds=budget,
            )
        )
    )
    protocol = json.loads((output / "protocol.json").read_text())
    assert protocol["budget_seconds"] == budget
    assert protocol["acquisition_timeout_seconds"] == acquire
    assert protocol["cleanup_reserve_seconds"] == 70


@pytest.mark.parametrize("failure", ["missing", "nonzero", "unreaped"])
def test_decay_collection_rejects_missing_or_failed_worker_receipts(
    failure, tmp_path, monkeypatch
):
    clean = {"pid": 41, "exit_code": 0, "reaped": True, "close_result": "normal"}
    bad = {**clean, "pid": 42}
    if failure == "nonzero":
        bad["exit_code"] = 7
    if failure == "unreaped":
        bad["reaped"] = False
    receipts = None if failure == "missing" else [clean, bad]

    async def measured_case(_manifests, config, *_options):
        counters = dict.fromkeys(decay_warm.WORK_FIELDS, 0)
        expected = config["history_rows"] + 30 * config["append_rows"]
        counters.update(transitions=expected, certified_rows=expected)
        empty = {"before_status": {}, "after_status": {}}
        candidate = {
            "before_status": {},
            "after_status": {
                "rolling_metrics": {"roll": {"data": {"_decay_work": counters}}}
            },
        }
        return {
            "worker_receipts": receipts,
            "terminal": [empty, candidate],
            "paired_speedup_median": 1.1,
        }, []

    monkeypatch.setattr(
        decay_warm.profile,
        "_fresh_case",
        AsyncMock(side_effect=measured_case),
    )
    with pytest.raises((ValueError, RuntimeError), match="worker"):
        asyncio.run(
            decay_warm.collect_cases(
                SimpleNamespace(output=tmp_path), [], {"cases": []}
            )
        )


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


def test_native_decay_witness_rejects_missing_copy_counter():
    counters = dict.fromkeys(decay_warm.WORK_FIELDS, 0)
    counters.update(transitions=1, certified_rows=1)
    del counters["cache_copied_samples"]
    status = {"rolling_metrics": {"roll": {"data": {"_decay_work": counters}}}}
    with pytest.raises(ValueError, match="fields"):
        decay_warm.total_work(status)


def test_native_decay_witness_rejects_overflowed_node():
    counters = dict.fromkeys(decay_warm.WORK_FIELDS, 0)
    counters.update(transitions=1, certified_rows=1)
    status = {
        "rolling_metrics": {
            "roll": {"overflowed": True, "data": {"_decay_work": counters}}
        }
    }
    with pytest.raises(ValueError, match="overflow"):
        decay_warm.total_work(status)


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
