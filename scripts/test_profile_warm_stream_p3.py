"""Focused guards for the fixed P3 warm case adapter, without native acquisition."""

from __future__ import annotations

import asyncio
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

import pytest

from benchmarks import warm_stream as warm
from scripts import profile_warm_stream as profile


def test_p3_scalar_cli_dimensions_select_only_the_explicit_cases() -> None:
    arguments = [
        "profile_warm_stream.py",
        "compare",
        "--baseline-build",
        "left.json",
        "--candidate-build",
        "right.json",
        "--output",
        "result.json",
        "--history-rows",
        "1000",
        "--append-rows",
        "1024",
        "--append-entities",
        "1",
        "--indicators",
        "dual_sma_spread",
        "rolling_mean",
        "--gc-modes",
        "normal",
        "--entities",
        "1",
        "--window",
        "60",
        "--fast-window",
        "20",
        "--history-segment-rows",
        "1000",
    ]
    compared = AsyncMock()
    with (
        patch.object(profile.sys, "argv", arguments),
        patch.object(profile, "compare", compared),
    ):
        profile.main()
    (options,) = compared.call_args.args
    assert options.output == Path("result.json")
    cases = profile.matrix_cases(options)
    assert len(cases) == 2
    for config, mode in cases:
        assert mode == "normal"
        assert config["history_rows"] == 1000
        assert config["append_rows"] == 1024
        assert config["append_entities"] == 1
        assert {
            key: config[key]
            for key in ("entities", "window", "fast_window", "history_segment_rows")
        } == {
            "entities": 1,
            "window": 60,
            "fast_window": 20,
            "history_segment_rows": 1000,
        }


def _config() -> dict:
    return {
        "history_rows": 1000,
        "append_rows": 1024,
        "append_entities": 1,
        "entities": 1,
        "window": 60,
        "fast_window": 20,
        "history_segment_rows": 1000,
        "indicator": "dual_sma_spread",
    }


def _start(config: dict) -> dict:
    return {
        "config": dict(config),
        "warm_seconds": 0.1,
        "float64_digest_format": warm.FLOAT64_DIGEST_FORMAT,
        "warm_output_digest": "a" * 64,
        "plan_identity": {
            "program_fingerprint": "d" * 64,
            "plan_fingerprint": "e" * 64,
            "project_sha256": "f" * 64,
            "preflight": {"engine": "streaming"},
        },
    }


def _sample(index: int) -> dict:
    return {
        "start_row": 1000 + 1024 * index,
        "seconds": 0.004,
        "correctness": {"passed": True, "float64_digest": "b" * 64},
        "phases_seconds": {name: 0.001 for name in profile.PHASE_NAMES},
    }


def _worker(start: dict, sample_digest: str | None = "b" * 64):
    measured = []

    async def request(**message):
        operation = message["operation"]
        if operation == "start":
            return start
        if operation == "sample":
            sample = _sample(len(measured))
            if sample_digest is None:
                del sample["correctness"]["float64_digest"]
            else:
                sample["correctness"]["float64_digest"] = sample_digest
            measured.append(sample)
            return sample
        return {"state": "completed"}

    return SimpleNamespace(request=AsyncMock(side_effect=request), measured=measured)


def test_p3_worker_start_acknowledges_expanded_config_and_warm_digest(tmp_path) -> None:
    async def exercise():
        config = _config()
        scenario = SimpleNamespace(
            warm_seconds=0.1,
            warm_output_digest="a" * 64,
            plan_identity=_start(config)["plan_identity"],
        )
        with patch.object(warm.WarmScenario, "start", AsyncMock(return_value=scenario)):
            response = await profile.WorkerSession(tmp_path).start(config)
        assert response == _start(config)

    asyncio.run(exercise())


@pytest.mark.parametrize(
    "mismatch", ("warm", "format", "config", "plan", "sample", "missing_sample")
)
def test_p3_paired_controller_rejects_inexact_results_even_when_oracle_passes(
    mismatch,
) -> None:
    async def exercise():
        config = _config()
        right = _start(config)
        if mismatch == "warm":
            right["warm_output_digest"] = "c" * 64
        elif mismatch == "format":
            right["float64_digest_format"] = "unversioned"
        elif mismatch == "config":
            right["config"]["entities"] = 64
        elif mismatch == "plan":
            right["plan_identity"]["plan_fingerprint"] = "0" * 64
        digest = {"sample": "c" * 64, "missing_sample": None}.get(mismatch, "b" * 64)
        workers = [_worker(_start(config)), _worker(right, digest)]
        with pytest.raises(ValueError, match="paired|digest|configuration"):
            await profile._measure_case(workers, config, 2, False)
        if mismatch in ("warm", "format", "config", "plan"):
            assert all(not worker.measured for worker in workers)

    asyncio.run(exercise())


def test_p3_matching_digests_retain_raw_receipts_and_alternating_pairs() -> None:
    async def exercise():
        config = _config()
        workers = [_worker(_start(config)), _worker(_start(config))]
        order = []

        def recorded(side, request):
            async def call(**message):
                order.append((side, message["operation"]))
                return await request(**message)

            return call

        for side, worker in enumerate(workers):
            worker.request = recorded(side, worker.request)
        result = await profile._measure_case(workers, config, 2, False)
        assert result["start"] == [_start(config), _start(config)]
        assert result["samples"] == [[_sample(0), _sample(1)]] * 2
        assert result["config"] == config
        assert result["paired_speedup_median"] == 1.0
        assert order == [
            (0, "start"),
            (1, "start"),
            (0, "sample"),
            (1, "sample"),
            (1, "sample"),
            (0, "sample"),
            (0, "finish"),
            (1, "finish"),
        ]

    asyncio.run(exercise())


@pytest.mark.parametrize("indicator", ("rolling_decay", "decay_mean"))
def test_decay_controller_preserves_existing_value_oracle_acceptance(indicator) -> None:
    async def exercise():
        config = {**_config(), "indicator": indicator, "append_entities": None}
        right = {**_start(config), "warm_output_digest": "c" * 64}
        workers = [_worker(_start(config)), _worker(right, "c" * 64)]
        result = await profile._measure_case(workers, config, 2, False)
        assert (
            result["start"][0]["warm_output_digest"]
            != result["start"][1]["warm_output_digest"]
        )
        assert result["samples"][0][0]["correctness"] == {
            "passed": True,
            "float64_digest": "b" * 64,
        }
        assert result["samples"][1][0]["correctness"] == {
            "passed": True,
            "float64_digest": "c" * 64,
        }

    asyncio.run(exercise())


def test_p3_explicit_t2_cli_does_not_add_an_implicit_history_workload() -> None:
    arguments = [
        "profile_warm_stream.py",
        "compare",
        "--baseline-build",
        "left.json",
        "--candidate-build",
        "right.json",
        "--output",
        "result.json",
        "--history-rows",
        "600000",
        "--append-rows",
        "1000",
        "--append-entities",
        "1000",
        "--indicators",
        "dual_sma_spread",
        "--gc-modes",
        "normal",
        "--entities",
        "10000",
        "--window",
        "60",
        "--fast-window",
        "20",
        "--history-segment-rows",
        "1000",
    ]
    compared = AsyncMock()
    with (
        patch.object(profile.sys, "argv", arguments),
        patch.object(profile, "compare", compared),
    ):
        profile.main()
    (options,) = compared.call_args.args
    assert profile.matrix_cases(options) == [
        (
            {
                **_config(),
                "history_rows": 600_000,
                "append_rows": 1000,
                "append_entities": 1000,
                "entities": 10_000,
            },
            "normal",
        )
    ]


@pytest.mark.parametrize(
    "flag", ("entities", "window", "fast-window", "history-segment-rows")
)
def test_p3_scalar_flags_reject_nonpositive_sizes_before_comparison(flag) -> None:
    arguments = [
        "profile_warm_stream.py",
        "compare",
        "--baseline-build",
        "left.json",
        "--candidate-build",
        "right.json",
        "--output",
        "result.json",
        f"--{flag}",
        "0",
    ]
    compared = AsyncMock()
    with (
        patch.object(profile.sys, "argv", arguments),
        patch.object(profile, "compare", compared),
        pytest.raises(SystemExit) as failure,
    ):
        profile.main()
    assert failure.value.code == 2
    compared.assert_not_awaited()
