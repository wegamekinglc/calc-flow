"""Focused guards for the fixed P3 warm case adapter, without native acquisition."""

from __future__ import annotations

import asyncio
import json
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
        "exact_projection": False,
    }


def _start(config: dict, driver: str = "runner") -> dict:
    return {
        "config": dict(config),
        "driver": driver,
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


def _protocol_arguments(*cases: str, extra: tuple[str, ...] = ()) -> list[str]:
    arguments = [
        "profile_warm_stream.py",
        "compare",
        "--baseline-build",
        "left.json",
        "--candidate-build",
        "right.json",
        "--output",
        "result.json",
        "--samples",
        "30",
        "--seed",
        "20261115",
        "--budget-seconds",
        "600",
        "--history-rows",
        "4096",
        "--history-segment-rows",
        "4096",
        "--window",
        "20",
        "--fast-window",
        "5",
        "--indicators",
        "dual_sma_spread",
        "--gc-modes",
        "forced",
        "--exact-projection",
        *extra,
    ]
    for case in cases:
        arguments.extend(("--case", case))
    return arguments


PROTOCOL_CASES = (
    "name=1,append_rows=1,entities=1,drivers=runner:session,"
    "reference=ta_lib,p50_limit_us=20",
    "name=2,append_rows=99,entities=1,drivers=runner:session",
    "name=3,append_rows=6400,entities=64,drivers=runner:session,reference=ta_lib",
    "name=4,append_rows=1,entities=1,drivers=runner:runner,regression_below=0.95",
)


def _parsed(arguments: list[str]):
    compared = AsyncMock()
    with (
        patch.object(profile.sys, "argv", arguments),
        patch.object(profile, "compare", compared),
    ):
        profile.main()
    (options,) = compared.call_args.args
    return options


def test_explicit_cases_select_only_the_protocol_points_and_arms() -> None:
    options = _parsed(_protocol_arguments(*PROTOCOL_CASES))
    selected = profile.selected_cases(options)
    assert selected == profile.selected_cases(options)
    by_name = {arms["name"]: (config, mode, arms) for config, mode, arms in selected}
    assert sorted(by_name) == ["1", "2", "3", "4"]
    for name, rows, entities, drivers in (
        ("1", 1, 1, ("runner", "session")),
        ("2", 99, 1, ("runner", "session")),
        ("3", 6400, 64, ("runner", "session")),
        ("4", 1, 1, ("runner", "runner")),
    ):
        config, mode, arms = by_name[name]
        assert mode == "forced"
        assert arms["drivers"] == drivers
        # No implicit H=1,024,000 matrix point is crossed into explicit cases.
        assert config == {
            "history_rows": 4096,
            "append_rows": rows,
            "entities": entities,
            "indicator": "dual_sma_spread",
            "exact_projection": True,
            "window": 20,
            "fast_window": 5,
            "history_segment_rows": 4096,
        }
    assert [by_name[name][2]["reference"] for name in "1234"] == [
        "ta_lib",
        None,
        "ta_lib",
        None,
    ]
    assert by_name["1"][2]["p50_limit_seconds"] == pytest.approx(20e-6)
    assert by_name["4"][2]["regression_below"] == 0.95
    assert options.budget_seconds == 600


@pytest.mark.parametrize(
    "arguments",
    (
        _protocol_arguments(PROTOCOL_CASES[0], extra=("--append-rows", "1")),
        _protocol_arguments(PROTOCOL_CASES[0], extra=("--entities", "1")),
        _protocol_arguments(PROTOCOL_CASES[0], extra=("--append-entities", "1")),
        _protocol_arguments(PROTOCOL_CASES[0], extra=("--worker-threads", "1", "4")),
        _protocol_arguments(PROTOCOL_CASES[0], extra=("--history-rows", "1", "2")),
        _protocol_arguments(PROTOCOL_CASES[0], PROTOCOL_CASES[0]),
        _protocol_arguments("name=1,append_rows=1,entities=1"),
        _protocol_arguments("name=1,append_rows=1,entities=1,drivers=runner"),
        _protocol_arguments("name=1,append_rows=1,entities=1,drivers=runner:batch"),
        _protocol_arguments("name=1,append_rows=0,entities=1,drivers=runner:runner"),
        _protocol_arguments(PROTOCOL_CASES[1] + ",reference=polars"),
        _protocol_arguments(PROTOCOL_CASES[1] + ",p50_limit_us=-1"),
        _protocol_arguments(PROTOCOL_CASES[1] + ",unknown=1"),
        _protocol_arguments(PROTOCOL_CASES[1] + ",name=5"),
        _protocol_arguments("name=a b,append_rows=1,entities=1,drivers=runner:runner"),
        [
            argument
            for argument in _protocol_arguments(PROTOCOL_CASES[1])
            if argument != "--exact-projection"
        ],
    ),
)
def test_explicit_cases_reject_matrix_options_and_malformed_fields(arguments) -> None:
    compared = AsyncMock()
    with (
        patch.object(profile.sys, "argv", arguments),
        patch.object(profile, "compare", compared),
        pytest.raises(SystemExit) as failure,
    ):
        profile.main()
    assert failure.value.code == 2
    compared.assert_not_awaited()


def _session_sample(index: int, *, attributed: bool = True) -> dict:
    sample = {
        "start_row": 1000 + 1024 * index,
        "seconds": 0.000015,
        "correctness": {"passed": True, "float64_digest": "b" * 64},
        "phases_seconds": {},
        "phase_attribution": "unavailable: observed 0 matching native calls",
    }
    if attributed:
        sample["phases_seconds"] = {
            "argument_handling": 0.000002,
            "native_call": 0.00001,
            "result_assembly": 0.000003,
        }
        sample["phase_attribution"] = "native_call_probe"
    return sample


def _session_worker(start: dict, samples: list[dict]):
    async def request(**message):
        if message["operation"] == "start":
            return start
        if message["operation"] == "sample":
            return samples.pop(0)
        return {"state": "completed"}

    return SimpleNamespace(request=AsyncMock(side_effect=request))


def test_session_pair_summarizes_driver_phases_and_runs_the_reference_last() -> None:
    async def exercise():
        config = _config()
        baseline = _worker(_start(config))
        reference = {"samples_seconds": [0.000006, 0.000007], "median_seconds": 6.5e-6}
        original = baseline.request.side_effect

        async def baseline_request(**message):
            if message["operation"] == "ta_lib_reference":
                assert message["config"] == config
                assert (message["samples"], message["collect_gc"]) == (2, True)
                return reference
            return await original(**message)

        baseline.request = AsyncMock(side_effect=baseline_request)
        candidate = _session_worker(
            _start(config, "session"),
            [_session_sample(0), _session_sample(1, attributed=False)],
        )
        result = await profile._measure_case(
            [baseline, candidate],
            config,
            2,
            True,
            options=(("runner", "session"), "ta_lib", False),
        )
        assert result["drivers"] == ["runner", "session"]
        starts = [call.kwargs for call in candidate.request.await_args_list[:1]]
        assert starts == [{"operation": "start", "config": config, "driver": "session"}]
        assert baseline.request.await_args_list[-1].kwargs["operation"] == (
            "ta_lib_reference"
        )
        assert result["reference"] == {"kind": "ta_lib", **reference}
        phases = result["phase_intervals"]["candidate"]
        assert phases["attributed_samples"] == 1
        assert phases["unattributed_reasons"] == [
            "unavailable: observed 0 matching native calls"
        ]
        assert phases["phases"]["native_call"]["p50_seconds"] == pytest.approx(1e-5)
        assert set(result["phase_intervals"]["baseline"]) == set(profile.PHASE_NAMES)

    asyncio.run(exercise())


@pytest.mark.parametrize(
    "mismatch", ("driver", "uncovered", "unexplained", "reference")
)
def test_session_pair_rejects_wrong_driver_phases_or_reference(mismatch) -> None:
    async def exercise():
        config = _config()
        start = _start(config, "runner" if mismatch == "driver" else "session")
        sample = _session_sample(0)
        if mismatch == "uncovered":
            sample["phases_seconds"]["native_call"] = 1.0
        elif mismatch == "unexplained":
            sample["phase_attribution"] = "missing"
        candidate = _session_worker(start, [sample, _session_sample(1)])
        baseline = _worker(_start(config))
        with pytest.raises(ValueError, match="driver|session|reference"):
            await profile._measure_case(
                [baseline, candidate],
                config,
                2,
                False,
                options=(
                    ("runner", "session"),
                    "ta_lib" if mismatch == "reference" else None,
                    False,
                ),
            )

    asyncio.run(exercise())


@pytest.mark.parametrize(
    ("ci", "arms", "expected"),
    (
        ((0.8, 0.99), {}, {"paired_speedup": "slower"}),
        ((0.9, 1.1), {}, {"paired_speedup": "inconclusive"}),
        ((1.5, 40.0), {}, {"paired_speedup": "faster"}),
        (
            (0.9, 0.96),
            {"regression_below": 0.95},
            {
                "paired_speedup": "slower",
                "regression_below": 0.95,
                "confirmed_regression": False,
            },
        ),
        (
            (0.8, 0.94),
            {"regression_below": 0.95},
            {
                "paired_speedup": "slower",
                "regression_below": 0.95,
                "confirmed_regression": True,
            },
        ),
    ),
)
def test_case_verdicts_apply_the_preregistered_rules(ci, arms, expected) -> None:
    case = {"paired_speedup_ci95": list(ci), "candidate": {"p50_seconds": 1.9e-5}}
    assert profile.case_verdict(case, {**profile.MATRIX_ARMS, **arms}) == expected
    limited = {**profile.MATRIX_ARMS, "p50_limit_seconds": 2e-5}
    assert profile.case_verdict(case, limited)["candidate_p50_within_limit"]
    case["candidate"]["p50_seconds"] = 2.1e-5
    assert not profile.case_verdict(case, limited)["candidate_p50_within_limit"]


def _table_blocks(text: str) -> list[list[str]]:
    blocks: list[list[str]] = []
    previous = False
    for line in text.splitlines():
        table = line.startswith("|")
        if table and not previous:
            blocks.append([])
        if table:
            blocks[-1].append(line)
        previous = table
    return blocks


def _assert_aligned_tables(text: str) -> None:
    for block in _table_blocks(text):
        assert len({len(line) for line in block}) == 1
        assert set(block[1]) == {"|", "-"}


def _fixture_workers(drivers: tuple[str, str]) -> list:
    """Return fake workers for the shared fixture rows under the given drivers."""
    if drivers[1] == "runner":
        return [_worker(_start(_config())), _worker(_start(_config()))]
    return [
        _worker(_start(_config())),
        _session_worker(
            _start(_config(), "session"), [_session_sample(0), _session_sample(1)]
        ),
    ]


def test_budget_stop_settles_the_running_case_and_reports_unmeasured_cases(
    tmp_path,
) -> None:
    output = tmp_path / "report" / "p1.json"
    options = _parsed(
        [
            *_protocol_arguments(PROTOCOL_CASES[0], PROTOCOL_CASES[3]),
            "--output",
            str(output),
            "--work-dir",
            str(tmp_path / "work"),
        ]
    )
    options.budget_seconds = 0.5
    first, *_ = profile.selected_cases(options)
    settled = []

    async def fresh_case(manifests, config, samples, collect_gc, pair):
        # Cases 1 and 4 share a workload; the drivers tell them apart.
        if pair.drivers != first[2]["drivers"]:
            try:
                await asyncio.sleep(30)
            finally:
                settled.append(pair.drivers)
        # Fake workers follow the shared fixture rows, not the protocol sizes.
        case = await profile._measure_case(
            _fixture_workers(pair.drivers),
            _config(),
            2,
            collect_gc,
            options=(pair.drivers, None, False),
        )
        assert pair.directory == (tmp_path / "work").resolve()
        return case, []

    async def exercise():
        with (
            patch.object(profile, "_compatible_builds", AsyncMock(return_value=[])),
            patch.object(profile, "_fresh_case", fresh_case),
        ):
            await profile.compare(options)

    asyncio.run(exercise())
    report = json.loads(output.read_text(encoding="utf-8"))
    measured, remaining = first[2]["name"], {"1": "4", "4": "1"}[first[2]["name"]]
    assert report["status"] == "stopped"
    assert report["case_selection"] == "explicit"
    assert [case["name"] for case in report["cases"]] == [measured]
    assert [case["name"] for case in report["unmeasured_cases"]] == [remaining]
    assert len(settled) == 1
    # Case 4 pairs equal runner timings; case 1's fixture session is faster.
    assert (
        report["cases"][0]["verdict"]["paired_speedup"]
        == ({"1": "faster", "4": "inconclusive"}[measured])
    )
    text = output.with_suffix(".md").read_text(encoding="utf-8")
    assert "Status: stopped" in text
    assert "Unmeasured after the budget stop" in text
    _assert_aligned_tables(text)


def test_case_timeouts_without_a_budget_are_not_budget_stops(tmp_path) -> None:
    options = _parsed(
        [
            *_protocol_arguments(PROTOCOL_CASES[3]),
            "--output",
            str(tmp_path / "p1.json"),
        ]
    )
    options.budget_seconds = None

    async def exercise():
        with (
            patch.object(profile, "_compatible_builds", AsyncMock(return_value=[])),
            patch.object(
                profile, "_fresh_case", AsyncMock(side_effect=TimeoutError("worker"))
            ),
            pytest.raises(TimeoutError, match="worker"),
        ):
            await profile.compare(options)

    asyncio.run(exercise())
