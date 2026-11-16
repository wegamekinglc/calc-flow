"""Guards for pairing release native-evidence binaries, without running them."""

from __future__ import annotations

import asyncio
import json
import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock, patch

import numpy as np
import pytest

from scripts import profile_warm_stream as profile

SAMPLES = 30
DISPATCH_CASES = (
    "1x1-dual",
    "99x1-dual",
    "6400x64-dual",
    "64x64-dual-tick",
    "6400x64-dual-lag1",
)


def _dispatch_case(name: str, rows: int, seconds: float, numeric_ns: float) -> dict:
    return {
        "name": name,
        "kind": "dispatch",
        "rows": rows,
        "window_outputs": 2,
        "history_rows": profile.NATIVE_HISTORY_ROWS,
        "warm_dispatches": profile.NATIVE_WARM_DISPATCHES,
        "workload": {
            "rows": rows,
            "entities": profile.NATIVE_CASE_ENTITIES[name],
            "window": 20,
            "outputs": profile.NATIVE_CASE_OUTPUTS[name],
            "min_periods": "full window",
            "input": "Float64 exact binary fractions",
            "first_payload_bytes": 0,
        },
        "samples": [
            {
                "index": index,
                "order": "AB" if index % 2 == 0 else "BA",
                "absent_seconds": seconds * (1 + index / 100),
                "present_seconds": seconds * (2 + index / 100),
                "present_numeric_update_seconds": rows * 2 * numeric_ns * 1e-9,
            }
            for index in range(SAMPLES)
        ],
    }


def _evidence(cases: tuple[str, ...], scale: float) -> dict:
    dimensions = {
        "1x1-dual": (1, 1e-6),
        "99x1-dual": (99, 2e-6),
        "6400x64-dual": (6_400, 4e-4),
        "64x64-dual-tick": (64, 4e-6),
        "6400x64-dual-lag1": (6_400, 5e-4),
    }
    return {
        "contract": profile.NATIVE_CONTRACT,
        "executable": "/bin/evidence",
        "debug_assertions": False,
        "selection": list(cases),
        "timed_samples": SAMPLES,
        "cases": [
            _dispatch_case(
                name, dimensions[name][0], dimensions[name][1] * scale, 6 * scale
            )
            for name in cases
        ],
    }


def _write_evidence(output: Path, evidence: dict) -> dict:
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(evidence), encoding="utf-8")
    return evidence


def _args(tmp_path: Path, cases: tuple[str, ...], rounds: int = 4) -> SimpleNamespace:
    binaries = []
    for name in ("baseline", "candidate"):
        binary = tmp_path / name
        binary.write_bytes(name.encode())
        binary.chmod(0o755)
        binaries.append(binary)
    return SimpleNamespace(
        baseline_binary=binaries[0],
        candidate_binary=binaries[1],
        cases=cases,
        rounds=rounds,
        output=tmp_path / "report" / "native.json",
        work_dir=tmp_path / "work",
        budget_seconds=None,
        numeric_contract_change=False,
    )


def _assert_native_calls(calls: list[tuple[str, str]]) -> None:
    assert [name for name, _ in calls] == [
        "baseline",
        "candidate",
        "candidate",
        "baseline",
    ] * 2
    assert len({output for _, output in calls}) == 8
    assert [output for _, output in calls] == [
        "round-1-baseline.json",
        "round-1-candidate.json",
        "round-2-candidate.json",
        "round-2-baseline.json",
        "round-3-baseline.json",
        "round-3-candidate.json",
        "round-4-candidate.json",
        "round-4-baseline.json",
    ]


def _assert_native_report(report: dict, args: SimpleNamespace) -> None:
    assert report["contract"] == "paired-native-evidence-v1"
    assert report["status"] == "completed"
    assert report["completed_rounds"] == 4
    assert [binary["sha256"] for binary in report["binaries"]] == [
        profile.sha256_file(args.baseline_binary),
        profile.sha256_file(args.candidate_binary),
    ]
    arms = {(arm["case"], arm["arm"]): arm for arm in report["arms"]}
    assert set(arms) == {
        (name, arm) for name in DISPATCH_CASES for arm in ("absent", "present")
    }
    absent = arms[("1x1-dual", "absent")]
    expected = np.median([1e-6 * (1 + index / 100) for index in range(SAMPLES)])
    assert absent["baseline"]["p50_seconds"] == pytest.approx(expected)
    assert absent["candidate"]["p50_seconds"] == pytest.approx(expected / 2)
    assert absent["paired_round_speedup_median"] == pytest.approx(2.0)
    assert absent["paired_round_speedup_ci95"] == pytest.approx([2.0, 2.0])
    assert absent["rounds"] == 4
    assert absent["verdict"] == {
        "paired_speedup": "faster",
        "confirmed_regression": False,
    }
    present = arms[("6400x64-dual", "present")]
    assert present["numeric_update_ns_per_window_row"] == pytest.approx(
        {"baseline": 6.0, "candidate": 3.0}
    )
    gates = report["gate_values"]
    assert gates["G1"]["value_seconds"] == pytest.approx(expected / 2)
    assert gates["G1"]["met"] is True
    assert gates["G2"]["met"] is True
    assert gates["G3"]["met"] is False


def _assert_aligned_markdown_tables(markdown: str) -> None:
    assert "| 1x1-dual" in markdown
    assert "Paired-round speedup" in markdown
    tables = [line for line in markdown.splitlines() if line.startswith("|")]
    assert tables
    blocks: list[list[str]] = [[]]
    for line in markdown.splitlines():
        if line.startswith("|"):
            blocks[-1].append(line)
        elif blocks[-1]:
            blocks.append([])
    for block in filter(None, blocks):
        assert len({len(line) for line in block}) == 1
        assert set(block[1]) == {"|", "-"}


def test_native_subcommand_parses_its_arguments_and_runs_the_pairing(
    tmp_path,
) -> None:
    argv = [
        "profile_warm_stream.py",
        "native",
        "--baseline-binary",
        str(tmp_path / "a"),
        "--candidate-binary",
        str(tmp_path / "b"),
        "--cases",
        ",".join(DISPATCH_CASES),
        "--output",
        str(tmp_path / "native.json"),
        "--budget-seconds",
        "590",
        "--numeric-contract-change",
    ]
    with (
        patch.object(sys, "argv", argv),
        patch.object(profile, "native", AsyncMock()) as native,
    ):
        profile.main()
    args = native.await_args.args[0]
    assert args.cases == DISPATCH_CASES
    assert (args.rounds, args.budget_seconds) == (10, 590.0)
    assert args.numeric_contract_change is True
    assert args.baseline_binary == tmp_path / "a"


@pytest.mark.parametrize(
    ("text", "message"),
    (
        ("1x1-dual,1x2-dual", "unknown"),
        ("session-g0", "unknown"),
        ("1x1-dual,", "empty"),
        ("99x1-dual,99x1-dual", "more than once"),
        (",".join((*DISPATCH_CASES, "1x1-dual")), "more than once"),
    ),
)
def test_native_case_selector_rejects_unknown_empty_and_duplicate_names(
    text, message
) -> None:
    with pytest.raises(profile.argparse.ArgumentTypeError, match=message):
        profile.parse_native_cases(text)


def test_native_rounds_alternate_binaries_and_pair_round_medians(tmp_path) -> None:
    args = _args(tmp_path, DISPATCH_CASES)
    calls = []

    async def run(binary, selected, output):
        calls.append((binary.name, output.name))
        assert selected == DISPATCH_CASES
        evidence = _evidence(DISPATCH_CASES, 1.0 if binary.name == "baseline" else 0.5)
        return _write_evidence(output, evidence)

    with patch.object(profile, "_run_native_evidence", AsyncMock(side_effect=run)):
        asyncio.run(profile.native(args))
    _assert_native_calls(calls)
    report = json.loads(args.output.read_text())
    _assert_native_report(report, args)
    markdown = args.output.with_suffix(".md").read_text()
    _assert_aligned_markdown_tables(markdown)


def test_g2_requires_a_positive_measured_numeric_update() -> None:
    arm = {
        "case": "6400x64-dual",
        "arm": "present",
        "numeric_update_ns_per_window_row": {"candidate": 0.0},
    }
    assert profile._native_g2_gate([arm], True)["met"] is False
    arm["numeric_update_ns_per_window_row"]["candidate"] = 3.0
    assert profile._native_g2_gate([arm], True)["met"] is True


@pytest.mark.parametrize(
    "defect", ("contract", "debug", "selection", "samples", "kind", "entities", "time")
)
def test_native_evidence_rejects_debug_mismatched_or_invalid_runs(defect) -> None:
    cases = ("1x1-dual", "6400x64-dual")
    evidence = _evidence(cases, 1.0)
    if defect == "contract":
        evidence["contract"] = "other"
    elif defect == "debug":
        evidence["debug_assertions"] = True
    elif defect == "selection":
        evidence["selection"] = ["1x1-dual"]
    elif defect == "samples":
        evidence["cases"][0]["samples"].pop()
    elif defect == "kind":
        evidence["cases"][1]["kind"] = "session"
    elif defect == "entities":
        evidence["cases"][1]["workload"]["entities"] = 1
    else:
        evidence["cases"][0]["samples"][3]["present_seconds"] = 0.0
    with pytest.raises(ValueError):
        profile._validate_native_evidence(evidence, cases)


def test_native_budget_stop_settles_the_running_process_and_keeps_completed_rounds(
    tmp_path,
) -> None:
    cases = ("1x1-dual",)
    args = _args(tmp_path, cases, rounds=10)
    args.budget_seconds = 0.5
    settled = []

    async def run(binary, selected, output):
        if not settled and output.name.startswith("round-2-"):
            try:
                await asyncio.sleep(30)
            finally:
                settled.append(output.name)
        return _write_evidence(output, _evidence(cases, 1.0))

    with patch.object(profile, "_run_native_evidence", AsyncMock(side_effect=run)):
        asyncio.run(profile.native(args))
    report = json.loads(args.output.read_text())
    assert settled == ["round-2-candidate.json"]
    assert report["status"] == "stopped"
    assert report["completed_rounds"] == 1
    assert {arm["rounds"] for arm in report["arms"]} == {1}
    assert {arm["paired_round_speedup_ci95"] for arm in report["arms"]} == {None}
    assert all(gate["met"] is None for gate in report["gate_values"].values())
    assert "Status: stopped" in args.output.with_suffix(".md").read_text()


def test_native_evidence_run_reaps_the_binary_when_cancelled(tmp_path) -> None:
    entered = asyncio.Event()
    process = SimpleNamespace(returncode=None, terminate=Mock(), kill=Mock())

    async def wait():
        entered.set()
        await asyncio.Event().wait()

    async def reaped():
        process.returncode = -15
        return -15

    process.wait = AsyncMock(side_effect=wait)

    async def exercise():
        with patch.object(
            profile.asyncio, "create_subprocess_exec", AsyncMock(return_value=process)
        ) as spawn:
            task = asyncio.create_task(
                profile._run_native_evidence(
                    tmp_path / "binary", ("1x1-dual",), tmp_path / "round.json"
                )
            )
            await entered.wait()
            process.wait.side_effect = reaped
            task.cancel()
            with pytest.raises(asyncio.CancelledError):
                await task
        command = spawn.await_args.args
        assert command[0] == str(tmp_path / "binary")
        assert command[1:] == (
            "--exact",
            profile.NATIVE_EVIDENCE_TEST,
            "--ignored",
            "--nocapture",
            "--test-threads=1",
        )
        environment = spawn.await_args.kwargs["env"]
        assert environment["CALC_FLOW_NATIVE_EVIDENCE_CASES"] == "1x1-dual"
        assert environment["CALC_FLOW_NATIVE_EVIDENCE_OUTPUT"] == str(
            tmp_path / "round.json"
        )

    asyncio.run(exercise())
    process.terminate.assert_called_once()
    assert process.returncode == -15


def test_numeric_contract_change_skips_only_cross_arm_digest_equality() -> None:
    config = {"indicator": "dual_sma_spread"}
    with pytest.raises(ValueError, match="digests differ"):
        profile._validate_pair_digest("0" * 64, "1" * 64, config, "sample")
    profile._validate_pair_digest(
        "0" * 64, "1" * 64, config, "sample", numeric_contract_change=True
    )
    with pytest.raises(ValueError, match="valid Float64 digest"):
        profile._validate_sample(
            {"start_row": 0, "correctness": {"passed": True, "float64_digest": None}},
            0,
        )


def test_compare_subcommand_parses_numeric_contract_change(tmp_path) -> None:
    argv = [
        "profile_warm_stream.py",
        "compare",
        "--baseline-build",
        str(tmp_path / "baseline.json"),
        "--candidate-build",
        str(tmp_path / "candidate.json"),
        "--output",
        str(tmp_path / "report.json"),
        "--numeric-contract-change",
    ]
    with (
        patch.object(sys, "argv", argv),
        patch.object(profile, "compare", AsyncMock()) as compare,
    ):
        profile.main()
    assert compare.await_args.args[0].numeric_contract_change is True
