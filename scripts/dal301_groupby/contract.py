from __future__ import annotations

import os
import platform
from pathlib import Path

from scripts.benchmark_suite.catalog import engine_cases
from scripts.benchmark_suite.release import load_release
from scripts.benchmark_suite.report import ROUNDS, SAMPLES, THRESHOLD_PERCENT

RUN_ID = 35596885420
SEALS = {
    "A": {
        "git_sha": "a594ad697a57947237dd289f3a1bc2272ef099e8",
        "wheel_sha256": (
            "3ad736f3ddc81a5732ca82fa2631c8729ea6418595c312940430dde48beadb35"
        ),
        "native_sha256": (
            "de32f46a22bd7b74185ca7656cb31bc39690e16d5fea108ec887f6533abcf370"
        ),
    },
    "B": {
        "git_sha": "b7e92cc58bc5000be0db5e6ec9a2decec052feb5",
        "wheel_sha256": (
            "64635726a174a73924924080230cd926bc752ea5930077def26d431d2d385e99"
        ),
        "native_sha256": (
            "d843f2a70af3866f0446698b7a252ce58509929e1927f3a1158b3e29f291109e"
        ),
    },
}
ARTIFACTS = {"A": 10636839917, "B": 10637665772}


def cases() -> list[dict]:
    return [
        case
        for rows in (10_000, 100_000)
        for case in engine_cases(rows)
        if case["backend"] == "calc-flow-stream" and case["scenario"] == "group_by"
    ]


def plan() -> dict:
    return {
        "contract": "dal301-groupby-v1",
        "cases": cases(),
        "comparisons": [["A", "A"], ["B", "B"], ["A", "B"]],
        "rounds": ROUNDS,
        "pairs": SAMPLES,
        "threshold_percent": THRESHOLD_PERCENT,
        "order": [["baseline", "candidate"], ["candidate", "baseline"]] * 5,
        "source_run": RUN_ID,
        "artifacts": ARTIFACTS,
        "seals": SEALS,
    }


def sealed(side: str, path: str | Path) -> dict:
    result = load_release(Path(path))
    if any(result.get(key) != value for key, value in SEALS[side].items()):
        raise ValueError("DAL-301 requires the original sealed A/B artifacts")
    return result


def host() -> dict:
    import psutil

    return {
        "system": platform.system(),
        "release": platform.release(),
        "cpus": os.cpu_count(),
        "affinity": psutil.Process().cpu_affinity(),
        "python": platform.python_version(),
        "runner": os.environ.get("RUNNER_NAME"),
        "runner_environment": os.environ.get("RUNNER_ENVIRONMENT"),
        "image": os.environ.get("ImageVersion"),  # noqa: SIM112
        "uname": list(platform.uname()),
    }


def require_host(value: dict) -> None:
    if (
        value["system"] != "Linux"
        or "microsoft" in value["release"].lower()
        or value["cpus"] != 4
        or len(value["affinity"]) != 4
        or value["runner_environment"] != "github-hosted"
    ):
        raise ValueError("requires actual GitHub-hosted Linux with four logical CPUs")
