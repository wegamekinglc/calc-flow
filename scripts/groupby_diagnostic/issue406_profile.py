"""Sample original stripped natives; retain addresses without claiming symbols."""

from __future__ import annotations

import asyncio
import re
from pathlib import Path

from scripts.benchmark_suite.diagnostics import request_contexts
from scripts.benchmark_suite.measure import _prepare
from scripts.benchmark_suite.process import _complete_owned
from scripts.groupby_diagnostic.issue406 import Issue406Worker, case
from scripts.groupby_diagnostic.profile import logged, profile_workload
from scripts.toolkit import write_json


def validate_trace(raw: str, decoded: str) -> None:
    if "PERF_RECORD_LOST" in raw or "PERF_RECORD_THROTTLE" in raw:
        raise ValueError("lost samples or frequency throttling")
    frames = re.finditer(
        r"(?m)^[ \t]*([0-9a-f]+)[ \t]+[^\n]*\([^\n)]*calc_flow/_native[^\n)]*\.so\)",
        decoded,
    )
    if "PERF_RECORD_SAMPLE" not in raw or not any(
        int(frame[1], 16) for frame in frames
    ):
        raise ValueError("profile lacks original native instruction addresses")


async def preflight(perf: str, root: Path) -> None:
    await logged([perf, "--version"], root, "perf-version")
    write_json(
        root / "permissions.json",
        {
            "perf_event_paranoid": Path(
                "/proc/sys/kernel/perf_event_paranoid"
            ).read_text(),
            "frequency_hz": 49,
            "unwind": "dwarf,16384",
        },
    )
    await logged(
        [
            perf,
            "record",
            "-e",
            "cpu-clock:u",
            "-F",
            "49",
            "--strict-freq",
            "--call-graph",
            "dwarf,16384",
            "-o",
            str(root / "permission-probe.data"),
            "--",
            "/bin/true",
        ],
        root,
        "permission-probe",
    )


async def record(worker: Issue406Worker, root: Path, perf: str) -> None:
    maps = Path(f"/proc/{worker.process.pid}/maps")
    (root / "maps-before.txt").write_bytes(maps.read_bytes())
    profiler = asyncio.create_task(
        logged(
            [
                perf,
                "record",
                "-e",
                "cpu-clock:u",
                "-F",
                "49",
                "--strict-freq",
                "--call-graph",
                "dwarf,16384",
                "-o",
                str(root / "perf.data"),
                "-p",
                str(worker.process.pid),
                "--",
                "sleep",
                "30",
            ],
            root,
            "perf-record",
        )
    )
    try:
        count = await profile_workload(worker, profiler)
    finally:
        await _complete_owned(profiler)
        (root / "maps-after.txt").write_bytes(maps.read_bytes())
    script = await logged(
        [
            perf,
            "script",
            "--show-lost-events",
            "-i",
            str(root / "perf.data"),
        ],
        root,
        "perf-script",
    )
    raw = await logged(
        [
            perf,
            "script",
            "-D",
            "-i",
            str(root / "perf.data"),
        ],
        root,
        "perf-raw",
    )
    validate_trace(raw, script)
    await logged(
        [
            perf,
            "report",
            "--stdio",
            "--no-children",
            "--show-nr-samples",
            "--sort",
            "dso,symbol",
            "--percent-limit",
            "0.5",
            "-i",
            str(root / "perf.data"),
        ],
        root,
        "perf-report",
    )
    write_json(
        root / "profile-result.json",
        {
            "workloads": count,
            "window_seconds": 30,
            "frequency_hz": 49,
            "lost_events": 0,
            "throttle_events": 0,
            "symbols": "original wheel is stripped; native offsets only",
            "lag": "not reported by perf; raw timestamps retained",
            "purpose": "profile-only, includes work outside paired timer",
        },
    )


async def collect(release: dict, site: Path, root: Path, perf: str) -> None:
    worker = await Issue406Worker.start(site, root)
    try:
        workload = case()
        await _prepare(
            {"candidate": worker},
            {"candidate": release},
            workload,
            request_contexts(workload, {"candidate": release}),
        )
        await preflight(perf, root)
        await record(worker, root, perf)
        completion = await worker.request(operation="finish")
        if completion["state"] != "completed":
            raise ValueError("profile worker did not complete")
    finally:
        await worker.close()
