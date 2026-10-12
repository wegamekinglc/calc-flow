"""Fixed original-wheel controls for the October 12 10M group_by regression."""

from __future__ import annotations

import argparse
import asyncio
import json
import platform
import signal
import sys
import time
from contextlib import contextmanager
from pathlib import Path

from scripts.benchmark_suite.catalog import engine_cases
from scripts.benchmark_suite.process import ROOT, _complete_owned, install
from scripts.benchmark_suite.provenance import harness_sha256
from scripts.benchmark_suite.release import load_release
from scripts.benchmark_suite.worker import dispatch
from scripts.groupby_diagnostic.contract import host, require_host
from scripts.groupby_diagnostic.controller import paired
from scripts.groupby_diagnostic.profile import logged
from scripts.groupby_diagnostic.runtime import (
    AuditWorker,
    input_hashes,
    monitor,
    resource_sample,
)
from scripts.toolkit import git_output, sha256_file, write_json

SEALS = {
    "A": {
        "git_sha": "89e511416d5793e11cefd74b1a7c09aae96f81bd",
        "wheel_sha256": (
            "fda1fce15eef465c8a45d4571713ef23c09b1fed4e268d71e82d9d45255da2ca"
        ),
        "native_sha256": (
            "48e63a99052bece0cf59309b9057e3e950a8673383a0fbf3b6ca5e1cdc5454a6"
        ),
    },
    "B": {
        "git_sha": "8ee5b71b7b862d4e44bd0be390d182cf950d17f7",
        "wheel_sha256": (
            "35e75476a92db376a959355be7c75123bcec522ec66a181fd51171ebfd838685"
        ),
        "native_sha256": (
            "ef24e1d8fefb075909f602a054b03a621200abe4230823fd8259ad2b3a9defa5"
        ),
    },
}
HARNESS_SHA256 = "e30672098c018ef1a650492403bf5ae4813d0182e2b4921cb9233b4d26b8b2e6"


def case() -> dict:
    return next(
        row
        for row in engine_cases(10_000_000)
        if row["backend"] == "calc-flow-stream" and row["scenario"] == "group_by"
    )


def plan() -> dict:
    return {
        "contract": "issue406-groupby-original-wheels-v1",
        "case": case(),
        "comparisons": [["A", "A"], ["B", "B"], ["A", "B"]],
        "rounds": 2,
        "pairs": 10,
        "warmups_per_side_per_round": 1,
        "profile_sides": ["A", "B"],
        "profile_seconds_per_side": 30,
        "profile_frequency_hz": 49,
        "budget_seconds": 600,
        "work_seconds": 550,
        "build_seconds": 0,
        "source_run": 38178694560,
        "seals": SEALS,
        "checkpoint": "enabled/24h; timer excludes terminal capture",
        "purpose": {
            "AA": "same baseline wheel tests process/host bias",
            "BB": "same candidate wheel tests process/host bias",
            "AB": "original failing wheel pair tests repeatable binary difference",
            "profiles": "original native instruction addresses; no symbol rebuild",
        },
    }


def sealed(side: str, path: Path) -> dict:
    value = load_release(path)
    if any(value.get(key) != expected for key, expected in SEALS[side].items()):
        raise ValueError("requires October 12 original sealed wheels")
    return value


class Issue406Worker(AuditWorker):
    worker_module = "scripts.groupby_diagnostic.issue406"


def worker(root: Path) -> None:
    active = None
    try:
        for line in sys.stdin:
            message = json.loads(line)
            if message["operation"] == "prepare" and message["case"] != case():
                raise ValueError("accepts only the fixed 10M group_by workload")
            response, active = dispatch(message, active, root)
            if message["operation"] == "prepare":
                response = {**response, "input_hashes": input_hashes(active, root)}
            print(json.dumps(response, allow_nan=False), flush=True)
    finally:
        if active is not None:
            active.close()


async def collect(root: Path, releases: dict, perf: str) -> None:
    sites = {
        side: await install(value, root / "sites" / side)
        for side, value in releases.items()
    }
    results = []
    for left, right in plan()["comparisons"]:
        selected = {"baseline": left, "candidate": right}
        row = await paired(
            case(),
            {side: sites[value] for side, value in selected.items()},
            {side: releases[value] for side, value in selected.items()},
            root / f"{left}{right}",
            worker_type=Issue406Worker,
        )
        results.append({"sources": selected, **row})
        write_json(root / "results.json", results)
    from scripts.groupby_diagnostic.issue406_profile import collect as profile

    for side in plan()["profile_sides"]:
        await profile(releases[side], sites[side], root / "profiles" / side, perf)


@contextmanager
def cancel_on_signal():
    loop, parent = asyncio.get_running_loop(), asyncio.current_task()
    previous = {sig: signal.getsignal(sig) for sig in (signal.SIGINT, signal.SIGTERM)}
    for sig in previous:
        loop.add_signal_handler(sig, parent.cancel)
    try:
        yield
    finally:
        for sig, handler in previous.items():
            loop.remove_signal_handler(sig)
            signal.signal(sig, handler)


async def preflight(root: Path, release_root: Path) -> dict:
    machine = host()
    write_json(root / "host.json", machine)
    require_host(machine)
    if platform.python_version() != "3.13.16":
        raise ValueError("requires original Python 3.13.16")
    if harness_sha256() != HARNESS_SHA256:
        raise ValueError("common harness differs from the failed morning run")
    if resource_sample()["builds"]:
        raise ValueError("native build active before measurement")
    await logged(["lscpu", "--json"], root, "cpu")
    releases = {
        side: sealed(side, release_root / side / "release.json") for side in SEALS
    }
    write_json(
        root / "identity.json",
        {
            "releases": releases,
            "harness_sha256": harness_sha256(),
            "checkout": git_output(ROOT, "rev-parse", "HEAD"),
            "diagnostic_files": {
                str(path.relative_to(ROOT)): sha256_file(path)
                for path in sorted((ROOT / "scripts/groupby_diagnostic").glob("*.py"))
            },
        },
    )
    return releases


def final_outcome(root: Path, started: float, outcome: dict) -> dict:
    samples = [
        json.loads(line) for line in (root / "resources.jsonl").read_text().splitlines()
    ]
    elapsed = time.monotonic() - started
    overlap = any(sample["builds"] for sample in samples)
    budget_satisfied = elapsed <= plan()["budget_seconds"]
    return {
        **outcome,
        "build_overlap": overlap,
        "elapsed_seconds": elapsed,
        "budget_satisfied": budget_satisfied,
        "status": "invalid" if overlap or not budget_satisfied else outcome["status"],
    }


async def run(root: Path, release_root: Path, perf: str) -> int:
    started = time.monotonic()
    root.mkdir(parents=True, exist_ok=True)
    write_json(root / "plan.json", plan())
    outcome = {"measurement_complete": False, "status": "failed"}
    stop = asyncio.Event()
    observer = asyncio.create_task(monitor(root / "resources.jsonl", stop))
    with cancel_on_signal():
        try:
            async with asyncio.timeout(plan()["work_seconds"]):
                releases = await preflight(root, release_root)
                await collect(root, releases, perf)
                outcome = {"measurement_complete": True, "status": "completed"}
        except (Exception, asyncio.CancelledError) as error:
            outcome = {**outcome, "error": f"{type(error).__name__}: {error}"}
        finally:
            stop.set()
            await _complete_owned(observer)
            outcome = final_outcome(root, started, outcome)
            write_json(root / "outcome.json", outcome)
    return 0 if outcome["status"] == "completed" else 1


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("operation", choices=("plan", "run", "worker"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--releases", type=Path)
    parser.add_argument("--perf", default="perf")
    args = parser.parse_args()
    if args.operation == "worker":
        worker(args.output)
    elif args.operation == "plan":
        write_json(args.output / "plan.json", plan())
    else:
        if args.releases is None:
            parser.error("run requires --releases")
        return asyncio.run(run(args.output, args.releases, args.perf))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
