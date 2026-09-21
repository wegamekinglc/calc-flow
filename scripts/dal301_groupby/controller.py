from __future__ import annotations

import asyncio
import json
import platform
from functools import partial
from pathlib import Path

from scripts.benchmark_suite.measure import _measured_row, _prepare, _samples
from scripts.benchmark_suite.process import install
from scripts.benchmark_suite.provenance import harness_sha256
from scripts.benchmark_suite.report import ROUNDS, comparison
from scripts.dal301_groupby.contract import cases, host, plan, require_host, sealed
from scripts.dal301_groupby.runtime import AuditWorker, monitor, resource_sample
from scripts.toolkit import git_output, sha256_file, write_json


async def round_(case: dict, sites: dict, releases: dict, root: Path) -> dict:
    workers = {}
    try:
        for side, site in sites.items():
            workers[side] = await AuditWorker.start(site, root / side)
        environment = await _prepare(workers, releases, case)
        if workers["baseline"].input_hashes != workers["candidate"].input_hashes:
            raise ValueError("paired input IPC bytes differ")
        samples = await _samples(workers)
        completion = {
            side: await w.request(operation="finish") for side, w in workers.items()
        }
        if any(value["state"] != "completed" for value in completion.values()):
            raise ValueError("worker did not finish")
        return {
            "environment": environment,
            "samples": samples,
            "completion": completion,
        }
    except Exception as error:
        write_json(root / "failure.json", {"error": repr(error)})
        raise
    finally:
        await close_workers(workers, root)


async def close_workers(workers: dict, root: Path) -> None:
    closed = await asyncio.gather(
        *(w.close() for w in workers.values()), return_exceptions=True
    )
    errors = [repr(value) for value in closed if isinstance(value, BaseException)]
    if errors:
        write_json(root / "close-errors.json", errors)
        raise RuntimeError(f"worker cleanup failed: {errors}")


async def paired(case: dict, sites: dict, releases: dict, root: Path) -> dict:
    evidence = []
    for index in range(ROUNDS):
        evidence.append(await round_(case, sites, releases, root / f"round-{index}"))
        write_json(root / "rounds.json", evidence)
    if evidence[0]["environment"] != evidence[1]["environment"]:
        raise ValueError("confirmation environment differs")
    row = _measured_row(case, evidence, "interleaved")
    return {**row, "result": comparison(row)}


async def collect(root: Path, releases: dict) -> None:
    sites = {
        side: await install(value, root / "sites" / side)
        for side, value in releases.items()
    }
    results = []
    for case in cases():
        for left, right in plan()["comparisons"]:
            selected = {"baseline": left, "candidate": right}
            row = await paired(
                case,
                {s: sites[v] for s, v in selected.items()},
                {s: releases[v] for s, v in selected.items()},
                root / f"{case['rows']}-{left}{right}",
            )
            results.append({"sources": selected, **row})
            write_json(root / "results.json", results)


async def observed_collect(root: Path, operation) -> None:
    root.mkdir(parents=True, exist_ok=True)
    if resource_sample()["builds"]:
        raise ValueError("native build active before sampling")
    stop = asyncio.Event()
    observer = asyncio.create_task(monitor(root / "resources.jsonl", stop))
    try:
        await operation()
    finally:
        stop.set()
        await observer
    samples = [
        json.loads(line) for line in (root / "resources.jsonl").read_text().splitlines()
    ]
    if any(sample["builds"] for sample in samples):
        raise ValueError("native build overlap invalidates this diagnostic window")


async def run(root: Path, release_root: Path, profiles: Path | None) -> int:
    write_json(root / "plan.json", plan())
    try:
        machine = host()
        write_json(root / "host.json", machine)
        require_host(machine)
        if platform.python_version() != "3.13.15":
            raise ValueError("requires sealed-release Python 3.13.15")
        releases = {s: sealed(s, release_root / s / "release.json") for s in ("A", "B")}
        write_json(
            root / "identity.json",
            {
                "releases": releases,
                "harness_sha256": harness_sha256(),
                "diagnostic_files": {
                    str(path): sha256_file(path)
                    for path in sorted(Path("scripts/dal301_groupby").glob("*.py"))
                },
                "checkout": git_output(Path.cwd(), "rev-parse", "HEAD"),
            },
        )
        await observed_collect(root, partial(collect, root, releases))
        if profiles is not None:
            from scripts.dal301_groupby.profile import collect_profiles

            await observed_collect(
                root / "profiles",
                partial(collect_profiles, profiles, root / "profiles"),
            )
        write_json(root / "outcome.json", {"status": "completed", "exit_code": 0})
        return 0
    except Exception as error:
        write_json(
            root / "outcome.json",
            {"status": "failed", "error": repr(error), "exit_code": 1},
        )
        return 1
