"""One prepared row through the public warmed dual-SMA session."""

from __future__ import annotations

import asyncio
import json
import os
import sys
import time
from datetime import datetime
from pathlib import Path

from benchmarks.native_release_identity import verify_package_files
from scripts.benchmark_suite.process import ROOT, child_environment, command, install
from scripts.benchmark_suite.release import load_release
from scripts.benchmark_suite.release_pairs import pair_order
from scripts.benchmark_suite.report import comparison
from scripts.sparse_changelog_ci import _resource
from scripts.sparse_changelog_wheels import _api, _wheel
from scripts.toolkit import (
    linear_percentile,
    sha256_file,
    wheel_native_sha256,
    write_json,
)

BUDGET = 480
SIDES = ("baseline", "candidate")
CONFIG = ROOT / "benchmarks/hosted-product-registration.json"
OUT = ROOT / "target/hosted-evidence"


def remaining_seconds(started: float, now: float) -> float:
    remaining = BUDGET - (now - started) - 45
    if remaining <= 0:
        raise TimeoutError("acquisition budget exhausted before worker startup")
    return remaining


def acquire() -> None:
    registration = json.loads(CONFIG.read_text())
    repository = os.environ["GITHUB_REPOSITORY"]
    run = int(os.environ["GITHUB_RUN_ID"])
    attempt = int(os.environ["GITHUB_RUN_ATTEMPT"])
    jobs = json.loads(
        _api(repository, f"actions/runs/{run}/attempts/{attempt}/jobs?per_page=100")
    )
    (job,) = (job for job in jobs["jobs"] if job["name"] == "Public stream evidence")
    started = datetime.fromisoformat(job["started_at"]).timestamp()
    record = {
        **registration,
        "workflow_sha": os.environ["GITHUB_SHA"],
        "job": job["id"],
        "workflow_run": run,
        "start_epoch": started,
        "job_budget_seconds": BUDGET,
        "reserved_cleanup_seconds": 45,
        "registration_sha256": sha256_file(CONFIG),
        "requirements_sha256": sha256_file(ROOT / "benchmarks/requirements.lock"),
    }
    write_json(OUT / "registration.json", record, exclusive=True)
    for side in SIDES:
        manifest = record[side]
        metadata = json.loads(
            _api(repository, f"actions/artifacts/{manifest['artifact']}")
        )
        if (
            metadata["digest"] != manifest["artifact_digest"]
            or metadata["expired"]
            or metadata["workflow_run"]["id"] != manifest["workflow_run"]
            or metadata["workflow_run"]["head_sha"] != manifest["requested_sha"]
        ):
            raise ValueError(f"published {side} artifact identity changed")
        directory = OUT / side
        directory.mkdir()
        wheel = _wheel(
            _api(repository, f"actions/artifacts/{manifest['artifact']}/zip"),
            manifest["artifact_digest"],
            directory,
        )
        write_json(
            directory / "release.json",
            {
                **manifest,
                "wheel": wheel.name,
                "wheel_sha256": sha256_file(wheel),
                "native_sha256": wheel_native_sha256(wheel),
            },
        )

        load_release(directory / "release.json")


def control(record: dict, cpu: int) -> tuple:
    sample = record["sample"]
    if (
        sample["start_row"] != 4096
        or not sample["correctness"]["passed"]
        or sample["seconds"] <= 0
        or not sample["phases_seconds"]
        or record["cpu"] != [cpu]
        or not record["phase_probe"]["installed"]
        or any(record["finish"].values())
    ):
        raise ValueError("invalid public warmed single-row session observation")
    return (
        record["warm_output_digest"],
        sample["correctness"]["float64_digest"],
        record["plan"],
    )


async def collect(sites: dict, cpu: int) -> list:
    observations = []
    expected = None
    for index in range(2):
        for pair in range(-1, 10):
            for arm in pair_order(index + pair + 1):
                site = sites[arm]
                log = (
                    OUT / "measurement" / f"round-{index}" / arm / f"{pair + 1:02}.log"
                )
                env = child_environment(
                    site, source=ROOT, tokio_threads=1, polars_threads=1
                )
                await command(
                    [
                        sys.executable,
                        str(ROOT / "scripts/hosted_session_record_worker.py"),
                        str(site / "calc_flow"),
                    ],
                    cwd=ROOT,
                    env=env,
                    log=log,
                )
                row = json.loads(log.read_text().splitlines()[-1])
                observed = control(row, cpu)
                if expected is not None and observed != expected:
                    raise ValueError("worker fixture, complete output or plan changed")
                expected = observed
                observations.append({"arm": arm, "round": index, "pair": pair, **row})
                write_json(OUT / "partial.json", observations)
    return observations


async def observe() -> None:
    releases = {side: load_release(OUT / side / "release.json") for side in SIDES}
    sites = {}
    for side, release in releases.items():
        site = await install(release, OUT / side)
        verify_package_files(Path(release["wheel_path"]), site)
        sites[side] = site
    cpu = await _resource(OUT / "resources.jsonl", None)
    os.sched_setaffinity(0, {cpu})
    async with asyncio.TaskGroup() as tasks:
        acquisition = tasks.create_task(collect(sites, cpu))
        while not acquisition.done():
            await _resource(OUT / "resources.jsonl", cpu)
    rows = acquisition.result()
    arms = {
        arm: [
            [
                row["sample"]["seconds"]
                for row in rows
                if row["arm"] == arm and row["round"] == index and row["pair"] >= 0
            ]
            for index in range(2)
        ]
        for arm in SIDES
    }
    if len(rows) != 44 or any(
        len(round_) != 10 for arm in arms.values() for round_ in arm
    ):
        raise ValueError("incomplete warmed session paired comparison")
    result = comparison(
        dict(
            id="public-single-row-session-record-export",
            rows=1,
            comparison="interleaved",
            correctness=True,
            **arms,
        )
    )
    write_json(
        OUT / "result.json",
        {
            **result,
            "base_p95": linear_percentile(
                [x for r in arms["baseline"] for x in r], 0.95
            ),
        },
    )


def main():
    if sys.argv[1] == "acquire":
        acquire()
        return
    if sys.argv[1] != "measure":
        raise ValueError("expected acquire or measure")
    registered = json.loads((OUT / "registration.json").read_text())
    started = registered["start_epoch"]
    outcome = {"state": "interrupted"}
    try:
        asyncio.run(
            asyncio.wait_for(observe(), remaining_seconds(started, time.time()))
        )
    except Exception as exc:
        outcome = {"state": "stopped", "error": repr(exc)}
        raise
    else:
        outcome = {"state": "completed"}
    finally:
        write_json(
            OUT / "outcome.json",
            {**outcome, "elapsed_since_job_start_seconds": time.time() - started},
        )


if __name__ == "__main__":
    main()
