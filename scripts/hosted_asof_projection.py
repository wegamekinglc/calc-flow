"""Measure a single-run public ASOF stream using two official Linux wheels."""

from __future__ import annotations

import asyncio
import json
import os
import statistics
import sys
import time
from datetime import datetime
from pathlib import Path

from benchmarks.native_release_identity import verify_package_files
from scripts.benchmark_suite.process import ROOT, child_environment, command, install
from scripts.benchmark_suite.release import load_release
from scripts.benchmark_suite.report import _paired_verdict
from scripts.benchmark_suite.statistics import paired_round
from scripts.sparse_changelog_ci import _resource
from scripts.sparse_changelog_wheels import _api, seal
from scripts.toolkit import linear_percentile, sha256_file, write_json

OUT = ROOT / "target/public-asof"
CONFIG = ROOT / "benchmarks/hosted-asof-registration.json"
SIDES = ("baseline", "candidate")


def read(path: Path) -> dict:
    return json.loads(path.read_text())


def schedule():
    for group in (1, 2):
        for pair in range(11):
            for side in SIDES if pair == 0 or pair % 2 else SIDES[::-1]:
                yield group, pair, side


def verify_worker(data: dict, cpu: int, native: str) -> None:
    if data["proof"] != {
        "rows": 1024,
        "matched": 764,
        "unmatched": 260,
        "columns": [
            f"{side}__{field}"
            for side in ("left", "right")
            for field in ("key", "time", "seq")
        ],
    }:
        raise ValueError("ASOF output proof differs")
    if type(data["stream_ns"]) is not int or data["stream_ns"] <= 0:
        raise ValueError("invalid ASOF stream clock")
    env = data["environment"]
    if (env["cpu_affinity"], env["tokio_worker_threads"], env["native_sha256"]) != (
        [cpu],
        "1",
        native,
    ):
        raise ValueError("worker CPU, thread count or native module differed")


def acquire() -> None:
    config = read(CONFIG)
    repo = os.environ["GITHUB_REPOSITORY"]
    run, attempt = os.environ["GITHUB_RUN_ID"], os.environ["GITHUB_RUN_ATTEMPT"]
    jobs = json.loads(
        _api(repo, f"actions/runs/{run}/attempts/{attempt}/jobs?per_page=100")
    )["jobs"]
    (job,) = (j for j in jobs if j["name"] == "Public ASOF projection")
    started = datetime.fromisoformat(job["started_at"]).timestamp()
    candidate = {
        "requested_sha": os.environ["CANDIDATE_SHA"],
        "artifact": int(os.environ["CANDIDATE_ARTIFACT"]),
    }
    write_json(
        OUT / "registration.json",
        {
            **config,
            "candidate": candidate,
            "workflow": os.environ["GITHUB_SHA"],
            "run": run,
            "job": job["id"],
            "start_epoch": started,
        },
        exclusive=True,
    )
    if any(sha256_file(ROOT / p) != h for p, h in config["source_hashes"].items()):
        raise ValueError("registered observer source changed")
    for side, wheel in (("baseline", config["baseline"]), ("candidate", candidate)):
        manifest = seal(repo, wheel["artifact"], wheel["requested_sha"], OUT / side)
        if (
            side == "baseline"
            and manifest["artifact_digest"] != wheel["artifact_digest"]
        ):
            raise ValueError("unexpected published baseline")


async def workers(cpu: int, sites: dict, releases: dict) -> list[dict]:
    observations = []
    initial_environment = None
    for group, pair, side in schedule():
        target = OUT / "workers" / f"r{group}-p{pair}-{side}.json"
        env = child_environment(sites[side], tokio_threads=1, polars_threads=1)
        await command(
            [
                sys.executable,
                "-m",
                "benchmarks.public_asof_projection",
                str(target),
            ],
            cwd=ROOT,
            log=target.with_suffix(".log"),
            env=env,
            timeout=25,
        )
        observation = read(target)
        verify_worker(observation, cpu, releases[side]["native_sha256"])
        environment = dict(observation["environment"])
        environment.pop("native_sha256")
        if initial_environment is not None and environment != initial_environment:
            raise ValueError("ASOF worker environments differ")
        initial_environment = environment
        observations.append({"round": group, "pair": pair, "side": side, **observation})
        write_json(OUT / "partial.json", {"observations": observations})
    return observations


def summarize(observations: list[dict]) -> dict:
    formal = [o for o in observations if o["pair"]]
    pooled, by_round = {}, {}
    for side in SIDES:
        samples = [o for o in formal if o["side"] == side]
        values = [o["stream_ns"] / 1_000_000 for o in samples]
        if len(values) != 20:
            raise ValueError("expected twenty public stream clocks per wheel")
        pooled[side] = {
            "p50_ms": statistics.median(values),
            "p95_ms": linear_percentile(values, 0.95),
        }
        by_round[side] = [
            [o["stream_ns"] for o in samples if o["round"] == r] for r in (1, 2)
        ]
    intervals = [
        paired_round(a, b) for a, b in zip(*(by_round[m] for m in SIDES), strict=True)
    ]
    return {
        "pooled": pooled,
        "paired_intervals": intervals,
        "verdict": _paired_verdict(intervals),
    }


async def observe() -> None:
    releases = {side: load_release(OUT / side / "release.json") for side in SIDES}
    base, candidate = (releases[side] for side in SIDES)
    if any(base[k] != candidate[k] for k in ("image", "image_version")):
        raise ValueError("wheel build images differ")
    if base["native_sha256"] == candidate["native_sha256"]:
        raise ValueError("expected distinct native builds")
    sites = {}
    for side, release in releases.items():
        site = await install(release, OUT / side)
        verify_package_files(Path(release["wheel_path"]), site)
        sites[side] = site
    cpu = await _resource(OUT / "resources.jsonl", None)
    os.sched_setaffinity(0, {cpu})
    async with asyncio.TaskGroup() as tasks:
        acquire_workers = tasks.create_task(workers(cpu, sites, releases))
        while not acquire_workers.done():
            await _resource(OUT / "resources.jsonl", cpu)
    write_json(OUT / "result.json", summarize(acquire_workers.result()))


def main():
    if sys.argv[1:] == ["acquire"]:
        acquire()
        return
    if sys.argv[1:] != ["measure"]:
        raise ValueError("expected acquire or measure")
    r = read(OUT / "registration.json")
    started = r["start_epoch"]
    outcome = {"state": "interrupted"}
    try:
        remaining = (
            started + r["budget_seconds"] - r["reserved_cleanup_seconds"] - time.time()
        )
        if remaining <= 0:
            raise TimeoutError("acquisition window exhausted during setup")
        asyncio.run(asyncio.wait_for(observe(), remaining))
    except Exception as error:
        outcome = {"state": "stopped", "error": repr(error)}
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
