"""Compare full snapshot latency on two official grouped SUM wheels."""

from __future__ import annotations

import asyncio
import gzip
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
from scripts.sparse_changelog_ci import FIXTURE, _resource
from scripts.sparse_changelog_wheels import _api, seal
from scripts.toolkit import linear_percentile, sha256_file, write_json

OUT = ROOT / "target/public-warm-update"
CONFIG = ROOT / "benchmarks/hosted-sparse-update-registration.json"
SIDES = ("baseline", "candidate")


def read(path: Path) -> dict:
    return json.loads(path.read_text())


def schedule():
    for group in (1, 2):
        for pair in range(11):
            for side in SIDES if pair == 0 or pair % 2 else SIDES[::-1]:
                yield group, pair, side


def verify_worker(data: dict, cpu: int, native: str) -> None:
    if data["mode"] != "snapshot" or data["output_rows"] != [10_000] * 21:
        raise ValueError("wrong mode or incomplete sparse output")
    if len(data["update_ns"]) != 20 or any(
        type(v) is not int or v <= 0 for v in data["update_ns"]
    ):
        raise ValueError("missing warm update clocks")
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
    (job,) = (j for j in jobs if j["name"] == "Public warm sparse update")
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


async def workers(cpu: int, sites: dict, releases: dict, fixture: Path) -> list[dict]:
    observations = []
    for group, pair, side in schedule():
        target = OUT / "workers" / f"r{group}-p{pair}-{side}.json"
        env = child_environment(sites[side], tokio_threads=1, polars_threads=1)
        await command(
            [
                sys.executable,
                "-m",
                "benchmarks.sparse_warm_update",
                "snapshot",
                str(fixture),
                str(target),
            ],
            cwd=ROOT,
            log=target.with_suffix(".log"),
            env=env,
            timeout=25,
        )
        observation = read(target)
        verify_worker(observation, cpu, releases[side]["native_sha256"])
        observations.append({"round": group, "pair": pair, "side": side, **observation})
        write_json(OUT / "partial.json", {"observations": observations})
    return observations


def summarize(observations: list[dict]) -> dict:
    formal = [o for o in observations if o["pair"]]
    pooled, by_round = {}, {}
    for side in SIDES:
        samples = [o for o in formal if o["side"] == side]
        values = [v / 1000 for o in samples for v in o["update_ns"]]
        if len(values) != 400:
            raise ValueError("an arm has fewer than 400 warm update clocks")
        pooled[side] = {
            "p50_us": statistics.median(values),
            "p95_us": linear_percentile(values, 0.95),
        }
        by_round[side] = [
            [statistics.median(o["update_ns"]) for o in samples if o["round"] == r]
            for r in (1, 2)
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
    fixture = OUT / "sparse.arrow"
    fixture.write_bytes(gzip.decompress(FIXTURE.read_bytes()))
    cpu = await _resource(OUT / "resources.jsonl", None)
    os.sched_setaffinity(0, {cpu})
    async with asyncio.TaskGroup() as tasks:
        acquire_workers = tasks.create_task(workers(cpu, sites, releases, fixture))
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
