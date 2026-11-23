"""Collect two public sparse SQL emission modes on one official wheel."""

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
MODES = ("snapshot", "changelog")


def read(path: Path) -> dict:
    return json.loads(path.read_text())


def schedule():
    for group in (1, 2):
        for pair in range(11):
            for mode in MODES if pair == 0 or pair % 2 else MODES[::-1]:
                yield group, pair, mode


def verify_worker(data: dict, mode: str, cpu: int, native: str) -> None:
    if (
        data["mode"] != mode
        or data["output_rows"]
        != [10_000] + [160 if mode == "changelog" else 10_000] * 20
    ):
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
    write_json(
        OUT / "registration.json",
        {
            **config,
            "workflow": os.environ["GITHUB_SHA"],
            "run": run,
            "job": job["id"],
            "start_epoch": started,
        },
        exclusive=True,
    )
    if any(sha256_file(ROOT / p) != h for p, h in config["source_hashes"].items()):
        raise ValueError("registered observer source changed")
    wheel = config["release"]
    manifest = seal(repo, wheel["artifact"], wheel["checkout"], OUT / "release")
    if any(
        wheel[k] != manifest[k]
        for k in (
            "artifact_digest",
            "job",
            "workflow_run",
            "tree",
            "checkout",
            "image",
            "image_version",
            "wheel_sha256",
            "native_sha256",
        )
    ):
        raise ValueError("unexpected published wheel")


async def workers(cpu: int, site: Path, native: str, fixture: Path) -> list[dict]:
    observations = []
    env = child_environment(site, tokio_threads=1, polars_threads=1)
    for group, pair, mode in schedule():
        target = OUT / "workers" / f"r{group}-p{pair}-{mode}.json"
        await command(
            [
                sys.executable,
                "-m",
                "benchmarks.sparse_warm_update",
                mode,
                str(fixture),
                str(target),
            ],
            cwd=ROOT,
            log=target.with_suffix(".log"),
            env=env,
            timeout=25,
        )
        observation = read(target)
        verify_worker(observation, mode, cpu, native)
        observations.append({"round": group, "pair": pair, **observation})
        write_json(OUT / "partial.json", {"observations": observations})
    return observations


def summarize(observations: list[dict]) -> dict:
    formal = [o for o in observations if o["pair"]]
    pooled, by_round = {}, {}
    for mode in MODES:
        samples = [o for o in formal if o["mode"] == mode]
        values = [v / 1000 for o in samples for v in o["update_ns"]]
        if len(values) != 400:
            raise ValueError("an arm has fewer than 400 warm update clocks")
        pooled[mode] = {
            "p50_us": statistics.median(values),
            "p95_us": linear_percentile(values, 0.95),
        }
        by_round[mode] = [
            [statistics.median(o["update_ns"]) for o in samples if o["round"] == r]
            for r in (1, 2)
        ]
    intervals = [
        paired_round(a, b) for a, b in zip(*(by_round[m] for m in MODES), strict=True)
    ]
    return {
        "pooled": pooled,
        "paired_intervals": intervals,
        "verdict": _paired_verdict(intervals),
    }


async def observe() -> None:
    manifest = load_release(OUT / "release/release.json")
    site = await install(manifest, OUT / "release")
    verify_package_files(Path(manifest["wheel_path"]), site)
    fixture = OUT / "sparse.arrow"
    fixture.write_bytes(gzip.decompress(FIXTURE.read_bytes()))
    cpu = await _resource(OUT / "resources.jsonl", None)
    os.sched_setaffinity(0, {cpu})
    async with asyncio.TaskGroup() as tasks:
        acquire_workers = tasks.create_task(
            workers(cpu, site, manifest["native_sha256"], fixture)
        )
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
