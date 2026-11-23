"""Verify both CI wheels and run the registered public SMA20 case."""

from __future__ import annotations

import asyncio
import json
import os
import subprocess
from pathlib import Path

from benchmarks.native_release_identity import verify_package_files
from scripts.benchmark_suite.catalog import engine_cases
from scripts.benchmark_suite.process import ROOT, command, install
from scripts.benchmark_suite.release import load_release
from scripts.sparse_changelog_ci import _resource, guarded_measure
from scripts.toolkit import sha256_file, write_json

SIDES = ("baseline", "candidate")
CASE = next(
    c
    for c in engine_cases(10_000)
    if c["id"] == "engines/10000/calc-flow-stream-1t/sma20"
)
OUT = ROOT / "target/partial-ticks"
PROTOCOL = ROOT / "benchmarks/rolling/20261124-partial-ticks/protocol.md"


def register():
    run = os.environ["GITHUB_RUN_ID"]
    attempt = os.environ["GITHUB_RUN_ATTEMPT"]
    artifacts = json.loads(
        subprocess.check_output(
            [
                "gh",
                "api",
                f"repos/{os.environ['GITHUB_REPOSITORY']}/actions/runs/{run}/artifacts?per_page=100",
            ]
        )
    )["artifacts"]
    refs = {}
    for side in SIDES:
        expected = os.environ[f"{side.upper()}_SHA"]
        (artifact,) = [
            a
            for a in artifacts
            if a["name"] == f"benchmark-release-{side}-{run}-{attempt}"
        ]
        release = load_release(OUT / side / "release.json")
        if (
            artifact["expired"]
            or release["git_sha"] != expected
            or not artifact["digest"].startswith("sha256:")
        ):
            raise ValueError("unexpected sealed build or artifact")
        refs[side] = {
            "sha": expected,
            "artifact": artifact["id"],
            "digest": artifact["digest"],
        }
    write_json(
        OUT / "registration.json",
        {
            "refs": refs,
            "case": CASE,
            "protocol": sha256_file(PROTOCOL),
            "rounds": 2,
            "pairs_per_round": 10,
            "job_seconds": 480,
            "workflow_sha": os.environ["GITHUB_SHA"],
        },
        exclusive=True,
    )


async def measure():
    sites, releases, tags = {}, {}, {}
    for side in SIDES:
        directory = OUT / side
        release = load_release(directory / "release.json")
        site = await install(release, directory)
        verify_package_files(Path(release["wheel_path"]), site)
        sites[side] = (site, ROOT)
        releases[side] = release
        (native,) = (site / "calc_flow").glob("_native*.so")
        log = directory / "compiler.log"
        await command(
            ["readelf", "-p", ".comment", str(native)], cwd=directory, log=log
        )
        comments = [
            line.partition("]")[2].strip()
            for line in log.read_text().splitlines()
            if "]" in line
        ]
        if not any(line.startswith("rustc version ") for line in comments):
            raise ValueError("missing compiler identity")
        image = (directory / "image.txt").read_text()
        compiler = (directory / "compiler.txt").read_text()
        if "ImageVersion=" not in image or "rustc 1.94.0" not in compiler:
            raise ValueError("missing build image or compiler")
        tags[side] = (image, compiler, sorted(comments))
    if (
        tags["baseline"] != tags["candidate"]
        or releases["baseline"]["native_sha256"]
        == releases["candidate"]["native_sha256"]
    ):
        raise ValueError("incompatible wheel pair")
    cpu = await _resource(OUT / "resources.jsonl", None)
    os.sched_setaffinity(0, {cpu})
    measured = await guarded_measure(OUT, cpu, CASE, sites, releases)
    write_json(OUT / "samples.json", measured)
    if measured["status"] != "ok":
        raise ValueError(measured)
    print(json.dumps(measured["result"]))


if __name__ == "__main__":
    result = {"state": "stopped"}
    try:
        register()
        asyncio.run(asyncio.wait_for(measure(), timeout=135))
        result["state"] = "completed"
    except BaseException as exc:
        result["reason"] = repr(exc)
        raise
    finally:
        write_json(OUT / "outcome.json", result)
