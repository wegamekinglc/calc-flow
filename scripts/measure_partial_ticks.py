"""Verify both CI wheels and run the registered public SMA20 case."""

from __future__ import annotations

import asyncio
import json
import os
import sys
from pathlib import Path

from benchmarks.native_release_identity import verify_package_files
from scripts.benchmark_suite.catalog import engine_cases
from scripts.benchmark_suite.process import ROOT, command, install
from scripts.benchmark_suite.release import load_release
from scripts.sparse_changelog_ci import _resource, guarded_measure
from scripts.sparse_changelog_wheels import seal
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
    refs = {
        s: {
            "sha": os.environ[f"{s.upper()}_SHA"],
            "artifact": int(os.environ[f"{s.upper()}_ARTIFACT"]),
        }
        for s in SIDES
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
    for s, ref in refs.items():
        seal(os.environ["GITHUB_REPOSITORY"], ref["artifact"], ref["sha"], OUT / s)


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
        tags[side] = (release["image"], release["image_version"], sorted(comments))
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
    if sys.argv[1] == "seal":
        register()
    else:
        result = {"state": "stopped"}
        try:
            asyncio.run(asyncio.wait_for(measure(), timeout=135))
            result["state"] = "completed"
        except BaseException as exc:
            result["reason"] = repr(exc)
            raise
        finally:
            write_json(OUT / "outcome.json", result)
