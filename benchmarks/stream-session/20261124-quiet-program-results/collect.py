"""Compare two installed products at the public Python session boundary."""

from __future__ import annotations

import asyncio
import json
import importlib.metadata
import os
import platform
import statistics
import sys
import time
from pathlib import Path

from benchmarks.native_release_identity import verify_package_files
from scripts.benchmark_suite.process import child_environment, command, install
from scripts.benchmark_suite.release import load_release
from scripts.benchmark_suite.report import _paired_verdict
from scripts.benchmark_suite.statistics import paired_round
from scripts.sparse_changelog_ci import _resource
from scripts.sparse_changelog_wheels import seal
from scripts.toolkit import linear_percentile, sha256_file, write_json

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "acquisition"
SIDES = ("baseline", "candidate")
CASES = ("quiet", "data")


def summarize(records):
    results = {}
    for case in CASES:
        pooled = {}
        for side in SIDES:
            samples = [
                n for r in records if r["side"] == side and r["pair"] for n in r[case]
            ]
            if len(samples) != 1280:
                raise ValueError("missing individual public calls")
            pooled[side] = {
                "p50_ns": statistics.median(samples),
                "p95_ns": linear_percentile(samples, 0.95),
            }
        intervals = []
        for group in (1, 2):
            sides = {}
            for side in SIDES:
                arm = [
                    r
                    for r in records
                    if r["side"] == side and r["round"] == group and r["pair"]
                ]
                if [r["pair"] for r in arm] != list(range(1, 11)):
                    raise ValueError("missing or reordered independent process pairs")
                sides[side] = [statistics.median(r[case]) for r in arm]
            intervals.append(paired_round(sides["baseline"], sides["candidate"]))
        results[case] = {
            "pooled": pooled,
            "paired_rounds": intervals,
            "verdict": _paired_verdict(intervals),
        }
    return results


async def workers(manifests, sites, cpu):
    samples = []
    arrow_version = importlib.metadata.version("pyarrow")
    worker = ROOT / "worker.py"
    env = child_environment(tokio_threads=1, polars_threads=1)
    env.update(OPENBLAS_NUM_THREADS="1", OMP_NUM_THREADS="1", MKL_NUM_THREADS="1")
    for group in (1, 2):
        for pair in range(11):
            order = SIDES if pair == 0 or pair % 2 else SIDES[::-1]
            for side in order:
                log = OUT / f"{group}-{pair:02d}-{side}.json"
                await command(
                    [
                        "taskset",
                        "-c",
                        str(cpu),
                        sys.executable,
                        str(worker),
                        str(sites[side]),
                    ],
                    cwd=OUT,
                    log=log,
                    env=env,
                    timeout=30,
                )
                r = json.loads(log.read_text())
                if (r["native"], r["affinity"], r["tokio"], r["outputs"], r["gc"]) != (
                    manifests[side]["native_sha256"],
                    [cpu],
                    "1",
                    32,
                    True,
                ):
                    raise ValueError(
                        "wheel identity, affinity, worker count, output count or GC mismatch"
                    )
                if r["python"] != sys.version or r["arrow"] != arrow_version:
                    raise ValueError("unexpected Python or Arrow version")
                if any(
                    len(r[c]) != 64 or any(type(n) != int or n <= 0 for n in r[c])
                    for c in CASES
                ):
                    raise ValueError("incorrect clocks")
                samples.append(dict(r, side=side, round=group, pair=pair))
                with (OUT / "workers.jsonl").open("a") as out:
                    print(json.dumps(samples[-1], separators=(",", ":")), file=out)
    return samples


async def acquire(reg):
    manifests, sites, compilers = {}, {}, {}
    for side in SIDES:
        source = reg[side]
        dest = OUT / side
        seal("wegamekinglc/calc-flow", source["artifact"], source["sha"], dest)
        m = load_release(dest / "release.json")
        if m["tree"] != source["tree"] or m["workflow_run"] != source["run"]:
            raise ValueError("unexpected CI checkout or workflow")
        manifests[side] = m
        sites[side] = await install(m, dest)
        verify_package_files(Path(m["wheel_path"]), sites[side])
        (native,) = (sites[side] / "calc_flow").glob("_native*.so")
        log = dest / "compiler.log"
        await command(["readelf", "-p", ".comment", str(native)], cwd=dest, log=log)
        compilers[side] = sorted(
            x.partition("]")[2].strip()
            for x in log.read_text().splitlines()
            if "]" in x
        )
        if (
            not any(line.startswith("rustc version ") for line in compilers[side])
            or sha256_file(sites[side] / "calc_flow/session.py")
            != source["session_sha256"]
        ):
            raise ValueError("unexpected compiler or installed session source")
    for key in ("image", "image_version"):
        if manifests["baseline"][key] != manifests["candidate"][key]:
            raise ValueError(f"wheel build {key} mismatch")
    if compilers["baseline"] != compilers["candidate"]:
        raise ValueError("native toolchains differ")
    resources = OUT / "resources.jsonl"
    cpu = await _resource(resources, None)
    os.sched_setaffinity(0, {cpu})
    async with asyncio.TaskGroup() as tg:
        worker = tg.create_task(workers(manifests, sites, cpu))
        while not worker.done():
            await _resource(resources, cpu)
    write_json(OUT / "statistics.json", summarize(worker.result()))


if __name__ == "__main__":
    reg = json.loads((ROOT / "registration.json").read_text())
    if any(
        sha256_file(ROOT / name) != expected
        for name, expected in reg["sources"].items()
    ):
        raise SystemExit("registered measurement code changed")
    OUT.mkdir(exist_ok=False)
    started = time.monotonic()
    cpu_model = next(
        (
            line.split(":", 1)[1].strip()
            for line in Path("/proc/cpuinfo").read_text().splitlines()
            if line.startswith("model name")
        ),
        "unknown",
    )
    state = {
        "state": "stopped",
        "host": platform.platform(),
        "cpu_model": cpu_model,
        "python": sys.version,
    }
    try:
        asyncio.run(asyncio.wait_for(acquire(reg), timeout=560))
        state["state"] = "completed"
    except BaseException as exc:
        state["reason"] = repr(exc)
        raise
    finally:
        state["wall_seconds"] = time.monotonic() - started
        write_json(OUT / "outcome.json", state)
