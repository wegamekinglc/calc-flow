"""Collect absolute timings for the two boundaries in protocol.md."""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import platform
import statistics
import sys
import time
from pathlib import Path

from scripts.benchmark_suite.process import ROOT, child_environment, command, install
from scripts.benchmark_suite.release import load_release
from scripts.sparse_changelog_ci import _resource
from scripts.toolkit import linear_percentile, sha256_file, write_json

HERE = Path(__file__).resolve().parent
ORDER = [(group, index) for group in (1, 2) for index in range(11)]


def verify(record):
    values = tuple(record.get(k) for k in ("setup", "chunks", "rows", "tail"))
    if values != ([260, 0, 261], [32] * 8 + [1], 257, [259]):
        raise ValueError("incomplete active-watermark fixture")
    ns = record.get("elapsed_ns")
    if type(ns) is not int or ns <= 0:
        raise ValueError("missing or invalid active-watermark timer")


def summarize(records):
    if [(r["round"], r["index"]) for r in records] != ORDER:
        raise ValueError("incomplete or misordered independent processes")
    values = [row["elapsed_ns"] / 1_000 for row in records if row["index"]]
    median = statistics.median(values)
    return {
        "p50_us": median,
        "p95_us": linear_percentile(values, 0.95),
        "round_p50_us": [
            statistics.median(values[:10]),
            statistics.median(values[10:]),
        ],
        "rows_per_second_at_p50": 257e6 / median,
    }


async def workers(out, argv, env, cpu, native_path):
    records = []
    for group, index in ORDER:
        raw = out / "workers" / f"round-{group}-{index:02d}.log"
        await command(argv, cwd=ROOT, log=raw, env=env, timeout=20)
        record = json.loads(raw.read_text())
        verify(record)
        if native_path and (
            record["affinity"] != [cpu]
            or record["tokio_threads"] != "1"
            or Path(record["native"]).resolve() != native_path
        ):
            raise ValueError("wheel module, thread count or worker affinity differed")
        row = dict(record, round=group, index=index, warmup=index == 0)
        with (out / "workers.jsonl").open("a", encoding="utf-8") as output:
            print(json.dumps(row), file=output)
        records.append(row)
    return records


async def acquire(args):
    out = args.output
    if args.mode == "rust":
        if sha256_file(args.worker) != args.sha:
            raise ValueError("release binary differs from sealed receipt")
        argv = [str(args.worker.resolve())]
        native_path = None
        env = child_environment(tokio_threads=1)
    else:
        from benchmarks.native_release_identity import verify_package_files

        release = load_release(out / "release" / "release.json")
        site = await install(release, out / "release")
        verify_package_files(Path(release["wheel_path"]), site)
        (native_path,) = (site / "calc_flow").glob("_native*.so")
        native_path = native_path.resolve()
        if sha256_file(native_path) != release["native_sha256"]:
            raise ValueError("installed native library differs from sealed wheel")
        argv = [sys.executable, str(HERE / "python_worker.py")]
        env = child_environment(site, tokio_threads=1)
    cpu = await _resource(out / "resources.jsonl", None)
    os.sched_setaffinity(0, {cpu})
    async with asyncio.TaskGroup() as group:
        samples = group.create_task(workers(out, argv, env, cpu, native_path))
        while not samples.done():
            await _resource(out / "resources.jsonl", cpu)
    return dict(summarize(samples.result()), cpu=cpu)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=("rust", "python"))
    parser.add_argument("output", type=Path)
    parser.add_argument("--worker", type=Path)
    parser.add_argument("--sha")
    args = parser.parse_args()
    out = args.output
    if args.mode == "rust":
        if args.worker is None or args.sha is None:
            parser.error("native acquisition requires the registered binary and SHA")
        started = time.time()
        out.mkdir(parents=True, exist_ok=False)
        budget = 240
    else:
        reg = json.loads((out / "registration.json").read_text())
        started = reg["start_epoch"]
        budget = 300
    result = {"mode": args.mode, "platform": platform.platform(), "started": started}
    try:
        remaining = started + budget - time.time() - 8
        if remaining <= 0:
            raise TimeoutError("budget exhausted during setup")
        result["statistics"] = asyncio.run(asyncio.wait_for(acquire(args), remaining))
    except BaseException as error:
        result.update(state="stopped", reason=repr(error))
        raise
    else:
        result["state"] = "completed"
    finally:
        result["elapsed_seconds"] = time.time() - started
        write_json(out / "outcome.json", result)


if __name__ == "__main__":
    main()
