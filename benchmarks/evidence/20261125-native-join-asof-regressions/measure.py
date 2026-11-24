"""Paired fresh-process samples for the registered Join and ASOF handler cases."""

from __future__ import annotations

import asyncio
import hashlib
import json
import math
import os
import platform
import statistics
import sys
import time
from pathlib import Path

from scripts.sparse_changelog_ci import _resource

from scripts.benchmark_suite.process import child_environment
from scripts.benchmark_suite.report import _paired_verdict
from scripts.benchmark_suite.statistics import paired_round
from scripts.toolkit import linear_percentile

ROOT = Path(__file__).resolve().parent
BASE = "92230f68c9024a053f2e7c74185c3a98f5003917"
expected_rows = {"join": 10_000, "asof": 100_000}


def write(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def sha(path):
    with path.open("rb") as source:
        return hashlib.file_digest(source, "sha256").hexdigest()


async def worker(executable, case):
    proc = await asyncio.create_subprocess_exec(
        str(executable),
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
        env=child_environment(tokio_threads=1, polars_threads=1),
    )
    try:
        out, err = await proc.communicate()
    finally:
        if proc.returncode is None:
            proc.kill()
            await proc.wait()
    if proc.returncode:
        raise RuntimeError(err.decode()[-3000:])
    data = json.loads(out)
    if not (
        data.get("validated_all_rows") is True
        and data.get("output_rows") == expected_rows[case]
        and math.isfinite(data["seconds"])
        and data["seconds"] > 0
    ):
        raise ValueError(f"invalid oracle or clock {data}")
    if (
        case == "asof"
        and data["evicted_right_rows"] + data["retained_right_rows"]
        != expected_rows[case]
    ):
        raise ValueError(f"invalid state accounting {data}")
    return data


async def run(case, head, directory):
    builds = {}
    for side, commit in (("base", BASE), ("head", head)):
        exe = ROOT / f"{case}-{commit[:10]}"
        receipt = json.loads(exe.with_suffix(".receipt.json").read_text())
        if receipt["commit"] != commit or receipt["binary_sha256"] != sha(exe):
            raise ValueError("unqualified executable")
        builds[side] = (exe, receipt)
    for field in (
        "case",
        "harness_sha256",
        "lock_sha256",
        "rustc",
        "cargo",
        "settings",
    ):
        if field == "cargo":
            continue  # base can compile both fixed examples at once
        if builds["base"][1][field] != builds["head"][1][field]:
            raise ValueError(field)
    for side, (_, receipt) in builds.items():
        write(directory / (side + ".receipt.json"), receipt)
    write(
        directory / "identity.json",
        {
            "base": BASE,
            "head": head,
            "case": case,
            "processor": platform.uname()._asdict(),
            "protocol_sha256": sha(ROOT / "protocol.md"),
            "adapter_sha256": sha(ROOT / ("stage_" + case + ".rs")),
        },
    )
    cpu = await _resource(directory / "resources.jsonl", None)
    os.sched_setaffinity(0, {cpu})
    obs = []

    async def collect():
        for rnd in (1, 2):
            for i in range(11):
                for side in ("base", "head") if i == 0 or i % 2 else ("head", "base"):
                    if os.sched_getaffinity(0) != {cpu}:
                        raise ValueError("affinity changed")
                    x = await worker(builds[side][0], case)
                    row = {"round": rnd, "pair": i, "side": side, "result": x}
                    with (directory / "observations.jsonl").open("a") as f:
                        f.write(json.dumps(row, allow_nan=False) + "\n")
                    obs.append(row)

    async with asyncio.TaskGroup() as tasks:
        work = tasks.create_task(collect())
        while not work.done():
            await _resource(directory / "resources.jsonl", cpu)

    def values(side, rnd=None):
        return [
            r["result"]["seconds"] * 1000
            for r in obs
            if r["pair"] and r["side"] == side and (rnd is None or r["round"] == rnd)
        ]

    assert len(obs) == 44 and all(len(values(side)) == 20 for side in builds)
    rounds = [paired_round(values("base", r), values("head", r)) for r in (1, 2)]
    return {
        "case": case,
        "cpu": cpu,
        "rounds": rounds,
        "verdict": _paired_verdict(rounds),
        "pooled_ms": {
            s: {
                "p50": statistics.median(values(s)),
                "p95": linear_percentile(values(s), 0.95),
            }
            for s in builds
        },
    }


if __name__ == "__main__":
    if not __debug__:
        raise SystemExit("Run with Python assertions enabled for all oracle checks.")
    case, head = sys.argv[1:]
    directory = ROOT / f"obs-{case}-{head[:10]}"
    directory.mkdir(exist_ok=False)
    start = time.monotonic()
    try:
        with asyncio.Runner() as runner:
            result = runner.run(asyncio.wait_for(run(case, head, directory), 90))
        out = {"state": "complete", **result}
    except BaseException as error:
        out = {"state": "stopped", "error": repr(error)}
        raise
    finally:
        out["wall_seconds"] = time.monotonic() - start
        write(directory / "outcome.json", out)
        print(out, flush=True)
