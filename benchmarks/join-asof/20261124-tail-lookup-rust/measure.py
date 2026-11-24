"""Compare the registered Rust watermark settlement workload on two builds."""

from __future__ import annotations

import asyncio
import hashlib
import json
import os
import platform
import statistics
import sys
import time
from pathlib import Path

from scripts.benchmark_suite.asof import CASES, _valid_sample
from scripts.benchmark_suite.process import child_environment
from scripts.benchmark_suite.report import _paired_verdict
from scripts.benchmark_suite.statistics import paired_round
from scripts.sparse_changelog_ci import _resource
from scripts.toolkit import linear_percentile, write_json

ROOT = Path(__file__).resolve().parent
CONFIG = CASES["balanced_512"]
SIDES = ("baseline", "candidate")


async def one_sample(executable, environment, prefix):
    process = await asyncio.create_subprocess_exec(
        str(executable),
        stdin=asyncio.subprocess.PIPE,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
        env=environment,
    )
    try:
        stdout, stderr = await asyncio.wait_for(
            process.communicate(b"sample\nquit\n"), 20
        )
    finally:
        if process.returncode is None:
            process.kill()
            await process.wait()
    (ROOT / f"{prefix}.log").write_bytes(stdout + stderr)
    if process.returncode:
        raise ValueError(f"worker exited {process.returncode}")
    ready, sample = stdout.decode().splitlines()
    for line, tag, oracle in [
        (ready, "ASOF_READY", True),
        (sample, "ASOF_SAMPLE", False),
    ]:
        actual_tag, value = line.split(" ", 1)
        data = json.loads(value)
        if (
            actual_tag != tag
            or data["validated_all_rows"] is not oracle
            or not _valid_sample(data, CONFIG, "latest-per-key")
        ):
            raise ValueError(f"invalid maintained ASOF report at {prefix}")
    return data


async def workers(cpu):
    env = child_environment(tokio_threads=1, polars_threads=1)
    env["CALC_FLOW_ASOF_CONFIG"] = json.dumps(CONFIG)
    observations = []
    for group in (1, 2):
        for index in range(11):
            for side in SIDES if index == 0 or index % 2 else SIDES[::-1]:
                if os.sched_getaffinity(0) != {cpu}:
                    raise ValueError("CPU affinity changed")
                data = await one_sample(ROOT / side, env, f"r{group}-{index}-{side}")
                observations.append(dict(round=group, pair=index, side=side, data=data))
                write_json(ROOT / "partial.json", observations)
    return observations


async def main():
    if (ROOT / "resources.jsonl").exists():
        raise ValueError("existing collection already recorded")
    identities = {}
    for side in SIDES:
        exe = ROOT / side
        digest = hashlib.sha256(exe.read_bytes()).hexdigest()
        assert digest == (ROOT / f"{side}.digest").read_text().split()[0]
        identities[side] = {
            "sha": (ROOT / f"{side}.sha").read_text().strip(),
            "sha256": digest,
            "rustc": (ROOT / f"{side}.rustc").read_text(),
        }
    assert (
        identities["baseline"]["sha"] == "a21d3eed7088795763f511d9aed2fd81920e8201"
        and identities["candidate"]["sha"] == sys.argv[1]
    )
    assert (
        identities["candidate"]["sha256"] != identities["baseline"]["sha256"]
        and identities["baseline"]["rustc"] == identities["candidate"]["rustc"]
    )
    write_json(
        ROOT / "identity.json",
        dict(
            builds=identities,
            case=CONFIG,
            machine=platform.uname()._asdict(),
            schedule="2 rounds; 1 warmup each + 10 AB/BA pairs",
            source=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        ),
    )
    cpu = await _resource(ROOT / "resources.jsonl", None)
    os.sched_setaffinity(0, {cpu})
    async with asyncio.TaskGroup() as tasks:
        task = tasks.create_task(workers(cpu))
        while not task.done():
            await _resource(ROOT / "resources.jsonl", cpu)
    obs = task.result()
    by_round = [
        [
            [
                r["data"]["seconds"] * 1e6
                for r in obs
                if r["round"] == group and r["side"] == side and r["pair"]
            ]
            for side in SIDES
        ]
        for group in (1, 2)
    ]
    assert len(obs) == 44 and all(len(v) == 10 for group in by_round for v in group)
    ci = [paired_round(*group) for group in by_round]
    values = {
        side: [
            r["data"]["seconds"] * 1e6 for r in obs if r["side"] == side and r["pair"]
        ]
        for side in SIDES
    }
    return dict(
        cpu=cpu,
        pooled_us={
            side: dict(p50=statistics.median(v), p95=linear_percentile(v, 0.95))
            for side, v in values.items()
        },
        rounds=ci,
        verdict=_paired_verdict(ci),
    )


if __name__ == "__main__":
    start = time.monotonic()
    try:
        summary = asyncio.run(asyncio.wait_for(main(), 228))
    except BaseException as error:
        write_json(
            ROOT / "outcome.json",
            dict(state="stopped", reason=repr(error), elapsed=time.monotonic() - start),
        )
        raise
    write_json(
        ROOT / "outcome.json",
        dict(state="completed", elapsed=time.monotonic() - start, **summary),
    )
