"""Observe the public session clock with a paired native-call probe."""

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
from scripts.sparse_changelog_ci import _resource
from scripts.sparse_changelog_wheels import _api, seal
from scripts.toolkit import linear_percentile, sha256_file, write_json

OUT = ROOT / "target/public-steady-session"
CONFIG = ROOT / "benchmarks/hosted-steady-session-registration.json"
MODES = ("plain", "probed")


def read(path: Path) -> dict:
    return json.loads(path.read_text())


def schedule():
    for group in (1, 2):
        for pair in range(11):
            for mode in MODES if pair == 0 or pair % 2 else MODES[::-1]:
                yield group, pair, mode


def verify_worker(data: dict, cpu: int, native: str) -> None:
    if (data["validated_rows"], data["finish_rows"], data["gc_enabled"]) != (
        4160,
        0,
        True,
    ):
        raise ValueError("incomplete steady session proof")
    if len(data["append_ns"]) != 64 or any(
        type(value) is not int or value <= 0 for value in data["append_ns"]
    ):
        raise ValueError("incomplete append clocks")
    env = data["environment"]
    if (env["cpu_affinity"], env["tokio_worker_threads"], env["native_sha256"]) != (
        [cpu],
        "1",
        native,
    ):
        raise ValueError("worker CPU, thread count or native module differed")


def verify_phases(data: dict, mode: str) -> None:
    if data["mode"] != mode:
        raise ValueError("wrong monitoring mode")
    phases = data["phases_ns"]
    if mode == "plain":
        if phases or data["calibration_ns"] is not None:
            raise ValueError("unmonitored control returned phases")
        return
    if len(phases) != 64 or data["calibration_ns"] < 0:
        raise ValueError("incomplete probe")
    for parts, total in zip(phases, data["append_ns"], strict=True):
        if len(parts) != 3 or min(parts) <= 0 or sum(parts) != total:
            raise ValueError("noncontiguous probe regions")


def verify_compiler(dump: str, expected: list[str]) -> None:
    actual = {
        line.partition("]")[2].strip() for line in dump.splitlines() if "]" in line
    }
    if set(expected) != actual:
        raise ValueError(
            "compiler, linker or GCC identity differs from the baseline ELF"
        )


def acquire() -> None:
    config = read(CONFIG)
    repo = os.environ["GITHUB_REPOSITORY"]
    run, attempt = os.environ["GITHUB_RUN_ID"], os.environ["GITHUB_RUN_ATTEMPT"]
    jobs = json.loads(
        _api(repo, f"actions/runs/{run}/attempts/{attempt}/jobs?per_page=100")
    )["jobs"]
    (job,) = (j for j in jobs if j["name"] == "Public steady session phases")
    started = datetime.fromisoformat(job["started_at"]).timestamp()
    write_json(
        OUT / "registration.json",
        {
            **config,
            "workflow": os.environ["GITHUB_SHA"],
            "run": run,
            "job": job["id"],
            "start_epoch": started,
            "runner_image": [os.environ.get("ImageOS"), os.environ.get("ImageVersion")],  # noqa: SIM112
            "cpu_model": next(
                line.partition(":")[2].strip()
                for line in Path("/proc/cpuinfo").read_text().splitlines()
                if line.startswith("model name")
            ),
        },
        exclusive=True,
    )
    if any(sha256_file(ROOT / p) != h for p, h in config["source_hashes"].items()):
        raise ValueError("registered observer source changed")
    for side, wheel in config["releases"].items():
        manifest = seal(repo, wheel["artifact"], wheel["requested_sha"], OUT / side)
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
            )
        ):
            raise ValueError("unexpected sealed release")


async def workers(cpu: int, site: Path, native: str, fixture: Path) -> list[dict]:
    observations = []
    for group, pair, mode in schedule():
        env = child_environment(site, tokio_threads=1, polars_threads=1)
        target = OUT / "workers" / f"r{group}-p{pair}-{mode}.json"
        await command(
            [
                sys.executable,
                "-m",
                "benchmarks.steady_phases",
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
        verify_worker(observation, cpu, native)
        verify_phases(observation, mode)
        observations.append({"round": group, "pair": pair, "mode": mode, **observation})
        write_json(OUT / "partial.json", {"observations": observations})
    return observations


def summarize(observations: list[dict]) -> dict:
    formal = [o for o in observations if o["pair"]]
    pooled, by_round = {}, {}
    for mode in MODES:
        samples = [o for o in formal if o["mode"] == mode]
        values = [v / 1000 for o in samples for v in o["append_ns"]]
        if len(values) != 1280:
            raise ValueError("an arm has fewer than 1,280 steady push clocks")
        pooled[mode] = {
            "p50_us": statistics.median(values),
            "p95_us": linear_percentile(values, 0.95),
        }
        by_round[mode] = [
            [statistics.median(o["append_ns"]) for o in samples if o["round"] == r]
            for r in (1, 2)
        ]
    intervals = [
        paired_round(a, b) for a, b in zip(*(by_round[m] for m in MODES), strict=True)
    ]
    probed = [entry for entry in formal if entry["mode"] == "probed"]
    phases = {}
    for index, name in enumerate(
        ("argument_handling", "native_call", "result_assembly")
    ):
        values = [
            parts[index] / 1000 for entry in probed for parts in entry["phases_ns"]
        ]
        phases[name] = {
            "p50": statistics.median(values),
            "p95": linear_percentile(values, 0.95),
        }
    return {
        "pooled": pooled,
        "pooled_phase_us": phases,
        "calibration_median_ns": statistics.median(
            entry["calibration_ns"] for entry in probed
        ),
        "paired_intervals": intervals,
        "verdict": _paired_verdict(intervals),
    }


async def observe() -> None:
    manifest = load_release(OUT / "release/release.json")
    site = await install(manifest, OUT / "release")
    verify_package_files(Path(manifest["wheel_path"]), site)
    (library,) = (site / "calc_flow").glob("_native*.so")
    report = OUT / "release/compiler.log"
    await command(
        ["readelf", "-p", ".comment", str(library)], cwd=ROOT, log=report, timeout=10
    )
    verify_compiler(report.read_text(), read(CONFIG)["elf_toolchain"])
    fixture = OUT / "fixture"
    fixture.mkdir()
    for name, digest in read(CONFIG)["fixture_sha256"].items():
        p = fixture / name
        p.write_bytes(
            gzip.decompress(
                (
                    ROOT / "benchmarks/steady-session-fixture" / (name + ".gz")
                ).read_bytes()
            )
        )
        if sha256_file(p) != digest:
            raise ValueError("native input/reference/project digest mismatch")
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
