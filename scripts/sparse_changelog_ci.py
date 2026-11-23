"""Run one public sparse changelog comparison from two verified Linux CI wheels."""

from __future__ import annotations

import argparse
import asyncio
import gzip
import json
import os
import time
from pathlib import Path

from benchmarks.native_release_identity import verify_package_files
from scripts.benchmark_suite.measure import measure_case
from scripts.benchmark_suite.process import ROOT, install
from scripts.benchmark_suite.release import load_release
from scripts.benchmark_suite.report import ROUNDS, SAMPLES
from scripts.sparse_changelog_wheels import seal
from scripts.toolkit import sha256_file, write_json

CASE = {
    "family": "sparse-changelog",
    "id": "python/public-sparse-changelog/fresh-stream",
    "backend": "calc-flow-stream",
    "rows": 12000,
}
FIXTURE = ROOT / "benchmarks/sum-state/public-sparse-changelog-20261123/sparse.arrow.gz"
SIDES = ("baseline", "candidate")


def interval(
    before: dict,
    after: dict,
    busy: list[float],
    available: int,
    cpu: int,
    siblings: tuple[int, ...],
    *,
    preflight: bool,
) -> dict:
    major = after["pgmajfault"] - before["pgmajfault"]
    swapped = sum(after[k] - before[k] for k in ("pswpin", "pswpout"))
    host = sum(busy) / len(busy)
    sibling = max((busy[i] for i in siblings), default=0.0)
    quiet = max(busy[cpu], sibling) <= 20 if preflight else sibling < 50
    return {
        "cpu": cpu,
        "cpu_busy": busy[cpu],
        "siblings": list(siblings),
        "sibling_busy": sibling,
        "host_busy": host,
        "available_bytes": available,
        "major_faults": major,
        "swap_pages": swapped,
        "swap_in_pages": after["pswpin"] - before["pswpin"],
        "swap_out_pages": after["pswpout"] - before["pswpout"],
        "suitable": quiet
        and host < 50
        and available >= 10 * 1024**3
        and major < 256
        and swapped < 1024,
    }


def _siblings(cpu: int) -> tuple[int, ...]:
    topology = Path(
        f"/sys/devices/system/cpu/cpu{cpu}/topology/thread_siblings_list"
    ).read_text()
    related = set()
    for item in topology.strip().split(","):
        bounds = [int(n) for n in item.split("-")]
        related.update(range(bounds[0], bounds[-1] + 1))
    return tuple(sorted(related - {cpu}))


def _paging() -> dict[str, int]:
    return {
        k: int(v)
        for k, v in (
            line.split() for line in Path("/proc/vmstat").read_text().splitlines()
        )
        if k in ("pgmajfault", "pswpin", "pswpout")
    }


async def _resource(path: Path, cpu: int | None) -> int:
    import psutil

    started = time.monotonic()
    before = _paging()
    psutil.cpu_percent(percpu=True)
    await asyncio.sleep(3 if cpu is None else 1)
    busy = psutil.cpu_percent(percpu=True)
    preflight = cpu is None
    if cpu is None:
        cpu = min(
            sorted(os.sched_getaffinity(0)),
            key=lambda i: max(busy[j] for j in (i, *_siblings(i))),
        )
    row = interval(
        before,
        _paging(),
        busy,
        psutil.virtual_memory().available,
        cpu,
        _siblings(cpu),
        preflight=preflight,
    )
    with path.open("a", encoding="utf-8") as log:
        log.write(
            json.dumps(
                {
                    "preflight": preflight,
                    "utc_seconds": time.time(),
                    "interval_seconds": time.monotonic() - started,
                    **row,
                }
            )
            + "\n"
        )
    if not row["suitable"]:
        raise ValueError(f"sparse changelog resource gate: {row}")
    return cpu


async def guarded_measure(
    root: Path, cpu: int, case: dict, sites: dict, releases: dict
) -> dict:
    async with asyncio.TaskGroup() as group:
        acquisition = group.create_task(
            measure_case(case, "interleaved", sites, releases, root / "measurement")
        )
        while not acquisition.done():
            await _resource(root / "resources.jsonl", cpu)
    return acquisition.result()


async def _measure(root: Path) -> None:
    releases = {k: load_release(root / k / "release.json") for k in SIDES}
    if releases["baseline"]["native_sha256"] == releases["candidate"]["native_sha256"]:
        raise ValueError("baseline and candidate must be distinct native builds")
    if any(
        releases["baseline"][k] != releases["candidate"][k]
        for k in ("image", "image_version")
    ):
        raise ValueError("wheel build images differ")
    sites = {k: (await install(releases[k], root / k), ROOT) for k in SIDES}
    for side, (site, _) in sites.items():
        verify_package_files(Path(releases[side]["wheel_path"]), site)
    fixture = root / "sparse.arrow"
    fixture.write_bytes(gzip.decompress(FIXTURE.read_bytes()))
    case = {**CASE, "fixture": str(fixture.resolve())}
    resource = root / "resources.jsonl"
    cpu = await _resource(resource, None)
    os.sched_setaffinity(0, {cpu})
    row = await guarded_measure(root, cpu, case, sites, releases)
    write_json(root / "samples.json", row)
    if row["status"] != "ok":
        raise ValueError(f"paired measurement incomplete: {row.get('error')}")
    result = row["result"]
    write_json(root / "result.json", result)
    print(json.dumps(result))


async def measure(root: Path, seconds: float = 135) -> None:
    started = time.monotonic()
    outcome = {"state": "interrupted"}
    try:
        await asyncio.wait_for(_measure(root), timeout=seconds)
    except Exception as error:
        outcome = {"state": "stopped", "error": repr(error)}
        raise
    else:
        outcome = {"state": "completed"}
    finally:
        write_json(
            root / "outcome.json",
            {**outcome, "elapsed_seconds": time.monotonic() - started},
        )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("operation", choices=("seal", "measure"))
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    root = args.output.resolve()
    if args.operation == "measure":
        asyncio.run(measure(root))
        return
    repository = os.environ["GITHUB_REPOSITORY"]
    requested = {
        s: {
            "sha": os.environ[f"{s.upper()}_SHA"],
            "artifact": int(os.environ[f"{s.upper()}_ARTIFACT"]),
        }
        for s in SIDES
    }
    write_json(
        root / "registration.json",
        {
            "requested": requested,
            "case": CASE["id"],
            "rounds": ROUNDS,
            "pairs_per_round": SAMPLES,
            "warmups_per_side_per_round": 1,
            "cleanup_reserve_seconds": 45,
            "acquisition_seconds": 180,
            "job_seconds": 600,
            "harness_sha256": sha256_file(ROOT / "benchmarks/sparse_changelog.py"),
            "fixture_gzip_sha256": sha256_file(FIXTURE),
            "workflow_sha": os.environ["GITHUB_SHA"],
            "workflow_run": os.environ["GITHUB_RUN_ID"],
            "runner_image": os.environ.get("ImageOS"),  # noqa: SIM112
            "runner_image_version": os.environ.get("ImageVersion"),  # noqa: SIM112
            "locked_requirements_sha256": sha256_file(
                ROOT / "benchmarks/requirements.lock"
            ),
        },
        exclusive=True,
    )
    for side, ref in requested.items():
        seal(repository, ref["artifact"], ref["sha"], root / side)


if __name__ == "__main__":
    main()
