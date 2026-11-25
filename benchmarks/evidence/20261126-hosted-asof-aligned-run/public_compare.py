"""Check the paired installed ASOF products."""

from __future__ import annotations

import argparse
import asyncio
import importlib.metadata
import json
import os
import platform
import re
import subprocess
import sys
import time
from pathlib import Path

from packaging.requirements import Requirement
from resources import observe_cpus

from benchmarks.native_release_identity import verify_package_files
from scripts.benchmark_suite.catalog import engine_cases
from scripts.benchmark_suite.measure import measure_case
from scripts.benchmark_suite.process import install
from scripts.benchmark_suite.release import load_release
from scripts.toolkit import linear_percentile, sha256_file, write_json

ROOT = Path(os.environ["GITHUB_WORKSPACE"])
RELEASE = "7012b5b409553df04e29e6d7f9a11abf919cbf93"
ROWS = 10_000_000
SIDES = ("baseline", "candidate")


def verify_build(release, source, expected):
    def git(*args):
        return subprocess.check_output(["git", "-C", source, *args], text=True).strip()

    if git("status", "--porcelain") or git("rev-parse", "HEAD") != expected:
        raise ValueError("source worktree does not identify the registered commit")
    if release["git_sha"] != expected or release["cargo_lock_sha256"] != sha256_file(
        source / "Cargo.lock"
    ):
        raise ValueError("source checkout differs from the sealed release build")
    return {
        "sha": expected,
        "tree": git("rev-parse", "HEAD^{tree}"),
        "release": release,
    }


def locked_versions():
    versions = {}
    for line in (ROOT / "benchmarks/requirements.lock").read_text().splitlines():
        if not re.match(r"^[A-Za-z0-9].*==", line):
            continue
        req = Requirement(line.rstrip(" " + chr(92)))
        if req.marker and not req.marker.evaluate():
            continue
        version = importlib.metadata.version(req.name)
        if version not in req.specifier:
            raise ValueError(f"wrong version of {req.name}")
        versions[req.name] = version
    return versions


async def run(args):
    selected = (
        row
        for row in engine_cases(ROWS)
        if row["id"] == f"engines/{ROWS}/calc-flow-stream/{args.scenario}"
    )
    (case,) = selected
    if "variant" in case:
        raise ValueError("expected the scheduled default managed workload")
    releases = dict(
        zip(
            SIDES,
            (load_release(manifest) for manifest in (args.baseline, args.candidate)),
            strict=True,
        )
    )
    identity = {
        "baseline": verify_build(
            releases["baseline"], args.baseline_source, args.baseline_sha
        ),
        "candidate": verify_build(
            releases["candidate"], args.candidate_source, args.sha
        ),
    }
    if releases["baseline"]["native_sha256"] == releases["candidate"]["native_sha256"]:
        raise ValueError("expected independent native builds")
    parent_sha = "5540d317884b59c53a8a9f80ac7129b78314c694"
    parent_lock = "8a72c638ce13b4ca2141d8b9b22c316fc4171813d28dc789e47a65a62ff80a72"
    registered_locks = {
        "baseline": {
            RELEASE: "d2b0e0ea1469156fffda9aa0fa5d80fae1d9d1cb1437c63256c7e584141bd5e5",
            parent_sha: parent_lock,
        }[args.baseline_sha],
        "candidate": parent_lock,
    }
    if {
        side: release["cargo_lock_sha256"] for side, release in releases.items()
    } != registered_locks:
        raise ValueError(
            "release or candidate Cargo.lock differs from its registered product"
        )
    for key in ("rustc", "python", "python_executable", "command"):
        # The output directory is the last argument; the build flags must match.
        a, b = (r[key] for r in releases.values())
        if (a[:-1] != b[:-1]) if key == "command" else (a != b):
            raise ValueError(f"release build identity differs at {key}")
    write_json(
        args.output / "qualification.json",
        {
            "builds": identity,
            "scenario": case,
            "controller_sha256": sha256_file(Path(__file__)),
            "bench_lock_sha256": sha256_file(ROOT / "benchmarks/requirements.lock"),
            "python": sys.version,
            "uname": tuple(platform.uname()),
            "dependencies": locked_versions(),
        },
        exclusive=True,
    )
    sites = {}
    for side, release in releases.items():
        site = await install(release, args.output / side)
        verify_package_files(Path(release["wheel_path"]), site)

        from setup_sites import SCRIPT

        from scripts.benchmark_suite.measure import validate_environment
        from scripts.benchmark_suite.process import child_environment, command

        log = args.output / side / "installed-interpreter.json"
        await command(
            [sys.executable, "-c", SCRIPT],
            cwd=ROOT,
            log=log,
            env=child_environment(site, source=ROOT),
            timeout=8,
        )
        env = json.loads(log.read_text())
        if (
            Path(env["native_path"]).parent != site / "calc_flow"
            or Path(env["package_path"]).parent != site / "calc_flow"
        ):
            raise ValueError("installed public entry point differs from qualified site")
        validate_environment(
            env, release, backend="calc-flow-stream", polars_threads=4, tokio_threads=4
        )
        sites[side] = (
            site,
            ROOT / ("target/post-timer-worker-" + os.environ["ASOF_BASELINE"]),
        )
    cpus = await observe_cpus(args.output)
    os.sched_setaffinity(0, set(cpus))
    async with asyncio.TaskGroup() as group:
        task = group.create_task(
            measure_case(
                case, "interleaved", sites, releases, args.output / "measurement"
            )
        )
        while not task.done():
            await observe_cpus(args.output, cpus)
    row = task.result()
    write_json(args.output / "samples.json", row)
    if row["status"] != "ok":
        raise ValueError(f"scheduled public case failed: {row.get('error')}")
    expected = {"passed": True, "rows": ROWS, "finite_rows": ROWS, "max_abs_error": 0.0}
    worker_files = list((args.output / "measurement").rglob("*.completed.json"))
    checks = {"prepare": [], "sample": []}
    for path in worker_files:
        record = json.loads(path.read_text())
        result = record.get("response", {})
        item = result.get("warmup", result)
        if isinstance(item, dict) and "seconds" in item and "correctness" in item:
            final = item["stream_evidence"]["terminal_asof_status"]
            if (
                item["stream_evidence"]["post_eof_output_rows"] != ROWS
                or final["state_rows"] != 0
                or final["matched_rows"] != ROWS
            ):
                raise ValueError("ASOF terminal oracle missing")
            checks[record["operation"]].append(item["correctness"])
    # Each request file retains the response and its sample context.
    if (
        len(checks["prepare"]) != 4
        or len(checks["sample"]) != 40
        or any(check != expected for items in checks.values() for check in items)
    ):
        raise ValueError("missing full, exact output for a warmup or formal execution")
    if len(row["evidence"]) != 2:
        raise ValueError("confirmation rounds missing")
    for evidence in row["evidence"]:
        if evidence["environment"]["cpu_affinity"] != cpus:
            raise ValueError("affinity changed")
        for side in SIDES:
            if len(evidence["samples"][side]) != 10:
                raise ValueError("incomplete sample inventory")
    stats = {
        **row["result"],
        "base_p95": linear_percentile([s for r in row["baseline"] for s in r], 0.95),
        "oracle_checked_executions": 44,
        "cpus": cpus,
    }
    write_json(args.output / "result.json", stats)
    print(json.dumps(stats), flush=True)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("scenario", choices=("asof_join",))
    for name in (
        "baseline",
        "candidate",
        "baseline-source",
        "candidate-source",
        "output",
    ):
        p.add_argument("--" + name, type=Path, required=True)
    p.add_argument("--sha", required=True)
    p.add_argument("--baseline-sha", required=True)
    args = p.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    start = time.monotonic()
    write_json(
        args.output / "started.json",
        {
            "epoch": time.time(),
            "sha": args.sha,
            "scenario": args.scenario,
            "budget": 250,
        },
    )
    state = {"state": "completed"}
    try:
        asyncio.run(asyncio.wait_for(run(args), timeout=250))
    except BaseException as exc:
        state = {"state": "stopped", "error": repr(exc)}
        raise
    finally:
        write_json(
            args.output / "outcome.json", {**state, "elapsed": time.monotonic() - start}
        )


if __name__ == "__main__":
    main()
