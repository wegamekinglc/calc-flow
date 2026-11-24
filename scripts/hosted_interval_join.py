"""Single-core public interval join comparison for the admitted-row iterator."""

from __future__ import annotations

import asyncio
import json
import os
import sys
import time
from datetime import datetime
from pathlib import Path

from benchmarks.native_release_identity import verify_package_files
from scripts.benchmark_suite.catalog import engine_cases
from scripts.benchmark_suite.process import ROOT, install
from scripts.benchmark_suite.release import load_release
from scripts.sparse_changelog_ci import _resource, guarded_measure
from scripts.sparse_changelog_wheels import _api, seal
from scripts.toolkit import (
    linear_percentile,
    sha256_file,
    write_json,
)

CASE_ID = "engines/1000000/calc-flow-stream-1t/interval_join"


def case():
    (chosen,) = (row for row in engine_cases(1_000_000) if row["id"] == CASE_ID)
    return chosen


BUDGET = 300
ROW_COUNT = 10998080
SIDES = ("baseline", "candidate")
CONFIG = ROOT / "benchmarks/hosted-product-registration.json"
OUT = ROOT / "target/hosted-evidence"


def remaining_seconds(started: float, now: float) -> float:
    remaining = BUDGET - (now - started) - 45
    if remaining <= 0:
        raise TimeoutError("acquisition budget exhausted before worker startup")
    return remaining


def acquire() -> None:
    registration = json.loads(CONFIG.read_text())
    repository = os.environ["GITHUB_REPOSITORY"]
    run = int(os.environ["GITHUB_RUN_ID"])
    attempt = int(os.environ["GITHUB_RUN_ATTEMPT"])
    jobs = json.loads(
        _api(repository, f"actions/runs/{run}/attempts/{attempt}/jobs?per_page=100")
    )
    (job,) = (job for job in jobs["jobs"] if job["name"] == "Public stream evidence")
    started = datetime.fromisoformat(job["started_at"]).timestamp()
    record = {
        **registration,
        "candidate": {
            "requested_sha": os.environ["CANDIDATE_SHA"],
            "artifact": int(os.environ["CANDIDATE_ARTIFACT"]),
        },
        "workflow_sha": os.environ["GITHUB_SHA"],
        "job": job["id"],
        "workflow_run": run,
        "start_epoch": started,
        "job_budget_seconds": BUDGET,
        "reserved_cleanup_seconds": 45,
        "registration_sha256": sha256_file(CONFIG),
        "requirements_sha256": sha256_file(ROOT / "benchmarks/requirements.lock"),
    }
    write_json(OUT / "registration.json", record, exclusive=True)
    if any(
        sha256_file(ROOT / path) != digest
        for path, digest in record["source_hashes"].items()
    ):
        raise ValueError("registered measurement sources changed")
    for side in SIDES:
        requested = record[side]
        manifest = seal(
            repository, requested["artifact"], requested["requested_sha"], OUT / side
        )
        if (
            side == "baseline"
            and requested["artifact_digest"] != manifest["artifact_digest"]
        ):
            raise ValueError("unexpected published baseline")
        load_release(OUT / side / "release.json")


async def observe() -> None:
    releases = {side: load_release(OUT / side / "release.json") for side in SIDES}
    if any(
        releases["baseline"][key] != releases["candidate"][key]
        for key in ("image", "image_version")
    ):
        raise ValueError("the two official wheel build images differ")
    if releases["baseline"]["native_sha256"] == releases["candidate"]["native_sha256"]:
        raise ValueError("expected two independent native builds")
    sites = {}
    for side, release in releases.items():
        site = await install(release, OUT / side)
        verify_package_files(Path(release["wheel_path"]), site)
        sites[side] = (site, ROOT)
    selected = case()

    cpu = await _resource(OUT / "resources.jsonl", None)
    os.sched_setaffinity(0, {cpu})
    row = await guarded_measure(OUT, cpu, selected, sites, releases)
    write_json(OUT / "samples.json", row)
    if row["status"] != "ok":
        raise ValueError(f"paired measurement failed: {row.get('error')}")
    samples = [s for r in row["evidence"] for arm in SIDES for s in r["samples"][arm]]
    if len(samples) != 40 or any(
        s["correctness"]["rows"] != ROW_COUNT for s in samples
    ):
        raise ValueError("complete output proof missing from measured pairs")
    base = [s for r in row["baseline"] for s in r]
    write_json(
        OUT / "result.json",
        {**row["result"], "base_p95": linear_percentile(base, 0.95)},
    )


def main():
    if sys.argv[1] == "acquire":
        acquire()
        return
    if sys.argv[1] != "measure":
        raise ValueError("expected acquire or measure")
    registered = json.loads((OUT / "registration.json").read_text())
    started = registered["start_epoch"]
    outcome = {"state": "interrupted"}
    try:
        asyncio.run(
            asyncio.wait_for(observe(), remaining_seconds(started, time.time()))
        )
    except Exception as exc:
        outcome = {"state": "stopped", "error": repr(exc)}
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
