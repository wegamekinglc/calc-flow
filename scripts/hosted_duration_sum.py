"""Single-core public single-duration-SUM comparison."""

from __future__ import annotations

import asyncio
import json
import os
import re
import sys
import time
from datetime import datetime
from pathlib import Path

from benchmarks.native_release_identity import verify_package_files
from scripts.benchmark_suite.process import ROOT, child_environment, command, install
from scripts.benchmark_suite.release import load_release
from scripts.sparse_changelog_ci import _resource, guarded_measure
from scripts.sparse_changelog_wheels import _api, _wheel, build_proof
from scripts.toolkit import (
    linear_percentile,
    sha256_file,
    wheel_native_sha256,
    write_json,
)


def case():
    return {
        "id": "python/public-duration-sum-60/fresh-stream",
        "family": "public-duration-sum",
        "rows": 100_000,
        "backend": "calc-flow-stream-1t",
    }


BUDGET = 105
ROW_COUNT = 100000
SIDES = ("baseline", "candidate")
CONFIG = ROOT / "benchmarks/hosted-product-registration.json"
OUT = ROOT / "target/hosted-evidence"


def remaining_seconds(started: float, now: float) -> float:
    remaining = BUDGET - (now - started) - 45
    if remaining <= 0:
        raise TimeoutError("acquisition budget exhausted before worker startup")
    return remaining


def download_receipt(manifest: dict, wheel: Path) -> dict:
    return {
        **manifest,
        "wheel": wheel.name,
        "wheel_sha256": sha256_file(wheel),
        "native_sha256": wheel_native_sha256(wheel),
    }


def ci_receipt(repository: str, registered: dict) -> dict:
    def api(path: str) -> dict:
        return json.loads(_api(repository, path))

    run_id = registered["workflow_run"]
    run = api(f"actions/runs/{run_id}")
    jobs = api(
        f"actions/runs/{run_id}/attempts/{run['run_attempt']}/jobs?per_page=100"
    )["jobs"]
    if not any(
        j["id"] == registered["job"]
        and j["name"] == "Build + isolated wheel smoke test"
        and j["conclusion"] == "success"
        for j in jobs
    ):
        raise ValueError("registered release smoke job did not succeed")
    log = _api(repository, f"actions/jobs/{registered['job']}/logs").decode("utf-8-sig")
    uploads = re.findall(
        r"Artifact python-distributions successfully finalized\. Artifact ID (\d+)\b",
        log,
    )
    if len(uploads) != 1:
        raise ValueError("expected one finalized release upload in the tested job")
    artifact = api(f"actions/artifacts/{uploads[0]}")
    proof = build_proof(artifact, run, registered["requested_sha"], log)
    if (
        proof["checkout"] != registered["checkout"]
        or api(f"git/commits/{proof['checkout']}")["tree"]["sha"] != registered["tree"]
    ):
        raise ValueError("wheel source differs from registered product commit or tree")
    return {
        **registered,
        **proof,
        "git_sha": proof["checkout"],
        "artifact": artifact["id"],
        "artifact_digest": artifact["digest"],
        "git_clean": True,
        "build_profile": "release",
        "contract": "benchmark-release-v1",
    }


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
    manifests = {side: ci_receipt(repository, record[side]) for side in SIDES}
    if any(
        manifests["baseline"][key] != manifests["candidate"][key]
        for key in ("image", "image_version")
    ):
        raise ValueError("wheel build images differ")
    for side, manifest in manifests.items():
        directory = OUT / side
        directory.mkdir()
        wheel = _wheel(
            _api(repository, f"actions/artifacts/{manifest['artifact']}/zip"),
            manifest["artifact_digest"],
            directory,
        )
        write_json(directory / "release.json", download_receipt(manifest, wheel))
        load_release(directory / "release.json")


def verify_fixture(path: Path) -> str:
    digest = sha256_file(path)
    if digest != "36847ac453bd58a3a0a918d993ec726d56bbddb8c3af6e8362ca940999286ab0":
        raise ValueError("duration fixture differs from committed PR 638 IPC bytes")
    return digest


async def observe() -> None:
    releases = {side: load_release(OUT / side / "release.json") for side in SIDES}
    sites = {}
    for side, release in releases.items():
        site = await install(release, OUT / side)
        verify_package_files(Path(release["wheel_path"]), site)
        sites[side] = (site, ROOT)
    selected = case()
    fixture_path = OUT / "duration.arrow"
    await command(
        [sys.executable, "-m", "benchmarks.public_duration_sum", str(fixture_path)],
        cwd=ROOT,
        log=OUT / "fixture.log",
        env=child_environment(sites["baseline"][0]),
    )
    selected = {**selected, "fixture": str(fixture_path)}
    write_json(
        OUT / "fixture.json",
        {"sha256": verify_fixture(fixture_path), "rows": ROW_COUNT},
    )
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
