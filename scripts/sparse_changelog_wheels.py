"""Verify sparse comparison wheels against their Linux CI uploads and Git trees."""

from __future__ import annotations

import hashlib
import io
import json
import re
import subprocess  # nosec B404
import zipfile
from pathlib import Path

from scripts.toolkit import (
    FULL_SHA,
    require_executable,
    sha256_file,
    wheel_native_sha256,
    write_json,
)


def build_proof(artifact: dict, run: dict, expected: str, log: str) -> dict:
    expected_identity = (
        run["id"],
        expected,
        expected,
        ".github/workflows/ci-linux.yml",
        "python-distributions",
        False,
    )
    identity = (
        artifact["workflow_run"]["id"],
        artifact["workflow_run"]["head_sha"],
        run["head_sha"],
        run["path"],
        artifact["name"],
        artifact["expired"],
    )
    if not FULL_SHA.fullmatch(expected) or identity != expected_identity:
        raise ValueError("artifact does not belong to the requested Linux CI commit")
    checkout = re.search(
        r"git log -1 --format=%H\s*\n[^\n]*? ([0-9a-f]{40})(?:\s|$)", log
    )
    image = re.search(r"Image: (\S+)\s*\n[^\n]*?Version: (\S+)", log)
    upload = rf"successfully finalized\. Artifact ID {artifact['id']}\b"
    expected_logs = (
        "Finished `release` profile [optimized]",
        "uv build --wheel --config-setting 'build-args=--features pyo3/abi3-py313'",
    )
    if (
        checkout is None
        or image is None
        or not re.search(upload, log)
        or not all(line in log for line in expected_logs)
    ):
        raise ValueError(
            "missing release profile, checkout, image, upload or ABI proof"
        )
    return {"checkout": checkout[1], "image": image[1], "image_version": image[2]}


def _api(repository: str, path: str) -> bytes:
    allowed = (
        r"actions/artifacts/[0-9]+(?:/zip)?|actions/jobs/[0-9]+/logs|"
        r"actions/runs/[0-9]+(?:/attempts/[0-9]+/jobs\?per_page=100)?|"
        r"git/commits/[0-9a-f]{40}"
    )
    if not re.fullmatch(r"\w[\w.-]*/\w[\w.-]*", repository) or not re.fullmatch(
        allowed, path
    ):
        raise ValueError("invalid repository or Linux CI evidence endpoint")
    options = []
    if path.endswith("/logs") and b"--allow-escape-sequences" in _capture_gh_api(
        "--help"
    ):
        options.append("--allow-escape-sequences")
    return _capture_gh_api(
        "--hostname",
        "github.com",
        "--method",
        "GET",
        *options,
        f"repos/{repository}/{path}",
    )


def _capture_gh_api(*arguments: str) -> bytes:
    # Capture fixed help and allowlisted GETs in memory, including raw CI logs.
    return subprocess.check_output(  # noqa: S603  # nosec B603  # nosemgrep
        [require_executable("gh"), "api", *arguments],
        shell=False,
        timeout=60,
    )


def _wheel(content: bytes, digest: str, directory: Path) -> Path:
    if "sha256:" + hashlib.sha256(content).hexdigest() != digest:
        raise ValueError("artifact ZIP digest mismatch")
    with zipfile.ZipFile(io.BytesIO(content)) as archive:
        members = [name for name in archive.namelist() if name.endswith(".whl")]
        if len(members) != 1 or not re.fullmatch(
            r"calc_flow_python-[\w.]+-cp313-abi3-linux_x86_64\.whl", members[0]
        ):
            raise ValueError("expected one Linux x86_64 CPython 3.13 ABI3 core wheel")
        path = directory / members[0]
        with path.open("xb") as target:
            target.write(archive.read(members[0]))
    return path


def product_tree(
    registration: dict, sha: str, artifact: int, checkout: str, tree: str
) -> str:
    observed = (sha, artifact, checkout, tree)
    fields = ("requested_sha", "artifact", "checkout", "tree")
    expected = [
        tuple(record[key] for key in fields) for record in registration.values()
    ]
    if observed not in expected:
        raise ValueError(
            "artifact, workflow, checkout or tree differs from sealed product pair"
        )
    return tree


def seal(repository: str, artifact_id: int, sha: str, directory: Path) -> dict:
    if artifact_id <= 0 or not FULL_SHA.fullmatch(sha):
        raise ValueError("a positive artifact ID and full head SHA are required")

    def api(path):
        return json.loads(_api(repository, path))

    artifact = api(f"actions/artifacts/{artifact_id}")
    run_id = artifact["workflow_run"]["id"]
    run = api(f"actions/runs/{run_id}")
    jobs = api(
        f"actions/runs/{run_id}/attempts/{run['run_attempt']}/jobs?per_page=100"
    )["jobs"]
    (job,) = [
        j
        for j in jobs
        if j["name"] == "Build + isolated wheel smoke test"
        and j["conclusion"] == "success"
    ]
    proof = build_proof(
        artifact,
        run,
        sha,
        _api(repository, f"actions/jobs/{job['id']}/logs").decode("utf-8-sig"),
    )
    registration = (
        Path(__file__).resolve().parents[1] / "benchmarks/steady-wheel-pair.json"
    )
    tree = product_tree(
        json.loads(registration.read_text()),
        sha,
        artifact_id,
        proof["checkout"],
        api(f"git/commits/{proof['checkout']}")["tree"]["sha"],
    )
    directory.mkdir(parents=True, exist_ok=False)
    wheel = _wheel(
        _api(repository, f"actions/artifacts/{artifact_id}/zip"),
        artifact["digest"],
        directory,
    )
    manifest = {
        "contract": "benchmark-release-v1",
        "build_profile": "release",
        "git_clean": True,
        "git_sha": proof["checkout"],
        "requested_sha": sha,
        "tree": tree,
        "wheel": wheel.name,
        "wheel_sha256": sha256_file(wheel),
        "native_sha256": wheel_native_sha256(wheel),
        "workflow_run": run_id,
        "job": job["id"],
        "artifact": artifact_id,
        "artifact_digest": artifact["digest"],
        **proof,
    }
    write_json(directory / "release.json", manifest)
    return manifest
