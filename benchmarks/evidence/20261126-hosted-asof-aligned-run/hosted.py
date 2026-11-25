"""Build one sealed product or compare one frozen pair on its own hosted VM."""

from __future__ import annotations

import gzip
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

from scripts.benchmark_suite.report import _paired_verdict
from scripts.benchmark_suite.statistics import paired_round
from scripts.toolkit import sha256_file

ROOT = Path(os.environ["GITHUB_WORKSPACE"])
STUDY = Path(__file__).resolve().parent
OUT = ROOT / "target" / "products"
RAW = ROOT / "target" / "evidence" / os.environ.get("ASOF_BASELINE", "candidate")
REFS = {
    "release": "7012b5b409553df04e29e6d7f9a11abf919cbf93",
    "parent": "f67422bac9c0e23a8d22e717422c7b5bb329f723",
    "candidate": "40689a3318ea866e5e948e5ad2fdc2788cfa39b5",
}


def command(*argv, cwd=ROOT, **opts):
    subprocess.run([str(a) for a in argv], cwd=cwd, check=True, **opts)


def checkout(path, sha):
    command("git", "worktree", "add", "--quiet", "--detach", path, sha)
    assert (
        subprocess.check_output(
            ["git", "-C", path, "rev-parse", "HEAD"], text=True
        ).strip()
        == sha
    )


def build(surface, name):
    sha = REFS[name]
    src = ROOT / "target/product-source"
    checkout(src, sha)
    OUT.mkdir()
    if surface == "rust":
        command(sys.executable, STUDY / "build.py", src, sha, "job")
        for f in [*STUDY.glob(f"job-{sha[:10]}*"), *STUDY.glob(f"{sha[:10]}-build.*")]:
            shutil.copy2(f, OUT)
    else:
        command(
            sys.executable,
            "-m",
            "scripts.benchmark_suite",
            "build",
            "--source",
            src,
            "--output",
            OUT,
        )


def compare(surface, name):
    assert name in ("release", "parent") and surface in ("rust", "python")
    base, head = REFS[name], REFS["candidate"]
    RAW.mkdir(exist_ok=True)
    for f in (
        *STUDY.glob("*.py"),
        *STUDY.glob("*.rs"),
        *STUDY.glob("*.md"),
        STUDY / "project.json",
        STUDY / "registered.sha256",
        STUDY / "plain-artifacts.json",
        STUDY / "worker.patch.gz",
    ):
        shutil.copy2(f, RAW)
    (RAW / "bound.json").write_text(
        json.dumps(dict(baseline=base, candidate=head, workload="10M-4T-64-keys"))
    )
    checkouts = {}
    for side in (name, "candidate"):
        checkout_path = ROOT / "target" / ("source-" + name + "-" + side)
        checkout(checkout_path, REFS[side])
        checkouts[side] = checkout_path
    if surface == "rust":
        for side, sha in ((name, base), ("candidate", head)):
            for f in (OUT / side).glob("*"):
                shutil.copy2(f, RAW)
            program = RAW / f"job-{sha[:10]}"
            program.chmod(0o755)
            receipt = json.loads(program.with_suffix(".receipt.json").read_text())
            assert (
                receipt["source"]
                == subprocess.check_output(
                    [
                        "git",
                        "-C",
                        checkouts[side],
                        "rev-parse",
                        "HEAD:crates/calc-flow/src",
                    ],
                    text=True,
                ).strip()
            )
            assert receipt["lock_sha256"] == sha256_file(checkouts[side] / "Cargo.lock")
            assert receipt["harness_sha256"] == sha256_file(RAW / "stage_job.rs")
        command(sys.executable, RAW / "measure.py", "job", head, base)
        command(sys.executable, RAW / "verify.py", next(RAW.glob("obs-job-*")))
    else:
        worker = ROOT / ("target/post-timer-worker-" + name)
        checkout(worker, "e3104f444688ab3890c4b45f6098c28e052de7a8")
        patch = gzip.decompress((STUDY / "worker.patch.gz").read_bytes())
        command("git", "apply", "--check", input=patch, cwd=worker)
        command("git", "apply", input=patch, cwd=worker)
        command(
            sys.executable,
            RAW / "public_compare.py",
            "asof_join",
            "--baseline",
            OUT / name / "release.json",
            "--baseline-sha",
            base,
            "--baseline-source",
            checkouts[name],
            "--candidate",
            OUT / "candidate/release.json",
            "--sha",
            head,
            "--candidate-source",
            checkouts["candidate"],
            "--output",
            RAW / "python",
        )
        results = json.loads((RAW / "python/samples.json").read_text())
        result = json.loads((RAW / "python/result.json").read_text())
        rounds = [
            paired_round(
                [x["seconds"] for x in r["samples"]["baseline"]],
                [x["seconds"] for x in r["samples"]["candidate"]],
            )
            for r in results["evidence"]
        ]
        assert (
            rounds == result["round_intervals"]
            and _paired_verdict(rounds) == result["verdict"]
        )


if __name__ == "__main__":
    {"build": build, "compare": compare}[sys.argv[1]](*sys.argv[2:])
