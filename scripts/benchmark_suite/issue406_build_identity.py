"""Validate the sealed controller correction against its original fresh build."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

from scripts.benchmark_suite.process import ROOT
from scripts.benchmark_suite.rust_harness import _clean_revision
from scripts.toolkit import command_output, sha256_file

REPAIR_FILES = {
    "crates/calc-flow/tests/stream_join_state.rs",
    "scripts/benchmark_suite/issue406_build_identity.py",
    "scripts/benchmark_suite/issue406_fresh_v3.py",
    "scripts/benchmark_suite/issue406_harness.py",
    "scripts/benchmark_suite/issue406_run.py",
    "scripts/test_issue406_build_identity.py",
    "scripts/test_issue406_fresh_harness.py",
    "scripts/test_issue406_fresh_v3.py",
}


def build_harness_commit(plan: dict) -> str:
    """Keep compiled artifact identity separate from controller identity."""
    return plan.get("build_harness_git_sha", plan["harness_git_sha"])


def _sealed(path: Path, files: dict) -> str:
    digest = sha256_file(path)
    if files.get(str(path)) != digest:
        raise ValueError(f"unsealed build/controller repair input: {path}")
    return digest


def _changed_files(plan: dict, source: Path, files: dict) -> dict:
    names = command_output(
        [
            "git",
            "diff",
            "--no-renames",
            "--name-only",
            build_harness_commit(plan),
            plan["harness_git_sha"],
        ],
        cwd=ROOT,
    ).splitlines()
    if not names or not set(names).issubset(REPAIR_FILES):
        raise ValueError("build reuse changes product, workloads or an unreviewed path")
    return {
        name: {
            "before_sha256": _sealed(source / name, files)
            if (source / name).exists()
            else None,
            "after_sha256": _sealed(ROOT / name, files),
        }
        for name in names
    }


def validate_build_harness(plan: dict, files: dict) -> None:
    """Require a reviewed exact diff and unchanged sealed original source."""
    keys = {"build_harness_git_sha", "build_harness_source", "execution_repair_witness"}
    if not keys.intersection(plan):
        return
    if not keys.issubset(plan):
        raise ValueError("split build identity requires its source and repair witness")
    source = Path(plan["build_harness_source"]).resolve()
    if _clean_revision(source) != build_harness_commit(plan):
        raise ValueError("original build harness revision changed")
    for name in command_output(["git", "ls-files"], cwd=source).splitlines():
        _sealed(source / name, files)
    witness = Path(plan["execution_repair_witness"]).resolve()
    _sealed(witness, files)
    patch = command_output(
        [
            "git",
            "diff",
            "--no-renames",
            "--no-ext-diff",
            "--binary",
            build_harness_commit(plan),
            plan["harness_git_sha"],
        ],
        cwd=ROOT,
    )
    expected = {
        "schema": "calc-flow.issue406-execution-repair.v1",
        "build_harness_git_sha": build_harness_commit(plan),
        "execution_git_sha": plan["harness_git_sha"],
        "patch_sha256": hashlib.sha256(patch.encode()).hexdigest(),
        "changed_files": _changed_files(plan, source, files),
    }
    if json.loads(witness.read_text()) != expected:
        raise ValueError("controller repair differs from the sealed reviewed diff")
