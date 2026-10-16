"""Explicit build/execution provenance for frozen milestone products."""

from __future__ import annotations

import re
from pathlib import Path

from scripts.toolkit import git_output, sha256_file

POLICY = "calc-flow.issue406-build-execution.v1"
COMPILED_INPUTS = (
    "crates",
    "python",
    "Cargo.toml",
    "Cargo.lock",
    "pyproject.toml",
    ".cargo",
    "rust-toolchain",
    "rust-toolchain.toml",
    "build.rs",
)
RECIPE_INPUTS = (
    "scripts/benchmark_suite/issue406_harness.py",
    "scripts/benchmark_suite/rust_harness.py",
    "scripts/benchmark_suite/rust_provenance.py",
    "scripts/toolkit.py",
)
REPORT_FIELDS = (
    "source_policy",
    "build_harness_source",
    "build_execution_diff",
    "preparation_main_git_sha",
)


def python_sources(source: Path) -> set[str]:
    return {
        str(path.resolve())
        for directory in (source / "benchmarks", source / "scripts/benchmark_suite")
        for path in directory.rglob("*.py")
    } | {str((source / "scripts/toolkit.py").resolve())}


def validate_declaration(plan: dict) -> None:
    policy = plan.get("source_policy")
    build = plan.get("build_harness_git_sha")
    if policy is None:
        if build != plan["harness_git_sha"]:
            raise ValueError("milestone requires one sealed tool SHA by default")
        return
    if policy != POLICY or re.fullmatch(r"[0-9a-f]{40}", str(build)) is None:
        raise ValueError("unknown milestone source policy or invalid build SHA")
    _validate_source_descriptor(plan)


def _validate_source_descriptor(plan: dict) -> None:
    source = plan.get("build_harness_source")
    descriptor = plan.get("build_execution_diff", {})
    if not isinstance(source, str) or not Path(source).is_absolute():
        raise ValueError("independent build source requires an absolute checkout path")
    if not isinstance(descriptor, dict):
        raise ValueError("independent build source requires a sealed exact Git diff")
    if (
        not isinstance(descriptor.get("path"), str)
        or not descriptor["path"]
        or re.fullmatch(r"[0-9a-f]{64}", str(descriptor.get("sha256"))) is None
    ):
        raise ValueError("independent build source requires a sealed exact Git diff")
    if re.fullmatch(r"[0-9a-f]{40}", str(plan.get("preparation_main_git_sha"))) is None:
        raise ValueError("independent sources require the preparation main SHA")


def _validate_checkout(source: Path, revision: str, files: dict) -> None:
    if not source.is_dir():
        raise ValueError("build/execution source checkout is missing")
    _validate_checkout_python(source, revision, files)


def _validate_checkout_python(source: Path, revision: str, files: dict) -> None:
    if git_output(source, "rev-parse", "HEAD^{commit}") != revision or git_output(
        source, "status", "--porcelain", "--untracked-files=no"
    ):
        raise ValueError("sealed build/execution checkout revision differs")
    if Path(git_output(source, "rev-parse", "--show-toplevel")).resolve() != source:
        raise ValueError("build/execution source must be the checkout root")
    tracked = set(git_output(source, "ls-files").splitlines())
    for name in python_sources(source):
        path = Path(name)
        relative = path.relative_to(source).as_posix()
        if relative not in tracked or files.get(name) != sha256_file(path):
            raise ValueError(
                "build/execution Python or fixture tool is not tracked and sealed"
            )


def source_diff(execution: Path, build_sha: str, execution_sha: str) -> str:
    """Return the canonical Git binary patch, without surrounding whitespace."""
    return git_output(
        execution,
        "diff",
        "--binary",
        "--no-color",
        "--no-renames",
        "--no-ext-diff",
        "--no-textconv",
        build_sha,
        execution_sha,
        "--",
    )


def _validate_diff(plan: dict, execution: Path, files: dict) -> None:
    descriptor = plan["build_execution_diff"]
    path = Path(descriptor["path"])
    if files.get(str(path)) != descriptor["sha256"]:
        raise ValueError("build/execution Git diff is not sealed")
    if sha256_file(path) != descriptor["sha256"] or path.read_text(
        encoding="utf-8"
    ) != source_diff(
        execution, plan["build_harness_git_sha"], plan["execution_git_sha"]
    ):
        raise ValueError("build/execution Git diff differs from the actual commits")


def validate_binding(plan: dict, execution: Path, files: dict) -> None:
    """Verify both real checkouts; never reinterpret a receipt's original identity."""
    if plan.get("source_policy") != POLICY:
        return
    validate_declaration(plan)
    execution = execution.resolve()
    build = Path(plan["build_harness_source"]).resolve()
    _validate_checkout(build, plan["build_harness_git_sha"], files)
    _validate_checkout(execution, plan["execution_git_sha"], files)
    paths = (*COMPILED_INPUTS, *RECIPE_INPUTS)
    original = git_output(build, "ls-tree", plan["build_harness_git_sha"], "--", *paths)
    current = git_output(execution, "ls-tree", plan["execution_git_sha"], "--", *paths)
    if original != current:
        raise ValueError("compiled inputs or original build recipe changed")
    _validate_main_inputs(
        execution, plan["execution_git_sha"], plan["preparation_main_git_sha"]
    )
    _validate_diff(plan, execution, files)


def validate_native_identity(plan, build, descriptor, product):
    if (plan.get("schema"), plan.get("source_policy")) != (
        "calc-flow.issue406-milestone.v1",
        POLICY,
    ):
        return
    if (
        build.get("harness_git_sha") != plan["build_harness_git_sha"]
        or build.get("product_commit") != product
        or build.get("binary_sha256") != descriptor["sha256"]
    ):
        raise ValueError("compiled native build/product/binary identity differs")


def report_fields(plan: dict) -> dict:
    return {key: plan[key] for key in REPORT_FIELDS if key in plan}


def _validate_main_inputs(execution: Path, revision: str, main: str) -> None:
    paths = (*COMPILED_INPUTS, "benchmarks")
    current = git_output(execution, "ls-tree", revision, "--", *paths)
    observed = git_output(execution, "ls-tree", main, "--", *paths)
    if current != observed:
        raise ValueError("preparation main compiled inputs or fixture tools differ")


def preparation_guard(
    execution: Path, expected_execution: str, expected_main: str
) -> dict:
    if not all(
        re.fullmatch(r"[0-9a-f]{40}", str(value))
        for value in (expected_execution, expected_main)
    ):
        raise ValueError("preparation source guard requires both exact SHAs")
    if git_output(
        execution, "rev-parse", "HEAD^{commit}"
    ) != expected_execution or git_output(
        execution, "status", "--porcelain", "--untracked-files=no"
    ):
        raise ValueError("preparation execution checkout differs")
    remote = git_output(execution, "ls-remote", "origin", "refs/heads/main").split()
    if not remote or remote[0] != expected_main:
        raise ValueError("preparation main/source guard failed")
    _validate_main_inputs(execution, expected_execution, expected_main)
    return {
        "execution_git_sha": expected_execution,
        "remote_main_git_sha": expected_main,
    }


def validate_anchor(plan: dict, anchor: dict) -> None:
    if plan.get("source_policy") == POLICY and anchor.get("source_guard") != {
        "execution_git_sha": plan["execution_git_sha"],
        "remote_main_git_sha": plan["preparation_main_git_sha"],
    }:
        raise ValueError("preparation anchor lacks its verified source guard")
