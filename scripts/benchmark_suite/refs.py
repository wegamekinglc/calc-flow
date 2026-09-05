"""Resolve immutable event refs before any build or measurement starts."""

from __future__ import annotations

import os
import re
import shutil
import subprocess  # nosec B404 -- SHA-validated, fixed git argv only
from pathlib import Path

from scripts.benchmark_suite.provenance import ROOT


def git_revision(revision: str) -> str:
    if revision != "HEAD^" and not re.fullmatch(r"[0-9a-f]{40}\^\{commit\}", revision):
        raise ValueError("unsupported benchmark revision")
    executable = shutil.which("git")
    if executable is None:
        raise ValueError("git is required to resolve benchmark refs")
    # Only the literal first parent or a checked full commit SHA reaches git.
    # Fixed command, absolute executable, strict revision allowlist, and no shell;
    # dynamic-command scanners cannot track the fullmatch guard above.
    return subprocess.check_output(  # nosec B603  # nosemgrep
        [
            str(Path(executable).resolve()),
            "rev-parse",
            "--verify",
            revision,
        ],  # nosemgrep
        cwd=ROOT,
        shell=False,
        text=True,
    ).strip()


def resolve_refs() -> tuple[str, str]:
    head = os.environ.get("BENCHMARK_HEAD_SHA", "")
    base = os.environ.get("BENCHMARK_BASE_SHA", "")
    if not re.fullmatch(r"[0-9a-f]{40}", head):
        raise ValueError("BENCHMARK_HEAD_SHA must be a full commit SHA")
    if not base or base == "0" * 40:
        base = git_revision("HEAD^")
    if not re.fullmatch(r"[0-9a-f]{40}", base):
        raise ValueError("BENCHMARK_BASE_SHA must be a full commit SHA")
    for value in (base, head):
        actual = git_revision(value + "^{commit}")
        if value != actual:
            raise ValueError("resolved commit does not match the requested SHA")
    return base, head


def write_refs(destination: Path) -> None:
    base, head = resolve_refs()
    with destination.open("a", encoding="utf-8") as output:
        output.write(f"base={base}\nhead={head}\n")
