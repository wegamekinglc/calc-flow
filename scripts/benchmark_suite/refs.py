"""Resolve immutable event refs before any build or measurement starts."""

from __future__ import annotations

import os
import re
import subprocess
from pathlib import Path

from scripts.benchmark_suite.provenance import ROOT


def resolve_refs() -> tuple[str, str]:
    head = os.environ.get("BENCHMARK_HEAD_SHA", "")
    base = os.environ.get("BENCHMARK_BASE_SHA", "")
    if not re.fullmatch(r"[0-9a-f]{40}", head):
        raise ValueError("BENCHMARK_HEAD_SHA must be a full commit SHA")
    if not base or base == "0" * 40:
        base = subprocess.check_output(
            ["git", "rev-parse", "HEAD^"], cwd=ROOT, shell=False, text=True
        ).strip()
    if not re.fullmatch(r"[0-9a-f]{40}", base):
        raise ValueError("BENCHMARK_BASE_SHA must be a full commit SHA")
    for value in (base, head):
        actual = subprocess.check_output(
            ["git", "rev-parse", "--verify", value + "^{commit}"],
            cwd=ROOT,
            shell=False,
            text=True,
        ).strip()
        if value != actual:
            raise ValueError("resolved commit does not match the requested SHA")
    return base, head


def write_refs(destination: Path) -> None:
    base, head = resolve_refs()
    with destination.open("a", encoding="utf-8") as output:
        output.write(f"base={base}\nhead={head}\n")
