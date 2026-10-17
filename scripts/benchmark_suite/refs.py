"""Resolve immutable event refs before any build or measurement starts."""

from __future__ import annotations

import os
import re
import shutil
import subprocess  # nosec B404 -- validated, fixed git argv only
from dataclasses import dataclass
from pathlib import Path

from scripts.benchmark_suite.provenance import ROOT
from scripts.toolkit import FULL_SHA

# Calendar release tags, as documented in docs/python-release.md.
RELEASE_TAG = re.compile(
    r"calc-flow-python-v(?:0|[1-9][0-9]*)\.(?:0|[1-9][0-9]*)\.(?:0|[1-9][0-9]*)"
)
_COMMIT_REVISION = re.compile(FULL_SHA.pattern + r"\^\{commit\}")
_RELEASE_REVISION = re.compile(r"refs/tags/" + RELEASE_TAG.pattern + r"\^\{commit\}")
# ref-filter exits 0 with empty output when no tag is reachable, unlike
# ``git describe``; version order ranks 2026.10.x above 2026.9.x.
_LATEST_RELEASE_TAG = (
    "for-each-ref",
    "--merged=HEAD",
    "--sort=-version:refname",
    "--count=1",
    "--format=%(refname:lstrip=2)",
    "refs/tags/calc-flow-python-v*",
)
_BASELINE_LABELS = {
    "explicit": "explicit",
    "release": "release",
    "first-parent": "first parent",
}


@dataclass(frozen=True)
class BenchmarkRefs:
    base: str
    head: str
    baseline_kind: str
    baseline_ref: str
    rust_base: str


def _git(*arguments: str) -> str:
    executable = shutil.which("git")
    if executable is None:
        raise ValueError("git is required to resolve benchmark refs")
    # Fixed command, absolute executable, caller-allowlisted arguments, and no
    # shell; dynamic-command scanners cannot track the callers' fullmatch guards.
    return subprocess.check_output(  # nosec B603  # nosemgrep
        [str(Path(executable).absolute()), *arguments],  # nosemgrep
        cwd=ROOT,
        shell=False,
        text=True,
    ).strip()


def git_revision(revision: str) -> str:
    # Only the literal first parent, a checked full commit SHA, or a checked
    # release tag reaches git.
    if revision != "HEAD^" and not (
        _COMMIT_REVISION.fullmatch(revision) or _RELEASE_REVISION.fullmatch(revision)
    ):
        raise ValueError("unsupported benchmark revision")
    return _git("rev-parse", "--verify", revision)


def latest_release_tag() -> str | None:
    """Return the newest release tag reachable from HEAD, or ``None``."""

    tag = _git(*_LATEST_RELEASE_TAG)
    if not tag:
        return None
    if not RELEASE_TAG.fullmatch(tag):
        raise ValueError("unsupported benchmark release tag")
    return tag


def _implicit_baseline(event: str, head: str) -> tuple[str, str, str, str]:
    # Scheduled runs stay anchored to the last release so a regression remains
    # visible until it is fixed or a release is cut deliberately. The Rust
    # shard compiles the candidate bench file against the baseline source, so
    # it keeps the first parent: a release checkout predates bench-file APIs
    # the candidate harness imports.
    if event == "schedule":
        tag = latest_release_tag()
        if tag is not None:
            release = git_revision(f"refs/tags/{tag}^{{commit}}")
            if release != head:
                return release, "release", tag, git_revision("HEAD^")
    parent = git_revision("HEAD^")
    return parent, "first-parent", "HEAD^", parent


def _verify_commit(sha: str) -> None:
    """Re-resolve one full SHA and refuse a rewritten history."""

    if git_revision(sha + "^{commit}") != sha:
        raise ValueError("resolved commit does not match the requested SHA")


def _verify_rust_base(rust_base: str, base: str) -> None:
    """Verify the Rust shard baseline only when it diverges from the base."""

    if rust_base == base:
        return
    if not FULL_SHA.fullmatch(rust_base):
        raise ValueError("the Rust baseline must be a full commit SHA")
    _verify_commit(rust_base)


def resolve_refs() -> BenchmarkRefs:
    head = os.environ.get("BENCHMARK_HEAD_SHA", "")
    base = os.environ.get("BENCHMARK_BASE_SHA", "")
    event = os.environ.get("BENCHMARK_EVENT", "")
    if not FULL_SHA.fullmatch(head):
        raise ValueError("BENCHMARK_HEAD_SHA must be a full commit SHA")
    if base and base != "0" * 40:
        base, kind, ref, rust_base = base, "explicit", base, base
    else:
        base, kind, ref, rust_base = _implicit_baseline(event, head)
    if not FULL_SHA.fullmatch(base):
        raise ValueError("BENCHMARK_BASE_SHA must be a full commit SHA")
    for value in (base, head):
        _verify_commit(value)
    _verify_rust_base(rust_base, base)
    return BenchmarkRefs(base, head, kind, ref, rust_base)


def describe_baseline(kind: str, ref: str, base: str) -> str:
    """Render resolved baseline provenance for the complete report header."""

    if kind not in _BASELINE_LABELS or not FULL_SHA.fullmatch(base):
        raise ValueError(f"unresolved benchmark baseline: kind={kind!r}")
    if kind == "release" and not RELEASE_TAG.fullmatch(ref):
        raise ValueError("release baseline requires a release tag")
    label = f"release {ref}" if kind == "release" else _BASELINE_LABELS[kind]
    return f"{label} (`{base}`)"


def write_refs(destination: Path) -> None:
    refs = resolve_refs()
    with destination.open("a", encoding="utf-8") as output:
        output.write(
            f"base={refs.base}\nhead={refs.head}\n"
            f"baseline_kind={refs.baseline_kind}\n"
            f"baseline_ref={refs.baseline_ref}\n"
            f"rust_base={refs.rust_base}\n"
        )
