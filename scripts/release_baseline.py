#!/usr/bin/env python3
"""Resolve an auditable performance baseline without skipping first-release gates."""

from __future__ import annotations

import argparse
import re

# Fixed Git commands with separate ref arguments; never invoke a shell.
import subprocess  # nosec B404
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
_SHA_RE = re.compile(r"[0-9a-f]{40}")
_RELEASE_TAG_RE = re.compile(
    r"v(?:0|[1-9][0-9]*)\.(?:0|[1-9][0-9]*)\.(?:0|[1-9][0-9]*)"
)


def _git(root: Path, *arguments: str) -> str:
    # Ref arguments are namespaced or validated full SHAs.
    result = subprocess.run(  # nosec B603
        ("git", *arguments),
        cwd=root,
        shell=False,
        check=False,
        capture_output=True,
        text=True,
    )
    if result.returncode:
        raise ValueError(f"git {arguments[0]} failed: {result.stderr.strip()}")
    return result.stdout.strip()


def _tag_baseline(
    root: Path, tag: str, candidate: str, initial: str | None
) -> str | None:
    ref = f"refs/tags/{tag}"
    if _git(root, "cat-file", "-t", ref) != "tag":
        raise ValueError("release tag must be annotated")
    if _git(root, "rev-parse", f"{ref}^{{commit}}") != candidate:
        raise ValueError("release tag must point at the candidate HEAD")
    message = _git(root, "for-each-ref", "--format=%(contents)", ref)
    return _annotation_baseline(message, initial)


def _annotation_baseline(message: str, initial: str | None) -> str | None:
    annotations = [
        line.removeprefix("Benchmark-Baseline:").strip()
        for line in message.splitlines()
        if line.startswith("Benchmark-Baseline:")
    ]
    if len(annotations) > 1:
        raise ValueError("duplicate Benchmark-Baseline annotations")
    if not annotations:
        return initial
    if initial is not None and initial != annotations[0]:
        raise ValueError("input and tag annotation baselines disagree")
    return annotations[0]


def _previous_release(root: Path, candidate: str) -> str | None:
    tags = _git(root, "tag", "--merged", "HEAD", "--list", "v[0-9]*").splitlines()
    previous_tags = [
        name
        for name in tags
        if _RELEASE_TAG_RE.fullmatch(name) is not None
        and _git(root, "rev-parse", f"refs/tags/{name}^{{commit}}") != candidate
    ]
    if not previous_tags:
        return None
    arguments = tuple(value for name in previous_tags for value in ("--match", name))
    previous = _git(root, "describe", "--tags", "--abbrev=0", *arguments, "HEAD")
    return _git(root, "rev-parse", f"refs/tags/{previous}^{{commit}}")


def _initial_baseline(root: Path, initial: str | None, candidate: str) -> str:
    if initial is None:
        raise ValueError("first release requires an explicit initial baseline SHA")
    if _SHA_RE.fullmatch(initial) is None:
        raise ValueError("initial baseline must be a full lowercase 40-character SHA")
    baseline = _git(root, "rev-parse", "--verify", f"{initial}^{{commit}}")
    if baseline == candidate:
        raise ValueError("initial baseline must be a strict ancestor of candidate HEAD")
    try:
        _git(root, "merge-base", "--is-ancestor", baseline, candidate)
    except ValueError as error:
        raise ValueError(
            "initial baseline must be an ancestor of candidate HEAD"
        ) from error
    return baseline


def resolve_release_baseline(
    root: Path = ROOT,
    *,
    initial: str | None = None,
    tag: str | None = None,
) -> str:
    candidate = _git(root, "rev-parse", "HEAD^{commit}")
    if tag is not None:
        initial = _tag_baseline(root, tag, candidate, initial)
    previous = _previous_release(root, candidate)
    if previous is not None:
        if initial is not None:
            raise ValueError("initial baseline cannot override a previous release")
        return previous
    return _initial_baseline(root, initial, candidate)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--initial-baseline")
    parser.add_argument("--tag")
    args = parser.parse_args()
    try:
        print(resolve_release_baseline(initial=args.initial_baseline, tag=args.tag))
    except (OSError, ValueError) as error:
        print(f"Release baseline validation failed: {error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
