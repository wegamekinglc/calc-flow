"""Record the exact source and run used by the DAL-296 coverage recovery."""

from __future__ import annotations

import argparse
import json
import os
from collections.abc import Mapping
from pathlib import Path

try:
    from scripts.toolkit import (
        FULL_SHA,
        command_output,
        require_executable,
        sha256_file,
        write_json,
    )
except ImportError:
    from toolkit import (
        FULL_SHA,
        command_output,
        require_executable,
        sha256_file,
        write_json,
    )

SOURCE_SHA = "2fcc36fd224dd1c8a9f6396bddd5fbe0b60b0987"
EXECUTION_REF = "refs/heads/fix/dal-296-coverage-recovery-execute"
REPORTS = {
    "rust": "rust-lcov.info",
    "python": "coverage.xml",
    "studio": "web-ui/backend/coverage.xml",
}


def _identity(environment: Mapping[str, str]) -> dict:
    expected = {
        "GITHUB_REPOSITORY": "wegamekinglc/calc-flow",
        "GITHUB_RUN_ATTEMPT": "1",
        "GITHUB_EVENT_NAME": "push",
        "GITHUB_REF": EXECUTION_REF,
    }
    if any(environment.get(key) != value for key, value in expected.items()):
        raise ValueError("recovery requires its dedicated push and original attempt")
    sha = environment["GITHUB_SHA"]
    number = environment["GITHUB_RUN_ID"]
    if not FULL_SHA.fullmatch(sha) or not number.isdecimal() or int(number) <= 0:
        raise ValueError("recovery workflow SHA or run ID is invalid")
    return {
        "repository": expected["GITHUB_REPOSITORY"],
        "source_sha": SOURCE_SHA,
        "workflow_sha": sha,
        "workflow_path": ".github/workflows/coverage-recovery.yml",
        "run_id": int(number),
        "run_attempt": 1,
        "build_number": number,
    }


def record(source: Path, control: Path, environment: Mapping[str, str]) -> dict:
    identity = _identity(environment)
    git = require_executable("git")
    head = command_output([git, "rev-parse", "HEAD"], cwd=source)
    tree = command_output([git, "rev-parse", "HEAD^{tree}"], cwd=source)
    dirty = command_output(
        [git, "status", "--porcelain", "--untracked-files=no"], cwd=source
    )
    if head != SOURCE_SHA or not FULL_SHA.fullmatch(tree) or dirty:
        raise ValueError("measured source is modified or is not the pinned SHA")
    control_head = command_output([git, "rev-parse", "HEAD"], cwd=control)
    control_dirty = command_output(
        [git, "status", "--porcelain", "--untracked-files=no"], cwd=control
    )
    if control_head != identity["workflow_sha"] or control_dirty:
        raise ValueError("workflow checkout is modified or differs from the run")
    return {**identity, "source_tree": tree}


def seal(
    source: Path, control: Path, environment: Mapping[str, str], flags: list[str]
) -> dict:
    identity = record(source, control, environment)
    if len(set(flags)) != len(flags) or not set(flags) <= REPORTS.keys():
        raise ValueError("unexpected or duplicate coverage flags")
    reports = {}
    for flag in flags:
        path = source / REPORTS[flag]
        if not path.is_file() or path.stat().st_size == 0:
            raise ValueError(f"missing or empty {flag} coverage report")
        reports[flag] = {"path": REPORTS[flag], "sha256": sha256_file(path)}
    return {**identity, "reports": reports}


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--control", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--flag", choices=REPORTS, action="append", default=[])
    mode.add_argument("--manifest", type=Path, action="append")
    args = parser.parse_args(argv)
    if args.manifest:
        records = [
            json.loads(path.read_text(encoding="utf-8")) for path in args.manifest
        ]
        result = validate(records, args.source, args.control, os.environ)
    else:
        result = seal(args.source, args.control, os.environ, args.flag)
    write_json(args.output, result, exclusive=True)


def validate(
    records: list[dict], source: Path, control: Path, environment: Mapping[str, str]
) -> dict:
    identity = record(source, control, environment)
    reports = {}
    for item in records:
        if {key: value for key, value in item.items() if key != "reports"} != identity:
            raise ValueError("coverage producers differ in source/workflow/run/attempt")
        for flag, report in item["reports"].items():
            if flag not in REPORTS or flag in reports:
                raise ValueError("unexpected or duplicate coverage flag")
            path = source / REPORTS[flag]
            if report != {"path": REPORTS[flag], "sha256": sha256_file(path)}:
                raise ValueError(f"{flag} coverage report digest differs")
            reports[flag] = report
    if reports.keys() != REPORTS.keys():
        raise ValueError("recovery requires all three flags in this run")
    return {**identity, "reports": reports}


if __name__ == "__main__":
    main()
