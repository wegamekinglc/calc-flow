"""Fail-closed aggregation across every catalog shard, including failed jobs."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

from scripts.benchmark_suite.catalog import get_shard, shard_cases, shards
from scripts.benchmark_suite.normalize import read_json
from scripts.benchmark_suite.report import (
    comparison,
    render_report,
    table,
    validate_metric,
    validate_shards,
)
from scripts.benchmark_suite.validation import validate_fragment, validate_shape


def case_failures(cases: list[dict]) -> list[str]:
    failures = []
    for case in cases:
        if case["status"] != "ok":
            failures.append(f"{case['id']}: {case.get('error', 'failed')}")
            continue
        try:
            if case.get("kind") == "metric":
                validate_metric(case)
                continue
            if comparison(case)["verdict"] == "regression":
                failures.append(f"{case['id']}: repeated regression exceeds +5%")
        except (KeyError, TypeError, ValueError) as error:
            failures.append(f"{case['id']}: {error}")
    return failures


def unavailable_cases(shard: dict, error: str, received: list[dict] = ()) -> list[dict]:
    expected = shard_cases(shard) if shard["family"] in ("engines", "warm") else []
    fallback = [
        {
            "id": case["id"],
            "scope": case.get("scope", "unavailable"),
            "rows": case.get("rows"),
        }
        for case in received
    ]
    descriptors = {case["id"]: case for case in [*fallback, *expected]}
    if not descriptors:
        descriptors = {shard["id"]: {"id": shard["id"]}}
    return [
        {
            **case,
            "scope": case.get("scope", "unavailable"),
            "status": "error",
            "error": error,
        }
        for case in descriptors.values()
    ]


def _read_fragment(
    path: Path, base: str, head: str
) -> tuple[dict | None, list[dict], list[str]]:
    try:
        report = read_json(path)
        shard = validate_shape(report)
    except (OSError, KeyError, TypeError, ValueError) as error:
        return None, [], [f"{path}: {error}"]
    report = {**report, "cases": fill_missing_cases(shard, report["cases"])}
    try:
        validate_fragment(report, base, head)
    except (KeyError, TypeError, ValueError) as error:
        message = f"{shard['id']}: {error}"
        return (
            report,
            unavailable_cases(shard, message, report["cases"]),
            [message, *report["errors"]],
        )
    return report, report["cases"], report["errors"]


def fill_missing_cases(shard: dict, cases: list[dict]) -> list[dict]:
    if shard["family"] not in ("engines", "warm"):
        return list(cases)
    received = {case["id"] for case in cases}
    return [
        *cases,
        *(
            {**case, "status": "error", "error": "case was not returned by the shard"}
            for case in shard_cases(shard)
            if case["id"] not in received
        ),
    ]


def collect(
    root: Path, base: str, head: str
) -> tuple[list[dict], list[str], list[dict]]:
    cases, errors, reports = [], [], []
    for path in sorted(root.glob("**/results.json")):
        report, rows, failures = _read_fragment(path, base, head)
        cases.extend(rows)
        errors.extend(failures)
        if report is not None:
            reports.append(report)
    expected = [shard["id"] for shard in shards()]
    received = [report["shard"]["id"] for report in reports]
    try:
        validate_shards(expected, received)
    except ValueError as error:
        errors.append(str(error))
    for name in sorted(set(expected) - set(received)):
        cases.extend(unavailable_cases(get_shard(name), "shard artifact is missing"))
    duplicates = [
        name
        for name, count in Counter(case["id"] for case in cases).items()
        if count > 1
    ]
    if duplicates:
        errors.append(f"duplicate benchmark cases: {duplicates}")
    errors.extend(case_failures(cases))
    return cases, errors, reports


def _provenance(reports: list[dict], base: str, head: str) -> str:
    rows = []
    for report in reports:
        rows.append(
            [
                report["shard"]["id"],
                _native_hash(report, "baseline"),
                _native_hash(report, "candidate"),
            ]
        )
    return "\n".join(
        [
            "## Provenance",
            "",
            f"Baseline: `{base}`. Candidate: `{head}`.",
            "",
            "Worker identities, dependency versions, raw samples and exact timing "
            "scopes are retained in the JSON artifacts. New engine SQL/DataFusion "
            "target partitions and Tokio/Polars worker settings are 32; BLAS helpers "
            "use one thread. Legacy fixtures retain their own query configuration. "
            "Each shard compares versions on its own runner, never "
            "across hosts.",
            "",
            table(["Shard", "Base native SHA256", "Head native SHA256"], rows),
            "",
        ]
    )


def _native_hash(report: dict, side: str) -> str:
    releases = report.get("releases")
    release = releases.get(side) if isinstance(releases, dict) else None
    return (
        release.get("native_sha256", "missing")
        if isinstance(release, dict)
        else "missing"
    )


def summarize(args: argparse.Namespace) -> int:
    cases, errors, reports = collect(args.input, args.expected_base, args.expected_head)
    report = (
        render_report(cases, errors)
        + "\n"
        + _provenance(reports, args.expected_base, args.expected_head)
    )
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "summary.md").write_text(report, encoding="utf-8")
    (args.output / "summary.json").write_text(
        json.dumps({"cases": cases, "errors": errors}, indent=2, allow_nan=False)
        + "\n",
        encoding="utf-8",
    )
    print(report, end="")
    if args.github_summary:
        if len(report.encode()) > 1_000_000:
            raise ValueError(
                "full summary exceeds the GitHub step limit; "
                "retained as artifact, never silently truncated"
            )
        with args.github_summary.open("a", encoding="utf-8") as output:
            output.write(report)
    return int(bool(errors))
