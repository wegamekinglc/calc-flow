"""Keep native Join allocation/RSS and bounded-edge observations with timings."""

from __future__ import annotations

import math
from pathlib import Path

from scripts.benchmark_suite.normalize import read_json

COUNT_FIELDS = (
    "allocation_peak_bytes",
    "allocation_total_bytes",
    "allocation_count",
    "rss_before_bytes",
    "rss_peak_bytes",
    "rss_first_emit_bytes",
    "chunks",
    "max_chunk_bytes",
    "queue_high_water_bytes",
    "blocked_sends",
)


def materialization_rows(path: Path) -> dict:
    report = read_json(path)
    if (
        report.get("schema") != "calc-flow.join-materialization.v1"
        or report.get("scope") != "operator-bounded-edge"
    ):
        raise ValueError("invalid Join materialization report contract")
    cases = report["cases"]
    names = [case["name"] for case in cases]
    if not names or len(names) != len(set(names)):
        raise ValueError("empty or duplicate Join materialization inventory")
    for case in cases:
        _validate_case(case)
    return {
        f"stream_join_materialization/{case['name']}": _case_row(case, report["scope"])
        for case in cases
    }


def _case_row(case: dict, scope: str) -> dict:
    return {
        "samples": [sample["seconds"] for sample in case["samples"]],
        "rows": case["config"]["incoming"] * case["config"]["fan"],
        "scope": scope,
        "metadata": {
            "config": case["config"],
            "oracle": case["oracle"],
            "observations": case["samples"],
        },
    }


def _validate_case(case: dict) -> None:
    rows = case["config"]["incoming"] * case["config"]["fan"]
    oracle = case["oracle"]
    if (
        oracle.get("validated_all_rows") is not True
        or oracle.get("output_rows") != rows
    ):
        raise ValueError("Join materialization row oracle failed")
    samples = case["samples"]
    if len(samples) < 20:
        raise ValueError("Join materialization samples are incomplete")
    if any(not _valid_sample(sample, rows) for sample in samples):
        raise ValueError("invalid Join materialization observation")


def _valid_sample(sample: dict, rows: int) -> bool:
    seconds = sample.get("seconds")
    return (
        type(seconds) in (int, float)
        and math.isfinite(seconds)
        and seconds > 0
        and sample.get("output_rows") == rows
        and _valid_diagnostics(sample)
    )


def _valid_diagnostics(sample: dict) -> bool:
    counts = [sample.get(field) for field in COUNT_FIELDS]
    blocked = sample.get("blocked_seconds")
    return (
        all(type(value) is int and value >= 0 for value in counts)
        and type(sample.get("rss_available")) is bool
        and type(blocked) in (int, float)
        and math.isfinite(blocked)
        and blocked >= 0
    )
