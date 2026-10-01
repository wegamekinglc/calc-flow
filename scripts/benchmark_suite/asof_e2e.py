"""Validate admission-through-settlement ASOF benchmark observations."""

from __future__ import annotations

import math
from pathlib import Path

from scripts.benchmark_suite.normalize import read_json

CASES = frozenset(
    {
        "admit_settle_100k",
        "eviction_ticks",
        "out_of_order_within_watermark",
        "composite_key",
    }
)
ROWS = 100_000


def asof_e2e_rows(path: Path) -> dict:
    """Keep timing and allocation samples for every validated ASOF workload."""

    report = read_json(path)
    cases = _validated_cases(report)
    return {
        f"stream_asof_e2e/{case['name']}": {
            "samples": [sample["seconds"] for sample in case["samples"]],
            "rows": ROWS,
            "scope": report["scope"],
            "metadata": {
                "oracle": case["oracle"],
                "observations": case["samples"],
            },
        }
        for case in cases
    }


def _validated_cases(report: dict) -> list[dict]:
    if (
        report.get("schema") != "calc-flow.asof-e2e.v1"
        or report.get("scope") != "operator-admission-settlement"
    ):
        raise ValueError("invalid ASOF e2e benchmark contract")
    cases = report.get("cases", [])
    if len(cases) != len(CASES) or {case.get("name") for case in cases} != CASES:
        raise ValueError("incomplete ASOF e2e inventory")
    for case in cases:
        _validate_case(case)
    return cases


def _validate_case(case: dict) -> None:
    if case.get("rows") != ROWS or len(case.get("samples", [])) < 20:
        raise ValueError("invalid ASOF e2e workload or sample count")
    for sample in (case.get("oracle"), *case["samples"]):
        if not _valid_sample(sample, case["name"]):
            raise ValueError("invalid ASOF e2e observation")


def _valid_sample(sample: object, case_name: str) -> bool:
    if not isinstance(sample, dict):
        return False
    seconds = sample.get("seconds")
    if type(seconds) not in (int, float) or not math.isfinite(seconds) or seconds <= 0:
        return False
    if (
        sample.get("output_rows") != ROWS
        or sample.get("validated_all_rows") is not True
    ):
        return False
    return _valid_right_rows(sample, case_name) and _valid_allocations(sample)


def _valid_right_rows(sample: dict, case_name: str) -> bool:
    evicted = sample.get("evicted_right_rows")
    retained = sample.get("retained_right_rows")
    if type(evicted) is not int or type(retained) is not int:
        return False
    if evicted < 0 or retained < 0 or evicted + retained != ROWS:
        return False
    return case_name != "eviction_ticks" or evicted > 0


def _valid_allocations(sample: dict) -> bool:
    return all(
        type(sample.get(field)) is int and sample[field] >= 0
        for field in (
            "allocation_total_bytes",
            "allocation_peak_bytes",
            "allocation_count",
        )
    )
