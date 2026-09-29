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
    if (
        report.get("schema") != "calc-flow.asof-e2e.v1"
        or report.get("scope") != "operator-admission-settlement"
    ):
        raise ValueError("invalid ASOF e2e benchmark contract")
    cases = report.get("cases", [])
    if len(cases) != len(CASES) or {case.get("name") for case in cases} != CASES:
        raise ValueError("incomplete ASOF e2e inventory")
    for case in cases:
        if case.get("rows") != ROWS or len(case.get("samples", [])) < 20:
            raise ValueError("invalid ASOF e2e workload or sample count")
        for sample in (case.get("oracle"), *case["samples"]):
            if not _valid_sample(sample, case["name"]):
                raise ValueError("invalid ASOF e2e observation")
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


def _valid_sample(sample: object, case_name: str) -> bool:
    if not isinstance(sample, dict):
        return False
    seconds = sample.get("seconds")
    return (
        type(seconds) in (int, float)
        and math.isfinite(seconds)
        and seconds > 0
        and sample.get("output_rows") == ROWS
        and sample.get("validated_all_rows") is True
        and type(sample.get("evicted_right_rows")) is int
        and type(sample.get("retained_right_rows")) is int
        and sample["evicted_right_rows"] >= 0
        and sample["retained_right_rows"] >= 0
        and sample["evicted_right_rows"] + sample["retained_right_rows"] == ROWS
        and (case_name != "eviction_ticks" or sample["evicted_right_rows"] > 0)
        and all(
            type(sample.get(field)) is int and sample[field] >= 0
            for field in (
                "allocation_total_bytes",
                "allocation_peak_bytes",
                "allocation_count",
            )
        )
    )
