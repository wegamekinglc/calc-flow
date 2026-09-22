"""Validate the maintained ASOF inventory and preserve every diagnostic sample."""

from __future__ import annotations

import math
from pathlib import Path

from scripts.benchmark_suite.normalize import read_json

CASES = {
    name: dict(pending=pending, retained=retained, skew=skew, restored=restored)
    for name, pending, retained, skew, restored in (
        ("balanced_512", 512, 512, False, False),
        ("balanced_2048", 2048, 2048, False, False),
        ("balanced_8192", 8192, 8192, False, False),
        ("fixed128_right512", 128, 512, False, False),
        ("fixed128_right2048", 128, 2048, False, False),
        ("fixed128_right8192", 128, 8192, False, False),
        ("skew_8192", 8192, 8192, True, False),
        ("restored_skew_8192", 8192, 8192, True, True),
    )
}
COUNTS = (
    "allocation_peak_bytes",
    "allocation_total_bytes",
    "allocation_count",
    "rss_before_bytes",
    "rss_peak_bytes",
    "checkpoint_before_bytes",
    "checkpoint_after_bytes",
    "max_chunk_bytes",
)


def asof_rows(path: Path) -> dict:
    report = read_json(path)
    if (
        report.get("schema") != "calc-flow.asof-finalization.v1"
        or report.get("scope") != "operator-watermark-settlement"
    ):
        raise ValueError("invalid ASOF benchmark contract")
    cases = report["cases"]
    if sorted(case["name"] for case in cases) != sorted(CASES):
        raise ValueError("incomplete or duplicate ASOF inventory")
    for case in cases:
        _validate_case(case)
    return {
        f"stream_asof_perf/{case['name']}": {
            "samples": [sample["seconds"] for sample in case["samples"]],
            "rows": case["config"]["pending"],
            "scope": report["scope"],
            "metadata": {
                "config": case["config"],
                "oracle": case["oracle"],
                "observations": case["samples"],
            },
        }
        for case in cases
    }


def _validate_case(case: dict) -> None:
    config = CASES[case["name"]]
    if case["config"] != config or case["oracle"].get("validated_all_rows") is not True:
        raise ValueError("ASOF workload or row oracle mismatch")
    if len(case["samples"]) < 20:
        raise ValueError("incomplete ASOF observations")
    for sample in (case["oracle"], *case["samples"]):
        if not _valid_sample(sample, config):
            raise ValueError("invalid ASOF observation")


def _valid_sample(sample: dict, config: dict) -> bool:
    return all(
        (
            sample.get("config") == config,
            sample.get("output_rows") == config["pending"],
            _valid_counts(sample),
            _valid_times(sample, config),
            _valid_chunks(sample, config),
            _valid_status(sample, config),
        )
    )


def _valid_counts(sample: dict) -> bool:
    counts = [sample.get(field) for field in COUNTS]
    return (
        all(type(value) is int and value >= 0 for value in counts)
        and type(sample.get("rss_available")) is bool
    )


def _nonnegative(value: object) -> bool:
    return type(value) in (int, float) and math.isfinite(value) and value >= 0


def _valid_times(sample: dict, config: dict) -> bool:
    fields = ("seconds", "admission_seconds_untimed", "capture_seconds_untimed")
    timings = all(_nonnegative(sample.get(field)) for field in fields)
    restore = sample.get("restore_seconds_untimed")
    restored = _nonnegative(restore) if config["restored"] else restore is None
    return timings and sample["seconds"] > 0 and restored


def _valid_chunks(sample: dict, config: dict) -> bool:
    chunks = sample.get("chunks", [])
    if len(chunks) != config["pending"] // 128:
        return False
    return all(
        len(chunk) == 2 and chunk[0] == 128 and _nonnegative(chunk[1])
        for chunk in chunks
    )


def _valid_status(sample: dict, config: dict) -> bool:
    status = sample.get("after_status", {})
    return status.get("pending_left_rows") == 0 and all(
        status.get(field) == config[dimension]
        for field, dimension in (
            ("matched_rows", "pending"),
            ("retained_right_rows", "retained"),
        )
    )
