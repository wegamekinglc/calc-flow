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
MAX_CHUNK_ROWS = 10_000  # The operator context's default output edge budget.
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
ADMISSION_PHASES = (
    "right_fixture_seconds_untimed",
    "right_admission_seconds_untimed",
    "left_fixture_seconds_untimed",
    "left_admission_seconds_untimed",
)


def asof_rows(path: Path) -> dict:
    report = read_json(path)
    if (
        report.get("schema") != "calc-flow.asof-finalization.v1"
        or report.get("scope") != "operator-watermark-settlement"
    ):
        raise ValueError("invalid ASOF benchmark contract")
    cases = report["cases"]
    retention = report.get("retention", "tolerance-window")
    if retention not in ("tolerance-window", "latest-per-key"):
        raise ValueError("invalid ASOF retention contract")
    if sorted(case["name"] for case in cases) != sorted(CASES):
        raise ValueError("incomplete or duplicate ASOF inventory")
    for case in cases:
        _validate_case(case, retention)
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


def _validate_case(case: dict, retention: str) -> None:
    config = CASES[case["name"]]
    if case["config"] != config or case["oracle"].get("validated_all_rows") is not True:
        raise ValueError("ASOF workload or row oracle mismatch")
    if len(case["samples"]) < 20:
        raise ValueError("incomplete ASOF observations")
    for sample in (case["oracle"], *case["samples"]):
        if not _valid_sample(sample, config, retention):
            raise ValueError("invalid ASOF observation")


def _valid_sample(sample: dict, config: dict, retention: str) -> bool:
    return all(
        (
            sample.get("config") == config,
            sample.get("output_rows") == config["pending"],
            _valid_counts(sample),
            _valid_times(sample, config),
            _valid_chunks(sample, config),
            _valid_status(sample, config, retention),
        )
    )


def _valid_counts(sample: dict) -> bool:
    counts = [sample.get(field) for field in COUNTS]
    return (
        all(type(value) is int and value >= 0 for value in counts)
        and type(sample.get("rss_available")) is bool
    )


def _nonnegative(value: object) -> bool:
    try:
        return type(value) in (int, float) and math.isfinite(value) and value >= 0
    except OverflowError:
        return False


def _valid_times(sample: dict, config: dict) -> bool:
    fields = ("seconds", "admission_seconds_untimed", "capture_seconds_untimed")
    timings = all(_nonnegative(sample.get(field)) for field in fields)
    restore = sample.get("restore_seconds_untimed")
    restored = _nonnegative(restore) if config["restored"] else restore is None
    return (
        timings
        and sample["seconds"] > 0
        and restored
        and _valid_admission_phases(sample)
    )


def _valid_admission_phases(sample: dict) -> bool:
    if not any(field in sample for field in ADMISSION_PHASES):
        return True
    values = [sample.get(field) for field in ADMISSION_PHASES]
    if not all(_nonnegative(value) for value in values):
        return False
    elapsed = sum(float(value) for value in values)
    total = sample["admission_seconds_untimed"]
    return math.isfinite(elapsed) and (
        elapsed <= total or elapsed - total <= 8 * math.ulp(total)
    )


def _valid_chunks(sample: dict, config: dict) -> bool:
    chunks = sample.get("chunks", [])
    if not isinstance(chunks, list) or not chunks:
        return False
    valid = all(
        isinstance(chunk, list)
        and len(chunk) == 2
        and type(chunk[0]) is int
        and 0 < chunk[0] <= MAX_CHUNK_ROWS
        and _nonnegative(chunk[1])
        for chunk in chunks
    )
    return valid and sum(chunk[0] for chunk in chunks) == config["pending"]


def _valid_status(sample: dict, config: dict, retention: str) -> bool:
    status = sample.get("after_status", {})
    retained = 32 if retention == "latest-per-key" else config["retained"]
    return (
        status.get("pending_left_rows") == 0
        and status.get("matched_rows") == config["pending"]
        and status.get("retained_right_rows") == retained
        and (
            retention != "latest-per-key"
            or status.get("evicted_right_rows") == config["retained"] - retained
        )
    )
