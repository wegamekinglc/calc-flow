"""Validate one isolated stream lifecycle pytest-benchmark artifact."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from pathlib import Path
from typing import Any

SCENARIO = "symbolic_stream_window_checkpoint"
PHASE_FIELDS = (
    "startup_duration_seconds",
    "steady_processing_duration_seconds",
    "checkpoint_duration_seconds",
    "cancel_duration_seconds",
    "recovery_duration_seconds",
    "shutdown_duration_seconds",
)
IDENTITY_FIELDS = ("machine", "dependency", "workload")


def _fingerprint(value: dict[str, object]) -> str:
    encoded = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    ).encode()
    return hashlib.sha256(encoded).hexdigest()


def _object(value: object, field: str) -> dict[str, Any]:
    if not isinstance(value, dict) or any(not isinstance(key, str) for key in value):
        raise ValueError(f"{field} must be an object")
    return value


def _positive_number(value: object, field: str) -> float:
    if (
        isinstance(value, bool)
        or not isinstance(value, int | float)
        or not math.isfinite(value)
        or value <= 0
    ):
        raise ValueError(f"{field} must be a positive finite number")
    return float(value)


def _non_negative_number(value: object, field: str) -> float:
    if (
        isinstance(value, bool)
        or not isinstance(value, int | float)
        or not math.isfinite(value)
        or value < 0
    ):
        raise ValueError(f"{field} must be a non-negative finite number")
    return float(value)


def _non_negative_integer(value: object, field: str) -> int:
    if type(value) is not int or value < 0:
        raise ValueError(f"{field} must be a non-negative integer")
    return value


def _matching_benchmark(document: dict[str, Any]) -> dict[str, Any]:
    benchmarks = document.get("benchmarks")
    if not isinstance(benchmarks, list):
        raise ValueError("report benchmarks must be an array")
    matching = []
    for entry in benchmarks:
        benchmark = _object(entry, "benchmark")
        extra = _object(benchmark.get("extra_info"), "benchmark extra_info")
        if extra.get("scenario") == SCENARIO:
            matching.append(benchmark)
    if len(matching) != 1:
        raise ValueError(f"report must contain exactly one {SCENARIO} case")
    return matching[0]


def _validate_stats(benchmark: dict[str, Any], minimum_rounds: int) -> None:
    stats = _object(benchmark.get("stats"), "benchmark stats")
    _positive_number(stats.get("mean"), "benchmark mean")
    _non_negative_number(stats.get("stddev"), "benchmark stddev")
    rounds = _non_negative_integer(stats.get("rounds"), "benchmark rounds")
    if rounds < minimum_rounds:
        raise ValueError(
            f"stream lifecycle evidence requires at least {minimum_rounds} rounds"
        )


def _validate_checkpoint_bytes(extra: dict[str, Any]) -> None:
    checkpoint_bytes = _non_negative_integer(
        extra.get("checkpoint_bytes"), "checkpoint_bytes"
    )
    if checkpoint_bytes == 0:
        raise ValueError("checkpoint_bytes must be positive")
    checkpoint_bytes_p50 = _non_negative_integer(
        extra.get("checkpoint_bytes_p50"), "checkpoint_bytes_p50"
    )
    checkpoint_bytes_p95 = _non_negative_integer(
        extra.get("checkpoint_bytes_p95"), "checkpoint_bytes_p95"
    )
    if (
        checkpoint_bytes != checkpoint_bytes_p50
        or checkpoint_bytes_p95 < checkpoint_bytes_p50
    ):
        raise ValueError("checkpoint byte quantiles are inconsistent")


def _validate_duration_quantiles(extra: dict[str, Any]) -> None:
    for phase in ("checkpoint", "recovery"):
        p50 = _positive_number(
            extra.get(f"{phase}_duration_p50_seconds"),
            f"{phase}_duration_p50_seconds",
        )
        p95 = _positive_number(
            extra.get(f"{phase}_duration_p95_seconds"),
            f"{phase}_duration_p95_seconds",
        )
        if p95 < p50:
            raise ValueError(f"{phase} p95 duration must be at least p50")


def _validate_lifecycle(extra: dict[str, Any], minimum_rounds: int) -> None:
    phase_total = sum(
        _positive_number(extra.get(field), field) for field in PHASE_FIELDS
    )
    lifecycle_total = _positive_number(
        extra.get("total_duration_seconds"), "total_duration_seconds"
    )
    if phase_total > lifecycle_total * 1.01:
        raise ValueError("lifecycle phase durations exceed the measured total")
    if _non_negative_integer(
        extra.get("recovery_resumed_batches"), "recovery_resumed_batches"
    ):
        raise ValueError("recovery evidence contains replayed batches")
    if (
        _non_negative_integer(extra.get("checkpoint_batches"), "checkpoint_batches")
        == 0
    ):
        raise ValueError("checkpoint_batches must be positive")
    _validate_checkpoint_bytes(extra)
    diagnostic_samples = _non_negative_integer(
        extra.get("diagnostic_samples"), "diagnostic_samples"
    )
    if diagnostic_samples < minimum_rounds:
        raise ValueError(
            f"stream lifecycle diagnostics require at least {minimum_rounds} samples"
        )
    _validate_duration_quantiles(extra)
    for field in ("rss_before_bytes", "rss_after_bytes", "peak_rss_bytes"):
        _positive_number(extra.get(field), field)


def _validate_identities(extra: dict[str, Any]) -> None:
    for prefix in IDENTITY_FIELDS:
        identity_field = f"{prefix}_identity"
        fingerprint_field = f"{prefix}_fingerprint"
        identity = _object(extra.get(identity_field), identity_field)
        fingerprint = extra.get(fingerprint_field)
        if not isinstance(fingerprint, str) or fingerprint != _fingerprint(identity):
            raise ValueError(f"{fingerprint_field} does not match {identity_field}")


def validate_report(path: Path, *, minimum_rounds: int = 20) -> dict[str, Any]:
    """Return validated lifecycle metadata or raise a fail-closed error."""
    if minimum_rounds < 2:
        raise ValueError("minimum rounds must be at least two")
    document = _object(json.loads(path.read_text(encoding="utf-8")), "report")
    benchmark = _matching_benchmark(document)
    _validate_stats(benchmark, minimum_rounds)
    extra = _object(benchmark.get("extra_info"), "benchmark extra_info")
    _validate_lifecycle(extra, minimum_rounds)
    _validate_identities(extra)
    return extra


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("report", type=Path)
    parser.add_argument("--minimum-rounds", type=int, default=20)
    options = parser.parse_args()
    try:
        evidence = validate_report(
            options.report, minimum_rounds=options.minimum_rounds
        )
    except (OSError, ValueError, json.JSONDecodeError) as error:
        print(f"Invalid stream lifecycle evidence: {error}", file=sys.stderr)
        return 1
    print(
        "Validated isolated stream lifecycle evidence: "
        f"checkpoint_bytes={evidence['checkpoint_bytes']} "
        f"rounds>={options.minimum_rounds}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
