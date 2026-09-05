"""Fail-closed validator for SQL versus raw DataFusion benchmark evidence."""

from __future__ import annotations

import argparse
import json
import math
import re
import statistics
import sys
from pathlib import Path
from typing import Any

GIT_SHA = re.compile(r"[0-9a-f]{40}")
FINGERPRINT = re.compile(r"[0-9a-f]{64}")
PROFILES = {
    "serial-control",
    "matched-adaptive",
    "p32-saturation",
    "matrix",
    "attribution",
}
PHASES = {
    "runtime_acquire",
    "session_state_create",
    "input_adapter",
    "table_register",
    "sql_parse",
    "logical_optimize",
    "physical_plan",
    "execution_to_first_batch",
    "execution_remaining",
    "collect_or_coalesce",
    "output_arrow_wrap",
    "audit",
    "metrics_traversal",
    "physical_plan_string",
    "batch_envelope",
    "run_result",
    "run_session_envelope",
}
ENGINE_FIELDS = {
    "parallelism_mode",
    "configured_partitions",
    "requested_partitions",
    "effective_partitions",
    "available_parallelism",
    "max_partitions",
    "min_rows_per_partition",
    "small_rows_threshold",
    "parallelism_decision_reused",
    "decision_input_rows",
    "decision_active_entities",
    "decision_active_entities_source",
    "partition_limit_reason",
    "batch_size",
    "input_logical_partitions",
    "input_batch_rows",
    "normalized_plan_hash",
    "bounded_window_agg_count",
    "samples_ms",
    "median_ms",
    "p25_ms",
    "p75_ms",
    "mad_ms",
    "cv",
    "cpu_time_ms",
    "peak_rss_bytes",
    "spill_bytes",
    "empty_partitions",
    "partition_rows",
    "partition_skew",
    "window_compute_ms",
    "repartition_sort_compute_ms",
    "window_operator_count",
    "repartition_operator_count",
    "sort_operator_count",
    "coalesce_operator_count",
    "phase_medians_ms",
    "phase_samples_ms",
}
CORRECTNESS_FIELDS = {
    "schema",
    "rows",
    "keys",
    "order",
    "null_nan_mask",
    "values",
    "rtol",
    "atol",
}
CASE_FIELDS = {
    "name",
    "rows",
    "active_entities",
    "window",
    "warmups",
    "rolling_rewrite_enabled",
    "sample_order",
    "calc_flow",
    "raw_datafusion",
    "paired_ratios",
    "paired_ratio_median",
    "paired_ratio_ci_low",
    "paired_ratio_ci_high",
    "correctness",
    "comparability",
    "speedup_conclusion",
}
ENVIRONMENT_FIELDS = {
    "machine_fingerprint",
    "dependency_fingerprint",
    "workload_fingerprint",
    "datafusion_version",
    "arrow_version",
    "build_profile",
    "allocator",
    "os",
    "arch",
    "cpu_model",
    "available_parallelism",
    "rust_version",
    "git_dirty",
}


def _exact_fields(value: object, expected: set[str], label: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ValueError(f"{label} must be an object")
    actual = set(value)
    if actual != expected:
        missing = sorted(expected - actual)
        extra = sorted(actual - expected)
        raise ValueError(f"{label} fields mismatch; missing={missing}, extra={extra}")
    return value


def _positive_int(value: object, label: str) -> int:
    if type(value) is not int or value <= 0:
        raise ValueError(f"{label} must be a positive integer")
    return value


def _nonnegative_int(value: object, label: str) -> int:
    if type(value) is not int or value < 0:
        raise ValueError(f"{label} must be a non-negative integer")
    return value


def _finite(value: object, label: str, *, positive: bool = False) -> float:
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise ValueError(f"{label} must be finite")
    result = float(value)
    if not math.isfinite(result) or (positive and result <= 0):
        raise ValueError(f"{label} must be {'positive and ' if positive else ''}finite")
    return result


def _median(values: list[float]) -> float:
    return statistics.median(values)


def _cv(values: list[float]) -> float:
    mean = statistics.fmean(values)
    return statistics.pstdev(values) / mean


def _verify_engine(
    raw: object,
    *,
    label: str,
    rows: int,
    minimum_samples: int,
    require_stable: bool,
) -> tuple[dict[str, Any], list[float]]:
    # The evidence contract deliberately checks every field in one fail-closed
    # engine boundary and reports the precise malformed path.
    # #lizard forgives
    engine = _exact_fields(raw, ENGINE_FIELDS, label)
    for field in (
        "configured_partitions",
        "requested_partitions",
        "effective_partitions",
        "available_parallelism",
        "max_partitions",
        "min_rows_per_partition",
        "small_rows_threshold",
        "batch_size",
        "input_logical_partitions",
        "peak_rss_bytes",
    ):
        _positive_int(engine[field], f"{label}.{field}")
    if engine["parallelism_mode"] not in {"fixed", "auto"}:
        raise ValueError(f"{label}.parallelism_mode must be fixed or auto")
    if type(engine["parallelism_decision_reused"]) is not bool:
        raise ValueError(f"{label}.parallelism_decision_reused must be a bool")
    _nonnegative_int(engine["decision_input_rows"], f"{label}.decision_input_rows")
    decision_entities = engine["decision_active_entities"]
    if decision_entities is not None:
        _positive_int(decision_entities, f"{label}.decision_active_entities")
    if (
        not isinstance(engine["decision_active_entities_source"], str)
        or not engine["decision_active_entities_source"]
    ):
        raise ValueError(f"{label}.decision_active_entities_source must be non-empty")
    _nonnegative_int(engine["spill_bytes"], f"{label}.spill_bytes")
    _nonnegative_int(engine["empty_partitions"], f"{label}.empty_partitions")
    plan_hash = engine["normalized_plan_hash"]
    if not isinstance(plan_hash, str) or FINGERPRINT.fullmatch(plan_hash) is None:
        raise ValueError(f"{label}.normalized_plan_hash must be a SHA-256 digest")
    raw_samples = engine["samples_ms"]
    if not isinstance(raw_samples, list) or len(raw_samples) < minimum_samples:
        raise ValueError(
            f"{label}.samples_ms requires at least {minimum_samples} samples"
        )
    samples = [
        _finite(value, f"{label}.samples_ms[{index}]", positive=True)
        for index, value in enumerate(raw_samples)
    ]
    summary = {
        "median_ms": _median(samples),
        "p25_ms": _percentile(samples, 0.25),
        "p75_ms": _percentile(samples, 0.75),
        "mad_ms": _median([abs(value - _median(samples)) for value in samples]),
        "cv": _cv(samples),
    }
    for field, expected in summary.items():
        reported = _finite(engine[field], f"{label}.{field}")
        if not math.isclose(reported, expected, rel_tol=1e-12, abs_tol=1e-12):
            raise ValueError(f"{label}.{field} is inconsistent")
    if require_stable and summary["cv"] > 0.10:
        raise ValueError(f"{label} CV exceeds 10%")
    _finite(engine["cpu_time_ms"], f"{label}.cpu_time_ms")
    if engine["cpu_time_ms"] < 0:
        raise ValueError(f"{label}.cpu_time_ms must be non-negative")
    if (
        not isinstance(engine["partition_limit_reason"], str)
        or not engine["partition_limit_reason"]
    ):
        raise ValueError(f"{label}.partition_limit_reason must be non-empty")
    input_batch_rows = engine["input_batch_rows"]
    if not isinstance(input_batch_rows, list) or any(
        type(value) is not int or value <= 0 for value in input_batch_rows
    ):
        raise ValueError(f"{label}.input_batch_rows must contain positive integers")
    if sum(input_batch_rows) != rows:
        raise ValueError(f"{label}.input_batch_rows must sum to rows")
    _nonnegative_int(
        engine["bounded_window_agg_count"], f"{label}.bounded_window_agg_count"
    )
    for field in (
        "window_operator_count",
        "repartition_operator_count",
        "sort_operator_count",
        "coalesce_operator_count",
    ):
        _nonnegative_int(engine[field], f"{label}.{field}")
    for field in ("window_compute_ms", "repartition_sort_compute_ms"):
        if _finite(engine[field], f"{label}.{field}") < 0:
            raise ValueError(f"{label}.{field} must be non-negative")
    partition_rows = engine["partition_rows"]
    if not isinstance(partition_rows, list) or any(
        type(value) is not int or value < 0 for value in partition_rows
    ):
        raise ValueError(f"{label}.partition_rows must contain non-negative integers")
    if len(partition_rows) != engine["effective_partitions"]:
        raise ValueError(f"{label}.partition_rows must cover every effective partition")
    if sum(partition_rows) != rows:
        raise ValueError(f"{label}.partition_rows must sum to rows")
    if sum(value == 0 for value in partition_rows) != engine["empty_partitions"]:
        raise ValueError(f"{label}.empty_partitions is inconsistent")
    average_rows = rows / len(partition_rows)
    expected_skew = max(partition_rows) / average_rows
    partition_skew = _finite(engine["partition_skew"], f"{label}.partition_skew")
    if not math.isclose(partition_skew, expected_skew, rel_tol=1e-12, abs_tol=1e-12):
        raise ValueError(f"{label}.partition_skew is inconsistent")
    phases = _exact_fields(
        engine["phase_medians_ms"], PHASES, f"{label}.phase_medians_ms"
    )
    for phase, value in phases.items():
        if _finite(value, f"{label}.phase_medians_ms.{phase}") < 0:
            raise ValueError(f"{label}.phase_medians_ms.{phase} must be non-negative")
    phase_samples = _exact_fields(
        engine["phase_samples_ms"], PHASES, f"{label}.phase_samples_ms"
    )
    for phase, raw_values in phase_samples.items():
        if not isinstance(raw_values, list) or len(raw_values) != len(samples):
            raise ValueError(
                f"{label}.phase_samples_ms.{phase} must cover every sample"
            )
        values = [
            _finite(value, f"{label}.phase_samples_ms.{phase}[{index}]")
            for index, value in enumerate(raw_values)
        ]
        if any(value < 0 for value in values):
            raise ValueError(f"{label}.phase_samples_ms.{phase} must be non-negative")
        if not math.isclose(
            float(phases[phase]), _median(values), rel_tol=1e-12, abs_tol=1e-12
        ):
            raise ValueError(f"{label}.phase_medians_ms.{phase} is inconsistent")
    return engine, samples


def _percentile(values: list[float], fraction: float) -> float:
    ordered = sorted(values)
    position = (len(ordered) - 1) * fraction
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    weight = position - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def _verify_correctness(raw: object, label: str) -> None:
    correctness = _exact_fields(raw, CORRECTNESS_FIELDS, f"{label}.correctness")
    for field in ("schema", "rows", "keys", "order", "null_nan_mask", "values"):
        if correctness[field] is not True:
            raise ValueError(f"{label} correctness failed for {field}")
    for field in ("rtol", "atol"):
        tolerance = _finite(correctness[field], f"{label}.correctness.{field}")
        if tolerance < 0 or tolerance > 1e-10:
            raise ValueError(f"{label}.correctness.{field} exceeds 1e-10")


def _actual_mismatches(calc_flow: dict[str, Any], raw: dict[str, Any]) -> list[str]:
    pairs = {
        "parallelism_mode": (calc_flow["parallelism_mode"], raw["parallelism_mode"]),
        "configured_partitions": (
            calc_flow["configured_partitions"],
            raw["configured_partitions"],
        ),
        "effective_partitions": (
            calc_flow["effective_partitions"],
            raw["effective_partitions"],
        ),
        "available_parallelism": (
            calc_flow["available_parallelism"],
            raw["available_parallelism"],
        ),
        "max_partitions": (calc_flow["max_partitions"], raw["max_partitions"]),
        "min_rows_per_partition": (
            calc_flow["min_rows_per_partition"],
            raw["min_rows_per_partition"],
        ),
        "small_rows_threshold": (
            calc_flow["small_rows_threshold"],
            raw["small_rows_threshold"],
        ),
        "decision_input_rows": (
            calc_flow["decision_input_rows"],
            raw["decision_input_rows"],
        ),
        "decision_active_entities": (
            calc_flow["decision_active_entities"],
            raw["decision_active_entities"],
        ),
        "decision_active_entities_source": (
            calc_flow["decision_active_entities_source"],
            raw["decision_active_entities_source"],
        ),
        "parallelism_decision_reused": (
            calc_flow["parallelism_decision_reused"],
            raw["parallelism_decision_reused"],
        ),
        "batch_size": (calc_flow["batch_size"], raw["batch_size"]),
        "input_logical_partitions": (
            calc_flow["input_logical_partitions"],
            raw["input_logical_partitions"],
        ),
        "input_batch_rows": (
            calc_flow["input_batch_rows"],
            raw["input_batch_rows"],
        ),
        "normalized_plan_hash": (
            calc_flow["normalized_plan_hash"],
            raw["normalized_plan_hash"],
        ),
    }
    return [field for field, (left, right) in pairs.items() if left != right]


def _verify_case(
    raw: object,
    *,
    index: int,
    minimum_samples: int,
    require_stable: bool,
) -> None:
    # Paired ordering, correctness, comparability, and conclusions are one
    # atomic evidence contract; keep all rejection paths together.
    # #lizard forgives
    label = f"cases[{index}]"
    case = _exact_fields(raw, CASE_FIELDS, label)
    if not isinstance(case["name"], str) or not case["name"]:
        raise ValueError(f"{label}.name must be non-empty")
    rows = _positive_int(case["rows"], f"{label}.rows")
    _positive_int(case["active_entities"], f"{label}.active_entities")
    _positive_int(case["window"], f"{label}.window")
    _positive_int(case["warmups"], f"{label}.warmups")
    if case["rolling_rewrite_enabled"] is not False:
        raise ValueError(
            f"{label} rolling rewrite must be disabled for fair comparison"
        )
    calc_flow, calc_samples = _verify_engine(
        case["calc_flow"],
        label=f"{label}.calc_flow",
        rows=rows,
        minimum_samples=minimum_samples,
        require_stable=require_stable,
    )
    raw_datafusion, raw_samples = _verify_engine(
        case["raw_datafusion"],
        label=f"{label}.raw_datafusion",
        rows=rows,
        minimum_samples=minimum_samples,
        require_stable=require_stable,
    )
    if case["name"] == "dual_sma_spread":
        for engine_name, engine in (
            ("calc_flow", calc_flow),
            ("raw_datafusion", raw_datafusion),
        ):
            if engine["bounded_window_agg_count"] != 1:
                raise ValueError(
                    f"{label}.{engine_name} must contain one BoundedWindowAggExec"
                )
    if len(calc_samples) != len(raw_samples):
        raise ValueError(f"{label} paired engines have different sample counts")
    order = case["sample_order"]
    expected_order = [
        "ab" if sample % 2 == 0 else "ba" for sample in range(len(calc_samples))
    ]
    if order != expected_order:
        raise ValueError(f"{label}.sample_order must alternate AB/BA")
    ratios = case["paired_ratios"]
    if not isinstance(ratios, list) or len(ratios) != len(calc_samples):
        raise ValueError(f"{label}.paired_ratios must cover every sample pair")
    verified_ratios = []
    for sample, (reported, calc, datafusion) in enumerate(
        zip(ratios, calc_samples, raw_samples, strict=True)
    ):
        ratio = _finite(reported, f"{label}.paired_ratios[{sample}]", positive=True)
        expected = calc / datafusion
        if not math.isclose(ratio, expected, rel_tol=1e-12, abs_tol=1e-12):
            raise ValueError(f"{label}.paired_ratios[{sample}] is inconsistent")
        verified_ratios.append(ratio)
    ratio_median = _finite(
        case["paired_ratio_median"], f"{label}.paired_ratio_median", positive=True
    )
    if not math.isclose(
        ratio_median, _median(verified_ratios), rel_tol=1e-12, abs_tol=1e-12
    ):
        raise ValueError(f"{label}.paired_ratio_median is inconsistent")
    interval_low = _finite(
        case["paired_ratio_ci_low"], f"{label}.paired_ratio_ci_low", positive=True
    )
    interval_high = _finite(
        case["paired_ratio_ci_high"], f"{label}.paired_ratio_ci_high", positive=True
    )
    if not interval_low <= ratio_median <= interval_high:
        raise ValueError(f"{label} paired ratio interval is inconsistent")
    _verify_correctness(case["correctness"], label)
    actual_mismatches = _actual_mismatches(calc_flow, raw_datafusion)
    comparability = _exact_fields(
        case["comparability"], {"comparable", "mismatches"}, f"{label}.comparability"
    )
    if comparability["mismatches"] != actual_mismatches:
        mismatch = actual_mismatches[0] if actual_mismatches else "comparability"
        raise ValueError(f"{label} {mismatch} mismatch is not reported exactly")
    if comparability["comparable"] is not (not actual_mismatches):
        raise ValueError(f"{label}.comparability.comparable is inconsistent")
    conclusion = case["speedup_conclusion"]
    if actual_mismatches:
        if conclusion is not None:
            raise ValueError(
                f"{label}.speedup_conclusion must be null when incomparable"
            )
    elif not isinstance(conclusion, str) or not conclusion:
        raise ValueError(f"{label}.speedup_conclusion is required when comparable")


def _case_key(case: dict[str, Any]) -> tuple[object, ...]:
    return (
        case["name"],
        case["rows"],
        case["active_entities"],
        case["window"],
        case["calc_flow"]["effective_partitions"],
        case["calc_flow"]["batch_size"],
    )


def _verify_repeat(report: dict[str, Any], repeat: dict[str, Any]) -> None:
    # Independent-repeat validation accumulates the complete comparable case
    # identity before accepting latency or memory stability.
    # #lizard forgives
    if report["git_sha"] != repeat["git_sha"]:
        raise ValueError("repeat report git_sha does not match")
    for field in (
        "machine_fingerprint",
        "dependency_fingerprint",
        "workload_fingerprint",
    ):
        if report["environment"][field] != repeat["environment"][field]:
            raise ValueError(f"repeat report {field} does not match")
    first_cases = {_case_key(case): case for case in report["cases"]}
    second_cases = {_case_key(case): case for case in repeat["cases"]}
    if first_cases.keys() != second_cases.keys():
        raise ValueError("repeat report cases do not match")
    for key, first in first_cases.items():
        second = second_cases[key]
        for engine in ("calc_flow", "raw_datafusion"):
            if (
                first[engine]["normalized_plan_hash"]
                != second[engine]["normalized_plan_hash"]
            ):
                raise ValueError(
                    "repeat report normalized_plan_hash does not match for "
                    f"{key} {engine}"
                )
            first_median = _median(
                [float(value) for value in first[engine]["samples_ms"]]
            )
            second_median = _median(
                [float(value) for value in second[engine]["samples_ms"]]
            )
            median_ratio = max(first_median, second_median) / min(
                first_median, second_median
            )
            if median_ratio > 1.10:
                raise ValueError(
                    f"independent median differs by more than 10% for {key} {engine}"
                )
            first_rss = first[engine]["peak_rss_bytes"]
            second_rss = second[engine]["peak_rss_bytes"]
            if max(first_rss, second_rss) / min(first_rss, second_rss) > 1.15:
                raise ValueError(
                    f"independent peak RSS differs by more than 15% for {key} {engine}"
                )


def verify_p1(report: object, serial_control: object) -> None:
    """Apply the P1 matched-p16 latency, ratio, and memory requirements."""
    # P1 is a single conjunctive gate across provenance, both workloads,
    # latency, paired ratio, and peak memory.
    # #lizard forgives
    if not isinstance(report, dict) or report.get("profile") != "matched-adaptive":
        raise ValueError("P1 report must use matched-adaptive profile")
    if (
        not isinstance(serial_control, dict)
        or serial_control.get("profile") != "serial-control"
    ):
        raise ValueError("P1 memory comparison requires serial-control profile")
    if report["git_sha"] != serial_control["git_sha"]:
        raise ValueError("P1 reports must share one git_sha")
    for field in ("machine_fingerprint", "dependency_fingerprint"):
        if report["environment"][field] != serial_control["environment"][field]:
            raise ValueError(f"P1 reports must share {field}")
    serial_cases = {
        (case["name"], case["rows"], case["active_entities"]): case
        for case in serial_control["cases"]
    }
    limits = {
        "sma_20": (90.0, 1.30),
        "dual_sma_spread": (110.0, 1.20),
    }
    observed = set()
    for case in report["cases"]:
        name = case["name"]
        if (
            name not in limits
            or case["rows"] != 1_000_000
            or case["active_entities"] != 64
        ):
            continue
        observed.add(name)
        if case["calc_flow"]["effective_partitions"] != 16:
            raise ValueError(f"P1 {name} requires effective p16")
        latency_limit, ratio_limit = limits[name]
        if case["calc_flow"]["median_ms"] > latency_limit:
            raise ValueError(
                f"P1 {name} Calc Flow latency exceeds {latency_limit:g} ms"
            )
        if case["paired_ratio_median"] > ratio_limit:
            raise ValueError(f"P1 {name} ratio exceeds {ratio_limit:.2f}x")
        key = (name, case["rows"], case["active_entities"])
        serial = serial_cases.get(key)
        if serial is None:
            raise ValueError(f"P1 {name} is missing serial-control evidence")
        if serial["calc_flow"]["effective_partitions"] != 1:
            raise ValueError(f"P1 {name} serial control must use p1")
        if (
            case["calc_flow"]["peak_rss_bytes"]
            > serial["calc_flow"]["peak_rss_bytes"] * 1.5
        ):
            raise ValueError(f"P1 {name} p16 peak RSS exceeds 1.5x p1")
    missing = sorted(set(limits) - observed)
    if missing:
        raise ValueError(f"P1 report is missing workloads: {', '.join(missing)}")


def verify_report(
    raw: object,
    *,
    minimum_samples: int,
    require_stable: bool = False,
    repeat: object | None = None,
) -> None:
    """Validate one report and optionally its independent repeat."""
    # Top-level provenance, environment, cases, and repeat validation form one
    # fail-closed admission boundary for a report.
    # #lizard forgives
    report = _exact_fields(
        raw,
        {"schema_version", "git_sha", "profile", "environment", "cases"},
        "report",
    )
    if report["schema_version"] != 1:
        raise ValueError("report.schema_version must be 1")
    git_sha = report["git_sha"]
    if not isinstance(git_sha, str) or GIT_SHA.fullmatch(git_sha) is None:
        raise ValueError("report.git_sha must be a lowercase full git SHA")
    if report["profile"] not in PROFILES:
        raise ValueError("report.profile is unsupported")
    environment = _exact_fields(
        report["environment"], ENVIRONMENT_FIELDS, "environment"
    )
    for field in (
        "machine_fingerprint",
        "dependency_fingerprint",
        "workload_fingerprint",
    ):
        value = environment[field]
        if not isinstance(value, str) or FINGERPRINT.fullmatch(value) is None:
            raise ValueError(f"environment.{field} must be a SHA-256 digest")
    if environment["datafusion_version"] != "54.0.0":
        raise ValueError("environment.datafusion_version must be 54.0.0")
    for field in ("arrow_version", "build_profile", "allocator"):
        if not isinstance(environment[field], str) or not environment[field]:
            raise ValueError(f"environment.{field} must be non-empty")
    for field in ("os", "arch", "cpu_model", "rust_version"):
        if not isinstance(environment[field], str) or not environment[field]:
            raise ValueError(f"environment.{field} must be non-empty")
    _positive_int(
        environment["available_parallelism"], "environment.available_parallelism"
    )
    if type(environment["git_dirty"]) is not bool:
        raise ValueError("environment.git_dirty must be a bool")
    if require_stable and environment["build_profile"] != "release":
        raise ValueError("stable evidence requires release build profile")
    if require_stable and environment["git_dirty"]:
        raise ValueError("stable evidence requires a clean Git worktree")
    cases = report["cases"]
    if not isinstance(cases, list) or not cases:
        raise ValueError("report.cases must not be empty")
    for index, case in enumerate(cases):
        _verify_case(
            case,
            index=index,
            minimum_samples=minimum_samples,
            require_stable=require_stable,
        )
    if repeat is not None:
        verify_report(
            repeat, minimum_samples=minimum_samples, require_stable=require_stable
        )
        _verify_repeat(report, repeat)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("report", type=Path)
    parser.add_argument("--repeat", type=Path)
    parser.add_argument("--minimum-samples", type=int, default=20)
    parser.add_argument("--require-stable", action="store_true")
    parser.add_argument("--require-p1", action="store_true")
    parser.add_argument("--serial-control", type=Path)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        report = json.loads(args.report.read_text(encoding="utf-8"))
        repeat = (
            json.loads(args.repeat.read_text(encoding="utf-8"))
            if args.repeat is not None
            else None
        )
        verify_report(
            report,
            minimum_samples=args.minimum_samples,
            require_stable=args.require_stable,
            repeat=repeat,
        )
        if args.require_p1:
            if args.serial_control is None:
                raise ValueError("--require-p1 requires --serial-control")
            serial_control = json.loads(args.serial_control.read_text(encoding="utf-8"))
            verify_report(
                serial_control,
                minimum_samples=args.minimum_samples,
                require_stable=args.require_stable,
            )
            verify_p1(report, serial_control)
    except (OSError, json.JSONDecodeError, ValueError) as error:
        print(f"SQL/DataFusion performance evidence failed: {error}", file=sys.stderr)
        return 1
    print("SQL/DataFusion performance evidence passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
