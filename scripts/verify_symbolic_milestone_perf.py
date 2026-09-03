"""Verify alternating same-process symbolic milestone benchmarks.

Each selected pytest-benchmark record must carry raw hand-built/symbolic
timings collected as alternating pairs in one process.  The verifier fails
closed on dirty or mismatched commits, machine drift, workload drift, missing
pairs, invalid order, or insufficient samples.  A fixed-seed paired bootstrap
classifies the symbolic regression against the configured threshold:

- ``pass``: the 95% upper bound is at or below the threshold;
- ``fail``: the 95% lower bound is above the threshold;
- ``inconclusive``: the interval crosses the threshold and needs more data.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
import struct
import sys
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any, Literal, TypedDict

DEFAULT_THRESHOLD = 0.05
DEFAULT_BOOTSTRAP_RESAMPLES = 20_000
BOOTSTRAP_SEED = 20_260_829
MIN_PAIRED_SAMPLES = 20
ROLLING_MIN_PAIRED_SAMPLES = 60
GIT_SHA = re.compile(r"[0-9a-f]{40}")
FINGERPRINT = re.compile(r"[0-9a-f]{64}")
COMPARISON_CONTRACT = "same-process-alternating-v1"
ROLLING_SCENARIOS = frozenset({"rolling_kernel_sma20", "rolling_kernel_dual_sma_5_20"})

Decision = Literal["pass", "fail", "inconclusive"]

_VOLATILE_EXTRA_FIELDS = frozenset(
    {
        "datafusion_execution_ns",
        "datafusion_planning_ns",
        "paired_samples",
        "peak_rss_bytes",
        "process_rss_bytes",
    }
)


class PairedSample(TypedDict):
    order: Literal["hand-built-first", "symbolic-first"]
    hand_built_seconds: float
    symbolic_seconds: float


class ScenarioEvidence(TypedDict):
    workload: dict[str, object]
    pairs: list[PairedSample]


def _read_report(path: Path) -> dict[str, Any]:
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"cannot read benchmark report {path}: {error}") from error
    if not isinstance(raw, dict):
        raise ValueError(f"benchmark report {path} must contain a JSON object")
    return raw


def _validated_commit(report: Mapping[str, object], path: Path) -> str:
    commit = report.get("commit_info")
    if not isinstance(commit, dict):
        raise ValueError(f"benchmark report {path} has no commit provenance")
    commit_id = commit.get("id")
    if not isinstance(commit_id, str) or GIT_SHA.fullmatch(commit_id) is None:
        raise ValueError(f"benchmark report {path} requires a full commit id")
    if commit.get("dirty") is not False:
        raise ValueError(f"benchmark report {path} was captured from a dirty tree")
    return commit_id


def _machine_fingerprint(report: Mapping[str, object], path: Path) -> dict[str, object]:
    machine = report.get("machine_info")
    if not isinstance(machine, dict):
        raise ValueError(f"benchmark report {path} has no machine identity")
    cpu = machine.get("cpu")
    if not isinstance(cpu, dict):
        raise ValueError(f"benchmark report {path} has no CPU identity")
    fingerprint = {
        "node": machine.get("node"),
        "machine": machine.get("machine"),
        "system": machine.get("system"),
        "release": machine.get("release"),
        "python_implementation": machine.get("python_implementation"),
        "python_version": machine.get("python_version"),
        "cpu_arch": cpu.get("arch"),
        "cpu_bits": cpu.get("bits"),
        "cpu_brand": cpu.get("brand_raw"),
        "cpu_count": cpu.get("count"),
    }
    missing = sorted(key for key, value in fingerprint.items() if value in (None, ""))
    if missing:
        raise ValueError(
            f"benchmark report {path} has incomplete machine identity: "
            f"{', '.join(missing)}"
        )
    return fingerprint


def _workload(extra: Mapping[str, object], path: Path) -> dict[str, object]:
    required = {
        "scenario",
        "comparison_contract",
        "workload_contract",
        "scale",
        "input_rows",
        "output_rows",
    }
    missing = sorted(required - set(extra))
    if missing:
        raise ValueError(
            f"benchmark report {path} has incomplete workload metadata: "
            f"{', '.join(missing)}"
        )
    if extra["comparison_contract"] != COMPARISON_CONTRACT:
        raise ValueError(f"benchmark report {path} has unsupported comparison contract")
    if extra["scenario"] in ROLLING_SCENARIOS:
        _validate_rolling_evidence(extra, path)
    return {
        key: value
        for key, value in sorted(extra.items())
        if key not in _VOLATILE_EXTRA_FIELDS
    }


def _validate_rolling_evidence(extra: Mapping[str, object], path: Path) -> None:
    scenario = str(extra["scenario"])
    required = {
        "machine_fingerprint",
        "dependency_fingerprint",
        "workload_fingerprint",
        "oracle",
        "oracle_checked_rows",
        "oracle_finite_rows",
        "oracle_rtol",
        "oracle_atol",
        "optimized_kernel",
        "optimized_shared_state_groups",
        "reference_rolling_rewrites",
        "fast_window",
        "slow_window",
    }
    missing = sorted(required - set(extra))
    if missing:
        raise ValueError(
            f"rolling benchmark report {path} lacks evidence: {', '.join(missing)}"
        )
    for field in (
        "machine_fingerprint",
        "dependency_fingerprint",
        "workload_fingerprint",
    ):
        value = extra[field]
        if not isinstance(value, str) or FINGERPRINT.fullmatch(value) is None:
            raise ValueError(f"rolling benchmark report {path} has invalid {field}")
    checked_rows = extra["oracle_checked_rows"]
    finite_rows = extra["oracle_finite_rows"]
    if (
        type(checked_rows) is not int
        or checked_rows <= 0
        or checked_rows != extra["input_rows"]
        or checked_rows != extra["output_rows"]
    ):
        raise ValueError(f"rolling benchmark report {path} has vacuous oracle coverage")
    if type(finite_rows) is not int or not 0 < finite_rows <= checked_rows:
        raise ValueError(
            f"rolling benchmark report {path} has no finite oracle outputs"
        )
    if extra["oracle"] != "independent_direct_window_v1":
        raise ValueError(f"rolling benchmark report {path} has an unknown oracle")
    for field in ("oracle_rtol", "oracle_atol"):
        value = extra[field]
        if (
            isinstance(value, bool)
            or not isinstance(value, int | float)
            or not math.isfinite(float(value))
            or not 0.0 <= float(value) <= 1e-10
        ):
            raise ValueError(f"rolling benchmark report {path} has invalid {field}")
    expected_fast = 5 if scenario == "rolling_kernel_dual_sma_5_20" else None
    expected_groups = 2 if expected_fast is not None else 1
    if (
        extra["optimized_kernel"] != "ordered_primitive"
        or extra["optimized_shared_state_groups"] != expected_groups
        or extra["reference_rolling_rewrites"] != 0
        or extra["slow_window"] != 20
        or extra["fast_window"] != expected_fast
    ):
        raise ValueError(f"rolling benchmark report {path} has invalid kernel evidence")


def _seconds(value: object, path: Path) -> float:
    if not isinstance(value, int | float):
        raise ValueError(f"benchmark report {path} has a non-numeric paired sample")
    seconds = float(value)
    if not math.isfinite(seconds) or seconds <= 0.0:
        raise ValueError(f"benchmark report {path} has an invalid paired sample")
    return seconds


def _paired_samples(extra: Mapping[str, object], path: Path) -> list[PairedSample]:
    raw_pairs = extra.get("paired_samples")
    if not isinstance(raw_pairs, list) or not raw_pairs:
        raise ValueError(f"benchmark report {path} lacks a same-process pair")
    pairs: list[PairedSample] = []
    expected_order = "hand-built-first"
    for index, raw_pair in enumerate(raw_pairs):
        if not isinstance(raw_pair, dict):
            raise ValueError(f"benchmark report {path} has an invalid paired sample")
        order = raw_pair.get("order")
        if order != expected_order:
            raise ValueError(
                f"benchmark report {path} violates alternating order at pair {index}"
            )
        pairs.append(
            {
                "order": order,
                "hand_built_seconds": _seconds(
                    raw_pair.get("hand_built_seconds"), path
                ),
                "symbolic_seconds": _seconds(raw_pair.get("symbolic_seconds"), path),
            }
        )
        expected_order = (
            "symbolic-first"
            if expected_order == "hand-built-first"
            else "hand-built-first"
        )
    return pairs


def _report_evidence(
    report: Mapping[str, object],
    path: Path,
    scenarios: Sequence[str],
) -> dict[str, ScenarioEvidence]:
    benchmarks = report.get("benchmarks")
    if not isinstance(benchmarks, list):
        raise ValueError(f"benchmark report {path} has no benchmark records")
    selected: dict[str, ScenarioEvidence] = {}
    for raw_benchmark in benchmarks:
        if not isinstance(raw_benchmark, dict):
            continue
        extra = raw_benchmark.get("extra_info")
        if not isinstance(extra, dict):
            continue
        scenario = extra.get("scenario")
        if scenario not in scenarios:
            continue
        scenario_name = str(scenario)
        if scenario_name in selected:
            raise ValueError(
                f"benchmark report {path} repeats scenario {scenario_name!r}"
            )
        selected[scenario_name] = {
            "workload": _workload(extra, path),
            "pairs": _paired_samples(extra, path),
        }
    missing = sorted(set(scenarios) - set(selected))
    if missing:
        raise ValueError(
            f"benchmark report {path} lacks a same-process pair for "
            f"{', '.join(missing)}"
        )
    return selected


def _quantile(values: Sequence[float], probability: float) -> float:
    if not values:
        raise ValueError("cannot calculate a quantile without values")
    position = (len(values) - 1) * probability
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return values[lower]
    weight = position - lower
    return values[lower] * (1.0 - weight) + values[upper] * weight


def _paired_log_ratios(pairs: Sequence[PairedSample]) -> list[float]:
    return [
        math.log(pair["symbolic_seconds"] / pair["hand_built_seconds"])
        for pair in pairs
    ]


def _bootstrap_regression_interval(
    log_ratios: Sequence[float], *, resamples: int
) -> tuple[float, float]:
    if resamples < 1_000:
        raise ValueError("bootstrap_resamples must be at least 1000")
    regressions: list[float] = []
    sample_size = len(log_ratios)
    for resample_index in range(resamples):
        random_bytes = hashlib.shake_256(
            f"{BOOTSTRAP_SEED}:{resample_index}".encode()
        ).digest(sample_size * 8)
        mean_log_ratio = (
            math.fsum(
                log_ratios[value[0] % sample_size]
                for value in struct.iter_unpack(">Q", random_bytes)
            )
            / sample_size
        )
        regressions.append(math.exp(mean_log_ratio) - 1.0)
    regressions.sort()
    return _quantile(regressions, 0.025), _quantile(regressions, 0.975)


def _decision(interval: tuple[float, float], threshold: float) -> Decision:
    lower, upper = interval
    if upper <= threshold:
        return "pass"
    if lower > threshold:
        return "fail"
    return "inconclusive"


def _overall_decision(decisions: Sequence[Decision]) -> Decision:
    if "fail" in decisions:
        return "fail"
    if "inconclusive" in decisions:
        return "inconclusive"
    return "pass"


def _report_digest(path: Path) -> str:
    try:
        return hashlib.sha256(path.read_bytes()).hexdigest()
    except OSError as error:
        raise ValueError(f"cannot hash benchmark report {path}: {error}") from error


def _validate_comparison_request(
    report_paths: Sequence[Path], scenarios: Sequence[str], threshold: float
) -> None:
    if not report_paths:
        raise ValueError("at least one benchmark report is required")
    if not scenarios or len(set(scenarios)) != len(scenarios):
        raise ValueError("scenarios must be a non-empty unique sequence")
    if not math.isfinite(threshold) or threshold < 0.0:
        raise ValueError("threshold must be finite and non-negative")


def _validated_reports(
    report_paths: Sequence[Path],
) -> tuple[str, dict[str, object], list[tuple[Path, dict[str, Any]]]]:
    reports = [(path, _read_report(path)) for path in report_paths]
    first_path, first_report = reports[0]
    commit_id = _validated_commit(first_report, first_path)
    machine = _machine_fingerprint(first_report, first_path)
    for path, report in reports[1:]:
        if _validated_commit(report, path) != commit_id:
            raise ValueError("benchmark reports do not share one exact commit")
        if _machine_fingerprint(report, path) != machine:
            raise ValueError("benchmark reports do not share one stable machine")
    return commit_id, machine, reports


def _combined_evidence(
    reports: Sequence[tuple[Path, Mapping[str, object]]],
    scenarios: Sequence[str],
) -> tuple[dict[str, ScenarioEvidence], list[dict[str, str]]]:
    combined: dict[str, ScenarioEvidence] = {
        scenario: {"workload": {}, "pairs": []} for scenario in scenarios
    }
    report_digests: list[dict[str, str]] = []
    for path, report in reports:
        evidence = _report_evidence(report, path, scenarios)
        for scenario, current in evidence.items():
            target = combined[scenario]
            if target["workload"] and target["workload"] != current["workload"]:
                raise ValueError(
                    f"benchmark reports have a workload mismatch for {scenario!r}"
                )
            target["workload"] = current["workload"]
            target["pairs"].extend(current["pairs"])
        report_digests.append({"path": path.as_posix(), "sha256": _report_digest(path)})
    return combined, report_digests


def _scenario_result(
    scenario: str,
    evidence: ScenarioEvidence,
    *,
    threshold: float,
    bootstrap_resamples: int,
) -> tuple[dict[str, object], Decision]:
    pairs = evidence["pairs"]
    minimum_pairs = (
        ROLLING_MIN_PAIRED_SAMPLES
        if scenario in ROLLING_SCENARIOS
        else MIN_PAIRED_SAMPLES
    )
    if len(pairs) < minimum_pairs:
        raise ValueError(
            f"scenario {scenario!r} requires at least {minimum_pairs} paired samples"
        )
    log_ratios = _paired_log_ratios(pairs)
    interval = _bootstrap_regression_interval(log_ratios, resamples=bootstrap_resamples)
    decision = _decision(interval, threshold)
    mean_log_ratio = math.fsum(log_ratios) / len(log_ratios)
    return (
        {
            "scenario": scenario,
            "decision": decision,
            "threshold_percent": threshold * 100.0,
            "geometric_regression_percent": (math.exp(mean_log_ratio) - 1.0) * 100.0,
            "regression_interval_percent": [
                interval[0] * 100.0,
                interval[1] * 100.0,
            ],
            "paired_samples": len(pairs),
            "hand_built_first_samples": sum(
                pair["order"] == "hand-built-first" for pair in pairs
            ),
            "symbolic_first_samples": sum(
                pair["order"] == "symbolic-first" for pair in pairs
            ),
            "workload": evidence["workload"],
        },
        decision,
    )


def compare_reports(
    report_paths: Sequence[Path],
    *,
    scenarios: Sequence[str],
    threshold: float = DEFAULT_THRESHOLD,
    bootstrap_resamples: int = DEFAULT_BOOTSTRAP_RESAMPLES,
    expected_commit: str | None = None,
) -> dict[str, object]:
    """Validate reports and return a deterministic comparison summary."""
    _validate_comparison_request(report_paths, scenarios, threshold)
    commit_id, machine, reports = _validated_reports(report_paths)
    if expected_commit is not None:
        if GIT_SHA.fullmatch(expected_commit) is None:
            raise ValueError("expected_commit must be a lowercase full git SHA")
        if commit_id != expected_commit:
            raise ValueError("benchmark reports do not match the expected commit")
    combined, report_digests = _combined_evidence(reports, scenarios)
    results = [
        _scenario_result(
            scenario,
            combined[scenario],
            threshold=threshold,
            bootstrap_resamples=bootstrap_resamples,
        )
        for scenario in scenarios
    ]
    scenario_results = [result for result, _decision_name in results]
    decisions = [decision for _result, decision in results]
    return {
        "format_version": 1,
        "decision": _overall_decision(decisions),
        "benchmark_commit": commit_id,
        "machine": machine,
        "bootstrap_seed": BOOTSTRAP_SEED,
        "bootstrap_resamples": bootstrap_resamples,
        "reports": report_digests,
        "scenarios": scenario_results,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--report",
        action="append",
        type=Path,
        required=True,
        help="pytest-benchmark JSON report with alternating paired samples",
    )
    parser.add_argument("--expected-commit")
    parser.add_argument(
        "--scenario",
        action="append",
        required=True,
        help="exact scenario name to verify",
    )
    parser.add_argument(
        "--threshold-percent",
        type=float,
        default=DEFAULT_THRESHOLD * 100.0,
    )
    parser.add_argument(
        "--bootstrap-resamples",
        type=int,
        default=DEFAULT_BOOTSTRAP_RESAMPLES,
    )
    parser.add_argument("--output", type=Path)
    options = parser.parse_args()
    try:
        summary = compare_reports(
            options.report,
            scenarios=options.scenario,
            threshold=options.threshold_percent / 100.0,
            bootstrap_resamples=options.bootstrap_resamples,
            expected_commit=options.expected_commit,
        )
    except ValueError as error:
        print(f"Invalid symbolic performance evidence: {error}", file=sys.stderr)
        return 1

    rendered = json.dumps(summary, indent=2, sort_keys=True) + "\n"
    if options.output is not None:
        options.output.write_text(rendered, encoding="utf-8")
    print(rendered, end="")
    return 0 if summary["decision"] == "pass" else 1


if __name__ == "__main__":
    raise SystemExit(main())
