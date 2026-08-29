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
import random
import re
import sys
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any, Literal, TypedDict

DEFAULT_THRESHOLD = 0.05
DEFAULT_BOOTSTRAP_RESAMPLES = 20_000
BOOTSTRAP_SEED = 20_260_829
MIN_PAIRED_SAMPLES = 20
GIT_SHA = re.compile(r"[0-9a-f]{40}")
COMPARISON_CONTRACT = "same-process-alternating-v1"

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
    return {
        key: value
        for key, value in sorted(extra.items())
        if key not in _VOLATILE_EXTRA_FIELDS
    }


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
    generator = random.Random(BOOTSTRAP_SEED)
    regressions: list[float] = []
    for _ in range(resamples):
        mean_log_ratio = math.fsum(
            generator.choice(log_ratios) for _ in range(len(log_ratios))
        ) / len(log_ratios)
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


def compare_reports(
    report_paths: Sequence[Path],
    *,
    scenarios: Sequence[str],
    threshold: float = DEFAULT_THRESHOLD,
    bootstrap_resamples: int = DEFAULT_BOOTSTRAP_RESAMPLES,
) -> dict[str, object]:
    """Validate reports and return a deterministic comparison summary."""
    if not report_paths:
        raise ValueError("at least one benchmark report is required")
    if not scenarios or len(set(scenarios)) != len(scenarios):
        raise ValueError("scenarios must be a non-empty unique sequence")
    if not math.isfinite(threshold) or threshold < 0.0:
        raise ValueError("threshold must be finite and non-negative")

    commit_id: str | None = None
    machine: dict[str, object] | None = None
    combined: dict[str, ScenarioEvidence] = {
        scenario: {"workload": {}, "pairs": []} for scenario in scenarios
    }
    report_digests: list[dict[str, str]] = []
    for path in report_paths:
        report = _read_report(path)
        current_commit = _validated_commit(report, path)
        current_machine = _machine_fingerprint(report, path)
        if commit_id is None:
            commit_id = current_commit
            machine = current_machine
        elif current_commit != commit_id:
            raise ValueError("benchmark reports do not share one exact commit")
        elif current_machine != machine:
            raise ValueError("benchmark reports do not share one stable machine")

        evidence = _report_evidence(report, path, scenarios)
        for scenario, current in evidence.items():
            target = combined[scenario]
            if not target["workload"]:
                target["workload"] = current["workload"]
            elif target["workload"] != current["workload"]:
                raise ValueError(
                    f"benchmark reports have a workload mismatch for {scenario!r}"
                )
            target["pairs"].extend(current["pairs"])
        report_digests.append({"path": path.as_posix(), "sha256": _report_digest(path)})

    scenario_results: list[dict[str, object]] = []
    decisions: list[Decision] = []
    for scenario in scenarios:
        evidence = combined[scenario]
        pairs = evidence["pairs"]
        if len(pairs) < MIN_PAIRED_SAMPLES:
            raise ValueError(
                f"scenario {scenario!r} requires at least "
                f"{MIN_PAIRED_SAMPLES} paired samples"
            )
        log_ratios = _paired_log_ratios(pairs)
        interval = _bootstrap_regression_interval(
            log_ratios, resamples=bootstrap_resamples
        )
        decision = _decision(interval, threshold)
        decisions.append(decision)
        mean_log_ratio = math.fsum(log_ratios) / len(log_ratios)
        scenario_results.append(
            {
                "scenario": scenario,
                "decision": decision,
                "threshold_percent": threshold * 100.0,
                "geometric_regression_percent": (math.exp(mean_log_ratio) - 1.0)
                * 100.0,
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
            }
        )

    if commit_id is None or machine is None:
        raise AssertionError("validated reports must establish provenance")
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
