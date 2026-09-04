"""Produce fail-closed P4 attribution and P5/P6/P7 gate decisions."""

from __future__ import annotations

import argparse
import json
import math
import statistics
import sys
from pathlib import Path
from typing import Any

try:
    from scripts.verify_sql_datafusion_performance import verify_report
except ModuleNotFoundError:  # Direct `python scripts/...` invocation.
    from verify_sql_datafusion_performance import verify_report

FIXED_PHASES = (
    "runtime_acquire",
    "session_state_create",
    "input_adapter",
    "table_register",
    "sql_parse",
    "logical_optimize",
    "physical_plan",
    "metrics_traversal",
    "run_result",
    "run_session_envelope",
)
MATERIALIZATION_PHASES = ("output_arrow_wrap", "batch_envelope")
EXECUTION_PHASES = ("execution_to_first_batch", "execution_remaining")


def _phase_total(engine: dict[str, Any], phases: tuple[str, ...]) -> float:
    timings = engine["phase_medians_ms"]
    return sum(float(timings[phase]) for phase in phases)


def _phase_sample_totals(
    engine: dict[str, Any], phases: tuple[str, ...]
) -> list[float]:
    samples = engine["phase_samples_ms"]
    return [
        sum(float(samples[phase][index]) for phase in phases)
        for index in range(len(engine["samples_ms"]))
    ]


def _decision(enabled: bool, reason: str) -> dict[str, str]:
    return {"decision": "go" if enabled else "no-go", "reason": reason}


def _analyze_case(case: dict[str, Any]) -> dict[str, Any]:
    # Attribution and all P5/P6/P7 decisions share one fail-closed case boundary
    # so no component can be reported without the matching residual checks.
    # #lizard forgives
    if case["comparability"]["comparable"] is not True:
        raise ValueError(f"attribution case {case['name']} must be comparable")
    calc = case["calc_flow"]
    raw = case["raw_datafusion"]
    gap_samples = [
        float(calc_sample) - float(raw_sample)
        for calc_sample, raw_sample in zip(
            calc["samples_ms"], raw["samples_ms"], strict=True
        )
    ]
    gap_ms = statistics.median(gap_samples)
    calc_execution = _phase_total(calc, EXECUTION_PHASES)
    raw_execution = _phase_total(raw, EXECUTION_PHASES)
    calc_fixed = _phase_total(calc, FIXED_PHASES)
    raw_fixed = _phase_total(raw, FIXED_PHASES)
    calc_materialization = _phase_total(calc, MATERIALIZATION_PHASES)
    raw_materialization = _phase_total(raw, MATERIALIZATION_PHASES)
    component_samples = {
        "execution": [
            left - right
            for left, right in zip(
                _phase_sample_totals(calc, EXECUTION_PHASES),
                _phase_sample_totals(raw, EXECUTION_PHASES),
                strict=True,
            )
        ],
        "fixed_envelope": [
            left - right
            for left, right in zip(
                _phase_sample_totals(calc, FIXED_PHASES),
                _phase_sample_totals(raw, FIXED_PHASES),
                strict=True,
            )
        ],
        "materialization": [
            left - right
            for left, right in zip(
                _phase_sample_totals(calc, MATERIALIZATION_PHASES),
                _phase_sample_totals(raw, MATERIALIZATION_PHASES),
                strict=True,
            )
        ],
    }
    components = {
        name: statistics.median(values) for name, values in component_samples.items()
    }
    if gap_ms <= 0:
        residual_ms = 0.0
        explained_ms = gap_ms
        explained_fraction = 1.0
        gap_status = "no_positive_gap"
    else:
        residual_samples = [
            gap
            - component_samples["execution"][index]
            - component_samples["fixed_envelope"][index]
            - component_samples["materialization"][index]
            for index, gap in enumerate(gap_samples)
        ]
        residual_ms = statistics.median(residual_samples)
        explained_ms = gap_ms - residual_ms
        explained_fraction = max(
            0.0,
            1.0
            - statistics.median(abs(value) for value in residual_samples)
            / max(statistics.median(abs(value) for value in gap_samples), 1e-12),
        )
        gap_status = "positive_gap"
    if gap_status == "positive_gap" and explained_fraction < 0.90:
        raise ValueError(
            f"attribution case {case['name']} explains only "
            f"{explained_fraction:.1%}; at least 90% is required"
        )

    window_share = float(calc["window_compute_ms"]) / max(calc_execution, 1e-12)
    fixed_share = calc_fixed / float(calc["median_ms"])
    p5_duplicate = any(
        int(calc[field]) > int(raw[field])
        for field in (
            "repartition_operator_count",
            "sort_operator_count",
            "coalesce_operator_count",
        )
    )
    p5_share = max(0.0, components["materialization"]) / gap_ms if gap_ms > 0 else 0.0
    p5_go = p5_duplicate and p5_share >= 0.20
    p6_go = window_share >= 0.50
    p7_go = gap_ms > 0 and (fixed_share >= 0.10 or calc_fixed > 1.0)

    return {
        "name": case["name"],
        "normalized_plan_hash": calc["normalized_plan_hash"],
        "total_gap_ms": gap_ms,
        "gap_status": gap_status,
        "components_ms": components,
        "explained_ms": explained_ms,
        "residual_ms": residual_ms,
        "explained_fraction": explained_fraction,
        "diagnostic_subphases_ms": {
            phase: float(calc["phase_medians_ms"][phase])
            for phase in (
                "audit",
                "physical_plan_string",
                "collect_or_coalesce",
            )
        },
        "execution": {
            "calc_flow_ms": calc_execution,
            "raw_datafusion_ms": raw_execution,
            "window_compute_ms": float(calc["window_compute_ms"]),
            "window_share": window_share,
        },
        "fixed_envelope": {
            "calc_flow_ms": calc_fixed,
            "raw_datafusion_ms": raw_fixed,
            "calc_flow_total_share": fixed_share,
        },
        "materialization": {
            "calc_flow_ms": calc_materialization,
            "raw_datafusion_ms": raw_materialization,
            "gap_share": p5_share,
            "calc_has_extra_shuffle_or_merge": p5_duplicate,
        },
        "gates": {
            "p5": _decision(
                p5_go,
                "extra repartition/sort/coalesce and materialization gap >=20%"
                if p5_go
                else (
                    "no Calc Flow-only duplicate shuffle/merge meeting the 20% "
                    "threshold"
                ),
            ),
            "p6": _decision(
                p6_go,
                "WindowAgg compute is at least 50% of execution"
                if p6_go
                else "WindowAgg compute is below 50% of execution",
            ),
            "p7": _decision(
                p7_go,
                "fixed envelope is at least 10% of total or exceeds 1 ms"
                if p7_go
                else (
                    "no positive gap, or fixed envelope is below 10% and does "
                    "not exceed 1 ms"
                ),
            ),
        },
    }


def analyze_report(report: object) -> dict[str, Any]:
    if not isinstance(report, dict) or report.get("profile") != "attribution":
        raise ValueError("P4 analysis requires an attribution report")
    verify_report(report, minimum_samples=20, require_stable=True)
    cases = report.get("cases")
    if not isinstance(cases, list) or not cases:
        raise ValueError("attribution report must contain cases")
    analyzed = [_analyze_case(case) for case in cases]
    if any(not math.isfinite(case["explained_fraction"]) for case in analyzed):
        raise ValueError("attribution output contains a non-finite value")
    return {
        "schema_version": 1,
        "git_sha": report["git_sha"],
        "source_profile": "attribution",
        "machine_fingerprint": report["environment"]["machine_fingerprint"],
        "dependency_fingerprint": report["environment"]["dependency_fingerprint"],
        "source_git_dirty": report["environment"]["git_dirty"],
        "cases": analyzed,
    }


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("report", type=Path)
    parser.add_argument("--output", required=True, type=Path)
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    try:
        report = json.loads(args.report.read_text(encoding="utf-8"))
        analysis = analyze_report(report)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        temporary = args.output.with_suffix(args.output.suffix + ".tmp")
        temporary.write_text(json.dumps(analysis, indent=2) + "\n", encoding="utf-8")
        temporary.replace(args.output)
    except (OSError, ValueError, json.JSONDecodeError) as error:
        print(f"SQL/DataFusion attribution failed: {error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
