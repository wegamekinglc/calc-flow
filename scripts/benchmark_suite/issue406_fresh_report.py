"""Separate cff retained-OFF phase comparisons with verified primary references."""

from __future__ import annotations

import math

from scripts.benchmark_suite.issue406_acceptance import validate_observation
from scripts.benchmark_suite.issue406_fresh import (
    CANDIDATE,
    PREVIOUS,
    retained_cases,
    validate_retained_case,
)
from scripts.benchmark_suite.report import ROUNDS, SAMPLES, comparison, table

PHASES = ("processing_seconds", "terminal_seconds", "full_cycle_seconds")


def _record(records: list[dict], case, side: str) -> dict:
    matches = [
        row for row in records if row.get("case") == case.id and row.get("side") == side
    ]
    return matches[0] if len(matches) == 1 else {}


def _validation_errors(report: dict, case, records: dict) -> list[str]:
    errors = []
    try:
        validate_retained_case(report, case)
    except (KeyError, TypeError, ValueError, IndexError) as error:
        errors.append(str(error))
    for side, record in records.items():
        samples = [
            *record.get("warmups", []),
            *(sample for group in record.get("rounds", []) for sample in group),
        ]
        for sample in samples:
            try:
                # Both roles are OFF-capable; their actual product identities
                # and comparison roles remain checked by the ledger validator.
                validate_observation(case, "candidate", sample)
                for field in PHASES:
                    value = sample.get(field)
                    if (
                        type(value) not in (int, float)
                        or not math.isfinite(value)
                        or value <= 0
                    ):
                        raise ValueError(f"{side}: invalid {field}")
            except ValueError as error:
                errors.append(f"{side}: {error}")
    return sorted(set(errors))


def _phase(case, metric: str, records: dict, *, valid: bool, suitable: bool) -> dict:
    from scripts.benchmark_suite.issue406_report import _stats

    rounds = {
        side: [
            [sample.get(metric) for sample in group]
            for group in record.get("rounds", [])
        ]
        for side, record in records.items()
    }
    result = {"verdict": "unmeasured-or-invalid", "round_intervals": []}
    error = None
    if valid:
        try:
            paired = comparison(
                {
                    "id": f"{case.id}/{metric}",
                    "rows": case.rows,
                    "correctness": True,
                    "comparison": "interleaved",
                    "baseline": rounds["previous"],
                    "candidate": rounds["candidate"],
                }
            )
            result = {
                **paired,
                "maintained_verdict": paired["verdict"],
                "verdict": paired["verdict"] if suitable else "inconclusive",
            }
        except ValueError as failure:
            error = str(failure)
    return {
        "metric": metric,
        **{
            side: _stats([value for group in values for value in group])
            for side, values in rounds.items()
        },
        "comparison": result,
        "validation_error": error,
    }


def _case(report: dict, case, suitable: bool) -> dict:
    control = report.get("retained_control", {})
    records = {
        "previous": _record(control.get("measurements", []), case, "previous"),
        "candidate": _record(report.get("measurements", []), case, "candidate"),
    }
    counts = {
        side: [len(group) for group in record.get("rounds", [])]
        for side, record in records.items()
    }
    complete = all(
        counts[side] == [SAMPLES] * ROUNDS and len(record.get("warmups", [])) == ROUNDS
        for side, record in records.items()
    )
    errors = _validation_errors(report, case, records)
    phases = [
        _phase(case, metric, records, valid=complete and not errors, suitable=suitable)
        for metric in PHASES
    ]
    errors.extend(
        phase["validation_error"] for phase in phases if phase["validation_error"]
    )
    return {
        "case": case.id,
        "rows": case.rows,
        "mode": "off",
        "threads": case.threads,
        "status": "incomplete" if not complete else "invalid" if errors else "complete",
        "sample_counts": counts,
        "validation_errors": sorted(set(errors)),
        "phases": phases,
    }


def summarize_retained_control(report: dict) -> dict:
    """Keep each size's complete pairs independent of other missing controls."""
    assessment = report.get("assessment", {})
    suitable = bool(
        assessment.get("compatibility_verified")
        and assessment.get("environment_suitable")
        and not report.get("settlement", {}).get("forced_kill_pids")
    )
    return {
        "schema": "calc-flow.issue406-retained-control-summary.v1",
        "previous": {"label": "cff retained-OFF control", "product_commit": PREVIOUS},
        "candidate": {"label": "current candidate", "product_commit": CANDIDATE},
        "cases": [_case(report, case, suitable) for case in retained_cases()],
        "scope": (
            "Fresh cff/current temporal pairs. Candidate observations reference the "
            "primary ledger; they are not additional independent samples or a main "
            "comparison."
        ),
    }


def render_retained_control(summary: dict) -> list[str]:
    """Render all phase clocks and retain incomplete/invalid control rows."""
    from scripts.benchmark_suite.issue406_report import _stat_cell

    rows = []
    for case in summary["cases"]:
        for phase in case["phases"]:
            paired = phase["comparison"]
            intervals = "; ".join(
                f"r{index + 1}: {item['median']:.3f}% "
                f"[{item['low']:.3f}%, {item['high']:.3f}%], "
                f"coverage {item['coverage']:.6f}"
                for index, item in enumerate(paired["round_intervals"])
            )
            rows.append(
                [
                    case["rows"],
                    phase["metric"],
                    case["status"],
                    _stat_cell(phase["previous"], 1000),
                    _stat_cell(phase["candidate"], 1000),
                    paired["verdict"],
                    intervals or "unmeasured",
                ]
            )
    return [
        "## Supplementary retained-OFF comparison",
        "",
        f"cff control `{summary['previous']['product_commit']}` versus current "
        f"`{summary['candidate']['product_commit']}`.",
        "",
        summary["scope"],
        "",
        table(
            [
                "Rows",
                "Phase",
                "Evidence",
                "cff p50/p95 [median interval], ms",
                "Current p50/p95 [median interval], ms",
                "Paired verdict",
                "Round paired median changes [interval]",
            ],
            rows,
        ),
        "",
        "Warmups are excluded from statistics. Terminal includes the final watermark "
        "before EOF; the full-cycle clock is reported directly. Partial or invalid "
        "controls have no acceptance verdict; incompatible or unsuitable comparisons "
        "remain inconclusive.",
        "",
    ]
