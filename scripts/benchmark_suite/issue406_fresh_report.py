"""Separate cff retained-OFF phase comparisons with verified primary references."""

from __future__ import annotations

import math

from scripts.benchmark_suite.issue406_acceptance import validate_observation
from scripts.benchmark_suite.issue406_fresh import (
    PREVIOUS,
    retained_cases,
    validate_retained_case,
)
from scripts.benchmark_suite.issue406_fresh_v3 import candidate_commit
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
                _validate_phase_sample(case, side, sample)
            except ValueError as error:
                errors.append(f"{side}: {error}")
    return sorted(set(errors))


def _validate_phase_sample(case, side, sample):
    # Both roles are OFF-capable; their actual product identities
    # and comparison roles remain checked by the ledger validator.
    validate_observation(case, "candidate", sample)
    for field in PHASES:
        value = sample.get(field)
        if type(value) not in (int, float) or not math.isfinite(value) or value <= 0:
            raise ValueError(f"{side}: invalid {field}")


def _phase(case, metric: str, records: dict, *, valid: bool, suitable: bool) -> dict:
    from scripts.benchmark_suite.issue406_report import _stats

    rounds = _phase_rounds(records, metric)
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


def _phase_rounds(records, metric):
    return {
        side: [
            [sample.get(metric) for sample in group]
            for group in record.get("rounds", [])
        ]
        for side, record in records.items()
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
    complete = _complete_records(counts, records)
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
        "status": _case_status(complete, errors),
        "sample_counts": counts,
        "validation_errors": sorted(set(errors)),
        "phases": phases,
    }


def _complete_records(counts, records):
    return all(
        counts[side] == [SAMPLES] * ROUNDS and len(record.get("warmups", [])) == ROUNDS
        for side, record in records.items()
    )


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
        "candidate": {
            "label": "current candidate",
            "product_commit": candidate_commit(report),
        },
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


def _case_status(complete: bool, errors: list) -> str:
    if not complete:
        return "incomplete"
    return "invalid" if errors else "complete"


def summarize_v3(report: dict) -> dict:
    """Retain every target while failed diagnostics never enter valid statistics."""
    from scripts.benchmark_suite import issue406_fresh_v3 as v3
    from scripts.benchmark_suite.issue406_report import summarize_acceptance

    clean, invalid = _valid_v3_ledger(report)
    summary = summarize_acceptance(clean)
    errors = list(summary["evidence_validation_errors"])
    try:
        v3.validate_complete_results(report)
    except (KeyError, TypeError, ValueError, IndexError) as error:
        errors.append(str(error))
    outcomes = _v3_outcomes(report, invalid)
    cases = [
        {
            **case,
            **{
                side: _v3_arm_state(case, side, outcomes)
                for side in ("baseline", "candidate")
            },
        }
        for case in summary["cases"]
    ]
    complete = summary["evidence_complete"] and not errors
    return {
        **summary,
        "schema": "calc-flow.issue406-summary.v3",
        "cases": cases,
        "evidence_complete": complete,
        "evidence_validation_errors": sorted(set(errors)),
        "acquisition_status": "complete" if complete else "incomplete",
        "acceptance_status": _v3_acceptance_status(report, complete),
        "failed_attempts": report.get("failed_attempts", []),
        "failed_arms": report.get("failed_arms", []),
        "dependent_unattempted": report.get("dependent_unattempted", []),
        "candidate_product_commit": candidate_commit(report),
    }


def _valid_v3_ledger(report):
    invalid = set()

    def validated(records):
        output = []
        for record in records:
            samples = [
                *record.get("warmups", []),
                *(sample for group in record.get("rounds", []) for sample in group),
            ]
            failed = any(sample.get("lifecycle") != "completed" for sample in samples)
            if failed:
                invalid.add((record["case"], record["side"]))
            output.append(
                {**record, "warmups": [], "rounds": [[], []]} if failed else record
            )
        return output

    primary = validated(report.get("measurements", []))
    control = report.get("retained_control", {})
    return {
        **report,
        "measurements": primary,
        **(
            {
                "retained_control": {
                    **control,
                    "measurements": validated(control.get("measurements", [])),
                }
            }
            if control
            else {}
        ),
    }, invalid


def _v3_outcomes(report, invalid):
    outcomes = {
        (record["case"], record["side"]): "attempted"
        for record in report.get("measurements", [])
        if record.get("warmups") or any(record.get("rounds", []))
    }
    failed = {
        (case, failure["side"])
        for failure in [
            *report.get("failed_arms", []),
            *report.get("failed_attempts", []),
        ]
        for case in failure["cases"]
    }
    return {**outcomes, **{key: "failed" for key in failed | invalid}}


def _v3_arm_state(case, side, outcomes):
    arm = case[side]
    if arm["status"] == "unsupported":
        return {**arm, "attempt_status": "unsupported", "validity": "unsupported"}
    attempt = outcomes.get((case["case"], side), "unattempted")
    valid = arm["complete"] and attempt != "failed"
    return {
        **arm,
        "attempt_status": attempt,
        "validity": "valid" if valid else "incomplete",
    }


def render_v3(summary: dict) -> str:
    """Expose acquisition and per-arm outcomes alongside the maintained eight rows."""
    from scripts.benchmark_suite.issue406_report import render_acceptance_markdown

    states = [
        [case["case"], side, case[side]["attempt_status"], case[side]["validity"]]
        for case in summary["cases"]
        for side in ("baseline", "candidate")
    ]
    return (
        render_acceptance_markdown(summary)
        + "\n"
        + f"Acquisition: {summary['acquisition_status']}; "
        + f"acceptance: {summary['acceptance_status']}.\n\n"
        + table(["Case", "Arm", "Attempt", "Evidence"], states)
        + "\n"
    )


def _v3_acceptance_status(report, complete):
    assessment = report.get("assessment", {})
    valid = (
        complete
        and report.get("state") == "complete"
        and report.get("budget_satisfied") is True
        and assessment.get("environment_suitable") is True
        and assessment.get("compatibility_verified") is True
    )
    return "eligible-for-review" if valid else "inconclusive"


def aborted_v3_summary(report):
    """Keep the fixed inventory explicit when no analysis time remains."""
    from dataclasses import asdict

    from scripts.benchmark_suite.issue406_acceptance import acceptance_inventory
    from scripts.benchmark_suite.issue406_report import TITLES

    outcomes = _v3_outcomes(report, set())
    cases = []
    for case in acceptance_inventory():
        arms = {
            side: {
                "status": "incomplete" if side in case.sides else "unsupported",
                "complete": False,
                "statistics": None,
            }
            for side in ("baseline", "candidate")
        }
        row = {**asdict(case), "case": case.id, **arms}
        cases.append(
            {**row, **{side: _v3_arm_state(row, side, outcomes) for side in arms}}
        )
    return {
        "schema": "calc-flow.issue406-summary-aborted.v3",
        "evidence_complete": False,
        "acquisition_status": "incomplete",
        "acceptance_status": "inconclusive",
        "failed_attempts": report.get("failed_attempts", []),
        "reason": report["error"],
        "cases": cases,
        "targets": [
            {"row": row, "title": title, "status": "incomplete"}
            for row, title in TITLES.items()
        ],
    }
