"""Current B/C summaries with absolute mode goals and separate planning targets."""

from __future__ import annotations

from scripts.benchmark_suite import issue406_milestone as policy
from scripts.benchmark_suite import issue406_report as report_tools
from scripts.benchmark_suite import issue406_sources as sources
from scripts.benchmark_suite.report import ROUNDS, SAMPLES, table

FORMAL = {
    1: ((1, 0.060), (32, 0.015)),
    2: ((1, 0.600), (32, 0.150)),
    6: ((1, 0.100), (32, 0.030)),
}
PLANNING = {
    1: ((1, 0.025), (32, 0.012)),
    2: ((1, 0.250), (32, 0.130)),
    6: ((1, 0.040), (32, 0.030)),
    9: ((1, 0.400),),
    10: ((1, 0.300), (32, 0.150)),
}


def mode_target(arm: dict, mode: str, off_goal: float) -> dict:
    """Derive LOW/ON limits from the goal, never from achieved OFF performance."""
    limit = off_goal / {"off": 1.0, "low": 0.90, "on": 0.75}[mode]
    value = arm["statistics"]["p50"] if arm["status"] == "ok" else None
    status = report_tools._target_status(value, limit)
    periodic = mode != "on" or arm.get("advancing_epoch_samples") == ROUNDS * SAMPLES
    return {
        "mode": mode,
        "off_goal_seconds": off_goal,
        "limit_seconds": limit,
        "configuration_target_status": status,
        "target_status": status if periodic else "unmeasured",
        "advancing_epoch_samples": arm.get("advancing_epoch_samples", 0),
        "periodic_coverage": "not-applicable"
        if mode != "on"
        else "observed"
        if periodic
        else "unmeasured",
        "statistics": arm["statistics"],
    }


def _sample_errors(case, side, record, report):
    errors = []
    samples = [*record.get("warmups", []), *report_tools._samples(record)]
    for sample in samples:
        try:
            policy.validate_sample(case, side, sample, report)
        except (ValueError, KeyError, TypeError) as error:
            errors.append(str(error))
    return errors


def _validated_status(status, errors, complete, cpu):
    if errors:
        return "invalid"
    if not complete:
        return "incomplete"
    if not cpu:
        return "inconclusive"
    return status


def _validated_arm(case, side, arm, record, report):
    if record is None:
        return arm
    errors = [*arm["validation_errors"], *_sample_errors(case, side, record, report)]
    complete = arm["complete"] and len(record.get("warmups", [])) == ROUNDS
    cpu = arm.get("cpu_evidence_status") == {"verified": ROUNDS * SAMPLES}
    return {
        **arm,
        "validation_errors": sorted(set(errors)),
        "complete": complete and not errors,
        "status": _validated_status(arm["status"], errors, complete, cpu),
    }


def _absolute_modes(cases, goals):
    return [
        {
            "case": case["case"],
            "side": side,
            "rows": case["rows"],
            "threads": case["threads"],
            **mode_target(case[side], case["mode"], goal),
        }
        for case in cases
        if case["row"] in goals
        for threads, goal in goals[case["row"]]
        if case["threads"] == threads
        for side in policy.SIDES
    ]


def _overhead(cases, records, report):
    result = []
    for side in policy.SIDES:
        for mode in ("off", "low"):
            components = {
                name: report_tools._find(cases, 7, name, mode)
                for name in ("asof", "asof_operator", "projection")
            }
            arms = {name: case[side] for name, case in components.items()}
            result.append(
                report_tools._measured_overhead(
                    components, records, report, (side, mode), arms
                )
                if all(arm["status"] == "ok" for arm in arms.values())
                else {"side": side, "mode": mode, "target_status": "unmeasured"}
            )
    return result


def _absolute_target(row, cases, modes):
    return {
        "mode_effects": [
            item for item in modes if item["case"].startswith(f"row-{row}/")
        ],
        "fresh_main": [
            {"mode": case["mode"], "threads": case["threads"], **case["baseline"]}
            for case in cases
            if case["row"] == row
        ],
    }


def _formal_target(target, cases, records, report, modes):
    row = target["row"]
    # Historical mode ratios remain available only in historical summaries.
    current = {key: value for key, value in target.items() if key != "mode_effects"}
    if row in FORMAL:
        current.update(_absolute_target(row, cases, modes))
    if row == 3:
        current["scaling"] = _scaling(cases)
    if row == 7:
        current["overhead"] = _overhead(cases, records, report)
    return current


def _formal_targets(cases, records, report):
    targets = report_tools._acceptance_targets(cases, records, report)
    modes = _absolute_modes(cases, FORMAL)
    return [_formal_target(target, cases, records, report, modes) for target in targets]


def _scaling(cases):
    result = []
    for mode in ("off", "low", "on"):
        small = report_tools._find(
            cases, 3, "join_retained", mode, shape=(1_000_000, 1)
        )
        large = report_tools._find(
            cases, 3, "join_retained", mode, shape=(4_000_000, 1)
        )
        for side in policy.SIDES:
            valid = small[side]["status"] == large[side]["status"] == "ok"
            ratio = (
                large[side]["statistics"]["p50"] / small[side]["statistics"]["p50"]
                if valid
                else None
            )
            result.append(
                {
                    "side": side,
                    "mode": mode,
                    "ratio_of_p50": ratio,
                    "target": 4.4 if mode == "off" else None,
                    "target_status": report_tools._target_status(ratio, 4.4)
                    if mode == "off"
                    else "diagnostic",
                    "uncertainty": (
                        "Separate size-group medians; no paired scaling interval."
                    ),
                }
            )
    return result


def _runtime_batch_values(samples):
    return [
        item["processing_seconds"] / item["runtime_input_data_batches"]
        for item in samples
        if type(item.get("runtime_input_data_batches")) is int
        and item["runtime_input_data_batches"] > 0
    ]


def _runtime_floor_arm(case, side, record):
    samples = report_tools._samples(record)
    values = _runtime_batch_values(samples)
    arm = {**case[side], "statistics": report_tools._stats(values)}
    if len(values) != ROUNDS * SAMPLES:
        arm = {**arm, "status": "incomplete"}
    return {
        "case": case["case"],
        "side": side,
        "unit": "seconds/actual-input-data-batch",
        "batch_counts": [item.get("runtime_input_data_batches") for item in samples],
        **mode_target(arm, case["mode"], 0.0001),
    }


def _runtime_floor(report, cases):
    records = {
        (record["case"], record["side"]): record
        for record in report.get("measurements", [])
    }
    return [
        _runtime_floor_arm(case, side, records.get((case["case"], side)))
        for case in cases
        if case["row"] == 7 and case["workload"] == "projection"
        for side in policy.SIDES
    ]


def summarize(report: dict) -> dict:
    """Retain all 94 arms and all formal/planning goals even after an early stop."""
    summary = report_tools.summarize_acceptance(
        report,
        inventory=policy.inventory(),
        validator=policy.validate_results,
        target_builder=_formal_targets,
        arm_validator=_validated_arm,
    )
    return {
        **summary,
        **sources.report_fields(report),
        "schema": "calc-flow.issue406-milestone-summary.v1",
        **{
            key: report.get(key)
            for key in (
                "baseline_product_commit",
                "candidate_product_commit",
                "harness_git_sha",
                "build_harness_git_sha",
                "execution_git_sha",
                "preparation_anchor",
            )
        },
        "planning_targets": _absolute_modes(summary["cases"], PLANNING),
        "runtime_floor": _runtime_floor(report, summary["cases"]),
        "acceptance": "inconclusive",
        "acceptance_reason": (
            "During-compaction handler remains unmeasured; other unmet/missing "
            "targets are retained independently."
        ),
        "invocation_inventory": {
            "semantic_cases": 47,
            "physical_groups": 45,
            "semantic_arms": 94,
            "physical_arms": 90,
            "measured_calls": 1800,
            "warmup_calls": 180,
        },
    }


def aborted_summary(report):
    """On deadline exhaustion retain every row without computing late statistics."""
    from dataclasses import asdict

    cases = [
        {
            **asdict(case),
            "case": case.id,
            **{
                side: {"status": "incomplete", "statistics": None}
                for side in policy.SIDES
            },
            "revision_comparison": {"verdict": "unmeasured-or-invalid"},
        }
        for case in policy.inventory()
    ]
    return {
        "schema": "calc-flow.issue406-milestone-summary.v1",
        "evidence_complete": False,
        **sources.report_fields(report),
        **{
            key: report.get(key)
            for key in (
                "baseline_product_commit",
                "candidate_product_commit",
                "harness_git_sha",
                "build_harness_git_sha",
                "execution_git_sha",
                "preparation_anchor",
            )
        },
        "acceptance": "inconclusive",
        "reason": report.get("error"),
        "targets": [
            {"row": row, "title": title, "status": "unmeasured"}
            for row, title in report_tools.TITLES.items()
        ],
        "planning_targets": [
            {
                "row": row,
                "threads": threads,
                "off_goal_seconds": goal,
                "target_status": "unmeasured",
            }
            for row, limits in PLANNING.items()
            for threads, goal in limits
        ],
        "runtime_floor": [{"off_goal_seconds": 0.0001, "target_status": "unmeasured"}],
        "cases": cases,
    }


def _case_table(cases):
    return table(
        [
            "Case",
            "B evidence",
            "B p50/p95/p99 [median CI], ms",
            "C evidence",
            "C p50/p95/p99 [median CI], ms",
            "Two-round paired verdict",
        ],
        [
            [
                case["case"],
                case["baseline"]["status"],
                report_tools._stat_cell(case["baseline"]["statistics"], 1000),
                case["candidate"]["status"],
                report_tools._stat_cell(case["candidate"]["statistics"], 1000),
                case["revision_comparison"]["verdict"],
            ]
            for case in cases
        ],
    )


def _absolute_component_row(row, item, side, by_case):
    stats = item[side]
    value = stats["p50"] if stats is not None else None
    return [
        row,
        item["metric"],
        side,
        item["mode"],
        item["target"],
        value,
        item["unit"],
        item["target_status"]
        if side == "candidate"
        else report_tools._target_status(
            value if by_case[item["case"]][side]["status"] == "ok" else None,
            item["target"],
        ),
    ]


def _scaling_component_rows(target):
    return [
        [
            target["row"],
            "retained 1M to 4M",
            item["side"],
            item["mode"],
            item["target"],
            item["ratio_of_p50"],
            "ratio",
            item["target_status"],
        ]
        for item in target.get("scaling", [])
    ]


def _overhead_component_rows(target):
    return [
        [
            target["row"],
            "matched residual",
            item["side"],
            item["mode"],
            item.get("target_seconds"),
            item.get("overhead_seconds"),
            "seconds",
            item["target_status"],
        ]
        for item in target.get("overhead", [])
    ]


def _target_component_rows(target, by_case):
    row = target["row"]
    if row in (4, 8):
        return [
            _absolute_component_row(row, item, side, by_case)
            for item in target.get("absolute", [])
            for side in policy.SIDES
        ]
    if row == 3:
        return _scaling_component_rows(target)
    if row == 5:
        return [
            [
                row,
                "genuinely during compaction",
                "both",
                "enabled",
                2,
                None,
                "ratio",
                "unmeasured",
            ]
        ]
    if row == 7:
        return _overhead_component_rows(target)
    return []


def _component_table(summary):
    by_case = {case["case"]: case for case in summary["cases"]}
    return table(
        ["Row", "Metric", "Side", "Mode", "Limit", "Observed p50", "Unit", "Result"],
        [
            row
            for target in summary["targets"]
            for row in _target_component_rows(target, by_case)
        ],
    )


def _goal_section(label, entries):
    return [
        f"## {label}",
        "",
        table(
            ["Case", "Side", "Mode", "Limit, ms", "Target result"],
            [
                [
                    item.get("case", f"row-{item.get('row')}"),
                    item.get("side", "both"),
                    item.get("mode", "off"),
                    1000 * item.get("limit_seconds", item.get("off_goal_seconds", 0)),
                    item["target_status"],
                ]
                for item in entries
            ],
        ),
        "",
    ]


def render(summary):
    """Render current B/C arms without historical unsupported-mode claims."""
    lines = [
        "# Current milestone Join/ASOF acceptance",
        "",
        f"B `{summary.get('baseline_product_commit')}`; "
        f"C `{summary.get('candidate_product_commit')}`.",
        f"Build tool `{summary.get('build_harness_git_sha')}`; "
        f"execution tool `{summary.get('execution_git_sha')}`.",
        "",
        "Acceptance inconclusive. During-compaction target unmeasured; "
        "post-compaction, preparation, queue and combined clocks are diagnostics.",
        "",
        "Formal target rows: "
        + ", ".join(
            str(item["row"]) + " " + item["title"] for item in summary["targets"]
        ),
        "",
        "Both sealed revisions have OFF/LOW24h/ON100ms arms where applicable. "
        "LOW/ON limits derive from absolute goals; ON also needs advancing "
        "nonterminal epochs.",
        "",
        _case_table(summary["cases"]),
        "",
        "## Formal scaling and component targets",
        "",
        _component_table(summary),
        "",
    ]
    goals = [
        item for target in summary["targets"] for item in target.get("mode_effects", [])
    ]
    for label, entries in (
        ("Formal absolute mode goals", goals),
        ("Stricter planning goals", summary["planning_targets"]),
        ("Runtime planning floor per actual input batch", summary["runtime_floor"]),
    ):
        lines.extend(_goal_section(label, entries))
    lines.extend(
        [
            "Raw JSON retains phase clocks, CPU/thread evidence, checkpoint "
            "coverage, baseline OFF scaling, capture/restore and callback units, "
            "matched residual controls and validation errors. Cross-mode/size/"
            "component group medians are not temporal pairs. Required CI/90% "
            "coverage is separate from local acceptance. This tool preparation "
            "does not certify a performance target.",
            "",
        ]
    )
    return "\n".join(lines)
