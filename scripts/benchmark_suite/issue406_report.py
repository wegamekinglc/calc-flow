"""Eight-target acceptance summaries without inventing cross-group timing pairs."""

from __future__ import annotations

import math
import statistics
from collections import Counter
from dataclasses import asdict

from scripts.benchmark_suite.issue406_acceptance import (
    acceptance_inventory,
    matched_native_identity,
    matched_overhead,
    validate_complete_results,
    validate_observation,
)
from scripts.benchmark_suite.report import ROUNDS, SAMPLES, comparison, table
from scripts.benchmark_suite.statistics import _median_ranks
from scripts.toolkit import linear_percentile

PHASES = (
    "setup_seconds",
    "oracle_seconds",
    "dispatch_trace_setup_seconds",
    "dispatch_trace_capture_seconds",
    "processing_seconds",
    "settled_processing_seconds",
    "preparation_seconds",
    "capture_seconds",
    "restore_seconds",
    "handler_seconds",
    "combined_seconds",
    "startup_seconds",
    "terminal_seconds",
    "cleanup_seconds",
    "full_cycle_seconds",
    "job_seconds",
    "job_cpu_seconds",
    "sample_wall_seconds",
)
TITLES = {
    1: "Sealed-dimension Join, 1M quotes",
    2: "Sealed-dimension Join, 10M quotes",
    3: "Fully retained Join, 1M to 4M scaling",
    4: "Join snapshot and restore, 1M resident rows",
    5: "500-row Join handler after compaction, 1M resident rows",
    6: "ASOF, 1M rows per side",
    7: "ASOF instrumented harness overhead, 1M per side / 1024-row messages",
    8: "Join callback, 1M quotes / 1024-row messages",
}


def _finite(value) -> bool:
    return type(value) in (int, float) and math.isfinite(value)


def _stats(values: list[float]) -> dict | None:
    if not values or not all(_finite(value) for value in values):
        return None
    ordered = sorted(values)
    interval = None
    try:
        rank, coverage = _median_ranks(len(values))
        interval = {
            "low": ordered[rank - 1],
            "high": ordered[-rank],
            "coverage": coverage,
        }
    except ValueError:
        pass
    return {
        "samples": len(values),
        "p50": statistics.median(values),
        "p95": linear_percentile(values, 0.95),
        "median_interval": interval,
    }


def _samples(record: dict | None) -> list[dict]:
    if record is None:
        return []
    return [sample for group in record.get("rounds", []) for sample in group]


def _arm(case, side: str, record: dict | None) -> dict:
    if side not in case.sides:
        return {"status": "unsupported", "statistics": None, "complete": False}
    samples = _samples(record)
    values = [sample.get(case.metric) for sample in samples]
    complete = (
        record is not None
        and len(record.get("rounds", [])) == ROUNDS
        and all(len(group) == SAMPLES for group in record["rounds"])
    )
    problems = []
    for sample in samples:
        try:
            validate_observation(case, side, sample)
        except ValueError as error:
            problems.append(str(error))
    lifecycle = Counter(sample.get("lifecycle", "missing") for sample in samples)
    status = (
        "unmeasured"
        if not samples
        else "incomplete"
        if not complete
        else "invalid"
        if problems
        else "diagnostic"
        if lifecycle.get("failed")
        else "ok"
    )
    phases = {
        field: _stats([sample[field] for sample in samples])
        for field in PHASES
        if samples and all(_finite(sample.get(field)) for sample in samples)
    }
    return {
        "status": status,
        "complete": complete and not problems,
        "statistics": _stats(values),
        "metric": case.metric,
        "lifecycle_counts": dict(lifecycle),
        "validation_errors": sorted(set(problems)),
        "phases": phases,
        "advancing_epoch_samples": sum(
            bool(sample.get("advancing_processing_epochs")) for sample in samples
        ),
        "measured_samples": len(samples),
        "checkpoint_bytes": _stats(
            [sample["checkpoint_evidence"]["state_bytes"] for sample in samples]
        )
        if samples
        and all(
            _finite(sample.get("checkpoint_evidence", {}).get("state_bytes"))
            for sample in samples
        )
        else None,
    }


def _revision(case, base: dict, head: dict, records: dict, assessment: dict) -> dict:
    if "baseline" not in case.sides:
        return {"verdict": "candidate-only", "round_intervals": []}
    if not base["complete"] or not head["complete"]:
        return {"verdict": "unmeasured-or-invalid", "round_intervals": []}
    values = {
        side: [
            [sample[case.metric] for sample in group]
            for group in records[case.id, side]["rounds"]
        ]
        for side in ("baseline", "candidate")
    }
    result = comparison(
        {
            "id": case.id,
            "rows": case.rows,
            "correctness": True,
            "comparison": "interleaved",
            **values,
        }
    )
    maintained = result["verdict"]
    if base["status"] != "ok" or head["status"] != "ok":
        verdict = "diagnostic"
    elif not assessment.get("compatibility_verified") or not assessment.get(
        "environment_suitable"
    ):
        verdict = "inconclusive"
    else:
        verdict = maintained
    return {
        "verdict": verdict,
        "maintained_verdict": maintained,
        "change_percent": result["change_percent"],
        "round_intervals": result["round_intervals"],
    }


def _find(
    cases: list[dict], row: int, workload: str, mode: str, *, shape=(1_000_000, 1)
) -> dict:
    return next(
        case
        for case in cases
        if case["row"] == row
        and case["workload"] == workload
        and case["mode"] == mode
        and (case["rows"], case["threads"]) == shape
    )


def _scaled(stats: dict | None, factor: float) -> dict | None:
    if stats is None:
        return None
    interval = stats["median_interval"]
    return {
        **stats,
        "p50": stats["p50"] * factor,
        "p95": stats["p95"] * factor,
        "median_interval": None
        if interval is None
        else {
            **interval,
            "low": interval["low"] * factor,
            "high": interval["high"] * factor,
        },
    }


def _target_status(value: float | None, limit: float, *, upper=True) -> str:
    if value is None:
        return "unmeasured"
    within = value <= limit if upper else value >= limit
    return (
        "observed-within-target"
        if within
        else "observed-above-target"
        if upper
        else "observed-below-target"
    )


def _absolute(case: dict, limit: float, *, factor=1.0, unit="seconds") -> dict:
    candidate = case["candidate"]
    stats = _scaled(candidate["statistics"], factor)
    value = stats["p50"] if stats is not None and candidate["status"] == "ok" else None
    return {
        "case": case["case"],
        "mode": case["mode"],
        "threads": case["threads"],
        "metric": case["workload"],
        "unit": unit,
        "target": limit,
        "candidate": stats,
        "denominator_rows": case["rows"],
        "baseline": _scaled(case["baseline"]["statistics"], factor),
        "target_status": _target_status(value, limit),
    }


def _mode_effect(off: dict, enabled: dict, mode: str) -> dict:
    left, right = off["candidate"], enabled["candidate"]
    available = left["status"] == right["status"] == "ok"
    ratio = (
        left["statistics"]["p50"] / right["statistics"]["p50"] if available else None
    )
    limit = 0.9 if mode == "low" else 0.75
    coverage = right.get("advancing_epoch_samples", 0)
    periodic = (
        "not-applicable"
        if mode == "low"
        else "observed"
        if right["complete"] and coverage == ROUNDS * SAMPLES
        else "mixed-coverage"
        if coverage
        else "unmeasured"
    )
    return {
        "mode": mode,
        "threads": enabled["threads"],
        "rows": enabled["rows"],
        "throughput_ratio": ratio,
        "target_ratio": limit,
        "configuration_target_status": _target_status(ratio, limit, upper=False),
        "periodic_impact": periodic,
        "advancing_epoch_samples": coverage,
        "off_processing": left["statistics"],
        "enabled_processing": right["statistics"],
        "phases": right.get("phases", {}),
        "checkpoint_bytes": right.get("checkpoint_bytes"),
        "uncertainty": (
            "Independent group median intervals; no paired mode confidence interval."
        ),
    }


def _runtime_target(
    cases: list[dict], row: int, workload: str, rows: int, limits: tuple
) -> dict:
    absolute, modes = [], []
    for threads, limit in limits:
        off = _find(cases, row, workload, "off", shape=(rows, threads))
        absolute.append(_absolute(off, limit))
        modes.extend(
            _mode_effect(
                off, _find(cases, row, workload, mode, shape=(rows, threads)), mode
            )
            for mode in ("low", "on")
        )
    main = [
        {
            "mode": mode,
            "threads": threads,
            **_find(cases, row, workload, mode, shape=(rows, threads))["baseline"],
        }
        for threads, _limit in limits
        for mode in ("low", "on")
    ]
    return {
        "row": row,
        "title": TITLES[row],
        "absolute": absolute,
        "mode_effects": modes,
        "fresh_main": main,
    }


def _scaling_target(cases: list[dict]) -> dict:
    scaling, effects = [], []
    for mode in ("off", "low", "on"):
        small, large = (
            _find(cases, 3, "join_retained", mode, shape=(rows, 1))
            for rows in (1_000_000, 4_000_000)
        )
        available = small["candidate"]["status"] == large["candidate"]["status"] == "ok"
        ratio = (
            large["candidate"]["statistics"]["p50"]
            / small["candidate"]["statistics"]["p50"]
            if available
            else None
        )
        base_valid = small["baseline"]["complete"] and large["baseline"]["complete"]
        base_ratio = (
            large["baseline"]["statistics"]["p50"]
            / small["baseline"]["statistics"]["p50"]
            if base_valid
            else None
        )
        baseline = {
            "ratio_of_p50": base_ratio,
            "one_million": small["baseline"]["statistics"],
            "four_million": large["baseline"]["statistics"],
            "status": "diagnostic"
            if base_valid
            and (
                small["baseline"]["status"] != "ok"
                or large["baseline"]["status"] != "ok"
            )
            else "ok"
            if base_valid
            else "unsupported"
            if mode == "off"
            else "unmeasured",
        }
        scaling.append(
            {
                "mode": mode,
                "baseline": baseline,
                "ratio_of_p50": ratio,
                "target": 4.4 if mode == "off" else None,
                "target_status": _target_status(ratio, 4.4)
                if mode == "off"
                else "informational",
                "one_million": small["candidate"]["statistics"],
                "four_million": large["candidate"]["statistics"],
                "uncertainty": (
                    "Separate size-group median intervals; "
                    "no paired scaling confidence interval."
                ),
            }
        )
        if mode != "off":
            effects.extend(
                _mode_effect(
                    _find(cases, 3, "join_retained", "off", shape=(rows, 1)),
                    _find(cases, 3, "join_retained", mode, shape=(rows, 1)),
                    mode,
                )
                for rows in (1_000_000, 4_000_000)
            )
    return {"row": 3, "title": TITLES[3], "scaling": scaling, "mode_effects": effects}


def _compaction_target(cases: list[dict]) -> dict:
    armed = _find(cases, 5, "join_handler_compacting", "enabled")
    steady = _find(cases, 5, "join_handler_steady", "enabled")
    valid = armed["candidate"]["status"] == steady["candidate"]["status"] == "ok"
    ratio = (
        armed["candidate"]["statistics"]["p50"]
        / steady["candidate"]["statistics"]["p50"]
        if valid
        else None
    )
    base_valid = armed["baseline"]["complete"] and steady["baseline"]["complete"]
    base_ratio = (
        armed["baseline"]["statistics"]["p50"] / steady["baseline"]["statistics"]["p50"]
        if base_valid
        else None
    )
    baseline = {
        "ratio_of_p50": base_ratio,
        "armed": armed["baseline"],
        "steady": steady["baseline"],
        "status": "diagnostic"
        if base_valid
        and (
            armed["baseline"]["status"] != "ok" or steady["baseline"]["status"] != "ok"
        )
        else "ok"
        if base_valid
        else "unmeasured",
    }
    return {
        "row": 5,
        "title": TITLES[5],
        "mode": "intrinsically-enabled",
        "baseline": baseline,
        "off": "not-applicable: compaction is absent",
        "ratio_of_p50": ratio,
        "target": 2.0,
        "target_status": _target_status(ratio, 2.0),
        "armed": armed["candidate"],
        "steady": steady["candidate"],
        "uncertainty": (
            "Armed/steady phases share each physical invocation; the displayed "
            "target is the ratio of their group medians. Revision AB/BA is a "
            "separate comparison."
        ),
    }


def _overhead(
    cases: list[dict], records: dict, report: dict, side: str, mode: str
) -> dict:
    components = {
        name: _find(cases, 7, name, mode)
        for name in ("asof", "asof_operator", "projection")
    }
    if side == "baseline" and mode == "off":
        return {"side": side, "mode": mode, "target_status": "unsupported"}
    arms = {name: case[side] for name, case in components.items()}
    if any(not arm["complete"] for arm in arms.values()):
        return {"side": side, "mode": mode, "target_status": "unmeasured"}
    proof = report.get("scheduling_proofs", {}).get(f"{side}/{mode}")
    errors = []
    # Match source references by their recorded location, including warmups.
    # These are semantic replay links; timing aggregation remains independent.
    try:
        normalized = _normalize_replays(components, records, side, mode)
        native_reference = normalized["asof_operator"][ROUNDS]
        e2e_reference = normalized["asof"][ROUNDS]
        projection = arms["projection"]["statistics"]["p50"]
        if len({item["observation_id"] for item in normalized["asof"]}) != 22:
            raise ValueError("ASOF replay reused a source observation")
        for source, replay in zip(
            normalized["asof"], normalized["asof_operator"], strict=True
        ):
            if source["match_identity"] != e2e_reference["match_identity"]:
                raise ValueError("ASOF workload identity changed across observations")
            matched = matched_overhead(
                source, replay, projection, scheduling_proof=proof
            )
            if matched["schedule_status"] != "recorded-dispatch-chain-replayed":
                raise ValueError("ASOF observation lacks its verified recorded replay")
        e2e = {**e2e_reference, "seconds": arms["asof"]["statistics"]["p50"]}
        native = {
            **native_reference,
            "seconds": arms["asof_operator"]["statistics"]["p50"],
        }
        derived = matched_overhead(e2e, native, projection, scheduling_proof=proof)
        derived["semantic_replays_validated"] = len(normalized["asof"])
    except (KeyError, ValueError) as error:
        errors.append(str(error))
        derived = {"target_status": "inconclusive"}
    if any(arm["status"] != "ok" for arm in arms.values()):
        derived["target_status"] = "diagnostic"
    return {
        "side": side,
        "mode": mode,
        **derived,
        "components": {name: arm["statistics"] for name, arm in arms.items()},
        "errors": errors,
        "uncertainty": (
            "Signed difference of independent component p50 values; component "
            "median intervals are retained separately, with no paired residual CI."
        ),
    }


def _overhead_target(cases: list[dict], records: dict, report: dict) -> dict:
    off = _find(cases, 7, "asof", "off")
    return {
        "row": 7,
        "title": TITLES[7],
        "overhead": [
            _overhead(cases, records, report, side, mode)
            for side in ("baseline", "candidate")
            for mode in ("off", "low")
        ],
        "mode_effects": [
            _mode_effect(off, _find(cases, 7, "asof", mode), mode)
            for mode in ("low", "on")
        ],
        "on_residual": "not-applicable: checkpoint work is not harness overhead",
    }


def summarize_acceptance(report: dict) -> dict:
    """Return every inventory row, including partial and unsupported evidence."""
    from scripts.benchmark_suite.issue406_fresh_report import summarize_retained_control

    inventory = acceptance_inventory()
    known = {case.id: case for case in inventory}
    records = {}
    for record in report.get("measurements", []):
        key = (record.get("case"), record.get("side"))
        if key[0] not in known or key[1] not in known[key[0]].sides or key in records:
            raise ValueError("unexpected or duplicate acceptance result")
        records[key] = record
    assessment = {**report.get("assessment", {})}
    if report.get("settlement", {}).get("forced_kill_pids"):
        assessment["environment_suitable"] = False
    cases = []
    for case in inventory:
        base = _arm(case, "baseline", records.get((case.id, "baseline")))
        head = _arm(case, "candidate", records.get((case.id, "candidate")))
        cases.append(
            {
                **asdict(case),
                "case": case.id,
                "baseline": base,
                "candidate": head,
                "revision_comparison": _revision(case, base, head, records, assessment),
            }
        )
    targets = [
        _runtime_target(cases, 1, "join_sealed", 1_000_000, ((1, 0.060), (32, 0.015))),
        _runtime_target(cases, 2, "join_sealed", 10_000_000, ((1, 0.600), (32, 0.150))),
        _scaling_target(cases),
        {
            "row": 4,
            "title": TITLES[4],
            "mode": "intrinsically-enabled",
            "absolute": [
                _absolute(
                    _find(cases, 4, name, "enabled"),
                    limit,
                    factor=1_000_000 / _find(cases, 4, name, "enabled")["rows"],
                    unit="microseconds/row",
                )
                for name, limit in (("join_capture", 0.1), ("join_restore", 0.2))
            ],
        },
        _compaction_target(cases),
        _runtime_target(cases, 6, "asof", 1_000_000, ((1, 0.100), (32, 0.030))),
        _overhead_target(cases, records, report),
        {
            "row": 8,
            "title": TITLES[8],
            "absolute": [
                _absolute(
                    _find(cases, 8, "join_small_batch", mode),
                    50.0,
                    factor=1_000_000,
                    unit="microseconds/full-batch callback",
                )
                for mode in ("off", "enabled")
            ],
            "low_on": (
                "Native contexts have no periodic interval; "
                "enabled context is reported separately."
            ),
        },
    ]
    validation_errors = []
    try:
        validate_complete_results(report)
    except ValueError as error:
        validation_errors.append(str(error))
    complete = not validation_errors and all(
        case[side]["complete"] for case in cases for side in known[case["case"]].sides
    )
    return {
        "schema": "calc-flow.issue406-summary.v1",
        "evidence_complete": complete,
        "assessment": assessment,
        "targets": targets,
        "cases": cases,
        "evidence_validation_errors": validation_errors,
        "run_context": {
            key: report.get(key)
            for key in (
                "state",
                "plan_fingerprint",
                "budget_satisfied",
                "elapsed_seconds",
                "elapsed_seconds_including_sealing",
            )
        },
        "statistical_scope": (
            "Only same-configuration baseline/candidate AB/BA observations use "
            "paired confidence intervals. Absolute median intervals use the "
            "maintained exact order-statistic construction and assume iid "
            "observations. Cross-mode, cross-size and cross-component metrics "
            "use group medians without pseudo-pairing."
        ),
        "historical_values": (
            "Historical issue figures identify targets only; every displayed "
            "current/base statistic comes from this report."
        ),
        **(
            {"retained_control": summarize_retained_control(report)}
            if "retained_control" in report
            else {}
        ),
    }


def _number(value: float | None, factor=1.0) -> str:
    return "—" if value is None else f"{value * factor:.6g}"


def _stat_cell(stats: dict | None, factor=1.0) -> str:
    if stats is None:
        return "unmeasured"
    interval = stats["median_interval"]
    bounds = (
        ""
        if interval is None
        else (
            f" [{_number(interval['low'], factor)}, "
            f"{_number(interval['high'], factor)}]"
        )
    )
    return (
        f"p50 {_number(stats['p50'], factor)}, "
        f"p95 {_number(stats['p95'], factor)}{bounds}"
    )


def _target_cell(target: dict) -> str:
    row = target["row"]
    if "absolute" in target:
        return "; ".join(
            f"{item['mode']}/{item['threads']}CPU {item['metric']}: "
            f"{_stat_cell(item['candidate'])} {item['unit']} "
            f"(≤{item['target']}: {item['target_status']})"
            for item in target["absolute"]
        )
    if row == 3:
        item = target["scaling"][0]
        return (
            f"OFF ratio of p50 {_number(item['ratio_of_p50'])} "
            f"(≤4.4: {item['target_status']})"
        )
    if row == 5:
        return (
            f"Enabled ratio of p50 {_number(target['ratio_of_p50'])} "
            f"(≤2: {target['target_status']})"
        )
    return "; ".join(
        f"{item['side']}/{item['mode']}: "
        f"H={_number(item.get('overhead_seconds'), 1000)} ms, "
        f"2×floor={_number(item.get('target_seconds'), 1000)} ms "
        f"({item['target_status']})"
        for item in target["overhead"]
        if item["side"] == "candidate"
    )


def _main_cell(target: dict) -> str:
    if "fresh_main" in target:
        return "; ".join(
            f"{item['mode']}/{item['threads']}CPU: "
            f"{_stat_cell(item['statistics'])} s ({item['status']})"
            for item in target["fresh_main"]
        )
    if target["row"] == 3:
        return "; ".join(
            f"{item['mode']}: ratio of p50 "
            f"{_number(item['baseline']['ratio_of_p50'])} "
            f"({item['baseline']['status']})"
            for item in target["scaling"]
            if item["mode"] != "off"
        )
    if target["row"] == 5:
        baseline = target["baseline"]
        return (
            f"Enabled ratio of p50 {_number(baseline['ratio_of_p50'])} "
            f"({baseline['status']}); "
            f"armed {_stat_cell(baseline['armed']['statistics'])} s; "
            f"steady {_stat_cell(baseline['steady']['statistics'])} s"
        )
    if "absolute" in target:
        return "; ".join(
            f"{item['mode']}/{item['metric']}: "
            f"{_stat_cell(item['baseline'])} {item['unit']}"
            for item in target["absolute"]
            if item["mode"] != "off"
        )
    return "; ".join(
        f"{item['mode']}: H={_number(item.get('overhead_seconds'), 1000)} ms, "
        f"2×floor={_number(item.get('target_seconds'), 1000)} ms "
        f"({item['target_status']})"
        for item in target["overhead"]
        if item["side"] == "baseline" and item["mode"] != "off"
    )


def render_acceptance_markdown(summary: dict) -> str:
    """Render all eight targets and every supported main/candidate comparison."""
    parts = [
        "# Join and ASOF acceptance",
        "",
        "Complete observations."
        if summary["evidence_complete"]
        else (
            "Incomplete observations; missing controls and target rows remain explicit."
        ),
        "",
        f"Run state: {summary['run_context'].get('state')}; "
        f"budget satisfied: {summary['run_context'].get('budget_satisfied')}; "
        "environment suitable: "
        f"{summary['assessment'].get('environment_suitable')}; "
        "compatibility verified: "
        f"{summary['assessment'].get('compatibility_verified')}; "
        f"machine: {summary['assessment'].get('machine_type', 'unrecorded')}. "
        "Observed target values are descriptive; an unsuitable environment or "
        "incomplete evidence cannot establish acceptance.",
        "",
        summary["statistical_scope"],
        "",
        table(
            [
                "Row",
                "Workload",
                "Fresh main (explicit policy)",
                "Candidate target observation",
            ],
            [
                [
                    target["row"],
                    target["title"],
                    _main_cell(target),
                    _target_cell(target),
                ]
                for target in summary["targets"]
            ],
        ),
        "",
        "Main has no checkpoint-off arm. Its low/on values have their policies "
        "labeled and are not revision pairs with candidate off. Checkpoint mode "
        "comparisons use ratios of independently collected p50 values. ON without "
        "advancing processing epochs measures configuration overhead; periodic "
        "impact remains unmeasured.",
        "",
        "Row 7 reports an instrumented matched-component residual: end-to-end "
        "processing extends through final output projection watermark completion, "
        "and native replay follows each recorded ASOF dispatch chain. Recorder "
        "hook costs remain in both clocks. The original ready-to-Arrow clock, "
        "recorder setup and trace capture costs remain separate. Replay links "
        "establish semantic matching; the residual uses independent group "
        "medians and does not claim identical global task scheduling or causal "
        "CPU attribution.",
        "",
    ]
    modes = [
        [
            target["row"],
            f"N={item['rows']}/{item['threads']}CPU/{item['mode']}",
            _number(item["throughput_ratio"]),
            f"≥{item['target_ratio']}",
            item["configuration_target_status"],
            f"{item['periodic_impact']} "
            f"({item['advancing_epoch_samples']}/20 observations)",
            _stat_cell(item["phases"].get("terminal_seconds"), 1000),
            _stat_cell(item["phases"].get("full_cycle_seconds"), 1000),
            _stat_cell(item["checkpoint_bytes"]),
        ]
        for target in summary["targets"]
        for item in target.get("mode_effects", [])
    ]
    parts.extend(
        [
            table(
                [
                    "Row",
                    "Mode",
                    "OFF-relative throughput",
                    "Goal",
                    "Observed configuration",
                    "Periodic impact",
                    "EOF p50/p95 [interval], ms",
                    "Full cycle p50/p95 [interval], ms",
                    "Checkpoint bytes p50/p95 [interval]",
                ],
                modes,
            ),
            "",
            "EOF includes terminal completion; full cycle runs from processing "
            "start through owned cleanup. [Machine-readable phase costs]"
            "(summary.json) retain main/off/native setup, oracle, preparation, "
            "processing, terminal, cleanup and CPU clocks where collected. "
            "[Raw observations](result.json), [process resource samples]"
            "(resources.json) and [final budget ledger](budget.json) retain "
            "their separate evidence.",
            "",
        ]
    )
    comparisons = []
    for case in summary["cases"]:
        revision = case["revision_comparison"]
        intervals = (
            "; ".join(
                f"{item['median']:+.3f}% [{item['low']:+.3f}, {item['high']:+.3f}]"
                for item in revision["round_intervals"]
            )
            or "—"
        )
        comparisons.append(
            [
                case["case"],
                _stat_cell(case["baseline"]["statistics"]),
                case["baseline"]["status"],
                _stat_cell(case["candidate"]["statistics"]),
                case["candidate"]["status"],
                intervals,
                revision["verdict"],
            ]
        )
    parts.extend(
        [
            "All timing statistics below are seconds; native checkpoint metrics "
            "retain their complete measured boundary. Invalid baseline lifecycle "
            "results are diagnostic.",
            "",
            table(
                [
                    "Case",
                    "Main p50/p95 [median interval]",
                    "Main state",
                    "Candidate p50/p95 [median interval]",
                    "Candidate state",
                    "Revision paired change [CI]",
                    "Verdict",
                ],
                comparisons,
            ),
            "",
        ]
    )
    if "retained_control" in summary:
        from scripts.benchmark_suite.issue406_fresh_report import (
            render_retained_control,
        )

        parts.extend(render_retained_control(summary["retained_control"]))
    return "\n".join(parts)


def _normalize_replays(components: dict, records: dict, side: str, mode: str) -> dict:
    normalized = {"asof": [], "asof_operator": []}
    for name in normalized:
        record = records[components[name]["case"], side]
        if len(record.get("warmups", [])) != ROUNDS:
            raise ValueError("ASOF replay requires both recorded warmups")
        for sample in [*record["warmups"], *_samples(record)]:
            item = {
                **sample,
                "seconds": sample["settled_processing_seconds"]
                if name == "asof"
                else sample["processing_seconds"],
            }
            if name == "asof_operator" and "match_identity" not in item:
                item["match_identity"] = matched_native_identity(
                    item, mode, item["cpu_affinity"]
                )
            normalized[name].append(item)
    return normalized
