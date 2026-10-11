"""The fixed fresh eight-row acquisition and its two retained-state controls."""

from __future__ import annotations

import asyncio
import math
import time
from dataclasses import asdict
from pathlib import Path

from scripts.benchmark_suite.issue406_acceptance import acceptance_inventory
from scripts.benchmark_suite.release_pairs import pair_order
from scripts.benchmark_suite.report import ROUNDS, SAMPLES
from scripts.toolkit import write_json

SCHEMA = "calc-flow.issue406-execution.v2"
MAIN = "588726c8bce0f1c57b90077064834c4a472ee5f2"
CANDIDATE = "ce77237f2e91e97651eb9d1ec203fe00f328678d"
PREVIOUS = "cffcf57be9c5cada667c6109156bcbfe5b084f27"


def retained_cases() -> tuple:
    return tuple(
        case for case in acceptance_inventory() if case.row == 3 and case.mode == "off"
    )


def validate_envelope(plan: dict) -> None:
    """Reject any expansion of the reviewed fresh acquisition."""
    expected = {
        "schema": SCHEMA,
        "budget_seconds": 5400,
        "settlement_seconds": 300,
        "collection_cutoff_seconds": 4800,
        "settlement_deadline_seconds": 5100,
        "startup_allowance_seconds": 5,
        "final_seal_allowance_seconds": 5,
        "row5_arm_seconds": 360,
        "row5_group_seconds": 720,
        "cases": [asdict(case) for case in acceptance_inventory()],
    }
    if any(plan.get(key) != value for key, value in expected.items()):
        raise ValueError("fresh acquisition differs from the reviewed envelope")
    precharged = plan.get("precharged_seconds", 0)
    if (
        type(precharged) not in (int, float)
        or not math.isfinite(precharged)
        or not 0 <= precharged < 4795
    ):
        raise ValueError("invalid fresh preparation precharge")
    _validate_product_revisions(plan)


def _validate_product_revisions(plan: dict) -> None:
    if (
        set(plan.get("releases", {})) != {"baseline", "candidate"}
        or plan["releases"]["baseline"].get("product_commit") != MAIN
        or plan["releases"]["candidate"].get("product_commit") != CANDIDATE
        or plan.get("retained_control", {}).get("product_commit") != PREVIOUS
    ):
        raise ValueError("fresh acquisition product revisions changed")


class Row5Budget:
    """Own cumulative occupied time for the two fixed compaction arms."""

    def __init__(self, plan: dict):
        self.started = time.monotonic()
        self.arm_seconds = plan["row5_arm_seconds"]
        self.group_seconds = plan["row5_group_seconds"]
        self.spent = {"baseline": 0.0, "candidate": 0.0}

    async def invoke(self, side: str, operation):
        started = time.monotonic()
        remaining = min(
            self.arm_seconds - self.spent[side],
            self.group_seconds - (started - self.started),
        )
        if remaining <= 0:
            raise TimeoutError("row-5 cumulative budget exhausted")
        try:
            async with asyncio.timeout(remaining):
                result = await operation()
            if time.monotonic() - started > remaining:
                raise TimeoutError("row-5 cumulative budget exhausted")
            return result
        finally:
            self.spent[side] += time.monotonic() - started


def _record(case, side: str) -> dict:
    return {"case": case.id, "side": side, "rounds": [[], []], "warmups": []}


def _reference(sample: dict) -> dict:
    return {key: sample[key] for key in ("observation_id", "raw_observation_sha256")}


async def collect_fresh(runner, output: Path) -> dict:
    """Collect new pairs once, retaining precisely recognized main failures."""
    from scripts.benchmark_suite.issue406_run import (
        _invoke_collection,
        _record_collection,
        collection_groups,
    )

    records = {
        (case.id, side): _record(case, side)
        for case in acceptance_inventory()
        for side in case.sides
    }
    previous = {
        (case.id, "previous"): _record(case, "previous") for case in retained_cases()
    }
    report = {
        "state": "running",
        "measurements": list(records.values()),
        "retained_control": {
            "product_commit": PREVIOUS,
            "measurements": list(previous.values()),
            "pairs": [],
        },
        "known_failures": [],
    }
    from scripts.benchmark_suite import issue406_fresh_v3 as v3

    fresh_v3 = runner.plan.get("schema") == v3.SCHEMA
    report.update(_v3_report_fields(runner.plan))
    exhausted = set()
    output.mkdir(parents=True, exist_ok=True)

    async def invoke(group, side, round_index, index, budget):
        key = (group[0].id, side)
        if key in exhausted:
            return None

        async def operation():
            try:
                result = await _invoke_collection(
                    runner, output, group, side, (round_index, index)
                )
                _require_fresh_lifecycle(runner.plan, result)
                return result
            except BaseException as error:
                _record_failed_attempt(report, group, (side, round_index, index), error)
                if not isinstance(error, Exception):
                    raise
                failure = await _recognized_failure(
                    runner, group, (side, round_index, index), error
                )
                if failure is None:
                    raise
                exhausted.add(key)
                report["failed_arms" if fresh_v3 else "known_failures"].append(
                    _exhausted_arm(group, side, round_index, index, failure)
                )
                return None

        result = await (
            operation() if budget is None else budget.invoke(side, operation)
        )
        if result is None:
            return None
        _record_collection(
            previous if side == "previous" else records,
            output,
            group,
            (side, round_index, index),
            result,
        )
        return result

    try:
        for group in collection_groups():
            _skip_failed_dependencies(runner.plan, group, exhausted, report)
            await _collect_group(runner, group, invoke, report)
            write_json(output / "partial.json", report)
        validate_retained_control(report)
        report["state"] = _collection_state(fresh_v3, exhausted)
        return report
    except BaseException as error:
        report.update(state="stopped", error=f"{type(error).__name__}: {error}")
        raise
    finally:
        write_json(output / "partial.json", report)


def _v3_report_fields(plan):
    from scripts.benchmark_suite.issue406_fresh_v3 import SCHEMA, report_identity

    return (
        {
            **report_identity(plan),
            "failed_attempts": [],
            "failed_arms": [],
            "dependent_unattempted": [],
        }
        if plan.get("schema") == SCHEMA
        else {}
    )


def _record_failed_attempt(report, group, position, error):
    if "failed_attempts" not in report:
        return
    side, round_index, index = position
    report["failed_attempts"].append(
        {
            "cases": [case.id for case in group],
            "side": side,
            "round": round_index,
            "index": index,
            "phase": "warmup" if index is None else "sample",
            "error": f"{type(error).__name__}: {error}",
        }
    )


async def _recognized_failure(runner, group, position, error):
    from scripts.benchmark_suite.issue406_fresh_v3 import SCHEMA

    side, round_index, index = position
    if not _first_baseline(side, round_index, index):
        return None
    recognize = (
        runner.known_fresh_failure
        if runner.plan.get("schema") == SCHEMA
        else runner.known_main_failure
    )
    return await recognize(group, side, round_index, index, error)


def _require_fresh_lifecycle(plan, result):
    from scripts.benchmark_suite.issue406_fresh_v3 import SCHEMA, require_completed

    if plan.get("schema") == SCHEMA:
        require_completed(result)


def _first_baseline(side, round_index, index):
    return (side, round_index, index) == ("baseline", 0, None)


def _skip_failed_dependencies(plan, group, exhausted, report):
    from scripts.benchmark_suite import issue406_fresh_v3 as v3

    if plan.get("schema") != v3.SCHEMA:
        return
    source = v3.replay_source(group[0])
    for side in group[0].sides:
        if (source, side) in exhausted:
            exhausted.add((group[0].id, side))
            report["dependent_unattempted"].append(
                v3.unattempted_dependency(group, side, source)
            )


def _collection_state(fresh_v3, exhausted):
    if not exhausted:
        return "collected"
    return "incomplete" if fresh_v3 else "collected-with-failures"


def _exhausted_arm(group, side, round_index, index, failure):
    return {
        "cases": [case.id for case in group],
        "side": side,
        "round": round_index,
        "index": index,
        "evidence": failure,
        "unattempted_slots": [
            [r, i]
            for r in range(ROUNDS)
            for i in (None, *range(SAMPLES))
            if (r, i) != (round_index, index)
        ],
    }


async def _collect_group(runner, group, invoke, report):
    sides = (
        ("previous", "candidate") if group[0] in retained_cases() else group[0].sides
    )
    budget = Row5Budget(runner.plan) if group[0].row == 5 else None
    for round_index in range(ROUNDS):
        try:
            await _collect_round(group, sides, (round_index, budget), invoke, report)
        finally:
            try:
                await _finish_round(runner, sides, budget)
            finally:
                if budget is not None:
                    report["row5_occupied_seconds"] = dict(budget.spent)


async def _finish_round(runner, sides, budget):
    for side in sides:
        cleanup = asyncio.create_task(runner.finish_round(side))
        runner.cleanup_tasks.add(cleanup)
        if budget is None:
            await asyncio.shield(cleanup)
        else:
            await budget.invoke(side, lambda task=cleanup: asyncio.shield(task))


async def _collect_round(group, sides, position, invoke, report):
    round_index, budget = position
    controlled = "previous" in sides
    for index in (None, *range(SAMPLES)):
        order = _collection_order(sides, controlled, index)
        samples = {}
        for side in order:
            samples[side] = await invoke(group, side, round_index, index, budget)
        if controlled:
            report["retained_control"]["pairs"].append(
                {
                    "case": group[0].id,
                    "round": round_index,
                    "index": index,
                    "order": list(order),
                    **{side: _reference(samples[side][group[0].id]) for side in sides},
                }
            )


def _collection_order(sides, controlled, index):
    return (
        sides
        if index is None
        else tuple(
            "previous" if controlled and side == "baseline" else side
            for side in pair_order(index)
            if controlled or side in sides
        )
    )


def validate_retained_case(report: dict, case) -> None:
    """Validate one completed retained control independently of other cases."""
    if case not in retained_cases():
        raise ValueError("unknown retained control case")
    control = report.get("retained_control", {})
    selected = {
        **report,
        "measurements": [
            row for row in report.get("measurements", []) if row.get("case") == case.id
        ],
        "retained_control": {
            **control,
            "measurements": [
                row
                for row in control.get("measurements", [])
                if row.get("case") == case.id
            ],
            "pairs": [
                row for row in control.get("pairs", []) if row.get("case") == case.id
            ],
        },
    }
    _validate_retained(selected, {case.id})


def validate_retained_control(report: dict) -> None:
    """Bind all 44 control links to existing primary candidate observations."""
    _validate_retained(report, {case.id for case in retained_cases()})


def _validate_retained(report: dict, selected: set[str]) -> None:
    from scripts.benchmark_suite.issue406_fresh_v3 import candidate_commit

    control = report.get("retained_control", {})
    primary = {
        (record["case"], record["side"]): record
        for record in report.get("measurements", [])
    }
    records = {
        (record["case"], record["side"]): record
        for record in control.get("measurements", [])
    }
    _validate_retained_inventory(control, records, selected)
    pairs = control.get("pairs", [])
    slots = _retained_slots(selected)
    if len(pairs) != len(slots):
        raise ValueError("incomplete retained control pairs")
    seen = set()
    observations = set()
    for pair in pairs:
        slot = pair["case"], pair["round"], pair["index"]
        if slot not in slots or slot in seen:
            raise ValueError("duplicate or unexpected retained control slot")
        seen.add(slot)
        case, round_index, index = slot
        _validate_retained_pair_order(pair, index)
        for side, source, revision in (
            ("previous", records, PREVIOUS),
            ("candidate", primary, candidate_commit(report)),
        ):
            sample = _retained_sample(source, case, side, round_index, index)
            _validate_retained_reference(pair, side, sample, revision, observations)


def _validate_retained_inventory(control, records, selected):
    if (
        control.get("product_commit") != PREVIOUS
        or set(records) != {(case, "previous") for case in selected}
        or len(control["measurements"]) != len(selected)
    ):
        raise ValueError("invalid retained control inventory")


def _retained_slots(selected):
    return {
        (case, r, i)
        for case in selected
        for r in range(ROUNDS)
        for i in (None, *range(SAMPLES))
    }


def _validate_retained_pair_order(pair, index):
    expected_order = (
        ["previous", "candidate"]
        if index is None
        else ["previous" if side == "baseline" else side for side in pair_order(index)]
    )
    if pair.get("order") != expected_order:
        raise ValueError("retained control pair order changed")


def _retained_sample(source, case, side, round_index, index):
    record = source.get((case, side), {})
    if (
        len(record.get("warmups", [])) != ROUNDS
        or [len(values) for values in record.get("rounds", [])] != [SAMPLES] * ROUNDS
    ):
        raise ValueError("retained control has incomplete observations")
    sample = (
        record["warmups"][round_index]
        if index is None
        else record["rounds"][round_index][index]
    )
    return sample


def _validate_retained_reference(pair, side, sample, revision, observations):
    reference = pair.get(side)
    if (
        reference != _reference(sample)
        or sample.get("product_commit") != revision
        or not reference["observation_id"]
        or reference["observation_id"] in observations
    ):
        raise ValueError("retained control reference or product differs")
    observations.add(reference["observation_id"])
