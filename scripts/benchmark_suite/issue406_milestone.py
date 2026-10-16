"""Current-milestone acquisition policy, separate from historical v2/v3 runs."""

from __future__ import annotations

import json
import math
import re
import time
from dataclasses import asdict
from pathlib import Path

from scripts.benchmark_suite import issue406_sources as sources
from scripts.benchmark_suite.issue406_acceptance import (
    MODES,
    AcceptanceCase,
    acceptance_inventory,
    validate_observation,
)
from scripts.benchmark_suite.issue406_fresh import Row5Budget
from scripts.benchmark_suite.issue406_measurement import (
    cpu_evidence_status,
    sample_cpu_evidence,
    validate_cpu_evidence,
)
from scripts.benchmark_suite.report import ROUNDS, SAMPLES
from scripts.toolkit import sha256_file, write_json

SCHEMA = "calc-flow.issue406-milestone.v1"
SIDES = ("baseline", "candidate")
ENVELOPE = {
    "budget_seconds": 5400,
    "collection_cutoff_seconds": 4800,
    "settlement_seconds": 300,
    "settlement_deadline_seconds": 5100,
    "startup_allowance_seconds": 5,
    "final_seal_allowance_seconds": 5,
    "row5_arm_seconds": 360,
    "row5_group_seconds": 720,
    "rounds": ROUNDS,
    "pairs_per_round": SAMPLES,
    "warmups_per_round": 1,
    "resource_limits": {
        "owned_rss_bytes": 24 << 30,
        "available_memory_bytes": 4 << 30,
        "output_bytes": 20 << 30,
        "free_disk_bytes": 10 << 30,
    },
}


class MilestoneCase(AcceptanceCase):
    """Both sealed revisions support OFF; preserve historical case semantics."""

    __slots__ = ()

    @property
    def sides(self) -> tuple[str, ...]:
        return SIDES


def inventory() -> tuple[MilestoneCase, ...]:
    """Keep all eight targets and add the nine planning configurations."""
    return (
        *(MilestoneCase(**asdict(case)) for case in acceptance_inventory()),
        *(MilestoneCase(9, "asof", 10_000_000, 64_000, mode) for mode in MODES),
        *(
            MilestoneCase(10, "interval_join", 1_000_000, 64_000, mode, threads)
            for threads in (1, 32)
            for mode in MODES
        ),
    )


def cpu_budgets() -> dict[str, float]:
    return {case.id: 2.0 if case.threads == 1 else 32.0 for case in inventory()}


def _revision(value) -> bool:
    return isinstance(value, str) and re.fullmatch(r"[0-9a-f]{40}", value) is not None


def _validate_revisions(plan: dict) -> list[str]:
    revisions = [plan.get(f"{side}_product_commit") for side in SIDES]
    harness = plan.get("harness_git_sha")
    if (
        not all(_revision(value) for value in (*revisions, harness))
        or revisions[0] == revisions[1]
        or plan.get("execution_git_sha") != harness
    ):
        raise ValueError(
            "milestone requires exact distinct B/C and one sealed execution SHA"
        )
    sources.validate_declaration(plan)
    return revisions


def _validate_release_revisions(plan: dict, revisions: list[str]) -> None:
    releases = plan.get("releases", {})
    if set(releases) != set(SIDES) or any(
        releases[side].get("product_commit") != revision
        for side, revision in zip(SIDES, revisions, strict=True)
    ):
        raise ValueError("milestone release differs from its sealed product revision")


def _validate_milestone_exceptions(plan: dict) -> None:
    forbidden = {
        "retained_control",
        "native_harness_reuse",
        "row7_harness_reuse",
        "historical_harness_witness",
        "known_main_failure_identity",
        "known_main_failure_identity_file",
        "execution_repair_witness",
        "precharged_seconds",
    }
    if forbidden.intersection(plan):
        raise ValueError(
            "milestone forbids third products and historical/precharge exceptions"
        )


def _validate_anchor_descriptor(anchor: dict) -> None:
    if (
        not isinstance(anchor.get("path"), str)
        or not anchor["path"]
        or re.fullmatch(r"[0-9a-f]{64}", str(anchor.get("sha256"))) is None
    ):
        raise ValueError("milestone requires a sealed preparation anchor")


def validate_envelope(plan: dict) -> None:
    """Reject drift in sources, modes, inventory, CPU limits or the fixed budget."""
    if plan.get("schema") != SCHEMA or any(
        plan.get(key) != value for key, value in ENVELOPE.items()
    ):
        raise ValueError("milestone acquisition differs from its fixed envelope")
    revisions = _validate_revisions(plan)
    _validate_release_revisions(plan, revisions)
    if plan.get("cases") != [asdict(case) for case in inventory()]:
        raise ValueError("milestone inventory must contain exactly 47 semantic cases")
    if plan.get("cpu_budgets") != cpu_budgets():
        raise ValueError("milestone requires serial 2x and explicit parallel 32x CPU")
    _validate_milestone_exceptions(plan)
    _validate_anchor_descriptor(plan.get("preparation_anchor", {}))


def begin_preparation(path: Path, *, source_guard: dict | None = None) -> None:
    """Start accounting after builds, before fixture, preflight and plan preparation."""
    record = {
        "schema": "calc-flow.issue406-preparation.v1",
        "clock": "CLOCK_MONOTONIC",
        "boot_id": Path("/proc/sys/kernel/random/boot_id").read_text().strip(),
        "started_monotonic": time.monotonic(),
        **({"source_guard": source_guard} if source_guard is not None else {}),
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as stream:
        json.dump(record, stream, indent=2, allow_nan=False)
        stream.write("\n")


def _sealed_preparation_anchor(plan: dict) -> dict:
    descriptor = plan["preparation_anchor"]
    path = Path(descriptor["path"])
    digest = sha256_file(path)
    if (
        digest != descriptor["sha256"]
        or plan.get("sealed_files", {}).get(str(path)) != digest
    ):
        raise ValueError("preparation anchor is not sealed")
    anchor = json.loads(path.read_text())
    if (
        anchor.get("schema") != "calc-flow.issue406-preparation.v1"
        or anchor.get("clock") != "CLOCK_MONOTONIC"
        or anchor.get("boot_id")
        != Path("/proc/sys/kernel/random/boot_id").read_text().strip()
    ):
        raise ValueError("preparation anchor clock or boot differs")
    sources.validate_anchor(plan, anchor)
    return anchor


def acquisition_start(plan: dict) -> float:
    """Charge all elapsed preparation on the same kernel boot; never assert zero."""
    anchor = _sealed_preparation_anchor(plan)
    started = anchor.get("started_monotonic")
    if (
        type(started) not in (int, float)
        or not math.isfinite(started)
        or not 0 <= started <= time.monotonic()
        or time.monotonic() - started >= ENVELOPE["collection_cutoff_seconds"]
    ):
        raise ValueError("preparation anchor is invalid or collection budget exhausted")
    return started


def report_identity(plan: dict) -> dict:
    return {
        "execution_schema": SCHEMA,
        **sources.report_fields(plan),
        **{
            key: plan[key]
            for key in (
                "baseline_product_commit",
                "candidate_product_commit",
                "harness_git_sha",
                "execution_git_sha",
                "build_harness_git_sha",
                "preparation_anchor",
            )
        },
    }


def validate_releases(plan: dict, files: dict) -> None:
    """Audit both ordinary wheels against their sealed original build tool."""
    from scripts.benchmark_suite.issue406_reuse import validate_ordinary_release

    if any(
        set(plan.get(key, {})) != set(SIDES)
        for key in ("row7_releases", "native_binaries")
    ):
        raise ValueError("milestone build inventory must contain only B/C")
    builds = [
        validate_ordinary_release(plan["releases"][side], files) for side in SIDES
    ]
    if builds[0] != builds[1] or builds[0]["rustc"] != plan["expected_native_rustc"]:
        raise ValueError(
            "milestone ordinary release dependencies/compiler/settings differ"
        )
    for side in SIDES:
        build = json.loads(Path(plan["releases"][side]["build_metadata"]).read_text())
        if build.get("harness_git_sha") != plan["build_harness_git_sha"]:
            raise ValueError(
                "ordinary release was not built by the sealed milestone tool"
            )


def _expected_input_rows(case) -> dict[str, int]:
    if case.workload in ("asof", "interval_join"):
        return {"reference.input": case.rows, "quotes.input": case.rows}
    if case.workload == "projection":
        return {"input": case.rows}
    return {"left": case.rows, "right": 64}


def _validate_event_geometry(event: dict) -> None:
    rows = event.get("record_batch_rows", [])
    if (
        not rows
        or any(type(value) is not int or value <= 0 for value in rows)
        or sum(rows) != event.get("rows")
        or len(rows) != event.get("record_batch_count")
    ):
        raise ValueError("runtime event RecordBatch geometry differs")


def _observed_input_rows(layout: list[dict]) -> dict[str, int]:
    observed = {}
    for event in layout:
        _validate_event_geometry(event)
        observed[event["input"]] = observed.get(event["input"], 0) + event["rows"]
    return observed


def _validate_interval_fixture(case, fixture: dict) -> None:
    if (
        fixture.get("schema") != "calc-flow.interval-acceptance-fixture.v1"
        or fixture.get("rows_per_side") != case.rows
        or fixture.get("batch_rows") != case.batch_rows
        or fixture.get("checkpoint_enabled") is not (case.mode != "off")
    ):
        raise ValueError("interval sample fixture differs from the declared workload")


def _validate_runtime_sample(case, sample: dict) -> None:
    layout = sample.get("event_layout", {}).get("data_events", [])
    if (
        not layout
        or type(sample.get("runtime_input_data_batches")) is not int
        or sample["runtime_input_data_batches"] != len(layout)
    ):
        raise ValueError("missing actual runtime input batch denominator")
    if _observed_input_rows(layout) != _expected_input_rows(case):
        raise ValueError(
            "runtime fixture input row counts differ from the declared workload"
        )
    if not set(sample.get("advancing_processing_epochs", [])).issubset(
        sample.get("completed_processing_epochs", [])
    ):
        raise ValueError("advancing epochs were not completed during processing")
    if case.workload == "interval_join":
        _validate_interval_fixture(case, sample.get("fixture", {}))


def validate_sample(case, side: str, sample: dict, identity: dict) -> None:
    """Require B/C ownership, complete lifecycle, CPU and real OFF evidence."""
    validate_observation(case, side, sample)
    if (
        sample.get("product_commit") != identity.get(f"{side}_product_commit")
        or sample.get("lifecycle") != "completed"
    ):
        raise ValueError("milestone sample source or lifecycle differs")
    validate_cpu_evidence(
        sample_cpu_evidence(sample, scope=case.cpu_scope),
        threads=case.threads,
        budget=cpu_budgets()[case.id],
    )
    if sample.get("cpu_budget") != cpu_budgets()[case.id]:
        raise ValueError("sample CPU budget differs from the frozen declaration")
    if case.workload in (
        "join_sealed",
        "join_retained",
        "asof",
        "projection",
        "interval_join",
    ):
        _validate_runtime_sample(case, sample)
    elif case.mode == "off" and sample.get("checkpoint_enabled") is not False:
        raise ValueError("native OFF sample did not disable checkpointing")


def validate_results(report: dict) -> None:
    """A partial inventory, third product or failed lifecycle never completes a run."""
    from scripts.benchmark_suite.issue406_acceptance import validate_complete_results

    if "retained_control" in report:
        raise ValueError("milestone cannot include third-product controls")
    validate_complete_results(report, cases=inventory())
    known = {case.id: case for case in inventory()}
    for record in report["measurements"]:
        for sample in [
            *record["warmups"],
            *(item for group in record["rounds"] for item in group),
        ]:
            case = known[record["case"]]
            validate_sample(case, record["side"], sample, report)
            if (
                cpu_evidence_status(
                    sample_cpu_evidence(sample, scope=case.cpu_scope),
                    threads=case.threads,
                    budget=sample["cpu_budget"],
                )
                != "verified"
            ):
                raise ValueError(
                    "milestone CPU precision leaves the budget inconclusive"
                )


async def collect(runner, output: Path) -> dict:
    """Collect 1,800 samples and 180 warmups once, without historical replacement."""
    from scripts.benchmark_suite.issue406_run import (
        _collect_round,
        collection_groups,
    )

    records = {
        (case.id, side): {
            "case": case.id,
            "side": side,
            "rounds": [[], []],
            "warmups": [],
        }
        for case in inventory()
        for side in SIDES
    }
    report = {"state": "running", "measurements": list(records.values())}
    output.mkdir(parents=True, exist_ok=True)
    try:
        for group in collection_groups(inventory()):
            budget = Row5Budget(runner.plan) if group[0].row == 5 else None
            for round_index in range(ROUNDS):
                await _collect_round(
                    runner, output, group, (round_index, budget), records
                )
                write_json(output / "partial.json", report)
        report["state"] = "collected"
        return report
    except BaseException as error:
        report.update(state="stopped", error=f"{type(error).__name__}: {error}")
        raise
    finally:
        write_json(output / "partial.json", report)
