"""The four preregistered P2 warm-decay cases, using maintained warm workers."""

from __future__ import annotations

import argparse
import asyncio
import os
import time
from pathlib import Path

import psutil

from scripts import profile_warm_stream as profile
from scripts.release_performance import harness_identity
from scripts.toolkit import write_json

CASES = tuple(
    {
        "history_rows": 1000,
        "history_segment_rows": 1000,
        "entities": 1,
        "window": window,
        "append_rows": append,
        "indicator": indicator,
    }
    for window, append, indicator in (
        (20, 1024, "rolling_decay"),
        (1000, 1024, "rolling_decay"),
        (1000, 1024, "decay_mean"),
        (1000, 1, "rolling_decay"),
    )
)
ROW_FIELDS = (
    "shortcut_rows",
    "certified_rows",
    "guard_fallback_rows",
    "quota_reference_rows",
    "scheduled_reseed_rows",
)
WORK_FIELDS = (
    "transitions",
    *ROW_FIELDS,
    "initial_seed_samples",
    "scheduled_reseed_samples",
    "guard_reseed_samples",
    "reference_scan_samples",
    "cache_copied_samples",
    "reference_history_samples",
)


def valid_counters(work):
    if set(work) - set(WORK_FIELDS):
        raise ValueError("unexpected native decay counter")
    if any(type(value) is not int or value < 0 for value in work.values()):
        raise ValueError("invalid native decay counter")
    return work


def total_work(status):
    totals = dict.fromkeys(WORK_FIELDS, 0)
    available = False
    for node in status.get("rolling_metrics", {}).values():
        for phase in ("data", "watermark", "end"):
            work = node.get(phase, {}).get("_decay_work")
            if work is None:
                continue
            available = True
            for name, value in valid_counters(work).items():
                totals[name] += value
    if sum(totals[name] for name in ROW_FIELDS) != totals["transitions"]:
        raise ValueError("native decay row classes do not cover transitions")
    return totals if available else None


def work_witness(terminal):
    before = total_work(terminal["before_status"])
    after = total_work(terminal["after_status"])
    if after is None:
        return {"available": False, "before": None, "after": None, "delta": None}
    before = before or dict.fromkeys(WORK_FIELDS, 0)
    delta = {name: after[name] - before[name] for name in WORK_FIELDS}
    if any(value < 0 for value in delta.values()):
        raise ValueError("native decay counters moved backwards")
    return {"available": True, "before": before, "after": after, "delta": delta}


def acceptance(cases):
    if len(cases) != len(CASES):
        raise ValueError("all four cases are required for acceptance")
    improved = [
        index
        for index, case in enumerate(cases[:3])
        if case["paired_speedup_median"] > 1.05 and case["paired_speedup_ci95"][0] > 1
    ]
    regressed = [
        index
        for index, case in enumerate(cases)
        if case["paired_speedup_ci95"][1] < 1 / 1.05
    ]
    return {
        "improved_targets": improved,
        "confirmed_regressions": regressed,
        "accepted": bool(improved) and not regressed,
    }


async def collect(args):
    started = time.monotonic()
    args.output.mkdir(parents=True, exist_ok=False)
    deadline = started + 530
    report = {"status": "running", "cases": []}
    write_json(args.output / "report.json", report)
    try:
        manifests = await profile._compatible_builds(
            (args.baseline_build, args.candidate_build)
        )
        original_harness = harness_identity()
        compilers = [
            {"pid": process.info["pid"], "name": process.info["name"]}
            for process in psutil.process_iter(["pid", "name"])
            if process.info["name"] in {"cargo", "rustc", "maturin"}
        ]
        write_json(
            args.output / "protocol.json",
            {
                "cases": CASES,
                "sample_pairs": 30,
                "gc": "normal",
                "numerical_profile": "stable_v1 (default)",
                "budget_seconds": 600,
                "acquisition_timeout_seconds": 530,
                "cleanup_reserve_seconds": 70,
                "harness_sha256": original_harness,
                "compiler_processes": compilers,
                "cpu_affinity": sorted(os.sched_getaffinity(0)),
                "available_memory_bytes": psutil.virtual_memory().available,
                "load_average": list(os.getloadavg()),
                "builds": manifests,
            },
        )
        if compilers:
            raise ValueError("competing compiler at acquisition preflight")
        await asyncio.wait_for(
            collect_cases(args, manifests, report),
            timeout=max(1, deadline - time.monotonic()),
        )
        if harness_identity() != original_harness:
            raise ValueError("shared measurement harness changed")
        report["status"] = "completed"
        report["acceptance"] = acceptance(report["cases"])
    except BaseException as error:
        report["status"] = "stopped"
        report["failure"] = f"{type(error).__name__}: {error}"
        raise
    finally:
        report["elapsed_seconds"] = time.monotonic() - started
        write_json(args.output / "report.json", report)


async def collect_cases(args, manifests, report):
    for config in CASES:
        case, environments = await profile._fresh_case(
            manifests, config, 30, False, profile.WorkerPairOptions(args.output)
        )
        if "environments" not in report:
            report.update(profile._new_report(manifests, environments, 30))
        elif report["environments"] != environments:
            raise ValueError("worker environment changed between cases")
        case["native_work"] = [work_witness(terminal) for terminal in case["terminal"]]
        candidate = case["native_work"][1]
        if not candidate["available"] or candidate["delta"]["certified_rows"] == 0:
            raise ValueError("candidate native decay work was not witnessed")
        expected = config["history_rows"] + 30 * config["append_rows"]
        if candidate["after"]["transitions"] != expected:
            raise ValueError("native decay transition count differs from actual input")
        report["cases"].append(case)
        write_json(args.output / "report.json", report)
        print(f"{config}: {case['paired_speedup_median']:.4f}x", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline-build", required=True, type=Path)
    parser.add_argument("--candidate-build", required=True, type=Path)
    parser.add_argument(
        "--output", required=True, type=lambda path: Path(path).resolve()
    )
    asyncio.run(collect(parser.parse_args()))


if __name__ == "__main__":
    main()
