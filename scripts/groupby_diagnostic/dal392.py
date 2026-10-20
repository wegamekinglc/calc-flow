"""Fixed 1M group_by controls using the original October 20 release wheels."""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import platform
import subprocess
import sys
import time
from pathlib import Path
from unittest.mock import patch

from scripts.benchmark_suite.catalog import engine_cases
from scripts.benchmark_suite.issue406_measurement import PhaseClock
from scripts.benchmark_suite.measure import (
    _measured_row,
    validate_environment,
    validate_sample,
)
from scripts.benchmark_suite.process import ROOT, install
from scripts.benchmark_suite.provenance import harness_sha256
from scripts.benchmark_suite.release import load_release
from scripts.benchmark_suite.report import comparison
from scripts.benchmark_suite.worker import dispatch
from scripts.groupby_diagnostic.contract import host, require_host
from scripts.groupby_diagnostic.controller import observed_collect, paired
from scripts.groupby_diagnostic.runtime import AuditWorker, input_hashes
from scripts.toolkit import git_output, sha256_file, write_json

SEALS = {
    "A": {
        "git_sha": "86c74e9a7fa93c4987f6b603ce8b3a5ae33bccbe",
        "wheel_sha256": (
            "91c7f8e286fbfcd8a9524756361eba548d2193d363847c40bad0ba363a558731"
        ),
        "native_sha256": (
            "48b66530bbf75824e257ad05cabcbf779516250d4583e19f3938be49fae46eba"
        ),
    },
    "B": {
        "git_sha": "1085c651d704585d0d1afeff57fce0f7ba53a2d3",
        "wheel_sha256": (
            "4d6f6530f726260b8d13c28420d4f4ae81ee5ab19410371dac28bbbd8be92c83"
        ),
        "native_sha256": (
            "85a5db3a3ab66d6ec94b67f5636a919a587b4d2f71c3d4c9c30c0c38febb8066"
        ),
    },
}
HARNESS = "88f7acd751817ace6bbc4b24e957d0fb9b954e40d561510e0fc00ae5c5228dea"


def case() -> dict:
    return next(
        row
        for row in engine_cases(1_000_000)
        if row["backend"] == "calc-flow-stream" and row["scenario"] == "group_by"
    )


def plan() -> dict:
    return {
        "contract": "dal392-original-wheel-diagnostic-v1",
        "case": case(),
        "source_run": 39423189143,
        "seals": SEALS,
        "harness_sha256": HARNESS,
        "comparisons": ["AA", "BB", "AB"],
        "phase_comparison": "AB",
        "fresh_comparisons": ["AA", "AB"],
        "fresh_observation": (
            "One warmup and one sample per newly started worker; no process reuse."
        ),
        "rounds": 2,
        "pairs_per_round": 10,
        "budget_seconds": 600,
        "builds": "none",
        "interpretation": (
            "Uninstrumented pairs retain the suite gate. "
            "Phase pairs are attribution only."
        ),
        "phases": ["setup-readiness", "processing", "terminal", "oracle"],
        "processing": (
            "Original ready-enqueue-to-arrow timer; "
            "native node durations are wall time, not CPU."
        ),
        "phase_cpu": "CLOCK_PROCESS_CPUTIME_ID plus Linux task schedstat clock.",
    }


def sealed(side: str, path: Path) -> dict:
    value = load_release(path)
    if any(value.get(k) != v for k, v in SEALS[side].items()):
        raise ValueError(f"{side}: original October 20 release required")
    return value


class DiagnosticWorker(AuditWorker):
    worker_module = "scripts.groupby_diagnostic.dal392"


class PhaseWorker(DiagnosticWorker):
    worker_module = "scripts.groupby_diagnostic.dal392_phase"


class PhaseCapture:
    """Own diagnostic clocks outside the maintained processing timer."""

    def __init__(self) -> None:
        self.phases = {}
        self.setup = PhaseClock("setup-readiness")
        self.terminal = None

    async def processing(self, operation, bindings, **kwargs):
        def start(scope):
            self.phases["setup-readiness"] = self.setup.finish()
            return PhaseClock(scope)

        table, seconds, evidence = await operation(
            bindings, **kwargs, phase_clock=start
        )
        self.phases["processing"] = evidence["cpu_evidence"]
        self.phases["native_status_after_processing"] = bindings.job.status()
        self.terminal = PhaseClock("terminal")
        return table, seconds, evidence

    def validate(self, operation, table):
        if self.terminal is None:
            raise ValueError("oracle reached before processing clock completed")
        self.phases["terminal"] = self.terminal.finish()
        clock = PhaseClock("oracle")
        result = operation(table)
        self.phases["oracle"] = clock.finish()
        return result


def phase_sample(active) -> dict:
    from functools import partial

    from benchmarks import engine_stream

    capture = PhaseCapture()
    with (
        patch.object(
            engine_stream,
            "_measure_ready_clock",
            partial(capture.processing, engine_stream._measure_ready_clock),
        ),
        patch.object(active, "validate", partial(capture.validate, active.validate)),
    ):
        result = active.sample()
    return {**result, "phase_evidence": capture.phases}


def persist_phases(sample: dict, root: Path, number: int) -> dict:
    """Keep full task snapshots off the bounded JSON-lines IPC channel."""

    evidence = sample["phase_evidence"]
    path = root / f"phase-sample-{number:04d}.json"
    write_json(path, evidence)
    summary = {
        phase: {
            key: evidence[phase][key]
            for key in ("status", "wall_seconds", "process_cpu_seconds")
        }
        for phase in plan()["phases"]
    }
    return {
        **sample,
        "phase_evidence": {
            "file": path.name,
            "sha256": sha256_file(path),
            "summary": summary,
            "operators": evidence["native_status_after_processing"]["operators"],
        },
    }


def worker(root: Path, *, phases: bool = False) -> None:
    active = None
    try:
        for line in sys.stdin:
            message = json.loads(line)
            if message["operation"] == "prepare" and message["case"] != case():
                raise ValueError("only fixed group_by/1M workload is permitted")
            if phases and message["operation"] == "sample":
                response = persist_phases(phase_sample(active), root, active.count)
            else:
                response, active = dispatch(message, active, root)
            if message["operation"] == "prepare":
                response = {**response, "input_hashes": input_hashes(active, root)}
            print(json.dumps(response, allow_nan=False), flush=True)
    finally:
        if active is not None:
            active.close()


def preflight(root: Path, release_root: Path) -> dict:
    machine = host()
    write_json(root / "host.json", machine)
    require_host(machine)
    cpu = json.loads(subprocess.check_output(["lscpu", "--json"], text=True))
    write_json(root / "host-cpu.json", cpu)
    model = next(x["data"] for x in cpu["lscpu"] if x["field"] == "Model name:")
    if model != "AMD EPYC 7763 64-Core Processor":
        raise ValueError(f"CPU model differs from failed CI: {model}")
    if platform.python_version() != "3.13.16" or harness_sha256() != HARNESS:
        raise ValueError("Python or common harness differs from original CI")
    releases = {
        side: sealed(side, release_root / side / "release.json") for side in SEALS
    }
    write_json(
        root / "identity.json",
        {
            "releases": releases,
            "checkout": git_output(ROOT, "rev-parse", "HEAD"),
            "harness_sha256": harness_sha256(),
            "diagnostic_sha256": {
                name: sha256_file(Path(__file__).with_name(name))
                for name in ("dal392.py", "dal392_phase.py")
            },
        },
    )
    return releases


async def prepare_one(worker, release: dict, context: dict) -> dict:
    identity = validate_environment(
        await worker.request(operation="hello", context={**context, "phase": "hello"}),
        release,
        backend=case()["backend"],
    )
    response = await worker.request(
        operation="prepare",
        case=case(),
        context={**context, "phase": "warmup", "sample": None},
    )
    if response["case"] != case():
        raise ValueError("fresh worker prepared a different workload")
    validate_sample(response["warmup"])
    return identity


async def fresh_round(sites: dict, releases: dict, root: Path, index: int) -> dict:
    samples = {side: [] for side in releases}
    environment = None
    hashes = None
    processes = []
    for number in range(plan()["pairs_per_round"]):
        order = (
            ("baseline", "candidate") if number % 2 == 0 else ("candidate", "baseline")
        )
        for side in order:
            worker = await DiagnosticWorker.start(
                sites[side], root / f"pair-{number}" / side
            )
            context = {
                side: {
                    "side": side,
                    "round": index,
                    "case_id": case()["id"],
                    "release": releases[side],
                }
            }
            try:
                identity = await prepare_one(worker, releases[side], context[side])
                if environment is not None and identity != environment:
                    raise ValueError("fresh-worker environment differs")
                if hashes is not None and worker.input_hashes != hashes:
                    raise ValueError("fresh-worker input IPC differs")
                environment, hashes = identity, worker.input_hashes
                sample = await worker.request(
                    operation="sample",
                    context={**context[side], "phase": "measurement", "sample": number},
                )
                validate_sample(sample)
                samples[side].append(sample)
                processes.append(
                    {"side": side, "sample": number, "pid": worker.process.pid}
                )
                complete = await worker.request(operation="finish")
                if complete["state"] != "completed":
                    raise ValueError("fresh worker did not finish")
                write_json(
                    root / "partial.json", {"samples": samples, "workers": processes}
                )
            finally:
                await worker.close()
    return {
        "environment": environment,
        "samples": samples,
        "workers": processes,
        "input_hashes": hashes,
    }


async def fresh_paired(sites: dict, releases: dict, root: Path) -> dict:
    rounds = []
    for index in range(plan()["rounds"]):
        rounds.append(
            await fresh_round(sites, releases, root / f"round-{index}", index)
        )
        write_json(root / "rounds.json", rounds)
    if rounds[0]["environment"] != rounds[1]["environment"]:
        raise ValueError("fresh-worker confirmation environment changed")
    row = _measured_row(case(), rounds, "interleaved")
    return {**row, "result": comparison(row)}


async def collect(root: Path, releases: dict, *, part: str) -> None:
    sites = {}
    for side, value in releases.items():
        sites[side] = await install(value, root / "sites" / side)
    results = []
    if part == "fresh":
        labels = [f"fresh-{pair}" for pair in plan()["fresh_comparisons"]]
    else:
        labels = (
            ["phase-AB"] if part == "phases" else [*plan()["comparisons"], "phase-AB"]
        )
    for label in labels:
        left, right = label.rsplit("-", 1)[-1]
        selected = {"baseline": left, "candidate": right}
        selected_sites = {side: sites[value] for side, value in selected.items()}
        selected_releases = {side: releases[value] for side, value in selected.items()}
        if part == "fresh":
            row = await fresh_paired(selected_sites, selected_releases, root / label)
        else:
            row = await paired(
                case(),
                selected_sites,
                selected_releases,
                root / label,
                worker_type=PhaseWorker if label == "phase-AB" else DiagnosticWorker,
            )
        results.append({"label": label, "sources": selected, **row})
        write_json(root / "results.json", results)
        print(label, json.dumps(row["result"]), flush=True)


async def run(root: Path, releases: Path) -> int:
    started = float(os.environ["DAL392_START_UNIX"])
    root.mkdir(parents=True, exist_ok=True)
    part = os.environ.get("DAL392_PART", "all")
    if part not in ("all", "phases", "fresh"):
        raise ValueError("unknown diagnostic part")
    write_json(
        root / "plan.json",
        {
            **plan(),
            "job_started_unix": started,
            "part": part,
            "reused_control_run": 39458710899 if part != "all" else None,
        },
    )
    outcome = {"status": "failed"}
    try:
        remaining = started + plan()["budget_seconds"] - time.time() - 10
        if remaining <= 0:
            raise TimeoutError("preparation exhausted measurement budget")
        async with asyncio.timeout(remaining):
            manifests = preflight(root, releases)
            await observed_collect(root, lambda: collect(root, manifests, part=part))
        outcome = {"status": "completed"}
    except (Exception, asyncio.CancelledError) as error:
        outcome = {"status": "failed", "error": f"{type(error).__name__}: {error}"}
    finally:
        outcome["elapsed_seconds_including_preparation"] = time.time() - started
        outcome["budget_satisfied"] = (
            outcome["elapsed_seconds_including_preparation"] <= plan()["budget_seconds"]
        )
        if not outcome["budget_satisfied"]:
            outcome["status"] = "invalid"
        write_json(root / "outcome.json", outcome)
    return 0 if outcome["status"] == "completed" else 1


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("operation", choices=("worker", "run", "plan"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--releases", type=Path)
    args = parser.parse_args()
    if args.operation == "worker":
        worker(args.output)
    elif args.operation == "plan":
        write_json(args.output / "plan.json", plan())
    elif args.releases is None:
        parser.error("run requires --releases")
    else:
        return asyncio.run(run(args.output, args.releases))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
