"""Pair PR #617's two convenience-stream cases across sealed release wheels.

Both sides run the candidate checkout's maintained
``benchmarks/test_stream_operators.py``; only the installed wheel differs.
Pairs follow ``scripts.benchmark_suite.release_pairs``: two rounds of ten
adjacent AB/BA invocations, each in a fresh pytest process, and the unified
paired-median verdict. Run from the candidate checkout with ``PYTHONPATH`` set
to that checkout.
"""

from __future__ import annotations

import argparse
import asyncio
import os
import platform
import sys
import time
from pathlib import Path

from scripts.benchmark_suite.identity import IDENTITIES, validate_component
from scripts.benchmark_suite.normalize import pytest_rows, read_json
from scripts.benchmark_suite.process import ROOT, command, install
from scripts.benchmark_suite.release_collectors import (
    pytest_command,
    python_environment,
)
from scripts.benchmark_suite.release_pairs import (
    RELEASE_ROUNDS,
    RELEASE_SAMPLES,
    SIDES,
    _pair_observations,
    _sample_median,
    _unique_worker,
    collect_case,
)
from scripts.benchmark_suite.report import comparison
from scripts.toolkit import sha256_file, wheel_native_sha256, write_json

DESCRIPTORS = ("scenario", "scope", "backend", "scale")
HARNESS = (
    "benchmarks/test_stream_operators.py",
    "benchmarks/warm_stream.py",
    "benchmarks/support.py",
    "scripts/benchmark_suite/release_pairs.py",
    "scripts/benchmark_suite/release_collectors.py",
    "scripts/benchmark_suite/report.py",
    "scripts/benchmark_suite/statistics.py",
)


def sealed_build(path: Path, expected_sha: str) -> dict:
    manifest = read_json(path)
    if (
        manifest.get("contract") != "warm-stream-build-v1"
        or manifest.get("build_profile") != "release"
    ):
        raise ValueError(f"{path}: expected a sealed release build manifest")
    if not manifest["git_sha"].startswith(expected_sha):
        raise ValueError(f"{path}: built {manifest['git_sha']}, not {expected_sha}")
    wheel = Path(manifest["wheel"])
    if (
        sha256_file(wheel) != manifest["wheel_sha256"]
        or wheel_native_sha256(wheel) != manifest["native_sha256"]
    ):
        raise ValueError(f"{path}: wheel or native seal changed after build")
    return manifest


def check_descriptors(metadata: dict, case: dict, workload: dict) -> None:
    if workload.get("scale") != case["scale"] or any(
        metadata.get(field) != workload.get(field) for field in DESCRIPTORS
    ):
        raise ValueError("incomparable workload identity: descriptor disagrees")


def check_row_counts(metadata: dict, workload: dict) -> None:
    rows = workload.get("rows_per_input")
    if metadata.get("input_rows") != rows or metadata.get("output_rows") != rows:
        raise ValueError("incomparable workload identity: row counts disagree")


def check_reference_fingerprints(metadata: dict, reference: dict | None) -> None:
    if reference is not None and any(
        metadata[f"{name}_fingerprint"] != reference[f"{name}_fingerprint"]
        for name in IDENTITIES
    ):
        raise ValueError("incomparable identity: fingerprint mismatch")


def check_identity(metadata: dict, case: dict, reference: dict | None) -> None:
    """Fail closed on corrupt, mismatched, or mislabeled identities.

    ``identity.validate_identity`` rejects this harness because its extra_info
    records ``input_rows``/``output_rows`` while the workload identity names the
    count ``rows_per_input``; check that agreement explicitly instead.
    """
    for name in IDENTITIES:
        validate_component(metadata, name)
    workload = metadata["workload_identity"]
    check_descriptors(metadata, case, workload)
    check_row_counts(metadata, workload)
    check_reference_fingerprints(metadata, reference)


async def observe(case: dict, site: Path, native: str, output: Path) -> dict:
    """Mirror ``release_collectors.python_observation`` at the case's scale."""
    output.mkdir(parents=True, exist_ok=True)
    environment = {
        **python_environment(site, native, output),
        "CALC_FLOW_BENCHMARK_SCALE": case["scale"],
    }
    await command(
        pytest_command([case["id"]], output),
        cwd=ROOT,
        log=output / "run.log",
        env=environment,
    )
    raw = read_json(output / "pytest.json")
    if any(not row.get("stats", {}).get("data") for row in raw["benchmarks"]):
        raise ValueError("saved Python raw samples are required")
    rows = pytest_rows(output / "pytest.json")
    if set(rows) != {case["id"]}:
        raise ValueError("invocation must produce exactly its selected case")
    observed = read_json(output / "native.json")["native_sha256"]
    if observed != native:
        raise ValueError("wrong sealed native SHA in Python observation")
    row = rows[case["id"]]
    check_identity(row["metadata"], case, None)
    return {
        "samples": row["samples"],
        "metadata": row["metadata"],
        "binary_sha256": observed,
        "worker": str(output),
        "correctness": True,
    }


def validate_observation(
    value: dict, context: tuple[dict, dict, dict], side: str, workers: set[str]
) -> None:
    case, seals, reference = context
    workers.add(_unique_worker(value, workers))
    if value.get("binary_sha256") != seals[side]:
        raise ValueError("incomparable observation: wrong native SHA")
    if value.get("correctness") is not True:
        raise ValueError("observation correctness was not established")
    check_identity(value["metadata"], case, reference)


def require_complete_rounds(evidence: dict) -> list:
    rounds = evidence["rounds"]
    if len(rounds) != RELEASE_ROUNDS or any(
        len(pairs) != RELEASE_SAMPLES for pairs in rounds
    ):
        raise ValueError("evidence requires two rounds of ten actual pairs")
    return rounds


def observation_values(case: dict, evidence: dict, seals: dict) -> dict:
    rounds = require_complete_rounds(evidence)
    reference = rounds[0][0]["observations"]["baseline"]["metadata"]
    values = {side: [[] for _ in rounds] for side in SIDES}
    workers: set[str] = set()
    for round_index, pairs in enumerate(rounds):
        for index, pair in enumerate(pairs):
            observations = _pair_observations(pair, index)
            for side in SIDES:
                value = observations[side]
                validate_observation(value, (case, seals, reference), side, workers)
                values[side][round_index].append(_sample_median(value["samples"]))
    return values


def evaluate(case: dict, evidence: dict, seals: dict) -> dict:
    """``release_pairs.evaluate_case`` with ``check_identity`` substituted."""
    return comparison(
        {
            "id": case["id"],
            "comparison": "interleaved",
            "correctness": True,
            **observation_values(case, evidence, seals),
        },
        minimum_samples=RELEASE_SAMPLES,
    )


def host() -> dict:
    cpu = next(
        (
            line.split(":", 1)[1].strip()
            for line in Path("/proc/cpuinfo").read_text().splitlines()
            if line.startswith("model name")
        ),
        platform.processor(),
    )
    memory = next(
        line.split(":", 1)[1].strip()
        for line in Path("/proc/meminfo").read_text().splitlines()
        if line.startswith("MemTotal")
    )
    return {
        "platform": platform.platform(),
        "cpu": cpu,
        "logical_cpus": os.cpu_count(),
        "affinity": len(os.sched_getaffinity(0)),
        "memory": memory,
        "python": platform.python_version(),
        "python_executable": sys.executable,
    }


async def measure_case(
    case: dict, sites: dict, seals: dict, output: Path, started: float
) -> dict:
    entry = {
        "label": case["label"],
        "id": case["id"],
        "scale": case["scale"],
        "load_before": os.getloadavg(),
        "started_seconds": time.monotonic() - started,
    }

    async def measure(side: str, destination: Path) -> dict:
        return await observe(case, sites[side], seals[side], destination)

    directory = output / f"case-{case['label']}"
    try:
        evidence = await collect_case(case["id"], measure, directory)
        result = evaluate(case, evidence, seals)
        entry = {**entry, "evidence": str(directory / "pairs.json"), "result": result}
    except Exception as error:
        entry = {**entry, "error": f"{type(error).__name__}: {error}"}
    return {
        **entry,
        "load_after": os.getloadavg(),
        "finished_seconds": time.monotonic() - started,
    }


def load_seals(protocol: dict) -> tuple[dict, dict]:
    builds = {
        side: sealed_build(
            Path(protocol[side]["build_manifest"]), protocol[side]["git_sha"]
        )
        for side in SIDES
    }
    if any(
        builds["baseline"][key] != protocol["baseline"][key]
        for key in ("wheel_sha256", "native_sha256")
    ):
        raise ValueError("baseline build does not match the registered seals")
    seals = {side: builds[side]["native_sha256"] for side in SIDES}
    if seals["baseline"] == seals["candidate"]:
        raise ValueError("baseline and candidate natives must differ")
    return builds, seals


def initial_report(
    args: argparse.Namespace, protocol: dict, builds: dict, seals: dict, budget: int
) -> dict:
    return {
        "contract": protocol["contract"],
        "argv": sys.argv,
        "protocol_sha256": sha256_file(args.protocol),
        "collector_sha256": sha256_file(Path(__file__)),
        "harness_sha256": {name: sha256_file(ROOT / name) for name in HARNESS},
        "builds": builds,
        "seals": seals,
        "rounds": RELEASE_ROUNDS,
        "pairs_per_round": RELEASE_SAMPLES,
        "budget_seconds": budget,
        "host": host(),
        "cases": [],
        "errors": [],
    }


async def install_sites(builds: dict, output: Path) -> dict:
    return {
        side: await install(
            {"wheel_path": builds[side]["wheel"]}, output / "install" / side
        )
        for side in SIDES
    }


async def measure_cases(
    report: dict,
    protocol: dict,
    bindings: tuple[dict, dict],
    output: Path,
    started: float,
) -> dict:
    sites, seals = bindings
    for case in protocol["cases"]:
        print(f"case {case['label']} {case['id']}", flush=True)
        entry = await measure_case(case, sites, seals, output, started)
        report = {**report, "cases": [*report["cases"], entry]}
        write_json(output / "results.json", report)
    return report


def finalize_report(report: dict, protocol: dict, started: float) -> dict:
    measured = {case["id"] for case in report["cases"] if "result" in case}
    return {
        **report,
        "unmeasured": [
            case["id"] for case in protocol["cases"] if case["id"] not in measured
        ],
        "elapsed_seconds": time.monotonic() - started,
    }


async def collect(args: argparse.Namespace) -> int:
    started = time.monotonic()
    protocol = read_json(args.protocol)
    output = args.output.resolve()
    if output.exists() and any(output.iterdir()):
        raise ValueError("measurement requires a fresh output directory")
    builds, seals = load_seals(protocol)
    budget = protocol["budget"]["measurement_seconds"]
    report = initial_report(args, protocol, builds, seals, budget)
    output.mkdir(parents=True, exist_ok=True)
    try:
        async with asyncio.timeout(budget):
            sites = await install_sites(builds, output)
            report = await measure_cases(
                report, protocol, (sites, seals), output, started
            )
    except TimeoutError:
        report = {
            **report,
            "errors": [
                *report["errors"],
                "measurement budget exhausted; remaining invocations unmeasured",
            ],
        }
    finally:
        report = finalize_report(report, protocol, started)
        write_json(output / "results.json", report)
    return int(bool(report["errors"] or report["unmeasured"]))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    raise SystemExit(asyncio.run(collect(parser.parse_args())))
