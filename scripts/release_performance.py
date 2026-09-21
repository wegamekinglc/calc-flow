"""Exact-release paired timing gate and durable acceptance-step summaries."""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import os
import sys
from functools import partial
from pathlib import Path

from scripts.benchmark_suite.identity import compare_identity
from scripts.benchmark_suite.migrations import declared_migrations, load_migrations
from scripts.benchmark_suite.normalize import read_json
from scripts.benchmark_suite.process import ROOT, child_environment, command, install
from scripts.benchmark_suite.release_collectors import (
    pytest_command,
    python_environment,
    python_inventory,
    python_observation,
    rust_inventory,
    rust_observation,
)
from scripts.benchmark_suite.release_pairs import SIDES, collect_case, evaluate_case
from scripts.benchmark_suite.rust import allocation, build_binaries
from scripts.benchmark_suite.rust_provenance import with_compiled_dependencies
from scripts.toolkit import fingerprint_json, sha256_file, write_json
from scripts.verify_perf_gates import (
    check_stream_lifecycle_regression,
    load_stream_lifecycle,
)
from scripts.write_criterion_provenance import build_provenance

TARGETS = ("core", "stream_join_perf")
LIFECYCLE = (
    "benchmarks/test_symbolic_baseline.py::test_stream_window_checkpoint_and_recovery"
)
ROLLING = ("rolling_kernel_sma20", "rolling_kernel_dual_sma_5_20")


def matching_inventory(inventories: dict) -> list[str]:
    for names in inventories.values():
        if not names or len(names) != len(set(names)):
            raise ValueError("incomparable empty/duplicate case inventory")
    if inventories["baseline"] != inventories["candidate"]:
        raise ValueError("incomparable baseline/candidate case inventories")
    return inventories["candidate"]


def harness_identity() -> dict:
    paths = [
        *ROOT.glob("benchmarks/*.py"),
        *ROOT.glob("scripts/benchmark_suite/*.py"),
        ROOT / "scripts/release_performance.py",
        ROOT / "scripts/verify_perf_gates.py",
        ROOT / "scripts/verify_stream_lifecycle_evidence.py",
        ROOT / "scripts/verify_symbolic_milestone_perf.py",
        ROOT / "scripts/toolkit.py",
    ]
    return {str(path.relative_to(ROOT)): sha256_file(path) for path in sorted(paths)}


async def prepare(options: argparse.Namespace) -> dict:
    from scripts.benchmark_suite.release import load_release

    output = options.output
    roots = {"baseline": options.baseline_source.resolve(), "candidate": ROOT}
    releases, sites, binaries, provenance = {}, {}, {}, {}
    for side, source in roots.items():
        release = output / "builds" / side
        await command(
            [
                sys.executable,
                "-m",
                "scripts.benchmark_suite",
                "build",
                "--source",
                str(source),
                "--output",
                str(release),
            ],
            cwd=ROOT,
            log=release / "command.log",
            env=child_environment(),
        )
        releases[side] = load_release(release / "release.json")
        sites[side] = await install(releases[side], output / "install" / side)
    if releases["baseline"]["git_sha"] == releases["candidate"]["git_sha"]:
        raise ValueError("formal release baseline and candidate SHAs must be distinct")
    for side, source in roots.items():
        destination = output / "rust-builds" / side
        binaries[side] = await build_binaries(
            source,
            destination,
            ROOT / "target/release-benchmark-build",
            targets=(*TARGETS, "allocation_regression"),
        )
        provenance[side] = with_compiled_dependencies(
            build_provenance(
                source,
                [Path(f"crates/calc-flow/benches/{target}.rs") for target in TARGETS],
            ),
            source,
            {target: destination / f"build-{target}.jsonl" for target in TARGETS},
        )
        if provenance[side]["git_sha"] != releases[side]["git_sha"]:
            raise ValueError(f"{side} Rust source differs from the sealed release")
    write_json(output / "rust-provenance.json", provenance)
    return {
        "roots": roots,
        "releases": releases,
        "sites": sites,
        "binaries": binaries,
        "provenance": provenance,
    }


def rust_identities(context: dict, target: str) -> dict:
    provenance = context["provenance"]
    applied = declared_migrations(provenance, load_migrations(ROOT))
    path = f"crates/calc-flow/benches/{target}.rs"
    identities = {}
    for side in SIDES:
        raw = provenance[side]
        workload_source = provenance["candidate"] if target in applied else raw
        values = {
            "machine": raw["machine_identity"],
            "dependency": raw["compiled_dependency_identity"],
            "workload": {path: workload_source["workload_identity"][path]},
        }
        identities[side] = {
            key: value
            for name, identity in values.items()
            for key, value in (
                (f"{name}_identity", identity),
                (f"{name}_fingerprint", fingerprint_json(identity)),
            )
        }
        if target in applied:
            identities[side]["workload_migration"] = applied[target]
    compare_identity(identities["baseline"], identities["candidate"])
    return identities


async def _python_measure(name: str, context: dict, side: str, output: Path) -> dict:
    return await python_observation(
        name, context["sites"][side], context["releases"][side]["native_sha256"], output
    )


async def _rust_measure(name: str, inputs: dict, side: str, output: Path) -> dict:
    selected = inputs[side]
    return await rust_observation(
        name,
        selected["binary"],
        selected["source"],
        selected["identity"],
        output,
    )


def _rust_inputs(context: dict, target: str, identities: dict) -> dict:
    return {
        side: {
            "binary": context["binaries"][side][target],
            "source": context["roots"][side],
            "identity": identities[side],
        }
        for side in SIDES
    }


async def _record_case(
    name: str, measure, seals: dict, output: Path, previous: dict
) -> dict:
    report = {
        **previous,
        "cases": list(previous["cases"]),
        "errors": list(previous["errors"]),
    }
    destination = output / "cases" / hashlib.sha256(name.encode()).hexdigest()[:20]
    try:
        evidence = await collect_case(name, measure, destination)
        result = evaluate_case(evidence, seals)
        report["cases"].append(
            {
                "id": name,
                "evidence": str(destination / "pairs.json"),
                "seals": seals,
                "result": result,
            }
        )
    except Exception as error:
        report["errors"].append(f"{name}: {type(error).__name__}: {error}")
    write_json(output / "results.json", report)
    return report


async def measure(context: dict, output: Path, previous: dict) -> dict:
    report = {
        **previous,
        "cases": list(previous["cases"]),
        "errors": list(previous["errors"]),
    }
    inventories = {
        side: await python_inventory(
            context["sites"][side],
            context["releases"][side]["native_sha256"],
            output / "inventory" / side,
        )
        for side in SIDES
    }
    names = matching_inventory(inventories)
    report["inventory"] = {"python": inventories}
    seals = {side: context["releases"][side]["native_sha256"] for side in SIDES}
    for name in names:
        report = await _record_case(
            f"python/{name}",
            partial(_python_measure, name, context),
            seals,
            output,
            report,
        )
    for target in TARGETS:
        inventories = {
            side: await rust_inventory(
                context["binaries"][side][target],
                context["roots"][side],
                output / "inventory" / side / target,
            )
            for side in SIDES
        }
        names = matching_inventory(inventories)
        report["inventory"][target] = inventories
        identities = rust_identities(context, target)
        inputs = _rust_inputs(context, target, identities)
        seals = {side: sha256_file(context["binaries"][side][target]) for side in SIDES}
        for name in names:
            report = await _record_case(
                f"rust/{target}/{name}",
                partial(_rust_measure, name, inputs),
                seals,
                output,
                report,
            )

    return report


async def specialized(
    context: dict, options: argparse.Namespace, previous: dict
) -> dict:
    report = {**previous, "allocation": {}, "errors": list(previous["errors"])}
    output = options.output
    for side in SIDES:
        destination = output / "lifecycle" / side
        destination.mkdir(parents=True, exist_ok=True)
        await command(
            pytest_command([LIFECYCLE], destination),
            cwd=ROOT,
            log=destination / "run.log",
            env=python_environment(
                context["sites"][side],
                context["releases"][side]["native_sha256"],
                destination,
            ),
        )
        await command(
            [
                sys.executable,
                "scripts/verify_stream_lifecycle_evidence.py",
                str(destination / "pytest.json"),
                "--minimum-rounds",
                "20",
            ],
            cwd=ROOT,
            log=destination / "verify.log",
            env=child_environment(),
        )
        report.setdefault("allocation", {})[side] = await allocation(
            context["binaries"][side]["allocation_regression"],
            context["roots"][side],
            output / "allocation" / side,
            side,
        )
    await command(
        [
            str(context["binaries"]["candidate"]["allocation_regression"]),
            "--compare",
            str(output / "allocation/baseline/allocation.json"),
            str(output / "allocation/candidate/allocation.json"),
        ],
        cwd=context["roots"]["candidate"],
        log=output / "allocation/compare.log",
        env=child_environment(),
    )
    regressions = check_stream_lifecycle_regression(
        load_stream_lifecycle(output / "lifecycle/baseline"),
        load_stream_lifecycle(output / "lifecycle/candidate"),
        allow_dependency_drift=options.allow_dependency_drift,
    )
    report["lifecycle_regressions"] = regressions
    for scenario in ROLLING:
        matches = []
        for path in (output / "cases").glob("*/round-0/pair-0/candidate/pytest.json"):
            raw = read_json(path)
            if any(
                row.get("extra_info", {}).get("scenario") == scenario
                for row in raw["benchmarks"]
            ):
                matches.append(path)
        if len(matches) != 1:
            raise ValueError(
                f"expected one candidate rolling-kernel report: {scenario}"
            )
        await command(
            [
                sys.executable,
                "scripts/verify_symbolic_milestone_perf.py",
                "--report",
                str(matches[0]),
                "--scenario",
                scenario,
                "--expected-commit",
                context["releases"]["candidate"]["git_sha"],
                "--output",
                str(output / f"{scenario}.json"),
            ],
            cwd=ROOT,
            log=output / f"verify-{scenario}.log",
            env=child_environment(),
        )

    return report


async def run_gate(options: argparse.Namespace) -> int:
    options = argparse.Namespace(
        **{**vars(options), "output": options.output.resolve()}
    )
    output = options.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    report = {
        "contract": "release-paired-v1",
        "cases": [],
        "errors": [],
        "harness": harness_identity(),
        "dependency_lock_sha256": sha256_file(ROOT / "benchmarks/requirements.lock"),
        "load_average": list(os.getloadavg()) if hasattr(os, "getloadavg") else None,
        "power_configuration": "not controlled by the release harness",
    }
    write_json(output / "results.json", report)
    try:
        context = await prepare(options)
        report["releases"] = context["releases"]
        report = await measure(context, output, report)
        report = await specialized(context, options, report)
    except Exception as error:
        report["errors"].append(f"{type(error).__name__}: {error}")
    write_json(output / "results.json", report)
    summary = _gate_summary(report)
    (output / "summary.md").write_text(summary, encoding="utf-8")
    print(summary)
    return _gate_exit_code(report)


def _gate_summary(report: dict) -> str:
    lines = [
        "Release paired timing: two rounds of ten adjacent AB/BA case "
        "invocations; +5% gate.",
        "Timing inconclusive is not evidence of equivalence. "
        "Missing/invalid evidence blocks acceptance.",
        "",
    ]
    lines.extend(
        f"- {row['id']}: {row['result']['verdict']}" for row in report["cases"]
    )
    lines.extend(
        f"- incomparable / evidence failure: {error}" for error in report["errors"]
    )
    if report.get("lifecycle_regressions"):
        lines.append(f"- lifecycle: regression {report['lifecycle_regressions']}")
    return "\n".join(lines) + "\n"


def _gate_exit_code(report: dict) -> int:
    return int(
        bool(
            report["errors"]
            or report.get("lifecycle_regressions")
            or not report["cases"]
            or any(row["result"]["verdict"] == "regression" for row in report["cases"])
        )
    )


def acceptance_summary(results: dict) -> str:
    lines, failed = [], []
    for step in ("performance", "security", "soak"):
        result = results.get(step, {})
        outcome, conclusion = (
            result.get("outcome", "skipped"),
            result.get("conclusion", "skipped"),
        )
        reason = ""
        if outcome == "skipped":
            reason = (
                f"; skipped after prior step {', '.join(failed)}"
                if failed
                else "; skipped after setup/ref selection failure or cancellation"
            )
        lines.append(f"- {step}: outcome={outcome}, conclusion={conclusion}{reason}")
        if outcome not in ("success", "skipped"):
            failed.append(step)
    return "\n".join(lines) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--baseline-source", type=Path)
    parser.add_argument(
        "--allow-dependency-drift",
        action="store_true",
        help="Record acknowledgement; incompatible dependencies still block release",
    )
    parser.add_argument("--acceptance-summary", action="store_true")
    options = parser.parse_args()
    if options.acceptance_summary:
        options.output.mkdir(parents=True, exist_ok=True)
        results = json.loads(os.environ["ACCEPTANCE_STEPS"])
        write_json(options.output / "acceptance.json", results)
        summary = acceptance_summary(results)
        (options.output / "acceptance.md").write_text(summary, encoding="utf-8")
        with Path(os.environ["GITHUB_STEP_SUMMARY"]).open(
            "a", encoding="utf-8"
        ) as handle:
            handle.write(summary)
        return 0
    if options.baseline_source is None:
        parser.error("--baseline-source is required for measurement")
    return asyncio.run(run_gate(options))


if __name__ == "__main__":
    raise SystemExit(main())
