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
from scripts.benchmark_suite.release_pairs import (
    RELEASE_ROUNDS,
    RELEASE_SAMPLES,
    SIDES,
    collect_case,
    evaluate_case,
)
from scripts.benchmark_suite.rust import allocation, build_binaries
from scripts.benchmark_suite.rust_provenance import with_compiled_dependencies
from scripts.toolkit import FULL_SHA, fingerprint_json, sha256_file, write_json
from scripts.verify_perf_gates import (
    check_stream_lifecycle_regression,
    load_stream_lifecycle,
)
from scripts.write_criterion_provenance import build_provenance

TARGETS = ("core", "stream_join_perf")
SUITES = ("python", *TARGETS)
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
    releases, sites, binaries, provenance, binary_sha256 = {}, {}, {}, {}, {}
    suite = getattr(options, "suite", None) or "all"
    rust_targets = TARGETS if suite == "all" else (suite,) if suite in TARGETS else ()
    build_targets = (
        (*rust_targets, "allocation_regression")
        if suite in ("all", "python")
        else rust_targets
    )
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
            targets=build_targets,
        )
        binary_sha256[side] = {
            target: sha256_file(binary) for target, binary in binaries[side].items()
        }
        write_json(destination / "binary-sha256.json", binary_sha256[side])
        if rust_targets:
            provenance[side] = with_compiled_dependencies(
                build_provenance(
                    source,
                    [
                        Path(f"crates/calc-flow/benches/{target}.rs")
                        for target in rust_targets
                    ],
                ),
                source,
                {
                    target: destination / f"build-{target}.jsonl"
                    for target in rust_targets
                },
            )
            if provenance[side]["git_sha"] != releases[side]["git_sha"]:
                raise ValueError(f"{side} Rust source differs from the sealed release")
    if provenance:
        write_json(output / "rust-provenance.json", provenance)
    return {
        "roots": roots,
        "releases": releases,
        "sites": sites,
        "binaries": binaries,
        "binary_sha256": binary_sha256,
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


async def _measure_python(context: dict, output: Path, report: dict) -> dict:
    inventories = {
        side: await python_inventory(
            context["sites"][side],
            context["releases"][side]["native_sha256"],
            output / "inventory" / side,
        )
        for side in SIDES
    }
    names = matching_inventory(inventories)
    report["inventory"]["python"] = inventories
    seals = {side: context["releases"][side]["native_sha256"] for side in SIDES}
    for name in names:
        report = await _record_case(
            f"python/{name}",
            partial(_python_measure, name, context),
            seals,
            output,
            report,
        )
    return report


async def _measure_rust_target(
    context: dict, output: Path, report: dict, target: str
) -> dict:
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
    seals = {side: context["binary_sha256"][side][target] for side in SIDES}
    for side in SIDES:
        if sha256_file(context["binaries"][side][target]) != seals[side]:
            raise ValueError(f"{target} {side} Rust binary changed after build")
    report["rust_binary_sha256"][target] = seals
    for name in names:
        report = await _record_case(
            f"rust/{target}/{name}",
            partial(_rust_measure, name, inputs),
            seals,
            output,
            report,
        )
    return report


async def measure(
    context: dict, output: Path, previous: dict, suite: str = "all"
) -> dict:
    report = {
        **previous,
        "cases": list(previous["cases"]),
        "errors": list(previous["errors"]),
        "inventory": {},
        "rust_binary_sha256": dict(previous.get("rust_binary_sha256", {})),
    }
    if suite in ("all", "python"):
        report = await _measure_python(context, output, report)
    for target in TARGETS:
        if suite in ("all", target):
            report = await _measure_rust_target(context, output, report, target)
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
        "suite": getattr(options, "suite", None) or "all",
        "cases": [],
        "errors": [],
        "rust_binary_sha256": {},
        "harness": harness_identity(),
        "dependency_lock_sha256": sha256_file(ROOT / "benchmarks/requirements.lock"),
        "load_average": list(os.getloadavg()) if hasattr(os, "getloadavg") else None,
        "power_configuration": "not controlled by the release harness",
    }
    write_json(output / "results.json", report)
    suite = report["suite"]
    try:
        context = await prepare(options)
        report["releases"] = context["releases"]
        report = await measure(context, output, report, suite)
        if suite in ("all", "python"):
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
        f"Release paired timing: {RELEASE_ROUNDS} rounds of {RELEASE_SAMPLES} "
        "adjacent AB/BA case invocations; +5% gate.",
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


def _verify_rust_provenance(source: Path, suite: str, shas: dict) -> None:
    path = source / "rust-provenance.json"
    if not path.is_file():
        raise ValueError(f"missing {suite} Rust provenance")
    provenance = read_json(path)
    if set(provenance) != set(SIDES):
        raise ValueError(f"incomplete {suite} Rust provenance")
    for side in SIDES:
        if provenance[side].get("git_sha") != shas[side]:
            raise ValueError(f"{suite} {side} Rust source differs from sealed release")
    rust_identities({"provenance": provenance}, suite)


def _verify_sealed_release(
    source: Path, collected: dict, suite: str, side: str
) -> None:
    from scripts.benchmark_suite.release import load_release

    manifest = source / "builds" / side / "release.json"
    if not manifest.is_file():
        raise ValueError(f"missing {suite} {side} sealed release manifest")
    sealed = load_release(manifest)
    if any(
        sealed[key] != collected["releases"][side][key]
        for key in ("git_sha", "wheel_sha256", "native_sha256")
    ):
        raise ValueError(f"{suite} {side} sealed release identity differs")


def _verify_rust_build(source: Path, collected: dict, suite: str, side: str) -> None:
    target = "allocation_regression" if suite == "python" else suite
    log = source / "rust-builds" / side / f"build-{target}.jsonl"
    if not log.is_file():
        raise ValueError(f"missing {suite} {side} Rust build log")
    digest = source / "rust-builds" / side / "binary-sha256.json"
    if not digest.is_file():
        raise ValueError(f"missing {suite} {side} Rust build digest")
    if (
        suite in TARGETS
        and read_json(digest).get(suite) != _expected_case_seals(collected, suite)[side]
    ):
        raise ValueError(f"{suite} {side} Rust build digest differs")


def _verify_suite_builds(source: Path, collected: dict, suite: str, shas: dict) -> None:
    for side in SIDES:
        _verify_sealed_release(source, collected, suite, side)
        _verify_rust_build(source, collected, suite, side)
    if suite in TARGETS:
        _verify_rust_provenance(source, suite, shas)


def _load_suite_report(
    source: Path, suite: str, expected_harness: dict, expected_lock: str
) -> dict:
    path = source / "results.json"
    if not path.is_file():
        raise ValueError(f"missing {suite} suite results")
    collected = read_json(path)
    if (
        collected.get("contract") != "release-paired-v1"
        or collected.get("suite") != suite
    ):
        raise ValueError(f"invalid {suite} suite contract")
    if collected.get("harness") != expected_harness:
        raise ValueError(f"{suite} suite harness differs from the candidate")
    if collected.get("dependency_lock_sha256") != expected_lock:
        raise ValueError(f"{suite} suite dependency lock differs from the candidate")
    return collected


def _suite_shas(collected: dict, suite: str) -> dict | None:
    try:
        shas = {side: collected["releases"][side]["git_sha"] for side in SIDES}
    except (KeyError, TypeError):
        if collected.get("errors"):
            return None
        raise ValueError(f"missing {suite} sealed release identity") from None
    if any(
        not isinstance(sha, str) or not FULL_SHA.fullmatch(sha) for sha in shas.values()
    ):
        raise ValueError(f"invalid {suite} sealed release SHA")
    if shas["baseline"] == shas["candidate"]:
        raise ValueError(f"{suite} baseline and candidate SHAs are equal")
    return shas


def _check_release_set(report: dict, collected: dict, shas: dict, suite: str) -> None:
    if "releases" not in report:
        report["releases"] = collected["releases"]
    elif shas != {side: report["releases"][side]["git_sha"] for side in SIDES}:
        raise ValueError(f"{suite} suite release SHAs disagree")


def _suite_inventory(collected: dict, suite: str) -> tuple[dict, list[str]] | None:
    inventory = collected.get("inventory", {})
    if set(inventory) != {suite}:
        if collected.get("errors"):
            return None
        raise ValueError(f"invalid {suite} case inventory")
    return inventory[suite], matching_inventory(inventory[suite])


def _expected_case_seals(collected: dict, suite: str) -> dict:
    if suite == "python":
        return {side: collected["releases"][side]["native_sha256"] for side in SIDES}
    seals = collected["rust_binary_sha256"][suite]
    if set(seals) != set(SIDES) or any(
        not isinstance(seal, str) or len(seal) != 64 for seal in seals.values()
    ):
        raise ValueError(f"incomplete {suite} Rust binary seals")
    return seals


def _verified_case(source: Path, suite: str, row: dict, expected_seals: dict) -> dict:
    case_id = row["id"]
    digest = hashlib.sha256(case_id.encode()).hexdigest()[:20]
    relative = Path("cases") / digest / "pairs.json"
    evidence = source / relative
    if Path(row["evidence"]).parts[-3:] != relative.parts or not evidence.is_file():
        raise ValueError(f"missing paired evidence for {case_id}")
    raw = read_json(evidence)
    if raw.get("id") != case_id:
        raise ValueError(f"paired case identity differs for {case_id}")
    if row["seals"] != expected_seals:
        kind = "Python native" if suite == "python" else f"{suite} Rust binary"
        raise ValueError(f"{kind} seal differs for {case_id}")
    if evaluate_case(raw, expected_seals) != row["result"]:
        raise ValueError(f"paired verdict differs from raw evidence for {case_id}")
    return {**row, "evidence": str(Path("suites") / source.name / relative)}


def _merge_case_rows(
    source: Path, suite: str, collected: dict, names: list[str]
) -> list[dict]:
    prefix = "python/" if suite == "python" else f"rust/{suite}/"
    expected_ids = {f"{prefix}{name}" for name in names}
    expected_seals = _expected_case_seals(collected, suite)
    seen = set()
    rows = []
    for row in collected.get("cases", []):
        case_id = row["id"]
        if case_id not in expected_ids or case_id in seen:
            raise ValueError(f"unexpected or duplicate {suite} case: {case_id}")
        seen.add(case_id)
        rows.append(_verified_case(source, suite, row, expected_seals))
    if seen != expected_ids and not collected.get("errors"):
        raise ValueError(f"incomplete {suite} case results")
    return rows


def _merge_python_gates(report: dict, collected: dict) -> None:
    report["allocation"] = collected.get("allocation", {})
    report["lifecycle_regressions"] = collected.get("lifecycle_regressions", [])
    if not collected.get("errors") and set(report["allocation"]) != set(SIDES):
        raise ValueError("missing Python suite allocation gate")


def _append_suite(
    report: dict, source: Path, suite: str, collected: dict, shas: dict
) -> None:
    if not collected.get("errors") or collected.get("cases"):
        _verify_suite_builds(source, collected, suite, shas)
    report["errors"].extend(
        f"{suite}: {error}" for error in collected.get("errors", [])
    )
    inventory = _suite_inventory(collected, suite)
    if inventory is None:
        report["errors"].append(f"{suite}: missing case inventory")
        return
    report["inventory"][suite], names = inventory
    if collected.get("cases") or not collected.get("errors"):
        report["cases"].extend(_merge_case_rows(source, suite, collected, names))
    if suite == "python":
        _merge_python_gates(report, collected)


def merge_reports(sources: Path) -> dict:
    """Verify and combine the independently collected release suite evidence."""
    report = {
        "contract": "release-paired-v1",
        "suite": "all",
        "cases": [],
        "errors": [],
        "inventory": {},
        "harness": harness_identity(),
        "dependency_lock_sha256": sha256_file(ROOT / "benchmarks/requirements.lock"),
    }
    for suite in SUITES:
        source = sources / f"release-performance-{suite}"
        collected = _load_suite_report(
            source, suite, report["harness"], report["dependency_lock_sha256"]
        )
        shas = _suite_shas(collected, suite)
        if shas is None:
            report["errors"].extend(
                f"{suite}: {error}" for error in collected["errors"]
            )
            continue
        _check_release_set(report, collected, shas, suite)
        _append_suite(report, source, suite, collected, shas)
    return report


def run_merge(sources: Path, output: Path) -> int:
    output.mkdir(parents=True, exist_ok=True)
    try:
        report = merge_reports(sources)
    except Exception as error:
        report = {
            "contract": "release-paired-v1",
            "suite": "all",
            "cases": [],
            "errors": [f"{type(error).__name__}: {error}"],
        }
    write_json(output / "results.json", report)
    summary = _gate_summary(report)
    (output / "summary.md").write_text(summary, encoding="utf-8")
    print(summary)
    return _gate_exit_code(report)


def acceptance_summary(
    results: dict, steps: tuple[str, ...] = ("performance", "security", "soak")
) -> str:
    lines, failed = [], []
    for step in steps:
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
    parser.add_argument("--suite", choices=SUITES)
    parser.add_argument("--merge-suites", type=Path)
    parser.add_argument(
        "--allow-dependency-drift",
        action="store_true",
        help="Record acknowledgement; incompatible dependencies still block release",
    )
    parser.add_argument("--acceptance-summary", action="store_true")
    parser.add_argument(
        "--steps",
        default="performance,security,soak",
        help="comma-separated acceptance step ids owned by this job",
    )
    options = parser.parse_args()
    if options.acceptance_summary:
        options.output.mkdir(parents=True, exist_ok=True)
        results = json.loads(os.environ["ACCEPTANCE_STEPS"])
        write_json(options.output / "acceptance.json", results)
        summary = acceptance_summary(results, steps=tuple(options.steps.split(",")))
        (options.output / "acceptance.md").write_text(summary, encoding="utf-8")
        with Path(os.environ["GITHUB_STEP_SUMMARY"]).open(
            "a", encoding="utf-8"
        ) as handle:
            handle.write(summary)
        return 0
    if options.merge_suites is not None:
        return run_merge(options.merge_suites, options.output)
    if options.baseline_source is None:
        parser.error("--baseline-source is required for measurement")
    return asyncio.run(run_gate(options))


if __name__ == "__main__":
    raise SystemExit(main())
