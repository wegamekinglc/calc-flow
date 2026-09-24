"""Discover and retain every Rust bench target, including specialized reports."""

from __future__ import annotations

import hashlib
import json
import shutil
import tomllib
from pathlib import Path

from scripts.benchmark_suite.asof import asof_rows
from scripts.benchmark_suite.catalog import CONTRACT
from scripts.benchmark_suite.join_materialization import materialization_rows
from scripts.benchmark_suite.legacy import combine_blocks
from scripts.benchmark_suite.migrations import declared_migrations, load_migrations
from scripts.benchmark_suite.normalize import criterion_rows, read_json
from scripts.benchmark_suite.process import ROOT, child_environment, command
from scripts.benchmark_suite.provenance import harness_sha256
from scripts.benchmark_suite.rust_provenance import (
    target_dependency_fingerprint,
    with_compiled_dependencies,
)
from scripts.verify_sql_datafusion_performance import verify_report
from scripts.write_criterion_provenance import build_provenance


def clear_stale_bench_binary(shared: Path, target: str) -> None:
    """Remove restored bench executables so each side links its own source.

    The shared target directory is cached across runs and hosts builds from
    both source trees. A restored executable whose unit hash collides with
    this side's can otherwise be re-selected as fresh, silently measuring the
    other revision's benchmark.
    """

    for stale in shared.glob(f"release/deps/{target}-*"):
        stale.unlink()


def bench_targets(source: Path) -> list[str]:
    manifest = tomllib.loads(
        (source / "crates/calc-flow/Cargo.toml").read_text(encoding="utf-8")
    )
    targets = [entry["name"] for entry in manifest["bench"]]
    if not targets or len(targets) != len(set(targets)):
        raise ValueError("invalid Rust benchmark inventory")
    return targets


def clear_stale_product_library(shared: Path) -> None:
    # Cargo unit hashes can collide across worktrees with different product code.
    for suffix in ("rlib", "rmeta"):
        for stale in shared.glob(f"release/deps/libcalc_flow-*.{suffix}"):
            stale.unlink()


async def build_binaries(
    source: Path, output: Path, shared: Path, *, targets: tuple[str, ...] | None = None
) -> dict:
    targets = targets if targets is not None else bench_targets(source)
    clear_stale_product_library(shared)
    environment = {
        **child_environment(),
        "CARGO_TARGET_DIR": str(shared),
        "CARGO_BUILD_JOBS": "2",
        "CARGO_INCREMENTAL": "0",
    }
    binaries = {}
    for target in targets:
        log = output / f"build-{target}.jsonl"
        clear_stale_bench_binary(shared, target)
        # The sub-nanosecond core getter benchmark changes when its loop crosses
        # a 64-byte instruction-cache line. Align only the bench target's loops;
        # cargo rustc passes the extra codegen flag to that target, not its deps.
        argv = (
            [
                "cargo",
                "rustc",
                "--locked",
                "--profile",
                "bench",
                "-p",
                "calc-flow",
                "--bench",
                target,
                "--message-format=json",
                "--",
                "-C",
                "llvm-args=--align-loops=64",
            ]
            if target == "core"
            else [
                "cargo",
                "bench",
                "--locked",
                "-p",
                "calc-flow",
                "--bench",
                target,
                "--no-run",
                "--message-format=json",
            ]
        )
        await command(
            argv,
            cwd=source,
            log=log,
            env=environment,
        )
        destination = output / target
        shutil.copy2(_compiled_executable(log, target), destination)
        binaries[target] = destination
    return binaries


def _compiled_executable(log: Path, target: str) -> str:
    artifacts = []
    for line in log.read_text(encoding="utf-8").splitlines():
        if not line.startswith("{"):
            continue
        item = json.loads(line)
        if (
            item.get("reason") == "compiler-artifact"
            and item["target"]["name"] == target
            and item.get("executable")
        ):
            artifacts.append(item["executable"])
    if len(artifacts) != 1:
        raise ValueError(f"expected one compiled executable for {target}")
    return artifacts[0]


SQL_MINIMUM_SAMPLES = 20


def _require_baseline_samples(
    case_name: object, engine_name: str, samples: object
) -> list[float]:
    if not isinstance(samples, list):
        raise ValueError(f"baseline {case_name}/{engine_name} samples are incomplete")
    if len(samples) < SQL_MINIMUM_SAMPLES or any(
        isinstance(value, bool) or not isinstance(value, int | float) or value <= 0
        for value in samples
    ):
        raise ValueError(f"baseline {case_name}/{engine_name} samples are incomplete")
    return samples


def _baseline_report_rows(report: dict) -> dict:
    """Read one frozen baseline report under its own legacy field contract.

    The baseline evidence was verified under its contemporary contract when
    that revision was the candidate; this comparison consumes only per-case
    samples, so an older field set (for example before ``output_rows``)
    stays readable without weakening the candidate verifier.
    """

    if not isinstance(report.get("cases"), list) or not report["cases"]:
        raise ValueError("baseline sql report has no cases")
    rows = {}
    for case in report["cases"]:
        for engine in ("calc_flow", "raw_datafusion"):
            samples = _require_baseline_samples(
                case.get("name"), engine, case.get(engine, {}).get("samples_ms")
            )
            rows[f"sql_datafusion_performance/{case['name']}/{engine}"] = {
                "samples": [value / 1000 for value in samples],
                "rows": case["rows"],
                "scope": "native-sql-paired-boundary",
                "metadata": {
                    "environment": report["environment"],
                    "correctness": case["correctness"],
                    "engine": case[engine],
                },
            }
    return rows


def sql_rows(path: Path, *, side: str) -> dict:
    report = read_json(path)
    if side == "baseline":
        return _baseline_report_rows(report)
    verify_report(report, minimum_samples=SQL_MINIMUM_SAMPLES, require_stable=False)
    return {
        f"sql_datafusion_performance/{case['name']}/{engine}": {
            "samples": [value / 1000 for value in case[engine]["samples_ms"]],
            "rows": case["rows"],
            "scope": "native-sql-paired-boundary",
            "metadata": {
                "environment": report["environment"],
                "correctness": case["correctness"],
                "engine": case[engine],
            },
        }
        for case in report["cases"]
        for engine in ("calc_flow", "raw_datafusion")
    }


async def run_binary(
    target: str, binary: Path, source: Path, output: Path, side: str
) -> dict:
    environment = {**child_environment(), "CRITERION_HOME": str(output / "criterion")}
    if target == "stream_asof_perf":
        path = output / "asof.json"
        await command(
            [str(binary), "--output", str(path)],
            cwd=source,
            log=output / "run.log",
            env=environment,
        )
        return asof_rows(path)
    if target == "stream_join_materialization":
        path = output / "materialization.json"
        await command(
            [str(binary), "--output", str(path)],
            cwd=source,
            log=output / "run.log",
            env=environment,
        )
        return materialization_rows(path)
    if target == "sql_datafusion_performance":
        path = output / "sql.json"
        await command(
            [
                str(binary),
                "--profile",
                "matched-adaptive",
                "--samples",
                "20",
                "--warmups",
                "1",
                "--output",
                str(path),
            ],
            cwd=source,
            log=output / "run.log",
            env=environment,
        )
        return sql_rows(path, side=side)
    await command(
        [str(binary), "--bench"], cwd=source, log=output / "run.log", env=environment
    )
    return {
        f"{target}/{name}": row
        for name, row in criterion_rows(output / "criterion").items()
    }


async def allocation(binary: Path, source: Path, output: Path, side: str) -> dict:
    path = output / "allocation.json"
    role = "baseline" if side == "baseline" else "candidate"
    await command(
        [
            str(binary),
            "--warmup-dispatches",
            "1000",
            "--measured-dispatches",
            "10000",
            "--repetitions",
            "10",
            "--cases",
            "all-existing-data",
            "--role",
            role,
            "--output",
            str(path),
        ],
        cwd=source,
        log=output / "allocation.log",
        env=child_environment(),
    )
    report = read_json(path)
    if report["valid"] is not True or not report["cases"]:
        raise ValueError("allocation evidence is invalid or empty")
    return report


def _validate_allocation_report(side: str, report: dict) -> None:
    names = [case["name"] for case in report["cases"]]
    if report["role"] != side or report["valid"] is not True:
        raise ValueError("invalid allocation report or version role")
    if not names or len(names) != len(set(names)):
        raise ValueError("empty or duplicate allocation case inventory")


def _allocation_metric(name: str, metric: str, sides: dict) -> dict:
    values = {
        side: [rep["normalized"][metric] for rep in cases[name]["repetitions"]]
        for side, cases in sides.items()
    }
    return {
        "id": f"rust/allocation/{name}/{metric}",
        "family": "rust",
        "backend": "calc-flow",
        "scenario": name,
        "scope": "allocation-counter",
        "status": "ok",
        "kind": "metric",
        "metric": metric,
        "baseline_value": min(values["baseline"]),
        "candidate_value": min(values["candidate"]),
        "correctness": True,
    }


def _allocation_case_index(reports: dict) -> dict:
    for side, report in reports.items():
        _validate_allocation_report(side, report)
    sides = {
        side: {case["name"]: case for case in report["cases"]}
        for side, report in reports.items()
    }
    if set(sides["baseline"]) != set(sides["candidate"]):
        raise ValueError("allocation case sets differ")
    return sides


def allocation_rows(reports: dict) -> list[dict]:
    sides = _allocation_case_index(reports)
    rows = []
    for name, case in sides["candidate"].items():
        if not case["valid"] or not sides["baseline"][name]["valid"]:
            raise ValueError(f"invalid allocation case {name}")
        for metric in ("calls_per_dispatch", "bytes_per_dispatch"):
            rows.append(_allocation_metric(name, metric, sides))
    return rows


def _rust_provenance(roots: dict, output: Path) -> dict:
    return {
        side: with_compiled_dependencies(
            build_provenance(
                source,
                [
                    Path(f"crates/calc-flow/benches/{name}.rs")
                    for name in bench_targets(source)
                ],
            ),
            source,
            {
                name: output / side / f"build-{name}.jsonl"
                for name in bench_targets(source)
            },
        )
        for side, source in roots.items()
    }


async def measure_rust(shard: dict, releases: dict, roots: dict, output: Path) -> dict:
    shared = ROOT / "target/benchmark-rust-build"
    binaries = {
        side: await build_binaries(source, output / side, shared)
        for side, source in roots.items()
    }
    provenance = _rust_provenance(roots, output)
    if set(binaries["baseline"]) - set(binaries["candidate"]):
        raise ValueError(
            "Rust benchmark targets removed; an explicit migration is required"
        )
    blocks = {side: [] for side in roots}
    errors = []
    applied = declared_migrations(provenance, load_migrations(ROOT))
    stamps = _stamp_fingerprints(provenance, applied)
    for index, side in enumerate(("baseline", "candidate", "candidate", "baseline")):
        block, failures = await _rust_block(
            binaries[side],
            roots[side],
            output / f"block-{index}-{side}",
            stamps[side],
            side,
        )
        errors.extend(f"{side}/{error}" for error in failures)
        blocks[side].append(block)
        (output / "blocks.json").write_text(
            json.dumps(blocks, indent=2) + "\n", encoding="utf-8"
        )
    cases = combine_blocks(shard, blocks)
    try:
        reports = await _allocation_reports(binaries, roots, output)
        cases.extend(allocation_rows(reports))
    except Exception as error:
        errors.append(f"allocation_regression: {error}")
    return {
        "contract": CONTRACT,
        "harness_sha256": harness_sha256(),
        "provenance": provenance,
        "binary_sha256": _binary_hashes(binaries),
        "shard": shard,
        "releases": releases,
        "cases": cases,
        "errors": errors,
        "expected_case_ids": [case["id"] for case in cases],
        "workload_migrations": sorted(
            applied.values(), key=lambda entry: entry["target"]
        ),
    }


async def _allocation_reports(binaries: dict, roots: dict, output: Path) -> dict:
    return {
        side: await allocation(
            values["allocation_regression"], roots[side], output / side, side
        )
        for side, values in binaries.items()
    }


def _binary_hashes(binaries: dict) -> dict:
    return {
        side: {
            name: hashlib.sha256(path.read_bytes()).hexdigest()
            for name, path in values.items()
        }
        for side, values in binaries.items()
    }


def _with_fingerprints(measured: dict, fingerprints: dict) -> dict:
    return {
        name: {**row, "metadata": {**row["metadata"], **fingerprints}}
        for name, row in measured.items()
    }


def _migration_marker(declared: dict | None) -> dict:
    """Mark rows accepted through one declared migration with its reference."""
    return {} if declared is None else {"workload_migration": declared["reference"]}


def _stamp_fingerprints(provenance: dict, applied: dict) -> dict:
    """Per-side, per-target fingerprint metadata for row stamping.

    Each target's rows carry its own scoped workload fingerprint, so a changed
    bench source invalidates only that target; a declared migration re-baselines
    the stamp to the candidate identity and records its reference instead.
    """
    candidate_scoped = provenance["candidate"]["scoped_workload_fingerprints"]
    return {
        side: {
            target: {
                "machine_fingerprint": identity["machine_fingerprint"],
                "dependency_fingerprint": target_dependency_fingerprint(
                    identity, target
                ),
                "workload_fingerprint": (
                    candidate_scoped[target] if target in applied else scoped
                ),
                **_migration_marker(applied.get(target)),
            }
            for target, scoped in identity["scoped_workload_fingerprints"].items()
        }
        for side, identity in provenance.items()
    }


async def _rust_block(
    binaries: dict, source: Path, output: Path, stamps: dict, side: str
) -> tuple[dict, list[str]]:
    block, errors = {}, []
    for target, binary in binaries.items():
        if target == "allocation_regression":
            continue
        try:
            measured = await run_binary(target, binary, source, output / target, side)
            block.update(_with_fingerprints(measured, stamps[target]))
        except Exception as error:
            errors.append(f"{target}: {error}")
    return block, errors
