"""Discover and retain every Rust bench target, including specialized reports."""

from __future__ import annotations

import hashlib
import json
import shutil
import tomllib
from pathlib import Path

from scripts.benchmark_suite.catalog import CONTRACT
from scripts.benchmark_suite.legacy import combine_blocks
from scripts.benchmark_suite.normalize import criterion_rows, read_json
from scripts.benchmark_suite.process import ROOT, child_environment, command
from scripts.benchmark_suite.provenance import harness_sha256
from scripts.verify_sql_datafusion_performance import verify_report
from scripts.write_criterion_provenance import build_provenance


def bench_targets(source: Path) -> list[str]:
    manifest = tomllib.loads(
        (source / "crates/calc-flow/Cargo.toml").read_text(encoding="utf-8")
    )
    targets = [entry["name"] for entry in manifest["bench"]]
    if not targets or len(targets) != len(set(targets)):
        raise ValueError("invalid Rust benchmark inventory")
    return targets


async def build_binaries(source: Path, output: Path, shared: Path) -> dict:
    targets = bench_targets(source)
    environment = {
        **child_environment(),
        "CARGO_TARGET_DIR": str(shared),
        "CARGO_BUILD_JOBS": "2",
        "CARGO_INCREMENTAL": "0",
    }
    binaries = {}
    for target in targets:
        log = output / f"build-{target}.jsonl"
        await command(
            [
                "cargo",
                "bench",
                "--locked",
                "-p",
                "calc-flow",
                "--bench",
                target,
                "--no-run",
                "--message-format=json",
            ],
            cwd=source,
            log=log,
            env=environment,
        )
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
        destination = output / target
        shutil.copy2(artifacts[0], destination)
        binaries[target] = destination
    return binaries


def sql_rows(path: Path) -> dict:
    report = read_json(path)
    verify_report(report, minimum_samples=20, require_stable=False)
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


async def run_binary(target: str, binary: Path, source: Path, output: Path) -> dict:
    environment = {**child_environment(), "CRITERION_HOME": str(output / "criterion")}
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
        return sql_rows(path)
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


def allocation_rows(reports: dict) -> list[dict]:
    for side, report in reports.items():
        names = [case["name"] for case in report["cases"]]
        if report["role"] != side or report["valid"] is not True:
            raise ValueError("invalid allocation report or version role")
        if not names or len(names) != len(set(names)):
            raise ValueError("empty or duplicate allocation case inventory")
    sides = {
        side: {case["name"]: case for case in report["cases"]}
        for side, report in reports.items()
    }
    if set(sides["baseline"]) != set(sides["candidate"]):
        raise ValueError("allocation case sets differ")
    rows = []
    for name, case in sides["candidate"].items():
        if not case["valid"] or not sides["baseline"][name]["valid"]:
            raise ValueError(f"invalid allocation case {name}")
        for metric in ("calls_per_dispatch", "bytes_per_dispatch"):
            values = {
                side: [rep["normalized"][metric] for rep in cases[name]["repetitions"]]
                for side, cases in sides.items()
            }
            rows.append(
                {
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
            )
    return rows


async def measure_rust(shard: dict, releases: dict, roots: dict, output: Path) -> dict:
    shared = ROOT / "target/benchmark-rust-build"
    provenance = {
        side: build_provenance(
            source,
            [
                Path(f"crates/calc-flow/benches/{name}.rs")
                for name in bench_targets(source)
            ],
        )
        for side, source in roots.items()
    }
    binaries = {
        side: await build_binaries(source, output / side, shared)
        for side, source in roots.items()
    }
    if set(binaries["baseline"]) != set(binaries["candidate"]):
        raise ValueError(
            "Rust benchmark targets changed; an explicit migration is required"
        )
    blocks = {side: [] for side in roots}
    errors = []
    for index, side in enumerate(("baseline", "candidate", "candidate", "baseline")):
        block = {}
        for target, binary in binaries[side].items():
            if target == "allocation_regression":
                continue
            destination = output / f"block-{index}-{side}" / target
            try:
                measured = await run_binary(target, binary, roots[side], destination)
                identity = provenance[side]
                fingerprints = {
                    key: identity[key]
                    for key in (
                        "machine_fingerprint",
                        "dependency_fingerprint",
                        "workload_fingerprint",
                    )
                }
                block.update(
                    {
                        name: {**row, "metadata": {**row["metadata"], **fingerprints}}
                        for name, row in measured.items()
                    }
                )
            except Exception as error:
                errors.append(f"{side}/{target}: {error}")
        blocks[side].append(block)
        (output / "blocks.json").write_text(
            json.dumps(blocks, indent=2) + "\n", encoding="utf-8"
        )
    cases = combine_blocks(shard, blocks)
    try:
        reports = {
            side: await allocation(
                values["allocation_regression"], roots[side], output / side, side
            )
            for side, values in binaries.items()
        }
        cases.extend(allocation_rows(reports))
    except Exception as error:
        errors.append(f"allocation_regression: {error}")
    return {
        "contract": CONTRACT,
        "harness_sha256": harness_sha256(),
        "provenance": provenance,
        "binary_sha256": {
            side: {
                name: hashlib.sha256(path.read_bytes()).hexdigest()
                for name, path in values.items()
            }
            for side, values in binaries.items()
        },
        "shard": shard,
        "releases": releases,
        "cases": cases,
        "errors": errors,
        "expected_case_ids": [case["id"] for case in cases],
    }
