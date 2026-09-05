"""Preserve existing fixture boundaries; compare whole-suite blocks honestly."""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import sys
from pathlib import Path

from scripts.benchmark_suite.catalog import CONTRACT
from scripts.benchmark_suite.normalize import pytest_rows, read_json, vitest_rows
from scripts.benchmark_suite.process import ROOT, child_environment, command, install
from scripts.benchmark_suite.provenance import harness_sha256
from scripts.benchmark_suite.report import comparison, validate_shards


def pytest_arguments(family: str) -> list[str]:
    if family == "studio":
        return ["web-ui/backend/benchmarks/test_performance.py"]
    if family == "lifecycle":
        return [
            "benchmarks/test_symbolic_baseline.py::test_stream_window_checkpoint_and_recovery"
        ]
    return ["benchmarks", "-m", "not stream_lifecycle"]


async def _pytest_run(shard: dict, source: Path, site: Path, output: Path) -> dict:
    environment = {
        **child_environment(site),
        "PYTHONPATH": os.pathsep.join(
            str(path) for path in (site, source, ROOT, source / "web-ui/backend/src")
        ),
        "CALC_FLOW_BENCHMARK_SCALE": shard.get("scale", "standard"),
        "CALC_FLOW_SUITE_INVENTORY": str(output / "inventory.json"),
    }
    await command(
        [
            sys.executable,
            "-m",
            "pytest",
            *pytest_arguments(shard["family"]),
            "-q",
            "-p",
            "scripts.benchmark_suite.pytest_plugin",
            "--benchmark-only",
            "--benchmark-save-data",
            f"--benchmark-json={output / 'pytest.json'}",
        ],
        cwd=source,
        log=output / "run.log",
        env=environment,
    )
    rows = pytest_rows(output / "pytest.json")
    validate_shards(read_json(output / "inventory.json"), list(rows))
    benchmark_root = (
        source / "web-ui/backend" if shard["family"] == "studio" else source
    )
    rows = {
        name: {
            **row,
            "metadata": {
                **row["metadata"],
                "benchmark_source_sha256": hashlib.sha256(
                    (benchmark_root / name.split("::", 1)[0]).read_bytes()
                ).hexdigest(),
            },
        }
        for name, row in rows.items()
    }
    if shard["family"] == "lifecycle":
        await command(
            [
                sys.executable,
                str(ROOT / "scripts/verify_stream_lifecycle_evidence.py"),
                str(output / "pytest.json"),
                "--minimum-rounds",
                "20",
            ],
            cwd=source,
            log=output / "verify.log",
            env=environment,
        )
    return rows


async def _frontend_run(source: Path, output: Path) -> dict:
    frontend = source / "web-ui"
    runner = frontend / "node_modules/.cache/calc-flow-benchmark.mjs"
    runner.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(ROOT / "scripts/benchmark_suite/frontend.mjs", runner)
    await command(
        ["node", str(runner)],
        cwd=frontend,
        log=output / "run.log",
        env=child_environment(),
    )
    shutil.copyfile(
        source / "target/benchmark-suite/vitest.json", output / "vitest.json"
    )
    rows = vitest_rows(output / "vitest.json")
    lock = hashlib.sha256(
        (source / "web-ui/package-lock.json").read_bytes()
    ).hexdigest()
    return {
        name: {**row, "metadata": {**row["metadata"], "dependency_fingerprint": lock}}
        for name, row in rows.items()
    }


async def _setup(shard: dict, releases: dict, roots: dict, output: Path) -> dict:
    sites = {}
    for side in ("baseline", "candidate"):
        sites[side] = await install(releases[side], output / side)
        if shard["family"] == "frontend":
            await command(
                ["npm", "ci"],
                cwd=roots[side] / "web-ui",
                log=output / side / "npm-install.log",
                env=child_environment(),
            )
    return sites


def combine_blocks(shard: dict, blocks: dict) -> list[dict]:
    available = [block for values in blocks.values() for block in values]
    names = sorted({name for block in available for name in block})
    result = []
    for name in names:
        descriptor = _latest_descriptor(name, available)
        result.append(_block_row(shard, name, descriptor, blocks))
    if not result:
        raise ValueError("legacy suite produced no cases")
    return result


def _latest_descriptor(name: str, available: list[dict]) -> dict:
    return next(block[name] for block in reversed(available) if name in block)


def _block_row(shard: dict, name: str, descriptor: dict, blocks: dict) -> dict:
    row = {
        "id": f"{shard['id']}/{name}",
        "family": shard["family"],
        "backend": "calc-flow",
        "scenario": name,
        "rows": descriptor["rows"],
        "scope": descriptor["scope"],
        "metadata": descriptor["metadata"],
    }
    problem = block_problem(name, blocks)
    if problem:
        return {**row, "status": "error", "error": problem}
    row = {
        **row,
        "status": "ok",
        "correctness": True,
        "comparison": "suite-blocks",
        **{
            side: [block[name]["samples"] for block in values]
            for side, values in blocks.items()
        },
    }
    return {**row, "result": comparison(row)}


def _missing_block(name: str, blocks: dict) -> bool:
    return any(
        len(values) != 2 or any(name not in block for block in values)
        for values in blocks.values()
    )


def block_problem(name: str, blocks: dict) -> str | None:
    if _missing_block(name, blocks):
        return "benchmark missing from a base/head confirmation block"
    rows = [block[name] for values in blocks.values() for block in values]
    for key in ("rows", "scope"):
        if any(row[key] != rows[0][key] for row in rows):
            return f"benchmark {key} changed; no timing classification"
    return _fingerprint_problem([row["metadata"] for row in rows])


def _fingerprint_problem(metadata: list[dict]) -> str | None:
    for key in (
        "machine_fingerprint",
        "dependency_fingerprint",
        "workload_fingerprint",
        "benchmark_source_sha256",
    ):
        if any(row.get(key) != metadata[0].get(key) for row in metadata):
            return f"benchmark {key} changed; no timing classification"
    return None


async def validate_sources(roots: dict, releases: dict) -> None:
    from scripts.profile_warm_stream import source_identity

    for side, root in roots.items():
        identity = await source_identity(root)
        if identity["git_clean"] is not True or any(
            identity[key] != releases[side][key]
            for key in ("git_sha", "source_sha256", "cargo_lock_sha256")
        ):
            raise ValueError(f"{side} source checkout differs from the sealed release")


async def observe_workers(sites: dict, releases: dict, output: Path) -> dict:
    from scripts.benchmark_suite.measure import validate_environment
    from scripts.benchmark_suite.process import Worker

    observed = {}
    for side, site in sites.items():
        worker = await Worker.start(site, output / side / "identity")
        try:
            observed[side] = validate_environment(
                await worker.request(operation="hello"), releases[side]
            )
        finally:
            await worker.close()
    if observed["baseline"] != observed["candidate"]:
        raise ValueError("legacy worker environments differ")
    return observed["candidate"]


async def measure_legacy(
    shard: dict, releases: dict, output: Path, baseline: Path | None
) -> dict:
    if baseline is None:
        raise ValueError("legacy comparison requires a baseline source checkout")
    roots = {"baseline": baseline.resolve(), "candidate": ROOT}
    await validate_sources(roots, releases)
    if shard["family"] == "rust":
        from scripts.benchmark_suite.rust import measure_rust

        return await measure_rust(shard, releases, roots, output)
    sites = await _setup(shard, releases, roots, output)
    environment = await observe_workers(sites, releases, output)
    blocks = {"baseline": [], "candidate": []}
    report = {
        "contract": CONTRACT,
        "harness_sha256": harness_sha256(),
        "environment": environment,
        "shard": shard,
        "releases": releases,
        "errors": [],
        "cases": [],
    }
    for index, side in enumerate(("baseline", "candidate", "candidate", "baseline")):
        destination = output / f"block-{index}-{side}"
        destination.mkdir(parents=True, exist_ok=True)
        try:
            block = (
                await _frontend_run(roots[side], destination)
                if shard["family"] == "frontend"
                else await _pytest_run(shard, roots[side], sites[side], destination)
            )
            blocks[side].append(block)
        except Exception as error:
            report["errors"].append(f"{side} block {index}: {error}")
            blocks[side].append({})
        (output / "blocks.json").write_text(
            json.dumps(blocks, indent=2) + "\n", encoding="utf-8"
        )
    report["cases"] = combine_blocks(shard, blocks)
    report["expected_case_ids"] = [case["id"] for case in report["cases"]]
    return report
