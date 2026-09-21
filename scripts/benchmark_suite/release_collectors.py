"""Run a single release case per subprocess, preserving its native timing boundary."""

from __future__ import annotations

import sys
from pathlib import Path

from scripts.benchmark_suite.identity import validate_identity
from scripts.benchmark_suite.normalize import criterion_rows, pytest_rows, read_json
from scripts.benchmark_suite.process import ROOT, child_environment, command
from scripts.toolkit import fingerprint_json, sha256_file


def python_environment(site: Path, native: str, output: Path) -> dict:
    return {
        **child_environment(site),
        "CALC_FLOW_BENCHMARK_SCALE": "overhead",
        "CALC_FLOW_RELEASE_NATIVE": native,
        "CALC_FLOW_RELEASE_OBSERVED": str(output / "native.json"),
        "CALC_FLOW_SUITE_INVENTORY": str(output / "inventory.json"),
    }


def pytest_command(arguments: list[str], output: Path) -> list[str]:
    return [
        sys.executable,
        "-m",
        "pytest",
        *arguments,
        "-q",
        "-p",
        "scripts.benchmark_suite.release_pytest",
        "--benchmark-only",
        "--benchmark-save-data",
        f"--benchmark-json={output / 'pytest.json'}",
    ]


async def python_inventory(site: Path, native: str, output: Path) -> list[str]:
    output.mkdir(parents=True, exist_ok=True)
    await command(
        pytest_command(
            ["benchmarks", "-m", "not stream_lifecycle", "--collect-only"], output
        ),
        cwd=ROOT,
        log=output / "collect.log",
        env=python_environment(site, native, output),
    )
    names = read_json(output / "inventory.json")
    if not names or len(names) != len(set(names)):
        raise ValueError("empty or duplicate release Python inventory")
    return sorted(names)


async def python_observation(name: str, site: Path, native: str, output: Path) -> dict:
    output.mkdir(parents=True, exist_ok=True)
    await command(
        pytest_command([name], output),
        cwd=ROOT,
        log=output / "run.log",
        env=python_environment(site, native, output),
    )
    raw = read_json(output / "pytest.json")
    if any(not row.get("stats", {}).get("data") for row in raw["benchmarks"]):
        raise ValueError("saved Python raw samples are required")
    rows = pytest_rows(output / "pytest.json")
    if set(rows) != {name}:
        raise ValueError(
            "release Python invocation must produce exactly its selected case"
        )
    row = rows[name]
    validate_identity(row["metadata"])
    observed = read_json(output / "native.json")["native_sha256"]
    if observed != native:
        raise ValueError("wrong sealed native SHA in Python observation")
    return {
        "samples": row["samples"],
        "metadata": row["metadata"],
        "binary_sha256": observed,
        "worker": str(output),
        "correctness": True,
    }


async def rust_inventory(binary: Path, source: Path, output: Path) -> list[str]:
    await command(
        [str(binary), "--list"],
        cwd=source,
        log=output / "inventory.log",
        env=child_environment(),
    )
    names = [
        line.removesuffix(": benchmark")
        for line in (output / "inventory.log").read_text(encoding="utf-8").splitlines()
        if line.endswith(": benchmark")
    ]
    if not names or len(names) != len(set(names)):
        raise ValueError("empty or duplicate Criterion inventory")
    return sorted(names)


async def rust_observation(
    name: str, binary: Path, source: Path, identity: dict, output: Path
) -> dict:
    validate_identity(identity)
    await command(
        [str(binary), name, "--exact", "--bench"],
        cwd=source,
        log=output / "run.log",
        env={**child_environment(), "CRITERION_HOME": str(output / "criterion")},
    )
    rows = criterion_rows(output / "criterion")
    if set(rows) != {name}:
        raise ValueError(
            "release Criterion invocation must produce exactly its selected case"
        )
    workload = {
        **identity["workload_identity"],
        "case": name,
        "scope": "criterion-native-boundary",
        "descriptor": rows[name]["metadata"],
    }
    metadata = {
        **identity,
        "workload_identity": workload,
        "workload_fingerprint": fingerprint_json(workload),
    }
    return {
        "samples": rows[name]["samples"],
        "metadata": metadata,
        "binary_sha256": sha256_file(binary),
        "worker": str(output),
        "correctness": True,
    }
