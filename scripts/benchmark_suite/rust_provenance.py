"""Identify dependencies actually compiled by the Rust benchmark commands."""

from __future__ import annotations

import hashlib
import json
import tomllib
from pathlib import Path


def _encoded(value: object) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"))


def _locked_packages(root: Path) -> dict:
    lock = tomllib.loads((root / "Cargo.lock").read_text(encoding="utf-8"))
    return {
        f"{item['source']}#{item['name']}@{item['version']}": {
            key: item[key] for key in ("name", "version", "source", "checksum")
        }
        for item in lock["package"]
        if item.get("source", "").startswith("registry+")
    }


def _package_identity(root: Path, artifact: dict, locked: dict) -> dict:
    manifest = Path(artifact["manifest_path"])
    if manifest == root / "crates/calc-flow/Cargo.toml":
        return {"workspace_package": "crates/calc-flow"}
    package = locked.get(artifact["package_id"])
    if package is None:
        raise ValueError(f"unrecognized compiled dependency: {artifact['package_id']}")
    return package


def _artifact_identity(root: Path, artifact: dict, locked: dict) -> dict:
    return {
        "package": _package_identity(root, artifact, locked),
        "target": {
            key: artifact["target"][key]
            for key in ("name", "kind", "crate_types", "edition")
        },
        "features": sorted(artifact["features"]),
        "profile": artifact["profile"],
    }


def _build_identity(root: Path, name: str, log: Path, locked: dict) -> list[dict]:
    messages = [
        json.loads(line)
        for line in log.read_text(encoding="utf-8").splitlines()
        if line.startswith("{")
    ]
    finished = [row for row in messages if row.get("reason") == "build-finished"]
    if len(finished) != 1 or finished[0]["success"] is not True:
        raise ValueError(f"missing successful build completion: {name}")
    artifacts = [
        _artifact_identity(root, row, locked)
        for row in messages
        if row.get("reason") == "compiler-artifact"
    ]
    _validate_artifacts(name, artifacts)
    return [json.loads(row) for row in sorted({_encoded(row) for row in artifacts})]


def _validate_artifacts(name: str, artifacts: list[dict]) -> None:
    benchmarks = [
        row
        for row in artifacts
        if row["target"]["kind"] == ["bench"]
        and row["target"]["name"] == name
        and row["package"].get("workspace_package") == "crates/calc-flow"
    ]
    dependencies = [row for row in artifacts if "source" in row["package"]]
    if len(benchmarks) != 1 or not dependencies:
        raise ValueError(f"incomplete compiled dependency inventory: {name}")


def compiled_dependencies(root: Path, logs: dict[str, Path]) -> dict:
    """Retain package checksums, enabled features and profiles per bench build.

    Cargo's compiler-artifact messages cover fresh and cached builds alike.
    Only the measured core package is normalized across checkout locations;
    unsupported dependency sources fail closed pending an explicit migration.
    """
    if not logs:
        raise ValueError("missing Rust benchmark build logs")
    locked = _locked_packages(root)
    return {
        name: _build_identity(root, name, log, locked)
        for name, log in sorted(logs.items())
    }


def with_compiled_dependencies(identity: dict, root: Path, logs: dict) -> dict:
    """Preserve full-lock provenance and add the compiled comparison identity."""
    dependency_identity = {
        "schema": "calc-flow.compiled-benchmark-dependencies.v1",
        "rustc": identity["dependency_identity"]["rustc"],
        "cargo": identity["dependency_identity"]["cargo"],
        "builds": compiled_dependencies(root, logs),
    }
    return {
        **identity,
        "compiled_dependency_identity": dependency_identity,
        "compiled_dependency_fingerprint": hashlib.sha256(
            _encoded(dependency_identity).encode()
        ).hexdigest(),
    }
