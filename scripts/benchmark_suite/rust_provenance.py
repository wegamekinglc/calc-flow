"""Identify dependencies actually compiled by the Rust benchmark commands."""

from __future__ import annotations

import json
import tomllib
from pathlib import Path

from scripts.toolkit import canonical_json, fingerprint_json


def _encoded(value: object) -> str:
    return canonical_json(value, ensure_ascii=True)


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


def _build_messages(name: str, log: Path) -> list[dict]:
    messages = [
        json.loads(line)
        for line in log.read_text(encoding="utf-8").splitlines()
        if line.startswith("{")
    ]
    finished = [row for row in messages if row.get("reason") == "build-finished"]
    if len(finished) != 1 or finished[0]["success"] is not True:
        raise ValueError(f"missing successful build completion: {name}")
    return messages


def _build_identity(root: Path, name: str, log: Path, locked: dict) -> list[dict]:
    artifacts = [
        _artifact_identity(root, row, locked)
        for row in _build_messages(name, log)
        if row.get("reason") == "compiler-artifact"
    ]
    _validate_artifacts(name, artifacts)
    return [json.loads(row) for row in sorted({_encoded(row) for row in artifacts})]


def _is_benchmark(name: str, row: dict) -> bool:
    return (
        row["target"]["kind"] == ["bench"]
        and row["target"]["name"] == name
        and row["package"].get("workspace_package") == "crates/calc-flow"
    )


def _validate_artifacts(name: str, artifacts: list[dict]) -> None:
    benchmarks = [row for row in artifacts if _is_benchmark(name, row)]
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
        "compiled_dependency_fingerprint": fingerprint_json(
            dependency_identity, ensure_ascii=True
        ),
    }


def target_dependency_fingerprint(identity: dict, target: str) -> str:
    """Scope comparison to one compiled target, retaining aggregate provenance."""
    compiled = identity["compiled_dependency_identity"]
    if compiled["schema"] != "calc-flow.compiled-benchmark-dependencies.v1":
        raise ValueError("invalid compiled dependency schema")
    if any(
        not isinstance(compiled[key], str) or not compiled[key]
        for key in ("rustc", "cargo")
    ):
        raise ValueError("missing compiled dependency compiler identity")
    if set(compiled["builds"]) != set(identity["scoped_workload_fingerprints"]):
        raise ValueError("incomplete compiled target inventory")
    artifacts = compiled["builds"][target]
    _validate_artifacts(target, artifacts)
    if not all(_valid_build_settings(row) for row in artifacts):
        raise ValueError("incomplete compiled artifact features or profile")
    return fingerprint_json(
        {**compiled, "builds": {target: artifacts}}, ensure_ascii=True
    )


def _valid_build_settings(row: dict) -> bool:
    return (
        isinstance(row.get("features"), list)
        and all(isinstance(feature, str) for feature in row["features"])
        and isinstance(row.get("profile"), dict)
        and bool(row["profile"])
    )
