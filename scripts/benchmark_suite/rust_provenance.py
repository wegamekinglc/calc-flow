"""Identify dependencies actually compiled by the Rust benchmark commands."""

from __future__ import annotations

import json
import tomllib
from pathlib import Path

from scripts.toolkit import canonical_json, fingerprint_json, sha256_file


def _encoded(value: object) -> str:
    return canonical_json(value, ensure_ascii=True)


def _locked_packages(root: Path) -> dict:
    lock = tomllib.loads((root / "Cargo.lock").read_text(encoding="utf-8"))
    return {
        **{
            f"{item['source']}#{item['name']}@{item['version']}": {
                key: item[key] for key in ("name", "version", "source", "checksum")
            }
            for item in lock["package"]
            if item.get("source", "").startswith("registry+")
        },
        **_vendored_packages(root, lock["package"]),
    }


def _vendored_packages(root: Path, locked: list[dict]) -> dict:
    manifest = root / "Cargo.toml"
    if not manifest.is_file():
        return {}
    workspace = tomllib.loads(manifest.read_text(encoding="utf-8"))
    patches = workspace.get("patch", {}).get("crates-io", {})
    packages = {}
    for name, patch in patches.items():
        if not isinstance(patch, dict) or "path" not in patch:
            continue
        relative = Path(patch["path"])
        if not relative.is_relative_to("vendor") or ".." in relative.parts:
            continue
        source = f"path+{relative.as_posix()}"
        packages[source] = _vendored_package(
            root, relative, patch.get("package", name), locked
        )
    return packages


def _vendored_package(root: Path, relative: Path, name: str, locked: list[dict]) -> dict:
    directory = root / relative
    if directory.is_symlink() or not directory.resolve().is_relative_to(
        root.resolve() / "vendor"
    ):
        raise ValueError(f"vendored dependency escapes source tree: {relative}")
    package = tomllib.loads((directory / "Cargo.toml").read_text(encoding="utf-8"))[
        "package"
    ]
    identity = {key: package[key] for key in ("name", "version")}
    _validate_vendored_identity(identity, name, locked, relative)
    return {
        **identity,
        "source": f"path+{relative.as_posix()}",
        "source_sha256": _vendored_source_fingerprint(directory),
    }


def _validate_vendored_identity(
    identity: dict, name: str, locked: list[dict], relative: Path
) -> None:
    if identity["name"] != name or not any(
        all(item[key] == value for key, value in identity.items())
        and "source" not in item
        for item in locked
    ):
        raise ValueError(f"vendored dependency differs from Cargo.lock: {relative}")


def _vendored_source_fingerprint(directory: Path) -> str:
    paths = sorted(directory.rglob("*"))
    if any(path.is_symlink() for path in paths):
        raise ValueError(f"vendored dependency contains a symlink: {directory}")
    return fingerprint_json(
        {
            path.relative_to(directory).as_posix(): sha256_file(path)
            for path in paths
            if path.is_file()
        }
    )


def _package_identity(root: Path, artifact: dict, locked: dict) -> dict:
    manifest = Path(artifact["manifest_path"])
    if manifest == root / "crates/calc-flow/Cargo.toml":
        return {"workspace_package": "crates/calc-flow"}
    package_id = artifact["package_id"]
    package = locked.get(package_id) if package_id.startswith("registry+") else None
    if (
        package is None
        and manifest.name == "Cargo.toml"
        and manifest.is_relative_to(root)
    ):
        source = f"path+{manifest.parent.relative_to(root).as_posix()}"
        package = locked.get(source)
        if package is not None and package_id not in (
            f"path+{manifest.parent.as_uri()}#{package['version']}",
            f"path+{manifest.parent.as_uri()}#{package['name']}@{package['version']}",
        ):
            package = None
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


def compiled_dependencies(
    root: Path, logs: dict[str, Path], *, build_roots: dict[str, Path] | None = None
) -> dict:
    """Retain package checksums, enabled features and profiles per bench build.

    Cargo's compiler-artifact messages cover fresh and cached builds alike.
    Registry checksums and declared vendor patch contents identify dependencies
    independently of checkout locations. Other dependency sources fail closed.
    """
    if not logs:
        raise ValueError("missing Rust benchmark build logs")
    locked = _locked_packages(root)
    return {
        name: _build_identity((build_roots or {}).get(name, root), name, log, locked)
        for name, log in sorted(logs.items())
    }


def with_compiled_dependencies(
    identity: dict,
    root: Path,
    logs: dict,
    *,
    build_roots: dict[str, Path] | None = None,
) -> dict:
    """Preserve full-lock provenance and add the compiled comparison identity."""
    dependency_identity = {
        "schema": "calc-flow.compiled-benchmark-dependencies.v1",
        "rustc": identity["dependency_identity"]["rustc"],
        "cargo": identity["dependency_identity"]["cargo"],
        "builds": compiled_dependencies(root, logs, build_roots=build_roots),
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
