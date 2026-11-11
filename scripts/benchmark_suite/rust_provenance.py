"""Identify dependencies actually compiled by the Rust benchmark commands."""

from __future__ import annotations

import json
import tomllib
from pathlib import Path

from scripts.toolkit import canonical_json, fingerprint_json, git_output, sha256_file


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


def _vendored_package(
    root: Path, relative: Path, name: str, locked: list[dict]
) -> dict:
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


def _clean_vendor_tree(root: Path, relative: Path) -> str:
    if git_output(
        root,
        "status",
        "--porcelain",
        "--untracked-files=all",
        "--ignored=matching",
        "--",
        relative.as_posix(),
    ):
        raise ValueError("canonical vendored dependency has dirty or untracked sources")
    tracked = set(
        git_output(root, "ls-files", "-z", "--", relative.as_posix()).split("\0")
    )
    tracked.discard("")
    actual = {
        path.relative_to(root).as_posix()
        for path in (root / relative).rglob("*")
        if path.is_file()
    }
    if not actual or tracked != actual:
        raise ValueError("canonical vendored dependency file inventory is not tracked")
    return git_output(root, "rev-parse", f"HEAD:{relative.as_posix()}")


def _canonical_vendor_package(root: Path, artifact: dict, locked: dict) -> tuple:
    common = Path(
        git_output(root, "rev-parse", "--path-format=absolute", "--git-common-dir")
    ).resolve()
    canonical = common.parent
    if (
        common.name != ".git"
        or Path(git_output(canonical, "rev-parse", "--show-toplevel")).resolve()
        != canonical
    ):
        raise ValueError("canonical vendored dependency needs the main worktree")
    if (
        Path(
            git_output(
                canonical, "rev-parse", "--path-format=absolute", "--git-common-dir"
            )
        ).resolve()
        != common
    ):
        raise ValueError("canonical vendored dependency belongs to another repository")
    manifest = Path(artifact["manifest_path"])
    try:
        relative = manifest.parent.relative_to(canonical)
    except ValueError as error:
        raise ValueError(
            "canonical vendored dependency is outside the main worktree"
        ) from error
    source = f"path+{relative.as_posix()}"
    package = locked.get(source)
    if (
        package is None
        or not relative.is_relative_to("vendor")
        or manifest.name != "Cargo.toml"
        or manifest.resolve() != manifest
        or artifact["package_id"]
        not in (
            f"path+{manifest.parent.as_uri()}#{package['version']}",
            f"path+{manifest.parent.as_uri()}#{package['name']}@{package['version']}",
        )
    ):
        raise ValueError("invalid canonical vendored dependency path or package ID")
    measured_tree = _clean_vendor_tree(root, relative)
    canonical_tree = _clean_vendor_tree(canonical, relative)
    if (
        canonical_tree != measured_tree
        or _vendored_source_fingerprint(manifest.parent) != package["source_sha256"]
    ):
        raise ValueError(
            "canonical vendored dependency content differs from the worktree"
        )
    return package, {
        "package_id": artifact["package_id"],
        "manifest_path": artifact["manifest_path"],
        "logical_source": source,
        "canonical_root": str(canonical),
        "measured_root": str(root),
        "git_common_dir": str(common),
        "vendor_tree": canonical_tree,
        "source_sha256": package["source_sha256"],
    }


def _package_identity(root: Path, artifact: dict, locked: dict) -> tuple:
    manifest = Path(artifact["manifest_path"])
    if manifest == root / "crates/calc-flow/Cargo.toml":
        return {"workspace_package": "crates/calc-flow"}, None
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
        if manifest.name != "Cargo.toml" or manifest.is_relative_to(root):
            raise ValueError(
                f"unrecognized compiled dependency: {artifact['package_id']}"
            )
        return _canonical_vendor_package(root, artifact, locked)
    return package, None


def _artifact_identity(root: Path, artifact: dict, locked: dict) -> tuple:
    package, mapping = _package_identity(root, artifact, locked)
    return {
        "package": package,
        "target": {
            key: artifact["target"][key]
            for key in ("name", "kind", "crate_types", "edition")
        },
        "features": sorted(artifact["features"]),
        "profile": artifact["profile"],
    }, mapping


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


def _build_identity(root: Path, name: str, log: Path, locked: dict) -> tuple:
    records = [
        _artifact_identity(root, row, locked)
        for row in _build_messages(name, log)
        if row.get("reason") == "compiler-artifact"
    ]
    artifacts = [record[0] for record in records]
    mappings = [record[1] for record in records if record[1] is not None]
    _validate_artifacts(name, artifacts)
    return (
        [json.loads(row) for row in sorted({_encoded(row) for row in artifacts})],
        [json.loads(row) for row in sorted({_encoded(row) for row in mappings})],
    )


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
    return _compiled_dependency_records(root, logs, build_roots=build_roots)[0]


def _compiled_dependency_records(
    root: Path, logs: dict[str, Path], *, build_roots: dict[str, Path] | None = None
) -> tuple:
    if not logs:
        raise ValueError("missing Rust benchmark build logs")
    locked = _locked_packages(root)
    records = {
        name: _build_identity((build_roots or {}).get(name, root), name, log, locked)
        for name, log in sorted(logs.items())
    }
    return (
        {name: record[0] for name, record in records.items()},
        {name: record[1] for name, record in records.items() if record[1]},
    )


def with_compiled_dependencies(
    identity: dict,
    root: Path,
    logs: dict,
    *,
    build_roots: dict[str, Path] | None = None,
) -> dict:
    """Preserve full-lock provenance and add the compiled comparison identity."""
    builds, mappings = _compiled_dependency_records(root, logs, build_roots=build_roots)
    dependency_identity = {
        "schema": "calc-flow.compiled-benchmark-dependencies.v1",
        "rustc": identity["dependency_identity"]["rustc"],
        "cargo": identity["dependency_identity"]["cargo"],
        "builds": builds,
    }
    return {
        **identity,
        "canonical_vendor_mappings": mappings,
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
