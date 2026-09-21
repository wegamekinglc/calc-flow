"""Versioned npm dependency identity with unmodified lockfile provenance."""

from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path

from scripts.benchmark_suite.identity import validate_component
from scripts.toolkit import fingerprint_json, sha256_file

FINGERPRINT_PROTOCOL = "frontend-npm-lock-v1"


def dependency_metadata(raw: bytes) -> dict:
    lock = _parse_lock(raw)
    packages = _lock_packages(lock)
    project_version = _version(lock.get("version"), "version")
    root_version = _version(packages[""].get("version"), 'packages[""].version')
    normalized = {
        **{key: value for key, value in lock.items() if key != "version"},
        "packages": {
            **packages,
            "": {key: value for key, value in packages[""].items() if key != "version"},
        },
    }
    return {
        "dependency_identity": {"protocol": FINGERPRINT_PROTOCOL, "lock": normalized},
        "dependency_fingerprint": fingerprint_json(
            {"protocol": FINGERPRINT_PROTOCOL, "lock": normalized},
            ensure_ascii=True,
            allow_nan=False,
        ),
        "dependency_fingerprint_protocol": FINGERPRINT_PROTOCOL,
        "package_lock_sha256": hashlib.sha256(raw).hexdigest(),
        "project_version": project_version,
        "root_package_version": root_version,
    }


def _parse_lock(raw: bytes) -> dict:
    try:
        lock = json.loads(raw, object_pairs_hook=_unique_object)
    except ValueError as error:
        raise ValueError(f"invalid npm lockfile JSON: {error}") from error
    if not isinstance(lock, dict):
        raise ValueError("npm lockfile must be a JSON object")
    return lock


def _unique_object(pairs: list[tuple[str, object]]) -> dict:
    result = dict(pairs)
    if len(result) != len(pairs):
        raise ValueError("duplicate JSON object key")
    return result


def _lock_packages(lock: dict) -> dict:
    version = lock.get("lockfileVersion")
    if type(version) is not int or version != 3:
        raise ValueError("unsupported npm lockfileVersion; expected 3")
    packages = lock.get("packages")
    if not isinstance(packages, dict) or "" not in packages:
        raise ValueError("npm lockfile packages must contain the root package")
    if any(not isinstance(package, dict) for package in packages.values()):
        raise ValueError("npm lockfile package records must be JSON objects")
    return packages


def _version(value: object, field: str) -> str:
    if not isinstance(value, str) or not value:
        raise ValueError(f"npm lockfile {field} must be a nonempty string")
    return value


def metadata_problem(metadata: list[dict]) -> str | None:
    if any(
        row.get("dependency_fingerprint_protocol") != FINGERPRINT_PROTOCOL
        for row in metadata
    ):
        return (
            "benchmark frontend fingerprint protocol missing or unsupported; "
            "no timing classification"
        )
    if not all(_valid_provenance(row) for row in metadata):
        return (
            "benchmark frontend dependency provenance missing or invalid; "
            "no timing classification"
        )
    try:
        for row in metadata:
            for name in ("machine", "workload"):
                validate_component(row, name)
    except ValueError as error:
        return str(error)
    return None


def _valid_provenance(row: dict) -> bool:
    for key in ("dependency_fingerprint", "package_lock_sha256"):
        value = row.get(key)
        if not isinstance(value, str) or re.fullmatch(r"[0-9a-f]{64}", value) is None:
            return False
    identity = row.get("dependency_identity")
    if (
        not isinstance(identity, dict)
        or fingerprint_json(identity, ensure_ascii=True, allow_nan=False)
        != row["dependency_fingerprint"]
    ):
        return False
    return all(
        isinstance(row.get(key), str) and row[key]
        for key in ("project_version", "root_package_version")
    )


def workload_sources(frontend: Path, runner: Path) -> dict[str, str]:
    sources = sorted((frontend / "src").rglob("*.bench.*"))
    if not sources:
        raise ValueError("frontend benchmark sources are missing")
    return {
        **{
            path.relative_to(frontend).as_posix(): sha256_file(path) for path in sources
        },
        "vite.config.ts": sha256_file(frontend / "vite.config.ts"),
        "runner": sha256_file(runner),
        "identity_collector": sha256_file(runner.with_name("frontend_identity.mjs")),
    }


def case_identity(machine: dict, sources: dict, name: str, group: str) -> dict:
    _require_machine(machine)
    workload = {
        "case": name,
        "group": group,
        "scope": "vitest-native-boundary",
        "sources": sources,
    }
    return {
        "machine_identity": machine,
        "machine_fingerprint": fingerprint_json(machine),
        "workload_identity": workload,
        "workload_fingerprint": fingerprint_json(workload),
    }


def _require_machine(machine: dict) -> None:
    fields = ("platform", "architecture", "node_version", "v8_version")
    if not isinstance(machine, dict) or not all(
        isinstance(machine.get(key), str) and machine[key] for key in fields
    ):
        raise ValueError("frontend machine/runtime identity is missing or malformed")
    _require_cpus(machine)


def _require_cpus(machine: dict) -> None:
    if type(machine.get("logical_cpus")) is not int or machine["logical_cpus"] <= 0:
        raise ValueError("frontend machine CPU count is missing or malformed")
    models = machine.get("cpu_models")
    if not isinstance(models, list) or not models or not all(models):
        raise ValueError("frontend machine CPU models are missing or malformed")
