"""Versioned npm dependency identity with unmodified lockfile provenance."""

from __future__ import annotations

import hashlib
import json
import re

FINGERPRINT_PROTOCOL = "frontend-npm-lock-v1"


def dependency_metadata(raw: bytes) -> dict[str, str]:
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
    identity = json.dumps(
        {"protocol": FINGERPRINT_PROTOCOL, "lock": normalized},
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode()
    return {
        "dependency_fingerprint": hashlib.sha256(identity).hexdigest(),
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
    return None


def _valid_provenance(row: dict) -> bool:
    for key in ("dependency_fingerprint", "package_lock_sha256"):
        value = row.get(key)
        if not isinstance(value, str) or re.fullmatch(r"[0-9a-f]{64}", value) is None:
            return False
    return all(
        isinstance(row.get(key), str) and row[key]
        for key in ("project_version", "root_package_version")
    )
