"""Declared accept or re-baseline records for Rust benchmark source changes."""

from __future__ import annotations

import json
import re
from pathlib import Path

SCHEMA = "calc-flow.rust-workload-migrations.v1"
REGISTRY = Path("benchmarks/rust-workload-migrations.json")
ENTRY_KEYS = frozenset(
    ("target", "baseline_sha256", "candidate_sha256", "reason", "reference")
)
SHA256 = re.compile(r"[0-9a-f]{64}")
BENCH_SOURCE = "crates/calc-flow/benches/{target}.rs"


def load_migrations(root: Path) -> list[dict]:
    """Load and validate the declared workload migration registry."""
    path = root / REGISTRY
    if not path.is_file():
        raise ValueError(f"missing workload migration registry: {REGISTRY}")
    migrations = [_validated(entry) for entry in _registry_entries(path)]
    identities = [
        (item["target"], item["baseline_sha256"], item["candidate_sha256"])
        for item in migrations
    ]
    if len(identities) != len(set(identities)):
        raise ValueError("duplicate workload migration declarations")
    return migrations


def _registry_entries(path: Path) -> list[object]:
    document = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(document, dict) or set(document) != {"schema", "migrations"}:
        raise ValueError(
            "workload migration registry must contain only schema and migrations"
        )
    if document["schema"] != SCHEMA:
        raise ValueError(
            f"unsupported workload migration schema: {document['schema']!r}"
        )
    if not isinstance(document["migrations"], list):
        raise ValueError("workload migration entries must be a list")
    return document["migrations"]


def _validated(declared: object) -> dict:
    if not isinstance(declared, dict) or set(declared) != ENTRY_KEYS:
        raise ValueError(
            f"workload migration entries must declare exactly {sorted(ENTRY_KEYS)}"
        )
    _validated_text_fields(declared)
    _validated_digest_fields(declared)
    return dict(declared)


def _validated_text_fields(declared: dict) -> None:
    for key in ("target", "reason", "reference"):
        if not isinstance(declared[key], str) or not declared[key].strip():
            raise ValueError(f"workload migration {key} must be a non-empty string")


def _validated_digest_fields(declared: dict) -> None:
    for key in ("baseline_sha256", "candidate_sha256"):
        if (
            not isinstance(declared[key], str)
            or SHA256.fullmatch(declared[key]) is None
        ):
            raise ValueError(
                f"workload migration {key} must be a lowercase SHA-256 digest"
            )


def match_migration(
    migrations: list[dict], target: str, baseline_sha256: str, candidate_sha256: str
) -> dict | None:
    """Return the declaration pinning exactly these observed source bytes."""
    return next(
        (
            item
            for item in migrations
            if (item["target"], item["baseline_sha256"], item["candidate_sha256"])
            == (target, baseline_sha256, candidate_sha256)
        ),
        None,
    )


def declared_migrations(provenance: dict, migrations: list[dict]) -> dict:
    """Apply declarations only where both sides' benchmark bytes are pinned.

    Undeclared source changes stay unapplied, so only their own benchmark
    target fails the workload fingerprint comparison closed.
    """
    applied = {}
    shared = (
        provenance["baseline"]["scoped_workload_fingerprints"].keys()
        & provenance["candidate"]["scoped_workload_fingerprints"].keys()
    )
    for target in shared:
        path = BENCH_SOURCE.format(target=target)
        baseline_sha = provenance["baseline"]["workload_identity"].get(path)
        candidate_sha = provenance["candidate"]["workload_identity"].get(path)
        if (
            baseline_sha is None
            or candidate_sha is None
            or baseline_sha == candidate_sha
        ):
            continue
        matched = match_migration(migrations, target, baseline_sha, candidate_sha)
        if matched is not None:
            applied[target] = matched
    return applied
