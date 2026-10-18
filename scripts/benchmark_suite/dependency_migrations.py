"""Explicit Rust release comparisons across pinned, genuinely different dependencies."""

from __future__ import annotations

import json
import re
from pathlib import Path

from scripts.benchmark_suite.identity import validate_component
from scripts.benchmark_suite.migrations import BENCH_SOURCE, SHA256
from scripts.benchmark_suite.process import ROOT
from scripts.benchmark_suite.rust_provenance import target_dependency_fingerprint
from scripts.toolkit import fingerprint_json

SCHEMA = "calc-flow.rust-dependency-migrations.v1"
REGISTRY = Path("benchmarks/rust-dependency-migrations.json")
ENDPOINT_KEYS = {"dependency_fingerprint", "cargo_lock_sha256", "workload_fingerprint"}
ENTRY_KEYS = {
    "target",
    "baseline",
    "candidate",
    "measured_workload_fingerprint",
    "reason",
    "reference",
}
CHANGED = "benchmark dependency_fingerprint changed; no timing classification"
INVALID = "invalid Rust dependency migration; no timing classification"


def load_dependency_migrations(root: Path = ROOT) -> list[dict]:
    """Load closed declarations; neither wildcard identities nor directions exist."""
    document = json.loads((root / REGISTRY).read_text(encoding="utf-8"))
    if (
        not isinstance(document, dict)
        or set(document) != {"schema", "migrations"}
        or document["schema"] != SCHEMA
        or not isinstance(document["migrations"], list)
    ):
        raise ValueError("invalid Rust dependency migration registry")
    entries = document["migrations"]
    for entry in entries:
        _validate_entry(entry)
    identities = [
        fingerprint_json({k: e[k] for k in ENTRY_KEYS - {"reason", "reference"}})
        for e in entries
    ]
    if len(set(identities)) != len(identities):
        raise ValueError("duplicate Rust dependency migrations")
    return entries


def _validate_entry(entry: object) -> None:
    if not isinstance(entry, dict) or set(entry) != ENTRY_KEYS:
        raise ValueError("invalid Rust dependency migration entry")
    for key in ("target", "reason", "reference"):
        if not isinstance(entry[key], str) or not entry[key].strip():
            raise ValueError(f"invalid dependency migration {key}")
    if re.fullmatch(r"[a-zA-Z0-9_]+", entry["target"]) is None:
        raise ValueError("invalid dependency migration target")
    for side in ("baseline", "candidate"):
        endpoint = entry[side]
        if not isinstance(endpoint, dict) or set(endpoint) != ENDPOINT_KEYS:
            raise ValueError("invalid dependency migration endpoint")
        for value in endpoint.values():
            _validate_digest(value)
    _validate_digest(entry["measured_workload_fingerprint"])
    if (
        entry["baseline"]["dependency_fingerprint"]
        == entry["candidate"]["dependency_fingerprint"]
    ):
        raise ValueError("dependency migration requires distinct compiled identities")


def _validate_digest(value: object) -> None:
    if not isinstance(value, str) or SHA256.fullmatch(value) is None:
        raise ValueError("dependency migration requires lowercase SHA-256 identities")


def _endpoint(identity: dict, target: str) -> dict:
    """Recompute observed identities before accepting any declaration."""
    for name in ("dependency", "workload"):
        validate_component(identity, name)
    compiled = identity["compiled_dependency_identity"]
    dependency = identity["dependency_identity"]
    if fingerprint_json(compiled, ensure_ascii=True) != identity[
        "compiled_dependency_fingerprint"
    ] or any(compiled[key] != dependency[key] for key in ("rustc", "cargo")):
        raise ValueError("corrupt compiled Rust dependency identity")
    path = BENCH_SOURCE.format(target=target)
    workload = fingerprint_json({path: identity["workload_identity"][path]})
    if workload != identity["scoped_workload_fingerprints"][target]:
        raise ValueError("corrupt scoped Rust workload identity")
    if target in identity.get("measured_harnesses", {}):
        _validate_measured_workload(identity, target, path)
    return {
        "dependency_fingerprint": target_dependency_fingerprint(identity, target),
        "cargo_lock_sha256": dependency["cargo_lock_sha256"],
        "workload_fingerprint": workload,
    }


def _validate_measured_workload(identity: dict, target: str, path: str) -> None:
    record = identity["measured_harnesses"][target]
    expected = {
        "schema": "calc-flow.common-rust-harness.v1",
        "target": target,
        "path": path,
        "product_git_sha": identity["git_sha"],
        "original_sha256": identity["workload_identity"][path],
    }
    if any(record.get(key) != value for key, value in expected.items()):
        raise ValueError("corrupt measured Rust workload identity")
    support = record.get("support_files", {})
    if not isinstance(support, dict) or path in support:
        raise ValueError("corrupt measured Rust workload support")
    measured = {path: record["measured_sha256"], **support}
    for digest in measured.values():
        _validate_digest(digest)
    if fingerprint_json(measured) != identity["measured_workload_fingerprints"][target]:
        raise ValueError("corrupt measured Rust workload fingerprint")


def stamp_dependency_migrations(
    provenance: dict, stamps: dict, declarations: list[dict]
) -> dict:
    """Add an attestation without replacing either side's original fingerprint."""
    applied = {}
    for target in stamps["baseline"].keys() & stamps["candidate"].keys():
        pair = {side: stamps[side][target] for side in ("baseline", "candidate")}
        if (
            pair["baseline"]["dependency_fingerprint"]
            == pair["candidate"]["dependency_fingerprint"]
        ):
            continue
        endpoints = {side: _endpoint(provenance[side], target) for side in pair}
        for side in pair:
            if (
                pair[side]["dependency_fingerprint"]
                != endpoints[side]["dependency_fingerprint"]
            ):
                raise ValueError("Rust dependency stamp differs from compiled identity")
        matched = next(
            (
                entry
                for entry in declarations
                if entry["target"] == target
                and all(entry[side] == endpoints[side] for side in pair)
                and all(
                    entry["measured_workload_fingerprint"]
                    == pair[side]["workload_fingerprint"]
                    for side in pair
                )
            ),
            None,
        )
        if matched is not None:
            applied[target] = matched
    return {
        side: {
            target: {
                **stamp,
                **(
                    {"dependency_migration": applied[target]}
                    if target in applied
                    else {}
                ),
            }
            for target, stamp in targets.items()
        }
        for side, targets in stamps.items()
    }


def dependency_migration_problem(name: str, blocks: dict) -> str | None:
    """Validate the exact declaration and B/B/C/C direction on all four blocks."""
    rows = {
        side: [block[name]["metadata"] for block in blocks[side]]
        for side in ("baseline", "candidate")
    }
    metadata = [row for values in rows.values() for row in values]
    markers = [row.get("dependency_migration") for row in metadata]
    if not any(marker is not None for marker in markers):
        return (
            CHANGED
            if any(
                row["dependency_fingerprint"] != metadata[0]["dependency_fingerprint"]
                for row in metadata
            )
            else None
        )
    declared = markers[0]
    if (
        not isinstance(declared, dict)
        or any(marker != declared for marker in markers)
        or declared not in load_dependency_migrations()
        or declared["target"] != name.split("/", 1)[0]
        or any(
            row["dependency_fingerprint"] != declared[side]["dependency_fingerprint"]
            or row["workload_fingerprint"] != declared["measured_workload_fingerprint"]
            for side, values in rows.items()
            for row in values
        )
    ):
        return INVALID
    return None


def applied_dependency_migrations(stamps: dict) -> list[dict]:
    return [
        stamp["dependency_migration"]
        for _, stamp in sorted(stamps["candidate"].items())
        if "dependency_migration" in stamp
    ]


def validate_dependency_migrations(report: dict) -> None:
    """Recheck persisted attestations against real provenance before aggregation."""
    if "dependency_migrations" not in report and not any(
        "dependency_migration" in case.get("metadata", {}) for case in report["cases"]
    ):
        if any(
            case["status"] == "ok" and case.get("kind") != "metric"
            for case in report["cases"]
        ):
            raise ValueError("missing Rust dependency comparison attestations")
        return
    from scripts.benchmark_suite.migrations import declared_migrations, load_migrations
    from scripts.benchmark_suite.rust import _stamp_fingerprints
    from scripts.benchmark_suite.rust_harness import with_harness_migrations

    provenance = report["provenance"]
    for side, identity in provenance.items():
        if (
            identity["git_sha"] != report["releases"][side]["git_sha"]
            or identity["tracked_worktree_clean"] is not True
            or identity["dependency_identity"]["cargo_lock_sha256"]
            != report["releases"][side]["cargo_lock_sha256"]
        ):
            raise ValueError("Rust dependency provenance differs from sealed release")
    migrations = load_migrations(ROOT)
    applied = with_harness_migrations(
        declared_migrations(provenance, migrations), provenance, migrations
    )
    stamps = stamp_dependency_migrations(
        provenance,
        _stamp_fingerprints(provenance, applied),
        load_dependency_migrations(),
    )
    if report.get("dependency_migrations") != applied_dependency_migrations(stamps):
        raise ValueError(
            "Rust dependency migration attestations differ from provenance"
        )
    for case in report["cases"]:
        if case.get("kind") == "metric":
            continue
        target = case["scenario"].split("/", 1)[0]
        stamp = stamps["candidate"][target]
        if case["status"] == "ok" and case.get("comparison") != "new":
            baseline = stamps["baseline"].get(target)
            if baseline is None or (
                baseline["dependency_fingerprint"] != stamp["dependency_fingerprint"]
                and "dependency_migration" not in stamp
            ):
                raise ValueError(
                    "uncompared Rust dependencies cannot have an ok result"
                )
        if any(
            case["metadata"].get(key) != stamp.get(key)
            for key in (*stamp, "dependency_migration")
        ):
            raise ValueError("Rust case dependency attestation differs from provenance")
