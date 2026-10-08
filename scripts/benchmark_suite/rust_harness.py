"""Compile the reviewed Join lifecycle harness against unchanged product revisions."""

from __future__ import annotations

import json
import re
from contextlib import contextmanager
from pathlib import Path
from tempfile import TemporaryDirectory

from scripts.benchmark_suite.migrations import load_migrations, match_migration
from scripts.benchmark_suite.process import ROOT
from scripts.toolkit import command_output, fingerprint_json, sha256_file, write_json

TARGET = "stream_join_perf"
BENCH = Path(f"crates/calc-flow/benches/{TARGET}.rs")
SCHEMA = "calc-flow.common-rust-harness.v1"


def _clean_revision(source: Path) -> str:
    revision = command_output(["git", "rev-parse", "HEAD^{commit}"], cwd=source)
    status = command_output(
        ["git", "status", "--porcelain", "--untracked-files=no"], cwd=source
    )
    if not re.fullmatch(r"[0-9a-f]{40}", revision) or status:
        raise ValueError("common Rust harness requires a clean tracked worktree")
    return revision


def _declaration(original: str, measured: str) -> dict | None:
    if original == measured:
        return None
    matched = match_migration(load_migrations(ROOT), TARGET, original, measured)
    if matched is None:
        raise ValueError(
            "common Join harness requires an exact declared workload migration"
        )
    return matched


@contextmanager
def common_join_harness(source: Path, output: Path, target: str):
    """Overlay only the Join bench in an owned exact-revision build mirror.

    Both product checkouts remain read-only. The detached clone has the same
    tracked product files and lock; only its declared benchmark is replaced.
    """
    if target != TARGET:
        yield source
        return
    source = source.resolve()
    product_sha = _clean_revision(source)
    harness_sha = _clean_revision(ROOT)
    original = sha256_file(source / BENCH)
    measured = sha256_file(ROOT / BENCH)
    declaration = _declaration(original, measured)
    output.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(prefix=f"{TARGET}-", dir=output) as raw:
        mirror = Path(raw).resolve() / "source"
        command_output(
            [
                "git",
                "clone",
                "--shared",
                "--no-checkout",
                "--quiet",
                str(source),
                str(mirror),
            ]
        )
        command_output(
            [
                "git",
                "-c",
                "filter.lfs.process=",
                "-c",
                "filter.lfs.smudge=",
                "-c",
                "filter.lfs.required=false",
                "checkout",
                "--detach",
                "--quiet",
                product_sha,
            ],
            cwd=mirror,
        )
        if (
            _clean_revision(mirror) != product_sha
            or sha256_file(mirror / BENCH) != original
            or sha256_file(mirror / "Cargo.lock") != sha256_file(source / "Cargo.lock")
        ):
            raise ValueError(
                "owned Rust build mirror differs from the product revision"
            )
        (mirror / BENCH).write_bytes((ROOT / BENCH).read_bytes())
        write_json(
            output / f"build-{TARGET}.harness.json",
            {
                "schema": SCHEMA,
                "target": TARGET,
                "path": BENCH.as_posix(),
                "product_git_sha": product_sha,
                "harness_git_sha": harness_sha,
                "original_sha256": original,
                "measured_sha256": measured,
                "build_source": str(mirror),
                "build_output": str(output.resolve()),
                "migration_reference": None
                if declaration is None
                else declaration["reference"],
            },
        )
        yield mirror


def with_measured_harness(identity: dict, output: Path) -> dict:
    """Retain original product/source provenance and attest the effective bench."""
    if TARGET not in identity["scoped_workload_fingerprints"]:
        return identity
    try:
        record = json.loads((output / f"build-{TARGET}.harness.json").read_text())
    except (OSError, ValueError) as error:
        raise ValueError("missing common Rust harness attestation") from error
    return _with_validated_harness(identity, record, output)


def revalidate_measured_harness(identity: dict) -> dict:
    """Recheck persisted effective source evidence before paired acceptance."""
    if TARGET not in identity["scoped_workload_fingerprints"]:
        return identity
    record = identity.get("measured_harnesses", {}).get(TARGET)
    if not isinstance(record, dict) or not isinstance(record.get("build_output"), str):
        raise ValueError("invalid common Rust harness attestation")
    checked = _with_validated_harness(identity, record, Path(record["build_output"]))
    if checked["measured_workload_fingerprints"] != identity.get(
        "measured_workload_fingerprints"
    ):
        raise ValueError("invalid common Rust harness attestation")
    return checked


def _with_validated_harness(identity: dict, record: object, output: Path) -> dict:
    if not isinstance(record, dict):
        raise ValueError("invalid common Rust harness attestation")
    expected = {
        "schema": SCHEMA,
        "target": TARGET,
        "path": BENCH.as_posix(),
        "product_git_sha": identity["git_sha"],
        "original_sha256": identity["workload_identity"][BENCH.as_posix()],
        "build_output": str(output.resolve()),
    }
    if any(record.get(key) != value for key, value in expected.items()):
        raise ValueError("invalid common Rust harness attestation")
    if not _matches_current_harness(record, output):
        raise ValueError("invalid common Rust harness attestation")
    declaration = _declaration(record["original_sha256"], record["measured_sha256"])
    reference = None if declaration is None else declaration["reference"]
    if record.get("migration_reference") != reference:
        raise ValueError("invalid common Rust harness attestation")
    return {
        **identity,
        "measured_harnesses": {TARGET: record},
        "measured_workload_fingerprints": {
            TARGET: fingerprint_json({BENCH.as_posix(): record["measured_sha256"]})
        },
    }


def _matches_current_harness(record: dict, output: Path) -> bool:
    return (
        record.get("harness_git_sha") == _clean_revision(ROOT)
        and record.get("measured_sha256") == sha256_file(ROOT / BENCH)
        and _owned_build_path(record.get("build_source"), output)
    )


def _owned_build_path(raw: object, output: Path) -> bool:
    if not isinstance(raw, str):
        return False
    path = Path(raw).resolve()
    return (
        path.name == "source"
        and path.parent.name.startswith(f"{TARGET}-")
        and path.parent.parent == output.resolve()
    )


def with_harness_migrations(
    applied: dict, provenance: dict, migrations: list[dict]
) -> dict:
    """Record declared original-to-measured overlays even for equal originals."""
    combined = dict(applied)
    for identity in provenance.values():
        for target, record in identity.get("measured_harnesses", {}).items():
            original, measured = record["original_sha256"], record["measured_sha256"]
            if original == measured:
                continue
            matched = match_migration(migrations, target, original, measured)
            if matched is None or matched["reference"] != record["migration_reference"]:
                raise ValueError(
                    "measured Rust harness has no exact declared migration"
                )
            if target in combined and combined[target] != matched:
                raise ValueError(f"conflicting Rust workload migrations: {target}")
            combined[target] = matched
    return combined
