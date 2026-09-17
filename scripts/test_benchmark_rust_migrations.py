"""Declared accept/re-baseline records for changed Rust bench sources."""

from __future__ import annotations

import json
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from scripts.benchmark_suite.migrations import (
    declared_migrations,
    load_migrations,
    match_migration,
)

SCHEMA = "calc-flow.rust-workload-migrations.v1"
REGISTRY = Path("benchmarks/rust-workload-migrations.json")
CORE = "crates/calc-flow/benches/core.rs"
JOIN = "crates/calc-flow/benches/join.rs"


def entry(**changes):
    declared = {
        "target": "core",
        "baseline_sha256": "a" * 64,
        "candidate_sha256": "b" * 64,
        "reason": "harness pipeline only: report output anchoring",
        "reference": "DAL-258",
    }
    return {**declared, **changes}


def write_registry(root: Path, migrations: list[dict]) -> None:
    path = root / REGISTRY
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps({"schema": SCHEMA, "migrations": migrations}, indent=2) + "\n",
        encoding="utf-8",
    )


def side(scoped: dict[str, str], workload_identity: dict[str, str]) -> dict:
    return {
        "scoped_workload_fingerprints": scoped,
        "workload_identity": workload_identity,
    }


def provenance() -> dict:
    return {
        "baseline": side(
            {"core": "core-baseline", "join": "join-shared"},
            {CORE: "a" * 64, JOIN: "c" * 64},
        ),
        "candidate": side(
            {"core": "core-candidate", "join": "join-shared"},
            {CORE: "b" * 64, JOIN: "c" * 64},
        ),
    }


class LoadMigrationTests(unittest.TestCase):
    def test_valid_registry_loads_its_declarations(self):
        with TemporaryDirectory() as raw:
            root = Path(raw)
            write_registry(root, [entry()])
            self.assertEqual(load_migrations(root), [entry()])

    def test_missing_registry_fails_closed(self):
        with (
            TemporaryDirectory() as raw,
            self.assertRaisesRegex(ValueError, "migration registry"),
        ):
            load_migrations(Path(raw))

    def test_registry_shape_and_schema_are_validated(self):
        with TemporaryDirectory() as raw:
            root = Path(raw)
            write_registry(root, [entry()])
            document = json.loads((root / REGISTRY).read_text(encoding="utf-8"))
            invalid = [
                {"migrations": []},
                {**document, "schema": "calc-flow.rust-workload-migrations.v2"},
                {**document, "extra": 1},
                {**document, "migrations": {}},
            ]
            for replaced in invalid:
                with self.subTest(replaced=replaced):
                    (root / REGISTRY).write_text(json.dumps(replaced), encoding="utf-8")
                    with self.assertRaises(ValueError):
                        load_migrations(root)

    def test_entry_shape_is_validated(self):
        with TemporaryDirectory() as raw:
            root = Path(raw)
            invalid = [
                {key: value for key, value in entry().items() if key != "reason"},
                {**entry(), "extra": True},
                {**entry(), "target": "  "},
                {**entry(), "reason": ""},
                {**entry(), "reference": 3},
                {**entry(), "baseline_sha256": "A" * 64},
                {**entry(), "candidate_sha256": "a" * 63},
                [entry(), entry()],
            ]
            for declarations in invalid:
                with self.subTest(declarations=declarations):
                    write_registry(root, declarations)
                    with self.assertRaises(ValueError):
                        load_migrations(root)


class MatchMigrationTests(unittest.TestCase):
    def test_match_requires_the_exact_declared_byte_pair(self):
        migrations = [entry()]
        self.assertIsNotNone(match_migration(migrations, "core", "a" * 64, "b" * 64))
        self.assertIsNone(match_migration(migrations, "join", "a" * 64, "b" * 64))
        self.assertIsNone(match_migration(migrations, "core", "a" * 64, "d" * 64))
        self.assertIsNone(match_migration(migrations, "core", "d" * 64, "b" * 64))


class DeclaredMigrationTests(unittest.TestCase):
    def test_only_declared_targets_are_applied(self):
        applied = declared_migrations(provenance(), [entry()])
        self.assertEqual(applied, {"core": entry()})

    def test_undeclared_byte_changes_are_not_applied(self):
        undeclared = [
            [],
            [entry(target="join")],
            [entry(candidate_sha256="d" * 64)],
        ]
        for declarations in undeclared:
            with self.subTest(declarations=declarations):
                self.assertEqual(declared_migrations(provenance(), declarations), {})

    def test_unchanged_targets_are_never_applied(self):
        applied = declared_migrations(
            provenance(), [entry(target="join", baseline_sha256="c" * 64)]
        )
        self.assertEqual(applied, {})


if __name__ == "__main__":
    unittest.main()
