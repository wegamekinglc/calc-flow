"""Exact, directional attestations keep Rust dependency upgrades fail closed."""

from __future__ import annotations

import copy
import json
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

from scripts.benchmark_suite.dependency_migrations import (
    REGISTRY,
    SCHEMA,
    applied_dependency_migrations,
    load_dependency_migrations,
    stamp_dependency_migrations,
    validate_dependency_migrations,
)
from scripts.benchmark_suite.legacy import combine_blocks
from scripts.benchmark_suite.report import render_report
from scripts.benchmark_suite.rust import _stamp_fingerprints
from scripts.benchmark_suite.rust_provenance import target_dependency_fingerprint
from scripts.toolkit import fingerprint_json, sha256_file

LOADER = "scripts.benchmark_suite.dependency_migrations.load_dependency_migrations"
SHARD = {"id": "rust", "family": "rust"}


def refresh(identity: dict) -> dict:
    result = copy.deepcopy(identity)
    for name in ("machine", "dependency", "workload"):
        result[f"{name}_fingerprint"] = fingerprint_json(result[f"{name}_identity"])
    result["compiled_dependency_fingerprint"] = fingerprint_json(
        result["compiled_dependency_identity"], ensure_ascii=True
    )
    result["scoped_workload_fingerprints"] = {
        Path(path).stem: fingerprint_json({path: digest})
        for path, digest in result["workload_identity"].items()
    }
    return result


def provenance() -> dict:
    sides = {}
    for side, version, lock in (
        ("baseline", "58.3.0", "a"),
        ("candidate", "59.2.0", "b"),
    ):
        artifact = {
            "package": {
                "name": "arrow",
                "version": version,
                "source": "registry+crates.io",
                "checksum": lock * 64,
            },
            "target": {
                "name": "arrow",
                "kind": ["lib"],
                "crate_types": ["lib"],
                "edition": "2024",
            },
            "features": ["default"],
            "profile": {"opt_level": "3", "test": False},
        }
        sides[side] = refresh(
            {
                "git_sha": lock * 40,
                "tracked_worktree_clean": True,
                "machine_identity": {"cpu": "same-runner"},
                "dependency_identity": {
                    "cargo_lock_sha256": lock * 64,
                    "rustc": "rustc 1.94",
                    "cargo": "cargo 1.94",
                },
                "workload_identity": {
                    f"crates/calc-flow/benches/{target}.rs": "c" * 64
                    for target in ("core", "other")
                },
                "compiled_dependency_identity": {
                    "schema": "calc-flow.compiled-benchmark-dependencies.v1",
                    "rustc": "rustc 1.94",
                    "cargo": "cargo 1.94",
                    "builds": {
                        target: [
                            artifact,
                            {
                                **artifact,
                                "package": {"workspace_package": "crates/calc-flow"},
                                "target": {"name": target, "kind": ["bench"]},
                            },
                        ]
                        for target in ("core", "other")
                    },
                },
            }
        )
    return sides


def declaration(identities: dict) -> dict:
    return {
        "target": "core",
        **{
            side: {
                "dependency_fingerprint": target_dependency_fingerprint(
                    identity, "core"
                ),
                "cargo_lock_sha256": identity["dependency_identity"][
                    "cargo_lock_sha256"
                ],
                "workload_fingerprint": identity["scoped_workload_fingerprints"][
                    "core"
                ],
            }
            for side, identity in identities.items()
        },
        "measured_workload_fingerprint": identities["candidate"][
            "scoped_workload_fingerprints"
        ]["core"],
        "reason": "Exact release upgrade; identical fixture and timing source bytes",
        "reference": "DAL-384 regression fixture",
    }


def blocks(stamps: dict) -> dict:
    return {
        side: [
            {
                f"{target}/case": {
                    "metadata": stamp,
                    "rows": 1,
                    "scope": "criterion-native-boundary",
                    "samples": [1.0 if side == "baseline" else 1.2] * 10,
                }
                for target, stamp in targets.items()
            }
            for _ in range(2)
        ]
        for side, targets in stamps.items()
    }


def without_attestation_field(report: dict, field: str) -> dict:
    result = copy.deepcopy(report)
    if field == "declarations":
        del result["dependency_migrations"]
    elif field == "marker":
        del result["cases"][0]["metadata"]["dependency_migration"]
    else:
        result["cases"][0]["metadata"]["dependency_fingerprint"] = "f" * 64
    return result


class DependencyMigrationTests(unittest.TestCase):
    def setUp(self):
        self.provenance = provenance()
        self.entry = declaration(self.provenance)
        self.stamps = _stamp_fingerprints(self.provenance, {})

    def accepted(self):
        return stamp_dependency_migrations(self.provenance, self.stamps, [self.entry])

    def cases(self, measured: dict, *, family: str = "rust") -> list[dict]:
        with patch(LOADER, return_value=[self.entry]):
            return combine_blocks({"id": family, "family": family}, measured)

    def test_exact_declaration_retains_observed_fingerprints_and_inputs(self):
        original = copy.deepcopy((self.provenance, self.stamps, self.entry))
        accepted = self.accepted()
        self.assertEqual((self.provenance, self.stamps, self.entry), original)
        for side in accepted:
            self.assertEqual(
                accepted[side]["core"],
                {**self.stamps[side]["core"], "dependency_migration": self.entry},
            )
            self.assertEqual(accepted[side]["other"], self.stamps[side]["other"])
        cases = self.cases(blocks(accepted))
        migrated = next(case for case in cases if case["scenario"] == "core/case")
        self.assertEqual(migrated["result"]["verdict"], "informational-slowdown")
        self.assertEqual(migrated["baseline"], [[1.0] * 10] * 2)
        self.assertEqual(migrated["candidate"], [[1.2] * 10] * 2)
        self.assertIn("cross-dependency release", render_report(cases, []))
        self.assertEqual(
            next(case for case in cases if case["scenario"] == "other/case")["status"],
            "error",
        )

    def test_unknown_direction_target_workload_and_lock_cannot_migrate(self):
        invalid = [
            [],
            [{**self.entry, "target": "other"}],
            [
                {
                    **self.entry,
                    "baseline": self.entry["candidate"],
                    "candidate": self.entry["baseline"],
                }
            ],
            [{**self.entry, "measured_workload_fingerprint": "d" * 64}],
            [
                {
                    **self.entry,
                    "candidate": {
                        **self.entry["candidate"],
                        "cargo_lock_sha256": "e" * 64,
                    },
                }
            ],
        ]
        for entries in invalid:
            with self.subTest(entries=entries):
                self.assertEqual(
                    stamp_dependency_migrations(self.provenance, self.stamps, entries),
                    self.stamps,
                )
                self.assertTrue(
                    all(
                        case["status"] == "error"
                        for case in self.cases(blocks(self.stamps))
                    )
                )

    def test_unknown_compiled_package_checksum_features_profile_and_compiler_reject(
        self,
    ):
        for field, value in (
            ("version", "99.0.0"),
            ("checksum", "e" * 64),
            ("features", ["simd"]),
            ("profile", {"opt_level": "0"}),
            ("rustc", "rustc 1.95"),
            ("cargo", "cargo 1.95"),
        ):
            changed = copy.deepcopy(self.provenance)
            identity = changed["candidate"]
            compiled = identity["compiled_dependency_identity"]
            artifact = compiled["builds"]["core"][0]
            if field in ("version", "checksum"):
                artifact["package"][field] = value
            elif field in ("rustc", "cargo"):
                compiled[field] = value
                identity["dependency_identity"][field] = value
            else:
                artifact[field] = value
            with self.subTest(field=field):
                with self.assertRaisesRegex(ValueError, "identity"):
                    stamp_dependency_migrations(changed, self.stamps, [self.entry])
                changed["candidate"] = refresh(identity)
                stamps = _stamp_fingerprints(changed, {})
                result = stamp_dependency_migrations(changed, stamps, [self.entry])
                self.assertNotIn("dependency_migration", result["candidate"]["core"])
                self.assertTrue(
                    all(
                        case["status"] == "error" for case in self.cases(blocks(result))
                    )
                )

    def test_corrupt_lock_workload_and_stamp_cannot_reuse_a_declaration(self):
        for field in (
            "dependency",
            "workload",
            "scoped_workload",
            "compiled_dependency",
        ):
            changed = copy.deepcopy(self.provenance)
            changed["candidate"][
                f"{field}_fingerprint"
                if field != "scoped_workload"
                else "scoped_workload_fingerprints"
            ] = (
                "e" * 64
                if field != "scoped_workload"
                else {"core": "e" * 64, "other": "e" * 64}
            )
            with self.subTest(field=field), self.assertRaises(ValueError):
                stamp_dependency_migrations(changed, self.stamps, [self.entry])
        changed = copy.deepcopy(self.stamps)
        changed["candidate"]["core"]["dependency_fingerprint"] = "e" * 64
        with self.assertRaisesRegex(ValueError, "stamp"):
            stamp_dependency_migrations(self.provenance, changed, [self.entry])

    def test_forged_unknown_partial_or_reversed_attestations_are_not_trusted(self):
        accepted = blocks(self.accepted())
        invalid = []
        for field, value in (
            ("reference", "unknown"),
            ("target", "other"),
            ("measured_workload_fingerprint", "f" * 64),
        ):
            changed = copy.deepcopy(accepted)
            for values in changed.values():
                for block in values:
                    block["core/case"]["metadata"]["dependency_migration"][field] = (
                        value
                    )
            invalid.append(changed)
        changed = copy.deepcopy(accepted)
        del changed["baseline"][0]["core/case"]["metadata"]["dependency_migration"]
        invalid.append(changed)
        invalid.append(
            {"baseline": accepted["candidate"], "candidate": accepted["baseline"]}
        )
        changed = copy.deepcopy(accepted)
        for values in changed.values():
            for block in values:
                block["other/case"] = block.pop("core/case")
        invalid.append(changed)
        for changed in invalid:
            with self.subTest(changed=changed):
                self.assertEqual(self.cases(changed)[0]["status"], "error")

    def test_measured_harness_support_and_product_identity_are_bound(self):
        identities = copy.deepcopy(self.provenance)
        path = "crates/calc-flow/benches/core.rs"
        for identity in identities.values():
            identity["measured_workload_fingerprints"] = {
                "core": identity["scoped_workload_fingerprints"]["core"]
            }
            identity["measured_harnesses"] = {
                "core": {
                    "schema": "calc-flow.common-rust-harness.v1",
                    "target": "core",
                    "path": path,
                    "product_git_sha": identity["git_sha"],
                    "harness_git_sha": "f" * 40,
                    "original_sha256": "c" * 64,
                    "measured_sha256": "c" * 64,
                    "support_files": {},
                }
            }
        stamps = _stamp_fingerprints(identities, {})
        self.assertIn(
            "dependency_migration",
            stamp_dependency_migrations(identities, stamps, [self.entry])["candidate"][
                "core"
            ],
        )
        for field, value in (
            ("product_git_sha", "f" * 40),
            ("target", "other"),
            ("path", "other.rs"),
            ("support_files", {"helper.rs": "f" * 64}),
            ("support_files", {path: "c" * 64}),
        ):
            changed = copy.deepcopy(identities)
            changed["candidate"]["measured_harnesses"]["core"][field] = value
            with self.subTest(field=field, value=value), self.assertRaises(ValueError):
                stamp_dependency_migrations(changed, stamps, [self.entry])

    def test_attestation_does_not_bypass_other_identity_sample_or_family_checks(self):
        for field in ("machine_fingerprint", "workload_fingerprint"):
            changed = blocks(self.accepted())
            changed["baseline"][0] = copy.deepcopy(changed["baseline"][0])
            changed["baseline"][0]["core/case"]["metadata"][field] = "f" * 64
            with self.subTest(field=field):
                self.assertEqual(self.cases(changed)[0]["status"], "error")
        changed = blocks(self.accepted())
        changed["baseline"].pop()
        self.assertTrue(all(case["status"] == "error" for case in self.cases(changed)))
        self.assertTrue(
            all(
                case["status"] == "error"
                for case in self.cases(blocks(self.accepted()), family="python")
            )
        )
        changed = blocks(self.accepted())
        changed["candidate"][0]["core/case"]["samples"] = [0.0]
        with self.assertRaisesRegex(ValueError, "positive"):
            self.cases(changed)

    def test_fragment_attestation_is_recomputed_from_original_provenance(self):
        accepted = self.accepted()
        report = {
            "provenance": self.provenance,
            "releases": {
                side: {
                    "git_sha": identity["git_sha"],
                    "cargo_lock_sha256": identity["dependency_identity"][
                        "cargo_lock_sha256"
                    ],
                }
                for side, identity in self.provenance.items()
            },
            "dependency_migrations": applied_dependency_migrations(accepted),
            "cases": self.cases(blocks(accepted)),
        }
        with patch(LOADER, return_value=[self.entry]):
            validate_dependency_migrations(report)
            invalid = copy.deepcopy(report)
            invalid["provenance"]["candidate"]["compiled_dependency_identity"][
                "builds"
            ]["core"][0]["features"] = ["forged"]
            with self.assertRaises(ValueError):
                validate_dependency_migrations(invalid)
            invalid["provenance"]["candidate"] = refresh(
                invalid["provenance"]["candidate"]
            )
            with self.assertRaises(ValueError):
                validate_dependency_migrations(invalid)
            invalid["dependency_migrations"] = []
            unknown_stamps = _stamp_fingerprints(invalid["provenance"], {})
            for case in invalid["cases"]:
                target = case["scenario"].split("/", 1)[0]
                case["metadata"] = unknown_stamps["candidate"][target]
            with self.assertRaisesRegex(ValueError, "uncompared"):
                validate_dependency_migrations(invalid)
            for field in ("git_sha", "cargo_lock_sha256"):
                invalid = copy.deepcopy(report)
                invalid["releases"]["candidate"][field] = "f" * (
                    40 if field == "git_sha" else 64
                )
                with self.subTest(field=field), self.assertRaises(ValueError):
                    validate_dependency_migrations(invalid)
            for remove in ("declarations", "marker", "fingerprint"):
                invalid = without_attestation_field(report, remove)
                with self.subTest(remove=remove), self.assertRaises(ValueError):
                    validate_dependency_migrations(invalid)
            invalid = copy.deepcopy(report)
            del invalid["dependency_migrations"]
            for case in invalid["cases"]:
                case["metadata"].pop("dependency_migration", None)
            with self.assertRaisesRegex(ValueError, "missing Rust dependency"):
                validate_dependency_migrations(invalid)


class DependencyRegistryTests(unittest.TestCase):
    def test_repository_declarations_pin_existing_targets_and_workloads(self):
        root = Path(__file__).resolve().parents[1]
        entries = load_dependency_migrations(root)
        self.assertEqual(len(entries), 12)
        for entry in entries:
            path = f"crates/calc-flow/benches/{entry['target']}.rs"
            self.assertEqual(
                fingerprint_json({path: sha256_file(root / path)}),
                entry["candidate"]["workload_fingerprint"],
            )

    def test_registry_rejects_unknown_schema_fields_endpoints_and_duplicates(self):
        entry = declaration(provenance())
        valid = {"schema": SCHEMA, "migrations": [entry]}
        invalid = [
            [],
            {**valid, "extra": True},
            {**valid, "schema": "unknown"},
            {**valid, "migrations": {}},
            {**valid, "migrations": [entry, entry]},
        ]
        for change in (
            {"target": "../core"},
            {"extra": True},
            {"reason": " "},
            {"reference": None},
            {"measured_workload_fingerprint": "A" * 64},
            {"baseline": {**entry["baseline"], "extra": True}},
            {"candidate": {**entry["candidate"], "dependency_fingerprint": "*"}},
            {"baseline": entry["candidate"]},
        ):
            invalid.append({**valid, "migrations": [{**entry, **change}]})
        with TemporaryDirectory() as raw:
            root = Path(raw)
            (root / REGISTRY).parent.mkdir(parents=True)
            with self.assertRaises(OSError):
                load_dependency_migrations(root)
            (root / REGISTRY).write_text(json.dumps(valid))
            self.assertEqual(load_dependency_migrations(root), [entry])
            for document in invalid:
                (root / REGISTRY).write_text(json.dumps(document))
                with self.subTest(document=document), self.assertRaises(ValueError):
                    load_dependency_migrations(root)


if __name__ == "__main__":
    unittest.main()
