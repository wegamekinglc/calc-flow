from __future__ import annotations

import asyncio
import hashlib
import json
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

from scripts.benchmark_suite import rust_harness
from scripts.benchmark_suite.migrations import declared_migrations
from scripts.benchmark_suite.rust import _stamp_fingerprints, build_binaries
from scripts.benchmark_suite.rust_harness import (
    common_join_harness,
    with_measured_harness,
)
from scripts.benchmark_suite.rust_provenance import compiled_dependencies
from scripts.test_benchmark_rust_provenance import write_inputs
from scripts.toolkit import command_output, fingerprint_json

TARGET = "stream_join_perf"
BENCH = Path(f"crates/calc-flow/benches/{TARGET}.rs")


def digest(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def checkout(root: Path, contents: bytes, lock: str = "product lock") -> str:
    path = root / BENCH
    path.parent.mkdir(parents=True)
    path.write_bytes(contents)
    (root / "Cargo.lock").write_text(lock, encoding="utf-8")
    for argv in (
        ["git", "init", "-q", str(root)],
        ["git", "add", "."],
        [
            "git",
            "-c",
            "user.name=Test",
            "-c",
            "user.email=test@example.invalid",
            "commit",
            "-qm",
            "Initial product",
        ],
    ):
        command_output(argv, cwd=root)
    return command_output(["git", "rev-parse", "HEAD"], cwd=root)


def registry(root: Path, baseline: bytes, candidate: bytes) -> None:
    path = root / "benchmarks/rust-workload-migrations.json"
    path.parent.mkdir(parents=True)
    path.write_text(
        json.dumps(
            {
                "schema": "calc-flow.rust-workload-migrations.v1",
                "migrations": [
                    {
                        "target": TARGET,
                        "baseline_sha256": digest(baseline),
                        "candidate_sha256": digest(candidate),
                        "reason": "Prepare checkpoint before handler on both products",
                        "reference": "benchmark followup",
                    }
                ],
            }
        ),
        encoding="utf-8",
    )


class CommonJoinHarnessTests(unittest.TestCase):
    def test_same_old_harness_versions_keep_overlay_migration_markers(self):
        declaration = {
            "target": TARGET,
            "baseline_sha256": digest(b"old"),
            "candidate_sha256": digest(b"current"),
            "reason": "common checkpoint preparation lifecycle",
            "reference": "benchmark followup",
        }
        provenance = {
            side: {
                "machine_fingerprint": "machine",
                "workload_identity": {BENCH.as_posix(): digest(b"old")},
                "scoped_workload_fingerprints": {TARGET: "old"},
                "measured_workload_fingerprints": {TARGET: "common"},
                "measured_harnesses": {
                    TARGET: {
                        "original_sha256": digest(b"old"),
                        "measured_sha256": digest(b"current"),
                        "harness_git_sha": "b" * 40,
                        "migration_reference": "benchmark followup",
                    }
                },
            }
            for side in ("baseline", "candidate")
        }
        original = declared_migrations(provenance, [declaration])
        self.assertEqual(original, {})
        applied = rust_harness.with_harness_migrations(
            original, provenance, [declaration]
        )
        self.assertEqual(original, {})
        self.assertEqual(applied, {TARGET: declaration})
        with patch(
            "scripts.benchmark_suite.rust.target_dependency_fingerprint",
            return_value="compiled",
        ):
            stamps = _stamp_fingerprints(provenance, applied)
        for side in ("baseline", "candidate"):
            self.assertEqual(
                stamps[side][TARGET]["workload_migration"], "benchmark followup"
            )

    def test_overlay_migrations_keep_existing_declarations_and_fail_on_conflicts(self):
        declaration = {
            "target": TARGET,
            "baseline_sha256": digest(b"old"),
            "candidate_sha256": digest(b"current"),
            "reason": "common checkpoint preparation lifecycle",
            "reference": "benchmark followup",
        }
        provenance = {
            "baseline": {
                "measured_harnesses": {
                    TARGET: {
                        "original_sha256": digest(b"old"),
                        "measured_sha256": digest(b"current"),
                        "migration_reference": "benchmark followup",
                    }
                }
            }
        }
        existing = {"core": {"reference": "existing declaration"}}
        applied = rust_harness.with_harness_migrations(
            existing, provenance, [declaration]
        )
        self.assertEqual(applied, {**existing, TARGET: declaration})
        self.assertEqual(existing, {"core": {"reference": "existing declaration"}})
        with self.assertRaisesRegex(ValueError, "migration"):
            rust_harness.with_harness_migrations(existing, provenance, [])
        with self.assertRaisesRegex(ValueError, "conflicting"):
            rust_harness.with_harness_migrations(
                {TARGET: {"reference": "other"}}, provenance, [declaration]
            )

    def test_both_products_compile_common_bytes_without_mutating_sources(self):
        with TemporaryDirectory() as raw:
            root = Path(raw)
            baseline, candidate = root / "baseline", root / "candidate"
            old, current = b"old handler harness", b"prepared checkpoint harness"
            revisions = {
                "baseline": checkout(baseline, old),
                "candidate": checkout(candidate, current),
            }
            registry(candidate, old, current)
            for side, source in (("baseline", baseline), ("candidate", candidate)):
                original = (source / BENCH).read_bytes()
                output = root / "output" / side
                with patch("scripts.benchmark_suite.rust_harness.ROOT", candidate):
                    with common_join_harness(source, output, TARGET) as build_source:
                        self.assertNotEqual(build_source, source)
                        self.assertEqual((build_source / BENCH).read_bytes(), current)
                        self.assertEqual((source / BENCH).read_bytes(), original)
                        self.assertEqual(
                            (build_source / "Cargo.lock").read_text(), "product lock"
                        )
                    self.assertFalse(build_source.exists())
                self.assertEqual((source / BENCH).read_bytes(), original)
                identity = {
                    "git_sha": revisions[side],
                    "workload_identity": {BENCH.as_posix(): digest(original)},
                    "scoped_workload_fingerprints": {TARGET: "original"},
                }
                with patch("scripts.benchmark_suite.rust_harness.ROOT", candidate):
                    measured = with_measured_harness(identity, output)
                self.assertEqual(measured["git_sha"], revisions[side])
                self.assertEqual(
                    measured["workload_identity"], identity["workload_identity"]
                )
                evidence = measured["measured_harnesses"][TARGET]
                self.assertEqual(evidence["product_git_sha"], revisions[side])
                self.assertEqual(evidence["harness_git_sha"], revisions["candidate"])
                self.assertEqual(evidence["measured_sha256"], digest(current))
                self.assertEqual(
                    measured["measured_workload_fingerprints"][TARGET],
                    fingerprint_json({BENCH.as_posix(): digest(current)}),
                )

    def test_failed_or_cancelled_build_removes_owned_mirror(self):
        with TemporaryDirectory() as raw:
            root = Path(raw)
            baseline, candidate = root / "baseline", root / "candidate"
            checkout(baseline, b"old")
            checkout(candidate, b"current")
            registry(candidate, b"old", b"current")
            for exception in (RuntimeError("build failed"), asyncio.CancelledError()):
                with self.subTest(exception=type(exception).__name__):
                    with (
                        patch("scripts.benchmark_suite.rust_harness.ROOT", candidate),
                        self.assertRaises(type(exception)),
                        common_join_harness(
                            baseline, root / "output", TARGET
                        ) as build_source,
                    ):
                        self.assertEqual(
                            (build_source / BENCH).read_bytes(), b"current"
                        )
                        self.assertEqual((baseline / BENCH).read_bytes(), b"old")
                        raise exception
                    self.assertFalse(build_source.exists())
                    self.assertEqual((baseline / BENCH).read_bytes(), b"old")

    def test_unknown_baseline_bytes_fail_before_overlay(self):
        with TemporaryDirectory() as raw:
            root = Path(raw)
            baseline, candidate = root / "baseline", root / "candidate"
            checkout(baseline, b"unknown")
            checkout(candidate, b"current")
            registry(candidate, b"declared old", b"current")
            with (
                patch("scripts.benchmark_suite.rust_harness.ROOT", candidate),
                self.assertRaisesRegex(ValueError, "declared.*migration"),
                common_join_harness(baseline, root / "output", TARGET),
            ):
                self.fail("unknown workload entered a build")
            self.assertEqual((baseline / BENCH).read_bytes(), b"unknown")

    def test_dirty_product_checkout_is_rejected_without_modification(self):
        with TemporaryDirectory() as raw:
            root = Path(raw)
            baseline, candidate = root / "baseline", root / "candidate"
            checkout(baseline, b"old")
            checkout(candidate, b"current")
            registry(candidate, b"old", b"current")
            (baseline / "Cargo.lock").write_text("edited lock", encoding="utf-8")
            with (
                patch("scripts.benchmark_suite.rust_harness.ROOT", candidate),
                self.assertRaisesRegex(ValueError, "clean tracked worktree"),
                common_join_harness(baseline, root / "output", TARGET),
            ):
                self.fail("dirty product entered a build")
            self.assertEqual((baseline / BENCH).read_bytes(), b"old")

    def test_missing_or_mismatched_harness_attestation_fails_closed(self):
        with TemporaryDirectory() as raw:
            root = Path(raw)
            identity = {
                "git_sha": "a" * 40,
                "workload_identity": {BENCH.as_posix(): digest(b"old")},
                "scoped_workload_fingerprints": {TARGET: "original"},
            }
            with self.assertRaisesRegex(ValueError, "harness.*attestation"):
                with_measured_harness(identity, root)

    def test_attestation_revalidates_harness_revision_bytes_migration_and_owned_path(
        self,
    ):
        with TemporaryDirectory() as raw:
            root = Path(raw)
            baseline, candidate = root / "baseline", root / "candidate"
            product_sha = checkout(baseline, b"old")
            checkout(candidate, b"current")
            registry(candidate, b"old", b"current")
            output = root / "output"
            with patch("scripts.benchmark_suite.rust_harness.ROOT", candidate):
                with common_join_harness(baseline, output, TARGET):
                    pass
                path = output / f"build-{TARGET}.harness.json"
                original = json.loads(path.read_text())
                identity = {
                    "git_sha": product_sha,
                    "workload_identity": {BENCH.as_posix(): digest(b"old")},
                    "scoped_workload_fingerprints": {TARGET: "original"},
                }
                invalid = [
                    {**original, "measured_sha256": "a" * 64},
                    {**original, "harness_git_sha": "b" * 40},
                    {**original, "migration_reference": "undeclared acceptance"},
                    {**original, "build_source": str(output / ".." / "foreign-source")},
                    [],
                ]
                for record in invalid:
                    with self.subTest(record=record):
                        path.write_text(json.dumps(record))
                        with self.assertRaises(ValueError):
                            with_measured_harness(identity, output)
                unknown = {**original, "original_sha256": "c" * 64}
                path.write_text(json.dumps(unknown))
                with self.assertRaisesRegex(ValueError, "declared.*migration"):
                    with_measured_harness(
                        {
                            **identity,
                            "workload_identity": {BENCH.as_posix(): "c" * 64},
                        },
                        output,
                    )

            document = {
                "schema": "calc-flow.common-rust-harness.v1",
                "target": TARGET,
                "path": BENCH.as_posix(),
                "product_git_sha": "b" * 40,
                "harness_git_sha": "c" * 40,
                "original_sha256": digest(b"old"),
                "measured_sha256": digest(b"current"),
                "migration_reference": "benchmark followup",
            }
            (root / f"build-{TARGET}.harness.json").write_text(json.dumps(document))
            with self.assertRaisesRegex(ValueError, "harness.*attestation"):
                with_measured_harness(identity, root)

    def test_owned_build_artifacts_preserve_original_locked_dependencies(self):
        with TemporaryDirectory() as raw:
            root = Path(raw)
            log, messages = write_inputs(root)
            mirror = root / "owned-source"
            messages[1]["manifest_path"] = str(mirror / "crates/calc-flow/Cargo.toml")
            log.write_text("\n".join(json.dumps(row) for row in messages))
            result = compiled_dependencies(
                root, {"core": log}, build_roots={"core": mirror}
            )
            self.assertEqual(result["core"][0]["package"].get("checksum"), "a" * 64)
            self.assertTrue(
                any(
                    row["package"] == {"workspace_package": "crates/calc-flow"}
                    for row in result["core"]
                )
            )

    def test_pairing_uses_common_measured_harness_and_rejects_differing_bytes(self):
        provenance = {
            side: {
                "machine_fingerprint": "machine",
                "scoped_workload_fingerprints": {TARGET: side},
                "measured_workload_fingerprints": {TARGET: "common"},
                "measured_harnesses": {
                    TARGET: {"measured_sha256": "a" * 64, "harness_git_sha": "b" * 40}
                },
            }
            for side in ("baseline", "candidate")
        }
        with patch(
            "scripts.benchmark_suite.rust.target_dependency_fingerprint",
            return_value="compiled",
        ):
            stamps = _stamp_fingerprints(provenance, {})
            self.assertEqual(
                stamps["baseline"][TARGET]["workload_fingerprint"], "common"
            )
            self.assertEqual(stamps["baseline"][TARGET], stamps["candidate"][TARGET])
            provenance["baseline"]["measured_harnesses"][TARGET]["measured_sha256"] = (
                "c" * 64
            )
            with self.assertRaisesRegex(ValueError, "common.*harness"):
                _stamp_fingerprints(provenance, {})

    def test_candidate_only_join_is_new_coverage_but_shared_join_requires_attestation(
        self,
    ):
        provenance = {
            "baseline": {
                "machine_fingerprint": "machine",
                "scoped_workload_fingerprints": {"core": "same"},
            },
            "candidate": {
                "machine_fingerprint": "machine",
                "scoped_workload_fingerprints": {"core": "same", TARGET: "join"},
                "measured_workload_fingerprints": {TARGET: "common"},
                "measured_harnesses": {
                    TARGET: {
                        "measured_sha256": "a" * 64,
                        "harness_git_sha": "b" * 40,
                    }
                },
            },
        }
        with patch(
            "scripts.benchmark_suite.rust.target_dependency_fingerprint",
            return_value="compiled",
        ):
            stamps = _stamp_fingerprints(provenance, {})
            self.assertNotIn(TARGET, stamps["baseline"])
            self.assertEqual(
                stamps["candidate"][TARGET]["workload_fingerprint"], "common"
            )
            provenance["baseline"]["scoped_workload_fingerprints"][TARGET] = "join"
            with self.assertRaisesRegex(ValueError, "common.*harness"):
                _stamp_fingerprints(provenance, {})

    def test_shared_join_missing_both_attestations_fails_closed(self):
        provenance = {
            side: {
                "machine_fingerprint": "machine",
                "scoped_workload_fingerprints": {TARGET: "shared"},
            }
            for side in ("baseline", "candidate")
        }
        with (
            patch(
                "scripts.benchmark_suite.rust.target_dependency_fingerprint",
                return_value="compiled",
            ),
            self.assertRaisesRegex(ValueError, "common.*harness"),
        ):
            _stamp_fingerprints(provenance, {})


class CommonJoinBuildTests(unittest.IsolatedAsyncioTestCase):
    async def test_build_binaries_compiles_the_owned_common_harness(self):
        with TemporaryDirectory() as raw:
            root = Path(raw)
            baseline, candidate = root / "baseline", root / "candidate"
            checkout(baseline, b"old")
            checkout(candidate, b"current")
            registry(candidate, b"old", b"current")
            output = root / "output"
            output.mkdir()

            async def command(_argv, **kwargs):
                source = kwargs["cwd"]
                self.assertNotEqual(source, baseline)
                self.assertEqual((baseline / BENCH).read_bytes(), b"old")
                self.assertEqual((source / BENCH).read_bytes(), b"current")
                executable = output / "compiled"
                executable.write_bytes(b"compiled original product with common harness")
                kwargs["log"].write_text(
                    json.dumps(
                        {
                            "reason": "compiler-artifact",
                            "target": {"name": TARGET},
                            "executable": str(executable),
                        }
                    )
                )

            with (
                patch("scripts.benchmark_suite.rust_harness.ROOT", candidate),
                patch("scripts.benchmark_suite.rust.command", side_effect=command),
            ):
                binaries = await build_binaries(
                    baseline, output, root / "cache", targets=(TARGET,)
                )
            self.assertEqual(
                binaries[TARGET].read_bytes(),
                b"compiled original product with common harness",
            )
            self.assertEqual((baseline / BENCH).read_bytes(), b"old")


if __name__ == "__main__":
    unittest.main()
