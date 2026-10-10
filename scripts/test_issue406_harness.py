from __future__ import annotations

import json
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

from scripts.benchmark_suite import issue406_harness as harness
from scripts.test_benchmark_rust_harness import checkout
from scripts.toolkit import command_output


def commit(root: Path) -> None:
    command_output(["git", "add", "."], cwd=root)
    command_output(
        [
            "git",
            "-c",
            "user.name=Test",
            "-c",
            "user.email=test@example.invalid",
            "commit",
            "-qm",
            "Fixture source",
        ],
        cwd=root,
    )


def fixtures(root: Path) -> tuple[Path, Path]:
    baseline, candidate = root / "baseline", root / "candidate"
    for source, label in ((baseline, "main"), (candidate, "candidate")):
        checkout(source, b"old bench", lock="version = 4\npackage = []\n")
        for relative in harness.COPIED_FILES:
            path = source / relative
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(f"// {label} {relative}\n")
        for relative in harness.INSERTIONS:
            path = source / relative
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(
                '[package]\nname = "calc-flow"\n'
                if relative.endswith(".toml")
                else f"// {label} production implementation\n"
            )
        for relative, patches in harness.LITERAL_PATCHES.items():
            path = source / relative
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text("".join(before for before, _after in patches))
        (source / "production.rs").write_text(f"// {label} must remain intact\n")
        commit(source)
    return baseline, candidate


class AcceptanceHarnessTests(unittest.TestCase):
    def test_python_correction_reuses_only_unchanged_historical_native_overlay(self):
        with TemporaryDirectory() as raw:
            root = Path(raw)
            baseline, candidate = fixtures(root)
            recipe = candidate / "scripts/benchmark_suite/issue406_harness.py"
            recipe.parent.mkdir(parents=True, exist_ok=True)
            recipe.write_text(Path(harness.__file__).read_text())
            commit(candidate)
            output = root / "output"
            with patch.object(harness, "ROOT", candidate):
                with harness.common_acceptance_harness(
                    baseline, output, "stream_asof_e2e"
                ):
                    pass
                original = harness.validate_acceptance_harness(
                    output, "stream_asof_e2e"
                )
                (candidate / "python_driver.py").write_text(
                    "fingerprint_before_run = True\n"
                )
                commit(candidate)
                reused = harness.validate_acceptance_harness(output, "stream_asof_e2e")
                self.assertEqual(reused, original)
                copied = candidate / harness.COPIED_FILES[0]
                copied.write_text(copied.read_text() + "// changed native work\n")
                commit(candidate)
                with self.assertRaisesRegex(ValueError, "attestation"):
                    harness.validate_acceptance_harness(output, "stream_asof_e2e")

    def test_historical_reuse_rejects_recipe_drift_and_missing_history(self):
        with TemporaryDirectory() as raw:
            root = Path(raw)
            baseline, candidate = fixtures(root)
            recipe = candidate / "scripts/benchmark_suite/issue406_harness.py"
            recipe.parent.mkdir(parents=True, exist_ok=True)
            recipe.write_text(Path(harness.__file__).read_text())
            commit(candidate)
            output = root / "output"
            with patch.object(harness, "ROOT", candidate):
                with harness.common_acceptance_harness(
                    baseline, output, "stream_asof_e2e"
                ):
                    pass
                record = harness.validate_acceptance_harness(output, "stream_asof_e2e")
                recipe.write_text(
                    recipe.read_text().replace(
                        "Closed, benchmark-only", "Changed, benchmark-only", 1
                    )
                )
                commit(candidate)
                with self.assertRaisesRegex(ValueError, "attestation"):
                    harness.validate_acceptance_harness(output, "stream_asof_e2e")
                evidence = output / "build-stream_asof_e2e.acceptance-harness.json"
                evidence.write_text(json.dumps({**record, "harness_git_sha": "0" * 40}))
                with self.assertRaisesRegex(ValueError, "attestation"):
                    harness.validate_acceptance_harness(output, "stream_asof_e2e")

    def test_traced_wheel_requires_both_actual_release_compiler_invocations(self):
        def invocation(crate, features):
            flags = " ".join(f"--cfg 'feature=\"{name}\"'" for name in features)
            return f"Running `rustc --crate-name {crate} -C opt-level=3 {flags}`"

        record = {
            "actual_core_invocation": invocation("calc_flow", ["benchmark-internals"]),
            "actual_binding_invocation": invocation(
                "calc_flow_python", ["benchmark-internals", "connector-file", "default"]
            ),
        }
        harness.validate_traced_wheel_invocations(record)
        for mutation in (
            {**record, "actual_core_invocation": invocation("calc_flow", [])},
            {
                **record,
                "actual_binding_invocation": record[
                    "actual_binding_invocation"
                ].replace("opt-level=3", "opt-level=0"),
            },
            {
                **record,
                "actual_core_invocation": record["actual_core_invocation"].replace(
                    " -C", " --cfg test -C"
                ),
            },
            {
                **record,
                "actual_core_invocation": invocation(
                    "calc_flow", ["benchmark-internals", "benchmark-checkpoint-off"]
                ),
            },
        ):
            with (
                self.subTest(mutation=mutation),
                self.assertRaisesRegex(ValueError, "wheel"),
            ):
                harness.validate_traced_wheel_invocations(mutation)

    def test_lock_identity_ignores_dependency_order_but_preserves_versions(self):
        with TemporaryDirectory() as raw:
            root = Path(raw)
            path = root / "Cargo.lock"
            prefix = 'version = 4\n[[package]]\nname = "calc-flow"\nversion = "1"\n'
            path.write_text(prefix + 'dependencies = ["fs2 1", "flatbuffers 2"]\n')
            original = harness._lock_identity(root)
            path.write_text(prefix + 'dependencies = ["flatbuffers 2", "fs2 1"]\n')
            self.assertEqual(harness._lock_identity(root), original)
            path.write_text(prefix + 'dependencies = ["flatbuffers 3", "fs2 1"]\n')
            self.assertNotEqual(harness._lock_identity(root), original)

    def test_only_closed_benchmark_overlay_changes_product_mirror(self):
        with TemporaryDirectory() as raw:
            root = Path(raw)
            baseline, candidate = fixtures(root)
            output = root / "output"
            with patch.object(harness, "ROOT", candidate):
                with harness.common_acceptance_harness(
                    baseline, output, "stream_asof_e2e"
                ) as mirror:
                    self.assertEqual(
                        (mirror / "production.rs").read_bytes(),
                        (baseline / "production.rs").read_bytes(),
                    )
                    for relative in (
                        harness.INSERTIONS.keys() | harness.LITERAL_PATCHES.keys()
                    ):
                        restored = (mirror / relative).read_text()
                        for before, after in reversed(
                            harness.LITERAL_PATCHES.get(relative, ())
                        ):
                            restored = restored.replace(after, before, 1)
                        insertion = harness.INSERTIONS.get(relative)
                        if insertion is not None:
                            restored = restored.replace(insertion, "", 1)
                        self.assertEqual(
                            restored,
                            (baseline / relative).read_text(),
                        )
                    for relative in harness.COPIED_FILES:
                        self.assertEqual(
                            (mirror / relative).read_bytes(),
                            (candidate / relative).read_bytes(),
                        )
                self.assertFalse(mirror.exists())
                record = harness.validate_acceptance_harness(output, "stream_asof_e2e")
                self.assertTrue(record["product_files_verified"])
                self.assertEqual(
                    record["product_git_sha"],
                    command_output(["git", "rev-parse", "HEAD"], cwd=baseline),
                )
                evidence = output / "build-stream_asof_e2e.acceptance-harness.json"
                invalid = {**record, "effective_files": {}}
                evidence.write_text(json.dumps(invalid))
                with self.assertRaisesRegex(ValueError, "attestation"):
                    harness.validate_acceptance_harness(output, "stream_asof_e2e")

    def test_unapproved_mutation_rejects_build_and_removes_mirror(self):
        with TemporaryDirectory() as raw:
            root = Path(raw)
            baseline, candidate = fixtures(root)
            with patch.object(harness, "ROOT", candidate):
                with (
                    self.assertRaisesRegex(ValueError, "product files"),
                    harness.common_acceptance_harness(
                        baseline, root / "output", "stream_join_perf"
                    ) as mirror,
                ):
                    (mirror / "production.rs").write_text("changed algorithm")
                self.assertFalse(mirror.exists())

    def test_failed_build_never_publishes_verified_attestation(self):
        with TemporaryDirectory() as raw:
            root = Path(raw)
            baseline, candidate = fixtures(root)
            output = root / "output"
            with patch.object(harness, "ROOT", candidate):
                with (
                    self.assertRaisesRegex(RuntimeError, "build failed"),
                    harness.common_acceptance_harness(
                        baseline, output, "stream_join_perf"
                    ) as mirror,
                ):
                    raise RuntimeError("build failed")
                self.assertFalse(mirror.exists())
                with self.assertRaisesRegex(ValueError, "attestation"):
                    harness.validate_acceptance_harness(output, "stream_join_perf")

    def test_compiled_core_must_use_release_without_test_configuration(self):
        base = {
            "package": {"workspace_package": "crates/calc-flow"},
            "target": {"kind": ["lib"], "name": "calc_flow"},
            "features": ["benchmark-internals"],
            "profile": {"opt_level": "3", "test": False},
        }
        harness.validate_native_profile([base], checkpoint_off=False)
        for invalid in (
            {**base, "profile": {"opt_level": "3", "test": True}},
            {**base, "features": []},
            {**base, "features": ["benchmark-internals", "benchmark-checkpoint-off"]},
        ):
            with (
                self.subTest(invalid=invalid),
                self.assertRaisesRegex(ValueError, "release core"),
            ):
                harness.validate_native_profile([invalid], checkpoint_off=False)
