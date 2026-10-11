from __future__ import annotations

import json
import tomllib
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

from scripts.benchmark_suite import issue406_harness as harness
from scripts.test_issue406_harness import commit, fixtures
from scripts.toolkit import command_output

DIAGNOSTIC_FEATURE = 'benchmark-checkpoint-diagnostic = ["benchmark-internals"]\n'
CARGO_PATH = "crates/calc-flow/Cargo.toml"


class FreshAcceptanceHarnessTests(unittest.TestCase):
    def test_fresh_overlay_uses_identical_helpers_and_opt_in_diagnostic(self):
        with TemporaryDirectory() as raw:
            root = Path(raw)
            baseline, candidate = fixtures(root)
            cargo = candidate / CARGO_PATH
            cargo.write_text(
                cargo.read_text() + harness.INSERTIONS[CARGO_PATH] + DIAGNOSTIC_FEATURE
            )
            commit(candidate)
            revision = command_output(["git", "rev-parse", "HEAD"], cwd=candidate)
            original = {
                source: harness._files(source) for source in (baseline, candidate)
            }
            with patch.object(harness, "ROOT", candidate):
                for source in (baseline, candidate):
                    output = root / f"{source.name}-output"
                    with harness.fresh_acceptance_harness(
                        source, output, "stream_join_perf"
                    ) as mirror:
                        features = tomllib.loads((mirror / CARGO_PATH).read_text())[
                            "features"
                        ]
                        self.assertEqual(
                            features,
                            {
                                "benchmark-internals": [],
                                "benchmark-checkpoint-off": ["benchmark-internals"],
                                "benchmark-checkpoint-diagnostic": [
                                    "benchmark-internals"
                                ],
                            },
                        )
                        for relative in harness.COPIED_FILES:
                            self.assertEqual(
                                (mirror / relative).read_bytes(),
                                (candidate / relative).read_bytes(),
                            )
                        self.assertEqual(
                            (mirror / "production.rs").read_bytes(),
                            (source / "production.rs").read_bytes(),
                        )
                    record = harness.validate_fresh_acceptance_harness(
                        output, "stream_join_perf", harness_git_sha=revision
                    )
                    self.assertEqual(
                        record["schema"], "calc-flow.issue406-native-harness.v2"
                    )
                    self.assertIsNone(record["compatibility"])
                    self.assertIsNone(record["helper_reconstruction"])
                    self.assertFalse(mirror.exists())
                    self.assertEqual(harness._files(source), original[source])

    def test_fresh_attestation_explicitly_excludes_historical_reuse(self):
        with TemporaryDirectory() as raw:
            root = Path(raw)
            baseline, candidate = fixtures(root)
            revision = command_output(["git", "rev-parse", "HEAD"], cwd=candidate)
            output = root / "output"
            with patch.object(harness, "ROOT", candidate):
                with harness.fresh_acceptance_harness(
                    baseline, output, "calc_flow_python"
                ):
                    pass
                evidence = output / "build-calc_flow_python.acceptance-harness.json"
                original = json.loads(evidence.read_text())
                for field in ("compatibility", "helper_reconstruction"):
                    for label, invalid in (
                        (
                            "missing",
                            {
                                key: value
                                for key, value in original.items()
                                if key != field
                            },
                        ),
                        ("empty", {**original, field: {}}),
                        ("historical", {**original, field: {"build": "historical"}}),
                    ):
                        evidence.write_text(json.dumps(invalid))
                        with (
                            self.subTest(field=field, mutation=label),
                            self.assertRaisesRegex(ValueError, "attestation"),
                        ):
                            harness.validate_fresh_acceptance_harness(
                                output, "calc_flow_python", harness_git_sha=revision
                            )

    def test_fresh_attestation_requires_the_exact_clean_harness_revision(self):
        with TemporaryDirectory() as raw:
            root = Path(raw)
            baseline, candidate = fixtures(root)
            revision = command_output(["git", "rev-parse", "HEAD"], cwd=candidate)
            output = root / "output"
            with (
                patch.object(harness, "ROOT", candidate),
                patch.object(
                    harness, "native_harness_reuse", side_effect=AssertionError
                ),
            ):
                with harness.fresh_acceptance_harness(
                    baseline, output, "stream_asof_e2e"
                ):
                    pass
                evidence = output / "build-stream_asof_e2e.acceptance-harness.json"
                original = json.loads(evidence.read_text())
                for mutation in (
                    {"schema": harness.SCHEMA},
                    {"harness_git_sha": "0" * 40},
                    {"effective_files": {}},
                    {"product_git_sha": "0" * 40},
                ):
                    evidence.write_text(json.dumps({**original, **mutation}))
                    with (
                        self.subTest(mutation=mutation),
                        self.assertRaisesRegex(ValueError, "attestation"),
                    ):
                        harness.validate_fresh_acceptance_harness(
                            output, "stream_asof_e2e", harness_git_sha=revision
                        )
                evidence.write_text(json.dumps(original))
                changed = candidate / "production.rs"
                changed.write_text(changed.read_text() + "// changed\n")
                with self.assertRaisesRegex(ValueError, "attestation"):
                    harness.validate_fresh_acceptance_harness(
                        output, "stream_asof_e2e", harness_git_sha=revision
                    )
                commit(candidate)
                current = command_output(["git", "rev-parse", "HEAD"], cwd=candidate)
                for supplied in (revision, current, "0" * 40):
                    with (
                        self.subTest(supplied=supplied),
                        self.assertRaisesRegex(ValueError, "attestation"),
                    ):
                        harness.validate_fresh_acceptance_harness(
                            output, "stream_asof_e2e", harness_git_sha=supplied
                        )

    def test_fresh_overlay_rejects_unreviewed_feature_declarations(self):
        with TemporaryDirectory() as raw:
            baseline, candidate = fixtures(Path(raw))
            cargo = baseline / CARGO_PATH
            original = cargo.read_text()
            legacy = harness.INSERTIONS[CARGO_PATH]
            with patch.object(harness, "ROOT", candidate):
                for declaration in (
                    legacy + DIAGNOSTIC_FEATURE + "unreviewed = []\n",
                    legacy + DIAGNOSTIC_FEATURE * 2,
                    legacy + "benchmark-checkpoint-diagnostic = []\n",
                    legacy + "# altered recipe\n" + DIAGNOSTIC_FEATURE,
                ):
                    cargo.write_text(original + declaration)
                    with (
                        self.subTest(declaration=declaration),
                        self.assertRaises(ValueError),
                    ):
                        harness._effective_fresh(baseline)

    def test_fresh_build_failure_and_mutation_leave_unverified_evidence(self):
        with TemporaryDirectory() as raw:
            root = Path(raw)
            baseline, candidate = fixtures(root)
            revision = command_output(["git", "rev-parse", "HEAD"], cwd=candidate)
            with patch.object(harness, "ROOT", candidate):
                for fail_build in (False, True):
                    output = root / str(fail_build)
                    with self.subTest(fail_build=fail_build):
                        with (
                            self.assertRaises((ValueError, RuntimeError)),
                            harness.fresh_acceptance_harness(
                                baseline, output, "stream_join_perf"
                            ) as mirror,
                        ):
                            if fail_build:
                                raise RuntimeError("build failed")
                            (mirror / "production.rs").write_text("changed algorithm")
                        self.assertFalse(mirror.exists())
                        with self.assertRaisesRegex(ValueError, "attestation"):
                            harness.validate_fresh_acceptance_harness(
                                output, "stream_join_perf", harness_git_sha=revision
                            )
                        evidence = (
                            output / "build-stream_join_perf.acceptance-harness.json"
                        )
                        original = evidence.read_bytes()
                        with (
                            self.assertRaisesRegex(ValueError, "fresh build evidence"),
                            harness.fresh_acceptance_harness(
                                baseline, output, "stream_join_perf"
                            ),
                        ):
                            pass
                        self.assertEqual(evidence.read_bytes(), original)

    def test_acceptance_profiles_exclude_checkpoint_diagnostic(self):
        for off in (False, True):
            features = ["benchmark-internals"] + (
                ["benchmark-checkpoint-off"] if off else []
            )
            row = {
                "package": {"workspace_package": "crates/calc-flow"},
                "target": {"kind": ["lib"], "name": "calc_flow"},
                "features": features,
                "profile": {"opt_level": "3", "test": False},
            }
            harness.validate_native_profile([row], checkpoint_off=off)
            with (
                self.subTest(off=off),
                self.assertRaisesRegex(ValueError, "release core"),
            ):
                harness.validate_native_profile(
                    [
                        {
                            **row,
                            "features": features + ["benchmark-checkpoint-diagnostic"],
                        }
                    ],
                    checkpoint_off=off,
                )
        record = {
            "actual_core_invocation": (
                "Running `rustc --crate-name calc_flow -C opt-level=3 "
                "--cfg 'feature=\"benchmark-internals\"' "
                "--cfg 'feature=\"benchmark-checkpoint-diagnostic\"'`"
            ),
            "actual_binding_invocation": (
                "Running `rustc --crate-name calc_flow_python -C opt-level=3 "
                "--cfg 'feature=\"benchmark-internals\"' "
                "--cfg 'feature=\"connector-file\"' --cfg 'feature=\"default\"'`"
            ),
        }
        with self.assertRaisesRegex(ValueError, "wheel"):
            harness.validate_traced_wheel_invocations(record)
