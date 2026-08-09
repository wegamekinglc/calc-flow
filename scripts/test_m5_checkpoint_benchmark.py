from __future__ import annotations

import math
import shutil
import tempfile
import unittest
from pathlib import Path

from scripts import m5_checkpoint_benchmark as benchmark


def _run(
    label: str,
    median: float,
    lower: float,
    upper: float,
) -> dict[str, object]:
    return {
        "label": label,
        "case": benchmark.COMMON_CASE,
        "median_ns": median,
        "median_confidence_interval_ns": [lower, upper],
        "confidence_level": 0.95,
        "sample_count": benchmark.COMMON_SAMPLE_COUNT,
    }


class CommonDecisionTests(unittest.TestCase):
    def test_common_case_pass_requires_both_pairing_intervals_below_five_percent(
        self,
    ) -> None:
        decision = benchmark.evaluate_common_case(
            [
                _run("B1", 100.0, 99.9, 100.1),
                _run("C1", 103.0, 102.9, 103.1),
                _run("B2", 100.2, 100.1, 100.3),
                _run("C2", 103.1, 103.0, 103.2),
            ]
        )

        self.assertEqual(decision["decision"], "pass")
        self.assertLessEqual(decision["candidate_min_regression_percent"], 5.0)

    def test_common_case_regression_must_be_sustained_and_exceed_noise(self) -> None:
        decision = benchmark.evaluate_common_case(
            [
                _run("B1", 100.0, 99.9, 100.1),
                _run("C1", 107.0, 106.9, 107.1),
                _run("B2", 100.2, 100.1, 100.3),
                _run("C2", 107.2, 107.1, 107.3),
            ]
        )

        self.assertEqual(decision["decision"], "regression")
        self.assertGreater(decision["candidate_min_regression_percent"], 5.0)
        self.assertTrue(decision["sustained_in_both_pairings"])
        self.assertTrue(decision["exceeds_twice_baseline_spread"])

    def test_common_case_ci_crossing_or_unsustained_change_is_inconclusive(
        self,
    ) -> None:
        crossing = benchmark.evaluate_common_case(
            [
                _run("B1", 100.0, 99.0, 101.0),
                _run("C1", 106.0, 104.0, 108.0),
                _run("B2", 100.0, 99.0, 101.0),
                _run("C2", 106.0, 104.0, 108.0),
            ]
        )
        unsustained = benchmark.evaluate_common_case(
            [
                _run("B1", 100.0, 99.9, 100.1),
                _run("C1", 107.0, 106.9, 107.1),
                _run("B2", 100.0, 99.9, 100.1),
                _run("C2", 103.0, 102.9, 103.1),
            ]
        )

        self.assertEqual(crossing["decision"], "inconclusive")
        self.assertEqual(unsustained["decision"], "inconclusive")

    def test_common_case_rejects_nonfinite_or_unordered_confidence_data(self) -> None:
        valid = [
            _run("B1", 100.0, 99.9, 100.1),
            _run("C1", 103.0, 102.9, 103.1),
            _run("B2", 100.2, 100.1, 100.3),
            _run("C2", 103.1, 103.0, 103.2),
        ]
        invalid = [dict(run) for run in valid]
        invalid[1]["median_ns"] = math.inf
        with self.assertRaisesRegex(ValueError, "finite"):
            benchmark.evaluate_common_case(invalid)

        invalid = [dict(run) for run in valid]
        invalid[2]["median_confidence_interval_ns"] = [101.0, 99.0]
        with self.assertRaisesRegex(ValueError, "ordered"):
            benchmark.evaluate_common_case(invalid)

    def test_pairing_rejects_unvalidated_confidence_interval_shape(self) -> None:
        baseline = _run("B1", 100.0, 99.9, 100.1)
        candidate = _run("C1", 103.0, 102.9, 103.1)
        baseline["median_confidence_interval_ns"] = (99.9, 100.1)

        with self.assertRaisesRegex(ValueError, "confidence interval"):
            benchmark._pairing(baseline, candidate)

    def test_cargo_executable_is_fixed(self) -> None:
        resolved = Path(benchmark._validated_cargo_executable("cargo"))
        self.assertEqual(resolved, Path(shutil.which("cargo") or ""))
        self.assertTrue(resolved.is_file())
        version = benchmark._run_command(
            benchmark.TrustedCommand(resolved, ("--version",)),
            cwd=benchmark.SCRIPT_ROOT.parent,
        )
        self.assertRegex(version, r"^cargo \d")
        for executable in ("/tmp/cargo", "cargo-nightly", "sh"):
            with (
                self.subTest(executable=executable),
                self.assertRaisesRegex(ValueError, "cargo executable"),
            ):
                benchmark._validated_cargo_executable(executable)


class CommonHarnessSourceTests(unittest.TestCase):
    def test_common_harness_is_frozen_public_no_checkpoint_workload(self) -> None:
        source = benchmark.COMMON_HARNESS_SOURCE.read_text(encoding="utf-8")
        manifest = benchmark.COMMON_HARNESS_MANIFEST.read_text(encoding="utf-8")

        self.assertIn('env!("CALC_FLOW_M5_SOURCE_COMMIT")', source)
        self.assertIn('env!("CALC_FLOW_M5_SOURCE_TREE")', source)
        self.assertIn('env!("CALC_FLOW_M5_HARNESS_SHA256")', source)
        self.assertIn("assert_eq!(harness_hash(), EXPECTED_HARNESS_SHA256)", source)
        self.assertIn("edge_channel", source)
        self.assertIn(benchmark.COMMON_CASE, source)
        self.assertIn("CALC_FLOW_M5_COMMON_EXECUTABLE", source)
        self.assertNotIn("current_exe", source)
        self.assertNotIn("checkpoint", source.lower())
        self.assertIn('calc-flow = { path = "../../../../crates/calc-flow" }', manifest)

    def test_materialized_harness_bytes_are_identical_for_every_run(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            hashes = []
            for label in benchmark.RUN_ORDER:
                materialized = benchmark.materialize_common_harness(root / label)
                hashes.append(benchmark.hash_harness_files(materialized))

        self.assertEqual(len(set(hashes)), 1)


class ArtifactIntegrityTests(unittest.TestCase):
    def test_hashed_json_rejects_report_tampering(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            report = Path(directory) / "report.json"
            benchmark.write_hashed_json(report, {"schema": "test", "value": 1})
            self.assertEqual(benchmark.load_hashed_json(report)["value"], 1)

            report.write_text("{}", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "hash"):
                benchmark.load_hashed_json(report)

    def test_hashed_json_rejects_symlinked_artifacts(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            report = root / "report.json"
            benchmark.write_hashed_json(report, {"schema": "test"})
            alias = root / "alias.json"
            alias.symlink_to(report)
            Path(f"{alias}.sha256").symlink_to(Path(f"{report}.sha256"))

            with self.assertRaisesRegex(ValueError, "regular file"):
                benchmark.load_hashed_json(alias)

    def test_hashed_json_output_is_canonical_and_deterministic(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            first = root / "first.json"
            second = root / "second.json"
            benchmark.write_hashed_json(first, {"z": 1, "a": {"b": 2}})
            benchmark.write_hashed_json(second, {"a": {"b": 2}, "z": 1})

            self.assertEqual(first.read_bytes(), second.read_bytes())
            self.assertTrue(first.read_bytes().endswith(b"\n"))

    def test_common_run_validation_recomputes_raw_statistics(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            executable = root / "common-benchmark"
            executable.write_bytes(b"ref-specific-executable")
            samples = [
                100.0 + float(index) for index in range(benchmark.COMMON_SAMPLE_COUNT)
            ]
            statistics = benchmark.common_statistics(samples)
            report = root / "B1.json"
            payload = {
                "schema": "calc-flow.m5-common-benchmark-run.v1",
                "label": "B1",
                "case": benchmark.COMMON_CASE,
                "source_commit": "a" * 40,
                "source_tree": "b" * 40,
                "harness_sha256": "c" * 64,
                "workload_sha256": "d" * 64,
                "workload_contract": "fixed",
                "executable": str(executable.resolve()),
                "executable_sha256": benchmark.sha256_file(executable),
                "sample_count": benchmark.COMMON_SAMPLE_COUNT,
                "confidence_level": 0.95,
                "raw_samples_ns": samples,
                **statistics,
            }
            benchmark.write_hashed_json(report, payload)
            validated = benchmark.validate_common_run_report(
                report,
                label="B1",
                commit="a" * 40,
                tree="b" * 40,
                harness_sha256="c" * 64,
            )
            self.assertEqual(validated["median_ns"], statistics["median_ns"])

            payload["median_ns"] = float(payload["median_ns"]) + 1.0
            forged = root / "forged.json"
            benchmark.write_hashed_json(forged, payload)
            with self.assertRaisesRegex(ValueError, "recomputed"):
                benchmark.validate_common_run_report(
                    forged,
                    label="B1",
                    commit="a" * 40,
                    tree="b" * 40,
                    harness_sha256="c" * 64,
                )

            with self.assertRaisesRegex(ValueError, "source identity"):
                benchmark.validate_common_run_report(
                    report,
                    label="B1",
                    commit="e" * 40,
                    tree="b" * 40,
                    harness_sha256="c" * 64,
                )

            executable.write_bytes(b"stale-ref-specific-executable")
            with self.assertRaisesRegex(ValueError, "executable hash"):
                benchmark.validate_common_run_report(
                    report,
                    label="B1",
                    commit="a" * 40,
                    tree="b" * 40,
                    harness_sha256="c" * 64,
                )


def _write_private_absolute_report(
    root: Path,
    candidate: benchmark.RefSnapshot,
) -> tuple[Path, Path]:
    executable = root / "debug" / "calc-flow-test"
    executable.parent.mkdir(parents=True)
    executable.write_bytes(b"candidate-test-binary")
    measurements = []
    for index, case in enumerate(benchmark.M5_ABSOLUTE_CASES):
        case_root = root / "criterion" / str(index)
        case_root.mkdir(parents=True)
        sample = case_root / "sample.json"
        estimates = case_root / "estimates.json"
        sample.write_bytes(f"sample-{index}".encode())
        estimates.write_bytes(f"estimates-{index}".encode())
        measurements.append(
            {
                "case": case,
                "comparison": "none",
                "median_ns": 100.0 + index,
                "median_confidence_interval_ns": [99.0 + index, 101.0 + index],
                "confidence_level": 0.95,
                "sample_count": benchmark.PRIVATE_SAMPLE_COUNT,
                "decision": "absolute_only",
                "artifacts": {
                    "sample": {
                        "path": str(sample.relative_to(root)),
                        "sha256": benchmark.sha256_file(sample),
                    },
                    "estimates": {
                        "path": str(estimates.relative_to(root)),
                        "sha256": benchmark.sha256_file(estimates),
                    },
                },
            }
        )
    provenance: dict[str, object] = {
        "commit": candidate.commit,
        "tree": candidate.tree,
        "clean": True,
        "harness_hash": "1" * 64,
        "config_hash": "2" * 64,
        "executable": str(executable.resolve()),
        "executable_sha256": benchmark.sha256_file(executable),
        "toolchain_hash": "3" * 64,
        "environment_hash": "4" * 64,
    }
    provenance["build_identity_hash"] = benchmark.private_build_identity_hash(
        provenance
    )
    report = {
        "schema": "calc-flow.m5-checkpoint-absolute-benchmark.v1",
        "commit": candidate.commit,
        "comparison": "none",
        "absolute_cases": list(benchmark.M5_ABSOLUTE_CASES),
        "provenance": provenance,
        "measurements": measurements,
        "overall_result": "absolute_only",
    }
    report_path = root / "m5-checkpoint-benchmark" / "candidate.json"
    report_path.parent.mkdir()
    benchmark.write_hashed_json(report_path, report)
    return report_path, executable


class PrivateAbsoluteReportTests(unittest.TestCase):
    def test_private_report_rejects_replaced_binary_and_forged_build_identity(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            candidate = benchmark.RefSnapshot(
                role="candidate",
                commit="a" * 40,
                tree="b" * 40,
                worktree=root,
            )
            report_path, executable = _write_private_absolute_report(root, candidate)
            validated = benchmark.validate_private_absolute_report(
                report_path,
                target_root=root,
                candidate=candidate,
            )
            self.assertEqual(validated["commit"], candidate.commit)

            executable.write_bytes(b"stale-candidate-test-binary")
            with self.assertRaisesRegex(ValueError, "executable hash"):
                benchmark.validate_private_absolute_report(
                    report_path,
                    target_root=root,
                    candidate=candidate,
                )

            report, executable = _write_private_absolute_report(
                root / "forged",
                candidate,
            )
            payload = benchmark.load_hashed_json(report)
            provenance = payload["provenance"]
            self.assertIsInstance(provenance, dict)
            provenance["build_identity_hash"] = "f" * 64
            forged = root / "forged-identity.json"
            benchmark.write_hashed_json(forged, payload)
            with self.assertRaisesRegex(ValueError, "build identity"):
                benchmark.validate_private_absolute_report(
                    forged,
                    target_root=root / "forged",
                    candidate=candidate,
                )


def _provenance_run(label: str) -> dict[str, object]:
    baseline = label.startswith("B")
    return {
        "label": label,
        "source_commit": ("a" if baseline else "c") * 40,
        "source_tree": ("b" if baseline else "d") * 40,
        "harness_sha256": "1" * 64,
        "workload_sha256": "2" * 64,
        "executable": f"/targets/{label}/common-benchmark",
        "executable_sha256": ("3" if baseline else "4") * 64,
        "target_dir": f"/targets/{label}",
        "evidence_root": f"/evidence/{label}",
        "source_cargo_lock_sha256": "5" * 64,
        "harness_cargo_lock_sha256": "6" * 64,
        "dependency_graph_sha256": "7" * 64,
        "toolchain_sha256": "8" * 64,
        "machine_sha256": "9" * 64,
        "environment_sha256": "e" * 64,
        "git_status_short": "",
    }


class ProvenanceContractTests(unittest.TestCase):
    def test_reference_pair_requires_distinct_ancestor_and_exact_merge_base(
        self,
    ) -> None:
        benchmark.validate_reference_contract("a" * 40, "c" * 40, "a" * 40)

        with self.assertRaisesRegex(ValueError, "distinct"):
            benchmark.validate_reference_contract("a" * 40, "a" * 40, "a" * 40)
        with self.assertRaisesRegex(ValueError, "ancestor"):
            benchmark.validate_reference_contract("a" * 40, "c" * 40, "f" * 40)

    def test_matrix_requires_fresh_roots_and_ref_specific_executables(self) -> None:
        runs = [_provenance_run(label) for label in benchmark.RUN_ORDER]
        benchmark.validate_matrix_provenance(
            runs,
            baseline_commit="a" * 40,
            baseline_tree="b" * 40,
            candidate_commit="c" * 40,
            candidate_tree="d" * 40,
        )

        shared_target = [dict(run) for run in runs]
        shared_target[1]["target_dir"] = shared_target[0]["target_dir"]
        with self.assertRaisesRegex(ValueError, "fresh target"):
            benchmark.validate_matrix_provenance(
                shared_target,
                baseline_commit="a" * 40,
                baseline_tree="b" * 40,
                candidate_commit="c" * 40,
                candidate_tree="d" * 40,
            )

        same_executable = [dict(run) for run in runs]
        same_executable[1]["executable_sha256"] = "3" * 64
        same_executable[3]["executable_sha256"] = "3" * 64
        with self.assertRaisesRegex(ValueError, "distinct executable"):
            benchmark.validate_matrix_provenance(
                same_executable,
                baseline_commit="a" * 40,
                baseline_tree="b" * 40,
                candidate_commit="c" * 40,
                candidate_tree="d" * 40,
            )


class OrchestrationPlanTests(unittest.TestCase):
    def test_plan_is_interleaved_and_keeps_every_binary_in_its_ref_worktree(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "evidence"
            baseline = benchmark.RefSnapshot(
                role="baseline",
                commit="a" * 40,
                tree="b" * 40,
                worktree=root / "worktrees" / "baseline",
            )
            candidate = benchmark.RefSnapshot(
                role="candidate",
                commit="c" * 40,
                tree="d" * 40,
                worktree=root / "worktrees" / "candidate",
            )

            plan = benchmark.build_run_plan(root, baseline, candidate)

        self.assertEqual([run.label for run in plan], list(benchmark.RUN_ORDER))
        self.assertEqual(
            [run.snapshot.role for run in plan],
            ["baseline", "candidate", "baseline", "candidate"],
        )
        self.assertEqual(len({run.target_dir for run in plan}), 4)
        self.assertEqual(len({run.evidence_root for run in plan}), 4)
        for run in plan:
            self.assertTrue(run.harness_root.is_relative_to(run.snapshot.worktree))
            self.assertTrue(run.target_dir.is_relative_to(run.snapshot.worktree))
            self.assertEqual(run.cwd, run.snapshot.worktree)

    def test_private_run_embeds_candidate_identity_and_uses_fresh_target(self) -> None:
        candidate = benchmark.RefSnapshot(
            role="candidate",
            commit="a" * 40,
            tree="b" * 40,
            worktree=Path("/evidence/worktrees/candidate"),
        )
        target = candidate.worktree / "target" / "private-run"

        environment = benchmark.private_run_environment(
            candidate,
            run_id="run-1",
            target_root=target,
        )

        self.assertEqual(
            environment["CALC_FLOW_M5_PRIVATE_SOURCE_COMMIT"], candidate.commit
        )
        self.assertEqual(
            environment["CALC_FLOW_M5_PRIVATE_SOURCE_TREE"], candidate.tree
        )
        self.assertEqual(environment["CARGO_TARGET_DIR"], str(target))


if __name__ == "__main__":
    unittest.main()
