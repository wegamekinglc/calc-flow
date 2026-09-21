"""Keep recovery measurements tied to an unchanged source and one original run."""

from __future__ import annotations

import copy
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from scripts import coverage_recovery as recovery


def _environment() -> dict[str, str]:
    return {
        "GITHUB_REPOSITORY": "wegamekinglc/calc-flow",
        "GITHUB_SHA": "a" * 40,
        "GITHUB_RUN_ID": "123",
        "GITHUB_RUN_ATTEMPT": "1",
        "GITHUB_EVENT_NAME": "push",
        "GITHUB_REF": "refs/heads/fix/dal-296-coverage-recovery-execute",
    }


class CoverageRecoveryTests(unittest.TestCase):
    def test_workflow_requires_reviewed_push_and_validated_reports_before_upload(self):
        root = Path(__file__).resolve().parents[1]
        workflow = (root / ".github/workflows/coverage-recovery.yml").read_text(
            encoding="utf-8"
        )
        self.assertIn("branches: [fix/dal-296-coverage-recovery-execute]", workflow)
        self.assertNotIn("pull_request:", workflow)
        self.assertNotIn("workflow_dispatch:", workflow)
        self.assertIn(recovery.SOURCE_SHA, workflow)
        publish = workflow.split("  publish:\n", 1)[1]
        self.assertIn("needs: [rust, python-studio]", publish)
        self.assertLess(publish.index("--manifest"), publish.index("flag-name: rust"))
        self.assertLess(
            publish.index("flag-name: studio"), publish.index("parallel-finished: true")
        )
        self.assertEqual(publish.count("git-commit: ${{ env.SOURCE_SHA }}"), 4)
        self.assertEqual(publish.count("build-number: ${{ github.run_id }}"), 4)
        self.assertNotIn("carryforward:", workflow)
        self.assertNotIn("compare-sha:", workflow)
        self.assertNotIn("continue-on-error:", workflow)
        self.assertIn("github.run_attempt == 1", workflow)

    def test_workflow_retains_full_coverage_commands_and_services(self):
        root = Path(__file__).resolve().parents[1]
        workflow = (root / ".github/workflows/coverage-recovery.yml").read_text(
            encoding="utf-8"
        )
        for command in (
            "python3.13 scripts/run_rust_coverage.py",
            "python/tests benchmarks/test_warm_stream.py",
            "--cov=calc_flow --cov-report=term-missing --cov-report=xml",
            "--cov=calc_flow_studio --cov-report=term-missing --cov-report=xml",
        ):
            self.assertIn(command, workflow)
        for image in (
            "mysql:8.4",
            "apache/kafka:3.9.0",
            "postgres:16",
            "clickhouse/clickhouse-server:24.12",
        ):
            self.assertIn(image, workflow)
        self.assertIn("--no-install-workspace", workflow)
        self.assertIn("uv build --wheel", workflow)
        self.assertNotIn("coverage-baseline", workflow)

    def test_record_distinguishes_measured_source_from_workflow_revision(self):
        with (
            patch.object(
                recovery,
                "command_output",
                side_effect=[recovery.SOURCE_SHA, "b" * 40, "", "a" * 40, ""],
            ),
            patch.object(recovery, "require_executable", return_value="/usr/bin/git"),
        ):
            result = recovery.record(Path("source"), Path("control"), _environment())
        self.assertEqual(result["source_sha"], recovery.SOURCE_SHA)
        self.assertEqual(result["workflow_sha"], "a" * 40)
        self.assertEqual(result["source_tree"], "b" * 40)
        self.assertEqual(result["run_id"], 123)
        self.assertEqual(result["run_attempt"], 1)

    def test_record_rejects_a_changed_measured_tree(self):
        with (
            patch.object(
                recovery,
                "command_output",
                side_effect=[recovery.SOURCE_SHA, "b" * 40, " M pyproject.toml"],
            ),
            patch.object(recovery, "require_executable", return_value="/usr/bin/git"),
            self.assertRaisesRegex(ValueError, "modified"),
        ):
            recovery.record(Path("source"), Path("control"), _environment())

    def test_record_rejects_retries_and_unreviewed_entry_points(self):
        for name, value in (
            ("GITHUB_RUN_ATTEMPT", "2"),
            ("GITHUB_EVENT_NAME", "pull_request"),
            ("GITHUB_REF", "refs/heads/main"),
        ):
            with self.subTest(name=name), self.assertRaises(ValueError):
                recovery.record(
                    Path("source"), Path("control"), {**_environment(), name: value}
                )

    def test_bundle_requires_all_flags_from_one_run_and_unchanged_reports(self):
        identity = {"source_sha": recovery.SOURCE_SHA, "run_id": 123}
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory)
            for path in recovery.REPORTS.values():
                report = source / path
                report.parent.mkdir(parents=True, exist_ok=True)
                report.write_text("real report", encoding="utf-8")
            with patch.object(recovery, "record", return_value=identity):
                first = recovery.seal(
                    source, source, _environment(), ["python", "studio"]
                )
                second = recovery.seal(source, source, _environment(), ["rust"])
                records = [first, second]
                before = copy.deepcopy(records)
                result = recovery.validate(records, source, source, _environment())
                self.assertEqual(set(result["reports"]), {"rust", "python", "studio"})
                self.assertEqual(records, before)
                for invalid in (
                    [first],
                    [first, second, second],
                    [first, {**second, "run_id": 124}],
                ):
                    with self.subTest(invalid=invalid), self.assertRaises(ValueError):
                        recovery.validate(invalid, source, source, _environment())
                (source / recovery.REPORTS["python"]).write_text(
                    "changed", encoding="utf-8"
                )
                with self.assertRaisesRegex(ValueError, "digest"):
                    recovery.validate(records, source, source, _environment())

    def test_cli_preserves_evidence_and_validates_before_publication(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "identity.json"
            args = [
                "--source",
                directory,
                "--control",
                directory,
                "--output",
                str(output),
            ]
            with patch.object(recovery, "seal", return_value={"run_id": 123}) as seal:
                recovery.main([*args, "--flag", "rust"])
                self.assertEqual(seal.call_args.args[-1], ["rust"])
                self.assertEqual(
                    json.loads(output.read_text(encoding="utf-8")), {"run_id": 123}
                )
                with self.assertRaises(FileExistsError):
                    recovery.main([*args, "--flag", "rust"])
            with (
                patch.object(recovery, "validate", side_effect=ValueError("mixed run")),
                self.assertRaisesRegex(ValueError, "mixed run"),
            ):
                recovery.main([*args, "--manifest", str(output)])


if __name__ == "__main__":
    unittest.main()
