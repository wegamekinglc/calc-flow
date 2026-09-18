from __future__ import annotations

import tomllib
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


class BenchmarkWorkflowTests(unittest.TestCase):
    def test_dependency_lock_excludes_the_current_workspace_distribution(self):
        project = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))[
            "project"
        ]["name"]
        suite = (ROOT / ".github/workflows/benchmark-suite.yml").read_text(
            encoding="utf-8"
        )
        self.assertIn(f"--no-emit-package {project} \\", suite)
        lock = (ROOT / "benchmarks/requirements.lock").read_text(encoding="utf-8")
        self.assertIn(f"--no-emit-package {project} ", lock)
        self.assertFalse(any(line.startswith("-e ") for line in lock.splitlines()))

    def test_nightly_paired_outputs_anchor_at_workspace_root(self):
        workflow = (ROOT / ".github/workflows/benchmarks.yml").read_text(
            encoding="utf-8"
        )
        nightly = workflow.split("  sql-datafusion-nightly:\n", 1)[1].split(
            "  sql-datafusion-weekly-matrix:\n", 1
        )[0]
        # `cargo bench -p calc-flow` runs the harness from the package root, so
        # a relative --output escapes the workspace-root report directory.
        self.assertNotIn('--output "benchmark-results/', nightly)
        self.assertIn(
            '--output "${{ github.workspace }}/benchmark-results/sql-datafusion/',
            nightly,
        )

    def test_nightly_sql_datafusion_reports_upload_even_after_failure(self):
        workflow = (ROOT / ".github/workflows/benchmarks.yml").read_text(
            encoding="utf-8"
        )
        nightly = workflow.split("  sql-datafusion-nightly:\n", 1)[1].split(
            "  sql-datafusion-weekly-matrix:\n", 1
        )[0]
        upload = nightly.split("- name: Upload SQL/DataFusion reports\n", 1)[1]
        # P1 gate failures must still leave the measured JSON evidence behind.
        self.assertIn("if: always()", upload)

    def test_matrix_sql_datafusion_reports_upload_even_after_failure(self):
        workflow = (ROOT / ".github/workflows/benchmarks.yml").read_text(
            encoding="utf-8"
        )
        matrix = workflow.split("  sql-datafusion-weekly-matrix:\n", 1)[1].split(
            "  benchmark:\n", 1
        )[0]
        upload = matrix.split("- name: Upload SQL/DataFusion matrix reports\n", 1)[1]
        # Stability-gate failures must still leave the measured screening and
        # candidate JSON behind for tolerance recalibration.
        self.assertIn("if: always()", upload)

    def test_nightly_sql_datafusion_paired_runs_use_two_warmups(self):
        workflow = (ROOT / ".github/workflows/benchmarks.yml").read_text(
            encoding="utf-8"
        )
        nightly = workflow.split("  sql-datafusion-nightly:\n", 1)[1].split(
            "  sql-datafusion-weekly-matrix:\n", 1
        )[0]
        self.assertIn("--warmups 2", nightly)

    def test_regular_ci_and_schedule_call_the_same_complete_suite(self):
        for name in ("ci-linux.yml", "benchmarks.yml"):
            workflow = (ROOT / ".github/workflows" / name).read_text(encoding="utf-8")
            self.assertIn("uses: ./.github/workflows/benchmark-suite.yml", workflow)
        suite = (ROOT / ".github/workflows/benchmark-suite.yml").read_text(
            encoding="utf-8"
        )
        self.assertIn("workflow_call:", suite)
        self.assertIn("scripts.benchmark_suite catalog", suite)
        self.assertIn("fromJSON(needs.prepare.outputs.matrix)", suite)
        self.assertIn("fail-fast: false", suite)

    def test_final_tables_and_artifacts_are_emitted_on_failure(self):
        suite = (ROOT / ".github/workflows/benchmark-suite.yml").read_text(
            encoding="utf-8"
        )
        summary = suite.split("  summary:\n", 1)[1]
        self.assertIn("if: always()", summary)
        self.assertIn("needs: [prepare, build, suite]", summary)
        self.assertIn("--github-summary", summary)
        self.assertIn("--expected-base", summary)
        self.assertIn("--expected-head", summary)
        self.assertIn("retention-days: 30", suite)
        self.assertIn("name: Retain every measured result", suite)

    def test_measurements_use_release_wheels_and_exact_base_head(self):
        suite = (ROOT / ".github/workflows/benchmark-suite.yml").read_text(
            encoding="utf-8"
        )
        self.assertIn("side: [baseline, candidate]", suite)
        self.assertIn("scripts.benchmark_suite build", suite)
        self.assertIn("--baseline", suite)
        self.assertIn("--candidate", suite)
        self.assertIn("--require-hashes benchmarks/requirements.lock", suite)
        self.assertNotIn("--benchmark-disable", suite)


if __name__ == "__main__":
    unittest.main()
